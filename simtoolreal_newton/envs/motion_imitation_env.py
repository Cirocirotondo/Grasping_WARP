"""UR5e + DG5F motion-imitation environment on Isaac Lab (Newton / PhysX).

A port of the Isaac Gym ``MotionImitationEnv`` to Isaac Lab's direct
workflow. The task -- reference state initialisation from a retargeted
transform bank, operational-space arm control through an in-loop IK,
object-centric keypoint rewards, grace-period terminations -- is unchanged and
uses the same plain-PyTorch helper modules. Only the simulator plumbing moved:

* the robot is an :class:`~isaaclab.assets.Articulation` imported from the
  URDF (fixed joints merged, like Isaac Gym's ``collapse_fixed_joints``);
* the cuboid is a :class:`~isaaclab.assets.RigidObject`, the table a static
  collider, both cloned per environment;
* joint targets, joint-state resets, root-state resets and external wrenches
  go through the articulation / rigid-object APIs;
* the palm Jacobian comes from :mod:`simtoolreal_newton.envs.kinematics`
  instead of a simulator tensor, so it is identical on every backend.

The class keeps the AnimRL attribute surface (``num_obs``, ``num_actions``,
``reference``, ``transform_bank``, ``rsi_*``, ``reset_idx`` ...) so the PPO
runner and the evaluators work through :class:`AnimRLVecEnv`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

from simtoolreal_newton import ROOT_DIR
from simtoolreal_newton.cfg import SimToolRealCfg
from simtoolreal_newton.envs.adaptive_sigma import AdaptiveSigma
from simtoolreal_newton.envs.contact import (
    fingertip_contact_diagnostics,
    fingertip_force_norms,
    fingertip_force_observation,
    fingertip_force_observation_dim,
    select_fingertip_forces,
)
from simtoolreal_newton.envs.controller import (
    ARM_BODY_NAMES,
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    JOINT_NAMES,
    PALM_LINK_NAME,
    WRIST_BODY_NAME,
    pd_gain_arrays,
)
from simtoolreal_newton.envs.self_collision import filtered_body_pairs
from simtoolreal_newton.envs.object_scale import (
    inertia_factor,
    mass_factor,
    object_scale_observation_dim,
    observe_scale,
    observed_scale_override,
    reference_height_shift,
    sample_scales,
    scale_observation,
    scale_randomization_enabled,
    scale_range,
)
from simtoolreal_newton.envs.cuboid_symmetry import (
    apply_cuboid_symmetry,
    canonicalize_cuboid_orientation,
    cuboid_rotation_symmetries,
)
from simtoolreal_newton.envs.demonstration import JointDemonstration60Hz
from simtoolreal_newton.envs.disturbance import sample_impulses
from simtoolreal_newton.envs.domain_randomization import DomainRandomization, UNAPPLIED_PARAMETERS
from simtoolreal_newton.envs.keypoints import (
    hand_keypoints,
    keypoint_gaussian,
    keypoint_tracking_error,
    keypoints_in_object_frame,
    split_palm_and_fingertips,
)
from simtoolreal_newton.envs.kinematics import PalmKinematics
from simtoolreal_newton.envs.motion_imitation_env_cfg import (
    ROBOT_URDF,
    MotionImitationEnvCfg,
    build_env_cfg,
)
from simtoolreal_newton.envs.object_assist import (
    assist_scale_at,
    object_assist_wrench,
    object_reward_gate,
    resolve_object_assist_settings,
)
from simtoolreal_newton.envs.operational_space import (
    damped_least_squares_step,
    saturate_direction_preserving,
)
from simtoolreal_newton.envs.proximity import fingertip_cuboid_proximity
from simtoolreal_newton.envs.rotations import (
    normalize_canonical_quaternion as _normalize_canonical_quaternion,
    quat_conjugate as _quat_conjugate,
    quat_multiply as _quat_multiply,
    quat_rotate as _quat_rotate,
    quat_rotate_inverse as _quat_rotate_inverse,
    quat_to_matrix as _quat_to_matrix,
    quat_to_rotation_6d,
)
from simtoolreal_newton.envs.rsi import resolve_rsi_settings, sample_rsi_indices
from simtoolreal_newton.envs.rsi_noise import perturb_reference_pose
from simtoolreal_newton.envs.sensing import (
    ActionDelay,
    add_observation_noise,
    perturb_cube_pose_observation,
    sample_cube_pose_bias,
    sample_position_bias,
)
from simtoolreal_newton.envs.transform_bank import TransformBank, nearest_transform_indices

logger = logging.getLogger(__name__)

# The fixed wrist -> mount -> base -> palm chain is merged while importing the
# robot asset. These constants reconstruct the exact rl_dg_palm URDF frame
# from the surviving wrist_3_link rigid body (xyzw quaternion convention); the
# kinematics module re-derives them from the URDF and the environment checks
# that the two agree.
PALM_PARENT_BODY_NAME = WRIST_BODY_NAME
PALM_POSITION_IN_WRIST = (0.0, 0.0, 0.0738)
PALM_ORIENTATION_IN_WRIST = (0.0, 0.0, 0.5, 0.8660254037844386)
FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME = {
    "thumb": "rl_dg_1_4",
    "index": "rl_dg_2_4",
    "middle": "rl_dg_3_4",
    "ring": "rl_dg_4_4",
    "pinky": "rl_dg_5_4",
}
FINGERTIP_BODY_NAMES = tuple(FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME.values())
FINGERTIP_OFFSETS = (
    (0.0, 0.0363, 0.0),
    (0.0, 0.0, 0.0255),
    (0.0, 0.0, 0.0255),
    (0.0, 0.0, 0.0255),
    (0.0, 0.0, 0.0363),
)


def _apply_overrides(cfg, overrides: Dict[str, object]):
    for path, value in (overrides or {}).items():
        node = cfg
        parts = [part for part in str(path).split(".") if part]
        for part in parts[:-1]:
            node = getattr(node, part)
        if not hasattr(node, parts[-1]):
            raise KeyError("Unknown configuration field {!r}".format(path))
        setattr(node, parts[-1], value)
    return cfg


class MotionImitationEnv(DirectRLEnv):
    """AnimRL-compatible motion-imitation task as an Isaac Lab direct environment."""

    cfg: MotionImitationEnvCfg
    JOINT_NAMES = JOINT_NAMES
    FINGERTIP_NAMES = tuple(FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME)

    def __init__(self, cfg: MotionImitationEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.animrl_cfg is None or cfg.robot is None:
            animrl = cfg.animrl_cfg or _apply_overrides(SimToolRealCfg(), cfg.animrl_overrides)
            animrl.env.num_envs = int(cfg.scene.num_envs)
            cfg = build_env_cfg(animrl, num_envs=int(cfg.scene.num_envs), device=str(cfg.sim.device))
        self.animrl_cfg = cfg.animrl_cfg
        acfg = self.animrl_cfg
        self._validate_task_config(acfg)
        self.num_envs_requested = int(acfg.env.num_envs)
        self.headless = True
        self.sim_device = str(cfg.sim.device)
        self._max_episode_length = int(acfg.env.episode_length)
        self.dt = float(acfg.sim.dt) * int(acfg.control.decimation)
        self._torch_device = torch.device(self.sim_device)

        torch.manual_seed(int(acfg.seed))
        np.random.seed(int(acfg.seed))

        demo_path = ROOT_DIR / acfg.motion.file
        self.reference = JointDemonstration60Hz.load(
            demo_path, device=self._torch_device, expected_hz=float(acfg.motion.frequency_hz)
        )
        if self.reference.sample_count < 2:
            raise ValueError("A demonstration needs at least two samples")
        self._load_transform_bank()
        (
            self.rsi_distribution,
            self.rsi_max_start_index,
            self.rsi_pregrasp_start_index,
            self.rsi_early_probability,
        ) = resolve_rsi_settings(acfg.env, self.reference.last_index)
        snap_from = getattr(acfg.env, "rsi_snap_placement_from_index", None)
        self.rsi_snap_placement_from_index = None if snap_from is None else int(snap_from)
        self.object_assist_settings = resolve_object_assist_settings(acfg.object_assist, self.reference.last_index)
        self.object_assist_enabled = self.object_assist_settings.enabled
        self.object_assist_gates_object_reward = bool(getattr(acfg.object_assist, "gate_object_reward", True))
        self.object_assist_scale = assist_scale_at(self.object_assist_settings, 0)
        self.reference_ghost_enabled = bool(getattr(acfg.viewer, "reference_ghost", False))
        if self.reference_ghost_enabled:
            logger.warning("The reference ghost robot is not available on the Isaac Lab port yet; ignoring it.")
            self.reference_ghost_enabled = False
        # A Warp-rendered camera (kit-less) backs the training/evaluation video.
        self.training_camera_enabled = cfg.camera is not None
        self.training_camera_env_index = int(getattr(acfg.viewer, "training_camera_env_index", 0))
        self.self_collision_enabled = bool(getattr(acfg.asset, "self_collision", False))

        # Kinematic model of the arm-to-palm chain, on the simulation device.
        self.kinematics = PalmKinematics(ROBOT_URDF, device=self._torch_device, dtype=torch.float32)

        super().__init__(cfg, render_mode, **kwargs)

        self._resolve_articulation_layout()
        self._read_joint_limits()
        self._allocate_buffers()
        self._apply_domain_randomization()
        self._setup_object_scale()
        self._check_kinematic_conventions()

    # ------------------------------------------------------------------
    # Configuration validation (mirrors the Isaac Gym constructor)
    # ------------------------------------------------------------------

    def _validate_task_config(self, acfg) -> None:
        self.num_obs = int(acfg.env.num_observations)
        self.num_privileged_obs = acfg.env.num_privileged_obs
        self.num_actions = int(acfg.env.num_actions)
        if self.num_actions != len(JOINT_NAMES):
            raise ValueError(
                "The policy drives every joint and requires exactly {} actions".format(len(JOINT_NAMES))
            )
        if acfg.control.action_parameterization != "operational_space_arm":
            raise ValueError(
                "Only the operational-space arm contract is supported. The arm's six actions are an "
                "end-effector twist [dx, dy, dz, wx, wy, wz]"
            )
        self.hand_action_scale = float(acfg.control.scale_hand_joint_target)
        self.action_target_clip = float(acfg.control.clip_joint_target)
        self.arm_translation_speed = float(acfg.control.arm_translation_speed_m_per_s)
        self.arm_rotation_speed = float(acfg.control.arm_rotation_speed_rad_per_s)
        self.ik_damping = float(acfg.control.ik_damping)
        self.ik_max_joint_delta = float(acfg.control.ik_max_joint_delta_rad)
        if (
            self.hand_action_scale <= 0.0
            or self.action_target_clip <= 0.0
            or self.arm_translation_speed <= 0.0
            or self.arm_rotation_speed <= 0.0
            or self.ik_damping <= 0.0
            or self.ik_max_joint_delta <= 0.0
        ):
            raise ValueError("Action scales, clips and IK parameters must be positive")
        self.domain_randomization = DomainRandomization(
            getattr(acfg, "domain_randomization", object()),
            int(acfg.env.num_envs),
            seed=int(getattr(acfg, "seed", 0) or 0),
            unapplied=UNAPPLIED_PARAMETERS,
        )
        contact = acfg.contact
        self.contact_enabled = bool(contact.enabled)
        self.contact_force_threshold_n = float(contact.force_threshold_n)
        self.contact_reward_per_finger = float(contact.reward_per_finger)
        self.contact_fingertip_names = tuple(str(n).strip().lower() for n in contact.fingertip_names)
        if not math.isfinite(self.contact_force_threshold_n) or self.contact_force_threshold_n <= 0.0:
            raise ValueError("contact.force_threshold_n must be finite and positive")
        if not math.isfinite(self.contact_reward_per_finger) or self.contact_reward_per_finger < 0.0:
            raise ValueError("contact.reward_per_finger must be finite and non-negative")
        if not self.contact_fingertip_names or len(set(self.contact_fingertip_names)) != len(
            self.contact_fingertip_names
        ):
            raise ValueError("contact.fingertip_names must be a non-empty list without duplicates")
        unknown = set(self.contact_fingertip_names).difference(FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME)
        if unknown:
            raise ValueError("Unknown contact fingertip names: {}".format(sorted(unknown)))
        self.contact_reward_enabled = bool(getattr(contact, "reward_enabled", False))
        if self.contact_reward_enabled and not self.contact_enabled:
            raise ValueError("contact.reward_enabled needs contact.enabled")
        self.contact_observation_enabled = bool(getattr(contact, "observe_fingertip_forces", False))
        self.contact_shaping_weight = self.contact_reward_per_finger if self.contact_reward_enabled else 0.0
        self.contact_observation_force_scale_n = float(getattr(contact, "observation_force_scale_n", 10.0))
        self.contact_observation_clip = float(getattr(contact, "observation_clip", 5.0))
        if self.contact_observation_enabled:
            if not self.contact_enabled:
                raise ValueError("contact.observe_fingertip_forces needs contact.enabled")
            if not math.isfinite(self.contact_observation_force_scale_n) or self.contact_observation_force_scale_n <= 0:
                raise ValueError("contact.observation_force_scale_n must be finite and positive")
            if not math.isfinite(self.contact_observation_clip) or self.contact_observation_clip <= 0:
                raise ValueError("contact.observation_clip must be finite and positive")
        rewards_cfg = acfg.rewards
        self.adaptive_sigmas = {}
        if bool(getattr(rewards_cfg, "adaptive_sigma_enabled", False)):
            target = float(rewards_cfg.adaptive_sigma_target_reward)
            decay = float(rewards_cfg.adaptive_sigma_decay)
            for name, initial, floor in (
                ("position_hand", rewards_cfg.position_hand_std_rad, rewards_cfg.adaptive_sigma_position_hand_floor),
                ("ee_action_rate", rewards_cfg.ee_action_rate_std, rewards_cfg.adaptive_sigma_ee_action_rate_floor),
                (
                    "hand_action_rate",
                    rewards_cfg.hand_action_rate_std,
                    rewards_cfg.adaptive_sigma_hand_action_rate_floor,
                ),
            ):
                self.adaptive_sigmas[name] = AdaptiveSigma(
                    initial=initial,
                    floor=floor,
                    target_reward=target,
                    decay=decay,
                    slack=float(getattr(rewards_cfg, "adaptive_sigma_slack", 1.5)),
                )
        self.contact_observation_dim = fingertip_force_observation_dim(contact)
        self.num_obs += self.contact_observation_dim
        # Cuboid scale (generalize_size): validated here so a bad range fails
        # at construction, applied per episode in reset_idx.
        randomization_cfg = acfg.object_randomization
        self.object_scale_range = scale_range(randomization_cfg)
        self.object_scale_enabled = scale_randomization_enabled(randomization_cfg)
        self.object_scale_observed = observe_scale(randomization_cfg)
        # Ablation: what the policy is *told* the scale is. None = the truth.
        self.object_scale_observation_override = observed_scale_override(randomization_cfg)
        self.object_scale_mass_with_volume = bool(getattr(randomization_cfg, "scale_mass_with_volume", True))
        self.num_obs += object_scale_observation_dim(randomization_cfg)
        self.critic_force_observation_dim = (
            3 * len(contact.fingertip_names)
            if bool(getattr(contact, "critic_observes_fingertip_forces", False))
            else 0
        )
        self.critic_parameter_dim = self.domain_randomization.privileged_dim
        self.critic_parameter_table = None
        if self.critic_force_observation_dim or self.critic_parameter_dim:
            if self.critic_force_observation_dim and not self.contact_enabled:
                raise ValueError("critic_observes_fingertip_forces requires contact.enabled")
            self.num_privileged_obs = self.num_obs + self.critic_force_observation_dim + self.critic_parameter_dim
        self.palm_keypoint_anchor = str(getattr(rewards_cfg, "palm_keypoint_anchor", "measured")).strip().lower()
        if self.palm_keypoint_anchor not in ("measured", "reference"):
            raise ValueError("rewards.palm_keypoint_anchor must be 'measured' or 'reference'")
        self.proximity_fingertip_names = tuple(
            str(n).strip().lower() for n in rewards_cfg.fingertip_object_distance_names
        )
        self.proximity_std_m = float(rewards_cfg.fingertip_object_distance_std_m)
        self.proximity_weight = float(rewards_cfg.fingertip_object_distance_weight)
        if not self.proximity_fingertip_names or len(set(self.proximity_fingertip_names)) != len(
            self.proximity_fingertip_names
        ):
            raise ValueError("rewards.fingertip_object_distance_names must be a non-empty list without duplicates")
        unknown = set(self.proximity_fingertip_names).difference(FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME)
        if unknown:
            raise ValueError("Unknown proximity fingertip names: {}".format(sorted(unknown)))
        if not math.isfinite(self.proximity_std_m) or self.proximity_std_m <= 0.0:
            raise ValueError("rewards.fingertip_object_distance_std_m must be finite and positive")
        if not math.isfinite(self.proximity_weight) or self.proximity_weight < 0.0:
            raise ValueError("rewards.fingertip_object_distance_weight must be finite and non-negative")
        control_dt = float(acfg.sim.dt) * int(acfg.control.decimation)
        if not math.isclose(control_dt, 1.0 / float(acfg.motion.frequency_hz), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Control dt {:.9g} does not match the {} Hz demonstration".format(control_dt, acfg.motion.frequency_hz)
            )

    def _load_transform_bank(self) -> None:
        randomization = self.animrl_cfg.object_randomization
        bank_path = ROOT_DIR / str(randomization.bank_path)
        if not bank_path.is_file():
            raise FileNotFoundError(
                "No transform bank at {}. Build one first with scripts/build_transform_bank.py".format(bank_path)
            )
        bank = TransformBank.load(bank_path)
        if bank.sample_count != self.reference.sample_count:
            raise ValueError(
                "The transform bank has {} frames but the demonstration has {}".format(
                    bank.sample_count, self.reference.sample_count
                )
            )
        if bank.reference_keypoints.shape[-2:] != (9, 3):
            raise ValueError("The transform bank carries malformed keypoints")
        self.transform_bank = bank.to(device=self._torch_device, dtype=torch.float32)
        self.palm_lever_arm_m = float(self.animrl_cfg.rewards.palm_lever_arm_m)
        print(
            "Transform bank: {} transforms, {:.1f}% of sampled transforms were feasible".format(
                self.transform_bank.transform_count, 100.0 * self.transform_bank.acceptance
            )
        )
        self.cuboid_symmetries = cuboid_rotation_symmetries(
            [0.5 * float(v) for v in self.animrl_cfg.object.size_m]
        ).to(device=self._torch_device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        acfg = self.animrl_cfg
        self.robot = Articulation(self.cfg.robot)
        self.cube = RigidObject(self.cfg.cube)
        table_prim = "/World/envs/env_0/Table"
        self.cfg.table.func(table_prim, self.cfg.table, translation=tuple(self.cfg.table_pos))
        self._author_collision_filters(table_prim)
        self._bind_fingertip_material(
            float(acfg.asset.fingertip_friction),
            float(acfg.asset.restitution),
            getattr(acfg.asset, "fingertip_torsional_friction", None),
            getattr(acfg.asset, "fingertip_rolling_friction", None),
        )
        self.contact_sensor = None
        if self.cfg.contact_sensor is not None:
            self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.cube_contact_sensor = None
        if self.cfg.debug_cube_contacts:
            from isaaclab.sensors import ContactSensorCfg

            self.cube_contact_sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Cube",
                    filter_prim_paths_expr=["/World/envs/env_.*/Robot/.*"],
                    update_period=0.0,
                )
            )
        self.camera = None
        if self.cfg.camera is not None:
            from isaaclab.sensors import Camera

            self.camera = Camera(self.cfg.camera)
        global_paths = ()
        if self.cfg.spawn_ground_plane:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
            global_paths = ("/World/ground",)
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        positions = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing)[0]
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, positions, global_paths=global_paths)
        cloner.replicate(plan)
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=list(global_paths))
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cube"] = self.cube
        if self.contact_sensor is not None:
            self.scene.sensors["contact"] = self.contact_sensor
        if self.cube_contact_sensor is not None:
            self.scene.sensors["cube_contact"] = self.cube_contact_sensor
        if self.camera is not None:
            self.scene.sensors["camera"] = self.camera
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _robot_body_prims(self):
        from pxr import UsdPhysics

        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        bodies = {}
        for prim in root.GetAllChildren() if False else [p for p in stage.Traverse() if str(p.GetPath()).startswith("/World/envs/env_0/Robot")]:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies[prim.GetName()] = prim
        return stage, bodies

    def _author_collision_filters(self, table_prim: str) -> None:
        """Reproduce the Isaac Gym collision graph with ``physics:filteredPairs``.

        Robot-table contacts are disabled everywhere, and arm-cube contacts up
        to ``wrist_2_link``. The hand assembly (``wrist_3_link`` carries the
        merged palm shapes) and the table keep colliding with the cuboid.
        Newton's USD importer honours the pairs on body and collider prims and
        replicates them with the prototype environment.

        With ``asset.self_collision`` on, the articulation no longer filters
        its own bodies wholesale (see ``make_robot_cfg``), so every robot body
        pair outside :func:`~simtoolreal_newton.envs.self_collision.allowed_body_pairs`
        is filtered here instead, once, on the lexicographically smaller body.
        """
        from pxr import UsdPhysics

        stage, bodies = self._robot_body_prims()
        if not bodies:
            logger.warning("No robot rigid bodies found under env_0; collision filters were not authored")
            return
        table_colliders = [
            p for p in stage.Traverse() if str(p.GetPath()).startswith(table_prim) and p.HasAPI(UsdPhysics.CollisionAPI)
        ]
        cube_prim = stage.GetPrimAtPath("/World/envs/env_0/Cube")
        self_pairs = {}
        if self.self_collision_enabled:
            for pair in filtered_body_pairs(self.animrl_cfg.asset, bodies):
                first, second = sorted(pair)
                self_pairs.setdefault(first, []).append(second)
        for name, prim in bodies.items():
            api = UsdPhysics.FilteredPairsAPI.Apply(prim)
            rel = api.CreateFilteredPairsRel()
            for collider in table_colliders:
                rel.AddTarget(collider.GetPath())
            if name in ARM_BODY_NAMES and cube_prim.IsValid():
                rel.AddTarget(cube_prim.GetPath())
            for other in sorted(self_pairs.get(name, ())):
                rel.AddTarget(bodies[other].GetPath())
        self.arm_collision_body_names = tuple(n for n in bodies if n in ARM_BODY_NAMES)
        self.hand_collision_body_names = tuple(n for n in bodies if n not in ARM_BODY_NAMES)

    def _bind_fingertip_material(
        self, friction: float, restitution: float, torsional_friction=None, rolling_friction=None
    ) -> None:
        """Give the five distal phalanges their own, grippier, physics material."""
        stage, bodies = self._robot_body_prims()
        material_path = "/World/envs/env_0/Robot/fingertip_material"
        material_cfg = RigidBodyMaterialBaseCfg(
            static_friction=friction, dynamic_friction=friction, restitution=restitution
        )
        if torsional_friction is None and rolling_friction is None:
            material_cfg.func(material_path, material_cfg)
        else:
            # Newton-only fragment (newton:torsionalFriction) composed with the
            # solver-common friction/restitution on the same material prim.
            from isaaclab.sim.spawners.materials import (
                UsdPhysicsRigidBodyMaterialCfg,
                spawn_rigid_body_material_from_fragments,
            )
            from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

            spawn_rigid_body_material_from_fragments(
                material_path,
                [
                    UsdPhysicsRigidBodyMaterialCfg(
                        static_friction=friction, dynamic_friction=friction, restitution=restitution
                    ),
                    NewtonMaterialCfg(
                        torsional_friction=None if torsional_friction is None else float(torsional_friction),
                        rolling_friction=None if rolling_friction is None else float(rolling_friction),
                    ),
                ],
                stage=stage,
            )
        for name in FINGERTIP_BODY_NAMES:
            prim = bodies.get(name)
            if prim is None:
                raise ValueError("Fingertip body {!r} was not found in the imported robot".format(name))
            sim_utils.bind_physics_material(str(prim.GetPath()), material_path, stronger_than_descendants=True)

    # ------------------------------------------------------------------
    # Post-construction resolution
    # ------------------------------------------------------------------

    def _resolve_articulation_layout(self) -> None:
        names = list(self.robot.joint_names)
        if names != list(JOINT_NAMES):
            raise ValueError(
                "Robot DOFs are not in demonstration order: {} vs {}".format(names, list(JOINT_NAMES))
            )
        wrist_ids, _ = self.robot.find_bodies(PALM_PARENT_BODY_NAME)
        if len(wrist_ids) != 1:
            raise ValueError("Robot rigid body {!r} was not found".format(PALM_PARENT_BODY_NAME))
        self.wrist_body_index = int(wrist_ids[0])
        tip_ids, tip_names = self.robot.find_bodies(list(FINGERTIP_BODY_NAMES), preserve_order=True)
        if list(tip_names) != list(FINGERTIP_BODY_NAMES):
            raise ValueError("Fingertip bodies resolved to {}".format(tip_names))
        self.fingertip_body_indices = torch.as_tensor(tip_ids, dtype=torch.long, device=self.device)
        # Contact forces are gathered into a (num_envs, 5, 3) buffer in
        # FINGERTIP_BODY_NAMES order, so the helper modules index into that.
        self.contact_fingertip_indices = torch.as_tensor(
            [FINGERTIP_BODY_NAMES.index(FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME[n]) for n in self.contact_fingertip_names],
            dtype=torch.long,
            device=self.device,
        )
        self.proximity_fingertip_indices = torch.as_tensor(
            [FINGERTIP_BODY_NAMES.index(FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME[n]) for n in self.proximity_fingertip_names],
            dtype=torch.long,
            device=self.device,
        )
        self._contact_sensor_order = None
        if self.contact_sensor is not None:
            sensor_names = list(self.contact_sensor.body_names or [])
            order = []
            for name in FINGERTIP_BODY_NAMES:
                matches = [i for i, s in enumerate(sensor_names) if s.endswith(name)]
                if len(matches) != 1:
                    raise ValueError(
                        "Contact sensor bodies {} do not resolve fingertip {!r} uniquely".format(sensor_names, name)
                    )
                order.append(matches[0])
            self._contact_sensor_order = torch.as_tensor(order, dtype=torch.long, device=self.device)
        self.robot_body_indices = torch.arange(self.robot.num_bodies, dtype=torch.long, device=self.device)
        # Impulse targets: the arm proper (base to wrist_2) and the phalanges.
        arm_ids, _ = self.robot.find_bodies(list(ARM_BODY_NAMES), preserve_order=True)
        self.arm_body_indices = torch.as_tensor(arm_ids, dtype=torch.long, device=self.device)
        finger_ids, _ = self.robot.find_bodies(["rl_dg_[1-5]_[1-4]"])
        self.finger_body_indices = torch.as_tensor(finger_ids, dtype=torch.long, device=self.device)
        self.cube_body_index_tensor = torch.zeros(1, dtype=torch.long, device=self.device)

    def _read_joint_limits(self) -> None:
        acfg = self.animrl_cfg
        limits = self.robot.data.joint_pos_limits.torch[0].detach().clone()
        lower, upper = limits[:, 0], limits[:, 1]
        if torch.any(~torch.isfinite(lower)) or torch.any(~torch.isfinite(upper)):
            raise ValueError("Robot position limits must be finite")
        if torch.any(lower >= upper):
            raise ValueError("Robot contains an invalid position interval")
        self.joint_lower_limits = lower.to(torch.float32)
        self.joint_upper_limits = upper.to(torch.float32)
        self.arm_lower_limits = self.joint_lower_limits[: len(ARM_JOINT_NAMES)]
        self.arm_upper_limits = self.joint_upper_limits[: len(ARM_JOINT_NAMES)]
        default_arm = torch.as_tensor(acfg.init_state.default_arm_joint_angles, dtype=torch.float32, device=self.device)
        default_hand = torch.as_tensor(
            acfg.init_state.default_hand_joint_angles, dtype=torch.float32, device=self.device
        )
        if default_arm.shape != (len(ARM_JOINT_NAMES),) or default_hand.shape != (len(HAND_JOINT_NAMES),):
            raise ValueError("Default pose must contain 6 arm and 20 hand angles")
        self.default_positions = torch.cat((default_arm, default_hand))
        self.default_arm_positions = self.default_positions[: len(ARM_JOINT_NAMES)]
        self.default_hand_positions = self.default_positions[len(ARM_JOINT_NAMES) :]
        if torch.any(self.default_positions < self.joint_lower_limits) or torch.any(
            self.default_positions > self.joint_upper_limits
        ):
            raise ValueError("Default pose exceeds the URDF position limits")
        q = self.reference.q
        if torch.any(q < self.joint_lower_limits - 1e-6) or torch.any(q > self.joint_upper_limits + 1e-6):
            raise ValueError("The demonstration exceeds the robot position limits")
        stiffness, damping = pd_gain_arrays(
            arm_stiffness_scale=float(getattr(acfg.control, "arm_stiffness_scale", 1.0)),
            arm_damping_scale=float(getattr(acfg.control, "arm_damping_scale", 1.0)),
            hand_stiffness_scale=float(getattr(acfg.control, "hand_stiffness_scale", 1.0)),
            hand_damping_scale=float(getattr(acfg.control, "hand_damping_scale", 1.0)),
        )
        self.pd_properties = {"stiffness": stiffness, "damping": damping}

    def _allocate_buffers(self) -> None:
        n, device = self.num_envs, self.device
        self.policy_obs = torch.zeros((n, self.num_obs), dtype=torch.float32, device=device)
        self.critic_obs = (
            torch.zeros((n, self.num_privileged_obs), dtype=torch.float32, device=device)
            if (self.critic_force_observation_dim or self.critic_parameter_dim)
            else None
        )
        self._critic_force_features = None
        self.action_delay = ActionDelay(n, self.num_actions, self.domain_randomization.action_delay_steps, device)
        self.observation_position_bias = sample_position_bias(
            n,
            self.num_actions,
            self.domain_randomization.obs_q_bias_rad if self.domain_randomization.enabled else 0.0,
            device,
        )
        # Per-episode offset of the observed bar pose (position, rotation
        # vector), redrawn at every reset; zero unless the noise family is on.
        self.cube_observation_position_bias = torch.zeros((n, 3), dtype=torch.float32, device=device)
        self.cube_observation_rotation_bias = torch.zeros((n, 3), dtype=torch.float32, device=device)
        if self.critic_parameter_dim:
            self.critic_parameter_table = self.domain_randomization.privileged_table(device=device)
        self.rew_buf = torch.zeros(n, dtype=torch.float32, device=device)
        self.reference_index = torch.zeros(n, dtype=torch.long, device=device)
        self.transform_index = torch.zeros_like(self.reference_index)
        self.episode_translation = torch.zeros(n, 3, dtype=torch.float32, device=device)
        self.episode_yaw_rad = torch.zeros(n, dtype=torch.float32, device=device)
        self.symmetry_index = torch.zeros_like(self.reference_index)
        self.arm_violation_steps = torch.zeros_like(self.reference_index)
        self.hand_violation_steps = torch.zeros_like(self.reference_index)
        self.object_violation_steps = torch.zeros_like(self.reference_index)
        bools = torch.zeros(n, dtype=torch.bool, device=device)
        self.arm_violation = bools.clone()
        self.hand_violation = bools.clone()
        self.object_violation = bools.clone()
        self.episode_initial_object_com_height_m = torch.zeros(n, dtype=torch.float32, device=device)
        self.episode_peak_object_com_height_m = torch.zeros_like(self.episode_initial_object_com_height_m)
        self.object_assist_force_n = torch.zeros(n, dtype=torch.float32, device=device)
        self.object_assist_torque_nm = torch.zeros_like(self.object_assist_force_n)
        self.actions = torch.zeros((n, self.num_actions), dtype=torch.float32, device=device)
        self.previous_actions = torch.zeros_like(self.actions)
        # The low-passed action the environment executes (see
        # control.action_filter_alpha); equals the raw action at alpha 1.
        self.action_filter_alpha = float(getattr(self.animrl_cfg.control, "action_filter_alpha", 1.0))
        if not 0.0 < self.action_filter_alpha <= 1.0:
            raise ValueError("control.action_filter_alpha must lie in (0, 1]")
        self.filtered_actions = torch.zeros_like(self.actions)
        self.position_targets = torch.zeros((n, self.num_actions), dtype=torch.float32, device=device)
        # What the drives are actually fed: the commanded targets, slewed at
        # the URDF joint velocity limits (see _apply_action).
        self.applied_targets = torch.zeros_like(self.position_targets)
        velocity_limits = self.robot.data.joint_velocity_limits.torch[0].detach().clone().to(torch.float32)
        if torch.any(~torch.isfinite(velocity_limits)) or torch.any(velocity_limits <= 0.0):
            raise ValueError("Robot joint velocity limits must be finite and positive")
        self.target_slew_per_step = velocity_limits * self.dt
        if not bool(getattr(self.animrl_cfg.control, "slew_targets_at_velocity_limit", True)):
            self.target_slew_per_step = torch.full_like(self.target_slew_per_step, float("inf"))
        self.requested_twist = torch.zeros((n, 6), dtype=torch.float32, device=device)
        self.achieved_twist = torch.zeros_like(self.requested_twist)
        self.applied_arm_q_delta = torch.zeros_like(self.requested_twist)
        self.ik_residual_norm = torch.zeros(n, dtype=torch.float32, device=device)
        self.arm_joint_delta_norm = torch.zeros_like(self.ik_residual_norm)
        self.arm_joint_delta_clipped = torch.zeros_like(self.ik_residual_norm)
        self.suppress_ee_action_rate = torch.zeros(n, dtype=torch.bool, device=device)
        self._last_command_targets = torch.zeros((n, self.num_actions), dtype=torch.float32, device=device)
        self.palm_position_in_wrist = torch.tensor(PALM_POSITION_IN_WRIST, dtype=torch.float32, device=device).expand(n, -1)
        self.palm_orientation_in_wrist = torch.tensor(
            PALM_ORIENTATION_IN_WRIST, dtype=torch.float32, device=device
        ).expand(n, -1)
        self.fingertip_offsets = (
            torch.tensor(FINGERTIP_OFFSETS, dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1, -1)
        )
        self.world_axis_sign = torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float32, device=device)
        self.world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device).expand(n, -1)
        self.robot_base_position = torch.tensor(self.animrl_cfg.init_state.pos, dtype=torch.float32, device=device)
        self.object_half_extents = torch.tensor(self.animrl_cfg.object.size_m, dtype=torch.float32, device=device) / 2.0
        # Per-episode scale factor and the resulting per-environment half
        # extents (generalize_size). Both stay at the nominal values until a
        # reset draws a factor.
        self.object_scale = torch.ones(n, dtype=torch.float32, device=device)
        self.object_half_extents_per_env = self.object_half_extents.unsqueeze(0).repeat(n, 1)
        self.gravity_vector = torch.tensor(self.animrl_cfg.sim.gravity, dtype=torch.float32, device=device)
        mj = getattr(self.animrl_cfg.sim, "mjwarp", None)
        self.object_max_linear_velocity = float(getattr(mj, "object_max_linear_velocity", 0.0) or 0.0)
        self.object_max_angular_velocity = float(getattr(mj, "object_max_angular_velocity", 0.0) or 0.0)
        self.net_contact_forces = (
            torch.zeros((n, len(FINGERTIP_BODY_NAMES), 3), dtype=torch.float32, device=device)
            if self.contact_enabled
            else None
        )
        self.cube_body_forces = torch.zeros((n, 3), dtype=torch.float32, device=device)
        self.cube_body_torques = torch.zeros_like(self.cube_body_forces)
        self.robot_body_forces = torch.zeros((n, self.robot.num_bodies, 3), dtype=torch.float32, device=device)
        self._wrench_pending = False
        self.episode_sums = {
            name: torch.zeros(n, dtype=torch.float32, device=device)
            for name in (
                "reward",
                "palm_keypoint_reward",
                "fingertip_keypoint_reward",
                "palm_keypoint_error_m",
                "fingertip_keypoint_error_m",
                "palm_tilt_reward",
                "palm_tilt_error_rad",
                "ee_action_rate_reward",
                "arm_joint_rate_reward",
                "ik_residual_reward",
                "ik_residual_norm",
                "arm_joint_delta_norm",
                "arm_joint_delta_clipped",
                "hand_position_reward",
                "hand_velocity_reward",
                "hand_action_rate_reward",
                "object_position_reward",
                "object_orientation_reward",
                "fingertip_object_distance_reward",
                "fingertip_object_distance_m",
                "rms_position_error",
                "rms_velocity_error",
                "rms_ee_action_rate",
                "rms_arm_joint_rate",
                "rms_hand_position_error",
                "rms_hand_velocity_error",
                "rms_hand_action_rate",
                "object_position_error_m",
                "object_orientation_error_rad",
                "fingertip_contact_reward",
                "fingertip_contact_fraction",
                "fingertip_contact_force_n",
                "object_assist_force_n",
                "object_assist_torque_nm",
            )
        }
        self._metrics = None
        self._early = torch.zeros_like(bools)
        self._timeout = torch.zeros_like(bools)
        self.extras = {}

    def _apply_domain_randomization(self) -> None:
        randomization = self.domain_randomization
        if not randomization.enabled:
            return
        print("Domain randomisation active:")
        for name, (low, high) in randomization.summary().items():
            print("  {:<20s} x[{:.3f}, {:.3f}]".format(name, low, high))
        n = self.num_envs
        stiffness = torch.as_tensor(self.pd_properties["stiffness"], device=self.device).unsqueeze(0).repeat(n, 1)
        damping = torch.as_tensor(self.pd_properties["damping"], device=self.device).unsqueeze(0).repeat(n, 1)
        arm, hand = slice(0, len(ARM_JOINT_NAMES)), slice(len(ARM_JOINT_NAMES), None)
        for env_index in range(n):
            stiffness[env_index, arm] *= randomization.multiplier("arm_stiffness", env_index)
            damping[env_index, arm] *= randomization.multiplier("arm_damping", env_index)
            stiffness[env_index, hand] *= randomization.multiplier("hand_stiffness", env_index)
            damping[env_index, hand] *= randomization.multiplier("hand_damping", env_index)
        self.robot.write_joint_stiffness_to_sim_index(stiffness=stiffness.contiguous())
        self.robot.write_joint_damping_to_sim_index(damping=damping.contiguous())
        mass_scale = torch.tensor(
            [randomization.multiplier("object_mass", i) for i in range(n)], dtype=torch.float32, device=self.device
        )
        default_mass = self.cube.data.default_mass.torch.to(self.device)
        self.cube.set_masses_index(masses=(default_mass * mass_scale.view(n, 1)).contiguous())
        default_inertia = self.cube.data.default_inertia.torch.to(self.device)
        self.cube.set_inertias_index(inertias=(default_inertia * mass_scale.view(n, 1, 1)).contiguous())
        # Kept for the per-episode cuboid scale, which multiplies on top of
        # the domain-randomization mass multiplier (generalize_size).
        self._object_nominal_mass = (default_mass * mass_scale.view(n, 1)).clone()
        self._object_nominal_inertia = (default_inertia * mass_scale.view(n, 1, 1)).clone()
        link_scale = torch.tensor(
            [randomization.multiplier("robot_link_mass", i) for i in range(n)], dtype=torch.float32, device=self.device
        )
        if bool(torch.any(link_scale != 1.0)):
            robot_mass = self.robot.data.default_mass.torch.to(self.device)
            self.robot.set_masses_index(masses=(robot_mass * link_scale.view(n, 1)).contiguous())
        self._apply_friction_randomization()
        for key in UNAPPLIED_PARAMETERS:
            values = [randomization.multiplier(key, i) for i in range(n)]
            if any(abs(v - 1.0) > 1e-9 for v in values):
                logger.warning(
                    "domain_randomization.%s is not applied on this backend; every environment keeps the "
                    "nominal value and the critic is not told about this parameter.",
                    key,
                )

    def _apply_friction_randomization(self) -> None:
        """Per-environment contact friction, written into the solver's per-world material arrays.

        The same route the per-episode bar scale takes: bind the Newton model's
        ``shape_material_mu`` (and the torsional/rolling terms) through the
        asset views, multiply the rows of each world, and notify
        ``SHAPE_PROPERTIES`` so MuJoCo-Warp refreshes ``geom_friction`` per
        world. Contacts combine the two shapes' friction by the maximum, so
        the fingertip (1.0) decides the fingertip-bar pair and the bar and
        table (0.5 each) decide theirs: each side carries its own multiplier.
        The fingertip multiplier scales the sliding and torsional terms
        together, the torsional one being what holds the bar's gravity moment
        in a two-finger pinch.
        """
        randomization = self.domain_randomization
        keys = ("fingertip_friction", "object_friction", "table_friction")
        multipliers = {
            key: torch.tensor(
                [randomization.multiplier(key, i) for i in range(self.num_envs)],
                dtype=torch.float32,
                device=self.device,
            )
            for key in keys
        }
        if all(bool(torch.all(values == 1.0)) for values in multipliers.values()):
            return
        if "physx" in self.scene.physics_backend:
            raise ValueError("Per-environment friction randomization is only implemented on the Newton backend")
        import warp as wp  # noqa: PLC0415
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
        from newton import ModelFlags  # noqa: PLC0415

        model = NewtonManager.get_model()
        n = self.num_envs

        def bind(view, attribute):
            binding = wp.to_torch(view.get_attribute(attribute, model))
            return binding.reshape(n, -1)

        # Fingertips: every shape of the five distal phalanges (the merged tip
        # included), the ones _bind_fingertip_material gave the grippier pad.
        robot_view = self.robot._root_view
        if not getattr(robot_view, "shapes_contiguous", True):
            raise RuntimeError("The robot's shapes are not contiguous in the Newton model; cannot bind friction")
        # Every shape of the distal phalanges carries the pad material: the
        # collision hulls (``rl_dg_<f>_4_c``, ``rl_dg_<f>_tip_c``) and, on
        # imports that keep them as shapes, the visual meshes of the same
        # links -- hence a prefix match and a lower bound, not an exact count.
        shape_names = list(robot_view.shape_names)
        prefixes = tuple(
            "rl_dg_{}_{}".format(f, part) for f in range(1, 6) for part in ("4", "tip")
        )
        pad = [i for i, name in enumerate(shape_names) if name.startswith(prefixes)]
        if len(pad) < 10:
            raise RuntimeError("Expected at least the ten fingertip shapes, found {} in {}".format(len(pad), shape_names))
        pad = torch.as_tensor(pad, dtype=torch.long, device=self.device)
        for attribute in ("shape_material_mu", "shape_material_mu_torsional"):
            rows = bind(robot_view, attribute)
            rows[:, pad] = rows[:, pad] * multipliers["fingertip_friction"].unsqueeze(1)
        cube_rows = bind(self.cube._root_view, "shape_material_mu")
        cube_rows.mul_(multipliers["object_friction"].unsqueeze(1))
        # The table is a bare prim, wrapped by no view: locate its shapes by label.
        labels = list(model.shape_label)
        table = [None] * n
        for index, label in enumerate(labels):
            if "/Table/" not in label:
                continue
            env_index = int(label.split("/envs/env_")[1].split("/")[0])
            table[env_index] = index
        if any(index is None for index in table):
            raise RuntimeError("Could not locate one table shape per environment in the Newton model")
        table_mu = wp.to_torch(model.shape_material_mu)
        table_index = torch.as_tensor(table, dtype=torch.long, device=table_mu.device)
        table_mu[table_index] = table_mu[table_index] * multipliers["table_friction"].to(table_mu.device)
        NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        self._friction_multipliers = multipliers

    def _check_kinematic_conventions(self) -> None:
        """Assert what the task-space controller rests on, once, at startup."""
        offset = self.kinematics.palm_offset_in_wrist()
        expected_pos = torch.tensor(PALM_POSITION_IN_WRIST, dtype=torch.float32, device=self.device)
        if not torch.allclose(offset[:3, 3], expected_pos, atol=1e-6):
            raise ValueError("The URDF palm offset {} disagrees with PALM_POSITION_IN_WRIST".format(offset[:3, 3]))
        tip_offsets = self.kinematics.fingertip_offsets()
        if not torch.allclose(tip_offsets, self.fingertip_offsets[0], atol=1e-6):
            raise ValueError("The URDF fingertip offsets disagree with FINGERTIP_OFFSETS")
        root_orientation = self.robot_root_state[:, 3:7]
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).expand_as(root_orientation)
        if not torch.allclose(root_orientation.abs(), identity.abs(), atol=1e-4):
            raise ValueError(
                "Operational-space control assumes the robot base is unrotated in world, so that the "
                "base-frame Jacobian is also the world-frame Jacobian"
            )

    # ------------------------------------------------------------------
    # Properties: simulator state in the environment (per-env local) frame
    # ------------------------------------------------------------------

    @property
    def max_episode_length(self) -> int:
        return self._max_episode_length

    @max_episode_length.setter
    def max_episode_length(self, value: int) -> None:
        self._max_episode_length = int(value)

    @property
    def env_origins(self) -> torch.Tensor:
        return self.scene.env_origins

    @property
    def q(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch

    @property
    def dq(self) -> torch.Tensor:
        return self.robot.data.joint_vel.torch

    @property
    def previous_targets(self) -> torch.Tensor:
        """Most recently applied position targets, in demonstration order."""
        return self.position_targets

    @property
    def cube_root_state(self) -> torch.Tensor:
        pose = self.cube.data.root_link_pose_w.torch
        vel = self.cube.data.root_com_vel_w.torch
        position = pose[:, 0:3] - self.env_origins
        return torch.cat((position, pose[:, 3:7], vel), dim=1)

    @property
    def cube_position(self) -> torch.Tensor:
        return self.cube.data.root_link_pos_w.torch - self.env_origins

    @property
    def cube_orientation(self) -> torch.Tensor:
        return self.cube.data.root_link_quat_w.torch

    @property
    def cube_linear_velocity(self) -> torch.Tensor:
        return self.cube.data.root_com_lin_vel_w.torch

    @property
    def cube_angular_velocity(self) -> torch.Tensor:
        return self.cube.data.root_com_ang_vel_w.torch

    @property
    def robot_root_state(self) -> torch.Tensor:
        pose = self.robot.data.root_link_pose_w.torch
        return torch.cat((pose[:, 0:3] - self.env_origins, pose[:, 3:7]), dim=1)

    def _fingertip_positions_world(self) -> torch.Tensor:
        poses = self.robot.data.body_link_pose_w.torch[:, self.fingertip_body_indices]
        orientations = _normalize_canonical_quaternion(poses[..., 3:7])
        return poses[..., 0:3] - self.env_origins.unsqueeze(1) + _quat_rotate(orientations, self.fingertip_offsets)

    def _palm_pose_world(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The palm frame in the env frame, from the wrist plus a fixed offset."""
        wrist = self.robot.data.body_link_pose_w.torch[:, self.wrist_body_index]
        wrist_orientation = _normalize_canonical_quaternion(wrist[:, 3:7])
        palm_position = wrist[:, 0:3] - self.env_origins + _quat_rotate(wrist_orientation, self.palm_position_in_wrist)
        palm_orientation = _normalize_canonical_quaternion(
            _quat_multiply(wrist_orientation, self.palm_orientation_in_wrist)
        )
        return palm_position, palm_orientation

    def palm_pose_tracking(self) -> Dict[str, torch.Tensor]:
        """Actual and reference palm pose in the env frame, for evaluation plots.

        The reference palm is rebuilt from the transform bank's palm keypoints
        (origin plus three lever points in the reference bar frame) and placed
        in the env frame through the episode's reference bar pose -- the same
        anchoring the palm keypoint reward uses with anchor="reference", so
        the plotted target is exactly what the policy is scored against.
        Rotations are ``(N, 3, 3)`` matrices whose columns are the palm axes.
        """
        actual_position, actual_orientation = self._palm_pose_world()
        reference = self.transform_bank.sample(self.transform_index, self.reference_index)
        root = self._cube_reference_root_states(reference)
        root_orientation = _normalize_canonical_quaternion(root[:, 3:7])
        keypoints = self.transform_bank.keypoints_at(self.reference_index)
        origin = keypoints[:, 0]
        # hand_keypoints() puts the image of basis vector i at index i + 1, so
        # the differences are the columns of the palm rotation in the bar frame.
        axes_in_bar = (keypoints[:, 1:4] - origin.unsqueeze(1)) / float(self.palm_lever_arm_m)
        matrix_in_bar = axes_in_bar.transpose(1, 2)
        return {
            "actual_position": actual_position,
            "actual_matrix": _quat_to_matrix(actual_orientation),
            "reference_position": root[:, 0:3] + _quat_rotate(root_orientation, origin),
            "reference_matrix": _quat_to_matrix(root_orientation) @ matrix_in_bar,
        }

    def object_pose_tracking(self) -> Dict[str, torch.Tensor]:
        """Actual and reference object pose in the env frame, for evaluation plots.

        The reference is the episode's transformed demonstration bar pose --
        the same root state the object position and orientation rewards and
        the object termination compare against. Rotations are ``(N, 3, 3)``
        matrices whose columns are the bar axes.
        """
        reference = self.transform_bank.sample(self.transform_index, self.reference_index)
        root = self._cube_reference_root_states(reference)
        root_orientation = _normalize_canonical_quaternion(root[:, 3:7])
        actual_orientation = _normalize_canonical_quaternion(self.cube_orientation)
        return {
            "actual_position": self.cube_position,
            "actual_matrix": _quat_to_matrix(actual_orientation),
            "reference_position": root[:, 0:3],
            "reference_matrix": _quat_to_matrix(root_orientation),
        }

    def _palm_jacobian_arm(self) -> torch.Tensor:
        """``(num_envs, 6, 6)`` base/world-frame palm Jacobian over the six arm DOFs.

        Built from the URDF at the measured arm configuration, referenced at
        the palm origin, so no centre-of-mass transfer is needed.
        """
        return self.kinematics.jacobian(self.arm_q)

    def _refresh_contact_forces(self) -> None:
        if self.contact_sensor is None:
            return
        if self.self_collision_enabled:
            # Only the cuboid is a filter object, so the filter axis carries
            # the fingertip-cuboid force alone; finger-finger contacts, which
            # ``net_forces_w`` would now include, never reach the feature.
            forces = self.contact_sensor.data.force_matrix_w.torch.sum(dim=2)
        else:
            forces = self.contact_sensor.data.net_forces_w.torch
        self.net_contact_forces.copy_(forces[:, self._contact_sensor_order])

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _operational_space_arm_targets(self, arm_actions: torch.Tensor) -> torch.Tensor:
        max_translation = self.arm_translation_speed * self.dt
        max_rotation = self.arm_rotation_speed * self.dt
        desired_twist = torch.cat(
            (
                saturate_direction_preserving(arm_actions[:, 0:3] * max_translation, max_translation),
                saturate_direction_preserving(arm_actions[:, 3:6] * max_rotation, max_rotation),
            ),
            dim=1,
        )
        jacobian = self._palm_jacobian_arm()
        q_delta = damped_least_squares_step(jacobian, desired_twist, self.ik_damping)
        unclipped_q_delta = q_delta
        q_delta = q_delta.clamp(-self.ik_max_joint_delta, self.ik_max_joint_delta)
        previous = self.previous_arm_targets
        targets = (previous + q_delta).clamp(self.arm_lower_limits, self.arm_upper_limits)
        applied_q_delta = targets - previous
        achieved_twist = torch.bmm(jacobian, applied_q_delta.unsqueeze(-1)).squeeze(-1)
        self.requested_twist.copy_(desired_twist)
        self.achieved_twist.copy_(achieved_twist)
        self.applied_arm_q_delta.copy_(applied_q_delta)
        self.ik_residual_norm.copy_((desired_twist - achieved_twist).norm(dim=-1))
        self.arm_joint_delta_norm.copy_(applied_q_delta.norm(dim=-1))
        self.arm_joint_delta_clipped.copy_(
            (unclipped_q_delta.abs() > self.ik_max_joint_delta).any(dim=-1).to(dtype=torch.float32)
        )
        return targets

    def canonical_cube_orientation(self) -> torch.Tensor:
        return apply_cuboid_symmetry(
            _normalize_canonical_quaternion(self.cube_orientation), self.cuboid_symmetries, self.symmetry_index
        )

    def _hand_keypoints_world(self) -> torch.Tensor:
        palm_position, palm_orientation = self._palm_pose_world()
        return hand_keypoints(palm_position, palm_orientation, self._fingertip_positions_world(), self.palm_lever_arm_m)

    def _hand_keypoints_cube_frame(self) -> torch.Tensor:
        return keypoints_in_object_frame(
            self._hand_keypoints_world(), self.cube_position, self.canonical_cube_orientation()
        )

    def _palm_tilt(self) -> torch.Tensor:
        _, palm_orientation = self._palm_pose_world()
        return _quat_rotate_inverse(palm_orientation, self.world_up)

    def _task_space_observation_components(self) -> Tuple[torch.Tensor, ...]:
        palm_position, palm_orientation = self._palm_pose_world()
        if self.critic_force_observation_dim:
            self._critic_force_features = self._fingertip_force_features(palm_orientation)
        robot_root = self.robot_root_state
        robot_position = robot_root[:, 0:3]
        robot_orientation = _normalize_canonical_quaternion(robot_root[:, 3:7])
        palm_position_robot = _quat_rotate_inverse(robot_orientation, palm_position - robot_position)
        palm_orientation_robot = _normalize_canonical_quaternion(
            _quat_multiply(_quat_conjugate(robot_orientation), palm_orientation)
        )
        cube_center_palm = _quat_rotate_inverse(palm_orientation, self.cube_position - palm_position)
        cube_orientation_palm = _normalize_canonical_quaternion(
            _quat_multiply(_quat_conjugate(palm_orientation), self.canonical_cube_orientation())
        )
        if self.domain_randomization.cube_observation_noise_enabled:
            # What a pose estimator would report: the true pose plus per-step
            # noise and the episode's constant offset. Rewards and terminations
            # keep reading the true state.
            cube_center_palm, cube_orientation_palm = perturb_cube_pose_observation(
                cube_center_palm,
                cube_orientation_palm,
                self.domain_randomization.obs_cube_position_noise_m,
                self.domain_randomization.obs_cube_orientation_noise_rad,
                self.cube_observation_position_bias,
                self.cube_observation_rotation_bias,
            )
        fingertip_positions = self._fingertip_positions_world()
        fingertip_positions_palm = _quat_rotate_inverse(
            palm_orientation.unsqueeze(1).expand(-1, 5, -1), fingertip_positions - palm_position.unsqueeze(1)
        ).reshape(self.num_envs, 15)
        components = (
            palm_position_robot,
            quat_to_rotation_6d(palm_orientation_robot),
            fingertip_positions_palm,
            quat_to_rotation_6d(cube_orientation_palm),
            cube_center_palm,
        )
        if not self.contact_observation_enabled:
            return components
        return components + (self._fingertip_force_features(palm_orientation),)

    def _fingertip_force_features(self, palm_orientation: torch.Tensor) -> torch.Tensor:
        forces_world = select_fingertip_forces(self.net_contact_forces, self.contact_fingertip_indices)
        count = forces_world.shape[1]
        forces_palm = _quat_rotate_inverse(palm_orientation.unsqueeze(1).expand(-1, count, -1), forces_world)
        return fingertip_force_observation(
            forces_palm, self.contact_observation_force_scale_n, self.contact_observation_clip
        )

    @property
    def arm_q(self) -> torch.Tensor:
        return self.q[:, : len(ARM_JOINT_NAMES)]

    @property
    def arm_dq(self) -> torch.Tensor:
        return self.dq[:, : len(ARM_JOINT_NAMES)]

    @property
    def hand_q(self) -> torch.Tensor:
        return self.q[:, len(ARM_JOINT_NAMES) :]

    @property
    def hand_dq(self) -> torch.Tensor:
        return self.dq[:, len(ARM_JOINT_NAMES) :]

    @property
    def previous_arm_targets(self) -> torch.Tensor:
        return self.position_targets[:, : len(ARM_JOINT_NAMES)]

    @property
    def previous_hand_targets(self) -> torch.Tensor:
        return self.position_targets[:, len(ARM_JOINT_NAMES) :]

    # ------------------------------------------------------------------
    # Cuboid scale (generalize_size)
    # ------------------------------------------------------------------

    def _setup_object_scale(self) -> None:
        """Bind the cuboid's solver-side half extents so a reset can rewrite them.

        Newton keeps one ``shape_scale`` per shape; the MuJoCo-Warp backend
        copies it into the per-world ``geom_size`` whenever the solver is told
        that shape properties changed, so every environment can carry its own
        bar size. The binding is a torch view of the warp array: writing a row
        changes the model in place.
        """
        self._object_shape_scale = None
        if not self.object_scale_enabled:
            return
        import warp as wp  # noqa: PLC0415
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        if not hasattr(self, "_object_nominal_mass"):
            # Domain randomisation off: the nominal mass is the spawned one.
            self._object_nominal_mass = self.cube.data.default_mass.torch.to(self.device).clone()
            self._object_nominal_inertia = self.cube.data.default_inertia.torch.to(self.device).clone()
        model = NewtonManager.get_model()
        binding = self.cube._root_view.get_attribute("shape_scale", model)
        self._object_shape_scale = wp.to_torch(binding).reshape(self.num_envs, -1, 3)[:, 0]
        if self._object_shape_scale.shape != (self.num_envs, 3):
            raise RuntimeError(
                "Unexpected cuboid shape_scale binding shape {}".format(tuple(self._object_shape_scale.shape))
            )
        self._object_nominal_shape_scale = self._object_shape_scale.clone()

    def enable_object_scale(self) -> None:
        """Switch the per-episode scale machinery on after construction.

        Evaluation tools that change the bar size live (``scripts/evaluate_viser.py``)
        need the solver binding even for a run that trained at the nominal size
        only, where the configured range is ``[1, 1]`` and the constructor left
        it off. Idempotent; the scale itself is still what
        ``object_randomization.scale_min/scale_max`` say at the next reset.
        """
        if self.object_scale_enabled:
            return
        self.object_scale_enabled = True
        self._setup_object_scale()

    def _apply_object_scale(self, env_ids: torch.Tensor, scale: torch.Tensor) -> None:
        """Write the scale factor of ``env_ids`` into the solver (geometry, mass, inertia)."""
        scale = scale.to(device=self.device, dtype=torch.float32)
        self.object_scale[env_ids] = scale
        self.object_half_extents_per_env[env_ids] = self.object_half_extents.unsqueeze(0) * scale.unsqueeze(1)
        if self._object_shape_scale is None:
            return
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415
        from newton import ModelFlags  # noqa: PLC0415

        self._object_shape_scale[env_ids] = self._object_nominal_shape_scale[env_ids] * scale.unsqueeze(1)
        ids = env_ids.to(torch.int32)
        masses = self._object_nominal_mass[env_ids] * mass_factor(scale, self.object_scale_mass_with_volume).view(
            -1, 1
        )
        inertias = self._object_nominal_inertia[env_ids] * inertia_factor(
            scale, self.object_scale_mass_with_volume
        ).view(-1, 1, 1)
        self.cube.set_masses_index(masses=masses.contiguous(), env_ids=ids)
        self.cube.set_inertias_index(inertias=inertias.contiguous(), env_ids=ids)
        NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    def _cube_reference_root_states(self, sample, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert a bank track to the episode's exact continuous transform (env frame).

        A scaled bar rests higher on the table than the demonstration's, so
        the reference pose is lifted by ``half_height * (scale - 1)``; the
        rest of the track (carry, orientation) is the demonstration's.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        position = self.robot_base_position + sample.cube_pose[:, :3] * self.world_axis_sign
        x, y, z, w = sample.cube_pose[:, 3:7].unbind(dim=1)
        quaternion_world = torch.nn.functional.normalize(torch.stack((-y, x, w, -z), dim=1), dim=1)
        bank_yaw = self.transform_bank.yaw_rad[self.transform_index[env_ids]]
        bank_translation = self.transform_bank.translation[self.transform_index[env_ids]]
        half = 0.5 * (self.episode_yaw_rad[env_ids] - bank_yaw)
        delta_quaternion = torch.stack(
            (torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)), dim=1
        )
        bank_start_sample = self.transform_bank.sample(self.transform_index[env_ids], torch.zeros_like(env_ids))
        bank_start_position = self.robot_base_position + bank_start_sample.cube_pose[:, :3] * self.world_axis_sign
        actual_start_position = bank_start_position - bank_translation + self.episode_translation[env_ids]
        position = actual_start_position + _quat_rotate(delta_quaternion, position - bank_start_position)
        if self.object_scale_enabled:
            position = position.clone()
            position[:, 2] += reference_height_shift(self.object_scale[env_ids], float(self.object_half_extents[2]))
        quaternion_world = _quat_multiply(delta_quaternion, quaternion_world)
        linear_velocity = _quat_rotate(delta_quaternion, sample.cube_linear_velocity * self.world_axis_sign)
        angular_velocity = _quat_rotate(delta_quaternion, sample.cube_angular_velocity * self.world_axis_sign)
        return torch.cat((position, quaternion_world, linear_velocity, angular_velocity), dim=1)

    def _reset_cube_from_reference(self, env_ids: torch.Tensor, sample) -> None:
        state = self._cube_reference_root_states(sample, env_ids)
        pose = state[:, 0:7].clone()
        pose[:, 0:3] += self.env_origins[env_ids]
        ids = env_ids.to(torch.int32)
        self.cube.write_root_pose_to_sim_index(root_pose=pose.contiguous(), env_ids=ids)
        self.cube.write_root_velocity_to_sim_index(root_velocity=state[:, 7:13].contiguous(), env_ids=ids)

    def set_training_iteration(self, iteration: int) -> float:
        self.object_assist_scale = assist_scale_at(self.object_assist_settings, int(iteration))
        return self.object_assist_scale

    def object_reward_gate(self) -> float:
        return object_reward_gate(
            self.object_assist_enabled, self.object_assist_gates_object_reward, self.object_assist_scale
        )

    def set_object_assist_scale(self, scale: float) -> float:
        scale = float(scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("The object-assist scale must be finite and non-negative")
        self.object_assist_scale = scale
        return self.object_assist_scale

    def _compute_object_assist(self, reference) -> None:
        active = self.reference_index >= self.object_assist_settings.active_from_reference_index
        force, torque = object_assist_wrench(
            self.cube_position,
            self.cube_orientation,
            self.cube_linear_velocity,
            self.cube_angular_velocity,
            self._cube_reference_root_states(reference),
            self.object_assist_settings,
            self.object_assist_scale,
            float(self.animrl_cfg.object.mass_kg),
            self.gravity_vector,
            active,
        )
        self.cube_body_forces.copy_(force)
        self.cube_body_torques.copy_(torque)
        self.object_assist_force_n.copy_(torch.linalg.vector_norm(force, dim=1))
        self.object_assist_torque_nm.copy_(torch.linalg.vector_norm(torque, dim=1))
        self._wrench_pending = True

    def _apply_disturbances(self) -> None:
        randomization = self.domain_randomization
        if not randomization.impulses_enabled:
            return
        robot_pushed = False
        if randomization.robot_impulse_probability > 0.0 and randomization.robot_impulse_n > 0.0:
            # Arm links only: a phalanx weighs 5-45 g and the same force would
            # launch it (see finger_impulse_n).
            self.robot_body_forces.copy_(
                sample_impulses(
                    self.num_envs,
                    self.robot.num_bodies,
                    randomization.robot_impulse_probability,
                    randomization.robot_impulse_n,
                    self.device,
                    body_indices=self.arm_body_indices,
                )
            )
            robot_pushed = True
        if randomization.finger_impulses_enabled:
            finger_forces = sample_impulses(
                self.num_envs,
                self.robot.num_bodies,
                randomization.finger_impulse_probability,
                randomization.finger_impulse_n,
                self.device,
                body_indices=self.finger_body_indices,
            )
            if robot_pushed:
                self.robot_body_forces.add_(finger_forces)
            else:
                self.robot_body_forces.copy_(finger_forces)
            robot_pushed = True
        if robot_pushed:
            self.robot.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=self.robot_body_forces, torques=torch.zeros_like(self.robot_body_forces), is_global=True
            )
        if randomization.object_impulse_probability > 0.0 and randomization.object_impulse_n > 0.0:
            self.cube_body_forces += sample_impulses(
                self.num_envs,
                1,
                randomization.object_impulse_probability,
                randomization.object_impulse_n,
                self.device,
                body_indices=self.cube_body_index_tensor,
            )[:, 0]
            self._wrench_pending = True

    def scale_hand_actions(self, hand_actions: torch.Tensor) -> torch.Tensor:
        residual = (hand_actions * self.hand_action_scale).clamp(-self.action_target_clip, self.action_target_clip)
        return self.default_hand_positions + residual

    def saturated_actions(self, actions: torch.Tensor) -> torch.Tensor:
        arm = actions[:, : len(ARM_JOINT_NAMES)]
        translation_saturated = arm[:, 0:3].norm(dim=1, keepdim=True) > 1.0
        rotation_saturated = arm[:, 3:6].norm(dim=1, keepdim=True) > 1.0
        hand = actions[:, len(ARM_JOINT_NAMES) :].abs() * self.hand_action_scale > self.action_target_clip
        return torch.cat((translation_saturated.expand(-1, 3), rotation_saturated.expand(-1, 3), hand), dim=1)

    def command_targets(self, actions: torch.Tensor) -> torch.Tensor:
        targets = torch.cat(
            (
                self._operational_space_arm_targets(actions[:, : len(ARM_JOINT_NAMES)]),
                self.scale_hand_actions(actions[:, len(ARM_JOINT_NAMES) :]),
            ),
            dim=1,
        )
        self._last_command_targets.copy_(targets)
        return targets

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return (
            2.0 * (positions - self.joint_lower_limits) / (self.joint_upper_limits - self.joint_lower_limits) - 1.0
        ).clamp(-1.0, 1.0)

    def normalize_arm_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return (
            2.0 * (positions - self.arm_lower_limits) / (self.arm_upper_limits - self.arm_lower_limits) - 1.0
        ).clamp(-1.0, 1.0)

    def positions_to_hand_actions(self, hand_positions: torch.Tensor) -> torch.Tensor:
        return (hand_positions - self.default_hand_positions) / self.hand_action_scale

    def demonstration_hand_action_delta(self, reference_velocity: torch.Tensor) -> torch.Tensor:
        return reference_velocity[:, len(ARM_JOINT_NAMES) :] * self.dt / self.hand_action_scale

    @property
    def _palm_kinematics(self):
        return self.kinematics

    def next_reference_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The ideal 26-action command carrying the accumulated target onto the next reference."""
        from simtoolreal_newton.envs.retarget import pose_error

        next_indices = (self.reference_index + 1).clamp(max=self.reference.last_index)
        target = self.transform_bank.sample(self.transform_index, next_indices).q
        current_pose = self.kinematics.palm_matrices(self.previous_arm_targets)
        target_pose = self.kinematics.palm_matrices(target[:, : len(ARM_JOINT_NAMES)])
        twist = pose_error(current_pose, target_pose)
        arm_action = torch.cat(
            (
                twist[:, :3] / (self.arm_translation_speed * self.dt),
                twist[:, 3:] / (self.arm_rotation_speed * self.dt),
            ),
            dim=1,
        )
        arm_action = torch.cat(
            (
                saturate_direction_preserving(arm_action[:, 0:3], 1.0),
                saturate_direction_preserving(arm_action[:, 3:6], 1.0),
            ),
            dim=1,
        )
        hand_action = self.positions_to_hand_actions(target[:, len(ARM_JOINT_NAMES) :])
        return torch.cat((arm_action, hand_action), dim=1), target

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        """DirectRLEnv hook for the automatic resets inside :meth:`step`."""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self.reset_idx(env_ids)

    def reset_idx(
        self,
        env_ids: torch.Tensor,
        reference_indices: Optional[torch.Tensor] = None,
        transform_indices: Optional[torch.Tensor] = None,
        episode_translation: Optional[torch.Tensor] = None,
        episode_yaw_rad: Optional[torch.Tensor] = None,
    ) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device).to(dtype=torch.long)
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        randomization = self.animrl_cfg.object_randomization
        fixed_indices = list(getattr(randomization, "fixed_transform_indices", []) or [])
        if transform_indices is None and episode_translation is None and fixed_indices:
            # Single-pose curriculum: the pose comes from the named bank
            # entries only (uniformly when there are several).
            choices = torch.as_tensor(fixed_indices, device=self.device, dtype=torch.long)
            transform_indices = choices[torch.randint(len(fixed_indices), (count,), device=self.device)]
        if transform_indices is None:
            if (episode_translation is None) != (episode_yaw_rad is None):
                raise ValueError("episode_translation and episode_yaw_rad must be given together")
            if episode_translation is None:
                x = torch.empty(count, device=self.device).uniform_(
                    float(randomization.translation_x_min_m), float(randomization.translation_x_max_m)
                )
                y = torch.empty(count, device=self.device).uniform_(
                    float(randomization.translation_y_min_m), float(randomization.translation_y_max_m)
                )
                yaw = torch.empty(count, device=self.device).uniform_(
                    math.radians(float(randomization.yaw_min_deg)), math.radians(float(randomization.yaw_max_deg))
                )
                episode_translation = torch.stack((x, y, torch.zeros_like(x)), dim=1)
            else:
                episode_translation = episode_translation.to(device=self.device, dtype=torch.float32)
                yaw = episode_yaw_rad.to(device=self.device, dtype=torch.float32)
                if episode_translation.ndim == 1:
                    episode_translation = episode_translation.unsqueeze(0).repeat(count, 1)
                if yaw.ndim == 0:
                    yaw = yaw.repeat(count)
                if episode_translation.shape != (count, 3):
                    raise ValueError("episode_translation has the wrong shape")
                if yaw.shape != (count,):
                    raise ValueError("episode_yaw_rad has the wrong shape")
            transform_indices = nearest_transform_indices(
                episode_translation,
                yaw,
                self.transform_bank.translation,
                self.transform_bank.yaw_rad,
                float(randomization.nearest_yaw_lever_arm_m),
            )
            self.episode_translation[env_ids] = episode_translation
            self.episode_yaw_rad[env_ids] = yaw
        else:
            if episode_translation is not None or episode_yaw_rad is not None:
                raise ValueError("transform_indices and episode_translation are alternatives")
            transform_indices = torch.as_tensor(transform_indices, device=self.device).to(dtype=torch.long)
            if transform_indices.ndim == 0:
                transform_indices = transform_indices.repeat(count)
            if transform_indices.shape != (count,):
                raise ValueError("transform_indices has the wrong shape")
            if torch.any(transform_indices < 0) or torch.any(transform_indices >= self.transform_bank.transform_count):
                raise ValueError("Transform index outside the bank")
            self.episode_translation[env_ids] = self.transform_bank.translation[transform_indices]
            self.episode_yaw_rad[env_ids] = self.transform_bank.yaw_rad[transform_indices]
        if reference_indices is None:
            reference_indices = sample_rsi_indices(
                count,
                self.device,
                self.rsi_distribution,
                self.rsi_max_start_index,
                self.rsi_pregrasp_start_index,
                self.rsi_early_probability,
            )
        else:
            reference_indices = torch.as_tensor(reference_indices, device=self.device).to(dtype=torch.long)
            if reference_indices.ndim == 0:
                reference_indices = reference_indices.repeat(count)
            if reference_indices.shape != (count,):
                raise ValueError("reference_indices has the wrong shape")
            max_start = self.reference.last_index - 1
            if torch.any(reference_indices < 0) or torch.any(reference_indices > max_start):
                raise ValueError("RSI indices must lie in [0, {}]".format(max_start))

        # Episodes that start with the fingers already on the bar cannot
        # tolerate the continuous-vs-bank placement residual (median 16 mm):
        # the cuboid would be placed inside the retargeted fingers and the
        # contact solver fires it away (25-45% of placements at RSI 770-798,
        # measured with both contact models; 0% with the bank's own
        # transform). Those episodes use the bank entry's exact transform.
        if self.rsi_snap_placement_from_index is not None:
            snap = reference_indices >= int(self.rsi_snap_placement_from_index)
            if bool(snap.any()):
                snapped = env_ids[snap]
                self.episode_translation[snapped] = self.transform_bank.translation[transform_indices[snap]]
                self.episode_yaw_rad[snapped] = self.transform_bank.yaw_rad[transform_indices[snap]]

        # Isaac Lab bookkeeping: asset buffers, event manager, episode counter.
        DirectRLEnv._reset_idx(self, env_ids.to(torch.int32))

        # Indices drawn here are in range by construction (argmin over the
        # bank, RSI sampler); caller-supplied ones were checked above. The
        # bank's sample() itself never reads them back to the host.
        sample = self.transform_bank.sample(transform_indices, reference_indices)
        self.reference_index[env_ids] = reference_indices
        self.transform_index[env_ids] = transform_indices
        self.arm_violation_steps[env_ids] = 0
        self.hand_violation_steps[env_ids] = 0
        self.object_violation_steps[env_ids] = 0
        for values in self.episode_sums.values():
            values[env_ids] = 0.0
        reset_q, reset_dq = perturb_reference_pose(
            sample.q,
            sample.dq,
            len(ARM_JOINT_NAMES),
            self.animrl_cfg.env.rsi_position_noise_arm_rad,
            self.animrl_cfg.env.rsi_position_noise_hand_rad,
            self.animrl_cfg.env.rsi_velocity_noise_scale,
            lower_limits=self.joint_lower_limits,
            upper_limits=self.joint_upper_limits,
        )
        reset_action = torch.zeros((count, self.num_actions), dtype=torch.float32, device=self.device)
        reset_action[:, len(ARM_JOINT_NAMES) :] = self.positions_to_hand_actions(reset_q[:, len(ARM_JOINT_NAMES) :])
        self.actions[env_ids] = reset_action
        self.previous_actions[env_ids] = reset_action
        self.filtered_actions[env_ids] = reset_action
        if self.domain_randomization.cube_observation_noise_enabled:
            position_bias, rotation_bias = sample_cube_pose_bias(
                count,
                self.domain_randomization.obs_cube_position_bias_m,
                self.domain_randomization.obs_cube_orientation_bias_rad,
                self.device,
            )
            self.cube_observation_position_bias[env_ids] = position_bias
            self.cube_observation_rotation_bias[env_ids] = rotation_bias
        self.suppress_ee_action_rate[env_ids] = True
        self.action_delay.reset(env_ids, reset_action)
        self.requested_twist[env_ids] = 0.0
        self.achieved_twist[env_ids] = 0.0
        self.applied_arm_q_delta[env_ids] = 0.0
        self.ik_residual_norm[env_ids] = 0.0
        self.arm_joint_delta_norm[env_ids] = 0.0
        self.arm_joint_delta_clipped[env_ids] = 0.0

        ids = env_ids.to(torch.int32)
        self.robot.write_joint_state_to_sim_index(
            position=reset_q.contiguous(), velocity=reset_dq.contiguous(), env_ids=ids
        )
        self.position_targets[env_ids] = sample.q
        self.applied_targets[env_ids] = sample.q
        self.robot.actuators.target_command.set_position_index(value=sample.q.contiguous(), env_ids=ids)
        if self.object_scale_enabled:
            self._apply_object_scale(env_ids, sample_scales(count, randomization, self.device))
        self._reset_cube_from_reference(env_ids, sample)
        reference_root = self._cube_reference_root_states(sample, env_ids)
        _, chosen_symmetry = canonicalize_cuboid_orientation(
            _normalize_canonical_quaternion(self.cube_orientation[env_ids]),
            self.cuboid_symmetries,
            _normalize_canonical_quaternion(reference_root[:, 3:7]),
            return_index=True,
        )
        self.symmetry_index[env_ids] = chosen_symmetry.to(dtype=self.symmetry_index.dtype)
        reset_object_height = self.cube_position[env_ids, 2]
        self.episode_initial_object_com_height_m[env_ids] = reset_object_height
        self.episode_peak_object_com_height_m[env_ids] = reset_object_height

    def reset_all(
        self,
        reference_index: Optional[int] = None,
        translation_xy: Optional[Tuple[float, float]] = None,
        yaw_rad: Optional[float] = None,
    ) -> torch.Tensor:
        """AnimRL-style full reset returning the policy observation."""
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        indices = None
        if reference_index is not None:
            indices = torch.full((self.num_envs,), int(reference_index), dtype=torch.long, device=self.device)
        translation = yaw = None
        if translation_xy is not None or yaw_rad is not None:
            if translation_xy is None or yaw_rad is None:
                raise ValueError("translation_xy and yaw_rad must be given together")
            translation = torch.tensor(
                (float(translation_xy[0]), float(translation_xy[1]), 0.0), device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1)
            yaw = torch.full((self.num_envs,), float(yaw_rad), device=self.device)
        self.reset_idx(env_ids, indices, episode_translation=translation, episode_yaw_rad=yaw)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        self.compute_observations()
        self.obs_buf = self._observation_dict()
        return self.policy_obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Gymnasium reset (used by the Isaac Lab RL entry points)."""
        if seed is not None:
            self.seed(seed)
        self.reset_all()
        return self.obs_buf, self.extras

    # ------------------------------------------------------------------
    # Observations, rewards, terminations
    # ------------------------------------------------------------------

    def compute_observations(self) -> None:
        self._refresh_contact_forces()
        phase = (self.reference_index.float() / float(self.reference.last_index)).unsqueeze(1)
        measured_q, measured_dq = self.q, self.dq
        if self.domain_randomization.observation_noise_enabled:
            measured_q, measured_dq = add_observation_noise(
                self.q,
                self.dq,
                self.domain_randomization.obs_q_noise_rad,
                self.domain_randomization.obs_dq_noise_rad_s,
                self.domain_randomization.obs_q_bias_rad,
                bias=self.observation_position_bias,
            )
        parts = [
            self.normalize_positions(measured_q),
            self.previous_targets,
            measured_dq,
            phase,
            *self._task_space_observation_components(),
        ]
        if self.object_scale_observed:
            parts.append(
                scale_observation(self.object_scale, self.object_scale_observation_override)
            )
        self.policy_obs.copy_(torch.cat(parts, dim=1))
        # Last line of defence against a blown-up world: the env is being
        # terminated (see _get_dones), and the policy must never see NaN.
        torch.nan_to_num_(self.policy_obs, nan=0.0, posinf=0.0, neginf=0.0)
        if self.critic_obs is not None:
            parts = [self.policy_obs]
            if self.critic_force_observation_dim:
                if self._critic_force_features is None:
                    raise RuntimeError("Privileged critic observation requested but the forces were never computed")
                parts.append(self._critic_force_features)
            if self.critic_parameter_table is not None:
                parts.append(self.critic_parameter_table)
            self.critic_obs.copy_(torch.cat(parts, dim=1))
            torch.nan_to_num_(self.critic_obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _observation_dict(self) -> dict:
        observations = {"policy": self.policy_obs}
        if self.critic_obs is not None:
            observations["critic"] = self.critic_obs
        return observations

    def get_observations(self) -> torch.Tensor:
        return self.policy_obs

    def get_privileged_observations(self):
        return self.critic_obs

    def _get_observations(self) -> dict:
        self.compute_observations()
        return self._observation_dict()

    def _compute_reward_and_errors(self) -> Dict[str, torch.Tensor]:
        reference = self.transform_bank.sample(self.transform_index, self.reference_index)
        n_arm = len(ARM_JOINT_NAMES)
        arm_q_error = self.arm_q - reference.q[:, :n_arm]
        arm_dq_error = self.arm_dq - reference.dq[:, :n_arm]
        hand_q_error = self.hand_q - reference.q[:, n_arm:]
        hand_dq_error = self.hand_dq - reference.dq[:, n_arm:]
        action_delta = self.actions - self.previous_actions
        hand_action_delta_error = action_delta[:, n_arm:] - self.demonstration_hand_action_delta(reference.dq)
        ee_action_delta = torch.where(
            self.suppress_ee_action_rate.unsqueeze(1), torch.zeros_like(action_delta[:, :n_arm]), action_delta[:, :n_arm]
        )
        palm_tilt = self._palm_tilt()
        reference_palm_tilt = self.transform_bank.palm_tilt_at(self.reference_index)
        palm_tilt_mse = (palm_tilt - reference_palm_tilt).square().sum(dim=1)
        palm_tilt_error_rad = torch.arccos((palm_tilt * reference_palm_tilt).sum(dim=1).clamp(-1.0, 1.0))

        rewards_cfg = self.animrl_cfg.rewards
        position_mse = arm_q_error.square().mean(dim=1)
        velocity_mse = arm_dq_error.square().mean(dim=1)
        ee_action_rate_mse = ee_action_delta.square().mean(dim=1)
        arm_joint_rate_mse = self.applied_arm_q_delta.square().mean(dim=1)
        ik_residual_mse = self.ik_residual_norm.square()
        hand_position_mse = hand_q_error.square().mean(dim=1)
        hand_velocity_mse = hand_dq_error.square().mean(dim=1)
        hand_action_rate_mse = hand_action_delta_error.square().mean(dim=1)

        reference_cube_root_state = self._cube_reference_root_states(reference)
        object_position_error = self.cube_position - reference_cube_root_state[:, 0:3]
        object_position_error_m = torch.linalg.vector_norm(object_position_error, dim=1)
        object_com_height_m = self.cube_position[:, 2]
        object_com_lift_m = object_com_height_m - self.episode_initial_object_com_height_m
        actual_cube_orientation = _normalize_canonical_quaternion(self.cube_orientation)
        reference_cube_orientation = _normalize_canonical_quaternion(reference_cube_root_state[:, 3:7])
        object_orientation_dot = (actual_cube_orientation * reference_cube_orientation).sum(dim=1).abs().clamp(max=1.0)
        object_orientation_error_rad = 2.0 * torch.acos(object_orientation_dot)

        gaussian = lambda mse, std: torch.exp(-mse / (2.0 * float(std) ** 2))  # noqa: E731

        def width(name, configured, mse):
            tracker = self.adaptive_sigmas.get(name)
            if tracker is None:
                return float(configured)
            return tracker.update(float(mse.mean().item()))

        position_hand_std = width("position_hand", rewards_cfg.position_hand_std_rad, hand_position_mse)
        ee_action_rate_std = width("ee_action_rate", rewards_cfg.ee_action_rate_std, ee_action_rate_mse)
        hand_action_rate_std = width("hand_action_rate", rewards_cfg.hand_action_rate_std, hand_action_rate_mse)
        palm_tilt_reward = gaussian(palm_tilt_mse, rewards_cfg.palm_tilt_std_rad)
        ee_action_rate_reward = gaussian(ee_action_rate_mse, ee_action_rate_std)
        arm_joint_rate_reward = gaussian(arm_joint_rate_mse, rewards_cfg.arm_joint_rate_std_rad)
        ik_residual_reward = gaussian(ik_residual_mse, rewards_cfg.ik_residual_std)
        hand_position_reward = gaussian(hand_position_mse, position_hand_std)
        hand_velocity_reward = gaussian(hand_velocity_mse, rewards_cfg.velocity_hand_std_rad_per_s)
        hand_action_rate_reward = gaussian(hand_action_rate_mse, hand_action_rate_std)
        object_position_reward = gaussian(object_position_error_m.square(), rewards_cfg.object_position_std_m)
        object_orientation_reward = gaussian(
            object_orientation_error_rad.square(), rewards_cfg.object_orientation_std_rad
        )
        selected_fingertips_world = self._fingertip_positions_world()[:, self.proximity_fingertip_indices]
        cube_orientation_expanded = actual_cube_orientation.unsqueeze(1).expand(-1, selected_fingertips_world.shape[1], -1)
        selected_fingertips_cube = _quat_rotate_inverse(
            cube_orientation_expanded, selected_fingertips_world - self.cube_position.unsqueeze(1)
        )
        proximity_active = self.reference_index >= self.rsi_pregrasp_start_index
        (
            fingertip_object_distance_reward,
            fingertip_object_distance_m,
            fingertip_object_distance_per_finger_m,
        ) = fingertip_cuboid_proximity(
            selected_fingertips_cube, self.object_half_extents_per_env, self.proximity_std_m, proximity_active
        )

        keypoints_world = self._hand_keypoints_world()
        keypoints_cube_frame = keypoints_in_object_frame(
            keypoints_world, self.cube_position, self.canonical_cube_orientation()
        )
        reference_keypoints = self.transform_bank.keypoints_at(self.reference_index)
        _, actual_fingertips = split_palm_and_fingertips(keypoints_cube_frame)
        reference_palm, reference_fingertips = split_palm_and_fingertips(reference_keypoints)
        if self.palm_keypoint_anchor == "reference":
            keypoints_reference_frame = keypoints_in_object_frame(
                keypoints_world, reference_cube_root_state[:, 0:3], reference_cube_orientation
            )
            actual_palm, _ = split_palm_and_fingertips(keypoints_reference_frame)
        else:
            actual_palm, _ = split_palm_and_fingertips(keypoints_cube_frame)
        palm_keypoint_mse = keypoint_tracking_error(actual_palm, reference_palm)
        fingertip_keypoint_mse = keypoint_tracking_error(actual_fingertips, reference_fingertips)
        palm_keypoint_error_m = palm_keypoint_mse.clamp_min(0.0).sqrt()
        fingertip_keypoint_error_m = fingertip_keypoint_mse.clamp_min(0.0).sqrt()
        palm_keypoint_reward = keypoint_gaussian(palm_keypoint_mse, rewards_cfg.palm_keypoint_std_m)
        fingertip_keypoint_reward = keypoint_gaussian(fingertip_keypoint_mse, rewards_cfg.fingertip_keypoint_std_m)
        if self.contact_enabled:
            (
                fingertip_contact_reward,
                fingertip_contact_fraction,
                mean_fingertip_contact_force_n,
            ) = fingertip_contact_diagnostics(
                self.net_contact_forces, self.contact_fingertip_indices, self.contact_force_threshold_n
            )
            fingertip_force_n = fingertip_force_norms(self.net_contact_forces, self.fingertip_body_indices.new_tensor(range(5)))
        else:
            fingertip_contact_reward = torch.zeros_like(palm_keypoint_reward)
            fingertip_contact_fraction = torch.zeros_like(palm_keypoint_reward)
            mean_fingertip_contact_force_n = torch.zeros_like(palm_keypoint_reward)
            fingertip_force_n = palm_keypoint_reward.new_zeros((self.num_envs, len(FINGERTIP_BODY_NAMES)))
        self.rew_buf.copy_(
            float(rewards_cfg.palm_keypoint_weight) * palm_keypoint_reward
            + float(rewards_cfg.fingertip_keypoint_weight) * fingertip_keypoint_reward
            + float(rewards_cfg.palm_tilt_weight) * palm_tilt_reward
            + float(rewards_cfg.ee_action_rate_weight) * ee_action_rate_reward
            + float(rewards_cfg.arm_joint_rate_weight) * arm_joint_rate_reward
            + float(rewards_cfg.ik_residual_weight) * ik_residual_reward
            + float(rewards_cfg.position_hand_weight) * hand_position_reward
            + float(rewards_cfg.velocity_hand_weight) * hand_velocity_reward
            + float(rewards_cfg.hand_action_rate_weight) * hand_action_rate_reward
            + self.object_reward_gate()
            * (
                float(rewards_cfg.object_position_weight) * object_position_reward
                + float(rewards_cfg.object_orientation_weight) * object_orientation_reward
            )
            + self.proximity_weight * fingertip_object_distance_reward
            + self.contact_shaping_weight * fingertip_contact_reward
        )
        return {
            "palm_keypoint_error_m": palm_keypoint_error_m,
            "fingertip_keypoint_error_m": fingertip_keypoint_error_m,
            "palm_keypoint_reward": palm_keypoint_reward,
            "fingertip_keypoint_reward": fingertip_keypoint_reward,
            "q_error": arm_q_error,
            "dq_error": arm_dq_error,
            "hand_q_error": hand_q_error,
            "hand_dq_error": hand_dq_error,
            "position_mse": position_mse,
            "velocity_mse": velocity_mse,
            "ee_action_rate_mse": ee_action_rate_mse,
            "arm_joint_rate_mse": arm_joint_rate_mse,
            "hand_position_mse": hand_position_mse,
            "hand_velocity_mse": hand_velocity_mse,
            "hand_action_rate_mse": hand_action_rate_mse,
            "palm_tilt_reward": palm_tilt_reward,
            "palm_tilt_error_rad": palm_tilt_error_rad,
            "ee_action_rate_reward": ee_action_rate_reward,
            "arm_joint_rate_reward": arm_joint_rate_reward,
            "ik_residual_reward": ik_residual_reward,
            "ik_residual_norm": self.ik_residual_norm,
            "arm_joint_delta_norm": self.arm_joint_delta_norm,
            "arm_joint_delta_clipped": self.arm_joint_delta_clipped,
            "requested_twist": self.requested_twist,
            "achieved_twist": self.achieved_twist,
            "hand_position_reward": hand_position_reward,
            "hand_velocity_reward": hand_velocity_reward,
            "hand_action_rate_reward": hand_action_rate_reward,
            "object_position_error_m": object_position_error_m,
            "object_com_height_m": object_com_height_m,
            "object_com_lift_m": object_com_lift_m,
            "object_orientation_error_rad": object_orientation_error_rad,
            "object_position_reward": object_position_reward,
            "object_orientation_reward": object_orientation_reward,
            "fingertip_object_distance_reward": fingertip_object_distance_reward,
            "fingertip_object_distance_m": fingertip_object_distance_m,
            "fingertip_object_distance_per_finger_m": fingertip_object_distance_per_finger_m,
            "proximity_active": proximity_active.to(dtype=fingertip_object_distance_m.dtype),
            "fingertip_contact_reward": fingertip_contact_reward,
            "fingertip_contact_fraction": fingertip_contact_fraction,
            "mean_fingertip_contact_force_n": mean_fingertip_contact_force_n,
            "fingertip_force_n": fingertip_force_n,
        }

    def threshold_violation(self, palm_keypoint_error_m: torch.Tensor) -> torch.Tensor:
        return palm_keypoint_error_m > float(self.animrl_cfg.termination.palm_keypoint_threshold_m)

    def hand_threshold_violation(self, hand_q_error: torch.Tensor) -> torch.Tensor:
        return hand_q_error.abs().amax(dim=1) > float(self.animrl_cfg.termination.hand_position_threshold_rad)

    def object_threshold_violation(self, object_position_error_m: torch.Tensor) -> torch.Tensor:
        return object_position_error_m > float(self.animrl_cfg.termination.object_position_threshold_m)

    def _compute_termination(self, palm_keypoint_error_m, hand_q_error, object_position_error_m):
        termination = self.animrl_cfg.termination
        if bool(termination.enabled):
            self.arm_violation = self.threshold_violation(palm_keypoint_error_m)
            self.hand_violation = self.hand_threshold_violation(hand_q_error)
            if bool(termination.object_position_enabled):
                self.object_violation = self.object_threshold_violation(object_position_error_m)
            else:
                self.object_violation = torch.zeros_like(self.arm_violation)
            grace = int(termination.grace_steps)
            for flags, steps in (
                (self.arm_violation, self.arm_violation_steps),
                (self.hand_violation, self.hand_violation_steps),
                (self.object_violation, self.object_violation_steps),
            ):
                steps.copy_(torch.where(flags, steps + 1, torch.zeros_like(steps)))
            early = (
                (self.arm_violation_steps >= grace)
                | (self.hand_violation_steps >= grace)
                | (self.object_violation_steps >= grace)
            )
        else:
            early = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.arm_violation = torch.zeros_like(early)
            self.hand_violation = torch.zeros_like(early)
            self.object_violation = torch.zeros_like(early)
        reference_end = self.reference_index >= self.reference.last_index
        timeout = (self.episode_length_buf >= self.max_episode_length) | reference_end
        return early | timeout, early, timeout

    def _accumulate_episode_metrics(self, metrics: Dict[str, torch.Tensor]) -> None:
        self.episode_peak_object_com_height_m.copy_(
            torch.maximum(self.episode_peak_object_com_height_m, metrics["object_com_height_m"])
        )
        sums = self.episode_sums
        sums["reward"] += self.rew_buf
        for name in (
            "palm_keypoint_reward",
            "fingertip_keypoint_reward",
            "palm_keypoint_error_m",
            "fingertip_keypoint_error_m",
            "palm_tilt_reward",
            "palm_tilt_error_rad",
            "ee_action_rate_reward",
            "arm_joint_rate_reward",
            "ik_residual_reward",
            "ik_residual_norm",
            "arm_joint_delta_norm",
            "arm_joint_delta_clipped",
            "hand_position_reward",
            "hand_velocity_reward",
            "hand_action_rate_reward",
            "object_position_reward",
            "object_orientation_reward",
            "fingertip_object_distance_reward",
            "fingertip_object_distance_m",
            "object_position_error_m",
            "object_orientation_error_rad",
            "fingertip_contact_reward",
            "fingertip_contact_fraction",
        ):
            sums[name] += metrics[name]
        sums["fingertip_contact_force_n"] += metrics["mean_fingertip_contact_force_n"]
        sums["rms_hand_position_error"] += metrics["hand_position_mse"].sqrt()
        sums["rms_hand_velocity_error"] += metrics["hand_velocity_mse"].sqrt()
        sums["rms_hand_action_rate"] += metrics["hand_action_rate_mse"].sqrt()
        sums["rms_position_error"] += metrics["position_mse"].sqrt()
        sums["rms_velocity_error"] += metrics["velocity_mse"].sqrt()
        sums["rms_ee_action_rate"] += metrics["ee_action_rate_mse"].sqrt()
        sums["rms_arm_joint_rate"] += metrics["arm_joint_rate_mse"].sqrt()
        sums["object_assist_force_n"] += self.object_assist_force_n
        sums["object_assist_torque_nm"] += self.object_assist_torque_nm

    def _build_episode_summary(self, done, early, horizon_timeout, reference_end) -> Dict[str, torch.Tensor]:
        lengths = self.episode_length_buf[done].float().clamp_min(1.0)
        peak = self.episode_peak_object_com_height_m
        lift = peak - self.episode_initial_object_com_height_m
        summary = {
            "return": self.episode_sums["reward"][done].mean(),
            "length": lengths.mean(),
            "palm_keypoint_reward": (self.episode_sums["palm_keypoint_reward"][done] / lengths).mean(),
            "fingertip_keypoint_reward": (self.episode_sums["fingertip_keypoint_reward"][done] / lengths).mean(),
            "palm_keypoint_error_m": (self.episode_sums["palm_keypoint_error_m"][done] / lengths).mean(),
            "fingertip_keypoint_error_m": (self.episode_sums["fingertip_keypoint_error_m"][done] / lengths).mean(),
            "early_termination_fraction": early[done].float().mean(),
            "arm_failure_fraction": (early & self.arm_violation)[done].float().mean(),
            "hand_failure_fraction": (early & self.hand_violation)[done].float().mean(),
            "object_failure_fraction": (early & self.object_violation)[done].float().mean(),
            "horizon_fraction": horizon_timeout[done].float().mean(),
            "reference_end_fraction": reference_end[done].float().mean(),
            "completed_episodes": done.sum().to(dtype=torch.float32),
            "mean_peak_object_com_height_m": peak[done].mean(),
            "max_peak_object_com_height_m": peak[done].max(),
            "mean_peak_object_com_lift_m": lift[done].mean(),
            "max_peak_object_com_lift_m": lift[done].max(),
        }
        for name, values in self.episode_sums.items():
            summary["mean_{}".format(name)] = (values[done] / lengths).mean()
        return summary

    # ------------------------------------------------------------------
    # DirectRLEnv step hooks
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                "Actions have shape {}, expected {}".format(tuple(actions.shape), (self.num_envs, self.num_actions))
            )
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.to(device=self.device, dtype=torch.float32))
        delayed = self.action_delay(self.actions)
        if self.action_filter_alpha < 1.0:
            self.filtered_actions.lerp_(delayed, self.action_filter_alpha)
        else:
            self.filtered_actions.copy_(delayed)
        complete_target_q = self.command_targets(self.filtered_actions)
        self.position_targets.copy_(complete_target_q)
        if self.object_assist_enabled or self.domain_randomization.impulses_enabled:
            self.cube_body_forces.zero_()
            self.cube_body_torques.zero_()
        if self.object_assist_enabled:
            next_indices = (self.reference_index + 1).clamp(max=self.reference.last_index)
            next_reference = self.transform_bank.sample(self.transform_index, next_indices)
            self._compute_object_assist(next_reference)
        self._apply_disturbances()

    def _apply_action(self) -> None:
        # PhysX clamped every joint at the URDF velocity limit (3.14 rad/s);
        # MuJoCo-Warp does not enforce joint velocity limits, and a residual
        # hand target that jumps 0.4 rad would fling a 5 g phalanx at 80 rad/s
        # into the cuboid. Slewing the *applied* target at that limit gives the
        # same arrival time as the clamped joint had, with bounded drive
        # torque. The observation still reports the commanded target.
        delta = (self.position_targets - self.applied_targets).clamp(
            -self.target_slew_per_step, self.target_slew_per_step
        )
        self.applied_targets.add_(delta)
        self.robot.actuators.target_command.set_position_index(value=self.applied_targets)
        if self._wrench_pending:
            self.cube.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=self.cube_body_forces.unsqueeze(1),
                torques=self.cube_body_torques.unsqueeze(1),
                is_global=True,
            )
            self._wrench_pending = False

    def _clamp_object_velocity(self) -> None:
        """Bound the cuboid's velocity the way PhysX's depenetration cap did."""
        if self.object_max_linear_velocity <= 0.0 and self.object_max_angular_velocity <= 0.0:
            return
        velocity = self.cube.data.root_com_vel_w.torch
        linear, angular = velocity[:, 0:3], velocity[:, 3:6]
        clamped = velocity.clone()
        if self.object_max_linear_velocity > 0.0:
            speed = linear.norm(dim=1, keepdim=True)
            clamped[:, 0:3] = linear * (self.object_max_linear_velocity / speed.clamp_min(1e-9)).clamp(max=1.0)
        if self.object_max_angular_velocity > 0.0:
            rate = angular.norm(dim=1, keepdim=True)
            clamped[:, 3:6] = angular * (self.object_max_angular_velocity / rate.clamp_min(1e-9)).clamp(max=1.0)
        # Written every step. Deciding on the host whether anything changed
        # would cost a GPU->CPU synchronisation per step; writing an unchanged
        # velocity back costs one small copy.
        self.cube.write_root_velocity_to_sim_index(root_velocity=clamped.contiguous())

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._clamp_object_velocity()
        self.reference_index.add_(1).clamp_(max=self.reference.last_index)
        self._refresh_contact_forces()
        metrics = self._compute_reward_and_errors()
        self.suppress_ee_action_rate.fill_(False)
        self._accumulate_episode_metrics(metrics)
        _, early, timeout = self._compute_termination(
            metrics["palm_keypoint_error_m"], metrics["hand_q_error"], metrics["object_position_error_m"]
        )
        # A contact blow-up leaves NaN in one world's state. Left alone it
        # reaches the policy as a NaN observation and kills the run (twice in
        # this lineage: s2s_track3_palm08 at 14014, s2s_track5_quiet at
        # 17898). Terminate and reset that env instead; its reward and
        # observation are zeroed below so nothing non-finite is learned from.
        blown = ~(
            torch.isfinite(self.q).all(dim=1)
            & torch.isfinite(self.dq).all(dim=1)
            & torch.isfinite(self.cube_root_state).all(dim=1)
        )
        self.sim_blowup = blown
        if bool(blown.any()):
            early = early | blown
            # The metrics of a blown env were accumulated above with NaN in
            # them, and one NaN in the episode sums makes the whole
            # iteration's mean return NaN in the log and in TensorBoard.
            # Drop that env's partial episode from the statistics.
            for values in self.episode_sums.values():
                values[blown] = 0.0
            self.episode_peak_object_com_height_m[blown] = self.episode_initial_object_com_height_m[blown]
            print(
                "[env] {} env(s) with non-finite state at reference index {}; resetting them".format(
                    int(blown.sum()), int(self.reference_index[blown][0])
                ),
                flush=True,
            )
        self._metrics = metrics
        self._early = early
        self._timeout = timeout
        return early, timeout

    def _get_rewards(self) -> torch.Tensor:
        metrics = self._metrics
        early, timeout = self._early, self._timeout
        done = self.reset_buf
        reference_end = self.reference_index >= self.reference.last_index
        horizon_timeout = self.episode_length_buf >= self.max_episode_length
        extras = {
            "time_outs": timeout.clone(),
            "horizon_time_outs": horizon_timeout.clone(),
            "reference_end": reference_end.clone(),
            "early_termination": early.clone(),
            "arm_threshold_violation": self.arm_violation.clone(),
            "hand_threshold_violation": self.hand_violation.clone(),
            "object_threshold_violation": self.object_violation.clone(),
            "reference_index": self.reference_index.clone(),
            "max_abs_position_error": metrics["q_error"].abs().amax(dim=1),
            "max_abs_arm_position_error": metrics["q_error"].abs().amax(dim=1),
            "max_abs_hand_position_error": metrics["hand_q_error"].abs().amax(dim=1),
            "worst_joint_index": metrics["q_error"].abs().argmax(dim=1),
            "rms_position_error": metrics["position_mse"].sqrt(),
            "rms_velocity_error": metrics["velocity_mse"].sqrt(),
            "rms_ee_action_rate": metrics["ee_action_rate_mse"].sqrt(),
            "rms_arm_joint_rate": metrics["arm_joint_rate_mse"].sqrt(),
            "rms_hand_position_error": metrics["hand_position_mse"].sqrt(),
            "rms_hand_velocity_error": metrics["hand_velocity_mse"].sqrt(),
            "rms_hand_action_rate": metrics["hand_action_rate_mse"].sqrt(),
            "palm_keypoint_reward": metrics["palm_keypoint_reward"],
            "fingertip_keypoint_reward": metrics["fingertip_keypoint_reward"],
            "palm_keypoint_error_m": metrics["palm_keypoint_error_m"],
            "fingertip_keypoint_error_m": metrics["fingertip_keypoint_error_m"],
            "palm_tilt_reward": metrics["palm_tilt_reward"],
            "palm_tilt_error_rad": metrics["palm_tilt_error_rad"],
            "ee_action_rate_reward": metrics["ee_action_rate_reward"],
            "arm_joint_rate_reward": metrics["arm_joint_rate_reward"],
            "ik_residual_reward": metrics["ik_residual_reward"],
            "ik_residual_norm": metrics["ik_residual_norm"].clone(),
            "arm_joint_delta_norm": metrics["arm_joint_delta_norm"].clone(),
            "arm_joint_delta_clipped": metrics["arm_joint_delta_clipped"].clone(),
            "hand_position_reward": metrics["hand_position_reward"],
            "hand_velocity_reward": metrics["hand_velocity_reward"],
            "hand_action_rate_reward": metrics["hand_action_rate_reward"],
            "object_position_reward": metrics["object_position_reward"],
            "object_orientation_reward": metrics["object_orientation_reward"],
            "fingertip_object_distance_reward": metrics["fingertip_object_distance_reward"],
            "fingertip_object_distance_m": metrics["fingertip_object_distance_m"],
            "object_position_error_m": metrics["object_position_error_m"],
            "object_com_height_m": metrics["object_com_height_m"].clone(),
            "object_com_lift_m": metrics["object_com_lift_m"].clone(),
            "object_orientation_error_rad": metrics["object_orientation_error_rad"],
            "fingertip_contact_reward": metrics["fingertip_contact_reward"],
            "fingertip_contact_fraction": metrics["fingertip_contact_fraction"],
            "mean_fingertip_contact_force_n": metrics["mean_fingertip_contact_force_n"],
            "fingertip_force_n": metrics["fingertip_force_n"],
            "fingertip_object_distance_per_finger_m": metrics["fingertip_object_distance_per_finger_m"],
            "proximity_active": metrics["proximity_active"],
            "object_assist_force_n": self.object_assist_force_n.clone(),
            "object_assist_torque_nm": self.object_assist_torque_nm.clone(),
            "object_assist_scale": self.object_assist_scale,
        }
        if bool(done.any()):
            summary = self._build_episode_summary(done, early, horizon_timeout, reference_end)
            extras["episode"] = summary
            extras["log"] = {
                "Episode/return": summary["return"],
                "Episode/length": summary["length"],
                "Episode/early_termination_fraction": summary["early_termination_fraction"],
                "Episode/reference_end_fraction": summary["reference_end_fraction"],
                "Episode/palm_keypoint_error_m": summary["palm_keypoint_error_m"],
                "Episode/fingertip_keypoint_error_m": summary["fingertip_keypoint_error_m"],
                "Episode/mean_peak_object_com_lift_m": summary["mean_peak_object_com_lift_m"],
            }
        self.extras = extras
        # Non-finite state (see _get_dones) makes every term NaN; zero it so
        # the value target stays finite for the env being reset.
        torch.nan_to_num_(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
        return self.rew_buf.clone()

    # ------------------------------------------------------------------
    # Misc AnimRL surface
    # ------------------------------------------------------------------

    def viewer_closed(self) -> bool:
        return not self.sim.is_headless_or_exist_active_visualizer()

    def render(self, *args, **kwargs):
        return None

    def capture_training_camera_frame(self, env_index: int = 0) -> np.ndarray:
        """One RGB frame ``(H, W, 3)`` of ``env_index`` from the optional Warp camera."""
        if self.camera is None:
            raise RuntimeError("No camera was configured; build the environment with a camera cfg")
        self.camera.update(self.physics_dt, force_recompute=True)
        rgb = self.camera.data.output["rgb"]
        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
        frame = rgb[env_index]
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != torch.uint8:
            frame = (frame.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return frame.detach().cpu().numpy()

    def pd_gain_summary(self) -> Dict[str, np.ndarray]:
        stiffness = np.asarray(self.pd_properties["stiffness"], dtype=np.float64)
        damping = np.asarray(self.pd_properties["damping"], dtype=np.float64)
        n_arm = len(ARM_JOINT_NAMES)
        return {
            "arm_stiffness": stiffness[:n_arm],
            "arm_damping": damping[:n_arm],
            "hand_stiffness": stiffness[n_arm:],
            "hand_damping": damping[n_arm:],
        }
