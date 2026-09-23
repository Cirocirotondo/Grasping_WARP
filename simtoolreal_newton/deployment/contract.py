"""The policy contract off the simulator: what it observes, what it commands.

Deployment must feed the network the observation the environment built and
turn its 26 outputs into joint targets the way the environment did, or the
policy runs in a different world than the one it was trained in. Rather than
restate that arithmetic, this module calls the environment's own building
blocks -- the URDF kinematics and its palm Jacobian, the damped least-squares
step, the rotation helpers, the cuboid-symmetry canonicaliser -- on the CPU, one
sample per control step. Nothing here imports Isaac Lab, so it runs in a plain
torch process next to the robot; ``scripts/check_deployment_contract.py`` drives
the real environment alongside it and asserts that the two agree.

Frames. The URDF base frame is the environment's world frame up to the fixed
translation ``init_state.pos`` (the environment asserts the base is unrotated).
Every pose here is in that base frame. The demonstration and the transform bank
store the cuboid in the *UR controller's* base convention instead, and
:func:`~simtoolreal_newton.envs.retarget.cube_pose_to_base_frame` is the one
mapping between the two -- the same one the environment applies at reset.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch

from simtoolreal_newton import ROOT_DIR
from simtoolreal_newton.cfg import (
    SimToolRealCfg,
    SimToolRealTrainCfg,
    update_config_from_dict,
)
from simtoolreal_newton.envs.controller import (
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    JOINT_NAMES,
)
from simtoolreal_newton.envs.cuboid_symmetry import (
    apply_cuboid_symmetry,
    canonicalize_cuboid_orientation,
    cuboid_rotation_symmetries,
)
from simtoolreal_newton.envs.demonstration import JointDemonstration60Hz
from simtoolreal_newton.envs.kinematics import PalmKinematics
from simtoolreal_newton.envs.object_scale import (
    object_scale_observation_dim,
    observe_scale,
    observed_scale_override,
    reference_height_shift,
    scale_observation,
    scale_range,
)
from simtoolreal_newton.envs.operational_space import (
    damped_least_squares_step,
    saturate_direction_preserving,
)
from simtoolreal_newton.envs.retarget import cube_pose_to_base_frame, pose_error
from simtoolreal_newton.envs.rotations import (
    normalize_canonical_quaternion,
    quat_conjugate,
    quat_multiply,
    quat_rotate_inverse,
    quat_to_matrix,
    quat_to_rotation_6d,
)
from simtoolreal_newton.envs.transform_bank import (
    BankSample,
    TransformBank,
    nearest_transform_indices,
)
from simtoolreal_newton.runners.modules.normalizer import EmpiricalNormalization
from simtoolreal_newton.runners.modules.policy import Policy

ARM_DOF = len(ARM_JOINT_NAMES)
HAND_DOF = len(HAND_JOINT_NAMES)
ACTION_DIM = len(JOINT_NAMES)
# 26 normalised positions, 26 previous targets, 26 velocities, phase.
PROPRIOCEPTION_DIM = 3 * ACTION_DIM + 1
# Palm position (3) and 6D rotation (6) in the base frame, five fingertips in
# the palm frame (15), cuboid 6D rotation (6) and centre (3) in the palm frame.
TASK_SPACE_DIM = 3 + 6 + 15 + 6 + 3
# A checkpoint that observes the cuboid scale appends one more column.
OBSERVATION_DIM = PROPRIOCEPTION_DIM + TASK_SPACE_DIM

PathLike = Union[str, Path]


def load_saved_configuration(config_path: PathLike):
    """Rebuild the run's ``(env_cfg, train_cfg)`` from its ``config.json``."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        saved = json.load(stream)
    if "env_cfg" not in saved or "train_cfg" not in saved:
        raise ValueError("{} must contain env_cfg and train_cfg".format(config_path))
    env_cfg = SimToolRealCfg()
    train_cfg = SimToolRealTrainCfg()
    update_config_from_dict(env_cfg, saved["env_cfg"], strict=False)
    update_config_from_dict(train_cfg, saved["train_cfg"], strict=False)
    return env_cfg, train_cfg


def urdf_velocity_limits(urdf_path: PathLike, joint_names) -> torch.Tensor:
    """Per-joint ``velocity`` limits from the URDF, in ``joint_names`` order."""
    root = ET.parse(str(urdf_path)).getroot()
    limits = {}
    for joint in root.iter("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("velocity") is not None:
            limits[joint.get("name")] = float(limit.get("velocity"))
    missing = [name for name in joint_names if name not in limits]
    if missing:
        raise ValueError("The URDF declares no velocity limit for {}".format(missing))
    return torch.tensor([limits[name] for name in joint_names], dtype=torch.float32)


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def validate_object_scale(object_scale) -> float:
    scale = float(object_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("The object scale must be a positive number")
    return scale


class Placement(NamedTuple):
    """A cuboid start pose as the transform bank parameterises it."""

    translation: torch.Tensor  # (3,), base frame, relative to the demonstration's start
    yaw_rad: float  # about the vertical, relative to the demonstration's start
    tilt_rad: float  # how far the relative rotation is from a pure yaw (0 = flat)


class ObservationInputs(NamedTuple):
    joint_positions: np.ndarray  # (26,) rad, demonstration order
    joint_velocities: np.ndarray  # (26,) rad/s
    previous_targets: np.ndarray  # (26,) rad, the last commanded position targets
    reference_index: int
    cube_pose_base: torch.Tensor  # (7,) position + xyzw quaternion, base frame
    symmetry_index: int
    object_scale: Optional[float] = None  # the bar's size relative to the nominal one


class DeploymentRun:
    """A checkpoint, its configuration, and everything the contract needs from them."""

    def __init__(
        self,
        checkpoint: PathLike,
        config: Optional[PathLike] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("Checkpoint not found: {}".format(self.checkpoint_path))
        self.config_path = (
            Path(config).expanduser().resolve()
            if config is not None
            else self.checkpoint_path.parent / "config.json"
        )
        if not self.config_path.is_file():
            raise FileNotFoundError(
                "No config.json beside {} (pass --config)".format(self.checkpoint_path)
            )
        self.env_cfg, self.train_cfg = load_saved_configuration(self.config_path)
        self.device = torch.device(device)
        self.repo_root = ROOT_DIR

        env = self.env_cfg
        control = env.control
        if getattr(control, "action_parameterization", None) != "operational_space_arm":
            raise ValueError(
                "This contract implements the operational-space arm (the six arm "
                "actions are a palm twist); the checkpoint was trained with "
                "control.action_parameterization={!r}".format(
                    getattr(control, "action_parameterization", None)
                )
            )
        if int(env.env.num_actions) != ACTION_DIM:
            raise ValueError("The contract drives {} joints".format(ACTION_DIM))
        if bool(getattr(env.contact, "observe_fingertip_forces", False)):
            raise ValueError(
                "The checkpoint observes fingertip contact forces, which no sensor "
                "on the robot reports; this contract has no force block"
            )
        if int(env.env.num_observations) != OBSERVATION_DIM:
            raise ValueError(
                "Observation-contract mismatch: the checkpoint was trained on a {}D "
                "base observation, this contract builds {}D".format(
                    env.env.num_observations, OBSERVATION_DIM
                )
            )
        randomization = env.object_randomization
        self.observes_scale = observe_scale(randomization)
        self.observed_scale_override = observed_scale_override(randomization)
        self.object_scale_range = scale_range(randomization)
        self.observation_dim = OBSERVATION_DIM + object_scale_observation_dim(randomization)
        with self.config_path.open("r", encoding="utf-8") as stream:
            saved_dim = json.load(stream).get("observation_dim")
        if saved_dim is not None and int(saved_dim) != self.observation_dim:
            raise ValueError(
                "The run recorded a {}D observation, this contract builds {}D".format(
                    saved_dim, self.observation_dim
                )
            )

        self.dt = float(env.sim.dt) * int(control.decimation)
        self.hand_action_scale = float(control.scale_hand_joint_target)
        self.action_target_clip = float(control.clip_joint_target)
        self.arm_translation_speed = float(control.arm_translation_speed_m_per_s)
        self.arm_rotation_speed = float(control.arm_rotation_speed_rad_per_s)
        self.ik_damping = float(control.ik_damping)
        self.ik_max_joint_delta = float(control.ik_max_joint_delta_rad)
        self.action_filter_alpha = float(getattr(control, "action_filter_alpha", 1.0))
        if not 0.0 < self.action_filter_alpha <= 1.0:
            raise ValueError("control.action_filter_alpha must lie in (0, 1]")
        for name, value in (
            ("scale_hand_joint_target", self.hand_action_scale),
            ("clip_joint_target", self.action_target_clip),
            ("arm_translation_speed_m_per_s", self.arm_translation_speed),
            ("arm_rotation_speed_rad_per_s", self.arm_rotation_speed),
            ("ik_damping", self.ik_damping),
            ("ik_max_joint_delta_rad", self.ik_max_joint_delta),
        ):
            if not value > 0.0:
                raise ValueError("control.{} must be positive".format(name))

        self.urdf_path = self.resolve(env.asset.file)
        self.kinematics = PalmKinematics(self.urdf_path, device=self.device, dtype=torch.float32)
        self.joint_lower_limits = self.kinematics.lower_limits.to(torch.float32)
        self.joint_upper_limits = self.kinematics.upper_limits.to(torch.float32)
        self.arm_lower_limits = self.joint_lower_limits[:ARM_DOF]
        self.arm_upper_limits = self.joint_upper_limits[:ARM_DOF]
        velocity_limits = urdf_velocity_limits(self.urdf_path, JOINT_NAMES).to(self.device)
        if bool(getattr(control, "slew_targets_at_velocity_limit", True)):
            self.target_slew_per_step = velocity_limits * self.dt
        else:
            self.target_slew_per_step = torch.full_like(velocity_limits, float("inf"))

        default_arm = torch.as_tensor(
            env.init_state.default_arm_joint_angles, dtype=torch.float32, device=self.device
        )
        default_hand = torch.as_tensor(
            env.init_state.default_hand_joint_angles, dtype=torch.float32, device=self.device
        )
        if default_arm.shape != (ARM_DOF,) or default_hand.shape != (HAND_DOF,):
            raise ValueError("Default pose must contain 6 arm and 20 hand angles")
        self.default_positions = torch.cat((default_arm, default_hand))
        self.default_hand_positions = self.default_positions[ARM_DOF:]
        if torch.any(self.default_positions < self.joint_lower_limits) or torch.any(
            self.default_positions > self.joint_upper_limits
        ):
            raise ValueError("Default pose exceeds the URDF position limits")

        self.reference = JointDemonstration60Hz.load(
            self.resolve(env.motion.file),
            device=self.device,
            expected_hz=float(env.motion.frequency_hz),
        )
        self.bank = TransformBank.load(self.resolve(env.object_randomization.bank_path)).to(
            device=self.device, dtype=torch.float32
        )
        if self.bank.sample_count != self.reference.sample_count:
            raise ValueError(
                "The transform bank has {} frames but the demonstration has {}".format(
                    self.bank.sample_count, self.reference.sample_count
                )
            )
        self.last_index = self.reference.last_index
        self.motion_frequency_hz = float(self.reference.frequency_hz)
        self.robot_base_position = torch.tensor(
            env.init_state.pos, dtype=torch.float32, device=self.device
        )
        self.nearest_yaw_lever_arm_m = float(
            getattr(env.object_randomization, "nearest_yaw_lever_arm_m", 0.10)
        )
        self.object_half_extents = [0.5 * float(v) for v in env.object.size_m]
        # A uniform scale keeps the bar's proportions, hence its symmetry group.
        self.cuboid_symmetries = cuboid_rotation_symmetries(self.object_half_extents).to(
            device=self.device, dtype=torch.float32
        )
        demonstration_cube_base = cube_pose_to_base_frame(self.reference.cube_pose)
        # transform_points() yaws about the bar's frame-0 centre, then translates.
        self.placement_pivot = demonstration_cube_base[0, :3].clone()
        self.demonstration_start_orientation = normalize_canonical_quaternion(
            demonstration_cube_base[0, 3:7]
        )

        termination = env.termination
        self.hand_position_threshold_rad = float(
            getattr(termination, "hand_position_threshold_rad", 1.35)
        )
        self.palm_keypoint_threshold_m = float(
            getattr(termination, "palm_keypoint_threshold_m", 0.20)
        )

        self.policy, self.normalizer = self._load_policy()

    # -- files ---------------------------------------------------------------
    def resolve(self, configured: PathLike) -> Path:
        path = Path(configured)
        return path if path.is_absolute() else self.repo_root / path

    def _load_policy(self):
        try:
            loaded = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        except TypeError:
            loaded = torch.load(str(self.checkpoint_path), map_location=self.device)
        if not isinstance(loaded, dict) or "policy_dict" not in loaded:
            raise ValueError("{} is not an AnimRL checkpoint".format(self.checkpoint_path))
        first_layer = loaded["policy_dict"].get("policy_latent_net.0.weight")
        if first_layer is None or int(first_layer.shape[1]) != self.observation_dim:
            raise ValueError(
                "The network's first layer takes {} inputs, the contract builds {}".format(
                    None if first_layer is None else int(first_layer.shape[1]),
                    self.observation_dim,
                )
            )
        policy_cfg = self.train_cfg.policy
        policy = Policy(
            num_obs=self.observation_dim,
            num_actions=ACTION_DIM,
            hidden_dims=list(policy_cfg.actor_hidden_dims),
            activation=policy_cfg.activation,
            log_std_init=policy_cfg.log_std_init,
            max_action_std=policy_cfg.max_action_std,
            min_action_std=getattr(policy_cfg, "min_action_std", None),
            device=self.device,
        ).to(self.device)
        policy.load_state_dict(loaded["policy_dict"])
        policy.eval()
        if bool(self.train_cfg.runner.normalize_observation):
            normalizer = EmpiricalNormalization(shape=self.observation_dim, until=int(1.0e8)).to(
                self.device
            )
            normalizer.load_state_dict(loaded["actor_obs_normalizer"])
            infos = loaded.get("infos")
            if isinstance(infos, dict):
                normalizer.count = int(infos.get("actor_normalizer_count", 0))
        else:
            normalizer = torch.nn.Identity()
        normalizer.eval()
        self.checkpoint_infos = loaded.get("infos")
        return policy, normalizer

    # -- the network -----------------------------------------------------------
    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        """The deterministic (mean) action, as the evaluators use it."""
        observation = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), device=self.device
        ).reshape(1, -1)
        if observation.shape[1] != self.observation_dim:
            raise ValueError(
                "Observation has {} values, expected {}".format(
                    observation.shape[1], self.observation_dim
                )
            )
        actions = self.policy.act_inference(self.normalizer(observation))
        return actions[0].cpu().numpy().astype(np.float64)

    # -- the reference ---------------------------------------------------------
    def reference_sample(self, transform_index: int, reference_index: int) -> BankSample:
        """One frame of one bank clip, batch of one."""
        transform_index = int(transform_index)
        reference_index = int(reference_index)
        if not 0 <= transform_index < self.bank.transform_count:
            raise IndexError("Transform index {} outside the bank".format(transform_index))
        if not 0 <= reference_index <= self.last_index:
            raise IndexError("Reference index {} outside the demonstration".format(reference_index))
        return self.bank.sample(
            torch.tensor([transform_index], device=self.device),
            torch.tensor([reference_index], device=self.device),
        )

    def reference_q(self, transform_index: int, reference_index: int) -> np.ndarray:
        return self.reference_sample(transform_index, reference_index).q[0].cpu().numpy().astype(np.float64)

    def reference_cube_pose(self, transform_index: int, reference_index: int, object_scale: float = 1.0) -> np.ndarray:
        """The clip's bar pose (demonstration convention), lifted so a scaled bar rests on the table.

        ``_cube_reference_root_states`` raises the centre by ``half_height * (s - 1)``;
        the vertical axis is the same in both frame conventions, so the shift
        applies here directly.
        """
        pose = self.reference_sample(transform_index, reference_index).cube_pose[0].cpu().numpy().astype(np.float64)
        pose = pose.copy()
        pose[2] += self.reference_height_shift(object_scale)
        return pose

    def reference_height_shift(self, object_scale: float) -> float:
        scale = validate_object_scale(object_scale)
        return float(reference_height_shift(torch.tensor([scale]), self.object_half_extents[2]))

    def default_transform_index(self) -> int:
        """The bank entry nearest the demonstration's own placement."""
        translation = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        yaw = torch.zeros(1, dtype=torch.float32, device=self.device)
        return int(
            nearest_transform_indices(
                translation, yaw, self.bank.translation, self.bank.yaw_rad, self.nearest_yaw_lever_arm_m
            )[0]
        )

    def cube_pose_base(self, cube_pose_ur: np.ndarray) -> torch.Tensor:
        """A demonstration-convention cuboid pose, in the base frame."""
        pose = torch.as_tensor(np.asarray(cube_pose_ur, dtype=np.float32), device=self.device)
        if pose.shape != (7,):
            raise ValueError("A cuboid pose has 7 values (position, xyzw quaternion)")
        return cube_pose_to_base_frame(pose)

    def placement_from_cube_pose(self, cube_pose_ur: np.ndarray) -> Placement:
        """Where a cuboid stands, relative to where the demonstration's stood."""
        base = self.cube_pose_base(cube_pose_ur)
        orientation = canonicalize_cuboid_orientation(
            normalize_canonical_quaternion(base[3:7]),
            self.cuboid_symmetries,
            self.demonstration_start_orientation,
        )
        relative = quat_to_matrix(
            quat_multiply(orientation, quat_conjugate(self.demonstration_start_orientation))
        )
        yaw = math.atan2(float(relative[1, 0]), float(relative[0, 0]))
        tilt = math.acos(max(-1.0, min(1.0, float(relative[2, 2]))))
        return Placement(base[:3] - self.placement_pivot, yaw, tilt)

    def nearest_transform_index(self, placement: Placement) -> Tuple[int, float, float]:
        """The bank entry nearest a placement, with its planar and yaw residuals."""
        translation = placement.translation.reshape(1, 3).to(self.device)
        yaw = torch.tensor([placement.yaw_rad], dtype=torch.float32, device=self.device)
        index = int(
            nearest_transform_indices(
                translation, yaw, self.bank.translation, self.bank.yaw_rad, self.nearest_yaw_lever_arm_m
            )[0]
        )
        residual_xy = float(
            torch.linalg.vector_norm(translation[0, :2] - self.bank.translation[index, :2])
        )
        residual_yaw = wrap_angle(placement.yaw_rad - float(self.bank.yaw_rad[index]))
        return index, residual_xy, residual_yaw

    def choose_symmetry_index(self, cube_pose_base: torch.Tensor, reference_cube_pose_ur: torch.Tensor) -> int:
        """Pick the bar's symmetry label against the pose the episode starts at, once."""
        reference_base = cube_pose_to_base_frame(reference_cube_pose_ur.reshape(7))
        _, chosen = canonicalize_cuboid_orientation(
            normalize_canonical_quaternion(cube_pose_base[3:7]).reshape(1, 4),
            self.cuboid_symmetries,
            normalize_canonical_quaternion(reference_base[3:7]).reshape(1, 4),
            return_index=True,
        )
        return int(chosen[0])

    # -- the arithmetic the environment shares ---------------------------------
    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return (
            2.0 * (positions - self.joint_lower_limits) / (self.joint_upper_limits - self.joint_lower_limits)
            - 1.0
        ).clamp(-1.0, 1.0)

    def positions_to_hand_actions(self, hand_positions: torch.Tensor) -> torch.Tensor:
        return (hand_positions - self.default_hand_positions) / self.hand_action_scale

    def palm_pose(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Palm position and canonical xyzw orientation in the base frame."""
        q = torch.as_tensor(np.asarray(joint_positions, dtype=np.float32), device=self.device).reshape(1, -1)
        position, orientation = self.kinematics.palm_pose(q[:, :ARM_DOF])
        orientation = normalize_canonical_quaternion(orientation)
        return position[0].cpu().numpy().astype(np.float64), orientation[0].cpu().numpy().astype(np.float64)

    def next_reference_action(
        self, previous_arm_targets: np.ndarray, transform_index: int, reference_index: int
    ) -> np.ndarray:
        """The ideal 26-action command carrying the accumulated target onto the next reference."""
        next_index = min(int(reference_index) + 1, self.last_index)
        target = self.reference_sample(transform_index, next_index).q
        previous = torch.as_tensor(
            np.asarray(previous_arm_targets, dtype=np.float32), device=self.device
        ).reshape(1, ARM_DOF)
        current_pose = self.kinematics.palm_matrices(previous)
        target_pose = self.kinematics.palm_matrices(target[:, :ARM_DOF])
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
        hand_action = self.positions_to_hand_actions(target[:, ARM_DOF:])
        return torch.cat((arm_action, hand_action), dim=1)[0].cpu().numpy().astype(np.float64)


def build_observation(run: DeploymentRun, inputs: ObservationInputs) -> np.ndarray:
    """``MotionImitationEnv.compute_observations`` for one measured state."""
    device = run.device
    q = torch.as_tensor(np.asarray(inputs.joint_positions, dtype=np.float32), device=device).reshape(1, ACTION_DIM)
    dq = torch.as_tensor(np.asarray(inputs.joint_velocities, dtype=np.float32), device=device).reshape(1, ACTION_DIM)
    previous = torch.as_tensor(np.asarray(inputs.previous_targets, dtype=np.float32), device=device).reshape(1, ACTION_DIM)
    for name, values in (("joint_positions", q), ("joint_velocities", dq), ("previous_targets", previous)):
        if not torch.all(torch.isfinite(values)):
            raise ValueError("{} contains non-finite values".format(name))
    cube = inputs.cube_pose_base.to(device=device, dtype=torch.float32).reshape(1, 7)
    phase = torch.tensor([[float(inputs.reference_index) / float(run.last_index)]], device=device)

    palm_position, palm_orientation = run.kinematics.palm_pose(q[:, :ARM_DOF])
    palm_orientation = normalize_canonical_quaternion(palm_orientation)
    fingertips = run.kinematics.fingertip_positions(q)
    fingertips_palm = quat_rotate_inverse(
        palm_orientation.unsqueeze(1).expand(-1, fingertips.shape[1], -1),
        fingertips - palm_position.unsqueeze(1),
    ).reshape(1, -1)
    cube_orientation = apply_cuboid_symmetry(
        normalize_canonical_quaternion(cube[:, 3:7]),
        run.cuboid_symmetries,
        torch.tensor([int(inputs.symmetry_index)], device=device),
    )
    cube_center_palm = quat_rotate_inverse(palm_orientation, cube[:, :3] - palm_position)
    cube_orientation_palm = normalize_canonical_quaternion(
        quat_multiply(quat_conjugate(palm_orientation), cube_orientation)
    )
    parts = [
        run.normalize_positions(q),
        previous,
        dq,
        phase,
        palm_position,
        quat_to_rotation_6d(palm_orientation),
        fingertips_palm,
        quat_to_rotation_6d(cube_orientation_palm),
        cube_center_palm,
    ]
    if run.observes_scale:
        if inputs.object_scale is None:
            raise ValueError("This checkpoint observes the bar's scale; ObservationInputs.object_scale is required")
        scale = torch.tensor([validate_object_scale(inputs.object_scale)], dtype=torch.float32, device=device)
        parts.append(scale_observation(scale, run.observed_scale_override))
    observation = torch.cat(parts, dim=1)
    if observation.shape[1] != run.observation_dim:
        raise RuntimeError("Built a {}D observation".format(observation.shape[1]))
    return observation[0].cpu().numpy().astype(np.float64)


class ActionPipeline:
    """The state between the network's output and the drives.

    ``MotionImitationEnv._pre_physics_step`` and ``_apply_action`` for one
    robot: the first-order action filter, the operational-space arm targets
    accumulating on the previously *commanded* target, the residual hand
    targets, and the applied target slewed at the joint velocity limit. The
    commanded targets are what the observation reports as ``previous_targets``;
    the applied ones are what the drives receive.
    """

    def __init__(self, run: DeploymentRun) -> None:
        self.run = run
        self.filtered_actions: Optional[torch.Tensor] = None
        self.position_targets: Optional[torch.Tensor] = None
        self.applied_targets: Optional[torch.Tensor] = None
        self.last_info: Dict[str, object] = {}

    def reset(self, reference_q: np.ndarray) -> None:
        """Start on a reference frame: targets on it, the filter on its own action."""
        q = torch.as_tensor(np.asarray(reference_q, dtype=np.float32), device=self.run.device).reshape(ACTION_DIM)
        if not torch.all(torch.isfinite(q)):
            raise ValueError("Reference pose contains non-finite values")
        self.position_targets = q.clone()
        self.applied_targets = q.clone()
        reset_action = torch.zeros(ACTION_DIM, dtype=torch.float32, device=self.run.device)
        reset_action[ARM_DOF:] = self.run.positions_to_hand_actions(q[ARM_DOF:])
        self.filtered_actions = reset_action
        self.last_info = {}

    def _require_reset(self) -> None:
        if self.position_targets is None or self.filtered_actions is None or self.applied_targets is None:
            raise RuntimeError("reset() must be called before the pipeline is used")

    @property
    def previous_targets(self) -> np.ndarray:
        self._require_reset()
        return self.position_targets.cpu().numpy().astype(np.float64)

    @property
    def previous_arm_targets(self) -> np.ndarray:
        return self.previous_targets[:ARM_DOF]

    def command(
        self,
        raw_actions: np.ndarray,
        measured_arm_q: np.ndarray,
        arm_scale: float = 1.0,
        hand_scale: float = 1.0,
    ) -> np.ndarray:
        """Filter the raw action and turn it into the 26 commanded position targets.

        ``arm_scale`` and ``hand_scale`` are commissioning knobs (1.0 reproduces
        training): they scale the filtered arm twist request and the hand
        residual, after the filter so that the filter always sees the raw
        network output.
        """
        self._require_reset()
        run = self.run
        raw = torch.as_tensor(np.asarray(raw_actions, dtype=np.float32), device=run.device).reshape(ACTION_DIM)
        if not torch.all(torch.isfinite(raw)):
            raise ValueError("Policy produced a non-finite action")
        if run.action_filter_alpha < 1.0:
            self.filtered_actions = torch.lerp(self.filtered_actions, raw, run.action_filter_alpha)
        else:
            self.filtered_actions = raw.clone()
        filtered = self.filtered_actions
        arm_targets = self._arm_targets(filtered[:ARM_DOF] * float(arm_scale), measured_arm_q)
        hand_residual = (filtered[ARM_DOF:] * float(hand_scale) * run.hand_action_scale).clamp(
            -run.action_target_clip, run.action_target_clip
        )
        hand_targets = run.default_hand_positions + hand_residual
        self.position_targets = torch.cat((arm_targets, hand_targets))
        return self.previous_targets

    def _arm_targets(self, arm_actions: torch.Tensor, measured_arm_q: np.ndarray) -> torch.Tensor:
        run = self.run
        arm_q = torch.as_tensor(np.asarray(measured_arm_q, dtype=np.float32), device=run.device).reshape(1, ARM_DOF)
        if not torch.all(torch.isfinite(arm_q)):
            raise ValueError("Measured arm configuration contains non-finite values")
        max_translation = run.arm_translation_speed * run.dt
        max_rotation = run.arm_rotation_speed * run.dt
        actions = arm_actions.reshape(1, ARM_DOF)
        desired_twist = torch.cat(
            (
                saturate_direction_preserving(actions[:, 0:3] * max_translation, max_translation),
                saturate_direction_preserving(actions[:, 3:6] * max_rotation, max_rotation),
            ),
            dim=1,
        )
        jacobian = run.kinematics.jacobian(arm_q)
        q_delta = damped_least_squares_step(jacobian, desired_twist, run.ik_damping)[0]
        unclipped = q_delta
        q_delta = q_delta.clamp(-run.ik_max_joint_delta, run.ik_max_joint_delta)
        previous = self.position_targets[:ARM_DOF]
        targets = (previous + q_delta).clamp(run.arm_lower_limits, run.arm_upper_limits)
        applied_delta = targets - previous
        achieved_twist = jacobian[0] @ applied_delta
        self.last_info = {
            "requested_twist": desired_twist[0].cpu().numpy().astype(np.float64),
            "achieved_twist": achieved_twist.cpu().numpy().astype(np.float64),
            "ik_residual_norm": float(torch.linalg.vector_norm(desired_twist[0] - achieved_twist)),
            "arm_joint_delta_norm": float(torch.linalg.vector_norm(applied_delta)),
            "arm_joint_delta_clipped": bool(torch.any(unclipped.abs() > run.ik_max_joint_delta)),
        }
        return targets

    def apply(self) -> np.ndarray:
        """Advance the applied target toward the commanded one at the velocity limit."""
        self._require_reset()
        delta = (self.position_targets - self.applied_targets).clamp(
            -self.run.target_slew_per_step, self.run.target_slew_per_step
        )
        self.applied_targets = self.applied_targets + delta
        return self.applied_targets.cpu().numpy().astype(np.float64)
