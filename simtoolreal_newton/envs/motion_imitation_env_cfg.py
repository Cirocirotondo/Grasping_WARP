"""Isaac Lab configuration of the UR5e + DG5F motion-imitation scene.

The AnimRL-style Python configuration (:class:`SimToolRealCfg`) remains the
single source of truth for the task. This module translates the parts of it
that describe the *simulation* -- robot asset, drives, cuboid, table, solver
settings -- into the ``configclass`` objects Isaac Lab consumes, and leaves the
task logic (rewards, RSI, terminations) to the environment.
"""

from __future__ import annotations

import math
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import MujocoCollisionCfg, MujocoRigidBodyPropertiesCfg

from simtoolreal_newton import ROOT_DIR
from simtoolreal_newton.cfg import SimToolRealCfg
from simtoolreal_newton.envs.contact import fingertip_force_observation_dim
from simtoolreal_newton.envs.controller import (
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    JOINT_NAMES,
    pd_gain_tables,
)
from simtoolreal_newton.envs.domain_randomization import DomainRandomization

SOURCE_URDF = ROOT_DIR / "assets" / "urdf" / "ur5e_delto_description" / "ur5e_right_dg5f_mount_60deg.urdf"
ROBOT_URDF = ROOT_DIR / "assets" / "ur5e_right_dg5f.urdf"
USD_DIR = ROOT_DIR / "assets" / "usd"
ROBOT_USD = USD_DIR / "ur5e_right_dg5f" / "ur5e_right_dg5f.usda"

@configclass
class SimToolRealMJWarpSolverCfg(MJWarpSolverCfg):
    """MJWarp solver settings plus the multi-contact flag Isaac Lab does not expose.

    Fields are forwarded to :class:`newton.solvers.SolverMuJoCo` by name.
    """

    enable_multiccd: bool = False


ROBOT_PRIM = "{ENV_REGEX_NS}/Robot"
CUBE_PRIM = "{ENV_REGEX_NS}/Cube"
TABLE_PRIM = "{ENV_REGEX_NS}/Table"
PROTO_ENV = "/World/envs/env_0"


def make_physics_cfg(animrl_cfg):
    """The Isaac Lab physics-manager configuration for ``animrl_cfg.sim.physics``."""
    sim_cfg = animrl_cfg.sim
    name = str(getattr(sim_cfg, "physics", "newton_mjwarp"))
    if name == "newton_mjwarp":
        mj = sim_cfg.mjwarp
        return NewtonCfg(
            solver_cfg=SimToolRealMJWarpSolverCfg(
                enable_multiccd=bool(getattr(mj, "enable_multiccd", True)),
                njmax=int(mj.njmax),
                nconmax=int(mj.nconmax),
                iterations=int(mj.iterations),
                ls_iterations=int(mj.ls_iterations),
                solver=str(mj.solver),
                integrator=str(mj.integrator),
                cone=str(mj.cone),
                impratio=float(mj.impratio),
                ccd_iterations=int(mj.ccd_iterations),
                tolerance=float(mj.tolerance),
                use_mujoco_contacts=bool(getattr(mj, "use_mujoco_contacts", True)),
                use_mujoco_cpu=bool(getattr(mj, "use_mujoco_cpu", False)),
            ),
            num_substeps=int(sim_cfg.substeps),
            use_cuda_graph=bool(getattr(sim_cfg, "use_cuda_graph", True)),
            debug_mode=False,
            # Newton semantics: forces act below ``margin`` of separation, and
            # contacts are detected within a further ``gap``. This mirrors
            # PhysX's rest_offset / contact_offset pair of the original setup.
            default_shape_cfg=NewtonShapeCfg(
                margin=float(getattr(mj, "contact_margin", 0.0)),
                gap=float(getattr(mj, "contact_gap", 0.005)),
            ),
        )
    if name in ("physx", "isaacsim_physx", "ovphysx"):
        px = sim_cfg.physx
        if name == "ovphysx":
            from isaaclab_ov.physics import OvPhysxCfg

            return OvPhysxCfg()
        from isaaclab_physx.physics import PhysxCfg

        return PhysxCfg(
            solver_type=int(px.solver_type),
            bounce_threshold_velocity=float(px.bounce_threshold_velocity),
            gpu_max_rigid_contact_count=int(px.max_gpu_contact_pairs),
        )
    raise ValueError(
        "Unknown physics backend {!r}; expected newton_mjwarp, physx, ovphysx or isaacsim_physx".format(name)
    )


def _collision_props(animrl_cfg, nested: bool = False):
    """Collision settings for one spawned asset.

    On the PhysX backends the legacy contact/rest offsets are authored. On
    MJWarp the per-shape margin and gap come from :class:`NewtonShapeCfg`
    (see :func:`make_physics_cfg`); the collision API is still applied to
    every collider, plus the optional MuJoCo contact softness / friction
    dimensionality fragment. ``nested`` targets every collider below the
    spawned prim (USD robot assets) instead of the geometry prim itself
    (primitive shapes).
    """
    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg

    physx = animrl_cfg.sim.physx
    physics_name = str(getattr(animrl_cfg.sim, "physics", "newton_mjwarp"))
    if physics_name != "newton_mjwarp":
        return sim_utils.CollisionPropertiesCfg(
            contact_offset=float(getattr(physx, "contact_offset", 0.002)),
            rest_offset=float(getattr(physx, "rest_offset", 0.0)),
        )
    mj = animrl_cfg.sim.mjwarp
    solref = getattr(mj, "contact_solref", None)
    condim = getattr(mj, "contact_condim", None)
    fragments = [UsdPhysicsCollisionCfg(collision_enabled=True)]
    if solref is not None or (condim is not None and int(condim) != 3):
        fragments.append(
            MujocoCollisionCfg(
                solref=tuple(float(v) for v in solref) if solref is not None else None,
                condim=int(condim) if condim is not None else None,
            )
        )
    return {"/.*": fragments} if nested else fragments


def _robot_rigid_props(physics_name: str):
    """Gravity-free robot links, expressed the way each solver understands it.

    The real arm runs gravity compensation, and the Isaac Gym configuration
    disabled gravity on the robot for the same reason. MuJoCo says this with a
    per-body ``gravcomp`` of one; PhysX with ``disableGravity``.
    """
    if physics_name == "newton_mjwarp":
        return MujocoRigidBodyPropertiesCfg(disable_gravity=True, gravcomp=1.0)
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

    return PhysxRigidBodyPropertiesCfg(
        disable_gravity=True,
        linear_damping=0.01,
        angular_damping=0.01,
        max_depenetration_velocity=2.0,
    )


def _make_actuators(stiffness, damping):
    """Implicit PD drives for the arm and the hand (solver-integrated).

    Explicit torque actuators (Isaac Lab's ``DCMotorCfg`` torque-speed curve,
    which would have reproduced PhysX's joint velocity clamp) are unstable at
    the 60 Hz control rate with the hand's gains and 1e-5 kg m^2 phalanges, so
    the drives stay implicit and the environment slews the applied targets at
    the URDF velocity limit instead (see ``MotionImitationEnv._apply_action``).
    """
    return {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(ARM_JOINT_NAMES),
            stiffness={name: stiffness[name] for name in ARM_JOINT_NAMES},
            damping={name: damping[name] for name in ARM_JOINT_NAMES},
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=list(HAND_JOINT_NAMES),
            stiffness={name: stiffness[name] for name in HAND_JOINT_NAMES},
            damping={name: damping[name] for name in HAND_JOINT_NAMES},
        ),
    }


def make_robot_cfg(animrl_cfg, contact_enabled: bool) -> ArticulationCfg:
    control = animrl_cfg.control
    stiffness, damping = pd_gain_tables(
        arm_stiffness_scale=float(getattr(control, "arm_stiffness_scale", 1.0)),
        arm_damping_scale=float(getattr(control, "arm_damping_scale", 1.0)),
        hand_stiffness_scale=float(getattr(control, "hand_stiffness_scale", 1.0)),
        hand_damping_scale=float(getattr(control, "hand_damping_scale", 1.0)),
    )
    physics_name = str(getattr(animrl_cfg.sim, "physics", "newton_mjwarp"))
    asset = animrl_cfg.asset
    default_joint_pos = dict(zip(ARM_JOINT_NAMES, [float(v) for v in animrl_cfg.init_state.default_arm_joint_angles]))
    default_joint_pos.update(
        zip(HAND_JOINT_NAMES, [float(v) for v in animrl_cfg.init_state.default_hand_joint_angles])
    )
    if not ROBOT_USD.is_file():
        raise FileNotFoundError(
            "The robot USD {} does not exist. Convert the URDF once with\n"
            "    deps/IsaacLab/.venv/bin/python scripts/convert_urdf.py".format(ROBOT_USD)
        )
    spawn = sim_utils.UsdFileCfg(
        spawn_path=PROTO_ENV + "/Robot",
        usd_path=str(ROBOT_USD),
        # The converter instances the link geometry; the physics material and
        # collision offsets below are authored on those descendants.
        make_uninstanceable=True,
        rigid_props=_robot_rigid_props(physics_name),
        collision_props=_collision_props(animrl_cfg, nested=True),
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=float(asset.friction),
            dynamic_friction=float(asset.friction),
            restitution=float(asset.restitution),
        ),
        activate_contact_sensors=bool(contact_enabled),
    )
    return ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(v) for v in animrl_cfg.init_state.pos),
            rot=tuple(float(v) for v in animrl_cfg.init_state.rot),
            joint_pos=default_joint_pos,
            joint_vel={".*": 0.0},
        ),
        actuators=_make_actuators(stiffness, damping),
        # Public joint order == demonstration order, whatever the solver's
        # native traversal is. Bodies stay in backend order and are resolved
        # by name.
        joint_ordering=list(JOINT_NAMES),
        soft_joint_pos_limit_factor=1.0,
    )


def cube_pose_ur_base_to_world(pose_xyzw, robot_base_pos):
    """The demonstration's cube pose (UR base frame) in the world/env frame.

    Matches the Isaac Gym environment exactly: position mirrored in x and y,
    orientation left-multiplied by a yaw of pi.
    """
    x, y, z, qx, qy, qz, qw = [float(v) for v in pose_xyzw]
    bx, by, bz = [float(v) for v in robot_base_pos]
    pos = (bx - x, by - y, bz + z)
    quat = (-qy, qx, qw, -qz)
    norm = math.sqrt(sum(v * v for v in quat))
    return pos, tuple(v / norm for v in quat)


def make_cube_cfg(animrl_cfg, initial_pose_xyzw) -> RigidObjectCfg:
    obj = animrl_cfg.object
    pos, quat = cube_pose_ur_base_to_world(initial_pose_xyzw, animrl_cfg.init_state.pos)
    return RigidObjectCfg(
        prim_path=CUBE_PRIM,
        spawn=sim_utils.CuboidCfg(
            spawn_path=PROTO_ENV + "/Cube",
            size=tuple(float(v) for v in obj.size_m),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(obj.mass_kg)),
            collision_props=_collision_props(animrl_cfg),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=float(obj.friction),
                dynamic_friction=float(obj.friction),
                restitution=float(obj.restitution),
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(v) for v in obj.color)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
    )


def make_table_cfg(animrl_cfg) -> sim_utils.CuboidCfg:
    table = animrl_cfg.table
    return sim_utils.CuboidCfg(
        size=tuple(float(v) for v in table.size_m),
        collision_props=_collision_props(animrl_cfg),
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=float(table.friction),
            dynamic_friction=float(table.friction),
            restitution=float(table.restitution),
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(v) for v in table.color)),
    )


def table_position(animrl_cfg):
    table = animrl_cfg.table
    return (
        0.0,
        0.0,
        float(animrl_cfg.init_state.pos[2])
        - float(table.surface_below_robot_base_m)
        - float(table.size_m[2]) / 2.0,
    )


def make_contact_sensor_cfg() -> ContactSensorCfg:
    # The five distal phalanges (the fixed tips are merged into them). The
    # body prims are nested under their parents, hence the leading ``.*``.
    return ContactSensorCfg(
        prim_path=ROBOT_PRIM + "/.*rl_dg_[1-5]_4",
        update_period=0.0,
        history_length=0,
    )


def observation_dims(animrl_cfg):
    """``(policy_dim, critic_dim_or_0)`` for the configured features."""
    num_obs = int(animrl_cfg.env.num_observations) + fingertip_force_observation_dim(animrl_cfg.contact)
    critic_force = (
        3 * len(animrl_cfg.contact.fingertip_names)
        if bool(getattr(animrl_cfg.contact, "critic_observes_fingertip_forces", False))
        else 0
    )
    randomization = DomainRandomization(
        getattr(animrl_cfg, "domain_randomization", object()),
        int(animrl_cfg.env.num_envs),
        seed=int(getattr(animrl_cfg, "seed", 0) or 0),
    )
    critic_parameter = randomization.privileged_dim
    critic = num_obs + critic_force + critic_parameter if (critic_force or critic_parameter) else 0
    return num_obs, critic


@configclass
class MotionImitationEnvCfg(DirectRLEnvCfg):
    """Direct-workflow configuration; see :func:`build_env_cfg`."""

    decimation: int = 1
    episode_length_s: float = 6.0
    action_space: int = 26
    observation_space: int = 112
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=2),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=2.0, replicate_physics=True)

    # Filled by :func:`build_env_cfg` (the environment calls it when they are
    # None, so the class can be instantiated without the converted USD).
    robot: ArticulationCfg | None = None
    cube: RigidObjectCfg | None = None
    table: sim_utils.CuboidCfg | None = None
    table_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_sensor: ContactSensorCfg | None = None
    spawn_ground_plane: bool = False
    # Diagnostics only: report which robot bodies push on the cuboid.
    debug_cube_contacts: bool = False
    # Optional kit-less camera (Newton Warp raster renderer) looking at env 0,
    # for frame dumps and videos. ``None`` disables it.
    camera: Any = None


def make_camera_cfg(animrl_cfg, width: int = 640, height: int = 480):
    """A Warp-rendered camera framed like the Isaac Gym training camera."""
    from isaaclab.sensors import CameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    from simtoolreal_newton.envs.rotations import matrix_to_quat

    viewer = animrl_cfg.viewer
    eye = torch.tensor([float(v) for v in viewer.camera_position], dtype=torch.float64)
    target = torch.tensor([float(v) for v in viewer.camera_lookat], dtype=torch.float64)
    # World-convention camera frame: +x forward (view direction), +z up.
    forward = target - eye
    forward = forward / forward.norm()
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    left = torch.cross(up, forward, dim=0)
    left = left / left.norm()
    up_cam = torch.cross(forward, left, dim=0)
    rot = torch.stack((forward, left, up_cam), dim=1)
    quat = matrix_to_quat(rot.unsqueeze(0))[0].tolist()
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=tuple(eye.tolist()), rot=tuple(quat), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            spawn_path=PROTO_ENV + "/Camera",
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        width=int(width),
        height=int(height),
        renderer_cfg=NewtonWarpRendererCfg(),
        update_period=0.0,
    )
    # The AnimRL configuration object driving the task. Set by
    # :func:`build_env_cfg`; when None the environment builds the default
    # :class:`SimToolRealCfg` and applies ``animrl_overrides`` (``--set``
    # style ``path=value`` pairs) so the Hydra/CLI entry points work too.
    animrl_cfg: Any = None
    animrl_overrides: dict[str, Any] = {}


def build_env_cfg(animrl_cfg, num_envs: int | None = None, device: str | None = None) -> MotionImitationEnvCfg:
    """Translate an AnimRL-style :class:`SimToolRealCfg` into the Isaac Lab cfg."""
    from simtoolreal_newton.envs.demonstration import JointDemonstration60Hz

    if num_envs is not None:
        animrl_cfg.env.num_envs = int(num_envs)
    if device is not None:
        animrl_cfg.sim.device = str(device)
    contact_enabled = bool(animrl_cfg.contact.enabled)
    reference = JointDemonstration60Hz.load(
        ROOT_DIR / animrl_cfg.motion.file, device="cpu", expected_hz=float(animrl_cfg.motion.frequency_hz)
    )
    num_obs, num_critic = observation_dims(animrl_cfg)
    cfg = MotionImitationEnvCfg()
    cfg.seed = int(animrl_cfg.seed)
    cfg.decimation = int(animrl_cfg.control.decimation)
    cfg.episode_length_s = float(animrl_cfg.env.episode_length) * float(animrl_cfg.sim.dt) * cfg.decimation
    cfg.action_space = int(animrl_cfg.env.num_actions)
    cfg.observation_space = num_obs
    cfg.state_space = num_critic
    cfg.sim = SimulationCfg(
        dt=float(animrl_cfg.sim.dt),
        render_interval=cfg.decimation,
        gravity=tuple(float(v) for v in animrl_cfg.sim.gravity),
        device=str(getattr(animrl_cfg.sim, "device", "cuda:0")),
        physics=make_physics_cfg(animrl_cfg),
    )
    cfg.scene = InteractiveSceneCfg(
        num_envs=int(animrl_cfg.env.num_envs),
        env_spacing=float(animrl_cfg.env.env_spacing),
        replicate_physics=True,
    )
    cfg.robot = make_robot_cfg(animrl_cfg, contact_enabled)
    cfg.cube = make_cube_cfg(animrl_cfg, reference.cube_pose[0].tolist())
    cfg.table = make_table_cfg(animrl_cfg)
    cfg.table_pos = table_position(animrl_cfg)
    cfg.contact_sensor = make_contact_sensor_cfg() if contact_enabled else None
    if bool(getattr(animrl_cfg.viewer, "training_camera_enabled", False)):
        cfg.camera = make_camera_cfg(
            animrl_cfg,
            width=int(getattr(animrl_cfg.viewer, "training_camera_width", 640)),
            height=int(getattr(animrl_cfg.viewer, "training_camera_height", 480)),
        )
    cfg.animrl_cfg = animrl_cfg
    return cfg
