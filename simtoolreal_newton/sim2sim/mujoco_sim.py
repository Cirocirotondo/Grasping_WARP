"""Native MuJoCo scene matching the Newton training environment.

The UR5e + right DG5F from the training URDF, the table, the (scaled) bar, the
implicit PD drives with the training gains, the training contact model
(``sim.mjwarp``: solref/solimp, elliptic cones, impratio, condim, margin/gap,
substeps) and the training collision graph (robot-table off, arm-bar off,
hand-bar on, bar-table on, plus the selective finger self-collision of
``asset.self_collision``). What MuJoCo-Warp and the Newton collision pipeline
do differently from native MuJoCo is exactly what this backend measures.
"""

from __future__ import annotations

import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np

from simtoolreal_newton import ROOT_DIR
from simtoolreal_newton.envs.controller import pd_gain_arrays
from simtoolreal_newton.envs.object_scale import inertia_factor, mass_factor
from simtoolreal_newton.envs.self_collision import filtered_body_pairs

from .constants import (
    ACTION_DIM,
    CUBE_BIT,
    FINGERTIP_BODY_NAMES,
    FINGERTIP_LINK_NAMES,
    HAND_BIT,
    HAND_BODY_NAMES,
    JOINT_NAMES,
    MUJOCO_BODY_GROUPS,
    ROBOT_BASE_BODY_NAME,
    ROBOT_URDF,
    TABLE_BIT,
    WRIST_BODY_NAME,
)


def _xyzw(quaternion_wxyz) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    q = q / np.linalg.norm(q)
    return -q if q[3] < 0.0 else q


def _wxyz(quaternion_xyzw) -> np.ndarray:
    return np.asarray(quaternion_xyzw, dtype=np.float64)[[3, 0, 1, 2]]


@dataclass
class MujocoSceneConfig:
    """Everything the scene takes from the run's ``env_cfg`` plus the runner's choices."""

    robot_urdf_path: Path
    robot_position_world: np.ndarray
    robot_orientation_world_xyzw: np.ndarray
    object_size_m: np.ndarray
    object_mass_kg: float
    object_friction: float
    object_scale: float
    scale_mass_with_volume: bool
    table_size_m: np.ndarray
    table_surface_below_robot_base_m: float
    table_friction: float
    robot_friction: float
    fingertip_friction: float
    fingertip_torsional_friction: float
    stiffness: np.ndarray
    damping: np.ndarray
    sim_dt: float
    substeps: int
    contact_solref: Optional[tuple]
    contact_solimp: Optional[tuple]
    contact_condim: int
    contact_margin: float
    contact_gap: float
    cone: str
    impratio: float
    integrator: str
    iterations: int
    ls_iterations: int
    tolerance: float
    object_max_linear_velocity: float
    object_max_angular_velocity: float
    gravity: np.ndarray
    asset_cfg: object
    self_collision: bool
    reference_ghost_offset_world: np.ndarray = field(
        default_factory=lambda: np.asarray((0.8, 0.0, 0.0))
    )
    reference_ghost_color: np.ndarray = field(
        default_factory=lambda: np.asarray((0.15, 0.85, 0.25))
    )
    enable_viewer: bool = False
    enable_reference_ghost: bool = False

    @classmethod
    def from_env_cfg(
        cls,
        env_cfg,
        *,
        object_scale: float = 1.0,
        enable_viewer: bool = False,
        enable_reference_ghost: bool = False,
        repo_root: Path = ROOT_DIR,
    ) -> "MujocoSceneConfig":
        asset, obj, table = env_cfg.asset, env_cfg.object, env_cfg.table
        control, sim = env_cfg.control, env_cfg.sim
        mj = getattr(sim, "mjwarp", None)
        stiffness, damping = pd_gain_arrays(
            arm_stiffness_scale=float(getattr(control, "arm_stiffness_scale", 1.0)),
            arm_damping_scale=float(getattr(control, "arm_damping_scale", 1.0)),
            hand_stiffness_scale=float(getattr(control, "hand_stiffness_scale", 1.0)),
            hand_damping_scale=float(getattr(control, "hand_damping_scale", 1.0)),
        )
        substeps = int(getattr(sim, "substeps", 1))
        solref = getattr(mj, "contact_solref", None)
        solimp = getattr(mj, "contact_solimp", None)
        randomization = env_cfg.object_randomization
        viewer_cfg = getattr(env_cfg, "viewer", None)
        config = cls(
            robot_urdf_path=Path(ROBOT_URDF),
            robot_position_world=np.asarray(env_cfg.init_state.pos, dtype=np.float64),
            robot_orientation_world_xyzw=np.asarray(env_cfg.init_state.rot, dtype=np.float64),
            object_size_m=np.asarray(obj.size_m, dtype=np.float64),
            object_mass_kg=float(obj.mass_kg),
            object_friction=float(obj.friction),
            object_scale=float(object_scale),
            scale_mass_with_volume=bool(getattr(randomization, "scale_mass_with_volume", True)),
            table_size_m=np.asarray(table.size_m, dtype=np.float64),
            table_surface_below_robot_base_m=float(table.surface_below_robot_base_m),
            table_friction=float(table.friction),
            robot_friction=float(asset.friction),
            fingertip_friction=float(asset.fingertip_friction),
            fingertip_torsional_friction=float(getattr(asset, "fingertip_torsional_friction", 0.0) or 0.0),
            stiffness=np.asarray(stiffness, dtype=np.float64),
            damping=np.asarray(damping, dtype=np.float64),
            sim_dt=float(sim.dt) / substeps,
            substeps=substeps,
            contact_solref=tuple(float(v) for v in solref) if solref is not None else None,
            contact_solimp=tuple(float(v) for v in solimp) if solimp is not None else None,
            contact_condim=int(getattr(mj, "contact_condim", 3) or 3),
            contact_margin=float(getattr(mj, "contact_margin", 0.0) or 0.0),
            contact_gap=float(getattr(mj, "contact_gap", 0.0) or 0.0),
            cone=str(getattr(mj, "cone", "pyramidal")),
            impratio=float(getattr(mj, "impratio", 1.0)),
            integrator=str(getattr(mj, "integrator", "implicitfast")),
            iterations=int(getattr(mj, "iterations", 100)),
            ls_iterations=int(getattr(mj, "ls_iterations", 50)),
            tolerance=float(getattr(mj, "tolerance", 1.0e-8)),
            object_max_linear_velocity=float(getattr(mj, "object_max_linear_velocity", 0.0) or 0.0),
            object_max_angular_velocity=float(getattr(mj, "object_max_angular_velocity", 0.0) or 0.0),
            gravity=np.asarray(sim.gravity, dtype=np.float64),
            asset_cfg=asset,
            self_collision=bool(getattr(asset, "self_collision", False)),
            enable_viewer=bool(enable_viewer),
            enable_reference_ghost=bool(enable_reference_ghost),
        )
        if viewer_cfg is not None:
            config.reference_ghost_offset_world = np.asarray(
                getattr(viewer_cfg, "reference_ghost_offset", (0.8, 0.0, 0.0)), dtype=np.float64
            )
            config.reference_ghost_color = np.asarray(
                getattr(viewer_cfg, "reference_ghost_color", (0.15, 0.85, 0.25)), dtype=np.float64
            )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.robot_urdf_path.is_file():
            raise FileNotFoundError("Robot URDF not found: {}".format(self.robot_urdf_path))
        if self.object_scale <= 0.0 or not math.isfinite(self.object_scale):
            raise ValueError("The object scale must be a positive number")
        if np.any(self.object_size_m <= 0.0) or np.any(self.table_size_m <= 0.0):
            raise ValueError("Object and table dimensions must be positive")
        if self.object_mass_kg <= 0.0 or self.sim_dt <= 0.0 or self.substeps <= 0:
            raise ValueError("Object mass, timestep and substeps must be positive")
        if self.stiffness.shape != (ACTION_DIM,) or self.damping.shape != (ACTION_DIM,):
            raise ValueError("One stiffness and one damping per joint are required")
        if np.any(self.stiffness <= 0.0) or np.any(self.damping < 0.0):
            raise ValueError("PD gains must be positive")
        if self.contact_condim not in (1, 3, 4, 6):
            raise ValueError("contact_condim must be 1, 3, 4 or 6")

    @property
    def scaled_object_size_m(self) -> np.ndarray:
        return self.object_size_m * self.object_scale

    @property
    def scaled_object_mass_kg(self) -> float:
        import torch

        factor = mass_factor(torch.tensor([self.object_scale]), self.scale_mass_with_volume)
        return float(self.object_mass_kg * float(factor[0]))

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.substeps


class MujocoSim:
    """One MuJoCo world with the training robot, table and bar."""

    def __init__(self, config: MujocoSceneConfig) -> None:
        self.config = config
        self._urdf_limits = self._read_urdf_joint_limits()
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="newton_sim2sim_")
        self.viewer = None
        self._init_scene()

    # -- assets ----------------------------------------------------------------
    def _read_urdf_joint_limits(self) -> Dict[str, tuple]:
        import xml.etree.ElementTree as ET

        root = ET.parse(str(self.config.robot_urdf_path)).getroot()
        limits = {}
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            name = joint.get("name")
            if name in JOINT_NAMES and limit is not None:
                limits[name] = (
                    float(limit.get("lower")),
                    float(limit.get("upper")),
                    float(limit.get("effort")),
                    float(limit.get("velocity")),
                )
        missing = sorted(set(JOINT_NAMES) - set(limits))
        if missing:
            raise ValueError("URDF is missing joint limits for {}".format(missing))
        return limits

    def _make_mujoco_compatible_urdf(self) -> Path:
        text = self.config.robot_urdf_path.read_text(encoding="utf-8")
        if "<mujoco>" not in text:
            # Fixed links stay bodies of their own so the palm and the tips
            # can be read back; visual meshes (.dae) are dropped, MuJoCo has
            # no decoder for them and only the collision hulls matter here.
            text = re.sub(
                r'(<robot\s+name="[^"]+">)',
                r'\1\n  <mujoco><compiler strippath="false" fusestatic="false" discardvisual="true"/></mujoco>',
                text,
                count=1,
            )
        assets = self.config.robot_urdf_path.parent
        if assets.name != "assets":
            assets = ROOT_DIR / "assets"

        def absolute_mesh_path(match: re.Match) -> str:
            filename = match.group(1)
            path = Path(filename)
            if not path.is_absolute():
                path = assets / filename
            return 'filename="{}"'.format(path.resolve())

        text = re.sub(r'filename="([^"]+)"', absolute_mesh_path, text)
        counts = {"visual": 0, "collision": 0}

        def name_geometry(match: re.Match) -> str:
            kind = match.group(1)
            counts[kind] += 1
            return '<{} name="{}_{}">'.format(kind, kind, counts[kind] - 1)

        text = re.sub(r"<(visual|collision)>", name_geometry, text)
        output = Path(self._tmp_dir.name) / "ur5e_right_dg5f_mujoco.urdf"
        output.write_text(text, encoding="utf-8")
        return output

    # -- scene -------------------------------------------------------------------
    def _init_scene(self) -> None:
        robot_path = self._make_mujoco_compatible_urdf()
        spec = mujoco.MjSpec.from_file(str(robot_path))
        if self.config.enable_reference_ghost:
            ghost = mujoco.MjSpec.from_file(str(robot_path))
            frame = spec.worldbody.add_frame()
            frame.name = "reference_ghost_mount"
            frame.attach_body(ghost.worldbody.first_body(), prefix="ghost_")
        self._add_world(spec)
        self._add_position_actuators(spec)
        self._add_self_collision_excludes(spec)
        # Gravity compensation must be authored before compiling: MuJoCo counts
        # the compensated bodies at compile time (``ngravcomp``) and skips the
        # pass entirely when the count is zero, whatever ``body_gravcomp``
        # says afterwards. The training robot runs gravity free (Newton
        # authors ``gravcomp = 1`` on every link).
        for body in spec.bodies:
            if body.name not in ("world", "table", "cube"):
                body.gravcomp = 1.0
        self.model = spec.compile()
        if int(self.model.ngravcomp) == 0:
            raise RuntimeError("Gravity compensation was not compiled into the model")
        self.data = mujoco.MjData(self.model)
        self._configure_options()
        self._resolve_indices()
        self._place_and_configure_robot()
        self._configure_collisions()
        self._validate_object()
        mujoco.mj_forward(self.model, self.data)
        if self.config.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            offset = (
                0.5 * self.config.reference_ghost_offset_world
                if self.config.enable_reference_ghost
                else np.zeros(3)
            )
            self.viewer.cam.lookat[:] = self.config.robot_position_world + offset + np.asarray((0.0, -0.05, 0.10))
            self.viewer.cam.distance = 2.2
            self.viewer.cam.azimuth = 40.0
            self.viewer.cam.elevation = -20.0
            self.viewer.sync()

    def _add_world(self, spec) -> None:
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = np.asarray((1.5, 1.5, 0.05))
        floor.rgba = np.asarray((0.20, 0.25, 0.28, 1.0))

        table = spec.worldbody.add_body()
        table.name = "table"
        table.pos = np.asarray(
            (
                0.0,
                0.0,
                self.config.robot_position_world[2]
                - self.config.table_surface_below_robot_base_m
                - self.config.table_size_m[2] / 2.0,
            )
        )
        table_geom = table.add_geom()
        table_geom.name = "table_geom"
        table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        table_geom.size = self.config.table_size_m / 2.0
        table_geom.rgba = np.asarray((0.82, 0.56, 0.35, 1.0))

        cube = spec.worldbody.add_body()
        cube.name = "cube"
        joint = cube.add_joint()
        joint.name = "cube_free_joint"
        joint.type = mujoco.mjtJoint.mjJNT_FREE
        geom = cube.add_geom()
        geom.name = "cube_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        size = self.config.scaled_object_size_m
        geom.size = size / 2.0
        # Density keeps the inertia consistent with the mass the scale gives
        # (s^3 with the volume, i.e. the same material at every size).
        geom.density = self.config.scaled_object_mass_kg / float(np.prod(size))
        geom.rgba = np.asarray((0.78, 0.78, 0.82, 1.0))

        light = spec.worldbody.add_light()
        light.name = "key_light"
        light.pos = np.asarray((0.0, -1.0, 1.5))
        light.dir = np.asarray((0.0, 0.5, -1.0))
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

    def _add_position_actuators(self, spec) -> None:
        for index, joint_name in enumerate(JOINT_NAMES):
            actuator = spec.add_actuator()
            actuator.name = "{}_pos".format(joint_name)
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.target = joint_name
            lower, upper, effort, _ = self._urdf_limits[joint_name]
            actuator.ctrllimited = True
            actuator.ctrlrange = np.asarray((lower, upper))
            actuator.forcelimited = True
            actuator.forcerange = np.asarray((-effort, effort))
            kp = float(self.config.stiffness[index])
            kv = float(self.config.damping[index])
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.gainprm[0] = kp
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = -kv

    def _add_self_collision_excludes(self, spec) -> None:
        """Author the complement of the allowed hand pairs, as the env does on USD."""
        if not self.config.self_collision:
            return
        self._self_collision_excluded = set()
        for pair in filtered_body_pairs(self.config.asset_cfg, HAND_BODY_NAMES):
            a, b = sorted(pair)
            for body_a in MUJOCO_BODY_GROUPS.get(a, (a,)):
                for body_b in MUJOCO_BODY_GROUPS.get(b, (b,)):
                    exclude = spec.add_exclude()
                    exclude.name = "{}__{}".format(body_a, body_b)
                    exclude.bodyname1 = body_a
                    exclude.bodyname2 = body_b
                    self._self_collision_excluded.add(frozenset((body_a, body_b)))

    def _configure_options(self) -> None:
        opt = self.model.opt
        opt.timestep = float(self.config.sim_dt)
        opt.gravity[:] = self.config.gravity
        opt.integrator = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "rk4": mujoco.mjtIntegrator.mjINT_RK4,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }[self.config.integrator.lower()]
        opt.cone = (
            mujoco.mjtCone.mjCONE_ELLIPTIC
            if self.config.cone.lower() == "elliptic"
            else mujoco.mjtCone.mjCONE_PYRAMIDAL
        )
        opt.impratio = float(self.config.impratio)
        opt.iterations = int(self.config.iterations)
        opt.ls_iterations = int(self.config.ls_iterations)
        opt.tolerance = float(self.config.tolerance)

    def _resolve_indices(self) -> None:
        m = self.model
        self._joint_qpos_adrs = np.asarray([m.joint(n).qposadr[0] for n in JOINT_NAMES], dtype=np.int32)
        self._joint_dof_adrs = np.asarray([m.joint(n).dofadr[0] for n in JOINT_NAMES], dtype=np.int32)
        self._actuator_ids = np.asarray([m.actuator("{}_pos".format(n)).id for n in JOINT_NAMES], dtype=np.int32)
        self._base_body_id = m.body(ROBOT_BASE_BODY_NAME).id
        self._wrist_body_id = m.body(WRIST_BODY_NAME).id
        self._fingertip_body_ids = np.asarray([m.body(n).id for n in FINGERTIP_BODY_NAMES], dtype=np.int32)
        self._fingertip_link_ids = np.asarray([m.body(n).id for n in FINGERTIP_LINK_NAMES], dtype=np.int32)
        self._cube_body_id = m.body("cube").id
        cube_joint = m.joint("cube_free_joint")
        self._cube_qpos_adr = int(cube_joint.qposadr[0])
        self._cube_dof_adr = int(cube_joint.dofadr[0])
        if self.config.enable_reference_ghost:
            self._ghost_base_body_id = m.body("ghost_" + ROBOT_BASE_BODY_NAME).id
            self._ghost_qpos_adrs = np.asarray(
                [m.joint("ghost_" + n).qposadr[0] for n in JOINT_NAMES], dtype=np.int32
            )
            self._ghost_dof_adrs = np.asarray(
                [m.joint("ghost_" + n).dofadr[0] for n in JOINT_NAMES], dtype=np.int32
            )
        else:
            self._ghost_base_body_id = None
            self._ghost_qpos_adrs = np.empty(0, dtype=np.int32)
            self._ghost_dof_adrs = np.empty(0, dtype=np.int32)
        self.joint_lower_limits = np.asarray([m.jnt_range[m.joint(n).id, 0] for n in JOINT_NAMES])
        self.joint_upper_limits = np.asarray([m.jnt_range[m.joint(n).id, 1] for n in JOINT_NAMES])
        self.joint_velocity_limits = np.asarray([self._urdf_limits[n][3] for n in JOINT_NAMES])
        if np.any(self.joint_lower_limits >= self.joint_upper_limits):
            raise RuntimeError("MuJoCo imported invalid robot joint limits")

    def _body_is_descendant_of(self, body_id: int, ancestor_id: int) -> bool:
        current = int(body_id)
        while current != 0:
            if current == ancestor_id:
                return True
            current = int(self.model.body_parentid[current])
        return False

    def _robot_body_ids(self, base_id: int) -> np.ndarray:
        return np.asarray(
            [b for b in range(1, self.model.nbody) if self._body_is_descendant_of(b, base_id)],
            dtype=np.int32,
        )

    def _place_and_configure_robot(self) -> None:
        m = self.model
        quat = self.config.robot_orientation_world_xyzw / np.linalg.norm(self.config.robot_orientation_world_xyzw)
        m.body_pos[self._base_body_id] = self.config.robot_position_world
        m.body_quat[self._base_body_id] = _wxyz(quat)
        m.dof_damping[self._joint_dof_adrs] = 0.0
        if self.config.enable_reference_ghost:
            m.body_pos[self._ghost_base_body_id] = (
                self.config.robot_position_world + self.config.reference_ghost_offset_world
            )
            m.body_quat[self._ghost_base_body_id] = _wxyz(quat)
            m.dof_damping[self._ghost_dof_adrs] = 0.0
            rgba = np.concatenate((self.config.reference_ghost_color, (0.72,)))
            for geom_id in range(m.ngeom):
                if self._body_is_descendant_of(int(m.geom_bodyid[geom_id]), self._ghost_base_body_id):
                    m.geom_rgba[geom_id] = rgba

    def _configure_collisions(self) -> None:
        """The training collision graph, plus the training contact parameters on every collider."""
        m = self.model
        cube_geom = m.geom("cube_geom").id
        table_geom = m.geom("table_geom").id
        floor_geom = m.geom("floor").id
        fingertip_bodies = set(int(v) for v in self._fingertip_body_ids) | set(
            int(v) for v in self._fingertip_link_ids
        )
        hand_affinity = CUBE_BIT | (HAND_BIT if self.config.self_collision else 0)
        for geom_id in range(m.ngeom):
            body_id = int(m.geom_bodyid[geom_id])
            torsional = 0.0
            if geom_id == cube_geom:
                contype, conaffinity, friction = CUBE_BIT, HAND_BIT | TABLE_BIT, self.config.object_friction
            elif geom_id == table_geom:
                contype, conaffinity, friction = TABLE_BIT, CUBE_BIT, self.config.table_friction
            elif geom_id == floor_geom:
                contype, conaffinity, friction = 0, 0, 1.0
            elif self._body_is_descendant_of(body_id, self._wrist_body_id):
                contype, conaffinity = HAND_BIT, hand_affinity
                if body_id in fingertip_bodies:
                    friction = self.config.fingertip_friction
                    torsional = self.config.fingertip_torsional_friction
                else:
                    friction = self.config.robot_friction
            else:
                # Arm links and the reference ghost: visual only.
                contype, conaffinity, friction = 0, 0, self.config.robot_friction
            m.geom_contype[geom_id] = contype
            m.geom_conaffinity[geom_id] = conaffinity
            m.geom_friction[geom_id] = (friction, torsional, 0.0)
            if contype or conaffinity:
                m.geom_condim[geom_id] = self.config.contact_condim
                # Newton: forces once two shapes are closer than ``margin``
                # (0 = at penetration), detection a further ``gap`` away.
                # MuJoCo's margin is not a detection band: the solver drives
                # an active contact to ``dist == margin``, so a positive value
                # would float the bar above the table. Newton's detection gap
                # therefore has no MuJoCo counterpart and only the force
                # margin is carried over.
                m.geom_margin[geom_id] = self.config.contact_margin
                m.geom_gap[geom_id] = 0.0
                if self.config.contact_solref is not None:
                    m.geom_solref[geom_id] = self.config.contact_solref
                if self.config.contact_solimp is not None:
                    m.geom_solimp[geom_id] = self.config.contact_solimp

    def _validate_object(self) -> None:
        mass = float(self.model.body_mass[self._cube_body_id])
        expected = self.config.scaled_object_mass_kg
        if not np.isclose(mass, expected, rtol=1e-5):
            raise RuntimeError("Compiled bar mass {} does not match {}".format(mass, expected))

    # -- collision graph introspection (tests, reports) ---------------------------
    def geoms_can_collide(self, geom_a: int, geom_b: int) -> bool:
        """MuJoCo's static pair filter: masks, excludes and the parent-child rule."""
        m = self.model
        body_a, body_b = int(m.geom_bodyid[geom_a]), int(m.geom_bodyid[geom_b])
        if body_a == body_b:
            return False
        mask = (m.geom_contype[geom_a] & m.geom_conaffinity[geom_b]) or (
            m.geom_contype[geom_b] & m.geom_conaffinity[geom_a]
        )
        if not mask:
            return False
        weld_a, weld_b = int(m.body_weldid[body_a]), int(m.body_weldid[body_b])
        if weld_a == weld_b:
            return False
        parent_a = int(m.body_weldid[m.body_parentid[weld_a]])
        parent_b = int(m.body_weldid[m.body_parentid[weld_b]])
        if (parent_a == weld_b and weld_b != 0) or (parent_b == weld_a and weld_a != 0):
            return False
        for index in range(m.nexclude):
            signature = int(m.exclude_signature[index])
            first, second = signature >> 16, signature & 0xFFFF
            if {first, second} == {body_a, body_b}:
                return False
        return True

    def bodies_can_collide(self, body_a: str, body_b: str) -> bool:
        m = self.model
        ids_a = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body_a).id]
        ids_b = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(body_b).id]
        return any(self.geoms_can_collide(a, b) for a in ids_a for b in ids_b)

    # -- state -------------------------------------------------------------------
    def reset(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        cube_root_state_world: np.ndarray,
    ) -> None:
        """Joint state plus the bar's world root state (position, xyzw, linear, angular)."""
        q = np.asarray(joint_positions, dtype=np.float64).reshape(ACTION_DIM)
        dq = np.asarray(joint_velocities, dtype=np.float64).reshape(ACTION_DIM)
        root = np.asarray(cube_root_state_world, dtype=np.float64).reshape(13)
        self.data.qpos[self._joint_qpos_adrs] = q
        self.data.qvel[self._joint_dof_adrs] = dq
        adr, dof = self._cube_qpos_adr, self._cube_dof_adr
        self.data.qpos[adr : adr + 3] = root[0:3]
        self.data.qpos[adr + 3 : adr + 7] = _wxyz(root[3:7] / np.linalg.norm(root[3:7]))
        self.data.qvel[dof : dof + 3] = root[7:10]
        self.data.qvel[dof + 3 : dof + 6] = root[10:13]
        self.data.qacc_warmstart[:] = 0.0
        self.set_position_targets(q)
        mujoco.mj_forward(self.model, self.data)
        self.sync_viewer()

    def set_position_targets(self, targets: np.ndarray) -> None:
        self.data.ctrl[self._actuator_ids] = np.asarray(targets, dtype=np.float64).reshape(ACTION_DIM)

    def set_reference_ghost(self, joint_positions: np.ndarray) -> None:
        if not self.config.enable_reference_ghost:
            return
        self.data.qpos[self._ghost_qpos_adrs] = np.asarray(joint_positions, dtype=np.float64).reshape(ACTION_DIM)
        self.data.qvel[self._ghost_dof_adrs] = 0.0

    def step_control(self) -> None:
        """One control step: the configured substeps, then the training's velocity cap."""
        for _ in range(self.config.substeps):
            mujoco.mj_step(self.model, self.data)
        self._clamp_object_velocity()
        self.sync_viewer()

    def _clamp_object_velocity(self) -> None:
        dof = self._cube_dof_adr
        linear = self.data.qvel[dof : dof + 3]
        angular = self.data.qvel[dof + 3 : dof + 6]
        limit = self.config.object_max_linear_velocity
        if limit > 0.0:
            speed = float(np.linalg.norm(linear))
            if speed > limit:
                self.data.qvel[dof : dof + 3] = linear * (limit / speed)
        limit = self.config.object_max_angular_velocity
        if limit > 0.0:
            rate = float(np.linalg.norm(angular))
            if rate > limit:
                self.data.qvel[dof + 3 : dof + 6] = angular * (limit / rate)

    def robot_cube_contacts(self) -> list:
        """``(body name, distance)`` of every live hand-bar contact."""
        cube_geom = self.model.geom("cube_geom").id
        contacts = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geoms = (int(contact.geom1), int(contact.geom2))
            if cube_geom not in geoms:
                continue
            other = geoms[1] if geoms[0] == cube_geom else geoms[0]
            body = int(self.model.geom_bodyid[other])
            if self._body_is_descendant_of(body, self._wrist_body_id):
                contacts.append((self.model.body(body).name, float(contact.dist)))
        return contacts

    def get_state(self) -> Dict[str, np.ndarray]:
        d, m = self.data, self.model
        adr, dof = self._cube_qpos_adr, self._cube_dof_adr
        return {
            "joint_positions": d.qpos[self._joint_qpos_adrs].copy(),
            "joint_velocities": d.qvel[self._joint_dof_adrs].copy(),
            "robot_position_world": d.xpos[self._base_body_id].copy(),
            "robot_orientation_world_xyzw": _xyzw(d.xquat[self._base_body_id]),
            "wrist_position_world": d.xpos[self._wrist_body_id].copy(),
            "wrist_orientation_world_xyzw": _xyzw(d.xquat[self._wrist_body_id]),
            "fingertip_body_positions_world": d.xpos[self._fingertip_body_ids].copy(),
            "fingertip_body_orientations_world_xyzw": np.stack(
                [_xyzw(d.xquat[b]) for b in self._fingertip_body_ids]
            ),
            "fingertip_link_positions_world": d.xpos[self._fingertip_link_ids].copy(),
            "palm_link_position_world": d.xpos[m.body("rl_dg_palm").id].copy(),
            "palm_link_orientation_world_xyzw": _xyzw(d.xquat[m.body("rl_dg_palm").id]),
            "cube_position_world": d.xpos[self._cube_body_id].copy(),
            "cube_orientation_world_xyzw": _xyzw(d.xquat[self._cube_body_id]),
            "cube_linear_velocity_world": d.qvel[dof : dof + 3].copy(),
            "cube_angular_velocity_world": d.qvel[dof + 3 : dof + 6].copy(),
            "cube_com_height_world": float(d.xipos[self._cube_body_id][2]),
        }

    def sync_viewer(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def viewer_is_running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._tmp_dir.cleanup()

    def __enter__(self) -> "MujocoSim":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
