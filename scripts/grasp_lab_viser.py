#!/usr/bin/env python3
"""Hand-driven grasp laboratory in a Viser GUI.

No policy. The simulator is built from a run's ``config.json`` -- the same
contact model, frictions, object, joint drives and finger PD gains that run
trained with -- and the robot is placed at a late pre-grasp frame of the
demonstration (830 by default, fingers around the bar just before closure)
with the cuboid at the bank's base placement (x = y = yaw = 0). Every joint
then has a slider (arm and all twenty finger joints), plus a nudge panel for
fine +/- moves of one joint at a time.

Two modes, switched by one button:

* **Physics stopped** (the default): moving a slider writes the joint angle
  straight into the simulator and refreshes the scene. Nothing is integrated,
  so the cuboid stays where it is while the fingers are arranged around it.
* **Physics running**: the sliders are the position targets of the joint
  drives, and the simulator steps at the control rate with gravity, contacts
  and the finger PD controllers exactly as in training. This is where a
  candidate grasp is tested: does the bar stay in the hand, does it roll?

``Save pose`` writes the joint vector and the cuboid pose to a JSON file for
reuse; ``Reset scene`` returns to the pre-grasp pose.

The **Physics parameters** panel exposes the contact model (friction
coefficients, condim, cone, impratio, margin/gap, solref/solimp, multi-CCD,
MuJoCo contacts), the substep count, the object mass and the finger drive
gain. ``Apply`` rebuilds the simulator with the new values (about a second)
and restores the joints, the drive targets and the cuboid pose, so a grasp
can be re-tested under different physics without relaunching.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python scripts/grasp_lab_viser.py \\
        --config logs/simtoolreal/2026-09-16_153055_s2s_track8_palm04
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate import load_saved_configuration  # noqa: E402
from evaluate_viser import (  # noqa: E402
    CUBE_DIMENSIONS,
    DEFAULT_URDF,
    ROBOT_WXYZ_WORLD,
    load_urdf_for_viser,
    quaternion_xyzw_to_wxyz,
)
from simtoolreal_newton.envs.controller import (  # noqa: E402
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
)
from simtoolreal_newton.envs.contact import fingertip_force_norms  # noqa: E402
from simtoolreal_newton.envs.motion_imitation_env import (  # noqa: E402
    _quat_rotate_inverse,
)
from simtoolreal_newton.envs.proximity import fingertip_cuboid_proximity  # noqa: E402
from simtoolreal_newton.launch import make_env  # noqa: E402
from test_headless_env import apply_overrides  # noqa: E402

import torch  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "logs/simtoolreal/2026-09-16_153055_s2s_track8_palm04"
# Demonstration frame the scene starts from: 830 is the last RSI start index
# training uses, just before the pinch closes, so the fingers are already
# around the bar and only the closure is left to arrange by hand.
DEFAULT_PREGRASP_INDEX = 830
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
# rl_dg_<finger>_<phalanx>: four joints per finger, demonstration order.
HAND_JOINT_LABELS = ("spread", "mcp", "pip", "dip")


def resolve_config_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        raise FileNotFoundError("Configuration not found: {}".format(path))
    return path


class GraspLab:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import viser
            from viser.extras import ViserUrdf
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "This tool needs viser and yourdfpy in the project venv: "
                'ISAACLAB_EXTRAS="--extra viser" ./setup.sh, then '
                "uv pip install --python deps/IsaacLab/.venv/bin/python yourdfpy"
            ) from exc

        self.args = args
        self.config_path = resolve_config_path(args.config)
        # Overrides the GUI's Physics panel applies on top of the command line.
        self.gui_overrides: List[str] = []
        self._build_environment()

        self.lock = threading.Lock()
        self.physics_running = False
        self.updating_sliders = False
        self.steps = 0

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self.port = int(self.server.get_port())
        self.server.scene.add_grid(
            "/ground", width=3.0, height=3.0, cell_size=0.1, position=(0, 0, 0)
        )
        table_size = np.asarray([float(v) for v in self.env_cfg.table.size_m])
        self.server.scene.add_box(
            "/table",
            dimensions=table_size,
            position=(0.0, 0.0, self.table_top_z - table_size[2] / 2.0),
            color=(209, 143, 89),
            opacity=0.9,
        )
        self.server.scene.add_frame(
            "/robot", position=self.robot_position, wxyz=ROBOT_WXYZ_WORLD, show_axes=False
        )
        urdf = load_urdf_for_viser(args.urdf.expanduser().resolve())
        self.robot = ViserUrdf(self.server, urdf, root_node_name="/robot")
        expected = tuple(ARM_JOINT_NAMES) + tuple(HAND_JOINT_NAMES)
        actual = tuple(self.robot.get_actuated_joint_names())
        if actual != expected:
            raise ValueError(
                "URDF actuated-joint order does not match demonstration order:\n"
                "  urdf: {}\n  demo: {}".format(actual, expected)
            )
        self.cube = self.server.scene.add_box(
            "/cube",
            dimensions=CUBE_DIMENSIONS,
            color=(65, 115, 210),
            opacity=0.95,
            side="double",
        )
        self._build_gui()

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.9, -0.9, 1.1)
            client.camera.look_at = (0.0, 0.3, 0.6)

        self.reset_scene()

    # ------------------------------------------------------------------ setup

    def _build_environment(self) -> None:
        args = self.args
        env_cfg, train_cfg = load_saved_configuration(self.config_path)
        if args.demo is not None:
            # A demonstration other than the run's: the scene is then reset
            # from its raw frames rather than from the transform bank, which
            # was built from the run's own demonstration.
            env_cfg.motion.file = str(args.demo.expanduser().resolve())
        env_cfg.seed = int(args.seed)
        env_cfg.env.num_envs = 1
        env_cfg.env.play = True
        env_cfg.viewer.enable_viewer = False
        env_cfg.viewer.reference_ghost = False
        env_cfg.viewer.training_camera_enabled = False
        # Nothing here is an episode: no terminations, no assist, no
        # randomisation. The physics (sim.*, control.*, asset.*, object.*)
        # is exactly the saved run's.
        env_cfg.termination.enabled = False
        env_cfg.object_assist.enabled = False
        env_cfg.domain_randomization.enabled = False
        # Physics knobs to explore with, e.g. control.hand_stiffness_scale=0.5,
        # sim.mjwarp.contact_solref=[0.02,1.0], asset.fingertip_torsional_friction=0.05,
        # contact.enabled=true (shows fingertip forces).
        apply_overrides(env_cfg, list(args.overrides) + list(self.gui_overrides))
        train_cfg.runner.record_video = False
        self.env_cfg = env_cfg
        self.env = make_env(
            env_cfg, num_envs=None, device=args.sim_device, physics=args.physics, visualizer=None
        )
        self.inner = self.env.unwrapped
        self.robot_position = np.asarray([float(v) for v in env_cfg.init_state.pos], dtype=np.float64)
        self.table_top_z = float(env_cfg.init_state.pos[2]) - float(env_cfg.table.surface_below_robot_base_m)
        self.pregrasp_index = int(args.pregrasp_index)
        inner = self.inner
        self.lower = inner.joint_lower_limits.detach().cpu().numpy().astype(np.float64)
        self.upper = inner.joint_upper_limits.detach().cpu().numpy().astype(np.float64)
        self.num_arm = len(ARM_JOINT_NAMES)
        self.control_dt = float(self.env.dt)
        self.decimation = int(inner.cfg.decimation)
        self.physics_dt = float(inner.physics_dt)
        self.demo_path = Path(env_cfg.motion.file)
        if self.pregrasp_index > int(inner.reference.last_index):
            raise ValueError(
                "--pregrasp-index {} beyond the demonstration's last sample {}".format(
                    self.pregrasp_index, int(inner.reference.last_index)
                )
            )

    # -------------------------------------------------------------------- gui

    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.add_markdown(
            "**Grasp lab** — physics from `{}`{}  \n"
            "Demonstration `{}`, frame {}, identity placement.".format(
                self.config_path.parent.name,
                " with " + ", ".join("`{}`".format(o) for o in self.args.overrides) if self.args.overrides else "",
                self.demo_path.name,
                self.pregrasp_index,
            )
        )
        with gui.add_folder("Physics", expand_by_default=True):
            self.physics_button = gui.add_button("Start physics", color="green")
            self.physics_button.on_click(self._toggle_physics)
            self.physics_status = gui.add_markdown("**Physics:** stopped (sliders move the joints kinematically).")
            self.rate = gui.add_slider("Speed (x real time)", min=0.1, max=2.0, step=0.1, initial_value=1.0)
            reset_button = gui.add_button("Reset scene (pre-grasp)")
            reset_button.on_click(lambda _: self.reset_scene())
            reset_cube = gui.add_button("Reset cuboid only")
            reset_cube.on_click(lambda _: self.reset_cube())
            save_button = gui.add_button("Save pose")
            save_button.on_click(self._save_pose)
            self.save_note = gui.add_markdown("")
        with gui.add_folder("Measurements", expand_by_default=True):
            self.readout = gui.add_markdown("")
        self._build_physics_panel(gui)
        with gui.add_folder("Nudge one joint", expand_by_default=True):
            names = list(ARM_JOINT_NAMES) + list(self._hand_labels())
            self.nudge_joint = gui.add_dropdown("Joint", options=names, initial_value=names[self.num_arm])
            self.nudge_step = gui.add_slider("Step [rad]", min=0.005, max=0.2, step=0.005, initial_value=0.02)
            minus = gui.add_button("−")
            plus = gui.add_button("+")
            minus.on_click(lambda _: self._nudge(-1.0))
            plus.on_click(lambda _: self._nudge(+1.0))

        self.sliders: List = []
        with gui.add_folder("Arm joints [rad]", expand_by_default=False):
            for index, name in enumerate(ARM_JOINT_NAMES):
                self.sliders.append(self._add_joint_slider(gui, name, index))
        labels = self._hand_labels()
        for finger, finger_name in enumerate(FINGER_NAMES):
            with gui.add_folder("{} [rad]".format(finger_name.capitalize()), expand_by_default=(finger < 3)):
                for phalanx in range(4):
                    index = self.num_arm + 4 * finger + phalanx
                    self.sliders.append(self._add_joint_slider(gui, labels[4 * finger + phalanx], index))

    def _build_physics_panel(self, gui) -> None:
        """Controls for the contact model and drives; Apply rebuilds the sim."""
        cfg = self.env_cfg
        mj = cfg.sim.mjwarp
        torsional = cfg.asset.fingertip_torsional_friction
        rolling = cfg.asset.fingertip_rolling_friction
        solref = mj.contact_solref if mj.contact_solref is not None else [0.02, 1.0]
        solimp = mj.contact_solimp if mj.contact_solimp is not None else [0.9, 0.95, 0.001, 0.5, 2.0]
        with gui.add_folder("Physics parameters", expand_by_default=False):
            gui.add_markdown(
                "Edit, then **Apply**: the simulator is rebuilt (~1 s) and the "
                "joints, targets and cuboid pose are put back."
            )
            self.p_apply = gui.add_button("Apply parameters (rebuild sim)", color="orange")
            self.p_apply.on_click(self._apply_physics_parameters)
            self.p_note = gui.add_markdown("")
            with gui.add_folder("Friction", expand_by_default=True):
                self.p_friction = gui.add_slider(
                    "fingertip_friction", min=0.1, max=3.0, step=0.05, initial_value=float(cfg.asset.fingertip_friction)
                )
                self.p_object_friction = gui.add_slider(
                    "object.friction", min=0.1, max=3.0, step=0.05, initial_value=float(cfg.object.friction)
                )
                self.p_condim = gui.add_dropdown(
                    "contact_condim", options=("3", "4", "6"), initial_value=str(int(mj.contact_condim))
                )
                self.p_torsional = gui.add_slider(
                    "fingertip_torsional_friction (0 = off)", min=0.0, max=0.3, step=0.005,
                    initial_value=float(torsional) if torsional is not None else 0.0,
                )
                self.p_rolling = gui.add_slider(
                    "fingertip_rolling_friction (0 = off)", min=0.0, max=0.1, step=0.001,
                    initial_value=float(rolling) if rolling is not None else 0.0,
                )
                self.p_cone = gui.add_dropdown("cone", options=("elliptic", "pyramidal"), initial_value=str(mj.cone))
                self.p_impratio = gui.add_slider("impratio", min=1.0, max=20.0, step=0.5, initial_value=float(mj.impratio))
            with gui.add_folder("Contact generation", expand_by_default=True):
                self.p_multiccd = gui.add_checkbox("enable_multiccd", initial_value=bool(mj.enable_multiccd))
                self.p_mujoco_contacts = gui.add_checkbox(
                    "use_mujoco_contacts", initial_value=bool(mj.use_mujoco_contacts)
                )
                self.p_margin = gui.add_slider(
                    "contact_margin [m]", min=0.0, max=0.02, step=0.0005, initial_value=float(mj.contact_margin)
                )
                self.p_gap = gui.add_slider(
                    "contact_gap [m]", min=0.0, max=0.03, step=0.0005, initial_value=float(mj.contact_gap)
                )
            with gui.add_folder("Contact softness", expand_by_default=True):
                self.p_solref = gui.add_text(
                    "contact_solref (timeconst, dampratio)", initial_value=", ".join("{:g}".format(v) for v in solref)
                )
                self.p_solimp = gui.add_text(
                    "contact_solimp (d0, dwidth, width, midpoint, power)",
                    initial_value=", ".join("{:g}".format(v) for v in solimp),
                )
                self.p_substeps = gui.add_slider("substeps", min=1, max=16, step=1, initial_value=int(cfg.sim.substeps))
                self.p_max_lin_vel = gui.add_slider(
                    "object_max_linear_velocity [m/s] (0 = off)", min=0.0, max=10.0, step=0.25,
                    initial_value=float(mj.object_max_linear_velocity),
                )
            with gui.add_folder("Object and drives", expand_by_default=True):
                self.p_mass = gui.add_slider(
                    "object.mass_kg", min=0.02, max=1.0, step=0.01, initial_value=float(cfg.object.mass_kg)
                )
                self.p_hand_stiffness = gui.add_slider(
                    "hand_stiffness_scale", min=0.05, max=1.5, step=0.01,
                    initial_value=float(cfg.control.hand_stiffness_scale),
                )
                self.p_hand_damping = gui.add_slider(
                    "hand_damping_scale", min=0.1, max=5.0, step=0.1,
                    initial_value=float(cfg.control.hand_damping_scale),
                )

    def _physics_overrides_from_gui(self) -> List[str]:
        def numbers(text: str, count: int, label: str) -> str:
            values = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
            if len(values) != count:
                raise ValueError("{} needs {} numbers, got {}".format(label, count, len(values)))
            return json.dumps(values)

        torsional = float(self.p_torsional.value)
        rolling = float(self.p_rolling.value)
        return [
            "asset.fingertip_friction={:g}".format(self.p_friction.value),
            "object.friction={:g}".format(self.p_object_friction.value),
            "sim.mjwarp.contact_condim={}".format(int(self.p_condim.value)),
            "asset.fingertip_torsional_friction={}".format("{:g}".format(torsional) if torsional > 0.0 else "null"),
            "asset.fingertip_rolling_friction={}".format("{:g}".format(rolling) if rolling > 0.0 else "null"),
            "sim.mjwarp.cone={}".format(self.p_cone.value),
            "sim.mjwarp.impratio={:g}".format(self.p_impratio.value),
            "sim.mjwarp.enable_multiccd={}".format("true" if self.p_multiccd.value else "false"),
            "sim.mjwarp.use_mujoco_contacts={}".format("true" if self.p_mujoco_contacts.value else "false"),
            "sim.mjwarp.contact_margin={:g}".format(self.p_margin.value),
            "sim.mjwarp.contact_gap={:g}".format(self.p_gap.value),
            "sim.mjwarp.contact_solref={}".format(numbers(self.p_solref.value, 2, "contact_solref")),
            "sim.mjwarp.contact_solimp={}".format(numbers(self.p_solimp.value, 5, "contact_solimp")),
            "sim.substeps={}".format(int(self.p_substeps.value)),
            "sim.mjwarp.object_max_linear_velocity={:g}".format(self.p_max_lin_vel.value),
            "object.mass_kg={:g}".format(self.p_mass.value),
            "control.hand_stiffness_scale={:g}".format(self.p_hand_stiffness.value),
            "control.hand_damping_scale={:g}".format(self.p_hand_damping.value),
        ]

    def _apply_physics_parameters(self, _) -> None:
        """Rebuild the simulator with the panel's values, keeping the scene."""
        try:
            overrides = self._physics_overrides_from_gui()
        except ValueError as exc:
            self.p_note.content = "**Not applied:** {}".format(exc)
            return
        with self.lock:
            was_running = self.physics_running
            self.physics_running = False
            measured = self._measure()
            q = measured["q"]
            targets = self._targets()
            lift_baseline = float(self.inner.episode_initial_object_com_height_m[0])
            cube_pose = torch.cat(
                (
                    self.inner.cube_position[0],
                    self.inner.cube_orientation[0],
                )
            ).detach().cpu().numpy().astype(np.float64)
            self.p_note.content = "Rebuilding..."
            started = time.monotonic()
            self.env.close()
            self.gui_overrides = overrides
            failure = None
            try:
                self._build_environment()
            except Exception as exc:  # the old sim is gone; rebuild the previous one
                failure = exc
                self.gui_overrides = []
                self._build_environment()
        if failure is not None:
            self.reset_scene()
            self.p_note.content = "**Rebuild failed, previous physics restored:** {}".format(failure)
            return
        with self.lock:
            inner = self.inner
            inner.reset_all(reference_index=self.pregrasp_index, translation_xy=(0.0, 0.0), yaw_rad=0.0)
            self._write_kinematic_pose(q)
            pose = torch.as_tensor(cube_pose, dtype=torch.float32, device=inner.device).unsqueeze(0).clone()
            pose[:, 0:3] += inner.env_origins[:1]
            ids = torch.zeros(1, dtype=torch.int32, device=inner.device)
            inner.cube.write_root_pose_to_sim_index(root_pose=pose.contiguous(), env_ids=ids)
            inner.cube.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=inner.device), env_ids=ids
            )
            self._flush()
            inner.episode_initial_object_com_height_m[0] = lift_baseline
            # Drive targets are the sliders again; the applied targets start
            # from the restored joints so the preload ramps in through the slew.
            inner.position_targets[0] = torch.as_tensor(targets, dtype=torch.float32, device=inner.device)
            inner.applied_targets[0] = inner.q[0]
            self.steps = 0
            self.physics_running = was_running
        self.p_note.content = "Applied in {:.1f} s: {}".format(
            time.monotonic() - started, ", ".join("`{}`".format(o) for o in overrides)
        )
        print("[grasp-lab] physics rebuilt with: " + " ".join("--set " + o for o in overrides), flush=True)
        self._publish()

    def _hand_labels(self) -> List[str]:
        return [
            "{}_{}".format(finger, label)
            for finger in FINGER_NAMES
            for label in HAND_JOINT_LABELS
        ]

    def _add_joint_slider(self, gui, label: str, index: int):
        lower, upper = float(self.lower[index]), float(self.upper[index])
        slider = gui.add_slider(
            label, min=lower, max=upper, step=0.001, initial_value=float(np.clip(0.0, lower, upper))
        )

        @slider.on_update
        def _(_event, index=index):
            if self.updating_sliders:
                return
            self._slider_moved(index)

        return slider

    # ---------------------------------------------------------------- actions

    def _targets(self) -> np.ndarray:
        return np.asarray([float(s.value) for s in self.sliders], dtype=np.float64)

    def _set_sliders(self, q: np.ndarray) -> None:
        self.updating_sliders = True
        try:
            for slider, value in zip(self.sliders, q):
                slider.value = float(np.clip(value, slider.min, slider.max))
        finally:
            self.updating_sliders = False

    def _slider_moved(self, index: int) -> None:
        with self.lock:
            if self.physics_running:
                # Targets are read every physics step; nothing else to do.
                return
            self._write_kinematic_pose(self._targets())
        self._publish()

    def _nudge(self, direction: float) -> None:
        names = list(ARM_JOINT_NAMES) + list(self._hand_labels())
        index = names.index(self.nudge_joint.value)
        slider = self.sliders[index]
        slider.value = float(np.clip(slider.value + direction * float(self.nudge_step.value), slider.min, slider.max))
        # The slider's on_update fires for programmatic changes too, so the
        # kinematic write happens there.

    def _toggle_physics(self, _) -> None:
        with self.lock:
            self.physics_running = not self.physics_running
            running = self.physics_running
            if running:
                # The drives start from where the joints are: targets equal
                # the current sliders, applied targets equal the current
                # joint angles, so nothing jumps on the first step.
                inner = self.inner
                q = self._targets()
                inner.position_targets[0] = torch.as_tensor(q, dtype=torch.float32, device=inner.device)
                inner.applied_targets[0] = inner.q[0]
        self.physics_button.label = "Stop physics" if running else "Start physics"
        self.physics_button.color = "red" if running else "green"
        self.physics_status.content = (
            "**Physics:** running (sliders are the drive targets)."
            if running
            else "**Physics:** stopped (sliders move the joints kinematically)."
        )

    def _demo_sample(self):
        inner = self.inner
        indices = torch.full((1,), self.pregrasp_index, dtype=torch.long, device=inner.device)
        return inner.reference.sample(indices)

    def _demo_cube_root_state(self, sample) -> torch.Tensor:
        """The demonstration's cuboid pose in the env frame, identity placement.

        Same axis convention as MotionImitationEnv._cube_reference_root_states
        (UR base frame to env frame, xyzw quaternion), without the transform
        bank's placement correction, so the scene is the raw demonstration.
        """
        inner = self.inner
        position = inner.robot_base_position + sample.cube_pose[:, :3] * inner.world_axis_sign
        x, y, z, w = sample.cube_pose[:, 3:7].unbind(dim=1)
        quaternion = torch.nn.functional.normalize(torch.stack((-y, x, w, -z), dim=1), dim=1)
        linear = sample.cube_linear_velocity * inner.world_axis_sign
        angular = sample.cube_angular_velocity * inner.world_axis_sign
        return torch.cat((position, quaternion, linear, angular), dim=1)

    def _place_cube_from_demo(self) -> None:
        inner = self.inner
        state = self._demo_cube_root_state(self._demo_sample())
        pose = state[:, 0:7].clone()
        pose[:, 0:3] += inner.env_origins[:1]
        ids = torch.zeros(1, dtype=torch.int32, device=inner.device)
        inner.cube.write_root_pose_to_sim_index(root_pose=pose.contiguous(), env_ids=ids)
        inner.cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(state[:, 7:13]), env_ids=ids)

    def reset_scene(self) -> None:
        with self.lock:
            self.physics_running = False
            self.steps = 0
            # reset_all sets up every episode buffer from the bank; the robot
            # and cuboid are then overwritten with the raw demonstration frame.
            self.inner.reset_all(
                reference_index=self.pregrasp_index, translation_xy=(0.0, 0.0), yaw_rad=0.0
            )
            sample = self._demo_sample()
            self._write_kinematic_pose(sample.q[0].detach().cpu().numpy().astype(np.float64))
            self._place_cube_from_demo()
            self._flush()
            self.inner.episode_initial_object_com_height_m[0] = self.inner.cube_position[0, 2]
            q = self.inner.q[0].detach().cpu().numpy().astype(np.float64)
            self.cube_wxyz_reset = quaternion_xyzw_to_wxyz(
                self.inner.cube_orientation[0].detach().cpu().numpy().astype(np.float64)
            )
        self.physics_button.label = "Start physics"
        self.physics_button.color = "green"
        self.physics_status.content = "**Physics:** stopped (sliders move the joints kinematically)."
        self._set_sliders(q)
        self._publish()

    def reset_cube(self) -> None:
        """Put the cuboid back on the table at the base placement, robot untouched."""
        with self.lock:
            inner = self.inner
            self._place_cube_from_demo()
            self._flush()
            self.cube_wxyz_reset = quaternion_xyzw_to_wxyz(
                inner.cube_orientation[0].detach().cpu().numpy().astype(np.float64)
            )
        self._publish()

    # ---------------------------------------------------------------- physics

    def _write_kinematic_pose(self, q: np.ndarray) -> None:
        inner = self.inner
        ids = torch.zeros(1, dtype=torch.int32, device=inner.device)
        position = torch.as_tensor(q, dtype=torch.float32, device=inner.device).unsqueeze(0)
        velocity = torch.zeros_like(position)
        inner.robot.write_joint_state_to_sim_index(position=position.contiguous(), velocity=velocity, env_ids=ids)
        inner.position_targets[0] = position[0]
        inner.applied_targets[0] = position[0]
        inner.robot.actuators.target_command.set_position_index(value=position.contiguous(), env_ids=ids)
        self._flush()

    def _flush(self) -> None:
        inner = self.inner
        inner.scene.write_data_to_sim()
        inner.sim.forward()
        inner.scene.update(dt=self.physics_dt)

    def _physics_step(self) -> None:
        """One control step: drives track the sliders, physics integrates."""
        inner = self.inner
        q = self._targets()
        inner.position_targets[0] = torch.as_tensor(q, dtype=torch.float32, device=inner.device)
        for _ in range(self.decimation):
            inner._apply_action()
            inner.scene.write_data_to_sim()
            inner.sim.step(render=False)
            inner.scene.update(dt=self.physics_dt)
        inner._clamp_object_velocity()
        self.steps += 1

    # ---------------------------------------------------------------- readout

    def _measure(self) -> Dict[str, object]:
        inner = self.inner
        with torch.inference_mode():
            tips = inner._fingertip_positions_world()
            cube_position = inner.cube_position
            cube_orientation = inner.canonical_cube_orientation()
            tips_cube = _quat_rotate_inverse(
                cube_orientation.unsqueeze(1).expand(-1, tips.shape[1], -1), tips - cube_position.unsqueeze(1)
            )
            _, _, per_finger = fingertip_cuboid_proximity(
                tips_cube, inner.object_half_extents, 0.02, torch.ones(1, dtype=torch.bool, device=inner.device)
            )
            q = inner.q[0]
            forces = None
            if getattr(inner, "contact_enabled", False):
                inner._refresh_contact_forces()
                forces = fingertip_force_norms(
                    inner.net_contact_forces, inner.fingertip_body_indices.new_tensor(range(5))
                )[0]
        return {
            "q": q.detach().cpu().numpy().astype(np.float64),
            "cube_position": cube_position[0].detach().cpu().numpy().astype(np.float64),
            "cube_wxyz": quaternion_xyzw_to_wxyz(inner.cube_orientation[0].detach().cpu().numpy().astype(np.float64)),
            "tip_distance": per_finger[0].detach().cpu().numpy().astype(np.float64),
            "forces": None if forces is None else forces.detach().cpu().numpy().astype(np.float64),
            "lift": float(cube_position[0, 2] - inner.episode_initial_object_com_height_m[0]),
        }

    def _publish(self) -> None:
        m = self._measure()
        self.robot.update_cfg(m["q"])
        self.cube.position = m["cube_position"]
        self.cube.wxyz = m["cube_wxyz"]
        # How far the bar has rotated since the last reset (any axis).
        dot = abs(float(np.dot(m["cube_wxyz"], self.cube_wxyz_reset)))
        roll = 2.0 * np.arccos(min(1.0, dot))
        distances = "  ".join(
            "{} {:.1f} mm".format(name, 1000.0 * d) for name, d in zip(FINGER_NAMES, m["tip_distance"])
        )
        force_line = ""
        if m["forces"] is not None:
            force_line = "**Fingertip forces [N]:** " + "  ".join(
                "{} {:.2f}".format(name, f) for name, f in zip(FINGER_NAMES, m["forces"])
            ) + "  \n"
        beyond = [
            name
            for name, q, lo, hi in zip(
                list(ARM_JOINT_NAMES) + self._hand_labels(), m["q"], self.lower, self.upper
            )
            if q < lo - 0.02 or q > hi + 0.02
        ]
        self.readout.content = (
            "**Steps simulated:** {}  \n"
            "**Cuboid:** x {:+.3f} y {:+.3f} z {:.3f} m, lift {:+.3f} m, rotated {:.1f}° since reset  \n"
            "**Fingertip to surface:** {}  \n"
            "{}"
            "**Joints beyond URDF limits:** {}".format(
                self.steps,
                m["cube_position"][0],
                m["cube_position"][1],
                m["cube_position"][2],
                m["lift"],
                np.rad2deg(roll),
                distances,
                force_line,
                ", ".join(beyond) if beyond else "none",
            )
        )

    def _save_pose(self, _) -> None:
        m = self._measure()
        folder = REPO_ROOT / "logs" / "grasp_lab"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / time.strftime("pose_%Y%m%d_%H%M%S.json")
        payload = {
            "config": str(self.config_path),
            "overrides": list(self.args.overrides) + list(self.gui_overrides),
            "demo": str(self.demo_path),
            "pregrasp_index": self.pregrasp_index,
            "joint_names": list(ARM_JOINT_NAMES) + list(HAND_JOINT_NAMES),
            "q": m["q"].tolist(),
            "targets": self._targets().tolist(),
            "cube_position": m["cube_position"].tolist(),
            "cube_wxyz": m["cube_wxyz"].tolist(),
            "cube_lift_m": m["lift"],
            "fingertip_surface_distance_m": m["tip_distance"].tolist(),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("[grasp-lab] pose saved to {}".format(path), flush=True)
        self.save_note.content = "Saved `{}`".format(path.relative_to(REPO_ROOT))

    # -------------------------------------------------------------------- loop

    def run(self) -> None:
        print("Open http://localhost:{}".format(self.port), flush=True)
        next_tick = time.monotonic()
        try:
            while True:
                with self.lock:
                    running = self.physics_running
                if not running:
                    time.sleep(0.02)
                    next_tick = time.monotonic()
                    continue
                with self.lock:
                    self._physics_step()
                    if self.steps % 3 == 0:
                        self._publish()
                next_tick += self.control_dt / max(float(self.rate.value), 1e-3)
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        except KeyboardInterrupt:
            print("Grasp lab stopped.")
        finally:
            self.env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="A run directory or its config.json; the physics is taken from it (default: %(default)s).",
    )
    parser.add_argument(
        "--pregrasp-index",
        type=int,
        default=DEFAULT_PREGRASP_INDEX,
        help="Demonstration frame the scene starts from (default: %(default)s, fingers already around the bar).",
    )
    parser.add_argument(
        "--demo",
        type=Path,
        default=None,
        help="Demonstration .npz to take the start pose from (default: the run's motion.file).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override an env config value, e.g. --set control.hand_stiffness_scale=0.5 (repeatable).",
    )
    parser.add_argument("--sim-device", default=None)
    parser.add_argument(
        "--physics", default=None, choices=["newton_mjwarp", "physx", "ovphysx", "isaacsim_physx"]
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--port", type=int, default=8082)
    return parser.parse_args()


def main() -> None:
    GraspLab(parse_args()).run()


if __name__ == "__main__":
    main()
