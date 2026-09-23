#!/usr/bin/env python3
"""Deploy a SimToolReal-Newton policy on the UR5e + Tesollo DG5F.

Simulation is the default. Physical arm output and physical hand output are
armed separately and explicitly, so every rung of the commissioning ladder in
``simtoolreal_newton/deployment/README.md`` is a flag change rather than an edit.

The policy contract is not restated here: ``build_observation`` and
``ActionPipeline`` come from ``simtoolreal_newton.deployment.contract``, which
calls the environment's own kinematics, Jacobian, IK step and rotation helpers.
The arm's six actions are a palm twist; the deployment resolves it to joint
targets with the same damped least-squares step training used, accumulating on
the previously commanded target exactly as the environment does, because those
commanded targets are part of the next observation.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simtoolreal_newton.deployment import (  # noqa: E402
    ACTION_DIM,
    ARM_DOF,
    HAND_DOF,
    ActionPipeline,
    ArmClient,
    ArmClientError,
    CubeSourceError,
    DeploymentRun,
    DeploymentViewer,
    FrozenCube,
    HandClient,
    HandClientError,
    ObservationInputs,
    PoseEstimationCube,
    ReferenceCube,
    SafetyAbort,
    SpikeMonitor,
    TargetLimiter,
    ViewerUnavailable,
    build_observation,
    confirm_send,
    wait_for_key,
)
from simtoolreal_newton.deployment.hand_stiffness import (  # noqa: E402
    HandStiffnessError,
    format_gains,
    read_hand_gains,
    set_hand_stiffness,
)
from simtoolreal_newton.envs.controller import (  # noqa: E402
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    JOINT_NAMES,
)

DEFAULT_CONTROL_HZ = 60.0
DEFAULT_ARM_CONFIG = Path("/home/duplo/simone/SimToolReal/deployment/simtoolreal_real/pc_ur_new.json")


def smoothstep01(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


# ---------------------------------------------------------------- contract --
def load_run_or_explain(checkpoint: Path, config) -> DeploymentRun:
    try:
        return DeploymentRun(checkpoint, config)
    except ValueError as error:
        raise SystemExit(
            "\n".join(
                (
                    "",
                    "Cannot deploy this checkpoint: {}".format(error),
                    "",
                    "The checkpoint and simtoolreal_newton.deployment.contract disagree. A policy",
                    "must be deployed against the observation and action contract it was trained",
                    "on; there is no adapter between contracts. Check env_cfg.env.num_observations,",
                    "env_cfg.control.action_parameterization and env_cfg.contact in its config.json,",
                    "and run scripts/check_deployment_contract.py against it.",
                    "",
                )
            )
        )


# ------------------------------------------------------------------- state --
class SimulatedSide:
    """Stand-in state for a subsystem that is not being read from hardware.

    ``reference`` replays the bank clip, including its recorded velocities: it
    reproduces the observation the policy was trained against and is the
    default for that reason.

    ``target`` instead assumes the subsystem tracks the policy's own command.
    That has no physics behind it, so a joint would otherwise traverse the whole
    step within one control period and report an implied velocity of tens of
    rad/s -- far outside anything training ever showed the network. The implied
    velocity is therefore clamped, and the position advances only as far as
    that clamp allows.
    """

    def __init__(self, source: str, initial_q: np.ndarray, control_dt: float, max_velocity_rad_s: float) -> None:
        self.source = source
        self.q = np.asarray(initial_q, dtype=np.float64).copy()
        self.dq = np.zeros_like(self.q)
        self.control_dt = float(control_dt)
        self.max_velocity = float(max_velocity_rad_s)

    def update(self, commanded: np.ndarray, reference_q: np.ndarray, reference_dq: np.ndarray) -> None:
        if self.source == "reference":
            self.q = np.asarray(reference_q, dtype=np.float64).copy()
            self.dq = np.asarray(reference_dq, dtype=np.float64).copy()
            return
        requested = np.asarray(commanded, dtype=np.float64)
        velocity = np.clip((requested - self.q) / self.control_dt, -self.max_velocity, self.max_velocity)
        self.q = self.q + velocity * self.control_dt
        self.dq = velocity


def format_joint_list(indices, names) -> str:
    return ", ".join(str(names[int(i)]) for i in indices)


def quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation taking ``b`` to ``a``, in degrees (sign-agnostic)."""
    dot = float(np.clip(abs(float(np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


# --------------------------------------------------------------- placement --
def resolve_transform_index(args, run: DeploymentRun, live_cube) -> int:
    """Which bank clip this run plays: named, the demonstration's own, or the live cuboid's."""
    if args.bank_index != "nearest":
        index = int(args.bank_index)
        if not 0 <= index < run.bank.transform_count:
            raise SystemExit("--bank-index must lie in [0, {}]".format(run.bank.transform_count - 1))
        return index
    if live_cube is None:
        return run.default_transform_index()

    pose, _, _ = live_cube.cube_state(args.rsi_index)
    placement = run.placement_from_cube_pose(pose)
    index, residual_xy, residual_yaw = run.nearest_transform_index(placement)
    translation = placement.translation.cpu().numpy()
    print()
    print("Live cuboid placement (relative to the demonstration's start pose):")
    print("  translation x/y/z : {} m".format(np.round(translation, 4).tolist()))
    print("  yaw               : {:+.1f} deg   tilt off the table: {:.1f} deg".format(
        math.degrees(placement.yaw_rad), math.degrees(placement.tilt_rad)
    ))
    print("  nearest bank entry: {} (translation {}, yaw {:+.1f} deg)".format(
        index,
        np.round(run.bank.translation[index].cpu().numpy(), 4).tolist(),
        math.degrees(float(run.bank.yaw_rad[index])),
    ))
    print("  residual          : {:.1f} mm, {:+.1f} deg".format(1e3 * residual_xy, math.degrees(residual_yaw)))
    if placement.tilt_rad > math.radians(args.max_placement_tilt_deg):
        raise SafetyAbort(
            "The cuboid is tilted {:.1f} deg off the table (limit {:.1f}). Training only ever "
            "placed it flat.".format(math.degrees(placement.tilt_rad), args.max_placement_tilt_deg)
        )
    if residual_xy > args.max_placement_residual_m or abs(residual_yaw) > math.radians(args.max_placement_residual_deg):
        raise SafetyAbort(
            "The cuboid stands {:.1f} mm / {:+.1f} deg from the nearest bank entry (limits "
            "{:.1f} mm / {:.1f} deg). Training placed it exactly on a bank entry; move it "
            "closer or raise --max-placement-residual-*.".format(
                1e3 * residual_xy,
                math.degrees(residual_yaw),
                1e3 * args.max_placement_residual_m,
                args.max_placement_residual_deg,
            )
        )
    return index


def check_cube_frame(args, run: DeploymentRun, live_cube, transform_index: int) -> int:
    """Compare a live estimator pose with the bank clip's pose at an index."""
    live_pose, _, _ = live_cube.cube_state(args.rsi_index)
    reference_pose = run.reference_cube_pose(transform_index, args.rsi_index, args.object_scale)
    print()
    print("Cuboid frame check at reference index {} against bank entry {}".format(args.rsi_index, transform_index))
    print("  reference position : {}".format(np.round(reference_pose[:3], 4).tolist()))
    print("  estimator position : {}".format(np.round(live_pose[:3], 4).tolist()))
    print("  difference         : {}".format(np.round(live_pose[:3] - reference_pose[:3], 4).tolist()))
    print("  reference quat xyzw: {}".format(np.round(reference_pose[3:], 4).tolist()))
    print("  estimator quat xyzw: {}".format(np.round(live_pose[3:], 4).tolist()))
    print()
    print(
        "Place the real cuboid where the clip has it at this index. A residual of a "
        "few millimetres is calibration error; a sign flip or a swapped axis means the "
        "estimator frame is NOT the demonstration frame, and --cube-source "
        "pose-estimation must not be used until that is resolved."
    )
    return 0


# -------------------------------------------------------------------- main --
def main() -> int:
    args = parse_args()

    run = load_run_or_explain(args.checkpoint, args.config)
    if not 0 <= args.rsi_index < run.last_index:
        raise SystemExit("--rsi-index must lie in [0, {}]".format(run.last_index - 1))
    low, high = run.object_scale_range
    if run.observes_scale and not low - 1e-9 <= args.object_scale <= high + 1e-9:
        print("WARNING: --object-scale {:g} lies outside the trained range [{:g}, {:g}].".format(
            args.object_scale, low, high
        ))
    if not run.observes_scale and abs(args.object_scale - 1.0) > 1e-9:
        print("WARNING: this checkpoint does not observe the bar's scale; --object-scale only lifts the reference.")
    control_dt = 1.0 / args.control_hz
    if args.max_hand_reference_error_rad is None:
        args.max_hand_reference_error_rad = run.hand_position_threshold_rad

    if args.spike_mode is None:
        args.spike_mode = "warn" if args.debug_step else "stop"

    send_arm = bool(args.send_to_arm)
    send_hand = bool(args.send_to_hand)
    # Commanding a subsystem without reading it would close the loop on a
    # fiction, so hardware output implies hardware state.
    use_arm_state = bool(args.use_real_arm_state) or send_arm
    use_hand_state = bool(args.use_real_hand_state) or send_hand

    live_cube = None
    cube = None
    arm_client = None
    hand_client = None
    viewer = None
    exit_code = 0
    try:
        if args.cube_source == "pose-estimation":
            live_cube = PoseEstimationCube(
                args.pose_address,
                board_id=args.pose_board_id,
                minimum_confidence=args.pose_min_confidence,
                pose_timeout=args.pose_timeout,
                z_offset_m=args.pose_z_offset_m,
                median_window=args.pose_median_window,
                jump_reject_m=args.pose_jump_reject_m,
                jump_accept_samples=args.pose_jump_accept_samples,
                position_filter_alpha=args.pose_filter_alpha,
            )
            live_cube.wait_for_pose(timeout=args.pose_wait_seconds)
            print("Cuboid pose stream is live on {}".format(args.pose_address))
        elif args.check_cube_frame:
            raise SystemExit("--check-cube-frame needs --cube-source pose-estimation")

        transform_index = resolve_transform_index(args, run, live_cube)
        if args.check_cube_frame:
            return check_cube_frame(args, run, live_cube, transform_index)

        if args.cube_source == "reference":
            cube = ReferenceCube(run, transform_index, args.object_scale)
        elif args.cube_source == "frozen":
            cube = FrozenCube(run, transform_index, args.rsi_index, args.object_scale)
        else:
            cube = live_cube

        start_q = run.reference_q(transform_index, args.rsi_index)
        reference_cube = run.reference_cube_pose(transform_index, args.rsi_index, args.object_scale)
        # The bar's symmetry label is chosen once, against the pose the episode
        # starts at, and held -- see envs/cuboid_symmetry.py for why not per step.
        initial_pose, _, _ = cube.cube_state(args.rsi_index)
        symmetry_index = run.choose_symmetry_index(
            run.cube_pose_base(initial_pose), torch.as_tensor(reference_cube, dtype=torch.float32)
        )

        if use_arm_state:
            arm_client = ArmClient(
                args.arm_config,
                stream_hz=args.arm_stream_hz,
                state_timeout=args.state_timeout,
                connect_settle_seconds=args.arm_connect_settle_seconds,
                velocity_source=args.arm_velocity_source,
            )
            arm_client.wait_for_state(timeout=args.state_wait_seconds)
            print("UR5 state: q_deg={}".format(np.rad2deg(arm_client.positions).round(2).tolist()))
        if use_hand_state:
            hand_client = HandClient(
                bind_address=args.hand_bind_address,
                state_port=args.hand_state_port,
                command_address=args.hand_command_address,
                command_port=args.hand_command_port,
                state_timeout=args.state_timeout,
            )
            hand_client.wait_for_state(timeout=args.state_wait_seconds)
            print("DG5F state: max|q|={:.3f} rad".format(float(np.max(np.abs(hand_client.positions)))))
            # Homing needs authority, the rollout wants compliance: with i = 0 a
            # soft hand cannot close the last tenth of a radian (p = 0.5 turns a
            # 0.19 rad error into 9.5% duty, below the joints' stiction), so home
            # at the higher gain and drop to the run gain once it has arrived.
            homing_stiffness = args.hand_home_stiffness if send_hand else None
            initial_stiffness = homing_stiffness if homing_stiffness is not None else args.hand_stiffness
            if initial_stiffness is not None:
                gains = set_hand_stiffness(initial_stiffness)
                print("DG5F stiffness set: {}{}".format(
                    format_gains(gains), " (for homing)" if homing_stiffness is not None else ""
                ))
            else:
                print("DG5F stiffness now: {} (pass --hand-stiffness to change it)".format(
                    format_gains(read_hand_gains())
                ))

        if not args.no_viewer:
            try:
                viewer = DeploymentViewer(
                    run,
                    ghost=not args.no_ghost,
                    object_scale=args.object_scale,
                    update_hz=args.viewer_hz,
                )
                print("MuJoCo viewer open (closing the window stops the run; --no-viewer skips it).")
            except ViewerUnavailable as error:
                # Losing the picture must not strand a hardware session.
                print("WARNING: the viewer could not open ({}); continuing without it.".format(error))

        print_banner(args, run, transform_index, symmetry_index, send_arm, send_hand)

        if send_arm or send_hand:
            outputs = []
            if send_arm:
                outputs.append("UR5e arm  -> {}".format(args.arm_config))
            if send_hand:
                outputs.append("DG5F hand -> udp://{}:{}".format(args.hand_command_address, args.hand_command_port))
            confirm_send(outputs)

        # Homing runs through operator keypresses and can take tens of seconds.
        # Keep the cuboid stream drained across all of it, so an estimator that
        # dies in that window is caught here -- before the robot has been homed
        # and both outputs armed -- rather than on the first control step.
        cube_keepalive = live_cube.poll if live_cube is not None else None

        if send_arm:
            home_arm(arm_client, start_q[:ARM_DOF], args, on_wait=cube_keepalive)
        if send_hand:
            home_hand(hand_client, start_q[ARM_DOF:], args, on_wait=cube_keepalive)
            if args.hand_home_stiffness is not None:
                gains = set_hand_stiffness(args.hand_stiffness)
                print("DG5F stiffness lowered for the rollout: {}".format(format_gains(gains)))
                # The hand sags toward its new equilibrium; hold the start pose
                # while it settles so the first observation is not mid-transient.
                settle_until = time.monotonic() + args.hand_home_settle_seconds
                while time.monotonic() < settle_until:
                    hand_client.send_target(start_q[ARM_DOF:])
                    time.sleep(args.hand_home_step_seconds)
                measured = hand_client.require_fresh_state()
                print("  after softening, max|q-target| = {:.4f} rad".format(
                    float(np.max(np.abs(start_q[ARM_DOF:] - measured)))
                ))

        if live_cube is not None:
            # The last gate before the policy commands anything: prove the
            # estimator survived homing. Without this the first control step
            # reports an age measured from start-up, which says nothing about
            # when the stream actually stopped.
            live_cube.poll()
            live_cube.cube_state(args.rsi_index)

        exit_code = run_policy(
            args=args,
            run=run,
            cube=cube,
            transform_index=transform_index,
            symmetry_index=symmetry_index,
            arm_client=arm_client,
            hand_client=hand_client,
            viewer=viewer,
            start_q=start_q,
            control_dt=control_dt,
            send_arm=send_arm,
            send_hand=send_hand,
            use_arm_state=use_arm_state,
            use_hand_state=use_hand_state,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        exit_code = 130
    except (SafetyAbort, ArmClientError, HandClientError, CubeSourceError, HandStiffnessError) as error:
        print("\nSAFETY STOP: {}".format(error))
        exit_code = 2
    finally:
        if arm_client is not None:
            if send_arm:
                print("Braking the arm (zero-velocity hold)...")
                arm_client.brake(duration_s=args.brake_seconds)
            arm_client.close()
        if hand_client is not None:
            if send_hand:
                print("Holding the hand at its measured position...")
                hand_client.hold_measured(duration_s=args.brake_seconds)
            hand_client.close()
        if live_cube is not None:
            # Printed on the abort path too: these counters are exactly what
            # tells a dead estimator apart from a gate that is set too high.
            print(live_cube.statistics())
            live_cube.close()
        if viewer is not None:
            if args.viewer_hold_seconds > 0.0 and viewer.is_running():
                # After an abort the last drawn frame is the one that tripped
                # the monitor, and it is worth looking at.
                print("Viewer holding the final frame for {:g} s.".format(args.viewer_hold_seconds))
                time.sleep(args.viewer_hold_seconds)
            viewer.close()
    return exit_code


def print_banner(args, run: DeploymentRun, transform_index: int, symmetry_index: int, send_arm: bool, send_hand: bool) -> None:
    print()
    print("=" * 70)
    print("SimToolReal-Newton real-robot deployment")
    print("  checkpoint      : {}".format(run.checkpoint_path))
    print("  observation/act : {}/{}".format(run.observation_dim, ACTION_DIM))
    print("  demonstration   : {} ({} samples)".format(run.reference.path.name, run.reference.sample_count))
    print("  bank clip       : {} of {} (translation {}, yaw {:+.1f} deg), symmetry {}".format(
        transform_index,
        run.bank.transform_count,
        np.round(run.bank.translation[transform_index].cpu().numpy(), 4).tolist(),
        math.degrees(float(run.bank.yaw_rad[transform_index])),
        symmetry_index,
    ))
    print("  start index     : {}".format(args.rsi_index))
    print("  bar scale       : {:g} ({}; trained on [{:g}, {:g}]; reference lifted {:+.1f} mm)".format(
        args.object_scale,
        "observed by the policy" if run.observes_scale else "not observed",
        run.object_scale_range[0],
        run.object_scale_range[1],
        1e3 * run.reference_height_shift(args.object_scale),
    ))
    print("  control rate    : {:g} Hz (trained at {:g} Hz)".format(args.control_hz, 1.0 / run.dt))
    print("  action filter   : alpha {:g}".format(run.action_filter_alpha))
    print("  arm twist       : <= {:g} m/s, {:g} rad/s; IK damping {:g}, <= {:g} rad/step; scale {:g}".format(
        run.arm_translation_speed, run.arm_rotation_speed, run.ik_damping, run.ik_max_joint_delta, args.arm_action_scale
    ))
    print("  hand residual   : {:g} rad/unit, clip {:g}; scale {:g}".format(
        run.hand_action_scale, run.action_target_clip, args.hand_action_scale
    ))
    print("  hand stiffness  : {}{} (driver YAML default 2.0; training's soft hand ~5 N m/rad)".format(
        "p = {:g}".format(args.hand_stiffness) if args.hand_stiffness is not None else "unchanged",
        ", homing at p = {:g}".format(args.hand_home_stiffness) if args.hand_home_stiffness is not None else "",
    ))
    print("  step limit      : arm {:g} rad, hand {:g} rad".format(args.max_arm_step_rad, args.max_hand_step_rad))
    print("  target smoothing: {:g}".format(args.target_smoothing))
    print("  startup policy  : {:g} demo-s reference-to-policy blend".format(args.startup_policy_blend_seconds))
    print("  startup target  : {:g} s home-to-target ramp".format(args.startup_ramp_seconds))
    print("  cube source     : {}".format(args.cube_source))
    if args.cube_source == "pose-estimation":
        print("  cube filtering  : median over {} samples, reject jumps > {:g} m unless {} in a row{}{}".format(
            args.pose_median_window,
            args.pose_jump_reject_m,
            args.pose_jump_accept_samples,
            ", confidence >= {:g}".format(args.pose_min_confidence) if args.pose_min_confidence > 0.0 else "",
            ", position low-pass {:g}".format(args.pose_filter_alpha) if args.pose_filter_alpha > 0.0 else "",
        ))
        if args.pose_median_window <= 1:
            print("  WARNING: no median filter -- the estimator's 16-22 mm face-switching")
            print("           excursions reach the policy unfiltered. --pose-median-window 5.")
        if args.pose_min_confidence > 0.0:
            print("  WARNING: an absolute confidence gate falls with occlusion and can starve the")
            print("           stream during the approach. Prefer the median filter for a rollout.")
    if args.commission_arm_only_ideal_context:
        print("  hybrid context  : real arm + ideal hand/cube + URDF kinematics")
    elif args.commission_hand_only_ideal_context:
        print("  hybrid context  : real hand + ideal arm/cube + URDF kinematics")
    else:
        print("  simulated state : {}".format(args.simulated_state_source))
    print("  arm velocity    : {}".format(args.arm_velocity_source))
    print("  arm output      : {}".format("ARMED" if send_arm else "simulated"))
    print("  hand output     : {}".format("ARMED" if send_hand else "simulated"))
    if getattr(args, "low_safety", False):
        print("  " + "!" * 66)
        print("  LOW SAFETY (demo): reference-tracking monitors raised so the bar")
        print("  can be moved by hand mid-rollout. Motion limits are UNCHANGED.")
        for change in getattr(args, "low_safety_changes", []):
            print("    {}".format(change))
        print("  " + "!" * 66)
    print("  action spike    : {} above arm {:g} / hand {:g}{}".format(
        args.spike_mode, args.max_action_step, args.max_hand_action_step,
        " (stepping by hand; spikes are an artefact of the wait)" if args.debug_step else "",
    ))
    print("  ref deviation   : {} above arm {:g} rad / hand {:g} rad / palm {:g} m".format(
        args.reference_error_mode,
        args.max_arm_reference_error_rad,
        args.max_hand_reference_error_rad,
        args.max_palm_reference_error_m,
    ))
    print("=" * 70)


def home_arm(arm_client, target_q, args, on_wait=None) -> None:
    measured = arm_client.require_fresh_state()
    distance = float(np.max(np.abs(target_q - measured)))
    print()
    print("Arm homing to the start pose:")
    print("  measured q_deg: {}".format(np.rad2deg(measured).round(2).tolist()))
    print("  target   q_deg: {}".format(np.rad2deg(target_q).round(2).tolist()))
    print("  largest joint move: {:.3f} rad ({:.1f} deg)".format(distance, np.rad2deg(distance)))
    if distance > args.max_home_distance_rad:
        raise SafetyAbort(
            "Homing move of {:.3f} rad exceeds --max-home-distance-rad {:.3f}. "
            "Jog the arm closer to the start pose by hand first.".format(distance, args.max_home_distance_rad)
        )
    # Every stream this process polls itself has to be drained while the
    # operator inspects the pose; otherwise a deliberate pause at this prompt
    # makes a healthy stream appear stale immediately afterward. That goes for
    # the cuboid estimator as much as for the arm state -- homing through two
    # keypresses can easily take half a minute.
    def keepalive():
        arm_client.poll()
        if on_wait is not None:
            on_wait()

    wait_for_key("Press Space to send the homing trajectory, or q to abort: ", on_wait=keepalive)
    seconds = max(args.home_seconds, distance / max(args.home_speed_rad_s, 1e-6))
    # The controller's spline needs at least three points.
    midpoint = 0.5 * (measured + target_q)
    arm_client.send_trajectory(np.asarray([0.0, 0.5 * seconds, seconds]), np.stack([measured, midpoint, target_q]))
    deadline = time.monotonic() + seconds + args.home_settle_seconds
    while time.monotonic() < deadline:
        arm_client.poll()
        time.sleep(0.02)
    measured = arm_client.require_fresh_state()
    error = float(np.max(np.abs(target_q - measured)))
    print("  homing finished, max|q-target| = {:.4f} rad".format(error))
    if error > args.home_tolerance_rad:
        raise SafetyAbort(
            "Arm did not reach the start pose (error {:.4f} rad > tolerance {:.4f} rad).".format(
                error, args.home_tolerance_rad
            )
        )
    arm_client.start_streaming()
    arm_client.set_target(measured)


def home_hand(hand_client, target_q, args, on_wait=None) -> None:
    measured = hand_client.require_fresh_state()
    distance = float(np.max(np.abs(target_q - measured)))
    print()
    print("Hand homing to the start pose:")
    print("  largest joint move: {:.3f} rad".format(distance))
    wait_for_key("Press Space to ramp the hand to the start pose, or q to abort: ", on_wait=on_wait)
    steps = max(1, int(np.ceil(distance / args.hand_home_step_rad)))
    for index in range(1, steps + 1):
        blend = index / steps
        hand_client.send_target((1.0 - blend) * measured + blend * target_q)
        time.sleep(args.hand_home_step_seconds)
    for _ in range(int(args.home_settle_seconds / max(args.hand_home_step_seconds, 1e-3))):
        hand_client.send_target(target_q)
        time.sleep(args.hand_home_step_seconds)
    settle_deadline = time.monotonic() + args.hand_home_timeout_seconds
    while True:
        measured = hand_client.require_fresh_state()
        absolute_error = np.abs(target_q - measured)
        if float(np.max(absolute_error)) <= args.hand_home_tolerance_rad:
            break
        if time.monotonic() >= settle_deadline:
            break
        hand_client.send_target(target_q)
        time.sleep(args.hand_home_step_seconds)
    worst_index = int(np.argmax(absolute_error))
    error = float(absolute_error[worst_index])
    print(
        "  homing finished, max|q-target| = {:.4f} rad on {} (measured={:.4f}, target={:.4f})".format(
            error, HAND_JOINT_NAMES[worst_index], measured[worst_index], target_q[worst_index]
        )
    )
    short = np.flatnonzero(absolute_error > args.hand_home_tolerance_rad)
    if short.size:
        print("  joints still short of the pose: {}".format(
            ", ".join("{} {:+.3f}".format(HAND_JOINT_NAMES[int(i)], float(measured[int(i)] - target_q[int(i)]))
                      for i in short)
        ))
    if error > args.hand_home_tolerance_rad:
        raise SafetyAbort(
            "Hand did not reach the start pose within {:.1f} s: {} remains {:.4f} rad from "
            "target (limit {:.4f} rad). A soft hand (i = 0) keeps a steady-state error: "
            "home at a higher gain with --hand-home-stiffness (e.g. 2.0) and let the runner "
            "drop to --hand-stiffness for the rollout.".format(
                args.hand_home_timeout_seconds, HAND_JOINT_NAMES[worst_index], error, args.hand_home_tolerance_rad
            )
        )


def run_policy(
    *,
    args,
    run: DeploymentRun,
    cube,
    transform_index: int,
    symmetry_index: int,
    arm_client,
    hand_client,
    viewer,
    start_q,
    control_dt,
    send_arm,
    send_hand,
    use_arm_state,
    use_hand_state,
) -> int:
    pipeline = ActionPipeline(run)
    pipeline.reset(start_q)
    limiter = TargetLimiter(
        run.joint_lower_limits.cpu().numpy(),
        run.joint_upper_limits.cpu().numpy(),
        max_arm_step_rad=args.max_arm_step_rad,
        max_hand_step_rad=args.max_hand_step_rad,
        smoothing=args.target_smoothing,
    )
    limiter.reset(start_q)
    if send_arm and not send_hand:
        spike_slice = slice(0, ARM_DOF)
        spike_labels = list(ARM_JOINT_NAMES)
    elif send_hand and not send_arm:
        spike_slice = slice(ARM_DOF, ACTION_DIM)
        spike_labels = list(HAND_JOINT_NAMES)
    else:
        spike_slice = slice(None)
        spike_labels = list(JOINT_NAMES)
    # The hand carries its own limit: every spike seen on the robot so far has
    # been a finger reacting to the real hand's lag behind its target, which the
    # arm has no equivalent of.
    spike_thresholds = np.concatenate((
        np.full(ARM_DOF, float(args.max_action_step)),
        np.full(ACTION_DIM - ARM_DOF, float(args.max_hand_action_step)),
    ))[spike_slice]
    action_spikes = SpikeMonitor(
        spike_thresholds, mode=args.spike_mode, name="action", labels=spike_labels, grace_steps=args.spike_grace_steps
    )
    simulated_arm = SimulatedSide(args.simulated_state_source, start_q[:ARM_DOF], control_dt, args.simulated_max_velocity_rad_s)
    simulated_hand = SimulatedSide(args.simulated_state_source, start_q[ARM_DOF:], control_dt, args.simulated_max_velocity_rad_s)

    previous_targets = start_q.copy()
    reference_index = int(args.rsi_index)
    steps = 0
    started_at = time.perf_counter()
    overruns = 0
    if use_arm_state:
        arm_client.poll()
    if use_hand_state:
        hand_client.poll()

    print()
    print("Running. Ctrl+C stops and brakes.")
    while reference_index < run.last_index:
        if args.max_steps and steps >= args.max_steps:
            break
        if viewer is not None and not viewer.is_running():
            print("Viewer window closed; stopping.")
            break
        loop_started = time.perf_counter()

        # -- measured state ------------------------------------------------
        if use_arm_state:
            arm_q = arm_client.require_fresh_state()
            arm_dq = arm_client.velocities.copy()
        else:
            arm_q, arm_dq = simulated_arm.q.copy(), simulated_arm.dq.copy()
        if use_hand_state:
            hand_q = hand_client.require_fresh_state()
            hand_dq = hand_client.velocities.copy()
        else:
            hand_q, hand_dq = simulated_hand.q.copy(), simulated_hand.dq.copy()
        measured_q = np.concatenate((arm_q, hand_q))
        measured_dq = np.concatenate((arm_dq, hand_dq))

        # -- observation ---------------------------------------------------
        cube_pose, _, _ = cube.cube_state(reference_index)
        observation = build_observation(
            run,
            ObservationInputs(
                joint_positions=measured_q,
                joint_velocities=measured_dq,
                previous_targets=previous_targets,
                reference_index=reference_index,
                cube_pose_base=run.cube_pose_base(cube_pose),
                symmetry_index=symmetry_index,
                object_scale=args.object_scale,
            ),
        )

        # -- policy --------------------------------------------------------
        actions = run.act(observation)
        if args.startup_policy_blend_seconds > 0.0:
            # Measured in demonstration time, so a slowed commissioning run
            # hands over just as gradually per policy frame.
            progress = steps / (args.startup_policy_blend_seconds * run.motion_frequency_hz)
            if progress < 1.0:
                reference_actions = run.next_reference_action(pipeline.previous_arm_targets, transform_index, reference_index)
                weight = smoothstep01(progress)
                actions = (1.0 - weight) * reference_actions + weight * actions
        # A commissioning mode deliberately leaves one subsystem disconnected.
        # Its policy outputs remain useful diagnostics, but must not stop the
        # physically armed subsystem. When both are armed, monitor all 26.
        action_spikes.update(actions[spike_slice])

        # -- targets -------------------------------------------------------
        commanded_targets = pipeline.command(actions, measured_q[:ARM_DOF], args.arm_action_scale, args.hand_action_scale)
        raw_targets = pipeline.apply()
        if args.startup_ramp_seconds > 0.0:
            startup_progress = (steps + 1) * control_dt / args.startup_ramp_seconds
            if startup_progress < 1.0:
                startup_weight = smoothstep01(startup_progress)
                raw_targets = (1.0 - startup_weight) * start_q + startup_weight * raw_targets
        applied_targets, limit_info = limiter.apply(raw_targets)

        # -- output --------------------------------------------------------
        if send_arm:
            arm_client.set_target(applied_targets[:ARM_DOF])
        if send_hand:
            hand_client.send_target(applied_targets[ARM_DOF:])

        next_sample = run.reference_sample(transform_index, min(reference_index + 1, run.last_index))
        reference_q = next_sample.q[0].cpu().numpy().astype(np.float64)
        reference_dq = next_sample.dq[0].cpu().numpy().astype(np.float64)
        if not use_arm_state:
            simulated_arm.update(applied_targets[:ARM_DOF], reference_q[:ARM_DOF], reference_dq[:ARM_DOF])
        if not use_hand_state:
            simulated_hand.update(applied_targets[ARM_DOF:], reference_q[ARM_DOF:], reference_dq[ARM_DOF:])

        previous_targets = (commanded_targets if args.previous_target_source == "commanded" else applied_targets).copy()
        reference_index += 1
        steps += 1

        # -- monitors ------------------------------------------------------
        # Deviation is measured against the reference clip, not against the
        # commanded target: a soft-PD finger legitimately sits far from its
        # target. Training terminated on the hand's joint error and on the
        # palm keypoints' distance from the reference bar; the joint-space arm
        # criterion is the earlier runs' one, kept as a plain-language guard.
        arm_error = measured_q[:ARM_DOF] - reference_q[:ARM_DOF]
        hand_error = measured_q[ARM_DOF:] - reference_q[ARM_DOF:]
        arm_worst = int(np.argmax(np.abs(arm_error)))
        hand_worst = int(np.argmax(np.abs(hand_error)))
        arm_deviation = float(abs(arm_error[arm_worst]))
        hand_deviation = float(abs(hand_error[hand_worst]))
        palm_measured, palm_quat = run.palm_pose(measured_q)
        palm_reference, palm_quat_reference = run.palm_pose(reference_q)
        palm_deviation = float(np.linalg.norm(palm_measured - palm_reference))
        # Only the palm's position is a monitor; its orientation is reported so
        # that a wrist drifting under a still palm centre is visible.
        palm_angle_deg = quaternion_angle_deg(palm_quat, palm_quat_reference)
        # Drawn before the monitors may abort, so the window holds the state
        # that stopped the run rather than the one before it.
        if viewer is not None:
            viewer.update(measured_q, reference_q, run.cube_pose_base(cube_pose))
        if args.reference_error_mode != "off":
            for label, deviation, threshold, unit, where in (
                ("Arm", arm_deviation, args.max_arm_reference_error_rad, "rad",
                 " on {} (measured {:+.3f}, reference {:+.3f})".format(
                     ARM_JOINT_NAMES[arm_worst], measured_q[arm_worst], reference_q[arm_worst])),
                ("Hand", hand_deviation, args.max_hand_reference_error_rad, "rad",
                 " on {} (measured {:+.3f}, reference {:+.3f})".format(
                     HAND_JOINT_NAMES[hand_worst],
                     measured_q[ARM_DOF + hand_worst],
                     reference_q[ARM_DOF + hand_worst])),
                ("Palm", palm_deviation, args.max_palm_reference_error_m, "m",
                 " (orientation {:.1f} deg off)".format(palm_angle_deg)),
            ):
                if deviation <= threshold:
                    continue
                message = "{} deviates {:.3f} {} from the reference at step {}{} (limit {:.3f} {}).".format(
                    label, deviation, unit, steps, where, threshold, unit
                )
                if args.reference_error_mode == "stop":
                    raise SafetyAbort(message)
                print("WARNING: " + message)
        if use_hand_state and args.current_warning_ma > 0.0:
            hot = hand_client.over_current_joints(args.current_warning_ma)
            if hot.size:
                print("WARNING: DG5F motor current above {:g} mA on {}".format(
                    args.current_warning_ma, format_joint_list(hot, HAND_JOINT_NAMES)
                ))

        if args.print_every and (steps == 1 or steps % args.print_every == 0):
            tracking_error = float(np.max(np.abs(measured_q - previous_targets)))
            note = ""
            if limit_info["step_limited_joints"].size:
                note = " step-limited:{}".format(format_joint_list(limit_info["step_limited_joints"], JOINT_NAMES))
            if pipeline.last_info.get("arm_joint_delta_clipped"):
                note += " ik-clipped"
            print(
                "step={:4d} ref={:4d} phase={:.3f} max|arm-ref|={:.3f}({}) max|hand-ref|={:.3f} "
                "palm-ref={:.3f}m/{:.1f}deg ik-res={:.4f} max_step={:.4f} track={:.3f}{}".format(
                    steps,
                    reference_index,
                    reference_index / float(run.last_index),
                    arm_deviation,
                    ARM_JOINT_NAMES[arm_worst],
                    hand_deviation,
                    palm_deviation,
                    palm_angle_deg,
                    float(pipeline.last_info.get("ik_residual_norm", 0.0)),
                    limit_info["max_requested_step_rad"],
                    tracking_error,
                    note,
                )
            )

        if args.debug_step:
            hand_keepalive = None
            if send_hand:
                hand_target = applied_targets[ARM_DOF:].copy()

                def hand_keepalive() -> None:
                    hand_client.require_fresh_state()
                    hand_client.send_target(hand_target)

            wait_for_key("  [step {}] Space for the next step, q to stop: ".format(steps), on_wait=hand_keepalive)
        elif not args.no_realtime:
            remaining = control_dt - (time.perf_counter() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -0.5 * control_dt:
                overruns += 1

    elapsed = time.perf_counter() - started_at
    print()
    print("Finished {} steps in {:.1f} s at reference index {}.".format(steps, elapsed, reference_index))
    if action_spikes.worst_index >= 0:
        print(
            "Largest single-step action change: {:.4f} on {} (threshold {:g}, {} detection(s)).".format(
                action_spikes.worst,
                spike_labels[action_spikes.worst_index],
                float(np.broadcast_to(action_spikes.threshold, (len(spike_labels),))[action_spikes.worst_index]),
                action_spikes.detections,
            )
        )
    if overruns and not args.debug_step:
        print("WARNING: {} control steps overran the {:g} Hz budget.".format(overruns, args.control_hz))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    policy = parser.add_argument_group("policy")
    policy.add_argument("--checkpoint", type=Path, required=True)
    policy.add_argument("--config", type=Path, default=None, help="Defaults to config.json beside the checkpoint.")
    policy.add_argument("--rsi-index", type=int, default=0)
    policy.add_argument(
        "--bank-index",
        default="nearest",
        help=(
            "Which transform-bank clip to play: an entry index, or 'nearest' (default) for "
            "the demonstration's own placement -- or, with --cube-source pose-estimation, "
            "the entry nearest the live cuboid."
        ),
    )
    policy.add_argument("--max-steps", type=int, default=0)
    policy.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)

    output = parser.add_argument_group("physical output (off unless given)")
    output.add_argument("--send-to-arm", action="store_true")
    output.add_argument("--send-to-hand", action="store_true")
    output.add_argument("--use-real-arm-state", action="store_true")
    output.add_argument("--use-real-hand-state", action="store_true")
    commissioning = output.add_mutually_exclusive_group()
    commissioning.add_argument(
        "--commission-arm-only-ideal-context",
        action="store_true",
        help="Command/read only the arm; hand q/dq and the cuboid come from the reference clip.",
    )
    commissioning.add_argument(
        "--commission-hand-only-ideal-context",
        action="store_true",
        help="Command/read only the hand; arm q/dq and the cuboid come from the reference clip.",
    )

    scaling = parser.add_argument_group("motion scaling and limits")
    scaling.add_argument("--arm-action-scale", type=float, default=1.0, help="Scales the requested palm twist (1 = training).")
    scaling.add_argument("--hand-action-scale", type=float, default=1.0, help="Scales the hand residual (1 = training).")
    scaling.add_argument(
        "--hand-stiffness", type=float, default=None,
        help=(
            "Load this p gain on every joint of the DG5F position PID (the real hand's whole "
            "stiffness; driver default 2.0). Needs the hand to be read or armed. Unset leaves "
            "the controller's gains alone; scripts/set_hand_stiffness.py changes them at any time."
        ),
    )
    scaling.add_argument(
        "--hand-home-stiffness", type=float, default=None,
        help=(
            "Home the hand at this p gain, then drop to --hand-stiffness for the rollout. "
            "With i = 0 a soft hand keeps a steady-state error and can miss the homing "
            "tolerance; 2.0 (the driver default) reaches the pose. Requires --hand-stiffness."
        ),
    )
    scaling.add_argument("--max-arm-step-rad", type=float, default=0.02)
    scaling.add_argument("--max-hand-step-rad", type=float, default=0.05)
    scaling.add_argument("--target-smoothing", type=float, default=0.0, help="EMA weight on the previous target, in [0, 1). 0 disables it.")
    scaling.add_argument(
        "--startup-ramp-seconds", type=float, default=1.0,
        help="Smoothly ramp targets from the verified home pose to the policy target (default: 1.0 s; 0 disables).",
    )
    scaling.add_argument(
        "--startup-policy-blend-seconds", type=float, default=1.0,
        help=(
            "Start exactly on the reference clip's ideal action and smoothly transfer to the "
            "learned action over this many seconds of demonstration frames (default: 1.0; 0 disables)."
        ),
    )

    monitors = parser.add_argument_group("safety monitors")
    monitors.add_argument(
        "--max-action-step", type=float, default=1.8,
        help="Largest tolerated single-step change in a raw ARM action (the hand has its own, "
             "--max-hand-action-step). For scale, the whole ideal rollout on this checkpoint peaks "
             "at 0.3845, so 1.8 is about five times the nominal worst case.",
    )
    monitors.add_argument(
        "--max-hand-action-step", type=float, default=2.2,
        help="The same limit for the FINGER actions, which need more room: every spike seen on the "
             "robot has been a finger, because the policy observes measured hand joints that lag "
             "their targets by around 0.16 rad, and that offset alone is worth 1.14 of action. The "
             "arm has no equivalent, so it keeps the tighter --max-action-step.",
    )
    monitors.add_argument(
        "--spike-mode", choices=("off", "warn", "stop"), default=None,
        help="Default: 'stop', but 'warn' under --debug-step. Stepping by hand leaves seconds "
             "between policy steps, and the soft hand keeps creeping toward its target the whole "
             "time, so the observation advances far more than one control period's worth and the "
             "action follows -- a measured 0.16 rad of hand sag is worth 1.14 of action. That is "
             "an artefact of stepping, not of the robot, so it must not abort the run. Pass this "
             "flag explicitly to override either default.",
    )
    monitors.add_argument("--spike-grace-steps", type=int, default=2, help="Report but do not abort on spikes in the first N steps.")
    monitors.add_argument(
        "--max-arm-reference-error-rad", type=float, default=1.5,
        help="Joint-space arm deviation from the reference clip that stops the run. Loose on purpose: the "
             "policy is free to leave the clip in joint space, so this is a runaway backstop and the palm "
             "monitor (--max-palm-reference-error-m) is the criterion that should actually fire.",
    )
    monitors.add_argument(
        "--max-hand-reference-error-rad", type=float, default=None,
        help="Defaults to termination.hand_position_threshold_rad from the run config.",
    )
    monitors.add_argument(
        "--max-palm-reference-error-m", type=float, default=0.15,
        help="Palm position deviation from the reference clip's palm that stops the run. This is "
             "the criterion meant to fire, the arm-joint one being only a runaway backstop. It is "
             "deliberately NOT tied to termination.palm_keypoint_threshold_m (0.08 here) any more: "
             "training terminated on the RMS of four palm keypoints expressed in the cuboid's "
             "frame, which also carries orientation at 0.1 m/rad, while this is the distance "
             "between two palm centres in the robot base frame. Different quantity, so borrowing "
             "its number was misleading; 0.15 is an operational limit chosen on the robot. Note "
             "that 0.15 m is a whole bar length, so at this setting the monitor catches gross "
             "runaway rather than a drifting grasp.",
    )
    monitors.add_argument(
        "--low-safety", action="store_true",
        help="Demo mode: raise the reference-tracking monitors far enough that MOVING THE BAR BY "
             "HAND during the rollout does not abort the run. It loosens only the monitors that "
             "ask 'is the robot still doing what the demonstration did' -- palm/arm/hand deviation, "
             "action discontinuity, cuboid jump rejection and pose timeout -- because a robustness "
             "demo answers that question 'no' on purpose. It does NOT touch what the robot may "
             "physically do per tick: the step clamps, the velocity slew, the IK clamp, the joint "
             "limits, the arming prompt and the braking path are unchanged. Anything you set "
             "explicitly overrides the preset.",
    )
    monitors.add_argument("--reference-error-mode", choices=("off", "warn", "stop"), default="stop")
    monitors.add_argument("--current-warning-ma", type=float, default=170.0)
    monitors.add_argument("--state-timeout", type=float, default=0.25)
    monitors.add_argument("--state-wait-seconds", type=float, default=10.0)

    cube_group = parser.add_argument_group("cuboid observation")
    cube_group.add_argument("--cube-source", choices=("reference", "frozen", "pose-estimation"), default="reference")
    cube_group.add_argument(
        "--object-scale", type=float, default=1.0,
        help=(
            "The real bar's size relative to the nominal 0.15 x 0.05 x 0.05 m one (a 0.12 x 0.04 x "
            "0.04 bar is 0.8). Fed to the policy when it observes the scale, and lifts the "
            "reference clip's bar so it rests on the table."
        ),
    )
    cube_group.add_argument("--pose-address", default="tcp://127.0.0.1:5558")
    cube_group.add_argument("--pose-board-id", default="0")
    cube_group.add_argument("--pose-z-offset-m", type=float, default=0.03, help="Add this calibration offset to the live estimator Z coordinate.")
    cube_group.add_argument(
        "--pose-min-confidence", type=float, default=0.0,
        help="Drop estimator samples below this confidence (0 = off, the default). Do NOT use this "
             "as the noise filter: confidence falls as the robot occludes the board, so an "
             "absolute threshold goes from rejecting a few percent of samples to rejecting all of "
             "them exactly while the policy reaches for the bar, and the run stalls on "
             "--pose-timeout. --pose-median-window is the filter. Useful for a diagnostic.",
    )
    cube_group.add_argument(
        "--pose-median-window", type=int, default=5,
        help="Median over the last N live cuboid positions (1 = off). The noise filter: the "
             "estimator's excursions last one or two samples, so a median of 5 removes them "
             "whatever the confidence -- measured with the arm in the scene, worst case 1.89 mm "
             "against 14.8 mm raw. Costs about 80 ms of lag, which is free while the bar is "
             "standing still, and unlike a gate it can never starve the stream.",
    )
    cube_group.add_argument(
        "--pose-jump-reject-m", type=float, default=0.05,
        help="Drop a live pose this far from the held one; 0 disables. A gross-teleport backstop, "
             "not the noise filter -- the bar itself travels up to 16 mm between samples during "
             "transport, so a threshold near the noise scale would fight real motion.",
    )
    cube_group.add_argument(
        "--pose-jump-accept-samples", type=int, default=3,
        help="Accept a rejected jump once this many arrive in a row, so a bar that really moved "
             "gets through and the held pose can never go stale by more than this many samples.",
    )
    cube_group.add_argument(
        "--pose-filter-alpha", type=float, default=0.0,
        help="First-order low-pass on the live cuboid POSITION only (0 = off). Measurement says it "
             "is unnecessary once --pose-min-confidence is set, and it costs lag: the bar moves at "
             "up to 0.4 m/s during transport, so 40 ms of lag is 16 mm of error. Orientation is "
             "never low-passed -- see the note in cube_source.PoseEstimationCube.",
    )
    cube_group.add_argument(
        "--pose-timeout", type=float, default=2.0,
        help="Stop the run when the live cuboid pose is older than this. Generous on purpose: the "
             "estimator publishes nothing at all while it cannot see the board, and the hand "
             "occludes the tags exactly during the approach. Holding the last pose across that is "
             "sound because the bar does not move until it is grasped -- what the policy observes "
             "is the bar in the PALM frame, which keeps evolving correctly from the measured arm.",
    )
    cube_group.add_argument("--pose-wait-seconds", type=float, default=10.0)
    cube_group.add_argument("--check-cube-frame", action="store_true", help="Compare the live estimator pose with the reference clip, then exit.")
    cube_group.add_argument(
        "--max-placement-residual-m", type=float, default=0.07,
        help="Abort when the live cuboid stands farther than this from the nearest bank entry. "
             "Neighbouring entries are 2 mm apart, so a residual only grows large when the bar is "
             "OUTSIDE the bank's coverage (x -0.09..0.09, y 0.00..0.15 m) -- at 0.07 the guard "
             "therefore admits placements the policy was never trained on, and the clip it plays "
             "is a demonstration for a materially different pose.",
    )
    cube_group.add_argument("--max-placement-residual-deg", type=float, default=10.0)
    cube_group.add_argument("--max-placement-tilt-deg", type=float, default=10.0)

    hardware = parser.add_argument_group("hardware endpoints")
    hardware.add_argument("--arm-config", type=Path, default=DEFAULT_ARM_CONFIG)
    hardware.add_argument("--arm-stream-hz", type=float, default=100.0)
    hardware.add_argument(
        "--arm-connect-settle-seconds", type=float, default=1.0,
        help="Pause after binding the command socket so ZMQ PUB does not drop the first message before the controller has subscribed.",
    )
    hardware.add_argument(
        "--arm-velocity-source", choices=("controller", "finite-difference"), default="controller",
        help="Joint velocity fed to the observation: the controller's target Qd, or a finite difference of measured Q.",
    )
    hardware.add_argument("--hand-bind-address", default="127.0.0.1")
    hardware.add_argument("--hand-state-port", type=int, default=5563)
    hardware.add_argument("--hand-command-address", default="127.0.0.1")
    hardware.add_argument("--hand-command-port", type=int, default=5562)

    homing = parser.add_argument_group("homing")
    homing.add_argument("--home-seconds", type=float, default=5.0)
    homing.add_argument("--home-speed-rad-s", type=float, default=0.15)
    homing.add_argument("--home-settle-seconds", type=float, default=1.5)
    homing.add_argument("--home-tolerance-rad", type=float, default=0.05)
    homing.add_argument("--max-home-distance-rad", type=float, default=1.9)
    homing.add_argument("--hand-home-step-rad", type=float, default=0.03)
    homing.add_argument("--hand-home-step-seconds", type=float, default=0.02)
    homing.add_argument("--hand-home-tolerance-rad", type=float, default=0.18)
    homing.add_argument(
        "--hand-home-settle-seconds", type=float, default=1.0,
        help="Hold the start pose after softening the hand, so the first observation is not mid-transient.",
    )
    homing.add_argument("--hand-home-timeout-seconds", type=float, default=10.0)
    homing.add_argument("--brake-seconds", type=float, default=0.5)

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--debug-step", action="store_true")
    behaviour.add_argument("--no-realtime", action="store_true")
    behaviour.add_argument("--print-every", type=int, default=30)

    view = parser.add_argument_group("viewer (the MuJoCo window, on by default)")
    view.add_argument("--no-viewer", action="store_true", help="Run without the MuJoCo window.")
    view.add_argument("--no-ghost", action="store_true", help="Drop the green reference robot.")
    view.add_argument(
        "--viewer-hz", type=float, default=0.0,
        help="Cap the redraw rate (0, the default, draws every control step).",
    )
    view.add_argument(
        "--viewer-hold-seconds", type=float, default=5.0,
        help="Keep the window up after the run, so the frame that stopped it can be read.",
    )
    behaviour.add_argument(
        "--simulated-state-source", choices=("reference", "target"), default="reference",
        help=(
            "State fed back for a subsystem that is not read from hardware. 'reference' replays "
            "the bank clip and its recorded velocities, reproducing the training observation; "
            "'target' assumes the subsystem tracks the policy's own command, rate-limited by "
            "--simulated-max-velocity-rad-s."
        ),
    )
    behaviour.add_argument("--simulated-max-velocity-rad-s", type=float, default=3.0)
    behaviour.add_argument(
        "--previous-target-source", choices=("commanded", "applied"), default="commanded",
        help=(
            "Which target enters the observation. 'commanded' is what training fed back; "
            "'applied' reports what the safety limiter actually sent."
        ),
    )

    args = parser.parse_args()
    if args.commission_arm_only_ideal_context:
        if args.send_to_hand or args.use_real_hand_state:
            parser.error("arm-only ideal-context mode cannot send to or read the real hand")
        args.send_to_arm = True
        args.use_real_arm_state = True
        args.simulated_state_source = "reference"
        args.cube_source = "reference"
    elif args.commission_hand_only_ideal_context:
        if args.send_to_arm or args.use_real_arm_state:
            parser.error("hand-only ideal-context mode cannot send to or read the real arm")
        args.send_to_hand = True
        args.use_real_hand_state = True
        args.simulated_state_source = "reference"
        args.cube_source = "reference"
    if args.bank_index != "nearest":
        try:
            int(args.bank_index)
        except ValueError:
            parser.error("--bank-index must be an integer or 'nearest'")
    if args.control_hz <= 0.0:
        parser.error("--control-hz must be positive")
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    if not 0.0 <= args.target_smoothing < 1.0:
        parser.error("--target-smoothing must lie in [0, 1)")
    if args.startup_ramp_seconds < 0.0:
        parser.error("--startup-ramp-seconds cannot be negative")
    if args.startup_policy_blend_seconds < 0.0:
        parser.error("--startup-policy-blend-seconds cannot be negative")
    if args.hand_home_timeout_seconds <= 0.0:
        parser.error("--hand-home-timeout-seconds must be positive")
    for name in ("arm_action_scale", "hand_action_scale"):
        if getattr(args, name) <= 0.0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.object_scale <= 0.0:
        parser.error("--object-scale must be positive")
    if args.viewer_hz < 0.0:
        parser.error("--viewer-hz cannot be negative")
    if args.viewer_hold_seconds < 0.0:
        parser.error("--viewer-hold-seconds cannot be negative")
    if args.hand_stiffness is not None:
        if args.hand_stiffness <= 0.0:
            parser.error("--hand-stiffness must be positive")
        if not (args.send_to_hand or args.use_real_hand_state):
            parser.error("--hand-stiffness needs the hand to be read or armed (--use-real-hand-state / --send-to-hand)")
    if args.hand_home_stiffness is not None:
        if args.hand_home_stiffness <= 0.0:
            parser.error("--hand-home-stiffness must be positive")
        if args.hand_stiffness is None:
            parser.error("--hand-home-stiffness needs --hand-stiffness: it is the gain to drop to after homing")
        if not args.send_to_hand:
            parser.error("--hand-home-stiffness only applies when the hand is armed (--send-to-hand)")
    if args.hand_home_settle_seconds < 0.0:
        parser.error("--hand-home-settle-seconds cannot be negative")
    if args.low_safety:
        apply_low_safety(args, parser)
    return args


# Raised far enough that deliberately moving the bar mid-rollout cannot trip
# them. Every one of these is a *reference-tracking* monitor: it asks whether
# the robot is still doing what the demonstration did, which is exactly the
# question a robustness demo answers "no" to on purpose.
LOW_SAFETY_LIMITS = {
    "max_palm_reference_error_m": 1.00,
    "max_arm_reference_error_rad": 3.14,
    "max_hand_reference_error_rad": 3.14,
    "max_action_step": 5.0,
    "max_hand_action_step": 5.0,
    "pose_jump_reject_m": 1.00,
    "pose_timeout": 5.0,
}


def apply_low_safety(args, parser) -> None:
    """Loosen the reference-tracking monitors, leaving the motion limits alone.

    What the robot is physically allowed to do per tick is NOT touched:
    ``--max-arm-step-rad`` / ``--max-hand-step-rad``, the contract's velocity
    slew, the IK joint clamp, the joint limits, the arming prompt and the
    braking path all stay exactly as they are. A demo needs the run not to
    abort; it does not need the arm to move faster.
    """
    changed = []
    for name, loose in LOW_SAFETY_LIMITS.items():
        current = getattr(args, name)
        # Anything the operator set explicitly wins over the preset.
        if current is not None and current != parser.get_default(name):
            continue
        setattr(args, name, loose)
        changed.append("--{} {:g}".format(name.replace("_", "-"), loose))
    args.low_safety_changes = changed


if __name__ == "__main__":
    raise SystemExit(main())
