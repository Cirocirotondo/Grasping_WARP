"""One measured episode: the training loop's order of operations, in MuJoCo.

Per control step, exactly as ``MotionImitationEnv.step``: observation from the
current state -> deterministic action -> filtered action -> commanded and
slewed targets -> ``substeps`` physics steps -> bar velocity cap -> reference
index + 1 -> tracking metrics against the reference at the new index.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from simtoolreal_newton.envs.cuboid_symmetry import apply_cuboid_symmetry
from simtoolreal_newton.envs.rotations import normalize_canonical_quaternion

from .constants import ARM_JOINT_NAMES
from .controller import ActionPipeline
from .mujoco_sim import MujocoSim
from .observation import build_observation
from .reference import ReferenceTrack

ARM_DOF = len(ARM_JOINT_NAMES)


@dataclass
class EpisodeSpec:
    transform_index: int
    rsi_index: int = 0
    scale: float = 1.0
    episode_translation: Optional[np.ndarray] = None  # world-frame planar offset, None = bank entry
    episode_yaw_rad: Optional[float] = None
    max_steps: int = 0  # 0 = to the end of the demonstration
    terminate: bool = False  # stop when the training termination would have fired


@dataclass
class EpisodeResult:
    spec: EpisodeSpec
    steps: int
    final_reference_index: int
    max_object_error_m: float
    final_object_error_m: float
    final_orientation_error_rad: float
    peak_lift_m: float
    final_lift_m: float
    max_hand_q_error_rad: float
    first_object_violation_index: int  # -1 = never above termination.object_position_threshold_m
    blown: bool
    trace: Dict[str, List] = field(default_factory=dict)

    def failed(self, gate_m: float = 0.07) -> bool:
        return bool(self.blown or self.max_object_error_m > gate_m)

    def summary(self) -> dict:
        return {
            "transform_index": int(self.spec.transform_index),
            "rsi_index": int(self.spec.rsi_index),
            "scale": float(self.spec.scale),
            "steps": int(self.steps),
            "final_reference_index": int(self.final_reference_index),
            "max_object_error_m": float(self.max_object_error_m),
            "final_object_error_m": float(self.final_object_error_m),
            "final_orientation_error_rad": float(self.final_orientation_error_rad),
            "peak_lift_m": float(self.peak_lift_m),
            "final_lift_m": float(self.final_lift_m),
            "max_hand_q_error_rad": float(self.max_hand_q_error_rad),
            "first_object_violation_index": int(self.first_object_violation_index),
            "blown": bool(self.blown),
        }


def _orientation_error(actual_xyzw, reference_xyzw, symmetries, symmetry_index) -> float:
    actual = apply_cuboid_symmetry(
        normalize_canonical_quaternion(torch.as_tensor(np.asarray(actual_xyzw, dtype=np.float32)).reshape(1, 4)),
        symmetries,
        torch.tensor([int(symmetry_index)]),
    )
    reference = normalize_canonical_quaternion(
        torch.as_tensor(np.asarray(reference_xyzw, dtype=np.float32)).reshape(1, 4)
    )
    dot = float((actual * reference).sum().abs().clamp(max=1.0))
    return 2.0 * math.acos(dot)


def run_episode(
    run,
    policy: Callable[[np.ndarray], np.ndarray],
    track: ReferenceTrack,
    sim: MujocoSim,
    pipeline: ActionPipeline,
    spec: EpisodeSpec,
    *,
    realtime: bool = False,
    print_every: int = 0,
    on_step: Optional[Callable[[int, dict], None]] = None,
) -> EpisodeResult:
    env_cfg = run.env_cfg
    last_index = track.last_index
    if not 0 <= spec.rsi_index < last_index:
        raise ValueError("rsi_index must lie in [0, {}]".format(last_index - 1))
    threshold = float(env_cfg.termination.object_position_threshold_m)
    grace = int(getattr(env_cfg.termination, "grace_steps", 0) or 0)
    control_dt = pipeline.dt
    observes_scale = run.observes_scale
    override = run.observed_scale_override

    def reference_root(frame: int) -> np.ndarray:
        return track.cube_root_state(
            spec.transform_index, frame, spec.episode_translation, spec.episode_yaw_rad, spec.scale
        )

    sample = track.sample(spec.transform_index, spec.rsi_index)
    root = reference_root(spec.rsi_index)
    sim.reset(sample.q[0].numpy(), sample.dq[0].numpy(), root)
    pipeline.reset(sample.q[0].numpy())
    state = sim.get_state()
    symmetry = track.symmetry_index(state["cube_orientation_world_xyzw"], root[3:7])
    sim.set_reference_ghost(sample.q[0].numpy())
    sim.sync_viewer()
    initial_height = float(state["cube_com_height_world"])

    trace = {
        "reference_indices": [],
        "policy_actions": [],
        "filtered_actions": [],
        "actual_joint_positions": [],
        "commanded_targets": [],
        "applied_targets": [],
        "reference_joint_positions": [],
        "cube_position_error_m": [],
        "cube_lift_m": [],
        "hand_q_error_rad": [],
        "ik_residual": [],
    }
    reference_index = int(spec.rsi_index)
    steps = 0
    max_error = 0.0
    peak_lift = 0.0
    max_hand_error = 0.0
    violation_steps = 0
    first_violation = -1
    blown = False
    error = lift = orientation_error = hand_error = 0.0
    while reference_index < last_index and sim.viewer_is_running():
        if spec.max_steps and steps >= spec.max_steps:
            break
        started = time.perf_counter()
        phase = reference_index / float(last_index)
        observation = build_observation(
            state,
            pipeline.previous_targets,
            phase,
            sim.joint_lower_limits,
            sim.joint_upper_limits,
            track.symmetries,
            symmetry,
            spec.scale if observes_scale else None,
            override,
        )
        action = policy(observation)
        applied = pipeline.command(action, state["joint_positions"][:ARM_DOF])
        next_sample = track.sample(spec.transform_index, reference_index + 1)
        sim.set_reference_ghost(next_sample.q[0].numpy())
        sim.set_position_targets(applied)
        sim.step_control()
        reference_index += 1
        steps += 1
        state = sim.get_state()
        if not (
            np.all(np.isfinite(state["joint_positions"]))
            and np.all(np.isfinite(state["joint_velocities"]))
            and np.all(np.isfinite(state["cube_position_world"]))
        ):
            blown = True
            break
        root = reference_root(reference_index)
        error = float(np.linalg.norm(state["cube_position_world"] - root[0:3]))
        orientation_error = _orientation_error(
            state["cube_orientation_world_xyzw"], root[3:7], track.symmetries, symmetry
        )
        lift = float(state["cube_com_height_world"]) - initial_height
        reference_q = next_sample.q[0].numpy()
        hand_error = float(np.max(np.abs(state["joint_positions"][ARM_DOF:] - reference_q[ARM_DOF:])))
        max_error = max(max_error, error)
        peak_lift = max(peak_lift, lift)
        max_hand_error = max(max_hand_error, hand_error)
        if error > threshold:
            violation_steps += 1
            if violation_steps > grace and first_violation < 0:
                first_violation = reference_index
        else:
            violation_steps = 0
        trace["reference_indices"].append(reference_index)
        trace["policy_actions"].append(np.asarray(action, dtype=np.float64).copy())
        trace["filtered_actions"].append(pipeline.filtered_actions[0].numpy().astype(np.float64))
        trace["actual_joint_positions"].append(state["joint_positions"].copy())
        trace["commanded_targets"].append(pipeline.previous_targets.copy())
        trace["applied_targets"].append(np.asarray(applied, dtype=np.float64).copy())
        trace["reference_joint_positions"].append(reference_q.astype(np.float64))
        trace["cube_position_error_m"].append(error)
        trace["cube_lift_m"].append(lift)
        trace["hand_q_error_rad"].append(hand_error)
        trace["ik_residual"].append(pipeline.last_ik_residual)
        if print_every and (steps == 1 or steps % print_every == 0):
            print(
                "step={:4d} ref={:4d} phase={:.3f} cube_err={:.3f} m lift={:.3f} m "
                "max|hand-ref|={:.3f} rad contacts={}".format(
                    steps, reference_index, reference_index / float(last_index), error, lift, hand_error,
                    len(sim.robot_cube_contacts()),
                ),
                flush=True,
            )
        if on_step is not None:
            on_step(steps, state)
        if spec.terminate and first_violation >= 0:
            break
        if realtime:
            remaining = control_dt - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)
    return EpisodeResult(
        spec=spec,
        steps=steps,
        final_reference_index=reference_index,
        max_object_error_m=max_error,
        final_object_error_m=error,
        final_orientation_error_rad=orientation_error,
        peak_lift_m=peak_lift,
        final_lift_m=lift,
        max_hand_q_error_rad=max_hand_error,
        first_object_violation_index=first_violation,
        blown=blown,
        trace=trace,
    )
