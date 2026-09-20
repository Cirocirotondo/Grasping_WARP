"""The policy observation, built from a MuJoCo state exactly as the env builds it.

Layout (``MotionImitationEnv.compute_observations``), 112 entries plus the
optional cuboid scale column:

    normalised joint positions (26) | previous commanded targets (26) |
    joint velocities (26) | phase (1) | palm position in the robot frame (3) |
    palm rotation 6D in the robot frame (6) | fingertips in the palm frame (15)
    | canonical bar rotation 6D in the palm frame (6) | bar centre in the palm
    frame (3) | [bar scale (1)]

All arithmetic runs through ``envs/rotations.py`` and ``envs/cuboid_symmetry.py``
so the sim2sim and the training environment share one implementation.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch

from simtoolreal_newton.envs.cuboid_symmetry import apply_cuboid_symmetry
from simtoolreal_newton.envs.object_scale import scale_observation
from simtoolreal_newton.envs.rotations import (
    normalize_canonical_quaternion,
    quat_conjugate,
    quat_multiply,
    quat_rotate,
    quat_rotate_inverse,
    quat_to_rotation_6d,
)

from .constants import (
    ACTION_DIM,
    BASE_OBSERVATION_DIM,
    FINGERTIP_OFFSETS,
    PALM_ORIENTATION_IN_WRIST_XYZW,
    PALM_POSITION_IN_WRIST,
)


def _t(values, shape=None) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.float32)
    return tensor.reshape(shape) if shape is not None else tensor


def palm_pose_from_wrist(
    wrist_position_world, wrist_orientation_world_xyzw
) -> tuple[torch.Tensor, torch.Tensor]:
    """``((1, 3), (1, 4))`` palm frame from the wrist body plus the fixed offset."""
    wrist_orientation = normalize_canonical_quaternion(_t(wrist_orientation_world_xyzw, (1, 4)))
    position = _t(wrist_position_world, (1, 3)) + quat_rotate(
        wrist_orientation, _t(PALM_POSITION_IN_WRIST, (1, 3))
    )
    orientation = normalize_canonical_quaternion(
        quat_multiply(wrist_orientation, _t(PALM_ORIENTATION_IN_WRIST_XYZW, (1, 4)))
    )
    return position, orientation


def fingertip_positions_from_bodies(
    fingertip_body_positions_world, fingertip_body_orientations_world_xyzw
) -> torch.Tensor:
    """``(1, 5, 3)`` fingertips: distal phalanx origin plus the fixed tip offset."""
    positions = _t(fingertip_body_positions_world, (1, 5, 3))
    orientations = normalize_canonical_quaternion(_t(fingertip_body_orientations_world_xyzw, (1, 5, 4)))
    return positions + quat_rotate(orientations, _t(FINGERTIP_OFFSETS, (1, 5, 3)))


def normalize_joint_positions(
    positions: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    return (2.0 * (positions - lower) / (upper - lower) - 1.0).clamp(-1.0, 1.0)


def build_observation(
    state: Mapping[str, np.ndarray],
    previous_targets: np.ndarray,
    phase: float,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    symmetries: torch.Tensor,
    symmetry_index: int,
    scale: Optional[float] = None,
    observed_scale_override: Optional[float] = None,
) -> np.ndarray:
    """One observation row, ``float32`` of length 112 (+1 with ``scale``).

    ``state`` is what :meth:`~simtoolreal_newton.sim2sim.mujoco_sim.MujocoSim.get_state`
    returns. ``scale`` is the episode's bar scale when the checkpoint observes
    it (``object_randomization.observe_scale``) and ``None`` otherwise.
    """
    q = _t(state["joint_positions"], (1, ACTION_DIM))
    dq = _t(state["joint_velocities"], (1, ACTION_DIM))
    targets = _t(previous_targets, (1, ACTION_DIM))
    lower = _t(lower_limits, (1, ACTION_DIM))
    upper = _t(upper_limits, (1, ACTION_DIM))
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")

    palm_position, palm_orientation = palm_pose_from_wrist(
        state["wrist_position_world"], state["wrist_orientation_world_xyzw"]
    )
    robot_position = _t(state["robot_position_world"], (1, 3))
    robot_orientation = normalize_canonical_quaternion(_t(state["robot_orientation_world_xyzw"], (1, 4)))
    palm_position_robot = quat_rotate_inverse(robot_orientation, palm_position - robot_position)
    palm_orientation_robot = normalize_canonical_quaternion(
        quat_multiply(quat_conjugate(robot_orientation), palm_orientation)
    )
    fingertips = fingertip_positions_from_bodies(
        state["fingertip_body_positions_world"], state["fingertip_body_orientations_world_xyzw"]
    )
    fingertips_palm = quat_rotate_inverse(
        palm_orientation.unsqueeze(1).expand(-1, 5, -1), fingertips - palm_position.unsqueeze(1)
    ).reshape(1, 15)
    cube_position = _t(state["cube_position_world"], (1, 3))
    cube_orientation = apply_cuboid_symmetry(
        normalize_canonical_quaternion(_t(state["cube_orientation_world_xyzw"], (1, 4))),
        symmetries,
        torch.tensor([int(symmetry_index)], dtype=torch.long),
    )
    cube_center_palm = quat_rotate_inverse(palm_orientation, cube_position - palm_position)
    cube_orientation_palm = normalize_canonical_quaternion(
        quat_multiply(quat_conjugate(palm_orientation), cube_orientation)
    )
    parts = [
        normalize_joint_positions(q, lower, upper),
        targets,
        dq,
        torch.tensor([[float(np.clip(phase, 0.0, 1.0))]], dtype=torch.float32),
        palm_position_robot,
        quat_to_rotation_6d(palm_orientation_robot),
        fingertips_palm,
        quat_to_rotation_6d(cube_orientation_palm),
        cube_center_palm,
    ]
    if scale is not None:
        parts.append(
            scale_observation(torch.tensor([float(scale)], dtype=torch.float32), observed_scale_override)
        )
    observation = torch.cat(parts, dim=1)[0].numpy().astype(np.float32)
    expected = BASE_OBSERVATION_DIM + (1 if scale is not None else 0)
    if observation.shape != (expected,):
        raise RuntimeError(
            "Observation has shape {}, expected ({},)".format(observation.shape, expected)
        )
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("Observation contains NaN or infinite values")
    return observation
