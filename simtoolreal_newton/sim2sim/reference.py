"""The transform bank as the sim2sim's reference: where the bar and the hand go.

The training environment never plays the raw demonstration; every episode
follows one entry of the transform bank (the demonstration retargeted to one
planar bar transform) and the bar itself is placed by
``MotionImitationEnv._cube_reference_root_states``. The functions here are
that arithmetic on the CPU, in torch so the rotation helpers are the ones the
environment uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from simtoolreal_newton import ROOT_DIR
from simtoolreal_newton.envs.cuboid_symmetry import (
    canonicalize_cuboid_orientation,
    cuboid_rotation_symmetries,
)
from simtoolreal_newton.envs.demonstration import JointDemonstration60Hz
from simtoolreal_newton.envs.object_scale import reference_height_shift
from simtoolreal_newton.envs.rotations import (
    normalize_canonical_quaternion,
    quat_multiply,
    quat_rotate,
)
from simtoolreal_newton.envs.transform_bank import TransformBank, nearest_transform_indices

from .constants import WORLD_AXIS_SIGN


@dataclass
class ReferenceTrack:
    """Bank, demonstration and the frame constants needed to place the bar."""

    bank: TransformBank
    demonstration: JointDemonstration60Hz
    robot_base_position: torch.Tensor  # (3,)
    object_half_extents: torch.Tensor  # (3,) nominal
    symmetries: torch.Tensor  # (K, 4)
    yaw_lever_arm_m: float

    @classmethod
    def load(cls, env_cfg, repo_root: Path = ROOT_DIR) -> "ReferenceTrack":
        randomization = env_cfg.object_randomization
        bank_path = Path(str(randomization.bank_path))
        if not bank_path.is_absolute():
            bank_path = Path(repo_root) / bank_path
        demo_path = Path(str(env_cfg.motion.file))
        if not demo_path.is_absolute():
            demo_path = Path(repo_root) / demo_path
        demonstration = JointDemonstration60Hz.load(
            demo_path, device="cpu", expected_hz=float(env_cfg.motion.frequency_hz)
        )
        bank = TransformBank.load(bank_path).to("cpu", torch.float32)
        if bank.sample_count != demonstration.sample_count:
            raise ValueError(
                "The transform bank has {} frames but the demonstration has {}".format(
                    bank.sample_count, demonstration.sample_count
                )
            )
        half = torch.tensor(
            [0.5 * float(v) for v in env_cfg.object.size_m], dtype=torch.float32
        )
        return cls(
            bank=bank,
            demonstration=demonstration,
            robot_base_position=torch.tensor(
                [float(v) for v in env_cfg.init_state.pos], dtype=torch.float32
            ),
            object_half_extents=half,
            symmetries=cuboid_rotation_symmetries(half.tolist()).to(torch.float32),
            yaw_lever_arm_m=float(randomization.nearest_yaw_lever_arm_m),
        )

    @property
    def last_index(self) -> int:
        return int(self.bank.last_index)

    def nearest_transform(self, translation_xy, yaw_rad: float) -> int:
        """The bank entry serving a continuous placement (as the env's reset does)."""
        translation = torch.tensor(
            [float(translation_xy[0]), float(translation_xy[1]), 0.0], dtype=torch.float32
        ).unsqueeze(0)
        yaw = torch.tensor([float(yaw_rad)], dtype=torch.float32)
        return int(
            nearest_transform_indices(
                translation, yaw, self.bank.translation, self.bank.yaw_rad, self.yaw_lever_arm_m
            )[0]
        )

    def sample(self, transform_index: int, frame_index: int):
        """Joint state and bar track of one bank entry at one frame (tensors of shape (1, ...))."""
        t = torch.tensor([int(transform_index)], dtype=torch.long)
        f = torch.tensor([int(frame_index)], dtype=torch.long)
        self.bank.validate_indices(t, f)
        return self.bank.sample(t, f)

    def cube_root_state(
        self,
        transform_index: int,
        frame_index: int,
        episode_translation: Optional[np.ndarray] = None,
        episode_yaw_rad: Optional[float] = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """``(13,)`` bar root state in the world frame: position, xyzw, linear, angular.

        Port of ``MotionImitationEnv._cube_reference_root_states``: the bank
        track is moved from the entry's transform to the episode's exact
        continuous one (a residual yaw about the bar's start position plus a
        residual translation) and lifted so a scaled bar rests on the table.
        With no episode transform the entry's own transform is used, which is
        what the training reset does from ``rsi_snap_placement_from_index``.
        """
        sample = self.sample(transform_index, frame_index)
        sign = torch.tensor(WORLD_AXIS_SIGN, dtype=torch.float32)
        base = self.robot_base_position
        position = base + sample.cube_pose[:, :3] * sign
        x, y, z, w = sample.cube_pose[:, 3:7].unbind(dim=1)
        quaternion_world = normalize_canonical_quaternion(torch.stack((-y, x, w, -z), dim=1))
        bank_yaw = self.bank.yaw_rad[transform_index]
        bank_translation = self.bank.translation[transform_index]
        if episode_translation is None:
            episode_translation = bank_translation.clone()
        else:
            episode_translation = torch.as_tensor(
                np.asarray(episode_translation, dtype=np.float32)
            ).reshape(3)
        if episode_yaw_rad is None:
            episode_yaw = bank_yaw.clone()
        else:
            episode_yaw = torch.tensor(float(episode_yaw_rad), dtype=torch.float32)
        half = 0.5 * (episode_yaw - bank_yaw)
        delta = torch.stack(
            (torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half))
        ).reshape(1, 4)
        start = self.sample(transform_index, 0)
        bank_start_position = base + start.cube_pose[:, :3] * sign
        actual_start = bank_start_position - bank_translation + episode_translation
        position = actual_start + quat_rotate(delta, position - bank_start_position)
        position = position.clone()
        position[:, 2] += reference_height_shift(
            torch.tensor([float(scale)], dtype=torch.float32), float(self.object_half_extents[2])
        )
        orientation = normalize_canonical_quaternion(quat_multiply(delta, quaternion_world))
        linear = quat_rotate(delta, sample.cube_linear_velocity * sign)
        angular = quat_rotate(delta, sample.cube_angular_velocity * sign)
        return torch.cat((position, orientation, linear, angular), dim=1)[0].numpy().astype(np.float64)

    def symmetry_index(self, cube_orientation_xyzw: np.ndarray, reference_xyzw: np.ndarray) -> int:
        """The cuboid symmetry chosen at reset and held for the episode."""
        orientation = torch.as_tensor(np.asarray(cube_orientation_xyzw, dtype=np.float32)).reshape(1, 4)
        reference = torch.as_tensor(np.asarray(reference_xyzw, dtype=np.float32)).reshape(1, 4)
        _, chosen = canonicalize_cuboid_orientation(
            normalize_canonical_quaternion(orientation),
            self.symmetries,
            normalize_canonical_quaternion(reference),
            return_index=True,
        )
        return int(chosen[0])
