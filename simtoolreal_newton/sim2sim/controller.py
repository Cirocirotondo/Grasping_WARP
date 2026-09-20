"""The action -> drive-target pipeline of the training environment, on the CPU.

One control step of ``MotionImitationEnv`` (``_pre_physics_step`` then
``_apply_action``) does, in order:

1. low-pass the raw action: ``filtered = lerp(filtered, action, alpha)``;
2. arm: the six filtered actions are a palm twist, capped direction-preserving
   at ``arm_translation_speed * dt`` / ``arm_rotation_speed * dt``, turned into
   a joint step by one damped-least-squares solve on the palm Jacobian at the
   *measured* arm configuration, clamped per joint at ``ik_max_joint_delta_rad``
   and accumulated onto the previous *commanded* arm target within the limits;
3. hand: ``default + clip(action * scale_hand_joint_target, +-clip_joint_target)``;
4. the commanded 26 targets go into the observation as ``previous_targets``,
   while what the drives receive is slewed towards them at the URDF joint
   velocity limit (``slew_targets_at_velocity_limit``).

The same helpers the environment calls are used here (``envs/operational_space``,
``envs/kinematics``), so a difference between the two simulators cannot come
from the controller.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from simtoolreal_newton.envs.kinematics import PalmKinematics
from simtoolreal_newton.envs.operational_space import (
    damped_least_squares_step,
    saturate_direction_preserving,
)

from .constants import ACTION_DIM, ARM_JOINT_NAMES, ROBOT_URDF

ARM_DOF = len(ARM_JOINT_NAMES)


class ActionPipeline:
    def __init__(
        self,
        env_cfg,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        velocity_limits: np.ndarray,
        kinematics: Optional[PalmKinematics] = None,
    ) -> None:
        control = env_cfg.control
        if str(control.action_parameterization) != "operational_space_arm":
            raise ValueError("Only the operational-space arm contract is supported")
        self.dt = float(env_cfg.sim.dt) * int(control.decimation)
        self.alpha = float(getattr(control, "action_filter_alpha", 1.0))
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("control.action_filter_alpha must lie in (0, 1]")
        self.hand_scale = float(control.scale_hand_joint_target)
        self.clip = float(control.clip_joint_target)
        self.translation_speed = float(control.arm_translation_speed_m_per_s)
        self.rotation_speed = float(control.arm_rotation_speed_rad_per_s)
        self.ik_damping = float(control.ik_damping)
        self.ik_max_delta = float(control.ik_max_joint_delta_rad)
        delay = int(getattr(getattr(env_cfg, "domain_randomization", None), "action_delay_max_steps", 0) or 0)
        if delay:
            raise ValueError("Action delay is not supported by the sim2sim runner")
        self.lower = torch.as_tensor(np.asarray(lower_limits, dtype=np.float32)).reshape(1, ACTION_DIM)
        self.upper = torch.as_tensor(np.asarray(upper_limits, dtype=np.float32)).reshape(1, ACTION_DIM)
        slew = torch.as_tensor(np.asarray(velocity_limits, dtype=np.float32)).reshape(1, ACTION_DIM) * self.dt
        if not bool(getattr(control, "slew_targets_at_velocity_limit", True)):
            slew = torch.full_like(slew, float("inf"))
        self.slew_per_step = slew
        default = list(env_cfg.init_state.default_arm_joint_angles) + list(
            env_cfg.init_state.default_hand_joint_angles
        )
        self.default_positions = torch.tensor(default, dtype=torch.float32).reshape(1, ACTION_DIM)
        self.default_hand = self.default_positions[:, ARM_DOF:]
        self.kinematics = kinematics or PalmKinematics(ROBOT_URDF, device="cpu", dtype=torch.float32)
        self.filtered_actions = torch.zeros((1, ACTION_DIM), dtype=torch.float32)
        self.position_targets = torch.zeros((1, ACTION_DIM), dtype=torch.float32)
        self.applied_targets = torch.zeros((1, ACTION_DIM), dtype=torch.float32)
        self.last_ik_residual = 0.0

    # -- reset -------------------------------------------------------------
    def reset_action(self, hand_positions) -> np.ndarray:
        """The 26-action equivalent of a joint pose: arm zero, hand residual."""
        hand = torch.as_tensor(np.asarray(hand_positions, dtype=np.float32)).reshape(1, -1)
        action = torch.zeros((1, ACTION_DIM), dtype=torch.float32)
        action[:, ARM_DOF:] = (hand - self.default_hand) / self.hand_scale
        return action[0].numpy()

    def reset(self, joint_positions) -> None:
        q = torch.as_tensor(np.asarray(joint_positions, dtype=np.float32)).reshape(1, ACTION_DIM)
        self.position_targets = q.clone()
        self.applied_targets = q.clone()
        self.filtered_actions = torch.as_tensor(self.reset_action(q[0, ARM_DOF:])).reshape(1, ACTION_DIM)
        self.last_ik_residual = 0.0

    # -- one control step ----------------------------------------------------
    def command(self, actions: np.ndarray, measured_arm_q: np.ndarray) -> np.ndarray:
        """Filter the action, update the commanded targets, return the drive targets."""
        action = torch.as_tensor(np.asarray(actions, dtype=np.float32)).reshape(1, ACTION_DIM)
        if self.alpha < 1.0:
            self.filtered_actions = torch.lerp(self.filtered_actions, action, self.alpha)
        else:
            self.filtered_actions = action.clone()
        arm_q = torch.as_tensor(np.asarray(measured_arm_q, dtype=np.float32)).reshape(1, ARM_DOF)
        arm_targets = self._arm_targets(self.filtered_actions[:, :ARM_DOF], arm_q)
        hand_targets = self.default_hand + (self.filtered_actions[:, ARM_DOF:] * self.hand_scale).clamp(
            -self.clip, self.clip
        )
        self.position_targets = torch.cat((arm_targets, hand_targets), dim=1)
        delta = (self.position_targets - self.applied_targets).clamp(-self.slew_per_step, self.slew_per_step)
        self.applied_targets = self.applied_targets + delta
        return self.applied_targets[0].numpy().astype(np.float64)

    def _arm_targets(self, arm_actions: torch.Tensor, arm_q: torch.Tensor) -> torch.Tensor:
        max_translation = self.translation_speed * self.dt
        max_rotation = self.rotation_speed * self.dt
        twist = torch.cat(
            (
                saturate_direction_preserving(arm_actions[:, 0:3] * max_translation, max_translation),
                saturate_direction_preserving(arm_actions[:, 3:6] * max_rotation, max_rotation),
            ),
            dim=1,
        )
        jacobian = self.kinematics.jacobian(arm_q)
        q_delta = damped_least_squares_step(jacobian, twist, self.ik_damping)
        q_delta = q_delta.clamp(-self.ik_max_delta, self.ik_max_delta)
        previous = self.position_targets[:, :ARM_DOF]
        targets = (previous + q_delta).clamp(self.lower[:, :ARM_DOF], self.upper[:, :ARM_DOF])
        achieved = torch.bmm(jacobian, (targets - previous).unsqueeze(-1)).squeeze(-1)
        self.last_ik_residual = float((twist - achieved).norm())
        return targets

    @property
    def previous_targets(self) -> np.ndarray:
        """What the observation reports: the commanded (unslewed) targets."""
        return self.position_targets[0].numpy().astype(np.float64)
