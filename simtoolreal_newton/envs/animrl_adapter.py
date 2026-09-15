"""AnimRL vectorised-environment view of :class:`MotionImitationEnv`.

The PPO runner and the evaluators were written against the five-value step
contract of the Isaac Gym environment::

    observations, privileged_observations, rewards, dones, infos = env.step(actions)

Isaac Lab's :class:`~isaaclab.envs.DirectRLEnv` returns an observation
dictionary and separate ``terminated`` / ``truncated`` flags. This thin
wrapper translates between the two and forwards every other attribute, so
``env.reference``, ``env.reset_idx`` or ``env.cfg.rewards`` keep working.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class AnimRLVecEnv:
    def __init__(self, env, launch_context=None) -> None:
        object.__setattr__(self, "env", env)
        object.__setattr__(self, "_launch_context", launch_context)

    # -- attribute forwarding -------------------------------------------------

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        setattr(self.env, name, value)

    @property
    def cfg(self):
        """The AnimRL configuration object (rewards, termination, env ...)."""
        return self.env.animrl_cfg

    @property
    def isaaclab_cfg(self):
        return self.env.cfg

    @property
    def unwrapped(self):
        return self.env

    # -- AnimRL contract -----------------------------------------------------

    def step(self, actions: torch.Tensor):
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = terminated | truncated
        return observations["policy"], observations.get("critic"), rewards, dones, extras

    def reset(
        self,
        reference_index: Optional[int] = None,
        translation_xy: Optional[Tuple[float, float]] = None,
        yaw_rad: Optional[float] = None,
    ) -> torch.Tensor:
        return self.env.reset_all(reference_index=reference_index, translation_xy=translation_xy, yaw_rad=yaw_rad)

    def get_observations(self) -> torch.Tensor:
        return self.env.get_observations()

    def get_privileged_observations(self):
        return self.env.get_privileged_observations()

    def render(self, sync_frame_time: bool = True) -> None:
        return None

    def close(self) -> None:
        self.env.close()
        context = self._launch_context
        if context is not None:
            object.__setattr__(self, "_launch_context", None)
            context.close()
