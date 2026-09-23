"""The MuJoCo window of the earlier deployments, on this branch's sim2sim scene.

It shows what it always showed: the **robot in its measured pose** (the URDF
posed by the encoders), the **bar** at the pose the policy is being shown --
the live estimator's or the demonstration's, whichever the run is using -- and
the **green ghost** replaying the reference clip beside it.

Nothing is simulated. ``MujocoSim.reset`` writes the measured joints and the
observed bar pose and calls ``mj_forward``, so MuJoCo only answers "where is
everything, given this state"; the robot integrates reality. Closing the window
stops the run, as before.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np


class ViewerUnavailable(RuntimeError):
    """mujoco is missing, or the scene could not be built."""


class DeploymentViewer:
    """Poses the sim2sim scene at the measured state, once per control step."""

    def __init__(
        self,
        run,
        *,
        ghost: bool = True,
        ghost_offset: Optional[Sequence[float]] = None,
        object_scale: float = 1.0,
        update_hz: float = 0.0,
    ) -> None:
        try:
            from simtoolreal_newton.sim2sim.mujoco_sim import MujocoSceneConfig, MujocoSim
        except ImportError as error:
            raise ViewerUnavailable(
                "the viewer needs mujoco in the venv: {}. Install it with "
                "'uv pip install --python deps/IsaacLab/.venv/bin/python mujoco'".format(error)
            )
        try:
            config = MujocoSceneConfig.from_env_cfg(
                run.env_cfg,
                object_scale=float(object_scale),
                enable_viewer=True,
                enable_reference_ghost=bool(ghost),
            )
            if ghost_offset is not None:
                config.reference_ghost_offset_world = np.asarray(ghost_offset, dtype=np.float64)
            self.sim = MujocoSim(config)
        except Exception as error:
            raise ViewerUnavailable("{}: {}".format(type(error).__name__, error))
        self.robot_position = np.asarray(config.robot_position_world, dtype=np.float64)
        self.period = 1.0 / float(update_hz) if float(update_hz) > 0.0 else 0.0
        self._last_draw = 0.0
        self._zero_velocity = np.zeros(26, dtype=np.float64)
        self.failed = False

    def is_running(self) -> bool:
        """False once the operator closes the window. A broken viewer never stops the robot."""
        if self.failed:
            return True
        try:
            return bool(self.sim.viewer_is_running())
        except Exception:
            return True

    def update(self, measured_q, reference_q, cube_pose_base) -> None:
        """Pose the scene at the measured state. Never raises."""
        if self.failed:
            return
        if self.period > 0.0:
            now = time.monotonic()
            if now - self._last_draw < self.period:
                return
            self._last_draw = now
        try:
            pose = np.asarray(cube_pose_base, dtype=np.float64).reshape(7)
            root = np.zeros(13, dtype=np.float64)
            root[0:3] = self.robot_position + pose[0:3]
            root[3:7] = pose[3:7]
            self.sim.set_reference_ghost(np.asarray(reference_q, dtype=np.float64))
            # Writes the state and runs mj_forward; it does not step physics.
            self.sim.reset(np.asarray(measured_q, dtype=np.float64), self._zero_velocity, root)
        except Exception as error:  # a viewer must never stop a moving robot
            self.failed = True
            print("WARNING: the viewer stopped updating ({}: {}).".format(type(error).__name__, error))

    def close(self) -> None:
        try:
            self.sim.close()
        except Exception:
            pass
