"""Build a ready-to-train environment from an AnimRL-style configuration.

Isaac Lab needs the physics backend and the visualizer chosen *before* the
simulation context exists. :func:`make_env` wraps that ceremony so a script
only has to say::

    env = make_env(SimToolRealCfg(), num_envs=4096)          # headless, Newton
    env = make_env(cfg, num_envs=1, visualizer="newton")     # with the viewer
"""

from __future__ import annotations

import argparse
import contextlib
from typing import Optional

import warp as wp

# Isaac Lab does not use Warp autodiff; skipping adjoint codegen roughly halves
# kernel compilation on a cold cache. Must run before any Warp kernel module.
wp.config.enable_backward = False


def add_env_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI flags shared by every script that stands up a simulation."""
    parser.add_argument("--sim-device", default=None, help="Simulation device, e.g. cuda:0 (default: config).")
    parser.add_argument(
        "--physics",
        default=None,
        choices=["newton_mjwarp", "physx", "ovphysx", "isaacsim_physx"],
        help="Physics backend (default: sim.physics in the configuration, newton_mjwarp).",
    )
    parser.add_argument(
        "--viz",
        default=None,
        help="Visualizer to open: newton (kit-less viewer), rerun, viser, kit (needs Isaac Sim). Default: none.",
    )


def make_env(
    animrl_cfg,
    num_envs: Optional[int] = None,
    device: Optional[str] = None,
    physics: Optional[str] = None,
    visualizer: Optional[str] = None,
    debug_cube_contacts: bool = False,
    camera: bool = False,
):
    """Create a :class:`~simtoolreal_newton.envs.animrl_adapter.AnimRLVecEnv`."""
    from isaaclab.app import launch_simulation

    from simtoolreal_newton.envs.animrl_adapter import AnimRLVecEnv
    from simtoolreal_newton.envs.motion_imitation_env import MotionImitationEnv
    from simtoolreal_newton.envs.motion_imitation_env_cfg import build_env_cfg

    if physics is not None:
        animrl_cfg.sim.physics = str(physics)
    env_cfg = build_env_cfg(animrl_cfg, num_envs=num_envs, device=device)
    env_cfg.debug_cube_contacts = bool(debug_cube_contacts)
    if camera:
        from simtoolreal_newton.envs.motion_imitation_env_cfg import make_camera_cfg

        env_cfg.camera = make_camera_cfg(
            animrl_cfg,
            width=int(getattr(animrl_cfg.viewer, "training_camera_width", 640)),
            height=int(getattr(animrl_cfg.viewer, "training_camera_height", 480)),
        )
    launcher_args = {
        "device": env_cfg.sim.device,
        "headless": visualizer is None,
        # Isaac Lab's kitless launch path joins this with " ".join(...) before
        # re-splitting it; a bare string gets joined character-by-character
        # (e.g. "newton" -> "n e w t o n"), so it must be a list.
        "visualizer": [visualizer] if visualizer is not None else None,
        "visualizer_explicit": visualizer is not None,
    }
    stack = contextlib.ExitStack()
    stack.enter_context(launch_simulation(env_cfg, launcher_args))
    try:
        env = MotionImitationEnv(env_cfg)
    except Exception:
        stack.close()
        raise
    return AnimRLVecEnv(env, stack)
