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


def _bind_space_to_simulation_pause() -> None:
    """Rebind the viewer's space bar from pausing rendering to pausing the simulation.

    Newton binds space to ``ViewerGL._paused``, which only stops the viewer
    from drawing: Isaac Lab keeps calling ``env.step`` behind the frozen
    window, so releasing the pause reveals a policy and a robot that ran on
    without us. The pause that actually blocks the loop is Isaac Lab's
    ``_paused_training`` -- ``SimulationContext.update_visualizers`` spins on
    ``is_training_paused()`` until it clears -- and it is otherwise reachable
    only through the "Pause Simulation" button in the side panel.

    Patched on the class rather than the instance, and before any viewer
    exists: ``ViewerGL.__init__`` registers ``self.on_key_press`` as a bound
    method, so a patch applied after construction would never be called.
    """
    try:
        import pyglet
        from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL
    except Exception:  # visualizer extras absent, or a non-GL backend
        return
    if getattr(NewtonViewerGL, "_simtoolreal_space_pauses_sim", False):
        return

    inherited = NewtonViewerGL.on_key_press

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            # Mirrors the guard in the method we are replacing: a space typed
            # into an ImGui text field belongs to the field.
            if not self.ui.is_capturing():
                self._paused_training = not self._paused_training
            return
        inherited(self, symbol, modifiers)

    NewtonViewerGL.on_key_press = on_key_press
    NewtonViewerGL._simtoolreal_space_pauses_sim = True


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
    if visualizer is not None and str(visualizer).startswith("newton"):
        _bind_space_to_simulation_pause()
    try:
        env = MotionImitationEnv(env_cfg)
    except Exception:
        stack.close()
        raise
    return AnimRLVecEnv(env, stack)
