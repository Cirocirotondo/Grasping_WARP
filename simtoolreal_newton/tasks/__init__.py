"""Gymnasium registration so the task is reachable from Isaac Lab's own CLI.

The primary training path is ``scripts/train.py`` (the AnimRL PPO runner with
its checkpoint schema, evaluators and deployment scoring). This registration
additionally exposes the same environment to Isaac Lab's ``isaaclab train``
entry point, e.g. with RSL-RL::

    deps/IsaacLab/.venv/bin/isaaclab train --task SimToolReal-Grasp-Direct \\
        --rl_library rsl_rl --num_envs 4096 physics=newton_mjwarp
"""

import gymnasium as gym

from . import agents

gym.register(
    id="SimToolReal-Grasp-Direct",
    entry_point="simtoolreal_newton.envs.motion_imitation_env:MotionImitationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "simtoolreal_newton.envs.motion_imitation_env_cfg:MotionImitationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimToolRealPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
