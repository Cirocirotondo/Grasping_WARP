#!/usr/bin/env python3
"""Stage 1 of the commissioning ladder: the deployment contract against the environment.

Runs the deterministic policy in the real environment (Isaac Lab / Newton, one
environment) from a chosen reference frame and bank clip, and at every step
rebuilds the observation and the commanded targets from the environment's
measured state through ``simtoolreal_newton.deployment.contract`` -- the code
that runs on the robot -- then compares the two. The observation is rebuilt
from the environment's own previous targets, so that check isolates the
observation builder; the targets are checked twice, from a pipeline re-synced
to the environment's state before every step (one-step error) and from one that
runs free from the reset (accumulated error).

    deps/IsaacLab/.venv/bin/python scripts/check_deployment_contract.py \\
        --checkpoint deploy/policies/<run>/model_<N>.pt --rsi-index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from simtoolreal_newton.deployment import (  # noqa: E402
    ARM_DOF,
    ActionPipeline,
    DeploymentRun,
    ObservationInputs,
    build_observation,
)
from simtoolreal_newton.deployment.contract import load_saved_configuration  # noqa: E402
from simtoolreal_newton.launch import add_env_arguments, make_env  # noqa: E402

OBSERVATION_BLOCKS = (
    ("normalized q", 0, 26),
    ("previous targets", 26, 52),
    ("dq", 52, 78),
    ("phase", 78, 79),
    ("palm position", 79, 82),
    ("palm rotation 6d", 82, 88),
    ("fingertips in palm", 88, 103),
    ("cube rotation 6d", 103, 109),
    ("cube centre in palm", 109, 112),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--rsi-index", type=int, default=0)
    parser.add_argument("--bank-index", default="nearest", help="A bank entry, or 'nearest' for the demonstration's own placement.")
    parser.add_argument("--steps", type=int, default=0, help="0 runs to the end of the clip.")
    parser.add_argument("--object-scale", type=float, default=1.0, help="The bar's scale for this episode (pinned in the environment).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--obs-tolerance", type=float, default=2e-3)
    parser.add_argument("--target-tolerance", type=float, default=5e-4)
    parser.add_argument("--print-every", type=int, default=100)
    add_env_arguments(parser)
    return parser.parse_args()


def block_errors(difference: np.ndarray) -> str:
    blocks = list(OBSERVATION_BLOCKS)
    if difference.shape[0] > 112:
        blocks.append(("bar scale", 112, difference.shape[0]))
    return "  ".join(
        "{}={:.2e}".format(name, float(np.max(np.abs(difference[start:stop]))))
        for name, start, stop in blocks
    )


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config else checkpoint.parent / "config.json"

    run = DeploymentRun(checkpoint, config_path)
    if not 0 <= args.rsi_index < run.last_index:
        raise SystemExit("--rsi-index must lie in [0, {}]".format(run.last_index - 1))
    transform_index = run.default_transform_index() if args.bank_index == "nearest" else int(args.bank_index)

    env_cfg, _ = load_saved_configuration(config_path)
    env_cfg.seed = int(args.seed)
    env_cfg.env.num_envs = 1
    env_cfg.env.play = True
    env_cfg.termination.enabled = False
    env_cfg.object_assist.enabled = False
    env_cfg.viewer.enable_viewer = False
    # Pin the episode's bar scale and switch the training-time randomisation
    # off: the robot reports its state without noise, and the contract must
    # match the environment's clean observation.
    env_cfg.object_randomization.scale_min = float(args.object_scale)
    env_cfg.object_randomization.scale_max = float(args.object_scale)
    env_cfg.object_randomization.scale_nominal_probability = 0.0
    env_cfg.object_randomization.scale_anchors = []
    env_cfg.domain_randomization.enabled = False
    env = make_env(env_cfg, num_envs=1, device=args.sim_device, physics=args.physics, visualizer=args.viz)
    inner = env.unwrapped
    device = inner.device
    try:
        env.max_episode_length = int(run.last_index)
        env.cfg.env.episode_length = env.max_episode_length

        lower_error = float(torch.max(torch.abs(inner.joint_lower_limits.cpu() - run.joint_lower_limits)))
        upper_error = float(torch.max(torch.abs(inner.joint_upper_limits.cpu() - run.joint_upper_limits)))
        slew_error = float(torch.max(torch.abs(inner.target_slew_per_step.cpu() - run.target_slew_per_step)))
        print("joint limits: simulator vs URDF differ by {:.2e} / {:.2e} rad; slew per step by {:.2e}".format(
            lower_error, upper_error, slew_error
        ))

        env.reset()
        env_ids = torch.arange(1, device=device, dtype=torch.long)
        inner.reset_idx(
            env_ids,
            reference_indices=torch.tensor([args.rsi_index], device=device),
            transform_indices=torch.tensor([transform_index], device=device),
        )
        inner.scene.write_data_to_sim()
        inner.sim.forward()
        inner.scene.update(dt=inner.physics_dt)
        inner.compute_observations()
        inner.obs_buf = inner._observation_dict()
        observation_env = inner.policy_obs[0].cpu().numpy().astype(np.float64)
        symmetry_index = int(inner.symmetry_index[0])
        env_scale = float(inner.object_scale[0])
        if abs(env_scale - args.object_scale) > 1e-6:
            raise SystemExit("The environment drew scale {:.4f}, expected {:.4f}".format(env_scale, args.object_scale))
        print("bank clip {}, reference index {}, symmetry element {}, bar scale {:g} ({}D observation)".format(
            transform_index, args.rsi_index, symmetry_index, env_scale, run.observation_dim
        ))

        start_q = run.reference_q(transform_index, args.rsi_index)
        free_pipeline = ActionPipeline(run)
        free_pipeline.reset(start_q)
        synced_pipeline = ActionPipeline(run)
        synced_pipeline.reset(start_q)

        total = args.steps if args.steps > 0 else run.last_index - args.rsi_index
        worst = {"obs": 0.0, "action": 0.0, "targets_1step": 0.0, "applied_1step": 0.0, "targets_free": 0.0, "filter_free": 0.0}
        worst_blocks = np.zeros(run.observation_dim)
        steps = 0
        with torch.inference_mode():
            for _ in range(total):
                q = inner.q[0].cpu().numpy().astype(np.float64)
                dq = inner.dq[0].cpu().numpy().astype(np.float64)
                previous = inner.position_targets[0].cpu().numpy().astype(np.float64)
                reference_index = int(inner.reference_index[0])
                cube_base = torch.cat((inner.cube_position[0] - inner.robot_base_position, inner.cube_orientation[0])).cpu()
                observation_dep = build_observation(
                    run,
                    ObservationInputs(q, dq, previous, reference_index, cube_base, symmetry_index, args.object_scale),
                )
                difference = observation_dep - observation_env
                worst_blocks = np.maximum(worst_blocks, np.abs(difference))
                worst["obs"] = max(worst["obs"], float(np.max(np.abs(difference))))

                action = run.act(observation_env)
                worst["action"] = max(worst["action"], float(np.max(np.abs(run.act(observation_dep) - action))))

                # Re-sync one pipeline to the environment's state, so its error is one step's.
                synced_pipeline.position_targets = torch.as_tensor(previous, dtype=torch.float32)
                synced_pipeline.applied_targets = inner.applied_targets[0].cpu().clone()
                synced_pipeline.filtered_actions = inner.filtered_actions[0].cpu().clone()
                targets_synced = synced_pipeline.command(action, q[:ARM_DOF])
                applied_synced = synced_pipeline.apply()
                targets_free = free_pipeline.command(action, q[:ARM_DOF])
                free_pipeline.apply()

                observation_next, _, _, dones, _ = env.step(torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0))
                steps += 1
                if bool(dones[0]):
                    # The environment has already reset this world's targets and
                    # filter to a new episode; there is nothing left to compare.
                    print("episode ended at step {} (reference index {})".format(steps, int(inner.reference_index[0])))
                    break
                env_targets = inner.position_targets[0].cpu().numpy().astype(np.float64)
                env_applied = inner.applied_targets[0].cpu().numpy().astype(np.float64)
                env_filtered = inner.filtered_actions[0].cpu().numpy().astype(np.float64)
                worst["targets_1step"] = max(worst["targets_1step"], float(np.max(np.abs(targets_synced - env_targets))))
                worst["applied_1step"] = max(worst["applied_1step"], float(np.max(np.abs(applied_synced - env_applied))))
                worst["targets_free"] = max(worst["targets_free"], float(np.max(np.abs(targets_free - env_targets))))
                worst["filter_free"] = max(
                    worst["filter_free"],
                    float(np.max(np.abs(free_pipeline.filtered_actions.numpy() - env_filtered))),
                )
                observation_env = observation_next[0].cpu().numpy().astype(np.float64)
                if args.print_every and steps % args.print_every == 0:
                    print("step {:4d} ref {:4d}: obs {:.2e}  targets 1-step {:.2e}  free {:.2e}".format(
                        steps, int(inner.reference_index[0]), worst["obs"], worst["targets_1step"], worst["targets_free"]
                    ))

        print()
        print("Deployment contract vs environment over {} steps:".format(steps))
        print("  observation (max abs)       : {:.2e}   [{}]".format(worst["obs"], block_errors(worst_blocks)))
        print("  action from rebuilt obs     : {:.2e}".format(worst["action"]))
        print("  commanded targets, one step : {:.2e} rad".format(worst["targets_1step"]))
        print("  applied targets, one step   : {:.2e} rad".format(worst["applied_1step"]))
        print("  commanded targets, free run : {:.2e} rad".format(worst["targets_free"]))
        print("  filtered actions, free run  : {:.2e}".format(worst["filter_free"]))
        passed = (
            worst["obs"] <= args.obs_tolerance
            and worst["targets_1step"] <= args.target_tolerance
            and worst["applied_1step"] <= args.target_tolerance
        )
        print("CONTRACT {}".format("OK" if passed else "MISMATCH"))
        return 0 if passed else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
