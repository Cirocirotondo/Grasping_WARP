#!/usr/bin/env python3
"""Where in the cuboid-pose envelope does a policy fail?

    python scripts/sweep_pose_success.py \\
        --checkpoint logs/simtoolreal/<run>/model_23500.pt \\
        --rsi-index 760 --output sweep.json

One environment per grid point (yaw x translation), all reset to the same
reference index, the deterministic policy run to the end of the demonstration
with termination OFF so every episode reports the full trajectory. Per point:
the largest cuboid position error along the way (the training criterion fails
an episode above termination.object_position_threshold_m), the peak lift, and
the pose errors at the end.

With --bank-index the grid is replaced by --repeats copies of one transform
bank entry, which is what the single-pose curriculum trains on.

Training metrics average over randomly drawn poses, which is exactly what
hides a failure that lives in one corner of the envelope.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402

from evaluate import load_saved_configuration  # noqa: E402
from test_headless_env import apply_overrides  # noqa: E402

from simtoolreal_newton.launch import make_env  # noqa: E402
from simtoolreal_newton.runners import PPO  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--rsi-index", type=int, default=760)
    parser.add_argument("--yaw-steps", type=int, default=10)
    parser.add_argument("--x-steps", type=int, default=5)
    parser.add_argument("--y-steps", type=int, default=5)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--physics", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--bank-index",
        type=int,
        default=None,
        help="Replay this one bank entry in every env (single-pose curriculum) "
        "instead of the x/y/yaw grid",
    )
    parser.add_argument("--repeats", type=int, default=250,
                        help="Envs to run when --bank-index is given")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Config override applied after config.json, e.g. rewards.x=1",
    )
    return parser.parse_args()


def median(values):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return 0.5 * (float(ordered[middle - 1]) + float(ordered[middle]))


def reset_to_pose(env, env_ids, reference_indices, transform_indices=None,
                  episode_translation=None, episode_yaw_rad=None):
    """Reset every env off-cycle and refresh the observation.

    ``MotionImitationEnv.reset_idx`` only writes the new state into the asset
    buffers; ``reset_all`` is what pushes it to the simulation and rebuilds the
    observation. This mirrors that tail (motion_imitation_env.py:1362) for a
    reset that carries a caller-chosen pose.
    """
    inner = env.unwrapped
    inner.reset_idx(
        env_ids,
        reference_indices,
        transform_indices=transform_indices,
        episode_translation=episode_translation,
        episode_yaw_rad=episode_yaw_rad,
    )
    inner.scene.write_data_to_sim()
    inner.sim.forward()
    inner.scene.update(dt=inner.physics_dt)
    inner.compute_observations()
    inner.obs_buf = inner._observation_dict()
    return inner.policy_obs


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else checkpoint.parent / "config.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            "Training configuration not found: {}. Pass --config explicitly."
            .format(config_path)
        )
    env_cfg, train_cfg = load_saved_configuration(config_path)
    apply_overrides(env_cfg, args.overrides)

    rand = env_cfg.object_randomization
    yaws = torch.linspace(
        float(rand.yaw_min_deg), float(rand.yaw_max_deg), args.yaw_steps
    )
    xs = torch.linspace(
        float(rand.translation_x_min_m), float(rand.translation_x_max_m), args.x_steps
    )
    ys = torch.linspace(
        float(rand.translation_y_min_m), float(rand.translation_y_max_m), args.y_steps
    )
    grid = [
        (float(yaw), float(x), float(y)) for yaw in yaws for x in xs for y in ys
    ]
    if args.bank_index is not None:
        grid = [None] * int(args.repeats)
    num_envs = len(grid)
    if num_envs <= 0:
        raise ValueError("No poses to run: check --repeats / --*-steps")

    env_cfg.seed = int(args.seed)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.play = True
    env_cfg.viewer.enable_viewer = False
    env_cfg.viewer.reference_ghost = False
    env_cfg.viewer.training_camera_enabled = False
    # Every episode must run to the final reference sample, so the tracking
    # threshold is measured rather than enforced.
    env_cfg.termination.enabled = False
    env_cfg.object_assist.enabled = False
    train_cfg.runner.record_video = False

    env = make_env(
        env_cfg,
        num_envs=None,
        device=args.sim_device,
        physics=args.physics,
        visualizer=None,
    )
    try:
        max_start = int(env.reference.last_index - 1)
        if not 0 <= int(args.rsi_index) <= max_start:
            raise ValueError(
                "--rsi-index must lie in [0, {}], got {}".format(
                    max_start, args.rsi_index
                )
            )
        env.max_episode_length = int(env.reference.last_index)
        env.cfg.env.episode_length = env.max_episode_length

        runner = PPO(env, train_cfg, log_dir=None, device=env.device)
        runner.load(checkpoint, load_optimizer=False, load_normalizers=True)
        policy = runner.get_inference_policy(device=env.device)

        device = env.device
        inner = env.unwrapped
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        reference = torch.full(
            (num_envs,), int(args.rsi_index), device=device, dtype=torch.long
        )
        if args.bank_index is not None:
            index = torch.full(
                (num_envs,), int(args.bank_index), device=device, dtype=torch.long
            )
            observations = reset_to_pose(
                env, env_ids, reference, transform_indices=index
            )
        else:
            translation = torch.tensor(
                [(x, y, 0.0) for (_, x, y) in grid], device=device, dtype=torch.float32
            )
            yaw_rad = torch.tensor(
                [math.radians(yaw) for (yaw, _, _) in grid],
                device=device,
                dtype=torch.float32,
            )
            observations = reset_to_pose(
                env,
                env_ids,
                reference,
                episode_translation=translation,
                episode_yaw_rad=yaw_rad,
            )
        # What the environment actually placed. It is the request unless the
        # run snaps the placement onto the bank entry (rsi_snap_placement_from_index),
        # in which case the realised pose is the nearest bank transform.
        placed_translation = inner.episode_translation[env_ids].clone()
        placed_yaw_rad = inner.episode_yaw_rad[env_ids].clone()
        if args.bank_index is not None:
            grid = [
                (
                    math.degrees(float(placed_yaw_rad[i])),
                    float(placed_translation[i, 0]),
                    float(placed_translation[i, 1]),
                )
                for i in range(num_envs)
            ]

        steps = int(env.reference.last_index) - int(args.rsi_index)
        max_error = torch.zeros(num_envs, device=device)
        peak_lift = torch.full((num_envs,), -1.0, device=device)
        fail_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        blown = torch.zeros(num_envs, device=device, dtype=torch.bool)
        contact_steps = torch.zeros(num_envs, device=device)
        sum_ee_rate = torch.zeros(num_envs, device=device)
        threshold = float(env_cfg.termination.object_position_threshold_m)
        step = -1
        last = None
        print(
            "Checkpoint: {}\nConfiguration: {}\n{} pose(s) from reference index "
            "{} to {}: {} transitions".format(
                checkpoint, config_path, num_envs, args.rsi_index,
                int(env.reference.last_index), steps,
            )
        )
        with torch.inference_mode():
            for step in range(steps):
                actions = policy(observations)
                observations, _, _, dones, infos = env.step(actions)
                err = infos["object_position_error_m"]
                newly = (err > threshold) & (fail_step < 0)
                fail_step[newly] = int(args.rsi_index) + step
                max_error = torch.maximum(max_error, err)
                peak_lift = torch.maximum(peak_lift, infos["object_com_lift_m"])
                contact_steps += infos["fingertip_contact_fraction"]
                sum_ee_rate += infos["rms_ee_action_rate"]
                last = infos
                if step < steps - 1:
                    # Termination is off, so a done before the reference end
                    # is the solver blow-up guard resetting that one env (to a
                    # random RSI: its trajectory is worthless from here). Mark
                    # it and keep rolling the others; the reference end arrives
                    # for every env on the same, last step.
                    blown |= dones.to(dtype=torch.bool)
        if last is None:
            raise RuntimeError("No steps were taken: nothing to report")

        taken = step + 1
        rows = []
        for i, (yaw, x, y) in enumerate(grid):
            rows.append(
                {
                    "blown": bool(blown[i]),
                    "yaw_deg": yaw,
                    "x_m": x,
                    "y_m": y,
                    "placed_yaw_deg": math.degrees(float(placed_yaw_rad[i])),
                    "placed_x_m": float(placed_translation[i, 0]),
                    "placed_y_m": float(placed_translation[i, 1]),
                    "max_object_error_m": float(max_error[i]),
                    "failed": bool(fail_step[i] >= 0),
                    "fail_reference_index": int(fail_step[i]),
                    "peak_lift_m": float(peak_lift[i]),
                    "final_object_error_m": float(last["object_position_error_m"][i]),
                    "final_orientation_error_rad": float(
                        last["object_orientation_error_rad"][i]
                    ),
                    "contact_fraction": float(contact_steps[i] / max(taken, 1)),
                    "mean_rms_ee_action_rate": float(sum_ee_rate[i] / max(taken, 1)),
                }
            )

        # Rows of blown envs are kept for the record but excluded from every
        # statistic: the summary describes the policy, not the solver.
        all_rows = rows
        rows = [r for r in all_rows if not r["blown"]] or all_rows
        failed_rows = [r for r in rows if r["failed"]]
        summary = {
            "blown_count": sum(r["blown"] for r in all_rows),
            "fail_fraction": len(failed_rows) / len(rows),
            # Continuous companions of the pass/fail gate: the policies sit
            # close to the 7 cm threshold, so a run can swing from 0 to 30%
            # passes on a 2 cm difference. Compare variants on these too.
            "median_max_object_error_m": median([r["max_object_error_m"] for r in rows]),
            "fail_fraction_10cm": sum(r["max_object_error_m"] > 0.10 for r in rows) / len(rows),
            "fail_fraction_15cm": sum(r["max_object_error_m"] > 0.15 for r in rows) / len(rows),
            "median_peak_lift_m": median([r["peak_lift_m"] for r in rows]),
            "median_final_orientation_error_rad": median(
                [r["final_orientation_error_rad"] for r in rows]
            ),
            "mean_contact_fraction": sum(r["contact_fraction"] for r in rows) / len(rows),
            "mean_rms_ee_action_rate": sum(
                r["mean_rms_ee_action_rate"] for r in rows
            ) / len(rows),
            "median_fail_reference_index": median(
                [r["fail_reference_index"] for r in failed_rows]
            ) if failed_rows else -1.0,
        }

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "checkpoint": str(checkpoint),
                    "rsi_index": int(args.rsi_index),
                    "threshold_m": threshold,
                    "steps": taken,
                    "bank_index": args.bank_index,
                    "summary": summary,
                    "rows": all_rows,
                },
                handle,
                indent=1,
            )

        print(
            "{} poses, {} failed ({:.0%}), mean peak lift {:.3f} m".format(
                len(rows),
                len(failed_rows),
                len(failed_rows) / len(rows),
                sum(r["peak_lift_m"] for r in rows) / len(rows),
            )
        )

        def table(key, values):
            print("\n  by {}:".format(key))
            for v in values:
                sel = [r for r in rows if abs(r[key] - v) < 1e-6]
                if not sel:
                    continue
                f = sum(r["failed"] for r in sel) / len(sel)
                lift = sum(r["peak_lift_m"] for r in sel) / len(sel)
                ori = sum(r["final_orientation_error_rad"] for r in sel) / len(sel)
                print(
                    "    {:>7.2f}: fail {:>4.0%}  lift {:.3f}  final ori err {:.2f} rad".format(
                        v, f, lift, ori
                    )
                )

        if args.bank_index is not None:
            # One pose: the per-axis tables are empty by construction.
            print(
                "  bank entry {}: yaw {:.2f} deg, x {:+.4f} m, y {:+.4f} m".format(
                    args.bank_index, grid[0][0], grid[0][1], grid[0][2]
                )
            )
            print(
                "  median fail reference index: {}".format(
                    "none (no failure)" if not failed_rows
                    else "{:.1f}".format(summary["median_fail_reference_index"])
                )
            )
        else:
            table("yaw_deg", [float(v) for v in yaws])
            table("x_m", [float(v) for v in xs])
            table("y_m", [float(v) for v in ys])
        print("\n  output: {}".format(output))
    finally:
        env.close()


if __name__ == "__main__":
    main()
