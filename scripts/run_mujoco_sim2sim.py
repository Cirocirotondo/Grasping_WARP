#!/usr/bin/env python3
"""Run a Newton-trained checkpoint in native MuJoCo (sim2sim).

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python scripts/run_mujoco_sim2sim.py \\
        --checkpoint logs/staged/<run>/model_<N>.pt [--bank-index 122] [--rsi-index 0] \\
        [--cube-scale 1.2] [--headless --no-realtime] [--set sim.mjwarp.contact_solref=[0.02,1.0]]

The scene, the drives, the contact model and the collision graph come from
the run's ``config.json`` (see ``simtoolreal_newton/sim2sim/``); the policy
sees the same 112/113-D observation and drives the same 26 targets as in
training. Each episode reports what ``scripts/sweep_pose_success.py`` reports
in the training simulator: the largest bar position error along the way (the
7 cm gate), the peak lift and the final pose errors, so the two simulators
can be compared number for number.

Placement: ``--bank-index`` (repeatable) picks transform-bank entries; ``--x
--y --yaw`` instead requests one continuous planar placement served by its
nearest bank entry, as the training reset does. ``--repeats`` re-runs each
placement; native MuJoCo is deterministic, so repeats only differ with
``--seed``-driven noise, which this runner does not add.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simtoolreal_newton.sim2sim.controller import ActionPipeline  # noqa: E402
from simtoolreal_newton.sim2sim.mujoco_sim import MujocoSceneConfig, MujocoSim  # noqa: E402
from simtoolreal_newton.sim2sim.policy import InferencePolicy, load_saved_run  # noqa: E402
from simtoolreal_newton.sim2sim.reference import ReferenceTrack  # noqa: E402
from simtoolreal_newton.sim2sim.rollout import EpisodeSpec, run_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Defaults to config.json beside the checkpoint.")
    parser.add_argument("--device", default="cpu", help="Actor device: cpu or cuda.")
    parser.add_argument("--rsi-index", type=int, default=0)
    parser.add_argument(
        "--bank-index", type=int, nargs="+", default=None,
        help="Transform bank entries to play (default: 122, the demonstration's own pose).",
    )
    parser.add_argument("--x", type=float, default=None, help="Continuous placement: bar x offset [m].")
    parser.add_argument("--y", type=float, default=None, help="Continuous placement: bar y offset [m].")
    parser.add_argument("--yaw", type=float, default=None, help="Continuous placement: bar yaw [deg].")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cube-scale", type=float, default=1.0, help="Bar scale factor (1.0 = 15 x 5 x 5 cm).")
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs to the end of the demonstration.")
    parser.add_argument("--terminate", action="store_true", help="Stop when the training termination would fire.")
    parser.add_argument("--gate-m", type=float, default=0.07, help="Bar error above which an episode counts as failed.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--start-delay-seconds", type=float, default=2.0)
    parser.add_argument("--no-reference-ghost", "--no-ghost", dest="reference_ghost", action="store_false")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Default: sim2sim_plots/<episode> beside the checkpoint.")
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--no-show-plots", action="store_true")
    parser.add_argument("--print-every", type=int, default=60)
    parser.add_argument("--output", type=Path, default=None, help="Write the per-episode metrics as JSON.")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE",
        help="Override an env configuration field (e.g. sim.mjwarp.contact_solref=[0.02,1.0]). Repeatable.",
    )
    args = parser.parse_args()
    if args.rsi_index < 0 or args.max_steps < 0 or args.repeats < 1 or args.print_every < 0:
        parser.error("indices, steps, repeats and print intervals must be non-negative (repeats >= 1)")
    continuous = [v is not None for v in (args.x, args.y, args.yaw)]
    if any(continuous) and not all(continuous):
        parser.error("--x, --y and --yaw must be given together")
    if all(continuous) and args.bank_index is not None:
        parser.error("--bank-index and --x/--y/--yaw are alternatives")
    return args


def main() -> None:
    args = parse_args()
    run = load_saved_run(args.checkpoint, args.config, args.overrides)
    env_cfg = run.env_cfg
    track = ReferenceTrack.load(env_cfg)
    policy = InferencePolicy(run, device=args.device)
    scene = MujocoSceneConfig.from_env_cfg(
        env_cfg,
        object_scale=args.cube_scale,
        enable_viewer=not args.headless,
        enable_reference_ghost=args.reference_ghost and not args.headless,
    )
    if all(v is not None for v in (args.x, args.y, args.yaw)):
        translation = np.asarray((args.x, args.y, 0.0), dtype=np.float64)
        yaw = float(np.deg2rad(args.yaw))
        placements = [(track.nearest_transform(translation[:2], yaw), translation, yaw)]
    else:
        placements = [(int(index), None, None) for index in (args.bank_index or [122])]

    print(
        "MuJoCo sim2sim: checkpoint={} observations={} actions=26 scale={:.2f} "
        "({}observed) RSI={} control={:.0f} Hz physics={:.0f} Hz ({} substeps) "
        "self_collision={} contacts: solref={} solimp={} cone={} impratio={} condim={}".format(
            run.checkpoint_path, run.observation_dim, args.cube_scale,
            "" if run.observes_scale else "not ", args.rsi_index, 1.0 / scene.control_dt,
            1.0 / scene.sim_dt, scene.substeps, scene.self_collision, scene.contact_solref,
            scene.contact_solimp, scene.cone, scene.impratio, scene.contact_condim,
        )
    )
    results = []
    with MujocoSim(scene) as sim:
        pipeline = ActionPipeline(env_cfg, sim.joint_lower_limits, sim.joint_upper_limits, sim.joint_velocity_limits)
        for transform_index, translation, yaw in placements:
            for repeat in range(args.repeats):
                spec = EpisodeSpec(
                    transform_index=transform_index,
                    rsi_index=args.rsi_index,
                    scale=args.cube_scale,
                    episode_translation=translation,
                    episode_yaw_rad=yaw,
                    max_steps=args.max_steps,
                    terminate=args.terminate,
                )
                bank_t = track.bank.translation[transform_index]
                print(
                    "episode: bank entry {} (x {:+.3f}, y {:+.3f} m, yaw {:+.1f} deg){} repeat {}/{}".format(
                        transform_index, float(bank_t[0]), float(bank_t[1]),
                        float(np.rad2deg(float(track.bank.yaw_rad[transform_index]))),
                        "" if translation is None else " serving x {:+.3f} y {:+.3f} yaw {:+.1f}".format(
                            translation[0], translation[1], np.rad2deg(yaw)
                        ),
                        repeat + 1, args.repeats,
                    ),
                    flush=True,
                )
                if not args.headless and args.start_delay_seconds and not results:
                    deadline = time.perf_counter() + args.start_delay_seconds
                    while sim.viewer_is_running() and time.perf_counter() < deadline:
                        sim.sync_viewer()
                        time.sleep(0.01)
                result = run_episode(
                    run, policy, track, sim, pipeline, spec,
                    realtime=not args.no_realtime and not args.headless,
                    print_every=args.print_every,
                )
                results.append(result)
                print(
                    "  -> {} steps to ref {}: max bar error {:.3f} m ({}), final {:.3f} m / {:.3f} rad, "
                    "peak lift {:.3f} m, final lift {:.3f} m, max|hand-ref| {:.3f} rad{}{}".format(
                        result.steps, result.final_reference_index, result.max_object_error_m,
                        "FAIL" if result.failed(args.gate_m) else "pass @ {:.0f} cm".format(100 * args.gate_m),
                        result.final_object_error_m, result.final_orientation_error_rad,
                        result.peak_lift_m, result.final_lift_m, result.max_hand_q_error_rad,
                        "" if result.first_object_violation_index < 0
                        else ", termination would fire at {}".format(result.first_object_violation_index),
                        " BLOWN" if result.blown else "",
                    ),
                    flush=True,
                )
                if args.plots and result.trace["reference_indices"]:
                    from simtoolreal_newton.sim2sim.plotting import save_rollout_plots

                    stem = "bank{}_rsi{}_scale{:.2f}".format(transform_index, args.rsi_index, args.cube_scale)
                    if args.repeats > 1:
                        stem += "_r{}".format(repeat)
                    plot_dir = (args.plot_dir or (run.checkpoint_path.parent / "sim2sim_plots")) / stem
                    paths = save_rollout_plots(plot_dir, result.trace, show=not args.headless and not args.no_show_plots)
                    print("  diagnostics: {}".format(paths["data"].parent))
    if len(results) > 1:
        failed = sum(r.failed(args.gate_m) for r in results)
        errors = sorted(r.max_object_error_m for r in results)
        print(
            "{} episodes: {} failed at the {:.0f} cm gate ({:.3f}), median max bar error {:.3f} m, "
            "median peak lift {:.3f} m".format(
                len(results), failed, 100 * args.gate_m, failed / len(results), errors[len(errors) // 2],
                sorted(r.peak_lift_m for r in results)[len(results) // 2],
            )
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "checkpoint": str(run.checkpoint_path),
                    "gate_m": args.gate_m,
                    "overrides": list(args.overrides),
                    "episodes": [r.summary() for r in results],
                },
                stream,
                indent=2,
            )
        print("metrics: {}".format(args.output))


if __name__ == "__main__":
    main()
