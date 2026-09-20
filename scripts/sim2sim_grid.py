#!/usr/bin/env python3
"""The sim2sim verdict: 25 bar placements per scale, in native MuJoCo.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python scripts/sim2sim_grid.py \\
        --checkpoint logs/staged/<run>/model_<N>.pt --scales 0.8 1.0 1.2 \\
        [--newton-grid logs/simtoolreal/<run>/sweep_grid_rsi0_<N>_s{scale}.json] --output out.json

The placements are a fixed, stratified subset of the 250-pose grid
``scripts/sweep_pose_success.py`` plays in the training simulator (five y
bands x five yaws, x cycling through its five values), so the same poses are
compared across checkpoints and, with ``--newton-grid``, against the training
simulator pose by pose. Per scale it reports the fraction above the 7 cm gate
(the sweep's criterion), the bar never lifted 5 cm (no grasp), solver
blow-ups and the median of the largest bar error.
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

X_VALUES = (-0.09, -0.045, 0.0, 0.045, 0.09)
Y_VALUES = (0.0, 0.0375, 0.075, 0.1125, 0.15)
YAW_VALUES_DEG = (-22.5, -7.5, 7.5, 22.5, 45.0)
# Stratified 25 of the sweep's 250: every y band against every listed yaw,
# with x walking through its five values so no band repeats an x.
PLACEMENTS = tuple(
    (X_VALUES[(i + j) % 5], Y_VALUES[i], YAW_VALUES_DEG[j]) for i in range(5) for j in range(5)
)
GATE_M = 0.07
LIFT_M = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    parser.add_argument("--rsi-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--newton-grid", default=None,
        help="Pattern of the training simulator's grid JSON per scale, with {scale} (e.g. ..._s{scale}.json).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    return parser.parse_args()


def newton_lookup(pattern, scale):
    if pattern is None:
        return {}
    path = Path(pattern.format(scale="{:g}".format(scale)))
    if not path.is_file():
        return {}
    rows = json.load(path.open())["rows"]
    return {(round(r["x_m"], 4), round(r["y_m"], 4), round(r["yaw_deg"], 2)): r for r in rows}


def main():
    args = parse_args()
    run = load_saved_run(args.checkpoint, args.config, args.overrides)
    track = ReferenceTrack.load(run.env_cfg)
    policy = InferencePolicy(run, device=args.device)
    report = {"checkpoint": str(run.checkpoint_path), "placements": PLACEMENTS, "scales": {}}
    for scale in args.scales:
        newton = newton_lookup(args.newton_grid, scale)
        sim = MujocoSim(MujocoSceneConfig.from_env_cfg(run.env_cfg, object_scale=scale))
        pipeline = ActionPipeline(run.env_cfg, sim.joint_lower_limits, sim.joint_upper_limits, sim.joint_velocity_limits)
        started = time.time()
        rows = []
        for x, y, yaw_deg in PLACEMENTS:
            translation = np.asarray((x, y, 0.0))
            yaw = float(np.deg2rad(yaw_deg))
            spec = EpisodeSpec(
                transform_index=track.nearest_transform(translation[:2], yaw), rsi_index=args.rsi_index,
                scale=scale, episode_translation=translation, episode_yaw_rad=yaw,
            )
            result = run_episode(run, policy, track, sim, pipeline, spec)
            row = {"x_m": x, "y_m": y, "yaw_deg": yaw_deg, **result.summary()}
            reference = newton.get((round(x, 4), round(y, 4), round(yaw_deg, 2)))
            if reference is not None:
                row["newton_max_object_error_m"] = reference["max_object_error_m"]
                row["newton_peak_lift_m"] = reference["peak_lift_m"]
            rows.append(row)
        sim.close()
        n = len(rows)
        failed = sum(r["blown"] or r["max_object_error_m"] > GATE_M for r in rows)
        no_grasp = sum(r["peak_lift_m"] < LIFT_M for r in rows)
        blown = sum(r["blown"] for r in rows)
        median = float(np.median([r["max_object_error_m"] for r in rows]))
        summary = {
            "fail_fraction": failed / n, "failed": failed, "no_grasp": no_grasp, "blown": blown,
            "median_max_object_error_m": median, "count": n,
        }
        line = "scale {:.2f}: fail@7 {}/{} ({:.3f}), no-grasp {}, blown {}, median max error {:.1f} cm".format(
            scale, failed, n, failed / n, no_grasp, blown, 100 * median
        )
        compared = [r for r in rows if "newton_max_object_error_m" in r]
        if compared:
            newton_failed = sum(r["newton_max_object_error_m"] > GATE_M for r in compared)
            agree = sum((r["newton_max_object_error_m"] > GATE_M) == (r["blown"] or r["max_object_error_m"] > GATE_M) for r in compared)
            summary.update({"newton_failed": newton_failed, "agreement": agree, "compared": len(compared)})
            line += " | training sim on the same poses: fail@7 {}/{}, same verdict {}/{}".format(
                newton_failed, len(compared), agree, len(compared)
            )
        print(line + "  [{:.0f} s]".format(time.time() - started), flush=True)
        report["scales"]["{:g}".format(scale)] = {"summary": summary, "rows": rows}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, args.output.open("w"), indent=1)
        print("output: {}".format(args.output))


if __name__ == "__main__":
    main()
