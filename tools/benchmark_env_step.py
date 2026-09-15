"""Time the environment step alone, with the policy held at zero action.

    PY=deps/IsaacLab/.venv/bin/python
    $PY tools/benchmark_env_step.py --num-envs 4096 --steps 200 --trials 3

Reports the median milliseconds per environment step and env-steps per second.
Run it on an idle GPU: a concurrent training job makes every number noisy.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simtoolreal_newton.cfg import SimToolRealCfg  # noqa: E402
from simtoolreal_newton.launch import add_env_arguments, make_env  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200, help="Timed steps per trial.")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed steps before each trial.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", type=str, default=None, help="Also write the result to this file.")
    add_env_arguments(parser)
    return parser.parse_args()


def main():
    import torch

    args = parse_args()
    cfg = SimToolRealCfg()
    cfg.seed = int(args.seed)
    env = make_env(cfg, num_envs=args.num_envs, device=args.sim_device, physics=args.physics, visualizer=args.viz)
    num_envs = int(env.num_envs)
    actions = torch.zeros(num_envs, env.num_actions, device=env.device)
    trial_ms = []
    with torch.inference_mode():
        env.reset()
        for trial in range(int(args.trials)):
            for _ in range(int(args.warmup)):
                env.step(actions)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(int(args.steps)):
                env.step(actions)
            torch.cuda.synchronize()
            ms = 1000.0 * (time.perf_counter() - t0) / float(args.steps)
            trial_ms.append(ms)
            print("trial {}: {:.3f} ms/step".format(trial, ms), flush=True)
    env.close()
    median_ms = statistics.median(trial_ms)
    result = {
        "num_envs": num_envs,
        "steps": int(args.steps),
        "trial_ms": trial_ms,
        "median_ms": median_ms,
        "env_steps_per_s": 1000.0 * num_envs / median_ms,
    }
    print("RESULT " + json.dumps(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
