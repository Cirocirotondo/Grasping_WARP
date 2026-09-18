#!/usr/bin/env python3
"""Print the ``--set PATH=VALUE`` flags that reproduce a run's configuration.

    python3 scripts/config_diff_flags.py logs/staged/<seed>/config.json [--output flags.txt]

A warm start (``--seed-checkpoint``) does not inherit the seed's config: every
non-default value has to be passed again. This compares the saved
``env_cfg``/``train_cfg`` with the code defaults and prints one ``--set`` per
differing leaf, skipping the per-launch fields (env count, seed, run name,
devices, iteration budget, logging). Values are JSON, the form
``scripts/train.py --set`` expects.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from simtoolreal_newton.cfg import SimToolRealCfg, SimToolRealTrainCfg  # noqa: E402
from train import config_to_dict  # noqa: E402

# Chosen per launch, never part of a recipe.
SKIP_PREFIXES = (
    "env.num_envs",
    "env.play",
    "seed",
    "sim.device",
    "sim.physics",
    "sim.physx",
    "sim.use_cuda_graph",
    "viewer",
    "train.runner.run_name",
    "train.runner.experiment_name",
    "train.runner.max_iterations",
    "train.runner.save_interval",
    "train.runner.record_",
    "train.runner.tensorboard",
    "train.runner.wandb",
    "train.runner.resume",
    "train.runner.load_run",
    "train.runner.checkpoint",
    "train.runner.log",
    "train.seed",
    "train.runner.evaluation_num_envs",
    "train.runner.evaluation_seed",
    "train.runner.evaluation_interval",
)


def flatten(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from flatten(value, prefix + key + ".")
    else:
        yield prefix[:-1], node


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="write the flags, one per line")
    parser.add_argument("--include-launch", action="store_true", help="do not skip the per-launch fields")
    args = parser.parse_args()

    saved = json.loads(args.config.read_text())
    # Instances: the nested sections are classes (callable) until instantiated.
    defaults = {"env": config_to_dict(SimToolRealCfg()), "train": config_to_dict(SimToolRealTrainCfg())}
    current = {"env": saved["env_cfg"], "train": saved["train_cfg"]}
    flags = []
    for section, prefix in (("env", ""), ("train", "train.")):
        default_leaves = dict(flatten(defaults[section]))
        for path, value in flatten(current[section]):
            full = prefix + path
            if not args.include_launch and any(full.startswith(p) for p in SKIP_PREFIXES):
                continue
            if path not in default_leaves:
                continue  # a field the saved config has and the code no longer knows (or vice versa)
            if default_leaves[path] == value:
                continue
            flags.append("--set {}={}".format(full, json.dumps(value, separators=(",", ":"))))
    text = "\n".join(flags)
    if args.output:
        args.output.write_text(text + "\n")
        print("{} flags -> {}".format(len(flags), args.output))
    else:
        print(text)


if __name__ == "__main__":
    main()
