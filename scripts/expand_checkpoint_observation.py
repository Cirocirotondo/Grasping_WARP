#!/usr/bin/env python3
"""Widen a checkpoint's observation input so it can warm-start a run that observes the cuboid scale.

    python scripts/expand_checkpoint_observation.py \
        --checkpoint logs/staged/w6_s7_cont2_it17000/model_17000.pt \
        --output logs/staged/w6_s7_cont2_it17000_scale/model_17000.pt \
        --scale-min 0.8 --scale-max 1.2

The actor's and the critic's first linear layer get one extra input column
of zeros (the policy is unchanged on the old inputs), the observation
normalizers get one extra column with the mean and variance of the uniform
scale draw, and the Adam moments of those two layers are widened with zeros
so ``--load-optimizer`` resumes cleanly. Everything else is copied.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from simtoolreal_newton.envs.object_scale import (
    expand_first_layer,
    expand_normalizer_row,
    uniform_scale_statistics,
)

FIRST_LAYER_KEYS = ("policy_latent_net.0.weight", "value.0.weight")
NORMALIZER_KEYS = ("actor_obs_normalizer", "critic_obs_normalizer")


class _Range:
    def __init__(self, low, high):
        self.scale_min = float(low)
        self.scale_max = float(high)


def expand_checkpoint(checkpoint: dict, extra: int, mean: float, variance: float) -> dict:
    """Return a widened copy of ``checkpoint`` (see the module docstring)."""
    if extra <= 0:
        raise ValueError("extra must be positive")
    out = dict(checkpoint)
    widened_shapes = {}
    for section, key in (("policy_dict", FIRST_LAYER_KEYS[0]), ("value_dict", FIRST_LAYER_KEYS[1])):
        state = dict(out[section])
        weight = state[key]
        widened_shapes[tuple(weight.shape)] = extra
        state[key] = expand_first_layer(weight, extra)
        out[section] = state
    std = float(variance) ** 0.5
    for key in NORMALIZER_KEYS:
        stats = checkpoint.get(key)
        if stats is None:
            continue
        stats = dict(stats)
        stats["_mean"] = expand_normalizer_row(stats["_mean"], extra, mean)
        stats["_var"] = expand_normalizer_row(stats["_var"], extra, variance)
        if "_std" in stats:
            stats["_std"] = expand_normalizer_row(stats["_std"], extra, std)
        out[key] = stats
    optimizer = checkpoint.get("optimizer_state_dict")
    if optimizer is not None:
        optimizer = {"state": dict(optimizer["state"]), "param_groups": optimizer["param_groups"]}
        for index, entry in list(optimizer["state"].items()):
            entry = dict(entry)
            for moment in ("exp_avg", "exp_avg_sq"):
                tensor = entry.get(moment)
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tuple(tensor.shape) in widened_shapes:
                    entry[moment] = expand_first_layer(tensor, extra)
            optimizer["state"][index] = entry
        out["optimizer_state_dict"] = optimizer
    infos = dict(out.get("infos", {}))
    infos["observation_expanded_by"] = int(extra) + int(infos.get("observation_expanded_by", 0))
    out["infos"] = infos
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra", type=int, default=1, help="extra observation entries (default 1: the scale)")
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=1.2)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mean, variance = uniform_scale_statistics(_Range(args.scale_min, args.scale_max))
    widened = expand_checkpoint(checkpoint, args.extra, mean, variance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(widened, args.output)
    # A staged seed travels with its config; carry it over and record the new width.
    config_src = args.checkpoint.parent / "config.json"
    if config_src.exists():
        config = json.loads(config_src.read_text())
        config["observation_dim"] = int(config.get("observation_dim", 0)) + args.extra
        config.setdefault("lineage", {})["observation_expanded_from"] = str(args.checkpoint)
        (args.output.parent / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    else:
        for name in ("config.json",):
            src = args.checkpoint.parent / name
            if src.exists():
                shutil.copy(src, args.output.parent / name)
    width = widened["policy_dict"][FIRST_LAYER_KEYS[0]].shape[1]
    print("Wrote {} (policy input width {}, normalizer column mean {:.3f} var {:.4f})".format(
        args.output, width, mean, variance))


if __name__ == "__main__":
    main()
