"""Per-episode cuboid scale (branch generalize_size, 2026-09-18).

The bar keeps its proportions and is scaled by one factor ``s`` drawn per
episode from ``[scale_min, scale_max]``. Everything else -- the transform bank,
the demonstration, the reward terms -- stays as it is; the environment only

* writes the scaled half extents into the solver (Newton ``shape_scale``,
  refreshed per world by the MuJoCo-Warp backend),
* scales mass with the volume (``s^3``) and the inertia with ``s^5`` so the
  bar stays the same material,
* lifts the reference cuboid pose by ``half_height * (s - 1)`` so a scaled bar
  still rests on the table where the demonstration's bar rested,
* and, when asked, appends ``s`` to the policy observation.

The helpers here are pure so they can be unit-tested without a simulator.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def scale_range(randomization_cfg) -> Tuple[float, float]:
    low = float(getattr(randomization_cfg, "scale_min", 1.0))
    high = float(getattr(randomization_cfg, "scale_max", 1.0))
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError("object_randomization.scale_min/scale_max must be finite")
    if low <= 0.0 or high <= 0.0:
        raise ValueError("object_randomization.scale_min/scale_max must be positive")
    if low > high:
        raise ValueError("object_randomization.scale_min must not exceed scale_max")
    return low, high


def scale_randomization_enabled(randomization_cfg) -> bool:
    low, high = scale_range(randomization_cfg)
    return abs(low - 1.0) > 1e-9 or abs(high - 1.0) > 1e-9


def observe_scale(randomization_cfg) -> bool:
    return bool(getattr(randomization_cfg, "observe_scale", False))


def object_scale_observation_dim(randomization_cfg) -> int:
    """How many entries the scale adds to the policy observation (0 or 1)."""
    return 1 if observe_scale(randomization_cfg) else 0


def sample_scales(count: int, randomization_cfg, device, generator=None) -> torch.Tensor:
    low, high = scale_range(randomization_cfg)
    if count <= 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    if abs(high - low) <= 1e-12:
        return torch.full((count,), low, dtype=torch.float32, device=device)
    draw = torch.rand(count, dtype=torch.float32, device=device, generator=generator)
    scales = low + (high - low) * draw
    nominal = nominal_probability(randomization_cfg)
    if nominal > 0.0:
        pick = torch.rand(count, dtype=torch.float32, device=device, generator=generator) < nominal
        scales = torch.where(pick, torch.ones_like(scales), scales)
    return scales


def nominal_probability(randomization_cfg) -> float:
    """Fraction of episodes that keep the nominal bar (scale 1.0) -- the scale anchor."""
    value = float(getattr(randomization_cfg, "scale_nominal_probability", 0.0))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("object_randomization.scale_nominal_probability must lie in [0, 1]")
    return value


def mass_factor(scale: torch.Tensor, with_volume: bool) -> torch.Tensor:
    """Mass multiplier for a bar of the same material: ``s^3`` (or 1)."""
    if not with_volume:
        return torch.ones_like(scale)
    return scale.pow(3)


def inertia_factor(scale: torch.Tensor, with_volume: bool) -> torch.Tensor:
    """Inertia multiplier: ``s^5`` with the volume (mass ``s^3`` times ``s^2``), else ``s^2``."""
    return scale.pow(5) if with_volume else scale.pow(2)


def reference_height_shift(scale: torch.Tensor, half_height_m: float) -> torch.Tensor:
    """How much higher the centre of a scaled bar sits when it rests on the table."""
    return float(half_height_m) * (scale - 1.0)


def scale_observation(scale: torch.Tensor) -> torch.Tensor:
    """The observation column: the raw factor, one per environment."""
    return scale.reshape(-1, 1)


def expand_first_layer(weight: torch.Tensor, extra: int) -> torch.Tensor:
    """Append ``extra`` zero input columns to a ``(out, in)`` weight matrix.

    Zero columns keep the network's function identical on the old inputs, so
    a checkpoint trained without the scale observation starts the widened run
    as exactly the policy it was.
    """
    if weight.ndim != 2:
        raise ValueError("Expected a 2-D weight matrix")
    if extra <= 0:
        return weight.clone()
    pad = torch.zeros((weight.shape[0], extra), dtype=weight.dtype, device=weight.device)
    return torch.cat((weight, pad), dim=1)


def expand_normalizer_row(values: torch.Tensor, extra: int, fill: float) -> torch.Tensor:
    """Append ``extra`` entries with value ``fill`` to a ``(1, in)`` statistics row."""
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError("Expected a (1, in) statistics row")
    if extra <= 0:
        return values.clone()
    pad = torch.full((1, extra), float(fill), dtype=values.dtype, device=values.device)
    return torch.cat((values, pad), dim=1)


def uniform_scale_statistics(randomization_cfg) -> Tuple[float, float]:
    """Mean and variance of the uniform scale draw, for a normalizer's new column."""
    low, high = scale_range(randomization_cfg)
    mean = 0.5 * (low + high)
    variance = (high - low) ** 2 / 12.0
    return mean, max(variance, 1e-4)
