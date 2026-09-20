"""Saved diagnostics for one MuJoCo sim2sim rollout (joint tracking, actions, bar)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .constants import ACTION_DIM, ARM_JOINT_NAMES, HAND_JOINT_NAMES

TRACE_KEYS = (
    "reference_indices",
    "policy_actions",
    "filtered_actions",
    "actual_joint_positions",
    "commanded_targets",
    "applied_targets",
    "reference_joint_positions",
)
SCALAR_KEYS = ("cube_position_error_m", "cube_lift_m", "hand_q_error_rad", "ik_residual")


def _validated_arrays(trace: Mapping[str, Sequence]) -> dict:
    missing = sorted(set(TRACE_KEYS) - set(trace))
    if missing:
        raise ValueError("Rollout trace is missing fields: {}".format(missing))
    arrays = {key: np.asarray(trace[key]) for key in TRACE_KEYS}
    frames = arrays["reference_indices"]
    if frames.ndim != 1 or frames.size == 0:
        raise ValueError("reference_indices must be a non-empty 1-D array")
    for key in TRACE_KEYS[1:]:
        if arrays[key].shape != (frames.size, ACTION_DIM):
            raise ValueError(
                "{} has shape {}, expected ({}, {})".format(key, arrays[key].shape, frames.size, ACTION_DIM)
            )
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError("{} contains non-finite values".format(key))
    for key in SCALAR_KEYS:
        if key in trace:
            values = np.asarray(trace[key], dtype=np.float64)
            if values.shape != (frames.size,):
                raise ValueError("{} must have one value per frame".format(key))
            arrays[key] = values
    return arrays


def _plot_joint_grid(plt, frames, series, joint_names, title, ylabel, rows, columns):
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.25 * rows), sharex=True, layout="constrained")
    axes = np.asarray(axes).reshape(-1)
    for joint_index, name in enumerate(joint_names):
        axis = axes[joint_index]
        for label, values, style in series:
            axis.plot(frames, values[:, joint_index], style, linewidth=1.15, label=label)
        axis.set_title(name.replace("_joint", ""), fontsize=9)
        axis.grid(True, alpha=0.28)
        axis.set_ylabel(ylabel, fontsize=8)
        if joint_index == 0:
            axis.legend(loc="best", fontsize=7)
    for axis in axes[len(joint_names) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.supxlabel("demonstration frame")
    return figure


def save_rollout_plots(output_dir: Path, trace: Mapping[str, Sequence], *, show: bool) -> dict:
    """Write ``rollout_data.npz`` and the figures, then optionally show them."""
    arrays = _validated_arrays(trace)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    frames = arrays["reference_indices"]
    action_series = (
        ("policy action", arrays["policy_actions"], "-"),
        ("filtered action (executed)", arrays["filtered_actions"], "--"),
    )
    joint_series = (
        ("measured q", arrays["actual_joint_positions"], "-"),
        ("commanded target", arrays["commanded_targets"], "--"),
        ("applied (slewed) target", arrays["applied_targets"], "-."),
        ("reference q", arrays["reference_joint_positions"], ":"),
    )
    figures = {
        "arm_actions": _plot_joint_grid(
            plt, frames, tuple((l, v[:, :6], s) for l, v, s in action_series), ARM_JOINT_NAMES,
            "Arm actions (palm twist command)", "action", 3, 2,
        ),
        "hand_actions": _plot_joint_grid(
            plt, frames, tuple((l, v[:, 6:], s) for l, v, s in action_series), HAND_JOINT_NAMES,
            "Hand actions (joint residuals)", "action", 5, 4,
        ),
        "arm_joint_tracking": _plot_joint_grid(
            plt, frames, tuple((l, v[:, :6], s) for l, v, s in joint_series), ARM_JOINT_NAMES,
            "Arm joint tracking", "angle [rad]", 3, 2,
        ),
        "hand_joint_tracking": _plot_joint_grid(
            plt, frames, tuple((l, v[:, 6:], s) for l, v, s in joint_series), HAND_JOINT_NAMES,
            "Hand joint tracking", "angle [rad]", 5, 4,
        ),
    }
    scalar = [key for key in SCALAR_KEYS if key in arrays]
    if scalar:
        figure, axes = plt.subplots(len(scalar), 1, figsize=(8.0, 2.2 * len(scalar)), sharex=True, layout="constrained")
        axes = np.asarray(axes).reshape(-1)
        for axis, key in zip(axes, scalar):
            axis.plot(frames, arrays[key], linewidth=1.2)
            axis.set_ylabel(key.replace("_", " "), fontsize=8)
            axis.grid(True, alpha=0.28)
        figure.suptitle("Bar and hand tracking")
        figure.supxlabel("demonstration frame")
        figures["tracking"] = figure
    paths = {"data": output_dir / "rollout_data.npz"}
    np.savez_compressed(paths["data"], **arrays)
    for name, figure in figures.items():
        path = output_dir / "{}.png".format(name)
        figure.savefig(path, dpi=160)
        paths[name] = path
        try:
            figure.canvas.manager.set_window_title(name.replace("_", " "))
        except AttributeError:
            pass
    if show:
        plt.show()
    plt.close("all")
    return paths
