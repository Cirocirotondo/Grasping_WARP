"""Rate limits, discontinuity detection and the arming prompt.

``TargetLimiter`` is the hardware's own line of defence, on top of the training
contract's velocity-limit slew: even a target the contract considers legitimate
cannot leave the joint limits or move further than one step allowance per
tick. ``SpikeMonitor`` watches a per-step signal (the raw actions) for a jump no
smooth policy should produce -- an observation branch change, a startup
mismatch, an out-of-distribution state -- and, in ``stop`` mode, aborts before
the target reaches the robot.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np

ARM_DOF = 6
HAND_DOF = 20
ACTION_DIM = ARM_DOF + HAND_DOF


class SafetyAbort(RuntimeError):
    """A safety monitor stopped the rollout."""


class TargetLimiter:
    """Smooth, rate-limit and clip position targets before they are sent.

    Applied in that order: smoothing shapes the trajectory, the step clamp
    bounds per-tick motion, and the limit clip is the final hard bound. The
    clamp is measured against the previously *emitted* target, not the measured
    position, so a policy that ramps away is followed at a bounded rate rather
    than being repeatedly pulled back.
    """

    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        *,
        max_arm_step_rad: float = 0.02,
        max_hand_step_rad: float = 0.05,
        smoothing: float = 0.0,
    ) -> None:
        lower = np.asarray(lower_limits, dtype=np.float64)
        upper = np.asarray(upper_limits, dtype=np.float64)
        if lower.shape != (ACTION_DIM,) or upper.shape != (ACTION_DIM,):
            raise ValueError("Joint limits must have shape (26,)")
        if np.any(lower >= upper):
            raise ValueError("Joint limits must be non-empty intervals")
        if max_arm_step_rad <= 0.0 or max_hand_step_rad <= 0.0:
            raise ValueError("Step limits must be positive")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must lie in [0, 1)")
        self.lower = lower
        self.upper = upper
        self.step_limit = np.concatenate(
            (np.full(ARM_DOF, float(max_arm_step_rad)), np.full(HAND_DOF, float(max_hand_step_rad)))
        )
        self.smoothing = float(smoothing)
        self.previous: Optional[np.ndarray] = None
        self.filtered: Optional[np.ndarray] = None

    def reset(self, initial_target: np.ndarray) -> None:
        target = np.asarray(initial_target, dtype=np.float64)
        if target.shape != (ACTION_DIM,):
            raise ValueError("Initial target must have shape (26,)")
        self.previous = target.copy()
        self.filtered = target.copy()

    def apply(self, requested: np.ndarray) -> tuple:
        requested = np.asarray(requested, dtype=np.float64)
        if requested.shape != (ACTION_DIM,):
            raise ValueError("Requested target must have shape (26,)")
        if not np.all(np.isfinite(requested)):
            raise SafetyAbort("Policy produced a non-finite position target.")
        if self.previous is None or self.filtered is None:
            raise RuntimeError("reset() must be called before apply()")

        if self.smoothing > 0.0:
            self.filtered = self.smoothing * self.filtered + (1.0 - self.smoothing) * requested
            shaped = self.filtered.copy()
        else:
            self.filtered = requested.copy()
            shaped = requested.copy()

        raw_step = shaped - self.previous
        limited_step = np.clip(raw_step, -self.step_limit, self.step_limit)
        stepped = self.previous + limited_step
        final = np.clip(stepped, self.lower, self.upper)

        info = {
            "max_requested_step_rad": float(np.max(np.abs(raw_step))),
            "step_limited_joints": np.flatnonzero(np.abs(raw_step) > self.step_limit + 1e-12),
            "limit_clipped_joints": np.flatnonzero(np.abs(final - stepped) > 1e-12),
        }
        self.previous = final.copy()
        return final, info


class SpikeMonitor:
    """Detect a discontinuity in a per-step signal (raw actions, or targets)."""

    MODES = ("off", "warn", "stop")

    def __init__(
        self,
        threshold: float,
        *,
        mode: str = "warn",
        name: str = "action",
        labels: Optional[list] = None,
        grace_steps: int = 0,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError("mode must be one of {}".format(self.MODES))
        # A scalar covers every channel; an array gives one threshold per
        # channel, which is how the hand gets a looser limit than the arm.
        self.threshold = np.asarray(threshold, dtype=np.float64)
        if not np.all(self.threshold > 0.0):
            raise ValueError("threshold must be positive")
        self.mode = mode
        self.name = name
        self.labels = labels
        # The first policy steps after a reset carry an inherent transient: the
        # policy settles from the reference pose onto its own trajectory, and
        # that first step alone can exceed the threshold. Aborting on it would
        # stop every run at step 1, so the grace window suppresses the abort
        # (never the report) while the transient passes.
        self.grace_steps = max(0, int(grace_steps))
        self.updates = 0
        self.previous: Optional[np.ndarray] = None
        self.worst = 0.0
        self.worst_index = -1
        self.detections = 0

    def reset(self, initial: Optional[np.ndarray] = None) -> None:
        self.updates = 0
        self.previous = None if initial is None else np.asarray(initial, dtype=np.float64).copy()

    def _label(self, index: int) -> str:
        if self.labels is not None and 0 <= index < len(self.labels):
            return str(self.labels[index])
        return "index {}".format(index)

    def update(self, values: np.ndarray) -> Optional[dict]:
        current = np.asarray(values, dtype=np.float64)
        if self.mode == "off":
            self.previous = current.copy()
            return None
        if self.previous is None:
            self.previous = current.copy()
            return None
        self.updates += 1
        delta = np.abs(current - self.previous)
        # The end-of-run report wants the largest raw step; the abort wants the
        # channel furthest past *its own* threshold, which need not be the same
        # one once the thresholds differ.
        largest = int(np.argmax(delta))
        if float(delta[largest]) > self.worst:
            self.worst = float(delta[largest])
            self.worst_index = largest
        self.previous = current.copy()
        threshold = np.broadcast_to(self.threshold, delta.shape)
        exceeded = delta > threshold
        if not np.any(exceeded):
            return None
        index = int(np.argmax(np.where(exceeded, delta - threshold, -np.inf)))
        magnitude = float(delta[index])
        limit = float(threshold[index])
        self.detections += 1
        report = {
            "name": self.name,
            "magnitude": magnitude,
            "index": index,
            "label": self._label(index),
            "threshold": limit,
        }
        message = (
            "{} discontinuity: {} moved {:.4f} in one step (threshold {:.4f}). "
            "This may come from a startup mismatch or an out-of-distribution "
            "state; see simtoolreal_newton/deployment/README.md.".format(
                self.name.capitalize(), report["label"], magnitude, limit
            )
        )
        if self.mode == "stop" and self.updates > self.grace_steps:
            raise SafetyAbort(message)
        if self.updates <= self.grace_steps:
            message += " (within the {}-step startup grace window)".format(self.grace_steps)
        print("WARNING: " + message)
        return report


def wait_for_key(
    prompt: str,
    accept: tuple = (" ", "\r", "\n"),
    on_wait=None,
    wait_interval_s: float = 0.05,
) -> None:
    """Wait for a key while optionally servicing a safety keepalive."""
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        answer = input().strip().lower()
        if answer == "q":
            raise KeyboardInterrupt
        return
    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    settings = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        while True:
            readable, _, _ = select.select([descriptor], [], [], float(wait_interval_s))
            if not readable:
                if on_wait is not None:
                    on_wait()
                continue
            key = sys.stdin.read(1)
            if key.lower() == "q":
                print()
                raise KeyboardInterrupt
            if key in accept:
                print()
                return
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, settings)


def confirm_send(outputs: list) -> None:
    """Require a typed SEND before any physical output is armed."""
    print()
    print("=" * 70)
    print("ARMING PHYSICAL OUTPUT:")
    for output in outputs:
        print("  - {}".format(output))
    print("Clear the workspace. Keep the e-stop within reach.")
    print("=" * 70)
    answer = input("Type SEND to arm, anything else to abort: ")
    if answer.strip() != "SEND":
        raise KeyboardInterrupt("Aborted at the arming prompt.")
