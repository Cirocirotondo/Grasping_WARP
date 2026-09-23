"""Where the observed cuboid pose comes from.

The observation ends with the cuboid's rotation and centre in the palm frame, so
something must supply a cuboid pose every control step. Three sources exist, in
increasing order of risk:

``ReferenceCube``
    Replays the cuboid track of one transform-bank clip, indexed by reference
    frame. Nothing is measured; the pose is a function of the reference index
    alone. This is the hardware-in-the-loop source: it lets the arm and the hand
    be commissioned separately, with a repeatable observation, before any
    camera is trusted.

``FrozenCube``
    Holds one frame of one clip forever. Useful to check that a stationary
    observation produces a stationary action.

``PoseEstimationCube``
    Subscribes to the tag pose estimator. The only source that closes the loop
    on the real cuboid, and the only one whose frame convention has to be
    verified before use -- see the warning on that class.

All three return a pose in the **demonstration's** (UR controller base) frame
convention, which is what the bank stores; ``DeploymentRun.cube_pose_base``
maps it into the base frame the observation is built in.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Protocol, Tuple

import numpy as np

DEFAULT_POSE_ADDRESS = "tcp://127.0.0.1:5558"

CubeState = Tuple[np.ndarray, np.ndarray, np.ndarray]


class CubeSourceError(RuntimeError):
    """The cuboid pose stream is absent, stale or malformed."""


class CubeSource(Protocol):
    name: str

    def cube_state(self, reference_index: int) -> CubeState:
        """Return (pose_xyzw[7], linear_velocity[3], angular_velocity[3])."""


class ReferenceCube:
    """The cuboid track of one bank clip, indexed by reference frame.

    ``object_scale`` lifts the track as the environment lifts its reference
    for a scaled bar, so the replayed bar rests on the table.
    """

    name = "reference"

    def __init__(self, run, transform_index: int, object_scale: float = 1.0) -> None:
        self.run = run
        self.transform_index = int(transform_index)
        self.object_scale = float(object_scale)

    def cube_state(self, reference_index: int) -> CubeState:
        index = int(np.clip(reference_index, 0, self.run.last_index))
        sample = self.run.reference_sample(self.transform_index, index)
        return (
            self.run.reference_cube_pose(self.transform_index, index, self.object_scale),
            sample.cube_linear_velocity[0].cpu().numpy().astype(np.float64),
            sample.cube_angular_velocity[0].cpu().numpy().astype(np.float64),
        )


class FrozenCube:
    """One frame of one bank clip, held for the whole rollout."""

    name = "frozen"

    def __init__(self, run, transform_index: int, reference_index: int, object_scale: float = 1.0) -> None:
        index = int(np.clip(reference_index, 0, run.last_index))
        self._pose = run.reference_cube_pose(int(transform_index), index, float(object_scale))
        self._zero = np.zeros(3, dtype=np.float64)

    def cube_state(self, reference_index: int) -> CubeState:
        return self._pose.copy(), self._zero.copy(), self._zero.copy()


class PoseEstimationCube:
    """Live cuboid pose from the tag pose estimator.

    WARNING -- frame convention. The estimator is run with a ``*_robot_frame``
    config and publishes a position plus a rotation matrix. This class assumes
    that frame is the one the demonstration recorded its ``cube_pose`` in, since
    the demonstration was captured through the same estimator. That assumption
    is plausible but NOT verified by this code, and a wrong frame is silently
    wrong: the policy would receive a mirrored or rotated cuboid and reach for
    the wrong place. Before ever enabling this source, park the real cuboid
    where a bank clip has it and compare ``cube_state()`` against
    ``ReferenceCube.cube_state()`` for that frame. ``--check-cube-frame`` in
    ``scripts/run_policy_real.py`` performs exactly that comparison.

    Velocities are reported as zero: the estimator publishes pose only, and a
    finite difference of a noisy tag pose is worse than a zero. The trained
    observation does not read cuboid velocity, so this costs nothing.

    **Position noise.** Measured on a static bar at 25 Hz, the stream is bimodal
    rather than Gaussian: 95% of samples sit within 3 mm of each other with a
    0.16 mm median step, and the rest snap ~16-22 mm away when the set of
    recognised faces changes. Those excursions last one or two samples, never
    more. Three filters are available, in the order they should be reached for:

    ``median_window``
        The filter that does the work, because it acts on the geometry. The
        excursions last one or two samples, so a median over five rejects them
        outright: measured with the arm in the scene, p95 1.57 mm and a worst
        case of 1.89 mm, against 14.8 mm raw. It costs 80 ms of lag, which is
        free during the approach -- the bar is not moving then -- and, unlike a
        gate, it can never starve the stream: a median always has an answer.

    ``minimum_confidence``
        Off by default, and **not** the filter to lean on. The estimator's
        confidence does separate the modes (with the arm parked, low-confidence
        samples deviate 1.91 mm at the median against 0.19 mm for the rest), but
        the whole distribution slides down as the robot occludes the board: a
        median of 0.625 on a clear bar becomes 0.577 with the arm parked and
        falls below any fixed threshold during the approach. An absolute gate
        therefore goes from rejecting 2% of samples to rejecting all of them
        exactly when the policy is reaching for the bar, which stalls the run on
        ``pose_timeout``. Raise it only for a diagnostic, never for a rollout.

    ``jump_reject_m`` / ``jump_accept_samples``
        A gross-teleport backstop, the earlier stack's ``reject_count`` idea at
        a sane scale: a sample further than ``jump_reject_m`` from the held pose
        is dropped, unless ``jump_accept_samples`` of them arrive in a row, so a
        bar that really moved still gets through. Acceptance is forced after
        that count, which bounds how stale the held pose can go.

    ``position_filter_alpha``
        A first-order low-pass on position only, off by default. The median
        already removes the excursions, and an average over a bimodal signal
        returns a position the bar is never at. Orientation is deliberately never
        filtered -- the bar's 8-fold symmetry means two detections can differ by
        a symmetry element, and interpolating between them sweeps the bar
        through an orientation it was never in. ``canonicalize_cuboid_orientation``
        in the contract is what resolves that, correctly.
    """

    name = "pose-estimation"

    def __init__(
        self,
        address: str = DEFAULT_POSE_ADDRESS,
        *,
        board_id: str = "0",
        minimum_confidence: float = 0.0,
        pose_timeout: float = 0.5,
        z_offset_m: float = 0.03,
        median_window: int = 5,
        jump_reject_m: float = 0.05,
        jump_accept_samples: int = 3,
        position_filter_alpha: float = 0.0,
        context=None,
    ) -> None:
        import zmq

        self._owns_context = context is None
        self.context = context or zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(address)
        self._zmq = zmq
        self.address = address
        self.board_id = str(board_id)
        self.minimum_confidence = float(minimum_confidence)
        self.pose_timeout = float(pose_timeout)
        self.z_offset_m = float(z_offset_m)
        self.median_window = max(1, int(median_window))
        self._positions: Deque[np.ndarray] = deque(maxlen=self.median_window)
        self.jump_reject_m = float(jump_reject_m)
        self.jump_accept_samples = max(1, int(jump_accept_samples))
        self.position_filter_alpha = float(position_filter_alpha)
        if not 0.0 <= self.position_filter_alpha <= 1.0:
            raise ValueError("position_filter_alpha must lie in [0, 1]")
        self._pose: Optional[np.ndarray] = None
        self._last_pose_at: Optional[float] = None
        self._pending_jumps = 0
        self._zero = np.zeros(3, dtype=np.float64)
        # Reported at the end of a run so the gate can be re-tuned from data.
        self.received_samples = 0
        self.accepted_at_last_check = 0
        self.low_confidence_samples = 0
        self.rejected_jump_samples = 0
        self.minimum_seen_confidence = float("inf")

    @staticmethod
    def _rotation_matrix_to_xyzw(rotation: np.ndarray) -> np.ndarray:
        # Project numerical drift onto the closest proper rotation first.
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (rotation[2, 1] - rotation[1, 2]) / scale
            y = (rotation[0, 2] - rotation[2, 0]) / scale
            z = (rotation[1, 0] - rotation[0, 1]) / scale
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif rotation[1, 1] > rotation[2, 2]:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
        quaternion = np.asarray((x, y, z, w), dtype=np.float64)
        return quaternion / np.linalg.norm(quaternion)

    def poll(self) -> bool:
        updated = False
        while True:
            try:
                message = self.socket.recv_json(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            poses = message.get("poses")
            if not isinstance(poses, dict):
                continue
            pose = poses.get(self.board_id)
            if not isinstance(pose, dict):
                continue
            self.received_samples += 1
            confidence = float(pose.get("confidence", 0.0))
            self.minimum_seen_confidence = min(self.minimum_seen_confidence, confidence)
            if confidence < self.minimum_confidence:
                self.low_confidence_samples += 1
                continue
            position = np.asarray(pose.get("position"), dtype=np.float64)
            rotation = np.asarray(pose.get("rotation_matrix"), dtype=np.float64)
            if position.shape != (3,) or rotation.shape != (3, 3):
                continue
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
                continue
            position[2] += self.z_offset_m
            if not self._accept_position(position):
                continue
            # The median goes over the raw accepted samples, not over the
            # filtered output, so one excursion cannot drag the window with it.
            self._positions.append(position.copy())
            if self.median_window > 1:
                position = np.median(np.stack(self._positions), axis=0)
            if self._pose is not None and self.position_filter_alpha > 0.0:
                alpha = self.position_filter_alpha
                position = alpha * position + (1.0 - alpha) * self._pose[:3]
            self._pose = np.concatenate((position, self._rotation_matrix_to_xyzw(rotation)))
            self._last_pose_at = time.monotonic()
            updated = True
        return updated

    def _accept_position(self, position: np.ndarray) -> bool:
        """Drop a one- or two-sample excursion; let a sustained move through.

        Acceptance is forced once ``jump_accept_samples`` rejections have piled
        up, so the held pose can never go stale by more than that many samples
        however badly the estimator misbehaves -- the staleness guard in
        ``cube_state`` stays the thing that stops the run, not this filter.
        """
        if self._pose is None or self.jump_reject_m <= 0.0:
            self._pending_jumps = 0
            return True
        if float(np.linalg.norm(position - self._pose[:3])) <= self.jump_reject_m:
            self._pending_jumps = 0
            return True
        self._pending_jumps += 1
        if self._pending_jumps >= self.jump_accept_samples:
            self._pending_jumps = 0
            return True
        self.rejected_jump_samples += 1
        return False

    def statistics(self) -> str:
        """One line on what the filters did, for the end-of-run report."""
        if not self.received_samples:
            return "cuboid pose: no samples received"
        seen = "{:.3f}".format(self.minimum_seen_confidence) if np.isfinite(
            self.minimum_seen_confidence
        ) else "n/a"
        return (
            "cuboid pose: {} samples, {} below confidence {:g} ({:.1f}%), {} transient jumps "
            "rejected (>{:g} m); lowest confidence seen {}".format(
                self.received_samples,
                self.low_confidence_samples,
                self.minimum_confidence,
                100.0 * self.low_confidence_samples / self.received_samples,
                self.rejected_jump_samples,
                self.jump_reject_m,
                seen,
            )
        )

    def wait_for_pose(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            self.poll()
            if self._pose is not None:
                return
            time.sleep(0.01)
        raise CubeSourceError(
            "No cuboid pose on {} for board '{}' within {:.1f} s. Is "
            "run_pose_estimation.py running?".format(self.address, self.board_id, timeout)
        )

    def cube_state(self, reference_index: int) -> CubeState:
        self.poll()
        if self._pose is None:
            raise CubeSourceError("No cuboid pose has been received.")
        age = time.monotonic() - float(self._last_pose_at)
        if age > self.pose_timeout:
            if self.received_samples == self.accepted_at_last_check:
                cause = (
                    "No message at all has arrived since the last check. The estimator publishes "
                    "nothing when it detects no board -- it does not send an empty message -- so "
                    "the usual cause is the hand occluding the tags during the approach, not a "
                    "dead process. Check its terminal for 'Published 1 board poses': if it is "
                    "still printing, raise --pose-timeout (the bar is not moving while you reach "
                    "for it, so holding the last pose is sound); if it is silent or gone, restart "
                    "run_pose_estimation.py on {}"
                ).format(self.address)
            else:
                cause = (
                    "Messages are still arriving but none passed the filters: {} of the last "
                    "samples were below --pose-min-confidence {:g} (lowest seen {:.3f}). Lower "
                    "the gate rather than raising --pose-timeout"
                ).format(
                    self.low_confidence_samples,
                    self.minimum_confidence,
                    self.minimum_seen_confidence if np.isfinite(self.minimum_seen_confidence) else float("nan"),
                )
            raise CubeSourceError(
                "Cuboid pose is stale by {:.3f} s (limit {:.3f} s). {}.".format(
                    age, self.pose_timeout, cause
                )
            )
        self.accepted_at_last_check = self.received_samples
        return self._pose.copy(), self._zero.copy(), self._zero.copy()

    def close(self) -> None:
        self.socket.close(linger=0)
        if self._owns_context:
            self.context.term()
