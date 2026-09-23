"""The live cuboid pose filters, against the noise actually measured on the rig.

The estimator's position noise is bimodal, not Gaussian: on a static bar at
25 Hz, 95% of samples sit within 3 mm of one another and the rest snap 16-22 mm
away for one or two samples when the recognised face set changes. These tests
pin the behaviour that matters for that shape of noise -- transients are
dropped, sustained motion is not, and orientation is never interpolated.
"""

import unittest

import numpy as np

from simtoolreal_newton.deployment.cube_source import PoseEstimationCube


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def make_source(**kwargs):
    # zmq connects lazily, so an endpoint nobody publishes on is fine here.
    return PoseEstimationCube("tcp://127.0.0.1:1", **kwargs)


def message(position, confidence=1.0, rotation=None):
    return {
        "poses": {
            "0": {
                "position": list(position),
                "rotation_matrix": rotation or IDENTITY,
                "confidence": confidence,
            }
        }
    }


class FakeSocket:
    """Replays a fixed list of messages, then raises Again like a real SUB."""

    def __init__(self, messages):
        self.messages = list(messages)

    def recv_json(self, flags=0):
        if not self.messages:
            raise Again()
        return self.messages.pop(0)

    def close(self, linger=0):
        pass


class Again(Exception):
    pass


class PoseFilterTest(unittest.TestCase):
    def feed(self, source, messages):
        if not isinstance(source.socket, FakeSocket):
            source.socket.close(linger=0)
            source.context.term()
        source.socket = FakeSocket(messages)
        source._zmq = type("zmq", (), {"Again": Again, "NOBLOCK": 0})
        source.poll()

    def test_a_clean_stream_passes_through_untouched(self):
        source = make_source()
        self.feed(source, [message([0.1, 0.2, 0.3])])
        np.testing.assert_allclose(source._pose[:3], [0.1, 0.2, 0.3 + source.z_offset_m])

    def test_a_one_sample_excursion_is_rejected(self):
        source = make_source(median_window=1, jump_reject_m=0.05, jump_accept_samples=3)
        self.feed(source, [message([0.0, 0.0, 0.0])])
        held = source._pose.copy()
        # 20 cm away for a single sample, the shape of a face-switch outlier.
        self.feed(source, [message([0.2, 0.0, 0.0]), message([0.001, 0.0, 0.0])])
        self.assertEqual(source.rejected_jump_samples, 1)
        np.testing.assert_allclose(source._pose[:3], [0.001, 0.0, held[2]], atol=1e-9)

    def test_a_sustained_move_is_accepted(self):
        source = make_source(median_window=1, jump_reject_m=0.05, jump_accept_samples=3)
        self.feed(source, [message([0.0, 0.0, 0.0])])
        self.feed(source, [message([0.2, 0.0, 0.0])] * 3)
        # Forced through on the third, so the bar that really moved is not lost.
        np.testing.assert_allclose(source._pose[0], 0.2)

    def test_rejection_cannot_freeze_the_held_pose_indefinitely(self):
        source = make_source(jump_reject_m=0.001, jump_accept_samples=2)
        self.feed(source, [message([0.0, 0.0, 0.0])])
        # Every sample is a jump; acceptance must still happen every 2 samples.
        for step in range(1, 9):
            self.feed(source, [message([0.5 * step, 0.0, 0.0])])
        self.assertLessEqual(source.rejected_jump_samples, 4)
        self.assertGreater(source._pose[0], 0.0)

    def test_the_median_rejects_a_two_sample_excursion(self):
        source = make_source(median_window=5)
        for x in (0.0, 0.001, 0.0, 0.001, 0.0):
            self.feed(source, [message([x, 0.0, 0.0])])
        # Two samples 20 mm away, the shape measured on the rig; the median of
        # five must not follow them.
        self.feed(source, [message([0.02, 0.0, 0.0]), message([0.02, 0.0, 0.0])])
        self.assertLess(abs(source._pose[0]), 0.002)

    def test_the_median_follows_a_sustained_move(self):
        source = make_source(median_window=5)
        for _ in range(5):
            self.feed(source, [message([0.0, 0.0, 0.0])])
        for _ in range(5):
            self.feed(source, [message([0.02, 0.0, 0.0])])
        np.testing.assert_allclose(source._pose[0], 0.02, atol=1e-9)

    def test_the_median_never_starves_the_stream(self):
        # Unlike a confidence gate, a median always produces an output, so the
        # staleness guard cannot fire because of filtering alone.
        source = make_source(median_window=9)
        self.feed(source, [message([0.1, 0.0, 0.0])])
        self.assertIsNotNone(source._pose)
        self.assertIsNotNone(source._last_pose_at)

    def test_the_median_is_off_at_window_one(self):
        source = make_source(median_window=1)
        self.feed(source, [message([0.0, 0.0, 0.0])])
        self.feed(source, [message([0.02, 0.0, 0.0])])
        np.testing.assert_allclose(source._pose[0], 0.02)

    def test_the_confidence_gate_drops_low_confidence_samples(self):
        source = make_source(minimum_confidence=0.55)
        self.feed(source, [message([0.1, 0.0, 0.0], confidence=0.62)])
        self.feed(source, [message([0.9, 0.0, 0.0], confidence=0.47)])
        np.testing.assert_allclose(source._pose[0], 0.1)
        self.assertEqual(source.low_confidence_samples, 1)
        self.assertAlmostEqual(source.minimum_seen_confidence, 0.47)

    def test_the_low_pass_is_off_by_default(self):
        source = make_source(median_window=1)
        self.feed(source, [message([0.0, 0.0, 0.0])])
        self.feed(source, [message([0.01, 0.0, 0.0])])
        np.testing.assert_allclose(source._pose[0], 0.01)

    def test_the_low_pass_blends_position_when_enabled(self):
        source = make_source(median_window=1, position_filter_alpha=0.25)
        self.feed(source, [message([0.0, 0.0, 0.0])])
        self.feed(source, [message([0.04, 0.0, 0.0])])
        np.testing.assert_allclose(source._pose[0], 0.01)

    def test_orientation_is_never_low_passed(self):
        # Interpolating between two symmetry-equivalent detections would sweep
        # the bar through an orientation it was never in; the contract's
        # canonicaliser is what resolves that, not a filter here.
        source = make_source(median_window=1, position_filter_alpha=0.5)
        self.feed(source, [message([0.0, 0.0, 0.0])])
        quarter_turn = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        self.feed(source, [message([0.0, 0.0, 0.0], rotation=quarter_turn)])
        expected = PoseEstimationCube._rotation_matrix_to_xyzw(np.asarray(quarter_turn))
        np.testing.assert_allclose(np.abs(source._pose[3:]), np.abs(expected), atol=1e-9)

    def test_the_median_is_on_by_default(self):
        # The rollout filter: a gate can starve the stream, a median cannot.
        self.assertEqual(make_source().median_window, 5)
        self.assertEqual(make_source().minimum_confidence, 0.0)

    def test_an_invalid_alpha_is_refused(self):
        with self.assertRaises(ValueError):
            make_source(position_filter_alpha=1.5)


if __name__ == "__main__":
    unittest.main()
