"""Which bar pose the palm keypoints are measured from.

The palm half of the keypoint reward can be anchored on the bar as *measured*
or on the bar the reference says is there. The difference only shows up once
the two poses disagree -- which is exactly the Transport phase, where a firm
grasp makes the measured anchor blind to whether the bar is being lifted at
all.

These are pure tensor tests of the frame arithmetic the environment performs.
They do not need Isaac Gym, and they pin the one silent bug the arrangement
can hide: applying the symmetry element to the reference pose, which is
already in the demonstration's labelling and therefore must not receive it.
"""

import math
import unittest

import torch

from simtoolreal_newton.envs.cuboid_symmetry import (
    apply_cuboid_symmetry,
    canonicalize_cuboid_orientation,
    cuboid_rotation_symmetries,
)
from simtoolreal_newton.envs.keypoints import (
    hand_keypoints,
    keypoint_tracking_error,
    keypoints_in_object_frame,
    split_palm_and_fingertips,
)
from simtoolreal_newton.envs.rotations import normalize_canonical_quaternion


IDENTITY = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
# The bar this project uses, whose symmetry group is the one that matters.
HALF_EXTENTS = torch.tensor([0.075, 0.025, 0.025])
LEVER_ARM_M = 0.1


def _palm_keypoints(palm_position, palm_orientation=IDENTITY):
    """Nine world keypoints for a hand whose fingertips sit on the palm."""
    fingertips = palm_position.unsqueeze(1).repeat(1, 5, 1)
    return hand_keypoints(
        palm_position, palm_orientation, fingertips, LEVER_ARM_M
    )


def _palm_error_m(keypoints_world, bar_position, bar_orientation, reference_palm):
    """RMS palm keypoint distance when measured from the given bar pose."""
    in_frame = keypoints_in_object_frame(
        keypoints_world, bar_position, bar_orientation
    )
    actual_palm, _ = split_palm_and_fingertips(in_frame)
    return keypoint_tracking_error(actual_palm, reference_palm).sqrt()


class PalmKeypointAnchorTest(unittest.TestCase):
    def setUp(self):
        # The demonstration: the palm sits 5 cm above the bar's centre, and the
        # reference bar is lifted 15 cm during Transport.
        self.bar_start = torch.tensor([[0.0, 0.0, 0.0]])
        self.palm_offset = torch.tensor([[0.0, 0.0, 0.05]])
        self.lift = torch.tensor([[0.0, 0.0, 0.15]])
        reference_keypoints = _palm_keypoints(self.bar_start + self.palm_offset)
        self.reference_palm, _ = split_palm_and_fingertips(
            keypoints_in_object_frame(
                reference_keypoints, self.bar_start, IDENTITY
            )
        )

    def test_anchors_agree_while_the_bar_is_where_the_reference_says(self):
        """At reset the cuboid is placed at the reference pose, so both agree."""
        keypoints = _palm_keypoints(self.bar_start + self.palm_offset)
        measured = _palm_error_m(
            keypoints, self.bar_start, IDENTITY, self.reference_palm
        )
        reference = _palm_error_m(
            keypoints, self.bar_start, IDENTITY, self.reference_palm
        )
        self.assertAlmostEqual(float(measured), 0.0, places=6)
        self.assertAlmostEqual(float(reference), 0.0, places=6)

    def test_measured_anchor_is_blind_to_a_lift_that_never_happened(self):
        """The failure the change exists to fix.

        The reference bar has risen 15 cm; the hand has risen with it but the
        real bar has not moved, because the grasp slipped or never closed. The
        measured anchor sees a perfect hand-bar geometry offset by nothing at
        all, and bills zero error. The reference anchor bills the full 15 cm.
        """
        hand_followed_the_reference = _palm_keypoints(
            self.bar_start + self.lift + self.palm_offset
        )
        # The bar itself never moved.
        measured = _palm_error_m(
            hand_followed_the_reference,
            self.bar_start + self.lift,
            IDENTITY,
            self.reference_palm,
        )
        self.assertAlmostEqual(float(measured), 0.0, places=6)

        # A hand that stayed put with the bar scores the same under the
        # measured anchor -- the two cases are indistinguishable to it.
        hand_stayed = _palm_keypoints(self.bar_start + self.palm_offset)
        measured_no_lift = _palm_error_m(
            hand_stayed, self.bar_start, IDENTITY, self.reference_palm
        )
        self.assertAlmostEqual(float(measured_no_lift), float(measured), places=6)

        # Anchored on the reference bar, the hand that did not lift is 15 cm
        # from where the demonstration put it, and the error points up.
        reference_no_lift = _palm_error_m(
            hand_stayed,
            self.bar_start + self.lift,
            IDENTITY,
            self.reference_palm,
        )
        self.assertAlmostEqual(float(reference_no_lift), 0.15, places=6)

    def test_reference_anchor_rewards_the_hand_that_lifted(self):
        """The other half: lifting the bar zeroes the reference-anchored error."""
        hand_and_bar_lifted = _palm_keypoints(
            self.bar_start + self.lift + self.palm_offset
        )
        reference = _palm_error_m(
            hand_and_bar_lifted,
            self.bar_start + self.lift,
            IDENTITY,
            self.reference_palm,
        )
        self.assertAlmostEqual(float(reference), 0.0, places=6)

    def test_reference_pose_must_not_receive_the_symmetry_element(self):
        """The silent bug this arrangement can hide.

        ``symmetry_index`` is chosen at reset to carry the *measured*
        orientation into the demonstration's labelling. The reference pose is
        already in that labelling, so applying the element to it again rotates
        the palm's frame by a relabelling the reference never needed. With this
        bar a 180 degree relabelling about its long axis moves the palm point
        by twice its offset, which is far past anything the reward tolerates.
        """
        symmetries = cuboid_rotation_symmetries(HALF_EXTENTS)
        self.assertGreater(symmetries.shape[0], 1)

        # A measured orientation that is the reference relabelled: physically
        # the same bar, a different quaternion.
        relabelled = normalize_canonical_quaternion(
            apply_cuboid_symmetry(
                IDENTITY, symmetries, torch.tensor([symmetries.shape[0] - 1])
            )
        )
        _, chosen = canonicalize_cuboid_orientation(
            relabelled, symmetries, IDENTITY, return_index=True
        )

        keypoints = _palm_keypoints(self.bar_start + self.lift + self.palm_offset)
        bar_position = self.bar_start + self.lift

        # Correct: the reference orientation used as it arrives.
        correct = _palm_error_m(
            keypoints, bar_position, IDENTITY, self.reference_palm
        )
        self.assertAlmostEqual(float(correct), 0.0, places=6)

        # The bug: the element applied to the reference as well.
        doubly_relabelled = apply_cuboid_symmetry(IDENTITY, symmetries, chosen)
        wrong = _palm_error_m(
            keypoints, bar_position, doubly_relabelled, self.reference_palm
        )
        if float(torch.abs(doubly_relabelled - IDENTITY).max()) > 1.0e-6:
            self.assertGreater(
                float(wrong),
                0.05,
                "applying the symmetry element to the reference pose must move "
                "the palm frame; if this no longer holds the test has stopped "
                "exercising the bug",
            )

    def test_fingertips_keep_the_measured_anchor(self):
        """Relative geometry is the fingertips' actual subject matter.

        A bar nudged 3 cm sideways with the fingers still on it is a hand that
        is still holding correctly, and the fingertip half must keep saying so.
        """
        nudge = torch.tensor([[0.03, 0.0, 0.0]])
        palm_position = self.bar_start + nudge + self.palm_offset
        fingertips = palm_position.unsqueeze(1).repeat(1, 5, 1)
        keypoints = hand_keypoints(
            palm_position, IDENTITY, fingertips, LEVER_ARM_M
        )
        reference_fingertips = split_palm_and_fingertips(
            keypoints_in_object_frame(
                _palm_keypoints(self.bar_start + self.palm_offset),
                self.bar_start,
                IDENTITY,
            )
        )[1]
        in_frame = keypoints_in_object_frame(
            keypoints, self.bar_start + nudge, IDENTITY
        )
        _, actual_fingertips = split_palm_and_fingertips(in_frame)
        error = keypoint_tracking_error(
            actual_fingertips, reference_fingertips
        ).sqrt()
        self.assertAlmostEqual(float(error), 0.0, places=6)


class PalmKeypointAnchorDefaultTest(unittest.TestCase):
    def test_reference_is_the_default(self):
        """Pinned deliberately.

        Every training from 2026-09-14 on is meant to use the reference
        anchor; a silent revert to "measured" would be invisible in the logs
        and would put the run on a reward scale that is not comparable with
        anything around it.
        """
        from simtoolreal_newton.cfg.simtoolreal_config import (
            SimToolRealCfg,
        )

        self.assertEqual(
            SimToolRealCfg.rewards.palm_keypoint_anchor, "reference"
        )


if __name__ == "__main__":
    unittest.main()
