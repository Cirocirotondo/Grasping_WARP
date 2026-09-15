"""The folded chain path returns what the generic per-joint path returns."""

import unittest

import torch

from simtoolreal_newton import ROOT_DIR
from simtoolreal_newton.envs.controller import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from simtoolreal_newton.envs.kinematics import PalmKinematics, UrdfKinematics

URDF = ROOT_DIR / "assets" / "urdf" / "ur5e_delto_description" / "ur5e_right_dg5f_mount_60deg.urdf"


def _urdf_path():
    if URDF.is_file():
        return URDF
    candidates = sorted((ROOT_DIR / "assets").rglob("*.urdf"))
    return candidates[0] if candidates else None


class FoldedChainTest(unittest.TestCase):
    def setUp(self):
        path = _urdf_path()
        if path is None:
            self.skipTest("No URDF under assets/")
        self.path = path

    def _compare(self, dtype, atol):
        kinematics = UrdfKinematics(self.path, dtype=dtype)
        palm = PalmKinematics(self.path, dtype=dtype)
        generator = torch.Generator().manual_seed(0)
        q = (torch.rand(64, 6, generator=generator, dtype=dtype) - 0.5) * 2.0 * torch.pi
        pose_fast, jacobian_fast = kinematics.pose_and_jacobian(q, ARM_JOINT_NAMES, palm.palm_link)
        pose_slow, jacobian_slow = kinematics._pose_and_jacobian_generic(q, ARM_JOINT_NAMES, palm.palm_link)
        torch.testing.assert_close(pose_fast, pose_slow, atol=atol, rtol=0.0)
        torch.testing.assert_close(jacobian_fast, jacobian_slow, atol=atol, rtol=0.0)

    def test_float64_matches_the_generic_path_tightly(self):
        self._compare(torch.float64, atol=1e-12)

    def test_float32_matches_the_generic_path(self):
        self._compare(torch.float32, atol=1e-5)

    def test_the_arm_chain_takes_the_folded_path(self):
        kinematics = UrdfKinematics(self.path)
        palm = PalmKinematics(self.path)
        plan = kinematics._chain_plan(tuple(ARM_JOINT_NAMES), palm.palm_link)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.moving_count, 6)
        self.assertTrue(plan.columns_are_prefix)

    def test_joints_off_the_chain_get_zero_columns(self):
        """All 26 joints in, palm out: the hand joints must not move the palm."""
        kinematics = UrdfKinematics(self.path, dtype=torch.float64)
        palm = PalmKinematics(self.path, dtype=torch.float64)
        names = tuple(ARM_JOINT_NAMES) + tuple(HAND_JOINT_NAMES)
        q = torch.rand(8, len(names), dtype=torch.float64)
        pose_fast, jacobian_fast = kinematics.pose_and_jacobian(q, names, palm.palm_link)
        pose_slow, jacobian_slow = kinematics._pose_and_jacobian_generic(q, names, palm.palm_link)
        torch.testing.assert_close(pose_fast, pose_slow, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(jacobian_fast, jacobian_slow, atol=1e-12, rtol=0.0)
        self.assertEqual(tuple(jacobian_fast.shape), (8, 6, len(names)))
        self.assertTrue(torch.all(jacobian_fast[:, :, 6:] == 0.0))

    def test_a_fingertip_chain_through_hand_joints_matches_too(self):
        kinematics = UrdfKinematics(self.path, dtype=torch.float64)
        palm = PalmKinematics(self.path, dtype=torch.float64)
        names = tuple(ARM_JOINT_NAMES) + tuple(HAND_JOINT_NAMES)
        q = torch.rand(8, len(names), dtype=torch.float64)
        for link in palm.fingertip_links:
            pose_fast, jacobian_fast = kinematics.pose_and_jacobian(q, names, link)
            pose_slow, jacobian_slow = kinematics._pose_and_jacobian_generic(q, names, link)
            torch.testing.assert_close(pose_fast, pose_slow, atol=1e-12, rtol=0.0)
            torch.testing.assert_close(jacobian_fast, jacobian_slow, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
