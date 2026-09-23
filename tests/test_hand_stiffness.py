"""The DG5F stiffness parameter document and its read-back, without ROS."""

import unittest
from unittest import mock

from simtoolreal_newton.deployment import hand_stiffness
from simtoolreal_newton.envs.controller import HAND_JOINT_NAMES

DUMP = """/dg5f_right/rj_dg_pospid:
  ros__parameters:
    command_interface: effort
    gains:
      rj_dg_1_1:
        angle_wraparound: false
        d: 0.0
        i: 0.0
        p: 0.5
      rj_dg_1_2:
        d: 0.0
        p: 0.75
"""


class GainsYamlTest(unittest.TestCase):
    def test_every_hand_joint_gets_a_double_p(self):
        document = hand_stiffness.gains_yaml(1)
        self.assertTrue(document.startswith("/**:\n  ros__parameters:\n    gains:\n"))
        for joint in HAND_JOINT_NAMES:
            self.assertIn("      {}:\n        p: 1.0\n".format(joint), document)

    def test_a_non_positive_gain_is_refused(self):
        for value in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                hand_stiffness.gains_yaml(value)


class ReadBackTest(unittest.TestCase):
    def test_p_is_parsed_per_joint_from_a_dump(self):
        with mock.patch.object(hand_stiffness, "_ros2", return_value=DUMP):
            gains = hand_stiffness.read_hand_gains(joints=("rj_dg_1_1", "rj_dg_1_2"))
        self.assertEqual(gains, {"rj_dg_1_1": 0.5, "rj_dg_1_2": 0.75})

    def test_a_missing_joint_is_an_error(self):
        with mock.patch.object(hand_stiffness, "_ros2", return_value=DUMP):
            with self.assertRaises(hand_stiffness.HandStiffnessError):
                hand_stiffness.read_hand_gains(joints=("rj_dg_1_1", "rj_dg_5_4"))

    def test_set_loads_then_verifies(self):
        calls = []

        def fake_ros2(arguments, ros_setup, timeout):
            calls.append(list(arguments))
            return DUMP if arguments[1] == "dump" else ""

        with mock.patch.object(hand_stiffness, "_ros2", side_effect=fake_ros2):
            gains = hand_stiffness.set_hand_stiffness(0.5, joints=("rj_dg_1_1",))
            self.assertEqual(gains, {"rj_dg_1_1": 0.5})
            with self.assertRaises(hand_stiffness.HandStiffnessError):
                hand_stiffness.set_hand_stiffness(0.5, joints=("rj_dg_1_1", "rj_dg_1_2"))
        self.assertEqual(calls[0][:3], ["param", "load", hand_stiffness.DEFAULT_CONTROLLER_NODE])
        self.assertEqual(calls[1], ["param", "dump", hand_stiffness.DEFAULT_CONTROLLER_NODE])


if __name__ == "__main__":
    unittest.main()
