"""The randomization families added for the DR wave: extra pairs, impulses, bar-pose noise."""

import unittest

import numpy as np
import torch

from simtoolreal_newton.envs.domain_randomization import DomainRandomization
from simtoolreal_newton.envs.rotations import normalize_canonical_quaternion, quat_multiply, quat_to_matrix
from simtoolreal_newton.envs.self_collision import allowed_body_pairs, finger_body_names, filtered_body_pairs
from simtoolreal_newton.envs.sensing import (
    perturb_cube_pose_observation,
    rotation_vector_to_quaternion,
    sample_cube_pose_bias,
)


class _Asset:
    self_collision = True
    self_collision_fingers = [2, 3, 4, 5]
    self_collision_adjacent_fingers_only = False
    self_collision_with_palm = False
    self_collision_same_finger = False
    self_collision_extra_pairs = []


HAND = ("wrist_3_link",) + finger_body_names()


class ExtraPairsTest(unittest.TestCase):
    def test_ring_distal_against_palm_opens_exactly_one_pair(self):
        asset = _Asset()
        base = allowed_body_pairs(asset, HAND)
        asset.self_collision_extra_pairs = [["rl_dg_4_4", "wrist_3_link"]]
        opened = allowed_body_pairs(asset, HAND)
        self.assertEqual(opened - base, {frozenset(("rl_dg_4_4", "wrist_3_link"))})
        self.assertEqual(len(opened), 97)
        self.assertNotIn(frozenset(("rl_dg_4_4", "wrist_3_link")), filtered_body_pairs(asset, HAND))
        self.assertIn(frozenset(("rl_dg_3_4", "wrist_3_link")), filtered_body_pairs(asset, HAND))

    def test_extra_pairs_reject_unknown_or_degenerate_entries(self):
        asset = _Asset()
        asset.self_collision_extra_pairs = [["rl_dg_4_4", "no_such_body"]]
        with self.assertRaises(ValueError):
            allowed_body_pairs(asset, HAND)
        asset.self_collision_extra_pairs = [["rl_dg_4_4", "rl_dg_4_4"]]
        with self.assertRaises(ValueError):
            allowed_body_pairs(asset, HAND)

    def test_extra_pairs_need_the_master_switch(self):
        asset = _Asset()
        asset.self_collision = False
        asset.self_collision_extra_pairs = [["rl_dg_4_4", "wrist_3_link"]]
        self.assertEqual(allowed_body_pairs(asset, HAND), set())


class _DRCfg:
    enabled = True
    finger_impulse_probability = 0.02
    finger_impulse_n = 0.05
    robot_impulse_probability = 0.02
    robot_impulse_n = 10.0
    obs_cube_position_noise_m = 0.005
    obs_cube_orientation_noise_rad = 0.035
    obs_cube_position_bias_m = 0.005
    obs_cube_orientation_bias_rad = 0.035
    fingertip_friction_range = 0.3
    object_friction_range = 0.3
    table_friction_range = 0.3


class NewFieldsTest(unittest.TestCase):
    def test_families_are_read_and_gated_by_the_master_switch(self):
        dr = DomainRandomization(_DRCfg(), 8, seed=1)
        self.assertTrue(dr.cube_observation_noise_enabled)
        self.assertTrue(dr.finger_impulses_enabled)
        self.assertTrue(dr.impulses_enabled)
        # Friction is applied on this port: it reaches the critic table too.
        self.assertIn("fingertip_friction", dr.critic_parameters)
        low, high = dr.summary()["fingertip_friction"]
        self.assertGreaterEqual(low, 0.7)
        self.assertLessEqual(high, 1.3)
        cfg = _DRCfg()
        cfg.enabled = False
        off = DomainRandomization(cfg, 8, seed=1)
        self.assertFalse(off.cube_observation_noise_enabled)
        self.assertFalse(off.impulses_enabled)

    def test_invalid_values_are_rejected(self):
        cfg = _DRCfg()
        cfg.finger_impulse_probability = 1.5
        with self.assertRaises(ValueError):
            DomainRandomization(cfg, 4)
        cfg = _DRCfg()
        cfg.obs_cube_position_noise_m = -1.0
        with self.assertRaises(ValueError):
            DomainRandomization(cfg, 4)


class CubePoseNoiseTest(unittest.TestCase):
    def test_zero_noise_and_bias_leave_the_pose_untouched(self):
        position = torch.randn(5, 3)
        orientation = normalize_canonical_quaternion(torch.randn(5, 4))
        out_p, out_q = perturb_cube_pose_observation(position, orientation, 0.0, 0.0)
        torch.testing.assert_close(out_p, position)
        torch.testing.assert_close(out_q, orientation)

    def test_noise_has_the_requested_scale(self):
        generator = torch.Generator().manual_seed(0)
        position = torch.zeros(20000, 3)
        orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(20000, 4)
        out_p, out_q = perturb_cube_pose_observation(position, orientation, 0.005, 0.035, generator=generator)
        self.assertAlmostEqual(float(out_p.std()), 0.005, delta=0.0003)
        angle = 2.0 * torch.acos(out_q[:, 3].clamp(max=1.0))
        # |rotation vector| with three N(0, s) components has mean s*sqrt(8/pi).
        self.assertAlmostEqual(float(angle.mean()), 0.035 * float(np.sqrt(8.0 / np.pi)), delta=0.003)
        self.assertTrue(bool((out_q[:, 3] >= 0.0).all()))

    def test_bias_is_a_constant_rotation_and_offset(self):
        position = torch.zeros(3, 3)
        orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(3, 4)
        bias_p = torch.tensor([[0.004, 0.0, 0.0]] * 3)
        bias_r = torch.tensor([[0.0, 0.0, 0.1]] * 3)
        out_p, out_q = perturb_cube_pose_observation(position, orientation, 0.0, 0.0, bias_p, bias_r)
        torch.testing.assert_close(out_p, bias_p)
        expected = rotation_vector_to_quaternion(bias_r)
        torch.testing.assert_close(out_q, expected, atol=1e-6, rtol=0.0)
        matrix = quat_to_matrix(out_q[0])
        self.assertAlmostEqual(float(matrix[0, 0]), float(np.cos(0.1)), places=6)
        self.assertAlmostEqual(float(matrix[1, 0]), float(np.sin(0.1)), places=6)

    def test_rotation_vector_quaternion_composes_like_a_rotation(self):
        a = torch.tensor([[0.2, -0.1, 0.05]])
        q = rotation_vector_to_quaternion(a)
        torch.testing.assert_close(q.norm(dim=-1), torch.ones(1))
        q_twice = quat_multiply(q, q)
        torch.testing.assert_close(q_twice, rotation_vector_to_quaternion(2.0 * a), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(rotation_vector_to_quaternion(torch.zeros(1, 3)), torch.tensor([[0.0, 0.0, 0.0, 1.0]]))

    def test_bias_sampler_respects_the_bounds(self):
        generator = torch.Generator().manual_seed(3)
        position, rotation = sample_cube_pose_bias(5000, 0.005, 0.035, "cpu", generator=generator)
        self.assertLessEqual(float(position.abs().max()), 0.005)
        self.assertLessEqual(float(rotation.norm(dim=1).max()), 0.035 + 1e-6)
        self.assertGreater(float(rotation.norm(dim=1).max()), 0.03)


if __name__ == "__main__":
    unittest.main()
