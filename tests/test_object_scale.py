import sys
import unittest
from pathlib import Path

import torch

from simtoolreal_newton.cfg import SimToolRealCfg
from simtoolreal_newton.envs.object_scale import (
    expand_first_layer,
    expand_normalizer_row,
    inertia_factor,
    mass_factor,
    object_scale_observation_dim,
    reference_height_shift,
    sample_scales,
    scale_randomization_enabled,
    scale_range,
    uniform_scale_statistics,
)
from simtoolreal_newton.envs.proximity import fingertip_cuboid_proximity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from expand_checkpoint_observation import expand_checkpoint  # noqa: E402


class _Cfg:
    def __init__(self, low=1.0, high=1.0, observe=False):
        self.scale_min = low
        self.scale_max = high
        self.observe_scale = observe


class ObjectScaleTest(unittest.TestCase):
    def test_defaults_leave_every_existing_run_untouched(self):
        cfg = SimToolRealCfg.object_randomization
        self.assertEqual(scale_range(cfg), (1.0, 1.0))
        self.assertFalse(scale_randomization_enabled(cfg))
        self.assertEqual(object_scale_observation_dim(cfg), 0)
        self.assertEqual(SimToolRealCfg.env.num_observations, 112)

    def test_range_validation(self):
        with self.assertRaises(ValueError):
            scale_range(_Cfg(1.2, 0.8))
        with self.assertRaises(ValueError):
            scale_range(_Cfg(0.0, 1.0))
        self.assertTrue(scale_randomization_enabled(_Cfg(0.8, 1.2)))
        self.assertTrue(scale_randomization_enabled(_Cfg(1.2, 1.2)))
        self.assertEqual(object_scale_observation_dim(_Cfg(observe=True)), 1)

    def test_sampling_stays_in_range_and_fixed_when_degenerate(self):
        generator = torch.Generator().manual_seed(3)
        draws = sample_scales(4096, _Cfg(0.8, 1.2), "cpu", generator=generator)
        self.assertEqual(draws.shape, (4096,))
        self.assertGreaterEqual(float(draws.min()), 0.8)
        self.assertLessEqual(float(draws.max()), 1.2)
        self.assertAlmostEqual(float(draws.mean()), 1.0, delta=0.01)
        fixed = sample_scales(5, _Cfg(1.2, 1.2), "cpu")
        self.assertTrue(torch.allclose(fixed, torch.full((5,), 1.2)))
        self.assertEqual(sample_scales(0, _Cfg(0.8, 1.2), "cpu").numel(), 0)

    def test_nominal_anchor_fraction(self):
        cfg = _Cfg(0.8, 1.2)
        cfg.scale_nominal_probability = 0.25
        generator = torch.Generator().manual_seed(5)
        draws = sample_scales(20000, cfg, "cpu", generator=generator)
        nominal = float((draws == 1.0).float().mean())
        self.assertAlmostEqual(nominal, 0.25, delta=0.02)
        others = draws[draws != 1.0]
        self.assertGreaterEqual(float(others.min()), 0.8)
        self.assertLessEqual(float(others.max()), 1.2)
        cfg.scale_nominal_probability = 1.5
        with self.assertRaises(ValueError):
            sample_scales(4, cfg, "cpu")

    def test_anchor_table_with_extra_anchors(self):
        cfg = _Cfg(0.8, 1.2)
        cfg.scale_nominal_probability = 0.25
        cfg.scale_anchors = [[1.2, 0.15]]
        generator = torch.Generator().manual_seed(9)
        draws = sample_scales(40000, cfg, "cpu", generator=generator)
        self.assertAlmostEqual(float((draws == 1.0).float().mean()), 0.25, delta=0.015)
        self.assertAlmostEqual(float((draws == 1.2).float().mean()), 0.15, delta=0.015)
        cfg.scale_anchors = [[1.2, 0.9]]
        with self.assertRaises(ValueError):
            sample_scales(4, cfg, "cpu")

    def test_physical_factors(self):
        scale = torch.tensor([0.8, 1.0, 1.2])
        self.assertTrue(torch.allclose(mass_factor(scale, True), scale**3))
        self.assertTrue(torch.allclose(mass_factor(scale, False), torch.ones(3)))
        self.assertTrue(torch.allclose(inertia_factor(scale, True), scale**5))
        self.assertTrue(torch.allclose(inertia_factor(scale, False), scale**2))
        # A 20% bigger bar (half height 2.5 cm) rests 5 mm higher.
        shift = reference_height_shift(scale, 0.025)
        self.assertTrue(torch.allclose(shift, torch.tensor([-0.005, 0.0, 0.005])))

    def test_proximity_accepts_per_environment_extents(self):
        tips = torch.tensor([[[0.0, 0.0, 0.05]], [[0.0, 0.0, 0.05]]])  # 5 cm above the centre
        active = torch.ones(2, dtype=torch.bool)
        shared = torch.tensor([0.075, 0.025, 0.025])
        per_env = torch.stack((shared, shared * 2.0))
        _, shared_distance, _ = fingertip_cuboid_proximity(tips, shared, 0.02, active)
        _, per_env_distance, _ = fingertip_cuboid_proximity(tips, per_env, 0.02, active)
        self.assertAlmostEqual(float(shared_distance[0]), 0.025, places=6)
        self.assertAlmostEqual(float(per_env_distance[0]), 0.025, places=6)
        self.assertAlmostEqual(float(per_env_distance[1]), 0.0, places=6)
        with self.assertRaises(ValueError):
            fingertip_cuboid_proximity(tips, torch.zeros(3, 3), 0.02, active)

    def test_first_layer_and_normalizer_expansion(self):
        weight = torch.randn(4, 6)
        wide = expand_first_layer(weight, 1)
        self.assertEqual(wide.shape, (4, 7))
        x = torch.randn(2, 6)
        self.assertTrue(torch.allclose(x @ weight.T, torch.cat((x, torch.rand(2, 1)), dim=1) @ wide.T))
        row = expand_normalizer_row(torch.zeros(1, 6), 1, 1.0)
        self.assertEqual(row.shape, (1, 7))
        self.assertEqual(float(row[0, -1]), 1.0)
        mean, variance = uniform_scale_statistics(_Cfg(0.8, 1.2))
        self.assertAlmostEqual(mean, 1.0)
        self.assertAlmostEqual(variance, 0.4**2 / 12.0)

    def test_checkpoint_expansion_keeps_the_policy_identical(self):
        checkpoint = {
            "policy_dict": {"policy_latent_net.0.weight": torch.randn(8, 5), "policy_latent_net.0.bias": torch.zeros(8)},
            "value_dict": {"value.0.weight": torch.randn(8, 5)},
            "actor_obs_normalizer": {"_mean": torch.zeros(1, 5), "_var": torch.ones(1, 5), "_std": torch.ones(1, 5)},
            "critic_obs_normalizer": {"_mean": torch.zeros(1, 5), "_var": torch.ones(1, 5), "_std": torch.ones(1, 5)},
            "optimizer_state_dict": {
                "state": {0: {"exp_avg": torch.randn(8, 5), "exp_avg_sq": torch.rand(8, 5), "step": torch.tensor(3)}},
                "param_groups": [{"params": [0]}],
            },
            "infos": {"iteration": 17000},
        }
        wide = expand_checkpoint(checkpoint, 1, 1.0, 0.0133)
        self.assertEqual(wide["policy_dict"]["policy_latent_net.0.weight"].shape, (8, 6))
        self.assertEqual(wide["value_dict"]["value.0.weight"].shape, (8, 6))
        self.assertEqual(wide["actor_obs_normalizer"]["_mean"].shape, (1, 6))
        self.assertAlmostEqual(float(wide["actor_obs_normalizer"]["_mean"][0, -1]), 1.0)
        self.assertEqual(wide["optimizer_state_dict"]["state"][0]["exp_avg"].shape, (8, 6))
        self.assertEqual(int(wide["optimizer_state_dict"]["state"][0]["step"]), 3)
        self.assertEqual(wide["infos"]["observation_expanded_by"], 1)
        self.assertEqual(wide["infos"]["iteration"], 17000)
        # The original is not modified.
        self.assertEqual(checkpoint["policy_dict"]["policy_latent_net.0.weight"].shape, (8, 5))


if __name__ == "__main__":
    unittest.main()
