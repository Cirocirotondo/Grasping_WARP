"""The scale observation: the ablation override and its own learning rate.

Pure torch -- no simulator. The environment is a stub that reports only what
PPO reads from it during construction.
"""

import unittest

import torch
import torch.nn as nn

from simtoolreal_newton.cfg import SimToolRealCfg, SimToolRealTrainCfg
from simtoolreal_newton.envs.object_scale import (
    observed_scale_override,
    scale_observation,
)
from simtoolreal_newton.runners.algorithms.ppo import PPO
from simtoolreal_newton.runners.modules import (
    Policy,
    SplitInputLinear,
    Value,
    fused_parameter_slots,
    split_input_layers,
)


class _Randomization:
    def __init__(self, override=None):
        self.observed_scale_override = override


class _Env:
    """What PPO reads from the environment while it is being built."""

    num_obs = 7
    num_privileged_obs = 9  # A privileged block sits *after* the scale column.
    num_actions = 3
    num_envs = 2

    def __init__(self, observes_scale=True):
        self.object_scale_observed = bool(observes_scale)
        self.resets = 0

    def reset(self):
        self.resets += 1


def _train_cfg(multiplier, hidden=(8, 4)):
    cfg = SimToolRealTrainCfg()
    cfg.policy.scale_input_lr_multiplier = multiplier
    cfg.policy.actor_hidden_dims = list(hidden)
    cfg.policy.critic_hidden_dims = list(hidden)
    cfg.runner.tensorboard = False
    cfg.runner.num_steps_per_env = 2
    cfg.runner.record_video = False
    cfg.algorithm.learning_rate = 1.0e-4
    return cfg


def _runner(multiplier, observes_scale=True):
    return PPO(_Env(observes_scale), _train_cfg(multiplier), log_dir=None, device="cpu")


class ObservedScaleOverrideTest(unittest.TestCase):
    def test_default_is_the_true_scale(self):
        self.assertIsNone(
            observed_scale_override(SimToolRealCfg.object_randomization)
        )
        scale = torch.tensor([0.8, 1.0, 1.2])
        column = scale_observation(scale)
        self.assertEqual(column.shape, (3, 1))
        self.assertTrue(torch.allclose(column, scale.reshape(-1, 1)))

    def test_override_replaces_every_entry(self):
        scale = torch.tensor([0.8, 1.0, 1.2])
        override = observed_scale_override(_Randomization(1.2))
        self.assertEqual(override, 1.2)
        column = scale_observation(scale, override)
        self.assertEqual(column.shape, (3, 1))
        self.assertTrue(torch.allclose(column, torch.full((3, 1), 1.2)))
        # The physical scale itself is untouched.
        self.assertTrue(torch.allclose(scale, torch.tensor([0.8, 1.0, 1.2])))
        self.assertTrue(
            torch.allclose(
                scale_observation(scale, 0.8), torch.full((3, 1), 0.8)
            )
        )

    def test_override_validation(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                observed_scale_override(_Randomization(bad))


class SplitInputLinearTest(unittest.TestCase):
    @staticmethod
    def _pair(in_features=5, out_features=4, split_start=4):
        plain = nn.Linear(in_features, out_features)
        split = SplitInputLinear(in_features, out_features, split_start=split_start)
        split.load_state_dict(plain.state_dict())
        return plain, split

    def test_forward_matches_a_plain_linear(self):
        torch.manual_seed(7)
        for split_start in (4, 2, 0):
            plain, split = self._pair(split_start=split_start)
            inputs = torch.randn(11, 5)
            self.assertTrue(
                torch.equal(plain(inputs), split(inputs)),
                "split at {} changed the forward pass".format(split_start),
            )

    def test_state_dict_keys_and_values_round_trip(self):
        torch.manual_seed(11)
        plain, split = self._pair()
        saved = split.state_dict()
        self.assertEqual(list(saved), list(plain.state_dict()))
        self.assertEqual(tuple(saved["weight"].shape), (4, 5))
        self.assertTrue(torch.equal(saved["weight"], plain.weight))
        self.assertTrue(torch.equal(saved["bias"], plain.bias))
        # split -> plain: a checkpoint written here stays loadable by code
        # that expects nn.Linear.
        restored = nn.Linear(5, 4)
        restored.load_state_dict(saved)
        self.assertTrue(torch.equal(restored.weight, plain.weight))
        # plain -> split, strictly, with no missing or unexpected keys.
        fresh = SplitInputLinear(5, 4, split_start=4)
        fresh.load_state_dict(plain.state_dict())
        self.assertTrue(torch.equal(fresh.weight, plain.weight))
        self.assertTrue(torch.equal(fresh.weight_split, plain.weight[:, 4:]))
        with self.assertRaises(RuntimeError):
            fresh.load_state_dict({"weight": torch.zeros(4, 6), "bias": torch.zeros(4)})

    def test_networks_keep_the_original_checkpoint_keys(self):
        torch.manual_seed(13)
        policy = Policy(6, 2, [8, 4], "elu", split_input_index=5)
        plain_policy = Policy(6, 2, [8, 4], "elu")
        self.assertEqual(
            list(policy.state_dict()), list(plain_policy.state_dict())
        )
        self.assertEqual(
            tuple(policy.state_dict()["policy_latent_net.0.weight"].shape), (8, 6)
        )
        plain_policy.load_state_dict(policy.state_dict())
        policy.load_state_dict(plain_policy.state_dict())
        value = Value(6, [8, 4], "elu", split_input_index=5)
        plain_value = Value(6, [8, 4], "elu")
        self.assertEqual(list(value.state_dict()), list(plain_value.state_dict()))
        self.assertEqual(tuple(value.state_dict()["value.0.weight"].shape), (8, 6))
        plain_value.load_state_dict(value.state_dict())

        observations = torch.randn(5, 6)
        self.assertTrue(
            torch.equal(
                policy.policy_latent_net(observations),
                plain_policy.policy_latent_net(observations),
            )
        )

    def test_fused_parameter_slots_follow_the_plain_ordering(self):
        policy = Policy(6, 2, [8, 4], "elu", split_input_index=5)
        value = Value(6, [8, 4], "elu", split_input_index=5)
        plain = list(Policy(6, 2, [8, 4], "elu").parameters())
        plain += list(Value(6, [8, 4], "elu").parameters())
        slots = fused_parameter_slots(policy, value)
        self.assertEqual(len(slots), len(plain))
        for slot, reference in zip(slots, plain):
            columns = sum(part.shape[1] if part.ndim == 2 else 0 for part in slot)
            if len(slot) == 1:
                self.assertEqual(slot[0].shape, reference.shape)
            else:
                self.assertEqual(columns, reference.shape[1])
                self.assertEqual(slot[0].shape[0], reference.shape[0])

    def test_the_split_column_takes_its_own_learning_rate(self):
        """One Adam step: 10x the rate moves the column 10x as far."""
        torch.manual_seed(17)
        plain, split = self._pair()
        gradient = torch.randn(4, 5)
        rate = 1.0e-3
        plain_optimizer = torch.optim.Adam(plain.parameters(), lr=rate)
        split_optimizer = torch.optim.Adam(
            [
                {"params": [split.weight_head], "lr": rate},
                {"params": [split.weight_split], "lr": rate * 10.0},
            ],
            lr=rate,
        )
        before = split.weight.detach().clone()
        plain.weight.grad = gradient.clone()
        split.weight_head.grad = gradient[:, :4].clone()
        split.weight_split.grad = gradient[:, 4:].clone()
        plain_optimizer.step()
        split_optimizer.step()
        plain_delta = (plain.weight.detach() - before).abs()
        split_delta = (split.weight.detach() - before).abs()
        # The untouched columns move exactly as before.
        self.assertTrue(
            torch.allclose(split_delta[:, :4], plain_delta[:, :4], atol=1e-12)
        )
        ratio = split_delta[:, 4] / plain_delta[:, 4]
        self.assertTrue(torch.allclose(ratio, torch.full((4,), 10.0), rtol=1e-4))


class ScaleInputLearningRateRunnerTest(unittest.TestCase):
    def test_default_multiplier_changes_nothing(self):
        self.assertEqual(
            SimToolRealTrainCfg.policy.scale_input_lr_multiplier, 1.0
        )
        runner = _runner(1.0)
        self.assertIsInstance(runner.policy.policy_latent_net[0], nn.Linear)
        self.assertNotIsInstance(
            runner.policy.policy_latent_net[0], SplitInputLinear
        )
        self.assertEqual(split_input_layers(runner.policy, runner.value), [])
        self.assertEqual(len(runner.optimizer.param_groups), 1)
        self.assertEqual(runner.optimizer.param_groups[0]["lr"], 1.0e-4)

    def test_multiplier_without_the_scale_observation_is_ignored(self):
        runner = _runner(20.0, observes_scale=False)
        self.assertEqual(runner.scale_input_lr_multiplier, 1.0)
        self.assertEqual(len(runner.optimizer.param_groups), 1)

    def test_both_networks_split_the_last_policy_observation(self):
        runner = _runner(20.0)
        layers = split_input_layers(runner.policy, runner.value)
        self.assertEqual(len(layers), 2)
        actor, critic = runner.policy.policy_latent_net[0], runner.value.value[0]
        self.assertIsInstance(actor, SplitInputLinear)
        self.assertIsInstance(critic, SplitInputLinear)
        # The critic sees the policy observation first, its privileged block
        # after it: the scale column is #6 of 9, not the last one.
        self.assertEqual(actor.split_start, _Env.num_obs - 1)
        self.assertEqual(critic.split_start, _Env.num_obs - 1)
        self.assertEqual(actor.in_features, _Env.num_obs)
        self.assertEqual(critic.in_features, _Env.num_privileged_obs)
        self.assertIsNone(actor.weight_tail)
        self.assertEqual(critic.weight_tail.shape[1], 2)
        self.assertEqual(
            tuple(runner.policy.state_dict()["policy_latent_net.0.weight"].shape),
            (8, _Env.num_obs),
        )
        self.assertEqual(
            tuple(runner.value.state_dict()["value.0.weight"].shape),
            (8, _Env.num_privileged_obs),
        )

    def test_param_groups_and_the_adaptive_schedule_keep_the_ratio(self):
        runner = _runner(20.0)
        base, scale = runner.optimizer.param_groups
        self.assertEqual(len(scale["params"]), 2)
        self.assertEqual(base["lr_multiplier"], 1.0)
        self.assertEqual(scale["lr_multiplier"], 20.0)
        self.assertAlmostEqual(base["lr"], 1.0e-4)
        self.assertAlmostEqual(scale["lr"], 2.0e-3)
        # Every parameter is trained exactly once.
        trained = [id(p) for group in runner.optimizer.param_groups for p in group["params"]]
        expected = [
            id(p)
            for p in list(runner.policy.parameters()) + list(runner.value.parameters())
        ]
        self.assertEqual(sorted(trained), sorted(expected))
        self.assertEqual(len(trained), len(set(trained)))

        runner.learning_rate = runner.learning_rate / 1.5
        runner._apply_learning_rate()
        base, scale = runner.optimizer.param_groups
        self.assertAlmostEqual(base["lr"], 1.0e-4 / 1.5)
        self.assertAlmostEqual(scale["lr"], 20.0 * 1.0e-4 / 1.5)
        self.assertAlmostEqual(scale["lr"] / base["lr"], 20.0)

    def test_a_pre_split_optimizer_state_loads_without_its_fused_moments(self):
        """A widened checkpoint resumes: only the split layers lose their moments."""
        torch.manual_seed(23)
        fused = _runner(1.0)
        for parameter in (
            list(fused.policy.parameters()) + list(fused.value.parameters())
        ):
            parameter.grad = torch.randn_like(parameter)
        fused.optimizer.step()
        saved = fused.optimizer.state_dict()
        self.assertEqual(len(saved["param_groups"]), 1)

        split = _runner(20.0)
        split.policy.load_state_dict(fused.policy.state_dict())
        split.value.load_state_dict(fused.value.state_dict())
        split._load_optimizer_state(saved)
        groups = split.optimizer.param_groups
        self.assertEqual([g["lr_multiplier"] for g in groups], [1.0, 20.0])
        self.assertAlmostEqual(groups[1]["lr"] / groups[0]["lr"], 20.0)
        # Every parameter but the four first-layer blocks kept its moments.
        state = split.optimizer.state
        first_layer_blocks = [
            parameter
            for layer in split_input_layers(split.policy, split.value)
            for parameter in layer.chunks()
        ]
        self.assertEqual(len(first_layer_blocks), 5)  # head+split, head+split+tail
        block_ids = {id(parameter) for parameter in first_layer_blocks}
        for parameter in first_layer_blocks:
            self.assertNotIn(parameter, state)
        kept = [
            parameter
            for parameter in (
                list(split.policy.parameters()) + list(split.value.parameters())
            )
            if id(parameter) not in block_ids
        ]
        self.assertTrue(kept)
        for parameter in kept:
            self.assertIn(parameter, state)
            self.assertEqual(state[parameter]["exp_avg"].shape, parameter.shape)
        # The adapted state is usable: one more step must not raise.
        for parameter in (
            list(split.policy.parameters()) + list(split.value.parameters())
        ):
            parameter.grad = torch.randn_like(parameter)
        split.optimizer.step()

    def test_an_unrecognised_optimizer_state_is_skipped(self):
        runner = _runner(20.0)
        runner._load_optimizer_state({"state": {}, "param_groups": [{"params": [0, 1]}]})
        self.assertEqual(
            [g["lr_multiplier"] for g in runner.optimizer.param_groups], [1.0, 20.0]
        )


if __name__ == "__main__":
    unittest.main()
