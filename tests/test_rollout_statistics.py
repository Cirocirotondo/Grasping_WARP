"""The rollout statistics computed on the device equal the per-step floats they replaced."""

import unittest

import torch

from simtoolreal_newton.cfg import SimToolRealTrainCfg
from simtoolreal_newton.runners.algorithms.ppo import _ROLLOUT_STEP_MEANS, PPO


class _FakeEnv:
    """Three steps of known numbers; an episode summary on the second."""

    num_envs = 4
    num_obs = 3
    num_privileged_obs = None
    num_actions = 2
    device = torch.device("cpu")
    dt = 1.0 / 60.0

    def __init__(self):
        self.calls = 0

    def reset(self):
        return torch.zeros(self.num_envs, self.num_obs)

    def get_observations(self):
        return torch.zeros(self.num_envs, self.num_obs)

    def get_privileged_observations(self):
        return None

    def saturated_actions(self, actions):
        saturated = torch.zeros_like(actions, dtype=torch.bool)
        saturated[0, 0] = True  # one of eight action values per step
        return saturated

    def step(self, actions):
        self.calls += 1
        k = float(self.calls)
        n = self.num_envs
        infos = {info_key: torch.full((n,), k * (index + 1)) for index, (_, info_key) in enumerate(_ROLLOUT_STEP_MEANS)}
        infos["max_abs_arm_position_error"] = torch.tensor([0.0, k, 0.0, 0.0])
        infos["time_outs"] = torch.tensor([True, False, False, False])
        infos["early_termination"] = torch.tensor([False, True, True, False])
        if self.calls == 2:
            infos["episode"] = {
                "completed_episodes": torch.tensor(3.0),
                "return": torch.tensor(10.0),
                "max_peak_object_com_lift_m": torch.tensor(0.25),
            }
        rewards = torch.full((n,), k)
        dones = torch.tensor([True, True, True, False])
        return torch.zeros(n, self.num_obs), None, rewards, dones, infos


class RolloutStatisticsTest(unittest.TestCase):
    def test_the_device_side_accumulation_matches_the_hand_computation(self):
        torch.manual_seed(0)
        train_cfg = SimToolRealTrainCfg()
        train_cfg.runner.num_steps_per_env = 3
        train_cfg.runner.tensorboard = False
        env = _FakeEnv()
        ppo = PPO(env, train_cfg, log_dir=None, device="cpu")
        # Deterministic, bounded actions so the action statistics are
        # checkable; the real call still runs so the policy keeps the state
        # process_env_step() reads (action mean and std).
        real_act = ppo.policy.act_and_log_prob

        def fixed_actions(observations):
            _, log_prob = real_act(observations)
            return torch.full((env.num_envs, env.num_actions), 0.5), log_prob

        ppo.policy.act_and_log_prob = fixed_actions

        result = ppo.collect_rollout()

        steps = 3.0
        k_mean = (1.0 + 2.0 + 3.0) / steps  # mean of k over the three steps
        self.assertAlmostEqual(result["mean_reward"], k_mean)
        for index, (result_key, _) in enumerate(_ROLLOUT_STEP_MEANS):
            self.assertAlmostEqual(result[result_key], k_mean * (index + 1), places=5, msg=result_key)
        self.assertAlmostEqual(result["max_abs_position_error"], 3.0)
        self.assertEqual(result["done_count"], 9)
        self.assertEqual(result["timeout_count"], 3)
        self.assertEqual(result["early_termination_count"], 6)
        self.assertAlmostEqual(result["done_fraction"], 9 / 12)
        self.assertAlmostEqual(result["action_target_clipped_fraction"], 3 / 24)
        self.assertAlmostEqual(result["mean_abs_action"], 0.5)
        self.assertAlmostEqual(result["max_abs_action"], 0.5)
        self.assertEqual(result["episode_count"], 3)
        self.assertAlmostEqual(result["episode_return"], 10.0)
        self.assertAlmostEqual(result["episode_max_peak_object_com_lift_m"], 0.25)
        for value in result.values():
            self.assertNotIsInstance(value, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
