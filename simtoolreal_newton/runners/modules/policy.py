"""AnimRL-compatible Gaussian MLP policy."""

import math

import torch
import torch.nn as nn

from simtoolreal_newton.runners.utils.distributions import (
    DiagGaussianDistribution,
)

from .split_input_linear import SplitInputLinear


def first_linear(num_obs, out_features, split_input_index=None, device="cpu"):
    """The input layer: plain ``Linear``, or one with a column split out.

    ``split_input_index`` names the observation column that gets its own
    parameter (and so its own learning rate); ``None`` builds exactly the
    ``nn.Linear`` every earlier run used. Either way the layer saves and loads
    one fused ``weight`` of shape ``(out_features, num_obs)``.
    """
    if split_input_index is None:
        return nn.Linear(num_obs, out_features).to(device)
    return SplitInputLinear(
        num_obs, out_features, split_start=int(split_input_index), split_size=1
    ).to(device)


def get_activation(name):
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError("Unsupported activation: {!r}".format(name))
    return activations[name]()


class Policy(nn.Module):
    """Field and layer names intentionally match AnimRL checkpoints."""

    def __init__(
        self,
        num_obs,
        num_actions,
        hidden_dims=None,
        activation="elu",
        log_std_init=0.0,
        max_action_std=None,
        min_action_std=None,
        split_input_index=None,
        device="cpu",
        **kwargs
    ):
        del kwargs
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        activation_module = get_activation(activation)

        layers = [
            first_linear(num_obs, hidden_dims[0], split_input_index, device),
            activation_module,
        ]
        for index in range(len(hidden_dims) - 1):
            layers.append(
                nn.Linear(hidden_dims[index], hidden_dims[index + 1]).to(device)
            )
            layers.append(get_activation(activation))
        self.policy_latent_net = nn.Sequential(*layers)

        self.max_action_std = max_action_std
        self.min_action_std = min_action_std
        self.distribution = DiagGaussianDistribution(
            action_dim=num_actions,
            max_action_std=max_action_std,
            min_action_std=min_action_std,
        )
        self.action_mean_net, self.log_std = self.distribution.proba_distribution_net(
            latent_dim=hidden_dims[-1], log_std_init=log_std_init
        )
        self.action_mean_net = self.action_mean_net.to(device)

    def reset(self, dones=None):
        del dones

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.distribution.mean

    @property
    def action_std(self):
        return self.distribution.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy()

    def project_action_std(self):
        """Project the learned log standard deviation onto its configured bounds."""
        if self.max_action_std is None and self.min_action_std is None:
            return
        with torch.no_grad():
            self.log_std.clamp_(
                min=math.log(self.min_action_std) if self.min_action_std is not None else None,
                max=math.log(self.max_action_std) if self.max_action_std is not None else None,
            )

    def act_and_log_prob(self, observations):
        mean = self.action_mean_net(self.policy_latent_net(observations))
        return self.distribution.log_prob_from_params(mean, self.log_std)

    def act_inference(self, observations):
        mean = self.action_mean_net(self.policy_latent_net(observations))
        return self.distribution.actions_from_params(
            mean, self.log_std, deterministic=True
        )
