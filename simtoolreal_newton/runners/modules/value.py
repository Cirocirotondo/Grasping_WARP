"""AnimRL-compatible value MLP."""

import torch.nn as nn

from .policy import first_linear, get_activation


class Value(nn.Module):
    """Field and layer names intentionally match AnimRL checkpoints."""

    def __init__(
        self,
        num_obs,
        hidden_dims=None,
        activation="elu",
        split_input_index=None,
        device="cpu",
        **kwargs
    ):
        del kwargs
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = [
            first_linear(num_obs, hidden_dims[0], split_input_index, device),
            get_activation(activation),
        ]
        for index in range(len(hidden_dims)):
            if index == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[index], 1).to(device))
            else:
                layers.append(
                    nn.Linear(hidden_dims[index], hidden_dims[index + 1]).to(
                        device
                    )
                )
                layers.append(get_activation(activation))
        self.value = nn.Sequential(*layers)

    def forward(self, observations, mask=None):
        del mask
        return self.value(observations)
