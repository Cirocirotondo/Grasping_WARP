"""A linear layer that keeps one input column in its own parameter.

The cuboid scale is appended as the last policy observation column. A
checkpoint widened from the 112-input policy starts that column at zero
weight, and Adam bounds every element's step by the learning rate, so the
column crawls while the rest of the network is already trained: scaling the
gradient cannot help, only a larger learning rate can.

:class:`SplitInputLinear` is a plain ``nn.Linear`` whose weight is stored as
two (or three) contiguous column blocks, so the block holding the scale column
can be handed to the optimizer as its own parameter group with its own
learning rate. Two properties make it a drop-in:

* the forward pass concatenates the blocks and calls ``F.linear``, which is
  the same arithmetic a plain ``Linear`` performs, and
* the state dict keeps the original ``weight``/``bias`` keys and the fused
  ``(out, in)`` shape, so old checkpoints load into it and the checkpoints it
  writes stay loadable by anything expecting a plain ``Linear``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# The column blocks, in input order. ``weight_tail`` is absent when the split
# column is the last input, which is the usual case for the actor.
CHUNK_NAMES = ("weight_head", "weight_split", "weight_tail")


class SplitInputLinear(nn.Module):
    """``nn.Linear`` with input columns ``[start, start + size)`` split out."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        split_start: int,
        split_size: int = 1,
        bias: bool = True,
        device=None,
    ):
        super().__init__()
        in_features = int(in_features)
        out_features = int(out_features)
        split_start = int(split_start)
        split_size = int(split_size)
        if split_size <= 0:
            raise ValueError("split_size must be positive")
        if split_start < 0 or split_start + split_size > in_features:
            raise ValueError("The split columns must lie inside the input")
        self.in_features = in_features
        self.out_features = out_features
        self.split_start = split_start
        self.split_size = split_size

        # Initialised from a plain Linear of the full width so the weights are
        # drawn from exactly the distribution the fused layer would have used.
        reference = nn.Linear(in_features, out_features, bias=bias, device=device)
        stop = split_start + split_size
        with torch.no_grad():
            weight = reference.weight.detach()
            self.weight_head = nn.Parameter(weight[:, :split_start].clone())
            self.weight_split = nn.Parameter(weight[:, split_start:stop].clone())
            tail = weight[:, stop:]
            if tail.shape[1] > 0:
                self.weight_tail = nn.Parameter(tail.clone())
            else:
                self.register_parameter("weight_tail", None)
            if bias:
                self.bias = nn.Parameter(reference.bias.detach().clone())
            else:
                self.register_parameter("bias", None)

    # -- the fused view -------------------------------------------------
    def chunks(self) -> List[nn.Parameter]:
        """The column blocks in input order, skipping an empty tail."""
        blocks = [self.weight_head, self.weight_split]
        if self.weight_tail is not None:
            blocks.append(self.weight_tail)
        return blocks

    @property
    def weight(self) -> torch.Tensor:
        """The ``(out, in)`` matrix a plain ``Linear`` would hold."""
        return torch.cat(self.chunks(), dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, split=[{}, {}), bias={}".format(
            self.in_features,
            self.out_features,
            self.split_start,
            self.split_start + self.split_size,
            self.bias is not None,
        )

    # -- checkpoint compatibility ---------------------------------------
    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        """Write the ORIGINAL ``weight``/``bias`` keys, fused."""
        weight = self.weight
        destination[prefix + "weight"] = weight if keep_vars else weight.detach()
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Accept a fused ``weight`` by splitting it into the blocks.

        ``state_dict`` is the loader's own copy, so rewriting the key here and
        deferring to ``nn.Module`` keeps torch's shape checks and messages.
        """
        key = prefix + "weight"
        weight = state_dict.get(key)
        if weight is not None and torch.overrides.is_tensor_like(weight):
            del state_dict[key]
            stop = self.split_start + self.split_size
            if weight.ndim == 2 and weight.shape[1] == self.in_features:
                state_dict[prefix + "weight_head"] = weight[:, : self.split_start]
                state_dict[prefix + "weight_split"] = weight[:, self.split_start : stop]
                if self.weight_tail is not None:
                    state_dict[prefix + "weight_tail"] = weight[:, stop:]
            else:
                # Hand the whole thing to the first block so torch reports the
                # real mismatch rather than a silently wrong slice.
                state_dict[prefix + "weight_head"] = weight
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def split_input_layers(*modules: nn.Module) -> List[SplitInputLinear]:
    """Every :class:`SplitInputLinear` inside ``modules``, in module order."""
    found = []
    for module in modules:
        for layer in module.modules():
            if isinstance(layer, SplitInputLinear):
                found.append(layer)
    return found


def fused_parameter_slots(*modules: nn.Module) -> List[List[nn.Parameter]]:
    """The parameters grouped as a plain-``Linear`` model would have listed them.

    ``slots[i]`` holds the current parameters that make up what was the
    ``i``-th entry of ``list(module.parameters())`` before any layer was
    split: a single parameter for an untouched tensor, and the column blocks
    for the fused weight of a split layer. That is the map an optimizer state
    saved before the split has to be read through.
    """
    slots: List[List[nn.Parameter]] = []
    for module in modules:
        split_names = {
            name
            for name, layer in module.named_modules()
            if isinstance(layer, SplitInputLinear)
        }
        current_owner: Optional[str] = None
        for name, parameter in module.named_parameters():
            owner, _, leaf = name.rpartition(".")
            if owner in split_names and leaf in CHUNK_NAMES:
                if current_owner != owner:
                    current_owner = owner
                    slots.append([])
                slots[-1].append(parameter)
            else:
                current_owner = None
                slots.append([parameter])
    return slots
