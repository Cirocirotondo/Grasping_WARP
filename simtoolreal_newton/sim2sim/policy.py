"""Saved run loading and the deterministic actor, without a simulator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from simtoolreal_newton import ROOT_DIR
from simtoolreal_newton.cfg import SimToolRealCfg, SimToolRealTrainCfg, update_config_from_dict
from simtoolreal_newton.envs.object_scale import (
    object_scale_observation_dim,
    observe_scale,
    observed_scale_override,
)
from simtoolreal_newton.runners.modules.normalizer import EmpiricalNormalization
from simtoolreal_newton.runners.modules.policy import Policy

from .constants import ACTION_DIM, BASE_OBSERVATION_DIM


@dataclass
class LoadedRun:
    checkpoint_path: Path
    config_path: Path
    env_cfg: SimToolRealCfg
    train_cfg: SimToolRealTrainCfg
    saved: Mapping[str, Any]

    @property
    def observation_dim(self) -> int:
        return int(self.env_cfg.env.num_observations) + object_scale_observation_dim(
            self.env_cfg.object_randomization
        )

    @property
    def observes_scale(self) -> bool:
        return observe_scale(self.env_cfg.object_randomization)

    @property
    def observed_scale_override(self) -> Optional[float]:
        return observed_scale_override(self.env_cfg.object_randomization)


def apply_overrides(env_cfg, overrides) -> None:
    """``PATH=VALUE`` overrides onto the env config (the ``--set`` of the training scripts)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("--set expects PATH=VALUE, got {!r}".format(item))
        path, raw = item.split("=", 1)
        parts = [part for part in path.strip().split(".") if part]
        if not parts or parts[0] == "train":
            raise ValueError("Only env configuration paths can be overridden here: {!r}".format(item))
        node = env_cfg
        for part in parts[:-1]:
            if not hasattr(node, part):
                raise KeyError("Unknown configuration section {!r} in --set {}".format(part, item))
            node = getattr(node, part)
        if not hasattr(node, parts[-1]):
            raise KeyError("Unknown configuration field {!r} in --set {}".format(parts[-1], item))
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        setattr(node, parts[-1], value)


def load_saved_run(
    checkpoint_path: Path, config_path: Optional[Path] = None, overrides=None
) -> LoadedRun:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
    config = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else checkpoint.parent / "config.json"
    )
    if not config.is_file():
        raise FileNotFoundError("Saved configuration not found: {}".format(config))
    with config.open("r", encoding="utf-8") as stream:
        saved = json.load(stream)
    if "env_cfg" not in saved or "train_cfg" not in saved:
        raise ValueError("Saved configuration needs env_cfg and train_cfg")
    env_cfg = SimToolRealCfg()
    train_cfg = SimToolRealTrainCfg()
    update_config_from_dict(env_cfg, saved["env_cfg"], strict=False)
    update_config_from_dict(train_cfg, saved["train_cfg"], strict=False)
    apply_overrides(env_cfg, overrides)
    run = LoadedRun(checkpoint, config, env_cfg, train_cfg, saved)
    if int(env_cfg.env.num_observations) != BASE_OBSERVATION_DIM:
        raise ValueError(
            "This runner builds the {}-D observation, the run declares {}".format(
                BASE_OBSERVATION_DIM, env_cfg.env.num_observations
            )
        )
    if int(env_cfg.env.num_actions) != ACTION_DIM:
        raise ValueError("The run must drive all {} joints".format(ACTION_DIM))
    if bool(getattr(env_cfg.contact, "observe_fingertip_forces", False)):
        raise ValueError("Checkpoints that observe fingertip forces are not supported by the sim2sim")
    saved_dim = saved.get("observation_dim")
    if saved_dim is not None and int(saved_dim) != run.observation_dim:
        raise ValueError(
            "The checkpoint was trained on {} observations, the configuration builds {}".format(
                saved_dim, run.observation_dim
            )
        )
    return run


class InferencePolicy:
    """The actor and its frozen observation normalizer, deterministic actions."""

    def __init__(self, run: LoadedRun, device: str = "cpu") -> None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for inference but is unavailable")
        self.run = run
        self.device = requested
        self.observation_dim = run.observation_dim
        policy_cfg = run.train_cfg.policy
        self.actor = Policy(
            num_obs=self.observation_dim,
            num_actions=ACTION_DIM,
            hidden_dims=list(policy_cfg.actor_hidden_dims),
            activation=str(policy_cfg.activation),
            log_std_init=float(policy_cfg.log_std_init),
            max_action_std=getattr(policy_cfg, "max_action_std", None),
            min_action_std=getattr(policy_cfg, "min_action_std", None),
            device=str(self.device),
        ).to(self.device)
        checkpoint = torch.load(str(run.checkpoint_path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["policy_dict"], strict=True)
        if bool(run.train_cfg.runner.normalize_observation) and "actor_obs_normalizer" in checkpoint:
            self.normalizer = EmpiricalNormalization(shape=self.observation_dim).to(self.device)
            self.normalizer.load_state_dict(checkpoint["actor_obs_normalizer"], strict=True)
        else:
            self.normalizer = torch.nn.Identity()
        self.infos = checkpoint.get("infos")
        self.actor.eval()
        self.normalizer.eval()

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_dim,):
            raise ValueError(
                "Observation has shape {}, expected ({},)".format(observation.shape, self.observation_dim)
            )
        tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            action = self.actor.act_inference(self.normalizer(tensor))
        return action[0].detach().cpu().numpy().astype(np.float32)
