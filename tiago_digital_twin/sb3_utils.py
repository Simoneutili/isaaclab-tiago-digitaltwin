"""Stable-Baselines3 glue shared by the train and play scripts."""

import os

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper

ALGORITHMS = {"sac": SAC, "ppo": PPO}


def normalizer_path(checkpoint: str) -> str:
    """Path of the normalization statistics that belong to a model checkpoint.

    Follows the naming of the SB3 ``CheckpointCallback``: ``model_<n>_steps.zip`` is paired with
    ``model_vecnormalize_<n>_steps.pkl``.
    """
    directory, name = os.path.split(checkpoint)
    return os.path.join(directory, name.replace("model", "model_vecnormalize", 1).replace(".zip", ".pkl"))


def make_sb3_env(env, agent_cfg: dict, checkpoint: str | None = None) -> VecEnv:
    """Wrap an Isaac Lab environment for SB3, with the normalization requested by the agent config.

    The normalization keys are removed from ``agent_cfg``. If statistics were saved next to ``checkpoint``
    they are restored, otherwise they are estimated from scratch.
    """
    env = Sb3VecEnvWrapper(env)
    norm_obs = agent_cfg.pop("normalize_input", False)
    norm_reward = agent_cfg.pop("normalize_value", False)
    clip_obs = agent_cfg.pop("clip_obs", 10.0)
    if not (norm_obs or norm_reward):
        return env
    if checkpoint and os.path.exists(normalizer_path(checkpoint)):
        return VecNormalize.load(normalizer_path(checkpoint), env)
    return VecNormalize(
        env, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=clip_obs, gamma=agent_cfg["gamma"], clip_reward=np.inf
    )
