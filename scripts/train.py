# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the digital-twin agent with Stable-Baselines3 (adapted from the Isaac Lab SB3 workflow).

Environment and agent settings can be overridden with Hydra arguments, for example
``env.initial_active_joints=1 agent.learning_rate=3e-4``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the digital-twin agent with Stable-Baselines3.")
parser.add_argument("--task", type=str, default="Isaac-DigitalTwin-TiAGO-v0", help="Name of the task.")
parser.add_argument("--algo", choices=["sac", "ppo"], default="sac", help="RL algorithm.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint to resume training from.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.utils.io import dump_yaml
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.sb3 import process_sb3_cfg

import tiago_digital_twin  # noqa: F401
from tiago_digital_twin.sb3_utils import ALGORITHMS, make_sb3_env, normalizer_path


@hydra_task_config(args_cli.task, f"sb3_{args_cli.algo}_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, run_name)
    # the Isaac Lab Ray tuner locates the run by parsing these two lines
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_name}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    agent_cfg = process_sb3_cfg(agent_cfg)
    policy = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    env = make_sb3_env(gym.make(args_cli.task, cfg=env_cfg), agent_cfg, args_cli.checkpoint)
    if args_cli.checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args_cli.checkpoint}")
        agent = ALGORITHMS[args_cli.algo].load(args_cli.checkpoint, env=env)
    else:
        agent = ALGORITHMS[args_cli.algo](policy, env, verbose=1, **agent_cfg)
    # the Isaac Lab Ray tuner reads the TensorBoard scalars from <run>/summaries
    agent.set_logger(configure(os.path.join(log_dir, "summaries"), ["stdout", "tensorboard"]))

    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=log_dir, name_prefix="model", save_vecnormalize=True, verbose=2
    )
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)

    model_path = os.path.join(log_dir, "model.zip")
    agent.save(model_path)
    if agent.get_vec_normalize_env() is not None:
        agent.get_vec_normalize_env().save(normalizer_path(model_path))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
