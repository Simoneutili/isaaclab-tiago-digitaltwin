# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a trained digital-twin agent (adapted from the Isaac Lab SB3 workflow).

Environment settings can be overridden with Hydra arguments, for example
``env.reference_shape=square env.real_velocity_topic=/real_velocities``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a trained digital-twin agent.")
parser.add_argument("--task", type=str, default="Isaac-DigitalTwin-TiAGO-v0", help="Name of the task.")
parser.add_argument("--algo", choices=["sac", "ppo"], default="sac", help="RL algorithm of the checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint. Default: the latest run.")
parser.add_argument("--real-time", action="store_true", help="Step the simulation no faster than real time.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path

import tiago_digital_twin  # noqa: F401
from tiago_digital_twin.sb3_utils import ALGORITHMS, make_sb3_env


@hydra_task_config(args_cli.task, f"sb3_{args_cli.algo}_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    checkpoint = args_cli.checkpoint
    if checkpoint is None:
        log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
        checkpoint = get_checkpoint_path(log_root_path, ".*", "model.zip")
    print(f"[INFO] Loading checkpoint from: {checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    step_dt = env.unwrapped.step_dt
    env = make_sb3_env(env, agent_cfg, checkpoint)
    agent = ALGORITHMS[args_cli.algo].load(checkpoint, env)

    obs = env.reset()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        if args_cli.real_time:
            time.sleep(max(0.0, step_dt - (time.time() - start_time)))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
