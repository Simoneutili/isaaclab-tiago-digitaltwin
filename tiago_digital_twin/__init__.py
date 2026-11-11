"""Real2Sim digital twin of the TIAGo arm: an RL agent tunes joint friction and damping in Isaac Lab."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-DigitalTwin-TiAGO-v0",
    entry_point=f"{__name__}.tiago_env:TiagoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiago_env:TiagoEnvCfg",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
