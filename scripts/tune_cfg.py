"""Hyperparameter sweeps for the Isaac Lab Ray tuner (``source/standalone/workflows/ray/tuner.py``).

Pass this file as ``--cfg_file``, one of the classes below as ``--cfg_class`` and ``scripts/train.py`` as
``--workflow``.
"""

from ray import tune


def _sweep(algo: str, hyperparameters: dict) -> dict:
    return {
        "runner_args": {
            "headless_singleton": "--headless",
            "--task": "Isaac-DigitalTwin-TiAGO-v0",
            "--algo": algo,
            "--num_envs": tune.choice([128, 256, 512]),
        },
        "hydra_args": {f"agent.{name}": space for name, space in hyperparameters.items()},
    }


class TiagoSacJobCfg:
    def __init__(self):
        self.cfg = _sweep(
            "sac",
            {
                "learning_rate": tune.loguniform(1e-4, 1e-2),
                "batch_size": tune.choice([64, 128, 256, 512]),
                "gamma": tune.uniform(0.9, 0.999),
                "ent_coef": tune.loguniform(1e-3, 2e-1),
                "tau": tune.loguniform(1e-3, 0.1),
                "buffer_size": tune.choice([50000, 100000, 500000]),
            },
        )


class TiagoPpoJobCfg:
    def __init__(self):
        self.cfg = _sweep(
            "ppo",
            {
                "learning_rate": tune.loguniform(1e-5, 1e-3),
                "batch_size": tune.choice([32, 64, 128, 256, 512]),
                "gamma": tune.uniform(0.9, 0.9999),
                "ent_coef": tune.loguniform(1e-4, 1e-1),
                "n_steps": 64,
                "n_epochs": tune.randint(3, 20),
            },
        )
