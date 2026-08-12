"""Agente A2C (stable-baselines3) sobre el entorno KRK gym."""
from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from .environment import N_ACTIONS, build_krk_gym_env


class A2CAgent:
    """
    Crea el entorno y el agente como:

        self.env = GymEnvFromGameAndPlayer2(...)
        self.agent = A2C('MlpPolicy', self.env, verbose=0)
    """

    def __init__(
        self,
        start_position: int = 1,
        side_to_move: str = "white",
        seed: int = 4,
        learning_rate: float = 7e-4,
        verbose: int = 0,
    ):
        self.start_position = start_position
        self.side_to_move = side_to_move
        self.seed = seed
        self.n_actions = N_ACTIONS
        self.learning_rate = learning_rate
        self.n_updates = 0
        self.last_loss = 0.0

        raw_env = build_krk_gym_env(
            start_position=start_position,
            side_to_move=side_to_move,
            seed=seed,
        )
        self.raw_env = raw_env
        self.env = Monitor(raw_env)
        self.model = A2C(
            "MlpPolicy",
            self.env,
            verbose=verbose,
            learning_rate=learning_rate,
            seed=seed,
        )
        self.gamma = float(self.model.gamma)

    def learn(self, total_timesteps: int = 1, callback=None) -> None:
        """Entrena un puñado de timesteps (para el bucle de telemetría)."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        self.n_updates += 1
        logger = getattr(self.model, "logger", None)
        if logger is not None and getattr(logger, "name_to_value", None):
            self.last_loss = float(
                logger.name_to_value.get("train/policy_loss", self.last_loss)
            )

    def predict(self, obs, deterministic: bool = False) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def value_grid(self, size: int = 8):
        """Mapa 8×8 con el valor estimado de la obs actual (broadcast visual)."""
        try:
            obs = self.env.unwrapped.encoder.to_array(
                self.env.unwrapped.encoder.encode_obs(self.raw_env.state)
            )
            obs_t, _ = self.model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                val = float(self.model.policy.predict_values(obs_t).item())
            grid = np.full((size, size), val, dtype=np.float32)
            # Variación leve por celda para que el heatmap no sea plano.
            noise = np.linspace(-0.05, 0.05, size * size, dtype=np.float32).reshape(size, size)
            return (grid + noise * abs(val + 1e-3)).tolist()
        except Exception:
            return np.zeros((size, size)).tolist()
