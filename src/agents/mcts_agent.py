from stable_baselines3 import A2C
from typing import Any, Optional

from numpy.random import Generator, default_rng

from agents.base_classes import GameProtocol
from agents.random_agent import RandomAgent
from agents.random_policy import GameUniformPolicy
from agents.utils import ChessEncoder
from games.chess import KRK
from games.game_utils import GymEnvFromGameAndPlayer2


class StBlsA2C:
    """
    Agente A2C de stable-baselines3 sobre KRK con oponente aleatorio.

        self.game = KRK(start_position=...)
        pl2 = RandomAgent(GameUniformPolicy(...))
        self.env = GymEnvFromGameAndPlayer2(game, pl2, ChessEncoder())
        self.agent = A2C('MlpPolicy', self.env, verbose=0)
    """

    def __init__(
        self,
        start_position: int = 1,
        side_to_move: str = "white",
        seed: int = 4,
        rng: Optional[Generator] = None,
        verbose: int = 0,
    ) -> None:
        self.seed = seed
        self.rng = rng if rng is not None else default_rng(self.seed)

        self.game = KRK(start_position=start_position, side_to_move=side_to_move)
        random_policy = GameUniformPolicy(
            game=self.game,
            rng=self.rng,
            encoder=None,
        )
        pl2 = RandomAgent(
            policy=random_policy,
            rng=self.rng,
        )
        encoder = ChessEncoder()
        self.env = GymEnvFromGameAndPlayer2(
            game=self.game,
            other_player=pl2,
            encoder=encoder,
            rng=self.rng,
        )
        self.agent = A2C("MlpPolicy", self.env, verbose=verbose, seed=self.seed)

    def learn(self, total_timesteps: int = 10_000, **kwargs: Any) -> A2C:
        return self.agent.learn(total_timesteps=total_timesteps, **kwargs)

    def predict(self, observation: Any, deterministic: bool = False):
        return self.agent.predict(observation, deterministic=deterministic)
