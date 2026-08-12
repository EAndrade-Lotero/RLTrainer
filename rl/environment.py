"""Entorno KRK via GymEnvFromGameAndPlayer2 (oponente aleatorio + ChessEncoder)."""
from __future__ import annotations

import sys
from pathlib import Path

import chess
from numpy.random import default_rng

# Asegura que `src/` esté en el path.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.random_agent import RandomAgent
from agents.random_policy import GameUniformPolicy
from agents.utils import ChessEncoder
from games.chess import KRK
from games.game_utils import GymEnvFromGameAndPlayer2

SEED = 4
N_ACTIONS = ChessEncoder.n_actions


def build_krk_gym_env(
    start_position: int = 1,
    side_to_move: str = "white",
    seed: int = SEED,
    max_steps: int | None = 100,
) -> GymEnvFromGameAndPlayer2:
    """
    Construye el entorno exactamente como en el setup A2C:

        game = KRK(...)
        pl2 = RandomAgent(GameUniformPolicy(...))
        env = GymEnvFromGameAndPlayer2(game, pl2, ChessEncoder())
    """
    rng = default_rng(seed)
    game = KRK(start_position=start_position, side_to_move=side_to_move)
    random_policy = GameUniformPolicy(game=game, rng=rng, encoder=None)
    pl2 = RandomAgent(policy=random_policy, rng=rng)
    encoder = ChessEncoder()
    return GymEnvFromGameAndPlayer2(
        game=game,
        other_player=pl2,
        encoder=encoder,
        rng=rng,
        max_steps=max_steps,
    )


def board_snapshot(env: GymEnvFromGameAndPlayer2, *, start_position: int, side_to_move: str) -> dict:
    """Serializa el tablero actual del GymEnv para la UI."""
    board: chess.Board = env.state
    game = env.game
    steps = getattr(env, "_steps", 0)
    max_steps = env.max_steps or 0

    pieces = []
    for square, piece in board.piece_map().items():
        pieces.append(
            {
                "square": chess.square_name(square),
                "file": chess.square_file(square),
                "rank": chess.square_rank(square),
                "symbol": piece.symbol(),
                "type": piece.piece_type,
                "color": "white" if piece.color == chess.WHITE else "black",
            }
        )

    last_move = None
    move_trail: list[str] = []
    if board.move_stack:
        tmp = board.copy()
        undone: list[chess.Move] = []
        while tmp.move_stack and len(undone) < 12:
            undone.append(tmp.pop())
        for mv in reversed(undone):
            move_trail.append(tmp.san(mv))
            tmp.push(mv)
        last_move = move_trail[-1] if move_trail else None

    return {
        "kind": "krk",
        "size": 8,
        "fen": board.fen(),
        "turn": game.player(board),
        "pieces": pieces,
        "last_move": last_move,
        "move_trail": move_trail,
        "steps": steps,
        "max_steps": max_steps,
        "start_position": start_position,
        "side_to_move": side_to_move,
        "n_positions": len(KRK._POSITIONS),
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_terminal": game.is_terminal(board),
        "legal_moves": [board.san(m) for m in game.actions(board)[:24]],
        "svg": game.board_svg(board, size=480),
    }
