"""Controla el bucle de entrenamiento A2C y expone snapshots thread-safe."""
from __future__ import annotations

import threading
import time
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from .agent import A2CAgent
from .environment import board_snapshot

MAX_HISTORY = 300


class _TelemetryCallback(BaseCallback):
    """Recoge recompensas/longitudes de episodio durante `model.learn`."""

    def __init__(self, trainer: "Trainer"):
        super().__init__()
        self.trainer = trainer

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                with self.trainer.lock:
                    self.trainer.episode += 1
                    self.trainer.reward_history.append(float(ep["r"]))
                    self.trainer.steps_history.append(int(ep["l"]))
                    self.trainer.loss_history.append(self.trainer.agent.last_loss)
                    self.trainer.episode_reward = 0.0
        rewards = self.locals.get("rewards")
        if rewards is not None:
            with self.trainer.lock:
                self.trainer.episode_reward += float(np_asarray0(rewards))
                self.trainer.total_steps += 1
                self.trainer.last_action = self.trainer._last_move_san()
        return (not self.trainer._stop) and self.trainer.running


def np_asarray0(rewards) -> float:
    try:
        return float(rewards[0])
    except Exception:
        return float(rewards)


class Trainer:
    def __init__(self):
        self.lock = threading.Lock()
        self.start_position = 1
        self.side_to_move = "white"
        self._build(start_position=1, side_to_move="white")
        self.running = False
        self.speed = 8
        self.episode = 0
        self.episode_reward = 0.0
        self.reward_history = deque(maxlen=MAX_HISTORY)
        self.steps_history = deque(maxlen=MAX_HISTORY)
        self.loss_history = deque(maxlen=MAX_HISTORY)
        self.td_error_ema = 0.0
        self.last_action = None
        self.last_exploration = True
        self.total_steps = 0
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _build(self, start_position, side_to_move):
        self.agent = A2CAgent(
            start_position=start_position,
            side_to_move=side_to_move,
        )
        self.raw_env = self.agent.raw_env
        self.callback = _TelemetryCallback(self)

    def _last_move_san(self):
        board = self.raw_env.state
        if not board.move_stack:
            return None
        last = board.peek()
        board.pop()
        try:
            return board.san(last)
        finally:
            board.push(last)

    def _loop(self):
        while not self._stop:
            with self.lock:
                running = self.running
                delay = max(1.0 / max(self.speed, 1), 0.01)
            if not running:
                time.sleep(0.05)
                continue

            try:
                self.agent.learn(total_timesteps=1, callback=self.callback)
            except Exception:
                pass

            with self.lock:
                self.td_error_ema = (
                    0.9 * self.td_error_ema + 0.1 * abs(self.agent.last_loss)
                )

            time.sleep(delay)

    def start(self):
        with self.lock:
            self.running = True

    def pause(self):
        with self.lock:
            self.running = False

    def set_speed(self, speed):
        with self.lock:
            self.speed = max(1, min(int(speed), 60))

    def reset(self, start_position=None, side_to_move=None):
        with self.lock:
            if start_position is not None:
                self.start_position = int(start_position)
            if side_to_move is not None:
                self.side_to_move = side_to_move.lower()
            self._build(
                start_position=self.start_position,
                side_to_move=self.side_to_move,
            )
            self.episode = 0
            self.episode_reward = 0.0
            self.reward_history.clear()
            self.steps_history.clear()
            self.loss_history.clear()
            self.td_error_ema = 0.0
            self.total_steps = 0
            self.last_action = None
            self.running = False

    def state_snapshot(self):
        with self.lock:
            recent = list(self.reward_history)[-50:]
            avg_recent = sum(recent) / len(recent) if recent else 0.0
            best = max(self.reward_history) if self.reward_history else 0.0
            env_snap = board_snapshot(
                self.raw_env,
                start_position=self.start_position,
                side_to_move=self.side_to_move,
            )
            return {
                "running": self.running,
                "speed": self.speed,
                "environment": env_snap,
                "agent": {
                    "episode": self.episode,
                    "epsilon": round(self.agent.learning_rate, 6),
                    "alpha": self.agent.learning_rate,
                    "gamma": self.agent.gamma,
                    "total_steps": self.total_steps,
                    "current_episode_reward": round(self.episode_reward, 2),
                    "avg_reward_last_50": round(avg_recent, 2),
                    "best_episode_reward": round(best, 2),
                    "td_error_ema": round(self.td_error_ema, 4),
                    "last_action": self.last_action,
                    "last_exploration": self.last_exploration,
                    "reward_history": list(self.reward_history),
                    "steps_history": list(self.steps_history),
                    "epsilon_history": list(self.loss_history),
                    "value_grid": self.agent.value_grid(8),
                    "algo": "A2C",
                    "n_updates": self.agent.n_updates,
                },
            }


trainer = Trainer()
