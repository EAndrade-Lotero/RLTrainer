"""SB3 discrete-action networks expose Q-values for the Visualization chart."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app import app  # noqa: E402
from env_runner import _sb3_network_q_values, create_sb3_model  # noqa: E402
import gymnasium as gym  # noqa: E402


class Sb3NetworkQValueTests(unittest.TestCase):
    def test_dqn_returns_one_q_value_per_discrete_action(self) -> None:
        env = gym.make("CartPole-v1")
        self.addCleanup(env.close)
        model = create_sb3_model(
            "DQN",
            {"environment": "CartPole-v1", "learning_rate": 3e-4},
        )
        observation, _info = env.reset(seed=0)
        payload = _sb3_network_q_values(model, observation, env)
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["values"]), int(env.action_space.n))
        self.assertTrue(payload["greedy_actions"])
        self.assertTrue(all(np.isfinite(payload["values"])))

        serialized = _sb3_network_q_values(model, observation.tolist(), env)
        self.assertIsNotNone(serialized)
        np.testing.assert_allclose(payload["values"], serialized["values"])

    def test_ppo_returns_action_scores_for_discrete_actions(self) -> None:
        env = gym.make("CartPole-v1")
        self.addCleanup(env.close)
        model = create_sb3_model(
            "PPO",
            {"environment": "CartPole-v1", "learning_rate": 3e-4},
        )
        observation, _info = env.reset(seed=0)
        payload = _sb3_network_q_values(model, observation, env)
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["values"]), 2)

    def test_continuous_action_model_has_no_q_vector(self) -> None:
        env = gym.make("Pendulum-v1")
        self.addCleanup(env.close)
        model = create_sb3_model(
            "SAC",
            {"environment": "Pendulum-v1", "learning_rate": 3e-4},
        )
        observation, _info = env.reset(seed=0)
        self.assertIsNone(_sb3_network_q_values(model, observation, env))


class Sb3QValueApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_initial_frame_includes_dqn_q_values(self) -> None:
        response = self.client.post(
            "/api/config",
            json={
                "environment": "CartPole-v1",
                "agent": "DQN",
                "learning_rate": 0.1,
                "exploration_probability": 0.0,
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())

        payload = self.client.get("/api/environment/initial").get_json()
        q_values = payload.get("q_values")
        self.assertIsNotNone(q_values)
        self.assertEqual(len(q_values["values"]), 2)

        action = self.client.post(
            "/api/environment/run-action",
            json={"action": "policy"},
        )
        self.assertEqual(action.status_code, 200, action.get_json())
        self.assertEqual(len(action.get_json()["q_values"]["values"]), 2)

    def test_analysis_samples_discrete_sb3_states(self) -> None:
        response = self.client.post(
            "/api/config",
            json={
                "environment": "CartPole-v1",
                "agent": "DQN",
                "learning_rate": 0.1,
                "exploration_probability": 0.2,
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())

        payload = self.client.get("/api/agent/analysis").get_json()
        self.assertIsNone(payload.get("error"), payload)
        sampled = payload.get("sampled_states") or []
        self.assertGreater(len(sampled), 0)
        self.assertLessEqual(len(sampled), 10)
        self.assertEqual(payload.get("sample_episodes"), 10)
        self.assertEqual(payload.get("n_actions"), 2)
        self.assertFalse(payload.get("q_table"))
        for item in sampled:
            self.assertEqual(len(item["q_values"]), 2)
            self.assertTrue(item.get("image"))
            self.assertTrue(item.get("state_label"))

    def test_analysis_rejects_continuous_action_sb3(self) -> None:
        response = self.client.post(
            "/api/config",
            json={
                "environment": "Pendulum-v1",
                "agent": "SAC",
                "learning_rate": 0.1,
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        analysis = self.client.get("/api/agent/analysis")
        self.assertEqual(analysis.status_code, 400)
        self.assertIn("discrete action", analysis.get_json()["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
