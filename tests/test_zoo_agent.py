"""Load pretrained RL Baselines3 Zoo agents from the Hugging Face Hub."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app import app  # noqa: E402
from env_runner import zoo_env_candidates, zoo_hub_refs  # noqa: E402


class ZooNamingTests(unittest.TestCase):
    def test_hub_refs_match_zoo_enjoy_convention(self) -> None:
        refs = zoo_hub_refs("PPO", "CartPole-v1")
        self.assertEqual(refs["algo"], "ppo")
        self.assertEqual(refs["organization"], "sb3")
        self.assertEqual(refs["model_name"], "ppo-CartPole-v1")
        self.assertEqual(refs["repo_id"], "sb3/ppo-CartPole-v1")
        self.assertEqual(refs["filename"], "ppo-CartPole-v1.zip")

    def test_taxi_falls_back_to_older_gym_id(self) -> None:
        self.assertEqual(zoo_env_candidates("Taxi-v4"), ("Taxi-v4", "Taxi-v3"))


class ZooAgentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_tabular_agent_is_rejected(self) -> None:
        response = self.client.post(
            "/api/agent/zoo",
            json={"environment": "FrozenLake-v1", "agent": "Q_learning"},
        )
        self.assertEqual(response.status_code, 400, response.get_json())
        self.assertIn("Stable-Baselines3", response.get_json()["error"])

    def test_incompatible_pair_is_rejected(self) -> None:
        response = self.client.post(
            "/api/agent/zoo",
            json={"environment": "Pendulum-v1", "agent": "DQN"},
        )
        self.assertEqual(response.status_code, 400, response.get_json())

    @mock.patch("huggingface_hub.hf_hub_download")
    def test_missing_hub_model_returns_404(self, download) -> None:
        download.side_effect = Exception("404 Client Error: Not Found for url")
        response = self.client.post(
            "/api/agent/zoo",
            json={"environment": "CartPole-v1", "agent": "PPO"},
        )
        self.assertEqual(response.status_code, 404, response.get_json())
        payload = response.get_json()
        self.assertIn("No RL Baselines3 Zoo agent found", payload["error"])
        self.assertEqual(payload["repo_id"], "sb3/ppo-CartPole-v1")

    @mock.patch("huggingface_hub.hf_hub_download")
    def test_loads_checkpoint_and_can_run_an_episode(self, download) -> None:
        from stable_baselines3 import PPO

        tmp = Path(tempfile.mkdtemp())
        stem = tmp / "ppo-CartPole-v1"
        PPO("MlpPolicy", "CartPole-v1", n_steps=8, verbose=0).save(str(stem))
        zip_path = Path(f"{stem}.zip")
        self.assertTrue(zip_path.is_file())
        download.return_value = str(zip_path)

        response = self.client.post(
            "/api/agent/zoo",
            json={
                "environment": "CartPole-v1",
                "agent": "PPO",
                "learning_rate": 0.1,
                "exploration_probability": 0.0,
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload["status"], "loaded")
        self.assertEqual(payload["kind"], "Stable-Baselines3 Zoo model")
        self.assertEqual(payload["repo_id"], "sb3/ppo-CartPole-v1")
        self.assertEqual(payload["agent"], "PPO")
        self.assertEqual(payload["environment"], "CartPole-v1")
        download.assert_called()

        episode = self.client.post(
            "/api/environment/run-episode",
            json={"action": "policy"},
        )
        self.assertEqual(episode.status_code, 200, episode.get_json())
        body = episode.get_json()
        frames = body.get("frames") or []
        self.assertTrue(frames, "Expected episode frames after loading the Zoo agent")
        self.assertIn("image", frames[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
