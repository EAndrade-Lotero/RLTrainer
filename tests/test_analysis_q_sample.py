"""Analysis Q-table samples visited states without changing learned Q."""

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


CONFIG = {
    "environment": "CliffWalking-v1",
    "agent": "Q_learning",
    "learning_rate": 0.5,
    "exploration_probability": 0.3,
    "discount_factor": 0.99,
}


class AnalysisQTableSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        response = self.client.post("/api/config", json=CONFIG)
        self.assertEqual(response.status_code, 200, response.get_json())

    def test_q_table_endpoint_does_not_sample_states(self) -> None:
        payload = self.client.get("/api/agent/q-table").get_json()
        self.assertNotIn("sampled_states", payload)
        self.assertIn("q_table", payload)

    def test_analysis_samples_up_to_ten_visited_states(self) -> None:
        payload = self.client.get("/api/agent/analysis").get_json()
        self.assertEqual(payload.get("sample_episodes"), 10)
        self.assertEqual(payload.get("sample_timesteps"), 100)
        sampled = payload.get("sampled_states") or []
        self.assertGreater(len(sampled), 0)
        self.assertLessEqual(len(sampled), 10)
        self.assertEqual(payload.get("sample_size"), len(sampled))
        self.assertGreaterEqual(payload.get("visited_unique"), len(sampled))

        states = [item["state"] for item in sampled]
        self.assertEqual(len(states), len(set(states)))
        for item in sampled:
            self.assertEqual(len(item["q_values"]), payload["n_actions"])
            self.assertTrue(item.get("image"))
            self.assertIn("greedy_actions", item)
            self.assertGreaterEqual(item["state"], 0)
            self.assertLess(item["state"], payload["n_states"])

    def test_analysis_sampling_does_not_change_q_table(self) -> None:
        learn = self.client.post(
            "/api/environment/run-action",
            json={"action": 0, "allow_learning": True},
        )
        self.assertEqual(learn.status_code, 200, learn.get_json())
        trained = np.asarray(
            self.client.get("/api/agent/q-table").get_json()["q_table"],
            dtype=float,
        )
        analysis = self.client.get("/api/agent/analysis")
        self.assertEqual(analysis.status_code, 200, analysis.get_json())
        after = np.asarray(
            self.client.get("/api/agent/q-table").get_json()["q_table"],
            dtype=float,
        )
        np.testing.assert_allclose(after, trained)


if __name__ == "__main__":
    unittest.main(verbosity=2)
