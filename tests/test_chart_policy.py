"""Visualization chart policy must drive discrete action selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from env_runner import _sample_from_chart_policy  # noqa: E402


class ChartPolicySamplerTests(unittest.TestCase):
    def test_epsilon_zero_is_greedy(self) -> None:
        actions = {
            _sample_from_chart_policy([1.0, 0.2, 0.0], "epsilon", epsilon=0.0)
            for _ in range(40)
        }
        self.assertEqual(actions, {0})

    def test_epsilon_one_explores_all_actions(self) -> None:
        actions = {
            _sample_from_chart_policy([100.0, 0.0], "epsilon", epsilon=1.0)
            for _ in range(80)
        }
        self.assertEqual(actions, {0, 1})

    def test_softmax_low_temperature_is_greedy(self) -> None:
        actions = {
            _sample_from_chart_policy([5.0, 0.0], "softmax", temperature=0.01)
            for _ in range(40)
        }
        self.assertEqual(actions, {0})

    def test_unknown_mode_defaults_to_epsilon_greedy(self) -> None:
        actions = {
            _sample_from_chart_policy([0.0, 3.0], "q", epsilon=0.0)
            for _ in range(20)
        }
        self.assertEqual(actions, {1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
