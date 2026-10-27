"""Budget invariants: a bad timing or restart must not silently extend a job."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from control import choose_updates, init_budget


class BudgetTests(unittest.TestCase):
    def test_fast_run_is_capped(self):
        self.assertEqual(choose_updates(0.5, 10800), 2000)

    def test_slow_run_is_scaled_down_with_headroom(self):
        target = choose_updates(10, 3600)
        self.assertEqual(target, 250)
        self.assertLessEqual((target - 50) * 10, 3600 * 0.8)

    def test_no_time_never_adds_updates(self):
        self.assertEqual(choose_updates(1, 0), 50)
        self.assertEqual(choose_updates(1, -10), 50)

    def test_invalid_timing_is_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                choose_updates(value, 3600)

    def test_restart_cannot_reset_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "budget.json"
            old = {"started_at": time.time() - 3600, "deadline": time.time() + 60}
            path.write_text(json.dumps(old))
            state = init_budget(path, 7 * 3600)
            self.assertEqual(state["deadline"], old["deadline"])
            self.assertEqual(state["started_at"], old["started_at"])


if __name__ == "__main__":
    unittest.main()
