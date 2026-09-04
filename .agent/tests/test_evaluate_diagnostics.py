from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills" / "hif4-evaluate" / "scripts"
)
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("hif4_evaluate", SCRIPTS / "evaluate.py")
assert SPEC is not None and SPEC.loader is not None
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


class EvaluationDiagnosticsTest(unittest.TestCase):
    def test_evenly_spaced_screening_is_deterministic(self) -> None:
        selected = EVALUATE.select_groups(list(range(10)), 3)
        self.assertEqual([index for index, _ in selected], [0, 4, 9])
        self.assertEqual([value for _, value in selected], [0, 4, 9])

    def test_score_statistics_reports_tail_and_negative_count(self) -> None:
        cases = [
            {"score_percentage_points": value}
            for value in (-10.0, 0.0, 20.0, 30.0, 60.0)
        ]
        stats = EVALUATE.score_statistics(cases)
        self.assertEqual(stats["minimum"], -10.0)
        self.assertEqual(stats["median"], 20.0)
        self.assertEqual(stats["negative_count"], 1)

    def test_invalid_group_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EVALUATE.select_groups([1, 2], 0)


if __name__ == "__main__":
    unittest.main()
