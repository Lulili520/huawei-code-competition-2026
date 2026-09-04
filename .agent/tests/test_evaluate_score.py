from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCORING_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills" / "hif4-evaluate" / "scripts" / "scoring.py"
)
SPEC = importlib.util.spec_from_file_location("hif4_scoring", SCORING_PATH)
assert SPEC is not None and SPEC.loader is not None
SCORING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORING)


class WeightedScoreTest(unittest.TestCase):
    def test_attention_has_five_times_linear_weight(self) -> None:
        baseline = SCORING.weighted_total_score([0.0] * 5, [0.0] * 5)
        linear_gain = SCORING.weighted_total_score([1.0] * 5, [0.0] * 5)
        attention_gain = SCORING.weighted_total_score([0.0] * 5, [1.0] * 5)
        self.assertEqual(baseline, 0.0)
        self.assertAlmostEqual(attention_gain, 5.0 * linear_gain)

    def test_score_scale_stays_equal_for_equal_case_scores(self) -> None:
        score = SCORING.weighted_total_score([0.6] * 5, [0.6] * 5)
        self.assertAlmostEqual(score, 18000.0)

    def test_each_category_is_averaged_before_weighting(self) -> None:
        score = SCORING.weighted_total_score([1.0], [0.0] * 5)
        self.assertAlmostEqual(score, 5000.0)

    def test_score_scale_is_independent_of_sample_count(self) -> None:
        short = SCORING.weighted_total_score([0.5], [0.25])
        fixed = SCORING.weighted_total_score([0.5] * 5, [0.25] * 5)
        self.assertAlmostEqual(short, fixed)

    def test_both_categories_are_required(self) -> None:
        with self.assertRaises(ValueError):
            SCORING.weighted_total_score([], [1.0])


if __name__ == "__main__":
    unittest.main()
