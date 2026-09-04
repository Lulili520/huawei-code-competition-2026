from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


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

    def test_reference_cache_reuses_exact_candidate_independent_outputs(self) -> None:
        quant_weight = torch.ones((4, 64), dtype=torch.bfloat16)
        scale_weight = torch.ones((4, 4), dtype=torch.bfloat16)
        quant_activation = torch.ones((2, 64), dtype=torch.bfloat16)
        scale_activation = torch.ones((2, 4), dtype=torch.bfloat16)
        group = {
            "weight_quant": quant_weight,
            "weight_scale": scale_weight,
            "test_activation_list": [(quant_activation, scale_activation)],
        }
        with tempfile.TemporaryDirectory(prefix="hif4_cache_test_") as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps({"dataset": "synthetic"}), encoding="utf-8",
            )
            cache = EVALUATE.ReferenceCache(root, enabled=False)
            first = cache.get(
                "linear", 0, 1,
                lambda: EVALUATE.build_linear_reference_cases(group),
            )
            second = cache.get(
                "linear", 0, 1,
                lambda: self.fail("in-memory reference cache was not reused"),
            )
        direct = (
            EVALUATE.decode_nvfp4(quant_activation, scale_activation)
            @ EVALUATE.decode_nvfp4(quant_weight, scale_weight).T
        )
        self.assertTrue(torch.equal(first["cases"][0]["reference"], direct))
        self.assertIs(first, second)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)


if __name__ == "__main__":
    unittest.main()
