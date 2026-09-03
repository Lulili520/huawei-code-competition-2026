from __future__ import annotations

import sys
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))
from core.strategy import (  # noqa: E402
    adaptive_priority,
    build_pareto_archive,
    build_process_metrics,
    compact_research_context,
    proposal_fingerprint,
    select_diverse_batch,
    stagnation_length,
)


def node(score: float, linear: float, attention: float, seconds: float) -> dict:
    return {
        "status": "evaluated",
        "method": "test",
        "metrics": {
            "score": score,
            "linear_mse": linear,
            "attention_mse": attention,
            "seconds": seconds,
            "dataset": "datasets/combined",
            "linear_case_count": 50,
            "attention_case_count": 250,
        },
    }


class StrategyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = {
            "versions": {
                "v0_score": node(100.0, 0.10, 0.10, 20.0),
                "v0_linear": node(95.0, 0.08, 0.11, 22.0),
                "v0_dominated": node(90.0, 0.12, 0.12, 30.0),
            }
        }

    def task(self, **changes: object) -> dict:
        value = {
            "based_on": "v0_score",
            "algorithm_family": "new_family",
            "focus": "attention",
            "implementation_base": "based_on",
            "evidence_strength": 0.5,
            "novelty": 0.5,
            "uncertainty": 0.5,
            "expected_cost": 1.0,
            "priority_hint": 0.5,
        }
        value.update(changes)
        return value

    def test_pareto_discards_strictly_dominated_version(self) -> None:
        archive = build_pareto_archive(self.registry)
        names = {item["version"] for item in archive["front"]}
        self.assertEqual(names, {"v0_score", "v0_linear"})
        self.assertEqual(archive["champions"]["score"], "v0_score")
        self.assertEqual(archive["champions"]["linear_mse"], "v0_linear")

    def test_priority_rewards_evidence_and_penalizes_cost(self) -> None:
        low, _ = adaptive_priority(
            self.task(evidence_strength=0.1, expected_cost=2.5), self.registry, []
        )
        high, _ = adaptive_priority(
            self.task(evidence_strength=0.9, expected_cost=0.5), self.registry, []
        )
        self.assertGreater(high, low)

    def test_linear_task_rewards_linear_pareto_source(self) -> None:
        score_source, _ = adaptive_priority(
            self.task(based_on="v0_score", focus="linear"), self.registry, []
        )
        linear_source, _ = adaptive_priority(
            self.task(based_on="v0_linear", focus="linear"), self.registry, []
        )
        self.assertGreater(linear_source, score_source)

    def test_family_failure_reduces_priority(self) -> None:
        clean, _ = adaptive_priority(self.task(), self.registry, [])
        failed, _ = adaptive_priority(self.task(), self.registry, [{
            "outcome": "failed", "algorithm_family": "new_family",
        }])
        self.assertLess(failed, clean)

    def test_stagnation_stops_at_last_global_best(self) -> None:
        experiments = [
            {"outcome": "evaluated", "completed_at": "1", "new_global_best": True},
            {"outcome": "evaluated", "completed_at": "2", "new_global_best": False},
            {"outcome": "evaluated", "completed_at": "3", "new_global_best": False},
        ]
        self.assertEqual(stagnation_length(experiments), 2)

    def test_memory_context_is_bounded(self) -> None:
        experiments = [
            {
                "outcome": "evaluated", "completed_at": str(index),
                "focus": "attention", "score_delta": float(index - 5),
                "version": f"v{index}_x",
            }
            for index in range(12)
        ]
        context = compact_research_context(
            self.task(), [], experiments, {"front": []}, limit=3
        )
        self.assertLessEqual(len(context["relevant_positive_experiments"]), 3)
        self.assertLessEqual(len(context["relevant_negative_experiments"]), 3)

    def test_proposal_fingerprint_is_stable(self) -> None:
        left = proposal_fingerprint("QK_Search", "Joint  search", "change blocks")
        right = proposal_fingerprint("qk_search", "joint search", "change blocks")
        self.assertEqual(left, right)

    def test_process_metrics_measure_screening_and_budget_efficiency(self) -> None:
        metrics = build_process_metrics([
            {
                "outcome": "evaluated", "score_delta": 10.0, "seconds": 1800,
                "new_global_best": True, "algorithm_family": "a",
                "screening": {"winner_agreement": True}, "completed_at": "1",
            },
            {
                "outcome": "evaluated", "score_delta": -2.0, "seconds": 1800,
                "new_global_best": False, "algorithm_family": "b",
                "screening": {"winner_agreement": False}, "completed_at": "2",
            },
            {"outcome": "evaluation_timeout", "algorithm_family": "b"},
        ])
        self.assertEqual(metrics["screening_winner_agreement_rate"], 0.5)
        self.assertEqual(metrics["positive_score_gain_per_evaluation_hour"], 10.0)
        self.assertEqual(metrics["non_evaluation_outcomes"], {"evaluation_timeout": 1})

    def test_dispatch_batch_covers_focus_and_family(self) -> None:
        pending = [
            {"task_id": "a1", "focus": "attention", "algorithm_family": "a", "priority": 10},
            {"task_id": "a2", "focus": "attention", "algorithm_family": "a", "priority": 9},
            {"task_id": "b1", "focus": "linear", "algorithm_family": "b", "priority": 8},
            {"task_id": "c1", "focus": "format", "algorithm_family": "c", "priority": 7},
        ]
        selected = select_diverse_batch(pending, 3)
        self.assertEqual({task["focus"] for task in selected}, {
            "attention", "linear", "format",
        })
        self.assertNotIn("a2", {task["task_id"] for task in selected})


if __name__ == "__main__":
    unittest.main()
