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
    infer_search_mode,
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


def evaluated_record(**changes: object) -> dict:
    record = {
        "outcome": "evaluated",
        "evaluation_profile": "F300",
        "linear_case_count": 50,
        "attention_case_count": 250,
    }
    record.update(changes)
    return record


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

    def test_pareto_excludes_pruned_solution_artifacts(self) -> None:
        self.registry["versions"]["v0_score"]["artifact_state"] = "pruned"
        archive = build_pareto_archive(self.registry)
        names = {item["version"] for item in archive["front"]}
        self.assertEqual(names, {"v0_linear"})
        self.assertEqual(archive["champions"]["score"], "v0_linear")

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
            evaluated_record(completed_at="1", new_global_best=True),
            evaluated_record(completed_at="2", new_global_best=False),
            evaluated_record(completed_at="3", new_global_best=False),
        ]
        self.assertEqual(stagnation_length(experiments), 2)

    def test_partial_diagnostic_does_not_change_stagnation(self) -> None:
        experiments = [
            evaluated_record(completed_at="1", new_global_best=True),
            {
                "outcome": "evaluated", "evaluation_profile": "F10",
                "linear_case_count": 5, "attention_case_count": 5,
                "completed_at": "2", "new_global_best": False,
            },
        ]
        self.assertEqual(stagnation_length(experiments), 0)

    def test_memory_context_is_bounded(self) -> None:
        experiments = [
            evaluated_record(
                completed_at=str(index),
                focus="attention", score_delta=float(index - 5),
                version=f"v{index}_x",
            )
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
            evaluated_record(
                score_delta=10.0, seconds=1800,
                new_global_best=True, algorithm_family="a",
                screening={"winner_agreement": True}, completed_at="1",
            ),
            evaluated_record(
                score_delta=-2.0, seconds=1800,
                new_global_best=False, algorithm_family="b",
                screening={"winner_agreement": False}, completed_at="2",
            ),
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

    def test_search_mode_inference_prefers_explicit_then_scratch(self) -> None:
        self.assertEqual(infer_search_mode({"search_mode": "exploit", "novelty": 1}), "exploit")
        self.assertEqual(infer_search_mode({"implementation_base": "scratch"}), "explore")
        self.assertEqual(infer_search_mode({"novelty": 0.8}), "explore")
        self.assertEqual(infer_search_mode({"novelty": 0.4}), "exploit")

    def test_dispatch_enforces_four_explore_two_exploit_portfolio(self) -> None:
        pending = [
            {
                "task_id": f"e{index}", "search_mode": "explore",
                "focus": f"focus_e{index}", "algorithm_family": f"family_e{index}",
                "priority": 20 - index,
            }
            for index in range(5)
        ] + [
            {
                "task_id": f"x{index}", "search_mode": "exploit",
                "focus": f"focus_x{index}", "algorithm_family": f"family_x{index}",
                "priority": 10 - index,
            }
            for index in range(3)
        ]
        selected = select_diverse_batch(
            pending, 6, explore_slots=4, exploit_slots=2,
        )
        modes = [infer_search_mode(task) for task in selected]
        self.assertEqual(modes.count("explore"), 4)
        self.assertEqual(modes.count("exploit"), 2)

    def test_dispatch_quota_accounts_for_active_tasks(self) -> None:
        active = [
            {"search_mode": "explore"} for _ in range(3)
        ] + [{"search_mode": "exploit"}]
        pending = [
            {"task_id": "e", "search_mode": "explore", "priority": 2},
            {"task_id": "x", "search_mode": "exploit", "priority": 1},
        ]
        selected = select_diverse_batch(
            pending, 2, active=active, explore_slots=4, exploit_slots=2,
        )
        self.assertEqual(
            {infer_search_mode(task) for task in selected}, {"explore", "exploit"},
        )

    def test_dispatch_diversity_accounts_for_active_tasks(self) -> None:
        active = [
            {
                "search_mode": "explore", "focus": f"e{index}",
                "algorithm_family": f"ef{index}",
            }
            for index in range(4)
        ] + [{
            "search_mode": "exploit", "focus": "attention",
            "algorithm_family": "same_family",
        }]
        pending = [
            {
                "task_id": "duplicate", "search_mode": "exploit",
                "focus": "attention", "algorithm_family": "same_family",
                "priority": 10,
            },
            {
                "task_id": "diverse", "search_mode": "exploit",
                "focus": "linear", "algorithm_family": "new_family",
                "priority": 1,
            },
        ]
        selected = select_diverse_batch(
            pending, 1, active=active, explore_slots=4, exploit_slots=2,
        )
        self.assertEqual(selected[0]["task_id"], "diverse")

    def test_dispatch_borrows_unused_exploit_capacity(self) -> None:
        pending = [
            {
                "task_id": f"e{index}", "search_mode": "explore",
                "priority": 10 - index,
            }
            for index in range(6)
        ]
        selected = select_diverse_batch(
            pending, 6, explore_slots=4, exploit_slots=2,
        )
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(infer_search_mode(task) == "explore" for task in selected))


if __name__ == "__main__":
    unittest.main()
