from __future__ import annotations

import hashlib
import math
import re
from typing import Any


CURRENT_DATASET = "datasets/combined"
CURRENT_LINEAR_CASES = 50
CURRENT_ATTENTION_CASES = 250


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, float(value)))


def _words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value.lower()))


def proposal_fingerprint(
    family: str, hypothesis: str, structural_change: str = "",
) -> str:
    material = "|".join((_words(family), _words(hypothesis), _words(structural_change)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def infer_search_mode(task: dict[str, Any]) -> str:
    """Classify legacy proposals while preferring an explicit research role."""
    explicit = str(task.get("search_mode", "")).lower()
    if explicit in {"explore", "exploit"}:
        return explicit
    if task.get("implementation_base") == "scratch":
        return "explore"
    if float(task.get("novelty", 0.5)) >= 0.72:
        return "explore"
    return "exploit"


def metrics_are_current(metrics: dict[str, Any] | None) -> bool:
    return bool(
        metrics
        and metrics.get("dataset") == CURRENT_DATASET
        and int(metrics.get("linear_case_count", -1)) == CURRENT_LINEAR_CASES
        and int(metrics.get("attention_case_count", -1)) == CURRENT_ATTENTION_CASES
    )


def _eligible_versions(registry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    bad = {"failed", "draft", "environment_failed", "evaluation_timeout", "invalid_after_evaluation"}
    return [
        (name, node)
        for name, node in registry.get("versions", {}).items()
        if node.get("status") not in bad and metrics_are_current(node.get("metrics"))
    ]


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lm, rm = left["metrics"], right["metrics"]
    left_values = (
        float(lm["score"]),
        -float(lm["linear_mse"]),
        -float(lm["attention_mse"]),
        -float(lm.get("seconds", math.inf)),
    )
    right_values = (
        float(rm["score"]),
        -float(rm["linear_mse"]),
        -float(rm["attention_mse"]),
        -float(rm.get("seconds", math.inf)),
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def build_pareto_archive(registry: dict[str, Any]) -> dict[str, Any]:
    eligible = _eligible_versions(registry)
    front: list[dict[str, Any]] = []
    for name, node in eligible:
        if any(_dominates(other, node) for other_name, other in eligible if other_name != name):
            continue
        metrics = node["metrics"]
        front.append({
            "version": name,
            "algorithm_family": node.get("algorithm_family", node.get("method", "unknown")),
            "score": float(metrics["score"]),
            "linear_mse": float(metrics["linear_mse"]),
            "attention_mse": float(metrics["attention_mse"]),
            "seconds": float(metrics.get("seconds", math.inf)),
        })
    front.sort(key=lambda item: (-item["score"], item["seconds"], item["version"]))

    def champion(metric: str, *, maximize: bool = False) -> str | None:
        if not eligible:
            return None
        key = (
            (lambda item: -float(item[1]["metrics"][metric]))
            if maximize
            else (lambda item: float(item[1]["metrics"].get(metric, math.inf)))
        )
        return min(eligible, key=lambda item: (key(item), item[0]))[0]

    return {
        "schema_version": 1,
        "objectives": {
            "score": "maximize",
            "linear_mse": "minimize",
            "attention_mse": "minimize",
            "seconds": "minimize",
        },
        "front": front,
        "champions": {
            "score": champion("score", maximize=True),
            "linear_mse": champion("linear_mse"),
            "attention_mse": champion("attention_mse"),
            "seconds": champion("seconds"),
        },
    }


def stagnation_length(experiments: list[dict[str, Any]]) -> int:
    evaluated = [item for item in experiments if item.get("outcome") == "evaluated"]
    evaluated.sort(key=lambda item: item.get("completed_at", ""), reverse=True)
    count = 0
    for item in evaluated:
        if item.get("new_global_best"):
            break
        count += 1
    return count


def adaptive_priority(
    task: dict[str, Any], registry: dict[str, Any],
    experiments: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    """Bandit-like ranking over evidence, novelty, cost, and observed family yield."""
    eligible = _eligible_versions(registry)
    best_score = max((float(node["metrics"]["score"]) for _, node in eligible), default=1.0)
    source = registry.get("versions", {}).get(task.get("based_on"), {})
    source_metrics = source.get("metrics") or {}
    source_score = float(source_metrics.get("score", 0.0))
    score_quality = _clamp(source_score / max(abs(best_score), 1.0), 0.0, 1.1)
    focus = task.get("focus")
    target_quality = score_quality
    if eligible and focus in {"linear", "attention"}:
        metric = "linear_mse" if focus == "linear" else "attention_mse"
        best_error = min(float(node["metrics"][metric]) for _, node in eligible)
        source_error = float(source_metrics.get(metric, math.inf))
        target_quality = _clamp(
            best_error / max(source_error, 1e-30) if math.isfinite(source_error) else 0.0,
            0.0, 1.1,
        )
    elif eligible and task.get("target_metric") == "runtime":
        best_seconds = min(float(node["metrics"].get("seconds", math.inf)) for _, node in eligible)
        source_seconds = float(source_metrics.get("seconds", math.inf))
        target_quality = _clamp(
            best_seconds / max(source_seconds, 1e-30) if math.isfinite(source_seconds) else 0.0,
            0.0, 1.1,
        )

    evidence = _clamp(task.get("evidence_strength", 0.5))
    novelty = _clamp(task.get("novelty", 0.5))
    uncertainty = _clamp(task.get("uncertainty", 0.5))
    expected_cost = max(0.25, float(task.get("expected_cost", 1.0)))
    family = task.get("algorithm_family")
    family_runs = [
        item for item in experiments
        if item.get("outcome") == "evaluated" and item.get("algorithm_family") == family
    ]
    family_failures = [
        item for item in experiments
        if item.get("outcome") in {"failed", "evaluation_timeout"}
        and item.get("algorithm_family") == family
    ]
    relative_gains = []
    for item in family_runs:
        denominator = item.get("baseline_score")
        if not isinstance(denominator, (int, float)):
            denominator = best_score
        relative_gains.append(
            float(item.get("score_delta", 0.0))
            / max(abs(float(denominator)), 1.0)
        )
    family_reward = math.tanh(8.0 * (sum(relative_gains) / len(relative_gains))) if relative_gains else 0.0
    total_runs = sum(item.get("outcome") == "evaluated" for item in experiments)
    exploration = math.sqrt(math.log(total_runs + 2.0) / (len(family_runs) + 1.0))
    stagnation = stagnation_length(experiments)
    pivot_boost = min(stagnation, 6) / 6.0 * (
        0.7 * novelty + (0.3 if task.get("implementation_base") == "scratch" else 0.0)
    )
    manual_hint = _clamp(task.get("priority_hint", 0.5)) - 0.5
    components = {
        "source_score_quality": score_quality,
        "source_target_quality": target_quality,
        "evidence": 2.0 * evidence,
        "novelty": 1.5 * novelty,
        "uncertainty": 0.5 * uncertainty,
        "family_reward": 1.25 * family_reward,
        "exploration": 0.75 * exploration,
        "pivot_boost": pivot_boost,
        "failure_penalty": -0.4 * min(len(family_failures), 3),
        "cost_penalty": -0.75 * math.log1p(expected_cost),
        "manual_hint": 0.5 * manual_hint,
    }
    return sum(components.values()), {key: round(value, 6) for key, value in components.items()}


def compact_research_context(
    task: dict[str, Any], principles: list[dict[str, Any]],
    experiments: list[dict[str, Any]], pareto: dict[str, Any], *, limit: int = 4,
) -> dict[str, Any]:
    """Bound prompt growth while preserving relevant positive and negative evidence."""
    focus = task.get("focus")
    ranked = sorted(
        (
            item for item in experiments
            if item.get("outcome") in {"evaluated", "failed", "evaluation_timeout"}
            and not ("baseline" in item and item.get("baseline") is None)
        ),
        key=lambda item: (
            item.get("focus") == focus,
            abs(float(item.get("score_delta", 0.0))),
            item.get("completed_at", ""),
        ),
        reverse=True,
    )
    positives = [item for item in ranked if float(item.get("score_delta", 0.0)) > 0][:limit]
    negatives = [
        item for item in ranked
        if item.get("outcome") in {"failed", "evaluation_timeout"}
        or float(item.get("score_delta", 0.0)) <= 0
    ][:limit]
    active_principles = [
        {
            "id": item.get("id"),
            "statement": item.get("statement"),
            "status": item.get("status"),
            "support_count": item.get("support_count", 0),
            "contradiction_count": item.get("contradiction_count", 0),
        }
        for item in principles
        if item.get("status") in {"core", "active", "candidate"}
    ][:12]
    keep = (
        "version", "algorithm_family", "focus", "search_mode", "root_cause",
        "baseline", "baseline_score",
        "score", "score_delta", "linear_mse_delta", "attention_mse_delta",
        "seconds", "outcome", "failure", "takeaways", "hypothesis_outcomes",
    )
    compact = lambda item: {key: item[key] for key in keep if key in item}
    return {
        "active_principles": active_principles,
        "relevant_positive_experiments": [compact(item) for item in positives],
        "relevant_negative_experiments": [compact(item) for item in negatives],
        "pareto_front": pareto.get("front", [])[:8],
        "stagnation_length": stagnation_length(experiments),
        "context_policy": "只保留与当前方向最相关且有真实评测支撑的正反例；不要把记忆当作隐藏集事实。",
    }


def build_process_metrics(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    references = [
        item for item in experiments
        if item.get("outcome") == "evaluated"
        and "baseline" in item and item.get("baseline") is None
    ]
    evaluated = [
        item for item in experiments
        if item.get("outcome") == "evaluated" and item not in references
    ]
    failures: dict[str, int] = {}
    for item in experiments:
        outcome = str(item.get("outcome", "unknown"))
        if outcome != "evaluated":
            failures[outcome] = failures.get(outcome, 0) + 1
    positive = [item for item in evaluated if float(item.get("score_delta", 0.0)) > 0]
    seconds = sum(float(item.get("seconds") or 0.0) for item in evaluated)
    positive_gain = sum(max(0.0, float(item.get("score_delta", 0.0))) for item in evaluated)
    screened = [
        item for item in evaluated
        if item.get("screening", {}).get("winner_agreement") is not None
    ]
    agreement_count = sum(
        item["screening"]["winner_agreement"] is True for item in screened
    )
    families: dict[str, dict[str, Any]] = {}
    for item in experiments:
        if item in references:
            continue
        family = str(item.get("algorithm_family", "unknown"))
        entry = families.setdefault(family, {
            "attempts": 0, "evaluated": 0, "positive": 0,
            "score_delta_sum": 0.0, "seconds": 0.0,
        })
        entry["attempts"] += 1
        if item.get("outcome") == "evaluated":
            entry["evaluated"] += 1
            delta = float(item.get("score_delta", 0.0))
            entry["positive"] += delta > 0
            entry["score_delta_sum"] += delta
            entry["seconds"] += float(item.get("seconds") or 0.0)
    for entry in families.values():
        count = max(int(entry["evaluated"]), 1)
        entry["mean_score_delta"] = entry["score_delta_sum"] / count
        entry["positive_rate"] = entry["positive"] / count
    search_modes: dict[str, dict[str, Any]] = {}
    for item in experiments:
        if item in references:
            continue
        mode = infer_search_mode(item)
        entry = search_modes.setdefault(mode, {
            "attempts": 0, "evaluated": 0, "positive": 0,
            "score_delta_sum": 0.0, "seconds": 0.0,
        })
        entry["attempts"] += 1
        if item.get("outcome") == "evaluated":
            entry["evaluated"] += 1
            delta = float(item.get("score_delta", 0.0))
            entry["positive"] += delta > 0
            entry["score_delta_sum"] += delta
            entry["seconds"] += float(item.get("seconds") or 0.0)
    for entry in search_modes.values():
        count = max(int(entry["evaluated"]), 1)
        entry["mean_score_delta"] = entry["score_delta_sum"] / count
        entry["positive_rate"] = entry["positive"] / count
    return {
        "schema_version": 1,
        "reference_versions": len(references),
        "evaluated_versions": len(evaluated),
        "positive_versions": len(positive),
        "positive_rate": len(positive) / max(len(evaluated), 1),
        "new_global_bests": sum(item.get("new_global_best") is True for item in evaluated),
        "formal_evaluation_seconds": seconds,
        "positive_score_gain": positive_gain,
        "positive_score_gain_per_evaluation_hour": (
            positive_gain / (seconds / 3600.0) if seconds > 0 else 0.0
        ),
        "screening_comparisons": len(screened),
        "screening_winner_agreements": agreement_count,
        "screening_winner_agreement_rate": (
            agreement_count / len(screened) if screened else None
        ),
        "stagnation_length": stagnation_length(experiments),
        "non_evaluation_outcomes": failures,
        "families": families,
        "search_modes": search_modes,
    }


def select_diverse_batch(
    pending: list[dict[str, Any]], limit: int, *,
    active: list[dict[str, Any]] | None = None,
    explore_slots: int = 4, exploit_slots: int = 2,
) -> list[dict[str, Any]]:
    """Fill a global exploration/exploitation portfolio and diversify each side.

    Unused capacity may be borrowed when one side lacks valid candidates.  The
    scheduler never fabricates a weak task merely to fill a process slot.
    """
    if limit <= 0:
        return []
    if explore_slots < 0 or exploit_slots < 0 or explore_slots + exploit_slots < 1:
        raise ValueError("invalid exploration/exploitation slot configuration")
    ordered = sorted(
        pending,
        key=lambda task: (-float(task.get("priority", 0.0)), task.get("created_at", "")),
    )
    active = active or []
    active_explore = sum(infer_search_mode(task) == "explore" for task in active)
    active_exploit = sum(infer_search_mode(task) == "exploit" for task in active)
    need_explore = max(0, explore_slots - active_explore)
    need_exploit = max(0, exploit_slots - active_exploit)
    total_need = need_explore + need_exploit
    if total_need:
        explore_quota = min(
            need_explore,
            int(round(limit * need_explore / total_need)),
        )
        exploit_quota = min(need_exploit, limit - explore_quota)
        while explore_quota + exploit_quota < min(limit, total_need):
            if explore_quota < need_explore:
                explore_quota += 1
            elif exploit_quota < need_exploit:
                exploit_quota += 1
            else:
                break
    else:
        explore_quota = exploit_quota = 0

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def take_diverse(pool: list[dict[str, Any]], quota: int) -> None:
        start = len(selected)
        for field in ("focus", "algorithm_family"):
            seen = {
                str(task.get(field, "unknown")) for task in [*active, *selected]
            }
            for task in pool:
                value = str(task.get(field, "unknown"))
                task_id = str(task.get("task_id", ""))
                if value in seen or task_id in selected_ids or len(selected) - start >= quota:
                    continue
                seen.add(value)
                selected.append(task)
                selected_ids.add(task_id)
        for task in pool:
            if len(selected) - start >= quota:
                break
            task_id = str(task.get("task_id", ""))
            if task_id not in selected_ids:
                selected.append(task)
                selected_ids.add(task_id)

    take_diverse(
        [task for task in ordered if infer_search_mode(task) == "explore"],
        explore_quota,
    )
    take_diverse(
        [task for task in ordered if infer_search_mode(task) == "exploit"],
        exploit_quota,
    )
    # Borrow only after the intended mix has been honored as far as possible.
    take_diverse(ordered, limit - len(selected))
    return selected
