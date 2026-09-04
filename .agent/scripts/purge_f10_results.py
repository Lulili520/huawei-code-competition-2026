from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = ROOT / ".agent"
REGISTRY_PATH = AGENT_ROOT / "versions.json"
EXPERIMENTS_PATH = AGENT_ROOT / "knowledge" / "experiments.json"
PRINCIPLES_PATH = AGENT_ROOT / "knowledge" / "principles.json"
QUEUE_PATH = AGENT_ROOT / "runtime" / "queue.json"
RUNS_ROOT = AGENT_ROOT / "runtime" / "runs"
VERSION_PATTERN = re.compile(r"solution[\\/](v\d+_[^\\/]+)[\\/]", re.IGNORECASE)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def results(document: dict[str, Any]) -> list[dict[str, Any]]:
    formal = document.get("formal")
    if isinstance(formal, dict) and isinstance(formal.get("results"), list):
        return [item for item in formal["results"] if isinstance(item, dict)]
    if isinstance(formal, list):
        return [item for item in formal if isinstance(item, dict)]
    raw = document.get("results")
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def is_f10(result: dict[str, Any]) -> bool:
    try:
        return (
            int(result.get("linear_case_count", -1)) == 5
            and int(result.get("attention_case_count", -1)) == 5
        )
    except (TypeError, ValueError):
        return False


def version_from_result(result: dict[str, Any]) -> str | None:
    source = str(result.get("solution_path") or result.get("path") or "")
    match = VERSION_PATTERN.search(source)
    return match.group(1) if match else None


def assert_local_file(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise RuntimeError(f"refusing to remove unexpected path: {path}")


def purge(*, dry_run: bool) -> int:
    registry = read_json(REGISTRY_PATH)
    experiments = read_json(EXPERIMENTS_PATH)
    principles = read_json(PRINCIPLES_PATH)
    queue = read_json(QUEUE_PATH)

    f10_evidence_versions: set[str] = set()
    full_versions: set[str] = set()
    registry_f10_versions: set[str] = set()
    f10_run_ids: set[str] = {
        str(item.get("run_id"))
        for item in experiments.get("experiments", [])
        if item.get("outcome") == "evaluated"
        and item.get("evaluation_profile") == "F10"
        and item.get("run_id")
    }
    files_to_remove: set[Path] = set()

    for version, node in registry.get("versions", {}).items():
        metrics = node.get("metrics") or {}
        try:
            f10 = (
                int(metrics.get("linear_case_count", -1)) == 5
                and int(metrics.get("attention_case_count", -1)) == 5
            )
        except (TypeError, ValueError):
            f10 = False
        try:
            full = (
                int(metrics.get("linear_case_count", -1)) == 50
                and int(metrics.get("attention_case_count", -1)) == 250
            )
        except (TypeError, ValueError):
            full = False
        if f10:
            registry_f10_versions.add(version)
        if full:
            full_versions.add(version)

    evaluation_paths = list(RUNS_ROOT.glob("*/evaluation*.json"))
    fixed_profile = AGENT_ROOT / "runtime" / "fixed-profile-v22.json"
    if fixed_profile.is_file():
        evaluation_paths.append(fixed_profile)
    for path in evaluation_paths:
        try:
            document = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        document_results = results(document)
        for item in document_results:
            try:
                full = (
                    int(item.get("linear_case_count", -1)) == 50
                    and int(item.get("attention_case_count", -1)) == 250
                )
            except (TypeError, ValueError):
                full = False
            if full:
                version = version_from_result(item)
                if version:
                    full_versions.add(version)
        found = [item for item in document_results if is_f10(item)]
        if not found:
            continue
        files_to_remove.add(path)
        for item in found:
            version = version_from_result(item)
            if version:
                f10_evidence_versions.add(version)
        if path.parent.parent == RUNS_ROOT:
            f10_run_ids.add(path.parent.name)
            for name in ("checkpoint.json", "report-feedback.json"):
                companion = path.parent / name
                if companion.is_file():
                    files_to_remove.add(companion)

    invalid_versions = registry_f10_versions | (
        f10_evidence_versions - full_versions
    )
    for version in invalid_versions:
        report = ROOT / "solution" / version / "report.md"
        if report.is_file():
            files_to_remove.add(report)

    print("versions with F10 evidence: " + ", ".join(sorted(f10_evidence_versions)))
    print("versions requiring F300: " + ", ".join(sorted(invalid_versions)))
    print(f"F10 run ids: {len(f10_run_ids)}")
    print(f"files to remove: {len(files_to_remove)}")
    for path in sorted(files_to_remove):
        print("  " + str(path.relative_to(ROOT)))
    if dry_run:
        return 0

    for version, node in registry.get("versions", {}).items():
        if version in invalid_versions:
            for key in (
                "metrics", "evaluation_run", "evaluation_evidence",
                "selected_config", "diagnostics", "pareto_member",
            ):
                node.pop(key, None)
            node["status"] = "draft"
            node["requires_full_evaluation"] = True
        node.pop("historical_fixed_10_evaluation", None)
    registry["revision"] = int(registry.get("revision", 0)) + 1

    retained_experiments = [
        item for item in experiments.get("experiments", [])
        if not (
            item.get("outcome") == "evaluated"
            and (
                item.get("evaluation_profile") == "F10"
                or (
                    item.get("version") in invalid_versions
                    and item.get("evaluation_profile") != "F300"
                )
            )
        )
    ]
    experiments["experiments"] = retained_experiments
    experiments["revision"] = int(experiments.get("revision", 0)) + 1

    remaining_by_run = {
        str(item.get("run_id")): item
        for item in retained_experiments if item.get("run_id")
    }
    retained_principles: list[dict[str, Any]] = []
    for principle in principles.get("principles", []):
        old_runs = [str(value) for value in principle.get("evidence_runs", [])]
        if (
            principle.get("status") != "core"
            and any(value in f10_run_ids for value in old_runs)
        ):
            # A mixed principle may still have prose or counts derived from an
            # F10 run. Drop it conservatively instead of retaining leakage.
            continue
        if old_runs:
            kept_runs = [value for value in old_runs if value not in f10_run_ids]
            principle["evidence_runs"] = kept_runs
            if principle.get("status") != "core":
                families = sorted({
                    str(remaining_by_run[run_id].get("algorithm_family", "unknown"))
                    for run_id in kept_runs if run_id in remaining_by_run
                })
                principle["supporting_families"] = families
                principle["support_count"] = len(kept_runs)
                if not kept_runs:
                    continue
                if int(principle.get("contradiction_count", 0)) == 0:
                    principle["status"] = "active" if len(families) >= 2 else "candidate"
        retained_principles.append(principle)
    principles["principles"] = retained_principles
    principles["revision"] = int(principles.get("revision", 0)) + 1

    for task in queue.get("tasks", []):
        if task.get("version") not in invalid_versions and task.get("run_id") not in f10_run_ids:
            continue
        if (ROOT / "solution" / str(task.get("version")) / "solution.py").is_file():
            task.update(
                status="queued", stage="queued",
                resume_run_id=task.get("run_id"), resume_stage="evaluation",
                error="partial evaluation cleared; full F300 evaluation required",
            )
    queue["revision"] = int(queue.get("revision", 0)) + 1

    atomic_json(REGISTRY_PATH, registry)
    atomic_json(EXPERIMENTS_PATH, experiments)
    atomic_json(PRINCIPLES_PATH, principles)
    atomic_json(QUEUE_PATH, queue)
    for path in files_to_remove:
        assert_local_file(path)
        path.unlink()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove invalid 5+5 score evidence")
    parser.add_argument("--apply", action="store_true", help="perform the purge")
    args = parser.parse_args()
    return purge(dry_run=not args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
