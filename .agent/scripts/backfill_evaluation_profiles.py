from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_PATH = ROOT / ".agent" / "knowledge" / "experiments.json"
REGISTRY_PATH = ROOT / ".agent" / "versions.json"
RUNS_ROOT = ROOT / ".agent" / "runtime" / "runs"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def profile_name(linear_cases: int, attention_cases: int) -> str:
    if (linear_cases, attention_cases) == (50, 250):
        return "F300"
    if (linear_cases, attention_cases) == (10, 50):
        return "F60"
    if (linear_cases, attention_cases) == (5, 5):
        return "F10"
    return f"F{linear_cases + attention_cases}"


def counts_from_summary(run_id: str) -> tuple[int, int] | None:
    path = RUNS_ROOT / run_id / "evaluation-summary.json"
    if not path.is_file():
        return None
    summary = read_json(path)
    selected = summary.get("selected")
    results = summary.get("formal", {}).get("results", [])
    if not isinstance(results, list):
        return None
    chosen = next(
        (item for item in results if item.get("config") == selected),
        results[0] if results else None,
    )
    if not isinstance(chosen, dict):
        return None
    try:
        return int(chosen["linear_case_count"]), int(chosen["attention_case_count"])
    except (KeyError, TypeError, ValueError):
        return None


def evaluation_counts(
    experiment: dict[str, Any], registry: dict[str, Any],
) -> tuple[int, int] | None:
    version = str(experiment.get("version", ""))
    metrics = registry.get("versions", {}).get(version, {}).get("metrics") or {}
    try:
        return int(metrics["linear_case_count"]), int(metrics["attention_case_count"])
    except (KeyError, TypeError, ValueError):
        run_id = experiment.get("run_id")
        return counts_from_summary(str(run_id)) if run_id else None


def migrate(*, check: bool) -> int:
    document = read_json(EXPERIMENTS_PATH)
    registry = read_json(REGISTRY_PATH)
    changed = 0
    unresolved: list[str] = []
    for experiment in document.get("experiments", []):
        if experiment.get("outcome") != "evaluated":
            continue
        counts = evaluation_counts(experiment, registry)
        if counts is None:
            unresolved.append(str(experiment.get("version", "unknown")))
            continue
        linear_cases, attention_cases = counts
        expected = {
            "linear_case_count": linear_cases,
            "attention_case_count": attention_cases,
            "evaluation_profile": profile_name(linear_cases, attention_cases),
        }
        if any(experiment.get(key) != value for key, value in expected.items()):
            changed += 1
            if not check:
                experiment.update(expected)

    print(f"evaluated records requiring update: {changed}")
    if unresolved:
        print("unresolved evaluated records: " + ", ".join(sorted(unresolved)))
        return 2
    if check:
        return 1 if changed else 0
    if changed:
        document["revision"] = int(document.get("revision", 0)) + 1
        temporary = EXPERIMENTS_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, EXPERIMENTS_PATH)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill explicit F10/F60/F300 provenance in experiment memory",
    )
    parser.add_argument("--check", action="store_true", help="report drift without writing")
    args = parser.parse_args()
    return migrate(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
