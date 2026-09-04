from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / ".agent" / "runtime" / "runs"
REGISTRY = ROOT / ".agent" / "versions.json"


def weighted_score(result: dict) -> float:
    linear_mean = result["linear_score"] / result["linear_case_count"]
    attention_mean = result["attention_score"] / result["attention_case_count"]
    return result["case_count"] * (linear_mean + 5.0 * attention_mean) / 6.0


def replace_score(path: Path, old: float, new: float) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    updated = text.replace(repr(old), repr(new))
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def main() -> None:
    replacements: list[tuple[float, float]] = []
    for main_path in sorted(RUNS.glob("*/evaluation-main.json")):
        payload = json.loads(main_path.read_text(encoding="utf-8"))
        for result in payload["results"]:
            old = float(result["total_score"])
            new = weighted_score(result)
            result["total_score"] = new
            result["score_weights"] = {"linear": 1.0, "attention": 5.0}
            replacements.append((old, new))
            version = Path(result["path"]).parent.name
            replace_score(ROOT / "solution" / version / "report.md", old, new)
            print(f"{version}: {old:.12f} -> {new:.12f}")
        main_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary_path = main_path.with_name("evaluation-summary.json")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_results = summary.get("results", summary.get("all_configs", []))
            for result in summary_results:
                old = float(result["total_score"])
                result["total_score"] = weighted_score(result)
                result["score_weights"] = {"linear": 1.0, "attention": 5.0}
                replacements.append((old, float(result["total_score"])))
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    if REGISTRY.exists():
        registry_text = REGISTRY.read_text(encoding="utf-8")
        for old, new in replacements:
            registry_text = registry_text.replace(repr(old), repr(new))
        REGISTRY.write_text(registry_text, encoding="utf-8")

    # References to comparison-version scores can appear in later policies/reports.
    # Keep every textual score reference on the same weighting convention.
    replacements.extend([
        (5.983274, 4.88216190982013),
        (5.965549782471559, 4.876253991048614),
        (4.868103401373618, 3.0235450141326643),
        (6.009126046871927, 4.925249423296513),
        (4.539616515452819, 2.476066870931333),
        (5.938902230760779, 4.867371473811688),
        (6.00237023237718, 4.888527474350488),
        (5.715327894536726, 4.435585836071177),
    ])
    for document in (ROOT / "solution").glob("v*/*.md"):
        text = document.read_text(encoding="utf-8")
        for old, new in replacements:
            text = text.replace(repr(old), repr(new))
        document.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
