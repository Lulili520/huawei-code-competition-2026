from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from core.score_summary import render_score_summary  # noqa: E402


class ScoreSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hif4_score_summary_")
        self.root = Path(self.temporary.name)
        (self.root / ".agent" / "runtime" / "runs" / "run-a").mkdir(parents=True)
        (self.root / "solution" / "v1_test_method").mkdir(parents=True)
        (self.root / "solution" / "v1_test_method" / "policy.md").write_text(
            "# policy\n", encoding="utf-8"
        )
        (self.root / ".agent" / "versions.json").write_text(json.dumps({
            "versions": {
                "v0_hessian_repair": {
                    "status": "baseline", "based_on": None,
                    "metrics": {
                        "score": 10000.0, "linear_mse": 0.1,
                        "attention_mse": 0.01, "seconds": 2.0,
                        "linear_case_count": 50, "attention_case_count": 250,
                    },
                },
                "v1_test_method": {
                    "status": "workflow_failed", "based_on": "v0_hessian_repair",
                    "metrics": None,
                },
            }
        }), encoding="utf-8")
        (self.root / ".agent" / "runtime" / "queue.json").write_text(json.dumps({
            "tasks": [{
                "version": "v1_test_method", "status": "workflow_failed",
                "stage": "workflow_failed", "failed_stage": "reporting",
                "error": "report agent schema failure",
                "structural_change": "用真实输出门禁替换张量 MSE 门禁。",
            }]
        }), encoding="utf-8")
        evaluation = {
            "selected": "main",
            "formal": {"results": [{
                "solution_path": "solution/v1_test_method/solution.py",
                "config": "main", "total_score": 21000.0, "seconds": 1.5,
                "linear_output": {"mse": 0.02},
                "attention_output": {"mse": 0.002},
                "linear_case_count": 50, "attention_case_count": 250,
                "score_scale_case_count": 300,
            }]},
        }
        (self.root / ".agent" / "runtime" / "runs" / "run-a" / "evaluation-summary.json").write_text(
            json.dumps(evaluation), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_recovers_formal_score_after_report_failure(self) -> None:
        rendered = render_score_summary(
            self.root, generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc)
        )
        self.assertIn("`v1_test_method:F300:main`", rendered)
        self.assertIn("21000.00", rendered)
        self.assertIn("评测完成；流程失败：报告/登记", rendered)
        self.assertIn("从正式运行文件恢复", rendered)
        self.assertIn("本版变化", rendered)
        self.assertIn("用真实输出门禁替换张量 MSE 门禁", rendered)

    def test_ledger_exposes_only_full_profile(self) -> None:
        rendered = render_score_summary(
            self.root, generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc)
        )
        self.assertIn("F300", rendered)
        self.assertNotIn("F10", rendered)
        self.assertIn("v0_hessian_repair", rendered)

    def test_f10_diagnostic_cannot_enter_formal_ranking(self) -> None:
        path = self.root / ".agent" / "runtime" / "runs" / "run-a" / "evaluation-summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        result = summary["formal"]["results"][0]
        result["linear_case_count"] = 5
        result["attention_case_count"] = 5
        path.write_text(json.dumps(summary), encoding="utf-8")
        rendered = render_score_summary(
            self.root, generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc)
        )
        ranking = rendered.split("## 3. ", 1)[1].split("## 4. ", 1)[0]
        self.assertNotIn("v1_test_method", ranking)
        self.assertNotIn("`v1_test_method:F10:main`", rendered)

    def test_summary_prevents_unnamed_raw_trial_duplicate(self) -> None:
        run_dir = self.root / ".agent" / "runtime" / "runs" / "run-a"
        summary = json.loads((run_dir / "evaluation-summary.json").read_text(encoding="utf-8"))
        summary["selected"] = "conservative"
        summary["formal"]["results"][0]["config"] = "conservative"
        (run_dir / "evaluation-summary.json").write_text(json.dumps(summary), encoding="utf-8")
        raw = dict(summary["formal"]["results"][0])
        raw.pop("config")
        (run_dir / "evaluation-formal.json").write_text(
            json.dumps({"results": [raw]}), encoding="utf-8"
        )
        rendered = render_score_summary(
            self.root, generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc)
        )
        self.assertIn("`v1_test_method:F300:conservative`", rendered)
        self.assertNotIn("`v1_test_method:F300:main`", rendered)


if __name__ == "__main__":
    unittest.main()
