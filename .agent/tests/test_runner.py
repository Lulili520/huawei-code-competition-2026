from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))
import runner  # noqa: E402


class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hif4_runner_test_")
        root = Path(self.temporary.name)
        self.solution_root = root / "solution"; self.solution_root.mkdir()
        baseline = self.solution_root / "v0_hessian_repair"; baseline.mkdir()
        (baseline / "solution.py").write_text("# baseline\n", encoding="utf-8")
        self.tree_path = root / "version-tree.json"
        self.tree_path.write_text(json.dumps({
            "schema_version": 1, "root": "v0_hessian_repair",
            "nodes": {"v0_hessian_repair": {
                "parent": None, "children": [], "focus": "combined",
                "status": "baseline",
                "metrics": {"linear_mse": 0.1, "attention_mse": 0.1, "score": 1.0},
            }},
        }), encoding="utf-8")
        self.queue_path = root / "queue.json"; self.runs = root / "runs"; self.stop = root / "STOP"
        self.originals = {name: getattr(runner, name) for name in (
            "SOLUTION_ROOT", "TREE_PATH", "QUEUE_PATH", "RUNS", "STOP_PATH",
        )}
        runner.SOLUTION_ROOT = self.solution_root; runner.TREE_PATH = self.tree_path
        runner.QUEUE_PATH = self.queue_path; runner.RUNS = self.runs; runner.STOP_PATH = self.stop
        self.config = {"max_agents": 6, "max_children": 3, "max_hyperparameter_configs": 3}

    def tearDown(self) -> None:
        for name, value in self.originals.items(): setattr(runner, name, value)
        self.temporary.cleanup()

    def test_seed_creates_only_three_complete_algorithms(self) -> None:
        queue = runner.initial_queue(); runner.seed(queue, self.config); runner.save_queue(queue)
        self.assertEqual(len(queue["tasks"]), 3)
        self.assertTrue(all(task["kind"] == "algorithm" for task in queue["tasks"]))
        tree = runner.read_json(self.tree_path)
        self.assertEqual(len(tree["nodes"]["v0_hessian_repair"]["children"]), 3)

    def test_dry_run_uses_three_slots_and_leaves_three_idle(self) -> None:
        queue = runner.initial_queue(); runner.seed(queue, self.config); runner.save_queue(queue)
        asyncio.run(runner.Scheduler(self.config, dry_run=True, once=True).loop())
        self.assertEqual([task["status"] for task in runner.load_queue()["tasks"]], ["dry_run"] * 3)

    def test_followups_are_children_and_inherit_measured_priority(self) -> None:
        queue = runner.initial_queue()
        task = runner.add_algorithm_task(
            queue, self.config, parent="v0_hessian_repair", version="v1_structural",
            focus="linear", family="structural_repair", hypothesis="change error model",
            base="parent", priority=1.0,
        )
        proposals = {"next_algorithms": [{
            "algorithm_family": "residual_subspace", "version_suffix": "residual_subspace",
            "focus": "linear", "hypothesis": "repair a calibrated residual subspace",
            "implementation_base": "parent",
            "structural_change": "Add a residual low-rank subspace repair after block quantization.",
            "evidence": "Second-order error compensation supports output-aware residual repair.",
        }]}
        self.assertEqual(runner.enqueue_followups(queue, self.config, task, proposals, 8.5), 1)
        child = queue["tasks"][-1]
        self.assertEqual(child["parent"], "v1_structural")
        self.assertEqual(child["priority"], 8.5)

    def test_parameter_only_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runner.add_algorithm_task(
                runner.initial_queue(), self.config, parent="v0_hessian_repair",
                version="v1_alpha_tuning", focus="attention", family="alpha_tuning",
                hypothesis="only tune alpha", base="parent", priority=1.0,
            )

    def test_recover_requeues_interrupted_task(self) -> None:
        queue = runner.initial_queue()
        task = runner.add_algorithm_task(
            queue, self.config, parent="v0_hessian_repair", version="v1_interrupted",
            focus="linear", family="interrupted_algorithm", hypothesis="structural test",
            base="parent", priority=1.0,
        )
        task["status"] = "running"
        runner.save_queue(queue)
        self.assertEqual(runner.recover_queue(), 1)
        recovered = runner.load_queue()["tasks"][0]
        self.assertEqual(recovered["status"], "queued")
        self.assertIsNone(recovered["run_id"])

    def test_recover_requeues_environment_failure_without_algorithm_failure(self) -> None:
        queue = runner.initial_queue()
        task = runner.add_algorithm_task(
            queue, self.config, parent="v0_hessian_repair", version="v1_environment_retry",
            focus="linear", family="environment_retry", hypothesis="structural test",
            base="parent", priority=1.0,
        )
        task.update(status="environment_failed", error="executable unavailable")
        runner.save_queue(queue)
        self.assertEqual(runner.recover_queue(), 1)
        self.assertEqual(runner.load_queue()["tasks"][0]["status"], "queued")
        node = runner.read_json(self.tree_path)["nodes"]["v1_environment_retry"]
        self.assertEqual(node["status"], "draft")
        self.assertIsNone(node["metrics"])

    def test_recover_requeues_winerror2_evaluation_launch_failure(self) -> None:
        queue = runner.initial_queue()
        task = runner.add_algorithm_task(
            queue, self.config, parent="v0_hessian_repair", version="v1_eval_launch_retry",
            focus="linear", family="eval_launch_retry", hypothesis="structural test",
            base="parent", priority=1.0,
        )
        task.update(status="failed", error="[WinError 2] 系统找不到指定的文件。")
        tree = runner.read_json(self.tree_path)
        tree["nodes"][task["version"]].update(
            status="failed", failure=task["error"]
        )
        runner.atomic_json(self.tree_path, tree)
        runner.save_queue(queue)
        self.assertEqual(runner.recover_queue(), 1)
        self.assertEqual(runner.load_queue()["tasks"][0]["status"], "queued")
        node = runner.read_json(self.tree_path)["nodes"][task["version"]]
        self.assertEqual(node["status"], "draft")
        self.assertNotIn("failure", node)

    def test_codex_preflight_checks_executable(self) -> None:
        passed, detail = runner.codex_preflight({"codex": {"command": sys.executable}})
        self.assertTrue(passed, detail)

    def test_windows_sandbox_logon_error_is_environment_failure(self) -> None:
        result = {
            "status": "failed",
            "algorithm_summary": "CreateProcessWithLogonW failed: error 1385",
        }
        self.assertTrue(runner.result_is_environment_failure(result))
        self.assertFalse(runner.result_is_environment_failure({
            "status": "failed", "algorithm_summary": "algorithm interface mismatch",
        }))

    def test_file_lock_rejects_second_scheduler(self) -> None:
        lock_path = Path(self.temporary.name) / "scheduler.lock"
        with runner.file_lock(lock_path):
            with self.assertRaises(OSError):
                with runner.file_lock(lock_path, blocking=False):
                    pass


if __name__ == "__main__":
    unittest.main()
