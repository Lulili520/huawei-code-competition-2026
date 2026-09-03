from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))
import runner  # noqa: E402


class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hif4_runner_test_")
        root = Path(self.temporary.name)
        self.solution_root = root / "solution"
        self.solution_root.mkdir()
        baseline = self.solution_root / "v0_hessian_repair"
        baseline.mkdir()
        (baseline / "solution.py").write_text("# baseline\n", encoding="utf-8")
        self.registry_path = root / "versions.json"
        self.registry_path.write_text(json.dumps({
            "schema_version": 1,
            "versioning": "flat-sequential",
            "score_weights": {"linear": 1.0, "attention": 5.0},
            "versions": {"v0_hessian_repair": {
                "number": 0, "method": "hessian_repair", "based_on": None,
                "status": "baseline",
                "metrics": {
                    "linear_mse": 0.1, "attention_mse": 0.1, "score": 1.0,
                    "linear_case_count": 50, "attention_case_count": 250,
                    "dataset": "datasets/combined",
                },
            }},
        }), encoding="utf-8")
        self.queue_path = root / "queue.json"
        self.runs = root / "runs"
        self.stop = root / "STOP"
        self.originals = {name: getattr(runner, name) for name in (
            "SOLUTION_ROOT", "REGISTRY_PATH", "QUEUE_PATH", "RUNS", "STOP_PATH",
            "EVALUATION_LOCK", "SCHEDULER_LOCK", "RUNNER_PID_PATH",
        )}
        runner.SOLUTION_ROOT = self.solution_root
        runner.REGISTRY_PATH = self.registry_path
        runner.QUEUE_PATH = self.queue_path
        runner.RUNS = self.runs
        runner.STOP_PATH = self.stop
        runner.EVALUATION_LOCK = root / "evaluation.lock"
        runner.SCHEDULER_LOCK = root / "scheduler.lock"
        runner.RUNNER_PID_PATH = root / "runner.pid"
        self.config = {
            "max_agents": 6, "directions_per_version": 3,
            "target_score": 20000.0,
            "search_mix": {"explore_slots": 4, "exploit_slots": 2},
            "max_hyperparameter_configs": 3,
            "fixed_evaluation_cases": {"linear": 50, "attention": 250},
            "screening": {
                "enabled": True, "linear_groups": 2, "attention_groups": 10,
                "linear_cases": 10, "attention_cases": 50, "promote_top_k": 2,
            },
        }

    def tearDown(self) -> None:
        for name, value in self.originals.items():
            setattr(runner, name, value)
        self.temporary.cleanup()

    def test_next_version_uses_global_sequence(self) -> None:
        registry = runner.read_json(self.registry_path)
        self.assertEqual(runner.next_version("softmax_aware", registry), "v1_softmax_aware")

    def test_version_record_has_no_tree_fields(self) -> None:
        queue = runner.initial_queue()
        runner.add_algorithm_task(
            queue, self.config, based_on="v0_hessian_repair",
            version="v1_structural_repair", focus="linear",
            family="structural_repair", hypothesis="change error model",
            base="based_on", priority=1.0,
        )
        node = runner.read_json(self.registry_path)["versions"]["v1_structural_repair"]
        self.assertEqual(node["based_on"], "v0_hessian_repair")
        self.assertNotIn("parent", node)
        self.assertNotIn("children", node)

    def test_followups_receive_global_consecutive_numbers(self) -> None:
        queue = runner.initial_queue()
        task = runner.add_algorithm_task(
            queue, self.config, based_on="v0_hessian_repair",
            version="v1_structural", focus="linear", family="structural_repair",
            hypothesis="change error model", base="based_on", priority=1.0,
        )
        proposals = {"next_algorithms": [{
            "algorithm_family": "residual_subspace",
            "version_suffix": "residual_subspace", "focus": "linear",
            "based_on": "v1_structural", "search_mode": "exploit",
            "root_cause": "The output residual remains concentrated in a calibrated low-rank subspace.",
            "hypothesis": "repair a calibrated residual subspace",
            "implementation_base": "based_on",
            "structural_change": "Add a residual low-rank subspace repair after quantization.",
            "evidence": "Second-order reconstruction supports output-aware repair.",
            "evidence_strength": 0.8,
        }]}
        registry = runner.read_json(self.registry_path)
        registry["versions"]["v1_structural"]["metrics"] = {
            "linear_mse": 0.09, "attention_mse": 0.1, "score": 1.1,
            "linear_case_count": 50, "attention_case_count": 250,
            "dataset": "datasets/combined",
        }
        runner.atomic_json(self.registry_path, registry)
        self.assertEqual(runner.enqueue_followups(queue, self.config, task, proposals, 8.5), 1)
        self.assertEqual(queue["tasks"][-1]["version"], "v2_residual_subspace")
        self.assertEqual(queue["tasks"][-1]["based_on"], "v1_structural")

    def test_parameter_only_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runner.add_algorithm_task(
                runner.initial_queue(), self.config, based_on="v0_hessian_repair",
                version="v1_alpha_tuning", focus="attention", family="alpha_tuning",
                hypothesis="only tune alpha", base="based_on", priority=1.0,
            )

    def test_paused_dry_run_dispatches_nothing(self) -> None:
        queue = runner.initial_queue()
        runner.add_algorithm_task(
            queue, self.config, based_on="v0_hessian_repair",
            version="v1_structural", focus="linear", family="structural_repair",
            hypothesis="change error model", base="based_on", priority=1.0,
        )
        runner.save_queue(queue)
        self.stop.write_text("paused\n", encoding="utf-8")
        asyncio.run(runner.Scheduler(self.config, dry_run=True, once=True).loop())
        self.assertEqual(runner.load_queue()["tasks"][0]["status"], "queued")

    def test_target_reached_dry_run_does_not_write_stop(self) -> None:
        registry = runner.read_json(self.registry_path)
        registry["versions"]["v0_hessian_repair"]["metrics"]["score"] = 20000.0
        runner.atomic_json(self.registry_path, registry)
        runner.save_queue(runner.initial_queue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            asyncio.run(runner.Scheduler(self.config, dry_run=True, once=True).loop())
        self.assertIn("target reached", output.getvalue())
        self.assertFalse(self.stop.exists())

    def test_dry_run_is_non_mutating_and_capped_at_six_slots(self) -> None:
        queue = runner.initial_queue()
        for number in range(1, 8):
            runner.add_algorithm_task(
                queue, self.config, based_on="v0_hessian_repair",
                version=f"v{number}_structural_{number}", focus="linear",
                family=f"structural_{number}", hypothesis=f"change error model {number}",
                structural_change=f"Replace the block objective with structural method {number}.",
                base="based_on", priority=0.5,
            )
        runner.save_queue(queue)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            asyncio.run(runner.Scheduler(self.config, dry_run=True, once=True).loop())
        observed = runner.load_queue()
        self.assertTrue(all(task["status"] == "queued" for task in observed["tasks"]))
        self.assertEqual(output.getvalue().count("DRY-RUN slot="), 6)

    def test_workspace_snapshot_excludes_large_data_and_runtime(self) -> None:
        source = Path(self.temporary.name) / "snapshot_source"
        (source / "datasets").mkdir(parents=True)
        (source / "reference").mkdir()
        (source / ".agent" / "runtime").mkdir(parents=True)
        (source / ".agent" / "prompts").mkdir()
        (source / "datasets" / "large.pt").write_bytes(b"large")
        (source / "reference" / "archive.zip").write_bytes(b"archive")
        (source / ".agent" / "runtime" / "queue.json").write_text("{}")
        (source / ".agent" / "prompts" / "worker.md").write_text("keep")
        run_dir = Path(self.temporary.name) / "snapshot_run"
        run_dir.mkdir()
        workspace = runner.create_workspace(run_dir, source)
        self.assertFalse((workspace / "datasets").exists())
        self.assertFalse((workspace / "reference").exists())
        self.assertFalse((workspace / ".agent" / "runtime").exists())
        self.assertTrue((workspace / ".agent" / "prompts" / "worker.md").is_file())

    def test_codex_preflight_checks_executable(self) -> None:
        passed, detail = runner.codex_preflight({"codex": {"command": sys.executable}})
        self.assertTrue(passed, detail)

    def test_codex_argv_isolates_user_config_and_sets_role_effort(self) -> None:
        config = {
            "codex": {
                "command": "codex", "search": True,
                "ignore_user_config": True, "ephemeral": True,
                "model": "gpt-test", "sandbox": "danger-full-access",
                "service_tier": "priority",
                "reasoning_effort": {"explore": "high", "exploit": "max"},
            }
        }
        argv = runner.codex_argv(
            config, Path("schema.json"), Path("result.json"), Path("workspace"),
            role="exploit",
        )
        self.assertEqual(argv[:3], ["codex", "--search", "exec"])
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--ephemeral", argv)
        self.assertIn('model_reasoning_effort="max"', argv)
        self.assertNotIn("--disable", argv)
        self.assertEqual(argv[-1], "-")

    def test_recovery_requires_manifest_or_manual_scope_audit(self) -> None:
        run_dir = Path(self.temporary.name) / "recover"
        target = run_dir / "workspace" / "solution" / "v1_method"
        target.mkdir(parents=True)
        (target / "policy.md").write_text("p" * 512, encoding="utf-8")
        (target / "solution.py").write_text("s" * 512, encoding="utf-8")
        self.assertFalse(runner.recoverable_implementation_artifacts(run_dir, "v1_method"))
        runner.atomic_json(run_dir / "legacy-scope-audit.json", {
            "version": "v1_method", "outside_target_changes": 0,
        })
        self.assertTrue(runner.recoverable_implementation_artifacts(run_dir, "v1_method"))

    def test_recovery_preserves_original_scope_baseline(self) -> None:
        run_dir = Path(self.temporary.name) / "recover_scope"
        workspace = run_dir / "workspace"
        target = workspace / "solution" / "v1_method"
        target.mkdir(parents=True)
        outside = workspace / "README.md"
        outside.write_text("original", encoding="utf-8")
        before = runner.tree_manifest(workspace)
        runner.atomic_json(run_dir / "workspace-before.json", {"files": before})
        outside.write_text("escaped", encoding="utf-8")
        (target / "solution.py").write_text("allowed", encoding="utf-8")
        baseline = runner.recovery_workspace_baseline(run_dir, workspace, "v1_method")
        with self.assertRaises(RuntimeError):
            runner.assert_workspace_scope(
                workspace, baseline, "v1_method", context="recovery test",
            )

    def test_report_recovery_requires_complete_formal_checkpoint(self) -> None:
        run_dir = Path(self.temporary.name) / "report_recovery"
        (run_dir / "workspace").mkdir(parents=True)
        runner.atomic_json(run_dir / "worker-result.json", {
            "result": {"status": "implemented"},
        })
        runner.atomic_json(run_dir / "evaluation-summary.json", {
            "formal": {"results": []},
        })
        self.assertFalse(runner.recoverable_report_checkpoint(run_dir))
        runner.atomic_json(run_dir / "checkpoint.json", {
            "stage": "formal_evaluated", "solution_sha256": "abc",
        })
        self.assertTrue(runner.recoverable_report_checkpoint(run_dir))

    def test_record_recovery_requires_reported_feedback_and_report(self) -> None:
        run_dir = Path(self.temporary.name) / "record_recovery"
        (run_dir / "workspace").mkdir(parents=True)
        runner.atomic_json(run_dir / "worker-result.json", {
            "result": {"status": "implemented"},
        })
        runner.atomic_json(run_dir / "evaluation-summary.json", {})
        runner.atomic_json(run_dir / "checkpoint.json", {
            "stage": "reported", "solution_sha256": "abc",
        })
        runner.atomic_json(run_dir / "report-feedback.json", {"status": "written"})
        baseline = self.solution_root / "v0_hessian_repair"
        (baseline / "report.md").write_text("report", encoding="utf-8")
        self.assertTrue(runner.recoverable_record_checkpoint(
            run_dir, "v0_hessian_repair",
        ))

    def test_recover_running_task_uses_implementation_checkpoint(self) -> None:
        queue = runner.initial_queue()
        task = runner.add_algorithm_task(
            queue, self.config, based_on="v0_hessian_repair",
            version="v1_recoverable", focus="linear", family="recoverable_method",
            hypothesis="test a recoverable structural implementation",
            base="based_on", priority=0.5,
        )
        task.update(status="running", stage="implementing", run_id="recover-run")
        runner.save_queue(queue)
        target = self.runs / "recover-run" / "workspace" / "solution" / task["version"]
        target.mkdir(parents=True)
        (target / "policy.md").write_text("p" * 512, encoding="utf-8")
        (target / "solution.py").write_text("s" * 512, encoding="utf-8")
        runner.atomic_json(self.runs / "recover-run" / "workspace-before.json", {
            "files": runner.tree_manifest(self.runs / "recover-run" / "workspace"),
        })
        self.assertEqual(runner.recover_queue(), 1)
        recovered = runner.load_queue()["tasks"][0]
        self.assertEqual(recovered["resume_stage"], "implementation_finalize")
        self.assertEqual(recovered["resume_run_id"], "recover-run")

    def test_stale_runner_pid_is_removed(self) -> None:
        runner.RUNNER_PID_PATH.write_text("99999999\n", encoding="utf-8")
        self.assertTrue(runner.clean_stale_runner_pid())
        self.assertFalse(runner.RUNNER_PID_PATH.exists())

    def test_file_lock_rejects_second_scheduler(self) -> None:
        lock_path = Path(self.temporary.name) / "scheduler.lock"
        with runner.file_lock(lock_path):
            with self.assertRaises(OSError):
                with runner.file_lock(lock_path, blocking=False):
                    pass

    def test_scheduler_lock_only_translates_acquisition_failure(self) -> None:
        with runner.exclusive_scheduler_lock():
            with self.assertRaises(runner.SchedulerBusyError):
                with runner.exclusive_scheduler_lock():
                    pass
        with self.assertRaises(FileNotFoundError):
            with runner.exclusive_scheduler_lock():
                raise FileNotFoundError("body failure")

    def test_evaluation_rejects_wrong_case_count(self) -> None:
        with self.assertRaises(RuntimeError):
            runner.validate_evaluation_case_counts(self.config, {
                "linear_case_count": 5,
                "attention_case_count": 5,
            })

    def test_evaluation_accepts_combined_case_count(self) -> None:
        runner.validate_evaluation_case_counts(self.config, {
            "linear_case_count": 50,
            "attention_case_count": 250,
        })

    def test_multifidelity_promotes_only_screening_top_two(self) -> None:
        queue = runner.initial_queue()
        task = runner.add_algorithm_task(
            queue, self.config, based_on="v0_hessian_repair",
            version="v1_structural", focus="attention", family="structural_search",
            hypothesis="change the joint output objective", base="based_on", priority=0.5,
        )
        version_dir = self.solution_root / task["version"]
        for name, content in (("trial_a", "# a\n"), ("trial_b", "# b\n")):
            path = version_dir / "trials" / name
            path.mkdir(parents=True)
            (path / "solution.py").write_text(content, encoding="utf-8")
        calls: list[tuple[str, list[str]]] = []

        async def fake_batch(config, candidates, run_dir, *, fidelity, **kwargs):
            calls.append((fidelity, [label for label, _ in candidates]))
            screening_scores = {"main": 1.0, "trial_a": 3.0, "trial_b": 2.0}
            formal_scores = {"trial_a": 10.0, "trial_b": 20.0}
            scores = screening_scores if fidelity == "screening" else formal_scores
            count = (10, 50) if fidelity == "screening" else (50, 250)
            return [{
                "config": label,
                "total_score": scores[label],
                "seconds": 1.0,
                "linear_output": {"mse": 0.1},
                "attention_output": {"mse": 0.1},
                "linear_case_count": count[0],
                "attention_case_count": count[1],
                "solution_path": str(path),
                "solution_sha256": runner.solution_hash(path),
            } for label, path in candidates]

        run_dir = Path(self.temporary.name) / "run"
        run_dir.mkdir()
        with mock.patch.object(runner, "run_evaluation_batch", new=fake_batch):
            selected, record = asyncio.run(runner.evaluate_candidates(
                self.config, task, run_dir, asyncio.Lock()
            ))
        self.assertEqual(calls, [
            ("screening", ["main", "trial_a", "trial_b"]),
            ("formal", ["trial_a", "trial_b"]),
        ])
        self.assertEqual(selected["config"], "trial_b")
        self.assertEqual(record["selected"], "trial_b")
        self.assertEqual((version_dir / "solution.py").read_text(encoding="utf-8"), "# b\n")

    def test_legacy_metrics_are_not_a_current_reference(self) -> None:
        self.assertFalse(runner.metrics_use_current_profile({
            "score": 999.0,
            "linear_case_count": 5,
            "attention_case_count": 5,
        }))


if __name__ == "__main__":
    unittest.main()
