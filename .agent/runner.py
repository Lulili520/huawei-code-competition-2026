from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.integrity import (
    assert_changes_within,
    assert_dataset_signatures,
    assert_protected_unchanged,
    dataset_file_signatures,
    protected_manifest,
    tree_manifest,
    verify_dataset_manifest,
)
from core.strategy import (
    adaptive_priority,
    build_pareto_archive,
    build_process_metrics,
    compact_research_context,
    infer_search_mode,
    metrics_are_current,
    proposal_fingerprint,
    select_diverse_batch,
    stagnation_length,
)


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / ".agent"
CONFIG_PATH = AGENT_ROOT / "config.json"
REGISTRY_PATH = AGENT_ROOT / "versions.json"
RUNTIME = AGENT_ROOT / "runtime"
QUEUE_PATH = RUNTIME / "queue.json"
RUNS = RUNTIME / "runs"
STOP_PATH = AGENT_ROOT / "STOP"
SCHEDULER_LOCK = RUNTIME / "scheduler.lock"
EVALUATION_LOCK = RUNTIME / "evaluation.lock"
RUNNER_PID_PATH = RUNTIME / "runner.pid"
SOLUTION_ROOT = ROOT / "solution"
PROMPTS = AGENT_ROOT / "prompts"
SCHEMAS = AGENT_ROOT / "schemas"
KNOWLEDGE = AGENT_ROOT / "knowledge"
PRINCIPLES_PATH = KNOWLEDGE / "principles.json"
EXPERIMENTS_PATH = KNOWLEDGE / "experiments.json"
PARETO_PATH = KNOWLEDGE / "pareto.json"
PROCESS_METRICS_PATH = KNOWLEDGE / "process_metrics.json"
VERSION_RE = re.compile(r"^v([1-9][0-9]*)_([a-z0-9]+(?:_[a-z0-9]+)*)$")
PARAM_ONLY = re.compile(
    r"(?:^|_)(alpha|gain|threshold|coefficient|factor)(?:_|$)|阈值|系数|倍率|参数",
    re.IGNORECASE,
)
class EnvironmentLaunchError(RuntimeError):
    """Infrastructure failed before the quantization algorithm could run."""


class EvaluationTimeoutError(RuntimeError):
    """The algorithm exceeded the configured evaluation budget."""


class SchedulerBusyError(RuntimeError):
    """Another scheduler owns the shared-state lock."""


class AgentExecutionTimeout(RuntimeError):
    """A Codex workflow step timed out; this is not an algorithm score."""


class AgentProtocolError(RuntimeError):
    """Codex exited without the structured response required by the workflow."""


def utf8_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def result_is_environment_failure(result: dict[str, Any] | None) -> bool:
    if not result:
        return False
    evidence = json.dumps(result, ensure_ascii=False).lower()
    markers = (
        "createprocesswithlogonw", "error 1385", "winerror 2",
        "executable unavailable", "系统找不到指定的文件",
    )
    return any(marker in evidence for marker in markers)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def clean_stale_runner_pid() -> bool:
    if not RUNNER_PID_PATH.is_file():
        return False
    try:
        pid = int(RUNNER_PID_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pid = -1
    if process_exists(pid):
        return False
    RUNNER_PID_PATH.unlink(missing_ok=True)
    return True


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported .agent/config.json schema")
    if not 1 <= int(config["max_agents"]) <= 6:
        raise ValueError("max_agents must be between 1 and 6")
    if float(config.get("target_score", 0.0)) <= 0.0:
        raise ValueError("target_score must be positive")
    if not 60 <= int(config.get("agent_idle_timeout_seconds", 0)) <= int(
        config["worker_timeout_seconds"]
    ):
        raise ValueError("agent_idle_timeout_seconds must be within the worker timeout")
    search_mix = config.get("search_mix", {})
    explore_slots = int(search_mix.get("explore_slots", -1))
    exploit_slots = int(search_mix.get("exploit_slots", -1))
    if explore_slots < 0 or exploit_slots < 0:
        raise ValueError("search_mix slot counts must be non-negative")
    if explore_slots + exploit_slots != int(config["max_agents"]):
        raise ValueError("search_mix slots must sum to max_agents")
    if not 1 <= int(config["max_hyperparameter_configs"]) <= 3:
        raise ValueError("max_hyperparameter_configs must be between 1 and 3")
    if int(config.get("directions_per_version", 0)) != 3:
        raise ValueError("directions_per_version must remain exactly 3")
    if not 1 <= int(config.get("environment_launch_retries", 1)) <= 3:
        raise ValueError("environment_launch_retries must be between 1 and 3")
    screening = config.get("screening", {})
    if screening.get("enabled", False):
        if int(screening.get("linear_groups", 0)) < 1:
            raise ValueError("screening.linear_groups must be positive")
        if int(screening.get("attention_groups", 0)) < 1:
            raise ValueError("screening.attention_groups must be positive")
        if not 1 <= int(screening.get("promote_top_k", 1)) <= int(config["max_hyperparameter_configs"]):
            raise ValueError("screening.promote_top_k is outside the configuration limit")
        if int(screening.get("linear_cases", -1)) != 5 * int(screening["linear_groups"]):
            raise ValueError("screening Linear case count must match five tests per group")
        if int(screening.get("attention_cases", -1)) != 5 * int(screening["attention_groups"]):
            raise ValueError("screening Attention case count must match five tests per group")
    if not config.get("protected_files"):
        raise ValueError("protected_files must define the evaluation trust boundary")
    return config


def codex_preflight(config: dict[str, Any]) -> tuple[bool, str]:
    command = str(config["codex"]["command"])
    resolved = shutil.which(command)
    if not resolved:
        return False, f"Codex executable not found: {command}"
    try:
        completed = subprocess.run(
            [resolved, "--version"], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20,
            env=utf8_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"Codex preflight failed: {error}"
    version = (completed.stdout or completed.stderr).strip()
    return completed.returncode == 0, version or f"exit code {completed.returncode}"


def evaluation_python_preflight(config: dict[str, Any]) -> tuple[bool, str]:
    """Verify the exact environment Python used by official evaluation."""
    command = str(config.get("evaluation_python", sys.executable))
    resolved = shutil.which(command)
    if not resolved:
        return False, f"Evaluation Python not found: {command}"
    try:
        completed = subprocess.run(
            [resolved, "--version"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30, env=utf8_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"Evaluation Python preflight failed: {error}"
    detail = (completed.stdout or completed.stderr).strip()
    return completed.returncode == 0, detail or f"exit code {completed.returncode}"


def initial_queue() -> dict[str, Any]:
    return {"schema_version": 1, "revision": 0, "tasks": []}


def load_queue() -> dict[str, Any]:
    if not QUEUE_PATH.exists():
        atomic_json(QUEUE_PATH, initial_queue())
    queue = read_json(QUEUE_PATH)
    if queue.get("schema_version") != 1 or not isinstance(queue.get("tasks"), list):
        raise ValueError("invalid runtime queue")
    return queue


def save_queue(queue: dict[str, Any]) -> None:
    queue["revision"] = int(queue.get("revision", 0)) + 1
    atomic_json(QUEUE_PATH, queue)


def load_principles() -> dict[str, Any]:
    value = read_json(PRINCIPLES_PATH)
    if value.get("schema_version") != 1 or not isinstance(value.get("principles"), list):
        raise ValueError("invalid research principles")
    return value


def load_experiments() -> dict[str, Any]:
    value = read_json(EXPERIMENTS_PATH)
    if value.get("schema_version") != 1 or not isinstance(value.get("experiments"), list):
        raise ValueError("invalid experiment memory")
    return value


def refresh_pareto_archive() -> dict[str, Any]:
    archive = build_pareto_archive(read_json(REGISTRY_PATH))
    archive["updated_at"] = now()
    atomic_json(PARETO_PATH, archive)
    return archive


def refresh_process_metrics() -> dict[str, Any]:
    metrics = build_process_metrics(load_experiments()["experiments"])
    metrics["updated_at"] = now()
    atomic_json(PROCESS_METRICS_PATH, metrics)
    return metrics


def research_context(task: dict[str, Any]) -> dict[str, Any]:
    principles = load_principles()["principles"]
    experiments = load_experiments()["experiments"]
    pareto = read_json(PARETO_PATH) if PARETO_PATH.is_file() else build_pareto_archive(
        read_json(REGISTRY_PATH)
    )
    return compact_research_context(task, principles, experiments, pareto)


def apply_principle_updates(
    updates: list[dict[str, Any]], *, run_id: str, algorithm_family: str,
) -> None:
    """Evidence-gated guidance evolution; core safety rules can never be retired."""
    if not updates:
        return
    document = load_principles()
    principles = document["principles"]
    by_id = {item["id"]: item for item in principles}
    changed = False
    for update in updates[:3]:
        principle_id = str(update.get("principle_id", "")).strip()
        action = update.get("action")
        evidence = str(update.get("evidence", "")).strip()
        if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", principle_id) or len(evidence) < 20:
            continue
        item = by_id.get(principle_id)
        if action == "add":
            statement = str(update.get("statement", "")).strip()
            if len(statement) < 20:
                continue
            if item is None:
                item = {
                    "id": principle_id,
                    "statement": statement,
                    "rationale": evidence,
                    "status": "candidate",
                    "support_count": 0,
                    "contradiction_count": 0,
                    "sources": [],
                    "supporting_families": [],
                    "evidence_runs": [],
                }
                principles.append(item)
                by_id[principle_id] = item
            action = "reinforce"
        if item is None:
            continue
        evidence_runs = item.setdefault("evidence_runs", [])
        families = item.setdefault("supporting_families", [])
        if run_id in evidence_runs:
            continue
        evidence_runs.append(run_id)
        if action == "reinforce":
            item["support_count"] = int(item.get("support_count", 0)) + 1
            if algorithm_family not in families:
                families.append(algorithm_family)
            if item.get("status") == "candidate" and len(families) >= 2:
                item["status"] = "active"
        elif action == "challenge":
            item["contradiction_count"] = int(item.get("contradiction_count", 0)) + 1
            if item.get("status") == "active" and item["contradiction_count"] >= 2:
                item["status"] = "challenged"
        else:
            continue
        changed = True
    if changed:
        document["revision"] = int(document.get("revision", 0)) + 1
        atomic_json(PRINCIPLES_PATH, document)


def record_experiment(
    task: dict[str, Any], selected: dict[str, Any], run_id: str,
    best_before: tuple[str, float], report_feedback: dict[str, Any],
) -> None:
    registry = read_json(REGISTRY_PATH)
    baseline = registry["versions"].get(task.get("based_on"), {})
    baseline_metrics = baseline.get("metrics") or {}
    score = float(selected["total_score"])
    experiment = {
        "version": task["version"],
        "algorithm_family": task["algorithm_family"],
        "focus": task["focus"],
        "search_mode": task.get("search_mode", infer_search_mode(task)),
        "root_cause": task.get("root_cause", task.get("hypothesis")),
        "baseline": task.get("based_on"),
        "baseline_score": baseline_metrics.get("score"),
        "score": score,
        "score_delta": score - float(baseline_metrics.get("score", score)),
        "linear_mse": selected["linear_output"]["mse"],
        "attention_mse": selected["attention_output"]["mse"],
        "linear_mse_delta": (
            float(selected["linear_output"]["mse"])
            - float(baseline_metrics.get("linear_mse", selected["linear_output"]["mse"]))
        ),
        "attention_mse_delta": (
            float(selected["attention_output"]["mse"])
            - float(baseline_metrics.get("attention_mse", selected["attention_output"]["mse"]))
        ),
        "seconds": selected.get("seconds"),
        "outcome": "evaluated",
        "new_global_best": score > best_before[1],
        "best_before": best_before[0],
        "run_id": run_id,
        "completed_at": now(),
        "hypothesis_outcomes": report_feedback.get("hypothesis_outcomes", []),
        "takeaways": report_feedback.get("takeaways", []),
        "screening": report_feedback.get("screening_summary", {}),
    }
    document = load_experiments()
    document["experiments"] = [
        item for item in document["experiments"] if item.get("version") != task["version"]
    ]
    document["experiments"].append(experiment)
    document["revision"] = int(document.get("revision", 0)) + 1
    atomic_json(EXPERIMENTS_PATH, document)
    apply_principle_updates(
        report_feedback.get("principle_updates", []), run_id=run_id,
        algorithm_family=task["algorithm_family"],
    )
    refresh_pareto_archive()
    refresh_process_metrics()


def record_failed_experiment(
    task: dict[str, Any], run_id: str, error: str, *, outcome: str = "failed",
) -> None:
    document = load_experiments()
    document["experiments"] = [
        item for item in document["experiments"] if item.get("version") != task.get("version")
    ]
    document["experiments"].append({
        "version": task.get("version"),
        "algorithm_family": task.get("algorithm_family"),
        "focus": task.get("focus"),
        "search_mode": task.get("search_mode", infer_search_mode(task)),
        "root_cause": task.get("root_cause", task.get("hypothesis")),
        "baseline": task.get("based_on"),
        "outcome": outcome,
        "failure": error[:2000],
        "run_id": run_id,
        "completed_at": now(),
    })
    document["revision"] = int(document.get("revision", 0)) + 1
    atomic_json(EXPERIMENTS_PATH, document)
    refresh_process_metrics()


def solution_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluation_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Describe only inputs that can change formal numerical semantics."""
    dataset_dir = str(config["evaluation_datasets_dir"]).replace("\\", "/")
    critical_files = [
        ".agent/skills/hif4-evaluate/scripts/evaluate.py",
        ".agent/skills/hif4-evaluate/scripts/scoring.py",
        f"{dataset_dir}/manifest.json",
    ]
    return {
        "evaluation_python": str(config.get("evaluation_python", sys.executable)),
        "evaluation_datasets_dir": dataset_dir,
        "fixed_evaluation_cases": config["fixed_evaluation_cases"],
        "screening": config.get("screening", {}),
        "critical_file_sha256": protected_manifest(ROOT, critical_files),
    }


def write_trust_manifest(
    run_dir: Path, config: dict[str, Any], root_trust: dict[str, str],
) -> dict[str, Any]:
    contract = evaluation_contract(config)
    dataset_manifest = ROOT / config["evaluation_datasets_dir"] / "manifest.json"
    signatures = {
        name: {"bytes": values[0], "mtime_ns": values[1]}
        for name, values in dataset_file_signatures(dataset_manifest).items()
    }
    payload = {
        "schema_version": 1,
        "recorded_at": now(),
        "protected_file_sha256": root_trust,
        "evaluation_contract": contract,
        "evaluation_contract_sha256": canonical_digest(contract),
        "dataset_file_signatures": signatures,
    }
    atomic_json(run_dir / "trust-manifest.json", payload)
    return payload


def validate_formal_checkpoint(
    run_dir: Path, config: dict[str, Any], *, stages: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify evaluation provenance, exact selected config and full case counts."""
    checkpoint = read_json(run_dir / "checkpoint.json")
    if checkpoint.get("stage") not in stages:
        raise RuntimeError("formal checkpoint is at an unexpected stage")
    evaluation_path = run_dir / "evaluation-summary.json"
    evaluation = read_json(evaluation_path)
    selected = selected_evaluation_result(evaluation)
    validate_evaluation_case_counts(config, selected)
    if selected.get("config") != checkpoint.get("selected"):
        raise RuntimeError("formal checkpoint selected-config mismatch")
    if selected.get("solution_sha256") != checkpoint.get("solution_sha256"):
        raise RuntimeError("formal checkpoint candidate hash mismatch")

    trust_path = run_dir / "trust-manifest.json"
    if trust_path.is_file():
        trust = read_json(trust_path)
        recorded_contract = trust.get("evaluation_contract")
        recorded_digest = trust.get("evaluation_contract_sha256")
        if canonical_digest(recorded_contract) != recorded_digest:
            raise RuntimeError("trust manifest contract digest is invalid")
        if checkpoint.get("evaluation_contract_sha256") != recorded_digest:
            raise RuntimeError("checkpoint and trust manifest contract mismatch")
        if canonical_digest(evaluation) != checkpoint.get("evaluation_summary_sha256"):
            raise RuntimeError("evaluation summary changed after checkpoint")
        if canonical_digest(evaluation_contract(config)) != recorded_digest:
            raise RuntimeError("formal evaluation contract changed before recovery")
    else:
        legacy = run_dir / "legacy-evaluation-audit.json"
        if not legacy.is_file():
            raise RuntimeError("formal checkpoint has no trust provenance")
        audit = read_json(legacy)
        if not (
            audit.get("approved") is True
            and audit.get("dataset") == config["evaluation_datasets_dir"]
            and int(audit.get("linear_cases", -1))
            == int(config["fixed_evaluation_cases"]["linear"])
            and int(audit.get("attention_cases", -1))
            == int(config["fixed_evaluation_cases"]["attention"])
            and audit.get("selected") == checkpoint.get("selected")
            and audit.get("solution_sha256") == checkpoint.get("solution_sha256")
        ):
            raise RuntimeError("legacy evaluation checkpoint audit is invalid")
    return evaluation, selected


@contextmanager
def file_lock(path: Path, *, blocking: bool = True):
    """Cross-process single-byte lock for Windows and POSIX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b"0"); stream.flush()
    try:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield
    finally:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


@contextmanager
def exclusive_scheduler_lock():
    """Translate only lock-acquisition failures, not errors from the body."""
    manager = file_lock(SCHEDULER_LOCK, blocking=False)
    try:
        manager.__enter__()
    except OSError as error:
        raise SchedulerBusyError("another scheduler is running") from error
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


def policy_scaffold(task: dict[str, Any]) -> str:
    return f"""# {task['version']} 优化策略

## 基准版本

对比基准版本：`{task['based_on']}`。先记录基准版本统一 300 例的 Linear MSE、Attention MSE 和最终得分。该字段只表示比较或实现来源，不形成父子关系。

## 实现基础

- 实现方式：`{task['implementation_base']}`
- 搜索角色：`{task.get('search_mode', 'explore')}`（explore 寻找新机制；exploit 深化已有正式正证据）
- 实际代码来源、复用模块和重写模块：待说明

## 固定输入边界

NVFP4 反量化固定为 E2M1 值乘对应 E4M3 scale，每 16 个连续值共享一个 scale，再恢复原 shape。不得修改该规则。

## 问题分析

根因假设：{task.get('root_cause', task['hypothesis'])}

先把最终输出误差分解到可观测项，再写可定位的已验证事实：文件路径与函数/行号，或评测配置与精确指标。Linear 至少区分 `ΔXWᵀ`、`XΔWᵀ` 和 `ΔXΔWᵀ`；Attention 至少区分居中 logit 误差、Softmax Jacobian 放大和 V 路径误差；格式方向至少区分削顶与小值分辨率不足。结合 Runner 注入的正例、反例与 Pareto 起点，说明哪一项是待验证的主导根因。

## 相关方案调研

优先引用原始论文、标准或官方文档，记录标题、作者/机构、年份和直达链接，并说明其假设是否适用于本任务。

## 理论分析

区分已验证事实、理论推导和待验证假设。必要公式必须解释符号、来源、适用条件及其与目标 MSE 的关系。

| 问题证据 | 理论依据 | 算法动作 | 目标指标 | 否证条件 |
|---|---|---|---|---|
| 待填写可复查证据 | 待填写来源与适用假设 | 待填写具体流程变化 | 待填写指标与方向 | 待填写拒绝假设的结果 |

## 选定修改方案

### 核心算法

- 算法族：`{task['algorithm_family']}`
- 研究假设：{task['hypothesis']}

### 修改目标

待说明目标指标、比较基线和预期方向。

### 修改范围

待说明具体函数、数据流、结构性变化、校准期/动态期复杂度和运行时间上界。

### 保持不变

固定 NVFP4 输入解析、公开接口和未涉及路径保持不变。

## 算法内部超参数计划

最多测试三组有理论依据的配置。说明参数含义、候选值和选择规则；配置先经 10 Linear + 50 Attention 筛选，最多两个进入完整 300 例；不得把纯调参拆成新版本。

## 实施步骤

写成可执行顺序，包含边界检查、回退条件和可观测诊断。含搜索或候选选择时规划只读 `hif4_get_diagnostics()`。

## 预期结果

只写可证伪的方向性假设，不得伪造数值。

## 验收标准

官方格式检查通过；比较两项 MSE 和最终得分；逐项判定假设为结果支持、结果否证或证据不足。
"""


def metrics_use_current_profile(metrics: dict[str, Any] | None) -> bool:
    return metrics_are_current(metrics)


def best_reference(registry: dict[str, Any]) -> str:
    candidates: list[tuple[float, str]] = []
    for name, node in registry["versions"].items():
        metrics = node.get("metrics")
        if (
            metrics_use_current_profile(metrics)
            and node.get("status") not in {
                "failed", "draft", "evaluation_timeout", "invalid_after_evaluation"
            }
        ):
            candidates.append((float(metrics["score"]), name))
    if not candidates:
        raise RuntimeError("no evaluated version is available as a reference")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def best_score_state(registry: dict[str, Any]) -> tuple[str | None, float]:
    candidates = [
        (name, float(node["metrics"]["score"]))
        for name, node in registry.get("versions", {}).items()
        if metrics_use_current_profile(node.get("metrics"))
        and node.get("status") not in {
            "failed", "draft", "environment_failed", "evaluation_timeout",
            "invalid_after_evaluation",
        }
    ]
    if not candidates:
        return None, float("-inf")
    return max(candidates, key=lambda item: (item[1], item[0]))


def next_version(suffix: str, registry: dict[str, Any]) -> str:
    numbers = [int(node["number"]) for node in registry["versions"].values()]
    number = max(numbers, default=-1) + 1
    candidate = f"v{number}_{suffix}"
    if candidate in registry["versions"] or (SOLUTION_ROOT / candidate).exists():
        raise FileExistsError(candidate)
    return candidate


def reserve_algorithm(task: dict[str, Any], config: dict[str, Any]) -> None:
    registry = read_json(REGISTRY_PATH)
    based_on = task["based_on"]
    if based_on not in registry["versions"]:
        raise ValueError(f"unknown based_on version: {based_on}")
    version = task["version"]
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid algorithm version name: {version}")
    if version in registry["versions"] or (SOLUTION_ROOT / version).exists():
        raise FileExistsError(version)
    target = SOLUTION_ROOT / version
    target.mkdir(parents=False)
    base = task["implementation_base"]
    if base == "based_on":
        shutil.copy2(SOLUTION_ROOT / based_on / "solution.py", target / "solution.py")
    elif base == "v0_hessian_repair":
        shutil.copy2(
            SOLUTION_ROOT / "v0_hessian_repair" / "solution.py", target / "solution.py"
        )
    elif base == "scratch":
        (target / "solution.py").write_text(
            '"""From-scratch HiF4 algorithm; implemented by the assigned Agent."""\n',
            encoding="utf-8",
        )
    else:
        raise ValueError(f"unsupported implementation base: {base}")
    (target / "policy.md").write_text(policy_scaffold(task), encoding="utf-8")
    number = int(re.match(r"^v(\d+)_", version).group(1))
    registry["versions"][version] = {
        "number": number,
        "method": version.split("_", 1)[1],
        "based_on": based_on,
        "focus": task["focus"],
        "algorithm_family": task["algorithm_family"],
        "implementation_base": base,
        "search_mode": task.get("search_mode", "explore"),
        "hypothesis": task["hypothesis"],
        "root_cause": task.get("root_cause", task["hypothesis"]),
        "structural_change": task.get("structural_change", ""),
        "target_metric": task.get("target_metric", "score"),
        "falsification": task.get("falsification", ""),
        "proposal_fingerprint": task.get("proposal_fingerprint"),
        "status": "draft",
        "metrics": None,
        "task_id": task["task_id"],
    }
    atomic_json(REGISTRY_PATH, registry)


def add_algorithm_task(
    queue: dict[str, Any], config: dict[str, Any], *, based_on: str, version: str,
    focus: str, family: str, hypothesis: str, base: str, priority: float,
    structural_change: str = "", evidence: str = "",
    evidence_strength: float = 0.5, novelty: float = 0.5,
    uncertainty: float = 0.5, expected_cost: float = 1.0,
    target_metric: str = "score", falsification: str = "",
    search_mode: str | None = None, root_cause: str = "",
) -> dict[str, Any]:
    if PARAM_ONLY.search(family) or PARAM_ONLY.search(version):
        raise ValueError("pure parameter variants are not valid algorithm versions")
    fingerprint = proposal_fingerprint(family, hypothesis, structural_change)
    if any(
        task.get("proposal_fingerprint") == fingerprint
        and task.get("status") not in {"failed", "cancelled"}
        for task in queue["tasks"]
    ):
        raise ValueError(f"duplicate algorithm proposal: {family}")
    priority_hint = float(priority) if 0.0 <= float(priority) <= 1.0 else 0.5
    if search_mode is not None and search_mode not in {"explore", "exploit"}:
        raise ValueError("search_mode must be explore or exploit")
    task = {
        "task_id": uuid.uuid4().hex[:12],
        "kind": "algorithm",
        "version": version,
        "based_on": based_on,
        "focus": focus,
        "algorithm_family": family,
        "hypothesis": hypothesis,
        "root_cause": root_cause.strip() or hypothesis,
        "structural_change": structural_change,
        "evidence": evidence,
        "evidence_strength": min(1.0, max(0.0, float(evidence_strength))),
        "novelty": min(1.0, max(0.0, float(novelty))),
        "uncertainty": min(1.0, max(0.0, float(uncertainty))),
        "expected_cost": max(0.25, float(expected_cost)),
        "target_metric": target_metric,
        "falsification": falsification,
        "proposal_fingerprint": fingerprint,
        "implementation_base": base,
        "search_mode": search_mode,
        "priority": 0.0,
        "priority_hint": priority_hint,
        "priority_components": {},
        "status": "queued",
        "stage": "queued",
        "created_at": now(),
        "updated_at": now(),
        "run_id": None,
        "error": None,
    }
    task["search_mode"] = infer_search_mode(task)
    experiments = load_experiments()["experiments"] if EXPERIMENTS_PATH.is_file() else []
    priority_value, components = adaptive_priority(
        task, read_json(REGISTRY_PATH), experiments
    )
    task["priority"] = priority_value
    task["priority_components"] = components
    reserve_algorithm(task, config)
    queue["tasks"].append(task)
    return task


def codex_argv(
    config: dict[str, Any], schema: Path, output: Path, workspace: Path,
    sandbox: str | None = None, role: str = "explore",
) -> list[str]:
    codex = config["codex"]
    argv = [codex["command"]]
    if codex.get("search", False):
        argv.append("--search")
    argv.append("exec")
    if codex.get("ignore_user_config", False):
        argv.append("--ignore-user-config")
    if codex.get("ephemeral", False):
        argv.append("--ephemeral")
    if codex.get("model"):
        argv.extend(["--model", str(codex["model"])])
    effort = codex.get("reasoning_effort", {}).get(role)
    if effort:
        argv.extend(["-c", f'model_reasoning_effort="{effort}"'])
    if codex.get("service_tier"):
        argv.extend(["-c", f'service_tier="{codex["service_tier"]}"'])
    argv.extend([
        "--json", "--color", "never",
        "--sandbox", sandbox or codex["sandbox"], "--cd", str(workspace),
        "--output-schema", str(schema), "--output-last-message", str(output), "-",
    ])
    return argv


def extract_session(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("session_id", "thread_id", "threadId"):
            item = value.get(key)
            if isinstance(item, str) and item:
                return item
        for item in value.values():
            found = extract_session(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = extract_session(item)
            if found:
                return found
    return None


async def run_codex(
    config: dict[str, Any], run_dir: Path, prompt: str, schema: Path,
    timeout: int, workspace: Path | None = None, sandbox: str | None = None,
    role: str = "explore",
) -> tuple[int, str | None, dict[str, Any] | None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    stderr_path = run_dir / "stderr.log"
    last_path = run_dir / "last-message.json"
    # A resumed stage can reuse its directory.  Never accept a structured
    # response left by an earlier, timed-out Codex process.
    last_path.unlink(missing_ok=True)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = None
    launch_error: OSError | None = None
    for attempt in range(1, int(config.get("environment_launch_retries", 1)) + 1):
        try:
            process = await asyncio.create_subprocess_exec(
                *codex_argv(
                    config, schema, last_path, workspace or ROOT, sandbox, role,
                ),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=creationflags,
                env=utf8_environment(),
            )
            break
        except OSError as error:
            launch_error = error
            with (run_dir / "launch-errors.log").open("a", encoding="utf-8") as stream:
                stream.write(f"attempt={attempt} error={error}\n")
            if attempt < int(config.get("environment_launch_retries", 1)):
                await asyncio.sleep(1)
    if process is None:
        raise EnvironmentLaunchError(
            f"Codex could not start after {config.get('environment_launch_retries', 1)} attempts: {launch_error}"
        )
    assert process.stdin and process.stdout and process.stderr
    try:
        process.stdin.write(prompt.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
    except asyncio.CancelledError:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    session_id: str | None = None
    started = time.monotonic()
    last_progress = started

    async def stdout_reader() -> None:
        nonlocal session_id, last_progress
        with events_path.open("a", encoding="utf-8") as stream:
            while line := await process.stdout.readline():
                last_progress = time.monotonic()
                text = line.decode("utf-8", errors="replace")
                stream.write(text)
                stream.flush()
                try:
                    event = json.loads(text)
                    session_id = session_id or extract_session(event)
                except json.JSONDecodeError:
                    pass

    async def stderr_reader() -> None:
        nonlocal last_progress
        with stderr_path.open("a", encoding="utf-8") as stream:
            while line := await process.stderr.readline():
                last_progress = time.monotonic()
                stream.write(line.decode("utf-8", errors="replace"))
                stream.flush()

    readers = [asyncio.create_task(stdout_reader()), asyncio.create_task(stderr_reader())]
    wait_task = asyncio.create_task(process.wait())

    async def stop_child() -> None:
        if process.returncode is None:
            try:
                process.send_signal(
                    signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
                )
            except (ProcessLookupError, OSError):
                pass
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=10)
        except asyncio.TimeoutError:
            if process.returncode is None:
                process.kill()
            await wait_task

    idle_timeout = int(config.get("agent_idle_timeout_seconds", timeout))
    timeout_reason: str | None = None
    try:
        while not wait_task.done():
            await asyncio.wait({wait_task}, timeout=1.0)
            if wait_task.done():
                break
            moment = time.monotonic()
            if moment - started >= timeout:
                timeout_reason = f"Codex step exceeded {timeout} seconds"
                break
            if moment - last_progress >= idle_timeout:
                timeout_reason = (
                    f"Codex step produced no events for {idle_timeout} seconds"
                )
                break
        if timeout_reason:
            await stop_child()
    except asyncio.CancelledError:
        await stop_child()
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        raise
    try:
        await asyncio.wait_for(
            asyncio.gather(*readers, return_exceptions=True), timeout=10,
        )
    except asyncio.TimeoutError:
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
    if timeout_reason:
        raise AgentExecutionTimeout(timeout_reason)
    result = None
    if last_path.is_file():
        try:
            result = json.loads(last_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = None
    return_code = process.returncode or 0
    if return_code:
        stderr_tail = ""
        if stderr_path.is_file():
            stderr_tail = stderr_path.read_text(
                encoding="utf-8", errors="replace",
            )[-2000:].strip()
        raise EnvironmentLaunchError(
            f"Codex CLI exited with code {return_code}"
            + (f": {stderr_tail}" if stderr_tail else "")
        )
    if result is None:
        raise AgentProtocolError("Codex exited without a valid structured response")
    return return_code, session_id, result


def implementation_prompt(task: dict[str, Any]) -> str:
    base = (PROMPTS / "implement.md").read_text(encoding="utf-8")
    context = research_context(task)
    return (
        base
        + "\n\n当前算法任务：\n"
        + json.dumps(task, ensure_ascii=False, indent=2)
        + "\n\n经过裁剪的研究记忆（含真实正反例、Pareto 起点和停滞信号）：\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n\n研究记忆只用于形成可证伪方案；当前版本仍须独立核对代码、理论与评测证据。"
    )


def implementation_finalize_prompt(task: dict[str, Any]) -> str:
    return (
        "A previous implementation Agent timed out after writing substantial artifacts. "
        "Do not restart the research or redesign the algorithm. Read AGENTS.md, the policy "
        "skill, and the existing target version. Check policy.md, solution.py and optional "
        "trials for completeness; repair only concrete syntax, interface, numerical-safety or "
        "scope problems; run bounded non-official smoke checks; then return the worker-result "
        "JSON. Do not run datasets, create report.md, or propose follow-up algorithms.\n\n"
        "Target task:\n" + json.dumps(task, ensure_ascii=False, indent=2)
    )


def sync_workspace_instructions(workspace: Path) -> None:
    """Refresh trusted workflow instructions before resuming an old snapshot."""
    shutil.copy2(ROOT / "AGENTS.md", workspace / "AGENTS.md")
    for name in ("prompts", "schemas", "skills"):
        source = AGENT_ROOT / name
        target = workspace / ".agent" / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def recoverable_implementation_artifacts(run_dir: Path, version: str) -> bool:
    target = run_dir / "workspace" / "solution" / version
    policy = target / "policy.md"
    solution = target / "solution.py"
    artifacts_exist = (
        policy.is_file() and policy.stat().st_size >= 512
        and solution.is_file() and solution.stat().st_size >= 512
    )
    # New runs persist the pre-Agent manifest.  The marker exists only for
    # legacy snapshots that were manually audited before this guard existed.
    audit = run_dir / "legacy-scope-audit.json"
    legacy_audited = False
    if audit.is_file():
        try:
            payload = read_json(audit)
            legacy_audited = (
                payload.get("version") == version
                and int(payload.get("outside_target_changes", -1)) == 0
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            legacy_audited = False
    return artifacts_exist and (
        (run_dir / "workspace-before.json").is_file() or legacy_audited
    )


def recoverable_report_checkpoint(run_dir: Path) -> bool:
    """Check that every artifact required by the reporting resume path exists."""
    if not (run_dir / "workspace").is_dir():
        return False
    required = (
        run_dir / "worker-result.json",
        run_dir / "evaluation-summary.json",
        run_dir / "checkpoint.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        worker = read_json(required[0]).get("result") or {}
        checkpoint = read_json(required[2])
        return (
            worker.get("status") == "implemented"
            and checkpoint.get("stage") == "formal_evaluated"
            and bool(checkpoint.get("solution_sha256"))
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def recoverable_record_checkpoint(run_dir: Path, version: str) -> bool:
    """Check artifacts needed to finish registry/queue recording without Agents."""
    required = (
        run_dir / "worker-result.json",
        run_dir / "evaluation-summary.json",
        run_dir / "checkpoint.json",
        run_dir / "report-feedback.json",
        SOLUTION_ROOT / version / "solution.py",
        SOLUTION_ROOT / version / "report.md",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        worker = read_json(required[0]).get("result") or {}
        checkpoint = read_json(required[2])
        feedback = read_json(required[3])
        return (
            worker.get("status") == "implemented"
            and checkpoint.get("stage") == "reported"
            and feedback.get("status") == "written"
            and bool(checkpoint.get("solution_sha256"))
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def recovery_workspace_baseline(
    run_dir: Path, workspace: Path, version: str,
) -> dict[str, str]:
    """Restore the original write-scope baseline for implementation recovery.

    Trusted instructions may legitimately be refreshed between runs.  All
    other paths keep their original digests, so an Agent's pre-timeout
    out-of-scope writes remain observable after finalization.
    """
    manifest_path = run_dir / "workspace-before.json"
    if manifest_path.is_file():
        payload = read_json(manifest_path)
        original = payload.get("files")
        if not isinstance(original, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in original.items()
        ):
            raise RuntimeError("invalid workspace-before recovery manifest")
        baseline = dict(original)
    else:
        audit = run_dir / "legacy-scope-audit.json"
        if not audit.is_file():
            raise RuntimeError("implementation recovery lacks a scope baseline")
        payload = read_json(audit)
        if (
            payload.get("version") != version
            or int(payload.get("outside_target_changes", -1)) != 0
        ):
            raise RuntimeError("legacy implementation scope audit is invalid")
        baseline = tree_manifest(workspace)

    current = tree_manifest(workspace)
    trusted_prefixes = (
        "AGENTS.md", ".agent/prompts/", ".agent/schemas/", ".agent/skills/",
    )

    def trusted(path: str) -> bool:
        return path == trusted_prefixes[0] or path.startswith(trusted_prefixes[1:])

    for path in list(baseline):
        if trusted(path):
            baseline.pop(path)
    for path, digest in current.items():
        if trusted(path):
            baseline[path] = digest
    return baseline


def create_workspace(run_dir: Path, source_root: Path | None = None) -> Path:
    """Create a lightweight code snapshot; multi-GB datasets stay at the root evaluator."""
    source_root = (source_root or ROOT).resolve()
    source_agent_root = source_root / ".agent"
    workspace = run_dir / "workspace"

    def ignored(directory: str, names: list[str]) -> set[str]:
        result = {name for name in names if name in {".git", "__pycache__"} or name.endswith(".pyc")}
        if Path(directory).resolve() == source_root:
            result.update(name for name in names if name in {"datasets", "reference"})
        if Path(directory).resolve() == source_agent_root and "runtime" in names:
            result.add("runtime")
        return result

    shutil.copytree(source_root, workspace, ignore=ignored)
    return workspace


def assert_workspace_scope(
    workspace: Path, before: dict[str, str], version: str, *, context: str,
) -> dict[str, str]:
    after = tree_manifest(workspace)
    assert_changes_within(
        before, after, allowed_prefixes=[f"solution/{version}"], context=context,
    )
    return after


def import_version(workspace: Path, version: str, *, report_only: bool = False) -> None:
    source = workspace / "solution" / version
    target = SOLUTION_ROOT / version
    names = ["report.md"] if report_only else ["solution.py", "policy.md"]
    for name in names:
        if not (source / name).is_file():
            raise RuntimeError(f"Agent did not produce {name}")
        shutil.copy2(source / name, target / name)
    if not report_only:
        target_trials = target / "trials"
        if target_trials.exists():
            shutil.rmtree(target_trials)
        if (source / "trials").is_dir():
            shutil.copytree(source / "trials", target_trials)


async def structural_review(
    config: dict[str, Any], task: dict[str, Any], run_dir: Path, workspace: Path,
) -> None:
    prompt = (PROMPTS / "review.md").read_text(encoding="utf-8")
    prompt += "\n\nVersion task:\n" + json.dumps(task, ensure_ascii=False, indent=2)
    code, _, result = await run_codex(
        config, run_dir / "review", prompt, SCHEMAS / "review-result.schema.json",
        int(config["worker_timeout_seconds"]), workspace, role="review",
    )
    checks = result.get("checks", {}) if result else {}
    all_checks_passed = bool(checks) and all(value is True for value in checks.values())
    if code or not result or result.get("status") != "approved" or not all_checks_passed:
        reasons = result.get("reasons", []) if result else []
        failed_checks = [name for name, passed in checks.items() if passed is not True]
        reasons.extend(f"check failed: {name}" for name in failed_checks)
        raise RuntimeError("structural review rejected: " + "; ".join(reasons))


def validate_evaluation_case_counts(
    config: dict[str, Any], result: dict[str, Any], *, expected: dict[str, int] | None = None,
) -> None:
    expected = expected or config["fixed_evaluation_cases"]
    observed = {
        "linear": int(result.get("linear_case_count", -1)),
        "attention": int(result.get("attention_case_count", -1)),
    }
    if observed != expected:
        raise RuntimeError(
            f"evaluation case-count mismatch: expected {expected}, observed {observed}"
        )


async def run_evaluation_batch(
    config: dict[str, Any], candidates: list[tuple[str, Path]], run_dir: Path,
    *, fidelity: str, linear_groups: int | None = None,
    attention_groups: int | None = None, skip_self_check: bool = False,
) -> list[dict[str, Any]]:
    output = run_dir / f"evaluation-{fidelity}.json"
    command = [
        config.get("evaluation_python", sys.executable),
        str(AGENT_ROOT / "skills/hif4-evaluate/scripts/evaluate.py"),
        *(str(path) for _, path in candidates),
        "--datasets-dir", str(ROOT / config["evaluation_datasets_dir"]),
        "--json-output", str(output),
    ]
    if linear_groups is not None:
        command.extend(["--linear-groups", str(linear_groups)])
    if attention_groups is not None:
        command.extend(["--attention-groups", str(attention_groups)])
    if skip_self_check:
        command.append("--skip-self-check")

    trust = protected_manifest(ROOT, config["protected_files"])
    dataset_manifest_path = ROOT / config["evaluation_datasets_dir"] / "manifest.json"
    dataset_guard = config.get("_dataset_guard") or dataset_file_signatures(
        dataset_manifest_path
    )
    assert_dataset_signatures(
        dataset_manifest_path, dataset_guard, context=f"before {fidelity} evaluation"
    )
    candidate_hashes = {str(path): solution_hash(path) for _, path in candidates}
    timeout_key = "screening_timeout_seconds" if fidelity == "screening" else "evaluation_timeout_seconds"
    try:
        completed = await asyncio.to_thread(
            subprocess.run, command, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=int(config.get(timeout_key, config["evaluation_timeout_seconds"])),
            env=utf8_environment(),
        )
    except subprocess.TimeoutExpired as error:
        partial_stdout = error.stdout or ""
        partial_stderr = error.stderr or ""
        if isinstance(partial_stdout, bytes):
            partial_stdout = partial_stdout.decode("utf-8", errors="replace")
        if isinstance(partial_stderr, bytes):
            partial_stderr = partial_stderr.decode("utf-8", errors="replace")
        (run_dir / f"evaluation-{fidelity}.log").write_text(
            str(partial_stdout) + "\n" + str(partial_stderr), encoding="utf-8"
        )
        assert_protected_unchanged(ROOT, trust, context=f"timed-out {fidelity} evaluation")
        assert_dataset_signatures(
            dataset_manifest_path, dataset_guard,
            context=f"timed-out {fidelity} evaluation",
        )
        changed_candidates = [
            path for path, digest in candidate_hashes.items()
            if not Path(path).is_file() or solution_hash(Path(path)) != digest
        ]
        if changed_candidates:
            raise RuntimeError(
                "timed-out evaluator modified candidate solution(s): "
                + ", ".join(changed_candidates)
            ) from error
        raise EvaluationTimeoutError(
            f"{fidelity} evaluation exceeded {config.get(timeout_key)} seconds"
        ) from error
    except OSError as error:
        raise EnvironmentLaunchError(
            f"official evaluator could not start environment Python: {error}"
        ) from error
    (run_dir / f"evaluation-{fidelity}.log").write_text(
        completed.stdout + "\n" + completed.stderr, encoding="utf-8"
    )
    assert_protected_unchanged(ROOT, trust, context=f"{fidelity} evaluation")
    assert_dataset_signatures(
        dataset_manifest_path, dataset_guard, context=f"{fidelity} evaluation"
    )
    changed_candidates = [
        path for path, digest in candidate_hashes.items()
        if not Path(path).is_file() or solution_hash(Path(path)) != digest
    ]
    if changed_candidates:
        raise RuntimeError(
            "evaluator modified candidate solution(s): " + ", ".join(changed_candidates)
        )
    if completed.returncode or not output.is_file():
        raise RuntimeError(f"{fidelity} evaluation failed")
    raw_results = read_json(output).get("results", [])
    if len(raw_results) != len(candidates):
        raise RuntimeError(
            f"{fidelity} evaluator returned {len(raw_results)} result(s) for "
            f"{len(candidates)} candidate(s)"
        )
    results: list[dict[str, Any]] = []
    for (label, path), result in zip(candidates, raw_results):
        result["config"] = label
        result["solution_path"] = str(path.relative_to(ROOT))
        result["solution_sha256"] = candidate_hashes[str(path)]
        result["evaluation_fidelity"] = fidelity
        results.append(result)
    return results


def evaluation_rank(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(result["total_score"]),
        -float(result.get("seconds", float("inf"))),
        -float(result["attention_output"]["mse"]),
        -float(result["linear_output"]["mse"]),
    )


def selected_evaluation_result(evaluation: dict[str, Any]) -> dict[str, Any]:
    selected_name = evaluation.get("selected")
    for result in evaluation.get("formal", {}).get("results", []):
        if result.get("config") == selected_name:
            return result
    raise RuntimeError(f"selected formal result is missing: {selected_name}")


async def evaluate_candidates(
    config: dict[str, Any], task: dict[str, Any], run_dir: Path,
    evaluation_lock: asyncio.Lock,
) -> tuple[dict[str, Any], dict[str, Any]]:
    version_dir = SOLUTION_ROOT / task["version"]
    candidates = [("main", version_dir / "solution.py")]
    for path in sorted((version_dir / "trials").glob("*/solution.py")) if (version_dir / "trials").is_dir() else []:
        candidates.append((path.parent.name, path))
    limit = int(config["max_hyperparameter_configs"])
    if len(candidates) > limit:
        raise ValueError(f"algorithm produced {len(candidates)} configurations; maximum is {limit}")
    screening_results: list[dict[str, Any]] = []
    formal_results: list[dict[str, Any]] = []
    async with evaluation_lock:
        with file_lock(EVALUATION_LOCK):
            screening = config.get("screening", {})
            if (
                screening.get("enabled", False)
                and len(candidates) > int(screening["promote_top_k"])
            ):
                screening_results = await run_evaluation_batch(
                    config, candidates, run_dir, fidelity="screening",
                    linear_groups=int(screening["linear_groups"]),
                    attention_groups=int(screening["attention_groups"]),
                )
                expected = {
                    "linear": int(screening["linear_cases"]),
                    "attention": int(screening["attention_cases"]),
                }
                for result in screening_results:
                    validate_evaluation_case_counts(config, result, expected=expected)
                promoted_labels = {
                    item["config"]
                    for item in sorted(screening_results, key=evaluation_rank, reverse=True)[
                        : int(screening["promote_top_k"])
                    ]
                }
                promoted = [item for item in candidates if item[0] in promoted_labels]
                formal_results = await run_evaluation_batch(
                    config, promoted, run_dir, fidelity="formal", skip_self_check=True,
                )
            else:
                formal_results = await run_evaluation_batch(
                    config, candidates, run_dir, fidelity="formal",
                )
            for result in formal_results:
                validate_evaluation_case_counts(config, result)
    selected = max(formal_results, key=evaluation_rank)
    selected_path = ROOT / selected["solution_path"]
    if selected_path != version_dir / "solution.py":
        shutil.copy2(selected_path, version_dir / "solution.py")
        selected["solution_sha256"] = solution_hash(version_dir / "solution.py")
    evaluation_record = {
        "schema_version": 2,
        "selected": selected["config"],
        "screening": {
            "enabled": bool(screening_results),
            "results": screening_results,
        },
        "formal": {"results": formal_results},
    }
    atomic_json(run_dir / "evaluation-summary.json", evaluation_record)
    return selected, evaluation_record


async def write_report(
    config: dict[str, Any], task: dict[str, Any], run_dir: Path,
    selected: dict[str, Any], evaluation: dict[str, Any], workspace: Path,
) -> dict[str, Any]:
    def compact_result(item: dict[str, Any]) -> dict[str, Any]:
        view = {key: value for key, value in item.items() if key != "case_diagnostics"}
        cases = item.get("case_diagnostics", {})
        view["representative_cases"] = {}
        for category in ("linear", "attention"):
            ordered = sorted(
                cases.get(category, []),
                key=lambda case: float(case.get("score_percentage_points", 0.0)),
            )
            view["representative_cases"][category] = {
                "worst": ordered[:5],
                "best": ordered[-2:] if ordered else [],
            }
        return view

    compact_evaluation = {
        "schema_version": evaluation.get("schema_version"),
        "selected": evaluation.get("selected"),
        "screening": {
            "enabled": evaluation.get("screening", {}).get("enabled", False),
            "results": [
                compact_result(item)
                for item in evaluation.get("screening", {}).get("results", [])
            ],
        },
        "formal": {
            "results": [
                compact_result(item)
                for item in evaluation.get("formal", {}).get("results", [])
            ]
        },
    }
    prompt = (PROMPTS / "report.md").read_text(encoding="utf-8")
    prompt += "\n\n版本任务：\n" + json.dumps(task, ensure_ascii=False, indent=2)
    prompt += "\n\n裁剪后的全局研究记忆与可选 Pareto 来源：\n" + json.dumps(
        research_context(task), ensure_ascii=False, indent=2,
    )
    prompt += "\n\n真实评测结果：\n" + json.dumps(
        {"selected": compact_result(selected), "evaluation": compact_evaluation},
        ensure_ascii=False, indent=2
    )
    prompt += (
        "\n\n除写 report.md 外，请在结构化返回值中提炼可被下一轮复用的假设判定、"
        "正负经验、恰好三个后续结构方向和最多三个原则更新建议。后续方向必须在看到本次正式结果后形成，"
        "并保持两个 explore、一个 exploit；原则建议不会自动生效，Runner 会按独立算法族证据门禁处理。"
    )
    code, _, result = await run_codex(
        config, run_dir / "report", prompt, SCHEMAS / "report-result.schema.json",
        int(config["worker_timeout_seconds"]), workspace, role="report",
    )
    report = workspace / "solution" / task["version"] / "report.md"
    if code or not result or result.get("status") != "written" or not report.is_file():
        raise RuntimeError("report agent did not produce report.md")
    directions = result.get("next_algorithms", [])
    modes = [item.get("search_mode") for item in directions]
    if len(directions) != 3 or modes.count("explore") != 2 or modes.count("exploit") != 1:
        raise RuntimeError(
            "report agent must propose exactly two explore and one exploit direction"
        )
    import_version(workspace, task["version"], report_only=True)
    screening_rank = [
        item["config"]
        for item in sorted(
            evaluation.get("screening", {}).get("results", []),
            key=evaluation_rank, reverse=True,
        )
    ]
    formal_rank = [
        item["config"]
        for item in sorted(
            evaluation.get("formal", {}).get("results", []),
            key=evaluation_rank, reverse=True,
        )
    ]
    result["screening_summary"] = {
        "enabled": evaluation.get("screening", {}).get("enabled", False),
        "evaluated_configs": len(evaluation.get("screening", {}).get("results", [])),
        "promoted_configs": len(evaluation.get("formal", {}).get("results", [])),
        "selected": evaluation.get("selected"),
        "screening_ranking": screening_rank,
        "formal_ranking": formal_rank,
        "winner_agreement": (
            screening_rank[0] == formal_rank[0]
            if screening_rank and formal_rank else None
        ),
    }
    atomic_json(run_dir / "report-feedback.json", result)
    return result


def record_completed(
    task: dict[str, Any], selected: dict[str, Any], run_id: str,
) -> tuple[str, tuple[str, float]]:
    registry = read_json(REGISTRY_PATH)
    node = registry["versions"][task["version"]]
    previous_best = best_reference(registry)
    baseline_score = float(registry["versions"][previous_best]["metrics"]["score"])
    node["metrics"] = {
        "linear_mse": selected["linear_output"]["mse"],
        "attention_mse": selected["attention_output"]["mse"],
        "score": selected["total_score"],
        "seconds": selected.get("seconds"),
        "linear_case_count": selected["linear_case_count"],
        "attention_case_count": selected["attention_case_count"],
        "dataset": "datasets/combined",
    }
    node["status"] = (
        "promising" if float(selected["total_score"]) > baseline_score else "rejected"
    )
    node["solution_sha256"] = selected["solution_sha256"]
    node["evaluation_run"] = run_id
    node["selected_config"] = selected.get("config")
    node["diagnostics"] = {
        "score_statistics": selected.get("score_statistics", {}),
        "timings": selected.get("timings", {}),
        "implementation": selected.get("implementation_diagnostics"),
    }
    atomic_json(REGISTRY_PATH, registry)
    archive = refresh_pareto_archive()
    node["pareto_member"] = any(
        item["version"] == task["version"] for item in archive.get("front", [])
    )
    atomic_json(REGISTRY_PATH, registry)
    return node["status"], (previous_best, baseline_score)


def enqueue_followups(
    queue: dict[str, Any], config: dict[str, Any], task: dict[str, Any],
    result: dict[str, Any], _score: float,
) -> int:
    """Turn structural directions into independent flat-version tasks."""
    added = 0
    for proposal in result.get("next_algorithms", [])[: int(config["directions_per_version"])]:
        structural = str(proposal.get("structural_change", "")).strip()
        evidence = str(proposal.get("evidence", "")).strip()
        root_cause = str(proposal.get("root_cause", "")).strip()
        family = str(proposal.get("algorithm_family", ""))
        mode = str(proposal.get("search_mode", ""))
        if (
            len(structural) < 20 or len(evidence) < 10 or len(root_cause) < 20
            or PARAM_ONLY.search(family) or mode not in {"explore", "exploit"}
        ):
            continue
        registry = read_json(REGISTRY_PATH)
        based_on = str(proposal.get("based_on") or task["version"])
        source = registry.get("versions", {}).get(based_on)
        if not source or not metrics_use_current_profile(source.get("metrics")):
            continue
        if mode == "exploit" and float(proposal.get("evidence_strength", 0.0)) < 0.6:
            continue
        version = next_version(proposal["version_suffix"], registry)
        try:
            add_algorithm_task(
                queue, config, based_on=based_on, version=version,
                focus=proposal["focus"], family=family,
                hypothesis=proposal["hypothesis"],
                base=proposal["implementation_base"], priority=0.5,
                structural_change=structural,
                evidence=evidence,
                evidence_strength=float(proposal.get("evidence_strength", 0.5)),
                novelty=float(proposal.get("novelty", 0.5)),
                uncertainty=float(proposal.get("uncertainty", 0.5)),
                expected_cost=float(proposal.get("expected_cost", 1.0)),
                target_metric=str(proposal.get("target_metric", "score")),
                falsification=str(proposal.get("falsification", "")),
                search_mode=mode, root_cause=root_cause,
            )
        except (ValueError, FileExistsError):
            continue
        added += 1
    return added


def rerank_queued_tasks(queue: dict[str, Any]) -> bool:
    registry = read_json(REGISTRY_PATH)
    experiments = load_experiments()["experiments"] if EXPERIMENTS_PATH.is_file() else []
    changed = False
    for task in queue["tasks"]:
        if task.get("search_mode") not in {"explore", "exploit"}:
            task["search_mode"] = infer_search_mode(task)
            changed = True
        if not task.get("root_cause"):
            task["root_cause"] = task.get("hypothesis", "unspecified root cause")
            changed = True
        if task.get("status") != "queued":
            continue
        priority, components = adaptive_priority(task, registry, experiments)
        if abs(float(task.get("priority", 0.0)) - priority) > 1e-9 or task.get("priority_components") != components:
            task["priority"] = priority
            task["priority_components"] = components
            task["priority_updated_at"] = now()
            changed = True
    return changed


def mark_failed_version(
    task: dict[str, Any], error: str, *, status: str = "failed",
) -> None:
    if task.get("kind") != "algorithm" or not task.get("version"):
        return
    registry = read_json(REGISTRY_PATH)
    node = registry.get("versions", {}).get(task["version"])
    if node and node.get("metrics") is None:
        node["status"] = status
        node["failure"] = error
        atomic_json(REGISTRY_PATH, registry)


class Scheduler:
    def __init__(self, config: dict[str, Any], *, dry_run: bool, once: bool) -> None:
        self.config = config
        self.dry_run = dry_run
        self.once = once
        self.queue_lock = asyncio.Lock()
        self.evaluation_lock = asyncio.Lock()
        self.running: dict[str, asyncio.Task[None]] = {}
        self.dispatched_once = False

    async def update_task(self, task_id: str, **changes: Any) -> None:
        async with self.queue_lock:
            queue = load_queue()
            task = next(item for item in queue["tasks"] if item["task_id"] == task_id)
            task.update(changes)
            task["updated_at"] = now()
            save_queue(queue)

    async def run_algorithm(self, task: dict[str, Any], run_dir: Path) -> None:
        root_trust = protected_manifest(ROOT, self.config["protected_files"])
        resume_report = (
            task.get("resume_stage") == "reporting"
            and recoverable_report_checkpoint(run_dir)
        )
        resume_record = (
            task.get("resume_stage") == "recording"
            and recoverable_record_checkpoint(run_dir, task["version"])
        )
        resume_finalize = (
            task.get("resume_stage") == "implementation_finalize"
            and (run_dir / "workspace").is_dir()
            and recoverable_implementation_artifacts(run_dir, task["version"])
        )
        if resume_record:
            evaluation, selected = validate_formal_checkpoint(
                run_dir, self.config, stages={"reported"},
            )
            checkpoint = read_json(run_dir / "checkpoint.json")
            main_path = SOLUTION_ROOT / task["version"] / "solution.py"
            if solution_hash(main_path) != selected["solution_sha256"]:
                raise RuntimeError("cannot resume recording: evaluated solution hash changed")
            report_feedback = read_json(run_dir / "report-feedback.json")
            report_path = SOLUTION_ROOT / task["version"] / "report.md"
            if (
                checkpoint.get("report_feedback_sha256")
                and solution_hash(run_dir / "report-feedback.json")
                != checkpoint["report_feedback_sha256"]
            ):
                raise RuntimeError("cannot resume recording: report feedback changed")
            if (
                checkpoint.get("report_sha256")
                and solution_hash(report_path) != checkpoint["report_sha256"]
            ):
                raise RuntimeError("cannot resume recording: report changed")
            await self.update_task(task["task_id"], stage="recording")
        elif resume_report:
            workspace = run_dir / "workspace"
            sync_workspace_instructions(workspace)
            worker_payload = read_json(run_dir / "worker-result.json")
            result = worker_payload.get("result")
            if not result or result.get("status") != "implemented":
                raise RuntimeError("cannot resume: valid worker result is missing")
            evaluation, selected = validate_formal_checkpoint(
                run_dir, self.config, stages={"formal_evaluated"},
            )
            selected_source = ROOT / selected["solution_path"]
            if not selected_source.is_file() or solution_hash(selected_source) != selected["solution_sha256"]:
                raise RuntimeError("cannot resume: selected evaluated solution hash changed")
            main_path = SOLUTION_ROOT / task["version"] / "solution.py"
            if solution_hash(main_path) != selected["solution_sha256"]:
                shutil.copy2(selected_source, main_path)
            await self.update_task(task["task_id"], stage="reporting")
        else:
            if resume_finalize:
                workspace = run_dir / "workspace"
                sync_workspace_instructions(workspace)
                workspace_before = recovery_workspace_baseline(
                    run_dir, workspace, task["version"],
                )
                implementation_run_dir = run_dir / f"finalize-{int(time.time())}"
                prompt = implementation_finalize_prompt(task)
                role = "finalize"
            else:
                workspace = await asyncio.to_thread(create_workspace, run_dir)
                workspace_before = tree_manifest(workspace)
                atomic_json(run_dir / "workspace-before.json", {
                    "schema_version": 1,
                    "files": workspace_before,
                    "resume_finalize": False,
                    "recorded_at": now(),
                })
                implementation_run_dir = run_dir
                prompt = implementation_prompt(task)
                role = infer_search_mode(task)
            # This stage precedes formal evaluation, so its eventual score must
            # be bound to the current evaluation contract rather than a stale
            # pre-recovery one.
            write_trust_manifest(run_dir, self.config, root_trust)
            await self.update_task(task["task_id"], stage="implementing")
            code, session, result = await run_codex(
                self.config, implementation_run_dir, prompt,
                SCHEMAS / "worker-result.schema.json",
                int(self.config["worker_timeout_seconds"]), workspace, role=role,
            )
            atomic_json(run_dir / "worker-result.json", {
                "session_id": session, "exit_code": code, "result": result,
            })
            if result_is_environment_failure(result):
                raise EnvironmentLaunchError(
                    "Codex started, but its local tool process could not start: "
                    + str(result.get("algorithm_summary", "unknown environment failure"))
                )
            if code or not result or result.get("status") != "implemented":
                raise RuntimeError("implementation agent did not complete")
            workspace_after_implementation = assert_workspace_scope(
                workspace, workspace_before, task["version"], context="implementation",
            )
            assert_protected_unchanged(ROOT, root_trust, context="implementation Agent")
            await self.update_task(task["task_id"], stage="structural_review")
            await structural_review(self.config, task, run_dir, workspace)
            review_after = tree_manifest(workspace)
            review_changes = [
                path for path in set(workspace_after_implementation) | set(review_after)
                if workspace_after_implementation.get(path) != review_after.get(path)
            ]
            if review_changes:
                raise RuntimeError(
                    "read-only structural reviewer modified files: "
                    + ", ".join(sorted(review_changes))
                )
            import_version(workspace, task["version"])
            await self.update_task(task["task_id"], stage="evaluation")
            selected, evaluation = await evaluate_candidates(
                self.config, task, run_dir, self.evaluation_lock
            )
            trust = read_json(run_dir / "trust-manifest.json")
            atomic_json(run_dir / "checkpoint.json", {
                "schema_version": 1,
                "stage": "formal_evaluated",
                "selected": selected["config"],
                "solution_sha256": selected["solution_sha256"],
                "evaluation_contract_sha256": trust["evaluation_contract_sha256"],
                "evaluation_summary_sha256": canonical_digest(evaluation),
                "updated_at": now(),
            })
        if not resume_record:
            shutil.copy2(
                SOLUTION_ROOT / task["version"] / "solution.py",
                workspace / "solution" / task["version"] / "solution.py",
            )
            before_report = tree_manifest(workspace)
            root_version_before_report = tree_manifest(
                SOLUTION_ROOT / task["version"]
            )
            await self.update_task(task["task_id"], stage="reporting")
            report_feedback = await write_report(
                self.config, task, run_dir, selected, evaluation, workspace
            )
            after_report = tree_manifest(workspace)
            changed_by_report = sorted(
                path for path in set(before_report) | set(after_report)
                if before_report.get(path) != after_report.get(path)
            )
            expected_report = f"solution/{task['version']}/report.md"
            if any(path != expected_report for path in changed_by_report):
                raise RuntimeError(
                    "report Agent modified files other than report.md: "
                    + ", ".join(changed_by_report)
                )
            root_version_after_report = tree_manifest(
                SOLUTION_ROOT / task["version"]
            )
            root_changes = sorted(
                path
                for path in set(root_version_before_report) | set(root_version_after_report)
                if root_version_before_report.get(path)
                != root_version_after_report.get(path)
            )
            if any(path != "report.md" for path in root_changes):
                raise RuntimeError(
                    "report Agent modified root version files other than report.md: "
                    + ", ".join(root_changes)
                )
            if solution_hash(
                SOLUTION_ROOT / task["version"] / "solution.py"
            ) != selected["solution_sha256"]:
                raise RuntimeError("report Agent changed the formally evaluated solution")
            formal_checkpoint = read_json(run_dir / "checkpoint.json")
            atomic_json(run_dir / "checkpoint.json", {
                "schema_version": 1, "stage": "reported",
                "selected": selected["config"],
                "solution_sha256": selected["solution_sha256"],
                "evaluation_contract_sha256": formal_checkpoint.get(
                    "evaluation_contract_sha256"
                ),
                "evaluation_summary_sha256": formal_checkpoint.get(
                    "evaluation_summary_sha256"
                ),
                "report_feedback_sha256": solution_hash(
                    run_dir / "report-feedback.json"
                ),
                "report_sha256": solution_hash(
                    SOLUTION_ROOT / task["version"] / "report.md"
                ),
                "updated_at": now(),
            })
        # Re-validate the formal evidence immediately before the serialized
        # registry transition.  This catches post-evaluation file drift even
        # when no report Agent is involved in a recording-only recovery.
        _, verified_selected = validate_formal_checkpoint(
            run_dir, self.config, stages={"reported"},
        )
        if solution_hash(
            SOLUTION_ROOT / task["version"] / "solution.py"
        ) != verified_selected["solution_sha256"]:
            raise RuntimeError("evaluated solution changed before registry recording")
        assert_protected_unchanged(ROOT, root_trust, context="complete algorithm task")
        # Registry update and proposal creation are one serialized state transition.
        # Otherwise two workers can allocate the same global numeric version.
        await self.update_task(task["task_id"], stage="recording")
        async with self.queue_lock:
            _, best_before = record_completed(task, selected, run_dir.name)
            record_experiment(
                task, selected, run_dir.name, best_before, report_feedback,
            )
            queue = load_queue()
            enqueue_followups(
                queue, self.config, task, report_feedback,
                float(selected["total_score"]),
            )
            rerank_queued_tasks(queue)
            save_queue(queue)

    async def load_ranked_queue(
        self,
    ) -> tuple[dict[str, Any], tuple[str | None, float]]:
        async with self.queue_lock:
            queue = load_queue()
            if rerank_queued_tasks(queue):
                save_queue(queue)
            # The registry is updated under the same lock at task completion.
            # Returning the target state here prevents a newly completed winner
            # from racing with one more dispatch batch.
            return queue, best_score_state(read_json(REGISTRY_PATH))

    async def worker(self, task: dict[str, Any]) -> None:
        run_id = str(task.get("resume_run_id") or f"{task['task_id']}-{int(time.time())}")
        run_dir = RUNS / run_id
        atomic_json(run_dir / "run.json", {
            "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
            "kind": task["kind"], "state": "running", "started_at": now(),
            "resumed": bool(task.get("resume_run_id")),
        })
        await self.update_task(
            task["task_id"], status="running", stage="starting", run_id=run_id
        )
        try:
            await self.run_algorithm(task, run_dir)
            await self.update_task(
                task["task_id"], status="completed", stage="completed",
                resume_run_id=None, resume_stage=None,
            )
            atomic_json(run_dir / "run.json", {
                "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
                "kind": task["kind"], "state": "completed", "finished_at": now(),
            })
        except asyncio.CancelledError:
            # Preserve run_id and the last durable stage.  recover() decides
            # whether to reuse formal/report or implementation artifacts.
            async with self.queue_lock:
                queue = load_queue()
                queued_task = next(
                    item for item in queue["tasks"] if item["task_id"] == task["task_id"]
                )
                failed_stage = queued_task.get("stage")
                queued_task.update(
                    status="workflow_failed", stage="interrupted",
                    failed_stage=failed_stage,
                    error="scheduler interrupted; checkpoint preserved",
                    updated_at=now(),
                )
                save_queue(queue)
            atomic_json(run_dir / "run.json", {
                "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
                "kind": task["kind"], "state": "interrupted",
                "failed_stage": failed_stage,
                "finished_at": now(),
            })
            raise
        except (AgentExecutionTimeout, AgentProtocolError) as error:
            async with self.queue_lock:
                queue = load_queue()
                queued_task = next(
                    item for item in queue["tasks"] if item["task_id"] == task["task_id"]
                )
                failed_stage = queued_task.get("stage")
                failure_stage = (
                    "agent_timeout"
                    if isinstance(error, AgentExecutionTimeout)
                    else "agent_protocol_error"
                )
                mark_failed_version(task, str(error), status="workflow_failed")
                record_failed_experiment(
                    task, run_dir.name, str(error), outcome="workflow_failed"
                )
                queued_task.update(
                    status="workflow_failed", stage=failure_stage,
                    failed_stage=failed_stage, error=str(error), updated_at=now(),
                )
                rerank_queued_tasks(queue)
                save_queue(queue)
            atomic_json(run_dir / "run.json", {
                "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
                "kind": task["kind"], "state": "workflow_failed",
                "failed_stage": failed_stage, "finished_at": now(),
                "error": str(error),
            })
        except EnvironmentLaunchError as error:
            # Infrastructure failure is not evidence against the algorithm.
            await self.update_task(
                task["task_id"], status="environment_failed",
                stage="environment_failed", error=str(error)
            )
            atomic_json(run_dir / "run.json", {
                "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
                "kind": task["kind"], "state": "environment_failed",
                "finished_at": now(), "error": str(error),
            })
        except EvaluationTimeoutError as error:
            async with self.queue_lock:
                mark_failed_version(task, str(error), status="evaluation_timeout")
                record_failed_experiment(
                    task, run_dir.name, str(error), outcome="evaluation_timeout"
                )
                queue = load_queue()
                queued_task = next(
                    item for item in queue["tasks"] if item["task_id"] == task["task_id"]
                )
                queued_task.update(
                    status="evaluation_timeout", stage="evaluation_timeout",
                    error=str(error), updated_at=now(),
                )
                rerank_queued_tasks(queue)
                save_queue(queue)
            atomic_json(run_dir / "run.json", {
                "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
                "kind": task["kind"], "state": "evaluation_timeout",
                "finished_at": now(), "error": str(error),
            })
        except BaseException as error:
            async with self.queue_lock:
                queue = load_queue()
                queued_task = next(
                    item for item in queue["tasks"] if item["task_id"] == task["task_id"]
                )
                workflow_only = queued_task.get("stage") in {"reporting", "recording"}
                failure_status = "workflow_failed" if workflow_only else "failed"
                mark_failed_version(task, str(error), status=failure_status)
                record_failed_experiment(
                    task, run_dir.name, str(error), outcome=failure_status
                )
                queued_task.update(
                    status=failure_status, stage=failure_status,
                    error=str(error), updated_at=now()
                )
                rerank_queued_tasks(queue)
                save_queue(queue)
            atomic_json(run_dir / "run.json", {
                "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
                "kind": task["kind"], "state": failure_status,
                "finished_at": now(),
                "error": str(error),
            })

    async def loop(self) -> None:
        max_agents = int(self.config["max_agents"])
        while True:
            if STOP_PATH.exists():
                if not self.running:
                    return
                await asyncio.sleep(0.5)
                self.running = {k: v for k, v in self.running.items() if not v.done()}
                continue
            self.running = {k: v for k, v in self.running.items() if not v.done()}
            if self.once and self.dispatched_once:
                if not self.running:
                    return
                await asyncio.sleep(0.5)
                continue
            queue, (best_version, best_score) = await self.load_ranked_queue()
            target_score = float(self.config["target_score"])
            if best_score >= target_score:
                print(
                    f"target reached: {best_version} score={best_score:.6f} "
                    f">= {target_score:.6f}; new dispatch paused"
                )
                if not self.dry_run:
                    STOP_PATH.write_text(
                        f"target reached by {best_version}: {best_score:.12f}\n",
                        encoding="utf-8",
                    )
                if not self.running or self.dry_run:
                    return
                await asyncio.sleep(0.5)
                continue
            pending = [
                task for task in queue["tasks"]
                if task["status"] == "queued" and task["task_id"] not in self.running
            ]
            active = [
                task for task in queue["tasks"]
                if task["status"] == "running" or task["task_id"] in self.running
            ]
            available = max_agents - len(self.running)
            mix = self.config["search_mix"]
            dispatch_batch = select_diverse_batch(
                pending, available, active=active,
                explore_slots=int(mix["explore_slots"]),
                exploit_slots=int(mix["exploit_slots"]),
            )
            for offset, task in enumerate(dispatch_batch, start=1):
                if self.dry_run:
                    print(
                        f"DRY-RUN slot={len(self.running) + offset} {task['kind']} "
                        f"{task['task_id']} {task.get('version','')} "
                        f"priority={float(task.get('priority', 0.0)):.6f} "
                        f"mode={infer_search_mode(task)} "
                        f"cell={task.get('focus')}/{task.get('algorithm_family')}"
                    )
                    continue
                running = asyncio.create_task(self.worker(task))
                self.running[task["task_id"]] = running
            if dispatch_batch:
                self.dispatched_once = True
            if self.dry_run:
                return
            # Empty capacity deliberately waits for valid child directions from
            # completed algorithm tasks. The scheduler never fabricates work.
            await asyncio.sleep(0.5)


def seed(queue: dict[str, Any], config: dict[str, Any]) -> int:
    seeds = [
        {
            "based_on": "v0_softmax_aware_qk",
            "suffix": "discrete_attention_output_search",
            "focus": "attention",
            "family": "discrete_attention_output_search",
            "search_mode": "explore",
            "root_cause": "独立最小化 Q/K 张量误差忽略逐行平移不变性和 Softmax Jacobian，局部 MSE 与最终 Attention 输出损失错位。",
            "hypothesis": "直接以校准 softmax 输出误差联合选择合法 Q/K HiF4 块参数，可减少张量 MSE 与注意力输出目标错位",
            "structural_change": "把独立 Q/K 局部选择替换为受预算约束的联合离散坐标搜索，并以注意力输出损失接受或回退候选。",
            "evidence": "v0_softmax_aware_qk 的 300 例结果只改变 Attention 路径，Attention MSE 相对 v0_hessian_repair 降低约 1.210%。",
            "evidence_strength": 0.8, "novelty": 0.75, "uncertainty": 0.55,
            "expected_cost": 1.6, "target_metric": "attention_mse",
            "falsification": "完整评测 Attention MSE 不低于 v0_softmax_aware_qk，或运行时间越过预算。",
            "base": "based_on",
        },
        {
            "based_on": "v0_alternating_joint_fit",
            "suffix": "activation_whitened_linear_blocks",
            "focus": "linear",
            "family": "activation_whitened_linear_blocks",
            "search_mode": "exploit",
            "root_cause": "Linear 中 ΔXWᵀ 与 XΔWᵀ 沿激活协方差主方向被放大，独立量化两侧无法控制交叉项。",
            "hypothesis": "在保持矩阵乘等价的互逆变换中显式压平激活协方差，再做块量化，可降低高相关通道主导的 Linear 输出误差",
            "structural_change": "依据校准二阶矩构造受 HiF4 分组约束的可逆预条件，再在变换域联合量化权重与激活。",
            "evidence": "v0_alternating_joint_fit 的 Linear MSE 相对 v0_hessian_repair 降低约 3.158%，表明联合建模两侧误差有效但仍可能受相关通道限制。",
            "evidence_strength": 0.75, "novelty": 0.7, "uncertainty": 0.6,
            "expected_cost": 1.4, "target_metric": "linear_mse",
            "falsification": "完整评测 Linear MSE 不低于 v0_alternating_joint_fit，且综合分或耗时明显退化。",
            "base": "based_on",
        },
        {
            "based_on": "v0_hessian_repair",
            "suffix": "error_budgeted_hierarchy",
            "focus": "format",
            "family": "error_budgeted_hif4_hierarchy",
            "search_mode": "explore",
            "root_cause": "同一 64/8/4 值层级中离群值与主体值争用离散共享尺度，使削顶和小值分辨率不足集中在少数高损失块。",
            "hypothesis": "把 LV2/LV3 的局部范围选择改为下游敏感度约束的层级误差预算，可减少少数高损失 case 的主导误差",
            "structural_change": "从零构造合法 HiF4 层级参数选择器，将每层离散决策写成受共享关系约束的误差预算分配。",
            "evidence": "现有保留版本均以聚合目标报告收益，尚未专门优化逐例负收益尾部与 HiF4 层级共享造成的误差集中。",
            "evidence_strength": 0.55, "novelty": 0.9, "uncertainty": 0.8,
            "expected_cost": 2.0, "target_metric": "combined",
            "falsification": "筛选阶段负收益 case 数未下降，或完整综合分不超过实现基线。",
            "base": "scratch",
        },
        {
            "based_on": "v0_softmax_aware_qk",
            "suffix": "paired_qk_error_cancellation",
            "focus": "attention",
            "family": "paired_qk_error_cancellation",
            "search_mode": "explore",
            "root_cause": "Q 与 K 独立编码产生的 ΔQKᵀ、QΔKᵀ 和 ΔQΔKᵀ 在中心化 logit 空间可能同向叠加；单侧张量误差或单侧 Hessian 无法主动利用两侧误差抵消。",
            "hypothesis": "联合选择成对的合法 Q/K 离散修复，使一阶和二阶 logit 误差在逐行去均值后相互抵消，可降低 Softmax 输入的有效扰动。",
            "structural_change": "显式计算中心化 ΔQKᵀ+QΔKᵀ+ΔQΔKᵀ，并用有界成对匹配替代独立 Q/K repair 接受，从误差相关性而非各自幅值选择编码。",
            "evidence": "v0_softmax_aware_qk 完整 300 例 Attention MSE 为 0.0003740420031498385，优于 v0_hessian_repair 的 0.0003786237067026048；精确 QK 展开说明剩余误差含独立选择未控制的交叉项。",
            "evidence_strength": 0.72, "novelty": 0.86, "uncertainty": 0.70,
            "expected_cost": 1.8, "target_metric": "attention_mse",
            "falsification": "成对候选接受率接近零，或完整评测中心化 logit 代理下降但 Attention MSE 不低于 v0_softmax_aware_qk。",
            "base": "based_on",
        },
        {
            "based_on": "v0_softmax_aware_qk",
            "suffix": "attention_weighted_value_transport",
            "focus": "attention",
            "family": "attention_weighted_value_transport",
            "search_mode": "explore",
            "root_cause": "V 的元素级量化误差只有经注意力矩阵 A 传播后的 AΔV 才影响输出；当前 V 编码没有区分 AᵀA 的高敏感方向与近零空间。",
            "hypothesis": "用校准注意力矩阵诱导的 AᵀA 度量选择合法 V 层级参数，把离散误差迁移到低可见方向，可降低最终 Attention 输出误差。",
            "structural_change": "从校准 Q/K 构造每头的注意力传输 Gram，并以 ||AΔV||² 而非 ||ΔV||² 驱动 V 的块级 scale/LV2/LV3 与 mant 候选选择，保留无收益回退。",
            "evidence": "Attention 输出对固定 Q/K 下的 V 误差精确等于 AΔV；现有最优 v0_softmax_aware_qk 只改善 Q/K 曲率，V 路径仍沿用通用量化。",
            "evidence_strength": 0.68, "novelty": 0.82, "uncertainty": 0.66,
            "expected_cost": 1.5, "target_metric": "attention_mse",
            "falsification": "AᵀA 加权候选与原编码等价或回退占绝大多数，或完整 Attention MSE/综合分不优于 v0_softmax_aware_qk。",
            "base": "based_on",
        },
        {
            "based_on": "v0_softmax_aware_qk",
            "suffix": "softmax_trust_region_repair",
            "focus": "attention",
            "family": "softmax_trust_region_repair",
            "search_mode": "exploit",
            "root_cause": "v0 的局部 Softmax 曲率已带来正式正收益，但离散 repair 可能越出二阶近似可靠区域；缺少按头的真实校准输出门禁来拒绝代理失真的更新。",
            "hypothesis": "保留已验证的 Softmax-aware 候选生成，在独立校准切分上用真实输出损失和中心化 logit 半径做按头信赖域接受，可稳定深化已有收益。",
            "structural_change": "在 v0_softmax_aware_qk 的 Hessian repair 外增加校准数据分割、按头局部信赖域和精确 Attention 输出接受/回退；不扩大候选数量，不改变固定格式。",
            "evidence": "v0_softmax_aware_qk 相对 v0_hessian_repair 将完整 Attention MSE 从 0.0003786237067026048 降至 0.0003740420031498385，证明 Softmax-aware Q/K 方向已有正式正证据。",
            "evidence_strength": 0.86, "novelty": 0.62, "uncertainty": 0.48,
            "expected_cost": 1.4, "target_metric": "attention_mse",
            "falsification": "真实门禁几乎总回退，或完整 Attention MSE/综合分未超过 v0_softmax_aware_qk，或额外校准耗时超过预算。",
            "base": "based_on",
        },
    ]
    added = 0
    for seed_item in seeds:
        registry = read_json(REGISTRY_PATH)
        if any(
            node.get("algorithm_family") == seed_item["family"]
            for node in registry.get("versions", {}).values()
        ):
            continue
        version = next_version(seed_item["suffix"], registry)
        add_algorithm_task(
            queue, config, based_on=seed_item["based_on"], version=version,
            focus=seed_item["focus"], family=seed_item["family"],
            hypothesis=seed_item["hypothesis"], base=seed_item["base"], priority=0.5,
            structural_change=seed_item["structural_change"],
            evidence=seed_item["evidence"],
            evidence_strength=seed_item["evidence_strength"],
            novelty=seed_item["novelty"], uncertainty=seed_item["uncertainty"],
            expected_cost=seed_item["expected_cost"],
            target_metric=seed_item["target_metric"],
            falsification=seed_item["falsification"],
            search_mode=seed_item["search_mode"],
            root_cause=seed_item["root_cause"],
        )
        added += 1
    rerank_queued_tasks(queue)
    return added


def backfill_completed(queue: dict[str, Any], config: dict[str, Any]) -> int:
    """Recover unused structural proposals from completed historical runs."""
    registry = read_json(REGISTRY_PATH)
    completed = sorted(
        (task for task in queue["tasks"] if task["status"] == "completed"),
        key=lambda task: -float(
            registry["versions"].get(task["version"], {}).get("metrics", {}).get("score", 0.0)
        ),
    )
    added = 0
    for task in completed:
        node = registry["versions"].get(task["version"], {})
        run_dir = RUNS / str(task.get("run_id"))
        feedback_path = run_dir / "report-feedback.json"
        legacy_path = run_dir / "worker-result.json"
        result_path = feedback_path if feedback_path.is_file() else legacy_path
        if not result_path.is_file() or not node.get("metrics"):
            continue
        payload = read_json(result_path)
        directions = payload if feedback_path.is_file() else payload.get("result") or {}
        added += enqueue_followups(
            queue, config, task, directions, float(node["metrics"]["score"])
        )
        registry = read_json(REGISTRY_PATH)
    return added


def recover_queue() -> int:
    """Requeue tasks left running by an interrupted scheduler."""
    queue = load_queue()
    registry = read_json(REGISTRY_PATH)
    recovered = 0
    for task in queue["tasks"]:
        task["search_mode"] = infer_search_mode(task)
        task["root_cause"] = task.get("root_cause") or task.get(
            "hypothesis", "unspecified root cause"
        )
        error_text = str(task.get("error") or "").lower()
        infrastructure_failure = any(marker in error_text for marker in (
            "winerror 2", "系统找不到指定的文件", "createprocesswithlogonw",
        ))
        interrupted_failure = task["status"] == "failed" and (
            not task.get("error") or infrastructure_failure
        )
        if task["status"] not in {"running", "environment_failed", "workflow_failed"} and not interrupted_failure:
            continue
        node = registry["versions"].get(task.get("version"), {})
        if node:
            node["search_mode"] = task.get("search_mode", infer_search_mode(task))
            node["root_cause"] = task.get("root_cause", task.get("hypothesis"))
        report = SOLUTION_ROOT / task["version"] / "report.md"
        if node.get("metrics") and report.is_file():
            task["status"] = "completed"
            task["stage"] = "completed"
        elif (
            task.get("run_id")
            and recoverable_record_checkpoint(
                RUNS / str(task["run_id"]), task["version"]
            )
        ):
            task.update(
                status="queued", stage="queued",
                resume_run_id=task["run_id"], resume_stage="recording",
                error="recovered at registry-recording checkpoint",
            )
            if node and node.get("metrics") is None:
                node["status"] = "draft"
                node.pop("failure", None)
        elif (
            task.get("run_id")
            and recoverable_report_checkpoint(RUNS / str(task["run_id"]))
        ):
            task.update(
                status="queued", stage="queued",
                resume_run_id=task["run_id"], resume_stage="reporting",
                error="recovered at report checkpoint",
            )
            if node and node.get("metrics") is None:
                node["status"] = "draft"
                node.pop("failure", None)
        elif (
            task.get("run_id")
            and recoverable_implementation_artifacts(
                RUNS / str(task["run_id"]), task["version"]
            )
        ):
            task.update(
                status="queued", stage="queued",
                resume_run_id=task["run_id"],
                resume_stage="implementation_finalize",
                error="recovered substantial implementation artifacts for finalization",
            )
            if node and node.get("metrics") is None:
                node["status"] = "draft"
                node.pop("failure", None)
        else:
            task.update(
                status="queued", stage="queued", run_id=None,
                resume_run_id=None, resume_stage=None,
                error="recovered after interrupted scheduler",
            )
            if node and node.get("metrics") is None:
                node["status"] = "draft"
                node.pop("failure", None)
        task["updated_at"] = now()
        recovered += 1
    if recovered:
        save_queue(queue)
        atomic_json(REGISTRY_PATH, registry)
    return recovered


def doctor(*, deep: bool = False) -> int:
    config = load_config()
    codex_ok, codex_detail = codex_preflight(config)
    evaluation_python_ok, evaluation_python_detail = evaluation_python_preflight(config)
    dataset_ok, dataset_detail = verify_dataset_manifest(
        ROOT,
        ROOT / config.get("evaluation_datasets_dir", "datasets/combined") / "manifest.json",
        deep=deep,
    )
    try:
        principles_ok = bool(load_principles()["principles"])
        experiments_ok = isinstance(load_experiments()["experiments"], list)
        process_metrics_ok = (
            PROCESS_METRICS_PATH.is_file()
            and read_json(PROCESS_METRICS_PATH).get("schema_version") == 1
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        principles_ok = experiments_ok = process_metrics_ok = False
    protected_ok = all((ROOT / path).is_file() for path in config["protected_files"])
    screening = config.get("screening", {})
    checks = {
        "codex": codex_ok,
        "evaluation_python": evaluation_python_ok,
        "archive": (ROOT / "reference" / "本地调试参考-0818.zip").is_file(),
        "flat_registry": REGISTRY_PATH.is_file(),
        "v0_hessian_repair": (SOLUTION_ROOT / "v0_hessian_repair" / "solution.py").is_file(),
        "max_agents": int(config["max_agents"]) <= 6,
        "search_mix_4_2": (
            int(config.get("search_mix", {}).get("explore_slots", -1)) == 4
            and int(config.get("search_mix", {}).get("exploit_slots", -1)) == 2
        ),
        "target_score": float(config.get("target_score", 0.0)) == 20000.0,
        "isolated_codex_config": bool(
            config.get("codex", {}).get("ignore_user_config")
            and config.get("codex", {}).get("ephemeral")
        ),
        "evaluation_dataset": (
            ROOT / config.get("evaluation_datasets_dir", "datasets/combined") / "linear.pt"
        ).is_file() and (
            ROOT / config.get("evaluation_datasets_dir", "datasets/combined") / "attn.pt"
        ).is_file(),
        "dataset_manifest": dataset_ok,
        "evaluation_cases": config["fixed_evaluation_cases"] == {"linear": 50, "attention": 250},
        "screening_profile": (
            screening.get("enabled") is True
            and screening.get("linear_cases") == 10
            and screening.get("attention_cases") == 50
            and 1 <= int(screening.get("promote_top_k", 0)) <= 3
        ),
        "lightweight_snapshots": all(
            name in config.get("isolation", {}).get("ignore", [])
            for name in ("datasets", "reference")
        ),
        "protected_boundary": protected_ok,
        "research_principles": principles_ok,
        "experiment_memory": experiments_ok,
        "process_metrics": process_metrics_ok,
    }
    for name, passed in checks.items():
        print(f"{name}: {'OK' if passed else 'FAILED'}")
    print(f"codex_detail: {codex_detail}")
    print(f"evaluation_python_detail: {evaluation_python_detail}")
    print(f"dataset_detail: {dataset_detail}")
    return 0 if all(checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="HiF4 asynchronous local-Agent runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("seed")
    status = sub.add_parser("status")
    status.add_argument("--json", action="store_true")
    status.add_argument("--explain", action="store_true")
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--json", action="store_true")
    sub.add_parser("pause")
    sub.add_parser("resume")
    doctor_parser = sub.add_parser("doctor")
    doctor_parser.add_argument("--deep", action="store_true")
    sub.add_parser("recover")
    sub.add_parser("backfill")
    run = sub.add_parser("run")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--once", action="store_true")
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("--based-on", required=True)
    enqueue.add_argument("--version", required=True)
    enqueue.add_argument("--focus", choices=("linear", "attention", "format", "combined"), required=True)
    enqueue.add_argument("--algorithm-family", required=True)
    enqueue.add_argument("--search-mode", choices=("explore", "exploit"))
    enqueue.add_argument("--hypothesis", required=True)
    enqueue.add_argument("--root-cause", default="")
    enqueue.add_argument("--implementation-base", choices=("based_on", "v0_hessian_repair", "scratch"), default="based_on")
    enqueue.add_argument("--structural-change", default="")
    enqueue.add_argument("--evidence", default="manual proposal")
    enqueue.add_argument("--evidence-strength", type=float, default=0.5)
    enqueue.add_argument("--novelty", type=float, default=0.5)
    enqueue.add_argument("--uncertainty", type=float, default=0.5)
    enqueue.add_argument("--expected-cost", type=float, default=1.0)
    enqueue.add_argument(
        "--target-metric",
        choices=("score", "linear_mse", "attention_mse", "runtime", "combined"),
        default="score",
    )
    enqueue.add_argument("--falsification", default="")
    enqueue.add_argument("--priority", type=float, default=0.5, help="bounded 0..1 manual hint")
    args = parser.parse_args()
    config = load_config()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.command == "init":
        try:
            with exclusive_scheduler_lock():
                stale_pid_removed = clean_stale_runner_pid()
                if not QUEUE_PATH.exists():
                    atomic_json(QUEUE_PATH, initial_queue())
                refresh_pareto_archive()
                refresh_process_metrics()
        except SchedulerBusyError:
            print("scheduler is running; init refused", file=sys.stderr)
            return 2
        print(
            f"initialized {QUEUE_PATH.relative_to(ROOT)} and research archives"
            + ("; removed stale runner PID" if stale_pid_removed else "")
        )
    elif args.command == "seed":
        try:
            with exclusive_scheduler_lock():
                queue = load_queue()
                added = seed(queue, config)
                save_queue(queue)
        except SchedulerBusyError:
            print("scheduler is running; seed refused", file=sys.stderr)
            return 2
        print(f"queued {added} new algorithm task(s); seed portfolio targets 4 explore + 2 exploit")
    elif args.command == "status":
        # Status is intentionally read-only: a concurrent CLI query must never
        # overwrite a scheduler update with an older queue snapshot.
        queue = copy.deepcopy(load_queue())
        rerank_queued_tasks(queue)
        if args.json:
            print(json.dumps(queue, ensure_ascii=False, indent=2))
        else:
            counts: dict[str, int] = {}
            for task in queue["tasks"]:
                counts[task["status"]] = counts.get(task["status"], 0) + 1
            print(" ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "empty")
            if RUNNER_PID_PATH.is_file():
                print("runner_pid=" + RUNNER_PID_PATH.read_text(encoding="utf-8").strip())
            if args.explain:
                experiments = load_experiments()["experiments"]
                print(f"stagnation_length={stagnation_length(experiments)}")
                pareto = (
                    read_json(PARETO_PATH)
                    if PARETO_PATH.is_file()
                    else build_pareto_archive(read_json(REGISTRY_PATH))
                )
                print("pareto=" + ",".join(item["version"] for item in pareto.get("front", [])))
                best_version, best_score = best_score_state(read_json(REGISTRY_PATH))
                target_score = float(config["target_score"])
                print(
                    f"target={target_score:.6f} best={best_score:.6f} "
                    f"best_version={best_version} "
                    f"gap={max(0.0, target_score - best_score):.6f}"
                )
                mode_counts: dict[str, dict[str, int]] = {}
                for task in queue["tasks"]:
                    if task["status"] not in {"queued", "running"}:
                        continue
                    mode = infer_search_mode(task)
                    mode_counts.setdefault(mode, {"queued": 0, "running": 0})[
                        task["status"]
                    ] += 1
                print("search_mix=" + json.dumps(mode_counts, ensure_ascii=False))
                pending = sorted(
                    (task for task in queue["tasks"] if task["status"] == "queued"),
                    key=lambda task: -float(task.get("priority", 0.0)),
                )
                for task in pending[:10]:
                    print(
                        f"{task['version']} priority={task['priority']:.6f} "
                        f"mode={infer_search_mode(task)} "
                        + json.dumps(task.get("priority_components", {}), ensure_ascii=False)
                    )
    elif args.command == "audit":
        # Like status, audit is a pure snapshot calculation.  The scheduler is
        # the sole writer of persistent process metrics during active runs.
        metrics = build_process_metrics(load_experiments()["experiments"])
        if args.json:
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        else:
            print(
                f"evaluated={metrics['evaluated_versions']} "
                f"positive_rate={metrics['positive_rate']:.3f} "
                f"new_bests={metrics['new_global_bests']} "
                f"stagnation={metrics['stagnation_length']}"
            )
            agreement = metrics["screening_winner_agreement_rate"]
            print(
                "screening_agreement="
                + ("insufficient_data" if agreement is None else f"{agreement:.3f}")
            )
            print(
                "positive_score_gain_per_eval_hour="
                f"{metrics['positive_score_gain_per_evaluation_hour']:.6f}"
            )
            print(
                "non_evaluation_outcomes="
                + json.dumps(metrics["non_evaluation_outcomes"], ensure_ascii=False)
            )
            print(
                "search_mode_metrics="
                + json.dumps(metrics.get("search_modes", {}), ensure_ascii=False)
            )
    elif args.command == "pause":
        STOP_PATH.write_text(now() + "\n", encoding="utf-8")
        print("dispatch paused; running atomic steps are allowed to finish")
    elif args.command == "resume":
        if STOP_PATH.exists():
            STOP_PATH.unlink()
        print("dispatch resumed")
    elif args.command == "doctor":
        return doctor(deep=args.deep)
    elif args.command == "recover":
        try:
            with exclusive_scheduler_lock():
                recovered = recover_queue()
        except SchedulerBusyError:
            print("scheduler is running; recover refused", file=sys.stderr)
            return 2
        print(f"recovered {recovered} interrupted task(s)")
    elif args.command == "backfill":
        try:
            with exclusive_scheduler_lock():
                queue = load_queue()
                added = backfill_completed(queue, config)
                if added:
                    save_queue(queue)
        except SchedulerBusyError:
            print("scheduler is running; backfill refused", file=sys.stderr)
            return 2
        print(f"backfilled {added} structural algorithm task(s)")
    elif args.command == "enqueue":
        try:
            with exclusive_scheduler_lock():
                queue = load_queue()
                task = add_algorithm_task(
                    queue, config, based_on=args.based_on, version=args.version,
                    focus=args.focus, family=args.algorithm_family,
                    hypothesis=args.hypothesis, base=args.implementation_base,
                    priority=args.priority,
                    structural_change=args.structural_change,
                    evidence=args.evidence,
                    evidence_strength=args.evidence_strength,
                    novelty=args.novelty,
                    uncertainty=args.uncertainty,
                    expected_cost=args.expected_cost,
                    target_metric=args.target_metric,
                    falsification=args.falsification,
                    search_mode=args.search_mode,
                    root_cause=args.root_cause,
                )
                save_queue(queue)
        except SchedulerBusyError:
            print("scheduler is running; enqueue refused", file=sys.stderr)
            return 2
        print(f"queued {task['task_id']} {task['version']}")
    elif args.command == "run":
        codex_ok, codex_detail = codex_preflight(config)
        if not codex_ok:
            print(f"Codex preflight failed; queue was not modified: {codex_detail}", file=sys.stderr)
            return 3
        evaluation_python_ok, evaluation_python_detail = evaluation_python_preflight(config)
        if not evaluation_python_ok:
            print(
                "Evaluation Python preflight failed; queue was not modified: "
                f"{evaluation_python_detail}",
                file=sys.stderr,
            )
            return 4
        dataset_manifest_path = (
            ROOT / config["evaluation_datasets_dir"] / "manifest.json"
        )
        dataset_ok, dataset_detail = verify_dataset_manifest(
            ROOT, dataset_manifest_path, deep=not args.dry_run
        )
        if not dataset_ok:
            print(
                f"Dataset integrity preflight failed; queue was not modified: {dataset_detail}",
                file=sys.stderr,
            )
            return 5
        config["_dataset_guard"] = dataset_file_signatures(dataset_manifest_path)
        print(f"Codex preflight: {codex_detail}")
        print(f"Evaluation Python preflight: {evaluation_python_detail}")
        print(f"Dataset integrity preflight: {dataset_detail}")
        try:
            with exclusive_scheduler_lock():
                RUNNER_PID_PATH.write_text(str(os.getpid()) + "\n", encoding="utf-8")
                try:
                    asyncio.run(Scheduler(config, dry_run=args.dry_run, once=args.once).loop())
                finally:
                    try:
                        if (
                            RUNNER_PID_PATH.is_file()
                            and RUNNER_PID_PATH.read_text(encoding="utf-8").strip() == str(os.getpid())
                        ):
                            RUNNER_PID_PATH.unlink()
                    except OSError:
                        pass
        except SchedulerBusyError:
            print("another scheduler is already running", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
