from __future__ import annotations

import argparse
import asyncio
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


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / ".agent"
CONFIG_PATH = AGENT_ROOT / "config.json"
TREE_PATH = AGENT_ROOT / "version-tree.json"
RUNTIME = AGENT_ROOT / "runtime"
QUEUE_PATH = RUNTIME / "queue.json"
RUNS = RUNTIME / "runs"
STOP_PATH = AGENT_ROOT / "STOP"
SCHEDULER_LOCK = RUNTIME / "scheduler.lock"
EVALUATION_LOCK = RUNTIME / "evaluation.lock"
SOLUTION_ROOT = ROOT / "solution"
PROMPTS = AGENT_ROOT / "prompts"
SCHEMAS = AGENT_ROOT / "schemas"
VERSION_RE = re.compile(r"^v([1-9][0-9]*)_([a-z0-9]+(?:_[a-z0-9]+)*)$")
PARAM_ONLY = re.compile(
    r"(?:^|_)(alpha|gain|threshold|coefficient|factor)(?:_|$)|阈值|系数|倍率|参数",
    re.IGNORECASE,
)
TERMINAL = {"completed", "failed", "cancelled"}


class EnvironmentLaunchError(RuntimeError):
    """Infrastructure failed before the quantization algorithm could run."""


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
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
    if not 1 <= int(config["max_hyperparameter_configs"]) <= 3:
        raise ValueError("max_hyperparameter_configs must be between 1 and 3")
    if not 1 <= int(config.get("environment_launch_retries", 1)) <= 3:
        raise ValueError("environment_launch_retries must be between 1 and 3")
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


def solution_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def policy_scaffold(task: dict[str, Any]) -> str:
    return f"""# {task['version']} 优化策略

## 父版本

名义父版本：`{task['parent']}`。先记录父版本官方 10 例的 Linear MSE、Attention MSE 和最终得分。

## 实现基础

- 实现方式：`{task['implementation_base']}`
- 实际代码来源、复用模块和重写模块：待说明

## 固定输入边界

NVFP4 反量化固定为 E2M1 值乘对应 E4M3 scale，每 16 个连续值共享一个 scale，再恢复原 shape。不得修改该规则。

## 问题分析

只写可定位的已验证事实：文件路径与函数/行号，或评测配置与精确指标。说明它影响 Linear、Attention 还是格式搜索。

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

待说明具体函数、数据流与结构性变化。

### 保持不变

固定 NVFP4 输入解析、公开接口和未涉及路径保持不变。

## 算法内部超参数计划

最多测试三组有理论依据的配置。说明参数含义、候选值和选择规则；不得把纯调参拆成新版本。

## 实施步骤

写成可执行顺序，包含边界检查与回退条件。

## 预期结果

只写可证伪的方向性假设，不得伪造数值。

## 验收标准

官方格式检查通过；比较两项 MSE 和最终得分；逐项判定假设为结果支持、结果否证或证据不足。
"""


def best_parent(tree: dict[str, Any], max_children: int) -> str:
    candidates: list[tuple[float, str]] = []
    for name, node in tree["nodes"].items():
        metrics = node.get("metrics")
        if (
            metrics
            and node.get("status") in {"baseline", "evaluated", "promising"}
            and active_child_count(tree, name) < max_children
        ):
            candidates.append((float(metrics["score"]), name))
    if not candidates:
        raise RuntimeError("no evaluated parent has a free algorithm-child slot")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def active_child_count(tree: dict[str, Any], parent: str) -> int:
    return sum(
        tree["nodes"].get(child, {}).get("status") not in {"failed", "cancelled"}
        for child in tree["nodes"][parent].get("children", [])
    )


def next_version(parent: str, suffix: str, tree: dict[str, Any]) -> str:
    parent_match = re.match(r"^v(\d+)_", parent)
    generation = (int(parent_match.group(1)) if parent_match else 0) + 1
    candidate = f"v{generation}_{suffix}"
    serial = 2
    while candidate in tree["nodes"] or (SOLUTION_ROOT / candidate).exists():
        candidate = f"v{generation}_{suffix}_{serial}"
        serial += 1
    return candidate


def reserve_algorithm(task: dict[str, Any], config: dict[str, Any]) -> None:
    tree = read_json(TREE_PATH)
    parent = task["parent"]
    if parent not in tree["nodes"]:
        raise ValueError(f"unknown parent: {parent}")
    if active_child_count(tree, parent) >= int(config["max_children"]):
        raise ValueError(f"parent {parent} has no free child slot")
    version = task["version"]
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid algorithm version name: {version}")
    if version in tree["nodes"] or (SOLUTION_ROOT / version).exists():
        raise FileExistsError(version)
    target = SOLUTION_ROOT / version
    target.mkdir(parents=False)
    base = task["implementation_base"]
    if base == "parent":
        shutil.copy2(SOLUTION_ROOT / parent / "solution.py", target / "solution.py")
    elif base == "v0":
        shutil.copy2(
            SOLUTION_ROOT / tree["root"] / "solution.py", target / "solution.py"
        )
    elif base == "scratch":
        (target / "solution.py").write_text(
            '"""From-scratch HiF4 algorithm; implemented by the assigned Agent."""\n',
            encoding="utf-8",
        )
    else:
        raise ValueError(f"unsupported implementation base: {base}")
    (target / "policy.md").write_text(policy_scaffold(task), encoding="utf-8")
    tree["nodes"][parent].setdefault("children", []).append(version)
    tree["nodes"][version] = {
        "parent": parent,
        "children": [],
        "focus": task["focus"],
        "algorithm_family": task["algorithm_family"],
        "implementation_base": base,
        "hypothesis": task["hypothesis"],
        "status": "draft",
        "metrics": None,
        "task_id": task["task_id"],
    }
    atomic_json(TREE_PATH, tree)


def add_algorithm_task(
    queue: dict[str, Any], config: dict[str, Any], *, parent: str, version: str,
    focus: str, family: str, hypothesis: str, base: str, priority: float,
) -> dict[str, Any]:
    if PARAM_ONLY.search(family) or PARAM_ONLY.search(version):
        raise ValueError("pure parameter variants are not valid algorithm versions")
    if any(
        task.get("algorithm_family") == family and task.get("parent") == parent
        and task.get("status") not in {"failed", "cancelled"}
        for task in queue["tasks"]
    ):
        raise ValueError(f"duplicate algorithm family for parent: {family}")
    task = {
        "task_id": uuid.uuid4().hex[:12],
        "kind": "algorithm",
        "version": version,
        "parent": parent,
        "focus": focus,
        "algorithm_family": family,
        "hypothesis": hypothesis,
        "implementation_base": base,
        "priority": priority,
        "status": "queued",
        "created_at": now(),
        "updated_at": now(),
        "run_id": None,
        "error": None,
    }
    reserve_algorithm(task, config)
    queue["tasks"].append(task)
    return task


def codex_argv(
    config: dict[str, Any], schema: Path, output: Path, workspace: Path,
    sandbox: str | None = None,
) -> list[str]:
    codex = config["codex"]
    argv = [
        codex["command"], "exec", "--json", "--color", "never",
        "--sandbox", sandbox or codex["sandbox"], "--cd", str(workspace),
        "--output-schema", str(schema), "--output-last-message", str(output), "-",
    ]
    if codex.get("model"):
        argv[2:2] = ["--model", codex["model"]]
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
) -> tuple[int, str | None, dict[str, Any] | None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    stderr_path = run_dir / "stderr.log"
    last_path = run_dir / "last-message.json"
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = None
    launch_error: OSError | None = None
    for attempt in range(1, int(config.get("environment_launch_retries", 1)) + 1):
        try:
            process = await asyncio.create_subprocess_exec(
                *codex_argv(config, schema, last_path, workspace or ROOT, sandbox),
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
    process.stdin.write(prompt.encode("utf-8"))
    await process.stdin.drain()
    process.stdin.close()
    session_id: str | None = None

    async def stdout_reader() -> None:
        nonlocal session_id
        with events_path.open("a", encoding="utf-8") as stream:
            while line := await process.stdout.readline():
                text = line.decode("utf-8", errors="replace")
                stream.write(text)
                stream.flush()
                try:
                    event = json.loads(text)
                    session_id = session_id or extract_session(event)
                except json.JSONDecodeError:
                    pass

    async def stderr_reader() -> None:
        with stderr_path.open("a", encoding="utf-8") as stream:
            while line := await process.stderr.readline():
                stream.write(line.decode("utf-8", errors="replace"))
                stream.flush()

    readers = [asyncio.create_task(stdout_reader()), asyncio.create_task(stderr_reader())]
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.terminate()
            await process.wait()
    await asyncio.gather(*readers, return_exceptions=True)
    result = None
    if last_path.is_file():
        try:
            result = json.loads(last_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = None
    return process.returncode or 0, session_id, result


def implementation_prompt(task: dict[str, Any]) -> str:
    base = (PROMPTS / "implement.md").read_text(encoding="utf-8")
    return base + "\n\n当前算法任务：\n" + json.dumps(task, ensure_ascii=False, indent=2)


def create_workspace(run_dir: Path) -> Path:
    """Give each Agent an isolated repository snapshot."""
    workspace = run_dir / "workspace"

    def ignored(directory: str, names: list[str]) -> set[str]:
        result = {name for name in names if name in {".git", "__pycache__"} or name.endswith(".pyc")}
        if Path(directory).resolve() == AGENT_ROOT.resolve() and "runtime" in names:
            result.add("runtime")
        return result

    shutil.copytree(ROOT, workspace, ignore=ignored)
    return workspace


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
        int(config["worker_timeout_seconds"]), workspace,
    )
    if code or not result or result.get("status") != "approved":
        reasons = result.get("reasons", []) if result else []
        raise RuntimeError("structural review rejected: " + "; ".join(reasons))


async def evaluate_candidates(
    config: dict[str, Any], task: dict[str, Any], run_dir: Path,
    evaluation_lock: asyncio.Lock,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    version_dir = SOLUTION_ROOT / task["version"]
    candidates = [("main", version_dir / "solution.py")]
    for path in sorted((version_dir / "trials").glob("*/solution.py")) if (version_dir / "trials").is_dir() else []:
        candidates.append((path.parent.name, path))
    limit = int(config["max_hyperparameter_configs"])
    if len(candidates) > limit:
        raise ValueError(f"algorithm produced {len(candidates)} configurations; maximum is {limit}")
    results: list[dict[str, Any]] = []
    async with evaluation_lock:
        with file_lock(EVALUATION_LOCK):
            for label, path in candidates:
                output = run_dir / f"evaluation-{label}.json"
                command = [
                    config.get("evaluation_python", sys.executable),
                    str(AGENT_ROOT / "skills/hif4-evaluate/scripts/evaluate.py"),
                    str(path), "--json-output", str(output),
                ]
                try:
                    completed = await asyncio.to_thread(
                        subprocess.run, command, cwd=ROOT, capture_output=True, text=True,
                        encoding="utf-8", errors="replace",
                        timeout=int(config["evaluation_timeout_seconds"]),
                        env=utf8_environment(),
                    )
                except FileNotFoundError as error:
                    raise EnvironmentLaunchError(
                        f"official evaluator could not start environment Python: {error}"
                    ) from error
                (run_dir / f"evaluation-{label}.log").write_text(
                    completed.stdout + "\n" + completed.stderr, encoding="utf-8"
                )
                if completed.returncode or not output.is_file():
                    raise RuntimeError(f"evaluation failed for {label}")
                result = read_json(output)["results"][0]
                result["config"] = label
                result["solution_path"] = str(path.relative_to(ROOT))
                result["solution_sha256"] = solution_hash(path)
                results.append(result)
    selected = max(results, key=lambda item: (float(item["total_score"]), -float(item["linear_output"]["mse"]), -float(item["attention_output"]["mse"])))
    selected_path = ROOT / selected["solution_path"]
    if selected_path != version_dir / "solution.py":
        shutil.copy2(selected_path, version_dir / "solution.py")
        selected["solution_sha256"] = solution_hash(version_dir / "solution.py")
    atomic_json(run_dir / "evaluation-summary.json", {
        "schema_version": 1, "selected": selected["config"], "results": results,
    })
    return selected, results


async def write_report(
    config: dict[str, Any], task: dict[str, Any], run_dir: Path,
    selected: dict[str, Any], trials: list[dict[str, Any]], workspace: Path,
) -> None:
    prompt = (PROMPTS / "report.md").read_text(encoding="utf-8")
    prompt += "\n\n版本任务：\n" + json.dumps(task, ensure_ascii=False, indent=2)
    prompt += "\n\n真实评测结果：\n" + json.dumps(
        {"selected": selected, "all_configs": trials}, ensure_ascii=False, indent=2
    )
    # Report output itself is prose, so use a minimal JSON wrapper schema dynamically.
    report_schema = run_dir / "report-result.schema.json"
    atomic_json(report_schema, {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object", "additionalProperties": False,
        "required": ["status", "report_path"],
        "properties": {
            "status": {"type": "string", "enum": ["written", "failed"]},
            "report_path": {"type": "string"},
        },
    })
    code, _, result = await run_codex(
        config, run_dir / "report", prompt, report_schema,
        int(config["worker_timeout_seconds"]), workspace,
    )
    report = workspace / "solution" / task["version"] / "report.md"
    if code or not result or result.get("status") != "written" or not report.is_file():
        raise RuntimeError("report agent did not produce report.md")
    import_version(workspace, task["version"], report_only=True)


def record_completed(task: dict[str, Any], selected: dict[str, Any], run_id: str) -> str:
    tree = read_json(TREE_PATH)
    node = tree["nodes"][task["version"]]
    node["metrics"] = {
        "linear_mse": selected["linear_output"]["mse"],
        "attention_mse": selected["attention_output"]["mse"],
        "score": selected["total_score"],
    }
    parent_score = float(tree["nodes"][task["parent"]]["metrics"]["score"])
    node["status"] = "promising" if float(selected["total_score"]) > parent_score else "rejected"
    node["solution_sha256"] = selected["solution_sha256"]
    node["evaluation_run"] = run_id
    atomic_json(TREE_PATH, tree)
    return node["status"]


def enqueue_followups(
    queue: dict[str, Any], config: dict[str, Any], task: dict[str, Any],
    result: dict[str, Any], score: float,
) -> int:
    """Turn a successful Agent's structural directions into child tasks."""
    added = 0
    for proposal in result.get("next_algorithms", [])[: int(config["max_children"])]:
        structural = str(proposal.get("structural_change", "")).strip()
        evidence = str(proposal.get("evidence", "")).strip()
        family = str(proposal.get("algorithm_family", ""))
        if len(structural) < 20 or len(evidence) < 10 or PARAM_ONLY.search(family):
            continue
        tree = read_json(TREE_PATH)
        if active_child_count(tree, task["version"]) >= int(config["max_children"]):
            break
        version = next_version(task["version"], proposal["version_suffix"], tree)
        try:
            add_algorithm_task(
                queue, config, parent=task["version"], version=version,
                focus=proposal["focus"], family=family,
                hypothesis=proposal["hypothesis"],
                base=proposal["implementation_base"], priority=score,
            )
        except (ValueError, FileExistsError):
            continue
        added += 1
    return added


def mark_failed_version(task: dict[str, Any], error: str) -> None:
    if task.get("kind") != "algorithm" or not task.get("version"):
        return
    tree = read_json(TREE_PATH)
    node = tree.get("nodes", {}).get(task["version"])
    if node and node.get("metrics") is None:
        node["status"] = "failed"
        node["failure"] = error
        atomic_json(TREE_PATH, tree)


class Scheduler:
    def __init__(self, config: dict[str, Any], *, dry_run: bool, once: bool) -> None:
        self.config = config
        self.dry_run = dry_run
        self.once = once
        self.queue_lock = asyncio.Lock()
        self.evaluation_lock = asyncio.Lock()
        self.running: dict[str, asyncio.Task[None]] = {}

    async def update_task(self, task_id: str, **changes: Any) -> None:
        async with self.queue_lock:
            queue = load_queue()
            task = next(item for item in queue["tasks"] if item["task_id"] == task_id)
            task.update(changes)
            task["updated_at"] = now()
            save_queue(queue)

    async def run_algorithm(self, task: dict[str, Any], run_dir: Path) -> None:
        workspace = await asyncio.to_thread(create_workspace, run_dir)
        code, session, result = await run_codex(
            self.config, run_dir, implementation_prompt(task),
            SCHEMAS / "worker-result.schema.json",
            int(self.config["worker_timeout_seconds"]), workspace,
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
        await structural_review(self.config, task, run_dir, workspace)
        import_version(workspace, task["version"])
        selected, trials = await evaluate_candidates(
            self.config, task, run_dir, self.evaluation_lock
        )
        shutil.copy2(
            SOLUTION_ROOT / task["version"] / "solution.py",
            workspace / "solution" / task["version"] / "solution.py",
        )
        await write_report(self.config, task, run_dir, selected, trials, workspace)
        # Tree update and child creation are one serialized state transition.
        # Otherwise two workers finishing together can overwrite each other's tree data.
        async with self.queue_lock:
            record_completed(task, selected, run_dir.name)
            queue = load_queue()
            enqueue_followups(
                queue, self.config, task, result, float(selected["total_score"])
            )
            save_queue(queue)

    async def worker(self, task: dict[str, Any]) -> None:
        run_id = f"{task['task_id']}-{int(time.time())}"
        run_dir = RUNS / run_id
        atomic_json(run_dir / "run.json", {
            "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
            "kind": task["kind"], "state": "running", "started_at": now(),
        })
        await self.update_task(task["task_id"], status="running", run_id=run_id)
        try:
            await self.run_algorithm(task, run_dir)
            await self.update_task(task["task_id"], status="completed")
            atomic_json(run_dir / "run.json", {
                "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
                "kind": task["kind"], "state": "completed", "finished_at": now(),
            })
        except asyncio.CancelledError:
            # Scheduler shutdown is not evidence against the algorithm. Leave the
            # draft intact and make the complete task eligible for a clean retry.
            await self.update_task(
                task["task_id"], status="queued", run_id=None,
                error="requeued after scheduler interruption",
            )
            atomic_json(run_dir / "run.json", {
                "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
                "kind": task["kind"], "state": "interrupted",
                "finished_at": now(),
            })
            raise
        except EnvironmentLaunchError as error:
            # Infrastructure failure is not evidence against the algorithm.
            await self.update_task(
                task["task_id"], status="environment_failed", error=str(error)
            )
            atomic_json(run_dir / "run.json", {
                "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
                "kind": task["kind"], "state": "environment_failed",
                "finished_at": now(), "error": str(error),
            })
        except BaseException as error:
            async with self.queue_lock:
                mark_failed_version(task, str(error))
                queue = load_queue()
                queued_task = next(
                    item for item in queue["tasks"] if item["task_id"] == task["task_id"]
                )
                queued_task.update(status="failed", error=str(error), updated_at=now())
                save_queue(queue)
            atomic_json(run_dir / "run.json", {
                "schema_version": 1, "run_id": run_id, "task_id": task["task_id"],
                "kind": task["kind"], "state": "failed", "finished_at": now(),
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
            queue = load_queue()
            pending = sorted(
                (task for task in queue["tasks"] if task["status"] == "queued"),
                key=lambda task: (-float(task.get("priority", 0.0)), task["created_at"]),
            )
            dry_slot = len(self.running)
            while pending and len(self.running) < max_agents:
                task = pending.pop(0)
                if self.dry_run:
                    dry_slot += 1
                    print(f"DRY-RUN slot={dry_slot} {task['kind']} {task['task_id']} {task.get('version','')}")
                    await self.update_task(task["task_id"], status="dry_run")
                    continue
                running = asyncio.create_task(self.worker(task))
                self.running[task["task_id"]] = running
            if self.once and not self.running:
                return
            # Empty capacity deliberately waits for valid child directions from
            # completed algorithm tasks. The scheduler never fabricates work.
            await asyncio.sleep(0.5)


def seed(queue: dict[str, Any], config: dict[str, Any]) -> None:
    tree = read_json(TREE_PATH)
    root = tree["root"]
    priority = float(tree["nodes"][root]["metrics"]["score"])
    seeds = [
        ("v1_qk_spectral_basis", "attention", "orthogonal_qk_basis",
         "构建 Q/K 公共正交谱基底，在保持浮点点积不变的同时重排 HiF4 分块能量", "parent"),
        ("v1_softmax_aware_qk", "attention", "softmax_aware_qk",
         "利用 softmax 平移不变性与校准输出误差设计 Q/K 量化策略", "parent"),
        ("v1_output_aware_linear", "linear", "output_aware_linear",
         "使用校准矩阵的输出误差设计 Linear 量化与误差修复策略", "parent"),
    ]
    for version, focus, family, hypothesis, base in seeds:
        add_algorithm_task(
            queue, config, parent=root, version=version, focus=focus,
            family=family, hypothesis=hypothesis, base=base, priority=priority,
        )


def backfill_completed(queue: dict[str, Any], config: dict[str, Any]) -> int:
    """Recover unused structural proposals from completed historical runs."""
    tree = read_json(TREE_PATH)
    completed = sorted(
        (task for task in queue["tasks"] if task["status"] == "completed"),
        key=lambda task: -float(
            tree["nodes"].get(task["version"], {}).get("metrics", {}).get("score", 0.0)
        ),
    )
    added = 0
    for task in completed:
        node = tree["nodes"].get(task["version"], {})
        if active_child_count(tree, task["version"]) >= int(config["max_children"]):
            continue
        result_path = RUNS / str(task.get("run_id")) / "worker-result.json"
        if not result_path.is_file() or not node.get("metrics"):
            continue
        worker = read_json(result_path).get("result") or {}
        added += enqueue_followups(
            queue, config, task, worker, float(node["metrics"]["score"])
        )
        tree = read_json(TREE_PATH)
    return added


def recover_queue() -> int:
    """Requeue tasks left running by an interrupted scheduler."""
    queue = load_queue()
    tree = read_json(TREE_PATH)
    recovered = 0
    for task in queue["tasks"]:
        error_text = str(task.get("error") or "").lower()
        infrastructure_failure = any(marker in error_text for marker in (
            "winerror 2", "系统找不到指定的文件", "createprocesswithlogonw",
        ))
        interrupted_failure = task["status"] == "failed" and (
            not task.get("error") or infrastructure_failure
        )
        if task["status"] not in {"running", "environment_failed"} and not interrupted_failure:
            continue
        node = tree["nodes"].get(task.get("version"), {})
        report = SOLUTION_ROOT / task["version"] / "report.md"
        if node.get("metrics") and report.is_file():
            task["status"] = "completed"
        else:
            task.update(status="queued", run_id=None, error="recovered after interrupted scheduler")
            if node and node.get("metrics") is None:
                node["status"] = "draft"
                node.pop("failure", None)
        task["updated_at"] = now()
        recovered += 1
    if recovered:
        save_queue(queue)
        atomic_json(TREE_PATH, tree)
    return recovered


def doctor() -> int:
    config = load_config()
    codex_ok, codex_detail = codex_preflight(config)
    evaluation_python_ok, evaluation_python_detail = evaluation_python_preflight(config)
    checks = {
        "codex": codex_ok,
        "evaluation_python": evaluation_python_ok,
        "archive": (ROOT / "reference" / "本地调试参考-0818.zip").is_file(),
        "tree": TREE_PATH.is_file(),
        "v0": (SOLUTION_ROOT / "v0_hessian_repair" / "solution.py").is_file(),
        "max_agents": int(config["max_agents"]) <= 6,
        "official_cases": config["fixed_evaluation_cases"] == {"linear": 5, "attention": 5},
    }
    for name, passed in checks.items():
        print(f"{name}: {'OK' if passed else 'FAILED'}")
    print(f"codex_detail: {codex_detail}")
    print(f"evaluation_python_detail: {evaluation_python_detail}")
    return 0 if all(checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="HiF4 asynchronous local-Agent runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("seed")
    status = sub.add_parser("status")
    status.add_argument("--json", action="store_true")
    sub.add_parser("pause")
    sub.add_parser("resume")
    sub.add_parser("doctor")
    sub.add_parser("recover")
    sub.add_parser("backfill")
    run = sub.add_parser("run")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--once", action="store_true")
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("--parent", required=True)
    enqueue.add_argument("--version", required=True)
    enqueue.add_argument("--focus", choices=("linear", "attention", "format", "combined"), required=True)
    enqueue.add_argument("--algorithm-family", required=True)
    enqueue.add_argument("--hypothesis", required=True)
    enqueue.add_argument("--implementation-base", choices=("parent", "v0", "scratch"), default="parent")
    enqueue.add_argument("--priority", type=float, default=1.0)
    args = parser.parse_args()
    config = load_config()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.command == "init":
        if not QUEUE_PATH.exists():
            atomic_json(QUEUE_PATH, initial_queue())
        print(f"initialized {QUEUE_PATH.relative_to(ROOT)}")
    elif args.command == "seed":
        queue = load_queue()
        seed(queue, config)
        save_queue(queue)
        print("queued 3 complete algorithm tasks; 3 slots remain idle")
    elif args.command == "status":
        queue = load_queue()
        if args.json:
            print(json.dumps(queue, ensure_ascii=False, indent=2))
        else:
            counts: dict[str, int] = {}
            for task in queue["tasks"]:
                counts[task["status"]] = counts.get(task["status"], 0) + 1
            print(" ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "empty")
    elif args.command == "pause":
        STOP_PATH.write_text(now() + "\n", encoding="utf-8")
        print("dispatch paused; running atomic steps are allowed to finish")
    elif args.command == "resume":
        if STOP_PATH.exists():
            STOP_PATH.unlink()
        print("dispatch resumed")
    elif args.command == "doctor":
        return doctor()
    elif args.command == "recover":
        print(f"recovered {recover_queue()} interrupted task(s)")
    elif args.command == "backfill":
        queue = load_queue()
        added = backfill_completed(queue, config)
        if added:
            save_queue(queue)
        print(f"backfilled {added} structural algorithm task(s)")
    elif args.command == "enqueue":
        queue = load_queue()
        task = add_algorithm_task(
            queue, config, parent=args.parent, version=args.version,
            focus=args.focus, family=args.algorithm_family,
            hypothesis=args.hypothesis, base=args.implementation_base,
            priority=args.priority,
        )
        save_queue(queue)
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
        print(f"Codex preflight: {codex_detail}")
        print(f"Evaluation Python preflight: {evaluation_python_detail}")
        try:
            with file_lock(SCHEDULER_LOCK, blocking=False):
                asyncio.run(Scheduler(config, dry_run=args.dry_run, once=args.once).loop())
        except OSError:
            print("another scheduler is already running", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
