from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


VERSION_PATTERN = re.compile(r"^v(\d+)_(.+)$")
SOLUTION_PATTERN = re.compile(r"solution[\\/](v\d+_[^\\/]+)[\\/]", re.IGNORECASE)


@dataclass(frozen=True)
class EvaluationRecord:
    version: str
    profile: str
    config: str
    score: float
    linear_mse: float | None
    attention_mse: float | None
    seconds: float | None
    linear_cases: int | None
    attention_cases: int | None
    source: Path
    source_priority: int
    source_mtime_ns: int
    selected: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        return self.version, self.profile, self.config

    @property
    def evaluation_id(self) -> str:
        return f"{self.version}:{self.profile}:{self.config}"


def _read_json(path: Path, *, retries: int = 4) -> dict[str, Any]:
    """Read files that may be atomically replaced by the active Runner."""
    last_error: OSError | json.JSONDecodeError | None = None
    for attempt in range(retries):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"expected an object in {path}")
            return value
        except (OSError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(0.03 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _version_from_result(result: dict[str, Any]) -> str | None:
    source = str(result.get("solution_path") or result.get("path") or "")
    match = SOLUTION_PATTERN.search(source)
    return match.group(1) if match else None


def _profile_for(result: dict[str, Any], score: float) -> str:
    linear_cases = _optional_int(result.get("linear_case_count"))
    attention_cases = _optional_int(result.get("attention_case_count"))
    score_scale = _optional_int(result.get("score_scale_case_count"))
    if linear_cases == 50 and attention_cases == 250:
        return "F300"
    if linear_cases == 5 and attention_cases == 5:
        if score_scale == 300 or score > 100.0:
            return "F10"
        return "L10"
    total = (linear_cases or 0) + (attention_cases or 0)
    if score_scale == 300:
        return f"F{total or 'custom'}"
    return f"L{total or 'custom'}"


def _record_from_result(
    result: dict[str, Any], *, source: Path, source_priority: int,
    selected_config: str | None = None,
) -> EvaluationRecord | None:
    version = _version_from_result(result)
    score = _optional_float(result.get("total_score"))
    if version is None or score is None:
        return None
    config = str(result.get("config") or "main")
    linear = result.get("linear_output") or {}
    attention = result.get("attention_output") or {}
    return EvaluationRecord(
        version=version,
        profile=_profile_for(result, score),
        config=config,
        score=score,
        linear_mse=_optional_float(linear.get("mse")),
        attention_mse=_optional_float(attention.get("mse")),
        seconds=_optional_float(result.get("seconds")),
        linear_cases=_optional_int(result.get("linear_case_count")),
        attention_cases=_optional_int(result.get("attention_case_count")),
        source=source,
        source_priority=source_priority,
        source_mtime_ns=source.stat().st_mtime_ns,
        selected=(selected_config is not None and config == str(selected_config)),
    )


def _results_from_document(path: Path) -> list[EvaluationRecord]:
    document = _read_json(path)
    selected_config = document.get("selected")
    results: list[dict[str, Any]] = []
    formal = document.get("formal")
    if isinstance(formal, dict) and isinstance(formal.get("results"), list):
        results.extend(item for item in formal["results"] if isinstance(item, dict))
    elif isinstance(formal, list):
        results.extend(item for item in formal if isinstance(item, dict))
    if not results and isinstance(document.get("results"), list):
        results.extend(item for item in document["results"] if isinstance(item, dict))

    if path.name == "evaluation-summary.json":
        priority = 40
    elif path.name == "evaluation-formal.json":
        priority = 30
    elif path.name == "evaluation-main.json":
        priority = 20
    else:
        priority = 10
    records = [
        _record_from_result(
            result, source=path, source_priority=priority,
            selected_config=str(selected_config) if selected_config is not None else None,
        )
        for result in results
    ]
    valid = [record for record in records if record is not None]
    if len(valid) == 1 and selected_config is None:
        only = valid[0]
        valid[0] = EvaluationRecord(**{**only.__dict__, "selected": True})
    return valid


def _registry_record(
    version: str, node: dict[str, Any], registry_path: Path,
) -> EvaluationRecord | None:
    metrics = node.get("metrics") or {}
    score = _optional_float(metrics.get("score"))
    if score is None:
        return None
    result = {
        "path": f"solution/{version}/solution.py",
        "total_score": score,
        "linear_output": {"mse": metrics.get("linear_mse")},
        "attention_output": {"mse": metrics.get("attention_mse")},
        "seconds": metrics.get("seconds"),
        "linear_case_count": metrics.get("linear_case_count"),
        "attention_case_count": metrics.get("attention_case_count"),
        "score_scale_case_count": metrics.get("score_scale_case_count"),
        "config": node.get("selected_config") or "main",
    }
    return _record_from_result(
        result, source=registry_path, source_priority=10,
        selected_config=str(node.get("selected_config") or "main"),
    )


def _historical_registry_record(
    version: str, node: dict[str, Any], registry_path: Path,
) -> EvaluationRecord | None:
    metrics = node.get("historical_full_evaluation") or {}
    score = _optional_float(metrics.get("score"))
    if score is None:
        return None
    result = {
        "path": f"solution/{version}/solution.py",
        "total_score": score,
        "linear_output": {"mse": metrics.get("linear_mse")},
        "attention_output": {"mse": metrics.get("attention_mse")},
        "seconds": metrics.get("seconds"),
        "linear_case_count": metrics.get("linear_case_count"),
        "attention_case_count": metrics.get("attention_case_count"),
        "config": "main",
    }
    return _record_from_result(
        result, source=registry_path, source_priority=9, selected_config="main",
    )


def discover_evaluations(
    root: Path, registry: dict[str, Any],
) -> dict[tuple[str, str, str], EvaluationRecord]:
    """Merge registry and immutable run evidence without losing failed-report scores."""
    registry_path = root / ".agent" / "versions.json"
    candidates: list[EvaluationRecord] = []
    for version, node in registry.get("versions", {}).items():
        if not isinstance(node, dict):
            continue
        for record in (
            _registry_record(version, node, registry_path),
            _historical_registry_record(version, node, registry_path),
        ):
            if record is not None:
                candidates.append(record)

    runtime = root / ".agent" / "runtime"
    # The summary adds the selected config name to raw evaluation output.  Use
    # it as the canonical file for a run; only fall back to formal/main when a
    # workflow stopped before the summary was written.  Reading all three
    # would turn a named trial such as ``conservative`` into a second fake
    # ``main`` record.
    for run_dir in (runtime / "runs").glob("*"):
        if not run_dir.is_dir():
            continue
        for name in (
            "evaluation-summary.json",
            "evaluation-formal.json",
            "evaluation-main.json",
        ):
            path = run_dir / name
            if not path.is_file():
                continue
            try:
                run_records = _results_from_document(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if run_records:
                candidates.extend(run_records)
                break
    merged: dict[tuple[str, str, str], EvaluationRecord] = {}
    for record in candidates:
        previous = merged.get(record.key)
        rank = (record.source_priority, record.selected, record.source_mtime_ns)
        previous_rank = (
            previous.source_priority, previous.selected, previous.source_mtime_ns
        ) if previous else (-1, False, -1)
        if previous is None or rank > previous_rank:
            merged[record.key] = record
    # Partial evaluations are deliberately absent from the public score
    # ledger. They may be used transiently inside one version's configuration
    # screen, but they are neither scores nor historical ranking evidence.
    return {
        key: record for key, record in merged.items() if record.profile == "F300"
    }


def version_sort_key(version: str) -> tuple[int, int, str]:
    match = VERSION_PATTERN.match(version)
    if not match:
        return 10**9, 1, version
    number = int(match.group(1))
    v0_order = {
        "v0_hessian_repair": 0,
        "v0_softmax_aware_qk": 1,
        "v0_alternating_joint_fit": 2,
    }.get(version, 3)
    return number, v0_order, version


def choose_primary(
    version: str, node: dict[str, Any], records: Iterable[EvaluationRecord],
) -> EvaluationRecord | None:
    # Only a genuinely complete 50 + 250 evaluation is a version score.
    # Partial evaluations can never rank a version or enter the public ledger.
    options = [
        record for record in records
        if record.version == version and record.profile == "F300"
    ]
    if not options:
        return None
    selected_config = str(node.get("selected_config") or "")
    profile_order = {"F300": 3}
    return max(
        options,
        key=lambda record: (
            profile_order.get(record.profile, 1),
            record.selected or bool(selected_config and record.config == selected_config),
            record.source_priority,
            record.source_mtime_ns,
            record.score,
        ),
    )


def _relative_link(root: Path, source: Path, label: str) -> str:
    try:
        relative = source.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return f"`{source}`"
    return f"[{label}](../{relative})"


def _version_cell(root: Path, version: str, *, prefer_report: bool) -> str:
    version_dir = root / "solution" / version
    choices = ("report.md", "policy.md") if prefer_report else ("policy.md", "report.md")
    for name in choices:
        path = version_dir / name
        if path.is_file():
            return _relative_link(root, path, version)
    return f"`{version}`"


def _fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _fmt_mse(value: float | None) -> str:
    return "—" if value is None else f"{value:.6e}"


def _fmt_seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _review_failure_summary(error: str) -> str:
    mappings = {
        "structural_algorithm": "结构实现",
        "fixed_dequantization": "固定反量化",
        "evidence_chain": "证据链",
        "scope": "写入范围",
        "complexity_bound": "复杂度",
        "diagnostic_plan": "诊断",
    }
    checks = []
    for key in re.findall(r"check failed:\s*([a-z_]+)", error.lower()):
        label = mappings.get(key, key)
        if label not in checks:
            checks.append(label)
    return "、".join(checks) if checks else "结构审查"


def version_change_text(
    version: str, node: dict[str, Any], task: dict[str, Any] | None,
) -> str:
    """Return the explicit structural delta for every flat version."""
    value = str(
        node.get("structural_change")
        or (task or {}).get("structural_change")
        or ""
    ).strip()
    if not value:
        retained = {
            "v0_hessian_repair": (
                "固定根基线：保持 NVFP4 固定解码，使用 Hessian 感知的合法 HiF4 离散误差修复。"
            ),
            "v0_softmax_aware_qk": (
                "在根基线上加入 Softmax 感知的 Q/K 误差度量，使修复目标更贴近 Attention 输出。"
            ),
            "v0_alternating_joint_fit": (
                "在根基线上联合处理 Linear 激活与权重量化误差，采用交替拟合降低输出 MSE。"
            ),
        }
        value = retained.get(version, "账本尚未记录可核验的结构变化。")
    # Keep the generated Markdown table valid while preserving the full change
    # statement rather than replacing it with an opaque method tag.
    return " ".join(value.split()).replace("|", "\\|")


def status_text(
    node: dict[str, Any], task: dict[str, Any] | None,
    primary: EvaluationRecord | None,
) -> str:
    if task is None:
        status = str(node.get("status") or "unknown")
        return {
            "baseline": "固定根基线",
            "best": "当前最高分基准",
            "promising": "完成；有潜力",
            "rejected": "完成；未晋级",
            "retained": "保留",
            "linear_best": "保留；Linear 路线",
            "draft": "草稿",
        }.get(status, status)

    status = str(task.get("status") or node.get("status") or "unknown")
    stage = str(task.get("stage") or "")
    error = str(task.get("error") or node.get("failure") or "")
    if status == "running":
        stage_label = {
            "starting": "启动",
            "implementing": "实现",
            "structural_review": "结构审查",
            "evaluating": "正式评测",
            "reporting": "报告",
            "recording": "登记",
        }.get(stage, stage or "执行")
        return f"运行中：{stage_label}"
    if status == "queued":
        return "排队"
    if status == "completed":
        node_status = str(node.get("status") or "")
        if node_status == "promising":
            return "完成；有潜力"
        if node_status == "best":
            return "完成；最高分"
        return "完成；未晋级"
    if status == "pruned":
        return "已评测，后剪枝" if primary else "剪枝；未评测"
    if status == "failed":
        return f"审查失败：{_review_failure_summary(error)}"
    if status == "workflow_failed":
        prefix = "评测完成；" if primary else ""
        if "no events" in error.lower() or "timeout" in stage.lower():
            return prefix + "流程失败：Agent 空闲超时"
        if "report agent" in error.lower() or task.get("failed_stage") in {"reporting", "recording"}:
            return prefix + "流程失败：报告/登记"
        return prefix + "流程失败"
    if status == "environment_failed":
        return "环境失败；未计入算法"
    if status == "evaluation_timeout":
        return "评测超时；无有效分数"
    return status


def _counts_text(tasks: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for task in tasks:
        status = str(task.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return "、".join(f"`{key}={counts[key]}`" for key in sorted(counts)) or "无任务"


def _ranked_fixed_records(
    versions: dict[str, Any], records: dict[tuple[str, str, str], EvaluationRecord],
) -> list[EvaluationRecord]:
    ranked = []
    for version, node in versions.items():
        options = [
            record for record in records.values()
            if record.version == version and record.profile == "F300"
        ]
        if not options:
            continue
        ranked.append(choose_primary(version, node, options))
    return sorted((record for record in ranked if record is not None), key=lambda record: (-record.score, record.seconds or math.inf, record.version))


def _status_evidence(
    root: Path, task: dict[str, Any] | None, primary: EvaluationRecord | None,
) -> str:
    if primary is not None:
        return _relative_link(root, primary.source, "评测")
    if task is not None:
        return _relative_link(root, root / ".agent" / "runtime" / "queue.json", "队列")
    return _relative_link(root, root / ".agent" / "versions.json", "账本")


def render_score_summary(root: Path, *, generated_at: datetime | None = None) -> str:
    registry_path = root / ".agent" / "versions.json"
    queue_path = root / ".agent" / "runtime" / "queue.json"
    registry = _read_json(registry_path)
    queue = _read_json(queue_path)
    versions = registry.get("versions", {})
    tasks = queue.get("tasks", [])
    if not isinstance(versions, dict) or not isinstance(tasks, list):
        raise ValueError("invalid versions or queue document")
    task_by_version = {
        str(task.get("version")): task
        for task in tasks if isinstance(task, dict) and task.get("version")
    }
    records = discover_evaluations(root, registry)
    primary = {
        version: choose_primary(version, node, records.values())
        for version, node in versions.items() if isinstance(node, dict)
    }
    fixed = _ranked_fixed_records(versions, records)
    stamp = (generated_at or datetime.now().astimezone()).strftime("%Y-%m-%d %H:%M:%S %z")
    if len(stamp) >= 5:
        stamp = stamp[:-2] + ":" + stamp[-2:]

    lines = [
        "<!-- AUTO-GENERATED by .agent/scripts/update_score_summary.py; DO NOT EDIT BY HAND. -->",
        "# HiF4 算法迭代分数总账",
        "",
        f"> 自动更新时间：{stamp}",
        "> 每个算法版本在总账中只占一行；每次评测用 `version:profile:config` 唯一标识。",
        "",
        "## 1. 口径",
        "",
        "| 标记 | 样例与量纲 | 用途 |",
        "|---|---|---|",
        "| `F300` | 实际运行 50 个 Linear + 250 个 Attention，满分 30000 | 唯一正式分数与排名依据 |",
        "| `F60` | 10 个 Linear + 50 个 Attention | 只筛选同一版本的内部配置，不登记 |",
        "",
        "```text",
        "Score = 300 × (Linear 单例得分均值 + 5 × Attention 单例得分均值) / 6",
        "```",
        "",
        "`—` 表示没有正式分数，并不表示零分。平均 MSE 用于定位问题，不能单独反推出总分。",
        "",
        "## 2. 当前摘要",
        "",
    ]
    if fixed:
        best_score = fixed[0].score
        tied = [record for record in fixed if math.isclose(record.score, best_score, rel_tol=0.0, abs_tol=0.005)]
        best_names = "、".join(f"`{record.version}`" for record in tied)
        fastest_tie = min(tied, key=lambda record: record.seconds or math.inf)
        lines.extend([
            f"- 当前 `F300` 最高分：**{best_score:.2f}**，版本：{best_names}。",
            f"- 同分版本中耗时最低：`{fastest_tie.version}`，{_fmt_seconds(fastest_tie.seconds)} 秒。",
        ])
    scored_count = sum(record is not None for record in primary.values())
    missing_count = len(versions) - scored_count
    recovered = sorted(
        version for version, record in primary.items()
        if record is not None
        and not (versions[version].get("metrics") or {}).get("score")
    )
    lines.extend([
        f"- 当前版本名共 **{len(versions)}** 个；有主分数 **{scored_count}** 个，无正式分数 **{missing_count}** 个。",
        f"- Runner 快照：{_counts_text(tasks)}。",
    ])
    if recovered:
        lines.append(
            "- 从正式运行文件恢复、但尚未写入版本账本的分数："
            + "、".join(f"`{name}`" for name in recovered) + "。"
        )
    lines.extend([
        "",
        "## 3. 当前 F300 正式排名",
        "",
        "| 排名 | 评测 ID | based_on | 流程状态 | 分数 | 相对最高分 | Linear MSE | Attention MSE | 秒 | 证据 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    previous_score: float | None = None
    displayed_rank = 0
    best_score = fixed[0].score if fixed else 0.0
    for index, record in enumerate(fixed, start=1):
        if previous_score is None or not math.isclose(record.score, previous_score, rel_tol=0.0, abs_tol=0.005):
            displayed_rank = index
        previous_score = record.score
        node = versions[record.version]
        task = task_by_version.get(record.version)
        based_on = node.get("based_on") if isinstance(node, dict) else None
        delta = record.score - best_score
        lines.append(
            f"| {displayed_rank} | `{record.evaluation_id}` | "
            f"{f'`{based_on}`' if based_on else '—'} | {status_text(node, task, record)} | "
            f"{_fmt_score(record.score)} | {delta:+.2f} | {_fmt_mse(record.linear_mse)} | "
            f"{_fmt_mse(record.attention_mse)} | {_fmt_seconds(record.seconds)} | "
            f"{_relative_link(root, record.source, '评测')} |"
        )

    groups = (
        ("4.1 基线与 v1–v21", lambda number: number <= 21),
        ("4.2 v22–v39", lambda number: 22 <= number <= 39),
        ("4.3 v40 及以后", lambda number: number >= 40),
    )
    lines.extend([
        "",
        "## 4. 当前版本一一对应总账",
        "",
        "主分数只采用实际运行 50 Linear + 250 Attention 的 `F300`；任何部分评测都不得补位。",
    ])
    ordered_versions = sorted(versions, key=version_sort_key)
    for title, predicate in groups:
        subset = [
            version for version in ordered_versions
            if (match := VERSION_PATTERN.match(version)) and predicate(int(match.group(1)))
        ]
        if not subset:
            continue
        lines.extend([
            "",
            f"### {title}",
            "",
            "| 版本 | based_on | 本版变化 | 状态 | 档位 | 主分数 | Linear MSE | Attention MSE | 秒 | 证据 |",
            "|---|---|---|---|---|---:|---:|---:|---:|---|",
        ])
        for version in subset:
            node = versions[version]
            task = task_by_version.get(version)
            record = primary.get(version)
            based_on = node.get("based_on") if isinstance(node, dict) else None
            lines.append(
                f"| {_version_cell(root, version, prefer_report=record is not None)} | "
                f"{f'`{based_on}`' if based_on else '—'} | "
                f"{version_change_text(version, node, task)} | {status_text(node, task, record)} | "
                f"{record.profile if record else '—'} | {_fmt_score(record.score if record else None)} | "
                f"{_fmt_mse(record.linear_mse if record else None)} | "
                f"{_fmt_mse(record.attention_mse if record else None)} | "
                f"{_fmt_seconds(record.seconds if record else None)} | "
                f"{_status_evidence(root, task, record)} |"
            )

    primary_keys = {record.key for record in primary.values() if record is not None}
    supplements = sorted(
        (
            record for record in records.values()
            if record.version in versions
            and record.profile != "L10"
            and record.key not in primary_keys
        ),
        key=lambda record: (version_sort_key(record.version), record.profile, record.config),
    )
    lines.extend([
        "",
        "## 5. 同版本的补充评测",
        "",
        "这些记录使用不同档位或内部配置，不覆盖第 4 节的主分数。",
        "",
        "| 评测 ID | 分数 | Linear MSE | Attention MSE | 秒 | 证据 |",
        "|---|---:|---:|---:|---:|---|",
    ])
    if supplements:
        for record in supplements:
            lines.append(
                f"| `{record.evaluation_id}` | {_fmt_score(record.score)} | "
                f"{_fmt_mse(record.linear_mse)} | {_fmt_mse(record.attention_mse)} | "
                f"{_fmt_seconds(record.seconds)} | {_relative_link(root, record.source, '评测')} |"
            )
    else:
        lines.append("| — | — | — | — | — | 暂无 |")

    legacy = sorted(
        (record for record in records.values() if record.version not in versions),
        key=lambda record: (version_sort_key(record.version), record.profile, record.config),
    )
    lines.extend([
        "",
        "## 6. 重构前遗留评测",
        "",
        "旧命名可能与当前编号重复，因此统一加 `legacy:` 前缀；这里只保留完整 F300 结果。",
        "",
        "| 遗留评测 ID | 原始分数 | Linear MSE | Attention MSE | 秒 | 证据 |",
        "|---|---:|---:|---:|---:|---|",
    ])
    if legacy:
        for record in legacy:
            lines.append(
                f"| `legacy:{record.evaluation_id}` | {record.score:.6f} | "
                f"{_fmt_mse(record.linear_mse)} | {_fmt_mse(record.attention_mse)} | "
                f"{_fmt_seconds(record.seconds)} | {_relative_link(root, record.source, '评测')} |"
            )
    else:
        lines.append("| — | — | — | — | — | 暂无 |")

    lines.extend([
        "",
        "## 7. 自动更新",
        "",
        "```powershell",
        "# 单次重建",
        "python .agent/scripts/update_score_summary.py",
        "",
        "# 持续监听账本、队列和正式评测文件",
        "python .agent/scripts/update_score_summary.py --watch --interval 2",
        "```",
        "",
        "自动更新采用临时文件加原子替换；读取到 Runner 正在切换文件时会重试，不会把半份 JSON 写进报告。",
        "",
    ])
    return "\n".join(lines)


def write_score_summary(root: Path, output: Path | None = None) -> bool:
    target = output or root / "docs" / "迭代分数汇总.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_score_summary(root)
    try:
        if target.read_text(encoding="utf-8") == rendered:
            return False
    except FileNotFoundError:
        pass
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, target)
    return True


def source_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    paths = [
        root / ".agent" / "versions.json",
        root / ".agent" / "runtime" / "queue.json",
    ]
    runtime = root / ".agent" / "runtime"
    for pattern in (
        "runs/*/evaluation-summary.json",
        "runs/*/evaluation-formal.json",
        "runs/*/evaluation-main.json",
    ):
        paths.extend(runtime.glob(pattern))
    signature = []
    for path in sorted(set(paths)):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)
