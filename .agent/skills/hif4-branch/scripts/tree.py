from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
TREE_PATH = ROOT / ".agent" / "version-tree.json"
SOLUTION_ROOT = ROOT / "solution"
NAME_PATTERN = re.compile(r"^v[1-9][0-9]*_[a-z0-9]+(?:_[a-z0-9]+)*$")
FOCUS_VALUES = ("linear", "attention", "format", "combined")
STATUS_VALUES = ("draft", "evaluated", "promising", "rejected")
MAX_CHILDREN = 3


def load_tree() -> dict[str, Any]:
    data = json.loads(TREE_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("nodes"), dict):
        raise ValueError("Unsupported or invalid version-tree.json")
    return data


def save_tree(data: dict[str, Any]) -> None:
    TREE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def policy_scaffold(name: str, parent: str, hypothesis: str) -> str:
    return f"""# {name} 优化策略

## 父版本

`{parent}`。待从父版本 report 填入 Linear MSE、Attention MSE 和最终得分。

## 问题分析

待结合 `{parent}` 的 report、代码和评测结果，说明一个具体误差来源。

## 相关方案调研

待比较与问题直接相关的论文、官方资料或现有方法，并说明对本题的适用性。

## 理论分析

待建立修改方向与最终输出 MSE 之间的理论联系，并解释术语和公式。

## 选定修改方案

### 修改目标

该版本从 `{parent}` 分支，只验证这一主要假设：{hypothesis}。

### 修改范围

待明确需要修改的函数、参数和数据流。

### 保持不变

待明确为公平归因而不修改的模块。

## 实施步骤

待明确修改函数、参数候选和保持不变的对照部分。

## 预期结果

待写目标 MSE 的方向性变化，不得伪造测试结果。

## 验收标准

官方格式检查必须通过；目标 MSE 和最终得分必须与父版本比较。
"""


def create_branch(args: argparse.Namespace) -> None:
    data = load_tree()
    nodes = data["nodes"]
    if args.parent not in nodes:
        raise ValueError(f"Unknown parent: {args.parent}")
    if not NAME_PATTERN.fullmatch(args.name):
        raise ValueError(
            "name must match v<number>_<short_change_name>, for example "
            "v2_k_hessian or v3_scale_search"
        )
    if args.name in nodes or (SOLUTION_ROOT / args.name).exists():
        raise FileExistsError(f"Version already exists: {args.name}")
    if len(nodes[args.parent].get("children", [])) >= MAX_CHILDREN:
        raise ValueError(
            f"Parent {args.parent} already has the maximum {MAX_CHILDREN} children"
        )
    parent_solution = SOLUTION_ROOT / args.parent / "solution.py"
    if not parent_solution.is_file():
        raise FileNotFoundError(parent_solution)

    target = SOLUTION_ROOT / args.name
    target.mkdir()
    try:
        shutil.copy2(parent_solution, target / "solution.py")
        (target / "policy.md").write_text(
            policy_scaffold(args.name, args.parent, args.hypothesis), encoding="utf-8"
        )
        nodes[args.parent]["children"].append(args.name)
        nodes[args.name] = {
            "parent": args.parent,
            "children": [],
            "focus": args.focus,
            "hypothesis": args.hypothesis,
            "status": "draft",
            "metrics": None,
        }
        save_tree(data)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    print(f"Created {args.name} from {args.parent}")


def record_result(args: argparse.Namespace) -> None:
    data = load_tree()
    nodes = data["nodes"]
    if args.name not in nodes:
        raise ValueError(f"Unknown version: {args.name}")
    if not (SOLUTION_ROOT / args.name / "solution.py").is_file():
        raise FileNotFoundError(f"Missing solution/{args.name}/solution.py")
    nodes[args.name]["metrics"] = {
        "linear_mse": args.linear_mse,
        "attention_mse": args.attention_mse,
        "score": args.score,
    }
    nodes[args.name]["status"] = args.status
    save_tree(data)
    print(f"Recorded {args.name}: score={args.score:.6f}")


def show_tree() -> None:
    data = load_tree()
    nodes = data["nodes"]

    def visit(name: str, prefix: str, last: bool, root: bool = False) -> None:
        node = nodes[name]
        metrics = node.get("metrics")
        score = "-" if not metrics else f"{metrics['score']:.6f}"
        connector = "" if root else ("+- " if last else "|- ")
        print(f"{prefix}{connector}{name} [{node['focus']}/{node['status']}] score={score}")
        children = node.get("children", [])
        child_prefix = prefix + ("   " if root or last else "|  ")
        for index, child in enumerate(children):
            visit(child, child_prefix, index == len(children) - 1)

    visit(data["root"], "", True, root=True)


def show_queue() -> None:
    data = load_tree()
    candidates = []
    for name, node in data["nodes"].items():
        metrics = node.get("metrics")
        if (
            metrics
            and node.get("status") in {"baseline", "evaluated", "promising"}
            and len(node.get("children", [])) < MAX_CHILDREN
        ):
            candidates.append((float(metrics["score"]), name, node))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    for rank, (score, name, node) in enumerate(candidates, 1):
        slots = MAX_CHILDREN - len(node.get("children", []))
        print(
            f"{rank}. {name} score={score:.6f} "
            f"focus={node['focus']} child_slots={slots}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the HiF4 solution tree")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--parent", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--focus", choices=FOCUS_VALUES, required=True)
    create.add_argument("--hypothesis", required=True)
    create.set_defaults(handler=create_branch)

    record = subparsers.add_parser("record")
    record.add_argument("--name", required=True)
    record.add_argument("--linear-mse", type=float, required=True)
    record.add_argument("--attention-mse", type=float, required=True)
    record.add_argument("--score", type=float, required=True)
    record.add_argument("--status", choices=STATUS_VALUES, default="evaluated")
    record.set_defaults(handler=record_result)

    show = subparsers.add_parser("show")
    show.set_defaults(handler=lambda _: show_tree())

    queue = subparsers.add_parser("queue")
    queue.set_defaults(handler=lambda _: show_queue())

    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
