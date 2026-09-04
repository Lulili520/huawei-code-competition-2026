from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
REGISTRY = ROOT / ".agent" / "versions.json"
SOLUTION = ROOT / "solution"
METHOD = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
FOCUS = ("linear", "attention", "format", "combined")
STATUS = ("draft", "evaluated", "promising", "rejected", "failed")


def load() -> dict[str, Any]:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if value.get("versioning") != "flat-sequential" or not isinstance(value.get("versions"), dict):
        raise ValueError("invalid flat version registry")
    return value


def save(value: dict[str, Any]) -> None:
    temporary = REGISTRY.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, REGISTRY)


def allocate(method: str, value: dict[str, Any]) -> str:
    if not METHOD.fullmatch(method):
        raise ValueError("method must be lowercase words joined by underscores")
    number = max((int(item["number"]) for item in value["versions"].values()), default=-1) + 1
    return f"v{number}_{method}"


def create(args: argparse.Namespace) -> None:
    value = load()
    if args.based_on not in value["versions"]:
        raise ValueError(f"unknown based_on version: {args.based_on}")
    name = allocate(args.method, value)
    target = SOLUTION / name
    if target.exists():
        raise FileExistsError(target)
    target.mkdir()
    try:
        if args.implementation_base == "based_on":
            shutil.copy2(SOLUTION / args.based_on / "solution.py", target / "solution.py")
        elif args.implementation_base == "v0_hessian_repair":
            shutil.copy2(SOLUTION / "v0_hessian_repair" / "solution.py", target / "solution.py")
        else:
            (target / "solution.py").write_text(
                '"""From-scratch HiF4 algorithm."""\n', encoding="utf-8"
            )
        (target / "policy.md").write_text(
            f"# {name} 策略\n\n"
            f"- 对比/实现参考：`{args.based_on}`\n"
            f"- 实现来源：`{args.implementation_base}`\n"
            f"- 研究假设：{args.hypothesis}\n",
            encoding="utf-8",
        )
        value["versions"][name] = {
            "number": int(name.split("_", 1)[0][1:]), "method": args.method,
            "based_on": args.based_on, "focus": args.focus,
            "hypothesis": args.hypothesis, "implementation_base": args.implementation_base,
            "status": "draft", "metrics": None,
        }
        save(value)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    print(name)


def record(args: argparse.Namespace) -> None:
    value = load()
    if args.name not in value["versions"]:
        raise ValueError(f"unknown version: {args.name}")
    value["versions"][args.name]["metrics"] = {
        "linear_mse": args.linear_mse, "attention_mse": args.attention_mse,
        "score": args.score,
    }
    value["versions"][args.name]["status"] = args.status
    save(value)


def show(ranked: bool) -> None:
    entries = list(load()["versions"].items())
    if ranked:
        entries = [item for item in entries if item[1].get("metrics")]
        entries.sort(key=lambda item: (-float(item[1]["metrics"]["score"]), item[1]["number"]))
    else:
        entries.sort(key=lambda item: item[1]["number"])
    for name, item in entries:
        score = "-" if not item.get("metrics") else f"{item['metrics']['score']:.12f}"
        print(f"{name} status={item['status']} score={score} based_on={item.get('based_on')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage flat sequential HiF4 versions")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create")
    make.add_argument("--based-on", required=True)
    make.add_argument("--method", required=True)
    make.add_argument("--focus", choices=FOCUS, required=True)
    make.add_argument("--hypothesis", required=True)
    make.add_argument(
        "--implementation-base", choices=("based_on", "v0_hessian_repair", "scratch"),
        default="based_on",
    )
    make.set_defaults(handler=create)
    result = commands.add_parser("record")
    result.add_argument("--name", required=True)
    result.add_argument("--linear-mse", type=float, required=True)
    result.add_argument("--attention-mse", type=float, required=True)
    result.add_argument("--score", type=float, required=True)
    result.add_argument("--status", choices=STATUS, default="evaluated")
    result.set_defaults(handler=record)
    listing = commands.add_parser("list")
    listing.set_defaults(handler=lambda _: show(False))
    queue = commands.add_parser("queue")
    queue.set_defaults(handler=lambda _: show(True))
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
