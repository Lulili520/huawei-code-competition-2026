from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = AGENT_ROOT.parent
sys.path.insert(0, str(AGENT_ROOT))

from core.score_summary import source_signature, write_score_summary  # noqa: E402


PID_PATH = AGENT_ROOT / "runtime" / "score-summary.pid"


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def claim_watcher() -> None:
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PID_PATH.is_file():
        try:
            existing = int(PID_PATH.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            existing = -1
        if process_exists(existing):
            raise RuntimeError(f"score summary watcher is already running: pid={existing}")
        PID_PATH.unlink(missing_ok=True)
    descriptor = os.open(PID_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()) + "\n")


def release_watcher() -> None:
    try:
        if PID_PATH.read_text(encoding="utf-8").strip() == str(os.getpid()):
            PID_PATH.unlink()
    except OSError:
        pass


def update_once(output: Path | None) -> bool:
    changed = write_score_summary(ROOT, output)
    print(("updated " if changed else "already current ") + str(output or ROOT / "docs" / "迭代分数汇总.md"), flush=True)
    return changed


def watch(output: Path | None, interval: float) -> int:
    claim_watcher()
    previous: tuple[tuple[str, int, int], ...] | None = None
    try:
        while True:
            current = source_signature(ROOT)
            if current != previous:
                try:
                    update_once(output)
                except (OSError, ValueError) as error:
                    print(f"score summary update deferred: {error}", file=sys.stderr, flush=True)
                    time.sleep(interval)
                    continue
                previous = current
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    finally:
        release_watcher()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the live HiF4 version score ledger")
    parser.add_argument("--watch", action="store_true", help="refresh whenever source files change")
    parser.add_argument("--interval", type=float, default=2.0, help="watch polling interval in seconds")
    parser.add_argument("--output", type=Path, help="override the Markdown output path")
    args = parser.parse_args()
    if args.interval < 0.5:
        parser.error("--interval must be at least 0.5 seconds")
    if args.watch:
        return watch(args.output, args.interval)
    update_once(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
