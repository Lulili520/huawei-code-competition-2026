"""Formal single-path ablation: execute V51 with its new format path disabled."""

from pathlib import Path


_MAIN = Path(__file__).resolve().parents[2] / "solution.py"
exec(compile(_MAIN.read_text(encoding="utf-8"), str(_MAIN), "exec"), globals())
_FORMAT_PROJECTION_MODE = "off"
