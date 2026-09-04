from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "reference" / "本地调试参考-0818" / "example" / "mini_sample"
COVERAGE = ROOT / "reference" / "test_sample" / "expanded"
OUTPUT = ROOT / "datasets" / "combined"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_list(path: Path) -> list:
    value = torch.load(path, weights_only=True)
    if not isinstance(value, list):
        raise TypeError(f"expected list in {path}")
    return value


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    official_linear = load_list(OFFICIAL / "linear.pt")
    official_attention = load_list(OFFICIAL / "attn.pt")
    coverage_linear = [
        torch.load(path, weights_only=True)
        for path in sorted((COVERAGE / "linear").glob("*.pt"))
    ]
    coverage_attention = [
        torch.load(path, weights_only=True)
        for path in sorted((COVERAGE / "attention").glob("*.pt"))
    ]
    linear = official_linear + coverage_linear
    attention = official_attention + coverage_attention
    linear_path = output / "linear.pt"
    attention_path = output / "attn.pt"
    torch.save(linear, linear_path)
    torch.save(attention, attention_path)
    linear_cases = sum(len(group["test_activation_list"]) for group in linear)
    attention_cases = sum(len(group["test"]) for group in attention)
    manifest = {
        "schema_version": 1,
        "name": "combined-300",
        "linear_groups": len(linear),
        "attention_groups": len(attention),
        "linear_cases": linear_cases,
        "attention_cases": attention_cases,
        "total_cases": linear_cases + attention_cases,
        "files": {
            "linear.pt": {"sha256": sha256(linear_path), "bytes": linear_path.stat().st_size},
            "attn.pt": {"sha256": sha256(attention_path), "bytes": attention_path.stat().st_size},
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified 300-case HiF4 dataset")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
