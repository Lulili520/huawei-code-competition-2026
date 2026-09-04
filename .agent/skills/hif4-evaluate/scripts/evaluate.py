from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import torch

from scoring import (
    ATTENTION_SCORE_WEIGHT,
    LINEAR_SCORE_WEIGHT,
    SCORE_SCALE_CASES,
    weighted_total_score,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARCHIVE = ROOT / "reference" / "本地调试参考-0818.zip"
CACHE_SCHEMA_VERSION = 1
CACHE_ALGORITHM_VERSION = "cpu-reference-v1"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_nvfp4(quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    values = quant.float().unflatten(-1, (-1, 16))
    return (values * scale.float().unsqueeze(-1)).flatten(-2, -1)


def decode_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        params["scale_factor"].float()
        * params["scale_lv2"].float()
        * params["scale_lv3"].float()
        * params["sign"].float()
        * params["mant"].float()
    ).flatten(-4, -1)


def standard_hif4(tensor: torch.Tensor) -> torch.Tensor:
    """Local stateless standard: max-range E6M2 + hierarchical nearest rounding."""
    channels = int(tensor.shape[-1])
    blocks = tensor.float().reshape(*tensor.shape[:-1], channels // 64, 8, 2, 4)
    absolute = blocks.abs()
    block_max = absolute.amax(dim=(-3, -2, -1), keepdim=True)
    target = (block_max / 7.0).clamp(2.0 ** -48, 49152.0)
    exponent = torch.floor(torch.log2(target))
    step = torch.pow(2.0, exponent - 2.0)
    scale = (torch.ceil(target / step) * step).clamp(2.0 ** -48, 49152.0)
    max_eight = absolute.amax(dim=(-2, -1), keepdim=True)
    lv2 = torch.where(max_eight > scale * 3.5, 2.0, 1.0)
    max_four = absolute.amax(dim=-1, keepdim=True)
    lv3 = torch.where(max_four > scale * lv2 * 1.75, 2.0, 1.0)
    effective = scale * lv2 * lv3
    mant = torch.round(absolute / effective * 4.0).clamp(0.0, 7.0) * 0.25
    sign = torch.where(mant == 0.0, 0.0, torch.sign(blocks))
    return (sign * mant * effective).flatten(-4, -1)


def mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return (candidate.float() - reference.float()).square().mean().item()


def case_metrics(reference: torch.Tensor, standard: torch.Tensor,
                 candidate: torch.Tensor) -> dict[str, float]:
    return case_metrics_from_standard_mse(reference, candidate, mse(reference, standard))


def case_metrics_from_standard_mse(
    reference: torch.Tensor, candidate: torch.Tensor, mse_standard: float,
) -> dict[str, float]:
    mse_player = mse(reference, candidate)
    ratio = (mse_standard - mse_player) / max(mse_standard, 1e-30)
    return {
        "mse_standard": mse_standard,
        "mse_player": mse_player,
        "score_ratio": ratio,
        "score_percentage_points": 100.0 * ratio,
    }


def _reference_cache_key(datasets_dir: Path) -> str:
    manifest = datasets_dir / "manifest.json"
    if manifest.is_file():
        dataset_identity = manifest.read_bytes()
    else:
        dataset_identity = "|".join(
            f"{name}:{(datasets_dir / name).stat().st_size}:"
            f"{(datasets_dir / name).stat().st_mtime_ns}"
            for name in ("linear.pt", "attn.pt")
        ).encode("utf-8")
    material = b"|".join((
        str(CACHE_SCHEMA_VERSION).encode("ascii"),
        CACHE_ALGORITHM_VERSION.encode("ascii"),
        torch.__version__.encode("ascii"),
        dataset_identity,
    ))
    return hashlib.sha256(material).hexdigest()[:20]


class ReferenceCache:
    """Cache candidate-independent CPU reference outputs by dataset group."""

    def __init__(self, datasets_dir: Path, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.key = _reference_cache_key(datasets_dir)
        self.root = ROOT / ".agent" / "runtime" / "reference-cache" / self.key
        self.memory: dict[tuple[str, int], dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.build_seconds = 0.0
        self.load_seconds = 0.0

    def _valid(
        self, value: Any, kind: str, group_index: int, expected_cases: int,
    ) -> bool:
        return bool(
            isinstance(value, dict)
            and value.get("schema_version") == CACHE_SCHEMA_VERSION
            and value.get("cache_key") == self.key
            and value.get("kind") == kind
            and value.get("group_index") == group_index
            and isinstance(value.get("cases"), list)
            and len(value["cases"]) == expected_cases
        )

    def get(
        self, kind: str, group_index: int, expected_cases: int, builder: Any,
    ) -> dict[str, Any]:
        identity = (kind, group_index)
        if identity in self.memory:
            self.hits += 1
            return self.memory[identity]
        path = self.root / kind / f"{group_index:03d}.pt"
        if self.enabled and path.is_file():
            tick = time.perf_counter()
            try:
                value = torch.load(path, weights_only=True, map_location="cpu")
            except (OSError, RuntimeError, ValueError):
                value = None
            self.load_seconds += time.perf_counter() - tick
            if self._valid(value, kind, group_index, expected_cases):
                self.hits += 1
                self.memory[identity] = value
                return value

        tick = time.perf_counter()
        cases = builder()
        self.build_seconds += time.perf_counter() - tick
        value = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": self.key,
            "kind": kind,
            "group_index": group_index,
            "cases": cases,
        }
        if not self._valid(value, kind, group_index, expected_cases):
            raise RuntimeError(f"invalid {kind} reference cache payload for group {group_index}")
        self.misses += 1
        self.memory[identity] = value
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".pt.{os.getpid()}.tmp")
            torch.save(value, temporary)
            os.replace(temporary, path)
        return value

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cache_key": self.key,
            "memory_groups": len(self.memory),
            "hits": self.hits,
            "misses": self.misses,
            "build_seconds": self.build_seconds,
            "load_seconds": self.load_seconds,
        }


def build_linear_reference_cases(group: Any) -> list[dict[str, Any]]:
    weight = decode_nvfp4(group["weight_quant"], group["weight_scale"])
    standard_weight = standard_hif4(weight)
    cases = []
    for aq, scale in group["test_activation_list"]:
        activation = decode_nvfp4(aq, scale)
        reference = activation @ weight.T
        standard = standard_hif4(activation) @ standard_weight.T
        cases.append({
            "reference": reference.contiguous(),
            "mse_standard": mse(reference, standard),
        })
    return cases


def build_attention_reference_cases(group: Any) -> list[dict[str, Any]]:
    qh = group["q_num_heads"]
    kvh = group["kv_num_heads"]
    dim = group["head_dim"]
    repeat = qh // kvh
    cases = []
    for sample in group["test"]:
        q, k, value = (decode_nvfp4(*sample[name]) for name in ("q", "k", "v"))
        seq = q.shape[0]
        q = q.reshape(seq, qh, dim).transpose(0, 1)
        k = k.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
        value = value.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
        logits = q @ k.transpose(-1, -2) / math.sqrt(dim)
        reference = torch.softmax(logits, -1) @ value

        standard_q = standard_hif4(q.transpose(0, 1).reshape(seq, -1))
        standard_k = standard_hif4(
            k[::repeat].transpose(0, 1).reshape(seq, -1)
        )
        standard_v = standard_hif4(
            value[::repeat].transpose(0, 1).reshape(seq, -1)
        )
        sq = standard_q.reshape(seq, qh, dim).transpose(0, 1)
        sk = standard_k.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
        sv = standard_v.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
        standard_logits = sq @ sk.transpose(-1, -2) / math.sqrt(dim)
        standard = torch.softmax(standard_logits, -1) @ sv
        cases.append({
            "reference": reference.contiguous(),
            "mse_standard": mse(reference, standard),
        })
    return cases


class StreamingMSE:
    def __init__(self) -> None:
        self.squared_error = 0.0
        self.elements = 0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        error = candidate.float() - reference.float()
        self.squared_error += error.square().sum().item()
        self.elements += error.numel()

    def result(self) -> dict[str, float]:
        return {"mse": self.squared_error / max(self.elements, 1)}


def select_groups(groups: list[Any], limit: int | None) -> list[tuple[int, Any]]:
    """Choose a deterministic, evenly-spaced subset instead of a biased prefix."""
    if limit is None or limit >= len(groups):
        return list(enumerate(groups))
    if limit < 1:
        raise ValueError("group limit must be positive")
    if limit == 1:
        indices = [len(groups) // 2]
    else:
        indices = [round(index * (len(groups) - 1) / (limit - 1)) for index in range(limit)]
    if len(set(indices)) != len(indices):
        raise RuntimeError("group selection produced duplicate indices")
    return [(index, groups[index]) for index in indices]


def score_statistics(cases: list[dict[str, Any]]) -> dict[str, float | int]:
    scores = sorted(float(item["score_percentage_points"]) for item in cases)

    def percentile(fraction: float) -> float:
        if not scores:
            return math.nan
        position = fraction * (len(scores) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return scores[lower]
        weight = position - lower
        return scores[lower] * (1.0 - weight) + scores[upper] * weight

    return {
        "mean": sum(scores) / max(len(scores), 1),
        "minimum": scores[0] if scores else math.nan,
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "maximum": scores[-1] if scores else math.nan,
        "negative_count": sum(value < 0.0 for value in scores),
    }


def optional_implementation_diagnostics(module: Any) -> dict[str, Any] | None:
    hook = getattr(module, "hif4_get_diagnostics", None)
    if not callable(hook):
        return None
    value = hook()
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        return {"diagnostic_error": f"non-JSON diagnostics: {error}"}
    return value


def evaluate(
    candidate_path: Path,
    linear: list[tuple[int, Any]],
    attention: list[tuple[int, Any]],
    reference_cache: ReferenceCache,
) -> dict[str, Any]:
    reference_seconds_before = (
        reference_cache.build_seconds + reference_cache.load_seconds
    )
    module = load_module(f"candidate_{abs(hash(candidate_path))}", candidate_path)
    linear_outputs = StreamingMSE()
    attention_outputs = StreamingMSE()
    linear_scores: list[float] = []
    attention_scores: list[float] = []
    linear_cases: list[dict[str, Any]] = []
    attention_cases: list[dict[str, Any]] = []
    timings = {
        "linear_calibration_seconds": 0.0,
        "linear_cases_seconds": 0.0,
        "linear_dynamic_quantization_seconds": 0.0,
        "linear_output_metric_seconds": 0.0,
        "attention_calibration_seconds": 0.0,
        "attention_cases_seconds": 0.0,
        "attention_dynamic_quantization_seconds": 0.0,
        "attention_output_metric_seconds": 0.0,
    }
    started = time.perf_counter()

    for group_index, group in linear:
        cached = reference_cache.get(
            "linear", group_index, len(group["test_activation_list"]),
            lambda group=group: build_linear_reference_cases(group),
        )
        tick = time.perf_counter()
        calibrated = module.hif4_calibration_and_quantize_weight(
            group["weight_quant"], group["weight_scale"],
            group["calib_activation_list"],
        )
        timings["linear_calibration_seconds"] += time.perf_counter() - tick
        quantized_weight = decode_hif4(calibrated["weight_params"])
        state = calibrated["activation_state"]
        tick = time.perf_counter()
        for sample_index, (aq, scale) in enumerate(group["test_activation_list"]):
            dynamic_tick = time.perf_counter()
            quantized_activation = decode_hif4(
                module.hif4_dynamic_quantize_activation(aq, scale, state)
            )
            timings["linear_dynamic_quantization_seconds"] += (
                time.perf_counter() - dynamic_tick
            )
            metric_tick = time.perf_counter()
            player_output = quantized_activation @ quantized_weight.T
            reference_output = cached["cases"][sample_index]["reference"]
            linear_outputs.update(reference_output, player_output)
            detail = case_metrics_from_standard_mse(
                reference_output, player_output,
                float(cached["cases"][sample_index]["mse_standard"]),
            )
            detail.update(group_index=group_index, sample_index=sample_index)
            linear_cases.append(detail)
            linear_scores.append(detail["score_ratio"])
            timings["linear_output_metric_seconds"] += (
                time.perf_counter() - metric_tick
            )
        timings["linear_cases_seconds"] += time.perf_counter() - tick

    for group_index, group in attention:
        qh = group["q_num_heads"]
        kvh = group["kv_num_heads"]
        dim = group["head_dim"]
        cached = reference_cache.get(
            "attention", group_index, len(group["test"]),
            lambda group=group: build_attention_reference_cases(group),
        )
        tick = time.perf_counter()
        states = module.hif4_calibration_attention(
            group["calib"], qh, kvh, dim
        )
        timings["attention_calibration_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        for sample_index, sample in enumerate(group["test"]):
            dynamic_tick = time.perf_counter()
            q1 = decode_hif4(module.hif4_dynamic_quantize_q(
                *sample["q"], qh, dim, states["q_state"]
            ))
            k1 = decode_hif4(module.hif4_dynamic_quantize_k(
                *sample["k"], kvh, dim, states["k_state"]
            ))
            v_candidate = decode_hif4(module.hif4_dynamic_quantize_v(
                *sample["v"], kvh, dim, states["v_state"]
            ))
            timings["attention_dynamic_quantization_seconds"] += (
                time.perf_counter() - dynamic_tick
            )
            metric_tick = time.perf_counter()
            seq = q1.shape[0]
            repeat = qh // kvh
            q1 = q1.reshape(seq, qh, dim).transpose(0, 1)
            k1 = k1.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            v_candidate = v_candidate.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            logits1 = q1 @ k1.transpose(-1, -2) / math.sqrt(dim)
            player_output = torch.softmax(logits1, -1) @ v_candidate
            reference_output = cached["cases"][sample_index]["reference"]
            attention_outputs.update(reference_output, player_output)
            detail = case_metrics_from_standard_mse(
                reference_output, player_output,
                float(cached["cases"][sample_index]["mse_standard"]),
            )
            detail.update(group_index=group_index, sample_index=sample_index)
            attention_cases.append(detail)
            attention_scores.append(detail["score_ratio"])
            timings["attention_output_metric_seconds"] += (
                time.perf_counter() - metric_tick
            )
        timings["attention_cases_seconds"] += time.perf_counter() - tick

    wall_seconds = time.perf_counter() - started
    reference_seconds = max(
        0.0,
        reference_cache.build_seconds
        + reference_cache.load_seconds
        - reference_seconds_before,
    )
    total_score = weighted_total_score(linear_scores, attention_scores)
    return {
        "path": str(candidate_path),
        "seconds": max(0.0, wall_seconds - reference_seconds),
        "wall_seconds": wall_seconds,
        "reference_seconds": reference_seconds,
        "linear_output": linear_outputs.result(),
        "attention_output": attention_outputs.result(),
        "linear_score": sum(linear_scores),
        "attention_score": sum(attention_scores),
        "total_score": total_score,
        "score_statistics": {
            "linear": score_statistics(linear_cases),
            "attention": score_statistics(attention_cases),
        },
        "case_diagnostics": {
            "linear": linear_cases,
            "attention": attention_cases,
        },
        "timings": timings,
        "implementation_diagnostics": optional_implementation_diagnostics(module),
        "score_weights": {
            "linear": LINEAR_SCORE_WEIGHT,
            "attention": ATTENTION_SCORE_WEIGHT,
        },
        "score_scale_case_count": SCORE_SCALE_CASES,
        "case_count": len(linear_scores) + len(attention_scores),
        "linear_case_count": len(linear_scores),
        "attention_case_count": len(attention_scores),
        "selected_group_indices": {
            "linear": [index for index, _ in linear],
            "attention": [index for index, _ in attention],
        },
    }


def print_result(result: dict[str, Any]) -> None:
    print(f"\n{result['path']}")
    print(
        f"cases                  Linear={result['linear_case_count']} "
        f"Attention={result['attention_case_count']}"
    )
    weights = result["score_weights"]
    print(
        f"score weight           Linear={weights['linear']:g} "
        f"Attention={weights['attention']:g}"
    )
    print("metric                 MSE")
    for label, key in (
        ("Linear Output", "linear_output"),
        ("Attention Output", "attention_output"),
    ):
        print(f"{label:<22} {result[key]['mse']:.6e}")
    print(f"Final score: {result['total_score']:.6f} percentage-points")


def main() -> int:
    parser = argparse.ArgumentParser(description="统一比较多个 HiF4 solution.py")
    parser.add_argument("solutions", nargs="+", type=Path)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        help="evaluate one unified dataset directory containing linear.pt and attn.pt",
    )
    parser.add_argument(
        "--linear-groups", type=int,
        help="deterministically sample this many Linear groups",
    )
    parser.add_argument(
        "--attention-groups", type=int,
        help="deterministically sample this many Attention groups",
    )
    parser.add_argument(
        "--json-output", type=Path,
        help="write machine-readable evaluation results to this JSON file",
    )
    parser.add_argument(
        "--fidelity-label", choices=("formal", "screening", "manual"),
        help="explicitly label the result without changing the selected groups",
    )
    parser.add_argument(
        "--skip-self-check", action="store_true",
        help="跳过官方输出格式检查，只计算本地 MSE 与得分",
    )
    parser.add_argument(
        "--no-reference-cache", action="store_true",
        help="compute candidate-independent references in memory without disk reuse",
    )
    args = parser.parse_args()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hif4_eval_") as directory:
        with zipfile.ZipFile(args.archive) as archive:
            archive.extractall(directory)
        example = Path(directory) / "example"
        checker = load_module("official_self_check", example / "self_check.py")
        datasets_dir = (
            args.datasets_dir.resolve()
            if args.datasets_dir is not None
            else example / "mini_sample"
        )
        for required in ("linear.pt", "attn.pt"):
            if not (datasets_dir / required).is_file():
                parser.error(f"missing {required} in {datasets_dir}")
        linear_all = checker._normalize_linear_dataset(
            torch.load(datasets_dir / "linear.pt", weights_only=True, mmap=True)
        )
        attention_all = checker._normalize_attention_dataset(
            torch.load(datasets_dir / "attn.pt", weights_only=True, mmap=True)
        )
        try:
            linear = select_groups(linear_all, args.linear_groups)
            attention = select_groups(attention_all, args.attention_groups)
        except ValueError as error:
            parser.error(str(error))
        reference_cache = ReferenceCache(
            datasets_dir, enabled=not args.no_reference_cache,
        )
        for path in args.solutions:
            path = path.resolve()
            if not args.skip_self_check:
                # Validate interfaces and shapes on the compact official suite.
                # This check is separate from the fixed numeric evaluation.
                completed = subprocess.run([
                    sys.executable,
                    str(example / "self_check.py"),
                    "--solution_dir", str(path.parent),
                    "--datasets_dir", str(example / "mini_sample"),
                ], cwd=ROOT)
                if completed.returncode:
                    return completed.returncode
            result = evaluate(path, linear, attention, reference_cache)
            result["datasets_dir"] = str(datasets_dir)
            result["evaluation_fidelity"] = args.fidelity_label or (
                "full"
                if len(linear) == len(linear_all) and len(attention) == len(attention_all)
                else "screening"
            )
            print_result(result)
            results.append(result)
            gc.collect()
    cache_diagnostics = reference_cache.diagnostics()
    for result in results:
        result["reference_cache"] = cache_diagnostics
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({
                "schema_version": 2,
                "reference_cache": cache_diagnostics,
                "results": results,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
