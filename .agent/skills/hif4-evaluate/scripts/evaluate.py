from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
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
    weighted_total_score,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARCHIVE = ROOT / "reference" / "本地调试参考-0818.zip"


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
    mse_standard = mse(reference, standard)
    mse_player = mse(reference, candidate)
    ratio = (mse_standard - mse_player) / max(mse_standard, 1e-30)
    return {
        "mse_standard": mse_standard,
        "mse_player": mse_player,
        "score_ratio": ratio,
        "score_percentage_points": 100.0 * ratio,
    }


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
) -> dict[str, Any]:
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
        "attention_calibration_seconds": 0.0,
        "attention_cases_seconds": 0.0,
    }
    started = time.perf_counter()

    for group_index, group in linear:
        weight = decode_nvfp4(group["weight_quant"], group["weight_scale"])
        standard_weight = standard_hif4(weight)
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
            activation = decode_nvfp4(aq, scale)
            quantized_activation = decode_hif4(
                module.hif4_dynamic_quantize_activation(aq, scale, state)
            )
            reference_output = activation @ weight.T
            standard_output = standard_hif4(activation) @ standard_weight.T
            player_output = quantized_activation @ quantized_weight.T
            linear_outputs.update(reference_output, player_output)
            detail = case_metrics(reference_output, standard_output, player_output)
            detail.update(group_index=group_index, sample_index=sample_index)
            linear_cases.append(detail)
            linear_scores.append(detail["score_ratio"])
        timings["linear_cases_seconds"] += time.perf_counter() - tick

    for group_index, group in attention:
        qh = group["q_num_heads"]
        kvh = group["kv_num_heads"]
        dim = group["head_dim"]
        tick = time.perf_counter()
        states = module.hif4_calibration_attention(
            group["calib"], qh, kvh, dim
        )
        timings["attention_calibration_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        for sample_index, sample in enumerate(group["test"]):
            q0, k0, v_reference = (decode_nvfp4(*sample[x]) for x in ("q", "k", "v"))
            q1 = decode_hif4(module.hif4_dynamic_quantize_q(
                *sample["q"], qh, dim, states["q_state"]
            ))
            k1 = decode_hif4(module.hif4_dynamic_quantize_k(
                *sample["k"], kvh, dim, states["k_state"]
            ))
            v_candidate = decode_hif4(module.hif4_dynamic_quantize_v(
                *sample["v"], kvh, dim, states["v_state"]
            ))
            seq = q0.shape[0]
            repeat = qh // kvh
            q0 = q0.reshape(seq, qh, dim).transpose(0, 1)
            q1 = q1.reshape(seq, qh, dim).transpose(0, 1)
            k0 = k0.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            k1 = k1.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            v_reference = v_reference.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            v_candidate = v_candidate.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            logits0 = q0 @ k0.transpose(-1, -2) / math.sqrt(dim)
            logits1 = q1 @ k1.transpose(-1, -2) / math.sqrt(dim)
            reference_output = torch.softmax(logits0, -1) @ v_reference
            player_output = torch.softmax(logits1, -1) @ v_candidate
            attention_outputs.update(reference_output, player_output)
            standard_q = standard_hif4(q0.transpose(0, 1).reshape(seq, -1))
            standard_k = standard_hif4(
                k0[::repeat].transpose(0, 1).reshape(seq, -1)
            )
            standard_v = standard_hif4(
                v_reference[::repeat].transpose(0, 1).reshape(seq, -1)
            )
            sq = standard_q.reshape(seq, qh, dim).transpose(0, 1)
            sk = standard_k.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            sv = standard_v.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            standard_logits = sq @ sk.transpose(-1, -2) / math.sqrt(dim)
            standard_output = torch.softmax(standard_logits, -1) @ sv
            detail = case_metrics(reference_output, standard_output, player_output)
            detail.update(group_index=group_index, sample_index=sample_index)
            attention_cases.append(detail)
            attention_scores.append(detail["score_ratio"])
        timings["attention_cases_seconds"] += time.perf_counter() - tick

    total_score = weighted_total_score(linear_scores, attention_scores)
    return {
        "path": str(candidate_path),
        "seconds": time.perf_counter() - started,
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
        "--skip-self-check", action="store_true",
        help="跳过官方输出格式检查，只计算本地 MSE 与得分",
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
        for path in args.solutions:
            path = path.resolve()
            if not args.skip_self_check:
                # Validate interfaces and shapes on the compact official suite.
                # Using the 300-case scoring set here would execute it twice.
                completed = subprocess.run([
                    sys.executable,
                    str(example / "self_check.py"),
                    "--solution_dir", str(path.parent),
                    "--datasets_dir", str(example / "mini_sample"),
                ], cwd=ROOT)
                if completed.returncode:
                    return completed.returncode
            result = evaluate(path, linear, attention)
            result["datasets_dir"] = str(datasets_dir)
            result["evaluation_fidelity"] = (
                "full"
                if len(linear) == len(linear_all) and len(attention) == len(attention_all)
                else "screening"
            )
            print_result(result)
            results.append(result)
            gc.collect()
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"schema_version": 1, "results": results}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
