from __future__ import annotations

import argparse
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


def case_score(reference: torch.Tensor, standard: torch.Tensor,
               candidate: torch.Tensor) -> float:
    mse_standard = mse(reference, standard)
    mse_player = mse(reference, candidate)
    return (mse_standard - mse_player) / max(mse_standard, 1e-30)


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


def evaluate(
    candidate_path: Path,
    example: Path,
    checker: Any,
) -> dict[str, Any]:
    module = load_module(f"candidate_{abs(hash(candidate_path))}", candidate_path)
    linear = checker._normalize_linear_dataset(
        torch.load(example / "mini_sample" / "linear.pt", weights_only=True)
    )
    attention = checker._normalize_attention_dataset(
        torch.load(example / "mini_sample" / "attn.pt", weights_only=True)
    )
    linear_outputs = StreamingMSE()
    attention_outputs = StreamingMSE()
    linear_scores: list[float] = []
    attention_scores: list[float] = []
    started = time.perf_counter()

    for group in linear:
        weight = decode_nvfp4(group["weight_quant"], group["weight_scale"])
        standard_weight = standard_hif4(weight)
        calibrated = module.hif4_calibration_and_quantize_weight(
            group["weight_quant"], group["weight_scale"],
            group["calib_activation_list"],
        )
        quantized_weight = decode_hif4(calibrated["weight_params"])
        state = calibrated["activation_state"]
        for aq, scale in group["test_activation_list"]:
            activation = decode_nvfp4(aq, scale)
            quantized_activation = decode_hif4(
                module.hif4_dynamic_quantize_activation(aq, scale, state)
            )
            reference_output = activation @ weight.T
            standard_output = standard_hif4(activation) @ standard_weight.T
            player_output = quantized_activation @ quantized_weight.T
            linear_outputs.update(reference_output, player_output)
            linear_scores.append(case_score(
                reference_output, standard_output, player_output
            ))

    for group in attention:
        qh = group["q_num_heads"]
        kvh = group["kv_num_heads"]
        dim = group["head_dim"]
        states = module.hif4_calibration_attention(
            group["calib"], qh, kvh, dim
        )
        for sample in group["test"]:
            q0, k0, v0 = (decode_nvfp4(*sample[x]) for x in ("q", "k", "v"))
            q1 = decode_hif4(module.hif4_dynamic_quantize_q(
                *sample["q"], qh, dim, states["q_state"]
            ))
            k1 = decode_hif4(module.hif4_dynamic_quantize_k(
                *sample["k"], kvh, dim, states["k_state"]
            ))
            v1 = decode_hif4(module.hif4_dynamic_quantize_v(
                *sample["v"], kvh, dim, states["v_state"]
            ))
            seq = q0.shape[0]
            repeat = qh // kvh
            q0 = q0.reshape(seq, qh, dim).transpose(0, 1)
            q1 = q1.reshape(seq, qh, dim).transpose(0, 1)
            k0 = k0.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            k1 = k1.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            v0 = v0.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            v1 = v1.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            logits0 = q0 @ k0.transpose(-1, -2) / math.sqrt(dim)
            logits1 = q1 @ k1.transpose(-1, -2) / math.sqrt(dim)
            reference_output = torch.softmax(logits0, -1) @ v0
            player_output = torch.softmax(logits1, -1) @ v1
            attention_outputs.update(reference_output, player_output)
            standard_q = standard_hif4(q0.transpose(0, 1).reshape(seq, -1))
            standard_k = standard_hif4(
                k0[::repeat].transpose(0, 1).reshape(seq, -1)
            )
            standard_v = standard_hif4(
                v0[::repeat].transpose(0, 1).reshape(seq, -1)
            )
            sq = standard_q.reshape(seq, qh, dim).transpose(0, 1)
            sk = standard_k.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            sv = standard_v.reshape(seq, kvh, dim).transpose(0, 1).repeat_interleave(repeat, 0)
            standard_logits = sq @ sk.transpose(-1, -2) / math.sqrt(dim)
            standard_output = torch.softmax(standard_logits, -1) @ sv
            attention_scores.append(case_score(
                reference_output, standard_output, player_output,
            ))

    return {
        "path": str(candidate_path),
        "seconds": time.perf_counter() - started,
        "linear_output": linear_outputs.result(),
        "attention_output": attention_outputs.result(),
        "linear_score": sum(linear_scores),
        "attention_score": sum(attention_scores),
        "total_score": sum(linear_scores) + sum(attention_scores),
        "case_count": len(linear_scores) + len(attention_scores),
        "linear_case_count": len(linear_scores),
        "attention_case_count": len(attention_scores),
    }


def print_result(result: dict[str, Any]) -> None:
    print(f"\n{result['path']}")
    print(
        f"cases                  Linear={result['linear_case_count']} "
        f"Attention={result['attention_case_count']}"
    )
    print("metric                 MSE")
    for label, key in (
        ("Linear Output", "linear_output"),
        ("Attention Output", "attention_output"),
    ):
        print(f"{label:<22} {result[key]['mse']:.6e}")
    print(
        f"Final score: {result['total_score']:.6f} "
        f"({result['total_score'] * 100.0:.3f} percentage-points)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="统一比较多个 HiF4 solution.py")
    parser.add_argument("solutions", nargs="+", type=Path)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
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
        for path in args.solutions:
            path = path.resolve()
            if not args.skip_self_check:
                completed = subprocess.run([
                    sys.executable,
                    str(example / "self_check.py"),
                    "--solution_dir", str(path.parent),
                    "--datasets_dir", str(example / "mini_sample"),
                ], cwd=ROOT)
                if completed.returncode:
                    return completed.returncode
            result = evaluate(path, example, checker)
            print_result(result)
            results.append(result)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"schema_version": 1, "results": results}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
