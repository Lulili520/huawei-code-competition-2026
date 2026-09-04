"""HiF4 V51: output-sensitive legal resolution projection after V37 routing.

The six competition functions remain the complete quantization surface.  Linear is
unchanged from V37.  Attention preserves V37's frozen-V and held-out Q/K routes,
then proposes resolution-only V moves on the existing legal HiF4 lattice.  Two fit
folds score adjacent mantissa/LV3/LV2 moves with the exact fixed-QK V-output
quadratic; two selection-held-out folds must both improve exact Softmax output
before a static action is published.  Format failure is fail-closed to the complete
V37 result.  ``hif4_get_diagnostics`` is read-only metadata.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch


_BLOCK = 64
_CROSSBLOCK_WINDOW = 128
_CROSSBLOCK_ROW_CHUNK = 1024
_TOP4_ROW_CHUNK = 1024
_EDGE_ROW_CHUNK = 512
_CALIBRATION_SHARDS = 5
_MIN_SCALE = 2.0 ** -48
_MAX_SCALE = 49152.0
_BASE_FACTORS = (0.5, 0.625, 0.75, 0.875, 1.0, 1.25)
_SENSITIVE_FACTORS = (
    0.375,
    0.4375,
    0.5,
    0.5625,
    0.625,
    0.6875,
    0.75,
    0.8125,
    0.875,
    0.9375,
    1.0,
    1.125,
    1.25,
    1.375,
    1.5,
)
_BASE_FACTOR_SET = frozenset(_BASE_FACTORS)
_LINEAR_GAIN = 0.0025
_LINEAR_OUTPUT_DAMPING = 2.0 ** -12
_LINEAR_OUTPUT_FLOOR = 1.0e-8
_LINEAR_OUTPUT_SCHEMA = "aonly-pow2-output-window-h128-v3"
_LINEAR_OUTPUT_WINDOW = 128
_LINEAR_OUTPUT_CODE_ROW_CHUNK = 1024
_LINEAR_OUTPUT_CODE_DESCENT_STEPS = 16
_HESSIAN_GAIN = 0.0005
_HESSIAN_DAMPING = 0.003
_TOP4_GAIN = 0.03
_ATTENTION_GAIN = 0.05
_ATTENTION_ALPHA = 0.625
_ATTENTION_OUTPUT_ALPHA_ABLATION = "off"
_ATTENTION_OUTPUT_ALPHA_CANDIDATES = (0.375, 0.5, 0.625, 0.75, 0.875)
_ATTENTION_OUTPUT_ALPHA_MAX_TOKENS = 32
_SMOOTH_MIN = 2.0 ** -8
_SMOOTH_MAX = 2.0 ** 8
_PAIR_REGULARIZATION = 2.0 ** -12
_PAIR_REGULARIZATION_FLOOR = 2.0 ** -24
_PAIR_CONDITION_LIMIT = 2.0 ** 16
_PAIR_EQUIVALENCE_LIMIT = 2.0e-5
_PAIR_ABLATION = "full"
_V_PAIR_ABLATION = "full"
_V_HIERARCHY_ABLATION = "full"
_V_PROFILE_ABLATION = "tridiagonal-full"
_V_PROFILE_SCHEMA = "v281-length-matched-softmax-ptp-tridiagonal-v1"
_V_PROFILE_MAX_SEQUENCE = 128
_V_GLOBAL_DC_ABLATION = "full"
_V_GLOBAL_DC_LAMBDA = 2.0 ** -8
_K_GAUGE_ABLATION = "best-channel-bisection"
_K_BREAKPOINT_ABLATION = "short-full"
_K_BREAKPOINT_SEQUENCE_LIMIT = 128
_K_PAIR_CODE_ABLATION = "full"
_K_PAIR_CODE_SCREEN = 8
_K_PAIR_CODE_SEQUENCE_LIMIT = 512
_Q_HESSIAN_ABLATION = "softmax-nullspace-deflated"
_Q_CODE_DESCENT_ABLATION = "full"
_K_CODE_DESCENT_ABLATION = "full"
_K_GAUGE_BISECTION_STEPS = 8
_Q_CODE_DESCENT_STEPS = 3
_Q_CODE_DESCENT_THIRD_STEP_COORD_BUDGET = 16384
_K_CODE_DESCENT_STEPS = 4
_QK_WIDE_SCHEMA = "v290-shape-adaptive-fullhead-consumer-covariance-v1"
_QK_WIDE_MAX_WINDOW = 256
# Legacy screened prototype below remains fixed at H256 and is never dispatched.
_QK_WIDE_WINDOW = _QK_WIDE_MAX_WINDOW
_QK_WIDE_Q_STEPS = 4
_QK_WIDE_LARGE_Q_STEPS = 7
_QK_WIDE_K_STEPS = 24
_QK_WIDE_CALIBRATION_SHARDS = 4
_QK_WIDE_SCREEN = 64
_QK_WIDE_ROW_CHUNK = 32
_QK_OUTER_SCALE_ABLATION = "off"
_QK_PERMUTATION_ABLATION = "full"
_QK_PERMUTATION_SCHEMA = "v278-analytic-nested-envelope-permutation-v2"
_QK_PERMUTATION_SHARDS = 3
_V_PAIR_SCHEMA = "v222-stationary-first-lag-v1"
_V_PAIR_MAX_ANCHORS = 8
_V_PAIR_RHO_LIMIT = 0.45
_V_PAIR_SHARDS = 3
_V_LAG2_ABLATION = "short-full"
_V_LAG2_SCHEMA = "v286-stationary-band2-short-v1"
_V_LAG2_SEQUENCE_LIMIT = 128
_V_LAG2_SCRATCH_LIMIT = 64 << 20
_V_LOWRANK_ABLATION = "full"
_V_LOWRANK_SCHEMA = "v281-post-dc-normalized-dct-consumer-rank4-fast-gate-v1"
_V_LOWRANK_RANK = 4
_V_LOWRANK_SHARDS = 3
_V_LOWRANK_DIAGONAL_SHRINK = 0.125
_V_LOWRANK_EPS = 1.0e-12
_V_LOWRANK_DESCENT_STEPS = 4
_V_NYSTROM_SCHEMA = "v286-length-matched-probability-gram-nystrom-rank4-v1"
_V_NYSTROM_EIGEN_TOLERANCE = 1.0e-8
_V_NYSTROM_ABLATION = "off"
_REPAIR_ROUTER_SCHEMA = "v37-heldout-qk-rescue-cascade-v1"
_REPAIR_ROUTER_BUDGET_FRACTION = 0.30
_REPAIR_ROUTER_PILOT_TOKENS = 16
_REPAIR_ROUTER_PILOT_FOLDS = 2
_REPAIR_ROUTER_EPS = 1.0e-20
_QK_RESCUE_FIT_FOLDS = 2
_QK_RESCUE_HELDOUT_FOLDS = 2
_FORMAT_PROJECTION_MODE = "on"
_FORMAT_PROJECTION_SCHEMA = "v51-output-sensitive-resolution-projection-v1"
_FORMAT_FIT_FOLDS = 2
_FORMAT_HELDOUT_FOLDS = 2
_FORMAT_MANT_ACTIONS = 2 * _BLOCK
_FORMAT_LV3_ACTIONS = 16
_FORMAT_LV2_ACTIONS = 8
_FORMAT_ACTIONS_PER_BLOCK = (
    _FORMAT_MANT_ACTIONS + _FORMAT_LV3_ACTIONS + _FORMAT_LV2_ACTIONS
)
_HIF4_PARAMETER_FIELDS = (
    "scale_factor",
    "scale_lv2",
    "scale_lv3",
    "sign",
    "mant",
)


_HIF4_DIAGNOSTICS: dict[str, int | float] = {
    "router_calibration_calls": 0,
    "router_calibration_successes": 0,
    "router_calibration_fallbacks": 0,
    "router_searches": 0,
    "router_pilot_folds": 0,
    "router_multiwindow_calibrations": 0,
    "router_calibration_time_ns": 0,
    "router_candidate_total": 0,
    "router_selected_total": 0,
    "router_budget_cost_ns": 0,
    "router_full_cost_ns": 0,
    "router_selected_cost_ns": 0,
    "router_predicted_benefit": 0.0,
    "router_selected_predicted_benefit": 0.0,
    "qk_candidate_total": 0,
    "qk_selected_total": 0,
    "v_candidate_total": 0,
    "v_selected_total": 0,
    "qk_pilot_cost_ns": 0,
    "v_pilot_cost_ns": 0,
    "qk_pilot_gain_numerator": 0.0,
    "qk_pilot_gain_denominator": 0.0,
    "qk_pilot_positive_folds": 0,
    "v_pilot_gain_numerator": 0.0,
    "v_pilot_gain_denominator": 0.0,
    "v_pilot_positive_folds": 0,
    "legacy_qk_selected_total": 0,
    "frozen_v_selected_total": 0,
    "legacy_router_selected_cost_ns": 0,
    "frozen_v_selected_cost_ns": 0,
    "qk_rescue_fit_folds": 0,
    "qk_rescue_heldout_folds": 0,
    "qk_rescue_candidate_total": 0,
    "qk_rescue_candidate_evaluations": 0,
    "qk_rescue_changed_candidates": 0,
    "qk_rescue_fit_comparisons": 0,
    "qk_rescue_fit_improved": 0,
    "qk_rescue_fit_equal": 0,
    "qk_rescue_fit_worse": 0,
    "qk_rescue_fit_eligible": 0,
    "qk_rescue_fit_rejected": 0,
    "qk_rescue_crossfold_accepted": 0,
    "qk_rescue_crossfold_rejected": 0,
    "qk_rescue_heldout_comparisons": 0,
    "qk_rescue_heldout_improved": 0,
    "qk_rescue_heldout_equal": 0,
    "qk_rescue_heldout_worse": 0,
    "qk_rescue_no_heldout_fallbacks": 0,
    "qk_rescue_gate_fallbacks": 0,
    "qk_rescue_v_baseline_builds": 0,
    "qk_rescue_v_baseline_effective_units": 0,
    "qk_rescue_candidate_cost_ns": 0,
    "qk_rescue_selected_cost_ns": 0,
    "qk_rescue_parent_normalized_loss": 0.0,
    "qk_rescue_candidate_normalized_loss": 0.0,
    "qk_rescue_fit_gain": 0.0,
    "qk_rescue_heldout_gain": 0.0,
    "qk_rescue_centered_logit_parent_sse": 0.0,
    "qk_rescue_centered_logit_candidate_sse": 0.0,
    "centered_logit_q_energy": 0.0,
    "centered_logit_k_energy": 0.0,
    "centered_logit_cross_energy": 0.0,
    "jacobian_q_output_risk": 0.0,
    "jacobian_k_output_risk": 0.0,
    "jacobian_cross_output_risk": 0.0,
    "jacobian_qk_joint_output_risk": 0.0,
    "v_path_output_risk": 0.0,
    "probability_value_cross_risk": 0.0,
    "format_clipping_sse": 0.0,
    "format_resolution_sse": 0.0,
    "q_dynamic_calls": 0,
    "k_dynamic_calls": 0,
    "v_dynamic_calls": 0,
    "qk_dynamic_candidate_units": 0,
    "qk_dynamic_selected_units": 0,
    "qk_dynamic_effective_units": 0,
    "q_expert_candidate_units": 0,
    "q_expert_selected_units": 0,
    "q_expert_effective_units": 0,
    "k_expert_candidate_units": 0,
    "k_expert_selected_units": 0,
    "k_expert_effective_units": 0,
    "k_window_local_selected_units": 0,
    "v_dynamic_candidate_units": 0,
    "v_dynamic_selected_units": 0,
    "v_dynamic_effective_units": 0,
    "q_expert_time_ns": 0,
    "k_expert_time_ns": 0,
    "v_expert_time_ns": 0,
    "router_dynamic_state_fallbacks": 0,
    "router_dynamic_shape_fallbacks": 0,
    "router_dynamic_exceptions": 0,
    "router_empty_routes": 0,
    "format_calibration_calls": 0,
    "format_calibration_successes": 0,
    "format_calibration_fallbacks": 0,
    "format_disabled_calls": 0,
    "format_calibration_time_ns": 0,
    "format_fit_folds": 0,
    "format_heldout_folds": 0,
    "format_candidate_blocks": 0,
    "format_action_total": 0,
    "format_resolution_candidate_total": 0,
    "format_clipping_candidate_total": 0,
    "format_clipping_invariant_rejected": 0,
    "format_fit_comparisons": 0,
    "format_fit_improved": 0,
    "format_fit_equal": 0,
    "format_fit_worse": 0,
    "format_fit_proposed_blocks": 0,
    "format_fit_rejected_blocks": 0,
    "format_heldout_candidates": 0,
    "format_heldout_comparisons": 0,
    "format_heldout_improved": 0,
    "format_heldout_equal": 0,
    "format_heldout_worse": 0,
    "format_crossfold_accepted": 0,
    "format_crossfold_rejected": 0,
    "format_no_heldout_fallbacks": 0,
    "format_gate_fallbacks": 0,
    "format_static_selected_blocks": 0,
    "format_mant_proposed": 0,
    "format_lv3_proposed": 0,
    "format_lv2_proposed": 0,
    "format_mant_accepted": 0,
    "format_lv3_accepted": 0,
    "format_lv2_accepted": 0,
    "format_fit_gain": 0.0,
    "format_heldout_gain": 0.0,
    "format_resolution_fit_output_gain": 0.0,
    "format_resolution_heldout_output_gain": 0.0,
    "format_clipping_fit_output_gain": 0.0,
    "format_clipping_heldout_output_gain": 0.0,
    "format_candidate_clipping_before": 0.0,
    "format_candidate_clipping_after": 0.0,
    "format_candidate_resolution_before": 0.0,
    "format_candidate_resolution_after": 0.0,
    "format_fit_fold0_parent_loss": 0.0,
    "format_fit_fold0_candidate_loss": 0.0,
    "format_fit_fold1_parent_loss": 0.0,
    "format_fit_fold1_candidate_loss": 0.0,
    "format_heldout_fold0_parent_loss": 0.0,
    "format_heldout_fold0_candidate_loss": 0.0,
    "format_heldout_fold1_parent_loss": 0.0,
    "format_heldout_fold1_candidate_loss": 0.0,
    "format_dynamic_calls": 0,
    "format_dynamic_candidate_blocks": 0,
    "format_dynamic_selected_blocks": 0,
    "format_dynamic_effective_blocks": 0,
    "format_dynamic_mant_blocks": 0,
    "format_dynamic_lv3_blocks": 0,
    "format_dynamic_lv2_blocks": 0,
    "format_dynamic_protected_values": 0,
    "format_dynamic_missing_state_fallbacks": 0,
    "format_dynamic_shape_fallbacks": 0,
    "format_dynamic_exceptions": 0,
    "format_dynamic_clipping_before": 0.0,
    "format_dynamic_clipping_after": 0.0,
    "format_dynamic_resolution_before": 0.0,
    "format_dynamic_resolution_after": 0.0,
}


def _diagnostic_add(name: str, value: int | float) -> None:
    """Accumulate one finite Python scalar without exposing mutable state."""
    if name not in _HIF4_DIAGNOSTICS:
        return
    if isinstance(_HIF4_DIAGNOSTICS[name], int) and isinstance(value, int):
        _HIF4_DIAGNOSTICS[name] = int(_HIF4_DIAGNOSTICS[name]) + value
        return
    numeric = float(value)
    if math.isfinite(numeric):
        _HIF4_DIAGNOSTICS[name] = float(_HIF4_DIAGNOSTICS[name]) + numeric


def _diagnostic_set(name: str, value: int | float) -> None:
    """Publish one finite immutable-by-reader scalar, including per-case slots."""
    if isinstance(value, int):
        _HIF4_DIAGNOSTICS[name] = int(value)
        return
    numeric = float(value)
    if math.isfinite(numeric):
        _HIF4_DIAGNOSTICS[name] = numeric


def hif4_get_diagnostics() -> dict[str, int | float]:
    """Return a read-only, JSON-serializable snapshot of router diagnostics."""
    return dict(_HIF4_DIAGNOSTICS)


def _qk_wide_window(head_dim: int) -> int:
    """Choose the largest complete H64-multiple window no wider than H256."""
    for window in (256, 192, 128, 64):
        if head_dim >= window and head_dim % window == 0:
            return window
    return 0


def _qk_wide_q_steps(head_dim: int) -> int:
    """Keep V290 small-head behavior and expose only the H256 probe route."""
    if head_dim == _QK_WIDE_MAX_WINDOW:
        return _QK_WIDE_LARGE_Q_STEPS
    if 64 < head_dim < _QK_WIDE_MAX_WINDOW:
        return _QK_WIDE_Q_STEPS
    return 0


def _normalized_dct_basis(
    length: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the first normalized DCT-II columns for a token sequence."""
    active_rank = min(max(rank, 0), length)
    if active_rank == 0:
        return torch.empty(length, 0, device=device, dtype=dtype)
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    frequency = torch.arange(
        active_rank, device=device, dtype=dtype
    ).unsqueeze(0)
    basis = torch.cos(
        torch.pi * (position + 0.5) * frequency / float(length)
    )
    basis[:, 0] *= float(length) ** -0.5
    if active_rank > 1:
        basis[:, 1:] *= (2.0 / float(length)) ** 0.5
    return basis


def _recover_nystrom_consumer_state(
    stats: dict[int, dict[str, Any]], kv_num_heads: int
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Recover trace-normalized, diagonally shrunk token kernels by length."""
    lengths: list[int] = []
    factors: list[torch.Tensor] = []
    diagonals: list[torch.Tensor] = []
    valid_heads: list[torch.Tensor] = []
    for length in sorted(stats):
        entry = stats[length]
        active_rank = min(_V_LOWRANK_RANK, length)
        try:
            y = entry["y"].to(torch.float64)
            moment = entry["m"].to(torch.float64)
            tau = entry["tau"].to(torch.float64)
            count = int(entry["count"])
            if (
                tuple(y.shape) != (kv_num_heads, length, active_rank)
                or tuple(moment.shape)
                != (kv_num_heads, active_rank, active_rank)
                or tuple(tau.shape) != (kv_num_heads,)
            ):
                continue
            moment = 0.5 * (moment + moment.transpose(-2, -1))
            eigenvalues, eigenvectors = torch.linalg.eigh(moment)
            largest = eigenvalues[:, -1].clamp_min(0.0)
            retained = eigenvalues > (
                _V_NYSTROM_EIGEN_TOLERANCE * largest[:, None]
            )
            inverse_root = torch.where(
                retained,
                eigenvalues.clamp_min(_V_LOWRANK_EPS).rsqrt(),
                torch.zeros_like(eigenvalues),
            )
            inverse_root = torch.matmul(
                eigenvectors * inverse_root[:, None, :],
                eigenvectors.transpose(-2, -1),
            )
            raw_factor = torch.matmul(inverse_root, y.transpose(1, 2))
            _, singular, right = torch.linalg.svd(raw_factor, full_matrices=False)
            eigen_energy = singular.square()
            scale = (tau / float(length)).clamp_min(_V_LOWRANK_EPS)
            complement = (tau - eigen_energy.sum(dim=-1)).clamp_min(0.0)
            delta = complement / float(max(length - active_rank, 1))
            delta = delta.clamp_min(_V_LOWRANK_EPS * scale)
            centered_energy = (
                (eigen_energy - delta[:, None]).clamp_min(0.0)
                / scale[:, None]
            )
            base_factor = centered_energy.sqrt().unsqueeze(-1) * right
            factor = ((1.0 - _V_LOWRANK_DIAGONAL_SHRINK) ** 0.5) * base_factor
            diagonal = (
                delta[:, None] / scale[:, None]
                + _V_LOWRANK_DIAGONAL_SHRINK
                * base_factor.square().sum(dim=1)
            )
            valid = (
                torch.full((kv_num_heads,), count >= 1, dtype=torch.bool)
                & torch.isfinite(y).all(dim=(-2, -1))
                & torch.isfinite(moment).all(dim=(-2, -1))
                & torch.isfinite(tau)
                & torch.isfinite(factor).all(dim=(-2, -1))
                & torch.isfinite(diagonal).all(dim=-1)
                & (tau > _V_LOWRANK_EPS)
                & (largest > _V_LOWRANK_EPS)
                & (retained.sum(dim=-1) > 0)
                & (diagonal > 0.0).all(dim=-1)
                & (factor.square().sum(dim=(-2, -1)) > 1.0e-6)
            )
            padded_factor = torch.zeros(
                kv_num_heads, _V_LOWRANK_RANK, length, dtype=torch.float32
            )
            padded_factor[:, :active_rank] = torch.where(
                valid[:, None, None], factor, torch.zeros_like(factor)
            ).to(torch.float32)
            diagonal = torch.where(
                valid[:, None], diagonal, torch.zeros_like(diagonal)
            ).to(torch.float32)
        except (KeyError, TypeError, ValueError, RuntimeError):
            continue
        lengths.append(length)
        factors.append(padded_factor.cpu())
        diagonals.append(diagonal.cpu())
        valid_heads.append(valid.cpu())
    return (
        torch.tensor(lengths, dtype=torch.int64),
        tuple(factors),
        tuple(diagonals),
        tuple(valid_heads),
    )


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = 16,
) -> torch.Tensor:
    channels = int(quant_float.shape[-1])
    if channels % blk_size:
        raise ValueError("NVFP4 last dimension must be divisible by 16")
    values = quant_float.unflatten(-1, (-1, blk_size))
    return (values * scale_float.unsqueeze(-1)).flatten(-2, -1).to(
        torch.bfloat16
    )


def _ceil_e6m2(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(min=_MIN_SCALE, max=_MAX_SCALE)
    exponent = torch.floor(torch.log2(value))
    step = torch.pow(2.0, exponent - 2.0)
    return (torch.ceil(value / step) * step).clamp(
        min=_MIN_SCALE, max=_MAX_SCALE
    )


def _floor_e6m2(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(min=_MIN_SCALE, max=_MAX_SCALE)
    exponent = torch.floor(torch.log2(value))
    step = torch.pow(2.0, exponent - 2.0)
    return (torch.floor(value / step) * step).clamp(
        min=_MIN_SCALE, max=_MAX_SCALE
    )


def _reshape_blocks(tensor: torch.Tensor) -> torch.Tensor:
    channels = int(tensor.shape[-1])
    if channels % _BLOCK:
        raise ValueError("HiF4 last dimension must be divisible by 64")
    return tensor.to(torch.float32).reshape(
        *tensor.shape[:-1], channels // _BLOCK, 8, 2, 4
    )


def _quantize_hif4_v1(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """The stateless legal encoder used only for invalid Linear state."""
    blocks = _reshape_blocks(tensor)
    absolute = blocks.abs()
    block_max = absolute.amax(dim=(-3, -2, -1), keepdim=True)
    scale_factor = _ceil_e6m2(block_max / 7.0)
    max_eight = absolute.amax(dim=(-2, -1), keepdim=True)
    scale_lv2 = torch.where(max_eight > scale_factor * 3.5, 2.0, 1.0)
    max_four = absolute.amax(dim=-1, keepdim=True)
    scale_lv3 = torch.where(
        max_four > scale_factor * scale_lv2 * 1.75, 2.0, 1.0
    )
    effective = scale_factor * scale_lv2 * scale_lv3
    mant = torch.round(absolute / effective * 4.0).clamp_(0.0, 7.0) * 0.25
    sign = torch.where(mant == 0.0, 0.0, torch.sign(blocks))
    return {
        "scale_factor": scale_factor.to(torch.bfloat16),
        "scale_lv2": scale_lv2.to(torch.bfloat16),
        "scale_lv3": scale_lv3.to(torch.bfloat16),
        "sign": sign.to(torch.bfloat16),
        "mant": mant.to(torch.bfloat16),
    }


def _quantize_hif4(
    tensor: torch.Tensor,
    importance: torch.Tensor | None = None,
    factors: tuple[float, ...] = _BASE_FACTORS,
) -> dict[str, torch.Tensor]:
    """Choose legal hierarchy parameters by tensor reconstruction SSE."""
    blocks = _reshape_blocks(tensor)
    absolute = blocks.abs()
    block_max = absolute.amax(dim=(-3, -2, -1), keepdim=True)
    if importance is None:
        error_weight: torch.Tensor | float = 1.0
    else:
        channels = int(tensor.shape[-1])
        weight = importance.to(device=tensor.device, dtype=torch.float32)
        if weight.ndim == 1:
            weight = weight.reshape(*([1] * (tensor.ndim - 1)), channels)
        error_weight = torch.broadcast_to(weight, tensor.shape).reshape_as(blocks)

    selected_error = torch.full_like(block_max, torch.inf)
    selected_scale = torch.ones_like(block_max)
    for factor in factors:
        candidate_scale = _ceil_e6m2(block_max * (factor / 7.0))
        level_error: list[torch.Tensor] = []
        for multiplier in (1.0, 2.0, 4.0):
            effective = candidate_scale * multiplier
            candidate_mant = (
                torch.round(absolute / effective * 4.0).clamp_(0.0, 7.0)
                * 0.25
            )
            candidate_mant.mul_(effective).sub_(absolute).square_()
            if type(error_weight) is torch.Tensor:
                candidate_mant.mul_(error_weight)
            level_error.append(
                candidate_mant.sum(dim=-1, keepdim=True)
            )
        lv2_error = [
            torch.minimum(level_error[0], level_error[1]).sum(
                dim=-2, keepdim=True
            ),
            torch.minimum(level_error[1], level_error[2]).sum(
                dim=-2, keepdim=True
            ),
        ]
        candidate_error = torch.minimum(lv2_error[0], lv2_error[1]).sum(
            dim=-3, keepdim=True
        )
        take = candidate_error < selected_error
        selected_error = torch.where(take, candidate_error, selected_error)
        selected_scale = torch.where(take, candidate_scale, selected_scale)

    level_mants: list[torch.Tensor] = []
    level_errors: list[torch.Tensor] = []
    for multiplier in (1.0, 2.0, 4.0):
        effective = selected_scale * multiplier
        candidate_mant = (
            torch.round(absolute / effective * 4.0).clamp_(0.0, 7.0) * 0.25
        )
        level_mants.append(candidate_mant)
        reconstruction_error = candidate_mant * effective
        reconstruction_error.sub_(absolute).square_()
        if type(error_weight) is torch.Tensor:
            reconstruction_error.mul_(error_weight)
        level_errors.append(
            reconstruction_error.sum(dim=-1, keepdim=True)
        )
    use_lv3_two_for_lv2_one = level_errors[1] < level_errors[0]
    use_lv3_two_for_lv2_two = level_errors[2] < level_errors[1]
    lv2_lv3 = [
        torch.where(use_lv3_two_for_lv2_one, 2.0, 1.0),
        torch.where(use_lv3_two_for_lv2_two, 2.0, 1.0),
    ]
    lv2_mants = [
        torch.where(
            use_lv3_two_for_lv2_one, level_mants[1], level_mants[0]
        ),
        torch.where(
            use_lv3_two_for_lv2_two, level_mants[2], level_mants[1]
        ),
    ]
    lv2_errors = [
        torch.minimum(level_errors[0], level_errors[1]).sum(
            dim=-2, keepdim=True
        ),
        torch.minimum(level_errors[1], level_errors[2]).sum(
            dim=-2, keepdim=True
        ),
    ]

    use_lv2_two = lv2_errors[1] < lv2_errors[0]
    scale_lv2 = torch.where(use_lv2_two, 2.0, 1.0)
    scale_lv3 = torch.where(use_lv2_two, lv2_lv3[1], lv2_lv3[0])
    mant = torch.where(use_lv2_two, lv2_mants[1], lv2_mants[0])
    sign = torch.where(mant == 0.0, 0.0, torch.sign(blocks))
    return {
        "scale_factor": selected_scale.to(torch.bfloat16),
        "scale_lv2": scale_lv2.to(torch.bfloat16),
        "scale_lv3": scale_lv3.to(torch.bfloat16),
        "sign": sign.to(torch.bfloat16),
        "mant": mant.to(torch.bfloat16),
    }


def _dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    value = params["sign"] * params["mant"]
    value.mul_(params["scale_lv3"])
    value.mul_(params["scale_lv2"])
    value.mul_(params["scale_factor"])
    return value.flatten(-4, -1).to(torch.float32)


def _quantize_hif4_plain_sensitive_pair(
    tensor: torch.Tensor,
    importance: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Share overlapping scale scans without changing either encoder result."""
    blocks = _reshape_blocks(tensor)
    absolute = blocks.abs()
    block_max = absolute.amax(dim=(-3, -2, -1), keepdim=True)
    channels = int(tensor.shape[-1])
    weight = importance.to(device=tensor.device, dtype=torch.float32)
    if weight.ndim == 1:
        weight = weight.reshape(*([1] * (tensor.ndim - 1)), channels)
    error_weight = torch.broadcast_to(weight, tensor.shape).reshape_as(blocks)

    plain_error = torch.full_like(block_max, torch.inf)
    plain_scale = torch.ones_like(block_max)
    sensitive_error = torch.full_like(block_max, torch.inf)
    sensitive_scale = torch.ones_like(block_max)
    for factor in _SENSITIVE_FACTORS:
        candidate_scale = _ceil_e6m2(block_max * (factor / 7.0))
        plain_levels: list[torch.Tensor] | None = (
            [] if factor in _BASE_FACTOR_SET else None
        )
        sensitive_levels: list[torch.Tensor] = []
        for multiplier in (1.0, 2.0, 4.0):
            effective = candidate_scale * multiplier
            reconstruction_error = (
                torch.round(absolute / effective * 4.0).clamp_(0.0, 7.0)
                * 0.25
            )
            reconstruction_error.mul_(effective).sub_(absolute).square_()
            if plain_levels is not None:
                plain_levels.append(
                    reconstruction_error.sum(dim=-1, keepdim=True)
                )
            sensitive_levels.append(
                (reconstruction_error * error_weight).sum(
                    dim=-1, keepdim=True
                )
            )

        sensitive_lv2 = (
            torch.minimum(sensitive_levels[0], sensitive_levels[1]).sum(
                dim=-2, keepdim=True
            ),
            torch.minimum(sensitive_levels[1], sensitive_levels[2]).sum(
                dim=-2, keepdim=True
            ),
        )
        factor_sensitive_error = torch.minimum(
            sensitive_lv2[0], sensitive_lv2[1]
        ).sum(dim=-3, keepdim=True)
        take_sensitive = factor_sensitive_error < sensitive_error
        sensitive_error = torch.where(
            take_sensitive, factor_sensitive_error, sensitive_error
        )
        sensitive_scale = torch.where(
            take_sensitive, candidate_scale, sensitive_scale
        )

        if plain_levels is not None:
            plain_lv2 = (
                torch.minimum(plain_levels[0], plain_levels[1]).sum(
                    dim=-2, keepdim=True
                ),
                torch.minimum(plain_levels[1], plain_levels[2]).sum(
                    dim=-2, keepdim=True
                ),
            )
            factor_plain_error = torch.minimum(
                plain_lv2[0], plain_lv2[1]
            ).sum(dim=-3, keepdim=True)
            take_plain = factor_plain_error < plain_error
            plain_error = torch.where(
                take_plain, factor_plain_error, plain_error
            )
            plain_scale = torch.where(
                take_plain, candidate_scale, plain_scale
            )

    def finalize(
        selected_scale: torch.Tensor,
        selected_weight: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        level_mants: list[torch.Tensor] = []
        level_errors: list[torch.Tensor] = []
        for multiplier in (1.0, 2.0, 4.0):
            effective = selected_scale * multiplier
            candidate_mant = (
                torch.round(absolute / effective * 4.0).clamp_(0.0, 7.0)
                * 0.25
            )
            level_mants.append(candidate_mant)
            reconstruction_error = candidate_mant * effective
            reconstruction_error.sub_(absolute).square_()
            if selected_weight is not None:
                reconstruction_error.mul_(selected_weight)
            level_errors.append(
                reconstruction_error.sum(dim=-1, keepdim=True)
            )
        use_lv3_two_for_lv2_one = level_errors[1] < level_errors[0]
        use_lv3_two_for_lv2_two = level_errors[2] < level_errors[1]
        lv2_lv3 = (
            torch.where(use_lv3_two_for_lv2_one, 2.0, 1.0),
            torch.where(use_lv3_two_for_lv2_two, 2.0, 1.0),
        )
        lv2_mants = (
            torch.where(
                use_lv3_two_for_lv2_one, level_mants[1], level_mants[0]
            ),
            torch.where(
                use_lv3_two_for_lv2_two, level_mants[2], level_mants[1]
            ),
        )
        lv2_errors = (
            torch.minimum(level_errors[0], level_errors[1]).sum(
                dim=-2, keepdim=True
            ),
            torch.minimum(level_errors[1], level_errors[2]).sum(
                dim=-2, keepdim=True
            ),
        )
        use_lv2_two = lv2_errors[1] < lv2_errors[0]
        scale_lv2 = torch.where(use_lv2_two, 2.0, 1.0)
        scale_lv3 = torch.where(use_lv2_two, lv2_lv3[1], lv2_lv3[0])
        mant = torch.where(use_lv2_two, lv2_mants[1], lv2_mants[0])
        sign = torch.where(mant == 0.0, 0.0, torch.sign(blocks))
        return {
            "scale_factor": selected_scale.to(torch.bfloat16),
            "scale_lv2": scale_lv2.to(torch.bfloat16),
            "scale_lv3": scale_lv3.to(torch.bfloat16),
            "sign": sign.to(torch.bfloat16),
            "mant": mant.to(torch.bfloat16),
        }

    return finalize(plain_scale, None), finalize(
        sensitive_scale, error_weight
    )


def _block_error(
    tensor: torch.Tensor,
    params: dict[str, torch.Tensor],
    importance: torch.Tensor,
) -> torch.Tensor:
    blocks = _reshape_blocks(tensor)
    reconstructed = _dequantize_hif4(params).reshape_as(blocks)
    channels = int(tensor.shape[-1])
    weight = importance.to(device=tensor.device, dtype=torch.float32)
    if weight.ndim == 1:
        weight = weight.reshape(*([1] * (tensor.ndim - 1)), channels)
    weight = torch.broadcast_to(weight, tensor.shape).reshape_as(blocks)
    reconstructed.sub_(blocks).square_().mul_(weight)
    return reconstructed.sum(dim=(-3, -2, -1), keepdim=True)


def _select_sensitive(
    tensor: torch.Tensor,
    parent: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    importance: torch.Tensor,
    min_gain: float,
) -> dict[str, torch.Tensor]:
    parent_error = _block_error(tensor, parent, importance)
    candidate_error = _block_error(tensor, candidate, importance)
    use_candidate = candidate_error < parent_error * (1.0 - min_gain)
    return {
        key: torch.where(use_candidate, candidate[key], parent[key])
        for key in parent
    }


def _quantize_sensitive(
    tensor: torch.Tensor,
    importance: torch.Tensor,
    min_gain: float,
) -> dict[str, torch.Tensor]:
    parent, _ = _quantize_sensitive_branches(
        tensor, importance, min_gain, return_alternate=False
    )
    return parent


def _quantize_sensitive_branches(
    tensor: torch.Tensor,
    importance: torch.Tensor,
    min_gain: float,
    return_alternate: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return the original sensitive choice and its legal opposite branch."""
    parent, candidate = _quantize_hif4_plain_sensitive_pair(
        tensor, importance
    )
    parent_error = _block_error(tensor, parent, importance)
    candidate_error = _block_error(tensor, candidate, importance)
    use_candidate = candidate_error < parent_error * (1.0 - min_gain)
    selected = {
        key: torch.where(use_candidate, candidate[key], parent[key])
        for key in parent
    }
    if not return_alternate:
        return selected, selected
    alternate = {
        key: torch.where(use_candidate, parent[key], candidate[key])
        for key in parent
    }
    return selected, alternate


def _aonly_statistics(
    activations: list[torch.Tensor], channels: int
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    square_sum = torch.zeros(channels, dtype=torch.float32)
    rows = 0
    for activation in activations:
        if activation.shape[-1] != channels:
            return torch.ones(channels), torch.ones(channels), False
        value = activation.to(torch.float32).reshape(-1, channels)
        if not bool(torch.isfinite(value).all().item()):
            return torch.ones(channels), torch.ones(channels), False
        square_sum += value.square().sum(dim=0).cpu()
        rows += int(value.shape[0])
    if rows == 0:
        return torch.ones(channels), torch.ones(channels), False

    second = square_sum / rows
    radius = torch.sqrt(second + 2.0 ** -48)
    geometric = torch.exp(torch.log(radius).mean())
    if not bool(torch.isfinite(second).all().item()) or not bool(
        torch.isfinite(geometric).item()
    ) or float(geometric.item()) <= 0.0:
        return torch.ones(channels), torch.ones(channels), False
    exponent = torch.round(0.5 * torch.log2(radius / geometric)).clamp(-4, 4)
    scale = torch.pow(2.0, exponent)
    denominator = (second / scale.square()).mean()
    if not bool(torch.isfinite(scale).all().item()) or not bool(
        torch.isfinite(denominator).item()
    ) or float(denominator.item()) <= 0.0:
        return torch.ones(channels), torch.ones(channels), False
    importance = ((second / scale.square()) / denominator).clamp(
        1.0 / 16.0, 16.0
    )
    if not bool(torch.isfinite(importance).all().item()):
        return torch.ones(channels), torch.ones(channels), False
    return scale, importance, True


def _block_grams(
    activations: list[torch.Tensor], precondition_scale: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    channels = int(precondition_scale.numel())
    block_count = channels // _BLOCK
    gram = torch.zeros(block_count, _BLOCK, _BLOCK, dtype=torch.float32)
    shard_grams: list[torch.Tensor] = []
    rows = 0
    for activation in activations:
        value = (
            activation.to(torch.float32) / precondition_scale
        ).reshape(-1, block_count, _BLOCK)
        if not bool(torch.isfinite(value).all().item()):
            return None, None
        shard_rows = int(value.shape[0])
        if shard_rows == 0:
            return None, None
        shard_sum = torch.einsum("nbi,nbj->bij", value, value).cpu()
        gram += shard_sum
        shard_grams.append(shard_sum / shard_rows)
        rows += shard_rows
    if rows == 0:
        return None, None
    gram /= rows
    if not bool(torch.isfinite(gram).all().item()):
        return None, None
    shards = None
    if len(shard_grams) == _CALIBRATION_SHARDS:
        candidate_shards = torch.stack(shard_grams)
        if bool(torch.isfinite(candidate_shards).all().item()):
            shards = candidate_shards
    return gram, shards


def _crossblock_gram(
    activations: list[torch.Tensor],
    precondition_scale: torch.Tensor,
    block_gram: torch.Tensor,
) -> torch.Tensor | None:
    """Build adjacent-two-block A-only Hessians for Weight calibration."""
    channels = int(precondition_scale.numel())
    window_count = channels // _CROSSBLOCK_WINDOW
    block_count = channels // _BLOCK
    if window_count == 0 or tuple(block_gram.shape) != (
        block_count,
        _BLOCK,
        _BLOCK,
    ):
        return None
    covered_channels = window_count * _CROSSBLOCK_WINDOW
    gram = torch.zeros(
        window_count,
        _CROSSBLOCK_WINDOW,
        _CROSSBLOCK_WINDOW,
        dtype=torch.float32,
    )
    paired_blocks = block_gram[: window_count * 2].reshape(
        window_count, 2, _BLOCK, _BLOCK
    )
    gram[:, :_BLOCK, :_BLOCK] = paired_blocks[:, 0]
    gram[:, _BLOCK:, _BLOCK:] = paired_blocks[:, 1]
    rows = 0
    for activation in activations:
        value = activation.to(torch.float32) / precondition_scale
        value = value.reshape(-1, channels)[:, :covered_channels].reshape(
            -1, window_count, _CROSSBLOCK_WINDOW
        )
        left, right = value.split(_BLOCK, dim=-1)
        cross = torch.einsum("nwi,nwj->wij", left, right).cpu()
        gram[:, :_BLOCK, _BLOCK:] += cross
        gram[:, _BLOCK:, :_BLOCK] += cross.transpose(-2, -1)
        rows += int(value.shape[0])
    if rows == 0:
        return None
    gram[:, :_BLOCK, _BLOCK:] /= rows
    gram[:, _BLOCK:, :_BLOCK] /= rows
    return gram if bool(torch.isfinite(gram).all().item()) else None


def _ordered_hessian_reconstruction(
    weight_block: torch.Tensor,
    scale_block: torch.Tensor,
    inverse: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor:
    """Run one GPTQ order and restore its output to natural coordinates."""
    working = weight_block.index_select(1, order).clone()
    ordered_scale = scale_block.index_select(1, order)
    ordered_inverse = inverse.index_select(0, order).index_select(1, order)
    ordered_output = torch.empty_like(working)
    for position in range(_BLOCK):
        value = working[:, position]
        column_scale = ordered_scale[:, position]
        mant = (
            torch.round(value.abs() / column_scale * 4.0).clamp(0.0, 7.0)
            * 0.25
        )
        reconstructed = torch.sign(value) * mant * column_scale
        ordered_output[:, position] = reconstructed
        if position + 1 < _BLOCK:
            denominator = ordered_inverse[position, position].clamp_min(1.0e-12)
            propagated = (value - reconstructed) / denominator
            working[:, position + 1 :] -= propagated.unsqueeze(-1) * (
                ordered_inverse[position, position + 1 :].unsqueeze(0)
            )

    output = torch.empty_like(ordered_output)
    output.index_copy_(1, order, ordered_output)
    return output


def _batched_ordered_hessian_reconstruction(
    weight_blocks: torch.Tensor,
    scale_blocks: torch.Tensor,
    inverses: torch.Tensor,
    orders: torch.Tensor,
) -> torch.Tensor:
    """Run independent GPTQ orders for a small batch of H64 blocks."""
    batch, rows, width = weight_blocks.shape
    if orders.ndim == 1:
        orders = orders.unsqueeze(0).expand(batch, -1)
    value_index = orders[:, None, :].expand(batch, rows, width)
    working = torch.gather(weight_blocks, 2, value_index).clone()
    ordered_scale = torch.gather(scale_blocks, 2, value_index)
    row_index = orders[:, :, None].expand(batch, width, width)
    ordered_inverse = torch.gather(inverses, 1, row_index)
    column_index = orders[:, None, :].expand(batch, width, width)
    ordered_inverse = torch.gather(ordered_inverse, 2, column_index)
    ordered_output = torch.empty_like(working)
    for position in range(width):
        value = working[:, :, position]
        column_scale = ordered_scale[:, :, position]
        mant = (
            torch.round(value.abs() / column_scale * 4.0).clamp(0.0, 7.0)
            * 0.25
        )
        reconstructed = torch.sign(value) * mant * column_scale
        ordered_output[:, :, position] = reconstructed
        if position + 1 < width:
            denominator = ordered_inverse[:, position, position].clamp_min(
                1.0e-12
            )
            propagated = (value - reconstructed) / denominator[:, None]
            working[:, :, position + 1 :] -= propagated[:, :, None] * (
                ordered_inverse[:, None, position, position + 1 :]
            )

    output = torch.empty_like(ordered_output)
    output.scatter_(2, value_index, ordered_output)
    return output


def _multiorder_hessian(
    weight: torch.Tensor,
    base: dict[str, torch.Tensor],
    gram: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Select natural, reverse, then diag-desc Weight candidates per row-block."""
    out_features, channels = int(weight.shape[0]), int(weight.shape[1])
    block_count = channels // _BLOCK
    weight_blocks = weight.to(torch.float32).reshape(
        out_features, block_count, _BLOCK
    ).permute(1, 0, 2)
    effective_scale = (
        base["scale_factor"].to(torch.float32)
        * base["scale_lv2"].to(torch.float32)
        * base["scale_lv3"].to(torch.float32)
    )
    scale_blocks = effective_scale.expand(
        out_features, block_count, 8, 2, 4
    ).reshape(out_features, block_count, _BLOCK).permute(1, 0, 2)
    base_quantized = _dequantize_hif4(base).reshape(
        out_features, block_count, _BLOCK
    ).permute(1, 0, 2)
    selected_sign = base["sign"].reshape(
        out_features, block_count, _BLOCK
    ).clone()
    selected_mant = base["mant"].reshape(
        out_features, block_count, _BLOCK
    ).clone()
    identity = torch.eye(
        _BLOCK, dtype=torch.float32, device=weight_blocks.device
    )
    natural = torch.arange(
        _BLOCK, dtype=torch.long, device=weight_blocks.device
    )
    reverse = torch.flip(natural, dims=(0,))

    valid_records: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for block_index in range(block_count):
        hessian = gram[block_index].to(
            device=weight_blocks.device, dtype=torch.float32
        )
        diagonal = torch.diagonal(hessian)
        if not bool(torch.isfinite(hessian).all().item()) or bool(
            (diagonal < 0.0).any().item()
        ):
            continue
        damp = diagonal.mean() * _HESSIAN_DAMPING
        if not bool(torch.isfinite(damp).item()):
            continue
        try:
            factor = torch.linalg.cholesky(hessian + identity * (damp + 1.0e-8))
            inverse = torch.cholesky_inverse(factor)
        except RuntimeError:
            continue
        if not bool(torch.isfinite(inverse).all().item()):
            continue

        diag_desc = torch.argsort(
            diagonal, descending=True, stable=True
        ).to(device=weight_blocks.device)
        valid_records.append((block_index, hessian, inverse, diag_desc))

    # The reconstruction recurrence is independent across H64 blocks.  A
    # small block batch amortizes its 64 tensor launches without changing the
    # per-block Cholesky, proxy reduction, acceptance order, or legal codes.
    for record_start in range(0, len(valid_records), 4):
        records = valid_records[record_start : record_start + 4]
        block_indices = torch.tensor(
            [record[0] for record in records],
            device=weight_blocks.device,
            dtype=torch.long,
        )
        target_batch = weight_blocks.index_select(0, block_indices)
        scale_batch = scale_blocks.index_select(0, block_indices)
        inverse_batch = torch.stack([record[2] for record in records])
        order_batches = (
            natural.unsqueeze(0).expand(len(records), -1),
            reverse.unsqueeze(0).expand(len(records), -1),
            torch.stack([record[3] for record in records]),
        )
        current_proxies = []
        for local_index, record in enumerate(records):
            target = target_batch[local_index]
            base_residual = target - base_quantized[record[0]]
            current_proxies.append(
                torch.einsum(
                    "oi,ij,oj->o",
                    base_residual,
                    record[1],
                    base_residual,
                )
            )

        for order_batch in order_batches:
            reconstructed_batch = _batched_ordered_hessian_reconstruction(
                target_batch, scale_batch, inverse_batch, order_batch
            )
            for local_index, record in enumerate(records):
                block_index, hessian = record[0], record[1]
                target = target_batch[local_index]
                ordered_reconstruction = reconstructed_batch[local_index]
                block_scale = scale_batch[local_index]
                candidate_mant = (
                    torch.round(
                        ordered_reconstruction.abs() / block_scale * 4.0
                    ).clamp(0.0, 7.0)
                    * 0.25
                )
                candidate_sign = torch.where(
                    candidate_mant == 0.0,
                    0.0,
                    torch.sign(ordered_reconstruction),
                )
                candidate_quantized = candidate_sign * candidate_mant * block_scale
                candidate_residual = target - candidate_quantized
                candidate_proxy = torch.einsum(
                    "oi,ij,oj->o",
                    candidate_residual,
                    hessian,
                    candidate_residual,
                )
                current_proxy = current_proxies[local_index]
                use_candidate = (
                    torch.isfinite(current_proxy)
                    & torch.isfinite(candidate_proxy)
                    & (
                        candidate_proxy
                        < current_proxy * (1.0 - _HESSIAN_GAIN)
                    )
                )
                mask = use_candidate.unsqueeze(-1)
                selected_sign[:, block_index] = torch.where(
                    mask,
                    candidate_sign.to(selected_sign.dtype),
                    selected_sign[:, block_index],
                )
                selected_mant[:, block_index] = torch.where(
                    mask,
                    candidate_mant.to(selected_mant.dtype),
                    selected_mant[:, block_index],
                )
                current_proxies[local_index] = torch.where(
                    use_candidate, candidate_proxy, current_proxy
                )

    selected = dict(base)
    selected["sign"] = selected_sign.reshape_as(base["sign"])
    selected["mant"] = selected_mant.reshape_as(base["mant"])
    return selected


def _bipartite_top4_repair(
    target_windows: torch.Tensor,
    scale_windows: torch.Tensor,
    current_sign: torch.Tensor,
    current_mant: torch.Tensor,
    current_proxy: torch.Tensor,
    diagonal: torch.Tensor,
    proxy_hessian: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the fixed Top-2+Top-2 search in at most 1024-row slices."""
    window_count, row_count = int(target_windows.shape[0]), int(target_windows.shape[1])
    pattern_ids = torch.arange(16, device=target_windows.device, dtype=torch.int64)
    pattern_bits = (
        (pattern_ids.unsqueeze(-1) >> torch.arange(4, device=target_windows.device))
        & 1
    ).to(torch.float32)
    window_index = torch.arange(window_count, device=target_windows.device).reshape(
        window_count, 1, 1, 1
    )

    for row_start in range(0, row_count, _TOP4_ROW_CHUNK):
        row_stop = min(row_start + _TOP4_ROW_CHUNK, row_count)
        target = target_windows[:, row_start:row_stop]
        scale = scale_windows[:, row_start:row_stop]
        sign = current_sign[:, row_start:row_stop]
        mant = current_mant[:, row_start:row_stop]
        proxy = current_proxy[:, row_start:row_stop]
        residual = target - sign.to(torch.float32) * mant.to(torch.float32) * scale

        # Pick two residual-salient coordinates in each 64-wide half.  Stable
        # ordering plus fixed left-then-right layout defines every tie.
        saliency = diagonal.unsqueeze(1) * residual.square()
        saliency = torch.where(
            torch.isfinite(saliency), saliency, torch.zeros_like(saliency)
        )
        left_group = torch.argsort(
            saliency[:, :, :_BLOCK], dim=-1, descending=True, stable=True
        )[:, :, :2]
        right_group = torch.argsort(
            saliency[:, :, _BLOCK:], dim=-1, descending=True, stable=True
        )[:, :, :2] + _BLOCK
        group = torch.cat((left_group, right_group), dim=-1)

        levels = sign.to(torch.float32) * mant.to(torch.float32) * 4.0
        group_levels = torch.gather(levels, 2, group)
        group_residual = torch.gather(residual, 2, group)
        group_scale = torch.gather(scale, 2, group)
        alternate_levels = (
            group_levels + torch.sign(group_residual)
        ).clamp(-7.0, 7.0)
        level_delta = alternate_levels - group_levels

        # Evaluate all 16 current/adjacent patterns using the exact incremental
        # H128 quadratic without materializing candidate 128-wide residuals.
        projected = torch.einsum("wij,woj->woi", proxy_hessian, residual)
        projected_group = torch.gather(projected, 2, group)
        group_hessian = proxy_hessian[
            window_index, group.unsqueeze(-1), group.unsqueeze(-2)
        ]
        residual_delta = (
            -pattern_bits[:, None, None, :]
            * level_delta.unsqueeze(0)
            * group_scale.unsqueeze(0)
            * 0.25
        )
        linear_delta = 2.0 * (
            residual_delta * projected_group.unsqueeze(0)
        ).sum(dim=-1)
        quadratic_delta = torch.einsum(
            "pwok,wokl,pwol->pwo",
            residual_delta,
            group_hessian,
            residual_delta,
        )
        pattern_proxy = proxy.unsqueeze(0) + linear_delta + quadratic_delta
        pattern_proxy = torch.where(
            torch.isfinite(pattern_proxy),
            pattern_proxy,
            torch.full_like(pattern_proxy, float("inf")),
        )
        best_proxy, best_pattern = pattern_proxy.min(dim=0)
        chosen_levels = group_levels + pattern_bits[best_pattern] * level_delta
        scale_valid = torch.isfinite(group_scale).all(dim=-1) & (
            group_scale > 0.0
        ).all(dim=-1)
        accept = (
            valid.unsqueeze(-1)
            & scale_valid
            & torch.isfinite(proxy)
            & (proxy > 0.0)
            & torch.isfinite(best_proxy)
            & (best_proxy < proxy * (1.0 - _TOP4_GAIN))
        )
        accepted_levels = torch.where(
            accept.unsqueeze(-1), chosen_levels, group_levels
        )
        levels.scatter_(2, group, accepted_levels)
        refined_mant = levels.abs() * 0.25
        refined_sign = torch.where(
            refined_mant == 0.0, 0.0, torch.sign(levels)
        )
        current_sign[:, row_start:row_stop] = refined_sign.to(current_sign.dtype)
        current_mant[:, row_start:row_stop] = refined_mant.to(current_mant.dtype)

    return current_sign, current_mant


def _crossblock_hessian_repair(
    weight: torch.Tensor,
    parent: dict[str, torch.Tensor],
    gram: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Repair adjacent blocks, then refine one cross-half four-code group."""
    out_features, channels = int(weight.shape[0]), int(weight.shape[1])
    window_count = channels // _CROSSBLOCK_WINDOW
    if window_count == 0 or tuple(gram.shape) != (
        window_count,
        _CROSSBLOCK_WINDOW,
        _CROSSBLOCK_WINDOW,
    ):
        return parent

    covered_channels = window_count * _CROSSBLOCK_WINDOW
    hessian = gram.to(device=weight.device, dtype=torch.float32)
    diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)
    identity = torch.eye(
        _CROSSBLOCK_WINDOW, dtype=torch.float32, device=weight.device
    )
    valid = (
        torch.isfinite(hessian).all(dim=(-2, -1))
        & torch.isfinite(diagonal).all(dim=-1)
        & (diagonal >= 0.0).all(dim=-1)
    )
    damp = diagonal.mean(dim=-1) * _HESSIAN_DAMPING
    valid &= torch.isfinite(damp)
    regularized = hessian + identity.unsqueeze(0) * (
        damp + 1.0e-8
    ).reshape(-1, 1, 1)
    regularized = torch.where(
        valid.reshape(-1, 1, 1), regularized, identity.unsqueeze(0)
    )
    factor, info = torch.linalg.cholesky_ex(regularized)
    inverse = torch.cholesky_inverse(factor)
    valid &= (info == 0) & torch.isfinite(inverse).all(dim=(-2, -1))
    inverse = torch.where(
        valid.reshape(-1, 1, 1), inverse, identity.unsqueeze(0)
    )

    order = torch.argsort(diagonal, dim=-1, descending=True, stable=True)
    row_index = order.unsqueeze(-1).expand(
        window_count, _CROSSBLOCK_WINDOW, _CROSSBLOCK_WINDOW
    )
    column_index = order.unsqueeze(1).expand(
        window_count, _CROSSBLOCK_WINDOW, _CROSSBLOCK_WINDOW
    )
    ordered_inverse = torch.gather(inverse, 1, row_index)
    ordered_inverse = torch.gather(ordered_inverse, 2, column_index)
    proxy_hessian = torch.where(
        valid.reshape(-1, 1, 1), hessian, identity.unsqueeze(0)
    )
    selected_sign = parent["sign"].reshape(out_features, channels).clone()
    selected_mant = parent["mant"].reshape(out_features, channels).clone()
    parent_sign = parent["sign"].reshape(out_features, channels)
    parent_mant = parent["mant"].reshape(out_features, channels)

    for row_start in range(0, out_features, _CROSSBLOCK_ROW_CHUNK):
        row_stop = min(row_start + _CROSSBLOCK_ROW_CHUNK, out_features)
        row_count = row_stop - row_start
        target_windows = weight[
            row_start:row_stop, :covered_channels
        ].to(torch.float32).reshape(
            row_count, window_count, _CROSSBLOCK_WINDOW
        ).permute(1, 0, 2)
        effective_scale = (
            parent["scale_factor"][row_start:row_stop].to(torch.float32)
            * parent["scale_lv2"][row_start:row_stop].to(torch.float32)
            * parent["scale_lv3"][row_start:row_stop].to(torch.float32)
        )
        scale_windows = effective_scale.expand(
            row_count, channels // _BLOCK, 8, 2, 4
        ).reshape(
            row_count, window_count, _CROSSBLOCK_WINDOW
        ).permute(1, 0, 2)
        parent_sign_windows = parent_sign[
            row_start:row_stop, :covered_channels
        ].reshape(
            row_count, window_count, _CROSSBLOCK_WINDOW
        ).permute(1, 0, 2)
        parent_mant_windows = parent_mant[
            row_start:row_stop, :covered_channels
        ].reshape(
            row_count, window_count, _CROSSBLOCK_WINDOW
        ).permute(1, 0, 2)
        parent_quantized = (
            parent_sign_windows * parent_mant_windows * scale_windows
        )

        value_index = order.unsqueeze(1).expand(
            window_count, row_count, _CROSSBLOCK_WINDOW
        )
        working = torch.gather(target_windows, 2, value_index).clone()
        ordered_scale = torch.gather(scale_windows, 2, value_index)
        ordered_output = torch.empty_like(working)
        for position in range(_CROSSBLOCK_WINDOW):
            value = working[:, :, position]
            column_scale = ordered_scale[:, :, position]
            mant = (
                torch.round(value.abs() / column_scale * 4.0)
                .clamp(0.0, 7.0)
                * 0.25
            )
            reconstructed = torch.sign(value) * mant * column_scale
            ordered_output[:, :, position] = reconstructed
            if position + 1 < _CROSSBLOCK_WINDOW:
                denominator = ordered_inverse[
                    :, position, position
                ].clamp_min(1.0e-12)
                propagated = (value - reconstructed) / denominator.unsqueeze(-1)
                working[:, :, position + 1 :].addcmul_(
                    propagated.unsqueeze(-1),
                    ordered_inverse[
                        :, position, position + 1 :
                    ].unsqueeze(1),
                    value=-1.0,
                )

        reconstruction = torch.empty_like(ordered_output)
        reconstruction.scatter_(2, value_index, ordered_output)
        candidate_mant = (
            torch.round(reconstruction.abs() / scale_windows * 4.0)
            .clamp(0.0, 7.0)
            * 0.25
        )
        candidate_sign = torch.where(
            candidate_mant == 0.0, 0.0, torch.sign(reconstruction)
        )
        candidate_quantized = candidate_sign * candidate_mant * scale_windows
        parent_residual = target_windows - parent_quantized
        candidate_residual = target_windows - candidate_quantized
        parent_proxy = torch.einsum(
            "woi,wij,woj->wo", parent_residual, proxy_hessian, parent_residual
        )
        candidate_proxy = torch.einsum(
            "woi,wij,woj->wo",
            candidate_residual,
            proxy_hessian,
            candidate_residual,
        )
        use_candidate = (
            valid.unsqueeze(-1)
            & torch.isfinite(parent_proxy)
            & torch.isfinite(candidate_proxy)
            & (candidate_proxy < parent_proxy * (1.0 - _HESSIAN_GAIN))
        )
        mask = use_candidate.unsqueeze(-1)
        current_sign = torch.where(
            mask, candidate_sign.to(parent_sign_windows.dtype), parent_sign_windows
        )
        current_mant = torch.where(
            mask, candidate_mant.to(parent_mant_windows.dtype), parent_mant_windows
        )
        current_proxy = torch.where(
            use_candidate, candidate_proxy, parent_proxy
        )
        # End the V26 parent working-set lifetime before Top-4 allocates its
        # residual projection and pattern buffers.  The three tensors retained
        # below are the complete parent decision needed by the fixed refinement.
        del (
            working,
            ordered_scale,
            ordered_output,
            reconstruction,
            candidate_quantized,
            parent_quantized,
            parent_residual,
            candidate_residual,
            candidate_sign,
            candidate_mant,
            value_index,
        )
        refined_sign, refined_mant = _bipartite_top4_repair(
            target_windows,
            scale_windows,
            current_sign,
            current_mant,
            current_proxy,
            diagonal,
            proxy_hessian,
            valid,
        )
        selected_sign[
            row_start:row_stop, :covered_channels
        ] = refined_sign.to(parent_sign_windows.dtype).permute(1, 0, 2).reshape(
            row_count, covered_channels
        )
        selected_mant[
            row_start:row_stop, :covered_channels
        ] = refined_mant.to(parent_mant_windows.dtype).permute(1, 0, 2).reshape(
            row_count, covered_channels
        )

    selected = dict(parent)
    selected["sign"] = selected_sign.reshape_as(parent["sign"])
    selected["mant"] = selected_mant.reshape_as(parent["mant"])
    return selected


def _hessian_graph_matching(
    hessian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one deterministic maximum-correlation perfect matching per H64."""
    block_count = int(hessian.shape[0])
    matching = torch.zeros(block_count, _BLOCK // 2, 2, dtype=torch.int64)
    diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)
    valid = (
        torch.isfinite(hessian).all(dim=(-2, -1))
        & torch.isfinite(diagonal).all(dim=-1)
        & (diagonal >= 0.0).all(dim=-1)
    )
    denominator = torch.sqrt(
        (diagonal.unsqueeze(-1) + _MIN_SCALE)
        * (diagonal.unsqueeze(-2) + _MIN_SCALE)
    )
    correlation = hessian.abs() / denominator
    correlation.masked_fill_(
        torch.eye(_BLOCK, dtype=torch.bool).unsqueeze(0), float("-inf")
    )

    for block_index in range(block_count):
        if not bool(valid[block_index].item()):
            continue
        order = torch.argsort(
            correlation[block_index].reshape(-1),
            descending=True,
            stable=True,
        )
        used = [False] * _BLOCK
        edges: list[tuple[int, int]] = []
        for flat_index in order.tolist():
            left = flat_index // _BLOCK
            right = flat_index % _BLOCK
            if left == right or used[left] or used[right]:
                continue
            edges.append((left, right))
            used[left] = True
            used[right] = True
            if len(edges) == _BLOCK // 2:
                break
        if len(edges) != _BLOCK // 2:
            valid[block_index] = False
            continue
        matching[block_index] = torch.tensor(edges, dtype=torch.int64)
    return matching, valid


def _hessian_edge_exchange(
    weight: torch.Tensor,
    parent: dict[str, torch.Tensor],
    aggregate_gram: torch.Tensor,
    shard_grams: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply one graph-matched two-code move with five-shard unanimity."""
    out_features, channels = int(weight.shape[0]), int(weight.shape[1])
    block_count = channels // _BLOCK
    expected_aggregate = (block_count, _BLOCK, _BLOCK)
    expected_shards = (_CALIBRATION_SHARDS, block_count, _BLOCK, _BLOCK)
    if tuple(aggregate_gram.shape) != expected_aggregate or tuple(
        shard_grams.shape
    ) != expected_shards:
        return parent

    matching, valid = _hessian_graph_matching(aggregate_gram)
    if not bool(valid.any().item()):
        return parent

    device = weight.device
    hessian = aggregate_gram.to(device=device, dtype=torch.float32)
    shards = shard_grams.to(device=device, dtype=torch.float32)
    matching = matching.to(device=device)
    valid = valid.to(device=device)
    edge_count = _BLOCK // 2
    directions = torch.tensor(
        ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)),
        device=device,
        dtype=torch.float32,
    )
    block_index = torch.arange(block_count, device=device).reshape(-1, 1, 1, 1)
    edge_hessian = hessian[
        block_index,
        matching.unsqueeze(-1),
        matching.unsqueeze(-2),
    ]

    target_blocks = weight.to(torch.float32).reshape(
        out_features, block_count, _BLOCK
    ).permute(1, 0, 2)
    parent_sign = parent["sign"].reshape(out_features, channels)
    parent_mant = parent["mant"].reshape(out_features, channels)
    selected_sign = parent_sign.clone()
    selected_mant = parent_mant.clone()
    effective_scale = (
        parent["scale_factor"].to(torch.float32)
        * parent["scale_lv2"].to(torch.float32)
        * parent["scale_lv3"].to(torch.float32)
    ).expand_as(parent["mant"]).reshape(
        out_features, block_count, _BLOCK
    ).permute(1, 0, 2)
    sign_blocks = parent_sign.reshape(
        out_features, block_count, _BLOCK
    ).permute(1, 0, 2)
    mant_blocks = parent_mant.reshape(
        out_features, block_count, _BLOCK
    ).permute(1, 0, 2)

    for row_start in range(0, out_features, _EDGE_ROW_CHUNK):
        row_stop = min(row_start + _EDGE_ROW_CHUNK, out_features)
        row_count = row_stop - row_start
        target = target_blocks[:, row_start:row_stop]
        scale = effective_scale[:, row_start:row_stop]
        levels = (
            sign_blocks[:, row_start:row_stop].to(torch.float32)
            * mant_blocks[:, row_start:row_stop].to(torch.float32)
            * 4.0
        )
        residual = target - levels * scale * 0.25
        projection = torch.einsum("bij,brj->bri", hessian, residual)
        current_proxy = torch.einsum("bri,bri->br", residual, projection)

        gather_index = matching.reshape(block_count, 1, edge_count * 2).expand(
            block_count, row_count, edge_count * 2
        )
        edge_levels = torch.gather(levels, 2, gather_index).reshape(
            block_count, row_count, edge_count, 2
        )
        edge_scale = torch.gather(scale, 2, gather_index).reshape(
            block_count, row_count, edge_count, 2
        )
        edge_projection = torch.gather(projection, 2, gather_index).reshape(
            block_count, row_count, edge_count, 2
        )
        candidate_levels = edge_levels.unsqueeze(-2) + directions.reshape(
            1, 1, 1, 4, 2
        )
        pattern_valid = (candidate_levels >= -7.0).all(dim=-1) & (
            candidate_levels <= 7.0
        ).all(dim=-1)
        pattern_valid &= torch.isfinite(edge_scale).all(dim=-1).unsqueeze(-1)
        pattern_valid &= (edge_scale > 0.0).all(dim=-1).unsqueeze(-1)

        pattern_deltas: list[torch.Tensor] = []
        for pattern_index in range(4):
            residual_delta = (
                -directions[pattern_index].reshape(1, 1, 1, 2)
                * edge_scale
                * 0.25
            )
            linear_delta = 2.0 * (residual_delta * edge_projection).sum(dim=-1)
            quadratic_delta = torch.einsum(
                "bren,benm,brem->bre",
                residual_delta,
                edge_hessian,
                residual_delta,
            )
            delta = linear_delta + quadratic_delta
            pattern_deltas.append(
                torch.where(
                    pattern_valid[..., pattern_index] & torch.isfinite(delta),
                    delta,
                    torch.full_like(delta, float("inf")),
                )
            )
        aggregate_delta = torch.stack(pattern_deltas, dim=-1)
        best_delta, best_flat = aggregate_delta.reshape(
            block_count, row_count, edge_count * 4
        ).min(dim=-1)
        best_edge = torch.div(best_flat, 4, rounding_mode="floor")
        best_pattern = best_flat.remainder(4)
        candidate_proxy = current_proxy + best_delta
        aggregate_accept = (
            valid.unsqueeze(-1)
            & torch.isfinite(current_proxy)
            & (current_proxy > 0.0)
            & torch.isfinite(candidate_proxy)
            & (candidate_proxy < current_proxy * (1.0 - _HESSIAN_GAIN))
        )

        edge_table = matching.unsqueeze(1).expand(
            block_count, row_count, edge_count, 2
        )
        chosen_pair = torch.gather(
            edge_table,
            2,
            best_edge.unsqueeze(-1).unsqueeze(-1).expand(
                block_count, row_count, 1, 2
            ),
        ).squeeze(2)
        chosen_direction = directions[best_pattern]
        chosen_scale = torch.gather(scale, 2, chosen_pair)
        chosen_residual_delta = -chosen_direction * chosen_scale * 0.25

        shard_index = torch.arange(
            _CALIBRATION_SHARDS, device=device
        ).reshape(-1, 1, 1, 1)
        chosen_block = torch.arange(block_count, device=device).reshape(
            1, -1, 1, 1
        )
        shard_rows = shards[
            shard_index,
            chosen_block,
            chosen_pair.unsqueeze(0),
        ]
        shard_edge_projection = torch.einsum(
            "sbrik,brk->sbri", shard_rows, residual
        )
        shard_edge_hessian = torch.gather(
            shard_rows,
            4,
            chosen_pair.unsqueeze(0).unsqueeze(-2).expand(
                _CALIBRATION_SHARDS, block_count, row_count, 2, 2
            ),
        )
        shard_residual_delta = chosen_residual_delta.unsqueeze(0)
        shard_delta = 2.0 * (
            shard_residual_delta * shard_edge_projection
        ).sum(dim=-1) + torch.einsum(
            "sbri,sbrij,sbrj->sbr",
            shard_residual_delta,
            shard_edge_hessian,
            shard_residual_delta,
        )
        unanimous = torch.isfinite(shard_delta).all(dim=0) & (
            shard_delta < 0.0
        ).all(dim=0)
        accept = aggregate_accept & unanimous

        current_pair_levels = torch.gather(levels, 2, chosen_pair)
        accepted_pair_levels = torch.where(
            accept.unsqueeze(-1),
            current_pair_levels + chosen_direction,
            current_pair_levels,
        )
        levels.scatter_(2, chosen_pair, accepted_pair_levels)
        refined_mant = levels.abs() * 0.25
        refined_sign = torch.where(
            refined_mant == 0.0, 0.0, torch.sign(levels)
        )
        selected_sign[row_start:row_stop] = refined_sign.permute(
            1, 0, 2
        ).reshape(row_count, channels).to(selected_sign.dtype)
        selected_mant[row_start:row_stop] = refined_mant.permute(
            1, 0, 2
        ).reshape(row_count, channels).to(selected_mant.dtype)

    selected = dict(parent)
    selected["sign"] = selected_sign.reshape_as(parent["sign"])
    selected["mant"] = selected_mant.reshape_as(parent["mant"])
    return selected


def _linear_parent_state_valid(state: Any, channels: int) -> bool:
    if not isinstance(state, dict) or state.get("schema") not in (
        "aonly-pow2-v1",
        _LINEAR_OUTPUT_SCHEMA,
    ):
        return False
    if float(state.get("min_proxy_gain", -1.0)) != _LINEAR_GAIN:
        return False
    scale = state.get("precondition_scale")
    importance = state.get("activation_importance")
    if type(scale) is not torch.Tensor or type(importance) is not torch.Tensor:
        return False
    if scale.numel() != channels or importance.numel() != channels:
        return False
    return bool(torch.isfinite(scale).all().item()) and bool(
        torch.isfinite(importance).all().item()
    ) and bool((scale > 0.0).all().item())


def _linear_output_state_valid(state: Any, channels: int) -> bool:
    if not _linear_parent_state_valid(state, channels):
        return False
    if state.get("schema") != _LINEAR_OUTPUT_SCHEMA or not bool(
        state.get("output_valid", False)
    ):
        return False
    hessian = state.get("output_hessian")
    center_map = state.get("output_center_map")
    window_count = channels // _LINEAR_OUTPUT_WINDOW
    expected = (
        window_count,
        _LINEAR_OUTPUT_WINDOW,
        _LINEAR_OUTPUT_WINDOW,
    )
    if type(hessian) is not torch.Tensor or type(center_map) is not torch.Tensor:
        return False
    if tuple(hessian.shape) != expected or tuple(center_map.shape) != expected:
        return False
    return bool(torch.isfinite(hessian).all().item()) and bool(
        torch.isfinite(center_map).all().item()
    )


def _linear_output_statistics(
    transformed_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    output_channels, channels = transformed_weight.shape
    if channels % _LINEAR_OUTPUT_WINDOW:
        return None, None
    block_count = channels // _LINEAR_OUTPUT_WINDOW
    weight_blocks = transformed_weight.reshape(
        output_channels, block_count, _LINEAR_OUTPUT_WINDOW
    ).permute(1, 0, 2)
    quantized_blocks = quantized_weight.reshape(
        output_channels, block_count, _LINEAR_OUTPUT_WINDOW
    ).permute(1, 0, 2)
    hessian = torch.matmul(quantized_blocks.transpose(-1, -2), quantized_blocks)
    cross = torch.matmul(weight_blocks.transpose(-1, -2), quantized_blocks)
    diagonal_mean = torch.diagonal(hessian, dim1=-2, dim2=-1).mean(dim=-1)
    damping = (diagonal_mean * _LINEAR_OUTPUT_DAMPING).clamp_min(
        _LINEAR_OUTPUT_FLOOR
    )
    identity = torch.eye(
        _LINEAR_OUTPUT_WINDOW, device=hessian.device, dtype=torch.float32
    )
    regularized = hessian + damping[:, None, None] * identity
    factor, info = torch.linalg.cholesky_ex(regularized)
    if bool((info != 0).any().item()) or not bool(
        torch.isfinite(factor).all().item()
    ):
        return None, None
    center_map = torch.cholesky_solve(
        cross.transpose(-1, -2), factor
    ).transpose(-1, -2)
    if not bool(torch.isfinite(center_map).all().item()):
        return None, None
    return regularized, center_map


def _linear_output_select(
    transformed: torch.Tensor,
    parent: dict[str, torch.Tensor],
    hessian: torch.Tensor,
    center_map: torch.Tensor,
) -> dict[str, torch.Tensor]:
    channels = int(transformed.shape[-1])
    block_count = channels // _LINEAR_OUTPUT_WINDOW
    h64_count = channels // _BLOCK
    rows = transformed.numel() // channels
    source_blocks = transformed.reshape(
        rows, block_count, _LINEAR_OUTPUT_WINDOW
    )
    center = torch.matmul(source_blocks.unsqueeze(-2), center_map).squeeze(-2)
    output_importance = torch.diagonal(
        hessian, dim1=-2, dim2=-1
    ).clamp_min(_LINEAR_OUTPUT_FLOOR).reshape(channels)
    candidate = _quantize_sensitive(
        center.reshape_as(transformed), output_importance, _LINEAR_GAIN
    )
    parent_value = _dequantize_hif4(parent).reshape(
        rows, block_count, _LINEAR_OUTPUT_WINDOW
    )
    candidate_value = _dequantize_hif4(candidate).reshape(
        rows, block_count, _LINEAR_OUTPUT_WINDOW
    )
    parent_delta = parent_value - center
    candidate_delta = candidate_value - center
    parent_risk = torch.einsum(
        "rbi,bij,rbj->rb", parent_delta, hessian, parent_delta
    )
    candidate_risk = torch.einsum(
        "rbi,bij,rbj->rb", candidate_delta, hessian, candidate_delta
    )
    different = torch.zeros(
        (rows, block_count), device=transformed.device, dtype=torch.bool
    )
    for key in parent:
        different |= (parent[key].reshape(rows, block_count, -1) != candidate[key].reshape(
            rows, block_count, -1
        )).any(dim=-1)
    accept = (
        different
        & torch.isfinite(parent_risk)
        & torch.isfinite(candidate_risk)
        & (candidate_risk < parent_risk * (1.0 - 1.0e-6))
    )
    prefix_shape = transformed.shape[:-1]
    block_mask = accept.repeat_interleave(
        _LINEAR_OUTPUT_WINDOW // _BLOCK, dim=-1
    ).reshape(*prefix_shape, h64_count, 1, 1, 1)
    return {
        key: torch.where(block_mask, candidate[key], parent[key])
        for key in parent
    }


def _linear_output_all_block_one_coordinate_code_descent(
    transformed: torch.Tensor,
    parent: dict[str, torch.Tensor],
    hessian: torch.Tensor,
    center_map: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Take dependent legal output-risk steps in every paired-H64 window."""
    channels = int(transformed.shape[-1])
    block_count = channels // _LINEAR_OUTPUT_WINDOW
    rows = transformed.numel() // channels
    source_blocks = transformed.reshape(
        rows, block_count, _LINEAR_OUTPUT_WINDOW
    )
    center = torch.matmul(source_blocks.unsqueeze(-2), center_map).squeeze(-2)
    try:
        scale = (
            parent["scale_factor"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
            * 0.25
        ).expand_as(parent["mant"]).reshape(
            rows, block_count, _LINEAR_OUTPUT_WINDOW
        )
        raw_level = (
            parent["sign"].to(torch.float32)
            * parent["mant"].to(torch.float32)
            * 4.0
        ).reshape(rows, block_count, _LINEAR_OUTPUT_WINDOW)
    except (KeyError, RuntimeError, ValueError):
        return parent
    level = torch.round(raw_level)
    if not (
        bool(torch.isfinite(center).all().item())
        and bool(torch.isfinite(scale).all().item())
        and bool((scale > 0.0).all().item())
        and bool(torch.isfinite(raw_level).all().item())
        and bool(((raw_level - level).abs() <= 1.0e-6).all().item())
        and bool(((level >= -7.0) & (level <= 7.0)).all().item())
    ):
        return parent
    hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
    diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)
    if not (
        bool(torch.isfinite(hessian).all().item())
        and bool((diagonal > 0.0).all().item())
    ):
        return parent

    selected_level = level.clone()
    accepted = torch.zeros(
        (rows, block_count), device=transformed.device, dtype=torch.bool
    )
    block_offsets = (
        torch.arange(block_count, device=transformed.device, dtype=torch.long)
        * _LINEAR_OUTPUT_WINDOW
    ).unsqueeze(0)
    hessian_columns = hessian.transpose(-1, -2).contiguous().reshape(
        block_count * _LINEAR_OUTPUT_WINDOW, _LINEAR_OUTPUT_WINDOW
    )
    for row_start in range(0, rows, _LINEAR_OUTPUT_CODE_ROW_CHUNK):
        row_stop = min(rows, row_start + _LINEAR_OUTPUT_CODE_ROW_CHUNK)
        row_count = row_stop - row_start
        row_levels = selected_level[row_start:row_stop]
        row_scales = scale[row_start:row_stop]
        residual = row_levels * row_scales - center[row_start:row_stop]
        gradient = torch.einsum("rbi,bij->rbj", residual, hessian)
        cumulative_delta = torch.zeros(
            (row_count, block_count),
            device=transformed.device,
            dtype=torch.float32,
        )
        row_index = torch.arange(
            row_count, device=transformed.device
        ).unsqueeze(-1)
        block_index = torch.arange(
            block_count, device=transformed.device
        ).unsqueeze(0)
        for _ in range(_LINEAR_OUTPUT_CODE_DESCENT_STEPS):
            candidate_levels = torch.round(
                row_levels
                - gradient
                / diagonal.clamp_min(1.0e-24).unsqueeze(0)
                / row_scales
            ).clamp(-7.0, 7.0)
            reconstruction_delta = (
                candidate_levels - row_levels
            ) * row_scales
            candidate_delta = (
                2.0 * gradient * reconstruction_delta
                + diagonal.unsqueeze(0) * reconstruction_delta.square()
            )
            candidate_delta = torch.where(
                torch.isfinite(candidate_levels)
                & torch.isfinite(candidate_delta)
                & (candidate_levels != row_levels),
                candidate_delta,
                torch.full_like(candidate_delta, torch.inf),
            )
            best_delta, code_index = candidate_delta.min(dim=-1)
            accept = torch.isfinite(best_delta) & (best_delta < 0.0)
            current = row_levels[row_index, block_index, code_index]
            proposed = candidate_levels[
                row_index, block_index, code_index
            ]
            code_delta = torch.where(
                accept,
                proposed - current,
                torch.zeros_like(best_delta),
            )
            row_levels[row_index, block_index, code_index] = current + code_delta
            value_delta = (
                code_delta
                * row_scales[row_index, block_index, code_index]
            )
            selected_column = hessian_columns.index_select(
                0, (block_offsets + code_index).reshape(-1)
            ).reshape(row_count, block_count, _LINEAR_OUTPUT_WINDOW)
            gradient = gradient + selected_column * value_delta.unsqueeze(-1)
            cumulative_delta = cumulative_delta + torch.where(
                accept, best_delta, torch.zeros_like(best_delta)
            )
        accepted[row_start:row_stop] = (
            torch.isfinite(cumulative_delta) & (cumulative_delta < 0.0)
        )
    if not bool(accepted.any().item()):
        return parent
    selected = dict(parent)
    selected["mant"] = (selected_level.abs() * 0.25).to(
        parent["mant"].dtype
    ).reshape_as(parent["mant"])
    selected["sign"] = torch.where(
        selected_level == 0.0,
        torch.zeros_like(selected_level),
        torch.sign(selected_level),
    ).to(parent["sign"].dtype).reshape_as(parent["sign"])
    return selected


def _linear_output_block_scale_refine(
    transformed: torch.Tensor,
    parent: dict[str, torch.Tensor],
    hessian: torch.Tensor,
    center_map: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Project each H64's output-optimal scalar onto the legal E6M2 grid."""
    channels = int(transformed.shape[-1])
    block_count = channels // _BLOCK
    rows = transformed.numel() // channels
    source_blocks = transformed.reshape(rows, block_count, _BLOCK)
    center = torch.matmul(source_blocks.unsqueeze(-2), center_map).squeeze(-2)
    try:
        pattern = (
            parent["sign"].to(torch.float32)
            * parent["mant"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
        ).reshape(rows, block_count, _BLOCK)
        current_scale = parent["scale_factor"].to(torch.float32).reshape(
            rows, block_count
        )
    except (KeyError, RuntimeError, ValueError):
        return parent
    hp = torch.einsum("rbi,bij->rbj", pattern, hessian)
    denominator = (hp * pattern).sum(dim=-1)
    numerator = (hp * center).sum(dim=-1)
    optimum = numerator / denominator.clamp_min(1.0e-24)
    candidate_scales = torch.stack(
        (
            current_scale,
            _floor_e6m2(optimum),
            _ceil_e6m2(optimum),
        ),
        dim=-1,
    )
    residual = (
        pattern.unsqueeze(-2) * candidate_scales.unsqueeze(-1)
        - center.unsqueeze(-2)
    )
    risk = torch.einsum("rbsi,bij,rbsj->rbs", residual, hessian, residual)
    best_risk, best_index = risk.min(dim=-1)
    current_risk = risk[..., 0]
    best_scale = candidate_scales.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    accept = (
        torch.isfinite(best_risk)
        & torch.isfinite(current_risk)
        & torch.isfinite(best_scale)
        & (best_scale > 0.0)
        & (best_risk < current_risk)
    )
    if not bool(accept.any().item()):
        return parent
    selected = dict(parent)
    selected["scale_factor"] = torch.where(
        accept, best_scale, current_scale
    ).to(parent["scale_factor"].dtype).reshape_as(parent["scale_factor"])
    return selected


def _linear_output_pair_scale_refine(
    transformed: torch.Tensor,
    parent: dict[str, torch.Tensor],
    hessian: torch.Tensor,
    center_map: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Jointly refit both legal outer scales in every H128 output window."""
    channels = int(transformed.shape[-1])
    if channels % _LINEAR_OUTPUT_WINDOW or _LINEAR_OUTPUT_WINDOW != 2 * _BLOCK:
        return parent
    window_count = channels // _LINEAR_OUTPUT_WINDOW
    rows = transformed.numel() // channels
    try:
        source = transformed.reshape(rows, window_count, _LINEAR_OUTPUT_WINDOW)
        center = torch.matmul(source.unsqueeze(-2), center_map).squeeze(-2)
        pattern = (
            parent["sign"].to(torch.float32)
            * parent["mant"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
        ).reshape(rows, window_count, 2, _BLOCK)
        current_scale = parent["scale_factor"].to(torch.float32).reshape(
            rows, window_count, 2
        )
    except (KeyError, RuntimeError, ValueError):
        return parent
    if not (
        bool(torch.isfinite(center).all().item())
        and bool(torch.isfinite(pattern).all().item())
        and bool(torch.isfinite(current_scale).all().item())
        and bool((current_scale > 0.0).all().item())
    ):
        return parent

    hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
    h00 = hessian[:, :_BLOCK, :_BLOCK]
    h01 = hessian[:, :_BLOCK, _BLOCK:]
    h10 = hessian[:, _BLOCK:, :_BLOCK]
    h11 = hessian[:, _BLOCK:, _BLOCK:]
    p0, p1 = pattern[:, :, 0], pattern[:, :, 1]
    c0, c1 = center[:, :, :_BLOCK], center[:, :, _BLOCK:]
    try:
        m00 = torch.einsum("rbi,bij,rbj->rb", p0, h00, p0)
        m01 = torch.einsum("rbi,bij,rbj->rb", p0, h01, p1)
        m11 = torch.einsum("rbi,bij,rbj->rb", p1, h11, p1)
        b0 = torch.einsum("rbi,bij,rbj->rb", p0, h00, c0)
        b0 = b0 + torch.einsum("rbi,bij,rbj->rb", p0, h01, c1)
        b1 = torch.einsum("rbi,bij,rbj->rb", p1, h10, c0)
        b1 = b1 + torch.einsum("rbi,bij,rbj->rb", p1, h11, c1)
        determinant = m00 * m11 - m01.square()
        valid_system = (
            torch.isfinite(m00)
            & torch.isfinite(m01)
            & torch.isfinite(m11)
            & torch.isfinite(b0)
            & torch.isfinite(b1)
            & (m00 > 1.0e-24)
            & (m11 > 1.0e-24)
            & torch.isfinite(determinant)
            & (determinant > 1.0e-10 * m00 * m11)
        )
        safe_determinant = torch.where(
            valid_system, determinant, torch.ones_like(determinant)
        )
        optimum0 = (b0 * m11 - b1 * m01) / safe_determinant
        optimum1 = (b1 * m00 - b0 * m01) / safe_determinant
        floor0, ceil0 = _floor_e6m2(optimum0), _ceil_e6m2(optimum0)
        floor1, ceil1 = _floor_e6m2(optimum1), _ceil_e6m2(optimum1)
        candidates = torch.stack(
            (
                current_scale,
                torch.stack((floor0, floor1), dim=-1),
                torch.stack((floor0, ceil1), dim=-1),
                torch.stack((ceil0, floor1), dim=-1),
                torch.stack((ceil0, ceil1), dim=-1),
            ),
            dim=-2,
        ).to(torch.bfloat16).to(torch.float32)

        parent_value = torch.cat(
            (
                current_scale[:, :, :1] * p0,
                current_scale[:, :, 1:] * p1,
            ),
            dim=-1,
        )
        parent_residual = parent_value - center
        gradient = torch.einsum("rbi,bij->rbj", parent_residual, hessian)
        g0 = (gradient[:, :, :_BLOCK] * p0).sum(dim=-1)
        g1 = (gradient[:, :, _BLOCK:] * p1).sum(dim=-1)
        delta = candidates - current_scale.unsqueeze(-2)
        d0, d1 = delta[..., 0], delta[..., 1]
        risk_delta = (
            2.0 * (d0 * g0.unsqueeze(-1) + d1 * g1.unsqueeze(-1))
            + d0.square() * m00.unsqueeze(-1)
            + 2.0 * d0 * d1 * m01.unsqueeze(-1)
            + d1.square() * m11.unsqueeze(-1)
        )
        candidate_valid = (
            valid_system.unsqueeze(-1)
            & torch.isfinite(candidates).all(dim=-1)
            & (candidates > 0.0).all(dim=-1)
            & torch.isfinite(risk_delta)
        )
        risk_delta = torch.where(
            candidate_valid,
            risk_delta,
            torch.full_like(risk_delta, torch.inf),
        )
        best_risk, best_index = risk_delta.min(dim=-1)
        best_scale = candidates.gather(
            -2,
            best_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2),
        ).squeeze(-2)
        accept = (
            torch.isfinite(best_risk)
            & (best_risk < 0.0)
            & (best_scale != current_scale).any(dim=-1)
        )
        selected_scale = torch.where(
            accept.unsqueeze(-1), best_scale, current_scale
        )
    except (RuntimeError, ValueError, OverflowError):
        return parent

    selected = dict(parent)
    selected["scale_factor"] = selected_scale.reshape_as(
        parent["scale_factor"]
    ).to(parent["scale_factor"].dtype)
    return selected


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    weight = dequantize_nvfp4(weight_quant, weight_scale).to(torch.float32)
    if weight.ndim != 2 or int(weight.shape[1]) % _BLOCK:
        raise ValueError("Linear Weight must be 2D with a 64-aligned input size")
    if not bool(torch.isfinite(weight).all().item()):
        weight = torch.nan_to_num(weight)
    channels = int(weight.shape[1])
    activations: list[torch.Tensor] = []
    try:
        for pair in calib_activation_list:
            activations.append(dequantize_nvfp4(*pair).to(torch.float32))
    except (TypeError, ValueError, RuntimeError):
        activations = []

    precondition, activation_importance, statistics_valid = _aonly_statistics(
        activations, channels
    )
    transformed_weight = weight * precondition.to(weight.device)
    parent = _quantize_hif4(transformed_weight)
    gram, shard_grams = (
        _block_grams(activations, precondition)
        if statistics_valid
        else (None, None)
    )
    crossblock_gram = (
        _crossblock_gram(activations, precondition, gram)
        if statistics_valid and gram is not None
        else None
    )
    if gram is None:
        weight_params = parent
    else:
        diagonal_importance = torch.diagonal(gram, dim1=-2, dim2=-1).reshape(-1)
        denominator = diagonal_importance.mean()
        if bool(torch.isfinite(denominator).item()) and float(denominator.item()) > 0:
            diagonal_importance = (diagonal_importance / denominator).clamp(
                1.0 / 16.0, 16.0
            )
            base = _quantize_hif4(
                transformed_weight,
                importance=diagonal_importance,
                factors=_SENSITIVE_FACTORS,
            )
        else:
            base = parent
        weight_params = _multiorder_hessian(transformed_weight, base, gram)
    if crossblock_gram is not None:
        weight_params = _crossblock_hessian_repair(
            transformed_weight, weight_params, crossblock_gram
        )
    if gram is not None and shard_grams is not None:
        weight_params = _hessian_edge_exchange(
            transformed_weight, weight_params, gram, shard_grams
        )

    output_hessian: torch.Tensor | None = None
    output_center_map: torch.Tensor | None = None
    try:
        quantized_weight = _dequantize_hif4(weight_params).reshape_as(
            transformed_weight
        )
        output_hessian, output_center_map = _linear_output_statistics(
            transformed_weight, quantized_weight
        )
    except (RuntimeError, ValueError):
        output_hessian, output_center_map = None, None

    activation_state: dict[str, Any] = {
        "schema": _LINEAR_OUTPUT_SCHEMA,
        "precondition_scale": precondition.cpu(),
        "activation_importance": activation_importance.cpu(),
        "min_proxy_gain": _LINEAR_GAIN,
        "output_valid": output_hessian is not None and output_center_map is not None,
    }
    if output_hessian is not None and output_center_map is not None:
        activation_state["output_hessian"] = output_hessian.cpu()
        activation_state["output_center_map"] = output_center_map.cpu()

    return {
        "weight_params": weight_params,
        "activation_state": activation_state,
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    activation = dequantize_nvfp4(activation_quant, activation_scale).to(
        torch.float32
    )
    channels = int(activation.shape[-1])
    if not _linear_parent_state_valid(activation_state, channels):
        return _quantize_hif4_v1(torch.nan_to_num(activation))
    precondition = activation_state["precondition_scale"].to(
        device=activation.device, dtype=torch.float32
    )
    importance = activation_state["activation_importance"].to(
        device=activation.device, dtype=torch.float32
    )
    transformed = torch.nan_to_num(activation / precondition)
    parent = _quantize_sensitive(transformed, importance, _LINEAR_GAIN)
    if not _linear_output_state_valid(activation_state, channels):
        return parent
    hessian = activation_state["output_hessian"].to(
        device=activation.device, dtype=torch.float32
    )
    center_map = activation_state["output_center_map"].to(
        device=activation.device, dtype=torch.float32
    )
    try:
        selected = _linear_output_select(
            transformed, parent, hessian, center_map
        )
    except (RuntimeError, ValueError):
        return parent
    try:
        selected = _linear_output_all_block_one_coordinate_code_descent(
            transformed, selected, hessian, center_map
        )
    except (RuntimeError, ValueError, OverflowError):
        return selected
    return selected


def _attention_fallback_state(
    q_num_heads: int, kv_num_heads: int, head_dim: int
) -> dict[str, Any]:
    q_ones = torch.ones(q_num_heads * head_dim, dtype=torch.float32)
    kv_ones = torch.ones(kv_num_heads * head_dim, dtype=torch.float32)
    common = {
        "selected_alpha": _ATTENTION_ALPHA,
        "rotation_group_size": 0,
        "rotation_reason": "stable-v3-basis-no-rotation",
        "min_proxy_gain": _ATTENTION_GAIN,
        "fallback_to_parent": True,
    }
    return {
        "q_state": {
            "smooth_scale": q_ones,
            "importance": q_ones.clone(),
            "proxy_role": "q-error-weighted-by-k-second-moment",
            **common,
        },
        "k_state": {
            "smooth_scale": kv_ones,
            "importance": kv_ones.clone(),
            "proxy_role": "k-error-weighted-by-gqa-q-second-moment",
            **common,
        },
        "v_state": {
            "importance": kv_ones.clone(),
            "min_proxy_gain": _ATTENTION_GAIN,
            "proxy_role": "v-energy-stable-upper-bound",
            "pair_schema": _V_PAIR_SCHEMA,
            "pair_rho": torch.zeros(kv_num_heads, dtype=torch.float32),
            "pair_valid": torch.zeros(kv_num_heads, dtype=torch.bool),
            "pair_lag2_schema": _V_LAG2_SCHEMA,
            "pair_rho2": torch.zeros(kv_num_heads, dtype=torch.float32),
            "pair_lag2_valid": torch.zeros(kv_num_heads, dtype=torch.bool),
            "profile_schema": _V_PROFILE_SCHEMA,
            "profile_lengths": torch.empty(0, dtype=torch.int64),
            "profile_unary": torch.empty(
                0, kv_num_heads, _V_PROFILE_MAX_SEQUENCE, dtype=torch.float32
            ),
            "profile_edge": torch.empty(
                0, kv_num_heads, _V_PROFILE_MAX_SEQUENCE - 1, dtype=torch.float32
            ),
            "profile_valid": torch.empty(0, kv_num_heads, dtype=torch.bool),
            "lowrank_schema": _V_LOWRANK_SCHEMA,
            "lowrank_gram": torch.zeros(
                kv_num_heads,
                _V_LOWRANK_RANK,
                _V_LOWRANK_RANK,
                dtype=torch.float32,
            ),
            "lowrank_delta": torch.zeros(kv_num_heads, dtype=torch.float32),
            "lowrank_valid": torch.zeros(kv_num_heads, dtype=torch.bool),
            "nystrom_schema": _V_NYSTROM_SCHEMA,
            "nystrom_lengths": torch.empty(0, dtype=torch.int64),
            "nystrom_factors": (),
            "nystrom_diagonals": (),
            "nystrom_valid": (),
            "fallback_to_parent": True,
        },
    }


def _normalize_importance(value: torch.Tensor) -> torch.Tensor | None:
    if not bool(torch.isfinite(value).all().item()):
        return None
    denominator = value.to(torch.float32).mean()
    if not bool(torch.isfinite(denominator).item()) or float(denominator.item()) <= 0:
        return None
    result = (value.to(torch.float32) / denominator.clamp_min(1.0e-12)).clamp(
        1.0 / 16.0, 16.0
    )
    return result if bool(torch.isfinite(result).all().item()) else None


def _normalized_hadamard(order: int) -> torch.Tensor | None:
    """Return a deterministic block-Sylvester orthogonal basis in FP64."""
    if order <= 0 or order % _BLOCK:
        return None
    blocks = []
    remaining = order
    block_order = 1 << (remaining.bit_length() - 1)
    while remaining:
        if remaining >= block_order:
            result = torch.ones(1, 1, dtype=torch.float64)
            while result.shape[0] < block_order:
                result = torch.cat(
                    (
                        torch.cat((result, result), dim=1),
                        torch.cat((result, -result), dim=1),
                    ),
                    dim=0,
                )
            blocks.append(result / float(block_order) ** 0.5)
            remaining -= block_order
        block_order //= 2
        if block_order < _BLOCK and remaining:
            return None
    return torch.block_diag(*blocks)


def _symmetric_matrix_power(
    matrix: torch.Tensor, exponent: float, enforce_condition: bool = True
) -> tuple[torch.Tensor, float] | None:
    """Apply a fixed spectral power to a finite FP64 SPD matrix."""
    value = 0.5 * (matrix.to(torch.float64) + matrix.to(torch.float64).T)
    if not bool(torch.isfinite(value).all().item()):
        return None
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(value)
    except RuntimeError:
        return None
    if not bool(torch.isfinite(eigenvalues).all().item()):
        return None
    smallest = float(eigenvalues[0].item())
    largest = float(eigenvalues[-1].item())
    if smallest <= 0.0 or largest <= 0.0:
        return None
    condition = largest / smallest
    if not torch.isfinite(torch.tensor(condition)) or (
        enforce_condition and condition > _PAIR_CONDITION_LIMIT
    ):
        return None
    powered = (eigenvectors * eigenvalues.pow(exponent).unsqueeze(0)) @ eigenvectors.T
    powered = 0.5 * (powered + powered.T)
    if not bool(torch.isfinite(powered).all().item()):
        return None
    return powered, condition


def _pair_transform(
    q_covariance: torch.Tensor,
    k_covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """Construct one fixed non-orthogonal reciprocal Q/K transform."""
    head_dim = int(q_covariance.shape[-1])
    if (
        q_covariance.shape != (head_dim, head_dim)
        or k_covariance.shape != (head_dim, head_dim)
        or head_dim % _BLOCK
    ):
        return None
    hadamard = _normalized_hadamard(head_dim)
    if hadamard is None:
        return None
    if _PAIR_ABLATION == "orthogonal-only":
        identity_error = float(
            (hadamard @ hadamard.T - torch.eye(head_dim, dtype=torch.float64))
            .abs()
            .amax()
            .item()
        )
        return hadamard.to(torch.float32), hadamard.to(torch.float32), identity_error
    if _PAIR_ABLATION in ("mechanism-off", "diagonal-only"):
        return None

    identity = torch.eye(head_dim, dtype=torch.float64)
    q_value = 0.5 * (
        q_covariance.to(torch.float64) + q_covariance.to(torch.float64).T
    )
    k_value = 0.5 * (
        k_covariance.to(torch.float64) + k_covariance.to(torch.float64).T
    )
    q_epsilon = (
        _PAIR_REGULARIZATION * torch.trace(q_value) / head_dim
        + _PAIR_REGULARIZATION_FLOOR
    )
    k_epsilon = (
        _PAIR_REGULARIZATION * torch.trace(k_value) / head_dim
        + _PAIR_REGULARIZATION_FLOOR
    )
    if (
        not bool(torch.isfinite(q_epsilon).item())
        or not bool(torch.isfinite(k_epsilon).item())
        or float(q_epsilon.item()) <= 0.0
        or float(k_epsilon.item()) <= 0.0
    ):
        return None
    q_regularized = q_value + q_epsilon * identity
    k_regularized = k_value + k_epsilon * identity
    q_sqrt_pair = _symmetric_matrix_power(q_regularized, 0.5, False)
    q_inverse_pair = _symmetric_matrix_power(q_regularized, -0.5, False)
    if q_sqrt_pair is None or q_inverse_pair is None:
        return None
    q_sqrt, _ = q_sqrt_pair
    q_inverse_sqrt, _ = q_inverse_pair
    middle = q_sqrt @ k_regularized @ q_sqrt
    middle_quarter_pair = _symmetric_matrix_power(middle, 0.25, False)
    if middle_quarter_pair is None:
        return None
    middle_quarter, _ = middle_quarter_pair
    balance = q_inverse_sqrt @ middle_quarter
    try:
        singular_values = torch.linalg.svdvals(balance)
    except RuntimeError:
        return None
    balance_condition = float((singular_values[0] / singular_values[-1]).item())
    if (
        not bool(torch.isfinite(singular_values).all().item())
        or float(singular_values[-1].item()) <= 0.0
        or balance_condition > _PAIR_CONDITION_LIMIT
    ):
        return None
    right_basis = identity if _PAIR_ABLATION == "balance-without-H" else hadamard
    q_transform = balance @ right_basis
    try:
        k_transform = torch.linalg.solve(balance.T, right_basis)
    except RuntimeError:
        return None
    q_transform32 = q_transform.to(torch.float32)
    k_transform32 = k_transform.to(torch.float32)
    identity_error = float(
        (
            q_transform32.to(torch.float64)
            @ k_transform32.to(torch.float64).T
            - identity
        )
        .abs()
        .amax()
        .item()
    )
    if (
        not bool(torch.isfinite(q_transform32).all().item())
        or not bool(torch.isfinite(k_transform32).all().item())
        or identity_error > _PAIR_EQUIVALENCE_LIMIT
    ):
        return None
    return (
        q_transform32,
        k_transform32,
        max(balance_condition, identity_error),
    )


def _apply_head_transform(
    value: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    rows = int(value.numel() // (transform.shape[0] * transform.shape[-1]))
    shaped = value.reshape(rows, transform.shape[0], transform.shape[-1])
    return torch.einsum(
        "rhi,hij->rhj", shaped, transform.to(device=value.device, dtype=torch.float32)
    ).reshape_as(value)


def _cross_hessian_inverse(
    hessian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Validate and invert every fixed 64x64 calibration covariance once."""
    if hessian.ndim != 4 or hessian.shape[-2:] != (_BLOCK, _BLOCK):
        return None
    diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)
    if (
        not bool(torch.isfinite(hessian).all().item())
        or bool((diagonal < 0.0).any().item())
    ):
        return None
    damping = diagonal.mean(dim=-1) * _HESSIAN_DAMPING + 1.0e-8
    if not bool(torch.isfinite(damping).all().item()):
        return None
    identity = torch.eye(_BLOCK, dtype=torch.float32, device=hessian.device)
    try:
        factor = torch.linalg.cholesky(
            hessian + damping.unsqueeze(-1).unsqueeze(-1) * identity
        )
        inverse = torch.cholesky_inverse(factor)
    except RuntimeError:
        return None
    if not bool(torch.isfinite(inverse).all().item()):
        return None
    return hessian.to(torch.float32), inverse.to(torch.float32)


def _attention_cross_hessians(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_smooth: torch.Tensor,
    kv_smooth: torch.Tensor,
    q_transform: torch.Tensor | None = None,
    k_transform: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
] | None:
    """Build Q/K repair covariances only from the opposite calibration role."""
    if head_dim % _BLOCK:
        return None
    heads_per_kv = q_num_heads // kv_num_heads
    block_count = head_dim // _BLOCK
    k_hessian = torch.zeros(
        kv_num_heads, block_count, _BLOCK, _BLOCK, dtype=torch.float32
    )
    q_hessian = torch.zeros_like(k_hessian)
    q_rows = 0
    k_rows = 0
    try:
        for sample in calib_qkv_list:
            q = dequantize_nvfp4(*sample["q"]).to(torch.float32).reshape(
                -1, q_num_heads, head_dim
            )
            k = dequantize_nvfp4(*sample["k"]).to(torch.float32).reshape(
                -1, kv_num_heads, head_dim
            )
            if not (
                bool(torch.isfinite(q).all().item())
                and bool(torch.isfinite(k).all().item())
            ):
                return None
            q_blocks = (q / q_smooth).reshape(
                -1, q_num_heads, head_dim
            )
            k_blocks = (k * kv_smooth).reshape(-1, kv_num_heads, head_dim)
            if q_transform is not None and k_transform is not None:
                q_blocks = _apply_head_transform(q_blocks, q_transform)
                k_blocks = _apply_head_transform(k_blocks, k_transform)
            q_blocks = q_blocks.reshape(
                -1, kv_num_heads, heads_per_kv, block_count, _BLOCK
            )
            if (
                _Q_HESSIAN_ABLATION == "softmax-nullspace-deflated"
                and int(k_blocks.shape[0]) > 1
            ):
                centered_k = k_blocks - k_blocks.mean(dim=0, keepdim=True)
                centered_energy = centered_k.square().sum()
                if (
                    bool(torch.isfinite(centered_k).all().item())
                    and bool(torch.isfinite(centered_energy).item())
                    and float(centered_energy.item()) > 0.0
                ):
                    q_consumer_k = centered_k
                else:
                    q_consumer_k = k_blocks
            else:
                q_consumer_k = k_blocks
            k_blocks = q_consumer_k.reshape(
                -1, kv_num_heads, block_count, _BLOCK
            )
            q_hessian += torch.einsum(
                "rghbi,rghbj->gbij", q_blocks, q_blocks
            ).cpu()
            k_hessian += torch.einsum(
                "rgbi,rgbj->gbij", k_blocks, k_blocks
            ).cpu()
            q_rows += int(q.shape[0])
            k_rows += int(k.shape[0])
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None
    if q_rows == 0 or k_rows == 0:
        return None
    q_hessian /= q_rows * heads_per_kv
    k_hessian /= k_rows
    q_pair = _cross_hessian_inverse(q_hessian)
    k_pair = _cross_hessian_inverse(k_hessian)
    if q_pair is None or k_pair is None:
        return None
    grouped_q_hessian, grouped_q_inverse = q_pair
    grouped_k_hessian, grouped_k_inverse = k_pair
    return (
        grouped_k_hessian.repeat_interleave(heads_per_kv, dim=0),
        grouped_k_inverse.repeat_interleave(heads_per_kv, dim=0),
        grouped_q_hessian,
        grouped_q_inverse,
    )


def _attention_attach_wide_cross_hessians(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_smooth: torch.Tensor,
    kv_smooth: torch.Tensor,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Attach shape-adaptive full-window covariances for final Q/K descent."""
    need_q = _qk_wide_q_steps(head_dim) > 0
    need_k = _QK_WIDE_K_STEPS > 0
    wide_window = _qk_wide_window(head_dim)
    if (
        wide_window == 0
        or not isinstance(state, dict)
        or not (need_q or need_k)
    ):
        return state
    try:
        q_state = state["q_state"]
        k_state = state["k_state"]
        q_transform = q_state["pair_transform"].to(torch.float32)
        k_transform = k_state["pair_transform"].to(torch.float32)
    except (KeyError, AttributeError, RuntimeError, TypeError):
        return state
    if (
        tuple(q_transform.shape) != (q_num_heads, head_dim, head_dim)
        or tuple(k_transform.shape) != (kv_num_heads, head_dim, head_dim)
        or not bool(torch.isfinite(q_transform).all().item())
        or not bool(torch.isfinite(k_transform).all().item())
    ):
        return state

    cached_q_values = state.get("_hierarchy_q_values")
    cached_k_values = state.get("_hierarchy_k_values")
    cache_valid = (
        isinstance(cached_q_values, list)
        and isinstance(cached_k_values, list)
        and len(cached_q_values) == len(calib_qkv_list)
        and len(cached_k_values) == len(calib_qkv_list)
    )

    heads_per_kv = q_num_heads // kv_num_heads
    window_count = head_dim // wide_window
    q_consumer = torch.zeros(
        kv_num_heads,
        window_count,
        wide_window,
        wide_window,
        dtype=torch.float32,
    )
    k_consumer = torch.zeros_like(q_consumer)
    q_rows = 0
    k_rows = 0
    try:
        for sample_index, sample in enumerate(calib_qkv_list):
            if sample_index >= _QK_WIDE_CALIBRATION_SHARDS:
                break
            if cache_valid:
                q_basis = cached_q_values[sample_index]
                k_basis = cached_k_values[sample_index]
                if type(q_basis) is not torch.Tensor or type(k_basis) is not torch.Tensor:
                    return state
            else:
                q = dequantize_nvfp4(*sample["q"]).to(torch.float32).reshape(
                    -1, q_num_heads, head_dim
                )
                k = dequantize_nvfp4(*sample["k"]).to(torch.float32).reshape(
                    -1, kv_num_heads, head_dim
                )
                q_basis = _apply_head_transform(q / q_smooth, q_transform)
                k_basis = _apply_head_transform(k * kv_smooth, k_transform)
            if need_q and int(k_basis.shape[0]) > 1:
                centered_k = k_basis - k_basis.mean(dim=0, keepdim=True)
                if bool(torch.isfinite(centered_k).all().item()):
                    k_basis = centered_k
            if need_k:
                q_windows = q_basis.reshape(
                    -1,
                    kv_num_heads,
                    heads_per_kv,
                    window_count,
                    wide_window,
                )
                k_consumer += torch.einsum(
                    "rghwi,rghwj->gwij", q_windows, q_windows
                ).cpu()
                q_rows += int(q_windows.shape[0]) * heads_per_kv
            if need_q:
                k_windows = k_basis.reshape(
                    -1, kv_num_heads, window_count, wide_window
                )
                q_consumer += torch.einsum(
                    "rgwi,rgwj->gwij", k_windows, k_windows
                ).cpu()
                k_rows += int(k_windows.shape[0])
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        return state
    if (need_k and q_rows <= 0) or (need_q and k_rows <= 0):
        return state
    q_wide = None
    if need_q:
        q_consumer /= k_rows
        q_wide = q_consumer.repeat_interleave(heads_per_kv, dim=0)
        q_diagonal = torch.diagonal(q_wide, dim1=-2, dim2=-1)
        if (
            not bool(torch.isfinite(q_wide).all().item())
            or bool((q_diagonal < 0.0).any().item())
        ):
            return state
    if need_k:
        k_consumer /= q_rows
        k_diagonal = torch.diagonal(k_consumer, dim1=-2, dim2=-1)
        if (
            not bool(torch.isfinite(k_consumer).all().item())
            or bool((k_diagonal < 0.0).any().item())
        ):
            return state
    if not need_q and not need_k:
        return state
    selected = dict(state)
    selected_q = dict(q_state)
    selected_k = dict(k_state)
    if need_q and q_wide is not None:
        selected_q.update(
            {
                "wide_cross_schema": _QK_WIDE_SCHEMA,
                "wide_cross_window": wide_window,
                "wide_cross_hessian": q_wide.cpu(),
            }
        )
    if need_k:
        selected_k.update(
            {
                "wide_cross_schema": _QK_WIDE_SCHEMA,
                "wide_cross_window": wide_window,
                "wide_cross_hessian": k_consumer.cpu(),
            }
        )
    selected["q_state"] = selected_q
    selected["k_state"] = selected_k
    return selected


def _attention_proxy_reconstruction(
    value: torch.Tensor, importance: torch.Tensor
) -> torch.Tensor:
    # The gate needs a fixed legal reconstruction, not Dynamic candidate search.
    # Keeping it at the V31 base encoder makes calibration bounded and deterministic.
    del importance
    encoded = _quantize_hif4(value)
    return _dequantize_hif4(encoded).reshape_as(value)


def _pair_proxy_components(
    q_basis: torch.Tensor,
    k_basis: torch.Tensor,
    q_reconstruction: torch.Tensor,
    k_reconstruction: torch.Tensor,
    kv_num_heads: int,
    heads_per_kv: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    q_grouped = q_basis.reshape(
        -1, kv_num_heads, heads_per_kv, head_dim
    ).to(torch.float64)
    q_error = (q_basis - q_reconstruction).reshape_as(q_grouped).to(torch.float64)
    k_grouped = k_basis.reshape(-1, kv_num_heads, head_dim).to(torch.float64)
    k_error = (k_basis - k_reconstruction).reshape_as(k_grouped).to(torch.float64)
    q_count = int(q_grouped.shape[0]) * heads_per_kv
    k_count = int(k_grouped.shape[0])
    return (
        torch.einsum("rghi,rghj->gij", q_error, q_error),
        torch.einsum("rgi,rgj->gij", k_error, k_error),
        torch.einsum("rghi,rghj->gij", q_grouped, q_grouped),
        torch.einsum("rgi,rgj->gij", k_grouped, k_grouped),
        q_count,
        k_count,
    )


def _pair_proxy_score(
    components: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int
    ]
) -> torch.Tensor:
    q_error, k_error, q_covariance, k_covariance, q_count, k_count = components
    denominator = float(q_count * k_count)
    return (
        torch.einsum("gij,gji->g", q_error, k_covariance)
        + torch.einsum("gij,gji->g", k_error, q_covariance)
    ) / denominator


def _sum_pair_components(
    values: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    first = values[0]
    return (
        sum((value[0] for value in values[1:]), first[0].clone()),
        sum((value[1] for value in values[1:]), first[1].clone()),
        sum((value[2] for value in values[1:]), first[2].clone()),
        sum((value[3] for value in values[1:]), first[3].clone()),
        sum(value[4] for value in values),
        sum(value[5] for value in values),
    )


def _attention_pair_state(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_smooth: torch.Tensor,
    kv_smooth: torch.Tensor,
    parent_state: dict[str, Any],
) -> dict[str, Any]:
    """Build and shard-gate an atomic reciprocal transform for each KV group."""
    if head_dim % _BLOCK:
        return parent_state
    heads_per_kv = q_num_heads // kv_num_heads
    q_covariance = torch.zeros(
        kv_num_heads, head_dim, head_dim, dtype=torch.float64
    )
    k_covariance = torch.zeros_like(q_covariance)
    q_count = 0
    k_count = 0
    q_bases: list[torch.Tensor] = []
    k_bases: list[torch.Tensor] = []
    try:
        for sample in calib_qkv_list:
            q = dequantize_nvfp4(*sample["q"]).to(torch.float32).reshape(
                -1, q_num_heads, head_dim
            )
            k = dequantize_nvfp4(*sample["k"]).to(torch.float32).reshape(
                -1, kv_num_heads, head_dim
            )
            q_basis = (q / q_smooth).reshape(-1, q_num_heads, head_dim)
            k_basis = (k * kv_smooth).reshape(-1, kv_num_heads, head_dim)
            if not (
                bool(torch.isfinite(q_basis).all().item())
                and bool(torch.isfinite(k_basis).all().item())
            ):
                return parent_state
            q_grouped = q_basis.reshape(
                -1, kv_num_heads, heads_per_kv, head_dim
            ).to(torch.float64)
            k_grouped = k_basis.to(torch.float64)
            q_covariance += torch.einsum("rghi,rghj->gij", q_grouped, q_grouped)
            k_covariance += torch.einsum("rgi,rgj->gij", k_grouped, k_grouped)
            q_count += int(q_grouped.shape[0]) * heads_per_kv
            k_count += int(k_grouped.shape[0])
            q_bases.append(q_basis.cpu())
            k_bases.append(k_basis.cpu())
    except (KeyError, TypeError, ValueError, RuntimeError):
        return parent_state
    if q_count == 0 or k_count == 0 or len(q_bases) != _CALIBRATION_SHARDS:
        return parent_state
    q_covariance /= q_count
    k_covariance /= k_count

    identity = torch.eye(head_dim, dtype=torch.float32)
    q_group_transform = identity.repeat(kv_num_heads, 1, 1)
    k_group_transform = identity.repeat(kv_num_heads, 1, 1)
    transform_valid = torch.zeros(kv_num_heads, dtype=torch.bool)
    transform_condition = torch.full(
        (kv_num_heads,), float("inf"), dtype=torch.float64
    )
    identity_error = torch.full_like(transform_condition, float("inf"))
    for group in range(kv_num_heads):
        pair = _pair_transform(q_covariance[group], k_covariance[group])
        if pair is None:
            continue
        q_transform, k_transform, condition = pair
        error = float(
            (
                q_transform.to(torch.float64)
                @ k_transform.to(torch.float64).T
                - torch.eye(head_dim, dtype=torch.float64)
            )
            .abs()
            .amax()
            .item()
        )
        if error > _PAIR_EQUIVALENCE_LIMIT:
            continue
        q_group_transform[group] = q_transform
        k_group_transform[group] = k_transform
        transform_valid[group] = True
        transform_condition[group] = condition
        identity_error[group] = error
    if not bool(transform_valid.any().item()):
        return parent_state

    q_transform = q_group_transform.repeat_interleave(heads_per_kv, dim=0)
    k_transform = k_group_transform
    q_square_sum = torch.zeros(q_num_heads, head_dim, dtype=torch.float64)
    k_square_sum = torch.zeros(kv_num_heads, head_dim, dtype=torch.float64)
    transformed_q: list[torch.Tensor] = []
    transformed_k: list[torch.Tensor] = []
    for q_basis, k_basis in zip(q_bases, k_bases):
        q_value = _apply_head_transform(q_basis, q_transform).cpu()
        k_value = _apply_head_transform(k_basis, k_transform).cpu()
        if not (
            bool(torch.isfinite(q_value).all().item())
            and bool(torch.isfinite(k_value).all().item())
        ):
            return parent_state
        q_square_sum += q_value.to(torch.float64).square().sum(dim=0)
        k_square_sum += k_value.to(torch.float64).square().sum(dim=0)
        transformed_q.append(q_value)
        transformed_k.append(k_value)
    q_rows = sum(x.shape[0] for x in transformed_q)
    k_rows = sum(x.shape[0] for x in transformed_k)
    q_second = q_square_sum / q_rows
    k_second = k_square_sum / k_rows
    q_importance = _normalize_importance(
        k_second.to(torch.float32).repeat_interleave(heads_per_kv, dim=0)
    )
    k_importance = _normalize_importance(
        q_second.reshape(
            kv_num_heads, heads_per_kv, head_dim
        ).mean(dim=1).to(torch.float32)
    )
    if q_importance is None or k_importance is None:
        return parent_state

    parent_q_importance = parent_state["q_state"]["importance"].reshape(
        q_num_heads, head_dim
    )
    parent_k_importance = parent_state["k_state"]["importance"].reshape(
        kv_num_heads, head_dim
    )
    parent_components = []
    candidate_components = []
    for q_basis, k_basis, q_value, k_value in zip(
        q_bases, k_bases, transformed_q, transformed_k
    ):
        parent_q_reconstruction = _attention_proxy_reconstruction(
            q_basis, parent_q_importance
        )
        parent_k_reconstruction = _attention_proxy_reconstruction(
            k_basis, parent_k_importance
        )
        candidate_q_reconstruction = _attention_proxy_reconstruction(
            q_value, q_importance
        )
        candidate_k_reconstruction = _attention_proxy_reconstruction(
            k_value, k_importance
        )
        parent_components.append(
            _pair_proxy_components(
                q_basis,
                k_basis,
                parent_q_reconstruction,
                parent_k_reconstruction,
                kv_num_heads,
                heads_per_kv,
                head_dim,
            )
        )
        candidate_components.append(
            _pair_proxy_components(
                q_value,
                k_value,
                candidate_q_reconstruction,
                candidate_k_reconstruction,
                kv_num_heads,
                heads_per_kv,
                head_dim,
            )
        )
    parent_shard_scores = torch.stack(
        [_pair_proxy_score(value) for value in parent_components]
    )
    candidate_shard_scores = torch.stack(
        [_pair_proxy_score(value) for value in candidate_components]
    )
    parent_aggregate = _pair_proxy_score(_sum_pair_components(parent_components))
    candidate_aggregate = _pair_proxy_score(_sum_pair_components(candidate_components))
    shard_stable = (candidate_shard_scores <= parent_shard_scores).all(dim=0)
    if _PAIR_ABLATION == "no-shard-gate":
        shard_stable = torch.ones_like(shard_stable)
    accepted = (
        transform_valid
        & torch.isfinite(parent_aggregate)
        & torch.isfinite(candidate_aggregate)
        & (candidate_aggregate < parent_aggregate)
        & shard_stable
    )
    if not bool(accepted.any().item()):
        return parent_state

    final_q_group_transform = torch.where(
        accepted[:, None, None], q_group_transform, identity[None, :, :]
    )
    final_k_transform = torch.where(
        accepted[:, None, None], k_group_transform, identity[None, :, :]
    )
    final_q_transform = final_q_group_transform.repeat_interleave(
        heads_per_kv, dim=0
    )
    accepted_q = accepted.repeat_interleave(heads_per_kv)
    final_q_importance = torch.where(
        accepted_q[:, None], q_importance, parent_q_importance
    )
    final_k_importance = torch.where(
        accepted[:, None], k_importance, parent_k_importance
    )
    cross = _attention_cross_hessians(
        calib_qkv_list,
        q_num_heads,
        kv_num_heads,
        head_dim,
        q_smooth,
        kv_smooth,
        final_q_transform,
        final_k_transform,
    )
    if cross is None:
        return parent_state
    q_hessian, q_inverse, k_hessian, k_inverse = cross
    parent_q_hessian = parent_state["q_state"]["cross_hessian"]
    parent_q_inverse = parent_state["q_state"]["cross_inverse"]
    parent_k_hessian = parent_state["k_state"]["cross_hessian"]
    parent_k_inverse = parent_state["k_state"]["cross_inverse"]
    q_hessian = torch.where(
        accepted_q[:, None, None, None], q_hessian, parent_q_hessian
    )
    q_inverse = torch.where(
        accepted_q[:, None, None, None], q_inverse, parent_q_inverse
    )
    k_hessian = torch.where(
        accepted[:, None, None, None], k_hessian, parent_k_hessian
    )
    k_inverse = torch.where(
        accepted[:, None, None, None], k_inverse, parent_k_inverse
    )

    q_state = dict(parent_state["q_state"])
    k_state = dict(parent_state["k_state"])
    q_state.update(
        {
            "importance": final_q_importance.flatten().cpu(),
            "cross_hessian": q_hessian.cpu(),
            "cross_inverse": q_inverse.cpu(),
            "pair_transform": final_q_transform.cpu(),
            "pair_group_accepted": accepted.cpu(),
            "pair_identity_error": identity_error.to(torch.float32).cpu(),
            "pair_condition": transform_condition.to(torch.float32).cpu(),
            "pair_ablation": _PAIR_ABLATION,
            "rotation_group_size": head_dim,
            "rotation_reason": "full-covariance-geometric-reciprocal-pair",
        }
    )
    k_state.update(
        {
            "importance": final_k_importance.flatten().cpu(),
            "cross_hessian": k_hessian.cpu(),
            "cross_inverse": k_inverse.cpu(),
            "pair_transform": final_k_transform.cpu(),
            "pair_group_accepted": accepted.cpu(),
            "pair_identity_error": identity_error.to(torch.float32).cpu(),
            "pair_condition": transform_condition.to(torch.float32).cpu(),
            "pair_ablation": _PAIR_ABLATION,
            "rotation_group_size": head_dim,
            "rotation_reason": "full-covariance-geometric-reciprocal-pair",
        }
    )
    if bool(accepted_q.all().item()):
        hierarchy_q_values = transformed_q
    else:
        hierarchy_q_values = [
            torch.where(accepted_q.reshape(1, -1, 1), candidate, basis)
            for basis, candidate in zip(q_bases, transformed_q)
        ]
    if bool(accepted.all().item()):
        hierarchy_k_values = transformed_k
    else:
        hierarchy_k_values = [
            torch.where(accepted.reshape(1, -1, 1), candidate, basis)
            for basis, candidate in zip(k_bases, transformed_k)
        ]
    return {
        "q_state": q_state,
        "k_state": k_state,
        "v_state": parent_state["v_state"],
        "_hierarchy_q_values": hierarchy_q_values,
        "_hierarchy_k_values": hierarchy_k_values,
    }


def _permutation_matrix(permutation: torch.Tensor) -> torch.Tensor | None:
    if permutation.ndim != 2:
        return None
    groups, head_dim = permutation.shape
    natural = torch.arange(head_dim, dtype=torch.long)
    if not all(torch.equal(torch.sort(row).values, natural) for row in permutation):
        return None
    matrix = torch.zeros(groups, head_dim, head_dim, dtype=torch.float32)
    destination = natural.unsqueeze(0).expand(groups, head_dim)
    group_index = torch.arange(groups).unsqueeze(1).expand(groups, head_dim)
    matrix[group_index, permutation, destination] = 1.0
    return matrix


def _normalize_h64_maxima(
    maximum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Normalize each H64 by its positive median without crossing a block."""
    if (
        maximum.ndim != 3
        or maximum.shape[-1] != _BLOCK
        or not bool(torch.isfinite(maximum).all().item())
        or bool((maximum < 0.0).any().item())
    ):
        return None
    normalized = torch.zeros_like(maximum, dtype=torch.float32)
    valid = torch.zeros(maximum.shape[:2], dtype=torch.bool)
    for group in range(int(maximum.shape[0])):
        for block in range(int(maximum.shape[1])):
            values = maximum[group, block].to(torch.float32)
            positive = values[values > 0.0]
            if positive.numel() == 0:
                continue
            middle = positive.median().clamp_min(_MIN_SCALE)
            candidate = values / middle
            if not bool(torch.isfinite(candidate).all().item()):
                continue
            normalized[group, block] = candidate
            valid[group, block] = True
    return normalized, valid


def _nested_hierarchy_envelope(
    q_pressure: torch.Tensor,
    k_pressure: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor | None:
    """Evaluate the fixed H4 plus half-weight H8 Q/K range envelope."""
    if (
        q_pressure.ndim != 3
        or tuple(k_pressure.shape) != tuple(q_pressure.shape)
        or tuple(order.shape) != tuple(q_pressure.shape)
        or q_pressure.shape[-1] != _BLOCK
    ):
        return None
    try:
        q_ordered = torch.gather(q_pressure, 2, order)
        k_ordered = torch.gather(k_pressure, 2, order)
        q_h4 = q_ordered.reshape(*q_ordered.shape[:2], _BLOCK // 4, 4)
        k_h4 = k_ordered.reshape(*k_ordered.shape[:2], _BLOCK // 4, 4)
        q_h8 = q_ordered.reshape(*q_ordered.shape[:2], _BLOCK // 8, 8)
        k_h8 = k_ordered.reshape(*k_ordered.shape[:2], _BLOCK // 8, 8)
        score = (
            q_h4.amax(dim=-1).square().sum(dim=(-1, -2))
            + k_h4.amax(dim=-1).square().sum(dim=(-1, -2))
            + 0.5 * q_h8.amax(dim=-1).square().sum(dim=(-1, -2))
            + 0.5 * k_h8.amax(dim=-1).square().sum(dim=(-1, -2))
        ).to(torch.float64)
    except (RuntimeError, ValueError, OverflowError):
        return None
    if not bool(torch.isfinite(score).all().item()):
        return None
    return score


def _gather_h64_congruence(
    matrix: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor | None:
    """Apply P^T H P to batched H64 matrices using the permutation gather."""
    if (
        matrix.ndim != 4
        or tuple(matrix.shape[:2]) != tuple(order.shape[:2])
        or tuple(matrix.shape[-2:]) != (_BLOCK, _BLOCK)
        or order.shape[-1] != _BLOCK
    ):
        return None
    try:
        rows = order.unsqueeze(-1).expand(-1, -1, -1, _BLOCK)
        columns = order.unsqueeze(-2).expand(-1, -1, _BLOCK, -1)
        gathered = torch.gather(torch.gather(matrix, 2, rows), 3, columns)
    except (RuntimeError, ValueError):
        return None
    if not bool(torch.isfinite(gathered).all().item()):
        return None
    return gathered


def _gather_window_congruence(
    matrix: torch.Tensor,
    order: torch.Tensor,
    window: int,
) -> torch.Tensor | None:
    """Apply P^T H P to a batched square window covariance."""
    if (
        matrix.ndim != 4
        or tuple(matrix.shape[:2]) != tuple(order.shape[:2])
        or tuple(matrix.shape[-2:]) != (window, window)
        or order.shape[-1] != window
    ):
        return None
    try:
        rows = order.unsqueeze(-1).expand(-1, -1, -1, window)
        columns = order.unsqueeze(-2).expand(-1, -1, window, -1)
        gathered = torch.gather(torch.gather(matrix, 2, rows), 3, columns)
    except (RuntimeError, ValueError):
        return None
    if not bool(torch.isfinite(gathered).all().item()):
        return None
    return gathered


def _attention_hierarchy_envelope_state(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_smooth: torch.Tensor,
    kv_smooth: torch.Tensor,
    parent_state: dict[str, Any],
) -> dict[str, Any]:
    """Publish V278's common Q/K hierarchy permutation over the V281 state."""
    cached_q_values = parent_state.get("_hierarchy_q_values")
    cached_k_values = parent_state.get("_hierarchy_k_values")
    cache_valid = (
        isinstance(cached_q_values, list)
        and isinstance(cached_k_values, list)
        and len(cached_q_values) == len(calib_qkv_list)
        and len(cached_k_values) == len(calib_qkv_list)
    )
    if cached_q_values is not None or cached_k_values is not None:
        parent_state = dict(parent_state)
        parent_state.pop("_hierarchy_q_values", None)
        parent_state.pop("_hierarchy_k_values", None)
    if (
        _QK_PERMUTATION_ABLATION in ("off", "mechanism-off")
        or head_dim % _BLOCK
        or q_num_heads % kv_num_heads
        or not calib_qkv_list
    ):
        return parent_state
    heads_per_kv = q_num_heads // kv_num_heads
    block_count = head_dim // _BLOCK
    q_state = parent_state.get("q_state")
    k_state = parent_state.get("k_state")
    if (
        not isinstance(q_state, dict)
        or not isinstance(k_state, dict)
        or q_state.get("fallback_to_parent", False)
        or k_state.get("fallback_to_parent", False)
    ):
        return parent_state

    identity = torch.eye(head_dim, dtype=torch.float32)
    parent_q_transform = q_state.get("pair_transform")
    parent_k_transform = k_state.get("pair_transform")
    if parent_q_transform is None and parent_k_transform is None:
        parent_q_transform = identity.repeat(q_num_heads, 1, 1)
        parent_k_transform = identity.repeat(kv_num_heads, 1, 1)
    expected_q_transform = (q_num_heads, head_dim, head_dim)
    expected_k_transform = (kv_num_heads, head_dim, head_dim)
    expected_q_matrix = (q_num_heads, block_count, _BLOCK, _BLOCK)
    expected_k_matrix = (kv_num_heads, block_count, _BLOCK, _BLOCK)
    q_importance = q_state.get("importance")
    k_importance = k_state.get("importance")
    q_hessian = q_state.get("cross_hessian")
    q_inverse = q_state.get("cross_inverse")
    k_hessian = k_state.get("cross_hessian")
    k_inverse = k_state.get("cross_inverse")
    if (
        type(parent_q_transform) is not torch.Tensor
        or type(parent_k_transform) is not torch.Tensor
        or tuple(parent_q_transform.shape) != expected_q_transform
        or tuple(parent_k_transform.shape) != expected_k_transform
        or type(q_importance) is not torch.Tensor
        or type(k_importance) is not torch.Tensor
        or q_importance.numel() != q_num_heads * head_dim
        or k_importance.numel() != kv_num_heads * head_dim
        or type(q_hessian) is not torch.Tensor
        or type(q_inverse) is not torch.Tensor
        or type(k_hessian) is not torch.Tensor
        or type(k_inverse) is not torch.Tensor
        or tuple(q_hessian.shape) != expected_q_matrix
        or tuple(q_inverse.shape) != expected_q_matrix
        or tuple(k_hessian.shape) != expected_k_matrix
        or tuple(k_inverse.shape) != expected_k_matrix
    ):
        return parent_state
    state_tensors = (
        parent_q_transform,
        parent_k_transform,
        q_importance,
        k_importance,
        q_hessian,
        q_inverse,
        k_hessian,
        k_inverse,
    )
    if not all(bool(torch.isfinite(item).all().item()) for item in state_tensors):
        return parent_state

    joint_pressure = torch.zeros(
        kv_num_heads, block_count, _BLOCK, dtype=torch.float32
    )
    all_blocks_valid = torch.ones(
        kv_num_heads, block_count, dtype=torch.bool
    )
    normalized_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for sample_index, sample in enumerate(calib_qkv_list):
            if cache_valid:
                q_basis = cached_q_values[sample_index]
                k_basis = cached_k_values[sample_index]
                if type(q_basis) is not torch.Tensor or type(k_basis) is not torch.Tensor:
                    return parent_state
            else:
                q = dequantize_nvfp4(*sample["q"]).to(torch.float32).reshape(
                    -1, q_num_heads, head_dim
                )
                k = dequantize_nvfp4(*sample["k"]).to(torch.float32).reshape(
                    -1, kv_num_heads, head_dim
                )
                q_basis = _apply_head_transform(q / q_smooth, parent_q_transform)
                k_basis = _apply_head_transform(k * kv_smooth, parent_k_transform)
            if not (
                bool(torch.isfinite(q_basis).all().item())
                and bool(torch.isfinite(k_basis).all().item())
            ):
                return parent_state
            q_maximum = q_basis.reshape(
                -1, kv_num_heads, heads_per_kv, block_count, _BLOCK
            ).abs().amax(dim=(0, 2))
            k_maximum = k_basis.reshape(
                -1, kv_num_heads, block_count, _BLOCK
            ).abs().amax(dim=0)
            q_normalized = _normalize_h64_maxima(q_maximum.cpu())
            k_normalized = _normalize_h64_maxima(k_maximum.cpu())
            if q_normalized is None or k_normalized is None:
                return parent_state
            q_pressure, q_valid = q_normalized
            k_pressure, k_valid = k_normalized
            valid = q_valid & k_valid
            all_blocks_valid &= valid
            joint_pressure = torch.maximum(
                joint_pressure, torch.maximum(q_pressure, k_pressure)
            )
            normalized_samples.append((q_pressure, k_pressure))
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        return parent_state
    if not normalized_samples:
        return parent_state

    natural_order = torch.arange(_BLOCK, dtype=torch.long).reshape(1, 1, _BLOCK)
    natural_order = natural_order.expand(kv_num_heads, block_count, _BLOCK)
    try:
        packed_order = torch.argsort(
            joint_pressure, dim=-1, descending=True, stable=True
        )
    except (RuntimeError, TypeError):
        return parent_state
    packed_order = torch.where(
        all_blocks_valid.unsqueeze(-1), packed_order, natural_order
    )
    changed_groups = (packed_order != natural_order).any(dim=-1).any(dim=-1)
    if not bool(changed_groups.any().item()):
        return parent_state

    parent_shards = torch.zeros(
        _QK_PERMUTATION_SHARDS, kv_num_heads, dtype=torch.float64
    )
    candidate_shards = torch.zeros_like(parent_shards)
    for sample_index, (q_pressure, k_pressure) in enumerate(normalized_samples):
        parent_score = _nested_hierarchy_envelope(
            q_pressure, k_pressure, natural_order
        )
        candidate_score = _nested_hierarchy_envelope(
            q_pressure, k_pressure, packed_order
        )
        if parent_score is None or candidate_score is None:
            return parent_state
        shard = sample_index % _QK_PERMUTATION_SHARDS
        parent_shards[shard] += parent_score
        candidate_shards[shard] += candidate_score
    pooled_parent = parent_shards.sum(dim=0)
    pooled_candidate = candidate_shards.sum(dim=0)
    accepted = (
        changed_groups
        & torch.isfinite(pooled_parent)
        & torch.isfinite(pooled_candidate)
        & (pooled_candidate < pooled_parent)
        & (candidate_shards.amax(dim=0) <= parent_shards.amax(dim=0))
    )
    if not bool(accepted.any().item()):
        return parent_state

    full_permutation = torch.arange(head_dim, dtype=torch.long).repeat(
        kv_num_heads, 1
    )
    for block in range(block_count):
        start = block * _BLOCK
        full_permutation[:, start : start + _BLOCK] = packed_order[:, block] + start
    permutation_matrix = _permutation_matrix(full_permutation)
    if permutation_matrix is None:
        return parent_state
    q_permutation_matrix = permutation_matrix.repeat_interleave(
        heads_per_kv, dim=0
    )
    candidate_q_transform = torch.bmm(
        parent_q_transform.to(torch.float32), q_permutation_matrix
    )
    candidate_k_transform = torch.bmm(
        parent_k_transform.to(torch.float32), permutation_matrix
    )
    candidate_k_for_q = candidate_k_transform.repeat_interleave(
        heads_per_kv, dim=0
    )
    identity_error_per_head = (
        torch.bmm(candidate_q_transform, candidate_k_for_q.transpose(1, 2))
        - identity.unsqueeze(0)
    ).abs().amax(dim=(1, 2))
    identity_error = identity_error_per_head.reshape(
        kv_num_heads, heads_per_kv
    ).amax(dim=1)
    accepted &= torch.isfinite(identity_error) & (
        identity_error <= _PAIR_EQUIVALENCE_LIMIT
    )
    if not bool(accepted.any().item()):
        return parent_state

    q_full_order = full_permutation.repeat_interleave(heads_per_kv, dim=0)
    candidate_q_importance = torch.gather(
        q_importance.reshape(q_num_heads, head_dim), 1, q_full_order
    )
    candidate_k_importance = torch.gather(
        k_importance.reshape(kv_num_heads, head_dim), 1, full_permutation
    )
    q_block_order = packed_order.repeat_interleave(heads_per_kv, dim=0)
    candidate_q_hessian = _gather_h64_congruence(q_hessian, q_block_order)
    candidate_q_inverse = _gather_h64_congruence(q_inverse, q_block_order)
    candidate_k_hessian = _gather_h64_congruence(k_hessian, packed_order)
    candidate_k_inverse = _gather_h64_congruence(k_inverse, packed_order)
    if any(
        item is None
        for item in (
            candidate_q_hessian,
            candidate_q_inverse,
            candidate_k_hessian,
            candidate_k_inverse,
        )
    ):
        return parent_state

    q_wide_hessian = q_state.get("wide_cross_hessian")
    k_wide_hessian = k_state.get("wide_cross_hessian")
    candidate_q_wide = None
    candidate_k_wide = None
    wide_window = _qk_wide_window(head_dim)
    wide_windows = head_dim // wide_window if wide_window else 0
    if wide_window:
        window_offsets = (
            torch.arange(wide_windows, dtype=torch.long)
            * wide_window
        ).reshape(1, wide_windows, 1)
        if (
            type(q_wide_hessian) is torch.Tensor
            and tuple(q_wide_hessian.shape)
            == (q_num_heads, wide_windows, wide_window, wide_window)
        ):
            q_wide_order = q_full_order.reshape(
                q_num_heads, wide_windows, wide_window
            ) - window_offsets
            candidate_q_wide = _gather_window_congruence(
                q_wide_hessian, q_wide_order, wide_window
            )
        if (
            type(k_wide_hessian) is torch.Tensor
            and tuple(k_wide_hessian.shape)
            == (kv_num_heads, wide_windows, wide_window, wide_window)
        ):
            k_wide_order = full_permutation.reshape(
                kv_num_heads, wide_windows, wide_window
            ) - window_offsets
            candidate_k_wide = _gather_window_congruence(
                k_wide_hessian, k_wide_order, wide_window
            )

    accepted_q = accepted.repeat_interleave(heads_per_kv)
    final_q_state = dict(q_state)
    final_k_state = dict(k_state)
    common_metadata = {
        "hierarchy_permutation_schema": _QK_PERMUTATION_SCHEMA,
        "hierarchy_permutation_accepted": accepted.cpu(),
        "hierarchy_permutation": full_permutation.cpu(),
        "hierarchy_permutation_parent_envelope": pooled_parent.to(torch.float32),
        "hierarchy_permutation_candidate_envelope": pooled_candidate.to(torch.float32),
        "hierarchy_permutation_parent_shards": parent_shards.to(torch.float32),
        "hierarchy_permutation_candidate_shards": candidate_shards.to(torch.float32),
        "hierarchy_permutation_identity_error": identity_error.to(torch.float32),
    }
    final_q_state.update(common_metadata)
    final_k_state.update(common_metadata)
    final_q_state.update(
        {
            "importance": torch.where(
                accepted_q[:, None],
                candidate_q_importance,
                q_importance.reshape(q_num_heads, head_dim),
            ).flatten().cpu(),
            "cross_hessian": torch.where(
                accepted_q[:, None, None, None], candidate_q_hessian, q_hessian
            ).cpu(),
            "cross_inverse": torch.where(
                accepted_q[:, None, None, None], candidate_q_inverse, q_inverse
            ).cpu(),
            "pair_transform": torch.where(
                accepted_q[:, None, None], candidate_q_transform, parent_q_transform
            ).cpu(),
        }
    )
    final_k_state.update(
        {
            "importance": torch.where(
                accepted[:, None],
                candidate_k_importance,
                k_importance.reshape(kv_num_heads, head_dim),
            ).flatten().cpu(),
            "cross_hessian": torch.where(
                accepted[:, None, None, None], candidate_k_hessian, k_hessian
            ).cpu(),
            "cross_inverse": torch.where(
                accepted[:, None, None, None], candidate_k_inverse, k_inverse
            ).cpu(),
            "pair_transform": torch.where(
                accepted[:, None, None], candidate_k_transform, parent_k_transform
            ).cpu(),
        }
    )
    if candidate_q_wide is not None:
        final_q_state["wide_cross_hessian"] = torch.where(
            accepted_q[:, None, None, None],
            candidate_q_wide,
            q_wide_hessian,
        ).cpu()
    if candidate_k_wide is not None:
        final_k_state["wide_cross_hessian"] = torch.where(
            accepted[:, None, None, None],
            candidate_k_wide,
            k_wide_hessian,
        ).cpu()
    return {
        "q_state": final_q_state,
        "k_state": final_k_state,
        "v_state": parent_state["v_state"],
    }


def _attention_output_alpha_select(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_max: torch.Tensor,
    k_max: torch.Tensor,
) -> torch.Tensor:
    """Select a reciprocal SmoothQuant exponent by calibration output MSE.

    Selection is independent per KV group, but uses one fixed candidate grid and
    one shape-only token budget for every call.  The proxy evaluates the actual
    non-causal GQA output after legal HiF4 Q/K/V reconstruction; it never reads
    case identity or evaluator metadata.
    """
    if _ATTENTION_OUTPUT_ALPHA_ABLATION != "full":
        return torch.full(
            (kv_num_heads,), _ATTENTION_ALPHA, dtype=torch.float32
        )
    heads_per_kv = q_num_heads // kv_num_heads
    candidates = torch.tensor(
        _ATTENTION_OUTPUT_ALPHA_CANDIDATES, dtype=torch.float32
    )
    losses = torch.zeros(
        int(candidates.numel()), kv_num_heads, dtype=torch.float64
    )
    observations = 0
    try:
        for sample in calib_qkv_list:
            q = dequantize_nvfp4(*sample["q"]).to(torch.float32).reshape(
                -1, q_num_heads, head_dim
            )
            k = dequantize_nvfp4(*sample["k"]).to(torch.float32).reshape(
                -1, kv_num_heads, head_dim
            )
            v = dequantize_nvfp4(*sample["v"]).to(torch.float32).reshape(
                -1, kv_num_heads, head_dim
            )
            sequence = int(q.shape[0])
            if sequence > _ATTENTION_OUTPUT_ALPHA_MAX_TOKENS:
                indices = torch.linspace(
                    0,
                    sequence - 1,
                    _ATTENTION_OUTPUT_ALPHA_MAX_TOKENS,
                    dtype=torch.float32,
                    device=q.device,
                ).round().to(torch.long)
                q = q.index_select(0, indices)
                k = k.index_select(0, indices)
                v = v.index_select(0, indices)
                sequence = int(q.shape[0])

            q_grouped = q.reshape(
                sequence, kv_num_heads, heads_per_kv, head_dim
            ).permute(1, 2, 0, 3)
            k_grouped = k.permute(1, 0, 2)
            v_grouped = v.permute(1, 0, 2)
            reference_probability = torch.softmax(
                torch.matmul(q_grouped, k_grouped.transpose(-1, -2))
                * (float(head_dim) ** -0.5),
                dim=-1,
            )
            reference = torch.matmul(
                reference_probability, v_grouped.unsqueeze(1)
            )
            quantized_v = _dequantize_hif4(
                _quantize_hif4(v.reshape(sequence, -1))
            ).reshape_as(v)
            quantized_v_grouped = quantized_v.permute(1, 0, 2)

            for candidate_index, alpha in enumerate(candidates.tolist()):
                kv_smooth = (
                    q_max.pow(alpha) / k_max.pow(1.0 - alpha)
                ).clamp(_SMOOTH_MIN, _SMOOTH_MAX)
                q_smooth = kv_smooth.repeat_interleave(
                    heads_per_kv, dim=0
                )
                quantized_q = _dequantize_hif4(
                    _quantize_hif4(
                        (q / q_smooth).reshape(sequence, -1)
                    )
                ).reshape_as(q)
                quantized_k = _dequantize_hif4(
                    _quantize_hif4(
                        (k * kv_smooth).reshape(sequence, -1)
                    )
                ).reshape_as(k)
                candidate_q = quantized_q.reshape(
                    sequence, kv_num_heads, heads_per_kv, head_dim
                ).permute(1, 2, 0, 3)
                candidate_k = quantized_k.permute(1, 0, 2)
                candidate_probability = torch.softmax(
                    torch.matmul(
                        candidate_q, candidate_k.transpose(-1, -2)
                    )
                    * (float(head_dim) ** -0.5),
                    dim=-1,
                )
                candidate_output = torch.matmul(
                    candidate_probability,
                    quantized_v_grouped.unsqueeze(1),
                )
                losses[candidate_index] += (
                    candidate_output - reference
                ).to(torch.float64).square().mean(dim=(1, 2, 3)).cpu()
            observations += 1
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        return torch.full(
            (kv_num_heads,), _ATTENTION_ALPHA, dtype=torch.float32
        )
    if observations == 0 or not bool(torch.isfinite(losses).all().item()):
        return torch.full(
            (kv_num_heads,), _ATTENTION_ALPHA, dtype=torch.float32
        )
    return candidates.index_select(0, losses.argmin(dim=0)).to(torch.float32)


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    if (
        q_num_heads <= 0
        or kv_num_heads <= 0
        or head_dim <= 0
        or q_num_heads % kv_num_heads
        or not calib_qkv_list
    ):
        return _attention_fallback_state(q_num_heads, kv_num_heads, head_dim)
    heads_per_kv = q_num_heads // kv_num_heads
    q_max = torch.zeros(kv_num_heads, head_dim, dtype=torch.float32)
    k_max = torch.zeros_like(q_max)
    v_max = torch.zeros_like(q_max)
    q_square_sum = torch.zeros(q_num_heads, head_dim, dtype=torch.float32)
    k_square_sum = torch.zeros_like(q_max)
    v_square_sum = torch.zeros_like(q_max)
    pair_h0 = torch.zeros(
        _V_PAIR_SHARDS, kv_num_heads, dtype=torch.float64
    )
    pair_h1 = torch.zeros_like(pair_h0)
    pair_nonempty = torch.zeros(
        _V_PAIR_SHARDS, kv_num_heads, dtype=torch.bool
    )
    lag2_h0 = torch.zeros_like(pair_h0)
    lag2_h2 = torch.zeros_like(pair_h0)
    lag2_nonempty = torch.zeros_like(pair_nonempty)
    lowrank_gram_sum = torch.zeros(
        _V_LOWRANK_SHARDS,
        kv_num_heads,
        _V_LOWRANK_RANK,
        _V_LOWRANK_RANK,
        dtype=torch.float64,
    )
    lowrank_delta_sum = torch.zeros(
        _V_LOWRANK_SHARDS, kv_num_heads, dtype=torch.float64
    )
    lowrank_count = torch.zeros(
        _V_LOWRANK_SHARDS, kv_num_heads, dtype=torch.float64
    )
    nystrom_stats: dict[int, dict[str, Any]] = {}
    profile_diagonal: dict[int, torch.Tensor] = {}
    profile_edge: dict[int, torch.Tensor] = {}
    profile_count: dict[int, int] = {}
    q_rows = 0
    kv_rows = 0
    try:
        for sample_index, sample in enumerate(calib_qkv_list):
            q = dequantize_nvfp4(*sample["q"]).to(torch.float32).reshape(
                -1, q_num_heads, head_dim
            )
            k = dequantize_nvfp4(*sample["k"]).to(torch.float32).reshape(
                -1, kv_num_heads, head_dim
            )
            v = dequantize_nvfp4(*sample["v"]).to(torch.float32).reshape(
                -1, kv_num_heads, head_dim
            )
            if not (
                bool(torch.isfinite(q).all().item())
                and bool(torch.isfinite(k).all().item())
                and bool(torch.isfinite(v).all().item())
            ):
                raise ValueError("non-finite Attention calibration")
            q_square_sum += q.square().sum(dim=0)
            k_square_sum += k.square().sum(dim=0)
            v_square_sum += v.square().sum(dim=0)
            v_max = torch.maximum(v_max, v.abs().amax(dim=0))
            q_rows += int(q.shape[0])
            kv_rows += int(k.shape[0])
            q_grouped = q.reshape(-1, kv_num_heads, heads_per_kv, head_dim)
            q_max = torch.maximum(q_max, q_grouped.abs().amax(dim=(0, 2)))
            k_max = torch.maximum(k_max, k.abs().amax(dim=0))

            sequence = int(q.shape[0])
            nystrom_probability: torch.Tensor | None = None
            if 2 <= sequence <= _V_PROFILE_MAX_SEQUENCE:
                full_query = q_grouped.permute(1, 2, 0, 3)
                full_logits = torch.matmul(
                    full_query,
                    k.permute(1, 2, 0).unsqueeze(1),
                ) * (float(head_dim) ** -0.5)
                full_probability = torch.softmax(full_logits, dim=-1).to(
                    torch.float64
                )
                nystrom_probability = full_probability.reshape(
                    kv_num_heads, -1, sequence
                )
                diagonal = full_probability.square().sum(dim=(1, 2)).cpu()
                edge = (
                    full_probability[..., :-1] * full_probability[..., 1:]
                ).sum(dim=(1, 2)).cpu()
                if sequence in profile_diagonal:
                    profile_diagonal[sequence] += diagonal
                    profile_edge[sequence] += edge
                    profile_count[sequence] += 1
                else:
                    profile_diagonal[sequence] = diagonal
                    profile_edge[sequence] = edge
                    profile_count[sequence] = 1

            query_bank = q_grouped.permute(1, 0, 2, 3).reshape(
                kv_num_heads, -1, head_dim
            )
            anchor_count = min(_V_PAIR_MAX_ANCHORS, int(query_bank.shape[1]))
            if anchor_count > 0 and int(k.shape[0]) > 1:
                if anchor_count == 1:
                    anchor_index = torch.zeros(
                        1, dtype=torch.long, device=query_bank.device
                    )
                else:
                    anchor_index = (
                        torch.arange(
                            anchor_count,
                            dtype=torch.long,
                            device=query_bank.device,
                        )
                        * (int(query_bank.shape[1]) - 1)
                        // (anchor_count - 1)
                    )
                anchors = query_bank.index_select(1, anchor_index)
                logits = torch.bmm(
                    anchors,
                    k.permute(1, 2, 0).contiguous(),
                ) * (float(head_dim) ** -0.5)
                probability = torch.softmax(logits, dim=-1).to(torch.float64)
                if nystrom_probability is None:
                    nystrom_probability = probability
                shard = sample_index % _V_PAIR_SHARDS
                pair_h0[shard] += probability.square().sum(dim=(1, 2)).cpu()
                pair_h1[shard] += (
                    probability[..., :-1] * probability[..., 1:]
                ).sum(dim=(1, 2)).cpu()
                pair_nonempty[shard] = True
                if sample_index < _V_PAIR_SHARDS and int(k.shape[0]) > 2:
                    lag2_h0[sample_index] += probability.square().sum(
                        dim=(1, 2)
                    ).cpu()
                    lag2_h2[sample_index] += (
                        probability[..., :-2] * probability[..., 2:]
                    ).sum(dim=(1, 2)).cpu()
                    lag2_nonempty[sample_index] = True

                sequence_length = int(k.shape[0])
                active_rank = min(_V_LOWRANK_RANK, sequence_length)
                basis = _normalized_dct_basis(
                    sequence_length,
                    active_rank,
                    probability.device,
                    torch.float64,
                )
                projected = torch.matmul(probability, basis)
                active_gram = torch.einsum(
                    "har,has->hrs", projected, projected
                )
                tau = probability.square().sum(dim=(1, 2))
                scale_value = tau / float(sequence_length)
                complement = (
                    tau - active_gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                ).clamp_min(0.0)
                delta_value = complement / float(
                    max(sequence_length - active_rank, 1)
                )
                delta_value = delta_value.clamp_min(_V_LOWRANK_EPS)
                sample_gram = torch.zeros(
                    kv_num_heads,
                    _V_LOWRANK_RANK,
                    _V_LOWRANK_RANK,
                    dtype=torch.float64,
                    device=probability.device,
                )
                sample_gram[:, :active_rank, :active_rank] = (
                    active_gram / scale_value[:, None, None].clamp_min(
                        _V_LOWRANK_EPS
                    )
                )
                sample_delta = delta_value / scale_value.clamp_min(
                    _V_LOWRANK_EPS
                )
                sample_usable = (
                    torch.isfinite(sample_gram).all(dim=(-2, -1))
                    & torch.isfinite(sample_delta)
                    & torch.isfinite(tau)
                    & (tau > _V_LOWRANK_EPS)
                )
                lowrank_shard = sample_index % _V_LOWRANK_SHARDS
                lowrank_gram_sum[lowrank_shard] += torch.where(
                    sample_usable[:, None, None], sample_gram, 0.0
                ).cpu()
                lowrank_delta_sum[lowrank_shard] += torch.where(
                    sample_usable, sample_delta, 0.0
                ).cpu()
                lowrank_count[lowrank_shard] += sample_usable.to(
                    torch.float64
                ).cpu()
                if _V_NYSTROM_ABLATION != "off":
                    active_nystrom_rank = min(
                        _V_LOWRANK_RANK, sequence_length
                    )
                    nystrom_probe = _normalized_dct_basis(
                        sequence_length,
                        active_nystrom_rank,
                        nystrom_probability.device,
                        torch.float64,
                    )
                    nystrom_projected = torch.matmul(
                        nystrom_probability, nystrom_probe
                    )
                    nystrom_y = torch.bmm(
                        nystrom_probability.transpose(1, 2),
                        nystrom_projected,
                    )
                    nystrom_m = torch.einsum(
                        "tr,hts->hrs", nystrom_probe, nystrom_y
                    )
                    nystrom_tau = nystrom_probability.square().sum(dim=(1, 2))
                    entry = nystrom_stats.get(sequence_length)
                    if entry is None:
                        nystrom_stats[sequence_length] = {
                            "y": nystrom_y.cpu(),
                            "m": nystrom_m.cpu(),
                            "tau": nystrom_tau.cpu(),
                            "count": 1,
                        }
                    else:
                        entry["y"] += nystrom_y.cpu()
                        entry["m"] += nystrom_m.cpu()
                        entry["tau"] += nystrom_tau.cpu()
                        entry["count"] += 1
    except (KeyError, TypeError, ValueError, RuntimeError):
        return _attention_fallback_state(q_num_heads, kv_num_heads, head_dim)
    if q_rows == 0 or kv_rows == 0:
        return _attention_fallback_state(q_num_heads, kv_num_heads, head_dim)

    q_max.clamp_min_(2.0 ** -24)
    k_max.clamp_min_(2.0 ** -24)
    selected_alpha = _attention_output_alpha_select(
        calib_qkv_list,
        q_num_heads,
        kv_num_heads,
        head_dim,
        q_max,
        k_max,
    )
    alpha_view = selected_alpha.reshape(kv_num_heads, 1)
    kv_smooth = (
        q_max.pow(alpha_view) / k_max.pow(1.0 - alpha_view)
    ).clamp(_SMOOTH_MIN, _SMOOTH_MAX)
    q_smooth = kv_smooth.repeat_interleave(heads_per_kv, dim=0)
    q_second = q_square_sum / q_rows
    k_second = k_square_sum / kv_rows
    v_second = v_square_sum / kv_rows
    q_importance = _normalize_importance(
        (k_second * kv_smooth.square()).repeat_interleave(heads_per_kv, dim=0)
    )
    k_importance = _normalize_importance(
        (q_second / q_smooth.square())
        .reshape(kv_num_heads, heads_per_kv, head_dim)
        .mean(dim=1)
    )
    v_importance = _normalize_importance(v_second + 0.05 * v_max.square())
    shard_finite = torch.isfinite(pair_h0) & torch.isfinite(pair_h1)
    shard_usable = pair_nonempty & shard_finite & (pair_h0 > 1.0e-24)
    shard_rho = torch.where(
        shard_usable,
        pair_h1 / pair_h0.clamp_min(1.0e-24),
        torch.zeros_like(pair_h0),
    )
    shard_direction = (~pair_nonempty) | (
        shard_usable & (shard_rho >= 0.0)
    )
    pair_valid = (
        (shard_usable.sum(dim=0) >= 2)
        & shard_direction.all(dim=0)
    )
    pair_rho = (
        pair_h1.sum(dim=0) / pair_h0.sum(dim=0).clamp_min(1.0e-24)
    ).clamp(0.0, _V_PAIR_RHO_LIMIT)
    pair_valid &= torch.isfinite(pair_rho) & (pair_rho > 0.0)
    pair_rho = torch.where(pair_valid, pair_rho, torch.zeros_like(pair_rho))
    lag2_finite = torch.isfinite(lag2_h0) & torch.isfinite(lag2_h2)
    lag2_usable = lag2_nonempty & lag2_finite & (lag2_h0 > 2.0 ** -80)
    lag2_ratio = torch.where(
        lag2_usable,
        lag2_h2 / lag2_h0.clamp_min(2.0 ** -80),
        torch.zeros_like(lag2_h0),
    )
    lag2_valid = (
        lag2_nonempty.all(dim=0)
        & lag2_usable.all(dim=0)
        & (lag2_ratio >= 0.0).all(dim=0)
        & pair_valid
    )
    rho2_raw = lag2_h2.sum(dim=0) / lag2_h0.sum(dim=0).clamp_min(
        2.0 ** -80
    )
    rho2_headroom = (_V_PAIR_RHO_LIMIT - pair_rho).clamp_min(0.0)
    pair_rho2 = torch.minimum(rho2_raw.clamp_min(0.0), rho2_headroom)
    lag2_valid &= torch.isfinite(pair_rho2) & (pair_rho2 > 0.0)
    pair_rho2 = torch.where(
        lag2_valid, pair_rho2, torch.zeros_like(pair_rho2)
    )

    lowrank_shard_usable = lowrank_count > 0.0
    lowrank_shard_gram = lowrank_gram_sum / lowrank_count.clamp_min(1.0)[
        :, :, None, None
    ]
    lowrank_shard_delta = lowrank_delta_sum / lowrank_count.clamp_min(1.0)
    lowrank_shard_number = lowrank_shard_usable.sum(dim=0)
    lowrank_gram = (
        lowrank_shard_gram
        * lowrank_shard_usable[:, :, None, None].to(torch.float64)
    ).sum(dim=0) / lowrank_shard_number.clamp_min(1)[:, None, None]
    lowrank_delta = (
        lowrank_shard_delta
        * lowrank_shard_usable.to(torch.float64)
    ).sum(dim=0) / lowrank_shard_number.clamp_min(1)
    lowrank_gram = 0.5 * (
        lowrank_gram + lowrank_gram.transpose(-2, -1)
    )
    lowrank_gram = (
        (1.0 - _V_LOWRANK_DIAGONAL_SHRINK) * lowrank_gram
        + _V_LOWRANK_DIAGONAL_SHRINK
        * torch.diag_embed(lowrank_gram.diagonal(dim1=-2, dim2=-1))
    )
    lowrank_finite = (
        torch.isfinite(lowrank_shard_gram).all(dim=(-2, -1)).all(dim=0)
        & torch.isfinite(lowrank_shard_delta).all(dim=0)
        & torch.isfinite(lowrank_gram).all(dim=(-2, -1))
        & torch.isfinite(lowrank_delta)
    )
    lowrank_psd = torch.linalg.eigvalsh(lowrank_gram).amin(dim=-1) >= -1.0e-8
    lowrank_identity = torch.eye(
        _V_LOWRANK_RANK, dtype=torch.float64
    ).unsqueeze(0)
    lowrank_anisotropy = torch.linalg.matrix_norm(
        lowrank_gram - lowrank_delta[:, None, None] * lowrank_identity,
        ord="fro",
    )
    lowrank_energy = (
        lowrank_gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        + lowrank_delta * _V_LOWRANK_RANK
    )
    lowrank_valid = (
        (lowrank_shard_number >= 2)
        & lowrank_finite
        & lowrank_psd
        & (lowrank_delta > 0.0)
        & (lowrank_energy > _V_LOWRANK_EPS)
        & (lowrank_anisotropy > 1.0e-6)
    )
    lowrank_gram = torch.where(
        lowrank_valid[:, None, None], lowrank_gram, 0.0
    )
    lowrank_delta = torch.where(lowrank_valid, lowrank_delta, 0.0)
    (
        nystrom_lengths,
        nystrom_factors,
        nystrom_diagonals,
        nystrom_valid,
    ) = _recover_nystrom_consumer_state(nystrom_stats, kv_num_heads)
    profile_lengths = torch.tensor(
        sorted(profile_diagonal), dtype=torch.int64
    )
    profile_unary = torch.zeros(
        int(profile_lengths.numel()),
        kv_num_heads,
        _V_PROFILE_MAX_SEQUENCE,
        dtype=torch.float32,
    )
    profile_coupling = torch.zeros(
        int(profile_lengths.numel()),
        kv_num_heads,
        _V_PROFILE_MAX_SEQUENCE - 1,
        dtype=torch.float32,
    )
    profile_valid = torch.zeros(
        int(profile_lengths.numel()), kv_num_heads, dtype=torch.bool
    )
    for profile_index, length_value in enumerate(profile_lengths.tolist()):
        diagonal = profile_diagonal[length_value] / profile_count[length_value]
        edge = profile_edge[length_value] / profile_count[length_value]
        normalizer = diagonal.mean(dim=-1, keepdim=True)
        head_valid = (
            pair_valid
            & torch.isfinite(diagonal).all(dim=-1)
            & torch.isfinite(edge).all(dim=-1)
            & torch.isfinite(normalizer).flatten()
            & (diagonal > 0.0).all(dim=-1)
            & (normalizer.flatten() > 1.0e-24)
        )
        diagonal = diagonal / normalizer.clamp_min(1.0e-24)
        edge = edge / normalizer.clamp_min(1.0e-24)
        geometric = torch.sqrt(
            diagonal[:, :-1].clamp_min(1.0e-24)
            * diagonal[:, 1:].clamp_min(1.0e-24)
        )
        correlation = (edge / geometric).clamp(0.0, _V_PAIR_RHO_LIMIT)
        coupling = correlation * geometric
        head_valid &= (
            torch.isfinite(diagonal).all(dim=-1)
            & torch.isfinite(coupling).all(dim=-1)
            & (diagonal > 0.0).all(dim=-1)
            & (coupling >= 0.0).all(dim=-1)
        )
        profile_unary[profile_index, :, :length_value] = diagonal.to(
            torch.float32
        )
        profile_coupling[profile_index, :, : length_value - 1] = coupling.to(
            torch.float32
        )
        profile_valid[profile_index] = head_valid
    if (
        q_importance is None
        or k_importance is None
        or v_importance is None
        or not bool(torch.isfinite(kv_smooth).all().item())
    ):
        return _attention_fallback_state(q_num_heads, kv_num_heads, head_dim)

    cross = _attention_cross_hessians(
        calib_qkv_list,
        q_num_heads,
        kv_num_heads,
        head_dim,
        q_smooth,
        kv_smooth,
    )
    if cross is None:
        return _attention_fallback_state(q_num_heads, kv_num_heads, head_dim)
    q_cross_hessian, q_cross_inverse, k_cross_hessian, k_cross_inverse = cross

    parent_state = {
        "q_state": {
            "smooth_scale": q_smooth.flatten().cpu(),
            "selected_alpha": selected_alpha.repeat_interleave(
                heads_per_kv
            ).cpu(),
            "rotation_group_size": 0,
            "rotation_reason": "stable-v3-basis-no-rotation",
            "importance": q_importance.flatten().cpu(),
            "min_proxy_gain": _ATTENTION_GAIN,
            "proxy_role": "q-error-weighted-by-k-second-moment",
            "cross_hessian": q_cross_hessian.cpu(),
            "cross_inverse": q_cross_inverse.cpu(),
            "hessian_head_dim": head_dim,
        },
        "k_state": {
            "smooth_scale": kv_smooth.flatten().cpu(),
            "selected_alpha": selected_alpha.cpu(),
            "rotation_group_size": 0,
            "rotation_reason": "stable-v3-basis-no-rotation",
            "importance": k_importance.flatten().cpu(),
            "min_proxy_gain": _ATTENTION_GAIN,
            "proxy_role": "k-error-weighted-by-gqa-q-second-moment",
            "cross_hessian": k_cross_hessian.cpu(),
            "cross_inverse": k_cross_inverse.cpu(),
            "hessian_head_dim": head_dim,
        },
        "v_state": {
            "importance": v_importance.flatten().cpu(),
            "min_proxy_gain": _ATTENTION_GAIN,
            "proxy_role": "v-energy-stable-upper-bound",
            "pair_schema": _V_PAIR_SCHEMA,
            "pair_rho": pair_rho.to(torch.float32).cpu(),
            "pair_valid": pair_valid.cpu(),
            "pair_lag2_schema": _V_LAG2_SCHEMA,
            "pair_rho2": pair_rho2.to(torch.float32).cpu(),
            "pair_lag2_valid": lag2_valid.cpu(),
            "profile_schema": _V_PROFILE_SCHEMA,
            "profile_lengths": profile_lengths.cpu(),
            "profile_unary": profile_unary.cpu(),
            "profile_edge": profile_coupling.cpu(),
            "profile_valid": profile_valid.cpu(),
            "lowrank_schema": _V_LOWRANK_SCHEMA,
            "lowrank_gram": lowrank_gram.to(torch.float32).cpu(),
            "lowrank_delta": lowrank_delta.to(torch.float32).cpu(),
            "lowrank_valid": lowrank_valid.cpu(),
            "nystrom_schema": _V_NYSTROM_SCHEMA,
            "nystrom_lengths": nystrom_lengths.cpu(),
            "nystrom_factors": nystrom_factors,
            "nystrom_diagonals": nystrom_diagonals,
            "nystrom_valid": nystrom_valid,
        },
    }
    paired_state = _attention_pair_state(
        calib_qkv_list,
        q_num_heads,
        kv_num_heads,
        head_dim,
        q_smooth,
        kv_smooth,
        parent_state,
    )
    wide_paired_state = _attention_attach_wide_cross_hessians(
        calib_qkv_list,
        q_num_heads,
        kv_num_heads,
        head_dim,
        q_smooth,
        kv_smooth,
        paired_state,
    )
    final_state = _attention_hierarchy_envelope_state(
        calib_qkv_list,
        q_num_heads,
        kv_num_heads,
        head_dim,
        q_smooth,
        kv_smooth,
        wide_paired_state,
    )
    # Dynamic descent needs H transposed both for the initial residual
    # projection and for indexed coordinate columns.  Materialize that layout
    # once per calibration group rather than five times in Dynamic-K.
    selected_state = dict(final_state)
    for role in ("q_state", "k_state"):
        role_state = final_state.get(role)
        if not isinstance(role_state, dict):
            continue
        wide_hessian = role_state.get("wide_cross_hessian")
        if type(wide_hessian) is not torch.Tensor:
            continue
        selected_role = dict(role_state)
        selected_role.pop("wide_cross_hessian", None)
        wide_hessian_t = wide_hessian.transpose(-1, -2).contiguous().cpu()
        selected_role["wide_cross_hessian_t"] = wide_hessian_t
        # The initial residual projection is intentionally BF16.  Cache that
        # representation once per calibration group instead of converting the
        # same covariance in every Dynamic Q/K call.
        selected_role["wide_cross_hessian_t_bf16"] = wide_hessian_t.to(
            torch.bfloat16
        )
        selected_state[role] = selected_role
    return _attention_build_repair_router(
        calib_qkv_list,
        q_num_heads,
        kv_num_heads,
        head_dim,
        selected_state,
    )


def _attention_hessian_repair(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Apply one natural-order repair with an opposite-role calibration Hessian."""
    channels = int(value.shape[-1])
    if (
        num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or channels != num_heads * head_dim
        or state.get("hessian_head_dim") != head_dim
    ):
        return parent
    block_count = head_dim // _BLOCK
    expected_shape = (num_heads, block_count, _BLOCK, _BLOCK)
    hessian = state.get("cross_hessian")
    inverse = state.get("cross_inverse")
    if (
        type(hessian) is not torch.Tensor
        or type(inverse) is not torch.Tensor
        or tuple(hessian.shape) != expected_shape
        or tuple(inverse.shape) != expected_shape
    ):
        return parent
    hessian = hessian.to(device=value.device, dtype=torch.float32).reshape(
        -1, _BLOCK, _BLOCK
    )
    inverse = inverse.to(device=value.device, dtype=torch.float32).reshape(
        -1, _BLOCK, _BLOCK
    )
    total_blocks = channels // _BLOCK
    rows = int(value.numel() // channels)
    target = value.reshape(rows, total_blocks, _BLOCK)
    try:
        effective = (
            parent["scale_factor"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
        ).expand_as(parent["mant"])
        scale = effective.reshape(rows, total_blocks, _BLOCK)
        parent_quantized = _dequantize_hif4(parent).reshape_as(target)
    except (KeyError, RuntimeError, ValueError):
        return parent
    working = target.clone()
    repaired = torch.empty_like(working)
    for position in range(_BLOCK):
        current = working[:, :, position]
        column_scale = scale[:, :, position]
        mant = (
            torch.round(current.abs() / column_scale * 4.0).clamp(0.0, 7.0)
            * 0.25
        )
        reconstructed = torch.sign(current) * mant * column_scale
        repaired[:, :, position] = reconstructed
        if position + 1 < _BLOCK:
            propagated = (current - reconstructed) / inverse[
                :, position, position
            ].clamp_min(1.0e-12).unsqueeze(0)
            working[:, :, position + 1 :] -= propagated.unsqueeze(-1) * inverse[
                :, position, position + 1 :
            ].unsqueeze(0)

    candidate_mant = (
        torch.round(repaired.abs() / scale * 4.0).clamp(0.0, 7.0) * 0.25
    )
    candidate_sign = torch.where(
        candidate_mant == 0.0, 0.0, torch.sign(repaired)
    )
    candidate_quantized = candidate_sign * candidate_mant * scale
    parent_residual = target - parent_quantized
    candidate_residual = target - candidate_quantized
    parent_proxy = torch.einsum(
        "rgi,gij,rgj->rg", parent_residual, hessian, parent_residual
    )
    candidate_proxy = torch.einsum(
        "rgi,gij,rgj->rg", candidate_residual, hessian, candidate_residual
    )
    use_candidate = (
        torch.isfinite(parent_proxy)
        & torch.isfinite(candidate_proxy)
        & (candidate_proxy < parent_proxy * (1.0 - _HESSIAN_GAIN))
    )
    mask = use_candidate.unsqueeze(-1)
    selected = dict(parent)
    selected["sign"] = torch.where(
        mask,
        candidate_sign.to(parent["sign"].dtype),
        parent["sign"].reshape(rows, total_blocks, _BLOCK),
    ).reshape_as(parent["sign"])
    selected["mant"] = torch.where(
        mask,
        candidate_mant.to(parent["mant"].dtype),
        parent["mant"].reshape(rows, total_blocks, _BLOCK),
    ).reshape_as(parent["mant"])
    return selected


def _attention_wide_fixed_scale_code_descent(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
    descent_steps: int,
    coordinates_per_step: int = 1,
) -> dict[str, torch.Tensor] | None:
    """Optimize final codes jointly in each calibrated full-head window."""
    channels = int(value.shape[-1])
    wide_window = state.get("wide_cross_window")
    if (
        type(wide_window) is not int
        or wide_window != _qk_wide_window(head_dim)
        or wide_window <= 0
        or head_dim % wide_window
        or channels != num_heads * head_dim
        or state.get("wide_cross_schema") != _QK_WIDE_SCHEMA
    ):
        return None
    windows_per_head = head_dim // wide_window
    expected = (
        num_heads,
        windows_per_head,
        wide_window,
        wide_window,
    )
    hessian_t = state.get("wide_cross_hessian_t")
    if type(hessian_t) is not torch.Tensor or tuple(hessian_t.shape) != expected:
        return None
    hessian_t = hessian_t.to(device=value.device, dtype=torch.float32).reshape(
        -1, wide_window, wide_window
    )
    hessian_t_bf16 = state.get("wide_cross_hessian_t_bf16")
    if (
        type(hessian_t_bf16) is not torch.Tensor
        or tuple(hessian_t_bf16.shape) != expected
    ):
        hessian_t_bf16 = hessian_t.to(torch.bfloat16)
    else:
        hessian_t_bf16 = hessian_t_bf16.to(
            device=value.device, dtype=torch.bfloat16
        ).reshape(-1, wide_window, wide_window)
    diagonal = torch.diagonal(hessian_t, dim1=-2, dim2=-1)
    total_windows = channels // wide_window
    rows = int(value.numel() // channels)
    target = value.reshape(rows, total_windows, wide_window)
    try:
        scale = (
            parent["scale_factor"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
            * 0.25
        ).expand_as(parent["mant"]).reshape(
            rows, total_windows, wide_window
        )
        parent_code = torch.round(
            parent["sign"].to(torch.float32)
            * parent["mant"].to(torch.float32)
            * 4.0
        ).reshape(rows, total_windows, wide_window)
    except (KeyError, RuntimeError, ValueError):
        return None
    # parent_code is freshly materialized by round(), so in-place refinement is safe.
    selected_code = parent_code
    projected_residual = torch.bmm(
        (target - parent_code * scale)
        .permute(1, 0, 2)
        .to(torch.bfloat16),
        hessian_t_bf16,
    ).permute(1, 0, 2).to(torch.float32)
    usable = torch.isfinite(diagonal) & (diagonal > 0.0)
    inverse_diagonal = diagonal.clamp_min(1.0e-24).reciprocal().unsqueeze(0)
    inverse_code_step = inverse_diagonal / scale
    expanded_diagonal = diagonal.unsqueeze(0)
    expanded_usable = usable.unsqueeze(0)
    hessian_columns = hessian_t.reshape(
        total_windows * wide_window, wide_window
    )
    offsets = (
        torch.arange(total_windows, device=value.device, dtype=torch.long)
        * wide_window
    ).unsqueeze(0)
    coordinates_per_step = min(max(int(coordinates_per_step), 1), 2)
    window_batch = torch.arange(
        total_windows, device=value.device, dtype=torch.long
    ).unsqueeze(0).expand(rows, -1)
    paired_offsets = offsets.unsqueeze(-1)
    for _ in range(descent_steps):
        candidate_code = torch.round(projected_residual * inverse_code_step)
        candidate_code.add_(selected_code).clamp_(-7.0, 7.0)
        reconstruction_delta = candidate_code - selected_code
        reconstruction_delta.mul_(scale)
        linear_delta = reconstruction_delta * -2.0
        linear_delta.mul_(projected_residual)
        coordinate_delta = reconstruction_delta.square()
        coordinate_delta.mul_(expanded_diagonal).add_(linear_delta)
        valid = expanded_usable & (candidate_code != selected_code)
        coordinate_delta.masked_fill_(~valid, torch.inf)
        if coordinates_per_step == 1:
            best_delta, best_index = coordinate_delta.min(dim=-1)
            gather_index = best_index.unsqueeze(-1)
            current_code = selected_code.gather(-1, gather_index)
            proposed_code = candidate_code.gather(-1, gather_index)
            accept = torch.isfinite(best_delta) & (best_delta < 0.0)
            accepted_code = torch.where(
                accept.unsqueeze(-1), proposed_code, current_code
            )
            accepted_delta = (
                (accepted_code - current_code)
                * scale.gather(-1, gather_index)
            ).squeeze(-1)
            selected_code.scatter_(-1, gather_index, accepted_code)
            selected_column = hessian_columns.index_select(
                0, (offsets + best_index).reshape(-1)
            ).reshape(rows, total_windows, wide_window)
            projected_residual -= selected_column * accepted_delta.unsqueeze(-1)
            continue

        best_delta, best_index = torch.topk(
            coordinate_delta,
            coordinates_per_step,
            dim=-1,
            largest=False,
            sorted=True,
        )
        current_code = selected_code.gather(-1, best_index)
        proposed_code = candidate_code.gather(-1, best_index)
        selected_scale = scale.gather(-1, best_index)
        proposed_delta = (proposed_code - current_code) * selected_scale
        cross = hessian_t[
            window_batch,
            best_index[..., 0],
            best_index[..., 1],
        ]
        joint_delta = (
            best_delta.sum(dim=-1)
            + 2.0
            * proposed_delta[..., 0]
            * proposed_delta[..., 1]
            * cross
        )
        accept_pair = torch.isfinite(joint_delta) & (joint_delta < 0.0)
        accept_first = (
            ~accept_pair
            & torch.isfinite(best_delta[..., 0])
            & (best_delta[..., 0] < 0.0)
        )
        accepted_mask = torch.stack(
            (accept_pair | accept_first, accept_pair), dim=-1
        )
        accepted_code = torch.where(
            accepted_mask, proposed_code, current_code
        )
        proposed_delta.masked_fill_(~accepted_mask, 0.0)
        selected_code.scatter_(-1, best_index, accepted_code)
        selected_columns = hessian_columns.index_select(
            0, (paired_offsets + best_index).reshape(-1)
        ).reshape(
            rows,
            total_windows,
            coordinates_per_step,
            wide_window,
        )
        residual_update = (
            selected_columns[..., 0, :]
            * proposed_delta[..., 0].unsqueeze(-1)
        )
        residual_update.add_(
            selected_columns[..., 1, :]
            * proposed_delta[..., 1].unsqueeze(-1)
        )
        projected_residual.sub_(residual_update)

    # selected_code begins as parent_code and only changes on finite, strictly
    # negative moves.  Re-encoding every window directly is therefore exactly
    # equivalent to the former final mask, including untouched windows.
    selected = dict(parent)
    selected["sign"] = torch.sign(selected_code).to(
        parent["sign"].dtype
    ).reshape_as(parent["sign"])
    selected["mant"] = (selected_code.abs() * 0.25).to(
        parent["mant"].dtype
    ).reshape_as(parent["mant"])
    return selected


def _attention_wide_screened_code_descent(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
    descent_steps: int,
) -> dict[str, torch.Tensor] | None:
    """Screen with H64 risks, then score a few coordinates in exact H256 risk."""
    channels = int(value.shape[-1])
    if (
        head_dim % _QK_WIDE_WINDOW
        or channels != num_heads * head_dim
        or state.get("wide_cross_schema") != _QK_WIDE_SCHEMA
        or state.get("hessian_head_dim") != head_dim
    ):
        return None
    windows_per_head = head_dim // _QK_WIDE_WINDOW
    blocks_per_head = head_dim // _BLOCK
    expected_wide = (
        num_heads,
        windows_per_head,
        _QK_WIDE_WINDOW,
        _QK_WIDE_WINDOW,
    )
    expected_block = (num_heads, blocks_per_head, _BLOCK, _BLOCK)
    wide_hessian = state.get("wide_cross_hessian")
    block_hessian = state.get("cross_hessian")
    if (
        type(wide_hessian) is not torch.Tensor
        or tuple(wide_hessian.shape) != expected_wide
        or type(block_hessian) is not torch.Tensor
        or tuple(block_hessian.shape) != expected_block
    ):
        return None
    wide_hessian = wide_hessian.to(
        device=value.device, dtype=torch.float32
    ).reshape(-1, _QK_WIDE_WINDOW, _QK_WIDE_WINDOW)
    block_hessian = block_hessian.to(
        device=value.device, dtype=torch.float32
    ).reshape(-1, _BLOCK, _BLOCK)
    wide_diagonal = torch.diagonal(wide_hessian, dim1=-2, dim2=-1)
    block_diagonal = torch.diagonal(block_hessian, dim1=-2, dim2=-1)
    usable = torch.isfinite(block_diagonal) & (block_diagonal > 0.0)
    total_windows = channels // _QK_WIDE_WINDOW
    total_blocks = channels // _BLOCK
    rows = int(value.numel() // channels)
    target = value.reshape(rows, total_windows, _QK_WIDE_WINDOW)
    try:
        scale = (
            parent["scale_factor"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
            * 0.25
        ).expand_as(parent["mant"]).reshape(
            rows, total_windows, _QK_WIDE_WINDOW
        )
        parent_code = torch.round(
            parent["sign"].to(torch.float32)
            * parent["mant"].to(torch.float32)
            * 4.0
        ).reshape(rows, total_windows, _QK_WIDE_WINDOW)
    except (KeyError, RuntimeError, ValueError):
        return None
    selected_code = parent_code.clone()
    selected_quantized = parent_code * scale
    cumulative_delta = torch.zeros(
        rows, total_windows, device=value.device, dtype=torch.float32
    )
    screen = min(_QK_WIDE_SCREEN, _QK_WIDE_WINDOW)
    wide_batch = torch.arange(total_windows, device=value.device).reshape(
        1, total_windows, 1
    )
    for _ in range(descent_steps):
        residual_blocks = (target - selected_quantized).reshape(
            rows, total_blocks, _BLOCK
        )
        block_gradient = torch.einsum(
            "gij,rgj->rgi", block_hessian, residual_blocks
        )
        block_scale = scale.reshape(rows, total_blocks, _BLOCK)
        block_code = selected_code.reshape(rows, total_blocks, _BLOCK)
        block_quantized = selected_quantized.reshape(
            rows, total_blocks, _BLOCK
        )
        proposed_code = torch.round(
            (
                block_quantized
                + block_gradient
                / block_diagonal.clamp_min(1.0e-24).unsqueeze(0)
            )
            / block_scale
        ).clamp(-7.0, 7.0)
        proposed_delta = (proposed_code - block_code) * block_scale
        approximate_delta = (
            -2.0 * proposed_delta * block_gradient
            + block_diagonal.unsqueeze(0) * proposed_delta.square()
        )
        approximate_valid = (
            usable.unsqueeze(0)
            & torch.isfinite(proposed_code)
            & torch.isfinite(approximate_delta)
            & (proposed_code != block_code)
        )
        approximate_delta = torch.where(
            approximate_valid,
            approximate_delta,
            torch.full_like(approximate_delta, torch.inf),
        ).reshape(rows, total_windows, _QK_WIDE_WINDOW)
        screened_index = torch.topk(
            approximate_delta,
            screen,
            dim=-1,
            largest=False,
            sorted=False,
        ).indices
        proposed_wide = proposed_code.reshape(
            rows, total_windows, _QK_WIDE_WINDOW
        )
        for row_start in range(0, rows, _QK_WIDE_ROW_CHUNK):
            row_stop = min(row_start + _QK_WIDE_ROW_CHUNK, rows)
            chunk_index = screened_index[row_start:row_stop]
            chunk_rows = row_stop - row_start
            hessian_rows = wide_hessian[
                wide_batch.expand(chunk_rows, -1, screen), chunk_index
            ]
            chunk_residual = (
                target[row_start:row_stop]
                - selected_quantized[row_start:row_stop]
            )
            exact_gradient = (
                hessian_rows * chunk_residual.unsqueeze(2)
            ).sum(dim=-1)
            current = selected_code[row_start:row_stop].gather(-1, chunk_index)
            candidate = proposed_wide[row_start:row_stop].gather(
                -1, chunk_index
            )
            chunk_scale = scale[row_start:row_stop].gather(-1, chunk_index)
            delta = (candidate - current) * chunk_scale
            diagonal = wide_diagonal.unsqueeze(0).expand(
                chunk_rows, -1, -1
            ).gather(-1, chunk_index)
            exact_delta = (
                -2.0 * delta * exact_gradient + diagonal * delta.square()
            )
            exact_valid = (
                torch.isfinite(exact_delta)
                & torch.isfinite(candidate)
                & (candidate != current)
            )
            exact_delta = torch.where(
                exact_valid,
                exact_delta,
                torch.full_like(exact_delta, torch.inf),
            )
            best_delta, best_slot = exact_delta.min(dim=-1)
            best_index = chunk_index.gather(
                -1, best_slot.unsqueeze(-1)
            )
            current_best = selected_code[row_start:row_stop].gather(
                -1, best_index
            )
            candidate_best = proposed_wide[row_start:row_stop].gather(
                -1, best_index
            )
            accept = torch.isfinite(best_delta) & (best_delta < 0.0)
            accepted_code = torch.where(
                accept.unsqueeze(-1), candidate_best, current_best
            )
            accepted_delta = (
                (accepted_code - current_best)
                * scale[row_start:row_stop].gather(-1, best_index)
            )
            selected_code[row_start:row_stop].scatter_(
                -1, best_index, accepted_code
            )
            selected_quantized[row_start:row_stop].scatter_(
                -1,
                best_index,
                selected_quantized[row_start:row_stop].gather(-1, best_index)
                + accepted_delta,
            )
            cumulative_delta[row_start:row_stop] += torch.where(
                accept, best_delta, torch.zeros_like(best_delta)
            )

    use_candidate = (
        (selected_code != parent_code).any(dim=-1)
        & torch.isfinite(cumulative_delta)
        & (cumulative_delta < 0.0)
    )
    candidate_mant = selected_code.abs() * 0.25
    candidate_sign = torch.where(
        candidate_mant == 0.0,
        torch.zeros_like(selected_code),
        torch.sign(selected_code),
    )
    mask = use_candidate.unsqueeze(-1)
    selected = dict(parent)
    selected["sign"] = torch.where(
        mask,
        candidate_sign.to(parent["sign"].dtype),
        parent["sign"].reshape(rows, total_windows, _QK_WIDE_WINDOW),
    ).reshape_as(parent["sign"])
    selected["mant"] = torch.where(
        mask,
        candidate_mant.to(parent["mant"].dtype),
        parent["mant"].reshape(rows, total_windows, _QK_WIDE_WINDOW),
    ).reshape_as(parent["mant"])
    return selected


def _attention_k_fixed_scale_code_descent(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Run coupled legal-code descent while keeping all hierarchy scales frozen."""
    if _K_CODE_DESCENT_ABLATION not in ("full", "off", "mechanism-off"):
        return parent
    descent_steps = (
        1
        if _K_CODE_DESCENT_ABLATION in ("off", "mechanism-off")
        else _K_CODE_DESCENT_STEPS
    )
    channels = int(value.shape[-1])
    rows = int(value.numel() // channels)
    wide = (
        _attention_wide_fixed_scale_code_descent(
            value,
            parent,
            state,
            num_heads,
            head_dim,
            _QK_WIDE_K_STEPS,
        )
        if (
            _QK_WIDE_K_STEPS > 0
            and head_dim > 64
            and (
                state.get("repair_router_force_wide") is True
                or head_dim >= 256
                or rows > 10
            )
        )
        else None
    )
    if wide is not None:
        return wide
    if (
        num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or channels != num_heads * head_dim
        or state.get("hessian_head_dim") != head_dim
    ):
        return parent
    block_count = head_dim // _BLOCK
    expected_shape = (num_heads, block_count, _BLOCK, _BLOCK)
    hessian = state.get("cross_hessian")
    if type(hessian) is not torch.Tensor or tuple(hessian.shape) != expected_shape:
        return parent
    hessian = hessian.to(device=value.device, dtype=torch.float32).reshape(
        -1, _BLOCK, _BLOCK
    )
    diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)

    total_blocks = channels // _BLOCK
    target = value.reshape(rows, total_blocks, _BLOCK)
    try:
        scale = (
            parent["scale_factor"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
            * 0.25
        ).expand_as(parent["mant"]).reshape(rows, total_blocks, _BLOCK)
        parent_code = torch.round(
            parent["sign"].to(torch.float32)
            * parent["mant"].to(torch.float32)
            * 4.0
        ).reshape(rows, total_blocks, _BLOCK)
    except (KeyError, RuntimeError, ValueError):
        return parent
    parent_quantized = parent_code * scale
    parent_residual = target - parent_quantized
    selected_code = parent_code.clone()
    selected_quantized = parent_quantized.clone()
    projected_residual = torch.einsum(
        "gij,rgj->rgi", hessian, parent_residual
    )
    cumulative_delta = torch.zeros(
        rows, total_blocks, device=value.device, dtype=torch.float32
    )
    usable = torch.isfinite(diagonal) & (diagonal > 0.0)
    hessian_columns = hessian.transpose(-1, -2).contiguous().reshape(
        total_blocks * _BLOCK, _BLOCK
    )
    block_offsets = (
        torch.arange(total_blocks, device=value.device, dtype=torch.long)
        * _BLOCK
    ).unsqueeze(0)
    for _ in range(descent_steps):
        continuous_delta = (
            projected_residual
            / diagonal.clamp_min(1.0e-24).unsqueeze(0)
        )
        candidate_code = torch.round(
            (selected_quantized + continuous_delta) / scale
        ).clamp(-7.0, 7.0)
        reconstruction_delta = (candidate_code - selected_code) * scale
        coordinate_delta = (
            -2.0 * reconstruction_delta * projected_residual
            + diagonal.unsqueeze(0) * reconstruction_delta.square()
        )
        coordinate_valid = (
            usable.unsqueeze(0)
            & torch.isfinite(candidate_code)
            & torch.isfinite(coordinate_delta)
            & (candidate_code != selected_code)
        )
        coordinate_delta = torch.where(
            coordinate_valid,
            coordinate_delta,
            torch.full_like(coordinate_delta, torch.inf),
        )
        best_delta, best_index = coordinate_delta.min(dim=-1)
        best_index_expanded = best_index.unsqueeze(-1)
        current_at_best = selected_code.gather(-1, best_index_expanded)
        candidate_at_best = candidate_code.gather(-1, best_index_expanded)
        accept = torch.isfinite(best_delta) & (best_delta < 0.0)
        accepted_code = torch.where(
            accept.unsqueeze(-1), candidate_at_best, current_at_best
        )
        accepted_delta = (
            (accepted_code - current_at_best)
            * scale.gather(-1, best_index_expanded)
        ).squeeze(-1)
        cumulative_delta = cumulative_delta + torch.where(
            accept, best_delta, torch.zeros_like(best_delta)
        )
        selected_code.scatter_(-1, best_index_expanded, accepted_code)
        selected_quantized.scatter_(
            -1,
            best_index_expanded,
            selected_quantized.gather(-1, best_index_expanded)
            + accepted_delta.unsqueeze(-1),
        )
        selected_column = hessian_columns.index_select(
            0, (block_offsets + best_index).reshape(-1)
        ).reshape(rows, total_blocks, _BLOCK)
        projected_residual = (
            projected_residual
            - selected_column * accepted_delta.unsqueeze(-1)
        )

    use_candidate = (
        (selected_code != parent_code).any(dim=-1)
        & torch.isfinite(cumulative_delta)
        & (cumulative_delta < 0.0)
    )
    mask = use_candidate.unsqueeze(-1)
    candidate_mant = selected_code.abs() * 0.25
    candidate_sign = torch.where(
        candidate_mant == 0.0,
        torch.zeros_like(selected_code),
        torch.sign(selected_code),
    )
    selected = dict(parent)
    selected["sign"] = torch.where(
        mask,
        candidate_sign.to(parent["sign"].dtype),
        parent["sign"].reshape(rows, total_blocks, _BLOCK),
    ).reshape_as(parent["sign"])
    selected["mant"] = torch.where(
        mask,
        candidate_mant.to(parent["mant"].dtype),
        parent["mant"].reshape(rows, total_blocks, _BLOCK),
    ).reshape_as(parent["mant"])
    return selected


def _attention_k_two_coordinate_code_solve(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Solve screened two-code integer quadratic slices with scales frozen."""
    if _K_PAIR_CODE_ABLATION != "full":
        return parent
    if int(value.shape[-2]) > _K_PAIR_CODE_SEQUENCE_LIMIT:
        return parent
    channels = int(value.shape[-1])
    if (
        num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or channels != num_heads * head_dim
        or state.get("hessian_head_dim") != head_dim
    ):
        return parent
    block_count = head_dim // _BLOCK
    expected_shape = (num_heads, block_count, _BLOCK, _BLOCK)
    hessian = state.get("cross_hessian")
    if type(hessian) is not torch.Tensor or tuple(hessian.shape) != expected_shape:
        return parent
    hessian = hessian.to(device=value.device, dtype=torch.float32).reshape(
        -1, _BLOCK, _BLOCK
    )
    diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)
    if (
        not bool(torch.isfinite(hessian).all().item())
        or bool((diagonal < 0.0).any().item())
    ):
        return parent

    total_blocks = channels // _BLOCK
    rows = int(value.numel() // channels)
    target = value.reshape(rows, total_blocks, _BLOCK)
    try:
        scale = (
            parent["scale_factor"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
            * 0.25
        ).expand_as(parent["mant"]).reshape(rows, total_blocks, _BLOCK)
        parent_code = torch.round(
            parent["sign"].to(torch.float32)
            * parent["mant"].to(torch.float32)
            * 4.0
        ).reshape(rows, total_blocks, _BLOCK)
    except (KeyError, RuntimeError, ValueError):
        return parent
    if (
        not bool(torch.isfinite(scale).all().item())
        or not bool((scale > 0.0).all().item())
        or not bool(torch.isfinite(parent_code).all().item())
        or bool((parent_code.abs() > 7.0).any().item())
    ):
        return parent

    parent_quantized = parent_code * scale
    parent_residual = target - parent_quantized
    projected_residual = torch.einsum(
        "gij,rgj->rgi", hessian, parent_residual
    )
    usable = torch.isfinite(diagonal) & (diagonal > 0.0)
    continuous_delta = projected_residual / diagonal.clamp_min(1.0e-24).unsqueeze(0)
    free_code = torch.round(
        (parent_quantized + continuous_delta) / scale
    ).clamp(-7.0, 7.0)

    left_code = (parent_code - 1.0).clamp(-7.0, 7.0)
    right_code = (parent_code + 1.0).clamp(-7.0, 7.0)
    left_delta = (left_code - parent_code) * scale
    right_delta = (right_code - parent_code) * scale
    left_risk = (
        -2.0 * left_delta * projected_residual
        + diagonal.unsqueeze(0) * left_delta.square()
    )
    right_risk = (
        -2.0 * right_delta * projected_residual
        + diagonal.unsqueeze(0) * right_delta.square()
    )
    left_risk = torch.where(
        parent_code > -7.0, left_risk, torch.full_like(left_risk, torch.inf)
    )
    right_risk = torch.where(
        parent_code < 7.0, right_risk, torch.full_like(right_risk, torch.inf)
    )
    neighbor_code = torch.where(right_risk < left_risk, right_code, left_code)
    barrier_code = torch.where(
        free_code != parent_code, free_code, neighbor_code
    )
    barrier_delta = (barrier_code - parent_code) * scale
    barrier_risk = (
        -2.0 * barrier_delta * projected_residual
        + diagonal.unsqueeze(0) * barrier_delta.square()
    )
    barrier_valid = (
        usable.unsqueeze(0)
        & torch.isfinite(barrier_code)
        & torch.isfinite(barrier_risk)
        & (barrier_code != parent_code)
    )
    barrier_risk = torch.where(
        barrier_valid, barrier_risk, torch.full_like(barrier_risk, torch.inf)
    )
    screened = torch.topk(
        barrier_risk,
        k=_K_PAIR_CODE_SCREEN,
        dim=-1,
        largest=False,
        sorted=True,
    ).indices

    pair_left = torch.tensor(
        (0, 0, 0, 1, 1, 2), dtype=torch.long, device=value.device
    )
    pair_right = torch.tensor(
        (1, 2, 3, 2, 3, 3), dtype=torch.long, device=value.device
    )
    index_i = screened.index_select(-1, pair_left)
    index_j = screened.index_select(-1, pair_right)
    diagonal_view = diagonal.unsqueeze(0).expand(rows, -1, -1)
    group_index = torch.arange(
        total_blocks, dtype=torch.long, device=value.device
    ).reshape(1, total_blocks, 1).expand_as(index_i)

    barrier_i = torch.gather(barrier_code, -1, index_i)
    barrier_j = torch.gather(barrier_code, -1, index_j)
    parent_pair_i = torch.gather(parent_code, -1, index_i)
    parent_pair_j = torch.gather(parent_code, -1, index_j)
    scale_pair_i = torch.gather(scale, -1, index_i)
    scale_pair_j = torch.gather(scale, -1, index_j)
    gradient_pair_i = torch.gather(projected_residual, -1, index_i)
    gradient_pair_j = torch.gather(projected_residual, -1, index_j)
    hessian_pair_ii = torch.gather(diagonal_view, -1, index_i)
    hessian_pair_jj = torch.gather(diagonal_view, -1, index_j)
    hessian_pair_ij = hessian[group_index, index_i, index_j]
    barrier_delta_i = (barrier_i - parent_pair_i) * scale_pair_i
    barrier_delta_j = (barrier_j - parent_pair_j) * scale_pair_j
    screened_pair_delta = (
        -2.0 * barrier_delta_i * gradient_pair_i
        - 2.0 * barrier_delta_j * gradient_pair_j
        + hessian_pair_ii * barrier_delta_i.square()
        + hessian_pair_jj * barrier_delta_j.square()
        + 2.0 * hessian_pair_ij * barrier_delta_i * barrier_delta_j
    )
    screened_pair_valid = (
        barrier_valid.gather(-1, index_i)
        & barrier_valid.gather(-1, index_j)
        & torch.isfinite(screened_pair_delta)
    )
    screened_pair_delta = torch.where(
        screened_pair_valid,
        screened_pair_delta,
        torch.full_like(screened_pair_delta, torch.inf),
    )
    best_screen_pair = screened_pair_delta.argmin(dim=-1, keepdim=True)
    index_i = index_i.gather(-1, best_screen_pair).squeeze(-1)
    index_j = index_j.gather(-1, best_screen_pair).squeeze(-1)

    parent_i = torch.gather(parent_code, -1, index_i.unsqueeze(-1)).squeeze(-1)
    parent_j = torch.gather(parent_code, -1, index_j.unsqueeze(-1)).squeeze(-1)
    scale_i = torch.gather(scale, -1, index_i.unsqueeze(-1)).squeeze(-1)
    scale_j = torch.gather(scale, -1, index_j.unsqueeze(-1)).squeeze(-1)
    gradient_i = torch.gather(
        projected_residual, -1, index_i.unsqueeze(-1)
    ).squeeze(-1)
    gradient_j = torch.gather(
        projected_residual, -1, index_j.unsqueeze(-1)
    ).squeeze(-1)
    hessian_ii = torch.gather(
        diagonal_view, -1, index_i.unsqueeze(-1)
    ).squeeze(-1)
    hessian_jj = torch.gather(
        diagonal_view, -1, index_j.unsqueeze(-1)
    ).squeeze(-1)
    group_index = torch.arange(
        total_blocks, dtype=torch.long, device=value.device
    ).reshape(1, total_blocks).expand_as(index_i)
    hessian_ij = hessian[group_index, index_i, index_j]

    legal_code = torch.arange(
        -7.0, 8.0, dtype=torch.float32, device=value.device
    ).reshape(1, 1, 15)
    candidate_i = legal_code.expand(rows, total_blocks, -1)
    delta_i = (candidate_i - parent_i.unsqueeze(-1)) * scale_i.unsqueeze(-1)
    valid_i = candidate_i != parent_i.unsqueeze(-1)

    conditional_delta_j = (
        gradient_j.unsqueeze(-1) - hessian_ij.unsqueeze(-1) * delta_i
    ) / hessian_jj.clamp_min(1.0e-24).unsqueeze(-1)
    free_j = torch.round(
        parent_j.unsqueeze(-1) + conditional_delta_j / scale_j.unsqueeze(-1)
    ).clamp(-7.0, 7.0)
    left_j = (parent_j - 1.0).clamp(-7.0, 7.0).unsqueeze(-1)
    right_j = (parent_j + 1.0).clamp(-7.0, 7.0).unsqueeze(-1)
    delta_left_j = (left_j - parent_j.unsqueeze(-1)) * scale_j.unsqueeze(-1)
    delta_right_j = (right_j - parent_j.unsqueeze(-1)) * scale_j.unsqueeze(-1)
    left_conditional_risk = (
        -2.0 * delta_left_j * gradient_j.unsqueeze(-1)
        + hessian_jj.unsqueeze(-1) * delta_left_j.square()
        + 2.0 * hessian_ij.unsqueeze(-1) * delta_i * delta_left_j
    )
    right_conditional_risk = (
        -2.0 * delta_right_j * gradient_j.unsqueeze(-1)
        + hessian_jj.unsqueeze(-1) * delta_right_j.square()
        + 2.0 * hessian_ij.unsqueeze(-1) * delta_i * delta_right_j
    )
    left_conditional_risk = torch.where(
        (parent_j > -7.0).unsqueeze(-1),
        left_conditional_risk,
        torch.full_like(left_conditional_risk, torch.inf),
    )
    right_conditional_risk = torch.where(
        (parent_j < 7.0).unsqueeze(-1),
        right_conditional_risk,
        torch.full_like(right_conditional_risk, torch.inf),
    )
    neighbor_j = torch.where(
        right_conditional_risk < left_conditional_risk, right_j, left_j
    )
    candidate_j = torch.where(
        free_j != parent_j.unsqueeze(-1), free_j, neighbor_j
    )
    delta_j = (candidate_j - parent_j.unsqueeze(-1)) * scale_j.unsqueeze(-1)
    pair_delta = (
        -2.0 * delta_i * gradient_i.unsqueeze(-1)
        - 2.0 * delta_j * gradient_j.unsqueeze(-1)
        + hessian_ii.unsqueeze(-1) * delta_i.square()
        + hessian_jj.unsqueeze(-1) * delta_j.square()
        + 2.0 * hessian_ij.unsqueeze(-1) * delta_i * delta_j
    )
    valid_pair = (
        valid_i
        & (hessian_ii > 0.0).unsqueeze(-1)
        & (hessian_jj > 0.0).unsqueeze(-1)
        & torch.isfinite(pair_delta)
        & (candidate_j != parent_j.unsqueeze(-1))
    )
    pair_delta = torch.where(
        valid_pair, pair_delta, torch.full_like(pair_delta, torch.inf)
    )
    best_delta, best_code_slot = pair_delta.min(dim=-1)
    best_i = index_i.unsqueeze(-1)
    best_j = index_j.unsqueeze(-1)
    best_i_code = candidate_i.gather(-1, best_code_slot.unsqueeze(-1))
    best_j_code = candidate_j.gather(-1, best_code_slot.unsqueeze(-1))

    selected_code = parent_code.clone()
    selected_code.scatter_(-1, best_i, best_i_code)
    selected_code.scatter_(-1, best_j, best_j_code)
    changed = (selected_code != parent_code).sum(dim=-1)
    use_candidate = (
        torch.isfinite(best_delta)
        & (best_delta < 0.0)
        & (changed == 2)
    )
    mask = use_candidate.unsqueeze(-1)
    candidate_mant = selected_code.abs() * 0.25
    candidate_sign = torch.where(
        candidate_mant == 0.0,
        torch.zeros_like(selected_code),
        torch.sign(selected_code),
    )
    selected = dict(parent)
    selected["sign"] = torch.where(
        mask,
        candidate_sign.to(parent["sign"].dtype),
        parent["sign"].reshape(rows, total_blocks, _BLOCK),
    ).reshape_as(parent["sign"])
    selected["mant"] = torch.where(
        mask,
        candidate_mant.to(parent["mant"].dtype),
        parent["mant"].reshape(rows, total_blocks, _BLOCK),
    ).reshape_as(parent["mant"])
    return selected


def _attention_q_fixed_scale_code_descent(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Refine final Q codes under a bounded conditional-search budget."""
    if _Q_CODE_DESCENT_ABLATION in ("off", "mechanism-off"):
        return parent
    if _Q_CODE_DESCENT_ABLATION != "full":
        return parent
    rows = int(value.numel() // int(value.shape[-1]))
    total_blocks = int(value.shape[-1]) // _BLOCK
    descent_steps = min(
        _Q_CODE_DESCENT_STEPS,
        3
        + 2
        * int(
            rows * total_blocks
            <= _Q_CODE_DESCENT_THIRD_STEP_COORD_BUDGET
        ),
    )
    wide = (
        _attention_wide_fixed_scale_code_descent(
            value,
            parent,
            state,
            num_heads,
            head_dim,
            _qk_wide_q_steps(head_dim),
            2 if head_dim == _QK_WIDE_MAX_WINDOW else 1,
        )
        if (
            _qk_wide_q_steps(head_dim) > 0
            and (
                state.get("repair_router_force_wide") is True
                or (head_dim == _QK_WIDE_MAX_WINDOW and rows >= 128)
                or (64 < head_dim < _QK_WIDE_MAX_WINDOW and rows >= 512)
            )
        )
        else None
    )
    if wide is not None:
        return wide
    channels = int(value.shape[-1])
    if (
        num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or channels != num_heads * head_dim
        or state.get("hessian_head_dim") != head_dim
    ):
        return parent
    block_count = head_dim // _BLOCK
    expected_shape = (num_heads, block_count, _BLOCK, _BLOCK)
    hessian = state.get("cross_hessian")
    if type(hessian) is not torch.Tensor or tuple(hessian.shape) != expected_shape:
        return parent
    hessian = hessian.to(device=value.device, dtype=torch.float32).reshape(
        -1, _BLOCK, _BLOCK
    )
    diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)

    total_blocks = channels // _BLOCK
    rows = int(value.numel() // channels)
    target = value.reshape(rows, total_blocks, _BLOCK)
    try:
        scale = (
            parent["scale_factor"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
            * 0.25
        ).expand_as(parent["mant"]).reshape(rows, total_blocks, _BLOCK)
        parent_code = torch.round(
            parent["sign"].to(torch.float32)
            * parent["mant"].to(torch.float32)
            * 4.0
        ).reshape(rows, total_blocks, _BLOCK)
    except (KeyError, RuntimeError, ValueError):
        return parent
    parent_quantized = parent_code * scale
    parent_residual = target - parent_quantized
    selected_code = parent_code.clone()
    selected_quantized = parent_quantized.clone()
    projected_residual = torch.einsum(
        "gij,rgj->rgi", hessian, parent_residual
    )
    cumulative_delta = torch.zeros(
        rows, total_blocks, device=value.device, dtype=torch.float32
    )
    usable = torch.isfinite(diagonal) & (diagonal > 0.0)
    hessian_columns = hessian.transpose(-1, -2).contiguous().reshape(
        total_blocks * _BLOCK, _BLOCK
    )
    block_offsets = (
        torch.arange(total_blocks, device=value.device, dtype=torch.long)
        * _BLOCK
    ).unsqueeze(0)
    # A third conditional move is useful on modest Q tensors, but its dense
    # coordinate scan has diminishing system-level value once the live
    # row-by-H64 search space becomes large.  Keep two moves universally and
    # spend the third only inside a shape-derived work budget.  The route is
    # independent of case identity and data values.
    for _ in range(descent_steps):
        continuous_delta = (
            projected_residual
            / diagonal.clamp_min(1.0e-24).unsqueeze(0)
        )
        candidate_code = torch.round(
            (selected_quantized + continuous_delta) / scale
        ).clamp(-7.0, 7.0)
        reconstruction_delta = (candidate_code - selected_code) * scale
        coordinate_delta = (
            -2.0 * reconstruction_delta * projected_residual
            + diagonal.unsqueeze(0) * reconstruction_delta.square()
        )
        coordinate_valid = (
            usable.unsqueeze(0)
            & torch.isfinite(candidate_code)
            & torch.isfinite(coordinate_delta)
            & (candidate_code != selected_code)
        )
        coordinate_delta = torch.where(
            coordinate_valid,
            coordinate_delta,
            torch.full_like(coordinate_delta, torch.inf),
        )
        best_delta, best_index = coordinate_delta.min(dim=-1)
        best_index_expanded = best_index.unsqueeze(-1)
        current_at_best = selected_code.gather(-1, best_index_expanded)
        candidate_at_best = candidate_code.gather(-1, best_index_expanded)
        accept = torch.isfinite(best_delta) & (best_delta < 0.0)
        accepted_code = torch.where(
            accept.unsqueeze(-1), candidate_at_best, current_at_best
        )
        accepted_delta = (
            (accepted_code - current_at_best)
            * scale.gather(-1, best_index_expanded)
        ).squeeze(-1)
        cumulative_delta = cumulative_delta + torch.where(
            accept, best_delta, torch.zeros_like(best_delta)
        )
        selected_code.scatter_(-1, best_index_expanded, accepted_code)
        selected_quantized.scatter_(
            -1,
            best_index_expanded,
            selected_quantized.gather(-1, best_index_expanded)
            + accepted_delta.unsqueeze(-1),
        )
        selected_column = hessian_columns.index_select(
            0, (block_offsets + best_index).reshape(-1)
        ).reshape(rows, total_blocks, _BLOCK)
        projected_residual = (
            projected_residual
            - selected_column * accepted_delta.unsqueeze(-1)
        )

    use_candidate = (
        (selected_code != parent_code).any(dim=-1)
        & torch.isfinite(cumulative_delta)
        & (cumulative_delta < 0.0)
    )
    mask = use_candidate.unsqueeze(-1)
    candidate_mant = selected_code.abs() * 0.25
    candidate_sign = torch.where(
        candidate_mant == 0.0,
        torch.zeros_like(selected_code),
        torch.sign(selected_code),
    )
    selected = dict(parent)
    selected["sign"] = torch.where(
        mask,
        candidate_sign.to(parent["sign"].dtype),
        parent["sign"].reshape(rows, total_blocks, _BLOCK),
    ).reshape_as(parent["sign"])
    selected["mant"] = torch.where(
        mask,
        candidate_mant.to(parent["mant"].dtype),
        parent["mant"].reshape(rows, total_blocks, _BLOCK),
    ).reshape_as(parent["mant"])
    return selected


def _attention_qk_block_scale_refine(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Choose the Hessian-optimal legal outer E6M2 scale per row/H64."""
    if _QK_OUTER_SCALE_ABLATION != "full":
        return parent
    channels = int(value.shape[-1])
    if (
        num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or channels != num_heads * head_dim
        or state.get("hessian_head_dim") != head_dim
    ):
        return parent
    block_count = head_dim // _BLOCK
    total_blocks = num_heads * block_count
    expected_shape = (num_heads, block_count, _BLOCK, _BLOCK)
    hessian = state.get("cross_hessian")
    if type(hessian) is not torch.Tensor or tuple(hessian.shape) != expected_shape:
        return parent

    rows = int(value.numel() // channels)
    try:
        if parent["scale_factor"].numel() != rows * total_blocks:
            return parent
        hessian = hessian.to(device=value.device, dtype=torch.float32).reshape(
            total_blocks, _BLOCK, _BLOCK
        )
        diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)
        if not (
            bool(torch.isfinite(hessian).all().item())
            and bool((diagonal >= 0.0).all().item())
        ):
            return parent
        target = value.reshape(rows, total_blocks, _BLOCK)
        pattern = (
            parent["sign"].to(torch.float32)
            * parent["mant"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
        ).reshape(rows, total_blocks, _BLOCK)
        parent_outer = parent["scale_factor"].to(torch.float32).reshape(
            rows, total_blocks
        )
        h_target = torch.einsum("gij,rgj->rgi", hessian, target)
        h_pattern = torch.einsum("gij,rgj->rgi", hessian, pattern)
        numerator = (pattern * h_target).sum(dim=-1)
        denominator = (pattern * h_pattern).sum(dim=-1)
        optimal = numerator / denominator.clamp_min(1.0e-24)
        candidates = torch.stack(
            (parent_outer, _floor_e6m2(optimal), _ceil_e6m2(optimal)),
            dim=-1,
        )
        # The interface returns BF16 tensors. Score the exact representable
        # values that leave the function, not their FP32 precursors.
        candidates = candidates.to(torch.bfloat16).to(torch.float32)
        parent_residual = target - parent_outer.unsqueeze(-1) * pattern
        projected_parent_residual = torch.einsum(
            "gij,rgj->rgi", hessian, parent_residual
        )
        linear_term = (pattern * projected_parent_residual).sum(dim=-1)
        delta_scale = candidates - parent_outer.unsqueeze(-1)
        risk_delta = (
            -2.0 * delta_scale * linear_term.unsqueeze(-1)
            + delta_scale.square() * denominator.unsqueeze(-1)
        )
        valid = (
            torch.isfinite(candidates)
            & (candidates > 0.0)
            & torch.isfinite(risk_delta)
            & torch.isfinite(denominator).unsqueeze(-1)
            & (denominator > 1.0e-24).unsqueeze(-1)
        )
        risk_delta = torch.where(
            valid, risk_delta, torch.full_like(risk_delta, torch.inf)
        )
        best_risk, best_index = risk_delta.min(dim=-1)
        best_outer = candidates.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
        use_candidate = (
            torch.isfinite(best_risk)
            & (best_risk < 0.0)
            & (best_outer != parent_outer)
        )
        selected_outer = torch.where(use_candidate, best_outer, parent_outer)
    except (KeyError, RuntimeError, ValueError, OverflowError):
        return parent

    selected = dict(parent)
    selected["scale_factor"] = selected_outer.reshape_as(
        parent["scale_factor"]
    ).to(parent["scale_factor"].dtype)
    return selected


def _attention_v_hierarchy_min_sum(
    errors: torch.Tensor,
    h1: torch.Tensor,
) -> torch.Tensor:
    """Return stable parent-first states for two legal H64 hierarchy branches."""
    sequence, chains, states, block = errors.shape
    if (
        sequence <= 0
        or states != 2
        or block != _BLOCK
        or tuple(h1.shape) != (chains,)
    ):
        raise ValueError("invalid Attention-V binary hierarchy-chain shape")
    unary = errors.square().sum(dim=-1)
    message = unary[0]
    backpointers: list[torch.Tensor] = []
    coupling = 2.0 * h1.reshape(chains, 1, 1)
    for token in range(1, sequence):
        cross = (
            errors[token - 1].unsqueeze(-2)
            * errors[token].unsqueeze(-3)
        ).sum(dim=-1)
        transition = message.unsqueeze(-1) + coupling * cross
        best, predecessor = transition.min(dim=1)
        message = unary[token] + best
        backpointers.append(predecessor.to(torch.int8))
    current = message.argmin(dim=-1)
    choices = torch.empty(
        sequence, chains, dtype=torch.int64, device=errors.device
    )
    choices[-1] = current
    chain_index = torch.arange(chains, device=errors.device)
    for token in range(sequence - 2, -1, -1):
        current = backpointers[token][chain_index, current].to(torch.int64)
        choices[token] = current
    return choices


def _attention_v_stationary_hierarchy_chain_select(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    alternate: dict[str, torch.Tensor],
    state: dict[str, Any],
    kv_num_heads: int | None,
    head_dim: int | None,
) -> dict[str, torch.Tensor]:
    """V279 selector under the stationary consumer kernel."""
    if (
        _V_HIERARCHY_ABLATION in ("off", "mechanism-off")
        or state.get("pair_schema") != _V_PAIR_SCHEMA
        or kv_num_heads is None
        or head_dim is None
        or kv_num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or int(value.shape[-1]) != kv_num_heads * head_dim
        or value.ndim < 2
    ):
        return parent
    rho_value = state.get("pair_rho")
    valid_value = state.get("pair_valid")
    if (
        type(rho_value) is not torch.Tensor
        or type(valid_value) is not torch.Tensor
        or rho_value.numel() != kv_num_heads
        or valid_value.numel() != kv_num_heads
    ):
        return parent
    rho_head = rho_value.to(device=value.device, dtype=torch.float32).flatten()
    valid_head = valid_value.to(device=value.device, dtype=torch.bool).flatten()
    if (
        not bool(torch.isfinite(rho_head).all().item())
        or not bool(((rho_head >= 0.0) & (rho_head <= _V_PAIR_RHO_LIMIT)).all().item())
    ):
        return parent
    if _V_HIERARCHY_ABLATION == "unary-only":
        rho_head = torch.zeros_like(rho_head)
    elif _V_HIERARCHY_ABLATION != "full":
        return parent
    if not bool(valid_head.any().item()):
        return parent

    channels = int(value.shape[-1])
    sequence = int(value.shape[-2])
    if sequence < 2:
        return parent
    batch = int(value.numel() // (sequence * channels))
    blocks_per_head = head_dim // _BLOCK

    try:
        target = value.reshape(
            batch, sequence, kv_num_heads, blocks_per_head, _BLOCK
        )
        parent_reconstruction = _dequantize_hif4(parent).reshape_as(target)
        alternate_reconstruction = _dequantize_hif4(alternate).reshape_as(target)
    except (KeyError, RuntimeError, ValueError):
        return parent
    if not (
        bool(torch.isfinite(target).all().item())
        and bool(torch.isfinite(parent_reconstruction).all().item())
        and bool(torch.isfinite(alternate_reconstruction).all().item())
    ):
        return parent

    parent_error = parent_reconstruction - target
    alternate_error = alternate_reconstruction - target
    errors = torch.stack((parent_error, alternate_error), dim=-2)
    errors = errors.permute(1, 0, 2, 3, 4, 5).reshape(
        sequence, -1, 2, _BLOCK
    )
    chain_h1 = rho_head.reshape(1, kv_num_heads, 1).expand(
        batch, kv_num_heads, blocks_per_head
    ).reshape(-1)
    chain_valid = valid_head.reshape(1, kv_num_heads, 1).expand(
        batch, kv_num_heads, blocks_per_head
    ).reshape(-1)
    try:
        choices = _attention_v_hierarchy_min_sum(errors, chain_h1)
        selected_error = errors.gather(
            2,
            choices.unsqueeze(-1).unsqueeze(-1).expand(
                sequence, errors.shape[1], 1, _BLOCK
            ),
        ).squeeze(2)
        flat_parent_error = errors[:, :, 0]
        selected_risk = selected_error.square().sum(dim=(0, 2)) + 2.0 * chain_h1 * (
            selected_error[:-1] * selected_error[1:]
        ).sum(dim=(0, 2))
        parent_risk = flat_parent_error.square().sum(dim=(0, 2)) + 2.0 * chain_h1 * (
            flat_parent_error[:-1] * flat_parent_error[1:]
        ).sum(dim=(0, 2))
        changed = choices.to(torch.bool).any(dim=0)
        accept = (
            chain_valid
            & changed
            & torch.isfinite(selected_risk)
            & torch.isfinite(parent_risk)
            & (selected_risk < parent_risk)
        )
    except (RuntimeError, ValueError, OverflowError):
        return parent
    if not bool(accept.any().item()):
        return parent

    use_alternate = choices.reshape(
        sequence, batch, kv_num_heads, blocks_per_head
    ).permute(1, 0, 2, 3).to(torch.bool)
    use_alternate &= accept.reshape(
        batch, 1, kv_num_heads, blocks_per_head
    )
    selected = {}
    try:
        for key in parent:
            parent_field = parent[key]
            alternate_field = alternate[key]
            if parent_field.shape != alternate_field.shape:
                return parent
            hierarchy_shape = parent_field.shape[value.ndim - 1 :]
            if (
                not hierarchy_shape
                or int(hierarchy_shape[0]) != kv_num_heads * blocks_per_head
            ):
                return parent
            parent_view = parent_field.reshape(batch, sequence, *hierarchy_shape)
            alternate_view = alternate_field.reshape_as(parent_view)
            field_mask = use_alternate.reshape(
                batch,
                sequence,
                kv_num_heads * blocks_per_head,
                *([1] * (len(hierarchy_shape) - 1)),
            )
            selected[key] = torch.where(
                field_mask, alternate_view, parent_view
            ).reshape_as(parent_field)
    except (KeyError, RuntimeError, ValueError):
        return parent
    return selected


def _attention_v_profile_min_sum(
    errors: torch.Tensor,
    unary: torch.Tensor,
    edge: torch.Tensor,
) -> torch.Tensor:
    """Return parent-first binary states for a variable tridiagonal risk."""
    sequence, chains, states, block = errors.shape
    if (
        sequence <= 0
        or states != 2
        or block != _BLOCK
        or tuple(unary.shape) != (sequence, chains)
        or tuple(edge.shape) != (sequence - 1, chains)
    ):
        raise ValueError("invalid Attention-V variable hierarchy-chain shape")
    message = unary[0].unsqueeze(-1) * errors[0].square().sum(dim=-1)
    backpointers: list[torch.Tensor] = []
    for token in range(1, sequence):
        cross = (
            errors[token - 1].unsqueeze(-2)
            * errors[token].unsqueeze(-3)
        ).sum(dim=-1)
        transition = message.unsqueeze(-1) + 2.0 * edge[token - 1].reshape(
            chains, 1, 1
        ) * cross
        best, predecessor = transition.min(dim=1)
        message = (
            unary[token].unsqueeze(-1) * errors[token].square().sum(dim=-1)
            + best
        )
        backpointers.append(predecessor.to(torch.int8))
    current = message.argmin(dim=-1)
    choices = torch.empty(
        sequence, chains, dtype=torch.int64, device=errors.device
    )
    choices[-1] = current
    chain_index = torch.arange(chains, device=errors.device)
    for token in range(sequence - 2, -1, -1):
        current = backpointers[token][chain_index, current].to(torch.int64)
        choices[token] = current
    return choices


def _attention_v_hierarchy_chain_select(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    alternate: dict[str, torch.Tensor],
    state: dict[str, Any],
    kv_num_heads: int | None,
    head_dim: int | None,
) -> dict[str, torch.Tensor]:
    """Use a length-matched softmax P^T P tridiagonal hierarchy risk."""
    stationary = _attention_v_stationary_hierarchy_chain_select(
        value, parent, alternate, state, kv_num_heads, head_dim
    )
    if (
        _V_PROFILE_ABLATION in ("off", "mechanism-off", "stationary-only")
        or _V_HIERARCHY_ABLATION != "full"
        or _V_PROFILE_ABLATION not in ("diagonal-only", "tridiagonal-full")
        or state.get("pair_schema") != _V_PAIR_SCHEMA
        or state.get("profile_schema") != _V_PROFILE_SCHEMA
        or kv_num_heads is None
        or head_dim is None
        or kv_num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or value.ndim < 2
        or int(value.shape[-1]) != kv_num_heads * head_dim
    ):
        return stationary
    sequence = int(value.shape[-2])
    if sequence < 2 or sequence > _V_PROFILE_MAX_SEQUENCE:
        return stationary

    lengths_value = state.get("profile_lengths")
    unary_value = state.get("profile_unary")
    edge_value = state.get("profile_edge")
    valid_value = state.get("profile_valid")
    pair_valid_value = state.get("pair_valid")
    if not (
        type(lengths_value) is torch.Tensor
        and type(unary_value) is torch.Tensor
        and type(edge_value) is torch.Tensor
        and type(valid_value) is torch.Tensor
        and type(pair_valid_value) is torch.Tensor
    ):
        return stationary
    profile_count = int(lengths_value.numel())
    if (
        lengths_value.ndim != 1
        or tuple(unary_value.shape)
        != (profile_count, kv_num_heads, _V_PROFILE_MAX_SEQUENCE)
        or tuple(edge_value.shape)
        != (profile_count, kv_num_heads, _V_PROFILE_MAX_SEQUENCE - 1)
        or tuple(valid_value.shape) != (profile_count, kv_num_heads)
        or pair_valid_value.numel() != kv_num_heads
    ):
        return stationary
    lengths = lengths_value.to(device=value.device, dtype=torch.int64).flatten()
    if profile_count == 0:
        return stationary
    if not bool(
        ((lengths >= 2) & (lengths <= _V_PROFILE_MAX_SEQUENCE)).all().item()
    ):
        return stationary
    if profile_count > 1 and not bool((lengths[1:] > lengths[:-1]).all().item()):
        return stationary
    matches = torch.nonzero(lengths == sequence, as_tuple=False).flatten()
    if matches.numel() != 1:
        return stationary
    profile_index = int(matches[0].item())

    unary_head = unary_value[profile_index].to(
        device=value.device, dtype=torch.float32
    )
    edge_head = edge_value[profile_index].to(
        device=value.device, dtype=torch.float32
    )
    valid_head = valid_value[profile_index].to(
        device=value.device, dtype=torch.bool
    ).flatten()
    valid_head &= pair_valid_value.to(
        device=value.device, dtype=torch.bool
    ).flatten()
    active_unary = unary_head[:, :sequence]
    active_edge = edge_head[:, : sequence - 1]
    geometric = torch.sqrt(
        active_unary[:, :-1].clamp_min(1.0e-24)
        * active_unary[:, 1:].clamp_min(1.0e-24)
    )
    valid_head &= (
        torch.isfinite(active_unary).all(dim=-1)
        & torch.isfinite(active_edge).all(dim=-1)
        & (active_unary > 0.0).all(dim=-1)
        & (active_edge >= 0.0).all(dim=-1)
        & (
            active_edge
            <= (_V_PAIR_RHO_LIMIT + 1.0e-6) * geometric
        ).all(dim=-1)
        & ((active_unary.mean(dim=-1) - 1.0).abs() <= 1.0e-4)
    )
    if sequence < _V_PROFILE_MAX_SEQUENCE:
        valid_head &= (unary_head[:, sequence:] == 0.0).all(dim=-1)
    if sequence - 1 < _V_PROFILE_MAX_SEQUENCE - 1:
        valid_head &= (edge_head[:, sequence - 1 :] == 0.0).all(dim=-1)
    if not bool(valid_head.any().item()):
        return stationary
    if _V_PROFILE_ABLATION == "diagonal-only":
        active_edge = torch.zeros_like(active_edge)

    channels = int(value.shape[-1])
    batch = int(value.numel() // (sequence * channels))
    blocks_per_head = head_dim // _BLOCK
    try:
        target = value.reshape(
            batch, sequence, kv_num_heads, blocks_per_head, _BLOCK
        )
        parent_reconstruction = _dequantize_hif4(parent).reshape_as(target)
        alternate_reconstruction = _dequantize_hif4(alternate).reshape_as(target)
    except (KeyError, RuntimeError, ValueError):
        return stationary
    if not (
        bool(torch.isfinite(target).all().item())
        and bool(torch.isfinite(parent_reconstruction).all().item())
        and bool(torch.isfinite(alternate_reconstruction).all().item())
    ):
        return stationary

    errors = torch.stack(
        (parent_reconstruction - target, alternate_reconstruction - target),
        dim=-2,
    ).permute(1, 0, 2, 3, 4, 5).reshape(sequence, -1, 2, _BLOCK)
    chain_unary = active_unary.transpose(0, 1).reshape(
        sequence, 1, kv_num_heads, 1
    ).expand(sequence, batch, kv_num_heads, blocks_per_head).reshape(sequence, -1)
    chain_edge = active_edge.transpose(0, 1).reshape(
        sequence - 1, 1, kv_num_heads, 1
    ).expand(sequence - 1, batch, kv_num_heads, blocks_per_head).reshape(
        sequence - 1, -1
    )
    chain_valid = valid_head.reshape(1, kv_num_heads, 1).expand(
        batch, kv_num_heads, blocks_per_head
    ).reshape(-1)
    try:
        choices = _attention_v_profile_min_sum(errors, chain_unary, chain_edge)
        selected_error = errors.gather(
            2,
            choices.unsqueeze(-1).unsqueeze(-1).expand(
                sequence, errors.shape[1], 1, _BLOCK
            ),
        ).squeeze(2)
        parent_error = errors[:, :, 0]
        selected_risk = (
            chain_unary * selected_error.square().sum(dim=-1)
        ).sum(dim=0) + 2.0 * (
            chain_edge * (selected_error[:-1] * selected_error[1:]).sum(dim=-1)
        ).sum(dim=0)
        parent_risk = (
            chain_unary * parent_error.square().sum(dim=-1)
        ).sum(dim=0) + 2.0 * (
            chain_edge * (parent_error[:-1] * parent_error[1:]).sum(dim=-1)
        ).sum(dim=0)
        accept = (
            chain_valid
            & choices.to(torch.bool).any(dim=0)
            & torch.isfinite(selected_risk)
            & torch.isfinite(parent_risk)
            & (selected_risk < parent_risk)
        )
    except (RuntimeError, ValueError, OverflowError):
        return stationary

    use_alternate = choices.reshape(
        sequence, batch, kv_num_heads, blocks_per_head
    ).permute(1, 0, 2, 3).to(torch.bool)
    use_alternate &= accept.reshape(batch, 1, kv_num_heads, blocks_per_head)
    profile_scope = valid_head.reshape(1, 1, kv_num_heads, 1).expand(
        batch, sequence, kv_num_heads, blocks_per_head
    )
    selected: dict[str, torch.Tensor] = {}
    try:
        for key in parent:
            parent_field = parent[key]
            alternate_field = alternate[key]
            stationary_field = stationary[key]
            if not (
                parent_field.shape == alternate_field.shape == stationary_field.shape
            ):
                return stationary
            hierarchy_shape = parent_field.shape[value.ndim - 1 :]
            if (
                not hierarchy_shape
                or int(hierarchy_shape[0]) != kv_num_heads * blocks_per_head
            ):
                return stationary
            parent_view = parent_field.reshape(batch, sequence, *hierarchy_shape)
            alternate_view = alternate_field.reshape_as(parent_view)
            stationary_view = stationary_field.reshape_as(parent_view)
            field_tail = *([1] * (len(hierarchy_shape) - 1)),
            alternate_mask = use_alternate.reshape(
                batch,
                sequence,
                kv_num_heads * blocks_per_head,
                *field_tail,
            )
            scope_mask = profile_scope.reshape(
                batch,
                sequence,
                kv_num_heads * blocks_per_head,
                *field_tail,
            )
            profile_view = torch.where(
                alternate_mask, alternate_view, parent_view
            )
            selected[key] = torch.where(
                scope_mask, profile_view, stationary_view
            ).reshape_as(parent_field)
    except (KeyError, RuntimeError, ValueError):
        return stationary
    return selected


def _attention_v_exact_min_sum(
    errors: torch.Tensor,
    h0: torch.Tensor,
    h1: torch.Tensor,
) -> torch.Tensor:
    """Return stable parent-first states for independent ternary token chains."""
    sequence, chains, states = errors.shape
    if (
        sequence <= 0
        or states != 3
        or tuple(h0.shape) != (chains,)
        or tuple(h1.shape) != (chains,)
    ):
        raise ValueError("invalid Attention-V ternary-chain shape")
    unary = h0.reshape(1, chains, 1) * errors.square()
    message = unary[0]
    backpointers: list[torch.Tensor] = []
    coupling = 2.0 * h1.reshape(chains, 1, 1)
    for token in range(1, sequence):
        transition = (
            message.unsqueeze(-1)
            + coupling
            * errors[token - 1].unsqueeze(-1)
            * errors[token].unsqueeze(-2)
        )
        best, predecessor = transition.min(dim=1)
        message = unary[token] + best
        backpointers.append(predecessor.to(torch.int8))
    current = message.argmin(dim=-1)
    choices = torch.empty(
        sequence, chains, dtype=torch.int64, device=errors.device
    )
    choices[-1] = current
    chain_index = torch.arange(chains, device=errors.device)
    for token in range(sequence - 2, -1, -1):
        current = backpointers[token][chain_index, current].to(torch.int64)
        choices[token] = current
    return choices


def _attention_v_order2_exact_min_sum(
    errors: torch.Tensor,
    h0: torch.Tensor,
    h1: torch.Tensor,
    h2: torch.Tensor,
) -> torch.Tensor:
    """Solve a ternary band-2 token chain with stable parent-first ties."""
    sequence, chains, states = errors.shape
    if (
        sequence < 3
        or states != 3
        or tuple(h0.shape) != (chains,)
        or tuple(h1.shape) != (chains,)
        or tuple(h2.shape) != (chains,)
    ):
        raise ValueError("invalid Attention-V order-2 chain shape")
    unary = h0.reshape(1, chains, 1) * errors.square()
    coupling1 = 2.0 * h1.reshape(chains, 1, 1)
    message = (
        unary[0].unsqueeze(-1)
        + unary[1].unsqueeze(-2)
        + coupling1 * errors[0].unsqueeze(-1) * errors[1].unsqueeze(-2)
    )
    backpointers: list[torch.Tensor] = []
    coupling1_order2 = 2.0 * h1.reshape(chains, 1, 1, 1)
    coupling2_order2 = 2.0 * h2.reshape(chains, 1, 1, 1)
    for token in range(2, sequence):
        transition = (
            message.unsqueeze(-1)
            + coupling1_order2
            * errors[token - 1].reshape(chains, 1, states, 1)
            * errors[token].reshape(chains, 1, 1, states)
            + coupling2_order2
            * errors[token - 2].reshape(chains, states, 1, 1)
            * errors[token].reshape(chains, 1, 1, states)
        )
        best, predecessor = transition.min(dim=1)
        message = best + unary[token].unsqueeze(1)
        backpointers.append(predecessor.to(torch.int8))

    flat = message.reshape(chains, states * states).argmin(dim=-1)
    previous = torch.div(flat, states, rounding_mode="floor")
    current = torch.remainder(flat, states)
    choices = torch.empty(
        sequence, chains, dtype=torch.int64, device=errors.device
    )
    choices[-2] = previous
    choices[-1] = current
    chain_index = torch.arange(chains, device=errors.device)
    for token in range(sequence - 1, 1, -1):
        predecessor = backpointers[token - 2][
            chain_index, previous, current
        ].to(torch.int64)
        choices[token - 2] = predecessor
        current = previous
        previous = predecessor
    return choices


def _attention_v_disjoint_min_sum(
    errors: torch.Tensor,
    h0: torch.Tensor,
    h1: torch.Tensor,
) -> torch.Tensor:
    """Ablation: solve non-overlapping pairs under the same ternary objective."""
    sequence, chains, states = errors.shape
    if (
        sequence <= 0
        or states != 3
        or tuple(h0.shape) != (chains,)
        or tuple(h1.shape) != (chains,)
    ):
        raise ValueError("invalid Attention-V disjoint-chain shape")
    choices = torch.zeros(
        sequence, chains, dtype=torch.int64, device=errors.device
    )
    h0_view = h0.reshape(chains, 1, 1)
    h1_view = h1.reshape(chains, 1, 1)
    for token in range(0, sequence - 1, 2):
        left = errors[token]
        right = errors[token + 1]
        objective = h0_view * (
            left.unsqueeze(-1).square() + right.unsqueeze(-2).square()
        ) + 2.0 * h1_view * left.unsqueeze(-1) * right.unsqueeze(-2)
        flat = objective.reshape(chains, 9).argmin(dim=-1)
        choices[token] = torch.div(flat, 3, rounding_mode="floor")
        choices[token + 1] = torch.remainder(flat, 3)
    if sequence % 2:
        choices[-1] = (
            h0.reshape(chains, 1) * errors[-1].square()
        ).argmin(dim=-1)
    return choices


def _attention_v_chain_repair(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    kv_num_heads: int | None,
    head_dim: int | None,
) -> dict[str, torch.Tensor]:
    """Repair short V token chains under a stationary band-2 kernel."""
    if (
        _V_PAIR_ABLATION in ("off", "mechanism-off")
        or state.get("pair_schema") != _V_PAIR_SCHEMA
        or kv_num_heads is None
        or head_dim is None
        or kv_num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or int(value.shape[-1]) != kv_num_heads * head_dim
        or value.ndim < 2
    ):
        return parent
    rho_value = state.get("pair_rho")
    valid_value = state.get("pair_valid")
    if (
        type(rho_value) is not torch.Tensor
        or type(valid_value) is not torch.Tensor
        or rho_value.numel() != kv_num_heads
        or valid_value.numel() != kv_num_heads
    ):
        return parent
    rho_head = rho_value.to(device=value.device, dtype=torch.float32).flatten()
    valid_head = valid_value.to(device=value.device, dtype=torch.bool).flatten()
    if (
        not bool(torch.isfinite(rho_head).all().item())
        or not bool(((rho_head >= 0.0) & (rho_head <= _V_PAIR_RHO_LIMIT)).all().item())
    ):
        return parent
    if _V_PAIR_ABLATION == "cross-off":
        rho_head = torch.zeros_like(rho_head)
    elif _V_PAIR_ABLATION not in ("full", "disjoint-pair"):
        return parent
    if not bool(valid_head.any().item()):
        return parent

    rho2_head = torch.zeros_like(rho_head)
    lag2_valid_head = torch.zeros_like(valid_head)
    lag2_state_valid = False
    if _V_PAIR_ABLATION == "full" and _V_LAG2_ABLATION == "short-full":
        rho2_value = state.get("pair_rho2")
        lag2_valid_value = state.get("pair_lag2_valid")
        if (
            state.get("pair_lag2_schema") == _V_LAG2_SCHEMA
            and type(rho2_value) is torch.Tensor
            and type(lag2_valid_value) is torch.Tensor
            and rho2_value.numel() == kv_num_heads
            and lag2_valid_value.numel() == kv_num_heads
        ):
            candidate_rho2 = rho2_value.to(
                device=value.device, dtype=torch.float32
            ).flatten()
            candidate_valid = lag2_valid_value.to(
                device=value.device, dtype=torch.bool
            ).flatten()
            headroom = (_V_PAIR_RHO_LIMIT - rho_head).clamp_min(0.0)
            lag2_state_valid = bool(
                torch.isfinite(candidate_rho2).all().item()
                and (
                    (candidate_rho2 >= 0.0)
                    & (candidate_rho2 <= headroom)
                    & ((~candidate_valid) | valid_head)
                    & (candidate_valid | (candidate_rho2 == 0.0))
                ).all().item()
            )
            if lag2_state_valid:
                rho2_head = candidate_rho2
                lag2_valid_head = candidate_valid & (candidate_rho2 > 0.0)

    channels = int(value.shape[-1])
    sequence = int(value.shape[-2])
    if sequence < 2:
        return parent
    batch = int(value.numel() // (sequence * channels))

    try:
        target = value.reshape(batch, sequence, kv_num_heads, head_dim)
        parent_sign = parent["sign"].reshape_as(target)
        parent_mant = parent["mant"].reshape_as(target)
        effective = (
            parent["scale_factor"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
        ).expand_as(parent["mant"]).reshape_as(target)
    except (KeyError, RuntimeError, ValueError):
        return parent
    if not (
        bool(torch.isfinite(target).all().item())
        and bool(torch.isfinite(effective).all().item())
        and bool((effective > 0.0).all().item())
    ):
        return parent

    raw_parent_level = (
        parent_sign.to(torch.float32) * parent_mant.to(torch.float32) * 4.0
    )
    parent_level = torch.round(raw_parent_level)
    legal_parent = (
        torch.isfinite(raw_parent_level)
        & ((raw_parent_level - parent_level).abs() <= 1.0e-6)
        & (parent_level >= -7.0)
        & (parent_level <= 7.0)
    )
    if not bool(legal_parent.all().item()):
        return parent

    predecessor = (parent_level - 1.0).clamp(-7.0, 7.0)
    successor = (parent_level + 1.0).clamp(-7.0, 7.0)
    candidate_level = torch.stack(
        (parent_level, predecessor, successor), dim=-1
    )
    candidate_value = candidate_level * effective.unsqueeze(-1) * 0.25
    errors = (
        candidate_value - target.unsqueeze(-1)
    ).permute(1, 0, 2, 3, 4).reshape(sequence, -1, 3)
    chain_h0 = torch.ones(
        batch, kv_num_heads, head_dim, device=value.device, dtype=torch.float32
    ).reshape(-1)
    chain_h1 = rho_head.reshape(1, kv_num_heads, 1).expand(
        batch, kv_num_heads, head_dim
    ).reshape(-1)
    chain_valid = valid_head.reshape(1, kv_num_heads, 1).expand(
        batch, kv_num_heads, head_dim
    ).reshape(-1)
    chain_h2 = rho2_head.reshape(1, kv_num_heads, 1).expand(
        batch, kv_num_heads, head_dim
    ).reshape(-1)
    chain_lag2_valid = lag2_valid_head.reshape(1, kv_num_heads, 1).expand(
        batch, kv_num_heads, head_dim
    ).reshape(-1)
    try:
        if _V_PAIR_ABLATION == "disjoint-pair":
            choices = _attention_v_disjoint_min_sum(errors, chain_h0, chain_h1)
        else:
            lag2_active = (
                lag2_state_valid
                and sequence >= 3
                and sequence <= _V_LAG2_SEQUENCE_LIMIT
                and bool((chain_valid & chain_lag2_valid).any().item())
            )
            if lag2_active:
                active_mask = chain_valid & chain_lag2_valid
                active_count = int(active_mask.sum().item())
                scratch_bytes = active_count * (
                    sequence * (3 * 4 + 3 * 4 + 9 + 8) + 27 * 4
                )
                lag2_active = scratch_bytes <= _V_LAG2_SCRATCH_LIMIT
            if not lag2_active:
                choices = _attention_v_exact_min_sum(
                    errors, chain_h0, chain_h1
                )
                chain_h2 = torch.zeros_like(chain_h2)
            else:
                active_mask = chain_valid & chain_lag2_valid
                first_order_mask = chain_valid & ~active_mask
                if bool(active_mask.all().item()):
                    choices = _attention_v_order2_exact_min_sum(
                        errors, chain_h0, chain_h1, chain_h2
                    )
                else:
                    choices = torch.zeros(
                        sequence,
                        errors.shape[1],
                        dtype=torch.int64,
                        device=errors.device,
                    )
                    if bool(first_order_mask.any().item()):
                        choices[:, first_order_mask] = _attention_v_exact_min_sum(
                            errors[:, first_order_mask],
                            chain_h0[first_order_mask],
                            chain_h1[first_order_mask],
                        )
                    choices[:, active_mask] = _attention_v_order2_exact_min_sum(
                        errors[:, active_mask],
                        chain_h0[active_mask],
                        chain_h1[active_mask],
                        chain_h2[active_mask],
                    )
                chain_h2 = torch.where(
                    active_mask, chain_h2, torch.zeros_like(chain_h2)
                )
        flat_levels = candidate_level.permute(1, 0, 2, 3, 4).reshape(
            sequence, -1, 3
        )
        selected_level = flat_levels.gather(
            -1, choices.unsqueeze(-1)
        ).squeeze(-1)
        flat_effective = effective.permute(1, 0, 2, 3).reshape(sequence, -1)
        flat_target = target.permute(1, 0, 2, 3).reshape(sequence, -1)
        selected_error = selected_level * flat_effective * 0.25 - flat_target
        flat_parent_level = parent_level.permute(1, 0, 2, 3).reshape(
            sequence, -1
        )
        parent_error = flat_parent_level * flat_effective * 0.25 - flat_target
        selected_risk = selected_error.square().sum(dim=0) + 2.0 * chain_h1 * (
            selected_error[:-1] * selected_error[1:]
        ).sum(dim=0)
        parent_risk = parent_error.square().sum(dim=0) + 2.0 * chain_h1 * (
            parent_error[:-1] * parent_error[1:]
        ).sum(dim=0)
        if sequence >= 3:
            selected_risk = selected_risk + 2.0 * chain_h2 * (
                selected_error[:-2] * selected_error[2:]
            ).sum(dim=0)
            parent_risk = parent_risk + 2.0 * chain_h2 * (
                parent_error[:-2] * parent_error[2:]
            ).sum(dim=0)
        changed = (selected_level != flat_parent_level).any(dim=0)
        accept = (
            chain_valid
            & changed
            & torch.isfinite(selected_risk)
            & torch.isfinite(parent_risk)
            & (selected_risk < parent_risk)
        )
    except (RuntimeError, ValueError, OverflowError):
        return parent

    selected_level = selected_level.reshape(
        sequence, batch, kv_num_heads, head_dim
    ).permute(1, 0, 2, 3)
    candidate_mant = selected_level.abs() * 0.25
    candidate_sign = torch.where(
        selected_level == 0.0,
        torch.zeros_like(selected_level),
        torch.sign(selected_level),
    )
    mask = accept.reshape(batch, kv_num_heads, head_dim).unsqueeze(1)
    selected = dict(parent)
    selected["mant"] = torch.where(
        mask, candidate_mant.to(parent_mant.dtype), parent_mant
    ).reshape_as(parent["mant"])
    selected["sign"] = torch.where(
        mask, candidate_sign.to(parent_sign.dtype), parent_sign
    ).reshape_as(parent["sign"])
    return selected


def _attention_v_global_dc_repair(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    kv_num_heads: int | None,
    head_dim: int | None,
) -> dict[str, torch.Tensor]:
    """Run one live sweep under the first-lag plus global-DC V risk."""
    if (
        _V_GLOBAL_DC_ABLATION in ("off", "mechanism-off")
        or state.get("pair_schema") != _V_PAIR_SCHEMA
        or kv_num_heads is None
        or head_dim is None
        or kv_num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or int(value.shape[-1]) != kv_num_heads * head_dim
        or value.ndim < 2
    ):
        return parent
    rho_value = state.get("pair_rho")
    valid_value = state.get("pair_valid")
    if (
        type(rho_value) is not torch.Tensor
        or type(valid_value) is not torch.Tensor
        or rho_value.numel() != kv_num_heads
        or valid_value.numel() != kv_num_heads
    ):
        return parent
    rho_head = rho_value.to(device=value.device, dtype=torch.float32).flatten()
    valid_head = valid_value.to(device=value.device, dtype=torch.bool).flatten()
    if (
        not bool(torch.isfinite(rho_head).all().item())
        or not bool(((rho_head >= 0.0) & (rho_head <= _V_PAIR_RHO_LIMIT)).all().item())
        or _V_GLOBAL_DC_ABLATION not in ("full", "dc-off", "stale-global")
        or not bool(valid_head.any().item())
    ):
        return parent

    channels = int(value.shape[-1])
    sequence = int(value.shape[-2])
    if sequence < 2:
        return parent
    batch = int(value.numel() // (sequence * channels))
    try:
        target = value.reshape(batch, sequence, kv_num_heads, head_dim)
        parent_sign = parent["sign"].reshape_as(target)
        parent_mant = parent["mant"].reshape_as(target)
        effective = (
            parent["scale_factor"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
        ).expand_as(parent["mant"]).reshape_as(target)
    except (KeyError, RuntimeError, ValueError):
        return parent
    if not (
        bool(torch.isfinite(target).all().item())
        and bool(torch.isfinite(effective).all().item())
        and bool((effective > 0.0).all().item())
    ):
        return parent

    raw_parent_level = (
        parent_sign.to(torch.float32) * parent_mant.to(torch.float32) * 4.0
    )
    parent_level = torch.round(raw_parent_level)
    legal_parent = (
        torch.isfinite(raw_parent_level)
        & ((raw_parent_level - parent_level).abs() <= 1.0e-6)
        & (parent_level >= -7.0)
        & (parent_level <= 7.0)
    )
    if not bool(legal_parent.all().item()):
        return parent

    current_level = parent_level.clone()
    parent_error = current_level * effective * 0.25 - target
    current_error = parent_error.clone()
    live_sum = current_error.sum(dim=1)
    rho = rho_head.reshape(1, kv_num_heads, 1)
    dc_weight = 0.0 if _V_GLOBAL_DC_ABLATION == "dc-off" else _V_GLOBAL_DC_LAMBDA
    one_plus_dc = 1.0 + dc_weight
    try:
        # The first-lag chain is bipartite.  Updating one token parity at a time
        # preserves live opposite-parity neighbours while replacing O(T) Python
        # launches with four bounded tensor passes.  The final full-risk gate
        # rejects any chain whose simultaneous rank-1 updates do not improve.
        for _ in range(2):
            for parity in (0, 1):
                token_slice = slice(parity, sequence, 2)
                level = current_level[:, token_slice]
                old_error = current_error[:, token_slice]
                step = effective[:, token_slice] * 0.25
                neighbor = torch.zeros_like(old_error)
                token_indices = torch.arange(
                    parity, sequence, 2, device=value.device
                )
                left_mask = token_indices > 0
                right_mask = token_indices + 1 < sequence
                if bool(left_mask.any().item()):
                    neighbor[:, left_mask] += current_error[
                        :, token_indices[left_mask] - 1
                    ]
                if bool(right_mask.any().item()):
                    neighbor[:, right_mask] += current_error[
                        :, token_indices[right_mask] + 1
                    ]
                zero = torch.zeros_like(step)
                lower_delta = torch.where(level > -7.0, -step, zero)
                upper_delta = torch.where(level < 7.0, step, zero)
                common = old_error + rho.unsqueeze(1) * neighbor + (
                    dc_weight * live_sum.unsqueeze(1)
                )
                lower_score = lower_delta * (
                    2.0 * common + one_plus_dc * lower_delta
                )
                upper_score = upper_delta * (
                    2.0 * common + one_plus_dc * upper_delta
                )
                choose_lower = lower_score < 0.0
                best_score = torch.where(choose_lower, lower_score, zero)
                choose_upper = upper_score < best_score
                proposed_delta = torch.where(
                    choose_upper,
                    upper_delta,
                    torch.where(choose_lower, lower_delta, zero),
                )
                accepted_delta = proposed_delta
                current_level[:, token_slice] = level + accepted_delta / step
                current_error[:, token_slice] = old_error + accepted_delta
                live_sum = live_sum + accepted_delta.sum(dim=1)

        selected_risk = current_error.square().sum(dim=1) + 2.0 * rho * (
            current_error[:, :-1] * current_error[:, 1:]
        ).sum(dim=1)
        parent_risk = parent_error.square().sum(dim=1) + 2.0 * rho * (
            parent_error[:, :-1] * parent_error[:, 1:]
        ).sum(dim=1)
        selected_risk = selected_risk + _V_GLOBAL_DC_LAMBDA * (
            current_error.sum(dim=1).square()
        )
        parent_risk = parent_risk + _V_GLOBAL_DC_LAMBDA * (
            parent_error.sum(dim=1).square()
        )
        changed = (current_level != parent_level).any(dim=1)
        accept = (
            valid_head.reshape(1, kv_num_heads, 1)
            & changed
            & torch.isfinite(selected_risk)
            & torch.isfinite(parent_risk)
            & (selected_risk < parent_risk)
        )
    except (RuntimeError, ValueError, OverflowError):
        return parent
    if not bool(accept.any().item()):
        return parent

    candidate_mant = current_level.abs() * 0.25
    candidate_sign = torch.where(
        current_level == 0.0,
        torch.zeros_like(current_level),
        torch.sign(current_level),
    )
    mask = accept.unsqueeze(1)
    selected = dict(parent)
    selected["mant"] = torch.where(
        mask, candidate_mant.to(parent_mant.dtype), parent_mant
    ).reshape_as(parent["mant"])
    selected["sign"] = torch.where(
        mask, candidate_sign.to(parent_sign.dtype), parent_sign
    ).reshape_as(parent["sign"])
    return selected


def _attention_v_nystrom_repair(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    kv_num_heads: int | None,
    head_dim: int | None,
) -> dict[str, torch.Tensor] | None:
    """Descend a length-matched data-driven probability-Gram kernel."""
    if _V_NYSTROM_ABLATION != "full":
        return None
    if (
        state.get("nystrom_schema") != _V_NYSTROM_SCHEMA
        or value.ndim != 2
        or kv_num_heads is None
        or head_dim is None
        or kv_num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or int(value.shape[-1]) != kv_num_heads * head_dim
    ):
        return None
    sequence = int(value.shape[0])
    lengths_value = state.get("nystrom_lengths")
    factors_value = state.get("nystrom_factors")
    diagonals_value = state.get("nystrom_diagonals")
    valid_value = state.get("nystrom_valid")
    if (
        type(lengths_value) is not torch.Tensor
        or type(factors_value) is not tuple
        or type(diagonals_value) is not tuple
        or type(valid_value) is not tuple
    ):
        return None
    matches = torch.nonzero(
        lengths_value.to(torch.int64).flatten() == sequence
    ).flatten()
    if int(matches.numel()) != 1:
        return None
    profile_index = int(matches[0].item())
    if not (
        0 <= profile_index < len(factors_value)
        and profile_index < len(diagonals_value)
        and profile_index < len(valid_value)
    ):
        return None
    factor_value = factors_value[profile_index]
    diagonal_value = diagonals_value[profile_index]
    head_valid_value = valid_value[profile_index]
    if (
        type(factor_value) is not torch.Tensor
        or type(diagonal_value) is not torch.Tensor
        or type(head_valid_value) is not torch.Tensor
        or tuple(factor_value.shape)
        != (kv_num_heads, _V_LOWRANK_RANK, sequence)
        or tuple(diagonal_value.shape) != (kv_num_heads, sequence)
        or head_valid_value.numel() != kv_num_heads
    ):
        return None
    factor = factor_value.to(device=value.device, dtype=torch.float32)
    diagonal = diagonal_value.to(device=value.device, dtype=torch.float32)
    head_valid = head_valid_value.to(
        device=value.device, dtype=torch.bool
    ).flatten()
    if not (
        bool(torch.isfinite(factor).all().item())
        and bool(torch.isfinite(diagonal).all().item())
        and bool((diagonal > 0.0).all().item())
        and bool(head_valid.all().item())
    ):
        return None

    blocks_per_head = head_dim // _BLOCK
    try:
        parent_sign = parent["sign"].reshape(
            sequence, kv_num_heads, blocks_per_head, _BLOCK
        )
        parent_mant = parent["mant"].reshape_as(parent_sign)
        effective = (
            parent["scale_factor"]
            * parent["scale_lv2"]
            * parent["scale_lv3"]
        ).expand_as(parent["sign"]).reshape_as(parent_sign).to(torch.float32)
        target = value.reshape(
            sequence, kv_num_heads, blocks_per_head, _BLOCK
        )
    except (KeyError, RuntimeError, ValueError):
        return None
    if not (
        bool(torch.isfinite(target).all().item())
        and bool(torch.isfinite(effective).all().item())
        and bool((effective > 0.0).all().item())
    ):
        return None

    current_code = torch.round(
        parent_sign.to(torch.float32) * parent_mant.to(torch.float32) * 4.0
    ).clamp(-7.0, 7.0)
    current_error = current_code * (effective * 0.25) - target
    hessian_diagonal = diagonal.transpose(0, 1) + factor.square().sum(dim=1).transpose(0, 1)
    if not (
        bool(torch.isfinite(hessian_diagonal).all().item())
        and bool((hessian_diagonal > 0.0).all().item())
    ):
        return None

    coordinate_index = torch.arange(
        _BLOCK, device=value.device, dtype=torch.long
    ).reshape(1, 1, _BLOCK)
    token_index = torch.arange(
        sequence, device=value.device, dtype=torch.long
    ).reshape(sequence, 1, 1)
    projected_error = torch.einsum(
        "hrt,thbc->rhbc", factor, current_error
    )
    for _ in range(_V_LOWRANK_DESCENT_STEPS):
        correction = torch.einsum(
            "hrt,rhbc->thbc", factor, projected_error
        )
        half_gradient = (
            diagonal.transpose(0, 1)[:, :, None, None] * current_error
            + correction
        )
        neighbor_code = (
            current_code - torch.sign(half_gradient)
        ).clamp(-7.0, 7.0)
        reconstruction_delta = (
            neighbor_code - current_code
        ) * (effective * 0.25)
        coordinate_delta = (
            2.0 * reconstruction_delta * half_gradient
            + reconstruction_delta.square()
            * hessian_diagonal[:, :, None, None]
        )
        best_coordinate_delta, coordinate = coordinate_delta.min(
            dim=-1, keepdim=True
        )
        best_block_delta, selected_token = best_coordinate_delta.squeeze(-1).min(
            dim=0
        )
        accept = torch.isfinite(best_block_delta) & (best_block_delta < 0.0)
        if not bool(accept.any().item()):
            break
        sparse_token = (token_index == selected_token.unsqueeze(0)).unsqueeze(-1)
        publish = (
            (coordinate_index.unsqueeze(0) == coordinate)
            & sparse_token
            & accept[None, :, :, None]
        )
        accepted_delta = torch.where(
            publish, reconstruction_delta, torch.zeros_like(reconstruction_delta)
        )
        current_code = torch.where(publish, neighbor_code, current_code)
        current_error = current_error + accepted_delta
        selected_coordinate = coordinate.squeeze(-1).gather(
            0, selected_token.unsqueeze(0)
        ).squeeze(0)
        accepted_value = accepted_delta.sum(dim=(0, 3))
        selected_factor = factor.gather(
            2,
            selected_token[:, None, :].expand(
                kv_num_heads, _V_LOWRANK_RANK, blocks_per_head
            ),
        ).permute(1, 0, 2)
        channel_mask = (
            coordinate_index == selected_coordinate.unsqueeze(-1)
        ).to(torch.float32)
        projected_error = projected_error + (
            selected_factor[..., None]
            * accepted_value[None, :, :, None]
            * channel_mask[None, :, :, :]
        )

    selected = dict(parent)
    selected["mant"] = (current_code.abs() * 0.25).to(
        parent["mant"].dtype
    ).reshape_as(parent["mant"])
    selected["sign"] = torch.where(
        current_code == 0.0,
        torch.zeros_like(current_code),
        torch.sign(current_code),
    ).to(parent["sign"].dtype).reshape_as(parent["sign"])
    return selected


def _attention_v_lowrank_repair(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    kv_num_heads: int | None,
    head_dim: int | None,
) -> dict[str, torch.Tensor]:
    """Repair V with a role-local normalized-position consumer kernel."""
    nystrom = _attention_v_nystrom_repair(
        value, parent, state, kv_num_heads, head_dim
    )
    if nystrom is not None:
        return nystrom
    if (
        _V_LOWRANK_ABLATION == "off"
        or state.get("lowrank_schema") != _V_LOWRANK_SCHEMA
        or kv_num_heads is None
        or head_dim is None
        or kv_num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or int(value.shape[-1]) != kv_num_heads * head_dim
    ):
        return parent
    gram_value = state.get("lowrank_gram")
    delta_value = state.get("lowrank_delta")
    valid_value = state.get("lowrank_valid")
    if (
        type(gram_value) is not torch.Tensor
        or type(delta_value) is not torch.Tensor
        or type(valid_value) is not torch.Tensor
        or tuple(gram_value.shape)
        != (kv_num_heads, _V_LOWRANK_RANK, _V_LOWRANK_RANK)
        or delta_value.numel() != kv_num_heads
        or valid_value.numel() != kv_num_heads
    ):
        return parent
    gram = gram_value.to(device=value.device, dtype=torch.float32)
    delta = delta_value.to(device=value.device, dtype=torch.float32).flatten()
    valid = valid_value.to(device=value.device, dtype=torch.bool).flatten()
    if not (
        bool(torch.isfinite(gram).all().item())
        and bool(torch.isfinite(delta).all().item())
        and bool((delta >= 0.0).all().item())
        and bool(valid.any().item())
    ):
        return parent
    if _V_LOWRANK_ABLATION not in ("full", "isotropic-only"):
        return parent

    channels = int(value.shape[-1])
    rows = value.numel() // channels
    if rows <= 0:
        return parent
    blocks_per_head = head_dim // _BLOCK
    active_rank = min(_V_LOWRANK_RANK, rows)
    basis = _normalized_dct_basis(
        rows, active_rank, value.device, torch.float32
    )
    gram = gram[:, :active_rank, :active_rank]
    identity = torch.eye(
        active_rank, device=value.device, dtype=torch.float32
    ).unsqueeze(0)
    centered = gram - delta[:, None, None] * identity
    if _V_LOWRANK_ABLATION == "isotropic-only":
        centered = torch.zeros_like(centered)

    try:
        parent_sign = parent["sign"].reshape(
            rows, kv_num_heads, blocks_per_head, _BLOCK
        )
        parent_mant = parent["mant"].reshape_as(parent_sign)
        effective = (
            parent["scale_factor"]
            * parent["scale_lv2"]
            * parent["scale_lv3"]
        ).expand_as(parent["sign"]).reshape_as(parent_sign).to(torch.float32)
        target = value.reshape(
            rows, kv_num_heads, blocks_per_head, _BLOCK
        )
    except (KeyError, RuntimeError, ValueError):
        return parent
    if not (
        bool(torch.isfinite(target).all().item())
        and bool(torch.isfinite(effective).all().item())
        and bool((effective > 0.0).all().item())
    ):
        return parent

    parent_code = torch.round(
        parent_sign.to(torch.float32) * parent_mant.to(torch.float32) * 4.0
    ).clamp(-7.0, 7.0)
    current_code = parent_code.clone()
    current_error = current_code * (effective * 0.25) - target
    diagonal = delta[None, :] + torch.einsum(
        "tr,hru,tu->th", basis, centered, basis
    )
    gram_psd = torch.linalg.eigvalsh(gram).amin(dim=-1) >= -1.0e-5
    runtime_valid = (
        valid
        & gram_psd
        & (delta > 0.0)
        & torch.isfinite(diagonal).all(dim=0)
        & (diagonal > 0.0).all(dim=0)
    )
    if not bool(runtime_valid.any().item()):
        return parent

    coordinate_index = torch.arange(
        _BLOCK, device=value.device, dtype=torch.long
    ).reshape(1, 1, 1, _BLOCK)
    token_index = torch.arange(
        rows, device=value.device, dtype=torch.long
    ).reshape(rows, 1, 1)
    # Reproduce V281's first global Gauss-Southwell move exactly, then continue
    # from that legal endpoint.  Each conditional step selects one coordinate
    # per KV-head/H64, so the calibration consumer model—not elementwise MSE—
    # remains the sole acceptance objective throughout the trajectory.
    projected_error = torch.einsum(
        "tr,thbc->rhbc", basis, current_error
    )
    for _ in range(_V_LOWRANK_DESCENT_STEPS):
        correction = torch.einsum(
            "tr,hru,uhbc->thbc", basis, centered, projected_error
        )
        half_gradient = (
            delta[None, :, None, None] * current_error + correction
        )
        neighbor_code = (
            current_code - torch.sign(half_gradient)
        ).clamp(-7.0, 7.0)
        reconstruction_delta = (
            neighbor_code - current_code
        ) * (effective * 0.25)
        coordinate_delta = (
            2.0 * reconstruction_delta * half_gradient
            + reconstruction_delta.square()
            * diagonal[:, :, None, None]
        )
        best_coordinate_delta, coordinate = coordinate_delta.min(
            dim=-1, keepdim=True
        )
        best_block_delta, selected_token = best_coordinate_delta.squeeze(-1).min(
            dim=0
        )
        sparse_token = (token_index == selected_token.unsqueeze(0)).unsqueeze(-1)
        accept = (
            runtime_valid[:, None]
            & torch.isfinite(best_block_delta)
            & (best_block_delta < 0.0)
        )
        if not bool(accept.any().item()):
            break
        publish = (
            (coordinate_index == coordinate)
            & sparse_token
            & accept[None, :, :, None]
        )
        accepted_delta = torch.where(
            publish, reconstruction_delta, torch.zeros_like(reconstruction_delta)
        )
        current_code = torch.where(publish, neighbor_code, current_code)
        current_error = current_error + accepted_delta
        selected_coordinate = coordinate.squeeze(-1).gather(
            0, selected_token.unsqueeze(0)
        ).squeeze(0)
        accepted_value = accepted_delta.sum(dim=(0, 3))
        projected_update = torch.zeros_like(projected_error).permute(1, 2, 0, 3)
        projected_update.scatter_(
            -1,
            selected_coordinate[:, :, None, None].expand(
                kv_num_heads, blocks_per_head, active_rank, 1
            ),
            (
                basis[selected_token].permute(0, 1, 2)
                * accepted_value[:, :, None]
            ).unsqueeze(-1),
        )
        projected_error = projected_error + projected_update.permute(2, 0, 1, 3)

    selected_code = current_code
    selected = dict(parent)
    selected["mant"] = (selected_code.abs() * 0.25).to(
        parent["mant"].dtype
    ).reshape_as(parent["mant"])
    selected["sign"] = torch.where(
        selected_code == 0.0,
        torch.zeros_like(selected_code),
        torch.sign(selected_code),
    ).to(parent["sign"].dtype).reshape_as(parent["sign"])
    return selected


def _k_breakpoint_direction(
    target: torch.Tensor,
    step: torch.Tensor,
    direction: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve every channel's monotone fixed-hierarchy code path."""
    if (
        target.ndim != 3
        or tuple(step.shape) != tuple(target.shape)
        or int(target.shape[-1]) != _BLOCK
        or int(target.shape[1]) < 2
        or direction not in (-1, 1)
    ):
        raise ValueError("invalid K breakpoint solver input")
    chains, sequence, channels = target.shape
    x = target.permute(0, 2, 1).to(torch.float32)
    delta = step.permute(0, 2, 1).to(torch.float32)
    if not (
        bool(torch.isfinite(x).all().item())
        and bool(torch.isfinite(delta).all().item())
        and bool((delta > 0.0).all().item())
    ):
        raise ValueError("non-finite K breakpoint input")

    start = torch.round(x / delta).clamp(-7.0, 7.0)
    residual = x - start * delta
    residual_sum = residual.sum(dim=-1)
    residual_square = residual.square().sum(dim=-1)
    base_risk = residual_square - residual_sum.square() / float(sequence)

    ordinal = torch.arange(
        1, 15, device=x.device, dtype=x.dtype
    ).reshape(1, 1, 1, 14)
    start_e = start.unsqueeze(-1)
    x_e = x.unsqueeze(-1)
    delta_e = delta.unsqueeze(-1)
    residual_e = residual.unsqueeze(-1)
    if direction > 0:
        new_code = start_e - ordinal
        threshold = x_e - delta_e * (new_code + 0.5)
        event_sum = delta_e.expand_as(threshold)
        event_square = (
            2.0 * residual_e * delta_e
            + (2.0 * ordinal - 1.0) * delta_e.square()
        )
        valid_code = new_code >= -7.0
    else:
        new_code = start_e + ordinal
        threshold = delta_e * (new_code - 0.5) - x_e
        event_sum = -delta_e.expand_as(threshold)
        event_square = (
            -2.0 * residual_e * delta_e
            + (2.0 * ordinal - 1.0) * delta_e.square()
        )
        valid_code = new_code <= 7.0

    valid = valid_code & torch.isfinite(threshold) & (threshold > 0.0)
    infinity = torch.full_like(threshold, float("inf"))
    threshold = torch.where(valid, threshold, infinity).reshape(
        chains, channels, -1
    )
    event_sum = torch.where(valid, event_sum, torch.zeros_like(event_sum)).reshape(
        chains, channels, -1
    )
    event_square = torch.where(
        valid, event_square, torch.zeros_like(event_square)
    ).reshape(chains, channels, -1)

    sorted_threshold, order = torch.sort(threshold, dim=-1, stable=True)
    sorted_sum = event_sum.gather(-1, order)
    sorted_square = event_square.gather(-1, order)
    prefix_sum = residual_sum.unsqueeze(-1) + sorted_sum.cumsum(dim=-1)
    prefix_square = residual_square.unsqueeze(-1) + sorted_square.cumsum(dim=-1)
    prefix_risk = prefix_square - prefix_sum.square() / float(sequence)

    next_threshold = torch.cat(
        (
            sorted_threshold[..., 1:],
            torch.full_like(sorted_threshold[..., :1], float("inf")),
        ),
        dim=-1,
    )
    interval_valid = torch.isfinite(sorted_threshold) & (
        next_threshold > sorted_threshold
    )
    prefix_risk = torch.where(
        interval_valid,
        prefix_risk,
        torch.full_like(prefix_risk, float("inf")),
    )
    best_risk, best_index = prefix_risk.min(dim=-1)
    lower = sorted_threshold.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    upper = next_threshold.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    last_interval_step = delta.amin(dim=-1).clamp_min(_MIN_SCALE) * 0.5
    shift = torch.where(
        torch.isfinite(upper),
        (lower + upper) * 0.5,
        lower + last_interval_step,
    )
    has_event = torch.isfinite(lower) & torch.isfinite(best_risk)
    return torch.where(has_event, best_risk, base_risk), torch.where(
        has_event, shift, torch.zeros_like(shift)
    )


def _k_breakpoint_gauge(
    target: torch.Tensor,
    step: torch.Tensor,
    hessian_diagonal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a parent-first token-common gauge and its diagonal risks."""
    if (
        target.ndim != 3
        or tuple(step.shape) != tuple(target.shape)
        or tuple(hessian_diagonal.shape)
        != (int(target.shape[0]), _BLOCK)
    ):
        raise ValueError("invalid K breakpoint gauge input")
    sequence = int(target.shape[1])
    x = target.permute(0, 2, 1).to(torch.float32)
    delta = step.permute(0, 2, 1).to(torch.float32)
    start = torch.round(x / delta).clamp(-7.0, 7.0)
    residual = x - start * delta
    base_risk = residual.square().sum(dim=-1) - residual.sum(
        dim=-1
    ).square() / float(sequence)
    positive_risk, positive_shift = _k_breakpoint_direction(target, step, 1)
    negative_risk, negative_magnitude = _k_breakpoint_direction(target, step, -1)

    best_risk = base_risk
    shift = torch.zeros_like(base_risk)
    choose_positive = positive_risk < best_risk
    best_risk = torch.where(choose_positive, positive_risk, best_risk)
    shift = torch.where(choose_positive, positive_shift, shift)
    choose_negative = (negative_risk < best_risk) | (
        (negative_risk == best_risk) & (negative_magnitude < shift.abs())
    )
    shift = torch.where(choose_negative, -negative_magnitude, shift)

    usable = torch.isfinite(hessian_diagonal) & (hessian_diagonal > 0.0)
    shift = torch.where(usable, shift, torch.zeros_like(shift))
    shifted = target - shift.unsqueeze(1)
    shifted_code = torch.round(shifted / step).clamp(-7.0, 7.0)
    shifted_error = shifted - shifted_code * step
    actual_risk = shifted_error.square().sum(dim=1) - shifted_error.sum(
        dim=1
    ).square() / float(sequence)
    changed = (shifted_code != start.permute(0, 2, 1)).any(dim=1)
    improved = (
        usable
        & changed
        & torch.isfinite(actual_risk)
        & (actual_risk < base_risk)
    )
    shift = torch.where(improved, shift, torch.zeros_like(shift))

    shifted = target - shift.unsqueeze(1)
    shifted_code = torch.round(shifted / step).clamp(-7.0, 7.0)
    shifted_error = shifted - shifted_code * step
    actual_risk = shifted_error.square().sum(dim=1) - shifted_error.sum(
        dim=1
    ).square() / float(sequence)
    return shift, base_risk, actual_risk


def _attention_k_breakpoint_overlay(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Re-encode one short code-transition gauge chain per batch/KV head."""
    if _K_BREAKPOINT_ABLATION == "mechanism-off":
        return parent
    if (
        _K_BREAKPOINT_ABLATION != "short-full"
        or value.ndim not in (2, 3)
        or num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or int(value.shape[-1]) != num_heads * head_dim
        or int(value.shape[-2]) < 2
        or int(value.shape[-2]) > _K_BREAKPOINT_SEQUENCE_LIMIT
        or state.get("hessian_head_dim") != head_dim
    ):
        return parent
    block_count = head_dim // _BLOCK
    expected_shape = (num_heads, block_count, _BLOCK, _BLOCK)
    hessian = state.get("cross_hessian")
    inverse = state.get("cross_inverse")
    importance = state.get("importance")
    if (
        type(hessian) is not torch.Tensor
        or type(inverse) is not torch.Tensor
        or type(importance) is not torch.Tensor
        or tuple(hessian.shape) != expected_shape
        or tuple(inverse.shape) != expected_shape
        or importance.numel() != num_heads * head_dim
    ):
        return parent

    batched = value.unsqueeze(0) if value.ndim == 2 else value
    batch, sequence = int(batched.shape[0]), int(batched.shape[1])
    try:
        hessian = hessian.to(device=value.device, dtype=torch.float32)
        inverse = inverse.to(device=value.device, dtype=torch.float32)
        importance = importance.to(device=value.device, dtype=torch.float32).reshape(
            num_heads, block_count, _BLOCK
        )
        if not (
            bool(torch.isfinite(hessian).all().item())
            and bool(torch.isfinite(inverse).all().item())
            and bool(torch.isfinite(importance).all().item())
            and bool((torch.diagonal(hessian, dim1=-2, dim2=-1) >= 0.0).all().item())
            and bool((torch.diagonal(inverse, dim1=-2, dim2=-1) > 0.0).all().item())
        ):
            return parent

        target = batched.reshape(batch, sequence, num_heads, block_count, _BLOCK)
        parent_quantized = _dequantize_hif4(parent)
        parent_batched = (
            parent_quantized.unsqueeze(0) if value.ndim == 2 else parent_quantized
        ).reshape_as(target)
        residual = target - parent_batched
        centered = residual - residual.mean(dim=1, keepdim=True)
        parent_risk = torch.einsum(
            "bthgi,hgij,bthgj->bhg", centered, hessian, centered
        )
        finite_risk = torch.isfinite(parent_risk) & (parent_risk > 0.0)
        routed_risk = torch.where(
            finite_risk, parent_risk, torch.full_like(parent_risk, -1.0)
        )
        selected_block = routed_risk.argmax(dim=-1)
        selected_valid = finite_risk.gather(
            -1, selected_block.unsqueeze(-1)
        ).squeeze(-1)

        batch_index = torch.arange(
            batch, device=value.device, dtype=torch.long
        ).repeat_interleave(num_heads)
        head_index = torch.arange(
            num_heads, device=value.device, dtype=torch.long
        ).repeat(batch)
        flat_block = selected_block.reshape(-1)
        flat_valid = selected_valid.reshape(-1)
        selected_target = target[batch_index, :, head_index, flat_block, :]
        selected_hessian = hessian[head_index, flat_block]
        selected_inverse = inverse[head_index, flat_block]

        effective = (
            parent["scale_factor"].to(torch.float32)
            * parent["scale_lv2"].to(torch.float32)
            * parent["scale_lv3"].to(torch.float32)
        ).expand_as(parent["mant"])
        effective_batched = (
            effective.unsqueeze(0) if value.ndim == 2 else effective
        ).reshape(batch, sequence, num_heads, block_count, _BLOCK)
        selected_step = effective_batched[
            batch_index, :, head_index, flat_block, :
        ] * 0.25
        shift, diagonal_parent, diagonal_candidate = _k_breakpoint_gauge(
            selected_target,
            selected_step,
            torch.diagonal(selected_hessian, dim1=-2, dim2=-1),
        )
        diagonal = torch.diagonal(selected_hessian, dim1=-2, dim2=-1)
        surrogate_parent = (diagonal_parent * diagonal).sum(dim=-1)
        surrogate_candidate = (diagonal_candidate * diagonal).sum(dim=-1)
        gauge_valid = (
            flat_valid
            & torch.isfinite(surrogate_parent)
            & torch.isfinite(surrogate_candidate)
            & (surrogate_candidate < surrogate_parent)
            & (shift != 0.0).any(dim=-1)
        )
        shift = torch.where(gauge_valid.unsqueeze(-1), shift, torch.zeros_like(shift))
        candidate_target = selected_target - shift.unsqueeze(1)
        packed_target = candidate_target.permute(1, 0, 2).reshape(sequence, -1)
        packed_importance = importance[head_index, flat_block].reshape(-1)
        candidate = _quantize_sensitive(
            packed_target, packed_importance, _ATTENTION_GAIN
        )
        packed_state = {
            "cross_hessian": selected_hessian.reshape(-1, 1, _BLOCK, _BLOCK),
            "cross_inverse": selected_inverse.reshape(-1, 1, _BLOCK, _BLOCK),
            "hessian_head_dim": _BLOCK,
        }
        candidate = _attention_hessian_repair(
            packed_target,
            candidate,
            packed_state,
            int(selected_target.shape[0]),
            _BLOCK,
        )
        candidate_quantized = _dequantize_hif4(candidate).reshape(
            sequence, -1, _BLOCK
        ).permute(1, 0, 2)
        candidate_residual = candidate_target - candidate_quantized
        candidate_centered = candidate_residual - candidate_residual.mean(
            dim=1, keepdim=True
        )
        selected_parent_risk = parent_risk[batch_index, head_index, flat_block]
        candidate_risk = torch.einsum(
            "nti,nij,ntj->n", candidate_centered, selected_hessian, candidate_centered
        )

        parent_fields: dict[str, torch.Tensor] = {}
        candidate_fields: dict[str, torch.Tensor] = {}
        changed = torch.zeros(
            int(selected_target.shape[0]), device=value.device, dtype=torch.bool
        )
        for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
            field = parent[key]
            field_batched = field.unsqueeze(0) if value.ndim == 2 else field
            tail = tuple(field_batched.shape[3:])
            field_view = field_batched.reshape(
                batch, sequence, num_heads, block_count, *tail
            )
            parent_selected = field_view[batch_index, :, head_index, flat_block]
            candidate_field = candidate[key]
            candidate_tail = tuple(candidate_field.shape[2:])
            if candidate_tail != tail:
                return parent
            candidate_selected = candidate_field.reshape(
                sequence, -1, *tail
            ).permute(1, 0, *range(2, 2 + len(tail)))
            parent_fields[key] = field_view
            candidate_fields[key] = candidate_selected
            difference = candidate_selected != parent_selected
            changed |= difference.reshape(difference.shape[0], -1).any(dim=-1)

        accept = (
            gauge_valid
            & changed
            & torch.isfinite(selected_parent_risk)
            & torch.isfinite(candidate_risk)
            & (candidate_risk < selected_parent_risk * (1.0 - _HESSIAN_GAIN))
        )
        if not bool(accept.any().item()):
            return parent

        selected = dict(parent)
        for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
            field = parent[key]
            tail = tuple(parent_fields[key].shape[4:])
            updated = parent_fields[key].clone()
            current = updated[batch_index, :, head_index, flat_block]
            mask_shape = (int(accept.numel()), 1) + (1,) * len(tail)
            merged = torch.where(
                accept.reshape(mask_shape), candidate_fields[key], current
            )
            updated[batch_index, :, head_index, flat_block] = merged
            restored = updated.reshape(
                (batch,) + tuple(field.shape) if value.ndim == 2 else tuple(field.shape)
            )
            selected[key] = restored.squeeze(0) if value.ndim == 2 else restored
        return selected
    except (KeyError, RuntimeError, ValueError, OverflowError, IndexError):
        return parent


def _attention_k_gauge_route(
    value: torch.Tensor,
    num_heads: int | None,
    head_dim: int | None,
) -> torch.Tensor:
    """Refine V250's K gauge by one convex coordinate per H64."""
    if (
        value.ndim not in (2, 3)
        or num_heads is None
        or head_dim is None
        or num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or int(value.shape[-1]) != num_heads * head_dim
    ):
        return value
    sequence = int(value.shape[-2])
    if sequence < 2 or not bool(torch.isfinite(value).all().item()):
        return value

    batched = value.unsqueeze(0) if value.ndim == 2 else value
    batch = int(batched.shape[0])
    block_count = head_dim // _BLOCK
    parent_routed: torch.Tensor | None = None
    try:
        shaped = batched.reshape(
            batch, sequence, num_heads, block_count, _BLOCK
        )
        mean = shaped.mean(dim=1, keepdim=True)
        centered = shaped - mean
        if not (
            bool(torch.isfinite(mean).all().item())
            and bool(torch.isfinite(centered).all().item())
        ):
            return value

        identity_pressure = shaped.abs().amax(dim=-1).square().sum(dim=1)
        centered_pressure = centered.abs().amax(dim=-1).square().sum(dim=1)
        if not (
            bool(torch.isfinite(identity_pressure).all().item())
            and bool(torch.isfinite(centered_pressure).all().item())
        ):
            return value

        # This is byte-for-byte the V250 parent decision: identity wins ties.
        use_centered = centered_pressure < identity_pressure
        mask = use_centered.unsqueeze(1).unsqueeze(-1)
        parent = torch.where(mask, centered, shaped)
        parent_routed = parent.reshape_as(batched)
        if _K_GAUGE_ABLATION == "mechanism-off":
            return parent_routed.squeeze(0) if value.ndim == 2 else parent_routed
        if _K_GAUGE_ABLATION not in (
            "fixed-channel-zero",
            "best-channel-midrange",
            "best-channel-bisection",
        ):
            return parent_routed.squeeze(0) if value.ndim == 2 else parent_routed

        parent_offset = torch.where(mask, mean, torch.zeros_like(mean))
        parent_pressure = torch.where(
            use_centered, centered_pressure, identity_pressure
        )
        parent_square = parent.square()

        if _K_GAUGE_ABLATION == "fixed-channel-zero":
            candidate_value = shaped[..., :1]
            other_pressure = parent_square[..., 1:].amax(
                dim=-1, keepdim=True
            )
        else:
            top_values, top_indices = torch.topk(
                parent_square, k=2, dim=-1, largest=True, sorted=True
            )
            channel_indices = torch.arange(
                _BLOCK, device=value.device, dtype=top_indices.dtype
            ).reshape(1, 1, 1, 1, _BLOCK)
            other_pressure = torch.where(
                channel_indices == top_indices[..., :1],
                top_values[..., 1:2],
                top_values[..., :1],
            )
            candidate_value = shaped

        lower = candidate_value.amin(dim=1, keepdim=True)
        upper = candidate_value.amax(dim=1, keepdim=True)
        if _K_GAUGE_ABLATION == "best-channel-midrange":
            candidate_center = (lower + upper) * 0.5
        else:
            for _ in range(_K_GAUGE_BISECTION_STEPS):
                midpoint = (lower + upper) * 0.5
                difference = midpoint - candidate_value
                active = difference.square() > other_pressure
                gradient = torch.where(
                    active, difference, torch.zeros_like(difference)
                ).sum(dim=1, keepdim=True)
                move_lower = gradient <= 0.0
                lower = torch.where(move_lower, midpoint, lower)
                upper = torch.where(move_lower, upper, midpoint)
            candidate_center = (lower + upper) * 0.5

        candidate_pressure = torch.maximum(
            (candidate_value - candidate_center).square(), other_pressure
        ).sum(dim=1)
        if not (
            bool(torch.isfinite(candidate_center).all().item())
            and bool(torch.isfinite(candidate_pressure).all().item())
        ):
            return parent_routed.squeeze(0) if value.ndim == 2 else parent_routed

        if _K_GAUGE_ABLATION == "fixed-channel-zero":
            best_index = torch.zeros_like(parent_pressure, dtype=torch.int64)
            best_pressure = candidate_pressure.squeeze(-1)
            best_center = candidate_center.squeeze(1).squeeze(-1)
        else:
            best_pressure, best_index = candidate_pressure.min(dim=-1)
            best_center = candidate_center.squeeze(1).gather(
                -1, best_index.unsqueeze(-1)
            ).squeeze(-1)

        accept = best_pressure < parent_pressure
        candidate_offset = parent_offset.squeeze(1).scatter(
            -1, best_index.unsqueeze(-1), best_center.unsqueeze(-1)
        )
        selected_offset = torch.where(
            accept.unsqueeze(-1), candidate_offset, parent_offset.squeeze(1)
        ).unsqueeze(1)
        routed = (shaped - selected_offset).reshape_as(batched)
    except (RuntimeError, ValueError, OverflowError):
        if parent_routed is None:
            return value
        return parent_routed.squeeze(0) if value.ndim == 2 else parent_routed
    return routed.squeeze(0) if value.ndim == 2 else routed


def _router_gather_value_blocks(
    value: torch.Tensor, block_indices: torch.Tensor
) -> torch.Tensor:
    """Pack selected H64 blocks while preserving every leading dimension."""
    channels = int(value.shape[-1])
    if channels % _BLOCK or block_indices.ndim != 1 or block_indices.numel() == 0:
        raise ValueError("invalid router value-block selection")
    total_blocks = channels // _BLOCK
    indices = block_indices.to(device=value.device, dtype=torch.long)
    if bool(((indices < 0) | (indices >= total_blocks)).any().item()):
        raise ValueError("router value-block index out of range")
    selected = value.reshape(*value.shape[:-1], total_blocks, _BLOCK).index_select(
        -2, indices
    )
    return selected.flatten(-2, -1)


def _router_gather_parameters(
    params: dict[str, torch.Tensor], block_indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Pack the five legal HiF4 fields along their common H64 axis."""
    selected: dict[str, torch.Tensor] = {}
    for key in _HIF4_PARAMETER_FIELDS:
        field = params[key]
        if field.ndim < 4:
            raise ValueError("invalid HiF4 parameter rank")
        indices = block_indices.to(device=field.device, dtype=torch.long)
        selected[key] = field.index_select(field.ndim - 4, indices)
    return selected


def _router_scatter_parameters(
    parent: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    block_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Publish selected legal fields without touching any unselected block."""
    selected = dict(parent)
    for key in _HIF4_PARAMETER_FIELDS:
        parent_field = parent[key]
        candidate_field = candidate[key]
        axis = parent_field.ndim - 4
        indices = block_indices.to(device=parent_field.device, dtype=torch.long)
        expected = list(parent_field.shape)
        expected[axis] = int(indices.numel())
        if tuple(candidate_field.shape) != tuple(expected):
            raise ValueError("routed HiF4 parameter shape mismatch")
        updated = parent_field.clone()
        updated.index_copy_(
            axis,
            indices,
            candidate_field.to(device=updated.device, dtype=updated.dtype),
        )
        selected[key] = updated
    return selected


def _router_changed_units(
    parent: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    blocks_per_unit: int,
) -> int:
    """Count routed units whose emitted legal representation actually changed."""
    changed: torch.Tensor | None = None
    for key in _HIF4_PARAMETER_FIELDS:
        parent_field = parent[key]
        candidate_field = candidate[key]
        if parent_field.shape != candidate_field.shape or parent_field.ndim < 4:
            return 0
        axis = parent_field.ndim - 4
        difference = (parent_field != candidate_field).movedim(axis, 0)
        block_changed = difference.reshape(difference.shape[0], -1).any(dim=-1)
        changed = block_changed if changed is None else (changed | block_changed)
    if (
        changed is None
        or blocks_per_unit <= 0
        or changed.numel() % blocks_per_unit
    ):
        return 0
    return int(changed.reshape(-1, blocks_per_unit).any(dim=-1).sum().item())


def _router_unit_indices(
    mask: torch.Tensor, head_dim: int, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flat window ids and their ordered constituent H64 block ids."""
    if (
        mask.ndim != 2
        or head_dim <= 0
        or window <= 0
        or head_dim % window
        or window % _BLOCK
        or int(mask.shape[1]) != head_dim // window
    ):
        raise ValueError("invalid repair-router unit mask")
    unit_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten().cpu()
    if unit_indices.numel() == 0:
        return unit_indices, torch.empty(0, dtype=torch.long)
    windows_per_head = head_dim // window
    blocks_per_head = head_dim // _BLOCK
    blocks_per_window = window // _BLOCK
    head_indices = torch.div(unit_indices, windows_per_head, rounding_mode="floor")
    window_indices = unit_indices.remainder(windows_per_head)
    starts = head_indices * blocks_per_head + window_indices * blocks_per_window
    block_indices = (
        starts[:, None] + torch.arange(blocks_per_window, dtype=torch.long)[None, :]
    ).reshape(-1)
    return unit_indices, block_indices


def _router_slice_qk_state(
    state: dict[str, Any],
    unit_indices: torch.Tensor,
    block_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    window: int,
) -> dict[str, Any]:
    """Build the exact compact state expected by one packed Q/K expert."""
    units = int(unit_indices.numel())
    blocks_per_head = head_dim // _BLOCK
    blocks_per_window = window // _BLOCK
    windows_per_head = head_dim // window
    if units <= 0 or block_indices.numel() != units * blocks_per_window:
        raise ValueError("empty compact Q/K router state")
    selected = dict(state)
    for key in ("cross_hessian", "cross_inverse"):
        tensor = state.get(key)
        if (
            type(tensor) is not torch.Tensor
            or tuple(tensor.shape)
            != (num_heads, blocks_per_head, _BLOCK, _BLOCK)
        ):
            raise ValueError("invalid compact Q/K Hessian state")
        selected[key] = tensor.reshape(-1, _BLOCK, _BLOCK).index_select(
            0, block_indices.to(dtype=torch.long)
        ).reshape(units, blocks_per_window, _BLOCK, _BLOCK)
    importance = state.get("importance")
    if type(importance) is not torch.Tensor or importance.numel() != num_heads * head_dim:
        raise ValueError("invalid compact Q/K importance")
    selected["importance"] = importance.reshape(-1, _BLOCK).index_select(
        0, block_indices.to(dtype=torch.long)
    ).flatten()
    selected["hessian_head_dim"] = window
    # Router pilots intentionally use at most a small calibration token sample.
    # Force the same wide-window operator selected at dynamic time so measured
    # family costs cannot silently benchmark the cheaper H64 fallback.
    selected["repair_router_force_wide"] = True

    expected_wide = (num_heads, windows_per_head, window, window)
    for key in ("wide_cross_hessian_t", "wide_cross_hessian_t_bf16"):
        tensor = state.get(key)
        if type(tensor) is torch.Tensor and tuple(tensor.shape) == expected_wide:
            selected[key] = tensor.reshape(-1, window, window).index_select(
                0, unit_indices.to(dtype=torch.long)
            ).reshape(units, 1, window, window)
        else:
            selected.pop(key, None)
    if window > _BLOCK and "wide_cross_hessian_t" not in selected:
        raise ValueError("missing compact Q/K wide-window state")
    if "wide_cross_hessian_t" in selected:
        selected["wide_cross_schema"] = _QK_WIDE_SCHEMA
        selected["wide_cross_window"] = window
    return selected


def _router_slice_v_state(
    state: dict[str, Any],
    block_indices: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Treat every selected V H64 block as one compact consumer head."""
    blocks_per_head = head_dim // _BLOCK
    head_indices = torch.div(
        block_indices.to(dtype=torch.long), blocks_per_head, rounding_mode="floor"
    )
    if head_indices.numel() == 0:
        raise ValueError("empty compact V router state")
    selected = dict(state)
    importance = state.get("importance")
    if type(importance) is not torch.Tensor or importance.numel() != kv_num_heads * head_dim:
        raise ValueError("invalid compact V importance")
    selected["importance"] = importance.reshape(-1, _BLOCK).index_select(
        0, block_indices.to(dtype=torch.long)
    ).flatten()

    for key in ("pair_rho", "pair_valid", "pair_rho2", "pair_lag2_valid"):
        tensor = state.get(key)
        if type(tensor) is torch.Tensor and tensor.numel() == kv_num_heads:
            selected[key] = tensor.flatten().index_select(0, head_indices)
    for key in ("profile_unary", "profile_edge", "profile_valid"):
        tensor = state.get(key)
        if type(tensor) is torch.Tensor and tensor.ndim >= 2 and tensor.shape[1] == kv_num_heads:
            selected[key] = tensor.index_select(1, head_indices)
    for key in ("lowrank_gram", "lowrank_delta", "lowrank_valid"):
        tensor = state.get(key)
        if type(tensor) is torch.Tensor and tensor.shape[0] == kv_num_heads:
            selected[key] = tensor.index_select(0, head_indices)
    for key in ("nystrom_factors", "nystrom_diagonals", "nystrom_valid"):
        values = state.get(key)
        if isinstance(values, tuple):
            compact_values = []
            for tensor in values:
                if type(tensor) is not torch.Tensor or tensor.shape[0] != kv_num_heads:
                    raise ValueError("invalid compact V Nystrom state")
                compact_values.append(tensor.index_select(0, head_indices))
            selected[key] = tuple(compact_values)
    return selected


def _attention_k_base_gauge_route(
    value: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Keep V22's cheap identity-vs-centered K gauge, without bisection."""
    if (
        value.ndim not in (2, 3)
        or num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or int(value.shape[-1]) != num_heads * head_dim
        or int(value.shape[-2]) < 2
        or not bool(torch.isfinite(value).all().item())
    ):
        return value
    batched = value.unsqueeze(0) if value.ndim == 2 else value
    try:
        batch, sequence = int(batched.shape[0]), int(batched.shape[1])
        shaped = batched.reshape(
            batch, sequence, num_heads, head_dim // _BLOCK, _BLOCK
        )
        mean = shaped.mean(dim=1, keepdim=True)
        centered = shaped - mean
        identity_pressure = shaped.abs().amax(dim=-1).square().sum(dim=1)
        centered_pressure = centered.abs().amax(dim=-1).square().sum(dim=1)
        if not (
            bool(torch.isfinite(centered).all().item())
            and bool(torch.isfinite(identity_pressure).all().item())
            and bool(torch.isfinite(centered_pressure).all().item())
        ):
            return value
        use_centered = centered_pressure < identity_pressure
        routed = torch.where(
            use_centered.unsqueeze(1).unsqueeze(-1), centered, shaped
        ).reshape_as(batched)
    except (RuntimeError, ValueError, OverflowError):
        return value
    return routed.squeeze(0) if value.ndim == 2 else routed


def _router_apply_q_expert(
    value: torch.Tensor,
    state: dict[str, Any],
    units: int,
    window: int,
) -> dict[str, torch.Tensor]:
    importance = state["importance"].to(device=value.device, dtype=torch.float32)
    parent = _quantize_sensitive(value, importance, _ATTENTION_GAIN)
    selected = _attention_hessian_repair(value, parent, state, units, window)
    return _attention_q_fixed_scale_code_descent(
        value, selected, state, units, window
    )


def _router_apply_k_expert(
    value: torch.Tensor,
    state: dict[str, Any],
    units: int,
    window: int,
) -> dict[str, torch.Tensor]:
    importance = state["importance"].to(device=value.device, dtype=torch.float32)
    routed = _attention_k_gauge_route(value, units, window)
    parent = _quantize_sensitive(routed, importance, _ATTENTION_GAIN)
    parent = _attention_hessian_repair(routed, parent, state, units, window)
    parent = _attention_k_fixed_scale_code_descent(
        routed, parent, state, units, window
    )
    parent = _attention_k_two_coordinate_code_solve(
        routed, parent, state, units, window
    )
    return _attention_k_breakpoint_overlay(
        routed, parent, state, units, window
    )


def _router_apply_v_expert(
    value: torch.Tensor,
    state: dict[str, Any],
    units: int,
) -> dict[str, torch.Tensor]:
    importance = state["importance"].to(device=value.device, dtype=torch.float32)
    parent, alternate = _quantize_sensitive_branches(
        value, importance, _ATTENTION_GAIN
    )
    selected = _attention_v_hierarchy_chain_select(
        value, parent, alternate, state, units, _BLOCK
    )
    selected = _attention_v_chain_repair(
        value, selected, state, units, _BLOCK
    )
    selected = _attention_v_global_dc_repair(
        value, selected, state, units, _BLOCK
    )
    return _attention_v_lowrank_repair(
        value, selected, state, units, _BLOCK
    )


def _router_format_error_split(
    target: torch.Tensor, params: dict[str, torch.Tensor]
) -> tuple[float, float]:
    """Separate saturation residual from in-range grid-resolution residual."""
    reconstruction = _dequantize_hif4(params).reshape_as(target)
    effective = (
        params["scale_factor"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
    ).expand_as(params["mant"])
    maximum = (effective * 1.75).flatten(-4, -1).reshape_as(target)
    clipped = target.abs() > maximum
    clipping = torch.where(
        clipped, (target.abs() - maximum).clamp_min(0.0).square(), 0.0
    ).sum()
    resolution = torch.where(
        clipped, 0.0, (reconstruction - target).square()
    ).sum()
    return float(clipping.item()), float(resolution.item())


def _router_attention_risks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_reconstruction: torch.Tensor,
    k_reconstruction: torch.Tensor,
    v_reconstruction: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor | float]:
    """Measure joint window Q/K and exact fixed-P V output risks."""
    sequence = int(q.shape[0])
    heads_per_kv = q_num_heads // kv_num_heads
    blocks_per_head = head_dim // _BLOCK
    if (
        sequence <= 0
        or tuple(q.shape) != (sequence, q_num_heads, head_dim)
        or tuple(k.shape) != (sequence, kv_num_heads, head_dim)
        or tuple(v.shape) != (sequence, kv_num_heads, head_dim)
        or q_reconstruction.shape != q.shape
        or k_reconstruction.shape != k.shape
        or v_reconstruction.shape != v.shape
    ):
        raise ValueError("invalid repair-router risk inputs")

    q_grouped = q.reshape(
        sequence, kv_num_heads, heads_per_kv, head_dim
    ).permute(1, 2, 0, 3)
    k_grouped = k.permute(1, 0, 2)
    v_grouped = v.permute(1, 0, 2)
    probability = torch.softmax(
        torch.matmul(q_grouped, k_grouped.transpose(-1, -2))
        * (float(head_dim) ** -0.5),
        dim=-1,
    )

    q_error = (q_reconstruction - q).reshape(
        sequence, kv_num_heads, heads_per_kv, blocks_per_head, _BLOCK
    ).permute(1, 2, 3, 0, 4)
    k_error = (k_reconstruction - k).reshape(
        sequence, kv_num_heads, blocks_per_head, _BLOCK
    ).permute(1, 2, 0, 3)
    q_blocks = q.reshape(
        sequence, kv_num_heads, heads_per_kv, blocks_per_head, _BLOCK
    ).permute(1, 2, 3, 0, 4)
    k_blocks = k.reshape(
        sequence, kv_num_heads, blocks_per_head, _BLOCK
    ).permute(1, 2, 0, 3)
    inverse_temperature = float(head_dim) ** -0.5
    q_delta = torch.einsum(
        "ghbti,gbsi->ghbts", q_error, k_blocks
    ) * inverse_temperature
    k_delta = torch.einsum(
        "ghbti,gbsi->ghbts", q_blocks, k_error
    ) * inverse_temperature

    def response_and_risk(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        centered = delta - delta.mean(dim=-1, keepdim=True)
        expanded_probability = probability.unsqueeze(2)
        response = expanded_probability * (
            centered
            - (expanded_probability * centered).sum(dim=-1, keepdim=True)
        )
        output = torch.einsum("ghbts,gsd->ghbtd", response, v_grouped)
        risk = output.square().sum(dim=(-2, -1))
        energy = float(centered.square().sum().item())
        return response, risk, energy

    q_response, q_risk, q_energy = response_and_risk(q_delta)
    k_response, k_risk, k_energy = response_and_risk(k_delta)

    window = _qk_wide_window(head_dim)
    if window <= 0 or head_dim % window or window % _BLOCK:
        raise ValueError("invalid repair-router risk window")
    windows_per_head = head_dim // window
    blocks_per_window = window // _BLOCK

    # One routed Q/K item contains the matching Q and K coordinates of a KV
    # group/window.  Form its complete logit perturbation before applying the
    # Jacobian and squaring.  Summing H64 contributions first retains all
    # cross-block inner products in ||(J delta_L_i) V||^2; adding the local
    # dQ dK term also keeps the coupled quantization error in the route value.
    cross_delta_blocks = torch.einsum(
        "ghbti,gbsi->ghbts", q_error, k_error
    ) * inverse_temperature
    joint_delta = (q_delta + k_delta + cross_delta_blocks).reshape(
        kv_num_heads,
        heads_per_kv,
        windows_per_head,
        blocks_per_window,
        sequence,
        sequence,
    ).sum(dim=3)
    _, qk_window_risk_per_head, _ = response_and_risk(joint_delta)
    qk_window_risk = qk_window_risk_per_head.sum(dim=1)

    q_error_full = q_error.permute(0, 1, 3, 2, 4).reshape(
        kv_num_heads, heads_per_kv, sequence, head_dim
    )
    k_error_full = k_error.permute(0, 2, 1, 3).reshape(
        kv_num_heads, sequence, head_dim
    )
    cross_delta = torch.einsum(
        "ghti,gsi->ghts", q_error_full, k_error_full
    ) * inverse_temperature
    cross_centered = cross_delta - cross_delta.mean(dim=-1, keepdim=True)
    cross_response = probability * (
        cross_centered
        - (probability * cross_centered).sum(dim=-1, keepdim=True)
    )
    cross_output = torch.einsum("ghts,gsd->ghtd", cross_response, v_grouped)
    cross_risk = float(cross_output.square().sum().item())

    v_error = (v_reconstruction - v).reshape(
        sequence, kv_num_heads, blocks_per_head, _BLOCK
    ).permute(1, 2, 0, 3)
    v_output = torch.einsum("ghts,gbsi->ghtbi", probability, v_error)
    v_risk = v_output.square().sum(dim=(1, 2, 4))
    combined_response = q_response.sum(dim=2) + k_response.sum(dim=2) + cross_response
    probability_value_cross = torch.einsum(
        "ghts,gsd->ghtd",
        combined_response,
        (v_reconstruction - v).permute(1, 0, 2),
    )

    tensors = (q_risk, k_risk, qk_window_risk, v_risk)
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise ValueError("non-finite repair-router output risk")
    scalar_values = (
        q_energy,
        k_energy,
        float(cross_centered.square().sum().item()),
        float(q_risk.sum().item()),
        float(k_risk.sum().item()),
        cross_risk,
        float(qk_window_risk.sum().item()),
        float(v_risk.sum().item()),
        float(probability_value_cross.square().sum().item()),
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("non-finite repair-router scalar risk")
    return {
        "q_risk": q_risk.cpu(),
        "k_risk": k_risk.cpu(),
        "qk_window_risk": qk_window_risk.cpu(),
        "v_risk": v_risk.cpu(),
        "q_energy": scalar_values[0],
        "k_energy": scalar_values[1],
        "cross_energy": scalar_values[2],
        "q_output": scalar_values[3],
        "k_output": scalar_values[4],
        "cross_output": scalar_values[5],
        "qk_joint_output": scalar_values[6],
        "v_output": scalar_values[7],
        "probability_value_cross": scalar_values[8],
    }


def _router_pilot_state(
    sample: dict[str, Any],
    state: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Construct one bounded representative parent and its decomposed risks."""
    q_state = state["q_state"]
    k_state = state["k_state"]
    v_state = state["v_state"]
    q = dequantize_nvfp4(*sample["q"]).to(torch.float32).reshape(
        -1, q_num_heads, head_dim
    )
    k = dequantize_nvfp4(*sample["k"]).to(torch.float32).reshape(
        -1, kv_num_heads, head_dim
    )
    v = dequantize_nvfp4(*sample["v"]).to(torch.float32).reshape(
        -1, kv_num_heads, head_dim
    )
    sequence = min(int(q.shape[0]), int(k.shape[0]), int(v.shape[0]))
    if sequence <= 0:
        raise ValueError("empty Attention router pilot")
    if sequence > _REPAIR_ROUTER_PILOT_TOKENS:
        indices = torch.linspace(
            0,
            sequence - 1,
            _REPAIR_ROUTER_PILOT_TOKENS,
            device=q.device,
            dtype=torch.float32,
        ).round().to(torch.long)
        q = q.index_select(0, indices)
        k = k.index_select(0, indices)
        v = v.index_select(0, indices)
    else:
        q, k, v = q[:sequence], k[:sequence], v[:sequence]
    sequence = int(q.shape[0])

    q_smooth = q_state["smooth_scale"].to(
        device=q.device, dtype=torch.float32
    ).reshape(q_num_heads, head_dim)
    k_smooth = k_state["smooth_scale"].to(
        device=k.device, dtype=torch.float32
    ).reshape(kv_num_heads, head_dim)
    q = q / q_smooth
    k = k * k_smooth
    q_transform = q_state.get("pair_transform")
    k_transform = k_state.get("pair_transform")
    if q_transform is not None or k_transform is not None:
        if type(q_transform) is not torch.Tensor or type(k_transform) is not torch.Tensor:
            raise ValueError("incomplete reciprocal router transform")
        q = _apply_head_transform(q, q_transform)
        k = _apply_head_transform(k, k_transform)
    if not (
        bool(torch.isfinite(q).all().item())
        and bool(torch.isfinite(k).all().item())
        and bool(torch.isfinite(v).all().item())
    ):
        raise ValueError("non-finite Attention router pilot")

    q_flat = q.reshape(sequence, -1)
    k_flat = k.reshape(sequence, -1)
    v_flat = v.reshape(sequence, -1)
    q_parent = _quantize_hif4(q_flat)
    k_base = _attention_k_base_gauge_route(k_flat, kv_num_heads, head_dim)
    k_parent = _quantize_hif4(k_base)
    v_parent = _quantize_hif4(v_flat)
    q_reconstruction = _dequantize_hif4(q_parent).reshape_as(q)
    k_reconstruction = _dequantize_hif4(k_parent).reshape_as(k)
    v_reconstruction = _dequantize_hif4(v_parent).reshape_as(v)
    risks = _router_attention_risks(
        q,
        k,
        v,
        q_reconstruction,
        k_reconstruction,
        v_reconstruction,
        q_num_heads,
        kv_num_heads,
        head_dim,
    )
    for target, params in (
        (q_flat, q_parent),
        (k_base, k_parent),
        (v_flat, v_parent),
    ):
        clipping, resolution = _router_format_error_split(target, params)
        _diagnostic_add("format_clipping_sse", clipping)
        _diagnostic_add("format_resolution_sse", resolution)
    return {
        "q": q,
        "k": k,
        "v": v,
        "q_flat": q_flat,
        "k_flat": k_flat,
        "v_flat": v_flat,
        "q_parent": q_parent,
        "k_parent": k_parent,
        "v_parent": v_parent,
        "q_reconstruction": q_reconstruction,
        "k_reconstruction": k_reconstruction,
        "v_reconstruction": v_reconstruction,
        "risks": risks,
        "tokens": sequence,
    }


def _router_pilot_qk_expert(
    pilot: dict[str, Any],
    state: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    qk_risk: torch.Tensor,
    flat_index: int,
) -> tuple[int, float]:
    """Time one actual paired Q/K unit and return its observed risk fraction."""
    heads_per_kv = q_num_heads // kv_num_heads
    window = _qk_wide_window(head_dim)
    windows_per_head = head_dim // window
    if flat_index < 0 or flat_index >= qk_risk.numel():
        raise ValueError("Q/K pilot route index out of range")
    group = flat_index // windows_per_head
    window_index = flat_index % windows_per_head
    group_mask = torch.zeros(kv_num_heads, windows_per_head, dtype=torch.bool)
    group_mask[group, window_index] = True
    q_mask = group_mask.repeat_interleave(heads_per_kv, dim=0)
    q_units, q_blocks = _router_unit_indices(q_mask, head_dim, window)
    k_units, k_blocks = _router_unit_indices(group_mask, head_dim, window)
    q_value = _router_gather_value_blocks(pilot["q_flat"], q_blocks)
    k_value = _router_gather_value_blocks(pilot["k_flat"], k_blocks)
    q_compact = _router_slice_qk_state(
        state["q_state"], q_units, q_blocks, q_num_heads, head_dim, window
    )
    k_compact = _router_slice_qk_state(
        state["k_state"], k_units, k_blocks, kv_num_heads, head_dim, window
    )
    started = time.perf_counter_ns()
    q_candidate = _router_apply_q_expert(
        q_value, q_compact, int(q_units.numel()), window
    )
    k_candidate = _router_apply_k_expert(
        k_value, k_compact, int(k_units.numel()), window
    )
    elapsed = max(time.perf_counter_ns() - started, 1)
    q_merged = _router_scatter_parameters(
        pilot["q_parent"], q_candidate, q_blocks
    )
    k_merged = _router_scatter_parameters(
        pilot["k_parent"], k_candidate, k_blocks
    )
    candidate_risks = _router_attention_risks(
        pilot["q"],
        pilot["k"],
        pilot["v"],
        _dequantize_hif4(q_merged).reshape_as(pilot["q"]),
        _dequantize_hif4(k_merged).reshape_as(pilot["k"]),
        pilot["v_reconstruction"],
        q_num_heads,
        kv_num_heads,
        head_dim,
    )
    parent_risk = float(qk_risk[group, window_index].item())
    candidate_risk = float(
        candidate_risks["qk_window_risk"][group, window_index].item()
    )
    fraction = max(
        0.0,
        min(1.0, (parent_risk - candidate_risk) / max(parent_risk, _REPAIR_ROUTER_EPS)),
    )
    _diagnostic_add("qk_pilot_cost_ns", elapsed)
    _diagnostic_add("qk_pilot_gain_numerator", max(parent_risk - candidate_risk, 0.0))
    _diagnostic_add("qk_pilot_gain_denominator", parent_risk)
    return elapsed, fraction


def _router_pilot_v_expert(
    pilot: dict[str, Any],
    state: dict[str, Any],
    kv_num_heads: int,
    head_dim: int,
    v_risk: torch.Tensor,
    block_index: int,
) -> tuple[int, float]:
    """Time one actual V H64 unit and return its observed fixed-P gain."""
    if block_index < 0 or block_index >= v_risk.numel():
        raise ValueError("V pilot route index out of range")
    blocks = torch.tensor([block_index], dtype=torch.long)
    value = _router_gather_value_blocks(pilot["v_flat"], blocks)
    compact = _router_slice_v_state(
        state["v_state"], blocks, kv_num_heads, head_dim
    )
    started = time.perf_counter_ns()
    candidate = _router_apply_v_expert(value, compact, 1)
    elapsed = max(time.perf_counter_ns() - started, 1)
    merged = _router_scatter_parameters(pilot["v_parent"], candidate, blocks)
    candidate_risks = _router_attention_risks(
        pilot["q"],
        pilot["k"],
        pilot["v"],
        pilot["q_reconstruction"],
        pilot["k_reconstruction"],
        _dequantize_hif4(merged).reshape_as(pilot["v"]),
        int(pilot["q"].shape[1]),
        kv_num_heads,
        head_dim,
    )
    parent_risk = float(v_risk.reshape(-1)[block_index].item())
    candidate_risk = float(
        candidate_risks["v_risk"].reshape(-1)[block_index].item()
    )
    fraction = max(
        0.0,
        min(1.0, (parent_risk - candidate_risk) / max(parent_risk, _REPAIR_ROUTER_EPS)),
    )
    _diagnostic_add("v_pilot_cost_ns", elapsed)
    _diagnostic_add("v_pilot_gain_numerator", max(parent_risk - candidate_risk, 0.0))
    _diagnostic_add("v_pilot_gain_denominator", parent_risk)
    return elapsed, fraction


def _router_apply_v_mask_to_pilot(
    pilot: dict[str, Any],
    state: dict[str, Any],
    v_mask: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    record_rescue: bool = True,
) -> dict[str, torch.Tensor]:
    """Materialize the frozen V23 route before measuring any Q/K residual."""
    expected = (kv_num_heads, head_dim // _BLOCK)
    if tuple(v_mask.shape) != expected or v_mask.dtype != torch.bool:
        raise ValueError("invalid frozen V route")
    if record_rescue:
        _diagnostic_add("qk_rescue_v_baseline_builds", 1)
    selected_units = int(v_mask.sum().item())
    if selected_units == 0:
        return pilot["v_parent"]
    block_indices = torch.nonzero(
        v_mask.reshape(-1), as_tuple=False
    ).flatten().cpu()
    packed_value = _router_gather_value_blocks(pilot["v_flat"], block_indices)
    packed_parent = _router_gather_parameters(pilot["v_parent"], block_indices)
    compact_state = _router_slice_v_state(
        state["v_state"], block_indices, kv_num_heads, head_dim
    )
    candidate = _router_apply_v_expert(
        packed_value, compact_state, selected_units
    )
    if record_rescue:
        _diagnostic_add(
            "qk_rescue_v_baseline_effective_units",
            _router_changed_units(packed_parent, candidate, 1),
        )
    return _router_scatter_parameters(
        pilot["v_parent"], candidate, block_indices
    )


def _router_exact_attention_loss(
    pilot: dict[str, Any],
    q_params: dict[str, torch.Tensor],
    k_params: dict[str, torch.Tensor],
    v_params: dict[str, torch.Tensor],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-KV normalized exact-output loss and centered-logit SSE."""
    heads_per_kv = q_num_heads // kv_num_heads
    q_reference = pilot["q"]
    k_reference = pilot["k"]
    v_reference = pilot["v"]
    sequence = int(q_reference.shape[0])
    expected_q = (sequence, q_num_heads, head_dim)
    expected_kv = (sequence, kv_num_heads, head_dim)
    if (
        tuple(q_reference.shape) != expected_q
        or tuple(k_reference.shape) != expected_kv
        or tuple(v_reference.shape) != expected_kv
    ):
        raise ValueError("invalid exact-output pilot shape")

    q_candidate = _dequantize_hif4(q_params).reshape(expected_q)
    k_candidate = _dequantize_hif4(k_params).reshape(expected_kv)
    v_candidate = _dequantize_hif4(v_params).reshape(expected_kv)
    tensors = (
        q_reference,
        k_reference,
        v_reference,
        q_candidate,
        k_candidate,
        v_candidate,
    )
    if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
        raise ValueError("non-finite exact-output pilot")

    q_reference = q_reference.reshape(
        sequence, kv_num_heads, heads_per_kv, head_dim
    ).permute(1, 2, 0, 3)
    q_candidate = q_candidate.reshape(
        sequence, kv_num_heads, heads_per_kv, head_dim
    ).permute(1, 2, 0, 3)
    k_reference = k_reference.permute(1, 0, 2)
    k_candidate = k_candidate.permute(1, 0, 2)
    v_reference = v_reference.permute(1, 0, 2)
    v_candidate = v_candidate.permute(1, 0, 2)
    inverse_temperature = float(head_dim) ** -0.5
    reference_logits = torch.matmul(
        q_reference, k_reference.unsqueeze(1).transpose(-1, -2)
    ) * inverse_temperature
    candidate_logits = torch.matmul(
        q_candidate, k_candidate.unsqueeze(1).transpose(-1, -2)
    ) * inverse_temperature
    reference_output = torch.matmul(
        torch.softmax(reference_logits, dim=-1), v_reference.unsqueeze(1)
    )
    candidate_output = torch.matmul(
        torch.softmax(candidate_logits, dim=-1), v_candidate.unsqueeze(1)
    )
    logit_error = candidate_logits - reference_logits
    centered_logit_error = logit_error - logit_error.mean(
        dim=-1, keepdim=True
    )
    numerator = (candidate_output - reference_output).square().sum(
        dim=(1, 2, 3)
    )
    denominator = reference_output.square().sum(dim=(1, 2, 3)).clamp_min(
        _REPAIR_ROUTER_EPS
    )
    loss = numerator / denominator
    centered_sse = centered_logit_error.square().sum(dim=(1, 2, 3))
    if not (
        bool(torch.isfinite(loss).all().item())
        and bool(torch.isfinite(centered_sse).all().item())
    ):
        raise ValueError("non-finite exact-output loss")
    return loss.to(torch.float64).cpu(), centered_sse.to(torch.float64).cpu()


def _router_qk_unit_candidate(
    pilot: dict[str, Any],
    state: dict[str, Any],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    flat_index: int,
    record_rescue: bool = True,
) -> dict[str, Any]:
    """Run one unchanged paired expert and return compact legal fields."""
    heads_per_kv = q_num_heads // kv_num_heads
    window = _qk_wide_window(head_dim)
    windows_per_head = head_dim // window
    total_units = kv_num_heads * windows_per_head
    if flat_index < 0 or flat_index >= total_units:
        raise ValueError("Q/K rescue index out of range")
    group = flat_index // windows_per_head
    window_index = flat_index % windows_per_head
    group_mask = torch.zeros(kv_num_heads, windows_per_head, dtype=torch.bool)
    group_mask[group, window_index] = True
    q_mask = group_mask.repeat_interleave(heads_per_kv, dim=0)
    q_units, q_blocks = _router_unit_indices(q_mask, head_dim, window)
    k_units, k_blocks = _router_unit_indices(group_mask, head_dim, window)
    q_value = _router_gather_value_blocks(pilot["q_flat"], q_blocks)
    k_value = _router_gather_value_blocks(pilot["k_flat"], k_blocks)
    q_compact = _router_slice_qk_state(
        state["q_state"], q_units, q_blocks, q_num_heads, head_dim, window
    )
    k_compact = _router_slice_qk_state(
        state["k_state"], k_units, k_blocks, kv_num_heads, head_dim, window
    )
    q_parent = _router_gather_parameters(pilot["q_parent"], q_blocks)
    k_parent = _router_gather_parameters(pilot["k_parent"], k_blocks)
    started = time.perf_counter_ns()
    q_candidate = _router_apply_q_expert(
        q_value, q_compact, int(q_units.numel()), window
    )
    k_candidate = _router_apply_k_expert(
        k_value, k_compact, int(k_units.numel()), window
    )
    elapsed = max(time.perf_counter_ns() - started, 1)
    changed = (
        _router_changed_units(q_parent, q_candidate, window // _BLOCK)
        + _router_changed_units(k_parent, k_candidate, window // _BLOCK)
    ) > 0
    if record_rescue:
        _diagnostic_add("qk_rescue_candidate_evaluations", 1)
        _diagnostic_add("qk_rescue_candidate_cost_ns", elapsed)
    return {
        "group": group,
        "q_blocks": q_blocks,
        "k_blocks": k_blocks,
        "q_candidate": q_candidate,
        "k_candidate": k_candidate,
        "elapsed_ns": elapsed,
        "changed": changed,
    }


def _router_merge_qk_unit(
    q_params: dict[str, torch.Tensor],
    k_params: dict[str, torch.Tensor],
    candidate: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Overlay one non-overlapping paired unit on the current cascade state."""
    return (
        _router_scatter_parameters(
            q_params, candidate["q_candidate"], candidate["q_blocks"]
        ),
        _router_scatter_parameters(
            k_params, candidate["k_candidate"], candidate["k_blocks"]
        ),
    )


def _format_action_family(action: int) -> str:
    """Return the legal hierarchy field changed by one static action id."""
    if 1 <= action <= _FORMAT_MANT_ACTIONS:
        return "mant"
    if action <= _FORMAT_MANT_ACTIONS + _FORMAT_LV3_ACTIONS:
        return "lv3"
    if action <= _FORMAT_ACTIONS_PER_BLOCK:
        return "lv2"
    raise ValueError("format action out of range")


def _format_error_components(
    target: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> tuple[float, float]:
    """Measure clipping and in-range resolution SSE for one or more H64 blocks."""
    blocks = _reshape_blocks(target)
    reconstruction = (
        params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_factor"].to(torch.float32)
    )
    maximum = (
        params["scale_lv3"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_factor"].to(torch.float32)
        * 1.75
    ).expand_as(blocks)
    clipped = blocks.abs() > maximum
    clipping = torch.where(
        clipped, (blocks.abs() - maximum).clamp_min(0.0).square(), 0.0
    ).sum()
    resolution = torch.where(
        clipped, 0.0, (reconstruction - blocks).square()
    ).sum()
    values = (float(clipping.item()), float(resolution.item()))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite format error components")
    return values


def _format_apply_v_action(
    target: torch.Tensor,
    parent: dict[str, torch.Tensor],
    action: int,
) -> tuple[dict[str, torch.Tensor], bool, int, tuple[float, float, float, float]]:
    """Apply one resolution-only adjacent move to a packed single H64 block."""
    if (
        target.shape[-1] != _BLOCK
        or action <= 0
        or action > _FORMAT_ACTIONS_PER_BLOCK
    ):
        raise ValueError("invalid format action input")
    blocks = _reshape_blocks(target)
    if int(blocks.shape[-4]) != 1:
        raise ValueError("format action must receive exactly one H64 block")
    candidate = {key: parent[key].clone() for key in _HIF4_PARAMETER_FIELDS}
    protected_values = 0

    if action <= _FORMAT_MANT_ACTIONS:
        local = (action - 1) // 2
        direction = -1 if (action - 1) % 2 == 0 else 1
        lv2_index = local // 8
        within_lv2 = local % 8
        lv3_index = within_lv2 // 4
        mant_index = within_lv2 % 4
        target_slice = blocks[..., 0, lv2_index, lv3_index, mant_index]
        effective = (
            parent["scale_factor"][..., 0, 0, 0, 0].to(torch.float32)
            * parent["scale_lv2"][..., 0, lv2_index, 0, 0].to(torch.float32)
            * parent["scale_lv3"][..., 0, lv2_index, lv3_index, 0].to(
                torch.float32
            )
        )
        old_code = torch.round(
            parent["sign"][..., 0, lv2_index, lv3_index, mant_index].to(
                torch.float32
            )
            * parent["mant"][..., 0, lv2_index, lv3_index, mant_index].to(
                torch.float32
            )
            * 4.0
        ).clamp(-7.0, 7.0)
        new_code = (old_code + float(direction)).clamp(-7.0, 7.0)
        in_range = target_slice.abs() <= effective * 1.75
        take = in_range & (new_code != old_code)
        protected_values = int((~in_range).sum().item())
        candidate["mant"][..., 0, lv2_index, lv3_index, mant_index] = torch.where(
            take,
            (new_code.abs() * 0.25).to(candidate["mant"].dtype),
            parent["mant"][..., 0, lv2_index, lv3_index, mant_index],
        )
        candidate["sign"][..., 0, lv2_index, lv3_index, mant_index] = torch.where(
            take,
            torch.where(new_code == 0.0, 0.0, torch.sign(new_code)).to(
                candidate["sign"].dtype
            ),
            parent["sign"][..., 0, lv2_index, lv3_index, mant_index],
        )
    elif action <= _FORMAT_MANT_ACTIONS + _FORMAT_LV3_ACTIONS:
        group = action - _FORMAT_MANT_ACTIONS - 1
        lv2_index = group // 2
        lv3_index = group % 2
        target_slice = blocks[..., 0, lv2_index, lv3_index, :]
        old_level = parent["scale_lv3"][..., 0, lv2_index, lv3_index, 0].to(
            torch.float32
        )
        legal_level = (old_level == 1.0) | (old_level == 2.0)
        new_level = torch.where(old_level == 1.0, 2.0, 1.0)
        base_scale = (
            parent["scale_factor"][..., 0, 0, 0, 0].to(torch.float32)
            * parent["scale_lv2"][..., 0, lv2_index, 0, 0].to(torch.float32)
        )
        old_effective = base_scale * old_level
        new_effective = base_scale * new_level
        safe = (
            legal_level
            & (target_slice.abs() <= old_effective.unsqueeze(-1) * 1.75).all(dim=-1)
            & (target_slice.abs() <= new_effective.unsqueeze(-1) * 1.75).all(dim=-1)
        )
        protected_values = int((~safe).sum().item()) * 4
        new_mant = (
            torch.round(target_slice.abs() / new_effective.unsqueeze(-1) * 4.0)
            .clamp(0.0, 7.0)
            * 0.25
        )
        new_sign = torch.where(new_mant == 0.0, 0.0, torch.sign(target_slice))
        candidate["scale_lv3"][..., 0, lv2_index, lv3_index, 0] = torch.where(
            safe,
            new_level.to(candidate["scale_lv3"].dtype),
            parent["scale_lv3"][..., 0, lv2_index, lv3_index, 0],
        )
        candidate["mant"][..., 0, lv2_index, lv3_index, :] = torch.where(
            safe.unsqueeze(-1),
            new_mant.to(candidate["mant"].dtype),
            parent["mant"][..., 0, lv2_index, lv3_index, :],
        )
        candidate["sign"][..., 0, lv2_index, lv3_index, :] = torch.where(
            safe.unsqueeze(-1),
            new_sign.to(candidate["sign"].dtype),
            parent["sign"][..., 0, lv2_index, lv3_index, :],
        )
    else:
        lv2_index = action - _FORMAT_MANT_ACTIONS - _FORMAT_LV3_ACTIONS - 1
        target_slice = blocks[..., 0, lv2_index, :, :]
        old_level = parent["scale_lv2"][..., 0, lv2_index, 0, 0].to(
            torch.float32
        )
        legal_level = (old_level == 1.0) | (old_level == 2.0)
        new_level = torch.where(old_level == 1.0, 2.0, 1.0)
        outer = parent["scale_factor"][..., 0, 0, 0, 0].to(torch.float32)
        lv3 = parent["scale_lv3"][..., 0, lv2_index, :, 0].to(torch.float32)
        old_effective = outer.unsqueeze(-1) * old_level.unsqueeze(-1) * lv3
        new_effective = outer.unsqueeze(-1) * new_level.unsqueeze(-1) * lv3
        safe = (
            legal_level
            & (target_slice.abs() <= old_effective.unsqueeze(-1) * 1.75).all(
                dim=(-2, -1)
            )
            & (target_slice.abs() <= new_effective.unsqueeze(-1) * 1.75).all(
                dim=(-2, -1)
            )
        )
        protected_values = int((~safe).sum().item()) * 8
        new_mant = (
            torch.round(target_slice.abs() / new_effective.unsqueeze(-1) * 4.0)
            .clamp(0.0, 7.0)
            * 0.25
        )
        new_sign = torch.where(new_mant == 0.0, 0.0, torch.sign(target_slice))
        candidate["scale_lv2"][..., 0, lv2_index, 0, 0] = torch.where(
            safe,
            new_level.to(candidate["scale_lv2"].dtype),
            parent["scale_lv2"][..., 0, lv2_index, 0, 0],
        )
        candidate["mant"][..., 0, lv2_index, :, :] = torch.where(
            safe[..., None, None],
            new_mant.to(candidate["mant"].dtype),
            parent["mant"][..., 0, lv2_index, :, :],
        )
        candidate["sign"][..., 0, lv2_index, :, :] = torch.where(
            safe[..., None, None],
            new_sign.to(candidate["sign"].dtype),
            parent["sign"][..., 0, lv2_index, :, :],
        )

    before_clipping, before_resolution = _format_error_components(target, parent)
    after_clipping, after_resolution = _format_error_components(target, candidate)
    changed = _router_changed_units(parent, candidate, 1) > 0
    return candidate, changed, protected_values, (
        before_clipping,
        after_clipping,
        before_resolution,
        after_resolution,
    )


def _format_apply_qk_mask_to_pilot(
    pilot: dict[str, Any],
    state: dict[str, Any],
    qk_mask: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Materialize V37's published non-overlapping Q/K route without diagnostics."""
    q_params = pilot["q_parent"]
    k_params = pilot["k_parent"]
    for flat_index in torch.nonzero(
        qk_mask.reshape(-1), as_tuple=False
    ).flatten().tolist():
        unit = _router_qk_unit_candidate(
            pilot,
            state,
            q_num_heads,
            kv_num_heads,
            head_dim,
            int(flat_index),
            record_rescue=False,
        )
        q_params, k_params = _router_merge_qk_unit(q_params, k_params, unit)
    return q_params, k_params


def _format_v_output_context(
    pilot: dict[str, Any],
    q_params: dict[str, torch.Tensor],
    k_params: dict[str, torch.Tensor],
    v_params: dict[str, torch.Tensor],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Cache exact fixed-QK V Jacobian, output residual, and group normalizer."""
    sequence = int(pilot["q"].shape[0])
    heads_per_kv = q_num_heads // kv_num_heads
    q_candidate = _dequantize_hif4(q_params).reshape(
        sequence, kv_num_heads, heads_per_kv, head_dim
    ).permute(1, 2, 0, 3)
    k_candidate = _dequantize_hif4(k_params).reshape(
        sequence, kv_num_heads, head_dim
    ).permute(1, 0, 2)
    v_candidate = _dequantize_hif4(v_params).reshape(
        sequence, kv_num_heads, head_dim
    ).permute(1, 0, 2)
    q_reference = pilot["q"].reshape(
        sequence, kv_num_heads, heads_per_kv, head_dim
    ).permute(1, 2, 0, 3)
    k_reference = pilot["k"].permute(1, 0, 2)
    v_reference = pilot["v"].permute(1, 0, 2)
    inverse_temperature = float(head_dim) ** -0.5
    probability = torch.softmax(
        torch.matmul(q_candidate, k_candidate.unsqueeze(1).transpose(-1, -2))
        * inverse_temperature,
        dim=-1,
    )
    reference_probability = torch.softmax(
        torch.matmul(q_reference, k_reference.unsqueeze(1).transpose(-1, -2))
        * inverse_temperature,
        dim=-1,
    )
    reference_output = torch.matmul(
        reference_probability, v_reference.unsqueeze(1)
    )
    candidate_output = torch.matmul(probability, v_candidate.unsqueeze(1))
    residual = candidate_output - reference_output
    denominator = reference_output.square().sum(dim=(1, 2, 3)).clamp_min(
        _REPAIR_ROUTER_EPS
    )
    if not all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in (probability, residual, denominator)
    ):
        raise ValueError("non-finite format V-output context")
    return {
        "probability": probability,
        "residual": residual,
        "denominator": denominator,
        "base_loss": residual.square().sum(dim=(1, 2, 3)) / denominator,
    }


def _format_v_action_gain(
    context: dict[str, torch.Tensor],
    parent_block: dict[str, torch.Tensor],
    candidate_block: dict[str, torch.Tensor],
    block_index: int,
    head_dim: int,
) -> float:
    """Return the exact normalized output gain of one fixed-QK V-only move."""
    blocks_per_head = head_dim // _BLOCK
    group = block_index // blocks_per_head
    local_block = block_index % blocks_per_head
    delta = (
        _dequantize_hif4(candidate_block)
        - _dequantize_hif4(parent_block)
    ).reshape(-1, _BLOCK)
    delta_output = torch.einsum(
        "hts,sf->htf", context["probability"][group], delta
    )
    start = local_block * _BLOCK
    residual = context["residual"][group, :, :, start : start + _BLOCK]
    increment = (
        2.0 * (residual * delta_output).sum() + delta_output.square().sum()
    ) / context["denominator"][group]
    gain = -float(increment.item())
    if not math.isfinite(gain):
        raise ValueError("non-finite format action gain")
    return gain


def _format_scatter_one_block(
    parent: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    block_index: int,
) -> dict[str, torch.Tensor]:
    return _router_scatter_parameters(
        parent, candidate, torch.tensor([block_index], dtype=torch.long)
    )


def _build_heldout_v_resolution_projection(
    calib_qkv_list: list,
    fit_pilots: list[dict[str, Any]],
    state: dict[str, Any],
    qk_mask: torch.Tensor,
    v_mask: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Fit legal adjacent V moves, then publish only two-fold exact-safe actions."""
    total_blocks = kv_num_heads * (head_dim // _BLOCK)
    empty = torch.zeros(total_blocks, dtype=torch.int16)
    started = time.perf_counter_ns()
    _diagnostic_add("format_calibration_calls", 1)
    call_index = int(_HIF4_DIAGNOSTICS["format_calibration_calls"])
    if _FORMAT_PROJECTION_MODE == "off":
        _diagnostic_add("format_disabled_calls", 1)
        _diagnostic_add("format_calibration_time_ns", time.perf_counter_ns() - started)
        return empty
    if (
        len(fit_pilots) != _FORMAT_FIT_FOLDS
        or len(calib_qkv_list) < _FORMAT_FIT_FOLDS + _FORMAT_HELDOUT_FOLDS
    ):
        _diagnostic_add("format_no_heldout_fallbacks", 1)
        _diagnostic_add("format_calibration_fallbacks", 1)
        _diagnostic_add("format_calibration_time_ns", time.perf_counter_ns() - started)
        return empty
    try:
        heldout_pilots = [
            _router_pilot_state(
                calib_qkv_list[_FORMAT_FIT_FOLDS + index],
                state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            for index in range(_FORMAT_HELDOUT_FOLDS)
        ]
        pilots = fit_pilots + heldout_pilots
        _diagnostic_add("format_fit_folds", len(fit_pilots))
        _diagnostic_add("format_heldout_folds", len(heldout_pilots))
        q_params: list[dict[str, torch.Tensor]] = []
        k_params: list[dict[str, torch.Tensor]] = []
        v_params: list[dict[str, torch.Tensor]] = []
        contexts: list[dict[str, torch.Tensor]] = []
        for pilot in pilots:
            routed_q, routed_k = _format_apply_qk_mask_to_pilot(
                pilot,
                state,
                qk_mask,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            routed_v = _router_apply_v_mask_to_pilot(
                pilot,
                state,
                v_mask,
                kv_num_heads,
                head_dim,
                record_rescue=False,
            )
            q_params.append(routed_q)
            k_params.append(routed_k)
            v_params.append(routed_v)
            contexts.append(
                _format_v_output_context(
                    pilot,
                    routed_q,
                    routed_k,
                    routed_v,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                )
            )
        for fit_index in range(_FORMAT_FIT_FOLDS):
            _diagnostic_set(
                f"format_case{call_index}_fit_fold{fit_index}_base_loss",
                float(contexts[fit_index]["base_loss"].mean().item()),
            )

        _diagnostic_add("format_candidate_blocks", total_blocks)
        _diagnostic_add(
            "format_action_total", total_blocks * _FORMAT_ACTIONS_PER_BLOCK
        )
        _diagnostic_add(
            "format_resolution_candidate_total",
            total_blocks * _FORMAT_ACTIONS_PER_BLOCK,
        )
        fit_records: list[tuple[float, float, int, int]] = []
        for block_index in range(total_blocks):
            best_key: tuple[float, float, int] | None = None
            best_record: tuple[float, float, int, int] | None = None
            best_stats: list[tuple[float, float, float, float]] = []
            block_tensor = torch.tensor([block_index], dtype=torch.long)
            fit_values = [
                _router_gather_value_blocks(pilot["v_flat"], block_tensor)
                for pilot in fit_pilots
            ]
            fit_parents = [
                _router_gather_parameters(v_params[index], block_tensor)
                for index in range(_FORMAT_FIT_FOLDS)
            ]
            for action in range(1, _FORMAT_ACTIONS_PER_BLOCK + 1):
                gains: list[float] = []
                all_changed = True
                clipping_fixed = True
                action_stats: list[tuple[float, float, float, float]] = []
                for fold_index in range(_FORMAT_FIT_FOLDS):
                    candidate, changed, _, stats = _format_apply_v_action(
                        fit_values[fold_index], fit_parents[fold_index], action
                    )
                    gain = _format_v_action_gain(
                        contexts[fold_index],
                        fit_parents[fold_index],
                        candidate,
                        block_index,
                        head_dim,
                    )
                    group = block_index // (head_dim // _BLOCK)
                    parent_loss = float(
                        contexts[fold_index]["base_loss"][group].item()
                    )
                    candidate_loss = parent_loss - gain
                    _diagnostic_add("format_fit_comparisons", 1)
                    _diagnostic_add(
                        f"format_fit_fold{fold_index}_parent_loss", parent_loss
                    )
                    _diagnostic_add(
                        f"format_fit_fold{fold_index}_candidate_loss",
                        candidate_loss,
                    )
                    _diagnostic_add("format_fit_gain", gain)
                    _diagnostic_add("format_resolution_fit_output_gain", gain)
                    if gain > 0.0:
                        _diagnostic_add("format_fit_improved", 1)
                    elif gain == 0.0:
                        _diagnostic_add("format_fit_equal", 1)
                    else:
                        _diagnostic_add("format_fit_worse", 1)
                    tolerance = 1.0e-12 * max(1.0, abs(stats[0]))
                    clipping_fixed &= abs(stats[1] - stats[0]) <= tolerance
                    all_changed &= changed
                    gains.append(gain)
                    action_stats.append(stats)
                if not clipping_fixed:
                    _diagnostic_add("format_clipping_invariant_rejected", 1)
                if all_changed and clipping_fixed and min(gains) > 0.0:
                    minimum_gain = min(gains)
                    summed_gain = sum(gains)
                    key = (minimum_gain, summed_gain, -action)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_record = (
                            minimum_gain,
                            summed_gain,
                            block_index,
                            action,
                        )
                        best_stats = action_stats
            if best_record is None:
                _diagnostic_add("format_fit_rejected_blocks", 1)
                continue
            fit_records.append(best_record)
            _diagnostic_add("format_fit_proposed_blocks", 1)
            family = _format_action_family(best_record[3])
            _diagnostic_add(f"format_{family}_proposed", 1)
            for stats in best_stats:
                _diagnostic_add("format_candidate_clipping_before", stats[0])
                _diagnostic_add("format_candidate_clipping_after", stats[1])
                _diagnostic_add("format_candidate_resolution_before", stats[2])
                _diagnostic_add("format_candidate_resolution_after", stats[3])

        current_v = [
            v_params[_FORMAT_FIT_FOLDS + index]
            for index in range(_FORMAT_HELDOUT_FOLDS)
        ]
        current_losses = []
        for heldout_index, pilot in enumerate(heldout_pilots):
            loss, _ = _router_exact_attention_loss(
                pilot,
                q_params[_FORMAT_FIT_FOLDS + heldout_index],
                k_params[_FORMAT_FIT_FOLDS + heldout_index],
                current_v[heldout_index],
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            current_losses.append(loss)
        initial_heldout_losses = [loss.clone() for loss in current_losses]
        local_accepted = 0
        local_rejected = 0
        for minimum_gain, summed_gain, block_index, action in sorted(
            fit_records, key=lambda item: (-item[0], -item[1], item[2], item[3])
        ):
            del minimum_gain, summed_gain
            _diagnostic_add("format_heldout_candidates", 1)
            group = block_index // (head_dim // _BLOCK)
            block_tensor = torch.tensor([block_index], dtype=torch.long)
            trial_v: list[dict[str, torch.Tensor]] = []
            trial_losses: list[torch.Tensor] = []
            trial_stats: list[tuple[float, float, float, float]] = []
            safe = True
            for heldout_index, pilot in enumerate(heldout_pilots):
                target_block = _router_gather_value_blocks(
                    pilot["v_flat"], block_tensor
                )
                parent_block = _router_gather_parameters(
                    current_v[heldout_index], block_tensor
                )
                candidate, changed, _, stats = _format_apply_v_action(
                    target_block, parent_block, action
                )
                proposed_v = _format_scatter_one_block(
                    current_v[heldout_index], candidate, block_index
                )
                loss, _ = _router_exact_attention_loss(
                    pilot,
                    q_params[_FORMAT_FIT_FOLDS + heldout_index],
                    k_params[_FORMAT_FIT_FOLDS + heldout_index],
                    proposed_v,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                )
                parent_loss = float(current_losses[heldout_index][group].item())
                candidate_loss = float(loss[group].item())
                gain = parent_loss - candidate_loss
                tolerance = 1.0e-12 * max(1.0, abs(stats[0]))
                clipping_fixed = abs(stats[1] - stats[0]) <= tolerance
                safe &= changed and clipping_fixed and gain > 0.0
                _diagnostic_add("format_heldout_comparisons", 1)
                _diagnostic_add(
                    f"format_heldout_fold{heldout_index}_parent_loss", parent_loss
                )
                _diagnostic_add(
                    f"format_heldout_fold{heldout_index}_candidate_loss",
                    candidate_loss,
                )
                _diagnostic_add("format_heldout_gain", gain)
                _diagnostic_add("format_resolution_heldout_output_gain", gain)
                if gain > 0.0:
                    _diagnostic_add("format_heldout_improved", 1)
                elif gain == 0.0:
                    _diagnostic_add("format_heldout_equal", 1)
                else:
                    _diagnostic_add("format_heldout_worse", 1)
                trial_v.append(proposed_v)
                trial_losses.append(loss)
                trial_stats.append(stats)
                if not clipping_fixed:
                    _diagnostic_add("format_clipping_invariant_rejected", 1)
            if safe:
                empty[block_index] = action
                current_v = trial_v
                current_losses = trial_losses
                _diagnostic_add("format_crossfold_accepted", 1)
                local_accepted += 1
                _diagnostic_add(
                    f"format_{_format_action_family(action)}_accepted", 1
                )
                for stats in trial_stats:
                    _diagnostic_add("format_candidate_clipping_before", stats[0])
                    _diagnostic_add("format_candidate_clipping_after", stats[1])
                    _diagnostic_add("format_candidate_resolution_before", stats[2])
                    _diagnostic_add("format_candidate_resolution_after", stats[3])
            else:
                _diagnostic_add("format_crossfold_rejected", 1)
                local_rejected += 1
        for heldout_index in range(_FORMAT_HELDOUT_FOLDS):
            initial_value = float(initial_heldout_losses[heldout_index].mean().item())
            final_value = float(current_losses[heldout_index].mean().item())
            _diagnostic_set(
                f"format_case{call_index}_heldout_fold{heldout_index}_base_loss",
                initial_value,
            )
            _diagnostic_set(
                f"format_case{call_index}_heldout_fold{heldout_index}_final_loss",
                final_value,
            )
            _diagnostic_set(
                f"format_case{call_index}_heldout_fold{heldout_index}_gain",
                initial_value - final_value,
            )
        _diagnostic_set(
            f"format_case{call_index}_accepted_candidates", local_accepted
        )
        _diagnostic_set(
            f"format_case{call_index}_rejected_candidates", local_rejected
        )
        _diagnostic_add("format_static_selected_blocks", int((empty > 0).sum().item()))
        _diagnostic_add("format_calibration_successes", 1)
        _diagnostic_add("format_calibration_time_ns", time.perf_counter_ns() - started)
        return empty
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        RuntimeError,
        OverflowError,
    ):
        _diagnostic_add("format_gate_fallbacks", 1)
        _diagnostic_add("format_calibration_fallbacks", 1)
        _diagnostic_add("format_calibration_time_ns", time.perf_counter_ns() - started)
        return empty


def _build_heldout_qk_rescue(
    calib_qkv_list: list,
    fit_pilots: list[dict[str, Any]],
    state: dict[str, Any],
    v_mask: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, int]:
    """Fit a residual route, then publish only cross-heldout-safe Q/K units."""
    window = _qk_wide_window(head_dim)
    windows_per_head = head_dim // window
    empty = torch.zeros(kv_num_heads, windows_per_head, dtype=torch.bool)
    if (
        len(fit_pilots) != _QK_RESCUE_FIT_FOLDS
        or len(calib_qkv_list)
        < _QK_RESCUE_FIT_FOLDS + _QK_RESCUE_HELDOUT_FOLDS
    ):
        _diagnostic_add("qk_rescue_no_heldout_fallbacks", 1)
        return empty, 0

    try:
        heldout_pilots = [
            _router_pilot_state(
                calib_qkv_list[_QK_RESCUE_FIT_FOLDS + index],
                state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            for index in range(_QK_RESCUE_HELDOUT_FOLDS)
        ]
        _diagnostic_add("qk_rescue_fit_folds", len(fit_pilots))
        _diagnostic_add("qk_rescue_heldout_folds", len(heldout_pilots))
        all_pilots = fit_pilots + heldout_pilots
        v_params = [
            _router_apply_v_mask_to_pilot(
                pilot, state, v_mask, kv_num_heads, head_dim
            )
            for pilot in all_pilots
        ]
        base_losses: list[torch.Tensor] = []
        base_centered: list[torch.Tensor] = []
        for pilot, routed_v in zip(all_pilots, v_params):
            loss, centered = _router_exact_attention_loss(
                pilot,
                pilot["q_parent"],
                pilot["k_parent"],
                routed_v,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            base_losses.append(loss)
            base_centered.append(centered)

        total_units = int(empty.numel())
        _diagnostic_add("qk_rescue_candidate_total", total_units)
        fit_records: list[tuple[int, float, int]] = []
        for flat_index in range(total_units):
            group = flat_index // windows_per_head
            maximum_cost = 0
            minimum_gain = float("inf")
            every_fold_changed = True
            every_fold_improved = True
            for fold_index, pilot in enumerate(fit_pilots):
                candidate = _router_qk_unit_candidate(
                    pilot,
                    state,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    flat_index,
                )
                q_trial, k_trial = _router_merge_qk_unit(
                    pilot["q_parent"], pilot["k_parent"], candidate
                )
                trial_loss, trial_centered = _router_exact_attention_loss(
                    pilot,
                    q_trial,
                    k_trial,
                    v_params[fold_index],
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                )
                parent_value = float(base_losses[fold_index][group].item())
                candidate_value = float(trial_loss[group].item())
                gain = parent_value - candidate_value
                _diagnostic_add("qk_rescue_fit_comparisons", 1)
                _diagnostic_add("qk_rescue_parent_normalized_loss", parent_value)
                _diagnostic_add(
                    "qk_rescue_candidate_normalized_loss", candidate_value
                )
                _diagnostic_add(
                    "qk_rescue_centered_logit_parent_sse",
                    float(base_centered[fold_index][group].item()),
                )
                _diagnostic_add(
                    "qk_rescue_centered_logit_candidate_sse",
                    float(trial_centered[group].item()),
                )
                _diagnostic_add("qk_rescue_fit_gain", gain)
                if gain > 0.0:
                    _diagnostic_add("qk_rescue_fit_improved", 1)
                elif gain == 0.0:
                    _diagnostic_add("qk_rescue_fit_equal", 1)
                else:
                    _diagnostic_add("qk_rescue_fit_worse", 1)
                every_fold_changed &= bool(candidate["changed"])
                every_fold_improved &= gain > 0.0
                maximum_cost = max(maximum_cost, int(candidate["elapsed_ns"]))
                minimum_gain = min(minimum_gain, gain)
            if every_fold_changed:
                _diagnostic_add("qk_rescue_changed_candidates", 1)
            if every_fold_changed and every_fold_improved:
                _diagnostic_add("qk_rescue_fit_eligible", 1)
                fit_records.append((maximum_cost, -minimum_gain, flat_index))
            else:
                _diagnostic_add("qk_rescue_fit_rejected", 1)

        current_q = [pilot["q_parent"] for pilot in heldout_pilots]
        current_k = [pilot["k_parent"] for pilot in heldout_pilots]
        current_losses = [
            base_losses[_QK_RESCUE_FIT_FOLDS + index]
            for index in range(_QK_RESCUE_HELDOUT_FOLDS)
        ]
        current_centered = [
            base_centered[_QK_RESCUE_FIT_FOLDS + index]
            for index in range(_QK_RESCUE_HELDOUT_FOLDS)
        ]
        selected_cost = 0
        for fit_cost, _, flat_index in sorted(fit_records):
            group = flat_index // windows_per_head
            trial_q: list[dict[str, torch.Tensor]] = []
            trial_k: list[dict[str, torch.Tensor]] = []
            trial_losses: list[torch.Tensor] = []
            trial_centered_values: list[torch.Tensor] = []
            safe = True
            for heldout_index, pilot in enumerate(heldout_pilots):
                candidate = _router_qk_unit_candidate(
                    pilot,
                    state,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    flat_index,
                )
                q_trial, k_trial = _router_merge_qk_unit(
                    current_q[heldout_index],
                    current_k[heldout_index],
                    candidate,
                )
                loss, centered = _router_exact_attention_loss(
                    pilot,
                    q_trial,
                    k_trial,
                    v_params[_QK_RESCUE_FIT_FOLDS + heldout_index],
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                )
                parent_value = float(current_losses[heldout_index][group].item())
                candidate_value = float(loss[group].item())
                gain = parent_value - candidate_value
                _diagnostic_add("qk_rescue_heldout_comparisons", 1)
                _diagnostic_add("qk_rescue_parent_normalized_loss", parent_value)
                _diagnostic_add(
                    "qk_rescue_candidate_normalized_loss", candidate_value
                )
                _diagnostic_add(
                    "qk_rescue_centered_logit_parent_sse",
                    float(current_centered[heldout_index][group].item()),
                )
                _diagnostic_add(
                    "qk_rescue_centered_logit_candidate_sse",
                    float(centered[group].item()),
                )
                _diagnostic_add("qk_rescue_heldout_gain", gain)
                if gain > 0.0:
                    _diagnostic_add("qk_rescue_heldout_improved", 1)
                elif gain == 0.0:
                    _diagnostic_add("qk_rescue_heldout_equal", 1)
                else:
                    _diagnostic_add("qk_rescue_heldout_worse", 1)
                safe &= bool(candidate["changed"]) and gain > 0.0
                trial_q.append(q_trial)
                trial_k.append(k_trial)
                trial_losses.append(loss)
                trial_centered_values.append(centered)
            if safe:
                empty.reshape(-1)[flat_index] = True
                current_q = trial_q
                current_k = trial_k
                current_losses = trial_losses
                current_centered = trial_centered_values
                selected_cost += fit_cost
                _diagnostic_add("qk_rescue_crossfold_accepted", 1)
                _diagnostic_add("qk_rescue_selected_cost_ns", fit_cost)
            else:
                _diagnostic_add("qk_rescue_crossfold_rejected", 1)
        return empty, selected_cost
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        RuntimeError,
        OverflowError,
    ):
        _diagnostic_add("qk_rescue_gate_fallbacks", 1)
        # An exception can occur after an earlier cascade item was tentatively
        # accepted.  Discard the whole rescue mask so the failure is genuinely
        # fail-closed; the independently frozen V route is returned by caller.
        return torch.zeros_like(empty), 0


def _solve_repair_router(
    qk_benefit: torch.Tensor,
    v_benefit: torch.Tensor,
    qk_cost: int,
    v_cost: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Deterministic benefit-density packing under one measured-cost budget."""
    if qk_cost <= 0 or v_cost <= 0:
        raise ValueError("non-positive repair-router expert cost")
    qk_mask = torch.zeros_like(qk_benefit, dtype=torch.bool)
    v_mask = torch.zeros_like(v_benefit, dtype=torch.bool)
    items: list[tuple[float, float, int, int, int]] = []
    for kind, benefit, cost in (
        (0, qk_benefit, qk_cost),
        (1, v_benefit, v_cost),
    ):
        for index, value in enumerate(benefit.reshape(-1).tolist()):
            numeric = float(value)
            if math.isfinite(numeric) and numeric > 0.0:
                items.append((-numeric / float(cost), -numeric, kind, index, cost))
    full_cost = qk_cost * qk_benefit.numel() + v_cost * v_benefit.numel()
    budget = int(math.floor(_REPAIR_ROUTER_BUDGET_FRACTION * full_cost))
    # A repair unit is indivisible.  For a tiny shape, the nominal fractional
    # budget can be smaller than the single best unit; admit exactly that best
    # density item so the router does not perversely choose a cheaper low-value
    # expert or an empty route solely because of granularity.
    if items:
        budget = max(budget, sorted(items)[0][4])
    selected_cost = 0
    for _, _, kind, index, cost in sorted(items):
        if selected_cost + cost > budget:
            continue
        if kind == 0:
            qk_mask.reshape(-1)[index] = True
        else:
            v_mask.reshape(-1)[index] = True
        selected_cost += cost
    return qk_mask, v_mask, budget, selected_cost


def _attention_build_repair_router(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Attach static block routes learned only from bounded calibration work."""
    calibration_started = time.perf_counter_ns()
    _diagnostic_add("router_calibration_calls", 1)
    try:
        if head_dim % _BLOCK or len(calib_qkv_list) < _REPAIR_ROUTER_PILOT_FOLDS:
            raise ValueError("unsupported repair-router shape")
        heads_per_kv = q_num_heads // kv_num_heads
        window = _qk_wide_window(head_dim)
        blocks_per_window = window // _BLOCK
        windows_per_head = head_dim // window
        if windows_per_head > 1:
            _diagnostic_add("router_multiwindow_calibrations", 1)
        pilots = [
            _router_pilot_state(
                calib_qkv_list[index],
                state,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            for index in range(_REPAIR_ROUTER_PILOT_FOLDS)
        ]
        qk_fold_risks: list[torch.Tensor] = []
        v_fold_risks: list[torch.Tensor] = []
        for pilot in pilots:
            fold_risks = pilot["risks"]
            qk_fold_risks.append(fold_risks["qk_window_risk"])
            v_fold_risks.append(fold_risks["v_risk"])
        qk_risk = torch.stack(qk_fold_risks).mean(dim=0)
        v_risk = torch.stack(v_fold_risks).mean(dim=0)
        qk_route_index = int(qk_risk.reshape(-1).argmax().item())
        v_route_index = int(v_risk.reshape(-1).argmax().item())
        qk_measurements = [
            _router_pilot_qk_expert(
                pilot,
                state,
                q_num_heads,
                kv_num_heads,
                head_dim,
                fold_risk,
                qk_route_index,
            )
            for pilot, fold_risk in zip(pilots, qk_fold_risks)
        ]
        v_measurements = [
            _router_pilot_v_expert(
                pilot,
                state,
                kv_num_heads,
                head_dim,
                fold_risk,
                v_route_index,
            )
            for pilot, fold_risk in zip(pilots, v_fold_risks)
        ]
        qk_cost = max(value[0] for value in qk_measurements)
        v_cost = max(value[0] for value in v_measurements)
        # A block family is admitted only when the same static route direction
        # improves both disjoint calibration samples.  The conservative minimum
        # is then extrapolated to all blocks in that family.
        qk_fraction = min(value[1] for value in qk_measurements)
        v_fraction = min(value[1] for value in v_measurements)
        _diagnostic_add(
            "qk_pilot_positive_folds",
            sum(int(value[1] > 0.0) for value in qk_measurements),
        )
        _diagnostic_add(
            "v_pilot_positive_folds",
            sum(int(value[1] > 0.0) for value in v_measurements),
        )
        qk_benefit = qk_risk * qk_fraction
        v_benefit = v_risk * v_fraction
        legacy_qk_mask, v_mask, budget, legacy_selected_cost = _solve_repair_router(
            qk_benefit, v_benefit, qk_cost, v_cost
        )
        # Stage one is byte-for-byte V23 routing.  Freeze only its measured V
        # decision; Q/K no longer competes with V for this budget.  Stage-two
        # failure returns an empty Q/K mask while leaving this V mask intact.
        _diagnostic_add(
            "legacy_qk_selected_total", int(legacy_qk_mask.sum().item())
        )
        _diagnostic_add("frozen_v_selected_total", int(v_mask.sum().item()))
        _diagnostic_add("legacy_router_selected_cost_ns", legacy_selected_cost)
        frozen_v_cost = int(v_cost * int(v_mask.sum().item()))
        _diagnostic_add("frozen_v_selected_cost_ns", frozen_v_cost)
        qk_mask, rescue_selected_cost = _build_heldout_qk_rescue(
            calib_qkv_list,
            pilots,
            state,
            v_mask,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        format_actions = _build_heldout_v_resolution_projection(
            calib_qkv_list,
            pilots,
            state,
            qk_mask,
            v_mask,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        selected_cost = frozen_v_cost + rescue_selected_cost
        candidate_total = int(qk_mask.numel() + v_mask.numel())
        selected_total = int(qk_mask.sum().item() + v_mask.sum().item())
        _diagnostic_add("router_searches", 1)
        _diagnostic_add("router_pilot_folds", len(pilots))
        _diagnostic_add("router_candidate_total", candidate_total)
        _diagnostic_add("router_selected_total", selected_total)
        _diagnostic_add("qk_candidate_total", int(qk_mask.numel()))
        _diagnostic_add("qk_selected_total", int(qk_mask.sum().item()))
        _diagnostic_add("v_candidate_total", int(v_mask.numel()))
        _diagnostic_add("v_selected_total", int(v_mask.sum().item()))
        _diagnostic_add("router_budget_cost_ns", budget)
        full_cost = int(qk_cost * qk_mask.numel() + v_cost * v_mask.numel())
        predicted_benefit = float(qk_benefit.sum().item() + v_benefit.sum().item())
        selected_predicted_benefit = float(
            qk_benefit[qk_mask].sum().item() + v_benefit[v_mask].sum().item()
        )
        _diagnostic_add("router_full_cost_ns", full_cost)
        _diagnostic_add("router_selected_cost_ns", selected_cost)
        _diagnostic_add("router_predicted_benefit", predicted_benefit)
        _diagnostic_add(
            "router_selected_predicted_benefit", selected_predicted_benefit
        )
        for pilot in pilots:
            risks = pilot["risks"]
            _diagnostic_add("centered_logit_q_energy", risks["q_energy"])
            _diagnostic_add("centered_logit_k_energy", risks["k_energy"])
            _diagnostic_add("centered_logit_cross_energy", risks["cross_energy"])
            _diagnostic_add("jacobian_q_output_risk", risks["q_output"])
            _diagnostic_add("jacobian_k_output_risk", risks["k_output"])
            _diagnostic_add("jacobian_cross_output_risk", risks["cross_output"])
            _diagnostic_add("jacobian_qk_joint_output_risk", risks["qk_joint_output"])
            _diagnostic_add("v_path_output_risk", risks["v_output"])
            _diagnostic_add(
                "probability_value_cross_risk", risks["probability_value_cross"]
            )

        selected_state = dict(state)
        common = {
            "repair_router_schema": _REPAIR_ROUTER_SCHEMA,
            "repair_router_budget_fraction": _REPAIR_ROUTER_BUDGET_FRACTION,
            "repair_router_qk_mask": qk_mask.cpu(),
            "repair_router_qk_window": window,
            "repair_router_kv_num_heads": kv_num_heads,
            "repair_router_heads_per_kv": heads_per_kv,
            "repair_router_qk_cost_ns": int(qk_cost),
            "repair_router_v_cost_ns": int(v_cost),
            "repair_router_budget_cost_ns": int(budget),
            "repair_router_selected_cost_ns": int(selected_cost),
            "repair_router_legacy_selected_cost_ns": int(legacy_selected_cost),
            "repair_router_frozen_v_cost_ns": int(frozen_v_cost),
            "repair_router_qk_rescue_cost_ns": int(rescue_selected_cost),
            "repair_router_legacy_qk_mask": legacy_qk_mask.cpu(),
            "repair_router_v_route_frozen": True,
            "repair_router_qk_gain_fraction": float(qk_fraction),
            "repair_router_v_gain_fraction": float(v_fraction),
            "repair_router_predicted_benefit": predicted_benefit,
            "repair_router_selected_predicted_benefit": selected_predicted_benefit,
            "repair_router_pilot_folds": len(pilots),
            "repair_router_pilot_tokens": sum(
                int(pilot["tokens"]) for pilot in pilots
            ),
        }
        for role in ("q_state", "k_state"):
            role_state = dict(state[role])
            role_state.update(common)
            selected_state[role] = role_state
        v_state = dict(state["v_state"])
        v_state.update(common)
        v_state["repair_router_v_mask"] = v_mask.cpu()
        if _FORMAT_PROJECTION_MODE != "off":
            v_state["format_projection_schema"] = _FORMAT_PROJECTION_SCHEMA
            v_state["format_projection_actions"] = format_actions.cpu()
        selected_state["v_state"] = v_state
        _diagnostic_add("router_calibration_successes", 1)
        if selected_total == 0:
            _diagnostic_add("router_empty_routes", 1)
        _diagnostic_add(
            "router_calibration_time_ns",
            time.perf_counter_ns() - calibration_started,
        )
        return selected_state
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        RuntimeError,
        OverflowError,
    ):
        # Absence of the schema deliberately dispatches the byte-equivalent V22
        # full repair chain.  A calibration failure therefore loses speed, not
        # the known high-precision safety path.
        _diagnostic_add("router_calibration_fallbacks", 1)
        _diagnostic_add(
            "router_calibration_time_ns",
            time.perf_counter_ns() - calibration_started,
        )
        return state


def _attention_full_repair_from_value(
    value: torch.Tensor,
    importance: torch.Tensor,
    state: dict[str, Any],
    role: str,
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Preserve V22's complete chain for calibration/router safety fallback."""
    if role == "k":
        value = _attention_k_gauge_route(value, num_heads, head_dim)
    value = torch.nan_to_num(value)
    if role == "v":
        parent, alternate = _quantize_sensitive_branches(
            value, importance, _ATTENTION_GAIN
        )
        parent = _attention_v_hierarchy_chain_select(
            value, parent, alternate, state, num_heads, head_dim
        )
        parent = _attention_v_chain_repair(
            value, parent, state, num_heads, head_dim
        )
        parent = _attention_v_global_dc_repair(
            value, parent, state, num_heads, head_dim
        )
        return _attention_v_lowrank_repair(
            value, parent, state, num_heads, head_dim
        )
    parent = _quantize_sensitive(value, importance, _ATTENTION_GAIN)
    parent = _attention_hessian_repair(
        value, parent, state, num_heads, head_dim
    )
    if role == "k":
        parent = _attention_k_fixed_scale_code_descent(
            value, parent, state, num_heads, head_dim
        )
        parent = _attention_k_two_coordinate_code_solve(
            value, parent, state, num_heads, head_dim
        )
        return _attention_k_breakpoint_overlay(
            value, parent, state, num_heads, head_dim
        )
    if role == "q":
        return _attention_q_fixed_scale_code_descent(
            value, parent, state, num_heads, head_dim
        )
    return parent


def _attention_routed_qk_repair(
    value: torch.Tensor,
    importance: torch.Tensor,
    state: dict[str, Any],
    role: str,
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Run deep Q/K repair only on statically selected paired windows."""
    mask_value = state.get("repair_router_qk_mask")
    window = state.get("repair_router_qk_window")
    kv_num_heads = state.get("repair_router_kv_num_heads")
    heads_per_kv = state.get("repair_router_heads_per_kv")
    if (
        type(mask_value) is not torch.Tensor
        or type(window) is not int
        or type(kv_num_heads) is not int
        or type(heads_per_kv) is not int
        or window <= 0
        or window % _BLOCK
        or head_dim % window
        or tuple(mask_value.shape) != (kv_num_heads, head_dim // window)
        or heads_per_kv <= 0
        or (role == "q" and num_heads != kv_num_heads * heads_per_kv)
        or (role == "k" and num_heads != kv_num_heads)
    ):
        _diagnostic_add("router_dynamic_shape_fallbacks", 1)
        return _attention_full_repair_from_value(
            value, importance, state, role, num_heads, head_dim
        )
    group_mask = mask_value.to(dtype=torch.bool, device="cpu")
    role_mask = (
        group_mask.repeat_interleave(heads_per_kv, dim=0)
        if role == "q"
        else group_mask
    )
    _diagnostic_add("qk_dynamic_candidate_units", int(role_mask.numel()))
    selected_units = int(role_mask.sum().item())
    _diagnostic_add("qk_dynamic_selected_units", selected_units)
    _diagnostic_add(f"{role}_expert_candidate_units", int(role_mask.numel()))
    _diagnostic_add(f"{role}_expert_selected_units", selected_units)
    if role == "k" and head_dim > window:
        _diagnostic_add("k_window_local_selected_units", selected_units)

    if role == "k":
        base_value = _attention_k_base_gauge_route(value, num_heads, head_dim)
    else:
        base_value = value
    base_value = torch.nan_to_num(base_value)
    parent = _quantize_hif4(base_value)
    if selected_units == 0:
        return parent

    unit_indices, block_indices = _router_unit_indices(
        role_mask, head_dim, window
    )
    compact_state = _router_slice_qk_state(
        state,
        unit_indices,
        block_indices,
        num_heads,
        head_dim,
        window,
    )
    packed_value = _router_gather_value_blocks(value, block_indices)
    packed_parent = _router_gather_parameters(parent, block_indices)
    started = time.perf_counter_ns()
    if role == "q":
        candidate = _router_apply_q_expert(
            packed_value,
            compact_state,
            selected_units,
            window,
        )
        _diagnostic_add("q_expert_time_ns", time.perf_counter_ns() - started)
    else:
        candidate = _router_apply_k_expert(
            packed_value,
            compact_state,
            selected_units,
            window,
        )
        _diagnostic_add("k_expert_time_ns", time.perf_counter_ns() - started)
    effective = _router_changed_units(
        packed_parent, candidate, window // _BLOCK
    )
    _diagnostic_add("qk_dynamic_effective_units", effective)
    _diagnostic_add(f"{role}_expert_effective_units", effective)
    return _router_scatter_parameters(parent, candidate, block_indices)


def _attention_apply_v_resolution_projection(
    value: torch.Tensor,
    parent: dict[str, torch.Tensor],
    state: dict[str, Any],
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Apply the held-out static resolution policy after the complete V37 route."""
    if _FORMAT_PROJECTION_MODE == "off":
        return parent
    _diagnostic_add("format_dynamic_calls", 1)
    action_value = state.get("format_projection_actions")
    if (
        state.get("format_projection_schema") != _FORMAT_PROJECTION_SCHEMA
        or type(action_value) is not torch.Tensor
    ):
        _diagnostic_add("format_dynamic_missing_state_fallbacks", 1)
        return parent
    total_blocks = kv_num_heads * (head_dim // _BLOCK)
    if (
        action_value.ndim != 1
        or action_value.numel() != total_blocks
        or int(value.shape[-1]) != kv_num_heads * head_dim
    ):
        _diagnostic_add("format_dynamic_shape_fallbacks", 1)
        return parent
    actions = action_value.to(dtype=torch.int64, device="cpu")
    if bool(((actions < 0) | (actions > _FORMAT_ACTIONS_PER_BLOCK)).any().item()):
        _diagnostic_add("format_dynamic_shape_fallbacks", 1)
        return parent
    _diagnostic_add("format_dynamic_candidate_blocks", total_blocks)
    selected_indices = torch.nonzero(actions > 0, as_tuple=False).flatten().tolist()
    _diagnostic_add("format_dynamic_selected_blocks", len(selected_indices))
    if not selected_indices:
        return parent
    try:
        selected = {key: parent[key].clone() for key in _HIF4_PARAMETER_FIELDS}
        effective_blocks = 0
        for block_index in selected_indices:
            index_tensor = torch.tensor([block_index], dtype=torch.long)
            target_block = _router_gather_value_blocks(value, index_tensor)
            parent_block = _router_gather_parameters(selected, index_tensor)
            action = int(actions[block_index].item())
            candidate, changed, protected, stats = _format_apply_v_action(
                target_block, parent_block, action
            )
            _diagnostic_add("format_dynamic_protected_values", protected)
            _diagnostic_add("format_dynamic_clipping_before", stats[0])
            _diagnostic_add("format_dynamic_clipping_after", stats[1])
            _diagnostic_add("format_dynamic_resolution_before", stats[2])
            _diagnostic_add("format_dynamic_resolution_after", stats[3])
            tolerance = 1.0e-12 * max(1.0, abs(stats[0]))
            if not changed or abs(stats[1] - stats[0]) > tolerance:
                continue
            for key in _HIF4_PARAMETER_FIELDS:
                field = selected[key]
                axis = field.ndim - 4
                field.index_copy_(
                    axis,
                    index_tensor.to(device=field.device),
                    candidate[key].to(device=field.device, dtype=field.dtype),
                )
            effective_blocks += 1
            _diagnostic_add(
                f"format_dynamic_{_format_action_family(action)}_blocks", 1
            )
        _diagnostic_add("format_dynamic_effective_blocks", effective_blocks)
        return selected
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        RuntimeError,
        OverflowError,
    ):
        _diagnostic_add("format_dynamic_exceptions", 1)
        return parent


def _attention_routed_v_repair(
    value: torch.Tensor,
    importance: torch.Tensor,
    state: dict[str, Any],
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Run the complete V consumer stack on selected H64 blocks only."""
    mask_value = state.get("repair_router_v_mask")
    expected = (kv_num_heads, head_dim // _BLOCK)
    if type(mask_value) is not torch.Tensor or tuple(mask_value.shape) != expected:
        _diagnostic_add("router_dynamic_shape_fallbacks", 1)
        return _attention_full_repair_from_value(
            value, importance, state, "v", kv_num_heads, head_dim
        )
    mask = mask_value.to(dtype=torch.bool, device="cpu")
    _diagnostic_add("v_dynamic_candidate_units", int(mask.numel()))
    selected_units = int(mask.sum().item())
    _diagnostic_add("v_dynamic_selected_units", selected_units)
    value = torch.nan_to_num(value)
    parent = _quantize_hif4(value)
    if selected_units == 0:
        return _attention_apply_v_resolution_projection(
            value, parent, state, kv_num_heads, head_dim
        )
    block_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten().cpu()
    packed_value = _router_gather_value_blocks(value, block_indices)
    packed_parent = _router_gather_parameters(parent, block_indices)
    compact_state = _router_slice_v_state(
        state, block_indices, kv_num_heads, head_dim
    )
    started = time.perf_counter_ns()
    candidate = _router_apply_v_expert(
        packed_value,
        compact_state,
        selected_units,
    )
    _diagnostic_add("v_expert_time_ns", time.perf_counter_ns() - started)
    effective = _router_changed_units(packed_parent, candidate, 1)
    _diagnostic_add("v_dynamic_effective_units", effective)
    routed = _router_scatter_parameters(parent, candidate, block_indices)
    return _attention_apply_v_resolution_projection(
        value, routed, state, kv_num_heads, head_dim
    )


def _attention_dynamic(
    quant: torch.Tensor,
    scale: torch.Tensor,
    state: Any,
    role: str,
    num_heads: int | None = None,
    head_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    if role == "q":
        _diagnostic_add("q_dynamic_calls", 1)
    elif role == "k":
        _diagnostic_add("k_dynamic_calls", 1)
    elif role == "v":
        _diagnostic_add("v_dynamic_calls", 1)
    value = dequantize_nvfp4(quant, scale).to(torch.float32)
    channels = int(value.shape[-1])
    if not isinstance(state, dict) or state.get("fallback_to_parent", False):
        _diagnostic_add("router_dynamic_state_fallbacks", 1)
        return _quantize_hif4(torch.nan_to_num(value))
    importance = state.get("importance")
    if type(importance) is not torch.Tensor or importance.numel() != channels:
        _diagnostic_add("router_dynamic_state_fallbacks", 1)
        return _quantize_hif4(torch.nan_to_num(value))
    importance = importance.to(device=value.device, dtype=torch.float32)
    if not bool(torch.isfinite(importance).all().item()):
        _diagnostic_add("router_dynamic_state_fallbacks", 1)
        return _quantize_hif4(torch.nan_to_num(value))
    if role in ("q", "k"):
        smooth = state.get("smooth_scale")
        if type(smooth) is not torch.Tensor or smooth.numel() != channels:
            _diagnostic_add("router_dynamic_state_fallbacks", 1)
            return _quantize_hif4(torch.nan_to_num(value))
        smooth = smooth.to(device=value.device, dtype=torch.float32)
        if not bool(torch.isfinite(smooth).all().item()) or not bool(
            (smooth > 0.0).all().item()
        ):
            _diagnostic_add("router_dynamic_state_fallbacks", 1)
            return _quantize_hif4(torch.nan_to_num(value))
        value = value / smooth if role == "q" else value * smooth
        transform = state.get("pair_transform")
        if transform is not None:
            expected_shape = (num_heads, head_dim, head_dim)
            if (
                type(transform) is not torch.Tensor
                or num_heads is None
                or head_dim is None
                or tuple(transform.shape) != expected_shape
                or not bool(torch.isfinite(transform).all().item())
            ):
                _diagnostic_add("router_dynamic_state_fallbacks", 1)
                return _quantize_hif4(torch.nan_to_num(value))
            value = _apply_head_transform(value, transform)
    if (
        role not in ("q", "k", "v")
        or num_heads is None
        or head_dim is None
        or num_heads <= 0
        or head_dim <= 0
        or head_dim % _BLOCK
        or channels != num_heads * head_dim
    ):
        _diagnostic_add("router_dynamic_shape_fallbacks", 1)
        return _quantize_sensitive(
            torch.nan_to_num(value), importance, _ATTENTION_GAIN
        )

    if state.get("repair_router_schema") != _REPAIR_ROUTER_SCHEMA:
        _diagnostic_add("router_dynamic_state_fallbacks", 1)
        return _attention_full_repair_from_value(
            value, importance, state, role, num_heads, head_dim
        )
    try:
        if role == "v":
            return _attention_routed_v_repair(
                value, importance, state, num_heads, head_dim
            )
        return _attention_routed_qk_repair(
            value, importance, state, role, num_heads, head_dim
        )
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        RuntimeError,
        OverflowError,
    ):
        _diagnostic_add("router_dynamic_exceptions", 1)
        return _attention_full_repair_from_value(
            value, importance, state, role, num_heads, head_dim
        )


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _attention_dynamic(
        q_quant, q_scale, q_state, "q", q_num_heads, head_dim
    )


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _attention_dynamic(
        k_quant, k_scale, k_state, "k", kv_num_heads, head_dim
    )


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _attention_dynamic(
        v_quant, v_scale, v_state, "v", kv_num_heads, head_dim
    )
