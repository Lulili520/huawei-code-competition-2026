"""Alternating activation-code and output-aware weight fitting.

The six public functions in this file are the complete submission surface.  Linear
calibration fits blockwise weights to activations passed through the inference
codec; Weight-derived intermediates terminate inside Weight quantization.
"""

from __future__ import annotations

from typing import Any

import torch


_BLOCK = 64
_CROSSBLOCK_WINDOW = 128
_CROSSBLOCK_ROW_CHUNK = 1024
_TOP4_ROW_CHUNK = 1024
_TAIL_QUANTILE = 0.994
_TAIL_WEIGHT_CAP = 4.0
_TAIL_TEMP_LIMIT_BYTES = 256 * 1024 * 1024
_TAIL_GATE_ENABLED = True
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
_LINEAR_GAIN = 0.0025
_HESSIAN_GAIN = 0.0005
_HESSIAN_DAMPING = 0.003
_TOP4_GAIN = 0.03
_ATTENTION_GAIN = 0.05
_ATTENTION_ALPHA = 0.75
_SMOOTH_MIN = 2.0 ** -8
_SMOOTH_MAX = 2.0 ** 8
_OUTPUT_AWARE_DAMPING = 0.01
_OUTPUT_AWARE_MAX_ROWS = 2048
_ALTERNATING_ROUNDS = 2


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
    return (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    ).flatten(-4, -1).to(torch.float32)


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
    return ((blocks - reconstructed).square() * weight).sum(
        dim=(-3, -2, -1), keepdim=True
    )


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
    parent = _quantize_hif4(tensor)
    candidate = _quantize_hif4(
        tensor, importance=importance, factors=_SENSITIVE_FACTORS
    )
    return _select_sensitive(tensor, parent, candidate, importance, min_gain)


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


def _block_gram(
    activations: list[torch.Tensor], precondition_scale: torch.Tensor
) -> torch.Tensor | None:
    channels = int(precondition_scale.numel())
    block_count = channels // _BLOCK
    gram = torch.zeros(block_count, _BLOCK, _BLOCK, dtype=torch.float32)
    rows = 0
    for activation in activations:
        value = (
            activation.to(torch.float32) / precondition_scale
        ).reshape(-1, block_count, _BLOCK)
        if not bool(torch.isfinite(value).all().item()):
            return None
        gram += torch.einsum("nbi,nbj->bij", value, value).cpu()
        rows += int(value.shape[0])
    if rows == 0:
        return None
    gram /= rows
    return gram if bool(torch.isfinite(gram).all().item()) else None


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


def _tail_crossblock_gram(
    activations: list[torch.Tensor],
    precondition_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build exact-q0.994 row-tail H128 Hessians from calibration A only."""
    if not _TAIL_GATE_ENABLED:
        return None
    channels = int(precondition_scale.numel())
    window_count = channels // _CROSSBLOCK_WINDOW
    if window_count == 0:
        return None

    rows = 0
    for activation in activations:
        if activation.shape[-1] != channels:
            return None
        rows += int(activation.numel() // channels)
    if rows == 0:
        return None

    # Current Z, abs(Z), sort values/indices and the weighted work tensor are
    # bounded together.  Crossing the cap disables the new gate atomically;
    # it never changes the exact quantile into a sampled approximation.
    estimated_bytes = rows * _CROSSBLOCK_WINDOW * (4 + 4 + 4 + 8 + 4)
    if estimated_bytes > _TAIL_TEMP_LIMIT_BYTES:
        return None

    scale = precondition_scale.to(dtype=torch.float32, device="cpu")
    tail_gram = torch.zeros(
        window_count,
        _CROSSBLOCK_WINDOW,
        _CROSSBLOCK_WINDOW,
        dtype=torch.float32,
    )
    valid = torch.zeros(window_count, dtype=torch.bool)
    quantile_position = (rows - 1) * _TAIL_QUANTILE
    lower_index = int(quantile_position)
    upper_index = min(lower_index + 1, rows - 1)
    upper_weight = float(quantile_position - lower_index)

    for window in range(window_count):
        start = window * _CROSSBLOCK_WINDOW
        stop = start + _CROSSBLOCK_WINDOW
        pieces: list[torch.Tensor] = []
        window_ok = True
        for activation in activations:
            value = activation.to(dtype=torch.float32, device="cpu").reshape(
                -1, channels
            )[:, start:stop]
            value = value / scale[start:stop]
            if not bool(torch.isfinite(value).all().item()):
                window_ok = False
                break
            pieces.append(value)
        if not window_ok:
            continue

        z = torch.cat(pieces, dim=0)
        absolute = z.abs()
        sorted_absolute, sort_indices = torch.sort(absolute, dim=0, stable=True)
        lower = sorted_absolute[lower_index]
        upper = sorted_absolute[upper_index]
        quantile = (lower + (upper - lower) * upper_weight).clamp_min(_MIN_SCALE)
        del sorted_absolute, sort_indices, pieces
        if not bool(torch.isfinite(quantile).all().item()):
            continue

        rho = (absolute / quantile).amax(dim=1)
        omega = rho.clamp(1.0, _TAIL_WEIGHT_CAP).square()
        denominator = omega.sum()
        if not bool(torch.isfinite(omega).all().item()) or not bool(
            torch.isfinite(denominator).item()
        ) or float(denominator.item()) <= 0.0:
            continue
        weighted = z * torch.sqrt(omega).unsqueeze(-1)
        candidate = weighted.transpose(0, 1).matmul(weighted) / denominator
        if bool(torch.isfinite(candidate).all().item()):
            tail_gram[window] = candidate
            valid[window] = True

    return tail_gram, valid


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

        target = weight_blocks[block_index]
        base_residual = target - base_quantized[block_index]
        current_proxy = torch.einsum(
            "oi,ij,oj->o", base_residual, hessian, base_residual
        )
        diag_desc = torch.argsort(
            diagonal, descending=True, stable=True
        ).to(device=weight_blocks.device)
        for order in (natural, reverse, diag_desc):
            ordered_reconstruction = _ordered_hessian_reconstruction(
                target, scale_blocks[block_index], inverse, order
            )
            candidate_mant = (
                torch.round(
                    ordered_reconstruction.abs()
                    / scale_blocks[block_index]
                    * 4.0
                ).clamp(0.0, 7.0)
                * 0.25
            )
            candidate_sign = torch.where(
                candidate_mant == 0.0,
                0.0,
                torch.sign(ordered_reconstruction),
            )
            candidate_quantized = (
                candidate_sign * candidate_mant * scale_blocks[block_index]
            )
            candidate_residual = target - candidate_quantized
            candidate_proxy = torch.einsum(
                "oi,ij,oj->o", candidate_residual, hessian, candidate_residual
            )
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
            current_proxy = torch.where(
                use_candidate, candidate_proxy, current_proxy
            )

    selected = dict(base)
    selected["sign"] = selected_sign.reshape_as(base["sign"])
    selected["mant"] = selected_mant.reshape_as(base["mant"])
    return selected


def _tail_pareto_accept(
    ordinary_accept: torch.Tensor,
    residual: torch.Tensor,
    group: torch.Tensor,
    best_pattern: torch.Tensor,
    pattern_bits: torch.Tensor,
    level_delta: torch.Tensor,
    group_scale: torch.Tensor,
    tail_hessian: torch.Tensor,
    tail_valid: torch.Tensor,
) -> torch.Tensor:
    """Require the ordinary winner to strictly improve its q-tail quadratic."""
    joint_accept = ordinary_accept.clone()
    eligible = ordinary_accept & tail_valid.unsqueeze(-1)
    indices = torch.nonzero(eligible, as_tuple=False)
    if int(indices.shape[0]) == 0:
        return joint_accept

    window_index = indices[:, 0]
    row_index = indices[:, 1]
    selected_residual = residual[window_index, row_index]
    selected_hessian = tail_hessian[window_index]
    projected = torch.einsum(
        "kij,kj->ki", selected_hessian, selected_residual
    )
    current_proxy = (selected_residual * projected).sum(dim=-1)

    selected_group = group[window_index, row_index]
    projected_group = torch.gather(projected, 1, selected_group)
    batch_index = torch.arange(
        int(indices.shape[0]), device=residual.device
    ).reshape(-1, 1, 1)
    group_hessian = selected_hessian[
        batch_index,
        selected_group.unsqueeze(-1),
        selected_group.unsqueeze(-2),
    ]
    selected_bits = pattern_bits[best_pattern[window_index, row_index]]
    selected_delta = (
        -selected_bits
        * level_delta[window_index, row_index]
        * group_scale[window_index, row_index]
        * 0.25
    )
    candidate_proxy = current_proxy + 2.0 * (
        selected_delta * projected_group
    ).sum(dim=-1) + torch.einsum(
        "ki,kij,kj->k", selected_delta, group_hessian, selected_delta
    )
    tail_accept = (
        torch.isfinite(current_proxy)
        & (current_proxy > 0.0)
        & torch.isfinite(candidate_proxy)
        & (candidate_proxy < current_proxy)
    )
    joint_accept[window_index, row_index] = tail_accept
    return joint_accept


def _bipartite_top4_repair(
    target_windows: torch.Tensor,
    scale_windows: torch.Tensor,
    current_sign: torch.Tensor,
    current_mant: torch.Tensor,
    current_proxy: torch.Tensor,
    diagonal: torch.Tensor,
    proxy_hessian: torch.Tensor,
    valid: torch.Tensor,
    tail_hessian: torch.Tensor,
    tail_valid: torch.Tensor,
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
        accept = _tail_pareto_accept(
            accept,
            residual,
            group,
            best_pattern,
            pattern_bits,
            level_delta,
            group_scale,
            tail_hessian,
            tail_valid,
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
    tail_gram: torch.Tensor | None = None,
    tail_gram_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Repair adjacent blocks, then apply the dual-Hessian Top-4 gate."""
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
    tail_hessian = torch.zeros_like(hessian)
    tail_valid = torch.zeros(window_count, dtype=torch.bool, device=weight.device)
    if (
        type(tail_gram) is torch.Tensor
        and type(tail_gram_valid) is torch.Tensor
        and tuple(tail_gram.shape)
        == (window_count, _CROSSBLOCK_WINDOW, _CROSSBLOCK_WINDOW)
        and int(tail_gram_valid.numel()) == window_count
    ):
        candidate_tail = tail_gram.to(device=weight.device, dtype=torch.float32)
        candidate_valid = tail_gram_valid.to(
            device=weight.device, dtype=torch.bool
        ).reshape(-1)
        candidate_diagonal = torch.diagonal(
            candidate_tail, dim1=-2, dim2=-1
        )
        candidate_valid &= torch.isfinite(candidate_tail).all(dim=(-2, -1))
        candidate_valid &= torch.isfinite(candidate_diagonal).all(dim=-1)
        candidate_valid &= (candidate_diagonal >= 0.0).all(dim=-1)
        tail_hessian = torch.where(
            candidate_valid.reshape(-1, 1, 1),
            candidate_tail,
            tail_hessian,
        )
        tail_valid = candidate_valid
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
            tail_hessian,
            tail_valid,
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


def _linear_state_valid(state: Any, channels: int) -> bool:
    if not isinstance(state, dict) or state.get("schema") != "alternating-joint-v2":
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


def _output_aware_linear_target(
    transformed_weight: torch.Tensor,
    activations: list[torch.Tensor],
    precondition: torch.Tensor,
    importance: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Fit blockwise weights against activations produced by the inference codec."""
    channels = int(transformed_weight.shape[1])
    if not activations:
        return transformed_weight, activations
    precondition = precondition.to(torch.float32)
    importance = importance.to(torch.float32)
    original_parts: list[torch.Tensor] = []
    quantized_parts: list[torch.Tensor] = []
    pseudo_activations: list[torch.Tensor] = []
    remaining = _OUTPUT_AWARE_MAX_ROWS
    try:
        for activation in activations:
            if activation.shape[-1] != channels or remaining <= 0:
                continue
            transformed = torch.nan_to_num(
                activation.to(torch.float32) / precondition
            )
            params = _quantize_sensitive(
                transformed, importance, _LINEAR_GAIN
            )
            quantized = _dequantize_hif4(params).reshape_as(transformed)
            if not bool(torch.isfinite(quantized).all().item()):
                return transformed_weight, activations
            flat_original = transformed.reshape(-1, channels)
            flat_quantized = quantized.reshape(-1, channels)
            take = min(remaining, int(flat_original.shape[0]))
            if take:
                original_parts.append(flat_original[:take])
                quantized_parts.append(flat_quantized[:take])
                pseudo_activations.append(
                    (flat_quantized[:take] * precondition).to(torch.float32)
                )
                remaining -= take
    except (TypeError, ValueError, RuntimeError):
        return transformed_weight, activations
    if not original_parts:
        return transformed_weight, activations

    original = torch.cat(original_parts, dim=0)
    quantized = torch.cat(quantized_parts, dim=0)
    fitted = transformed_weight.clone()
    identity = torch.eye(
        _BLOCK, device=transformed_weight.device, dtype=torch.float32
    )
    try:
        for start in range(0, channels, _BLOCK):
            stop = start + _BLOCK
            x = original[:, start:stop].to(transformed_weight.device)
            a = quantized[:, start:stop].to(transformed_weight.device)
            cross = x.transpose(0, 1) @ a
            gram = a.transpose(0, 1) @ a
            mean_diagonal = torch.diagonal(gram).mean().clamp_min(_MIN_SCALE)
            factor = torch.linalg.cholesky(
                gram + identity * (mean_diagonal * _OUTPUT_AWARE_DAMPING)
            )
            rhs = cross.transpose(0, 1) @ transformed_weight[:, start:stop].T
            block = torch.cholesky_solve(rhs, factor).T
            if not bool(torch.isfinite(block).all().item()):
                return transformed_weight, activations
            fitted[:, start:stop] = block
    except RuntimeError:
        return transformed_weight, activations
    return fitted, pseudo_activations


def _alternating_joint_target(
    transformed_weight: torch.Tensor,
    activations: list[torch.Tensor],
    precondition: torch.Tensor,
    initial_importance: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Alternate legal activation coding and output-aware weight fitting.

    The activation update uses diag(U^T U) as an output-error metric.  Returning
    the final metric makes calibration and the dynamic inference codec identical.
    """
    importance = initial_importance.to(torch.float32)
    fitted = transformed_weight
    pseudo = activations
    for round_index in range(_ALTERNATING_ROUNDS):
        candidate, candidate_pseudo = _output_aware_linear_target(
            transformed_weight, activations, precondition, importance
        )
        if not bool(torch.isfinite(candidate).all().item()):
            break
        fitted, pseudo = candidate, candidate_pseudo
        if round_index + 1 >= _ALTERNATING_ROUNDS:
            break
        output_energy = fitted.square().sum(dim=0)
        denominator = output_energy.mean()
        if not bool(torch.isfinite(denominator).item()) or float(denominator) <= 0:
            break
        candidate_importance = (output_energy / denominator).clamp(
            1.0 / 16.0, 16.0
        )
        if not bool(torch.isfinite(candidate_importance).all().item()):
            break
        importance = candidate_importance
    return fitted, pseudo, importance


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
    target_weight, proxy_activations, activation_importance = _alternating_joint_target(
        transformed_weight,
        activations if statistics_valid else [],
        precondition,
        activation_importance,
    )
    parent = _quantize_hif4(target_weight)
    gram = (
        _block_gram(proxy_activations, precondition)
        if statistics_valid
        else None
    )
    crossblock_gram = (
        _crossblock_gram(proxy_activations, precondition, gram)
        if statistics_valid and gram is not None
        else None
    )
    tail_crossblock = (
        _tail_crossblock_gram(proxy_activations, precondition)
        if statistics_valid and crossblock_gram is not None
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
                target_weight,
                importance=diagonal_importance,
                factors=_SENSITIVE_FACTORS,
            )
        else:
            base = parent
        weight_params = _multiorder_hessian(target_weight, base, gram)
    if crossblock_gram is not None:
        tail_gram = tail_crossblock[0] if tail_crossblock is not None else None
        tail_gram_valid = (
            tail_crossblock[1] if tail_crossblock is not None else None
        )
        weight_params = _crossblock_hessian_repair(
            target_weight,
            weight_params,
            crossblock_gram,
            tail_gram,
            tail_gram_valid,
        )

    return {
        "weight_params": weight_params,
        "activation_state": {
            "schema": "alternating-joint-v2",
            "precondition_scale": precondition.cpu(),
            "activation_importance": activation_importance.cpu(),
            "min_proxy_gain": _LINEAR_GAIN,
        },
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
    if not _linear_state_valid(activation_state, channels):
        return _quantize_hif4_v1(torch.nan_to_num(activation))
    precondition = activation_state["precondition_scale"].to(
        device=activation.device, dtype=torch.float32
    )
    importance = activation_state["activation_importance"].to(
        device=activation.device, dtype=torch.float32
    )
    transformed = torch.nan_to_num(activation / precondition)
    return _quantize_sensitive(transformed, importance, _LINEAR_GAIN)


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
                -1, kv_num_heads, heads_per_kv, block_count, _BLOCK
            )
            k_blocks = (k * kv_smooth).reshape(
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
    q_rows = 0
    kv_rows = 0
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
    except (KeyError, TypeError, ValueError, RuntimeError):
        return _attention_fallback_state(q_num_heads, kv_num_heads, head_dim)
    if q_rows == 0 or kv_rows == 0:
        return _attention_fallback_state(q_num_heads, kv_num_heads, head_dim)

    q_max.clamp_min_(2.0 ** -24)
    k_max.clamp_min_(2.0 ** -24)
    kv_smooth = (
        q_max.pow(_ATTENTION_ALPHA) / k_max.pow(1.0 - _ATTENTION_ALPHA)
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

    return {
        "q_state": {
            "smooth_scale": q_smooth.flatten().cpu(),
            "selected_alpha": _ATTENTION_ALPHA,
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
            "selected_alpha": _ATTENTION_ALPHA,
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
        },
    }


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
    hessian_diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1)
    inverse_diagonal = torch.diagonal(inverse, dim1=-2, dim2=-1)
    if (
        not bool(torch.isfinite(hessian).all().item())
        or not bool(torch.isfinite(inverse).all().item())
        or bool((hessian_diagonal < 0.0).any().item())
        or bool((inverse_diagonal <= 0.0).any().item())
    ):
        return parent

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
    if not (
        bool(torch.isfinite(scale).all().item())
        and bool((scale > 0.0).all().item())
        and bool(torch.isfinite(parent_quantized).all().item())
    ):
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


def _attention_dynamic(
    quant: torch.Tensor,
    scale: torch.Tensor,
    state: Any,
    role: str,
    num_heads: int | None = None,
    head_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(quant, scale).to(torch.float32)
    channels = int(value.shape[-1])
    if not isinstance(state, dict) or state.get("fallback_to_parent", False):
        return _quantize_hif4(torch.nan_to_num(value))
    importance = state.get("importance")
    if type(importance) is not torch.Tensor or importance.numel() != channels:
        return _quantize_hif4(torch.nan_to_num(value))
    importance = importance.to(device=value.device, dtype=torch.float32)
    if not bool(torch.isfinite(importance).all().item()):
        return _quantize_hif4(torch.nan_to_num(value))
    if role in ("q", "k"):
        smooth = state.get("smooth_scale")
        if type(smooth) is not torch.Tensor or smooth.numel() != channels:
            return _quantize_hif4(torch.nan_to_num(value))
        smooth = smooth.to(device=value.device, dtype=torch.float32)
        if not bool(torch.isfinite(smooth).all().item()) or not bool(
            (smooth > 0.0).all().item()
        ):
            return _quantize_hif4(torch.nan_to_num(value))
        value = value / smooth if role == "q" else value * smooth
    value = torch.nan_to_num(value)
    parent = _quantize_sensitive(value, importance, _ATTENTION_GAIN)
    if role not in ("q", "k") or num_heads is None or head_dim is None:
        return parent
    return _attention_hessian_repair(value, parent, state, num_heads, head_dim)


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
    del kv_num_heads, head_dim
    return _attention_dynamic(v_quant, v_scale, v_state, "v")
