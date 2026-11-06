# Copyright (c) 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl
from flag_gems.utils import libentry, tl_extra_shim


@triton.jit
def _round_half_to_even(x):
    """Match torch.round without relying on a vendor-specific libdevice."""
    floor = tl.floor(x)
    fraction = x - floor
    is_odd = tl.abs(floor - 2.0 * tl.floor(floor / 2.0)) > 0.5
    round_up = (fraction > 0.5) | ((tl.abs(fraction - 0.5) < 1e-10) & is_odd)
    return tl.where(round_up, floor + 1.0, floor)


@libentry()
@triton.jit
def tagged_quant_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WARPS_TAG: tl.constexpr,
    ROUND_KIND: tl.constexpr,
):
    token_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    values = tl.load(
        input_ptr + token_idx * hidden_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    absmax = tl.max(tl.abs(values), axis=0)
    inv_scale = tl.where(absmax != 0.0, 127.0 / absmax, 0.0)
    if ROUND_KIND == 0:
        quantized = _round_half_to_even(values * inv_scale)
    else:
        quantized = tl_extra_shim.nearbyint(values * inv_scale)
    quantized = tl.minimum(tl.maximum(quantized, -128.0), 127.0)

    tl.store(
        output_ptr + token_idx * hidden_size + offsets,
        quantized.to(tl.int8),
        mask=mask,
    )
    tl.store(scale_ptr + token_idx, absmax / 127.0)


@triton.jit
def gate_pad_quant_kernel(X, Q, S, M, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    values = tl.load(X + row * K + cols, (row < M) & (cols < K), 0).to(tl.float32)
    amax = tl.max(tl.abs(values), 0)
    inv = tl.where(amax != 0.0, 127.0 / amax, 0.0)
    codes = tl.minimum(tl.maximum(tl_extra_shim.nearbyint(values * inv), -128.0), 127.0)
    tl.store(Q + row * K + cols, codes.to(tl.int8), cols < K)
    tl.store(S + row, amax / 127.0)
