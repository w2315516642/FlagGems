# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Functional RMSNorm followed by symmetric, per-token INT8 quantization.

The BF16/FP16 rounding boundary before quantization is intentional. With a
residual, variance is computed from the FP32 sum, while the returned residual
is rounded to the input dtype, matching fused_add_rms_norm. No input is mutated.
"""

import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import tl_extra_shim


@triton.jit
def _round_even(x):
    lo = tl.floor(x)
    fraction = x - lo
    odd = lo - 2.0 * tl.floor(lo / 2.0) != 0.0
    return tl.where((fraction > 0.5) | ((fraction == 0.5) & odd), lo + 1.0, lo)


@libentry()
@triton.jit
def _rms_norm_dynamic_int8_kernel(
    X,
    R,
    W,
    Q,
    S,
    R_OUT,
    N: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    x = tl.load(X + row * N + col, mask, 0).to(tl.float32)
    w = tl.load(W + col, mask, 0).to(tl.float32)
    if HAS_RESIDUAL:
        r = tl.load(R + row * N + col, mask, 0).to(tl.float32)
        x = x + r
        tl.store(R_OUT + row * N + col, x, mask)
        variance = tl.sum(x * x / N, 0)
    else:
        variance = tl.sum(x * x, 0) / N
    normalized = x * (1.0 / tl.sqrt(variance + EPS))
    if not HAS_RESIDUAL:
        # Match the separate RMSNorm operator's pre-weight rounding.
        normalized = normalized.to(X.dtype.element_ty).to(tl.float32)
    y = (normalized * w).to(X.dtype.element_ty).to(tl.float32)
    amax = tl.max(tl.where(mask, tl.abs(y), 0.0), 0)
    inverse = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    q = tl.minimum(tl.maximum(_round_even(y * inverse), -127.0), 127.0)
    tl.store(Q + row * N + col, q.to(tl.int8), mask)
    tl.store(S + row, amax / 127.0)


def rms_norm_dynamic_int8(x, weight, eps=1e-6, residual=None):
    """Return ``(int8[M,N], fp32[M,1], residual_out or None)``.

    Only the last dimension is normalized. Zero rows have zero codes and zero
    scale. Inputs must be finite; nonfinite values have no INT8 representation.
    Large hidden dimensions are rejected instead of silently changing the
    reduction/rounding contract to a multi-pass approximation.
    """
    if x.ndim != 2 or not 0 < x.shape[1] <= 8192:
        raise ValueError("x must be [tokens, hidden] with 1 <= hidden <= 8192")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("x must be FP16 or BF16")
    if (
        weight.ndim != 1
        or weight.numel() != x.shape[1]
        or weight.dtype != x.dtype
        or weight.device != x.device
    ):
        raise ValueError("weight must match the input's hidden size, dtype and device")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if residual is not None and (
        residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
    ):
        raise ValueError("residual must match x's shape, dtype and device")
    x, weight = x.contiguous(), weight.contiguous()
    residual = residual.contiguous() if residual is not None else None
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    residual_out = torch.empty_like(x) if residual is not None else None
    if x.shape[0]:
        with torch_device_fn.device(x.device):
            _rms_norm_dynamic_int8_kernel[(x.shape[0],)](
                x,
                residual,
                weight,
                q,
                scale,
                residual_out,
                x.shape[1],
                eps,
                residual is not None,
                triton.next_power_of_2(x.shape[1]),
                num_warps=4 if x.shape[1] <= 2048 else 8,
            )
    return q, scale, residual_out


@triton.jit
def _add_rms_norm_quantize_kernel(
    X, R, W, Q, S, R_OUT, M, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) + tl.arange(0, 1)
    # Preserve the two-element reduction layout of the BF16 norm boundary;
    # INT8 stores must not select a different sum tree through wider vectors.
    col = tl.max_contiguous(tl.arange(0, BLOCK), 2)
    mask = (row[:, None] < M) & (col[None, :] < N)
    value = tl.load(X + row[:, None] * N + col[None, :], mask, 0).to(tl.float32)
    value += tl.load(R + row[:, None] * N + col[None, :], mask, 0).to(tl.float32)
    tl.store(R_OUT + row[:, None] * N + col[None, :], value, mask)
    weight = tl.load(W + col, col < N, 0).to(tl.float32)
    variance = tl.sum(value * value, 1) / N
    normalized = value * tl_extra_shim.rsqrt(variance[:, None] + EPS)
    rounded = (normalized * weight[None, :]).to(X.dtype.element_ty).to(tl.float32)
    amax = tl.max(tl.abs(rounded), 1)
    inverse = tl.where(amax != 0, 127.0 / amax, 0.0)
    codes = tl_extra_shim.nearbyint(rounded * inverse[:, None])
    codes = tl.minimum(tl.maximum(codes, -128.0), 127.0)
    tl.store(Q + row[:, None] * N + col[None, :], codes.to(tl.int8), col[None, :] < N)
    tl.store(S + row, amax / 127.0)


def add_rms_norm_quantize_int8(x, residual, weight, eps=1e-6, *, padded_rows=None):
    """Residual RMSNorm -> rounded BF16/FP16 -> per-token INT8, without mutation.

    Return padded codes/scales and the input-dtype residual. Callers must
    require this materialized residual boundary; consumers relying on a deferred
    higher-precision residual expression cannot use this new fused operation.
    Padded rows contain zero codes and scales. Existing Norm APIs stay unchanged.
    """
    if x.ndim != 2 or not 0 < x.shape[1] <= 8192:
        raise ValueError("expected [tokens, hidden], hidden in [1,8192]")
    if x.device.type != "cuda" or x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("expected an accelerator BF16/FP16 input")
    if (
        residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
    ):
        raise ValueError("residual must match input shape, dtype and device")
    if (
        weight.shape != (x.shape[1],)
        or weight.dtype != x.dtype
        or weight.device != x.device
    ):
        raise ValueError("weight must match input hidden size, dtype and device")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    rows = x.shape[0] if padded_rows is None else padded_rows
    if type(rows) is not int or rows < x.shape[0]:
        raise ValueError("padded_rows must be an integer at least tokens")
    x, residual, weight = x.contiguous(), residual.contiguous(), weight.contiguous()
    q = torch.empty((rows, x.shape[1]), dtype=torch.int8, device=x.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    residual_out = torch.empty_like(x)
    if rows:
        with torch_device_fn.device(x.device):
            _add_rms_norm_quantize_kernel[(rows,)](
                x,
                residual,
                weight,
                q,
                scale,
                residual_out,
                x.shape[0],
                x.shape[1],
                eps,
                triton.next_power_of_2(x.shape[1]),
                num_warps=4 if x.shape[1] <= 2048 else 8,
                num_stages=1,
            )
    return q, scale, residual_out
