# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""INT8 gate/up projection with per-token/channel scales and BF16 SwiGLU.

The weight is K-contiguous [K, 2*N], gate columns followed by up columns.
Preserve the unfused projection's BF16 rounding before FP32 activation.
This inference operator does not implement autograd.
"""

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _scaled_int8_swiglu_kernel(
    X,
    W,
    ScaleX,
    ScaleW,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    WARPS_TAG: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    group_width = GROUP_M * grid_n
    first_m = pid // group_width * GROUP_M
    group_size = tl.minimum(grid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_width) % group_size
    pid_n = (pid % group_width) // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    pair_axis = tl.arange(0, 2 * BLOCK_N)
    if INTERLEAVED:
        paired_cols = pid_n * BLOCK_N + pair_axis // 2
        weight_cols = pid_n * (2 * BLOCK_N) + pair_axis
    else:
        paired_cols = pid_n * BLOCK_N + pair_axis % BLOCK_N
        weight_cols = paired_cols + (pair_axis // BLOCK_N) * N
    reduction = tl.arange(0, BLOCK_K)
    # Pair output columns through pointers; no runtime weight rearrangement.
    x_ptrs = X + rows[:, None].to(tl.int64) * K + reduction[None, :]
    w_ptrs = W + reduction[:, None] + weight_cols[None, :].to(tl.int64) * K
    acc = tl.zeros((BLOCK_M, 2 * BLOCK_N), tl.int32)
    for block in range(tl.cdiv(K, BLOCK_K)):
        valid_k = reduction < K - block * BLOCK_K
        x = tl.load(x_ptrs, mask=(rows[:, None] < M) & valid_k[None, :], other=0)
        w = tl.load(w_ptrs, mask=valid_k[:, None] & (paired_cols[None, :] < N), other=0)
        acc = tl.dot(x, w, acc, out_dtype=tl.int32, allow_tf32=False)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    sx = tl.load(ScaleX + rows, mask=rows < M, other=0)
    sw = tl.load(ScaleW + weight_cols, mask=paired_cols < N, other=0)
    # Match scaled_mm's operation order and BF16 output boundary. Convert
    # before redistributing gate/up so layout conversion uses 16-bit values.
    projection = (acc.to(tl.float32) * sx[:, None]) * sw[None, :]
    paired = projection.to(tl.bfloat16)
    if INTERLEAVED:
        gate_up = tl.reshape(paired, (BLOCK_M, BLOCK_N, 2))
    else:
        gate_up = tl.trans(tl.reshape(paired, (BLOCK_M, 2, BLOCK_N)), 0, 2, 1)
    gate_bf16, up_bf16 = tl.split(gate_up)
    gate, up = gate_bf16.to(tl.float32), up_bf16.to(tl.float32)
    result = (gate * tl.sigmoid(gate)) * up
    tl.store(
        Y + rows[:, None].to(tl.int64) * N + cols[None, :],
        result,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def scaled_int8_swiglu(input, weight, scale_x, scale_w, *, interleaved=False):
    """Fuse [M,K] INT8 @ [K,2*N] INT8, scaling and SwiGLU into BF16 [M,N].

    Requires contiguous input, transposed-contiguous weight, FP32 scales
    [M,1] and [2*N] on the same accelerator. Tail tiles and empty M/N are
    supported. K is bounded to prevent INT32 overflow for arbitrary INT8
    values. Finite scales are an input precondition, as for scaled_mm.
    Tile choices are provisional; profitability is a caller-side policy.
    """
    if type(interleaved) is not bool:
        raise ValueError("interleaved must be a boolean")
    if input.ndim != 2 or weight.ndim != 2:
        raise ValueError("scaled_int8_swiglu requires two-dimensional matrices")
    if input.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError("scaled_int8_swiglu requires INT8 input and weight")
    tensors = (input, weight, scale_x, scale_w)
    if input.device.type == "cpu" or any(t.device != input.device for t in tensors):
        raise ValueError("scaled_int8_swiglu requires one accelerator device")
    m, k = input.shape
    if weight.shape[0] != k or weight.shape[1] % 2 or not 0 < k <= 131071:
        raise ValueError("incompatible gate/up dimensions or unsafe INT32 reduction")
    n = weight.shape[1] // 2
    if not input.is_contiguous() or not weight.t().is_contiguous():
        raise ValueError("requires contiguous input and K-contiguous weight")
    if (
        scale_x.dtype != torch.float32
        or scale_w.dtype != torch.float32
        or scale_x.shape != (m, 1)
        or scale_w.shape != (2 * n,)
        or not scale_x.is_contiguous()
        or not scale_w.is_contiguous()
    ):
        raise ValueError("requires contiguous FP32 per-token/channel scales")
    output = torch.empty((m, n), dtype=torch.bfloat16, device=input.device)
    if m == 0 or n == 0:
        return output
    block_m = min(64, max(16, triton.next_power_of_2(m)))
    block_n = 32
    block_k = 128 if interleaved and runtime.device.vendor_name == "metax" else 64
    options = {"pipeline": "basic"} if runtime.device.vendor_name == "metax" else {}
    # Large interleaved tiles benefit from more threads on MetaX C500.
    # Keep the previous configuration for small M and other backends/layouts.
    num_warps = (
        8 if interleaved and runtime.device.vendor_name == "metax" and m >= 256 else 4
    )
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    with torch_device_fn.device(input.device):
        _scaled_int8_swiglu_kernel[grid](
            input,
            weight,
            scale_x,
            scale_w,
            output,
            m,
            n,
            k,
            block_m,
            block_n,
            block_k,
            8,
            interleaved,
            num_warps,  # Explicit LibEntry cache key, not only a launch option.
            num_warps=num_warps,
            num_stages=2,
            **options,
        )
    return output


def scaled_int8_swiglu_interleaved(input, weight, scale_x, scale_w):
    """Same rounding and tiling, with g0,u0,g1,u1 weight/scale channels."""
    return scaled_int8_swiglu(input, weight, scale_x, scale_w, interleaved=True)


@libentry()
@triton.jit
def _interleaved_swiglu(X, Y, size, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + 2 * i, i < size, other=0).to(tl.float32)
    up = tl.load(X + 2 * i + 1, i < size, other=0).to(tl.float32)
    tl.store(Y + i, (gate * tl.sigmoid(gate)) * up, i < size)


def interleaved_swiglu(x):
    """No materialized deinterleave; preserve BF16 projection then FP32 activation."""
    if x.ndim < 1 or x.shape[-1] % 2 or not x.is_contiguous():
        raise ValueError("expected contiguous input with paired last dimension")
    if x.dtype != torch.bfloat16 or x.device.type == "cpu":
        raise ValueError("expected BF16 input on an accelerator")
    y = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
    if y.numel():
        with torch_device_fn.device(x.device):
            _interleaved_swiglu[(triton.cdiv(y.numel(), 256),)](x, y, y.numel(), 256)
    return y
