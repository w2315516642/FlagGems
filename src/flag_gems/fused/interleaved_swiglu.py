# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Inference SwiGLU for contiguous [..., g0,u0,g1,u1,...] projection output."""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


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
