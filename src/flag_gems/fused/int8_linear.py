# Copyright (c) 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Local INT8 projections with explicit per-token/per-channel scale contracts.

These are new quantized graph boundaries. Existing mm/scaled_mm registrations,
backend ordering and their unsupported-input behavior are not changed.
"""

import torch
import triton

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

from ._dynamic_int8 import gate_pad_quant_kernel, tagged_quant_kernel


def quantize_int8_tokens(x, *, padded_rows=None, num_warps=8):
    """Quantize contiguous [M,K]; optionally zero-fill a larger row capacity."""
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("expected a BF16/FP16 [tokens, features] tensor")
    if x.device.type == "cpu" or not x.is_contiguous() or x.shape[1] <= 0:
        raise ValueError("expected contiguous activations on an accelerator")
    m, k = x.shape
    rows = m if padded_rows is None else padded_rows
    if type(rows) is not int or rows < m or rows < 0:
        raise ValueError("padded_rows must be an integer at least M")
    q = torch.empty((rows, k), device=x.device, dtype=torch.int8)
    scales = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    if not rows:
        return q, scales
    with torch_device_fn.device(x.device):
        if padded_rows is not None:
            gate_pad_quant_kernel[(rows,)](
                x,
                q,
                scales,
                m,
                k,
                triton.next_power_of_2(k),
                num_warps=num_warps,
            )
        else:
            tagged_quant_kernel[(m,)](
                x,
                q,
                scales,
                k,
                triton.next_power_of_2(k),
                num_warps,
                1,
                num_warps=num_warps,
            )
    return q, scales


def _validate(q, weight, scale_x, scale_w):
    if q.ndim != 2 or weight.ndim != 2:
        raise ValueError("expected matrices [M,K] and [K,N]")
    m, k = q.shape
    n = weight.shape[1]
    if weight.shape[0] != k or not 0 < k <= 131071:
        raise ValueError("incompatible or unsafe INT32 contraction dimension")
    if q.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError("projection input and weight must be INT8")
    if q.device.type == "cpu" or any(
        t.device != q.device for t in (weight, scale_x, scale_w)
    ):
        raise ValueError("all projection tensors must share an accelerator")
    if not q.is_contiguous() or not weight.t().is_contiguous():
        raise ValueError("input and weight contraction dimensions must be contiguous")
    if (
        scale_x.shape != (m, 1)
        or scale_w.shape != (n,)
        or scale_x.dtype != torch.float32
        or scale_w.dtype != torch.float32
        or not scale_x.is_contiguous()
        or not scale_w.is_contiguous()
    ):
        raise ValueError("expected contiguous FP32 [M,1] and [N] scales")
    return m, n, k


def _qkv_config(m, k):
    if m <= 16:
        return (16, 64, 128, 2, 2 if k > 128 else 1, 4)
    if m <= 128:
        return (32, 32, 128, 2, 1, 4)
    return (64, 128, 128, 2, 1, 8)


def _down_config(m):
    if m <= 16:
        return (16, 64, 128, 2, 8)
    if m <= 32:
        return (16, 64, 128, 2, 4)
    if m <= 64:
        return (32, 32, 128, 2, 2)
    if m <= 128:
        return (16, 64, 128, 2, 1)
    if m <= 256:
        return (64, 128, 128, 2, 1)
    if m <= 512:
        return (64, 64, 128, 2, 1)
    return (64, 128, 128, 2, 1)


def scaled_int8_linear(q, weight, scale_x, scale_w, *, kind="qkv"):
    """INT32 dot, FP32 rescaling and one final BF16 output conversion.

    ``kind`` is a tuning hint for a new projection, not a backend selector.
    The implementations accept tail tiles and never match model/layer names.
    """
    m, n, k = _validate(q, weight, scale_x, scale_w)
    vendor = runtime.device.vendor_name
    if vendor not in ("metax", "iluvatar"):
        raise NotImplementedError(
            "INT8 projection tuning is enabled only on MetaX/Iluvatar"
        )
    if kind not in ("gate", "qkv", "down", "o"):
        raise ValueError("unknown INT8 projection kind")
    if kind in ("down", "o") and vendor != "metax":
        raise NotImplementedError("Iluvatar down/O keep their original BF16 path")
    if kind == "gate" and vendor != "iluvatar":
        raise NotImplementedError(
            "MetaX gate uses scaled_mm or fused scaled_int8_swiglu"
        )
    if not m or not n:
        return torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
    with torch_device_fn.device(q.device):
        if kind == "down":
            from ._int8_down import launch

            return launch(q, weight, scale_x, scale_w, _down_config(m), False)
        if kind == "o":
            from ._int8_o import launch

            return launch(q, weight, scale_x, scale_w, _qkv_config(m, k), False)
        if kind == "qkv":
            if vendor == "iluvatar" and 17 <= m <= 64 and k == 2048 and n == 2560:
                return _iluvatar_small_qkv(q, weight, scale_x, scale_w)
            from ._int8_qkv import launch

            return launch(q, weight, scale_x, scale_w, _qkv_config(m, k), False)
        return _iluvatar_gate(q, weight, scale_x, scale_w)


def _iluvatar_small_qkv(q, weight, sx, sw):
    from ._int8_iluvatar import down_kernel, reduce_kernel

    m, k = q.shape
    n = weight.shape[1]
    y = torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
    partial = torch.empty((4, m, n), device=q.device, dtype=torch.int32)
    down_kernel[(triton.cdiv(m, 32) * triton.cdiv(n, 128), 4)](
        q,
        weight,
        sx,
        sw,
        y,
        partial,
        m,
        n,
        k,
        32,
        128,
        64,
        4,
        False,
        1,
        4,
        0,
        num_warps=4,
        num_stages=1,
    )
    reduce_kernel[(triton.cdiv(m * n, 256),)](
        partial,
        sx,
        sw,
        y,
        m,
        n,
        4,
        False,
        256,
        num_warps=4,
    )
    return y


def _iluvatar_gate(q, weight, sx, sw):
    from ._int8_gate import gate_int8_kernel
    from ._int8_iluvatar import down_kernel

    m, k = q.shape
    n = weight.shape[1]
    y = torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
    if m <= 32 and k == 2048 and n == 12288:
        down_kernel[(triton.cdiv(m, 32) * triton.cdiv(n, 128), 1)](
            q,
            weight,
            sx,
            sw,
            y,
            y,
            m,
            n,
            k,
            32,
            128,
            128,
            1,
            False,
            1,
            4,
            0,
            num_warps=4,
            num_stages=1,
        )
    else:
        gate_int8_kernel[(triton.cdiv(m, 128) * triton.cdiv(n, 256), 1)](
            q,
            weight,
            sx,
            sw,
            y,
            y,
            m,
            n,
            k,
            128,
            256,
            64,
            1,
            False,
            1,
            8,
            8,
            num_warps=8,
            num_stages=1,
        )
    return y


def dynamic_int8_linear(x, weight, scale_w, *, kind="down"):
    """Floating activations -> per-row INT8 -> scaled BF16 projection."""
    m, k = x.shape
    if kind == "gate":
        rows = triton.cdiv(m, 32 if m <= 32 else 128) * (32 if m <= 32 else 128)
        q, sx = quantize_int8_tokens(x, padded_rows=rows)
    else:
        q, sx = quantize_int8_tokens(x, num_warps=4 if kind == "o" and m > 128 else 8)
    return scaled_int8_linear(q, weight, sx, scale_w, kind=kind)[:m]
