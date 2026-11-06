# SPDX-License-Identifier: Apache-2.0
"""Triton-only top-k/top-p filtering; random sampling stays with the caller."""

import torch


def supports_topk_topp(logits, k=None, p=None):
    if (
        logits.ndim != 2
        or logits.dtype != torch.float32
        or logits.device.type != "cuda"
        or logits.stride(1) != 1
    ):
        return False
    for value, dtype in [(k, torch.int32), (p, torch.float32)]:
        if value is not None and not (
            isinstance(value, torch.Tensor)
            and value.device == logits.device
            and value.dtype == dtype
            and value.shape == (logits.shape[0],)
            and value.is_contiguous()
        ):
            return False
    return logits.shape[1] > 0


def topk_topp_filter(logits, k=None, p=None):
    """Mask [M,V] logits in-place, preserving the caller's RNG and logits mode.

    Tiled top-p uses bounded, stream-owned graph workspaces. Unresolved rows
    use the general Triton pivot kernel on the device, without a host read.
    No PyTorch/CUDA reference implementation is registered for this new op.
    """
    if not supports_topk_topp(logits, k, p):
        raise ValueError("expected GPU FP32 logits, INT32 top-k and FP32 top-p vectors")
    if logits.shape[0] == 0 or (k is None and p is None):
        return logits
    from ._top_p_tiled import apply_tiled_top_p, supported
    from ._topk_topp import apply_top_k_top_p_triton

    with torch.cuda.device(logits.device):
        if (
            k is None
            and supported(logits, p)
            and not torch.cuda.is_current_stream_capturing()
        ):
            return apply_tiled_top_p(logits, p)
        return apply_top_k_top_p_triton(logits, k, p)


def clear_topk_topp_cache():
    """Release only this operator's caches after their outstanding work."""
    from ._top_p_tiled import clear_cache
    from ._topk_topp import reset_buffer_cache

    clear_cache()
    reset_buffer_cache()
