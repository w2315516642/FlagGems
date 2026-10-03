# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BF16 gate/up projection with a fused SwiGLU epilogue.

Weights are [gate; up] in linear-layer layout [2 * N, K]. Accumulation
uses FP32; projection results are rounded to BF16 before layout conversion
and FP32 activation. This inference operator does not implement autograd.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _linear_swiglu_kernel(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # Group adjacent row tiles to improve reuse of gate/up weight tiles in L2.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    group_width = GROUP_M * grid_n
    group_id = pid // group_width
    first_m = group_id * GROUP_M
    group_size = tl.minimum(grid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_width) % group_size
    pid_n = (pid % group_width) // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # These unwrapped indices are contiguous and start at tile boundaries.
    # Keep masks for partial tiles instead of claiming wrapped tail indices
    # have the same contiguity properties as full tiles.
    load_rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_M), BLOCK_M)
    # Pair gate/up output columns locally without materializing new weights.
    pair_axis = tl.arange(0, 2 * BLOCK_N)
    projection_cols = pid_n * BLOCK_N + pair_axis % BLOCK_N
    load_cols = tl.max_contiguous(tl.multiple_of(projection_cols, BLOCK_N), BLOCK_N)
    weight_cols = load_cols + (pair_axis // BLOCK_N) * N
    reduction = tl.arange(0, BLOCK_K)

    x_ptrs = X + load_rows[:, None] * K + reduction[None, :]
    paired_ptrs = W + weight_cols[None, :] * K + reduction[:, None]

    paired_acc = tl.zeros((BLOCK_M, 2 * BLOCK_N), tl.float32)
    for block in range(tl.cdiv(K, BLOCK_K)):
        valid_k = reduction < K - block * BLOCK_K
        if EVEN_M and EVEN_K:
            x = tl.load(x_ptrs)
        elif EVEN_K:
            x = tl.load(x_ptrs, mask=rows[:, None] < M, other=0.0)
        elif EVEN_M:
            x = tl.load(x_ptrs, mask=valid_k[None, :], other=0.0)
        else:
            x = tl.load(x_ptrs, mask=(rows[:, None] < M) & valid_k[None, :], other=0.0)
        if EVEN_N and EVEN_K:
            paired_weight = tl.load(paired_ptrs)
        elif EVEN_K:
            paired_weight = tl.load(
                paired_ptrs, mask=projection_cols[None, :] < N, other=0.0
            )
        elif EVEN_N:
            paired_weight = tl.load(paired_ptrs, mask=valid_k[:, None], other=0.0)
        else:
            paired_weight = tl.load(
                paired_ptrs,
                mask=valid_k[:, None] & (projection_cols[None, :] < N),
                other=0.0,
            )
        paired_acc = tl.dot(
            x, paired_weight, paired_acc, out_dtype=tl.float32, allow_tf32=False
        )
        x_ptrs += BLOCK_K
        paired_ptrs += BLOCK_K

    # Preserve the projection's BF16 boundary before redistributing paired
    # results, so layout conversion can operate on BF16 instead of FP32.
    paired = paired_acc.to(X.dtype.element_ty)
    gate_up = tl.trans(tl.reshape(paired, (BLOCK_M, 2, BLOCK_N)), 0, 2, 1)
    gate_bf16, up_bf16 = tl.split(gate_up)
    gate = gate_bf16.to(tl.float32)
    up = up_bf16.to(tl.float32)
    # Match the observed C500 compiled pointwise path.  Rewriting this as a
    # division by exp(-gate)+1 can change FP32 operation ordering/rounding.
    silu = gate * tl.sigmoid(gate)
    result = silu * up
    out_ptrs = Y + rows[:, None] * N + cols[None, :]
    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, result)
    else:
        tl.store(out_ptrs, result, mask=(rows[:, None] < M) & (cols[None, :] < N))


def linear_swiglu(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fused BF16 linear projection and SwiGLU for contiguous 2D tensors.

    ``input`` is [M, K] and ``weight`` is [2 * N, K], with gate rows first.
    Returns [M, N]. The projection's BF16 rounding boundary is retained;
    activation and multiplication use FP32 before the final BF16 store.
    M, N and K need not be multiples of the tile sizes. There is no token
    whitelist or model-specific shape check. Large M is supported for
    correctness; callers decide whether fusion is profitable for a workload.
    """
    if input.ndim != 2 or weight.ndim != 2:
        raise ValueError("linear_swiglu requires two-dimensional tensors")
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("linear_swiglu currently supports BF16 only")
    if input.device != weight.device or input.device.type == "cpu":
        raise ValueError("linear_swiglu requires tensors on the same accelerator")
    if not input.is_contiguous() or not weight.is_contiguous():
        raise ValueError("linear_swiglu requires contiguous input and weight")
    m, k = input.shape
    if weight.shape[1] != k or weight.shape[0] % 2:
        raise ValueError("incompatible gate/up projection dimensions")
    n = weight.shape[0] // 2
    if k == 0:
        return torch.zeros((m, n), device=input.device, dtype=input.dtype)
    output = torch.empty((m, n), device=input.device, dtype=input.dtype)
    if m == 0 or n == 0:
        return output

    logger.debug("GEMS LINEAR SWIGLU")
    block_m = min(64, max(16, triton.next_power_of_2(m)))
    block_n, block_k = 32, 64
    # MACA's basic pipeline is validated for the paired-dot loop. Do not
    # forward this vendor-specific compiler option to other Triton backends.
    options = {"pipeline": "basic"} if runtime.device.vendor_name == "metax" else {}
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    with torch_device_fn.device(input.device):
        _linear_swiglu_kernel[grid](
            input,
            weight,
            output,
            m,
            n,
            k,
            block_m,
            block_n,
            block_k,
            8,
            m % block_m == 0,
            n % block_n == 0,
            k % block_k == 0,
            num_warps=4,
            num_stages=2,
            **options,
        )
    return output
