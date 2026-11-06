# Copyright (c) 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry


@libentry()
@triton.jit
def o_kernel(
    A,
    W,
    SX,
    SW,
    Y,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SP: tl.constexpr,
    MIXED: tl.constexpr,
    STAGES: tl.constexpr,
    WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    rm = (pid // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    rn = (pid % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    chunk = tl.cdiv(K, BK * SP)
    acc = tl.full((BM, BN), 0, tl.float32 if MIXED else tl.int32)
    for it in range(split * chunk, (split + 1) * chunk):
        rk = it * BK + tl.arange(0, BK)
        a = tl.load(
            A + rm[:, None] * K + rk[None, :], (rm[:, None] < M) & (rk[None, :] < K), 0
        )
        w = tl.load(
            W + rn[None, :] * K + rk[:, None], (rn[None, :] < N) & (rk[:, None] < K), 0
        )
        if MIXED:
            w = w.to(tl.bfloat16)
        acc += tl.dot(a, w, allow_tf32=False)
    offsets = rm[:, None] * N + rn[None, :]
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if SP == 1:
        value = acc.to(tl.float32)
        if not MIXED:
            sx = tl.load(SX + rm, rm < M, 0.0)
            value = value * sx[:, None]
        sw = tl.load(SW + rn, rn < N, 0.0)
        tl.store(Y + offsets, value * sw[None, :], mask)
    else:
        tl.store(P + split * M * N + offsets, acc, mask)


@libentry()
@triton.jit
def reduce_kernel(
    P,
    SX,
    SW,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    SP: tl.constexpr,
    MIXED: tl.constexpr,
    B: tl.constexpr,
):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    splits = tl.arange(0, SP)
    v = tl.load(P + splits[:, None] * M * N + idx[None, :], idx[None, :] < M * N, 0)
    value = tl.sum(v, 0).to(tl.float32)
    if not MIXED:
        value = value * tl.load(SX + idx // N, idx < M * N, 0.0)
    value = value * tl.load(SW + idx % N, idx < M * N, 0.0)
    tl.store(Y + idx, value, idx < M * N)


def launch(a, w, sx, sw, config, mixed=False):
    m, k = a.shape
    n = w.shape[1]
    bm, bn, bk, st, sp, warps = config
    y = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    p = (
        torch.empty(
            (sp, m, n), device=a.device, dtype=torch.float32 if mixed else torch.int32
        )
        if sp > 1
        else y
    )
    o_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn), sp)](
        a,
        w,
        sx,
        sw,
        y,
        p,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        sp,
        mixed,
        st,
        warps,
        num_warps=warps,
        num_stages=st,
        pipeline="basic",
    )
    if sp > 1:
        reduce_kernel[(triton.cdiv(m * n, 256),)](
            p, sx, sw, y, m, n, sp, mixed, 256, num_warps=4
        )
    return y
