# Copyright (c) 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def gate_int8_kernel(
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
    GROUP: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    gn = tl.cdiv(N, BN)
    if GROUP > 0:
        gm = tl.cdiv(M, BM)
        gid = pid // (GROUP * gn)
        gs = tl.minimum(gm - gid * GROUP, GROUP)
        pm = gid * GROUP + pid % gs
        pn = (pid % (GROUP * gn)) // gs
    else:
        pm = pid // gn
        pn = pid % gn
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
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
