# SPDX-License-Identifier: Apache-2.0
"""Tiled top-p filtering with device-side fallback and bounded graph reuse."""

import torch
import triton as tr
import triton.language as tl

from flag_gems import runtime


from collections import OrderedDict
import threading

from ._topk_topp import (
    _NORMAL_CDF_TO_SIGMA_TABLE,
    _PERCENTILE_TO_STD_TABLE,
    _topk_topp_kernel,
)


@tr.jit
def initial_part(
    X,
    PART,
    V: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
    INDIRECT: tl.constexpr = False,
):
    if INDIRECT:
        X = tl.load(X).to(tl.pointer_type(tl.float32))
    row = tl.program_id(0)
    tile = tl.program_id(1)
    j = tile * B + tl.arange(0, B)
    x = tl.load(X + row * V + j, j < V, -float("inf"))
    mx = tl.max(x, 0)
    center = tl.where(mx > -float("inf"), mx, 0.0)
    mass = tl.sum(tl.exp(x - center), 0)
    tl.store(PART + (row * T + tile) * 4, mx)
    tl.store(PART + (row * T + tile) * 4 + 1, mass)


@tr.jit
def initial_row(PART, P, STATE, T: tl.constexpr, BT: tl.constexpr):
    row = tl.program_id(0)
    t = tl.arange(0, BT)
    mxs = tl.load(PART + (row * T + t) * 4, t < T, -float("inf"))
    sums = tl.load(PART + (row * T + t) * 4 + 1, t < T, 0.0)
    mx = tl.max(mxs, 0)
    center = tl.where(mx > -float("inf"), mx, 0.0)
    mass = tl.sum(sums * tl.exp(mxs - center), 0)
    p = tl.load(P + row)
    valid = (mass == mass) & (mass < float("inf")) & (p > 0.0) & (p <= 1.0)
    active = (mass > 0) & (p < 1.0) & valid
    tl.store(STATE + row * 8 + 0, center)
    tl.store(STATE + row * 8 + 1, mass)
    tl.store(STATE + row * 8 + 2, p * mass)
    tl.store(STATE + row * 8 + 3, 0.0)
    tl.store(STATE + row * 8 + 4, 1.0)
    tl.store(STATE + row * 8 + 5, tl.where(active, 0.0, -1.0))
    tl.store(STATE + row * 8 + 6, tl.where(valid, active.to(tl.float32), -1.0))
    tl.store(STATE + row * 8 + 7, 0.0)


@tr.jit
def weights_kernel(
    X, W, STATE, V: tl.constexpr, B: tl.constexpr, INDIRECT: tl.constexpr = False
):
    if INDIRECT:
        X = tl.load(X).to(tl.pointer_type(tl.float32))
    row = tl.program_id(0)
    j = tl.program_id(1) * B + tl.arange(0, B)
    x = tl.load(X + row * V + j, j < V, -float("inf"))
    mx = tl.load(STATE + row * 8)
    tl.store(W + row * V + j, tl.exp(x - mx), j < V)


@tr.jit
def scan_part(W, STATE, PART, V: tl.constexpr, T: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    if tl.load(STATE + row * 8 + 6) > 0:
        lo = tl.load(STATE + row * 8 + 3)
        hi = tl.load(STATE + row * 8 + 4)
        pivot = (lo + hi) * 0.5
        j = tile * B + tl.arange(0, B)
        w = tl.load(W + row * V + j, j < V, 0.0)
        above = (w > pivot) & (j < V)
        mass = tl.sum(tl.where(above, w, 0.0), 0)
        mn = tl.min(tl.where(above, w, float("inf")), 0)
        count = tl.sum((above & (w == mn)).to(tl.int32), 0)
        below = tl.max(tl.where(~above & (j < V), w, 0.0), 0)
        tl.store(PART + (row * T + tile) * 4, mass)
        tl.store(PART + (row * T + tile) * 4 + 1, mn)
        tl.store(PART + (row * T + tile) * 4 + 2, count)
        tl.store(PART + (row * T + tile) * 4 + 3, below)


@tr.jit
def update_row(PART, STATE, T: tl.constexpr, BT: tl.constexpr, ITER):
    row = tl.program_id(0)
    if tl.load(STATE + row * 8 + 6) > 0:
        t = tl.arange(0, BT)
        mass = tl.sum(tl.load(PART + (row * T + t) * 4, t < T, 0.0), 0)
        mins = tl.load(PART + (row * T + t) * 4 + 1, t < T, float("inf"))
        mn = tl.min(mins, 0)
        counts = tl.load(PART + (row * T + t) * 4 + 2, t < T, 0.0)
        count = tl.sum(tl.where(mins == mn, counts, 0.0), 0)
        below = tl.max(tl.load(PART + (row * T + t) * 4 + 3, t < T, 0.0), 0)
        target = tl.load(STATE + row * 8 + 2)
        enough = mass >= target
        strict = mass - tl.where(count > 0, mn, 0.0) * count
        done = enough & (strict < target) & (count > 0)
        lo = tl.load(STATE + row * 8 + 3)
        hi = tl.load(STATE + row * 8 + 4)
        tl.store(STATE + row * 8 + 3, tl.where(enough, mn, lo))
        tl.store(STATE + row * 8 + 4, tl.where(enough, hi, below))
        tl.store(STATE + row * 8 + 5, tl.where(done, mn, 0.0))
        tl.store(STATE + row * 8 + 6, (~done).to(tl.float32))
        tl.store(STATE + row * 8 + 7, ITER + 1)


@tr.jit
def final_part(W, STATE, PART, V: tl.constexpr, T: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    j = tile * B + tl.arange(0, B)
    w = tl.load(W + row * V + j, j < V, 0.0)
    cut = tl.load(STATE + row * 8 + 5)
    mass = tl.sum(tl.where((w > cut) & (j < V), w, 0.0), 0)
    count = tl.sum(((w == cut) & (j < V)).to(tl.int32), 0)
    tl.store(PART + (row * T + tile) * 4, mass)
    tl.store(PART + (row * T + tile) * 4 + 1, count)


@tr.jit
def final_row(
    PART, STATE, PREFIX, KEEP, P, FALLBACK_P, T: tl.constexpr, BT: tl.constexpr
):
    row = tl.program_id(0)
    t = tl.arange(0, BT)
    mass = tl.sum(tl.load(PART + (row * T + t) * 4, t < T, 0.0), 0)
    counts = tl.load(PART + (row * T + t) * 4 + 1, t < T, 0.0).to(tl.int32)
    prefix = tl.cumsum(counts, 0) - counts
    count = tl.sum(counts, 0)
    cut = tl.load(STATE + row * 8 + 5)
    target = tl.load(STATE + row * 8 + 2)
    need = tl.ceil((target - mass) / tl.maximum(cut, 1e-30)).to(tl.int32)
    need = tl.minimum(count, tl.maximum(0, need))
    tl.store(PREFIX + row * T + t, prefix, t < T)
    tl.store(KEEP + row, need)
    unresolved = tl.load(STATE + row * 8 + 6) != 0
    tl.store(FALLBACK_P + row, tl.where(unresolved, tl.load(P + row), 1.0))


@tr.jit
def apply_mask(
    X,
    W,
    STATE,
    PREFIX,
    KEEP,
    Y,
    V: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
    INDIRECT: tl.constexpr = False,
):
    if INDIRECT:
        X = tl.load(X).to(tl.pointer_type(tl.float32))
        Y = tl.load(Y + 1).to(tl.pointer_type(tl.float32))
    row = tl.program_id(0)
    tile = tl.program_id(1)
    j = tile * B + tl.arange(0, B)
    x = tl.load(X + row * V + j, j < V, -float("inf"))
    w = tl.load(W + row * V + j, j < V, 0.0)
    cut = tl.load(STATE + row * 8 + 5)
    eq = (w == cut) & (j < V)
    rank = tl.cumsum(eq.to(tl.int32), 0) - 1 + tl.load(PREFIX + row * T + tile)
    keep = (
        (tl.load(STATE + row * 8 + 6) != 0)
        | (cut < 0)
        | (w > cut)
        | eq & (rank < tl.load(KEEP + row))
    )
    tl.store(Y + row * V + j, tl.where(keep, x, -float("inf")), j < V)


# Each thread owns its workspaces. Different devices/streams never share scratch.
_LOCAL = threading.local()
_CAPTURE_LOCK = threading.RLock()
_MAX_CACHE_BYTES = 256 * 1024 * 1024
_MAX_CACHE_ENTRIES = 8


class _Workspace:
    def __init__(self, logits, iterations=32, rows=None, *, indirect=None):
        self.m = logits.shape[0] if rows is None else rows
        self.v = logits.shape[1]
        self.b = 4096
        self.t = tr.cdiv(self.v, self.b)
        self.iterations = iterations
        self.x = logits.new_zeros((self.m, self.v))
        self.p = logits.new_empty(self.m)
        self.part = logits.new_empty((self.m, self.t, 4))
        self.state = logits.new_empty((self.m, 8))
        self.w = torch.empty_like(self.x)
        self.y = torch.empty_like(self.x)
        self.prefix = torch.empty(
            (self.m, self.t), device=logits.device, dtype=torch.int32
        )
        self.keep = torch.empty(self.m, device=logits.device, dtype=torch.int32)
        self.fallback_p = logits.new_empty(self.m)
        self.buffer = torch.empty_like(self.x)
        self.normal_table = logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)
        self.percentile_table = logits.new_tensor(_PERCENTILE_TO_STD_TABLE)
        # Enable external graph I/O only on the validated target. Other
        # vendors keep scratch I/O until their device regression is complete.
        self.indirect = (
            (runtime.device.vendor_name == "iluvatar") if indirect is None else indirect
        )
        self.io = torch.empty(2, dtype=torch.int64, device=logits.device)
        self.graph = None
        self.done = torch.cuda.Event()
        self.nbytes = sum(
            t.numel() * t.element_size()
            for t in vars(self).values()
            if isinstance(t, torch.Tensor)
        )

    def kernels(self):
        m, v, b, t = self.m, self.v, self.b, self.t
        grid, bt = (m, t), tr.next_power_of_2(t)
        x = self.io if self.indirect else self.x
        y = self.io if self.indirect else self.y
        initial_part[grid](x, self.part, v, t, b, INDIRECT=self.indirect, num_warps=4)
        initial_row[(m,)](self.part, self.p, self.state, t, bt, num_warps=4)
        weights_kernel[grid](
            x, self.w, self.state, v, b, INDIRECT=self.indirect, num_warps=4
        )
        for it in range(self.iterations):
            scan_part[grid](self.w, self.state, self.part, v, t, b, num_warps=4)
            update_row[(m,)](self.part, self.state, t, bt, it, num_warps=4)
        final_part[grid](self.w, self.state, self.part, v, t, b, num_warps=4)
        final_row[(m,)](
            self.part,
            self.state,
            self.prefix,
            self.keep,
            self.p,
            self.fallback_p,
            t,
            bt,
            num_warps=4,
        )
        apply_mask[grid](
            x,
            self.w,
            self.state,
            self.prefix,
            self.keep,
            y,
            v,
            t,
            b,
            INDIRECT=self.indirect,
            num_warps=4,
        )
        # Completed rows use p=1 and skip V5 search/masking. Unresolved rows
        # retain original logits and their original p, all without a CPU read.
        _topk_topp_kernel[(m,)](
            y,
            v,
            self.buffer,
            self.percentile_table,
            self.normal_table,
            self.y,
            self.fallback_p,
            BATCH_SIZE=m,
            VOCAB_SIZE=v,
            MASK_VALUE=float("-inf"),
            BLOCK_SIZE=8192,
            BLOCK_SIZE_TRUNC=4096,
            TOPK_ENABLED=False,
            TOPP_ENABLED=True,
            LOGITS_INDIRECT=self.indirect,
        )

    def copy_inputs(self, logits, p, *, external=True):
        rows = logits.shape[0]
        direct = self.indirect and external and rows == self.m
        if not direct:
            self.x[:rows].copy_(logits)
        if self.indirect:
            source, target = (logits, logits) if direct else (self.x, self.y)
            # A fresh pinned allocation avoids overwriting a host table while
            # its earlier asynchronous DMA may still be pending. PyTorch's
            # pinned allocator retains the allocation until the copy completes.
            addresses = torch.tensor(
                [source.data_ptr(), target.data_ptr()],
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            self.io.copy_(addresses, non_blocking=True)
            if direct:
                logits.record_stream(torch.cuda.current_stream(logits.device))
        # The padded rows are never returned. Reset their p even after a prior
        # call used more rows from this bucket, so they cannot run fallback.
        self.p.fill_(1.0)
        self.p[:rows].copy_(p)
        return direct

    def capture(self, logits, p):
        # Warmup/capture must never filter caller storage before apply().
        self.copy_inputs(logits, p, external=False)
        current = torch.cuda.current_stream(logits.device)
        stream = torch.cuda.Stream(device=logits.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            self.kernels()  # Compile and warm up before capture.
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.kernels()
        current.wait_stream(stream)

    def apply(self, logits, p):
        direct = self.copy_inputs(logits, p)
        self.graph.replay()
        # Full-capacity rows read/write the caller through the parameter table.
        # Partial buckets retain padded scratch and the existing copy contract.
        if not direct:
            logits.copy_(self.y[: logits.shape[0]])
        self.done.record()
        return logits


def supported(logits, p):
    return (
        logits.is_cuda
        and logits.ndim == 2
        and logits.dtype == torch.float32
        and logits.is_contiguous()
        and 0 < logits.shape[0] <= 128
        and 8192 <= logits.shape[1] <= 262144
        and isinstance(p, torch.Tensor)
        and p.device == logits.device
        and p.dtype == torch.float32
        and p.ndim == 1
        and p.numel() == logits.shape[0]
        and p.is_contiguous()
        and tr.next_power_of_2(logits.shape[0]) * logits.shape[1] * 16 + 1024 * 1024
        <= _MAX_CACHE_BYTES
    )


def _apply_tiled_top_p(logits, p):
    """Supported top-p-only inputs; caller retains V5 for other contracts.

    Scratch/graphs are bounded per calling thread. First use of a shape pays
    warmup/capture; steady state has no CPU read of device values. Eviction
    waits for that entry's last use before releasing captured allocations.
    """
    with torch.cuda.device(logits.device):
        cache = getattr(_LOCAL, "cache", None)
        if cache is None:
            cache = _LOCAL.cache = OrderedDict()
        bucket = tr.next_power_of_2(logits.shape[0])
        key = (
            logits.device.index,
            torch.cuda.current_stream().cuda_stream,
            (bucket, logits.shape[1]),
        )
        entry = cache.get(key)
        if entry is None:
            estimate = bucket * logits.shape[1] * 16 + 1024 * 1024
            while cache and (
                len(cache) >= _MAX_CACHE_ENTRIES
                or sum(w.nbytes for w in cache.values()) + estimate > _MAX_CACHE_BYTES
            ):
                _, old = cache.popitem(last=False)
                old.done.synchronize()
            entry = _Workspace(logits, rows=bucket)
            entry.capture(logits, p)
            cache[key] = entry
        cache.move_to_end(key)
        return entry.apply(logits, p)


def _clear_cache():
    """Release this thread's cached graphs after their last outstanding use."""
    cache = getattr(_LOCAL, "cache", {})
    for entry in cache.values():
        entry.done.synchronize()
    cache.clear()


def apply_tiled_top_p(logits, p):
    # Serialize capture setup against this module's other launches. GPU work
    # stays asynchronous; independent streams still own independent buffers.
    with _CAPTURE_LOCK:
        return _apply_tiled_top_p(logits, p)


def clear_cache():
    with _CAPTURE_LOCK:
        _clear_cache()
