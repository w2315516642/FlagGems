# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Projection rounding, tail masks and graph replay for fused dense SwiGLU."""

import math

import pytest
import torch
import torch.nn.functional as F

import flag_gems


def reference(x, w):
    projection = F.linear(x, w).float()
    gate, up = projection.chunk(2, dim=-1)
    return (F.silu(gate) * up).to(x.dtype)


def assert_numerics(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).abs()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-12)
    assert error.square().mean().sqrt() / scale <= 0.01
    assert error.max() / expected.float().abs().max().clamp_min(1e-12) <= 0.03


@pytest.mark.parametrize(
    "m", [1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129, 257, 2048]
)
def test_continuous_token_sizes(m):
    torch.manual_seed(17)
    x = torch.randn((m, 2048), device=flag_gems.device, dtype=torch.bfloat16)
    w = (
        torch.randn((12288, 2048), device=flag_gems.device) / math.sqrt(2048)
    ).bfloat16()
    assert_numerics(flag_gems.linear_swiglu(x, w), reference(x, w))


@pytest.mark.parametrize(
    "m,n,k", [(3, 1, 1), (17, 35, 67), (65, 129, 257), (131, 97, 130)]
)
def test_tail_masks(m, n, k):
    torch.manual_seed(23)
    x = torch.randn((m, k), device=flag_gems.device, dtype=torch.bfloat16)
    w = (torch.randn((2 * n, k), device=flag_gems.device) / math.sqrt(k)).bfloat16()
    assert_numerics(flag_gems.linear_swiglu(x, w), reference(x, w))


@pytest.mark.parametrize("m,n,k", [(0, 35, 67), (3, 0, 67), (3, 35, 0)])
def test_empty_dimensions(m, n, k):
    x = torch.empty((m, k), device=flag_gems.device, dtype=torch.bfloat16)
    w = torch.empty((2 * n, k), device=flag_gems.device, dtype=torch.bfloat16)
    result = flag_gems.linear_swiglu(x, w)
    assert result.shape == (m, n)
    if k == 0:
        assert torch.count_nonzero(result) == 0


def test_invalid_inputs():
    x = torch.zeros((3, 64), device=flag_gems.device, dtype=torch.bfloat16)
    w = torch.zeros((96, 64), device=flag_gems.device, dtype=torch.bfloat16)
    for xx, ww in [(x.float(), w), (x, w[:95]), (x, w[:, :63]), (x[:, ::2], w[:, ::2])]:
        with pytest.raises(ValueError):
            flag_gems.linear_swiglu(xx, ww)


@pytest.mark.skipif(flag_gems.device != "cuda", reason="CUDA-compatible graph test")
@pytest.mark.parametrize("m", [7, 33, 65])
def test_graph_replay(m):
    x = torch.randn((m, 67), device=flag_gems.device, dtype=torch.bfloat16)
    w = torch.randn((70, 67), device=flag_gems.device, dtype=torch.bfloat16) / 8
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flag_gems.linear_swiglu(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flag_gems.linear_swiglu(x, w)
    x.fill_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    assert_numerics(out, reference(x, w))
