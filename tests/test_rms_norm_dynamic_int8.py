# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flag_gems


def reference(x, w, eps, residual):
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    if residual is None:
        y = y.to(x.dtype).float()
    y = (y * w.float()).to(x.dtype).float()
    amax = y.abs().amax(-1, keepdim=True)
    q = (
        (y * torch.where(amax > 0, 127 / amax, 0))
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
    )
    return q, amax / 127, xf.to(x.dtype) if residual is not None else None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape", [(0, 2048), (1, 31), (7, 513), (64, 2048), (129, 4096)]
)
@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_rms_norm_quantized_contract(dtype, shape, with_residual):
    torch.manual_seed(17)
    x = torch.randn((shape[0], shape[1] * 2), device=flag_gems.device, dtype=dtype)[
        :, ::2
    ]
    w = torch.randn(shape[1], device=x.device, dtype=dtype)
    r = torch.randn_like(x) if with_residual else None
    x_before = x.clone()
    r_before = r.clone() if r is not None else None
    expected = reference(x.cpu(), w.cpu(), 1e-6, r.cpu() if r is not None else None)
    q, s, rr = flag_gems.rms_norm_dynamic_int8(x, w, 1e-6, r)
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    assert q.shape == x.shape and s.shape == (shape[0], 1)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    if r is not None:
        torch.testing.assert_close(r, r_before, rtol=0, atol=0)
        torch.testing.assert_close(rr.cpu(), expected[2], rtol=0, atol=0)
    else:
        assert rr is None
    torch.testing.assert_close(s.cpu(), expected[1], rtol=0.002, atol=1e-7)
    # A reduction-rounding boundary can move a code by one; count it explicitly.
    if q.numel():
        delta = (q.cpu().int() - expected[0].int()).abs()
        assert delta.max().item() <= 1
        assert (delta != 0).float().mean().item() < 0.005


@torch.inference_mode()
def test_zero_rows_and_cuda_graph():
    x = torch.zeros((8, 2048), device=flag_gems.device, dtype=torch.bfloat16)
    w = torch.ones(2048, device=x.device, dtype=x.dtype)
    r = torch.zeros_like(x)
    for _ in range(3):
        flag_gems.rms_norm_dynamic_int8(x, w, 1e-6, r)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, scale, rr = flag_gems.rms_norm_dynamic_int8(x, w, 1e-6, r)
    graph.replay()
    assert torch.count_nonzero(q).item() == 0
    assert torch.count_nonzero(scale).item() == 0
    assert torch.count_nonzero(rr).item() == 0
    x.fill_(2)
    graph.replay()
    assert torch.all(q == 127).item()
    torch.testing.assert_close(rr, x, rtol=0, atol=0)


@pytest.mark.parametrize("with_residual", [False, True])
def test_ties_round_to_even(with_residual):
    w = torch.tensor(
        [127, 0.5, 1.5, -0.5, -1.5, 126.5, -126.5, 0],
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    x = torch.ones((1, 8), device=w.device, dtype=w.dtype)
    r = torch.zeros_like(x) if with_residual else None
    q, scale, _ = flag_gems.rms_norm_dynamic_int8(x, w, 1e-6, r)
    torch.testing.assert_close(
        q.cpu(),
        torch.tensor([[127, 0, 2, 0, -2, 126, -126, 0]], dtype=torch.int8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(scale.cpu(), torch.ones(1, 1), rtol=0, atol=0)
