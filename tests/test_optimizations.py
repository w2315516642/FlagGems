# SPDX-License-Identifier: Apache-2.0
"Essential projection, SwiGLU, RMSNorm/quantization and filtering regressions."

import pytest
import torch
import flag_gems
from flag_gems import runtime
import math
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    runtime.device.vendor_name not in ("metax", "iluvatar"),
    reason="target-platform kernels",
)


def _shape(kind):
    return {
        "gate": (2048, 12288),
        "qkv": (2048, 2560),
        "down": (6144, 2048),
        "o": (2048, 2048),
    }[kind]


def _supported(kind):
    return kind in (
        ("gate", "qkv")
        if runtime.device.vendor_name == "iluvatar"
        else ("down", "o", "qkv")
    )


@pytest.mark.parametrize("kind", ["gate", "qkv", "down", "o"])
@pytest.mark.parametrize("m", [0, 1, 16, 17, 32, 64, 129, 4096])
@torch.inference_mode()
def test_scaled_projection_cpu_integer_reference(kind, m):
    if not _supported(kind):
        pytest.skip("this platform keeps the original projection for this kind")
    k, n = _shape(kind)
    torch.manual_seed(7301)
    q = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8).t()
    sx = torch.rand(m, 1, device="cuda", dtype=torch.float32) / 127
    sw = torch.rand(n, device="cuda", dtype=torch.float32) / 127
    y = flag_gems.scaled_int8_linear(q, weight, sx, sw, kind=kind)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    if not m:
        return
    rows = sorted({0, m // 2, m - 1})
    cols = sorted({0, 1, 31, n - 1})
    ref = q.cpu()[rows].long() @ weight.cpu()[:, cols].long()
    ref = ref.float() * sx.cpu()[rows] * sw.cpu()[cols][None, :]
    # BF16's final rounding and hardware FP32 scale multiplication may differ
    # at an ulp boundary; the integer contraction itself has no quantization.
    torch.testing.assert_close(
        y.cpu()[rows][:, cols].float(), ref, rtol=0.009, atol=0.01
    )


@pytest.mark.parametrize("m", [17, 129])
@torch.inference_mode()
def test_dynamic_projection_graph_changes_input(m):
    kind = "gate" if runtime.device.vendor_name == "iluvatar" else "down"
    k, n = _shape(kind)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8).t()
    scale = torch.full((n,), 0.001, device="cuda", dtype=torch.float32)
    for _ in range(2):
        flag_gems.dynamic_int8_linear(x, weight, scale, kind=kind)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = flag_gems.dynamic_int8_linear(x, weight, scale, kind=kind)
    x.neg_()
    graph.replay()
    torch.cuda.synchronize()
    expected = flag_gems.dynamic_int8_linear(x, weight, scale, kind=kind)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 33])
@torch.inference_mode()
def test_padded_quantization_codes_scales_and_zero_rows(m):
    k = 2048
    capacity = ((m + 31) // 32) * 32
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q, sx = flag_gems.quantize_int8_tokens(x, padded_rows=capacity)
    values = x.cpu().float()
    amax = values.abs().amax(1, keepdim=True)
    codes = (
        (values * torch.where(amax > 0, 127 / amax, 0))
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
    )
    # Reciprocal lowering may move a half-integer across the rounding boundary.
    assert (q[:m].cpu().short() - codes.short()).abs().max().item() <= 1
    torch.testing.assert_close(sx[:m].cpu(), amax / 127, rtol=1e-6, atol=0)
    assert torch.count_nonzero(q[m:]).item() == 0
    assert torch.count_nonzero(sx[m:]).item() == 0


@pytest.mark.parametrize("m", [3, 64])
@torch.inference_mode()
def test_top_p_mass_order_and_workspace_input_updates(m):
    torch.manual_seed(2301)
    x = torch.randn(m, 8192, device="cuda", dtype=torch.float32)
    p = torch.full((m,), 0.95, device="cuda", dtype=torch.float32)
    try:
        for multiplier in [1.0, -1.0, 0.5]:
            logits = x * multiplier
            before = logits.cpu().clone()
            output = flag_gems.topk_topp_filter(logits, None, p)
            assert output.data_ptr() == logits.data_ptr()
            actual = output.cpu()
            keep = torch.isfinite(actual)
            assert keep.any(dim=1).all()
            assert torch.equal(actual[keep], before[keep])
            probabilities = before.softmax(-1)
            retained = (probabilities * keep).sum(-1)
            least = torch.where(keep, probabilities, float("inf")).min(-1).values
            assert torch.all(retained >= 0.95 - 2e-5)
            assert torch.all(retained - least < 0.95 + 2e-5)
            smallest_kept = torch.where(keep, before, float("inf")).min(-1).values
            largest_removed = torch.where(~keep, before, -float("inf")).max(-1).values
            assert torch.all(largest_removed <= smallest_kept)
    finally:
        torch.cuda.synchronize()
        flag_gems.clear_topk_topp_cache()


@torch.inference_mode()
def test_top_k_and_top_p_general_kernel():
    logits = torch.randn(3, 8192, device="cuda", dtype=torch.float32)
    before = logits.cpu().clone()
    k = torch.tensor([1, 17, 64], device="cuda", dtype=torch.int32)
    p = torch.ones(3, device="cuda", dtype=torch.float32)
    try:
        result = flag_gems.topk_topp_filter(logits, k, p).cpu()
        for row, count in enumerate([1, 17, 64]):
            keep = torch.isfinite(result[row])
            assert keep.sum().item() == count
            expected = before[row].topk(count).values.sort().values
            torch.testing.assert_close(
                result[row][keep].sort().values, expected, rtol=0, atol=0
            )
    finally:
        torch.cuda.synchronize()
        flag_gems.clear_topk_topp_cache()


def _bf16_swiglu_reference(x, w):
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


@pytest.mark.parametrize("m", [1, 64, 129])
def test_continuous_token_sizes(m):
    torch.manual_seed(17)
    x = torch.randn((m, 2048), device=flag_gems.device, dtype=torch.bfloat16)
    w = (
        torch.randn((12288, 2048), device=flag_gems.device) / math.sqrt(2048)
    ).bfloat16()
    assert_numerics(flag_gems.linear_swiglu(x, w), _bf16_swiglu_reference(x, w))


@pytest.mark.parametrize("m,n,k", [(17, 35, 67)])
def test_tail_masks(m, n, k):
    torch.manual_seed(23)
    x = torch.randn((m, k), device=flag_gems.device, dtype=torch.bfloat16)
    w = (torch.randn((2 * n, k), device=flag_gems.device) / math.sqrt(k)).bfloat16()
    assert_numerics(flag_gems.linear_swiglu(x, w), _bf16_swiglu_reference(x, w))


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
@pytest.mark.parametrize("m", [33])
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
    assert_numerics(out, _bf16_swiglu_reference(x, w))


def _norm_quant_reference(x, w, eps, residual):
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
    expected = _norm_quant_reference(
        x.cpu(), w.cpu(), 1e-6, r.cpu() if r is not None else None
    )
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
