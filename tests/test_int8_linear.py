# SPDX-License-Identifier: Apache-2.0
"""On-device correctness checks; integer references are computed on CPU."""

import pytest
import torch

import flag_gems
from flag_gems import runtime

pytestmark = pytest.mark.skipif(
    runtime.device.vendor_name not in ("metax", "iluvatar"),
    reason="projection tuning targets MetaX and Iluvatar",
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
@pytest.mark.parametrize(
    "m", [0, 1, 16, 17, 32, 33, 64, 128, 129, 512, 1024, 2048, 4096]
)
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


@pytest.mark.parametrize("m", [1, 17, 33, 129])
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


@pytest.mark.parametrize("m", [1, 17, 32, 33, 64, 129])
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
