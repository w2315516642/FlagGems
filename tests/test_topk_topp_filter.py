# SPDX-License-Identifier: Apache-2.0
"""Device filtering tests independent of vLLM and of its random sampler."""

import pytest
import torch

import flag_gems
from flag_gems import runtime

pytestmark = pytest.mark.skipif(
    runtime.device.vendor_name not in ("metax", "iluvatar"),
    reason="current filter validation targets MetaX/Iluvatar",
)


@pytest.mark.parametrize("m", [1, 3, 17, 64])
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
