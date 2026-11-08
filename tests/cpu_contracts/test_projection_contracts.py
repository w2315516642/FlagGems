# SPDX-License-Identifier: Apache-2.0
"""Metadata contracts only: FakeTensor executes no accelerator kernels."""

import ast
import importlib.util
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

ROOT = Path(__file__).resolve().parents[2]


def functions(path, names):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    env = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


@pytest.fixture(scope="module")
def validate():
    p = ROOT / "src/flag_gems/fused/int8_linear.py"
    return functions(p, {"_validate"})["_validate"]


def tensors(m=5, k=128, n=256):
    return (
        torch.empty((m, k), device="cuda", dtype=torch.int8),
        torch.empty((n, k), device="cuda", dtype=torch.int8).t(),
        torch.empty((m, 1), device="cuda", dtype=torch.float32),
        torch.empty(n, device="cuda", dtype=torch.float32),
    )


@pytest.mark.parametrize("m,n", [(0, 256), (1, 256), (17, 256), (33, 256), (129, 0)])
def test_valid_metadata_and_empty_dimensions(validate, m, n):
    with FakeTensorMode():
        assert validate(*tensors(m=m, n=n)) == (m, n, 128)


@pytest.mark.parametrize(
    "bad",
    [
        "input_dtype",
        "weight_dtype",
        "scale_dtype",
        "scale_shape",
        "scale_stride",
        "weight_stride",
        "input_stride",
        "device",
    ],
)
def test_invalid_projection_contracts_rejected(validate, bad):
    with FakeTensorMode():
        q, w, sx, sw = tensors()
        if bad == "input_dtype":
            q = q.to(torch.bfloat16)
        elif bad == "weight_dtype":
            w = w.to(torch.bfloat16)
        elif bad == "scale_dtype":
            sx = sx.to(torch.bfloat16)
        elif bad == "scale_shape":
            sx = torch.empty(1, device="cuda")
        elif bad == "scale_stride":
            sw = torch.empty_strided((256,), (2,), device="cuda")
        elif bad == "weight_stride":
            w = torch.empty_strided(
                (128, 256), (1, 256), device="cuda", dtype=torch.int8
            )
        elif bad == "input_stride":
            q = torch.empty_strided((5, 128), (256, 2), device="cuda", dtype=torch.int8)
        elif bad == "device":
            # FakeTensor may initialize the requested GPU context on Torch
            # 2.10. A CPU scale tests the same mismatch on single-GPU hosts.
            sw = torch.empty(256, device="cpu")
        with pytest.raises(ValueError):
            validate(q, w, sx, sw)


@pytest.mark.parametrize("k", [0, 131072])
def test_int32_overflow_bound_and_empty_contraction_rejected(validate, k):
    with FakeTensorMode():
        with pytest.raises(ValueError, match="unsafe"):
            validate(*tensors(k=k))


def test_no_new_cpu_reference_fallback(validate):
    q = torch.empty((3, 8), dtype=torch.int8)
    w = torch.empty((16, 8), dtype=torch.int8).t()
    with pytest.raises(ValueError, match="accelerator"):
        validate(q, w, torch.ones(3, 1), torch.ones(16))


@pytest.fixture(scope="module")
def filter_api():
    p = ROOT / "src/flag_gems/fused/topk_topp.py"
    spec = importlib.util.spec_from_file_location("_filter_contracts", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("m", [0, 1, 17, 64, 128])
def test_filter_metadata_and_identity_without_sampling(filter_api, m):
    with FakeTensorMode():
        x = torch.empty((m, 8192), device="cuda", dtype=torch.float32)
        k = torch.empty(m, device="cuda", dtype=torch.int32)
        p = torch.empty(m, device="cuda", dtype=torch.float32)
        assert filter_api.supports_topk_topp(x, k, p)
        assert filter_api.topk_topp_filter(x) is x
        if m == 0:
            assert filter_api.topk_topp_filter(x, k, p) is x


def test_filter_preserves_original_path_for_unsupported_contracts(filter_api):
    assert not filter_api.supports_topk_topp(torch.empty(2, 8192))
    with FakeTensorMode():
        x = torch.empty((2, 8192), device="cuda", dtype=torch.float32)
        assert not filter_api.supports_topk_topp(
            x, torch.empty(2, device="cuda", dtype=torch.int64)
        )
        assert not filter_api.supports_topk_topp(x, p=torch.empty(1, device="cuda"))
        assert not filter_api.supports_topk_topp(x.to(torch.bfloat16))


def test_bf16_down_and_o_ranges_do_not_extend_to_other_shapes():
    p = ROOT / "src/flag_gems/runtime/backend/_iluvatar/ops/mm.py"
    select = functions(p, {"_tuned_bf16_config"})["_tuned_bf16_config"]
    assert select(16, 2048, 6144)[5] == 4
    assert select(17, 2048, 6144) is None
    assert select(128, 2048, 6144) is None
    assert select(129, 2048, 6144)[4:6] == (2, 1)
    assert select(1024, 2048, 6144) is not None
    assert select(1025, 2048, 6144) is None
    assert select(256, 2048, 2048) is None
    assert select(257, 2048, 2048)[4:6] == (1, 1)
    for m in [1, 16, 129, 257, 1024]:
        assert select(m, 4096, 6144) is None
