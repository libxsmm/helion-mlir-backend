"""ATen helpers typed at the call site.

Generic ATen nodes become calls to torch-mlir helpers whose signatures come from
the operands' MLIR types; these tests pin the properties of that design.
"""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel
from tests.inputs import boundary_tile
from tests.inputs import close_values

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.aten_bridge import helper_cache
from helion_mlir_backend._compiler.mlir.codegen import MLIRModuleBuilder
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError


def _kernel(*block_sizes: int) -> object:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=list(block_sizes)),
    )


@_kernel(8, 16)
def pointwise_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn]
        out[tm, tn] = torch.sigmoid(v) + torch.nn.functional.gelu(v) + torch.erf(v)
    return out


@_kernel(8, 16)
def exp_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = torch.exp(x[tm, tn])
    return out


@_kernel(16)
def runtime_scalar_kernel(x: torch.Tensor, alpha: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile]
        squared = torch.clamp_max(v * v, alpha)
        out[tile] = torch.where(v > alpha, squared, v)
    return out


@_kernel(16, 16, 16)
def square_matmul_kernel(x: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    out = torch.empty([m, m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, m]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], x[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@_kernel(8, 16)
def subscript_broadcast_kernel(
    x: torch.Tensor, row: torch.Tensor, col: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + row[tm][:, None] * col[tn][None, :]
    return out


@_kernel(8, 16)
def embedding_kernel(idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        [idx.size(0), weight.size(1)], dtype=weight.dtype, device=weight.device
    )
    for tile_b, tile_e in hl.tile([idx.size(0), weight.size(1)]):
        out[tile_b, tile_e] = weight[idx[tile_b], tile_e]
    return out


@_kernel(8, 16)
def scatter_kernel(idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile_b, tile_e in hl.tile(x.size()):
        out[idx[tile_b], tile_e] = x[tile_b, tile_e]
    return out


def test_pointwise_ops() -> None:
    torch.manual_seed(0)

    def reference(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) + torch.nn.functional.gelu(x) + torch.erf(x)

    check_kernel(pointwise_kernel, reference, [torch.randn(16, 32)], atol=1e-5)
    x = boundary_tile([0.0, 1.0], 16, 32, dtype=torch.float32)
    # f64's reference: torch's vectorized f32 gelu gives NaN at inf and inf at max.
    close_values(pointwise_kernel(x), reference(x.double()).float(), 1e-5)


@pytest.mark.parametrize("alpha", [0.25, 1.5])
def test_runtime_scalar_and_repeated_operand(alpha: float) -> None:
    """``clamp_max``/``gt`` read a runtime float; ``v * v`` reaches codegen as
    ``mul(v, None)`` and is restored from the pre-strip arguments."""
    torch.manual_seed(0)

    def reference(x: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.where(x > alpha, torch.clamp_max(x * x, alpha), x)

    check_kernel(runtime_scalar_kernel, reference, [torch.randn(64), alpha])


def test_runtime_scalar_compare_at_the_numbers_value() -> None:
    """``v > alpha`` as torch does it, in the tile's type. ``x * x`` isn't exact over
    the whole range, but is computed the same way in f32 by both."""
    alpha = 0.1

    def reference(x: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.where(x > alpha, torch.clamp_max(x * x, alpha), x)

    x = boundary_tile([alpha], 64, dtype=torch.float32)
    check_kernel(runtime_scalar_kernel, reference, [x, alpha], exact=True)


def test_contraction_with_repeated_operand() -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 32)
    check_kernel(square_matmul_kernel, lambda x: x @ x, [x], atol=1e-4, rtol=1e-4)


def test_subscript_new_axes() -> None:
    torch.manual_seed(0)
    args = [torch.randn(16, 32), torch.randn(16), torch.randn(32)]
    check_kernel(
        subscript_broadcast_kernel,
        lambda x, row, col: x + row[:, None] * col[None, :],
        args,
    )


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_gather_load(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    idx = torch.randint(0, 64, (16,), dtype=dtype)
    weight = torch.randn(64, 32)
    check_kernel(embedding_kernel, lambda idx, w: w[idx], [idx, weight])


def test_scatter_store_is_rejected() -> None:
    idx = torch.arange(16)
    with pytest.raises(UnsupportedOperationError, match="scatter"):
        generate_mlir(scatter_kernel, [idx, torch.randn(16, 32)])


def test_helion_node_meta_is_not_modified(monkeypatch: pytest.MonkeyPatch) -> None:
    build = MLIRModuleBuilder.build
    snapshots: list[list[tuple[str, str, str]]] = []

    def snapshot(builder: MLIRModuleBuilder) -> list[tuple[str, str, str]]:
        return [
            (node.name, repr(node.args), repr(node.meta.get("val")))
            for graph in builder.hf.device_ir.graphs
            for node in graph.graph.nodes
        ]

    def checked_build(builder: MLIRModuleBuilder) -> object:
        snapshots.append(snapshot(builder))
        module = build(builder)
        snapshots.append(snapshot(builder))
        return module

    monkeypatch.setattr(MLIRModuleBuilder, "build", checked_build)
    generate_mlir(runtime_scalar_kernel, [torch.randn(64), 0.5])
    before, after = snapshots
    assert before == after


def test_one_torch_mlir_run_per_compile_and_cache_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs: list[list[str]] = []
    run = helper_cache._run_torch_mlir

    def counted(requests: list[helper_cache.HelperRequest]) -> object:
        runs.append([request.name for request in requests])
        return run(requests)

    monkeypatch.setattr(helper_cache, "CACHE", helper_cache.HelperCache())
    monkeypatch.setattr(helper_cache, "_run_torch_mlir", counted)
    x = torch.randn(16, 32)
    generate_mlir(pointwise_kernel, [x])
    assert len(runs) == 1
    assert len(runs[0]) > 1  # every helper of the kernel in one batch
    generate_mlir(pointwise_kernel, [x])
    assert len(runs) == 1  # all helpers come from the cache


def test_unlowerable_helper_is_reported_at_its_source_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = helper_cache._run_torch_mlir

    def failing_sigmoid(requests: list[helper_cache.HelperRequest]) -> object:
        if any(
            request.target is torch.ops.aten.sigmoid.default for request in requests
        ):
            raise RuntimeError("no sigmoid today")
        return run(requests)

    monkeypatch.setattr(helper_cache, "CACHE", helper_cache.HelperCache())
    monkeypatch.setattr(helper_cache, "_run_torch_mlir", failing_sigmoid)
    x = torch.randn(16, 32)
    with pytest.raises(UnsupportedOperationError) as info:
        generate_mlir(pointwise_kernel, [x])
    message = str(info.value)
    assert "aten.sigmoid" in message
    assert "no sigmoid today" in message
    # The source line is shown with ``torch.sigmoid(v)`` underlined.
    assert "torch.sigmoid(v)" in message
    assert "^" * len("torch.sigmoid(v)") + "\n" in message
    # The other helpers were lowered on their own and stay usable.
    generate_mlir(exp_kernel, [x])
