"""Loop geometry from Helion metadata: begin/end/step, tile positions, forall shape."""

from __future__ import annotations

import re

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel
from tests.harness import opt_pipeline

from helion_mlir_backend import generate_mlir


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4))
def root_offset_1d(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(4, x.size(0)):
        out[tile] = x[tile] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4, 8))
def root_offset_2d(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tm, tn in hl.tile([4, 8], x.size()):
        out[tm, tn] = x[tm, tn] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 4))
def nested_offset(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(2, n):
            out[tm, tn] = x[tm, tn] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 4))
def nested_offset_end(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(4, n - 4):
            out[tm, tn] = x[tm, tn] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4))
def nested_grid_step(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tm in hl.tile(x.size(0)):
        for j in hl.grid(0, x.size(1), 2):
            out[tm, j] = x[tm, j] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4))
def root_grid_step(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for i in hl.grid(1, x.size(0), 3):
        out[i, :] = x[i, :] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def tile_begin_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(8, x.size(0)):
        out[tile] = x[tile] + tile.begin
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def tile_end_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(8, x.size(0)):
        out[tile] = x[tile] + tile.end
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def tile_id_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(8, x.size(0)):
        out[tile] = x[tile] + tile.id
    return out


def _add_one_in(*region: slice):
    """Reference for a ``zeros_like`` output with ``x + 1`` written to ``region``."""

    def reference(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[region] = x[region] + 1.0
        return out

    return reference


def _per_tile(value):
    """Reference for ``hl.tile(8, n)`` adding ``value(begin, end)`` to each tile."""

    def reference(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for begin in range(8, x.size(0), 8):
            end = min(begin + 8, x.size(0))
            out[begin:end] = x[begin:end] + value(begin, end)
        return out

    return reference


@pytest.mark.parametrize(
    ("kernel", "reference", "shape"),
    [
        pytest.param(root_offset_1d, _add_one_in(slice(4, None)), (20,), id="root_1d"),
        pytest.param(
            root_offset_2d,
            _add_one_in(slice(4, None), slice(8, None)),
            (12, 24),
            id="root_2d",
        ),
        pytest.param(
            nested_offset,
            _add_one_in(slice(None), slice(2, None)),
            (16, 10),
            id="nested_begin",
        ),
        pytest.param(
            nested_offset_end,
            _add_one_in(slice(None), slice(4, 12)),
            (16, 16),
            id="nested_begin_end",
        ),
        pytest.param(
            nested_grid_step,
            _add_one_in(slice(None), slice(None, None, 2)),
            (8, 6),
            id="nested_grid_step",
        ),
        pytest.param(
            root_grid_step,
            _add_one_in(slice(1, None, 3)),
            (10, 4),
            id="root_grid_step",
        ),
        pytest.param(
            tile_begin_kernel,
            _per_tile(lambda begin, end: begin),
            (32,),
            id="tile_begin",
        ),
        pytest.param(
            tile_end_kernel,
            _per_tile(lambda begin, end: end),
            (32,),
            id="tile_end",
        ),
        pytest.param(
            tile_id_kernel,
            _per_tile(lambda begin, end: begin // 8),
            (32,),
            id="tile_id",
        ),
    ],
)
def test_loop_geometry(kernel, reference, shape) -> None:
    torch.manual_seed(0)
    check_kernel(
        kernel,
        reference,
        [torch.randn(*shape)],
        paths=("direct", "generated"),
    )


def test_outer_forall_is_normalized() -> None:
    module = generate_mlir(root_offset_2d, [torch.randn(12, 24)])
    text = str(module)
    assert re.search(r"scf\.forall \(%\w+, %\w+\) in \(2, 2\)", text)
    assert "affine_map<(d0) -> (d0 * 4 + 4)>" in text
    assert "affine_map<(d0) -> (d0 * 8 + 8)>" in text


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4, 8))
def scale_2d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def test_partial_tiles_are_versioned() -> None:
    """A tile that may be partial is lowered twice per dim: full (static, no
    padded loads) and partial (Helion's tile mask)."""
    x = torch.randn(10, 20)
    torch.testing.assert_close(scale_2d(x), x * 2.0)
    text = str(generate_mlir(scale_2d, [x]))
    assert text.count("scf.if") == 3
    assert "tensor<4x8xf32>" in text
    assert text.count("tensor.pad") == 3


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(32, 32, 32))
def f32_matmul_32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@pytest.mark.isolated
@pytest.mark.slow
def test_sfc_remap_applies_to_normalized_forall() -> None:
    """The optimizing pipeline's SFC remap only rewrites a normalized 2-D forall."""
    from lighthouse.pipeline.driver import BackendDriver
    import mlir.ir as ir

    from helion_mlir_backend._compiler.execution import inline_module
    from helion_mlir_backend._compiler.targets.cpu import pipeline_descriptor

    module = inline_module(
        generate_mlir(f32_matmul_32, [torch.randn(128, 64), torch.randn(64, 128)])
    )
    remapped = False
    with module.context, ir.Location.unknown():
        driver = BackendDriver(
            module, "f32_matmul_32", result_to_args=False, benchmark=False
        )
        driver.add_stage(pipeline_descriptor("opt"))
        for stage in driver.stages:
            before = str(module)
            module = stage.apply(module)
            if "sfc_remap_forall" in str(stage):
                # A 4x4 tile grid is remapped through a 16-entry index table.
                remapped = str(module) != before and "tensor<16xi64>" in str(module)
                break
    assert remapped


@pytest.mark.isolated
@pytest.mark.slow
def test_sfc_matmul_executes_under_opt_pipeline() -> None:
    x, y = torch.randn(128, 64), torch.randn(64, 128)
    with opt_pipeline():
        result = f32_matmul_32(x, y)
    torch.testing.assert_close(result, x @ y, atol=1e-4, rtol=1e-4)


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(32, 1, 32, 32, 1))
def tile_expression_then_phase(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A tile expression in a local, read in an inner loop, with a phase after them."""
    rows, n = x.size()
    p = torch.empty_like(x)
    out = torch.empty_like(x)
    for tm, tb in hl.tile([n, rows]):
        h = tb.begin // 4
        acc = hl.zeros([tb, tm], dtype=torch.float32)
        for tk in hl.tile(n):
            acc = acc + y[h, tk].sum()
        p[tb, tm] = acc
    hl.barrier()
    for tm, tb in hl.tile([n, rows]):
        out[tb, tm] = p[tb, tm]
    return out


def test_tile_expression_then_phase() -> None:
    """Regression: code after the loop made ``h`` a host argument, which failed with
    "name 'h' is not defined"."""
    torch.manual_seed(0)
    check_kernel(
        tile_expression_then_phase,
        lambda x, y: y.sum(-1).repeat_interleave(4)[:, None].expand(x.shape),
        [torch.randn(8, 64), torch.randn(2, 64)],
    )


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(32, 1, 32, 32, 1, 32))
def tile_expression_past_its_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A tile expression read after its loop ended, where no position exists."""
    rows, n = x.size()
    p = torch.empty_like(x)
    out = torch.empty_like(x)
    for tm, tb in hl.tile([n, rows]):
        h = tb.begin // 4
        acc = hl.zeros([tb, tm], dtype=torch.float32)
        for tk in hl.tile(n):
            acc = acc + y[h, tk].sum()
        p[tb, tm] = acc
    hl.barrier()
    for tm, tb in hl.tile([n, rows]):
        acc = hl.zeros([tb, tm], dtype=torch.float32)
        for tk in hl.tile(n):
            acc = acc + y[h, tk].sum()
        out[tb, tm] = acc + p[tb, tm]
    return out


def test_tile_expression_past_its_loop_is_refused() -> None:
    """Helion's tracer refuses it; no host argument or stale position comes of it."""
    with pytest.raises(helion.exc.InternalError, match="AssertionError"):
        tile_expression_past_its_loop(torch.zeros(8, 64), torch.zeros(2, 64))
