"""Reshapes of a grid tile whose sizes are block-size expressions
(``tile.block_size // 32``): with the tile static in the IR they lower to views
(``tensor.expand_shape``/``collapse_shape``), not ``tensor.reshape``, which
bufferizes to a copy of the strided tile (the f32 fused-pack GEMM packed every
operand chunk twice). Dynamic tiles keep the symbolic lowering."""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from helion_mlir_backend import generate_mlir


def _split_rows(x: torch.Tensor) -> torch.Tensor:
    """Each ``[TM, 64]`` tile as 32x32 blocks ``[TM/32, 2, 32, 32]``, the
    blocks of a row transposed, back to ``[TM, 64]``."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        t = x[tm, tn].reshape(tm.block_size // 32, 32, tn.block_size // 32, 32)
        t = t.permute(0, 2, 1, 3).permute(0, 2, 1, 3)
        out[tm, tn] = t.reshape(tm.block_size, tn.block_size) * 2.0
    return out


def _lower(static: bool, shape: tuple[int, int]) -> tuple[str, torch.Tensor]:
    kernel = helion.kernel(
        _split_rows,
        backend="mlir",
        static_shapes=static,
        config=helion.Config(block_sizes=[64, 64]),
    )
    x = torch.randn(*shape)
    ir = str(generate_mlir(kernel, [x], config=helion.Config(block_sizes=[64, 64])))
    return ir, kernel(x) if static else None


def test_static_tile_reshape_is_a_view() -> None:
    ir, out = _lower(True, (128, 128))
    assert "tensor.expand_shape" in ir
    assert "tensor.collapse_shape" in ir
    assert "tensor.reshape" not in ir
    assert "tensor<2x32x2x32xf32>" in ir


def test_static_tile_reshape_runs() -> None:
    x = torch.randn(128, 192)
    kernel = helion.kernel(
        _split_rows,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 64]),
    )
    torch.testing.assert_close(kernel(x), x * 2.0, rtol=0, atol=0)


def test_padded_partial_tile_reshape_runs() -> None:
    """Dynamic shapes, ragged extents: partial tiles are padded to whole
    blocks, so they are static too and lower to views; results stay exact."""
    x = torch.randn(100, 160)
    kernel = helion.kernel(
        _split_rows,
        backend="mlir",
        static_shapes=False,
        config=helion.Config(block_sizes=[64, 64]),
    )
    ir = str(generate_mlir(kernel, [x], config=helion.Config(block_sizes=[64, 64])))
    assert "tensor.expand_shape" in ir
    assert "tensor.reshape" not in ir
    torch.testing.assert_close(kernel(x), x * 2.0, rtol=0, atol=0)


def _mixed(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``[TM, TN]`` tiles reshaped to ``[TM/2, 2*TN]`` (neither a pure split
    nor a merge), scaled per column of that shape, and back."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n]):
        t = x[tm, tn].reshape(tm.block_size // 2, 2 * tn.block_size) * w[None, :]
        out[tm, tn] = t.reshape(tm.block_size, tn.block_size)
    return out


def _mixed_reference(
    x: torch.Tensor, w: torch.Tensor, bm: int, bn: int
) -> torch.Tensor:
    """``_mixed`` per zero-padded ``[bm, bn]`` tile."""
    out = torch.empty_like(x)
    for i in range(0, x.shape[0], bm):
        for j in range(0, x.shape[1], bn):
            t = x[i : i + bm, j : j + bn]
            padded = torch.zeros(bm, bn)
            padded[: t.shape[0], : t.shape[1]] = t
            r = (padded.reshape(bm // 2, 2 * bn) * w[None, :]).reshape(bm, bn)
            out[i : i + bm, j : j + bn] = r[: t.shape[0], : t.shape[1]]
    return out


def test_mixed_tile_reshape_dynamic_ragged() -> None:
    """``[bm, bn] -> [bm/2, 2*bn]`` with dynamic shapes and ragged extents:
    the padded partial tiles are static, the reshape needs no symbolic sizes
    (before this change it failed to lower: "slice along dimension 0")."""
    torch.manual_seed(0)
    x, w = torch.randn(100, 160), torch.randn(128)
    config = helion.Config(block_sizes=[64, 64])
    kernel = helion.kernel(_mixed, backend="mlir", static_shapes=False, config=config)
    ir = str(generate_mlir(kernel, [x, w], config=config))
    assert "tensor<32x128xf32>" in ir
    torch.testing.assert_close(
        kernel(x, w), _mixed_reference(x, w, 64, 64), rtol=0, atol=0
    )


def _rows(x: torch.Tensor) -> torch.Tensor:
    """Row tiles of all (dynamic) columns, split into 32-row blocks."""
    m, _ = x.shape
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        t = x[tm, :].reshape(tm.block_size // 32, 32, -1)
        out[tm, :] = t.reshape(tm.block_size, -1) * 2.0
    return out


def test_dynamic_tile_reshape_stays_symbolic() -> None:
    """Negative: dynamic dims stay symbolic. The tile has a dynamic dim in the
    IR (all columns of a dynamic-shape input) and its result sizes keep a free
    symbol (``[2, 32, s]``: not ints with block sizes substituted), so nothing
    is resolved from the config and the reshape keeps the symbolic
    ``tensor.reshape``. IR only: running it aborts, also before
    this change (open item, not this path)."""
    kernel = helion.kernel(
        _rows,
        backend="mlir",
        static_shapes=False,
        config=helion.Config(block_sizes=[64]),
    )
    ir = str(
        generate_mlir(
            kernel, [torch.randn(128, 96)], config=helion.Config(block_sizes=[64])
        )
    )
    assert "tensor.reshape" in ir
    assert "tensor<2x32x" not in ir
