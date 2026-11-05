"""Nested tile loops of one block size: refused, as a block size is one symbol.

Regressions: both tiles indexed with the inner loop's position, so results were
silently wrong even with whole tiles, and with partial ones the sizes didn't match
(once a heap corruption). A tile loop over an enclosing tile of its own block size
hung the compiler (a cycle in ``enclosing_tiles``)."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir.support.errors import UnsupportedOperationError


def one_symbol(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    b = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m, block_size=b):
        low = hl.full([tm], float("inf"), dtype=torch.float32)
        for tn in hl.tile(n, block_size=b):
            low = torch.minimum(low, x[tm, tn].amin(-1))
        out[tm] = low
    return out


def two_symbols(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m, block_size=bm):
        low = hl.full([tm], float("inf"), dtype=torch.float32)
        for tn in hl.tile(n, block_size=bn):
            low = torch.minimum(low, x[tm, tn].amin(-1))
        out[tm] = low
    return out


def sub_tiles(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    outer = hl.register_block_size(n)
    inner = hl.register_block_size(outer)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        low = hl.full([tm, inner], float("inf"), dtype=torch.float32)
        for to in hl.tile(n, block_size=outer):
            for ti in hl.tile(to.begin, to.end, block_size=inner):
                low = torch.minimum(low, x[tm, ti])
        out[tm] = low.amin(-1)
    return out


def _kernel(fn: object, *blocks: int) -> helion.Kernel:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=list(blocks)),
    )(fn)


@pytest.mark.parametrize("shape", [(128, 256), (128, 252), (100, 256), (20, 60)])
def test_one_symbol_is_refused(shape: tuple[int, int]) -> None:
    x = torch.randn(*shape).abs() + 1.0
    with pytest.raises(UnsupportedOperationError, match="of one block size"):
        _kernel(one_symbol, 64)(x)


@pytest.mark.parametrize("shape", [(128, 256), (128, 252), (100, 256), (20, 60)])
def test_two_symbols_are_right(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape).abs() + 1.0
    got = _kernel(two_symbols, 64, 64)(x).double()
    assert (got - x.double().amin(-1)).abs().max() < 1e-5


@pytest.mark.isolated(timeout=120)
def test_sub_tiles_of_one_block_size_are_refused() -> None:
    x = torch.randn(16, 60).abs() + 1.0
    with pytest.raises(UnsupportedOperationError, match="of one block size"):
        _kernel(sub_tiles, 32, 8, 8)(x)


def root_grid_of_one_symbol(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    b = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n], block_size=[b, b]):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


@pytest.mark.isolated(timeout=120)
@pytest.mark.parametrize("shape", [(32, 64), (20, 60)], ids=["32x64", "20x60"])
def test_root_grid_of_one_block_size_refused(shape: tuple[int, int]) -> None:
    """The form with one ``hl.tile``: a heap corruption at 32x64 and a ``KeyError`` at
    20x60 without the refusal."""
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16]),
    )
    with pytest.raises(UnsupportedOperationError, match="nested tile loops"):
        kernel(root_grid_of_one_symbol)(torch.rand(*shape))


def root_grid_of_two_symbols(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, n], block_size=[bm, bn]):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def test_root_grid_of_two_block_sizes_of_one_value() -> None:
    x = torch.rand(32, 64)
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16, 16]),
    )
    assert torch.equal(kernel(root_grid_of_two_symbols)(x), x * 2.0)


def square_auto_grid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def test_square_auto_grid_is_not_refused() -> None:
    x = torch.rand(32, 32)
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16, 16]),
    )
    assert torch.equal(kernel(square_auto_grid)(x), x * 2.0)


def stored_sub_tiles(x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Inner tiles stored; the last inner tile of an outer tile ends inside ``o``."""
    m, n = x.size()
    outer = hl.register_block_size(n)
    inner = hl.register_block_size(outer)
    for tm in hl.tile(m):
        for to in hl.tile(n, block_size=outer):
            for ti in hl.tile(to.begin, to.end, block_size=inner):
                o[tm, ti] = x[tm, ti] * 2.0
    return o


@pytest.mark.isolated(timeout=120)
def test_stored_sub_tiles_do_not_hang() -> None:
    """Regression: the tile that owns a store was searched for around a cycle of
    enclosing tiles, forever."""
    x = torch.rand(32, 256)
    o = torch.full_like(x, 7.0)
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 128, 48]),
    )
    with pytest.raises(UnsupportedOperationError, match="bounded by the other"):
        kernel(stored_sub_tiles)(x, o)


def stored_sub_tiles_bounded_by_the_size(
    x: torch.Tensor, o: torch.Tensor
) -> torch.Tensor:
    """As :func:`stored_sub_tiles` with what the refusal names: right."""
    m, n = x.size()
    outer = hl.register_block_size(n)
    inner = hl.register_block_size(n)
    for tm in hl.tile(m):
        for to in hl.tile(n, block_size=outer):
            for ti in hl.tile(to.begin, to.end, block_size=inner):
                o[tm, ti] = x[tm, ti] * 2.0
    return o


@pytest.mark.parametrize("blocks", [[64, 128, 48], [64, 128, 32]], ids=["48", "32"])
def test_the_refusals_alternative_for_sub_tiles_is_right(blocks: list[int]) -> None:
    x = torch.rand(32, 256)
    o = torch.full_like(x, 7.0)
    kernel = helion.kernel(
        backend="mlir", static_shapes=True, config=helion.Config(block_sizes=blocks)
    )
    kernel(stored_sub_tiles_bounded_by_the_size)(x, o)
    assert torch.equal(o, x * 2.0)
