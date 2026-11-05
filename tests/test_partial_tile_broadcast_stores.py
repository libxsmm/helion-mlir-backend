"""A stored tile broadcast into a partial tile, against float64.

Regression: the broadcast's loop took the stored tile's size in its other dims, past
the shorter destination slice: heap corruption at partial rows (a process abort).
"""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

import helion_mlir_backend  # noqa: F401

# A corrupted heap can hang rather than abort.
pytestmark = pytest.mark.isolated(timeout=120)


def row_sums(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m, block_size=bm):
        for tn in hl.tile(n, block_size=bn):
            out[tm, tn] = x[tm, tn].sum(-1, keepdim=True)
    return out


def row_minima(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m, block_size=bm):
        for tn in hl.tile(n, block_size=bn):
            out[tm, tn] = x[tm, tn].amin(-1, keepdim=True)
    return out


def column_sums(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m, block_size=bm):
        for tn in hl.tile(n, block_size=bn):
            out[tm, tn] = x[tm, tn].sum(0, keepdim=True)
    return out


def added_row_sums(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    out = torch.zeros_like(x)
    for tm in hl.tile(m, block_size=bm):
        for tn in hl.tile(n, block_size=bn):
            out[tm, tn] += x[tm, tn].sum(-1, keepdim=True)
    return out


def grid_row_sums(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn].sum(-1, keepdim=True)
    return out


def row_sums_cast(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m, block_size=bm):
        for tn in hl.tile(n, block_size=bn):
            out[tm, tn] = x[tm, tn].to(torch.float32).sum(-1, keepdim=True).to(x.dtype)
    return out


def _tiles(
    x: torch.Tensor, op: object, blocks: tuple[int, int] = (16, 16)
) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in range(0, x.shape[0], blocks[0]):
        for col in range(0, x.shape[1], blocks[1]):
            tile = x[row : row + blocks[0], col : col + blocks[1]]
            out[row : row + blocks[0], col : col + blocks[1]] = op(tile)
    return out


FORMS = {
    "row_sums": (row_sums, lambda t: t.sum(-1, keepdim=True).expand_as(t)),
    "row_minima": (row_minima, lambda t: t.amin(-1, keepdim=True).expand_as(t)),
    "column_sums": (column_sums, lambda t: t.sum(0, keepdim=True).expand_as(t)),
    "added_row_sums": (added_row_sums, lambda t: t.sum(-1, keepdim=True).expand_as(t)),
    "grid_row_sums": (grid_row_sums, lambda t: t.sum(-1, keepdim=True).expand_as(t)),
}


@pytest.mark.parametrize(
    "shape",
    [(20, 64), (20, 60), (16, 60), (32, 64)],
    ids=lambda s: f"{s[0]}x{s[1]}",
)
@pytest.mark.parametrize("name", list(FORMS))
def test_broadcast_store_into_a_partial_tile(name: str, shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    fn, op = FORMS[name]
    x = torch.randn(*shape).abs() + 1.0
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16, 16]),
    )
    ref = _tiles(x.double(), op)
    assert (kernel(fn)(x).double() - ref).abs().max() / ref.abs().max() < 1e-5


def test_broadcast_store_with_a_cast() -> None:
    torch.manual_seed(0)
    x = (torch.randn(20, 64).abs() + 1.0).to(torch.bfloat16)
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16, 16]),
    )
    ref = _tiles(x.double(), lambda t: t.sum(-1, keepdim=True).expand_as(t))
    got = kernel(row_sums_cast)(x).double()
    assert (got - ref).abs().max() / ref.abs().max() < 2**-7
