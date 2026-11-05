"""Inner loops bounded by an enclosing tile's position (non-rectangular iteration):
results on the CPU, including partial tiles."""

from __future__ import annotations

import helion
import helion.language as hl
from mlir import ir
from mlir.dialects import arith
import pytest
import torch

from tests.harness import check_kernel

from helion_mlir_backend._compiler.mlir.analysis.index_facts import IndexFacts
from helion_mlir_backend._compiler.mlir.analysis.index_facts import index_facts
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError


def _config(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


def _kernel(fn: object, *block_sizes: int) -> object:
    return helion.kernel(
        fn, backend="mlir", static_shapes=True, config=_config(*block_sizes)
    )


def up_to_tile_end(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def behind_window(
    x: torch.Tensor, window: hl.constexpr, tile: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(max((tm.begin - window) // tile * tile, 0), tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def zero_span(x: torch.Tensor, width: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.begin, tm.begin + width):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def capped(x: torch.Tensor, cap: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(min(tm.end, cap)):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def _tiles(rows: int, tile: int) -> list[tuple[int, int]]:
    return [(b, min(b + tile, rows)) for b in range(0, rows, tile)]


def _expected(x: torch.Tensor, tile: int, keys: object) -> torch.Tensor:
    """``keys(begin, end)``: the range of keys of the row tile ``begin, end``."""
    out = torch.empty_like(x)
    for begin, end in _tiles(x.size(0), tile):
        first, last = keys(begin, end)
        out[begin:end] = x[first:last].sum(0)
    return out


@pytest.mark.parametrize(
    ("rows", "row_tile", "key_tile"),
    [
        (64, 16, 8),  # every tile full
        (64, 16, 16),  # one key tile per row tile
        (64, 12, 8),  # the bound is no multiple of the key tile: partial key tiles
        (60, 16, 8),  # a partial row tile
    ],
)
def test_loop_up_to_the_tile_end(rows: int, row_tile: int, key_tile: int) -> None:
    x = torch.randn(rows, 16)
    check_kernel(
        _kernel(up_to_tile_end, row_tile, key_tile),
        lambda a: _expected(a, row_tile, lambda _, end: (0, end)),
        [x],
    )


@pytest.mark.parametrize(
    ("window", "tile", "row_tile"),
    [(16, 8, 16), (20, 8, 16), (8, 8, 16), (0, 8, 16), (24, 8, 8)],
)
def test_loop_from_a_value(window: int, tile: int, row_tile: int) -> None:
    """A lower bound that is a value, clamped to 0 and aligned down."""
    x = torch.randn(64, 16)
    check_kernel(
        _kernel(behind_window, row_tile, tile),
        lambda a, *_: _expected(
            a,
            row_tile,
            lambda begin, end: (max((begin - window) // tile * tile, 0), end),
        ),
        [x, hl.constexpr(window), hl.constexpr(tile)],
    )


def test_loop_to_a_minimum_runs_no_key_past_it() -> None:
    x = torch.randn(64, 16)
    check_kernel(
        _kernel(capped, 16, 8),
        lambda a, *_: _expected(a, 16, lambda _, end: (0, min(end, 24))),
        [x, hl.constexpr(24)],
    )


def test_a_lower_bound_proven_in_the_tensor_is_not_refused() -> None:
    """Negative: from the second row tile on, the same window stays in the tensor."""
    x = torch.randn(64, 16)

    check_kernel(
        _kernel(behind_window, 16, 8),
        lambda a, *_: _expected(a, 16, lambda b, e: (max(b - 16, 0), e)),
        [x, hl.constexpr(16), hl.constexpr(8)],
    )


def test_a_loop_that_spans_nothing_runs_no_tile() -> None:
    """Regression: ``ZeroDivisionError`` in the proof of full tiles: the tile of a
    loop whose span is 0 has size 0."""
    x = torch.randn(32, 8)
    check_kernel(
        _kernel(zero_span, 8, 4),
        lambda a, *_: torch.zeros_like(a),
        [x, hl.constexpr(0)],
    )


def test_a_loop_over_part_of_its_tile() -> None:
    """The same kernel with a span, from the same review."""
    x = torch.randn(32, 8)

    def expected(a: torch.Tensor, _: object) -> torch.Tensor:
        out = torch.zeros_like(a)
        for begin in range(0, 32, 8):
            out[begin : begin + 8] = a[begin : begin + 8].sum(0)
        return out

    check_kernel(_kernel(zero_span, 8, 4), expected, [x, hl.constexpr(8)])


def behind_dynamic(x: torch.Tensor) -> torch.Tensor:
    dim = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, dim], dtype=x.dtype)
        for tn in hl.tile(tm.begin - 16, tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def three_levels(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.end):
            for tk in hl.tile(tn.begin - 8, tn.end):
                acc = acc + x[tk, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def from_tile_id(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.id * 16 - 16, tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def from_tile_id_small(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.id - 1, tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def from_runtime(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(n, tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def from_runtime_scaled(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(n * 8, tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def from_host_offset(x: torch.Tensor, n: int) -> torch.Tensor:
    offset = n - 8
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(offset, tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def store_from_runtime(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tm in hl.tile(x.size(0)):
        for tn in hl.tile(n, tm.end):
            out[tn, :] = x[tm, :].sum(0)[None, :]
    return out


def from_clamped_constexpr(x: torch.Tensor, n: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(max(n, 0), tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def from_tile_difference(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.end - tm.begin, tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def from_tile_remainder(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.begin - tm.begin // 16 * 16, tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def from_grid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for i in hl.grid(-1, 1):
            for tn in hl.tile(i * 8, tm.end):
                acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def from_runtime_grid(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for i in hl.grid(n, n + 2):
            for tn in hl.tile(i * 8, tm.end):
                acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def from_grid_at_zero(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for i in hl.grid(0, 2):
            for tn in hl.tile(i * 8, tm.end):
                acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def top_level_grid(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for i in hl.grid(-2, 2):
        for tn in hl.tile(i * 8, x.size(0)):
            out[tn, :] = x[tn, :]
    return out


def _dynamic_kernel(fn: object, *block_sizes: int) -> object:
    return helion.kernel(
        fn, backend="mlir", static_shapes=False, config=_config(*block_sizes)
    )


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("make", "fn", "blocks", "extra"),
    [
        (_dynamic_kernel, behind_dynamic, (16, 8), ()),
        (_kernel, three_levels, (8, 4, 4), ()),
        (_kernel, from_tile_id, (16, 8), ()),
        (_kernel, from_tile_id_small, (16, 1), ()),
        (_kernel, from_runtime, (16, 8), (-8,)),
        (_kernel, from_runtime_scaled, (16, 8), (-1,)),
        (_kernel, from_host_offset, (16, 8), (0,)),
        (_kernel, store_from_runtime, (16, 8), (-8,)),
        (_kernel, from_grid, (16, 8), ()),
        (_kernel, from_runtime_grid, (16, 8), (-1,)),
        (_kernel, top_level_grid, (8,), ()),
        # A bound the facts cannot tell is in the tensor is refused too: a positive one.
        (_kernel, from_runtime, (16, 8), (8,)),
    ],
    ids=[
        "dynamic-shapes",
        "three-levels",
        "tile-id",
        "tile-id-small",
        "runtime",
        "runtime-scaled",
        "host-offset",
        "store",
        "grid",
        "runtime-grid",
        "top-level-grid",
        "runtime-positive",
    ],
)
def test_a_lower_bound_not_proven_in_the_tensor_is_refused(
    make: object, fn: object, blocks: tuple[int, ...], extra: tuple[int, ...]
) -> None:
    """Regression: wrong rows with no error, and for the store a write before the
    buffer (``malloc(): invalid size``). The refusal names the alternative."""
    x = torch.randn(64, 8)
    with pytest.raises(
        UnsupportedOperationError, match=r"(?s)before the tensor.*max\("
    ):
        make(fn, *blocks)(x, *extra)


def test_a_grid_from_zero_still_lowers() -> None:
    x = torch.randn(64, 8)

    def expected(a: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(a)
        for begin in range(0, 64, 16):
            out[begin : begin + 16] = sum(
                a[i * 8 : begin + 16].sum(0) for i in range(2)
            )
        return out

    check_kernel(_kernel(from_grid_at_zero, 16, 8), expected, [x], atol=1e-3)


def scalar_runtime_index(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        out[tm, :] = x[n, :][None, :]
    return out


def test_a_runtime_negative_scalar_index_counts_from_the_end() -> None:
    """From the review: all zeros at ``n = -4`` before positions were checked."""
    x = torch.randn(64, 8) + 2.0
    check_kernel(
        _kernel(scalar_runtime_index, 16),
        lambda a, _: a[60].expand(64, -1),
        [x, -4],
    )


@pytest.mark.parametrize("n", [-8, 8, 0])
def test_a_constexpr_bound_clamped_at_zero_lowers(n: int) -> None:
    """The refusal's alternative, for a bound that is known when compiling."""
    x = torch.randn(64, 8)
    check_kernel(
        _kernel(from_clamped_constexpr, 16, 8),
        lambda a, _: _expected(a, 16, lambda _b, end: (min(max(n, 0), end), end)),
        [x, hl.constexpr(n)],
    )


def from_clamped_runtime(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(max(n, 0), tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


@pytest.mark.isolated
def test_a_clamped_runtime_bound_is_a_clear_error() -> None:
    """Pinned: Helion's trace does not pass ``n`` to the kernel when only ``max(n, 0)``
    uses it, so a runtime bound has no clamped form here."""
    with pytest.raises(UnsupportedOperationError, match="tile expression"):
        _kernel(from_clamped_runtime, 16, 8)(torch.randn(64, 8), 8)


@pytest.mark.isolated
@pytest.mark.parametrize(
    "fn", [from_tile_difference, from_tile_remainder], ids=["difference", "remainder"]
)
def test_bounds_the_facts_cannot_prove_stay_refused(fn: object) -> None:
    """Pinned: these never go negative, but the facts do not see it (a false refusal,
    accepted: refusing is safe). This test can change if the facts learn more."""
    with pytest.raises(UnsupportedOperationError, match="before the tensor"):
        _kernel(fn, 16, 8)(torch.randn(64, 8))


# --- index_facts ---------------------------------------------------------------------


def _facts_of(build: object, leaf: IndexFacts) -> IndexFacts:
    """``build(position, index)``'s facts, ``position`` a value with facts ``leaf``."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        index = ir.IndexType.get()
        with ir.InsertionPoint(module.body):
            position = arith.ConstantOp(index, 12345).result

            def constant(value: int) -> ir.Value:
                return arith.ConstantOp(index, value).result

            result = build(position, constant)
            return index_facts(result, lambda v: leaf if v == position else None)


_LEAF = IndexFacts(64, 0, 448)


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda p, c: arith.AddIOp(p, c(64)).result, IndexFacts(64, 64, 512)),
        (lambda p, c: arith.AddIOp(p, c(32)).result, IndexFacts(32, 32, 480)),
        (lambda p, c: arith.SubIOp(p, c(128)).result, IndexFacts(64, -128, 320)),
        (lambda p, c: arith.MulIOp(p, c(2)).result, IndexFacts(128, 0, 896)),
        (lambda p, c: arith.MulIOp(c(-1), p).result, IndexFacts(64, -448, 0)),
        (lambda p, c: arith.FloorDivSIOp(p, c(32)).result, IndexFacts(2, 0, 14)),
        (lambda p, c: arith.FloorDivSIOp(p, c(128)).result, IndexFacts(1, 0, 3)),
        (lambda p, c: arith.MaxSIOp(p, c(0)).result, IndexFacts(64, 0, 448)),
        (lambda p, c: arith.MinSIOp(p, c(96)).result, IndexFacts(32, 0, 96)),
        (
            lambda p, c: arith.MaxSIOp(arith.XOrIOp(p, c(64)).result, c(32)).result,
            IndexFacts(1, 32, None),
        ),
        (lambda p, c: arith.XOrIOp(p, c(1)).result, IndexFacts()),
    ],
    ids=[
        "add",
        "add-other",
        "sub",
        "mul",
        "mul-negative",
        "fdiv",
        "fdiv-more",
        "max",
        "min",
        "max-of-unknown",
        "other",
    ],
)
def test_index_facts(build: object, expected: IndexFacts) -> None:
    assert _facts_of(build, _LEAF) == expected


def behind_any_row(x: torch.Tensor, window: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        acc = hl.zeros([tm, x.size(1)], dtype=x.dtype)
        for tn in hl.tile(tm.begin - window, tm.end):
            acc = acc + x[tn, :].sum(0)[None, :]
        out[tm, :] = acc
    return out


def test_a_lower_bound_that_may_be_negative_is_refused() -> None:
    """Regression: the first row tiles read before the tensor, with no error and
    wrong rows."""
    with pytest.raises(UnsupportedOperationError, match="before the tensor"):
        _kernel(behind_any_row, 16, 8)(torch.randn(64, 16), hl.constexpr(16))
