"""Accesses at runtime positions: as torch, negative from the end, out of bounds an
``IndexError``, never a store outside the tensor."""

from __future__ import annotations

from unittest import mock

import helion
import helion.language as hl
import pytest
import torch

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError


def put_row(x: torch.Tensor, cache: torch.Tensor, pos: int) -> torch.Tensor:
    m, d = x.shape
    for tm, td in hl.tile([m, d]):
        cache[tm, pos, td] = x[tm, td] * 1.0
    return cache


def get_row(cache: torch.Tensor, pos: int) -> torch.Tensor:
    m, s, d = cache.shape
    out = torch.empty(m, d, dtype=cache.dtype, device=cache.device)
    for tm, td in hl.tile([m, d]):
        out[tm, td] = cache[tm, pos, td] * 1.0
    return out


def put_next_row(x: torch.Tensor, cache: torch.Tensor, pos: int) -> torch.Tensor:
    m, d = x.shape
    for tm, td in hl.tile([m, d]):
        cache[tm, pos + 1, td] = x[tm, td] * 1.0
    return cache


def guarded_put(x: torch.Tensor, cache: torch.Tensor, pos: int) -> torch.Tensor:
    """Out of bounds where the guard doesn't run: not an error."""
    m, d = x.shape
    for tm, td in hl.tile([m, d]):
        if pos < 8:
            cache[tm, pos, td] = x[tm, td] * 1.0
    return cache


def _kernel(fn: object, blocks: tuple[int, ...] = (1, 16)) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=list(blocks)),
    )


def _view() -> tuple[torch.Tensor, torch.Tensor]:
    """A cache of 8 rows of 64, a view with guard rows of the same allocation around it."""
    base = torch.zeros(10, 64, 16)
    return base, base[1:9]


@pytest.mark.parametrize("pos", [0, 1, 63, -1, -2, -64])
def test_store_in_bounds_matches_torch(pos: int) -> None:
    x = torch.randn(8, 16)
    base, cache = _view()
    _kernel(put_row)(x, cache, pos)
    expected = torch.zeros(10, 64, 16)
    expected[1:9][:, pos] = x
    assert torch.equal(base, expected)


@pytest.mark.parametrize("pos", [64, 65, -65, 1000, -1000])
def test_store_out_of_bounds_raises(pos: int) -> None:
    """Regression: 16 elements landed outside the view at 64 and 65, and -65 corrupted
    the heap (``double free or corruption``, process aborted)."""
    x = torch.randn(8, 16)
    base, cache = _view()
    with pytest.raises(IndexError, match=f"index {pos} is out of bounds"):
        _kernel(put_row)(x, cache, pos)
    assert not base.any()


@pytest.mark.parametrize("pos", [0, 63, -1, -64])
def test_load_in_bounds_matches_torch(pos: int) -> None:
    cache = torch.randn(8, 64, 16)
    assert torch.equal(_kernel(get_row)(cache, pos), cache[:, pos])


@pytest.mark.parametrize("pos", [64, -65])
def test_load_out_of_bounds_raises(pos: int) -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        _kernel(get_row)(torch.randn(8, 64, 16), pos)


def test_computed_position() -> None:
    x = torch.randn(8, 16)
    base, cache = _view()
    _kernel(put_next_row)(x, cache, 3)
    assert torch.equal(cache[:, 4], x)
    with pytest.raises(IndexError, match="index 64 is out of bounds"):
        _kernel(put_next_row)(x, cache, 63)


def test_repeated_calls_are_checked() -> None:
    """A later call with another position takes the same checks as the first."""
    kernel = _kernel(put_row)
    x = torch.randn(8, 16)
    _, cache = _view()
    kernel(x, cache, 1)
    kernel(x, cache, 1)
    with pytest.raises(IndexError):
        kernel(x, cache, 64)


def test_guarded_position_counts_from_the_end() -> None:
    x = torch.randn(8, 16)
    _, cache = _view()
    _kernel(guarded_put)(x, cache, -1)
    assert torch.equal(cache[:, -1], x)


def test_guarded_position_is_checked_though_it_may_not_run() -> None:
    """A guard can't be proved: -65 under a true guard crashed. So out of bounds is an
    error also where the guard would be false: conservative."""
    x = torch.randn(8, 16)
    base, cache = _view()
    with pytest.raises(IndexError, match="out of bounds"):
        _kernel(guarded_put)(x, cache, 64)
    assert not base.any()


def put_host_computed(x: torch.Tensor, c: torch.Tensor, p: int) -> torch.Tensor:
    m, d = x.shape
    i = p + 1
    for tm, td in hl.tile([m, d]):
        c[tm, i, td] = x[tm, td] * 1.0
    return c


def put_rebound(x: torch.Tensor, c: torch.Tensor, p: int) -> torch.Tensor:
    m, d = x.shape
    p = p + 1
    for tm, td in hl.tile([m, d]):
        c[tm, p, td] = x[tm, td] * 1.0
    return c


def put_host_computed_plus(x: torch.Tensor, c: torch.Tensor, p: int) -> torch.Tensor:
    m, d = x.shape
    i = p + 1
    for tm, td in hl.tile([m, d]):
        c[tm, i + 1, td] = x[tm, td] * 1.0
    return c


def put_reversed(x: torch.Tensor, c: torch.Tensor, p: int) -> torch.Tensor:
    m, d = x.shape
    i = 63 - p
    for tm, td in hl.tile([m, d]):
        c[tm, i, td] = x[tm, td] * 1.0
    return c


@pytest.mark.parametrize(
    ("fn", "valid", "at", "invalid"),
    [
        (put_host_computed, 3, 4, 63),
        (put_rebound, 3, 4, 63),
        (put_host_computed_plus, 3, 5, 62),
        (put_reversed, 3, 60, 128),
    ],
    ids=["host_plus_one", "rebound", "host_plus_two", "reversed"],
)
def test_host_computed_positions(fn: object, valid: int, at: int, invalid: int) -> None:
    """Regression (review): a scalar computed on the host went unchecked, 16 elements
    outside the view at 63 and ``free(): invalid size`` at -66."""
    x = torch.randn(8, 16)
    base, cache = _view()
    kernel = _kernel(fn)
    kernel(x, cache, valid)
    assert torch.equal(cache[:, at], x)
    base.zero_()
    with pytest.raises(IndexError, match="out of bounds"):
        kernel(x, cache, invalid)
    assert not base.any()


def put_floor(x: torch.Tensor, c: torch.Tensor, p: int) -> torch.Tensor:
    m, d = x.shape
    for tm, td in hl.tile([m, d]):
        c[tm, p // 2, td] = x[tm, td] * 1.0
    return c


def put_mod(x: torch.Tensor, c: torch.Tensor, p: int) -> torch.Tensor:
    m, d = x.shape
    for tm, td in hl.tile([m, d]):
        c[tm, p % 8, td] = x[tm, td] * 1.0
    return c


@pytest.mark.parametrize(("fn", "at"), [(put_floor, 10), (put_mod, 4)])
def test_floor_and_mod_positions(fn: object, at: int) -> None:
    """Regression (review): ``NameError: FloorDiv``, ``PythonMod`` at the check."""
    x = torch.randn(8, 16)
    _, cache = _view()
    _kernel(fn)(x, cache, 20)
    assert torch.equal(cache[:, at], x)


def load_with_none(c: torch.Tensor, p: int) -> torch.Tensor:
    s, d = c.shape
    out = torch.empty(1, d, dtype=c.dtype)
    for t0, td in hl.tile([1, d]):
        out[t0, td] = c[None, p, td] * 1.0
    return out


def test_none_in_the_index_keeps_the_dim() -> None:
    """Regression (review): ``None`` shifted the checked dim: false IndexError."""
    cache = torch.randn(64, 16)
    kernel = _kernel(load_with_none)
    assert torch.equal(kernel(cache, 20)[0], cache[20])
    with pytest.raises(IndexError, match="out of bounds"):
        kernel(cache, 64)


def put_tile_plus(x: torch.Tensor, c: torch.Tensor, p: int) -> torch.Tensor:
    m, d = x.shape
    for tm, td in hl.tile([m, d]):
        c[tm, tm.begin + p, td] = x[tm, td] * 1.0
    return c


def test_tile_position_plus_a_scalar_is_refused() -> None:
    """Regression (review): unchecked, outside the view at 100 and 60."""
    x = torch.randn(8, 16)
    _, cache = _view()
    with pytest.raises(UnsupportedOperationError, match="tile position"):
        _kernel(put_tile_plus)(x, cache, 60)


def put_literal(x: torch.Tensor, c: torch.Tensor, pos: hl.constexpr) -> torch.Tensor:
    m, d = x.shape
    for tm, td in hl.tile([m, d]):
        c[tm, pos, td] = x[tm, td] * 1.0
    return c


@pytest.mark.parametrize("pos", [64, -65])
def test_constant_out_of_bounds_is_refused(pos: int) -> None:
    """Regression (review): a constant 64 on 64 rows stored outside, on base too."""
    x = torch.randn(8, 16)
    _, cache = _view()
    with pytest.raises(UnsupportedOperationError, match="out of bounds"):
        _kernel(put_literal)(x, cache, hl.constexpr(pos))


def put_alias_row(x: torch.Tensor, c: torch.Tensor, pos: int) -> torch.Tensor:
    """``v`` is ``c`` reshaped: read through it, it's an alias with no argument."""
    m, d = x.shape
    v = c.view(m, 64, d)
    out = torch.empty_like(x)
    for tm, td in hl.tile([m, d]):
        out[tm, td] = x[tm, td] + v[tm, pos, td]
    return out


@pytest.mark.parametrize("pos", [0, 63, -1, -64])
def test_alias_in_bounds(pos: int) -> None:
    x, c = torch.randn(8, 16), torch.randn(8 * 64, 16)
    out = _kernel(put_alias_row)(x, c, pos)
    assert torch.equal(out, x + c.view(8, 64, 16)[:, pos])


@pytest.mark.parametrize("pos", [64, -65])
def test_alias_out_of_bounds_raises(pos: int) -> None:
    """Regression: the reshaped alias had no argument and went unchecked."""
    x, c = torch.randn(8, 16), torch.randn(8 * 64, 16)
    with pytest.raises(IndexError, match="'v' with size 64"):
        _kernel(put_alias_row)(x, c, pos)


def test_alias_of_a_runtime_shape_out_of_bounds_raises() -> None:
    """Without static shapes the view is sized at the call too, or refused."""
    kernel = helion.kernel(
        put_alias_row, backend="mlir", config=helion.Config(block_sizes=[1, 16])
    )
    with pytest.raises((IndexError, UnsupportedOperationError)):
        kernel(torch.randn(8, 16), torch.randn(8 * 64, 16), 64)


def test_alias_size_refused_when_the_call_cannot_read_it() -> None:
    """No argument and no static size: refused rather than unchecked."""
    from helion_mlir_backend._compiler.mlir.analysis import signature

    access = signature.ScalarIndex("v", 1, None, None)
    known = mock.Mock(refs={"c": None})
    with pytest.raises(UnsupportedOperationError, match="reshaped view of a runtime"):
        signature._check_alias_size(known, access)
    signature._check_alias_size(mock.Mock(refs={"v": None}), access)


def add_last_row(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    out = torch.empty_like(x)
    for tm, td in hl.tile([n, d]):
        out[tm, td] = x[tm, td] + c[n - 1, td]
    return out


def test_empty_tensor_is_not_checked() -> None:
    """Regression (review): dynamic shapes at ``n = 0``: "index -1 ... size 0", where
    the kernel is right and stores nothing."""
    kernel = helion.kernel(
        add_last_row,
        backend="mlir",
        static_shapes=False,
        config=helion.Config(block_sizes=[1, 16]),
    )
    assert kernel(torch.zeros(0, 16), torch.zeros(0, 16)).shape == (0, 16)
    x, c = torch.randn(8, 16), torch.randn(8, 16)
    assert torch.equal(kernel(x, c), x + c[-1])


def put_in_inner_tile(
    x: torch.Tensor, c: torch.Tensor, s: int, pos: int
) -> torch.Tensor:
    m, d = x.shape
    for tm, td in hl.tile([m, d]):
        for _ts in hl.tile(s):
            c[tm, pos, td] = x[tm, td] * 1.0
    return c


def test_position_in_a_loop_of_no_iterations_is_checked() -> None:
    """Conservative, as under a guard: the store wouldn't run at ``s = 0``, but the
    position is out of bounds, so the call raises."""
    kernel = helion.kernel(
        put_in_inner_tile, backend="mlir", config=helion.Config(block_sizes=[1, 16, 8])
    )
    x = torch.randn(8, 16)
    _, cache = _view()
    with pytest.raises(IndexError, match="index 70 is out of bounds"):
        kernel(x, cache, 0, 70)
