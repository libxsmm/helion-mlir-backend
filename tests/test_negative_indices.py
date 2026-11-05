"""Negative indices in subscripts.

A computed index that is negative at the start of its loops (``x[ti.begin -
1, tj]``, ``x[ti, tj.index - 1]``, ``x[ti, tj.index + kw - 1]``) read before
the tensor: the scalar form garbage (normalized error 4e27 in the conv
feasibility probe), the tile form wrapped around as torch indexing does.
Helion leaves such indices undefined (its codegen may add them to the
address unchecked; its eager reference mode wraps). A shifted window usually
means zero padding: Helion's masked load (``hl.load(x, [t.index - 1],
extra_mask=t.index >= 1)``) or an ``if`` guard gives it, the bare subscript
does not. So the backend rejects unconditional, unmasked accesses whose
index is provably negative at compile time (``UnsupportedOperationError``)
and keeps masked and guarded ones (their tests below were right before and
stay right). Literal negative integers count from the end, as in Python (before:
read before the tensor too).
"""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch
from torch import Tensor
import torch.nn.functional as F

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

# --- The probe's kernels (p2_neg.py) ------------------------------------------


def shift_row(x: Tensor) -> Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for ti, tj in hl.tile([m, n]):
        out[ti.begin, tj] = x[ti.begin - 1, tj]
    return out


def shift_col(x: Tensor) -> Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for ti, tj in hl.tile([m, n]):
        out[ti, tj] = x[ti, tj.index - 1]
    return out


def shift_col_kw(x: Tensor) -> Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for ti, tj in hl.tile([m, n]):
        for kw in hl.static_range(2):
            out[ti, tj] = x[ti, tj.index + kw - 1]
    return out


def store_before_start(x: Tensor) -> Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for ti, tj in hl.tile([m, n]):
        out[ti.begin - 1, tj] = x[ti.begin, tj]
    return out


# --- Controls: the same windows where the loop starts late enough --------------


def shift_row_from_one(x: Tensor) -> Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for ti, tj in hl.tile([1, 0], [m, n]):
        out[ti.begin, tj] = x[ti.begin - 1, tj]
    return out


def shift_col_from_one(x: Tensor) -> Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for ti, tj in hl.tile([0, 1], [m, n]):
        out[ti, tj] = x[ti, tj.index - 1]
    return out


def last_row(x: Tensor) -> Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for ti, tj in hl.tile([m, n]):
        out[ti, tj] = x[ti, tj] - x[-1, tj]
    return out


def _x() -> Tensor:
    return torch.arange(8 * 64, dtype=torch.float32).view(8, 64)


def _kernel(fn: object, block_sizes: list[int]) -> helion.Kernel:
    return helion.kernel(
        fn, backend="mlir", config=helion.Config(block_sizes=block_sizes)
    )


@pytest.mark.parametrize(
    ("fn", "block_sizes"),
    [
        (shift_row, [1, 32]),
        (shift_col, [4, 32]),
        (shift_col_kw, [4, 32]),
        (store_before_start, [1, 32]),
    ],
    ids=[
        "scalar_begin_minus_1",
        "tile_index_minus_1",
        "tile_index_plus_kw_minus_1",
        "store_begin_minus_1",
    ],
)
def test_negative_index_is_rejected(fn: object, block_sizes: list[int]) -> None:
    """Unfixed: garbage rows (scalar), torch-style wraparound (tile), or a
    store before the tensor start."""
    with pytest.raises(UnsupportedOperationError, match="negative"):
        _kernel(fn, block_sizes)(_x())


@pytest.mark.parametrize(
    ("fn", "block_sizes", "reference"),
    [
        (
            shift_row_from_one,
            [1, 32],
            lambda x: torch.cat([torch.zeros_like(x[:1]), x[:-1]]),
        ),
        (
            shift_col_from_one,
            [4, 32],
            lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1),
        ),
    ],
    ids=["scalar_from_one", "tile_from_one"],
)
def test_shift_within_bounds_is_kept(
    fn: object, block_sizes: list[int], reference: object
) -> None:
    """The same windows on loops starting at 1: never negative, computed."""
    x = _x()
    torch.testing.assert_close(_kernel(fn, block_sizes)(x), reference(x))


def test_negative_literal_counts_from_end() -> None:
    """``x[-1, tj]`` is the last row (unfixed: the row before the tensor)."""
    x = _x()
    torch.testing.assert_close(_kernel(last_row, [2, 32])(x), x - x[-1])


# --- Masked and guarded windows: valid, kept (right on the base commit too) ---
# The review's kernels and the conv probe's p2_pad.py ``c``/``d``.


def masked_prev(x: Tensor) -> Tensor:
    (n,) = x.shape
    out = torch.empty_like(x)
    for t in hl.tile(n):
        prev = hl.load(x, [t.index - 1], extra_mask=t.index >= 1)
        out[t] = x[t] - prev
    return out


def guarded_scalar(x: Tensor) -> Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for ti, tj in hl.tile([m, n]):
        if ti.begin >= 1:
            out[ti.begin, tj] = x[ti.begin - 1, tj]
    return out


def grid_prev(x: Tensor) -> Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for i in hl.grid(m):
        if i > 0:
            for tj in hl.tile(n):
                out[i, tj] = x[i - 1, tj]
    return out


def conv_padh_guard(x: Tensor, w: Tensor) -> Tensor:
    """3x3 conv, padding 1: x W-padded on the host, H padded by two guards."""
    n, c, h, wdp = x.shape
    kh_, kw_, oc, _ = w.shape
    KH = hl.specialize(kh_)
    KW = hl.specialize(kw_)
    ow = wdp - KW + 1
    out = torch.empty([n, oc, h, ow], dtype=x.dtype, device=x.device)
    for tn, toc, toh, tow in hl.tile([n, oc, h, ow]):
        acc = hl.zeros([toc, tow], dtype=torch.float32)
        for tic in hl.tile(c):
            for kh in hl.static_range(KH):
                if toh.begin + kh >= 1:
                    if toh.begin + kh <= h:
                        for kw in hl.static_range(KW):
                            acc = hl.dot(
                                w[kh, kw, toc, tic],
                                x[tn.begin, tic, toh.begin + kh - 1, tow.index + kw],
                                acc=acc,
                            )
        out[tn.begin, toc, toh.begin, tow] = acc.to(x.dtype)
    return out


def conv_padh_guard1(x: Tensor, w: Tensor) -> Tensor:
    """The same with only the top row guarded (the rest padded on the host)."""
    n, c, hp, wdp = x.shape
    kh_, kw_, oc, _ = w.shape
    KH = hl.specialize(kh_)
    KW = hl.specialize(kw_)
    ow = wdp - KW + 1
    oh = hp - 1
    out = torch.empty([n, oc, oh, ow], dtype=x.dtype, device=x.device)
    for tn, toc, toh, tow in hl.tile([n, oc, oh, ow]):
        acc = hl.zeros([toc, tow], dtype=torch.float32)
        for tic in hl.tile(c):
            for kh in hl.static_range(KH):
                if toh.begin + kh >= 1:
                    for kw in hl.static_range(KW):
                        acc = hl.dot(
                            w[kh, kw, toc, tic],
                            x[tn.begin, tic, toh.begin + kh - 1, tow.index + kw],
                            acc=acc,
                        )
        out[tn.begin, toc, toh.begin, tow] = acc.to(x.dtype)
    return out


def _shift_rows(x: Tensor) -> Tensor:
    return torch.cat([torch.zeros_like(x[:1]), x[:-1]])


@pytest.mark.parametrize(
    ("fn", "block_sizes", "x", "reference"),
    [
        (
            masked_prev,
            [16],
            torch.arange(64, dtype=torch.float32) + 100,
            lambda x: x - _shift_rows(x),
        ),
        (guarded_scalar, [1, 32], _x(), _shift_rows),
        (grid_prev, [32], _x(), _shift_rows),
    ],
    ids=["masked_load", "guarded_scalar", "guarded_grid_index"],
)
def test_masked_or_guarded_window_is_kept(
    fn: object, block_sizes: list[int], x: Tensor, reference: object
) -> None:
    torch.testing.assert_close(_kernel(fn, block_sizes)(x), reference(x))


# 3x3 conv, padding 1, f32: N=1, C=OC=32, H=4, W=32, blocks [1, 32, 1, 32, 32].
_CONV_PADS = {
    "conv_two_guards": (conv_padh_guard, (1, 1)),
    "conv_top_guard": (conv_padh_guard1, (1, 1, 0, 1)),
}


@pytest.mark.parametrize("name", list(_CONV_PADS))
def test_guarded_conv_padding_is_kept(name: str) -> None:
    fn, host_pad = _CONV_PADS[name]
    torch.manual_seed(0)
    x = torch.randn(1, 32, 4, 32)
    w = torch.randn(32, 32, 3, 3)
    kernel = _kernel(fn, [1, 32, 1, 32, 32])
    out = kernel(F.pad(x, host_pad), w.permute(2, 3, 0, 1).contiguous())
    expected = F.conv2d(x.double(), w.double(), padding=1)
    error = (out.double() - expected).abs().max() / expected.abs().max()
    assert error.item() < 1e-5
