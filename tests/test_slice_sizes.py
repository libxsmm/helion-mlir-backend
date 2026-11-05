"""``.size()`` of a full ``:`` slice is the size of the sliced dim.

Helion pads on-device tensor factories (``hl.zeros([tile, dim])``) to powers
of two. A dim used in one then had a reduction block of
next_pow2(dim), and every ``.size()`` of a ``:`` slice of it returned the
padded size; the MLIR backend turns the padding off. Dynamic dims keep
their run-time size (``test_dynamic_shapes.py``: ``row_size``,
``swap_sizes_view``)."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

import helion_mlir_backend  # noqa: F401


def with_zeros(x: torch.Tensor) -> torch.Tensor:
    heads, seq, dim = x.shape
    hl.specialize(dim)
    out = torch.empty_like(x)
    for h in hl.grid(heads):
        for tm in hl.tile(seq):
            acc = hl.zeros([tm, dim], dtype=torch.float32)
            t = x[h, tm, :]
            out[h, tm, :] = acc + t * t.size(1)
    return out


def zeros_after(x: torch.Tensor) -> torch.Tensor:
    heads, seq, dim = x.shape
    hl.specialize(dim)
    out = torch.empty_like(x)
    for h in hl.grid(heads):
        for tm in hl.tile(seq):
            t = x[h, tm, :]
            n = t.size(1)
            acc = hl.zeros([tm, dim], dtype=torch.float32)
            out[h, tm, :] = acc + t * n
    return out


@pytest.mark.parametrize("static_shapes", [True, False])
@pytest.mark.parametrize("fn", [with_zeros, zeros_after], ids=lambda fn: fn.__name__)
def test_size_of_a_slice_of_a_dim_in_a_factory(fn: object, static_shapes: bool) -> None:
    """D=40 is also in ``hl.zeros([tm, dim])``: ``t.size(1)`` was 64."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 40)
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=static_shapes,
        config=helion.Config(block_sizes=[8]),
    )
    torch.testing.assert_close(kernel(x), x * 40, atol=1e-5, rtol=1e-5)
