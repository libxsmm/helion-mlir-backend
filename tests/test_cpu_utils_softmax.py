"""``helion_mlir_utils.softmax`` over rows whose width is not a whole number
of 128-column chunks.

The kernel kept lane-wise ``[rows, 128]`` maxima and sums across chunks and
reduced them after the loop, so the zero padding of a partial last chunk
entered both: each padded lane added ``exp(-max)`` to the row sum.
AI-bench's L2/66 at 130x1001x515 (softmax of a [130, 515] GEMM output) was
off by 0.18 x max|ref|, by 1.0 for rows far below zero."""

from __future__ import annotations

from helion_mlir_utils import group_norm
from helion_mlir_utils import softmax
import pytest
import torch


@pytest.mark.parametrize(
    ("m", "n", "dtype", "shift"),
    [
        (130, 515, torch.float32, 0.0),  # L2/66's GEMM output, 130x1001x515
        (4, 129, torch.float32, -50.0),  # one padded chunk of 127 lanes
        (2, 200, torch.bfloat16, -50.0),
        (4, 256, torch.float32, -50.0),  # whole chunks only
        (3, 47, torch.float32, 0.0),  # one chunk narrower than 128
    ],
)
def test_softmax_partial_last_chunk(m, n, dtype, shift) -> None:
    torch.manual_seed(0)
    x = (torch.randn(m, n) + shift).to(dtype)
    out = softmax(x)
    ref = torch.softmax(x.double(), -1)
    error = ((out.double() - ref).abs().max() / ref.abs().max()).item()
    assert error < (1e-6 if dtype is torch.float32 else 1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softmax_one_column(dtype) -> None:
    """One column (AI-bench L2/66, 84, 99 with OUT_FEATURES = 1): the chunked
    kernel raised ControlFlowTensorMismatch. Softmax is 1, NaN for NaN and
    infinite inputs, as torch's."""
    x = torch.tensor([[0.5], [-3e4], [float("nan")], [float("inf")], [-float("inf")]])
    x = torch.cat([x, torch.randn(60, 1)]).to(dtype)
    out = softmax(x)
    ref = torch.softmax(x, -1)
    assert torch.equal(out.isnan(), ref.isnan())
    assert torch.equal(out[~ref.isnan()], ref[~ref.isnan()])


_SHAPES = [(32, 128), (64, 256), (16, 129), (33, 257), (2, 65), (1, 127), (5, 2)]


def _bits(t: torch.Tensor) -> torch.Tensor:
    """``t``'s bit pattern, so NaNs compare equal."""
    return t.to(torch.float64).nan_to_num(7.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", _SHAPES)
def test_softmax_in_place_is_the_out_of_place_result(
    shape: tuple[int, int], dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    x = (torch.randn(*shape) * 3).to(dtype)
    y = x.clone()
    pointer = y.data_ptr()
    out = softmax(y, in_place=True)
    assert out.data_ptr() == pointer
    assert torch.equal(_bits(out), _bits(softmax(x)))
    assert x.data_ptr() != pointer


def test_softmax_one_column_may_not_be_in_place() -> None:
    # One column is its own kernel: the result, not the input, is what counts.
    x = torch.randn(7, 1)
    assert torch.equal(softmax(x.clone(), in_place=True), softmax(x))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("m", "c", "groups"),
    [(32, 256, 16), (33, 256, 4), (64, 512, 512), (5, 96, 3), (2, 65, 5)],
)
def test_group_norm_in_place_is_the_out_of_place_result(
    m: int, c: int, groups: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    x = (torch.randn(m, c) * 3).to(dtype)
    weight, bias = torch.randn(c).to(dtype), torch.randn(c).to(dtype)
    y = x.clone()
    pointer = y.data_ptr()
    out = group_norm(y, groups, weight, bias, 1e-5, in_place=True)
    assert out.data_ptr() == pointer
    assert torch.equal(_bits(out), _bits(group_norm(x, groups, weight, bias, 1e-5)))


def test_out_of_place_leaves_the_input() -> None:
    x = torch.randn(32, 128)
    before = x.clone()
    assert softmax(x).data_ptr() != x.data_ptr()
    assert torch.equal(x, before)
