"""``helion_mlir_utils`` row sums, row dots and column sums at the shapes that
aborted LLVM's x86 code generation (AI-bench L2/14), and their contracts."""

from __future__ import annotations

import os

from helion_mlir_utils import column_sum
from helion_mlir_utils import row_dot
from helion_mlir_utils import row_sum
import pytest
import torch

_RAGGED = [(33, 65, 8), (33, 129, 8), (1025, 65, 64)]


@pytest.mark.isolated
@pytest.mark.parametrize(("m", "n", "threads"), _RAGGED)
def test_row_sum_ragged_rows_and_8k_plus_1_columns(
    m: int, n: int, threads: int
) -> None:
    """bf16 row sums with partial row tiles and 8k+1 > 32 columns aborted LLVM's
    x86 code generation."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    torch.manual_seed(0)
    x = torch.rand(m, n).to(torch.bfloat16)
    exact = x.double().sum(1, keepdim=True)
    assert (row_sum(x).double() - exact).abs().max() / exact.abs().max() < 2.0**-8


@pytest.mark.isolated
@pytest.mark.parametrize(("m", "n", "threads"), _RAGGED)
def test_row_dot_ragged_rows_and_8k_plus_1_columns(
    m: int, n: int, threads: int
) -> None:
    """The shapes of the row sum regression: partial row tiles aborted LLVM's x86
    code generation here too."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    torch.manual_seed(0)
    x = torch.rand(m, n).to(torch.bfloat16)
    v = torch.rand(n)
    exact = x.double() @ v.double()
    out = row_dot(x, v)[:, 0].double()
    assert (out - exact).abs().max() / exact.abs().max() < 2.0**-8


@pytest.mark.isolated
@pytest.mark.parametrize(
    ("k", "threads"),
    [
        (577, 1),
        (641, 1),
        (1089, 1),
        (1153, 1),
        (4161, 8),
        (4225, 8),
        (32833, 64),
        (32897, 64),
    ],
)
def test_column_sum_of_one_row_with_a_partial_column_tile(k: int, threads: int) -> None:
    """The column sum of a one-row bf16 weight with a partial column tile aborted
    LLVM's x86 code generation."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    x = torch.rand(1, k).to(torch.bfloat16)
    assert torch.equal(column_sum(x), x[0].float())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_column_sum_of_one_row_is_new_memory(dtype: torch.dtype) -> None:
    """As a sum: not a view of the input, and +0.0 where the row has -0.0."""
    x = torch.tensor([[-0.0, 1.0, -2.0]]).to(dtype)
    out = column_sum(x)
    assert out.dtype == torch.float32
    assert torch.equal(out, x.float()[0])
    assert torch.signbit(out).tolist() == [False, False, True]
    x[0, 1] = 5.0
    assert out[1].item() == 1.0


def test_row_dot_rejects_bad_operands() -> None:
    with pytest.raises(ValueError, match="row_dot"):
        row_dot(torch.ones(4, 8), torch.ones(4))
    with pytest.raises(ValueError, match="f32 vector"):
        row_dot(torch.ones(4, 8), torch.ones(8, dtype=torch.bfloat16))
