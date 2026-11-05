"""Tiles loading past their operand's end under a grid index: 512 rows padded to 96
columns aborted LLVM's instruction selection, and rows of 32k+1 elements read past
their end a masked bf16 load LLVM aborts on (AI-bench's L1/97 at D 66 and D 65)."""

from __future__ import annotations

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import lower_opt_with_amx

from helion_mlir_backend import generate_mlir


def _grid_padded_dots(
    q: torch.Tensor, k: torch.Tensor, padded_dim: hl.constexpr
) -> torch.Tensor:
    """Each row's sum of ``q @ k.T`` by heads, on AMX: the head dim's one tile
    padded, in VNNI layouts, as AI-bench's L1/97."""
    heads, seq, dim = q.shape
    hl.specialize(seq)
    hl.specialize(dim)
    out = torch.empty((heads, seq), dtype=torch.float32)
    for h in hl.grid(heads):
        for tile_m in hl.tile(seq):
            for tile_d in hl.tile(int(padded_dim)):
                q_tile = q[h, tile_m, tile_d]
                total = hl.zeros([tile_m], dtype=torch.float32)
                for tile_n in hl.tile(seq):
                    k_tile = k[h, tile_n, tile_d]
                    a3 = q_tile.reshape(q_tile.size(0), q_tile.size(1) // 2, 2)
                    b3 = k_tile.reshape(k_tile.size(0), k_tile.size(1) // 2, 2)
                    scores = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    scores = scores + torch.einsum(
                        "mcv,cnv->mn", a3, b3.permute(1, 0, 2)
                    )
                    total = total + scores.sum(-1)
                out[h, tile_m] = total
    return out


def _dots_kernel(block_m: int, padded_dim: int, block_n: int) -> helion.Kernel:
    return helion.kernel(
        _grid_padded_dots,
        backend="mlir",
        config=helion.Config(block_sizes=[block_m, padded_dim, block_n]),
    )


# 512 query rows padded to 96 columns in one vector aborted LLVM's instruction
# selection (SelectionDAG "too many operands"), AI-bench's L1/97 at D 66.
@pytest.mark.slow
@pytest.mark.isolated
def test_padded_tile_under_a_grid_index_runs() -> None:
    torch.manual_seed(0)
    q, k = (torch.randn(128, 2048, 66).bfloat16() for _ in range(2))
    out = _dots_kernel(512, 96, 64)(q, k, hl.constexpr(96))
    ref = (q.double() @ k.double().transpose(1, 2)).sum(-1)
    error = (out.double() - ref).abs().max() / ref.abs().max()
    assert error < 1e-2


# Rows of 32k+1 elements read past their end by a padded tile: a masked bf16 load
# LLVM aborted on (AI-bench's L1/97 at (1, 4, 129, 65), without its host pad).
@pytest.mark.slow
@pytest.mark.isolated
@pytest.mark.parametrize(("dim", "padded"), [(65, 96), (161, 192)])
def test_padded_tile_of_32k_plus_1_columns_runs(dim: int, padded: int) -> None:
    torch.manual_seed(0)
    q, k = (torch.randn(4, 129, dim).bfloat16() for _ in range(2))
    out = _dots_kernel(160, padded, 64)(q, k, hl.constexpr(padded))
    ref = (q.double() @ k.double().transpose(1, 2)).sum(-1)
    assert (out.double() - ref).abs().max() / ref.abs().max() < 1e-2


@pytest.mark.slow
@pytest.mark.isolated
def test_padded_tile_under_a_grid_index_reaches_amx() -> None:
    q, k = (torch.randn(4, 64, 66).bfloat16() for _ in range(2))
    kernel = _dots_kernel(32, 96, 32)
    module = generate_mlir(kernel, [q, k, hl.constexpr(96)])
    lowered = lower_opt_with_amx(
        module,
        kernel.fn.__name__,
        lambda _stage, after: "x86.amx.tile_mulf" in str(after),
    )
    assert lowered is not None
    assert "vector.contract" not in str(lowered[1])
