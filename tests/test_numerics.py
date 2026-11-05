"""Numerical stress: small odd f32 shapes, several block sizes, loop-carried values.

Each case guards a bug a larger randomized sweep of the same kind found.
"""

from __future__ import annotations

from itertools import starmap

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel


def _kernel(*block_sizes: int) -> object:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=list(block_sizes)),
    )


def _configured(fn: object, block_sizes: list[int], *, static: bool = True) -> object:
    return helion.kernel(
        fn,
        backend="mlir",
        static_shapes=static,
        config=helion.Config(block_sizes=block_sizes),
    )


@_kernel(8, 16)
def carried_before_invariant_kernel(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        scale = s[tm]
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].sum(dim=-1) * scale
        out[tm] = acc
    return out


@_kernel(8, 16)
def online_sum_exp_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        mx = hl.full([tm], float("-inf"), dtype=torch.float32)
        total = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            block = x[tm, tn]
            new_mx = torch.maximum(mx, block.amax(dim=-1))
            total = total * torch.exp(mx - new_mx) + torch.exp(
                block - new_mx[:, None]
            ).sum(dim=-1)
            mx = new_mx
        out[tm] = total
    return out


@_kernel(4, 8, 8)
def combined_inner_tile_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n, k = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn, tk in hl.tile([n, k]):
            acc = acc + x[tm, tn, tk].sum(dim=-1).sum(dim=-1)
        out[tm] = acc
    return out


def test_carried_value_followed_by_invariant_input() -> None:
    x, s = torch.randn(13, 37), torch.randn(13)
    check_kernel(
        carried_before_invariant_kernel,
        lambda x, s: x.sum(-1) * s,
        [x, s],
        atol=1e-4,
        rtol=1e-4,
    )


def test_carried_value_dead_after_loop() -> None:
    check_kernel(
        online_sum_exp_kernel,
        lambda x: torch.exp(x - x.amax(-1, keepdim=True)).sum(-1),
        [torch.randn(13, 37)],
        atol=1e-4,
        rtol=1e-4,
    )


def test_combined_inner_tile_carries_accumulator() -> None:
    check_kernel(
        combined_inner_tile_kernel,
        lambda x: x.sum((1, 2)),
        [torch.randn(6, 12, 20)],
        atol=1e-4,
        rtol=1e-4,
    )


def split_sum(x: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tk in hl.tile(k // 2):
            acc = acc + x[tm, tk].sum(-1)
        for tk in hl.tile(k // 2, k):
            acc = acc + x[tm, tk].sum(-1)
        out[tm] = acc
    return out


def row_var(x: torch.Tensor) -> torch.Tensor:
    # float64: torch-mlir's decomposition of var and std computes in f64. On the CPU; on the
    # XPU Helion stops these kernels before any IR (a bare AssertionError).
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tm in hl.tile(x.size(0)):
        out[tm] = torch.var(x[tm, :], dim=-1, unbiased=False)
    return out


def registered_block(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            out[tm, tn] = x[tm, tn] * 2.0
    return out


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, m, d = q.size()
    _, n, _ = k.size()
    out = torch.empty_like(q)
    for tb, tm in hl.tile([batch, m]):
        qt = q[tb, tm, :]
        mx = hl.full([tb, tm], float("-inf"), dtype=torch.float32)
        total = hl.zeros([tb, tm], dtype=torch.float32)
        acc = hl.zeros([tb, tm, d], dtype=torch.float32)
        for tn in hl.tile(n):
            scores = torch.bmm(qt, k[tb, tn, :].transpose(1, 2))
            new_mx = torch.maximum(mx, scores.amax(-1))
            p = torch.exp(scores - new_mx[:, :, None])
            alpha = torch.exp(mx - new_mx)
            total = total * alpha + p.sum(-1)
            acc = torch.baddbmm(acc * alpha[:, :, None], p, v[tb, tn, :])
            mx = new_mx
        out[tb, tm, :] = acc / total[:, :, None]
    return out


def softmax_loops(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        mx = hl.full([tm], float("-inf"), dtype=torch.float32)
        for tn in hl.tile(n):
            mx = torch.maximum(mx, x[tm, tn].amax(-1))
        total = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            total = total + torch.exp(x[tm, tn] - mx[:, None]).sum(-1)
        for tn in hl.tile(n):
            out[tm, tn] = torch.exp(x[tm, tn] - mx[:, None]) / total[:, None]
    return out


def matmul_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


def transpose_copy(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([n, m], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tn, tm] = x[tm, tn].t()
    return out


@pytest.mark.parametrize("shape", [(7, 1), (5, 3)])
def test_zero_trip_inner_loop(shape: tuple[int, int]) -> None:
    check_kernel(
        _configured(split_sum, [2, 1, 1]),
        lambda x: x.sum(-1),
        [torch.randn(*shape)],
        atol=1e-4,
        rtol=1e-4,
    )


def test_var_with_partial_tiles() -> None:
    check_kernel(
        _configured(row_var, [4]),
        lambda x: torch.var(x, dim=-1, unbiased=False),
        [torch.randn(13, 17)],
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("static", [True, False], ids=["static", "dynamic"])
def test_registered_block_size_of_a_unit_dim(static: bool) -> None:
    check_kernel(
        _configured(registered_block, [1, 4], static=static),
        lambda x: x * 2.0,
        [torch.randn(7, 1)],
    )


@pytest.mark.parametrize("shape", [(2, 3, 5, 4), (1, 9, 13, 8)])
@pytest.mark.parametrize("block_sizes", [[1, 4, 8], [2, 8, 4]])
def test_attention_small_shapes(
    shape: tuple[int, int, int, int], block_sizes: list[int]
) -> None:
    b, m, n, d = shape
    check_kernel(
        _configured(attention, block_sizes),
        lambda q, k, v: torch.softmax(q @ k.transpose(1, 2), -1) @ v,
        [torch.randn(b, m, d), torch.randn(b, n, d), torch.randn(b, n, d)],
        atol=1e-4,
        rtol=1e-4,
    )


_SWEEP = {
    "softmax_loops": (softmax_loops, lambda x: x.softmax(-1), [(13, 17)], 4),
    "matmul": (matmul_addmm, torch.matmul, [(17, 9), (9, 13)], 3),
    "transpose": (transpose_copy, lambda x: x.t(), [(13, 17)], 2),
}


@pytest.mark.parametrize("case", list(_SWEEP))
@pytest.mark.parametrize("pattern", [[1], [4], [8, 2]], ids=["1", "4", "8_2"])
def test_block_size_sweep(case: str, pattern: list[int]) -> None:
    fn, reference, shapes, count = _SWEEP[case]
    block_sizes = [pattern[i % len(pattern)] for i in range(count)]
    torch.manual_seed(0)
    check_kernel(
        _configured(fn, block_sizes),
        reference,
        list(starmap(torch.randn, shapes)),
        atol=1e-4,
        rtol=1e-4,
    )


def forward_difference(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0) - 1
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        out[tile] = x[tile.index + 1] - x[tile]
    return out


def shifted_columns(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n - 2], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n - 2]):
        out[tm, tn] = x[tm, tn.index + 2] + x[tm, tn]
    return out


def backward_difference(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(1, x.size(0)):
        out[tile] = x[tile] - x[tile.index - 1]
    return out


def runtime_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    n = x.size(0) - shift
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        out[tile] = x[tile.index + shift]
    return out


@pytest.mark.parametrize("static", [True, False], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    ("fn", "reference", "shape", "block_sizes"),
    [
        (forward_difference, lambda x: x[1:] - x[:-1], (17,), [4]),
        (shifted_columns, lambda x: x[:, 2:] + x[:, :-2], (5, 11), [2, 4]),
        (
            backward_difference,
            lambda x: torch.cat([x[:1] * 0, x[1:] - x[:-1]]),
            (17,),
            [4],
        ),
    ],
    ids=["forward", "columns", "backward"],
)
def test_tile_index_plus_constant(
    fn: object,
    reference: object,
    shape: tuple[int, ...],
    block_sizes: list[int],
    static: bool,
) -> None:
    check_kernel(
        _configured(fn, block_sizes, static=static), reference, [torch.randn(*shape)]
    )


def test_tile_index_plus_runtime_offset_is_rejected() -> None:
    from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

    with pytest.raises(UnsupportedOperationError, match="runtime offset"):
        _configured(runtime_shift, [4])(torch.randn(17), 3)
