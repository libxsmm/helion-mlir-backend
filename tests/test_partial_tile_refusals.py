"""Reductions at partial tiles that Helion leaves wrong, on every backend: refused
here, as a silent wrong result is the one outcome to avoid."""

from __future__ import annotations

import operator

import helion
import helion.language as hl
import pytest
import torch

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir.support.errors import UnsupportedOperationError


def _minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a, b)


def _multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return a * b


def _add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return a + b


def mean_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = tv.mean(-1, keepdim=True)
    return out


def var_tile(x: torch.Tensor) -> torch.Tensor:
    # float64: torch-mlir's decomposition of var and std computes in f64. On the CPU; on the
    # XPU Helion stops these kernels before any IR (a bare AssertionError).
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = tv.var(-1, keepdim=True)
    return out


def std_tile(x: torch.Tensor) -> torch.Tensor:
    # float64: torch-mlir's decomposition of var and std computes in f64. On the CPU; on the
    # XPU Helion stops these kernels before any IR (a bare AssertionError).
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = tv.std(-1, keepdim=True)
    return out


def reduce_min(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.reduce(_minimum, tv, dim=1, keep_dims=True)
    return out


def reduce_mul(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.reduce(_multiply, tv, dim=1, keep_dims=True)
    return out


def reduce_min_identity(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.reduce(
                _minimum, tv, dim=1, other=float("inf"), keep_dims=True
            )
    return out


def reduce_add(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.reduce(_add, tv, dim=1, keep_dims=True)
    return out


def scan_min_reverse(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.associative_scan(_minimum, tv, dim=1, reverse=True)
    return out


def scan_min(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=bn):
            tv = x[tm, tn]
            out[tm, tn] = hl.associative_scan(_minimum, tv, dim=1)
    return out


def std_mean_tile(x: torch.Tensor) -> torch.Tensor:
    # float64: torch-mlir's decomposition of var and std computes in f64. On the CPU; on the
    # XPU Helion stops these kernels before any IR (a bare AssertionError).
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            std, mean = torch.std_mean(x[tm, tn], dim=-1, keepdim=True)
            out[tm, tn] = std + mean
    return out


def var_mean_tile(x: torch.Tensor) -> torch.Tensor:
    # float64: torch-mlir's decomposition of var and std computes in f64. On the CPU; on the
    # XPU Helion stops these kernels before any IR (a bare AssertionError).
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            var, mean = torch.var_mean(x[tm, tn], dim=-1, keepdim=True)
            out[tm, tn] = var + mean
    return out


def last_amax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn]
        out[tm] = value.amax(-1)
    return out


def last_mean(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn] * 2.0
        out[tm] = value.mean(-1)
    return out


def last_sum_of_exp(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = torch.exp(x[tm, tn])
        out[tm] = value.sum(-1)
    return out


def last_sum_of_shift(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn] + 1.0
        out[tm] = value.sum(-1)
    return out


def last_sum_of_scaled(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn] * 2.0 - x[tm, tn]
        out[tm] = value.sum(-1)
    return out


def stored_exp(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 16], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = torch.exp(x[tm, tn])
        out[tm, :] = value
    return out


def stored_scaled(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 16], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn] * 2.0
        out[tm, :] = value
    return out


def stored_shifted_after(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 16], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn] * 2.0
        out[tm, :] = value + 1.0
    return out


def sum_shifted_after(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn] * 2.0
        out[tm] = (value + 1.0).sum(-1)
    return out


def einsum_of_exp(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = torch.exp(x[tm, tn])
        out[tm] = torch.einsum("mn,mn->m", value, value)
    return out


def carried_amax(x: torch.Tensor) -> torch.Tensor:
    m, a, c = x.shape
    block_a = hl.register_block_size(a)
    block_c = hl.register_block_size(c)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_c], dtype=torch.float32)
        for _ta in hl.tile(a, block_size=block_a):
            for tc in hl.tile(c, block_size=block_c):
                value = x[tm, 0, tc]
        out[tm] = value.amax(-1)
    return out


def carried_sum_of_exp(x: torch.Tensor) -> torch.Tensor:
    m, a, c = x.shape
    block_a = hl.register_block_size(a)
    block_c = hl.register_block_size(c)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_c], dtype=torch.float32)
        for _ta in hl.tile(a, block_size=block_a):
            for tc in hl.tile(c, block_size=block_c):
                value = torch.exp(x[tm, 0, tc])
        out[tm] = value.sum(-1)
    return out


def carried_stored_exp(x: torch.Tensor) -> torch.Tensor:
    m, a, c = x.shape
    block_a = hl.register_block_size(a)
    block_c = hl.register_block_size(c)
    out = torch.empty([m, 8], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_c], dtype=torch.float32)
        for _ta in hl.tile(a, block_size=block_a):
            for tc in hl.tile(c, block_size=block_c):
                value = torch.exp(x[tm, 0, tc])
        out[tm, :] = value
    return out


def carried_sum_of_scaled(x: torch.Tensor) -> torch.Tensor:
    m, a, c = x.shape
    block_a = hl.register_block_size(a)
    block_c = hl.register_block_size(c)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_c], dtype=torch.float32)
        for _ta in hl.tile(a, block_size=block_a):
            for tc in hl.tile(c, block_size=block_c):
                value = x[tm, 0, tc] * 2.0
        out[tm] = value.sum(-1)
    return out


def sibling_amax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn]
        for tk in hl.tile(n, block_size=block_k):
            out[tm, tk] = value.amax(-1, keepdim=True) + x[tm, tk] * 0.0
    return out


def sibling_sum_of_exp(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = torch.exp(x[tm, tn])
        for tk in hl.tile(n, block_size=block_k):
            out[tm, tk] = value.sum(-1, keepdim=True) + x[tm, tk] * 0.0
    return out


def sibling_same_block(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    block_k = block_n
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn]
        for tk in hl.tile(n, block_size=block_k):
            out[tm, tk] = value.amax(-1, keepdim=True) + x[tm, tk] * 0.0
    return out


def sibling_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn]
        for tk in hl.tile(n, block_size=block_k):
            out[tm, tk] = value.sum(-1, keepdim=True) + x[tm, tk] * 0.0
    return out


def sibling_running(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn]
        top = hl.full([tm, 1], float("-inf"), dtype=torch.float32)
        for _tk in hl.tile(32, block_size=block_k):
            top = torch.maximum(top, value.amax(-1, keepdim=True))
        out[tm] = top.squeeze(-1)
    return out


def sibling_two_levels(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    block_k = hl.register_block_size(n)
    block_j = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn]
        top = hl.full([tm, 1], float("-inf"), dtype=torch.float32)
        for _tk in hl.tile(16, block_size=block_k):
            for _tj in hl.tile(16, block_size=block_j):
                top = torch.maximum(top, value.amax(-1, keepdim=True))
        out[tm] = top.squeeze(-1)
    return out


def after_times_recip(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn]
        out[tm] = (value * (1.0 / value)).sum(-1)
    return out


def body_times_recip(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn] * (1.0 / x[tm, tn])
        out[tm] = value.sum(-1)
    return out


def after_times_exp(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, block_n], dtype=torch.float32)
        for tn in hl.tile(n, block_size=block_n):
            value = x[tm, tn]
        out[tm] = (value * torch.exp(-value)).sum(-1)
    return out


def dot_of_shifted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            acc = hl.dot(a[tm, tk] + 1.0, b[tk, tn] + 1.0, acc=acc)
        out[tm, tn] = acc
    return out


def dot_of_one_shifted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            acc = hl.dot(a[tm, tk], b[tk, tn] + 1.0, acc=acc)
        out[tm, tn] = acc
    return out


def addmm_of_shifted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            acc = torch.addmm(acc, a[tm, tk] + 1.0, b[tk, tn] + 1.0)
        out[tm, tn] = acc
    return out


def dot_of_loads(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            acc = hl.dot(a[tm, tk], b[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


def dot_of_scaled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            acc = hl.dot(a[tm, tk] * 2.0, -b[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


def dot_of_casts(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([a.size(0), b.size(1)]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            first = a[tm, tk].to(torch.float32)
            acc = hl.dot(first, b[tk, tn].to(torch.float32), acc=acc)
        out[tm, tn] = acc
    return out


def _logical_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.logical_or(a, b)


def _bitwise_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return a | b


def reduce_logical_or(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            tile = x[tm, tn]
            out[tm, tn] = (
                hl.reduce(_logical_or, tile > 1.5, dim=1, keep_dims=True).to(
                    torch.float32
                )
                + tile * 0.0
            )
    return out


def reduce_bitwise_or(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            tile = x[tm, tn]
            out[tm, tn] = (
                hl.reduce(_bitwise_or, tile > 1.5, dim=1, keep_dims=True).to(
                    torch.float32
                )
                + tile * 0.0
            )
    return out


def scan_sum_reverse(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        for tn in hl.tile(n, block_size=block_n):
            out[tm, tn] = hl.associative_scan(_add, x[tm, tn], dim=1, reverse=True)
    return out


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16, 16]),
    )(fn)


@pytest.mark.parametrize(
    ("fn", "match"),
    [
        (mean_tile, "over a partial tile"),
        (var_tile, "over a partial tile"),
        (std_tile, "over a partial tile"),
        (reduce_min, "other=0"),
        (reduce_mul, "other=0"),
        (scan_min_reverse, "reverse"),
        (std_mean_tile, "over a partial tile"),
        (var_mean_tile, "over a partial tile"),
        (last_amax, "after a loop"),
        (last_mean, "after a loop"),
        (last_sum_of_exp, "after a loop"),
        (last_sum_of_shift, "after a loop"),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_refused_at_a_partial_tile(fn: object, match: str) -> None:
    x = torch.randn(16, 60).abs() + 1.0
    with pytest.raises(UnsupportedOperationError, match=match):
        _kernel(fn)(x)


@pytest.mark.parametrize(
    "fn",
    [
        mean_tile,
        var_tile,
        std_tile,
        reduce_min,
        reduce_mul,
        scan_min_reverse,
        std_mean_tile,
        var_mean_tile,
        last_amax,
        last_mean,
        last_sum_of_exp,
        last_sum_of_shift,
    ],
    ids=lambda v: v.__name__,
)
def test_lowers_with_whole_tiles(fn: object) -> None:
    _kernel(fn)(torch.randn(16, 64).abs() + 1.0)


def mean_of_columns(x: torch.Tensor) -> torch.Tensor:
    """A mean over whole columns, with partial rows: right."""
    m, n = x.shape
    bm = hl.register_block_size(m)
    out = torch.empty([m, 1], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m, block_size=bm):
        out[tm, :] = x[tm, :].mean(-1, keepdim=True)
    return out


def test_mean_over_whole_columns_with_partial_rows() -> None:
    torch.manual_seed(0)
    x = torch.randn(60, 16)
    got = _kernel(mean_of_columns)(x).double()
    assert (got - x.double().mean(-1, keepdim=True)).abs().max() < 1e-5


def _chunk(x: torch.Tensor, op: object) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float64)
    for start in range(0, x.shape[1], 16):
        out[:, start : start + 16] = op(x.double()[:, start : start + 16])
    return out


@pytest.mark.parametrize(
    ("fn", "op"),
    [
        (reduce_min_identity, lambda v: v.amin(-1, keepdim=True)),
        (reduce_add, lambda v: v.sum(-1, keepdim=True)),
        (reduce_logical_or, lambda v: (v > 1.5).any(1, keepdim=True).double()),
        (reduce_bitwise_or, lambda v: (v > 1.5).any(1, keepdim=True).double()),
        (scan_sum_reverse, lambda v: v.flip(1).cumsum(1).flip(1)),
        (scan_min, lambda v: v.cummin(1).values),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_right_at_a_partial_tile(fn: object, op: object) -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 60).abs() + 1.0
    got = _kernel(fn)(x).double()
    assert (got - _chunk(x, op)).abs().max() < 1e-5


def test_last_tile_sum_of_padding_that_stays_zero() -> None:
    x = torch.randn(16, 60).abs() + 1.0
    ref = x[:, 48:].double().sum(-1)
    assert (
        _kernel(last_sum_of_scaled)(x).double() - ref
    ).abs().max() / ref.max() < 1e-5


@pytest.mark.parametrize("fn", [mean_tile, reduce_min, scan_min_reverse, last_amax])
def test_refusal_names_static_shapes(fn: object) -> None:
    """Runtime bounds count every tile as partial, so divisible sizes don't help: the
    message has to say so."""
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=False,
        config=helion.Config(block_sizes=[16, 16]),
    )
    with pytest.raises(UnsupportedOperationError) as refused:
        kernel(fn)(torch.randn(16, 64).abs() + 1.0)
    assert "static_shapes=True" in str(refused.value)


@pytest.mark.parametrize(
    "fn",
    [stored_exp, stored_shifted_after, sum_shifted_after, einsum_of_exp],
    ids=lambda v: v.__name__,
)
def test_last_tile_use_after_the_loop_refused(fn: object) -> None:
    """Padding that isn't 0 reaches a store or a reduction: padded lanes held 1.0, and
    sums and an einsum were off by 7.8 % and 1 %."""
    with pytest.raises(UnsupportedOperationError, match="after a loop"):
        _kernel(fn)(torch.rand(16, 60) + 1.0)


def test_last_tile_store_of_padding_that_stays_zero() -> None:
    x = torch.rand(16, 60) + 1.0
    got = _kernel(stored_scaled)(x).double()
    assert (got[:, :12] - x[:, 48:].double() * 2).abs().max() < 1e-5
    assert torch.count_nonzero(got[:, 12:]) == 0


def _kernel_3d(fn: object) -> helion.Kernel:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[8, 8, 16]),
    )(fn)


@pytest.mark.parametrize(
    "fn",
    [carried_amax, carried_sum_of_exp, carried_stored_exp],
    ids=lambda v: v.__name__,
)
def test_last_tile_value_through_an_enclosing_loop_refused(fn: object) -> None:
    """Carried out through the enclosing loop, off by 113 % as a sum of exp."""
    with pytest.raises(UnsupportedOperationError, match="after a loop"):
        _kernel_3d(fn)(-(torch.rand(16, 8, 30) + 1.0))


def test_last_tile_sum_through_an_enclosing_loop() -> None:
    x = torch.rand(16, 8, 30) + 1.0
    ref = (x[:, 0, 24:].double() * 2).sum(-1)
    got = _kernel_3d(carried_sum_of_scaled)(x).double()
    assert (got - ref).abs().max() / ref.max() < 1e-5


@pytest.mark.parametrize(
    "fn",
    [
        sibling_amax,
        sibling_sum_of_exp,
        sibling_same_block,
        sibling_running,
        sibling_two_levels,
    ],
    ids=lambda v: v.__name__,
)
def test_last_tile_value_read_by_a_sibling_loop_refused(fn: object) -> None:
    """The sibling loop's mask is its own tile's: maxima of negatives were off by
    100 % and a sum of exp by 2.3 %."""
    x = -(torch.rand(16, 60) + 1.0)
    blocks = {sibling_same_block: 2, sibling_two_levels: 4}.get(fn, 3)
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16] * blocks),
    )
    with pytest.raises(UnsupportedOperationError, match="after a loop"):
        kernel(fn)(x)


@pytest.mark.parametrize(
    "fn", [after_times_recip, body_times_recip], ids=lambda v: v.__name__
)
def test_zero_times_a_value_from_padding_refused(fn: object) -> None:
    """Padded lanes hold 0 * inf: NaN in the sum."""
    with pytest.raises(UnsupportedOperationError, match="after a loop"):
        _kernel(fn)(torch.rand(16, 60) + 1.0)


@pytest.mark.parametrize("fn", [sibling_sum, after_times_exp], ids=lambda v: v.__name__)
def test_right_but_refused_without_a_proof(fn: object) -> None:
    """Pinned: right without the refusal, and accepted as a false refusal: nothing
    proves the padding stays 0 there. This test can change if the walk learns more."""
    blocks = [16, 16, 16] if fn is sibling_sum else [16, 16]
    kernel = helion.kernel(
        backend="mlir", static_shapes=True, config=helion.Config(block_sizes=blocks)
    )
    with pytest.raises(UnsupportedOperationError, match="after a loop"):
        kernel(fn)(torch.rand(16, 60) + 1.0)


def _dot_kernel(fn: object) -> helion.Kernel:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16, 16, 16]),
    )(fn)


@pytest.mark.parametrize(
    "fn", [dot_of_shifted, dot_of_one_shifted], ids=lambda v: v.__name__
)
def test_dot_of_a_computed_operand_at_a_partial_k_refused(fn: object) -> None:
    """Regression: ``hl.dot`` gets no mask from Helion: 2.1 % off at 32x56x32, where
    ``torch.addmm`` is right. ``dot_of_one_shifted`` is
    right without the refusal: nothing proves the other operand finite."""
    a, b = torch.rand(32, 56) + 1.0, torch.rand(56, 32) + 1.0
    with pytest.raises(UnsupportedOperationError, match=r"torch\.addmm"):
        _dot_kernel(fn)(a, b)


@pytest.mark.parametrize(
    ("fn", "ref"),
    [
        (addmm_of_shifted, lambda a, b: (a + 1) @ (b + 1)),
        (dot_of_loads, operator.matmul),
        (dot_of_scaled, lambda a, b: (a * 2) @ -b),
    ],
    ids=["addmm", "loads", "scaled"],
)
def test_right_contractions_at_a_partial_k(fn: object, ref: object) -> None:
    torch.manual_seed(0)
    a, b = torch.rand(32, 56) + 1.0, torch.rand(56, 32) + 1.0
    want = ref(a.double(), b.double())
    got = _dot_kernel(fn)(a, b).double()
    assert (got - want).abs().max() / want.abs().max() < 1e-5


def test_dot_of_a_computed_operand_at_a_whole_k() -> None:
    a, b = torch.rand(32, 64) + 1.0, torch.rand(64, 32) + 1.0
    want = (a.double() + 1) @ (b.double() + 1)
    got = _dot_kernel(dot_of_shifted)(a, b).double()
    assert (got - want).abs().max() / want.abs().max() < 1e-5


def test_dot_of_casts_at_a_partial_k() -> None:
    torch.manual_seed(0)
    a = (torch.rand(32, 56) + 1.0).to(torch.bfloat16)
    b = (torch.rand(56, 32) + 1.0).to(torch.bfloat16)
    want = a.double() @ b.double()
    got = _dot_kernel(dot_of_casts)(a, b).double()
    assert (got - want).abs().max() / want.abs().max() < 1e-5
