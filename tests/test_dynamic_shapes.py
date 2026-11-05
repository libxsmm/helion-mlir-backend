"""``static_shapes=False`` kernels: one compiled entry per shape
bucket, sizes resolved at run time, never specialized to the example sizes."""

from __future__ import annotations

from itertools import starmap
import re

import helion
import helion.language as hl
import pytest
import sympy
import torch

from tests.harness import check_kernel
from tests.harness import lower_opt_with_amx
from tests.harness import opt_pipeline

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.backend import MLIRBackend
from helion_mlir_backend._compiler.mlir.codegen import MLIRModuleBuilder
from helion_mlir_backend._compiler.mlir.driver import _check_sizes
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend.api import _compile


def _kernel(*block_sizes: int) -> object:
    return helion.kernel(
        backend="mlir",
        static_shapes=False,
        config=helion.Config(block_sizes=list(block_sizes)),
    )


@_kernel(32, 32, 32)
def matmul_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@_kernel(32, 32, 32)
def matmul_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = acc
    return out


@_kernel(1, 32, 32, 32)
def batch_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tb, tm, tn in hl.tile([b, m, n]):
        acc = hl.zeros([tb, tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.baddbmm(acc, x[tb, tm, tk], y[tb, tk, tn])
        out[tb, tm, tn] = acc
    return out


@_kernel(16)
def add_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


@_kernel(8, 16)
def axpy_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0 + y[tm, tn]
    return out


@_kernel(8)
def row_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        row = x[tm, :]
        e = torch.exp(row - row.amax(dim=-1, keepdim=True))
        out[tm, :] = e / e.sum(dim=-1, keepdim=True)
    return out


@_kernel(8)
def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        row = x[tm, :]
        mean = row.mean(dim=-1, keepdim=True)
        var = ((row - mean) ** 2).mean(dim=-1, keepdim=True)
        out[tm, :] = (row - mean) * torch.rsqrt(var + 1e-5) * w[None, :] + b[None, :]
    return out


@_kernel(8, 16)
def row_sum_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].sum(dim=-1)
        out[tm] = acc
    return out


@_kernel(16)
def half_copy(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0) // 2
    out = torch.zeros([n], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        out[tile] = x[tile] * 2.0
    return out


@_kernel(8)
def ones_like_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        out[tm, :] = hl.zeros([tm, n], dtype=torch.float32) + 1.0
    return out


@_kernel(16)
def add_tile_count(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + tile.count
    return out


@_kernel(8)
def cumsum_rows(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        out[tm, :] = torch.cumsum(x[tm, :], dim=1)
    return out


@_kernel(8)
def divide_by_row_size(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        row = x[tm, :]
        out[tm, :] = row / row.size(1)
    return out


@_kernel(8)
def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    out = torch.empty([idx.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(idx.size(0)):
        out[tile] = x[idx[tile]]
    return out


@_kernel(8)
def flatten_rows(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        [x.size(0), x.size(1) * x.size(2)], dtype=x.dtype, device=x.device
    )
    for tm in hl.tile(x.size(0)):
        out[tm, :] = x[tm, :, :].reshape([tm, -1])
    return out


@_kernel(8)
def specialized_row(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    n = hl.specialize(n)
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        out[tm, :] = x[tm, :] * 3.0
    return out


@_kernel(32, 32)
def matmul_full_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.matmul(x[tm, :], y[:, tn])
    return out


@_kernel(32)
def matmul_column_tiles(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tn in hl.tile(n):
        out[:, tn] = torch.matmul(x[:, :], y[:, tn])
    return out


@_kernel(32)
def transpose_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([n, m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        out[:, tm] = x[tm, :].t()
    return out


@_kernel(1)
def swap_sizes_view(x: torch.Tensor) -> torch.Tensor:
    b, m, n = x.size()
    out = torch.empty([b, n, m], dtype=x.dtype, device=x.device)
    for tb in hl.tile(b):
        v = x[tb, :, :]
        out[tb, :, :] = v.reshape([v.size(0), v.size(2), v.size(1)])
    return out


@_kernel(8)
def last_column(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        out[tm] = x[tm, n - 1]
    return out


def _matmul_inputs() -> list[list[torch.Tensor]]:
    shapes = [((64, 64), (64, 64)), ((70, 100), (100, 50)), ((33, 129), (129, 65))]
    return [[torch.randn(*a), torch.randn(*b)] for a, b in shapes]


def _check_one_compile(
    kernel: object, runs: list[list[torch.Tensor]], reference, *, opt: bool = False
) -> None:
    for args in runs:
        if opt:
            # check_kernel forces the scalar pipeline.
            with opt_pipeline():
                actual = kernel(*[arg.clone() for arg in args])
            torch.testing.assert_close(actual, reference(*args), atol=1e-4, rtol=1e-4)
        else:
            check_kernel(kernel, reference, args, atol=1e-4, rtol=1e-4)
    (bound,) = kernel._bound_kernels.values()
    assert len(bound._compile_cache) == 1


@pytest.mark.parametrize("kernel", [matmul_addmm, matmul_dot], ids=["addmm", "hl_dot"])
def test_matmul_compiles_once_for_every_shape(kernel: object) -> None:
    torch.manual_seed(0)
    kernel.reset()
    _check_one_compile(kernel, _matmul_inputs(), torch.matmul)


def test_matmul_with_unit_and_ragged_sizes() -> None:
    torch.manual_seed(0)
    for a, b in (((5, 7), (7, 3)), ((1, 40), (40, 33)), ((40, 1), (1, 1))):
        check_kernel(
            matmul_addmm,
            torch.matmul,
            [torch.randn(*a), torch.randn(*b)],
            atol=1e-4,
            rtol=1e-4,
        )


def test_matmul_host_tensors_are_dynamic() -> None:
    text = str(generate_mlir(matmul_addmm, _matmul_inputs()[1]))
    assert "memref<?x?xf32>" in text
    assert "tensor.dim" in text


@pytest.mark.parametrize(
    ("kernel", "reference", "shapes"),
    [
        (add_1d, torch.add, [[(100,), (100,)], [(7,), (7,)], [(33,), (33,)]]),
        (axpy_2d, lambda x, y: x * 2.0 + y, [[(20, 33), (20, 33)], [(7, 5), (7, 5)]]),
        (row_softmax, lambda x: x.softmax(-1), [[(20, 33)], [(21, 40)], [(3, 5)]]),
        (
            layer_norm,
            lambda x, w, b: torch.nn.functional.layer_norm(x, [x.size(1)], w, b, 1e-5),
            [[(20, 33), (33,), (33,)], [(9, 64), (64,), (64,)]],
        ),
        (row_sum_loop, lambda x: x.sum(-1), [[(20, 33)], [(9, 64)]]),
        (half_copy, lambda x: x[: x.size(0) // 2] * 2.0, [[(100,)], [(37,)]]),
        (ones_like_rows, torch.ones_like, [[(20, 33)], [(9, 64)]]),
        (
            add_tile_count,
            lambda x: x + -(-x.size(0) // 16),
            [[(20,)], [(50,)], [(16,)]],
        ),
        (cumsum_rows, lambda x: x.cumsum(1), [[(20, 33)], [(9, 64)]]),
        (divide_by_row_size, lambda x: x / x.size(1), [[(20, 33)], [(9, 64)]]),
        (flatten_rows, lambda x: x.reshape(x.size(0), -1), [[(5, 3, 4)], [(9, 2, 7)]]),
        (
            matmul_full_k,
            torch.matmul,
            [[(64, 48), (48, 64)], [(40, 20), (20, 70)]],
        ),
        (
            matmul_column_tiles,
            torch.matmul,
            [[(40, 20), (20, 70)], [(9, 5), (5, 33)]],
        ),
        (transpose_rows, lambda x: x.t(), [[(40, 20)], [(70, 33)]]),
        (
            swap_sizes_view,
            lambda x: x.reshape(x.size(0), x.size(2), x.size(1)),
            [[(2, 3, 5)], [(3, 4, 6)]],
        ),
        (last_column, lambda x: x[:, x.size(1) - 1], [[(13, 17)], [(9, 5)]]),
    ],
    ids=[
        "add_1d",
        "axpy_2d",
        "row_softmax",
        "layer_norm",
        "row_sum_loop",
        "half_copy",
        "zeros_tile_by_n",
        "tile_count",
        "cumsum",
        "row_size",
        "flatten_view",
        "matmul_full_k",
        "matmul_runtime_rows",
        "transpose",
        "swap_sizes_view",
        "runtime_index",
    ],
)
def test_dynamic_kernel(kernel: object, reference, shapes: list) -> None:
    torch.manual_seed(0)
    kernel.reset()
    _check_one_compile(
        kernel, [list(starmap(torch.randn, run)) for run in shapes], reference
    )


def test_gather_from_a_dynamic_source() -> None:
    torch.manual_seed(0)
    for size, count in ((40, 20), (13, 33)):
        x, idx = torch.randn(size), torch.randint(0, size, (count,))
        check_kernel(gather, lambda x, idx: x[idx], [x, idx])


def test_specialized_size_stays_static() -> None:
    torch.manual_seed(0)
    text = str(generate_mlir(specialized_row, [torch.randn(20, 33)]))
    assert "tensor<?x33xf32>" in text
    for rows in (20, 9):
        check_kernel(specialized_row, lambda x: x * 3.0, [torch.randn(rows, 33)])


def test_sizes_are_never_specialized_to_the_example() -> None:
    torch.manual_seed(0)
    hf, config, env = _compile(row_softmax, [torch.randn(20, 33)], None)
    guards = len(env.shape_env.guards)
    with env:
        MLIRModuleBuilder(hf, config, env).build()
    assert len(env.shape_env.guards) == guards
    row_softmax.reset()
    for shape in ((20, 33), (21, 40), (64, 33)):
        check_kernel(row_softmax, lambda x: x.softmax(-1), [torch.randn(*shape)])


def test_sizes_sharing_a_symbol_must_be_equal() -> None:
    s0, s1 = sympy.symbols("s0 s1", integer=True, positive=True)
    sizes = {"x": [s0, s1], "out": [s0 // 2]}
    _check_sizes([("x", torch.empty(10, 3)), ("out", torch.empty(5))], sizes)
    with pytest.raises(ValueError, match="compiled for the same size"):
        _check_sizes(
            [("x", torch.empty(10, 3)), ("y", torch.empty(4))], {**sizes, "y": [s0]}
        )
    with pytest.raises(ValueError, match="computes it as"):
        _check_sizes([("x", torch.empty(10, 3)), ("out", torch.empty(4))], sizes)


def test_execute_mlir_rejects_host_created_dynamic_tensors() -> None:
    x = torch.randn(20, 33)
    module = generate_mlir(row_softmax, [x])
    with pytest.raises(
        UnsupportedOperationError, match="only known when the host code runs"
    ):
        MLIRBackend().execute_mlir(module, x, kernel_name="row_softmax")


# A block larger than its dim: this kernel took very long in LLVM (a 128x4 copy
# of the pad, its n-D transfer threaded through one scf.if per row), so the IR is
# checked before it's compiled.
@pytest.mark.isolated
@pytest.mark.parametrize("block", [128, 512])
def test_transpose_of_a_block_larger_than_its_dim_compiles(block: int) -> None:
    kernel = helion.kernel(
        transpose_rows.fn,
        backend="mlir",
        static_shapes=False,
        config=helion.Config(block_sizes=[block]),
    )
    torch.manual_seed(0)
    x = torch.randn(48, 48)
    module = generate_mlir(kernel, [x])
    lowered = lower_opt_with_amx(
        module, kernel.fn.__name__, lambda _stage, after: "omp." in str(after)
    )
    assert lowered is not None
    assert not re.search(r"vector<\d{3,}x\d+xf32>", str(lowered[1]))
    runs = list(starmap(torch.randn, ((48, 48), (48, 300), (300, 48), (7, 129))))
    for x in runs:
        with opt_pipeline():
            out = kernel(x.clone())
        assert torch.equal(out, x.t())


@pytest.mark.isolated
def test_matmul_on_the_optimizing_pipeline() -> None:
    torch.manual_seed(0)
    matmul_addmm.reset()
    _check_one_compile(matmul_addmm, _matmul_inputs(), torch.matmul, opt=True)


@pytest.mark.isolated
def test_dynamic_batch_matmul_on_the_optimizing_pipeline() -> None:
    torch.manual_seed(0)
    batch_matmul.reset()
    runs = [[torch.randn(b, 64, 64), torch.randn(b, 64, 64)] for b in (3, 8, 2)]
    _check_one_compile(batch_matmul, runs, torch.bmm, opt=True)


@_kernel(32)
def row_softmax_32(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        row = x[tm, :]
        e = torch.exp(row - row.amax(dim=-1, keepdim=True))
        out[tm, :] = e / e.sum(dim=-1, keepdim=True)
    return out


@_kernel(32)
def layer_norm_32(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm in hl.tile(x.size(0)):
        row = x[tm, :]
        mean = row.mean(dim=-1, keepdim=True)
        var = ((row - mean) ** 2).mean(dim=-1, keepdim=True)
        out[tm, :] = (row - mean) * torch.rsqrt(var + 1e-5) * w[None, :] + b[None, :]
    return out


@pytest.mark.isolated
def test_runtime_sized_linalg_ops_on_the_optimizing_pipeline() -> None:
    """Ops over a full slice of a runtime dim are vectorized with masks."""
    torch.manual_seed(0)
    row_softmax_32.reset()
    runs = [[torch.randn(*shape)] for shape in ((64, 96), (70, 45), (33, 200))]
    _check_one_compile(row_softmax_32, runs, lambda x: x.softmax(-1), opt=True)
    matmul_full_k.reset()
    runs = [
        [torch.randn(64, 48), torch.randn(48, 64)],
        [torch.randn(70, 45), torch.randn(45, 33)],
    ]
    _check_one_compile(matmul_full_k, runs, torch.matmul, opt=True)
    # `** 2` is a vector math.fpowi once vectorized.
    layer_norm_32.reset()
    runs = [[torch.randn(64, n), torch.randn(n), torch.randn(n)] for n in (96, 45)]
    _check_one_compile(
        layer_norm_32,
        runs,
        lambda x, w, b: torch.nn.functional.layer_norm(x, [x.size(1)], w, b, 1e-5),
        opt=True,
    )
