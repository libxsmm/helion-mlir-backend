"""End-to-end execution under the backend's optimizing lighthouse pipeline.

f32 only: bf16 needs AMX hardware to execute (covered at IR level by test_amx_ir_gate.py).
"""

from __future__ import annotations

import importlib
import re
from typing import TYPE_CHECKING

import helion
import helion.language as hl
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir import passmanager
from mlir.dialects import transform
from mlir.dialects.transform import structured
import pytest
import torch

from tests.harness import opt_pipeline

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.helion_transforms import align_operand_rows
from helion_mlir_backend._compiler.helion_transforms import amx_move_offsets
from helion_mlir_backend._compiler.helion_transforms import fold_empty_slices
from helion_mlir_backend._compiler.helion_transforms import forward_rows
from helion_mlir_backend._compiler.helion_transforms import hoist_allocas
from helion_mlir_backend._compiler.helion_transforms import legalize_for_llvm
from helion_mlir_backend._compiler.helion_transforms import linearize_vector_ops
from helion_mlir_backend._compiler.helion_transforms import lower_transposes
from helion_mlir_backend._compiler.helion_transforms import materialize_operand_pads
from helion_mlir_backend._compiler.helion_transforms import schedule_amx_loads
from helion_mlir_backend._compiler.helion_transforms import split_bf16_loads
from helion_mlir_backend._compiler.helion_transforms import split_contraction_operands
from helion_mlir_backend._compiler.helion_transforms import split_copies
from helion_mlir_backend._compiler.helion_transforms import split_transfers
from helion_mlir_backend._compiler.helion_transforms import unroll_transfers
from helion_mlir_backend._compiler.helion_transforms import vectorize_linalg
from helion_mlir_backend._compiler.helion_transforms import version_padded_operands
from helion_mlir_backend._compiler.mlir_transforms import HelionTransformDialect
from helion_mlir_backend._compiler.mlir_transforms import (
    forward_rows as forward_rows_op,
)
from helion_mlir_backend._compiler.mlir_transforms import match_scalar_contractions
from helion_mlir_backend._compiler.mlir_transforms import (
    split_contraction_operands as split_contraction_operands_op,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.isolated, pytest.mark.slow]


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 32])
)
def opt_elementwise_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = torch.relu(x[tm, tn] * 2.0 + y[tm, tn])
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 32, 32])
)
def opt_addmm_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        out[tm, tn] = acc
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 128])
)
def opt_row_sum_loop_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            acc = acc + x[tm, tn].sum(dim=-1)
        out[tm] = acc
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 128])
)
def opt_row_max_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        acc = hl.full([tm], float("-inf"), dtype=torch.float32)
        for tn in hl.tile(n):
            acc = torch.maximum(acc, x[tm, tn].amax(dim=-1))
        out[tm] = acc
    return out


@helion.kernel(
    backend="mlir",
    static_shapes=True,
    config=helion.Config(block_sizes=[1, 32, 32, 32]),
)
def opt_ragged_k_bmm_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batch, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([batch, m, n], dtype=x.dtype, device=x.device)
    for tb, tm, tn in hl.tile([batch, m, n]):
        acc = hl.zeros([tb, tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.baddbmm(acc, x[tb, tm, tk], y[tb, tk, tn])
        out[tb, tm, tn] = acc
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 32])
)
def opt_col_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    k, n = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tn in hl.tile(n):
        acc = hl.zeros([tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = acc + x[tk, tn].sum(dim=0)
        out[tn] = acc
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32])
)
def opt_tile_sums_kernel(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    out = torch.ones([(n + 31) // 32, x.size(1)], dtype=x.dtype, device=x.device)
    for t in hl.tile(n):
        out[t.id, :] = out[t.id, :] + x[t, :].sum(0)
    return out


@helion.kernel(
    backend="mlir",
    static_shapes=True,
    config=helion.Config(block_sizes=[32, 128, 128]),
)
def opt_online_softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        mi = hl.full([tm], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(n):
            values = x[tm, tn]
            mi_next = torch.maximum(mi, torch.amax(values, dim=1))
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tn in hl.tile(n):
            out[tm, tn] = torch.exp(x[tm, tn] - mi[:, None]) / di[:, None]
    return out


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[2, 4, 4])
)
def opt_packed_gemm_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pack A and B into 32x32 blocks, then a blocked contraction over tiles of
    blocks (partial ones too)."""
    m, k = a.size()
    _, n = b.size()
    mb, kb, nb = (m + 31) // 32, (k + 31) // 32, (n + 31) // 32
    a4 = torch.empty((mb, kb, 32, 32), dtype=a.dtype, device=a.device)
    b4 = torch.empty((nb, kb, 32, 32), dtype=b.dtype, device=b.device)
    out4 = torch.empty((mb, 32, nb, 32), dtype=a.dtype, device=a.device)
    for tm, tk in hl.tile([mb * 32, kb * 32], block_size=[32, 32]):
        a4[tm.id, tk.id, :, :] = a[tm, tk]
    for tn, tk in hl.tile([nb * 32, kb * 32], block_size=[32, 32]):
        b4[tn.id, tk.id, :, :] = b[tk, tn]
    hl.barrier()
    for tbm, tbn in hl.tile([mb, nb]):
        acc = hl.zeros([tbm, tbn, 32, 32], dtype=torch.float32)
        for tbk in hl.tile(kb):
            acc = acc + torch.einsum(
                "akmc,bkcn->abmn", a4[tbm, tbk, :, :], b4[tbn, tbk, :, :]
            )
        out4[tbm, :, tbn, :] = acc.permute(0, 2, 1, 3)
    return out4.view(mb * 32, nb * 32)[:m, :n]


def _cpu_utils_cases() -> dict[str, tuple[Callable[[], object], Callable[[], object]]]:
    import helion_mlir_utils as cpu

    a, b = torch.randn(256, 192), torch.randn(192, 320)
    bias = torch.randn(320)
    batch_a, batch_b = torch.randn(2, 64, 96), torch.randn(2, 96, 128)
    layer = torch.nn.Linear(192, 160)
    rows, ragged_rows = torch.randn(64, 1024), torch.randn(64, 1000)
    ragged_a, ragged_b = torch.randn(2, 32, 40), torch.randn(2, 40, 32)
    ragged_cols = torch.randn(70, 64)
    gemm_a, gemm_b = torch.randn(70, 200), torch.randn(200, 150)
    return {
        "matmul": (lambda: cpu.matmul(a, b), lambda: a @ b),
        "matmul_bias_relu": (
            lambda: cpu.matmul(a, b, bias=bias, epilogue=torch.relu),
            lambda: torch.relu(a @ b + bias),
        ),
        "matmul_trans_b": (
            lambda: cpu.matmul(a, b.t().contiguous(), trans_b=True),
            lambda: a @ b,
        ),
        "bmm": (lambda: cpu.bmm(batch_a, batch_b), lambda: batch_a @ batch_b),
        "linear": (lambda: cpu.linear(a, layer)[0], lambda: layer(a).detach()),
        "elementwise": (
            lambda: opt_elementwise_kernel(
                a[:, :64].contiguous(), a[:, 64:128].contiguous()
            ),
            lambda: torch.relu(a[:, :64] * 2.0 + a[:, 64:128]),
        ),
        "addmm": (
            lambda: opt_addmm_kernel(a[:64], b[:, :64].contiguous()),
            lambda: a[:64] @ b[:, :64],
        ),
        "row_sum_loop": (lambda: opt_row_sum_loop_kernel(rows), lambda: rows.sum(-1)),
        "row_max_ragged": (
            lambda: opt_row_max_kernel(ragged_rows),
            lambda: ragged_rows.amax(-1),
        ),
        "online_softmax": (
            lambda: opt_online_softmax_kernel(ragged_rows),
            lambda: ragged_rows.softmax(-1),
        ),
        # Padded batch_matmul operands (miscompiled to NaNs without vectorize_pads).
        "ragged_k_bmm": (
            lambda: opt_ragged_k_bmm_kernel(ragged_a, ragged_b),
            lambda: ragged_a @ ragged_b,
        ),
        # Reductions over padded tiles (wrong without vectorize_pads).
        "col_sum_ragged": (
            lambda: opt_col_sum_kernel(ragged_cols),
            lambda: ragged_cols.sum(0),
        ),
        "tile_sums_ragged": (
            lambda: opt_tile_sums_kernel(ragged_cols),
            lambda: (
                1
                + torch.nn.functional.pad(ragged_cols, (0, 0, 0, 26))
                .reshape(-1, 32, 64)
                .sum(1)
            ),
        ),
        "packed_gemm_ragged": (
            lambda: opt_packed_gemm_kernel(gemm_a, gemm_b),
            lambda: gemm_a @ gemm_b,
        ),
    }


@pytest.mark.parametrize(
    "case",
    [
        "matmul",
        "matmul_bias_relu",
        "matmul_trans_b",
        "bmm",
        "linear",
        "elementwise",
        "addmm",
        "row_sum_loop",
        "row_max_ragged",
        "online_softmax",
        "ragged_k_bmm",
        "col_sum_ragged",
        "tile_sums_ragged",
        "packed_gemm_ragged",
    ],
)
def test_optimizing_pipeline_f32(case: str) -> None:
    torch.manual_seed(0)
    run, reference = _cpu_utils_cases()[case]
    with opt_pipeline():
        actual = run()
    torch.testing.assert_close(actual, reference(), atol=1e-3, rtol=1e-3)


RANK_REDUCING_TRANSFERS = """
func.func @f(%m: memref<?x?xf32>, %i: index, %v: vector<32xf32>) -> vector<32xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %read = vector.transfer_read %m[%i, %c0], %pad : memref<?x?xf32>, vector<32xf32>
  vector.transfer_write %v, %m[%i, %c0] : vector<32xf32>, memref<?x?xf32>
  return %read : vector<32xf32>
}
"""


def _apply(make_schedule: Callable[[], ir.Module], source: str) -> str:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        schedule = make_schedule()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def test_split_transfers_rank_reducing() -> None:
    text = _apply(split_transfers, RANK_REDUCING_TRANSFERS)
    # The read and the write, each on its in-bounds path (cleanup merges the ifs).
    assert "scf.if" in text
    assert text.count("in_bounds = [true]") == 2


# The copy of a partial tile into a padded block (e.g. a pad or an operand pack),
# of a block into a partial tile, and of a whole block.
PARTIAL_TILE_COPIES = """
func.func @f(%src: memref<?x?xf32, strided<[33, 1], offset: ?>>,
             %dst: memref<32x32xf32, strided<[64, 1], offset: ?>>,
             %out: memref<?x32xf32>, %buf: memref<32x32xf32>,
             %copy: memref<32x32xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %tile = vector.transfer_read %src[%c0, %c0], %pad
      : memref<?x?xf32, strided<[33, 1], offset: ?>>, vector<32x32xf32>
  vector.transfer_write %tile, %dst[%c0, %c0] {in_bounds = [true, true]}
      : vector<32x32xf32>, memref<32x32xf32, strided<[64, 1], offset: ?>>
  %block = vector.transfer_read %buf[%c0, %c0], %pad {in_bounds = [true, true]}
      : memref<32x32xf32>, vector<32x32xf32>
  vector.transfer_write %block, %out[%c0, %c0] {in_bounds = [false, true]}
      : vector<32x32xf32>, memref<?x32xf32>
  %whole = vector.transfer_read %buf[%c0, %c0], %pad {in_bounds = [true, true]}
      : memref<32x32xf32>, vector<32x32xf32>
  vector.transfer_write %whole, %copy[%c0, %c0] {in_bounds = [true, true]}
      : vector<32x32xf32>, memref<32x32xf32>
  return
}
"""


def test_split_copies_copies_edge_tiles_by_rows() -> None:
    text = _apply(split_copies, PARTIAL_TILE_COPIES)
    # The partial copies row by row, in bounds if they fit, else with rows past
    # the source's end padding, past the destination's end not written, masked
    # past the source's last column. The whole copy is left as is.
    assert text.count("vector<32x32xf32>") == 2
    rows = re.findall(r"vector\.transfer_(read|write)[^\n]*vector<32xf32>", text)
    assert rows.count("read") == rows.count("write") == 2 * (32 + 32)
    assert text.count("vector.create_mask") == 1
    assert text.count("-> (vector<32xf32>)") == 32
    assert text.count("scf.if %") == 2 + 32 + 32


def test_split_copies_copies_many_rows_in_a_loop() -> None:
    # 128x4 of a padded tile: no unrolled row, and no n-D vector left.
    text = _apply(
        split_copies,
        """
func.func @f(%src: memref<?x?xf32>, %dst: memref<128x?xf32>) {
  %c0 = arith.constant 0 : index
  %z = arith.constant 0.0 : f32
  %r = vector.transfer_read %src[%c0, %c0], %z : memref<?x?xf32>, vector<128x4xf32>
  vector.transfer_write %r, %dst[%c0, %c0] {in_bounds = [true, false]} : vector<128x4xf32>, memref<128x?xf32>
  return
}
""",
    )
    assert "vector<128x4xf32>" not in text
    assert text.count("scf.for") == 2  # the in-bounds and the edge path
    assert text.count("vector.transfer_read") == 2


def test_unroll_split_copies_keeps_rows() -> None:
    # Unrolled, no copy is rebuilt from its rows (element by element).
    text = _apply(unroll_transfers, _apply(split_copies, PARTIAL_TILE_COPIES))
    assert "elements" not in text and "vector.insert " not in text


@pytest.mark.parametrize(
    ("name", "shape"),
    [("_pad_kernel", (33, 33)), ("_pack_a_kernel_t", (45, 70))],
    ids=["pad", "pack_a_t"],
)
def test_partial_tile_copies_lower_by_rows(name: str, shape: tuple[int, int]) -> None:
    """Regression: the edge tiles of a 32x32 copy lowered to about 10 MB of
    per-element LLVM code (shuffles of 1024 elements)."""
    matmul = importlib.import_module("helion_mlir_utils.matmul")
    padded = [hl.constexpr(-(-size // 32) * 32) for size in shape]
    if name.endswith("_t"):
        padded.reverse()
    text = str(lower_mlir(getattr(matmul, name), [torch.randn(shape), *padded]))
    assert not re.search(r"llvm\.shufflevector[^\n]*vector<\d{3,}x", text)
    assert text.count("llvm.insertelement") + text.count("llvm.extractelement") < 100
    assert len(text) < 200_000


# Loads of bf16 vectors: 33 elements (8k+1) and, left alone, 32 and 34.
BF16_VECTOR_LOADS = """
llvm.func @f(%p: !llvm.ptr) -> !llvm.struct<(vector<33xbf16>, vector<32xbf16>,
                                             vector<34xbf16>)> {
  %a = llvm.load %p <alignment = 2> : !llvm.ptr -> vector<33xbf16>
  %b = llvm.load %p <alignment = 2> : !llvm.ptr -> vector<32xbf16>
  %c = llvm.load %p <alignment = 2> : !llvm.ptr -> vector<34xbf16>
  %0 = llvm.mlir.poison : !llvm.struct<(vector<33xbf16>, vector<32xbf16>,
                                         vector<34xbf16>)>
  %1 = llvm.insertvalue %a, %0[0] : !llvm.struct<(vector<33xbf16>, vector<32xbf16>,
                                                    vector<34xbf16>)>
  %2 = llvm.insertvalue %b, %1[1] : !llvm.struct<(vector<33xbf16>, vector<32xbf16>,
                                                    vector<34xbf16>)>
  %3 = llvm.insertvalue %c, %2[2] : !llvm.struct<(vector<33xbf16>, vector<32xbf16>,
                                                    vector<34xbf16>)>
  llvm.return %3 : !llvm.struct<(vector<33xbf16>, vector<32xbf16>, vector<34xbf16>)>
}
"""


def test_split_bf16_loads_splits_the_last_element() -> None:
    # A load of a bf16 vector of 8k+1 elements aborts x86 code generation.
    text = _apply(split_bf16_loads, BF16_VECTOR_LOADS)
    loads = re.findall(r"llvm\.load [^\n]*-> vector<(\d+)xbf16>", text)
    assert sorted(loads) == ["1", "32", "32", "34"]
    assert "llvm.getelementptr %arg0[32] : (!llvm.ptr) -> !llvm.ptr, bf16" in text
    assert text.count("llvm.shufflevector") == 2


def _masked_bf16_load(width: int, mask: list[bool], element: str = "bf16") -> str:
    lanes = ", ".join("true" if lane else "false" for lane in mask)
    vector = f"vector<{width}x{element}>"
    return f"""
llvm.func @f(%p: !llvm.ptr, %z: {vector}) -> {vector} {{
  %m = llvm.mlir.constant(dense<[{lanes}]> : vector<{width}xi1>) : vector<{width}xi1>
  %r = llvm.intr.masked.load(%p, %m, %z), alignment(2) : (!llvm.ptr, vector<{width}xi1>, {vector}) -> {vector}
  llvm.return %r : {vector}
}}
"""


@pytest.mark.parametrize(("width", "kept"), [(64, 33), (96, 65), (192, 161), (40, 33)])
def test_split_bf16_loads_splits_a_masked_load_of_32k_plus_1(
    width: int, kept: int
) -> None:
    # Aborts x86 code generation (LegalizeDAG, a v32bf16 BUILD_VECTOR).
    mask = [lane < kept for lane in range(width)]
    text = _apply(split_bf16_loads, _masked_bf16_load(width, mask))
    assert f"llvm.getelementptr %arg0[{kept - 1}]" in text
    assert re.search(r"llvm\.load [^\n]*-> vector<1xbf16>", text)
    assert text.count("llvm.intr.masked.load") == 1
    assert text.count("llvm.shufflevector") == 2


@pytest.mark.parametrize(
    ("width", "mask", "element"),
    [
        # Other prefixes: selected as they are.
        (96, [lane < 64 for lane in range(96)], "bf16"),
        (96, [lane < 66 for lane in range(96)], "bf16"),
        (64, [lane < 17 for lane in range(64)], "bf16"),
        # No split part of one lane past the first.
        (32, [lane < 1 for lane in range(32)], "bf16"),
        # Not a prefix.
        (64, [lane != 0 and lane < 34 for lane in range(64)], "bf16"),
        # Another element type.
        (96, [lane < 65 for lane in range(96)], "f16"),
    ],
)
def test_split_bf16_loads_leaves_other_masked_loads(
    width: int, mask: list[bool], element: str
) -> None:
    text = _apply(split_bf16_loads, _masked_bf16_load(width, mask, element))
    assert "llvm.getelementptr" not in text
    assert text.count("llvm.intr.masked.load") == 1


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """AI-bench's 39_L2Norm_ kernel: ``x / ||x||_2`` per row, in two passes over
    column chunks."""
    m, n = x.shape
    hl.specialize(n)
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n].to(torch.float32)
            acc = acc + v * v
        inv = torch.rsqrt(acc.sum(-1, keepdim=True))
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = (x[tile_m, tile_n].to(torch.float32) * inv).to(
                x.dtype
            )
    return out


l2_normalize_kernel = helion.kernel(
    _l2_normalize, backend="mlir", config=helion.Config(block_sizes=[513, 8])
)


def test_ragged_bf16_l2_normalize_runs() -> None:
    """Regression: copying the partial last row tile of a 129x513 bf16 output
    by rows of 513 aborted LLVM code generation ("Do not know how to soft
    promote this operator's operand")."""
    torch.manual_seed(0)
    x = torch.randn(129, 513, dtype=torch.bfloat16)
    expected = x.float() / x.float().norm(dim=1, keepdim=True)
    actual = l2_normalize_kernel(x)
    torch.testing.assert_close(
        actual.float(), expected.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2
    )


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[8, 1024]))
def scalar_mul_kernel(a: torch.Tensor, s: hl.constexpr) -> torch.Tensor:
    """AI-bench's 5_Matrix_scalar_multiplication kernel."""
    out = torch.empty_like(a)
    for tile in hl.tile(a.shape):
        out[tile] = a[tile] * s
    return out


def test_ragged_bf16_scalar_mul_runs() -> None:
    """Regression: the rows of the partial last row tile of a 9x33 bf16 input,
    loaded as vectors of 33 elements, aborted LLVM code generation ("Do not
    know how to soft promote this operator's operand")."""
    torch.manual_seed(0)
    a = torch.randn(9, 33, dtype=torch.bfloat16)
    actual = scalar_mul_kernel(a, hl.constexpr(3.0))
    expected = (a.float() * 3.0).to(torch.bfloat16)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)


def _apply_bare(make_op: Callable[[ir.Value], object], source: str) -> tuple[str, str]:
    """``source`` as printed, and after the transform op alone (no cleanup)."""
    with ir.Context(), ir.Location.unknown():
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        before = str(module)
        with schedule_boilerplate() as (schedule, named_seq):
            make_op(lh_transform.match_op(named_seq.bodyTarget, "func.func"))
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return before, str(module)


# Rows of the 16x16x2 VNNI transpose, scaled down: four rows read and rebuilt
# before the bitcast and the row extracts (the input side), four shuffled rows
# rebuilt, bitcast and extracted to be written (the output side).
ROWS_INPUT_SIDE = """
func.func @f(%r0: vector<8xbf16>, %r1: vector<8xbf16>, %r2: vector<8xbf16>,
             %r3: vector<8xbf16>) -> (vector<4xi32>, vector<4xi32>) {
  %a:8 = vector.to_elements %r0 : vector<8xbf16>
  %b:8 = vector.to_elements %r1 : vector<8xbf16>
  %c:8 = vector.to_elements %r2 : vector<8xbf16>
  %d:8 = vector.to_elements %r3 : vector<8xbf16>
  %v = vector.from_elements %a#0, %a#1, %a#2, %a#3, %a#4, %a#5, %a#6, %a#7,
      %b#0, %b#1, %b#2, %b#3, %b#4, %b#5, %b#6, %b#7,
      %c#0, %c#1, %c#2, %c#3, %c#4, %c#5, %c#6, %c#7,
      %d#0, %d#1, %d#2, %d#3, %d#4, %d#5, %d#6, %d#7 : vector<4x4x2xbf16>
  %i = vector.bitcast %v : vector<4x4x2xbf16> to vector<4x4x1xi32>
  %s = vector.shape_cast %i : vector<4x4x1xi32> to vector<4x4xi32>
  %e0 = vector.extract %s[0] : vector<4xi32> from vector<4x4xi32>
  %e3 = vector.extract %s[3] : vector<4xi32> from vector<4x4xi32>
  return %e0, %e3 : vector<4xi32>, vector<4xi32>
}
"""

ROWS_OUTPUT_SIDE = """
func.func @f(%r0: vector<4xi32>, %r1: vector<4xi32>, %r2: vector<4xi32>,
             %r3: vector<4xi32>, %m: memref<4x64xbf16>, %j: index) {
  %c0 = arith.constant 0 : index
  %a:4 = vector.to_elements %r0 : vector<4xi32>
  %b:4 = vector.to_elements %r1 : vector<4xi32>
  %c:4 = vector.to_elements %r2 : vector<4xi32>
  %d:4 = vector.to_elements %r3 : vector<4xi32>
  %v = vector.from_elements %a#0, %a#1, %a#2, %a#3, %b#0, %b#1, %b#2, %b#3,
      %c#0, %c#1, %c#2, %c#3, %d#0, %d#1, %d#2, %d#3 : vector<4x4x1xi32>
  %p = vector.bitcast %v : vector<4x4x1xi32> to vector<4x4x2xbf16>
  %e0 = vector.extract %p[0] : vector<4x2xbf16> from vector<4x4x2xbf16>
  %f0 = vector.shape_cast %e0 : vector<4x2xbf16> to vector<8xbf16>
  vector.transfer_write %f0, %m[%c0, %j] {in_bounds = [true]}
      : vector<8xbf16>, memref<4x64xbf16>
  %e1 = vector.extract %p[1] : vector<4x2xbf16> from vector<4x4x2xbf16>
  %f1 = vector.shape_cast %e1 : vector<4x2xbf16> to vector<8xbf16>
  vector.transfer_write %f1, %m[%c0, %j] {in_bounds = [true]}
      : vector<8xbf16>, memref<4x64xbf16>
  return
}
"""

# Built from two rows, written whole: to a memref and to a tensor.
ROWS_WHOLE_WRITES = """
func.func @f(%r0: vector<4xf32>, %r1: vector<4xf32>, %m: memref<8x4xf32>,
             %t: tensor<8x4xf32>, %i: index) -> tensor<8x4xf32> {
  %c0 = arith.constant 0 : index
  %a:4 = vector.to_elements %r0 : vector<4xf32>
  %b:4 = vector.to_elements %r1 : vector<4xf32>
  %v = vector.from_elements %a#0, %a#1, %a#2, %a#3, %b#0, %b#1, %b#2, %b#3
      : vector<2x4xf32>
  vector.transfer_write %v, %m[%i, %c0] {in_bounds = [true, true]}
      : vector<2x4xf32>, memref<8x4xf32>
  %w = vector.transfer_write %v, %t[%i, %c0] {in_bounds = [true, true]}
      : vector<2x4xf32>, tensor<8x4xf32>
  return %w : tensor<8x4xf32>
}
"""


def test_forward_rows_input_side() -> None:
    _, text = _apply_bare(forward_rows_op, ROWS_INPUT_SIDE)
    assert "from_elements" not in text
    # Each row cast flat: no 2-D bitcast (lowered one element at a time).
    assert text.count("vector.bitcast %arg0 : vector<8xbf16> to vector<4xi32>") == 1
    assert text.count("vector.bitcast %arg3 : vector<8xbf16> to vector<4xi32>") == 1
    assert "x1xi32" not in text
    assert "return %4, %7 : vector<4xi32>, vector<4xi32>" in text


def test_forward_rows_output_side() -> None:
    text = _apply(forward_rows, ROWS_OUTPUT_SIDE)
    assert "from_elements" not in text and "to_elements" not in text
    assert "vector.bitcast %arg0 : vector<4xi32> to vector<8xbf16>" in text
    assert "vector.bitcast %arg1 : vector<4xi32> to vector<8xbf16>" in text
    assert text.count("vector.transfer_write") == 2
    assert "shape_cast" not in text


def test_forward_rows_whole_writes() -> None:
    text = _apply(forward_rows, ROWS_WHOLE_WRITES)
    assert "from_elements" not in text and "to_elements" not in text
    writes = re.findall(r"vector\.transfer_write (%\w+), (%\w+)\[", text)
    assert writes.count(("%arg0", "%arg2")) == writes.count(("%arg1", "%arg2")) == 1
    # On the tensor, chained: the second row into the first row's result.
    assert ("%arg0", "%arg3") in writes and ("%arg1", "%1") in writes


# Each left alone: not whole rows in order, or a use not row by row.
_TWO_ROWS = """
func.func @f(%r0: vector<4xf32>, %r1: vector<4xf32>, %s: f32, %m: memref<8x4xf32>,
             %i: index) -> {result_type} {{
  %c0 = arith.constant 0 : index
  %a:4 = vector.to_elements %r0 : vector<4xf32>
  %b:4 = vector.to_elements %r1 : vector<4xf32>
  %v = vector.from_elements {elements} : {built_type}
  {use}
}}
"""
_ROWS = "%a#0, %a#1, %a#2, %a#3, %b#0, %b#1, %b#2, %b#3"
_EXTRACT = (
    "%e = vector.extract %v[1] : vector<4xf32> from vector<2x4xf32>\n"
    "  return %e : vector<4xf32>"
)
_UNCHANGED = {
    "partial_row": ("%a#0, %a#1, %b#0, %b#1, %a#2, %a#3, %b#2, %b#3", _EXTRACT),
    "out_of_order": ("%a#1, %a#0, %a#2, %a#3, %b#0, %b#1, %b#2, %b#3", _EXTRACT),
    "row_twice_in_part": ("%a#0, %a#1, %a#2, %a#3, %a#0, %a#1, %b#2, %b#3", _EXTRACT),
    "scalar": ("%a#0, %a#1, %a#2, %a#3, %b#0, %b#1, %b#2, %s", _EXTRACT),
    "returned": (_ROWS, "return %v : vector<2x4xf32>"),
    "flattened": (
        _ROWS,
        (
            "%f = vector.shape_cast %v : vector<2x4xf32> to vector<8xf32>\n"
            "  return %f : vector<8xf32>"
        ),
    ),
    "leading_dim_changed": (
        _ROWS,
        (
            "%f = vector.shape_cast %v : vector<2x4xf32> to vector<4x2xf32>\n"
            "  %e = vector.extract %f[0] : vector<2xf32> from vector<4x2xf32>\n"
            "  return %e : vector<2xf32>"
        ),
    ),
    "element_extract": (
        _ROWS,
        "%e = vector.extract %v[1, 2] : f32 from vector<2x4xf32>\n  return %e : f32",
    ),
    "dynamic_extract": (
        _ROWS,
        (
            "%e = vector.extract %v[%i] : vector<4xf32> from vector<2x4xf32>\n"
            "  return %e : vector<4xf32>"
        ),
    ),
    "masked_write": (
        _ROWS,
        (
            "%k = vector.create_mask %i, %i : vector<2x4xi1>\n"
            "  vector.transfer_write %v, %m[%c0, %c0], %k {in_bounds = [true, true]}"
            " : vector<2x4xf32>, memref<8x4xf32>\n  return"
        ),
    ),
    "out_of_bounds_write": (
        _ROWS,
        (
            "vector.transfer_write %v, %m[%i, %c0] {in_bounds = [false, true]}"
            " : vector<2x4xf32>, memref<8x4xf32>\n  return"
        ),
    ),
    "permuted_write": (
        _ROWS,
        (
            "vector.transfer_write %v, %m[%c0, %c0] {in_bounds = [true, true],"
            " permutation_map = affine_map<(d0, d1) -> (d1, d0)>}"
            " : vector<2x4xf32>, memref<8x4xf32>\n  return"
        ),
    ),
    "one_row_use_other_not": (
        _ROWS,
        (
            "%e = vector.extract %v[1] : vector<4xf32> from vector<2x4xf32>\n"
            "  return %v : vector<2x4xf32>"
        ),
    ),
}


def _unchanged_case(elements: str, use: str) -> str:
    result = re.search(r"return( %\w+ : ([^\n]+))?$", use)
    result_type = result.group(2) if result.group(1) else "()"
    return _TWO_ROWS.format(
        elements=elements,
        built_type="vector<2x4xf32>",
        use=use,
        result_type=result_type,
    )


def test_forward_rows_forwards_whole_rows_of_the_template() -> None:
    # The cases below differ from this one only in what they name.
    before, after = _apply_bare(forward_rows_op, _unchanged_case(_ROWS, _EXTRACT))
    assert "from_elements" in before and "from_elements" not in after
    assert "return %arg1 : vector<4xf32>" in after


@pytest.mark.parametrize("case", sorted(_UNCHANGED))
def test_forward_rows_leaves_others_alone(case: str) -> None:
    before, after = _apply_bare(forward_rows_op, _unchanged_case(*_UNCHANGED[case]))
    assert after == before


RANK_1_ROWS = """
func.func @f(%r0: vector<4xf32>) -> vector<4xf32> {
  %a:4 = vector.to_elements %r0 : vector<4xf32>
  %v = vector.from_elements %a#0, %a#1, %a#2, %a#3 : vector<4xf32>
  return %v : vector<4xf32>
}
"""


def test_forward_rows_leaves_rank_1_alone() -> None:
    before, after = _apply_bare(forward_rows_op, RANK_1_ROWS)
    assert after == before


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[8]))
def reciprocal_rows_kernel(x: torch.Tensor) -> torch.Tensor:
    """Rows scaled by the reciprocal of their sum of magnitudes (AI-bench's
    38_L1Norm_, in one pass)."""
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        out[tile_m, :] = row * (1.0 / row.abs().sum(-1, keepdim=True))
    return out


def _softmax(x: torch.Tensor) -> torch.Tensor:
    """AI-bench's 23_Softmax kernel: row softmax in passes over column chunks."""
    m, n = x.shape
    hl.specialize(n)
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n].to(torch.float32))
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top)
        inv = 1.0 / total.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            e = torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top)
            out[tile_m, tile_n] = (e * inv).to(x.dtype)
    return out


softmax_kernel = helion.kernel(
    _softmax, backend="mlir", config=helion.Config(block_sizes=[64, 2])
)


def test_ragged_reciprocal_runs() -> None:
    """Regression: torch-mlir's reciprocal asserts its input is nonzero; the
    zero-padded rows of the partial last row tile (9 rows, tiles of 8) failed
    it and aborted the kernel."""
    torch.manual_seed(0)
    x = torch.randn(9, 16)
    expected = x / x.abs().sum(-1, keepdim=True)
    torch.testing.assert_close(reciprocal_rows_kernel(x), expected)


def test_ragged_softmax_runs() -> None:
    """Regression: as for the reciprocal, on a 33x64 input in tiles of 2 rows."""
    torch.manual_seed(0)
    x = torch.randn(33, 64)
    torch.testing.assert_close(softmax_kernel(x), torch.softmax(x, dim=-1))


def test_lowered_reciprocal_has_no_asserts() -> None:
    # A cf.assert lowers to a call to abort (its message is dropped).
    text = str(lower_mlir(reciprocal_rows_kernel, [torch.randn(9, 16)]))
    assert "@abort" not in text


# A cast of VNNI pairs between the reshapes of flattened transfers.
CAST_BETWEEN_RESHAPES = """
func.func @f(%src: memref<64xf32>, %dst: memref<64xbf16>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %read = vector.transfer_read %src[%c0], %pad {in_bounds = [true]}
      : memref<64xf32>, vector<32xf32>
  %pairs = vector.shape_cast %read : vector<32xf32> to vector<16x2xf32>
  %cast = arith.truncf %pairs : vector<16x2xf32> to vector<16x2xbf16>
  %flat = vector.shape_cast %cast : vector<16x2xbf16> to vector<32xbf16>
  vector.transfer_write %flat, %dst[%c0] {in_bounds = [true]}
      : vector<32xbf16>, memref<64xbf16>
  return
}
"""


def test_linearize_vector_ops_casts_whole_vectors() -> None:
    # Lowered 2-D, the cast would be unrolled into 16 casts of pairs.
    text = _apply(linearize_vector_ops, CAST_BETWEEN_RESHAPES)
    assert "arith.truncf %0 : vector<32xf32> to vector<32xbf16>" in text
    assert "vector.shape_cast" not in text


CONSTANT_OFFSET_READS = """
func.func @f(%a: memref<64x64xbf16>, %i: index)
    -> (vector<16x16xbf16>, vector<16x16xbf16>) {
  %pad = arith.constant 0.0 : bf16
  %i16 = affine.apply affine_map<(d0) -> (d0 + 16)>(%i)
  %0 = vector.transfer_read %a[%i, %i], %pad {in_bounds = [true, true]}
      : memref<64x64xbf16>, vector<16x16xbf16>
  %1 = vector.transfer_read %a[%i, %i16], %pad {in_bounds = [true, true]}
      : memref<64x64xbf16>, vector<16x16xbf16>
  return %0, %1 : vector<16x16xbf16>, vector<16x16xbf16>
}
"""


def _subviews(text: str) -> list[tuple[str, str]]:
    """The source and offsets of every ``memref.subview`` in ``text``."""
    return re.findall(r"memref\.subview (%\w+)\[([^\]]*)\]", text)


def test_amx_move_offsets_constant_offsets_share_subview() -> None:
    text = _apply(amx_move_offsets, CONSTANT_OFFSET_READS)
    assert _subviews(text) == [("%arg0", "%arg1, %arg1")]
    assert "vector.transfer_read %subview[%c0, %c16]" in text


# A K loop over a runtime-shape operand, at an offset of k + j + 16.
LOOP_OFFSET_READ = """
func.func @f(%a: memref<?x?xbf16>, %j: index, %n: index) -> vector<16x16xbf16> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %init = arith.constant dense<0.0> : vector<16x16xbf16>
  %r = scf.for %k = %c0 to %n step %c16 iter_args(%acc = %init) -> vector<16x16xbf16> {
    %off = affine.apply affine_map<(d0)[s0] -> (d0 + s0 + 16)>(%k)[%j]
    %v = vector.transfer_read %a[%c0, %off], %pad {in_bounds = [true, true]}
        : memref<?x?xbf16>, vector<16x16xbf16>
    %s = arith.addf %acc, %v : vector<16x16xbf16>
    scf.yield %s : vector<16x16xbf16>
  }
  return %r : vector<16x16xbf16>
}
"""


def test_amx_move_offsets_splits_loop_offset() -> None:
    text = _apply(amx_move_offsets, LOOP_OFFSET_READ)
    # The K loop's induction variable is the offset of an inner subview, the
    # rest of the offset of an outer one; the constant stays in the read.
    assert _subviews(text) == [("%arg0", "%c0, %arg1"), ("%subview", "%c0, %arg3")]
    assert "vector.transfer_read %subview_2[%c0, %c16]" in text
    assert text.count("memref.dim") == 2


# Reads whose bounds are checked or masked: a subview would change their bounds.
UNSAFE_READS = """
func.func @f(%a: memref<?x?xf32>, %i: index, %m: vector<16xi1>)
    -> (vector<16xf32>, vector<16xf32>, vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %a[%c0, %i], %pad : memref<?x?xf32>, vector<16xf32>
  %1 = vector.transfer_read %a[%c0, %i], %pad, %m {in_bounds = [true]}
      : memref<?x?xf32>, vector<16xf32>
  %2 = vector.mask %m {
    vector.transfer_read %a[%c0, %i], %pad {in_bounds = [true]}
        : memref<?x?xf32>, vector<16xf32>
  } : vector<16xi1> -> vector<16xf32>
  return %0, %1, %2 : vector<16xf32>, vector<16xf32>, vector<16xf32>
}
"""


def test_amx_move_offsets_keeps_bounded_reads() -> None:
    assert _subviews(_apply(amx_move_offsets, UNSAFE_READS)) == []


# A batch-reduce and a K loop reading at offsets of their bare induction variables.
TWO_REDUCTION_LOOPS_READ = """
func.func @f(%a: memref<16x64x64xbf16>, %m: index) -> vector<1x16x16xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.0 : bf16
  %init = arith.constant dense<0.0> : vector<1x16x16xbf16>
  %r = scf.for %b = %c0 to %c16 step %c1 iter_args(%acc = %init) -> vector<1x16x16xbf16> {
    %r2 = scf.for %k = %c0 to %c64 step %c16 iter_args(%acc2 = %acc) -> vector<1x16x16xbf16> {
      %v = vector.transfer_read %a[%b, %m, %k], %pad {in_bounds = [true, true, true]}
          : memref<16x64x64xbf16>, vector<1x16x16xbf16>
      %s = arith.addf %acc2, %v : vector<1x16x16xbf16>
      scf.yield %s : vector<1x16x16xbf16>
    }
    scf.yield %r2 : vector<1x16x16xbf16>
  }
  return %r : vector<1x16x16xbf16>
}
"""


def test_amx_move_offsets_keeps_reduction_ivs_in_one_subview() -> None:
    # The AMX rewrite finds the induction variables of both reduction loops in
    # the offsets of the same subview.
    text = _apply(amx_move_offsets, TWO_REDUCTION_LOOPS_READ)
    assert _subviews(text) == [("%arg0", "%arg2, %arg1, %arg4")]


# A K loop around a loop over N tiles: the read is in the N loop.
OUTER_REDUCTION_LOOP_READ = """
func.func @f(%a: memref<64x128xbf16>, %j: index, %out: memref<64x128xbf16>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.0 : bf16
  scf.for %k = %c0 to %c64 step %c16 {
    scf.for %n = %c0 to %c32 step %c16 {
      %off = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%k)[%j]
      %v = vector.transfer_read %a[%c0, %off], %pad {in_bounds = [true, true]}
          : memref<64x128xbf16>, vector<16x16xbf16>
      vector.transfer_write %v, %out[%c0, %n] {in_bounds = [true, true]}
          : vector<16x16xbf16>, memref<64x128xbf16>
    }
  }
  return
}
"""


def test_amx_move_offsets_splits_innermost_loop_only() -> None:
    # Only the innermost loop's induction variable is split out: the K offset
    # stays whole in a single subview.
    text = _apply(amx_move_offsets, OUTER_REDUCTION_LOOP_READ)
    assert _subviews(text) == [("%arg0", "0, %0")]
    assert "%0 = affine.apply #map(%arg3)[%arg1]" in text


# A 16x16 block of VNNI pairs transposed, as a pack of [N, K] into [K/2, N, 2].
PAIR_TRANSPOSE = """
func.func @f(%v: vector<16x16x2xbf16>) -> vector<16x16x2xbf16> {
  %t = vector.transpose %v, [1, 0, 2] : vector<16x16x2xbf16> to vector<16x16x2xbf16>
  return %t : vector<16x16x2xbf16>
}
"""


def test_lower_transposes_widens_kept_inner_dims() -> None:
    """Pairs kept whole are one i32 each: a 16x16 transpose of i32, lowered to
    the shuffle sequence, not per-element extracts."""
    text = _apply(lower_transposes, PAIR_TRANSPOSE)
    assert "vector.transpose" not in text
    assert "vector<16x16xi32>" in text or "vector<16xi32>" in text
    assert text.count("vector.bitcast") == 2
    assert "vector.shuffle" in text


# A 32x16 block of bf16 transposed (e.g. a transposed A tile).
NARROW_TRANSPOSE = """
func.func @f(%v: vector<32x16xbf16>) -> vector<16x32xbf16> {
  %t = vector.transpose %v, [1, 0] : vector<32x16xbf16> to vector<16x32xbf16>
  return %t : vector<16x32xbf16>
}
"""


def test_lower_transposes_pairs_narrow_elements() -> None:
    """Row pairs interleaved, then a 16x16 transpose of the i32 pairs: shuffles
    only."""
    text = _apply(lower_transposes, NARROW_TRANSPOSE)
    assert "vector.transpose" not in text
    assert "vector.bitcast" in text
    assert "vector.shuffle" in text
    assert ": bf16 from" not in text


# A [K, M] tile of bf16 packed into VNNI pairs [M, K/2, 2] (a transposed A).
PAIR_PACK_TRANSPOSE = """
func.func @f(%v: vector<16x2x32xbf16>) -> vector<32x16x2xbf16> {
  %t = vector.transpose %v, [2, 0, 1] : vector<16x2x32xbf16> to vector<32x16x2xbf16>
  return %t : vector<32x16x2xbf16>
}
"""


def test_lower_transposes_merges_adjacent_dims() -> None:
    """K/2 and the pair stay adjacent: a 32x32 transpose of bf16, lowered to
    shuffles, not per element."""
    text = _apply(lower_transposes, PAIR_PACK_TRANSPOSE)
    assert "vector.transpose" not in text
    assert "vector.shuffle" in text
    assert ": bf16 from" not in text


# A 16x16x2 block of a strided source: rows of 32 contiguous elements.
STRIDED_BLOCK_READ = """
func.func @f(%m: memref<128x1024x2xbf16, strided<[4096, 2, 1]>>) -> vector<16x16x2xbf16> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : bf16
  %r = vector.transfer_read %m[%c0, %c0, %c0], %pad {in_bounds = [true, true, true]}
      : memref<128x1024x2xbf16, strided<[4096, 2, 1]>>, vector<16x16x2xbf16>
  return %r : vector<16x16x2xbf16>
}
"""


def test_unroll_transfers_reads_contiguous_rows() -> None:
    text = _apply(unroll_transfers, STRIDED_BLOCK_READ)
    reads = re.findall(r"vector\.transfer_read[^\n]*(vector<[^>]*>)", text)
    assert reads and set(reads) == {"vector<32xbf16>"} and len(reads) == 16
    assert "memref.alloca" not in text


ZERO_D_TRANSFERS = """
func.func @f(%m: memref<1x1xf32, strided<[?, ?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %read = vector.transfer_read %m[%c0, %c0], %pad : memref<1x1xf32, strided<[?, ?], offset: ?>>, vector<f32>
  vector.transfer_write %read, %m[%c0, %c0] : vector<f32>, memref<1x1xf32, strided<[?, ?], offset: ?>>
  return
}
"""


def test_legalize_0d_transfers() -> None:
    text = _apply(legalize_for_llvm, ZERO_D_TRANSFERS)
    assert "vector.transfer" not in text
    assert "memref.load" in text
    assert "memref.store" in text


ALLOCA_IN_LOOP = """
func.func @f(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %buffer = memref.alloca() : memref<f32>
    %value = memref.load %buffer[] : memref<f32>
    memref.store %value, %buffer[] : memref<f32>
  }
  return
}
"""


def test_hoist_allocas_out_of_loops() -> None:
    text = _apply(hoist_allocas, ALLOCA_IN_LOOP)
    assert text.index("memref.alloca") < text.index("scf.for")


REGISTER_TILE_TEMPORARY = """
func.func @f(%x: tensor<4x32xf32>) -> tensor<4x32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %whole = tensor.empty() : tensor<4x32xf32>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%out = %x) -> (tensor<4x32xf32>) {
    %slice = tensor.extract_slice %whole[%i, 0] [1, 32] [1, 1] : tensor<4x32xf32> to tensor<1x32xf32>
    %zero = linalg.fill ins(%cst : f32) outs(%slice : tensor<1x32xf32>) -> tensor<1x32xf32>
    %next = tensor.insert_slice %zero into %out[%i, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
    scf.yield %next : tensor<4x32xf32>
  }
  return %r : tensor<4x32xf32>
}
"""


def test_fold_empty_slices_sizes_temporaries_per_register_tile() -> None:
    text = _apply(fold_empty_slices, REGISTER_TILE_TEMPORARY)
    assert "tensor.empty() : tensor<1x32xf32>" in text
    assert "tensor.empty() : tensor<4x32xf32>" not in text


# A register tile of a partial tile's padded operand, as tiling fuses the pad: a
# guard for an empty source slice, else a pad of the slice.
GUARDED_PADDED_OPERAND = """
func.func @f(%a: tensor<?x64xf32>, %b: tensor<64x32xf32>, %c: tensor<32x32xf32>,
             %n: index) -> tensor<32x32xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %empty = arith.cmpi eq, %n, %c0 : index
  %high = affine.apply affine_map<()[s0] -> (32 - s0)>()[%n]
  %lhs = scf.if %empty -> (tensor<32x64xf32>) {
    %g = tensor.generate {
    ^bb0(%i: index, %j: index):
      tensor.yield %zero : f32
    } : tensor<32x64xf32>
    scf.yield %g : tensor<32x64xf32>
  } else {
    %s = tensor.extract_slice %a[0, 0] [%n, 64] [1, 1] : tensor<?x64xf32> to tensor<?x64xf32>
    %p = tensor.pad %s low[0, 0] high[%high, 0] {
    ^bb0(%i: index, %j: index):
      tensor.yield %zero : f32
    } : tensor<?x64xf32> to tensor<32x64xf32>
    scf.yield %p : tensor<32x64xf32>
  }
  %r = linalg.matmul ins(%lhs, %b : tensor<32x64xf32>, tensor<64x32xf32>)
                     outs(%c : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}
"""


def test_version_padded_operands() -> None:
    """The contraction is sunk into the guard and branched on the pad padding
    nothing: all padding, the source in place, or the padded source."""
    text = _apply(version_padded_operands, GUARDED_PADDED_OPERAND)
    assert text.count("linalg.matmul") == 3
    assert text.count("scf.if") == 2
    assert text.count("tensor.pad") == 1
    assert "tensor.cast" in text or "[32, 64] [1, 1]" in text


def test_materialize_versioned_operand_pad_by_rows() -> None:
    """Only the padded version copies its operand, row by row into a register
    tile buffer, filling the rows past the source."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(GUARDED_PADDED_OPERAND)
        for make_schedule in (version_padded_operands, materialize_operand_pads):
            schedule = make_schedule()
            schedule.body.operations[0].apply(module.operation)
            module.operation.verify()
        text = str(module)
    assert "tensor.pad" not in text
    assert "tensor.generate" not in text
    assert "linalg.copy" in text
    assert "linalg.fill" in text
    assert text.count("linalg.matmul") == 3


# A K-tail register tile: rows padded at runtime, columns statically, read
# through reshapes (e.g. into VNNI pairs and back).
GUARDED_TAIL_OPERAND = """
func.func @f(%a: tensor<?x16xf32>, %b: tensor<64x32xf32>, %c: tensor<32x32xf32>,
             %n: index) -> tensor<32x32xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %empty = arith.cmpi eq, %n, %c0 : index
  %high = affine.apply affine_map<()[s0] -> (32 - s0)>()[%n]
  %lhs = scf.if %empty -> (tensor<32x64xf32>) {
    %g = tensor.generate {
    ^bb0(%i: index, %j: index):
      tensor.yield %zero : f32
    } : tensor<32x64xf32>
    scf.yield %g : tensor<32x64xf32>
  } else {
    %s = tensor.extract_slice %a[0, 0] [%n, 16] [1, 1] : tensor<?x16xf32> to tensor<?x16xf32>
    %p = tensor.pad %s low[0, 0] high[%high, 48] {
    ^bb0(%i: index, %j: index):
      tensor.yield %zero : f32
    } : tensor<?x16xf32> to tensor<32x64xf32>
    scf.yield %p : tensor<32x64xf32>
  }
  %pairs = tensor.expand_shape %lhs [[0], [1, 2]] output_shape [32, 32, 2]
      : tensor<32x64xf32> into tensor<32x32x2xf32>
  %rows = tensor.collapse_shape %pairs [[0], [1, 2]]
      : tensor<32x32x2xf32> into tensor<32x64xf32>
  %r = linalg.matmul ins(%rows, %b : tensor<32x64xf32>, tensor<64x32xf32>)
                     outs(%c : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}
"""


def test_version_padded_operands_sinks_reshaped_static_pad() -> None:
    """A statically padded operand is never read in place, but its guard is
    sunk so the pad, read through the reshapes, is materialized by rows."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(GUARDED_TAIL_OPERAND)
        for make_schedule in (version_padded_operands, materialize_operand_pads):
            schedule = make_schedule()
            schedule.body.operations[0].apply(module.operation)
            module.operation.verify()
        text = str(module)
    assert text.count("linalg.matmul") == 2
    assert "tensor.cast" not in text
    padded = re.findall(
        r"tensor\.pad[\s\S]*?\} : tensor<[^>]*> to (tensor<[^>]*>)", text
    )
    assert padded and set(padded) == {"tensor<64xf32>"}


# Register tiles along N reading the same A tile in place, from rows of
# ``{cols}`` bf16 elements.
_ROW_TILE_OPERAND = """
func.func @f(%a: tensor<64x{cols}xbf16>, %b: tensor<32x128xbf16>,
             %c: tensor<32x128xf32>, %m: index) -> tensor<32x128xf32> {{
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %r = scf.for %j = %c0 to %c128 step %c32 iter_args(%acc = %c) -> (tensor<32x128xf32>) {{
    %row = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%m]
    %s = tensor.extract_slice %a[%row, 0] [32, 32] [1, 1]
        : tensor<64x{cols}xbf16> to tensor<32x32xbf16>
    %bs = tensor.extract_slice %b[0, %j] [32, 32] [1, 1]
        : tensor<32x128xbf16> to tensor<32x32xbf16>
    %cs = tensor.extract_slice %acc[0, %j] [32, 32] [1, 1]
        : tensor<32x128xf32> to tensor<32x32xf32>
    %mm = linalg.matmul ins(%s, %bs : tensor<32x32xbf16>, tensor<32x32xbf16>)
                        outs(%cs : tensor<32x32xf32>) -> tensor<32x32xf32>
    %next = tensor.insert_slice %mm into %acc[0, %j] [32, 32] [1, 1]
        : tensor<32x32xf32> into tensor<32x128xf32>
    scf.yield %next : tensor<32x128xf32>
  }}
  return %r : tensor<32x128xf32>
}}
"""


def test_align_operand_rows_copies_misaligned_rows_once() -> None:
    """Rows of 36 bf16 (72 bytes) are copied into an aligned buffer, before
    the loop over N tiles: once per A tile."""
    text = _apply(align_operand_rows, _ROW_TILE_OPERAND.format(cols=36))
    assert "linalg.copy" in text
    assert text.index("-> (tensor<32x32xbf16>)") < text.index("-> (tensor<32x128xf32>)")


def test_align_operand_rows_reads_aligned_rows_in_place() -> None:
    text = _apply(align_operand_rows, _ROW_TILE_OPERAND.format(cols=64))
    assert "linalg.copy" not in text
    # Hoisted all the same: the A tile does not vary along N.
    assert text.index("tensor.extract_slice %arg0") < text.index("scf.for")


# GUARDED_PADDED_OPERAND of a source with misaligned rows (36 bf16).
GUARDED_MISALIGNED_OPERAND = """
func.func @f(%a: tensor<64x36xbf16>, %b: tensor<32x32xbf16>, %c: tensor<32x32xf32>,
             %n: index) -> tensor<32x32xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : bf16
  %empty = arith.cmpi eq, %n, %c0 : index
  %high = affine.apply affine_map<()[s0] -> (32 - s0)>()[%n]
  %lhs = scf.if %empty -> (tensor<32x32xbf16>) {
    %g = tensor.generate {
    ^bb0(%i: index, %j: index):
      tensor.yield %zero : bf16
    } : tensor<32x32xbf16>
    scf.yield %g : tensor<32x32xbf16>
  } else {
    %s = tensor.extract_slice %a[0, 0] [%n, 32] [1, 1] : tensor<64x36xbf16> to tensor<?x32xbf16>
    %p = tensor.pad %s low[0, 0] high[%high, 0] {
    ^bb0(%i: index, %j: index):
      tensor.yield %zero : bf16
    } : tensor<?x32xbf16> to tensor<32x32xbf16>
    scf.yield %p : tensor<32x32xbf16>
  }
  %r = linalg.matmul ins(%lhs, %b : tensor<32x32xbf16>, tensor<32x32xbf16>)
                     outs(%c : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}
"""


def test_misaligned_padded_operand_is_copied_unversioned() -> None:
    """Misaligned rows are never read in place: no version, and the guard of
    the empty slice folds into the row copy (which fills every row)."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(GUARDED_MISALIGNED_OPERAND)
        for make_schedule in (version_padded_operands, materialize_operand_pads):
            schedule = make_schedule()
            schedule.body.operations[0].apply(module.operation)
            module.operation.verify()
        text = str(module)
    assert text.count("linalg.matmul") == 1
    assert "tensor.generate" not in text
    assert "tensor.pad" not in text
    assert "tensor.cast" not in text
    assert "linalg.copy" in text


_TILE_A = "!x86.amx.tile<16x32xbf16>"
_TILE_C = "!x86.amx.tile<16x16xf32>"


def _amx_block(body: str, accumulators: int = 4) -> str:
    """A K loop of AMX dot-products on 32x64 operands ``%a``/``%b``."""
    accs = ", ".join(f"%x{i} = %zero" for i in range(accumulators))
    types = ", ".join([_TILE_C] * accumulators)
    return f"""
func.func @f(%a: memref<32x64xbf16>, %b: memref<32x128xbf16>, %c: memref<16x16xf32>, %n: index) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c96 = arith.constant 96 : index
  %zero = x86.amx.tile_zero : {_TILE_C}
  %r:{accumulators} = scf.for %i = %c0 to %n step %c1 iter_args({accs}) -> ({types}) {{
{body}
  }}
  x86.amx.tile_store %c[%c0, %c0], %r#0 : memref<16x16xf32>, {_TILE_C}
  return
}}
"""


def _load(name: str, buffer: str, row: str, col: str) -> str:
    shape = "32x64" if buffer == "a" else "32x128"
    return (
        f"    %{name} = x86.amx.tile_load %{buffer}[%{row}, %{col}] : "
        f"memref<{shape}xbf16> into {_TILE_A}"
    )


def _dot(name: str, lhs: str, rhs: str, acc: str) -> str:
    return (
        f"    %{name} = x86.amx.tile_mulf %{lhs}, %{rhs}, %{acc} : "
        f"{_TILE_A}, {_TILE_A}, {_TILE_C}"
    )


_AMX_2X2_BODY = "\n".join(
    [
        _load("a0", "a", "c0", "c0"),
        _load("b0", "b", "c0", "c0"),
        _load("b1", "b", "c0", "c32"),
        _load("a1", "a", "c16", "c0"),
        _dot("d0", "a0", "b0", "x0"),
        _dot("d1", "a0", "b1", "x1"),
        _dot("d2", "a1", "b0", "x2"),
        _dot("d3", "a1", "b1", "x3"),
        f"    scf.yield %d0, %d1, %d2, %d3 : {', '.join([_TILE_C] * 4)}",
    ]
)


def _amx_order(text: str) -> list[str]:
    """The AMX loads (``L``) and dot-products (``D``) of the K loop, in order."""
    body = text[text.index("scf.for") : text.index("scf.yield")]
    return [
        "L" if "tile_load" in line else "D"
        for line in body.splitlines()
        if "tile_load" in line or "tile_mulf" in line
    ]


def test_schedule_amx_loads_interleaves() -> None:
    text = _apply(schedule_amx_loads, _amx_block(_AMX_2X2_BODY))
    # oneDNN's order: A0, B0, B1, dot, A1, dot, dot, dot.
    assert _amx_order(text) == ["L", "L", "L", "D", "L", "D", "D", "D"]


def test_schedule_amx_loads_keeps_blocks_writing_memory() -> None:
    body = _AMX_2X2_BODY.replace(
        _dot("d1", "a0", "b1", "x1"),
        f"    x86.amx.tile_store %c[%c0, %c0], %d0 : memref<16x16xf32>, {_TILE_C}\n"
        + _dot("d1", "a0", "b1", "x1"),
    )
    text = _apply(schedule_amx_loads, _amx_block(body))
    assert _amx_order(text) == ["L", "L", "L", "L", "D", "D", "D", "D"]


def test_schedule_amx_loads_keeps_tile_register_limit() -> None:
    # 1x5 blocking: 5 accumulators, all 6 operand tiles loaded ahead would need 11.
    loads = [_load("a0", "a", "c0", "c0")] + [
        _load(f"b{j}", "b", f"c{16 * (j % 2)}", col)
        for j, col in enumerate(["c0", "c32", "c64", "c96", "c0"])
    ]
    dots = [_dot(f"d{j}", "a0", f"b{j}", f"x{j}") for j in range(5)]
    interleaved = [loads[0], loads[1], dots[0]]
    for j in range(1, 5):
        interleaved += [loads[j + 1], dots[j]]
    body = "\n".join(
        [
            *interleaved,
            f"    scf.yield {', '.join(f'%d{j}' for j in range(5))} : {', '.join([_TILE_C] * 5)}",
        ]
    )
    text = _apply(lambda: schedule_amx_loads(distance=5), _amx_block(body, 5))
    assert _amx_order(text) == ["L", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D"]


# A contraction generic, as elementwise fusion leaves it: $A, $B, $C are the
# element types of the inputs and the init, $BODY computes %r from %x, %y and
# %acc (the block arguments).
_CONTRACTION = """
#a = affine_map<(m, n, k) -> (m, k)>
#b = affine_map<(m, n, k) -> (k, n)>
#c = affine_map<(m, n, k) -> (m, n)>
func.func @f(%a: tensor<$Mx8x$A>, %b: tensor<8x16x$B>, %c: tensor<$Mx16x$C>)
    -> tensor<$Mx16x$C> {
  %0 = linalg.generic {indexing_maps = [#a, #b, #c],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<$Mx8x$A>, tensor<8x16x$B>) outs(%c : tensor<$Mx16x$C>) {
  ^bb0(%x: $A, %y: $B, %acc: $C):
$BODY
    linalg.yield %r : $C
  } -> tensor<$Mx16x$C>
  return %0 : tensor<$Mx16x$C>
}
"""


def _contraction(body: str, a: str = "f32", b: str = "f32", c: str = "f32") -> str:
    return (
        _CONTRACTION.replace("$BODY", body)
        .replace("$M", "4")
        .replace("$A", a)
        .replace("$B", b)
        .replace("$C", c)
    )


def _split_and_raise(source: str) -> tuple[str, str]:
    """``source`` after ``split_contraction_operands``, and then raised to
    named contractions as ``linalg_contract_fold_unit_dims`` does."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        schedule = split_contraction_operands()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        split = str(module)
        passmanager.PassManager.parse(
            "builtin.module(func.func(linalg-morph-ops{generic-to-category}))"
        ).run(module.operation)
        return split, str(module)


def _contract_inputs(text: str) -> str:
    """The input types of the one ``linalg.contract`` in ``text``."""
    (line,) = [line for line in text.splitlines() if "linalg.contract" in line]
    return line[line.index("ins(") : line.index(") outs(")].split(" : ")[1]


def test_split_contraction_operands_exp() -> None:
    body = """    %e = math.exp %x : f32
    %m = arith.mulf %e, %y : f32
    %r = arith.addf %acc, %m : f32"""
    split, raised = _split_and_raise(_contraction(body))
    # exp over A's 4x8 elements, then the contraction of its result.
    assert split.count("math.exp") == 1
    assert "outs(%0 : tensor<4x8xf32>)" in split
    assert "linalg.elementwise <exp> ins(%arg0 : tensor<4x8xf32>)" in raised
    assert _contract_inputs(raised) == "tensor<4x8xf32>, tensor<8x16xf32>"


def test_split_contraction_operands_keeps_trailing_cast() -> None:
    # The bf16 rounding of A (a fused `.to(bf16)`): split, A is a bf16 tensor.
    body = """    %t = arith.truncf %x : f32 to bf16
    %e = arith.extf %t : bf16 to f32
    %ye = arith.extf %y : bf16 to f32
    %m = arith.mulf %e, %ye : f32
    %r = arith.addf %acc, %m : f32"""
    split, raised = _split_and_raise(_contraction(body, b="bf16"))
    assert "arith.truncf %in : f32 to bf16" in split
    assert split.count("arith.extf") == 2
    assert _contract_inputs(raised) == "tensor<4x8xbf16>, tensor<8x16xbf16>"


def test_split_contraction_operands_splits_casts_of_mixed_inputs() -> None:
    # bf16 A widened, f32 B: the vectorizer casts no float inputs of a
    # linalg.contract, so A is widened in an elementwise op of its own.
    body = """    %e = arith.extf %x : bf16 to f32
    %m = arith.mulf %e, %y : f32
    %r = arith.addf %acc, %m : f32"""
    split, raised = _split_and_raise(_contraction(body, a="bf16"))
    assert "outs(%0 : tensor<4x8xf32>)" in split
    assert _contract_inputs(raised) == "tensor<4x8xf32>, tensor<8x16xf32>"


def test_split_contraction_operands_runtime_shape() -> None:
    body = """    %e = math.exp %y : f32
    %m = arith.mulf %x, %e : f32
    %r = arith.addf %m, %acc : f32"""
    source = _contraction(body).replace("tensor<4x", "tensor<?x")
    split, raised = _split_and_raise(source)
    # Only B is split: its shape is static.
    assert "tensor.dim" not in split and "outs(%0 : tensor<8x16xf32>)" in split
    source = source.replace("%e = math.exp %y", "%e = math.exp %x")
    source = source.replace("arith.mulf %x, %e", "arith.mulf %e, %y")
    split, raised = _split_and_raise(source)
    assert "tensor.dim %arg0, %c0" in split
    assert "tensor.empty(%dim) : tensor<?x8xf32>" in split
    assert _contract_inputs(raised) == "tensor<?x8xf32>, tensor<8x16xf32>"


_MUL_ADD = """    %m = arith.mulf %x, %y : f32
    %r = arith.addf %acc, %m : f32"""
_EXP_A = "    %e = math.exp %x : f32\n"
_SPLIT_UNCHANGED = {
    # Nothing to split: what linalg.contract computes.
    "plain": _contraction(_MUL_ADD),
    "widened_bf16": _contraction(
        """    %xe = arith.extf %x : bf16 to f32
    %ye = arith.extf %y : bf16 to f32
    %m = arith.mulf %xe, %ye : f32
    %r = arith.addf %acc, %m : f32""",
        a="bf16",
        b="bf16",
    ),
    "widened_i8": _contraction(
        """    %xe = arith.extsi %x : i8 to i32
    %ye = arith.extsi %y : i8 to i32
    %m = arith.muli %xe, %ye : i32
    %r = arith.addi %acc, %m : i32""",
        a="i8",
        b="i8",
        c="i32",
    ),
    # Not understood: an op on the product, the accumulator or the result
    # (no elementwise fusion makes them), a chain op with another use.
    "unary_product": _contraction(
        _EXP_A
        + """    %m = arith.mulf %e, %y : f32
    %n = arith.negf %m : f32
    %r = arith.addf %acc, %n : f32"""
    ),
    "unary_accumulator": _contraction(
        _EXP_A
        + """    %m = arith.mulf %e, %y : f32
    %n = arith.negf %acc : f32
    %r = arith.addf %n, %m : f32"""
    ),
    "unary_result": _contraction(
        _EXP_A
        + """    %m = arith.mulf %e, %y : f32
    %s = arith.addf %acc, %m : f32
    %r = arith.negf %s : f32"""
    ),
    "shared_chain_op": _contraction(
        _EXP_A
        + """    %m = arith.mulf %e, %y : f32
    %s = arith.addf %acc, %m : f32
    %r = arith.addf %s, %e : f32"""
    ),
    # Not a contraction of mul+add.
    "max_plus": _contraction(
        _EXP_A
        + """    %m = arith.addf %e, %y : f32
    %r = arith.maximumf %acc, %m : f32"""
    ),
    "square": _contraction(
        """    %e = math.exp %x : f32
    %m = arith.mulf %e, %e : f32
    %r = arith.addf %acc, %m : f32"""
    ),
    # A binary op of a captured value: no contraction body upstream raises.
    "scaled": _contraction(
        """    %cst = arith.constant 2.0 : f32
    %e = arith.mulf %x, %cst : f32
    %m = arith.mulf %e, %y : f32
    %r = arith.addf %acc, %m : f32"""
    ),
    # Maps that are not projected permutations.
    "strided_map": _contraction(_EXP_A + _MUL_ADD.replace("%x, %y", "%e, %y")).replace(
        "#a = affine_map<(m, n, k) -> (m, k)>",
        "#a = affine_map<(m, n, k) -> (m, k floordiv 2)>",
    ),
    # Buffers.
    "memref": """
#a = affine_map<(m, n, k) -> (m, k)>
#b = affine_map<(m, n, k) -> (k, n)>
#c = affine_map<(m, n, k) -> (m, n)>
func.func @f(%a: memref<4x8xf32>, %b: memref<8x16xf32>, %c: memref<4x16xf32>) {
  linalg.generic {indexing_maps = [#a, #b, #c],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<4x8xf32>, memref<8x16xf32>) outs(%c : memref<4x16xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %e = math.exp %x : f32
    %m = arith.mulf %e, %y : f32
    %r = arith.addf %acc, %m : f32
    linalg.yield %r : f32
  }
  return
}
""",
}


@pytest.mark.parametrize("case", sorted(_SPLIT_UNCHANGED))
def test_split_contraction_operands_leaves_others_alone(case: str) -> None:
    before, after = _apply_bare(split_contraction_operands_op, _SPLIT_UNCHANGED[case])
    assert after == before


_SCALAR_CONTRACTIONS = """
func.func @f(%a: tensor<16xf32>, %b: tensor<16xf32>, %c: tensor<f32>,
             %m: tensor<4x16xf32>, %n: tensor<16x4xf32>, %o: tensor<4x4xf32>)
    -> (tensor<f32>, tensor<f32>, tensor<4x4xf32>) {
  %0 = linalg.contract
      indexing_maps = [affine_map<(k) -> (k)>, affine_map<(k) -> (k)>,
                       affine_map<(k) -> ()>]
      ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%c : tensor<f32>)
      -> tensor<f32>
  %1 = linalg.dot ins(%b, %a : tensor<16xf32>, tensor<16xf32>)
      outs(%c : tensor<f32>) -> tensor<f32>
  %2 = linalg.contract
      indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                       affine_map<(i, j, k) -> (k, j)>,
                       affine_map<(i, j, k) -> (i, j)>]
      ins(%m, %n : tensor<4x16xf32>, tensor<16x4xf32>) outs(%o : tensor<4x4xf32>)
      -> tensor<4x4xf32>
  return %0, %1, %2 : tensor<f32>, tensor<f32>, tensor<4x4xf32>
}
"""


def test_match_scalar_contractions() -> None:
    def generalize(funcs: ir.Value) -> None:
        structured.structured_generalize(
            transform.AnyOpType.get(), match_scalar_contractions(funcs)
        )

    _, text = _apply_bare(generalize, _SCALAR_CONTRACTIONS)
    # The dot products only; the 4x4 matmul stays named.
    assert text.count("linalg.generic") == 2
    assert "linalg.dot" not in text
    assert text.count("linalg.contract") == 1


def test_vectorize_linalg_dot_products() -> None:
    # Named, the dot products vectorize to vector.contract ops of vector<f32>
    # accumulators, which do not verify.
    text = _apply(vectorize_linalg, _SCALAR_CONTRACTIONS)
    assert "linalg." not in text
    assert text.count("into f32") == 2
    assert text.count("into vector<4x4xf32>") == 1
