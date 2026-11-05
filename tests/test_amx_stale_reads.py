"""An AMX operand loaded at its contraction must not see a write after its read:
one-shot bufferization may give the query and key tiles one buffer, and a query
tile read above the key loop was loaded after the keys landed there, every row
silently wrong. ``widen_misread_amx_operands`` keeps such contractions off AMX."""

from __future__ import annotations

import helion
import helion.language as hl
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
import pytest
import torch

from tests.harness import amx_features

from helion_mlir_backend._compiler.mlir_transforms import HelionTransformDialect
from helion_mlir_backend._compiler.mlir_transforms import widen_misread_amx_operands


def _apply(source: str) -> str:
    with (
        TargetInfo.override(features=amx_features(True)),
        ir.Context(),
        ir.Location.unknown(),
    ):
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            widen_misread_amx_operands(
                lh_transform.match_op(named_seq.bodyTarget, "func.func")
            )
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


_MAPS = """
#a = affine_map<(m, n, k, v) -> (m, k, v)>
#b = affine_map<(m, n, k, v) -> (k, n, v)>
#c = affine_map<(m, n, k, v) -> (m, n)>
"""


def _function(between: str) -> str:
    return f"""{_MAPS}
func.func @f(%q: memref<32x16x2xbf16>, %r: memref<16x16x2xbf16>,
             %other: memref<32x16x2xbf16>, %acc: vector<16x16xf32>,
             %row: vector<16x16x2xbf16>, %n: index) -> vector<16x16xf32> {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %z = arith.constant 0.0 : bf16
  %lhs = vector.transfer_read %q[%c0, %c0, %c0], %z {{in_bounds = [true, true, true]}}
      : memref<32x16x2xbf16>, vector<16x16x2xbf16>
  {between}
  %rhs = vector.transfer_read %r[%c0, %c0, %c0], %z {{in_bounds = [true, true, true]}}
      : memref<16x16x2xbf16>, vector<16x16x2xbf16>
  %out = vector.contract {{indexing_maps = [#a, #b, #c],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"],
      kind = #vector.kind<add>}} %lhs, %rhs, %acc
      : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
  return %out : vector<16x16xf32>
}}
"""


_WRITE = """vector.transfer_write %row, {base}[%c0, %c0, %c0] {{in_bounds = [true, true, true]}}
      : vector<16x16x2xbf16>, memref<32x16x2xbf16>"""


@pytest.mark.parametrize(
    "between",
    [
        _WRITE.format(base="%q"),
        # Through a view, and in a nested op.
        """%view = memref.subview %q[0, 0, 0] [32, 16, 2] [1, 1, 1]
      : memref<32x16x2xbf16> to memref<32x16x2xbf16>
  scf.for %i = %c0 to %n step %c1 {
    vector.transfer_write %row, %view[%c0, %c0, %c0] {in_bounds = [true, true, true]}
      : vector<16x16x2xbf16>, memref<32x16x2xbf16>
  }""",
        "memref.copy %other, %q : memref<32x16x2xbf16> to memref<32x16x2xbf16>",
    ],
    ids=["write", "view_in_loop", "copy"],
)
def test_operand_written_after_its_read_stays_off_amx(between: str) -> None:
    assert "arith.extf" in _apply(_function(between))


@pytest.mark.parametrize(
    "between",
    [
        "",
        # Other memory, and only reads.
        _WRITE.format(base="%other"),
        """%again = vector.transfer_read %q[%c0, %c0, %c0], %z {in_bounds = [true, true, true]}
      : memref<32x16x2xbf16>, vector<16x16x2xbf16>""",
    ],
    ids=["nothing", "other_memory", "read"],
)
def test_operand_not_written_stays_on_amx(between: str) -> None:
    assert "arith.extf" not in _apply(_function(between))


def fn(
    setup, base, between, acc_decl="%acc: vector<16x16xf32>", acc="%acc", pre_acc=""
):
    return f"""{_MAPS}
func.func @f(%q: memref<32x16x2xbf16>, %r: memref<16x16x2xbf16>, %other: memref<32x16x2xbf16>, {acc_decl},
             %row: vector<16x16x2xbf16>, %n: index, %c: i1, %am: memref<16x16xf32>, %af: vector<16x16xf32>) -> vector<16x16xf32> {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %z = arith.constant 0.0 : bf16
  %zf = arith.constant 0.0 : f32
  {setup}
  {pre_acc}
  %lhs = vector.transfer_read {base}[%c0, %c0, %c0], %z {{in_bounds = [true, true, true]}}
      : memref<32x16x2xbf16>, vector<16x16x2xbf16>
  {between}
  %rhs = vector.transfer_read %r[%c0, %c0, %c0], %z {{in_bounds = [true, true, true]}}
      : memref<16x16x2xbf16>, vector<16x16x2xbf16>
  %out = vector.contract {{indexing_maps = [#a, #b, #c],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"],
      kind = #vector.kind<add>}} %lhs, %rhs, {acc}
      : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
  return %out : vector<16x16xf32>
}}
"""


W = _WRITE.format(base="%q")
IFALIAS = """%al = scf.if %c -> memref<32x16x2xbf16> { scf.yield %q : memref<32x16x2xbf16> } else { scf.yield %q : memref<32x16x2xbf16> }"""
SEL = """%al = arith.select %c, %q, %q : memref<32x16x2xbf16>"""
TR = """%t = memref.transpose %q (i, j, k) -> (j, i, k) : memref<32x16x2xbf16> to memref<16x32x2xbf16, affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>>"""
MISS = {
    "read_through_scf_if": fn(IFALIAS, "%al", W),
    "read_through_select": fn(SEL, "%al", W),
    "write_through_scf_if": fn(IFALIAS, "%q", _WRITE.format(base="%al")),
    "acc_read_stale": fn(
        "",
        "%q",
        "vector.transfer_write %af, %am[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32>",
        acc_decl="%acc0: vector<16x16xf32>",
        acc="%acc",
        pre_acc="%acc = vector.transfer_read %am[%c0, %c0], %zf {in_bounds = [true, true]} : memref<16x16xf32>, vector<16x16xf32>",
    ),
}


@pytest.mark.parametrize("name", MISS)
def test_unknown_or_stale_memory_stays_off_amx(name: str) -> None:
    assert "arith.extf" in _apply(MISS[name])


READ_ONLY = {
    "vector_load": "%vl = vector.load %q[%c0, %c0, %c0] : memref<32x16x2xbf16>, vector<2xbf16>",
    "copy_from": "memref.copy %q, %other : memref<32x16x2xbf16> to memref<32x16x2xbf16>",
    "maskedload": "%m = vector.constant_mask [2] : vector<4xi1>\n  %pt = arith.constant dense<0.0> : vector<4xbf16>\n  %ml = vector.maskedload %q[%c0, %c0, %c0], %m, %pt : memref<32x16x2xbf16>, vector<4xi1>, vector<4xbf16> into vector<4xbf16>",
    "extract_meta": "%mm = memref.extract_aligned_pointer_as_index %q : memref<32x16x2xbf16> -> index",
}


@pytest.mark.parametrize("name", READ_ONLY)
def test_reads_between_stay_on_amx(name: str) -> None:
    assert "arith.extf" not in _apply(fn("", "%q", READ_ONLY[name]))


def _padded_dots(
    q: torch.Tensor, k: torch.Tensor, rows: hl.constexpr, padded_dim: hl.constexpr
) -> torch.Tensor:
    """``q @ k.T`` by heads, query and key tiles of one shape, columns padded."""
    heads, seq, dim = q.shape
    hl.specialize(seq)
    hl.specialize(dim)
    out = torch.empty((heads, seq, seq), dtype=torch.float32)
    for h in hl.grid(heads):
        for tile_m in hl.tile(int(rows)):
            for tile_d in hl.tile(int(padded_dim)):
                q_tile = q[h, tile_m, tile_d]
                for tile_n in hl.tile(seq):
                    k_tile = k[h, tile_n, tile_d]
                    a3 = q_tile.reshape(q_tile.size(0), q_tile.size(1) // 2, 2)
                    b3 = k_tile.reshape(k_tile.size(0), k_tile.size(1) // 2, 2)
                    scores = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    scores = scores + torch.einsum(
                        "mcv,cnv->mn", a3, b3.permute(1, 0, 2)
                    )
                    out[h, tile_m, tile_n] = scores
    return out


@pytest.mark.slow
@pytest.mark.isolated
@pytest.mark.parametrize(("seq", "dim"), [(64, 7), (48, 7), (33, 7), (64, 16)], ids=str)
def test_query_and_key_tiles_of_one_buffer(seq: int, dim: int) -> None:
    torch.manual_seed(0)
    q, k = (torch.randn(2, seq, dim).bfloat16() for _ in range(2))
    kernel = helion.kernel(
        _padded_dots, backend="mlir", config=helion.Config(block_sizes=[32, 32, 32])
    )
    out = kernel(q, k, hl.constexpr(64), hl.constexpr(32))
    ref = q.double() @ k.double().transpose(1, 2)
    assert (out.double() - ref).abs().max() / ref.abs().max() < 1e-3
