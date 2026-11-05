"""Operand layouts upstream's x86 AMX rewrite (``vector_contract_to_amx_dot_product``)
assumes without checking, found on an AMX host (EMR, bf16):

- it reads a flat contraction's LHS as ``M x K``, RHS as ``K x N`` and
  accumulator as ``M x N`` rows (their two innermost dims), whatever the
  indexing maps say: ``x @ w.T`` (KernelBench level 2), ``q @ k.T``
  (attention), a transposed accumulator, linear layers on 3-D inputs and
  batch-reduce contractions of such operands were miscompiled.
  ``normalize_amx_contractions`` gives such operands transposed copies, or
  keeps the contraction off AMX.
- in a K loop with accumulators read from memory, it takes the accumulator
  read's memref as the register tile, from its origin: a ``64x32`` subview of a
  ``32x32`` register tile's buffer made it index past its results (an
  assertion abort in attention's P @ V). ``fit_subviews_to_reads`` fits the
  subviews ``move_offsets_to_subview`` makes to the tiles read.

The transform and IR-level tests force AMX on through lighthouse's
``TargetInfo.override`` and run on any x86 host. The numeric tests execute AMX
code and run on AMX hosts only (the KernelBench and attention numerics run in
``test_contraction_regressions.py``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import helion
import helion.language as hl
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import x86
import pytest
import torch

from tests.harness import amx_features
from tests.harness import attention_kernel
from tests.harness import lower_opt_with_amx

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend import generate_mlir
from helion_mlir_backend import inline_mlir
from helion_mlir_backend._compiler.mlir_transforms import AMX_OFF_ATTR_NAME
from helion_mlir_backend._compiler.mlir_transforms import HelionTransformDialect
from helion_mlir_backend._compiler.mlir_transforms import fit_subviews_to_reads
from helion_mlir_backend._compiler.mlir_transforms import normalize_amx_contractions
from helion_mlir_backend._compiler.mlir_transforms import pin_amx_off_tile_sizes

if TYPE_CHECKING:
    from collections.abc import Callable

_HAS_AMX = TargetInfo.host().is_supported("amx")
amx_host_only = pytest.mark.skipif(not _HAS_AMX, reason="executes AMX code")


def _apply(
    make_op: Callable[[ir.Value], object], source: str, *, amx: bool = True
) -> str:
    """``source`` after the transform op alone, on a target with or without AMX."""
    with (
        TargetInfo.override(features=amx_features(amx)),
        ir.Context(),
        ir.Location.unknown(),
    ):
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            make_op(lh_transform.match_op(named_seq.bodyTarget, "func.func"))
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def _parsed(source: str) -> str:
    with ir.Context():
        return str(ir.Module.parse(source))


# --- normalize_amx_contractions ---------------------------------------------------

_CONTRACT = """
func.func @f(%a: tensor<{a}x{t}>, %b: tensor<{b}x{t}>, %c: tensor<{c}x{acc}>)
    -> tensor<{c}x{acc}> {{
  %r = linalg.contract indexing_maps = [
      affine_map<({dims}) -> ({lhs})>, affine_map<({dims}) -> ({rhs})>,
      affine_map<({dims}) -> ({out})>]
      ins(%a, %b : tensor<{a}x{t}>, tensor<{b}x{t}>) outs(%c : tensor<{c}x{acc}>)
      -> tensor<{c}x{acc}>
  return %r : tensor<{c}x{acc}>
}}
"""


def _contract(
    lhs: str, rhs: str, out: str, shapes: str, t: str = "bf16", acc: str = "f32"
) -> str:
    """A ``linalg.contract`` over (m, n, k) = (d0, d1, d2) (and b = d3 if used),
    of operand shapes ``a;b;c``."""
    a, b, c = shapes.split(";")
    batched = "d3" in lhs + rhs + out
    dims = "d0, d1, d2, d3" if batched else "d0, d1, d2"
    return _CONTRACT.format(
        a=a, b=b, c=c, t=t, acc=acc, dims=dims, lhs=lhs, rhs=rhs, out=out
    )


NT = _contract("d0, d2", "d1, d2", "d0, d1", "32x32;64x32;32x64")
TN = _contract("d2, d0", "d2, d1", "d0, d1", "32x32;32x64;32x64")
BOTH_SWAPPED = _contract("d2, d0", "d1, d2", "d0, d1", "32x32;64x32;32x64")
NT_BATCHED = _contract(
    "d3, d0, d2", "d3, d1, d2", "d3, d0, d1", "2x32x32;2x64x32;2x32x64"
)
# out[n, m] = sum_k a[m, k] b[k, n]: a transposed accumulator.
ACC_NM = _contract("d0, d2", "d2, d1", "d1, d0", "32x32;32x64;64x32")
# out[b, m, n] = sum_k x[b, m, k] w[n, k]: linear on a 3-D input (two M dims).
LINEAR_3D = _contract("d3, d0, d2", "d1, d2", "d3, d0, d1", "2x32x32;64x32;2x32x64")
# out[m, n] = sum_{b, k} x[b, m, k] w[b, n, k]: batch-reduce with B as N x K.
BRGEMM_NT = _contract("d3, d0, d2", "d3, d1, d2", "d0, d1", "2x32x32;2x64x32;32x64")
NT_DYNAMIC = _contract("d0, d2", "d1, d2", "d0, d1", "?x?;?x?;?x?")


def _maps(text: str) -> list[tuple[str, str]]:
    return re.findall(r"affine_map<\(([^)]*)\) -> \(([^)]*)\)>", text)


def test_normalize_amx_contractions_rhs_n_by_k() -> None:
    """``x @ w.T``: B (64 x 32, N x K) is read from a 32 x 64 transposed copy."""
    text = _apply(normalize_amx_contractions, NT)
    assert re.search(
        r"linalg\.transpose ins\(%arg1 : tensor<64x32xbf16>\) "
        r"outs\(%\w+ : tensor<32x64xbf16>\) permutation = \[1, 0\]",
        text,
    )
    assert "ins(%arg0, %transposed : tensor<32x32xbf16>, tensor<32x64xbf16>)" in text
    rhs = [result for _, result in _maps(text) if result in ("d2, d1", "d1, d2")]
    assert rhs == ["d2, d1"]


def test_normalize_amx_contractions_lhs_k_by_m() -> None:
    """``x.T @ y``: A (K x M) is read from a transposed copy; B already K x N."""
    text = _apply(normalize_amx_contractions, TN)
    assert text.count("linalg.transpose") == 1
    assert "linalg.transpose ins(%arg0 : tensor<32x32xbf16>)" in text
    assert ("d0, d1, d2", "d0, d2") in _maps(text)
    assert ("d0, d1, d2", "d2, d1") in _maps(text)


def test_normalize_amx_contractions_both_inputs() -> None:
    text = _apply(normalize_amx_contractions, BOTH_SWAPPED)
    assert text.count("linalg.transpose") == 2
    assert (
        "ins(%transposed, %transposed_0 : tensor<32x32xbf16>, tensor<32x64xbf16>)"
        in text
    )
    assert ("d0, d1, d2", "d0, d2") in _maps(text)
    assert ("d0, d1, d2", "d2, d1") in _maps(text)


def test_normalize_amx_contractions_batched() -> None:
    """Batch dims lead every operand: only the inner two dims of B swap."""
    text = _apply(normalize_amx_contractions, NT_BATCHED)
    assert "permutation = [0, 2, 1]" in text
    assert "tensor<2x32x64xbf16>" in text
    assert ("d0, d1, d2, d3", "d3, d2, d1") in _maps(text)


def test_normalize_amx_contractions_accumulator_n_by_m() -> None:
    """The ``N x M`` accumulator init is transposed in, the ``M x N`` result
    transposed back for the function's result."""
    text = _apply(normalize_amx_contractions, ACC_NM)
    assert text.count("linalg.transpose") == 2
    assert "linalg.transpose ins(%arg2 : tensor<64x32xf32>) outs(" in text
    assert re.search(
        r"outs\(%transposed : tensor<32x64xf32>\) -> tensor<32x64xf32>", text
    )
    assert re.search(
        r"%(\w+) = linalg\.transpose ins\(%\w+ : tensor<32x64xf32>\) "
        r"outs\(%\w+ : tensor<64x32xf32>\)",
        text,
    )
    assert re.search(r"return %transposed_\d+ : tensor<64x32xf32>", text)
    assert ("d0, d1, d2", "d0, d1") in _maps(text)
    assert ("d0, d1, d2", "d1, d0") not in _maps(text)


def test_normalize_amx_contractions_linear_3d() -> None:
    """Two M dims (b, m): only B's inner pair (n, k) decides, and swaps."""
    text = _apply(normalize_amx_contractions, LINEAR_3D)
    assert "linalg.transpose ins(%arg1 : tensor<64x32xbf16>)" in text
    assert ("d0, d1, d2, d3", "d2, d1") in _maps(text)


def test_normalize_amx_contractions_batch_reduce() -> None:
    """Two K dims (b, k), the batch-reduce one leading: B (b, n, k) swaps its
    inner pair."""
    text = _apply(normalize_amx_contractions, BRGEMM_NT)
    assert "linalg.transpose ins(%arg1 : tensor<2x64x32xbf16>)" in text
    assert "permutation = [0, 2, 1]" in text
    assert ("d0, d1, d2, d3", "d3, d2, d1") in _maps(text)


def test_normalize_amx_contractions_dynamic_shapes() -> None:
    text = _apply(normalize_amx_contractions, NT_DYNAMIC)
    assert text.count("tensor.dim %arg1") == 2
    assert "linalg.transpose ins(%arg1 : tensor<?x?xbf16>)" in text
    assert ("d0, d1, d2", "d2, d1") in _maps(text)


# The same contraction as a linalg.generic (the transform also takes generics
# linalg-categorize-ops left as such).
NT_GENERIC = """
func.func @f(%a: tensor<32x32xbf16>, %b: tensor<64x32xbf16>, %c: tensor<32x64xf32>)
    -> tensor<32x64xf32> {
  %r = linalg.generic {indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<32x32xbf16>, tensor<64x32xbf16>) outs(%c : tensor<32x64xf32>) {
  ^bb0(%x: bf16, %y: bf16, %acc: f32):
    %xf = arith.extf %x : bf16 to f32
    %yf = arith.extf %y : bf16 to f32
    %p = arith.mulf %xf, %yf : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  } -> tensor<32x64xf32>
  return %r : tensor<32x64xf32>
}
"""


def test_normalize_amx_contractions_generic() -> None:
    text = _apply(normalize_amx_contractions, NT_GENERIC)
    assert "linalg.transpose ins(%arg1 : tensor<64x32xbf16>)" in text
    assert "ins(%arg0, %transposed : tensor<32x32xbf16>, tensor<32x64xbf16>)" in text
    assert ("d0, d1, d2", "d2, d1") in _maps(text)
    assert ("d0, d1, d2", "d1, d2") not in _maps(text)


def test_normalize_amx_contractions_generic_accumulator_n_by_m() -> None:
    """A generic is rebuilt for the transposed result, its body moved over."""
    source = (
        NT_GENERIC.replace(
            "affine_map<(d0, d1, d2) -> (d1, d2)>,\n      affine_map<(d0, d1, d2) -> (d0, d1)>",
            "affine_map<(d0, d1, d2) -> (d2, d1)>,\n      affine_map<(d0, d1, d2) -> (d1, d0)>",
        )
        .replace("tensor<64x32xbf16>", "tensor<32x64xbf16>")
        .replace("tensor<32x64xf32>", "tensor<64x32xf32>")
    )
    text = _apply(normalize_amx_contractions, source)
    assert text.count("linalg.transpose") == 2
    assert "arith.mulf" in text
    assert "linalg.generic" in text
    assert ("d0, d1, d2", "d1, d0") not in _maps(text)


# VNNI layouts, as upstream's isInVnniLayout takes them: the rewrite reads
# their maps itself.
_VNNI_CONTRACT = """
func.func @f(%a: tensor<{a}xbf16>, %b: tensor<{b}xbf16>, %c: tensor<{c}xf32>)
    -> tensor<{c}xf32> {{
  %r = linalg.contract indexing_maps = [
      affine_map<({dims}) -> ({lhs})>, affine_map<({dims}) -> ({rhs})>,
      affine_map<({dims}) -> ({out})>]
      ins(%a, %b : tensor<{a}xbf16>, tensor<{b}xbf16>)
      outs(%c : tensor<{c}xf32>) -> tensor<{c}xf32>
  return %r : tensor<{c}xf32>
}}
"""
VNNI = _VNNI_CONTRACT.format(
    a="32x16x2",
    b="16x64x2",
    c="32x64",
    dims="m, n, k, v",
    lhs="m, k, v",
    rhs="k, n, v",
    out="m, n",
)
# KernelBench level 2's 37 (``mcv,bcnv->mbn``): the accumulator ends in (N, N).
VNNI_MBN = _VNNI_CONTRACT.format(
    a="32x16x2",
    b="2x16x64x2",
    c="32x2x64",
    dims="m, b, n, c, v",
    lhs="m, c, v",
    rhs="b, c, n, v",
    out="m, b, n",
)
# A transposed accumulator of a VNNI contraction.
VNNI_ACC_NM = _VNNI_CONTRACT.format(
    a="32x16x2",
    b="16x64x2",
    c="64x32",
    dims="m, n, k, v",
    lhs="m, k, v",
    rhs="k, n, v",
    out="n, m",
)
# Split K (``mkj,kjn->mn``): two K dims innermost in A, but B's innermost is N,
# so it is no VNNI layout (upstream takes it as flat and ignores its maps).
SPLIT_K = _VNNI_CONTRACT.format(
    a="32x2x16",
    b="2x16x64",
    c="32x64",
    dims="m, n, k, j",
    lhs="m, k, j",
    rhs="k, j, n",
    out="m, n",
)

# Memref operands.
_MEMREF_CONTRACT = """
func.func @f(%a: memref<{a}xbf16>, %b: memref<{b}xbf16>, %c: memref<32x64xf32>) {{
  linalg.contract indexing_maps = [
      affine_map<(d0, d1, d2) -> ({lhs})>, affine_map<(d0, d1, d2) -> ({rhs})>,
      affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%a, %b : memref<{a}xbf16>, memref<{b}xbf16>) outs(%c : memref<32x64xf32>)
  return
}}
"""
NT_MEMREF = _MEMREF_CONTRACT.format(a="32x32", b="64x32", lhs="d0, d2", rhs="d1, d2")
NN_MEMREF = _MEMREF_CONTRACT.format(a="32x32", b="32x64", lhs="d0, d2", rhs="d2, d1")


@pytest.mark.parametrize(
    ("source", "amx"),
    [
        # No AMX: lighthouse picks no AMX tiles, the vector lowering reads the maps.
        (NT, False),
        # Already the expected layouts.
        (_contract("d0, d2", "d2, d1", "d0, d1", "32x32;32x64;32x64"), True),
        (NN_MEMREF, True),
        # f32, f16 and i8: lighthouse tiles only bf16 x bf16 -> f32 for AMX.
        (_contract("d0, d2", "d1, d2", "d0, d1", "32x32;64x32;32x64", t="f32"), True),
        (_contract("d0, d2", "d1, d2", "d0, d1", "32x32;64x32;32x64", t="f16"), True),
        (
            _contract(
                "d0, d2", "d1, d2", "d0, d1", "32x32;64x32;32x64", t="i8", acc="i32"
            ),
            True,
        ),
        (VNNI, True),
        (VNNI_MBN, True),
    ],
    ids=["no_amx", "nn", "nn_memref", "f32", "f16", "i8", "vnni", "vnni_mbn"],
)
def test_normalize_amx_contractions_leaves_others(source: str, amx: bool) -> None:
    assert _apply(normalize_amx_contractions, source, amx=amx) == _parsed(source)


def test_normalize_amx_contractions_vnni_accumulator_n_by_m() -> None:
    """VNNI inputs stay; the (N, M) accumulator is swapped as in flat layouts."""
    text = _apply(normalize_amx_contractions, VNNI_ACC_NM)
    assert text.count("linalg.transpose") == 2
    assert "linalg.transpose ins(%arg2 : tensor<64x32xf32>)" in text
    assert "ins(%arg0, %arg1 : tensor<32x16x2xbf16>, tensor<16x64x2xbf16>)" in text
    assert ("m, n, k, v", "n, m") not in _maps(text)


@pytest.mark.parametrize(
    "source",
    [
        # Decode-style batched dot (b, k), (b, n, k) -> (b, n): no M dim.
        _contract("d0, d2", "d0, d1, d2", "d0, d1", "2x32;2x64x32;2x64"),
        # Batched matrix-vector (b, m, k), (b, k) -> (b, m): no N dim.
        _contract("d0, d1, d2", "d0, d2", "d0, d1", "2x32x64;2x64;2x32"),
        # A batch dim innermost in A: (m, b, k).
        _contract("d0, d3, d2", "d3, d1, d2", "d3, d0, d1", "32x2x32;2x64x32;2x32x64"),
        # A batch dim innermost in the accumulator: (m, n, b).
        _contract("d3, d0, d2", "d3, d2, d1", "d0, d1, d3", "2x32x32;2x32x64;32x64x2"),
        SPLIT_K,
    ],
    ids=["decode", "bmv", "batch_inner_lhs", "batch_inner_acc", "split_k"],
)
def test_normalize_amx_contractions_keeps_others_off_amx(source: str) -> None:
    """Layouts the transform cannot normalize are marked to be kept off AMX
    (``pin_amx_off_tile_sizes``), otherwise unchanged."""
    text = _apply(normalize_amx_contractions, source)
    assert text.count(AMX_OFF_ATTR_NAME) == 1
    assert text.replace(f" {{{AMX_OFF_ATTR_NAME}}}", "") == _parsed(source)


def test_normalize_amx_contractions_keeps_memref_off_amx() -> None:
    """A memref ``x @ w.T`` (no tensors to transpose; lighthouse does not tile
    memref ops): both inputs widened into new buffers, an f32 contraction."""
    text = _apply(normalize_amx_contractions, NT_MEMREF)
    assert text.count("memref.alloc()") == 2
    assert "ins(%alloc, %alloc_0 : memref<32x32xf32>, memref<64x32xf32>)" in text


def test_normalize_amx_contractions_widens_generic_memref() -> None:
    """A memref generic's body reads the widened inputs through a truncation,
    so its bf16 ops stay valid."""
    source = """
func.func @f(%a: memref<32x32xbf16>, %b: memref<64x32xbf16>, %c: memref<32x64xf32>) {
  linalg.generic {indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<32x32xbf16>, memref<64x32xbf16>) outs(%c : memref<32x64xf32>) {
  ^bb0(%x: bf16, %y: bf16, %acc: f32):
    %xf = arith.extf %x : bf16 to f32
    %yf = arith.extf %y : bf16 to f32
    %p = arith.mulf %xf, %yf : f32
    %s = arith.addf %acc, %p : f32
    linalg.yield %s : f32
  }
  return
}
"""
    text = _apply(normalize_amx_contractions, source)
    assert re.search(r"\^bb0\(%\w+: f32, %\w+: f32, %\w+: f32\)", text)
    assert text.count("arith.truncf") == 2


# --- pin_amx_off_tile_sizes ---------------------------------------------------


def _tile_sizes(text: str) -> list[str]:
    return re.findall(r"transform_ext\.tile_sizes = array<i64: ([^>]*)>", text)


@pytest.mark.parametrize(
    ("strategy", "amx_sizes"),
    [
        ("register_parallel", "32, 32, 0"),
        ("register_unroll", "16, 16, 32"),
    ],
)
def test_pin_amx_off_tile_sizes(strategy: str, amx_sizes: str) -> None:
    """A marked contraction gets the strategy's tile sizes for the host without
    AMX, not lighthouse's AMX ones (``amx_sizes``); an unmarked one none."""
    from lighthouse.dialects.transform.transform_ext.utils import (
        tile_size_analysis as tsa,
    )

    from helion_mlir_backend._compiler.mlir_transforms.pin_amx_off_tile_sizes import (
        without_amx,
    )

    source = NT.replace(
        "]\n      ins(%a, %b", "] {helion.amx_off}\n      ins(%a, %b", 1
    )
    marked = _apply(lambda t: pin_amx_off_tile_sizes(t, strategy=strategy), source)
    unmarked = _apply(lambda t: pin_amx_off_tile_sizes(t, strategy=strategy), NT)
    assert unmarked == _parsed(NT)
    (sizes,) = _tile_sizes(marked)
    with (
        TargetInfo.override(features=amx_features()),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.parse(NT)
        contract = next(
            op
            for op in module.body.operations[0].regions[0].blocks[0].operations
            if op.name == "linalg.contract"
        )
        amx = tsa.compute_tile_sizes(
            contract, strategy=strategy, target=TargetInfo.host()
        )
        no_amx = tsa.compute_tile_sizes(
            contract, strategy=strategy, target=without_amx(TargetInfo.host())
        )
    assert ", ".join(map(str, amx)) == amx_sizes
    assert sizes == ", ".join(map(str, no_amx)) != amx_sizes


def test_pin_amx_off_tile_sizes_keeps_annotations() -> None:
    source = NT.replace(
        "]\n      ins(%a, %b",
        "] {helion.amx_off, transform_ext.tile_sizes = array<i64: 4, 4, 4>}"
        "\n      ins(%a, %b",
        1,
    )
    assert _apply(pin_amx_off_tile_sizes, source) == _parsed(source)


# --- fit_subviews_to_reads ----------------------------------------------------

_MK = "affine_map<(m, n, k) -> (m, k)>"
_KN = "affine_map<(m, n, k) -> (k, n)>"
_MN = "affine_map<(m, n, k) -> (m, n)>"
_ACC = "memref<64x32xf32, strided<[32, 1], offset: ?>>"
_V4 = ", ".join(["vector<16x16xf32>"] * 4)


def _contract_op(
    result: str,
    lhs: str,
    rhs: str,
    acc: str,
    m: int,
    n: int,
    t: str = "f32",
    k: int = 2,
) -> str:
    return (
        f"{result} = vector.contract {{indexing_maps = [{_MK}, {_KN}, {_MN}], "
        'iterator_types = ["parallel", "parallel", "reduction"], '
        f"kind = #vector.kind<add>}} {lhs}, {rhs}, {acc} : vector<{m}x{k}x{t}>, "
        f"vector<{k}x{n}x{t}> into vector<{m}x{n}xf32>"
    )


def _acc_reads(row: int, names: str = "i", subview: str = "%s") -> str:
    """Four 16x16 accumulator reads of a 32x32 tile at row ``row``."""
    return "\n".join(
        f"  %{names}{t} = vector.transfer_read {subview}[%c{r}, %c{c}], %pad "
        f"{{in_bounds = [true, true]}} : {_ACC}, vector<16x16xf32>"
        for t, (r, c) in enumerate([(row, 0), (row, 16), (row + 16, 0), (row + 16, 16)])
    )


def _acc_contracts(acc: str = "x", result: str = "y", indent: str = "    ") -> str:
    return "\n".join(
        indent + _contract_op(f"%{result}{t}", "%a", "%b", f"%{acc}{t}", 16, 16)
        for t in range(4)
    )


_CONSTANTS = "\n".join(
    f"  %c{v} = arith.constant {v} : index" for v in (0, 1, 16, 32, 48, 56)
)


def _k_loop(row: int) -> str:
    """A K loop accumulating the 32x32 register tile at ``row`` of a 64x32
    buffer, its four 16x16 accumulators read before it through a subview that
    keeps the buffer's sizes (as move_offsets_to_subview leaves it): the form
    the AMX rewrite aborted on."""
    return f"""
func.func @f(%buf: memref<64x32xf32>, %i: index, %n: index,
             %a: vector<16x2xf32>, %b: vector<2x16xf32>) -> ({_V4}) {{
{_CONSTANTS}
  %pad = arith.constant 0.0 : f32
  %s = memref.subview %buf[%i, 0] [64, 32] [1, 1] : memref<64x32xf32> to {_ACC}
{_acc_reads(row)}
  %r:4 = scf.for %k = %c0 to %n step %c1
      iter_args(%x0 = %i0, %x1 = %i1, %x2 = %i2, %x3 = %i3) -> ({_V4}) {{
{_acc_contracts()}
    scf.yield %y0, %y1, %y2, %y3 : {_V4}
  }}
  return %r#0, %r#1, %r#2, %r#3 : {_V4}
}}
"""


_FITTED = "memref<32x32xf32, strided<[32, 1], offset: ?>>"


def test_fit_subviews_to_reads_accumulator_tile() -> None:
    text = _apply(fit_subviews_to_reads, _k_loop(0))
    assert (
        f"memref.subview %arg0[%arg1, 0] [32, 32] [1, 1] : memref<64x32xf32> to "
        f"{_FITTED}" in text
    )
    assert text.count("memref.subview") == 1
    assert text.count(f"{_FITTED}, vector<16x16xf32>") == 4


@pytest.mark.parametrize("row", [16, 32])
def test_fit_subviews_to_reads_moves_tile_to_origin(row: int) -> None:
    """A tile not at the subview's origin: its row is added to the offset and
    taken off the reads' indices."""
    text = _apply(fit_subviews_to_reads, _k_loop(row))
    assert re.search(rf"%(\w+) = arith.addi %arg1, %c{row}", text)
    assert re.search(
        rf"memref\.subview %arg0\[%\w+, 0\] \[32, 32\] \[1, 1\] : "
        rf"memref<64x32xf32> to {re.escape(_FITTED)}",
        text,
    )
    reads = re.findall(
        r"vector\.transfer_read %\w+\[(%c\d+)(?:_\d+)?, (%c\d+)(?:_\d+)?\]", text
    )
    assert sorted(reads) == sorted(
        [("%c0", "%c0"), ("%c0", "%c16"), ("%c16", "%c0"), ("%c16", "%c16")]
    )


# Two register tiles (rows 0 and 32) read through one subview, each for a loop
# of its own: one fitted subview per tile.
TWO_TILES = f"""
func.func @f(%buf: memref<64x32xf32>, %i: index, %n: index,
             %a: vector<16x2xf32>, %b: vector<2x16xf32>) -> ({_V4}, {_V4}) {{
{_CONSTANTS}
  %pad = arith.constant 0.0 : f32
  %s = memref.subview %buf[%i, 0] [64, 32] [1, 1] : memref<64x32xf32> to {_ACC}
{_acc_reads(0)}
{_acc_reads(32, "j")}
  %r:4 = scf.for %k = %c0 to %n step %c1
      iter_args(%x0 = %i0, %x1 = %i1, %x2 = %i2, %x3 = %i3) -> ({_V4}) {{
{_acc_contracts()}
    scf.yield %y0, %y1, %y2, %y3 : {_V4}
  }}
  %q:4 = scf.for %k = %c0 to %n step %c1
      iter_args(%x0 = %j0, %x1 = %j1, %x2 = %j2, %x3 = %j3) -> ({_V4}) {{
{_acc_contracts()}
    scf.yield %y0, %y1, %y2, %y3 : {_V4}
  }}
  return %r#0, %r#1, %r#2, %r#3, %q#0, %q#1, %q#2, %q#3 : {_V4}, {_V4}
}}
"""


def test_fit_subviews_to_reads_one_subview_per_tile() -> None:
    text = _apply(fit_subviews_to_reads, TWO_TILES)
    subviews = re.findall(r"memref\.subview %arg0\[(%\w+), 0\] \[32, 32\]", text)
    assert len(subviews) == 2 and "%arg1" in subviews
    assert "arith.addi %arg1, %c32" in text
    assert text.count(f"{_FITTED}, vector<16x16xf32>") == 8


# Accumulators carried through two loops (a batch-reduce and a K loop).
TWO_LEVEL_K_LOOP = f"""
func.func @f(%buf: memref<64x32xf32>, %i: index, %n: index,
             %a: vector<16x2xf32>, %b: vector<2x16xf32>) -> ({_V4}) {{
{_CONSTANTS}
  %pad = arith.constant 0.0 : f32
  %s = memref.subview %buf[%i, 0] [64, 32] [1, 1] : memref<64x32xf32> to {_ACC}
{_acc_reads(0)}
  %r:4 = scf.for %g = %c0 to %n step %c1
      iter_args(%w0 = %i0, %w1 = %i1, %w2 = %i2, %w3 = %i3) -> ({_V4}) {{
    %q:4 = scf.for %k = %c0 to %n step %c1
        iter_args(%x0 = %w0, %x1 = %w1, %x2 = %w2, %x3 = %w3) -> ({_V4}) {{
{_acc_contracts(indent="      ")}
      scf.yield %y0, %y1, %y2, %y3 : {_V4}
    }}
    scf.yield %q#0, %q#1, %q#2, %q#3 : {_V4}
  }}
  return %r#0, %r#1, %r#2, %r#3 : {_V4}
}}
"""


def test_fit_subviews_to_reads_two_level_k_loop() -> None:
    text = _apply(fit_subviews_to_reads, TWO_LEVEL_K_LOOP)
    assert "memref.subview %arg0[%arg1, 0] [32, 32] [1, 1]" in text


# An accumulator read of a 2-D vector from a 3-D subview of a dynamic base, used
# directly: leading dims get the index plus one, the layout stays as it was.
_T3 = "memref<?x64x32xf32, strided<[2048, 32, 1], offset: ?>>"
# A 2x2 accumulator tile read as 2-D vectors from a 3-D subview of a dynamic
# base, used directly: leading dims get the index plus one, the layout stays.
RANK_REDUCED_READS = (
    f"""
func.func @f(%v: memref<?x64x32xf32>, %h: index, %a: vector<16x2xf32>,
             %b: vector<2x16xf32>) -> ({_V4}) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %pad = arith.constant 0.0 : f32
  %d = memref.dim %v, %c0 : memref<?x64x32xf32>
  %s = memref.subview %v[%h, 0, 0] [%d, 64, 32] [1, 1, 1]
      : memref<?x64x32xf32> to {_T3}
"""
    + "\n".join(
        f"  %x{t} = vector.transfer_read %s[%c1, %c{r}, %c{c}], %pad "
        f"{{in_bounds = [true, true]}} : {_T3}, vector<16x16xf32>"
        for t, (r, c) in enumerate([(0, 0), (0, 16), (16, 0), (16, 16)])
    )
    + f"""
{_acc_contracts(indent="  ")}
  return %y0, %y1, %y2, %y3 : {_V4}
}}
"""
)


def test_fit_subviews_to_reads_rank_reduced_reads() -> None:
    text = _apply(fit_subviews_to_reads, RANK_REDUCED_READS)
    assert re.search(
        r"memref\.subview %arg0\[%\w+, 0, 0\] \[1, 32, 32\] \[1, 1, 1\] : "
        r"memref<\?x64x32xf32> to memref<1x32x32xf32, strided<\[2048, 32, 1\], "
        r"offset: \?>>",
        text,
    )
    assert "arith.addi %arg1, %c1" in text
    assert re.search(r"transfer_read %\w+\[%c0(_\d+)?, %c0, %c0\]", text)


def _one_read(
    read: str,
    *,
    use: str = "acc",
    sizes: str = "64, 32",
    strides: str = "1, 1",
    shape: str = "64x32",
    layout: str = "32, 1",
    extra: str = "",
) -> str:
    """A subview of a 64x32 buffer, three 16x16 accumulator reads of a 32x32
    tile at its origin and the fourth, ``read`` (``%r``; ``{t}`` stands for
    the subview's type), used as a contraction's accumulator too, or as its
    RHS (``use="rhs"``)."""
    memref_type = f"memref<{shape}xf32, strided<[{layout}], offset: ?>>"
    fixed = "\n".join(
        f"  %x{t} = vector.transfer_read %s[%c{r}, %c{c}], %pad "
        f"{{in_bounds = [true, true]}} : {memref_type}, vector<16x16xf32>"
        for t, (r, c) in enumerate([(0, 0), (0, 16), (16, 0)])
    )
    contracts = "\n".join(
        "  " + _contract_op(f"%y{t}", "%a", "%b", f"%x{t}", 16, 16) for t in range(3)
    )
    if use == "acc":
        last = _contract_op("%y3", "%a", "%b", "%r", 16, 16)
    else:
        last = _contract_op("%y3", "%a1", "%r", "%acc", 16, 16, k=16)
    return f"""
func.func @f(%buf: memref<64x32xf32>, %i: index, %j: index, %m: vector<16x16xi1>,
             %a: vector<16x2xf32>, %b: vector<2x16xf32>, %a1: vector<16x16xf32>,
             %acc: vector<16x16xf32>) -> ({_V4}) {{
{_CONSTANTS}
  %pad = arith.constant 0.0 : f32
  %s = memref.subview %buf[%i, 0] [{sizes}] [{strides}]
      : memref<64x32xf32> to {memref_type}
  {extra}
{fixed}
  {read.format(t=memref_type)}
{contracts}
  {last}
  return %y0, %y1, %y2, %y3 : {_V4}
}}
"""


_IN_BOUNDS = "{{in_bounds = [true, true]}}"  # format()ted with the type
_READ_AT_16 = (
    f"%r = vector.transfer_read %s[%c16, %c16], %pad {_IN_BOUNDS} : "
    "{t}, vector<16x16xf32>"
)

# A rank-reducing subview: a row of the buffer.
RANK_REDUCING_SUBVIEW = """
func.func @f(%buf: memref<64x32xf32>, %i: index, %a: vector<1x2xf32>,
             %b: vector<2x16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %s = memref.subview %buf[%i, 0] [1, 32] [1, 1]
      : memref<64x32xf32> to memref<32xf32, strided<[1], offset: ?>>
  %r = vector.transfer_read %s[%c0], %pad {in_bounds = [true]}
      : memref<32xf32, strided<[1], offset: ?>>, vector<16xf32>
  %y = vector.contract {indexing_maps = [affine_map<(n, k) -> (k)>,
      affine_map<(n, k) -> (k, n)>, affine_map<(n, k) -> (n)>],
      iterator_types = ["parallel", "reduction"], kind = #vector.kind<add>}
      %v, %b, %r : vector<2xf32>, vector<2x16xf32> into vector<16xf32>
  return %y : vector<16xf32>
}
""".replace("%a: vector<1x2xf32>", "%v: vector<2xf32>")

# An iter arg also used by something else than a contraction.
ITER_ARG_OTHER_USE = _k_loop(0).replace(
    "    scf.yield %y0, %y1, %y2, %y3",
    "    %e = arith.addf %x0, %x1 : vector<16x16xf32>\n    scf.yield %e, %y1, %y2, %y3",
)

_ACC64 = "memref<64x64xf32, strided<[64, 1], offset: ?>>"
_V8 = ", ".join(["vector<16x16xf32>"] * 8)
# A 2x4 group of accumulators (32x64): not the tile the rewrite expects, whose
# C rows it reads 32 wide.
GROUP_2X4 = (
    f"""
func.func @f(%buf: memref<64x64xf32>, %i: index, %a: vector<16x2xf32>,
             %b: vector<2x16xf32>) -> ({_V8}) {{
{_CONSTANTS}
  %pad = arith.constant 0.0 : f32
  %s = memref.subview %buf[%i, 0] [64, 64] [1, 1] : memref<64x64xf32> to {_ACC64}
"""
    + "\n".join(
        f"  %x{t} = vector.transfer_read %s[%c{r}, %c{c}], %pad "
        f"{{in_bounds = [true, true]}} : {_ACC64}, vector<16x16xf32>"
        for t, (r, c) in enumerate((r, c) for r in (0, 16) for c in (0, 16, 32, 48))
    )
    + "\n"
    + "\n".join(
        "  " + _contract_op(f"%y{t}", "%a", "%b", f"%x{t}", 16, 16) for t in range(8)
    )
    + f"""
  return {", ".join(f"%y{t}" for t in range(8))} : {_V8}
}}
"""
)


# A 4x2 group of accumulators (64x32, 8 contractions in one block) of a
# 128x32 subview: not the 2x2 tile the rewrite handles.
_ACC128 = "memref<128x32xf32, strided<[32, 1], offset: ?>>"
GROUP_4X2 = (
    f"""
func.func @f(%buf: memref<128x32xf32>, %i: index, %a: vector<16x2xf32>,
             %b: vector<2x16xf32>) -> ({_V8}) {{
{_CONSTANTS}
  %pad = arith.constant 0.0 : f32
  %s = memref.subview %buf[%i, 0] [128, 32] [1, 1] : memref<128x32xf32> to {_ACC128}
"""
    + "\n".join(
        f"  %x{t} = vector.transfer_read %s[%c{r}, %c{c}], %pad "
        f"{{in_bounds = [true, true]}} : {_ACC128}, vector<16x16xf32>"
        for t, (r, c) in enumerate((r, c) for r in (0, 16, 32, 48) for c in (0, 16))
    )
    + "\n"
    + "\n".join(
        "  " + _contract_op(f"%y{t}", "%a", "%b", f"%x{t}", 16, 16) for t in range(8)
    )
    + f"""
  return {", ".join(f"%y{t}" for t in range(8))} : {_V8}
}}
"""
)


@pytest.mark.parametrize(
    "source",
    [
        # A runtime index.
        _one_read(
            f"%r = vector.transfer_read %s[%j, %c16], %pad {_IN_BOUNDS} "
            ": {t}, vector<16x16xf32>"
        ),
        # A bounds-checked read: its bounds are the subview's sizes.
        _one_read(
            "%r = vector.transfer_read %s[%c16, %c16], %pad : {t}, vector<16x16xf32>"
        ),
        # A masked read.
        _one_read(
            f"%r = vector.transfer_read %s[%c16, %c16], %pad, %m {_IN_BOUNDS} "
            ": {t}, vector<16x16xf32>"
        ),
        # A permuted read.
        _one_read(
            "%r = vector.transfer_read %s[%c16, %c16], %pad {{in_bounds = [true, true], "
            "permutation_map = affine_map<(d0, d1) -> (d1, d0)>}} : {t}, "
            "vector<16x16xf32>"
        ),
        # Another use (a nested subview).
        _one_read(
            _READ_AT_16,
            extra="%t = memref.subview %s[0, 0] [16, 16] [1, 1] : "
            "memref<64x32xf32, strided<[32, 1], offset: ?>> to "
            "memref<16x16xf32, strided<[32, 1], offset: ?>>",
        ),
        # A contraction operand: AMX tile loads read operand subviews in place.
        _one_read(_READ_AT_16, use="rhs"),
        # Reads that fill the subview already.
        _one_read(_READ_AT_16, sizes="32, 32", shape="32x32"),
        # A read past the subview's end (out of bounds).
        _one_read(
            f"%r = vector.transfer_read %s[%c48, %c16], %pad {_IN_BOUNDS} "
            ": {t}, vector<16x16xf32>",
            sizes="32, 32",
            shape="32x32",
        ),
        # A group that is no 2x2 tile (rows 0-16 and 32-48).
        _one_read(
            f"%r = vector.transfer_read %s[%c32, %c16], %pad {_IN_BOUNDS} "
            ": {t}, vector<16x16xf32>"
        ),
        # Non-unit strides.
        _one_read(
            _READ_AT_16,
            sizes="32, 32",
            strides="2, 1",
            shape="32x32",
            layout="64, 1",
        ),
        RANK_REDUCING_SUBVIEW,
        ITER_ARG_OTHER_USE,
        GROUP_2X4,
        GROUP_4X2,
    ],
    ids=[
        "runtime_index",
        "bounds_checked",
        "masked",
        "permuted",
        "other_use",
        "operand",
        "filled",
        "past_end",
        "not_a_tile",
        "strided",
        "rank_reducing",
        "iter_arg_other_use",
        "group_2x4",
        "group_4x2",
    ],
)
def test_fit_subviews_to_reads_leaves_others(source: str) -> None:
    assert _apply(fit_subviews_to_reads, source) == _parsed(source)


def test_fit_subviews_to_reads_negative_cases_are_fittable_reads() -> None:
    """The ``_one_read`` negative cases differ from a fitted group in one
    property each: with the fourth read in place, the 2x2 tile is fitted."""
    text = _apply(fit_subviews_to_reads, _one_read(_READ_AT_16))
    assert "[32, 32] [1, 1]" in text


# bf16 K loop of the AMX rewrite (as amx_acc_buffer_larger_than_tile.mlir in
# the upstream reproducer), its 32x32 accumulator tile at ``row`` of a 64x32
# subview.
def _bf16_k_loop(row: int) -> str:
    sa = "memref<32x64xbf16, strided<[64, 1], offset: ?>>"
    sb = "memref<64x32xbf16, strided<[32, 1], offset: ?>>"
    contracts = "\n".join(
        "    "
        + _contract_op(
            f"%y{t}", f"%a{t // 2}", f"%b{t % 2}", f"%x{t}", 16, 16, "bf16", 32
        )
        for t in range(4)
    )
    writes = "\n".join(
        f"  vector.transfer_write %r#{t}, %out[%c{r}, %c{c}] {{in_bounds = [true, true]}}"
        " : vector<16x16xf32>, memref<32x32xf32>"
        for t, (r, c) in enumerate([(0, 0), (0, 16), (16, 0), (16, 16)])
    )
    return f"""
func.func @f(%a: memref<32x64xbf16>, %b: memref<64x32xbf16>, %buf: memref<64x32xf32>,
             %out: memref<32x32xf32>, %i: index) {{
{_CONSTANTS}
  %c64 = arith.constant 64 : index
  %pb = arith.constant 0.0 : bf16
  %pad = arith.constant 0.0 : f32
  %s = memref.subview %buf[%i, 0] [64, 32] [1, 1] : memref<64x32xf32> to {_ACC}
{_acc_reads(row)}
  %r:4 = scf.for %k = %c0 to %c64 step %c32
      iter_args(%x0 = %i0, %x1 = %i1, %x2 = %i2, %x3 = %i3) -> ({_V4}) {{
    %sa = memref.subview %a[0, %k] [32, 64] [1, 1] : memref<32x64xbf16> to {sa}
    %sb = memref.subview %b[%k, 0] [64, 32] [1, 1] : memref<64x32xbf16> to {sb}
    %a0 = vector.transfer_read %sa[%c0, %c0], %pb {{in_bounds = [true, true]}} : {sa}, vector<16x32xbf16>
    %a1 = vector.transfer_read %sa[%c16, %c0], %pb {{in_bounds = [true, true]}} : {sa}, vector<16x32xbf16>
    %b0 = vector.transfer_read %sb[%c0, %c0], %pb {{in_bounds = [true, true]}} : {sb}, vector<32x16xbf16>
    %b1 = vector.transfer_read %sb[%c0, %c16], %pb {{in_bounds = [true, true]}} : {sb}, vector<32x16xbf16>
{contracts}
    scf.yield %y0, %y1, %y2, %y3 : {_V4}
  }}
{writes}
  return
}}
"""


@pytest.mark.isolated
@pytest.mark.parametrize("row", [0, 16, 32])
def test_fitted_accumulator_tile_lowers_to_amx(row: int) -> None:
    """Upstream's AMX rewrite after the fit: it rewrites the K loop (unfitted, a
    tile at row 0 aborts the process, one at row 16 or 32 too: 48 or 64 rows
    are taken as the tile, or the tile at the origin)."""
    with ir.Context(), ir.Location.unknown():
        HelionTransformDialect.load()
        module = ir.Module.parse(_bf16_k_loop(row))
        with schedule_boilerplate() as (schedule, named_seq):
            funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
            fit_subviews_to_reads(funcs)
            with ir.InsertionPoint(transform.ApplyPatternsOp(funcs).patterns):
                x86.apply_patterns_x86_vector_contract_to_amx_dot_product()
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        text = str(module)
    assert "x86.amx.tile_mulf" in text
    assert "vector.contract" not in text


# --- The opt pipeline with AMX ------------------------------------------------


def _lower_to_x86_vectorization(module: ir.Module, entry: str) -> tuple[str, str]:
    """The module before and after the x86 vector rewrites (AMX included), with
    AMX forced on."""
    lowered = lower_opt_with_amx(
        module,
        entry,
        lambda stage, _: "x86.vector_contract_to_amx_dot_product" in stage,
    )
    assert lowered is not None, "no x86_vectorization stage"
    return lowered[0], str(lowered[1])


def _contracts(text: str) -> list[str]:
    return [line for line in text.splitlines() if "vector.contract" in line]


@pytest.mark.slow
@pytest.mark.isolated
def test_linear_bf16_reaches_amx_with_k_by_n_operand() -> None:
    """KernelBench 40's ``x @ weight.T + bias`` (bf16, blocks [32, 64, 32], as
    ``test_level2_matches_eager_on_cpu``): the contractions reach AMX reading
    B as K x N (a transposed copy), not as the N x K tiles the rewrite packs
    as if K x N."""
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[32, 64, 32]),
    )(_linear_scaling_residual_add)
    x = torch.randn(64, 256, dtype=torch.bfloat16)
    weight = torch.randn(128, 256, dtype=torch.bfloat16)
    bias = torch.randn(128, dtype=torch.bfloat16)
    module = generate_mlir(kernel, [x, weight, bias, hl.constexpr(0.5)])
    before, after = _lower_to_x86_vectorization(module, kernel.fn.__name__)
    assert _contracts(before)
    # B tiles K x N (32 x 16); N x K tiles would be 16 x 32, as A tiles are.
    assert all(
        "vector<16x32xbf16>, vector<32x16xbf16>" in line for line in _contracts(before)
    )
    assert "x86.amx.tile_mulf" in after
    assert "vector.contract" not in after


@pytest.mark.slow
@pytest.mark.isolated
def test_attention_bf16_without_vnni_reaches_amx() -> None:
    """``test_attention_bf16_without_vnni``'s kernel (S=64, D=32, blocks 64):
    P @ V's K loop accumulates into a 32x32 register tile of a 64x32 buffer.
    Read through a 64x32 subview, the AMX rewrite aborted the process."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 64, 32).to(torch.bfloat16) for _ in range(3))
    kernel = attention_kernel(64, 64)
    args = [q, k, v, hl.constexpr(32**-0.5), hl.constexpr(False)]
    module = generate_mlir(kernel, args)
    before, after = _lower_to_x86_vectorization(module, kernel.fn.__name__)
    assert _contracts(before)
    assert all(
        "vector<16x32xbf16>, vector<32x16xbf16>" in line for line in _contracts(before)
    )
    assert after.count("x86.amx.tile_mulf") >= 8
    assert "vector.contract" not in after


def _x86_input(name: str, amx: bool) -> str:
    """``_case(name)``'s module as the x86 vector rewrites get it, on a target
    with or without AMX (the kept-off mark removed)."""
    kernel, args, _ = _case(name)
    module = generate_mlir(kernel, args)
    lowered = lower_opt_with_amx(
        module,
        kernel.fn.__name__,
        lambda stage, _: "x86.vector_contract_to_amx_dot_product" in stage,
        amx=amx,
    )
    assert lowered is not None
    return re.sub(rf"{AMX_OFF_ATTR_NAME}(, )?", "", lowered[0]).replace(" {}", "")


@pytest.mark.slow
@pytest.mark.isolated
@pytest.mark.parametrize(
    "name", ["decode_2x32", "decode_32x64", "bmv", "acc_b_inner", "split_k"]
)
def test_kept_off_amx_lowers_as_without_amx(name: str) -> None:
    """Kept off AMX, a contraction reaches the x86 rewrites exactly as on a
    host without AMX (where it is correct), on any host. Unfixed, decode at
    [32, 64] aborted the process (ArrayRef.h:439) and split K was miscompiled."""
    assert _x86_input(name, amx=True) == _x86_input(name, amx=False)


# --- AMX numerics (AMX hosts) -------------------------------------------------


def _snippet(lhs: str, rhs: str, acc: str, dims: str, ranks: tuple[int, ...]) -> str:
    """An ``inline_mlir`` contraction of dynamic-shape tiles, ``acc`` zeroed."""

    def t(rank: int, element: str) -> str:
        return "tensor<" + "x".join(["?"] * rank) + f"x{element}>"

    a, b, o = t(ranks[0], "bf16"), t(ranks[1], "bf16"), t(ranks[2], "f32")
    return f"""
func.func @f(%a: {a}, %b: {b}, %o: {o}) -> {o} {{
  %z = arith.constant 0.0 : f32
  %c = linalg.fill ins(%z : f32) outs(%o : {o}) -> {o}
  %r = linalg.contract indexing_maps = [affine_map<({dims}) -> ({lhs})>,
      affine_map<({dims}) -> ({rhs})>, affine_map<({dims}) -> ({acc})>]
      ins(%a, %b : {a}, {b}) outs(%c : {o}) -> {o}
  return %r : {o}
}}
"""


_ACC_NM_SNIPPET = _snippet("m, k", "k, n", "n, m", "m, n, k", (2, 2, 2))
_BOTH_SNIPPET = _snippet("k, m", "n, k", "m, n", "m, n, k", (2, 2, 2))
_LINEAR_3D_SNIPPET = _snippet("b, m, k", "n, k", "b, m, n", "b, m, n, k", (3, 2, 3))
_BRGEMM_NT_SNIPPET = _snippet("b, m, k", "b, n, k", "m, n", "b, m, n, k", (3, 3, 2))


def _config(*blocks: int) -> helion.Config:
    return helion.Config(block_sizes=list(blocks))


@helion.kernel(backend="mlir", static_shapes=True, config=_config(32, 32))
def _acc_nm_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, _ = a.size()
    _, n = b.size()
    out = torch.empty([n, m], dtype=torch.float32, device=a.device)
    for tn, tm in hl.tile([n, m]):
        out[tn, tm] = inline_mlir(
            _ACC_NM_SNIPPET, [a[tm, :], b[:, tn], out[tn, tm]], out[tn, tm]
        )
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_config(32, 32))
def _both_swapped_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _, m = a.size()
    n, _ = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = inline_mlir(
            _BOTH_SNIPPET, [a[:, tm], b[tn, :], out[tm, tn]], out[tm, tn]
        )
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_config(2, 32, 32))
def _linear_3d_kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    batch, m, _ = x.size()
    n, _ = w.size()
    out = torch.empty([batch, m, n], dtype=torch.float32, device=x.device)
    for tb, tm, tn in hl.tile([batch, m, n]):
        out[tb, tm, tn] = inline_mlir(
            _LINEAR_3D_SNIPPET,
            [x[tb, tm, :], w[tn, :], out[tb, tm, tn]],
            out[tb, tm, tn],
        )
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_config(32, 32))
def _brgemm_nt_kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    _, m, _ = x.size()
    _, n, _ = w.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = inline_mlir(
            _BRGEMM_NT_SNIPPET, [x[:, tm, :], w[:, tn, :], out[tm, tn]], out[tm, tn]
        )
    return out


# Plain Helion, dynamic shapes: x @ w.T accumulated over a K loop.
@helion.kernel(backend="mlir", static_shapes=False, config=_config(32, 32, 32))
def _linear_dynamic_kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n, _ = w.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tn, tk].T)
        out[tm, tn] = acc
    return out


_VNNI_ACC_NM_SNIPPET = _snippet("m, k, v", "k, n, v", "n, m", "m, n, k, v", (3, 3, 2))
_BMV_SNIPPET = _snippet("b, m, k", "b, k", "b, m", "b, m, k", (3, 2, 2))
_ACC_B_INNER_SNIPPET = _snippet(
    "b, m, k", "b, k, n", "m, n, b", "b, m, n, k", (3, 3, 3)
)


@helion.kernel(backend="mlir", static_shapes=True, config=_config(32, 32))
def _vnni_acc_nm_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, _, _ = a.size()
    _, n, _ = b.size()
    out = torch.empty([n, m], dtype=torch.float32, device=a.device)
    for tn, tm in hl.tile([n, m]):
        out[tn, tm] = inline_mlir(
            _VNNI_ACC_NM_SNIPPET, [a[tm, :, :], b[:, tn, :], out[tn, tm]], out[tn, tm]
        )
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_config(2, 32))
def _bmv_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batch, m, _ = x.size()
    out = torch.empty([batch, m], dtype=torch.float32, device=x.device)
    for tb, tm in hl.tile([batch, m]):
        out[tb, tm] = inline_mlir(
            _BMV_SNIPPET, [x[tb, tm, :], y[tb, :], out[tb, tm]], out[tb, tm]
        )
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_config(32, 32, 2))
def _acc_b_inner_kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    batch, m, _ = x.size()
    n = w.size(2)
    out = torch.empty([m, n, batch], dtype=torch.float32, device=x.device)
    for tm, tn, tb in hl.tile([m, n, batch]):
        out[tm, tn, tb] = inline_mlir(
            _ACC_B_INNER_SNIPPET,
            [x[tb, tm, :], w[tb, :, tn], out[tm, tn, tb]],
            out[tm, tn, tb],
        )
    return out


# Plain Helion split K: two K dims, B's innermost N (no VNNI layout).
@helion.kernel(backend="mlir", static_shapes=True, config=_config(32, 32))
def _split_k_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, _, _ = a.size()
    n = b.size(2)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = torch.einsum("mkj,kjn->mn", a[tm, :, :], b[:, :, tn]).to(
            torch.float32
        )
    return out


def _decode(q: torch.Tensor, kc: torch.Tensor) -> torch.Tensor:
    """Decode-style attention scores: a batched dot with no M dim."""
    batch, n, _ = kc.size()
    out = torch.empty([batch, n], dtype=torch.float32, device=q.device)
    for tb, tn in hl.tile([batch, n]):
        out[tb, tn] = (
            q[tb, None, :].to(torch.float32) * kc[tb, tn, :].to(torch.float32)
        ).sum(-1)
    return out


def _case(name: str) -> tuple[helion.Kernel, list[torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(0)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    if name == "acc_nm":
        a, b = randn(64, 64), randn(64, 64)
        return _acc_nm_kernel, [a, b], (a.float() @ b.float()).T
    if name == "both_swapped":
        a, b = randn(64, 64), randn(64, 64)
        return _both_swapped_kernel, [a, b], a.float().T @ b.float().T
    if name == "linear_3d":
        x, w = randn(2, 64, 64), randn(64, 64)
        return _linear_3d_kernel, [x, w], x.float() @ w.float().T
    if name == "brgemm_nt":
        x, w = randn(2, 64, 64), randn(2, 64, 64)
        return (
            _brgemm_nt_kernel,
            [x, w],
            torch.einsum("bmk,bnk->mn", x.float(), w.float()),
        )
    if name == "linear_dynamic":
        x, w = randn(64, 128), randn(64, 128)
        return _linear_dynamic_kernel, [x, w], x.float() @ w.float().T
    if name == "vnni_acc_nm":
        a, b = randn(64, 32, 2), randn(32, 64, 2)
        return (
            _vnni_acc_nm_kernel,
            [a, b],
            torch.einsum("mkv,knv->nm", a.float(), b.float()),
        )
    if name == "bmv":
        x, y = randn(4, 64, 64), randn(4, 64)
        return _bmv_kernel, [x, y], torch.einsum("bmk,bk->bm", x.float(), y.float())
    if name == "acc_b_inner":
        x, w = randn(2, 64, 64), randn(2, 64, 64)
        return (
            _acc_b_inner_kernel,
            [x, w],
            torch.einsum("bmk,bkn->mnb", x.float(), w.float()),
        )
    if name == "split_k":
        a, b = randn(64, 2, 32), randn(2, 32, 64)
        return (
            _split_k_kernel,
            [a, b],
            torch.einsum("mkj,kjn->mn", a.float(), b.float()),
        )
    blocks = [int(size) for size in name.split("_")[1].split("x")]
    kernel = helion.kernel(
        _decode,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=blocks),
    )
    q, kc = randn(64, 64), randn(64, 64, 64)
    return kernel, [q, kc], torch.einsum("bd,bnd->bn", q.float(), kc.float())


@amx_host_only
@pytest.mark.slow
@pytest.mark.isolated
@pytest.mark.parametrize(
    "name",
    [
        "acc_nm",
        "both_swapped",
        "linear_3d",
        "brgemm_nt",
        "linear_dynamic",
        "vnni_acc_nm",
    ],
)
def test_transposed_layouts_on_amx(name: str) -> None:
    """Layouts the AMX rewrite reads wrongly, normalized and executed on AMX:
    a transposed accumulator (flat and VNNI), both inputs swapped, linear on a
    3-D input (two M dims), batch-reduce with B as N x K (two K dims), and
    ``x @ w.T`` of runtime shapes. Unfixed, the normalized error was above 1."""
    kernel, args, expected = _case(name)
    out = kernel(*args)
    error = (out.float() - expected).abs().max() / expected.abs().max()
    # f32 accumulation of bf16 products: only the summation order differs.
    assert error.item() < 1e-4


@amx_host_only
@pytest.mark.slow
@pytest.mark.isolated
@pytest.mark.parametrize(
    ("name", "tolerance"),
    [
        # Decode at blocks (b, n) = (2, 32): no AMX tile; at (32, 64) the
        # rewrite aborted (ArrayRef.h:439).
        ("decode_2x32", 1e-4),
        ("decode_32x64", 1e-4),
        ("bmv", 1e-4),
        ("acc_b_inner", 1e-4),
        # bf16 einsum result: one bf16 rounding of the output.
        ("split_k", 1e-2),
    ],
)
def test_layouts_kept_off_amx(name: str, tolerance: float) -> None:
    """Layouts the transform cannot normalize run as f32 contractions, not on
    AMX: correct, no abort (split K was miscompiled, error 1.44; decode at
    32x64 aborted; the rest were correct and must stay so)."""
    kernel, args, expected = _case(name)
    out = kernel(*args)
    error = (out.float() - expected).abs().max() / expected.abs().max()
    assert error.item() < tolerance


def _linear_scaling_residual_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: hl.constexpr,
) -> torch.Tensor:
    """KernelBench level 2 40: ``y * scaling_factor + y`` of ``y = linear(x)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = (y * scaling_factor + y).to(x.dtype)
    return out
