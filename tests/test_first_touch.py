"""``mark_overwritten_args`` and ``first_touch_outputs``: fresh outputs faulted in
by each thread in a contiguous slice before the parallel loop writing them."""

from __future__ import annotations

import ctypes
import mmap
import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import helion
import helion.language as hl
from lighthouse.utils.lib_finder import find_openmp_library
from mlir import ir
from mlir import passmanager
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor
import numpy as np
import pytest
import torch

from tests.harness import opt_pipeline
from tests.harness import visits

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.helion_transforms import first_touch_outputs
from helion_mlir_backend._compiler.helion_transforms import mark_overwritten_args
from helion_mlir_backend._compiler.mlir_transforms import OVERWRITTEN_ATTR_NAME
from helion_mlir_backend._compiler.mlir_transforms import first_touch
from helion_mlir_backend._compiler.mlir_transforms.first_touch import MIN_REDUCTION
from helion_mlir_backend._compiler.mlir_transforms.first_touch import _fully_written
from helion_mlir_backend._compiler.mlir_transforms.first_touch import _reduction_length
from helion_mlir_backend._compiler.mlir_transforms.first_touch import _scattered

if TYPE_CHECKING:
    from collections.abc import Callable

MARK = OVERWRITTEN_ATTR_NAME
TOUCH = "llvm.call @madvise"


def _apply(make_schedule: Callable[[], ir.Module], source: str) -> str:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        schedule = make_schedule()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


# Total reduction length (K) of the contractions producing the test tiles: above
# the gate's MIN_REDUCTION.
LONG_K = 4096


def _gemm(
    name: str, m: int, n: int, k: int = LONG_K, a: str = "%ka", b: str = "%kb"
) -> str:
    """IR defining ``%name``, an ``m x n`` bf16 tile: the bf16 rounding (an
    all-parallel generic) of an f32 ``linalg.matmul`` of ``a`` (``m x k``) and
    ``b`` (``k x n``) into a zeroed accumulator, as a GEMM's epilogue."""
    return f"""%{name}_z = arith.constant 0.0 : f32
    %{name}_e = tensor.empty() : tensor<{m}x{n}xf32>
    %{name}_f = linalg.fill ins(%{name}_z : f32) outs(%{name}_e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    %{name}_m = linalg.matmul ins({a}, {b} : tensor<{m}x{k}xbf16>, tensor<{k}x{n}xbf16>) outs(%{name}_f : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    %{name}_b = tensor.empty() : tensor<{m}x{n}xbf16>
    %{name} = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} ins(%{name}_m : tensor<{m}x{n}xf32>) outs(%{name}_b : tensor<{m}x{n}xbf16>) {{
    ^bb0(%in: f32, %init: bf16):
      %rounded = arith.truncf %in : f32 to bf16
      linalg.yield %rounded : bf16
    }} -> tensor<{m}x{n}xbf16>"""


def _forall(
    trips: str = "2, 32",
    offsets: str = "%r, %c",
    *,
    sizes: str = "2048, 128",
    out: str = "4096x4096",
    writable: str = "restrict writable",
    bounds: str | None = None,
    dest: str = "%arg0",
    inserts: int = 1,
    nested: bool = False,
    k: int = LONG_K,
) -> str:
    """A function whose top-level forall writes tiles of ``sizes`` (static, or
    ragged with ``%m``) at ``offsets`` into its argument ``%arg0`` (IVs ``%i``,
    ``%j``; ``%r`` = ``%i * 2048``, ``%c`` = ``%j * 128``, ``%rj`` = ``%j * 2048``,
    ``%m`` = ``min(128, 4032 - %j * 128)``). Default: 2048x128 bf16 tiles of a
    4096x4096 output, as AI-bench's 4096 GEMM."""
    static = sizes.replace("%m", "?").replace(", ", "x")
    loop = bounds or f"(%i, %j) in ({trips})"
    insert = (
        f"tensor.parallel_insert_slice %s into %o[{offsets}] [{sizes}] [1, 1]"
        f" : tensor<{static}xbf16> into tensor<{out}xbf16>"
    )
    body = f"""
  %0 = bufferization.to_tensor %arg0 {writable} : memref<{out}xbf16> to tensor<{out}xbf16>
  %1 = scf.forall {loop} shared_outs(%o = %0) -> (tensor<{out}xbf16>) {{
    %r = affine.apply affine_map<(d0) -> (d0 * 2048)>(%i)
    %c = affine.apply affine_map<(d0) -> (d0 * 128)>(%j)
    %rj = affine.apply affine_map<(d0) -> (d0 * 2048)>(%j)
    %m = affine.min affine_map<(d0) -> (d0 * -128 + 4032, 128)>(%j)
    {_gemm("tc", 4096, 4096, k)}
    %s = tensor.extract_slice %tc[0, 0] [{sizes}] [1, 1] : tensor<4096x4096xbf16> to tensor<{static}xbf16>
    scf.forall.in_parallel {{
      {chr(10).join([insert] * inserts)}
    }}
  }}
  bufferization.materialize_in_destination %1 in restrict writable {dest} : (tensor<{out}xbf16>, memref<{out}xbf16>) -> ()"""
    if nested:
        body = f"""
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %k = %c0 to %c1 step %c1 {{{body}
  }}"""
    return f"""
func.func @f(%arg0: memref<{out}xbf16>, %arg1: memref<{out}xbf16>, %ka: tensor<4096x{k}xbf16>, %kb: tensor<{k}x4096xbf16>) {{{body}
  return
}}
"""


def _marked(source: str) -> list[bool]:
    """Per argument of ``@f``: whether ``mark_overwritten_args`` marked it."""
    text = _apply(mark_overwritten_args, source)
    signature = text[text.index("func.func @f(") : text.index(") {")]
    return [MARK in arg for arg in signature.split("%arg")[1:]]


def test_mark_overwritten_args_marks_tall_tiles() -> None:
    # Each tile spans 16 MiB of rows, 256 bytes of each: 32 threads per 2 MiB.
    assert _marked(_forall())[0]


def test_mark_overwritten_args_marks_ragged_tiles() -> None:
    # 32 tiles of 128 columns over 4032: the last one 64 wide.
    assert _marked(_forall(sizes="2048, %m", out="4096x4032"))[0]


@pytest.mark.parametrize(
    ("trips", "offsets", "sizes"),
    [
        # Whole rows: each slice contiguous (as an elementwise kernel's tiles).
        ("128, 1", "%r32, 0", "32, 4096"),
        # Partial rows spanning 512 KiB only: few threads per 2 MiB range.
        ("64, 32", "%r64, %c", "64, 128"),
    ],
)
def test_mark_overwritten_args_keeps_contiguous_or_short_slices(
    trips: str, offsets: str, sizes: str
) -> None:
    source = _forall(trips, offsets, sizes=sizes)
    source = source.replace(
        "    %m = ",
        "    %r32 = affine.apply affine_map<(d0) -> (d0 * 32)>(%i)\n"
        "    %r64 = affine.apply affine_map<(d0) -> (d0 * 64)>(%i)\n    %m = ",
    )
    assert not _marked(source)[0]


def test_mark_overwritten_args_keeps_unit_tiles_of_whole_inner_dims() -> None:
    # As the f32 pack kernels: an IV per outer dim, whole (contiguous) inner dims.
    source = """
func.func @f(%arg0: memref<128x128x32x32xf32>, %t: tensor<1x1x32x32xf32>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<128x128x32x32xf32> to tensor<128x128x32x32xf32>
  %1 = scf.forall (%i, %j) in (128, 128) shared_outs(%o = %0) -> (tensor<128x128x32x32xf32>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %t into %o[%i, %j, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<1x1x32x32xf32> into tensor<128x128x32x32xf32>
    }
  }
  bufferization.materialize_in_destination %1 in restrict writable %arg0 : (tensor<128x128x32x32xf32>, memref<128x128x32x32xf32>) -> ()
  return
}
"""
    text = _apply(mark_overwritten_args, source)
    assert MARK not in text


@pytest.mark.parametrize(
    "source",
    [
        _forall("1, 32"),  # rows 2048..4095 never written
        _forall("2, 31"),  # the last column tile never written
        _forall(offsets="%rj, %c"),  # %j in both dims, %i in none: a diagonal
        _forall(offsets="%r, 0"),  # every tile at column 0
        # min(128, 4032 - %j * 128) on a 4096-wide output: 64 columns unwritten.
        _forall(sizes="2048, %m"),
        _forall(inserts=2),  # two inserts into the output
        _forall(writable="restrict"),  # not a writable view of the argument
        _forall(dest="%arg1"),  # the result is stored into another argument
        _forall(bounds="(%i, %j) = (0, 0) to (4, 32) step (2, 1)"),  # step 2
        _forall(bounds="(%i, %j) = (1, 0) to (3, 32) step (1, 1)"),  # from 1
        _forall(nested=True),  # in a loop: not the function's top level
    ],
    ids=[
        "rows",
        "columns",
        "diagonal",
        "same-column",
        "ragged-bound",
        "two-inserts",
        "not-writable",
        "other-argument",
        "step",
        "lower-bound",
        "nested",
    ],
)
def test_mark_overwritten_args_keeps_other_writes(source: str) -> None:
    assert not any(_marked(source))


def test_mark_overwritten_args_marks_only_the_covered_output() -> None:
    # Two shared outputs: %arg0 written whole, %arg1 in its first column tile only.
    source = (
        """
func.func @f(%arg0: memref<4096x4096xbf16>, %arg1: memref<4096x4096xbf16>, %ka: tensor<2048x4096xbf16>, %kb: tensor<4096x128xbf16>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4096x4096xbf16> to tensor<4096x4096xbf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<4096x4096xbf16> to tensor<4096x4096xbf16>
  %2:2 = scf.forall (%i, %j) in (2, 32) shared_outs(%o = %0, %p = %1) -> (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) {
    %r = affine.apply affine_map<(d0) -> (d0 * 2048)>(%i)
    %c = affine.apply affine_map<(d0) -> (d0 * 128)>(%j)
    """
        + _gemm("tw", 2048, 128)
        + """
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tw into %o[%r, %c] [2048, 128] [1, 1] : tensor<2048x128xbf16> into tensor<4096x4096xbf16>
      tensor.parallel_insert_slice %tw into %p[%r, 0] [2048, 128] [1, 1] : tensor<2048x128xbf16> into tensor<4096x4096xbf16>
    }
  }
  bufferization.materialize_in_destination %2#0 in restrict writable %arg0 : (tensor<4096x4096xbf16>, memref<4096x4096xbf16>) -> ()
  bufferization.materialize_in_destination %2#1 in restrict writable %arg1 : (tensor<4096x4096xbf16>, memref<4096x4096xbf16>) -> ()
  return
}
"""
    )
    assert _marked(source)[:2] == [True, False]


def test_mark_overwritten_args_needs_the_result_in_the_argument() -> None:
    # The forall's result is returned, not materialized into the argument.
    source = (
        """
func.func @f(%arg0: memref<4096x4096xbf16>, %ka: tensor<2048x4096xbf16>, %kb: tensor<4096x128xbf16>) -> tensor<4096x4096xbf16> {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4096x4096xbf16> to tensor<4096x4096xbf16>
  %1 = scf.forall (%i, %j) in (2, 32) shared_outs(%o = %0) -> (tensor<4096x4096xbf16>) {
    %r = affine.apply affine_map<(d0) -> (d0 * 2048)>(%i)
    %c = affine.apply affine_map<(d0) -> (d0 * 128)>(%j)
    """
        + _gemm("tw", 2048, 128)
        + """
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tw into %o[%r, %c] [2048, 128] [1, 1] : tensor<2048x128xbf16> into tensor<4096x4096xbf16>
    }
  }
  return %1 : tensor<4096x4096xbf16>
}
"""
    )
    assert MARK not in _apply(mark_overwritten_args, source)


def _tile_body(tile: str, insert: str, *, elem: str = "bf16") -> str:
    """A 2x32 forall over 2048x128 tiles of a 4096x4096 ``elem`` output whose
    body is ``tile`` (``%old``: the old tile; ``%t``, ``%u``: GEMM tiles of
    K = LONG_K, ``_gemm``; ``%t0``, ``%u0``: argument tiles; ``%b``: a flag) and
    which inserts ``insert`` at the tile."""
    out = f"tensor<4096x4096x{elem}>"
    piece = f"tensor<2048x128x{elem}>"
    if elem == "bf16":
        tiles = _gemm("t", 2048, 128) + "\n    " + _gemm("u", 2048, 128)
    else:  # no bf16 GEMM of this element type: copies of the arguments
        tiles = (
            f"%t = linalg.copy ins(%t0 : {piece}) outs(%et : {piece}) -> {piece}\n"
            f"    %u = linalg.copy ins(%u0 : {piece}) outs(%et : {piece}) -> {piece}"
        )
    return f"""
func.func @f(%arg0: memref<4096x4096x{elem}>, %t0: {piece}, %u0: {piece}, %b: i1,
             %ka: tensor<2048x4096xbf16>, %kb: tensor<4096x128xbf16>) {{
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4096x4096x{elem}> to {out}
  %1 = scf.forall (%i, %j) in (2, 32) shared_outs(%o = %0) -> ({out}) {{
    %r = affine.apply affine_map<(d0) -> (d0 * 2048)>(%i)
    %c = affine.apply affine_map<(d0) -> (d0 * 128)>(%j)
    %old = tensor.extract_slice %o[%r, %c] [2048, 128] [1, 1] : {out} to {piece}
    %et = tensor.empty() : {piece}
    {tiles}
    {tile}
    scf.forall.in_parallel {{
      tensor.parallel_insert_slice {insert} into %o[%r, %c] [2048, 128] [1, 1] : {piece} into {out}
    }}
  }}
  bufferization.materialize_in_destination %1 in restrict writable %arg0 : ({out}, memref<4096x4096x{elem}>) -> ()
  return
}}
"""


# A conditional store (``if flag: out[tile] = ...``): the old tile in the else
# branch, which bufferization leaves in place without writing it.
CONDITIONAL_STORE = _tile_body(
    """%v = scf.if %b -> (tensor<2048x128xbf16>) {
      scf.yield %t : tensor<2048x128xbf16>
    } else {
      scf.yield %old : tensor<2048x128xbf16>
    }""",
    "%v",
)


@pytest.mark.parametrize(
    "source",
    [
        CONDITIONAL_STORE,
        _tile_body("", "%old"),  # the old tile stored back
        _tile_body("%e = tensor.empty() : tensor<2048x128xbf16>", "%e"),  # nothing
        # Half an empty tile written: eliminated into the output, half stays unwritten.
        _tile_body(
            "%e = tensor.empty() : tensor<2048x128xbf16>\n"
            "    %xs = tensor.extract_slice %t[0, 0] [1024, 128] [1, 1]"
            " : tensor<2048x128xbf16> to tensor<1024x128xbf16>\n"
            "    %v = tensor.insert_slice %xs into %e[0, 0] [1024, 128] [1, 1]"
            " : tensor<1024x128xbf16> into tensor<2048x128xbf16>",
            "%v",
        ),
        # An empty tile in one branch.
        _tile_body(
            """%e = tensor.empty() : tensor<2048x128xbf16>
    %v = scf.if %b -> (tensor<2048x128xbf16>) {
      scf.yield %t : tensor<2048x128xbf16>
    } else {
      scf.yield %e : tensor<2048x128xbf16>
    }""",
            "%v",
        ),
        # An empty tile through casts.
        _tile_body(
            "%e = tensor.empty() : tensor<2048x128xbf16>\n"
            "    %d = tensor.cast %e : tensor<2048x128xbf16> to tensor<?x128xbf16>\n"
            "    %v = tensor.cast %d : tensor<?x128xbf16> to tensor<2048x128xbf16>",
            "%v",
        ),
        # The argument also used through another view.
        _tile_body("", "%t").replace(
            "  %0 = bufferization.to_tensor",
            "  %view = memref.cast %arg0 : memref<4096x4096xbf16> to"
            " memref<?x4096xbf16>\n  %0 = bufferization.to_tensor",
        ),
        # A slice of the old tile written in part only.
        _tile_body(
            "%v = tensor.insert_slice %t into %old[0, 0] [2048, 128] [1, 1]"
            " : tensor<2048x128xbf16> into tensor<2048x128xbf16>\n"
            "    %w = tensor.extract_slice %v[0, 0] [1024, 128] [1, 1]"
            " : tensor<2048x128xbf16> to tensor<1024x128xbf16>\n"
            "    %x = tensor.insert_slice %w into %old[0, 0] [1024, 128] [1, 1]"
            " : tensor<1024x128xbf16> into tensor<2048x128xbf16>",
            "%x",
        ),
    ],
    ids=[
        "conditional",
        "old-tile",
        "empty",
        "half-empty",
        "empty-branch",
        "empty-cast",
        "other-view",
        "partial-destination",
    ],
)
def test_mark_overwritten_args_keeps_conditional_stores(source: str) -> None:
    assert not _marked(source)[0]


@pytest.mark.parametrize(
    "tile",
    [
        # As the reviewer's probe: an empty tile as a loop's init, a quarter written
        # per iteration (rows 1024-2047 never, after bufferization).
        """%e = tensor.empty() : tensor<2048x128xbf16>
    %xs = tensor.extract_slice %t[0, 0] [256, 128] [1, 1] : tensor<2048x128xbf16> to tensor<256x128xbf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %v = scf.for %k = %c0 to %c4 step %c1 iter_args(%a = %e) -> (tensor<2048x128xbf16>) {
      %off = arith.muli %k, %c256 : index
      %w = tensor.insert_slice %xs into %a[%off, 0] [256, 128] [1, 1] : tensor<256x128xbf16> into tensor<2048x128xbf16>
      scf.yield %w : tensor<2048x128xbf16>
    }""",
        # An empty tile reshaped, then written in half.
        """%e = tensor.empty() : tensor<2048x64x2xbf16>
    %ec = tensor.collapse_shape %e [[0], [1, 2]] : tensor<2048x64x2xbf16> into tensor<2048x128xbf16>
    %xs = tensor.extract_slice %t[0, 0] [1024, 128] [1, 1] : tensor<2048x128xbf16> to tensor<1024x128xbf16>
    %v = tensor.insert_slice %xs into %ec[0, 0] [1024, 128] [1, 1] : tensor<1024x128xbf16> into tensor<2048x128xbf16>""",
        # A reduction: the init is accumulated into, not overwritten.
        """%v = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%t3 : tensor<2048x128x4xbf16>) outs(%et : tensor<2048x128xbf16>) {
    ^bb0(%in: bf16, %acc: bf16):
      %sum = arith.addf %in, %acc : bf16
      linalg.yield %sum : bf16
    } -> tensor<2048x128xbf16>""",
        # An output map that is not a permutation of the loops (a projection).
        """%v = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%t3 : tensor<2048x128x4xbf16>) outs(%et : tensor<2048x128xbf16>) {
    ^bb0(%in: bf16, %init: bf16):
      linalg.yield %in : bf16
    } -> tensor<2048x128xbf16>""",
        # All parallel, but the payload reads the init (the old, unset values).
        """%v = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%t : tensor<2048x128xbf16>) outs(%et : tensor<2048x128xbf16>) {
    ^bb0(%in: bf16, %init: bf16):
      linalg.yield %init : bf16
    } -> tensor<2048x128xbf16>""",
        # Ops only forwarding an empty input: folded into it, nothing is written.
        """%e2 = tensor.empty() : tensor<2048x128xbf16>
    %v = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%e2 : tensor<2048x128xbf16>) outs(%et : tensor<2048x128xbf16>) {
    ^bb0(%in: bf16, %init: bf16):
      linalg.yield %in : bf16
    } -> tensor<2048x128xbf16>""",
        """%e2 = tensor.empty() : tensor<2048x128xbf16>
    %xs = tensor.extract_slice %t[0, 0] [1024, 128] [1, 1] : tensor<2048x128xbf16> to tensor<1024x128xbf16>
    %h = tensor.insert_slice %xs into %e2[0, 0] [1024, 128] [1, 1] : tensor<1024x128xbf16> into tensor<2048x128xbf16>
    %v = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%h : tensor<2048x128xbf16>) outs(%et : tensor<2048x128xbf16>) {
    ^bb0(%in: bf16, %init: bf16):
      linalg.yield %in : bf16
    } -> tensor<2048x128xbf16>""",
        """%e2 = tensor.empty() : tensor<2048x128xbf16>
    %v = linalg.copy ins(%e2 : tensor<2048x128xbf16>) outs(%et : tensor<2048x128xbf16>) -> tensor<2048x128xbf16>""",
    ],
    ids=[
        "loop-init",
        "reshaped-empty",
        "reduction",
        "not-permutation",
        "reads-init",
        "identity-of-empty",
        "identity-of-half-empty",
        "copy-of-empty",
    ],
)
def test_mark_overwritten_args_needs_a_fully_written_tile(tile: str) -> None:
    source = _tile_body(tile, "%v").replace(
        "%b: i1,", "%b: i1, %t3: tensor<2048x128x4xbf16>, %t1: tensor<2048xbf16>,"
    )
    assert not _marked(source)[0]


@pytest.mark.parametrize(
    "tile",
    [
        # A permuted output map (loops over columns, then rows).
        """%v2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%t0 : tensor<2048x128xbf16>) outs(%et : tensor<2048x128xbf16>) {
    ^bb0(%in: bf16, %init: bf16):
      linalg.yield %in : bf16
    } -> tensor<2048x128xbf16>""",
        """%z = arith.constant 0.0 : bf16
    %v2 = linalg.fill ins(%z : bf16) outs(%et : tensor<2048x128xbf16>) -> tensor<2048x128xbf16>""",
        """%v2 = linalg.copy ins(%t0 : tensor<2048x128xbf16>) outs(%et : tensor<2048x128xbf16>) -> tensor<2048x128xbf16>""",
        # As the GEMM's epilogue: a contraction into a zeroed accumulator, then the
        # bf16 rounding (a generic) into an empty tile.
        """%zf = arith.constant 0.0 : f32
    %ea = tensor.empty() : tensor<2048x128xf32>
    %acc = linalg.fill ins(%zf : f32) outs(%ea : tensor<2048x128xf32>) -> tensor<2048x128xf32>
    %mm = linalg.matmul ins(%ka, %kb : tensor<2048x4096xbf16>, tensor<4096x128xbf16>) outs(%acc : tensor<2048x128xf32>) -> tensor<2048x128xf32>
    %v2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%mm : tensor<2048x128xf32>) outs(%et : tensor<2048x128xbf16>) {
    ^bb0(%in: f32, %init: bf16):
      %rounded = arith.truncf %in : f32 to bf16
      linalg.yield %rounded : bf16
    } -> tensor<2048x128xbf16>""",
        # A cast of a written tile (as fold_full_insert_slices leaves a full
        # tile's static type cast to the ragged one).
        """%d = tensor.cast %t : tensor<2048x128xbf16> to tensor<?x128xbf16>
    %v2 = tensor.cast %d : tensor<?x128xbf16> to tensor<2048x128xbf16>""",
    ],
    ids=["permuted-generic", "fill", "copy", "matmul-epilogue", "cast"],
)
def test_fully_written_tiles(tile: str) -> None:
    # Fully written (``_fully_written``); marked only if a long contraction
    # produces them (the matmul epilogue; the others: no contraction).
    source = _tile_body(tile, "%v2")
    assert _insert_source(source, _fully_written)
    assert _marked(source)[0] is ("linalg.matmul" in tile or "tensor.cast" in tile)


def _insert_source(source: str, check: Callable[[ir.Value], object]) -> object:
    """``check`` of the source of the (first) ``parallel_insert_slice`` in ``source``."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        found: list[object] = []

        def visit(op: ir.Operation) -> ir.WalkResult:
            if op.name == "tensor.parallel_insert_slice" and not found:
                found.append(check(op.operands[0]))
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
        return found[0]


@pytest.mark.parametrize(
    ("k", "marked"),
    [(MIN_REDUCTION, True), (MIN_REDUCTION - 32, False), (64, False), ("?", False)],
    ids=["at-threshold", "below", "small-k", "dynamic-k"],
)
def test_mark_overwritten_args_needs_a_long_contraction(
    k: int | str, marked: bool
) -> None:
    # A short-K GEMM is memory-bound: the prologue costs more than it saves.
    # A runtime K is not known to be long.
    assert _marked(_forall(k=k))[0] is marked


K_LOOP = """
func.func @f(%a: tensor<2048x2112xbf16>, %b: tensor<2112x128xbf16>, %n: index) -> (tensor<2048x128xf32>, tensor<2048x128xf32>) {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c2048 = arith.constant 2048 : index
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2048x128xf32>
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<2048x128xf32>) -> tensor<2048x128xf32>
  %loop = scf.for %k = %c0 to %c2048 step %c512 iter_args(%acc = %f) -> (tensor<2048x128xf32>) {
    %as = tensor.extract_slice %a[0, %k] [2048, 512] [1, 1] : tensor<2048x2112xbf16> to tensor<2048x512xbf16>
    %bs = tensor.extract_slice %b[%k, 0] [512, 128] [1, 1] : tensor<2112x128xbf16> to tensor<512x128xbf16>
    %d = linalg.matmul ins(%as, %bs : tensor<2048x512xbf16>, tensor<512x128xbf16>) outs(%acc : tensor<2048x128xf32>) -> tensor<2048x128xf32>
    scf.yield %d : tensor<2048x128xf32>
  }
  %at = tensor.extract_slice %a[0, 2048] [2048, 64] [1, 1] : tensor<2048x2112xbf16> to tensor<2048x64xbf16>
  %bt = tensor.extract_slice %b[2048, 0] [64, 128] [1, 1] : tensor<2112x128xbf16> to tensor<64x128xbf16>
  %tail = linalg.matmul ins(%at, %bt : tensor<2048x64xbf16>, tensor<64x128xbf16>) outs(%loop : tensor<2048x128xf32>) -> tensor<2048x128xf32>
  %dyn = scf.for %k = %c0 to %n step %c512 iter_args(%acc = %f) -> (tensor<2048x128xf32>) {
    %as = tensor.extract_slice %a[0, %k] [2048, 512] [1, 1] : tensor<2048x2112xbf16> to tensor<2048x512xbf16>
    %bs = tensor.extract_slice %b[%k, 0] [512, 128] [1, 1] : tensor<2112x128xbf16> to tensor<512x128xbf16>
    %d = linalg.matmul ins(%as, %bs : tensor<2048x512xbf16>, tensor<512x128xbf16>) outs(%acc : tensor<2048x128xf32>) -> tensor<2048x128xf32>
    scf.yield %d : tensor<2048x128xf32>
  }
  return %tail, %dyn : tensor<2048x128xf32>, tensor<2048x128xf32>
}
"""


def test_reduction_length_of_k_loops() -> None:
    # 4 trips of K = 512 plus a tail of 64; a runtime trip count: unknown.
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(K_LOOP)
        ret = module.body.operations[0].regions[0].blocks[0].operations
        results = list(ret[len(ret) - 1].operands)
        assert _reduction_length(results[0]) == 4 * 512 + 64
        assert _reduction_length(results[1]) is None


K_HEADER = """
func.func @f(%a: tensor<128x512xbf16>, %b: tensor<512x128xbf16>, %x: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x128xf32>
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
"""


def _mm(init: str) -> str:
    """A K = 512 matmul accumulating into ``init``."""
    return (
        "linalg.matmul ins(%a, %b : tensor<128x512xbf16>, tensor<512x128xbf16>)"
        f" outs({init} : tensor<128x128xf32>) -> tensor<128x128xf32>"
    )


K_CASES = {
    # Accumulating: 8 trips of 512.
    "loop": (
        f"""%l = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %f) -> (tensor<128x128xf32>) {{
    %d = {_mm("%acc")}
    scf.yield %d : tensor<128x128xf32>
  }}""",
        8 * 512,
    ),
    # A fresh accumulator each trip, the carried value unused: one trip's K.
    "fresh-each-trip": (
        f"""%l = scf.for %k = %c0 to %c64 step %c1 iter_args(%acc = %f) -> (tensor<128x128xf32>) {{
    %ff = linalg.fill ins(%z : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
    %d = {_mm("%ff")}
    scf.yield %d : tensor<128x128xf32>
  }}""",
        512,
    ),
    # Nested accumulating loops: 4 x 8 trips.
    "nested": (
        f"""%l = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %f) -> (tensor<128x128xf32>) {{
    %in = scf.for %j = %c0 to %c8 step %c1 iter_args(%acc2 = %acc) -> (tensor<128x128xf32>) {{
      %d = {_mm("%acc2")}
      scf.yield %d : tensor<128x128xf32>
    }}
    scf.yield %in : tensor<128x128xf32>
  }}""",
        4 * 8 * 512,
    ),
    # The inner accumulator reset each outer trip: one outer trip's K.
    "inner-reset": (
        f"""%l = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %f) -> (tensor<128x128xf32>) {{
    %in = scf.for %j = %c0 to %c8 step %c1 iter_args(%acc2 = %f) -> (tensor<128x128xf32>) {{
      %d = {_mm("%acc2")}
      scf.yield %d : tensor<128x128xf32>
    }}
    scf.yield %in : tensor<128x128xf32>
  }}""",
        8 * 512,
    ),
    # No trips: the init's K, not a negative count.
    "zero-trips": (
        f"""%l = scf.for %k = %c64 to %c0 step %c1 iter_args(%acc = %f) -> (tensor<128x128xf32>) {{
    %d = {_mm("%acc")}
    scf.yield %d : tensor<128x128xf32>
  }}""",
        0,
    ),
    # Accumulating into data (a bias): its K, the init counting 0.
    "data-init": (f"%l = {_mm('%x')}", 512),
    # A matmul whose result the epilogue discards: 0 (it forwards data).
    "discarded": (
        f"""%d = {_mm("%f")}
  %l = linalg.generic {{indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]}} ins(%d, %x : tensor<128x128xf32>, tensor<128x128xf32>) outs(%e : tensor<128x128xf32>) {{
  ^bb0(%p: f32, %q: f32, %o: f32):
    linalg.yield %q : f32
  }} -> tensor<128x128xf32>""",
        0,
    ),
}


@pytest.mark.parametrize("case", list(K_CASES))
def test_reduction_length_counts(case: str) -> None:
    body, expected = K_CASES[case]
    source = K_HEADER + f"  {body}\n  return %l : tensor<128x128xf32>\n}}\n"
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        ops = module.body.operations[0].regions[0].blocks[0].operations
        assert _reduction_length(ops[len(ops) - 1].operands[0]) == expected


def _where_chain(levels: int) -> tuple[str, str]:
    """A 3-value epilogue of ``levels`` steps, each value a select of the three
    previous ones (as a chain of ``torch.where``): every path through it shares
    producers, 3^levels paths in all."""
    lines, prev = [], ["%t", "%t", "%t"]
    piece = "tensor<2048x128xbf16>"
    maps = ", ".join(["affine_map<(d0, d1) -> (d0, d1)>"] * 4)
    for level in range(levels):
        cur = []
        for j in range(3):
            name = f"%w{level}_{j}"
            ins = ", ".join(prev)
            lines.append(
                f"""{name} = linalg.generic {{indexing_maps = [{maps}], iterator_types = ["parallel", "parallel"]}} ins({ins} : {piece}, {piece}, {piece}) outs(%et : {piece}) {{
    ^bb0(%in0: bf16, %in1: bf16, %in2: bf16, %init: bf16):
      %gt = arith.cmpf ogt, %in2, %in0 : bf16
      %sel = arith.select %gt, %in0, %in1 : bf16
      linalg.yield %sel : bf16
    }} -> {piece}"""
            )
            cur.append(name)
        prev = cur
    return "\n    ".join(lines), prev[0]


def test_mark_overwritten_args_analyses_deep_epilogues_quickly() -> None:
    # Regression: the analysis visited every path through a 12-step where chain
    # (3^12 paths, past the compile timeout) and, capped,
    # left the GEMM unmarked. The chain has 36 ops: visits stay linear in them.
    tile, last = _where_chain(12)
    source = _tile_body(tile, last)
    # _ReductionLength is what walks the chain; _fully_written stops at its first op here.
    with visits(first_touch._ReductionLength, "_length", limit=100) as counts:
        assert _marked(source)[0]
    assert counts["_length"] >= 36


def test_mark_overwritten_args_keeps_reduction_tiles() -> None:
    # The bf16 rounding of a row-wise sum (as a norm's): fully written, but no
    # contraction produces it.
    tile = """%zr = arith.constant 0.0 : f32
    %er = tensor.empty() : tensor<2048x128xf32>
    %fr = linalg.fill ins(%zr : f32) outs(%er : tensor<2048x128xf32>) -> tensor<2048x128xf32>
    %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%t3 : tensor<2048x128x4096xf32>) outs(%fr : tensor<2048x128xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    } -> tensor<2048x128xf32>
    %v = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%sum : tensor<2048x128xf32>) outs(%et : tensor<2048x128xbf16>) {
    ^bb0(%in: f32, %init: bf16):
      %rounded = arith.truncf %in : f32 to bf16
      linalg.yield %rounded : bf16
    } -> tensor<2048x128xbf16>"""
    source = _tile_body(tile, "%v").replace(
        "%b: i1,", "%b: i1, %t3: tensor<2048x128x4096xf32>,"
    )
    assert _insert_source(source, _fully_written)
    assert not _marked(source)[0]


def test_mark_overwritten_args_marks_tiles_computed_into_empty() -> None:
    # As the GEMM's epilogue: a linalg op writes all of an empty tile.
    source = _tile_body(
        """%e = tensor.empty() : tensor<2048x128xbf16>
    %v = linalg.copy ins(%t : tensor<2048x128xbf16>) outs(%e : tensor<2048x128xbf16>) -> tensor<2048x128xbf16>""",
        "%v",
    )
    assert _marked(source)[0]


def test_mark_overwritten_args_keeps_ragged_versioning() -> None:
    # As 8_Matmul_with_irregular_shapes_: each version writes all of the old tile.
    source = _tile_body(
        """%v = scf.if %b -> (tensor<2048x128xbf16>) {
      %w = tensor.insert_slice %t into %old[0, 0] [2048, 128] [1, 1] : tensor<2048x128xbf16> into tensor<2048x128xbf16>
      scf.yield %w : tensor<2048x128xbf16>
    } else {
      %w = tensor.insert_slice %u into %old[0, 0] [2048, 128] [1, 1] : tensor<2048x128xbf16> into tensor<2048x128xbf16>
      scf.yield %w : tensor<2048x128xbf16>
    }""",
        "%v",
    )
    assert _marked(source)[0]


@pytest.mark.parametrize("elem", ["complex<f32>", "index"])
def test_mark_overwritten_args_skips_elements_without_width(elem: str) -> None:
    assert not _marked(_tile_body("", "%t", elem=elem))[0]


@pytest.mark.parametrize(
    ("tiles", "shape", "width", "expected"),
    [
        ([127, 128], [4096, 4160], 4, True),  # spans exactly 2 MiB
        ([127, 127], [4096, 4160], 4, False),  # 4 bytes short of 2 MiB
        ([2048, 4095], [4096, 4096], 2, True),  # one element short of the row
        ([2048, 4096], [4096, 4096], 2, False),  # whole rows: contiguous
        ([1, 128], [4096, 4096], 2, False),  # one row: contiguous
        ([4, 2048, 128], [4, 4096, 4096], 2, True),  # a batch of tall tiles
    ],
)
def test_scattered_gate_edges(
    tiles: list[int], shape: list[int], width: int, expected: bool
) -> None:
    assert _scattered(tiles, shape, width) is expected


def test_mark_overwritten_args_marks_covered_output_not_first() -> None:
    # The covered output is the second shared output.
    source = (
        """
func.func @f(%arg0: memref<4096x4096xbf16>, %arg1: memref<4096x4096xbf16>, %ka: tensor<2048x4096xbf16>, %kb: tensor<4096x128xbf16>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<4096x4096xbf16> to tensor<4096x4096xbf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<4096x4096xbf16> to tensor<4096x4096xbf16>
  %2:2 = scf.forall (%i, %j) in (2, 32) shared_outs(%p = %1, %o = %0) -> (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) {
    %r = affine.apply affine_map<(d0) -> (d0 * 2048)>(%i)
    %c = affine.apply affine_map<(d0) -> (d0 * 128)>(%j)
    """
        + _gemm("tw", 2048, 128)
        + """
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tw into %p[%r, 0] [2048, 128] [1, 1] : tensor<2048x128xbf16> into tensor<4096x4096xbf16>
      tensor.parallel_insert_slice %tw into %o[%r, %c] [2048, 128] [1, 1] : tensor<2048x128xbf16> into tensor<4096x4096xbf16>
    }
  }
  bufferization.materialize_in_destination %2#0 in restrict writable %arg1 : (tensor<4096x4096xbf16>, memref<4096x4096xbf16>) -> ()
  bufferization.materialize_in_destination %2#1 in restrict writable %arg0 : (tensor<4096x4096xbf16>, memref<4096x4096xbf16>) -> ()
  return
}
"""
    )
    assert _marked(source)[:2] == [True, False]
    # Crossed: each result materialized into the other argument.
    crossed = source.replace(
        "%2#0 in restrict writable %arg1", "%2#0 in restrict writable %argX"
    )
    crossed = crossed.replace(
        "%2#1 in restrict writable %arg0", "%2#1 in restrict writable %arg1"
    )
    crossed = crossed.replace("%argX", "%arg0")
    assert _marked(crossed)[:2] == [False, False]


def _parallel(
    arg_type: str = "memref<4096x4096xbf16>",
    attrs: str = "{helion.overwritten}",
    before: str = "",
) -> str:
    """An OpenMP loop storing into ``%arg0`` (as after ``omp.py``)."""
    return f"""
func.func @f(%arg0: {arg_type} {attrs}, %v: bf16) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  {before}
  omp.parallel {{
    omp.wsloop {{
      omp.loop_nest (%i) : index = (%c0) to (%c64) step (%c1) {{
        memref.store %v, %arg0[%i, %c0] : {arg_type}
        omp.yield
      }}
    }}
    omp.terminator
  }}
  return
}}
"""


def test_first_touch_outputs_populates_pages_first() -> None:
    text = _apply(first_touch_outputs, _parallel())
    assert MARK not in text
    # A static worksharing loop without a barrier over the 2 MiB-aligned ranges
    # before the loop, one madvise per range with MADV_POPULATE_WRITE (23).
    assert text.count("omp.wsloop nowait schedule(static)") == 1
    assert text.count("omp.wsloop") == 2
    assert "omp.barrier" not in text
    assert text.count(TOUCH) == 1
    assert text.index(TOUCH) < text.index("memref.store")
    # No OpenMP API calls: with torch's libgomp loaded first, they bind to it.
    assert "omp_get" not in text
    assert "arith.constant 23 : i32" in text
    assert "arith.constant -2097152 : index" in text
    assert "arith.constant 2097152 : index" in text
    assert re.search(r"llvm\.func @madvise\(!llvm\.ptr, i64, i32\) -> i32", text)
    # The result is ignored (the call is a hint).
    result = re.search(r"(%\S+) = llvm\.call @madvise", text).group(1)
    assert text.count(result) == 1
    assert "memref.extract_aligned_pointer_as_index %arg0" in text
    assert "arith.constant -4096 : index" in text
    # Its 32 MiB: the end of the range.
    assert "arith.constant 33554432 : index" in text


@pytest.mark.parametrize(
    ("arg_type", "attrs", "before"),
    [
        # Not marked by mark_overwritten_args.
        ("memref<4096x4096xbf16>", '{helion.role = "inout"}', ""),
        # 8 MiB: below the default min_bytes (16 MiB).
        ("memref<2048x2048xbf16>", "{helion.overwritten}", ""),
        # Not contiguous: the touched range would include bytes not in the view.
        ("memref<4096x4096xbf16, strided<[8192, 1]>>", "{helion.overwritten}", ""),
        # Runtime shape.
        ("memref<?x4096xbf16>", "{helion.overwritten}", ""),
        # First written outside the parallel region.
        (
            "memref<4096x4096xbf16>",
            "{helion.overwritten}",
            "memref.store %v, %arg0[%c0, %c0] : memref<4096x4096xbf16>",
        ),
    ],
)
def test_first_touch_outputs_leaves_other_arguments(
    arg_type: str, attrs: str, before: str
) -> None:
    source = _parallel(arg_type, attrs, before)
    text = _apply(first_touch_outputs, source)
    assert TOUCH not in text
    assert text.count("omp.wsloop") == 1
    # The marks are dropped: the function lowers as if never marked.
    assert MARK not in text


def test_first_touch_outputs_follows_views() -> None:
    # A view of the output made outside the region is the same memory.
    source = """
func.func @f(%arg0: memref<4096x4096xbf16> {helion.overwritten}, %v: bf16) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %flat = memref.collapse_shape %arg0 [[0, 1]] : memref<4096x4096xbf16> into memref<16777216xbf16>
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c64) step (%c1) {
        memref.store %v, %flat[%i] : memref<16777216xbf16>
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
"""
    assert _apply(first_touch_outputs, source).count(TOUCH) == 1


def test_first_touch_outputs_populates_each_output() -> None:
    # Two outputs first written in one region: one call each, declared once.
    source = """
func.func @f(%arg0: memref<4096x4096xbf16> {helion.overwritten},
             %arg1: memref<4096x4096xbf16> {helion.overwritten}, %v: bf16) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c64) step (%c1) {
        memref.store %v, %arg0[%i, %c0] : memref<4096x4096xbf16>
        memref.store %v, %arg1[%i, %c0] : memref<4096x4096xbf16>
        omp.yield
      }
    }
    omp.terminator
  }
  return
}
"""
    text = _apply(first_touch_outputs, source)
    assert text.count(TOUCH) == 2
    assert text.count("llvm.func @madvise") == 1
    assert text.count("omp.wsloop nowait schedule(static)") == 2
    assert "omp.barrier" not in text
    assert text.rindex(TOUCH) < text.index("memref.store")


@pytest.mark.parametrize(
    ("arg_type", "min_bytes", "touched"),
    [
        ("memref<16777215xi8>", None, False),  # 16 MiB - 1
        ("memref<16777216xi8>", None, True),  # 16 MiB
        ("memref<0xi8>", 0, False),  # nothing to touch
    ],
)
def test_first_touch_outputs_size_edges(
    arg_type: str, min_bytes: int | None, touched: bool
) -> None:
    source = f"""
func.func @f(%arg0: {arg_type} {{helion.overwritten}}, %v: i8) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  omp.parallel {{
    omp.wsloop {{
      omp.loop_nest (%i) : index = (%c0) to (%c1) step (%c1) {{
        memref.store %v, %arg0[%i] : {arg_type}
        omp.yield
      }}
    }}
    omp.terminator
  }}
  return
}}
"""
    schedule = (
        first_touch_outputs if min_bytes is None else lambda: first_touch_outputs(0)
    )
    assert (TOUCH in _apply(schedule, source)) is touched


def _regions(
    first: str, second: str, before: str = "", second_arg: str = "%arg0"
) -> str:
    """Two OpenMP regions: the first runs ``first`` on ``%arg0``, the second
    ``second`` on ``second_arg``; ``before`` runs at the top level first."""
    t = "memref<4096x4096xbf16>"
    return f"""
func.func @f(%arg0: {t} {{helion.overwritten}}, %arg1: {t} {{helion.overwritten}}, %v: bf16) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  {before}
  omp.parallel {{
    omp.wsloop {{
      omp.loop_nest (%i) : index = (%c0) to (%c64) step (%c1) {{
        {first}
        omp.yield
      }}
    }}
    omp.terminator
  }}
  omp.parallel {{
    omp.wsloop {{
      omp.loop_nest (%i) : index = (%c0) to (%c64) step (%c1) {{
        {second.replace("ARG", second_arg)}
        omp.yield
      }}
    }}
    omp.terminator
  }}
  return
}}
"""


STORE = "memref.store %v, %arg0[%i, %c0] : memref<4096x4096xbf16>"
LOAD = "%l = memref.load %arg0[%i, %c0] : memref<4096x4096xbf16>\n        " + STORE
SECOND = "memref.store %v, ARG[%i, %c1] : memref<4096x4096xbf16>"


def test_first_touch_outputs_touches_before_a_load() -> None:
    text = _apply(first_touch_outputs, _regions(LOAD, "", second_arg="%arg1"))
    assert text.count(TOUCH) == 1
    assert text.index(TOUCH) < text.index("memref.load")


def test_first_touch_outputs_touches_each_output_in_its_region() -> None:
    text = _apply(first_touch_outputs, _regions(STORE, SECOND, second_arg="%arg1"))
    assert text.count(TOUCH) == 2
    assert "omp.barrier" not in text
    second = text.index("omp.parallel", text.index("omp.parallel") + 1)
    assert text.index(TOUCH) < second < text.rindex(TOUCH)


def test_first_touch_outputs_touches_only_the_first_region() -> None:
    # %arg0 written in both regions (and %arg1 nowhere): one touch, in the first.
    text = _apply(first_touch_outputs, _regions(STORE, SECOND))
    second = text.index("omp.parallel", text.index("omp.parallel") + 1)
    assert text.count(TOUCH) == 1
    assert text.index(TOUCH) < second


@pytest.mark.parametrize(
    "before",
    [
        (
            "%s = memref.memory_space_cast %arg0 : memref<4096x4096xbf16> to"
            " memref<4096x4096xbf16, 1>"
        ),
        (
            "%b, %o, %sz:2, %st:2 = memref.extract_strided_metadata %arg0 :"
            " memref<4096x4096xbf16> -> memref<bf16>, index, index, index, index, index"
        ),
        (
            "%s = memref.cast %arg0 : memref<4096x4096xbf16> to memref<?x4096xbf16>\n"
            "  memref.store %v, %s[%c0, %c0] : memref<?x4096xbf16>"
        ),
    ],
    ids=["memory-space-cast", "strided-metadata", "write-through-cast"],
)
def test_first_touch_outputs_leaves_outputs_used_before(before: str) -> None:
    source = _regions(STORE, "", before=before)
    assert TOUCH not in _apply(first_touch_outputs, source)


def _scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 3.0
    return out


def _gemm_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
        # A fused epilogue (an all-parallel op over the K loop's result), as the
        # AI-bench GEMMs' bf16 rounding: a bare loop result is not on the allowlist.
        out[tm, tn] = torch.relu(acc)
    return out


# Tall partial-width tiles (as the GEMM's): each spans 8 MiB of rows, 256 bytes
# of each; whole-row tiles (as an elementwise kernel's): each contiguous.
tall_tiles_kernel = helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[1024, 64])
)(_scale)
row_tiles_kernel = helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[8, 2048])
)(_scale)
tall_gemm_kernel = helion.kernel(
    backend="mlir",
    static_shapes=True,
    config=helion.Config(block_sizes=[1024, 64, 256]),
)(_gemm_kernel)


@pytest.mark.isolated
def test_long_k_gemm_output_is_touched_and_correct() -> None:
    # 16.4 MiB of f32 (ragged: 2100 rows in tiles of 1024), K = 2048 in 8 chunks:
    # touched, the same result as without the prologue (an exact f32 check
    # against the scalar pipeline is not available: compare to torch loosely).
    torch.manual_seed(0)
    x, y = torch.randn(2100, 2048), torch.randn(2048, 2048)
    with opt_pipeline():
        text = str(lower_mlir(tall_gemm_kernel, [x, y]))
        assert text.count(TOUCH) == 1
        torch.testing.assert_close(
            tall_gemm_kernel(x, y), torch.relu(x @ y), rtol=1e-4, atol=1e-3
        )


# Lowers a transform-level module with an OpenMP region to the LLVM dialect.
_TO_LLVM = (
    "builtin.module(expand-strided-metadata, finalize-memref-to-llvm, "
    "convert-scf-to-cf, convert-openmp-to-llvm, convert-arith-to-llvm, "
    "convert-index-to-llvm, convert-cf-to-llvm, convert-func-to-llvm, "
    "reconcile-unrealized-casts)"
)
_MADV_NOHUGEPAGE = 15


@pytest.mark.isolated
def test_first_touch_outputs_populates_exactly_the_output() -> None:
    """The populated range is the output's pages, no more and no fewer: a region
    that only reads the output's first element, run on an output 64 bytes into
    the second page of a fresh mapping with one spare page on each side (no huge
    pages, which would populate 2 MiB at once); ``mincore`` then sees every page
    holding output bytes resident and the spare pages not."""
    rows, cols, page = 1024, 1024, 4096  # 4 MiB of f32, ending 64 bytes into a page
    source = f"""
func.func @f(%arg0: memref<{rows}x{cols}xf32> {{helion.overwritten}}) attributes {{llvm.emit_c_interface}} {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  omp.parallel {{
    omp.wsloop {{
      omp.loop_nest (%i) : index = (%c0) to (%c64) step (%c1) {{
        %v = memref.load %arg0[%c0, %c0] : memref<{rows}x{cols}xf32>
        omp.yield
      }}
    }}
    omp.terminator
  }}
  return
}}
"""
    libc = ctypes.CDLL(None, use_errno=True)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        schedule = first_touch_outputs(0)  # kept alive: its ops' parent
        schedule.body.operations[0].apply(module.operation)
        assert TOUCH in str(module)
        passmanager.PassManager.parse(_TO_LLVM).run(module.operation)
        engine = ExecutionEngine(
            module, opt_level=2, shared_libs=[find_openmp_library()]
        )
        size = rows * cols * 4
        offset = page + 64
        length = (offset + size + page - 1) // page * page + page
        buffer = mmap.mmap(-1, length, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        assert (
            libc.madvise(
                ctypes.c_void_p(address), ctypes.c_size_t(length), _MADV_NOHUGEPAGE
            )
            == 0
        )
        output = np.frombuffer(
            buffer, dtype=np.float32, count=rows * cols, offset=offset
        ).reshape(rows, cols)
        flags = (ctypes.c_ubyte * (length // page))()

        def resident() -> list[int]:
            assert (
                libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(length), flags)
                == 0
            )
            return [flag & 1 for flag in flags]

        assert not any(resident())
        argument = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(output)))
        engine.invoke("f", argument)
        pages = resident()
        first, last = offset // page, (offset + size - 1) // page
        assert all(pages[first : last + 1])
        assert not pages[first - 1] and not pages[last + 1]
        del output, argument


@pytest.mark.isolated
def test_failing_madvise_is_ignored() -> None:
    # The fallback path: an advice the kernel rejects (EINVAL, as Linux before
    # 5.14 does for MADV_POPULATE_WRITE) leaves the kernel correct.
    import helion_mlir_backend._compiler.mlir_transforms as ht

    original = ht.first_touch_outputs

    def invalid_advice(target: ir.Value, min_bytes: int, advice: int) -> object:
        return original(target, min_bytes, 1 << 20)

    torch.manual_seed(0)
    x, y = torch.randn(2100, 2048), torch.randn(2048, 2048)
    with opt_pipeline(), patch.object(ht, "first_touch_outputs", invalid_advice):
        text = str(lower_mlir(tall_gemm_kernel, [x, y]))
        assert text.count(TOUCH) == 1
        assert "1048576 : i32" in text
        torch.testing.assert_close(
            tall_gemm_kernel(x, y), torch.relu(x @ y), rtol=1e-4, atol=1e-3
        )


@pytest.mark.isolated
def test_elementwise_row_tiles_and_small_outputs_are_not_touched() -> None:
    # Tall-tile elementwise, row-tile elementwise and a small output: not touched.
    with opt_pipeline():
        x = torch.randn(2100, 2048)
        assert TOUCH not in str(lower_mlir(tall_tiles_kernel, [x]))
        assert TOUCH not in str(lower_mlir(row_tiles_kernel, [x]))
        small = [torch.randn(64, 1024), torch.randn(1024, 128)]
        assert TOUCH not in str(lower_mlir(tall_gemm_kernel, small))


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[1024, 64])
)
def cond_store_kernel(
    x: torch.Tensor, flag: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tm, tn in hl.tile(x.size()):
        if flag[0] > 0:
            out[tm, tn] = x[tm, tn] * 3.0
    return out


@pytest.mark.isolated
def test_conditional_store_leaves_unwritten_output_untouched() -> None:
    """Regression: a conditional store was marked as a whole overwrite, and with
    the flag off the prologue wrote every page of a read-only output (SIGSEGV)."""
    m, n = 4096, 2048  # 32 MiB of f32
    x = torch.randn(m, n)
    with opt_pipeline():
        assert TOUCH not in str(
            lower_mlir(cond_store_kernel, [x, torch.zeros(1), torch.empty(m, n)])
        )
        buffer = mmap.mmap(
            -1,
            m * n * 4,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            prot=mmap.PROT_READ,
        )
        out = torch.frombuffer(buffer, dtype=torch.float32).view(m, n)
        result = cond_store_kernel(x, torch.zeros(1), out)
        assert bool((result == 0).all())
