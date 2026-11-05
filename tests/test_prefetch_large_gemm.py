"""``prefetch_large_gemm``: the output lines of the register blocks of a short-K AMX
GEMM, of whole-row tiles and an output beyond the last-level cache, prefetched for
writing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl
from lighthouse.execution.target import TargetInfo
from mlir import ir
import pytest
import torch

from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.helion_transforms import prefetch_large_gemm
from helion_mlir_backend._compiler.mlir_transforms.prefetch_large_gemm import MAX_K
from helion_mlir_backend._compiler.mlir_transforms.prefetch_large_gemm import MIN_BYTES

if TYPE_CHECKING:
    from collections.abc import Callable


amx_host_only = pytest.mark.skipif(
    not TargetInfo.host().is_supported("amx"), reason="executes AMX code"
)


def _apply(make_schedule: Callable[[], ir.Module], source: str) -> str:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        schedule = make_schedule()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


_ROLE_OUT = '{helion.role = "inout"}'
_ROLE_IN = '{helion.role = "in"}'
# A row loop of 32-row blocks, a column loop, the K loop, then whole lines.
_BLOCK = """
func.func @f(%out: OUT_TYPE OUT_ROLE, %a: A_TYPE A_ROLE, %b: memref<128x32xbf16>,
             %v: VALUE_TYPE, %acc: !x86.amx.tile<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %KEND = arith.constant KVAL : index
  %c128 = arith.constant 128 : index
  %step = STEP
  %tile = memref.subview %out[%c0, 0] [TILE_ROWS, TILE_COLS] [1, 1] :
      OUT_TYPE to memref<TILE_ROWSxTILE_COLSxbf16, strided<[STRIDE, 1], offset: ?>>
  scf.for %i = %c0 to %c128 step %step {
    %r = arith.addi %i, %c32 : index
    %av = memref.subview %a[%r, 0] [32, 256] [1, 1] :
        A_TYPE to memref<32x256xbf16, strided<[256, 1], offset: ?>>
    scf.for %j = %c0 to %c128 step %c32 {
      %t = scf.for %k = %c0 to %KEND step %c16 iter_args(%x = %acc)
          -> (!x86.amx.tile<16x16xf32>) {
        %l = x86.amx.tile_load %av[%c0, %k] :
            memref<32x256xbf16, strided<[256, 1], offset: ?>>
            into !x86.amx.tile<16x32xbf16>
        %m = x86.amx.tile_load %b[%k, %c0] : memref<128x32xbf16>
            into !x86.amx.tile<16x32xbf16>
        %y = MULF
        scf.yield %y : !x86.amx.tile<16x16xf32>
      }
      %w = VIEW
      vector.transfer_write %v, %w[%c0, %c0] {in_bounds = [true]} :
          VALUE_TYPE, memref<32x32xbf16, strided<[STRIDE, 1], offset: ?>>
      vector.transfer_write %v, %w[%c1, %c0] {in_bounds = [true]} :
          VALUE_TYPE, memref<32x32xbf16, strided<[STRIDE, 1], offset: ?>>
    }
  }
  return
}
"""
_MULF = (
    "x86.amx.tile_mulf %l, %m, %x : !x86.amx.tile<16x32xbf16>, "
    "!x86.amx.tile<16x32xbf16>, !x86.amx.tile<16x16xf32>"
)


def _source(**changes: str) -> str:
    values = {
        "OUT_TYPE": "memref<524288x1024xbf16>",
        "OUT_ROLE": _ROLE_OUT,
        "A_TYPE": "memref<16384x256xbf16>",
        "A_ROLE": _ROLE_IN,
        "VALUE_TYPE": "vector<32xbf16>",
        "STEP": "arith.constant 32 : index",
        "MULF": _MULF,
        "STRIDE": "1024",
        "KEND": "kend",
        "KVAL": "128",
        "TILE_ROWS": "256",
        "TILE_COLS": "1024",
        "VIEW": (
            "memref.subview %tile[%i, %j] [32, 32] [1, 1] : "
            "memref<256xTILE_COLSxbf16, strided<[STRIDE, 1], offset: ?>> to "
            "memref<32x32xbf16, strided<[STRIDE, 1], offset: ?>>"
        ),
    }
    values.update(changes)
    source = _BLOCK.replace("VIEW", values.pop("VIEW"))
    for key, value in values.items():
        source = source.replace(key, value)
    return source


def test_block_prefetches_its_lines() -> None:
    assert MIN_BYTES <= 524288 * 1024 * 2
    text = _apply(prefetch_large_gemm, _source())
    assert text.count(", write, locality<3>, data") == 2
    assert ", read," not in text
    # Before the K loop that computes them, once each.
    assert text.index(", write,") < text.index("x86.amx.tile_mulf")


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param(
            {
                "OUT_TYPE": "memref<16384x1024xbf16>",
                "STRIDE": "1024",
                "TILE_COLS": "1024",
            },
            id="small",
        ),
        pytest.param(
            {
                "TILE_COLS": "512",
                "VIEW": (
                    "memref.subview %tile[%i, %j] [32, 32] [1, 1] : "
                    "memref<256x512xbf16, strided<[1024, 1], offset: ?>> to "
                    "memref<32x32xbf16, strided<[1024, 1], offset: ?>>"
                ),
            },
            id="narrow-tile",
        ),
        pytest.param(
            {
                "TILE_ROWS": "128",
                "VIEW": (
                    "memref.subview %tile[%i, %j] [32, 32] [1, 1] : "
                    "memref<128x1024xbf16, strided<[1024, 1], offset: ?>> to "
                    "memref<32x32xbf16, strided<[1024, 1], offset: ?>>"
                ),
            },
            id="few-rows",
        ),
        pytest.param(
            {
                "VIEW": (
                    "memref.subview %out[%i, %j] [32, 32] [1, 1] : "
                    "memref<524288x1024xbf16> to "
                    "memref<32x32xbf16, strided<[1024, 1], offset: ?>>"
                ),
            },
            id="no-tile",
        ),
        pytest.param({"OUT_ROLE": _ROLE_IN}, id="not-an-output"),
        pytest.param({"OUT_ROLE": ""}, id="no-role"),
        pytest.param({"VALUE_TYPE": "vector<16xbf16>"}, id="half-line"),
        # K over MAX_K: 9 steps of 32.
        pytest.param({"KVAL": "144"}, id="long-k"),
    ],
)
def test_other_output_writes_get_no_prefetch(changes: dict[str, str]) -> None:
    assert "memref.prefetch" not in _apply(prefetch_large_gemm, _source(**changes))


def test_few_columns_get_no_prefetch() -> None:
    """Whole rows of 128 columns: writes-only prefetches lose there (1.01-1.10)."""
    source = _source(OUT_TYPE="memref<4194304x128xbf16>", STRIDE="128", TILE_COLS="128")
    assert "memref.prefetch" not in _apply(prefetch_large_gemm, source)


def test_tile_accumulator_over_the_limit_gets_no_prefetch() -> None:
    """Whole rows with a 2 MiB accumulator lose (131072x2048x256, 8 threads: 1.35)."""
    source = _source(
        OUT_TYPE="memref<131072x2048xbf16>",
        STRIDE="2048",
        TILE_COLS="2048",
    )
    assert "memref.prefetch" not in _apply(prefetch_large_gemm, source)


def test_longest_k_gets_prefetches() -> None:
    assert MAX_K == 256
    text = _apply(prefetch_large_gemm, _source(KVAL="128"))
    assert text.count(", write,") == 2


_SECOND_K_LOOP = """      %t2 = scf.for %k = %c0 to %kend step %c16 iter_args(%x = %t)
          -> (!x86.amx.tile<16x16xf32>) {
        %l = x86.amx.tile_load %av[%c0, %k] :
            memref<32x256xbf16, strided<[256, 1], offset: ?>>
            into !x86.amx.tile<16x32xbf16>
        %m = x86.amx.tile_load %b[%k, %c0] : memref<128x32xbf16>
            into !x86.amx.tile<16x32xbf16>
        %y = x86.amx.tile_mulf %l, %m, %x : !x86.amx.tile<16x32xbf16>,
            !x86.amx.tile<16x32xbf16>, !x86.amx.tile<16x16xf32>
        scf.yield %y : !x86.amx.tile<16x16xf32>
      }
      %w = memref.subview"""


def test_two_k_loops_count_together_and_prefetch_once() -> None:
    def two_loops(pairs: str) -> str:
        return _source(KVAL=pairs).replace("      %w = memref.subview", _SECOND_K_LOOP)

    # 128 + 128 of K.
    assert _apply(prefetch_large_gemm, two_loops("64")).count(", write,") == 2
    # 256 + 256 of K, each within MAX_K alone.
    assert "memref.prefetch" not in _apply(prefetch_large_gemm, two_loops("128"))


@pytest.mark.parametrize(
    "bound",
    [
        pytest.param("%v_bound", id="dynamic-k"),
    ],
)
def test_dynamic_k_is_left_alone(bound: str) -> None:
    source = _source().replace("to %kend step %c16", f"to {bound} step %c16")
    source = source.replace(
        "%acc: !x86.amx.tile<16x16xf32>) {",
        f"%acc: !x86.amx.tile<16x16xf32>, {bound}: index) {{",
    )
    assert "memref.prefetch" not in _apply(prefetch_large_gemm, source)


def test_column_write_is_left_alone() -> None:
    """A vector written down a column touches 32 lines, not the line of its start."""
    source = _source().replace(
        "{in_bounds = [true]} :",
        "{in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} :",
    )
    assert "memref.prefetch" not in _apply(prefetch_large_gemm, source)


def test_memref_of_vectors_is_left_alone() -> None:
    source = f"""
func.func @f(%out: memref<8388608x8388608xvector<32xbf16>> {_ROLE_OUT},
             %v: vector<32xbf16>, %i: index) {{
  vector.transfer_write %v, %out[%i, %i] :
      vector<32xbf16>, memref<8388608x8388608xvector<32xbf16>>
  return
}}
"""
    assert "memref.prefetch" not in _apply(prefetch_large_gemm, source)


def test_kernel_without_a_contraction_is_left_alone() -> None:
    source = """
func.func @f(%out: memref<16384x16384xbf16> {helion.role = "inout"},
             %v: vector<32xbf16>, %c0: index, %c1: index) {
  scf.for %i = %c0 to %c1 step %c1 {
    vector.transfer_write %v, %out[%i, %c0] {in_bounds = [true]} :
        vector<32xbf16>, memref<16384x16384xbf16>
  }
  return
}
"""
    assert "memref.prefetch" not in _apply(prefetch_large_gemm, source)


def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 ``a @ b`` on AMX: ``b`` packed into VNNI pairs per tile."""
    m, k = a.shape
    n = b.size(1)
    k = hl.specialize(k)
    hl.specialize(b.size(0))
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a_tile = a[tile_m, tile_k]
            b_tile = b[tile_k, tile_n]
            a3 = a_tile.reshape(a_tile.size(0), a_tile.size(1) // 2, 2)
            b3 = b_tile.reshape(b_tile.size(0) // 2, 2, b_tile.size(1)).permute(0, 2, 1)
            acc = acc + torch.einsum("mcv,cnv->mn", a3, b3)
        out[tile_m, tile_n] = acc.to(a.dtype)
    return out


def _gemm_kernel(rows: int, columns: int, k: int) -> helion.Kernel:
    return helion.kernel(
        _gemm,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[rows, columns, k]),
    )


@pytest.mark.isolated
@amx_host_only
@pytest.mark.parametrize(("n", "fires"), [(1024, True), (992, True), (1040, False)])
def test_large_gemm_output_values(n: int, fires: bool) -> None:
    """At the size that turns the prefetches on, every element arrives. Powers of
    two: exact products in bf16. 1040 columns: a partial register block, no lines."""
    m = -(-MIN_BYTES // (n * 2))
    m = -(-m // 256) * 256
    k = 64
    kernel = _gemm_kernel(256, n, k)
    a = (2.0 ** (torch.arange(m) % 4)).reshape(m, 1).expand(m, k)
    a = a.to(torch.bfloat16).contiguous()
    b = (2.0 ** (torch.arange(n) % 3)).reshape(1, n).expand(k, n)
    b = b.to(torch.bfloat16).contiguous()
    assert ("llvm.intr.prefetch" in str(lower_mlir(kernel, [a, b]))) == fires
    out = kernel(a, b)
    expected = (a[:, 0].float() * k)[:, None] * b[0].float()[None, :]
    assert torch.equal(out, expected.to(torch.bfloat16))


def test_small_gemm_gets_no_prefetch() -> None:
    a = torch.ones(64, 64, dtype=torch.bfloat16)
    b = torch.ones(64, 64, dtype=torch.bfloat16)
    kernel = _gemm_kernel(32, 64, 64)
    assert "llvm.intr.prefetch" not in str(lower_mlir(kernel, [a, b]))


@pytest.mark.isolated
@amx_host_only
@pytest.mark.parametrize(
    ("m", "n", "k", "tiles"),
    [
        (524288, 1025, 256, (32, 32, 256)),
        (524289, 1024, 256, None),
        (16385, 32769, 64, None),
    ],
)
def test_helper_gemm_shapes_compile(
    m: int, n: int, k: int, tiles: tuple[int, int, int] | None
) -> None:
    """Regression: a view made outside the block was moved into it, and a use came
    before its operands' definitions (``does not dominate this use``)."""
    from helion_mlir_utils import matmul

    a = torch.zeros(m, k, dtype=torch.bfloat16)
    b = torch.zeros(k, n, dtype=torch.bfloat16)
    assert matmul(a, b, block_sizes=tiles).shape == (m, n)
