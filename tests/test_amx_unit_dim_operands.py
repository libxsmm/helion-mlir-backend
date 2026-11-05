"""AMX contraction operands read through views with unit dims.

A split-K GEMM (``_matmul_split_k``, below) contracts a
chunk ``a[tm, ts.begin, tk].unsqueeze(0)`` of a ``[M, S, Kc]`` view of A with
a 3-D ``hl.dot``: a ``linalg.batch_matmul`` of an ``expand_shape`` of the
``M x 1 x Kc`` slice. Register-tiled for AMX on the CPU, the LHS tiles are
read from the ``1 x M x 1 x Kc`` view with the map
``(d0, d1, d2, d3) -> (d1, d3)``, which skips the unit dim.

Upstream's AMX rewrite ignores the map: it loads the LHS tile with
``x86.amx.tile_load`` of the 4-D memref, whose row stride is the unit dim's
(``Kc``) instead of the rows' (``K``). At 64x1024 @ 1024x64, S = 8, blocks
``[1, 32, 32, 32]`` the result was silently wrong (normalized error 1.36).
With the K tile equal to ``Kc`` (no Helion K loop, e.g. blocks
``[1, 64, 64, 128]``) the rewrite instead aborted on the ``expand_shape``
base (``ArrayRef.h`` "Invalid index!").

The fix (``amx_move_offsets``): ``drop_contraction_read_unit_dims`` makes
such operand reads minor-identity reads of a rank-reducing view without the
unit dims, and ``move_offsets_to_subview`` keeps its subview of that view at
the register K loop's induction variable in bounds (with the base's whole
size, ``fold-memref-alias-ops`` composed the peeled iteration's subview into
one past the end, which failed to verify). Operand reads drop skipped dims
of any static size (the bf16 form of AI-bench's ``_split_k_partials`` skips
the size-8 S dim of its tile buffer). Last, ``widen_misread_amx_operands``
widens to f32 any bf16 contraction whose operand reads the rewrite would
still misread (other maps, masks, padding, a non-unit innermost stride).

The numeric regressions execute AMX code and run on AMX hosts only; the IR
and transform tests force AMX on (any x86 host).
"""

from __future__ import annotations

import re

import helion
import helion.language as hl
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
import pytest
import torch
from torch import Tensor

from tests.harness import amx_features
from tests.harness import lower_opt_with_amx

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir_transforms import HelionTransformDialect
from helion_mlir_backend._compiler.mlir_transforms import move_offsets_to_subview
from helion_mlir_backend._compiler.mlir_transforms import widen_misread_amx_operands

_HAS_AMX = TargetInfo.host().is_supported("amx")
amx_host_only = pytest.mark.skipif(not _HAS_AMX, reason="executes AMX code")


# --- The kernel ---------------------------------------------------------------
# The form of a GEMM split along K (backend tests do not import AI-bench).


def _matmul_split_k(a: Tensor, b: Tensor) -> Tensor:
    m, splits, kc = a.size()
    _, _, n = b.size()
    out = torch.empty([splits, m, n], dtype=torch.float32, device=a.device)
    for ts, tm, tn in hl.tile([splits, m, n]):
        acc = hl.zeros([ts, tm, tn], dtype=torch.float32)
        for tk in hl.tile(kc):
            chunk = a[tm, ts.begin, tk].unsqueeze(0)
            acc = hl.dot(chunk, b[ts, tk, tn], acc=acc)
        out[ts, tm, tn] = acc
    return out


_M, _K, _N, _SPLITS = 64, 1024, 64, 8

BLOCKS = {
    # Silently wrong: LHS tiles loaded with the unit dim's row stride.
    "k_loop_32x32x32": [1, 32, 32, 32],
    "k_loop_64x64x32": [1, 64, 64, 32],
    # K tile = Kc (no Helion K loop): aborted in the AMX rewrite.
    "no_k_loop_64x64x128": [1, 64, 64, 128],
}


def _case(name: str) -> tuple[helion.Kernel, list[Tensor], Tensor]:
    torch.manual_seed(0)
    a = torch.randn(_M, _K, dtype=torch.bfloat16)
    b = torch.randn(_K, _N, dtype=torch.bfloat16)
    chunk = _K // _SPLITS
    kernel = helion.kernel(
        _matmul_split_k,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=BLOCKS[name]),
    )
    args = [a.view(_M, _SPLITS, chunk), b.view(_SPLITS, chunk, _N)]
    expected = torch.stack(
        [
            a[:, s * chunk : (s + 1) * chunk].double()
            @ b[s * chunk : (s + 1) * chunk].double()
            for s in range(_SPLITS)
        ]
    )
    return kernel, args, expected


@amx_host_only
@pytest.mark.isolated
@pytest.mark.parametrize("name", list(BLOCKS))
def test_split_k_unsqueezed_chunks_match_reference(name: str) -> None:
    """The kernel at the configs that were wrong or aborted: f32 sums of
    bf16 products (normalized error ~1e-7; unfixed 1.36, or an abort)."""
    kernel, args, expected = _case(name)
    out = kernel(*args)
    error = (out.double() - expected).abs().max() / expected.abs().max()
    assert error.item() < 1e-5


def _amx_lowering(name: str, until=None) -> str:
    kernel, args, _ = _case(name)
    lowered = lower_opt_with_amx(
        generate_mlir(kernel, args),
        "_matmul_split_k",
        until or (lambda stage, module: "x86.amx.tile_mulf" in str(module)),
        verify=True,
    )
    assert lowered is not None, "no AMX contraction produced"
    return str(lowered[1])


@pytest.mark.isolated
def test_split_k_lhs_tiles_load_with_row_stride() -> None:
    """With AMX forced on, the LHS tiles are loaded in place from a view
    without the unit dim, rows ``K`` apart (unfixed: from the 4-D view, whose
    second innermost stride is the unit dim's ``Kc``)."""
    text = _amx_lowering("k_loop_32x32x32")
    assert "vector.contract" not in text
    lhs = [
        line
        for line in text.splitlines()
        if "x86.amx.tile_load" in line and "xbf16, strided" in line
    ]
    assert lhs
    assert all("strided<[32768, 1024, 1]" in line for line in lhs), lhs
    assert "strided<[32768, 1024, 128, 1]" not in "".join(lhs)


@pytest.mark.isolated
def test_split_k_without_k_loop_lowers() -> None:
    """With AMX forced on, the K-tile-equals-``Kc`` config gets through every
    stage, verified after each (unfixed: an abort in the AMX rewrite, or a
    subview past the end once its fix alone is in)."""
    assert (
        lower_opt_with_amx(
            generate_mlir(*_case("no_k_loop_64x64x128")[:2]),
            "_matmul_split_k",
            lambda stage, module: False,
            verify=True,
        )
        is None
    )


# --- move_offsets_to_subview: subviews of subviews at a loop's IV --------------


def _apply(source: str) -> str:
    with ir.Context(), ir.Location.unknown():
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            move_offsets_to_subview(
                lh_transform.match_op(named_seq.bodyTarget, "func.func")
            )
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def _k_loop_read(
    base: str = "%view",
    upper: str = "%c128",
    offset: str = "%k",
    step: str = "%c32",
    guarded: bool = False,
) -> str:
    """A read of ``base`` (a static ``32x128`` subview ``%view`` of ``%a``, or
    ``%a`` itself) at ``offset`` in a K loop up to ``upper`` (``guarded``:
    under an ``scf.if``)."""
    read = (
        f"vector.transfer_read {base}[%c0, {offset}], %pad "
        f"{{in_bounds = [true, true]}} : {_TYPES[base]}, vector<16x32xbf16>"
    )
    if guarded:
        read = (
            "scf.if %p -> vector<16x32xbf16> {\n"
            f"      %g = {read}\n"
            "      scf.yield %g : vector<16x32xbf16>\n"
            "    } else {\n"
            "      scf.yield %acc : vector<16x32xbf16>\n"
            "    }"
        )
    return f"""
func.func @f(%a: memref<64x256xbf16>, %j: index, %n: index, %p: i1)
    -> vector<16x32xbf16> {{
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %pad = arith.constant 0.0 : bf16
  %init = arith.constant dense<0.0> : vector<16x32xbf16>
  %view = memref.subview %a[0, 128] [32, 128] [1, 1]
      : memref<64x256xbf16> to memref<32x128xbf16, strided<[256, 1], offset: 128>>
  %r = scf.for %k = %c0 to {upper} step {step} iter_args(%acc = %init)
      -> vector<16x32xbf16> {{
    %v = {read}
    %s = arith.addf %acc, %v : vector<16x32xbf16>
    scf.yield %s : vector<16x32xbf16>
  }}
  return %r : vector<16x32xbf16>
}}
"""


_TYPES = {
    "%view": "memref<32x128xbf16, strided<[256, 1], offset: 128>>",
    "%a": "memref<64x256xbf16>",
}


def test_move_offsets_cuts_loop_dim_of_subview_base() -> None:
    """A read of a static subview at the K loop's IV: the moved subview ends
    where the base does on the last iteration (96 + 32 = 128)."""
    text = _apply(_k_loop_read())
    assert (
        "memref.subview %subview[0, %arg4] [32, 32] [1, 1] : memref<32x128xbf16, "
        "strided<[256, 1], offset: 128>> to memref<32x32xbf16" in text
    )
    assert re.search(r"vector.transfer_read %subview_\d+\[%c0, %c0", text)


@pytest.mark.parametrize(
    ("source", "sizes"),
    [
        # The base is not a subview: its subviews are not composed.
        (_k_loop_read(base="%a"), "[64, 256]"),
        # A runtime trip count: no last iteration known.
        (_k_loop_read(upper="%n"), "[32, 128]"),
        # One iteration: the subview stays at offset 0.
        (_k_loop_read(upper="%c32"), "[32, 128]"),
        # The offset is not the IV.
        (_k_loop_read(offset="%j"), "[32, 128]"),
        # Under an scf.if: the read need not run on the last iteration.
        (_k_loop_read(guarded=True), "[32, 128]"),
    ],
    ids=[
        "base_not_subview",
        "runtime_trip_count",
        "one_iteration",
        "not_the_iv",
        "under_if",
    ],
)
def test_move_offsets_keeps_other_sizes(source: str, sizes: str) -> None:
    text = _apply(source)
    subviews = [line for line in text.splitlines() if "memref.subview" in line]
    moved = subviews[-1]
    assert f"] {sizes} [1, 1]" in moved, moved


# --- Sibling kernels ------------------------------------------------------------
# AI-bench's f32 split-K partials (``_split_k_partials`` in
# backends/utils/helion_mlir_cpu_utils/matmul.py, copied) in bf16: an einsum of
# ``a3[tile_m, tile_s, tile_k]`` (an ``M x 1 x Kc`` tile of ``[M, S, Kc]``).
# Its LHS read skips the S dim of the whole ``[M, S, Kc]`` tile buffer
# (size 8, not 1), which only the any-size drop of operand reads handles.


def _split_k_partials(a3: Tensor, b3: Tensor) -> Tensor:
    m, splits, chunk = a3.shape
    n = b3.size(2)
    chunk = hl.specialize(chunk)
    hl.specialize(b3.size(1))
    out = torch.empty((splits, m, n), dtype=torch.float32, device=a3.device)
    for tile_s, tile_m, tile_n in hl.tile([splits, m, n]):
        acc = hl.zeros([tile_s, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(chunk):
            acc = acc + torch.einsum(
                "msc,scn->smn", a3[tile_m, tile_s, tile_k], b3[tile_s, tile_k, tile_n]
            )
        out[tile_s, tile_m, tile_n] = acc
    return out


def _chunk_innermost(a: Tensor, b: Tensor) -> Tensor:
    """``a`` laid out ``[M, Kc, S]``: the chunk's rows have stride S, not 1
    (``tile_load`` requires a unit innermost stride: the guard's case)."""
    m, kc, s = a.size()
    n = b.size(2)
    out = torch.empty([s, m, n], dtype=torch.float32, device=a.device)
    for ts, tm, tn in hl.tile([s, m, n]):
        acc = hl.zeros([ts, tm, tn], dtype=torch.float32)
        for tk in hl.tile(kc):
            acc = hl.dot(a[tm, tk, ts.begin].unsqueeze(0), b[ts, tk, tn], acc=acc)
        out[ts, tm, tn] = acc
    return out


def _chunk_mixed_use(a: Tensor, b: Tensor) -> Tensor:
    """``_matmul_split_k`` whose chunk is also summed (the chunk feeds the
    contraction and a reduction)."""
    m, s, kc = a.size()
    n = b.size(2)
    out = torch.empty([s, m, n], dtype=torch.float32, device=a.device)
    for ts, tm, tn in hl.tile([s, m, n]):
        acc = hl.zeros([ts, tm, tn], dtype=torch.float32)
        for tk in hl.tile(kc):
            chunk = a[tm, ts.begin, tk].unsqueeze(0)
            acc = hl.dot(chunk, b[ts, tk, tn], acc=acc)
            acc = acc + chunk.to(torch.float32).sum(-1, keepdim=True)
        out[ts, tm, tn] = acc
    return out


def _sibling(name: str) -> tuple[helion.Kernel, list[Tensor], Tensor]:
    torch.manual_seed(0)
    a = torch.randn(_M, _K, dtype=torch.bfloat16)
    b = torch.randn(_K, _N, dtype=torch.bfloat16)
    chunk = _K // _SPLITS
    a3 = a.view(_M, _SPLITS, chunk)
    b3 = b.view(_SPLITS, chunk, _N)
    expected = torch.stack(
        [
            a[:, s * chunk : (s + 1) * chunk].double()
            @ b[s * chunk : (s + 1) * chunk].double()
            for s in range(_SPLITS)
        ]
    )
    fn = {
        "split_k_partials": _split_k_partials,
        "chunk_innermost": _chunk_innermost,
        "chunk_mixed_use": _chunk_mixed_use,
    }[name]
    if name == "chunk_innermost":
        a3 = a3.permute(0, 2, 1).contiguous()
    if name == "chunk_mixed_use":
        expected = expected + torch.stack(
            [
                a[:, s * chunk : (s + 1) * chunk].double().sum(-1, keepdim=True)
                for s in range(_SPLITS)
            ]
        )
    kernel = helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[1, 32, 32, 32]),
    )
    return kernel, [a3, b3], expected


SIBLINGS = ["split_k_partials", "chunk_innermost", "chunk_mixed_use"]


@amx_host_only
@pytest.mark.isolated
@pytest.mark.parametrize("name", SIBLINGS)
def test_sibling_kernels_match_reference(name: str) -> None:
    """64x1024 @ 1024x64, S = 8, blocks [1, 32, 32, 32], bf16. Unfixed:
    split_k_partials 1.357 (also with only the unit-dim drop), chunk_innermost
    1.413 (or "tile_load requires unit innermost stride"). chunk_mixed_use
    is a control (right before too: the contraction gets its own read)."""
    kernel, args, expected = _sibling(name)
    out = kernel(*args)
    error = (out.double() - expected).abs().max() / expected.abs().max()
    assert error.item() < 1e-5


def _final_ir(kernel: helion.Kernel, args: list[Tensor], entry: str) -> str:
    """The module after every stage of the opt pipeline, AMX forced on."""
    seen: list[str] = []

    def until(stage: str, module: object) -> bool:
        seen[:] = [str(module)]
        return False

    lower_opt_with_amx(generate_mlir(kernel, args), entry, until, verify=True)
    return seen[0]


@pytest.mark.isolated
def test_split_k_partials_stays_on_amx() -> None:
    """The landed split-K form in bf16 reaches AMX in place: its LHS read is
    dropped onto a view, so the guard leaves it."""
    kernel, args, _ = _sibling("split_k_partials")
    lowered = lower_opt_with_amx(
        generate_mlir(kernel, args),
        "_split_k_partials",
        lambda stage, module: "x86.amx.tile_mulf" in str(module),
        verify=True,
    )
    assert lowered is not None, "no AMX contraction produced"
    assert "vector.contract" not in str(lowered[1])


@pytest.mark.isolated
def test_guard_moves_misread_kernel_off_amx() -> None:
    """With AMX forced on, the ``[M, Kc, S]`` chunk's contractions (rows S
    apart) are widened to f32 and lower without AMX (unfixed: "tile_load
    requires unit innermost stride", or wrong rows)."""
    kernel, args, _ = _sibling("chunk_innermost")
    text = _final_ir(kernel, args, kernel.fn.__name__)
    assert "amx" not in text


# --- widen_misread_amx_operands -------------------------------------------------

_CONTRACT_MAPS = (
    "indexing_maps = [affine_map<(m, n, k) -> (m, k)>, "
    "affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], "
    'iterator_types = ["parallel", "parallel", "reduction"], '
    "kind = #vector.kind<add>"
)
_PLAIN_LHS = (
    "%l = vector.transfer_read %a[%c0, %c0], %pb {in_bounds = [true, true]} : "
    "memref<16x32xbf16>, vector<16x32xbf16>"
)
_PLAIN_RHS = (
    "%r = vector.transfer_read %b[%c0, %c0], %pb {in_bounds = [true, true]} : "
    "memref<32x16xbf16>, vector<32x16xbf16>"
)


def _contraction(lhs: str = _PLAIN_LHS, rhs: str = _PLAIN_RHS, t: str = "bf16") -> str:
    """One 16x16x32 contraction of ``%l`` and ``%r`` (``lhs``/``rhs`` define
    them) into an accumulator read from ``%c``, written back."""
    body = f"""
  {lhs}
  {rhs}
  %v = vector.transfer_read %c[%c0, %c0], %pf {{in_bounds = [true, true]}}
      : memref<16x16xf32>, vector<16x16xf32>
  %y = vector.contract {{{_CONTRACT_MAPS}}} %l, %r, %v
      : vector<16x32x{t}>, vector<32x16x{t}> into vector<16x16xf32>
  vector.transfer_write %y, %c[%c0, %c0] {{in_bounds = [true, true]}}
      : vector<16x16xf32>, memref<16x16xf32>
  return
"""
    if t != "bf16":
        body = body.replace("bf16", t)
    return f"""
func.func @f(%a: memref<16x32xbf16>, %b: memref<32x16xbf16>, %c: memref<16x16xf32>,
             %at: memref<32x16xbf16>, %a3: memref<16x1x32xbf16>,
             %as: memref<16x32xbf16, strided<[64, 2]>>, %ab: memref<8x32xbf16>,
             %a1: memref<1x32xbf16>,
             %m: vector<16x32xi1>, %x: vector<16x32xbf16>) {{
  %c0 = arith.constant 0 : index
  %pb = arith.constant 0.0 : bf16
  %pf = arith.constant 0.0 : f32
{body}}}
"""


def _apply_guard(source: str, *, amx: bool = True, rewrite: bool = False) -> str:
    """``source`` after ``widen_misread_amx_operands`` (then upstream's AMX
    rewrite if ``rewrite``), with AMX forced on or off."""
    from mlir.dialects.transform import x86

    with (
        TargetInfo.override(features=amx_features(amx)),
        ir.Context(),
        ir.Location.unknown(),
    ):
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
            widen_misread_amx_operands(funcs)
            if rewrite:
                with ir.InsertionPoint(transform.ApplyPatternsOp(funcs).patterns):
                    x86.apply_patterns_x86_vector_contract_to_amx_dot_product()
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def test_guard_keeps_plain_reads_on_amx() -> None:
    """Plain operand reads: left alone, and the AMX rewrite takes the
    contraction (tile loads, tile_mulf)."""
    text = _apply_guard(_contraction(), rewrite=True)
    assert "arith.extf" not in text
    assert "x86.amx.tile_mulf" in text
    assert "vector.contract" not in text


_MISREAD = {
    # A transposing map (K x M buffer read as M x K).
    "transposed": "%l = vector.transfer_read %at[%c0, %c0], %pb {in_bounds = "
    "[true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : "
    "memref<32x16xbf16>, vector<16x32xbf16>",
    # A map skipping a unit dim (left by drop_contraction_read_unit_dims,
    # e.g. a read with another use).
    "skips_unit_dim": "%l = vector.transfer_read %a3[%c0, %c0, %c0], %pb "
    "{in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> "
    "(d0, d2)>} : memref<16x1x32xbf16>, vector<16x32xbf16>",
    # A broadcast dim (rows of the 1x32 buffer repeated).
    "broadcast": "%l = vector.transfer_read %a1[%c0, %c0], %pb {in_bounds = "
    "[true, true], permutation_map = affine_map<(d0, d1) -> (0, d1)>} : "
    "memref<1x32xbf16>, vector<16x32xbf16>",
    # A mask operand.
    "masked": "%l = vector.transfer_read %a[%c0, %c0], %pb, %m {in_bounds = "
    "[true, true]} : memref<16x32xbf16>, vector<16x32xbf16>",
    # In vector.mask.
    "vector_mask": "%l = vector.mask %m { vector.transfer_read %a[%c0, %c0], %pb "
    "{in_bounds = [true, true]} : memref<16x32xbf16>, vector<16x32xbf16> } : "
    "vector<16x32xi1> -> vector<16x32xbf16>",
    # Out-of-bounds rows padded (the 8-row buffer read as 16 rows).
    "padded": "%l = vector.transfer_read %ab[%c0, %c0], %pb {in_bounds = "
    "[false, true]} : memref<8x32xbf16>, vector<16x32xbf16>",
    # A non-unit innermost stride.
    "strided": "%l = vector.transfer_read %as[%c0, %c0], %pb {in_bounds = "
    "[true, true]} : memref<16x32xbf16, strided<[64, 2]>>, vector<16x32xbf16>",
    # Not a read (a value computed in registers).
    "not_a_read": "%l = arith.negf %x : vector<16x32xbf16>",
}


@pytest.mark.parametrize("form", list(_MISREAD))
def test_guard_widens_misread_operands(form: str) -> None:
    """Each LHS form the rewrite misreads: both operands are widened to f32
    and the rewrite leaves the contraction (no AMX ops)."""
    text = _apply_guard(_contraction(lhs=_MISREAD[form]), rewrite=True)
    assert text.count("arith.extf") == 2
    assert "x86.amx" not in text
    assert "vector<16x32xf32>, vector<32x16xf32> into vector<16x16xf32>" in text


def test_guard_widens_misread_rhs() -> None:
    rhs = (
        "%r = vector.transfer_read %a[%c0, %c0], %pb {in_bounds = [true, true], "
        "permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : "
        "memref<16x32xbf16>, vector<32x16xbf16>"
    )
    text = _apply_guard(_contraction(rhs=rhs))
    assert text.count("arith.extf") == 2


_F32_CONTRACTION = f"""
func.func @f(%x: vector<16x32xf32>, %r: vector<32x16xf32>,
             %v: vector<16x16xf32>) -> vector<16x16xf32> {{
  %y = vector.contract {{{_CONTRACT_MAPS}}} %x, %r, %v
      : vector<16x32xf32>, vector<32x16xf32> into vector<16x16xf32>
  return %y : vector<16x16xf32>
}}
"""


@pytest.mark.parametrize(
    ("source", "amx"),
    [
        # No AMX on the target: the rewrite matches no register tile there.
        (_contraction(lhs=_MISREAD["transposed"]), False),
        # f32 operands (not reads): the rewrite does not take them.
        (_F32_CONTRACTION, True),
    ],
    ids=["no_amx", "f32"],
)
def test_guard_leaves_others(source: str, amx: bool) -> None:
    with ir.Context():
        parsed = str(ir.Module.parse(source))
    assert _apply_guard(source, amx=amx) == parsed
