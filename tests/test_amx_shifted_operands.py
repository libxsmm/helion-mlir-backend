"""AMX contractions whose operand tiles are shifted by a constant, or read
from a buffer the K loop fills.

A bf16 ``hl.dot`` with a column-shifted B tile (``b[tk, tn.index + 1]``, one
N tile of 32, B 64/96/128 wide) was silently wrong on AMX hosts (normalized
error 1.39). ``move_offsets_to_subview`` keeps the constant in the reads'
indices (columns 1 and 17 of the K loop's view of B), and upstream's AMX
rewrite of a K loop (``x86.vector_contract_to_amx_dot_product``) packs a flat
RHS to VNNI from the origin of the read's memref, ignoring its indices: B was
read from column 0. ``fit_subviews_to_reads`` now re-bases operand views whose
reads do not start at their origin (the view at column 1, the reads at 0 and
16), so the contraction stays on AMX.

With a B row of 33, 34, 48 or 66 elements and a K tile of 32,
``align_operand_rows`` copies each K block of B into a stack buffer of aligned
rows, viewed above the K loop. The rewrite re-creates the RHS view per
iteration by substituting the K loop's induction variable for one of the
view's offset operands; this view has none (``MutableArrayRef<OpOperand>::
operator[]`` "Invalid index!" abort), and the rewrite's pipelining would pack
the buffer before the loop fills it. ``widen_misread_amx_operands`` now moves
contractions in a K loop off AMX when an operand's memref is no view at the
loop's induction variable, or a flat RHS is read elsewhere than its view's
origin (and column 16), as a safety net for what the fit leaves.

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
from helion_mlir_backend._compiler.mlir_transforms import fit_subviews_to_reads
from helion_mlir_backend._compiler.mlir_transforms import widen_misread_amx_operands

_HAS_AMX = TargetInfo.host().is_supported("amx")
amx_host_only = pytest.mark.skipif(not _HAS_AMX, reason="executes AMX code")


# --- The kernel (the conv feasibility probe's p3_shift_mm.py) -----------------


def mm_shift(a: Tensor, b: Tensor, SHIFT: hl.constexpr, N: hl.constexpr) -> Tensor:
    m, k = a.shape
    out = torch.empty([m, N], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, N]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(a[tm, tk], b[tk, tn.index + SHIFT], acc=acc)
        out[tm, tn] = acc.to(a.dtype)
    return out


def _case(width: int, shift: int, block_k: int) -> tuple[helion.Kernel, list, Tensor]:
    """``a`` 32x64 @ the 32 columns of a 64 x ``width`` ``b`` from ``shift``,
    blocks ``[32, 32, block_k]``, bf16 (the probe's arguments)."""
    torch.manual_seed(0)
    kernel = helion.kernel(
        mm_shift,
        backend="mlir",
        config=helion.Config(block_sizes=[32, 32, block_k]),
    )
    a = torch.randn(32, 64, dtype=torch.bfloat16)
    b = torch.randn(64, width, dtype=torch.bfloat16)
    expected = a.double() @ b[:, shift : shift + 32].double()
    return kernel, [a, b, shift, 32], expected


def _ir_args(args: list) -> list:
    """``args`` for ``generate_mlir``: the ints as Helion constexprs."""
    return [hl.constexpr(arg) if isinstance(arg, int) else arg for arg in args]


# (b's width, shift, K block): unfixed errors 1.39 (64), 1.79 (96), wrong
# (128). B rows 128-256 bytes: no copy, the shifted view is fitted.
SHIFTED = [(64, 1, 64), (64, 1, 32), (96, 1, 32), (128, 1, 32)]
# B rows copied to an aligned buffer above a K loop of one 32 block: unfixed
# the AMX rewrite aborted (also unshifted).
BUFFERED = [(33, 1, 32), (33, 0, 32), (48, 0, 32), (34, 1, 32), (66, 1, 32)]


def _ids(cases: list[tuple[int, int, int]]) -> list[str]:
    return [f"b{w}_shift{s}_bk{k}" for w, s, k in cases]


def _error(case: tuple[int, int, int]) -> float:
    kernel, args, expected = _case(*case)
    out = kernel(*args)
    return ((out.double() - expected).abs().max() / expected.abs().max()).item()


@amx_host_only
@pytest.mark.isolated
@pytest.mark.parametrize("case", SHIFTED, ids=_ids(SHIFTED))
def test_shifted_rhs_matches_reference(case: tuple[int, int, int]) -> None:
    """bf16 output of f32 sums: ~2e-3 (unfixed 1.39 / 1.79)."""
    assert _error(case) < 1e-2


@amx_host_only
@pytest.mark.isolated
@pytest.mark.parametrize("case", BUFFERED, ids=_ids(BUFFERED))
def test_buffered_rhs_matches_reference(case: tuple[int, int, int]) -> None:
    """Unfixed: an abort in the AMX rewrite (the lowering process died)."""
    assert _error(case) < 1e-2


def _final_ir(case: tuple[int, int, int]) -> str:
    """The module after every stage of the opt pipeline, AMX forced on."""
    kernel, args, _ = _case(*case)
    seen: list[str] = []

    def until(stage: str, module: object) -> bool:
        seen[:] = [str(module)]
        return False

    lower_opt_with_amx(
        generate_mlir(kernel, _ir_args(args)), "mm_shift", until, verify=True
    )
    return seen[0]


@pytest.mark.isolated
def test_shifted_rhs_stays_on_amx() -> None:
    """With AMX forced on, the shifted kernel's contractions become AMX tile
    products (2 K blocks x 2x2 tiles) of the B view re-based at column 1."""
    kernel, args, _ = _case(64, 1, 64)
    lowered = lower_opt_with_amx(
        generate_mlir(kernel, _ir_args(args)),
        "mm_shift",
        lambda stage, module: "x86.amx.tile_mulf" in str(module),
        verify=True,
    )
    assert lowered is not None, "no AMX contraction produced"
    before, after = lowered[0], str(lowered[1])
    assert "vector.contract" not in after
    assert after.count("x86.amx.tile_mulf") == 8
    assert "[32, 32] [1, 1] : memref<64x64xbf16>" in before
    assert ", 1] [32, 32] [1, 1] : memref<64x64xbf16>" in before


@pytest.mark.isolated
def test_buffered_rhs_lowers_off_amx() -> None:
    """With AMX forced on, the buffered kernel gets through every stage,
    verified after each, on the f32 path (unfixed: an abort)."""
    text = _final_ir((33, 1, 32))
    assert "amx" not in text


# --- fit_subviews_to_reads and widen_misread_amx_operands on a K loop ---------

_MAPS = (
    "indexing_maps = [affine_map<(m, n, k) -> (m, k)>, "
    "affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], "
    'iterator_types = ["parallel", "parallel", "reduction"], '
    "kind = #vector.kind<add>"
)
_B_VIEW = (
    "%sb = memref.subview %b[%k, 0] [32, 64] [1, 1] : memref<64x64xbf16> "
    "to memref<32x64xbf16, strided<[64, 1], offset: ?>>"
)
_B_TYPE = "memref<32x64xbf16, strided<[64, 1], offset: ?>>"


def _k_loop(
    rhs_rows: tuple[str, str] = ("%c0", "%c0"),
    rhs_cols: tuple[str, str] = ("%c1", "%c17"),
    b_view: str = _B_VIEW,
    b_type: str = _B_TYPE,
    above: str = "",
    extra: str = "",
    b_arg: str = "memref<64x64xbf16>",
) -> str:
    """A K loop over a 2x2 register tile of 16x16 contractions (as
    ``amx_move_offsets`` gets it), the RHS read at ``rhs_rows``/``rhs_cols``
    of ``%sb`` (``b_view``, in the loop; ``above`` the loop)."""
    contracts = "\n".join(
        f"    %d{i} = vector.contract {{{_MAPS}}} %va{i // 2}, %vb{i % 2}, %a{i} "
        ": vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32>"
        for i in range(4)
    )
    reads = "\n".join(
        f"    %vb{j} = vector.transfer_read %sb[{rhs_rows[j]}, {rhs_cols[j]}], %pb "
        f"{{in_bounds = [true, true]}} : {b_type}, vector<32x16xbf16>"
        for j in range(2)
    )
    acc_reads = "\n".join(
        f"  %i{i} = vector.transfer_read %c[{r}, {c}], %pf {{in_bounds = [true, true]}}"
        " : memref<32x32xf32>, vector<16x16xf32>"
        for i, (r, c) in enumerate(
            [("%c0", "%c0"), ("%c0", "%c16"), ("%c16", "%c0"), ("%c16", "%c16")]
        )
    )
    writes = "\n".join(
        f"  vector.transfer_write %r#{i}, %c[{r}, {c}] {{in_bounds = [true, true]}}"
        " : vector<16x16xf32>, memref<32x32xf32>"
        for i, (r, c) in enumerate(
            [("%c0", "%c0"), ("%c0", "%c16"), ("%c16", "%c0"), ("%c16", "%c16")]
        )
    )
    accs = ", ".join(f"%a{i} = %i{i}" for i in range(4))
    vec = "vector<16x16xf32>"
    return f"""
func.func @f(%a: memref<32x64xbf16>, %b: {b_arg}, %c: memref<32x32xf32>,
             %n: index) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c17 = arith.constant 17 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %pb = arith.constant 0.0 : bf16
  %pf = arith.constant 0.0 : f32
{above}
{acc_reads}
  %r:4 = scf.for %k = %c0 to %c64 step %c32 iter_args({accs})
      -> ({vec}, {vec}, {vec}, {vec}) {{
    %sa = memref.subview %a[0, %k] [32, 32] [1, 1] : memref<32x64xbf16>
        to memref<32x32xbf16, strided<[64, 1], offset: ?>>
    {b_view}
{extra}
    %va0 = vector.transfer_read %sa[%c0, %c0], %pb {{in_bounds = [true, true]}}
        : memref<32x32xbf16, strided<[64, 1], offset: ?>>, vector<16x32xbf16>
    %va1 = vector.transfer_read %sa[%c16, %c0], %pb {{in_bounds = [true, true]}}
        : memref<32x32xbf16, strided<[64, 1], offset: ?>>, vector<16x32xbf16>
{reads}
{contracts}
    scf.yield %d0, %d1, %d2, %d3 : {vec}, {vec}, {vec}, {vec}
  }}
{writes}
  return
}}
"""


# The RHS as align_operand_rows leaves it: a buffer viewed above the loop,
# filled with the K block in it.
_BUFFERED = {
    "above": "  %buf = memref.alloca() : memref<32x32xbf16>\n"
    "  %sb = memref.subview %buf[0, 0] [32, 32] [1, 1] : memref<32x32xbf16> "
    "to memref<32x32xbf16>",
    "b_view": "%bk = memref.subview %b[%k, 1] [32, 32] [1, 1] : memref<64x64xbf16> "
    "to memref<32x32xbf16, strided<[64, 1], offset: ?>>",
    "extra": "    memref.copy %bk, %buf : memref<32x32xbf16, strided<[64, 1], "
    "offset: ?>> to memref<32x32xbf16>",
    "b_type": "memref<32x32xbf16>",
    "rhs_cols": ("%c0", "%c16"),
}


def _apply(source: str, *, fit: bool, guard: bool, rewrite: bool) -> str:
    """``source`` after ``fit_subviews_to_reads``, ``widen_misread_amx_operands``
    and upstream's AMX rewrite (each if asked), AMX forced on."""
    from mlir.dialects.transform import x86

    with (
        TargetInfo.override(features=amx_features(True)),
        ir.Context(),
        ir.Location.unknown(),
    ):
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
            if fit:
                fit_subviews_to_reads(funcs)
            if guard:
                widen_misread_amx_operands(funcs)
            if rewrite:
                with ir.InsertionPoint(transform.ApplyPatternsOp(funcs).patterns):
                    x86.apply_patterns_x86_vector_contract_to_amx_dot_product()
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def test_fit_rebases_shifted_rhs_view() -> None:
    """Reads at columns 1 and 17 of the K loop's view of B: the view moves to
    column 1, cut to the 32x32 the reads cover, and the reads to columns 0
    and 16. The rewrite then takes the loop (tile products, no widening)."""
    text = _apply(_k_loop(), fit=True, guard=False, rewrite=False)
    assert (
        "memref.subview %arg1[%arg4, 1] [32, 32] [1, 1] : memref<64x64xbf16> to "
        "memref<32x32xbf16, strided<[64, 1], offset: ?>>" in text
    )
    rhs = [line for line in text.splitlines() if "vector<32x16xbf16>" in line]
    rhs = [line for line in rhs if "transfer_read" in line]
    assert len(rhs) == 2
    assert re.search(r"\[%c0, %c0(_\d+)?\]", rhs[0]), rhs
    assert re.search(r"\[%c0, %c16(_\d+)?\]", rhs[1]), rhs
    rewritten = _apply(_k_loop(), fit=True, guard=True, rewrite=True)
    assert "arith.extf" not in rewritten
    assert "x86.amx.tile_mulf" in rewritten
    assert "vector.contract" not in rewritten


@pytest.mark.isolated
def test_fit_adds_runtime_offset_above_k_loop() -> None:
    """A shifted view at a runtime column ``%n`` (not a loop IV): the add of
    the shift is placed at ``%n``'s definition, above the K loop, so the
    rewrite, which clones only the view per iteration, takes the loop with
    no LICM in between (an add in the loop made it abort in
    ``RewriterBase::eraseOp``)."""
    source = _k_loop(
        b_view="%sb = memref.subview %b[%k, %n] [32, 64] [1, 1] : "
        "memref<64x128xbf16> to memref<32x64xbf16, strided<[128, 1], offset: ?>>",
        b_type="memref<32x64xbf16, strided<[128, 1], offset: ?>>",
        b_arg="memref<64x128xbf16>",
    )
    fitted = _apply(source, fit=True, guard=False, rewrite=False)
    body = fitted[fitted.index("scf.for") :]
    assert "arith.addi" not in body
    assert "arith.addi %arg3, %c1" in fitted
    text = _apply(source, fit=True, guard=True, rewrite=True)
    assert "arith.extf" not in text
    assert "x86.amx.tile_mulf" in text
    assert "vector.contract" not in text


@pytest.mark.parametrize(
    "source",
    [
        # Reads at the view's origin (every lighthouse register tile).
        _k_loop(rhs_cols=("%c0", "%c16")),
        # Shifted along K: the row dim's offset is the loop's IV.
        _k_loop(rhs_rows=("%c1", "%c1"), rhs_cols=("%c0", "%c16")),
        # A runtime column.
        _k_loop(rhs_cols=("%k", "%c17")),
        # The view has another use (a copy of it).
        _k_loop(
            extra="    memref.copy %sb, %sb : " + _B_TYPE + " to " + _B_TYPE,
        ),
    ],
    ids=["at_origin", "row_at_iv", "runtime_index", "other_use"],
)
def test_fit_leaves_other_operand_views(source: str) -> None:
    before = _apply(source, fit=False, guard=False, rewrite=False)
    assert _apply(source, fit=True, guard=False, rewrite=False) == before


@pytest.mark.parametrize(
    "source",
    [
        # Columns 1 and 17 (unfitted): packed from column 0.
        _k_loop(),
        # Rows 1 (a K shift the fit leaves): packed from row 0.
        _k_loop(rhs_rows=("%c1", "%c1"), rhs_cols=("%c0", "%c16")),
        # A buffer viewed above the loop: no IV offset to re-create it by.
        _k_loop(**_BUFFERED),
    ],
    ids=["shifted_columns", "shifted_rows", "buffer_above_loop"],
)
def test_guard_widens_k_loop_misreads(source: str) -> None:
    """Each form the K-loop rewrite misreads: widened, then left by the
    rewrite (4 contractions x 2 operands; no AMX ops)."""
    text = _apply(source, fit=False, guard=True, rewrite=True)
    assert text.count("arith.extf") == 8
    assert "x86.amx" not in text


def test_guard_keeps_k_loop_reads_at_origin() -> None:
    """The register tile as lighthouse emits it: left on AMX."""
    text = _apply(
        _k_loop(rhs_cols=("%c0", "%c16")), fit=False, guard=True, rewrite=True
    )
    assert "arith.extf" not in text
    assert "x86.amx.tile_mulf" in text
