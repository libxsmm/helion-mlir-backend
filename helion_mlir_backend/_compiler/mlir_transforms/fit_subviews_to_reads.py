"""``fit_subviews_to_reads``: a subview only read by in-bounds vector reads of
contraction accumulators at constant indices, or of contraction operands not
starting at its origin, shrunk to the region those reads cover.

``move_offsets_to_subview`` re-bases reads on subviews that keep their base's
sizes (they extend past its end). Upstream's x86 AMX rewrite of a K loop
(``x86.vector_contract_to_amx_dot_product``, flat operands, accumulators read
from memory) takes the accumulator read's memref as the register tile: it
stores ``M x N`` result tiles for the memref's two inner sizes, one per
contraction. A ``32x32`` register tile of a ``64x32`` accumulator buffer read
through a ``64x32`` subview gives 8 tile stores for 4 contractions, and the
rewrite indexes past its results (``SmallVector::operator[]`` assertion, or
memory corruption without assertions); it also reads the accumulator rows
from the memref's origin. Fitted, the subview is the ``32x32`` tile.

Matched: a ``memref.subview`` with unit strides, not rank-reducing, whose
every use is the base of a ``vector.transfer_read`` that is in bounds, unmasked
(no mask operand, not in ``vector.mask``) and minor identity, with constant
indices, and only used as ``vector.contract`` accumulators (directly, or as the
init of ``scf.for`` iter args only used so). Its reads are grouped by the
loops they are inits of, or the blocks of the contractions they accumulate
for (one register tile each). Each group gets a subview of its own covering exactly ``[min, max)`` of its reads
per dim (a read covers its indices plus its extent, 1 in dims the vector does
not cover): the minimum is added to the offsets and subtracted from the read
indices, so the tile starts at the subview's origin, as the rewrite assumes.
The subviews keep the strides (an explicit strided layout). Only groups that
are the accumulator tile the rewrite expects are fitted: 16x16 reads (unit dims
aside) tiling a region of 32x32 inner extents without overlap, lighthouse's AMX
register tile (2x2 contractions of 16x16).

Left alone: subviews with other uses (other ops, nested subviews), reads of
anything but accumulators (contraction operands: see below), reads with runtime indices, permuted, masked
or bounds-checked reads, other strides, rank-reducing subviews, a single group
that already fills its subview, subviews some read would exceed (it would be
out of bounds), and subviews with a group of another shape. None of those
reaches the rewrite silently from lighthouse: it tiles AMX contractions 32x32
in registers, unrolled to 16x16 (``strategy_register_parallel`` /
``_unroll``), so its groups are 2x2, or 1x2 / 2x1 for a dim of 16, never
taller or wider. The rewrite pairs contractions along N and leaves 2x1 groups
alone (it fails to find the pair); a 1x2 group (16x32) unfitted keeps its
buffer's sizes, so in a taller buffer the rewrite would store more result
tiles than it has results and assert (the eudsl wheel has assertions), as all
tiles did before this fit: loud, not silent.

Operand subviews whose reads do not start at their origin are fitted too.
In a K loop, the rewrite packs a flat RHS (``K x N``) to VNNI from the
origin of its read's memref: the 32 x 32 block at row 0, column 0, whatever
the reads' indices (it only pairs the two reads 16 columns apart). A
column-shifted tile (``b[tk, tn.index + 1]``: reads at columns 1 and 17)
was packed from column 0, silently wrong. Matched: a subview as above
whose every use is such a read (constant indices) used only as a
``vector.contract`` LHS or RHS, with the minimum of its reads' region not
at the origin. It gets one subview of exactly that region (in bounds of
the old one), the minimum moved from the indices to the offsets, so the
reads start at the origin. Dims whose offset is a loop's induction
variable are not shifted (the rewrite finds the K loop's tile by that
offset; an ``arith.addi`` of it would hide it): such subviews are left
(``widen_misread_amx_operands`` then moves them off AMX). Operand reads
that start at their origin, as every lighthouse register tile's do, are
left as they are."""

from collections.abc import Callable
import math

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import memref
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import int_constant
from .utils import is_accumulator
from .utils import opview
from .utils import payload_ops


def _accumulated_by(value: ir.Value) -> frozenset:
    """Where ``value`` accumulates: the loops it is an init of, and the blocks
    of the contractions it is the accumulator of (one register tile each)."""
    keys = set()
    for use in value.uses:
        owner = opview(use.owner)
        if isinstance(owner, vector.ContractionOp):
            keys.add(owner.operation.block)
        else:
            keys.add(owner)
    return frozenset(keys)


def _read_region(
    use: ir.OpOperand, rank: int, used_as: Callable[[ir.Value], bool] = is_accumulator
) -> tuple[list[int], list[int]] | None:
    """The start and end, per dim, of the region the read ``use`` covers;
    ``None`` if it is no read as matched (its result used as ``used_as``
    accepts: accumulators by default)."""
    read = opview(use.owner)
    if not isinstance(read, vector.TransferReadOp) or use.operand_number != 0:
        return None
    if read.mask is not None or read.operation.parent.name == "vector.mask":
        return None
    if not all(ir.BoolAttr(flag).value for flag in read.in_bounds):
        return None
    shape = ir.VectorType(read.vector.type).shape
    permutation = read.permutation_map.value
    if permutation != ir.AffineMap.get_minor_identity(rank, len(shape)):
        return None
    if not used_as(read.result):
        return None
    indices = [int_constant(index) for index in read.indices]
    if len(indices) != rank or any(index is None for index in indices):
        return None
    extents = [1] * (rank - len(shape)) + list(shape)
    return indices, [i + e for i, e in zip(indices, extents, strict=True)]


# The rows and columns of an AMX accumulator tile (f32).
_TILE = 16

# A group of reads to fit one subview to: their uses, the region's start and end.
_Group = tuple[list[ir.OpOperand], list[int], list[int]]


def _is_register_tile(
    reads: list[ir.OpOperand], low: list[int], high: list[int]
) -> bool:
    """Whether ``reads`` tile ``[low, high)`` as the accumulator tile the AMX
    rewrite expects: 16x16 tiles (unit dims aside), not overlapping, of a
    region whose two inner extents are 32x32: lighthouse's 2x2 register tile.
    The rewrite reads the C rows 32 wide (two halves, at columns 0 and 16)
    and pairs the contractions of such a tile; a taller group (e.g. 64x32, 8
    contractions) is no tile it handles."""
    for use in reads:
        shape = ir.VectorType(opview(use.owner).vector.type).shape
        if [size for size in shape if size != 1] != [_TILE, _TILE]:
            return False
    extents = [h - lo for lo, h in zip(low, high, strict=True)]
    return extents[-2:] == [2 * _TILE, 2 * _TILE] and math.prod(
        extents
    ) == _TILE * _TILE * len(reads)


def _fitted_groups(subview: memref.SubViewOp) -> list[_Group] | None:
    """The reads of ``subview`` grouped by the register tile they accumulate
    (``_accumulated_by``), with the region each group covers, if ``subview`` is
    matched, every group is such a tile and some covers less than it."""
    source_type = ir.MemRefType(subview.source.type)
    result_type = ir.MemRefType(subview.result.type)
    rank = result_type.rank
    if rank != source_type.rank or list(subview.static_strides) != [1] * rank:
        return None
    uses = list(subview.result.uses)
    if not uses:
        return None
    groups: dict[frozenset, _Group] = {}
    for use in uses:
        region = _read_region(use, rank)
        if region is None:
            return None
        start, end = region
        key = _accumulated_by(opview(use.owner).result)
        if key not in groups:
            groups[key] = ([], start, end)
        reads, low, high = groups[key]
        reads.append(use)
        groups[key] = (
            reads,
            list(map(min, low, start)),
            list(map(max, high, end)),
        )
    shape = result_type.shape
    dynamic = ir.ShapedType.get_dynamic_size()
    for reads, low, high in groups.values():
        if any(h > d for h, d in zip(high, shape, strict=True) if d != dynamic):
            return None
        if not _is_register_tile(reads, low, high):
            return None
    if len(groups) == 1:
        ((_, low, high),) = groups.values()
        if low == [0] * rank and high == list(shape):
            return None
    return list(groups.values())


def _is_operand(value: ir.Value) -> bool:
    """Whether every use of ``value`` is a ``vector.contract`` LHS or RHS."""
    uses = list(value.uses)
    return bool(uses) and all(
        isinstance(opview(use.owner), vector.ContractionOp)
        and use.operand_number in (0, 1)
        for use in uses
    )


def _is_loop_iv(value: ir.Value) -> bool:
    """Whether ``value`` is the induction variable of an ``scf.for``."""
    if not isinstance(value, ir.BlockArgument):
        return False
    owner = value.owner.owner
    return owner is not None and owner.name == "scf.for" and value.arg_number == 0


def _fitted_operand_group(subview: memref.SubViewOp) -> list[_Group] | None:
    """The reads of ``subview`` as one group with the region they cover, if
    ``subview`` is matched as an operand subview (see the module docstring)
    and that region does not start at its origin."""
    source_type = ir.MemRefType(subview.source.type)
    result_type = ir.MemRefType(subview.result.type)
    rank = result_type.rank
    if rank != source_type.rank or list(subview.static_strides) != [1] * rank:
        return None
    uses = list(subview.result.uses)
    if not uses:
        return None
    low: list[int] | None = None
    high: list[int] | None = None
    for use in uses:
        region = _read_region(use, rank, _is_operand)
        if region is None:
            return None
        start, end = region
        low = start if low is None else list(map(min, low, start))
        high = end if high is None else list(map(max, high, end))
    if not any(low):
        return None
    dynamic = ir.ShapedType.get_dynamic_size()
    shape = result_type.shape
    if any(h > d for h, d in zip(high, shape, strict=True) if d != dynamic):
        return None
    dynamic_offset = ir.ShapedType.get_dynamic_stride_or_offset()
    values = iter(subview.offsets)
    for offset, shift in zip(subview.static_offsets, low, strict=True):
        if offset != dynamic_offset:
            continue
        value = next(values)
        if shift and _is_loop_iv(value):
            return None
    return [(uses, low, high)]


def _after_definition(value: ir.Value) -> ir.InsertionPoint:
    """An insertion point right after ``value`` is defined."""
    if isinstance(value, ir.BlockArgument):
        return ir.InsertionPoint.at_block_begin(value.owner)
    return ir.InsertionPoint.after(value.owner)


def _shifted_offsets(
    subview: memref.SubViewOp, low: list[int]
) -> tuple[list[ir.Value], list[int]]:
    """The subview's dynamic and static offsets, each moved by ``low``."""
    dynamic_offset = ir.ShapedType.get_dynamic_stride_or_offset()
    index = ir.IndexType.get()
    values = list(subview.offsets)
    dynamic: list[ir.Value] = []
    static: list[int] = []
    for offset, shift in zip(subview.static_offsets, low, strict=True):
        if offset != dynamic_offset:
            static.append(offset + shift)
            continue
        value = values.pop(0)
        if shift:
            # At the offset's definition: an add in the loop body would be
            # cloned with the subview by the AMX rewrite (which only clones
            # the subview), and abort without LICM in between.
            with _after_definition(value):
                value = arith.addi(value, arith.constant(index, shift))
        dynamic.append(value)
        static.append(dynamic_offset)
    return dynamic, static


def _fit(subview: memref.SubViewOp, groups: list[_Group]) -> None:
    result_type = ir.MemRefType(subview.result.type)
    strides, offset = result_type.get_strides_and_offset()
    dynamic_offset = ir.ShapedType.get_dynamic_stride_or_offset()
    index = ir.IndexType.get()
    for reads, low, high in groups:
        fitted_offset = offset
        if offset != dynamic_offset and dynamic_offset not in strides:
            fitted_offset = offset + sum(
                lo * st for lo, st in zip(low, strides, strict=True)
            )
        elif any(low):
            fitted_offset = dynamic_offset
        fitted_type = ir.MemRefType.get(
            [h - lo for lo, h in zip(low, high, strict=True)],
            result_type.element_type,
            ir.StridedLayoutAttr.get(fitted_offset, strides),
            result_type.memory_space,
        )
        with ir.InsertionPoint(subview), subview.location:
            offsets, static_offsets = _shifted_offsets(subview, low)
            fitted = memref.SubViewOp(
                fitted_type,
                subview.source,
                offsets,
                [],
                list(subview.strides),
                static_offsets,
                [h - lo for lo, h in zip(low, high, strict=True)],
                subview.static_strides,
            )
        for use in reads:
            read = use.owner
            read.operands[0] = fitted.result
            if not any(low):
                continue
            with ir.InsertionPoint(read), read.location:
                for dim, shift in enumerate(low):
                    if shift:
                        old = int_constant(read.operands[1 + dim])
                        read.operands[1 + dim] = arith.constant(index, old - shift)
    subview.operation.erase()


@transform_op(modifies_payload=True)
class FitSubviewsToReadsOp(
    HelionTransformDialect.Operation, name="fit_subviews_to_reads"
):
    """Shrink every matched ``memref.subview`` in the target (accumulator
    tiles, and operand subviews whose reads do not start at their origin) to
    the region its reads cover (see the module docstring)."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "FitSubviewsToReadsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for subview in payload_ops(state, op.target, memref.SubViewOp):
            groups = _fitted_groups(subview) or _fitted_operand_group(subview)
            if groups is not None:
                _fit(subview, groups)
        return DiagnosedSilenceableFailure.Success


def fit_subviews_to_reads(target: ir.Value) -> FitSubviewsToReadsOp:
    """snake_case wrapper to create a FitSubviewsToReadsOp."""
    return FitSubviewsToReadsOp(target=target)
