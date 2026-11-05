"""``drop_contraction_read_unit_dims``: a contraction operand or accumulator
read whose permutation map skips unit dims of its buffer, moved onto a
rank-reducing subview without those dims, as a minor-identity read.

Upstream's x86 AMX rewrite (``x86.vector_contract_to_amx_dot_product``)
ignores the permutation maps of the reads it consumes:
- Operands: it loads an LHS tile in place with ``x86.amx.tile_load`` of the
  read's memref, whose row stride is the stride of the memref's second
  innermost dim, and packs an RHS to VNNI with row loads of the read's memref
  (rows taken along that dim). A read that skips a unit dim between its
  vector dims, e.g. an ``M x 1 x K`` view of ``a[m, s, k]`` at a fixed ``s``
  read with ``(d0, d1, d2) -> (d0, d2)``, then gets rows at the unit dim's
  stride: the wrong elements, silently (split-K over ``a.view(M, S, K / S)``
  with an ``unsqueeze``d chunk into a 3-D ``hl.dot``). Such reads come from
  ``simplify_vector_ops`` folding the rank-reducing slices of the
  contraction's operands into the reads.
- Accumulators: it loads only minor-identity reads in place. Any other read
  makes it start from zero tiles and add the accumulator from memory
  afterwards, with the read's memref taken as the register tile: its two
  inner sizes as the tile's rows and columns, the tile at the memref's
  origin. A second K loop accumulating onto the first one's ``M x 1 x N``
  result (a panel dim of size 1) reads it at ``[%c16, %c0, %n]`` with the
  same map; the rewrite then writes past a ``1 x N`` stack buffer (a
  segfault) or fails an ``operator[]`` assertion.

Moved onto the subview, such a read is minor identity: the rewrite loads an
operand with the right row stride, and ``move_offsets_to_subview`` and
``fit_subviews_to_reads`` (``amx_move_offsets``, which runs this first) give
an accumulator register tile its own subview at its origin, which the
rewrite loads in place. The step also runs on hosts without AMX and for any
element type, where it is harmless: the read is the same read of the same
memory, through a view.

Matched: a ``vector.transfer_read`` of a strided ``memref``, without mask (no
mask operand, not in ``vector.mask``), whose permutation map is no minor
identity but becomes one once dims it does not use are dropped, whose result
is only used as ``vector.contract`` operands (LHS, RHS or accumulator) or as
the init of ``scf.for`` iter args only used as accumulators, and whose
indices of the dropped dims are not constants outside the dim (an invalid
static subview). A dim the map does not use is read at the read's index (a
projection; broadcasts are constant results), so a size-1 subview at that
index drops it exactly: reads used only as LHS or RHS drop such dims of any
static size (e.g. ``a[m, s, k]`` read at a fixed ``s`` with ``(d0, d1, d2)
-> (d0, d2)`` from an ``M x S x K`` buffer); other reads only dims of static
size 1 (accumulator subviews become register tiles). The subview takes the
read's indices of the dropped dims as offsets and the whole extent of the
other dims; the read keeps its other indices, padding and ``in_bounds``
(the dims it bounds-checks keep their sizes).

Left alone: reads used otherwise (e.g. through an ``arith.extf``, or passed
on), masked reads, maps that permute dims, skip a dynamic dim, or (for
accumulators) skip a dim that is not of static size 1 (unless leading, as a
minor identity allows), broadcast dims, reads of tensors or of memrefs
without a strided layout (e.g. an affine-map layout), and writes. Contractions
whose operand reads stay in a form the rewrite misreads are widened to f32
(``widen_misread_amx_operands``, last in ``amx_move_offsets``).
"""

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import memref
from mlir.dialects import scf
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import int_constant
from .utils import is_accumulator
from .utils import opview
from .utils import payload_ops


def _is_contraction_read(value: ir.Value) -> bool:
    """Whether every use of ``value`` is a ``vector.contract`` operand (LHS,
    RHS or accumulator), or an ``scf.for`` init of an iter arg only used as
    accumulators (:func:`is_accumulator`)."""
    uses = list(value.uses)
    if not uses:
        return False
    for use in uses:
        if isinstance(opview(use.owner), vector.ContractionOp):
            continue
        if not _is_accumulator_init(use):
            return False
    return True


def _is_operand_read(value: ir.Value) -> bool:
    """Whether every use of ``value`` is a ``vector.contract`` LHS or RHS."""
    uses = list(value.uses)
    return bool(uses) and all(
        isinstance(opview(use.owner), vector.ContractionOp) and use.operand_number < 2
        for use in uses
    )


def _is_accumulator_init(use: ir.OpOperand) -> bool:
    """Whether ``use`` is an ``scf.for`` init whose iter arg only feeds
    accumulators."""
    owner = opview(use.owner)
    if not isinstance(owner, scf.ForOp) or use.operand_number < 3:
        return False
    return is_accumulator(owner.inner_iter_args[use.operand_number - 3])


def _dropped_dims(read: vector.TransferReadOp) -> list[int] | None:
    """The dims of ``read``'s buffer to drop for a minor-identity map;
    ``None`` if ``read`` is not matched."""
    if not isinstance(read.base.type, ir.MemRefType) or read.mask is not None:
        return None
    if read.operation.parent.name == "vector.mask":
        return None
    base_type = ir.MemRefType(read.base.type)
    try:
        base_type.get_strides_and_offset()
    except (RuntimeError, ValueError):
        # No strided layout (e.g. an affine map): no subview to take.
        return None
    permutation = read.permutation_map.value
    rank = base_type.rank
    results = len(permutation.results)
    if permutation == ir.AffineMap.get_minor_identity(rank, results):
        return None
    used = []
    for expr in permutation.results:
        if not isinstance(expr, ir.AffineDimExpr):
            return None
        used.append(expr.position)
    if not used or used != sorted(set(used)):
        return None
    # A dim the map skips is read at the read's index (a projection, not a
    # broadcast: those are constant results, rejected above), so a size-1
    # subview at that index drops it exactly, whatever its size. Operand
    # reads drop any static skipped dim; accumulator reads only unit dims
    # (their subviews become register tiles, see the module docstring).
    operand = _is_operand_read(read.result)
    dynamic = ir.ShapedType.get_dynamic_size()
    dropped = [
        dim
        for dim in range(used[0] + 1, rank)
        if dim not in used
        and (base_type.shape[dim] == 1 or (operand and base_type.shape[dim] != dynamic))
    ]
    kept = [dim for dim in range(rank) if dim not in dropped]
    # Without the dropped dims the map must be a minor identity: the dims it
    # uses are the trailing kept ones.
    if kept[len(kept) - results :] != used:
        return None
    indices = list(read.indices)
    for dim in dropped:
        constant = int_constant(indices[dim])
        if constant is not None and not 0 <= constant < base_type.shape[dim]:
            return None
    if not _is_contraction_read(read.result):
        return None
    return dropped


def _rank_reduced_view(
    base: ir.Value, dropped: list[int], indices: list[ir.Value]
) -> ir.Value:
    """The subview of ``base`` at ``indices`` in the ``dropped`` dims (size
    1 there), whole in the others, without the dropped dims."""
    base_type = ir.MemRefType(base.type)
    strides, offset = base_type.get_strides_and_offset()
    dynamic = ir.ShapedType.get_dynamic_size()
    dynamic_offset = ir.ShapedType.get_dynamic_stride_or_offset()
    index = ir.IndexType.get()
    offsets: list[ir.Value | int] = []
    sizes: list[ir.Value | int] = []
    # As memref.subview infers it: a dim adds nothing at index 0 or with a
    # stride of 0, else a dynamic index or stride makes the offset dynamic.
    view_offset = offset
    for dim, size in enumerate(base_type.shape):
        if dim in dropped:
            constant = int_constant(indices[dim])
            offsets.append(indices[dim] if constant is None else constant)
            sizes.append(1)
            if constant == 0 or strides[dim] == 0:
                continue
            if dynamic_offset in (view_offset, strides[dim]) or constant is None:
                view_offset = dynamic_offset
            else:
                view_offset += constant * strides[dim]
            continue
        offsets.append(0)
        sizes.append(
            memref.dim(base, arith.constant(index, dim)) if size == dynamic else size
        )
    kept = [dim for dim in range(base_type.rank) if dim not in dropped]
    view_type = ir.MemRefType.get(
        [base_type.shape[dim] for dim in kept],
        base_type.element_type,
        ir.StridedLayoutAttr.get(view_offset, [strides[dim] for dim in kept]),
        base_type.memory_space,
    )
    return memref.subview(
        base, offsets, sizes, [1] * base_type.rank, result_type=view_type
    )


def _drop(read: vector.TransferReadOp, dropped: list[int]) -> None:
    indices = list(read.indices)
    with ir.InsertionPoint(read), read.location:
        view = _rank_reduced_view(read.base, dropped, indices)
        kept = [index for dim, index in enumerate(indices) if dim not in dropped]
        vector_rank = len(read.permutation_map.value.results)
        moved = vector.transfer_read(
            read.vector.type,
            view,
            kept,
            ir.AffineMapAttr.get(
                ir.AffineMap.get_minor_identity(len(kept), vector_rank)
            ),
            read.padding,
            read.in_bounds,
        )
    read.result.replace_all_uses_with(moved)
    read.operation.erase()


@transform_op(modifies_payload=True)
class DropContractionReadUnitDimsOp(
    HelionTransformDialect.Operation, name="drop_contraction_read_unit_dims"
):
    """Move every matched contraction operand or accumulator read in the
    target onto a rank-reducing subview without the unit dims its map skips
    (see the module docstring)."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "DropContractionReadUnitDimsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for read in payload_ops(state, op.target, vector.TransferReadOp):
            dropped = _dropped_dims(read)
            if dropped is not None:
                _drop(read, dropped)
        return DiagnosedSilenceableFailure.Success


def drop_contraction_read_unit_dims(
    target: ir.Value,
) -> DropContractionReadUnitDimsOp:
    """snake_case wrapper to create a DropContractionReadUnitDimsOp."""
    return DropContractionReadUnitDimsOp(target=target)
