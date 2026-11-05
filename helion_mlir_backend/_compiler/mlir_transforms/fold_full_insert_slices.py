"""``fold_full_insert_slices``: a ``tensor.insert_slice`` that overwrites all of
its destination, with dynamic sizes, replaced by its source.

A grid tile that may be partial is versioned on being full (``_lower_tile`` in
``lowering/loops.py``); the partial version stores its tile into the tile's
region of the output, a ``tensor.extract_slice`` of dynamic size, as
``insert_slice %tile into %region[0, 0] [%rows, 128] [1, 1]``. Upstream folds
such a full overwrite only when the sizes are static (``InsertSliceOp::fold``).
Left in place, one-shot bufferization keeps ``%region``'s old value for the
insert while the other version writes the region: it copies the region into a
heap buffer (dynamic size) before every tile, full or partial, and copies it
back after. Folded, each version yields its tile and only
the copy into the region after the ``if`` is left
(``sink_copies_into_branches`` then keeps it to partial tiles).

Matched exactly: all offsets static 0, all strides static 1, the source of the
destination's type, and each size the destination's: equal static sizes, or
for a dynamic dim the same SSA value as the size of the destination's
``tensor.extract_slice`` (or a ``tensor.dim`` of the destination). Other
inserts are left unchanged.
"""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op

_DYNAMIC = ir.ShapedType.get_dynamic_size()


def _ops_named(root: ir.Operation, name: str) -> list[ir.Operation]:
    """The ``name`` ops nested in ``root`` (``root`` included), in walk order."""
    found: list[ir.Operation] = []

    def visit(nested: ir.Operation) -> ir.WalkResult:
        if nested.name == name:
            found.append(nested)
        return ir.WalkResult.ADVANCE

    root.walk(visit)
    return found


def _attr_ints(op: ir.Operation, name: str) -> list[int]:
    return list(ir.DenseI64ArrayAttr(op.attributes[name]))


def _segments(op: ir.Operation) -> list[int]:
    return list(ir.DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))


def _dynamic_sizes(op: ir.Operation, first: int) -> list[ir.Value]:
    """The dynamic size operands of an extract/insert slice; ``first`` is the
    index of the size segment in ``operandSegmentSizes``."""
    segments = _segments(op)
    start = sum(segments[:first])
    return list(op.operands[start : start + segments[first]])


def _is_dim_of(value: ir.Value, tensor: ir.Value, index: int) -> bool:
    """Whether ``value`` is ``tensor.dim %tensor, %c<index>``."""
    if not isinstance(value, ir.OpResult) or value.owner.name != "tensor.dim":
        return False
    dim = value.owner
    if dim.operands[0] != tensor:
        return False
    position = dim.operands[1]
    if not isinstance(position, ir.OpResult) or position.owner.name != "arith.constant":
        return False
    return ir.IntegerAttr(position.owner.attributes["value"]).value == index


def _covers(insert: ir.Operation) -> bool:
    """Whether ``insert`` (a ``tensor.insert_slice``) overwrites all of its dest."""
    source, dest = insert.operands[0], insert.operands[1]
    if source.type != dest.type:
        return False
    if any(offset != 0 for offset in _attr_ints(insert, "static_offsets")):
        return False
    if any(stride != 1 for stride in _attr_ints(insert, "static_strides")):
        return False
    shape = ir.RankedTensorType(dest.type).shape
    sizes = _attr_ints(insert, "static_sizes")
    # operandSegmentSizes: source, dest, offsets, sizes, strides.
    dynamic = iter(_dynamic_sizes(insert, 3))
    producer = dest.owner if isinstance(dest, ir.OpResult) else None
    region_sizes: list[ir.Value] | None = None
    if producer is not None and producer.name == "tensor.extract_slice":
        static = _attr_ints(producer, "static_sizes")
        if len(static) == len(shape) and _attr_ints(producer, "static_strides") == [
            1
        ] * len(shape):
            # operandSegmentSizes: source, offsets, sizes, strides.
            values = iter(_dynamic_sizes(producer, 2))
            region_sizes = [next(values) if s == _DYNAMIC else None for s in static]
    for index, (size, extent) in enumerate(zip(sizes, shape, strict=True)):
        if extent != _DYNAMIC:
            if size != extent:
                return False
            continue
        if size != _DYNAMIC:
            return False
        value = next(dynamic)
        if region_sizes is not None and region_sizes[index] == value:
            continue
        if not _is_dim_of(value, dest, index):
            return False
    return True


@transform_op(modifies_payload=True)
class FoldFullInsertSlicesOp(
    HelionTransformDialect.Operation, name="fold_full_insert_slices"
):
    """In the target ops, replace each ``tensor.insert_slice`` that overwrites
    all of its destination (zero offsets, unit strides, the destination's type
    and sizes, dynamic sizes matched by SSA value) by its source, and erase it.
    Other inserts are left unchanged."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "FoldFullInsertSlicesOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for target in state.get_payload_ops(op.target):
            inserts = _ops_named(target.operation, "tensor.insert_slice")
            for insert in [insert for insert in inserts if _covers(insert)]:
                insert.results[0].replace_all_uses_with(insert.operands[0])
                insert.erase()
        return DiagnosedSilenceableFailure.Success


def fold_full_insert_slices(target: ir.Value) -> FoldFullInsertSlicesOp:
    """snake_case wrapper to create a FoldFullInsertSlicesOp."""
    return FoldFullInsertSlicesOp(target=target)
