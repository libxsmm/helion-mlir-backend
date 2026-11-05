"""``forward_rows``: canonicalization folds a chain of row inserts that builds an
n-D vector into one ``vector.from_elements`` of every element of every row
(each row taken apart by ``vector.to_elements``). Nothing re-forms the rows
when the vector is then cast and used row by row, as around a VNNI pack's
transpose: its rows read and rebuilt before the bitcast, its shuffled rows
rebuilt, bitcast and written. Lowered to LLVM, that is one extract and one
insert per element. Forwarded to its users, each row is cast and used on its
own: no n-D vector is built.

Matched: a ``vector.from_elements`` of static, fixed-size type of rank 2 or
more whose operands are, row by row (the slices along its leading dim), all
the results of one ``vector.to_elements``, in order, of a vector of the same
element type, and whose uses, through ``vector.bitcast`` and
``vector.shape_cast`` ops that keep the leading dim (and rank 2 or more), are
all:

- ``vector.extract`` of one row at a static position (forwarded the row);
- ``vector.transfer_write`` of the whole vector, unmasked, in bounds and
  minor identity, to a memref or a tensor (one write per row).

Left alone: partial rows (a row's elements from several vectors, or only some
of a vector's), rows out of order, mixed element types, scalable or dynamic
shapes, rank 0 and 1 vectors, more than 64 rows, and any other use: an extract of less than a
row or at a dynamic position, a masked, out-of-bounds or permuted write, a cast
changing the leading dim, an ``scf.yield``, or any other op."""

import math

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import payload_ops

# Rows a vector is forwarded as at most.
_MAX_ROWS = 64
_CASTS = ("vector.bitcast", "vector.shape_cast")


def _is_row_vector(type_: ir.Type) -> bool:
    """A static, fixed-size vector type of rank 2 or more."""
    if not isinstance(type_, ir.VectorType):
        return False
    vector_type = ir.VectorType(type_)
    return (
        vector_type.rank >= 2
        and vector_type.has_static_shape
        and not vector_type.scalable
    )


def _row_type(vector_type: ir.VectorType) -> ir.VectorType:
    """The type of a row of ``vector_type``, flattened."""
    width = math.prod(vector_type.shape[1:])
    return ir.VectorType.get([width], vector_type.element_type)


def _shape_cast(value: ir.Value, type_: ir.VectorType) -> ir.Value:
    return value if value.type == type_ else vector.ShapeCastOp(type_, value).result


def _rows(built: vector.FromElementsOp) -> list[ir.Value] | None:
    """The vectors whose elements are, in order, the rows of ``built``."""
    vector_type = ir.VectorType(built.result.type)
    width = math.prod(vector_type.shape[1:])
    operands = list(built.operands)
    rows = []
    for start in range(0, len(operands), width):
        group = operands[start : start + width]
        owner = group[0].owner
        if not isinstance(owner, ir.Operation | ir.OpView):
            return None
        owner = owner.operation
        if owner.name != "vector.to_elements" or list(owner.results) != group:
            return None
        source = owner.operands[0]
        source_type = source.type
        if (
            not isinstance(source_type, ir.VectorType)
            or ir.VectorType(source_type).scalable
            or ir.VectorType(source_type).element_type != vector_type.element_type
        ):
            return None
        rows.append(source)
    return rows


def _is_row_extract(extract: vector.ExtractOp, leading: int) -> bool:
    position = list(extract.static_position)
    return (
        not list(extract.dynamic_position)
        and len(position) == 1
        and 0 <= position[0] < leading
    )


def _is_whole_write(write: vector.TransferWriteOp) -> bool:
    vector_type = ir.VectorType(write.valueToStore.type)
    base_rank = ir.ShapedType(write.base.type).rank
    minor_identity = ir.AffineMap.get_minor_identity(base_rank, vector_type.rank)
    return (
        write.mask is None
        and write.permutation_map.value == minor_identity
        and all(ir.BoolAttr(flag).value for flag in write.in_bounds)
    )


def _users(value: ir.Value, leading: int) -> list[ir.Operation] | None:
    """The casts from ``value`` and the row-wise users of it and of them, in
    pre-order; None if any use is not one."""
    found: list[ir.Operation] = []
    for use in value.uses:
        user = use.owner
        user = user.operation if isinstance(user, ir.OpView) else user
        if user.name in _CASTS:
            result_type = user.results[0].type
            if not _is_row_vector(result_type) or (
                ir.VectorType(result_type).shape[0] != leading
            ):
                return None
            nested = _users(user.results[0], leading)
            if nested is None:
                return None
            found += [user, *nested]
        elif user.name == "vector.extract":
            if not _is_row_extract(user.opview, leading):
                return None
            found.append(user)
        elif user.name == "vector.transfer_write":
            if use.operand_number != 0 or not _is_whole_write(user.opview):
                return None
            found.append(user)
        else:
            return None
    return found


def _forward(
    built: vector.FromElementsOp, rewriter: transform.TransformRewriter
) -> None:
    if not _is_row_vector(built.result.type):
        return
    vector_type = ir.VectorType(built.result.type)
    leading = vector_type.shape[0]
    if leading > _MAX_ROWS:
        return
    rows = _rows(built)
    users = _users(built.result, leading) if rows is not None else None
    if rows is None or users is None:
        return
    # The rows of every vector in the tree, flattened (a row is contiguous: a
    # bitcast of its last dim is one of the flat row, a shape cast keeping the
    # leading dim none), cast as it is, before its op.
    with ir.InsertionPoint(built), built.location:
        row_type = _row_type(vector_type)
        forwarded = {built.result: [_shape_cast(row, row_type) for row in rows]}
    casts = [user for user in users if user.name in _CASTS]
    sinks = [user for user in users if user.name not in _CASTS]
    for cast in casts:
        source, result = cast.operands[0], cast.results[0]
        rows = forwarded[source]
        if cast.name == "vector.bitcast":
            cast_type = _row_type(ir.VectorType(result.type))
            with ir.InsertionPoint(cast), cast.location:
                rows = [vector.BitCastOp(cast_type, row).result for row in rows]
        forwarded[result] = rows
    for sink in sinks:
        if sink.name == "vector.extract":
            row = forwarded[sink.operands[0]][sink.opview.static_position[0]]
            with ir.InsertionPoint(sink), sink.location:
                row = _shape_cast(row, ir.VectorType(sink.results[0].type))
            rewriter.replace_op(sink, [row])
        else:
            _write_rows(sink.opview, forwarded[sink.operands[0]], rewriter)
    # Nested casts first: each is unused once its users are gone.
    for cast in reversed(casts):
        rewriter.erase_op(cast)
    rewriter.erase_op(built)


def _write_rows(
    write: vector.TransferWriteOp,
    rows: list[ir.Value],
    rewriter: transform.TransformRewriter,
) -> None:
    """``write`` as one write per row."""
    base_type = write.base.type
    base_rank = ir.ShapedType(base_type).rank
    rank = ir.VectorType(write.valueToStore.type).rank
    leading = base_rank - rank
    row_map = ir.AffineMap.get_minor_identity(base_rank, rank - 1)
    is_tensor = isinstance(base_type, ir.RankedTensorType)
    index = ir.IndexType.get()
    dest = write.base
    vector_type = ir.VectorType(write.valueToStore.type)
    slice_type = ir.VectorType.get(vector_type.shape[1:], vector_type.element_type)
    with ir.InsertionPoint(write), write.location:
        for position, row in enumerate(rows):
            row = _shape_cast(row, slice_type)
            indices = list(write.indices)
            if position:
                offset = arith.ConstantOp(index, position).result
                indices[leading] = arith.AddIOp(indices[leading], offset).result
            row_write = vector.TransferWriteOp(
                base_type if is_tensor else None,
                row,
                dest,
                indices,
                row_map,
                [True] * (rank - 1),
            )
            if is_tensor:
                dest = row_write.result
    if is_tensor:
        rewriter.replace_op(write, [dest])
    else:
        rewriter.erase_op(write)


@transform_op(modifies_payload=True)
class ForwardRowsOp(HelionTransformDialect.Operation, name="forward_rows"):
    """Forward the rows of every ``vector.from_elements`` in the target built
    from whole rows and used only row by row (see the module docstring) to its
    users: casts of the rows, the rows themselves for row extracts, one write
    per row for whole writes."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "ForwardRowsOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for built in payload_ops(state, op.target, vector.FromElementsOp):
            _forward(built, rewriter)
        return DiagnosedSilenceableFailure.Success


def forward_rows(target: ir.Value) -> ForwardRowsOp:
    """snake_case wrapper to create a ForwardRowsOp."""
    return ForwardRowsOp(target=target)
