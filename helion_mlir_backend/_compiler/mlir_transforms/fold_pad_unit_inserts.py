"""``fold_pad_unit_inserts``: a padded tile put back into the unit dims it was
stripped of, by a ``tensor.insert_slice`` into a ``tensor.empty``, as a
``tensor.expand_shape``.

Upstream's unit-dim folding rewrites a pad with unit dims (a tile's load under a
grid index, ``q[h, tile_m, tile_d]``, past the end of the last dim) as a pad of
fewer dims inserted whole into an empty tensor. The insert hides the pad's readers
from ``materialize_operand_pads``, which follows reshapes: an operand left a pad
reaches ``vectorize_pads`` whole, as one vector of all its elements, copied row by
row after bufferizing. The
insert only adds unit dims, so the reshape is the same value with no copy.

Exact pattern: a ``tensor.insert_slice`` of a static ``tensor.pad`` into a
``tensor.empty`` of the same shape plus unit dims, at zero offsets and unit
strides. Anything else (another source, a destination with contents, a slice not
covering it) is left alone.
"""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import payload_ops


def _unit_dim_groups(source: list[int], result: list[int]) -> list[list[int]] | None:
    """The reassociation expanding ``source`` to ``result`` by adding unit dims,
    or ``None`` if ``result`` isn't ``source`` plus unit dims."""
    if not source or len(result) <= len(source):
        return None
    groups: list[list[int]] = []
    group: list[int] = []
    for dim, size in enumerate(result):
        group.append(dim)
        if len(groups) < len(source) and size == source[len(groups)]:
            groups.append(group)
            group = []
        elif size != 1:
            return None
    if len(groups) != len(source):
        return None
    groups[-1].extend(group)
    return groups


def _fold(insert: tensor.InsertSliceOp, rewriter: transform.TransformRewriter) -> None:
    source, dest = insert.source, insert.dest
    if not (
        isinstance(source, ir.OpResult)
        and isinstance(source.owner.opview, tensor.PadOp)
        and isinstance(dest, ir.OpResult)
        and isinstance(dest.owner.opview, tensor.EmptyOp)
    ):
        return
    source_type = ir.RankedTensorType(source.type)
    result_type = ir.RankedTensorType(insert.result.type)
    if (
        not source_type.has_static_shape
        or not result_type.has_static_shape
        or any(offset != 0 for offset in insert.static_offsets)
        or any(stride != 1 for stride in insert.static_strides)
        or list(insert.static_sizes) != list(result_type.shape)
    ):
        return
    groups = _unit_dim_groups(list(source_type.shape), list(result_type.shape))
    if groups is None:
        return
    with ir.InsertionPoint(insert), insert.location:
        expanded = tensor.ExpandShapeOp(
            result_type,
            source,
            groups,
            [],
            static_output_shape=list(result_type.shape),
        )
    rewriter.replace_op(insert, [expanded.result])


@transform_op(modifies_payload=True)
class FoldPadUnitInsertsOp(
    HelionTransformDialect.Operation, name="fold_pad_unit_inserts"
):
    """Replace every matching ``tensor.insert_slice`` in the target (see the module
    docstring) by a ``tensor.expand_shape``."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "FoldPadUnitInsertsOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for insert in payload_ops(state, op.target, tensor.InsertSliceOp):
            _fold(insert, rewriter)
        return DiagnosedSilenceableFailure.Success


def fold_pad_unit_inserts(target: ir.Value) -> FoldPadUnitInsertsOp:
    """snake_case wrapper to create a FoldPadUnitInsertsOp."""
    return FoldPadUnitInsertsOp(target=target)
