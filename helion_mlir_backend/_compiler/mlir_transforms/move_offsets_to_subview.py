"""``move_offsets_to_subview``: the x86 AMX contraction rewrite matches operand
tiles read from a subview, rebuilding the reduction loop around a subview
offset that is exactly its induction variable, and needs the tiles of one
operand to share their subview. Memref reads move their offsets to subviews
in that form.

A local version of lighthouse's ``transform_ext.move_offsets_to_subview``,
extended for AMX operands: runtime shapes and strides, constant offsets kept
in the indices and loop offsets split into subviews of their own.

The subviews only re-base the indices: they keep the base's sizes, so extend
past its end (``amx_move_offsets`` then fits accumulators' subviews to their
reads: ``fit_subviews_to_reads``). One exception: a subview at a loop's
induction variable of a base that is itself a subview, read directly in the
loop's body, ends where the base does on the loop's last iteration
(``_in_bounds_sizes``). Only in-bounds, unmasked reads are rewritten, as the
bounds of any other read would be checked against the subview's sizes.
"""

from mlir import ir
from mlir.dialects import affine
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
from .utils import opview
from .utils import payload_ops


def _enclosing_loop(op: ir.OpView) -> scf.ForOp | None:
    """The innermost ``scf.for`` around ``op``."""
    parent = op.operation.parent
    while parent is not None:
        if parent.name == "scf.for":
            return opview(parent)
        if parent.name in ("func.func", "omp.loop_nest"):
            return None
        parent = parent.parent
    return None


def _last_iv(loop: scf.ForOp) -> int | None:
    """The last value of ``loop``'s induction variable, if its bounds are
    constants and it runs at least once."""
    bounds = [int_constant(v) for v in (loop.lowerBound, loop.upperBound, loop.step)]
    if None in bounds:
        return None
    lower, upper, step = bounds
    if step <= 0 or upper <= lower:
        return None
    return lower + (upper - lower - 1) // step * step


def _apply_op(value: ir.Value) -> affine.AffineApplyOp | None:
    """The single-result ``affine.apply`` defining ``value``, if any."""
    if isinstance(value.owner, ir.Block):
        return None
    apply = opview(value.owner)
    if not isinstance(apply, affine.AffineApplyOp):
        return None
    return apply if len(apply.map.value.results) == 1 else None


def _sum_terms(expr: ir.AffineExpr, affine_map: ir.AffineMap) -> list[int] | None:
    """Operand positions of ``expr`` if it is a sum of distinct dims/symbols."""
    if isinstance(expr, ir.AffineAddExpr):
        lhs = _sum_terms(expr.lhs, affine_map)
        rhs = _sum_terms(expr.rhs, affine_map)
        if lhs is None or rhs is None:
            return None
        return lhs + rhs
    if isinstance(expr, ir.AffineDimExpr):
        return [expr.position]
    if isinstance(expr, ir.AffineSymbolExpr):
        return [affine_map.n_dims + expr.position]
    return None


def _split_loop_offset(
    offset: ir.Value, iv: ir.Value | None
) -> tuple[ir.Value, list[ir.Value]] | None:
    """Split ``affine.apply(iv + rest...)`` into ``iv`` and the other summands."""
    apply = _apply_op(offset)
    if iv is None or apply is None:
        return None
    affine_map = apply.map.value
    terms = _sum_terms(affine_map.results[0], affine_map)
    operands = list(apply.mapOperands)
    if terms is None or len(set(terms)) != len(terms):
        return None
    loop_terms = [t for t in terms if operands[t] == iv]
    if len(loop_terms) != 1:
        return None
    return iv, [operands[t] for t in terms if t != loop_terms[0]]


def _split_constant(offset: ir.Value) -> tuple[ir.Value, int]:
    """``offset`` as ``rest + constant`` when it is an ``affine.apply`` adding a
    positive constant: tiles of one operand then share a subview."""
    apply = _apply_op(offset)
    if apply is None:
        return offset, 0
    affine_map = apply.map.value
    expr = affine_map.results[0]
    if not isinstance(expr, ir.AffineAddExpr):
        return offset, 0
    operands = list(apply.mapOperands)
    for constant, rest in ((expr.rhs, expr.lhs), (expr.lhs, expr.rhs)):
        if not isinstance(constant, ir.AffineConstantExpr) or constant.value <= 0:
            continue
        if isinstance(rest, ir.AffineDimExpr):
            return operands[rest.position], constant.value
        if isinstance(rest, ir.AffineSymbolExpr):
            return operands[affine_map.n_dims + rest.position], constant.value
        rest_map = ir.AffineMap.get(affine_map.n_dims, affine_map.n_symbols, [rest])
        return affine.apply(rest_map, operands), constant.value
    return offset, 0


def _sum_values(values: list[ir.Value]) -> ir.Value:
    if len(values) == 1:
        return values[0]
    expr = ir.AffineSymbolExpr.get(0)
    for i in range(1, len(values)):
        expr = expr + ir.AffineSymbolExpr.get(i)
    return affine.apply(ir.AffineMap.get(0, len(values), [expr]), values)


def _unit_stride_subview(
    base: ir.Value, offsets: list[ir.Value], sizes: list[ir.Value | int]
) -> ir.Value:
    """A unit-stride ``memref.subview`` of ``base``: its strides, any offset."""
    base_type = ir.MemRefType(base.type)
    strides, _ = base_type.get_strides_and_offset()
    unit = [1] * len(sizes)
    dynamic_stride = ir.ShapedType.get_dynamic_stride_or_offset()
    if base_type.has_static_shape and dynamic_stride not in strides:
        return memref.subview(base, offsets, sizes, unit)
    dynamic = ir.ShapedType.get_dynamic_size()
    result_type = ir.MemRefType.get(
        [size if isinstance(size, int) else dynamic for size in sizes],
        base_type.element_type,
        ir.StridedLayoutAttr.get(dynamic_stride, strides),
        base_type.memory_space,
    )
    return memref.subview(base, offsets, sizes, unit, result_type=result_type)


def _in_bounds_sizes(
    base: ir.Value,
    sizes: list[ir.Value | int],
    offsets: list[ir.Value],
    loop: scf.ForOp | None,
) -> list[ir.Value | int]:
    """``sizes`` (``base``'s) of the subview of ``base`` at ``offsets``, with
    each static dim at ``loop``'s induction variable cut to end where
    ``base`` does on the last iteration, if ``base`` is itself a
    ``memref.subview``.

    The cut dim then stays in bounds on every iteration. With ``base``'s
    whole size it does not: once the AMX rewrite peels the last iteration,
    its offset folds to a constant, and ``fold-memref-alias-ops`` composes
    it with ``base`` into one subview with a static offset past the end,
    which fails to verify. (Other bases are left as they were: their
    subviews are not composed.) The reads stay in bounds of the cut dim:
    they were in bounds of ``base`` at the last offset."""
    if not isinstance(base, ir.OpResult) or base.owner.name != "memref.subview":
        return sizes
    last_iv = None if loop is None else _last_iv(loop)
    if last_iv is None:
        return sizes
    iv = loop.induction_variable
    return [
        size - last_iv
        if isinstance(size, int) and offset == iv and 0 < last_iv < size
        else size
        for size, offset in zip(sizes, offsets, strict=True)
    ]


def _is_rewritable(read: vector.TransferReadOp) -> bool:
    if not isinstance(read.base.type, ir.MemRefType) or read.mask is not None:
        return False
    if read.operation.parent.name == "vector.mask":
        return False
    if not all(ir.BoolAttr(flag).value for flag in read.in_bounds):
        return False
    permutation = read.permutation_map.value
    return permutation == ir.AffineMap.get_minor_identity(
        permutation.n_dims, len(permutation.results)
    )


def _move_offsets(
    read: vector.TransferReadOp, rewriter: transform.TransformRewriter
) -> None:
    base = read.base
    base_type = ir.MemRefType(base.type)
    index = ir.IndexType.get()
    sizes = [
        memref.dim(base, arith.ConstantOp(index, dim).result)
        if base_type.is_dynamic_dim(dim)
        else size
        for dim, size in enumerate(base_type.shape)
    ]
    zero = arith.ConstantOp(index, 0).result

    # Constant indices, and constants added to an offset, stay in the read.
    offsets: list[ir.Value] = []
    indices: list[ir.Value] = []
    for value in read.indices:
        if isinstance(opview(value.owner), arith.ConstantOp):
            offsets.append(zero)
            indices.append(value)
            continue
        rest, constant = _split_constant(value)
        offsets.append(rest)
        indices.append(arith.ConstantOp(index, constant).result if constant else zero)

    loop = _enclosing_loop(read)
    iv = None if loop is None else loop.induction_variable
    splits = [_split_loop_offset(offset, iv) for offset in offsets]
    if any(splits):
        outer_offsets: list[ir.Value] = []
        inner_offsets: list[ir.Value] = []
        for offset, split in zip(offsets, splits, strict=True):
            if split is None:
                outer_offsets.append(offset)
                inner_offsets.append(zero)
                continue
            loop_iv, rest = split
            outer_offsets.append(_sum_values(rest) if rest else zero)
            inner_offsets.append(loop_iv)
        outer = _unit_stride_subview(base, outer_offsets, sizes)
        subview = _unit_stride_subview(outer, inner_offsets, sizes)
    else:
        # Cut only for a read directly in the loop's body: one under an
        # scf.if need not run on the last iteration.
        direct = loop is not None and read.operation.parent == loop.operation
        subview = _unit_stride_subview(
            base,
            offsets,
            _in_bounds_sizes(base, sizes, offsets, loop) if direct else sizes,
        )

    moved = vector.transfer_read(
        read.vector.type,
        subview,
        indices,
        read.permutation_map,
        read.padding,
        read.in_bounds,
    )
    rewriter.replace_op(read, [moved])


@transform_op(modifies_payload=True)
class MoveOffsetsToSubviewOp(
    HelionTransformDialect.Operation, name="move_offsets_to_subview"
):
    """Move the non-constant offsets of every in-bounds, unmasked,
    minor-identity memref ``vector.transfer_read`` in the target to subviews:
    a loop's induction variable to an inner subview of its own, the rest of
    the offset to an outer one, constants added to it to the read's indices."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "MoveOffsetsToSubviewOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for read in payload_ops(state, op.target, vector.TransferReadOp):
            if not _is_rewritable(read):
                continue
            with ir.InsertionPoint(read), read.location:
                _move_offsets(read, rewriter)
        return DiagnosedSilenceableFailure.Success


def move_offsets_to_subview(target: ir.Value) -> MoveOffsetsToSubviewOp:
    """snake_case wrapper to create a MoveOffsetsToSubviewOp."""
    return MoveOffsetsToSubviewOp(target=target)
