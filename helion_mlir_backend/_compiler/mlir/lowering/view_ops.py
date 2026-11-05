"""Views and reshapes without an ATen helper round-trip, ``hl.subscript``, and
``hl.split``/``hl.join``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language.view_ops as helion_view_ops
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import torch

from ..support import UnsupportedOperationError
from ..support import ValueNotFoundError
from ..support import static_dim
from . import emit
from .registry import NOT_APPLICABLE
from .registry import lowers

if TYPE_CHECKING:
    import sympy

    from ..build_context import BuildContext

aten = torch.ops.aten


@lowers(aten.view.default, aten.reshape.default)
def lower_view(ctx: BuildContext, node: torch.fx.Node) -> object:
    reshaped = view(ctx, node)
    return NOT_APPLICABLE if reshaped is None else reshaped


def view(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """``node``'s view or reshape of its first argument without a helper, else ``None``.

    With runtime (``?``) dims only unit dims may be added or removed, and since
    ``?`` types cannot tell sizes apart, the other dims' size symbols must match.
    """
    from ..aten_bridge import infer_results

    source = node.args[0]
    value = ctx.get_value(source)
    (result,) = infer_results(ctx, node)
    shape = _static_shape(ctx, source, node, value) or [
        static_dim(dim) for dim in result.shape
    ]
    dynamic = any(
        ir.ShapedType.is_dynamic_size(dim)
        for dim in [*ir.RankedTensorType(value.type).shape, *shape]
    )
    if dynamic and _non_unit_sizes(ctx, source) != _non_unit_sizes(ctx, node):
        return None
    return reshape(value, shape)


def _static_shape(
    ctx: BuildContext, source: torch.fx.Node, node: torch.fx.Node, value: ir.Value
) -> list[int] | None:
    """``node``'s shape as ints when its sizes are symbolic but ``value``, the
    lowered source, is static: e.g. a full ``[256, 256]`` tile reshaped to
    ``[tile.block_size // 32, 32, ...]``. Only if the source's sizes, with block
    sizes substituted, are exactly ``value``'s dims (so the substitution holds
    here) and the element counts match. Partial tiles under dynamic shapes are
    padded to whole blocks, so they are static and qualify too; a source with
    a dynamic dim (e.g. all columns of a dynamic extent) keeps its symbolic
    sizes. A static result lowers to a view (``expand_shape``,
    ``collapse_shape``), not a ``tensor.reshape``, which bufferizes to a copy of
    a strided source."""
    source_shape = list(ir.RankedTensorType(value.type).shape)
    result_val = node.meta.get("val")
    source_val = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
    if not isinstance(result_val, torch.Tensor) or not isinstance(
        source_val, torch.Tensor
    ):
        return None
    if all(isinstance(dim, int) for dim in result_val.shape):
        return None
    if any(ir.ShapedType.is_dynamic_size(dim) for dim in source_shape):
        return None

    def ints(sizes: object) -> list[int] | None:
        exprs = [ctx.sizes.expr(size) for size in sizes]
        if any(expr.free_symbols or not expr.is_integer for expr in exprs):
            return None
        return [int(expr) for expr in exprs]

    if ints(source_val.shape) != source_shape:
        return None
    shape = ints(result_val.shape)
    if shape is None or any(dim <= 0 for dim in shape):
        return None
    return shape if _numel(shape) == _numel(source_shape) else None


def _non_unit_sizes(ctx: BuildContext, node: torch.fx.Node) -> list[sympy.Expr]:
    sizes = (ctx.sizes.expr(size) for size in node.meta["val"].shape)
    return [size for size in sizes if size != 1]


@lowers(helion_view_ops.subscript)
def lower_subscript(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``x[None, :]``: ``x`` with a unit dim at each ``None`` (Helion only admits
    ``None`` and ``:`` here)."""
    source_node, index = node.args[:2]
    value = ctx.get_value(source_node)
    if value is None:
        raise ValueNotFoundError(source_node, context="subscripted tensor")
    dims = iter(ir.RankedTensorType(value.type).shape)
    shape = []
    for item in index:
        if item is None:
            shape.append(1)
        elif item == slice(None):
            shape.append(next(dims))
        else:
            raise UnsupportedOperationError("subscript", reason=f"index {item!r}")
    return reshape(value, shape)


@lowers(helion_view_ops.split)
def lower_split(ctx: BuildContext, node: torch.fx.Node) -> emit.Results:
    """``hl.split(x)``: the two halves of ``x``'s last dimension (of size 2)."""
    value = ctx.get_value(node.args[0])
    last = ir.RankedTensorType(value.type).rank - 1
    return emit.Results(
        [take(ctx, value, last, ctx.index_const(half)) for half in (0, 1)]
    )


@lowers(helion_view_ops.join)
def lower_join(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``hl.join(a, b)``: ``a`` and ``b`` stacked along a new last dimension."""
    halves = [ctx.get_value(arg) for arg in node.args[:2]]
    value_type = ir.RankedTensorType(halves[0].type)
    rank = value_type.rank
    joined = emit.empty([*emit.sizes(halves[0]), 2], value_type.element_type)
    for half, value in enumerate(halves):
        joined = put(ctx, joined, value, rank, ctx.index_const(half))
    return joined


def take(ctx: BuildContext, value: ir.Value, dim: int, position: ir.Value) -> ir.Value:
    """``value`` at ``position`` along ``dim``, without that dimension."""
    shape = list(ir.RankedTensorType(value.type).shape)
    offsets = [ctx.index_const(0)] * len(shape)
    offsets[dim] = position
    sizes = emit.sizes(value)
    sizes[dim] = 1
    return emit.extract_slice(value, offsets, sizes, shape[:dim] + shape[dim + 1 :])


def put(
    ctx: BuildContext, dest: ir.Value, item: ir.Value, dim: int, position: ir.Value
) -> ir.Value:
    """``dest`` with ``item`` stored at ``position`` along ``dim``."""
    shape = list(ir.RankedTensorType(dest.type).shape)
    offsets = [ctx.index_const(0)] * len(shape)
    offsets[dim] = position
    sizes = emit.sizes(dest)
    sizes[dim] = 1
    shape[dim] = 1
    # Not rank-reducing: that trips an MLIR assertion (areEquivalentSlices) in the opt pipeline.
    return emit.insert_slice(reshape(item, shape), dest, offsets, sizes)


def reshape(value: ir.Value, result_shape: list[int]) -> ir.Value | None:
    """``value`` reshaped without a helper, else ``None``: an ``expand_shape`` or
    ``collapse_shape`` when one applies to static shapes, else a ``tensor.reshape``;
    with runtime (``?``) dims, only unit dims may be added or removed, and the
    other dims keep their sizes (the caller knows they do)."""
    source_type = ir.RankedTensorType(value.type)
    if list(source_type.shape) == list(result_shape):
        return value
    if any(
        ir.ShapedType.is_dynamic_size(dim)
        for dim in [*source_type.shape, *result_shape]
    ):
        return _unit_dim_reshape(value, list(result_shape))
    if any(dim <= 0 for dim in result_shape) or _numel(source_type.shape) != _numel(
        result_shape
    ):
        return None
    result_type = ir.RankedTensorType.get(result_shape, source_type.element_type)
    # A tensor.reshape bufferizes to a copy of a strided source; these stay views.
    if (groups := _refining_groups(source_type.shape, result_shape)) is not None:
        return tensor_d.ExpandShapeOp(
            result_type, value, groups, [], result_shape
        ).result
    if (groups := _refining_groups(result_shape, source_type.shape)) is not None:
        return tensor_d.CollapseShapeOp(result_type, value, groups).result
    i32 = ir.IntegerType.get_signless(32)
    shape = tensor_d.FromElementsOp(
        ir.RankedTensorType.get([len(result_shape)], i32),
        [emit.constant(i32, dim) for dim in result_shape],
    ).result
    return tensor_d.ReshapeOp(result_type, value, shape).result


def _refining_groups(coarse: object, fine: object) -> list[list[int]] | None:
    """Groups of consecutive ``fine`` dims whose products are the ``coarse`` dims
    in order (trailing unit dims join the last group), else ``None``."""
    coarse, fine = [int(dim) for dim in coarse], [int(dim) for dim in fine]
    if not coarse or len(fine) <= len(coarse):
        return None
    groups: list[list[int]] = []
    position = 0
    for dim in coarse:
        group, product = [], 1
        while position < len(fine) and (not group or product < dim):
            group.append(position)
            product *= fine[position]
            position += 1
        if product != dim:
            return None
        groups.append(group)
    while position < len(fine) and fine[position] == 1:
        groups[-1].append(position)
        position += 1
    return groups if position == len(fine) else None


def _unit_dim_reshape(value: ir.Value, result_shape: list[int]) -> ir.Value | None:
    """Collapse ``value``'s unit dims, then expand to ``result_shape``'s, if the
    other dims agree in order."""
    source_type = ir.RankedTensorType(value.type)
    source_shape = list(source_type.shape)
    core = [dim for dim in source_shape if dim != 1]
    if core != [dim for dim in result_shape if dim != 1]:
        return None
    element_type = source_type.element_type
    core_sizes = [
        size
        for size, dim in zip(emit.sizes(value), source_shape, strict=True)
        if dim != 1
    ]
    if len(core) != len(source_shape):
        value = tensor_d.CollapseShapeOp(
            ir.RankedTensorType.get(core, element_type),
            value,
            _unit_groups(source_shape),
        ).result
    if len(core) == len(result_shape):
        return value
    remaining = iter(core_sizes)
    output = [1 if dim == 1 else next(remaining) for dim in result_shape]
    return tensor_d.ExpandShapeOp(
        ir.RankedTensorType.get(result_shape, element_type),
        value,
        _unit_groups(result_shape),
        [size for size in output if isinstance(size, ir.Value)],
        result_shape,
    ).result


def _unit_groups(shape: list[int]) -> list[list[int]]:
    return emit.reassociation(
        len(shape), [dim for dim, size in enumerate(shape) if size == 1]
    )


def _numel(shape: object) -> int:
    product = 1
    for dim in shape:
        product *= int(dim)
    return product
