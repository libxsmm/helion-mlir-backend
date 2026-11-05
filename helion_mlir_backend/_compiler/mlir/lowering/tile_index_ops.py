"""Lower Helion tile-index operations and scalar shape queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
import helion.language.tile_ops as tile_ops
from mlir.dialects import arith as arith_d
from mlir.dialects import linalg as linalg_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import sympy
import torch
from torch.utils._sympy.functions import FloorDiv

from ..support import NodeLoweringError
from ..support import UnsupportedOperationError
from ..support import ValueNotFoundError
from ..support import block_id_from_key
from ..support import resolve_index_descriptor
from ..support import torch_dtype_to_mlir
from .registry import lowers

if TYPE_CHECKING:
    from ..build_context import BuildContext

_SCALAR_KINDS = {
    tile_ops.tile_begin: "tile_begin",
    tile_ops.tile_end: "tile_end",
    tile_ops.tile_id: "tile_id",
    tile_ops.tile_count: "tile_count",
    tile_ops.tile_block_size: "block_size",
}


@lowers(tracing_ops._get_symnode)
def lower_get_symnode(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``_get_symnode(key)``: a block-size constant, runtime scalar or tile position."""
    key = node.args[0]
    if key in ctx.scalars:
        return ctx.scalars[key]
    block_id = block_id_from_key(key)
    if block_id is not None:
        if block_id not in ctx.geometry.blocks:
            raise ValueNotFoundError(node, context=f"unknown block key: {key!r}")
        # The tile's size, as in every shape built from this symbol.
        return ctx.as_index(ctx.sizes.tile(block_id))
    value = node.meta.get("val")
    if isinstance(value, torch.SymInt) and not value.node.expr.free_symbols:
        value = int(value.node.expr)
    if isinstance(value, int):
        # A symbol Helion specialized to a constant.
        return ctx.index_const(value)
    # ``hl.grid`` and tile position symbols have no ``block_size_`` key; they resolve
    # to a scalar index through their Helion symbol origin.
    info = ctx.node_symbol_info(node)
    resolved = scalar_tile_value(ctx, *info) if info is not None else None
    if resolved is not None:
        return resolved
    if isinstance(value, torch.SymInt):
        resolved = _symbolic_value(ctx, value.node.expr)
        if resolved is None:
            raise UnsupportedOperationError(
                f"the tile expression {value.node.expr} in an inner loop",
                reason=(
                    "arithmetic on tile positions reaches an inner loop as an "
                    "expression; +, -, *, //, min and max of tile positions and "
                    "constants lower"
                ),
                alternatives=["compute it in the loop that uses it"],
            )
        return resolved
    raise ValueNotFoundError(node, context=f"invalid block key: {key!r}")


def _symbolic_value(ctx: BuildContext, expr: object) -> ir.Value | None:
    """``expr``'s ``index`` value: arithmetic on tile positions captured by an inner
    loop (``h = th.begin // group`` above it), which reaches the inner graph as its
    symbolic expression. ``None`` for any other expression."""
    if isinstance(expr, sympy.Integer):
        return ctx.index_const(int(expr))
    if isinstance(expr, sympy.Symbol):
        info = ctx.symbol_info(expr)
        return scalar_tile_value(ctx, *info) if info is not None else None
    if isinstance(expr, FloorDiv):
        combine, operands = arith_d.FloorDivSIOp, expr.args
    elif isinstance(expr, sympy.Add):
        combine, operands = arith_d.AddIOp, expr.args
    elif isinstance(expr, sympy.Mul):
        combine, operands = arith_d.MulIOp, expr.args
    elif isinstance(expr, sympy.Max):
        combine, operands = arith_d.MaxSIOp, expr.args
    elif isinstance(expr, sympy.Min):
        combine, operands = arith_d.MinSIOp, expr.args
    else:
        return None
    values = [_symbolic_value(ctx, operand) for operand in operands]
    if any(value is None for value in values):
        return None
    result = values[0]
    for other in values[1:]:
        result = combine(result, other).result
    return result


@lowers(torch.ops.aten.sym_size.int)
def lower_sym_size(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``sym_size.int(tensor, dim)``: the operand's dimension, a constant if static."""
    tensor, dim = node.args[:2]
    value = ctx.get_value(tensor)
    if value is None:
        raise ValueNotFoundError(tensor, context="sym_size operand")
    size = ir.RankedTensorType(value.type).shape[dim]
    if not ir.ShapedType.is_dynamic_size(size):
        return ctx.index_const(size)
    return tensor_d.DimOp(value, ctx.index_const(dim)).result


@lowers(tile_ops.tile_index)
def lower_tile_index(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """``tile.index``: ``offset + i`` for each tile position, as a ``linalg.generic``."""

    from . import emit

    if not node.args:
        return None

    block_id = resolve_index_descriptor(ctx, node.args[0]).block_id
    if block_id is None or block_id not in ctx.geometry.blocks:
        raise NodeLoweringError(node, reason="cannot resolve the tile's block id")
    base = ctx.block_id_to_iv.get(block_id)
    if base is None:
        raise NodeLoweringError(
            node, reason=f"block_id {block_id} has no active loop offset"
        )
    shape = [ctx.geometry.tile_extent(block_id)]

    element_type: ir.Type = ir.IndexType.get()
    metadata_value = node.meta.get("val")
    if isinstance(metadata_value, torch.Tensor):
        metadata_type = torch_dtype_to_mlir(metadata_value.dtype)
        if isinstance(metadata_type, (ir.IntegerType, ir.IndexType)):
            element_type = metadata_type

    generic = linalg_d.GenericOp(
        [ir.RankedTensorType.get(shape, element_type)],
        [],
        [emit.empty(shape, element_type)],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(ir.AffineMap.get_identity(1))]),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")]),
    )
    body = generic.regions[0].blocks.append(element_type)
    with ir.InsertionPoint(body):
        position = arith_d.AddIOp(base, linalg_d.IndexOp(0).result).result
        linalg_d.YieldOp([emit.cast_scalar(position, element_type)])
    return generic.result


def scalar_tile_value(ctx: BuildContext, block_id: int, kind: str) -> ir.Value | None:
    """Build the scalar ``index`` value for a grid or tile position op.

    Matches Helion: ``begin`` is the tile's absolute offset, ``end`` is clamped
    to the loop end, ``id`` is ``begin // block_size`` and ``count`` is
    ``cdiv(end - begin, block_size)``.
    """

    geometry = ctx.geometry
    if block_id not in geometry.blocks:
        return None
    block = geometry.block(block_id)
    block_size = block.block_size
    bounds = ctx.block_id_to_bounds.get(block_id)

    if kind == "block_size":
        return ctx.index_const(block_size)

    if kind == "tile_count":
        if block.span is not None:
            return ctx.index_const(geometry.trip_count(block_id))
        if bounds is None:
            return None
        extent = arith_d.SubIOp(ctx.as_index(bounds[1]), ctx.as_index(bounds[0])).result
        return arith_d.CeilDivUIOp(extent, ctx.index_const(block_size)).result

    offset = ctx.block_id_to_iv.get(block_id)
    if offset is None:
        return None

    if kind in ("grid", "tile_begin"):
        return offset

    if kind == "tile_end":
        valid = ctx.block_id_to_valid.get(block_id, ctx.sizes.tile(block_id))
        return arith_d.AddIOp(offset, ctx.as_index(valid)).result

    if kind == "tile_id":
        trip = ctx.block_id_to_trip_iv.get(block_id)
        if (
            trip is not None
            and bounds is not None
            and isinstance(bounds[0], int)
            and bounds[0] == 0
        ):
            return trip
        if block_size == 1:
            return offset
        return arith_d.DivUIOp(offset, ctx.index_const(block_size)).result

    return None


@lowers(*_SCALAR_KINDS)
def lower_tile_scalar_op(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """Lower ``tile.begin`` / ``tile.end`` / ``tile.id`` / ``tile.count``."""
    kind = _SCALAR_KINDS[node.target]
    info = ctx.node_symbol_info(node)
    if info is None:
        raise NodeLoweringError(node, reason="cannot resolve the tile's block id")
    block_id = info[0]

    value = scalar_tile_value(ctx, block_id, kind)
    if value is None:
        raise NodeLoweringError(
            node, reason=f"no active loop bounds for block_id {block_id} ({kind})"
        )
    return value
