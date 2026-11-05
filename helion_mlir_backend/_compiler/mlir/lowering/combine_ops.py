"""``hl.reduce`` and ``hl.associative_scan`` (also ``torch.cumsum``) with a
combine function, as a sequential ``scf.for`` along the dimension.

The combine graph is lowered once per step on whole slices, which is valid
because a combine function is elementwise in its two arguments. A reduction whose
combine function is one known op of its two arguments is a ``linalg.reduce``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from helion.language import _tracing_ops as tracing_ops
from helion.language import reduce_ops
from helion.language import scan_ops
from mlir.dialects import affine as affine_d
from mlir.dialects import arith as arith_d
from mlir.dialects import linalg as linalg_d
from mlir.dialects import scf as scf_d
import mlir.ir as ir
import torch

from ..support import UnsupportedOperationError
from . import emit
from .control_flow import lower_subgraph
from .registry import lowers
from .view_ops import put
from .view_ops import reshape
from .view_ops import take

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..build_context import BuildContext

aten = torch.ops.aten

_ARITH: dict[object, tuple[type | None, type | None, type | None]] = {
    # target: (float op, signed integer op, bool op)
    aten.add.Tensor: (arith_d.AddFOp, arith_d.AddIOp, None),
    aten.mul.Tensor: (arith_d.MulFOp, arith_d.MulIOp, None),
    aten.maximum.default: (arith_d.MaximumFOp, arith_d.MaxSIOp, None),
    aten.minimum.default: (arith_d.MinimumFOp, arith_d.MinSIOp, None),
    aten.logical_and.default: (None, None, arith_d.AndIOp),
    aten.logical_or.default: (None, None, arith_d.OrIOp),
    aten.bitwise_or.Tensor: (None, arith_d.OrIOp, arith_d.OrIOp),
}

_ZERO_NEUTRAL = (arith_d.AddFOp, arith_d.AddIOp, arith_d.OrIOp)
"""Combiners padding's 0 doesn't change."""


def _partial_tile(ctx: BuildContext, source: torch.fx.Node, dim: int) -> bool:
    """``source``'s dim ``dim`` is a tile dim with a partial tile here."""
    if source.op == "call_function" and source.target is tracing_ops._mask_to:
        source = source.args[0]
    size = source.meta["val"].shape[dim]
    return ctx.block_id_to_valid.get(ctx.env.resolve_block_id(size)) is not None


def _refuse_partial_tile(ctx: BuildContext, source: torch.fx.Node, dim: int) -> None:
    """A reverse scan starts in the partial tile's padding: wrong values, silently."""
    if _partial_tile(ctx, source, dim):
        raise UnsupportedOperationError(
            "a reverse hl.associative_scan over a partial tile",
            reason="the scan would start in the tile's padding",
            alternatives=[
                "block sizes that divide the sizes they tile, with static_shapes=True"
            ],
        )


def _refuse_masked_with_zero(
    ctx: BuildContext, graph_id: int, source: torch.fx.Node, dim: int
) -> None:
    """``hl.reduce``'s ``other`` is what padded lanes hold, 0 by default, and then
    Helion leaves no mask: wrong for any combiner but a sum, silently."""
    masked = source.op == "call_function" and source.target is tracing_ops._mask_to
    if (masked and source.args[1] != 0) or not _partial_tile(ctx, source, dim):
        return
    known = _known_combiner(ctx, graph_id, source.meta["val"].dtype)
    if known is None or known[0] not in _ZERO_NEUTRAL:
        raise UnsupportedOperationError(
            "hl.reduce over a partial tile with other=0",
            reason="padded lanes hold 0, which only a sum or an OR ignores",
            alternatives=[
                "other= the combiner's identity",
                "a sum",
                "block sizes that divide the sizes they tile, with static_shapes=True",
            ],
        )


@lowers(reduce_ops._reduce)
def lower_reduce(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``_reduce(combine_graph, x, dim, keep_dims, is_tuple_input)``."""
    graph_id, source, dim, keep_dims, is_tuple = _operands(node, "hl.reduce")
    if dim is None:
        raise UnsupportedOperationError("hl.reduce", reason="dim=None")
    _refuse_masked_with_zero(ctx, graph_id, source, dim)
    value, dim, extent = _source(ctx, source, dim)
    known = _known_combiner(ctx, graph_id, source.meta["val"].dtype)
    if known is not None:
        result = _linalg_reduce(value, dim, *known)
    else:
        combine = _combine(ctx, graph_id)
        loop = scf_d.ForOp(
            ctx.index_const(1),
            ctx.as_index(extent),
            ctx.index_const(1),
            iter_args=[take(ctx, value, dim, ctx.index_const(0))],
        )
        with ir.InsertionPoint(loop.body):
            item = take(ctx, value, dim, loop.induction_variable)
            scf_d.YieldOp([combine(loop.inner_iter_args[0], item)])
        result = loop.results[0]
    if keep_dims:
        shape = list(ir.RankedTensorType(value.type).shape)
        shape[dim] = 1
        result = reshape(result, shape)
    return result


@lowers(scan_ops._associative_scan)
def lower_scan(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``_associative_scan(combine_graph, x, dim, reverse, is_tuple_input)``."""
    graph_id, source, dim, reverse, _ = _operands(node, "hl.associative_scan")
    known = _known_combiner(ctx, graph_id, source.meta["val"].dtype)
    if reverse and (known is None or known[0] not in _ZERO_NEUTRAL):
        _refuse_partial_tile(ctx, source, dim)
    value, dim, extent = _source(ctx, source, dim)
    combine = _combine(ctx, graph_id)
    value_type = ir.RankedTensorType(value.type)
    last = (
        extent - 1
        if isinstance(extent, int)
        else arith_d.SubIOp(extent, ctx.index_const(1)).result
    )
    first = ctx.as_index(last) if reverse else ctx.index_const(0)
    head = take(ctx, value, dim, first)
    out = emit.empty(emit.sizes(value), value_type.element_type)
    loop = scf_d.ForOp(
        ctx.index_const(1),
        ctx.as_index(extent),
        ctx.index_const(1),
        iter_args=[head, put(ctx, out, head, dim, first)],
    )
    with ir.InsertionPoint(loop.body):
        step = loop.induction_variable
        if reverse and isinstance(extent, int):
            d0 = ir.AffineDimExpr.get(0)
            position = affine_d.AffineApplyOp(
                ir.AffineMap.get(1, 0, [ir.AffineConstantExpr.get(extent - 1) - d0]),
                [step],
            ).result
        elif reverse:
            position = arith_d.SubIOp(last, step).result
        else:
            position = step
        acc, partial = loop.inner_iter_args
        item = take(ctx, value, dim, position)
        acc = combine(item, acc) if reverse else combine(acc, item)
        scf_d.YieldOp([acc, put(ctx, partial, acc, dim, position)])
    return loop.results[1]


def _operands(node: torch.fx.Node, name: str) -> tuple:
    defaults = (None, None, None, False, False)
    graph_id, source, dim, flag, is_tuple = (*node.args, *defaults[len(node.args) :])
    if is_tuple:
        raise UnsupportedOperationError(name, reason="tuple inputs")
    return graph_id, source, dim, flag, is_tuple


def _source(
    ctx: BuildContext, source: torch.fx.Node, dim: int
) -> tuple[ir.Value, int, emit.Size]:
    value = ctx.get_value(source)
    dim %= ir.RankedTensorType(value.type).rank
    return value, dim, emit.sizes(value)[dim]


def _combine(
    ctx: BuildContext, graph_id: int
) -> Callable[[ir.Value, ir.Value], ir.Value]:
    graph = ctx.host_function.device_ir.graphs[graph_id].graph

    def combine(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        (result,) = lower_subgraph(ctx, graph, [lhs, rhs])
        return result

    return combine


def _known_combiner(
    ctx: BuildContext, graph_id: int, dtype: torch.dtype
) -> tuple[type, float] | None:
    """The ``arith`` op and identity of a combine graph that is one known op of its
    two arguments, else ``None``."""
    nodes = list(ctx.host_function.device_ir.graphs[graph_id].graph.nodes)
    if len(nodes) != 4:
        return None
    lhs, rhs, op, output = nodes
    if (
        (lhs.op, rhs.op, op.op) != ("placeholder", "placeholder", "call_function")
        or op.target not in _ARITH
        or op.kwargs
        or len(op.args) != 2
        or set(op.args) != {lhs, rhs}
        or output.args[0] is not op
    ):
        return None
    float_op, int_op, bool_op = _ARITH[op.target]
    if dtype.is_floating_point:
        arith_op, info = float_op, None
    elif dtype == torch.bool:
        arith_op, info = bool_op, None
    elif dtype.is_signed and not dtype.is_complex:
        arith_op, info = int_op, torch.iinfo(dtype)
    else:
        return None
    if arith_op is None:
        return None
    if op.target is aten.maximum.default:
        return arith_op, -math.inf if info is None else info.min
    if op.target is aten.minimum.default:
        return arith_op, math.inf if info is None else info.max
    return arith_op, 1 if op.target in (
        aten.mul.Tensor,
        aten.logical_and.default,
    ) else 0


def _linalg_reduce(
    value: ir.Value, dim: int, arith_op: type, identity: float
) -> ir.Value:
    value_type = ir.RankedTensorType(value.type)
    element_type = value_type.element_type
    shape = [size for index, size in enumerate(emit.sizes(value)) if index != dim]
    init = emit.filled(shape, element_type, identity)
    reduce = linalg_d.ReduceOp([init.type], [value], [init], [dim])
    body = reduce.regions[0].blocks.append(element_type, element_type)
    with ir.InsertionPoint(body):
        linalg_d.YieldOp([arith_op(*body.arguments).result])
    return reduce.results[0]
