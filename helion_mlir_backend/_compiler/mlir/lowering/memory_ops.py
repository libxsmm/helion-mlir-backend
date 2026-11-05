"""Memory-related MLIR lowering helpers."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
import helion.language.memory_ops as memory_ops
from mlir.dialects import linalg as linalg_d
import mlir.ir as ir
import torch

from ..analysis.tensor_effects import host_tensor_name
from ..aten_bridge import call_helper
from ..support import NodeLoweringError
from ..support import UnsupportedOperationError
from ..support import ValueNotFoundError
from . import emit
from .load_slice_ops import load_tile
from .registry import lowers
from .slice_plan import SlicePlan
from .slice_plan import plan_slice
from .view_ops import reshape

if TYPE_CHECKING:
    from ..build_context import BuildContext


@lowers(operator.getitem)
def lower_getitem(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """One result of a loop, ``if`` or ``split`` (:class:`emit.Results`), of a
    multi-result helper call, or of a multi-result ``inline_mlir``."""
    container_value = ctx.get_value(node.args[0])
    if container_value is None:
        # An item of an Inductor-internal buffer (``_inductor_lowering_extra``).
        return None
    return container_value.results[int(node.args[1])]


@lowers(tracing_ops._mask_to)
def lower_mask_to(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """``_mask_to(x, other)``: ``other`` outside the loop along each tile dim of ``x``.

    A dim belongs to a tile through its size's symbol; only loops with a partial
    last tile need a mask.
    """
    source, other = node.args
    value = ctx.get_value(source)
    if value is None:
        # A mask of an Inductor-internal buffer (``_inductor_lowering_extra``).
        return None
    bounds = {}
    for dim, size in enumerate(source.meta["val"].shape):
        block_id = ctx.env.resolve_block_id(size)
        valid = ctx.block_id_to_valid.get(block_id)
        if valid is not None:
            bounds[dim] = ctx.as_index(valid)
    return emit.mask(value, bounds, other) if bounds else value


@lowers(tracing_ops._inductor_lowering_extra)
def lower_inductor_extra(ctx: BuildContext, node: torch.fx.Node) -> None:
    """No value: an Inductor intermediate buffer of an ATen op (e.g. the sum of
    ``mean``) that only feeds that op's ``_extra_args``; the op's helper
    recomputes it."""
    return None


@lowers(memory_ops.store)
def lower_store(ctx: BuildContext, node: torch.fx.Node) -> None:
    """``hl.store(tensor, index, value, extra_mask)``: an ``insert_slice`` into the
    tensor's current value, of the tile's real part where ``extra_mask`` holds."""
    name = host_tensor_name(node.args[0])
    if name is None or name not in ctx.tensors:
        raise NodeLoweringError(node, reason="the store target is not a host tensor")
    index_nodes, value_node = node.args[1], node.args[2]
    extra_mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("extra_mask")
    state = ctx.tensors.value(name)
    plan = plan_slice(
        ctx,
        index_nodes,
        state,
        ctx.tensors.owned(name),
        name,
        masked=extra_mask is not None,
    )
    if plan.gathers():
        raise UnsupportedOperationError(
            "store", reason="stores indexed by a tensor (scatter) are not supported"
        )
    element_type = ir.RankedTensorType(state.type).element_type
    value = ctx.get_value(value_node)
    if value is None:
        raise ValueNotFoundError(value_node, context="stored value")
    if extra_mask is None and _broadcasts(value, plan):
        # Broadcast straight into the destination slice, never as its own tile.
        kept = [i for i, dim in enumerate(plan.dims) if not dim.reduces]
        dest = emit.extract_slice(state, plan.offsets(), plan.sizes())
        stored = _broadcast(_real_part(value, plan), dest, kept)
        ctx.tensors.rebind(
            name, emit.insert_slice(stored, state, plan.offsets(), plan.sizes())
        )
        return
    value = _store_value(value, element_type, plan)
    if extra_mask is not None or plan.is_partial():
        value = reshape(value, plan.value_shape())
    if extra_mask is not None:
        current = load_tile(state, plan)
        value = call_helper(
            ctx, node, torch.ops.aten.where.self, (extra_mask, value, current), {}
        )
    if plan.is_partial():
        zeros = [ctx.index_const(0)] * len(plan.value_shape())
        value = emit.extract_slice(value, zeros, plan.value_sizes())
    ctx.tensors.rebind(
        name, emit.insert_slice(value, state, plan.offsets(), plan.sizes())
    )


def _real_part(value: ir.Value, plan: SlicePlan) -> ir.Value:
    """A stored tile without its padding, in its dims that aren't broadcast: the
    destination slice of a partial tile is shorter than they are."""
    if not plan.is_partial():
        return value
    sizes = [
        size if extent != 1 else 1
        for size, extent in zip(
            plan.value_sizes(), ir.RankedTensorType(value.type).shape, strict=True
        )
    ]
    zeros = [emit.constant(ir.IndexType.get(), 0)] * len(sizes)
    return emit.extract_slice(value, zeros, sizes)


def _store_value(value: ir.Value, element_type: ir.Type, plan: SlicePlan) -> ir.Value:
    """The stored value as a tile of the plan's shape (with or without the reduced
    dims) and the destination dtype."""
    shape = plan.value_shape()
    if not isinstance(value.type, ir.RankedTensorType):
        scalar = emit.cast_scalar(value, element_type)
        return linalg_d.fill(
            scalar, outs=[emit.empty(plan.value_tile_sizes(), element_type)]
        )
    if _broadcasts(value, plan):
        return _broadcast(
            value,
            emit.empty(plan.value_tile_sizes(), element_type),
            list(range(len(shape))),
        )
    value_shape = list(value.type.shape)
    if value_shape not in (shape, plan.tile_shape()):
        raise UnsupportedOperationError(
            "store with transposed or mismatched tile layout",
            reason=(
                f"storing a tile of shape {value_shape} into a slice of shape "
                f"{shape}; the stored value's tile order does not "
                "match the order the destination is indexed"
            ),
            alternatives=[
                "reorder explicitly, e.g. out[a, b] = src[b, a].permute(1, 0)",
                "index the destination in the same order the value is loaded",
            ],
        )
    return emit.cast_tensor(value, element_type)


def _broadcasts(value: ir.Value, plan: SlicePlan) -> bool:
    """A tile of the stored rank with size 1 where the slice is wider (Helion
    broadcasts it, like ``tl.store``)."""
    if not isinstance(value.type, ir.RankedTensorType):
        return False
    shape, target = list(value.type.shape), plan.value_shape()
    return (
        len(shape) == len(target)
        and shape != target
        and all(size in (1, extent) for size, extent in zip(shape, target, strict=True))
    )


def _broadcast(value: ir.Value, dest: ir.Value, dest_dims: list[int]) -> ir.Value:
    """``dest`` overwritten by ``value`` broadcast along its size-1 dims and cast to
    ``dest``'s dtype; value dim ``j`` is ``dest`` dim ``dest_dims[j]``."""
    dest_type = ir.RankedTensorType(dest.type)
    rank = dest_type.rank
    value_map = ir.AffineMap.get(
        rank,
        0,
        [
            ir.AffineConstantExpr.get(0) if size == 1 else ir.AffineDimExpr.get(dim)
            for size, dim in zip(
                ir.RankedTensorType(value.type).shape, dest_dims, strict=True
            )
        ],
    )
    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>")
    generic = linalg_d.GenericOp(
        [dest_type],
        [value],
        [dest],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(value_map),
                ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank)),
            ]
        ),
        ir.ArrayAttr.get([parallel] * rank),
    )
    element_type = dest_type.element_type
    body = generic.regions[0].blocks.append(
        ir.RankedTensorType(value.type).element_type, element_type
    )
    with ir.InsertionPoint(body):
        linalg_d.YieldOp([emit.cast_scalar(body.arguments[0], element_type)])
    return generic.result
