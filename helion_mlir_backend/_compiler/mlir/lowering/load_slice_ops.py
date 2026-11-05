"""Helion tile loads: an extract slice of the tensor's current value."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language.memory_ops as memory_ops
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import torch

from ..analysis.tensor_effects import host_tensor_name
from ..aten_bridge import call_helper
from ..support import UnsupportedOperationError
from ..support import ValueNotFoundError
from . import emit
from .registry import lowers
from .view_ops import reshape

if TYPE_CHECKING:
    from ..build_context import BuildContext
    from .slice_plan import SlicePlan


@lowers(memory_ops.load)
def lower_load(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    """``hl.load(tensor, index, extra_mask)``: a tile of the tensor's current value.

    Elements outside the loop or the tensor, or where ``extra_mask`` is false, are 0.
    """
    from .slice_plan import plan_slice

    tensor_node, index_nodes = node.args[:2]
    extra_mask = _arg(node, 2, "extra_mask")
    name = host_tensor_name(tensor_node)
    if name is not None and name in ctx.tensors:
        tensor_value = ctx.tensors.value(name)
        owned = ctx.tensors.owned(name)
    else:
        tensor_value = ctx.get_value(tensor_node)
        owned = {}
    if tensor_value is None:
        raise ValueNotFoundError(tensor_node, context="loaded tensor")

    plan = plan_slice(
        ctx,
        [item for item in index_nodes if item is not None],
        tensor_value,
        owned,
        name,
        masked=extra_mask is not None,
    )
    loaded = load_tile(tensor_value, plan)
    gathers = plan.gathers()
    if gathers:
        # Helion and torch agree on the result shape for one index tensor that
        # is 1-D or indexes a 1-D tensor; they differ beyond that.
        (dimension, index), *others = gathers
        if others or (len(plan.dims) > 1 and ir.RankedTensorType(index.type).rank != 1):
            raise UnsupportedOperationError(
                "load", reason="only one 1-D index tensor per load is supported"
            )
        reduced = plan.reduced_dims()
        position = dimension - sum(1 for dim in reduced if dim < dimension)
        loaded = _gather(ctx, node, loaded, position, index)
    if None in index_nodes:
        loaded = reshape(loaded, _with_new_axes(loaded, index_nodes, plan))
    if extra_mask is None:
        return loaded
    return call_helper(
        ctx, node, torch.ops.aten.where.ScalarOther, (extra_mask, loaded, 0), {}
    )


def _gather(
    ctx: BuildContext, node: torch.fx.Node, loaded: ir.Value, dim: int, index: ir.Value
) -> ir.Value:
    """``loaded`` indexed along ``dim`` by the ``index`` tensor, via ``aten.index.Tensor``.

    The index is widened to ``i64``: torch-mlir fails on narrower ones.
    """
    wide = emit.cast_tensor(index, ir.IntegerType.get_signless(64))
    indices = [None] * dim + [wide]
    return call_helper(ctx, node, torch.ops.aten.index.Tensor, (loaded, indices), {})


def _with_new_axes(
    loaded: ir.Value, index_nodes: list[object], plan: SlicePlan
) -> list[int]:
    """The shape of ``loaded`` with a unit dim at each ``None`` of the index."""
    sizes = iter(ir.RankedTensorType(loaded.type).shape)
    dims = iter(plan.dims)
    shape = []
    for item in index_nodes:
        if item is None:
            shape.append(1)
        elif not next(dims).reduces:
            shape.append(next(sizes))
    return shape + list(sizes)


def load_tile(tensor: ir.Value, plan: SlicePlan) -> ir.Value:
    """The tile ``plan`` selects, zero-padded past the real part, reduced dims dropped.

    The extract keeps every dim; only the scalar-indexed ones are collapsed, by
    position, since letting MLIR infer dropped unit dims is ambiguous when a
    kept tile dim also has extent 1.
    """
    loaded = emit.extract_slice(tensor, plan.offsets(), plan.sizes())
    if plan.is_partial():
        loaded = emit.pad_high(loaded, plan.sizes(), plan.tile_shape())
    reduced_dims = set(plan.reduced_dims())
    if not reduced_dims:
        return loaded
    element_type = ir.RankedTensorType(tensor.type).element_type
    result_type = ir.RankedTensorType.get(plan.value_shape() or [1], element_type)
    rank = len(plan.dims)
    reassociation = emit.reassociation(rank, reduced_dims) or [list(range(rank))]
    return tensor_d.CollapseShapeOp(result_type, loaded, reassociation).result


def _arg(node: torch.fx.Node, position: int, name: str) -> object:
    if len(node.args) > position:
        return node.args[position]
    return node.kwargs.get(name)
