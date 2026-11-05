"""Loops: one ``scf.forall`` (or a sequential ``scf.for`` nest) per root graph,
one ``scf.for`` per block id of a nested ``_for_loop``.

Written host tensors are threaded as SSA values (``tensor_state``): a forall
iteration works on its owned region of each written tensor and inserts it back
in the terminator; an ``scf.for`` carries the tensors its body writes.
"""

from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING

from helion.language import memory_ops as hl_memory
import helion.language._tracing_ops as tracing_ops
from mlir.dialects import affine as affine_d
from mlir.dialects import arith as arith_d
from mlir.dialects import scf as scf_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import torch

from ..analysis.geometry import LOOP_TARGETS
from ..analysis.geometry import LoopBounds
from ..analysis.geometry import loop_block_ids
from ..analysis.tensor_effects import accessed_tensor
from ..support import NodeLoweringError
from ..support import UnsupportedOperationError
from . import emit
from .control_flow import carried_positions
from .control_flow import lower_subgraph
from .control_flow import with_carried
from .registry import COUNTING_REDUCTIONS
from .registry import lowers
from .slice_plan import tile_window
from .tensor_state import OwnedDim
from .tensor_state import owned_dims
from .tile_index_ops import scalar_tile_value

if TYPE_CHECKING:
    from ..build_context import BuildContext

log = logging.getLogger(__name__)


def build_phase_body(
    ctx: BuildContext, root_positions: list[int], outputs: tuple[str, ...]
) -> list[ir.Value]:
    """Lower the root graphs at ``root_positions`` in order; return ``outputs``' values.

    Every written host tensor (``outputs``) starts as its function argument.
    """
    device_ir = ctx.host_function.device_ir
    for name in outputs:
        ctx.tensors.bind(name, ctx.param_to_value[name])
    for position in root_positions:
        _lower_root(
            ctx, device_ir.root_ids[position], list(device_ir.grid_block_ids[position])
        )
    return [ctx.tensors.value(name) for name in outputs]


_SUMS = (torch.ops.aten.sum.dim_IntList, torch.ops.aten.sum.default)
_KEEP_ZERO_AFTER = (
    tracing_ops._phi,
    operator.getitem,
    tracing_ops._mask_to,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.neg.default,
)
"""Ops after a loop that leave a padded 0 at 0."""
_SAME_ELEMENTS = (
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.clone.default,
    torch.ops.prims.convert_element_type.default,
)
"""Ops that move or cast elements: a 0 stays a 0."""


def _lower_root(ctx: BuildContext, graph_id: int, grid_ids: list[int]) -> None:
    """Parallel ``scf.forall`` if each iteration writes disjoint regions, else sequential."""
    if len(set(grid_ids)) != len(grid_ids):
        raise UnsupportedOperationError(
            "nested tile loops of one block size",
            reason=(
                "both loops have one block id, from one block size or from one "
                "bounded by the other: their tiles would index with the inner "
                "loop's position"
            ),
            alternatives=[
                (
                    "hl.register_block_size for each loop, bounded by a tensor's size, "
                    "not by another block size"
                )
            ],
        )
    geometry = ctx.geometry
    bounds = [
        LoopBounds(ctx.sizes.value(bound.begin), ctx.sizes.value(bound.end), bound.step)
        for bound in (geometry.root_bounds[block_id] for block_id in grid_ids)
    ]
    trips = [_trip_count(bound) for bound in bounds]

    names = list(ctx.effects.writes(graph_id))
    accesses = ctx.effects.accesses(graph_id)
    owned = {
        name: owned_dims(
            ctx,
            [node for node in accesses if accessed_tensor(node) == name],
            ir.RankedTensorType(ctx.tensors.value(name).type).rank,
            grid_ids,
        )
        for name in names
    }
    unowned = [
        (name, block_id)
        for name in names
        for block_id in grid_ids
        if block_id not in {dim.block_id for dim in owned[name].values()}
    ]
    graph = ctx.host_function.device_ir.graphs[graph_id].graph
    if not unowned:
        _emit_forall(ctx, graph, grid_ids, bounds, trips, names, owned)
        return
    log.debug(
        "root graph %d runs sequentially: %s is not partitioned by block_id %d",
        graph_id,
        *unowned[0],
    )
    _emit_sequential(ctx, graph, grid_ids, bounds, trips, names)


def _trip_count(bound: LoopBounds) -> int | ir.Value:
    """``ceildiv(end - begin, step)``, static when both bounds are."""
    if isinstance(bound.begin, int) and isinstance(bound.end, int):
        return -(-(bound.end - bound.begin) // bound.step)
    operands: list[ir.Value] = []
    exprs = []
    for value in (bound.end, bound.begin):
        if isinstance(value, ir.Value):
            exprs.append(ir.AffineSymbolExpr.get(len(operands)))
            operands.append(value)
        else:
            exprs.append(ir.AffineConstantExpr.get(value))
    end, begin = exprs
    expr = ir.AffineExpr.get_ceil_div(
        end - begin, ir.AffineConstantExpr.get(bound.step)
    )
    return affine_d.AffineApplyOp(
        ir.AffineMap.get(0, len(operands), [expr]), operands
    ).result


def _emit_forall(
    ctx: BuildContext,
    graph: torch.fx.Graph,
    grid_ids: list[int],
    bounds: list[LoopBounds],
    trips: list[int | ir.Value],
    names: list[str],
    owned: dict[str, dict[int, OwnedDim]],
) -> None:
    rank = len(grid_ids)
    forall = scf_d.ForallOp(
        [0] * rank,
        trips,
        [1] * rank,
        shared_outs=[ctx.tensors.value(name) for name in names],
    )
    with ir.InsertionPoint(forall.body):
        for block_id, bound, trip_iv in zip(
            grid_ids, bounds, forall.induction_variables, strict=True
        ):
            _bind_grid_iv(ctx, block_id, bound, trip_iv)
        shared = dict(zip(names, forall.inner_iter_args, strict=True))
        regions = {
            name: _owned_region(ctx, name, shared[name], owned[name]) for name in names
        }
        region_values = {
            name: emit.extract_slice(shared[name], *regions[name]) for name in names
        }
        partial = [
            block_id for block_id in grid_ids if block_id in ctx.block_id_to_valid
        ]
        values = _lower_tile(ctx, graph, region_values, owned, partial)
        in_parallel = scf_d.InParallelOp()
        with ir.InsertionPoint(in_parallel.block):
            for name, value in zip(names, values, strict=True):
                emit.parallel_insert_slice(value, shared[name], *regions[name])
    for name, result in zip(names, forall.results, strict=True):
        ctx.tensors.bind(name, result)


def _lower_tile(
    ctx: BuildContext,
    graph: torch.fx.Graph,
    region_values: dict[str, ir.Value],
    owned: dict[str, dict[int, OwnedDim]],
    partial: list[int],
) -> list[ir.Value]:
    """Lower one forall iteration on ``region_values``, the regions it owns of
    the tensors it writes; their final values.

    A tile that may be partial along ``partial`` grid dims is versioned on being
    full, one dim at a time: a full tile is lowered with static sizes (no padded
    loads or masked stores), only a partial one with Helion's tile mask. Every
    version works on the same region values, so they stay in place.
    """
    if not partial:
        for name, value in region_values.items():
            ctx.tensors.bind(name, value, owned[name])
        ctx.lower_graph(graph)
        return [ctx.tensors.value(name) for name in region_values]
    block_id, rest = partial[0], partial[1:]
    valid = ctx.block_id_to_valid[block_id]
    tile = ctx.geometry.tile_extent(block_id)
    is_full = arith_d.cmpi(arith_d.CmpIPredicate.eq, valid, ctx.index_const(tile))
    if_op = scf_d.IfOp(
        is_full, [value.type for value in region_values.values()], has_else=True
    )
    with ir.InsertionPoint(if_op.then_block):
        ctx.bind_loop(
            block_id,
            ctx.block_id_to_iv[block_id],
            ctx.block_id_to_bounds[block_id],
            span=tile,
        )
        # The full version's regions have the tile's static extent: its stores
        # then overwrite them whole and bufferize in place.
        full_values = {
            name: _cast_dims(value, owned[name], block_id, tile, valid)
            for name, value in region_values.items()
        }
        results = _lower_tile(ctx, graph, full_values, owned, rest)
        scf_d.YieldOp(
            [
                tensor_d.CastOp(value.type, result).result
                if result.type != value.type
                else result
                for result, value in zip(results, region_values.values(), strict=True)
            ]
        )
    ctx.block_id_to_valid[block_id] = valid
    with ir.InsertionPoint(if_op.else_block):
        scf_d.YieldOp(_lower_tile(ctx, graph, region_values, owned, rest))
    return list(if_op.results)


def _cast_dims(
    value: ir.Value,
    owned: dict[int, OwnedDim],
    block_id: int,
    extent: int,
    valid: ir.Value,
) -> ir.Value:
    """``value`` (a region) cast to the static ``extent`` along its dynamic dims
    that tile ``block_id`` and whose size is the tile's ``valid`` size itself
    (``value`` itself if there are none).

    A full tile (``valid`` = ``extent``) fills exactly those dims. A dim whose
    size is clamped to the tensor (an output shorter than the loop: the region
    is ``min(valid, size - offset)``) may be shorter even then, and is left
    dynamic: cast, its store would write past the tensor.
    """
    value_type = ir.RankedTensorType(value.type)
    shape = list(value_type.shape)
    for dim, owner in owned.items():
        if (
            owner.block_id == block_id
            and not owner.point
            and shape[dim] == ir.ShapedType.get_dynamic_size()
            and _region_size(value, dim) == valid
        ):
            shape[dim] = extent
    if shape == list(value_type.shape):
        return value
    cast_type = ir.RankedTensorType.get(shape, value_type.element_type)
    return tensor_d.CastOp(cast_type, value).result


def _region_size(value: ir.Value, dim: int) -> ir.Value | None:
    """The dynamic size of dim ``dim`` of the ``tensor.extract_slice`` that
    ``value`` is (through ``tensor.cast``s), ``None`` if it is static or
    ``value`` is no such slice."""
    while isinstance(value, ir.OpResult) and value.owner.name == "tensor.cast":
        value = value.owner.operands[0]
    if not isinstance(value, ir.OpResult) or value.owner.name != "tensor.extract_slice":
        return None
    slice_op = value.owner
    static = list(ir.DenseI64ArrayAttr(slice_op.attributes["static_sizes"]))
    # A rank-reducing slice's result dims are not its size list's dims.
    if len(static) != ir.RankedTensorType(value.type).rank:
        return None
    if static[dim] != ir.ShapedType.get_dynamic_size():
        return None
    # operandSegmentSizes: source, offsets, sizes, strides.
    segments = list(ir.DenseI32ArrayAttr(slice_op.attributes["operandSegmentSizes"]))
    start = sum(segments[:2])
    sizes = list(slice_op.operands[start : start + segments[2]])
    dynamic_index = sum(
        1 for size in static[:dim] if size == ir.ShapedType.get_dynamic_size()
    )
    return sizes[dynamic_index]


def _emit_sequential(
    ctx: BuildContext,
    graph: torch.fx.Graph,
    grid_ids: list[int],
    bounds: list[LoopBounds],
    trips: list[int | ir.Value],
    names: list[str],
    level: int = 0,
) -> None:
    """One ``scf.for`` per grid dim carrying the full written tensors."""
    if level == len(grid_ids):
        ctx.lower_graph(graph)
        return
    for_op = scf_d.ForOp(
        ctx.index_const(0),
        ctx.as_index(trips[level]),
        ctx.index_const(1),
        iter_args=[ctx.tensors.value(name) for name in names],
    )
    with ir.InsertionPoint(for_op.body):
        _bind_grid_iv(ctx, grid_ids[level], bounds[level], for_op.induction_variable)
        for name, arg in zip(names, for_op.inner_iter_args, strict=True):
            ctx.tensors.rebind(name, arg)
        _emit_sequential(ctx, graph, grid_ids, bounds, trips, names, level + 1)
        scf_d.YieldOp([ctx.tensors.value(name) for name in names])
    for name, result in zip(names, for_op.results, strict=True):
        ctx.tensors.rebind(name, result)


def _bind_grid_iv(
    ctx: BuildContext, block_id: int, bound: LoopBounds, trip_iv: ir.Value
) -> None:
    ctx.block_id_to_trip_iv[block_id] = trip_iv
    ctx.bind_loop(
        block_id,
        _tile_offset(trip_iv, bound.begin, bound.step),
        (bound.begin, bound.end),
    )


def _tile_offset(trip_iv: ir.Value, begin: int | ir.Value, step: int) -> ir.Value:
    """Absolute tile offset ``begin + trip_iv * step`` for a normalized loop IV."""
    if isinstance(begin, int) and begin == 0 and step == 1:
        return trip_iv
    scaled = ir.AffineExpr.get_mul(
        ir.AffineDimExpr.get(0), ir.AffineConstantExpr.get(step)
    )
    if isinstance(begin, ir.Value):
        expr = ir.AffineExpr.get_add(scaled, ir.AffineSymbolExpr.get(0))
        return affine_d.AffineApplyOp(
            ir.AffineMap.get(1, 1, [expr]), [trip_iv, begin]
        ).result
    expr = ir.AffineExpr.get_add(scaled, ir.AffineConstantExpr.get(begin))
    return affine_d.AffineApplyOp(ir.AffineMap.get(1, 0, [expr]), [trip_iv]).result


def _owned_region(
    ctx: BuildContext, name: str, tensor: ir.Value, owned: dict[int, OwnedDim]
) -> tuple[list[ir.Value], list[emit.Size]]:
    """Offsets and sizes of one iteration's region of host tensor ``name``: the
    real part of its tile along owned dims, everything elsewhere."""
    offsets: list[ir.Value] = []
    sizes: list[emit.Size] = []
    for dim in range(ir.RankedTensorType(tensor.type).rank):
        extent = ctx.sizes.extent(tensor, dim, name)
        owner = owned.get(dim)
        if owner is None:
            offsets.append(ctx.index_const(0))
            sizes.append(extent)
        elif owner.point:
            offsets.append(scalar_tile_value(ctx, owner.block_id, owner.kind))
            sizes.append(1)
        else:
            offset, size, _ = tile_window(ctx, owner.block_id, 0, extent)
            offsets.append(offset)
            sizes.append(size)
    return offsets, sizes


@lowers(tracing_ops._for_loop, tracing_ops._for_loop_step)
def lower_nested_for_loop(ctx: BuildContext, node: torch.fx.Node) -> emit.Results:
    """Lower a (possibly multi-dimensional) nested ``_for_loop``/``_for_loop_step``
    to one ``scf.for`` per block id.

    ``_for_loop(graph_id, begin, end, args[, step])``: the body graph's
    placeholders are ``args``. Each ``scf.for`` carries the variables the body
    assigns, then the host tensors it writes. Block ids come from the loop's
    ``ForLoopGraphInfo``.
    """
    graph_id, args = node.args[0], node.args[3]
    body = ctx.host_function.device_ir.graphs[graph_id].graph
    block_ids = loop_block_ids(ctx.host_function, node)
    bounds = ctx.geometry.loop_bounds(node, block_ids)
    inputs = [ctx.get_value(arg) for arg in args]
    positions = carried_positions(node, args, body)
    names = list(ctx.effects.writes(graph_id))
    running = _running_nodes(ctx, args, positions, body)

    def emit_loop(
        level: int,
        begin: int | ir.Value,
        end: int | ir.Value,
        carried: list[ir.Value],
    ) -> list[ir.Value]:
        # A lower bound that is a value counts from 0, with the offset in the body.
        moved = isinstance(begin, ir.Value)
        for_op = scf_d.ForOp(
            ctx.index_const(0) if moved else ctx.as_index(begin),
            arith_d.SubIOp(ctx.as_index(end), begin).result
            if moved
            else ctx.as_index(end),
            ctx.index_const(bounds[level].step),
            iter_args=[*carried, *(ctx.tensors.value(name) for name in names)],
        )
        count = len(carried)
        block_args = list(for_op.inner_iter_args)
        with ir.InsertionPoint(for_op.body):
            offset = for_op.induction_variable
            if moved:
                offset = arith_d.AddIOp(begin, offset).result
        with (
            ir.InsertionPoint(for_op.body),
            ctx.enter_for_loop(
                block_ids[level],
                offset,
                (begin, end),
                _enclosing_span(ctx, block_ids[level]),
            ),
        ):
            _refuse_unmaskable_running(ctx, block_ids[level], running, body, positions)
            _refuse_reduced_after_loop(ctx, block_ids[level], node, running, body)
            for name, arg in zip(names, block_args[count:], strict=True):
                ctx.tensors.rebind(name, arg)
            yields = emit_level(level + 1, block_args[:count])
            yields = _keep_past_tile(
                ctx, block_ids[level], yields, block_args[:count], running
            )
            scf_d.YieldOp([*yields, *(ctx.tensors.value(name) for name in names)])
        results = list(for_op.results)
        for name, result in zip(names, results[count:], strict=True):
            ctx.tensors.rebind(name, result)
        return results[:count]

    def emit_level(level: int, carried: list[ir.Value]) -> list[ir.Value]:
        if level == len(block_ids):
            return lower_subgraph(ctx, body, with_carried(inputs, positions, carried))
        begin = _resolve_loop_bound(ctx, node, bounds[level].begin)
        end = _resolve_loop_bound(ctx, node, bounds[level].end)
        step = bounds[level].step
        # Whole tiles first, as if no tile were partial, so that a running value
        # is updated in place; only the partial tile keeps elements of the old one.
        whole_end = (
            _whole_tiles_end(begin, end, step)
            if _tile_dims(ctx, block_ids[level], running)
            else end
        )
        if whole_end is not end:
            carried = emit_loop(level, begin, whole_end, carried)
            begin = whole_end
        return emit_loop(level, begin, end, carried)

    return emit.Results(emit_level(0, [inputs[position] for position in positions]))


def _running_nodes(
    ctx: BuildContext,
    args: list[torch.fx.Node],
    positions: list[int],
    body: torch.fx.Graph,
) -> list[torch.fx.Node | None]:
    """Per carried variable its arg, ``None`` if the body's new value doesn't read
    the old one: that is the last tile's value, not a running one."""
    placeholders = list(body.find_nodes(op="placeholder"))
    (outputs,) = next(iter(body.find_nodes(op="output"))).args
    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    running: list[torch.fx.Node | None] = []
    for position, output in zip(positions, outputs, strict=True):
        seen: set[torch.fx.Node] = set()
        stack = [output] if isinstance(output, torch.fx.Node) else []
        while stack:
            current = stack.pop()
            # Only its sizes, as a load's tile extents, aren't the old value.
            if (
                current not in seen
                and current.target is not torch.ops.aten.sym_size.int
            ):
                seen.add(current)
                stack.extend(_data_inputs(ctx, current))
        running.append(args[position] if placeholders[position] in seen else None)
    return running


def _is_loop(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in LOOP_TARGETS


def _data_inputs(ctx: BuildContext, node: torch.fx.Node) -> list[torch.fx.Node]:
    """``node``'s inputs, without a loop's args its body only takes the sizes of."""
    if node.op == "call_function" and node.target is tracing_ops._phi:
        before, after = node.args[0], node.args[1]
        if isinstance(after, torch.fx.Node) and _is_loop(after.args[0]):
            # The value a nested loop assigns: "before" only if it has no trip.
            return [after]
        return [before, after]
    if not _is_loop(node):
        return list(node.all_input_nodes)
    inner = ctx.host_function.device_ir.graphs[node.args[0]].graph
    placeholders = list(inner.find_nodes(op="placeholder"))
    skipped = {
        arg
        for arg, placeholder in zip(node.args[3], placeholders, strict=False)
        if not _reads_data(ctx, placeholder)
    }
    return [value for value in node.all_input_nodes if value not in skipped]


def _reads_data(ctx: BuildContext, placeholder: torch.fx.Node) -> bool:
    """A value of ``placeholder`` is read, not just its sizes."""
    for user in placeholder.users:
        if user.target is torch.ops.aten.sym_size.int:
            continue
        if not _is_loop(user):
            return True
        inner = ctx.host_function.device_ir.graphs[user.args[0]].graph
        for arg, inner_placeholder in zip(
            user.args[3], inner.find_nodes(op="placeholder"), strict=False
        ):
            if arg is placeholder and _reads_data(ctx, inner_placeholder):
                return True
    return False


def _refuse_unmaskable_running(
    ctx: BuildContext,
    block_id: int,
    running: list[torch.fx.Node | None],
    body: torch.fx.Graph,
    positions: list[int],
) -> None:
    """Refuse a running value a partial tile leaves wrong, silently."""
    if block_id not in ctx.block_id_to_valid:
        return
    placeholders = list(body.find_nodes(op="placeholder"))
    tile = ctx.geometry.tile_extent(block_id)
    for position, node in zip(positions, running, strict=True):
        if node is None or not isinstance(node.meta["val"], torch.Tensor):
            continue
        shape = node.meta["val"].shape
        if any(
            isinstance(size, int) and size == tile for size in shape
        ) and not _tile_has_symbol(ctx, block_id, body):
            raise UnsupportedOperationError(
                "a running value sized by a number over a partial tile",
                reason=(
                    f"its dim of {tile} isn't tied to the loop's tile, so the "
                    "tile's padding can't be kept out of it"
                ),
                alternatives=[
                    "the size from hl.register_block_size, for both",
                    "block sizes that divide the sizes they tile, with static_shapes=True",
                ],
            )
        if not any(ctx.env.resolve_block_id(size) == block_id for size in shape):
            continue
        if _masked_inside(ctx, placeholders[position], body):
            raise UnsupportedOperationError(
                "a running value reduced inside a loop with a partial tile",
                reason=(
                    "the tile's mask would hide its elements past the partial "
                    "tile, which hold earlier tiles' values"
                ),
                alternatives=[
                    "the reduction after the loop",
                    "block sizes that divide the sizes they tile, with static_shapes=True",
                ],
            )


def _refuse_reduced_after_loop(
    ctx: BuildContext,
    block_id: int,
    loop: torch.fx.Node,
    running: list[torch.fx.Node | None],
    body: torch.fx.Graph,
) -> None:
    """Refuse a use, after a loop with a partial tile, of the value of its last tile
    that reduces or stores it or that a sibling loop reads: nothing masks the padding
    there. Only a direct sum or store of padding provably still 0 is allowed; others
    that happen to be right (``x * exp(-x)``, a sum in a sibling loop) are refused."""
    if block_id not in ctx.block_id_to_valid:
        return
    (outputs,) = next(iter(body.find_nodes(op="output"))).args
    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    for index, (node, output) in enumerate(zip(running, outputs, strict=True)):
        if node is None and not all(
            kept and keeps_zero(ctx, output)
            for kept in _uses_after_loop(ctx, block_id, loop, index)
        ):
            raise UnsupportedOperationError(
                "a use after a loop of its last partial tile's value",
                reason="the loop's tile isn't active there, so its padding counts",
                alternatives=[
                    "the reduction or store inside the loop",
                    (
                        "block sizes that divide the sizes they tile, with "
                        "static_shapes=True"
                    ),
                ],
            )


def _uses_after_loop(
    ctx: BuildContext, block_id: int, loop: torch.fx.Node, index: int
) -> list[bool]:
    """For the uses that end the loop's result ``index`` (reductions over the tile's
    dim, stores), through enclosing loops too: whether padding that was 0 still counts
    as 0 there."""
    uses: list[bool] = []
    stack = [(user, True) for user in _result_users(loop, index)]
    seen: set[tuple[torch.fx.Node, bool]] = set()
    while stack:
        current, clean = stack.pop()
        if (current, clean) in seen:
            continue
        seen.add((current, clean))
        for user in current.users:
            if user.op == "output":
                (outputs,) = user.args
                outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                enclosing = _enclosing_loop(ctx, user.graph)
                if enclosing is not None and current in outputs:
                    stack.extend(
                        (next_user, clean)
                        for next_user in _result_users(
                            enclosing, outputs.index(current)
                        )
                    )
            elif user.target is torch.ops.aten.sym_size.int:
                continue
            elif _is_loop(user):
                inner = ctx.host_function.device_ir.graphs[user.args[0]].graph
                if any(
                    arg is current and _reads_data(ctx, placeholder)
                    for arg, placeholder in zip(
                        user.args[3], inner.find_nodes(op="placeholder"), strict=False
                    )
                ):
                    uses.append(False)
            elif any(
                reader.target in COUNTING_REDUCTIONS for reader in (user, *user.users)
            ):
                uses.append(False)
            elif user.target is hl_memory.store:
                uses.append(clean)
            elif current.target is tracing_ops._mask_to:
                kept = clean and current.args[1] == 0 and user.target in _SUMS
                if _has_tile_dim(ctx, block_id, user):
                    # It reduces another dim: the tile's is still there.
                    stack.append((user, kept))
                else:
                    uses.append(kept)
            elif user.target is tracing_ops._mask_to or (
                _has_tile_dim(ctx, block_id, user)
            ):
                kept = user.target in (*_KEEP_ZERO_AFTER, *_SUMS)
                stack.append((user, clean and kept))
            else:
                uses.append(clean and user.target in _SUMS)
    return uses


def _result_users(loop: torch.fx.Node, index: int) -> list[torch.fx.Node]:
    return [
        user
        for user in loop.users
        if user.target is operator.getitem and user.args[1] == index
    ]


def _enclosing_loop(ctx: BuildContext, body: torch.fx.Graph) -> torch.fx.Node | None:
    """The loop whose body is ``body``."""
    graphs = ctx.host_function.device_ir.graphs
    for info in graphs.values() if isinstance(graphs, dict) else graphs:
        for node in info.graph.nodes:
            if _is_loop(node) and graphs[node.args[0]].graph is body:
                return node
    return None


def _has_tile_dim(ctx: BuildContext, block_id: int, node: torch.fx.Node) -> bool:
    value = node.meta.get("val")
    return isinstance(value, torch.Tensor) and any(
        ctx.env.resolve_block_id(size) == block_id for size in value.shape
    )


def keeps_zero(ctx: BuildContext, node: object) -> bool:
    """``node``, in a loop's body, is 0 where the tiles it loads are padded."""
    if not isinstance(node, torch.fx.Node):
        return False
    if node.target is hl_memory.load:
        return True
    if node.op == "placeholder":
        enclosing = _enclosing_loop(ctx, node.graph)
        if enclosing is None:
            return False
        position = list(node.graph.find_nodes(op="placeholder")).index(node)
        return keeps_zero(ctx, enclosing.args[3][position])
    if node.target is tracing_ops._mask_to:
        return node.args[1] == 0
    if node.target in _SAME_ELEMENTS:
        return keeps_zero(ctx, node.args[0])
    if node.target is tracing_ops._phi:
        return keeps_zero(ctx, node.args[1])
    if node.target is operator.getitem and _is_loop(node.args[0]):
        inner = ctx.host_function.device_ir.graphs[node.args[0].args[0]].graph
        (outputs,) = next(iter(inner.find_nodes(op="output"))).args
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        return keeps_zero(ctx, outputs[node.args[1]])
    operands = [keeps_zero(ctx, arg) for arg in node.args]
    if node.target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar):
        # 0 times a value computed from padding may not be 0: 0 / 0, 0 * inf.
        return any(operands) and all(
            kept or not isinstance(arg, torch.fx.Node)
            for kept, arg in zip(operands, node.args, strict=False)
        )
    if node.target in (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.neg.default,
    ):
        return all(operands)
    return False


def _tile_has_symbol(ctx: BuildContext, block_id: int, body: torch.fx.Graph) -> bool:
    """Some value in ``body`` has a dim of the loop's tile by its block size's symbol:
    numbers equal to the tile are other dims (a head dim). A loop with a number as
    its block size has no such value."""
    for node in body.nodes:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and any(
            ctx.env.resolve_block_id(size) == block_id for size in value.shape
        ):
            return True
    return False


def _masked_inside(
    ctx: BuildContext, placeholder: torch.fx.Node, body: torch.fx.Graph
) -> bool:
    """A value computed from ``placeholder`` reaches a ``_mask_to`` in ``body`` or
    in a loop nested in it."""
    reached = {placeholder}
    for node in body.nodes:
        if node.target is torch.ops.aten.sym_size.int:
            continue
        if any(value in reached for value in node.all_input_nodes):
            if node.target is tracing_ops._mask_to:
                return True
            if _is_loop(node):
                inner = ctx.host_function.device_ir.graphs[node.args[0]].graph
                for arg, inner_arg in zip(
                    node.args[3], inner.find_nodes(op="placeholder"), strict=False
                ):
                    if arg in reached and _masked_inside(ctx, inner_arg, inner):
                        return True
            reached.add(node)
    return False


def _tile_dims(
    ctx: BuildContext, block_id: int, nodes: list[torch.fx.Node | None]
) -> list[list[int]]:
    """Per running value in ``nodes``, its dims in the loop's tile, where it has
    any; nothing if the loop has no such value."""
    dims = [
        [
            dim
            for dim, size in enumerate(node.meta["val"].shape)
            if ctx.env.resolve_block_id(size) == block_id
        ]
        if node is not None and isinstance(node.meta["val"], torch.Tensor)
        else []
        for node in nodes
    ]
    return dims if any(dims) else []


def _whole_tiles_end(
    begin: int | ir.Value, end: int | ir.Value, step: int
) -> int | ir.Value:
    """Where the loop's whole tiles end: ``end`` itself unless both bounds are static
    and the last tile is partial. Runtime bounds stay one loop: a loop with no trip
    has to stay so, and ``tile.count`` counts that loop's tiles."""
    if isinstance(begin, int) and isinstance(end, int) and begin < end:
        whole = begin + (end - begin) // step * step
        return end if whole == end else whole
    return end


def _keep_past_tile(
    ctx: BuildContext,
    block_id: int,
    yields: list[ir.Value],
    carried: list[ir.Value],
    nodes: list[torch.fx.Node | None],
) -> list[ir.Value]:
    """A running value (of the loop's body, reading its old value) with a dim of the
    loop's tile keeps its elements past a partial tile's end: an iteration has no
    elements there, so padding must not enter it (``top = maximum(top, x[tile])``).
    Outside the loop nothing masks it, as the loop's tile isn't active there.
    Without a partial tile there is no ``valid``, and nothing changes."""
    valid = ctx.block_id_to_valid.get(block_id)
    if valid is None:
        return yields
    dims = _tile_dims(ctx, block_id, nodes) or [[] for _ in nodes]
    return [
        emit.keep_past(value, old, dict.fromkeys(tile_dims, ctx.as_index(valid)))
        if tile_dims
        else value
        for value, old, tile_dims in zip(yields, carried, dims, strict=True)
    ]


def _enclosing_span(ctx: BuildContext, block_id: int) -> int | None:
    """The static width of the enclosing tile a loop over one tile's range spans,
    if that tile is full (its loop has no partial tile)."""
    enclosing = ctx.geometry.enclosing_tiles.get(block_id)
    if enclosing is None or enclosing not in ctx.block_id_to_iv:
        return None
    if enclosing in ctx.block_id_to_valid:
        return None
    return ctx.geometry.tile_extent(enclosing)


def _resolve_loop_bound(
    ctx: BuildContext,
    node: torch.fx.Node,
    source: object,
) -> int | ir.Value:
    """Resolve a ``_for_loop`` begin/end (an int or an FX node) to a static int
    or an index value.

    A runtime scalar carrying a size resolves through ``ctx.sizes``, so a loop
    over a tensor's extent shares that extent's value.
    """
    if isinstance(source, int):
        return source
    value = None
    if isinstance(source, torch.fx.Node):
        meta_val = source.meta.get("val")
        if (
            source.target is tracing_ops._get_symnode
            and source.args[0] in ctx.scalars
            and isinstance(meta_val, torch.SymInt)
        ):
            return ctx.sizes.value(meta_val)
        value = ctx.get_value(source)
    if value is None:
        raise NodeLoweringError(
            node,
            reason=f"Unsupported loop bound: {source!r}",
            recovery_hint="Ensure loop bounds are integer constants or scalar tensor values",
        )
    return ctx.cast_to_index(value)
