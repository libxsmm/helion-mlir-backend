"""Control flow inside a loop body: ``if`` (``scf.if``), ``while`` (``scf.while``),
``_phi``/``_new_var``, and the subgraph and loop-carried value helpers loops share.

Written host tensors are threaded as SSA values (``tensor_state``): an
``scf.if`` or ``scf.while`` carries the tensors its body writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
from mlir.dialects import scf as scf_d
import mlir.ir as ir

from ..support import NodeLoweringError
from ..support import UnsupportedOperationError
from . import emit
from .registry import lowers
from .scalar_ops import condition

if TYPE_CHECKING:
    import torch

    from ..build_context import BuildContext


@lowers(tracing_ops._new_var)
def lower_new_var(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    return ctx.get_value(node.args[0])


@lowers(tracing_ops._phi)
def lower_phi(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """``_phi(before, after)``: the loop or ``if`` result (``after``) replaces the value."""
    return ctx.get_value(node.args[1])


@lowers(tracing_ops._if)
def lower_if(ctx: BuildContext, node: torch.fx.Node) -> emit.Results:
    """``_if(test, if_graph, else_graph, if_args, else_args)`` as an ``scf.if``.

    It yields each of Helion's ``branches_outputs`` (a branch output index, or
    the unchanged outer value of a variable only the other branch assigns),
    then the tensors either branch writes. The result types are only known once
    the branches are lowered, so they are lowered into a draft ``scf.if`` first.
    ``_if``'s outputs are the then-side values, then the else-side ones: both
    are the same ``scf.if`` results.
    """
    test, if_id, else_id, if_args, else_args = node.args
    graphs = ctx.host_function.device_ir.graphs
    info = graphs[if_id]
    if info.predicate_is_tensor:
        raise UnsupportedOperationError(
            "if", reason="the condition is a tensor of more than one element"
        )
    predicate = condition(ctx, test)
    outer = dict(zip(info.if_arg_names, if_args, strict=True))
    outer.update(zip(info.else_arg_names, else_args, strict=True))
    names = list(
        dict.fromkeys([*ctx.effects.writes(if_id), *ctx.effects.writes(else_id)])
    )
    before = {name: ctx.tensors.value(name) for name in names}

    draft = scf_d.IfOp(predicate, [], has_else=True)
    branch_yields: list[list[ir.Value]] = []
    for side, (block, graph_id, args) in enumerate(
        ((draft.then_block, if_id, if_args), (draft.else_block, else_id, else_args))
    ):
        with ir.InsertionPoint(block):
            for name in names:
                ctx.tensors.rebind(name, before[name])
            outputs = _guarded_subgraph(
                ctx, graphs[graph_id].graph, [ctx.get_value(arg) for arg in args]
            )
            branch_yields.append(
                [
                    outputs[pick[side]]
                    if isinstance(pick[side], int)
                    else ctx.get_value(outer[pick[side]])
                    for pick in info.branches_outputs
                ]
                + [ctx.tensors.value(name) for name in names]
            )

    if_op = scf_d.IfOp(
        predicate, [value.type for value in branch_yields[0]], has_else=True
    )
    for draft_block, block, values in zip(
        (draft.then_block, draft.else_block),
        (if_op.then_block, if_op.else_block),
        branch_yields,
        strict=True,
    ):
        with ir.InsertionPoint(block):
            terminator = scf_d.YieldOp(values)
        for op in list(draft_block.operations):
            if op.name != "scf.yield":
                op.move_before(terminator)
    draft.operation.erase()

    results = list(if_op.results)
    count = len(info.branches_outputs)
    for name, value in zip(names, results[count:], strict=True):
        ctx.tensors.rebind(name, value)
    return emit.Results(results[:count] * 2)


@lowers(tracing_ops._while_loop)
def lower_while(ctx: BuildContext, node: torch.fx.Node) -> emit.Results:
    """``_while_loop(cond_graph, body_graph, args)`` as an ``scf.while`` carrying
    the variables the body assigns, then the tensors the loop writes."""
    cond_id, body_id, args, orelse_id = node.args
    if orelse_id is not None:
        raise UnsupportedOperationError("while", reason="`else` clause")
    graphs = ctx.host_function.device_ir.graphs
    names = list(
        dict.fromkeys([*ctx.effects.writes(cond_id), *ctx.effects.writes(body_id)])
    )
    inputs = [ctx.get_value(arg) for arg in args]
    positions = carried_positions(node, args, graphs[body_id].graph)
    inits = [inputs[position] for position in positions]
    inits += [ctx.tensors.value(name) for name in names]
    types = [value.type for value in inits]
    while_op = scf_d.WhileOp(types, inits)
    count = len(positions)

    before = while_op.before.blocks.append(*types)
    with ir.InsertionPoint(before):
        carried = list(before.arguments)
        for name, value in zip(names, carried[count:], strict=True):
            ctx.tensors.rebind(name, value)
        (test,) = lower_subgraph(
            ctx,
            graphs[cond_id].graph,
            with_carried(inputs, positions, carried[:count]),
        )
        predicate = condition(ctx, test)
        scf_d.ConditionOp(
            predicate, carried[:count] + [ctx.tensors.value(n) for n in names]
        )

    after = while_op.after.blocks.append(*types)
    with ir.InsertionPoint(after):
        carried = list(after.arguments)
        for name, value in zip(names, carried[count:], strict=True):
            ctx.tensors.rebind(name, value)
        outputs = _guarded_subgraph(
            ctx,
            graphs[body_id].graph,
            with_carried(inputs, positions, carried[:count]),
        )
        scf_d.YieldOp(outputs + [ctx.tensors.value(n) for n in names])

    results = list(while_op.results)
    for name, value in zip(names, results[count:], strict=True):
        ctx.tensors.rebind(name, value)
    return emit.Results(results[:count])


def _guarded_subgraph(
    ctx: BuildContext, graph: torch.fx.Graph, inputs: list[ir.Value]
) -> list[ir.Value]:
    """``lower_subgraph`` of an ``if`` branch or a ``while`` body: its accesses
    run only where the condition holds (``ctx.guard_depth``)."""
    ctx.guard_depth += 1
    try:
        return lower_subgraph(ctx, graph, inputs)
    finally:
        ctx.guard_depth -= 1


def lower_subgraph(
    ctx: BuildContext, graph: torch.fx.Graph, inputs: list[ir.Value]
) -> list[ir.Value]:
    """Lower an ``if``/``else``/``while``/combine graph with its placeholders
    bound to ``inputs``; its output values."""
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    for placeholder, value in zip(placeholders, inputs, strict=True):
        ctx.set_value(placeholder, value)
    ctx.lower_graph(graph)
    outputs = next(node for node in graph.nodes if node.op == "output").args[0]
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return [ctx.get_value(value) for value in outputs]


def carried_positions(
    node: torch.fx.Node, args: list[torch.fx.Node], body: torch.fx.Graph
) -> list[int]:
    """Position in the loop's ``args`` of the variable each ``body`` output assigns.

    Helion merges output ``i`` into its variable as
    ``_phi(before, getitem(loop, i))`` (``_phi`` is never dead-code eliminated),
    where ``before`` is the variable's pre-loop value, which the loop also takes
    as an arg. Outputs follow the args' order.
    """
    before: dict[int, torch.fx.Node] = {}
    for item in node.users:
        for phi in item.users:
            if phi.target is tracing_ops._phi and phi.args[1] is item:
                before[item.args[1]] = phi.args[0]
    (outputs,) = next(iter(body.find_nodes(op="output"))).args
    count = len(outputs) if isinstance(outputs, (list, tuple)) else 1
    positions: list[int] = []
    for index in range(count):
        start = positions[-1] + 1 if positions else 0
        position = next(
            (p for p in range(start, len(args)) if args[p] is before.get(index)),
            None,
        )
        if position is None:
            raise NodeLoweringError(
                node,
                reason=f"loop output {index} does not update one of the loop's args",
            )
        positions.append(position)
    return positions


def with_carried(
    inputs: list[ir.Value], positions: list[int], carried: list[ir.Value]
) -> list[ir.Value]:
    """``inputs`` with the carried ones (at ``positions``) replaced by ``carried``."""
    values = list(inputs)
    for position, value in zip(positions, carried, strict=True):
        values[position] = value
    return values
