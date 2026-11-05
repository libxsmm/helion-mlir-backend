"""Identity-keyed dispatch from FX node targets to MLIR lowering handlers.

Handlers register with :func:`lowers` against the exact target objects Helion's
device IR uses: Helion API functions (``hl.load``, ``_tracing_ops._for_loop``...),
ATen ``OpOverload`` objects, or an ``OpOverloadPacket`` restricted to named
overloads (so ``aten.div`` with ``overloads=("Tensor",)`` never sees
``div.Tensor_mode``). Nothing is matched by name.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch._ops import OpOverload
from torch._ops import OpOverloadPacket

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

    import mlir.ir as ir

    from ..build_context import BuildContext

    Handler = Callable[[BuildContext, torch.fx.Node], object]


class _NotApplicable:
    def __repr__(self) -> str:
        return "NOT_APPLICABLE"


NOT_APPLICABLE = _NotApplicable()
"""Returned by a pattern handler that declines a node, so the next handler runs."""


@dataclass(frozen=True)
class _Entry:
    handler: Handler
    overloads: frozenset[str] | None


_ENTRIES: dict[object, list[_Entry]] = {}


def lowers(
    *targets: object, overloads: Iterable[str] | None = None
) -> Callable[[Handler], Handler]:
    """Register a handler for each target (functions, overloads or packets)."""
    if overloads is not None and not all(
        isinstance(target, OpOverloadPacket) for target in targets
    ):
        raise TypeError("overloads= only applies to OpOverloadPacket targets")
    entry_overloads = frozenset(overloads) if overloads is not None else None

    def register(handler: Handler) -> Handler:
        for target in targets:
            _ENTRIES.setdefault(target, []).append(_Entry(handler, entry_overloads))
        return handler

    return register


def handlers_for(target: object) -> list[Handler]:
    """Handlers for ``target`` in registration order: exact target, then packet."""
    entries = list(_ENTRIES.get(target, ()))
    if isinstance(target, OpOverload):
        name = target._overloadname
        entries.extend(
            entry
            for entry in _ENTRIES.get(target.overloadpacket, ())
            if entry.overloads is None or name in entry.overloads
        )
    return [entry.handler for entry in entries]


def lower_node(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    """Lower one FX node, attributing any error to the kernel source line."""
    if node.op == "placeholder":
        return ctx.node_to_value.get(node)
    if node.op == "output":
        return None
    location = node.meta.get("location")
    with location if location is not None else contextlib.nullcontext():
        return _dispatch(ctx, node)


COUNTING_REDUCTIONS = {
    torch.ops.aten.mean.dim,
    torch.ops.aten.mean.default,
    torch.ops.aten.var.dim,
    torch.ops.aten.var.correction,
    torch.ops.aten.std.dim,
    torch.ops.aten.std.correction,
    torch.ops.aten.var_mean.dim,
    torch.ops.aten.var_mean.correction,
    torch.ops.aten.std_mean.dim,
    torch.ops.aten.std_mean.correction,
}
"""Reductions that count their elements: Helion masks a tile's padding but they
would still divide by the tile's size."""


def _reduced_sizes(node: torch.fx.Node) -> list[object]:
    """The sizes a counting reduction reduces: its source sits behind the Inductor
    buffer of ``_extra_args``, itself masked or not."""
    dims = node.args[1] if len(node.args) > 1 else None
    buffers = node.kwargs.get("_extra_args") or []
    sizes: list[object] = []
    for buffer in buffers:
        if not isinstance(buffer, torch.fx.Node):
            continue
        for source in buffer.all_input_nodes:
            shape = source.meta["val"].shape
            picked = range(len(shape)) if dims is None else dims
            sizes.extend(shape[dim] for dim in picked)
    return sizes


def refuse_counting_partial_tiles(ctx: BuildContext, node: torch.fx.Node) -> None:
    """Refuse a reduction of :data:`COUNTING_REDUCTIONS` over a partial tile's dim:
    a wrong result, silently (a mean of 12 elements of a tile of 16 is 12/16 of
    it)."""
    from ..support import UnsupportedOperationError

    if node.target not in COUNTING_REDUCTIONS:
        return
    for size in _reduced_sizes(node):
        if ctx.block_id_to_valid.get(ctx.env.resolve_block_id(size)) is not None:
            raise UnsupportedOperationError(
                f"{node.target} over a partial tile",
                reason="it would divide by the tile's size, padding included",
                alternatives=[
                    "a sum over the tile divided by the count of real elements",
                    "block sizes that divide the sizes they tile, with static_shapes=True",
                ],
            )


def _dispatch(ctx: BuildContext, node: torch.fx.Node) -> ir.Value | None:
    from ..aten_bridge import is_aten_op
    from ..aten_bridge import lower_via_aten_helper
    from ..support import UnsupportedOperationError

    if node.op != "call_function":
        raise UnsupportedOperationError(node.op, reason=f"FX node kind of {node.name}")
    refuse_counting_partial_tiles(ctx, node)
    for handler in handlers_for(node.target):
        result = handler(ctx, node)
        if result is not NOT_APPLICABLE:
            return result
    if is_aten_op(node):
        return lower_via_aten_helper(ctx, node)
    raise UnsupportedOperationError(
        str(getattr(node.target, "__name__", node.target)),
        reason="no MLIR lowering is registered for this operation",
    )
