"""Authoritative (block_id, bias, is_scalar) resolution for index nodes.

Single source of truth for mapping a Helion device-IR index expression to the
block id / bias it represents, built entirely from Helion's own device-IR
metadata (``tile_with_offset``, symbol origins) instead of name-based or
id()-keyed heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops

from .block_ids import OFFSET_SYMBOL_KINDS
from .block_ids import SCALAR_SYMBOL_KINDS
from .block_ids import block_id_from_key
from .errors import UnsupportedOperationError

if TYPE_CHECKING:
    from ..build_context import BuildContext


@dataclass(frozen=True)
class IndexDescriptor:
    """Resolved identity of a single subscript index position."""

    block_id: int | None
    bias: int
    is_scalar: bool
    kind: str | None = None
    """The symbol kind (``grid``, ``tile_id``, ...) of a scalar position."""

    @property
    def is_offset(self) -> bool:
        """A scalar position equal to its loop's current offset."""
        return self.kind in OFFSET_SYMBOL_KINDS


_UNRESOLVED = IndexDescriptor(block_id=None, bias=0, is_scalar=False)


def resolve_index_descriptor(ctx: BuildContext, index_node: object) -> IndexDescriptor:
    """Resolve an index expression to its block id, bias, and scalar-ness.

    Resolution order (all authoritative, no name/heuristic matching):
    1. Literal int -> scalar constant offset.
    2. ``meta['tile_with_offset']`` (``tile.index + k``) -> block id + constant
       offset, set for every backend by Helion's ``add_tile_with_offset_metadata``.
    3. Symbol origin (``HostFunction.expr_to_origin`` via
       ``BuildContext.symbol_info``) -> block id, plus whether it denotes a
       scalar grid/tile position or a tile extent.
    4. ``_get_symnode('block_size_N')`` key -> block id directly; a constant
       key names the active tile loop whose block size Helion specialized to
       it (``hl.register_block_size`` of a size-1 dim).
    """
    import torch
    import torch.fx

    if isinstance(index_node, int):
        return IndexDescriptor(block_id=None, bias=index_node, is_scalar=True)
    if not isinstance(index_node, torch.fx.Node):
        return _UNRESOLVED

    tile_meta = index_node.meta.get("tile_with_offset")
    if tile_meta is not None:
        offset = tile_meta.get("offset", 0)
        if isinstance(offset, torch.SymInt) and not offset.node.expr.free_symbols:
            offset = int(offset.node.expr)
        if not isinstance(offset, int):
            raise UnsupportedOperationError(
                "tile index", reason=f"tile.index plus the runtime offset {offset}"
            )
        return IndexDescriptor(
            block_id=tile_meta.get("block_id"), bias=offset, is_scalar=False
        )

    symbol_info = ctx.symbol_info(index_node.meta.get("val"))
    if symbol_info is not None:
        block_id, kind = symbol_info
        return IndexDescriptor(
            block_id=block_id,
            bias=0,
            is_scalar=kind in SCALAR_SYMBOL_KINDS,
            kind=kind,
        )

    target = index_node.target
    if target is tracing_ops._get_symnode and index_node.args:
        key = index_node.args[0]
        block_id = block_id_from_key(key)
        if block_id is None and key.isdigit():
            block_id = _specialized_block(ctx, key)
        if block_id is not None:
            return IndexDescriptor(block_id=block_id, bias=0, is_scalar=False)

    return _UNRESOLVED


def _specialized_block(ctx: BuildContext, key: str) -> int | None:
    """The one active tile loop whose block size symbol is the constant ``key``."""
    import torch

    matches = [
        info.block_id
        for info in ctx.env.block_sizes
        if isinstance(info.var, torch.SymInt)
        and str(info.var.node.expr) == key
        and info.block_id in ctx.block_id_to_bounds
    ]
    return matches[0] if len(matches) == 1 else None


def index_lower_bound(ctx: BuildContext, index_node: object) -> int | None:
    """A lower bound of the values the index expression ``index_node`` takes,
    if one follows statically from the lower bounds of the loops it uses.

    Understood: integers, tile indices (``tile.index``, plus a constant:
    ``tile_with_offset``) and scalar loop positions (``tile.begin``, grid
    indices: a SymInt linear in them with positive coefficients) at their
    loop's static lower bound, and sums and differences of those with
    constants (``tile.index - 1``, ``tile.begin - 1``, ``tile.index + kw - 1``).
    Anything else (loads, runtime sizes, products) has no bound (``None``).
    """
    import operator

    import torch
    import torch.fx

    if isinstance(index_node, bool):
        return None
    if isinstance(index_node, int):
        return index_node
    if isinstance(index_node, torch.SymInt):
        return _symint_lower_bound(ctx, index_node)
    if not isinstance(index_node, torch.fx.Node):
        return None

    tile_meta = index_node.meta.get("tile_with_offset")
    if tile_meta is not None:
        begin = _loop_lower_bound(ctx, tile_meta.get("block_id"))
        offset = tile_meta.get("offset", 0)
        if begin is None or not isinstance(offset, int):
            return None
        return begin + offset

    target = index_node.target
    args = index_node.args
    if target is tracing_ops._get_symnode or index_node.op != "call_function":
        return _symint_lower_bound(ctx, index_node.meta.get("val"))
    if _is_tile_index(target) and args:
        from helion._compiler.compile_environment import CompileEnvironment

        tile = args[0].meta.get("val") if isinstance(args[0], torch.fx.Node) else None
        if not isinstance(tile, torch.SymInt):
            return None
        return _loop_lower_bound(ctx, CompileEnvironment.current().get_block_id(tile))
    if index_node.kwargs or len(args) != 2:
        return _symint_lower_bound(ctx, index_node.meta.get("val"))
    add = (operator.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
    sub = (operator.sub, torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)
    if target in add:
        bounds = [index_lower_bound(ctx, arg) for arg in args]
        return None if None in bounds else bounds[0] + bounds[1]
    if target in sub:
        lhs = index_lower_bound(ctx, args[0])
        rhs = _constant(args[1])
        return None if lhs is None or rhs is None else lhs - rhs
    return _symint_lower_bound(ctx, index_node.meta.get("val"))


def _is_tile_index(target: object) -> bool:
    import helion.language as hl

    return target is hl.tile_index or getattr(target, "__name__", None) == "tile_index"


def _constant(value: object) -> int | None:
    """``value`` as an integer, if it is one (an int, a constant SymInt, or an
    FX node of one)."""
    import torch
    import torch.fx

    if isinstance(value, torch.fx.Node):
        value = value.meta.get("val")
    if isinstance(value, torch.SymInt) and not value.node.expr.free_symbols:
        value = int(value.node.expr)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _loop_lower_bound(ctx: BuildContext, block_id: int | None) -> int | None:
    """The static lower bound of ``block_id``'s active loop, if any."""
    if block_id is None:
        return None
    bounds = ctx.block_id_to_bounds.get(block_id)
    if bounds is None or not isinstance(bounds[0], int):
        return None
    return bounds[0]


def _symint_lower_bound(ctx: BuildContext, value: object) -> int | None:
    """A lower bound of the SymInt ``value``: linear with positive integer
    coefficients in loop positions (``tile.begin``, grid indices), each at
    its loop's static lower bound."""
    import sympy
    import torch

    from .block_ids import symbol_origin_info

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, torch.SymInt):
        return None
    expr = value.node.expr
    lowest = {}
    for symbol in expr.free_symbols:
        info = symbol_origin_info(ctx.host_function, symbol)
        if info is None or info[1] not in OFFSET_SYMBOL_KINDS:
            return None
        slope = sympy.diff(expr, symbol)
        if not (slope.is_Integer and slope > 0):
            return None
        begin = _loop_lower_bound(ctx, info[0])
        if begin is None:
            return None
        lowest[symbol] = begin
    bound = expr.subs(lowest)
    return int(bound) if bound.is_Integer else None
