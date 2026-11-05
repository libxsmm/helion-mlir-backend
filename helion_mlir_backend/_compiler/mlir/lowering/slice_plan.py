"""Authoritative per-dimension slice plan from Helion index metadata.

Instead of re-deriving geometry from sizes/extents or heuristics, a SlicePlan
captures what each index position actually means (block id, scalar, full slice)
and gives one canonical (offsets, sizes) pair for tensor extract/insert
operations. A tile has a static extent (see ``BlockGeometry.tile_extent``), or
the runtime extent of a full slice of a runtime-sized dim; at the end of its
loop or tensor only its first ``size`` elements are real (loads zero-pad the
rest, stores drop it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

from mlir.dialects import arith as arith_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir
import torch

from . import emit

if TYPE_CHECKING:
    from ..build_context import BuildContext
    from .tensor_state import OwnedDim


@dataclass(frozen=True)
class DimSlice:
    """Descriptor of a single source/destination dimension slice."""

    kind: Literal["scalar", "tile", "full", "gather"]
    offset: ir.Value
    size: emit.Size
    """Elements taken from the tensor; fewer than ``tile`` at a boundary."""
    tile: int
    """Extent of this dimension in the loaded or stored tile; the dynamic size
    sentinel when it is the tensor's runtime extent (then ``size`` is all of it)."""
    block_id: int | None = None
    reduces: bool = False
    index: ir.Value | None = None
    """For ``gather``: the index tensor (the slice spans the whole dimension)."""


@dataclass(frozen=True)
class SlicePlan:
    """Complete descriptor of a load/store operation's index→dimension mapping."""

    dims: list[DimSlice]

    def offsets(self) -> list:
        """Dynamic offset values for extract/insert_slice, one per base-tensor dimension."""
        return [dim.offset for dim in self.dims]

    def sizes(self) -> list[emit.Size]:
        """Slice sizes for extract/insert_slice, one per base-tensor dimension."""
        return [dim.size for dim in self.dims]

    def tile_shape(self) -> list[int]:
        """Shape of the tile at full rank (``?`` for a runtime extent)."""
        return [dim.tile for dim in self.dims]

    def value_shape(self) -> list[int]:
        """Shape of the loaded/stored value tile (reduced dims omitted)."""
        return [dim.tile for dim in self.dims if not dim.reduces]

    def value_sizes(self) -> list[emit.Size]:
        """Real part of the value tile (reduced dims omitted)."""
        return [dim.size for dim in self.dims if not dim.reduces]

    def value_tile_sizes(self) -> list[emit.Size]:
        """The value tile's dims: static extents, or the runtime size of a dynamic one."""
        return [
            dim.size if ir.ShapedType.is_dynamic_size(dim.tile) else dim.tile
            for dim in self.dims
            if not dim.reduces
        ]

    def is_partial(self) -> bool:
        """Whether the tile may extend past its loop or tensor."""
        return any(
            not ir.ShapedType.is_dynamic_size(dim.tile)
            and (isinstance(dim.size, ir.Value) or dim.size != dim.tile)
            for dim in self.dims
        )

    def reduced_dims(self) -> list[int]:
        """Dimension indices that are scalar-indexed and dropped from the result."""
        return [i for i, dim in enumerate(self.dims) if dim.reduces]

    def gathers(self) -> list[tuple[int, ir.Value]]:
        """``(dimension, index tensor)`` of each tensor-indexed dimension."""
        return [
            (i, dim.index) for i, dim in enumerate(self.dims) if dim.kind == "gather"
        ]


def plan_slice(
    ctx: BuildContext,
    index_nodes: list | tuple,
    base: ir.Value,
    owned: dict[int, OwnedDim] | None = None,
    name: str | None = None,
    *,
    masked: bool = False,
) -> SlicePlan:
    """Build a SlicePlan from authoritative index metadata.

    Every index position d maps to base tensor dimension d. For each index:
    - Full slice (None) → full: offset 0, size = base extent.
    - Scalar index (grid/tile.begin) → scalar: block_id from symbol, size 1, reduces.
    - Tile index (block_id) → tile: block_id from symbol, tile-extent wide.
    - Literal int → scalar constant offset, size 1, reduces.
    - Scalar computed at run time (``n - 1``) → scalar at that offset, reduces.
    - Index tensor → gather: the whole dimension, gathered after slicing.

    ``owned`` marks dims where ``base`` is only the current iteration's region
    of a larger tensor (see ``tensor_state``); those are indexed from its origin.
    ``name`` is the host tensor ``base`` holds, for its runtime extents.
    ``masked``: the access has an ``extra_mask`` (Helion's zero-padding idiom
    ``hl.load(x, [t.index - 1], extra_mask=t.index >= 1)``), so its indices
    may be negative where the mask is false.

    Raises NodeLoweringError if a tile index cannot be resolved to a block id.
    """
    from ..analysis.signature import runtime_index_expr
    from ..support.errors import NodeLoweringError
    from ..support.index_meta import resolve_index_descriptor

    owned = owned or {}
    base_shape = ir.RankedTensorType(base.type).shape
    base_rank = len(base_shape)
    dims: list[DimSlice] = []

    def full(dimension: int, index: slice) -> DimSlice:
        extent = base_shape[dimension]
        if ir.ShapedType.is_dynamic_size(extent):
            return _dynamic_full(
                ctx, index, ctx.sizes.extent(base, dimension, name), extent
            )
        start, stop = _static_slice_bounds(index, extent)
        return DimSlice("full", ctx.index_const(start), stop - start, stop - start)

    for dimension, index_node in enumerate(index_nodes):
        if dimension >= base_rank:
            break
        extent = base_shape[dimension]

        if isinstance(index_node, slice):
            dims.append(full(dimension, index_node))
            continue

        descriptor = resolve_index_descriptor(ctx, index_node)
        owner = owned.get(dimension)
        if owner is not None:
            inner = (
                descriptor.block_id is not None
                and descriptor.block_id != owner.block_id
                and not descriptor.is_scalar
                and ctx.geometry.owning_block(descriptor.block_id, {owner.block_id})
                is not None
            )
            if (descriptor.block_id != owner.block_id and not inner) or descriptor.bias:
                raise NodeLoweringError(
                    index_node,
                    reason=(
                        f"dimension {dimension} is owned by block_id "
                        f"{owner.block_id} but indexed by another expression"
                    ),
                )
            zero = ctx.index_const(0)
            if owner.point:
                dims.append(DimSlice("scalar", zero, 1, 1, owner.block_id, True))
                continue
            # The region is the tile's real part; its extent is dynamic when partial.
            size = (
                tensor_d.DimOp(base, ctx.index_const(dimension)).result
                if ir.ShapedType.is_dynamic_size(extent)
                else extent
            )
            if inner:
                dims.append(_inner_tile(ctx, descriptor.block_id, owner.block_id, size))
                continue
            tile = ctx.geometry.tile_extent(owner.block_id)
            dims.append(DimSlice("tile", zero, size, tile, owner.block_id))
            continue

        if isinstance(index_node, int) and not isinstance(index_node, bool):
            dims.append(
                _literal_scalar(
                    ctx, index_node, ctx.sizes.extent(base, dimension, name)
                )
            )
            continue
        if not masked and not ctx.guard_depth:
            _reject_negative(ctx, index_node, dimension, base_shape[dimension])

        if descriptor.is_scalar:
            block_id = descriptor.block_id
            if descriptor.is_offset and block_id in ctx.block_id_to_iv:
                scalar_value = ctx.block_id_to_iv[block_id]
            else:
                scalar_value = ctx.get_value(index_node)
            offset = (
                ctx.cast_to_index(scalar_value)
                if scalar_value is not None
                else ctx.index_const(descriptor.bias)
            )
            dims.append(DimSlice("scalar", offset, 1, 1, block_id, reduces=True))
            continue

        block_id, bias = descriptor.block_id, descriptor.bias
        index = ctx.get_value(index_node) if block_id is None else None
        if index is not None and isinstance(index.type, ir.RankedTensorType):
            size = ctx.sizes.extent(base, dimension, name)
            dims.append(
                DimSlice("gather", ctx.index_const(0), size, extent, index=index)
            )
            continue
        if index is not None:
            # A position computed from runtime sizes, e.g. ``n - 1``.
            offset = ctx.cast_to_index(index)
            if (
                ctx.signature is not None
                and runtime_index_expr(ctx.signature, index_node) is not None
            ):
                extent_value = ctx.as_index(ctx.sizes.extent(base, dimension, name))
                offset = _from_the_end(ctx, offset, extent_value)
            dims.append(DimSlice("scalar", offset, 1, 1, reduces=True))
            continue
        if block_id is None:
            raise NodeLoweringError(
                index_node,
                reason=f"Tile index at dimension {dimension} has no resolvable block id",
                recovery_hint="Ensure all tile indices are in hl.tile() loops with configured block_sizes",
            )
        if block_id not in ctx.geometry.blocks:
            raise NodeLoweringError(
                index_node,
                reason=f"Tile index at dimension {dimension} names unknown block_id {block_id}",
            )
        offset, size, tile = tile_window(
            ctx, block_id, bias, ctx.sizes.extent(base, dimension, name)
        )
        dims.append(DimSlice("tile", offset, size, tile, block_id))

    while len(dims) < base_rank:
        dims.append(full(len(dims), slice(None)))

    return SlicePlan(dims)


def _from_the_end(ctx: BuildContext, offset: ir.Value, extent: ir.Value) -> ir.Value:
    """A runtime position, counted from the end if negative, as a literal one is and
    as torch does. The call refuses one out of bounds (``driver.call_entry``)."""
    negative = arith_d.cmpi(arith_d.CmpIPredicate.slt, offset, ctx.index_const(0))
    return arith_d.select(negative, arith_d.addi(offset, extent), offset)


def _literal_scalar(ctx: BuildContext, index: int, extent: emit.Size) -> DimSlice:
    """A literal integer position; a negative one counts from the end, as in
    Python (``x[-1]`` is the last element)."""
    if isinstance(extent, int) and index >= extent:
        from ..support.errors import UnsupportedOperationError

        raise UnsupportedOperationError(
            "subscript",
            reason=f"index {index} is out of bounds of a dimension of {extent}",
        )
    if index >= 0:
        offset = ctx.index_const(index)
    elif isinstance(extent, int):
        if index < -extent:
            from ..support.errors import UnsupportedOperationError

            raise UnsupportedOperationError(
                "subscript",
                reason=f"index {index} is out of bounds of a dimension of {extent}",
            )
        offset = ctx.index_const(extent + index)
    else:
        offset = arith_d.addi(ctx.as_index(extent), ctx.index_const(index))
    return DimSlice("scalar", offset, 1, 1, None, reduces=True)


def _reject_negative(
    ctx: BuildContext, index_node: object, dimension: int, extent: int
) -> None:
    """Raise if the computed index ``index_node`` of an unconditional, unmasked
    access is negative at its loops' lower bounds (statically:
    ``index_lower_bound``).

    Helion leaves such indices undefined: its codegen may add them to the
    tensor's address unchecked, while torch (and Helion's eager reference
    mode) count them from the end. A shifted window (``x[tile.index - 1]``)
    usually means zero padding, which only a mask or a guard gives
    (``hl.load(..., extra_mask=...)``, an ``if``; those accesses are not
    checked), so the kernel is rejected rather than read before the tensor
    start or wrap silently."""
    from ..support.errors import UnsupportedOperationError
    from ..support.index_meta import index_lower_bound

    bound = index_lower_bound(ctx, index_node)
    if bound is not None and bound < 0:
        size = "?" if ir.ShapedType.is_dynamic_size(extent) else str(extent)
        raise UnsupportedOperationError(
            "subscript",
            reason=(
                f"the index of dimension {dimension} (size {size}) is as low as "
                f"{bound} on the first iteration of its loops, before the "
                "tensor start, and the access is neither masked nor guarded"
            ),
            alternatives=[
                "mask it: hl.load(x, [t.index - 1], extra_mask=t.index >= 1)",
                "guard it with an if",
                "start the loop later (hl.tile(1, n))",
                "pad the tensor on the host",
            ],
        )


def _inner_tile(
    ctx: BuildContext, block_id: int, owner: int, region: emit.Size
) -> DimSlice:
    """``block_id``'s current tile, inside the region of ``owner``'s tile that
    encloses it: offset from the region's origin, real part clamped to the region."""
    tile = ctx.geometry.tile_extent(block_id)
    d0, d1 = ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)
    offset = emit.affine_apply(
        d0 - d1, [ctx.block_id_to_iv[block_id], ctx.block_id_to_iv[owner]]
    )
    size = ctx.block_id_to_valid.get(block_id, tile)
    if (
        isinstance(region, int)
        and isinstance(size, int)
        and region % tile == 0
        and size == tile
    ):
        return DimSlice("tile", offset, size, tile, block_id)
    operands = [offset, ctx.as_index(region), ctx.as_index(size)]
    clamped = emit.affine_min([d1 - d0, ir.AffineDimExpr.get(2)], operands)
    return DimSlice("tile", offset, clamped, tile, block_id)


def tile_window(
    ctx: BuildContext, block_id: int, bias: int, extent: emit.Size
) -> tuple[ir.Value, emit.Size, int]:
    """Offset, real size and tile size of ``block_id``'s current tile, shifted by
    ``bias``, in a dimension of ``extent``.

    The real part is what lies inside both the loop and the tensor. The tensor
    bound is only checked when the loop may run past it (e.g. an input smaller
    than the iteration domain, whose excess is then read as zeros); a loop whose
    end is the tensor's extent (the same size value) never does.
    """
    from ..support.errors import UnsupportedOperationError

    tile = ctx.geometry.tile_extent(block_id)
    offset = ctx.block_id_to_iv.get(block_id)
    if offset is None:
        raise UnsupportedOperationError(
            "tile index", reason=f"block_id {block_id} is used outside its loop"
        )
    if bias:
        offset = arith_d.addi(offset, ctx.index_const(bias))
    size = ctx.block_id_to_valid.get(block_id, tile)
    begin, end = ctx.block_id_to_bounds[block_id]
    # A loop over one enclosing tile ends within the enclosing loop.
    outer = block_id
    seen = {block_id}
    while (
        isinstance(end, ir.Value)
        and (outer := ctx.geometry.enclosing_tiles.get(outer)) is not None
        and outer not in seen
        and outer in ctx.block_id_to_bounds
    ):
        seen.add(outer)
        end = ctx.block_id_to_bounds[outer][1]
    low = ctx.index_facts(begin).low
    if low is None or low + bias < 0:
        where = "" if low is None else f" at {low + bias}"
        raise UnsupportedOperationError(
            "tile index",
            reason=f"the loop may start{where}, before the tensor start",
            alternatives=[
                (
                    "a lower bound of tile positions and constants clamped at 0: "
                    "max(tm.begin - 16, 0)"
                ),
                "a runtime bound as hl.constexpr, clamped at 0",
            ],
        )
    if bias < 0 and not (isinstance(begin, int) and begin + bias >= 0):
        raise UnsupportedOperationError(
            "tile index", reason=f"offset {bias} may index before the tensor start"
        )
    if isinstance(end, int) and isinstance(extent, int) and end + bias <= extent:
        return offset, size, tile
    if bias == 0 and isinstance(end, ir.Value) and end == extent:
        return offset, size, tile
    # A loop ending at an enclosing tile's position stays inside that loop's range.
    high = ctx.index_facts(end).high
    if isinstance(extent, int) and high is not None and high + bias <= extent:
        return offset, size, tile
    d0 = ir.AffineDimExpr.get(0)
    if isinstance(extent, int):
        limit, limit_operands = ir.AffineConstantExpr.get(extent), []
    else:
        limit, limit_operands = ir.AffineDimExpr.get(1), [extent]
    offset = emit.affine_min([d0, limit], [offset, *limit_operands])
    remaining = limit - d0
    if isinstance(size, int):
        size = emit.affine_min(
            [ir.AffineConstantExpr.get(size), remaining], [offset, *limit_operands]
        )
    else:
        size = emit.affine_min(
            [remaining, ir.AffineDimExpr.get(1 + len(limit_operands))],
            [offset, *limit_operands, size],
        )
    return offset, size, tile


def _dynamic_full(
    ctx: BuildContext, index: slice, extent: ir.Value, dynamic: int
) -> DimSlice:
    """All of a dimension whose extent is only known at run time."""
    from ..support.errors import UnsupportedOperationError

    if (
        index.step not in (None, 1)
        or index.start not in (None, 0)
        or index.stop is not None
    ):
        raise UnsupportedOperationError(
            "subscript slice",
            reason=f"{index} of a dimension whose size is only known at run time",
        )
    return DimSlice("full", ctx.index_const(0), extent, dynamic)


def _static_slice_bounds(index: slice, extent: int) -> tuple[int, int]:
    """``[start, stop)`` of a unit-step slice with static bounds, Python-style."""
    from ..support.errors import UnsupportedOperationError

    bounds = []
    for value in (index.start, index.stop):
        if isinstance(value, torch.SymInt) and not value.node.expr.free_symbols:
            value = int(value.node.expr)
        bounds.append(value)
    if index.step not in (None, 1) or not all(
        bound is None or isinstance(bound, int) for bound in bounds
    ):
        raise UnsupportedOperationError(
            "subscript slice", reason=f"only static unit-step slices, got {index}"
        )
    start, stop, _ = slice(*bounds).indices(extent)
    return start, max(start, stop)
