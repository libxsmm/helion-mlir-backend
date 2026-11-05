"""Helpers shared by the transform ops of ``mlir_transforms``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mlir import ir
from mlir.dialects import affine
from mlir.dialects import arith
from mlir.dialects import linalg
from mlir.dialects import scf
from mlir.dialects import tensor
from mlir.dialects import vector

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mlir.dialects import transform

# Largest vector, in elements, ops are vectorized to: LLVM takes many seconds on
# large vectors and on odd-width masked ones.
MAX_VECTOR_ELEMENTS = 4096
# Tile size of runtime extents without an evident bound, of odd runtime bounds, and
# of all extents of ops whose vectors would exceed MAX_VECTOR_ELEMENTS.
_TILE = 32
# Bytes of a vector register: the contiguous run each transpose tile moves.
_VECTOR_BYTES = 64
# Bytes of a cache line: rows of operand tiles read in place start at multiples.
_LINE_BYTES = 64

# An operand read through reshapes: (op, operand index of the value read) from
# the first reshape to the consumer.
Chain = list[tuple[ir.OpView, int]]


def payload_ops(
    state: transform.TransformState, handle: ir.Value, op_types: type | tuple
) -> list[ir.OpView]:
    """The ``op_types`` ops nested in the payload of ``handle``, in pre-order."""
    found: list[ir.OpView] = []

    def collect(visited: ir.Operation) -> ir.WalkResult:
        if isinstance(visited.opview, op_types):
            found.append(visited.opview)
        return ir.WalkResult.ADVANCE

    for target in state.get_payload_ops(handle):
        target.operation.walk(collect, ir.WalkOrder.PRE_ORDER)
    return found


def opview(owner: object) -> ir.OpView:
    return owner.opview if isinstance(owner, ir.Operation) else owner


def is_linalg(owner: object) -> bool:
    if isinstance(owner, ir.Operation):
        owner = owner.opview
    return isinstance(owner, ir.OpView) and linalg.get_indexing_maps(owner) is not None


def has_static_shape(op: ir.OpView) -> bool:
    return all(
        ir.ShapedType(value.type).has_static_shape
        for value in [*op.operands, *op.results]
        if isinstance(value.type, ir.ShapedType)
    )


def is_transpose(op: ir.OpView) -> bool:
    """A static linalg op only moving its one input to a permuted layout."""
    if len(op.operands) != 2 or not has_static_shape(op):
        return False
    maps = linalg.get_indexing_maps(op)
    if maps is None or len(maps) != 2:
        return False
    source, result = (ir.AffineMapAttr(affine_map).value for affine_map in maps)
    if source == result or not (source.is_permutation and result.is_permutation):
        return False
    body = list(op.regions[0].blocks[0].operations)
    return (
        len(body) == 1 and body[0].operands[0] == op.regions[0].blocks[0].arguments[0]
    )


def is_contraction_input(use: ir.OpOperand) -> bool:
    """Whether ``use`` reads its value, possibly through slices, as an input of
    a contraction."""
    owner = opview(use.owner)
    if isinstance(owner, tensor.ExtractSliceOp):
        uses = list(owner.result.uses)
        return bool(uses) and all(is_contraction_input(u) for u in uses)
    return (
        is_linalg(owner)
        and linalg.isa_contraction_op(owner)
        and use.operand_number < len(owner.operands) - 1
    )


def is_operand_pack(op: ir.OpView) -> bool:
    """A static transpose only read as a contraction operand: a pack into the
    layout a contraction kernel reads (e.g. VNNI). Fused into the contraction's
    tiles, its values would not be in memory, which AMX tile loads need."""
    if not is_transpose(op):
        return False
    uses = list(op.results[0].uses)
    return bool(uses) and all(is_contraction_input(use) for use in uses)


def bits(element_type: ir.Type) -> int:
    if isinstance(element_type, (ir.FloatType, ir.IntegerType)):
        return element_type.width
    return 64


def pad_value(pad: tensor.PadOp | tensor.GenerateOp) -> ir.Value | None:
    """The pad's (or generate's) constant value, usable before it, if it has one."""
    body = pad.regions[0].blocks[0]
    value = list(body.operations)[-1].operands[0]
    if isinstance(value, ir.OpResult) and isinstance(
        value.owner.opview, arith.ConstantOp
    ):
        return arith.ConstantOp(value.type, value.owner.opview.value).result
    if isinstance(value, ir.BlockArgument) and value.owner == body:
        return None
    if isinstance(value, ir.OpResult) and value.owner.block == body:
        return None
    return value


def pad_amount(pad: tensor.PadOp, high: bool, dim: int) -> int | ir.Value:
    """The low or high padding of ``dim``, as a constant or a value."""
    static = list(pad.static_high if high else pad.static_low)
    if static[dim] != ir.ShapedType.get_dynamic_size():
        return static[dim]
    dynamic = list(pad.high if high else pad.low)
    return dynamic[
        sum(size == ir.ShapedType.get_dynamic_size() for size in static[:dim])
    ]


def int_constant(value: ir.Value) -> int | None:
    """The value of the integer ``arith.constant`` defining ``value``, if any."""
    if isinstance(value, ir.OpResult) and isinstance(
        value.owner.opview, arith.ConstantOp
    ):
        attr = value.owner.opview.value
        if isinstance(attr, ir.IntegerAttr):
            return attr.value
    return None


def is_accumulator(value: ir.Value) -> bool:
    """Whether every use of ``value`` is a ``vector.contract`` accumulator, or
    an ``scf.for`` init of an iter arg whose uses all are."""
    uses = list(value.uses)
    if not uses:
        return False
    for use in uses:
        owner = opview(use.owner)
        if isinstance(owner, vector.ContractionOp) and use.operand_number == 2:
            continue
        if not isinstance(owner, scf.ForOp) or use.operand_number < 3:
            return False
        iter_arg = owner.inner_iter_args[use.operand_number - 3]
        if not is_accumulator(iter_arg):
            return False
    return True


def _min(bounds: Sequence[int | None]) -> int | None:
    known = [bound for bound in bounds if bound is not None]
    return min(known) if known else None


def _size_bound(value: ir.Value, depth: int = 0) -> int | None:
    """A constant upper bound of the index ``value``, if one is evident."""
    if (constant := int_constant(value)) is not None:
        return constant
    if depth > 16 or not isinstance(value, ir.OpResult):
        return None
    op = value.owner.opview
    if isinstance(op, affine.AffineMinOp):
        affine_map = ir.AffineMapAttr(op.attributes["map"]).value
        operands = list(op.operands)
        bounds = []
        for expr in affine_map.results:
            if isinstance(expr, ir.AffineConstantExpr):
                bounds.append(expr.value)
            elif isinstance(expr, ir.AffineDimExpr):
                operand = operands[expr.position]
                bounds.append(_size_bound(operand, depth + 1))
            elif isinstance(expr, ir.AffineSymbolExpr):
                operand = operands[affine_map.n_dims + expr.position]
                bounds.append(_size_bound(operand, depth + 1))
        return _min(bounds)
    if isinstance(op, arith.MinSIOp | arith.MinUIOp):
        return _min([_size_bound(operand, depth + 1) for operand in op.operands])
    if isinstance(op, arith.AddIOp):
        bounds = [_size_bound(operand, depth + 1) for operand in op.operands]
        return None if None in bounds else sum(bounds)
    if isinstance(op, tensor.DimOp) and (dim := int_constant(op.index)) is not None:
        return dim_bound(op.source, dim, depth + 1)
    return None


def _dynamic_size(
    sizes: Sequence[ir.Value], shaped: ir.ShapedType, dim: int
) -> ir.Value:
    """The size operand of dynamic ``dim`` of an op listing only dynamic sizes."""
    return sizes[sum(shaped.is_dynamic_dim(i) for i in range(dim))]


def _pad_amount_bound(
    pad: tensor.PadOp, high: bool, dim: int, depth: int
) -> int | None:
    amount = pad_amount(pad, high, dim)
    return amount if isinstance(amount, int) else _size_bound(amount, depth)


def dim_bound(value: ir.Value, dim: int, depth: int = 0) -> int | None:
    """A constant upper bound of ``dim`` of the tensor ``value``, if one is evident."""
    shaped = ir.ShapedType(value.type)
    if not shaped.is_dynamic_dim(dim):
        return shaped.get_dim_size(dim)
    if depth > 16:
        return None
    if isinstance(value, ir.BlockArgument):
        owner = value.owner.owner.operation.opview
        if isinstance(owner, scf.ForOp):
            return dim_bound(owner.initArgs[value.arg_number - 1], dim, depth + 1)
        if isinstance(owner, scf.ForallOp):
            ivs = len(owner.induction_variables)
            return dim_bound(owner.outputs[value.arg_number - ivs], dim, depth + 1)
        return None
    op = value.owner.opview
    if isinstance(op, tensor.ExtractSliceOp | tensor.EmptyOp | tensor.GenerateOp):
        sizes = {
            tensor.ExtractSliceOp: lambda: op.sizes,
            tensor.EmptyOp: lambda: op.dynamicSizes,
            tensor.GenerateOp: lambda: op.dynamicExtents,
        }[type(op)]()
        # Dims an extract_slice drops are unit dims, so dynamic sizes stay in order.
        return _size_bound(_dynamic_size(list(sizes), shaped, dim), depth + 1)
    if isinstance(op, tensor.CastOp):
        return dim_bound(op.source, dim, depth + 1)
    if isinstance(op, tensor.PadOp):
        bounds = [
            dim_bound(op.source, dim, depth + 1),
            *(_pad_amount_bound(op, high, dim, depth + 1) for high in (False, True)),
        ]
        return None if None in bounds else sum(bounds)
    if isinstance(op, tensor.InsertSliceOp):
        return dim_bound(op.dest, dim, depth + 1)
    if isinstance(op, vector.TransferWriteOp):
        return dim_bound(op.base, dim, depth + 1)
    if isinstance(op, scf.ForOp):
        return dim_bound(op.initArgs[value.result_number], dim, depth + 1)
    if isinstance(op, scf.ForallOp):
        return dim_bound(op.outputs[value.result_number], dim, depth + 1)
    if isinstance(op, scf.IfOp):
        if any(not region.blocks for region in op.regions):
            return None
        yields = [
            list(region.blocks[0].operations)[-1].operands[value.result_number]
            for region in op.regions
        ]
        bounds = [dim_bound(yielded, dim, depth + 1) for yielded in yields]
        return None if None in bounds else max(bounds)
    if linalg.get_indexing_maps(op) is not None:
        outputs = list(op.operands)[len(op.operands) - len(op.results) :]
        return dim_bound(outputs[value.result_number], dim, depth + 1)
    return None


def linalg_loop_bounds(op: ir.OpView) -> list[int | None] | None:
    """Constant upper bounds of a linalg op's loop ranges (``None`` where not
    evident), or ``None`` for an op without indexing maps."""
    maps = linalg.get_indexing_maps(op)
    operands = list(op.operands)
    if maps is None or len(maps) != len(operands):
        return None
    maps = [ir.AffineMapAttr(affine_map).value for affine_map in maps]
    bounds: list[int | None] = [None] * maps[0].n_dims
    for operand, affine_map in zip(operands, maps, strict=True):
        if not isinstance(operand.type, ir.ShapedType):
            continue
        for position, expr in enumerate(affine_map.results):
            if isinstance(expr, ir.AffineDimExpr):
                loop_dim = expr.position
                bounds[loop_dim] = _min(
                    [bounds[loop_dim], dim_bound(operand, position)]
                )
    return bounds


def linalg_tile_sizes(op: ir.OpView) -> list[int] | None:
    """Tile sizes making a linalg op's loop ranges bounded and its vectors at most
    ``MAX_VECTOR_ELEMENTS``, those of an op of runtime shape multiples of
    ``_TILE``, and those of a transpose 1 on all but the inner dims of its source
    and result (0: untiled)."""
    bounds = linalg_loop_bounds(op)
    if bounds is None:
        return None
    too_large = None not in bounds and math.prod(bounds) > MAX_VECTOR_ELEMENTS
    static = has_static_shape(op)
    sizes = [
        _TILE
        if bound is None
        or (bound > _TILE and (too_large or (not static and bound % _TILE)))
        else 0
        for bound in bounds
    ]
    # LLVM takes many seconds on large n-D vector transposes. Each tile reads and
    # writes runs of a vector (e.g. a VNNI pack: 2 rows into pairs; a transposed
    # one: 16x16 blocks of pairs).
    if is_transpose(op):
        tiles = transpose_tiles(op)
        for dim, bound in enumerate(bounds):
            size = tiles.get(dim, 1)
            sizes[dim] = size if bound > size else 0
    return sizes


def transpose_tiles(op: ir.OpView) -> dict[int, int]:
    """Tile sizes of a static transpose's loop dims reading and writing runs of
    at least ``_VECTOR_BYTES``: its source's and result's inner dims, as far as
    needed (others 1)."""
    bounds = linalg_loop_bounds(op)
    element_type = ir.ShapedType(op.operands[0].type).element_type
    run = max(1, _VECTOR_BYTES * 8 // bits(element_type))
    tiles: dict[int, int] = {}
    for affine_map in linalg.get_indexing_maps(op):
        need = run
        for expr in reversed(ir.AffineMapAttr(affine_map).value.results):
            if need <= 1:
                break
            dim = ir.AffineDimExpr(expr).position
            size = _largest_divisor_at_most(bounds[dim], need)
            tiles[dim] = max(tiles.get(dim, 1), size)
            need = -(-need // size)
    return tiles


def _largest_divisor_at_most(extent: int, limit: int) -> int:
    return max(d for d in range(1, min(extent, limit) + 1) if extent % d == 0)


def dim_sizes(value: ir.Value) -> list[ir.Value | int]:
    """The sizes of the tensor ``value``: constants, or ``tensor.dim`` values."""
    value_type = ir.RankedTensorType(value.type)
    index_type = ir.IndexType.get()
    return [
        tensor.DimOp(value, arith.ConstantOp(index_type, dim).result).result
        if value_type.is_dynamic_dim(dim)
        else value_type.shape[dim]
        for dim in range(value_type.rank)
    ]


def copy_rows(
    source: ir.Value,
    dest: ir.Value,
    sizes: list[ir.Value | int],
    padding: ir.Value | None = None,
) -> ir.Value:
    """Copy ``source`` of ``sizes`` into the leading corner of ``dest`` one
    innermost row at a time: 1-D copies vectorize to plain masked loads and
    stores, n-D ones to transfers lowered through memory. With ``padding``, every
    row of the static ``dest`` is written: a row of ``source`` padded past its
    end, other rows filled."""
    index_type = ir.IndexType.get()
    dynamic = ir.ShapedType.get_dynamic_size()
    rank = len(sizes)
    cols = sizes[-1]
    static_cols = cols if isinstance(cols, int) else dynamic
    dynamic_cols = [] if isinstance(cols, int) else [cols]
    element_type = ir.RankedTensorType(source.type).element_type
    row_type = ir.RankedTensorType.get([static_cols], element_type)
    dest_shape = ir.RankedTensorType(dest.type).shape
    width = dest_shape[-1]
    zero = arith.ConstantOp(index_type, 0).result
    one = arith.ConstantOp(index_type, 1).result

    def constant(size: ir.Value | int) -> ir.Value:
        return (
            size
            if not isinstance(size, int)
            else arith.ConstantOp(index_type, size).result
        )

    def write_row(buffer: ir.Value, rows: list[ir.Value]) -> ir.Value:
        args = (
            rows,
            dynamic_cols,
            [],
            [dynamic] * len(rows) + [0],
            [1] * len(rows) + [static_cols],
            [1] * rank,
        )
        source_row = tensor.extract_slice(row_type, source, *args)
        if padding is None or static_cols == width:
            dest_row = tensor.extract_slice(row_type, buffer, *args)
            copied = linalg.copy(source_row, outs=[dest_row])
            return tensor.insert_slice(copied, buffer, *args)
        high = arith.SubIOp(
            arith.ConstantOp(index_type, width).result,
            tensor.DimOp(source_row, zero).result,
        ).result
        padded = tensor.PadOp(
            ir.RankedTensorType.get([width], element_type),
            source_row,
            [],
            [high],
            [0],
            [dynamic],
        )
        body = padded.regions[0].blocks.append(index_type)
        with ir.InsertionPoint(body):
            tensor.YieldOp(padding)
        return tensor.insert_slice(padded.result, buffer, *full_row(rows))

    def full_row(rows: list[ir.Value]) -> tuple:
        return (
            rows,
            [],
            [],
            [dynamic] * len(rows) + [0],
            [1] * len(rows) + [width],
            [1] * rank,
        )

    def fill_row(buffer: ir.Value, rows: list[ir.Value]) -> ir.Value:
        dest_row = tensor.extract_slice(
            ir.RankedTensorType.get([width], element_type), buffer, *full_row(rows)
        )
        filled = linalg.fill(padding, outs=[dest_row])
        return tensor.insert_slice(filled, buffer, *full_row(rows))

    def copy(buffer: ir.Value, rows: list[ir.Value]) -> ir.Value:
        if len(rows) == rank - 1:
            inside = None
            if padding is not None:
                for row, size, extent in zip(rows, sizes, dest_shape, strict=False):
                    if size == extent:
                        continue
                    row_inside = arith.CmpIOp(
                        arith.CmpIPredicate.ult, row, constant(size)
                    ).result
                    inside = (
                        row_inside
                        if inside is None
                        else arith.AndIOp(inside, row_inside).result
                    )
            if inside is None:
                return write_row(buffer, rows)
            branch = scf.IfOp(inside, [buffer.type], has_else=True)
            with ir.InsertionPoint(branch.then_block):
                scf.YieldOp([write_row(buffer, rows)])
            with ir.InsertionPoint(branch.else_block):
                scf.YieldOp([fill_row(buffer, rows)])
            return branch.results[0]
        bound = constant(sizes[len(rows)] if padding is None else dest_shape[len(rows)])
        loop = scf.ForOp(zero, bound, one, [buffer])
        with ir.InsertionPoint(loop.body):
            scf.YieldOp(
                [copy(loop.inner_iter_args[0], [*rows, loop.induction_variable])]
            )
        return loop.results[0]

    return copy(dest, [])


def operand_chain(consumer: ir.OpView, index: int) -> tuple[Chain, ir.Value]:
    """The reshapes in ``consumer``'s block, each only read by the next, that
    operand ``index`` of ``consumer`` is read through; the value reshaped."""
    chain = [(consumer, index)]
    value = consumer.operands[index]
    while isinstance(value, ir.OpResult):
        op = value.owner.opview
        if (
            not isinstance(op, (tensor.ExpandShapeOp, tensor.CollapseShapeOp))
            or op.operation.block != consumer.operation.block
            or len(list(value.uses)) != 1
        ):
            break
        chain.insert(0, (op, 0))
        value = op.operands[0]
    return chain, value


def expanded_slice(
    value: ir.Value, shape: Sequence[int]
) -> tuple[tensor.ExtractSliceOp, list[int], ir.OpView | None] | None:
    """The slice ``value`` of static ``shape`` is, through casts and expanded
    shapes; the slice's static shape and the op reading it on the way (``None``
    if ``value`` is the slice), else ``None``."""
    shape, user = list(shape), None
    while isinstance(value, ir.OpResult):
        op = value.owner.opview
        if isinstance(op, tensor.ExtractSliceOp):
            return op, shape, user
        if isinstance(op, tensor.CastOp):
            value = op.source
        elif isinstance(op, tensor.ExpandShapeOp):
            shape = [
                math.prod(
                    shape[ir.IntegerAttr(dim).value] for dim in ir.ArrayAttr(group)
                )
                for group in ir.ArrayAttr(op.reassociation)
            ]
            value = op.src
        else:
            return None
        user = op
    return None


def misaligned_rows(slice_op: tensor.ExtractSliceOp, sizes: Sequence[int]) -> bool:
    """Whether ``slice_op``, of static ``sizes``, reads rows not starting at
    cache-line multiples of each other: the stride of a dim outside the
    slice's contiguous inner run, in the static tensor it slices (through
    slices) stored row-major, is not a multiple of a cache line."""
    source = slice_op.source
    while isinstance(source, ir.OpResult) and isinstance(
        source.owner.opview, tensor.ExtractSliceOp
    ):
        source = source.owner.opview.source
    source_type = ir.RankedTensorType(source.type)
    if not source_type.has_static_shape or source_type.rank != len(sizes):
        return False
    source_shape = source_type.shape
    element_bits = bits(source_type.element_type)
    run = len(sizes) - 1
    while run > 0 and sizes[run] == source_shape[run]:
        run -= 1
    stride = math.prod(source_shape[run:])
    for dim in range(run - 1, -1, -1):
        if sizes[dim] > 1 and stride * element_bits // 8 % _LINE_BYTES:
            return True
        stride *= source_shape[dim]
    return False
