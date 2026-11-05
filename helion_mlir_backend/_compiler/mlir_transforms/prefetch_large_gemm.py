"""``prefetch_large_gemm``: a short-K AMX GEMM of whole-row tiles whose output is
beyond the last-level cache prefetches each register block's output lines for writing.

Every line of such an output misses, and the store needs the line from DRAM first, so
it's fetched for writing ahead of the store.

Match, left alone if anything differs from it: a block holds AMX K loops and, after
them, writes whole lines through a subview of a whole-row tile of an ``inout``
argument. Those lines are prefetched for writing just before the first K loop. No
prefetch changes a result.

The gate is empirical, and narrow on purpose: the prefetches often lose elsewhere.
They fire for K of at most ``MAX_K``, an output of at least ``MIN_BYTES``, and tiles of
at least ``MIN_ROWS`` x ``MIN_COLUMNS`` whose f32 accumulator stays within
``MAX_TILE_BYTES``. Inside it they helped at every shape, thread count and allocator
tried on EMR. Outside it they lost: default tiles, whole rows of few columns, a large
accumulator, or a longer K.
"""

import re

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import memref
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .first_touch import _VIEW_OPS
from .first_touch import _contiguous_bytes
from .utils import payload_ops

MIN_BYTES = 320 << 20
MAX_K = 256
MIN_ROWS = 256
MIN_COLUMNS = 256
MAX_TILE_BYTES = 1 << 20
_LINE_BYTES = 64
_WRITE_LOCALITY = 3
_MULF = "x86.amx.tile_mulf"
_LOAD = "x86.amx.tile_load"
_WRITE = "vector.transfer_write"
_TILE = re.compile(r"!x86\.amx\.tile<\d+x(\d+)x\w+>")


def _element_bytes(type_: ir.Type) -> int | None:
    if isinstance(type_, ir.BF16Type):
        return 2
    if isinstance(type_, ir.FloatType):
        return type_.width // 8
    return None


def _inout(func: ir.OpView, arg: ir.BlockArgument) -> bool:
    if "arg_attrs" not in func.attributes:
        return False
    attrs = ir.DictAttr(func.attributes["arg_attrs"][arg.arg_number])
    return (
        "helion.role" in attrs and ir.StringAttr(attrs["helion.role"]).value == "inout"
    )


def _root(value: ir.Value) -> ir.BlockArgument | None:
    """The argument ``value`` is a chain of views of."""
    while isinstance(value, ir.OpResult):
        owner = value.owner.operation
        if owner.name not in _VIEW_OPS:
            return None
        value = owner.operands[0]
    return ir.BlockArgument(value) if isinstance(value, ir.BlockArgument) else None


def _line_write(write: ir.Operation) -> bool:
    """Whether ``write`` stores one cache line, unmasked and in bounds, along the
    last, contiguous dim of a memref."""
    if write.name != _WRITE or len(write.operands) < 3:
        return False
    value_type = write.operands[0].type
    base_type = write.operands[1].type
    if not isinstance(value_type, ir.VectorType) or not isinstance(
        base_type, ir.MemRefType
    ):
        return False
    vector_type = ir.VectorType(value_type)
    memref_type = ir.MemRefType(base_type)
    bytes_ = _element_bytes(vector_type.element_type)
    if (
        vector_type.rank != 1
        or vector_type.scalable
        or bytes_ is None
        or vector_type.shape[0] * bytes_ != _LINE_BYTES
        or len(write.operands) != 2 + memref_type.rank
        or memref_type.get_strides_and_offset()[0][-1] != 1
        or memref_type.element_type != vector_type.element_type
        or not ir.BoolAttr(write.attributes["in_bounds"][0]).value
    ):
        return False
    map_ = ir.AffineMapAttr(write.attributes["permutation_map"]).value
    return map_ == ir.AffineMap.get_minor_identity(memref_type.rank, 1)


def _mulfs(loop: ir.Operation) -> list[ir.Operation]:
    """The AMX multiplies in the body of ``loop``, not in a nested loop."""
    body = loop.regions[0].blocks[0]
    return [op.operation for op in body.operations if op.operation.name == _MULF]


def _constant(value: ir.Value) -> int | None:
    if not isinstance(value, ir.OpResult):
        return None
    owner = value.owner.operation
    if owner.name != "arith.constant":
        return None
    return ir.IntegerAttr(owner.attributes["value"]).value


def _k_elements(loop: ir.Operation) -> int | None:
    """K of an AMX K loop: its steps by the elements of a left tile's row; ``None``
    unless the bounds are constant and the left tiles come from tile loads."""
    mulfs = _mulfs(loop)
    if loop.name != "scf.for" or not mulfs:
        return None
    lower, upper, step = map(_constant, list(loop.operands)[:3])
    if lower is None or upper is None or not step:
        return None
    left = mulfs[0].operands[0]
    if not isinstance(left, ir.OpResult) or left.owner.operation.name != _LOAD:
        return None
    tile = _TILE.fullmatch(str(left.type))
    if tile is None:
        return None
    columns = int(tile.group(1))
    return -(-(upper - lower) // step) * columns


def _before(value: ir.Value, operations: list[ir.Operation], end: int) -> bool:
    """Whether ``value`` is defined outside these operations of a block, or by one
    of the first ``end``."""
    if isinstance(value, ir.BlockArgument):
        return True
    owner = ir.OpResult(value).owner.operation
    return not any(owner == op for op in operations) or any(
        owner == op for op in operations[:end]
    )


def _whole_row_tile(view: ir.Operation, arg: ir.BlockArgument) -> bool:
    """Whether the register block ``view`` is cut from an output tile of whole rows
    of ``arg``: of at least ``MIN_ROWS`` rows and ``MIN_COLUMNS`` columns, its f32
    accumulator within ``MAX_TILE_BYTES``."""
    tile = view.operands[0]
    if not isinstance(tile, ir.OpResult):
        return False
    owner = tile.owner.operation
    if owner.name != "memref.subview" or owner.operands[0] != arg:
        return False
    tile_type = ir.MemRefType(tile.type)
    if tile_type.rank != 2 or not tile_type.has_static_shape:
        return False
    rows, columns = tile_type.shape
    row_elements = 1
    for extent in ir.MemRefType(arg.type).shape[1:]:
        row_elements *= extent
    return (
        rows >= MIN_ROWS
        and columns >= MIN_COLUMNS
        and columns == row_elements
        and rows * columns * 4 <= MAX_TILE_BYTES
    )


def _prefetch_block(
    func: ir.OpView, block: ir.Block, min_bytes: int, max_k: int
) -> None:
    """Prefetches before the first K loop of ``block`` of the lines of the output
    it writes after its last, if it runs at most ``max_k`` of K."""
    operations = [op.operation for op in block.operations]
    loops = [
        i for i, op in enumerate(operations) if op.name == "scf.for" and _mulfs(op)
    ]
    if not loops:
        return
    ks = [_k_elements(operations[i]) for i in loops]
    if None in ks or sum(ks) > max_k:
        return
    first, last = loops[0], loops[-1]
    lines: list[tuple[ir.Operation, ir.Operation]] = []
    for op in operations[last + 1 :]:
        if not _line_write(op):
            continue
        base = op.operands[1]
        if not isinstance(base, ir.OpResult):
            continue
        view = base.owner.operation
        if view.name != "memref.subview":
            continue
        arg = _root(base)
        if arg is None or not _inout(func, arg):
            continue
        size = _contiguous_bytes(arg.type)
        if size is None or size < min_bytes:
            continue
        if not _whole_row_tile(view, arg):
            continue
        if not all(_before(operand, operations, first) for operand in view.operands):
            continue
        if not all(_before(index, operations, first) for index in op.operands[2:]):
            continue
        lines.append((view, op))
    anchor = operations[first]
    for view, _ in lines:
        # A view made in the block moves up to the loop; one from outside stays.
        if any(view == op for op in operations[first + 1 :]):
            view.move_before(anchor)
    for view, write in lines:
        with ir.InsertionPoint(anchor), view.location:
            memref.PrefetchOp(
                view.results[0], list(write.operands[2:]), True, _WRITE_LOCALITY, True
            )


@transform_op(modifies_payload=True)
class PrefetchLargeGemmOp(HelionTransformDialect.Operation, name="prefetch_large_gemm"):
    """In the ``func.func`` targets, for each block holding AMX K loops of at most
    ``max_k`` elements in all and, after them, unmasked in-bounds writes of a 1-D
    vector of 64 bytes along the last, contiguous dim of a subview of an ``inout``
    argument, a static contiguous memref of at least ``min_bytes``: a
    ``memref.prefetch`` for writing of each such line before the first K loop. Other
    IR is left untouched."""

    target: ext.Operand[transform.AnyOpType]
    min_bytes: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), MIN_BYTES
        )
    )
    max_k: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), MAX_K
        )
    )

    @staticmethod
    def run(
        op: "PrefetchLargeGemmOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        min_bytes = ir.IntegerAttr(op.min_bytes).value
        max_k = ir.IntegerAttr(op.max_k).value
        for func in payload_ops(state, op.target, ir.OpView):
            if func.operation.name != "func.func" or not func.regions[0].blocks:
                continue
            blocks: list[ir.Block] = []

            def collect(visited: ir.Operation, blocks: list = blocks) -> ir.WalkResult:
                if visited.name == "scf.for" and _mulfs(visited):
                    block = visited.block
                    if not any(block == seen for seen in blocks):
                        blocks.append(block)
                return ir.WalkResult.ADVANCE

            func.operation.walk(collect, ir.WalkOrder.PRE_ORDER)
            for block in blocks:
                _prefetch_block(func, block, min_bytes, max_k)
        return DiagnosedSilenceableFailure.Success


def prefetch_large_gemm(
    target: ir.Value, min_bytes: int = MIN_BYTES, max_k: int = MAX_K
) -> PrefetchLargeGemmOp:
    """snake_case wrapper to create a PrefetchLargeGemmOp."""
    i64 = ir.IntegerType.get_signless(64)
    return PrefetchLargeGemmOp(
        target=target,
        min_bytes=ir.IntegerAttr.get(i64, min_bytes),
        max_k=ir.IntegerAttr.get(i64, max_k),
    )
