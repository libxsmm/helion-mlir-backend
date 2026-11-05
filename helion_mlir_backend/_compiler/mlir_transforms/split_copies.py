"""``split_copies``: an n-D copy (a memref read only written back) of a partial
tile, e.g. a pad or an operand pack, reads out of bounds. Split as a lone read
(``split_transfers``), its edge path yields the whole n-D vector from an
``scf.if``; unrolled to rows, that vector is rebuilt element by element, then
taken apart again for the write: per-element code, megabytes of LLVM IR per
copy. Split together with its write, the copy is one row at a time, with no n-D
vector in flight: in bounds too, as a copy between a contiguous and a strided
memref is otherwise flattened on one side, unrolled to rows on the other, and
rebuilt element by element in between."""

from collections.abc import Callable
import itertools
import math

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import memref
from mlir.dialects import scf
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import payload_ops

# Rows a copy is unrolled to at most; one of more runs them in a loop nest. Left
# as an n-D transfer, each row of an edge tile's read threads the whole vector
# through an ``scf.if``, and a large tile's LLVM IR grows too big to compile.
_MAX_UNROLLED_ROWS = 64
# Views of memory: a copy's read is from the memref they view.
_VIEWS = (
    "memref.subview",
    "memref.expand_shape",
    "memref.collapse_shape",
    "memref.cast",
    "memref.reinterpret_cast",
)
# Ops writing no memory read before (views, allocations, index computations):
# a copy's read may move past them to its write.
_MOVABLE_PAST = (*_VIEWS, "memref.alloc", "memref.alloca", "memref.dim", "ub.poison")
_PURE_DIALECTS = ("arith", "affine", "index")


def _root(value: ir.Value) -> ir.Value:
    """The memref ``value`` is a view of."""
    while isinstance(value.owner, ir.Operation) and value.owner.name in _VIEWS:
        value = value.owner.operands[0]
    return value


def _is_minor_identity(
    transfer: vector.TransferReadOp | vector.TransferWriteOp, rank: int
) -> bool:
    base_rank = ir.MemRefType(transfer.base.type).rank
    minor_identity = ir.AffineMap.get_minor_identity(base_rank, rank)
    return transfer.permutation_map.value == minor_identity


def _write_of(read: vector.TransferReadOp) -> vector.TransferWriteOp | None:
    """The write of the copy ``read`` starts, if it is one to split."""
    vector_type = ir.VectorType(read.result.type)
    if (
        vector_type.rank < 2
        or not isinstance(read.base.type, ir.MemRefType)
        or read.mask is not None
        or not _is_minor_identity(read, vector_type.rank)
    ):
        return None
    uses = list(read.result.uses)
    if len(uses) != 1 or uses[0].operand_number != 0:
        return None
    write = uses[0].owner.opview
    if (
        not isinstance(write, vector.TransferWriteOp)
        or write.operation.block != read.operation.block
        or not isinstance(write.base.type, ir.MemRefType)
        or write.mask is not None
        or not _is_minor_identity(write, vector_type.rank)
        or _root(write.base) == _root(read.base)
    ):
        return None
    flags = [*read.in_bounds, *write.in_bounds]
    if all(ir.BoolAttr(flag).value for flag in flags):
        return None
    # The read moves to the write: nothing in between may write memory.
    between = False
    for op in read.operation.block.operations:
        if op == write.operation:
            return write
        if between and not (
            op.name in _MOVABLE_PAST or op.name.split(".")[0] in _PURE_DIALECTS
        ):
            return None
        between = between or op == read.operation
    return None


class _Side:
    """The memref, indices and bounds of one side of a copy."""

    def __init__(
        self,
        transfer: vector.TransferReadOp | vector.TransferWriteOp,
        vector_rank: int,
        in_bounds: bool = False,
    ) -> None:
        """With ``in_bounds``, the transfer is known to be in bounds."""
        self.base = transfer.base
        self.indices = list(transfer.indices)
        self.in_bounds = [
            in_bounds or ir.BoolAttr(flag).value for flag in transfer.in_bounds
        ]
        self.leading = ir.MemRefType(self.base.type).rank - vector_rank
        self.row_map = ir.AffineMap.get_minor_identity(
            ir.MemRefType(self.base.type).rank, 1
        )
        self.extents: dict[int, ir.Value] = {}

    def extent(self, dim: int) -> ir.Value:
        """The size of the base along vector dim ``dim``."""
        if dim not in self.extents:
            index = ir.IndexType.get()
            dim_index = arith.ConstantOp(index, self.leading + dim).result
            self.extents[dim] = memref.DimOp(self.base, dim_index).result
        return self.extents[dim]

    def fits(self, shape: list[int], fits: ir.Value | None = None) -> ir.Value | None:
        """Whether a vector of ``shape`` is in bounds, and ``fits`` (None: true)."""
        index = ir.IndexType.get()
        for dim, (size, flag) in enumerate(zip(shape, self.in_bounds, strict=True)):
            if flag:
                continue
            start = self.indices[self.leading + dim]
            end = arith.AddIOp(start, arith.ConstantOp(index, size)).result
            dim_fits = arith.CmpIOp(
                arith.CmpIPredicate.sle, end, self.extent(dim)
            ).result
            fits = dim_fits if fits is None else arith.AndIOp(fits, dim_fits).result
        return fits

    def row_mask(self, width: int) -> ir.Value | None:
        """The lanes of a row in bounds (None: all are)."""
        if self.in_bounds[-1]:
            return None
        start = self.indices[-1]
        lanes = arith.SubIOp(self.extent(len(self.in_bounds) - 1), start).result
        mask_type = ir.VectorType.get([width], ir.IntegerType.get_signless(1))
        return vector.CreateMaskOp(mask_type, [lanes]).result

    def row(
        self, position: tuple[int | ir.Value, ...]
    ) -> tuple[list[ir.Value], ir.Value | None]:
        """The indices of row ``position`` (constants or loop values) and whether
        it is in bounds (None: it always is)."""
        index = ir.IndexType.get()
        indices = list(self.indices)
        inside = None
        for dim, offset in enumerate(position):
            start = indices[self.leading + dim]
            if isinstance(offset, ir.Value):
                start = arith.AddIOp(start, offset).result
            elif offset:
                start = arith.AddIOp(start, arith.ConstantOp(index, offset)).result
            indices[self.leading + dim] = start
            if self.in_bounds[dim]:
                continue
            row_inside = arith.CmpIOp(
                arith.CmpIPredicate.slt, start, self.extent(dim)
            ).result
            inside = (
                row_inside
                if inside is None
                else arith.AndIOp(inside, row_inside).result
            )
        return indices, inside


def _copy_rows(
    source: _Side, dest: _Side, padding: ir.Value, vector_type: ir.VectorType
) -> None:
    """Copy one row at a time: rows past the source's end are the padding, rows
    past the destination's end are not written."""
    width = vector_type.shape[-1]
    row_type = ir.VectorType.get([width], vector_type.element_type)
    read_mask, write_mask = source.row_mask(width), dest.row_mask(width)

    def copy_row(position: tuple[int | ir.Value, ...]) -> None:
        indices, inside = source.row(position)
        if inside is None:
            row = _read_row(source, indices, padding, row_type, read_mask)
        else:
            branch = scf.IfOp(inside, [row_type], has_else=True)
            with ir.InsertionPoint(branch.then_block):
                scf.YieldOp([_read_row(source, indices, padding, row_type, read_mask)])
            with ir.InsertionPoint(branch.else_block):
                scf.YieldOp([vector.BroadcastOp(row_type, padding).result])
            row = branch.results[0]
        indices, inside = dest.row(position)
        if inside is None:
            _write_row(row, dest, indices, write_mask)
        else:
            guard = scf.IfOp(inside)
            with ir.InsertionPoint(guard.then_block):
                _write_row(row, dest, indices, write_mask)
                scf.YieldOp([])

    leading = vector_type.shape[:-1]
    if math.prod(leading) <= _MAX_UNROLLED_ROWS:
        for position in itertools.product(*map(range, leading)):
            copy_row(position)
    else:
        _in_loops(leading, copy_row)


def _in_loops(sizes: list[int], body: Callable, outer: tuple = ()) -> None:
    """``body(position)`` in a nest of loops over ``sizes``."""
    if not sizes:
        body(outer)
        return
    index = ir.IndexType.get()
    zero = arith.ConstantOp(index, 0).result
    one = arith.ConstantOp(index, 1).result
    loop = scf.ForOp(zero, arith.ConstantOp(index, sizes[0]).result, one, [])
    with ir.InsertionPoint(loop.body):
        _in_loops(sizes[1:], body, (*outer, loop.induction_variable))
        scf.YieldOp([])


def _read_row(
    source: _Side,
    indices: list[ir.Value],
    padding: ir.Value,
    row_type: ir.VectorType,
    mask: ir.Value | None,
) -> ir.Value:
    return vector.TransferReadOp(
        row_type, source.base, indices, source.row_map, padding, [True], mask=mask
    ).result


def _write_row(
    row: ir.Value, dest: _Side, indices: list[ir.Value], mask: ir.Value | None
) -> None:
    vector.TransferWriteOp(
        None, row, dest.base, indices, dest.row_map, [True], mask=mask
    )


def _split(read: vector.TransferReadOp, rewriter: transform.TransformRewriter) -> None:
    write = _write_of(read)
    if write is None:
        return
    vector_type = ir.VectorType(read.result.type)
    shape, rank = list(vector_type.shape), vector_type.rank
    source, dest = _Side(read, rank), _Side(write, rank)
    with ir.InsertionPoint(write), write.location:
        fits = source.fits(shape, dest.fits(shape))
        branch = scf.IfOp(fits, [], has_else=True)
        # In bounds too, rows: a copy between a contiguous and a strided memref
        # unrolls one side to rows and flattens the other, element by element.
        with ir.InsertionPoint(branch.then_block):
            _copy_rows(
                _Side(read, rank, in_bounds=True),
                _Side(write, rank, in_bounds=True),
                read.padding,
                vector_type,
            )
            scf.YieldOp([])
        with ir.InsertionPoint(branch.else_block):
            _copy_rows(source, dest, read.padding, vector_type)
            scf.YieldOp([])
    rewriter.erase_op(write)
    rewriter.erase_op(read)


@transform_op(modifies_payload=True)
class SplitCopiesOp(HelionTransformDialect.Operation, name="split_copies"):
    """Split every n-D copy in the target, a minor-identity, unmasked memref
    ``vector.transfer_read`` possibly out of bounds whose only user writes it to
    another memref, on an in-bounds check, into copies row by row: in bounds if
    it passes, with rows padded, masked and skipped otherwise."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "SplitCopiesOp",
        rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for read in payload_ops(state, op.target, vector.TransferReadOp):
            _split(read, rewriter)
        return DiagnosedSilenceableFailure.Success


def split_copies(target: ir.Value) -> SplitCopiesOp:
    """snake_case wrapper to create a SplitCopiesOp."""
    return SplitCopiesOp(target=target)
