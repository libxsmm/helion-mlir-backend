"""``widen_misread_amx_operands``: on an AMX target, a bf16 x bf16 -> f32
``vector.contract`` whose LHS or RHS upstream's AMX rewrite would misread is
moved off AMX: both operands are widened to f32 (``arith.extf``).

Upstream's x86 AMX rewrite (``x86.vector_contract_to_amx_dot_product``) takes
an operand's ``vector.transfer_read`` apart: it loads the tile (or the rows it
packs to VNNI) from the read's memref at the read's indices, row by row along
the memref's two inner dims. It ignores the read's permutation map, its mask
(operand or enclosing ``vector.mask``) and its ``in_bounds`` flags (the
padding of an out-of-bounds read), and ``x86.amx.tile_load`` requires a unit
innermost stride. Any other operand read is loaded as if it were the plain
read: silently wrong values (a transposing or unit-dim-skipping map, a mask,
padding), or a failure to compile (``tile_load requires unit innermost
stride``). The other layout fixes (``normalize_amx_contractions``,
``drop_contraction_read_unit_dims``) make the reads they match plain reads;
this is the safety net for every read they leave.

Matched: a ``vector.contract`` with bf16 LHS and RHS and an f32 accumulator,
on a target with AMX, whose LHS or RHS is not a plain read: the result of a
``vector.transfer_read`` of a memref with a strided layout and a static unit
innermost stride, without mask (no mask operand, not in ``vector.mask``),
with every dim in bounds and a minor-identity permutation map. Both operands
are then widened to f32: the rewrite takes only bf16 operands, so the
contraction lowers on the f32 path (exact products of bf16 values, f32
sums), as on a host without AMX.

In a K loop (the accumulator an iter arg of the ``scf.for`` around the
contraction), the rewrite rebuilds the loop: it re-creates each operand's
view per iteration by cloning the op defining the read's memref with the
loop's induction variable substituted for the offset it finds equal to it,
and packs a flat RHS (``K x N``) to VNNI from the origin of that view,
ignoring the read's indices (also software-pipelined: the next iteration's
RHS is packed before the current one's tile products). Also misread there,
so also matched:
- an LHS or RHS whose memref is not a ``memref.subview`` with the loop's
  induction variable among its offsets: e.g. a buffer the loop copies the
  RHS block into, viewed above the loop (``align_operand_rows``, K tile of
  32). The rewrite takes an offset operand the view does not have
  (``MutableArrayRef<OpOperand>::operator[]`` "Invalid index!"), or another
  offset for the induction variable, or packs the buffer before the loop
  writes it;
- a flat RHS read at other indices than its view's origin, or column 16 of
  it (the pair the rewrite packs together): e.g. columns 1 and 17 of a
  column-shifted tile ``b[tk, tn.index + 1]`` whose view could not be
  re-based (``fit_subviews_to_reads``), rows of a K-shifted one.
VNNI operands (``... x 2``) are loaded with tile loads at the read's
indices, so only the first applies to them.

The rewrite loads an operand, and the accumulator, at the contraction, not at
its read. Also matched: a read above the contraction of a memref written between
the two, in a loop around the contraction too, or of memory that can't be told
apart from what is written: a root other than an allocation or a function
argument, read or written (e.g. through an ``scf.if``). One-shot bufferization may
give two pads of one shape one buffer, as a read vector needs no memory once it's
read: a query tile read above a key loop whose tile lands there would load the
keys, silently (a 32x32 query and key tile of padded columns,
``tests/test_amx_stale_reads.py``).

Left alone: contractions with only plain operand reads the rewrite reads
right, of other element types, and every contraction on a target without
AMX (the rewrite matches no register tile there).
"""

from lighthouse.execution.target import TargetInfo
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
from .utils import int_constant
from .utils import opview
from .utils import payload_ops

# The column of the second RHS read of a pair the rewrite packs together.
_PAIR_COLUMN = 16


# Views of memory, and ops that only read it.
_VIEWS = (
    "memref.subview",
    "memref.expand_shape",
    "memref.collapse_shape",
    "memref.cast",
    "memref.reinterpret_cast",
)
_NOT_WRITING = (
    *_VIEWS,
    "vector.transfer_read",
    "vector.load",
    "vector.maskedload",
    "memref.dim",
    "memref.load",
    "memref.extract_aligned_pointer_as_index",
)


def _element_type(value: ir.Value) -> ir.Type | None:
    if not isinstance(value.type, ir.VectorType):
        return None
    return ir.VectorType(value.type).element_type


def _is_plain_read(value: ir.Value) -> bool:
    """Whether ``value`` is a read the AMX rewrite loads right (see the module
    docstring)."""
    if not isinstance(value, ir.OpResult):
        return False
    read = opview(value.owner)
    if not isinstance(read, vector.TransferReadOp):
        return False
    if read.mask is not None or read.operation.parent.name == "vector.mask":
        return False
    if not isinstance(read.base.type, ir.MemRefType):
        return False
    if not all(ir.BoolAttr(flag).value for flag in read.in_bounds):
        return False
    base_type = ir.MemRefType(read.base.type)
    try:
        strides, _ = base_type.get_strides_and_offset()
    except (RuntimeError, ValueError):
        return False
    if not strides or strides[-1] != 1:
        return False
    permutation = read.permutation_map.value
    return permutation == ir.AffineMap.get_minor_identity(
        base_type.rank, len(permutation.results)
    )


def _k_loop(op: vector.ContractionOp) -> scf.ForOp | None:
    """The ``scf.for`` whose body holds ``op`` and whose iter arg is its
    accumulator: the K loop the AMX rewrite rebuilds."""
    acc = op.acc
    if not isinstance(acc, ir.BlockArgument) or acc.owner != op.operation.block:
        return None
    loop = acc.owner.owner
    if loop is None or loop.name != "scf.for" or acc.arg_number == 0:
        return None
    return opview(loop)


def _is_loop_view(read: vector.TransferReadOp, loop: scf.ForOp) -> bool:
    """Whether ``read``'s memref is a subview at ``loop``'s induction
    variable (the view the rewrite re-creates per iteration)."""
    base = read.base
    if not isinstance(base, ir.OpResult):
        return False
    view = opview(base.owner)
    return isinstance(view, memref.SubViewOp) and any(
        offset == loop.induction_variable for offset in view.offsets
    )


def _is_packed_from_origin(read: vector.TransferReadOp) -> bool:
    """Whether the rewrite's VNNI packing of the flat RHS ``read`` (32 x 32
    from its view's origin, paired along N) gets its elements: indices all
    0 but the last, 0 or 16."""
    indices = [int_constant(index) for index in read.indices]
    return all(index == 0 for index in indices[:-1]) and indices[-1:] in (
        [0],
        [_PAIR_COLUMN],
    )


def _is_vnni(value: ir.Value) -> bool:
    shape = ir.VectorType(value.type).shape
    return len(shape) > 2 and shape[-1] == 2


def _is_misread_in_k_loop(op: vector.ContractionOp) -> bool:
    loop = _k_loop(op)
    if loop is None:
        return False
    lhs, rhs = (opview(value.owner) for value in (op.lhs, op.rhs))
    if not (_is_loop_view(lhs, loop) and _is_loop_view(rhs, loop)):
        return True
    return not _is_vnni(op.rhs) and not _is_packed_from_origin(rhs)


def _root(value: ir.Value) -> ir.Value:
    """The memref ``value`` is a view of."""
    while isinstance(value, ir.OpResult) and value.owner.name in _VIEWS:
        value = value.owner.operands[0]
    return value


def _is_distinct_root(root: ir.Value) -> bool:
    """Whether memory of ``root`` can be told apart from that of other roots: an
    allocation or a function argument."""
    if isinstance(root, ir.OpResult):
        return root.owner.name in ("memref.alloca", "memref.alloc")
    return root.owner.owner.operation.name == "func.func"


def _may_write(op: ir.Operation, root: ir.Value) -> bool:
    """Whether ``op`` or an op nested in it may write ``root`` or a view of it."""
    found = False

    def visit(nested: ir.Operation) -> ir.WalkResult:
        nonlocal found
        if nested.name in _NOT_WRITING:
            return ir.WalkResult.ADVANCE
        operands = list(nested.operands)
        if nested.name == "memref.copy":
            operands = operands[1:]
        for operand in operands:
            if not isinstance(operand.type, ir.MemRefType):
                continue
            written = _root(operand)
            if written == root or not _is_distinct_root(written):
                found = True
                return ir.WalkResult.INTERRUPT
        return ir.WalkResult.ADVANCE

    op.walk(visit)
    return found


def _is_stale_read(value: ir.Value, op: vector.ContractionOp) -> bool:
    """Whether the read ``value`` of ``op`` is of memory written after it, before
    ``op`` or in a loop around it, or of memory that can't be told apart."""
    if not isinstance(value, ir.OpResult):
        return False
    read = opview(value.owner)
    if not isinstance(read, vector.TransferReadOp):
        return False
    root = _root(read.base)
    if not _is_distinct_root(root):
        return True
    block = read.operation.block
    ancestor = op.operation
    while ancestor.block != block:
        ancestor = ancestor.parent
        if ancestor is None or not isinstance(ancestor, ir.Operation):
            return False
    after = False
    for between in block.operations:
        if between == read.operation:
            after = True
        elif after and _may_write(between, root):
            return True
        if between == ancestor:
            return False
    return False


def _is_misread(op: vector.ContractionOp) -> bool:
    lhs, rhs, acc = op.lhs, op.rhs, op.acc
    bf16 = ir.BF16Type
    if not (
        isinstance(_element_type(lhs), bf16)
        and isinstance(_element_type(rhs), bf16)
        and isinstance(_element_type(acc), ir.F32Type)
    ):
        return False
    if not (_is_plain_read(lhs) and _is_plain_read(rhs)):
        return True
    return _is_misread_in_k_loop(op) or any(
        _is_stale_read(value, op) for value in (lhs, rhs, acc)
    )


def _widen(op: vector.ContractionOp) -> None:
    f32 = ir.F32Type.get()
    with ir.InsertionPoint(op), op.location:
        for index in (0, 1):
            value = op.operation.operands[index]
            shape = ir.VectorType(value.type).shape
            op.operation.operands[index] = arith.extf(
                ir.VectorType.get(shape, f32), value
            )


@transform_op(modifies_payload=True)
class WidenMisreadAmxOperandsOp(
    HelionTransformDialect.Operation, name="widen_misread_amx_operands"
):
    """Widen to f32 the operands of every bf16 contraction in the target that
    upstream's AMX rewrite would misread (see the module docstring)."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "WidenMisreadAmxOperandsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        if not TargetInfo.host().is_supported("amx"):
            return DiagnosedSilenceableFailure.Success
        for contract in payload_ops(state, op.target, vector.ContractionOp):
            if _is_misread(contract):
                _widen(contract)
        return DiagnosedSilenceableFailure.Success


def widen_misread_amx_operands(target: ir.Value) -> WidenMisreadAmxOperandsOp:
    """snake_case wrapper to create a WidenMisreadAmxOperandsOp."""
    return WidenMisreadAmxOperandsOp(target=target)
