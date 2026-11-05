"""``sink_copies_into_branches``: the copy of an ``scf.if`` result into a
target, made at the end of each branch instead, and not at all in a branch
that yields the target itself.

A grid tile that may be partial is versioned on being full (``lowering/loops.py``): the
full version writes the output's region in place, the partial one a padded temporary.
Bufferized, the ``scf.if`` yields either buffer and its result is copied into the region
after it, for every tile: a full tile copies its region onto itself, through the runner
library's generic strided ``memrefCopy``. Canonicalization does not erase such a copy
when a cast between the two changes the static shape or layout, so the transform does
not make it.

Matched exactly:
- a ``memref.copy`` whose source is a result of an ``scf.if`` with an else
  branch, or a ``memref.cast`` of one, in the same block;
- the result (and the cast) used by nothing else, except the yield of an
  enclosing ``if`` whose result is unused (left by a sinking before);
- between the ``if`` and the copy, only ops without memory effects (subviews,
  casts, index arithmetic), so moving the copy before them reorders no read
  or write of any buffer;
- a target available before the ``if``, or a ``memref.subview`` of values
  that are, used only by the copy (moved before the ``if``).
Repeated until none matches, for the copies made in the branches of nested
``if``s (a tile versioned along two dims). Other copies are left unchanged.

At run time one branch executes, and its copy is its last op: the target gets
the value the original copy wrote, after the same reads and writes.
"""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import memref
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op


def _ops_named(root: ir.Operation, name: str) -> list[ir.Operation]:
    """The ``name`` ops nested in ``root`` (``root`` included), in walk order."""
    found: list[ir.Operation] = []

    def visit(nested: ir.Operation) -> ir.WalkResult:
        if nested.name == name:
            found.append(nested)
        return ir.WalkResult.ADVANCE

    root.walk(visit)
    return found


def _defined_before(value: ir.Value, op: ir.Operation) -> bool:
    """Whether ``value`` is available at ``op``: an argument of a block
    enclosing ``op``, or the result of an op before ``op`` (or before an op
    enclosing it) in the same block."""
    if isinstance(value, ir.BlockArgument):
        block = value.owner
        current = op
        while current is not None:
            if current.operation.block == block:
                return True
            current = current.parent
        return False
    defining = value.owner
    current = op
    while current is not None:
        if defining.operation.block == current.operation.block:
            return _precedes(defining, current)
        current = current.parent
    return False


def _precedes(first: ir.Operation, second: ir.Operation) -> bool:
    """Whether ``first`` comes before ``second`` in their common block."""
    for nested in first.operation.block.operations:
        if nested == second:
            return False
        if nested == first:
            return True
    return False


def _cast_of_if(value: ir.Value, copy: ir.Operation) -> ir.Operation | None:
    """The ``memref.cast`` of an ``scf.if`` result that ``value`` is, if only
    ``copy`` and dead yields (:func:`_dead_yield`) use it: a copy sunk into a
    branch that yields a nested ``if``'s result cast to the outer type."""
    if not isinstance(value, ir.OpResult) or value.owner.name != "memref.cast":
        return None
    operand = value.owner.operands[0]
    if not isinstance(operand, ir.OpResult) or operand.owner.name != "scf.if":
        return None
    if any(use.owner != copy and not _dead_yield(use) for use in value.uses):
        return None
    return value.owner


def _dead_yield(use: ir.OpOperand) -> bool:
    """Whether ``use`` is an operand of the ``scf.yield`` of an ``scf.if``
    whose result for it is unused (its copy was sunk before)."""
    user = use.owner
    if user.name != "scf.yield" or user.parent.name != "scf.if":
        return False
    return not list(user.parent.results[use.operand_number].uses)


# Ops without memory effects that may sit between the ``if`` and its copy.
_PURE_PREFIXES = ("arith.", "index.")
_PURE_OPS = frozenset(
    {
        "memref.subview",
        "memref.cast",
        "memref.dim",
        "memref.extract_strided_metadata",
        "affine.apply",
        "affine.min",
        "affine.max",
    }
)


def _only_pure_between(first: ir.Operation, second: ir.Operation) -> bool:
    """Whether every op strictly between ``first`` and ``second`` (in one block,
    ``first`` before ``second``) has no memory effects."""
    inside = False
    for nested in first.operation.block.operations:
        if nested == second:
            return inside
        if inside and not (
            nested.name in _PURE_OPS or nested.name.startswith(_PURE_PREFIXES)
        ):
            return False
        if nested == first:
            inside = True
    return False


def _sinkable(copy: ir.Operation) -> tuple[ir.Operation, int] | None:
    """The ``scf.if`` and result index whose copy ``copy`` is, if it matches."""
    source, target = copy.operands[0], copy.operands[1]
    cast = _cast_of_if(source, copy)
    if cast is not None:
        source = cast.operands[0]
    if not isinstance(source, ir.OpResult) or source.owner.name != "scf.if":
        return None
    if_op = source.owner
    if len(if_op.regions[1].blocks) == 0:
        return None
    if if_op.operation.block != copy.operation.block:
        return None
    if not _only_pure_between(if_op, copy):
        return None
    user = copy if cast is None else cast
    if any(use.owner != user and not _dead_yield(use) for use in source.uses):
        return None
    if not _defined_before(target, if_op):
        producer = target.owner if isinstance(target, ir.OpResult) else None
        if producer is None or producer.name != "memref.subview":
            return None
        if producer.operation.block != copy.operation.block:
            return None
        if len(list(target.uses)) != 1:
            return None
        if not all(_defined_before(operand, if_op) for operand in producer.operands):
            return None
    return if_op, source.result_number


def _uncast(value: ir.Value) -> ir.Value:
    """``value`` through any ``memref.cast``s."""
    while isinstance(value, ir.OpResult) and value.owner.name == "memref.cast":
        value = value.owner.operands[0]
    return value


def _same_view(first: ir.Value, second: ir.Value) -> bool:
    """Whether ``first`` and ``second`` are the same buffer: the same value, or
    ``memref.subview``s of the same operands and static offsets, sizes and
    strides (not yet merged by CSE)."""
    if first == second:
        return True
    views = [
        value.owner
        for value in (first, second)
        if isinstance(value, ir.OpResult) and value.owner.name == "memref.subview"
    ]
    if len(views) != 2:
        return False
    a, b = views
    if list(a.operands) != list(b.operands) or a.results[0].type != b.results[0].type:
        return False
    names = ("static_offsets", "static_sizes", "static_strides")
    return all(str(a.attributes[name]) == str(b.attributes[name]) for name in names)


def _sink(copy: ir.Operation, if_op: ir.Operation, index: int) -> None:
    target = copy.operands[1]
    cast = _cast_of_if(copy.operands[0], copy)
    if not _defined_before(target, if_op):
        target.owner.move_before(if_op)
    for region in if_op.regions:
        block = region.blocks[0]
        terminator = block.operations[len(block.operations) - 1]
        yielded = terminator.operands[index]
        if _same_view(_uncast(yielded), _uncast(target)):
            # The branch wrote the target in place: no copy onto itself.
            continue
        with ir.InsertionPoint(terminator), copy.location:
            memref.CopyOp(yielded, target)
    copy.erase()
    if cast is not None and not list(cast.results[0].uses):
        cast.erase()


@transform_op(modifies_payload=True)
class SinkCopiesIntoBranchesOp(
    HelionTransformDialect.Operation, name="sink_copies_into_branches"
):
    """In the target ops, replace each matching ``memref.copy`` of an ``scf.if``
    result (see the module docstring) by a copy of the yielded buffer at the
    end of each branch that does not yield the target itself. Other copies are
    left unchanged."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "SinkCopiesIntoBranchesOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for target in state.get_payload_ops(op.target):
            # Until none is left: a copy made in a branch may copy a nested
            # ``if`` (a tile versioned along two dims).
            while True:
                copies = _ops_named(target.operation, "memref.copy")
                matches = [(copy, _sinkable(copy)) for copy in copies]
                matches = [(copy, match) for copy, match in matches if match]
                if not matches:
                    break
                for copy, match in matches:
                    _sink(copy, *match)
        return DiagnosedSilenceableFailure.Success


def sink_copies_into_branches(target: ir.Value) -> SinkCopiesIntoBranchesOp:
    """snake_case wrapper to create a SinkCopiesIntoBranchesOp."""
    return SinkCopiesIntoBranchesOp(target=target)
