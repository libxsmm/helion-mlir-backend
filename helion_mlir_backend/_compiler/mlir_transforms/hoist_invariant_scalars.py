"""``hoist_invariant_scalars``: scalar code that does not depend on a loop moved
out of it, e.g. the casts of runtime scalars in a GPU kernel's grid loop, so the
kernel takes the values it uses (an ``f32`` scale, not the ``f64`` argument it
is computed from)."""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op


def is_scalar(value_type: ir.Type) -> bool:
    """An integer, index or float type: no vector, tensor or buffer."""
    return isinstance(value_type, (ir.IntegerType, ir.IndexType, ir.FloatType))


def is_pure(op: ir.Operation) -> bool:
    """No memory effects and speculatable (upstream LICM's condition): the op can
    run where its operands are defined, even if the loop runs no iteration."""
    try:
        effects = ir.MemoryEffectsOpInterface(op).get_effects()
        speculatability = ir.ConditionallySpeculatable(op).getSpeculatability()
    except ValueError:
        return False
    return not effects and speculatability == ir.Speculatability.Speculatable


def defined_outside(value: ir.Value, op: ir.Operation) -> bool:
    """Whether ``value`` is defined outside ``op`` (not in its regions)."""
    owner = value.owner
    current = owner.owner if isinstance(owner, ir.Block) else owner
    current = current.operation if current is not None else None
    while current is not None:
        if current == op:
            return False
        current = current.parent
    return True


def _nested(op: ir.Operation) -> list[ir.Operation]:
    """The ops nested in ``op``, in pre-order (``op`` excluded)."""
    found: list[ir.Operation] = []

    def visit(nested: ir.Operation) -> ir.WalkResult:
        if nested != op:
            found.append(nested)
        return ir.WalkResult.ADVANCE

    op.walk(visit, ir.WalkOrder.PRE_ORDER)
    return found


def _hoistable(op: ir.Operation, loop: ir.Operation) -> bool:
    operands = list(op.operands)
    return (
        not op.regions
        # Constants stay: they are cheaper to rematerialize than to pass along.
        and bool(operands)
        and all(is_scalar(value.type) for value in [*operands, *op.results])
        and all(defined_outside(value, loop) for value in operands)
        and is_pure(op)
    )


@transform_op(modifies_payload=True)
class HoistInvariantScalarsOp(
    HelionTransformDialect.Operation, name="hoist_invariant_scalars"
):
    """Move the scalar code nested in each target op (a loop) that does not depend
    on it to right before it.

    An op moves if it has no regions, takes and returns only scalars (integer,
    index, float), takes at least one operand and only operands defined outside
    the target (or by ops moved before), and is free of memory effects and
    speculatable. Vector, tensor and buffer code stays, as do constants.
    """

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "HoistInvariantScalarsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for loop in state.get_payload_ops(op.target):
            loop = loop.operation
            # Pre-order: definitions move before their uses are visited.
            for candidate in _nested(loop):
                if _hoistable(candidate, loop):
                    candidate.move_before(loop)
        return DiagnosedSilenceableFailure.Success


def hoist_invariant_scalars(target: ir.Value) -> HoistInvariantScalarsOp:
    """snake_case wrapper to create a HoistInvariantScalarsOp."""
    return HoistInvariantScalarsOp(target=target)
