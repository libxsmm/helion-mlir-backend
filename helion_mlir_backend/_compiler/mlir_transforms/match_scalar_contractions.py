"""``match_scalar_contractions``: the named contractions into a rank-0 result,
i.e. dot products (``linalg.dot``, or a ``linalg.contract`` left by folding the
unit dims of a 1x1 output tile). Upstream vectorizes such a named contraction
(``create_named_contraction``) into a ``vector.contract`` with a ``vector<f32>``
accumulator, which does not verify: a rank-0 accumulator must be a scalar.
Generalized, they vectorize as a multi-reduction (raised back to a valid
``vector.contract`` of a scalar).

Matched: ``linalg.contract`` and ``linalg.dot`` ops with one init, a rank-0
tensor or memref. Left alone: other ops, and contractions into rank 1 or more."""

from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op

_CONTRACTIONS = ("linalg.contract", "linalg.dot")


def _is_scalar_contraction(op: ir.Operation) -> bool:
    if op.name not in _CONTRACTIONS:
        return False
    inits = op.opview.outputs
    return (
        len(inits) == 1
        and isinstance(inits[0].type, ir.ShapedType)
        and ir.ShapedType(inits[0].type).rank == 0
    )


@transform_op(modifies_payload=False)
class MatchScalarContractionsOp(
    HelionTransformDialect.Operation, name="match_scalar_contractions"
):
    """The named contractions into a rank-0 result nested in the target (see
    the module docstring)."""

    target: ext.Operand[transform.AnyOpType]
    matched: ext.Result[transform.AnyOpType]

    @staticmethod
    def run(
        op: "MatchScalarContractionsOp",
        _rewriter: transform.TransformRewriter,
        results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        found: list[ir.Operation] = []

        def collect(visited: ir.Operation) -> ir.WalkResult:
            if _is_scalar_contraction(visited):
                found.append(visited)
            return ir.WalkResult.ADVANCE

        for target in state.get_payload_ops(op.target):
            target.operation.walk(collect, ir.WalkOrder.PRE_ORDER)
        results.set_ops(op.matched, found)
        return DiagnosedSilenceableFailure.Success


def match_scalar_contractions(target: ir.Value) -> ir.Value:
    """snake_case wrapper to create a MatchScalarContractionsOp; its result."""
    return MatchScalarContractionsOp(
        target=target, matched=transform.AnyOpType.get()
    ).matched
