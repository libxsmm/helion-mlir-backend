"""NaN-propagating minima and maxima in torch-mlir's ATen helpers.

torch-mlir (wheel 20261001) lowers ``aten.clamp`` and its relatives to
compare-and-select pairs that pick the bound for a NaN input:
``select(cmpf ult x, lo, lo, x)`` and ``select(cmpf ugt x, hi, hi, x)``
(``torch.clamp(nan, -1, 1)`` gives -1), and ``aten.maximum`` / ``aten.minimum``
to ``select(cmpf ogt a, b, a, b)``, which gives ``b`` for a NaN ``a``. Torch's
minima and maxima propagate NaN from any operand.

Only the helpers of :data:`NAN_PROPAGATING_OPS` are rewritten: in them, every
``select(cmpf P a, b, X, Y)`` with ``{X, Y} = {a, b}`` and an ordering predicate
``P`` is a minimum or a maximum (the same pattern means something else in
other ops: ``torch.where(x > 0, x, 0)`` gives 0 for NaN). For ordered operands
nothing changes (also signed zeros, which ``arith.maximumf`` would order);
a NaN operand now gives NaN:

- the operand picked when the comparison is unordered (``X`` for ``u``
  predicates, ``Y`` for ``o`` ones) is NaN whenever the other is a non-NaN
  constant: left alone (e.g. ReLU's ``select(ugt x, 0, x, 0)``);
- if the picked one is a non-NaN constant (a clamp bound), the predicate's
  ordered and unordered forms are swapped, so a NaN input is picked;
- otherwise (two runtime values) a NaN of the other operand is selected
  after the pair: ``select(cmpf uno v, v, v, r)``.
"""

from __future__ import annotations

import math

from mlir import ir
from mlir.dialects import arith
import torch

_aten = torch.ops.aten
NAN_PROPAGATING_OPS = frozenset(
    {
        _aten.clamp,
        _aten.clamp_min,
        _aten.clamp_max,
        _aten.hardtanh,
        _aten.relu6,
        _aten.hardsigmoid,
        _aten.maximum,
        _aten.minimum,
    }
)
"""ATen ops whose results are NaN-propagating minima/maxima of their operands
(or of a value computed from them, e.g. ``hardsigmoid``'s ``x / 6 + 0.5``)."""

# arith.cmpf predicates: the ordered forms ogt, oge, olt, ole are 2-5, their
# unordered forms ugt, uge, ult, ule 9-12.
_ORDERED = range(2, 6)
_UNORDERED = range(9, 13)
_UNO = 14


def _non_nan_constant(value: ir.Value) -> bool:
    """Whether ``value`` is a float constant (scalar or splat) that is not NaN."""
    if not isinstance(value, ir.OpResult) or value.owner.name != "arith.constant":
        return False
    attribute = value.owner.attributes["value"]
    if isinstance(attribute, ir.FloatAttr):
        return not math.isnan(attribute.value)
    if isinstance(attribute, ir.DenseFPElementsAttr) and attribute.is_splat:
        return not math.isnan(ir.FloatAttr(attribute.get_splat_value()).value)
    return False


def _propagate(select: ir.Operation) -> None:
    condition, true_value, false_value = select.operands
    if not isinstance(condition, ir.OpResult) or condition.owner.name != "arith.cmpf":
        return
    compare = condition.owner
    lhs, rhs = compare.operands
    if true_value == false_value or not (
        (true_value == lhs and false_value == rhs)
        or (true_value == rhs and false_value == lhs)
    ):
        return
    predicate = ir.IntegerAttr(compare.attributes["predicate"]).value
    if predicate in _UNORDERED:
        picked, other = true_value, false_value
    elif predicate in _ORDERED:
        picked, other = false_value, true_value
    else:
        return
    if _non_nan_constant(other):
        return
    if _non_nan_constant(picked):
        swapped = predicate + 7 if predicate in _ORDERED else predicate - 7
        with ir.InsertionPoint(compare), compare.location:
            flipped = arith.CmpFOp(swapped, lhs, rhs).result
        select.operands[0] = flipped
        if not list(compare.results[0].uses):
            compare.erase()
        return
    result = select.results[0]
    with ir.InsertionPoint.after(select), select.location:
        is_nan = arith.CmpFOp(_UNO, other, other).result
        guarded = arith.SelectOp(is_nan, other, result)
    result.replace_all_uses_except(guarded.result, guarded.operation)


def propagate_nan(function: ir.Operation) -> None:
    """Make the compare-and-select minima and maxima in ``function`` (a helper
    of one of :data:`NAN_PROPAGATING_OPS`) propagate NaN (see the module)."""
    selects: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "arith.select":
            selects.append(op)
        return ir.WalkResult.ADVANCE

    function.walk(visit)
    for select in selects:
        _propagate(select)
