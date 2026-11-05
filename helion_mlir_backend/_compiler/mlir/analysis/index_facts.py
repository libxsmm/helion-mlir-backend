"""What the build context can prove about an ``index`` value of a loop bound.

A loop may end at an enclosing tile's position (``hl.tile(tm.end)``, a triangular
iteration). Its bounds are then values, not constants, but they are still multiples
of the tile sizes they are built from and stay within the enclosing loop. These
facts decide whether its tiles are full and inside the tensor, which targets without
masked loads need (:meth:`BuildContext.bind_loop`, ``tile_window``).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import TYPE_CHECKING

from mlir import ir

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class IndexFacts:
    """``value % multiple == 0`` and ``low <= value <= high`` (``None``: unknown).
    ``multiple`` 0 means the value is 0 modulo everything: it is zero."""

    multiple: int = 1
    low: int | None = None
    high: int | None = None


def _add(a: IndexFacts, b: IndexFacts) -> IndexFacts:
    low = a.low + b.low if a.low is not None and b.low is not None else None
    high = a.high + b.high if a.high is not None and b.high is not None else None
    return IndexFacts(gcd(a.multiple, b.multiple), low, high)


def _negate(a: IndexFacts) -> IndexFacts:
    return IndexFacts(
        a.multiple,
        -a.high if a.high is not None else None,
        -a.low if a.low is not None else None,
    )


def _scale(a: IndexFacts, factor: int) -> IndexFacts:
    """``a`` times a constant."""
    if factor < 0:
        return _scale(_negate(a), -factor)
    return IndexFacts(
        a.multiple * factor,
        a.low * factor if a.low is not None else None,
        a.high * factor if a.high is not None else None,
    )


def _constant(value: int) -> IndexFacts:
    return IndexFacts(abs(value), value, value)


def _floordiv(a: IndexFacts, divisor: int) -> IndexFacts:
    if divisor <= 0:
        return IndexFacts()
    return IndexFacts(
        a.multiple // divisor if a.multiple % divisor == 0 else 1,
        a.low // divisor if a.low is not None else None,
        a.high // divisor if a.high is not None else None,
    )


def _max(a: IndexFacts, b: IndexFacts) -> IndexFacts:
    """Either value, so their common multiple; at least the larger known low bound."""
    lows = [x.low for x in (a, b) if x.low is not None]
    high = max(a.high, b.high) if a.high is not None and b.high is not None else None
    return IndexFacts(gcd(a.multiple, b.multiple), max(lows, default=None), high)


def _min(a: IndexFacts, b: IndexFacts) -> IndexFacts:
    """Either value, so their common multiple; at most the smaller known high bound."""
    highs = [x.high for x in (a, b) if x.high is not None]
    low = min(a.low, b.low) if a.low is not None and b.low is not None else None
    return IndexFacts(gcd(a.multiple, b.multiple), low, min(highs, default=None))


def _affine_facts(
    expr: ir.AffineExpr, operands: list[IndexFacts], dims: int
) -> IndexFacts:
    if isinstance(expr, ir.AffineConstantExpr):
        return _constant(expr.value)
    if isinstance(expr, ir.AffineDimExpr):
        return operands[expr.position]
    if isinstance(expr, ir.AffineSymbolExpr):
        return operands[dims + expr.position]
    if isinstance(expr, ir.AffineAddExpr):
        return _add(
            _affine_facts(expr.lhs, operands, dims),
            _affine_facts(expr.rhs, operands, dims),
        )
    if isinstance(expr, ir.AffineMulExpr) and isinstance(
        expr.rhs, ir.AffineConstantExpr
    ):
        return _scale(_affine_facts(expr.lhs, operands, dims), expr.rhs.value)
    return IndexFacts()


def index_facts(
    value: int | ir.Value, known: Callable[[ir.Value], IndexFacts | None]
) -> IndexFacts:
    """Facts of ``value``, from constants through ``+``, ``-``, ``*`` and ``//`` by a
    constant, ``min``, ``max`` and ``affine.apply``; ``known`` gives those of a loop's
    position (``None`` for any other value)."""
    if isinstance(value, int):
        return _constant(value)
    if (facts := known(value)) is not None:
        return facts
    if not isinstance(value, ir.OpResult):
        return IndexFacts()
    op = value.owner
    name = op.name
    args = [index_facts(operand, known) for operand in op.operands]

    def constant(position: int) -> int | None:
        fact = args[position]
        return fact.low if fact.low is not None and fact.low == fact.high else None

    if name == "arith.constant":
        attribute = op.attributes["value"]
        return (
            _constant(attribute.value)
            if isinstance(attribute, ir.IntegerAttr)
            else IndexFacts()
        )
    if name == "arith.addi":
        return _add(args[0], args[1])
    if name == "arith.subi":
        return _add(args[0], _negate(args[1]))
    if name == "arith.muli":
        for position in (0, 1):
            if (factor := constant(position)) is not None:
                return _scale(args[1 - position], factor)
        return IndexFacts(args[0].multiple * args[1].multiple)
    if name == "tensor.dim":
        return IndexFacts(low=0)
    if name in ("arith.floordivsi", "arith.divsi", "arith.divui") and (
        divisor := constant(1)
    ):
        return _floordiv(args[0], divisor)
    if name == "arith.maxsi":
        return _max(args[0], args[1])
    if name == "arith.minsi":
        return _min(args[0], args[1])
    if name == "affine.apply":
        affine_map = op.attributes["map"].value
        if len(affine_map.results) == 1:
            return _affine_facts(affine_map.results[0], args, affine_map.n_dims)
    return IndexFacts()
