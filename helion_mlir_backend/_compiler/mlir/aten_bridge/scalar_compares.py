"""Compares of a tile with a Python number, in the tile's type, as torch does them.

Torch converts the number to the tile's type and compares there: with ``x`` holding 0.1 of
its type, ``x <= 0.1`` is true. torch-mlir (wheel 20261001) instead widens a float tile to
f64 and compares with the f64 number, where 0.1 of f32 is above 0.1; for an integer tile it
truncates a float number, where ``x > -0.5`` is ``x > 0``, and compares an integer number in
i64, where torch wraps it to the tile's width (``i8 < 200`` is ``i8 < -56``). Those are wrong
answers, on the elements at or next to the number.

:func:`compare_in_tile_type` rewrites one of those ATen helpers, as ``nan_minmax`` does:

- a float tile: ``cmpf P, extf(in : T to f64), n : f64`` becomes a compare of the tile with
  ``n`` converted to ``T``, through f32 like torch's own conversion. bf16 and f16 compare
  extended to f32, which holds them exactly, as neither target compares them directly. A
  number the IR only has at run time is converted in the body, and hoisted out later.
- an integer or bool tile and a float number: torch promotes the tile to float (its default
  type, f32): a ``cmpf`` of ``sitofp`` (``uitofp`` for bool) of the tile with ``n`` in f32.
- an integer tile and an integer number: the number wrapped to the tile's width.
- a bool tile and an integer number: torch promotes the tile to i64.

The predicate of an integer tile's compare comes from the op, not the IR: what torch-mlir
leaves can have folded with the truncated number. Only with the number known here. Anything
else is left alone: a tile of f64, a tile against a tile, a number known at run time on an
integer tile, whose helper already compares in f64.
"""

from __future__ import annotations

from mlir import ir
import torch

_aten = torch.ops.aten
COMPARE_OPS = frozenset(
    {
        _aten.eq.Scalar,
        _aten.ne.Scalar,
        _aten.lt.Scalar,
        _aten.le.Scalar,
        _aten.gt.Scalar,
        _aten.ge.Scalar,
    }
)
_PREDICATES = {
    # The op's name: arith.cmpi's signed predicate, arith.cmpf's: ordered, "ne" unordered.
    "eq": (0, 1),
    "ne": (1, 13),
    "lt": (2, 4),
    "le": (3, 5),
    "gt": (4, 2),
    "ge": (5, 3),
}


def _torch_type(float_type: ir.Type) -> torch.dtype:
    if isinstance(float_type, ir.F32Type):
        return torch.float32
    if isinstance(float_type, ir.F16Type):
        return torch.float16
    if isinstance(float_type, ir.BF16Type):
        return torch.bfloat16
    raise ValueError(float_type)


def _walk(function: ir.Operation) -> list[ir.Operation]:
    ops: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        ops.append(op)
        return ir.WalkResult.ADVANCE

    function.walk(visit, ir.WalkOrder.PRE_ORDER)
    return ops


def _defining(value: ir.Value) -> ir.Operation | None:
    owner = value.owner
    return owner.operation if isinstance(owner, ir.OpView) else None


def _create(
    name: str, result: ir.Type, *operands: ir.Value, **attributes: object
) -> ir.Value:
    return ir.Operation.create(
        name, results=[result], operands=list(operands), attributes=attributes
    ).results[0]


def _f32_constant(value: float) -> ir.Value:
    f32 = ir.F32Type.get()
    return _create("arith.constant", f32, value=ir.FloatAttr.get(f32, value))


def _in_tile_type(number: ir.Value, tile: ir.Type) -> ir.Value:
    """``number`` (f64) in ``tile``'s type, as f32, at the insertion point."""
    constant = _defining(number)
    if constant is not None and constant.name == "arith.constant":
        value = ir.FloatAttr(constant.attributes["value"]).value
        return _f32_constant(torch.tensor(value, dtype=_torch_type(tile)).item())
    f32 = ir.F32Type.get()
    narrowed = _create("arith.truncf", f32, number)
    if isinstance(tile, ir.F32Type):
        return narrowed
    return _create("arith.extf", f32, _create("arith.truncf", tile, narrowed))


def _float_tile(compare: ir.Operation) -> None:
    widened = _defining(compare.operands[0])
    number = compare.operands[1]
    if (
        widened is None
        or widened.name != "arith.extf"
        or not isinstance(number.type, ir.F64Type)
    ):
        return
    source = widened.operands[0]
    tile = source.type
    if not isinstance(tile, ir.F32Type | ir.F16Type | ir.BF16Type):
        return
    f32 = ir.F32Type.get()
    with ir.InsertionPoint(compare):
        scalar = _in_tile_type(number, tile)
        lhs = (
            source
            if isinstance(tile, ir.F32Type)
            else _create("arith.extf", f32, source)
        )
    compare.operands[0] = lhs
    compare.operands[1] = scalar
    if not any(True for _ in widened.results[0].uses):
        widened.erase()


def _i64(value: int) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)


def _wrapped(number: int, width: int) -> int:
    number &= (1 << width) - 1
    return number - (1 << width) if number >> (width - 1) else number


def _integer_compare(
    source: ir.Value, number: object, name: str, result: ir.Type
) -> ir.Value:
    """``source``, an element of an integer or bool tile, against ``number``, at the insertion
    point."""
    element = source.type
    bool_tile = element.width == 1
    signed, ordered = _PREDICATES[name]
    if isinstance(number, float):
        f32 = ir.F32Type.get()
        lhs = _create("arith.uitofp" if bool_tile else "arith.sitofp", f32, source)
        rhs = _f32_constant(torch.tensor(number, dtype=torch.float32).item())
        opcode, predicate = "arith.cmpf", ordered
    else:
        i64 = ir.IntegerType.get_signless(64)
        if bool_tile:
            lhs, width_type, value = _create("arith.extui", i64, source), i64, number
        else:
            lhs, width_type = source, element
            value = _wrapped(number, element.width)
        rhs = _create(
            "arith.constant", width_type, value=ir.IntegerAttr.get(width_type, value)
        )
        opcode, predicate = "arith.cmpi", signed
    return ir.Operation.create(
        opcode,
        results=[result],
        operands=[lhs, rhs],
        attributes={"predicate": _i64(predicate)},
    ).results[0]


def _integer_tile(compare: ir.Operation, number: object, name: str) -> None:
    source = compare.operands[0]
    extended = _defining(source)
    if extended is not None and extended.name in ("arith.extsi", "arith.extui"):
        source = extended.operands[0]
    with ir.InsertionPoint(compare):
        new = _integer_compare(source, number, name, compare.results[0].type)
    compare.results[0].replace_all_uses_with(new)
    compare.erase()


def _folded_compare(function: ir.Operation, number: object, name: str) -> None:
    """torch-mlir folded a float number that no integer equals to ``false`` or ``true``,
    but torch compares in f32, where 1e-50 is 0."""
    for op in _walk(function):
        if op.name != "linalg.yield":
            continue
        result = op.operands[0]
        owner = _defining(result)
        if owner is None or owner.name != "arith.constant":
            continue
        source = op.parent.regions[0].blocks[0].arguments[0]
        if not isinstance(source.type, ir.IntegerType):
            continue
        with ir.InsertionPoint(op):
            op.operands[0] = _integer_compare(source, number, name, result.type)


def compare_in_tile_type(function: ir.Operation, number: object, name: str) -> None:
    """Rewrite the helper ``function`` of the op ``name`` (``lt``, ...) of :data:`COMPARE_OPS`
    (see the module). ``number``: its Python number, if the call has one."""
    if isinstance(number, bool):
        number = int(number)
    compares = [op for op in _walk(function) if op.name in ("arith.cmpf", "arith.cmpi")]
    if not compares and isinstance(number, float):
        _folded_compare(function, number, name)
    for op in compares:
        if op.name == "arith.cmpf":
            _float_tile(op)
        elif isinstance(number, float | int):
            _integer_tile(op, number, name)
