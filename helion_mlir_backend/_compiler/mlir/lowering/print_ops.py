"""``hl.device_print``, which is ``print`` in a kernel: ``gpu.printf`` for every target.

Each target lowers the op its own way (:mod:`...mlir_transforms.prints`). What Helion's
type rule accepts is what prints: a label and tensors. A tensor prints an element per
line; a partial tile's padding prints too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language as hl
from mlir.dialects import arith as arith_d
from mlir.dialects import scf as scf_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir

from ..support import UnsupportedOperationError
from . import emit
from .registry import lowers

if TYPE_CHECKING:
    import torch

    from ..build_context import BuildContext


def _printable(value: ir.Value) -> tuple[str, ir.Value]:
    """The conversion and the operand that print the scalar ``value``."""
    i64 = ir.IntegerType.get_signless(64)
    type_ = value.type
    if isinstance(type_, ir.FloatType):
        if type_.width < 32:
            value = arith_d.ExtFOp(ir.F32Type.get(), value).result
        return "%f", value
    if isinstance(type_, ir.IndexType):
        return "%ld", arith_d.IndexCastOp(i64, value).result
    if isinstance(type_, ir.IntegerType):
        if type_.width == 1:
            value = arith_d.ExtUIOp(i64, value).result
        elif type_.width < 64:
            value = arith_d.ExtSIOp(i64, value).result
        return "%ld", value
    raise UnsupportedOperationError(
        "hl.device_print", reason=f"cannot print a value of type {type_}"
    )


def _printf(text: str, operands: list[ir.Value]) -> None:
    ir.Operation.create(
        "gpu.printf",
        operands=operands,
        attributes={"format": ir.StringAttr.get(text)},
    )


def _print_tensor(label: str, value: ir.Value) -> None:
    """``label[i, j]: v`` per element, in nested loops: the read is of one element, so
    tiling and vectorization see the tile's other uses as they are."""
    index = ir.IndexType.get()
    i64 = ir.IntegerType.get_signless(64)
    sizes = emit.sizes(value)
    zero = emit.constant(index, 0)
    one = emit.constant(index, 1)

    def nest(dim: int, indices: list[ir.Value]) -> None:
        if dim == len(sizes):
            conversion, operand = _printable(tensor_d.ExtractOp(value, indices).result)
            positions = [arith_d.IndexCastOp(i64, i).result for i in indices]
            where = ", ".join("%ld" for _ in indices)
            _printf(f"{label}[{where}]: {conversion}\n", [*positions, operand])
            return
        size = sizes[dim]
        bound = emit.constant(index, size) if isinstance(size, int) else size
        loop = scf_d.ForOp(zero, bound, one)
        with ir.InsertionPoint(loop.body):
            nest(dim + 1, [*indices, loop.induction_variable])
            scf_d.YieldOp([])

    nest(0, [])


def _leaves(args: object) -> list[object]:
    """The frontend wraps the varargs: ``('label', ((a, b),))``."""
    if isinstance(args, (tuple, list)):
        return [leaf for arg in args for leaf in _leaves(arg)]
    return [args]


@lowers(hl.device_print)
def lower_device_print(ctx: BuildContext, node: torch.fx.Node) -> None:
    """Scalars on one line after the label, then each tensor; a label alone is a line."""
    label = node.args[0].replace("%", "%%")
    values = [ctx.get_value(arg) for arg in _leaves(node.args[1:])]
    tensors = [v for v in values if isinstance(v.type, ir.RankedTensorType)]
    scalars = [v for v in values if not isinstance(v.type, ir.RankedTensorType)]
    if scalars or not tensors:
        parts = [_printable(value) for value in scalars]
        line = " ".join([label, *(conversion for conversion, _ in parts)])
        _printf(line + "\n", [operand for _, operand in parts])
    for tensor in tensors:
        if ir.RankedTensorType(tensor.type).rank == 0:
            conversion, operand = _printable(tensor_d.ExtractOp(tensor, []).result)
            _printf(f"{label} {conversion}\n", [operand])
        else:
            _print_tensor(label, tensor)
