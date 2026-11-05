"""The single lowering for contractions matched by ``analysis/contractions.py``.

Identity 2-D/batched maps of ATen/``hl.dot`` contractions emit ``linalg.matmul``/
``linalg.batch_matmul``; everything else (transposed operands, einsum) emits
``linalg.contract``. All are lighthouse tiling anchors. With an accumulator the op
writes into ``outs = acc`` (in-place loop-carried update, one bf16 x bf16 -> f32 op
for AMX). Without one it accumulates sub-32-bit floats in f32 and casts the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language.matmul_ops as matmul_ops
from mlir.dialects import linalg as linalg_d
import mlir.ir as ir
import torch

from ..support import UnsupportedOperationError
from ..support import ValueNotFoundError
from ..support import torch_dtype_to_mlir
from ..support.einsum_spec import ContractSpec
from ..support.einsum_spec import EinsumNotContractible
from ..support.einsum_spec import build_contract_spec
from ..trace_mode import einsum_op_target
from . import emit
from .loops import keeps_zero
from .registry import NOT_APPLICABLE
from .registry import lowers

if TYPE_CHECKING:
    from ..analysis.contractions import Contraction
    from ..build_context import BuildContext

aten = torch.ops.aten


class _NotFusable(Exception):
    """The accumulator cannot be the contraction's destination."""


@lowers(
    aten.mm.default,
    aten.bmm.default,
    aten.matmul.default,
    aten.addmm.default,
    aten.baddbmm.default,
    matmul_ops.dot,
    einsum_op_target(),
)
def lower_contraction_node(ctx: BuildContext, node: torch.fx.Node) -> object:
    plan = ctx.contractions
    if node in plan.absorbed:
        return None
    contraction = plan.roots.get(node)
    if contraction is None:
        if node.target is matmul_ops.dot or node.target is einsum_op_target():
            raise UnsupportedOperationError(
                str(node.target),
                reason="only two same-rank (2-D or 3-D) operands are supported",
            )
        return NOT_APPLICABLE
    try:
        return emit_contraction(ctx, contraction)
    except _NotFusable as error:
        raise UnsupportedOperationError(str(node.target), reason=str(error)) from None


@lowers(aten.add.Tensor)
def lower_accumulating_add(ctx: BuildContext, node: torch.fx.Node) -> object:
    """``acc + contraction`` as one contraction into ``acc`` when types allow."""
    contraction = ctx.contractions.roots.get(node)
    if contraction is None:
        return NOT_APPLICABLE
    try:
        return emit_contraction(ctx, contraction)
    except _NotFusable:
        inner = contraction.unfused
        ctx.set_value(inner.root, emit_contraction(ctx, inner))
        return NOT_APPLICABLE


def emit_contraction(ctx: BuildContext, contraction: Contraction) -> ir.Value:
    lhs = _operand(ctx, contraction.lhs)
    rhs = _operand(ctx, contraction.rhs)
    lhs_type = ir.RankedTensorType(lhs.type)
    rhs_type = ir.RankedTensorType(rhs.type)
    if lhs_type.element_type != rhs_type.element_type:
        raise UnsupportedOperationError(
            f"contraction '{contraction.equation}'",
            reason=(
                f"operand element types differ ({lhs_type.element_type} vs "
                f"{rhs_type.element_type})"
            ),
            alternatives=["cast both operands to one dtype first"],
        )
    try:
        spec = build_contract_spec(
            contraction.equation, [list(lhs_type.shape), list(rhs_type.shape)]
        )
    except EinsumNotContractible as error:
        raise UnsupportedOperationError(
            f"contraction '{contraction.equation}'", reason=str(error)
        ) from error

    _refuse_dot_of_padding(ctx, contraction, spec)
    operand_type = lhs_type.element_type
    result_type = _result_element_type(contraction.root, operand_type)
    if contraction.acc is None:
        accumulate_type = _accumulation_type(result_type)
        init = emit.filled(_out_sizes(spec, lhs, rhs), accumulate_type, 0)
        result = _emit_op(contraction, spec, lhs, rhs, init)
        return emit.cast_tensor(result, result_type)

    acc = _operand(ctx, contraction.acc)
    _check_accumulator(contraction, spec, operand_type, result_type, acc)
    return _emit_op(contraction, spec, lhs, rhs, acc)


def _refuse_dot_of_padding(
    ctx: BuildContext, contraction: Contraction, spec: ContractSpec
) -> None:
    """Helion masks ``torch.addmm``'s operands but not ``hl.dot``'s: over a partial
    tile's K, a computed operand's padding joins the sum. Refused unless both operands
    provably keep it at 0, as loads do."""
    dots = (contraction.root, *contraction.absorbed)
    if not any(node.target is matmul_ops.dot for node in dots):
        return
    shape = contraction.lhs.meta["val"].shape
    contracted = set(spec.lhs) & set(spec.rhs) - set(spec.out)
    partial = any(
        ctx.block_id_to_valid.get(ctx.env.resolve_block_id(shape[spec.lhs.index(dim)]))
        is not None
        for dim in contracted
    )
    if partial and not (
        keeps_zero(ctx, contraction.lhs) and keeps_zero(ctx, contraction.rhs)
    ):
        raise UnsupportedOperationError(
            "hl.dot of a computed operand over a partial tile",
            reason="Helion doesn't mask its operands, so padding joins the sum",
            alternatives=[
                "torch.addmm, whose operands Helion masks",
                "block sizes that divide the sizes they tile, with static_shapes=True",
            ],
        )


def _operand(ctx: BuildContext, node: torch.fx.Node) -> ir.Value:
    value = ctx.get_value(node)
    if value is None:
        raise ValueNotFoundError(node, context="contraction operand")
    return value


def _out_sizes(spec: ContractSpec, lhs: ir.Value, rhs: ir.Value) -> list[emit.Size]:
    """The result's dims; a runtime one is the size of the operand dim it comes from."""
    lhs_sizes, rhs_sizes = emit.sizes(lhs), emit.sizes(rhs)
    return [
        lhs_sizes[spec.lhs.index(position)]
        if position in spec.lhs
        else rhs_sizes[spec.rhs.index(position)]
        for position in spec.out
    ]


def _result_element_type(root: torch.fx.Node, operand_type: ir.Type) -> ir.Type:
    value = root.meta.get("val")
    if isinstance(value, torch.Tensor):
        return torch_dtype_to_mlir(value.dtype)
    return operand_type


def _accumulation_type(result_type: ir.Type) -> ir.Type:
    if isinstance(result_type, ir.FloatType) and result_type.width < 32:
        return ir.F32Type.get()
    return result_type


def _check_accumulator(
    contraction: Contraction,
    spec: ContractSpec,
    operand_type: ir.Type,
    result_type: ir.Type,
    acc: ir.Value,
) -> None:
    acc_type = ir.RankedTensorType(acc.type)
    if list(acc_type.shape) != spec.out_shape:
        raise _NotFusable(
            f"accumulator shape {list(acc_type.shape)} differs from the "
            f"contraction result shape {spec.out_shape}"
        )
    acc_element = acc_type.element_type
    if acc_element != result_type:
        raise _NotFusable(
            f"result type {result_type} differs from the accumulator type {acc_element}"
        )
    if not _is_valid_accumulator(operand_type, acc_element):
        raise _NotFusable(
            f"cannot accumulate {operand_type} operands into {acc_element}"
        )
    root = contraction.root
    if root.target is matmul_ops.dot:
        out_dtype = root.args[3] if len(root.args) > 3 else root.kwargs.get("out_dtype")
        if out_dtype is not None and torch_dtype_to_mlir(out_dtype) != acc_element:
            raise _NotFusable(
                f"hl.dot out_dtype={out_dtype} differs from the accumulator type"
            )


def _is_valid_accumulator(operand_type: ir.Type, accumulator_type: ir.Type) -> bool:
    if operand_type == accumulator_type:
        return True
    for kind in (ir.FloatType, ir.IntegerType):
        if isinstance(operand_type, kind) and isinstance(accumulator_type, kind):
            return accumulator_type.width >= operand_type.width
    return False


def _emit_op(
    contraction: Contraction,
    spec: ContractSpec,
    lhs: ir.Value,
    rhs: ir.Value,
    out: ir.Value,
) -> ir.Value:
    if contraction.named and _has_maps(spec, (0, 2), (2, 1), (0, 1)):
        return linalg_d.matmul(lhs, rhs, outs=[out])
    if contraction.named and _has_maps(spec, (0, 1, 3), (0, 3, 2), (0, 1, 2)):
        return linalg_d.batch_matmul(lhs, rhs, outs=[out])
    return linalg_d.contract(lhs, rhs, outs=[out], indexing_maps=indexing_maps(spec))


def indexing_maps(spec: ContractSpec) -> list[ir.AffineMap]:
    total = len(spec.iteration_dims)
    return [
        ir.AffineMap.get(total, 0, [ir.AffineDimExpr.get(p) for p in positions])
        for positions in (spec.lhs, spec.rhs, spec.out)
    ]


def _has_maps(spec: ContractSpec, lhs: tuple, rhs: tuple, out: tuple) -> bool:
    return (spec.lhs, spec.rhs, spec.out) == (lhs, rhs, out)
