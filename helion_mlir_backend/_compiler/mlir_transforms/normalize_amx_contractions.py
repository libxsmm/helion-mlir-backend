"""``normalize_amx_contractions``: contractions lowered to AMX in the layouts
upstream's AMX rewrite assumes, or kept off AMX.

Lighthouse picks AMX register tiles for every bf16 x bf16 -> f32 contraction
on an AMX target (``is_amx_bf16_contraction``), and upstream's
``x86.vector_contract_to_amx_dot_product`` then checks the shapes of a flat
(non-VNNI) ``vector.contract``, never its indexing maps: it loads the LHS as
``M x K`` rows, packs the RHS to VNNI from ``K x N`` rows and reads and writes
the accumulator as ``M x N`` rows, from the two innermost dims of each. Any
other layout that passes the shape checks (``16x32`` / ``16x16`` either way)
is miscompiled (``x @ w.T``, attention's ``q @ k.T``, a transposed
accumulator, split-K operands) or aborts the rewrite (decode-style batched
dots: an LHS without an M dim).

The dims are classified as M (LHS and accumulator), N (RHS and accumulator),
K (both inputs, not the accumulator; a batch-reduce dim is one) and batch (all
three); there may be several of each kind.

- **VNNI**, exactly as upstream's ``isInVnniLayout`` (``X86Utils.cpp``)
  decides it, for bf16 (factor 2): both inputs of rank 3 or more, the same K
  dim innermost in both, the LHS's second innermost dim the RHS's third
  innermost one and a K dim, the RHS's second innermost dim parallel, and a
  static innermost size of 2 in both. The rewrite reads these maps itself; an
  ``(N, M)`` accumulator is swapped as below, any other kept (e.g. KernelBench
  level 2's 37, ``mcv,bcnv->mbn``, whose accumulator ends in ``(N, N)``).
- **Flat**, deciding per operand from its two innermost dims (all the rewrite
  reads): LHS ``(M, K)``, RHS ``(K, N)`` and accumulator ``(M, N)`` are kept;
  LHS ``(K, M)``, RHS ``(N, K)`` are transposed (the two innermost dims
  swapped) into a new tensor, a ``linalg.transpose`` only read by the
  contraction (an operand pack, kept in memory by ``pin_transposes`` and
  ``isolate_operand_packs``); an ``(N, M)`` accumulator's init is transposed
  in, the contraction computes the ``(M, N)`` result, which is transposed back
  for its users. As the accumulator is loop-carried across K steps, that adds
  two transposes per K step of the source loop.
- **Kept off AMX**: any other layout (an innermost pair of other kinds, e.g. a
  batch dim innermost or no M dim) is marked with ``AMX_OFF_ATTR_NAME``, and
  ``pin_amx_off_tile_sizes`` (before each register tiling stage) tiles it as on
  a host without AMX, where the same IR is correct; the rewrite gets no
  AMX-shaped contraction of it. Memref contractions not in the kept layouts
  (lighthouse does not tile memref ops, but the rewrite takes them once
  vectorized) get both inputs widened to f32 into new buffers instead, the
  body truncating them back (exact: the values are bf16): an f32 contraction.

Left alone: other targets and element types (f32, f16, i8), and contractions
already in a kept layout."""

from lighthouse.dialects.transform.transform_ext.utils.tiling.target_caps import (
    is_amx_bf16_contraction,
)
from lighthouse.execution.target import TargetInfo
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import memref
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .pin_amx_off_tile_sizes import AMX_OFF_ATTR_NAME
from .utils import payload_ops

_KEEP = "keep"
_SWAP = "swap"
# The VNNI factor of bf16.
_VNNI_FACTOR = 2


def _dims(affine_map: ir.AffineMap) -> list[int] | None:
    """The dims of a projected permutation's results, else ``None``."""
    if affine_map.n_symbols or not affine_map.is_projected_permutation:
        return None
    return [ir.AffineDimExpr(expr).position for expr in affine_map.results]


def _kinds(dims: list[list[int]], n_dims: int) -> dict[int, str | None]:
    lhs, rhs, acc = (set(d) for d in dims)
    return {
        dim: {
            (True, True, True): "B",
            (True, False, True): "M",
            (False, True, True): "N",
            (True, True, False): "K",
        }.get((dim in lhs, dim in rhs, dim in acc))
        for dim in range(n_dims)
    }


def _is_vnni(op: ir.OpView, dims: list[list[int]], kinds: dict) -> bool:
    """Upstream's ``isInVnniLayout`` for bf16 (see the module docstring)."""
    lhs, rhs, _ = dims
    if len(lhs) < 3 or len(rhs) < 3:
        return False
    if sum(kind == "K" for kind in kinds.values()) < 2:
        return False
    if lhs[-1] != rhs[-1] or kinds[lhs[-1]] != "K":
        return False
    if lhs[-2] != rhs[-3] or kinds[lhs[-2]] != "K":
        return False
    if kinds[rhs[-2]] not in ("M", "N", "B"):
        return False
    shapes = [ir.ShapedType(op.operands[i].type).shape for i in (0, 1)]
    return shapes[0][-1] == shapes[1][-1] == _VNNI_FACTOR


def _actions(op: ir.OpView) -> list[str] | None:
    """What to do with each operand (LHS, RHS, accumulator), ``_KEEP`` or
    ``_SWAP`` (its two innermost dims); ``None``: keep the contraction off AMX."""
    maps = [ir.AffineMapAttr(m).value for m in linalg.get_indexing_maps(op)]
    if len(maps) != 3:
        return None
    dims = [_dims(m) for m in maps]
    if any(d is None or len(d) < 2 for d in dims):
        return None
    kinds = _kinds(dims, maps[0].n_dims)
    pairs = [tuple(kinds[d] for d in operand[-2:]) for operand in dims]
    if _is_vnni(op, dims, kinds):
        return [_KEEP, _KEEP, _SWAP if pairs[2] == ("N", "M") else _KEEP]
    expected = (("M", "K"), ("K", "N"), ("M", "N"))
    actions = []
    for pair, want in zip(pairs, expected, strict=True):
        if pair == want:
            actions.append(_KEEP)
        elif pair == tuple(reversed(want)):
            actions.append(_SWAP)
        else:
            return None
    return actions


def _sizes(source: ir.Value, order: list[int]) -> list[ir.Value | int]:
    """The sizes of ``source``'s dims in ``order`` (``tensor.dim`` /
    ``memref.dim`` for runtime ones)."""
    source_type = ir.ShapedType(source.type)
    index = ir.IndexType.get()
    dim_op = tensor.dim if isinstance(source.type, ir.RankedTensorType) else memref.dim
    return [
        dim_op(source, arith.constant(index, dim))
        if source_type.is_dynamic_dim(dim)
        else source_type.shape[dim]
        for dim in order
    ]


def _transposed(source: ir.Value) -> ir.Value:
    """``source`` with its two innermost dims swapped, in a new tensor."""
    source_type = ir.RankedTensorType(source.type)
    rank = source_type.rank
    permutation = [*range(rank - 2), rank - 1, rank - 2]
    init = tensor.EmptyOp(_sizes(source, permutation), source_type.element_type)
    transposed = linalg.transpose(source, outs=[init.result], permutation=permutation)
    return transposed.results[0]


def _widened(source: ir.Value) -> ir.Value:
    """The memref ``source`` extended to f32, in a new buffer."""
    source_type = ir.MemRefType(source.type)
    rank = source_type.rank
    f32 = ir.F32Type.get()
    sizes = _sizes(source, list(range(rank)))
    dynamic = [s for s in sizes if isinstance(s, ir.Value)]
    shape = [
        ir.ShapedType.get_dynamic_size() if isinstance(s, ir.Value) else s
        for s in sizes
    ]
    buffer = memref.alloc(ir.MemRefType.get(shape, f32), dynamic, [])
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank))
    widen = linalg.GenericOp(
        [],
        [source],
        [buffer],
        ir.ArrayAttr.get([identity, identity]),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank
        ),
    )
    body = widen.regions[0].blocks.append(source_type.element_type, f32)
    with ir.InsertionPoint(body):
        linalg.YieldOp([arith.extf(f32, body.arguments[0])])
    return buffer


def _widen_inputs(op: ir.OpView) -> None:
    """Contract memref ``op``'s inputs widened to f32: no bf16 contraction
    for AMX."""
    with ir.InsertionPoint(op), op.location:
        for number in (0, 1):
            op.operation.operands[number] = _widened(op.operands[number])
    block = op.regions[0].blocks[0]
    f32 = ir.F32Type.get()
    for number in (0, 1):
        argument = block.arguments[number]
        element_type = argument.type
        uses = [(use.owner, use.operand_number) for use in argument.uses]
        argument.set_type(f32)
        with ir.InsertionPoint.at_block_begin(block), op.location:
            narrowed = arith.truncf(element_type, argument)
        for owner, operand_number in uses:
            owner.operands[operand_number] = narrowed


def _swap_inner_dims(affine_map: ir.AffineMap) -> ir.AffineMap:
    results = list(affine_map.results)
    results[-2], results[-1] = results[-1], results[-2]
    return ir.AffineMap.get(affine_map.n_dims, affine_map.n_symbols, results)


def _maps_attr(maps: list[ir.AffineMap]) -> ir.ArrayAttr:
    return ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in maps])


def _rebuild(
    op: ir.OpView, operands: list[ir.Value], maps: list[ir.AffineMap]
) -> ir.Operation:
    """``op`` with ``operands`` (its result of the accumulator's type) and
    ``maps``, its body moved over, created at the current insertion point."""
    old = op.operation.attributes
    attributes = {name: old[name] for name in old}
    attributes["indexing_maps"] = _maps_attr(maps)
    new = ir.Operation.create(
        op.operation.name,
        results=[operands[2].type],
        operands=operands,
        attributes=attributes,
        regions=len(op.operation.regions),
    )
    for old_region, new_region in zip(op.operation.regions, new.regions, strict=True):
        for old_block in old_region.blocks:
            types = [arg.type for arg in old_block.arguments]
            new_block = new_region.blocks.append(*types)
            for old_arg, new_arg in zip(
                old_block.arguments, new_block.arguments, strict=True
            ):
                old_arg.replace_all_uses_with(new_arg)
            for body_op in list(old_block.operations):
                body_op.detach_from_parent()
                new_block.append(body_op)
    return new


def _normalize(op: ir.OpView, actions: list[str]) -> None:
    maps = [ir.AffineMapAttr(m).value for m in linalg.get_indexing_maps(op)]
    operands = list(op.operation.operands)
    with ir.InsertionPoint(op), op.location:
        for number, action in enumerate(actions):
            if action == _SWAP:
                operands[number] = _transposed(operands[number])
                maps[number] = _swap_inner_dims(maps[number])
        if actions[2] != _SWAP:
            for number in (0, 1):
                op.operation.operands[number] = operands[number]
            op.operation.attributes["indexing_maps"] = _maps_attr(maps)
            return
        new = _rebuild(op, operands, maps)
        result = _transposed(new.results[0])
    op.results[0].replace_all_uses_with(result)
    # Erased outside the transform rewriter: the schedule's only handle is to
    # the functions, which stay valid, and the walk collected the contractions
    # beforehand, each visited once (the rebuilt one is not in the list).
    op.operation.erase()


def _is_candidate(op: ir.OpView, target: TargetInfo) -> bool:
    """A bf16 x bf16 -> f32 contraction of matrices on an AMX target: what
    lighthouse tiles for AMX (``is_amx_bf16_contraction``), also on memrefs
    (which the predicate skips, having no results, but which the AMX rewrite
    still takes once vectorized)."""
    if not linalg.isa_contraction_op(op) or len(op.operands) != 3:
        return False
    if not all(isinstance(v.type, ir.ShapedType) for v in op.operands):
        return False
    if _is_tensor_op(op):
        return is_amx_bf16_contraction(op, target)
    if not target.is_supported("amx"):
        return False
    types = [ir.ShapedType(v.type) for v in op.operands]
    return (
        all(t.rank >= 2 for t in types)
        and isinstance(types[0].element_type, ir.BF16Type)
        and isinstance(types[1].element_type, ir.BF16Type)
        and isinstance(types[2].element_type, ir.F32Type)
    )


def _is_tensor_op(op: ir.OpView) -> bool:
    return all(isinstance(v.type, ir.RankedTensorType) for v in op.operands)


@transform_op(modifies_payload=True)
class NormalizeAmxContractionsOp(
    HelionTransformDialect.Operation, name="normalize_amx_contractions"
):
    """Normalize every AMX contraction of the target to the layouts upstream's
    AMX rewrite reads, or keep it off AMX (see the module docstring)."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "NormalizeAmxContractionsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        target = TargetInfo.host()
        for contraction in payload_ops(
            state, op.target, (linalg.ContractOp, linalg.GenericOp)
        ):
            if not _is_candidate(contraction, target):
                continue
            actions = _actions(contraction)
            if actions is not None and _SWAP not in actions:
                continue
            if _is_tensor_op(contraction):
                if actions is None:
                    contraction.operation.attributes[AMX_OFF_ATTR_NAME] = (
                        ir.UnitAttr.get()
                    )
                else:
                    _normalize(contraction, actions)
            else:
                _widen_inputs(contraction)
        return DiagnosedSilenceableFailure.Success


def normalize_amx_contractions(target: ir.Value) -> NormalizeAmxContractionsOp:
    """snake_case wrapper to create a NormalizeAmxContractionsOp."""
    return NormalizeAmxContractionsOp(target=target)
