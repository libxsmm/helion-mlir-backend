"""``split_contraction_operands``: elementwise fusion can move an operand's
producer (an ``exp``, the ``truncf`` of a ``.to(bf16)``, ...) into the body of a
``linalg.generic`` contraction. Upstream's raising of generics to
``linalg.contract`` (``linalg-morph-ops{generic-to-category}``, also
``linalg-categorize-ops``) matches the body's operands modulo any
side-effect-free unary op, and the ``linalg.contract`` it builds only casts its
operands: the ``exp`` is dropped, and so is a ``truncf``+``extf`` rounding,
which leaves a contraction of mixed operand types. Split out first, each such
operand's computation is an elementwise ``linalg.generic`` of its own, which
the contraction reads.

Matched: a ``linalg.generic`` on tensors with two inputs and one init, all maps
projected permutations, whose body is exactly

    yield(add(acc, mul(chain_a(a), chain_b(b))))

(``arith.mulf``+``arith.addf`` or ``arith.muli``+``arith.addi``, either operand
order), each chain a run of single-use, single-operand, region-free ``arith``
and ``math`` ops from its block argument (possibly none). A chain the raised
``linalg.contract`` reproduces exactly is no op, or one cast it applies itself
(``arith.extf``, ``extsi``, ``extui``, ``sitofp``, ``uitofp``; of two kept
casts of different signedness, upstream raises nothing). Any other chain
moves to an elementwise generic over the operand's shape, of the type it
computes; a trailing exact cast stays in the contraction (so a ``.to(bf16)``
leaves a bf16 tensor, and a bf16 x bf16 contraction). Unless the two inputs
would then differ in element type (e.g. a widened bf16 times an f32): the
vectorizer casts no float inputs of a ``linalg.contract`` and builds an invalid
``vector.contract``, so all casts move too and both inputs are of the
multiply's type.

Left alone: generics whose chains are all exact (and inputs of one type), more or fewer operands,
memref operands, other maps, other combiner pairs (``andi``/``ori``,
``complex``), and bodies with anything else (unary ops on the product, the
accumulator or the result, chain ops of other dialects or with other uses,
extra ops). Elementwise fusion only fuses producers of inputs, so it makes no
ops on the product, accumulator or result side."""

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import payload_ops

_PAIRS = {"arith.mulf": "arith.addf", "arith.muli": "arith.addi"}
# Operand casts a linalg.contract applies itself (by its `cast` attribute).
_EXACT_CASTS = frozenset(
    ("arith.extf", "arith.extsi", "arith.extui", "arith.sitofp", "arith.uitofp")
)
_CHAIN_DIALECTS = ("arith.", "math.")


def _defining_op(value: ir.Value) -> ir.Operation | None:
    owner = value.owner
    if isinstance(owner, ir.OpView):
        return owner.operation
    return owner if isinstance(owner, ir.Operation) else None


def _single_use(value: ir.Value) -> bool:
    return len(list(value.uses)) == 1


def _chain(value: ir.Value, block: ir.Block) -> tuple[list[ir.Operation], int] | None:
    """The unary ops from an input block argument of ``block`` to ``value``,
    nearest to ``value`` first, and that argument's number."""
    chain: list[ir.Operation] = []
    while not isinstance(value, ir.BlockArgument):
        op = _defining_op(value)
        if (
            op is None
            or not op.name.startswith(_CHAIN_DIALECTS)
            or len(op.operands) != 1
            or len(op.results) != 1
            or len(op.regions) != 0
            or not _single_use(value)
        ):
            return None
        chain.append(op)
        value = op.operands[0]
    if value.owner != block or value.arg_number > 1 or not _single_use(value):
        return None
    return chain, value.arg_number


def _matches(generic: linalg.GenericOp) -> dict[int, list[ir.Operation]] | None:
    """The chain of each input of a contraction the raising would match."""
    if len(generic.inputs) != 2 or len(generic.outputs) != 1:
        return None
    if len(generic.results) != 1 or not all(
        isinstance(value.type, ir.RankedTensorType)
        for value in [*generic.inputs, *generic.outputs]
    ):
        return None
    maps = [ir.AffineMapAttr(attr).value for attr in generic.indexing_maps]
    if not all(map_.is_projected_permutation for map_ in maps):
        return None
    block = generic.regions[0].blocks[0]
    ops = list(block.operations)
    yielded = ops[-1].operands
    if len(yielded) != 1:
        return None
    add = _defining_op(yielded[0])
    if add is None or add.name not in _PAIRS.values() or not _single_use(yielded[0]):
        return None
    acc = block.arguments[2]
    if not _single_use(acc) or acc not in list(add.operands):
        return None
    product = add.operands[1] if add.operands[0] == acc else add.operands[0]
    mul = _defining_op(product)
    if mul is None or _PAIRS.get(mul.name) != add.name or not _single_use(product):
        return None
    chains = {}
    for operand in mul.operands:
        found = _chain(operand, block)
        if found is None:
            return None
        chain, number = found
        chains[number] = chain
    if len(chains) != 2 or len(ops) != sum(map(len, chains.values())) + 3:
        return None
    return chains


def _kept(chain: list[ir.Operation]) -> list[ir.Operation]:
    """The trailing exact cast of ``chain``, if any."""
    return chain[:1] if chain and chain[0].name in _EXACT_CASTS else []


def _read_type(
    generic: linalg.GenericOp, number: int, chain: list[ir.Operation]
) -> ir.Type:
    """The element type input ``number`` has once its chain is split."""
    kept = _kept(chain)
    if kept:
        return kept[0].operands[0].type
    if chain:
        return chain[0].results[0].type
    return generic.regions[0].blocks[0].arguments[number].type


def _split(
    generic: linalg.GenericOp, number: int, chain: list[ir.Operation], keep: bool
) -> None:
    """Computes input ``number``'s ``chain`` (but, with ``keep``, a trailing
    exact cast) in an elementwise generic before ``generic``, which reads its
    result."""
    moved = chain[len(_kept(chain)) :] if keep else chain
    if not moved:
        return
    source = generic.inputs[number]
    source_type = ir.RankedTensorType(source.type)
    element_type = moved[0].results[0].type
    rank = source_type.rank
    identity = ir.AffineMap.get_identity(rank)
    with ir.InsertionPoint(generic), generic.location:
        index = ir.IndexType.get()
        sizes = [
            tensor.dim(source, arith.constant(index, dim))
            if source_type.is_dynamic_dim(dim)
            else source_type.shape[dim]
            for dim in range(rank)
        ]
        init = tensor.EmptyOp(sizes, element_type)
        computed = linalg.GenericOp(
            [init.result.type],
            [source],
            [init.result],
            ir.ArrayAttr.get([ir.AffineMapAttr.get(identity)] * 2),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank
            ),
        )
        body = computed.regions[0].blocks.append(source_type.element_type, element_type)
    with ir.InsertionPoint(body):
        value = body.arguments[0]
        for op in reversed(moved):
            clone = op.clone()
            clone.operands[0] = value
            value = clone.results[0]
        linalg.YieldOp([value])
    generic.operation.operands[number] = computed.result
    argument = generic.regions[0].blocks[0].arguments[number]
    argument.set_type(element_type)
    # The kept cast (or the multiply) reads the argument; the rest is dead.
    moved[0].results[0].replace_all_uses_with(argument)
    for op in moved:
        op.erase()


def _split_operands(generic: linalg.GenericOp) -> None:
    chains = _matches(generic)
    if chains is None:
        return
    exact = all(len(chain) == len(_kept(chain)) for chain in chains.values())
    # The raised contraction casts each input to the result type, but the
    # vectorizer does not cast float inputs: of two types, the casts go too.
    keep = len({_read_type(generic, n, chain) for n, chain in chains.items()}) == 1
    if exact and keep:
        return
    for number, chain in sorted(chains.items()):
        _split(generic, number, chain, keep)


@transform_op(modifies_payload=True)
class SplitContractionOperandsOp(
    HelionTransformDialect.Operation, name="split_contraction_operands"
):
    """Split the operand computations out of every ``linalg.generic``
    contraction in the target that the raising to ``linalg.contract`` would
    drop (see the module docstring): each into an elementwise generic."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "SplitContractionOperandsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for generic in payload_ops(state, op.target, linalg.GenericOp):
            _split_operands(generic)
        return DiagnosedSilenceableFailure.Success


def split_contraction_operands(target: ir.Value) -> SplitContractionOperandsOp:
    """snake_case wrapper to create a SplitContractionOperandsOp."""
    return SplitContractionOperandsOp(target=target)
