"""``pin_amx_off_tile_sizes``: contractions kept off AMX tiled as on a target
without AMX.

``normalize_amx_contractions`` marks the bf16 contractions upstream's AMX rewrite
would miscompile or abort on, and that it cannot normalize, with
``AMX_OFF_ATTR_NAME``. Lighthouse would give them AMX register tiles (it decides
by element types and the host's features, ``is_amx_bf16_contraction``). Run
before each of lighthouse's register tiling stages, this op annotates every
marked op with the tile sizes lighthouse's ``strategy`` computes for the host
without its AMX features; lighthouse keeps existing annotations
(``assign_tile_sizes``), so the op is tiled as on a host without AMX, and the
rewrite never sees AMX-shaped contractions of it (as on such hosts, where the
same IR is correct). The mark is a discardable attribute, which tiling copies
to the tiled ops, so each stage finds the ops the previous one made.

Matched: linalg ops carrying ``AMX_OFF_ATTR_NAME`` and no tile sizes yet. Ops
the strategy computes no tile sizes for are a definite failure: they would be
tiled for AMX."""

from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa
from lighthouse.execution.target import TargetInfo
from mlir import ir
from mlir.dialects import ext
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import is_linalg
from .utils import payload_ops

AMX_OFF_ATTR_NAME = "helion.amx_off"


def without_amx(target: TargetInfo) -> TargetInfo:
    """``target`` without its AMX features."""
    return TargetInfo(
        arch=target.arch,
        features=[f for f in target.features if not f.startswith("amx")],
        core_count=target.core_count(),
    )


@transform_op(modifies_payload=True)
class PinAmxOffTileSizesOp(
    HelionTransformDialect.Operation, name="pin_amx_off_tile_sizes"
):
    """Annotate every ``AMX_OFF_ATTR_NAME`` op of the target with ``strategy``'s
    tile sizes for the host without AMX (see the module docstring)."""

    target: ext.Operand[transform.AnyOpType]
    strategy: ir.StringAttr = ext.attribute(
        default_factory=lambda: ir.StringAttr.get("register_parallel")
    )

    @staticmethod
    def run(
        op: "PinAmxOffTileSizesOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        target = without_amx(TargetInfo.host())
        strategy = op.strategy.value
        for marked in payload_ops(state, op.target, ir.OpView):
            attributes = marked.operation.attributes
            if AMX_OFF_ATTR_NAME not in attributes or not is_linalg(marked):
                continue
            if tsa.get_tile_sizes_attr(marked) is not None:
                continue
            sizes = tsa.compute_tile_sizes(marked, strategy=strategy, target=target)
            if sizes is None:
                marked.location.emit_error(
                    "pin_amx_off_tile_sizes: no non-AMX tile sizes for a "
                    f"contraction kept off AMX (strategy {strategy})"
                )
                return DiagnosedSilenceableFailure.DefiniteFailure
            tsa.set_tile_sizes_attr(marked, sizes)
        return DiagnosedSilenceableFailure.Success


def pin_amx_off_tile_sizes(
    target: ir.Value, strategy: str = "register_parallel"
) -> PinAmxOffTileSizesOp:
    """snake_case wrapper to create a PinAmxOffTileSizesOp."""
    return PinAmxOffTileSizesOp(target=target, strategy=ir.StringAttr.get(strategy))
