"""Transform schedules of the opt pipeline (``pipeline.yaml``), built from
lighthouse's and this backend's transform ops (``mlir_transforms``)."""

from __future__ import annotations

from contextlib import contextmanager
import inspect
from typing import TYPE_CHECKING

from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import loop as transform_loop
from mlir.dialects.transform import math as transform_math
from mlir.dialects.transform import structured
from mlir.dialects.transform import tensor as transform_tensor
from mlir.dialects.transform import vector as transform_vector
from mlir.dialects.transform import x86 as transform_x86

from helion_mlir_backend._compiler import mlir_transforms as ht
from helion_mlir_backend._compiler.mlir_transforms.first_touch import (
    MADV_POPULATE_WRITE,
)
from helion_mlir_backend._compiler.mlir_transforms.first_touch import (
    MIN_BYTES as FIRST_TOUCH_MIN_BYTES,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _suppressing(op: ir.Value) -> Iterator[ir.Value]:
    """A sequence on ``op`` whose silenceable failures are ignored."""
    sequence = transform.SequenceOp(transform.FailurePropagationMode.Suppress, [], op)
    with ir.InsertionPoint(sequence.body):
        yield sequence.bodyTarget
        transform.yield_()


def vectorize_linalg() -> ir.Module:
    """Schedule: lighthouse's ``vectorization.py[gen=vectorize_linalg]``, also
    vectorizing ops of runtime shape, with masks. Runtime extents without an
    evident bound and ops of too large vectors are first tiled. Ops that cannot
    be tiled or vectorized (e.g. argmax, gathers) are left to the loop lowering.
    Dot products are generalized first: upstream vectorizes a named
    contraction into a rank-0 result to an invalid ``vector.contract``."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        structured.structured_generalize(
            transform.AnyOpType.get(), ht.match_scalar_contractions(funcs)
        )
        to_tile = ht.partition_linalg(funcs)[-1]
        with lh_transform.foreach(to_tile) as op:
            sizes = ht.tile_sizes(op)
            with _suppressing(op) as target:
                structured.TileUsingForOp(target, sizes=sizes)
            transform.yield_()
        static, *dynamic, _ = ht.partition_linalg(funcs)
        with lh_transform.foreach(static) as op:
            with _suppressing(op) as target:
                structured.structured_vectorize(
                    target, [], create_named_contraction=True
                )
            transform.yield_()
        for loops, group in enumerate(dynamic, 1):
            with lh_transform.foreach(group) as op:
                with _suppressing(op) as target:
                    bounds = ht.loop_bounds(target, loops)
                    structured.structured_vectorize(
                        target,
                        bounds,
                        static_vector_sizes=[ir.ShapedType.get_dynamic_size()] * loops,
                        scalable_sizes=[False] * loops,
                        create_named_contraction=True,
                    )
                transform.yield_()
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            # Masked transfers as transfers with a mask operand: later patterns
            # would otherwise build ops inside vector.mask regions.
            transform_vector.apply_patterns_vector_lower_masked_transfers()
            transform_vector.apply_patterns_vector_reduction_to_contract()
            transform_vector.apply_patterns_vector_transfer_permutation_patterns()
            transform_vector.apply_patterns_vector_fold_arith_extension()
        ht.unmask_contractions(funcs)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def vectorize_pads() -> ir.Module:
    """Schedule: vectorize the pads of every function."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.vectorize_pads(funcs)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def lower_transposes(strategy: str = "Shuffle16x16") -> ir.Module:
    """Schedule: every function's vector transposes as transposes of their
    widest elements (``widen_transposes``), lowered by upstream's ``strategy``
    (``Shuffle16x16``: 2-D ones as shuffles, 16x16 ones of 32 bits as AVX-512's
    unpack/permute sequence)."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.widen_transposes(funcs)
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            transform_vector.apply_patterns_vector_lower_transpose(
                lowering_strategy=transform_vector.VectorTransposeLowering[strategy]
            )
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule


def unroll_transfers(max_rank: int = 2) -> ir.Module:
    """Schedule: every function's transfers of rank above ``max_rank`` unrolled
    to ones of that rank (upstream ``transfer_to_scf``), flattened where
    contiguous, and the rest unrolled to 1-D: e.g. a 16x16x2 block of a pack's
    strided source as 16 reads of 32 elements, not staged through memory element
    pair by pair."""
    with schedule_boilerplate() as (schedule, named_seq):
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            transform_vector.apply_patterns_vector_transfer_to_scf(
                max_transfer_rank=max_rank, full_unroll=True
            )
        lh_transform.flatten_vector_ops(named_seq.bodyTarget)
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            transform_vector.apply_patterns_vector_transfer_to_scf(
                max_transfer_rank=1, full_unroll=True
            )
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule


def linearize_vector_ops() -> ir.Module:
    """Schedule: every function's n-D elementwise ops and constants as 1-D ones
    (upstream vector linearization), e.g. a cast between the reshapes of
    flattened transfers: lowered n-D, it is unrolled row by row."""
    with schedule_boilerplate() as (schedule, named_seq):
        linearize = transform.ApplyConversionPatternsOp(
            named_seq.bodyTarget,
            num_default_type_converter_region=1,
            legal_dialects=["vector"],
            partial_conversion=True,
            preserve_handles=True,
        )
        with ir.InsertionPoint(linearize.patterns.blocks.append()):
            transform_vector.apply_conversion_patterns_vector_linearize(
                shuffle_like_ops=False
            )
            transform_loop.apply_conversion_patterns_scf_structural_conversions()
        with ir.InsertionPoint(
            linearize.default_type_converter_region[0].blocks.append()
        ):
            transform_vector.VectorLinearizeTypeConverterOp()
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule


def materialize_copies() -> ir.Module:
    """Schedule: large slice moves of every function as ``linalg.copy`` ops."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.materialize_copies(funcs)
        transform.yield_()
    return schedule


def fold_pad_unit_inserts() -> ir.Module:
    """Schedule: pads put back into unit dims by an insert into an empty tensor,
    in every function, as reshapes."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.fold_pad_unit_inserts(funcs)
        transform.yield_()
    return schedule


def materialize_operand_pads(packed_only: bool = False) -> ir.Module:
    """Schedule: every function's zero-padded contraction operands in memory
    (with ``packed_only``, those read through an operand pack)."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.materialize_operand_pads(funcs, packed_only=packed_only)
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule


def version_padded_operands() -> ir.Module:
    """Schedule: branch every function's contractions on their runtime-padded
    inputs padding nothing."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.version_padded_operands(funcs)
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule


def align_operand_rows() -> ir.Module:
    """Schedule: copy contraction inputs read in place from misaligned rows
    into buffers of aligned rows."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.align_operand_rows(funcs)
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule


def prefetch_large_gemm() -> ir.Module:
    """Schedule: in short-K AMX GEMMs of whole-row tiles with outputs beyond the
    last-level cache, each register block's output lines prefetched for writing."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.prefetch_large_gemm(funcs)
        transform.yield_()
    return schedule


def split_bf16_loads() -> ir.Module:
    """Schedule: every LLVM load of a bf16 vector of 8k+1 elements (k > 0) as
    loads of the first 8k and of the last."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        ht.split_bf16_loads(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def split_contraction_operands() -> ir.Module:
    """Schedule: every function's contraction operands computed in the body
    of a ``linalg.generic`` (e.g. fused ``exp`` or bf16 rounding), which the
    raising to ``linalg.contract`` would drop, as elementwise generics."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.split_contraction_operands(funcs)
        transform.yield_()
    return schedule


def normalize_amx_contractions() -> ir.Module:
    """Schedule: every function's bf16 contractions that lighthouse tiles for
    AMX in the layouts upstream's AMX rewrite assumes but does not check:
    inputs with swapped innermost dims (e.g. ``x @ w.T``) and ``(N, M)``
    accumulators get transposed copies; layouts it cannot normalize are marked
    to be tiled as without AMX (``pin_amx_off_tile_sizes``); memref ones not in
    the expected layout get their inputs widened to f32."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.normalize_amx_contractions(funcs)
        transform.yield_()
    return schedule


def pin_amx_off_tile_sizes(strategy: str = "register_parallel") -> ir.Module:
    """Schedule: every function's contractions kept off AMX annotated with the
    ``strategy`` tile sizes of the host without AMX."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.pin_amx_off_tile_sizes(funcs, strategy=strategy)
        transform.yield_()
    return schedule


def split_copies() -> ir.Module:
    """Schedule: every function's out-of-bounds n-D memref copies row by row,
    split on an in-bounds check."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.split_copies(funcs)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def forward_rows() -> ir.Module:
    """Schedule: every function's vectors built from whole rows and used row by
    row forwarded as rows."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.forward_rows(funcs)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def split_transfers() -> ir.Module:
    """Schedule: split every function's out-of-bounds memref transfers on an
    in-bounds check."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.split_transfers(funcs)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def amx_move_offsets() -> ir.Module:
    """Schedule: move the offsets of every function's in-bounds memref reads to
    subviews, in the form the x86 AMX contraction rewrite matches, then fit
    the subviews to the reads: the rewrite takes an accumulator read's memref
    as the register tile. Contraction operand and accumulator reads that
    skip unit dims of their buffer are first made minor-identity reads of a
    view without them (the rewrite ignores read maps: it loads operand rows
    along the memref's inner dims, and accumulators in place only from
    minor-identity reads). Last, on an AMX target, bf16 contractions with an
    operand read the rewrite would still misread are widened to f32."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.drop_contraction_read_unit_dims(funcs)
        ht.move_offsets_to_subview(funcs)
        transform.apply_cse(funcs)
        ht.fit_subviews_to_reads(funcs)
        ht.widen_misread_amx_operands(funcs)
        transform.yield_()
    return schedule


def pin_transposes() -> ir.Module:
    """Schedule: keep every function's small static transposes untiled."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.pin_transposes(funcs)
        transform.yield_()
    return schedule


def isolate_operand_packs() -> ir.Module:
    """Schedule: tile every operand pack on its outer loop, by its transpose
    tile (``outer_transpose_tile``).

    A loop result is no producer tile-and-fuse can fuse: the pack stays outside
    the contraction's register loops and runs once per contraction, not once
    per register tile reading it.
    """
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.mark_operand_packs(funcs)
        packs = structured.MatchOp(
            transform.any_op_t(),
            named_seq.bodyTarget,
            op_attrs=ir.DictAttr.get({ht.OPERAND_PACK_ATTR_NAME: ir.UnitAttr.get()}),
        )
        with lh_transform.foreach(packs) as pack:
            size = ht.outer_transpose_tile(pack)
            structured.TileUsingForOp(pack, sizes=[size])
            transform.yield_()
        transform.yield_()
    return schedule


def mark_overwritten_args() -> ir.Module:
    """Schedule: mark the arguments each function's top-level parallel loop
    overwrites whole, for ``first_touch_outputs``."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.mark_overwritten_args(funcs)
        transform.yield_()
    return schedule


def first_touch_outputs(
    min_bytes: int = FIRST_TOUCH_MIN_BYTES, advice: int = MADV_POPULATE_WRITE
) -> ir.Module:
    """Schedule: the pages of outputs marked by ``mark_overwritten_args`` (of at
    least ``min_bytes``) faulted in by each thread in a contiguous share
    (``madvise(advice)``) at the start of the OpenMP region first writing them:
    faulting a fresh output in, dozens of threads then neither zero the same
    huge page nor contend on one page table lock."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.first_touch_outputs(funcs, min_bytes, advice)
        transform.yield_()
    return schedule


def hoist_allocas() -> ir.Module:
    """Schedule: hoist every function's static stack buffers out of loops."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.hoist_allocas(funcs)
        transform.yield_()
    return schedule


def fold_full_insert_slices() -> ir.Module:
    """Schedule: inserts overwriting all of a dynamically sized destination
    (the full and partial versions of a grid tile storing their whole region)
    replaced by their source, so bufferization writes the tile in place."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.fold_full_insert_slices(funcs)
        transform.yield_()
    return schedule


def sink_copies_into_branches() -> ir.Module:
    """Schedule: copies of an ``scf.if`` result made in its branches, where a
    full tile's copy onto its own region folds away."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.sink_copies_into_branches(funcs)
        transform.yield_()
    return schedule


def fold_empty_slices() -> ir.Module:
    """Schedule: slices of empty tensors as empty tensors of the slice shape.

    A per-register-tile op writing a slice of a whole-tile temporary (e.g. a
    fused accumulator or epilogue) then gets a register-tile-sized buffer, not
    one the size of the whole tile streaming through the caches. A tile
    rank-expanded into an empty tensor before its insert into the output is
    inserted directly, so bufferization writes it in place.
    """
    with schedule_boilerplate() as (schedule, named_seq):
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            transform_tensor.apply_patterns_tensor_fold_tensor_empty()
            transform_tensor.apply_patterns_tensor_merge_consecutive_insert_extract_slice()
            transform.apply_patterns_canonicalization()
        transform.yield_()
    return schedule


def schedule_amx_loads(distance: int = 1) -> ir.Module:
    """Schedule: interleave every function's AMX operand loads with its
    dot-products, each ``distance`` dot-products ahead of its first use."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.schedule_amx_loads(funcs, distance=distance)
        transform.yield_()
    return schedule


def promote_buffers_to_stack(max_alloc_size_in_bytes: int = 262144) -> ir.Module:
    """Schedule: ``promote-buffers-to-stack`` on every function, with a size limit
    (the pipeline descriptor cannot nest a pass with options)."""
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        transform.apply_registered_pass(
            transform.AnyOpType.get(),
            funcs,
            "promote-buffers-to-stack",
            options={"max-alloc-size-in-bytes": max_alloc_size_in_bytes},
        )
        transform.yield_()
    return schedule


def _polynomial_approximation_patterns() -> None:
    """Math's polynomial approximation patterns with the AVX2 ``rsqrt`` one, at
    the insertion point. MLIR moved the AVX2 ``rsqrt`` approximation out of
    ``polynomial_approximation{enable_avx2}`` into the x86 transform op
    ``math_rsqrt_approximation`` (llvm-project #236807): use whichever form the
    bindings have."""
    op = transform_math.ApplyPolynomialApproximationPatternsOp
    if "enable_avx2" in inspect.signature(op.__init__).parameters:
        op(enable_avx2=True)
        return
    op()
    transform_x86.ApplyMathRsqrtApproximationPatternsOp()


def approximate_math() -> ir.Module:
    """Schedule: math functions (``exp``, ``tanh``, ``erf``, ...) as polynomial
    approximations: LLVM lowers a vector one to a libm call per element."""
    with schedule_boilerplate() as (schedule, named_seq):
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            transform_math.ApplyF32ExpansionPatternsOp()
            _polynomial_approximation_patterns()
        transform.yield_()
    return schedule


def legalize_for_llvm() -> ir.Module:
    """Schedule: legalize every function's vector ops for the LLVM lowering."""
    ht.HelionTransformDialect.load()
    with schedule_boilerplate() as (schedule, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ht.legalize_for_llvm(funcs)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule
