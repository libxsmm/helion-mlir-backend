"""A K loop accumulating onto an earlier K loop's result, on AMX.

AI-bench's prepacked VNNI GEMM (``_matmul_prepacked_vnni``, copied below) runs
K in chunks: a first K loop, then a second (and third) loop over the rest,
accumulating onto the first one's ``M x 1 x N`` result (a panel dim of size
1). Once ``simplify_vector_ops`` folds the rank-reducing slices of the second
loop's register tiles into its accumulator reads, they read the buffer
directly, with a map skipping the unit dim (``(d0, d1, d2) -> (d0, d2)``).
Upstream's AMX rewrite loads only minor-identity accumulator reads in place;
for others it takes the read's memref as the register tile (``1 x N``), so it
stored the result tiles past a ``1 x N`` stack buffer (segfault) or indexed past
its results (``SmallVector::operator[]`` assertion), whenever the first loop
ran two or more iterations. ``drop_contraction_read_unit_dims`` (first step of
``amx_move_offsets``) moves such reads onto a rank-reducing view without the
unit dims, which ``move_offsets_to_subview`` and ``fit_subviews_to_reads`` then
fit to the register tile, as for any other accumulator.

The transform and IR-level tests force AMX on (any x86 host); the numeric
regressions execute AMX code and run on AMX hosts only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import x86
import pytest
import torch
from torch import Tensor

from tests.harness import amx_features
from tests.harness import lower_opt_with_amx

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler import helion_transforms
from helion_mlir_backend._compiler.mlir_transforms import HelionTransformDialect
from helion_mlir_backend._compiler.mlir_transforms import (
    drop_contraction_read_unit_dims,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_HAS_AMX = TargetInfo.host().is_supported("amx")
amx_host_only = pytest.mark.skipif(not _HAS_AMX, reason="executes AMX code")


# --- The kernel ---------------------------------------------------------------
# Copied from AI-bench's backends/utils/helion_mlir_cpu_utils/matmul.py
# (backend tests do not import AI-bench).


def identity_epilogue(x: Tensor) -> Tensor:
    return x


def _panel_dot(
    a3: Tensor, b4: Tensor, tile_m: object, tile_p: object, tile_kp: object
) -> Tensor:
    return torch.einsum(
        "mcv,bcnv->mbn", a3[tile_m, tile_kp, :], b4[tile_p, tile_kp, :, :]
    )


def _matmul_prepacked_vnni(
    a3: Tensor,
    b4: Tensor,
    bias2: Tensor | None,
    scale2: Tensor | None,
    shift2: Tensor | None,
    epilogue: Callable[[Tensor], Tensor],
    pairs_chunked: hl.constexpr,
    pairs_even: hl.constexpr,
    out: Tensor | None = None,
) -> Tensor:
    m, pairs, _ = a3.shape
    panels, _, panel, _ = b4.shape
    pairs = hl.specialize(pairs)
    panel = hl.specialize(panel)
    hl.specialize(b4.size(1))
    hl.specialize(a3.size(2))
    hl.specialize(b4.size(3))

    if out is None:
        result = torch.empty((m, panels, panel), dtype=a3.dtype, device=a3.device)
    else:
        result = out
    for tile_m, tile_p in hl.tile([m, panels]):
        acc = hl.zeros([tile_m, tile_p, panel], dtype=torch.float32)
        if pairs_chunked > 0:
            for tile_kp in hl.tile(pairs_chunked):
                acc = acc + _panel_dot(a3, b4, tile_m, tile_p, tile_kp)
        if pairs_even > pairs_chunked:
            for tile_rest in hl.tile(pairs_chunked, pairs_even):
                acc = acc + _panel_dot(a3, b4, tile_m, tile_p, tile_rest)
        if pairs > pairs_even:
            for tile_tail in hl.tile(pairs_even, pairs_even + 32, block_size=32):
                acc = acc + _panel_dot(a3, b4, tile_m, tile_p, tile_tail)
        if bias2 is not None:
            acc = acc + bias2[tile_p, :]
        if scale2 is not None:
            acc = acc * scale2[tile_p, :]
        if shift2 is not None:
            acc = acc + shift2[tile_p, :]
        result[tile_m, tile_p, :] = epilogue(acc).to(a3.dtype)
    return result


def _pack_b_vnni_t(w: Tensor, panel: int) -> Tensor:
    """``[N, K]`` into the kernel's ``[N/panel, K/2, panel, 2]`` VNNI panels
    (as AI-bench's ``pack_b_vnni_t`` for N a multiple of the panel)."""
    n, k = w.shape
    return w.view(n // panel, panel, k // 2, 2).permute(0, 2, 1, 3).contiguous()


# (M, K, N, panel, block_sizes, pairs_chunked, pairs_even) as AI-bench's
# _prepacked_bound picks them; the first K loop runs 2 or more iterations.
CASES = {
    # The reproducer: 2 chunks of 384 pairs, then one of 256 (segfault).
    "chunks_2x384_rest_256": (32, 2048, 64, 64, [32, 1, 384, 256], 768, 1024),
    # linear() at 128x4160x4096 with 64 threads (segfault).
    "linear_4160_64_threads": (128, 4160, 4096, 64, [128, 1, 704, 672], 1408, 2080),
    # linear() at 128x4160x4096 with 8 threads (SmallVector assertion).
    "linear_4160_8_threads": (128, 4160, 4096, 256, [128, 1, 256, 32], 2048, 2080),
    # 128x4096x4096 in 320-pair chunks (SmallVector assertion).
    "chunks_6x320_rest_128": (128, 4096, 4096, 64, [128, 1, 320, 128], 1920, 2048),
    # 2 chunks of 512 pairs, no rest, then the padded tail loop of 32 pairs
    # (SmallVector assertion).
    "chunks_2x512_tail_32": (32, 2080, 256, 256, [32, 1, 512, 32], 1024, 1024),
}


def _case(name: str) -> tuple[helion.Kernel, list[object], Tensor]:
    m, k, n, panel, block_sizes, pairs_chunked, pairs_even = CASES[name]
    kernel = helion.kernel(
        _matmul_prepacked_vnni,
        backend="mlir",
        config=helion.Config(block_sizes=block_sizes),
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    args = [
        a.view(m, k // 2, 2),
        _pack_b_vnni_t(w, panel),
        None,
        None,
        None,
        identity_epilogue,
        hl.constexpr(pairs_chunked),
        hl.constexpr(pairs_even),
    ]
    return kernel, args, a.float() @ w.float().T


@amx_host_only
@pytest.mark.isolated
@pytest.mark.parametrize("name", list(CASES))
def test_second_k_loop_matches_reference(name: str) -> None:
    """The kernel at the configs that crashed: correct (one bf16 rounding of
    the output; the scalar pipeline gives the same error)."""
    kernel, args, expected = _case(name)
    out = kernel(*args).view(expected.shape)
    error = (out.float() - expected).abs().max() / expected.abs().max()
    assert error.item() < 1e-2


# The six AI-bench linear() shapes (bias + relu, 64 threads, small-M panel
# rule) that aborted in lowering (review temp/review-cpu-mlp sw64, cases
# 134/148/155/159/162/166). One entry per kernel call: the bulk rows, and the
# zero-padded rest rows where M is not a whole number of row tiles.
# (M, K, N, panel, block_sizes, pairs_chunked, pairs_even)
UTILS_CASES = {
    "linear_128x3200x4096": (128, 3200, 4096, 128, [64, 1, 416, 352], 1248, 1600),
    "linear_192x3200x4096": (192, 3200, 4096, 128, [96, 1, 416, 352], 1248, 1600),
    "linear_200x3200x4096_bulk": (128, 3200, 4096, 128, [128, 1, 416, 352], 1248, 1600),
    "linear_200x3200x4096_rest": (96, 3200, 4096, 128, [96, 1, 416, 352], 1248, 1600),
    "linear_200x2080x2048_bulk": (192, 2080, 2048, 128, [64, 1, 512, 32], 1024, 1024),
    "linear_200x2080x2048_rest": (32, 2080, 2048, 128, [32, 1, 512, 32], 1024, 1024),
    "linear_255x3200x4096_rest": (128, 3200, 4096, 128, [128, 1, 416, 352], 1248, 1600),
    "linear_255x2080x2048_rest": (64, 2080, 2048, 128, [64, 1, 512, 32], 1024, 1024),
}


def _utils_case(name: str) -> tuple[helion.Kernel, list[object], Tensor]:
    m, k, n, panel, block_sizes, pairs_chunked, pairs_even = UTILS_CASES[name]
    kernel = helion.kernel(
        _matmul_prepacked_vnni,
        backend="mlir",
        config=helion.Config(block_sizes=block_sizes),
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.bfloat16)
    # As pack_b_vnni_t: K padded to whole 32-pair chunks; the kernel's tail
    # chunk reads A past its last pair (masked) and B's zero padding.
    k_pad = -(-k // 64) * 64
    w_pad = torch.nn.functional.pad(w, (0, k_pad - k))
    args = [
        a.view(m, k // 2, 2),
        _pack_b_vnni_t(w_pad, panel),
        bias.view(n // panel, panel),
        None,
        None,
        torch.relu,
        hl.constexpr(pairs_chunked),
        hl.constexpr(pairs_even),
    ]
    return kernel, args, torch.relu(a.float() @ w.float().T + bias.float())


@amx_host_only
@pytest.mark.isolated
@pytest.mark.parametrize("name", list(UTILS_CASES))
def test_utils_small_m_shapes_match_reference(name: str) -> None:
    """AI-bench linear() configs with 128-column panels and a second K loop
    (rest chunk or pairs tail) that aborted in lowering."""
    kernel, args, expected = _utils_case(name)
    out = kernel(*args).view(expected.shape)
    error = (out.float() - expected).abs().max() / expected.abs().max()
    assert error.item() < 1e-2


@pytest.mark.isolated
@pytest.mark.parametrize("name", ["chunks_2x384_rest_256", "linear_4160_8_threads"])
def test_second_k_loop_accumulates_in_place(name: str) -> None:
    """With AMX forced on, every K loop's accumulator tiles are loaded in place
    and no contraction is left: no zero tiles added from memory afterwards (the
    rewrite's path that took a ``1 x N`` row of the buffer as the tile, or
    aborted the process)."""
    kernel, args, _ = _case(name)
    module = generate_mlir(kernel, args)
    lowered = lower_opt_with_amx(
        module,
        kernel.fn.__name__,
        lambda stage, _: "x86.vector_contract_to_amx_dot_product" in stage,
    )
    assert lowered is not None
    after = str(lowered[1])
    assert "x86.amx.tile_mulf" in after
    assert "vector.contract" not in after
    assert "x86.amx.tile_zero" not in after


# --- drop_contraction_read_unit_dims --------------------------------------------

_MK = "affine_map<(m, n, k) -> (m, k)>"
_KN = "affine_map<(m, n, k) -> (k, n)>"
_MN = "affine_map<(m, n, k) -> (m, n)>"
_V4 = ", ".join(["vector<16x16xf32>"] * 4)
_SKIP_D1 = "affine_map<(d0, d1, d2) -> (d0, d2)>"


def _apply(make_op: Callable[[ir.Value], object], source: str) -> str:
    """``source`` after the transform op alone, AMX forced on."""
    with (
        TargetInfo.override(features=amx_features()),
        ir.Context(),
        ir.Location.unknown(),
    ):
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            make_op(lh_transform.match_op(named_seq.bodyTarget, "func.func"))
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def _parsed(source: str) -> str:
    with ir.Context():
        return str(ir.Module.parse(source))


def _contract(
    result: str, lhs: str, rhs: str, acc: str, t: str = "f32", k: int = 2
) -> str:
    return (
        f"{result} = vector.contract {{indexing_maps = [{_MK}, {_KN}, {_MN}], "
        'iterator_types = ["parallel", "parallel", "reduction"], '
        f"kind = #vector.kind<add>}} {lhs}, {rhs}, {acc} : vector<16x{k}x{t}>, "
        f"vector<{k}x16x{t}> into vector<16x16xf32>"
    )


def _k_loop(
    buffer: str = "memref<32x1x64xf32>",
    index: Callable[[str, str], str] = lambda r, c: f"%c{r}, %c0, {c}",
    permutation: str = _SKIP_D1,
    last: str | None = None,
    use: str = "acc",
) -> str:
    """The second K loop's accumulators: four 16x16 reads of a 32x32 register
    tile of ``buffer`` at column ``%n`` (``index`` gives a read's indices from
    its row and column), each the init of a K loop's iter arg. ``last``
    replaces the fourth read (``%i3``; ``{t}`` stands for the buffer's type);
    ``use="rhs"`` makes the reads contraction operands instead."""
    reads = [
        f"  %i{t} = vector.transfer_read %buf[{index(r, c)}], %pad "
        f"{{in_bounds = [true, true], permutation_map = {permutation}}} : "
        f"{buffer}, vector<16x16xf32>"
        for t, (r, c) in enumerate([(0, "%n"), (0, "%n16"), (16, "%n"), (16, "%n16")])
    ]
    if last is not None:
        reads[3] = "  " + last.format(t=buffer)
    if use == "acc":
        inits = "%x0 = %i0, %x1 = %i1, %x2 = %i2, %x3 = %i3"
        contracts = [_contract(f"%y{t}", "%a", "%b", f"%x{t}") for t in range(4)]
    else:
        inits = "%x0 = %z, %x1 = %z, %x2 = %z, %x3 = %z"
        contracts = [
            _contract(f"%y{t}", "%a16", f"%i{t}", f"%x{t}", k=16) for t in range(4)
        ]
    body = "\n".join("    " + line for line in contracts)
    return f"""
func.func @f(%buf: {buffer}, %n: index, %k: index, %a: vector<16x2xf32>,
             %b: vector<2x16xf32>, %a16: vector<16x16xf32>, %z: vector<16x16xf32>,
             %m: vector<16x16xi1>) -> ({_V4}) {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %pad = arith.constant 0.0 : f32
  %n16 = affine.apply affine_map<()[s0] -> (s0 + 16)>()[%n]
{chr(10).join(reads)}
  %r:4 = scf.for %kk = %c0 to %k step %c1
      iter_args({inits}) -> ({_V4}) {{
{body}
    scf.yield %y0, %y1, %y2, %y3 : {_V4}
  }}
  return %r#0, %r#1, %r#2, %r#3 : {_V4}
}}
"""


def _reads(text: str) -> list[str]:
    return [
        line.strip() for line in text.splitlines() if "vector.transfer_read" in line
    ]


def test_drop_unit_dims_of_second_k_loop_reads() -> None:
    """The kernel's form: reads of a ``32x1x64`` buffer skipping its unit dim
    become minor-identity reads of a ``32x64`` view, at the same rows and
    columns."""
    text = _apply(drop_contraction_read_unit_dims, _k_loop())
    assert (
        "memref.subview %arg0[0, 0, 0] [32, 1, 64] [1, 1, 1] : memref<32x1x64xf32> "
        "to memref<32x64xf32, strided<[64, 1]>>" in text
    )
    reads = _reads(text)
    assert len(reads) == 4
    assert all("permutation_map" not in read for read in reads)
    assert all(
        "memref<32x64xf32, strided<[64, 1]>>, vector<16x16xf32>" in r for r in reads
    )
    assert [read.split("[")[1].split("]")[0] for read in reads] == [
        "%c0, %arg1",
        "%c0, %0",
        "%c16, %arg1",
        "%c16, %0",
    ]


def test_drop_unit_dims_runtime_index_dynamic_layout() -> None:
    """A runtime index of the unit dim becomes the view's offset (dynamic);
    a dynamic leading dim keeps its runtime size; strides stay."""
    buffer = "memref<?x1x64xf32, strided<[128, 64, 1], offset: ?>>"
    text = _apply(
        drop_contraction_read_unit_dims,
        _k_loop(buffer, index=lambda r, c: f"%c{r}, %k, {c}"),
    )
    assert "memref.dim %arg0, %c0" in text
    assert (
        f"[0, %arg2, 0] [%dim, 1, 64] [1, 1, 1] : {buffer} to "
        "memref<?x64xf32, strided<[128, 1], offset: ?>>" in text
    )
    assert all("permutation_map" not in read for read in _reads(text))


def test_drop_unit_dims_keeps_leading_dims() -> None:
    """Leading dims the map skips stay (a minor identity allows them); unit
    dims between the vector's dims go, several at once."""
    buffer = "memref<4x32x1x1x64xf32>"
    text = _apply(
        drop_contraction_read_unit_dims,
        _k_loop(
            buffer,
            index=lambda r, c: f"%c1, %c{r}, %c0, %c0, {c}",
            permutation="affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>",
        ),
    )
    assert (
        f"[0, 0, 0, 0, 0] [4, 32, 1, 1, 64] [1, 1, 1, 1, 1] : {buffer} to "
        "memref<4x32x64xf32, strided<[2048, 64, 1]>>" in text
    )
    reads = _reads(text)
    # Minor identity: the map is left out.
    assert all("permutation_map" not in read for read in reads)
    assert all(
        "memref<4x32x64xf32, strided<[2048, 64, 1]>>, vector<16x16xf32>" in r
        for r in reads
    )
    assert reads[2].split("[")[1].split("]")[0] == "%c1, %c16, %arg1"


def test_amx_move_offsets_fits_second_k_loop_tile() -> None:
    """The whole ``amx_move_offsets`` schedule: the register tile gets a
    ``32x32`` subview at its column, read from its origin, as the AMX rewrite
    expects (unfixed, the reads kept the ``32x1x64`` buffer)."""
    with (
        TargetInfo.override(features=amx_features()),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.parse(_k_loop())
        schedule = helion_transforms.amx_move_offsets()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        text = str(module)
    assert "[0, %arg1] [32, 32] [1, 1]" in text
    reads = _reads(text)
    assert all("memref<32x32xf32, strided<[64, 1], offset: ?>>" in r for r in reads)
    assert sorted(read.split("[")[1].split("]")[0] for read in reads) == [
        "%c0, %c0",
        "%c0, %c16",
        "%c16, %c0",
        "%c16, %c16",
    ]


_IN_BOUNDS = f"{{{{in_bounds = [true, true], permutation_map = {_SKIP_D1}}}}}"


@pytest.mark.parametrize(
    "source",
    [
        # Permuted dims (transposed).
        _k_loop(permutation="affine_map<(d0, d1, d2) -> (d2, d0)>"),
        # A skipped dim that is not unit.
        _k_loop("memref<32x2x64xf32>"),
        # A skipped dynamic dim.
        _k_loop("memref<32x?x64xf32>"),
        # Minor identity already (a leading unit dim skipped).
        _k_loop(
            "memref<1x32x64xf32>",
            index=lambda r, c: f"%c0, %c{r}, {c}",
            permutation="affine_map<(d0, d1, d2) -> (d1, d2)>",
        ),
        # Tensor reads.
        _k_loop("tensor<32x1x64xf32>"),
    ],
    ids=[
        "transposed",
        "non_unit_dim",
        "dynamic_dim",
        "minor_identity",
        "tensor",
    ],
)
def test_drop_unit_dims_leaves_others(source: str) -> None:
    assert _apply(drop_contraction_read_unit_dims, source) == _parsed(source)


def test_drop_any_static_dim_of_operand_reads() -> None:
    """Operand reads drop a skipped dim of any static size (a projection at
    the read's index): an ``a[m, s, k]`` chunk at a runtime ``s`` of a
    ``32x8x64`` buffer becomes a read of a ``32x64`` view at ``s``."""
    text = _apply(
        drop_contraction_read_unit_dims,
        _k_loop(
            "memref<32x8x64xf32>",
            index=lambda r, c: f"%c{r}, %k, {c}",
            use="rhs",
        ),
    )
    assert (
        "memref.subview %arg0[0, %arg2, 0] [32, 1, 64] [1, 1, 1] : "
        "memref<32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>" in text
    )
    assert all("permutation_map" not in read for read in _reads(text))


def test_drop_any_static_dim_constant_index_in_range() -> None:
    """A constant index inside the skipped dim (7 of 8) is the view's static
    offset."""
    text = _apply(
        drop_contraction_read_unit_dims,
        _k_loop(
            "memref<32x8x64xf32>",
            index=lambda r, c: f"%c{r}, %c7, {c}",
            use="rhs",
        ).replace(
            "%c16 = arith.constant 16 : index",
            "%c16 = arith.constant 16 : index\n  %c7 = arith.constant 7 : index",
        ),
    )
    assert "[0, 7, 0] [32, 1, 64] [1, 1, 1]" in text
    assert "strided<[512, 1], offset: 448>" in text


@pytest.mark.parametrize(
    "source",
    [
        # Accumulators drop unit dims only (their subviews are register tiles).
        _k_loop("memref<32x8x64xf32>", index=lambda r, c: f"%c{r}, %k, {c}"),
        # A dynamic skipped dim.
        _k_loop("memref<32x?x64xf32>", use="rhs"),
        # A constant index past the skipped dim (8 of 8).
        _k_loop(
            "memref<32x8x64xf32>",
            index=lambda r, c: f"%c{r}, %c8, {c}",
            use="rhs",
        ).replace(
            "%c16 = arith.constant 16 : index",
            "%c16 = arith.constant 16 : index\n  %c8 = arith.constant 8 : index",
        ),
        # A permuted (transposing) operand read.
        _k_loop(
            "memref<64x8x32xf32>",
            index=lambda r, c: f"{c}, %c0, %c{r}",
            permutation="affine_map<(d0, d1, d2) -> (d2, d0)>",
            use="rhs",
        ),
    ],
    ids=[
        "acc_non_unit",
        "operand_dynamic",
        "operand_index_past_end",
        "operand_permuted",
    ],
)
def test_drop_any_static_dim_leaves_others(source: str) -> None:
    assert _apply(drop_contraction_read_unit_dims, source) == _parsed(source)


_FOURTH = "%i3 = vector.transfer_read %buf[%c16, %c0, %n16], %pad"
_FOURTH_TYPE = " : {t}, vector<16x16xf32>"


@pytest.mark.parametrize(
    ("last", "callee"),
    [
        # A read also used otherwise (passed to a call).
        (
            (
                f"{_FOURTH} {_IN_BOUNDS}{_FOURTH_TYPE}\n"
                "  func.call @g(%i3) : (vector<16x16xf32>) -> ()"
            ),
            True,
        ),
        # A masked read.
        (f"{_FOURTH}, %m {_IN_BOUNDS}{_FOURTH_TYPE}", False),
        # The plain read: rewritten like the others (the control).
        (f"{_FOURTH} {_IN_BOUNDS}{_FOURTH_TYPE}", False),
    ],
    ids=["other_use", "masked", "control"],
)
def test_drop_unit_dims_leaves_other_reads(last: str, callee: bool) -> None:
    """Matched per read: a read used otherwise or masked keeps its buffer and
    map, the other three of its tile are rewritten (and, in the control, it
    too)."""
    source = _k_loop(last=last)
    if callee:
        source = source.replace(
            "func.func @f", "func.func private @g(vector<16x16xf32>)\nfunc.func @f"
        )
    reads = _reads(_apply(drop_contraction_read_unit_dims, source))
    assert all("permutation_map" not in read for read in reads[:3])
    control = "%c16, %c0" not in reads[3]
    assert control == ("%m" not in last and not callee)
    assert ("permutation_map" in reads[3]) != control


def _one_acc_read(buffer: str, read: str, *, args: str = "") -> str:
    """One accumulator read ``%r`` of ``%buf: buffer`` (``read`` is its
    attributes and types after the indices), a contraction's accumulator."""
    return f"""
func.func @f(%buf: {buffer}, %i: index, %a: vector<16x2xf32>, %b: vector<2x16xf32>,
             %m: vector<16x16xi1>{args}) -> vector<16x16xf32> {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %pad = arith.constant 0.0 : f32
  %r = vector.transfer_read %buf{read}
  {_contract("%y", "%a", "%b", "%r")}
  return %y : vector<16x16xf32>
}}
"""


_SKIP_READ = f"{{in_bounds = [true, true], permutation_map = {_SKIP_D1}}}"


@pytest.mark.parametrize(
    ("buffer", "index", "view"),
    [
        # A dynamic stride of the dropped dim at index 0 adds nothing: the
        # static offset stays (as memref.subview infers it).
        (
            "memref<32x1x64xf32, strided<[64, ?, 1], offset: 7>>",
            "%c0",
            "memref<32x64xf32, strided<[64, 1], offset: 7>>",
        ),
        # A runtime index with a stride of 0 adds nothing either.
        (
            "memref<32x1x64xf32, strided<[64, 0, 1], offset: 7>>",
            "%i",
            "memref<32x64xf32, strided<[64, 1], offset: 7>>",
        ),
        # A runtime index with a static stride: a dynamic offset.
        (
            "memref<32x1x64xf32, strided<[128, 64, 1], offset: 7>>",
            "%i",
            "memref<32x64xf32, strided<[128, 1], offset: ?>>",
        ),
        # A dynamic base offset stays dynamic.
        (
            "memref<32x1x64xf32, strided<[64, 64, 1], offset: ?>>",
            "%c0",
            "memref<32x64xf32, strided<[64, 1], offset: ?>>",
        ),
    ],
    ids=[
        "dynamic_stride_index_0",
        "stride_0_runtime_index",
        "runtime_index",
        "dynamic_offset",
    ],
)
def test_drop_unit_dims_view_offset(buffer: str, index: str, view: str) -> None:
    """The view's offset is the one memref.subview infers (else it does not
    verify: "mismatch of result layout")."""
    source = _one_acc_read(
        buffer, f"[%c1, {index}, %c3], %pad {_SKIP_READ} : {buffer}, vector<16x16xf32>"
    )
    text = _apply(drop_contraction_read_unit_dims, source)
    assert f"{buffer} to {view}" in text
    assert "permutation_map" not in _reads(text)[0]


_AFFINE = "memref<32x1x64xf32, affine_map<(d0, d1, d2) -> (d2, d1, d0)>>"


@pytest.mark.parametrize(
    "source",
    [
        # No strided layout (an affine map): no subview to take.
        _one_acc_read(
            _AFFINE,
            f"[%c0, %c0, %c0], %pad {_SKIP_READ} : {_AFFINE}, vector<16x16xf32>",
        ),
        # A constant index other than 0 on the unit dim (out of bounds).
        _one_acc_read(
            "memref<32x1x64xf32>",
            f"[%c0, %c1, %c0], %pad {_SKIP_READ} : memref<32x1x64xf32>, "
            "vector<16x16xf32>",
        ),
        # A broadcast dim.
        _one_acc_read(
            "memref<32x1x64xf32>",
            "[%c0, %c0, %c0], %pad {in_bounds = [true, true], permutation_map = "
            "affine_map<(d0, d1, d2) -> (d0, 0)>} : memref<32x1x64xf32>, "
            "vector<16x16xf32>",
        ),
    ],
    ids=["affine_layout", "constant_index_1", "broadcast"],
)
def test_drop_unit_dims_leaves_other_layouts(source: str) -> None:
    assert _apply(drop_contraction_read_unit_dims, source) == _parsed(source)


_PLAIN = (
    "%r = vector.transfer_read %buf[%c0, %c0, %c0], %pad "
    f"{_SKIP_READ} : memref<32x1x64xf32>, vector<16x16xf32>"
)


def _func(body: str) -> str:
    return f"""
func.func @f(%buf: memref<32x1x64xf32>, %k: index, %a: vector<16x2xf32>,
             %b: vector<2x16xf32>, %z: vector<16x16xf32>,
             %m: vector<16x16xi1>) -> vector<16x16xf32> {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %pad = arith.constant 0.0 : f32
{body}
}}
"""


_LHS_MAPS = (
    f"indexing_maps = [{_MK}, {_KN}, {_MN}], "
    'iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>'
)


@pytest.mark.parametrize(
    "source",
    [
        # Inside vector.mask.
        _func(
            "  %r = vector.mask %m { vector.transfer_read %buf[%c0, %c0, %c0], %pad "
            f"{_SKIP_READ} : memref<32x1x64xf32>, vector<16x16xf32> }} : "
            "vector<16x16xi1> -> vector<16x16xf32>\n"
            f"  {_contract('%y', '%a', '%b', '%r')}\n"
            "  return %y : vector<16x16xf32>"
        ),
        # A K loop's iter arg also yielded as another result.
        _func(
            f"  {_PLAIN}\n"
            "  %o:2 = scf.for %kk = %c0 to %k step %c1 iter_args(%x = %r, %w = %r)"
            " -> (vector<16x16xf32>, vector<16x16xf32>) {\n"
            f"    {_contract('%y', '%a', '%b', '%x')}\n"
            "    scf.yield %y, %x : vector<16x16xf32>, vector<16x16xf32>\n"
            "  }\n"
            "  return %o#1 : vector<16x16xf32>"
        ),
        # A contraction operand through another op (a negation).
        _func(
            f"  {_PLAIN}\n"
            "  %e = arith.negf %r : vector<16x16xf32>\n"
            f"  %y = vector.contract {{{_LHS_MAPS}}} %e, %z, %z : vector<16x16xf32>, "
            "vector<16x16xf32> into vector<16x16xf32>\n"
            "  return %y : vector<16x16xf32>"
        ),
    ],
    ids=["vector_mask", "iter_arg_yielded", "operand_through_other_op"],
)
def test_drop_unit_dims_leaves_other_uses(source: str) -> None:
    assert _apply(drop_contraction_read_unit_dims, source) == _parsed(source)


@pytest.mark.parametrize(
    "body",
    [
        # One accumulator read.
        (
            f"  {_PLAIN}\n  {_contract('%y', '%a', '%b', '%r')}\n"
            "  return %y : vector<16x16xf32>"
        ),
        # The LHS and the accumulator of one contraction.
        (
            f"  {_PLAIN}\n"
            f"  %y = vector.contract {{{_LHS_MAPS}}} %r, %z, %r : vector<16x16xf32>, "
            "vector<16x16xf32> into vector<16x16xf32>\n"
            "  return %y : vector<16x16xf32>"
        ),
        # The LHS of one contraction and the RHS of another.
        (
            f"  {_PLAIN}\n"
            f"  %y = vector.contract {{{_LHS_MAPS}}} %r, %z, %z : vector<16x16xf32>, "
            "vector<16x16xf32> into vector<16x16xf32>\n"
            f"  %w = vector.contract {{{_LHS_MAPS}}} %z, %r, %y : vector<16x16xf32>, "
            "vector<16x16xf32> into vector<16x16xf32>\n"
            "  return %w : vector<16x16xf32>"
        ),
    ],
    ids=["accumulator", "lhs_and_accumulator", "lhs_and_rhs"],
)
def test_drop_unit_dims_other_uses_control(body: str) -> None:
    """The ``_func`` cases' plain forms (contraction uses only) are
    rewritten."""
    assert (
        "permutation_map"
        not in _reads(_apply(drop_contraction_read_unit_dims, _func(body)))[0]
    )


def test_drop_unit_dims_of_operand_reads() -> None:
    """Contraction operand reads skipping a unit dim become minor-identity
    reads of a ``32x64`` view, at the same rows and columns, as accumulator
    reads do (upstream's AMX rewrite loads operand rows along the memref's
    two inner dims, ignoring the map)."""
    text = _apply(drop_contraction_read_unit_dims, _k_loop(use="rhs"))
    assert (
        "memref.subview %arg0[0, 0, 0] [32, 1, 64] [1, 1, 1] : memref<32x1x64xf32> "
        "to memref<32x64xf32, strided<[64, 1]>>" in text
    )
    reads = _reads(text)
    assert len(reads) == 4
    assert all("permutation_map" not in read for read in reads)


# The kernel's second K loop as ``amx_move_offsets`` gets it (case
# "chunks_2x384_rest_256"): accumulators read from the first loop's 32x1x64
# result skipping its unit dim, K pairs from 768, results written to a copy.
_ACC_MAPS = (
    "indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, "
    "affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>, "
    "affine_map<(d0, d1, d2, d3) -> (d0, d1)>], "
    'iterator_types = ["parallel", "parallel", "reduction", "reduction"], '
    "kind = #vector.kind<add>"
)


def _bf16_second_k_loop() -> str:
    acc = "memref<32x1x64xf32>"
    a = "memref<32x1024x2xbf16>"
    b = "memref<1x1024x64x2xbf16>"
    tiles = [(0, "%n"), (0, "%n16"), (16, "%n"), (16, "%n16")]
    reads = "\n".join(
        f"    %i{t} = vector.transfer_read %acc[%c{r}, %c0, {c}], %pad "
        f"{{in_bounds = [true, true], permutation_map = {_SKIP_D1}}} : {acc}, "
        "vector<16x16xf32>"
        for t, (r, c) in enumerate(tiles)
    )
    lhs = [
        f"      %a{i} = vector.transfer_read %a[%c{16 * i}, %kk, %c0], %pb "
        f"{{in_bounds = [true, true, true]}} : {a}, vector<16x16x2xbf16>"
        for i in range(2)
    ]
    rhs = [
        f"      %b{j} = vector.transfer_read %b[%c0, %kk, {c}, %c0], %pb "
        f"{{in_bounds = [true, true, true]}} : {b}, vector<16x16x2xbf16>"
        for j, c in enumerate(["%n", "%n16"])
    ]
    contracts = [
        f"      %y{t} = vector.contract {{{_ACC_MAPS}}} %a{t // 2}, %b{t % 2}, %x{t} : "
        "vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>"
        for t in range(4)
    ]
    body = "\n".join(
        [
            lhs[0],
            rhs[0],
            contracts[0],
            rhs[1],
            contracts[1],
            lhs[1],
            contracts[2],
            contracts[3],
        ]
    )
    writes = "\n".join(
        f"    vector.transfer_write %r#{t}, %out[%c{r}, %c{c}] "
        "{in_bounds = [true, true]} : vector<16x16xf32>, memref<32x32xf32>"
        for t, (r, c) in enumerate([(0, 0), (0, 16), (16, 0), (16, 16)])
    )
    return f"""
func.func @f(%a: {a}, %b: {b}, %acc: {acc}, %out: memref<32x32xf32>) {{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %pb = arith.constant 0.0 : bf16
  %pad = arith.constant 0.0 : f32
  scf.for %n = %c0 to %c64 step %c32 {{
    %n16 = affine.apply affine_map<()[s0] -> (s0 + 16)>()[%n]
{reads}
    %r:4 = scf.for %k = %c0 to %c256 step %c16
        iter_args(%x0 = %i0, %x1 = %i1, %x2 = %i2, %x3 = %i3) -> ({_V4}) {{
      %kk = affine.apply affine_map<()[s0] -> (s0 + 768)>()[%k]
{body}
      scf.yield %y0, %y1, %y2, %y3 : {_V4}
    }}
{writes}
  }}
  return
}}
"""


def _amx_rewrite(source: str, *, fix: bool) -> str:
    """``source`` after ``amx_move_offsets`` (without its first step unless
    ``fix``) and upstream's AMX rewrite."""
    from helion_mlir_backend._compiler.mlir_transforms import fit_subviews_to_reads
    from helion_mlir_backend._compiler.mlir_transforms import move_offsets_to_subview

    with ir.Context(), ir.Location.unknown():
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
            if fix:
                drop_contraction_read_unit_dims(funcs)
            move_offsets_to_subview(funcs)
            transform.apply_cse(funcs)
            fit_subviews_to_reads(funcs)
            with ir.InsertionPoint(transform.ApplyPatternsOp(funcs).patterns):
                x86.apply_patterns_x86_vector_contract_to_amx_dot_product()
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


@pytest.mark.isolated
def test_second_k_loop_reads_reach_amx_in_place() -> None:
    """Upstream's AMX rewrite of the second loop's form after the fix loads the
    accumulator tiles from a ``32x32`` view of the buffer and stores them to
    the output; unfixed (checked here too), it starts from zero tiles and
    stores the four result tiles to a ``1x64`` stack buffer."""
    fixed = _amx_rewrite(_bf16_second_k_loop(), fix=True)
    assert fixed.count("x86.amx.tile_mulf") == 4
    assert "vector.contract" not in fixed
    assert "x86.amx.tile_zero" not in fixed
    assert fixed.count("x86.amx.tile_load %subview") == 4
    assert "memref<1x64xf32>" not in fixed
    unfixed = _amx_rewrite(_bf16_second_k_loop(), fix=False)
    assert "x86.amx.tile_zero" in unfixed
    assert "memref.alloca() : memref<1x64xf32>" in unfixed
