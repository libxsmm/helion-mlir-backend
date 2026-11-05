"""Stores of grid tiles versioned on being full: ``fold_full_insert_slices``,
``sink_copies_into_branches`` and the full version's static region
(``lowering/loops.py``), which together keep a ragged GEMM's full tiles in
place (no per-tile heap copy of the output region, no self-copy)."""

from __future__ import annotations

import helion
import helion.language as hl
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
import pytest
import torch

from tests.harness import lower_opt_with_amx

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir_transforms import HelionTransformDialect
from helion_mlir_backend._compiler.mlir_transforms import fold_full_insert_slices
from helion_mlir_backend._compiler.mlir_transforms import sink_copies_into_branches


def _ops_named(root: ir.Operation, name: str) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def visit(nested: ir.Operation) -> ir.WalkResult:
        if nested.name == name:
            found.append(nested)
        return ir.WalkResult.ADVANCE

    root.walk(visit)
    return found


def _apply(source: str, op) -> str:
    with ir.Context(), ir.Location.unknown():
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            op(lh_transform.match_op(named_seq.bodyTarget, "func.func"))
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def _parsed(source: str) -> str:
    with ir.Context():
        return str(ir.Module.parse(source))


def _insert(sizes: str, offsets: str = "0, 0", dims: str = "%n, 128") -> str:
    """An insert of ``%x`` into a ``[%n, 128]`` region of ``%t`` at ``offsets``
    with ``sizes``."""
    return f"""
func.func @f(%t: tensor<64x256xf32>, %x: tensor<?x128xf32>, %i: index, %n: index,
             %m: index) -> tensor<?x128xf32> {{
  %r = tensor.extract_slice %t[%i, 0] [{dims}] [1, 1] : tensor<64x256xf32> to tensor<?x128xf32>
  %y = tensor.insert_slice %x into %r[{offsets}] [{sizes}] [1, 1] : tensor<?x128xf32> into tensor<?x128xf32>
  return %y : tensor<?x128xf32>
}}
"""


def test_fold_full_insert_slices_dynamic_region() -> None:
    text = _apply(_insert("%n, 128"), fold_full_insert_slices)
    assert "tensor.insert_slice" not in text
    assert "return %arg1" in text


def test_fold_full_insert_slices_tensor_dim() -> None:
    source = """
func.func @f(%r: tensor<?x128xf32>, %x: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %c0 = arith.constant 0 : index
  %d = tensor.dim %r, %c0 : tensor<?x128xf32>
  %y = tensor.insert_slice %x into %r[0, 0] [%d, 128] [1, 1] : tensor<?x128xf32> into tensor<?x128xf32>
  return %y : tensor<?x128xf32>
}
"""
    assert "tensor.insert_slice" not in _apply(source, fold_full_insert_slices)


@pytest.mark.parametrize(
    "source",
    [_insert("%m, 128"), _insert("%n, 128", offsets="1, 0")],
    ids=["other_size", "offset"],
)
def test_fold_full_insert_slices_leaves_partial(source: str) -> None:
    assert _apply(source, fold_full_insert_slices) == _parsed(source)


def _copy_of_if(
    extra_use: bool = False,
    target_after: bool = True,
    then_row: str = "%i",
    between: str = "",
) -> str:
    use = (
        "memref.copy %s, %buf : memref<?x128xf32, strided<[?, ?], offset: ?>> to memref<?x128xf32, strided<[?, ?], offset: ?>>"
        if extra_use
        else ""
    )
    target = "%v = memref.subview %out[%i, 0] [%n, 128] [1, 1] : memref<64x256xf32> to memref<?x128xf32, strided<[256, 1], offset: ?>>"
    before, after = ("", target) if target_after else (target, "")
    return f"""
func.func @f(%out: memref<64x256xf32>, %tmp: memref<64x128xf32>,
             %buf: memref<?x128xf32, strided<[?, ?], offset: ?>>, %c: i1, %i: index, %n: index) {{
  {before}
  %s = scf.if %c -> (memref<?x128xf32, strided<[?, ?], offset: ?>>) {{
    %w = memref.subview %out[{then_row}, 0] [%n, 128] [1, 1] : memref<64x256xf32> to memref<?x128xf32, strided<[256, 1], offset: ?>>
    %wc = memref.cast %w : memref<?x128xf32, strided<[256, 1], offset: ?>> to memref<?x128xf32, strided<[?, ?], offset: ?>>
    scf.yield %wc : memref<?x128xf32, strided<[?, ?], offset: ?>>
  }} else {{
    %p = memref.subview %tmp[0, 0] [%n, 128] [1, 1] : memref<64x128xf32> to memref<?x128xf32, strided<[128, 1]>>
    %pc = memref.cast %p : memref<?x128xf32, strided<[128, 1]>> to memref<?x128xf32, strided<[?, ?], offset: ?>>
    scf.yield %pc : memref<?x128xf32, strided<[?, ?], offset: ?>>
  }}
  {after}
  {between}
  memref.copy %s, %v : memref<?x128xf32, strided<[?, ?], offset: ?>> to memref<?x128xf32, strided<[256, 1], offset: ?>>
  {use}
  return
}}
"""


@pytest.mark.parametrize(
    "target_after", [True, False], ids=["subview_after", "target_before"]
)
def test_sink_copies_into_branches(target_after: bool) -> None:
    """The full branch wrote the target in place (an identical subview): no
    copy; the partial branch copies its temporary."""
    text = _apply(_copy_of_if(target_after=target_after), sink_copies_into_branches)
    then, _, rest = text.partition("} else {")
    els, _, tail = rest.partition("}\n")
    assert "memref.copy" not in then
    assert els.count("memref.copy") == 1
    assert "memref.copy" not in tail
    # The subview is before the if.
    assert text.index("memref.subview %arg0") < text.index("scf.if")


def test_sink_copies_into_branches_other_region() -> None:
    """A branch yielding another region of the output (other offsets) keeps
    its copy."""
    text = _apply(_copy_of_if(then_row="%n"), sink_copies_into_branches)
    then, _, rest = text.partition("} else {")
    assert then.count("memref.copy") == 1
    assert rest.count("memref.copy") == 1


_NESTED = """
func.func @f(%out: memref<64x256xf32>, %tmp: memref<64x128xf32>, %c: i1, %d: i1,
             %i: index, %n: index, %w: index) {
  %s = scf.if %c -> (memref<?x?xf32, strided<[?, ?], offset: ?>>) {
    %t = scf.if %d -> (memref<64x?xf32, strided<[?, ?], offset: ?>>) {
      %v = memref.subview %out[%i, 0] [%n, %w] [1, 1] : memref<64x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %vc = memref.cast %v : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<64x?xf32, strided<[?, ?], offset: ?>>
      scf.yield %vc : memref<64x?xf32, strided<[?, ?], offset: ?>>
    } else {
      %p = memref.subview %tmp[0, 0] [64, %w] [1, 1] : memref<64x128xf32> to memref<64x?xf32, strided<[128, 1]>>
      %pc = memref.cast %p : memref<64x?xf32, strided<[128, 1]>> to memref<64x?xf32, strided<[?, ?], offset: ?>>
      scf.yield %pc : memref<64x?xf32, strided<[?, ?], offset: ?>>
    }
    %tc = memref.cast %t : memref<64x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    scf.yield %tc : memref<?x?xf32, strided<[?, ?], offset: ?>>
  } else {
    %q = memref.subview %tmp[0, 0] [%n, %w] [1, 1] : memref<64x128xf32> to memref<?x?xf32, strided<[128, 1]>>
    %qc = memref.cast %q : memref<?x?xf32, strided<[128, 1]>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    scf.yield %qc : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
  %v = memref.subview %out[%i, 0] [%n, %w] [1, 1] : memref<64x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
  memref.copy %s, %v : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
  return
}
"""


def test_sink_copies_into_branches_nested() -> None:
    """A tile versioned along two dims: the copy sinks through the nested
    ``if`` (and the cast of its result); the full-full branch, which wrote the
    target in place (an identical subview), gets no copy, the others one."""
    text = _apply(_NESTED, sink_copies_into_branches)
    assert text.count("memref.copy") == 2
    # Both copies are of the temporary's views, into the hoisted target.
    assert "memref.copy %cast_1, %subview " in text
    assert "memref.copy %cast, %subview " in text
    assert text.index("%subview = memref.subview %arg0") < text.index("scf.if")


_C0 = "%c0 = arith.constant 0 : index"
_TARGET = "memref<?x128xf32, strided<[256, 1], offset: ?>>"


@pytest.mark.parametrize(
    "between",
    [
        f"{_C0}\n  %r = memref.load %v[%c0, %c0] : {_TARGET}",
        (
            f"{_C0}\n  %f = arith.constant 1.0 : f32\n"
            "  memref.store %f, %tmp[%c0, %c0] : memref<64x128xf32>"
        ),
    ],
    ids=["reads_target", "writes_temporary"],
)
def test_sink_copies_into_branches_leaves_effects_between(between: str) -> None:
    """A read or write of a buffer between the ``if`` and its copy: moving the
    copy before it would reorder them, so the copy stays."""
    source = _copy_of_if(target_after=False, between=between)
    assert _apply(source, sink_copies_into_branches) == _parsed(source)


def test_sink_copies_into_branches_pure_between() -> None:
    """Index arithmetic between the ``if`` and its copy does not block it."""
    source = _copy_of_if(target_after=False, between="%z = arith.addi %i, %n : index")
    text = _apply(source, sink_copies_into_branches)
    _, _, rest = text.partition("} else {")
    els, _, tail = rest.partition("}\n")
    assert els.count("memref.copy") == 1
    assert "memref.copy" not in tail


def test_sink_copies_into_branches_leaves_shared_result() -> None:
    source = _copy_of_if(extra_use=True)
    assert _apply(source, sink_copies_into_branches) == _parsed(source)


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[64, 64])
)
def _ragged_store(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@pytest.mark.parametrize("shape", [(130, 70), (128, 64), (20, 200)], ids=str)
def test_versioned_tile_stores_numerics(shape: tuple[int, int]) -> None:
    x = torch.randn(*shape)
    torch.testing.assert_close(_ragged_store(x), x * 2.0, rtol=0, atol=0)


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[64, 128, 64])
)
def _ragged_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = acc + torch.matmul(a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(a.dtype)
    return out


def test_ragged_grid_tiles_write_in_place() -> None:
    """Regression: a grid tile partial along M copied the output region into a
    heap buffer of dynamic size before every tile, full or not, and copied it
    back after (through ``memrefCopy``). Now only the partial version copies."""
    a = torch.randn(130, 128, dtype=torch.bfloat16)
    b = torch.randn(128, 256, dtype=torch.bfloat16)
    module = generate_mlir(_ragged_matmul, (a, b))
    lowered = lower_opt_with_amx(
        module, "_ragged_matmul", lambda stage, _: "hoist_allocas" in stage
    )
    assert lowered is not None
    after = str(lowered[1])
    assert "memref.alloc(%" not in after
    # The one copy is the partial version's, made in its branch: none after
    # the ``if`` (``sink_copies_into_branches``), none in the full branch.
    copies = _ops_named(lowered[1].operation, "memref.copy")
    assert len(copies) == 1
    branch = copies[0].parent
    assert branch.name == "scf.if"
    assert copies[0].operation.block == branch.regions[1].blocks[0]
    out = _ragged_matmul(a, b)
    expected = a.float() @ b.float()
    assert ((out.float() - expected).abs().max() / expected.abs().max()).item() < 1e-2


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 64])
)
def _short_output(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@pytest.mark.isolated
def test_full_tile_of_short_output_stays_in_bounds() -> None:
    """Regression (review of the full-tile cast): an output shorter than the
    grid's range (70 of 100 rows) has regions clamped to its end, shorter than
    a full tile. Cast to the full tile's extent, the store wrote past it (rows
    70-95 of the buffer the output is a view of)."""
    x = torch.randn(100, 64)
    guard = torch.full((200, 64), -7.0)
    out = guard[:70]
    _short_output(x, out)
    assert torch.equal(out, x[:70] * 2.0)
    assert bool((guard[70:] == -7.0).all())
