"""``mark_overwritten_args`` and ``first_touch_outputs``: a fresh output's pages
first touched by each thread in a contiguous slice, before the parallel loop
writing it.

A fresh large output (e.g. glibc ``mmap``s every buffer of 32 MiB or more per
call) is faulted in by the threads that first write it. A loop of tall tiles,
one per thread, has every thread write the same rows at once, so dozens of
threads fault in each 2 MiB range together: the kernel allocates and zeroes a
huge page for each of them and keeps one (``__do_huge_pmd_anonymous_page``
rechecks the PMD only after zeroing), and the 4 KiB pages at the unaligned ends
of the mapping are faulted by up to 16 threads each, which serialize on their
page table's lock. A prologue in the same parallel region makes each thread
fault in its own contiguous share of the output's pages first: a static
worksharing loop over the output's 2 MiB-aligned ranges, one
``madvise(MADV_POPULATE_WRITE)`` call per range, without a barrier. Each 2 MiB
range is faulted (one huge page zeroed) by one thread, and pages already mapped
(e.g. under TCMalloc) cost a page table walk. Smaller chunks contend: the
populate takes the PMD lock of each huge page, and x86's split PMD lock covers
1 GiB, so many threads at chunks under 2 MiB spin on it.

The call accesses no memory: it faults the pages in for writing (contents unchanged;
``EFAULT`` on read-only memory), so it cannot race with the loop's stores and needs no
barrier, which would be costly when the OpenMP workers sleep
(``OMP_WAIT_POLICY=passive``, or a torch op between the kernels). It needs Linux 5.14;
older kernels return ``EINVAL`` and errors are ignored: the call is only a hint. (An
earlier version touched one byte per page with a volatile ``atomicrmw or 0`` and a
barrier.)

It runs only on outputs that every iteration
space of a top-level loop writes whole (``mark_overwritten_args``): each element
is stored to by the function anyway, so the touch adds no page the function
does not write itself (no fault on read-only memory, no copy-on-write after a
fork, no dirtied shared file page that would stay clean). The inserted tile must
be provably written whole, by an allowlist of producers (``_fully_written``:
linalg ops writing all of their init, whose only-forwarded inputs are data or
written whole, slices and whole-slice inserts of such
tiles, ``scf.if`` versions of them), and the old output must not flow into it
(a conditional store yields the old slice, which bufferization leaves in place
without writing); anything else (empty tensors, loop results, reshapes, casts,
unknown ops) is not marked. And only when its slices are narrower than their
rows and span a huge page: threads writing contiguous slices (e.g. whole rows,
as an elementwise kernel does) already first-touch distinct ranges, and the
prologue only slows them.
"""

import math

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import ext
from mlir.dialects import linalg
from mlir.dialects import llvm
from mlir.dialects import memref
from mlir.dialects import openmp as omp
from mlir.dialects import transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from .dialect import HelionTransformDialect
from .dialect import transform_op
from .utils import int_constant
from .utils import payload_ops

# Deepest chain of producers the analyses follow (unknown beyond: unmarked).
_MAX_DEPTH = 200
# Contractions whose reduction length _reduction_length counts.
_CONTRACTIONS = ("linalg.contract", "linalg.matmul", "linalg.batch_matmul")

OVERWRITTEN_ATTR_NAME = "helion.overwritten"
# The base page size of x86-64.
_PAGE = 4096
# A huge (PMD) page of x86-64: one iteration's slice spanning this much while not
# contiguous (narrower than its rows) means many threads write each 2 MiB range.
_HUGE_PAGE = 2 * 1024 * 1024
# Outputs below this are not touched. The prologue makes one madvise call per
# 2 MiB even when the pages are mapped (a page table walk), and glibc reuses
# smaller buffers from its heap more often than not.
MIN_BYTES = 16 * 1024 * 1024
# Tiles not produced by a contraction of at least this total static reduction
# length (K) are not marked. With the pages already mapped (TCMalloc), the
# prologue costs a roughly fixed time per output page, while the GEMM's time per
# output element grows with K: short K pays for it and long K doesn't notice,
# and where pages are fresh (glibc) it pays off at every K. Elementwise ops,
# reductions and norms (no contraction) are memory-bound like a short-K GEMM.
MIN_REDUCTION = 2048
# Views of a memref: their results alias their source.
_VIEW_OPS = (
    "memref.subview",
    "memref.expand_shape",
    "memref.collapse_shape",
    "memref.cast",
    "memref.reinterpret_cast",
    "memref.transpose",
)


def _entry_ops(func: ir.OpView) -> list[ir.OpView]:
    return [op.opview for op in func.regions[0].blocks[0].operations]


def _static_ints(attr: ir.Attribute) -> list[int]:
    return list(ir.DenseI64ArrayAttr(attr))


def _affine_map(value: ir.Value, name: str) -> tuple[ir.AffineMap, list] | None:
    """The map and operands of the ``name`` op defining ``value``, if it is one."""
    if not isinstance(value, ir.OpResult) or value.owner.name != name:
        return None
    owner = value.owner
    return ir.AffineMapAttr(owner.attributes["map"]).value, list(owner.operands)


def _tile_of(offset: ir.Value | int, ivs: list[ir.Value]) -> tuple[int, int] | None:
    """``(iv index, tile)`` if ``offset`` is ``iv * tile`` of a loop IV."""
    if isinstance(offset, int):
        return None
    for index, iv in enumerate(ivs):
        if offset == iv:
            return index, 1
    applied = _affine_map(offset, "affine.apply")
    if applied is None:
        return None
    amap, operands = applied
    if len(operands) != 1 or operands[0] not in ivs:
        return None
    if amap.n_dims != 1 or amap.n_symbols != 0:
        return None
    expr = amap.results[0] if len(amap.results) == 1 else None
    if (
        isinstance(expr, ir.AffineMulExpr)
        and isinstance(expr.lhs, ir.AffineDimExpr)
        and isinstance(expr.rhs, ir.AffineConstantExpr)
        and expr.rhs.value > 0
    ):
        return ivs.index(operands[0]), expr.rhs.value
    return None


def _covers_dim(
    offset: ir.Value | int,
    size: ir.Value | int,
    extent: int,
    ivs: list[ir.Value],
    trips: list[int],
    used: set[int],
) -> bool:
    """Whether slices at ``offset`` of ``size`` over the loop cover ``extent``."""
    if offset == 0 and size == extent:
        return True
    tiled = _tile_of(offset, ivs)
    if tiled is None or tiled[0] in used:
        return False
    index, tile = tiled
    used.add(index)
    if trips[index] * tile < extent:
        return False
    if size == tile:
        return trips[index] * tile == extent
    # A ragged tile: min(tile, extent - iv * tile).
    bounded = _affine_map(size, "affine.min") if not isinstance(size, int) else None
    if bounded is None:
        return False
    amap, operands = bounded
    if operands != [ivs[index]] or amap.n_dims != 1 or amap.n_symbols != 0:
        return False
    d0 = ir.AffineDimExpr.get(0)
    ragged = [d0 * -tile + extent, ir.AffineConstantExpr.get(tile)]
    return amap in (
        ir.AffineMap.get(1, 0, ragged),
        ir.AffineMap.get(1, 0, ragged[::-1]),
    )


def _mixed(static: list[int], dynamic: list[ir.Value]) -> list[ir.Value | int]:
    values = iter(dynamic)
    return [
        next(values) if value == ir.ShapedType.get_dynamic_size() else value
        for value in static
    ]


def _overwritten_slice(op: ir.OpView) -> bool:
    """Whether ``op`` is a ``tensor.extract_slice`` (of the old output) whose
    result is only the destination of ``tensor.insert_slice`` ops writing all of
    it (offset 0, its sizes, unit strides; e.g. a ragged tile's versions): its
    old values never reach the tile."""
    if op.operation.name != "tensor.extract_slice":
        return False
    sizes = _mixed(_static_ints(op.attributes["static_sizes"]), list(op.sizes))
    for use in op.result.uses:
        user = use.owner.opview
        if user.operation.name != "tensor.insert_slice" or user.dest != op.result:
            return False
        if any(_static_ints(user.attributes["static_offsets"])) or list(user.offsets):
            return False
        if any(
            stride != 1 for stride in _static_ints(user.attributes["static_strides"])
        ):
            return False
        user_sizes = _mixed(
            _static_ints(user.attributes["static_sizes"]), list(user.sizes)
        )
        if user_sizes != sizes:
            return False
    return True


# Named linalg ops writing every element of their init without reading it.
_WRITING_OPS = ("linalg.copy", "linalg.fill")


def _writes_all(op: ir.Operation) -> bool:
    """Whether a ``linalg.generic`` writes every element of its single init
    without reading it: all loops parallel, the output map a permutation of
    them, the payload not using the init's element."""
    if op.name in _WRITING_OPS:
        return len(op.results) == 1
    if op.name != "linalg.generic" or len(op.results) != 1:
        return False
    iterators = [str(attr) for attr in ir.ArrayAttr(op.attributes["iterator_types"])]
    if any("parallel" not in iterator for iterator in iterators):
        return False
    maps = [
        ir.AffineMapAttr(attr).value
        for attr in ir.ArrayAttr(op.attributes["indexing_maps"])
    ]
    if not maps[-1].is_permutation or maps[-1].n_dims != len(iterators):
        return False
    payload = op.regions[0].blocks[0]
    return not list(payload.arguments[len(payload.arguments) - 1].uses)


def _forwarded_inputs(op: ir.Operation) -> list[ir.Value]:
    """The inputs a ``linalg`` op only forwards: a ``linalg.copy``'s, or those
    whose element a ``linalg.generic`` payload yields as is (canonicalization
    may replace such an op by its input)."""
    if op.name == "linalg.copy":
        return [op.operands[0]]
    if op.name != "linalg.generic":
        return []
    payload = op.regions[0].blocks[0]
    args = list(payload.arguments)
    inputs = len(op.operands) - len(op.results)
    terminator = payload.operations[len(payload.operations) - 1]
    return [
        op.operands[args.index(value)]
        for value in terminator.operands
        if value in args[:inputs]
    ]


# Views of a tensor: their elements are their source's.
_TENSOR_VIEWS = (
    "tensor.extract_slice",
    "tensor.expand_shape",
    "tensor.collapse_shape",
    "tensor.cast",
)


def _is_data(value: ir.Value) -> bool:
    """Whether ``value`` is a view of the function's data (a ``to_tensor`` or a
    tensor argument of the function), not of an empty tensor."""
    for _ in range(16):
        if isinstance(value, ir.BlockArgument):
            # FuncOp.name is the symbol name: compare the op name.
            return value.owner.owner.operation.name == "func.func"
        name = value.owner.name
        if name == "bufferization.to_tensor":
            return True
        if name not in _TENSOR_VIEWS:
            return False
        value = value.owner.operands[0]
    return False


def _same_sizes(insert: ir.OpView, dest: ir.Value) -> bool:
    """Whether ``tensor.insert_slice`` ``insert`` writes all of ``dest``: at
    offset 0, unit strides, the sizes of ``dest`` (its static shape, or the
    sizes of the slice it is)."""
    if any(_static_ints(insert.attributes["static_offsets"])) or list(insert.offsets):
        return False
    if any(stride != 1 for stride in _static_ints(insert.attributes["static_strides"])):
        return False
    sizes = _mixed(_static_ints(insert.attributes["static_sizes"]), list(insert.sizes))
    dest_type = ir.RankedTensorType(dest.type)
    if dest_type.has_static_shape:
        return sizes == list(dest_type.shape)
    if isinstance(dest, ir.OpResult) and dest.owner.name == "tensor.extract_slice":
        slice_op = dest.owner.opview
        return sizes == _mixed(
            _static_ints(slice_op.attributes["static_sizes"]), list(slice_op.sizes)
        )
    return False


def _fully_written(
    value: ir.Value, depth: int = 0, memo: dict[ir.Value, bool] | None = None
) -> bool:
    """Whether every element of the tile ``value`` is provably written (an
    allowlist; anything else is not):

    - the result of a ``linalg`` op writing all of its init (``_writes_all``);
    - a ``tensor.extract_slice`` or ``tensor.cast`` of such a value;
    - a ``tensor.insert_slice`` of such a value writing all of its destination
      (a ragged tile's versions into the old tile);
    - an ``scf.if`` result whose every branch yields such a value.

    Empty tensors (also cast), loop results, reshapes and unknown ops are not:
    bufferized, parts of them may never be written."""
    if not isinstance(value, ir.OpResult) or depth > _MAX_DEPTH:
        return False
    memo = {} if memo is None else memo
    if value not in memo:
        memo[value] = False  # a cycle (not in valid IR) is not written
        memo[value] = _fully_written_uncached(value, depth, memo)
    return memo[value]


def _fully_written_uncached(value: ir.Value, depth: int, memo: dict) -> bool:
    owner = value.owner
    if _writes_all(owner):
        # An op only forwarding an input (linalg.copy, an identity generic) folds
        # away into it: then that input must be written whole itself.
        return all(
            _fully_written(forwarded, depth + 1, memo) or _is_data(forwarded)
            for forwarded in _forwarded_inputs(owner)
        )
    if owner.name in ("tensor.extract_slice", "tensor.cast"):
        # A slice or a cast (e.g. a full tile's static type to the ragged one,
        # once fold_full_insert_slices replaced the ragged store) of a written tile.
        return _fully_written(owner.operands[0], depth + 1, memo)
    if owner.name == "tensor.insert_slice":
        insert = owner.opview
        return _same_sizes(insert, insert.dest) and _fully_written(
            insert.source, depth + 1, memo
        )
    if owner.name == "scf.if":
        index = list(owner.results).index(value)
        branches = [block for region in owner.regions for block in region.blocks]
        return len(branches) == 2 and all(
            _fully_written(
                block.operations[len(block.operations) - 1].operands[index],
                depth + 1,
                memo,
            )
            for block in branches
        )
    return False


def _contraction_length(op: ir.Operation) -> int | None:
    """The static reduction length of a contraction: the product of the extents
    of the dims its output map does not use (``None`` if one is dynamic)."""
    # Named ops (e.g. linalg.matmul) may leave their default maps implicit.
    indexing_maps = linalg.get_indexing_maps(op.opview)
    if indexing_maps is None:
        return None
    maps = [ir.AffineMapAttr(attr).value for attr in indexing_maps]
    output = {
        expr.position for expr in maps[-1].results if isinstance(expr, ir.AffineDimExpr)
    }
    inputs = list(op.operands)[: len(maps) - 1]
    length = 1
    for dim in range(maps[-1].n_dims):
        if dim in output:
            continue
        extents = [
            ir.RankedTensorType(operand.type).shape[position]
            for operand, amap in zip(inputs, maps, strict=False)
            for position, expr in enumerate(amap.results)
            if isinstance(expr, ir.AffineDimExpr) and expr.position == dim
        ]
        if not extents or extents[0] == ir.ShapedType.get_dynamic_size():
            return None
        length *= extents[0]
    return length


def _reduction_length(value: ir.Value, iter_arg: ir.Value | None = None) -> int | None:
    """The total static reduction length (K) of the contraction producing the
    tile ``value`` (``_ReductionLength``); ``iter_arg``: a K loop's carried
    value, which counts 0. ``None``: not a contraction, or not static. Only
    gates the prologue for speed: ``_fully_written`` gates it for correctness."""
    return _ReductionLength().length(value, iter_arg)


class _ReductionLength:
    """K of tiles, memoised per (value, carried value): an epilogue's inputs
    share producers (e.g. a chain of ``torch.where``), so a plain recursion
    would visit every path through it.

    - Data (a function argument or ``to_tensor``), ``linalg.fill`` and
      ``tensor.empty`` count 0, as does the K loop's carried value.
    - A contraction adds its static reduction extents to its init's.
    - A static ``scf.for`` whose yielded value is accumulated from its carried
      value (``_accumulates``) counts trips x the body's K plus its init's; one
      that does not (e.g. a fresh accumulator each trip) counts one trip's; no
      trips: its init's.
    - An all-parallel op of the epilogue counts the longest K of the inputs its
      payload uses; ``linalg.copy`` and views their source's; ``scf.if`` the
      shortest of its branches'."""

    def __init__(self) -> None:
        self.lengths: dict[tuple[ir.Value, ir.Value | None], int | None] = {}
        self.accumulated: dict[tuple[ir.Value, ir.Value], bool] = {}
        self.depth = 0

    def length(self, value: ir.Value, carried: ir.Value | None) -> int | None:
        key = (value, carried)
        if key not in self.lengths:
            self.lengths[key] = None  # a cycle (not in valid IR) counts as unknown
            if self.depth < _MAX_DEPTH:
                self.depth += 1
                self.lengths[key] = self._length(value, carried)
                self.depth -= 1
        return self.lengths[key]

    def _length(self, value: ir.Value, carried: ir.Value | None) -> int | None:
        if isinstance(value, ir.BlockArgument):
            if carried is not None and value == carried:
                return 0
            return 0 if value.owner.owner.operation.name == "func.func" else None
        owner = value.owner
        name = owner.name
        if name in ("linalg.fill", "tensor.empty", "bufferization.to_tensor"):
            return 0
        if name in _CONTRACTIONS:
            length = _contraction_length(owner)
            base = self.length(list(owner.operands)[len(owner.operands) - 1], carried)
            return None if length is None or base is None else length + base
        if name == "scf.for":
            return self._loop_length(owner, list(owner.results).index(value), carried)
        if name == "linalg.generic" and _writes_all(owner):
            lengths = [self.length(operand, carried) for operand in _used_inputs(owner)]
            known = [length for length in lengths if length is not None]
            if known:
                return max(known)
            return None if lengths else 0
        if name in ("linalg.copy", "tensor.insert_slice", *_TENSOR_VIEWS):
            return self.length(owner.operands[0], carried)
        if name == "scf.if":
            index = list(owner.results).index(value)
            lengths = [
                self.length(_yielded(block, index), carried)
                for region in owner.regions
                for block in region.blocks
            ]
            return None if None in lengths or not lengths else min(lengths)
        return None

    def _loop_length(
        self, loop: ir.Operation, index: int, carried: ir.Value | None
    ) -> int | None:
        lower, upper, step = (
            int_constant(operand) for operand in list(loop.operands)[:3]
        )
        if lower is None or upper is None or not step or step < 0:
            return None
        trips = max(0, -(-(upper - lower) // step))
        init = self.length(list(loop.operands)[3 + index], carried)
        if trips == 0:
            return init
        body = loop.regions[0].blocks[0]
        inner = body.arguments[1 + index]
        yielded = _yielded(body, index)
        per_trip = self.length(yielded, inner)
        if per_trip is None:
            return None
        if not self.accumulates(yielded, inner):
            return per_trip
        return None if init is None else trips * per_trip + init

    def accumulates(self, value: ir.Value, carried: ir.Value) -> bool:
        """Whether ``value`` is accumulated from ``carried``: reached through
        contraction inits, the epilogue ops' used inputs, copies, views, both
        ``scf.if`` branches and the inits of inner loops."""
        key = (value, carried)
        if key not in self.accumulated:
            self.accumulated[key] = False
            if self.depth < _MAX_DEPTH:
                self.depth += 1
                self.accumulated[key] = self._accumulates(value, carried)
                self.depth -= 1
        return self.accumulated[key]

    def _accumulates(self, value: ir.Value, carried: ir.Value) -> bool:
        if isinstance(value, ir.BlockArgument):
            return value == carried
        owner = value.owner
        name = owner.name
        if name in _CONTRACTIONS:
            return self.accumulates(
                list(owner.operands)[len(owner.operands) - 1], carried
            )
        if name == "scf.for":
            index = list(owner.results).index(value)
            return self.accumulates(list(owner.operands)[3 + index], carried)
        if name == "linalg.generic" and _writes_all(owner):
            return any(
                self.accumulates(operand, carried) for operand in _used_inputs(owner)
            )
        if name in ("linalg.copy", "tensor.insert_slice", *_TENSOR_VIEWS):
            return self.accumulates(owner.operands[0], carried)
        if name == "scf.if":
            index = list(owner.results).index(value)
            return all(
                self.accumulates(_yielded(block, index), carried)
                for region in owner.regions
                for block in region.blocks
            )
        return False


def _yielded(block: ir.Block, index: int) -> ir.Value:
    return block.operations[len(block.operations) - 1].operands[index]


def _used_inputs(op: ir.Operation) -> list[ir.Value]:
    """The inputs of a ``linalg.generic`` whose element its payload uses."""
    payload = op.regions[0].blocks[0]
    inputs = list(op.operands)[: len(op.operands) - len(op.results)]
    return [
        operand
        for operand, argument in zip(inputs, payload.arguments, strict=False)
        if list(argument.uses)
    ]


def _covers(forall: ir.OpView, out_index: int) -> bool:
    """Whether the iterations of ``forall`` write all of shared output
    ``out_index``, in the pattern the prologue helps (``_scattered``): one
    ``parallel_insert_slice`` into it, of unit strides, whose offsets are
    ``iv * tile`` (or 0 for whole dims) and whose sizes are the tile (ragged at
    the end: ``min(tile, extent - iv * tile)``), every IV in one dim."""
    lower = _static_ints(forall.attributes["staticLowerBound"])
    trips = _static_ints(forall.attributes["staticUpperBound"])
    steps = _static_ints(forall.attributes["staticStep"])
    if (
        len(forall.operands) != len(forall.results)  # bounds of runtime values
        or any(lower)
        or any(step != 1 for step in steps)
    ):
        return False
    block = forall.regions[0].blocks[0]
    ivs = list(block.arguments)[: len(trips)]
    out = block.arguments[len(trips) + out_index]
    terminator = block.operations[len(block.operations) - 1]
    inserts = [
        op.opview
        for op in terminator.regions[0].blocks[0].operations
        if op.name == "tensor.parallel_insert_slice" and op.opview.dest == out
    ]
    if len(inserts) != 1:
        return False
    insert = inserts[0]
    # Elements without a bit width (complex, index): not handled.
    width = getattr(ir.RankedTensorType(out.type).element_type, "width", 0) // 8
    if width == 0:
        return False
    # The old output must not flow into the tile: a conditional store yields the
    # old slice, which bufferization leaves in place, writing nothing (the touch
    # would then write pages the function never writes: a read-only mapping
    # faults, a forked copy or a shared file mapping gets dirtied). Only slices
    # it overwrites whole as destinations are allowed. An empty tensor inserted
    # is the same: empty-tensor elimination writes nothing.
    if not all(
        use.owner == insert.operation or _overwritten_slice(use.owner.opview)
        for use in out.uses
    ):
        return False
    if not _fully_written(insert.source):
        return False
    # Only tiles of long contractions: a memory-bound kernel (a short-K GEMM, an
    # elementwise op) pays the prologue's pass over the pages for nothing when
    # the pages are already mapped (TCMalloc), more than its faults cost.
    length = _reduction_length(insert.source)
    if length is None or length < MIN_REDUCTION:
        return False
    shape = ir.RankedTensorType(out.type).shape
    if any(stride != 1 for stride in _static_ints(insert.attributes["static_strides"])):
        return False
    offsets = _mixed(
        _static_ints(insert.attributes["static_offsets"]), list(insert.offsets)
    )
    sizes = _mixed(_static_ints(insert.attributes["static_sizes"]), list(insert.sizes))
    used: set[int] = set()
    tiles: list[int] = []
    for offset, size, extent in zip(offsets, sizes, shape, strict=True):
        if not _covers_dim(offset, size, extent, ivs, trips, used):
            return False
        tiled = _tile_of(offset, ivs)
        tiles.append(size if tiled is None else tiled[1])
    return _scattered(tiles, shape, width)


def _scattered(tiles: list[int], shape: list[int], width: int) -> bool:
    """Whether a slice of ``tiles`` (full-size) of a row-major ``shape`` of
    ``width``-byte elements is not contiguous (narrower than the rows it spans)
    and spans at least a huge page: then the threads writing their slices at
    once all write in the same 2 MiB ranges. A contiguous slice (e.g. whole
    rows) or a short one is first-touched by few threads per range already."""
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    # Contiguous: every dim after the first one of more than one element is whole.
    first = next((d for d, tile in enumerate(tiles) if tile > 1), len(shape))
    contiguous = all(tiles[d] == shape[d] for d in range(first + 1, len(shape)))
    span = (
        sum((tile - 1) * stride for tile, stride in zip(tiles, strides, strict=True))
        + 1
    ) * width
    return not contiguous and span >= _HUGE_PAGE


def _overwritten_args(func: ir.OpView) -> set[int]:
    """Indices of ``func``'s memref arguments a top-level ``scf.forall`` writes
    whole in place: its shared output is a writable ``to_tensor`` of the
    argument and its result is materialized into it."""
    args = list(func.regions[0].blocks[0].arguments)
    found: set[int] = set()
    ops = _entry_ops(func)
    for forall in (op for op in ops if op.operation.name == "scf.forall"):
        inits = list(forall.operands)[len(forall.operands) - len(forall.results) :]
        for index, (init, result) in enumerate(zip(inits, forall.results, strict=True)):
            if not isinstance(init, ir.OpResult) or init.owner.name != (
                "bufferization.to_tensor"
            ):
                continue
            source = init.owner.operands[0]
            if source not in args or "writable" not in init.owner.attributes:
                continue
            if len(list(init.uses)) != 1:  # read or written elsewhere too
                continue
            stores = [
                use.owner
                for use in result.uses
                if use.owner.name == "bufferization.materialize_in_destination"
                and use.owner.operands[1] == source
                and use.owner.operation.block == forall.operation.block
            ]
            # The argument used only here and by its store: another view of it
            # could bring old values back or write it elsewhere.
            others = [
                use.owner
                for use in source.uses
                if use.owner != init.owner and use.owner not in stores
            ]
            if stores and not others and _covers(forall, index):
                found.add(args.index(source))
    return found


@transform_op(modifies_payload=True)
class MarkOverwrittenArgsOp(
    HelionTransformDialect.Operation, name="mark_overwritten_args"
):
    """Mark with ``OVERWRITTEN_ATTR_NAME`` each memref argument of the target
    functions that a top-level ``scf.forall`` overwrites whole in slices that are
    narrower than their rows and span a huge page (``_covers``), for
    ``first_touch_outputs``. Runs on the IR before tiling."""

    target: ext.Operand[transform.AnyOpType]

    @staticmethod
    def run(
        op: "MarkOverwrittenArgsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        for func in payload_ops(state, op.target, ir.OpView):
            if func.operation.name != "func.func" or not func.regions[0].blocks:
                continue
            indices = _overwritten_args(func)
            if not indices:
                continue
            count = len(func.regions[0].blocks[0].arguments)
            attrs = func.attributes.get("arg_attrs", None)
            dicts = [
                {named.name: named.attr for named in ir.DictAttr(attrs[i])}
                if attrs is not None
                else {}
                for i in range(count)
            ]
            for index in indices:
                dicts[index][OVERWRITTEN_ATTR_NAME] = ir.UnitAttr.get()
            func.attributes["arg_attrs"] = ir.ArrayAttr.get(
                [ir.DictAttr.get(named) for named in dicts]
            )
        return DiagnosedSilenceableFailure.Success


def mark_overwritten_args(target: ir.Value) -> MarkOverwrittenArgsOp:
    """snake_case wrapper to create a MarkOverwrittenArgsOp."""
    return MarkOverwrittenArgsOp(target=target)


def _contiguous_bytes(memref_type: ir.Type) -> int | None:
    """Bytes of a static, row-major contiguous memref of byte-sized elements."""
    if not isinstance(memref_type, ir.MemRefType):
        return None
    if not memref_type.has_static_shape or memref_type.memory_space is not None:
        return None
    width = getattr(memref_type.element_type, "width", None)
    if width is None or width % 8:
        return None
    strides, offset = memref_type.get_strides_and_offset()
    shape = memref_type.shape
    expected = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    if offset != 0 or strides != expected:
        return None
    return math.prod(shape) * width // 8


def _uses_any(op: ir.OpView, values: list[ir.Value]) -> bool:
    found = False

    def visit(nested: ir.Operation) -> ir.WalkResult:
        nonlocal found
        if any(operand in values for operand in nested.operands):
            found = True
            return ir.WalkResult.INTERRUPT
        return ir.WalkResult.ADVANCE

    op.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return found


def _first_user(func: ir.OpView, arg: ir.Value) -> ir.OpView | None:
    """The first op of ``func``'s entry block using ``arg`` (through views)."""
    aliases = [arg]
    for op in _entry_ops(func):
        if op.operation.name in _VIEW_OPS and op.operands and op.operands[0] in aliases:
            aliases.extend(op.results)
            continue
        if _uses_any(op, aliases):
            return op
    return None


def _index(value: int) -> ir.Value:
    return arith.ConstantOp(ir.IndexType.get(), value).result


# madvise(2) advice that faults a range in for writing without accessing it
# (Linux 5.14 and later; older kernels return EINVAL, which is ignored).
MADV_POPULATE_WRITE = 23
# libc's madvise, which the JIT resolves in the process. (Not omp_get_thread_num
# and friends: with torch's libgomp loaded first, those names bind to libgomp,
# which inside a libomp team reports one thread: every thread then populated
# the whole output. The worksharing loop goes through
# libomp's own __kmpc entry points.)
_MADVISE = "llvm.func @madvise(!llvm.ptr, i64, i32) -> i32"


def _declare_madvise(module: ir.Operation) -> None:
    """Declare ``madvise`` in ``module`` unless already there."""
    body = module.regions[0].blocks[0]
    declared = {
        ir.StringAttr(op.attributes["sym_name"]).value
        for op in body.operations
        if "sym_name" in op.attributes
    }
    if "madvise" not in declared:
        ir.InsertionPoint(body).insert(ir.Operation.parse(_MADVISE))


def _populate(arg: ir.Value, size: int, advice: int) -> None:
    """A static worksharing loop without a barrier (``nowait``) over the 2 MiB-
    aligned ranges the ``size`` bytes at ``arg``'s aligned pointer overlap: each
    iteration ``madvise(advice)``s its range clipped to the output's pages
    (from the page holding its first byte to its end), each thread consecutive
    ranges. The result is ignored: the call only faults pages in."""
    i64 = ir.IntegerType.get_signless(64)
    base = memref.ExtractAlignedPointerAsIndexOp(arg).result
    first = arith.AndIOp(base, _index(-_PAGE)).result
    start = arith.AndIOp(base, _index(-_HUGE_PAGE)).result
    end = arith.AddIOp(base, _index(size)).result
    span = arith.SubIOp(end, start).result
    count = arith.DivUIOp(
        arith.AddIOp(span, _index(_HUGE_PAGE - 1)).result, _index(_HUGE_PAGE)
    )
    # A loop wrapper's region holds only its loop nest: bounds are defined before.
    lower, unit = _index(0), _index(1)
    advice_value = arith.ConstantOp(ir.IntegerType.get_signless(32), advice).result
    loop = omp.WsloopOp(
        [],
        [],
        [],
        [],
        [],
        [],
        nowait=True,
        schedule_kind=ir.Attribute.parse("#omp.schedulekind<static>"),
    )
    with ir.InsertionPoint(loop.regions[0].blocks.append()):
        nest = omp.LoopNestOp([lower], [count.result], [unit])
        block = nest.regions[0].blocks.append(ir.IndexType.get())
        with ir.InsertionPoint(block):
            offset = arith.MulIOp(block.arguments[0], _index(_HUGE_PAGE)).result
            aligned = arith.AddIOp(start, offset).result
            address = arith.MaxUIOp(aligned, first).result
            limit = arith.AddIOp(aligned, _index(_HUGE_PAGE)).result
            stop = arith.MinUIOp(limit, end).result
            # madvise rounds the length up to a page: the last one is the output's.
            length = arith.SubIOp(stop, address).result
            pointer = llvm.IntToPtrOp(
                llvm.PointerType.get(), arith.IndexCastOp(i64, address).result
            ).result
            llvm.CallOp(
                ir.IntegerType.get_signless(32),
                [pointer, arith.IndexCastOp(i64, length).result, advice_value],
                [],
                [],
                callee="madvise",
            )
            omp.YieldOp([])


@transform_op(modifies_payload=True)
class FirstTouchOutputsOp(HelionTransformDialect.Operation, name="first_touch_outputs"):
    """For each argument marked by ``mark_overwritten_args`` that is a static,
    contiguous memref of at least ``min_bytes``, whose first user in the entry
    block of its function is an ``omp.parallel``: at the start of that region,
    each thread ``madvise``s its contiguous share of the output's pages with
    ``advice`` (``MADV_POPULATE_WRITE``; see the module docstring), one call per
    2 MiB-aligned range in a static worksharing loop without a barrier
    (``nowait``): the call accesses no memory, so it cannot race with the loop's
    stores.
    Other IR is left untouched."""

    target: ext.Operand[transform.AnyOpType]
    min_bytes: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), MIN_BYTES
        )
    )
    advice: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), MADV_POPULATE_WRITE
        )
    )

    @staticmethod
    def run(
        op: "FirstTouchOutputsOp",
        _rewriter: transform.TransformRewriter,
        _results: transform.TransformResults,
        state: transform.TransformState,
    ) -> DiagnosedSilenceableFailure:
        min_bytes = ir.IntegerAttr(op.min_bytes).value
        advice = ir.IntegerAttr(op.advice).value
        for func in payload_ops(state, op.target, ir.OpView):
            if func.operation.name != "func.func" or not func.regions[0].blocks:
                continue
            if "arg_attrs" not in func.attributes:
                continue
            attrs = func.attributes["arg_attrs"]
            # Outputs per region, populated at its start.
            regions: dict[ir.Operation, list[tuple[ir.Value, int]]] = {}
            for index, arg in enumerate(func.regions[0].blocks[0].arguments):
                if OVERWRITTEN_ATTR_NAME not in ir.DictAttr(attrs[index]):
                    continue
                size = _contiguous_bytes(arg.type)
                if size is None or size <= 0 or size < min_bytes:
                    continue
                user = _first_user(func, arg)
                if user is None or user.operation.name != "omp.parallel":
                    continue
                regions.setdefault(user.operation, []).append((arg, size))
            if regions:
                _declare_madvise(func.operation.parent)
            for region, outputs in regions.items():
                body = region.regions[0].blocks[0]
                with ir.InsertionPoint.at_block_begin(body), region.location:
                    for arg, size in outputs:
                        _populate(arg, size, advice)
            # The marks are used up: an output not touched lowers as without them.
            stripped = ir.ArrayAttr.get(
                [
                    ir.DictAttr.get(
                        {
                            named.name: named.attr
                            for named in ir.DictAttr(attrs[index])
                            if named.name != OVERWRITTEN_ATTR_NAME
                        }
                    )
                    for index in range(len(attrs))
                ]
            )
            if any(len(ir.DictAttr(named)) for named in stripped):
                func.attributes["arg_attrs"] = stripped
            else:
                del func.attributes["arg_attrs"]
        return DiagnosedSilenceableFailure.Success


def first_touch_outputs(
    target: ir.Value, min_bytes: int = MIN_BYTES, advice: int = MADV_POPULATE_WRITE
) -> FirstTouchOutputsOp:
    """snake_case wrapper to create a FirstTouchOutputsOp."""
    i64 = ir.IntegerType.get_signless(64)
    return FirstTouchOutputsOp(
        target=target,
        min_bytes=ir.IntegerAttr.get(i64, min_bytes),
        advice=ir.IntegerAttr.get(i64, advice),
    )
