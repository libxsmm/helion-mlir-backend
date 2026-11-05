"""A pool of output buffers for CPU kernels: fresh tensors reused without page faults.

Why: a large buffer is new memory on every call unless the allocator keeps it.
glibc mmaps what is over 32 MiB per call, and two smaller buffers per call trim
its heap and refault. Threads writing tiles of one huge page make those faults
far dearer than the zeroing. TCMalloc keeps its memory, so it never pays this.

What: :func:`empty` and :func:`empty_like` are ``torch.empty`` and
``torch.empty_like`` from mappings of our own, for ``MIN_BYTES`` or more. A block
returns when
the last reference to the tensor's storage dies, and is reused for the same size
(whole 4 KiB pages; whole 2 MiB-aligned huge pages from 2 MiB up, so no 4 KiB
pages at its ends, the ones many threads fault at once). Contents are unspecified,
as ``torch.empty``'s. Kernels' host code calls them directly; the native call path
gets its fresh tensors through :func:`_native_allocate` (``native/call.cpp`` has
only that seam, and knows nothing of this module's policy).

Limits: at most ``max_bytes`` and ``_MAX_FREE_BLOCKS`` free blocks are kept, oldest out
first: :func:`set_output_pool`, ``HELION_MLIR_OUTPUT_POOL`` (bytes, read at import; 0
turns it off), by default the smaller of ``DEFAULT_MAX_BYTES`` and 1/8 of the memory
(RAM or the cgroup's limit). Kernels need 8-256 MiB free, in 1-2 blocks, a few up to 1.5
GiB. Two sizes that do not fit together thrash, oldest out first: [1536, 1024] MiB at 2
GiB has no hits; no spec model does this. When a mapping fails, every free block goes
and it is tried again; a forked child starts with none. :func:`release` frees them on
request and :func:`output_pool_stats` shows them.

Everything else gets torch's function: other devices, sizes or argument forms,
wrapper tensors, active modes and transforms (:func:`_plain_dispatch`), and
deterministic mode, which fills empty tensors.

The storage hook: ``torch.frombuffer`` keeps the object it is given until the
storage dies, then drops it, holding the GIL. So each tensor gets a throwaway
ctypes array over its block, whose ``__del__`` returns the block. The tensor is
shaped with ``resize_`` (no view: same as ``torch.empty``'s, not ``_is_view()``),
but its storage cannot grow: ``resize_`` or ``resize_as_`` beyond the block, or an
``out=`` argument that would grow the tensor, raises, as on a ``from_numpy`` tensor.

Nothing here waits on another allocation on this thread: blocks may return or be
asked for inside the garbage collector, a finalizer or a signal handler, so
:func:`_take` and :func:`_return_block` only try the lock (failing, it is a miss, or
the block waits in ``_returned``). A signal handler calling :func:`release` or the
limit functions can still wait on the lock its own thread holds.
"""

from __future__ import annotations

from collections import OrderedDict
from collections import deque
import contextlib
import ctypes
import math
import mmap
import os
from pathlib import Path
import threading
import warnings

import torch

OUTPUT_POOL_ENV = "HELION_MLIR_OUTPUT_POOL"
# Below this torch allocates: a pool hit is a visible share of a small GEMM, and a
# single buffer this small stays on glibc's heap unfaulted anyway. Two smaller buffers
# per call still refault under glibc: kernels overwrite their input instead where they
# can.
MIN_BYTES = 4 * 1024 * 1024
_PAGE = 4096
_HUGE_PAGE = 2 * 1024 * 1024
DEFAULT_MAX_BYTES = 2 << 30
_MAX_FREE_BLOCKS = 64
_CPU = torch.device("cpu")
# What torch.frombuffer and resize_ were checked on.
_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }
)


class _Block:
    """A mapping of ours, ``length`` bytes at ``address``."""

    __slots__ = ("address", "length", "lease", "mapping")

    def __init__(self, length: int) -> None:
        huge = length >= _HUGE_PAGE
        padding = _HUGE_PAGE if huge else 0
        self.mapping = mmap.mmap(
            -1, length + padding, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
        self.address = -(-base // _HUGE_PAGE) * _HUGE_PAGE if huge else base
        self.length = length
        if huge:  # Huge pages also where THP is madvise-only. Only a hint.
            # Refused, or no such advice here.
            with contextlib.suppress(OSError, AttributeError):
                self.mapping.madvise(mmap.MADV_HUGEPAGE, self.address - base, length)
        # The class of this block's throwaway arrays: its __del__ knows the block.
        self.lease = type(
            "_Lease",
            (ctypes.c_char * length,),
            {"__slots__": (), "__del__": _return_block, "block": self},
        )

    def close(self) -> None:
        self.lease = None  # breaks the block <-> class cycle
        self.mapping.close()


def _return_block(lease: ctypes.Array) -> None:
    # Runs wherever the storage dies, possibly inside the garbage collector during
    # _take on this very thread: only an atomic append, then a try at the lock.
    _returned.append(lease.block)
    _settle()


# Returned blocks wait here for the lock's holder.
_returned: deque[_Block] = deque()
_lock = threading.Lock()
# Oldest first; by address. _by_length: the same blocks' addresses, per length.
_free: OrderedDict[int, _Block] = OrderedDict()
_by_length: dict[int, list[int]] = {}
_free_bytes = 0
_max_bytes = 0
_hits = 0
_misses = 0


def _settle_locked() -> None:
    """File returned blocks; evict what is over the limits."""
    global _free_bytes
    while _returned:
        block = _returned.popleft()
        if block.length > _max_bytes:  # alone over the limit: nothing else need go
            block.close()
            continue
        _free[block.address] = block
        _by_length.setdefault(block.length, []).append(block.address)
        _free_bytes += block.length
    while _free and (_free_bytes > _max_bytes or len(_free) > _MAX_FREE_BLOCKS):
        address, block = _free.popitem(last=False)
        _by_length[block.length].remove(address)
        _free_bytes -= block.length
        block.close()


def _release_locked() -> None:
    global _free_bytes
    for block in _free.values():
        block.close()
    _free.clear()
    _by_length.clear()
    _free_bytes = 0


def _settle() -> None:
    # Never waits: the holder files the block on its next pass, or the next call does.
    if _lock.acquire(blocking=False):
        try:
            _settle_locked()
        finally:
            _lock.release()


def _take(length: int) -> _Block:
    global _free_bytes, _hits, _misses
    if _lock.acquire(blocking=False):
        try:
            _settle_locked()
            addresses = _by_length.get(length)
            if addresses:
                block = _free.pop(addresses.pop())
                _free_bytes -= length
                _hits += 1
                return block
            _misses += 1
        finally:
            _lock.release()
    # A miss, also when the lock is taken: it may be this thread's, under us.
    try:
        return _Block(length)
    except OSError:  # out of memory or address space: what we keep goes first
        if _lock.acquire(blocking=False):
            try:
                _settle_locked()
                _release_locked()
            finally:
                _lock.release()
        return _Block(length)


def _allocate(sizes: tuple[int, ...], count: int, dtype: torch.dtype) -> torch.Tensor:
    size = dtype.itemsize * count
    unit = _HUGE_PAGE if size >= _HUGE_PAGE else _PAGE
    block = _take(-(-size // unit) * unit)
    lease = block.lease.from_address(block.address)
    # frombuffer keeps the lease until the storage dies. The shape set in place, as
    # torch.empty's: not a view of a flat tensor.
    return torch.frombuffer(lease, dtype=dtype, count=count).resize_(sizes)


def _plain_dispatch() -> bool:
    """No torch-function or dispatch mode, no transform and no trace (a traced
    allocation becomes a constant: two calls would share its memory): then
    ``torch.empty`` would dispatch as ``frombuffer`` does."""
    return not (
        torch._C._is_torch_function_mode_enabled()
        or torch._C._len_torch_dispatch_stack()
        or torch._C._functorch.peek_interpreter_stack() is not None
        or torch.jit.is_tracing()
    )


def _deterministic_fill() -> bool:
    return (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    )


def _takes(count: int, dtype: torch.dtype) -> bool:
    # Not what it could not keep: it would be mapped and unmapped per call.
    return (
        dtype in _DTYPES
        and MIN_BYTES <= dtype.itemsize * count <= _max_bytes
        and not _deterministic_fill()
    )


def _native_allocate(sizes: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor | None:
    """The native call path's allocator, for a fresh tensor of ``MIN_BYTES`` or
    more; ``None``: use ``at::empty``. It checked the dispatch and the sizes."""
    count = math.prod(sizes)
    if not _takes(count, dtype) or torch.jit.is_tracing():
        return None
    try:
        return _allocate(sizes, count, dtype)
    except OSError:
        return None


def _default_strides(sizes: tuple[int, ...]) -> tuple[int, ...]:
    """A new tensor's strides. ``empty_like`` keeps other ones of a contiguous
    tensor, in size-1 dims: those it leaves to torch."""
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * max(sizes[i + 1], 1)
    return tuple(strides)


def empty(
    sizes: tuple[int, ...] | list[int], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """``torch.empty(sizes, dtype=dtype, device=device)``, from the pool if it
    takes the tensor."""
    if device == _CPU and all(type(size) is int and size >= 0 for size in sizes):
        count = math.prod(sizes)
        if _takes(count, dtype) and _plain_dispatch():
            try:
                return _allocate(tuple(sizes), count, dtype)
            except OSError:
                pass  # torch's own attempt raises its error
    return torch.empty(sizes, dtype=dtype, device=device)


def empty_like(x: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """``torch.empty_like(x, dtype=dtype)``, from the pool if it takes the tensor:
    a plain contiguous CPU tensor's result is as ``torch.empty``'s."""
    kind = x.dtype if dtype is None else dtype
    if (
        not torch.jit.is_tracing()  # x.shape would be traced tensors
        and type(x) is torch.Tensor
        and x.device == _CPU
        and x.layout == torch.strided
        and not (x.is_conj() or x.is_neg() or x.is_quantized)
        and x.is_contiguous()
        and x.stride() == _default_strides(x.shape)
    ):
        count = x.numel()
        if _takes(count, kind) and _plain_dispatch():
            try:
                return _allocate(tuple(x.shape), count, kind)
            except OSError:
                pass  # torch's own attempt raises its error
    return torch.empty_like(x, dtype=kind)


def set_output_pool(max_bytes: int) -> None:
    """Keep at most ``max_bytes`` of free blocks; 0 turns the pool off and frees
    them. Tensors using a block are unaffected."""
    global _max_bytes
    with _lock:
        _max_bytes = max(0, int(max_bytes))
        _settle_locked()
    from . import native  # imports this module

    native.refresh_allocator()


def release() -> None:
    """Free every free block now; tensors using a block are unaffected."""
    with _lock:
        _settle_locked()
        _release_locked()


def output_pool_stats() -> dict[str, int]:
    with _lock:
        _settle_locked()
        return {
            "hits": _hits,
            "misses": _misses,
            "free_blocks": len(_free),
            "free_bytes": _free_bytes,
            "max_bytes": _max_bytes,
        }


def enabled() -> bool:
    return _max_bytes > 0


def _after_fork_in_child() -> None:
    # The child's copies of the free blocks are its own and nobody will reuse them
    # soon: it starts empty. Blocks of tensors it inherited return as usual.
    _returned.clear()
    _release_locked()
    _lock.release()


# A fork holds the lock over the copy, so a child sees the pool whole and unlocked,
# whatever other threads were doing.
os.register_at_fork(
    before=_lock.acquire,
    after_in_parent=_lock.release,
    after_in_child=_after_fork_in_child,
)


def _cgroup_limit() -> int | None:
    """The tightest memory limit of this process's cgroups (v2 and v1), if any."""
    limits = []
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            root, name = "/sys/fs/cgroup", "memory.max"
        elif "memory" in controllers.split(","):
            root, name = "/sys/fs/cgroup/memory", "memory.limit_in_bytes"
        else:
            continue
        # A process may see only part of the hierarchy: read what exists.
        parts = path.strip("/").split("/") if path.strip("/") else []
        for depth in range(len(parts), -1, -1):
            # No such file, or "max".
            with contextlib.suppress(OSError, ValueError):
                limits.append(int(Path(root, *parts[:depth], name).read_text()))
    return min(limits) if limits else None


def _default_max_bytes() -> int:
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = _cgroup_limit()
    memory = ram if limit is None else min(ram, limit)
    default = min(DEFAULT_MAX_BYTES, memory // 8)
    value = os.environ.get(OUTPUT_POOL_ENV, "").strip()
    if not value:
        return default
    try:
        chosen = int(value)
        if chosen < 0:
            raise ValueError
    except ValueError:
        warnings.warn(
            f"{OUTPUT_POOL_ENV}={value!r} is not a number of bytes: using {default}",
            stacklevel=2,
        )
        return default
    return chosen


_max_bytes = _default_max_bytes()
