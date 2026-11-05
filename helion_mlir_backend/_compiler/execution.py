"""Lower a generated module with lighthouse and JIT its memref-ABI entry function.

:func:`compile_entry` returns a :class:`CompiledEntry` that takes the entry's
arguments in order (tensors for memrefs, Python numbers for runtime scalars,
passed by value); :func:`lower_entry` only lowers, possibly to a stage before
the end of the target's pipeline (IR tests). The device target
(``targets/``) supplies the lighthouse lowering and the JIT runner. Failures keep
their exception type and get a note naming the stage. Lighthouse lowering runs in a forked process, killed after
the compile timeout; only the lowered module comes back. Compiled entries are
cached in-process by module text, target and pipeline, so equal modules (e.g.
configs that clamp to the same tiles) JIT once.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import functools
import gc
import hashlib
import io
import logging
import math
import multiprocessing
import operator
import os
import pickle
import signal
import tempfile
import threading
from typing import TYPE_CHECKING
import warnings

import mlir.ir as ir
from mlir.passmanager import PassManager
import sympy
import torch

from helion_mlir_backend._compiler import native
from helion_mlir_backend._compiler.mlir.in_place import update_carried_values_in_place
from helion_mlir_backend._compiler.mlir.support.debug import LOWERING_LOG_ATTR
from helion_mlir_backend._compiler.mlir.support.debug import LOWERING_WARNINGS_ATTR
from helion_mlir_backend._compiler.mlir.support.debug import DebugOptions
from helion_mlir_backend._compiler.mlir.support.debug import compile_timeout
from helion_mlir_backend._compiler.mlir.support.errors import CompileTimeoutError
from helion_mlir_backend._compiler.mlir.support.errors import LoweringStoppedError
from helion_mlir_backend._compiler.mlir.support.type_utils import mlir_dtype_to_torch
from helion_mlir_backend._compiler.mlir_transforms.prints import warn_on_each_compile
from helion_mlir_backend._compiler.targets import resolve_target
from helion_mlir_backend._compiler.targets.base import MLIRTarget

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence
    from multiprocessing.connection import Connection
    from typing import BinaryIO

    from lighthouse.execution.runner import Runner

    from .mlir.analysis.signature import ReadOrder
    from .mlir.analysis.signature import ScalarIndex
    from helion_mlir_backend._compiler.targets.base import Lowering

log = logging.getLogger(__name__)

_JIT_CACHE: OrderedDict[tuple[str, str, str, bool, str, str], CompiledEntry] = (
    OrderedDict()
)
_JIT_CACHE_SIZE = 128


def _dump_if(enabled: bool, label: str, module: ir.Module) -> None:
    if enabled:
        print(f"=== {label} ===", flush=True)
        print(module, flush=True)


def inline_module(module: ir.Module) -> ir.Module:
    """Inline phase and helper functions into the entry and update loop-carried
    values in place."""
    with module.context, ir.Location.unknown():
        PassManager.parse("builtin.module(inline,canonicalize)").run(module.operation)
        update_carried_values_in_place(module)
        PassManager.parse("builtin.module(canonicalize)").run(module.operation)
    return module


@dataclass(frozen=True)
class EntryArg:
    """One argument of the entry function, from its ``helion.*`` argument attributes."""

    name: str
    """The host expression that produces the argument."""
    role: str
    """``in``, ``inout`` or ``scalar``."""
    tensor_param: int | None
    shape: tuple[int, ...]
    """``-1`` for a size only known at run time; ``()`` for a scalar."""
    dtype: torch.dtype


_LAMBDIFY = [
    {
        "FloorDiv": operator.floordiv,
        "CleanDiv": operator.floordiv,
        "PythonMod": operator.mod,
        "Mod": operator.mod,
    },
    "math",
]
"""What a runtime position's ``sympy`` expression may use."""


@dataclass(frozen=True)
class CompiledEntry:
    name: str
    args: tuple[EntryArg, ...]
    runner: Runner
    target: MLIRTarget
    """The target compiled for: its checks and hooks apply to every call."""
    read_order: ReadOrder | None = None
    """What its reads see (``None``: no host code known, inputs as at the call)."""
    scalar_indices: tuple[ScalarIndex, ...] = ()
    """Its accesses at runtime positions, which each call checks."""
    scalar_exprs: tuple[tuple[object, str], ...] = ()
    """Each runtime scalar's ``expr`` and argument name, for those positions."""

    def __post_init__(self) -> None:
        # The ctypes callable of the packed C interface, looked up once.
        object.__setattr__(self, "_func", self.runner.engine.lookup(self.name))
        object.__setattr__(self, "_static", _StaticCall.build(self.args))
        object.__setattr__(self, "_native", self._native_call())
        object.__setattr__(self, "arg_names", tuple(arg.name for arg in self.args))
        object.__setattr__(self, "index_checks", self._index_checks())

    def _index_checks(self) -> tuple[tuple, ...]:
        """Per access at a runtime position, without repeats: ``(tensor, dim, size, ...)``
        with the position as the name of the scalar argument it is, or as a function
        of the named ones: compiled once, so a call pays no ``sympy``."""
        names = {expr: name for expr, name in self.scalar_exprs if expr is not None}
        checks: dict[tuple, tuple] = {}
        for access in self.scalar_indices:
            if access.expr in names:
                position, operands = names[access.expr], None
            elif access.expr.free_symbols <= names.keys():
                symbols = sorted(access.expr.free_symbols, key=str)
                position = sympy.lambdify(symbols, access.expr, modules=_LAMBDIFY)
                operands = tuple(names[symbol] for symbol in symbols)
            else:  # a host-computed scalar plus a constant
                base = next(k for k in names if (access.expr - k).is_number)
                position = functools.partial(operator.add, int(access.expr - base))
                operands = (names[base],)
            key = (access.tensor, access.dim, str(access.expr))
            checks.setdefault(
                key, (access.tensor, access.dim, access.size, position, operands)
            )
        return tuple(checks.values())

    def _native_call(self) -> object | None:
        """The native ``EntryCall`` of a static entry on a target with native
        paths, if the extension is available."""
        static = self._static
        if static is None or static.scalars or not self.target.native_in_place:
            return None
        if (extension := native.module()) is None:
            return None
        address = ctypes.cast(self._func, ctypes.c_void_p).value
        return extension.EntryCall(
            address,
            [
                (arg.shape, arg.dtype, arg.role == "inout")
                for arg in self.args  # every arg a memref of static shape
            ],
            self.target.buffer_alignment,
        )

    def call_in_place(self, values: Sequence[object]) -> list[torch.Tensor] | None:
        """Call the entry and return its inout tensors if every argument but
        the runtime scalars is a tensor that it can take as is -- the static
        shape and dtype, contiguous strides, no negative or conjugate bit,
        data aligned to
        :attr:`MLIRTarget.buffer_alignment`,
        :meth:`MLIRTarget.passes_in_place` and a target without call
        hooks (:meth:`MLIRTarget.prepare_call`) or one whose hooks run here
        (:attr:`MLIRTarget.hooks_in_place`), no inout sharing memory with
        another argument except as the general path last passed them
        (:meth:`note_passed_as_is`) -- else return ``None`` without calling
        it: the caller then takes the general path (``driver.call_entry``),
        which stages, checks and reports. Only the data pointers and scalar
        values change between calls, so the arguments are packed once per
        entry. On a target with
        :attr:`MLIRTarget.native_in_place`, the native ``EntryCall``
        (``native/``) does these checks and the call first (descriptors on
        the stack, no lock); a call it refuses gets the Python checks."""
        static = self._static
        if static is None:
            return None
        # Checked per call, not once: entries are shared through the JIT
        # cache, and a target's hooks may be replaced after one was built.
        target = self.target
        hooks = type(target)
        prepare = hooks.prepare_call is not MLIRTarget.prepare_call
        finish = hooks.finish_call is not MLIRTarget.finish_call
        if (prepare or finish) and not target.hooks_in_place:
            return None
        if (native_call := self._native) is not None and not (prepare or finish):
            # The checks below and the call, natively (it runs no hooks); a
            # value it doesn't take (e.g. a non-plain tensor that
            # passes_in_place admits) gets the Python checks.
            if native_call.call(values):
                return [values[i] for i in static.inouts]  # type: ignore[misc]
        passes = target.passes_in_place
        alignment = target.buffer_alignment
        scalars = static.scalars
        tensors = [values[i] for i in static.tensors] if scalars else values
        for value, shape, strides, dtype in zip(
            tensors, static.shapes, static.strides, static.dtypes, strict=True
        ):
            if not (
                isinstance(value, torch.Tensor)
                and value.dtype is dtype
                and value.shape == shape
                and value.stride() == strides
                and not value.is_neg()
                and not value.is_conj()
                and passes(value)
            ):
                return None
        pointers = [value.data_ptr() for value in tensors]  # type: ignore[union-attr]
        if alignment != 1 and any(pointer % alignment for pointer in pointers):
            return None
        overlaps = static.overlaps(pointers)
        if overlaps and overlaps not in static.passed_as_is:
            return None
        numbers = [python_type(values[i]) for i, _, python_type in scalars]
        if prepare:
            target.prepare_call(tensors)  # type: ignore[arg-type]
        # One packed buffer per entry: a call from another thread meanwhile
        # takes the general path.
        if not static.lock.acquire(blocking=False):
            return None
        try:
            buffer = static.buffer
            for slot, pointer in zip(static.slots, pointers, strict=True):
                buffer[slot] = pointer
                buffer[slot + 1] = pointer
            for (_, storage, _), number in zip(scalars, numbers, strict=True):
                storage.value = number
            self._func(static.packed)
        except Exception as exc:
            exc.add_note(f"Helion MLIR backend: failed while running '{self.name}'")
            raise
        finally:
            static.lock.release()
        if finish:
            target.finish_call(tensors)  # type: ignore[arg-type]
        return [values[i] for i in static.inouts]  # type: ignore[misc]

    def note_passed_as_is(self, values: Sequence[torch.Tensor]) -> None:
        """The general path passed ``values``, one per tensor argument, as they are.
        What it decides about tensors sharing memory depends only on which
        contiguous tensors overlap (the entry's read order is fixed), so
        :meth:`call_in_place` may pass the same overlaps as they are too,
        without repeating the decision."""
        static = self._static
        if static is not None:
            pointers = [value.data_ptr() for value in values]
            if overlaps := static.overlaps(pointers):
                static.passed_as_is.add(overlaps)

    def __call__(self, values: Sequence[torch.Tensor | float | int | bool]) -> None:
        """Call the entry with one value per argument: a tensor for a memref, a
        Python number for a runtime scalar."""
        packed, _storage = _packed_args(self.args, values)
        try:
            self._func(packed)
        except Exception as exc:
            exc.add_note(f"Helion MLIR backend: failed while running '{self.name}'")
            raise


_WORD = ctypes.sizeof(ctypes.c_int64)


class _StaticCall:
    """The packed arguments of an entry that takes only memrefs of static,
    non-empty shapes and runtime scalars, built once: the descriptors as
    :func:`_packed_args` builds them for contiguous tensors, with the data
    pointers (``allocated`` and ``aligned``) and the scalars' values filled in
    per call. ``shapes`` and the like are per tensor argument, in order."""

    def __init__(self, args: Sequence[EntryArg]) -> None:
        memrefs = [arg for arg in args if arg.role != "scalar"]
        self.tensors = tuple(i for i, arg in enumerate(args) if arg.role != "scalar")
        self.scalars = tuple(
            (i, _SCALAR_CTYPES[arg.dtype][0](), _SCALAR_CTYPES[arg.dtype][1])
            for i, arg in enumerate(args)
            if arg.role == "scalar"
        )
        self.shapes = tuple(torch.Size(arg.shape) for arg in memrefs)
        self.strides = tuple(_contiguous_strides(arg.shape) for arg in memrefs)
        self.dtypes = tuple(arg.dtype for arg in memrefs)
        self.nbytes = tuple(
            math.prod(arg.shape) * arg.dtype.itemsize for arg in memrefs
        )
        self.inouts = tuple(i for i, arg in enumerate(args) if arg.role == "inout")
        words: list[int] = []
        slots: list[int] = []
        for shape, strides in zip(self.shapes, self.strides, strict=True):
            slots.append(len(words))
            words += (0, 0, 0, *shape, *strides)
        self.slots = tuple(slots)
        descriptors = len(words)
        self.buffer = (ctypes.c_int64 * (descriptors + len(memrefs) + len(args)))()
        base = ctypes.addressof(self.buffer)
        words += [base + _WORD * slot for slot in slots]
        pointer = iter(range(descriptors, descriptors + len(memrefs)))
        scalar = {i: storage for i, storage, _ in self.scalars}
        words += [
            ctypes.addressof(scalar[i]) if i in scalar else base + _WORD * next(pointer)
            for i in range(len(args))
        ]
        self.buffer[:] = words
        self.packed = base + _WORD * (descriptors + len(memrefs))
        self.lock = threading.Lock()
        self.written = tuple(arg.role == "inout" for arg in memrefs)
        self.passed_as_is: set[int] = set()
        """Results of :meth:`overlaps` the general path passed as they are."""
        self._last_overlaps: tuple[list[int] | None, int] = (None, 0)

    def overlaps(self, pointers: list[int]) -> int:
        """Which tensor arguments at ``pointers`` overlap an inout: a bitmask with
        one bit per pair of their positions, whatever their addresses. Zero: none.
        A kernel's fresh tensors tend to be at the same addresses from call to
        call, which skips the sweep."""
        last, mask = self._last_overlaps
        if pointers == last:
            return mask
        count = len(pointers)
        order = sorted(range(count), key=pointers.__getitem__)
        nbytes = self.nbytes
        written = self.written
        mask = 0
        for position, i in enumerate(order):
            end = pointers[i] + nbytes[i]
            for later in range(position + 1, count):
                j = order[later]
                if pointers[j] >= end:
                    break
                if written[i] or written[j]:
                    mask |= 1 << (min(i, j) * count + max(i, j))
        self._last_overlaps = (pointers, mask)
        return mask

    @classmethod
    def build(cls, args: Sequence[EntryArg]) -> _StaticCall | None:
        if not args or any(-1 in arg.shape or 0 in arg.shape for arg in args):
            return None
        return cls(args)


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """The strides torch gives a contiguous tensor of ``shape``."""
    strides = []
    step = 1
    for size in reversed(shape):
        strides.append(step)
        step *= max(size, 1)
    return tuple(reversed(strides))


# The C type of each runtime scalar type the entry takes by value, and the
# Python type its values are converted to (as ``torch.tensor(value, dtype)`` did).
_SCALAR_CTYPES: dict[torch.dtype, tuple[type, type]] = {
    torch.float64: (ctypes.c_double, float),
    torch.float32: (ctypes.c_float, float),
    torch.int64: (ctypes.c_int64, int),
    torch.int32: (ctypes.c_int32, int),
    torch.bool: (ctypes.c_bool, bool),
}


def _packed_args(
    args: Sequence[EntryArg], values: Sequence[torch.Tensor | float | int | bool]
) -> tuple[int, tuple[object, ...]]:
    """The address of the packed arguments of a C-interface entry -- one pointer
    per argument: to a pointer to its memref descriptor, or to the value of a
    runtime scalar -- and the storage they point to, which must outlive the call.

    The descriptors (``allocated``, ``aligned``, ``offset``, sizes, strides), the
    descriptor pointers and the packed arguments are all 64-bit words of one
    buffer; each scalar is a C value of its own type.
    """
    words: list[int] = []
    starts: list[int] = []
    scalars: dict[int, ctypes._SimpleCData] = {}
    for index, (arg, value) in enumerate(zip(args, values, strict=True)):
        if arg.role == "scalar":
            c_type, python_type = _SCALAR_CTYPES[arg.dtype]
            scalars[index] = c_type(python_type(value))
            continue
        starts.append(len(words))
        data = value.data_ptr()
        words += (data, data, 0, *value.shape, *value.stride())
    pointers = len(words)
    buffer = (ctypes.c_int64 * (pointers + len(starts) + len(args)))()
    base = ctypes.addressof(buffer)
    words += [base + _WORD * start for start in starts]
    descriptor = iter(range(pointers, pointers + len(starts)))
    words += [
        ctypes.addressof(scalars[index])
        if index in scalars
        else base + _WORD * next(descriptor)
        for index in range(len(args))
    ]
    buffer[:] = words
    return base + _WORD * (pointers + len(starts)), (buffer, scalars)


def entry_args(module: ir.Module, entry: str) -> tuple[EntryArg, ...]:
    """The entry function's arguments, read before lowering."""
    fn = next(
        (
            op
            for op in module.body.operations
            if "sym_name" in op.attributes
            and ir.StringAttr(op.attributes["sym_name"]).value == entry
        ),
        None,
    )
    if fn is None:
        raise ValueError(f"no function named '{entry}' in the module")
    arg_types = ir.FunctionType(
        ir.TypeAttr(fn.attributes["function_type"]).value
    ).inputs
    arg_attrs = fn.attributes.get("arg_attrs")
    if arg_attrs is None:
        raise ValueError(f"'{entry}' is not a Helion MLIR entry function")
    args = []
    for arg_type, attrs in zip(arg_types, ir.ArrayAttr(arg_attrs), strict=True):
        attrs = ir.DictAttr(attrs)
        role = ir.StringAttr(attrs["helion.role"]).value
        if role == "scalar":
            shape: tuple[int, ...] = ()
            element_type = arg_type
        else:
            memref = ir.MemRefType(arg_type)
            shape = tuple(
                -1 if ir.ShapedType.is_dynamic_size(size) else size
                for size in memref.shape
            )
            element_type = memref.element_type
        dtype = mlir_dtype_to_torch(str(element_type))
        if dtype is None or (role == "scalar" and dtype not in _SCALAR_CTYPES):
            kind = "scalar" if role == "scalar" else "memref"
            raise ValueError(f"'{entry}' takes a {kind} of unsupported {element_type}")
        args.append(
            EntryArg(
                ir.StringAttr(attrs["helion.name"]).value,
                role,
                ir.IntegerAttr(attrs["helion.param"]).value
                if "helion.param" in attrs
                else None,
                shape,
                dtype,
            )
        )
    return tuple(args)


def compile_entry(
    module: ir.Module,
    entry: str,
    *,
    target: MLIRTarget | None = None,
    pipeline: str | None = None,
    read_order: ReadOrder | None = None,
    scalar_indices: tuple[ScalarIndex, ...] = (),
    scalar_exprs: tuple[tuple[object, str], ...] = (),
) -> CompiledEntry:
    """Inline, lower and JIT-compile ``entry`` (consumes ``module``) for ``target``
    (default: :func:`resolve_target` for the CPU), or reuse the entry compiled from
    an identical module for the same target and pipeline. ``pipeline`` defaults to
    the target's. A target whose lowerings stop early by default lowers and raises :class:`LoweringStoppedError`."""
    if target is None:
        target = resolve_target()
    pipeline = target.resolve_pipeline(pipeline)
    indices = (scalar_indices, scalar_exprs)
    debug = DebugOptions.from_env()
    stop_at = target.default_stop_at()
    if (
        debug.dump_ir
        or debug.dump_pre_lowering
        or debug.dump_lowered
        or debug.dump_stages
        or not target.is_full(stop_at)
    ):
        return _compile_entry(
            module, entry, target, pipeline, stop_at, debug, read_order, indices
        )
    # The native switch is part of the key: entries compiled with it on carry
    # native call paths.
    key = (
        hashlib.sha256(str(module).encode()).hexdigest(),
        target.name,
        pipeline,
        native.enabled(),
        repr(read_order),
        repr(indices),
    )
    if (compiled := _JIT_CACHE.get(key)) is not None:
        _JIT_CACHE.move_to_end(key)
        return compiled
    compiled = _compile_entry(
        module, entry, target, pipeline, stop_at, debug, read_order, indices
    )
    _JIT_CACHE[key] = compiled
    if len(_JIT_CACHE) > _JIT_CACHE_SIZE:
        _JIT_CACHE.popitem(last=False)
    return compiled


def lower_entry(
    module: ir.Module,
    entry: str,
    *,
    target: MLIRTarget | None = None,
    pipeline: str | None = None,
    stop_at: str | None = None,
) -> ir.Module:
    """Inline and lower ``entry`` (consumes ``module``) for ``target`` (default:
    the CPU) with ``pipeline`` (default: the target's), to the target's stage
    ``stop_at`` (default: :meth:`MLIRTarget.default_stop_at`; ``None`` there: the
    full lowering, the LLVM dialect on the CPU). Nothing is JIT-compiled."""
    if target is None:
        target = resolve_target()
    pipeline = target.resolve_pipeline(pipeline)
    if stop_at is None:
        stop_at = target.default_stop_at()
    target.check_stop_at(stop_at)
    if stop_at is None and target.stop_points:
        stop_at = target.stop_points[-1]
    target.validate_environment(stop_at)
    return _inline_and_lower(
        module, entry, target, pipeline, stop_at, DebugOptions.from_env()
    )


def _compile_entry(
    module: ir.Module,
    entry: str,
    target: MLIRTarget,
    pipeline: str,
    stop_at: str | None,
    debug: DebugOptions,
    read_order: ReadOrder | None,
    indices: tuple[tuple[ScalarIndex, ...], tuple[tuple[object, str], ...]],
) -> CompiledEntry:
    full = target.is_full(stop_at)
    target.validate_environment(None if full else stop_at)
    args = entry_args(module, entry)
    lowered = _inline_and_lower(module, entry, target, pipeline, stop_at, debug)
    if not full:
        raise LoweringStoppedError(target.name, stop_at)
    with _stage("JIT compilation"):
        runner = target.runner(lowered)
    return CompiledEntry(entry, args, runner, target, read_order, *indices)


def _inline_and_lower(
    module: ir.Module,
    entry: str,
    target: MLIRTarget,
    pipeline: str,
    stop_at: str | None,
    debug: DebugOptions,
) -> ir.Module:
    _dump_if(debug.dump_ir, "MLIR before inlining", module)
    with _stage("inlining"):
        inline_module(module)
    _dump_if(debug.dump_pre_lowering, "MLIR before lighthouse lowering", module)
    until = "" if target.is_full(stop_at) else f" up to '{stop_at}'"
    with _stage(f"lowering with the lighthouse '{pipeline}' pipeline{until}"):
        lowered = _lower(module, entry, target, pipeline, stop_at)
    _dump_if(debug.dump_lowered, f"MLIR after lighthouse lowering{until}", lowered)
    return lowered


def _lower(
    module: ir.Module,
    entry: str,
    target: MLIRTarget,
    pipeline: str,
    stop_at: str | None = None,
) -> ir.Module:
    """Lower ``module`` with the target's lighthouse ``pipeline`` (to ``stop_at``),
    in a forked process under the compile timeout: native passes cannot be
    interrupted, only killed with their process. The lowering is built here and
    applied there."""
    with module.context, ir.Location.unknown():
        lowering = target.lowering(module, entry, pipeline, stop_at)
        timeout = compile_timeout()
        if timeout is None or "fork" not in multiprocessing.get_all_start_methods():
            lowered = lowering.apply(module)
            # Already logged, in this process.
            _take_lowering_log(lowered)
            _warn_from_lowering(lowered)
            return lowered
    receiver, sender = multiprocessing.Pipe(duplex=False)
    with _anonymous_file() as stderr:
        child = multiprocessing.get_context("fork").Process(
            target=_lower_in_child,
            args=(target, lowering, module, sender, stderr.fileno()),
            daemon=True,
        )
        with warnings.catch_warnings():
            # The child uses only the module's context, which runs no threads.
            warnings.filterwarnings("ignore", "This process .* is multi-threaded")
            child.start()
        sender.close()
        error: Exception | None = None
        died = False
        try:
            if receiver.poll(timeout):
                ok, result = receiver.recv()
                if not ok:
                    error = result
            else:
                child.kill()
                error = CompileTimeoutError(pipeline, timeout)
        except EOFError:
            died = True
        finally:
            receiver.close()
            child.join()
        stderr.seek(0)
        output = stderr.read().decode(errors="replace").strip()
    if died:
        error = RuntimeError(_death_message(child.exitcode, output))
    if error is not None:
        if child.exitcode and not died and not isinstance(error, CompileTimeoutError):
            error.add_note(f"exit code {child.exitcode}")
        if output:
            error.add_note(f"lighthouse output:\n{output}")
        raise error
    if output:
        log.debug("lighthouse output:\n%s", output)
    lowered = ir.Module.parse(result, context=module.context)
    for line in _take_lowering_log(lowered):
        log.warning("%s", line)
    _warn_from_lowering(lowered)
    return lowered


def _take_lowering_log(module: ir.Module) -> list[str]:
    """The lines the lowering recorded on ``module``, removed from it."""
    attributes = module.operation.attributes
    if LOWERING_LOG_ATTR not in attributes:
        return []
    lines = [
        ir.StringAttr(line).value
        for line in ir.ArrayAttr(attributes[LOWERING_LOG_ATTR])
    ]
    del attributes[LOWERING_LOG_ATTR]
    return lines


def _warn_from_lowering(module: ir.Module) -> None:
    """The warnings the lowering recorded on ``module``, raised here and removed from it."""
    attributes = module.operation.attributes
    if LOWERING_WARNINGS_ATTR not in attributes:
        return
    for line in ir.ArrayAttr(attributes[LOWERING_WARNINGS_ATTR]):
        warn_on_each_compile(ir.StringAttr(line).value)
    del attributes[LOWERING_WARNINGS_ATTR]


def _death_message(exitcode: int | None, output: str) -> str:
    """What killed a lowering process: its signal or exit code, and the assertion
    it printed (or else its last line of output)."""
    if exitcode is not None and exitcode < 0:
        how = f"on {signal.Signals(-exitcode).name}"
    else:
        how = f"with exit code {exitcode}"
    lines = output.splitlines()
    cause = next((line for line in lines if "Assertion" in line), None)
    if cause is None and lines:
        cause = lines[-1]
    message = f"lighthouse lowering process died {how}"
    return f"{message}: {cause.strip()}" if cause else message


def _anonymous_file() -> BinaryIO:
    """A file with no name, in memory where the OS supports it (Linux)."""
    if hasattr(os, "memfd_create"):
        return os.fdopen(os.memfd_create("lighthouse-stderr"), "w+b")
    return tempfile.TemporaryFile()


def _lower_in_child(
    target: MLIRTarget,
    lowering: Lowering,
    module: ir.Module,
    sender: Connection,
    stderr: int,
) -> None:
    """Forked: apply the parent's lowering to the parent's module (both copied on
    write) and send back the lowered module as bytecode. What lighthouse prints
    goes to ``stderr``: diagnostics of a lowering that succeeds are not errors."""
    os.dup2(stderr, 2)
    # Freeing an MLIR context whose threads did not survive the fork waits forever.
    gc.disable()
    target.enter_lowering_child()
    try:
        with module.context, ir.Location.unknown():
            lowered = lowering.apply(module)
        sender.send((True, _bytecode(lowered)))
    except Exception as exc:
        sender.send((False, _picklable(exc)))
    finally:
        sender.close()


def _bytecode(module: ir.Module) -> bytes:
    buffer = io.BytesIO()
    module.operation.write_bytecode(buffer)
    return buffer.getvalue()


def _picklable(exc: Exception) -> Exception:
    try:
        return pickle.loads(pickle.dumps(exc))
    except Exception:
        fallback = RuntimeError(f"{type(exc).__name__}: {exc}")
        for note in getattr(exc, "__notes__", ()):
            fallback.add_note(note)
        return fallback


@contextmanager
def _stage(what: str) -> Iterator[None]:
    try:
        yield
    except Exception as exc:
        exc.add_note(f"Helion MLIR backend: failed while {what}")
        raise
