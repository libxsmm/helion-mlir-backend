"""Runtime backend registration helpers for external Helion MLIR backend."""

from __future__ import annotations

import dataclasses
import logging
import numbers
import re
import struct
import sys
import types
from typing import Callable
from typing import Sequence
import warnings

from helion.language.constexpr import ConstExpr
import torch

log = logging.getLogger(__name__)


def install() -> bool:
    """Register external MLIR backend into Helion registry.

    Returns True when registration is applied successfully.
    """
    try:
        from helion._compiler.backend_registry import register_compiler_backend
        from helion.runtime.kernel import BoundKernel

        from helion_mlir_backend._compiler import call_trace
        from helion_mlir_backend._compiler.mlir.aten_bridge.original_args import (
            install_original_args_capture,
        )
        from helion_mlir_backend._compiler.mlir.backend import MLIRBackend
        from helion_mlir_backend._compiler.mlir.driver import mlir_compile_config
        from helion_mlir_backend._compiler.mlir.trace_mode import install_trace_mode
    except ImportError as exc:
        log.debug("External MLIR backend registration unavailable: %s", exc)
        return False

    register_compiler_backend(MLIRBackend)
    _patch_bound_kernel(BoundKernel, MLIRBackend, mlir_compile_config)
    _allow_config_keys()
    _register_autotune_cache()
    _accept_mlir_module_globals()
    _keep_runtime_slice_sizes(MLIRBackend)
    _key_function_args_by_globals_and_defaults(MLIRBackend)
    _key_constexpr_by_type_and_bits(MLIRBackend)
    _key_ints_read_at_compile_time(BoundKernel, MLIRBackend)
    _fast_key_functions_and_constexpr(MLIRBackend)
    _check_module_lookups(MLIRBackend)
    call_trace.install(MLIRBackend)
    install_trace_mode()
    install_original_args_capture()
    return True


def _accept_mlir_module_globals() -> None:
    """Let kernels read ``mlir.ir.Module`` globals (``inline_mlir`` sources).

    Helion rejects globals of unknown types; like Triton ``JITFunction`` globals,
    a module becomes its source text.
    """
    from helion._compiler.compile_environment import CompileEnvironment
    import mlir.ir as ir

    from helion_mlir_backend._compiler.mlir.snippets import source_text

    if getattr(CompileEnvironment, "_helion_mlir_module_globals", False):
        return
    original = CompileEnvironment.to_fake

    def to_fake(self: CompileEnvironment, obj: object, origin: object) -> object:
        if isinstance(obj, ir.Module):
            return source_text(obj)
        return original(self, obj, origin)

    CompileEnvironment.to_fake = to_fake
    CompileEnvironment._helion_mlir_module_globals = True


def _keep_runtime_slice_sizes(MLIRBackend: type) -> None:
    """Give a ``:`` slice of a dim known only at run time a reduction block.

    The MLIR backend does not pad factory shapes to powers of two
    (``MLIRBackend.pad_factory_tensors_to_power_of_2``). On such backends
    Helion sizes ``:`` slices with ``try_concretize_symint``, which turns a
    dim of a ``static_shapes=False`` kernel into the example input's size
    without a guard: ``row / row.size(1)`` then divides by that size for
    every input of the shape bucket. A dim that is a compile-time constant
    (static or ``hl.specialize``) stays that constant; one known only at
    run time gets a reduction block, as with padding, whose size the kernel
    reads at run time.
    """
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.compile_environment import _has_unbacked
    import sympy
    import torch

    if getattr(CompileEnvironment, "_helion_mlir_runtime_slice_sizes", False):
        return
    original = CompileEnvironment.try_concretize_symint

    def try_concretize_symint(
        self: CompileEnvironment, size: int | torch.SymInt
    ) -> int | torch.SymInt:
        if not isinstance(self.backend, MLIRBackend) or not isinstance(
            size, torch.SymInt
        ):
            return original(self, size)
        expr = self.shape_env.replace(size._sympy_())
        if isinstance(expr, sympy.Integer):
            return int(expr)
        if _has_unbacked(expr):
            return original(self, size)
        return self.allocate_reduction_dimension(size).var

    CompileEnvironment.try_concretize_symint = try_concretize_symint
    CompileEnvironment._helion_mlir_runtime_slice_sizes = True


class _SameObject:
    """Key part that matches only the very object it holds, unhashable or not.
    Holding it keeps its id from being reused while the key lives."""

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SameObject) and self.value is other.value

    def __hash__(self) -> int:
        return id(self.value)

    def __repr__(self) -> str:
        return f"<{type(self.value).__name__} object>"


def _default_key(value: object) -> object:
    """Defaults are compiled in as constants, so equal values may share a key.
    Floats by their bits: 0.0 == -0.0, and NaN differs from itself. Another object
    by identity and the ints in it, which may be mutated: a default is read at
    compile time, and records no guard to go by."""
    if type(value) is float:
        return float, value.hex()
    if value is None or type(value) in (bool, int, str):
        return type(value), value
    return _SameObject(value), _cell_ints(value)


def _constexpr_key(value: object) -> object:
    """An ``hl.constexpr`` value: its type too (``1 == 1.0 == True``), floats of any
    type by their bits (``0.0 == -0.0``, and NaN differs from itself), each element
    of a tuple likewise."""
    kind = type(value)
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return kind, float(value).hex()
    if kind is tuple:
        return tuple, tuple(map(_constexpr_key, value))  # type: ignore[arg-type]
    return kind, value


def _cell_ints(value: object) -> object:
    """The ints in a value, nested in tuples, lists, dicts, dataclasses and
    namedtuples; ``None`` where there are none. Everything else stays with Helion's
    key: floats and bools (reading them at compile time raises) and ``hl.constexpr``
    (its own extractor). A function counts by the ints of its cells: it may be held
    in a container or a default, where a guard's source stops."""
    kind = type(value)
    if kind is int:
        return value
    if kind is ConstExpr:
        return None
    if kind is types.FunctionType:
        return _function_ints(value)
    if isinstance(value, (tuple, list)):
        parts = tuple(map(_cell_ints, value))
    elif isinstance(value, dict):
        parts = tuple((key, _cell_ints(value[key])) for key in sorted(value, key=repr))
        if all(part[1] is None for part in parts):
            return None
        return parts
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        parts = tuple(
            _cell_ints(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    else:
        return None
    return parts if any(part is not None for part in parts) else None


def _key_function_args_by_globals_and_defaults(MLIRBackend: type) -> None:
    """Keep kernels from reusing a compile for a function that differs from it.

    Helion keys a function argument (an epilogue) by its code object and closure
    cells. Code objects compare by content, ignoring the file, so two modules'
    functions of the same text and line share a key, as do functions of one
    ``def`` with different defaults. The second call then ran the first one's
    compile: other module globals, or the first default, without an error.
    """
    import types

    from helion.runtime.kernel import _specialization_extractors as extractors

    original = extractors[types.FunctionType]
    if getattr(original, "_helion_mlir_globals_and_defaults", False):
        return
    backend_name = MLIRBackend().name

    def function_key(kernel: object, fn: types.FunctionType) -> object:
        key = original(kernel, fn)
        if kernel.settings.backend != backend_name:  # type: ignore[attr-defined]
            return key
        defaults = tuple(map(_default_key, fn.__defaults__ or ()))
        return key, _SameObject(fn.__globals__), defaults

    function_key._helion_mlir_globals_and_defaults = True  # type: ignore[attr-defined]
    extractors[types.FunctionType] = function_key


_MANY_INT_COMPILES = 16
"""Bound kernels of one kernel from guarded int values before it warns, once."""


def _source_steps(source: object) -> tuple[str, object] | None:
    """How to read a recorded symbol's value from the arguments: ``(parameter,
    steps)``, each step ``("attr", name)`` or ``("item", index)``. Helion's
    ``make_extractor`` can't read a closure cell, e.g.
    ``L['fn'].__closure__[0].cell_contents``. ``None``: no value of an argument, or
    a tensor's size, which the key holds anyway. Steps ``None``: an argument's value
    by a chain this can't follow, so the caller keys all the ints in that argument
    rather than leave them stale."""
    from torch._dynamo.source import AttrSource
    from torch._dynamo.source import GetItemSource
    from torch._dynamo.source import LocalSource
    from torch._dynamo.source import TensorPropertySource

    if isinstance(source, TensorPropertySource):
        return None
    steps: list[tuple[str, object]] = []
    current = source
    while type(current) is not LocalSource:
        if type(current) is AttrSource:
            steps.append(("attr", current.member))
        elif type(current) is GetItemSource and not current.index_is_slice:
            steps.append(("item", current.index))
        else:
            root = re.match(r"L\['(\w+)'\]", getattr(source, "name", ""))
            return (root.group(1), None) if root else None
        current = current.base
    return current.local_name, tuple(reversed(steps))


def _function_ints(fn: types.FunctionType) -> object:
    """The ints of a function's cells, in the functions in its cells too. An empty
    cell holds none: a default may be such a function, and the key must not raise."""
    parts: list[object] = []
    for cell in fn.__closure__ or ():
        try:
            parts.append(_cell_ints(cell.cell_contents))
        except ValueError:
            parts.append(None)
    return tuple(parts)


def _walked_ints(value: object, steps: tuple[tuple[str, object], ...] | None) -> object:
    """The ints at the end of ``steps``. A function of other code may have no such
    cell, and steps may be ``None``: then all the ints in ``value``, as a finer key
    costs a compile and a coarser one a wrong answer."""
    if steps is None:
        return _cell_ints(value)
    current = value
    try:
        for kind, step in steps:
            current = getattr(current, step) if kind == "attr" else current[step]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return _cell_ints(value)
    return _cell_ints(current)


def _guarded_ints(kernel: object, args: Sequence[object]) -> object:
    """The kernel's guarded ints in ``args``, each by its value. A short call binds
    with its defaults applied but is keyed as made."""
    guarded = kernel.__dict__.get("_helion_mlir_guarded")
    if not guarded:
        return ()
    return tuple(
        [
            _walked_ints(args[index] if index < len(args) else default, steps)
            for index, default, steps in guarded
        ]
    )


def _note_guarded_ints(bound: object) -> None:
    """Add what this compile recorded a guard on to its kernel's list. A new list:
    another thread may be reading the old one."""
    shape_env = bound.env.shape_env  # type: ignore[attr-defined]
    symbols = {
        symbol for guard in shape_env.guards for symbol in guard.expr.free_symbols
    }
    kernel = bound.kernel  # type: ignore[attr-defined]
    parameters = kernel.signature.parameters
    names = list(parameters)
    found = kernel.__dict__.setdefault("_helion_mlir_guarded_sources", set())
    for symbol in symbols:
        for source in shape_env.var_to_sources.get(symbol, ()):
            step = _source_steps(source)
            if step is not None and step[0] in parameters:
                found.add(step)
    # Against the current list, not this call's additions: with no lock, a thread may
    # have assigned an older list after a newer one, and the next compile repairs it.
    if len(found) != len(kernel.__dict__.get("_helion_mlir_guarded", ())):
        known = list(found)  # another thread may be adding
        kernel._helion_mlir_guarded = [
            (names.index(name), parameters[name].default, steps)
            for name, steps in sorted(known, key=repr)
        ]


def _warn_on_many_int_compiles(bound: object, args: tuple[object, ...]) -> None:
    """A compile per value is right but costs: say so once a
    kernel has many, as nothing evicts them."""
    kernel = bound.kernel  # type: ignore[attr-defined]
    values = _guarded_ints(kernel, args)
    if not values:
        return
    seen = kernel.__dict__.setdefault("_helion_mlir_int_values", set())
    seen.add(values)
    if len(seen) > _MANY_INT_COMPILES and not kernel.__dict__.get(
        "_helion_mlir_warned_ints"
    ):
        kernel._helion_mlir_warned_ints = True
        warnings.warn(
            f"Helion MLIR backend: kernel {kernel.name} compiled for more than "
            f"{_MANY_INT_COMPILES} values of an int that it reads at compile time, "
            "in a closure cell or passed to a called function: one compile per "
            "value. If such an int changes per call, read it as a value only, or "
            "as a plain int or Python-number parameter of the kernel.",
            stacklevel=2,
        )


def _key_ints_read_at_compile_time(BoundKernel: type, MLIRBackend: type) -> None:
    """Keep an int read at compile time from reusing another value's compile.

    A function the kernel calls is traced, so an ``if n > 2`` on an int in a closure
    cell, or on an int argument passed to that function, takes the example value's
    branch at compile time and records a guard. Helion keys an int by its type and
    nothing acts on the guard: the next value ran that branch without an error. Such
    an int is keyed by value through Helion's extras, with the other ints of its
    cell or container argument; one without a guard, read as a value at run time or
    in the kernel's own body, keeps one compile, as in Helion's model.

    Helion asks for a signature's extras once, from its first bound kernel, but a
    later compile may take another branch and record a guard on another int
    (``if n > 2 and m > 2``) or reach an ``hl.specialize``. So the extras read what
    every compile adds to: the kernel's list of guarded ints, which applies to any
    closure of the signature, first seen before or after, and Helion's own extras of
    each compile of the signature. Calls that reach one compile then agree on every
    guard that compile recorded: its trace is theirs. A compile keyed before a list
    grew stays reachable through the dispatch cache and is right for those calls;
    ``bind()`` compiles such values once more. Any guard counts, also one the value
    can't change the outcome of: that costs a compile per value, not a wrong result.

    Guards are read at bind: a read of the value that records no guard then, in the
    trace or later in lowering, or under the profiler, which suppresses guards,
    stays stale. So do a global's reads. Reading the guards rather than changing
    ``specialized_vars`` leaves the lowering as it is.

    One race, in threads only, not fixed (it would need a lock around Helion's
    ``bind``): a thread computes its key, another thread's first compile of a branch
    stores itself under that same key and adds a guard, and the first thread's lookup
    then finds that compile, for a value it took the other branch for, and the
    dispatch cache keeps the wrong answer for those arguments. The window is between
    ``_get_bound_kernel_cache_key`` and ``_bound_kernels.get`` in ``Kernel.bind``,
    during a first compile of that branch; it was only seen forced.
    """
    original = BoundKernel._specialize_extra
    if getattr(original, "_helion_mlir_guarded_ints", False):
        return
    original_init = BoundKernel.__init__

    def extras_of(bound: object) -> dict[object, list[Callable[..., object]]]:
        """Helion's extras of each compile of this signature, once per set of
        sources: a later compile may specialize on what an earlier one didn't."""
        env = bound.env  # type: ignore[attr-defined]
        sources = env.shape_env.var_to_sources
        kernel = bound.kernel  # type: ignore[attr-defined]
        by_signature = kernel.__dict__.setdefault("_helion_mlir_extras", {})
        collected = by_signature.setdefault(bound._base_spec_key, {})  # type: ignore[attr-defined]
        what = (
            sorted(sources[symbol][0].name for symbol in env.specialized_vars),
            sorted(map(repr, env.specialized_strides)),
            sorted(map(repr, env.tensor_descriptor_layout_guards)),
        )
        collected.setdefault(repr(what), original(bound))
        return collected

    def init(self: object, kernel: object, args: object, *rest: object) -> None:
        original_init(self, kernel, args, *rest)
        if isinstance(self.env.backend, MLIRBackend):  # type: ignore[attr-defined]
            extras_of(self)
            _note_guarded_ints(self)
            _warn_on_many_int_compiles(self, args)  # type: ignore[arg-type]

    def specialize_extra(self: object) -> list[Callable[..., object]]:
        if not isinstance(self.env.backend, MLIRBackend):  # type: ignore[attr-defined]
            return original(self)
        kernel = self.kernel  # type: ignore[attr-defined]
        parameters = kernel.signature.parameters
        names = list(parameters)
        collected = extras_of(self)

        def helion_extras(args: Sequence[object]) -> object:
            # A short call is keyed as made, but a later compile's extras may read a
            # parameter it left to its default. A list: a compile in another thread
            # may add to these meanwhile.
            full = (
                *args,
                *(parameters[name].default for name in names[len(args) :]),
            )
            return tuple(
                [extra(full) for extras in list(collected.values()) for extra in extras]
            )

        def guarded_ints(args: Sequence[object]) -> object:
            return _guarded_ints(kernel, args)

        return [helion_extras, guarded_ints]

    specialize_extra._helion_mlir_guarded_ints = True  # type: ignore[attr-defined]
    BoundKernel.__init__ = init
    BoundKernel._specialize_extra = specialize_extra


def _key_constexpr_by_type_and_bits(MLIRBackend: type) -> None:
    """Keep ``hl.constexpr`` values of different type or sign of zero apart.

    Helion keys them by the value: ``0.0 == -0.0``, so ``x / hl.constexpr(-0.0)``
    after ``x / hl.constexpr(0.0)`` ran the first compile without an error, and
    ``1``, ``1.0`` and ``True`` shared one. The key of an argument (wrapped, or
    plain to an annotated parameter) is built in ``_base_specialization_key``; of
    a constexpr in a cell or container, in the extractor table.
    """
    from helion.runtime.kernel import Kernel
    from helion.runtime.kernel import _specialization_extractors as extractors

    original_extractor = extractors[ConstExpr]
    if getattr(original_extractor, "_helion_mlir_constexpr_key", False):
        return
    backend_name = MLIRBackend().name
    original_base = Kernel._base_specialization_key

    def constexpr_extractor(kernel: Kernel, value: ConstExpr) -> object:
        if kernel.settings.backend != backend_name:
            return original_extractor(kernel, value)
        return _constexpr_key(value.value)

    def base_specialization_key(self: Kernel, args: object) -> tuple[object, ...]:
        key = original_base(self, args)
        if self.settings.backend != backend_name:
            return key
        typed = [
            _constexpr_key(value.value if isinstance(value, ConstExpr) else value)
            for value, annotation in zip(args, self._annotations, strict=False)
            if isinstance(value, ConstExpr) or annotation is ConstExpr
        ]
        return (*key, tuple(typed)) if typed else key

    constexpr_extractor._helion_mlir_constexpr_key = True  # type: ignore[attr-defined]
    extractors[ConstExpr] = constexpr_extractor
    Kernel._base_specialization_key = base_specialization_key


_MAX_FAST_KEYS = 4096
"""Entries of a kernel's dispatch cache before the function path clears it: a
float argument or guarded int that changes per call would otherwise add one entry
per call, for as long as the kernel lives. A cleared cache costs a ``bind()`` per
key, nothing else."""

_float_bits = struct.Struct("d").pack


class _NoFastKey(Exception):
    """An argument the fast key doesn't handle: the call takes ``bind()``."""


def _scalar_fast_key(value: object) -> object:
    """A scalar's type and value, floats by their bits (0.0 == -0.0, and NaN
    differs from itself); ``_NoFastKey`` for anything else."""
    kind = type(value)
    if kind is float:
        return float, _float_bits(value)  # type: ignore[arg-type]
    if value is None or kind is int or kind is bool or kind is str:
        return kind, value
    raise _NoFastKey


def _tensor_fast_key(tensor: torch.Tensor) -> object:
    """The same metadata as ``Kernel._fast_dispatch_key`` for a tensor."""
    indices = getattr(tensor, "_dynamo_static_indices", None)
    return (
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.device,
        None if indices is None else frozenset(indices),
    )


def _cell_fast_key(kernel: object, value: object) -> object:
    """A closure cell: a number by its type, as in the full key, so a value that
    changes per call, read at run time, keeps one entry and takes no ``bind()``. What
    this doesn't know takes Helion's own key of the value."""
    kind = type(value)
    if kind is torch.Tensor or kind is torch.nn.Parameter:
        return _tensor_fast_key(value)  # type: ignore[arg-type]
    if kind is ConstExpr:
        return ConstExpr, _scalar_fast_key(value.value)  # type: ignore[attr-defined]
    if kind is types.FunctionType:
        return _function_fast_key(kernel, value)  # type: ignore[arg-type]
    if kind is torch.dtype or kind is torch.device:
        return value
    if kind is int or kind is float or kind is bool:
        return kind
    if value is None or kind is str:
        return kind, value
    key = kernel._specialization_key(value)  # type: ignore[attr-defined]
    hash(key)  # what bind() raises on takes bind(), which reports it
    return kind, key


def _function_fast_key(kernel: object, fn: types.FunctionType) -> object:
    """At least what ``function_key`` holds: code, cells, globals and defaults."""
    closure = fn.__closure__
    return (
        fn.__code__,
        ()
        if closure is None
        else tuple([_cell_fast_key(kernel, cell.cell_contents) for cell in closure]),
        _SameObject(fn.__globals__),
        tuple(map(_default_key, fn.__defaults__ or ())),
    )


def _extended_fast_key(kernel: object, args: tuple[object, ...]) -> object | None:
    """``Kernel._fast_dispatch_key`` that also handles functions and
    ``hl.constexpr`` arguments, part by part at least as fine as the full key and
    with the same guarded ints:
    two calls with one key bind to one ``BoundKernel``. ``None``: the call takes
    ``bind()``, which is always right."""
    annotations = kernel._annotations  # type: ignore[attr-defined]
    if len(args) > len(annotations):
        return None
    key: list[object] = []
    has_tensor = False
    try:
        for value, annotation in zip(args, annotations, strict=False):
            kind = type(value)
            if kind is torch.Tensor or kind is torch.nn.Parameter:
                if annotation is ConstExpr:  # the full key holds this very tensor
                    return None
                has_tensor = True
                key.append(_tensor_fast_key(value))  # type: ignore[arg-type]
            elif kind is types.FunctionType:
                if annotation is ConstExpr:  # the full key holds this very function
                    return None
                key.append(_function_fast_key(kernel, value))  # type: ignore[arg-type]
            elif kind is types.BuiltinFunctionType:
                if annotation is ConstExpr:
                    return None
                key.append(value)
            elif kind is ConstExpr:
                key.append((ConstExpr, _scalar_fast_key(value.value)))  # type: ignore[attr-defined]
            elif kind is torch.dtype or kind is torch.device:
                key.append(value)
            else:
                key.append(_scalar_fast_key(value))
    except Exception:
        return None
    if not has_tensor:
        return None
    key += [_guarded_ints(kernel, args)]
    key.append(kernel._compute_is_distributed(args))  # type: ignore[attr-defined]
    if kernel._key_fn is not None:  # type: ignore[attr-defined]
        key.append(kernel._key_fn(*args))  # type: ignore[attr-defined]
    return tuple(key)


def _fast_key_functions_and_constexpr(MLIRBackend: type) -> None:
    """Give mlir kernels called with a function or ``hl.constexpr`` Helion's fast
    dispatch: its ``_fast_dispatch_key`` returns ``None`` for them, so every call
    pays for ``bind()``. Other kernels and other arguments keep Helion's key, except
    a kernel with a constexpr-annotated parameter: Helion's key of a plain float
    has ``0.0 == -0.0``, coarser than this backend's full key
    (:func:`_key_constexpr_by_type_and_bits`)."""
    from helion.runtime.kernel import Kernel

    original = Kernel._fast_dispatch_key
    if getattr(original, "_helion_mlir_extended", False):
        return
    backend_name = MLIRBackend().name

    def pure_key(self: Kernel, args: tuple[object, ...]) -> object:
        """The key of a call, changing nothing (``explain``)."""
        if self.settings.backend != backend_name:
            return original(self, args)
        if ConstExpr in self._annotations:
            return _extended_fast_key(self, args)
        key = original(self, args)
        return _extended_fast_key(self, args) if key is None else key

    def fast_dispatch_key(self: Kernel, args: tuple[object, ...]) -> object:
        if self.settings.backend != backend_name:
            return original(self, args)
        if ConstExpr not in self._annotations:
            key = original(self, args)
            if key is not None:
                return key
        key = _extended_fast_key(self, args)
        if key is not None and len(self._dispatch_cache) >= _MAX_FAST_KEYS:
            self._dispatch_cache.clear()
        return key

    fast_dispatch_key._helion_mlir_pure = pure_key  # type: ignore[attr-defined]
    fast_dispatch_key._helion_mlir_extended = True  # type: ignore[attr-defined]
    fast_dispatch_key._helion_mlir_original = original  # type: ignore[attr-defined]
    Kernel._fast_dispatch_key = fast_dispatch_key


def _check_module_lookups(MLIRBackend: type) -> None:
    """Refuse a module global when ``sys.modules`` names another module.

    Helion finds a global's module through ``sys.modules``, by name. Its check
    that it found the right one is an ``assert``, gone under ``python -O``, and
    is skipped for a name it already resolved in this kernel; host code then
    reads the other module's value.
    """
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion._compiler.type_propagation import LocalScope

    from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

    if getattr(HostFunction, "_helion_mlir_checked_lookups", False):
        return

    def check(scope: dict[str, object], global_name: str) -> None:
        if not isinstance(CompileEnvironment.current().backend, MLIRBackend):
            return
        name = scope["__name__"]
        module = sys.modules.get(name)
        if module is None or vars(module) is not scope:
            raise UnsupportedOperationError(
                f"module global {global_name}",
                reason=f"sys.modules[{name!r}] is "
                f"{'missing' if module is None else 'another module'}; "
                "give each loaded module its own name",
            )

    global_scope_origin = HostFunction.global_scope_origin
    import_from_module = HostFunction.import_from_module
    maybe_get = LocalScope.maybe_get

    def checked_global_scope_origin(self: HostFunction, name: str) -> object:
        check(self.fn.__globals__, name)
        return global_scope_origin(self, name)

    def local_maybe_get(self: LocalScope, name: str) -> object:
        """Assigning a local looks its name up first, to compare types with what it
        rebinds: only a local is rebound. A global of that name isn't read: it may
        be of a type no kernel can take, or of a module that isn't loaded. A name the
        function declares ``global`` is no local, and goes on to Helion's checks."""
        if not isinstance(CompileEnvironment.current().backend, MLIRBackend):
            return maybe_get(self, name)
        scope = self
        while isinstance(scope, LocalScope):
            if name in scope.variables:
                return scope.variables[name]
            scope = scope.parent
        if name in scope.function.fn.__code__.co_varnames:
            return None
        return maybe_get(self, name)

    def checked_import_from_module(
        self: HostFunction, module_scope: dict[str, object], name: str
    ) -> object:
        check(module_scope, name)
        return import_from_module(self, module_scope, name)

    LocalScope.maybe_get = local_maybe_get
    HostFunction.global_scope_origin = checked_global_scope_origin
    HostFunction.import_from_module = checked_import_from_module
    HostFunction._helion_mlir_checked_lookups = True


def _register_autotune_cache() -> None:
    from helion import autotuner

    from helion_mlir_backend._compiler.mlir.autotune import AUTOTUNE_CACHE
    from helion_mlir_backend._compiler.mlir.autotune import CPU_AUTOTUNE_CACHE
    from helion_mlir_backend._compiler.mlir.autotune import MLIRAutotuneCache

    autotuner.cache_classes[AUTOTUNE_CACHE] = MLIRAutotuneCache
    autotuner.cache_classes[CPU_AUTOTUNE_CACHE] = MLIRAutotuneCache


def _allow_config_keys() -> None:
    """Let configs carry ``mlir_pipeline`` and the targets' lowering keys: Helion accepts only keys in ``VALID_KEYS``. Backends
    that do not support them drop them from configs."""
    import helion.autotuner.config_spec as config_spec

    from helion_mlir_backend._compiler.mlir.support.debug import PIPELINE_CONFIG_KEY
    from helion_mlir_backend._compiler.targets import target_config_keys

    config_spec.VALID_KEYS = (
        config_spec.VALID_KEYS | {PIPELINE_CONFIG_KEY} | target_config_keys()
    )


def _patch_bound_kernel(
    BoundKernel: type,
    MLIRBackend: type,
    mlir_compile_config: Callable[..., object],
) -> None:
    """Patch BoundKernel.compile_config to route MLIR-backend kernels through lighthouse."""
    if getattr(BoundKernel, "_helion_mlir_compile_config_patched", False):
        return

    _original = BoundKernel.compile_config

    def _compile_config(
        self: object,
        config: object = None,
        *,
        allow_print: bool = True,
    ) -> object:
        if isinstance(self.env.backend, MLIRBackend):
            return mlir_compile_config(self, config, allow_print=allow_print)
        return _original(self, config, allow_print=allow_print)

    BoundKernel.compile_config = _compile_config
    BoundKernel._helion_mlir_compile_config_patched = True
