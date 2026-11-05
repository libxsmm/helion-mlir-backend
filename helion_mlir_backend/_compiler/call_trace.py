"""Why a call of an mlir kernel ran the way it did.

:func:`explain` says what a call with given arguments would do: which dispatch,
which run, and which check would refuse it. It runs and compiles nothing.

``HELION_MLIR_TRACE_CALLS=1`` (read once, at import) prints to stderr the same
for every call through a kernel or bound kernel, once per kernel, compile, path
and reason, each compile's outcome, and at exit the counts per compiled kernel.
Calls of a ``prepared_call`` show up in the counts only.

Unset, tracing patches nothing, and a native call does no work for it: the native
run keeps its records (``KernelRun.refusals``, ``.last_refusal``) on the path that
leaves it, and this module's records are set once per compile.
"""

from __future__ import annotations

import atexit
import dataclasses
import os
import sys
import threading
from typing import TYPE_CHECKING

import torch

from . import native

if TYPE_CHECKING:
    from collections.abc import Callable

TRACE_CALLS_ENV = "HELION_MLIR_TRACE_CALLS"

_REFUSALS = (
    "the number of arguments differs from the plan's",
    "a torch dispatch or torch-function mode is active",
    "the target's call hooks were replaced",
    "an argument differs from the plan's",
    "an entry argument is unaligned or differs from the entry's",
    "an output shares memory with another argument",
)
_NATIVE_RUN = "_helion_mlir_native_run"


def _from_environment() -> bool:
    value = os.environ.get(TRACE_CALLS_ENV, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


_enabled = _from_environment()
_compiled: list[tuple[str, object, object]] = []
_seen: set[tuple[object, ...]] = set()
_local = threading.local()


def enabled() -> bool:
    """Whether ``HELION_MLIR_TRACE_CALLS`` was set at import."""
    return _enabled


# -- records of a compile and its first call: on the Python run ---------------


def set_no_native(run: Callable[..., object], reason: str) -> None:
    """Why this compile got no native run; called at compile time."""
    run._helion_mlir_no_native = reason  # type: ignore[attr-defined]


def set_no_plan(run: Callable[..., object], reason: str) -> None:
    """Why the first call recorded no plan: every call takes the general path."""
    run._helion_mlir_no_plan = reason  # type: ignore[attr-defined]
    if _enabled:
        _print(f"{run.__name__}: no plan: {reason}")


def set_plan_params(run: Callable[..., object], params: list[tuple]) -> None:
    """The plan's parameters, which name what a refused argument differs in."""
    run._helion_mlir_plan_params = params  # type: ignore[attr-defined]


# -- explain ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CallExplanation:
    """What a call would do, see :func:`explain`.

    ``dispatch``: ``fast`` (the kernel's dispatch cache has this key), ``bind``
    (the full key: an existing or a new bound kernel) or ``direct`` (a bound
    kernel was given). ``run``: ``native``, ``plan`` (the first call of a native
    run: on the general path, it records the plan), ``general`` (the Python path of
    the compile), or ``compile`` (no compile yet). ``reason``: why not ``native``.
    """

    kernel: str
    backend: str | None
    dispatch: str
    dispatch_reason: str | None
    run: str
    reason: str | None
    config: object
    planned: bool
    native_calls: int
    refusals: dict[str, int]
    last_refusal: str | None

    def __str__(self) -> str:
        lines = [f"{self.kernel}: dispatch {self.dispatch}, run {self.run}"]
        if self.dispatch_reason:
            lines.append(f"  dispatch: {self.dispatch_reason}")
        if self.reason:
            lines.append(f"  run: {self.reason}")
        if self.config is not None:
            lines.append(f"  config: {self.config}")
        lines.append(
            f"  planned: {self.planned}, native calls so far: {self.native_calls}"
        )
        for why, count in self.refusals.items():
            lines.append(f"  refused {count}x: {why}")
        return "\n".join(lines)


def _describe_param(param: tuple) -> str:
    kind, value, sizes, strides = param
    if kind == 0:
        return f"{value} tensor of shape {sizes}, strides {strides}"
    if kind == 4:
        return f"function {value.__qualname__} with the planned cell values"
    return f"{value!r}"


def _describe_value(value: object) -> str:
    if isinstance(value, torch.Tensor):
        return (
            f"{type(value).__name__} {value.dtype} of shape {tuple(value.shape)}, "
            f"strides {value.stride() if value.layout == torch.strided else '?'}"
        )
    return f"{value!r}"


def _describe_function_difference(param: tuple, value: object) -> str | None:
    """Which cell of a function differs from the plan, if it is one."""
    planned, cells = param[1], param[3]
    if not hasattr(value, "__code__"):
        return None
    if value.__code__ != planned.__code__:
        return "other code than the planned function"
    names = value.__code__.co_freevars
    for name, cell, old in zip(names, value.__closure__ or (), cells, strict=False):
        new = cell.cell_contents
        if new is not old and (type(new) is not type(old) or repr(new) != repr(old)):
            return f"cell value {name}: {new!r}, planned {old!r}"
    return "a cell differs from the planned one (compared by identity)"


def _refusal_text(
    general: object, refusal: tuple[int, int], args: tuple[object, ...]
) -> str:
    what, index = refusal
    text = _REFUSALS[what]
    params = getattr(general, "_helion_mlir_plan_params", None)
    if what != 3 or params is None or not 0 <= index < len(args):
        return f"{text} (index {index})" if index >= 0 else text
    param, value = params[index], args[index]
    if param[0] == 4 and (detail := _describe_function_difference(param, value)):
        return f"argument {index}: {detail}"
    return (
        f"argument {index}: {_describe_value(value)}, planned {_describe_param(param)}"
    )


def _fast_key_refusal(kernel: object, args: tuple[object, ...]) -> str:
    """Why this call has no fast key."""
    if not any(isinstance(arg, torch.Tensor) for arg in args):
        return "no tensor argument to pin the device"
    return "an argument the fast key doesn't handle: " + ", ".join(
        type(arg).__name__
        for arg in args
        if not isinstance(arg, (torch.Tensor, int, float, str, type(None)))
    )


def _pure_fast_key(kernel: object, args: tuple[object, ...]) -> object:
    """The fast key of a call, changing no cache."""
    method = type(kernel)._fast_dispatch_key  # type: ignore[attr-defined]
    return getattr(method, "_helion_mlir_pure", method)(kernel, args)


def _find_bound(kernel: object, args: tuple[object, ...]) -> object | None:
    """The bound kernel that ``bind`` would return, if it exists: makes none."""
    for candidate in (args, kernel.normalize_args(*args)):  # type: ignore[attr-defined]
        signature = kernel._base_specialization_key(candidate)  # type: ignore[attr-defined]
        key = kernel._get_bound_kernel_cache_key(candidate, signature)  # type: ignore[attr-defined]
        if key is not None and (bound := kernel._bound_kernels.get(key)) is not None:  # type: ignore[attr-defined]
            return bound
    return None


def _explain_run(bound: object, args: tuple[object, ...]) -> tuple[str, str | None]:
    run = bound._run  # type: ignore[attr-defined]
    if run is None:
        return "compile", "the bound kernel has no compile yet"
    if type(run).__name__ != "KernelRun":
        reason = getattr(run, "_helion_mlir_no_native", "not an mlir compile")
        return "general", f"no native run: {reason}"
    general = run.general
    if not run.planned:
        no_plan = getattr(general, "_helion_mlir_no_plan", None)
        if no_plan is not None:
            return "general", f"no plan: {no_plan}"
        return "plan", "the first call runs on the general path and records the plan"
    if not native.enabled():
        return "general", "native calls are off (set_native_call)"
    refusal = run.check(*args)
    if refusal is None:
        return "native", None
    return "general", _refusal_text(general, refusal, args)


def explain(kernel: object, *args: object, **kwargs: object) -> CallExplanation:
    """What ``kernel(*args, **kwargs)`` would do now, for a ``helion.kernel`` or a
    bound kernel: runs and compiles nothing, changes no cache. Allocation-dependent
    checks (an output's alignment and aliasing) can't be known before the call; a
    call they refuse shows in ``refusals`` and ``last_refusal`` afterwards."""
    from helion.runtime.kernel import BoundKernel

    if isinstance(kernel, BoundKernel):
        bound, dispatch, dispatch_reason = kernel, "direct", None
        owner = kernel.kernel
    else:
        owner = kernel
        bound = None
        dispatch, dispatch_reason = "bind", None
        if kwargs:
            args = kernel.normalize_args(*args, **kwargs)  # type: ignore[attr-defined]
            dispatch_reason = "keyword arguments take bind()"
        else:
            fast_key = _pure_fast_key(kernel, args)
            if fast_key is None:
                dispatch_reason = _fast_key_refusal(kernel, args)
            elif (hit := kernel._dispatch_cache.get(fast_key)) is not None:  # type: ignore[attr-defined]
                if hit._run is not None:
                    bound, dispatch = hit, "fast"
            else:
                dispatch_reason = "first call with this key"
        if bound is None:
            bound = _find_bound(kernel, args)
    name = owner.name  # type: ignore[attr-defined]
    backend = owner.settings.backend  # type: ignore[attr-defined]
    if bound is None:
        return CallExplanation(
            name, backend, dispatch, dispatch_reason, "compile",
            "no bound kernel yet: the call binds and compiles",
            None, False, 0, {}, None,
        )  # fmt: skip
    run_kind, reason = _explain_run(bound, args)
    run = bound._run  # type: ignore[attr-defined]
    native_run = type(run).__name__ == "KernelRun"
    last = run.last_refusal if native_run else None
    return CallExplanation(
        name,
        backend,
        dispatch,
        dispatch_reason,
        run_kind,
        reason,
        bound._config,  # type: ignore[attr-defined]
        native_run and run.planned,
        run.native_calls if native_run else 0,
        {
            why: count
            for why, count in zip(_REFUSALS, run.refusals, strict=True)
            if count
        }
        if native_run
        else {},
        None
        if last is None
        else (
            f"{_REFUSALS[last[0]]} (index {last[1]})"
            if last[1] >= 0
            else _REFUSALS[last[0]]
        ),
    )


# -- HELION_MLIR_TRACE_CALLS --------------------------------------------------


def _print(text: str) -> None:
    sys.stderr.write(f"[helion-mlir] {text}\n")


def note_compile(bound: object, config: object, run: object) -> None:
    """A compile happened, tracing on."""
    name = bound.kernel.name  # type: ignore[attr-defined]
    _compiled.append((name, config, run))
    if type(run).__name__ == "KernelRun":
        _print(f"{name}: compiled, native run {config}")
    else:
        reason = getattr(run, "_helion_mlir_no_native", None)
        _print(f"{name}: compiled, no native run: {reason} {config}")


def _trace_call(owner: object, kernel: object, args: tuple, call: Callable) -> object:
    """``call()``, reporting the first call of each path and reason."""
    if getattr(_local, "depth", 0):
        return call()
    try:
        before = explain(kernel, *args)
    except Exception:
        return call()
    _local.depth = 1
    try:
        result = call()
    finally:
        _local.depth = 0
    try:
        after = explain(kernel, *args)
    except Exception:
        return result
    ran_native = after.native_calls > before.native_calls
    path = "native" if ran_native else before.run
    reason = None if ran_native else before.reason
    if before.run == "native" and not ran_native:  # refused after the argument checks
        path, reason = "general", after.last_refusal
    # The reason's kind, not the values it names: one line per kernel, compile, path
    # and reason, however many values go by.
    kind = None if reason is None else reason.split(":", 1)[0]
    event = (id(owner), str(after.config), before.dispatch, path, kind)
    if event not in _seen:
        _seen.add(event)
        line = f"{before.kernel}: dispatch {before.dispatch}, run {path}"
        if before.dispatch_reason:
            line += f" (dispatch: {before.dispatch_reason})"
        if reason:
            line += f" ({reason})"
        _print(line)
    return result


def _summary() -> None:
    for name, config, run in _compiled:
        if type(run).__name__ != "KernelRun":
            _print(f"at exit: {name} {config}: no native run")
            continue
        counts = ", ".join(
            f"{why}: {count}"
            for why, count in zip(_REFUSALS, run.refusals, strict=True)
            if count
        )
        _print(
            f"at exit: {name} {config}: {run.native_calls} native calls"
            + (f"; refused: {counts}" if counts else "")
        )


def install(MLIRBackend: type) -> None:
    """With tracing on, report the calls of mlir kernels. Off: nothing is patched."""
    if not _enabled:
        return
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

    if getattr(Kernel.__call__, "_helion_mlir_traced", False):
        return
    backend_name = MLIRBackend().name
    kernel_call, bound_call = Kernel.__call__, BoundKernel.__call__

    def traced_kernel(self: Kernel, *args: object, **kwargs: object) -> object:
        if self.settings.backend != backend_name:
            return kernel_call(self, *args, **kwargs)
        shown = self.normalize_args(*args, **kwargs) if kwargs else args
        return _trace_call(
            self, self, shown, lambda: kernel_call(self, *args, **kwargs)
        )

    def traced_bound(self: BoundKernel, *args: object) -> object:
        if self.kernel.settings.backend != backend_name:
            return bound_call(self, *args)
        return _trace_call(self, self, args, lambda: bound_call(self, *args))

    traced_kernel._helion_mlir_traced = True  # type: ignore[attr-defined]
    Kernel.__call__ = traced_kernel
    BoundKernel.__call__ = traced_bound
    atexit.register(_summary)
