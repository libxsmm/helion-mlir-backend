"""Call path of a compiled kernel: host code, argument binding, entry call, return.

Every call runs the kernel's host code up to its device loops (``host_code.py``),
evaluates each entry argument's host expression in the host locals, calls the
memref-ABI entry (runtime scalars by value), then finishes the host code and returns its return value.
"""

from __future__ import annotations

import bisect
import difflib
from itertools import pairwise
from itertools import starmap
import keyword
import operator
from typing import TYPE_CHECKING
import warnings

import helion.autotuner.config_spec as helion_config_spec
import sympy
import torch

from .. import call_trace
from .. import native
from ..execution import compile_entry
from ..targets import resolve_target
from .analysis.signature import read_order
from .analysis.signature import scalar_indices
from .codegen import MLIRModuleBuilder
from .host_code import build_host_function
from .support import UnsupportedOperationError
from .support.debug import PIPELINE_CONFIG_KEY

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config
    from helion.runtime.kernel import BoundKernel

    from ..execution import CompiledEntry
    from ..execution import EntryArg
    from ..targets import MLIRTarget


_SPELLING_CUTOFF = 0.8
"""``difflib`` ratio from which an unknown config key counts as a misspelling. Every typo
tried of a valid key is above it (``block_size``, ``mlir_pipline``), and none of the keys that AI-bench and this repo's tests, examples and
tools pass or name; at 0.7 ``reduce_block_sizes`` would match."""
_warned_spellings: set[tuple[str, str]] = set()
"""``(kernel, key)`` pairs warned about, so a kernel warns once however often it compiles."""


def warn_misspelt_config_keys(
    kernel: str, config: Config, env: CompileEnvironment
) -> None:
    """Warn once per kernel and key about a config key that looks like a typo of a valid
    one, since the MLIR backend ignores keys it doesn't read. Any other unknown key stays
    silent and nothing raises: a kernel may carry keys meant for another backend."""
    valid = helion_config_spec.VALID_KEYS | env.config_spec.user_defined_tunables.keys()
    for key in config:
        if key in valid or (kernel, key) in _warned_spellings:
            continue
        close = difflib.get_close_matches(key, valid, n=1, cutoff=_SPELLING_CUTOFF)
        if close:
            _warned_spellings.add((kernel, key))
            warnings.warn(
                f"{kernel}: config key {key!r} is being ignored; did you mean {close[0]!r}?",
                stacklevel=2,
            )


def mlir_compile_config(
    bound_kernel: BoundKernel,
    config: Config | None = None,
    *,
    allow_print: bool = True,
) -> Callable[..., object]:
    """``BoundKernel.compile_config`` for the MLIR backend (one module per config)."""
    if config is None:
        config = bound_kernel._require_implicit_config()
    config = bound_kernel._normalize_config(config)
    if (run := bound_kernel._compile_cache.get(config)) is not None:
        return run
    warn_misspelt_config_keys(bound_kernel.host_function.name, config, bound_kernel.env)
    run = compile_kernel(bound_kernel.host_function, config, bound_kernel.env)
    bound_kernel._compile_cache[config] = run
    if call_trace.enabled():
        call_trace.note_compile(bound_kernel, config, run)
    return run


def compile_kernel(
    hf: HostFunction,
    config: Config,
    env: CompileEnvironment,
    *,
    target: str | None = None,
    pipeline: str | None = None,
) -> Callable[..., object]:
    """A callable with the kernel's own signature and Helion call semantics.

    ``target`` names the device target (default: ``HELION_MLIR_TARGET``, else
    the one for ``env.device``). ``pipeline`` defaults to the config's
    ``mlir_pipeline``, then the target's default (``HELION_MLIR_PIPELINE``, else
    ``opt`` on the CPU). The target's config keys
    apply to its lowering; other targets ignore them.
    """
    mlir_target = resolve_target(env.device, target)
    if pipeline is None:
        pipeline = config.get(PIPELINE_CONFIG_KEY)
    with env:
        builder = MLIRModuleBuilder(hf, config, env)
        module = builder.build()
        host_function = build_host_function(
            hf, builder.context.geometry.block_size, config
        )
        sizes = {
            name: builder.context.sizes.ref(name)
            for name in builder.context.signature.refs
        }
    mlir_target.configure(module, config)
    with env:
        order = read_order(hf, builder.context.signature, builder.context)
        indices = scalar_indices(hf, builder.context.signature)
    entry = compile_entry(
        module,
        hf.name,
        target=mlir_target,
        pipeline=pipeline,
        read_order=order,
        scalar_indices=indices,
        scalar_exprs=tuple(
            (scalar.expr, scalar.host_expr)
            for scalar in builder.context.signature.scalars.values()
        ),
    )
    host_globals = host_function.__globals__
    aliases = host_function._helion_mlir_aliases
    arg_exprs = [
        compile(arg.name, f"<helion-mlir-arg:{arg.name}>", "eval") for arg in entry.args
    ]
    # Only sizes computed from symbols need checks at run time.
    symbolic_sizes = (
        sizes
        if any(expr.free_symbols for exprs in sizes.values() for expr in exprs)
        else None
    )

    # An argument named by a host local is looked up; others are evaluated.
    local_names = [
        arg.name
        if arg.name.isidentifier() and not keyword.iskeyword(arg.name)
        else None
        for arg in entry.args
    ]
    arg_sources = list(zip(local_names, arg_exprs, strict=True))

    def run(*args: object) -> object:
        host = host_function(*args)
        try:
            local_vars = next(host)
            if aliases:
                local_vars.update(aliases)
            call_entry(
                entry,
                [
                    local_vars[name]
                    if name is not None and name in local_vars
                    else eval(expr, host_globals, local_vars)
                    for name, expr in arg_sources
                ],
                symbolic_sizes,
            )
            next(host)
        except StopIteration as done:
            return done.value
        raise AssertionError("the host function yields once")

    return _native_run(run, entry, host_function, arg_sources, symbolic_sizes)


def _native_run(
    run: Callable[..., object],
    entry: CompiledEntry,
    host_function: Callable[..., object],
    arg_sources: list[tuple[str | None, object]],
    symbolic_sizes: object,
) -> Callable[..., object]:
    """``run``, or a native ``KernelRun`` in front of it if the kernel
    qualifies: its first call records a plan (:func:`_plan`); later calls
    whose arguments match it run natively, all others take ``run``. The first
    call decides: one that can't be planned leaves every call on ``run``."""
    run.__name__ = entry.name  # names it in the call trace
    native_call = getattr(entry, "_native", None)
    if (
        native_call is None
        or symbolic_sizes is not None
        or not getattr(host_function, "_helion_mlir_metadata_only", False)
        or any(name is None for name, _ in arg_sources)
        or (extension := native.module()) is None
    ):
        call_trace.set_no_native(
            run, _no_native_reason(entry, host_function, arg_sources, symbolic_sizes)
        )
        return run
    names = [name for name, _ in arg_sources]
    defaults = (
        *(host_function.__defaults__ or ()),
        *(host_function.__kwdefaults__ or {}).values(),
    )

    def planner(native_run: object, *args: object) -> object:
        return _plan(
            native_run, run, entry, native_call, host_function, names, defaults, args
        )

    return extension.KernelRun(run, planner)


def _no_native_reason(
    entry: CompiledEntry,
    host_function: Callable[..., object],
    arg_sources: list[tuple[str | None, object]],
    symbolic_sizes: object,
) -> str:
    """Why a compiled kernel gets no ``KernelRun``, for :mod:`~..call_trace`. Loads
    nothing: a target without native paths must not bring the extension in."""
    if not entry.target.native_in_place:
        return f"target {type(entry.target).__name__} has no native call path"
    if not native.enabled():
        return "native calls are off (set_native_call, HELION_MLIR_NATIVE_CALL)"
    if (error := native.load_error()) is not None:
        return f"the native extension is unavailable: {error}"
    if getattr(entry, "_native", None) is None:
        static = getattr(entry, "_static", None)
        if static is None:
            return "an entry argument has no static shape"
        if static.scalars:
            names = ", ".join(str(host) for _, host in entry.scalar_exprs)
            return f"the entry takes run-time scalars ({names})"
        return "the entry has no native call"
    if symbolic_sizes is not None:
        return "array sizes are computed from symbols and checked per call"
    if not getattr(host_function, "_helion_mlir_metadata_only", False):
        refusal = getattr(host_function, "_helion_mlir_host_refusal", None)
        return f"host code does more than read metadata: {refusal}"
    names = [str(expr) for name, expr in arg_sources if name is None]
    return f"an entry argument is not a host local: {', '.join(names)}"


_ScalarTypes = (int, float, bool, str, type(None))


def _plan(
    native_run: object,
    run: Callable[..., object],
    entry: CompiledEntry,
    native_call: object,
    host_function: Callable[..., object],
    names: list[str],
    defaults: tuple[object, ...],
    args: tuple[object, ...],
) -> object:
    """The first call: as ``run``, recording which argument or fresh host
    tensor each entry argument is and what is returned. Installs that plan in
    ``native_run`` if each entry argument and the result is exactly one
    argument (no two arguments the same object) or a fresh contiguous CPU
    tensor (not a default argument), tensor arguments are plain CPU tensors
    and other arguments immutable scalars, scalar ``hl.constexpr`` values or
    functions. Without a plan, every call takes ``run``."""
    why: list[str] = []
    try:
        params = _plan_params(args, why)
    except Exception as exc:
        params = None
        why.append(f"reading the arguments raised {type(exc).__name__}: {exc}")
    if params is None:
        call_trace.set_no_plan(run, why[0] if why else "the arguments can't be planned")
        return run(*args)
    host = host_function(*args)
    try:
        local_vars = next(host)
        if not all(name in local_vars for name in names):
            # Not a host local after all (run evaluates it): no plan.
            host.close()
            missing = [name for name in names if name not in local_vars]
            call_trace.set_no_plan(run, f"not host locals: {', '.join(missing)}")
            return run(*args)
        values = [local_vars[name] for name in names]
        call_entry(entry, values)
        next(host)
    except StopIteration as done:
        result = done.value
    else:
        raise AssertionError("the host function yields once")
    plan = _plan_sources(args, values, result, defaults, names, why)
    if plan is None:
        call_trace.set_no_plan(run, why[0] if why else "the plan's sources")
    else:
        from ..targets.base import MLIRTarget

        hooks = (type(entry.target), MLIRTarget.prepare_call, MLIRTarget.finish_call)
        native_run.install(native_call, hooks, params, *plan)  # type: ignore[attr-defined]
        call_trace.set_plan_params(run, params)
    return result


def _plan_params(
    args: tuple[object, ...], why: list[str] | None = None
) -> list[tuple[object, ...]] | None:
    """Per argument ``(kind, dtype or value, sizes, strides)``: 0 a plain CPU
    tensor, 1 an object compared by identity (a builtin function), 2 a scalar
    (type and ``==``, floats by their bits), 3 an ``hl.constexpr`` of a scalar
    (its type, and its value as 2: a helper may make a new one per call), 4 a
    function ``(4, function, cell kinds, cell contents)``: its code and its
    closure's contents when planned, each as 1-3 (as Helion keys functions:
    another instance of a model has its own epilogue closures); ``None``: no
    plan, with the reason appended to ``why``."""
    import types

    from helion.language.constexpr import ConstExpr

    def refuse(index: int, reason: str) -> None:
        if why is not None:
            why.append(f"argument {index}: {reason}")

    params: list[tuple[object, ...]] = []
    for index, arg in enumerate(args):
        if type(arg) in (torch.Tensor, torch.nn.Parameter):
            tensor: torch.Tensor = arg  # type: ignore[assignment]
            if tensor.device.type != "cpu" or tensor.layout != torch.strided:
                refuse(index, f"a {tensor.device.type} or {tensor.layout} tensor")
                return None
            params.append((0, tensor.dtype, tuple(tensor.shape), tensor.stride()))
        elif type(arg) in _ScalarTypes:
            if arg != arg:  # NaN never matches: no plan
                refuse(index, "NaN")
                return None
            params.append((2, arg, (), ()))
        elif type(arg) is ConstExpr and type(arg.value) in _ScalarTypes:  # type: ignore[attr-defined]
            if arg.value != arg.value:  # type: ignore[attr-defined]
                refuse(index, "hl.constexpr(NaN)")
                return None
            params.append((3, arg, (), ()))
        elif isinstance(arg, types.FunctionType):
            try:
                contents = tuple(cell.cell_contents for cell in arg.__closure__ or ())
            except ValueError:  # an empty cell
                refuse(index, "a function with an empty closure cell")
                return None
            params.append((4, arg, tuple(map(_value_kind, contents)), contents))
        elif isinstance(arg, types.BuiltinFunctionType):
            params.append((1, arg, (), ()))
        else:
            refuse(index, f"a {type(arg).__name__}")
            return None
    return params


def _value_kind(value: object) -> int:
    """How a closure cell's content is compared (as :func:`_plan_params`)."""
    from helion.language.constexpr import ConstExpr

    if type(value) in _ScalarTypes and value == value:
        return 2
    if type(value) is ConstExpr and type(value.value) in _ScalarTypes:  # type: ignore[attr-defined]
        return 3 if value.value == value.value else 1  # type: ignore[attr-defined]
    return 1


def _plan_sources(
    args: tuple[object, ...],
    values: list[object],
    result: object,
    defaults: tuple[object, ...],
    names: list[str] | None = None,
    why: list[str] | None = None,
) -> (
    tuple[
        list[tuple[int, int]],
        list[tuple[torch.dtype, tuple[int, ...]]],
        tuple[int, int],
    ]
    | None
):
    """``(sources, fresh, result)`` for ``KernelRun.install``: per entry
    argument and for the result, ``(0, argument index)`` or ``(1, fresh
    index)`` (``(0, -1)``: returns None); ``None``: no plan, with the reason
    appended to ``why``."""
    from ..execution import _contiguous_strides

    fresh: list[torch.Tensor] = []
    arg_storages = {
        arg.untyped_storage().data_ptr()
        for arg in args
        if isinstance(arg, torch.Tensor)
    }

    def source(value: object) -> tuple[int, int] | None:
        matches = [index for index, arg in enumerate(args) if value is arg]
        if matches:
            # The same object for two arguments: a later call may pass two.
            return (0, matches[0]) if len(matches) == 1 else None
        for index, tensor in enumerate(fresh):
            if value is tensor:
                return (1, index)
        if (
            type(value) is torch.Tensor
            and not any(value is default for default in defaults)
            and value.device.type == "cpu"
            and value.layout == torch.strided
            # The strides at::empty gives (torch's contiguity ignores the
            # strides of size-1 dims).
            and value.stride() == _contiguous_strides(value.shape)
            and value.storage_offset() == 0
            and value.untyped_storage().data_ptr() not in arg_storages
            and not value.requires_grad
        ):
            fresh.append(value)
            return (1, len(fresh) - 1)
        return None

    def refused(what: str, value: object) -> str:
        if sum(value is arg for arg in args) > 1:
            return f"{what}: the same object is passed for two arguments"
        return (
            f"{what}: neither exactly one argument nor a fresh contiguous CPU "
            f"tensor at offset 0 (e.g. a view, or a default argument)"
        )

    sources = [source(value) for value in values]
    result_source = (0, -1) if result is None else source(result)
    if result_source is None or any(s is None for s in sources):
        if why is not None:
            why.extend(
                refused(f"entry argument {names[i] if names else i}", value)
                for i, (value, s) in enumerate(zip(values, sources, strict=True))
                if s is None
            )
            if result_source is None:
                why.append(refused("the result", result))
        return None
    return (
        sources,  # type: ignore[return-value]
        [(t.dtype, tuple(t.shape)) for t in fresh],
        result_source,
    )


def call_entry(
    entry: CompiledEntry,
    values: list[object],
    sizes: dict[str, list[sympy.Expr]] | None = None,
) -> list[torch.Tensor]:
    """Call ``entry`` with one value per argument; returns the inout tensors.

    The entry's target checks each tensor (e.g. its device) and prepares and
    finishes the call. Tensors are passed contiguous (a copy for strided ones,
    copied back for inouts) and with their data aligned to the target's
    ``buffer_alignment`` (a copy likewise). An input that overlaps an inout is
    cloned so the ``restrict`` arguments hold, unless it reads what the kernel wrote
    earlier (:func:`_reads_earlier_writes`); overlapping inouts are rejected. Runtime
    scalars are passed by value. ``sizes`` (each argument's size expressions) lets
    runtime sizes the kernel assumes equal be checked. Calls that need none of this
    (:meth:`CompiledEntry.call_in_place`) skip it: per-call host time is a large
    part of a small kernel's time.
    """
    if entry.scalar_indices:
        _check_scalar_indices(entry, values)
    if sizes is None and (inouts := entry.call_in_place(values)) is not None:
        return inouts
    target = entry.target
    tensors = [
        _checked_tensor(target, arg, value)
        for arg, value in zip(entry.args, values, strict=True)
        if arg.role != "scalar"
    ]
    if sizes is not None:
        _check_sizes(
            [
                (arg.name, value)
                for arg, value in zip(entry.args, values, strict=True)
                if arg.role != "scalar"
            ],
            sizes,
        )
    roles = [arg.role for arg in entry.args if arg.role != "scalar"]
    inouts = [t for t, role in zip(tensors, roles, strict=True) if role == "inout"]
    inout_spans = sorted(_byte_span(t) for t in inouts if t.numel())
    if any(starmap(_spans_overlap, pairwise(inout_spans))):
        raise UnsupportedOperationError(
            "kernel writes two tensors that share memory",
            reason="each written tensor must be its own buffer",
        )
    tensor_args = [arg for arg in entry.args if arg.role != "scalar"]
    written = sorted(
        (
            (_byte_span(tensor), arg.name, tensor)
            for arg, tensor, role in zip(tensor_args, tensors, roles, strict=True)
            if role == "inout" and tensor.numel()
        ),
        key=operator.itemgetter(0),
    )
    passed_as_is = [
        role == "in"
        and _overlaps_any(tensor, inout_spans)
        and _reads_earlier_writes(entry, arg.name, tensor, written)
        for arg, tensor, role in zip(tensor_args, tensors, roles, strict=True)
    ]
    staged = [
        _aligned(
            # A negative or conjugate view's storage holds the values unnegated
            # (unconjugated): the kernel gets a materialized copy (written back
            # for inouts), which never aliases an inout.
            tensor.resolve_conj().resolve_neg().contiguous()
            if tensor.is_neg() or tensor.is_conj()
            else tensor.clone(memory_format=torch.contiguous_format)
            if role == "in" and not as_is and _overlaps_any(tensor, inout_spans)
            else tensor.contiguous(),
            target.buffer_alignment,
        )
        for tensor, role, as_is in zip(tensors, roles, passed_as_is, strict=True)
    ]
    buffers = iter(staged)
    arguments = [
        value if arg.role == "scalar" else next(buffers)
        for arg, value in zip(entry.args, values, strict=True)
    ]
    target.prepare_call(staged)
    entry(arguments)
    target.finish_call(staged)
    if all(map(operator.is_, staged, tensors)):
        entry.note_passed_as_is(staged)
    for tensor, copy, role in zip(tensors, staged, roles, strict=True):
        if role == "inout" and copy is not tensor:
            tensor.copy_(copy)
    return inouts


def _check_scalar_indices(entry: CompiledEntry, values: list[object]) -> None:
    """Raise ``IndexError`` for an access at a runtime position out of its tensor's
    bounds, as torch does: the kernels don't check, and a position out of bounds
    stores outside the tensor. Negative positions count from the end (the lowering's
    ``_from_the_end``). A reshaped alias of another argument takes its static size.
    A tensor with no elements is skipped: nothing can be stored outside it, and
    ``c[n - 1, t]`` at ``n = 0`` is right. Conservative: an access in a loop that runs
    0 times, or under a guard or mask that stays false, is still checked; a position
    out of bounds there raises, though it wouldn't run."""
    by_name = dict(zip(entry.arg_names, values, strict=True))
    for tensor, dim, size, position, operands in entry.index_checks:
        value = by_name.get(tensor)
        if value is not None and not value.numel():
            continue
        if size is None:
            size = by_name[tensor].shape[dim]
        elif not size:
            continue
        at = (
            by_name[position]
            if operands is None
            else position(*map(by_name.get, operands))
        )
        if not -size <= at < size:
            raise IndexError(
                f"index {at} is out of bounds for dimension {dim} of '{tensor}' "
                f"with size {size}"
            )


def _reads_earlier_writes(
    entry: CompiledEntry,
    name: str,
    tensor: torch.Tensor,
    written: list[tuple[tuple[int, int], str, torch.Tensor]],
) -> bool:
    """Whether input ``tensor``, whose bytes overlap a written tensor, must be passed
    as it is: it shares elements with a tensor stored before it is read, and a copy
    would not hold what was stored. A kernel whose reads the target can't honour is
    refused: never a stale read. No such store: ``False``, its value at the call."""
    order = entry.read_order
    if order is None:
        return False
    after = order.after.get(name, frozenset())
    if not after:
        return False
    span = _byte_span(tensor)
    start = bisect.bisect_left(written, span, key=operator.itemgetter(0))
    near = []
    for stored_span, other, stored in written[max(start - 1, 0) :]:
        if stored_span[0] >= span[1]:
            break
        near.append((other, stored))
    if not any(
        other in after and _share_elements(tensor, stored) for other, stored in near
    ):
        return False
    shared = [
        (other, stored) for other, stored in near if _share_elements(tensor, stored)
    ]
    same = order.same.get(name, frozenset())
    alignment = entry.target.buffer_alignment
    if (
        entry.target.phases_see_aliases
        and _passes_as_is(tensor, alignment)
        and all(_passes_as_is(other, alignment) for _, other in shared)
        and not any(other in same for other, _ in shared)
    ):
        return True
    raise UnsupportedOperationError(
        "a tensor read after the kernel wrote into its elements",
        reason=(
            f"'{name}' shares elements with a tensor the kernel stores before it "
            "reads it: on this target, or with this view's strides, alignment or a "
            "store in the same loop, the read can't see that store"
        ),
        alternatives=["two kernel calls, the second one reading the view"],
    )


def _passes_as_is(tensor: torch.Tensor, alignment: int) -> bool:
    """``call_entry`` passes ``tensor``'s own storage (no staged copy)."""
    return (
        tensor.is_contiguous()
        and not tensor.is_neg()
        and not tensor.is_conj()
        and (alignment <= 1 or tensor.data_ptr() % alignment == 0)
    )


def _share_elements(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors share an element (rare path: only for tensors whose byte
    spans overlap). Different element sizes count as shared."""
    if not a.numel() or not b.numel():
        return False
    if not _spans_overlap(_byte_span(a), _byte_span(b)):
        return False
    if a.element_size() != b.element_size():
        return True
    if a.is_contiguous() and b.is_contiguous():
        return True

    def elements(tensor: torch.Tensor) -> torch.Tensor:
        offsets = torch.zeros((), dtype=torch.int64)
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
            steps = torch.arange(size) * stride * tensor.element_size()
            offsets = offsets.unsqueeze(-1) + steps
        return offsets.flatten() + tensor.data_ptr()

    return bool(torch.isin(elements(a), elements(b)).any())


def _aligned(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    """``tensor``, or a copy of it whose data starts at a multiple of
    ``alignment`` bytes (the caching allocators return aligned blocks)."""
    if alignment <= 1 or not tensor.numel() or tensor.data_ptr() % alignment == 0:
        return tensor
    copy = tensor.clone(memory_format=torch.contiguous_format)
    if copy.data_ptr() % alignment:
        raise UnsupportedOperationError(
            f"cannot pass a {alignment}-byte aligned copy of a tensor",
            reason=f"the allocator returned data at {copy.data_ptr():#x}",
        )
    return copy


def _checked_tensor(target: MLIRTarget, arg: EntryArg, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"'{arg.name}' must be a torch.Tensor, got {type(value).__name__}"
        )
    target.check_tensor(arg.name, value)
    if (
        value.dim() != len(arg.shape)
        or value.dtype != arg.dtype
        or any(
            expected != -1 and expected != actual
            for expected, actual in zip(arg.shape, value.shape, strict=False)
        )
    ):
        expected = tuple("?" if size == -1 else size for size in arg.shape)
        raise ValueError(
            f"'{arg.name}' has shape {tuple(value.shape)} and dtype {value.dtype}; the "
            f"compiled kernel expects {expected} and {arg.dtype}"
        )
    return value


def _check_sizes(
    tensors: list[tuple[str, torch.Tensor]], sizes: dict[str, list[sympy.Expr]]
) -> None:
    """Runtime sizes with one symbol are equal, and sizes computed from symbols
    (``n // 2``) match them."""
    bound: dict[sympy.Symbol, int] = {}
    computed = []
    for name, tensor in tensors:
        for dim, expr in enumerate(sizes.get(name, ())):
            if not expr.free_symbols:
                continue
            actual = tensor.shape[dim]
            if isinstance(expr, sympy.Symbol):
                expected = bound.setdefault(expr, actual)
                if expected != actual:
                    raise ValueError(
                        f"'{name}' has size {actual} in dim {dim}, but the kernel was "
                        f"compiled for the same size as another argument's ({expected})"
                    )
            else:
                computed.append((name, dim, expr, actual))
    for name, dim, expr, actual in computed:
        if expr.free_symbols <= bound.keys():
            expected = int(expr.xreplace(bound))
            if expected != actual:
                raise ValueError(
                    f"'{name}' has size {actual} in dim {dim}; the kernel computes "
                    f"it as {expr} = {expected}"
                )


def _overlaps_any(tensor: torch.Tensor, spans: list[tuple[int, int]]) -> bool:
    """``spans``: sorted and disjoint, as ``call_entry`` checks the written tensors'.
    A kernel of many tensors: a search per input, not a scan."""
    if not spans or tensor.numel() == 0:
        return False
    span = _byte_span(tensor)
    index = bisect.bisect_left(spans, span)
    near = spans[max(index - 1, 0) : index + 1]
    return any(_spans_overlap(span, other) for other in near)


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _byte_span(t: torch.Tensor) -> tuple[int, int]:
    """The bytes ``t`` spans in memory; ``t`` is not empty."""
    start = t.data_ptr()
    if t.is_contiguous():
        return start, start + t.numel() * t.element_size()
    last = sum(
        (size - 1) * stride for size, stride in zip(t.shape, t.stride(), strict=True)
    )
    return start, start + (last + 1) * t.element_size()


def prepared_call(
    bound_kernel: BoundKernel,
    args: Sequence[object],
    varying: Sequence[int],
    fallback: Callable[..., object],
    *,
    views: Sequence[Sequence[int] | None] | None = None,
    result_shape: Sequence[int] | None = None,
) -> Callable[..., object] | None:
    """``lambda *v: bound_kernel(*args with v at the varying positions)``,
    natively: a call checks the varying tensors (and everything else the
    native run checks: the bound tensors' metadata, their current data, that
    ``bound_kernel`` still runs the same compiled kernel) and calls the
    kernel; any other call returns ``fallback(*v)``. ``None`` if the kernel
    has no native run (not static, host code not metadata-only, no
    extension).

    ``views[k]`` (optional, per varying argument): the caller passes a
    contiguous tensor of that shape, standing for its contiguous view of the
    argument's shape (e.g. ``x`` for ``x.view(m, k // 2, 2)``).
    ``result_shape``: the result, a fresh tensor of the host code, is
    allocated with that shape (as ``result.view(result_shape)``). Neither may
    change the element count. The first native call happens after the kernel
    ran once (its plan is recorded on its first call)."""
    if bound_kernel._run is None:
        bound_kernel.ensure_config_exists(tuple(args))
    run = bound_kernel._run
    if native.module() is None or type(run).__name__ != "KernelRun":
        return None
    varying = list(varying)
    bound = tuple(None if i in varying else arg for i, arg in enumerate(args))
    return run.prepare(  # type: ignore[union-attr]
        bound_kernel,
        bound,
        varying,
        [None if v is None else tuple(v) for v in views]
        if views is not None
        else [None] * len(varying),
        None if result_shape is None else tuple(result_shape),
        fallback,
    )
