"""The device-specific half of compiling and running a kernel.

Code generation is target-neutral: one Linalg-on-Tensors module serves every
device. An :class:`MLIRTarget` owns everything else -- the lighthouse lowering
pipelines, the JIT runner, the device checks and hooks around an entry call, and
what autotuning searches and caches. Exactly one target compiles a kernel.
"""

from __future__ import annotations

import abc
import ctypes
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Protocol

import mlir.ir as ir

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from helion._compiler.device_ir import DeviceIR
    from helion.autotuner.config_spec import ConfigSpec
    from lighthouse.execution.runner import Runner
    import torch


_THREAD_ATEXIT = "__cxa_thread_atexit_impl"


def exit_before_teardown() -> bool:
    """Makes C ``exit`` in this thread end the process at once, with status 1:
    ``_exit`` as the next thread-local destructor, and those run first and in
    reverse order, before ``atexit`` handlers and static destructors. False if
    this libc has no such registration (a glibc-private symbol)."""
    libc = ctypes.CDLL(None)
    register = getattr(libc, _THREAD_ATEXIT, None)
    if register is None:
        return False
    register.argtypes = [ctypes.c_void_p] * 3
    leave = ctypes.cast(libc._exit, ctypes.c_void_p).value
    # The object is ``_exit``'s argument; the symbol names the owning library.
    register(leave, 1, leave)
    return True


class Lowering(Protocol):
    """A lowering ready to apply (e.g. a lighthouse ``BackendDriver``)."""

    def apply(self, module: ir.Module) -> ir.Module: ...


class MLIRTarget(abc.ABC):
    """A device the MLIR backend compiles kernels for.

    Targets hold no per-kernel state: one instance serves every kernel.
    """

    name: ClassVar[str]
    """The target's name (``cpu``); also part of JIT and autotune cache keys."""
    pipelines: ClassVar[tuple[str, ...]]
    """The names of the target's lighthouse pipelines."""
    config_keys: ClassVar[frozenset[str]] = frozenset()
    """Config keys of the target's lowering, besides ``block_sizes`` and
    ``mlir_pipeline`` (other targets ignore them: a config can serve several)."""
    stop_points: ClassVar[tuple[str, ...]] = ()
    """The stages, in order, a lowering can stop after to inspect its IR; the last
    is the full lowering. Empty: only full lowerings."""

    @abc.abstractmethod
    def default_pipeline(self) -> str:
        """The pipeline of configs that name none (may read the environment)."""

    def resolve_pipeline(self, pipeline: str | None) -> str:
        """``pipeline``, or the default one for ``None``; ``ValueError`` if the
        target has no such pipeline."""
        if pipeline is None:
            return self.default_pipeline()
        if pipeline not in self.pipelines:
            raise ValueError(
                f"unknown pipeline {pipeline!r} for the MLIR target '{self.name}'; "
                f"expected one of {self.pipelines}"
            )
        return pipeline

    def default_stop_at(self) -> str | None:
        """Where lowerings that name no stop point stop (default: ``None``, the
        full lowering; may read the environment)."""
        return None

    def check_stop_at(self, stop_at: str | None) -> None:
        """Raise ``ValueError`` for a stop point the target has not."""
        if stop_at is not None and stop_at not in self.stop_points:
            expected = self.stop_points or "none"
            raise ValueError(
                f"unknown stop point {stop_at!r} for the MLIR target "
                f"'{self.name}'; expected one of {expected}"
            )

    def is_full(self, stop_at: str | None) -> bool:
        """Whether a lowering stopping at ``stop_at`` is complete (JIT-able)."""
        return stop_at is None or stop_at == self.stop_points[-1]

    def configure(self, module: ir.Module, config: Mapping[str, object]) -> None:
        """Record in ``module`` (generated with ``config``) what its
        :attr:`config_keys` ask of :meth:`lowering`, as module attributes: equal
        modules then lower equally (the JIT cache is keyed by module text).
        Raises ``MLIRBackendError`` for malformed values (default: nothing to
        record)."""
        return

    @abc.abstractmethod
    def lowering(
        self,
        module: ir.Module,
        entry: str,
        pipeline: str,
        stop_at: str | None = None,
    ) -> Lowering:
        """The lowering of ``module`` with ``pipeline``, down to the LLVM dialect
        or to the end of stage ``stop_at`` (one of :attr:`stop_points`).

        Called in ``module``'s context. Unknown pipelines raise ``ValueError``; a
        module the pipeline cannot compile may raise ``UnsupportedOperationError``.
        The result may be applied in a forked process: it must not need the threads
        of the calling one.
        """

    def lower(
        self,
        module: ir.Module,
        entry: str,
        pipeline: str,
        stop_at: str | None = None,
    ) -> ir.Module:
        """``module`` lowered with ``pipeline`` (to ``stop_at``), in this process
        with no time limit."""
        with module.context, ir.Location.unknown():
            return self.lowering(module, entry, pipeline, stop_at).apply(module)

    def enter_lowering_child(self) -> None:
        """Called in a forked lowering process before it lowers. ``LLVM ERROR`` ends
        in C ``exit``, whose destructors can't run in a fork: libomp's wait forever
        on threads that did not survive it or crash, so the error would arrive as a
        compile timeout or a signal, and the driver's reach the parent's GPU."""
        exit_before_teardown()

    @abc.abstractmethod
    def runner(self, module: ir.Module) -> Runner:
        """The lighthouse JIT runner of a lowered module."""

    @abc.abstractmethod
    def check_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Reject a tensor argument (named by its host expression) that the
        compiled kernel cannot access, e.g. one on another device."""

    def passes_in_place(self, tensor: torch.Tensor) -> bool:
        """Whether the kernel can take ``tensor`` as is, with no
        :meth:`check_tensor` error: a call whose tensors all pass skips those
        checks, and on a target that keeps the default no-op
        :meth:`prepare_call` and :meth:`finish_call` those hooks too (default:
        never)."""
        return False

    hooks_in_place: bool = False
    """Whether a call that passes as is still runs here if the target has
    :meth:`prepare_call` or :meth:`finish_call` hooks, which then get the
    arguments themselves (default: such a call takes the general path)."""

    native_in_place: bool = False
    """Whether the native call paths (``native/``) may be used. They take only
    plain dense CPU tensors (strided; no conjugate or negative bit,
    functionalization, functorch or Python wrapper), each of which must pass
    :meth:`passes_in_place`; any other tensor takes the Python paths
    (default: no native paths)."""

    phases_see_aliases: bool = False
    """Whether a kernel's loops read what earlier loops stored through an input
    sharing their elements, passed as it is: each launch takes device pointers. Other
    targets refuse such a kernel (``call_entry``)."""
    buffer_alignment: int = 1
    """The byte alignment the compiled kernels assume for the data of every
    tensor argument; ``call_entry`` passes a staged copy of a tensor whose data
    is not aligned (copied back for inouts). Default: no assumption."""

    def prepare_call(self, buffers: Sequence[torch.Tensor]) -> None:
        """Before an entry call: make ``buffers`` ready for the kernel (default:
        nothing to do)."""
        return

    def finish_call(self, buffers: Sequence[torch.Tensor]) -> None:
        """After an entry call: make the kernel's writes visible to the host
        (default: nothing to do)."""
        return

    @abc.abstractmethod
    def hardware_name(self) -> str:
        """The device model, keying autotuned configs."""

    def adjust_config_spec(self, config_spec: ConfigSpec, device_ir: DeviceIR) -> None:
        """Narrow the autotuning search space of a kernel to what the target
        compiles well, before a search (default: unchanged)."""
        return

    def validate_environment(self, stop_at: str | None = None) -> None:
        """Raise if this process cannot lower kernels for the target to
        ``stop_at``, or for ``None`` compile and run them (default: it can)."""
        return

    def __repr__(self) -> str:
        return f"<MLIR target {self.name!r}>"
