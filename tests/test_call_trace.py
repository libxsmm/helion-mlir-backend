"""``explain`` and ``HELION_MLIR_TRACE_CALLS``: what a call of an mlir kernel does
and why. Marked ``helion_internals``: this reads the native run's records."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import helion
import helion.language as hl
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import Kernel
import pytest
import torch

import helion_mlir_backend
from helion_mlir_backend import explain
from helion_mlir_backend._compiler import native

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.helion_internals,
    pytest.mark.skipif(
        native.module() is None, reason="the native extension could not be built"
    ),
]


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] + y[tile]
    return out


def _apply(x: torch.Tensor, fn: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = fn(x[tile])
    return out


def _times(x: torch.Tensor, alpha: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] * alpha
    return out


def _torch_in_host(x: torch.Tensor) -> torch.Tensor:
    y = x + x
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = y[tile]
    return out


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(fn, backend="mlir", static_shapes=True, config=_cfg(8))


def test_the_paths_of_a_plain_kernel_in_order() -> None:
    kernel = _kernel(_add)
    x, y = torch.ones(32), torch.ones(32)

    first = explain(kernel, x, y)
    assert (first.dispatch, first.run) == ("bind", "compile")
    assert "first call" in first.dispatch_reason
    assert not kernel._bound_kernels, "explain bound a kernel"

    bound = kernel.bind((x, y))
    assert explain(kernel, x, y).run == "compile"
    bound.ensure_config_exists((x, y))
    planning = explain(kernel, x, y)
    assert planning.run == "plan"
    assert planning.config == _cfg(8)

    kernel(x, y)
    native_call = explain(kernel, x, y)
    assert (native_call.dispatch, native_call.run) == ("fast", "native")
    assert native_call.planned and native_call.reason is None
    assert native_call.native_calls == 0

    kernel(x, y)
    assert explain(kernel, x, y).native_calls == 1
    assert "dispatch fast, run native" in str(explain(kernel, x, y))


def test_explain_runs_compiles_and_caches_nothing() -> None:
    kernel = _kernel(_add)
    x, y = torch.ones(32), torch.ones(32)
    kernel(x, y)
    kernel(x, y)
    bound = kernel.bind((x, y))
    run = bound._run
    state = (
        run.native_calls,
        run.refusals,
        run.last_refusal,
        dict(kernel._dispatch_cache),
        dict(kernel._bound_kernels),
        dict(bound._compile_cache),
    )
    other = torch.ones(16), torch.ones(16)
    for args in ((x, y), other, (x, y.double())):
        explain(kernel, *args)
    assert state == (
        run.native_calls,
        run.refusals,
        run.last_refusal,
        dict(kernel._dispatch_cache),
        dict(kernel._bound_kernels),
        dict(bound._compile_cache),
    )


def test_a_refused_argument_is_named_by_explain_and_recorded_by_the_call() -> None:
    kernel = _kernel(_add)
    x, y = torch.ones(32), torch.ones(32)
    kernel(x, y)
    bound = kernel.bind((x, y))
    run = bound._run
    # A strided tensor of the same shape: the plan's check refuses it, not Helion.
    assert run.last_refusal is None
    strided = torch.ones(64)[::2]
    told = explain(bound, x, strided)
    assert told.dispatch == "direct"
    assert told.run == "general"
    assert "argument 1" in told.reason
    assert "strides (2,)" in told.reason
    assert "planned" in told.reason
    assert run.last_refusal is None, "explain changed the records"

    assert bound(x, strided).equal(x + 1)
    assert run.last_refusal == (3, 1)
    assert run.refusals == (0, 0, 0, 1, 0, 0)
    after = explain(bound, x, y)
    assert after.run == "native"
    assert "an argument differs from the plan's" in next(iter(after.refusals))
    assert "index 1" in after.last_refusal


def test_a_constexpr_function_cell_that_differs_is_named() -> None:
    def make(value: object) -> object:
        return lambda t: t * value

    kernel = _kernel(_apply)
    x = torch.ones(32)
    fn = make(hl.constexpr(2.0))
    kernel(x, fn)
    bound = kernel.bind((x, fn))
    told = explain(bound, x, make(hl.constexpr(3.0)))
    assert told.run == "general"
    assert "cell value" in told.reason
    assert "3.0" in told.reason
    assert "2.0" in told.reason


def test_no_plan_says_why() -> None:
    kernel = _kernel(_add)
    x = torch.ones(32)
    kernel(x, x)
    told = explain(kernel, x, x)
    assert told.run == "general"
    assert "no plan" in told.reason
    assert "same object" in told.reason


def test_no_native_run_says_why() -> None:
    scalars = _kernel(_times)
    x = torch.ones(32)
    scalars(x, 2.0)
    told = explain(scalars, x, 2.0)
    assert told.run == "general"
    assert "no native run" in told.reason
    assert "run-time scalars" in told.reason
    assert "alpha" in told.reason

    host = helion.kernel(
        _torch_in_host,
        backend="mlir",
        static_shapes=True,
        config=_cfg(8),
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    host(x)
    told = explain(host, x)
    assert "host code does more than read metadata" in told.reason
    assert "y = " in told.reason


def test_native_calls_off_says_so() -> None:
    kernel = _kernel(_add)
    x, y = torch.ones(32), torch.ones(32)
    kernel(x, y)
    helion_mlir_backend.set_native_call(False)
    try:
        told = explain(kernel, x, y)
        assert told.run == "general"
        assert "off" in told.reason
    finally:
        helion_mlir_backend.set_native_call(True)


def test_refusal_kinds_name_the_check_that_refused() -> None:
    from helion_mlir_backend._compiler.targets.base import MLIRTarget

    kernel = _kernel(_add)
    x, y = torch.ones(32), torch.ones(32)
    kernel(x, y)
    bound = kernel.bind((x, y))
    run = bound._run
    assert run.check(x, y) is None
    assert run.check(x) == (0, -1)

    class Mode(torch.overrides.TorchFunctionMode):
        pass

    with Mode():
        assert run.check(x, y) == (1, -1)
    entry_target = type(next(iter(_entries())).target)
    original = entry_target.prepare_call
    entry_target.prepare_call = lambda self, *args: MLIRTarget.prepare_call(self, *args)
    try:
        assert run.check(x, y) == (2, -1)
    finally:
        if "prepare_call" in vars(MLIRTarget) and original is MLIRTarget.prepare_call:
            del entry_target.prepare_call
        else:
            entry_target.prepare_call = original
    assert run.check(x, torch.ones(16)) == (3, 1)


def _entries() -> object:
    from helion_mlir_backend._compiler import execution

    return execution._JIT_CACHE.values()


def test_keyword_arguments_are_explained_as_bind() -> None:
    kernel = _kernel(_add)
    x, y = torch.ones(32), torch.ones(32)
    kernel(x, y)
    kernel(x, y)
    told = explain(kernel, x, y=y)
    assert told.dispatch == "bind"
    assert "keyword" in told.dispatch_reason
    assert explain(kernel, x, y).dispatch == "fast"


def test_explain_empties_no_cache_at_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helion_mlir_backend import inject

    def scaled(t):
        return t * 2.0

    kernel = _kernel(_apply)
    x = torch.ones(32)
    kernel(x, scaled)
    monkeypatch.setattr(inject, "_MAX_FAST_KEYS", 1)
    assert explain(kernel, x, scaled).dispatch == "fast"
    assert len(kernel._dispatch_cache) == 1
    kernel(x, scaled)  # a call may clear it
    assert explain(kernel, x, scaled).run == "native"


def test_no_native_reason_on_another_target_loads_nothing() -> None:
    """A target without native paths must not bring the extension in, even to say
    why it has none (a compile on XPU went through here). In a new process: this
    file loads the extension."""
    script = textwrap.dedent(
        """
        import sys
        from types import SimpleNamespace
        from helion_mlir_backend._compiler import native
        from helion_mlir_backend._compiler.mlir.driver import _no_native_reason

        entry = SimpleNamespace(target=SimpleNamespace(native_in_place=False))
        assert "no native call path" in _no_native_reason(
            entry, lambda: None, [("x", None)], None
        )
        assert not native._loaded
        assert not any("helion_mlir_native" in name for name in sys.modules)
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=300)


def test_dispatch_reasons() -> None:
    kernel = _kernel(_add)
    x = torch.ones(32)
    assert "first call" in explain(kernel, x, x.clone()).dispatch_reason
    told = explain(kernel, x, [x])
    assert told.dispatch == "bind"
    assert "list" in told.dispatch_reason


def test_tracing_off_patches_nothing() -> None:
    if os.environ.get("HELION_MLIR_TRACE_CALLS"):
        pytest.skip("tracing is on in this process")
    assert not getattr(Kernel.__call__, "_helion_mlir_traced", False)
    assert "traced" not in BoundKernel.__call__.__name__


def test_trace_calls_reports_each_path_once_and_counts_at_exit(tmp_path: Path) -> None:
    script = tmp_path / "traced.py"
    script.write_text(
        textwrap.dedent(
            """
            import helion
            import helion.language as hl
            import torch
            import helion_mlir_backend

            def add(x, y):
                out = torch.empty_like(x)
                for tile in hl.tile(x.shape[0]):
                    out[tile] = x[tile] + y[tile]
                return out

            kernel = helion.kernel(
                add, backend="mlir", static_shapes=True,
                config=helion.Config(block_sizes=[8]),
            )
            x, y = torch.ones(32), torch.ones(32)
            for _ in range(5):
                kernel(x, y)
            kernel.bind((x, y))(x, torch.ones(64)[::2])
            kernel(x, x)
            def make(c):
                return lambda t: t * c
            def apply(x, fn):
                out = torch.empty_like(x)
                for tile in hl.tile(x.shape[0]):
                    out[tile] = fn(x[tile])
                return out
            other = helion.kernel(
                apply, backend="mlir", static_shapes=True,
                config=helion.Config(block_sizes=[8]),
            )
            for i in range(4):
                bound = other.bind((x, make(hl.constexpr(2.0))))
                other(x, make(hl.constexpr(2.0)))
            bound(x, make(hl.constexpr(3.0)))
            bound(x, make(hl.constexpr(4.0)))
            """
        )
    )
    done = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "HELION_MLIR_TRACE_CALLS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    lines = [
        line for line in done.stderr.splitlines() if line.startswith("[helion-mlir]")
    ]
    text = "\n".join(lines)
    assert "add: compiled, native run" in text
    assert sum("add: dispatch fast, run native" in line for line in lines) == 1
    assert "dispatch bind, run compile" in text
    assert "argument 1" in text
    assert "strides (2,)" in text
    assert "at exit: add" in text
    # One line for the refusal, whichever value was refused.
    assert sum("(argument 1: cell value" in line for line in lines) == 1
    # 4 after the first call, and kernel(x, x) matches the plan of (x, y).
    assert "5 native calls" in text
    assert "refused: an argument differs from the plan's: 1" in text
