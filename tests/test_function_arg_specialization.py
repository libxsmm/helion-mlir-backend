"""A kernel taking a function must not reuse the compile of a different one."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import helion
import helion.language as hl
import pytest
import torch

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

if TYPE_CHECKING:
    from pathlib import Path


# These pin how Helion keys function arguments and finds module globals.
pytestmark = pytest.mark.helion_internals


def _apply_kernel(x: torch.Tensor, fn: object) -> torch.Tensor:
    m = x.shape[0]
    out = torch.empty_like(x)
    for tile in hl.tile(m):
        out[tile] = fn(x[tile])
    return out


@pytest.fixture
def apply_kernel() -> helion.Kernel:
    return helion.kernel(
        _apply_kernel, backend="mlir", config=helion.Config(block_sizes=[8])
    )


def _load(path: Path, text: str, name: str) -> object:
    path.write_text(textwrap.dedent(text))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _module_pair(tmp_path: Path, request: pytest.FixtureRequest, text: str) -> tuple:
    names = [f"_fn_arg_{request.node.name}_{tag}" for tag in "ab"]
    for name in names:
        request.addfinalizer(lambda name=name: sys.modules.pop(name, None))
    return tuple(
        _load(tmp_path / f"{tag}.py", text % value, name)
        for tag, value, name in zip("ab", ("2.0", "3.0"), names, strict=True)
    )


def _value(out: torch.Tensor) -> float:
    return out.flatten()[0].item()


def test_equal_code_reading_other_module_globals(
    tmp_path: Path, request: pytest.FixtureRequest, apply_kernel: helion.Kernel
) -> None:
    a, b = _module_pair(
        tmp_path,
        request,
        """
        def scale(x): return x * %s
        def fn(x): return scale(x)
        """,
    )
    assert a.fn.__code__ == b.fn.__code__
    x = torch.ones(32)
    assert _value(apply_kernel(x, a.fn)) == 2.0
    assert _value(apply_kernel(x, b.fn)) == 3.0
    assert _value(apply_kernel(x, a.fn)) == 2.0


def test_equal_code_with_other_default(apply_kernel: helion.Kernel) -> None:
    def make(scale: float) -> object:
        def fn(x, scale=scale):
            return x * scale

        return fn

    x = torch.ones(32)
    assert _value(apply_kernel(x, make(2.0))) == 2.0
    assert _value(apply_kernel(x, make(3.0))) == 3.0
    assert _value(apply_kernel(x, make(2.0))) == 2.0


def _make_divide(scale: float) -> object:
    def fn(x, scale=scale):
        return x / scale

    return fn


@pytest.mark.parametrize("first,second", [(0.0, -0.0), (-0.0, 0.0)])
def test_signed_zero_defaults_compile_apart(
    apply_kernel: helion.Kernel, first: float, second: float
) -> None:
    x = torch.ones(32)
    for scale in (first, second):
        assert _value(apply_kernel(x, _make_divide(scale))) == math.copysign(
            math.inf, scale
        )
    assert len(apply_kernel._bound_kernels) == 2


def test_nan_defaults_share_a_compile(apply_kernel: helion.Kernel) -> None:
    x = torch.ones(32)
    for _ in range(2):
        assert math.isnan(_value(apply_kernel(x, _make_divide(float("nan")))))
    assert len(apply_kernel._bound_kernels) == 1


def _int_cell(n: int) -> object:
    return lambda t: t * n


def _float_cell(value: float) -> object:
    return lambda t: t * value


def test_helion_consults_the_function_key(apply_kernel: helion.Kernel) -> None:
    """The wrapper replaces an entry of Helion's extractor table: if Helion
    stopped dispatching through that table, it would go inert while the entry
    stayed. So this goes through the key the kernel's cache uses."""
    x = torch.ones(32)
    key = apply_kernel._base_specialization_key
    assert key((x, _make_divide(2.0))) != key((x, _make_divide(3.0)))
    assert key((x, _make_divide(2.0))) == key((x, _make_divide(2.0)))
    # A number in a cell by its type, as in Helion: compile-time reads of an int go
    # through the extras (test_constexpr_and_int_cell_keys.py).
    assert key((x, _int_cell(1))) == key((x, _int_cell(3)))
    assert key((x, _float_cell(1.0))) == key((x, _float_cell(3.0)))


def test_equal_functions_share_a_compile(apply_kernel: helion.Kernel) -> None:
    """Closures of one def, on runtime scalars or equal defaults, stay one
    specialization: a new epilogue object per call must not recompile."""

    def make(scale: float) -> object:
        def fn(x, offset=1.0):
            return x * scale + offset

        return fn

    x = torch.ones(32)
    assert _value(apply_kernel(x, make(2.0))) == 3.0
    assert _value(apply_kernel(x, make(3.0))) == 4.0
    assert len(apply_kernel._bound_kernels) == 1


def test_other_backends_keep_helions_key() -> None:
    def fn(x, scale=2.0):
        return x * scale

    kernel = helion.kernel(_apply_kernel, backend="triton")
    assert kernel._specialization_key(fn) == fn.__code__


_SHADOWED_GLOBAL = """
import helion, helion.language as hl, torch
import helion_mlir_backend
SCALE = %s
def _kernel(x):
    m = x.shape[0]
    out = torch.empty_like(x)
    for tile in hl.tile(m):
        out[tile] = x[tile] * SCALE
    return out
KERNEL = helion.kernel(
    _kernel, backend="mlir", config=helion.Config(block_sizes=[8])
)
"""


def test_global_of_a_module_whose_name_was_retaken_raises(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    a, b = _module_pair(tmp_path, request, _SHADOWED_GLOBAL)
    sys.modules[a.__name__] = b
    with pytest.raises(UnsupportedOperationError, match="another module"):
        a.KERNEL(torch.ones(32))


def test_global_of_a_module_whose_name_was_retaken_raises_under_O(
    tmp_path: Path,
) -> None:
    """``python -O`` strips Helion's assert: without our check the kernel reads
    the other module's SCALE and returns 3s."""
    script = tmp_path / "run.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import importlib.util, sys, torch
            assert False, "this script needs python -O"
            from helion_mlir_backend._compiler.mlir.support import (
                UnsupportedOperationError,
            )
            TEXT = {_SHADOWED_GLOBAL!r}
            def load(tag, value):
                path = {str(tmp_path)!r} + f"/{{tag}}.py"
                open(path, "w").write(TEXT % value)
                spec = importlib.util.spec_from_file_location("shared", path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["shared"] = module
                spec.loader.exec_module(module)
                return module
            a, b = load("a", "2.0"), load("b", "3.0")
            assert b.KERNEL(torch.ones(32)).flatten()[0].item() == 3.0
            try:
                print("got", a.KERNEL(torch.ones(32)).flatten()[0].item())
            except UnsupportedOperationError as error:
                print("raised", error)
            """
        )
    )
    done = subprocess.run(
        [sys.executable, "-O", str(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert "raised" in done.stdout and "another module" in done.stdout, done.stdout


def test_source_module_check_accepts_the_right_module(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    a, b = _module_pair(tmp_path, request, _SHADOWED_GLOBAL)
    assert _value(a.KERNEL(torch.ones(32))) == 2.0
    assert _value(b.KERNEL(torch.ones(32))) == 3.0
