"""Host code reads the module's globals on every call, as plain Python and
stock Helion do; a recorded native plan must not freeze such a read."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING
from unittest import mock

import helion
import helion.language as hl
import pytest
import torch

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler import native
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

if TYPE_CHECKING:
    from pathlib import Path

CONFIG = helion.Config(block_sizes=[8])
S = 2.0
FLAG = True
N = 32


def _return_times_global(x):
    m = x.shape[0]
    out = torch.empty_like(x)
    for tile in hl.tile(m):
        out[tile] = x[tile]
    return out * S


def _local_from_global(x):
    m = x.shape[0]
    out = torch.empty_like(x)
    s = S
    for tile in hl.tile(m):
        out[tile] = x[tile] * s
    return out


def _local_from_flag(x):
    m = x.shape[0]
    out = torch.empty_like(x)
    s = 2.0 if FLAG else 3.0
    for tile in hl.tile(m):
        out[tile] = x[tile] * s
    return out


def _shape_from_global(x):
    out = torch.empty(N, dtype=x.dtype)
    for tile in hl.tile(N):
        out[tile] = x[tile] * 2.0
    return out


def _metadata_only(x):
    m = x.shape[0]
    out = torch.empty_like(x)
    for tile in hl.tile(m):
        out[tile] = x[tile] * 2.0
    return out


@pytest.fixture(params=[True, False], ids=["native", "native-off"])
def native_call(request: pytest.FixtureRequest) -> object:
    before = native.enabled()
    native.set_native_call(request.param)
    yield request.param
    native.set_native_call(before)


@pytest.fixture(params=[False, True], ids=["dynamic", "static"])
def static_shapes(request: pytest.FixtureRequest) -> bool:
    return request.param


def _kernel(fn: object, static_shapes: bool) -> helion.Kernel:
    return helion.kernel(fn, backend="mlir", config=CONFIG, static_shapes=static_shapes)


@pytest.mark.parametrize(
    "fn,name,first,second",
    [
        (_return_times_global, "S", 2.0, 3.0),
        (_local_from_global, "S", 2.0, 3.0),
        (_local_from_flag, "FLAG", True, False),
    ],
)
def test_host_code_reads_a_changed_global(
    monkeypatch: pytest.MonkeyPatch,
    native_call: bool,
    static_shapes: bool,
    fn: object,
    name: str,
    first: object,
    second: object,
) -> None:
    kernel = _kernel(fn, static_shapes)
    x = torch.ones(32)
    monkeypatch.setitem(globals(), name, first)
    # Enough calls for a plan to be recorded and replayed, if one were.
    for _ in range(3):
        assert kernel(x).flatten()[0].item() == 2.0
    monkeypatch.setitem(globals(), name, second)
    for _ in range(3):
        assert kernel(x).flatten()[0].item() == 3.0
    monkeypatch.setitem(globals(), name, first)
    assert kernel(x).flatten()[0].item() == 2.0


def test_host_allocation_follows_a_changed_global(
    monkeypatch: pytest.MonkeyPatch, native_call: bool, static_shapes: bool
) -> None:
    kernel = _kernel(_shape_from_global, static_shapes)
    x = torch.ones(32)
    for _ in range(3):
        assert kernel(x).shape == (32,)
    monkeypatch.setitem(globals(), "N", 16)
    for _ in range(3):
        assert kernel(x).shape == (16,)


@pytest.mark.parametrize(
    "fn",
    [_return_times_global, _local_from_global, _local_from_flag, _shape_from_global],
)
def test_a_plan_does_not_apply_to_host_code_reading_globals(
    static_shapes: bool, fn: object
) -> None:
    kernel = _kernel(fn, static_shapes)
    x = torch.ones(32)
    kernel(x)
    (bound,) = kernel._bound_kernels.values()
    assert type(bound._run).__name__ != "KernelRun"


@pytest.mark.skipif(native.module() is None, reason="needs the native call path")
def test_host_code_without_globals_keeps_its_plan() -> None:
    kernel = _kernel(_metadata_only, True)
    x = torch.ones(32)
    kernel(x)
    (bound,) = kernel._bound_kernels.values()
    assert type(bound._run).__name__ == "KernelRun"


def test_nothing_is_added_to_the_module(tmp_path: Path) -> None:
    module = _load(
        tmp_path,
        "host_globals_clean",
        """
        import helion, helion.language as hl, torch
        W = torch.full((32,), 2.0)
        S = 3.0
        def _k(x):
            m = x.shape[0]
            out = torch.empty_like(x)
            for t in hl.tile(m):
                out[t] = x[t] * W[t]
            return out * S
        K = helion.kernel(_k, backend="mlir", config=helion.Config(block_sizes=[8]))
        """,
    )
    before = set(vars(module))
    assert module.K(torch.ones(32)).flatten()[0].item() == 6.0
    assert set(vars(module)) == before


def _load(tmp_path: Path, name: str, text: str) -> object:
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(text))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load(tmp_path: Path) -> object:
    names = []

    def loader(name: str, text: str) -> object:
        names.append(name)
        return _load(tmp_path, name, text)

    sys.path.insert(0, str(tmp_path))
    yield loader
    sys.path.remove(str(tmp_path))
    for name in names:
        sys.modules.pop(name, None)


def _apply(x, fn):
    m = x.shape[0]
    out = torch.empty_like(x)
    for tile in hl.tile(m):
        out[tile] = fn(x[tile])
    return out


@pytest.fixture
def apply_kernel(native_call: bool) -> helion.Kernel:
    return helion.kernel(_apply, backend="mlir", config=CONFIG)


_READS_GLOBAL = """
S = %s
def fn(y): return y * S
"""


def _value(out: torch.Tensor) -> float:
    return out.flatten()[0].item()


def test_epilogues_read_their_own_modules_globals(
    load: object, apply_kernel: helion.Kernel
) -> None:
    """Was a NameError on ``_global_source0``."""
    a = load("alias_a", _READS_GLOBAL % "2.0")
    b = load("alias_b", _READS_GLOBAL % "3.0")
    x = torch.ones(32)
    for _ in range(2):
        assert _value(apply_kernel(x, a.fn)) == 2.0
        assert _value(apply_kernel(x, b.fn)) == 3.0
    assert len(apply_kernel._bound_kernels) == 2


def test_epilogue_reads_a_changed_global(
    load: object, apply_kernel: helion.Kernel
) -> None:
    module = load("alias_changed", _READS_GLOBAL % "2.0")
    x = torch.ones(32)
    for _ in range(3):
        assert _value(apply_kernel(x, module.fn)) == 2.0
    module.S = 3.0
    for _ in range(3):
        assert _value(apply_kernel(x, module.fn)) == 3.0


def test_kernel_global_times_epilogue_global(load: object, native_call: bool) -> None:
    kernel_module = load(
        "alias_kernel",
        """
        import helion, helion.language as hl, torch
        LOCAL = 10.0
        def _k(x, fn):
            m = x.shape[0]
            out = torch.empty_like(x)
            for t in hl.tile(m):
                out[t] = fn(x[t]) * LOCAL
            return out
        K = helion.kernel(_k, backend="mlir", config=helion.Config(block_sizes=[8]))
        """,
    )
    a = load("alias_epilogue", _READS_GLOBAL % "2.0")
    x = torch.ones(32)
    assert _value(kernel_module.K(x, a.fn)) == 20.0
    a.S = 3.0
    kernel_module.LOCAL = 100.0
    assert _value(kernel_module.K(x, a.fn)) == 300.0


def test_helper_in_another_module_reads_its_own_global(
    load: object, native_call: bool
) -> None:
    helper = load("alias_helper", _READS_GLOBAL % "2.0")
    module = load(
        "alias_caller",
        """
        import helion, helion.language as hl, torch
        import alias_helper
        def _k(x):
            m = x.shape[0]
            out = torch.empty_like(x)
            for t in hl.tile(m):
                out[t] = alias_helper.fn(x[t])
            return out
        K = helion.kernel(_k, backend="mlir", config=helion.Config(block_sizes=[8]))
        """,
    )
    # Inlined at compile time, so a later change of S doesn't reach it: the same
    # as in stock Helion, unlike a global read by host code or an epilogue.
    assert helper.S == 2.0
    assert _value(module.K(torch.ones(32))) == 2.0


def test_epilogue_global_of_a_retaken_module_name_raises_under_O(
    tmp_path: Path,
) -> None:
    """``python -O`` strips Helion's assert in ``import_from_module``: without
    our check the epilogue of the first module reads the second one's S."""
    script = tmp_path / "run.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import importlib.util, sys
            import helion, helion.language as hl, torch
            import helion_mlir_backend
            assert False, "this script needs python -O"
            def load(tag, value):
                path = {str(tmp_path)!r} + f"/{{tag}}.py"
                open(path, "w").write({_READS_GLOBAL!r} % value)
                spec = importlib.util.spec_from_file_location("shared", path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["shared"] = module
                spec.loader.exec_module(module)
                return module
            def _apply(x, fn):
                m = x.shape[0]
                out = torch.empty_like(x)
                for tile in hl.tile(m):
                    out[tile] = fn(x[tile])
                return out
            a, b = load("a", "2.0"), load("b", "3.0")
            kernel = helion.kernel(
                _apply, backend="mlir", config=helion.Config(block_sizes=[8])
            )
            try:
                print("got", kernel(torch.ones(32), a.fn).flatten()[0].item())
            except Exception as error:
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


LOCAL_NAMED_LIKE_A_GLOBAL = """
import torch, helion, helion.language as hl

n = 99
glob = 3.0

def local(x):
    n, k = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([n, k]):
        out[tm, tn] = x[tm, tn] + 1.0
    return out

def read_global(x):
    m, k = x.shape
    out = torch.empty_like(x)
    for tm, tn in hl.tile([m, k]):
        out[tm, tn] = x[tm, tn] * glob
    return out
"""


def _kernel_of(fn: object) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[2, 8]),
    )


def test_local_named_like_a_global_of_an_unloaded_module(load) -> None:
    """Regression: a local's assignment looks its name up, only to compare types, and
    that lookup was refused as a read of a module global ("module global n") when
    the kernel's module isn't in ``sys.modules``."""
    module = load("shadow_unregistered", LOCAL_NAMED_LIKE_A_GLOBAL)
    del sys.modules["shadow_unregistered"]
    x = torch.randn(4, 8)
    assert torch.equal(_kernel_of(module.local)(x), x + 1.0)


def test_read_of_a_global_of_an_unloaded_module_is_refused(load) -> None:
    module = load("shadow_unregistered_read", LOCAL_NAMED_LIKE_A_GLOBAL)
    del sys.modules["shadow_unregistered_read"]
    with pytest.raises(UnsupportedOperationError, match="module global glob"):
        _kernel_of(module.read_global)(torch.randn(4, 8))


def test_local_named_like_a_global_of_an_unsupported_type(load) -> None:
    """Regression: a local named like a module global no kernel can take, e.g. a
    ``torch.Generator``, failed with "unsupported argument type": assigning the
    local looked up the global."""
    module = load(
        "shadow_generator",
        """
        import torch, helion, helion.language as hl

        g = torch.Generator()

        def local(x):
            m, k = x.shape
            out = torch.empty_like(x)
            for tm, tn in hl.tile([m, k]):
                g = x[tm, tn] + 1.0
                out[tm, tn] = g
            return out
        """,
    )
    x = torch.randn(4, 8)
    assert torch.equal(_kernel_of(module.local)(x), x + 1.0)


def test_global_statement_assigned_on_the_device_is_refused(load) -> None:
    """A ``global`` is no local: assigning it in a device loop stays refused (it ran
    with the module's value unchanged while every lookup skipped globals)."""
    module = load(
        "global_statement",
        """
        import torch, helion, helion.language as hl

        g = 1.0

        def kernel(x):
            global g
            m, k = x.shape
            out = torch.empty_like(x)
            for tm, tn in hl.tile([m, k]):
                g = x[tm, tn] + 1.0
                out[tm, tn] = g
            return out
        """,
    )
    with pytest.raises(Exception, match="(?i)host variable|global"):
        _kernel_of(module.kernel)(torch.randn(4, 8))
    assert module.g == 1.0


def test_other_backends_keep_helions_lookup() -> None:
    """Only this backend's kernels skip globals: importing it changes nothing for
    Helion's other backends."""
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.type_propagation import GlobalScope
    from helion._compiler.type_propagation import LocalScope

    parent = mock.Mock(spec=GlobalScope)
    parent.get.return_value = "from the global scope"
    scope = LocalScope(parent=parent)
    backend = mock.Mock()
    with mock.patch.object(
        CompileEnvironment, "current", return_value=mock.Mock(backend=backend)
    ):
        assert scope.maybe_get("g") == "from the global scope"
