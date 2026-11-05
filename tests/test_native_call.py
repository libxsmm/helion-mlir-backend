"""The native call paths (``_compiler/native``).

A compiled kernel whose entry is static and whose host code is metadata-only
runs as a native ``KernelRun``: its first call records a plan (which argument
or fresh host tensor each entry argument is, what is returned); later calls
whose arguments match it exactly are checked, allocated and called natively.
Every other call must take the Python run, with its results and errors
unchanged. ``prepared_call`` binds a KernelRun's other arguments.
"""

from __future__ import annotations

import ast
import threading

import helion
import helion.language as hl
import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from tests.harness import run_direct

import helion_mlir_backend
from helion_mlir_backend._compiler import execution
from helion_mlir_backend._compiler import native
from helion_mlir_backend._compiler.mlir import driver
from helion_mlir_backend._compiler.mlir.host_code import _MetadataOnly
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.targets.cpu import CpuTarget

pytestmark = pytest.mark.skipif(
    native.module() is None, reason="the native extension could not be built"
)


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def scaled_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0 + y[tm, tn]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def copy_plus_one(a: torch.Tensor, b: torch.Tensor) -> None:
    for tile in hl.tile(a.size(0)):
        b[tile] = a[tile] + 1.0


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def write_both(a: torch.Tensor, b: torch.Tensor) -> None:
    for tile in hl.tile(a.size(0)):
        a[tile] = a[tile] + 1.0
        b[tile] = b[tile] * 2.0


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def add_rows_into(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    m, n = x.shape
    if out is None:
        result = torch.empty([m, n], dtype=x.dtype, device=x.device)
    else:
        result = out
    for tm, tn in hl.tile([m, n]):
        result[tm, tn] = x[tm, tn] + y[tm, tn]
    return result


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def scaled_by(x: torch.Tensor, scale: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * scale
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def host_op_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y2 = y * 1.0  # a torch op in the host code: not metadata-only
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + y2[tm, tn]
    return out


def _run(kernel: helion.Kernel, args: list[object]) -> object:
    return kernel.bind(tuple(args))._run


@pytest.fixture
def general_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The entry argument names the general path checked, in order."""
    seen: list[str] = []
    checked = driver._checked_tensor

    def spy(target, arg, value):
        seen.append(arg.name)
        return checked(target, arg, value)

    monkeypatch.setattr(driver, "_checked_tensor", spy)
    return seen


def _calls(kernel: helion.Kernel, args: list[object]) -> int:
    return _run(kernel, args).native_calls


# -- the native run is taken ------------------------------------------------


def test_static_call_runs_natively(general_runs: list[str]) -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    torch.testing.assert_close(run_direct(scaled_add, [x, y]), x * 2.0 + y)
    run = _run(scaled_add, [x, y])
    assert type(run).__name__ == "KernelRun" and run.planned
    before = run.native_calls
    general_runs.clear()
    for _ in range(3):
        x, y = torch.randn(16, 16), torch.randn(16, 16)
        out = run_direct(scaled_add, [x, y])
        assert out.is_contiguous() and out.dtype == x.dtype
        torch.testing.assert_close(out, x * 2.0 + y)
    assert run.native_calls == before + 3
    assert general_runs == []


def test_native_result_matches_the_python_run_bit_for_bit() -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    run = _run(scaled_add, [x, y])
    assert torch.equal(run(x, y), run.general(x, y))


def test_inout_kernel_writes_in_place_and_returns_none() -> None:
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    before = _calls(copy_plus_one, [a, b])
    a, b = torch.randn(16), torch.zeros(16)
    assert run_direct(copy_plus_one, [a, b]) is None
    torch.testing.assert_close(b, a + 1.0)
    assert _calls(copy_plus_one, [a, b]) == before + 1


def test_parameter_returned_is_the_same_tensor() -> None:
    x, y, out = torch.randn(8, 8), torch.randn(8, 8), torch.empty(8, 8)
    run_direct(add_rows_into, [x, y, out])
    before = _calls(add_rows_into, [x, y, out])
    out2 = torch.empty(8, 8)
    assert run_direct(add_rows_into, [x, y, out2]) is out2
    torch.testing.assert_close(out2, x + y)
    assert _calls(add_rows_into, [x, y, out2]) == before + 1


def test_storage_swapped_between_calls_is_read() -> None:
    # The plan holds no data pointers: a tensor whose storage changes
    # (set_, .data) between calls is read from its current storage.
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    run_direct(scaled_add, [x, y])
    before = _calls(scaled_add, [x, y])
    x.set_(torch.randn(16, 16))
    torch.testing.assert_close(run_direct(scaled_add, [x, y]), x * 2.0 + y)
    y.data = torch.randn(16, 16)
    torch.testing.assert_close(run_direct(scaled_add, [x, y]), x * 2.0 + y)
    assert _calls(scaled_add, [x, y]) == before + 2


def test_inference_mode_and_deterministic_fill_run_natively() -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    run = _run(scaled_add, [x, y])
    with torch.inference_mode():
        before = run.native_calls
        out = run(x, y)
        assert run.native_calls == before + 1
        assert out.is_inference() == run.general(x, y).is_inference()
    torch.testing.assert_close(out, x * 2.0 + y)
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.testing.assert_close(run(x, y), x * 2.0 + y)
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.utils.deterministic.fill_uninitialized_memory = fill


def test_concurrent_native_calls_are_correct() -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    run = _run(scaled_add, [x, y])
    assert run.planned
    before = run.native_calls
    wrong: list[int] = []

    def work(seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        for _ in range(50):
            a = torch.randn(16, 16, generator=generator)
            b = torch.randn(16, 16, generator=generator)
            if not torch.equal(run_direct(scaled_add, [a, b]), a * 2.0 + b):
                wrong.append(seed)

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert wrong == []
    assert run.native_calls == before + 8 * 50


# -- regressions (review of the first version) ------------------------------


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def scaled_add_alias(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0 + y[tm, tn]
    return out


def test_aliased_first_call_records_no_plan() -> None:
    # Regression: the first call scaled_add(a, a) mapped both entry inputs to
    # argument 0, so a later scaled_add(a, b) computed a * 2 + a.
    a = torch.randn(16, 16)
    torch.testing.assert_close(scaled_add_alias(a, a), a * 3.0)
    b = torch.randn(16, 16)
    assert torch.equal(scaled_add_alias(a, b), a * 2.0 + b)
    run = _run(scaled_add_alias, [a, b])
    assert not run.planned and run.native_calls == 0


BIAS = torch.randn(16, 16)


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def add_default_bias(x: torch.Tensor, b: torch.Tensor = BIAS) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + b[tm, tn]
    return out


def test_tensor_default_argument_is_not_a_fresh_tensor() -> None:
    # Regression: a tensor default argument (not passed, so not an argument)
    # was planned as a fresh host tensor: later calls read an empty one.
    for _ in range(3):
        x = torch.randn(16, 16)
        assert torch.equal(add_default_bias(x), x + BIAS)
    assert not _run(add_default_bias, [x]).planned


ACC = torch.zeros(16)


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def into_or_global(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        result = torch.empty_like(x)
    else:
        print("else branch ran")
        result = ACC  # a global tensor, not a fresh one
    for tile in hl.tile(x.size(0)):
        result[tile] = x[tile] + 1.0
    return result


def test_untaken_literal_branch_is_checked(capsys: pytest.CaptureFixture[str]) -> None:
    # Regression: only the branch Helion typed as taken was checked; the same
    # BoundKernel called with out given ran an unchecked else branch (here a
    # print, and a global tensor planned as fresh).
    x = torch.zeros(16)
    bound = into_or_global.bind((x, None))
    torch.testing.assert_close(bound(x, None), x + 1.0)
    for _ in range(3):
        assert bound(x, torch.zeros(16)) is ACC
    assert capsys.readouterr().out.count("else branch ran") == 3
    assert type(bound._run).__name__ != "KernelRun"


_UNTAKEN_SOURCE = """
import torch, helion, helion.language as hl
ACC = torch.zeros(16)
@helion.kernel(backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[8]))
def k(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        result = torch.empty_like(x)
    else:
        {body}
    for tile in hl.tile(x.size(0)):
        result[tile] = x[tile] + 1.0
    return result
"""


@pytest.mark.parametrize(
    ("body", "metadata_only"),
    [
        pytest.param("result = out", True, id="rename"),
        pytest.param("result = (y := out)", False, id="walrus"),
        pytest.param("result = out[0:16]", False, id="subscript"),
        pytest.param(
            "s = (*x.shape,)\n        result = out", False, id="starred_tuple"
        ),
        pytest.param(
            "result = torch.empty(*x.shape, dtype=x.dtype, device=x.device)",
            False,
            id="starred_empty",
        ),
        pytest.param("result = ACC", False, id="global"),
    ],
)
def test_untaken_branch_forms(tmp_path, body: str, metadata_only: bool) -> None:
    # The review's attempts to run work through the untaken branch: only
    # renames and plain allocations pass; the rest keep the Python run.
    import importlib.util

    path = tmp_path / "untaken_kernel.py"
    path.write_text(_UNTAKEN_SOURCE.replace("{body}", body))
    spec = importlib.util.spec_from_file_location(f"untaken_{id(path)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.zeros(16)
    bound = module.k.bind((x, None))
    bound(x, None)
    run = bound._run
    general = getattr(run, "general", run)
    cells = dict(zip(general.__code__.co_freevars, general.__closure__, strict=True))
    assert (
        cells["host_function"].cell_contents._helion_mlir_metadata_only is metadata_only
    )
    assert (type(run).__name__ == "KernelRun") is metadata_only
    out = torch.zeros(16)
    result = bound(x, out)
    expected = general(x, torch.zeros(16))
    assert torch.equal(result, expected) and result.shape == expected.shape


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(1, 8))
def like_template(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(y.size()):
        out[tm, tn] = y[tm, tn] + 1.0
    return out


def test_fresh_tensor_strides_match_the_python_run() -> None:
    # Regression: empty_like of a template whose size-1 dim has stride 7 is
    # contiguous for torch but not at::empty's strides: the native run
    # returned other strides than the Python run.
    x = torch.as_strided(torch.randn(16), (1, 16), (7, 1))
    y = torch.randn(1, 16)
    results = [like_template(x, y) for _ in range(3)]
    assert {r.stride() for r in results} == {torch.empty_like(x).stride()}
    assert all(torch.equal(r, y + 1.0) for r in results)
    assert not _run(like_template, [x, y]).planned


# -- the extension build -------------------------------------------------------


@pytest.fixture
def unloaded(monkeypatch: pytest.MonkeyPatch, tmp_path) -> object:
    """native.module() as on first use, building into an empty cache."""
    monkeypatch.setattr(native, "_loaded", [])
    monkeypatch.setenv(native.NATIVE_CACHE_ENV, str(tmp_path))
    return tmp_path


def _fake_compiler(tmp_path, body: str) -> str:
    path = tmp_path / "fake-cxx"
    path.write_text(
        f'#!/bin/sh\n[ "$1" = --version ] && {{ echo fake 1; exit 0; }}\n{body}\n'
    )
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize(
    "body",
    [
        "exit 1",  # the build fails
        # "builds" a library that doesn't load
        'for a; do [ "$prev" = -o ] && echo garbage > "$a"; prev=$a; done; exit 0',
    ],
)
def test_failed_build_falls_back_to_the_python_paths(
    unloaded, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    cache = unloaded
    monkeypatch.setenv("CXX", _fake_compiler(cache, body))
    assert native.module() is None
    # Nothing that doesn't load enters the cache.
    assert not [p for p in cache.iterdir() if p.suffix == ".so"]

    @helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
    def scaled_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tm, tn in hl.tile(x.size()):
            out[tm, tn] = x[tm, tn] * y[tm, tn]
        return out

    x, y = torch.randn(8, 16), torch.randn(8, 16)
    for _ in range(2):
        assert torch.equal(run_direct(scaled_mul, [x, y]), x * y)
    assert type(_run(scaled_mul, [x, y])).__name__ == "function"


def test_build_lock_timeout_falls_back(
    unloaded, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    monkeypatch.setattr(native, "_BUILD_TIMEOUT", 0.5)
    compiler = native._compiler()
    key = native._key(compiler, native._command(compiler, "{output}"))
    with open(unloaded / f"{native._MODULE_NAME}-{key}.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # another builder holds it
        assert native.module() is None


def test_uninitialized_native_objects_raise() -> None:
    extension = native.module()
    for cls, call in (
        (extension.EntryCall, lambda o: o.call([])),
        (extension.KernelRun, lambda o: o()),
    ):
        obj = cls.__new__(cls)
        with pytest.raises(RuntimeError, match="not initialized"):
            call(obj)


# -- every other call takes the Python run, with its result or error ------


def _planned_scaled_add() -> tuple[object, torch.Tensor, torch.Tensor]:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    run = _run(scaled_add, [x, y])
    assert run.planned
    return run, x, y


@pytest.mark.parametrize(
    "case",
    ["transposed", "unit_stride_view", "negative_view", "subclass", "requires_grad"],
)
def test_other_tensors_take_the_python_run(case: str, general_runs: list[str]) -> None:
    run, x, y = _planned_scaled_add()
    if case == "transposed":
        x = x.t()
    elif case == "unit_stride_view":
        x = torch.randn(16, 32)[:, ::2]
    elif case == "negative_view":
        x = torch._neg_view(x)
    elif case == "subclass":

        class Sub(torch.Tensor):
            pass

        x = x.as_subclass(Sub)
    before = run.native_calls
    general_runs.clear()
    if case == "requires_grad":
        # Not a fallback: the Python path ignores autograd too (same result).
        x = x.clone().requires_grad_()
        with torch.no_grad():
            torch.testing.assert_close(run(x, y), x * 2.0 + y)
        assert run.native_calls == before + 1
        return
    expected = x.detach() * 2.0 + y
    plain = torch.Tensor
    torch.testing.assert_close(
        run(x, y).as_subclass(plain), expected.as_subclass(plain)
    )
    assert run.native_calls == before
    # A subclass passes the Python in-place checks (as before); the others
    # need the general path's staging.
    assert bool(general_runs) == (case != "subclass")


def test_wrong_shape_and_dtype_report_the_general_errors() -> None:
    run, x, y = _planned_scaled_add()
    before = run.native_calls
    with pytest.raises(ValueError, match="shape"):
        run(torch.randn(8, 16), y)
    with pytest.raises(ValueError, match="dtype"):
        run(x.double(), y)
    with pytest.raises(NotImplementedError, match="only CPU is supported"):
        run(x.to("meta"), y)
    assert run.native_calls == before


def test_wrong_arity_takes_the_python_run() -> None:
    run, x, y = _planned_scaled_add()
    before = run.native_calls
    with pytest.raises(TypeError):
        run(x)
    with pytest.raises(TypeError):
        run(x, y=y)
    assert run.native_calls == before


def test_aliasing_arguments_take_the_python_run(general_runs: list[str]) -> None:
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    run_direct(copy_plus_one, [a, b])
    before = _calls(copy_plus_one, [a, b])
    general_runs.clear()
    x = torch.randn(16)
    expected = x + 1.0
    run_direct(copy_plus_one, [x, x])
    torch.testing.assert_close(x, expected)
    buffer = torch.randn(24)
    a, b = buffer[:16], buffer[8:]
    expected = a + 1.0
    run_direct(copy_plus_one, [a, b])
    torch.testing.assert_close(b, expected)
    assert general_runs
    assert _calls(copy_plus_one, [a, b]) == before


def test_inouts_sharing_memory_are_still_rejected() -> None:
    a, b = torch.randn(16), torch.randn(16)
    run_direct(write_both, [a, b])
    run_direct(write_both, [a, b])
    x = torch.randn(16)
    with pytest.raises(UnsupportedOperationError, match="share memory"):
        run_direct(write_both, [x, x])


def test_changed_none_argument_takes_the_python_run() -> None:
    x, y = torch.randn(8, 8), torch.randn(8, 8)
    run_direct(add_rows_into, [x, y, None])
    run = _run(add_rows_into, [x, y, None])
    assert run.planned
    before = run.native_calls
    torch.testing.assert_close(run(x, y, None), x + y)
    assert run.native_calls == before + 1
    out = torch.empty(8, 8)
    assert run(x, y, out) is out  # the plan had None: Python run
    torch.testing.assert_close(out, x + y)
    assert run.native_calls == before + 1


def test_constexpr_arguments_match_by_value_and_type() -> None:
    x = torch.randn(16, 16)
    run_direct(scaled_by, [x, hl.constexpr(3.0)])
    run = _run(scaled_by, [x, hl.constexpr(3.0)])
    assert run.planned
    before = run.native_calls
    # A new, equal constexpr (helpers make one per call): native.
    torch.testing.assert_close(run(x, hl.constexpr(3.0)), x * 3.0)
    assert run.native_calls == before + 1
    # Equal values of another type (3, True for 1) or another value: the
    # Python run (here the same compiled kernel).
    torch.testing.assert_close(run(x, hl.constexpr(3)), x * 3.0)
    torch.testing.assert_close(run(x, hl.constexpr(4.0)), x * 3.0)
    assert run.native_calls == before + 1


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def with_epilogue(x: torch.Tensor, epilogue) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = epilogue(x[tm, tn])
    return out


_EPILOGUE_SOURCE = """
def make(scale):
    def epilogue(t):
        return t * scale

    return epilogue
"""
_epilogues: dict[str, object] = {}
exec(_EPILOGUE_SOURCE, _epilogues)
_make_epilogue = _epilogues["make"]


def test_epilogues_of_the_same_code_and_closure_share_the_plan() -> None:
    # Two instances of a model each make their own epilogue closure; Helion
    # binds both to one BoundKernel (it keys functions by code and closure),
    # so the plan matches functions the same way, not by identity.
    x = torch.randn(16, 16)
    first, second = _make_epilogue(hl.constexpr(2.0)), _make_epilogue(hl.constexpr(2.0))
    run_direct(with_epilogue, [x, first])
    run = _run(with_epilogue, [x, first])
    assert _run(with_epilogue, [x, second]) is run and run.planned
    before = run.native_calls
    assert torch.equal(run_direct(with_epilogue, [x, second]), x * 2.0)
    assert torch.equal(run_direct(with_epilogue, [x, first]), x * 2.0)
    assert run.native_calls == before + 2
    # Equal code from another definition (a model loaded from two files).
    namespace: dict[str, object] = {}
    exec(_EPILOGUE_SOURCE, namespace)
    third = namespace["make"](hl.constexpr(2.0))
    assert third.__code__ == first.__code__ and third.__code__ is not first.__code__
    assert torch.equal(run(x, third), x * 2.0)
    assert run.native_calls == before + 3
    # Same code, a closure of another value type or another code: the
    # Python run of this BoundKernel (calling it directly), same result.
    other = _make_epilogue(hl.constexpr(2))

    def doubled(t):
        return t * 2.0

    for epilogue in (other, doubled):
        assert torch.equal(run(x, epilogue), run.general(x, epilogue))
    assert run.native_calls == before + 3


@pytest.mark.parametrize("mode", ["torch_function", "torch_dispatch"])
def test_active_modes_take_the_python_run(mode: str) -> None:
    run, x, y = _planned_scaled_add()
    before = run.native_calls
    seen: list[object] = []

    class FunctionSpy(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            seen.append(func)
            return func(*args, **(kwargs or {}))

    class DispatchSpy(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(func)
            return func(*args, **(kwargs or {}))

    with FunctionSpy() if mode == "torch_function" else DispatchSpy():
        out = run(x, y)
    torch.testing.assert_close(out, x * 2.0 + y)
    assert run.native_calls == before
    assert seen  # the mode saw the host code's torch.empty_like


def test_replaced_call_hooks_take_the_python_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, x, y = _planned_scaled_add()
    before = run.native_calls
    prepared: list[int] = []
    monkeypatch.setattr(
        CpuTarget, "prepare_call", lambda self, buffers: prepared.append(len(buffers))
    )
    torch.testing.assert_close(run(x, y), x * 2.0 + y)
    assert prepared == [3]
    assert run.native_calls == before


def test_host_code_with_torch_ops_has_no_native_run() -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    torch.testing.assert_close(run_direct(host_op_add, [x, y]), x + y)
    assert type(_run(host_op_add, [x, y])).__name__ != "KernelRun"


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def scaled_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0 - y[tm, tn]
    return out


@pytest.fixture
def native_switch():
    """native.set_native_call, switched back on after the test."""
    try:
        yield native.set_native_call
    finally:
        native.set_native_call(True)


def test_switch_off_uses_nothing_native(native_switch) -> None:
    # Compiled while off: the same kernel and inputs, nothing native, the
    # same bits.
    inputs = [(torch.randn(24, 8), torch.randn(24, 8)) for _ in range(3)]
    scaled_sub.reset()
    native_results = [run_direct(scaled_sub, [x, y]) for x, y in inputs]
    run = _run(scaled_sub, list(inputs[0]))
    assert run.native_calls == len(inputs) - 1
    native_switch(False)
    assert native.module() is None
    scaled_sub.reset()
    off_results = [run_direct(scaled_sub, [x, y]) for x, y in inputs]
    general = _run(scaled_sub, list(inputs[0]))
    assert type(general).__name__ == "function"
    cells = dict(zip(general.__code__.co_freevars, general.__closure__, strict=True))
    assert cells["entry"].cell_contents._native is None
    assert run.native_calls == len(inputs) - 1  # untouched
    for on, off in zip(native_results, off_results, strict=True):
        assert torch.equal(on, off)
    scaled_sub.reset()


def test_switch_off_applies_to_compiled_kernels(native_switch) -> None:
    # Off takes effect at once, also for kernels compiled natively: every
    # native path (KernelRun, PreparedRun, EntryCall) checks it per call.
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    run = _run(scaled_add, [x, y])
    assert run.planned
    bound = scaled_add.bind((x, y))
    prepared = helion_mlir_backend.prepared_call(
        bound, (x, y), [0], lambda a: bound(a, y)
    )
    cells = dict(
        zip(run.general.__code__.co_freevars, run.general.__closure__, strict=True)
    )
    entry_call = cells["entry"].cell_contents._native
    expected = run(x, y)
    before = run.native_calls
    native_switch(False)
    for _ in range(2):
        assert torch.equal(run_direct(scaled_add, [x, y]), expected)
        assert torch.equal(prepared(x), expected)
    assert not entry_call.call([torch.empty(16, 16), x, y])
    assert run.native_calls == before
    native_switch(True)
    assert torch.equal(run_direct(scaled_add, [x, y]), expected)
    assert run.native_calls == before + 1


def test_switch_environment_is_read_at_import() -> None:
    import os
    import subprocess
    import sys

    code = (
        "import os; from helion_mlir_backend._compiler import native as n; "
        "a = n.enabled(); os.environ[n.NATIVE_CALL_ENV] = '1'; print(a, n.enabled())"
    )
    env = {**os.environ, native.NATIVE_CALL_ENV: "0"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["False", "False"]


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def scaled_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] / y[tm, tn]
    return out


def test_planning_failure_leaves_the_call_to_the_python_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: an exception while classifying the first call's arguments
    # (_plan_params; e.g. metadata a tensor can't give) propagated instead of
    # leaving the call to the Python run.
    from helion_mlir_backend._compiler.mlir import driver as drv

    def unreadable(args: tuple[object, ...]) -> None:
        raise RuntimeError("an argument whose metadata can't be read")

    monkeypatch.setattr(drv, "_plan_params", unreadable)
    x, y = torch.randn(2, 16), torch.randn(2, 16)
    bound = scaled_div.bind((x, y))
    bound.ensure_config_exists((x, y))
    run = bound._run
    assert type(run).__name__ == "KernelRun" and not run.planned
    assert torch.equal(run(x, y), x / y)
    assert not run.planned  # the first call decides: no plan
    assert torch.equal(run(x, y), x / y)
    assert run.native_calls == 0


def test_entry_call_checks_match_call_in_place() -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    general = _run(scaled_add, [x, y]).general
    cells = dict(zip(general.__code__.co_freevars, general.__closure__, strict=True))
    entry: execution.CompiledEntry = cells["entry"].cell_contents
    native_call = entry._native
    out = torch.empty(16, 16)
    assert native_call.call([out, x, y])
    torch.testing.assert_close(out, x * 2.0 + y)
    for values in (
        [out, x.t(), y],  # strides
        [out, x.double(), y],  # dtype
        [out, torch.randn(8, 16), y],  # shape
        [out, torch._neg_view(x), y],  # negative bit
        [out, out, y],  # an input aliasing the inout
        [out, x],  # arity
        [out, x, 1.0],  # not a tensor
    ):
        assert not native_call.call(values)


# -- prepared calls ----------------------------------------------------------


# Each prepared-call test has its own shapes, so its own BoundKernel and plan.


def test_prepared_call_views_its_argument_and_result() -> None:
    x3, y3 = torch.randn(16, 4, 2), torch.randn(16, 4, 2)
    bound = add_rows_into.bind((x3.view(16, 8), y3.view(16, 8)))
    calls: list[torch.Tensor] = []

    def fallback(x: torch.Tensor) -> torch.Tensor:
        calls.append(x)
        return bound(x.reshape(16, 8), y3.view(16, 8)).view(128)

    prepared = helion_mlir_backend.prepared_call(
        bound,
        (x3.view(16, 8), y3.view(16, 8)),
        [0],
        fallback,
        views=[(16, 4, 2)],
        result_shape=(128,),
    )
    assert prepared is not None
    first = prepared(x3)  # before the plan exists: the fallback records it
    assert len(calls) == 1
    calls.clear()
    before = prepared.native_calls
    x3 = torch.randn(16, 4, 2)
    out = prepared(x3)
    assert prepared.native_calls == before + 1 and not calls
    assert out.shape == (128,) and out.is_contiguous()
    assert torch.equal(out, (x3 + y3).view(128))
    assert first.shape == out.shape
    # Anything else: the fallback.
    for other in (x3.view(16, 8), x3.transpose(0, 1).contiguous().transpose(0, 1)):
        assert torch.equal(
            prepared(other), (other.reshape(16, 8) + y3.view(16, 8)).view(128)
        )
    assert len(calls) == 2


def test_prepared_call_reads_bound_tensors_current_storage() -> None:
    x, y = torch.randn(24, 8), torch.randn(24, 8)
    bound = add_rows_into.bind((x, y))
    prepared = helion_mlir_backend.prepared_call(
        bound, (x, y), [0], lambda x: bound(x, y)
    )
    prepared(x)
    before = prepared.native_calls
    assert torch.equal(prepared(x), x + y)
    y.set_(torch.randn(24, 8))  # the bound tensor's storage changes
    assert torch.equal(prepared(x), x + y)
    assert prepared.native_calls == before + 2
    y.set_(torch.randn(8, 24).t())  # its metadata changes: the fallback
    assert torch.equal(prepared(x), x + y)
    assert prepared.native_calls == before + 2


def test_prepared_call_follows_a_recompiled_bound_kernel() -> None:
    x, y = torch.randn(32, 8), torch.randn(32, 8)
    bound = add_rows_into.bind((x, y))
    used: list[str] = []

    def fallback(x: torch.Tensor) -> torch.Tensor:
        used.append("fallback")
        return bound(x, y)

    prepared = helion_mlir_backend.prepared_call(bound, (x, y), [0], fallback)
    prepared(x)
    prepared(x)
    assert used == ["fallback"]  # the first call only
    run = bound._run
    before = prepared.native_calls
    try:
        bound._run = run.general  # e.g. set_config: another run
        assert torch.equal(prepared(x), x + y)
    finally:
        bound._run = run
    assert used == ["fallback", "fallback"]
    assert prepared.native_calls == before


def test_prepared_call_rejects_views_that_change_the_size() -> None:
    x, y = torch.randn(40, 8), torch.randn(40, 8)
    bound = add_rows_into.bind((x, y))
    calls: list[int] = []

    def fallback(x: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return bound(x[:40], y)

    prepared = helion_mlir_backend.prepared_call(
        bound, (x, y), [0], fallback, views=[(80, 8)]
    )
    big = torch.randn(80, 8)
    prepared(big)
    assert torch.equal(prepared(big), big[:40] + y)
    assert len(calls) == 2


# -- the host-code check -------------------------------------------------------


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def half_cols(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    half = max(int(n // 2), 1) if x.ndim == 2 else n
    out = torch.empty([m, half], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, half]):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def _host_function(kernel: helion.Kernel, args: list[object]) -> object:
    run = _run(kernel, args)
    general = getattr(run, "general", run)
    cells = dict(zip(general.__code__.co_freevars, general.__closure__, strict=True))
    return cells["host_function"].cell_contents


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        (scaled_add, lambda: [torch.randn(16, 16), torch.randn(16, 16)]),
        (add_rows_into, lambda: [torch.randn(8, 8), torch.randn(8, 8)]),
        (scaled_by, lambda: [torch.randn(16, 16), hl.constexpr(2.0)]),
        (half_cols, lambda: [torch.randn(16, 16)]),
    ],
)
def test_metadata_only_host_code(kernel: helion.Kernel, args) -> None:
    values = args()
    result = run_direct(kernel, values)
    assert _host_function(kernel, values)._helion_mlir_metadata_only
    if kernel is half_cols:
        torch.testing.assert_close(result, values[0][:, :8] * 2.0)


def test_host_code_with_a_tensor_op_is_not_metadata_only() -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(host_op_add, [x, y])
    assert not _host_function(host_op_add, [x, y])._helion_mlir_metadata_only


class _Typed:
    """A non-tensor Helion type for hand-written host code."""


def _metadata_only(
    source: str, namespace: dict | None = None, tensor: str = ""
) -> bool:
    """Check ``source`` as host code typed as Helion types it: every
    expression a non-tensor, except the names in ``tensor`` and what is
    computed from them."""
    from helion._compiler.type_info import TensorType

    func = ast.parse("def f(x, y, out):\n" + source).body[0]
    assert isinstance(func, ast.FunctionDef)
    tensor_type = TensorType.__new__(TensorType)
    for node in ast.walk(func):
        if isinstance(node, ast.expr):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            node._type_info = tensor_type if names & set(tensor.split()) else _Typed()  # type: ignore[attr-defined]
    return _MetadataOnly(func.args, namespace or {"torch": torch}).check(func.body)


def test_typed_shape_arithmetic_is_metadata_only() -> None:
    source = "    k = max(int(x.shape[1] // 2), 1) * x.stride(0)\n    yield locals()\n    return k\n"
    assert _metadata_only(source)
    # The same names as tensors: torch ops.
    assert not _metadata_only("    r = x * 2\n    return r\n", tensor="x")
    assert not _metadata_only("    r = x[0]\n    return r\n", tensor="x")
    assert not _metadata_only("    r = x.sum()\n    return r\n", tensor="x")


def test_untyped_computation_is_not_metadata_only() -> None:
    func = ast.parse("def f(x):\n    k = x.shape[0] + 1\n    return k\n").body[0]
    assert isinstance(func, ast.FunctionDef)
    assert not _MetadataOnly(func.args, {"torch": torch}).check(func.body)


@pytest.mark.parametrize(
    "source",
    [
        "    r = torch.zeros((4,), dtype=x.dtype, device=x.device)\n    return r\n",
        "    r = torch.empty((4,))\n    return r\n",  # default dtype and device
        "    r = torch.empty((4,), dtype=x.dtype, device=x.device, requires_grad=True)\n",
        "    x.shape[0] = 1\n",  # subscript assignment
        "    for i in range(2):\n        pass\n",
        "    r = GLOBAL\n    return r\n",  # a global
        "    return x.view(4)\n",
        "    r = len(x.shape)\n    return r.bit_length()\n",
        "    yield x\n",
        "    del x\n",
    ],
)
def test_host_code_that_is_not_metadata_only(source: str) -> None:
    assert not _metadata_only(source)


def test_shadowed_builtin_is_not_metadata_only() -> None:
    source = "    k = len(x.shape)\n    return k\n"
    assert _metadata_only(source)
    assert not _metadata_only(source, {"torch": torch, "len": print})
