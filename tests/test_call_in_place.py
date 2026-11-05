"""The in-place call path (``CompiledEntry.call_in_place``).

A call whose arguments are all tensors of the entry's static shapes and dtypes,
contiguous, on the target's device and not aliasing an inout (except as the
general path passed it) skips the general path of ``driver.call_entry`` (checks,
staging, and hooks unless the target runs them in place): its arguments are
packed once per entry. Every other call must take the general path, with its
results and errors unchanged.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import run_direct

from helion_mlir_backend._compiler import execution
from helion_mlir_backend._compiler.mlir import driver
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.targets.cpu import CpuTarget

if TYPE_CHECKING:
    from collections.abc import Sequence


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


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def add_int(x: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + k
    return out


@helion.kernel(backend="mlir", static_shapes=False, config=_cfg(8))
def add_one_dynamic(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def add_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _, n = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = x[tm, 0, tn] + y[tm, 0, tn]
    return out


@pytest.fixture
def general_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The argument names the general path checked, in order."""
    seen: list[str] = []
    checked = driver._checked_tensor

    def spy(target, arg, value):
        seen.append(arg.name)
        return checked(target, arg, value)

    monkeypatch.setattr(driver, "_checked_tensor", spy)
    return seen


def test_static_contiguous_call_is_in_place(general_calls: list[str]) -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    torch.testing.assert_close(run_direct(scaled_add, [x, y]), x * 2.0 + y)
    general_calls.clear()
    # Twice: the packed arguments are reused with new data pointers.
    for _ in range(2):
        x, y = torch.randn(16, 16), torch.randn(16, 16)
        torch.testing.assert_close(run_direct(scaled_add, [x, y]), x * 2.0 + y)
    assert general_calls == []


def test_inout_parameter_is_written_in_place(general_calls: list[str]) -> None:
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    general_calls.clear()
    a, b = torch.randn(16), torch.zeros(16)
    assert run_direct(copy_plus_one, [a, b]) is None
    torch.testing.assert_close(b, a + 1.0)
    assert general_calls == []


def test_non_contiguous_input_takes_general_path(general_calls: list[str]) -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    general_calls.clear()
    xt = x.t()
    torch.testing.assert_close(run_direct(scaled_add, [xt, y]), xt * 2.0 + y)
    assert general_calls


def test_unit_dim_with_other_stride_takes_general_path(
    general_calls: list[str],
) -> None:
    x, y = torch.randn(8, 1, 16), torch.randn(8, 1, 16)
    torch.testing.assert_close(run_direct(add_rows, [x, y]), (x + y)[:, 0])
    general_calls.clear()
    # Contiguous for torch, but not the strides the packed descriptor holds.
    odd = torch.randn(8, 16).as_strided((8, 1, 16), (16, 1, 1))
    assert odd.is_contiguous()
    torch.testing.assert_close(run_direct(add_rows, [odd, y]), (odd + y)[:, 0])
    assert general_calls


def test_input_aliasing_an_inout_takes_general_path(general_calls: list[str]) -> None:
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    general_calls.clear()
    x = torch.randn(16)
    expected = x + 1.0
    run_direct(copy_plus_one, [x, x])
    torch.testing.assert_close(x, expected)
    assert general_calls
    # A partial overlap too.
    general_calls.clear()
    buffer = torch.randn(24)
    a, b = buffer[:16], buffer[8:]
    expected = a + 1.0
    run_direct(copy_plus_one, [a, b])
    torch.testing.assert_close(b, expected)
    assert general_calls


def test_inouts_sharing_memory_are_still_rejected() -> None:
    a, b = torch.randn(16), torch.randn(16)
    run_direct(write_both, [a, b])
    x = torch.randn(16)
    with pytest.raises(UnsupportedOperationError, match="share memory"):
        run_direct(write_both, [x, x])


def test_wrong_dtype_reports_the_general_error() -> None:
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    entry = _entry(copy_plus_one, [a, b])
    with pytest.raises(ValueError, match="dtype"):
        driver.call_entry(entry, [b, a.double()])


def test_dynamic_entries_have_no_in_place_path(general_calls: list[str]) -> None:
    x = torch.randn(16)
    torch.testing.assert_close(run_direct(add_one_dynamic, [x]), x + 1.0)
    assert _entry(add_one_dynamic, [x])._static is None
    general_calls.clear()
    torch.testing.assert_close(run_direct(add_one_dynamic, [x]), x + 1.0)
    assert general_calls


def test_scalar_entries_take_each_call_s_value_in_place(
    general_calls: list[str],
) -> None:
    # A static entry with a runtime scalar: no native path, the Python in-place
    # path, and the value of this call (never an earlier call's).
    x = torch.randn(16)
    torch.testing.assert_close(run_direct(add_int, [x, 3]), x + 3)
    entry = _entry(add_int, [x, 3])
    assert entry._static is not None and entry._native is None
    general_calls.clear()
    for k in (4, -7, 0, 2**40):
        torch.testing.assert_close(run_direct(add_int, [x, k]), x + k)
    x2 = torch.randn(16)
    torch.testing.assert_close(run_direct(add_int, [x2, 5]), x2 + 5)
    assert not general_calls
    # Not a number: the same error as on the general path.
    out = torch.empty(16)
    with pytest.raises(ValueError, match="invalid literal"):
        driver.call_entry(
            entry,
            [
                out if a.role == "inout" else {"x": x, "k": "a"}[a.name]
                for a in entry.args
            ],
        )


def _fake_entry(
    target: execution.MLIRTarget,
    roles: list[str],
    calls: list[str],
    read_order: object | None = None,
    size: int = 16,
) -> execution.CompiledEntry:
    """A ``CompiledEntry`` over 1D float32 tensors of ``size`` elements, named
    ``a0``, ``a1``..., whose JIT function only records the call."""
    args = tuple(
        execution.EntryArg(f"a{i}", role, None, (size,), torch.float32)
        for i, role in enumerate(roles)
    )
    runner = mock.Mock()
    runner.engine.lookup.return_value = lambda packed: calls.append("run")
    return execution.CompiledEntry("fake", args, runner, target, read_order)


class _HookedTarget(CpuTarget):
    """The CPU's checks, no native paths and hooks that run on the in-place path."""

    hooks_in_place = True
    native_in_place = False

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    def prepare_call(self, buffers: Sequence[torch.Tensor]) -> None:
        self.calls.append(f"prepare:{len(buffers)}")

    def finish_call(self, buffers: Sequence[torch.Tensor]) -> None:
        self.calls.append(f"finish:{len(buffers)}")


def test_hooks_in_place_run_around_the_call() -> None:
    calls: list[str] = []
    entry = _fake_entry(_HookedTarget(calls), ["in", "inout"], calls)
    a, b = torch.zeros(16), torch.zeros(16)
    assert entry.call_in_place([a, b]) == [b]
    assert calls == ["prepare:2", "run", "finish:2"]


def test_hooks_in_place_run_on_a_target_with_native_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The native call runs no hooks: it must not go first and skip them.
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    entry = _entry(copy_plus_one, [a, b])
    if entry._native is None:
        pytest.skip("no native call path")
    calls: list[str] = []
    monkeypatch.setattr(CpuTarget, "hooks_in_place", True)
    monkeypatch.setattr(
        CpuTarget, "prepare_call", lambda self, buffers: calls.append("prepare")
    )
    monkeypatch.setattr(
        CpuTarget, "finish_call", lambda self, buffers: calls.append("finish")
    )
    b.zero_()
    values = [{"a": a, "b": b}[arg.name] for arg in entry.args]
    (written,) = entry.call_in_place(values)
    assert written is b
    assert calls == ["prepare", "finish"]
    torch.testing.assert_close(b, a + 1.0)


def test_hooks_not_in_place_refuse_the_call() -> None:
    calls: list[str] = []
    target = _HookedTarget(calls)
    target.hooks_in_place = False
    entry = _fake_entry(target, ["in", "inout"], calls)
    assert entry.call_in_place([torch.zeros(16), torch.zeros(16)]) is None
    assert not calls


def test_a_failing_prepare_hook_runs_nothing() -> None:
    calls: list[str] = []
    target = _HookedTarget(calls)
    target.prepare_call = mock.Mock(side_effect=ValueError("several devices"))
    entry = _fake_entry(target, ["in", "inout"], calls)
    with pytest.raises(ValueError, match="several devices"):
        entry.call_in_place([torch.zeros(16), torch.zeros(16)])
    assert not calls
    assert not entry._static.lock.locked()


@pytest.mark.parametrize("seed", range(20))
def test_overlaps_match_a_scan_of_all_pairs(seed: int) -> None:
    random = torch.Generator().manual_seed(seed)
    roles = [
        "inout" if int(v) else "in" for v in torch.randint(0, 2, (9,), generator=random)
    ]
    static = _fake_entry(_HookedTarget([]), roles, [])._static
    for _ in range(50):
        pointers = [int(v) for v in torch.randint(0, 200, (9,), generator=random)]
        expected = 0
        for i in range(9):
            for j in range(i + 1, 9):
                if roles[i] == "inout" or roles[j] == "inout":
                    if (
                        pointers[i] < pointers[j] + 64
                        and pointers[j] < pointers[i] + 64
                    ):
                        expected |= 1 << (i * 9 + j)
        assert static.overlaps(pointers) == expected


def test_overlaps_ignore_where_the_tensors_are_and_input_pairs() -> None:
    static = _fake_entry(_HookedTarget([]), ["in", "in", "inout"], [])._static
    assert static.overlaps([0, 8, 1000]) == 0  # two inputs overlap: nothing to do
    near = static.overlaps([0, 1000, 8])
    assert near != 0
    assert static.overlaps([5000, 9000, 5008]) == near
    assert static.overlaps([5008, 9000, 5000]) == near  # the other way round
    # The memo of the last pointers answers for those pointers only.
    assert static.overlaps([5008, 9000, 7000]) == 0
    assert static.overlaps([5008, 9000, 5000]) == near


_READS_AFTER_WRITES = SimpleNamespace(after={"a0": frozenset({"a2"})}, same={})
"""A ``ReadOrder``: ``a0`` is read after ``a2`` is written."""


def _shared(offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """An input overlapping an output by all but ``offset`` elements, aligned."""
    storage = torch.zeros(16 + offset)
    return storage[:16], storage[offset:]


def test_overlap_passed_as_is_repeats_in_place() -> None:
    # The kernel reads the view after writing its base: the general path passes
    # it as is, and later calls with the same overlap (at other addresses) go
    # in place, as the general path would pass them.
    calls: list[str] = []
    target = _HookedTarget(calls)
    target.phases_see_aliases = True
    entry = _fake_entry(target, ["in", "in", "inout"], calls, _READS_AFTER_WRITES)
    view, out = _shared()
    other = torch.zeros(16)
    assert entry.call_in_place([view, other, out]) is None
    assert not calls
    driver.call_entry(entry, [view, other, out])
    assert calls == ["prepare:3", "run", "finish:3"]
    calls.clear()
    view2, out2 = _shared()
    assert entry.call_in_place([view2, other, out2]) == [out2]
    assert calls == ["prepare:3", "run", "finish:3"]
    # Another overlap isn't the one decided: general path, which copies it
    # (a1 isn't read after a write) and sees the call's own values.
    calls.clear()
    view3, out3 = _shared()
    assert entry.call_in_place([view3, out3, out3]) is None
    assert not calls
    # Tensors that don't overlap need no decision.
    assert entry.call_in_place([torch.zeros(16), other, torch.zeros(16)]) is not None


def test_overlap_the_general_path_copies_stays_on_it() -> None:
    # No read-after-write: each call's input is a copy, so never in place.
    calls: list[str] = []
    entry = _fake_entry(_HookedTarget(calls), ["in", "inout"], calls)
    view, out = _shared()
    driver.call_entry(entry, [view, out])
    assert not entry._static.passed_as_is
    calls.clear()
    assert entry.call_in_place([view, out]) is None
    assert not calls


def test_refused_overlap_stays_refused_on_every_call() -> None:
    # A read after a write that this target can't honour: the error each time.
    calls: list[str] = []
    entry = _fake_entry(
        _HookedTarget(calls), ["in", "in", "inout"], calls, _READS_AFTER_WRITES
    )
    view, out = _shared()
    other = torch.zeros(16)
    for _ in range(2):
        assert entry.call_in_place([view, other, out]) is None
        with pytest.raises(UnsupportedOperationError, match="read after"):
            driver.call_entry(entry, [view, other, out])
    assert not entry._static.passed_as_is
    assert not calls


def test_two_written_views_of_one_buffer_stay_refused() -> None:
    calls: list[str] = []
    entry = _fake_entry(_HookedTarget(calls), ["inout", "inout"], calls)
    a, b = _shared(4)
    for _ in range(2):
        assert entry.call_in_place([a, b]) is None
        with pytest.raises(UnsupportedOperationError, match="share memory"):
            driver.call_entry(entry, [a, b])
    assert not calls


def test_busy_packed_arguments_take_general_path(general_calls: list[str]) -> None:
    # The Python in-place path (no native EntryCall) packs into one buffer
    # per entry; the native paths pack on the stack and take no lock.
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    entry = _entry(scaled_add, [x, y])
    values = [{"x": x, "y": y}.get(arg.name, torch.empty(16, 16)) for arg in entry.args]
    static = entry._static
    native_call = entry._native
    object.__setattr__(entry, "_native", None)
    general_calls.clear()
    assert static.lock.acquire(blocking=False)
    try:
        assert entry.call_in_place(values) is None
        driver.call_entry(entry, values)
    finally:
        static.lock.release()
        object.__setattr__(entry, "_native", native_call)
    torch.testing.assert_close(values[0], x * 2.0 + y)
    assert general_calls


def test_concurrent_calls_are_correct() -> None:
    inputs = [(torch.randn(16, 16), torch.randn(16, 16)) for _ in range(8)]
    run_direct(scaled_add, list(inputs[0]))
    results: list[torch.Tensor | None] = [None] * len(inputs)

    def work(i: int) -> None:
        for _ in range(50):
            results[i] = run_direct(scaled_add, list(inputs[i]))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(inputs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for (x, y), result in zip(inputs, results, strict=True):
        torch.testing.assert_close(result, x * 2.0 + y)


def test_packed_descriptors_match_the_general_packing() -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    entry = _entry(scaled_add, [x, y])
    out = torch.empty(16, 16)
    values = [
        out if arg.role == "inout" else {"x": x, "y": y}[arg.name] for arg in entry.args
    ]
    static = entry._static
    for slot, value in zip(static.slots, values, strict=True):
        static.buffer[slot] = static.buffer[slot + 1] = value.data_ptr()
    _, (general, _) = execution._packed_args(entry.args, values)
    descriptors = len(general) - 2 * len(values)
    assert list(static.buffer)[:descriptors] == list(general)[:descriptors]


def _entry(kernel: helion.Kernel, args: list[object]) -> execution.CompiledEntry:
    """The compiled entry ``kernel`` runs for ``args``."""
    run = kernel.bind(tuple(args))._run
    run = getattr(run, "general", run)  # the Python run behind a native KernelRun
    cells = dict(zip(run.__code__.co_freevars, run.__closure__, strict=True))
    return cells["entry"].cell_contents


def test_negative_view_input_is_read_negated() -> None:
    # Regression: a negative view (torch._neg_view) shares its storage with
    # the bit set; both call paths passed the storage as is (wrong sign).
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    neg = torch._neg_view(x)
    assert neg.is_neg()
    torch.testing.assert_close(run_direct(scaled_add, [neg, y]), -x * 2.0 + y)
    # The general path (a non-contiguous negative view).
    neg_t = torch._neg_view(x.t())
    torch.testing.assert_close(run_direct(scaled_add, [neg_t, y]), -x.t() * 2.0 + y)


def test_negative_view_inout_is_written_negated() -> None:
    a, b = torch.randn(16), torch.zeros(16)
    run_direct(copy_plus_one, [a, b])
    storage = torch.zeros(16)
    out = torch._neg_view(storage)
    run_direct(copy_plus_one, [a, out])
    torch.testing.assert_close(out, a + 1.0)
    torch.testing.assert_close(storage, -(a + 1.0))


def test_tensor_the_target_rejects_reports_the_device_error(
    general_calls: list[str],
) -> None:
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    run_direct(scaled_add, [x, y])
    entry = _entry(scaled_add, [x, y])
    values = [{"x": x, "y": y}.get(arg.name, torch.empty(16, 16)) for arg in entry.args]
    general_calls.clear()
    # Static shape, dtype and strides match: only passes_in_place refuses it.
    values[0] = values[0].to("meta")
    with pytest.raises(NotImplementedError, match="only CPU is supported"):
        driver.call_entry(entry, values)
    assert general_calls
