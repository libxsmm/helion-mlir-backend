"""The pool of large fresh tensors of the native CPU call paths (``native/``):
blocks reused once a tensor's last reference is gone, whatever the allocator."""

from __future__ import annotations

import copy
import gc
import itertools
import mmap
import os
import pickle
import resource
import subprocess
import sys
import textwrap
import threading
from typing import TYPE_CHECKING

import helion
import helion.language as hl
import pytest
import torch
from torch.overrides import TorchFunctionMode

from tests.harness import run_direct

import helion_mlir_backend
from helion_mlir_backend import output_pool
from helion_mlir_backend._compiler import native

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_HUGE = 2 * 1024 * 1024


@helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[32, 32])
)
def scaled(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def _stats() -> dict[str, int]:
    return output_pool.output_pool_stats()


@pytest.fixture(autouse=True)
def _pool(monkeypatch: pytest.MonkeyPatch) -> object:
    """A large empty pool per test, taking small tensors so tests stay small; the
    limit restored after."""
    limit = _stats()["max_bytes"]
    monkeypatch.setattr(output_pool, "MIN_BYTES", 128 * 1024)
    helion_mlir_backend.set_output_pool(0)
    helion_mlir_backend.set_output_pool(1 << 30)
    yield
    monkeypatch.undo()
    helion_mlir_backend.set_output_pool(0)
    helion_mlir_backend.set_output_pool(limit)


def _need_native() -> None:
    if native.module() is None:
        pytest.skip("the native extension is off or could not be built")


def _native_run(x: torch.Tensor) -> object:
    """The kernel's native run, its plan recorded; its first (general) output freed."""
    _need_native()
    run_direct(scaled, [x])
    run = scaled.bind((x,))._run
    assert run.planned
    return run


def _quiet_window_faults(work: Callable[[], None], budget: int) -> int:
    """Minor faults of the quietest of a few runs of ``work``, stopping at the first
    within ``budget``. The count covers the whole process: the NUMA balancer's scans
    can fault resident pages back in (``x`` alone is 1024 4 KiB pages): a rare run
    takes hundreds of faults with no pool miss, also with no kernel or pool at all.
    Such noise only adds; a block that lost its pages faults in every run."""
    fewest = None
    for _ in range(5):
        before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        work()
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before
        fewest = faults if fewest is None else min(fewest, faults)
        if fewest < budget:
            break
    return fewest


def test_freed_output_block_is_reused() -> None:
    x = torch.randn(256, 256)  # 256 KiB
    run = _native_run(x)
    calls = run.native_calls
    first = run(x)
    pointer = first.data_ptr()
    torch.testing.assert_close(first, x * 2.0, rtol=0, atol=0)
    del first
    assert _stats()["free_blocks"] == 1
    hits = _stats()["hits"]
    second = run(x)
    assert second.data_ptr() == pointer
    assert _stats()["hits"] == hits + 1
    assert _stats()["free_blocks"] == 0
    assert run.native_calls == calls + 2


def test_reused_block_is_written_whole() -> None:
    x = torch.randn(256, 256)
    run = _native_run(x)
    out = run(x)
    pointer = out.data_ptr()
    out.fill_(float("nan"))
    del out
    again = run(x)
    assert again.data_ptr() == pointer
    assert torch.equal(again, x * 2.0)


def test_held_output_keeps_its_block() -> None:
    x = torch.randn(256, 256)
    run = _native_run(x)
    first = run(x)
    second = run(x)
    assert first.data_ptr() != second.data_ptr()
    pointer = first.data_ptr()
    expected = first.clone()
    view = first[3:5]  # a view keeps the storage alive
    del first
    third = run(x)
    assert third.data_ptr() != pointer
    assert view.data_ptr() == pointer + 3 * 256 * 4
    assert torch.equal(view, expected[3:5])
    del view
    fourth = run(x)
    assert fourth.data_ptr() == pointer


def test_small_tensors_are_not_pooled() -> None:
    x = torch.randn(128, 128)  # 64 KiB
    run = _native_run(x)
    before = _stats()
    out = run(x)
    del out
    after = _stats()
    assert after["hits"] == before["hits"]
    assert after["misses"] == before["misses"]
    assert after["free_blocks"] == before["free_blocks"]


def test_huge_blocks_are_aligned_whole_huge_pages() -> None:
    x = torch.randn(512, 1024)  # exactly 2 MiB
    run = _native_run(x)
    out = run(x)
    assert out.data_ptr() % _HUGE == 0
    del out
    assert _stats()["free_bytes"] == _HUGE
    y = torch.randn(512 + 32, 1024)  # 2 MiB + 128 KiB: rounded up
    run = _native_run(y)
    out = run(y)
    assert out.data_ptr() % _HUGE == 0
    del out
    assert _stats()["free_bytes"] == 3 * _HUGE


def test_two_buffers_per_call_do_not_refault() -> None:
    """L2/66's pattern: two 4 MiB buffers per call, freed by the call's end. Under
    glibc they used to trim its heap: 1024+ faults per buffer on every call."""
    x = torch.randn(1024, 1024)
    run = _native_run(x)

    def call() -> None:
        y = run(x)
        z = run(y)
        del y, z

    def calls() -> None:
        for _ in range(20):
            call()

    for _ in range(3):
        call()
    misses = _stats()["misses"]
    # A few pages for Python's and the runtime's own use; not 2048 per call.
    assert _quiet_window_faults(calls, 500) < 500
    assert _stats()["misses"] == misses


def test_limit_evicts_oldest_and_keeps_nothing_over_it() -> None:
    small, large = torch.randn(256, 256), torch.randn(512, 256)  # 256, 512 KiB
    run_small, run_large = _native_run(small), _native_run(large)
    helion_mlir_backend.set_output_pool(0)
    helion_mlir_backend.set_output_pool(512 * 1024)
    out = run_small(small)
    del out
    assert _stats()["free_bytes"] == 256 * 1024
    out = run_large(large)  # evicts the older block
    del out
    assert _stats()["free_bytes"] == 512 * 1024
    assert _stats()["free_blocks"] == 1
    helion_mlir_backend.set_output_pool(256 * 1024)  # frees what is over
    assert _stats()["free_bytes"] == 0
    out = run_large(large)  # larger than the limit
    del out
    assert _stats()["free_blocks"] == 0


def test_free_block_count_is_bounded() -> None:
    x = torch.randn(256, 256)
    run = _native_run(x)
    held = [run(x) for _ in range(70)]
    del held
    assert _stats()["free_blocks"] == 64
    assert _stats()["free_bytes"] == 64 * 256 * 1024


def test_off_uses_torch_allocator_and_live_tensors_survive() -> None:
    x = torch.randn(256, 256)
    run = _native_run(x)
    live = run(x)
    helion_mlir_backend.set_output_pool(0)
    assert _stats()["free_blocks"] == 0
    before = _stats()
    other = run(x)
    assert torch.equal(other, x * 2.0)
    assert _stats()["misses"] == before["misses"]
    assert torch.equal(live, x * 2.0)
    del live  # over the limit now: unmapped, not kept
    assert _stats()["free_blocks"] == 0


def test_pooled_tensor_behaves_as_a_tensor() -> None:
    x = torch.randn(256, 256)
    run = _native_run(x)
    out = run(x)
    expected = x * 2.0
    array = out.numpy()  # shares the storage, keeps it alive
    pointer = out.data_ptr()
    del out
    held = run(x)
    assert held.data_ptr() != pointer
    assert torch.equal(torch.from_numpy(array), expected)
    held.resize_(128, 256)  # within the block; a pooled storage cannot grow
    torch.testing.assert_close(held[0, 0], expected[0, 0])
    with pytest.raises(RuntimeError, match="not resizable"):
        held.resize_(512, 512)
    assert not run(x).requires_grad
    assert torch.equal(held[:0], held[:0].clone())
    with torch.inference_mode():
        assert run(x).is_inference()
    shared = run(x).share_memory_()
    assert torch.equal(shared, expected)


def test_deterministic_fill_bypasses_the_pool() -> None:
    x = torch.randn(256, 256)
    run = _native_run(x)
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    before = _stats()
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        assert torch.equal(run(x), x * 2.0)
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.utils.deterministic.fill_uninitialized_memory = fill
    assert _stats()["misses"] == before["misses"]
    assert _stats()["hits"] == before["hits"]


def test_threads_never_share_a_block() -> None:
    x = torch.randn(256, 256)
    run = _native_run(x)
    expected = x * 2.0
    wrong: list[int] = []

    def work(seed: int) -> None:
        for _ in range(100):
            out = run(x)
            if not torch.equal(out, expected):
                wrong.append(seed)
            out.fill_(float(seed))
            del out

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert wrong == []


def _script(path: Path, body: str, **env: str) -> subprocess.CompletedProcess[str]:
    _need_native()
    head = """
        import os, torch, helion, helion.language as hl
        import helion_mlir_backend
        from helion_mlir_backend._compiler import native

        @helion.kernel(backend="mlir", static_shapes=True,
                       config=helion.Config(block_sizes=[32, 32]))
        def scaled(x):
            out = torch.empty_like(x)
            for tm, tn in hl.tile(x.size()):
                out[tm, tn] = x[tm, tn] * 2.0
            return out

        x = torch.randn(256, 256)
        scaled(x)
        run = scaled.bind((x,))._run
        helion_mlir_backend.output_pool.MIN_BYTES = 128 * 1024
        native.refresh_allocator()
        stats = helion_mlir_backend.output_pool.output_pool_stats
        """
    script = path / "script.py"  # Helion reads a kernel's source from its file
    script.write_text(textwrap.dedent(head) + textwrap.dedent(body))
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, **env},
    )


def test_environment_sets_the_limit(tmp_path: Path) -> None:
    done = _script(
        tmp_path,
        """
        print(stats()["max_bytes"])
        out = run(x)
        del out
        print(stats()["misses"], stats()["free_blocks"])
        """,
        HELION_MLIR_OUTPUT_POOL="0",
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[-3:] == ["0", "0", "0"]


def test_forked_child_can_free_to_the_pool(tmp_path: Path) -> None:
    """Forks while another thread takes the pool's lock: the child's lock is
    free, so a tensor freed in the child (no kernel run: no OpenMP) doesn't hang."""
    done = _script(
        tmp_path,
        """
        import threading
        held = run(x)
        stop = threading.Event()
        def churn():
            while not stop.is_set():
                run(x)
        thread = threading.Thread(target=churn)
        thread.start()
        for _ in range(50):
            pid = os.fork()
            if pid == 0:
                del held
                os._exit(0)
            _, status = os.waitpid(pid, 0)
            assert status == 0, status
        stop.set()
        thread.join()
        """,
    )
    assert done.returncode == 0, done.stderr


# -- host code asks the pool directly ---------------------------------------


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32]))
def scaled_host(x: torch.Tensor) -> torch.Tensor:
    out = output_pool.empty_like(x)
    assert x.dim() == 2, "2-D only"
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32]))
def scaled_host_empty(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = output_pool.empty((m, n), dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32]))
def host_op(x: torch.Tensor) -> torch.Tensor:
    y = x * 1.0  # a torch op: general path, not metadata-only
    out = output_pool.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = y[tm, tn] * 2.0
    return out


@pytest.mark.parametrize("kernel", [scaled_host, scaled_host_empty, host_op])
def test_host_code_reuses_blocks(kernel: helion.Kernel) -> None:
    x = torch.randn(256, 256)
    run_direct(kernel, [x])
    first = kernel(x)
    pointer = first.data_ptr()
    assert torch.equal(first, x * 2.0)
    del first
    hits = _stats()["hits"]
    second = kernel(x)
    assert second.data_ptr() == pointer
    assert _stats()["hits"] == hits + 1
    assert torch.equal(second, x * 2.0)


def test_pool_call_and_assert_are_metadata_only() -> None:
    _need_native()
    x = torch.randn(256, 256)
    for kernel in (scaled_host, scaled_host_empty):
        run_direct(kernel, [x])
        assert type(kernel.bind((x,))._run).__name__ == "KernelRun"
    run_direct(host_op, [x])
    assert type(host_op.bind((x,))._run).__name__ != "KernelRun"


def test_native_off_still_pools_host_code() -> None:
    x = torch.randn(256, 256)
    run_direct(scaled_host, [x])
    helion_mlir_backend.set_native_call(False)
    try:
        first = scaled_host(x)
        pointer = first.data_ptr()
        del first
        assert scaled_host(x).data_ptr() == pointer
    finally:
        helion_mlir_backend.set_native_call(True)


def test_host_code_sees_torch_function_modes() -> None:
    x = torch.randn(256, 256)
    run_direct(host_op, [x])
    seen: list[str] = []

    class Spy(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            seen.append(getattr(func, "__name__", ""))
            return func(*args, **(kwargs or {}))

    before = _stats()
    with Spy():
        out = host_op(x)
    assert "empty_like" in seen
    assert _stats()["hits"] == before["hits"]
    assert _stats()["misses"] == before["misses"]
    assert torch.equal(out, x * 2.0)


def test_functions_fall_back_to_torch() -> None:
    """Whatever the pool doesn't take gives what torch's functions give."""
    before = _stats()
    big = (256, 256)
    f32 = torch.float32
    cpu = torch.device("cpu")
    assert output_pool.empty((8, 8), dtype=f32, device=cpu).shape == (8, 8)
    assert output_pool.empty(big, dtype=f32, device=torch.device("meta")).is_meta
    with pytest.raises(RuntimeError):
        output_pool.empty((-1, 256), dtype=f32, device=cpu)
    with pytest.raises(TypeError):
        output_pool.empty(("a", 256), dtype=f32, device=cpu)
    assert output_pool.empty(big, dtype=torch.qint8, device=cpu).is_quantized
    x = torch.randn(256, 256)
    assert output_pool.empty_like(x[::2]).shape == (128, 256)
    assert output_pool.empty_like(x.t()).shape == big
    conjugate = torch.randn(256, 256, dtype=torch.complex64).conj()
    assert output_pool.empty_like(conjugate).is_conj() == (
        torch.empty_like(conjugate).is_conj()
    )

    class Sub(torch.Tensor):
        pass

    assert type(output_pool.empty_like(x.as_subclass(Sub))) is Sub
    assert _stats()["hits"] == before["hits"]
    assert _stats()["misses"] == before["misses"]


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ],
)
def test_every_dtype_is_a_plain_multi_dim_tensor(dtype: torch.dtype) -> None:
    sizes = (4, 256, 256)
    made = output_pool.empty(sizes, dtype=dtype, device=torch.device("cpu"))
    plain = torch.empty(sizes, dtype=dtype)
    assert made.shape == plain.shape and made.dtype == dtype
    assert made.is_contiguous()
    assert made.stride() == plain.stride()
    assert made._is_view() is False and made._base is None
    assert made.requires_grad is False and made.grad_fn is None
    made.fill_(1)  # in place under grad mode, as on a plain tensor
    assert bool(made[3, 255, 255])


def test_in_place_and_autograd_as_a_plain_tensor() -> None:
    made = output_pool.empty(
        (256, 256), dtype=torch.float32, device=torch.device("cpu")
    )
    source = torch.ones(256, 256, requires_grad=True)
    with torch.no_grad():
        no_grad = output_pool.empty(
            (256, 256), dtype=torch.float32, device=torch.device("cpu")
        )
    no_grad.copy_(source * 2.0)  # under grad mode: a view made in no_grad would raise
    made.copy_(source * 3.0)
    made.sum().backward()
    assert torch.equal(source.grad, torch.full((256, 256), 3.0))
    with torch.inference_mode():
        assert output_pool.empty_like(source).is_inference()


def test_storage_operations() -> None:
    cpu = torch.device("cpu")
    x = output_pool.empty((256, 256), dtype=torch.float32, device=cpu).fill_(2.0)
    # Pickling and deepcopy copy the data out.
    again = pickle.loads(pickle.dumps(x))
    twin = copy.deepcopy(x)
    assert torch.equal(again, x) and torch.equal(twin, x)
    assert again.data_ptr() != x.data_ptr() != twin.data_ptr()
    # set_ onto another storage: the block returns; set_ onto ours keeps it alive.
    other = torch.zeros(4)
    pointer = x.data_ptr()
    y = torch.empty(0).set_(x.untyped_storage(), 0, (256, 256))
    x.set_(other)
    del x
    assert _stats()["free_blocks"] == 0
    assert y.data_ptr() == pointer and bool((y == 2.0).all())
    del y
    assert _stats()["free_blocks"] == 1
    # share_memory_ moves the data to shared memory and gives the block back.
    z = output_pool.empty((256, 256), dtype=torch.float32, device=cpu).fill_(3.0)
    pointer = z.data_ptr()
    z.share_memory_()
    assert z.is_shared() and z.data_ptr() != pointer and bool((z == 3.0).all())
    assert _stats()["free_blocks"] == 1


def test_block_returned_inside_the_garbage_collector() -> None:
    """A tensor in a cycle dies at a collection, on any thread and inside any
    allocation, also while this thread holds the pool's lock: it must not wait."""
    cpu = torch.device("cpu")
    gc.collect()

    class Cycle:
        pass

    cycle = Cycle()
    cycle.me = cycle
    cycle.tensor = output_pool.empty((256, 256), dtype=torch.float32, device=cpu)
    del cycle
    with output_pool._lock:  # as _take holds it
        gc.collect()
    assert _stats()["free_blocks"] == 1


def test_interpreter_shutdown_with_live_blocks(tmp_path: Path) -> None:
    done = _script(
        tmp_path,
        """
        import atexit
        held = [run(x), run(x)]        # module globals at shutdown
        cycle = {"t": run(x)}; cycle["me"] = cycle
        atexit.register(lambda t=run(x): None)   # alive until atexit runs
        del cycle
        print("end")
        """,
    )
    assert done.returncode == 0, done.stderr
    assert "end" in done.stdout


def test_quiet_window_faults_skips_noise_but_not_lost_pages() -> None:
    def touch_fresh_pages() -> None:
        pages = mmap.mmap(-1, 512 * mmap.PAGESIZE)
        pages.madvise(mmap.MADV_NOHUGEPAGE)
        for page in range(512):
            pages[page * mmap.PAGESIZE] = 1
        pages.close()

    runs = []

    def noisy_once() -> None:
        runs.append(1)
        if len(runs) == 1:
            touch_fresh_pages()

    assert _quiet_window_faults(noisy_once, 200) < 200
    assert len(runs) >= 2
    assert _quiet_window_faults(touch_fresh_pages, 200) >= 200


def test_faults_only_on_first_use_of_a_block() -> None:
    """64 MiB, over glibc's largest mmap threshold (32 MiB): without the pool every call
    maps new pages. Smaller, glibc keeps one buffer and this would pass without the pool."""
    x = torch.ones(4096, 4096)
    run = _native_run(x)
    out = run(x)
    del out

    def calls() -> None:
        for _ in range(20):
            out = run(x)
            del out

    misses = _stats()["misses"]
    assert _quiet_window_faults(calls, 200) < 200
    assert _stats()["misses"] == misses


def test_extension_gets_the_allocator_only_while_on() -> None:
    seen: list[tuple] = []
    original = output_pool._native_allocate

    def spy(sizes, dtype):
        seen.append((sizes, dtype))
        return original(sizes, dtype)

    x = torch.randn(256, 256)
    run = _native_run(x)
    try:
        output_pool._native_allocate = spy
        native.refresh_allocator()
        run(x)
        assert seen == [((256, 256), torch.float32)]
        helion_mlir_backend.set_output_pool(0)
        seen.clear()
        assert torch.equal(run(x), x * 2.0)
        assert seen == []
    finally:
        output_pool._native_allocate = original
        native.refresh_allocator()


def test_allocator_failure_takes_at_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.randn(256, 256)
    run = _native_run(x)

    def fail(sizes, dtype):
        raise MemoryError

    monkeypatch.setattr(output_pool, "_native_allocate", fail)
    native.refresh_allocator()
    try:
        assert torch.equal(run(x), x * 2.0)
    finally:
        monkeypatch.undo()
        native.refresh_allocator()


# -- retention -------------------------------------------------------------


def test_failed_mapping_evicts_every_free_block_and_retries(tmp_path: Path) -> None:
    """One buffer live at a time, with free blocks taking the address space: as
    without a pool, every size must succeed."""
    done = _script(
        tmp_path,
        """
        import gc, resource
        helion_mlir_backend.output_pool.set_output_pool(8 << 30)
        vm = [l for l in open("/proc/self/status") if l.startswith("VmSize")][0]
        limit = (int(vm.split()[1]) // 1024 + 3000) << 20
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        cpu = torch.device("cpu")
        P = helion_mlir_backend.output_pool
        for mb in (300, 500, 700, 900, 1100, 1300):
            t = P.empty((mb << 20,), dtype=torch.uint8, device=cpu)
            t[0] = 1; t[-1] = 1
            del t; gc.collect()
            print(mb, P.output_pool_stats()["free_blocks"], flush=True)
        print("END")
        """,
        OMP_NUM_THREADS="1",
    )
    assert done.returncode == 0, done.stderr
    assert "END" in done.stdout
    counts = [int(line.split()[1]) for line in done.stdout.split("\n")[:-2]]
    assert len(counts) == 6
    assert any(b < a for a, b in itertools.pairwise(counts))  # evicted


def test_block_over_the_limit_leaves_the_others() -> None:
    small = torch.randn(256, 256)
    run = _native_run(small)
    helion_mlir_backend.set_output_pool(0)
    helion_mlir_backend.set_output_pool(512 * 1024)
    out = run(small)
    del out
    big = output_pool.empty((1 << 20,), dtype=torch.uint8, device=torch.device("cpu"))
    del big
    assert _stats()["free_bytes"] == 256 * 1024


def test_release_frees_and_live_tensors_survive() -> None:
    cpu = torch.device("cpu")
    live = output_pool.empty((256, 256), dtype=torch.float32, device=cpu).fill_(1.0)
    gone = output_pool.empty((256, 256), dtype=torch.float32, device=cpu)
    del gone
    assert _stats()["free_blocks"] == 1
    output_pool.release()
    assert _stats()["free_blocks"] == 0
    assert bool((live == 1.0).all())
    del live
    assert _stats()["free_blocks"] == 1


def test_forked_child_starts_with_no_free_blocks(tmp_path: Path) -> None:
    done = _script(
        tmp_path,
        """
        gone = run(x); held = run(x); del gone
        before = stats()["free_blocks"]
        value = float(held[0, 0])
        pid = os.fork()
        if pid == 0:
            ok = stats()["free_blocks"] == 0 and float(held[0, 0]) == value
            del held
            ok = ok and stats()["free_blocks"] == 1
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
        assert status == 0, status
        assert stats()["free_blocks"] == before
        print("OK")
        """,
    )
    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout


def test_default_threshold_leaves_small_buffers_to_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    native.refresh_allocator()
    assert output_pool.MIN_BYTES == 4 * 1024 * 1024
    before = _stats()
    cpu = torch.device("cpu")
    output_pool.empty((1024, 1024 - 1), dtype=torch.float32, device=cpu)
    assert _stats()["misses"] == before["misses"]
    output_pool.empty((1024, 1024), dtype=torch.float32, device=cpu)
    assert _stats()["misses"] == before["misses"] + 1


def test_default_limit() -> None:
    limit = output_pool._default_max_bytes()
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert 0 < limit <= min(output_pool.DEFAULT_MAX_BYTES, ram // 8)


def test_cgroup_limit_is_taken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(output_pool, "_cgroup_limit", lambda: 800 << 20)
    assert output_pool._default_max_bytes() == 100 << 20


@pytest.mark.parametrize("value", ["4G", "-1", "many"])
def test_garbage_environment_warns_and_keeps_the_default(
    tmp_path: Path, value: str
) -> None:
    done = _script(
        tmp_path,
        """
        from helion_mlir_backend._compiler import native as n
        out = run(x)
        assert n.module() is not None
        assert stats()["max_bytes"] == helion_mlir_backend.output_pool._default_max_bytes()
        assert stats()["max_bytes"] > 0
        print("OK")
        """,
        HELION_MLIR_OUTPUT_POOL=value,
    )
    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout
    assert "HELION_MLIR_OUTPUT_POOL" in done.stderr


# -- torch's own forms of the allocation calls -----------------------------


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32]))
def _dtype_none_empty(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=None, device=x.device)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32]))
def _dtype_none_empty_like(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=None)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32]))
def _dtype_none_pool(x: torch.Tensor) -> torch.Tensor:
    out = output_pool.empty(x.shape, dtype=None, device=x.device)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32]))
def _dtype_none_pool_like(x: torch.Tensor) -> torch.Tensor:
    out = output_pool.empty_like(x, dtype=None)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


@pytest.mark.parametrize(
    "kernel",
    [
        _dtype_none_empty,
        _dtype_none_empty_like,
        _dtype_none_pool,
        _dtype_none_pool_like,
    ],
)
def test_dtype_none_in_a_kernel_as_on_base(kernel: helion.Kernel) -> None:
    x = torch.randn(256, 256)
    for _ in range(3):
        assert torch.equal(kernel(x), x * 2.0)


@pytest.mark.parametrize("kind", [float, int, bool, None])
def test_python_types_as_dtype(kind: object) -> None:
    x = torch.randn(256, 256)
    cpu = torch.device("cpu")
    plain = torch.empty((256, 256), dtype=kind, device=cpu)
    made = output_pool.empty((256, 256), dtype=kind, device=cpu)
    assert made.dtype == plain.dtype
    assert output_pool.empty_like(x, dtype=kind).dtype == (
        torch.empty_like(x, dtype=kind).dtype
    )


def test_device_as_string_and_size_object() -> None:
    made = output_pool.empty(torch.Size((256, 256)), dtype=torch.float32, device="cpu")
    assert made.shape == (256, 256) and made.dtype == torch.float32


def test_non_main_thread_reuses_blocks() -> None:
    x = torch.randn(256, 256)
    run = _native_run(x)
    pointers: list[int] = []

    def work() -> None:
        for _ in range(3):
            out = run(x)
            pointers.append(out.data_ptr())
            del out

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert len(set(pointers)) == 1


# -- review: what the pool must leave to torch, and must not wait for ------


def test_output_over_the_limit_is_left_to_torch() -> None:
    """Mapped and unmapped per call it would cost more than it saves."""
    cpu = torch.device("cpu")
    helion_mlir_backend.set_output_pool(0)
    helion_mlir_backend.set_output_pool(512 * 1024)
    before = _stats()
    made = output_pool.empty((1024, 1024), dtype=torch.float32, device=cpu)
    del made
    after = _stats()
    assert after["misses"] == before["misses"] and after["hits"] == before["hits"]
    assert output_pool._native_allocate((1024, 1024), torch.float32) is None
    assert output_pool._takes(1024 * 128, torch.float32)
    assert not output_pool._takes(1024 * 128 + 1, torch.float32)


def test_impossible_size_raises_as_torch_does() -> None:
    cpu = torch.device("cpu")
    with pytest.raises(RuntimeError):
        torch.empty((1 << 62, 2), dtype=torch.float32, device=cpu)
    with pytest.raises(RuntimeError):
        output_pool.empty((1 << 62, 2), dtype=torch.float32, device=cpu)


def test_take_does_not_wait_on_a_held_lock() -> None:
    """A signal handler or finalizer allocating while this thread is in ``_settle``."""
    cpu = torch.device("cpu")
    gone = output_pool.empty((256, 256), dtype=torch.float32, device=cpu)
    del gone
    before = _stats()
    with output_pool._lock:
        made = output_pool.empty((256, 256), dtype=torch.float32, device=cpu)
    assert _stats()["hits"] == before["hits"]  # a miss, though a block was free
    del made
    assert _stats()["free_blocks"] == before["free_blocks"] + 1


def test_handler_allocating_during_allocations_does_not_hang(tmp_path: Path) -> None:
    done = _script(
        tmp_path,
        """
        import faulthandler, signal, time
        P = helion_mlir_backend.output_pool
        faulthandler.dump_traceback_later(60, exit=True)
        cpu = torch.device("cpu")
        hits = [0]
        busy = [False]
        def handler(*_):
            if not busy[0]:
                busy[0] = True
                hits[0] += 1
                y = P.empty((1 << 19,), dtype=torch.float32, device=cpu)
                del y
                busy[0] = False
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, 0.0003, 0.0003)
        start = time.time()
        while time.time() - start < 3:
            t = P.empty((1 << 18,), dtype=torch.float32, device=cpu)
            del t
        signal.setitimer(signal.ITIMER_REAL, 0)
        print("handler ran", hits[0])
        """,
    )
    assert done.returncode == 0, done.stderr
    assert "handler ran" in done.stdout


def test_traced_allocation_is_left_to_torch() -> None:
    def twice(x: torch.Tensor) -> torch.Tensor:
        return output_pool.empty_like(x)

    x = torch.randn(256, 256)
    before = _stats()
    torch.jit.trace(twice, x, check_trace=False)
    assert _stats()["misses"] == before["misses"]
    assert _stats()["hits"] == before["hits"]


def test_missing_huge_page_advice_is_only_a_lost_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(mmap, "MADV_HUGEPAGE")
    output_pool.release()
    cpu = torch.device("cpu")
    made = output_pool.empty((1024, 1024), dtype=torch.float32, device=cpu)
    assert made.shape == (1024, 1024)


def test_empty_like_keeps_torchs_strides_of_size_one_dims() -> None:
    for x in (
        torch.randn(1, 65536),
        torch.randn(65536, 1).t(),
        torch.randn(65536, 1),
        torch.randn(1, 65536).t(),
        torch.randn(4, 1, 65536).permute(1, 0, 2),
    ):
        assert output_pool.empty_like(x).stride() == torch.empty_like(x).stride()
