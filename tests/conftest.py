"""Pytest configuration for the MLIR backend suite."""

from __future__ import annotations

import ctypes
import importlib
import multiprocessing
import multiprocessing.forkserver
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

# Imported once by the fork server, so an isolated test only pays for its own work.
_PRELOAD = [
    "torch",
    "helion",
    "helion.language",
    "mlir.ir",
    "lighthouse.pipeline.driver",
    "lighthouse.execution.runner",
    "torch_mlir.extras.fx_importer",
    "helion_mlir_backend",
    "helion_mlir_backend._compiler.execution",
    "helion_mlir_backend._compiler.mlir.driver",
    "tests.harness",
    "tests.ir_invariants",
]
_context: multiprocessing.context.ForkServerContext | None = None
_SKIP_EXIT = 77
"""Exit code of an isolated child whose test skipped; its last line says why."""
_SKIPPED = "isolated test skipped: "

if (_workers := os.environ.get("PYTEST_XDIST_WORKER_COUNT")) is not None:
    # xdist workers share the cores; per-process OpenMP/torch pools would oversubscribe.
    os.environ.setdefault(
        "OMP_NUM_THREADS", str(max(2, (os.cpu_count() or 2) // int(_workers)))
    )
# Test kernels run fixed configs; autotuning tests re-enable the search explicitly.
os.environ["HELION_DISALLOW_AUTOTUNING"] = "1"
# IR invariants on every lowering of the suite: see tests/ir_invariants.py.
_MODE = os.environ.setdefault("HELION_MLIR_TEST_IR_CHECKS", "fail")
# Node-local, as on NFS a worker can hang in nfs_lookup_revalidate, and with it the whole run.
# Hypothesis reads this at first use, after its pytest plugin has loaded and after this file.
HYPOTHESIS_STORAGE_NAMED = "HYPOTHESIS_STORAGE_DIRECTORY" in os.environ
os.environ.setdefault(
    "HYPOTHESIS_STORAGE_DIRECTORY", f"/dev/shm/hypothesis-{os.getuid()}"
)
# The same hang on bytecode, which workers write by replacing files next to the sources.
# The variable reaches xdist workers, the fork server and subprocesses.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def _isolated_context() -> multiprocessing.context.ForkServerContext:
    global _context
    if _context is None:
        _context = multiprocessing.get_context("forkserver")
        _context.set_forkserver_preload(_PRELOAD)
    return _context


def _run_isolated(
    module: str, name: str, kwargs: dict, env: dict[str, str], log: str
) -> None:
    """Child side: run one test function with the parent's environment."""
    _disable_core_dumps()
    fd = os.open(log, os.O_WRONLY)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.environ.clear()
    os.environ.update(env)
    # The fork server does not run pytest's session hooks.
    from tests import ir_invariants

    ir_invariants.install()
    try:
        getattr(importlib.import_module(module), name)(**kwargs)
    except pytest.skip.Exception as skipped:
        print(f"{_SKIPPED}{skipped.msg}", flush=True)
        sys.stderr.flush()
        os._exit(_SKIP_EXIT)
    sys.stdout.flush()
    sys.stderr.flush()


def _bounded(command: list[str]) -> str:
    """``command``'s output, whatever becomes of it: a hung machine still fails the test."""
    try:
        return subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return f"{command[0]}: {error}\n"


def _live_processes(pid: int) -> str:
    """A hung child and its own children (a forked lowering), by what ``ps`` shows: a
    non-dumpable child can't be attached to."""
    columns = "pid,ppid,etimes,time,stat,nlwp,comm"
    return (
        _bounded(["ps", "-o", columns, "-p", str(pid)])
        + _bounded(["ps", "-o", columns, "--no-headers", "--ppid", str(pid)])
    ).rstrip()


def _kill_hung(process: multiprocessing.Process) -> str:
    """Kills ``process`` and its children, and returns what they were: read before the kill,
    as its children are no longer its children after it. Bounded, and the kill always runs."""
    try:
        return _live_processes(process.pid)
    finally:
        _bounded(["pkill", "-KILL", "-P", str(process.pid)])
        process.kill()
        process.join()


def _disable_core_dumps() -> None:
    """A crashing probe would otherwise dump the whole process (seconds per crash)."""
    if sys.platform.startswith("linux"):
        pr_set_dumpable = 4
        ctypes.CDLL(None).prctl(pr_set_dumpable, 0, 0, 0, 0)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="re-record tests/golden/*.txt IR signatures instead of comparing",
    )


class _ExpectedViolations:
    """``expect_violation``: the call must hit the check, also in a fork or isolated child,
    so a named exemption that no longer fires fails instead of going stale."""

    @pytest.hookimpl(wrapper=True)
    def pytest_pyfunc_call(self, pyfuncitem: pytest.Function) -> object:
        marker = pyfuncitem.get_closest_marker("expect_violation")
        if marker is None or _MODE == "off":
            return (yield)
        with tempfile.NamedTemporaryFile(suffix=".fired", delete=False) as handle:
            fired = pathlib.Path(handle.name)
        os.environ["HELION_MLIR_TEST_EXPECTED_VIOLATION"] = f"{marker.args[0]}:{fired}"
        try:
            result = yield
            if not fired.read_text():
                pytest.fail(
                    f"expect_violation({marker.args[0]!r}) did not fire", pytrace=False
                )
            return result
        finally:
            del os.environ["HELION_MLIR_TEST_EXPECTED_VIOLATION"]
            fired.unlink()


def pytest_report_header() -> str:
    return f"IR invariants (HELION_MLIR_TEST_IR_CHECKS): {_MODE}"


def pytest_configure(config: pytest.Config) -> None:
    from tests import ir_invariants

    if not ir_invariants.valid_mode(_MODE):
        raise pytest.UsageError(
            f"HELION_MLIR_TEST_IR_CHECKS={_MODE!r}; expected fail, off or report:PATH"
        )
    config.pluginmanager.register(_ExpectedViolations())
    config.addinivalue_line(
        "markers",
        "expect_violation(check): the test's kernel breaks that IR invariant, for a reason "
        "stated by the test; the test fails if the check doesn't fire",
    )
    config.addinivalue_line(
        "markers",
        "isolated(timeout=None): run the test body in a forked child process so a "
        "native crash only fails that test; with a timeout in seconds, so does a hang",
    )
    config.addinivalue_line("markers", "slow: long-running test")
    config.addinivalue_line(
        "markers",
        "helion_internals: pins Helion internals the native kernel run relies on "
        "(host-code types, binding, call conventions); run first on a Helion upgrade",
    )


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """``-n auto`` (the default) parallelizes whole-suite runs only.

    Runs that name test files or node ids, or select with ``-k``, stay in-process:
    starting workers costs more than such runs take. Pass ``-n <count>`` to force.
    """
    targeted = bool(config.option.keyword) or any(
        not (config.invocation_params.dir / arg.split("::")[0]).is_dir()
        for arg in config.args
    )
    return 0 if targeted else None


def pytest_sessionstart(session: pytest.Session) -> None:
    """A native extension that fails to build only logs a warning, and its tests
    skip: fail unless it is disabled on purpose (``HELION_MLIR_NATIVE_CALL=0``)."""
    from tests import ir_invariants

    from helion_mlir_backend._compiler import native

    ir_invariants.install()
    if native.enabled() and native.module() is None:
        raise pytest.UsageError(
            f"the native extension could not be built: {native.load_error()}; set "
            "HELION_MLIR_NATIVE_CALL=0 to run without it"
        )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Isolated tests last, so the fork server finishes importing meanwhile."""
    items.sort(key=lambda item: item.get_closest_marker("isolated") is not None)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Start the fork server early so its imports overlap the in-process tests."""
    if any(item.get_closest_marker("isolated") for item in session.items):
        _isolated_context()
        multiprocessing.forkserver.ensure_running()


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    marker = pyfuncitem.get_closest_marker("isolated")
    if marker is None:
        return None
    if pyfuncitem.cls is not None:
        raise pytest.UsageError(
            "@pytest.mark.isolated supports module-level tests only"
        )
    kwargs = {
        name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames
    }
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as handle:
        log = handle.name
    try:
        process = _isolated_context().Process(
            target=_run_isolated,
            args=(
                pyfuncitem.module.__name__,
                pyfuncitem.originalname,
                kwargs,
                dict(os.environ),
                log,
            ),
        )
        process.start()
        process.join(marker.kwargs.get("timeout"))
        hung = process.is_alive()
        if hung:
            children = _kill_hung(process)
        output = pathlib.Path(log).read_text()
    finally:
        os.unlink(log)
    if hung:
        tail = "\n".join(output.splitlines()[-25:])
        pytest.fail(
            f"isolated test still running after {marker.kwargs['timeout']} s, killed:\n"
            f"{children}\n{tail}",
            pytrace=False,
        )
    if process.exitcode == _SKIP_EXIT:
        reasons = [
            line[len(_SKIPPED) :]
            for line in output.splitlines()
            if line.startswith(_SKIPPED)
        ]
        if reasons:
            pytest.skip(reasons[-1])
    if process.exitcode != 0:
        tail = "\n".join(output.splitlines()[-25:])
        how = (
            f"crashed with signal {-process.exitcode}"
            if process.exitcode < 0
            else f"failed (exit code {process.exitcode})"
        )
        pytest.fail(f"isolated test {how}:\n{tail}", pytrace=False)
    return True
