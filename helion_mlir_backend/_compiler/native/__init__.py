"""Native call paths of compiled CPU entries: a small C++ extension.

Per-call Python host work is a real share of a small kernel's call, and in a loop
after a many-threaded kernel it runs slower than alone: the master thread's caches
are cold and its clock is the all-core one. ``call.cpp`` does a call's checks
and argument packing natively:

- ``EntryCall``: :meth:`CompiledEntry.call_in_place`'s checks and call;
- ``KernelRun``: a compiled kernel's ``run(*args)`` for the argument metadata
  of the plan its first call recorded (``driver``), skipping the host
  generator;
- ``PreparedRun``: a KernelRun with its other arguments bound
  (``driver.prepared_call``).

All take the unchanged general path for anything they don't handle exactly.

The extension is built with the system C++ compiler on first use, once per
torch, Python, compiler, flags and source version (for the generic ISA, so
the library loads on every node sharing the cache), into
``HELION_MLIR_NATIVE_CACHE`` (default ``~/.cache/helion_mlir_backend/native``)
under a file lock: concurrent processes (pytest workers) build it once.
Without a compiler or on a build failure or timeout, the Python paths are
used. :func:`set_native_call` (``helion_mlir_backend.set_native_call``) turns
every native path off or on; ``HELION_MLIR_NATIVE_CALL=0`` starts the process
with them off (read once, at import). Off applies to every call at once.

A kernel's first call decides: if it can't be planned (e.g. two arguments the
same tensor), that kernel (BoundKernel) never runs natively in the process,
which costs speed only.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from typing import TYPE_CHECKING

import torch

from .. import output_pool

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

log = logging.getLogger(__name__)

NATIVE_CALL_ENV = "HELION_MLIR_NATIVE_CALL"
NATIVE_CACHE_ENV = "HELION_MLIR_NATIVE_CACHE"
_MODULE_NAME = "_helion_mlir_native"
_SOURCE = Path(__file__).with_name("call.cpp")
_BUILD_TIMEOUT = 300

_lock = threading.Lock()
_loaded: list[ModuleType | None] = []
_load_error: list[str] = []


def _from_environment() -> bool:
    value = os.environ.get(NATIVE_CALL_ENV, "").strip().lower()
    return value not in ("0", "false", "no", "off")


_enabled = _from_environment()


def enabled() -> bool:
    """Whether the native paths are on (:func:`set_native_call`)."""
    return _enabled


def set_native_call(enabled: bool) -> None:
    """Turn the native call paths on or off for the whole process.
    ``HELION_MLIR_NATIVE_CALL`` gives the initial state, read once at import.
    Off takes effect at once for every kernel, compiled or not (each native
    call checks it): every call takes the Python paths. On applies to kernels
    compiled from then on (compiled while off, a kernel has no native run)."""
    global _enabled
    _enabled = bool(enabled)
    if _loaded and _loaded[0] is not None:
        _loaded[0].set_enabled(_enabled)


def refresh_allocator() -> None:
    """Give the extension the output pool's allocator, or none while it is off."""
    if _loaded and _loaded[0] is not None:
        _loaded[0].set_fresh_allocator(
            output_pool._native_allocate if output_pool.enabled() else None,
            output_pool.MIN_BYTES,
        )


def module() -> ModuleType | None:
    """The extension module, built on first use; ``None`` if disabled or it
    can't be built (logged once)."""
    if not enabled():
        return None
    if _loaded:
        return _loaded[0]
    with _lock:
        if not _loaded:
            try:
                _loaded.append(_load())
            except Exception as exc:
                log.warning(
                    "Helion MLIR backend: native call paths unavailable, using the "
                    "Python ones: %s",
                    exc,
                )
                _load_error.append(str(exc))
                _loaded.append(None)
            refresh_allocator()
    return _loaded[0]


def load_error() -> str | None:
    """Why :func:`module` found no extension module, if it failed to build."""
    return _load_error[0] if _load_error else None


def _compiler() -> str:
    for name in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if name and (path := shutil.which(name)):
            return path
    raise RuntimeError("no C++ compiler found (set CXX)")


def _torch_dir() -> Path:
    return Path(torch.__file__).resolve().parent


def _command(compiler: str, output: str) -> list[str]:
    torch_dir = _torch_dir()
    lib = torch_dir / "lib"
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    return [
        compiler,
        "-O2",
        "-std=c++17",
        "-shared",
        "-fPIC",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        "-isystem",
        str(torch_dir / "include"),
        "-isystem",
        str(torch_dir / "include" / "torch" / "csrc" / "api" / "include"),
        "-isystem",
        sysconfig.get_paths()["include"],
        str(_SOURCE),
        "-o",
        output,
        f"-L{lib}",
        f"-Wl,-rpath,{lib}",
        "-lc10",
        "-ltorch",
        "-ltorch_cpu",
        "-ltorch_python",
    ]


def _key(compiler: str, command: list[str]) -> str:
    """The cache key: source, torch, Python, compiler and flags. No host: the
    build targets the generic ISA (no ``-march``), so its library loads on
    every node sharing the cache."""
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True, timeout=60
    ).stdout
    parts = [
        _SOURCE.read_bytes().decode(),
        torch.__version__,
        str(torch.version.git_version),
        sys.version,
        platform.machine(),
        version,
        *command,
    ]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:20]


def _cache_dir() -> Path:
    root = os.environ.get(NATIVE_CACHE_ENV)
    if root:
        return Path(root)
    return Path.home() / ".cache" / "helion_mlir_backend" / "native"


def _load() -> ModuleType:
    """The cached library, or one built now: under a file lock (one builder
    at a time, waiting at most the build timeout), into a temporary file that
    is loaded before it is renamed into place, so a library that doesn't
    load never enters the cache."""
    compiler = _compiler()
    directory = _cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    output = "{output}"
    key = _key(compiler, _command(compiler, output))
    path = directory / f"{_MODULE_NAME}-{key}{suffix}"
    if path.exists():
        return _import(path)
    with _file_lock(directory / f"{_MODULE_NAME}-{key}.lock"):
        if path.exists():
            return _import(path)
        fd, temporary = tempfile.mkstemp(dir=directory, suffix=suffix)
        os.close(fd)
        try:
            result = subprocess.run(
                _command(compiler, temporary),
                capture_output=True,
                text=True,
                timeout=_BUILD_TIMEOUT,
            )
            if result.returncode:
                raise RuntimeError(
                    f"building {_SOURCE.name} failed:\n{result.stderr[-4000:]}"
                )
            loaded = _import(Path(temporary))
            os.replace(temporary, path)
            return loaded
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _import(path: Path) -> ModuleType:
    loader = importlib.machinery.ExtensionFileLoader(_MODULE_NAME, str(path))
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path, loader=loader)
    assert spec is not None
    loaded = importlib.util.module_from_spec(spec)
    loader.exec_module(loaded)
    loaded.set_enabled(_enabled)
    return loaded


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """An exclusive ``fcntl`` lock on ``path``; TimeoutError after the build
    timeout (the caller then uses the Python paths)."""
    with open(path, "a") as handle:
        deadline = time.monotonic() + _BUILD_TIMEOUT
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"waited {_BUILD_TIMEOUT} s for {path}"
                    ) from None
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
