"""Shared helpers for MLIR backend tests."""

from __future__ import annotations

import contextlib
import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import helion
import helion.language as hl
from lighthouse.execution.target import TargetInfo
from lighthouse.pipeline.driver import BackendDriver
import torch
from torch._subclasses import FakeTensorMode

from tests.inputs import same_values

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.backend import MLIRBackend

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from types import ModuleType

    from mlir import ir


@contextlib.contextmanager
def use_pipeline(pipeline: str | None) -> Iterator[None]:
    """Default compiles in the block to ``pipeline`` (``None``: unchanged)."""
    if pipeline is None:
        yield
        return
    with patch.dict(os.environ, {"HELION_MLIR_PIPELINE": pipeline}):
        yield


def scalar_pipeline() -> contextlib.AbstractContextManager[None]:
    return use_pipeline("scalar")


def opt_pipeline() -> contextlib.AbstractContextManager[None]:
    return use_pipeline("opt")


_FAKE_MODE = FakeTensorMode()


def load_example(relative_path: str) -> ModuleType:
    """The module of a script under ``examples/``, run as a library."""
    path = Path(__file__).resolve().parent.parent / "examples" / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def visits(owner: object, *names: str, limit: int) -> Iterator[dict[str, int]]:
    """Counts the calls of each ``owner.name`` in the block, and fails on call
    ``limit + 1``: an analysis's work as a number, not a time, and an exponential
    walk fails at once instead of running for its minutes."""
    counts = dict.fromkeys(names, 0)

    def counted(name: str, function: Callable[..., object]) -> Callable[..., object]:
        def call(*args: object, **kwargs: object) -> object:
            counts[name] += 1
            assert counts[name] <= limit, f"{name} called over {limit} times"
            return function(*args, **kwargs)

        return call

    patches = [
        patch.object(owner, name, counted(name, getattr(owner, name))) for name in names
    ]
    for started in patches:
        started.start()
    try:
        yield counts
    finally:
        for started in patches:
            started.stop()


def execute_module(module: object, *tensors: torch.Tensor, kernel_name: str) -> object:
    """Run a module from ``generate_mlir``."""
    return MLIRBackend().execute_mlir(module, *tensors, kernel_name=kernel_name)


def run_generated(
    kernel: helion.Kernel,
    args: list[object],
    *,
    config: helion.Config | None = None,
) -> object:
    """Run ``kernel`` through the explicit ``generate_mlir``/``execute_mlir`` flow."""
    module = generate_mlir(kernel, list(args), config=config)
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    return execute_module(module, *tensors, kernel_name=kernel.fn.__name__)


def run_direct(
    kernel: helion.Kernel, args: list[object], *, pipeline: str | None = None
) -> object:
    """Run ``kernel`` through the direct ``@helion.kernel(backend="mlir")`` call path."""
    with use_pipeline(pipeline):
        return kernel(*args)


def check_kernel(
    kernel: helion.Kernel,
    reference: Callable[..., object],
    args: list[object],
    *,
    paths: tuple[str, ...] = ("direct",),
    config: helion.Config | None = None,
    pipeline: str | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    exact: bool = False,
) -> None:
    """Assert that ``kernel(*args)`` matches ``reference(*args)`` on every requested path.

    ``reference`` runs on clones so in-place kernels and references cannot see each
    other's writes. ``pipeline`` overrides the default lighthouse pipeline.
    ``exact``: bit for bit (:func:`tests.inputs.same_values`), for exact ops and
    inputs with NaN or signed zeros.
    """
    expected = reference(*_clone_args(args))
    for path in paths:
        with use_pipeline(pipeline):
            if path == "direct":
                actual = run_direct(kernel, _clone_args(args))
            elif path == "generated":
                actual = run_generated(kernel, _clone_args(args), config=config)
            else:
                raise ValueError(f"unknown execution path {path!r}")
        if exact:
            same_values(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=path)


def _clone_args(args: list[object]) -> list[object]:
    return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]


def blocked_matmul_cases() -> dict[str, tuple[helion.Kernel, list[object]]]:
    """Production bf16 blocked-matmul kernels from ``helion_mlir_utils`` with inputs."""
    import importlib

    cpu_matmul = importlib.import_module("helion_mlir_utils.matmul")

    def bf16(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.bfloat16)

    vector = [torch.randn(2, 1, 32) for _ in range(3)]
    return {
        "blocked_matmul": (
            cpu_matmul._matmul_blocked_kernel,
            [bf16(2, 2, 32, 32), bf16(2, 2, 32, 32), cpu_matmul.identity_epilogue],
        ),
        "blocked_matmul_bias": (
            cpu_matmul._matmul_blocked_kernel_bias,
            [
                bf16(2, 2, 32, 32),
                bf16(2, 2, 32, 32),
                vector[0],
                cpu_matmul.identity_epilogue,
            ],
        ),
        "blocked_matmul_affine": (
            cpu_matmul._matmul_blocked_kernel_affine,
            [
                bf16(2, 2, 32, 32),
                bf16(2, 2, 32, 32),
                *vector,
                cpu_matmul.identity_epilogue,
            ],
        ),
    }


# --- AMX forced on (IR level: runs on any x86 host) -----------------------------

AMX_FEATURES = ["amx_bf16", "amx_tile", "amx_int8"]


def amx_features(amx: bool = True) -> list[str]:
    """The host's CPU features with AMX's added (or removed, ``amx=False``)."""
    host = [f for f in TargetInfo.host().features if not f.startswith("amx")]
    return host + AMX_FEATURES if amx else host


def lower_opt_with_amx(
    module: ir.Module,
    entry: str,
    until: Callable[[str, ir.Module], bool],
    *,
    verify: bool = False,
    amx: bool = True,
) -> tuple[str, ir.Module] | None:
    """Inline ``module`` and apply the opt pipeline's stages to it with AMX
    forced on, until ``until(str(stage), module after it)`` holds: the module
    before that stage (as text) and after it (``None`` if it never holds).
    ``verify``: verify after every stage. ``amx=False``: with AMX forced off
    instead."""
    from mlir import ir as mlir_ir

    from helion_mlir_backend._compiler.execution import inline_module
    from helion_mlir_backend._compiler.targets.cpu import pipeline_descriptor

    with (
        TargetInfo.override(features=amx_features(amx)),
        module.context,
        mlir_ir.Location.unknown(),
    ):
        inline_module(module)
        driver = BackendDriver(module, entry, result_to_args=False, benchmark=False)
        driver.add_stage(pipeline_descriptor("opt"))
        for stage in driver.stages:
            before = str(module)
            module = stage.apply(module)
            if verify:
                module.operation.verify()
            if until(str(stage), module):
                return before, module
    return None


# Copied from AI-bench's
# backends/helion_mlir/cpu/KernelBench/level1/97_ScaledDotProductAttention.py
# (backend tests do not import AI-bench).

# Growth of a row maximum (in scaled logits) below which the stale maximum is
# kept: exponentials stay below e^8, well within f32 and bf16 range.
_RESCALE_THRESHOLD = 8.0


def _dot_nt(a: torch.Tensor, b: torch.Tensor, vnni: bool) -> torch.Tensor:
    """``a @ b.T`` of ``[M, K]`` and ``[N, K]``; bf16 operands in AMX's VNNI
    layouts, ``[M, K/2, 2]`` and ``[K/2, N, 2]``."""
    if not vnni:
        return torch.einsum("mk,nk->mn", a, b)
    a3 = a.reshape(a.size(0), a.size(1) // 2, 2)
    b3 = b.reshape(b.size(0), b.size(1) // 2, 2).permute(1, 0, 2)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _dot_nn(a: torch.Tensor, b: torch.Tensor, vnni: bool) -> torch.Tensor:
    """``a @ b`` of ``[M, K]`` and ``[K, N]``, VNNI layouts as in ``_dot_nt``."""
    if not vnni:
        return torch.einsum("mk,kn->mn", a, b)
    a3 = a.reshape(a.size(0), a.size(1) // 2, 2)
    b3 = b.reshape(b.size(0) // 2, 2, b.size(1)).permute(0, 2, 1)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    vnni: hl.constexpr,
) -> torch.Tensor:
    """Softmax attention of ``[BH, S, D]`` heads with logits scaled by ``scale``,
    online over key blocks: the running row maxima rescale the sums and the
    output accumulator, only when one of them grows past ``_RESCALE_THRESHOLD``."""
    heads, seq, dim = q.shape
    hl.specialize(seq)
    hl.specialize(dim)
    out = torch.empty_like(q)
    for h in hl.grid(heads):
        for tile_m in hl.tile(seq):
            q_tile = q[h, tile_m, :]
            # Finite: Helion's value range analysis of the growth takes -inf - -inf.
            top = hl.full([tile_m], -1.0e30, dtype=torch.float32)
            total = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, dim], dtype=torch.float32)
            for tile_n in hl.tile(seq):
                scores = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                scores = scores + _dot_nt(q_tile, k[h, tile_n, :], vnni)
                new_top = torch.maximum(top, torch.amax(scores, dim=-1) * scale)
                if torch.amax(new_top - top) > _RESCALE_THRESHOLD:
                    alpha = torch.exp(top - new_top)
                    total = total * alpha
                    acc = acc * alpha[:, None]
                    top = new_top
                p = torch.exp(scores * scale - top[:, None])
                total = total + p.sum(-1)
                acc = acc + _dot_nn(p.to(q.dtype), v[h, tile_n, :], vnni)
            out[h, tile_m, :] = (acc / total[:, None]).to(q.dtype)
    return out


def attention_kernel(block_m: int, block_n: int) -> helion.Kernel:
    """AI-bench 97's online-softmax attention kernel (CPU) at blocks
    ``(block_m, block_n)``: ``kernel(q, k, v, constexpr(scale), constexpr(vnni))``."""
    return helion.kernel(
        _attention,
        backend="mlir",
        config=helion.Config(block_sizes=[block_m, block_n]),
    )
