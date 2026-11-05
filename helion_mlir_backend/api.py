"""Public MLIR API for the external Helion MLIR backend package."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    import mlir.ir as ir


def generate_mlir(
    kernel: object,
    args: list[object],
    *,
    config: object | None = None,
    target: str | None = None,
) -> ir.Module:
    """Lower a Helion kernel to an MLIR Linalg-on-Tensors module.

    The module holds one private tensor function per phase and a public
    memref-ABI entry function named after the kernel. ``target`` (``"cpu"``;
    default: ``HELION_MLIR_TARGET``, else the device of ``args``)
    names the device target the module is generated for.
    """
    from helion._compiler.backend_registry import get_backend_class

    from helion_mlir_backend._compiler.targets import resolve_target

    hf, config, env = _compile(kernel, args, config)
    # Codegen is target-neutral; the target is only checked to exist.
    resolve_target(env.device, target)
    with env:
        return get_backend_class("mlir")().generate_mlir(hf, config, env)


def compile_mlir(
    kernel: object,
    args: list[object],
    *,
    config: object | None = None,
    target: str | None = None,
    pipeline: str | None = None,
) -> Callable[..., object]:
    """Compile a Helion kernel for ``args``' shapes and dtypes into a callable.

    The callable has the kernel's signature and Helion semantics: it runs the
    kernel's host code on each call and returns the kernel's return value.
    ``target`` (``"cpu"``; default: ``HELION_MLIR_TARGET``, else the
    device of ``args``) names the device target. ``pipeline`` is one of the
    target's pipelines; on the CPU ``"opt"`` or ``"scalar"`` (default:
    ``HELION_MLIR_PIPELINE``, else ``"opt"``).
    """
    from helion_mlir_backend._compiler.mlir.driver import compile_kernel

    hf, config, env = _compile(kernel, args, config)
    return compile_kernel(hf, config, env, target=target, pipeline=pipeline)


def lower_mlir(
    kernel: object,
    args: list[object],
    *,
    config: object | None = None,
    target: str | None = None,
    pipeline: str | None = None,
    stop_at: str | None = None,
) -> ir.Module:
    """Generate, inline and lower a Helion kernel for ``args``' shapes and dtypes
    without JIT-compiling it: the IR a call would run, or an earlier stage of it.

    ``target`` and ``pipeline`` as for :func:`compile_mlir` (``pipeline``
    defaults to the config's ``mlir_pipeline``, then the target's). ``stop_at`` is
    a stage of the target's lowering; default: the full lowering (the LLVM dialect
    on the CPU). Unsupported kernels raise ``UnsupportedOperationError``.
    """
    from helion._compiler.backend_registry import get_backend_class

    from helion_mlir_backend._compiler.execution import lower_entry
    from helion_mlir_backend._compiler.mlir.support.debug import PIPELINE_CONFIG_KEY
    from helion_mlir_backend._compiler.targets import resolve_target

    hf, config, env = _compile(kernel, args, config)
    mlir_target = resolve_target(env.device, target)
    if pipeline is None:
        pipeline = config.get(PIPELINE_CONFIG_KEY)
    with env:
        module = get_backend_class("mlir")().generate_mlir(hf, config, env)
    mlir_target.configure(module, config)
    return lower_entry(
        module, hf.name, target=mlir_target, pipeline=pipeline, stop_at=stop_at
    )


def _compile(
    kernel: object, args: list[object], config: object | None
) -> tuple[HostFunction, object, CompileEnvironment]:
    try:
        import mlir.ir  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mlir-python-bindings is required for MLIR code generation. "
            "Install it with: pip install mlir-python-bindings"
        ) from exc

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.kernel_compiler import KernelCompiler
    from helion._compiler.variable_origin import ArgumentOrigin
    from helion.runtime.settings import Settings
    import torch

    from helion_mlir_backend._compiler.mlir.driver import warn_misspelt_config_keys

    fn = getattr(kernel, "fn", None)
    if fn is None:
        raise ValueError(
            f"Expected a @helion.kernel-decorated function; got {type(kernel).__name__}"
        )

    settings: Settings = getattr(kernel, "settings", Settings())
    settings = dataclasses.replace(settings, backend="mlir")

    if config is None:
        # `@helion.kernel(config=...)` is normalized into `Kernel.configs`.
        kernel_configs = getattr(kernel, "configs", None)
        if kernel_configs:
            config = kernel_configs[0]

    device = next(
        (arg.device for arg in args if isinstance(arg, torch.Tensor)),
        torch.device("cpu"),
    )
    env = CompileEnvironment(device, settings)

    with env:
        fake_args = [
            env.to_fake(arg, ArgumentOrigin(name))
            for name, arg in zip(kernel.signature.parameters, args, strict=False)
        ]
        host_function = KernelCompiler(env).compile(fn, fake_args, {})
        if config is None:
            config = env.config_spec.default_config()
    warn_misspelt_config_keys(host_function.name, config, env)
    return host_function, config, env
