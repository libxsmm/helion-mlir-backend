"""MLIR backend class for Helion.

Registers as a Helion compiler backend named "mlir".  The backend:

- Reuses Helion's backend-neutral compilation pipeline (type propagation,
    device IR construction, etc.).
- Replaces the final code-generation step with an MLIR module builder that
  produces Linalg-on-Tensors IR instead of Triton Python source code.

The generated MLIR is intentionally high-level: Linalg-on-Tensors phase
functions plus a thin memref entry function that fixes the calling convention
(no tiling or lowering to LLVM), so a downstream MLIR compiler can apply its
own tiling, vectorization, and memory-placement passes. What depends on the
device (lowering, runtime, search space) is the kernel's target (``targets/``).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from helion import exc
from helion._compiler.backend import Backend

from ..targets import resolve_target
from ..targets import target_config_keys
from .support.debug import PIPELINE_CONFIG_KEY
from .support.errors import MLIRBackendError

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion.autotuner.config_fragment import ConfigSpecFragment
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.config import Config
    from helion.runtime.kernel import BoundKernel

_CONFIG_KEYS = frozenset({"block_sizes", PIPELINE_CONFIG_KEY})


class MLIRBackend(Backend):
    """Helion backend that emits MLIR Linalg-on-Tensors IR.

    Compilation (parsing, type propagation, device IR construction) uses
    Helion's backend-neutral pipeline. Only the final codegen step is replaced.
    Autotuning is Helion's (one given config is used as is), timed on the host.
    """

    @property
    def name(self) -> str:
        return "mlir"

    @property
    def experimental(self) -> bool:
        return True

    def supports_config_key(self, key: str) -> bool:
        """Only block sizes shape the generated IR; ``mlir_pipeline`` picks the
        target's lighthouse pipeline (default on the CPU: ``HELION_MLIR_PIPELINE``,
        else ``opt``). Each target's lowering keys are
        accepted on every device: the other targets ignore them."""
        return key in _CONFIG_KEYS or key in target_config_keys()

    def supports_precompile(self) -> bool:
        return False

    @property
    def pad_factory_tensors_to_power_of_2(self) -> bool:
        """No: MLIR needs no power-of-two shapes, and the
        padding changes semantics. With it, a dim used in an on-device
        factory (``hl.zeros([tile, dim])``) gets a reduction block of
        next_pow2(dim), and every ``.size()`` of a ``:`` slice of that dim
        anywhere in the loop returns the padded size (64 for 40). Without
        it, Helion keeps ``:`` slices and factory dims at their real sizes,
        as for Pallas."""
        return False

    def autotune(
        self,
        bound_kernel: BoundKernel,
        args: Sequence[object],
        *,
        force: bool = True,
        **kwargs: object,
    ) -> Config:
        """Helion's autotuning; its default local cache becomes the MLIR one,
        keyed by the target's hardware. A search runs in the space the kernel's
        target narrows (on the CPU under the optimizing pipeline: tiles of at
        least 32 except for the leading dim of each root loop)."""
        from .autotune import AUTOTUNE_CACHE

        settings = bound_kernel.settings
        if settings.autotune_cache == "LocalAutotuneCache":
            settings.autotune_cache = AUTOTUNE_CACHE
        if force or settings.force_autotune or not bound_kernel.kernel.configs:
            target = resolve_target(bound_kernel.env.device)
            target.adjust_config_spec(
                bound_kernel.config_spec, bound_kernel.host_function.device_ir
            )
        return super().autotune(bound_kernel, args, force=force, **kwargs)

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]]:
        from helion.autotuner.benchmarking import do_bench_generic

        return do_bench_generic

    def get_interleaved_bench(self) -> Callable[..., list[float]]:
        from helion.autotuner.benchmarking import interleaved_bench_generic

        return interleaved_bench_generic

    def classify_autotune_exception(self, err: BaseException) -> str | None:
        """A config the backend or lighthouse cannot compile is skipped."""
        if isinstance(err, MLIRBackendError) or any(
            note.startswith("Helion MLIR backend:")
            for note in getattr(err, "__notes__", ())
        ):
            return "warn"
        return None

    def config_value_priors(
        self, config_spec: ConfigSpec
    ) -> dict[str, Callable[[ConfigSpecFragment, int], object]]:
        """Bias random block sizes toward divisors of the dimension (no pad/mask)."""
        hints = [spec.size_hint for spec in config_spec.block_sizes]

        def divisors_first(fragment: ConfigSpecFragment, position: int) -> object:
            low, high = getattr(fragment, "low", None), getattr(fragment, "high", None)
            if (
                not isinstance(low, int)
                or not isinstance(high, int)
                or position >= len(hints)
            ):
                return None
            sizes = [1 << bit for bit in range(low.bit_length() - 1, high.bit_length())]
            weights = [4.0 if hints[position] % size == 0 else 1.0 for size in sizes]
            return random.choices(sizes, weights=weights)[0]

        return {"block_sizes": divisors_first}

    def dtype_str(self, dtype: object) -> str:
        import torch

        dtype_names = {
            torch.bool: "torch.bool",
            torch.float16: "torch.float16",
            torch.bfloat16: "torch.bfloat16",
            torch.float32: "torch.float32",
            torch.float64: "torch.float64",
            torch.int8: "torch.int8",
            torch.int16: "torch.int16",
            torch.int32: "torch.int32",
            torch.int64: "torch.int64",
            torch.uint8: "torch.uint8",
        }
        return dtype_names.get(dtype, str(dtype))

    def acc_type(self, dtype: object) -> str:
        return self.dtype_str(dtype)

    @property
    def function_decorator(self) -> str:
        raise exc.BackendUnsupported(self.name, "Python function decorators")

    @property
    def constexpr_type(self) -> str:
        raise exc.BackendUnsupported(self.name, "Python constexpr annotations")

    @property
    def default_launcher_name(self) -> str:
        raise exc.BackendUnsupported(self.name, "Python launchers")

    @property
    def library_imports(self) -> dict[str, str]:
        raise exc.BackendUnsupported(self.name, "Python library imports")

    # ------------------------------------------------------------------
    # MLIR generation entry point
    # ------------------------------------------------------------------

    def generate_mlir(
        self,
        host_function: HostFunction,
        config: object,
        env: CompileEnvironment,
    ) -> object:
        """Build and return an ``mlir.ir.Module`` for the compiled kernel.

        Parameters
        ----------
        host_function:
            The :class:`~helion._compiler.host_function.HostFunction` produced
            by :func:`~helion._compiler.kernel_compiler.KernelCompiler.compile`.
        config:
            The :class:`~helion.runtime.config.Config` specifying block sizes
            and other tuning parameters.
        env:
            The :class:`~helion._compiler.compile_environment.CompileEnvironment`
            that was active during compilation.

        Returns
        -------
        mlir.ir.Module
            Parsed and verified MLIR module containing a ``func.func`` with
            Linalg-on-Tensors IR equivalent to the Helion kernel.
        """
        from .codegen import MLIRModuleBuilder

        builder = MLIRModuleBuilder(host_function, config, env)
        return builder.build()

    # ------------------------------------------------------------------
    # MLIR execution entry point (experimental)
    # ------------------------------------------------------------------

    def execute_mlir(
        self,
        mlir_module: object,
        *input_tensors: object,
        kernel_name: str = "kernel",
        pipeline: str | None = None,
    ) -> object:
        """Execute a module from :meth:`generate_mlir` via lighthouse (consumes it).

        ``input_tensors`` are the kernel's tensor parameters in declaration order.
        No host code runs: tensors the kernel creates on the host and writes are
        zero-initialized, and kernels that read host-computed tensors or take
        runtime scalars are rejected. Use ``compile_mlir`` or a direct kernel call
        for full Helion semantics. The target follows the tensors' device (or
        ``HELION_MLIR_TARGET``); ``pipeline`` defaults to the target's.

        Inputs hold their values at the call: no host code is known here, so an
        input sharing elements with a written tensor is a copy of it.

        Returns every tensor the kernel writes, in entry argument order: one
        tensor, or a list when there are several.
        """
        import torch

        from ..execution import compile_entry
        from .driver import call_entry
        from .support import UnsupportedOperationError

        if not input_tensors or not all(
            isinstance(tensor, torch.Tensor) for tensor in input_tensors
        ):
            raise TypeError("input_tensors must be non-empty torch.Tensor instances")

        device = input_tensors[0].device
        if any(tensor.device != device for tensor in input_tensors[1:]):
            raise ValueError("all input tensors must be on the same device")
        target = resolve_target(device)
        if pipeline is not None and pipeline not in target.pipelines:
            raise ValueError(
                f"unknown pipeline {pipeline!r}; expected one of {target.pipelines}"
            )

        entry = compile_entry(
            mlir_module, kernel_name, target=target, pipeline=pipeline
        )
        values: list[object] = []
        alternatives = [
            "helion_mlir_backend.compile_mlir(kernel, args)",
            "call the kernel with @helion.kernel(backend='mlir')",
        ]
        for arg in entry.args:
            if arg.tensor_param is not None:
                values.append(input_tensors[arg.tensor_param])
            elif arg.role == "inout" and -1 in arg.shape:
                raise UnsupportedOperationError(
                    f"execute_mlir cannot create '{arg.name}'",
                    reason="its shape is only known when the host code runs",
                    alternatives=alternatives,
                )
            elif arg.role == "inout":
                values.append(torch.zeros(arg.shape, dtype=arg.dtype, device=device))
            else:
                raise UnsupportedOperationError(
                    f"execute_mlir cannot supply host value '{arg.name}'",
                    reason="execute_mlir runs no host code",
                    alternatives=alternatives,
                )
        outputs = call_entry(entry, values)
        return outputs[0] if len(outputs) == 1 else outputs
