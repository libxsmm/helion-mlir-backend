"""The CPU target: lighthouse's x86 pipelines, OpenMP, the host CPU's tensors."""

from __future__ import annotations

import functools
from pathlib import Path
import platform
from typing import TYPE_CHECKING

from lighthouse.execution.runner import Runner
from lighthouse.ingress.torch.compile import TorchMemoryManager
from lighthouse.pipeline.descriptor import Descriptor
from lighthouse.pipeline.driver import BackendDriver

from ..mlir.support.debug import PIPELINES
from ..mlir.support.debug import default_pipeline
from ..mlir_transforms.prints import lower_prints_to_libc
from ..mlir_transforms.prints import warn_print_is_experimental
from .base import MLIRTarget

if TYPE_CHECKING:
    from collections.abc import Collection

    from helion._compiler.device_ir import DeviceIR
    from helion.autotuner.config_spec import ConfigSpec
    import mlir.ir as ir
    import torch

_PIPELINE_FILES = {"opt": "./pipeline.yaml", "scalar": "./scalar.yaml"}
_PIPELINE_DIR = str(Path(__file__).parent.parent)
"""Where pipeline files and the local schedules they name live (``_compiler/``)."""
_OPT_MIN_TILE = 32
"""Inner tiles narrower than 32 waste vector lanes."""


def pipeline_descriptor(pipeline: str | None = None) -> Descriptor:
    """The lighthouse pipeline by name (default: :func:`default_pipeline`)."""
    if pipeline is None:
        pipeline = default_pipeline()
    if pipeline not in _PIPELINE_FILES:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected one of {PIPELINES}")
    return Descriptor(_PIPELINE_FILES[pipeline], base_path=_PIPELINE_DIR)


def raise_block_minimums(
    config_spec: ConfigSpec, outer_block_ids: Collection[int] = ()
) -> None:
    """Search (and default) tiles of at least 32 where the dimension allows, but
    any size for ``outer_block_ids``: small outer tiles add parallel work and keep
    the compiled inner work small."""
    for spec in config_spec.block_sizes:
        if not set(spec.block_ids) & set(outer_block_ids):
            spec.update_min(min(_OPT_MIN_TILE, spec.max_size))


@functools.cache
def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


class CpuTarget(MLIRTarget):
    """Kernels JIT-compiled for the host CPU and parallelized with OpenMP.

    Pipelines: ``opt`` (``pipeline.yaml``: tiling, packing, vectorization, OpenMP)
    and ``scalar`` (``scalar.yaml``: plain loops); ``HELION_MLIR_PIPELINE`` picks
    the default.
    """

    name = "cpu"
    pipelines = PIPELINES

    def default_pipeline(self) -> str:
        return default_pipeline()

    def lowering(
        self,
        module: ir.Module,
        entry: str,
        pipeline: str,
        stop_at: str | None = None,
    ) -> BackendDriver:
        self.check_stop_at(stop_at)
        if lower_prints_to_libc(module):
            warn_print_is_experimental(entry)
        driver = BackendDriver(module, entry, result_to_args=False, benchmark=False)
        driver.add_stage(pipeline_descriptor(pipeline))
        return driver

    def runner(self, module: ir.Module) -> Runner:
        return Runner(module, mem_manager_cls=TorchMemoryManager, shared_libs=[])

    def check_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu":
            raise NotImplementedError(
                f"'{name}' is on {tensor.device}; only CPU is supported"
            )

    def passes_in_place(self, tensor: torch.Tensor) -> bool:
        return tensor.is_cpu

    # The native paths take only plain dense CPU tensors: a subset of
    # passes_in_place (anything else takes the Python paths).
    native_in_place = True

    def hardware_name(self) -> str:
        return cpu_name()

    def adjust_config_spec(self, config_spec: ConfigSpec, device_ir: DeviceIR) -> None:
        """Under the optimizing pipeline, tiles of at least 32 except for the
        leading dim of each root loop."""
        if self.default_pipeline() == "opt":
            outer = [ids[0] for ids in device_ir.grid_block_ids if ids]
            raise_block_minimums(config_spec, outer)
