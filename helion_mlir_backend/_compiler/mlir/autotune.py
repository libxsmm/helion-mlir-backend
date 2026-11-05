"""Autotuning: Helion's local autotune cache, keyed by the kernel's target.

Helion's :class:`LocalAutotuneCache` derives its key from a GPU (or TPU/MPS)
device and has no CPU entry, so the MLIR backend uses this subclass instead: the
kernel's target names the hardware and its default pipeline the runtime.
"""

from __future__ import annotations

import hashlib
import inspect
import textwrap

from helion.autotuner.base_cache import LooseAutotuneCacheKey
from helion.autotuner.local_cache import LocalAutotuneCache

from ..targets import resolve_target

AUTOTUNE_CACHE = "MLIRAutotuneCache"
"""Name under which the cache is registered in ``helion.autotuner.cache_classes``."""
CPU_AUTOTUNE_CACHE = "MLIRCpuAutotuneCache"
"""The name the cache had when it served only the CPU (still registered)."""


class MLIRAutotuneCache(LocalAutotuneCache):
    """Best configs on disk, keyed by the target's hardware and default pipeline."""

    def _generate_key(self) -> LooseAutotuneCacheKey:
        bound = self.kernel
        in_memory = bound.kernel._create_bound_kernel_cache_key(
            bound, tuple(self.args), bound.kernel._base_specialization_key(self.args)
        )
        source = textwrap.dedent(inspect.getsource(bound.kernel.fn))
        target = resolve_target(bound.env.device)
        pipeline = target.default_pipeline()
        return LooseAutotuneCacheKey(
            specialization_key=in_memory.specialization_key,
            extra_results=in_memory.extra_results,
            kernel_source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            hardware=target.hardware_name(),
            runtime_name=f"lighthouse-{target.name}-{pipeline}",
            backend=bound.env.backend.name,
            config_spec_hash=bound.config_spec.structural_fingerprint_hash(
                advanced_controls_files=self.autotuner.settings.autotune_search_acf
                or None
            ),
            extra_cache_key=bound.extra_cache_key(),
            best_of_k=self.autotuner.settings.autotune_best_of_k,
        )
