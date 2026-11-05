"""Kernels of one body that differ by tile config."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import helion

if TYPE_CHECKING:
    from collections.abc import Callable


def per_config(
    **kernel_args: object,
) -> Callable[[Callable[..., object]], Callable[[tuple[int, ...]], helion.Kernel]]:
    """A kernel per tile config: ``@per_config(backend="mlir")`` on a kernel body
    makes its name take a hashable ``block_sizes`` and return that config's
    kernel. The raw body stays reachable as ``.body``."""

    def wrap(body: Callable[..., object]) -> Callable[[tuple[int, ...]], helion.Kernel]:
        @functools.cache
        def for_config(block_sizes: tuple[int, ...]) -> helion.Kernel:
            return helion.kernel(
                body, config=helion.Config(block_sizes=list(block_sizes)), **kernel_args
            )

        for_config.body = body
        return for_config

    return wrap
