"""Helion operations this backend rejects, each with its reason."""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language as hl

from ..support import UnsupportedOperationError
from .registry import lowers

if TYPE_CHECKING:
    import torch

    from ..build_context import BuildContext

_ATOMICS = "atomics need memory semantics; value-semantic tensor IR has none"
_REASONS = {
    hl.atomic_add: _ATOMICS,
    hl.atomic_and: _ATOMICS,
    hl.atomic_cas: _ATOMICS,
    hl.atomic_max: _ATOMICS,
    hl.atomic_min: _ATOMICS,
    hl.atomic_or: _ATOMICS,
    hl.atomic_xchg: _ATOMICS,
    hl.atomic_xor: _ATOMICS,
    hl.rand: "no device random number generator",
    hl.rand4x: "no device random number generator",
    hl.randint: "no device random number generator",
    hl.inline_asm_elementwise: "inline assembly is target specific",
    hl.inline_triton: "inline Triton code needs the Triton backend",
}


@lowers(*_REASONS)
def reject(ctx: BuildContext, node: torch.fx.Node) -> None:
    raise UnsupportedOperationError(
        f"hl.{node.target.__name__}", reason=_REASONS[node.target]
    )
