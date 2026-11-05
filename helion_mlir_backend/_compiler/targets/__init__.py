"""Device targets of the MLIR backend and how a kernel's target is chosen.

The target is, in order: the one named by the caller (``target=`` of the public
API), ``HELION_MLIR_TARGET``, or the one for the device of the kernel's tensors
(``auto``). ``meta`` tensors have no device to follow, so they need a named target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..mlir.support.debug import target_override
from ..mlir.support.errors import UnsupportedTargetError
from .base import MLIRTarget
from .cpu import CpuTarget

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

__all__ = [
    "CpuTarget",
    "MLIRTarget",
    "get_target",
    "resolve_target",
    "target_config_keys",
]

_FACTORIES: dict[str, Callable[[], MLIRTarget] | None] = {
    "cpu": CpuTarget,
}
_TARGET_CLASSES: tuple[type[MLIRTarget], ...] = (CpuTarget,)
_DEVICE_TARGETS = {"cpu": "cpu"}
"""The target of each ``torch.device`` type under ``auto``."""
_INSTANCES: dict[str, MLIRTarget] = {}


def target_config_keys() -> frozenset[str]:
    """The config keys of every target's lowering (:attr:`MLIRTarget.config_keys`)."""
    return frozenset().union(*(cls.config_keys for cls in _TARGET_CLASSES))


def get_target(name: str) -> MLIRTarget:
    """The target named ``name`` (``cpu``)."""
    if name not in _FACTORIES:
        raise ValueError(
            f"unknown MLIR target {name!r}; expected one of {tuple(_FACTORIES)}"
        )
    if (target := _INSTANCES.get(name)) is None:
        if (factory := _FACTORIES[name]) is None:
            raise UnsupportedTargetError(
                f"MLIR target '{name}'",
                reason=f"the '{name}' target is not implemented yet",
                alternatives=["target='cpu' with CPU tensors"],
            )
        target = _INSTANCES[name] = factory()
    return target


def resolve_target(
    device: torch.device | None = None, override: str | None = None
) -> MLIRTarget:
    """The target compiling a kernel whose tensors are on ``device`` (``None``:
    the CPU). ``override`` names the target (``auto``: follow the device); if it is
    ``None``, ``HELION_MLIR_TARGET`` may (read on every call)."""
    name = override if override is not None else target_override()
    if name is not None and name != "auto":
        return get_target(name)
    device_type = "cpu" if device is None else device.type
    if device_type == "meta":
        raise UnsupportedTargetError(
            "MLIR compilation for meta tensors",
            reason="meta tensors have no device to pick the target from",
            alternatives=["target='cpu'", "HELION_MLIR_TARGET"],
        )
    if device_type not in _DEVICE_TARGETS:
        raise UnsupportedTargetError(
            f"MLIR compilation for device '{device_type}'",
            reason=(
                "the MLIR backend compiles for "
                f"{', '.join(name.upper() for name in _DEVICE_TARGETS)} tensors"
            ),
        )
    return get_target(_DEVICE_TARGETS[device_type])
