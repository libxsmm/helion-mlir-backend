"""Error types for the MLIR backend.

Every error is a :class:`helion.exc.BaseError`: raised while a node's
``meta["location"]`` is active (see ``lowering/registry.py``), Helion appends the
kernel source line to the message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.exc

if TYPE_CHECKING:
    import torch.fx


class MLIRBackendError(helion.exc.BaseError):
    """Base exception for all MLIR backend errors."""

    message = "{0}"


class UnsupportedOperationError(MLIRBackendError):
    """Raised when an unsupported operation is encountered."""

    def __init__(
        self,
        op_name: str,
        reason: str | None = None,
        alternatives: list[str] | None = None,
    ) -> None:
        msg = f"Unsupported operation: {op_name}"
        if reason:
            msg += f"\nReason: {reason}"
        if alternatives:
            msg += f"\nAlternatives: {', '.join(alternatives)}"
        super().__init__(msg)
        self.op_name = op_name
        self.reason = reason
        self.alternatives = alternatives

    def __reduce__(self) -> tuple:
        # Rebuilt from its parts, not its message: it crosses process boundaries
        # (lowering in a forked child).
        args = (self.op_name, self.reason, self.alternatives)
        return type(self), args, self.__dict__


class UnsupportedTargetError(UnsupportedOperationError, NotImplementedError):
    """Raised for a device (or target name) no MLIR target compiles for.

    Also a :class:`NotImplementedError`, what device checks raised before targets.
    """


class ShapeError(MLIRBackendError):
    """Raised when shape inference fails."""

    def __init__(
        self,
        shape: object,
        reason: str | None = None,
        constraint: str | None = None,
    ) -> None:
        msg = f"Invalid shape: {shape}"
        if reason:
            msg += f"\nReason: {reason}"
        if constraint:
            msg += f"\nConstraint: {constraint}"
        super().__init__(msg)
        self.shape = shape
        self.reason = reason
        self.constraint = constraint


class DynamicShapeError(ShapeError):
    """Raised when a size is not known at compile time and cannot be computed at run time."""

    def __init__(self, shape: object, symbol_name: str | None = None) -> None:
        reason = f"cannot resolve {symbol_name or 'a symbolic size'}"
        constraint = "Use static_shapes=True, or hl.specialize() the size"
        super().__init__(shape, reason, constraint)
        self.symbol_name = symbol_name


class ValueNotFoundError(MLIRBackendError):
    """Raised when a value lookup fails."""

    def __init__(self, node: object, context: str | None = None) -> None:
        msg = f"Value not found for node: {node}"
        if context:
            msg += f"\nContext: {context}"
        super().__init__(msg)
        self.node = node
        self.context = context


class NodeLoweringError(MLIRBackendError):
    """Raised when lowering a single FX node fails."""

    def __init__(
        self,
        node: torch.fx.Node,
        reason: str | None = None,
        recovery_hint: str | None = None,
    ) -> None:
        msg = f"Failed to lower FX node: {node.op}[{node.name}]"
        if hasattr(node, "target"):
            msg += f"\nTarget: {node.target}"
        if reason:
            msg += f"\nReason: {reason}"
        if recovery_hint:
            msg += f"\nHint: {recovery_hint}"
        super().__init__(msg)
        self.node = node
        self.reason = reason
        self.recovery_hint = recovery_hint


class CompileTimeoutError(MLIRBackendError):
    """Raised when lowering a module takes longer than the compile timeout."""

    def __init__(self, pipeline: str, timeout: float) -> None:
        super().__init__(
            f"Lowering with the lighthouse '{pipeline}' pipeline took over "
            f"{timeout:g} s (HELION_MLIR_COMPILE_TIMEOUT); try other block sizes"
        )
        self.pipeline = pipeline
        self.timeout = timeout


class LoweringStoppedError(MLIRBackendError):
    """Raised when a kernel to run was only lowered to an earlier stage: there is
    nothing to JIT-compile."""

    def __init__(self, target: str, stop_at: str | None) -> None:
        super().__init__(
            f"Lowering for the MLIR target '{target}' stops at '{stop_at}' by "
            "default, so the kernel cannot run; "
            "helion_mlir_backend.lower_mlir() returns the IR"
        )
        self.target = target
        self.stop_at = stop_at
