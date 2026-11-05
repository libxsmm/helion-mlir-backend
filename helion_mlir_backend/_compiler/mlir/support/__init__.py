"""Support utilities for MLIR code generation.

This module contains utility functions and classes used throughout the MLIR
compilation pipeline:
- block_ids: Block dimension identification and mapping
- type_utils: Type conversion between torch and MLIR
- errors: Error types and diagnostics
- index_meta: Index expressions to block ids
"""

from __future__ import annotations

from .block_ids import SCALAR_SYMBOL_KINDS
from .block_ids import block_id_from_key
from .block_ids import symbol_origin_info
from .errors import CompileTimeoutError
from .errors import DynamicShapeError
from .errors import LoweringStoppedError
from .errors import MLIRBackendError
from .errors import NodeLoweringError
from .errors import ShapeError
from .errors import UnsupportedOperationError
from .errors import UnsupportedTargetError
from .errors import ValueNotFoundError
from .index_meta import IndexDescriptor
from .index_meta import resolve_index_descriptor
from .type_utils import mlir_dtype_to_torch
from .type_utils import static_dim
from .type_utils import torch_dtype_to_mlir

__all__ = [
    "SCALAR_SYMBOL_KINDS",
    "CompileTimeoutError",
    "DynamicShapeError",
    "IndexDescriptor",
    "LoweringStoppedError",
    "MLIRBackendError",
    "NodeLoweringError",
    "ShapeError",
    "UnsupportedOperationError",
    "UnsupportedTargetError",
    "ValueNotFoundError",
    "block_id_from_key",
    "mlir_dtype_to_torch",
    "resolve_index_descriptor",
    "static_dim",
    "symbol_origin_info",
    "torch_dtype_to_mlir",
]
