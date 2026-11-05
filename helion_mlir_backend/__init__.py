"""External MLIR backend package for Helion.

This package hosts the MLIR backend implementation outside the Helion tree and
registers an external MLIR backend with vanilla Helion when imported.
"""

from __future__ import annotations

from ._compiler import output_pool as output_pool
from ._compiler.call_trace import CallExplanation as CallExplanation
from ._compiler.call_trace import explain as explain
from ._compiler.mlir.driver import prepared_call as prepared_call
from ._compiler.native import set_native_call as set_native_call
from ._compiler.output_pool import set_output_pool as set_output_pool
from .api import compile_mlir as compile_mlir
from .api import generate_mlir as generate_mlir
from .api import lower_mlir as lower_mlir
from .inject import install as install
from .language import InlineMLIRHint as InlineMLIRHint
from .language import inline_mlir as inline_mlir

__all__ = [
    "CallExplanation",
    "InlineMLIRHint",
    "compile_mlir",
    "explain",
    "generate_mlir",
    "inline_mlir",
    "install",
    "lower_mlir",
    "output_pool",
    "prepared_call",
    "set_native_call",
    "set_output_pool",
]

# Import-time side effect: make backend name "mlir" available in Helion.
install()
