"""MLIR lowering of device IR nodes.

Importing this package registers every handler with :mod:`.registry`.
"""

from __future__ import annotations

from . import combine_ops as combine_ops
from . import contraction_ops as contraction_ops
from . import control_flow as control_flow
from . import elementwise_ops as elementwise_ops
from . import host_tensor_ops as host_tensor_ops
from . import inline_mlir_ops as inline_mlir_ops
from . import load_slice_ops as load_slice_ops
from . import loops as loops
from . import memory_ops as memory_ops
from . import print_ops as print_ops
from . import scalar_ops as scalar_ops
from . import tensor_creation_ops as tensor_creation_ops
from . import tile_index_ops as tile_index_ops
from . import transpose_ops as transpose_ops
from . import unsupported_ops as unsupported_ops
from . import view_ops as view_ops
from .loops import build_phase_body
from .registry import lower_node

__all__ = ["build_phase_body", "lower_node"]
