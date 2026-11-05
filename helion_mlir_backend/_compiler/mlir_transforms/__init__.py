"""Python-defined transform ops of the Helion MLIR backend, one per module,
with snake_case wrappers to build them in schedules (see ``helion_transforms``).

The op modules are imported before the dialect is loaded: a Python-defined
dialect only registers the ops defined before it is loaded."""

from .align_operand_rows import align_operand_rows
from .dialect import HelionTransformDialect
from .drop_contraction_read_unit_dims import drop_contraction_read_unit_dims
from .first_touch import OVERWRITTEN_ATTR_NAME
from .first_touch import first_touch_outputs
from .first_touch import mark_overwritten_args
from .fit_subviews_to_reads import fit_subviews_to_reads
from .fold_full_insert_slices import fold_full_insert_slices
from .fold_pad_unit_inserts import fold_pad_unit_inserts
from .forward_rows import forward_rows
from .hoist_allocas import hoist_allocas
from .hoist_invariant_scalars import hoist_invariant_scalars
from .legalize_for_llvm import legalize_for_llvm
from .loop_bounds import loop_bounds
from .mark_operand_packs import OPERAND_PACK_ATTR_NAME
from .mark_operand_packs import mark_operand_packs
from .match_scalar_contractions import match_scalar_contractions
from .materialize_copies import materialize_copies
from .materialize_operand_pads import materialize_operand_pads
from .move_offsets_to_subview import move_offsets_to_subview
from .normalize_amx_contractions import normalize_amx_contractions
from .outer_transpose_tile import outer_transpose_tile
from .partition_linalg import partition_linalg
from .pin_amx_off_tile_sizes import AMX_OFF_ATTR_NAME
from .pin_amx_off_tile_sizes import pin_amx_off_tile_sizes
from .pin_transposes import pin_transposes
from .prefetch_large_gemm import prefetch_large_gemm
from .schedule_amx_loads import schedule_amx_loads
from .sink_copies_into_branches import sink_copies_into_branches
from .split_bf16_loads import split_bf16_loads
from .split_contraction_operands import split_contraction_operands
from .split_copies import split_copies
from .split_transfers import split_transfers
from .tile_sizes import tile_sizes
from .unmask_contractions import unmask_contractions
from .vectorize_pads import vectorize_pads
from .version_padded_operands import version_padded_operands
from .widen_misread_amx_operands import widen_misread_amx_operands
from .widen_transposes import widen_transposes

__all__ = [
    "AMX_OFF_ATTR_NAME",
    "OPERAND_PACK_ATTR_NAME",
    "OVERWRITTEN_ATTR_NAME",
    "HelionTransformDialect",
    "align_operand_rows",
    "drop_contraction_read_unit_dims",
    "first_touch_outputs",
    "fit_subviews_to_reads",
    "fold_full_insert_slices",
    "fold_pad_unit_inserts",
    "forward_rows",
    "hoist_allocas",
    "hoist_invariant_scalars",
    "legalize_for_llvm",
    "loop_bounds",
    "mark_operand_packs",
    "mark_overwritten_args",
    "match_scalar_contractions",
    "materialize_copies",
    "materialize_operand_pads",
    "move_offsets_to_subview",
    "normalize_amx_contractions",
    "outer_transpose_tile",
    "partition_linalg",
    "pin_amx_off_tile_sizes",
    "pin_transposes",
    "prefetch_large_gemm",
    "schedule_amx_loads",
    "sink_copies_into_branches",
    "split_bf16_loads",
    "split_contraction_operands",
    "split_copies",
    "split_transfers",
    "tile_sizes",
    "unmask_contractions",
    "vectorize_pads",
    "version_padded_operands",
    "widen_misread_amx_operands",
    "widen_transposes",
]
