"""Shared mutable state for MLIR code generation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
from mlir.dialects import arith as arith_d
from mlir.dialects import tensor as tensor_d
import mlir.ir as ir

from .analysis.index_facts import IndexFacts
from .analysis.index_facts import index_facts
from .lowering.tensor_state import TensorState
from .sizes import Sizes
from .support import DynamicShapeError
from .support import block_id_from_key
from .support.errors import UnsupportedOperationError

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config
    import torch.fx

    from .analysis.contractions import ContractionPlan
    from .analysis.geometry import KernelGeometry
    from .analysis.signature import KernelSignature
    from .analysis.tensor_effects import TensorEffects
    from .aten_bridge import AtenHelperTable


@dataclass
class BuildContext:
    """State shared by the MLIR builder and its lowering helpers."""

    host_function: HostFunction
    config: Config | object
    env: CompileEnvironment

    node_to_value: dict[torch.fx.Node, ir.Value] = field(default_factory=dict)
    param_to_value: dict[str, ir.Value] = field(default_factory=dict)

    geometry: KernelGeometry | None = None

    # Written directly once per outer grid block id (loops._bind_grid_iv) then
    # save/restored per nested scf.for level via enter_for_loop(); read
    # everywhere a block id's current induction variable is needed. The value is
    # the tile's absolute offset (``begin + trip * step``).
    block_id_to_iv: dict[int, ir.Value] = field(default_factory=dict)
    # The active loop's (begin, end) per block id, as static ints or index values.
    block_id_to_bounds: dict[int, tuple[int | ir.Value, int | ir.Value]] = field(
        default_factory=dict
    )
    # Elements of the current tile inside its loop, for loops whose last tile is
    # partial (Helion's tile mask); absent when every tile is full.
    block_id_to_valid: dict[int, int | ir.Value] = field(default_factory=dict)
    # Normalized (unit-step) trip index of an outer forall dimension, when known.
    block_id_to_trip_iv: dict[int, ir.Value] = field(default_factory=dict)
    tensors: TensorState = field(default_factory=TensorState)
    effects: TensorEffects | None = None
    signature: KernelSignature | None = None
    # Runtime scalar values of the current function, by ``_get_symnode`` key.
    scalars: dict[str, ir.Value] = field(default_factory=dict)
    # Nesting depth of the ``if`` branches and ``while`` loops being lowered:
    # accesses there run only where their condition holds.
    guard_depth: int = 0

    mlir_module: ir.Module | None = None
    mlir_context: ir.Context | None = None
    aten_helpers: AtenHelperTable | None = None
    contractions: ContractionPlan | None = None
    lower_node_callback: Callable[[torch.fx.Node], ir.Value | None] | None = None
    sizes: Sizes = field(init=False)

    def __post_init__(self) -> None:
        self.sizes = Sizes(self)

    def get_value(self, node_or_value: object) -> ir.Value | None:
        """The MLIR value of an FX node, or an ``index`` constant for an int."""
        import torch.fx

        if isinstance(node_or_value, torch.fx.Node):
            return self.node_to_value.get(node_or_value)
        if isinstance(node_or_value, int):
            return self.index_const(node_or_value)
        return None

    def set_value(self, node: torch.fx.Node, value: ir.Value) -> None:
        """Associate an FX node with its generated MLIR value."""
        self.node_to_value[node] = value

    def index_const(self, value: int) -> ir.Value:
        """Create an MLIR index constant."""

        index_type = ir.IndexType.get()
        return arith_d.ConstantOp(
            index_type,
            ir.IntegerAttr.get(index_type, value),
        ).result

    def as_index(self, value: int | ir.Value) -> ir.Value:
        """An index value for a static int or an existing scalar value."""
        if isinstance(value, int):
            return self.index_const(value)
        return self.cast_to_index(value)

    def cast_to_index(self, value: ir.Value) -> ir.Value:
        """Cast an integer or rank-zero tensor value to MLIR index type."""

        if isinstance(value.type, ir.IndexType):
            return value
        if isinstance(value.type, ir.RankedTensorType) and value.type.rank == 0:
            element_type = value.type.element_type
            if isinstance(element_type, (ir.IntegerType, ir.IndexType)):
                value = tensor_d.ExtractOp(value, []).result
                if isinstance(value.type, ir.IndexType):
                    return value
        return arith_d.IndexCastOp(ir.IndexType.get(), value).result

    def shape_from_nodes(
        self, shape_nodes: list, operation_name: str = "op"
    ) -> list[int | ir.Value]:
        """Dims of a shape list: ints, tile sizes, constants or runtime sizes."""
        import torch.fx

        shape: list[int | ir.Value] = []
        for shape_node in shape_nodes:
            if isinstance(shape_node, int):
                shape.append(shape_node)
                continue
            if isinstance(shape_node, torch.fx.Node):
                if shape_node.target is tracing_ops._get_symnode:
                    block_id = block_id_from_key(shape_node.args[0])
                    if block_id is not None and block_id in self.geometry.blocks:
                        shape.append(self.sizes.tile(block_id))
                        continue
                value = self.get_value(shape_node)
                owner = value.owner if value is not None else None
                if isinstance(owner, ir.OpView) and owner.name == "arith.constant":
                    shape.append(ir.IntegerAttr(owner.attributes["value"]).value)
                    continue
                meta = shape_node.meta.get("val")
                if isinstance(meta, torch.SymInt):
                    shape.append(self.sizes.value(meta))
                    continue
            raise DynamicShapeError(
                shape_nodes, symbol_name=f"{shape_node} in {operation_name}"
            )
        return shape

    def symbol_info(self, value: object) -> tuple[int, str] | None:
        """Resolve a SymInt to ``(block_id, kind)`` using Helion symbol origins."""
        from .support import symbol_origin_info

        return symbol_origin_info(self.host_function, value)

    def node_symbol_info(self, node: object) -> tuple[int, str] | None:
        """Resolve an FX node's SymInt metadata to ``(block_id, kind)``."""
        import torch
        import torch.fx

        if not isinstance(node, torch.fx.Node):
            return None
        value = node.meta.get("val")
        if not isinstance(value, torch.SymInt):
            return None
        return self.symbol_info(value)

    def lower_graph(self, graph: torch.fx.Graph) -> None:
        """Lower an FX graph through the builder callback."""
        if self.lower_node_callback is None:
            raise RuntimeError("BuildContext.lower_node_callback is not configured")
        for node in graph.nodes:
            value = self.lower_node_callback(node)
            if value is not None:
                self.set_value(node, value)

    def reset_for_new_function(self) -> None:
        """Clear per-function state before building another ``func.func``.

        Needed only when compiling more than one function against the same
        ``BuildContext`` (one per phase): a previous function's parameter
        bindings, induction variables, and tensor states are SSA values
        scoped to that function and must not leak into the next one.
        """
        self.param_to_value.clear()
        self.block_id_to_iv.clear()
        self.block_id_to_bounds.clear()
        self.block_id_to_valid.clear()
        self.block_id_to_trip_iv.clear()
        self.tensors.clear()
        self.scalars.clear()

    def bind_loop(
        self,
        block_id: int,
        offset: ir.Value,
        bounds: tuple[int | ir.Value, int | ir.Value],
        span: int | None = None,
    ) -> None:
        """Make ``offset`` the current tile of ``block_id`` in a loop over ``bounds``
        (``span`` wide when known statically though the bounds are runtime values)."""
        self.block_id_to_iv[block_id] = offset
        self.block_id_to_bounds[block_id] = bounds
        valid = self._valid_size(block_id, offset, *bounds, span)
        if valid is None:
            self.block_id_to_valid.pop(block_id, None)
        else:
            self.block_id_to_valid[block_id] = valid

    def _valid_size(
        self,
        block_id: int,
        offset: ir.Value,
        begin: int | ir.Value,
        end: int | ir.Value,
        span: int | None = None,
    ) -> int | ir.Value | None:
        """``min(tile, end - offset)``, or ``None`` if every tile is full.

        The tile is at most ``end - begin`` wide (``BlockGeometry.tile_extent``).
        """
        from .lowering import emit

        if self.geometry.is_grid(block_id):
            return None
        size = self.geometry.tile_extent(block_id)
        d0, d1 = ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)
        if span is not None and span % size == 0:
            return None
        if isinstance(begin, int) and isinstance(end, int):
            if end <= begin or (end - begin) % size == 0:
                return None
            return emit.affine_min(
                [ir.AffineConstantExpr.get(size), ir.AffineConstantExpr.get(end) - d0],
                [offset],
            )
        first, last = self.index_facts(begin), self.index_facts(end)
        if (
            size  # A tile of a loop that spans nothing has size 0.
            and first.multiple % size == 0
            and last.multiple % size == 0
            and first.low is not None
            and first.low >= 0
        ):
            return None  # A triangular loop, on tiles of the same grid.
        return emit.affine_min(
            [ir.AffineConstantExpr.get(size), d1 - d0], [offset, self.as_index(end)]
        )

    def index_facts(self, value: int | ir.Value) -> IndexFacts:
        """What is known of ``value``, as a multiple and bounds, from the loops it is
        built from."""

        def position(candidate: ir.Value) -> IndexFacts | None:
            for block_id, offset in self.block_id_to_iv.items():
                if offset != candidate:
                    continue
                if self.geometry.is_grid(block_id):
                    # hl.grid may start below 0.
                    first = self.block_id_to_bounds[block_id][0]
                    return IndexFacts(low=self.index_facts(first).low)
                begin, end = self.block_id_to_bounds[block_id]
                size = self.geometry.tile_extent(block_id)
                if not size:
                    return None
                first = self.index_facts(begin)
                if not isinstance(begin, int) or not isinstance(end, int):
                    return IndexFacts(
                        size if not first.multiple % size else 1, first.low, None
                    )
                full = not (end - begin) % size
                return IndexFacts(
                    size if not begin % size else 1,
                    begin,
                    end - size if full else end - 1,
                )
            return None

        return index_facts(value, position)

    @contextmanager
    def enter_for_loop(
        self,
        block_id: int,
        induction_variable: ir.Value,
        bounds: tuple[int | ir.Value, int | ir.Value],
        span: int | None = None,
    ) -> Generator[None]:
        """Bind a loop induction variable and bounds, restoring the previous ones."""
        if block_id in self.block_id_to_iv:
            raise UnsupportedOperationError(
                "nested tile loops of one block size",
                reason=(
                    "both loops have one block id, from one block size or from one "
                    "bounded by the other: their tiles would index with the inner "
                    "loop's position"
                ),
                alternatives=[
                    (
                        "hl.register_block_size for each loop, bounded by a tensor's size, "
                        "not by another block size"
                    )
                ],
            )
        saved = [
            (table, table.get(block_id))
            for table in (
                self.block_id_to_iv,
                self.block_id_to_bounds,
                self.block_id_to_valid,
            )
        ]
        self.bind_loop(block_id, induction_variable, bounds, span)
        try:
            yield
        finally:
            for table, previous in saved:
                if previous is None:
                    table.pop(block_id, None)
                else:
                    table[block_id] = previous
