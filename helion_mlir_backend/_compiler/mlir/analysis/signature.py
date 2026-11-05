"""Kernel calling convention: the host tensors and runtime scalars device code uses.

A tensor ref is named by its ``_host_tensor`` name, which is the host expression
that produces it (``x``, ``out``, ``self.weight``). Refs are ordered declared tensor
parameters first (declaration order), then other host tensors in first-use order.
A ref written anywhere is ``inout``; every other ref is ``in``. A read-only host
view of a declared parameter holding the same elements in the same order (a
reshape) is not a ref: it lowers to a reshape of that parameter. Any other view
(a slice, a transpose) is a ref of its own, which the host code computes.

Runtime scalars are symbolic ints/floats/bools with a host origin (non-constexpr scalar
parameters); Helion keys them by type only, so their values are call arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import helion.language._tracing_ops as tracing_ops
import helion.language.memory_ops as memory_ops
import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ..support.block_ids import SCALAR_SYMBOL_KINDS
from ..support.block_ids import block_id_from_key
from ..support.block_ids import symbol_origin_info
from ..support.errors import UnsupportedOperationError
from ..support.index_meta import resolve_index_descriptor
from .tensor_effects import accessed_tensor
from .tensor_effects import host_tensor_name
from .tensor_effects import subgraph_ids

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping

    from helion._compiler.host_function import HostFunction
    import sympy

    from ..build_context import BuildContext
    from .tensor_effects import TensorEffects


@dataclass(frozen=True)
class TensorRef:
    name: str
    fake: torch.Tensor
    written: bool
    tensor_param: int | None
    """Position among the kernel's tensor parameters, if the ref is one."""


@dataclass(frozen=True)
class ScalarArg:
    key: str
    """The ``_get_symnode`` key device code uses."""
    host_expr: str
    dtype: torch.dtype
    expr: object = None
    """The ``sympy`` expression an int scalar carries (a size source)."""


@dataclass(frozen=True)
class PhaseSignature:
    root_positions: tuple[int, ...]
    ins: tuple[str, ...]
    inouts: tuple[str, ...]
    scalars: tuple[str, ...]


@dataclass(frozen=True)
class KernelSignature:
    refs: dict[str, TensorRef]
    scalars: dict[str, ScalarArg]
    aliases: dict[str, str]
    """Read-only host reshapes of a declared parameter -> that parameter."""
    phases: tuple[PhaseSignature, ...]

    @property
    def inouts(self) -> list[TensorRef]:
        return [ref for ref in self.refs.values() if ref.written]

    @property
    def ins(self) -> list[TensorRef]:
        return [ref for ref in self.refs.values() if not ref.written]

    @classmethod
    def from_host_function(
        cls, hf: HostFunction, effects: TensorEffects
    ) -> KernelSignature:
        tensor_params = [
            name
            for name, value in hf.params.arguments.items()
            if isinstance(value, torch.Tensor)
        ]
        device_ir = hf.device_ir
        phase_nodes = [
            [
                node
                for position in phase.roots
                for node in _program_order(hf, device_ir.root_ids[position])
            ]
            for phase in device_ir.phases
        ]
        all_nodes = [node for nodes in phase_nodes for node in nodes]
        used = dict.fromkeys(
            name for node in all_nodes if (name := host_tensor_name(node)) is not None
        )
        written = _stored(all_nodes)
        scalars: dict[str, ScalarArg] = {}
        for node in all_nodes:
            if (scalar := _runtime_scalar(hf, node)) is not None:
                scalars.setdefault(scalar.key, scalar)

        aliases = {
            name: base
            for name in used
            if name not in tensor_params
            and name not in written
            and (base := _declared_base(hf, effects.fakes[name], tensor_params))
            and _same_elements(effects.fakes[name], hf.params.arguments[base])
        }
        ordered = [
            name for name in tensor_params if name in used or name in aliases.values()
        ]
        ordered += [
            name for name in used if name not in tensor_params and name not in aliases
        ]
        refs = {
            name: TensorRef(
                name,
                effects.fakes.get(name, hf.params.arguments.get(name)),
                name in written,
                tensor_params.index(name) if name in tensor_params else None,
            )
            for name in ordered
        }

        phases = []
        for phase, nodes in zip(device_ir.phases, phase_nodes, strict=True):
            touched = {
                aliases.get(name, name)
                for node in nodes
                if (name := host_tensor_name(node)) is not None
            }
            stored = _stored(nodes)
            keys = {
                node.args[0] for node in nodes if _runtime_scalar(hf, node) is not None
            }
            phases.append(
                PhaseSignature(
                    tuple(phase.roots),
                    tuple(n for n in refs if n in touched and n not in stored),
                    tuple(n for n in refs if n in stored),
                    tuple(key for key in scalars if key in keys),
                )
            )
        return cls(refs, scalars, aliases, tuple(phases))


@dataclass(frozen=True)
class ReadOrder:
    """What a kernel's reads see, by tensor name, in program order."""

    after: Mapping[str, frozenset[str]]
    """The written tensors stored before a read of this tensor: if they share
    elements with it, the read must see those stores."""
    same: Mapping[str, frozenset[str]]
    """The tensors stored by a root loop that also reads this tensor."""


def read_order(
    hf: HostFunction, signature: KernelSignature, ctx: BuildContext
) -> ReadOrder:
    """The kernel's :class:`ReadOrder`, over its root loops in program order. A loop
    body that runs again reads what its earlier iterations stored."""
    after: dict[str, set[str]] = {}
    same: dict[str, set[str]] = {}
    written: set[str] = set()

    def base(name: str) -> str:
        return signature.aliases.get(name, name)

    def visit(graph_id: int, stored: set[str]) -> None:
        for node in hf.device_ir.graphs[graph_id].graph.nodes:
            if (name := accessed_tensor(node)) is not None:
                if node.target is memory_ops.store:
                    written.add(base(name))
                else:
                    after.setdefault(base(name), set()).update(written)
                    same.setdefault(base(name), set()).update(stored)
            for subgraph_id in subgraph_ids(node):
                written.update(map(base, _revisited(hf, node, subgraph_id, ctx)))
                visit(subgraph_id, stored)

    for root_id in hf.device_ir.root_ids:
        stored = set(map(base, _stored(list(_program_order(hf, root_id)))))
        visit(root_id, stored)
    return ReadOrder(
        {name: frozenset(names) for name, names in after.items()},
        {name: frozenset(names) for name, names in same.items()},
    )


def _revisited(
    hf: HostFunction, node: torch.fx.Node, graph_id: int, ctx: BuildContext
) -> set[str]:
    """The tensors a nested body stores at elements its next iteration runs over: all
    of a ``while``'s stores, a ``for``'s stores not indexed by its block ids (``range``
    and a nested ``hl.tile`` look alike in the device graph). An ``if`` runs once."""
    if node.target is tracing_ops._if:
        return set()
    stores = [
        (name, store)
        for store in _program_order(hf, graph_id)
        if (name := accessed_tensor(store)) is not None
        and store.target is memory_ops.store
    ]
    if node.target is tracing_ops._while_loop:
        return {name for name, _ in stores}
    block_ids = list(hf.device_ir.graphs[graph_id].block_ids or ())
    return {name for name, store in stores if not _indexed_by(store, block_ids, ctx)}


def _indexed_by(store: torch.fx.Node, block_ids: list[int], ctx: BuildContext) -> bool:
    """A store's index uses one of ``block_ids`` or a tile nested in them; an index
    that doesn't resolve counts as not, which refuses rather than reads stale."""
    for index in store.args[1]:
        if index is None or isinstance(index, slice):
            continue
        try:
            block_id = resolve_index_descriptor(ctx, index).block_id
        except UnsupportedOperationError:
            continue
        if (
            block_id is not None
            and ctx.geometry.owning_block(block_id, block_ids) is not None
        ):
            return True
    return False


@dataclass(frozen=True)
class ScalarIndex:
    """A tensor access at a position given by runtime scalars."""

    tensor: str
    dim: int
    expr: object
    """The position, a ``sympy`` expression of the scalars' ``expr``."""
    size: int | None
    """The accessed dim's size, if static; a reshaped alias of another tensor has no
    argument of its own to read it from at the call."""


def runtime_index_expr(
    signature: KernelSignature, node: object, ctx: BuildContext | None = None
) -> object | None:
    """The expression of an index node computed from runtime integer scalars alone
    (their symbols, or a host-computed scalar like ``i = p + 1`` plus a constant), a
    position the host can check; else ``None``. A tile's position plus a scalar is
    refused: no check is written for its range."""
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    if not isinstance(value, torch.SymInt):
        return None
    expr = value.node.expr
    symbols = expr.free_symbols
    if not symbols:
        return None
    known = {s.expr for s in signature.scalars.values() if s.expr is not None}
    if expr in known or symbols <= known or any((expr - k).is_number for k in known):
        return expr
    if known and any(k.free_symbols & symbols for k in known):
        raise UnsupportedOperationError(
            "a tensor index adding a tile position and a runtime scalar",
            reason=f"the position {expr} isn't checked against the tensor's bounds",
            alternatives=["a runtime scalar alone, or a tile position alone"],
        )
    return None


def scalar_indices(
    hf: HostFunction, signature: KernelSignature
) -> tuple[ScalarIndex, ...]:
    """Every access at a runtime position, masked or guarded too: a call checks them
    whether or not they would run."""
    found: list[ScalarIndex] = []

    def visit(graph_id: int) -> None:
        for node in hf.device_ir.graphs[graph_id].graph.nodes:
            name = accessed_tensor(node)
            if name is not None:
                indices = (index for index in node.args[1] if index is not None)
                for dim, index in enumerate(indices):
                    if (expr := runtime_index_expr(signature, index)) is not None:
                        access = ScalarIndex(name, dim, expr, _static_size(node, dim))
                        found.append(access)
                        _check_alias_size(signature, access)
            for subgraph_id in subgraph_ids(node):
                visit(subgraph_id)

    for root_id in hf.device_ir.root_ids:
        visit(root_id)
    return tuple(dict.fromkeys(found))


def _static_size(node: torch.fx.Node, dim: int) -> int | None:
    """Dim ``dim`` of the tensor a load or store ``node`` accesses, if static."""
    value = (
        node.args[0].meta.get("val")
        if isinstance(node.args[0], torch.fx.Node)
        else None
    )
    if value is None or dim >= value.dim():
        return None
    size = value.shape[dim]
    if isinstance(size, torch.SymInt):
        return None if size.node.expr.free_symbols else int(size.node.expr)
    return int(size)


def _check_alias_size(signature: KernelSignature, access: ScalarIndex) -> None:
    """Refuse a runtime position into a tensor the call can't size: a reshaped alias
    of another tensor, of a runtime shape, has no argument, and a position out of
    bounds would store outside it."""
    if access.size is None and access.tensor not in signature.refs:
        raise UnsupportedOperationError(
            "a runtime position into a reshaped view of a runtime shape",
            reason=(
                f"'{access.tensor}' shares its elements with another argument and "
                "the call has no size to check the position against"
            ),
            alternatives=["static shapes", "a position on the tensor itself"],
        )


def _program_order(hf: HostFunction, graph_id: int) -> Iterator[torch.fx.Node]:
    """The graph's nodes, with each subgraph's nodes at its loop, if or while node."""
    for node in hf.device_ir.graphs[graph_id].graph.nodes:
        yield node
        for subgraph_id in subgraph_ids(node):
            yield from _program_order(hf, subgraph_id)


def _stored(nodes: list[torch.fx.Node]) -> set[str]:
    return {
        name
        for node in nodes
        if node.op == "call_function"
        and node.target is memory_ops.store
        and (name := accessed_tensor(node)) is not None
    }


def _runtime_scalar(hf: HostFunction, node: torch.fx.Node) -> ScalarArg | None:
    if node.op != "call_function" or node.target is not tracing_ops._get_symnode:
        return None
    key = node.args[0]
    value = node.meta.get("val")
    if (
        not isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool))
        or not value.node.expr.free_symbols
        or block_id_from_key(key) is not None
        or symbol_origin_info(hf, value) is not None
        or _on_tile_positions(hf, value.node.expr)
    ):
        return None
    symbol_origin = hf.expr_to_origin.get(value.node.expr)
    if symbol_origin is None or not symbol_origin.origin.is_host():
        return None
    if isinstance(value, torch.SymFloat):
        return ScalarArg(key, symbol_origin.origin.host_str(), torch.float64)
    if isinstance(value, torch.SymBool):
        return ScalarArg(key, symbol_origin.origin.host_str(), torch.bool)
    return ScalarArg(key, symbol_origin.origin.host_str(), torch.int64, value.node.expr)


def _on_tile_positions(hf: HostFunction, expr: sympy.Basic) -> bool:
    """Whether ``expr`` depends on a loop's position. Such a value exists on the device
    alone, though Helion names it by a host local when host code follows its loop."""
    return any(
        info[1] in SCALAR_SYMBOL_KINDS
        for symbol in expr.free_symbols
        if (info := symbol_origin_info(hf, symbol)) is not None
    )


def _declared_base(
    hf: HostFunction, tensor: torch.Tensor, tensor_params: list[str]
) -> str | None:
    """The declared parameter a host tensor is a view of, via origins and ``._base``."""
    seen: set[int] = set()
    current: torch.Tensor | None = tensor
    while isinstance(current, torch.Tensor) and id(current) not in seen:
        seen.add(id(current))
        origin = hf.tensor_to_origin.get(current)
        if origin is not None and origin.host_str() in tensor_params:
            return origin.host_str()
        current = getattr(current, "_base", None)
    return None


def _same_elements(view: torch.Tensor, base: torch.Tensor) -> bool:
    """Whether ``view`` provably holds all of ``base``'s elements in row-major order."""
    return (
        statically_known_true(view.storage_offset() == 0)
        and statically_known_true(view.numel() == base.numel())
        and _row_major(view)
        and _row_major(base)
    )


def _row_major(tensor: torch.Tensor) -> bool:
    expected: int | torch.SymInt = 1
    for size, stride in reversed(list(zip(tensor.shape, tensor.stride(), strict=True))):
        if not statically_known_true(size == 1) and not statically_known_true(
            stride == expected
        ):
            return False
        expected = expected * size
    return True
