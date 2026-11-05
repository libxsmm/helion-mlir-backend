"""Questions about IR by its ops and attributes, not by substrings of its text: a test
says which op it means, and survives value renames, attribute order and printing."""

from __future__ import annotations

from dataclasses import dataclass
import re

from mlir import ir


def parse(text: str) -> ir.Module:
    """The module of ``text``, IR a stage printed (unknown ops allowed: only read)."""
    context = ir.Context()
    context.allow_unregistered_dialects = True
    return ir.Module.parse(text, context)


def walk(module: ir.Module | ir.Operation) -> list[ir.Operation]:
    """Every op under ``module``, outer ones first."""
    root = module.operation if isinstance(module, ir.Module) else module
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        found.append(op)
        return ir.WalkResult.ADVANCE

    root.walk(visit, ir.WalkOrder.PRE_ORDER)
    return found


def ops_named(module: ir.Module | ir.Operation, *names: str) -> list[ir.Operation]:
    """The ops under ``module`` named one of ``names``; valid while ``module`` lives (keep
    it in a variable: an op doesn't keep it alive)."""
    return [op for op in walk(module) if op.name in names]


@dataclass(frozen=True)
class Layout:
    """A ``#xegpu.layout``; a field it doesn't set is ``None``."""

    sg_layout: tuple[int, ...] | None
    sg_data: tuple[int, ...] | None
    inst_data: tuple[int, ...] | None
    order: tuple[int, ...] | None


def layout_of(op: ir.Operation, attribute: str = "layout") -> Layout | None:
    """The layout ``op`` carries as ``attribute`` (``layout_a``, ``layout_b`` or
    ``layout_cd`` on a DPAS), ``None`` without one."""
    if attribute not in op.attributes:
        return None
    text = str(op.attributes[attribute])

    def field(name: str) -> tuple[int, ...] | None:
        match = re.search(rf"\b{name} = \[([\d, ]*)\]", text)
        if match is None:
            return None
        return tuple(int(number) for number in match.group(1).split(","))

    return Layout(
        field("sg_layout"), field("sg_data"), field("inst_data"), field("order")
    )


def static_sizes(op: ir.Operation) -> list[int]:
    """The static sizes of a slice op."""
    return list(ir.DenseI64ArrayAttr(op.attributes["static_sizes"]))
