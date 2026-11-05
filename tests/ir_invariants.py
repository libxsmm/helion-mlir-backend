"""IR invariants checked on the lowerings the suite already does.

:func:`install` wraps the module builder (``MLIRModuleBuilder.build``, every kernel call
and ``generate_mlir``), for tests only: ``tests/conftest.py`` sets
``HELION_MLIR_TEST_IR_CHECKS``. Its values:

- ``fail`` (the suite's default): a violation raises, failing the test;
- ``report:PATH``: each violation is one JSON line in ``PATH``, and the test goes on
  (one run lists them all);
- ``off``.

A kernel that means to compute in f64 says so: an f64 tensor argument, or ``float64`` or
``double`` as a name in its code (``torch.float64``, ``.double()``; a word in a comment or
docstring doesn't count). A lowering runs once per kernel and signature, so a declaration on
one test would depend on the order of tests.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import textwrap
from typing import TYPE_CHECKING
from unittest.mock import patch

from mlir import ir

from tests.ir_query import walk

from helion_mlir_backend._compiler.mlir.codegen import MLIRModuleBuilder

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

MODE_VARIABLE = "HELION_MLIR_TEST_IR_CHECKS"
EXPECTED_VARIABLE = "HELION_MLIR_TEST_EXPECTED_VIOLATION"
"""``check:path``: set by ``@pytest.mark.expect_violation(check)``: a violation of that check
is noted in ``path``, not raised, and the test fails if none is."""


_F64_DECLARED = {"float64", "double"}
"""Names in the kernel's code (not in its comments or docstring) that ask for f64."""
_F64_HELPERS = ("_aten_var_correction_", "_aten_var_mean_correction_")
"""torch-mlir's decomposition of ``var`` and ``std`` computes in f64, whatever the tile's type;
the kernel's source can't say otherwise."""
_F64_HARMLESS = {"arith.constant", "func.func", "func.return", "func.call"}
"""Only values: what uses them is what computes in f64."""


class IRInvariantError(AssertionError):
    """The IR breaks an invariant (a test-only failure)."""


def _element_type(value_type: ir.Type) -> ir.Type:
    return (
        ir.ShapedType(value_type).element_type
        if isinstance(value_type, ir.ShapedType)
        else value_type
    )


def _is_f64(value: ir.Value) -> bool:
    return isinstance(_element_type(value.type), ir.F64Type)


def _narrows_a_scalar(op: ir.Operation) -> bool:
    """A scalar parameter or constant taken out of f64: no data is widened."""
    return op.name == "arith.truncf" and not isinstance(
        op.operands[0].type, ir.ShapedType
    )


def _helper_ops(module: ir.Module) -> Iterator[ir.Operation]:
    """The ops under ``module``, but those of the functions ``_F64_HELPERS`` names."""
    for op in module.body.operations:
        operation = op.operation
        name = (
            str(operation.attributes["sym_name"])
            if operation.name == "func.func"
            else ""
        )
        if not name.strip('"').startswith(_F64_HELPERS):
            yield from walk(operation)


def f64_ops(module: ir.Module) -> list[str]:
    """The ops computing in f64 (a bare constant or scalar argument does not)."""
    if "f64" not in str(module):
        return []
    return [
        str(op).split("\n", 1)[0].strip()
        for op in _helper_ops(module)
        if op.name not in _F64_HARMLESS
        and not _narrows_a_scalar(op)
        and any(_is_f64(value) for value in (*op.operands, *op.results))
    ]


def _declares_f64(module: ir.Module, function: Callable[..., object] | None) -> bool:
    for op in walk(module):
        # The entry: ATen helpers of a widened tile take f64 tensors without a declaration.
        if op.name == "func.func" and "public" in str(op.attributes["sym_visibility"]):
            arguments = op.regions[0].blocks[0].arguments
            if any(
                isinstance(arg.type, ir.ShapedType) and _is_f64(arg)
                for arg in arguments
            ):
                return True
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return False
    names = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute | ast.Name)
    }
    return not names.isdisjoint(_F64_DECLARED)


def _violation(check: str, where: str, findings: list[str]) -> None:
    expected, _, fired = os.environ.get(EXPECTED_VARIABLE, "").partition(":")
    if check == expected:
        with open(fired, "a") as log:
            log.write(f"{where}: {findings[:2]}\n")
        return
    mode = os.environ.get(MODE_VARIABLE, "off")
    if mode.startswith("report:"):
        line = json.dumps(
            {
                "check": check,
                "where": where,
                "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
                "count": len(findings),
                "first": findings[:4],
            }
        )
        with open(mode.removeprefix("report:"), "a") as log:
            log.write(line + "\n")
        return
    if mode == "fail":
        shown = "\n  ".join(findings[:8])
        raise IRInvariantError(
            f"{check} at {where}: {len(findings)} violations (tests/ir_invariants.py)"
            f":\n  {shown}"
        )


def _checked_build(self: MLIRModuleBuilder) -> ir.Module:
    module = _BUILD(self)
    definition = getattr(self.hf, "definition", None)
    function = getattr(definition, "fn", None)
    with module.context:
        if (found := f64_ops(module)) and not _declares_f64(module, function):
            name = getattr(definition, "name", "?")
            _violation("f64-not-asked-for", f"frontend of {name}", found)
    return module


_BUILD = MLIRModuleBuilder.build


def valid_mode(mode: str) -> bool:
    return mode in ("fail", "off") or (
        mode.startswith("report:") and bool(mode.removeprefix("report:"))
    )


def install() -> None:
    """Hooks on, for this process and its forks (nothing to undo: tests only)."""
    if os.environ.get(MODE_VARIABLE, "off") == "off":
        return
    patch.object(MLIRModuleBuilder, "build", _checked_build).start()
