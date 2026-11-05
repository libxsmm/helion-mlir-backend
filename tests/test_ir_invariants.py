"""Each hook of ``tests/ir_invariants.py`` fires, and leaves declared forms alone."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch

from tests import ir_invariants

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.codegen import MLIRModuleBuilder

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get(ir_invariants.MODE_VARIABLE) == "off",
    reason="the hooks are off in this run",
)
_CONFIG = helion.Config(block_sizes=[32, 32])


@pytest.fixture(autouse=True)
def _failing_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests' violations must raise, also in a run in ``report:`` mode."""
    monkeypatch.setenv(ir_invariants.MODE_VARIABLE, "fail")


_WIDE = torch.float64


def _widens_by_a_name(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(_WIDE) * 2.0).to(x.dtype)
    return out


def _declared(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(torch.float64) * 2.0).to(x.dtype)
    return out


def _times_two(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(fn, backend="mlir", static_shapes=True, config=_CONFIG)


def _tensor(device: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(64, 64, dtype=dtype)


@pytest.mark.parametrize("device", ["cpu"])
def test_undeclared_f64_kernel_fails_the_frontend(device: str) -> None:
    with pytest.raises(ir_invariants.IRInvariantError, match="f64-not-asked-for"):
        generate_mlir(_kernel(_widens_by_a_name), [_tensor(device)], target=device)


def test_compare_of_a_tile_widened_to_f64_is_flagged() -> None:
    """The IR torch-mlir made of ``x <= 0.1``: what the hook first found."""
    module = ir.Module.parse(
        """
func.func @f(%x: f32) -> i1 {
  %cst = arith.constant 1.000000e-01 : f64
  %wide = arith.extf %x : f32 to f64
  %le = arith.cmpf ole, %wide, %cst : f64
  return %le : i1
}""",
        ir.Context(),
    )
    assert ir_invariants.f64_ops(module) == [
        "%0 = arith.extf %arg0 : f32 to f64",
        "%1 = arith.cmpf ole, %0, %cst : f64",
    ]


def _word_in_a_comment(x: torch.Tensor) -> torch.Tensor:
    # float64 and double: words here are no declaration.
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(_WIDE) * 2.0).to(x.dtype)
    return out


def _word_in_a_docstring(x: torch.Tensor) -> torch.Tensor:
    """Not a double precision kernel: float64 is only a word here."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(_WIDE) * 2.0).to(x.dtype)
    return out


@pytest.mark.parametrize("fn", [_word_in_a_comment, _word_in_a_docstring])
def test_a_word_in_text_is_no_declaration(fn: object) -> None:
    with pytest.raises(ir_invariants.IRInvariantError, match="f64-not-asked-for"):
        generate_mlir(_kernel(fn), [_tensor("cpu")], target="cpu")


@pytest.mark.parametrize(
    ("helper", "found"),
    [
        ("_aten_var_correction_0", 0),
        ("_aten_var_mean_correction_0", 0),
        ("_aten_add_Tensor_0", 1),
    ],
)
def test_only_var_helpers_may_compute_in_f64(helper: str, found: int) -> None:
    module = ir.Module.parse(
        f"""
func.func private @{helper}(%x: f32) -> f64 {{
  %0 = arith.extf %x : f32 to f64
  return %0 : f64
}}""",
        ir.Context(),
    )
    assert len(ir_invariants.f64_ops(module)) == found


def test_scalar_numbers_narrowed_are_not_flagged() -> None:
    module = ir.Module.parse(
        """
func.func @f(%alpha: f64) -> f32 {
  %cst = arith.constant 1.000000e-01 : f64
  %0 = arith.truncf %alpha : f64 to f32
  %1 = arith.truncf %cst : f64 to f32
  %2 = arith.addf %0, %1 : f32
  return %2 : f32
}""",
        ir.Context(),
    )
    assert ir_invariants.f64_ops(module) == []


@pytest.mark.parametrize("device", ["cpu"])
def test_f64_in_the_source_or_arguments_is_declared(device: str) -> None:
    generate_mlir(_kernel(_declared), [_tensor(device)], target=device)
    generate_mlir(_kernel(_times_two), [_tensor(device, torch.float64)], target=device)


@pytest.mark.parametrize("device", ["cpu"])
def test_a_kernel_call_goes_through_the_f64_hook(device: str) -> None:
    with pytest.raises(ir_invariants.IRInvariantError, match="f64-not-asked-for"):
        _kernel(_widens_by_a_name)(_tensor(device))


@pytest.mark.isolated
def test_a_kernel_call_goes_through_the_f64_hook_in_an_isolated_child() -> None:
    with pytest.raises(ir_invariants.IRInvariantError, match="f64-not-asked-for"):
        _kernel(_widens_by_a_name)(_tensor("cpu"))


@pytest.mark.xfail(strict=True, reason="a named exemption that doesn't fire fails")
@pytest.mark.expect_violation("f64-not-asked-for")
def test_an_exemption_that_does_not_fire_fails() -> None:
    pass


def test_unknown_modes_are_refused() -> None:
    assert ir_invariants.valid_mode("fail")
    assert ir_invariants.valid_mode("off")
    assert ir_invariants.valid_mode("report:/tmp/x")
    assert not ir_invariants.valid_mode("report:")
    assert not ir_invariants.valid_mode("Fail")
    assert not ir_invariants.valid_mode("")


def test_report_mode_records_and_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "findings.jsonl"
    monkeypatch.setenv(ir_invariants.MODE_VARIABLE, f"report:{report}")
    generate_mlir(_kernel(_widens_by_a_name), [_tensor("cpu")], target="cpu")
    finding = json.loads(report.read_text())
    assert finding["check"] == "f64-not-asked-for"
    assert finding["where"] == "frontend of _widens_by_a_name"
    assert "test_report_mode_records_and_goes_on" in finding["test"]
    assert any("arith.extf" in line for line in finding["first"])


@pytest.mark.isolated
def test_hooks_are_on_in_isolated_children() -> None:
    assert MLIRModuleBuilder.build is ir_invariants._checked_build
