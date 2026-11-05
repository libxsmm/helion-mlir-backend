"""NaN through minima and maxima: torch-mlir's compare-and-select clamps and
``aten.maximum``/``aten.minimum`` made NaN-propagating (``nan_minmax``), on the
CPU.

Regression: ``torch.clamp(x * 1.5, -1.0, 1.0)`` on bf16 (``[32, 256]`` tiles)
gave -1.0 at NaN inputs on the CPU and the XPU; torch gives NaN.
"""

from __future__ import annotations

import helion
import helion.language as hl
from mlir import ir
import pytest
import torch
import torch.nn.functional as F

from tests.inputs import boundary_tile

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir.aten_bridge.nan_minmax import propagate_nan

BF16 = torch.bfloat16
TILE = helion.Config(block_sizes=[32, 256])
_BOUNDS = (0.5, 1.0, 1.0 / 1.5, 3.0, 6.0)


def _kernel(config: helion.Config) -> object:
    return helion.kernel(backend="mlir", static_shapes=True, config=config)


@_kernel(TILE)
def clamp_scaled(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for t in hl.tile(x.shape):
        out[t] = torch.clamp(x[t] * 1.5, -1.0, 1.0)
    return out


@_kernel(TILE)
def clamp_family(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outs = [torch.empty_like(x) for _ in range(5)]
    lower, upper, hardtanh, relu6, hardsigmoid = outs
    for t in hl.tile(x.shape):
        lower[t] = torch.clamp_min(x[t], 0.5)
        upper[t] = torch.clamp_max(x[t], 0.5)
        hardtanh[t] = F.hardtanh(x[t])
        relu6[t] = F.relu6(x[t])
        hardsigmoid[t] = F.hardsigmoid(x[t])
    return lower, upper, hardtanh, relu6, hardsigmoid


def clamp_family_reference(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return (
        torch.clamp_min(x, 0.5),
        torch.clamp_max(x, 0.5),
        F.hardtanh(x),
        F.relu6(x),
        F.hardsigmoid(x),
    )


@_kernel(TILE)
def binary_minmax(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum, minimum, clamped = (torch.empty_like(x) for _ in range(3))
    for t in hl.tile(x.shape):
        maximum[t] = torch.maximum(x[t], y[t])
        minimum[t] = torch.minimum(x[t], y[t])
        clamped[t] = torch.clamp(x[t], min=y[t])
    return maximum, minimum, clamped


def binary_minmax_reference(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    return torch.maximum(x, y), torch.minimum(x, y), torch.clamp(x, min=y)


@_kernel(TILE)
def relu_and_where(
    x: torch.Tensor, out: torch.Tensor, out2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    for t in hl.tile(x.shape):
        out[t] = torch.relu(x[t])
        out2[t] = torch.where(x[t] > 0, x[t], 0.0)
    return out, out2


def nan_inputs(
    *shape: int, dtype: torch.dtype, device: str = "cpu", seed: int = 0
) -> torch.Tensor:
    """The values these kernels turn on: the bounds of their clamps, of ``x * 1.5`` too,
    and next to them, with NaN, infinities, signed zeros, ``tiny`` and ``max``. Rows and
    16-column lanes differ (``inputs.lanes``), so a broadcast row or lane can't pass.
    ``seed``: another placement of them."""
    x = boundary_tile(_BOUNDS, *shape, dtype=dtype, device=device)
    return x.roll(seed * 5, dims=-1)


def assert_same_values(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The same NaN pattern, and the other elements bit for bit."""
    assert out.dtype == ref.dtype and out.shape == ref.shape
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(out), nan)
    as_int = torch.int16 if out.element_size() == 2 else torch.int32
    assert torch.equal(out[~nan].view(as_int), ref[~nan].view(as_int))


# --- the rewrite ---------------------------------------------------------------


def _function(body: str, args: str, result: str) -> str:
    return f"func.func @f({args}) -> {result} {{\n{body}\n}}"


def _rewritten(body: str, args: str = "%x: f32, %y: f32", result: str = "f32") -> str:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(_function(body, args, result))
        propagate_nan(module.body.operations[0])
        module.operation.verify()
        return str(module)


def _parsed(body: str) -> str:
    with ir.Context():
        return str(ir.Module.parse(_function(body, "%x: f32, %y: f32", "f32")))


@pytest.mark.parametrize(
    ("predicate", "swapped"), [("ult", "olt"), ("ugt", "ogt"), ("ole", "ule")]
)
def test_constant_bound_swaps_the_predicate(predicate: str, swapped: str) -> None:
    """``select(cmpf ult x, c, c, x)`` gives ``c`` for NaN ``x``: the ordered
    form gives ``x``; ordered operands are unchanged."""
    picked = "%c, %x" if predicate.startswith("u") else "%x, %c"
    body = f"""
  %c = arith.constant -1.0 : f32
  %p = arith.cmpf {predicate}, %x, %c : f32
  %r = arith.select %p, {picked} : f32
  return %r : f32"""
    text = _rewritten(body)
    assert f"arith.cmpf {swapped}, %arg0, %cst" in text
    assert f"arith.cmpf {predicate}," not in text
    assert text.count("arith.select") == 1


def test_two_runtime_operands_select_a_nan() -> None:
    """``aten.maximum``'s ``select(cmpf ogt a, b, a, b)`` gives ``b`` for NaN
    ``a``: a NaN ``a`` is selected after it."""
    body = """
  %p = arith.cmpf ogt, %x, %y : f32
  %r = arith.select %p, %x, %y : f32
  return %r : f32"""
    text = _rewritten(body)
    assert "%0 = arith.cmpf ogt, %arg0, %arg1 : f32" in text
    assert "%1 = arith.select %0, %arg0, %arg1 : f32" in text
    assert "%2 = arith.cmpf uno, %arg0, %arg0 : f32" in text
    assert "%3 = arith.select %2, %arg0, %1 : f32" in text
    assert "return %3 : f32" in text


def test_vector_operands() -> None:
    args = "%x: vector<4xf32>, %y: vector<4xf32>"
    body = """
  %c = arith.constant dense<1.0> : vector<4xf32>
  %p = arith.cmpf ugt, %x, %c : vector<4xf32>
  %r = arith.select %p, %c, %x : vector<4xi1>, vector<4xf32>
  %q = arith.cmpf olt, %r, %y : vector<4xf32>
  %s = arith.select %q, %r, %y : vector<4xi1>, vector<4xf32>
  return %s : vector<4xf32>"""
    text = _rewritten(body, args, "vector<4xf32>")
    assert "arith.cmpf ogt, %arg0, %cst : vector<4xf32>" in text
    # The second pair picks y when unordered: a NaN r (%1) is selected.
    assert "arith.cmpf uno, %1, %1 : vector<4xf32>" in text


@pytest.mark.parametrize(
    "body",
    [
        # ReLU: a NaN x is already picked (the other operand is a constant).
        """
  %c = arith.constant 0.0 : f32
  %p = arith.cmpf ugt, %x, %c : f32
  %r = arith.select %p, %x, %c : f32
  return %r : f32""",
        # Not a minimum or maximum: the selected values are not the compared ones.
        """
  %c = arith.constant 0.0 : f32
  %p = arith.cmpf ogt, %x, %c : f32
  %r = arith.select %p, %x, %y : f32
  return %r : f32""",
        # An equality predicate.
        """
  %p = arith.cmpf oeq, %x, %y : f32
  %r = arith.select %p, %x, %y : f32
  return %r : f32""",
        # The same value on both sides.
        """
  %p = arith.cmpf ogt, %x, %y : f32
  %r = arith.select %p, %x, %x : f32
  return %r : f32""",
    ],
    ids=["relu", "other_values", "equality", "same_value"],
)
def test_other_selects_are_left_alone(body: str) -> None:
    assert _rewritten(body) == _parsed(body)


def test_only_minmax_ops_are_rewritten() -> None:
    """``torch.where(x > 0, x, 0)`` gives 0 for NaN (the same compare-and-select
    shape as a maximum): its helpers are not rewritten."""
    x = nan_inputs(64, 512, dtype=torch.float32)
    text = str(generate_mlir(relu_and_where, [x, x, x]))
    assert "arith.cmpf uno" not in text
    assert "arith.cmpf ugt" in text  # ReLU's, kept


def test_clamp_ir_uses_ordered_predicates() -> None:
    x = nan_inputs(64, 512, dtype=BF16)
    text = str(generate_mlir(clamp_scaled, [x, x]))
    assert "arith.cmpf olt" in text
    assert "arith.cmpf ogt" in text
    assert "arith.cmpf ult" not in text
    assert "arith.cmpf ugt" not in text


# --- the CPU -------------------------------------------------------------------


def test_clamp_propagates_nan() -> None:
    """Regression (exact kernel and config): -1.0 at NaN inputs."""
    x = nan_inputs(64, 256, dtype=BF16)
    out = clamp_scaled(x, torch.full_like(x, 7.0))
    assert_same_values(out, torch.clamp(x * 1.5, -1.0, 1.0))


@pytest.mark.parametrize("dtype", [BF16, torch.float32], ids=["bf16", "f32"])
def test_clamp_family_propagates_nan(dtype: torch.dtype) -> None:
    x = nan_inputs(64, 512, dtype=dtype)
    for out, ref in zip(clamp_family(x), clamp_family_reference(x), strict=True):
        assert_same_values(out, ref)


def test_binary_minmax_propagates_nan() -> None:
    x = nan_inputs(64, 512, dtype=torch.float32)
    y = nan_inputs(64, 512, dtype=torch.float32, seed=1)
    # In row 0's first columns NaN is in x only; in row 3's, in y whatever x holds.
    y[0, :16] = 1.0
    y[3, :16] = float("nan")
    for out, ref in zip(
        binary_minmax(x, y), binary_minmax_reference(x, y), strict=True
    ):
        assert_same_values(out, ref)


def test_where_keeps_its_value_at_nan() -> None:
    x = nan_inputs(64, 512, dtype=torch.float32)
    relu, where = relu_and_where(x, torch.empty_like(x), torch.empty_like(x))
    assert torch.isnan(relu[x.isnan()]).all()
    # +0.0: -0.0 is the next test's.
    assert_same_values(relu + 0.0, torch.relu(x) + 0.0)
    assert_same_values(where, torch.where(x > 0, x, 0.0))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="relu(-0.0) is +0.0 here, -0.0 in torch: test_relu_keeps_negative_zero's "
    "finding, 2026-11-03, also on the XPU (test_unit_clamps_match_torch's docstring)",
)
def test_relu_keeps_negative_zero() -> None:
    x = nan_inputs(64, 512, dtype=torch.float32)
    relu, _ = relu_and_where(x, torch.empty_like(x), torch.empty_like(x))
    assert_same_values(relu, torch.relu(x))
