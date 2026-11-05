"""Compares of a tile with a Python number, against torch, element by element, at the elements
where it matters: the number itself in the tile's type, and its neighbours. On the CPU. ``HELION_TEST_FULL_SWEEP=1``: every number and type."""

from __future__ import annotations

import helion
import pytest
import torch

from tests import scalar_compare_kernels as kernels

from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[32, 32]),
    )


def _call(kernel: helion.Kernel, *inputs: object) -> torch.Tensor:
    return kernel(*inputs)


def _run(fn: object, *inputs: object) -> torch.Tensor:
    return _call(_kernel(fn), *inputs)


def _same_outputs(fn: object, x: torch.Tensor, expected: list[torch.Tensor]) -> None:
    outputs = kernels.as_outputs(_run(fn, x))
    assert len(outputs) == len(expected)
    for out, want in zip(outputs, expected, strict=True):
        assert torch.equal(out, want)


@pytest.mark.parametrize("dtype", kernels.FLOAT_TYPES, ids=str)
def test_float_tile_as_torch(dtype: torch.dtype) -> None:
    """Every op against every number, packed in one kernel."""
    x = kernels.packed_input(kernels.NUMBERS, dtype)
    _same_outputs(
        kernels.compare_kernel(kernels.NUMBERS),
        x,
        kernels.all_compares(x, kernels.NUMBERS),
    )


@pytest.mark.parametrize("op", list(kernels.OPS))
@pytest.mark.parametrize("dtype", kernels.FLOAT_TYPES, ids=str)
def test_float_tile_one_compare(dtype: torch.dtype, op: str) -> None:
    """Unpacked: fusion with the others can't be what hides a bug."""
    x = kernels.packed_input((0.1,), dtype)
    _same_outputs(
        kernels.compare_kernel((0.1,), (op,)), x, kernels.all_compares(x, (0.1,), (op,))
    )


@pytest.mark.parametrize("dtype", kernels.INT_TILES, ids=str)
def test_integer_tile_as_torch(dtype: torch.dtype) -> None:
    x = kernels.packed_input(kernels.INT_NUMBERS, dtype)
    _same_outputs(
        kernels.compare_kernel(kernels.INT_NUMBERS),
        x,
        kernels.all_compares(x, kernels.INT_NUMBERS),
    )


@pytest.mark.parametrize("op", list(kernels.OPS))
@pytest.mark.parametrize("dtype", kernels.INT_TILES, ids=str)
def test_integer_tile_one_compare(dtype: torch.dtype, op: str) -> None:
    x = kernels.packed_input((-0.5, -3), dtype)
    _same_outputs(
        kernels.compare_kernel((-0.5,), (op,)),
        x,
        kernels.all_compares(x, (-0.5,), (op,)),
    )


def test_bool_tile_as_torch() -> None:
    x = kernels.bool_input()
    _same_outputs(
        kernels.compare_kernel(kernels.BOOL_NUMBERS),
        x,
        kernels.all_compares(x, kernels.BOOL_NUMBERS),
    )


@pytest.mark.parametrize("op", list(kernels.OPS))
def test_bool_tile_one_compare(op: str) -> None:
    x = kernels.bool_input()
    _same_outputs(
        kernels.compare_kernel((0.5,), (op,)), x, kernels.all_compares(x, (0.5,), (op,))
    )


@pytest.mark.parametrize("dtype", kernels.FLOAT_TYPES, ids=str)
@pytest.mark.parametrize("name", list(kernels.COMBINED))
def test_combined_compares_as_torch(name: str, dtype: torch.dtype) -> None:
    """Boundary values, and the lane patterns of IGC's one-row ``fabs``, right in row 0 only:
    exact regressions, on one lowering."""
    fn, ref = kernels.COMBINED[name]
    kernel = _kernel(fn)
    for x in (
        kernels.boundary_input(1.0, dtype, (128, 256)),
        kernels.lane_input((128, 256), dtype, first_inside=True),
        kernels.lane_input((128, 256), dtype, first_inside=False),
    ):
        assert kernels.same(_call(kernel, x), ref(x))


@pytest.mark.parametrize("name", list(kernels.INT_ABS_KERNELS))
def test_int_abs_as_torch(name: str) -> None:
    """Exact regression: IGC's one-row ``llvm.abs.v16i32``, 30471 of 32768 wrong."""
    kernel = _kernel(kernels.INT_ABS_KERNELS[name])
    for x in (
        kernels.integer_input((128, 256)),
        (kernels.lane_input((128, 256), torch.float32, first_inside=True) * 50).to(
            torch.int32
        ),
        (kernels.lane_input((128, 256), torch.float32, first_inside=False) * 50).to(
            torch.int32
        ),
    ):
        assert torch.equal(_call(kernel, x), x.abs())


@pytest.mark.parametrize("name", list(kernels.F64_KERNELS))
def test_f64_math_narrowed(name: str) -> None:
    fn, ref = kernels.F64_KERNELS[name]
    x = kernels.boundary_input(1.0, torch.float32, (128, 256))
    assert kernels.same(_run(fn, x), ref(x))


@pytest.mark.parametrize("dtype", kernels.FLOAT_TYPES, ids=str)
def test_runtime_number_as_torch(dtype: torch.dtype) -> None:
    kernel = _kernel(kernels.compare_runtime)
    for number in (0.1, -0.1):
        x = kernels.boundary_input(number, dtype)
        assert torch.equal(_call(kernel, x, number), (x <= number).float())


@pytest.mark.parametrize("dtype", kernels.FLOAT_TYPES, ids=str)
def test_compare_with_other_users_as_torch(dtype: torch.dtype) -> None:
    x = kernels.boundary_input(0.1, dtype)
    c = x <= 0.1
    expected = torch.where(c, x, x * 0.0) + c.to(dtype)
    assert kernels.same(_run(kernels.compare_with_other_users, x), expected)


_BYTE_LOWERS = {
    "add": "t + 1",
    "rsub": "1 - t",
    "add_alpha": "torch.add(t, 1, alpha=2)",
    "and": "t & 1",
    "argmax": "torch.argmax(t, 1, keepdim=True)",
    "clamp": "torch.clamp(t, -5, 5)",
    "clamp_min": "torch.clamp(t, min=0)",
    "div": "t / 2",
    "where_abs": "torch.where(t < 0, 0 - t, t)",
    "shift_left": "t << 1",
    "shift_right": "t >> 1",
    "unsqueeze": "t.unsqueeze(0).squeeze(0)",
}
_BYTE_CRASHES = {
    "times_zero": "torch.where(t > 0, t, t * 0)",
    "times_two": "t * 2",
    "two_times": "2 * t",
    "minus_one": "t - 1",
    "floor_divide": "t // 2",
    "remainder_op": "t % 3",
    "remainder": "torch.remainder(t, 3)",
    "mul": "torch.mul(t, 2)",
    "sub": "torch.sub(t, 1)",
}


def _byte_input() -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    return torch.randint(-100, 100, (32, 32), generator=generator, dtype=torch.int8)


@pytest.mark.parametrize("name", list(_BYTE_LOWERS))
def test_i8_tile_with_an_int_that_lowers_as_torch(name: str) -> None:
    """Exact regression: an allow-list that never matched refused all of these."""
    expression = _BYTE_LOWERS[name]
    fn = kernels.expression_kernel(expression)
    x = _byte_input()
    expected = eval(expression, {"t": x, "torch": torch}).to(torch.float32)
    assert torch.equal(_kernel(fn)(x), expected.expand(x.size()))


@pytest.mark.isolated
@pytest.mark.parametrize("name", list(_BYTE_CRASHES))
def test_i8_tile_with_an_int_is_refused_not_a_crash(name: str) -> None:
    """Exact regression: SIGSEGV in torch-mlir's lowering, in this process."""
    fn = kernels.expression_kernel(_BYTE_CRASHES[name])
    with pytest.raises(
        UnsupportedOperationError, match="i8 tensor with a Python integer"
    ):
        _kernel(fn)(_byte_input())


def test_f64_tile_stays_in_f64() -> None:
    x = kernels.boundary_input(0.1, torch.float64)
    out = _run(kernels.compare_kernel((0.1,), ("le",)), x)
    assert torch.equal(out.double(), (x <= 0.1).double())
