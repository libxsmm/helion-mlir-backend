"""``tests/inputs.py``: the values it promises, and where it puts them."""

from __future__ import annotations

import math

import pytest
import torch

from tests import inputs

DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_boundary_has_the_constant_in_the_tiles_type_and_its_neighbours(
    dtype: torch.dtype,
) -> None:
    values = inputs.boundary([0.1], dtype)
    at = torch.tensor(0.1, dtype=torch.float64).to(dtype)
    up = torch.nextafter(at, torch.tensor(math.inf, dtype=dtype))
    down = torch.nextafter(at, torch.tensor(-math.inf, dtype=dtype))
    for number in (at, up, down, -at, -up, -down):
        assert number.double().item() in values
    assert any(math.isnan(value) for value in values)
    # Each is a value of the type: rounding to it changes nothing.
    for value in values:
        if not math.isnan(value):
            assert torch.tensor(value, dtype=torch.float64).to(dtype).item() == value


def test_boundary_keeps_both_zeros_once_each() -> None:
    zeros = [v for v in inputs.boundary([0.0], torch.float32) if v == 0.0]
    assert sorted(math.copysign(1.0, zero) for zero in zeros) == [-1.0, 1.0]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", [(64, 64), (32, 256), (17, 33)])
def test_rows_and_lanes_differ_and_every_value_is_placed(
    dtype: torch.dtype, shape: tuple[int, int]
) -> None:
    x = inputs.boundary_tile([0.5, 1.0], *shape, dtype=dtype)
    assert x.shape == shape and x.dtype == dtype
    bits = x.view({2: torch.int16, 4: torch.int32}[x.element_size()])
    for row in range(1, shape[0]):
        assert not bits[row].equal(bits[row - 1])
    for column in (1, 16):
        if column < shape[1]:
            assert not bits[:, column].equal(bits[:, 0])
    values = inputs.boundary([0.5, 1.0], dtype)
    if shape[1] >= len(values):
        assert len(bits[0].unique()) == len(values)


@pytest.mark.parametrize("count", [1, 2, 4, 8, 16, 32])
def test_columns_16_apart_differ_for_any_number_of_values(count: int) -> None:
    x = inputs.lanes([float(i + 10) for i in range(count)], 4, 64, dtype=torch.float32)
    assert not x[:, 16].equal(x[:, 0])


def test_same_values_compares_bits() -> None:
    x = torch.tensor([0.0, math.nan, 1.0], dtype=torch.float32)
    inputs.same_values(x, x.clone())
    with pytest.raises(AssertionError, match="1 of 3 elements differ"):
        inputs.same_values(x, torch.tensor([-0.0, math.nan, 1.0]))
    with pytest.raises(AssertionError, match="1 of 3 elements differ"):
        inputs.same_values(x, torch.tensor([0.0, 2.0, 1.0]))


def test_close_values_refuses_bf16() -> None:
    x = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="normalized by max"):
        inputs.close_values(x, x, 1e-5)
