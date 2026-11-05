"""Inputs that find wrong elements: the values where an op changes its answer, placed so
that rows and 16-column lanes differ (a kernel that broadcasts row 0 or lane 0 can't pass).

Deterministic: no seed needed, and the same tensor on every run and device.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable


def _rounded(value: float, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64).to(dtype)


def specials(dtype: torch.dtype) -> list[float]:
    """±0, ±inf, NaN, ``tiny`` and ``max`` of ``dtype``, both signs."""
    info = torch.finfo(dtype)
    tiny, largest = float(info.tiny), float(info.max)
    return [
        0.0,
        -0.0,
        math.inf,
        -math.inf,
        math.nan,
        tiny,
        -tiny,
        largest,
        -largest,
    ]


def boundary(constants: Iterable[float], dtype: torch.dtype) -> list[float]:
    """For each constant its value rounded to ``dtype``, the numbers of ``dtype`` next to
    it, their negations; and :func:`specials`. All are values of ``dtype``, returned in
    f64, which holds each exactly, without duplicates (by bits, so -0.0 stays)."""
    values = specials(dtype)
    for constant in constants:
        at = _rounded(constant, dtype)
        below = torch.nextafter(at, _rounded(-math.inf, dtype))
        above = torch.nextafter(at, _rounded(math.inf, dtype))
        for neighbour in (at, below, above):
            number = neighbour.double().item()
            values += [number, -number]
    seen: set[bytes] = set()
    distinct = []
    for value in values:
        bits = torch.tensor(value, dtype=torch.float64).numpy().tobytes()
        if bits not in seen:
            seen.add(bits)
            distinct.append(value)
    return distinct


def lanes(
    values: list[float],
    *shape: int,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """``values`` over ``shape``: each column steps through them, and each row starts one
    further on, so a row (or, across rows, a column) of at least ``len(values)`` elements
    has them all, and no two neighbouring rows or columns, or columns 16 apart, are alike
    (rows ``len(values)`` apart are).
    The last two dims place them; earlier dims repeat them, shifted again."""
    values = list(values)
    while 16 % len(values) == 0:  # Columns 16 apart would be alike: one more value.
        values.append(max(abs(v) for v in values if math.isfinite(v)) * 2.0 + 1.0)
    count = len(values)
    grids = torch.meshgrid(*(torch.arange(size) for size in shape), indexing="ij")
    picks = sum(grids, torch.zeros(shape, dtype=torch.long)) % count
    return torch.tensor(values, dtype=torch.float64)[picks].to(dtype).to(device)


def boundary_tile(
    constants: Iterable[float],
    *shape: int,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """:func:`boundary` placed by :func:`lanes`."""
    return lanes(boundary(constants, dtype), *shape, dtype=dtype, device=device)


def same_values(out: torch.Tensor, ref: torch.Tensor) -> None:
    """``out`` is ``ref`` bit for bit, in its dtype: NaN by mask, signed zero by bits. The
    failure says how many elements differ, and the first ones."""
    out, ref = out.cpu(), ref.cpu()
    assert out.dtype == ref.dtype and out.shape == ref.shape
    nan = torch.isnan(ref.float())
    integer = {2: torch.int16, 4: torch.int32, 8: torch.int64}[out.element_size()]
    wrong = (out.view(integer) != ref.view(integer)) & ~nan | (
        torch.isnan(out.float()) != nan
    )
    if wrong.any():
        first = wrong.nonzero()[:4]
        shown = [
            (at.tolist(), out[tuple(at)].item(), ref[tuple(at)].item()) for at in first
        ]
        raise AssertionError(
            f"{int(wrong.sum())} of {wrong.numel()} elements differ; "
            f"first (index, out, ref): {shown}"
        )


def close_values(out: torch.Tensor, ref: torch.Tensor, bound: float) -> None:
    """The same NaNs and infinities, by sign; the finite elements' error at most ``bound``,
    for rounding ops: per element, relative to ``|ref|``, or absolute below 1. Inputs here
    reach ``max``, so one scale from ``max|ref|`` would hide every other element's error."""
    assert out.dtype != torch.bfloat16 and ref.dtype != torch.bfloat16, (
        "element-wise error misleads on bf16's rounding: use the error normalized by max|ref|"
    )
    out, ref = out.cpu().double(), ref.cpu().double()
    assert out.shape == ref.shape
    same_special = (out.isnan() == ref.isnan()) & (out.isposinf() == ref.isposinf())
    same_special &= out.isneginf() == ref.isneginf()
    if not same_special.all():
        at = (~same_special).nonzero()[:4]
        shown = [(a.tolist(), out[tuple(a)].item(), ref[tuple(a)].item()) for a in at]
        raise AssertionError(
            f"{int((~same_special).sum())} of {out.numel()} NaN or infinity elements "
            f"differ; first (index, out, ref): {shown}"
        )
    finite = ref.isfinite()
    if finite.any():
        scale = ref[finite].abs().clamp_min(1.0)
        error = ((out[finite] - ref[finite]).abs() / scale).amax().item()
        assert error <= bound, f"error {error:.3g} over {bound:.3g}"
