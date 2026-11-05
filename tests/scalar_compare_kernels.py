"""Kernels of compares with a Python number, and inputs with elements at the number."""

from __future__ import annotations

import importlib.util
import math
import operator
from pathlib import Path
import sys
import tempfile
from typing import TYPE_CHECKING

import helion.language as hl
import numpy
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

F32 = torch.float32
FLOAT_TYPES = (torch.float32, torch.bfloat16, torch.float16)
NUMBERS = (0.1, -0.1, 0.5, 1.0, 1e-3, 3.3, 1e30, 1, -3, 0)
INT_NUMBERS = (
    0.5,
    -0.5,
    2.5,
    1.0,
    16777217.0,
    1e-50,
    -1e-50,
    1,
    0,
    -3,
    200,
    -200,
    70000,
    2**40,
)
BOOL_NUMBERS = (0.5, -0.5, 1.0, 1e-50, -1e-50, 2.5, 0, 1, 2, -3)
INT_TILES = (torch.int32, torch.int64, torch.int16, torch.int8)
OPS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_BITS_PER_OUTPUT = 24
"""An f32 holds this many bits exactly."""
_SOURCE = """
import torch
import helion.language as hl


def kernel(x):
{empties}
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
{stores}
    return {outputs}
"""


def _triples(
    numbers: Sequence[object], ops: Sequence[str]
) -> list[list[tuple[str, object]]]:
    pairs = [(op, number) for number in numbers for op in ops]
    return [
        pairs[start : start + _BITS_PER_OUTPUT]
        for start in range(0, len(pairs), _BITS_PER_OUTPUT)
    ]


def compare_kernel(
    numbers: Sequence[object], ops: Sequence[str] = tuple(OPS)
) -> Callable[..., object]:
    """Every op against every number in one kernel, so one lowering covers them: a bit each in
    f32 outputs. In its source, for the numbers to be constants to the lowering (Helion takes
    no closures). See :func:`packed_input` for its input, :func:`all_compares` for its result."""
    groups = _triples(numbers, ops)
    empties = "\n".join(
        f"    o{i} = torch.empty(x.size(), dtype=torch.float32, device=x.device)"
        for i in range(len(groups))
    )
    stores = "\n".join(
        f"        o{i}[tm, tn] = "
        + " + ".join(
            f"(t {OPS[op]} {number!r}).to(torch.float32) * {float(1 << bit)!r}"
            for bit, (op, number) in enumerate(group)
        )
        for i, group in enumerate(groups)
    )
    source = _SOURCE.format(
        empties=empties,
        stores=stores,
        outputs=", ".join(f"o{i}" for i in range(len(groups))),
    )
    return _load(source)


def _load(source: str) -> Callable[..., object]:
    directory = Path(tempfile.mkdtemp(prefix="scalar_compare_"))
    path = directory / f"compare_{abs(hash(source))}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module.kernel


def expression_kernel(expression: str) -> Callable[..., object]:
    """``expression`` of ``t``, a whole tile, on a one-tile grid, as f32."""
    source = (
        "import torch\nimport helion.language as hl\n\n\ndef kernel(x):\n"
        "    o = torch.empty(x.size(), dtype=torch.float32, device=x.device)\n"
        "    for tm, tn in hl.tile(x.size()):\n"
        "        t = x[tm, tn]\n"
        f"        o[tm, tn] = ({expression}).to(torch.float32)\n"
        "    return o\n"
    )
    return _load(source)


def all_compares(
    x: torch.Tensor, numbers: Sequence[object], ops: Sequence[str] = tuple(OPS)
) -> list[torch.Tensor]:
    """What :func:`compare_kernel` gives, by torch, in ``x``'s type."""
    results = []
    for group in _triples(numbers, ops):
        total = torch.zeros(x.size(), dtype=torch.float32)
        for bit, (op, number) in enumerate(group):
            total += _COMPARES[op](x, number).float() * float(1 << bit)
        results.append(total)
    return results


def as_outputs(result: object) -> list[torch.Tensor]:
    return list(result) if isinstance(result, tuple | list) else [result]


_COMPARES = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def compare_runtime(x: torch.Tensor, number: float) -> torch.Tensor:
    out = torch.empty(x.size(), dtype=F32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn] <= number).to(F32)
    return out


def compare_with_other_users(x: torch.Tensor) -> torch.Tensor:
    """One compare, as a cast and as a condition."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        c = t <= 0.1
        out[tm, tn] = torch.where(c, t, t * 0.0) + c.to(x.dtype)
    return out


def either_side(x: torch.Tensor) -> torch.Tensor:
    """Exact kernel of the review: 9830 of 32768 wrong on the XPU."""
    out = torch.empty(x.size(), dtype=F32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        out[tm, tn] = ((t > 1.0) | (t < -1.0)).to(F32)
    return out


def either_side_where(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        out[tm, tn] = torch.where((t > 1.0) | (t < -1.0), t * 0.0 + 1.0, t * 0.0)
    return out


def between(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(), dtype=F32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        out[tm, tn] = ((t <= 1.0) & (t >= -1.0)).to(F32)
    return out


def between_where(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        out[tm, tn] = torch.where((t <= 1.0) & (t >= -1.0), t * 0.0 + 1.0, t * 0.0)
    return out


def one_side_twice(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(), dtype=F32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        out[tm, tn] = ((t > 1.0) | (t > 2.0)).to(F32)
    return out


def abs_through_f64(x: torch.Tensor) -> torch.Tensor:
    """Exact kernel of the review: 16384 of 32768 wrong on the XPU."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = torch.abs(x[tm, tn].to(torch.float64)).to(F32)
    return out


def abs_of_computed_f64(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = torch.abs(x[tm, tn].to(torch.float64) * 2.0 - 1.0).to(F32)
    return out


def range_check_in_f64(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn].to(torch.float64)
        out[tm, tn] = ((t > 1.0) | (t < -1.0)).to(F32)
    return out


def both_sides_by_chain(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        one = t * 0.0 + 1.0
        out[tm, tn] = torch.where(t > 1.0, one, torch.where(t < -1.0, one, one * 0.0))
    return out


def both_sides_by_chain_mirror(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        one = t * 0.0 + 1.0
        out[tm, tn] = torch.where(
            t > 1.0, one * 0.0, torch.where(t < -1.0, one * 0.0, one)
        )
    return out


def not_either(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(), dtype=F32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        out[tm, tn] = ((~(t > 1.0)) & (~(t < -1.0))).to(F32)
    return out


def not_both(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(), dtype=F32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        out[tm, tn] = ((~(t <= 1.0)) | (~(t >= -1.0))).to(F32)
    return out


def inside_or(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(), dtype=F32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        out[tm, tn] = ((t < 1.0) | (t > -1.0)).to(F32)
    return out


def where_int_abs(x: torch.Tensor) -> torch.Tensor:
    """``llvm.abs.v16i32`` in IGC: 30471 of 32768 wrong on the XPU."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        out[tm, tn] = torch.where(t < 0, 0 - t, t)
    return out


def where_int_abs_mirror(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn]
        out[tm, tn] = torch.where(t > 0, t, 0 - t)
    return out


def sqrt_of_abs_f64(x: torch.Tensor) -> torch.Tensor:
    """Scalar ``fptrunc`` of f64 after the ``sqrt`` builtin: 16k of 32768 wrong."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = torch.sqrt(torch.abs(x[tm, tn].to(torch.float64))).to(F32)
    return out


def sqrt_of_square_f64(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        t = x[tm, tn].to(torch.float64)
        out[tm, tn] = torch.sqrt(t * t).to(F32)
    return out


COMBINED: dict[str, tuple[Callable, Callable]] = {
    "by_chain": (
        both_sides_by_chain,
        lambda x: torch.where((x > 1.0) | (x < -1.0), x * 0.0 + 1.0, x * 0.0),
    ),
    "by_chain_mirror": (
        both_sides_by_chain_mirror,
        lambda x: torch.where((x > 1.0) | (x < -1.0), x * 0.0, x * 0.0 + 1.0),
    ),
    "not_either": (not_either, lambda x: ((~(x > 1.0)) & (~(x < -1.0))).float()),
    "not_both": (not_both, lambda x: ((~(x <= 1.0)) | (~(x >= -1.0))).float()),
    "inside_or": (inside_or, lambda x: ((x < 1.0) | (x > -1.0)).float()),
    "either_side": (either_side, lambda x: ((x > 1.0) | (x < -1.0)).float()),
    "either_side_where": (
        either_side_where,
        lambda x: torch.where((x > 1.0) | (x < -1.0), x * 0.0 + 1.0, x * 0.0),
    ),
    "between": (between, lambda x: ((x <= 1.0) & (x >= -1.0)).float()),
    "between_where": (
        between_where,
        lambda x: torch.where((x <= 1.0) & (x >= -1.0), x * 0.0 + 1.0, x * 0.0),
    ),
    "one_side_twice": (one_side_twice, lambda x: ((x > 1.0) | (x > 2.0)).float()),
}


def packed_input(
    numbers: Sequence[object], dtype: torch.dtype, shape: tuple[int, int] = (64, 64)
) -> torch.Tensor:
    """Each element drawn at random from the numbers in the tile's type, their negatives and
    doubles, zero, NaN and infinities where the type has them, so rows and 16-column lanes all
    differ: a broadcast of row 0 or lane 0 can't pass."""
    pool: list[float] = [0.0]
    for number in numbers:
        at = torch.tensor(number, dtype=torch.float64).to(dtype)
        pool += [at.double().item(), -at.double().item()]
        if dtype.is_floating_point:
            above = torch.nextafter(
                at.double(), torch.tensor(math.inf, dtype=torch.float64)
            )
            pool += [above.to(dtype).double().item(), at.double().item() * 2.0]
        else:
            pool += [at.double().item() + 1.0, at.double().item() - 1.0]
    if dtype.is_floating_point:
        pool += [math.nan, math.inf, -math.inf]
    generator = torch.Generator().manual_seed(13)
    picks = torch.randint(0, len(pool), shape, generator=generator)
    return torch.tensor(pool, dtype=torch.float64)[picks].to(dtype)


def boundary_input(number: float, dtype: torch.dtype, shape=(64, 64)) -> torch.Tensor:
    """Random, with every 4th element the number in ``dtype`` and its neighbours in it, its
    negative, zero and values either side in f64."""
    generator = torch.Generator().manual_seed(7)
    x = (torch.randn(*shape, generator=generator).double() * max(abs(number), 1.0)).to(
        dtype
    )
    at = torch.tensor(number, dtype=dtype)
    above = torch.nextafter(
        at.double(), torch.tensor(math.inf, dtype=torch.float64)
    ).to(dtype)
    fill = [
        at,
        -at,
        above,
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(math.nan, dtype=dtype),
        torch.tensor(math.inf, dtype=dtype),
        torch.tensor(-math.inf, dtype=dtype),
    ]
    for index, value in enumerate(fill):
        x[index::8, ::4] = value
    return x


def integer_input(shape=(64, 64), dtype: torch.dtype = torch.int32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    x = (torch.randn(*shape, generator=generator) * 3).to(dtype)
    if dtype in (torch.int32, torch.int64):
        x[::8, ::4] = 16777217
        x[1::8, ::4] = 16777216
    x[2::8, ::4] = -1
    x[3::8, ::4] = 0
    x[4::8, ::4] = 100
    x[5::8, ::4] = -56
    return x


def lane_input(
    shape: tuple[int, int], dtype: torch.dtype, *, first_inside: bool
) -> torch.Tensor:
    """Row 0 of each 16 rows and column 0 of each 16 columns on one side of the range check
    ``|x| > 1``, the rest on the other, with signs mixed: a compare that sees one row of a
    lane's vector, or one lane of a row, gets the rest wrong."""
    generator = torch.Generator().manual_seed(11)
    inside = torch.rand(*shape, generator=generator) * 1.6 - 0.8
    outside = (torch.rand(*shape, generator=generator) + 1.5) * torch.sign(
        torch.rand(*shape, generator=generator) - 0.5
    )
    first = torch.zeros(shape, dtype=torch.bool)
    first[::16, :] = True
    first[:, ::16] = True
    if first_inside:
        x = torch.where(first, inside, outside)
    else:
        x = torch.where(first, outside, inside)
    return x.to(dtype)


def bool_input(shape=(64, 64)) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(*shape, generator=generator) > 0.0


INT_ABS_KERNELS: dict[str, Callable] = {
    "where": where_int_abs,
    "where_mirror": where_int_abs_mirror,
}


def _exact_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Not ``torch.sqrt``: MKL's threaded first call can be wrong in one chunk
    (agent-wiki/pytorch/mkl-sqrt-first-call.md)."""
    return torch.from_numpy(numpy.sqrt(x.cpu().numpy()))


F64_KERNELS: dict[str, tuple[Callable, Callable]] = {
    "sqrt_of_abs": (sqrt_of_abs_f64, lambda x: _exact_sqrt(x.double().abs()).float()),
    "sqrt_of_square": (
        sqrt_of_square_f64,
        lambda x: _exact_sqrt(x.double() * x.double()).float(),
    ),
    "abs_through_f64": (abs_through_f64, lambda x: x.abs()),
    "abs_of_computed_f64": (
        abs_of_computed_f64,
        lambda x: (x.double() * 2 - 1).abs().float(),
    ),
    "range_check_in_f64": (
        range_check_in_f64,
        lambda x: ((x > 1.0) | (x < -1.0)).float(),
    ),
}


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """The same elements, with NaN equal to NaN."""
    return torch.equal(a.isnan(), b.isnan()) and torch.equal(
        a.nan_to_num(nan=0.0, posinf=math.inf, neginf=-math.inf),
        b.nan_to_num(nan=0.0, posinf=math.inf, neginf=-math.inf),
    )
