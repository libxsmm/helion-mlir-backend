"""``print`` in a kernel (``hl.device_print``): the printed values, the kernel's
result unchanged, the refusals, and no change to a kernel without a print."""

from __future__ import annotations

import contextlib
import ctypes
import os
import re
import tempfile
from typing import TYPE_CHECKING
import warnings

import helion
from helion._compiler.backend_registry import get_backend_class
import helion.language as hl
from mlir import ir
import pytest
import torch

from helion_mlir_backend import api
from helion_mlir_backend import lower_mlir
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError
from helion_mlir_backend._compiler.mlir_transforms.prints import lower_prints_to_libc

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

BLOCKS_2D = [32, 32]
BLOCKS_GEMM = [32, 32, 32]


def _kernel(fn: Callable, blocks: list[int]) -> helion.Kernel:
    return helion.kernel(
        fn, backend="mlir", static_shapes=True, config=helion.Config(block_sizes=blocks)
    )


@contextlib.contextmanager
def _stdout_lines() -> Iterator[list[str]]:
    """The lines written to fd 1 inside the block, by C or the driver, in the list after it.
    Works in an isolated child, whose fds aren't the ones pytest's fixtures capture."""
    lines: list[str] = []
    flush = ctypes.CDLL(None).fflush
    flush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile("w+") as file:
        os.dup2(file.fileno(), 1)
        try:
            yield lines
        finally:
            flush(None)
            os.dup2(saved, 1)
            os.close(saved)
        file.seek(0)
        lines.extend(file.read().splitlines())


def _elements(lines: list[str], label: str) -> list[tuple[tuple[int, ...], float]]:
    """``label[i, j]: v`` lines; every line must be whole: threads must not split one."""
    elements = []
    for line in lines:
        match = re.fullmatch(rf"{re.escape(label)}\[([\d, ]+)\]: (-?[\d.]+)", line)
        assert match, f"not a whole line: {line!r}"
        where = tuple(int(i) for i in match[1].split(", "))
        elements.append((where, float(match[2])))
    return elements


def _check_positions(
    elements: list[tuple[tuple[int, ...], float]],
    expected: torch.Tensor,
    block: int,
    *,
    calls: int = 1,
) -> None:
    """Each position of a tile holds the values of that position in all the tiles, so a
    swapped index or a value printed for the wrong element shows: tiles run in any order,
    which is all a sorted comparison can't tell."""
    printed: dict[tuple[int, ...], list[float]] = {}
    for where, value in elements:
        printed.setdefault(where, []).append(value)
    positions = {(i, j) for i in range(block) for j in range(block)}
    assert set(printed) == positions
    tiles = torch.tensor(
        expected.double().unfold(0, block, block).unfold(1, block, block)
    )
    for (i, j), values in printed.items():
        want = sorted(tiles[..., i, j].flatten().tolist() * calls)
        assert sorted(values) == pytest.approx(want, abs=1e-6)


def _normalized_error(out: torch.Tensor, reference: torch.Tensor) -> float:
    reference = reference.double()
    error = (out.double().cpu() - reference).abs().max() / reference.abs().max()
    return error.item()


def _x(size: int = 64) -> torch.Tensor:
    return torch.rand(size, size, generator=torch.Generator().manual_seed(1))


def _ab(size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    a = torch.rand(size, size, generator=generator, dtype=torch.float64)
    b = torch.rand(size, size, generator=generator, dtype=torch.float64)
    return a.to(torch.bfloat16), b.to(torch.bfloat16)


def scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def scale_label(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        print("50% tile é")
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def scale_tile(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn] * 2.0
        print("v", v)
        out[tm, tn] = v
    return out


def two_tensors(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn]
        print("both", v, v > 0.5)
        out[tm, tn] = v
    return out


def row_sums(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn]
        print("sum", v.sum(0).sum(0))
        out[tm, tn] = v
    return out


def int_tile(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn]
        print("i", v)
        out[tm, tn] = v
    return out


def gemm_k_label(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            print("k")
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def gemm_input(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            print("in", a[tm, tk])
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def gemm_k_label_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """No register-tiled consumer of the K loop: the label prints once per K tile."""
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            print("k")
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc
    return out


def in_host_code(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.device_print("start", x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn]
    return out


def full_print(x: torch.Tensor) -> torch.Tensor:
    """The review's kernel, exact, printing wrong values on the B50: 11549 of 16384."""
    out = torch.full_like(x, 7.0)
    for tm, tn in hl.tile(x.size()):
        print("f", out[tm, tn])
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def l2_plain(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n, _ = w.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = torch.relu((y - 0.25) * 2.0).to(x.dtype)
    return out


def l2_k_acc(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n, _ = w.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], w[tn, tk].T)
            print("acc", acc)
        y = acc + bias[tn]
        out[tm, tn] = torch.relu((y - 0.25) * 2.0).to(x.dtype)
    return out


def two_labels(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        print("one")
        print("two")
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def host_print(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    print("host", "only")
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def position(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        print("at", tm.begin)
        out[tm, tn] = x[tm, tn]
    return out


def position_tensor(x: torch.Tensor) -> torch.Tensor:
    """A tile position, which Helion's rule takes as a 0-d tensor (a computed one)."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        print("at", torch.zeros([], dtype=torch.int64, device=x.device) + tm.begin)
        out[tm, tn] = x[tm, tn]
    return out


# --- The CPU target ----------------------------------------------------------------


def test_label_once_per_tile_and_result_unchanged() -> None:
    x = _x()
    with _stdout_lines() as lines:
        out = _kernel(scale_label, BLOCKS_2D)(x)
    assert lines == ["50% tile é"] * 4
    assert torch.equal(out, _kernel(scale, BLOCKS_2D)(x))


def test_tile_prints_its_values_on_whole_lines() -> None:
    x = _x()
    with _stdout_lines() as lines:
        out = _kernel(scale_tile, BLOCKS_2D)(x)
    elements = _elements(lines, "v")
    assert len(elements) == x.numel()
    _check_positions(elements, x * 2, 32)
    assert torch.equal(out, x * 2)


def test_each_tensor_prints_bool_as_zero_or_one() -> None:
    x = _x()
    with _stdout_lines() as lines:
        _kernel(two_tensors, BLOCKS_2D)(x)
    elements = _elements(lines, "both")
    assert len(elements) == 2 * x.numel()
    values = [value for _, value in elements]
    integers = [v for line, v in zip(lines, values, strict=True) if "." not in line]
    assert len(integers) == x.numel()
    assert sum(integers) == (x > 0.5).sum().item()
    assert set(integers) == {0.0, 1.0}


def test_integer_tile() -> None:
    x = (_x() * 1000).to(torch.int32) - 500
    with _stdout_lines() as lines:
        _kernel(int_tile, BLOCKS_2D)(x)
    printed = sorted(int(value) for _, value in _elements(lines, "i"))
    assert printed == sorted(x.flatten().tolist())


def test_reduced_scalar() -> None:
    x = _x()
    with _stdout_lines() as lines:
        _kernel(row_sums, BLOCKS_2D)(x)
    printed = sorted(float(line.split()[1]) for line in lines)
    tiles = [
        x[i : i + 32, j : j + 32].double().sum().item()
        for i in (0, 32)
        for j in (0, 32)
    ]
    assert printed == pytest.approx(sorted(tiles), rel=1e-5)


def test_scalar_pipeline() -> None:
    x = _x()
    kernel = helion.kernel(
        scale_tile,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=BLOCKS_2D, mlir_pipeline="scalar"),
    )
    with _stdout_lines() as lines:
        out = kernel(x)
    assert len(_elements(lines, "v")) == x.numel()
    assert torch.equal(out, x * 2)


def test_label_once_per_k_tile() -> None:
    a, b = _ab()
    with _stdout_lines() as lines:
        out = _kernel(gemm_k_label_f32, BLOCKS_GEMM)(a, b)
    assert lines == ["k"] * (4 * 2)
    assert _normalized_error(out, a.double() @ b.double()) < 2**-7


def test_k_loop_may_repeat_but_the_result_stands() -> None:
    """A documented limit: fusion can copy a K loop with a print for a register-tiled
    consumer, so its lines repeat. Whole repeats only, and a right result."""
    a, b = _ab()
    with _stdout_lines() as lines:
        out = _kernel(gemm_k_label, BLOCKS_GEMM)(a, b)
    assert lines
    assert set(lines) == {"k"}
    assert len(lines) % (4 * 2) == 0
    assert _normalized_error(out.to(torch.float32), a.double() @ b.double()) < 2**-7


def test_loaded_tile_in_the_k_loop() -> None:
    a, b = _ab()
    with _stdout_lines() as lines:
        out = _kernel(gemm_input, BLOCKS_GEMM)(a, b)
    printed = {value for _, value in _elements(lines, "in")}
    assert printed <= {round(value, 6) for value in a.double().flatten().tolist()}
    assert len(printed) > 1
    assert _normalized_error(out.to(torch.float32), a.double() @ b.double()) < 2**-7


def test_print_in_a_k_loop_can_change_the_path_within_rounding() -> None:
    """The review's kernel, exact: this print in the K loop took an L2 from FMAs to AMX at
    these tiles on AMX hosts, changing a bf16 unit on some inputs, so its result can't be
    compared bit for bit; against fp64 it stays right."""
    torch.manual_seed(0)
    x, w = (
        torch.rand(128, 128, dtype=torch.float64).to(torch.bfloat16) for _ in range(2)
    )
    bias = torch.rand(128, dtype=torch.float64).to(torch.bfloat16)
    reference = torch.relu(((x.double() @ w.double().T + bias.double()) - 0.25) * 2.0)
    with _stdout_lines() as lines:
        out = _kernel(l2_k_acc, BLOCKS_GEMM)(x, w, bias)
    assert lines
    assert _normalized_error(out.to(torch.float32), reference) < 2**-7
    plain = _kernel(l2_plain, BLOCKS_GEMM)(x, w, bias)
    assert _normalized_error(plain.to(torch.float32), reference) < 2**-7


def test_cpu_prints_a_stored_buffer_before_its_store() -> None:
    """Tiles don't overlap and each thread runs a whole tile: right on the CPU."""
    x = _x()
    with _stdout_lines() as lines:
        out = _kernel(full_print, BLOCKS_2D)(x)
    assert {value for _, value in _elements(lines, "f")} == {7.0}
    assert torch.equal(out, x * 2)


def test_position_through_a_tensor_on_the_cpu() -> None:
    x = _x()
    with _stdout_lines() as lines:
        _kernel(position_tensor, BLOCKS_2D)(x)
    assert sorted(lines) == ["at 0", "at 0", "at 32", "at 32"]


# --- Refusals ----------------------------------------------------------------------


@pytest.mark.helion_internals
def test_refused_in_host_code() -> None:
    with pytest.raises(UnsupportedOperationError, match=r"hl\.device_print"):
        _kernel(in_host_code, BLOCKS_2D)(_x())


@pytest.mark.helion_internals
def test_refused_non_tensor_argument() -> None:
    """Helion's rule, untouched: this backend doesn't override it."""
    with pytest.raises(Exception, match="only supports runtime tensor values"):
        lower_mlir(_kernel(position, BLOCKS_2D), [_x()], target="cpu")


STORED_BUFFER_WARNING = "may show wrong values"


def test_cpu_has_no_such_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower_mlir(_kernel(full_print, BLOCKS_2D), [_x()], target="cpu")
    assert not [w for w in caught if STORED_BUFFER_WARNING in str(w.message)]


def test_cpu_lowers_a_print_before_the_store() -> None:
    text = str(lower_mlir(_kernel(full_print, BLOCKS_2D), [_x()], target="cpu"))
    assert "@printf" in text


# --- The experimental warning ------------------------------------------------------


EXPERIMENTAL = "(hl.device_print) is experimental"


def _experimental(recwarn: pytest.WarningsRecorder) -> list[str]:
    return [str(w.message) for w in recwarn.list if EXPERIMENTAL in str(w.message)]


def _lower_for(target: str, fn: Callable) -> None:
    lower_mlir(_kernel(fn, BLOCKS_2D), [_x()], target=target)


@pytest.mark.parametrize("target", ["cpu"])
def test_experimental_warning_once_per_compile(
    target: str, recwarn: pytest.WarningsRecorder
) -> None:
    warnings.simplefilter("always")
    _lower_for(target, two_labels)
    messages = _experimental(recwarn)
    assert len(messages) == 1
    assert "two_labels" in messages[0]
    assert "hl.device_print" in messages[0]
    assert "'Printing in a kernel'" in messages[0]


@pytest.mark.parametrize("target", ["cpu"])
def test_no_experimental_warning_without_a_print(
    target: str, recwarn: pytest.WarningsRecorder
) -> None:
    warnings.simplefilter("always")
    _lower_for(target, scale)
    assert _experimental(recwarn) == []


def test_host_print_is_no_kernel_print(recwarn: pytest.WarningsRecorder) -> None:
    warnings.simplefilter("always")
    _lower_for("cpu", host_print)
    assert _experimental(recwarn) == []


@pytest.mark.parametrize("target", ["cpu"])
def test_every_compile_warns_under_the_default_filter(
    target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python's default filter shows a text once per location: a plain ``warnings.warn``
    from one line stays silent for a second compile of the same kernel. No lowering process:
    forking one changes the filters, which happens to reset that memory."""
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        _lower_for(target, two_labels)
        _lower_for(target, two_labels)
    assert len([w for w in caught if EXPERIMENTAL in str(w.message)]) == 2


# --- A kernel without a print ------------------------------------------------------


def _generated(fn: Callable, blocks: list[int], x: torch.Tensor) -> str:
    """The module before any pipeline stage."""
    hf, config, env = api._compile(_kernel(fn, blocks), [x], None)
    with env:
        module = get_backend_class("mlir")().generate_mlir(hf, config, env)
    return str(module)


def test_generated_module_differs_by_the_print_alone() -> None:
    """A label emits one op and nothing else, so a kernel without one is untouched."""

    def k_printing(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tm, tn in hl.tile(x.size()):
            print("tile")
            out[tm, tn] = x[tm, tn] * 2.0
        return out

    def k_plain(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tm, tn in hl.tile(x.size()):
            out[tm, tn] = x[tm, tn] * 2.0
        return out

    x = _x()
    printing = _generated(k_printing, BLOCKS_2D, x)
    assert printing.count("gpu.printf") == 1
    without = "".join(
        line for line in printing.splitlines(True) if "gpu.printf" not in line
    )
    assert without.replace("k_printing", "k_plain") == _generated(k_plain, BLOCKS_2D, x)


NO_PRINT = """
module {
  func.func @f(%x: index) -> index { return %x : index }
}
"""

TWO_PRINTS = """
module {
  func.func @f(%x: f32, %y: i64) {
    gpu.printf "a %f\\n", %x : f32
    gpu.printf "a %f\\n", %x : f32
    gpu.printf "b %ld\\n", %y : i64
    return
  }
}
"""


@pytest.fixture
def _context():
    with ir.Context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        yield context


@pytest.mark.usefixtures("_context")
@pytest.mark.parametrize("rewrite", [lower_prints_to_libc])
def test_rewrites_leave_a_module_without_a_print_alone(rewrite: Callable) -> None:
    module = ir.Module.parse(NO_PRINT)
    before = str(module)
    assert rewrite(module) == 0
    assert str(module) == before


@pytest.mark.usefixtures("_context")
def test_libc_calls_share_strings_and_pass_doubles() -> None:
    module = ir.Module.parse(TWO_PRINTS)
    assert lower_prints_to_libc(module) == 3
    module.operation.verify()
    text = str(module)
    assert "gpu.printf" not in text
    assert text.count("llvm.mlir.global") == 2
    assert text.count("llvm.call @printf") == 3
    assert text.count("llvm.call @fflush") == 3
    assert text.count("arith.extf %arg0 : f32 to f64") == 2


# --- On the device -----------------------------------------------------------------
