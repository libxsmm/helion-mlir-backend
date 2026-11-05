"""Running values over a partial tile, against float64: a kernel's result doesn't
depend on its block sizes. Regression: the padding of the last tile entered values
an ``scf.for`` carries, which no mask reaches outside the loop. KernelBench's
softmax form had every row wrong (8 % at 128x252)."""

from __future__ import annotations

import subprocess
import sys

import helion
import helion.language as hl
import pytest
import torch

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir.support.errors import UnsupportedOperationError


def softmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.exp(x[tile_m, tile_n] - row_top)
        inv = 1.0 / total.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = torch.exp(x[tile_m, tile_n] - row_top) * inv
    return out


def _rows(x: torch.Tensor) -> tuple[int, int, torch.Tensor]:
    m, n = x.shape
    return m, n, torch.empty([m], dtype=torch.float32, device=x.device)


def running_min(x: torch.Tensor) -> torch.Tensor:
    m, n, out = _rows(x)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        low = hl.full([tile_m, block_n], float("inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            low = torch.minimum(low, x[tile_m, tile_n])
        out[tile_m] = torch.amin(low, dim=-1)
    return out


def running_product(x: torch.Tensor) -> torch.Tensor:
    m, n, out = _rows(x)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        product = hl.full([tile_m, block_n], 1.0, dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            product = product * x[tile_m, tile_n]
        out[tile_m] = product.sum(-1)
    return out


def running_sum(x: torch.Tensor) -> torch.Tensor:
    m, n, out = _rows(x)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + x[tile_m, tile_n] * 2.0 + 1.0
        out[tile_m] = total.sum(-1)
    return out


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[64, 64]),
    )(fn)


def _negative(*shape: int) -> torch.Tensor:
    return -torch.randn(*shape).abs() - 1.0


def _chunks(x: torch.Tensor, block: int) -> list[torch.Tensor]:
    return list(x.split(block, dim=1))


def _chunked_product(x: torch.Tensor) -> torch.Tensor:
    """Elements at one position of each column chunk multiply, chunks don't pad."""
    x = x.double()
    product = torch.ones(x.shape[0], 64, dtype=torch.float64)
    for chunk in _chunks(x, 64):
        product[:, : chunk.shape[1]] *= chunk
    return product.sum(-1)


FORMS = {
    "softmax": (softmax, lambda x: torch.softmax(x.double(), -1)),
    "min": (running_min, lambda x: x.double().amin(-1)),
    "product": (running_product, _chunked_product),
    "sum": (running_sum, lambda x: (x.double() * 2 + 1).sum(-1)),
}


@pytest.mark.parametrize(
    "shape", [(128, 252), (125, 252), (128, 256)], ids=lambda s: f"{s[0]}x{s[1]}"
)
@pytest.mark.parametrize("name", list(FORMS))
def test_running_values_ignore_padding(name: str, shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    fn, reference = FORMS[name]
    x = _negative(*shape) / 2 if name == "product" else _negative(*shape)
    if name == "min":
        x = -x  # A padded zero would be each row's minimum.
    got, ref = _kernel(fn)(x).double(), reference(x)
    if got.dim() == 1:
        errors = (got - ref).abs() / ref.abs().clamp_min(1e-30)
    else:
        errors = (got - ref).abs().amax(-1) / ref.abs().amax(-1)
    assert errors.max() < 1e-5


def last_chunk_sum(x: torch.Tensor) -> torch.Tensor:
    """A value that never reads its old value is the last tile's: padded zeros and
    all, without elements of earlier tiles."""
    m, n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        value = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            value = x[tile_m, tile_n] * 2.0
        out[tile_m] = value.sum(-1)
    return out


def last_chunk_amax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        value = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            value = x[tile_m, tile_n] * 2.0
        out[tile_m] = value.amax(-1)
    return out


@pytest.mark.parametrize("shape", [(16, 60), (16, 64)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_last_chunk_sum(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape).abs() + 1.0
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16, 16]),
    )
    got = kernel(last_chunk_sum)(x).double()
    start = (shape[1] - 1) // 16 * 16
    ref = (x[:, start:].double() * 2).sum(-1)
    assert ((got - ref).abs() / ref).max() < 1e-5


def test_last_chunk_amax_whole_tiles() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 64).abs() + 1.0
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[16, 16]),
    )
    got = kernel(last_chunk_amax)(x).double()
    ref = (x[:, 48:].double() * 2).amax(-1)
    assert ((got - ref).abs() / ref).max() < 1e-5


def _rows_of(x: torch.Tensor) -> torch.Tensor:
    return torch.empty([x.shape[0]], dtype=torch.float32, device=x.device)


def literal_min(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = _rows_of(x)
    for tile_m in hl.tile(m):
        low = hl.full([tile_m, 64], float("inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=64):
            low = torch.minimum(low, x[tile_m, tile_n])
        out[tile_m] = low.amin(-1)
    return out


def reduced_inside(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n])
            out[tile_m, tile_n] = top.amax(-1, keepdim=True)
    return out


@pytest.mark.parametrize(
    ("fn", "match"),
    [(literal_min, "sized by a number"), (reduced_inside, "reduced inside")],
    ids=["literal_min", "reduced_inside"],
)
def test_refused_at_a_partial_tile(fn: object, match: str) -> None:
    x = torch.randn(128, 252).abs() + 1.0
    with pytest.raises(UnsupportedOperationError, match=match):
        _kernel(fn)(x)


@pytest.mark.parametrize(
    "fn", [literal_min, reduced_inside], ids=["literal_min", "reduced_inside"]
)
def test_lowers_with_whole_tiles(fn: object) -> None:
    _kernel(fn)(torch.randn(128, 256).abs() + 1.0)


def _spans(size: int, block: int, begin: int = 0) -> list[tuple[int, int]]:
    return [(at, min(at + block, size)) for at in range(begin, size, block)]


def window(x: torch.Tensor, lo: hl.constexpr) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        low = hl.full([tile_m, block_n], 100.0, dtype=torch.float32)
        for tile_n in hl.tile(tile_m.begin + lo, tile_m.end, block_size=block_n):
            low = torch.minimum(low, x[tile_m, tile_n])
        out[tile_m] = low.amin(-1)
    return out


@pytest.mark.parametrize(
    ("lo", "block_m", "block_n"),
    [(10, 8, 16), (11, 8, 5), (3, 16, 8)],
    ids=["empty_lo10", "empty_lo11", "lo3"],
)
def test_window_runs_only_its_tiles(lo: int, block_m: int, block_n: int) -> None:
    """A window starting past its end runs no tile: splitting its bounds ran some."""
    x = torch.rand(40, 40) + 1.0
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[block_n, block_m]),
    )
    got = kernel(window)(x, hl.constexpr(lo)).double()
    ref = torch.empty(40, dtype=torch.float64)
    source = x.double()
    for begin, end in _spans(40, block_m):
        low = torch.full((end - begin, block_n), 100.0, dtype=torch.float64)
        for at, stop in _spans(end, block_n, begin + lo):
            low[:, : stop - at] = torch.minimum(
                low[:, : stop - at], source[begin:end, at:stop]
            )
        ref[begin:end] = low.amin(-1)
    assert (got - ref).abs().max() / ref.abs().max() < 1e-5


def counted(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + x[tile_m, tile_n]
            out[tile_m, tile_n] = total + tile_n.end + tile_n.count * 1000
    return out


@pytest.mark.parametrize("static", [True, False], ids=["static", "dynamic"])
def test_tile_count_counts_all_tiles(static: bool) -> None:
    """``tile.count`` counts the loop's tiles, with a partial one among them."""
    x = torch.rand(16, 60) + 1.0
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=static,
        config=helion.Config(block_sizes=[16, 16]),
    )
    source = x.double()
    ref = torch.empty_like(source)
    total = torch.zeros(16, 16, dtype=torch.float64)
    spans = _spans(60, 16)
    for begin, end in spans:
        total[:, : end - begin] += source[:, begin:end]
        ref[:, begin:end] = total[:, : end - begin] + end + len(spans) * 1000.0
    got = kernel(counted)(x).double()
    assert (got - ref).abs().max() / ref.abs().max() < 1e-5


def nested_last_chunk(x: torch.Tensor) -> torch.Tensor:
    m, a, c = x.shape
    block_a = hl.register_block_size(a)
    block_c = hl.register_block_size(c)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        value = hl.zeros([tile_m, block_a, block_c], dtype=torch.float32)
        for tile_a in hl.tile(a, block_size=block_a):
            for tile_c in hl.tile(c, block_size=block_c):
                value = x[tile_m, tile_a, tile_c] * 2.0
        out[tile_m] = value.sum(-1).sum(-1)
    return out


@pytest.mark.parametrize(
    "shape",
    [(16, 28, 30), (16, 32, 30), (16, 28, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_nested_last_chunk(shape: tuple[int, int, int]) -> None:
    """A value assigned in a nested loop, not read, is the last tile's: only a read
    of the old value makes it a running one."""
    x = torch.rand(*shape) + 1.0
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[8, 8, 16]),
    )
    first_a = (shape[1] - 1) // 8 * 8
    first_c = (shape[2] - 1) // 8 * 8
    ref = (x.double()[:, first_a:, first_c:] * 2).sum(-1).sum(-1)
    got = kernel(nested_last_chunk)(x).double()
    assert (got - ref).abs().max() / ref.abs().max() < 1e-5


def triangle_reduced_inside(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.zeros_like(x)
    for tile_m in hl.tile(m):
        low = hl.full([tile_m, block_n], float("inf"), dtype=torch.float32)
        for tile_n in hl.tile(tile_m.end, block_size=block_n):
            low = torch.minimum(low, x[tile_m, tile_n])
            out[tile_m, tile_n] = low.amin(-1, keepdim=True) + x[tile_m, tile_n] * 0.0
    return out


def test_reduced_inside_a_loop_bounded_by_a_whole_enclosing_tile() -> None:
    """No partial tile, so nothing to refuse: the end is an enclosing tile's value."""
    x = torch.rand(64, 64) + 1.0
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[8, 16]),
    )
    source = x.double()
    ref = source * 0.0
    for begin, end in _spans(64, 16):
        low = torch.full((end - begin, 8), float("inf"), dtype=torch.float64)
        for at, stop in _spans(end, 8):
            low[:, : stop - at] = torch.minimum(
                low[:, : stop - at], source[begin:end, at:stop]
            )
            ref[begin:end, at:stop] = low.amin(-1, keepdim=True)
    got = kernel(triangle_reduced_inside)(x).double()
    assert (got - ref).abs().max() / ref.abs().max() < 1e-5


def reduced_in_a_nested_loop(x: torch.Tensor) -> torch.Tensor:
    m, a, c = x.shape
    block_a = hl.register_block_size(a)
    block_c = hl.register_block_size(c)
    out = torch.empty([m, a], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_a], float("-inf"), dtype=torch.float32)
        for tile_a in hl.tile(a, block_size=block_a):
            for tile_c in hl.tile(c, block_size=block_c):
                top = torch.maximum(top, x[tile_m, tile_a, tile_c].amax(-1))
                out[tile_m, tile_a] = top.amax(-1, keepdim=True)
    return out


def test_refused_reduced_in_a_nested_loop() -> None:
    """The running value's mask is in a loop nested in the one with the partial tile
    (0.69 off without the refusal)."""
    kernel = helion.kernel(
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[8, 8, 8]),
    )
    with pytest.raises(UnsupportedOperationError, match="reduced inside"):
        kernel(reduced_in_a_nested_loop)(torch.rand(8, 12, 16))


_COMPILE_GUARD = """
import helion, helion.language as hl, torch
import helion_mlir_backend
from tests.test_partial_tile_running_values import softmax
x = torch.randn(4096, 4092, generator=torch.Generator().manual_seed(0))
kernel = helion.kernel(
    backend="mlir", static_shapes=True, config=helion.Config(block_sizes={blocks})
)
ref = torch.softmax(x.double(), -1)
assert ((kernel(softmax)(x).double() - ref).abs().max() / ref.abs().max()) < 1e-5
"""


@pytest.mark.parametrize(
    "blocks", [[1024, 256], [64, 256]], ids=["block1024", "block64"]
)
def test_large_partial_tile_compiles_in_time(blocks: list[int]) -> None:
    """A per-element select on the kept part took far longer to compile at this
    size."""
    subprocess.run(
        [sys.executable, "-c", _COMPILE_GUARD.format(blocks=blocks)],
        check=True,
        timeout=90,
    )
