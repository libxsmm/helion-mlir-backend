"""``helion_mlir_utils`` f32 GEMM paths: split-K for few output tiles, the
batched kernel of ``bmm`` (batch as a grid dim), and their gates."""

from __future__ import annotations

import importlib

import pytest
import torch

# The package exports functions named like its modules.
mm = importlib.import_module("helion_mlir_utils.matmul")


def _error(out: torch.Tensor, expected: torch.Tensor) -> float:
    return ((out.double() - expected).abs().max() / expected.abs().max()).item()


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list[object]:
    """Calls of ``mm.<name>`` (still run), recorded by their return value."""
    calls: list[object] = []
    original = getattr(mm, name)

    def wrapper(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(mm, name, wrapper)
    return calls


@pytest.mark.parametrize(
    ("m", "n", "k", "split"),
    [
        (64, 128, 4096, True),  # one 64x128 tile: K split for the threads
        (96, 128, 4096, False),  # rows not whole 64-row tiles: one K loop
    ],
    ids=["split", "ragged_rows"],
)
def test_split_k_gate(
    monkeypatch: pytest.MonkeyPatch, m: int, n: int, k: int, split: bool
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    calls = _spy(monkeypatch, "_matmul_split_k")
    a, b = torch.randn(m, k), torch.randn(k, n)
    out = mm.matmul(a, b)
    assert (len(calls) == 1) is split
    assert out.shape == (m, n)
    assert _error(out, a.double() @ b.double()) < 1e-5


def test_split_k_splits_fill_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """S x tiles fill the threads, K chunks stay at least 512 long, and only
    the first level of a nested ``OMP_NUM_THREADS`` list counts."""
    args = (False, False, None, None, None, mm.identity_epilogue)
    monkeypatch.setenv("OMP_NUM_THREADS", "64,8")
    assert mm._split_k(256, 256, 8192, *args) == 8
    assert mm._split_k(64, 128, 2048, *args) == 4
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert mm._split_k(64, 128, 4096, *args) == 2
    # Not split: transposed, an epilogue, or enough tiles already.
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    assert (
        mm._split_k(256, 256, 8192, True, False, None, None, None, mm.identity_epilogue)
        == 0
    )
    assert mm._split_k(256, 256, 8192, False, False, None, None, None, torch.relu) == 0
    assert mm._split_k(2048, 2048, 2048, *args) == 0


@pytest.mark.parametrize(
    ("shape", "direct"),
    [
        ((2, 160, 96, 224), True),  # 32-multiples: the batched kernel
        ((2, 65, 129, 129), False),  # no dividing tiles: per-slice fallback
    ],
    ids=["direct", "ragged_fallback"],
)
def test_bmm_f32_gate(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], direct: bool
) -> None:
    calls = _spy(monkeypatch, "_bmm_f32")
    batch, m, n, k = shape
    a, b = torch.randn(batch, m, k), torch.randn(batch, k, n)
    out = mm.bmm(a, b)
    assert len(calls) == 1
    assert (calls[0] is not None) is direct
    assert out.shape == (batch, m, n)
    assert _error(out, torch.bmm(a.double(), b.double())) < 1e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "b_shape", [(1, 64, 64), (3, 32, 64)], ids=["batch", "contracted_dim"]
)
def test_bmm_rejects_mismatched_operands(
    dtype: torch.dtype, b_shape: tuple[int, ...]
) -> None:
    """Raises like ``torch.bmm`` (it read past ``b`` and returned zeros)."""
    a = torch.randn(3, 64, 64, dtype=dtype)
    b = torch.randn(*b_shape, dtype=dtype)
    with pytest.raises(RuntimeError, match="batch2"):
        mm.bmm(a, b)


@pytest.mark.parametrize(
    ("shape", "tiles"),
    [
        ((8, 64, 128, 256), (1, 64, 128, 256)),  # whole slices, K in one chunk
        ((4, 64, 64, 128), (1, 64, 64, 128)),  # the fewest slices it takes
        ((3, 64, 64, 128), None),  # fewer: a kernel per slice
        ((8, 65, 128, 256), None),  # rows of no whole AMX block
        ((8, 64, 129, 256), None),  # columns of none
        ((8, 64, 128, 250), None),  # K of none
    ],
    ids=["batched", "min_batch", "few_slices", "ragged_m", "ragged_n", "ragged_k"],
)
def test_bmm_bf16_gate(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    tiles: tuple[int, ...] | None,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert mm._bmm_batched_tiles(*shape) == tiles


@pytest.mark.parametrize(
    ("threads", "shape", "tiles"),
    [
        ("64", (8, 512, 1024, 2048), (1, 256, 256, 512)),
        ("64", (128, 512, 1024, 2048), (1, 512, 512, 512)),  # the accumulator's limit
        ("64", (128, 1024, 1024, 1024), (1, 512, 512, 512)),
        ("64", (4, 32, 32, 64), (1, 32, 32, 64)),  # no tile left to halve: threads idle
        # One AMX block per tile keeps K whole: in chunks it doesn't reach AMX.
        ("16", (4, 64, 32, 1024), (1, 32, 32, 1024)),
        ("64", (8, 32, 32, 1024), (1, 32, 32, 1024)),
        ("64", (8, 64, 64, 1024), (1, 32, 32, 1024)),
        ("64", (8, 128, 128, 1024), (1, 64, 32, 512)),  # two blocks: chunks reach AMX
    ],
)
def test_bmm_bf16_tiles_give_each_thread_one(
    monkeypatch: pytest.MonkeyPatch,
    threads: str,
    shape: tuple[int, ...],
    tiles: tuple[int, ...] | None,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", threads)
    assert mm._bmm_batched_tiles(*shape) == tiles


@pytest.mark.parametrize(
    "shape",
    [(8, 64, 128, 256), (4, 64, 64, 128), (6, 96, 160, 192), (8, 32, 32, 1024)],
    ids=["whole", "min_batch", "not_powers_of_two", "one_block"],
)
def test_bmm_bf16_batched_is_the_per_slice_result(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    """Every slice, against the per-slice kernels and float64."""
    batch, m, n, k = shape
    generator = torch.Generator().manual_seed(batch * m)
    a = torch.randn(batch, m, k, generator=generator, dtype=torch.float64)
    b = torch.randn(batch, k, n, generator=generator, dtype=torch.float64)
    a16, b16 = a.to(torch.bfloat16), b.to(torch.bfloat16)
    tiles = mm._bmm_batched_tiles(batch, m, n, k)
    assert tiles is not None
    mm._BMM_KERNELS.pop(tiles, None)
    # Different slices must reach different outputs: a wrong batch index shows.
    out = mm.bmm(a16, b16).clone()
    assert tiles in mm._BMM_KERNELS, "the batched kernel didn't run"
    reference = torch.bmm(a16.double(), b16.double())
    assert _error(out, reference) < 2e-2
    for i in range(batch):
        assert _error(out[i : i + 1], reference[i : i + 1]) < 2e-2
    with monkeypatch.context() as patch:
        patch.setattr(mm, "_bmm_batched_tiles", lambda *_: None)
        per_slice = mm.bmm(a16, b16).clone()
    assert _error(out, per_slice.double()) < 2e-2
