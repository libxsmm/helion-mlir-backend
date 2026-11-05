"""``helion_mlir_utils`` f32 GEMMs packed per tile (``_matmul_f32_fused_pack``):
the tile gate, results bit-identical to the blocked path it replaces (the same
per-element K order), transposes, epilogues and the fallbacks."""

from __future__ import annotations

import importlib

import pytest
import torch

# The package exports functions named like its modules.
mm = importlib.import_module("helion_mlir_utils.matmul")


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


def _operands(
    m: int, n: int, k: int, trans_a: bool, trans_b: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a = torch.randn(k, m) if trans_a else torch.randn(m, k)
    b = torch.randn(n, k) if trans_b else torch.randn(k, n)
    expected = (a.T if trans_a else a).double() @ (b.T if trans_b else b).double()
    return a, b, expected


def test_f32_fused_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    assert mm._f32_fused_tiles(4096, 4096, 4096) == (256, 256, 128)
    assert mm._f32_fused_tiles(2048, 4096, 8192) == (256, 256, 128)
    assert mm._f32_fused_tiles(4096, 4096, 64) == (256, 256, 64)
    # Fewer tiles than threads: the longer side halves, down to 64.
    assert mm._f32_fused_tiles(1024, 1024, 1024) == (128, 128, 128)
    assert mm._f32_fused_tiles(128, 4096, 4096) == (64, 128, 128)
    assert mm._f32_fused_tiles(128, 192, 96) == (64, 64, 32)
    # Not whole tiles or K chunks: the blocked path.
    assert mm._f32_fused_tiles(100, 256, 256) is None
    assert mm._f32_fused_tiles(256, 96, 256) is None
    assert mm._f32_fused_tiles(256, 256, 48) is None
    # Only the first level of a nested list counts.
    monkeypatch.setenv("OMP_NUM_THREADS", "4,8")
    assert mm._f32_fused_tiles(1024, 1024, 1024) == (256, 256, 128)


@pytest.mark.parametrize(
    ("trans_a", "trans_b"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["nn", "tn", "nt", "tt"],
)
def test_f32_fused_matches_blocked(
    monkeypatch: pytest.MonkeyPatch, trans_a: bool, trans_b: bool
) -> None:
    """Bit-identical to the blocked path: each output element sums its K
    products in the same order, only the packing moved into the tile."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    m, n, k = 128, 192, 160  # tiles 64x64, K chunks of 32
    a, b, expected = _operands(m, n, k, trans_a, trans_b)
    calls = _spy(monkeypatch, "_matmul_f32_fused")
    out = mm.matmul(a, b, trans_a=trans_a, trans_b=trans_b)
    assert len(calls) == 1 and calls[0] is not None
    assert out.shape == (m, n)
    monkeypatch.setattr(mm, "_f32_fused_tiles", lambda *_: None)
    blocked = mm.matmul(a, b, trans_a=trans_a, trans_b=trans_b)
    assert torch.equal(out, blocked)
    assert ((out.double() - expected).abs().max() / expected.abs().max()) < 1e-6


def test_f32_fused_two_k_chunks_and_epilogue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accumulator carried over K chunks (256 = 2 x 128), epilogue fused."""
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    a, b, expected = _operands(256, 256, 256, False, False)
    calls = _spy(monkeypatch, "_matmul_f32_fused")
    out = mm.matmul(a, b, epilogue=torch.relu)
    assert mm._f32_fused_tiles(256, 256, 256) == (128, 128, 128)
    assert len(calls) == 1 and calls[0] is not None
    expected = torch.relu(expected)
    assert ((out.double() - expected).abs().max() / expected.abs().max()) < 1e-6


@pytest.mark.parametrize(
    ("trans_a", "trans_b"),
    [(True, False), (False, True), (True, True)],
    ids=["tn", "nt", "tt"],
)
def test_f32_fused_large_tiles_match_blocked(
    monkeypatch: pytest.MonkeyPatch, trans_a: bool, trans_b: bool
) -> None:
    """128x128 tiles over three 128-deep K chunks, transposed operands and an
    epilogue: bit-identical to the blocked path too."""
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    m, n, k = 256, 256, 384
    assert mm._f32_fused_tiles(m, n, k) == (128, 128, 128)
    a, b, expected = _operands(m, n, k, trans_a, trans_b)
    calls = _spy(monkeypatch, "_matmul_f32_fused")
    out = mm.matmul(a, b, trans_a=trans_a, trans_b=trans_b, epilogue=torch.relu)
    assert len(calls) == 1 and calls[0] is not None
    monkeypatch.setattr(mm, "_f32_fused_tiles", lambda *_: None)
    blocked = mm.matmul(a, b, trans_a=trans_a, trans_b=trans_b, epilogue=torch.relu)
    assert torch.equal(out, blocked)
    expected = torch.relu(expected)
    assert ((out.double() - expected).abs().max() / expected.abs().max()) < 1e-6


def test_f32_fused_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ragged shapes and a bias take the blocked path, still correct."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    calls = _spy(monkeypatch, "_matmul_f32_fused")
    a, b, expected = _operands(100, 96, 160, False, False)
    out = mm.matmul(a, b)
    assert calls == [None]
    assert ((out.double() - expected).abs().max() / expected.abs().max()) < 1e-6
    a, b, expected = _operands(128, 128, 128, False, False)
    bias = torch.randn(128)
    out = mm.matmul(a, b, bias=bias)
    assert calls == [None]
    expected = expected + bias.double()
    assert ((out.double() - expected).abs().max() / expected.abs().max()) < 1e-6


def test_f32_fused_non_contiguous_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Views (a transposed view without ``trans_a``) are made contiguous."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    a, b, expected = _operands(128, 128, 128, True, False)
    calls = _spy(monkeypatch, "_matmul_f32_fused")
    out = mm.matmul(a.T, b)
    assert len(calls) == 1 and calls[0] is not None
    assert ((out.double() - expected).abs().max() / expected.abs().max()) < 1e-6
