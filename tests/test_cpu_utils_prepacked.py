"""``helion_mlir_utils`` prepacked GEMMs: the f32 rows kernel, the prepared
call that :func:`linear` caches, and the small-M VNNI panel choice."""

from __future__ import annotations

import importlib
import os

from helion_mlir_utils.linear import linear
from helion_mlir_utils.matmul import identity_epilogue
import pytest
import torch

# The package exports functions named like these modules.
mm = importlib.import_module("helion_mlir_utils.matmul")


def _old_blocked(
    a: torch.Tensor, b4: torch.Tensor, n: int, bias: torch.Tensor | None, epilogue
) -> torch.Tensor:
    """The f32 path before the rows kernel: A packed per call, blocked kernel."""
    m, k = a.shape
    m_pad, k_pad = mm._round_up(m, mm.BLOCK_M), mm._round_up(k, mm.BLOCK_K)
    blocks_n = int(b4.shape[0])
    a4 = mm.pack_a_blocked(a, m_pad=m_pad, k_pad=k_pad)
    if bias is None:
        out4 = mm._matmul_blocked_kernel(a4, b4, epilogue)
    else:
        n_pad = blocks_n * mm.BLOCK_N
        bias3 = mm._pad_to(bias.reshape(1, -1), (1, n_pad)).reshape(
            blocks_n, 1, mm.BLOCK_N
        )
        out4 = mm._matmul_blocked_kernel_bias(a4, b4, bias3, epilogue)
    return out4.reshape(m_pad, blocks_n * mm.BLOCK_N)[:m, :n]


@pytest.mark.parametrize(
    ("m", "k", "n", "bias"),
    [
        (64, 256, 96, True),  # whole row tiles: one call
        (33, 256, 64, True),  # ragged M: bulk rows + a padded rest
        (130, 96, 70, False),  # ragged M and N
        (17, 64, 32, False),  # fewer rows than one tile: the rest only
        (40, 100, 64, True),  # K % 32 != 0: the blocked path
    ],
)
def test_f32_rows_kernel_matches_blocked_path(
    m: int, k: int, n: int, bias: bool
) -> None:
    torch.manual_seed(m + k + n)
    a = torch.randn(m, k)
    w = torch.randn(n, k)
    b = torch.randn(n) if bias else None
    b4 = mm.pack_b_blocked_t(w)
    out = mm.matmul_prepacked_b(a, b4, n, b, torch.relu)
    assert torch.equal(out, _old_blocked(a, b4, n, b, torch.relu))
    if k % mm.BLOCK_K == 0:
        assert mm._matmul_prepacked_rows.cache_info().currsize > 0, (
            "the rows kernel ran"
        )
    expected = torch.relu(a @ w.T + (0 if b is None else b))
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_reuses_its_call_until_the_weight_changes(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(128, 64).to(dtype)
    x = torch.randn(32, 128, dtype=dtype)
    with torch.no_grad():
        y, cache = linear(x, layer, torch.relu)
        call = cache.call
        x2 = torch.randn(32, 128, dtype=dtype)
        y2, cache = linear(x2, layer, torch.relu, cache=cache)
        assert cache.call is call
        assert torch.equal(
            y2,
            mm.matmul_prepacked_b(x2, cache.packed_weight, 64, cache.bias, torch.relu),
        )
        # Another epilogue: a new call on the same packed weight.
        _, cache2 = linear(x2, layer, identity_epilogue, cache=cache)
        assert cache2.call is not call
        assert cache2.packed_weight is cache.packed_weight
        # A weight mutated in place: repacked, and the output follows.
        layer.weight.mul_(-1.0)
        y3, cache3 = linear(x2, layer, torch.relu, cache=cache)
        assert cache3.packed_weight is not cache.packed_weight
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(
            y3.float(),
            torch.relu(x2.float() @ layer.weight.float().T + layer.bias.float()),
            atol=tol,
            rtol=tol,
        )


def test_linear_call_falls_back_for_other_activations() -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(64, 64)
    with torch.no_grad():
        _, cache = linear(torch.randn(32, 64), layer, torch.relu)
        xt = torch.randn(64, 32).t()  # same shape, not contiguous
        y, _ = linear(xt, layer, torch.relu, cache=cache)
    torch.testing.assert_close(y, torch.relu(layer(xt)), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("case", ["bias_length", "bias_rank", "dtype"])
def test_prepared_call_raises_like_matmul_prepacked_b(case: str) -> None:
    a = torch.randn(32, 64)
    b4 = mm.pack_b_blocked_t(torch.randn(64, 64))
    bias = None
    if case == "bias_length":
        bias = torch.randn(32)
    elif case == "bias_rank":
        bias = torch.randn(64, 1)
    else:
        a = a.bfloat16()
    with pytest.raises(ValueError) as general:
        mm.matmul_prepacked_b(a, b4, 64, bias, torch.relu)
    with pytest.raises(ValueError) as prepared:
        mm.prepacked_b_call(a, b4, 64, bias, torch.relu)(a)
    assert str(prepared.value) == str(general.value)


def test_small_m_vnni_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    # 128 rows: 64-row x 128-column tiles (64 of them), not 128 x 64.
    panel = mm.vnni_panel(128, 4096)
    assert panel == 128
    assert mm._prepacked_tile_m(128, 4096 // panel, panel) == 64
    # The N = 2048 layer keeps 64-column panels (a tie: the narrower one).
    assert mm.vnni_panel(128, 2048) == 64
    # From 256 rows on, panels that fit keep the rule for shared rows.
    assert mm.vnni_panel(1024, 8192) == 256
    # None fits: a quarter fewer operand bytes per tile than 64 columns, the
    # widest without K (with K: test_vnni_panel_without_a_fitting_panel).
    assert mm.vnni_panel(256, 4096) == 256


def test_linear_panel_override() -> None:
    """``panel`` sets the bf16 weight's VNNI panel width (the output tile's),
    is part of the cache key, and leaves the result within bf16 rounding of
    the default panels' (256 is the rule's choice at 128x16384x16384)."""
    torch.manual_seed(0)
    layer = torch.nn.Linear(512, 512).to(torch.bfloat16)
    x = torch.rand(64, 512).to(torch.bfloat16)
    default, cache = linear(x, layer)
    wide, wide_cache = linear(x, layer, cache=cache, panel=256)
    assert wide_cache is not cache
    assert int(wide_cache.packed_weight.shape[2]) == 256
    assert int(cache.packed_weight.shape[2]) != 256
    _, again = linear(x, layer, cache=wide_cache, panel=256)
    assert again.packed_weight is wide_cache.packed_weight
    ref = torch.nn.functional.linear(
        x.float(), layer.weight.float(), layer.bias.float()
    )
    for out in (default, wide):
        error = ((out.float() - ref).abs().max() / ref.abs().max()).item()
        assert error < 1e-2


@pytest.mark.parametrize("post_bias", [False, True])
def test_linear_affine_reuses_its_call(
    monkeypatch: pytest.MonkeyPatch, post_bias: bool
) -> None:
    """bf16 ``linear_affine`` (L2/33, L2/39) prepares its call once, like
    ``linear``: later calls skip ``matmul_prepacked_b``'s per-call staging and
    give the same bits."""
    from helion_mlir_utils.linear import linear_affine

    torch.manual_seed(1)
    dtype = torch.bfloat16
    layer = torch.nn.Linear(128, 64).to(dtype)
    scale = torch.randn(64, dtype=dtype)
    shift = torch.randn(64, dtype=dtype) if post_bias else None
    x = torch.randn(32, 128, dtype=dtype)
    with torch.no_grad():
        y, cache = linear_affine(x, layer, scale, shift, torch.relu)
        assert cache.call is not None
        general = mm.matmul_prepacked_b(
            x, cache.packed_weight, 64, cache.bias, torch.relu, scale, shift
        )
        assert torch.equal(y, general)

        def no_general(*args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("the prepared call took the general path")

        monkeypatch.setattr(mm, "matmul_prepacked_b", no_general)
        x2 = torch.randn(32, 128, dtype=dtype)
        y2, cache2 = linear_affine(x2, layer, scale, shift, torch.relu, cache=cache)
        assert cache2.call is cache.call
        monkeypatch.undo()
        assert torch.equal(
            y2,
            mm.matmul_prepacked_b(
                x2, cache.packed_weight, 64, cache.bias, torch.relu, scale, shift
            ),
        )
        # Another epilogue, then another thread count: each rebuilds the
        # call, which stays bit-equal to the general path.
        y5, cache5 = linear_affine(
            x2, layer, scale, shift, identity_epilogue, cache=cache2
        )
        assert cache5.call is not cache2.call
        assert torch.equal(
            y5,
            mm.matmul_prepacked_b(
                x2, cache.packed_weight, 64, cache.bias, identity_epilogue, scale, shift
            ),
        )
        threads = os.environ.get("OMP_NUM_THREADS")
        monkeypatch.setenv("OMP_NUM_THREADS", "3" if threads != "3" else "5")
        y6, cache6 = linear_affine(x2, layer, scale, shift, torch.relu, cache=cache5)
        assert cache6.call is not cache5.call
        assert cache6.call[1] == os.environ["OMP_NUM_THREADS"]
        assert torch.equal(
            y6,
            mm.matmul_prepacked_b(
                x2, cache.packed_weight, 64, cache.bias, torch.relu, scale, shift
            ),
        )
        # Another activation layout takes the general path, same result.
        xt = torch.randn(128, 32, dtype=dtype).t()
        y3, _ = linear_affine(xt, layer, scale, shift, torch.relu, cache=cache)
        assert torch.equal(
            y3, linear_affine(xt.contiguous(), layer, scale, shift, torch.relu)[0]
        )
        # A scale mutated in place: a new cache, and the output follows.
        scale.mul_(-1.0)
        y4, cache4 = linear_affine(x2, layer, scale, shift, torch.relu, cache=cache)
        assert cache4.key != cache.key
        expected = x2.float() @ layer.weight.float().T + layer.bias.float()
        expected = expected.to(dtype).float() * scale.float()
        if shift is not None:
            expected = expected + shift.float()
        err = (y4.float() - torch.relu(expected)).abs().max() / expected.abs().max()
        assert err < 2e-2


def test_linear_affine_f32_keeps_the_general_path() -> None:
    from helion_mlir_utils.linear import linear_affine

    torch.manual_seed(2)
    layer = torch.nn.Linear(64, 64)
    scale = torch.randn(64)
    x = torch.randn(32, 64)
    with torch.no_grad():
        y, cache = linear_affine(x, layer, scale, None, torch.relu)
        y2, _ = linear_affine(x, layer, scale, None, torch.relu, cache=cache)
    assert torch.equal(y, y2)
    torch.testing.assert_close(y, torch.relu(layer(x) * scale), atol=1e-4, rtol=1e-4)


def test_row_sum_binds_once_per_shape_and_epilogue() -> None:
    red = importlib.import_module("helion_mlir_utils.reduction")

    def half(t: torch.Tensor) -> torch.Tensor:
        return t * 0.5

    torch.manual_seed(3)
    x = torch.randn(40, 96, dtype=torch.bfloat16)
    red._ROW_SUM_BOUND.clear()
    out = red.row_sum(x, half)
    assert len(red._ROW_SUM_BOUND) == 1
    out2 = red.row_sum(x.clone(), half)
    assert len(red._ROW_SUM_BOUND) == 1
    assert torch.equal(out, out2)
    # The unbound kernel call (the previous path) gives the same bits.
    (block_sizes,) = {key[0] for key in red._ROW_SUM_BOUND}
    assert torch.equal(out, red._row_sum_kernel(block_sizes)(x, half))
    red.row_sum(x, identity_epilogue)
    red.row_sum(torch.randn(8, 96, dtype=torch.bfloat16), half)
    assert len(red._ROW_SUM_BOUND) == 3
    ref = x.float().sum(1, keepdim=True) * 0.5
    torch.testing.assert_close(out.float(), ref, atol=0.1, rtol=1e-2)


def test_row_sum_cache_evicts_least_recently_used() -> None:
    """Past ``_ROW_SUM_BOUND_SIZE`` keys the least recently used bound kernel
    goes, a key in use stays, and evicted keys rebind with the right result."""
    red = importlib.import_module("helion_mlir_utils.reduction")
    size = red._ROW_SUM_BOUND_SIZE

    def scale_by(factor: float):
        def epilogue(t: torch.Tensor) -> torch.Tensor:
            return t * factor

        return epilogue

    torch.manual_seed(4)
    x = torch.randn(8, 32, dtype=torch.bfloat16)
    epilogues = [scale_by(float(i + 1)) for i in range(size + 1)]
    red._ROW_SUM_BOUND.clear()
    outs = {}
    for i, epilogue in enumerate(epilogues):
        # Key 0, the oldest, is used again before every new key: the new key
        # past the size evicts key 1, not key 0 (as first-in first-out would).
        if i:
            red.row_sum(x, epilogues[0])
        outs[i] = red.row_sum(x, epilogue)
    assert len(red._ROW_SUM_BOUND) == size
    keys = [key[-1] for key in red._ROW_SUM_BOUND]
    assert epilogues[0] in keys
    assert epilogues[1] not in keys  # the least recently used
    ref = x.float().sum(1, keepdim=True)
    for i in (0, 1, 2, size):
        out = red.row_sum(x, epilogues[i])
        assert torch.equal(out, outs[i])
        torch.testing.assert_close(out.float(), ref * (i + 1.0), atol=0.5, rtol=2e-2)


def _native_extension() -> bool:
    from helion_mlir_backend._compiler import native

    return native.module() is not None


@pytest.mark.skipif(not _native_extension(), reason="no native extension")
@pytest.mark.parametrize(
    ("dtype", "affine"),
    [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)],
)
def test_prepared_linear_calls_run_natively(dtype: torch.dtype, affine: bool) -> None:
    """The prepared call of ``linear`` / ``linear_affine`` is the backend's
    native prepared run: later calls with the same activation layout run
    natively, bit-equal to the general path; others take the Python call."""
    from helion_mlir_utils.linear import linear_affine

    torch.manual_seed(2)
    layer = torch.nn.Linear(128, 64).to(dtype)
    scale = torch.randn(64, dtype=dtype)
    shift = torch.randn(64, dtype=dtype)

    def run(x: torch.Tensor, cache: object) -> tuple[torch.Tensor, object]:
        if affine:
            return linear_affine(x, layer, scale, shift, torch.relu, cache=cache)
        return linear(x, layer, torch.relu, cache=cache)

    def general(x: torch.Tensor, cache: object) -> torch.Tensor:
        extra = (scale, shift) if affine else ()
        return mm.matmul_prepacked_b(
            x, cache.packed_weight, 64, cache.bias, torch.relu, *extra
        )

    with torch.no_grad():
        x = torch.randn(32, 128, dtype=dtype)
        _, cache = run(x, None)
        prepared = cache.call[2]
        assert type(prepared).__name__ == "PreparedRun"
        _, cache = run(x, cache)  # (the kernel's first call records its plan)
        before = prepared.native_calls
        x2 = torch.randn(32, 128, dtype=dtype)
        y2, cache = run(x2, cache)
        assert prepared.native_calls == before + 1
        assert y2.shape == (32, 64) and y2.is_contiguous()
        assert torch.equal(y2, general(x2, cache))
        # A non-contiguous activation of the same shape: the Python call.
        xt = torch.randn(128, 32, dtype=dtype).t()
        y3, cache = run(xt, cache)
        assert torch.equal(y3, general(xt.contiguous(), cache))


@pytest.mark.skipif(not _native_extension(), reason="no native extension")
def test_prepared_call_keeps_the_closure_for_padded_columns() -> None:
    # N = 70 is padded to whole panels: the result is a column slice, which
    # the native prepared run does not make.
    a = torch.randn(32, 64)
    b4 = mm.pack_b_blocked_t(torch.randn(70, 64))
    call = mm.prepacked_b_call(a, b4, 70, None, torch.relu)
    assert type(call).__name__ == "function"
    assert torch.equal(call(a), mm.matmul_prepacked_b(a, b4, 70, None, torch.relu))


@pytest.mark.parametrize(
    ("pairs_even", "panel", "tile_m", "chunk"),
    [
        (1024, 256, 1024, 1024),  # K 2048, 1 MiB panel: one chunk
        (2048, 128, 2048, 2048),  # K 4096, 1 MiB panel: one chunk
        (2048, 64, 64, 2048),  # few rows, 512 KiB panel: one chunk
        (2048, 256, 1024, 512),  # 2 MiB panel: 512 KiB chunks, as before
        (4096, 64, 1024, 2048),  # K 8192 at a 1 MiB panel: chunks, as before
        (4096, 64, 128, 1024),  # K 8192, few rows: 256 KiB chunks, as before
        (16384, 256, 64, 256),  # 128x32768x32768: 256 KiB chunks, as before
    ],
)
def test_prepacked_k_chunk(
    pairs_even: int, panel: int, tile_m: int, chunk: int
) -> None:
    assert mm._prepacked_k_chunk(pairs_even, panel, tile_m) == chunk


def test_vnni_panel_prefers_one_k_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    # K = 4096: 128-column panels run K as one chunk; 256 would need chunks.
    assert mm.vnni_panel(16384, 4096, 4096) == 128
    assert mm.vnni_panel(16384, 1024, 4096) == 128
    # K = 2048: 256 columns already run as one chunk.
    assert mm.vnni_panel(16384, 4096, 2048) == 256
    # No panel runs K = 8192 as one chunk, and without K: the widest that fits.
    assert mm.vnni_panel(1024, 8192, 8192) == 256
    assert mm.vnni_panel(16384, 4096) == 256
    # No panel fits, and 128 columns would save no operand bytes, whatever
    # that is worth here: 256 rows x 1024 columns.
    assert mm.vnni_panel(256, 1024, 4096) == 64


@pytest.mark.parametrize(
    "threads,m,n,k,panel",
    [
        # A quarter fewer operand bytes, nothing added: L3/3's layers and others.
        (64, 1024, 1024, 1024, 256),
        (64, 1000, 1000, 1000, 256),
        (64, 1024, 700, 700, 256),
        (56, 1024, 1024, 1024, 256),
        (8, 256, 512, 512, 256),
        (2, 256, 128, 128, 128),
        (32, 1024, 512, 8192, 128),
        # The fewest K chunks first (L3/3's first layer).
        (64, 1024, 1024, 8192, 128),
        (64, 256, 4096, 4096, 128),
        (64, 384, 4096, 4096, 128),
        # Removes 64 columns' rest rows, whatever it saves (L2/98 at 56 threads).
        (64, 768, 768, 768, 128),
        (64, 448, 1024, 1024, 128),
        (56, 1024, 512, 8192, 256),
        # 256 columns would add padding or rest rows: 128.
        (64, 320, 4096, 1024, 128),
        # K in chunks where 64 columns run one, with even tiles: 64 columns stay.
        (64, 288, 4096, 4096, 64),
        # Uneven tiles at 64 columns: chunks pay.
        (3, 288, 256, 4096, 256),
        # Less than a quarter saved: 64 columns stay, though some would gain.
        (64, 1024, 512, 8192, 64),
        (64, 512, 1024, 1024, 64),
        (64, 300, 1000, 1000, 64),
        (3, 256, 256, 256, 64),
        (128, 1024, 1024, 1024, 64),
    ],
)
def test_vnni_panel_without_a_fitting_panel(
    monkeypatch: pytest.MonkeyPatch, threads: int, m: int, n: int, k: int, panel: int
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", str(threads))
    assert mm.vnni_panel(m, n, k) == panel


@pytest.mark.parametrize(
    ("m", "n", "k", "panel"),
    [
        # One tile per thread: L2/55, 56, 59 and L2/66, 68 at bench-cpu.
        (128, 32768, 32768, 512),
        (128, 16384, 16384, 256),
        (128, 8192, 8192, 128),
        # From K = 6144 on, with few rows and A of at least 1.5 MiB.
        (128, 8192, 6144, 128),
        (128, 8192, 6112, 64),
        (224, 16384, 8192, 256),
        (64, 16384, 12288, 256),
        (96, 32768, 8192, 512),
        # A below 1.5 MiB: the cost rule.
        (96, 32768, 8190, 64),
        (64, 16384, 8192, 64),
        (32, 32768, 8192, 64),
        (32, 16384, 6144, 64),
        (64, 16384, 6144, 64),
        # Smaller K (the MLP layers): the cost rule, unchanged.
        (128, 4096, 4096, 128),
        (128, 8192, 4096, 64),
        (128, 32768, 4096, 64),
        # Too few columns for a tile per thread above 64: the cost rule.
        (128, 4096, 8192, 128),
        (128, 2048, 32768, 64),
        # No K given: the cost rule.
        (128, 32768, None, 64),
        # Padding would leave threads with two tiles beside threads with one.
        (128, 32768 + 64, 32768, 64),
        # 256 rows and more: the rule for shared rows.
        (256, 32768, 32768, 256),
    ],
)
def test_vnni_panel_one_tile_per_thread(
    monkeypatch: pytest.MonkeyPatch, m: int, n: int, k: int | None, panel: int
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    assert mm.vnni_panel(m, n, k) == panel


@pytest.mark.parametrize(
    ("threads", "panel"),
    [
        (32, 512),  # two tiles each
        (48, 256),  # 3 tiles each (1/8 above even); 512 columns would leave 2 or 1
        (56, 128),
        (128, 256),
        (256, 128),
        (512, 128),  # no panel gives each thread a tile: the cost rule
    ],
)
def test_vnni_panel_tile_per_thread_counts_threads(
    monkeypatch: pytest.MonkeyPatch, threads: int, panel: int
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", str(threads))
    assert mm.vnni_panel(128, 32768, 32768) == panel


def test_vnni_panel_invalid_thread_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OMP_NUM_THREADS=0`` takes the cost rule, as it did before the
    one-tile rule, instead of dividing by zero."""
    monkeypatch.setenv("OMP_NUM_THREADS", "0")
    assert mm.vnni_panel(128, 32768, 32768) == 64


@pytest.mark.parametrize(
    ("threads", "m", "n", "k", "panel"),
    [
        (1, 128, 456, 6144, 512),  # N below the panel
        (2, 129, 1000, 6146, 512),  # ragged in every dim
        (3, 255, 1500, 6200, 512),
        (7, 224, 3500, 6146, 512),
    ],
)
def test_one_tile_rule_matches_narrow_panel_bits(
    monkeypatch: pytest.MonkeyPatch,
    threads: int,
    m: int,
    n: int,
    k: int,
    panel: int,
) -> None:
    """Shapes on which the rule fires give the bits of 64-column panels, over
    three calls, fused epilogue and affine linears included."""
    from helion_mlir_utils.linear import linear_affine

    linear_module = importlib.import_module("helion_mlir_utils.linear")

    monkeypatch.setenv("OMP_NUM_THREADS", str(threads))
    assert mm.vnni_panel(m, n, k) == panel
    torch.manual_seed(m + n + k)
    x = (torch.rand(m, k) - 0.5).to(torch.bfloat16)
    layer = torch.nn.Linear(k, n).to(torch.bfloat16)
    wide = narrow = None
    for _ in range(3):
        y, wide = linear(x, layer, torch.sigmoid, cache=wide)
        y64, narrow = linear(x, layer, torch.sigmoid, cache=narrow, panel=64)
        assert int(wide.packed_weight.shape[2]) == panel
        assert torch.equal(y.float(), y64.float())
    scale = torch.tensor([1.5], dtype=torch.bfloat16)
    y, _ = linear_affine(x, layer, scale)
    monkeypatch.setattr(linear_module, "vnni_panel", lambda *_: 64)
    y64, _ = linear_affine(x, layer, scale)
    assert torch.equal(y.float(), y64.float())


@pytest.mark.parametrize(("m", "n"), [(32, 1024), (40, 1000)])
def test_wide_panels_match_default_bits(m: int, n: int) -> None:
    """A 512-column panel (one tile per thread) gives the bits of 64-column
    panels, with partial rows and columns too: the f32 sums run in the same K
    order."""
    torch.manual_seed(m)
    k = 512
    x = torch.randn(m, k, dtype=torch.bfloat16)
    layer = torch.nn.Linear(k, n)
    wide, _ = linear(x, layer, panel=512)
    narrow, _ = linear(x, layer, panel=64)
    assert wide.shape == (m, n)
    assert torch.equal(wide.float(), narrow.float())


@pytest.mark.parametrize("k", [2048, 2080])
def test_one_k_chunk_matches_chunked_bits(k: int) -> None:
    """The whole K as one chunk (the prepared call's choice) gives the bits of
    1024-deep chunks: the f32 sums run in the same K order. 2080 adds a
    padded tail chunk after the one chunk."""
    import helion
    import helion.language as hl

    m, n, panel = 64, 256, 256
    torch.manual_seed(k)
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.bfloat16)
    b4 = mm.pack_b_vnni_t(w, panel)
    pairs_even = int(b4.shape[1]) // mm.BLOCK_K * mm.BLOCK_K
    assert mm._prepacked_k_chunk(pairs_even, panel, m) == pairs_even
    one = mm.prepacked_b_call(a, b4, n, bias, torch.relu)(a)
    kernel = helion.kernel(
        mm._matmul_prepacked_vnni.body,
        backend="mlir",
        config=helion.Config(block_sizes=[m, 1, 512, 32]),
        ignore_warnings=mm._OUT_CHECK_WARNINGS,
    )
    chunked = kernel(
        a.view(m, k // 2, 2),
        b4,
        bias.view(n // panel, panel),
        None,
        None,
        torch.relu,
        hl.constexpr(pairs_even // 512 * 512),
        hl.constexpr(pairs_even),
    ).view(m, n)
    assert torch.equal(one, chunked)
    expected = torch.relu(a.float() @ w.float().T + bias.float())
    error = (one.float() - expected).abs().max() / expected.abs().max()
    assert error.item() < 1e-2


def test_whole_row_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    # L2/45's second layer: 256 rows per thread, all four 256-column panels.
    assert mm.whole_row_tiles(16384, 1024) == (256, 256)
    # The f32 accumulator stays within 1 MiB: narrower panels for more rows.
    assert mm.whole_row_tiles(65536, 1024) == (1024, 256)
    assert mm.whole_row_tiles(131072, 1024) == (2048, 128)
    assert mm.whole_row_tiles(16384, 4160) == (256, 64)
    # Fewer than 256 rows per thread, rows that do not split evenly or into
    # whole register tiles, no dividing panel: None.
    assert mm.whole_row_tiles(128, 1024) is None
    assert mm.whole_row_tiles(8192, 1024) is None
    assert mm.whole_row_tiles(16384 + 64, 1024) is None
    assert mm.whole_row_tiles(16384 + 1, 1024) is None
    assert mm.whole_row_tiles(16384, 1000) is None
    # A given panel: its columns or nothing.
    assert mm.whole_row_tiles(16384, 1024, 128) == (256, 128)
    assert mm.whole_row_tiles(131072, 1024, 256) is None
    assert mm.whole_row_tiles(16384, 1024, 96) is None


def test_linear_whole_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """``whole_rows`` packs and tiles for one row of tiles per thread, with the
    default's bits; shapes it does not cover keep the default."""
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    torch.manual_seed(0)
    layer = torch.nn.Linear(256, 512).bfloat16()
    with torch.no_grad():
        for m, tiles in ((512, (256, 256)), (520, None)):
            x = torch.randn(m, 256, dtype=torch.bfloat16)
            ref, ref_cache = linear(x, layer, torch.relu)
            out, cache = linear(x, layer, torch.relu, whole_rows=True)
            assert torch.equal(out, ref)
            assert mm.whole_row_tiles(m, 512) == tiles
            panel = int(cache.packed_weight.shape[2])
            if tiles is None:
                assert panel == int(ref_cache.packed_weight.shape[2])
                continue
            assert panel == tiles[1]
            # The default runs one 512-row tile per panel here.
            assert mm._prepacked_tile_m(m, 512 // panel, panel) == 512
            assert any(
                key[0] == "prepacked" and key[1] == m and key[5] == tiles[0]
                for key in mm._FUSED_PACK_BOUND
            )
            # Later calls reuse the prepared call.
            out2, cache2 = linear(x, layer, torch.relu, cache, whole_rows=True)
            assert cache2.call is cache.call
            assert torch.equal(out2, ref)


def test_prepared_call_rejects_partial_register_tiles() -> None:
    a = torch.randn(64, 64, dtype=torch.bfloat16)
    b4 = mm.pack_b_vnni_t(torch.randn(64, 64, dtype=torch.bfloat16), 64)
    with pytest.raises(ValueError, match="tile_m"):
        mm.prepacked_b_call(a, b4, 64, tile_m=48)


def test_panel_rule_agrees_with_chunk_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """K = 2080 pads to 2080: 2048 in whole pairs of AMX steps plus a tail.
    The panel picker and the chunk rule both see one chunk at 256 columns."""
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    assert mm.vnni_panel(16384, 4096, 2080) == 256
    pairs_even = 2080 // 2 // mm.BLOCK_K * mm.BLOCK_K
    assert mm._prepacked_k_chunk(pairs_even, 256, 1024) == pairs_even
    # 4128 in the same way: 4096 at 128 columns.
    assert mm.vnni_panel(16384, 4096, 4128) == 128


@pytest.mark.parametrize(
    ("k", "chunk", "forced"),
    [
        (8224, 2048, 1024),  # two chunks and a padded tail
        (4192, 1056, 512),  # chunk, rest chunk and a padded tail
    ],
)
def test_two_k_chunks_and_tail_match_forced_chunks(
    monkeypatch: pytest.MonkeyPatch, k: int, chunk: int, forced: int
) -> None:
    """K past the one-chunk limit: the chosen chunks, the rest and the tail
    give the bits of other chunks, and the right result."""
    import helion
    import helion.language as hl

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    m, n, panel = 256, 64, 64
    torch.manual_seed(k)
    a = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.bfloat16)
    b4 = mm.pack_b_vnni_t(w, panel)
    pairs = int(b4.shape[1])
    pairs_even = pairs // mm.BLOCK_K * mm.BLOCK_K
    assert pairs > pairs_even, "a padded tail"
    assert mm._prepacked_k_chunk(pairs_even, panel, m) == chunk
    out = mm.prepacked_b_call(a, b4, n, bias, torch.relu)(a)
    chunked = pairs_even // forced * forced
    kernel = helion.kernel(
        mm._matmul_prepacked_vnni.body,
        backend="mlir",
        config=helion.Config(
            block_sizes=[m, 1, forced, max(pairs_even - chunked, mm.BLOCK_K)]
        ),
        ignore_warnings=mm._OUT_CHECK_WARNINGS,
    )
    other = kernel(
        a.view(m, k // 2, 2),
        b4,
        bias.view(n // panel, panel),
        None,
        None,
        torch.relu,
        hl.constexpr(chunked),
        hl.constexpr(pairs_even),
    ).view(m, n)
    assert torch.equal(out, other)
    expected = torch.relu(a.double() @ w.double().T + bias.double())
    error = (out.double() - expected).abs().max() / expected.abs().max()
    assert error.item() < 2e-2


def test_whole_rows_keep_the_accumulator_limit_after_a_thread_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packed for 8 threads (256x256 tiles), then run on one: 2048x256 would
    be a 2 MiB accumulator, so the default rows take over, with the same bits.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    torch.manual_seed(0)
    m, n = 2048, 256
    layer = torch.nn.Linear(64, n).bfloat16()
    x = torch.randn(m, 64, dtype=torch.bfloat16)
    mm._FUSED_PACK_BOUND.clear()
    with torch.no_grad():
        ref, cache = linear(x, layer, torch.relu, whole_rows=True)
        assert int(cache.packed_weight.shape[2]) == 256
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        assert mm.whole_row_tiles(m, n, 256) is None
        out, cache = linear(x, layer, torch.relu, cache, whole_rows=True)
    assert torch.equal(out, ref)
    assert int(cache.packed_weight.shape[2]) == 256
    rows = {key[5] for key in mm._FUSED_PACK_BOUND if key[0] == "prepacked"}
    assert all(tile * 256 * 4 <= 1 << 20 for tile in rows)
    assert rows == {256, 1024}


def test_linear_panel_and_whole_rows_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``panel`` sets the columns, ``whole_rows`` the rows; each is in the key
    of the cache, and the bits stay those of the default."""
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    torch.manual_seed(0)
    m, n = 512, 512
    layer = torch.nn.Linear(256, n).bfloat16()
    x = torch.randn(m, 256, dtype=torch.bfloat16)
    mm._FUSED_PACK_BOUND.clear()
    with torch.no_grad():
        ref, _ = linear(x, layer, torch.relu)
        out, cache = linear(x, layer, torch.relu, panel=128, whole_rows=True)
        assert torch.equal(out, ref)
        assert int(cache.packed_weight.shape[2]) == 128
        assert any(
            key[0] == "prepacked" and key[3] == 128 and key[5] == 256
            for key in mm._FUSED_PACK_BOUND
        )
        # Either argument changing rebuilds the cache.
        out, panel_only = linear(x, layer, torch.relu, cache, panel=128)
        assert panel_only.key != cache.key
        assert panel_only.packed_weight is not cache.packed_weight
        assert torch.equal(out, ref)
        out, rows_only = linear(x, layer, torch.relu, cache, whole_rows=True)
        assert rows_only.key != cache.key
        assert int(rows_only.packed_weight.shape[2]) == 256
        assert torch.equal(out, ref)
        # Same arguments: the prepared call is reused.
        out, again = linear(x, layer, torch.relu, cache, panel=128, whole_rows=True)
        assert again.call is cache.call
        assert torch.equal(out, ref)
        # A panel not dividing N has no whole-row tiles: the default rows.
        assert mm.whole_row_tiles(m, n, 96) is None
        out, odd = linear(x, layer, torch.relu, panel=96, whole_rows=True)
        assert int(odd.packed_weight.shape[2]) == 96
        assert torch.equal(out, ref)


@pytest.mark.parametrize(
    "threads,m,n,k,panel",
    [
        # Panels that fit, and the few-rows rules.
        (64, 1024, 8192, 1024, 256),
        (64, 1024, 8192, 8192, 256),
        (64, 16384, 4096, 2048, 256),
        (64, 16384, 1024, 4096, 128),
        (64, 128, 4096, 4096, 128),
        (64, 128, 8192, 4096, 64),
        (64, 128, 32768, 32768, 512),
    ],
)
def test_vnni_panel_with_a_fitting_panel(
    monkeypatch: pytest.MonkeyPatch, threads: int, m: int, n: int, k: int, panel: int
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", str(threads))
    assert mm.vnni_panel(m, n, k) == panel


@pytest.mark.parametrize("panel", [None, 64, 128, 256])
def test_linear_chain_any_panel_matches_fp64(panel: int | None) -> None:
    """AI-bench L3/3's shape class: the result must not depend on the tile the
    panel sets (checked against fp64)."""
    torch.manual_seed(7)
    layer = torch.nn.Linear(1024, 1024).to(torch.bfloat16)
    x = torch.rand(512, 1024).to(torch.bfloat16)
    out, _ = linear(x, layer, torch.relu, panel=panel)
    ref = torch.relu(
        x.to(torch.float64) @ layer.weight.detach().to(torch.float64).T
        + layer.bias.detach().to(torch.float64)
    )
    error = ((out.double() - ref).abs().max() / ref.abs().max()).item()
    assert error < 1e-2


@pytest.mark.parametrize(
    ("m", "n", "k", "threads", "expected"),
    [
        (1048576, 768, 256, 64, (256, 768, 256)),  # AI-bench L1/11 bench-cpu
        (16384, 768, 256, 64, (256, 768, 256)),  # one tile per thread
        (16320, 768, 256, 64, None),  # fewer tiles than threads
        (1048320, 768, 256, 64, (256, 768, 256)),
        (1048321, 768, 256, 64, None),  # a partial row tile
        (1048576, 760, 256, 64, None),  # partial register tiles
        (1048576, 768, 264, 64, None),  # a K tail
        (1048576, 2048, 512, 64, None),  # a 2 MiB B chunk
        (1048576, 2048, 256, 64, None),  # 128 rows: too few to share
    ],
)
def test_tall_output_tiles(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
    n: int,
    k: int,
    threads: int,
    expected: tuple[int, int, int] | None,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", str(threads))
    assert mm.tall_output_tiles(m, n, k) == expected
