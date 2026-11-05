"""Tests for MLIR kernel execution via lighthouse (CPU).

Covers both execution paths:
- ``execute_mlir``  – explicit generate_mlir + MLIRBackend.execute_mlir
- ``direct call``   – @helion.kernel(backend="mlir") called like a normal Python function
"""

from __future__ import annotations

import math
import re

import helion
import helion.language as hl
import pytest
import torch

from tests.inputs import boundary_tile
from tests.inputs import same_values
from tests.ir_query import ops_named
from tests.ir_query import static_sizes

from helion_mlir_backend import generate_mlir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend():
    from helion._compiler.backend_registry import get_backend_class

    return get_backend_class("mlir")()


def _tile_rows(module: object) -> set[int]:
    """The row counts of the tiles read from the kernel's inputs."""
    return {static_sizes(op)[0] for op in ops_named(module, "tensor.extract_slice")}


def _allclose(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ab_32x32():
    torch.manual_seed(0)
    return torch.randn(32, 32), torch.randn(32, 32)


@pytest.fixture
def abc_16x64():
    torch.manual_seed(1)
    return torch.randn(16, 64), torch.randn(16, 64), torch.randn(16, 64)


# ---------------------------------------------------------------------------
# execute_mlir path
# ---------------------------------------------------------------------------


class TestExecuteMlir:
    """Tests for MLIRBackend.execute_mlir (explicit two-step API)."""

    def test_execute_mlir_rejects_empty_inputs(self):
        with pytest.raises(TypeError, match="non-empty"):
            _backend().execute_mlir(None)

    def test_execute_mlir_rejects_non_tensor_inputs(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            _backend().execute_mlir(None, torch.ones(2), 1)

    def test_execute_mlir_rejects_mixed_devices(self):
        cpu_tensor = torch.ones(2)
        meta_tensor = torch.ones(2, device="meta")
        with pytest.raises(ValueError, match="same device"):
            _backend().execute_mlir(None, cpu_tensor, meta_tensor)

    def test_add_execute_mlir(self, ab_32x32):
        """Elementwise add via execute_mlir matches torch reference."""
        A, B = ab_32x32

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(add_kernel, [A, B])
        C = _backend().execute_mlir(mlir_module, A, B, kernel_name="add_kernel")
        assert _allclose(C, A + B)

    def test_mul_execute_mlir(self, ab_32x32):
        """Elementwise multiply via execute_mlir matches torch reference."""
        A, B = ab_32x32

        @helion.kernel(static_shapes=True)
        def mul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(mul_kernel, [A, B])
        C = _backend().execute_mlir(mlir_module, A, B, kernel_name="mul_kernel")
        assert _allclose(C, A * B)

    def test_scale_add_execute_mlir(self, abc_16x64):
        """Fused scale-add (x * 2 + y) via execute_mlir."""
        X, Y, _ = abc_16x64

        @helion.kernel(static_shapes=True)
        def scale_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            # 1D tile over M to stay within the scalar-lowering pipeline.
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] * 2.0 + y[tile_m, :]
            return out

        mlir_module = generate_mlir(scale_add_kernel, [X, Y])
        C = _backend().execute_mlir(mlir_module, X, Y, kernel_name="scale_add_kernel")
        assert _allclose(C, X * 2.0 + Y)

    def test_output_shape_execute_mlir(self, ab_32x32):
        """Output tensor has correct shape and dtype."""
        A, B = ab_32x32

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(add_kernel, [A, B])
        C = _backend().execute_mlir(mlir_module, A, B, kernel_name="add_kernel")
        assert C.shape == A.shape
        assert C.dtype == torch.float32

    def test_matmul_full_rhs_slice_execute_mlir_regression(self):
        """Regression: full RHS slice in tiled matmul compiles and runs correctly.

        Mirrors the example pattern where B is loaded with full slices inside
        an outer tile loop and multiplied with a tiled slice of A.
        """

        @helion.kernel(static_shapes=True)
        def matmul_full_rhs_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            assert k == k2

            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                a_tile = hl.load(x, [tile_m, slice(None)])
                b_full = hl.load(y, [slice(None), slice(None)])
                c_tile = a_tile @ b_full
                hl.store(out, [tile_m, slice(None)], c_tile)
            return out

        torch.manual_seed(123)
        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        ref = a @ b

        mlir_module = generate_mlir(matmul_full_rhs_slice, [a, b])
        c = _backend().execute_mlir(
            mlir_module,
            a,
            b,
            kernel_name="matmul_full_rhs_slice",
        )

        assert torch.isfinite(c).all()
        assert _allclose(c, ref)

    def test_concat2d_dim1_simple_generate_mlir_regression(self):
        """Regression: concat2d_dim1_simple pattern lowers and emits kernel symbol."""

        @helion.kernel(static_shapes=True)
        def concat2d_dim1_simple_kernel(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            assert x.size(0) == y.size(0)
            out = torch.empty(
                [x.size(0), x.size(1) + y.size(1)],
                dtype=x.dtype,
                device=x.device,
            )
            n1 = x.size(1)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :n1] = x[tile_m, :]
                out[tile_m, n1:] = y[tile_m, :]
            return out

        torch.manual_seed(29)
        x = torch.randn(64, 40)
        y = torch.randn(64, 24)

        mlir_module = generate_mlir(concat2d_dim1_simple_kernel, [x, y])
        assert "concat2d_dim1_simple_kernel" in str(mlir_module)
        # TODO(helion-mlir): Runtime broken

    def test_broadcast_matmul_generate_mlir_regression(self):
        """Regression: broadcasted batch matmul kernel lowers and emits the kernel symbol."""

        @helion.kernel(static_shapes=True)
        def broadcast_matmul_kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            b, m, k = x.size()
            k2, n = w.size()
            assert k == k2

            x_2d = x.reshape([b * m, k])
            out_2d = torch.empty(
                [b * m, n],
                device=x.device,
                dtype=torch.promote_types(x.dtype, w.dtype),
            )
            for tile_bm, tile_n in hl.tile([b * m, n]):
                acc = hl.zeros([tile_bm, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x_2d[tile_bm, tile_k], w[tile_k, tile_n])
                out_2d[tile_bm, tile_n] = acc

            return out_2d.view(b, m, n)

        torch.manual_seed(41)
        x = torch.randn(4, 8, 16)
        w = torch.randn(16, 12)

        mlir_module = generate_mlir(broadcast_matmul_kernel, [x, w])
        assert "broadcast_matmul_kernel" in str(mlir_module)
        # TODO(helion-mlir): Runtime broken

    def test_flat_gather_row_sum_indexing_compile(self):
        """Regression: flattened gather + reduction with standard hl.tile matches reference."""

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[32, 16]),
        )
        def flat_gather_row_sum_kernel(
            x_data: torch.Tensor,
            flat_idx: torch.Tensor,
        ) -> torch.Tensor:
            num_rows, m = x_data.shape
            x_flat = x_data.view(-1)
            out = torch.zeros((num_rows,), dtype=x_data.dtype, device=x_data.device)

            for tile_b in hl.tile(num_rows):
                idx_tile = hl.load(flat_idx, [tile_b, slice(None)])
                x_slice = hl.load(x_flat, [idx_tile])
                out[tile_b] = x_slice.sum(dim=1)
            return out

        torch.manual_seed(7)
        x_data = torch.randn(64, 16)
        row_ids = torch.arange(x_data.shape[0], dtype=torch.int64)[:, None]
        col_ids = torch.arange(x_data.shape[1], dtype=torch.int64)[None, :]
        flat_idx = row_ids * x_data.shape[1] + col_ids
        expected = x_data.sum(dim=1)

        module = generate_mlir(flat_gather_row_sum_kernel, [x_data, flat_idx])
        actual = _backend().execute_mlir(
            module,
            x_data,
            flat_idx,
            kernel_name="flat_gather_row_sum_kernel",
        )

        assert _allclose(actual, expected)

    def test_flat_gather_2d_copy_indexing_compile(self):
        """Regression: flattened 2D gather copy with standard hl.tile preserves values."""

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[32, 16]),
        )
        def flat_gather_2d_copy_kernel(
            x_data: torch.Tensor,
            flat_idx: torch.Tensor,
        ) -> torch.Tensor:
            num_rows, m = x_data.shape
            x_flat = x_data.view(-1)
            out = torch.empty((num_rows, m), dtype=x_data.dtype, device=x_data.device)

            for tile_b in hl.tile(num_rows):
                idx_tile = hl.load(flat_idx, [tile_b, slice(None)])
                out[tile_b, :] = hl.load(x_flat, [idx_tile])
            return out

        torch.manual_seed(11)
        x_data = torch.randn(64, 16)
        row_ids = torch.arange(x_data.shape[0], dtype=torch.int64)[:, None]
        col_ids = torch.arange(x_data.shape[1], dtype=torch.int64)[None, :]
        flat_idx = row_ids * x_data.shape[1] + col_ids

        module = generate_mlir(flat_gather_2d_copy_kernel, [x_data, flat_idx])
        actual = _backend().execute_mlir(
            module,
            x_data,
            flat_idx,
            kernel_name="flat_gather_2d_copy_kernel",
        )

        assert _allclose(actual, x_data)

    def test_tile_index_usage_execute_mlir(self):
        """tile.index can be used for indexed load/store in a 1D tiled loop."""

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[16]),
        )
        def tile_index_kernel(x: torch.Tensor) -> torch.Tensor:
            n = x.shape[0]
            out = torch.empty((n,), dtype=x.dtype, device=x.device)

            for tile_n in hl.tile(n):
                out[tile_n] = hl.load(x, [tile_n.index]) + 1.0

            return out

        torch.manual_seed(23)
        x = torch.randn(64)
        expected = x + 1.0

        module = generate_mlir(tile_index_kernel, [x])
        actual = _backend().execute_mlir(
            module,
            x,
            kernel_name="tile_index_kernel",
        )

        assert _allclose(actual, expected)

    def test_grid_batched_matmul_execute_mlir(self):
        """`hl.grid` scalar indices produce numerically correct batched matmul."""

        @helion.kernel(static_shapes=True)
        def grid_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            nb, m, k = a.shape
            out = torch.zeros((nb, m, b.shape[2]), dtype=torch.float32, device=a.device)
            for i in hl.grid(nb):
                out[i, :, :] = torch.matmul(a[i, :, :], b[i, :, :])
            return out

        torch.manual_seed(7)
        a = torch.randn(4, 32, 32)
        b = torch.randn(4, 32, 32)

        module = generate_mlir(grid_bmm, [a, b])
        actual = _backend().execute_mlir(module, a, b, kernel_name="grid_bmm")

        assert _allclose(actual, torch.bmm(a, b))

    def test_block_packing_kernels_execute_mlir(self):
        """Nested tiled A/B panel packing matches contiguous PyTorch references."""
        from benchmarks.helion_block_pack import pack_a
        from benchmarks.helion_block_pack import pack_a_panels
        from benchmarks.helion_block_pack import pack_b
        from benchmarks.helion_block_pack import pack_b_panels

        torch.manual_seed(41)
        a = torch.randn(64, 64, dtype=torch.bfloat16)
        b = torch.randn(64, 64, dtype=torch.bfloat16)

        packed_a = pack_a(a, 32)
        packed_b = pack_b(b, 32)

        torch.testing.assert_close(
            packed_a,
            a.view(64, 2, 32).permute(1, 0, 2).contiguous(),
        )
        torch.testing.assert_close(
            packed_b,
            b.view(64, 2, 32).permute(1, 0, 2).contiguous(),
        )

        b_module = generate_mlir(pack_b_panels, [b.view(64, 2, 32).contiguous()])
        b_ir = str(b_module)
        assert re.search(r"scf\.forall \(%\w+, %\w+, %\w+\)", b_ir)
        assert (
            "affine_map<(d0) -> (d0 * 8)>" in b_ir
            and "affine_map<(d0) -> (d0 * 32)>" in b_ir
        )
        assert "linalg.transpose" in b_ir
        assert "permutation = [1, 0, 2]" in b_ir

        a_module = generate_mlir(pack_a_panels, [a.view(64, 2, 32).contiguous()])
        a_ir = str(a_module)
        assert re.search(r"scf\.forall \(%\w+, %\w+, %\w+\)", a_ir)
        assert (
            "affine_map<(d0) -> (d0 * 8)>" in a_ir
            and "affine_map<(d0) -> (d0 * 32)>" in a_ir
        )
        assert "linalg.transpose" in a_ir
        assert "permutation = [1, 0, 2]" in a_ir

    def test_unpack_grid_tile_reordered_store_execute_mlir(self):
        """Store index order can differ from loop declaration order.

        ``out[tm, panel, :]`` writes the tile index before the grid index even
        though the grid loop is declared first; the grid block id must still
        land on output dimension 1 (not 0).
        """

        @helion.kernel(static_shapes=True)
        def unpack_panels(src: torch.Tensor) -> torch.Tensor:
            n_panels, m, bn = src.shape
            out = torch.empty((m, n_panels, bn), dtype=src.dtype, device=src.device)
            for panel in hl.grid(n_panels):
                for tm in hl.tile(m):
                    out[tm, panel, :] = src[panel, tm, :]
            return out

        torch.manual_seed(37)
        src = torch.randn(3, 8, 8)
        module = generate_mlir(
            unpack_panels,
            [src],
            config=helion.Config(block_sizes=[1, 4]),
        )
        actual = _backend().execute_mlir(module, src, kernel_name="unpack_panels")

        assert _allclose(actual, src.permute(1, 0, 2).contiguous())

    def test_unpack_triple_nested_grid_execute_mlir(self):
        """``grid -> grid -> tile`` (3 levels deep) chains accumulators correctly.

        Each level's synthetic accumulator must flush into its parent's at the
        right offset, with ancestor dimensions already reduced to a single
        local slot forced to offset 0 rather than an out-of-scope/absolute iv.
        """

        @helion.kernel(static_shapes=True)
        def unpack_triple_nested(src: torch.Tensor) -> torch.Tensor:
            mb, np_, bm, bn = src.shape
            out = torch.empty((mb, bm, np_, bn), dtype=src.dtype, device=src.device)
            for m_block in hl.grid(mb):
                for panel in hl.grid(np_):
                    for tile_m in hl.tile(bm):
                        out[m_block, tile_m, panel, :] = src[m_block, panel, tile_m, :]
            return out

        torch.manual_seed(53)
        src = torch.randn(2, 3, 4, 8)
        module = generate_mlir(
            unpack_triple_nested,
            [src],
            config=helion.Config(block_sizes=[1, 1, 4]),
        )
        actual = _backend().execute_mlir(
            module, src, kernel_name="unpack_triple_nested"
        )

        assert _allclose(actual, src.permute(0, 2, 1, 3).contiguous())

    def test_grid_combined_2d_tile_execute_mlir(self):
        """``grid -> tile([a, b])`` (one loop node, two block ids) lowers to
        nested ``scf.for`` loops, one per combined tile dimension."""

        @helion.kernel(static_shapes=True)
        def copy_grid_combined_tile(src: torch.Tensor) -> torch.Tensor:
            mb, np_, bm, bn = src.shape
            out = torch.empty((mb, np_, bm, bn), dtype=src.dtype, device=src.device)
            for m_block in hl.grid(mb):
                for tp, tm in hl.tile([np_, bm]):
                    out[m_block, tp, tm, :] = src[m_block, tp, tm, :]
            return out

        torch.manual_seed(59)
        src = torch.randn(2, 3, 4, 8)
        module = generate_mlir(
            copy_grid_combined_tile,
            [src],
            config=helion.Config(block_sizes=[1, 4, 1]),
        )
        actual = _backend().execute_mlir(
            module, src, kernel_name="copy_grid_combined_tile"
        )

        assert _allclose(actual, src)

    def test_scalar_grid_index_transpose_execute_mlir(self):
        """``.transpose()`` on a value loaded via a scalar ``hl.grid`` index.

        Exercises the ATen-helper prebuild path for a permute/transpose op
        whose input is a scalar-index-reduced load (previously mismatched
        rank against Helion's own tracked shape).
        """

        @helion.kernel(static_shapes=True)
        def grid_load_transpose(src: torch.Tensor) -> torch.Tensor:
            nb, m, n = src.shape
            out = torch.empty((nb, n, m), dtype=src.dtype, device=src.device)
            for i in hl.grid(nb):
                out[i, :, :] = src[i, :, :].transpose(0, 1)
            return out

        torch.manual_seed(61)
        src = torch.randn(3, 5, 7)
        module = generate_mlir(
            grid_load_transpose,
            [src],
            config=helion.Config(block_sizes=[1]),
        )
        actual = _backend().execute_mlir(module, src, kernel_name="grid_load_transpose")

        assert _allclose(actual, src.transpose(1, 2))

    def test_grid_combined_2d_tile_explicit_transpose_execute_mlir(self):
        """``grid -> tile([a, b])`` with an explicit ``.permute()`` reconciling
        a transposed store — the Helion-idiomatic way to write an unpack that
        reorders dimensions between load and store (as opposed to relying on
        differing index order, which is not valid Helion semantics)."""

        @helion.kernel(static_shapes=True)
        def unpack_combined_tile_transpose(src: torch.Tensor) -> torch.Tensor:
            mb, np_, bm, bn = src.shape
            out = torch.empty((mb, bm, np_, bn), dtype=src.dtype, device=src.device)
            for m_block in hl.grid(mb):
                for tm, tp in hl.tile([bm, np_]):
                    loaded = src[m_block, tp, tm, :]
                    out[m_block, tm, tp, :] = loaded.permute(1, 0, 2)
            return out

        torch.manual_seed(67)
        src = torch.randn(2, 3, 4, 8)
        module = generate_mlir(
            unpack_combined_tile_transpose,
            [src],
            config=helion.Config(block_sizes=[1, 4, 1]),
        )
        actual = _backend().execute_mlir(
            module, src, kernel_name="unpack_combined_tile_transpose"
        )

        assert _allclose(actual, src.permute(0, 2, 1, 3).contiguous())

    def test_grid_combined_tile_separate_reduction_execute_mlir(self):
        """``grid -> tile([m, n]) -> tile(k)`` batched matmul: the combined
        tile's own loop node never carries an accumulator (Helion always
        gives a reduction its own separate, single-block loop nested inside),
        so this validates that pattern executes correctly end to end."""

        @helion.kernel(static_shapes=True)
        def batched_matmul_grid_combined_tile(
            a: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            nb, m, k = a.shape
            _, k2, n = b.shape
            out = torch.zeros((nb, m, n), dtype=torch.float32, device=a.device)
            for i in hl.grid(nb):
                for tm, tn in hl.tile([m, n]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(k):
                        acc = torch.addmm(acc, a[i, tm, tk], b[i, tk, tn])
                    out[i, tm, tn] = acc
            return out

        torch.manual_seed(71)
        a = torch.randn(2, 8, 8)
        b = torch.randn(2, 8, 8)
        module = generate_mlir(
            batched_matmul_grid_combined_tile,
            [a, b],
            config=helion.Config(block_sizes=[1, 4, 4, 4]),
        )
        actual = _backend().execute_mlir(
            module, a, b, kernel_name="batched_matmul_grid_combined_tile"
        )

        assert _allclose(actual, torch.bmm(a, b))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_unpack_grid_tile_reordered_store_dtypes(self, dtype):
        """Reordered-store unpack (Form 2) stays correct across dtypes."""

        @helion.kernel(static_shapes=True)
        def unpack_panels_dtype(src: torch.Tensor) -> torch.Tensor:
            n_panels, m, bn = src.shape
            out = torch.empty((m, n_panels, bn), dtype=src.dtype, device=src.device)
            for panel in hl.grid(n_panels):
                for tm in hl.tile(m):
                    out[tm, panel, :] = src[panel, tm, :]
            return out

        torch.manual_seed(73)
        src = torch.randn(3, 8, 8).to(dtype)
        module = generate_mlir(
            unpack_panels_dtype,
            [src],
            config=helion.Config(block_sizes=[1, 4]),
        )
        actual = _backend().execute_mlir(module, src, kernel_name="unpack_panels_dtype")

        assert _allclose(actual, src.permute(1, 0, 2).contiguous())

    @pytest.mark.parametrize("shape", [(2, 3, 4, 8), (3, 2, 5, 6), (2, 4, 3, 7)])
    def test_unpack_triple_nested_grid_shapes(self, shape):
        """Triple-nested-grid unpack (Form 1) across non-uniform shapes."""

        @helion.kernel(static_shapes=True)
        def unpack_triple_nested_shape(src: torch.Tensor) -> torch.Tensor:
            mb, np_, bm, bn = src.shape
            out = torch.empty((mb, bm, np_, bn), dtype=src.dtype, device=src.device)
            for m_block in hl.grid(mb):
                for panel in hl.grid(np_):
                    for tile_m in hl.tile(bm):
                        out[m_block, tile_m, panel, :] = src[m_block, panel, tile_m, :]
            return out

        torch.manual_seed(sum(shape) + 100)
        src = torch.randn(*shape)
        module = generate_mlir(
            unpack_triple_nested_shape,
            [src],
            config=helion.Config(block_sizes=[1, 1, 2]),
        )
        actual = _backend().execute_mlir(
            module, src, kernel_name="unpack_triple_nested_shape"
        )

        assert _allclose(actual, src.permute(0, 2, 1, 3).contiguous())

    @pytest.mark.parametrize(
        "block_sizes",
        [
            [1, 4, 1],  # tm covers full extent in one shot, tp one-per-iteration
            [1, 1, 3],  # tm one-per-iteration, tp covers full extent in one shot
            [1, 2, 2],  # both dims iterate multiple times
            [1, 8, 8],  # both dims larger than their extent (clamped)
        ],
    )
    def test_grid_combined_2d_tile_block_sizes(self, block_sizes):
        """Combined 2D tile (Form 3) stays correct across block-size configs
        stressing the multi-block disambiguation and offset-clamping logic."""

        @helion.kernel(static_shapes=True)
        def copy_grid_combined_tile_bs(src: torch.Tensor) -> torch.Tensor:
            mb, np_, bm, bn = src.shape
            out = torch.empty((mb, np_, bm, bn), dtype=src.dtype, device=src.device)
            for m_block in hl.grid(mb):
                for tp, tm in hl.tile([np_, bm]):
                    out[m_block, tp, tm, :] = src[m_block, tp, tm, :]
            return out

        torch.manual_seed(79)
        src = torch.randn(2, 3, 4, 8)
        module = generate_mlir(
            copy_grid_combined_tile_bs,
            [src],
            config=helion.Config(block_sizes=block_sizes),
        )
        actual = _backend().execute_mlir(
            module, src, kernel_name="copy_grid_combined_tile_bs"
        )

        assert _allclose(actual, src)

    def test_grid_grid_combined_2d_tile_execute_mlir(self):
        """``grid -> grid -> tile([a, b])``: arbitrary-depth recursion and
        multi-block disambiguation stressed together in one kernel."""

        @helion.kernel(static_shapes=True)
        def copy_grid_grid_combined_tile(src: torch.Tensor) -> torch.Tensor:
            mb, nb, bm, bn = src.shape
            out = torch.empty((mb, nb, bm, bn), dtype=src.dtype, device=src.device)
            for i in hl.grid(mb):
                for j in hl.grid(nb):
                    for tm, tn in hl.tile([bm, bn]):
                        out[i, j, tm, tn] = src[i, j, tm, tn] + 1.0
            return out

        torch.manual_seed(83)
        src = torch.randn(2, 3, 4, 5)
        module = generate_mlir(
            copy_grid_grid_combined_tile,
            [src],
            config=helion.Config(block_sizes=[1, 1, 2, 3]),
        )
        actual = _backend().execute_mlir(
            module, src, kernel_name="copy_grid_grid_combined_tile"
        )

        assert _allclose(actual, src + 1.0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_grid_combined_2d_tile_explicit_transpose_dtypes(self, dtype):
        """Explicit-permute combined-tile unpack stays correct across dtypes."""

        @helion.kernel(static_shapes=True)
        def unpack_combined_tile_transpose_dtype(src: torch.Tensor) -> torch.Tensor:
            mb, np_, bm, bn = src.shape
            out = torch.empty((mb, bm, np_, bn), dtype=src.dtype, device=src.device)
            for m_block in hl.grid(mb):
                for tm, tp in hl.tile([bm, np_]):
                    loaded = src[m_block, tp, tm, :]
                    out[m_block, tm, tp, :] = loaded.permute(1, 0, 2)
            return out

        torch.manual_seed(89)
        src = torch.randn(2, 3, 4, 8).to(dtype)
        module = generate_mlir(
            unpack_combined_tile_transpose_dtype,
            [src],
            config=helion.Config(block_sizes=[1, 4, 1]),
        )
        actual = _backend().execute_mlir(
            module, src, kernel_name="unpack_combined_tile_transpose_dtype"
        )

        assert _allclose(actual, src.permute(0, 2, 1, 3).contiguous())

    @pytest.mark.parametrize("shape", [(4, 6), (5, 3), (8, 8)])
    def test_scalar_grid_index_transpose_shapes(self, shape):
        """Grid-index-scoped transpose across non-square/square 2D tiles."""

        @helion.kernel(static_shapes=True)
        def grid_load_transpose_shape(src: torch.Tensor) -> torch.Tensor:
            nb, m, n = src.shape
            out = torch.empty((nb, n, m), dtype=src.dtype, device=src.device)
            for i in hl.grid(nb):
                out[i, :, :] = src[i, :, :].t()
            return out

        torch.manual_seed(sum(shape) + 200)
        src = torch.randn(3, *shape)
        module = generate_mlir(
            grid_load_transpose_shape,
            [src],
            config=helion.Config(block_sizes=[1]),
        )
        actual = _backend().execute_mlir(
            module, src, kernel_name="grid_load_transpose_shape"
        )

        assert _allclose(actual, src.transpose(1, 2))

    @pytest.mark.parametrize(
        "block_sizes",
        [[1, 4, 4, 4], [1, 2, 2, 2], [1, 8, 8, 8]],
    )
    def test_grid_combined_tile_separate_reduction_block_sizes(self, block_sizes):
        """Batched matmul via combined tile + separate reduction loop across
        block-size configs (exact tiling, sub-tiling, and single-shot)."""

        @helion.kernel(static_shapes=True)
        def batched_matmul_bs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            nb, m, k = a.shape
            _, k2, n = b.shape
            out = torch.zeros((nb, m, n), dtype=torch.float32, device=a.device)
            for i in hl.grid(nb):
                for tm, tn in hl.tile([m, n]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(k):
                        acc = torch.addmm(acc, a[i, tm, tk], b[i, tk, tn])
                    out[i, tm, tn] = acc
            return out

        torch.manual_seed(97)
        a = torch.randn(2, 8, 8)
        b = torch.randn(2, 8, 8)
        module = generate_mlir(
            batched_matmul_bs,
            [a, b],
            config=helion.Config(block_sizes=block_sizes),
        )
        actual = _backend().execute_mlir(module, a, b, kernel_name="batched_matmul_bs")

        assert _allclose(actual, torch.bmm(a, b))

    @pytest.mark.parametrize("size", [64, 128])
    def test_packed_rhs_matmul_execute_mlir(self, size):
        """A packed RHS remains numerically correct through tiled matmul consumption."""
        from benchmarks.helion_block_packed_f32_repro import matmul_packed_b_f32
        from benchmarks.helion_block_packed_f32_repro import pack_b_panels_f32
        from benchmarks.helion_block_packed_f32_repro import unpack_panel_major

        torch.manual_seed(43)
        block_n = 32
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        b3 = b.view(size, size // block_n, block_n).contiguous()

        packed_b = pack_b_panels_f32(b3)
        panel_result = matmul_packed_b_f32(a, packed_b)
        actual = unpack_panel_major(panel_result)

        assert _allclose(actual, torch.mm(a, b))

    def test_nested_grid_copy_execute_mlir(self):
        """Nested unit-step grid indices preserve both outer dimensions."""

        @helion.kernel(static_shapes=True)
        def nested_grid_copy(a: torch.Tensor) -> torch.Tensor:
            nb, nc, m = a.shape
            out = torch.zeros_like(a)
            for i in hl.grid(nb):
                for j in hl.grid(nc):
                    out[i, j, :] = a[i, j, :] + 1.0
            return out

        torch.manual_seed(31)
        a = torch.randn(2, 3, 8)
        module = generate_mlir(
            nested_grid_copy,
            [a],
            config=helion.Config(block_sizes=[1, 1, 8]),
        )
        actual = _backend().execute_mlir(module, a, kernel_name="nested_grid_copy")

        assert _allclose(actual, a + 1.0)

    @pytest.mark.parametrize("shape", [(2, 3, 8), (3, 2, 10)])
    def test_grid_tile_slice_execute_mlir(self, shape):
        """Scalar grid plus exact/ragged trailing tile slices stay numerical."""

        @helion.kernel(static_shapes=True)
        def grid_tile_copy(a: torch.Tensor) -> torch.Tensor:
            nb, nc, m = a.shape
            out = torch.zeros_like(a)
            for i in hl.grid(nb):
                for tm in hl.tile(m):
                    out[i, :, tm] = a[i, :, tm] * 2.0 + 1.0
            return out

        torch.manual_seed(sum(shape))
        a = torch.randn(*shape)
        module = generate_mlir(
            grid_tile_copy,
            [a],
            config=helion.Config(block_sizes=[1, 1, 4]),
        )
        actual = _backend().execute_mlir(module, a, kernel_name="grid_tile_copy")

        assert _allclose(actual, a * 2.0 + 1.0)

    def test_tile_begin_execute_mlir(self):
        """`tile.begin` contributes the correct per-tile offset value."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[16]))
        def tile_begin_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm in hl.tile(m):
                out[tm, :] = x[tm, :] + tm.begin
            return out

        torch.manual_seed(11)
        x = torch.randn(64, 32)

        module = generate_mlir(
            tile_begin_kernel, [x], config=helion.Config(block_sizes=[16])
        )
        actual = _backend().execute_mlir(module, x, kernel_name="tile_begin_kernel")

        offsets = (torch.arange(64) // 16 * 16).to(torch.float32).unsqueeze(1)
        assert _allclose(actual, x + offsets)

    @pytest.mark.parametrize(
        "block_sizes", [[16, 32, 64], [32, 32, 32], [16, 16, 16], [32, 64, 32]]
    )
    def test_nested_tile_block_sizes_execute_mlir(self, block_sizes):
        """Repeated nested block sizes still compute a correct matmul."""

        @helion.kernel(static_shapes=True)
        def mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm, tn in hl.tile([m, n]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], y[tk, tn])
                out[tm, tn] = acc
            return out

        torch.manual_seed(3)
        x = torch.randn(128, 128)
        y = torch.randn(128, 128)

        module = generate_mlir(
            mm, [x, y], config=helion.Config(block_sizes=block_sizes)
        )
        actual = _backend().execute_mlir(module, x, y, kernel_name="mm")

        assert _allclose(actual, x @ y)

    def test_transposed_rhs_matmul_execute_mlir(self):
        """A transposed RHS lowered via linalg.contract is numerically correct."""

        @helion.kernel(static_shapes=True)
        def mm_transposed_b(x: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            n, k2 = yt.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tm, tn in hl.tile([m, n]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, x[tm, tk], yt[tn, tk].permute(1, 0))
                out[tm, tn] = acc
            return out

        torch.manual_seed(5)
        x = torch.randn(64, 64)
        yt = torch.randn(64, 64)

        module = generate_mlir(
            mm_transposed_b, [x, yt], config=helion.Config(block_sizes=[16, 32, 64])
        )
        actual = _backend().execute_mlir(module, x, yt, kernel_name="mm_transposed_b")

        assert _allclose(actual, x @ yt.t())

    def test_dtype_cast_epilogue_execute_mlir(self):
        """An explicit `.to(bfloat16)` epilogue produces a bf16 result."""

        @helion.kernel(static_shapes=True)
        def cast_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            for tm, tn in hl.tile([m, n]):
                out[tm, tn] = (x[tm, tn] * 2.0).to(torch.bfloat16)
            return out

        torch.manual_seed(13)
        x = torch.randn(64, 64)

        module = generate_mlir(cast_kernel, [x])
        actual = _backend().execute_mlir(module, x, kernel_name="cast_kernel")

        assert actual.dtype == torch.bfloat16
        assert _allclose(actual, (x * 2.0).to(torch.bfloat16))

    def test_cpu_only_guard(self, ab_32x32):
        """execute_mlir raises NotImplementedError for non-CPU tensors."""
        A, B = ab_32x32
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(add_kernel, [A, B])
        with pytest.raises(NotImplementedError, match="CPU"):
            _backend().execute_mlir(
                mlir_module, A.cuda(), B.cuda(), kernel_name="add_kernel"
            )


# ---------------------------------------------------------------------------
# Direct-call path (backend="mlir" integrated into helion runtime)
# ---------------------------------------------------------------------------


class TestDirectCall:
    """Tests for @helion.kernel(backend="mlir") called directly."""

    def test_add_direct(self, ab_32x32):
        """Direct add kernel call matches torch reference."""
        A, B = ab_32x32

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[32, 32]),
        )
        def add_direct(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        assert _allclose(add_direct(A, B), A + B)

    def test_mul_direct(self, ab_32x32):
        """Direct multiply kernel call matches torch reference."""
        A, B = ab_32x32

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[32, 32]),
        )
        def mul_direct(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * y[tile_m, tile_n]
            return out

        assert _allclose(mul_direct(A, B), A * B)

    def test_three_input_direct(self, abc_16x64):
        """Direct call with three tensor inputs."""
        X, Y, Z = abc_16x64

        @helion.kernel(
            static_shapes=True, backend="mlir", config=helion.Config(block_sizes=[16])
        )
        def add3_direct(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            # 1D tile over M to stay within the scalar-lowering pipeline.
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :] + z[tile_m, :]
            return out

        assert _allclose(add3_direct(X, Y, Z), X + Y + Z)

    def test_result_is_cached(self, ab_32x32):
        """Second call reuses the compiled JIT function (no recompilation)."""
        A, B = ab_32x32

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[32, 32]),
        )
        def add_cached(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        C1 = add_cached(A, B)
        C2 = add_cached(A, B)
        assert _allclose(C1, C2)

    def test_direct_matches_execute_mlir(self, ab_32x32):
        """Direct call and execute_mlir produce identical results."""
        A, B = ab_32x32

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[32, 32]),
        )
        def add_both(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        mlir_module = generate_mlir(add_both, [A, B])
        C_explicit = _backend().execute_mlir(mlir_module, A, B, kernel_name="add_both")
        C_direct = add_both(A, B)
        assert torch.equal(C_explicit, C_direct)

    def test_matmul_full_rhs_slice_direct_regression(self):
        """Regression: direct MLIR backend call handles full RHS slice matmul."""

        @helion.kernel(
            static_shapes=True, backend="mlir", config=helion.Config(block_sizes=[32])
        )
        def matmul_full_rhs_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            assert k == k2

            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                a_tile = hl.load(x, [tile_m, slice(None)])
                b_full = hl.load(y, [slice(None), slice(None)])
                c_tile = a_tile @ b_full
                hl.store(out, [tile_m, slice(None)], c_tile)
            return out

        torch.manual_seed(321)
        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        ref = a @ b

        c = matmul_full_rhs_slice(a, b)
        assert torch.isfinite(c).all()
        assert _allclose(c, ref)

    def test_unpack_combined_tile_transpose_direct(self):
        """Direct call (full runtime pipeline, not generate_mlir+execute_mlir)
        for a combined 2D tile with an explicit transpose reconciling the
        store, matching the reordered-unpack pattern used in production."""

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[1, 4, 1]),
        )
        def unpack_combined_tile_transpose_direct(src: torch.Tensor) -> torch.Tensor:
            mb, np_, bm, bn = src.shape
            out = torch.empty((mb, bm, np_, bn), dtype=src.dtype, device=src.device)
            for m_block in hl.grid(mb):
                for tm, tp in hl.tile([bm, np_]):
                    loaded = src[m_block, tp, tm, :]
                    out[m_block, tm, tp, :] = loaded.permute(1, 0, 2)
            return out

        torch.manual_seed(103)
        src = torch.randn(2, 3, 4, 8)
        actual = unpack_combined_tile_transpose_direct(src)

        assert _allclose(actual, src.permute(0, 2, 1, 3).contiguous())

    def test_batched_matmul_combined_tile_direct(self):
        """Direct call for grid + combined tile + separate reduction loop
        (batched matmul), the realistic combined-tile-plus-accumulator case."""

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[1, 4, 4, 4]),
        )
        def batched_matmul_direct(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            nb, m, k = a.shape
            _, k2, n = b.shape
            out = torch.zeros((nb, m, n), dtype=torch.float32, device=a.device)
            for i in hl.grid(nb):
                for tm, tn in hl.tile([m, n]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(k):
                        acc = torch.addmm(acc, a[i, tm, tk], b[i, tk, tn])
                    out[i, tm, tn] = acc
            return out

        torch.manual_seed(107)
        a = torch.randn(2, 8, 8)
        b = torch.randn(2, 8, 8)
        actual = batched_matmul_direct(a, b)

        assert _allclose(actual, torch.bmm(a, b))


# ---------------------------------------------------------------------------
# Configurable block sizes
# ---------------------------------------------------------------------------


class TestConfigurableBlockSizes:
    """Tests that block_sizes from the config propagate into the generated MLIR."""

    def test_block_size_in_ir(self):
        """Config block_sizes are reflected as the forall tile-offset map in the IR."""

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
            return out

        torch.manual_seed(0)
        A = torch.randn(64, 32)
        B = torch.randn(64, 32)

        for block_size in (8, 16, 32, 64):
            cfg = helion.Config(block_sizes=[block_size])
            mlir_module = generate_mlir(add_kernel, [A, B], config=cfg)
            ir_str = str(mlir_module)
            assert f"affine_map<(d0) -> (d0 * {block_size})>" in ir_str, (
                f"Expected tile size {block_size} in IR, got:\n{ir_str}"
            )

    def test_decorator_config_is_used_without_kwarg(self):
        """A config on @helion.kernel applies even when generate_mlir omits it."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[8]))
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
            return out

        torch.manual_seed(0)
        A = torch.randn(64, 32)
        B = torch.randn(64, 32)

        module = generate_mlir(add_kernel, [A, B])

        assert _tile_rows(module) == {8}, f"block_sizes=[8] was ignored:\n{module}"

    def test_explicit_config_overrides_decorator_config(self):
        """An explicit config= argument wins over the decorator's config."""

        @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[8]))
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
            return out

        torch.manual_seed(0)
        A = torch.randn(64, 32)
        B = torch.randn(64, 32)

        module = generate_mlir(
            add_kernel, [A, B], config=helion.Config(block_sizes=[32])
        )

        assert _tile_rows(module) == {32}

    def test_different_block_sizes_same_result(self):
        """Different block_sizes produce numerically identical results via execute_mlir."""

        @helion.kernel(static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
            return out

        torch.manual_seed(42)
        A = torch.randn(64, 32)
        B = torch.randn(64, 32)
        ref = A + B

        for block_size in (8, 16, 32, 64):
            cfg = helion.Config(block_sizes=[block_size])
            mlir_module = generate_mlir(add_kernel, [A, B], config=cfg)
            C = _backend().execute_mlir(mlir_module, A, B, kernel_name="add_kernel")
            assert _allclose(C, ref), (
                f"block_size={block_size}: max_err={torch.max(torch.abs(C - ref)).item():.2e}"
            )

    def test_block_size_via_direct_call(self):
        """Block size from Config propagates through the direct helion kernel call."""
        from contextlib import redirect_stdout
        import io
        import os
        from unittest import mock

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[16]),
        )
        def add_bs16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
            return out

        torch.manual_seed(7)
        A = torch.randn(64, 32)
        B = torch.randn(64, 32)

        # Capture the pre-lowering IR emitted by the direct call to verify
        # that block_size=16 from the Config decorator reaches the codegen.
        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HELION_MLIR_DUMP_PRE_LOWERING": "1"}),
            redirect_stdout(buf),
        ):
            result = add_bs16(A, B)

        ir_dump = buf.getvalue()
        assert "affine_map<(d0) -> (d0 * 16)>" in ir_dump, (
            f"Expected tile size 16 from Config in pre-lowering IR dump, got:\n{ir_dump[:500]}"
        )
        assert _allclose(result, A + B)

    def test_outer_forall_inner_scf_for_block_sizes(self):
        """Outer hl.tile([m,n]) → scf.forall; inner hl.tile(k) → scf.for.

        Config(block_sizes=[bs_mn, bs_k]) controls both steps independently.
        Verifies both step values appear in the pre-lowering IR and results are correct.
        """
        from contextlib import redirect_stdout
        import io
        import os
        from unittest import mock

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[16, 8, 32]),
        )
        def matmul_tiled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        torch.manual_seed(3)
        # Non-square dimensions are required: all three tile blocks (tile_m, tile_n, tile_k)
        # must have distinct hint values so _block_hint_to_id maps them unambiguously.
        # Choose k=64 with block_k=32 so the inner reduction loop must execute
        # multiple iterations and remains visible in pre-lowering IR.
        A = torch.randn(32, 64)
        B = torch.randn(64, 48)

        # Clear helion's in-memory kernel cache to force fresh compilation.
        matmul_tiled._bound_kernels.clear()
        matmul_tiled._dispatch_cache.clear()

        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HELION_MLIR_DUMP_PRE_LOWERING": "1"}),
            redirect_stdout(buf),
        ):
            result = matmul_tiled(A, B)

        ir_dump = buf.getvalue()
        assert (
            "affine_map<(d0) -> (d0 * 16)>" in ir_dump
            and "affine_map<(d0) -> (d0 * 8)>" in ir_dump
        ), "Expected outer tile sizes (16, 8)"
        assert _allclose(result, A @ B)

    def test_outer_forall_inner_scf_for_block_sizes_matmul_accumulate(self):
        """Nested tiling with explicit matmul + add accumulation.

        Same structure as addmm test but uses:
        ``acc = acc + torch.matmul(dequant, act_group)``.
        """
        from contextlib import redirect_stdout
        import io
        import os
        from unittest import mock

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[16, 8, 32]),
        )
        def matmul_acc_tiled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            k2, n = y.shape
            out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    dequant = x[tile_m, tile_k]
                    act_group = y[tile_k, tile_n]
                    acc = acc + torch.matmul(dequant, act_group)
                out[tile_m, tile_n] = acc
            return out

        torch.manual_seed(11)
        A = torch.randn(32, 64)
        B = torch.randn(64, 48)

        # Clear helion's in-memory kernel cache to force fresh compilation.
        matmul_acc_tiled._bound_kernels.clear()
        matmul_acc_tiled._dispatch_cache.clear()

        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HELION_MLIR_DUMP_PRE_LOWERING": "1"}),
            redirect_stdout(buf),
        ):
            result = matmul_acc_tiled(A, B)

        ir_dump = buf.getvalue()
        assert (
            "affine_map<(d0) -> (d0 * 16)>" in ir_dump
            and "affine_map<(d0) -> (d0 * 8)>" in ir_dump
        ), "Expected outer tile sizes (16, 8)"
        assert _allclose(result, A @ B)

    def test_outer_inner_loops_eltwise_block_sizes(self):
        """Nested outer/inner loops with eltwise ops honor configured steps."""
        from contextlib import redirect_stdout
        import io
        import os
        from unittest import mock

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[16, 8]),
        )
        def add_nested(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty((m, n), dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                for tile_n in hl.tile(n):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        torch.manual_seed(19)
        A = torch.randn(64, 48)
        B = torch.randn(64, 48)

        # Clear helion's in-memory kernel cache to force fresh compilation.
        add_nested._bound_kernels.clear()
        add_nested._dispatch_cache.clear()

        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"HELION_MLIR_DUMP_PRE_LOWERING": "1"}),
            redirect_stdout(buf),
        ):
            result = add_nested(A, B)

        ir_dump = buf.getvalue()
        assert "affine_map<(d0) -> (d0 * 16)>" in ir_dump, "Expected outer tile size 16"
        assert "step %c8" in ir_dump or "step (8)" in ir_dump, (
            "Expected inner loop step 8"
        )

        assert _allclose(result, A + B)


class TestPackCopyTileLayouts:
    """Regressions for logical-transpose pack/copy kernels.

    ``source: [K, PANELS, BN] -> out: [PANELS, K, BN]`` exercises nested loop
    block-id resolution and store/insert tile geometry validation.
    """

    @staticmethod
    def _inputs(panels: int = 4, k: int = 64, bn: int = 16):
        torch.manual_seed(7)
        return torch.randn(k, panels, bn), panels, k, bn

    def test_grid_panel_tile_k_scalar_grid_n_is_exact(self):
        """grid(panel) + tile(k) + scalar grid(n) must not swap block ids.

        The outer ``hl.tile`` body is a pure wrapper whose own symbol only
        appears in the innermost graph, so block-id resolution previously
        adopted the inner grid's id and emitted swapped induction variables.
        """

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[32]),
        )
        def pack(source: torch.Tensor) -> torch.Tensor:
            k, panels, bn = source.shape
            out = torch.empty((panels, k, bn), dtype=source.dtype, device=source.device)
            for panel in hl.grid(panels):
                for tile_k in hl.tile(k):
                    for n_idx in hl.grid(bn):
                        out[panel, tile_k, n_idx] = source[tile_k, panel, n_idx]
            return out

        source, _, _, _ = self._inputs()
        result = pack(source)
        expected = source.permute(1, 0, 2)
        assert result.shape == expected.shape
        assert torch.equal(result, expected)

    def test_grid_panel_tile_k_and_n_is_exact(self):
        """grid(panel) + tile([k, n]) lowers and is numerically exact."""

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[32, 16]),
        )
        def pack(source: torch.Tensor) -> torch.Tensor:
            k, panels, bn = source.shape
            out = torch.empty((panels, k, bn), dtype=source.dtype, device=source.device)
            for panel in hl.grid(panels):
                for tile_k, tile_n in hl.tile([k, bn]):
                    out[panel, tile_k, tile_n] = source[tile_k, panel, tile_n]
            return out

        source, _, _, _ = self._inputs()
        result = pack(source)
        assert torch.equal(result, source.permute(1, 0, 2))

    def test_explicit_permute_pack_copy_is_exact(self):
        """The supported spelling of a transposing pack uses ``.permute()``."""

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[2, 32, 16]),
        )
        def pack(source: torch.Tensor) -> torch.Tensor:
            k, panels, bn = source.shape
            out = torch.empty((panels, k, bn), dtype=source.dtype, device=source.device)
            for tile_p, tile_k, tile_n in hl.tile([panels, k, bn]):
                out[tile_p, tile_k, tile_n] = source[tile_k, tile_p, tile_n].permute(
                    1, 0, 2
                )
            return out

        source, _, _, _ = self._inputs()
        result = pack(source)
        assert torch.equal(result, source.permute(1, 0, 2))

    def test_implicit_transpose_store_raises_clear_diagnostic(self):
        """An implicit transpose store is rejected instead of writing OOB.

        Helion traces ``out[a, b, c] = src[b, a, c]`` without complaint even
        though the value's tile order does not match the destination slice.
        The backend must diagnose it rather than emit an out-of-bounds insert.
        """
        from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

        @helion.kernel(static_shapes=True)
        def pack(source: torch.Tensor) -> torch.Tensor:
            k, panels, bn = source.shape
            out = torch.empty((panels, k, bn), dtype=source.dtype, device=source.device)
            for tile_p, tile_k, tile_n in hl.tile([panels, k, bn]):
                out[tile_p, tile_k, tile_n] = source[tile_k, tile_p, tile_n]
            return out

        source, _, _, _ = self._inputs()
        with pytest.raises(UnsupportedOperationError, match="transposed or mismatched"):
            generate_mlir(pack, [source], config=helion.Config(block_sizes=[2, 32, 16]))


class TestHostSideViewAlias:
    """Regressions for host-side ``.view()`` aliases used inside the loop."""

    def test_rank_preserving_host_view_is_materialized(self):
        """A 2-D -> 2-D host view must change the value's MLIR shape.

        Previously the alias silently resolved to the base parameter's SSA
        value, producing an out-of-bounds insert of ``tensor<8x16>`` into
        ``tensor<16x8>``.
        """

        @helion.kernel(static_shapes=True)
        def reshape_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((16, 8), dtype=x.dtype, device=x.device)
            reshaped = x.view(16, 8)
            for _ in hl.tile(16):
                out[:, :] = reshaped
            return out

        module = generate_mlir(
            reshape_kernel,
            [torch.randn(8, 16)],
            config=helion.Config(block_sizes=[8, 16]),
        )
        assert module.operation.verify()
        assert ops_named(module, "tensor.expand_shape")
        stores = ops_named(module, "tensor.insert_slice")
        assert [static_sizes(store) for store in stores] == [[16, 8]]
        assert str(stores[0].results[0].type) == "tensor<16x8xf32>"

    def test_rank_preserving_host_view_executes_exactly(self):
        """The materialized host view produces correct numerics."""

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[8, 16]),
        )
        def reshape_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((16, 8), dtype=x.dtype, device=x.device)
            reshaped = x.view(16, 8)
            for _ in hl.tile(16):
                out[:, :] = reshaped
            return out

        torch.manual_seed(3)
        x = torch.randn(8, 16)
        assert torch.equal(reshape_kernel(x), x.view(16, 8))


class TestGenericAtenHelperBoundaryTileRegression:
    """Regression: generic ATen-helper ops must work for tiles > 64 elements.

    Helper signatures used to be rebuilt from Helion's symbolic metadata, whose
    unresolved block-size hints default to 64, so tiles wider than 64 got a
    mismatching ``tensor<...x64>`` helper. Helpers are now typed from the call
    site's MLIR operand types.

    These ops (``relu``, ``abs``, ``maximum``, elementwise scalar multiply)
    all go through the generic ATen-helper path -- unlike matmul/einsum,
    which have dedicated direct lowering and never hit this bug -- so they
    are the right coverage for this regression, independent of any one op.
    """

    @staticmethod
    def _relu_kernel():
        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[8, 1024]),
        )
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.relu(x[tile])
            return out

        return kernel

    @staticmethod
    def _abs_kernel():
        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[8, 1024]),
        )
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.abs(x[tile])
            return out

        return kernel

    @pytest.mark.parametrize("n", [64, 65, 96, 512, 1024, 4096])
    def test_relu_above_64_elements(self, n):
        kernel = self._relu_kernel()
        x = boundary_tile([0.0, 1.0], 16, n, dtype=torch.float32)
        # -0.0's sign: test_nan_minmax.py's test_relu_keeps_negative_zero.
        same_values(kernel(x) + 0.0, torch.relu(x) + 0.0)

    @pytest.mark.parametrize("n", [64, 96, 512, 4096])
    def test_abs_above_64_elements(self, n):
        kernel = self._abs_kernel()
        x = boundary_tile([0.0, 1.0], 16, n, dtype=torch.float32)
        same_values(kernel(x), torch.abs(x))

    @pytest.mark.parametrize("n", [64, 96, 512, 4096])
    def test_scalar_mul_above_64_elements(self, n):
        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[8, 1024]),
        )
        def scalar_mul_kernel(a: torch.Tensor, s: hl.constexpr) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(a.shape):
                out[tile] = a[tile] * s
            return out

        torch.manual_seed(2)
        a = torch.randn(16, n)
        assert _allclose(scalar_mul_kernel(a, hl.constexpr(math.pi)), a * math.pi)

    @pytest.mark.parametrize("n", [64, 96, 512, 4096])
    def test_maximum_above_64_elements(self, n):
        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[8, 1024]),
        )
        def maximum_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.maximum(x[tile], y[tile])
            return out

        x = boundary_tile([0.0, 1.0], 16, n, dtype=torch.float32)
        same_values(maximum_kernel(x, x.roll(3, -1)), torch.maximum(x, x.roll(3, -1)))

    def test_leaky_relu_above_64_elements(self):
        """``torch.nn.functional.leaky_relu`` needs both the shape fix and

        `hl.constexpr` for its ``negative_slope`` scalar (a dynamic float arg
        can't be lowered by this backend at all -- see ``elementwise.py``).
        """

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[8, 1024]),
        )
        def leaky_relu_kernel(
            x: torch.Tensor, negative_slope: hl.constexpr
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = torch.nn.functional.leaky_relu(
                    x[tile], negative_slope=negative_slope
                )
            return out

        torch.manual_seed(4)
        x = torch.randn(16, 4096)
        actual = leaky_relu_kernel(x, hl.constexpr(0.01))
        expected = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        assert _allclose(actual, expected)


class TestPaddedPackingAndMultiPhaseExecution:
    """Regression tests for 4D blocked packing, irregular shape matmul,

    and multi-phase in-place buffer preservation.
    """

    def test_pack_b_blocked_shape_and_square_matmul_execution(self):
        """Regression test for pack_b_blocked 4D view return and square matmul."""
        from helion_mlir_utils.matmul import matmul
        from helion_mlir_utils.matmul import pack_b_blocked

        b = torch.randn(128, 128, dtype=torch.float32)
        b4 = pack_b_blocked(b)
        assert b4.ndim == 4
        assert b4.shape == (4, 4, 32, 32)

        a = torch.randn(128, 128, dtype=torch.float32)
        actual = matmul(a, b)
        expected = a @ b
        assert _allclose(actual, expected)

    def test_prepacked_rhs_matmul_matches_runtime_packing(self):
        """A cached packed weight preserves fused bias and epilogue results."""
        from helion_mlir_utils.matmul import matmul
        from helion_mlir_utils.matmul import matmul_prepacked_b
        from helion_mlir_utils.matmul import pack_b_blocked_t

        torch.manual_seed(4)
        a = torch.randn(32, 64, dtype=torch.bfloat16)
        weight = torch.randn(96, 64, dtype=torch.bfloat16)
        bias = torch.randn(96, dtype=torch.bfloat16)
        packed_weight = pack_b_blocked_t(weight)

        expected = matmul(
            a,
            weight,
            trans_b=True,
            bias=bias,
            epilogue=torch.relu,
        )
        actual = matmul_prepacked_b(
            a,
            packed_weight,
            n=96,
            bias=bias,
            epilogue=torch.relu,
        )

        torch.testing.assert_close(actual, expected)

    def test_prepacked_rhs_affine_matmul_matches_reference(self):
        """Explicit scale/post-bias tensors fuse without tensor closures."""
        from helion_mlir_utils.matmul import matmul_prepacked_b_affine
        from helion_mlir_utils.matmul import pack_b_blocked_t

        torch.manual_seed(8)
        a = torch.randn(32, 64, dtype=torch.bfloat16)
        weight = torch.randn(96, 64, dtype=torch.bfloat16)
        bias = torch.randn(96, dtype=torch.bfloat16)
        scale = torch.rand(96, dtype=torch.bfloat16)
        post_bias = torch.randn(1, dtype=torch.bfloat16)

        actual = matmul_prepacked_b_affine(
            a,
            pack_b_blocked_t(weight),
            96,
            bias,
            scale,
            post_bias,
            torch.relu,
        )
        expected = torch.relu(
            (a.float() @ weight.float().T + bias.float()) * scale.float()
            + post_bias.float()
        ).to(torch.bfloat16)

        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=1e-2, atol=4e-2
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matmul_irregular_shapes_padded_packing(self, dtype):
        """Irregular (non-32-divisible) shapes pad inside packing kernels."""
        from helion_mlir_utils.matmul import matmul

        torch.manual_seed(5)
        a = torch.randn(37, 19, dtype=dtype)
        b = torch.randn(19, 45, dtype=dtype)

        actual = matmul(a, b)
        expected = a @ b
        atol = 1e-4 if dtype == torch.float32 else 0.5
        assert torch.allclose(actual.float(), expected.float(), atol=atol)

    @pytest.mark.parametrize("trans_b", [False, True])
    @pytest.mark.parametrize(
        "mkn", [(64, 32, 64), (32, 32, 32), (64, 64, 64), (64, 96, 70)]
    )
    def test_matmul_bf16_short_k(self, mkn, trans_b):
        """bf16 matmuls whose K is one or a few AMX K steps (a single-step K
        chunk miscompiled upstream)."""
        from helion_mlir_utils.matmul import matmul

        m, k, n = mkn
        torch.manual_seed(6)
        a = torch.randn(m, k, dtype=torch.bfloat16)
        b = torch.randn(k, n, dtype=torch.bfloat16)
        actual = matmul(a, b.T.contiguous() if trans_b else b, trans_b=trans_b)
        expected = (a.float() @ b.float()).to(torch.bfloat16)
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=1e-2, atol=0.5
        )

    @pytest.mark.parametrize("trans_b", [False, True])
    @pytest.mark.parametrize(
        "mkn, block_sizes",
        [
            ((64, 128, 96), None),
            ((64, 128, 96), (32, 64, 64)),
            ((40, 198, 33), None),
        ],
    )
    def test_matmul_bf16_trans_a(self, mkn, block_sizes, trans_b):
        """bf16 with A given as ``[K, M]``: packed per tile by the fused kernel,
        with its tile choice or the given ``block_sizes``. ``(40, 198, 33)``:
        partial M and N tiles (several, with remainders) and a K tail."""
        from helion_mlir_utils.matmul import matmul

        m, k, n = mkn
        torch.manual_seed(10)
        a = torch.randn(m, k, dtype=torch.bfloat16)
        b = torch.randn(k, n, dtype=torch.bfloat16)
        actual = matmul(
            a.T.contiguous(),
            b.T.contiguous() if trans_b else b,
            trans_a=True,
            trans_b=trans_b,
            block_sizes=block_sizes,
        )
        expected = (a.float() @ b.float()).to(torch.bfloat16)
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=1e-2, atol=0.5
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bmm_and_matvec(self, dtype):
        """bmm writes each batch slice in place; matvec is a row reduction."""
        from helion_mlir_utils.matmul import bmm
        from helion_mlir_utils.reduction import matvec

        torch.manual_seed(11)
        a = torch.randn(3, 100, 130, dtype=dtype)
        b = torch.randn(3, 130, 70, dtype=dtype)
        atol = 1e-3 if dtype == torch.float32 else 0.5
        torch.testing.assert_close(
            bmm(a, b).float(), torch.bmm(a, b).float(), rtol=1e-2, atol=atol
        )
        x = torch.randn(130, 1, dtype=dtype)
        torch.testing.assert_close(
            matvec(a[0], x).float(),
            (a[0].float() @ x.float()).to(dtype).float(),
            rtol=1e-2,
            atol=atol,
        )

    @pytest.mark.parametrize("linear", [False, True])
    @pytest.mark.parametrize(
        "mkn",
        [
            (100, 130, 100),
            (33, 66, 35),
            (70, 201, 40),
            (20, 50, 40),
        ],
    )
    def test_matmul_bf16_ragged(self, mkn, linear):
        """bf16 matmuls of unaligned shapes, odd K included: partial edge tiles,
        the K tail masked, the epilogue fused; ``linear``: B transposed (packed
        per tile from ``[N, K]``) with a fused bias. Several partial tiles per
        dim (100x130x100), two with remainders of 1 and 3 (33x66x35), odd K
        (201), M below one tile (20, the zero-padded path). Each shape is a
        compile of seconds: shapes giving the same tiles, chunks and tail as
        one of these are left out."""
        from helion_mlir_utils.matmul import matmul

        m, k, n = mkn
        torch.manual_seed(7)
        a = torch.randn(m, k, dtype=torch.bfloat16)
        b = torch.randn(k, n, dtype=torch.bfloat16)
        if linear:
            bias = torch.randn(n, dtype=torch.bfloat16)
            actual = matmul(
                a, b.T.contiguous(), trans_b=True, bias=bias, epilogue=torch.relu
            )
            expected = torch.relu(a.float() @ b.float() + bias.float())
        else:
            actual = matmul(a, b, epilogue=torch.relu)
            expected = torch.relu(a.float() @ b.float())
        torch.testing.assert_close(
            actual.float(), expected.to(torch.bfloat16).float(), rtol=1e-2, atol=0.5
        )

    @pytest.mark.parametrize(
        "mkn",
        [(64, 256, 128), (40, 66, 100), (64, 4096, 64), (300, 128, 64)],
    )
    def test_matmul_prepacked_vnni(self, mkn):
        """A weight packed once into VNNI panels (``pack_b_vnni_t``) is read in
        place by the fused kernel: ragged M (a second call for the rows past the
        last whole row tile), N and K included."""
        from helion_mlir_utils.matmul import matmul_prepacked_b
        from helion_mlir_utils.matmul import pack_b_vnni_t

        m, k, n = mkn
        torch.manual_seed(9)
        a = torch.randn(m, k, dtype=torch.bfloat16)
        weight = torch.randn(n, k, dtype=torch.bfloat16)
        bias = torch.randn(n, dtype=torch.bfloat16)
        actual = matmul_prepacked_b(
            a, pack_b_vnni_t(weight), n=n, bias=bias, epilogue=torch.relu
        )
        expected = torch.relu(a.float() @ weight.float().T + bias.float())
        torch.testing.assert_close(
            actual.float(), expected.to(torch.bfloat16).float(), rtol=1e-2, atol=0.5
        )

    def test_multiphase_inplace_buffer_preservation(self):
        """Multi-phase kernel preserves Phase 0's zeros in unwritten slice."""

        @helion.kernel(
            static_shapes=True,
            backend="mlir",
            config=helion.Config(block_sizes=[32, 32, 32, 32]),
        )
        def pad_copy_two_phase(
            a: torch.Tensor, m_pad: hl.constexpr, k_pad: hl.constexpr
        ) -> torch.Tensor:
            m, k = a.shape
            out = torch.empty((int(m_pad), int(k_pad)), dtype=a.dtype, device=a.device)
            for tile_m, tile_k in hl.tile([int(m_pad), int(k_pad)]):
                out[tile_m, tile_k] = hl.zeros([tile_m, tile_k], dtype=a.dtype)
            hl.barrier()
            for tile_m, tile_k in hl.tile([m, k]):
                out[tile_m, tile_k] = a[tile_m, tile_k]
            return out

        torch.manual_seed(6)
        a = torch.randn(32, 19)
        actual = pad_copy_two_phase(a, hl.constexpr(64), hl.constexpr(32))

        expected = torch.zeros(64, 32)
        expected[:32, :19] = a
        assert torch.equal(actual, expected)

    def test_pack_kernels_match_block_layouts(self):
        """Fused packing (padding included) gives the blocked and VNNI layouts."""
        import helion_mlir_utils as cpu_matmul

        torch.manual_seed(7)
        for rows, cols in [(128, 96), (70, 45)]:
            x = torch.randn(rows, cols, dtype=torch.float32)
            r, c = -(-rows // 32) * 32, -(-cols // 32) * 32
            padded = torch.nn.functional.pad(x, (0, c - cols, 0, r - rows))
            a4 = padded.reshape(r // 32, 32, c // 32, 32).permute(0, 2, 1, 3)
            b4 = padded.reshape(r // 32, 32, c // 32, 32).permute(2, 0, 1, 3)
            vnni = b4.reshape(c // 32, r // 32, 16, 2, 32).transpose(3, 4)
            x_t = x.T.contiguous()
            assert torch.equal(cpu_matmul.pack_a_blocked(x), a4)
            assert torch.equal(cpu_matmul.pack_a_blocked_t(x_t), a4)
            assert torch.equal(cpu_matmul.pack_b_blocked(x), b4)
            assert torch.equal(cpu_matmul.pack_b_blocked_t(x_t), b4)
            assert torch.equal(cpu_matmul.pack_b_blocked_vnni(x), vnni)
            assert torch.equal(cpu_matmul.pack_b_blocked_vnni_t(x_t), vnni)

    def test_ragged_panel_count_matmul_execution(self):
        """Regression: ragged block-8 panel stores corrupted the native heap."""
        from helion_mlir_utils.matmul import matmul

        torch.manual_seed(9)
        a = torch.randn(65, 47, dtype=torch.bfloat16)
        b = torch.randn(47, 372, dtype=torch.bfloat16)

        actual = matmul(a, b)
        expected = a.float() @ b.float()
        torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=0.5)
