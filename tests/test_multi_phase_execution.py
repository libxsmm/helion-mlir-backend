"""End-to-end tests for multi-phase (hl.barrier()) and host-tensor interop kernels.

The whole kernel is one module (one function per phase plus the entry); every
call runs the kernel's host code and returns its own ``return`` value.
"""

from __future__ import annotations

import helion
import helion.language as hl
import torch

import helion_mlir_backend  # noqa: F401 registers "mlir" backend


def _tiles_of_8(count: int) -> helion.Config:
    return helion.Config(block_sizes=[8] * count)


def test_two_phase_barrier_kernel_direct_call():
    @helion.kernel(static_shapes=True, backend="mlir", config=_tiles_of_8(4))
    def two_phase_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = mid[tm, tn] * 2.0
        return out

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = two_phase_kernel(x, y)
    torch.testing.assert_close(result, (x + y) * 2.0)


def test_three_phase_chained_dependency_direct_call():
    @helion.kernel(static_shapes=True, backend="mlir", config=_tiles_of_8(6))
    def three_phase_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        a = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        b = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            a[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            b[tm, tn] = a[tm, tn] * 2.0
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = b[tm, tn] - x[tm, tn]
        return out

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = three_phase_kernel(x, y)
    torch.testing.assert_close(result, ((x + y) * 2.0) - x)


def test_host_tensor_interop_direct_call():
    @helion.kernel(static_shapes=True, backend="mlir", config=_tiles_of_8(2))
    def host_side_scale(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        scale = x.mean() * 100.0
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            s = hl.load(scale, [])
            out[tm, tn] = x[tm, tn] + s
        return out

    x = torch.randn(8, 8)
    result = host_side_scale(x)
    torch.testing.assert_close(result, x + x.mean() * 100.0)


def test_host_tensor_computed_before_phases_and_used_in_later_phase():
    """Host work can precede phases and feed a later phase by ``hl.load``.

    Helion disallows ordinary statements between top-level device loops, so
    this is the valid syntax for host-side interop used by a later phase.
    """

    @helion.kernel(
        static_shapes=True,
        backend="mlir",
        config=_tiles_of_8(4),
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    def later_phase_uses_host_scale(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        scale = x.mean() * 3.0
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            s = hl.load(scale, [])
            out[tm, tn] = mid[tm, tn] * s
        return out

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = later_phase_uses_host_scale(x, y)
    torch.testing.assert_close(result, (x + y) * (x.mean() * 3.0))


def test_multiphase_grid_packing_then_blocked_matmul():
    """Phase-local block IDs must supersede a stale ID reused from phase 0.

    The nested `tm` loop in phase 1 can be declared with phase 0's raw grid
    block ID even though its body identifies a new tile block ID. The
    recovery must use that unique body-derived ID, or synthetic accumulator
    geometry expands the final 8-wide panel dimension to 16 and emits an
    invalid tensor.cast.
    """

    n, block_m, block_n, tile_m, tile_k = 32, 16, 8, 8, 8
    m_blocks, n_panels = n // block_m, n // block_n

    @helion.kernel(
        static_shapes=True,
        backend="mlir",
        config=helion.Config(block_sizes=[tile_k, tile_m, tile_k]),
    )
    def packed_blocked_matmul(a3: torch.Tensor, b3: torch.Tensor) -> torch.Tensor:
        m_blocks_, block_m_, k = a3.shape
        k2, n_panels_, block_n_ = b3.shape
        packed_b = torch.empty(
            (n_panels_, k, block_n_), dtype=b3.dtype, device=b3.device
        )
        blocked_out = torch.empty(
            (m_blocks_, n_panels_, block_m_, block_n_),
            dtype=torch.float32,
            device=a3.device,
        )
        for panel in hl.grid(n_panels_):
            for tile_k_ in hl.tile(k):
                packed_b[panel, tile_k_, :] = b3[tile_k_, panel, :]
        hl.barrier()
        for m_block, panel in hl.grid([m_blocks_, n_panels_]):
            for tile_m_ in hl.tile(block_m_):
                acc = hl.zeros([tile_m_, block_n_], dtype=torch.float32)
                for tile_k_ in hl.tile(k):
                    acc = torch.addmm(
                        acc,
                        a3[m_block, tile_m_, tile_k_],
                        packed_b[panel, tile_k_, :],
                    )
                blocked_out[m_block, panel, tile_m_, :] = acc
        return blocked_out

    a = torch.randn(n, n, dtype=torch.bfloat16)
    b = torch.randn(n, n, dtype=torch.bfloat16)
    a3 = a.view(m_blocks, block_m, n)
    b3 = b.view(n, n_panels, block_n).contiguous()

    actual = packed_blocked_matmul(a3, b3)
    expected = (
        (a.float() @ b.float())
        .view(m_blocks, block_m, n_panels, block_n)
        .permute(0, 2, 1, 3)
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1.0)


def test_multiphase_reordered_store_uses_nested_terminal_mapping():
    """A phase-1 terminal store below `grid -> tile` maps grid to dim 1.

    The phase root itself has no store, so terminal-store lookup must include
    nested loop bodies to derive `out[tm, panel, :]` correctly rather than
    falling back to loop declaration order.
    """

    @helion.kernel(
        static_shapes=True,
        backend="mlir",
        config=helion.Config(block_sizes=[4, 4]),
    )
    def unpack_after_barrier(src: torch.Tensor) -> torch.Tensor:
        panels, rows, width = src.shape
        mid = torch.empty((panels, rows, width), dtype=src.dtype, device=src.device)
        out = torch.empty((rows, panels, width), dtype=src.dtype, device=src.device)
        for panel in hl.grid(panels):
            for tm in hl.tile(rows):
                mid[panel, tm, :] = src[panel, tm, :]
        hl.barrier()
        for panel in hl.grid(panels):
            for tm in hl.tile(rows):
                out[tm, panel, :] = mid[panel, tm, :]
        return out

    src = torch.randn(3, 8, 5)
    actual = unpack_after_barrier(src)
    torch.testing.assert_close(actual, src.permute(1, 0, 2).contiguous())


def test_multi_phase_generate_mlir_has_one_function_per_phase():
    from helion_mlir_backend import generate_mlir
    from helion_mlir_backend._compiler.mlir.backend import MLIRBackend

    @helion.kernel(static_shapes=True)
    def two_phase_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = mid[tm, tn] * 2.0
        return out

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    module = generate_mlir(two_phase_kernel, [x, y])
    text = str(module)
    assert "func.func private @two_phase_kernel__phase0" in text
    assert "func.func private @two_phase_kernel__phase1" in text
    mid, out = MLIRBackend().execute_mlir(module, x, y, kernel_name="two_phase_kernel")
    torch.testing.assert_close(mid, x + y)
    torch.testing.assert_close(out, (x + y) * 2.0)


def test_two_phase_kernel_with_multi_output_final_phase():
    """Combines both features: a 2-phase kernel whose final phase returns
    two tensors (return out1, out2), exercising the N-output-per-phase
    machinery together with cross-phase threading in the same kernel."""

    @helion.kernel(static_shapes=True, backend="mlir", config=_tiles_of_8(4))
    def two_phase_multi_output(x: torch.Tensor, y: torch.Tensor):
        m, n = x.shape
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out1 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out2 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out1[tm, tn] = mid[tm, tn] * 2.0
            out2[tm, tn] = mid[tm, tn] - x[tm, tn]
        return out1, out2

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = two_phase_multi_output(x, y)
    assert isinstance(result, tuple) and len(result) == 2
    mid = x + y
    torch.testing.assert_close(result[0], mid * 2.0)
    torch.testing.assert_close(result[1], mid - x)


def test_three_output_kernel_direct_call():
    """Multi-output stress test: 3 outputs from one hl.tile() loop, via the
    direct call path (which takes the unchanged single-phase fast path
    since there's no hl.barrier()/extra host tensor here)."""

    @helion.kernel(static_shapes=True, backend="mlir", config=_tiles_of_8(2))
    def three_outputs(x: torch.Tensor, y: torch.Tensor):
        m, n = x.shape
        out1 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out2 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out3 = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            out1[tm, tn] = x[tm, tn] + y[tm, tn]
            out2[tm, tn] = x[tm, tn] - y[tm, tn]
            out3[tm, tn] = x[tm, tn] * y[tm, tn]
        return out1, out2, out3

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = three_outputs(x, y)
    assert isinstance(result, tuple) and len(result) == 3
    torch.testing.assert_close(result[0], x + y)
    torch.testing.assert_close(result[1], x - y)
    torch.testing.assert_close(result[2], x * y)


def test_four_phase_chained_dependency_direct_call():
    """Stress test beyond 2-3 phases: 4 phases, each depending on the last."""

    @helion.kernel(static_shapes=True, backend="mlir", config=_tiles_of_8(8))
    def four_phase_kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        a = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        b = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        c = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            a[tm, tn] = x[tm, tn] + 1.0
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            b[tm, tn] = a[tm, tn] * 2.0
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            c[tm, tn] = b[tm, tn] - 1.0
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = c[tm, tn] / 2.0
        return out

    x = torch.randn(8, 8)
    result = four_phase_kernel(x)
    expected = (((x + 1.0) * 2.0) - 1.0) / 2.0
    torch.testing.assert_close(result, expected)


def test_return_expression_is_evaluated_on_host():
    @helion.kernel(static_shapes=True, backend="mlir", config=_tiles_of_8(4))
    def two_phase_expression_return(x: torch.Tensor, y: torch.Tensor):
        m, n = x.shape
        mid = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        for tm, tn in hl.tile([m, n]):
            mid[tm, tn] = x[tm, tn] + y[tm, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            out[tm, tn] = mid[tm, tn] * 2.0
        return out.view(-1) + 1

    x = torch.randn(8, 8)
    y = torch.randn(8, 8)
    result = two_phase_expression_return(x, y)
    torch.testing.assert_close(result, ((x + y) * 2.0).view(-1) + 1)


def test_local_tensor_named_like_a_scalar_key():
    """Regression: a host tensor named ``u0`` and a runtime scalar whose key is
    ``u0`` shared one name table, so a later phase's call took the scalar for the
    tensor ("operand type mismatch"), found on a generated Llama step."""

    def two_phases(x: torch.Tensor, c: torch.Tensor, pos: int) -> torch.Tensor:
        m, n = x.shape
        u0 = torch.empty_like(x)
        for tm, tn in hl.tile([m, n]):
            u0[tm, tn] = x[tm, tn] * c[pos, tn]
        hl.barrier()
        for tm, tn in hl.tile([m, n]):
            x[tm, tn] = u0[tm, tn] + 1.0
        return x

    kernel = helion.kernel(
        two_phases,
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[8] * 4),
    )
    x, c = torch.randn(8, 8), torch.randn(4, 8)
    expected = x * c[2] + 1.0
    torch.testing.assert_close(kernel(x.clone(), c, 2), expected)
