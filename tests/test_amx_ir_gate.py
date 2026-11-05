"""AMX IR gate: production bf16 contractions must reach AMX ops in the optimizing pipeline.

Runs on any x86 host: AMX is forced on through lighthouse's ``TargetInfo.override`` and
the pipeline is applied stage by stage until ``x86.amx`` ops appear. Nothing is executed.
"""

from __future__ import annotations

import pytest

from tests.harness import blocked_matmul_cases
from tests.harness import lower_opt_with_amx

from helion_mlir_backend import generate_mlir


def _amx_ops(module: object) -> set[str]:
    return {
        token.split("(")[0]
        for token in str(module).split()
        if token.startswith("x86.amx.")
    }


def _first_amx_ops(module: object, entry: str) -> set[str]:
    lowered = lower_opt_with_amx(
        module, entry, lambda stage, after: "x86" in stage and bool(_amx_ops(after))
    )
    return set() if lowered is None else _amx_ops(lowered[1])


def _until_amx_contraction(stage: str, module: object) -> bool:
    return "x86.amx.tile_mulf" in str(module)


@pytest.mark.slow
@pytest.mark.isolated
@pytest.mark.parametrize(
    "case", ["blocked_matmul", "blocked_matmul_bias", "blocked_matmul_affine"]
)
def test_blocked_bf16_matmul_reaches_amx(case: str) -> None:
    kernel, args = blocked_matmul_cases()[case]
    ops = _first_amx_ops(generate_mlir(kernel, args), kernel.fn.__name__)
    assert "x86.amx.tile_mulf" in ops, (
        f"no AMX contraction produced (saw {sorted(ops)})"
    )


@pytest.mark.slow
@pytest.mark.isolated
def test_partial_tile_pads_only_edge_register_tiles(monkeypatch) -> None:
    """In a partial M tile, register tiles of full rows read A in place by AMX;
    only the edge ones copy A rows into a register-tile buffer."""
    import importlib

    import helion
    import torch

    mm = importlib.import_module("helion_mlir_utils.matmul")

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    m, n, k = 232, 64, 128
    block_sizes, k_chunked, k_even = mm._fused_pack_tiles(m, n, k)
    tile_m, chunk = block_sizes[0], block_sizes[2]
    assert m % tile_m > 32  # a partial tile with a full and a partial register tile
    kernel = helion.kernel(
        mm._matmul_fused_pack.body,
        static_shapes=True,
        backend="mlir",
        config=helion.Config(block_sizes=block_sizes),
    )
    args = [
        torch.randn(m, k, dtype=torch.bfloat16),
        torch.randn(k, n, dtype=torch.bfloat16),
        None,
        None,
        None,
        mm.identity_epilogue,
        helion.language.constexpr(k_chunked),
        helion.language.constexpr(k_even),
        helion.language.constexpr(False),
        helion.language.constexpr(False),
    ]
    lowered = lower_opt_with_amx(
        generate_mlir(kernel, args),
        "_matmul_fused_pack",
        _until_amx_contraction,
        verify=True,
    )
    assert lowered is not None, "no AMX contraction produced"
    text = str(lowered[1])
    assert "vector.contract" not in text
    loads = [line for line in text.splitlines() if "x86.amx.tile_load" in line]
    # Full register tiles of the partial tile read A, of runtime rows, in place.
    assert any(f"memref<?x{chunk}xbf16, strided<[{k}, 1]" in line for line in loads)
    # No copy of the whole partial tile's A panel.
    assert f"memref<{tile_m}x{chunk}xbf16>" not in text


@pytest.mark.slow
@pytest.mark.isolated
def test_prepacked_ragged_k_contracts_only_with_amx(monkeypatch) -> None:
    """A K pair count off the AMX step: whole chunks and a padded tail chunk all
    reach AMX; no contraction is left to lower per element (it took minutes)."""
    import importlib

    import helion
    import torch

    mm = importlib.import_module("helion_mlir_utils.matmul")

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    m, k, n = 64, 130, 64
    pairs = k // 2
    pairs_even = pairs // 32 * 32
    kernel = helion.kernel(
        mm._matmul_prepacked_vnni.body,
        static_shapes=True,
        backend="mlir",
        config=helion.Config(block_sizes=[32, 1, pairs_even, 32]),
    )
    args = [
        torch.randn(m, k, dtype=torch.bfloat16).view(m, pairs, 2),
        mm.pack_b_vnni_t(torch.randn(n, k, dtype=torch.bfloat16)),
        None,
        None,
        None,
        mm.identity_epilogue,
        helion.language.constexpr(pairs_even),
        helion.language.constexpr(pairs_even),
    ]
    lowered = lower_opt_with_amx(
        generate_mlir(kernel, args), "_matmul_prepacked_vnni", _until_amx_contraction
    )
    assert lowered is not None, "no AMX contraction produced"
    text = str(lowered[1])
    assert "vector.contract" not in text
