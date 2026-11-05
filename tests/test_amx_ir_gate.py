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
