"""Runtime scalars: float, int and bool kernel parameters passed to the entry by
value (``f64``/``i64``/``i1`` arguments), read in device and host code."""

from __future__ import annotations

import re

import helion
import helion.language as hl
import pytest
import torch

from tests.harness import check_kernel
from tests.harness import run_direct
from tests.inputs import boundary_tile

from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler.mlir import driver
from helion_mlir_backend._compiler.mlir.backend import MLIRBackend
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def scale_shift(x: torch.Tensor, alpha: float, beta: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * alpha + beta
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def scale_positive(x: torch.Tensor, flag: bool) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile]
        out[tile] = torch.where(v > 0, v * flag, v)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(4, 8))
def scaled_row_sums(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """``alpha`` read in the inner (K) loop."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n] * alpha
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def fill_then_scale(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """``alpha`` read by the host code (the fill) and the device code."""
    out = torch.full_like(x, alpha)
    for tile in hl.tile(x.size(0)):
        out[tile] = out[tile] + x[tile] * alpha
    return out


@pytest.mark.parametrize(("alpha", "beta"), [(0.5, 3), (-2.25, -7)])
def test_float_and_int_scalars(alpha: float, beta: int) -> None:
    check_kernel(scale_shift, lambda x, a, b: x * a + b, [torch.randn(32), alpha, beta])


@pytest.mark.parametrize("flag", [True, False])
def test_bool_scalar(flag: bool) -> None:
    check_kernel(
        scale_positive,
        lambda x, f: torch.where(x > 0, x * f, x),
        [boundary_tile([0.0, 1.0], 32, dtype=torch.float32), flag],
        exact=True,
    )


def test_scalar_in_inner_loop() -> None:
    x = torch.randn(8, 32)
    check_kernel(
        scaled_row_sums, lambda x, a: (x * a).sum(-1), [x, 1.5], atol=1e-4, rtol=1e-4
    )


def test_scalar_in_host_and_device_code() -> None:
    check_kernel(fill_then_scale, lambda x, a: a + x * a, [torch.randn(32), 0.75])


def test_scalar_values_change_without_recompiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiles: list[str] = []
    compile_kernel = driver.compile_kernel

    def counting(*args: object, **kwargs: object) -> object:
        compiles.append("compile")
        return compile_kernel(*args, **kwargs)

    monkeypatch.setattr(driver, "compile_kernel", counting)
    scale_shift.reset()
    x = torch.randn(32)
    torch.testing.assert_close(run_direct(scale_shift, [x, 2.0, 1]), x * 2.0 + 1)
    torch.testing.assert_close(run_direct(scale_shift, [x, -3.0, 5]), x * -3.0 + 5)
    torch.testing.assert_close(run_direct(scale_shift, [x, 0.0, 0]), x * 0.0)
    assert compiles == ["compile"]


def test_entry_takes_scalars_by_value() -> None:
    text = str(generate_mlir(scale_shift, [torch.randn(32), 0.5, 3]))
    entry = re.search(r"func\.func public @scale_shift\(([^\n]*)\) \{", text)
    assert entry, text
    assert re.search(
        r'%arg\d+: f64 \{helion\.name = "alpha", helion\.role = "scalar"\}',
        entry.group(1),
    )
    assert re.search(
        r'%arg\d+: i64 \{helion\.name = "beta", helion\.role = "scalar"\}',
        entry.group(1),
    )
    # The phase function takes the values too: no 0-d tensors or memrefs.
    assert re.search(r"@scale_shift__phase0\([^)]*: f64, %arg\d+: i64\)", text)
    assert "memref<f64>" not in text and "tensor<f64>" not in text
    assert "tensor.extract %" not in text
    flag = str(generate_mlir(scale_positive, [torch.randn(32), True]))
    assert re.search(
        r'%arg\d+: i1 \{helion\.name = "flag", helion\.role = "scalar"\}', flag
    )


def test_execute_mlir_rejects_runtime_scalars() -> None:
    x = torch.randn(32)
    module = generate_mlir(scale_shift, [x, 0.5, 3])
    with pytest.raises(UnsupportedOperationError, match="host value 'alpha'"):
        MLIRBackend().execute_mlir(module, x, kernel_name="scale_shift")
