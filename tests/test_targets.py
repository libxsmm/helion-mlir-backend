"""Device targets: how a kernel's target is resolved, and the CPU target's flow."""

from __future__ import annotations

import hashlib

import helion
import helion.language as hl
import pytest
import torch

from helion_mlir_backend import compile_mlir
from helion_mlir_backend import generate_mlir
from helion_mlir_backend._compiler import execution
from helion_mlir_backend._compiler.mlir.support import UnsupportedTargetError
from helion_mlir_backend._compiler.targets import CpuTarget
from helion_mlir_backend._compiler.targets import get_target
from helion_mlir_backend._compiler.targets import resolve_target
from helion_mlir_backend._compiler.targets.cpu import cpu_name
from helion_mlir_backend.api import _compile

CPU = torch.device("cpu")
META = torch.device("meta")


def axpy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0 + y[tm, tn]
    return out


def _kernel() -> helion.Kernel:
    return helion.kernel(backend="mlir", config=helion.Config(block_sizes=[32, 32]))(
        axpy
    )


@pytest.fixture(autouse=True)
def auto_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HELION_MLIR_TARGET", raising=False)


def test_cpu_tensors_resolve_to_the_cpu_target() -> None:
    target = resolve_target(CPU)
    assert isinstance(target, CpuTarget)
    assert target.name == "cpu"
    assert resolve_target() is target
    assert resolve_target(CPU, "auto") is target


def test_cpu_target_describes_todays_cpu_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    target = get_target("cpu")
    assert target.pipelines == ("opt", "scalar")
    assert target.default_pipeline() == "opt"
    monkeypatch.setenv("HELION_MLIR_PIPELINE", "scalar")
    assert target.default_pipeline() == "scalar"
    assert target.hardware_name() == cpu_name()
    target.validate_environment()
    with pytest.raises(NotImplementedError, match="only CPU is supported"):
        target.check_tensor("x", torch.empty(2, device="meta"))


def test_pipelines_and_stop_points_belong_to_their_target() -> None:
    cpu = get_target("cpu")
    assert cpu.resolve_pipeline(None) == "opt"
    assert cpu.resolve_pipeline("scalar") == "scalar"
    with pytest.raises(ValueError, match="pipeline 'xegpu' for the MLIR target 'cpu'"):
        cpu.resolve_pipeline("xegpu")
    cpu.check_stop_at(None)
    with pytest.raises(ValueError, match="stop point 'xevm' for the MLIR target 'cpu'"):
        cpu.check_stop_at("xevm")
    assert cpu.is_full(None)


@pytest.mark.parametrize("value", ["cpu", "auto", ""])
def test_environment_names_the_target(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("HELION_MLIR_TARGET", value)
    device = META if value == "cpu" else CPU
    assert resolve_target(device).name == "cpu"


def test_unknown_target_in_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELION_MLIR_TARGET", "gpu")
    with pytest.raises(ValueError, match="HELION_MLIR_TARGET='gpu'"):
        resolve_target(CPU)


def test_explicit_target_overrides_environment_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELION_MLIR_TARGET", "gpu")
    assert resolve_target(META, "cpu").name == "cpu"
    assert resolve_target(CPU, "auto").name == "cpu"


def test_unknown_target_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown MLIR target 'gpu'"):
        resolve_target(CPU, "gpu")


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(UnsupportedTargetError, match="device 'cuda'"):
        resolve_target(torch.device("cuda"))


def test_meta_tensors_need_a_named_target() -> None:
    with pytest.raises(UnsupportedTargetError, match="meta tensors"):
        resolve_target(META)
    args = [torch.empty(64, 64, device="meta"), torch.empty(64, 64, device="meta")]
    with pytest.raises(UnsupportedTargetError, match="meta tensors"):
        generate_mlir(_kernel(), args)
    assert "func.func public @axpy" in str(generate_mlir(_kernel(), args, target="cpu"))


def test_jit_cache_keys_differ_by_target_and_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OtherCpuTarget(CpuTarget):
        name = "other-cpu"

    monkeypatch.setattr(execution, "_JIT_CACHE", type(execution._JIT_CACHE)())
    x, y = torch.randn(64, 64), torch.randn(64, 64)
    hf, config, env = _compile(_kernel(), [x, y], None)
    targets = [
        (get_target("cpu"), "opt"),
        (get_target("cpu"), "scalar"),
        (OtherCpuTarget(), "opt"),
    ]
    with env:
        module_text = str(env.backend.generate_mlir(hf, config, env))
    for target, pipeline in [*targets, targets[0]]:
        with env:
            module = env.backend.generate_mlir(hf, config, env)
        compiled = execution.compile_entry(
            module, "axpy", target=target, pipeline=pipeline
        )
        assert compiled.target is target
    keys = list(execution._JIT_CACHE)
    assert len(keys) == 3
    assert {key[0] for key in keys} == {
        hashlib.sha256(module_text.encode()).hexdigest()
    }
    assert [key[1:3] for key in keys] == [
        ("cpu", "scalar"),
        ("other-cpu", "opt"),
        ("cpu", "opt"),
    ]


def test_compile_mlir_for_the_cpu_target(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def hook(name: str) -> object:
        return lambda self, buffers: calls.append(f"{name}:{len(buffers)}")

    monkeypatch.setattr(CpuTarget, "prepare_call", hook("prepare"))
    monkeypatch.setattr(CpuTarget, "finish_call", hook("finish"))
    # The explicit target wins over the environment's.
    monkeypatch.setenv("HELION_MLIR_TARGET", "gpu")
    x, y = torch.randn(64, 64), torch.randn(64, 64)
    run = compile_mlir(_kernel(), [x, y], target="cpu", pipeline="scalar")
    torch.testing.assert_close(run(x, y), x * 2.0 + y)
    assert calls == ["prepare:3", "finish:3"]
    with pytest.raises(NotImplementedError, match="only CPU is supported"):
        run(x.to("meta"), y.to("meta"))
