"""A lowering process that dies says how: its signal and the assertion it printed."""

from __future__ import annotations

import faulthandler
import os
import signal

from mlir import ir
import pytest

from helion_mlir_backend._compiler import execution
from helion_mlir_backend._compiler.execution import _death_message

ASSERTION = (
    "mlir-opt: Types.cpp:125: unsigned int getIntOrFloatBitWidth() const: "
    "Assertion `isIntOrFloat()' failed."
)


class _Dying:
    def __init__(self, die: object) -> None:
        self.die = die

    def apply(self, module: ir.Module) -> ir.Module:
        self.die()
        return module


class _Target:
    def __init__(self, die: object) -> None:
        self.die = die

    def lowering(self, *_: object) -> _Dying:
        return _Dying(self.die)

    def enter_lowering_child(self) -> None:
        pass


def _abort_after_assertion() -> None:
    os.write(2, f"noise\n{ASSERTION}\nmore noise\n".encode())
    # pytest's handler writes to the session's stderr, not to the child's fd 2: an
    # abort this test expects would print a "Fatal Python error" in a passing run.
    faulthandler.disable()
    os.abort()


def _exit_after_output() -> None:
    os.write(2, b"first\nlast line\n")
    os._exit(3)


def _lower(die: object) -> None:
    with ir.Context():
        module = ir.Module.parse("module {}")
        execution._lower(module, "entry", _Target(die), "pipeline")


@pytest.fixture(autouse=True)
def _forked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HELION_MLIR_COMPILE_TIMEOUT", "60")


def test_abort_names_the_signal_and_the_assertion() -> None:
    with pytest.raises(RuntimeError) as info:
        _lower(_abort_after_assertion)
    assert "died on SIGABRT" in str(info.value)
    assert ASSERTION in str(info.value)
    assert any("more noise" in note for note in info.value.__notes__)


def test_exit_names_the_code_and_the_last_line() -> None:
    with pytest.raises(RuntimeError) as info:
        _lower(_exit_after_output)
    assert str(info.value).endswith("with exit code 3: last line")


def test_kill_without_output_names_the_signal() -> None:
    with pytest.raises(RuntimeError) as info:
        _lower(lambda: os.kill(os.getpid(), signal.SIGKILL))
    assert str(info.value) == "lighthouse lowering process died on SIGKILL"


def test_message_without_an_exit_code() -> None:
    assert _death_message(None, "") == (
        "lighthouse lowering process died with exit code None"
    )
