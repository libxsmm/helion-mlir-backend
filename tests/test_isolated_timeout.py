"""A hung isolated test fails instead of holding the run."""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from tests.conftest import _live_processes


@pytest.mark.xfail(strict=True, reason="the timeout fails it")
@pytest.mark.isolated(timeout=2)
def test_hang_is_killed_and_fails() -> None:
    time.sleep(600)


@pytest.mark.isolated(timeout=60)
def test_timeout_leaves_a_quick_test_alone() -> None:
    assert True


def test_a_hang_report_shows_the_process_and_its_children() -> None:
    child = subprocess.Popen(["sleep", "60"])
    try:
        shown = _live_processes(os.getpid())
    finally:
        child.kill()
        child.wait()
    assert f"{os.getpid()} " in shown
    assert f"{child.pid} " in shown
    assert "sleep" in shown
