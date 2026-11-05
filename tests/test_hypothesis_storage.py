"""Hypothesis and bytecode stay off NFS, in every kind of test process."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from hypothesis.configuration import storage_directory
import pytest

from tests.conftest import HYPOTHESIS_STORAGE_NAMED


def _check_storage() -> None:
    path = storage_directory("x", intent_to_write=False).path
    assert path.parent == Path(os.environ["HYPOTHESIS_STORAGE_DIRECTORY"])
    assert Path(__file__).resolve().parent.parent not in path.resolve().parents


def test_storage_is_the_configured_directory() -> None:
    _check_storage()
    if not HYPOTHESIS_STORAGE_NAMED:
        assert os.environ["HYPOTHESIS_STORAGE_DIRECTORY"].startswith("/dev/shm/")


@pytest.mark.isolated
def test_storage_is_the_configured_directory_in_an_isolated_child() -> None:
    _check_storage()


def _check_no_bytecode() -> None:
    assert sys.dont_write_bytecode
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"


def test_no_bytecode_is_written() -> None:
    _check_no_bytecode()


@pytest.mark.isolated
def test_no_bytecode_is_written_in_an_isolated_child() -> None:
    _check_no_bytecode()
