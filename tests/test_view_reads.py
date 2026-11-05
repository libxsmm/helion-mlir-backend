"""Reads through views of written tensors: right or refused, never silently stale.

Each case in ``view_read_cases`` is honoured or refused on the CPU, with a new input
per call: an output's block is reused, so a repeated input would pass by accident.
"""

# ruff: noqa: FURB118
from __future__ import annotations

from unittest import mock

import helion
import pytest
import torch

from tests import view_read_cases as cases

import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir import driver
from helion_mlir_backend._compiler.mlir.support import UnsupportedOperationError

REFUSED = frozenset(
    {
        "A_w1_r2_reshape",
        "C4_same_tile_w_then_r_1loop",
        "C5_w1_r2_and_w2",
        "D_two_views",
        "F1_two_written_views",
        "F2_write_through_view_read_base",
        "G1_transpose_w1_r2",
        "G2_every_other_w1_r2",
        "G4_strided_inout_w1_r2",
        "G5_misaligned_view_w1_r2",
        "G6_neg_view_w1_r2",
        "G7_gemm_through_view_w1_r2",
        "J1_nobarrier_w_then_r",
        "L1_seq_w1_forall_r2",
        "L2_forall_w1_seq_r2",
        "L3_seq_w1_seq_r2",
        "N2_caller_view_w1_r2",
        "N3_inplace_read_later_phase",
        "O_w1_r2_w3",
        "P1_host_slice_written_param_read_later",
        "P2_host_reshape_written_param_read_later",
        "P3_host_slice_of_caller_alias",
        "P4_host_reshape_alias_of_caller_alias",
        "P5_view_of_view_local",
        "P6_view_of_view_of_written_param",
        "Q1_global_w1_r2",
        "Q4_default_arg_w1_r2",
        "Q5_global_view_written_by_host_local",
        "T4_partial_store_elsewhere",
        "T5_exclusive_if_branches",
        "U2_range_read_then_store",
        "U5_dynamic_while_read_then_store",
        "U6_range_second_phase",
    }
)
"""On the CPU, a read after a store into its elements is refused, whatever took the
view: T4 and T5 too, conservatively (store and read miss each other at run time). The
others are right on the CPU, as they were before the read order rule."""


WRITTEN_TWICE = frozenset({"F1_two_written_views"})
"""Two written tensors sharing memory: refused before the read order is looked at."""


def _kernel(case: dict) -> helion.Kernel:
    return helion.kernel(
        case["fn"],
        backend="mlir",
        static_shapes=True,
        config=helion.Config(block_sizes=[8] * case["blocks"]),
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def _as_list(value: object) -> list[torch.Tensor]:
    return list(value) if isinstance(value, (tuple, list)) else [value]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (lambda t: t[:, ::2], lambda t: t[:, 1::2], False),
        (lambda t: t[:, ::2], lambda t: t[:, ::4], True),
        (lambda t: t[:4], lambda t: t[4:], False),
        (lambda t: t[:5], lambda t: t[4:], True),
        (lambda t: t.t(), lambda t: t, True),
        (lambda t: t[:, :2], lambda t: t[:, 2:4], False),
        (lambda t: t[:0], lambda t: t, False),
    ],
)
def test_share_elements(left: object, right: object, expected: bool) -> None:
    """Elements, not byte spans: interleaved views share none."""
    tensor = torch.zeros(8, 8)
    assert driver._share_elements(left(tensor), right(tensor)) is expected


def test_share_elements_of_other_element_sizes_are_shared() -> None:
    tensor = torch.zeros(8, dtype=torch.float32)
    bytes_view = torch.frombuffer(tensor.numpy(), dtype=torch.uint8)
    assert driver._share_elements(tensor, bytes_view) is True


@pytest.mark.parametrize("name", sorted(cases.CASES))
def test_case(name: str) -> None:
    case = cases.CASES[name]
    kernel = _kernel(case)
    for call in range(2):
        torch.manual_seed(call)
        x = torch.randn(cases.M, cases.N)
        make = case["args"] or (lambda value: (value,))
        expected = _as_list(case["ref"](*[v.clone() for v in make(x.clone())]))
        if name in REFUSED:
            message = "share memory" if name in WRITTEN_TWICE else "two kernel calls"
            with pytest.raises(UnsupportedOperationError, match=message):
                kernel(*make(x))
            continue
        for out, ref in zip(_as_list(kernel(*make(x))), expected, strict=True):
            assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), name


def test_read_after_a_store_is_refused_on_the_plan_path_too() -> None:
    """The native plan's first call goes through ``call_entry``; a call it can't plan
    goes through it each time: a second call is refused like the first."""
    kernel = _kernel(cases.CASES["N2_caller_view_w1_r2"])
    make = cases.CASES["N2_caller_view_w1_r2"]["args"]
    for call in range(3):
        torch.manual_seed(call)
        with pytest.raises(UnsupportedOperationError, match="two kernel calls"):
            kernel(*make(torch.randn(cases.M, cases.N)))


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        ((0, 10), False),
        ((0, 11), True),
        ((15, 20), True),
        ((30, 40), False),
        ((0, 100), True),
        ((25, 26), False),
        ((39, 41), True),
        ((55, 60), False),
    ],
)
def test_overlaps_any_of_sorted_spans(span: tuple[int, int], expected: bool) -> None:
    """The search covers an input wider than the spans it overlaps, and each side."""
    tensor = mock.Mock()
    tensor.numel.return_value = span[1] - span[0]
    with mock.patch.object(driver, "_byte_span", return_value=span):
        assert driver._overlaps_any(tensor, [(10, 20), (40, 50)]) is expected
