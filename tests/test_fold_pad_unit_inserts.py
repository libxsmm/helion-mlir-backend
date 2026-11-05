"""``fold_pad_unit_inserts``: a pad put back into unit dims by an insert into an
empty tensor as a reshape, so that ``materialize_operand_pads`` reads through it."""

from __future__ import annotations

from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from mlir import ir
from mlir.dialects import transform
import pytest

from helion_mlir_backend._compiler.mlir_transforms import HelionTransformDialect
from helion_mlir_backend._compiler.mlir_transforms import fold_pad_unit_inserts


def _apply(source: str) -> str:
    with ir.Context(), ir.Location.unknown():
        HelionTransformDialect.load()
        module = ir.Module.parse(source)
        with schedule_boilerplate() as (schedule, named_seq):
            fold_pad_unit_inserts(
                lh_transform.match_op(named_seq.bodyTarget, "func.func")
            )
            transform.yield_()
        schedule.body.operations[0].apply(module.operation)
        module.operation.verify()
        return str(module)


def _parsed(source: str) -> str:
    with ir.Context():
        return str(ir.Module.parse(source))


_PAD = """
  %c0 = arith.constant 0 : index
  %z = arith.constant 0.0 : bf16
  %p = tensor.pad %t low[0, 0] high[0, 30] {{
  ^bb0(%i: index, %j: index):
    tensor.yield %z : bf16
  }} : tensor<512x66xbf16> to tensor<512x96xbf16>"""


def _insert_of_pad(result: str, insert: str, dest: str | None = None) -> str:
    empty = "" if dest else f"%e = tensor.empty() : tensor<{result}xbf16>"
    return f"""
func.func @f(%t: tensor<512x66xbf16>, %d: tensor<{result}xbf16>, %s: index) -> tensor<{result}xbf16> {{{_PAD.format()}
  {empty}
  %r = tensor.insert_slice %p into {dest or "%e"}{insert} : tensor<512x96xbf16> into tensor<{result}xbf16>
  return %r : tensor<{result}xbf16>
}}
"""


@pytest.mark.parametrize(
    ("result", "insert", "groups"),
    [
        ("1x512x96", "[0, 0, 0] [1, 512, 96] [1, 1, 1]", "[[0, 1], [2]]"),
        ("512x1x96", "[0, 0, 0] [512, 1, 96] [1, 1, 1]", "[[0], [1, 2]]"),
        ("512x96x1", "[0, 0, 0] [512, 96, 1] [1, 1, 1]", "[[0], [1, 2]]"),
        (
            "1x512x1x96x1",
            "[0, 0, 0, 0, 0] [1, 512, 1, 96, 1] [1, 1, 1, 1, 1]",
            "[[0, 1], [2, 3, 4]]",
        ),
    ],
)
def test_insert_of_pad_into_empty_is_a_reshape(
    result: str, insert: str, groups: str
) -> None:
    out = _apply(_insert_of_pad(result, insert))
    assert "tensor.insert_slice" not in out
    assert f"tensor.expand_shape %padded {groups}" in out.replace("%0", "%padded")


@pytest.mark.parametrize(
    ("result", "insert", "dest"),
    [
        # Not at the origin, or not covering the destination.
        ("2x512x96", "[1, 0, 0] [1, 512, 96] [1, 1, 1]", None),
        ("1x512x96", "[%s, 0, 0] [1, 512, 96] [1, 1, 1]", None),
        # A destination with contents.
        ("1x512x96", "[0, 0, 0] [1, 512, 96] [1, 1, 1]", "%d"),
    ],
)
def test_other_inserts_of_a_pad_stay(
    result: str, insert: str, dest: str | None
) -> None:
    source = _insert_of_pad(result, insert, dest)
    assert _apply(source) == _parsed(source)


def test_insert_of_another_source_stays() -> None:
    source = """
func.func @f(%t: tensor<512x96xbf16>) -> tensor<1x512x96xbf16> {
  %e = tensor.empty() : tensor<1x512x96xbf16>
  %r = tensor.insert_slice %t into %e[0, 0, 0] [1, 512, 96] [1, 1, 1] : tensor<512x96xbf16> into tensor<1x512x96xbf16>
  return %r : tensor<1x512x96xbf16>
}
"""
    assert _apply(source) == _parsed(source)
