"""What the full key of an mlir kernel tells apart beyond stock Helion: ints in
closure cells by value and ``hl.constexpr`` by type and bits. Each of these ran an
earlier compile silently, on stock Helion too
(agent-wiki/upstream-repro/helion-function-key). Marked ``helion_internals``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple
import warnings

import helion
import helion.language as hl
import numpy as np
import pytest
import torch

import helion_mlir_backend
from helion_mlir_backend import inject
from helion_mlir_backend._compiler import native

pytestmark = pytest.mark.helion_internals


def _kernel(fn: object) -> helion.Kernel:
    return helion.kernel(
        fn, backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[8])
    )


def _apply(x: torch.Tensor, fn: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = fn(x[tile])
    return out


def _value(out: torch.Tensor) -> float:
    return out.flatten()[0].item()


@dataclasses.dataclass
class _Field:
    n: int


def _divide(x: torch.Tensor, s: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] / s
    return out


def _divide_annotated(x: torch.Tensor, s: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] / s
    return out


def _divide_by(s: object) -> object:
    return lambda t: t / s


@pytest.mark.parametrize("zeros", [(0.0, -0.0), (-0.0, 0.0)], ids=["pos", "neg"])
@pytest.mark.parametrize("form", ["plain", "wrapped", "cell"])
def test_signed_zero_constexpr_compiles_apart(
    form: str, zeros: tuple[float, float]
) -> None:
    x = torch.ones(32)
    if form == "plain":
        kernel = _kernel(_divide_annotated)
        call = lambda s: kernel(x, s)  # noqa: E731
    elif form == "wrapped":
        kernel = _kernel(_divide)
        call = lambda s: kernel(x, hl.constexpr(s))  # noqa: E731
    else:
        kernel = _kernel(_apply)
        call = lambda s: kernel(x, _divide_by(hl.constexpr(s)))  # noqa: E731
    for s in (*zeros, *zeros):
        for _ in range(2):  # the second call takes the dispatch cache
            assert _value(call(s)) == math.copysign(math.inf, s)
    assert len(kernel._bound_kernels) == 2


def test_constexpr_type_is_part_of_the_key() -> None:
    kernel = _kernel(_divide)
    x = torch.ones(8)
    key = kernel._base_specialization_key
    wrapped = [hl.constexpr(v) for v in (1, 1.0, True)]
    assert len({key((x, v)) for v in wrapped}) == 3
    assert key((x, hl.constexpr(2.0))) == key((x, hl.constexpr(2.0)))
    annotated = _kernel(_divide_annotated)
    assert annotated._base_specialization_key((x, 0.0)) != (
        annotated._base_specialization_key((x, -0.0))
    )
    assert annotated._base_specialization_key((x, 1)) != (
        annotated._base_specialization_key((x, 1.0))
    )


@pytest.mark.parametrize("kind", [np.float64, np.float32], ids=["f64", "f32"])
def test_numpy_float_in_a_constexpr_cell_keeps_its_sign_of_zero(kind: type) -> None:
    key = _kernel(_apply)._base_specialization_key
    x = torch.ones(8)

    def cell(value: float) -> object:
        return _divide_by(hl.constexpr(kind(value)))

    assert key((x, cell(0.0))) != key((x, cell(-0.0)))
    assert key((x, cell(0.0))) == key((x, cell(0.0)))


def test_numpy_float64_constexpr_compiles_apart() -> None:
    kernel = _kernel(_apply)
    x = torch.ones(32)
    for value in (0.0, -0.0):
        out = kernel(x, _divide_by(hl.constexpr(np.float64(value))))
        assert _value(out) == math.copysign(math.inf, value)


def test_constexpr_tuple_keys_its_elements() -> None:
    key = _kernel(_divide)._base_specialization_key
    x = torch.ones(8)
    assert key((x, hl.constexpr((0.0,)))) != key((x, hl.constexpr((-0.0,))))
    assert key((x, hl.constexpr((1,)))) != key((x, hl.constexpr((1.0,))))


def test_other_backends_keep_helions_constexpr_key() -> None:
    kernel = helion.kernel(_divide, backend="triton")
    x = torch.ones(8)
    assert kernel._specialization_key(hl.constexpr(0.0)) == 0.0
    base = kernel._base_specialization_key
    assert base((x, hl.constexpr(0.0))) == base((x, hl.constexpr(-0.0)))


def test_mutated_list_default_compiles_apart() -> None:
    held = [1]

    def fn(t, h=held):
        if h[0] > 2:
            return t * 10.0
        return t * 20.0

    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert _value(kernel(x, fn)) == 20.0
    held[0] = 3
    assert _value(kernel(x, fn)) == 10.0


def test_mutated_dataclass_default_compiles_apart() -> None:
    held = _Field(1)

    def fn(t, h=held):
        if h.n > 2:
            return t * 10.0
        return t * 20.0

    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert _value(kernel(x, fn)) == 20.0
    held.n = 3
    assert _value(kernel(x, fn)) == 10.0


def _scale_ints(x: torch.Tensor, v: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] * v
    return out


def _ir(value: object) -> str:
    x = torch.ones(8, dtype=torch.int32)
    return str(helion_mlir_backend.generate_mlir(_kernel(_scale_ints), [x, value]))


def test_constexpr_type_changes_the_ir() -> None:
    """Why the type belongs in the key: on an int tensor, ``3`` multiplies as ints
    and ``3.0`` promotes to floats; ``1`` and ``1.0`` fold differently; ``True``
    has its own helper."""
    assert "arith.muli" in _ir(hl.constexpr(3))
    assert "arith.mulf" in _ir(hl.constexpr(3.0))
    one, one_float, true = (
        _ir(hl.constexpr(1)),
        _ir(hl.constexpr(1.0)),
        _ir(hl.constexpr(True)),
    )
    assert "arith.sitofp" not in one
    assert "arith.sitofp" in one_float
    assert one != true


@pytest.mark.parametrize(
    "order", [(1, 1.0), (1.0, 1)], ids=["int-first", "float-first"]
)
def test_results_follow_the_constexpr_type(order: tuple[object, object]) -> None:
    """Through f32, 2**24 + 1 rounds: multiplying by ``1.0`` differs from ``1``, so
    sharing a compile gave a wrong number."""
    kernel = _kernel(_scale_ints)
    x = torch.full((8,), 2**24 + 1, dtype=torch.int32)
    for value in (*order, *order):
        want = (x * value).to(torch.int32)
        assert kernel(x, hl.constexpr(value)).equal(want), value
    assert len(kernel._bound_kernels) == 2
    assert kernel.bind((x, hl.constexpr(1))) is not kernel.bind((x, hl.constexpr(1.0)))


@pytest.mark.skipif(
    native.module() is None, reason="the native extension could not be built"
)
def test_negative_zero_is_no_native_hit_of_a_plan_on_zero() -> None:
    kernel = _kernel(_divide)
    x = torch.ones(32)
    positive = hl.constexpr(0.0)
    kernel(x, positive)
    kernel(x, positive)
    bound = kernel.bind((x, positive))
    run = bound._run
    assert run.planned and run.native_calls == 1
    # Through the kernel: its own bound kernel.
    assert _value(kernel(x, hl.constexpr(-0.0))) == -math.inf
    assert run.native_calls == 1
    # A bound kernel called directly: the plan's own check refuses it.
    bound(x, hl.constexpr(-0.0))
    assert run.native_calls == 1
    bound(x, hl.constexpr(0.0))
    assert run.native_calls == 2


# -- ints read at compile time -------------------------------------------------


class _Pair(NamedTuple):
    n: int
    other: float


def _gt(n: int) -> object:
    def fn(t):
        if n > 2:
            return t * 10.0
        return t * 20.0

    return fn


def _eq(n: int) -> object:
    def fn(t):
        if n == 3:
            return t * 10.0
        return t * 20.0

    return fn


def _mod(n: int) -> object:
    def fn(t):
        if n % 2 == 1:
            return t * 10.0
        return t * 20.0

    return fn


def _in_tuple(n: int) -> object:
    held = (n, 1)

    def fn(t):
        if held[0] > 2:
            return t * 10.0
        return t * 20.0

    return fn


def _in_list(n: int) -> object:
    held = [n]

    def fn(t):
        if held[0] > 2:
            return t * 10.0
        return t * 20.0

    return fn


def _in_dict(n: int) -> object:
    held = {"n": n}

    def fn(t):
        if held["n"] > 2:
            return t * 10.0
        return t * 20.0

    return fn


def _in_dataclass(n: int) -> object:
    held = _Field(n)

    def fn(t):
        if held.n > 2:
            return t * 10.0
        return t * 20.0

    return fn


def _in_namedtuple(n: int) -> object:
    held = _Pair(n, 1.0)

    def fn(t):
        if held.n > 2:
            return t * 10.0
        return t * 20.0

    return fn


def _in_inner_function(n: int) -> object:
    inner = _gt(n)
    return lambda t: inner(t)


@pytest.mark.parametrize(
    "make,low,high",
    [
        (_gt, 1, 3),
        (_eq, 1, 3),
        (_mod, 2, 3),
        (_in_tuple, 1, 3),
        (_in_list, 1, 3),
        (_in_dict, 1, 3),
        (_in_dataclass, 1, 3),
        (_in_namedtuple, 1, 3),
        (_in_inner_function, 1, 3),
    ],
    ids=lambda v: getattr(v, "__name__", None) and v.__name__.lstrip("_"),
)
def test_int_in_a_cell_compiles_apart(make: object, low: int, high: int) -> None:
    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert [_value(kernel(x, make(n))) for n in (low, high, low)] == [20.0, 10.0, 20.0]
    assert len(kernel._bound_kernels) == 2


def test_rebinding_an_int_cell_compiles_apart() -> None:
    n = 1

    def fn(t):
        if n > 2:
            return t * 10.0
        return t * 20.0

    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert _value(kernel(x, fn)) == 20.0
    n = 3
    assert _value(kernel(x, fn)) == 10.0
    n = 1
    assert _value(kernel(x, fn)) == 20.0


def test_int_cell_as_a_value_keeps_one_compile() -> None:
    """Read at run time, as Helion's model has it: nothing to key."""
    kernel = _kernel(_apply)
    x = torch.ones(32)
    for n in (2, 3, 2):
        assert _value(kernel(x, _times(n))) == float(n)
    assert len(kernel._bound_kernels) == 1
    assert len(kernel._dispatch_cache) == 1


@pytest.mark.parametrize(
    "make,change",
    [
        (lambda: [1], lambda held: held.__setitem__(0, 3)),
        (lambda: _Field(1), lambda held: setattr(held, "n", 3)),
    ],
    ids=["list", "dataclass"],
)
def test_mutated_cell_branched_on_compiles_apart(make: object, change: object) -> None:
    held = make()

    def fn(t):
        if (held[0] if isinstance(held, list) else held.n) > 2:
            return t * 10.0
        return t * 20.0

    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert _value(kernel(x, fn)) == 20.0
    change(held)
    assert _value(kernel(x, fn)) == 10.0


def _value_or_branch(gate: int, n: int) -> object:
    """``n`` is read as a value while ``gate`` is 0, branched on otherwise."""

    def fn(t):
        if gate > 0:
            if n > 2:
                return t * 10.0
            return t * 20.0
        return t * n

    return fn


def test_a_guard_on_a_cell_reaches_later_closures_of_the_same_code() -> None:
    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert _value(kernel(x, _value_or_branch(0, 2))) == 2.0
    assert _value(kernel(x, _value_or_branch(0, 3))) == 3.0
    assert len(kernel._bound_kernels) == 1
    assert _value(kernel(x, _value_or_branch(1, 1))) == 20.0
    assert _value(kernel(x, _value_or_branch(1, 3))) == 10.0
    assert _value(kernel(x, _value_or_branch(1, 5))) == 10.0
    assert _value(kernel(x, _value_or_branch(0, 7))) == 7.0


def test_a_guard_in_a_nested_closure_keeps_the_outer_value_only_cell_free() -> None:
    def make(n: int, scale: int) -> object:
        inner = _gt(n)
        return lambda t: inner(t) * scale

    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert _value(kernel(x, make(1, 1))) == 20.0
    assert _value(kernel(x, make(1, 2))) == 40.0
    assert len(kernel._bound_kernels) == 1
    assert _value(kernel(x, make(3, 2))) == 20.0
    assert len(kernel._bound_kernels) == 2


def test_a_source_that_cant_be_walked_keys_the_whole_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail safe: a finer key costs a compile, a coarser one a wrong answer."""
    walk = inject._source_steps

    def unwalkable(source: object) -> object:
        step = walk(source)
        return None if step is None else (step[0], None)

    monkeypatch.setattr(inject, "_source_steps", unwalkable)
    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert [_value(kernel(x, _gt(n))) for n in (1, 3, 1)] == [20.0, 10.0, 20.0]
    assert len(kernel._bound_kernels) == 2


def _held_at_0(x: torch.Tensor, held: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = held[0](x[tile])
    return out


def _held_at_f(x: torch.Tensor, held: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = held["f"](x[tile])
    return out


def _held_at_0_0(x: torch.Tensor, held: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = held[0][0](x[tile])
    return out


def _outer(inner: object) -> object:
    return lambda t: inner(t)


@pytest.mark.parametrize(
    "fn,make",
    [
        (_held_at_0, lambda n: (_gt(n),)),
        (_held_at_0, lambda n: [_gt(n)]),
        (_held_at_f, lambda n: {"f": _gt(n)}),
        (_held_at_0_0, lambda n: ((_gt(n),),)),
        (_held_at_0, lambda n: (_outer(_gt(n)),)),
    ],
    ids=["tuple", "list", "dict", "nested-tuple", "nested-function"],
)
def test_a_guard_in_a_function_held_in_a_container_argument(
    fn: object, make: object
) -> None:
    kernel = _kernel(fn)
    x = torch.ones(32)
    assert [_value(kernel(x, make(n))) for n in (1, 3, 1)] == [20.0, 10.0, 20.0]
    assert len(kernel._bound_kernels) == 2


def test_a_guard_in_a_function_in_a_tuple_in_a_cell() -> None:
    def make(n: int) -> object:
        held = (_gt(n),)
        return lambda t: held[0](t)

    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert [_value(kernel(x, make(n))) for n in (1, 3, 1)] == [20.0, 10.0, 20.0]
    assert len(kernel._bound_kernels) == 2


def test_a_function_default_with_a_rebound_cell_compiles_apart() -> None:
    n = 1

    def inner(t):
        if n > 2:
            return t * 10.0
        return t * 20.0

    def fn(t, g=inner):
        return g(t)

    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert _value(kernel(x, fn)) == 20.0
    n = 3
    assert _value(kernel(x, fn)) == 10.0


def test_a_function_with_an_empty_cell_has_no_ints_and_doesnt_raise() -> None:
    def make() -> object:
        def fn(t):
            return later

        if False:
            later = 1
        return fn

    assert inject._cell_ints((make(),)) == ((None,),)


def test_the_next_compile_repairs_a_stale_guarded_list() -> None:
    """With no lock, a thread may assign an older list after a newer one."""
    kernel = _kernel(_apply)
    kernel.bind((torch.ones(32), _gt(1)))
    assert len(kernel._helion_mlir_guarded) == 1
    kernel._helion_mlir_guarded = []
    kernel.bind((torch.ones(16), _gt(1)))
    assert len(kernel._helion_mlir_guarded) == 1


def test_source_steps() -> None:
    from torch._dynamo.source import AttrSource
    from torch._dynamo.source import GetItemSource
    from torch._dynamo.source import GlobalSource
    from torch._dynamo.source import LocalSource
    from torch._dynamo.source import NNModuleSource

    local = LocalSource("fn")
    cell = AttrSource(
        GetItemSource(AttrSource(local, "__closure__"), 0), "cell_contents"
    )
    assert inject._source_steps(local) == ("fn", ())
    assert inject._source_steps(cell) == (
        "fn",
        (("attr", "__closure__"), ("item", 0), ("attr", "cell_contents")),
    )
    assert inject._source_steps(AttrSource(GlobalSource("G"), "member")) is None
    assert inject._source_steps(NNModuleSource(local)) == ("fn", None)


def test_walking_a_function_of_other_code_keys_all_its_ints() -> None:
    steps = (("attr", "__closure__"), ("item", 5), ("attr", "cell_contents"))
    assert inject._walked_ints(_gt(3), steps) == (3,)
    assert inject._walked_ints(_gt(3), None) == (3,)


def test_the_fast_key_holds_guarded_cell_values() -> None:
    kernel = _kernel(_apply)
    x = torch.ones(32)
    key = kernel._fast_dispatch_key
    assert key((x, _gt(1))) == key((x, _gt(3)))  # no guard seen yet
    kernel(x, _gt(1))
    assert key((x, _gt(1))) != key((x, _gt(3)))
    held = (x, _in_tuple(1)), (x, _in_tuple(3))
    other = _kernel(_apply)
    other(*held[0])
    assert other._fast_dispatch_key(held[0]) != other._fast_dispatch_key(held[1])


def _times(n: object) -> object:
    return lambda t: t * n


def test_float_and_bool_cells_stay_keyed_by_type() -> None:
    kernel = _kernel(_apply)
    x = torch.ones(32)
    assert _value(kernel(x, _times(2.0))) == 2.0
    assert _value(kernel(x, _times(3.0))) == 3.0
    assert len(kernel._bound_kernels) == 1
    other = _kernel(_apply)
    key = other._base_specialization_key
    assert key((x, _times(True))) == key((x, _times(False)))


# -- an int argument into a traced function -----------------------------------


def _branch_on(t: torch.Tensor, n: int) -> torch.Tensor:
    if n > 2:
        return t * 10.0
    return t * 20.0


def _int_arg_to_called(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = _branch_on(x[tile], n)
    return out


def _int_arg_to_argument(x: torch.Tensor, fn: object, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = fn(x[tile], n)
    return out


def _int_arg_direct_if(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        if n > 2:
            out[tile] = x[tile] * 10.0
        else:
            out[tile] = x[tile] * 20.0
    return out


def _int_arg_as_value(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] * n
    return out


def _int_arg_unused(x: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] * 2.0
    return out


def test_int_argument_into_a_called_function() -> None:
    kernel = _kernel(_int_arg_to_called)
    x = torch.ones(32)
    assert [_value(kernel(x, n)) for n in (1, 3, 1)] == [20.0, 10.0, 20.0]
    assert len(kernel._bound_kernels) == 2


def test_int_argument_into_a_function_argument() -> None:
    kernel = _kernel(_int_arg_to_argument)
    x = torch.ones(32)
    values = [_value(kernel(x, _branch_on, n)) for n in (1, 3, 1)]
    assert values == [20.0, 10.0, 20.0]
    assert len(kernel._bound_kernels) == 2


def _int_arg_with_default(x: torch.Tensor, n: int = 1) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = _branch_on(x[tile], n)
    return out


def test_guarded_int_argument_left_to_its_default() -> None:
    kernel = _kernel(_int_arg_with_default)
    x = torch.ones(32)
    assert _value(kernel(x)) == 20.0
    assert _value(kernel(x, 3)) == 10.0
    assert _value(kernel(x)) == 20.0
    assert _value(kernel(x, 1)) == 20.0


def _int_in_tuple_arg_to_called(x: torch.Tensor, held: tuple) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = _branch_on(x[tile], held[0])
    return out


@pytest.mark.parametrize(
    "make", [lambda n: (n, 0), lambda n: [n, 0]], ids=["tuple", "list"]
)
def test_int_in_a_container_argument_into_a_called_function(make: object) -> None:
    kernel = _kernel(_int_in_tuple_arg_to_called)
    x = torch.ones(32)
    assert [_value(kernel(x, make(n))) for n in (1, 3, 1)] == [20.0, 10.0, 20.0]
    assert len(kernel._bound_kernels) == 2


@pytest.mark.parametrize(
    "fn,want",
    [(_int_arg_direct_if, (20.0, 10.0, 20.0)), (_int_arg_as_value, (1.0, 3.0, 1.0))],
    ids=["direct-if", "value"],
)
def test_int_argument_without_a_guard_keeps_one_compile(
    fn: object, want: tuple[float, ...]
) -> None:
    kernel = _kernel(fn)
    x = torch.ones(32)
    assert tuple(_value(kernel(x, n)) for n in (1, 3, 1)) == want
    assert len(kernel._bound_kernels) == 1


@pytest.mark.skipif(
    native.module() is None, reason="the native extension could not be built"
)
def test_int_argument_without_a_guard_keeps_its_native_hit() -> None:
    kernel = _kernel(_int_arg_unused)
    x = torch.ones(32)
    kernel(x, 1)
    kernel(x, 1)
    run = kernel.bind((x, 1))._run
    assert run.planned and run.native_calls == 1
    kernel(x, 3)  # one compile: this value takes its general path
    assert len(kernel._bound_kernels) == 1
    assert run.native_calls == 1
    kernel(x, 1)
    assert run.native_calls == 2


@pytest.mark.skipif(
    native.module() is None, reason="the native extension could not be built"
)
def test_a_guarded_int_argument_never_runs_natively_into_another_compile() -> None:
    """After a plan on n=1, n=3 misses the dispatch cache (it holds the value) and
    its extras give another bound kernel with its own plan."""
    kernel = _kernel(_int_arg_to_called)
    x = torch.ones(32)
    kernel(x, 1)
    kernel(x, 1)
    low = kernel.bind((x, 1))
    native_calls = getattr(low._run, "native_calls", None)
    assert _value(kernel(x, 3)) == 10.0
    assert _value(kernel(x, 3)) == 10.0
    high = kernel.bind((x, 3))
    assert high is not low
    assert high._run is not low._run
    assert getattr(low._run, "native_calls", None) == native_calls
    assert _value(kernel(x, 1)) == 20.0


def _both(t: torch.Tensor, n: int, m: int) -> torch.Tensor:
    if n > 2 and m > 2:
        return t * 10.0
    return t * 20.0


def _two_ints(x: torch.Tensor, n: int, m: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = _both(x[tile], n, m)
    return out


def test_a_guard_first_recorded_by_a_later_compile_reaches_the_key() -> None:
    """``n > 2`` short-circuits on the first values: only a later compile records a
    guard on ``m``, and Helion asks for a signature's extras once."""
    kernel = _kernel(_two_ints)
    x = torch.ones(32)
    for args, want in [
        ((1, 1), 20.0),
        ((3, 1), 20.0),
        ((3, 5), 10.0),
        ((3, 1), 20.0),
        ((1, 5), 20.0),
        ((3, 7), 10.0),
    ]:
        assert _value(kernel(x, *args)) == want, args


def _gated_by_cell(gate: object) -> object:
    def fn(t, n):
        if gate > 0:
            if n > 2:
                return t * 10.0
            return t * 20.0
        return t * 30.0

    return fn


@pytest.mark.parametrize("wrap", [int, hl.constexpr], ids=["int", "constexpr"])
def test_a_guard_gated_by_a_cell_reaches_the_key(wrap: object) -> None:
    kernel = _kernel(_int_arg_to_argument)
    x = torch.ones(32)
    closed, opened = _gated_by_cell(wrap(0)), _gated_by_cell(wrap(1))
    for fn, n, want in [
        (closed, 1, 30.0),
        (closed, 3, 30.0),
        (opened, 1, 20.0),
        (opened, 3, 10.0),
        (opened, 5, 10.0),
        (opened, 1, 20.0),
        (closed, 5, 30.0),
    ]:
        assert _value(kernel(x, fn, n)) == want, (fn, n)


def _gated_on_size(t: torch.Tensor, size: int, n: int) -> torch.Tensor:
    if size > 20:
        if n > 2:
            return t * 10.0
        return t * 20.0
    return t * 30.0


def _shape_gate(x: torch.Tensor, n: int) -> torch.Tensor:
    size = x.shape[0]
    out = torch.empty_like(x)
    for tile in hl.tile(size):
        out[tile] = _gated_on_size(x[tile], size, n)
    return out


def _specialized_gate(x: torch.Tensor, n: int) -> torch.Tensor:
    size = hl.specialize(x.shape[0])
    out = torch.empty_like(x)
    for tile in hl.tile(size):
        out[tile] = _gated_on_size(x[tile], size, n)
    return out


@pytest.mark.parametrize(
    "kernel_of",
    [
        lambda: _kernel(_shape_gate),
        lambda: helion.kernel(
            _specialized_gate, backend="mlir", config=helion.Config(block_sizes=[8])
        ),
    ],
    ids=["static-shapes", "specialized"],
)
def test_a_guard_gated_by_a_shape_reaches_the_key(kernel_of: object) -> None:
    kernel = kernel_of()
    small, large = torch.ones(16), torch.ones(32)
    for x, n, want in [
        (small, 1, 30.0),
        (small, 3, 30.0),
        (large, 1, 20.0),
        (large, 3, 10.0),
        (large, 5, 10.0),
        (large, 1, 20.0),
        (small, 5, 30.0),
    ]:
        assert _value(kernel(x, n)) == want, (x.shape, n)


def _specializes_when(t: torch.Tensor, n: int, size: object) -> torch.Tensor:
    if n > 2:
        if hl.specialize(size) > 20:
            return t * 10.0
        return t * 20.0
    return t * 30.0


def _specialized_under_a_branch(x: torch.Tensor, n: int) -> torch.Tensor:
    size = x.shape[0]
    out = torch.empty_like(x)
    for tile in hl.tile(size):
        out[tile] = _specializes_when(x[tile], n, size)
    return out


def test_a_specialize_first_reached_by_a_later_compile_reaches_the_key() -> None:
    """Helion's extras too are asked for once, from the first bound kernel."""
    kernel = helion.kernel(
        _specialized_under_a_branch,
        backend="mlir",
        static_shapes=False,
        config=helion.Config(block_sizes=[8]),
    )
    assert _value(kernel(torch.ones(32), 1)) == 30.0
    assert _value(kernel(torch.ones(30), 3)) == 10.0
    assert _value(kernel(torch.ones(18), 3)) == 20.0


def _specializes_m_when(t: torch.Tensor, n: int, m: object) -> torch.Tensor:
    if n > 2:
        if hl.specialize(m) > 6:
            return t * 10.0
        return t * 20.0
    return t * 2.0


def _with_defaults(x: torch.Tensor, n: int = 1, m: int = 5) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = _specializes_m_when(x[tile], n, m)
    return out


def test_a_short_call_after_a_later_compiles_extras() -> None:
    """Keyed as made: a later compile's extra may read a parameter left to its
    default, which raised ``IndexError`` here."""
    kernel = _kernel(_with_defaults)
    x = torch.ones(32)
    assert _value(kernel(x, 3, 7)) == 10.0
    assert _value(kernel(x)) == 2.0
    assert _value(kernel(x, 3)) == 20.0
    assert _value(kernel(x, 3, 7)) == 10.0


def test_other_backends_get_no_extra() -> None:
    kernel = helion.kernel(_int_arg_as_value, backend="triton")
    assert kernel.bind((torch.ones(8), 1))._specialize_extra() == []


# -- a compile per value warns ------------------------------------------------


def test_many_compiles_from_guarded_int_cells_warn_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inject, "_MANY_INT_COMPILES", 2)
    kernel = _kernel(_apply)
    x = torch.ones(32)
    for n in (1, 3):
        kernel.bind((x, _gt(n)))
    with pytest.warns(UserWarning, match="one compile per value") as caught:
        for n in (4, 5, 6):
            kernel.bind((x, _gt(n)))
    assert len(caught) == 1
    assert "_apply" in str(caught[0].message)


def test_value_only_int_cells_dont_count_towards_the_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inject, "_MANY_INT_COMPILES", 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*one compile per value")
        kernel = _kernel(_apply)
        x = torch.ones(32)
        for n in range(4):
            kernel.bind((x, _times(n)))


def test_many_compiles_from_guarded_int_arguments_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inject, "_MANY_INT_COMPILES", 2)
    kernel = _kernel(_int_arg_to_called)
    x = torch.ones(32)
    with pytest.warns(UserWarning, match="one compile per value"):
        for n in (1, 2, 3, 4):
            kernel.bind((x, n))


def test_value_only_int_cells_of_new_compiles_dont_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each shape compiles, and each has another value-only int."""
    monkeypatch.setattr(inject, "_MANY_INT_COMPILES", 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*one compile per value")
        kernel = _kernel(_apply)
        for size in (8, 16, 32):
            kernel.bind((torch.ones(size), _times(size)))


def test_other_compiles_dont_count_towards_the_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inject, "_MANY_INT_COMPILES", 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*one compile per value")
        kernel = _kernel(_apply)
        for size in (8, 16, 32):
            kernel.bind((torch.ones(size), _times(hl.constexpr(float(size)))))
