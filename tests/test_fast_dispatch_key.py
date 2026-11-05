"""The fast dispatch key of mlir kernels called with functions and
``hl.constexpr`` (``inject._fast_key_functions_and_constexpr``).

The danger is a key coarser than Helion's full key: a call then silently runs
another call's compile. Marked ``helion_internals``: this reads ``bind``, the
dispatch cache and ``_run``.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
import threading
import types

import helion
import helion.language as hl
from helion.language.constexpr import ConstExpr
from helion.runtime.kernel import Kernel
import pytest
import torch

from helion_mlir_backend import inject
from helion_mlir_backend._compiler import native

pytestmark = [
    pytest.mark.helion_internals,
    pytest.mark.skipif(
        native.module() is None, reason="the native extension could not be built"
    ),
]


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


def _apply(x: torch.Tensor, fn: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = fn(x[tile])
    return out


def _apply_then_scale(x: torch.Tensor, fn: object, scale: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = fn(x[tile]) * scale
    return out


def _annotated(x: torch.Tensor, v: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile] / v
    return out


def _kernel(fn: object = _apply, *, block: int = 8, **options: object) -> helion.Kernel:
    return helion.kernel(
        fn, backend="mlir", static_shapes=True, config=_cfg(block), **options
    )


def _scaled_by(value: object) -> object:
    return lambda t: t * value


def _first(out: torch.Tensor) -> float:
    return out.flatten()[0].item()


def _run_of(kernel: helion.Kernel, *args: object) -> object:
    return kernel.bind(args)._run


@pytest.fixture
def count_binds(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    binds: list[tuple] = []
    original = Kernel.bind

    def counting(self: Kernel, args: tuple) -> object:
        binds.append(args)
        return original(self, args)

    monkeypatch.setattr(Kernel, "bind", counting)
    return binds


# -- the property: one fast key, one BoundKernel ------------------------------


@dataclasses.dataclass
class _Field:
    value: float


def _other_module_globals() -> dict[str, object]:
    return {"__name__": __name__, "torch": torch}


def _random_cell(rng: random.Random) -> object:
    pool: list[object] = [
        1,
        2,
        True,
        False,
        1.0,
        2.0,
        0.0,
        -0.0,
        "a",
        "b",
        None,
        torch.float32,
        torch.float64,
        torch.device("cpu"),
        hl.constexpr(1),
        hl.constexpr(1.0),
        hl.constexpr(True),
        hl.constexpr(0.0),
        hl.constexpr(-0.0),
        hl.constexpr("a"),
        (1, 2),
        (1, 2.0),
        [1, 2],
        {"k": 1},
        _Field(1.0),
        torch.ones(4),
        torch.ones(5),
        torch.ones(4, dtype=torch.float64),
        torch.ones(2, 3),
        torch.ones(2, 3).t(),
        torch.nn.Parameter(torch.ones(4)),
        torch.relu,
    ]
    return rng.choice(pool)


def _random_function(rng: random.Random, depth: int = 0) -> object:
    a = _random_cell(rng)
    b = _random_cell(rng)
    kind = rng.randrange(7)
    if kind == 0:

        def fn(t):
            return t

    elif kind == 1:

        def fn(t):
            return t, a

    elif kind == 2:

        def fn(t):
            return t, a, b

    elif kind == 3:  # another function of other code, same cell count

        def fn(t):
            return a, t

    elif kind == 4:  # defaults
        default = rng.choice([1.0, 2.0, 0.0, -0.0])

        def fn(t, d=default):
            return t, d, a

    elif kind == 5 and depth < 2:  # a function in a cell
        inner = _random_function(rng, depth + 1)

        def fn(t):
            return inner(t)

    else:  # other globals
        fn = types.FunctionType(
            (lambda t: t).__code__, _GLOBALS[rng.randrange(len(_GLOBALS))]
        )
    return fn


_GLOBALS = [_other_module_globals(), _other_module_globals()]


def _random_tensor(rng: random.Random) -> torch.Tensor:
    shape = rng.choice([(8,), (16,), (9,), (4, 8), (8, 4), (4, 4)])
    dtype = rng.choice([torch.float32, torch.float64, torch.bfloat16])
    tensor = torch.ones(*shape, dtype=dtype)
    form = rng.randrange(4)
    if form == 1 and tensor.dim() == 2:
        return tensor.t()
    if form == 2:
        return torch.nn.Parameter(tensor)
    if form == 3 and tensor.dim() == 1:
        return torch.ones(2 * shape[0], dtype=dtype)[::2]
    return tensor


def _random_argument(rng: random.Random) -> object:
    kind = rng.randrange(5)
    if kind == 0:
        return _random_function(rng)
    if kind == 1:
        return rng.choice(
            [
                hl.constexpr(1),
                hl.constexpr(1.0),
                hl.constexpr(True),
                hl.constexpr(2),
                hl.constexpr(0.0),
                hl.constexpr(-0.0),
                hl.constexpr("a"),
                hl.constexpr(None),
            ]
        )
    if kind == 2:
        return rng.choice([1, 2, 1.0, 2.0, 0.0, -0.0, True, "a", None])
    if kind == 3:
        return rng.choice([torch.relu, torch.neg, torch.float32])
    return _random_tensor(rng)


def _ignores_two(x: torch.Tensor, a0: object, a1: object) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape[0]):
        out[tile] = x[tile]
    return out


@pytest.mark.parametrize("seed", range(2))
def test_one_fast_key_binds_to_one_bound_kernel(seed: int) -> None:
    rng = random.Random(seed)
    kernel = helion.kernel(
        _ignores_two, backend="mlir", static_shapes=True, config=_cfg(8)
    )
    seen: dict[object, object] = {}
    repeats = keyed = 0
    for _ in range(100):
        x = rng.choice([torch.ones(8), torch.ones(16), torch.ones(16).double()])
        args = (x, _random_argument(rng), _random_argument(rng))
        key = kernel._fast_dispatch_key(args)
        bound = kernel.bind(args)
        if key is None:
            continue
        keyed += 1
        if key in seen:
            repeats += 1
            assert seen[key] is bound, (
                f"one fast key for two bound kernels: {args}, {key}"
            )
        seen.setdefault(key, bound)
    assert keyed > 50 and repeats > 0  # not vacuous


def test_randomized_fast_hits_match_a_fresh_compile() -> None:
    """A fast hit's result is what a new kernel compiles for the same arguments,
    bit for bit."""
    rng = random.Random(0)
    kernel = _kernel(_apply_then_scale)
    made = [
        (rng.choice([2.0, 3.0]), rng.choice([hl.constexpr(5.0), hl.constexpr(7)]))
        for _ in range(6)
    ]
    for cell, scale in made:
        x = torch.tensor([rng.random() for _ in range(32)])
        before = len(kernel._dispatch_cache)
        out = kernel(x, _scaled_by(hl.constexpr(cell)), scale)
        again = kernel(x, _scaled_by(hl.constexpr(cell)), scale)
        fresh = _kernel(_apply_then_scale)(x, _scaled_by(hl.constexpr(cell)), scale)
        assert len(kernel._dispatch_cache) >= before
        assert out.equal(fresh) and again.equal(fresh)


def test_keys_tell_apart_what_the_full_key_tells_apart() -> None:
    kernel = _kernel()
    x = torch.ones(8)

    def key(*args: object) -> object:
        made = kernel._fast_dispatch_key(args)
        assert made is not None
        return made

    def make(cell: object) -> object:
        return lambda t: t * cell

    assert key(x, make(hl.constexpr(1))) != key(x, make(hl.constexpr(1.0)))
    assert key(x, make(hl.constexpr(0.0))) != key(x, make(hl.constexpr(-0.0)))
    # A number by its type, as in the full key: read at run time.
    assert key(x, make(1)) == key(x, make(2))
    assert key(x, make(1)) != key(x, make(1.0))
    assert key(x, make(0.0)) == key(x, make(-0.0))
    assert key(x, make(torch.ones(4))) != key(x, make(torch.ones(5)))
    assert key(x, make(torch.ones(4))) == key(x, make(torch.zeros(4)))
    assert key(x, make(torch.ones(2, 3))) != key(x, make(torch.ones(2, 3).t()))
    assert key(x, make(make(1))) == key(x, make(make(2)))
    assert key(x, make(make(1))) != key(x, make(make(1.0)))
    assert key(x, hl.constexpr(1)) != key(x, 1)
    assert key(x, hl.constexpr(0.0)) != key(x, hl.constexpr(-0.0))
    assert key(x, make(1)) == key(x, make(1))
    assert key(x, hl.constexpr(2.0)) == key(x, hl.constexpr(2.0))
    assert key(x, make(math.nan)) == key(x, make(math.nan))

    def with_default(default: float) -> object:
        def fn(t, default=default):
            return t * default

        return fn

    assert key(x, with_default(0.0)) != key(x, with_default(-0.0))
    g = _other_module_globals()
    code = (lambda t: t).__code__
    assert key(x, types.FunctionType(code, g)) != key(
        x, types.FunctionType(code, _other_module_globals())
    )
    assert key(x, types.FunctionType(code, g)) == key(x, types.FunctionType(code, g))


class _SubTensor(torch.Tensor):
    pass


class _OtherSubTensor(torch.Tensor):
    pass


def test_keys_tell_apart_cells_of_other_kinds_code_and_contents() -> None:
    """Each pair would silently share a compile if its fast key matched."""
    kernel = _kernel()
    x = torch.ones(8)
    key = kernel._fast_dispatch_key

    def scale(c: object) -> object:
        return lambda t: t * c

    def add(c: object) -> object:
        return lambda t: t + c

    # Other code, equal cells.
    assert key((x, scale(1))) != key((x, add(1)))
    # Containers: what Helion keys, beyond the ints.
    assert key((x, scale([torch.ones(4)]))) != key((x, scale([torch.ones(5)])))
    assert key((x, scale((1.0,)))) != key((x, scale(("a",))))
    # Tensor subclasses: Helion's key is the same, the type isn't.
    plain = torch.ones(4)
    assert key((x, scale(plain))) != key((x, scale(plain.as_subclass(_SubTensor))))
    assert key((x, scale(plain))) == key((x, scale(torch.zeros(4))))
    assert key((x, scale(plain.as_subclass(_SubTensor)))) != key(
        (x, scale(plain.as_subclass(_OtherSubTensor)))
    )
    # A number by its type too, wrapped apart from plain.
    assert key((x, scale(1))) != key((x, scale(True)))
    assert key((x, scale(hl.constexpr(1)))) != key((x, scale(1)))
    # A constexpr in a cell apart from a plain value of its key's form.
    assert key((x, scale(hl.constexpr("a")))) != key((x, scale("a")))
    # Next to a function, the arguments go through the same path: by type, wrapped
    # apart from plain.
    with_scale = _kernel(_apply_then_scale)._fast_dispatch_key
    fn = scale(hl.constexpr(2.0))
    assert with_scale((x, fn, 1)) != with_scale((x, fn, True))
    assert with_scale((x, fn, hl.constexpr(1))) != with_scale((x, fn, 1))


def test_other_code_with_equal_cells_runs_its_own_compile() -> None:
    def scale(c: object) -> object:
        return lambda t: t * c

    def add(c: object) -> object:
        return lambda t: t + c

    kernel = _kernel()
    x = torch.full((32,), 2.0)
    for _ in range(2):
        assert _first(kernel(x, scale(hl.constexpr(3.0)))) == 6.0
        assert _first(kernel(x, add(hl.constexpr(3.0)))) == 5.0


def test_a_cell_key_that_cant_be_hashed_takes_bind() -> None:
    fn = _scaled_by([hl.constexpr([1])])
    assert _kernel()._fast_dispatch_key((torch.ones(8), fn)) is None


def test_the_distributed_flag_stays_in_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _kernel()
    args = (torch.ones(8), _scaled_by(1.0))
    first = kernel._fast_dispatch_key(args)
    monkeypatch.setattr(kernel, "_compute_is_distributed", lambda args: True)
    assert kernel._fast_dispatch_key(args) != first


def test_the_key_fn_and_distributed_flag_stay_in_the_key() -> None:
    state = {"key": 1}
    kernel = _kernel(key=lambda *args: state["key"])
    x = torch.ones(8)
    fn = _scaled_by(hl.constexpr(2.0))
    first = kernel._fast_dispatch_key((x, fn))
    state["key"] = 2
    assert first != kernel._fast_dispatch_key((x, fn))
    assert first is not None


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(hl.constexpr([1, 2]), id="constexpr-list"),
        pytest.param(hl.constexpr((1, 2)), id="constexpr-tuple"),
        pytest.param(hl.constexpr(torch.ones(1)), id="constexpr-tensor"),
        pytest.param([1, 2], id="list-argument"),
        pytest.param({"a": 1}, id="dict-argument"),
    ],
)
def test_values_without_a_fast_key_take_bind(value: object) -> None:
    assert _kernel()._fast_dispatch_key((torch.ones(8), value)) is None


@pytest.mark.parametrize(
    "value",
    [lambda t: t, torch.relu, torch.ones(1)],
    ids=["function", "builtin", "tensor"],
)
def test_constexpr_position_keyed_by_identity_takes_bind(value: object) -> None:
    """The full key holds this very object: an equal fast key would be coarser."""
    kernel = _kernel(_annotated)
    assert kernel._fast_dispatch_key((torch.ones(8), value)) is None


def test_no_tensor_argument_has_no_fast_key() -> None:
    assert _kernel()._fast_dispatch_key((_scaled_by(1.0), 1)) is None


@pytest.mark.parametrize(
    "unsupported",
    [object(), types.SimpleNamespace(a=1), type("Int", (int,), {})(1)],
    ids=["object", "namespace", "int-subclass"],
)
def test_a_cell_helion_refuses_is_reported_by_bind(unsupported: object) -> None:
    kernel = _kernel()
    x = torch.ones(8)
    kernel(x, _scaled_by(hl.constexpr(2.0)))
    fn = _scaled_by(unsupported)
    assert kernel._fast_dispatch_key((x, fn)) is None
    with pytest.raises(TypeError, match="unsupported argument type"):
        kernel(x, fn)


def test_an_empty_cell_takes_bind() -> None:
    def make() -> object:
        def fn(t):
            return t * later

        if False:
            later = 1
        return fn

    assert _kernel()._fast_dispatch_key((torch.ones(8), make())) is None


def test_other_backends_keep_helions_key() -> None:
    kernel = helion.kernel(_apply, config=_cfg(8))
    assert kernel._fast_dispatch_key((torch.ones(8), _scaled_by(1.0))) is None
    assert kernel._fast_dispatch_key((torch.ones(8), hl.constexpr(1.0))) is None


def test_helions_key_of_plain_arguments_is_used_as_it_is() -> None:
    """A tensor's key parts here are Helion's: if this fails, Helion's key changed,
    and ``inject._tensor_fast_key`` must follow."""
    original = Kernel._fast_dispatch_key._helion_mlir_original
    kernel = _kernel()
    x = torch.ones(8, 4)
    y = torch.nn.Parameter(torch.ones(4, 8).t())
    key = original(kernel, (x, y))
    assert key is not None
    assert kernel._fast_dispatch_key((x, y)) == key
    assert key[:2] == (inject._tensor_fast_key(x), inject._tensor_fast_key(y))
    assert key[2:] == (kernel._compute_is_distributed((x, y)),)
    assert original(kernel, (x, _scaled_by(1.0))) is None, (
        "Helion now keys functions itself: drop the backend's fast key"
    )
    assert original(kernel, (x, hl.constexpr(1.0))) is None, (
        "Helion now keys constexpr itself: drop the backend's fast key"
    )


# -- a new closure per call ---------------------------------------------------


def test_new_closure_per_call_binds_once_and_runs_natively(
    count_binds: list[tuple],
) -> None:
    kernel = _kernel()
    x = torch.arange(32.0)
    want = x * 2.0
    calls = 6
    for _ in range(calls):
        assert kernel(x, _scaled_by(hl.constexpr(2.0))).equal(want)
    assert len(count_binds) == 1
    run = _run_of(kernel, x, _scaled_by(hl.constexpr(2.0)))
    assert run.planned
    assert run.native_calls == calls - 1
    assert len(kernel._dispatch_cache) == 1


def test_constexpr_argument_binds_once_and_runs_natively(
    count_binds: list[tuple],
) -> None:
    kernel = _kernel(_annotated)
    x = torch.arange(32.0)
    for _ in range(4):
        assert kernel(x, hl.constexpr(2.0)).equal(x / 2.0)
    assert len(count_binds) == 1
    assert _run_of(kernel, x, hl.constexpr(2.0)).native_calls == 3


def test_same_function_object_stays_native(count_binds: list[tuple]) -> None:
    kernel = _kernel()
    x = torch.arange(32.0)
    fn = _scaled_by(hl.constexpr(2.0))
    for _ in range(4):
        kernel(x, fn)
    assert len(count_binds) == 1
    assert _run_of(kernel, x, fn).native_calls == 3


def test_a_number_changing_per_call_binds_once_and_stays_one_entry(
    count_binds: list[tuple],
) -> None:
    kernel = _kernel()
    x = torch.arange(32.0)
    for value in (2.0, 3.0, 4.0, 5.0):
        assert kernel(x, _scaled_by(value)).equal(x * value)
    assert len(count_binds) == 1
    assert len(kernel._dispatch_cache) == 1


def test_function_without_cells_is_native(count_binds: list[tuple]) -> None:
    def fn(t):
        return t * 3.0

    kernel = _kernel()
    x = torch.arange(32.0)
    for _ in range(3):
        assert kernel(x, fn).equal(x * 3.0)
    assert len(count_binds) == 1
    assert _run_of(kernel, x, fn).native_calls == 2


# -- changing cells -----------------------------------------------------------


@pytest.mark.parametrize("first,second", [(2.0, 3.0), (2, 3)], ids=["float", "int"])
def test_rebinding_a_cell_used_as_a_value_is_honoured(
    first: float, second: float
) -> None:
    value = first

    def fn(t):
        return t * value

    kernel = _kernel()
    x = torch.arange(32.0)
    assert kernel(x, fn).equal(x * first)
    assert kernel(x, fn).equal(x * first)
    value = second
    assert kernel(x, fn).equal(x * second)
    assert kernel(x, fn).equal(x * second)
    value = first
    assert kernel(x, fn).equal(x * first)
    assert len(kernel._bound_kernels) == 1
    assert len(kernel._dispatch_cache) == 1


def test_rebinding_a_constexpr_cell_compiles_and_comes_back() -> None:
    value = hl.constexpr(2.0)

    def fn(t):
        return t * value

    kernel = _kernel()
    x = torch.arange(32.0)
    assert kernel(x, fn).equal(x * 2.0)
    first_run = _run_of(kernel, x, fn)
    assert kernel(x, fn).equal(x * 2.0)
    assert first_run.native_calls == 1
    value = hl.constexpr(3.0)
    assert kernel(x, fn).equal(x * 3.0)
    assert kernel(x, fn).equal(x * 3.0)
    assert len(kernel._bound_kernels) == 2
    assert _run_of(kernel, x, fn) is not first_run
    value = hl.constexpr(2.0)
    assert kernel(x, fn).equal(x * 2.0)
    assert first_run.native_calls == 2


def test_constexpr_cell_read_at_compile_time_follows_a_rebinding() -> None:
    value = hl.constexpr(1)

    def fn(t):
        if value > 2:
            return t * 10.0
        return t * 20.0

    kernel = _kernel()
    x = torch.ones(32)
    assert _first(kernel(x, fn)) == 20.0
    value = hl.constexpr(3)
    assert _first(kernel(x, fn)) == 10.0
    value = hl.constexpr(1)
    assert _first(kernel(x, fn)) == 20.0


@pytest.mark.parametrize(
    "first,second",
    [("a", "b"), (None, 2.0), (torch.float32, torch.float64)],
    ids=["str", "none", "dtype"],
)
def test_rebinding_str_none_and_dtype_cells_compile_apart(
    first: object, second: object
) -> None:
    value = first

    def fn(t, _marker=None):
        if value == first:
            return t * 20.0
        return t * 10.0

    kernel = _kernel()
    x = torch.ones(32)
    assert _first(kernel(x, fn)) == 20.0
    assert _first(kernel(x, fn)) == 20.0
    value = second
    assert _first(kernel(x, fn)) == 10.0
    assert _first(kernel(x, fn)) == 10.0
    value = first
    assert _first(kernel(x, fn)) == 20.0


def test_tensor_cell_of_other_metadata_compiles_apart_and_new_data_is_read() -> None:
    def make(cell: torch.Tensor) -> object:
        def fn(t):
            return t * cell.shape[0]

        return fn

    kernel = _kernel()
    x = torch.ones(32)
    assert _first(kernel(x, make(torch.ones(4)))) == 4.0
    assert _first(kernel(x, make(torch.ones(5)))) == 5.0
    assert _first(kernel(x, make(torch.ones(4)))) == 4.0

    def reading(y: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        def fn(t):
            return t * cell.sum()

        return fn

    data = torch.full((32,), 2.0)
    other = torch.full((32,), 3.0)

    def combine(x: torch.Tensor, fn: object) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.shape[0]):
            out[tile] = fn(x[tile], tile)
        return out

    def by_cell(cell: torch.Tensor) -> object:
        return lambda t, tile: t * cell[tile]

    kernel = _kernel(combine)
    assert _first(kernel(x, by_cell(data))) == 2.0
    assert _first(kernel(x, by_cell(data))) == 2.0
    assert _first(kernel(x, by_cell(other))) == 3.0
    other[:] = 4.0
    assert _first(kernel(x, by_cell(other))) == 4.0


@pytest.mark.parametrize(
    "first,second",
    [
        (_scaled_by(hl.constexpr(0.0)), _scaled_by(hl.constexpr(-0.0))),
        (_scaled_by(hl.constexpr(1)), _scaled_by(hl.constexpr(1.0))),
    ],
    ids=["constexpr-signed-zero", "constexpr-type"],
)
def test_the_fast_key_runs_what_bind_runs(first: object, second: object) -> None:
    kernel = _kernel()
    x = torch.ones(32)
    assert kernel._fast_dispatch_key((x, first)) != kernel._fast_dispatch_key(
        (x, second)
    )
    kernel(x, first)
    kernel(x, second)
    for fn in (first, second, first):
        hit = kernel._dispatch_cache[kernel._fast_dispatch_key((x, fn))]
        assert hit is kernel.bind((x, fn))
    assert kernel.bind((x, first)) is not kernel.bind((x, second))


def test_annotated_signed_zero_is_coarse_in_neither_key() -> None:
    kernel = _kernel(_annotated)
    x = torch.ones(32)
    for value in (0.0, -0.0, 0.0, -0.0):
        out = kernel(x, value)
        assert _first(out) == math.copysign(math.inf, value)
    assert len(kernel._bound_kernels) == 2


# -- the surroundings ---------------------------------------------------------


def test_omp_num_threads_change_keeps_the_compile_and_the_bits(
    monkeypatch: pytest.MonkeyPatch, count_binds: list[tuple]
) -> None:
    """The fast key doesn't read the environment, as ``bind()`` doesn't: a config
    that depends on the thread count is the caller's to key."""
    kernel = _kernel()
    x = torch.randn(32)
    fn = _scaled_by(hl.constexpr(2.0))
    first = kernel(x, fn)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert kernel(x, fn).equal(first)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert kernel(x, fn).equal(first)
    assert len(count_binds) == 1
    assert _run_of(kernel, x, fn).native_calls == 2


def test_key_fn_on_the_thread_count_splits_compiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _kernel(key=lambda *args: os.environ.get("OMP_NUM_THREADS"))
    x = torch.randn(32)
    fn = _scaled_by(hl.constexpr(2.0))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    first = kernel(x, fn)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert kernel(x, fn).equal(first)
    assert len(kernel._bound_kernels) == 2


def test_set_config_after_fast_hits_runs_the_new_compile() -> None:
    kernel = _kernel()
    x = torch.randn(32)
    fn = _scaled_by(hl.constexpr(2.0))
    first = kernel(x, fn)
    kernel(x, fn)
    bound = kernel.bind((x, fn))
    old = bound._run
    assert old.native_calls == 1
    bound.set_config(_cfg(16))
    assert bound._run is not old
    out = kernel(x, fn)
    assert out.equal(first)
    assert old.native_calls == 1


def test_reset_clears_the_new_keys_too() -> None:
    kernel = _kernel()
    x = torch.randn(32)
    kernel(x, _scaled_by(hl.constexpr(2.0)))
    assert kernel._dispatch_cache
    kernel.reset()
    assert not kernel._dispatch_cache
    assert kernel(x, _scaled_by(hl.constexpr(2.0))).equal(x * 2.0)


def test_two_kernels_sharing_a_jit_entry_keep_their_own_runs() -> None:
    from helion_mlir_backend._compiler import execution

    one, two = _kernel(), _kernel()
    x = torch.randn(32)
    fn = _scaled_by(hl.constexpr(2.0))
    entries = len(execution._JIT_CACHE)
    assert one(x, fn).equal(x * 2.0)
    after_first = len(execution._JIT_CACHE)
    assert after_first >= entries
    assert two(x, fn).equal(x * 2.0)
    assert len(execution._JIT_CACHE) == after_first, "the second kernel compiled anew"
    assert two(x, _scaled_by(hl.constexpr(3.0))).equal(x * 3.0)
    assert one(x, fn).equal(x * 2.0)
    assert two(x, fn).equal(x * 2.0)
    run_one, run_two = _run_of(one, x, fn), _run_of(two, x, fn)
    assert run_one is not run_two
    assert run_one.native_calls == 1
    assert run_two.native_calls == 1


@pytest.mark.isolated(timeout=60)
def test_fork_after_a_planned_call() -> None:
    """On one thread: with a worker pool, the child hangs inside the compiled
    kernel, natively or not, with or without the fast key (seen at cd134f0)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    kernel = _kernel()
    x = torch.arange(32.0)
    fn = _scaled_by(hl.constexpr(2.0))
    kernel(x, fn)
    kernel(x, fn)
    run = _run_of(kernel, x, fn)
    assert run.native_calls == 1
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            os.close(read)
            good = kernel(x, _scaled_by(hl.constexpr(2.0))).equal(x * 2.0)
            os.write(write, bytes([good, run.native_calls]))
            status = 0
        finally:
            os._exit(status)
    os.close(write)
    data = os.read(read, 2)
    os.close(read)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert data == bytes([1, 2]), "the child's call was wrong or not native"
    assert run.native_calls == 1  # the child's count stays in the child
    assert kernel(x, fn).equal(x * 2.0)


def test_cache_stays_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A float argument, and a guarded int, change the key per value."""
    monkeypatch.setattr(inject, "_MAX_FAST_KEYS", 4)
    kernel = _kernel()
    args = (torch.ones(8), _scaled_by(2.0))
    kernel._dispatch_cache.update(dict.fromkeys(range(3)))
    kernel._fast_dispatch_key(args)
    assert len(kernel._dispatch_cache) == 3
    kernel._dispatch_cache[3] = None
    assert kernel._fast_dispatch_key(args) is not None
    assert not kernel._dispatch_cache


def test_two_threads_with_a_cache_of_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inject, "_MAX_FAST_KEYS", 1)
    kernel = _kernel()
    xs = (torch.arange(32.0), torch.arange(16.0))
    errors: list[object] = []
    for x in xs:  # both compiled before the threads start
        kernel(x, _scaled_by(hl.constexpr(2.0)))

    def work(x: torch.Tensor, value: float) -> None:
        try:
            for _ in range(300):
                out = kernel(x, _scaled_by(hl.constexpr(2.0)))
                if not out.equal(x * value):
                    errors.append((x.shape, out))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(x, 2.0)) for x in xs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors[:1]


def test_constexpr_in_the_annotated_position_and_wrapped_share_nothing_wrong() -> None:
    kernel = _kernel(_annotated)
    x = torch.arange(1.0, 33.0)
    assert kernel(x, 2.0).equal(x / 2.0)
    assert kernel(x, hl.constexpr(2.0)).equal(x / 2.0)
    assert kernel(x, hl.constexpr(4.0)).equal(x / 4.0)
    assert kernel(x, 4.0).equal(x / 4.0)
    assert ConstExpr is type(hl.constexpr(1))


def test_defaults_on_a_short_call_take_the_same_compile() -> None:
    def scale(x: torch.Tensor, fn: object, v: hl.constexpr = 2.0) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.shape[0]):
            out[tile] = fn(x[tile]) * v
        return out

    kernel = _kernel(scale)
    x = torch.arange(32.0)
    fn = _scaled_by(hl.constexpr(3.0))
    assert kernel(x, fn).equal(x * 6.0)
    assert kernel(x, fn).equal(x * 6.0)
    assert kernel(x, fn, 2.0).equal(x * 6.0)
    assert kernel(x, fn, hl.constexpr(5.0)).equal(x * 15.0)
    assert kernel(x, fn).equal(x * 6.0)


def test_a_specialized_tensor_argument_next_to_a_function() -> None:
    def spec(x: torch.Tensor, fn: object) -> torch.Tensor:
        n = hl.specialize(x.shape[0])
        out = torch.empty_like(x)
        for tile in hl.tile(n):
            out[tile] = fn(x[tile]) * n
        return out

    kernel = helion.kernel(spec, backend="mlir", config=_cfg(8))
    fn = _scaled_by(hl.constexpr(2.0))
    for n in (32, 40, 32, 40):
        x = torch.ones(n)
        assert _first(kernel(x, fn)) == 2.0 * n
