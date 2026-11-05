"""The Helion internals the native kernel run (``driver._native_run``,
``host_code._MetadataOnly``) relies on, and a matrix of host-code shapes run
natively and with ``HELION_MLIR_NATIVE_CALL=0``, bit for bit.

Marked ``helion_internals``: on a Helion upgrade, run ``pytest -m
helion_internals`` first. A failing guard names the assumption that broke; the
native run must then be revisited before the rest of the suite means much.
"""

from __future__ import annotations

import ast
from itertools import starmap

import helion
import helion.language as hl
from helion.language.constexpr import ConstExpr
import pytest
import torch

from helion_mlir_backend._compiler import native
from helion_mlir_backend._compiler.mlir.host_code import _HostCode
from helion_mlir_backend._compiler.mlir.host_code import _is_device_statement

pytestmark = [
    pytest.mark.helion_internals,
    pytest.mark.skipif(
        native.module() is None, reason="the native extension could not be built"
    ),
]


def _cfg(*block_sizes: int) -> helion.Config:
    return helion.Config(block_sizes=list(block_sizes))


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def add_into(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    m, n = x.shape
    if out is None:
        result = torch.empty([m, n], dtype=x.dtype, device=x.device)
    else:
        result = out
    for tm, tn in hl.tile([m, n]):
        result[tm, tn] = x[tm, tn] + y[tm, tn]
    return result


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def times(x: torch.Tensor, scale: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * scale
    return out


def _nodes(body: list[ast.stmt], kind: type) -> list[ast.AST]:
    return [n for stmt in body for n in ast.walk(stmt) if isinstance(n, kind)]


# -- guards ------------------------------------------------------------------


def test_host_ast_nodes_carry_helion_types() -> None:
    from helion._compiler.type_info import LiteralType
    from helion._compiler.type_info import TensorType

    x, y = torch.randn(8, 8), torch.randn(8, 8)
    bound = add_into.bind((x, y))
    body = bound.host_function.body
    calls = [
        n
        for n in _nodes(body, ast.Call)
        if isinstance(n.func, ast.Attribute) and n.func.attr == "empty"
    ]
    shapes = [n for n in _nodes(body, ast.Attribute) if n.attr == "shape"]
    tests = [s.test for s in body if isinstance(s, ast.If)]
    assumption = (
        "Helion no longer types host-code AST nodes (_type_info: TensorType for "
        "tensors, LiteralType for constant tests): host_code._MetadataOnly "
        "relies on these to tell int arithmetic from torch ops"
    )
    assert calls and all(isinstance(c._type_info, TensorType) for c in calls), (
        assumption
    )
    assert shapes and all(
        getattr(s, "_type_info", None) is not None
        and not isinstance(s._type_info, TensorType)
        for s in shapes
    ), assumption
    assert len(tests) == 1 and isinstance(tests[0]._type_info, LiteralType), assumption
    assert tests[0]._type_info.value is True, assumption
    # The backend's copy of the host code keeps the types.
    copied = [
        _HostCode(lambda block_id: 8, None).visit(stmt)
        for stmt in body
        if not _is_device_statement(stmt)
    ]
    copied_calls = [
        n
        for n in _nodes(copied, ast.Call)
        if isinstance(n.func, ast.Attribute) and n.func.attr == "empty"
    ]
    assert [c._type_info for c in copied_calls] == [c._type_info for c in calls], (
        "host_code._HostCode no longer keeps Helion's _type_info on copied nodes"
    )


def test_untaken_literal_branch_is_untyped() -> None:
    x, y = torch.randn(8, 8), torch.randn(8, 8)
    body = add_into.bind((x, y)).host_function.body
    (branch,) = [s for s in body if isinstance(s, ast.If)]
    untaken = [n for s in branch.orelse for n in ast.walk(s) if isinstance(n, ast.expr)]
    assert untaken
    assert all(getattr(n, "_type_info", None) is None for n in untaken), (
        "Helion now types the untaken branch of a literal if: revisit "
        "_MetadataOnly._untyped (it checks that branch syntactically)"
    )


def test_constexpr_is_a_one_value_named_tuple_bound_per_value() -> None:
    c = hl.constexpr(3.0)
    assert type(c) is ConstExpr and isinstance(c, tuple), (
        "hl.constexpr is no longer a ConstExpr NamedTuple: the native run "
        "compares it as a tuple of one value (call.cpp, kConstExpr)"
    )
    assert len(c) == 1 and c[0] == 3.0 and c.value == 3.0, "ConstExpr layout changed"
    x = torch.randn(8, 8)
    assert times.bind((x, hl.constexpr(3.0))) is times.bind((x, hl.constexpr(3.0))), (
        "equal constexpr values no longer bind the same BoundKernel"
    )
    assert times.bind((x, hl.constexpr(3.0))) is not times.bind(
        (x, hl.constexpr(4.0))
    ), (
        "constexpr values no longer specialize the BoundKernel: the plan's "
        "constexpr check would not match the compiled kernel"
    )


def test_run_is_called_positionally_as_given() -> None:
    x, y = torch.randn(16, 8), torch.randn(16, 8)
    add_into.reset()
    bound = add_into.bind((x, y))
    bound(x, y)
    seen: list[tuple[int, bool]] = []
    run = bound._run

    def spy(*args: object, **kwargs: object) -> object:
        seen.append((len(args), bool(kwargs)))
        return run(*args, **kwargs)

    bound._run = spy
    try:
        bound(x, y)
        add_into(x, y)  # Kernel.__call__ (dispatch cache)
        add_into(x, y=y)  # keywords: normalized, defaults filled in
    finally:
        bound._run = run
        add_into.reset()
    assert seen == [(2, False), (2, False), (3, False)], (
        "BoundKernel/Kernel calls no longer reach _run as (positional args as "
        f"given; keywords normalized with defaults): {seen}. The native run "
        "records one plan per BoundKernel from the first call's arity"
    )


def test_defaulted_argument_binds_the_same_bound_kernel() -> None:
    x, y = torch.randn(8, 8), torch.randn(8, 8)
    assert add_into.bind((x, y)) is add_into.bind((x, y, None)), (
        "f(x, y) and f(x, y, None) no longer bind the same BoundKernel"
    )


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def apply(x: torch.Tensor, epilogue) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = epilogue(x[tm, tn])
    return out


def test_functions_bind_by_code_and_closure() -> None:
    def make(scale: object):
        return lambda t: t * scale

    x = torch.randn(8, 8)
    a, b = make(hl.constexpr(2.0)), make(hl.constexpr(2.0))
    assert apply.bind((x, a)) is apply.bind((x, b)), (
        "Helion no longer binds functions of the same code and equal closures to "
        "one BoundKernel: the plan's function check (call.cpp, kFunction) is then "
        "looser than needed but still safe; revisit it"
    )
    assert apply.bind((x, a)) is not apply.bind((x, make(hl.constexpr(3.0)))), (
        "Helion no longer specializes on a closure's constexpr value"
    )


# -- the matrix: native vs HELION_MLIR_NATIVE_CALL=0 --------------------------


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def scale_default(x: torch.Tensor, scale: hl.constexpr = 2.0) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * scale
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def alpha_default(x: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * alpha
    return out


BIAS = torch.randn(16, 8)


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def bias_default(x: torch.Tensor, b: torch.Tensor = BIAS) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + b[tm, tn]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def two_ins(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0 + y[tm, tn]
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def empty_tuple(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        out[tm, tn] = x[tm, tn] - 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def empty_f32(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), x.size(1)], dtype=torch.float32, device=x.device)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn].to(torch.float32) * 2.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def empty_like_f32(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float32)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn].to(torch.float32) + 0.5
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(1, 8))
def like_strided(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(y.size()):
        out[tm, tn] = y[tm, tn] + 1.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def half_cols(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    half = max(int(n // 2), 1) if x.ndim == 2 else n
    out = torch.empty([m, half], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, half]):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8, 8))
def two_outs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        a[tm, tn] = x[tm, tn] + 1.0
        b[tm, tn] = x[tm, tn] * 3.0
    return a, b


@helion.kernel(backend="mlir", static_shapes=True, config=_cfg(8))
def write_into(a: torch.Tensor, b: torch.Tensor) -> None:
    for tile in hl.tile(a.size(0)):
        b[tile] = a[tile] + 1.0


def _r(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(*shape).to(dtype)


_X = (16, 8)
_SHARED: dict[str, torch.Tensor] = {}


def _same(name: str) -> torch.Tensor:
    """One tensor per name and run (for calls passing an object twice)."""
    if name not in _SHARED:
        _SHARED[name] = _r(*_X)
    return _SHARED[name]


# (id, kernel, calls: each () -> (args, kwargs), native expected)
CASES = [
    ("literal_if_out_omitted", add_into, [lambda: ((_r(*_X), _r(*_X)), {})] * 3, True),
    (
        "literal_if_none_then_out",
        add_into,
        [lambda: ((_r(*_X), _r(*_X), None), {})] * 2
        + [lambda: ((_r(*_X), _r(*_X), torch.empty(_X)), {})] * 2,
        True,
    ),
    ("default_constexpr", scale_default, [lambda: ((_r(*_X),), {})] * 3, True),
    ("default_runtime_scalar", alpha_default, [lambda: ((_r(*_X),), {})] * 3, False),
    ("default_tensor", bias_default, [lambda: ((_r(*_X),), {})] * 3, False),
    ("kwargs", two_ins, [lambda: ((_r(*_X),), {"y": _r(*_X)})] * 3, True),
    (
        "kwargs_and_positional",
        add_into,
        [lambda: ((_r(*_X), _r(*_X)), {}), lambda: ((_r(*_X),), {"y": _r(*_X)})] * 2,
        True,
    ),
    (
        "aliased_first_call",
        two_ins,
        [lambda: ((_same("a"), _same("a")), {})]
        + [lambda: ((_r(*_X), _r(*_X)), {})] * 2,
        False,
    ),
    (
        "aliased_later_call",
        two_ins,
        [lambda: ((_r(*_X), _r(*_X)), {}), lambda: ((_same("c"), _same("c")), {})] * 2,
        True,
    ),
    (
        "constexpr_new_objects",
        times,
        [lambda: ((_r(*_X), hl.constexpr(3.0)), {})] * 3,
        True,
    ),
    ("empty_tuple_shape", empty_tuple, [lambda: ((_r(*_X),), {})] * 3, True),
    (
        "empty_other_dtype",
        empty_f32,
        [lambda: ((_r(*_X, dtype=torch.bfloat16),), {})] * 3,
        True,
    ),
    (
        "empty_like_dtype",
        empty_like_f32,
        [lambda: ((_r(*_X, dtype=torch.bfloat16),), {})] * 3,
        True,
    ),
    (
        "empty_like_strided_template",
        like_strided,
        [lambda: ((torch.as_strided(_r(16), (1, 16), (7, 1)), _r(1, 16)), {})] * 3,
        False,
    ),
    ("shape_arithmetic", half_cols, [lambda: ((_r(16, 16),), {})] * 3, True),
    ("several_outputs", two_outs, [lambda: ((_r(*_X),), {})] * 3, False),
    (
        "out_parameter",
        add_into,
        [lambda: ((_r(*_X), _r(*_X), torch.empty(_X)), {})] * 3,
        True,
    ),
    (
        "inout_returns_none",
        write_into,
        [lambda: ((_r(16), torch.zeros(16)), {})] * 3,
        True,
    ),
]


def _snapshot(result: object, args: tuple[object, ...], kwargs: dict) -> list[object]:
    def copy(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return (value.clone(), value.stride(), value.dtype)
        if isinstance(value, tuple):
            return tuple(copy(v) for v in value)
        return value

    tensors = [a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)]
    return [copy(result), [copy(t) for t in tensors]]


def _same_bits(a: object, b: object) -> bool:
    if isinstance(a, tuple) and len(a) == 3 and isinstance(a[0], torch.Tensor):
        return torch.equal(a[0], b[0]) and a[1:] == b[1:]
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(starmap(_same_bits, zip(a, b, strict=True)))
    return a == b


def _run_calls(kernel: helion.Kernel, calls: list) -> tuple[list[object], int]:
    torch.manual_seed(0)
    _SHARED.clear()
    kernel.reset()
    results = []
    for make in calls:
        args, kwargs = make()
        result = kernel(*args, **kwargs)
        results.append(_snapshot(result, args, kwargs))
    runs = {id(b): b._run for b in kernel._bound_kernels.values()}
    native_calls = sum(getattr(r, "native_calls", 0) for r in runs.values())
    kernel.reset()
    return results, native_calls


@pytest.mark.parametrize(
    ("kernel", "calls", "native_expected"),
    [pytest.param(k, c, n, id=i) for i, k, c, n in CASES],
)
def test_native_matches_switch_off(
    monkeypatch: pytest.MonkeyPatch,
    kernel: helion.Kernel,
    calls: list,
    native_expected: bool,
) -> None:
    on, native_calls = _run_calls(kernel, calls)
    native.set_native_call(False)
    try:
        off, off_native_calls = _run_calls(kernel, calls)
    finally:
        native.set_native_call(True)
    assert off_native_calls == 0
    assert _same_bits(on, off), "native and switched-off results differ"
    if native_expected:
        assert native_calls > 0, "the plan was not replayed"
    else:
        assert native_calls == 0, "a plan that must be refused ran natively"


# A condition on a tensor is data, not metadata: frozen in a plan, it would keep
# its first value (found in review: ``assert g`` stopped raising on replay).
_TRUTH_SOURCE = """
import torch, helion, helion.language as hl
@helion.kernel(backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[8]))
def k(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    {body}
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1.0
    return out
"""


def _truth_kernel(tmp_path, body: str):
    import importlib.util

    path = tmp_path / "truth_kernel.py"
    path.write_text(_TRUTH_SOURCE.replace("{body}", body))
    spec = importlib.util.spec_from_file_location(f"truth_{id(path)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.k


@pytest.mark.parametrize(
    ("body", "passing", "failing"),
    [
        ("assert g", 1.0, 0.0),
        ("assert not g", 0.0, 1.0),
        ("assert g, 'g is zero'", 1.0, 0.0),
        ("assert x.dim() == 1 and g", 1.0, 0.0),
        ("assert x.dim() == 2 or g", 1.0, 0.0),
        ("assert (g if x.dim() == 1 else True)", 1.0, 0.0),
        ("n = 16 if g else 32\n    assert n == 16", 1.0, 0.0),
        ("n = 32\n    if g:\n        n = 16\n    assert n == 16", 1.0, 0.0),
        ("h = g\n    assert h", 1.0, 0.0),
        ("assert abs(g)", 1.0, 0.0),
        ("assert g > 0", 1.0, 0.0),
        ("assert g[0]", 1.0, 0.0),
        ("assert g.sum() > 0", 1.0, 0.0),
    ],
)
def test_condition_on_a_tensor_is_not_frozen(tmp_path, body, passing, failing) -> None:
    k = _truth_kernel(tmp_path, body)
    x = torch.zeros(16)
    good, bad = torch.tensor([passing]), torch.tensor([failing])
    bound = k.bind((x, good))
    expected = x + 1.0
    assert torch.equal(bound(x, good), expected)
    assert torch.equal(bound(x, good), expected)
    with pytest.raises(AssertionError):
        bound(x, bad)


@pytest.mark.parametrize("body", ["assert g and x.dim() == 1", "assert int(g)"])
def test_condition_helion_refuses_is_not_planned(tmp_path, body: str) -> None:
    k = _truth_kernel(tmp_path, body)
    x, g = torch.zeros(16), torch.tensor([1.0])
    with pytest.raises((helion.exc.TypeInferenceError, helion.exc.TorchOpTracingError)):
        k.bind((x, g))(x, g)


def test_assert_on_one_element_input_is_not_frozen(tmp_path) -> None:
    source = """
import torch, helion, helion.language as hl
@helion.kernel(backend="mlir", static_shapes=True, config=helion.Config(block_sizes=[1]))
def k(x: torch.Tensor) -> torch.Tensor:
    assert x
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1.0
    return out
"""
    import importlib.util

    path = tmp_path / "one_element.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"one_{id(path)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    one, zero = torch.tensor([1.0]), torch.tensor([0.0])
    bound = module.k.bind((one,))
    bound(one)
    bound(one)
    with pytest.raises(AssertionError):
        bound(zero)


@pytest.mark.parametrize(
    "body",
    ["assert x.dim() == 1", "assert x.size(0) == 16, 'msg'", "assert not x.dim() == 2"],
)
def test_assert_on_metadata_stays_native(tmp_path, body: str) -> None:
    k = _truth_kernel(tmp_path, body)
    x, g = torch.zeros(16), torch.tensor([1.0])
    bound = k.bind((x, g))
    bound(x, g)
    assert type(bound._run).__name__ == "KernelRun"
    assert bound._run.planned
