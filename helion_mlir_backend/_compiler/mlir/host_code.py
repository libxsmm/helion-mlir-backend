"""The kernel's host code as a re-executable generator function.

Helion's own codegen keeps the kernel's host statements as real Python that
runs on every call around the device kernel launch (see
``helion._compiler.generate_ast``). The MLIR backend does the same: the host
function is a copy of the kernel's AST body in which the top-level device loops
(``ast.For`` tagged ``LoopType.GRID``; Helion requires them at top level, with
only ``hl.barrier()`` between them) become one ``yield locals()``, and every
host-side block size (``hl.register_block_size``), ``hl.specialize`` and
``hl.register_tunable`` becomes its compile-time value, as in Helion's host
codegen. The driver runs it to the ``yield``, calls the
compiled kernel with tensors taken from the yielded locals, then resumes it; the
generator's return value is the kernel's return value.
"""

from __future__ import annotations

import ast
import copy
import types
from typing import TYPE_CHECKING
from typing import Callable

from .. import output_pool
from .support import UnsupportedOperationError

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion._compiler.host_function import HostFunction
    from helion.runtime.config import Config


def build_host_function(
    hf: HostFunction, block_size: Callable[[int], int], config: Config | None = None
) -> Callable[..., Generator[dict[str, object], None, object]]:
    """Compile *hf*'s host code; ``hf.body`` is not modified.

    ``config`` gives the values of ``hl.register_tunable`` calls.
    """
    body: list[ast.stmt] = []
    yielded = False
    for stmt in hf.body:
        if _is_device_statement(stmt):
            if not yielded:
                body.append(ast.copy_location(_yield_locals(), stmt))
                yielded = True
            continue
        body.append(_HostCode(block_size, config).visit(stmt))
    if not yielded:
        body.append(_yield_locals())

    func_name = f"_helion_mlir_host_{hf.name}"
    func_def = ast.FunctionDef(
        name=func_name,
        args=hf.definition.args,
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None,
        lineno=1,
        col_offset=0,
    )
    # Host expressions of globals go through these aliases (``_source_module.W``).
    # They are the factory's parameters, so that the host function runs on the
    # module's own globals, as the kernel does in plain Python and in Helion's
    # codegen, and they stay out of the module.
    aliases = {
        imported.alias: imported.value for imported in hf.global_imports.values()
    }
    factory = ast.FunctionDef(
        name=f"{func_name}_factory",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=alias) for alias in aliases],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[func_def, ast.Return(ast.Name(id=func_name, ctx=ast.Load()))],
        decorator_list=[],
        returns=None,
        type_comment=None,
        lineno=1,
        col_offset=0,
    )
    module = ast.Module(body=[factory], type_ignores=[])
    ast.fix_missing_locations(module)

    code = compile(module, filename=f"<helion-mlir-host:{hf.name}>", mode="exec")
    factory_code = next(c for c in code.co_consts if isinstance(c, types.CodeType))
    function = types.FunctionType(factory_code, hf.fn.__globals__)(*aliases.values())
    function._helion_mlir_aliases = aliases  # type: ignore[attr-defined]
    metadata_only = _MetadataOnly(hf.definition.args, hf.fn.__globals__)
    function._helion_mlir_metadata_only = metadata_only.check(body)  # type: ignore[attr-defined]
    function._helion_mlir_host_refusal = metadata_only.refusal  # type: ignore[attr-defined]
    return function  # type: ignore[return-value]


def _is_device_statement(stmt: ast.stmt) -> bool:
    from helion._compiler.ast_extension import LoopType
    from helion._compiler.type_info import BarrierResultType

    if isinstance(stmt, ast.For):
        return getattr(stmt, "_loop_type", None) == LoopType.GRID
    return isinstance(stmt, ast.Expr) and isinstance(
        getattr(stmt.value, "_type_info", None), BarrierResultType
    )


def _yield_locals() -> ast.stmt:
    call = ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[])
    return ast.Expr(value=ast.Yield(value=call))


class _HostCode(ast.NodeTransformer):
    """Copy a host statement, replacing compile-time Helion values by constants.

    Every visited node is shallow-copied (with its list fields) before its
    children are transformed, so Helion's own AST stays intact.
    """

    def __init__(self, block_size: Callable[[int], int], config: Config | None) -> None:
        self.block_size = block_size
        self.config = config

    def generic_visit(self, node: ast.AST) -> ast.AST:
        from helion._compiler.ast_extension import ExtendedAST

        node = node.copy() if isinstance(node, ExtendedAST) else copy.copy(node)
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                setattr(node, field, list(value))
        return super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        from helion._compiler.type_info import BlockSizeType
        from helion._compiler.type_info import SequenceType
        from helion._compiler.type_info import TileIndexType

        block_types = (BlockSizeType, TileIndexType)
        type_info = getattr(node, "_type_info", None)
        if isinstance(type_info, block_types):
            return ast.copy_location(
                ast.Constant(self.block_size(type_info.block_id)), node
            )
        if isinstance(type_info, SequenceType) and all(
            isinstance(item, block_types) for item in type_info.unpack()
        ):
            sizes = [
                ast.Constant(self.block_size(item.block_id))
                for item in type_info.unpack()
            ]
            return ast.copy_location(ast.List(elts=sizes, ctx=ast.Load()), node)
        api = _api_function(node)
        if api is not None:
            return ast.copy_location(self._api_value(api, node), node)
        return self.generic_visit(node)

    def _api_value(self, api: object, node: ast.Call) -> ast.expr:
        from helion.language.constexpr import _convert_specializable
        from helion.language.constexpr import specialize
        from helion.language.tunable_ops import register_tunable

        if api is specialize:
            value = _convert_specializable(node.args[0]._type_info.proxy())
        elif api is register_tunable and self.config is not None:
            value = self.config[node.args[0]._type_info.proxy()]
        else:
            raise UnsupportedOperationError(
                f"hl.{api.__name__}", reason="not supported in host code"
            )
        return ast.parse(repr(value), mode="eval").body


def _api_function(node: ast.Call) -> object | None:
    from helion._compiler.type_info import CallableType
    from helion.language._decorators import is_api_func

    func_type = getattr(node.func, "_type_info", None)
    if isinstance(func_type, CallableType) and is_api_func(func_type.value):
        return func_type.value
    return None


class _MetadataOnly:
    """Whether host code is a pure function of its arguments' metadata, so a
    call's plan (the driver's native run) is valid for every call with the
    same metadata: it reads only the arguments' ``shape``, ``dtype``,
    ``device``, ``ndim``, ``size()``, ``stride()``, ``dim()``, ``numel()``
    and ``element_size()``, computes with them (arithmetic, comparisons,
    subscripts, ``if``, the builtins ``len``/``min``/``max``/``int``/``abs``)
    and allocates tensors only with ``torch.empty(shape, dtype=, device=)`` or
    ``torch.empty_like(x)`` or the same of :mod:`~.output_pool` (uninitialized;
    dtype and device given, not defaults), and may ``assert`` on all of that (a
    plan has the metadata of a call that passed); it returns a name or ``None``. Anything else (any other call
    or global, loops, ``del``, attribute or subscript assignment, ...) fails
    the check and keeps every call on the general path, which reads globals
    anew. ``torch``, ``output_pool`` and the builtins above are the only global
    reads a plan fixes, as checked at compile time."""

    _ATTRIBUTES = frozenset({"shape", "dtype", "device", "ndim"})
    _METHODS = frozenset({"size", "stride", "dim", "numel", "element_size"})
    _BUILTINS = frozenset({"len", "min", "max", "int", "abs"})

    def __init__(self, args: ast.arguments, namespace: dict[str, object]) -> None:
        self.namespace = namespace
        self.names = {
            arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
        }
        self.ok = not (args.vararg or args.kwarg)
        self.refusal: str | None = None
        """Why :meth:`check` failed: the first top-level statement it refused."""

    def check(self, body: list[ast.stmt]) -> bool:
        if not self.ok:
            self.refusal = "the kernel takes *args or **kwargs"
            return False
        try:
            for stmt in body:
                if not self.statement(stmt):
                    self.refusal = ast.unparse(stmt)
                    return False
        except RecursionError:
            self.refusal = "host code nested too deeply to check"
            return False
        return True

    def statement(self, stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Expr):
            value = stmt.value
            if isinstance(value, ast.Yield):  # the driver's ``yield locals()``
                return (
                    isinstance(value.value, ast.Call)
                    and isinstance(value.value.func, ast.Name)
                    and value.value.func.id == "locals"
                    and not value.value.args
                )
            return isinstance(value, ast.Constant)
        if isinstance(stmt, ast.Assign):
            return self.expr(stmt.value) and all(
                self.target(target) for target in stmt.targets
            )
        if isinstance(stmt, ast.AnnAssign):
            return (
                stmt.value is not None
                and self.expr(stmt.value)
                and self.target(stmt.target)
            )
        if isinstance(stmt, ast.If):
            if not self.truth(stmt.test):
                return False
            # A test Helion typed as a literal (e.g. ``out is None``) is fixed
            # for the bound kernel and the plan's argument metadata: only its
            # branch runs, and Helion leaves the other one untyped.
            from helion._compiler.type_info import LiteralType

            test_type = getattr(stmt.test, "_type_info", None)
            if not isinstance(test_type, LiteralType):
                return all(self.statement(s) for s in (*stmt.body, *stmt.orelse))
            taken, other = (
                (stmt.body, stmt.orelse)
                if test_type.value
                else (stmt.orelse, stmt.body)
            )
            # The other branch is untyped, but a call of the same bound kernel
            # (e.g. ``out`` given after binding with None) may run it: there,
            # only renames of known names and plain allocations.
            names = set(self.names)
            return all(self.statement(s) for s in taken) and all(
                self._untyped(s, names) for s in other
            )
        if isinstance(stmt, ast.Assert):
            return self.truth(stmt.test) and (
                stmt.msg is None or isinstance(stmt.msg, ast.Constant)
            )
        if isinstance(stmt, ast.Return):
            return stmt.value is None or isinstance(
                stmt.value, (ast.Name, ast.Constant)
            )
        return isinstance(stmt, ast.Pass)

    def _untyped(self, stmt: ast.stmt, names: set[str]) -> bool:
        """An untyped statement: ``pass``, or ``name = `` a name, a constant, a
        tuple or list of them, a metadata attribute of a name, or a
        ``torch.empty``/``torch.empty_like`` of those (no arithmetic, no
        other call)."""
        if isinstance(stmt, ast.Pass):
            return True
        if not (
            isinstance(stmt, ast.Assign)
            and all(isinstance(t, ast.Name) for t in stmt.targets)
            and self._plain(stmt.value, names, allocation=True)
        ):
            return False
        names.update(t.id for t in stmt.targets)  # type: ignore[attr-defined]
        return True

    def _plain(self, node: ast.expr, names: set[str], allocation: bool = False) -> bool:
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return node.id in names
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(self._plain(e, names) for e in node.elts)
        if isinstance(node, ast.Attribute):
            if self._is_torch(node.value):
                return isinstance(
                    getattr(torch_module(), node.attr, None), torch_dtype()
                )
            return (
                node.attr in self._ATTRIBUTES
                and isinstance(node.value, ast.Name)
                and (node.value.id in names)
            )
        if not (allocation and self._allocation(node)):
            return False
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        keywords = {k.arg for k in node.keywords}
        if node.func.attr == "empty":
            form = keywords == {"dtype", "device"} and len(node.args) >= 1
        else:
            form = keywords <= {"dtype"} and len(node.args) == 1
        return (
            form
            and not any(isinstance(a, ast.Starred) for a in node.args)
            and all(self._plain(a, names) for a in node.args)
            and all(self._plain(k.value, names) for k in node.keywords)
        )

    def target(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            self.names.add(node.id)
            return True
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(self.target(elt) for elt in node.elts)
        return False

    _COMPUTED = (
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        ast.Subscript,
        ast.Call,
    )

    def truth(self, node: ast.expr) -> bool:
        """``node`` as a condition. A tensor's truth is its data, which a plan
        doesn't record: replayed, the condition would keep its first value."""
        from helion._compiler.type_info import TensorType

        if isinstance(node, ast.Name) and isinstance(
            getattr(node, "_type_info", None), TensorType
        ):
            return False
        return self.expr(node)

    def expr(self, node: ast.expr) -> bool:
        from helion._compiler.type_info import TensorType

        # Helion's types tell int arithmetic from torch ops: a tensor value
        # may only be named or allocated, and a computed value of unknown
        # type fails the check.
        type_info = getattr(node, "_type_info", None)
        if isinstance(type_info, TensorType):
            if not (isinstance(node, ast.Name) or self._allocation(node)):
                return False
        elif isinstance(node, self._COMPUTED) and type_info is None:
            return False
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return node.id in self.names
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(self.expr(elt) for elt in node.elts)
        if isinstance(node, ast.Attribute):
            if self._is_torch(node.value):
                return isinstance(
                    getattr(torch_module(), node.attr, None), torch_dtype()
                )
            return node.attr in self._ATTRIBUTES and self.expr(node.value)
        if isinstance(node, ast.Subscript):
            return self.expr(node.value) and self.expr(node.slice)
        if isinstance(node, ast.Slice):
            return all(
                part is None or self.expr(part)
                for part in (node.lower, node.upper, node.step)
            )
        if isinstance(node, ast.BinOp):
            return self.expr(node.left) and self.expr(node.right)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return self.truth(node.operand)
            return self.expr(node.operand)
        if isinstance(node, ast.BoolOp):
            return all(self.truth(value) for value in node.values)
        if isinstance(node, ast.Compare):
            return self.expr(node.left) and all(self.expr(c) for c in node.comparators)
        if isinstance(node, ast.IfExp):
            return (
                self.truth(node.test)
                and self.expr(node.body)
                and self.expr(node.orelse)
            )
        if isinstance(node, ast.Call):
            return self.call(node)
        return False

    def call(self, node: ast.Call) -> bool:
        func = node.func
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            return False
        args_ok = all(self.expr(arg) for arg in node.args) and all(
            self.expr(keyword.value) for keyword in node.keywords
        )
        if isinstance(func, ast.Name):
            import builtins

            return (
                func.id in self._BUILTINS
                and func.id not in self.names
                and self.namespace.get(func.id, getattr(builtins, func.id))
                is getattr(builtins, func.id)
                and not node.keywords
                and args_ok
            )
        if not isinstance(func, ast.Attribute):
            return False
        if self._allocates(func.value):
            keywords = {keyword.arg for keyword in node.keywords}
            if func.attr == "empty":
                return (
                    keywords == {"dtype", "device"} and len(node.args) >= 1 and args_ok
                )
            if func.attr == "empty_like":
                return (
                    keywords <= {"dtype"}
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and args_ok
                )
            return False
        return func.attr in self._METHODS and self.expr(func.value) and args_ok

    def _allocation(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("empty", "empty_like")
            and self._allocates(node.func.value)
        )

    def _allocates(self, node: ast.expr) -> bool:
        return self._is_torch(node) or (
            isinstance(node, ast.Name)
            and node.id not in self.names
            and self.namespace.get(node.id) is output_pool
        )

    def _is_torch(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Name)
            and node.id == "torch"
            and node.id not in self.names
            and self.namespace.get("torch") is torch_module()
        )


def torch_module() -> object:
    import torch

    return torch


def torch_dtype() -> type:
    import torch

    return torch.dtype
