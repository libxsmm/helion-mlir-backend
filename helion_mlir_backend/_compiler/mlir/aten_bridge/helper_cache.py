"""Helper requests, their lowering by torch-mlir, and the process-wide helper cache.

A :class:`HelperRequest` is an ATen call typed by its operands' MLIR types. The
cache lowers the requests it lacks in one torch-mlir run (each alone if that
run fails, to name the failing node) and keeps the lowered functions, which live
in the shared MLIR context, for every later module.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from dataclasses import field
import hashlib
import operator
from typing import TYPE_CHECKING

import mlir.ir as ir
import torch
import torch.fx
from torch.fx.passes.shape_prop import TensorMetadata

from ..support import UnsupportedOperationError
from .nan_minmax import NAN_PROPAGATING_OPS
from .nan_minmax import propagate_nan
from .samples import substitute
from .scalar_compares import COMPARE_OPS
from .scalar_compares import compare_in_tile_type

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch._ops import OpOverload


@dataclass(frozen=True)
class HelperRequest:
    target: OpOverload
    args: tuple
    kwargs: tuple[tuple[str, object], ...]
    operand_types: tuple[str, ...]
    result_types: tuple[str, ...]
    samples: tuple = field(compare=False, hash=False, repr=False)
    """Meta or fake tensor, or Python scalar, standing in for each operand."""
    result: object = field(compare=False, hash=False, repr=False)
    """The call's result (a tensor or a tuple of them) on ``samples``."""

    @property
    def name(self) -> str:
        key = repr((str(self.target), self.args, self.kwargs, self.operand_types))
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return f"_aten_{self.target.__name__.replace('.', '_')}_{digest}"

    @property
    def function_type(self) -> ir.FunctionType:
        return ir.FunctionType.get(
            [ir.Type.parse(text) for text in self.operand_types],
            [ir.Type.parse(text) for text in self.result_types],
        )


class HelperCache:
    """Lowered helper functions by name, and the modules that own them."""

    def __init__(self) -> None:
        self._functions: dict[str, ir.Operation] = {}
        self._modules: list[ir.Module] = []

    def __contains__(self, name: str) -> bool:
        return name in self._functions

    def __getitem__(self, name: str) -> ir.Operation:
        return self._functions[name]

    def lower(self, requests: list[tuple[HelperRequest, torch.fx.Node]]) -> None:
        """Lower ``requests`` (with the node each came from) into the cache."""
        for request, node in requests:
            _refuse_what_crashes_torch_mlir(request, node)
        batch = [request for request, _ in requests]
        try:
            helpers = _run_torch_mlir(batch)
        except Exception:
            pass
        else:
            self._add(helpers, batch)
            return
        for request, node in requests:
            try:
                helpers = _run_torch_mlir([request])
            except Exception as error:
                message = str(error).strip().splitlines()
                with located(node):
                    raise UnsupportedOperationError(
                        str(request.target),
                        reason=(
                            f"torch-mlir cannot lower it for operand types "
                            f"{list(request.operand_types)}: "
                            f"{message[0] if message else type(error).__name__}"
                        ),
                    ) from error
            self._add(helpers, [request])

    def _add(self, helpers: ir.Module, requests: list[HelperRequest]) -> None:
        self._modules.append(helpers)
        for operation in helpers.body.operations:
            if "sym_name" in operation.attributes:
                name = ir.StringAttr(operation.attributes["sym_name"]).value
                self._functions[name] = operation
        # torch-mlir's minima and maxima drop NaN: see nan_minmax.
        for request in requests:
            if request.target.overloadpacket in NAN_PROPAGATING_OPS:
                propagate_nan(self._functions[request.name])
            # torch-mlir's compares with a number aren't in the tile's type: see there.
            if request.target in COMPARE_OPS:
                compare_in_tile_type(
                    self._functions[request.name],
                    request.args[1] if len(request.args) > 1 else None,
                    request.target.__name__.split(".")[-2],
                )


_BYTE_TILE_CRASHES = frozenset({"mul", "remainder"})
"""torch-mlir's ops that crash on an i8 tile and a Python integer, with ``div.Tensor_mode`` and
``sub`` with the integer second. Measured, wheel 20261001: its ``convertScalarToDtype`` reports
"for conversion to byte or char type dstOriginalDtype has to be passed", and then the process
dies with SIGSEGV in ``lower_mlir_module``: nothing here could catch it. ``t + 1``, ``1 - t``,
``t & 1``, ``torch.clamp``, ``t / 2``, ``t << 1`` and others lower, and are right."""


def _refuse_what_crashes_torch_mlir(
    request: HelperRequest, node: torch.fx.Node
) -> None:
    samples = request.samples
    byte_tile = any(
        isinstance(sample, torch.Tensor) and sample.dtype == torch.int8
        for sample in samples
    )
    an_int = any(
        isinstance(arg, int) and not isinstance(arg, bool) for arg in request.args
    )
    name = request.target.__name__
    op = name.split(".")[0]
    second_is_int = len(request.args) > 1 and isinstance(request.args[1], int)
    crashes = (
        op in _BYTE_TILE_CRASHES
        or name == "div.Tensor_mode"
        or (op == "sub" and second_is_int)
    )
    if byte_tile and an_int and crashes:
        with located(node):
            raise UnsupportedOperationError(
                str(request.target),
                reason="torch-mlir crashes on an i8 tensor with a Python integer in this op",
                alternatives=[
                    "a tensor of the integer, of the same type",
                    "another type",
                ],
            )


CACHE = HelperCache()


@contextlib.contextmanager
def located(node: torch.fx.Node) -> Iterator[None]:
    """Attribute errors raised inside to ``node``'s kernel source line."""
    location = node.meta.get("location")
    with location if location is not None else contextlib.nullcontext():
        yield


def _run_torch_mlir(requests: list[HelperRequest]) -> ir.Module:
    """Import and lower ``requests`` with torch-mlir; the result lives in this context."""
    from torch_mlir.compiler_utils import OutputType
    from torch_mlir.compiler_utils import lower_mlir_module
    from torch_mlir.compiler_utils import run_pipeline_with_repro_report
    from torch_mlir.dialects import torch as torch_d
    from torch_mlir.extras.fx_importer import FxImporter
    import torch_mlir.ir as tm_ir

    context = tm_ir.Context()
    torch_d.register_dialect(context)
    importer = FxImporter(context=context)
    for request in requests:
        importer.import_stateless_graph(_fx_graph(request), func_name=request.name)
    run_pipeline_with_repro_report(
        importer.module,
        "builtin.module(func.func(torch-match-quantized-custom-ops),"
        " torchdynamo-export-to-torch-backend-pipeline{})",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=False,
    )
    lower_mlir_module(False, OutputType.LINALG_ON_TENSORS, importer.module)
    return ir.Module.parse(
        importer.module.operation.get_asm(binary=False, enable_debug_info=False)
    )


def _fx_graph(request: HelperRequest) -> torch.fx.Graph:
    graph = torch.fx.Graph()
    placeholders = []
    for index, sample in enumerate(request.samples):
        placeholder = graph.placeholder(f"arg{index}")
        _set_meta(placeholder, sample)
        placeholders.append(placeholder)
    args = substitute(request.args, placeholders)
    kwargs = substitute(dict(request.kwargs), placeholders)
    call = graph.call_function(request.target, args, kwargs)
    if isinstance(request.result, torch.Tensor):
        _set_meta(call, request.result)
        graph.output((call,))
        return graph
    call.meta["val"] = tuple(request.result)
    outputs = []
    for index, result in enumerate(request.result):
        item = graph.call_function(operator.getitem, (call, index))
        _set_meta(item, result)
        outputs.append(item)
    graph.output(tuple(outputs))
    return graph


def _set_meta(node: torch.fx.Node, value: object) -> None:
    node.meta["val"] = value
    if isinstance(value, torch.Tensor):
        node.meta["tensor_meta"] = TensorMetadata(
            value.shape, value.dtype, False, value.stride(), None, False, {}
        )
