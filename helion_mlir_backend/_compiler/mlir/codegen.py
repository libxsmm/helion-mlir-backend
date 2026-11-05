"""Core MLIR module builder for Helion kernels.

Walks the :class:`~helion._compiler.device_ir.DeviceIR` produced by the
standard Helion compilation pipeline and emits an ``mlir.ir.Module`` using
Linalg-on-Tensors abstraction.

Mapping summary
---------------
Helion construct          → MLIR construct
──────────────────────────────────────────────────────────────────────────────
Outer ``hl.tile([m,n])``  → normalized ``scf.forall`` + ``shared_outs``
Inner ``hl.tile(k)``      → ``scf.for``   (sequential, carries accumulator)
``hl.zeros([bm,bn])``     → ``tensor.empty`` + ``linalg.fill``
``hl.load(t, idx)``       → ``tensor.extract_slice`` of ``t``'s current value
``hl.store(t, idx, v)``   → ``tensor.insert_slice`` into ``t``'s current value
matmul family, ``hl.dot``,
einsum, ``acc + mm``      → one ``linalg.matmul``/``batch_matmul``/``contract``
Pointwise aten ops        → ``linalg.generic`` or ``arith.*``
``_host_tensor(name)``    → the phase function's argument for that host tensor
``_get_symnode(bs_N)``    → the concrete block-size integer constant
``_get_symnode(scalar)``  → the runtime scalar argument (by value)
``_phi``                  → the result of the enclosing ``scf.for``
``_new_var``              → pass-through (same MLIR value)

ATen lowering architecture
--------------------------
Every node is dispatched by target identity through ``lowering/registry.py``.
Helion-specific device-IR nodes determine the tile/control-flow structure and
lower directly to ``scf``/``tensor``/``linalg`` operations. Any other ATen node
becomes a ``func.call`` to a private helper typed by its operands' MLIR types at
the call site (``aten_bridge/helpers.py``); after all functions are built, the
helpers are lowered through torch-mlir's FX importer and Torch-to-Linalg
pipeline in one batch and cloned into this module. Helion's node metadata is
read, never modified. A small set of direct ATen lowerings remains for
contractions, views, casts and index-scalar arithmetic.

This split is necessary because ``FxImporter`` cannot import Helion's
non-ATen FX targets, while torch-mlir provides broad, reusable coverage for
the ordinary ATen portions of each tile body.
"""

from __future__ import annotations

import functools
from itertools import starmap
import logging
from typing import TYPE_CHECKING

from mlir.dialects import bufferization as bufferization_d
from mlir.dialects import func as func_d
import mlir.ir as ir

from .analysis.signature import KernelSignature
from .aten_bridge import AtenHelperTable
from .build_context import BuildContext
from .lowering import build_phase_body
from .lowering import lower_node
from .support import MLIRBackendError
from .support import torch_dtype_to_mlir

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.host_function import HostFunction

    from .analysis.signature import PhaseSignature
    from .analysis.signature import ScalarArg
    from .analysis.signature import TensorRef

log = logging.getLogger(__name__)


@functools.cache
def _get_shared_mlir_context() -> ir.Context:
    """The process-wide ``mlir.ir.Context``, created on first use.

    Creating and destroying contexts per compile races their background thread
    pools and has segfaulted; one context hosts every module instead. It runs no
    threads (a module has one function once inlined), so a forked lowering can
    use it.
    """
    context = ir.Context()
    context.enable_multithreading(False)
    return context


class MLIRModuleBuilder:
    """Builds an ``mlir.ir.Module`` from a compiled :class:`HostFunction`.

    The module holds one private tensor function per ``hl.barrier()`` phase and
    a public memref-ABI entry function named after the kernel that calls them in
    order.

    Parameters
    ----------
    host_function:
        Compiled HostFunction (has ``device_ir`` populated).
    config:
        Helion Config with concrete block sizes.
    env:
        Active CompileEnvironment.
    """

    def __init__(
        self,
        host_function: HostFunction,
        config: object,
        env: CompileEnvironment,
    ) -> None:
        self.hf = host_function
        self.config = config
        self.env = env
        self.context = BuildContext(host_function, config, env)
        self.context.lower_node_callback = functools.partial(lower_node, self.context)

    def build(self) -> ir.Module:
        """Build and return the generated MLIR module."""

        try:
            ctx = _get_shared_mlir_context()

            with ir.Location.unknown(ctx):
                module = ir.Module.create()
                self.context.mlir_module = module
                self.context.mlir_context = ctx
                self.context.aten_helpers = AtenHelperTable()
                with ir.InsertionPoint(module.body), self.hf:
                    self._resolve_geometry()
                    phases = list(
                        starmap(
                            self._build_phase_function,
                            enumerate(self.context.signature.phases),
                        )
                    )
                    self._build_entry_function(phases)
                    self.context.aten_helpers.materialize(module)
            return module
        except MLIRBackendError:
            raise
        except Exception as exc:
            exc.add_note(f"While generating MLIR for Helion kernel '{self.hf.name}'")
            raise

    def _build_phase_function(self, index: int, phase: PhaseSignature) -> func_d.FuncOp:
        """``(ins..., inouts..., scalars...) -> inouts...`` over tensors."""
        ctx = self.context
        refs = [*phase.ins, *phase.inouts]
        arg_types = [self._tensor_type(name) for name in refs]
        arg_types += [_scalar_type(ctx.signature.scalars[k]) for k in phase.scalars]
        result_types = [self._tensor_type(name) for name in phase.inouts]
        fn = func_d.FuncOp(
            f"{self.hf.name}__phase{index}",
            ir.FunctionType.get(arg_types, result_types),
        )
        fn.attributes["sym_visibility"] = ir.StringAttr.get("private")
        entry = fn.add_entry_block()
        with ir.InsertionPoint(entry):
            ctx.reset_for_new_function()
            args = list(entry.arguments)
            ctx.param_to_value.update(zip(refs, args[: len(refs)], strict=True))
            ctx.scalars.update(zip(phase.scalars, args[len(refs) :], strict=True))
            ctx.sizes.begin_function(entry)
            results = build_phase_body(ctx, list(phase.root_positions), phase.inouts)
            func_d.ReturnOp(results)
        return fn

    def _build_entry_function(self, phases: list[func_d.FuncOp]) -> None:
        """Public ``@<kernel>(inouts..., ins..., scalars...)``: memrefs, then
        runtime scalars by value (``f64``, ``i64``, ...).

        Inputs are ``restrict`` tensors, inouts ``restrict writable`` ones; each
        final inout value is committed with ``materialize_in_destination``.
        """
        signature = self.context.signature
        refs = [*signature.inouts, *signature.ins]
        scalars = list(signature.scalars.values())
        tensor_types = [self._tensor_type(ref.name) for ref in refs]
        arg_types = [
            ir.MemRefType.get(list(t.shape), t.element_type) for t in tensor_types
        ]
        arg_types += [_scalar_type(scalar) for scalar in scalars]
        fn = func_d.FuncOp(self.hf.name, ir.FunctionType.get(arg_types, []))
        fn.attributes["sym_visibility"] = ir.StringAttr.get("public")
        fn.attributes["arg_attrs"] = ir.ArrayAttr.get(
            [_arg_attr(ref.name, _role(ref), ref.tensor_param) for ref in refs]
            + [_arg_attr(scalar.host_expr, "scalar", None) for scalar in scalars]
        )
        entry = fn.add_entry_block()
        with ir.InsertionPoint(entry):
            args = list(entry.arguments)
            values: dict[str, ir.Value] = {}
            for ref, tensor_type, arg in zip(refs, tensor_types, args, strict=False):
                values[ref.name] = bufferization_d.ToTensorOp(
                    tensor_type, arg, restrict=True, writable=ref.written or None
                ).result
            # Apart from the tensors: a scalar's key may equal a tensor's name.
            scalar_values = {
                scalar.key: arg
                for scalar, arg in zip(scalars, args[len(refs) :], strict=True)
            }
            for phase, fn_op in zip(signature.phases, phases, strict=True):
                operands = [values[name] for name in (*phase.ins, *phase.inouts)]
                operands += [scalar_values[key] for key in phase.scalars]
                call = func_d.CallOp(fn_op, operands)
                values.update(zip(phase.inouts, call.results, strict=True))
            for ref, arg in zip(refs, args, strict=False):
                if ref.written:
                    bufferization_d.MaterializeInDestinationOp(
                        None, values[ref.name], arg, restrict=True, writable=True
                    )
            func_d.ReturnOp([])

    def _tensor_type(self, name: str) -> ir.RankedTensorType:
        """The host tensor's type: ``?`` for a size only known at run time. A size
        computed from block sizes on the host (``n // block_n``) takes the config's
        block sizes, as the host code does."""
        fake = self.context.signature.refs[name].fake
        shape = [
            ir.ShapedType.get_dynamic_size() if size.free_symbols else int(size)
            for size in self.context.sizes.ref(name)
        ]
        return ir.RankedTensorType.get(shape, torch_dtype_to_mlir(fake.dtype))

    def _resolve_geometry(self) -> None:
        """Record loop geometry, tensor effects, signature and contractions.

        Needs ``with self.hf:``.
        """
        from .analysis.contractions import ContractionPlan
        from .analysis.geometry import KernelGeometry
        from .analysis.tensor_effects import TensorEffects

        self.context.effects = TensorEffects.from_host_function(self.hf)
        self.context.signature = KernelSignature.from_host_function(
            self.hf, self.context.effects
        )
        self.context.geometry = KernelGeometry.from_host_function(
            self.hf, self.config, self.env
        )
        self.context.contractions = ContractionPlan.from_graphs(
            [graph_info.graph for graph_info in self.hf.device_ir.graphs]
        )


def _scalar_type(scalar: ScalarArg) -> ir.Type:
    return torch_dtype_to_mlir(scalar.dtype)


def _role(ref: TensorRef) -> str:
    return "inout" if ref.written else "in"


def _arg_attr(name: str, role: str, tensor_param: int | None) -> ir.DictAttr:
    """Entry argument metadata: host expression, role, tensor-parameter position."""
    attrs = {
        "helion.name": ir.StringAttr.get(name),
        "helion.role": ir.StringAttr.get(role),
    }
    if tensor_param is not None:
        attrs["helion.param"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), tensor_param
        )
    return ir.DictAttr.get(attrs)
