"""``gpu.printf`` on the CPU (:mod:`..mlir.lowering.print_ops` emits it). Nothing here
changes a module without a print.

- **CPU:** a libc ``printf`` and ``fflush`` per line. libc locks the stream per call, so
  lines stay whole under OpenMP, in no fixed order; the flush keeps a line ahead of a
  later crash and of Python's own output.
"""

from __future__ import annotations

import warnings

from mlir import ir
from mlir.dialects import arith as arith_d
from mlir.dialects import llvm as llvm_d


def warn_on_each_compile(message: str) -> None:
    """A ``UserWarning`` for this compile. ``warnings.warn`` with Python's default filter shows
    one text once per location, so a later compile from here would stay silent; a registry
    of its own for each call leaves that to the user's filters."""
    warnings.warn_explicit(message, UserWarning, __file__, 0)


def warn_print_is_experimental(entry: str) -> None:
    """Once per compile of a kernel with a print, whatever its number of prints."""
    warn_on_each_compile(
        f"print in kernel {entry} (hl.device_print) is experimental: see "
        "'Printing in a kernel' in the helion-mlir-backend README."
    )


def _ops_named(root: ir.Operation, names: frozenset[str]) -> list[ir.Operation]:
    found: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name in names:
            found.append(op)
        return ir.WalkResult.ADVANCE

    root.walk(visit)
    return found


def _prints(module: ir.Module) -> list[ir.Operation]:
    return _ops_named(module.operation, frozenset({"gpu.printf"}))


def _c_string(text: str) -> str:
    """``text`` as an MLIR string attribute's body, NUL-terminated."""
    return "".join(f"\\{byte:02X}" for byte in text.encode()) + "\\00"


def lower_prints_to_libc(module: ir.Module) -> int:
    """Each ``gpu.printf`` as ``printf`` and ``fflush``; the number rewritten. Variadic
    floats are doubles."""
    prints = _prints(module)
    if not prints:
        return 0
    ptr = ir.Type.parse("!llvm.ptr")
    i32 = ir.IntegerType.get_signless(32)
    printf = ir.FlatSymbolRefAttr.get("printf")
    fflush = ir.FlatSymbolRefAttr.get("fflush")
    printf_type = ir.TypeAttr.get(ir.Type.parse("!llvm.func<i32 (ptr, ...)>"))
    start = ir.InsertionPoint.at_block_begin(module.body)
    start.insert(ir.Operation.parse("llvm.func @printf(!llvm.ptr, ...) -> i32"))
    start.insert(ir.Operation.parse("llvm.func @fflush(!llvm.ptr) -> i32"))
    strings: dict[str, str] = {}
    for op in prints:
        text = op.attributes["format"].value
        if text not in strings:
            strings[text] = f"__helion_print_{len(strings)}"
            start.insert(
                ir.Operation.parse(
                    f"llvm.mlir.global private constant @{strings[text]}"
                    f'("{_c_string(text)}") {{addr_space = 0 : i32}}'
                )
            )
        with ir.InsertionPoint(op):
            symbol = ir.FlatSymbolRefAttr.get(strings[text])
            operands = [llvm_d.AddressOfOp(ptr, symbol).result]
            for value in op.operands:
                if isinstance(value.type, ir.FloatType) and value.type.width < 64:
                    value = arith_d.ExtFOp(ir.F64Type.get(), value).result
                operands.append(value)
            llvm_d.CallOp(
                i32, operands, [], [], var_callee_type=printf_type, callee=printf
            )
            llvm_d.CallOp(i32, [llvm_d.ZeroOp(ptr).result], [], [], callee=fflush)
        op.erase()
    return len(prints)


_MEMREF_VIEWS = frozenset(
    {
        "memref.cast",
        "memref.collapse_shape",
        "memref.expand_shape",
        "memref.reinterpret_cast",
        "memref.subview",
    }
)
_MEMREF_READS = frozenset({"memref.load", "vector.load", "vector.transfer_read"})
_MEMREF_NEUTRAL = frozenset({"memref.dim", "gpu.launch_func", "gpu.return"})


_PRINT_LOOP_OPS = frozenset(
    {
        "arith.addi",
        "arith.cmpi",
        "arith.constant",
        "arith.extf",
        "arith.extsi",
        "arith.extui",
        "arith.index_cast",
        "gpu.thread_id",
        "memref.load",
        "scf.for",
        "scf.if",
        "scf.yield",
        "vector.load",
    }
)
