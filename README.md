# helion-mlir-backend

External MLIR backend package for Helion.

Currently, only CPU execution is supported.

## Usage

When `helion_mlir_backend` is imported, it registers the `mlir` backend in Helion.

Usable by standard Helion kernels with `mlir` backend selected:

```python
import helion
import helion_mlir_backend  # noqa: F401

@helion.kernel(backend="mlir")
def kernel():
  ...
```

Following Helion configurations (`config=helion.Config()`) are supported:
* **block\_sizes** (`list[int]`):
Controls tile sizes corresponding to each dimension passed `hl.tile` or call
to `hl.register_block_size` in the kernel.

Additional MLIR backend kernel (`helion.kernel(...)`) options:
* **mlir\_pipeline** (`"opt"`, `"scalar"`):
Selects compilation pipeline:
  - `opt`: Optimizing pipeline (default). Vectorization, parallelization etc.
  - `scalar`: Scalar lowering. A simple serial fallback; mostly useful for debugging.

Keys the MLIR backend doesn't read are ignored, since a kernel may carry keys for another
backend. A key that looks like a typo of a valid one (`mlir_pipline`) warns once per kernel.

## Install

The project is using [uv](https://docs.astral.sh/uv/) package manager.

`uv` can be [installed](https://docs.astral.sh/uv/getting-started/installation/) locally using:

```bash
pip install uv
```

The project can be installed using:

```bash
uv sync
```

## Requirements

The project extended Helion which targets Linux systems and requires a recent Python, PyTorch, and MLIR environment:
- Linux-based OS
- Python 3.11 or later
- [PyTorch](https://github.com/pytorch/pytorch) 2.12 or later
- [Helion](https://github.com/pytorch/helion) 1.4 or later
- [llvm/lighthouse](https://github.com/llvm/lighthouse)

For development, this repo is developed and tested against dependencies pulled by the `dev` dependency-group.
External users must supply compatible versions themselves.

## Testing

Run tests with pytest:

```bash
pytest -v
```

A whole-suite run is parallel (`pytest-xdist`, `-n auto`: one worker per physical core).

## Linting

This repo uses `pre-commit` with linter rules aligned with Helion style.

Initial config:

```bash
pip install pre-commit
pre-commit install
```

All checks can be run using:

```bash
pre-commit run -a
```

## Config variables

| Variable | Description |
|---|---|
| `HELION_MLIR_COMPILE_TIMEOUT` | Bounds MLIR lowering of each kernel, in seconds (default `30`; `0` disables the limit) |
| `HELION_MLIR_NATIVE_CALL=0` | Starts the process with the native CPU call paths off (`_compiler/native/`: C++ argument checks and calls, the native kernel run and prepared calls); every call then takes the Python paths. Read once, at import; `helion_mlir_backend.set_native_call(False/True)` switches later (off: at once for every call; on: for kernels compiled from then on). Default: on, if the extension builds. A kernel whose first call can't be planned (e.g. two arguments the same tensor) never runs natively in that process (speed only) |
| `HELION_MLIR_OUTPUT_POOL=<bytes>` | Most bytes of free blocks the CPU output pool keeps (`_compiler/output_pool.py`: tensors of 4 MiB or more that kernels allocate, reused once their storage dies; a call then pays no page faults, whatever the allocator). `0`: off. Read once, at import; a bad value warns and keeps the default; `helion_mlir_backend.set_output_pool(bytes)` changes it later. Default: the smaller of 1 GiB and 1/8 of the memory (RAM or cgroup limit) |
| `HELION_MLIR_NATIVE_CACHE` | Where the native extension is built and cached (default `~/.cache/helion_mlir_backend/native`) |
| `HELION_MLIR_DUMP_IR` | Raw generated IR before inlining (shows `@_aten_*` helpers and tile loops) |
| `HELION_MLIR_DUMP_PRE_LOWERING` | After inlining / all pre-passes, immediately before the lighthouse lowering schedule |
| `HELION_MLIR_DUMP_LOWERED` | After lighthouse lowering (LLVM dialect, expanded memref descriptors) |
| `HELION_MLIR_PIPELINE=opt\|scalar` | Selects compilation pipeline (default: `opt`) |
| `HELION_MLIR_TARGET=auto\|cpu` | Selects the device target (default: `auto`, from the kernel's tensors; an explicit `target=` of `generate_mlir`/`compile_mlir`/`lower_mlir` wins). `meta` tensors need a named target |

## License

MIT License - see [LICENSE](LICENSE) for details.
