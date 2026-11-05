// Native call paths of compiled CPU entries (see native/__init__.py).
//
// EntryCall: the checks of CompiledEntry.call_in_place and the packed
// C-interface call, without Python objects per call.
// KernelRun: a compiled kernel's run(*args) for the argument metadata of the
// plan its first call recorded (driver._plan): checks the arguments,
// allocates the host code's fresh tensors, calls the entry.
// PreparedRun: a KernelRun with its other arguments bound
// (driver.prepared_call).
//
// A check that fails makes the caller take the unchanged general path: these
// objects never raise for an argument they cannot take.

#include <cstring>
#include <Python.h>

#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>

#include <cstdint>
#include <vector>

namespace {

using EntryFunc = void (*)(void**);
using Ints = std::vector<int64_t>;

// native.set_native_call(): off, every call takes the general path (checked
// per call: one load).
bool g_enabled = true;

// The dispatch keys of a plain dense CPU tensor (with or without autograd
// metadata, inference tensors included). Anything else -- conjugate or
// negative views, functionalization, functorch wrappers, Python (wrapper)
// subclasses, sparse, meta, other devices -- has more keys and is rejected.
const c10::DispatchKeySet kPlainCpu({
    c10::DispatchKey::CPU,
    c10::DispatchKey::AutogradCPU,
    c10::DispatchKey::ADInplaceOrView,
    c10::DispatchKey::AutocastCPU,
});

// set_fresh_allocator: fresh tensors of min_bytes or more come from
// allocator(sizes, dtype) when it returns a tensor; else from at::empty.
PyObject* g_fresh_allocator = nullptr;
size_t g_fresh_min_bytes = 0;

at::Tensor fresh_tensor(const Ints& sizes, c10::ScalarType dtype) {
  int64_t count = 1;
  for (int64_t size : sizes) count *= size;
  if (g_fresh_allocator && static_cast<size_t>(count) * c10::elementSize(dtype) >= g_fresh_min_bytes) {
    PyObject* shape = PyTuple_New(static_cast<Py_ssize_t>(sizes.size()));
    for (size_t i = 0; shape && i < sizes.size(); ++i)
      PyTuple_SET_ITEM(shape, static_cast<Py_ssize_t>(i), PyLong_FromLongLong(sizes[i]));
    PyObject* made = shape ? PyObject_CallFunctionObjArgs(
                                 g_fresh_allocator, shape,
                                 reinterpret_cast<PyObject*>(torch::getTHPDtype(dtype)), nullptr)
                           : nullptr;
    Py_XDECREF(shape);
    if (!made) PyErr_Clear();
    if (made && THPVariable_Check(made)) {
      at::Tensor tensor = THPVariable_Unpack(made);
      Py_DECREF(made);
      return tensor;
    }
    Py_XDECREF(made);
  }
  return at::empty(sizes, at::TensorOptions().dtype(dtype).device(at::kCPU));
}

Ints contiguous_strides(const Ints& sizes) {
  Ints strides(sizes.size());
  int64_t step = 1;
  for (size_t i = sizes.size(); i-- > 0;) {
    strides[i] = step;
    step *= sizes[i] > 1 ? sizes[i] : 1;
  }
  return strides;
}

int64_t numel(const Ints& sizes) {
  int64_t n = 1;
  for (auto s : sizes) n *= s;
  return n;
}

bool read_ints(PyObject* seq, Ints& out) {
  PyObject* fast = PySequence_Fast(seq, "expected a sequence of ints");
  if (!fast) return false;
  const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
  out.resize(n);
  for (Py_ssize_t i = 0; i < n; ++i) {
    out[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(fast, i));
    if (out[i] == -1 && PyErr_Occurred()) {
      Py_DECREF(fast);
      return false;
    }
  }
  Py_DECREF(fast);
  return true;
}

bool read_dtype(PyObject* obj, at::ScalarType& out) {
  if (!THPDtype_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "expected a torch.dtype");
    return false;
  }
  out = reinterpret_cast<THPDtype*>(obj)->scalar_type;
  return true;
}

// A plain tensor of exactly these dtype, sizes and strides, its data aligned;
// its data pointer in ``ptr``. A tensor whose metadata cannot be read (a c10
// error) does not match.
bool tensor_matches(const at::Tensor& t, at::ScalarType dtype, const Ints& sizes,
                    const Ints& strides, int64_t alignment, int64_t& ptr) {
  try {
    if (!t.defined() || (t.key_set() | kPlainCpu) != kPlainCpu ||
        t.layout() != at::kStrided || !t.has_storage() || t.scalar_type() != dtype)
      return false;
    const auto* impl = t.unsafeGetTensorImpl();
    if (impl->has_symbolic_sizes_strides()) return false;
    const auto t_sizes = impl->sizes();
    const auto t_strides = impl->strides();
    if (t_sizes.size() != sizes.size()) return false;
    for (size_t i = 0; i < sizes.size(); ++i)
      if (t_sizes[i] != sizes[i] || t_strides[i] != strides[i]) return false;
    if (t.is_neg() || t.is_conj()) return false;
    ptr = reinterpret_cast<int64_t>(t.data_ptr());
    return ptr != 0 && (alignment <= 1 || ptr % alignment == 0);
  } catch (const c10::Error&) {
    return false;
  }
}

bool initialized(const void* field, const char* type) {
  if (field) return true;
  PyErr_Format(PyExc_RuntimeError, "%s was not initialized", type);
  return false;
}

// --------------------------------------------------------------------------
// EntryCall

struct ArgSpec {
  at::ScalarType dtype;
  Ints sizes, strides;
  int64_t nbytes;
  bool inout;
};

struct EntryCall {
  PyObject_HEAD
  EntryFunc func;
  int64_t alignment;
  std::vector<ArgSpec>* specs;
  Ints* words;                 // descriptor template: sizes and strides set
  std::vector<size_t>* slots;  // start of each descriptor in words
};

PyTypeObject EntryCallType = {PyVarObject_HEAD_INIT(nullptr, 0)};

// Checks the pointers for aliasing (no inout overlaps another argument),
// fills the descriptors and calls the entry without the GIL.
bool call_entry(EntryCall* self, const Ints& pointers) {
  const auto& specs = *self->specs;
  const size_t n = specs.size();
  for (size_t i = 0; i < n; ++i) {
    if (!specs[i].inout) continue;
    const int64_t start = pointers[i], end = start + specs[i].nbytes;
    for (size_t j = 0; j < n; ++j)
      if (j != i && pointers[j] < end && start < pointers[j] + specs[j].nbytes)
        return false;
  }
  Ints words(*self->words);
  const size_t descriptors = words.size() - 2 * n;
  const auto base = reinterpret_cast<int64_t>(words.data());
  for (size_t i = 0; i < n; ++i) {
    const size_t slot = (*self->slots)[i];
    words[slot] = words[slot + 1] = pointers[i];
    words[descriptors + i] = base + 8 * static_cast<int64_t>(slot);
    words[descriptors + n + i] = base + 8 * static_cast<int64_t>(descriptors + i);
  }
  void** packed = reinterpret_cast<void**>(words.data() + descriptors + n);
  EntryFunc func = self->func;
  Py_BEGIN_ALLOW_THREADS
  func(packed);
  Py_END_ALLOW_THREADS
  return true;
}

// EntryCall(address, [(sizes, dtype, inout), ...], alignment)
int EntryCall_init(EntryCall* self, PyObject* args, PyObject*) {
  unsigned long long address;
  PyObject* spec_list;
  long long alignment;
  if (!PyArg_ParseTuple(args, "KOL", &address, &spec_list, &alignment)) return -1;
  if (self->specs) {
    PyErr_SetString(PyExc_RuntimeError, "EntryCall is already initialized");
    return -1;
  }
  if (!address) {
    PyErr_SetString(PyExc_ValueError, "EntryCall: address 0");
    return -1;
  }
  std::vector<ArgSpec> specs;
  Ints words;
  std::vector<size_t> slots;
  PyObject* fast = PySequence_Fast(spec_list, "specs must be a sequence");
  if (!fast) return -1;
  for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fast); ++i) {
    PyObject *sizes, *dtype;
    int inout;
    ArgSpec spec;
    if (!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(fast, i), "OOp", &sizes, &dtype,
                          &inout) ||
        !read_ints(sizes, spec.sizes) || !read_dtype(dtype, spec.dtype)) {
      Py_DECREF(fast);
      return -1;
    }
    spec.strides = contiguous_strides(spec.sizes);
    spec.nbytes = static_cast<int64_t>(c10::elementSize(spec.dtype)) * numel(spec.sizes);
    spec.inout = inout;
    slots.push_back(words.size());
    words.insert(words.end(), {0, 0, 0});
    words.insert(words.end(), spec.sizes.begin(), spec.sizes.end());
    words.insert(words.end(), spec.strides.begin(), spec.strides.end());
    specs.push_back(std::move(spec));
  }
  Py_DECREF(fast);
  words.resize(words.size() + 2 * specs.size(), 0);
  self->func = reinterpret_cast<EntryFunc>(address);
  self->alignment = alignment;
  self->specs = new std::vector<ArgSpec>(std::move(specs));
  self->words = new Ints(std::move(words));
  self->slots = new std::vector<size_t>(std::move(slots));
  return 0;
}

void EntryCall_dealloc(EntryCall* self) {
  delete self->specs;
  delete self->words;
  delete self->slots;
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

// call(values) -> bool: True if the entry ran with ``values`` as they are.
PyObject* EntryCall_call(EntryCall* self, PyObject* values) {
  HANDLE_TH_ERRORS
  if (!initialized(self->specs, "EntryCall")) return nullptr;
  if (!g_enabled) Py_RETURN_FALSE;
  PyObject* fast = PySequence_Fast(values, "values must be a sequence");
  if (!fast) return nullptr;
  const auto& specs = *self->specs;
  const size_t n = specs.size();
  bool ok = static_cast<size_t>(PySequence_Fast_GET_SIZE(fast)) == n;
  Ints pointers(n);
  for (size_t i = 0; ok && i < n; ++i) {
    PyObject* value = PySequence_Fast_GET_ITEM(fast, i);
    ok = THPVariable_CheckExact(value) &&
        tensor_matches(THPVariable_Unpack(value), specs[i].dtype, specs[i].sizes,
                       specs[i].strides, self->alignment, pointers[i]);
  }
  // ``fast`` (the list or a new reference to it) keeps the tensors alive.
  ok = ok && call_entry(self, pointers);
  Py_DECREF(fast);
  return PyBool_FromLong(ok);
  END_HANDLE_TH_ERRORS
}

PyMethodDef EntryCall_methods[] = {
    {"call", reinterpret_cast<PyCFunction>(EntryCall_call), METH_O,
     "call(values) -> bool: run the entry if every value is a tensor it takes as is"},
    {nullptr, nullptr, 0, nullptr}};

// --------------------------------------------------------------------------
// KernelRun

// Parameters: a tensor, an object compared by identity, a scalar compared by
// type and == (floats by their bits: 0.0 == -0.0), an hl.constexpr compared by
// type and its value's type and likewise,
// a function compared by its code (==) and its closure cells' contents (each by
// identity, or as kEqual / kConstExpr), as Helion keys it.
enum Kind { kTensor = 0, kIdentity = 1, kEqual = 2, kConstExpr = 3, kFunction = 4 };
enum Source { kParam = 0, kFresh = 1 };                // entry arguments, result

struct Param {
  int kind;
  PyObject* value;  // other kinds: the recorded object (owned)
  at::ScalarType dtype;
  Ints sizes, strides;  // kFunction: sizes holds the cells' kinds
  PyObject* cells;      // kFunction: the cells' contents when planned (owned)
};

// 1 if ``a`` matches the recorded ``b`` of ``kind`` (kIdentity, kEqual or
// kConstExpr), 0 if not, -1 on a Python error.
int same_value(int kind, PyObject* a, PyObject* b) {
  if (kind == kIdentity) return a == b;
  if (Py_TYPE(a) != Py_TYPE(b)) return 0;
  if (kind == kConstExpr) {  // a NamedTuple (value,)
    if (!PyTuple_Check(a) || PyTuple_GET_SIZE(a) != 1) return 0;
    a = PyTuple_GET_ITEM(a, 0);
    b = PyTuple_GET_ITEM(b, 0);
    if (Py_TYPE(a) != Py_TYPE(b)) return 0;
  }
  if (PyFloat_CheckExact(a)) {
    const double x = PyFloat_AS_DOUBLE(a), y = PyFloat_AS_DOUBLE(b);
    return std::memcmp(&x, &y, sizeof x) == 0;
  }
  return PyObject_RichCompareBool(a, b, Py_EQ);
}

int same_function(const Param& p, PyObject* fn) {
  if (fn == p.value) return 1;
  if (!PyFunction_Check(fn)) return 0;
  // Equal code and closure cells, e.g. a new closure per call. The kernel's
  // function key already told apart functions of other globals or defaults
  // (inject.py) and led here only if they matched; a bound kernel called
  // directly with another function is not checked for them, on any path.
  const int same_code =
      PyObject_RichCompareBool(PyFunction_GET_CODE(fn), PyFunction_GET_CODE(p.value), Py_EQ);
  if (same_code <= 0) return same_code;
  PyObject* closure = PyFunction_GET_CLOSURE(fn);
  const Py_ssize_t n = closure ? PyTuple_GET_SIZE(closure) : 0;
  if (n != PyTuple_GET_SIZE(p.cells)) return 0;
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* content = PyCell_GET(PyTuple_GET_ITEM(closure, i));
    if (!content) return 0;
    const int same = same_value(static_cast<int>(p.sizes[i]), content,
                                PyTuple_GET_ITEM(p.cells, i));
    if (same <= 0) return same;
  }
  return 1;
}

struct Fresh {
  at::ScalarType dtype;
  Ints sizes;
};

struct Plan {
  std::vector<Param> params;
  std::vector<std::pair<int, int>> sources;  // (Source, index) per entry argument
  std::vector<Fresh> fresh;
  std::pair<int, int> result;  // index -1: None
};

enum State { kUnplanned = 0, kPlanning = 1, kPlanned = 2, kNoPlan = 3 };

// Which check sent a planned call to the general path (call_trace.py names them).
// Written on that path only: a native call pays nothing for them.
enum Refusal {
  kRefusedArgCount = 0,
  kRefusedDispatch = 1,
  kRefusedHooks = 2,
  kRefusedArgument = 3,  // index: the argument
  kRefusedEntryArg = 4,  // index: the entry argument (alignment, or its own metadata)
  kRefusedAliasing = 5,
  kRefusalKinds = 6,
};

struct KernelRun {
  PyObject_HEAD
  PyObject* general;  // the Python run(*args)
  PyObject* planner;  // planner(kernel_run, *args): the first call
  PyObject* entry;    // EntryCall
  PyObject* hooks;    // (target type, base prepare_call, base finish_call)
  Plan* plan;
  int state;
  unsigned long long native_calls;  // calls that ran natively (tests, profiling)
  unsigned long long refusals[kRefusalKinds];  // refused calls, per check
  int last_refusal, last_refusal_index;        // the latest; kind -1: none yet
};

PyTypeObject KernelRunType = {PyVarObject_HEAD_INIT(nullptr, 0)};

void free_plan(Plan* plan) {
  if (!plan) return;
  for (auto& p : plan->params) {
    Py_XDECREF(p.value);
    Py_XDECREF(p.cells);
  }
  delete plan;
}

int KernelRun_init(KernelRun* self, PyObject* args, PyObject*) {
  PyObject *general, *planner;
  if (!PyArg_ParseTuple(args, "OO", &general, &planner)) return -1;
  if (self->general) {
    PyErr_SetString(PyExc_RuntimeError, "KernelRun is already initialized");
    return -1;
  }
  Py_INCREF(general);
  Py_INCREF(planner);
  self->general = general;
  self->planner = planner;
  self->last_refusal = -1;
  return 0;
}

void KernelRun_dealloc(KernelRun* self) {
  Py_XDECREF(self->general);
  Py_XDECREF(self->planner);
  Py_XDECREF(self->entry);
  Py_XDECREF(self->hooks);
  free_plan(self->plan);
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

// The target's call hooks are still the defaults (they may be replaced after
// the plan was built; CompiledEntry.call_in_place checks the same).
bool default_hooks(KernelRun* self) {
  PyObject* type = PyTuple_GET_ITEM(self->hooks, 0);
  static const char* names[2] = {"prepare_call", "finish_call"};
  for (int i = 0; i < 2; ++i) {
    PyObject* hook = PyObject_GetAttrString(type, names[i]);
    if (!hook) {
      PyErr_Clear();
      return false;
    }
    const bool same = hook == PyTuple_GET_ITEM(self->hooks, 1 + i);
    Py_DECREF(hook);
    if (!same) return false;
  }
  return true;
}

// No Python torch-function or dispatch mode and no thread-local dispatch keys
// included beyond the defaults (e.g. functorch's transforms): torch.empty
// could then dispatch differently from at::empty.
bool plain_dispatch() {
  if (at::impl::torch_function_mode_enabled() ||
      c10::impl::TorchDispatchModeTLS::stack_len() > 0)
    return false;
  const auto included = c10::impl::tls_local_dispatch_key_set().included_;
  return (included | c10::default_included_set) == c10::default_included_set;
}

// A prepared call's argument views: the caller passes a contiguous tensor of
// ``sizes`` for a tensor parameter whose plan is contiguous with the same
// number of elements; it stands for its view of the plan's sizes (same data).
struct View {
  bool set = false;
  Ints sizes, strides;
};

// Why ``args`` can't run the plan, as far as is known before allocating: the
// checks of run_plan, without running anything or changing the plan's records.
// Returns the Refusal with ``index`` set, -1 if the checks pass, -2 on a Python
// error.
int check_arguments(KernelRun* self, PyObject* const* args, Py_ssize_t nargs,
                    const std::vector<View>* views, int& index) {
  const Plan* plan = self->plan;
  index = -1;
  if (nargs != static_cast<Py_ssize_t>(plan->params.size())) return kRefusedArgCount;
  if (!plain_dispatch()) return kRefusedDispatch;
  if (!default_hooks(self)) return kRefusedHooks;
  for (Py_ssize_t i = 0; i < nargs; ++i) {
    PyObject* arg = args[i];
    const Param& p = plan->params[i];
    if (p.kind == kTensor) {
      const bool viewed = views && (*views)[i].set;
      int64_t ptr;
      if (!THPVariable_CheckExact(arg) ||
          !tensor_matches(THPVariable_Unpack(arg), p.dtype,
                          viewed ? (*views)[i].sizes : p.sizes,
                          viewed ? (*views)[i].strides : p.strides, 1, ptr)) {
        index = static_cast<int>(i);
        return kRefusedArgument;
      }
    } else {
      const int same =
          p.kind == kFunction ? same_function(p, arg) : same_value(p.kind, arg, p.value);
      if (same < 0) return -2;
      if (same == 0) {
        index = static_cast<int>(i);
        return kRefusedArgument;
      }
    }
  }
  return -1;
}

void note_refusal(KernelRun* self, int refusal, int index) {
  ++self->refusals[refusal];
  self->last_refusal = refusal;
  self->last_refusal_index = index;
}

// Runs the plan natively for ``args``. Returns a new reference, or nullptr
// with ``fallback`` set if the call must take the general path (nullptr
// without it: a Python error).
PyObject* run_plan(KernelRun* self, PyObject* const* args, Py_ssize_t nargs,
                   const std::vector<View>* views, const Ints* result_sizes,
                   bool& fallback) {
  fallback = true;
  const Plan* plan = self->plan;
  int refused_index;
  const int refused = check_arguments(self, args, nargs, views, refused_index);
  if (refused != -1) {
    if (refused == -2)
      fallback = false;
    else
      note_refusal(self, refused, refused_index);
    return nullptr;
  }
  const auto& r = plan->result;
  const bool reshaped = result_sizes && r.first == kFresh;
  std::vector<at::Tensor> fresh;
  fresh.reserve(plan->fresh.size());
  for (size_t k = 0; k < plan->fresh.size(); ++k) {
    const auto& f = plan->fresh[k];
    const bool resized = reshaped && r.second == static_cast<int>(k);
    fresh.push_back(fresh_tensor(resized ? *result_sizes : f.sizes, f.dtype));
  }
  EntryCall* entry = reinterpret_cast<EntryCall*>(self->entry);
  const auto& specs = *entry->specs;
  Ints pointers(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    const auto& s = plan->sources[i];
    const bool viewed = s.first == kParam ? views && (*views)[s.second].set
                                          : reshaped && r.second == s.second;
    const at::Tensor& t =
        s.first == kParam ? THPVariable_Unpack(args[s.second]) : fresh[s.second];
    // A viewed tensor was checked above (or allocated) with its own sizes:
    // contiguous with the entry's number of elements (validate checked).
    const Ints own_sizes = viewed ? Ints(t.sizes().begin(), t.sizes().end()) : Ints();
    const Ints own_strides = viewed ? contiguous_strides(own_sizes) : Ints();
    if (!tensor_matches(t, specs[i].dtype, viewed ? own_sizes : specs[i].sizes,
                        viewed ? own_strides : specs[i].strides, entry->alignment,
                        pointers[i])) {
      note_refusal(self, kRefusedEntryArg, static_cast<int>(i));
      return nullptr;
    }
  }
  // ``args`` (owned by our caller) and ``fresh`` keep the tensors alive.
  if (!call_entry(entry, pointers)) {
    note_refusal(self, kRefusedAliasing, -1);
    return nullptr;
  }
  fallback = false;
  ++self->native_calls;
  if (r.second < 0) Py_RETURN_NONE;
  if (r.first == kParam) {
    Py_INCREF(args[r.second]);
    return args[r.second];
  }
  return THPVariable_Wrap(std::move(fresh[r.second]));
}

PyObject* KernelRun_call(KernelRun* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  if (!initialized(self->general, "KernelRun")) return nullptr;
  if (!g_enabled) return PyObject_Call(self->general, args, kwargs);
  const bool keywords = kwargs && PyDict_Size(kwargs);
  if (self->state == kPlanned && !keywords) {
    bool fallback;
    PyObject* result = run_plan(self, &PyTuple_GET_ITEM(args, 0), PyTuple_GET_SIZE(args),
                                nullptr, nullptr, fallback);
    if (result || !fallback) return result;
  }
  if (self->state != kUnplanned || keywords) return PyObject_Call(self->general, args, kwargs);
  // The first call: the planner runs it (general path) and may install a
  // plan; one that raises leaves the next call to plan.
  self->state = kPlanning;
  PyObject* head = PyTuple_Pack(1, reinterpret_cast<PyObject*>(self));
  PyObject* planner_args = head ? PySequence_Concat(head, args) : nullptr;
  Py_XDECREF(head);
  PyObject* result =
      planner_args ? PyObject_Call(self->planner, planner_args, nullptr) : nullptr;
  Py_XDECREF(planner_args);
  if (self->state == kPlanning) self->state = result ? kNoPlan : kUnplanned;
  return result;
  END_HANDLE_TH_ERRORS
}

// install(entry, hooks, params, sources, fresh, result), arguments checked by
// the caller (driver._plan): params [(kind, dtype or value, sizes, strides)]
// (kFunction: (kind, function, cell kinds, cell contents)),
// sources [(kind, index)], fresh [(dtype, sizes)], result (kind, index), index
// -1 for None.
PyObject* KernelRun_install(KernelRun* self, PyObject* args) {
  PyObject *entry, *hooks, *params, *sources, *fresh;
  int result_kind, result_index;
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!(ii)", &EntryCallType, &entry, &PyTuple_Type,
                        &hooks, &PyList_Type, &params, &PyList_Type, &sources,
                        &PyList_Type, &fresh, &result_kind, &result_index))
    return nullptr;
  if (self->state != kPlanning || PyTuple_GET_SIZE(hooks) != 3 ||
      !reinterpret_cast<EntryCall*>(entry)->specs ||
      PyList_GET_SIZE(sources) !=
          static_cast<Py_ssize_t>(reinterpret_cast<EntryCall*>(entry)->specs->size())) {
    PyErr_SetString(PyExc_RuntimeError, "KernelRun.install: not planning or bad plan");
    return nullptr;
  }
  auto* plan = new Plan();
  bool ok = true;
  for (Py_ssize_t i = 0; ok && i < PyList_GET_SIZE(params); ++i) {
    Param p{};
    PyObject *value, *sizes, *strides;
    ok = PyArg_ParseTuple(PyList_GET_ITEM(params, i), "iOOO", &p.kind, &value, &sizes,
                          &strides);
    if (ok && p.kind == kTensor) {
      ok = read_dtype(value, p.dtype) && read_ints(sizes, p.sizes) &&
          read_ints(strides, p.strides);
    } else if (ok) {
      Py_INCREF(value);
      p.value = value;
      if (p.kind == kFunction) {
        ok = PyFunction_Check(value) && PyTuple_Check(strides) && read_ints(sizes, p.sizes) &&
            static_cast<Py_ssize_t>(p.sizes.size()) == PyTuple_GET_SIZE(strides);
        Py_INCREF(strides);
        p.cells = strides;
        if (!ok && !PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "bad function param");
      }
    }
    plan->params.push_back(std::move(p));
  }
  for (Py_ssize_t i = 0; ok && i < PyList_GET_SIZE(sources); ++i) {
    std::pair<int, int> s;
    ok = PyArg_ParseTuple(PyList_GET_ITEM(sources, i), "ii", &s.first, &s.second);
    plan->sources.push_back(s);
  }
  for (Py_ssize_t i = 0; ok && i < PyList_GET_SIZE(fresh); ++i) {
    Fresh f;
    PyObject *dtype, *sizes;
    ok = PyArg_ParseTuple(PyList_GET_ITEM(fresh, i), "OO", &dtype, &sizes) &&
        read_dtype(dtype, f.dtype) && read_ints(sizes, f.sizes);
    plan->fresh.push_back(std::move(f));
  }
  if (!ok) {
    free_plan(plan);
    return nullptr;
  }
  plan->result = {result_kind, result_index};
  Py_INCREF(entry);
  Py_INCREF(hooks);
  self->entry = entry;
  self->hooks = hooks;
  self->plan = plan;
  self->state = kPlanned;
  Py_RETURN_NONE;
}

PyObject* KernelRun_planned(KernelRun* self, void*) {
  return PyBool_FromLong(self->state == kPlanned);
}

PyObject* KernelRun_general(KernelRun* self, void*) {
  if (!initialized(self->general, "KernelRun")) return nullptr;
  Py_INCREF(self->general);
  return self->general;
}

PyObject* KernelRun_native_calls(KernelRun* self, void*) {
  return PyLong_FromUnsignedLongLong(self->native_calls);
}

// refusals -> (count per Refusal, ...)
PyObject* KernelRun_refusals(KernelRun* self, void*) {
  PyObject* out = PyTuple_New(kRefusalKinds);
  for (int i = 0; out && i < kRefusalKinds; ++i)
    PyTuple_SET_ITEM(out, i, PyLong_FromUnsignedLongLong(self->refusals[i]));
  return out;
}

// last_refusal -> (Refusal, index) or None
PyObject* KernelRun_last_refusal(KernelRun* self, void*) {
  if (self->last_refusal < 0) Py_RETURN_NONE;
  return Py_BuildValue("(ii)", self->last_refusal, self->last_refusal_index);
}

// check(*args) -> None if the plan's checks on the arguments pass (a call may still
// be refused on allocation-dependent checks), else (Refusal, index). Runs
// nothing, records nothing. For explain(): no plan installed raises.
PyObject* KernelRun_check(KernelRun* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (self->state != kPlanned) {
    PyErr_SetString(PyExc_RuntimeError, "KernelRun.check: no plan installed");
    return nullptr;
  }
  int index;
  const int refused = check_arguments(self, &PyTuple_GET_ITEM(args, 0),
                                      PyTuple_GET_SIZE(args), nullptr, index);
  if (refused == -2) return nullptr;
  if (refused == -1) Py_RETURN_NONE;
  return Py_BuildValue("(ii)", refused, index);
  END_HANDLE_TH_ERRORS
}

// --------------------------------------------------------------------------
// PreparedRun: every call re-checks everything the KernelRun checks, plus
// that the bound kernel still runs that KernelRun.

struct PreparedRun {
  PyObject_HEAD
  KernelRun* run;
  PyObject* bound;     // the BoundKernel: its _run must still be ``run``
  PyObject* args;      // tuple: the bound arguments (None at varying positions)
  PyObject* fallback;  // fallback(*varying)
  std::vector<int>* varying;
  std::vector<View>* views;  // one per argument
  Ints* result_sizes;        // or nullptr
  int valid;                 // the views fit the plan: -1 not yet checked
};

PyTypeObject PreparedRunType = {PyVarObject_HEAD_INIT(nullptr, 0)};

void PreparedRun_dealloc(PreparedRun* self) {
  Py_XDECREF(reinterpret_cast<PyObject*>(self->run));
  Py_XDECREF(self->bound);
  Py_XDECREF(self->args);
  Py_XDECREF(self->fallback);
  delete self->varying;
  delete self->views;
  delete self->result_sizes;
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

// The views and result sizes fit the plan: each viewed parameter is a
// contiguous tensor of the same element count, taken by the entry as is and
// not returned; the result is a fresh tensor of the same element count.
bool validate(const PreparedRun* self) {
  const Plan* plan = self->run->plan;
  const auto& views = *self->views;
  if (views.size() != plan->params.size()) return false;
  const auto& specs = *reinterpret_cast<EntryCall*>(self->run->entry)->specs;
  for (size_t i = 0; i < views.size(); ++i) {
    if (!views[i].set) continue;
    const Param& p = plan->params[i];
    if (p.kind != kTensor || p.strides != contiguous_strides(p.sizes) ||
        numel(p.sizes) != numel(views[i].sizes) ||
        (plan->result.first == kParam && plan->result.second == static_cast<int>(i)))
      return false;
    for (size_t j = 0; j < specs.size(); ++j) {
      const auto& s = plan->sources[j];
      if (s.first == kParam && s.second == static_cast<int>(i) && specs[j].sizes != p.sizes)
        return false;
    }
  }
  const auto& r = plan->result;
  return !self->result_sizes ||
      (r.first == kFresh && r.second >= 0 &&
       numel(plan->fresh[r.second].sizes) == numel(*self->result_sizes));
}

PyObject* PreparedRun_call(PreparedRun* self, PyObject* vargs, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  const auto& varying = *self->varying;
  KernelRun* run = self->run;
  bool native = g_enabled && run->state == kPlanned && !(kwargs && PyDict_Size(kwargs)) &&
      PyTuple_GET_SIZE(vargs) == static_cast<Py_ssize_t>(varying.size());
  if (native) {
    PyObject* current = PyObject_GetAttrString(self->bound, "_run");
    if (!current) return nullptr;
    native = current == reinterpret_cast<PyObject*>(run);
    Py_DECREF(current);
  }
  if (native && self->valid < 0) self->valid = validate(self);
  if (native && self->valid) {
    const Py_ssize_t nargs = PyTuple_GET_SIZE(self->args);
    std::vector<PyObject*> array(nargs);
    for (Py_ssize_t i = 0; i < nargs; ++i) array[i] = PyTuple_GET_ITEM(self->args, i);
    for (size_t k = 0; k < varying.size(); ++k) array[varying[k]] = PyTuple_GET_ITEM(vargs, k);
    bool fallback;
    PyObject* result =
        run_plan(run, array.data(), nargs, self->views, self->result_sizes, fallback);
    if (result || !fallback) return result;
  }
  return PyObject_Call(self->fallback, vargs, kwargs);
  END_HANDLE_TH_ERRORS
}

PyObject* PreparedRun_native_calls(PreparedRun* self, void*) {
  return PyLong_FromUnsignedLongLong(self->run->native_calls);
}

// prepare(bound, args, varying, views, result_sizes, fallback) -> PreparedRun;
// views: one sizes tuple or None per varying position.
PyObject* KernelRun_prepare(KernelRun* self, PyObject* a) {
  PyObject *bound, *args, *varying, *views, *result_sizes, *fallback;
  if (!initialized(self->general, "KernelRun") ||
      !PyArg_ParseTuple(a, "OO!OOOO", &bound, &PyTuple_Type, &args, &varying, &views,
                        &result_sizes, &fallback))
    return nullptr;
  Ints positions, sizes;
  if (!read_ints(varying, positions) ||
      (result_sizes != Py_None && !read_ints(result_sizes, sizes)))
    return nullptr;
  const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
  std::vector<View> view_list(nargs);
  PyObject* fv = PySequence_Fast(views, "views");
  if (!fv) return nullptr;
  bool ok = PySequence_Fast_GET_SIZE(fv) == static_cast<Py_ssize_t>(positions.size());
  for (size_t k = 0; ok && k < positions.size(); ++k) {
    ok = positions[k] >= 0 && positions[k] < nargs;
    PyObject* v = ok ? PySequence_Fast_GET_ITEM(fv, k) : Py_None;
    if (ok && v != Py_None) {
      View& view = view_list[positions[k]];
      ok = read_ints(v, view.sizes);
      view.strides = contiguous_strides(view.sizes);
      view.set = true;
    }
  }
  Py_DECREF(fv);
  if (!ok) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "prepare: bad views");
    return nullptr;
  }
  auto* out = PyObject_New(PreparedRun, &PreparedRunType);
  if (!out) return nullptr;
  for (PyObject* o : {reinterpret_cast<PyObject*>(self), bound, args, fallback}) Py_INCREF(o);
  out->run = self;
  out->bound = bound;
  out->args = args;
  out->fallback = fallback;
  out->varying = new std::vector<int>(positions.begin(), positions.end());
  out->views = new std::vector<View>(std::move(view_list));
  out->result_sizes = result_sizes == Py_None ? nullptr : new Ints(std::move(sizes));
  out->valid = -1;
  return reinterpret_cast<PyObject*>(out);
}

PyMethodDef KernelRun_methods[] = {
    {"install", reinterpret_cast<PyCFunction>(KernelRun_install), METH_VARARGS,
     "install the plan (from the planner)"},
    {"prepare", reinterpret_cast<PyCFunction>(KernelRun_prepare), METH_VARARGS,
     "prepare(bound, args, varying, views, result_sizes, fallback) -> PreparedRun"},
    {"check", reinterpret_cast<PyCFunction>(KernelRun_check), METH_VARARGS,
     "check(*args) -> None or (refusal, index): the plan's checks, running nothing"},
    {nullptr, nullptr, 0, nullptr}};

PyGetSetDef KernelRun_getset[] = {
    {"planned", reinterpret_cast<getter>(KernelRun_planned), nullptr, "a plan is installed", nullptr},
    {"general", reinterpret_cast<getter>(KernelRun_general), nullptr, "the Python run", nullptr},
    {"native_calls", reinterpret_cast<getter>(KernelRun_native_calls), nullptr,
     "the number of calls that ran natively", nullptr},
    {"refusals", reinterpret_cast<getter>(KernelRun_refusals), nullptr,
     "planned calls sent to the general path, per check", nullptr},
    {"last_refusal", reinterpret_cast<getter>(KernelRun_last_refusal), nullptr,
     "(check, index) of the latest refusal, or None", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

PyGetSetDef PreparedRun_getset[] = {
    {"native_calls", reinterpret_cast<getter>(PreparedRun_native_calls), nullptr,
     "native calls of the underlying KernelRun", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

PyObject* set_enabled(PyObject*, PyObject* value) {
  const int enabled = PyObject_IsTrue(value);
  if (enabled < 0) return nullptr;
  g_enabled = enabled;
  Py_RETURN_NONE;
}

PyObject* set_fresh_allocator(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(PyExc_TypeError, "set_fresh_allocator(allocator or None, min_bytes)");
    return nullptr;
  }
  const size_t min_bytes = PyLong_AsSize_t(args[1]);
  if (PyErr_Occurred()) return nullptr;
  PyObject* old = g_fresh_allocator;
  g_fresh_allocator = args[0] == Py_None ? nullptr : Py_NewRef(args[0]);
  g_fresh_min_bytes = min_bytes;
  Py_XDECREF(old);
  Py_RETURN_NONE;
}

PyMethodDef module_methods[] = {
    {"set_enabled", set_enabled, METH_O, "turn the native paths on or off (every call)"},
    {"set_fresh_allocator", reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(set_fresh_allocator)),
     METH_FASTCALL, "the allocator of fresh tensors of min_bytes or more"},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_helion_mlir_native", nullptr, -1,
                          module_methods};

void set_type(PyTypeObject& type, const char* name, size_t size, destructor dealloc) {
  type.tp_name = name;
  type.tp_basicsize = static_cast<Py_ssize_t>(size);
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  type.tp_dealloc = dealloc;
}

}  // namespace

PyMODINIT_FUNC PyInit__helion_mlir_native(void) {
  set_type(EntryCallType, "_helion_mlir_native.EntryCall", sizeof(EntryCall),
           reinterpret_cast<destructor>(EntryCall_dealloc));
  EntryCallType.tp_new = PyType_GenericNew;
  EntryCallType.tp_init = reinterpret_cast<initproc>(EntryCall_init);
  EntryCallType.tp_methods = EntryCall_methods;
  set_type(KernelRunType, "_helion_mlir_native.KernelRun", sizeof(KernelRun),
           reinterpret_cast<destructor>(KernelRun_dealloc));
  KernelRunType.tp_new = PyType_GenericNew;
  KernelRunType.tp_init = reinterpret_cast<initproc>(KernelRun_init);
  KernelRunType.tp_call = reinterpret_cast<ternaryfunc>(KernelRun_call);
  KernelRunType.tp_methods = KernelRun_methods;
  KernelRunType.tp_getset = KernelRun_getset;
  // No tp_new: made only by KernelRun.prepare.
  set_type(PreparedRunType, "_helion_mlir_native.PreparedRun", sizeof(PreparedRun),
           reinterpret_cast<destructor>(PreparedRun_dealloc));
  PreparedRunType.tp_call = reinterpret_cast<ternaryfunc>(PreparedRun_call);
  PreparedRunType.tp_getset = PreparedRun_getset;
  if (PyType_Ready(&EntryCallType) < 0 || PyType_Ready(&KernelRunType) < 0 ||
      PyType_Ready(&PreparedRunType) < 0)
    return nullptr;
  PyObject* m = PyModule_Create(&module_def);
  if (!m) return nullptr;
  for (auto [name, type] : {std::pair{"EntryCall", &EntryCallType},
                            std::pair{"KernelRun", &KernelRunType},
                            std::pair{"PreparedRun", &PreparedRunType}}) {
    Py_INCREF(type);
    PyModule_AddObject(m, name, reinterpret_cast<PyObject*>(type));
  }
  return m;
}
