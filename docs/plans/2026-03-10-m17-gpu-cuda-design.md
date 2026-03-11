# M17 Design: GPU/CUDA + `kernel` Keyword

**Date:** 2026-03-10
**Milestone:** M17
**Deliverable:** Train a model end-to-end on GPU with custom kernels
**Deferred to M19:** `@fuse` (operator fusion), `@autotune` (parameter search), `nsl profile`

---

## 1. Architecture Decisions

### 1.1 Multi-Backend Strategy (CUDA First)

CUDA is the first and only GPU backend for M17. The architecture does not introduce
a `GpuBackend` trait or vtable — dispatch is a simple `if device == 0 { cpu } else { cuda }`
branch inside existing runtime functions. A trait abstraction will be introduced when a
second backend (Metal, ROCm) is actually implemented in a later milestone.

**Rationale:** Premature abstraction over backends that don't exist yet adds complexity
without value. The unified tensor with device tag (Section 1.2) provides enough structure
to add backends later without a trait.

### 1.2 Unified Tensor with Device Tag

A single `NslTensor` struct serves both CPU and GPU. No separate `GpuTensor` type.

```rust
#[repr(C)]
pub struct NslTensor {
    pub(crate) data: *mut c_void,   // Opaque: CPU f64 or GPU f32
    pub(crate) shape: *mut i64,
    pub(crate) strides: *mut i64,
    pub(crate) ndim: i64,
    pub(crate) len: i64,
    pub(crate) refcount: i64,
    pub(crate) device: u8,          // 0 = CPU, 1+ = CUDA device ID
    pub(crate) dtype: u8,           // 0 = f64, 1 = f32
}
```

**Critical ABI note:** The existing field layout (`data`, `shape`, `strides`, `ndim`, `len`,
`refcount`) must be preserved exactly. `device` and `dtype` are appended at the end.
All FFI signatures remain `i64` pointer-passing — no ABI changes to Cranelift codegen
or the autodiff tape.

### 1.3 Asymmetric Dtype Strategy

For M17, dtypes are restricted by hardware:

- **CPU:** Always `f64` (`dtype == 0`). All existing CPU ops assert `dtype == 0` and cast
  `data` to `*mut f64` via a `data_f64()` accessor method. Changing `data` from `*mut f64`
  to `*mut c_void` requires updating ~50 cast sites in `tensor.rs` and `autodiff.rs` to use
  this accessor — this is a mechanical sweep, not a logic refactor.
- **GPU:** Always `f32` (`dtype == 1`). All GPU PTX kernels use `.f32` registers and
  4-byte addressing.

`nsl_tensor_to_device` handles the f64 ↔ f32 conversion during transfer. This avoids
the "dtype explosion" refactor (rewriting 50+ CPU functions to dispatch on dtype) while
delivering practical GPU training at standard ML precision.

### 1.4 Kernel Compilation: PTX Templates (Evolving to Full Codegen)

User-defined `kernel` blocks are compiled to PTX text strings at NSL compile time.
PTX uses unlimited virtual registers — the NVIDIA assembler (`ptxas`, inside the driver)
handles physical register allocation. This avoids building a full GPU register allocator
for M17.

Long-term (post-M19), the `KernelCompiler` evolves toward direct PTX codegen from
the AST. For M17, the compiler generates PTX from a combination of AST walking and
parameterized templates (dtype-aware instruction selection, element size calculations).

### 1.5 CUDA as Optional Cargo Feature

```toml
[dependencies]
cudarc = { version = "0.12", default-features = false, features = ["driver"], optional = true }

[features]
default = []
cuda = ["cudarc"]
```

Without `--features cuda`, the compiler builds and runs without any CUDA dependency.
If an NSL program uses `cuda` device annotations or kernel blocks, the semantic checker
detects this and checks `cfg!(feature = "cuda")`. If the feature is absent, a compile-time
error directs the user to rebuild with `--features cuda`. No `--cuda` CLI flag needed —
the type system is the source of truth.

---

## 2. Runtime: CUDA Memory + Context

### 2.1 CUDA Initialization

Lazy initialization on first GPU tensor allocation:

- `cuInit(0)`, `cuDeviceGet`, `cuCtxCreate`
- Stored in a global `OnceLock<CudaContext>`
- Exposed as `nsl_cuda_init()` FFI function

### 2.2 Memory: CUDA Unified Memory

Initial implementation uses `cudaMallocManaged` (Unified Memory). The driver handles
page migration between CPU and GPU automatically. This eliminates separate host/device
allocation paths during early development.

**Allocation:** Uses `cudarc::driver::sys` raw bindings (not `CudaSlice`) to avoid the
ownership trap — `CudaSlice`'s `Drop` impl would free memory when the Rust FFI function
returns, leaving the NSL runtime with a dangling pointer. Manual alloc/free via:

- Allocate: `cudarc::driver::sys::cuMemAllocManaged(&mut ptr, size, CU_MEM_ATTACH_GLOBAL)`
- Free: `cudarc::driver::sys::cuMemFree(ptr)`

Lifecycle is managed by NSL's existing `refcount` system, not Rust's `Drop`.

### 2.3 Device Transfer

`nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64`:

All FFI parameters use `i64` (consistent with project convention). The function casts
`target_device` to `u8` internally.

1. Allocates a new tensor on `target_device`
2. Casts data: f64 → f32 (CPU→GPU) or f32 → f64 (GPU→CPU)
3. Copies data via `cuMemcpy`
4. Calls `cuMemPrefetchAsync(ptr, size, device, NULL)` to eagerly migrate pages to GPU
   VRAM (avoids PCIe page faults on first kernel launch). Note: `cuMemPrefetchAsync`
   requires `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS` support. On devices
   without this capability (e.g., Maxwell/Pascal under WDDM), the call returns
   `CUDA_ERROR_NOT_SUPPORTED` — this error is silently ignored as the driver will
   handle page migration on demand instead.
5. Returns new tensor pointer (original unchanged — value semantics)

---

## 3. The Tracer Bullet: Kernel Codegen Pipeline

### 3.1 Stage A: PTX Generation (Compile Time)

New module: `crates/nsl-codegen/src/kernel.rs`

The `KernelCompiler` walks a `KernelDef` AST node and emits a PTX string:

- **Registers:** PTX virtual registers (`.reg .f32 %fs<N>`, `.reg .u64 %rd<N>`)
- **Intrinsic mapping:** `thread_id()` → `%tid.x`, `block_id()` → `%ctaid.x`, etc.
- **Dtype-aware instructions:** `f32` → `ld.global.f32`, `shl idx 2`; `f64` → `ld.global.f64`, `shl idx 3`
- **Shared memory:** `shared let buf: [f32; 256]` → `.shared .f32 buf[256];`

Example output for a vector add kernel:

```
.version 7.0
.target sm_52
.address_size 64

.visible .entry vec_add(
    .param .u64 a, .param .u64 b, .param .u64 c, .param .u64 N
) {
    .reg .u32 %r<2>;
    .reg .u64 %rd<8>;
    .reg .f32 %fs<4>;
    .reg .pred %p1;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [c];
    ld.param.u64 %rd4, [N];

    mov.u32 %r1, %tid.x;
    cvt.u64.u32 %rd5, %r1;
    setp.ge.u64 %p1, %rd5, %rd4;
    @%p1 bra DONE;

    shl.b64 %rd6, %rd5, 2;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.f32 %fs1, [%rd7];
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %fs2, [%rd7];
    add.f32 %fs3, %fs1, %fs2;
    add.u64 %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %fs3;

DONE:
    ret;
}
```

### 3.2 Stage B: PTX Embedding (Compile Time)

Cranelift embeds the PTX string as a null-terminated byte array in `.rodata`:

```rust
let mut ptx_bytes = ptx_string.into_bytes();
ptx_bytes.push(0); // Null-terminate for CUDA driver API

let data_id = module.declare_data(
    &format!("__nsl_ptx_{}", kernel_name),
    Linkage::Local, false, false  // writable=false, .rodata
)?;
```

The kernel's entry point name is also embedded as a separate null-terminated string:

```rust
let name_id = module.declare_data(
    &format!("__nsl_ptx_name_{}", kernel_name),
    Linkage::Local, false, false
)?;
```

### 3.3 Stage C: Kernel Launch FFI (Runtime)

New FFI function in `crates/nsl-runtime/src/cuda.rs`:

```rust
#[no_mangle]
pub extern "C" fn nsl_kernel_launch(
    ptx_ptr: *const u8,      // pointer to PTX string in .rodata
    ptx_len: i64,            // length of PTX string
    name_ptr: *const u8,     // pointer to kernel entry point name
    name_len: i64,           // length of name string
    grid_x: i64, grid_y: i64, grid_z: i64,
    block_x: i64, block_y: i64, block_z: i64,
    args: *const *mut u8,    // array of pointers to kernel arguments
    num_args: i64,
) -> i64  // 0 = success, nonzero = CUDA error
```

Internally:
1. `cuModuleLoadData(&module, ptx_ptr)` — JIT compile PTX
2. `cuModuleGetFunction(&func, module, name_ptr)` — get entry point by name
3. `cuLaunchKernel(func, grid, block, args, ...)` — launch
4. Module cache: `HashMap<*const u8, CUmodule>` keyed by PTX pointer address
   (static `.rodata` addresses are stable, zero-cost cache keys).
   **Concurrency model:** The cache is stored in a `static Mutex<HashMap<...>>`.
   NSL is currently single-threaded, so lock contention is not a concern. If
   multi-threading is added later, this can be upgraded to a lock-free map or
   per-thread cache. For standard-op PTX (`const &str` in the runtime), the
   pointer addresses are also stable process-wide.

### 3.4 Stage D: Kernel Call Sites (Codegen)

NSL kernel calls use explicit grid/block configuration:

```python
vec_add(a, b, c, grid=N/256, block=256)
tiled_matmul(a, b, c, grid=(M/64, N/64), block=(16, 16))
```

`grid` and `block` are special named arguments on kernel call expressions. The parser
extracts them.

**Scalar vs tuple normalization:** A scalar value like `grid=N/256` maps to
`grid_x=N/256, grid_y=1, grid_z=1`. A tuple like `grid=(M/64, N/64)` maps to
`grid_x=M/64, grid_y=N/64, grid_z=1`. A 3-tuple sets all three dimensions.
The parser distinguishes tuples from parenthesized expressions by checking for
commas inside the parentheses — `(expr)` is a paren group, `(expr, expr)` is a tuple.

Codegen:

1. Evaluates `grid` and `block` expressions, normalizes to 3D
2. Loads `__nsl_ptx_vec_add` and `__nsl_ptx_name_vec_add` pointers from `.rodata`
3. Marshals tensor arguments into the `args` array
4. Calls `nsl_kernel_launch(ptx, len, name, name_len, grid, block, args, num_args)`

If `grid` or `block` are omitted at a kernel call site, the compiler emits an error.

---

## 4. Standard GPU Tensor Ops

### 4.1 Dispatch Pattern

Existing runtime functions get device branches:

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_add(a_ptr: i64, b_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(a_ptr);
    let b = NslTensor::from_ptr(b_ptr);
    assert_eq!(a.device, b.device, "device mismatch");

    if a.device == 0 {
        cpu::tensor_add(a_ptr, b_ptr)
    } else {
        #[cfg(feature = "cuda")]
        { cuda::tensor_add(a_ptr, b_ptr) }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled. Rebuild with --features cuda") }
    }
}
```

### 4.2 Standard PTX Kernels

PTX templates for standard ops live as `const &str` in `crates/nsl-runtime/src/cuda/kernels.rs`.
They feed into the same `cuModuleLoadData` caching pipeline as user-defined kernels.

| Op | PTX Kernel | Notes |
|---|---|---|
| add/sub/mul/div | elementwise binary | Same template, different instruction |
| relu/sigmoid/tanh/exp/log | elementwise unary | |
| matmul | tiled GEMM | Shared memory tiles, most complex kernel |
| sum/mean | parallel reduction | Tree reduction in shared memory |
| softmax | fused | Per-row max, subtract, exp, sum, divide |
| transpose | shared memory tiled | Avoid uncoalesced memory access |
| gather | indexed load | Embedding lookups |
| mul_scalar | scalar broadcast | Reuses existing `nsl_tensor_mul_scalar` |

### 4.3 CPU Ops Extraction

Existing CPU implementations in `tensor.rs` are extracted into `crates/nsl-runtime/src/cpu.rs`.
Same functions, same logic, just moved. `tensor.rs` becomes the dispatch layer. No trait,
no behavior change — pure file reorganization.

---

## 5. GPU Autodiff

### 5.1 Why It Mostly Just Works

The autodiff tape stores `TapeOp` variants with `i64` tensor pointers. During `.backward()`,
the tape calls the same FFI functions (`nsl_tensor_add`, `nsl_tensor_mul`, etc.) to compute
gradients. Since those functions now dispatch on `device`, if the forward pass ran on GPU,
the backward pass runs on GPU automatically.

No new `TapeOp` variants are needed. GPU ops don't change differentiation rules.

### 5.2 Required Changes

**Gradient allocation respects device:**

```rust
// New: allocate on same device/dtype as the tensor
nsl_tensor_zeros_like(tensor_ptr) -> i64
nsl_tensor_ones_like(tensor_ptr) -> i64
```

`zeros_like` and `ones_like` inherit `device` and `dtype` from the input tensor.

**Backward seed:** The loss gradient seed must use `ones_like(loss_ptr)` instead of
`tensor_ones(shape)` to ensure the seed tensor is on the same device as the loss.
Otherwise, the first backward step panics on device mismatch.

**Critical: update internal `ones_like` in `autodiff.rs`:** The private `ones_like` helper
function in `autodiff.rs` (used at the backward seed site and in SumReduce/MeanReduce
backward helpers) currently calls `tensor_ones(shape_list)`, which always allocates on CPU.
This internal helper must be updated to call the new `nsl_tensor_ones_like` FFI function
that inherits device/dtype. Failing to update this private function will cause GPU backward
to panic on device mismatch regardless of the new FFI functions existing.

**All backward helpers get PTX dispatch:** Every backward function that touches `data`
in a loop (clamp_backward, relu_backward, softmax_backward, etc.) must dispatch to
a GPU PTX kernel. CPU pointer dereference on Unified Memory triggers PCIe page faults,
causing ~100x slowdown.

**Scalar broadcast:** Reuse existing `nsl_tensor_mul_scalar` with a CUDA dispatch branch
and scalar multiply PTX template. No new function needed.

### 5.3 No Synchronization Needed

CUDA kernels on the same stream (Stream 0, the default) execute in strict order.
Forward and backward kernels are queued sequentially — the GPU executes them in order
without CPU intervention. `cuCtxSynchronize()` is only called when data is pulled back
to CPU (print, checkpoint save, `.to(cpu)`).

---

## 6. Semantic Validation

### 6.1 Kernel Body Restrictions

New module: `crates/nsl-semantic/src/kernel.rs`

The semantic checker tracks an `in_kernel: bool` flag. Inside kernel bodies:

**Allowed:**
- Scalar arithmetic and comparisons
- Tensor element indexing (`a[idx]`)
- Hardware intrinsics (see Section 6.2)
- Local variable declarations (`let`)
- Control flow (`if`/`else`, `while`)
- Shared memory declarations (`shared let buf: [f32; 256]`)

**Not allowed (compile error):**
- Function calls to non-intrinsic functions
- Closures and higher-order functions
- Heap allocation
- Print, file I/O, or any non-GPU runtime op

### 6.2 Hardware Intrinsics

Registered in `crates/nsl-semantic/src/builtins.rs`, valid only inside kernel bodies:

| NSL Intrinsic | PTX Mapping | Type Signature |
|---|---|---|
| `thread_id()` | `%tid.x` | `() -> int` |
| `thread_id_y()` | `%tid.y` | `() -> int` |
| `block_id()` | `%ctaid.x` | `() -> int` |
| `block_id_y()` | `%ctaid.y` | `() -> int` |
| `block_dim()` | `%ntid.x` | `() -> int` |
| `sync_threads()` | `bar.sync 0` | `() -> void` |
| `atomic_add(ptr, val)` | `atom.global.add` | `(*mut f32, f32) -> f32` |

Using an intrinsic outside a kernel body is a compile error.
Calling a non-intrinsic function inside a kernel body is a compile error.

### 6.3 Device Requirement

All tensor parameters in a kernel must have `device: cuda` (or another GPU device).
If a kernel parameter has `device: cpu` or `device: unknown`, the semantic checker
emits an error.

### 6.4 CUDA Feature Detection

When the semantic checker encounters a `cuda` device annotation or kernel block:

```rust
if !cfg!(feature = "cuda") {
    return Err(CompilerError::new(
        span,
        "This NSL script requires CUDA, but the NSL compiler was built without \
         GPU support. Recompile with `cargo build --features cuda`."
    ));
}
```

**Note:** `cfg!(feature = "cuda")` is a Rust compile-time constant evaluated when the
NSL compiler binary itself is built, not when the user runs `nsl run`. The `if !cfg!(...)`
branch is dead code in a CUDA-enabled binary. This means the `nsl` binary effectively
ships in two variants: CPU-only (default `cargo build`) and CUDA-enabled
(`cargo build --features cuda`). End users install the appropriate variant; there is no
runtime flag to toggle GPU support.

No `--cuda` CLI flag needed — the type system drives compilation mode.

---

## 7. File Organization

### 7.1 New Files

```
crates/nsl-runtime/src/
├── cuda.rs              # CUDA context init, kernel launch, module cache
├── cuda/
│   └── kernels.rs       # Static PTX strings for standard ops
├── cpu.rs               # Extracted CPU implementations (from tensor.rs)
├── tensor.rs            # Dispatch layer (device → cpu:: or cuda::)
└── ...

crates/nsl-codegen/src/
├── kernel.rs            # KernelCompiler: AST → PTX string generation
└── ...

crates/nsl-semantic/src/
├── kernel.rs            # Kernel body validation, intrinsic scoping
└── ...
```

### 7.2 Dependency Changes

```toml
# crates/nsl-runtime/Cargo.toml
[dependencies]
cudarc = { version = "0.12", default-features = false, features = ["driver"], optional = true }

[features]
default = []
cuda = ["cudarc"]
```

---

## 8. Implementation Order (Vertical Slice)

The implementation follows a tracer-bullet strategy, solving the highest-risk
technical unknowns first:

1. **Struct update** — Add `device`, `dtype` to `NslTensor`, change `data` to `*mut c_void`,
   add CPU dtype assertions
2. **Tracer bullet** — Minimal kernel: PTX generation → `.rodata` embedding → `nsl_kernel_launch`
   FFI → `cuModuleLoadData` → `cuLaunchKernel`. Prove the JIT pipeline works end-to-end.
3. **Device transfer** — `nsl_tensor_to_device` with f64↔f32 cast + `cuMemPrefetchAsync`
4. **Core GPU ops** — Device dispatch in existing functions, PTX templates for all standard ops
5. **GPU autodiff** — `zeros_like`/`ones_like` respecting device, backward helpers with PTX dispatch
6. **Semantic validation** — `in_kernel` flag, intrinsic scoping, kernel body restrictions,
   `cfg!` feature detection
7. **Kernel call syntax** — Parser support for `grid`/`block` named args, codegen for launch sites

---

## 9. What This Enables for Users

```python
# CPU model definition (unchanged from M14)
model MLP:
    linear1: Linear(784, 256)
    linear2: Linear(256, 10)

    fn forward(self, x: Tensor) -> Tensor:
        x |> self.linear1 |> relu |> self.linear2

# GPU training — only change is .to(cuda)
let m = MLP()
let x = load_data().to(cuda)
let y = load_labels().to(cuda)
let m_gpu = m.to(cuda)

train m_gpu:
    data: x, y
    optimizer: Adam(lr=0.001)
    epochs: 10

# Custom GPU kernel
kernel fused_gelu(x: Tensor<[N], f32, cuda>, out: Tensor<[N], f32, cuda>):
    let idx = thread_id()
    if idx < N:
        let v = x[idx]
        out[idx] = 0.5 * v * (1.0 + tanh(0.7978845 * (v + 0.044715 * v * v * v)))

fused_gelu(activations, result, grid=N/256, block=256)
```
