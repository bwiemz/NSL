<!-- owner: @bwiemz -->

# Runtime Internals

What happens at runtime after the compiler hands off. Tensor representation, memory allocation, GPU execution, autodiff tape, and FFI ownership conventions. For the compile-time story, see [Compiler-Pipeline](Compiler-Pipeline.md); for optimization passes and PTX synthesis rules, see [Optimization-Passes](Optimization-Passes.md).

## The `NslTensor` struct

Source: [`crates/nsl-runtime/src/tensor/mod.rs:154`](../../crates/nsl-runtime/src/tensor/mod.rs#L154)

```rust
#[repr(C)]
pub struct NslTensor {
    pub(crate) magic: u32,          // MUST be first — 0x4E534C54 ("NSLT") when live, 0x0000DEAD after free
    pub(crate) data: *mut c_void,   // Opaque: CPU f64 or GPU f32
    pub(crate) shape: *mut i64,
    pub(crate) strides: *mut i64,
    pub(crate) ndim: i64,
    pub(crate) len: i64,
    pub(crate) refcount: AtomicI64,
    pub(crate) device: u8,          // 0 = CPU, 1+ = CUDA device ID
    pub(crate) dtype: u16,          // 0 = f64, 1 = f32; 256+ = custom user-defined dtypes
    pub(crate) owns_data: u8,       // 1 = heap-owned (free on drop), 0 = borrowed/mmap
    pub(crate) data_owner: i64,     // 0 = owns data; non-zero = pointer to owning NslTensor (view)
    pub(crate) slab_managed: u8,    // 1 = data is offset into GPU slab (do NOT free individually)
    pub(crate) tape_id: i64,        // monotonic ID assigned during tape recording (0 = unassigned)
}
```

Each field serves a specific role:

- **`magic`** — sentinel that spells `"NSLT"` when the struct is live and `0x0000DEAD` after free. Because `#[repr(C)]` places it first, a freed-tensor dereference produces a recognizable poison value rather than silent data corruption.
- **`data`** — opaque pointer to element storage. Interpretation depends on `device` and `dtype` (see below).
- **`shape`** / **`strides`** — heap-allocated `i64` arrays of length `ndim`, row-major byte strides.
- **`ndim`** / **`len`** — rank and total element count.
- **`refcount`** — atomic reference counter. When it reaches zero the allocator frees `data` (if `owns_data == 1` and `slab_managed == 0`) and the struct itself.
- **`device`** — `0` means CPU; values `1+` are CUDA device IDs (matching the ordinals returned by `cuDeviceGet`).
- **`dtype`** — built-in codes are `0`=f64, `1`=f32, `2`=fp16, `3`=bf16, `4`=int8, `5`=fp8e4m3, `6`=fp8e5m2, `7`=u16 token, `8`=u16 segment. Custom dtypes from `datatype` blocks start at `256`.
- **`owns_data`** — `1` means the struct is responsible for freeing `data`; `0` means the memory is borrowed (e.g. from a view, mmap, or external allocation).
- **`data_owner`** — non-zero means this tensor is a view; the value is a raw `i64` pointer to the NslTensor that actually allocated the buffer. When a view is freed, the owner's refcount is decremented.
- **`slab_managed`** — when `1`, `data` is an interior pointer into a persistent GPU slab. The free path skips `cuMemFree_v2` for this block; the slab is released once at program exit via `nsl_slab_destroy`.
- **`tape_id`** — monotonic integer assigned when an op involving this tensor is recorded onto the autodiff tape. Decouples tensor identity from memory address so that intermediate tensors freed during compilation still participate correctly in gradient bookkeeping. `0` means unassigned (never recorded).

### dtype convention

- **CPU path** — elements are `f64` (`dtype = 0`, `DTYPE_F64`). The `data` pointer is safe to cast to `*mut f64` from CPU code.
- **GPU path** — elements are `f32` (`dtype = 1`, `DTYPE_F32`). The `data` pointer is a CUDA device pointer; see the `data` pointer rules below.

The asymmetry is load-bearing: `.to(cuda)` converts element values from `f64` to `f32` as it copies to device memory. CPU code that casts a GPU tensor's `data` pointer and reads it will mis-decode the values because `f32` and `f64` have different bit widths and layouts.

### `data` pointer rules

- `device == 0` (CPU) → `data` is a host heap pointer, allocated via `checked_alloc`. Safe to dereference from any CPU thread holding a valid refcount.
- `device >= 1` (CUDA) → `data` is a **CUDA device pointer** returned by `cuMemAlloc_v2` (or an interior offset into a GPU slab). The CPU **cannot** dereference it. Use `memcpy_dtoh` to read values to host. Any CPU dereference of a GPU `data` pointer is undefined behavior.

## Allocator

Source: [`crates/nsl-runtime/src/cuda/caching_allocator.rs`](../../crates/nsl-runtime/src/cuda/caching_allocator.rs)

The caching allocator is GPU-only. CPU tensor data uses the system allocator (`checked_alloc` / `checked_alloc_zeroed` wrapping `std::alloc`). The GPU allocator is a PyTorch-style block-splitting, coalescing pool that minimises `cuMemAlloc_v2` syscalls and internal fragmentation.

```rust
pub(crate) struct CachingAllocator<D: DriverAlloc = CudaDriverAlloc> {
    small_free: BTreeSet<FreeBlockKey>,   // free blocks for ≤ 1 MB allocations
    large_free: BTreeSet<FreeBlockKey>,   // free blocks for > 1 MB allocations
    allocated_blocks: HashMap<usize, *mut Block>,
    all_blocks: HashMap<usize, *mut Block>,
    segments: Vec<Segment>,
    stats: AllocStats,
    total_reserved: usize,
    total_allocated: usize,
    memory_limit: usize,
    driver: D,
}
```

Key properties:

- **Two size classes.** Small allocations (≤ 1 MB) are 512-byte aligned; large allocations (> 1 MB) are 2 MB-aligned. Each class has its own free-list `BTreeSet` for O(log n) best-fit search.
- **Segments.** Each `cuMemAlloc_v2` call returns a large region (`SMALL_SEGMENT_SIZE = 2 MB` or `LARGE_SEGMENT_SIZE = 20 MB`) that is subdivided into `Block`s. Blocks form a doubly-linked list within their parent segment ordered by GPU address.
- **Coalescing.** When a block is freed, adjacent free blocks merge in O(1) via the linked-list pointers. Fully-free segments are returned to the driver on `drain_all`.
- **Two pool tags.** `AllocPool::Persistent` covers model weights, optimizer moment buffers, and DataLoader tensors — these segments survive `drain_all`. `AllocPool::Transient` covers forward/backward intermediates and gradients — these segments are released once fully free.
- **Single global lock.** `CACHING_ALLOCATOR` is a `LazyLock<Mutex<CachingAllocator>>`. All alloc/free paths are serialised through this mutex; every CUDA call site must call `ensure_context()` before acquiring it.
- **Memory limit.** `NSL_GPU_MEMORY_LIMIT` environment variable sets a soft ceiling; an over-limit alloc triggers `drain_all` and retries before returning OOM.

## GPU path

Source: [`crates/nsl-runtime/src/cuda/mod.rs`](../../crates/nsl-runtime/src/cuda/mod.rs)

Binds to CUDA via [cudarc](https://github.com/coreylowman/cudarc) `0.19.4` with the `driver`, `cuda-version-from-build-system`, and `dynamic-linking` features enabled.

### Initialization and context management

`ensure_context()` (line 67) must be called before any CUDA driver API call. It is a free function in [`crates/nsl-runtime/src/cuda/mod.rs`](../../crates/nsl-runtime/src/cuda/mod.rs) — no arguments, acquires the global `CudaState` mutex internally. Call it as `crate::cuda::ensure_context()` at the top of any new CUDA call site. It locks `CudaState` and invokes `cuCtxSetCurrent(guard.context)`. Because CUDA contexts are **thread-local state in the driver**, failing to call this on a new thread produces "invalid context" errors deep in cudarc with no useful stack trace.

### CUDA 13.x workarounds

These invariants come from hard-won debugging and are enforced in the code:

- **`cuCtxCreate` was removed in CUDA 13.** NSL uses `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` instead. See [`crates/nsl-runtime/src/cuda/mod.rs:94`](../../crates/nsl-runtime/src/cuda/mod.rs#L94) where `cuDevicePrimaryCtxRetain` is called during `state()` initialization, and line 71 and 101 where `cuCtxSetCurrent` activates the retained context.

- **PTX ISA 7.0 `mad.lo.u32` is invalid in newer CUDA runtimes.** The NSL PTX backend emits `mul.lo.u32` followed by `add.u32` instead. See [`crates/nsl-codegen/src/backend_ptx.rs:376`](../../crates/nsl-codegen/src/backend_ptx.rs#L376) for the comment and line 382 for the emission. A test at line 658 asserts `ptx.contains("mul.lo.u32")` and line 661 asserts `!ptx.contains("mad.lo.u32")`.

- **Contexts are thread-local.** Every call path that touches a CUDA driver API must call `ensure_context()` (which wraps `cuCtxSetCurrent`) before any cudarc or raw driver call. Forgetting this produces "invalid context" errors (`CUDA_ERROR_INVALID_CONTEXT`) with no actionable stack trace from cudarc.

### Module caching

PTX modules are cached in `CudaState::module_cache` keyed by the FNV-1a hash of the PTX content string — not by raw pointer — because heap reuse between sequential test invocations can place different PTX `Vec`s at the same address, producing stale cache hits (`CUDA_ERROR_NOT_FOUND`, rc=500) on the old pointer-keyed design.

## Autodiff tape

Source: [`crates/nsl-runtime/src/autodiff/mod.rs`](../../crates/nsl-runtime/src/autodiff/mod.rs)

The fallback autodiff path. Used when the AOT path (see [Optimization-Passes § AOT autodiff](Optimization-Passes.md#aot-autodiff--the-wengertlist)) cannot prove a construct. The tape is a `thread_local! { static TAPE: RefCell<Tape> }` — each thread has its own recording state.

```rust
pub(crate) struct Tape {
    pub(crate) ops: Vec<TapeOp>,
    pub(crate) param_set: HashSet<i64>,
    pub(crate) recording: bool,
    pub(crate) pause_depth: i32,
    pub(crate) next_id: i64,
}
```

`TapeOp` is a large enum with one variant per differentiable operation (Add, Sub, Mul, MatMul, ReLU, Softmax, FlashAttention, Checkpoint, etc.). Fields within each variant are `i64` values — these are **tape IDs**, not raw pointer casts.

### tape_id indirection

When an op is recorded, `assign_ids` converts each operand pointer into a `tape_id` (the `tape_id` field of `NslTensor`). This monotonic integer decouples tensor identity from memory address: if an intermediate tensor is freed during the forward pass (e.g., by statement-level cleanup), the backward pass can still match grad-map entries by ID rather than by address that may have been reused.

### Load-bearing invariant — pointer vs. ID misuse

`TapeOp` fields named `a`, `b`, `out` store **tape IDs** after `assign_ids` runs. Fields named `saved_a`, `saved_b`, `saved_out` store **live tensor pointers with a refcount bump** — the runtime retains these tensors so backward can read their data.

- **Use `a`, `b`, `out` only as gradient-map keys** — call `accumulate_grad(&mut grad_map, *a, ...)`, never dereference them as `NslTensor` pointers.
- **If backward needs shape info**, store it in the `TapeOp` struct as a `Vec<i64>` (e.g., `input_shape` in `SumReduce`). Do not reconstruct from the `a` pointer.
- **If backward needs tensor data**, add a `saved_a` field with a refcount bump at record time, as `Mul`, `Div`, `MatMul`, `Exp`, `Log`, and `FlashAttention` do.

Two distinct failure modes can result from misusing these fields:

- **Tape ID dereferenced as a pointer** → the "address" is a small integer (e.g. `3`). On modern OSes the low page is unmapped, so the dereference produces an **immediate SIGSEGV / access violation**. If `loss.backward()` dies with a segfault pointing into a backward implementation, the canonical cause is a bare `*a` where `a` is a tape ID.
- **Stale freed pointer dereferenced as live tensor data** → the pointer was valid at `tape_start` time but its backing storage was freed and possibly reallocated. The read succeeds, but the bytes are whatever the allocator reused for — the backward pass silently produces **non-deterministic NaN or garbage gradients**, detectable as divergence between runs on the same seed.

The canonical anti-pattern: a backward implementation calls `ones_like(*a)` to generate a gradient seed. The `ones_from_shape(input_shape, dtype)` helper was introduced specifically to avoid both failure modes — it takes shape and dtype directly from the `TapeOp` record, never touching `a` as a pointer.

### Tape scoping

`nsl_tape_start(param_list)` begins recording; `nsl_tape_stop()` ends the forward pass. Tensors promoted to `TapeHeld` ownership during the forward pass (see [ELTLS ownership](#ownership-side-channel-strict-unknown-fallback) below) are freed after `nsl_tape_backward` completes. The tape itself does not own any tensor storage — it records identities and references.

Strongly prefer AOT autodiff via `--source-ad` when possible. The tape is unused in that path; see [Optimization-Passes § AOT autodiff](Optimization-Passes.md#aot-autodiff--the-wengertlist).

## FFI conventions

### Prefer flags over variants

Source: [`crates/nsl-runtime/src/tensor/fbip_flags.rs`](../../crates/nsl-runtime/src/tensor/fbip_flags.rs)

Every binary tensor FFI takes a single `flags: u8` third argument rather than one function per ownership combination. The two defined bits are:

```rust
/// Bit 0: caller relinquishes operand A.
pub const RELINQUISH_A: u8 = 0x01;
/// Bit 1: caller relinquishes operand B.
pub const RELINQUISH_B: u8 = 0x02;
```

A concrete example from [`crates/nsl-runtime/src/tensor/arithmetic.rs:29`](../../crates/nsl-runtime/src/tensor/arithmetic.rs#L29):

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_add(a: i64, b: i64, flags: u8) -> i64
```

When a bit is set, the runtime may reuse the operand's storage in-place (if shape-compatible) or free it after use. When a bit is clear, the runtime must leave the operand untouched.

The alternative — per-op consume variants like `nsl_tensor_add_consume_lhs` / `_consume_rhs` / `_consume_both` — would produce ~100+ near-identical stubs across the ~20 binary tensor ops. The flags byte scales linearly and costs zero at runtime: Cranelift inlines the constant-fold, and the runtime must check shape for in-place eligibility anyway.

Do not add per-op consume variants. Extend `fbip_flags.rs` with new bit definitions if new ownership modes are needed.

### Ownership side-channel: strict Unknown fallback

Source: [`crates/nsl-codegen/src/ownership_expr.rs`](../../crates/nsl-codegen/src/ownership_expr.rs), [`crates/nsl-codegen/src/compiler/ownership_api.rs`](../../crates/nsl-codegen/src/compiler/ownership_api.rs)

The ELTLS (Expression-Level Tensor Lifetime System) tracks tensor ownership via a `HashMap<ir::Value, Ownership>` side-channel populated by producer sites. The `Ownership` enum has five variants:

```rust
pub enum Ownership {
    Owned,
    BorrowedFromVar(Variable),
    BorrowedWeight,
    TapeHeld,
    Unknown,
}
```

The accessor `get_ownership` returns `Ownership::Unknown` for any value absent from the map:

```rust
pub fn get_ownership(&self, state: &FuncState, val: ir::Value) -> Ownership {
    state.cleanup.expr_ownership.get(&val).copied().unwrap_or(Ownership::Unknown)
}
```

`Unknown` is not a gap or an error state — it is an **explicit conservative fallback** with defined consumer behavior:

- Consumer wants to mutate the tensor → emit `nsl_tensor_clone` first.
- Consumer wants to store the tensor in a variable → emit `nsl_tensor_retain`.
- Consumer is the final use in a statement → leave it alone; the function epilog sweep (`nsl_tensor_free_if_valid`) will handle it.

In `promote_to_tape_held`, an `Unknown` value is given `nsl_tensor_retain` before being promoted to `TapeHeld`, because the originating producer scope may end before the tape region exits. Silent-free on `Unknown` is the root cause of a class of use-after-free bugs; the strict fallback was introduced explicitly to prevent recurrence.

Every `Unknown` hit at a consumer increments `state.cleanup.unknown_ownership_count`, which is observable as a rollout metric.

## Related

- PTX comment encoding rules (ASCII-only in emitted PTX) — see [Optimization-Passes § Load-bearing invariants](Optimization-Passes.md#load-bearing-invariants)
- `@flash_attention` causal default — see [Optimization-Passes § Load-bearing invariants](Optimization-Passes.md#load-bearing-invariants)
- AOT autodiff preferred over tape — see [Optimization-Passes § AOT autodiff](Optimization-Passes.md#aot-autodiff--the-wengertlist)
- Pass ordering and pipeline stages — see [Compiler-Pipeline](Compiler-Pipeline.md)

---

*Last structurally verified against commit `9a1b512e` on 2026-04-21. If the crate graph or struct layout in this page no longer matches reality, open an issue tagged `docs-rot`.*
