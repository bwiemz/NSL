# nsl-runtime architecture

This is the contributor-level map of `crates/nsl-runtime`. It extends
`crates/nsl-runtime/ARCHITECTURE.md` (the FFI safety contract and the facade
table — read that first; nothing there is repeated here) and
`docs/architecture/compiler-state.md` (the machine-checked inventory of every
`thread_local!` in the workspace, including the runtime's own). The
host-facing ABI rules live in `docs/abi/README.md`.

## Overview

`nsl-runtime` builds as `crate-type = ["staticlib", "rlib"]` (`Cargo.toml`).
The static library is what every compiled NSL program links: `nsl build`
emits a Cranelift object file and `nsl_codegen::linker::link` hands `cc` the
object plus `libnsl_runtime.a` (`find_runtime_lib` in
`crates/nsl-codegen/src/linker.rs`). The rlib is what `nsl-cli` and the test
binaries use to call the same functions from Rust.

The crate is the **C-ABI boundary**. Generated code knows nothing about Rust
types: it calls `extern "C"` functions by symbol name with `i64` handles and
raw pointers. Measured on the current tree:

- ~134K lines under `src/`, ~17K lines across 71 integration-test files
  under `tests/`, one criterion bench (`benches/tape.rs`).
- 813 `extern "C"` entry points (794 `pub extern "C" fn` plus 19
  `pub unsafe extern "C" fn`), 813 `no_mangle` attributes (810
  `#[unsafe(no_mangle)]`, 3 legacy `#[no_mangle]`). The count matters because
  the codegen declares each one in a `RUNTIME_FUNCTIONS*` table and the two
  sides are tied together only by name — see "Where to add a new X".

The one contract every entry point upholds (ownership, nullability,
lifetimes, layout, sentinel errors, "panics are aborts") is stated in
`src/lib.rs`'s crate docs and summarized in `ARCHITECTURE.md`. Two
consequences shape the code you will read:

1. Fatal conditions do not `panic!`. They print, flush stderr, and either
   `std::process::abort()` (`bad_handle` in `src/tensor/mod.rs`,
   `src/assert.rs`) or `std::process::exit(code)` through `src/fatal.rs`
   (`Fatal::{GpuOom, CudaDriver, CudaAsync, Cublas}`, one exit code each —
   see "Fatal exits" below). A panic inside an `extern "C"` frame cannot
   unwind; it becomes SIGABRT after a second backtrace and, on a GPU box, a
   multi-GB core dump.
2. Recoverable failures are sentinel returns (`0`, `-1`, null) and, on the
   host-facing C API, a per-thread error string readable through
   `nsl_get_last_error` (`src/c_api/mod.rs`).

### Feature flags

From `Cargo.toml`:

| Feature | Effect |
|---|---|
| `cuda` | The CUDA driver backend (`cudarc` with `driver`, `cublas`, `cublasLt`). Without it the `cuda` module compiles to nothing and every GPU path refuses or falls back to CPU. |
| `nccl` | Implies `cuda`; real NCCL collectives via `cudarc/nccl-02030`. Needs `libnccl.so` at link and run time. |
| `interop` | safetensors / Hugging Face Hub (`hf-hub`) / ONNX (`prost`) bridges. When off, `src/interop_stubs.rs` provides the same symbols (`nsl_safetensors_load`, `nsl_hf_load`, `nsl_trace_*`, ...) so the codegen's declarations still link; the stubs abort with a message if called. |
| `onnx-rt-op` | ONNX Runtime custom-op registration (`src/onnx_rt_op/`), vendored against ORT 1.22. |
| `test-hooks` | Narrow test-only inspectors, e.g. `test_tape_ops_len` / `test_drain_tape_and_params` in `src/autodiff/mod.rs`. |
| `bench-internal`, `strict-matmul`, `csha_cycle19_probe` | Bench-only surfaces; force `CUBLAS_PEDANTIC_MATH` at compile time; a research probe FFI variant. All off by default. |

`nsl-cli` depends on the runtime with `features = ["interop"]`, so the `nsl`
binary always has the bridges; a program compiled by `nsl build` links the
same library.

## The tensor handle model

Everything a program calls a "tensor" is an `i64` that is really a
`*mut NslTensor` (`src/tensor/mod.rs`). The struct is `#[repr(C)]`:

```
magic: u32          // MUST be first: 0x4E534C54 ("NSLT") live, 0x0000DEAD freed
data: *mut c_void   // opaque: CPU f64 or GPU f32 by default
shape: *mut i64
strides: *mut i64
ndim: i64
len: i64
refcount: AtomicI64
device: u8          // 0 = CPU, 1+ = CUDA device ordinal + 1
dtype: u16          // DTYPE_* tag; 256+ = custom
owns_data: u8       // 1 = heap-owned, 0 = borrowed / mmap / view / DLPack import
data_owner: i64     // non-zero: this is a view; the owner's refcount is held
slab_managed: u8    // data is an offset into the GPU slab; never freed alone
tape_id: i64        // assigned on first tape record, 0 = unassigned
```

`NSL_TENSOR_DATA_OFFSET` is `offset_of!(NslTensor, data)`. The codegen reads
`data` straight out of the struct at that offset in generated code, which is
why the field order is ABI and why `nsl-codegen` and `nsl-runtime` must be
built from the same revision.

**Construction.** `NslTensor::new(...)` is the only sanctioned constructor:
it sets `magic = TENSOR_MAGIC` and `refcount = 1`. `NslTensor::publish(Box)`
turns the box into the `i64` handle, records the allocation in the peak/stat
counters (`crate::math::track_alloc`), and registers it with the current
scope (`scope_track`, backed by the `TENSOR_SCOPE` thread-local).

**The magic word (PR #584).** `NslTensor::from_ptr(i64) -> &'static mut
NslTensor` and `from_ptr_ref` are the funnel every `extern "C"` tensor op
goes through. Both call `check_handle`, which in release builds too does:
null check; plausibility check (address below the first page or not
8-aligned cannot be a tensor and reading it would fault); then one load of
the first `u32` compared against `TENSOR_MAGIC`. Failure goes to the
`#[cold]` `bad_handle`, which names the handle and which of the four
failures it was (`null handle`, implausible address, `use-after-free` when
the word is `TENSOR_FREED`, or `not a tensor (magic 0x...)`), prints a Rust
backtrace when `RUST_BACKTRACE` is set, and **aborts**. `nsl_tensor_free`
writes `TENSOR_FREED` into the word before releasing the box, which is what
makes the use-after-free diagnosis possible. The gate is
`crates/nsl-runtime/tests/tensor_handle_magic_gate.rs`: each scenario re-execs the test binary
with `NSL_HANDLE_GATE_SCENARIO` set (`null | poisoned | garbage | freed |
valid`) and asserts on the child's exit status and stderr, because an abort
cannot be observed with `#[should_panic]`.

**`&'static mut` is known soundness debt.** `from_ptr` hands out a
`&'static mut NslTensor` from a raw pointer with no lifetime tied to
anything. Two calls on the same handle in one function produce aliasing
`&mut` references, and nothing stops a reference from outliving the
`nsl_tensor_free` that poisons the struct. The magic check catches stale
handles at the *next* entry point; it does not make the reference itself
sound. This is tracked as roadmap item C2. When touching a call site, prefer
`from_ptr_ref` for read-only access and keep the reference's scope as short
as the op allows; do not introduce new long-lived `&'static mut` borrows.
Concretely: never hold the reference across a call that takes the same
handle — the callee's own `from_ptr` invalidates yours under Stacked
Borrows (Miri reports it as "trying to retag … but that tag does not exist
in the borrow stack"); re-derive with `from_ptr` / `from_ptr_ref` after the
call instead. `scripts/miri-cpu-tensor.sh` is the check (see *Tests and
gates*).

**Dtype tags.** `src/tensor/mod.rs` declares the `u16` wire tags (roadmap
A3 moves the declaration into `nsl-abi`, with the runtime re-exporting
every name at this path, so readers are unaffected): `DTYPE_F64 = 0`, `DTYPE_F32 = 1`, `DTYPE_FP16 = 2`,
`DTYPE_BF16 = 3`, `DTYPE_INT8 = 4`, `DTYPE_FP8E4M3 = 5`, `DTYPE_FP8E5M2 = 6`,
`DTYPE_U16_TOKEN = 7`, `DTYPE_U16_SEGMENT = 8`, `DTYPE_I32 = 9`,
`DTYPE_INT8_BLOCKWISE = 10` (the blockwise-quantized int8 buffer: values
padded to 4 bytes plus one f32 scale per 64-value block, sized by
`data_byte_size` through `int8_blockwise_byte_size`; it had been tagged
`DTYPE_INT8` and freed as `len` bytes until Miri caught the mismatched
layout), custom dtypes from `DTYPE_CUSTOM_START = 256` (registered once at init through the
`STAGING_REGISTRY` / `CUSTOM_DTYPE_REGISTRY` pair). The C API's
`NslTensorDesc.dtype` uses the same space verbatim; the `dtype_abi_lock` unit
test in the same file fails if any value moves. Add new tags at the next free
slot; never reuse one.

**The opaque default.** `data` is "CPU f64 or GPU f32": a tensor created on
the host with no dtype is `DTYPE_F64`, and `nsl_tensor_to_device` produces a
`DTYPE_F32` device buffer. Typed accessors (`data_f64`, `data_f32`,
`data_f16_bits`, ...) assert the dtype they read. Kernels that need bf16 or
fp8 storage live behind explicit modes (`src/sr_bf16.rs`, `src/fp8.rs`,
`src/tensor/precision_cast.rs`) rather than changing the default.

**Strides and contiguity (PR #585).** `compute_strides` produces row-major
strides; `is_contiguous` compares the stored strides against that expectation
for every rank ≥ 1 (rank 0 is trivially contiguous). Until #585 rank 1 was
short-circuited to `true`, so a stride-0 `[N]` view from `nsl_tensor_expand`
flat-copied one element plus N-1 words of neighbouring heap; two call sites
had open-coded the missing check and the predicate itself was the fix.
`strides_are_row_major` is the allocation-free twin used on hot paths, and
`nsl_tensor_contiguous` materialises a copy only when needed. Views hold a
refcount on `data_owner`; freeing a view decrements the owner, never frees
the buffer.

**Lifetime.** Handles are refcounted (`nsl_tensor_retain`,
`nsl_tensor_free`). `nsl_tensor_free_transient` is the codegen's "this
intermediate is dead" hint; while the tape is recording it is deferred into
`Tape::deferred_transients` and released at `nsl_tape_stop`, because tape
nodes hold bare pointers as gradient-map keys and an address reused mid-step
would alias two nodes.

## Memory

**Host allocation** (`src/memory.rs`). `nsl_alloc` / `nsl_free` are the
program-facing malloc/free. `nsl_free` reconstructs the `Layout` from the
thread-local `ALLOC_REGISTRY` (ptr → size); that map is thread-affine for
correctness, which is why the inventory doc classes it RUNTIME-OK rather than
MIGRATE. `checked_alloc*` are the crate-internal helpers that abort with a
message on allocation failure. `memory::peak` tracks a CPU high-water mark
(`nsl_cpu_peak_bytes`); `memory::stats` are `#[cfg(test)]` counters the fuzz
harness (`src/fuzz.rs`) balances after every random op sequence.

**Slabs** (`src/slab.rs`). `nsl_slab_alloc` / `nsl_slab_offset` carve a
host slab; `nsl_gpu_slab_init` / `nsl_gpu_slab_destroy` reserve one device
region that parameter tensors are placed into with `slab_managed = 1`, so
they are freed once at exit rather than individually.

**The CUDA caching allocator** (`src/cuda/caching_allocator.rs`). Device
memory never goes back to the driver on `free`; blocks are returned to
`CACHING_ALLOCATOR` and reused. Every allocation carries `AllocationMetadata`
built by `current_alloc_metadata`: a `SurfaceTag` (`Other`, `Weights`,
`OptimM`, `OptimV`, `MPartial`, `Grads`, `Activations`, `AttnWorkspace`,
set through the RAII `SurfaceGuard` / `set_alloc_surface`), the (op, tensor)
identity from `set_alloc_identity`, and an `AllocationLifetime`. The
`CURRENT_POOL` selector with its `PoolGuard` splits persistent from transient
pools. `NSL_GPU_MEM_LIMIT` caps the reservation; `NSL_ASYNC_ALLOC=1` switches
to stream-ordered `cuMemAllocAsync`; `NSL_DEBUG_MEM_TRACE=1` prints every
block handout. `NSL_MEMSTATS=1` registers an `atexit` handler in
`cuda::inner::state()` that prints `print_memory_summary` — per-surface peak
bytes (`surface_table_string`), allocated-block summaries, and the external
(non-allocator) share at peak. `nsl_gpu_mem_*` accessors in
`src/tensor/mod.rs` expose the same numbers programmatically so gates do not
scrape stderr.

**Pinned host memory.** `cuda::inner::try_alloc_pinned` allocates page-locked
staging buffers for the optimizer-state offload path and records them in
`PINNED_HOST_SET`; `is_pinned` lets the tensor free path route those buffers
to the driver free instead of `dealloc`.

**The transient arena** (`src/transient_arena.rs`). The runtime half of the
placed transient arena (`--transient-arena`): the compiler admits only
statically-sized, non-escaping backward temporaries and pins each to a fixed
slot so it gets the same device address every step (the precondition for CUDA
graph replay). Slots are bracketed by `REDZONE` (256) bytes of `POISON`
(`0xA5`) that `nsl_arena_check` re-reads. The `PIN` / `PLACED_AT`
thread-locals are the single-shot channel that steers the *next* allocation,
because the `extern "C"` allocation signature has no out-parameter; a bind
whose size differs from the plan aborts rather than silently falling back to
the heap. `scripts/arena-parity.sh` is the byte-identity certification.

**Fatal exits** (`src/fatal.rs`). A condition the runtime cannot continue
from goes through `fatal::die(kind, msg)`: the message is printed verbatim,
then a `[nsl] fatal: <kind>, exiting with code <n>` line, stderr is flushed,
and the process exits with the kind's code. The codes are a contract with
whatever supervises a compiled program — they tell the conditions apart from
each other and from a crash (a panic's 101, SIGABRT's 134) without parsing
stderr, and `fatal::tests` pins them:

| `Fatal` | Exit code | When |
|---|---|---|
| `GpuOom` | **12** | the allocator could not satisfy a request after draining the pool and retrying |
| `CudaDriver` | **13** | a driver call failed for a reason other than OOM: `cuMemAlloc` (e.g. `CUDA_ERROR_ILLEGAL_ADDRESS` after a faulting kernel), `cuMemcpyHtoD` |
| `CudaAsync` | **14** | the `cuCtxSynchronize` that `--cuda-sync` inserts after a kernel or cuBLAS call reported an asynchronous device error |
| `Cublas` | **15** | a cuBLAS call failed on an in-place operation (the fused wgrad accumulate), where no partial result is safe to continue from |
| `CudaNotCompiled` | **16** | a device tensor reached a tensor op in a runtime built without the `cuda` feature — the `#[cfg(not(feature = "cuda"))]` arm of every GPU-capable op (`fatal::cuda_not_compiled`) and the cast paths' "compiled without the `cuda` feature" checks |
| `UnsupportedDtype` | **17** | a tensor op was asked to work on a dtype it does not implement (the cast family in `tensor/precision_cast.rs`, the scalar readers in `tensor/mod.rs`): a compiler/runtime contract violation, not a user error |

**GPU OOM** is the first of these: the allocator's failure path builds
`oom_diagnostic` (the request, the current `OOM_CONTEXT` description set by
`set_oom_context`, the pool breakdown, and `oom_contention_line` when the
driver's free/total says another process holds the card) and calls
`oom_fatal`, which is `die(Fatal::GpuOom, …)`.

## Autodiff tape

`src/autodiff/mod.rs` owns the tape; `src/autodiff/backward.rs` owns the
adjoint rules; `src/autodiff/grad_utils.rs` holds broadcasting and reduction
helpers.

**Recording.** `TapeOp` is an enum with one variant per differentiable
primitive (`Add`, `Mul`, `MatMul`, `Fp8MatMul`, `SumReduce`, `Gather`,
`Slice`, `Reshape`, `MaxPool2d`, `FlashAttention`, `Checkpoint`, ...). Each
variant stores the operand handles as **`i64` identities, never dereferenced
during backward**, and the shapes it needs as `TapeShape` — a
`SmallVec<[i64; 4]>`, so a ≤4-D shape costs no heap allocation (PR #572,
roadmap C4). The thread-local `TAPE: RefCell<Tape>` holds `ops`, the
`param_set` handed to `nsl_tape_start`, `recording`, `pause_depth`,
`next_id`, and `deferred_transients`. `maybe_record(op)` appends only when
`is_recording()` (recording on, pause depth zero, `TRAINING_MODE` set) and
first rewrites every identity field through `Tape::get_or_assign_id`, which
stamps a monotonic `tape_id` into the tensor. Identity is therefore decoupled
from the address, and an intermediate can be freed after recording without
its node becoming ambiguous.

The program-facing FFIs are `nsl_tape_start(param_list)`, `nsl_tape_stop`,
`nsl_tape_pause` / `nsl_tape_resume` (the `@no_grad` region; `TapePause` is
the RAII form used inside the runtime), and `nsl_tape_backward` /
`nsl_tape_backward_train(loss, param_list)`.

**Backward.** `run_backward_core_strict(ops, loss, params, strict)` takes the
ops *by value*, `ops.reverse()`s them, and walks once, accumulating into a
`grad_map: HashMap<i64, i64>` keyed by tape id. Every rule that needs a saved
operand reads it from the node's `saved_*` field, not from the live tensor.
`strict = true` (the train-block entry) aborts if every parameter fell back
to `zeros_like`, because a disconnected graph "training" on weight decay
alone was a real silent failure. The elementwise alloc budget
(`crates/nsl-runtime/tests/elementwise_alloc_budget.rs`) pins that a CPU elementwise op makes
exactly four heap allocations whether or not the tape records — the C4
number, down from thirteen.

**Per-call contexts.** For the host-facing C API the tape is not left in the
thread-local: `nsl_model_forward_grad` (`src/grad_context.rs`) records, then
moves the ops out into a heap `GradContext` (magic `NSL_GRAD_CONTEXT_MAGIC`
`0x4E534C47` "NSLG", poisoned to `NSL_GRAD_CONTEXT_FREED` on destroy) that
`nsl_model_backward` consumes and `nsl_grad_context_destroy` frees. A
`GradContext` is `Send`, so the backward can run on another thread
(`crates/nsl-runtime/tests/cross_thread_backward.rs`); the RAII guard inside `forward_grad`
clears the tape on every exit path including panic
(`crates/nsl-runtime/tests/raii_guard_clears_tape_on_panic.rs`); and
`crates/nsl-runtime/tests/backward_does_not_consult_live_tape.rs` pins that backward never reads
the live `TAPE`.

**How source-AD bypasses the tape.** Under `--source-ad` the codegen
(`crates/nsl-codegen/src/source_ad.rs`, `wengert.rs`, `wengert_lower.rs`)
derives the backward at compile time and emits it as ordinary Cranelift code.
The runtime's role shrinks to: `src/backward_context.rs`
(`nsl_backward_ctx_new/save/load/free`, a per-grad-block slot table for
saved-for-backward tensors) and `src/tensor/ad_ops.rs` (the primitive
backward steps the lowering calls — comparisons, ternary select, scatter-add,
log-softmax, inverse RoPE). `nsl_tape_start` is never emitted on that path,
`is_recording()` stays false, and `maybe_record` is a no-op. The `[tape-ad]`
stderr marker (registered in `crates/nsl-cli/src/exec_markers.rs`) is how
tests prove which path a run took.

## CUDA backend

All of it lives under `src/cuda/` behind `feature = "cuda"`; `src/cpu.rs`
is the host fallback and `src/gpu_backend.rs` is the (compile-time) backend
trait.

**The process-global driver state.** `cuda::inner::CudaState` holds the
`CUdevice`, the primary `CUcontext`, a `module_cache: HashMap<u64, CUmodule>`
keyed by an FNV-1a hash of the PTX text, and a `func_cache` keyed by
(module hash, entry-name hash). It is a `static CUDA_STATE:
OnceLock<Mutex<CudaState>>`, initialised on first use by `state()`:
`cuInit`, `cuDeviceGet(select_device_ordinal())`,
`cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`. `select_device_ordinal`
honours `NSL_CUDA_DEVICE` and otherwise stripes by `NSL_LOCAL_RANK` only
under the SPMD spawner protocol (`NSL_TP_SHM_PATH` set). There is exactly one
device, one context, and one cuBLAS handle (`cublas_handle()`, lazily
created) per process. Multi-GPU today means one process per device; a single
process driving two devices is blocked on this singleton — roadmap item A4.

**Streams and workspaces** are per-thread cells in `src/cuda/mod.rs`:
`COMPUTE_STREAM`, `TRANSFER_STREAM` (with `transfer_stream_synchronize` and
event-based cross-stream waits such as `compute_stream_wait_event`), plus
persistent kernel workspaces (`WS`, `SR_WS`, `CE_SCRATCH`, `MUON_STATS_BUF`).
`NSL_CUDA_SYNC=1` synchronises after every launch for bisecting async bugs.

**Kernel launch path.** PTX is embedded as NUL-terminated `&str` constants:
`src/cuda/kernels.rs` (elementwise, `.target sm_70`),
`src/cuda/fused_kernels.rs` (embedding, bias, layernorm, rmsnorm, `sm_80`),
`fused_ce_kernels.rs` / `fused_kl_ce_kernels.rs` (fused linear-CE and KL-CE
losses), `precision_cast_kernels.rs`, `strided_copy.rs`, `tier_b1_prepass.rs`,
and `kernels_hopper.rs` (`sm_90a`, wgmma/TMA FlashAttention-3). A launch is
`load_module_once(ptx)` → `get_function(module, name)` →
`launch_function_raw(...)`; the caches mean a module is JIT-compiled by the
driver once per process. Hand-written PTX is **frozen**: `scripts/hand-ptx-freeze.sh`
(roadmap A2) fails CI if a new hand-written kernel appears, because the
intended path for new kernels is the compiler's `KernelIR` →
`backend_ptx.rs` lowering in `nsl-codegen`.

**CUDA graphs** (`src/cuda/graph_capture.rs`). Under `--cuda-graphs` the
codegen brackets each Wengert region with
`nsl_cuda_graph_region_begin/end(id)`. The per-region state machine
(`ACTIVE`, `REGIONS`, `OCCURRENCE`, `NESTED_SKIP`) records the pseudo-op
sequence eagerly for two steps, captures on the third with
`cuStreamBeginCapture_v2`, then replays with one `cuGraphLaunch` per region,
verifying every issued op against the capture and eager-repairing on any
mismatch. Host-side bookkeeping runs identically in every mode; only GPU
issuance is diverted — that is the correctness invariant the module's header
states. Deferred frees inside a region are queued until the region's work is
on the stream.

**Matmul configuration and the bf16 GEMM path.** `src/matmul_config.rs`
holds the arithmetic mode set once by `nsl_set_matmul_config`, which the
codegen emits before the first user statement from compile options — so the
mode is part of the execution fingerprint and a resume can refuse a silent
switch. `NSL_MATMUL_BF16`, `NSL_MATMUL_TF32`, `NSL_MATMUL_PEDANTIC` and
friends survive as a deprecated fallback that warns once. In bf16 mode,
`src/cuda/bf16_cast_cache.rs` keeps one persistent bf16 image per parameter
and re-casts only after the fused optimizer step calls `note_param_stepped`;
it registers only tensors that own their storage, so arena views and DLPack
imports can never serve a stale image. `src/cuda/lt_matmul.rs` is the
cublasLt issue path (`NSL_MATMUL_BF16_LT=1`,
`NSL_MATMUL_BF16_LT_WORKSPACE_MIB`): it asks the heuristic for several
candidates, times each once per shape, caches the winner, and falls back to
`cublasGemmEx` on any decline. The doc comment in `matmul_config.rs` is
explicit that matching the configuration is recorded as the *inputs* to
kernel selection, not a bit-identity promise.

**Determinism** (`src/deterministic_ops.rs`). `--deterministic` sets
`DETERMINISTIC_MODE` and `RNG_SEED` (`nsl_rng_seed`; `RNG_SEED_DEFAULT = 42`),
swaps reductions and scatter-add for sequential/output-owned PTX variants
(`nsl_det_global_sum_f32`, `nsl_det_scatter_add_f32`), and disables the
timed cublasLt plan selection — which is what makes the Lt path
reproducible.

## Training state

**`.nslm` and the `.optim` sidecar** (`src/checkpoint.rs`). `nsl_model_save`
writes magic `NSLM`, version 1, a JSON header of `{name, shape, dtype,
offset, nbytes}` entries, and 64-byte-aligned raw data; `nsl_model_load`
hard-aborts on an unknown version. `nsl_train_checkpoint_save` writes θ
through `nsl_model_save` plus `<path>.optim` with magic `OPTIM_MAGIC` =
`NSLO`: a header `{"step_count", "model_sig", "resume": {...}, "params":
[...]}` followed by all `m` entries then all `v` entries. Both files go to
`<path>.tmp` and are `rename`d, and the sidecar carries a signature of the
`.nslm` it was written beside so a crash between the two renames is detected
at load. The `resume` block is what makes a resume a continuation rather than
a warm start:

- loader position (`loader_epoch`, `loader_slot`) and `loader_id`, the corpus
  and geometry fingerprint from `nsl_dataloader_identity`; a different corpus
  refuses;
- the RNG streams from `src/rng_state.rs` (`RngSnapshot`: the ChaCha word
  position of the sampling RNG, the GPU dropout counter, the SR-BF16 dither
  counter) and the `--seed` scalar; a different seed refuses;
- the execution fingerprint (`src/exec_fingerprint.rs`, `exec_fingerprint()`,
  PR #519): `arithmetic_diff` (source-AD vs tape, `--deterministic`, dtype,
  fusion flags, matmul mode) refuses; `placement_diff` (`--transient-arena`,
  `--cuda-graphs`, `--checkpoint-blocks`, `--optim-state-offload`) warns;
- the resolved train config (`src/train_config_record.rs`, installed by the
  codegen through `nsl_set_train_config_record`): `MOMENT_KEYS` drift
  (optimizer, accum, betas, eps, wd, ...) aborts; `TRAJECTORY_KEYS` drift
  (lr, schedule, clip) refuses unless `NSL_RESUME_ALLOW_TRAJECTORY_DRIFT=1`.

`nsl_train_checkpoint_load` restores all of it and `nsl_train_resume_epoch`
tells the emitted epoch loop where to continue. `epochs` is the run total; a
budget already spent refuses rather than exiting 0 having done nothing.

**ZeRO** (`src/zero.rs`). `ZeROStage` 1/2/3; `nsl_zero_init(stage,
world_size)`, `nsl_zero_partition[_bytes]` (`partition_params_balanced`),
`nsl_zero_owns_param`, `nsl_zero_reduce_grads`, `nsl_zero_step`,
`nsl_zero_sync_params`. Stage 3 adds a per-parameter residency table
(`ParameterResidency`, `nsl_zero3_note_param`, `nsl_zero3_residency`) and the
elementwise-sharded moments (`nsl_zero3_alloc_elem_moment`). Collectives are
real NCCL under `nccl` and shared-memory simulation otherwise.

**Fused optimizer step** (`src/fase_step.rs`). `nsl_fase_fused_adamw_step`
and the `_multi` / `_multi_idx` batched forms perform the whole AdamW update
in one launch per parameter, bit-exact with the interpreted `UpdateProgram`
the codegen (`stmt_fase.rs`) otherwise emits; the counters
`nsl_fase_fused_step_count` etc. are what the `[fase-fused]` gates read.

**The parameter plan** (`src/param_plan.rs`). Three independent tables can
own a parameter's storage — `MIRRORS` (`src/weight_stream.rs`),
`ZERO3_TABLE` (`src/zero.rs`), `SRBF16_TABLE` (`src/sr_bf16.rs`) — and the
codegen decides per parameter which one should. `nsl_param_plan_declare`
records that decision (`PLAN_STREAMED`, `PLAN_BF16_SR`, `PLAN_SHARDED`,
`PLAN_ELEMENTWISE`) and `nsl_param_plan_verify` checks it against where the
parameter actually landed, turning a silent-wrong-numerics class into an
abort with the `[param-plan]` marker.

**Standalone weights** (`src/weight_provider.rs`). A `.nslweights` blob
(magic `NSLW`) embedded in the binary (`nsl_standalone_init_embedded`) or
mmapped beside it (`nsl_standalone_init_sidecar`); when a provider is set,
`nsl_model_load` reads from it and skips file I/O. The same module hosts the
`nsl_standalone_arg_*` CLI-argument accessors a standalone binary uses.

## Data and tokenization

**DataLoader** (`src/dataloader.rs`). `nsl_dataloader_create` builds a
multi-threaded loader with a reorder buffer keyed by batch id, so
`nsl_dataloader_next_batch` yields batches in a deterministic sequential order
regardless of worker timing. The resumable surface is `nsl_dataloader_epoch`,
`nsl_dataloader_slot`, `nsl_dataloader_identity`, and
`nsl_dataloader_resume_to(epoch, slot)`. Under data parallelism every rank
derives the same global permutation from `DEFAULT_DP_SHUFFLE_SEED ^ epoch`
so strided per-rank shards stay disjoint and complete.

**Corpus format.** `src/data_source.rs` provides `nsl_load_jsonl`,
`nsl_load_csv`, and `nsl_load_mmap(path, dtype)`, which memory-maps a raw
little-endian file (`memmap2`) into a borrowed tensor (`owns_data = 0`).
Token corpora are `DTYPE_U16_TOKEN` streams; the loader's fast path
(`supports_flat_value_dtype`) is exactly f64, f32 and u16 tokens. `src/data/`
holds the sharded/GDS/multimodal pipeline pieces (`shards.rs`, `gds.rs`,
`pipeline.rs`).

**Tokenizers.** `src/tokenizer.rs` wraps the Hugging Face `tokenizers` crate
(byte-level and BPE) behind boxed `TokenizerKind` handles.
`src/tokenizer_bpe.rs` is the two-stage BPE *trainer* behind `nsl tokenize`
(merges may cross the pre-tokenizer boundary so indentation compresses).
`src/tokenizer_fast.rs` is the byte-domain encoder specialised to the single
configuration NSL's shipped tokenizers use; `tests/` in `nsl-cli`
(`tokbench_reference_encoder_gate.rs`) pins that it matches the reference
encoder token for token.

**Calibration data** (`src/calibration_data.rs`). `load(path)` /
`peek_shape(path)` read either the NSL-native `.bin` (magic `NSLB`) or a
safetensors archive with a `calibration` tensor, for the AWQ/GPTQ pipelines
(`src/awq.rs`, `src/gptq.rs`, `src/quantize.rs`).

## Serving and interop

**Serving** (`src/serving/`). `scheduler.rs`, `request.rs`, `preemption.rs`
and `ragged.rs` implement continuous batching; `ffi.rs` exposes
`nsl_serve_init`, `nsl_serve_enqueue`, `nsl_serve_step`,
`nsl_serve_record_token`, `nsl_serve_drain_completed`, `nsl_serve_has_work`.
`src/paged_kv/` (`block_alloc.rs`, `page_table.rs`, `cow.rs`, `manager.rs`)
is the paged KV cache with copy-on-write block sharing; `src/speculative/`
(`draft.rs`, `verify.rs`, `tree.rs`, `lookahead.rs`, `ffi.rs`) is speculative
decoding; `src/kv_compress/`, `src/disaggregated/`, `src/elastic/` are the
remaining `inference` facade members.

**The stable C API** (`src/c_api/mod.rs`). This is the surface a C/C++/Python
host uses against a shared library built by `nsl build --shared`:

- `NslTensorDesc` (`#[repr(C)]`; data pointer, shape, dtype in the canonical
  tag space, device) and `NslModel`;
- `nsl_model_create` / `nsl_model_create_with_lib` (dlopens the model's own
  `.so`, enumerates `nsl_get_num_exports` / `nsl_get_export_name`, and builds
  the read-only `ExportRegistry` in `src/c_api/exports.rs`),
  `nsl_model_call` / `nsl_model_call_into` / `nsl_model_call_alloc` /
  `nsl_model_call_dlpack` (named-export dispatch; the `Into`/`Alloc`
  ownership mode is armed for the synchronous span of one call through the
  `DISPATCH_MODE` thread-local and `nsl_dispatch_ownership_arm`),
  `nsl_model_forward[_dlpack]`, `nsl_model_lookup_function`,
  `nsl_model_get_export_signature`, `nsl_model_get_weight`,
  `nsl_model_destroy`;
- the autograd pair `nsl_model_forward_grad` / `nsl_model_backward` from
  `src/grad_context.rs`;
- `nsl_abi_version`, and the error slot: `nsl_get_last_error` returns the
  thread's last message (a `LAST_ERROR` thread-local `CString`),
  `nsl_clear_error` resets it, `nsl_set_error_cstr` lets generated wrappers
  fill it.

**DLPack** (`src/dlpack.rs`). `nsl_dlpack_export` / `nsl_dlpack_import` /
`nsl_dlpack_free` implement DLPack v0.8 zero-copy in both directions; an
imported tensor has `owns_data = 0` and its buffer dies in the foreign
allocator. Unsupported dtypes are refused, not coerced
(`crates/nsl-runtime/tests/dlpack_unsupported_dtype_refusal.rs`).

**Interop bridges** (`feature = "interop"`). `src/safetensors_io.rs`
(`nsl_safetensors_load` / `nsl_safetensors_save`), `src/huggingface.rs`
(`nsl_hf_load`, via `hf-hub`'s `ureq` backend — note the rustls root-store
caveat in `Cargo.toml`), `src/onnx.rs` + `src/onnx_proto.rs`
(`nsl_onnx_export` from a recorded `src/trace.rs` trace),
`src/weight_map.rs` for name mapping. `src/onnx_rt_op/` (feature
`onnx-rt-op`) exports `RegisterCustomOps` for ONNX Runtime.

**The Python bridge** lives outside the crate in `python/nslpy/`:
`_core.py` loads the shared library with `ctypes.CDLL`, binds the
`nsl_model_*` and `nsl_get_last_error` prototypes, and wraps them in
`NslModel`; `_bridge.py` mirrors `NslTensorDesc` byte for byte and implements
the DLPack exchange with its defensive-copy guard; `autograd.py` wraps the
grad-context pair as a `torch.autograd.Function`; `hub.py` and `onnxrt.py`
are the Hub and ORT conveniences. `python/tests/` is run by the
`python-interop` CI job. When `@export` functions exist, `nsl build --shared`
also writes a C header next to the library
(`nsl_codegen::c_header::emit`, called from
`crates/nsl-cli/src/commands/build/shared_lib.rs`).

## Observability

**Structured events** (`src/events.rs`). `NSL_EVENTS=<path>` appends one
JSON object per line: `{"v":1,"seq":N,"rank":R,"kind":"...","step":S|null,
"fields":{...}}`. `EVENTS_VERSION` is bumped only for envelope changes;
per-kind fields may grow and consumers must ignore unknown ones. `seq` is
per-process (multi-rank consumers key on `(rank, seq)`), writes are one
`write(2)` per line on an `O_APPEND` fd, and emission is best-effort: the
first failure prints one `[nsl] warning:` line and disables the sink. Call
sites use `events::emit(kind, step, &[(name, value)])` and build the JSON
and the stderr line from one counter snapshot so the two cannot disagree.
The registry of kinds and fields is `EVENT_SCHEMAS` in
`crates/nsl-cli/src/exec_markers.rs`, next to the marker registry. The
writer is the compiled program, one process per rank: the `nsl` CLI passes
`NSL_EVENTS` through to the program it spawns and opts its own process out
(`events::opt_out_this_process`, first thing in `main`), so its
compile-time `nsl_log!` lines never land in the program's file with a
second `seq` sequence.

**Logging** (`src/log.rs`, roadmap C3). Diagnostic lines go through
`nsl_log!(LEVEL, "target", "…")`, a `tracing` event whose target names the
subsystem and whose message is the line. The crate's own subscriber
(`NslSubscriber`, installed on the first line by `ensure_installed`) writes
the message plus one newline to stderr and nothing else, so the marker
lines the CLI gates compare byte for byte are unchanged from the
`eprintln!` they replaced; when `NSL_EVENTS` is on, the same line is
appended to the stream as a `log` event (`level`, `target`, `message`), the
one kind whose `message` is its own marker (`LINE_IS_THE_MARKER` in
`exec_markers.rs`). A host that installed a global `tracing` subscriber
first keeps it and receives the runtime's lines as events. Every
diagnostic `eprintln!` in the crate is migrated: the bracketed-marker
family (`[zero3]`, `[cuda-graph]`, `[weight-stream]`, `[arena]`, …), the
`nsl: …` and `[nsl] …` families (target `nsl`; `ERROR` where the line
precedes an abort or exit, `WARN` where the entry point returns instead),
and the per-subsystem lines (`cfie`, `flash-attention` / `flash-bwd`,
`fused-linear-ce`, `cuda`, `tensor`, `huggingface`, …; a line that starts
with its own `[marker]` uses the marker as its target). Program output —
the `print` builtin, the tensor printer, the health JSON — stays on
`println!` because it is stdout, not a diagnostic. nsl-codegen is next.
The stderr path allocates nothing — the
message is formatted straight into the locked handle — so the
`nsl: out of memory` line in `memory.rs` still prints. A
new line in a migrated family uses the macro; a new family picks a target
and a level (ERROR before an abort or a lost result, WARN for degraded-but-
continuing, INFO for the rest) and keeps the text it would have printed.

**The stderr marker contract with nsl-cli.** Subsystems announce engagement
with a bracketed tag at the start of a stderr line — `[zero3]`,
`[cuda-graph]`, `[weight-stream]`, `[arena]`, `[sr-bf16]`,
`[fused-lce-gemm]`, `[nsl-gpu-launch-count]`, `[nsl-kernel-count]`,
`[muon-state]`, `[wgrad-accum]` and the general `[nsl]` prefix all originate
in this crate. The tags are an API: `EXEC_MARKERS` in
`crates/nsl-cli/src/exec_markers.rs` records each token, the files that emit
it (`emitted_by`), and what it means, and a gate fails if an emitter
disappears. Rename a tag only by changing both. Gates compare these lines
byte for byte, so formatting is part of the contract.

**Profilers.** `src/profiler/` (`nsl_profile_kernel_begin/end`, `CUevent`
timing under `cuda`, a host `NanoClock` otherwise) feeds `nsl run --monitor`
and `--profile`; `src/kernel_profiler.rs` and `src/profiling.rs` are the
older per-kernel and phase timers; `src/muon_prof.rs` (`NSL_MUON_PROF`) is
the Muon scope profiler; `src/tensor_trace.rs` / `src/trace_diff.rs` record
and diff tensor-op traces; `src/inspect/` is the `nsl debug` inspection
surface.

**Health.** `src/health/` (`collector.rs`, `ffi.rs`) accumulates loss EMA,
slope, gradient norm and NaN/Inf window counts through
`nsl_health_record_loss` and the `nsl_health_get_*` readers, which the CLI's
live monitor polls. `src/grad_integrity.rs` (`--grad-integrity`,
`NSL_GRAD_INTEGRITY=1`) verifies every optimizer step that each parameter
received a usable gradient and prints a worst-case snapshot at exit under the
`[grad-integrity]` marker.

## Invariants

Each of these is enforced by code or a test named here; if you change one,
change the gate.

- **No unwinding across the ABI.** Fatal paths abort or `exit`; recoverable
  ones return sentinels. `bad_handle`, `oom_fatal`, `src/assert.rs`, and
  `write_or_abort` in `checkpoint.rs` are the reference implementations.
- **Sentinel error returns on the C API.** `nsl_model_*` return `0`/`-1` and
  set `LAST_ERROR`; typed refusals rather than crashes
  (`crates/nsl-runtime/tests/double_backward_returns_typed_error.rs`,
  `crates/nsl-runtime/tests/model_call_returns_error_on_unknown_name.rs`).
- **Every tensor entry point checks the magic**, in release builds
  (`crates/nsl-runtime/tests/tensor_handle_magic_gate.rs`); the free path poisons it.
  `GradContext` carries its own magic (`crates/nsl-runtime/tests/grad_context_magic_validation.rs`).
- **Dtype tags never move** (`dtype_abi_lock` in `src/tensor/mod.rs`).
- **`NslTensor` field order is ABI**; `NSL_TENSOR_DATA_OFFSET` is the only
  way the codegen learns where `data` is.
- **Tape ordering.** Ops are appended in forward order and consumed by one
  reversed walk over a moved-out `Vec`; backward never reads the live tape
  (`crates/nsl-runtime/tests/backward_does_not_consult_live_tape.rs`,
  `crates/nsl-runtime/tests/run_backward_core_matches_tape_backward.rs`). Tape nodes hold
  identities, not dereferenceable pointers.
- **Transient frees are deferred while recording** (`Tape::deferred_transients`),
  so an address cannot be recycled under a live node.
- **Checkpoints are atomic**: `.tmp` + `rename`, and the sidecar's
  `model_sig` refuses a mismatched pair.
- **Resume refuses drift**: corpus/geometry, seed, arithmetic fingerprint,
  moment-meaning config; trajectory drift needs an explicit acknowledgement
  (`NSL_RESUME_ALLOW_TRAJECTORY_DRIFT=1`).
- **Byte-identical stderr for gates**: marker lines are part of the CLI
  contract; the JSON events file is the machine-readable twin and must not
  replace them.
- **`NSL_*` variables are registered** in `crates/nsl-env/src/registry.rs`
  or the workspace `registry_agreement` gate fails.
- **Every `thread_local!` is inventoried** in
  `docs/architecture/compiler-state.md` (`crates/nsl-runtime/tests/thread_local_inventory_drift.rs`).
- **Every `#[ignore]` carries a reason** and is in
  `ci/gpu-cert-manifest.tsv` (`scripts/gpu-cert.sh --check-reasons /
  --check-inventory`).
- **Hand-written PTX is frozen** (`scripts/hand-ptx-freeze.sh`).
- **CUDA graph replay is sound only because host bookkeeping runs eagerly
  in every mode**; never move an allocation or free behind a replay skip.
- **Cast-cache eligibility is ownership, not inference** (`owns_data != 0`,
  contiguous f32, on device) — see `src/cuda/bf16_cast_cache.rs`.

## Tests and gates

**Unit tests** are `#[cfg(test)]` modules inside `src/` — the dtype lock,
`caching_allocator` tests (run in the `cuda-feature` CI job even without a
device), the PTX assembly gate, and the `src/fuzz.rs` lifecycle fuzzer that
balances the `memory::stats` counters after random FFI sequences
(`cargo test -p nsl-runtime fuzz`).

**Integration tests** are the 71 files under `crates/nsl-runtime/tests/`.
They call the runtime through the rlib and fall into: handle and context
safety (`tensor_handle_magic_gate.rs`, `grad_context_*`,
`multi_context_in_flight.rs`, `forward_in_progress_returned_on_reentry.rs`);
tape semantics (`backward_does_not_consult_live_tape.rs`,
`cross_thread_backward.rs`, `raii_guard_clears_tape_on_panic.rs`,
`desc_tape_id_roundtrip.rs`, `elementwise_alloc_budget.rs`); the C API and
export registry (`export_registry_populates_eagerly.rs`,
`model_call_dispatches_to_named_export.rs`, `export_tuple_marshaling.rs`,
`dlpack_unsupported_dtype_refusal.rs`); matmul modes
(`matmul_bf16_lt.rs`, `matmul_bf16_cast_cache.rs`,
`matmul_cublas_equivalence.rs`, `matmul_tf32_mode.rs`); FP8, MoE, PCA and
Tier-B dispatch; health and profiler collectors; workspace drift gates
(`thread_local_inventory_drift.rs`, `cpu_only_build.rs`). Shared helpers are
in `crates/nsl-runtime/tests/common/`.

**GPU gates.** 27 of the 71 files contain `#[ignore = "requires CUDA GPU"]`
(or a variant naming the spawned child) arms; they run only with
`--features cuda -- --ignored`. `scripts/gpu-cert.sh` (roadmap item 19) is
the certification lane: `--list` inventories every ignored test through
`scripts/gpu-gate-inventory.awk` (classes such as `gpu`, `broken`,
`cpu-stub`), `--write-manifest` regenerates `ci/gpu-cert-manifest.tsv`
(~470 rows across the workspace), and the GPU-free checks
`--check-inventory`, `--check-reasons`, `--check-long-arms` run in the
`gpu-gate-inventory` CI job. `scripts/gpu-tier.sh` layers `smoke` / `certify`
tiers over it; `scripts/gpu-guard.sh` serialises device access.

**Benches.** `benches/tape.rs` (criterion, host-only): tape record/backward
on a small MLP, the elementwise record chain, the tiled f32 matmul, and
allocation churn. `cargo bench -p nsl-runtime`.

**Miri.** `scripts/miri-cpu-tensor.sh` runs the `tensor::tests` module
(the CPU path is pure Rust, so it interprets end to end) under Miri's
Stacked Borrows model; `--each` runs one process per test so one error
does not hide the rest. First run (2026-09-07, roadmap C2): 45 tests, 35
clean, 10 undefined behaviour — every one the same shape, in the tests
themselves: a `&'static mut` from `from_ptr` held across an op that
re-derives the same handle, then read again. No production op tripped
it, which is consistent with each op deriving its own reference and
using it within the call. Those ten tests now re-derive after the call.
The follow-up runs widened the filter to the whole `tensor::` namespace
and then (`--sweep`) to every module of the crate, one process each: 69
modules, 50 clean, 8 with reports, 6 that Miri cannot run (process
spawn, file-backed mmap, `atexit`; listed in the script). The production
findings, each fixed with the run that found it: `tensor_elementwise_op`
and `nsl_tensor_matmul` derived two `&mut` for their two inputs, so
`mul(y, y)` / `matmul(x, x)` aliased them (now shared references);
blockwise-int8 tensors were freed and cloned as `len` bytes while their
buffer is padded values plus per-block scales (now their own dtype tag);
`nsl_tensor_reshape` read its input through a reference it had held
across `new_view_i64` (now re-derived); the sparse value buffer was a
byte allocation read as `&[f64]` (now an f64 allocation); and the KV
transfer header was written to the socket as the struct's raw bytes,
padding included (now field by field); and the owned DLPack export held
a `&mut` to the tensor across `storage_is_nsl_owned`, which re-derived
the same handle (the export entry points now take shared references).
Everything else was test-side:
tests holding a handle across a call, an unaligned test buffer for
`ShmHeader`, stack tensors handed out through `&T` rather than `&mut T`.
Three `tensor::activation` / `flash_attention` / `context_parallel` tests
assert bit-exact float results and fail under Miri's deliberately
perturbed float operations; those are not findings.
`tensor::alias_tests` is the standing gate that came out of it: one
probe per multi-input CPU op, each passing the same handle for every
input (`cat([x, x])`, `x == x`, `where(x, x, x)`, `x += x`, the
activation backwards with `grad` and `x` aliased, …), which is exactly
the pattern that makes two `&mut` derivations from one handle observable.
Twelve of the sixteen ops it covers reported undefined behaviour on the
first run and now take shared references; run it with
`scripts/miri-cpu-tensor.sh --each tensor::alias_tests` after touching a
multi-input op.
`fase_step`, `host_profile` and the two seeded `fuzz` loops exceed the
per-module time cap and are skipped for time, not for a limitation of
Miri (the four small `fuzz` tests are clean). The script passes
`-Zmiri-ignore-leaks`: the tests leak shape lists and tensors on purpose
(110 allocations at exit), which is test hygiene, not the aliasing
question. Not in CI only because it needs a nightly toolchain: once the
crate is built for Miri the whole module interprets in about ten seconds
(`--each` pays a process start per test and takes a minute or so each).

## Where to add a new X

**A new `extern "C"` runtime function.**
1. Implement it in the module its subject belongs to as
   `#[unsafe(no_mangle)] pub extern "C" fn nsl_...`. Take `i64` handles and
   validate them with `NslTensor::from_ptr[_ref]`; report failure with a
   documented sentinel, never a panic; if it can allocate device memory, set
   the OOM context (`cuda::inner::set_oom_context`).
2. Declare it once as a row of the typed ABI table,
   `crates/nsl-abi/src/table.rs` — `[group] nsl_…(i64, …) -> i64 =
   module::path::nsl_…;` in the group its subsystem belongs to (`[interop]`
   after the path if it lives behind the `interop` feature). The codegen
   renders the row into its declaration and this crate renders it into a
   compile-time check (`src/abi_check.rs`): a row whose arity, slot types or
   path disagree with the implementation fails `cargo build -p nsl-runtime`
   naming the function.
3. Run `cargo test -p nsl-abi --test signature_agreement`: it parses the
   runtime's `extern "C"` items and cross-checks them against the table,
   reporting every drift at once.
4. If the function is part of the host-facing C API, add it to
   `src/c_api/mod.rs`, bind it in `python/nslpy/_core.py` (argtypes/restype),
   document it in `docs/abi/README.md`, and — if it is an `@export`-visible
   signature — make sure `nsl_codegen::c_header` renders it.
5. If the codegen only ever calls it when a feature is off, add a stub to
   `src/interop_stubs.rs` under the inverse `cfg`.

**A new tensor op with a backward rule (tape path).**
1. Forward in the right `src/tensor/*.rs` file (`arithmetic.rs`,
   `activation.rs`, `reduction.rs`, `shape_ops.rs`, ...); CPU kernel plus a
   `cuda` arm that dispatches through `src/cuda/mod.rs`.
2. Add a `TapeOp` variant in `src/autodiff/mod.rs` storing operand
   identities and `TapeShape`s (never a `Vec` for a shape); extend
   `assign_ids` so every identity field is rewritten; call
   `maybe_record(...)` at the end of the forward.
3. Add the adjoint arm in `run_backward_core_strict`
   (`src/autodiff/backward.rs`) using `accumulate_grad` and the
   `grad_utils` helpers; read saved operands from the node, not the live
   tensor.
4. If source-AD must also support it, add the `PrimalOp` and adjoint rule in
   `crates/nsl-codegen/src/wengert.rs` / `source_ad.rs` and any primitive it
   needs to `src/tensor/ad_ops.rs`.
5. Keep `crates/nsl-runtime/tests/elementwise_alloc_budget.rs` green (four allocations per
   elementwise op) and add a CPU/GPU parity test; a GPU arm needs a reasoned
   `#[ignore = "requires CUDA GPU"]` and a manifest regeneration.

**A new CUDA kernel.** Prefer the compiler path: build it as `KernelIR` in
`crates/nsl-codegen/src/kernel_ir.rs` and let `backend_ptx.rs` lower it.
A hand-written PTX constant in `src/cuda/*.rs` trips
`scripts/hand-ptx-freeze.sh` and needs an explicit freeze-list change. Either
way: load with `load_module_once` / `get_function`, launch with
`launch_function_raw` on `COMPUTE_STREAM`, give a `--deterministic` variant
in `src/deterministic_ops.rs` if it uses atomics, make it graph-capture-safe
(no readbacks inside a region), and add a `ptxas` syntax check to the
`cuda-feature` job's PTX assembly gate.

**A new `NSL_*` environment variable.**
1. Read it with `std::env::var("NSL_...")` (or a `const` the scanner in
   `crates/nsl-env/src/scan.rs` can see); `"1"`-only semantics are the
   convention for booleans.
2. Add a `var!(...)` row to `crates/nsl-env/src/registry.rs` with the right
   `Tier` (`Behavior` if it changes numerics or what executes, `Perf`,
   `Safety` for allow/force overrides, `Platform`, `Diagnostic`, `Test`) and
   `ReadAt` (`Compile`, `Runtime`, `Both`, `Test`). The
   `registry_agreement` gate in `crates/nsl-env/tests/` fails on a read that
   is not registered or a row that is no longer read.
3. Regenerate `docs/wiki/Environment-Variables.md` with
   `nsl env list --markdown`; the nsl-cli test compares the page with the
   registry.
4. A `Behavior`-tier variable that changes training arithmetic must also be
   folded into `exec_fingerprint()` or, better, replaced by a compile option
   the way `matmul_config.rs` did.

**A new checkpoint sidecar field.**
1. Extend the `resume` block written in `nsl_train_checkpoint_save`
   (`src/checkpoint.rs`) and the reader in `nsl_train_checkpoint_load`.
   Numbers go in as integers; anything with a decimal point rides inside a
   string, because the header's needle scanners are digit-only (see
   `train_config_record.rs`).
2. Decide the drift class on load — refuse (arithmetic / moment-meaning),
   refuse-unless-acknowledged (trajectory), or warn (placement) — and route
   it through `exec_fingerprint::diff` / `train_config_record::check_on_resume`
   rather than a bespoke comparison.
3. Keep version-1 and version-2 sidecars loadable with a warning; bump the
   sidecar version only if the layout changes.
4. Add a gate in `crates/nsl-cli/tests/` alongside
   `train_checkpoint_gate.rs`, `train_config_resume_gate.rs`,
   `exec_fingerprint_resume_gate.rs`, and update
   `docs/summaries/04-training-and-autodiff.md`'s restore table.

**A new stderr marker or event kind.** Emit the tag from the runtime, register
it in `EXEC_MARKERS` (token, `emitted_by` files, meaning) and, if it has a
JSON twin, in `EVENT_SCHEMAS` in `crates/nsl-cli/src/exec_markers.rs`;
build both renderings from one counter snapshot at the call site.
