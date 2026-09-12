# Roadmap A4 — `CudaContext`: one value per device, no process singleton

**Roadmap criterion:** *Introduce a `CudaContext { device, streams:
StreamPool, cublas, cublaslt, allocator: CachingAllocator, events }` value
type; thread it through a `RuntimeCtx` handle that generated code receives
once (`nsl_runtime_init() -> *mut RuntimeCtx`) and passes to every FFI call
(or stores in a per-device registry keyed by `device` byte). Keep a
compatibility shim `state()` that returns device 0's context so the
migration is incremental. Fold the 30 `thread_local!`s into `RuntimeCtx` at
the same time (this is the migration `STATUS.md` already promises). Do it
after A1 so the emission side is small.*

This is a design spec, not a plan: it says what the singleton actually is
today (it is both narrower and wider than the roadmap sentence), what the
per-device value holds, how existing code reaches it without an ABI change,
which of the thread-locals fold into it and which deliberately do not, and
the order of steps that get from one device per process to several under
gates that already exist. The roadmap's two forms — a handle passed to every
FFI call, or a per-device registry keyed by the tensor's `device` byte —
are not equivalent for this codebase; the second is chosen, and the
"Why a registry" section says why.

## Where it stands

Everything below was read from the tree at the time of writing
(`crates/nsl-runtime/src/cuda/mod.rs` and its siblings); line numbers are
approximate and will drift.

**The singleton is four fields.** `cuda::inner::CudaState` holds the
`CUdevice`, the primary `CUcontext`, the `module_cache: HashMap<u64,
CUmodule>` keyed by an FNV-1a hash of the PTX text, and the `func_cache`
keyed by (module hash, entry-name hash). It lives in `static CUDA_STATE:
OnceLock<Mutex<CudaState>>` and is initialised once by `state()`: `cuInit`,
`select_device_ordinal()`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`,
`cuCtxSetCurrent`, every step asserted (a hard panic on failure, which is why
`context_initialized()` exists as the non-forcing probe the diagnostics use).
`select_device_ordinal` honours `NSL_CUDA_DEVICE`, otherwise returns 0 unless
the SPMD spawner protocol is active (`NSL_TP_SHM_PATH` set), in which case
it stripes `NSL_LOCAL_RANK % cuDeviceGetCount()`. `CUDA_VISIBLE_DEVICES` is
never read.

**The doors are few.** Seven functions lock the state directly
(`ensure_context`, `current_device_ordinal`, `detect_sm_version`,
`async_alloc_enabled`, `prefetch_to_device`, `kernel_launch`,
`load_module_once`); everything else goes through `ensure_context()`, which
has about 110 call sites in 21 files (46 in `cuda/mod.rs`, 17 in `zero.rs`,
10 in `weight_stream.rs`, 7 in `sr_bf16.rs`, …). One lock-ordering rule is
documented: `CUDA_STATE` before `CACHING_ALLOCATOR`.

**But the device state is wider than the singleton.** What a second device
would need duplicated is not the four fields; it is these, each its own
static or thread-local today:

| Kind | Where | What |
|---|---|---|
| Streams (3, all thread-local, lazily created, no pool) | `cuda/mod.rs` `COMPUTE_STREAM`, `TRANSFER_STREAM`; `inspect/stream.rs` `INSPECT_STREAM` | the launch stream (blocking, so it orders against the legacy NULL stream), the offload copy-back stream (non-blocking), the debug-copy stream |
| Library handles | `cuda/mod.rs` `CUBLAS_HANDLE`, `RESOLVED_MATH_MODE`; `cuda/lt_matmul.rs` `HANDLE`, `WS`, `PLANS` | one cuBLAS handle, one cublasLt handle plus its device workspace and per-shape plan cache, per process |
| Allocator | `cuda/caching_allocator.rs` `CACHING_ALLOCATOR` (+ thread-local `CURRENT_POOL`, `CURRENT_SURFACE`, `CURRENT_ALLOC_IDENTITY`) | the transient/persistent pools — device memory |
| Free machinery | `cuda/mod.rs` `DEFERRED_FREES`, `FREE_EVENT_POOL`, `ASYNC_ALLOC_RESULT`, `ASYNC_ALLOC_SET`, `CUDA_ALLOC_SET` | stream-ordered frees, the recycled-event pool, the `cuMemAllocAsync` probe and its pointer set |
| Pinned host memory | `cuda/mod.rs` `PINNED_HOST_SET`, `PINNED_LIVE_COUNT` | host-side; routes frees to `cuMemFreeHost` |
| Device-pointer workspaces (thread-local) | `cuda/mod.rs` `WS`, `SR_WS`, `CE_SCRATCH`, `MUON_STATS_BUF`; `muon_batch.rs` `WS_CACHE` | persistent kernel scratch, one pointer each |
| Device-pointer caches | `cuda/bf16_cast_cache.rs` `CACHE`, `cuda/strided_copy.rs` `CACHE`, `cuda/tier_b1_prepass.rs` `CACHE`, `cuda/mod.rs` `:8609 CACHE` (already keyed by ordinal), `fp8.rs` `FP8_SCALES` | bf16 parameter images, resident copy plans, prepass buffers |
| Regions | `slab.rs` `GPU_SLAB_BASE/SIZE`; `transient_arena.rs` `ARENA_BASE/SIZE` (+ thread-local `PIN`, `PLACED_AT`) | one slab and one transient arena per process |
| Graph capture | `cuda/graph_capture.rs` thread-local `ACTIVE`, `REGIONS`, `OCCURRENCE`, `NESTED_SKIP`; `CACHE` (param info by `CUfunction`) | the record/capture/replay state machine |
| Collectives | `tensor_parallel/ffi.rs` `TP_CTX` (the NCCL communicator), `pipeline/comm.rs` `PIPELINE_CTX`, `zero.rs` `ZERO_CTX`, `disaggregated/kv_transfer.rs` `KV_TRANSFER_CTX` | one communicator per process, bound to the one device |

Roughly 25 globals beyond `CudaState`. `CudaContext` has to absorb the
device-bound ones — everything that holds a `CUdeviceptr`, a `CUstream`, a
`CUevent`, a library handle or a communicator — or a second context would
share device 0's memory and streams with device 1.

**Multi-GPU today is one process per device.** `nsl run --devices N`
(`nsl-cli/src/commands/run.rs`) re-executes the CLI N times with
`NSL_LOCAL_RANK`, `NSL_WORLD_SIZE`, `NSL_TP_SHM_PATH` and `NSL_COLLECTIVES`
set; each child binds one device through `select_device_ordinal`. NCCL
(`tensor_parallel/collective.rs`, `feature = "nccl"`) calls
`ensure_context()` before `ncclCommInitRank` and issues on
`current_stream()`; the communicator sits in `TP_CTX`. Six modules read
`NSL_LOCAL_RANK` independently; there is no single rank accessor, no
`nsl_cuda_set_device`, no `nsl_cuda_device_count`.

**The ABI already carries a device byte and nothing else.** `NslTensor.device`
is `0 = CPU, 1+ = CUDA ordinal + 1`, and `nsl_tensor_to_device(tensor,
target_device)` takes the byte — but only 0 and 1 ever occur, and the
runtime resolves every launch to *the* context regardless of the byte. Of
the 682 rows in `nsl_abi::RUNTIME_ABI` (A3), none carries a stream handle
or a device ordinal beyond that byte: `nsl_cuda_device_synchronize()` takes
nothing, `nsl_tp_init()` and `nsl_tp_rank()` take nothing (the rank comes
from the environment), the graph-region rows take a region id. Emitted code
never names a stream. `docs/architecture/runtime.md` ("CUDA backend")
already states the single-device limitation and points at A4.

**What pins the single-device assumption in tests.** `csha_cuda_launch_classic`
requires `--test-threads=1` because a failed PTX load "can corrupt the
context for subsequent launches in the same process"; `m34_v1_context_parallel_single_node`
asserts the exact warning strings `single-device ring math is verified` /
`multi-device distribution`; `zero_grad_gpu` asserts `nsl_zero_init` refuses
`world_size > 1` without `NSL_TP_SHM_PATH`; `zero_gpu_collectives_gate` is
the two-rank NCCL parity gate. None greps for "device 0". The thread-local
inventory in `docs/architecture/compiler-state.md` is machine-checked by
`thread_local_inventory_drift`, so every cell this spec moves is a row to
update there.

## Why a registry, not a handle on every call

The roadmap offers two shapes. A `*mut RuntimeCtx` parameter on every FFI
call would change the signature of several hundred of the 682 rows, every
emitted call site in `nsl-codegen`, the C header, the Python bridge, and the
`ExportRegistry` transmute that the shared-library dispatch path relies on —
for a benefit (an explicit context) that the tensors already carry
implicitly. The `device` byte on every `NslTensor` is that context: an op on
device-k tensors runs on device k's context, and an op with no tensor
argument (`nsl_cuda_device_synchronize`, an allocation, a collective) runs on
the *calling thread's current device*, which is what CUDA itself does with
`cuCtxSetCurrent`.

So the shape is:

```rust
// crates/nsl-runtime/src/cuda/context.rs
pub struct CudaContext {
    pub ordinal: u8,                    // the tensor `device` byte minus 1
    device: CUdevice,
    primary: CUcontext,                 // cuDevicePrimaryCtxRetain
    pub sm_version: u32,
    modules: Mutex<ModuleCache>,        // module_cache + func_cache, as today
    pub streams: StreamPool,            // §Streams
    cublas: OnceLock<CublasHandle>,
    cublaslt: OnceLock<Option<LtHandle>>, lt_workspace: OnceLock<(u64, usize)>, lt_plans: Mutex<…>,
    pub allocator: Mutex<CachingAllocator>,
    frees: Mutex<DeferredFrees>,        // DEFERRED_FREES + FREE_EVENT_POOL + ASYNC_ALLOC_*
    workspaces: Workspaces,             // WS, SR_WS, CE_SCRATCH, MUON_STATS_BUF, WS_CACHE — per (context, thread), see §Streams
    caches: DeviceCaches,               // bf16 images, resident copy plans, prepass buffers, fp8 scales
    regions: Regions,                   // slab and transient-arena base/size
    capture: CaptureState,              // graph_capture's four cells + param-info cache
}

static DEVICES: OnceLock<Box<[OnceLock<CudaContext>]>> = …;   // one slot per cuDeviceGetCount()

pub fn device(ordinal: u8) -> &'static CudaContext;            // init on first use
pub fn current() -> &'static CudaContext;                      // the thread's current ordinal
pub fn set_current(ordinal: u8);                               // cuCtxSetCurrent + the thread-local ordinal
pub fn for_tensor(t: &NslTensor) -> &'static CudaContext;      // device byte → context, CPU → error
```

The thread's current ordinal is a `thread_local! Cell<Option<u8>>` that
defaults, on first read, to `select_device_ordinal()` — exactly the value
the singleton computes today, so a process that never calls `set_current`
behaves as it does now. `cuCtxSetCurrent` is per thread in the driver too,
which is why "current device" is the right granularity and why a process
driving two devices from one thread is a sequence of `set_current` calls, not
a race.

**The compatibility shims** keep every existing door working through the
whole migration: `inner::ensure_context()` becomes `current().activate()`;
`inner::state()` returns `current()`'s module cache; `current_stream()`
becomes `current().streams.compute()`; `cublas_handle()` becomes
`current().cublas()`; `CACHING_ALLOCATOR.lock()` becomes
`current().allocator.lock()`. The ~110 `ensure_context` callers and the 7
`state()` lockers do not change in the first step, and the lock order stays
"context before allocator" (now: the context's module mutex before its
allocator mutex; two contexts are never locked together, which is the rule
that makes the second device safe).

**`RuntimeCtx`** — the roadmap's session handle — is a different object from
the device context and is kept separate on purpose (compiler-state.md,
"do NOT build one mega-context"). It is the autodiff/execution session:
`TAPE`, `TRAINING_MODE`, `TENSOR_SCOPE`, `INPLACE_SUPPRESS_DEPTH`. Two NSL
programs in one Python process need two of *these*, on one or two devices;
two devices under one program need one of these and two `CudaContext`s. §Steps
puts the session object last, after A1's `TrainPlan` has made the emission
side small enough that giving emitted code a session handle is a contained
change.

## Streams

Today's contract (the doc comment on `current_stream`): the compute stream is
per thread, blocking, and "every consumer that records ordering events
against the work that touched this buffer must run on the same thread that
launched that work". That contract is kept — it is what makes the
event-based cross-stream waits (`transfer_stream_wait_null_stream`,
`compute_stream_wait_event`) correct — and generalised from "per thread" to
"per (thread, device)":

```rust
pub struct StreamPool {                 // one per CudaContext
    // The three named streams a thread has today, created on first use,
    // keyed by thread id: compute (blocking), transfer (non-blocking), inspect.
    named: Mutex<HashMap<ThreadId, NamedStreams>>,
    // Dedicated streams for work that must not serialise behind compute:
    // CADENCE's STAC copies, CADRE's ENGRAM prefetch, weight streaming.
    free: Mutex<Vec<CUstream>>,         // CU_STREAM_NON_BLOCKING
}
pub struct StreamLease<'a> { pool: &'a StreamPool, stream: CUstream }   // returns to `free` on drop
impl StreamPool {
    pub fn compute(&self) -> CUstream;  pub fn transfer(&self) -> CUstream;  pub fn inspect(&self) -> CUstream;
    pub fn lease(&self) -> StreamLease<'_>;
    pub fn record(&self, stream: CUstream) -> CUevent;               // from the context's event pool
    pub fn wait(&self, stream: CUstream, ev: CUevent);               // cuStreamWaitEvent
}
```

The device-pointer workspaces (`WS`, `SR_WS`, `CE_SCRATCH`, `MUON_STATS_BUF`,
`WS_CACHE`) are thread-local today for the same reason the streams are: a
kernel on thread A's compute stream must not reuse thread B's scratch. They
move into the same per-(thread, device) slot as the named streams, so the
ownership model is unchanged and a second device gets its own scratch.

`NSL_LEGACY_NULL_STREAM=1` (the kill-switch that makes `current_stream()`
return the NULL stream) and `NSL_CUDA_SYNC=1` (sync after every launch) keep
their meaning per context.

## What folds into the context, and what does not

Of the 30 `thread_local!` cells in `nsl-runtime`, the inventory in
compiler-state.md already classifies each. This spec is the "Phase 3"
that document defers to, for the runtime clusters it names:

| Cells | Class today | Destination |
|---|---|---|
| `COMPUTE_STREAM`, `TRANSFER_STREAM`, `INSPECT_STREAM`, `WS`, `SR_WS`, `CE_SCRATCH`, `MUON_STATS_BUF`, `WS_CACHE` | RUNTIME-OK (thread-affine for correctness) | `StreamPool`'s per-(thread, device) slot. Still thread-affine; now also device-affine. |
| `CURRENT_POOL`, transient arena `PIN` / `PLACED_AT`, graph-capture `ACTIVE` / `REGIONS` / `OCCURRENCE` / `NESTED_SKIP` | MIGRATE (cluster 3) | `CaptureState` and the allocator's placement channel on the context, behind the same RAII guards. The "steer the next allocation through a global because the FFI has no out-parameter" shape is solved once, on the context, as the doc asks. |
| `CURRENT_SURFACE`, `CURRENT_ALLOC_IDENTITY` | RUNTIME-OK (metadata) | Stay thread-local: they tag allocations on whatever context the allocation lands on. |
| `OFFLOAD_DRAIN_TENSORS`, `OFFLOAD_DRAIN_DEVICE_BUFS` | RUNTIME-OK (paired with `TRANSFER_STREAM`) | Move with the transfer stream into the per-(thread, device) slot. |
| `OOM_CONTEXT`, `LAST_ERROR`, `DISPATCH_MODE`, `RNG`, `STAGING_REGISTRY`, `ALLOC_REGISTRY`, `INSPECT` / `MUON_PROF` scopes, the TEST cells | FFI/RUNTIME-OK, TEST | Stay. None holds device state. |
| `TAPE`, `TRAINING_MODE`, `TENSOR_SCOPE`, `INPLACE_SUPPRESS_DEPTH`, `PACKING_METADATA` | MIGRATE (clusters 1–2) | `RuntimeCtx`, the session object (§Steps, last). |

So "fold the 30 thread-locals" resolves to: 8 + 2 move into the device
context's per-thread slots, 7 move into the context proper, 5 move into the
session object, and 8 stay because thread affinity is their correctness
model. The inventory table is updated in the step that moves each row; the
drift gate is what proves the table and the tree agree.

## Device selection and the tensor byte

Two rows join the ABI table (A3), under a new `[cuda]` group:

```text
[cuda] nsl_cuda_device_count() -> i64            cuda::context::nsl_cuda_device_count ;
[cuda] nsl_cuda_set_device(i64) -> i64           cuda::context::nsl_cuda_set_device   ;   // status: 0 ok, RuntimeError code otherwise
```

`nsl_cuda_init()` keeps its meaning (initialise the current device).
`select_device_ordinal()` becomes the *default* for a thread that never
called `nsl_cuda_set_device`, so the SPMD spawner protocol is untouched: a
rank still binds `NSL_LOCAL_RANK % count` without any call.

The runtime then starts honouring the byte it already carries. Every launch
path resolves its context with `for_tensor(t)` instead of `current()`; an op
whose tensor arguments live on different devices returns the C1
`RuntimeError` (`DeviceMismatch`) rather than launching on the wrong
context; `nsl_tensor_to_device(t, k)` for `k >= 2` becomes a real
device-to-device copy (`cuMemcpyPeerAsync` on the source's transfer stream,
with an event the destination's compute stream waits on) instead of the
`GPU-to-GPU transfer` panic that sits at that site today. The allocation
paths take the context from the calling thread, as they must: a fresh
tensor has no byte yet.

Codegen is unchanged by this: emitted code passes tensors, and the byte
travels with them. The one compile-time consumer of a device —
`gpu_specs::local_device_identity()` for the autotune cache key and the
CSHA spec lookup — keeps probing device 0 of the compiling machine, which is
what it means.

## Multi-device in one process

With contexts per device and a current-device per thread, the SPMD protocol
can run in-process: `nsl run --devices N --in-process` spawns N threads
instead of N processes, each calling `nsl_cuda_set_device(rank)` first, with
the shm-file collective backend replaced by an in-memory one (the
`SimulatedBackend` already abstracts the transport) and one NCCL communicator
per context (`TP_CTX` becomes a field of `CudaContext`, created by the rank's
thread). The process-per-device spawner stays the default; in-process is the
mode the distributed tier, CADENCE and a Python host driving two devices
need, and it is also the mode where a failed PTX load on one device does not
take the others down — the thing `csha_cuda_launch_classic` serialises
tests to avoid.

Two NSL programs in one process (the Python-embedding case) is the session
object's job, not the context's: each program gets a `RuntimeCtx` and both
may share one `CudaContext`.

## Measured outcomes

- `nsl run --devices 2 --in-process` trains `coder50m` on two GPUs of the
  CachyOS box with the same loss trajectory as the two-process run
  (`scripts/gpu-tier.sh certify` gains the lane; `score_trajectory.sh` diffs).
- A `StreamLease` copy overlapping a compute kernel shows up as two concurrent
  streams in `nsys` (the STAC/ENGRAM prerequisite), where today the offload
  path is the only second stream.
- The thread-local inventory shrinks from 30 rows to the 8 that stay plus the
  per-(thread, device) slot, and `thread_local_inventory_drift` pins it.
- `csha_cuda_launch_classic` drops its `--test-threads=1` requirement once a
  test can bind a throwaway context.

## Steps

Each is one PR. CPU CI cannot see a GPU, so the gates are: the `cuda` job
(compile + GPU-free tests) on every step, `scripts/gpu-tier.sh smoke` on the
CachyOS box for any step that changes a launch path, and the two-rank
`zero_gpu_collectives_gate` / `certify` lanes for steps 5–6. Every step is
behaviour-preserving on one device; the `m34` warning strings and the
`zero_grad_gpu` refusal semantics are untouched until step 6 gives them a
new true statement.

1. **`CudaContext` with one entry.** `cuda/context.rs` holds the struct,
   `DEVICES` with one slot, `current()` / `device(0)`; `CudaState` and its
   `OnceLock` are replaced by the shims above. `DEFERRED_FREES`,
   `FREE_EVENT_POOL`, the async-alloc probe/set and `CUDA_ALLOC_SET` move
   in. No caller changes. **Done.**

   Two deviations from the sketch above, both because this step has to be
   provably behaviour-preserving on CPU CI, where no lane can observe a GPU
   regression:

   - The context keeps **one mutex per former static** (`frees`,
     `free_events`, `async_allocs`, `allocs`) rather than the single
     `frees: Mutex<DeferredFrees>` the struct sketch shows. Merging them
     changes lock granularity on the allocation hot path; that belongs to a
     step that can run `scripts/gpu-tier.sh`. Today's drain paths call
     `recycle_free_event` deliberately *outside* the queue lock so the queue
     and the allocator never nest — collapsing the two locks would have to
     preserve that by construction.
   - `sm_version` is a method that queries the driver per call, not the
     cached `pub sm_version: u32` field. Caching is safe (a live device's
     capability cannot change) but is a behaviour change this step does not
     need.

   One incidental behaviour change was unavoidable and is called out here:
   `prefetch_to_device` used to hold the singleton's mutex across its
   `cuMemPrefetchAsync_v2` call while reading no field of it — the lock was
   only how the function reached `state()` to force initialisation.
   `context::current()` forces initialisation directly, so that incidental
   serialisation is gone. The driver call is itself thread-safe.
2. **`StreamPool`.** The three named streams and the five workspaces move
   into the per-(thread, device) slot; `StreamLease` and the event helpers
   are added; `current_stream()`, `transfer_stream()`,
   `current_inspect_stream()` become shims. First user of a lease: the
   weight-streaming prefetch. Inventory rows updated.
3. **Handles, allocator, caches, regions.** cuBLAS, cublasLt (handle,
   workspace, plans), `CACHING_ALLOCATOR`, the bf16/strided-copy/prepass
   caches, `FP8_SCALES`, the slab and arena bases move in; the lock order
   is restated on the context. `caching_allocator`'s public `LazyLock` is
   replaced by `current().allocator`; its tests keep working through the
   shim.
4. **Capture and placement state.** graph_capture's four cells and its
   param-info cache, `CURRENT_POOL`, the arena's `PIN` / `PLACED_AT` become
   `CaptureState` and the allocator's placement channel on the context
   (compiler-state.md cluster 3), RAII guards unchanged.
5. **Honour the device byte.** `DEVICES` gets `cuDeviceGetCount()` slots;
   the two `[cuda]` rows; `for_tensor` at every launch and cuBLAS site;
   `DeviceMismatch`; peer copies in `nsl_tensor_to_device`. Gate: the
   existing single-device lanes plus a two-device `to_device` round-trip
   test under `certify`.
6. **In-process SPMD.** `--in-process`, the in-memory collective backend,
   `TP_CTX` per context, the `certify` two-device trajectory lane. The
   `m34` and `zero_grad_gpu` assertions are updated in this step to their
   new true statements.
7. **`RuntimeCtx`.** The session object for `TAPE`, `TRAINING_MODE`,
   `TENSOR_SCOPE`, `INPLACE_SUPPRESS_DEPTH` (deciding the last one's fate
   first, as compiler-state.md asks) and `PACKING_METADATA`'s 3b half:
   stored thread-locally with `nsl_runtime_init() -> *mut RuntimeCtx` /
   `nsl_runtime_set_current(ctx)` rows so a host can switch programs, and
   an emitted `nsl_runtime_init` at program entry. After A1's `TrainPlan`
   steps, per the roadmap's ordering.

Effort: steps 1–4 are mechanical moves behind shims, about a week each
including the GPU lanes; step 5 is the first behaviour change and needs the
two-device box; steps 6–7 are the roadmap's "6–8 weeks" tail.

## Non-goals

- No signature changes to existing ABI rows, no stream or device parameters
  on emitted calls, no change to the C header beyond the two new rows.
- No change to the default multi-GPU mode: the process-per-device spawner
  and the shm collective backend stay, and every `NSL_LOCAL_RANK` reader
  keeps reading it.
- No adoption of cudarc's own `CudaContext` / `CudaStream` wrappers: the
  runtime drives the driver API directly, and the contexts here are primary
  contexts retained per device, as today.
- No multi-device *placement* decisions in the compiler (which device a
  parameter lives on is the distributed tier's planner's job, after this).
- Pinned host memory and the C-ABI error slot stay process- and
  thread-global respectively; neither is device state.
