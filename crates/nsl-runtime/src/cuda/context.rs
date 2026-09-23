//! Roadmap A4 step 1 — `CudaContext`, the per-device value.
//!
//! Before this module the runtime's CUDA state was a process singleton:
//! `inner::CudaState` in a `static OnceLock<Mutex<…>>`, plus a scattering of
//! sibling statics holding the deferred-free queue, the recycled-event pool,
//! the async-allocation probe and the two allocation-tracking sets. Every one
//! of those holds *device* state — a `CUcontext`, a `CUmodule`, a `CUevent`,
//! or a set of `CUdeviceptr`s — so a second device could not exist without
//! silently sharing device 0's.
//!
//! [`CudaContext`] is that state as a value, one per device, reached through a
//! registry rather than through a handle threaded onto every FFI call. The
//! design spec (`docs/superpowers/specs/2026-09-09-a4-cuda-context-design.md`,
//! "Why a registry, not a handle on every call") says why: every `NslTensor`
//! already carries a `device` byte, so the context an op needs is implied by
//! its arguments, and a handle parameter would change several hundred ABI rows
//! for a fact the tensors already state.
//!
//! **This step is one slot.** [`DEVICES`] is sized 1, [`current`] is that
//! slot, and the ordinal it binds is exactly what the singleton computed
//! before — `select_device_ordinal()`, which honours `NSL_CUDA_DEVICE` and the
//! SPMD spawner's rank striping. So a process behaves precisely as it did; the
//! shape is what changed, not the behaviour. Step 5 grows the registry to
//! `cuDeviceGetCount()` slots, adds the thread-current-ordinal cell, and
//! starts resolving each launch through the tensor's byte.
//!
//! **Slot index is not the CUDA ordinal.** In this step slot 0 holds whatever
//! ordinal `select_device_ordinal()` picked, which under the SPMD spawner is
//! `NSL_LOCAL_RANK % count` and need not be 0. The two coincide only once the
//! registry has one slot per device, which is step 5's job — and step 5 is
//! also where the context grows an `ordinal` field. It is deliberately absent
//! here: with one slot nothing reads it, and the crate denies dead code rather
//! than carrying speculative state. The `CUdevice` handle that
//! `cuDeviceGet(ordinal)` returned is kept, because the shims do read it.
//! (Step 2's [`StreamPool`] carries the *registry slot* index, which is the
//! key into the per-thread table — still not the CUDA ordinal.)
//!
//! **Step 2 adds [`StreamPool`].** The three named streams (compute, transfer,
//! inspect) and the five device-pointer workspaces were `thread_local!` cells
//! scattered across `cuda/mod.rs`, `inspect/stream.rs` and `muon_batch.rs`.
//! They are thread-affine for a correctness reason that has not changed — two
//! blocking streams do not synchronise with each other, so the thread that
//! launched work is the only one that may record ordering events against it —
//! and they are now *also* device-affine: one [`ThreadSlot`] per (thread,
//! device), reached through the context. The slot index is what step 1
//! deliberately left off the context; it earns its place here.
//!
//! **Step 4 adds the capture state** to the same slot: `graph_capture`'s
//! four cells become one `CaptureState` field, and its `cuFuncGetParamInfo`
//! answers — keyed by `CUfunction`, so device state — move onto
//! [`CudaContext::with_cache`]. Step 4b adds the allocator's placement
//! channel — the pool selector and the transient arena's pin — as the
//! slot's [`Placement`] field, reached through [`with_placement`], which
//! unlike every other door here does not create a context.

use cudarc::driver::sys::*;
use std::any::{Any, TypeId};

use super::caching_allocator::AllocPool;
use super::graph_capture::CaptureState;
use crate::device_region::Region;
use crate::transient_arena::ArenaPin;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Mutex, OnceLock};

/// The PTX module cache and the resolved-function cache for one device.
///
/// Both are keyed by content hash rather than by pointer: different PTX `Vec`s
/// can land at the same heap address between sequential calls, and a key
/// derived from the address then produces a stale hit — observed as
/// `CUDA_ERROR_NOT_FOUND` (rc=500). Modules are never unloaded, so a
/// `CUfunction` stays valid for as long as its module's entry lives here.
pub(crate) struct ModuleCache {
    /// Keyed by FNV-1a of the PTX text.
    pub(crate) modules: HashMap<u64, CUmodule>,
    /// Keyed by (module content hash, FNV-1a of the entry name).
    pub(crate) funcs: HashMap<(u64, u64), CUfunction>,
}

// SAFETY: `CUmodule` and `CUfunction` are opaque driver handles that the Rust
// side never dereferences. The cache is only ever reached through the
// context's `Mutex`, so the handles are not shared concurrently, and every
// driver call that consumes one first makes the owning context current.
unsafe impl Send for ModuleCache {}

/// One deferred-free batch: buffers sharing a lifetime, guarded by a single
/// completion event.
pub(crate) struct DeferredFree {
    /// Buffers all consumed by the same preceding kernels.
    pub(crate) ptrs: Vec<usize>,
    pub(crate) event: CUevent,
}

// SAFETY: `CUevent` is an opaque driver handle (a raw pointer) that is never
// dereferenced on the Rust side; every driver call touching it first
// re-establishes the owning primary context. Moving the handle between threads
// — the queue lives in a `static` — is therefore sound.
unsafe impl Send for DeferredFree {}

/// Everything one CUDA device owns.
///
/// **Lock order (restated in roadmap A4 step 3d, when the allocator moved
/// in).** The singleton documented "CUDA_STATE before CACHING_ALLOCATOR",
/// because `ensure_context` locked the state and `alloc_managed` called it
/// with the allocator held. Step 1 made `ensure_context` lock-free
/// (`activate` is one `cuCtxSetCurrent`), so that pair can no longer nest,
/// and what the code actually does now is stronger than an order:
///
/// - **[`allocator`](Self::allocator) is a leaf.** While it is held, the
///   code calls the driver (`cuMemAlloc_v2` / `cuMemFree_v2` in `drain_all`
///   and the grow path), reads thread-locals and writes the log — never
///   another of this context's mutexes and never the allocator's own entry
///   points (`caching_allocator::the_allocator_methods_take_no_lock` holds
///   its methods to that). Every caller releases it before touching
///   `allocs`, `async_allocs` or `frees`: `caching_alloc` drops the guard
///   before `register_cuda_alloc`, `free_managed` releases `allocs` before
///   taking it, and the deferred-free drain collects under `frees` and frees
///   after.
/// - **[`with_cache`](Self::with_cache) never reaches the allocator** (its
///   doc says why: `free_managed` opens with a `with_cache` call).
/// - If a future path must nest, the rule the singleton wrote down still
///   applies — this context's other mutexes outside, the allocator inside —
///   and it must be added to this list.
///
/// Two contexts are never locked together — that is the invariant that makes
/// a second device safe, and step 5 depends on it. With one allocator per
/// context it also means a device's blocks can only ever be filed in that
/// device's free lists.
///
/// The mutexes are deliberately one-per-former-static rather than the single
/// `frees: Mutex<DeferredFrees>` the design spec sketches. Merging them would
/// change lock granularity on the allocation hot path, and this step is meant
/// to be provably behaviour-preserving on CPU CI, where no GPU lane can
/// observe a regression. The merge belongs to a step that can run
/// `scripts/gpu-tier.sh`.
pub(crate) struct CudaContext {
    /// The `CUdevice` handle from `cuDeviceGet`.
    pub(crate) device: CUdevice,
    /// The retained primary context. Made current by [`CudaContext::activate`].
    primary: CUcontext,
    /// Roadmap A4 step 2: the three named streams (per thread), the shared
    /// lease free list and the completion-event pool.
    pub(crate) streams: StreamPool,
    /// Was `CudaState`'s `module_cache` + `func_cache`.
    pub(crate) modules: Mutex<ModuleCache>,
    /// Was `DEFERRED_FREES`.
    pub(crate) frees: Mutex<VecDeque<DeferredFree>>,
    /// Was `FREE_EVENT_POOL` — recycled disable-timing events.
    pub(crate) free_events: Mutex<Vec<usize>>,
    /// Was `ASYNC_ALLOC_RESULT` — the `cuMemAllocAsync` support probe.
    pub(crate) async_alloc: OnceLock<bool>,
    /// Was `ASYNC_ALLOC_SET` — pointers to free via `cuMemFreeAsync`.
    pub(crate) async_allocs: Mutex<HashSet<usize>>,
    /// Was `CUDA_ALLOC_SET` — every allocation, so frees can be validated.
    pub(crate) allocs: Mutex<HashSet<usize>>,
    /// Roadmap A4 step 3a: was `inner`'s `CUBLAS_HANDLE`. A cuBLAS handle
    /// binds whichever CUDA context was current when it was created, so it is
    /// device state; `OnceLock` keeps the exactly-once creation the static
    /// had.
    pub(crate) cublas: OnceLock<CublasHandle>,
    /// Was `lt_matmul`'s `HANDLE`. `None` means `cublasLtCreate` failed and
    /// every GEMM falls back — cached so the failure is diagnosed once.
    pub(crate) cublaslt: OnceLock<Option<LtHandle>>,
    /// Was `lt_matmul`'s `WS`: the cublasLt workspace as `(device ptr, bytes)`,
    /// `(0, 0)` if the allocation failed. A device pointer, hence per device.
    /// Never freed — captured graph nodes may reference it for the life of the
    /// process, exactly as before.
    pub(crate) lt_workspace: OnceLock<(u64, usize)>,
    /// Was `lt_matmul`'s `PLANS`, reached through [`CudaContext::with_cache`]
    /// so the plan type stays declared in `lt_matmul`.
    caches: Mutex<HashMap<TypeId, Box<dyn Any + Send>>>,
    /// The compile-time-planned transient arena for this device (roadmap A4
    /// step 3c). Was `transient_arena`'s `ARENA_BASE` / `ARENA_SIZE`.
    ///
    /// Plain atomics rather than a slot in `caches`: `Region::contains` runs
    /// at the top of every `free_managed` and `base` is read once per wrapped
    /// op, so this is read-hot state, where the plan cache is read-rare. See
    /// [`crate::device_region`].
    pub(crate) arena: Region,
    /// The compile-time-planned GPU slab for this device (roadmap A4 step
    /// 3c). Was `slab`'s `GPU_SLAB_BASE` / `GPU_SLAB_SIZE`. Cold by
    /// comparison — allocated at program start, freed at exit — but it is the
    /// same kind of thing and shares the type.
    pub(crate) slab: Region,
    /// Roadmap A4 step 3d: was `caching_allocator`'s
    /// `pub static CACHING_ALLOCATOR: LazyLock<Mutex<CachingAllocator>>`.
    /// The pools hold this device's pointers, so they are this device's.
    /// Reached through `caching_allocator::allocator()` (forcing, for the
    /// alloc/free paths) or `allocator_if_initialized()` (for the stats rows
    /// and reports a CPU-only run can reach). A leaf lock — see the struct
    /// doc.
    pub(crate) allocator: Mutex<super::caching_allocator::CachingAllocator>,
}

// SAFETY: the two raw driver handles (`primary`, and `device` which is an
// `i32`) are opaque and never dereferenced here; `primary` is only ever handed
// back to `cuCtxSetCurrent`. Every interior-mutable field is behind its own
// `Mutex`. The context is created once and then shared by `&'static`
// reference, so nothing else can move or alias it.
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    /// Make this context current on the calling thread.
    ///
    /// `cuCtxSetCurrent` is per-thread in the driver, which is why "current
    /// device" is thread granularity and why one thread driving two devices is
    /// a sequence of these calls rather than a race.
    pub(crate) fn activate(&self) {
        unsafe {
            cuCtxSetCurrent(self.primary);
        }
    }

    /// Compute capability as `major * 10 + minor` — 90 for Hopper H100, 89 for
    /// Ada RTX 4090, 100 for Blackwell B200.
    ///
    /// Queried on each call, as the singleton did. It cannot change for a
    /// live device, so caching it is safe but is a behaviour change this step
    /// does not need.
    pub(crate) fn sm_version(&self) -> u32 {
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        unsafe {
            cuDeviceGetAttribute(
                &mut major,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self.device,
            );
            cuDeviceGetAttribute(
                &mut minor,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self.device,
            );
        }
        (major * 10 + minor) as u32
    }

    /// Is stream-ordered async allocation enabled *and* supported here?
    ///
    /// `OnceLock` rather than a re-probe: the answer is a property of the
    /// device and the environment, and probing twice would race.
    pub(crate) fn async_alloc_enabled(&self) -> bool {
        *self.async_alloc.get_or_init(|| {
            let env_enabled = std::env::var("NSL_ASYNC_ALLOC")
                .map(|v| v == "1")
                .unwrap_or(false);
            if !env_enabled {
                return false;
            }
            let supported = unsafe {
                let mut pool: CUmemoryPool = std::ptr::null_mut();
                let r = cuDeviceGetDefaultMemPool(&mut pool, self.device);
                r == CUresult::CUDA_SUCCESS && !pool.is_null()
            };
            if supported {
                crate::nsl_log!(
                    INFO,
                    "nsl",
                    "[nsl] Async GPU allocation ENABLED (cuMemAllocAsync)"
                );
            } else {
                crate::nsl_log!(
                    INFO,
                    "nsl",
                    "[nsl] NSL_ASYNC_ALLOC=1 but driver does not support memory pools — using sync alloc"
                );
            }
            supported
        })
    }

    /// Run `f` against this (thread, device)'s workspace of type `T`,
    /// creating it with `Default` on first use.
    ///
    /// This is the generic replacement for the four `thread_local!` cells
    /// that held one device pointer each (`WS`, `SR_WS`, `CE_SCRATCH`,
    /// `MUON_STATS_BUF`) plus `muon_batch`'s keyed `WS_CACHE`. Keying by
    /// `TypeId` means each workspace type stays declared where it is used —
    /// `MultiWs` is still a local type inside `fase_fused_adamw_multi` — and
    /// this module never has to name any of them. Steps 3 and 4 move more
    /// per-thread device state in without touching this function.
    ///
    /// The `&mut T` is exclusive for the duration of `f` *provided* `f` does
    /// not itself ask for the same `T`, which is precisely the contract the
    /// raw-pointer cells had before (they handed out `&mut *ptr` and held it
    /// across the whole body). Asking for a *different* `T`, or for a stream,
    /// is fine: the map's borrow is released before `f` runs.
    ///
    /// `T` must not implement `Drop` in a way that calls the driver — the
    /// boxes are freed at thread exit, when the context may already be gone.
    /// Device memory a workspace owns is leaked at thread exit, exactly as it
    /// was before this step.
    pub(crate) fn with_workspace<T: Default + 'static, R>(
        &self,
        f: impl FnOnce(&mut T) -> R,
    ) -> R {
        self.streams.with_workspace(f)
    }

    /// Run `f` against this DEVICE's cache of type `T`, creating it with
    /// `Default` on first use.
    ///
    /// The device-level counterpart to [`CudaContext::with_workspace`], and
    /// the difference is the whole point: a workspace is per (thread, device)
    /// because a kernel on thread A's stream must not reuse thread B's
    /// scratch, whereas a cache like `lt_matmul`'s plan map is shared by every
    /// thread on the device and needs a lock rather than thread affinity. So
    /// this one hands out `&mut T` *while holding the mutex* — `f` must not
    /// re-enter for the same `T`, and should not do long GPU work under it.
    ///
    /// **`f` must not call the allocator.** One mutex guards every cache on
    /// the device, so `alloc_managed` or `free_managed` inside any closure can
    /// come back through another cache and self-deadlock — `free_managed`'s
    /// first act is `bf16_cast_cache::evict`, which is a `with_cache` call,
    /// and `alloc_managed`'s OOM recovery frees. This is not the theoretical
    /// hazard it looks like: two of step 3b's caches published a duplicate
    /// device buffer by calling `free_managed` under their own (then separate)
    /// lock, and both had to move that free after the critical section to be
    /// migrated here. The shape that works is short closures with the driver
    /// calls *between* them: probe, allocate, install, and release the loser
    /// outside. Step 3a's plan cache already followed it, for the different
    /// reason that `build_plan` syncs the device while timing.
    ///
    /// Keyed by `TypeId` for the same reason as the workspaces: `lt_matmul`'s
    /// `Plan` stays declared in `lt_matmul`, and this module names no caller's
    /// type. Roadmap A4 steps 3a and 3b.
    pub(crate) fn with_cache<T: Default + Send + 'static, R>(
        &self,
        f: impl FnOnce(&mut T) -> R,
    ) -> R {
        let mut map = self.caches.lock().unwrap();
        let entry = map
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(T::default()) as Box<dyn Any + Send>);
        let t: &mut T = entry
            .downcast_mut::<T>()
            .expect("device cache map keyed by TypeId cannot hold another type");
        f(t)
    }
}

// ---------------------------------------------------------------------------
// Roadmap A4 step 3a — the library handles.
// ---------------------------------------------------------------------------

/// A cuBLAS handle.
///
/// Defined here rather than in `cuda::mod`'s `inner` because the context owns
/// the field; `inner` imports it back. A handle binds the CUDA context that
/// was current when `cublasCreate_v2` ran, which is exactly why it is device
/// state and not process state.
#[derive(Copy, Clone)]
pub(crate) struct CublasHandle(pub cudarc::cublas::sys::cublasHandle_t);

// SAFETY: `cublasHandle_t` is an opaque library-managed pointer that the Rust
// side never dereferences. cuBLAS documents handles as usable from multiple
// threads with external serialization; NSL serializes GPU dispatch.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

/// A cublasLt handle. Same ownership story as [`CublasHandle`].
#[derive(Copy, Clone)]
pub(crate) struct LtHandle(pub cudarc::cublaslt::sys::cublasLtHandle_t);

// SAFETY: as [`CublasHandle`] — opaque, never dereferenced here, and
// documented thread-safe by cublasLt.
unsafe impl Send for LtHandle {}
unsafe impl Sync for LtHandle {}

// ---------------------------------------------------------------------------
// Roadmap A4 step 2 — the stream pool and the per-(thread, device) slot.
// ---------------------------------------------------------------------------

/// The streams and device-pointer workspaces belonging to one (thread,
/// device) pair.
///
/// Every cell here was a `thread_local!` of its own before this step. They
/// stay thread-affine — that is their correctness model, not an accident:
/// the compute stream is a BLOCKING stream, and two blocking streams do not
/// synchronise with each other, so only the thread that launched work may
/// record ordering events against it. What changes is that the affinity is
/// now to a *pair*: thread B on device 1 no longer reuses thread B's device-0
/// scratch, which is what makes step 5's second device possible.
///
/// Handles are stored as `usize` because the driver's are `!Send` raw
/// pointers and a `thread_local!` initialiser must be `const` to stay on the
/// fast path; `0` means "not created yet", as it did before.
pub(crate) struct ThreadSlot {
    /// Was `cuda::mod`'s `COMPUTE_STREAM` — blocking (`CU_STREAM_DEFAULT`).
    compute: Cell<usize>,
    /// Was `cuda::mod`'s `TRANSFER_STREAM` — `CU_STREAM_NON_BLOCKING`.
    transfer: Cell<usize>,
    /// Was `inspect::stream`'s `INSPECT_STREAM` — blocking, debug copies.
    inspect: Cell<usize>,
    /// Was `WS`, `SR_WS`, `CE_SCRATCH`, `MUON_STATS_BUF` and `muon_batch`'s
    /// `WS_CACHE`, keyed by the workspace's own type so no workspace type has
    /// to be named here. See [`CudaContext::with_workspace`].
    workspaces: RefCell<HashMap<TypeId, Box<dyn Any>>>,
    /// Roadmap A4 step 4: was `graph_capture`'s `ACTIVE`, `REGIONS`,
    /// `OCCURRENCE` and `NESTED_SKIP`. A named field rather than a
    /// `workspaces` entry because every launch consults it while a capture
    /// run is armed, and a field is one pointer hop where the map is a hash.
    /// See [`StreamPool::with_capture`].
    capture: CaptureState,
    /// Roadmap A4 step 4b: was `caching_allocator`'s `CURRENT_POOL` and the
    /// transient arena's `PIN` / `PLACED_AT`. See [`with_placement`].
    placement: Placement,
}

/// The allocator's placement channel for one (thread, device): what steers
/// the *next* allocation on this thread, set by a caller that cannot pass it
/// down because the allocation happens inside an `extern "C"` row whose
/// signature has no room for it.
///
/// Both halves were `thread_local!` cells and both stay thread-affine: a
/// `PoolGuard` brackets allocations the same thread makes, and an arena bind
/// is consumed by the next allocation the same thread makes. What the slot
/// adds is the device half — a bracket on device 1 must not retag device 0's
/// allocations once step 5 lets a thread switch devices.
///
/// Step 5 note: the guards restore through the slot that is current when
/// they DROP. A bracket that switches device in the middle would restore the
/// other device's selector; nothing does that today (there is one slot), and
/// step 5 is where it becomes possible and where the guards learn their slot.
pub(crate) struct Placement {
    /// Was `CURRENT_POOL`. Read by every allocation, written by `PoolGuard`
    /// and the two `nsl_gpu_set_*_pool` rows.
    pub(crate) pool: Cell<AllocPool>,
    /// Was `PIN` and `PLACED_AT`.
    pub(crate) arena: ArenaPin,
}

impl Placement {
    const fn new() -> Self {
        Self {
            pool: Cell::new(AllocPool::Transient),
            arena: ArenaPin::new(),
        }
    }
}

impl ThreadSlot {
    fn new() -> Self {
        Self {
            compute: Cell::new(0),
            transfer: Cell::new(0),
            inspect: Cell::new(0),
            workspaces: RefCell::new(HashMap::new()),
            capture: CaptureState::default(),
            placement: Placement::new(),
        }
    }
}

thread_local! {
    /// This thread's slots, indexed by registry slot. Boxed so a slot's
    /// address is stable once created: `StreamPool::with_slot` hands out a
    /// reference *after* releasing the `RefCell` borrow, which is what makes
    /// the re-entrant paths legal (the `WS` realloc closure calls
    /// `current_stream()`, which reaches back into the same slot).
    ///
    /// Dropped at thread exit, which frees the `Box`es and nothing else — no
    /// workspace type may implement `Drop` with a driver call in it, because
    /// the context may already be gone by then. That matches the previous
    /// behaviour exactly: these cells leaked their device memory too.
    ///
    /// The one exception is the capture state, and it is the exception it
    /// already was: a captured region's pinned staging buffers are freed by
    /// `cuMemFreeHost` on drop, and the `REGIONS` cell they lived in before
    /// step 4 dropped them at thread exit in the same way. The call ignores
    /// its result, and primary contexts are never released, so it is a
    /// successful free or a no-op.
    static THREAD_SLOTS: RefCell<Vec<Option<Box<ThreadSlot>>>> =
        const { RefCell::new(Vec::new()) };
}

/// Streams that are *not* one of the three named ones: a shared free list of
/// non-blocking streams handed out as [`StreamLease`]s, plus a pool of
/// recycled completion events for [`StreamPool::record`].
///
/// Shared across threads, unlike [`ThreadSlot`]: a lease exists precisely so
/// a bounded piece of work can run off the thread's named streams, and a
/// recycled `CUstream` is FIFO, so the next borrower is ordered *behind* the
/// previous borrower's tail. That can cost overlap; it can never race.
pub(crate) struct StreamPool {
    /// Which registry slot this pool belongs to — the index into
    /// [`THREAD_SLOTS`] for the named streams.
    slot: usize,
    /// Idle `CU_STREAM_NON_BLOCKING` streams, as `usize`.
    free: Mutex<Vec<usize>>,
    /// EVERY stream this pool has created, idle or out on lease, as `usize`.
    ///
    /// [`StreamPool::synchronize_leases`] walks this rather than `free` on
    /// purpose: a drain that only covered idle streams would be correct only
    /// as long as every borrower happened to drop its lease first, which is a
    /// property of today's one caller and not of the API.
    all: Mutex<Vec<usize>>,
    /// Idle `CU_EVENT_DISABLE_TIMING` events, as `usize`.
    ///
    /// Deliberately *not* the context's `free_events`: that pool belongs to
    /// the deferred-free machinery, whose drain path recycles outside the
    /// queue lock on purpose. Sharing one pool would couple two schedules for
    /// no gain.
    events: Mutex<Vec<usize>>,
}

// SAFETY: every field is either a `usize` or a `Mutex` of `usize`s. The
// driver handles they encode are opaque and never dereferenced on the Rust
// side; each is only handed back to a driver call that first makes the owning
// context current.
unsafe impl Send for StreamPool {}
unsafe impl Sync for StreamPool {}

/// A non-blocking stream borrowed from a [`StreamPool`], returned on drop.
///
/// The borrower owns the ordering of its own work, exactly as the transfer
/// stream's callers do today: record an event when the work is issued and
/// have the consumer wait on it. Drop does **not** synchronise — a lease
/// whose work is still in flight is fine, because a `CUstream` is FIFO and
/// the next borrower's work simply queues behind it.
pub(crate) struct StreamLease<'a> {
    pool: &'a StreamPool,
    stream: CUstream,
}

impl StreamLease<'_> {
    pub(crate) fn stream(&self) -> CUstream {
        self.stream
    }
}

impl Drop for StreamLease<'_> {
    fn drop(&mut self) {
        self.pool.free.lock().unwrap().push(self.stream as usize);
    }
}

impl StreamPool {
    fn new(slot: usize) -> Self {
        Self {
            slot,
            free: Mutex::new(Vec::new()),
            all: Mutex::new(Vec::new()),
            events: Mutex::new(Vec::new()),
        }
    }

    /// Run `f` against the calling thread's slot for this pool's device,
    /// creating the slot on first use. See [`with_thread_slot`].
    fn with_slot<R>(&self, f: impl FnOnce(&ThreadSlot) -> R) -> R {
        with_thread_slot(self.slot, f)
    }

    /// Run `f` against this (thread, device)'s workspace of type `T`. See
    /// [`CudaContext::with_workspace`], which is the door callers use; the
    /// implementation lives here because it is slot bookkeeping and touches
    /// no driver state, which is also what makes it testable without a GPU.
    fn with_workspace<T: Default + 'static, R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let ptr = self.with_slot(|s| {
            let mut map = s.workspaces.borrow_mut();
            let entry = map
                .entry(TypeId::of::<T>())
                .or_insert_with(|| Box::new(T::default()) as Box<dyn Any>);
            let t: &mut T = entry
                .downcast_mut::<T>()
                .expect("workspace map keyed by TypeId cannot hold another type");
            &raw mut *t
        });
        // SAFETY: the box is owned by this thread's slot, is never replaced
        // or moved out, and outlives this frame. Exclusivity is the caller
        // contract documented on `CudaContext::with_workspace`.
        f(unsafe { &mut *ptr })
    }

    /// Run `f` against this (thread, device)'s CUDA-graph capture state.
    ///
    /// A shared reference, not `&mut`: the state keeps the interior
    /// mutability its four `thread_local!` cells had, so nested calls — the
    /// capture hooks reach back in from inside each other — are exactly as
    /// legal as nested `LocalKey::with` calls were. Roadmap A4 step 4.
    pub(crate) fn with_capture<R>(&self, f: impl FnOnce(&CaptureState) -> R) -> R {
        self.with_slot(|s| f(&s.capture))
    }

    /// Lazily create and return this (thread, device)'s blocking compute
    /// stream — where `kernel_launch` issues, and where the profiler and the
    /// deferred-free machinery record.
    ///
    /// `activate` runs only when the stream has to be created, so a warm
    /// launch costs a thread-local read and no driver call — which is what
    /// the `COMPUTE_STREAM` cell cost before this step.
    ///
    /// `CU_STREAM_DEFAULT` (flags = 0) is load-bearing: a blocking stream is
    /// an implicit two-way barrier against the legacy NULL stream, so the
    /// runtime's synchronous memcpys interleave in exactly the total order
    /// they did before the p8 stream migration.
    pub(crate) fn compute(&self, activate: impl FnOnce()) -> CUstream {
        self.with_slot(|s| named(&s.compute, 0, "compute stream", activate))
    }

    /// Lazily create and return this (thread, device)'s offload transfer
    /// stream.
    ///
    /// `CU_STREAM_NON_BLOCKING` (0x1): no implicit synchronisation with the
    /// legacy NULL stream, so a copy-back overlaps the next parameter's
    /// update kernels. Correctness ordering is per copy, via an event the
    /// transfer stream waits on.
    pub(crate) fn transfer(&self, activate: impl FnOnce()) -> CUstream {
        self.with_slot(|s| named(&s.transfer, 0x1, "offload transfer stream", activate))
    }

    /// This (thread, device)'s transfer stream if it was ever created, and
    /// `None` otherwise — the non-creating read the offload drain needs so
    /// that synchronising an unused stream is a no-op rather than a reason to
    /// build one.
    pub(crate) fn transfer_if_created(&self) -> Option<CUstream> {
        self.with_slot(|s| match s.transfer.get() {
            0 => None,
            raw => Some(raw as CUstream),
        })
    }

    /// Lazily create and return this (thread, device)'s inspect stream — the
    /// debug-copy stream the emitted inspect hooks issue on.
    pub(crate) fn inspect(&self, activate: impl FnOnce()) -> CUstream {
        self.with_slot(|s| named(&s.inspect, 0, "inspect stream", activate))
    }

    /// Borrow a non-blocking stream for a bounded piece of work.
    ///
    /// Reuses an idle one when the free list has it, so a steady-state
    /// prefetch/copy loop creates its stream once. Returned on drop.
    pub(crate) fn lease(&self) -> StreamLease<'_> {
        if let Some(raw) = self.free.lock().unwrap().pop() {
            return StreamLease {
                pool: self,
                stream: raw as CUstream,
            };
        }
        let mut stream: CUstream = std::ptr::null_mut();
        let r = unsafe { cuStreamCreate(&mut stream, 0x1) };
        if r != CUresult::CUDA_SUCCESS {
            crate::fatal::die(
                crate::fatal::Fatal::CudaDriver,
                &format!("cuStreamCreate (leased stream) failed: {r:?}"),
            );
        }
        self.all.lock().unwrap().push(stream as usize);
        StreamLease {
            pool: self,
            stream,
        }
    }

    /// Block the host until every stream this pool ever leased has drained.
    ///
    /// The counterpart to `inner::transfer_stream_synchronize` for work that
    /// went out on a lease instead of on a named stream: teardown paths that
    /// free a buffer a lease may still be copying into must call this, and it
    /// is a no-op when nothing was ever leased.
    pub(crate) fn synchronize_leases(&self) {
        let streams = self.all.lock().unwrap().clone();
        for raw in streams {
            let r = unsafe { cuStreamSynchronize(raw as CUstream) };
            if r != CUresult::CUDA_SUCCESS {
                crate::fatal::die(
                    crate::fatal::Fatal::CudaDriver,
                    &format!("cuStreamSynchronize (leased stream) failed: {r:?}"),
                );
            }
        }
    }

    /// Record a completion event on `stream`, taking one from the pool when
    /// there is a spare.
    ///
    /// The event is the caller's to hand to [`StreamPool::wait`] and then to
    /// [`StreamPool::recycle`]; dropping it on the floor leaks one driver
    /// event, as the open-coded `cuEventCreate` sites did before.
    pub(crate) fn record(&self, stream: CUstream) -> CUevent {
        let ev = match self.events.lock().unwrap().pop() {
            Some(raw) => raw as CUevent,
            None => {
                let mut ev: CUevent = std::ptr::null_mut();
                // 0x2 = CU_EVENT_DISABLE_TIMING (the cheapest flavour).
                let r = unsafe { cuEventCreate(&mut ev, 0x2) };
                if r != CUresult::CUDA_SUCCESS {
                    crate::fatal::die(
                        crate::fatal::Fatal::CudaDriver,
                        &format!("cuEventCreate (stream pool) failed: {r:?}"),
                    );
                }
                ev
            }
        };
        let r = unsafe { cuEventRecord(ev, stream) };
        if r != CUresult::CUDA_SUCCESS {
            crate::fatal::die(
                crate::fatal::Fatal::CudaDriver,
                &format!("cuEventRecord (stream pool) failed: {r:?}"),
            );
        }
        ev
    }

    /// Make `stream` wait for `ev` without blocking the host.
    pub(crate) fn wait(&self, stream: CUstream, ev: CUevent) {
        let r = unsafe { cuStreamWaitEvent(stream, ev, 0) };
        if r != CUresult::CUDA_SUCCESS {
            crate::fatal::die(
                crate::fatal::Fatal::CudaDriver,
                &format!("cuStreamWaitEvent (stream pool) failed: {r:?}"),
            );
        }
    }

    /// Return an event to the pool.
    ///
    /// Safe as soon as every wait that needed it has been *enqueued*, not
    /// only once they have completed: `cuStreamWaitEvent` captures the
    /// event's state at call time, so a later `cuEventRecord` on the same
    /// handle cannot retroactively change what an already-issued wait waits
    /// for. A caller that has neither waited nor synchronised must not
    /// recycle.
    pub(crate) fn recycle(&self, ev: CUevent) {
        self.events.lock().unwrap().push(ev as usize);
    }
}

/// Shared body of [`StreamPool::compute`] / `transfer` / `inspect`: read the
/// cell, create the stream on a miss.
///
/// `activate` runs ONLY on the miss, because that is when a live context is
/// needed. Calling it on every read would put a `cuCtxSetCurrent` on the
/// launch path, where before this step the hit was a thread-local read and
/// nothing else.
fn named(cell: &Cell<usize>, flags: u32, what: &str, activate: impl FnOnce()) -> CUstream {
    let cur = cell.get();
    if cur != 0 {
        return cur as CUstream;
    }
    activate();
    let mut stream: CUstream = std::ptr::null_mut();
    let r = unsafe { cuStreamCreate(&mut stream, flags) };
    if r != CUresult::CUDA_SUCCESS {
        crate::fatal::die(
            crate::fatal::Fatal::CudaDriver,
            &format!("cuStreamCreate ({what}) failed: {r:?}"),
        );
    }
    cell.set(stream as usize);
    stream
}

/// One slot per device. Sized 1 in this step; `cuDeviceGetCount()` in step 5.
static DEVICES: OnceLock<Box<[OnceLock<CudaContext>]>> = OnceLock::new();

/// How many registry slots exist. One until step 5 grows it.
const SLOTS: usize = 1;

fn slots() -> &'static [OnceLock<CudaContext>] {
    DEVICES.get_or_init(|| (0..SLOTS).map(|_| OnceLock::new()).collect::<Vec<_>>().into_boxed_slice())
}

/// The context in registry slot `slot`, initialising the device on first use.
///
/// Panics on a driver failure, exactly as the singleton's lazy init did — an
/// unusable CUDA device is not a recoverable condition for a call that has
/// already committed to running on the GPU. Callers that must *not* force
/// initialisation use [`initialized`] instead.
pub(crate) fn device(slot: usize) -> &'static CudaContext {
    let all = slots();
    assert!(
        slot < all.len(),
        "CUDA device slot {slot} out of range (registry has {} slot(s))",
        all.len()
    );
    all[slot].get_or_init(|| unsafe { init_device(slot) })
}

/// Run `f` against the calling thread's slot for registry slot `slot`,
/// creating the slot on first use.
///
/// The `RefCell` borrow is released *before* `f` runs, so a closure that
/// reaches back in (the `WS` realloc path calls `current_stream()`, and every
/// allocation made inside a workspace closure reads the pool selector) does
/// not hit a double borrow. The `Box` keeps the address stable for the
/// thread's life, and the higher-ranked closure bound stops the reference
/// escaping.
///
/// A free function rather than a `StreamPool` method because it needs no
/// context: the slot is plain thread-local bookkeeping and creating one makes
/// no driver call. [`with_placement`] depends on that.
fn with_thread_slot<R>(slot: usize, f: impl FnOnce(&ThreadSlot) -> R) -> R {
    let ptr = THREAD_SLOTS.with(|slots| slot_ptr(slots, slot));
    // SAFETY: the box was just created or already existed in this thread's
    // `THREAD_SLOTS`, is never moved out or replaced, and lives until the
    // thread exits — which cannot happen while this frame is on that
    // thread's stack.
    f(unsafe { &*ptr })
}

fn slot_ptr(slots: &RefCell<Vec<Option<Box<ThreadSlot>>>>, slot: usize) -> *const ThreadSlot {
    let mut v = slots.borrow_mut();
    if v.len() <= slot {
        v.resize_with(slot + 1, || None);
    }
    let entry = v[slot].get_or_insert_with(|| Box::new(ThreadSlot::new()));
    &raw const **entry
}

/// The registry slot [`current`] resolves to, computed without creating a
/// context. One slot in this step; step 5 reads the thread-current ordinal
/// here, and both [`current`] and [`with_placement`] follow it.
fn current_slot() -> usize {
    0
}

/// Run `f` against the calling thread's placement channel for the current
/// device. Roadmap A4 step 4b.
///
/// NON-FORCING, unlike [`current`]: no context is created, and that is
/// load-bearing. The pool selector is written by `nsl_gpu_set_*_pool`, which
/// codegen emits around every train block whatever device the program runs
/// on, and the arena's unbind row runs whether or not an arena exists. A
/// forcing door would make a cuda-featured binary running a CPU-only program
/// on a machine with no driver abort in `cuInit` — the failure
/// `initialized()` exists to prevent. The slot needs no context, so there is
/// nothing to force.
///
/// Once this thread's slots have been destroyed (thread-local teardown), `f`
/// runs against a fresh channel instead: the pool reads `Transient` and no
/// pin is armed, which is what the cells answered on a new thread, and a
/// write there is dropped with the thread. The cells this replaced were
/// const-initialised and never destroyed, so a late allocation in some other
/// thread-local's destructor could still read them; this keeps that working
/// rather than panicking inside a destructor.
pub(crate) fn with_placement<R>(f: impl FnOnce(&Placement) -> R) -> R {
    with_placement_in(current_slot(), f)
}

/// The registry slot [`with_placement`] reaches right now. A bracket that must
/// restore what it changed records this when it arms and hands it back to
/// [`with_placement_in`] when it restores, so the restore lands on the device
/// it armed even if the thread's device changed in between (`PoolGuard`).
/// Always 0 until step 5 makes the device switchable.
pub(crate) fn placement_slot() -> usize {
    current_slot()
}

/// [`with_placement`] for an explicit registry slot, as recorded by
/// [`placement_slot`]. Non-forcing and teardown-safe in the same way.
pub(crate) fn with_placement_in<R>(slot: usize, f: impl FnOnce(&Placement) -> R) -> R {
    match THREAD_SLOTS.try_with(|slots| slot_ptr(slots, slot)) {
        // SAFETY: as in `with_thread_slot`.
        Ok(ptr) => f(unsafe { &(*ptr).placement }),
        Err(_) => f(&Placement::new()),
    }
}

/// The context bound to the calling thread's current device.
///
/// One slot in this step, so this is slot 0 and matches the singleton exactly.
/// Step 5 makes it read a thread-local ordinal that defaults, on first read,
/// to `select_device_ordinal()` — so a process that never calls
/// `nsl_cuda_set_device` keeps binding the device it binds today.
pub(crate) fn current() -> &'static CudaContext {
    device(current_slot())
}

/// Has any context been initialised yet?
///
/// Non-forcing, and that is the whole point: diagnostics paths (the
/// `NSL_PHASE_TIMING` device sync, the offload-pageable probe) must not
/// force-initialise CUDA, because [`device`]'s lazy init asserts on `cuInit`
/// failure — which would abort a pure-CPU run of a cuda-featured binary on a
/// GPU-less machine from inside an instrumentation path.
pub(crate) fn initialized() -> bool {
    DEVICES
        .get()
        .is_some_and(|all| all.iter().any(|slot| slot.get().is_some()))
}

/// Bind `slot` to a real device: `cuInit`, pick the ordinal, retain the
/// primary context, make it current.
///
/// # Safety
/// Calls the CUDA driver API. Must run at most once per slot, which
/// `OnceLock::get_or_init` guarantees.
unsafe fn init_device(slot: usize) -> CudaContext {
    unsafe {
        let result = cuInit(0);
        assert_eq!(result, CUresult::CUDA_SUCCESS, "cuInit failed: {result:?}");

        // Step 1 has a single slot, so the ordinal is the one the singleton
        // computed. When the registry grows, the slot index *is* the ordinal.
        let ordinal = if SLOTS == 1 {
            super::inner::select_device_ordinal()
        } else {
            slot as i32
        };

        let mut dev: CUdevice = 0;
        let result = cuDeviceGet(&mut dev, ordinal);
        assert_eq!(
            result,
            CUresult::CUDA_SUCCESS,
            "cuDeviceGet (ordinal {ordinal}) failed: {result:?}"
        );

        let mut primary: CUcontext = std::ptr::null_mut();
        let result = cuDevicePrimaryCtxRetain(&mut primary, dev);
        assert_eq!(
            result,
            CUresult::CUDA_SUCCESS,
            "cuDevicePrimaryCtxRetain failed: {result:?}"
        );

        let result = cuCtxSetCurrent(primary);
        assert_eq!(
            result,
            CUresult::CUDA_SUCCESS,
            "cuCtxSetCurrent failed: {result:?}"
        );

        // Process-wide side effects of the first context coming up. They stay
        // in `inner` because the state they touch (the sync-mode flag, the
        // allocator's memstats hook) is process state, not device state.
        super::inner::on_first_context();

        CudaContext {
            device: dev,
            primary,
            streams: StreamPool::new(slot),
            modules: Mutex::new(ModuleCache {
                modules: HashMap::new(),
                funcs: HashMap::new(),
            }),
            frees: Mutex::new(VecDeque::new()),
            free_events: Mutex::new(Vec::new()),
            async_alloc: OnceLock::new(),
            async_allocs: Mutex::new(HashSet::new()),
            allocs: Mutex::new(HashSet::new()),
            cublas: OnceLock::new(),
            cublaslt: OnceLock::new(),
            lt_workspace: OnceLock::new(),
            caches: Mutex::new(HashMap::new()),
            arena: Region::new(),
            slab: Region::new(),
            // Pure: no driver call, so building it inside this lazy init
            // cannot re-enter `device()`.
            allocator: Mutex::new(super::caching_allocator::CachingAllocator::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slab and arena teardown rows must not bring a context into being.
    ///
    /// Roadmap A4 step 3c moved both regions onto the context, and the first
    /// cut of that reached them through `current()`, which creates one. The
    /// emitted `nsl_gpu_slab_destroy` runs at the end of EVERY program, a
    /// CPU-only one included, so a cuda-featured binary on a machine with no
    /// driver would have aborted at exit where it used to no-op — `device()`
    /// asserts on `cuInit` failure. Both accessors are gated on
    /// [`initialized`] for that reason, and this is the gate that says so.
    ///
    /// Stated as an implication so it is honest on a box that has a GPU: if a
    /// context already exists some other test made it, and there is nothing
    /// here to prove. On CI's CUDA lane, which has stub libraries and no
    /// device, it can never exist — which is exactly where the regression
    /// would bite.
    #[test]
    fn the_teardown_rows_do_not_create_a_context() {
        if initialized() {
            return;
        }
        crate::slab::nsl_gpu_slab_destroy();
        crate::transient_arena::nsl_arena_destroy();
        assert!(
            !initialized(),
            "a teardown row created a CUDA context; on a driverless machine \
             that is an abort at program exit, not a slow path"
        );
    }

    /// Roadmap A4 step 3d: the allocator moved onto the context, and the rows
    /// that only READ it — the `nsl_gpu_*` stats getters gates call, the
    /// per-step reset, the debug summary, the `NSL_MEMSTATS` report — must
    /// not create one. The old process-wide static never touched the driver,
    /// so a CPU-only run of a cuda binary could call them on a machine with
    /// no GPU; forcing a context there is a `cuInit` assertion. Each answers
    /// what a fresh allocator would. Stated as an implication, like the
    /// teardown test above.
    #[test]
    fn the_allocator_stats_rows_do_not_create_a_context() {
        if initialized() {
            return;
        }
        assert_eq!(crate::tensor::nsl_gpu_peak_allocated_bytes(), 0);
        assert_eq!(crate::tensor::nsl_gpu_cumulative_alloc_count(), 0);
        assert_eq!(crate::tensor::nsl_gpu_surface_peak_bytes(0), 0);
        assert_eq!(crate::tensor::nsl_gpu_surface_at_peak_bytes(0), 0);
        crate::tensor::nsl_gpu_reset_mem_stats();
        crate::tensor::nsl_debug_gpu_alloc_summary(0);
        crate::cuda::caching_allocator::print_memory_summary();
        assert!(crate::cuda::caching_allocator::allocator_if_initialized().is_none());
        assert!(
            !initialized(),
            "an allocator stats row created a CUDA context; on a driverless \
             machine that is an abort, where the old static answered zero"
        );
    }

    /// The registry is one slot in this step. When step 5 changes this, the
    /// `SLOTS == 1` branch in `init_device` must go with it — that branch is
    /// what keeps the SPMD spawner's ordinal striping working while there is
    /// only one slot to put it in.
    #[test]
    fn the_registry_has_one_slot_in_this_step() {
        assert_eq!(SLOTS, 1);
        assert_eq!(slots().len(), 1);
    }

    /// `initialized()` must not create a context: it is the probe that
    /// diagnostics use precisely because they must not force `cuInit` on a
    /// GPU-less machine, where `device()`'s lazy init would assert and abort
    /// an otherwise pure-CPU run from inside an instrumentation path.
    ///
    /// Asserted as the property rather than as "the registry is empty", so
    /// the test does not depend on which other test in this binary ran
    /// first: whatever the answer is, calling the probe — and touching the
    /// slot table it reads — must not change it.
    /// The slot bookkeeping is reachable without a device, which is the
    /// whole reason `with_slot` / `with_workspace` live on `StreamPool`
    /// rather than behind `CudaContext`'s driver handles: CPU CI can run
    /// them. A `StreamPool` touches the driver only in `lease`, `record`,
    /// `wait` and the three named-stream accessors — none of which these
    /// tests call.
    fn test_pool(slot: usize) -> StreamPool {
        StreamPool::new(slot)
    }

    #[derive(Default, PartialEq, Debug)]
    struct ProbeA(u64);
    #[derive(Default, PartialEq, Debug)]
    struct ProbeB(u64);

    /// A workspace is created once and then reused: the whole point of the
    /// cells this replaced.
    #[test]
    fn a_workspace_is_created_once_and_then_reused() {
        let pool = test_pool(0);
        assert_eq!(pool.with_workspace(|w: &mut ProbeA| w.0), 0, "not default");
        pool.with_workspace(|w: &mut ProbeA| w.0 = 7);
        assert_eq!(pool.with_workspace(|w: &mut ProbeA| w.0), 7, "not reused");
    }

    /// Two workspace types in one slot are two workspaces. Keying by
    /// `TypeId` is what lets `WS`, `SR_WS`, `CE_SCRATCH`, `MUON_STATS_BUF`
    /// and `WS_CACHE` share one map without this module naming any of them.
    #[test]
    fn workspace_types_do_not_collide() {
        let pool = test_pool(0);
        pool.with_workspace(|w: &mut ProbeA| w.0 = 1);
        pool.with_workspace(|w: &mut ProbeB| w.0 = 2);
        assert_eq!(pool.with_workspace(|w: &mut ProbeA| w.0), 1);
        assert_eq!(pool.with_workspace(|w: &mut ProbeB| w.0), 2);
    }

    /// Re-entrancy is the property the `Box` + released-borrow dance exists
    /// for: the real `WS` closure calls `current_stream()`, which reaches
    /// back into the same slot. A `RefCell` still held across `f` would
    /// panic here instead.
    #[test]
    fn a_workspace_closure_may_reach_back_into_its_slot() {
        let pool = test_pool(0);
        let inner = pool.with_workspace(|a: &mut ProbeA| {
            a.0 = 3;
            // Another workspace, and the slot's stream cells, from inside.
            let b = pool.with_workspace(|b: &mut ProbeB| {
                b.0 = 4;
                b.0
            });
            let created = pool.transfer_if_created();
            assert!(created.is_none(), "no driver call was made");
            a.0 + b
        });
        assert_eq!(inner, 7);
        assert_eq!(pool.with_workspace(|a: &mut ProbeA| a.0), 3);
    }

    /// Two registry slots are two sets of per-thread state — the device
    /// affinity this step adds on top of the thread affinity that was
    /// already there. With `SLOTS == 1` nothing in production exercises
    /// this yet; step 5 does, and this is what says it works.
    #[test]
    fn slots_do_not_share_workspaces() {
        let a = test_pool(0);
        let b = test_pool(3);
        a.with_workspace(|w: &mut ProbeA| w.0 = 11);
        b.with_workspace(|w: &mut ProbeA| w.0 = 22);
        assert_eq!(a.with_workspace(|w: &mut ProbeA| w.0), 11);
        assert_eq!(b.with_workspace(|w: &mut ProbeA| w.0), 22);
    }

    /// The capture state is per (thread, device), like everything else in
    /// the slot, and re-entrant: the capture hooks nest their accesses, and
    /// a slot borrow held across `f` would panic on the inner one.
    #[test]
    fn capture_state_is_per_slot_and_reentrant() {
        let a = test_pool(0);
        let b = test_pool(4);
        a.with_capture(|c| {
            c.nested_skip().set(2);
            // A nested reach into the same slot, then into another.
            assert_eq!(a.with_capture(|c| c.nested_skip().get()), 2);
            assert_eq!(b.with_capture(|c| c.nested_skip().get()), 0);
        });
        b.with_capture(|c| c.nested_skip().set(5));
        assert_eq!(a.with_capture(|c| c.nested_skip().get()), 2);
        assert_eq!(b.with_capture(|c| c.nested_skip().get()), 5);
        a.with_capture(|c| c.nested_skip().set(0));
        b.with_capture(|c| c.nested_skip().set(0));
    }

    /// The placement channel is per (thread, device) and reachable with no
    /// context at all. The second half is what the stub-library lane proves:
    /// there every `cuInit` fails, so a door that created a context would
    /// panic here rather than pass. Roadmap A4 step 4b.
    #[test]
    fn placement_is_per_slot_and_needs_no_context() {
        use super::super::caching_allocator::{get_alloc_pool, PoolGuard};

        assert_eq!(get_alloc_pool(), AllocPool::Transient, "a fresh thread starts Transient");
        {
            let _g = PoolGuard::new(AllocPool::Persistent);
            assert_eq!(with_placement(|p| p.pool.get()), AllocPool::Persistent);
            // Another device's slot on this thread is untouched...
            assert_eq!(
                with_thread_slot(current_slot() + 1, |s| s.placement.pool.get()),
                AllocPool::Transient
            );
            // ...and so is this device's slot on another thread.
            let other = std::thread::spawn(get_alloc_pool).join().unwrap();
            assert_eq!(other, AllocPool::Transient);
        }
        assert_eq!(get_alloc_pool(), AllocPool::Transient, "the guard restored through the slot");
    }

    /// A pool guard restores the slot it armed, not whichever slot is
    /// current when it drops. Nothing can switch the thread's device yet, so
    /// the guard is armed on a slot other than the current one directly —
    /// what a mid-bracket device switch would leave it holding. Preparation
    /// for roadmap A4 step 5.
    #[test]
    fn pool_guard_restores_the_slot_it_armed() {
        use super::super::caching_allocator::{get_alloc_pool, PoolGuard};

        let here = placement_slot();
        let there = here + 1;
        with_placement_in(there, |p| p.pool.set(AllocPool::Persistent));
        {
            let _g = PoolGuard::in_slot(there, AllocPool::Transient);
            assert_eq!(with_placement_in(there, |p| p.pool.get()), AllocPool::Transient);
            // The current slot is neither armed...
            assert_eq!(get_alloc_pool(), AllocPool::Transient);
            // ...nor restored into when the guard drops.
            with_placement(|p| p.pool.set(AllocPool::Persistent));
        }
        assert_eq!(
            with_placement_in(there, |p| p.pool.get()),
            AllocPool::Persistent,
            "the armed slot got its previous pool back"
        );
        assert_eq!(get_alloc_pool(), AllocPool::Persistent, "the current slot was left alone");
        with_placement(|p| p.pool.set(AllocPool::Transient));
        with_placement_in(there, |p| p.pool.set(AllocPool::Transient));
    }

    /// `transfer_if_created` must not create: it is what keeps
    /// `transfer_stream_synchronize` a no-op on a thread that never
    /// transferred, which is in turn what keeps the offload drain safe to
    /// call unconditionally.
    #[test]
    fn the_transfer_probe_does_not_create_a_stream() {
        let pool = test_pool(2);
        assert!(pool.transfer_if_created().is_none());
        assert!(pool.transfer_if_created().is_none());
    }

    /// A recycled event comes back out before a new one is made — the pool
    /// half of `record`, without the driver half.
    #[test]
    fn recycled_events_are_reused_before_new_ones() {
        let pool = test_pool(0);
        pool.recycle(0xbeef as CUevent);
        assert_eq!(pool.events.lock().unwrap().pop(), Some(0xbeef));
        assert!(pool.events.lock().unwrap().is_empty());
    }

    #[test]
    fn the_probe_does_not_force_initialisation() {
        let before = initialized();
        // Touching the slot table is allowed; filling a slot is not.
        let _ = slots();
        assert_eq!(initialized(), before, "reading the slot table initialised a device");
        for _ in 0..4 {
            assert_eq!(initialized(), before, "the probe initialised a device");
        }
    }
}
