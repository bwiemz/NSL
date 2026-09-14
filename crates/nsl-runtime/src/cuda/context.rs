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

use cudarc::driver::sys::*;
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
/// Lock order within a context: **modules before allocator**, the same rule
/// the singleton documented as "CUDA_STATE before CACHING_ALLOCATOR". Two
/// contexts are never locked together — that is the invariant that makes a
/// second device safe, and step 5 depends on it.
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

/// The context bound to the calling thread's current device.
///
/// One slot in this step, so this is slot 0 and matches the singleton exactly.
/// Step 5 makes it read a thread-local ordinal that defaults, on first read,
/// to `select_device_ordinal()` — so a process that never calls
/// `nsl_cuda_set_device` keeps binding the device it binds today.
pub(crate) fn current() -> &'static CudaContext {
    device(0)
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
            modules: Mutex::new(ModuleCache {
                modules: HashMap::new(),
                funcs: HashMap::new(),
            }),
            frees: Mutex::new(VecDeque::new()),
            free_events: Mutex::new(Vec::new()),
            async_alloc: OnceLock::new(),
            async_allocs: Mutex::new(HashSet::new()),
            allocs: Mutex::new(HashSet::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
