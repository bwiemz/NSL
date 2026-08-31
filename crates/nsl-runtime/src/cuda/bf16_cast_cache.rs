//! Weight-operand cast cache for the BF16 matmul mode.
//!
//! `prepare_bf16_operands` casts BOTH GEMM operands f32 -> bf16 into fresh
//! scratch on every call. For activations that is the only correct thing —
//! the values are new each micro-batch. For parameters it is pure re-work:
//! theta changes only at the optimizer step, yet at grad_accum=8 each weight
//! is re-cast 8x per accumulation window in the forward, 8x more as `W^T` in
//! dgrad, every window. This module keeps ONE persistent bf16 image per
//! parameter and re-casts it only when the optimizer has actually moved the
//! weights.
//!
//! ## Identity, not inference
//!
//! The cache never guesses what is a parameter. A buffer becomes cacheable
//! ONLY because the fused AdamW step named it ([`note_param_stepped`]) — the
//! same call that makes its cached image stale. Everything else (activations,
//! gradients, workspace) misses and takes the fresh-scratch path unchanged.
//! Consequences of that rule:
//!
//! * Non-FASE update paths (interpreted axpy, Muon, bf16-sr mirrors, ZeRO
//!   slices, streamed working views) never register, so runs using them
//!   behave exactly as before this module existed.
//! * The FIRST accumulation window of a run misses everywhere — nothing has
//!   been registered yet. The cache warms at optimizer step 1.
//!
//! ## Staleness — the three ways theta's bytes can change
//!
//! 1. **The optimizer step**: [`note_param_stepped`] marks the entry invalid;
//!    the next GEMM re-casts into the SAME persistent buffer.
//! 2. **The buffer is freed** (model teardown, `to_device` migration): the
//!    [`on_free`] hook in `free_managed` evicts the entry BEFORE the pointer
//!    can be recycled — a caching-allocator block that comes back at the same
//!    address for unrelated data must not inherit a weight's bf16 image.
//! 3. **An in-place write** (`copy_data` into a live parameter — checkpoint
//!    restore into an existing model, `.copy_(...)`): the write sites call
//!    [`evict`] on their destination. `set_element` too, same reason.
//!
//! Anything that mutates device memory through a path that neither frees the
//! buffer nor goes through those externs (a future fused kernel writing theta
//! directly) MUST either call [`evict`]/[`note_param_stepped`] or not be a
//! parameter update. This is the module's one load-bearing assumption, and
//! `stepping_a_param_invalidates_its_cached_cast` is the gate that catches a
//! forgotten optimizer-side call.
//!
//! ## cuda-graphs
//!
//! A cache hit issues NO cast launches; a miss issues them into a pointer
//! that is STABLE for the life of the parameter. Capture state is keyed on
//! (region, occurrence % accum_window) and the optimizer runs at window
//! boundaries, so the miss lands in the same phase slot every window: slot 0
//! records cast+gemm, slots 1..N-1 record gemm alone, and each slot's digest
//! is self-consistent. The persistent pointer is strictly BETTER for replay
//! than the old scratch, whose digest stability leaned on the allocator
//! handing back the same block sequence.
//!
//! ## Concurrency
//!
//! All GPU work is single-threaded (the process invariant documented at the
//! launch path), so the map sees no concurrent GEMMs. The mutex is for the
//! free hook, which any thread may enter. Lock ordering: this module's lock
//! is always taken WITHOUT allocator locks held (the free hook runs at the
//! TOP of `free_managed`), and this module drops its lock before calling
//! `alloc_managed`/`free_managed` (whose OOM recovery can re-enter the free
//! hook — a nested probe on our own just-freed buffer must find nothing, not
//! a poisoned mutex).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

struct Entry {
    /// Element count of the parameter. A lookup with a different count (a
    /// sub-view GEMM, or a recycled pointer that slipped registration) is a
    /// MISS, not a resize — resizing on lookup would let a stale image
    /// masquerade under a new shape.
    elems: usize,
    /// Persistent bf16 image (`elems * 2` bytes), allocated on first use so
    /// parameters that never feed a bf16 GEMM (norm scales, biases) cost
    /// nothing. Null until then.
    buf16: u64,
    /// False = theta has moved since the image was last cast; the next
    /// acquire re-casts.
    valid: bool,
}

static CACHE: Mutex<Option<HashMap<u64, Entry>>> = Mutex::new(None);

/// Fast-path gate: false until the first registration, so runs outside BF16
/// mode (and the first window inside it) pay one relaxed load per free and
/// per GEMM operand, nothing more.
static ACTIVE: AtomicBool = AtomicBool::new(false);

// Test/diagnostic counters (relaxed; read by the gates and the teardown
// banner). `HITS` counts operand acquisitions served without a cast launch;
// `RECASTS` counts invalid->valid transitions (one per param per optimizer
// step, once warm).
static HITS: AtomicU64 = AtomicU64::new(0);
static RECASTS: AtomicU64 = AtomicU64::new(0);
static EVICTIONS: AtomicU64 = AtomicU64::new(0);

/// `NSL_MATMUL_BF16_CAST_CACHE=0` kills the cache (fresh scratch every GEMM,
/// the pre-cache behavior). Same read-once discipline as the mode resolvers.
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NSL_MATMUL_BF16_CAST_CACHE").ok().as_deref() != Some("0")
    })
}

/// The fused AdamW step just moved (or is about to move — order within the
/// step does not matter, no GEMM runs between the mark and the launch) the
/// parameter at `data`. Registers it on first sight, marks its image stale
/// always.
///
/// Called ONLY from theta-updating optimizer paths, with `data` a device
/// pointer to `elems` contiguous f32 elements. Gated internally on BF16 mode
/// + the kill switch so every other configuration stays zero-cost.
pub(crate) fn note_param_stepped(data: u64, elems: usize) {
    if data == 0 || elems == 0 || !enabled() {
        return;
    }
    if super::cublas_inner::resolved_math_mode() != super::cublas_inner::CublasMathMode::Bf16 {
        return;
    }
    let mut guard = CACHE.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    match map.get_mut(&data) {
        Some(e) if e.elems == elems => e.valid = false,
        Some(e) => {
            // Same pointer, different length: the old registration is for a
            // buffer that no longer means what it did (freed + recycled into
            // a differently-sized param without the free hook seeing it, or
            // a reshaped model). Replace wholesale; free the old image
            // outside the lock.
            let old = e.buf16;
            e.elems = elems;
            e.buf16 = 0;
            e.valid = false;
            drop(guard);
            if old != 0 {
                super::inner::free_managed(old as *mut std::ffi::c_void);
            }
            ACTIVE.store(true, Ordering::Release);
            return;
        }
        None => {
            map.insert(data, Entry { elems, buf16: 0, valid: false });
        }
    }
    drop(guard);
    ACTIVE.store(true, Ordering::Release);
}

/// Operand probe for `prepare_bf16_operands`. `Some((buf16, needs_cast))`
/// means the operand is a registered parameter: use `buf16` as the GEMM's
/// bf16 operand, DO NOT free it, and issue a cast into it first iff
/// `needs_cast`. `None` means take the fresh-scratch path.
///
/// Marks the entry valid on the `needs_cast` handoff: the caller launches
/// the cast unconditionally on that flag, and no other GEMM can interleave
/// (single-threaded GPU work).
pub(crate) fn acquire(data: u64, elems: usize) -> Option<(u64, bool)> {
    if !ACTIVE.load(Ordering::Acquire) {
        return None;
    }
    // First pass under the lock: identify, and detect a missing image.
    {
        let mut guard = CACHE.lock().unwrap();
        let map = guard.as_mut()?;
        let e = map.get_mut(&data)?;
        if e.elems != elems {
            return None;
        }
        if e.buf16 != 0 {
            let needs = !e.valid;
            e.valid = true;
            if needs {
                RECASTS.fetch_add(1, Ordering::Relaxed);
            } else {
                HITS.fetch_add(1, Ordering::Relaxed);
            }
            return Some((e.buf16, needs));
        }
    }
    // Image not allocated yet. Allocate WITHOUT the lock (alloc_managed's
    // OOM recovery may re-enter the free hook), then install. If the entry
    // vanished meanwhile (free hook on another thread), release the buffer.
    let fresh = super::inner::alloc_managed(elems * 2) as u64;
    if fresh == 0 {
        return None;
    }
    let mut guard = CACHE.lock().unwrap();
    let map = match guard.as_mut() {
        Some(m) => m,
        None => {
            drop(guard);
            super::inner::free_managed(fresh as *mut std::ffi::c_void);
            return None;
        }
    };
    match map.get_mut(&data) {
        Some(e) if e.elems == elems => {
            debug_assert_eq!(e.buf16, 0, "image raced into existence");
            e.buf16 = fresh;
            e.valid = true;
            RECASTS.fetch_add(1, Ordering::Relaxed);
            Some((fresh, true))
        }
        _ => {
            drop(guard);
            super::inner::free_managed(fresh as *mut std::ffi::c_void);
            None
        }
    }
}

/// Drop the entry for `data` (if any) and free its image. Called by the
/// in-place-write externs on their destination, and by [`on_free`].
pub(crate) fn evict(data: u64) {
    if !ACTIVE.load(Ordering::Acquire) || data == 0 {
        return;
    }
    let removed = {
        let mut guard = CACHE.lock().unwrap();
        guard.as_mut().and_then(|m| m.remove(&data))
    };
    if let Some(e) = removed {
        EVICTIONS.fetch_add(1, Ordering::Relaxed);
        if e.buf16 != 0 {
            // Free-after-enqueue is safe by the process stream invariant
            // (see Bf16Scratch::drop); the nested free-hook probe finds no
            // entry for the image pointer and falls straight through.
            super::inner::free_managed(e.buf16 as *mut std::ffi::c_void);
        }
    }
}

/// The `free_managed` hook: a device buffer is being returned to the
/// allocator. If it is a registered parameter, its image must die with it —
/// the allocator WILL hand this address out again.
#[inline]
pub(crate) fn on_free(ptr: *mut std::ffi::c_void) {
    if ACTIVE.load(Ordering::Acquire) {
        evict(ptr as u64);
    }
}

/// (hits, recasts, evictions) — see the counter docs.
pub(crate) fn stats() -> (u64, u64, u64) {
    (
        HITS.load(Ordering::Relaxed),
        RECASTS.load(Ordering::Relaxed),
        EVICTIONS.load(Ordering::Relaxed),
    )
}

/// Test-only reset: drop every entry and image, clear counters, disarm the
/// fast path. Lets gates in one process not see each other's registrations.
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn reset_for_test() {
    let drained: Vec<Entry> = {
        let mut guard = CACHE.lock().unwrap();
        guard.take().map(|m| m.into_values().collect()).unwrap_or_default()
    };
    ACTIVE.store(false, Ordering::Release);
    for e in drained {
        if e.buf16 != 0 {
            super::inner::free_managed(e.buf16 as *mut std::ffi::c_void);
        }
    }
    HITS.store(0, Ordering::Relaxed);
    RECASTS.store(0, Ordering::Relaxed);
    EVICTIONS.store(0, Ordering::Relaxed);
}
