//! Weight-operand cast cache for the BF16 matmul mode.
//!
//! `prepare_bf16_operands` casts BOTH GEMM operands f32 -> bf16 into fresh
//! scratch on every call. For activations that is the only correct thing —
//! the values are new each micro-batch. For parameters it is pure re-work:
//! theta changes only at the optimizer step, yet at grad_accum=8 each weight
//! is re-cast 8x per window in the forward and 8x more as `W^T` in dgrad.
//! This module keeps ONE persistent bf16 image per parameter and re-casts it
//! only when the optimizer has actually moved the weights.
//!
//! ## Identity, not inference
//!
//! The cache never guesses what is a parameter. A buffer becomes cacheable
//! ONLY because the fused AdamW step named it ([`note_param_stepped`]) — the
//! same call that makes its cached image stale — and the call sites name only
//! tensors that OWN their storage (`owns_data != 0`), are contiguous f32, and
//! live on device. That ownership gate is load-bearing twice over:
//!
//! * a weight-stream ARENA view (`upload_pack_inner` sets `owns_data = 0`)
//!   must never register — arena slots are recycled by raw DMA without
//!   passing `free_managed`, so a registered interior pointer would serve one
//!   layer's image for another layer's weights;
//! * a DLPack-imported tensor (`owns_data = 0`) must never register — its
//!   buffer dies in the FOREIGN allocator, invisible to the free hook, and
//!   the recycled address would inherit a dead parameter's image.
//!
//! Everything else (activations, gradients, Muon/ZeRO/streamed/interpreted
//! postures) misses and takes the fresh-scratch path unchanged. The FIRST
//! accumulation window of a run misses everywhere — nothing has been
//! registered yet; the cache warms at optimizer step 1.
//!
//! ## Compositions REFUSED (registration declines, whole run stays eager)
//!
//! * **`--cuda-graphs`**: a cache hit records no cast op, so the bf16 image
//!   address appears in NO verified record — only inside the captured
//!   graph's GemmEx node. A mid-window evict could then free memory a
//!   captured graph still reads, with `mismatches=0` (the exact HookOutcome
//!   failure shape, reincarnated); and any out-of-region GEMM (an in-run
//!   val forward) relocates the recast, burning capture attempts until
//!   regions go permanently eager. Until the cache can taint affected
//!   regions on evict, the two features do not compose — same rule as SR.
//! * **`NSL_MATMUL_BF16_ROUND=sr`**: rng_state.rs documents per-LAUNCH
//!   re-dithering as SR's contract (forward and dgrad deliberately draw
//!   independent dithers). A cached image would freeze one dither per step
//!   and share it across every GEMM in the window, changing the estimator
//!   PR #540 measured. SR runs therefore bypass the cache entirely.
//!
//! ## Staleness — how theta's bytes can change, and who tells the cache
//!
//! 1. **The optimizer step**: [`note_param_stepped`] marks the entry invalid;
//!    the next GEMM re-casts into the SAME persistent buffer.
//! 2. **The buffer is freed** (model teardown, `to_device` migration):
//!    `free_managed` calls [`evict`] at its TOP, before the allocator can
//!    recycle the address into unrelated data.
//! 3. **An addressed device write through the transport helpers**
//!    (`memcpy_htod{,_immediate,_async}`, `memcpy_dtod`, `memset_d8*`):
//!    those helpers probe [`evict`] on their destination, which covers
//!    checkpoint/model restore into a live model, `copy_data`, `zero_`, and
//!    every future transport-mediated writer automatically.
//! 4. **In-place writes that bypass transport** — a PTX kernel mutating its
//!    destination (`add_inplace`, `mul_scalar_inplace`, `scalar_mul_add`) or
//!    a host deref of managed memory (`set_element`, `slice_assign`) — carry
//!    explicit [`evict`] hooks at their externs.
//!
//! The remaining load-bearing assumption is only class 4: a FUTURE in-place
//! KERNEL that writes theta must call [`evict`]/[`note_param_stepped`] or
//! not target parameters. Check 3 of `zz_probe_child` in
//! `tests/matmul_bf16_cast_cache.rs` (driven by `cast_cache_gates_gpu`) is
//! the gate that catches a forgotten optimizer-side call.
//!
//! ## VRAM accounting
//!
//! Each cached image pins `2 bytes x elems` for the life of the parameter —
//! at 500M that is ~1 GiB, at 1B ~2 GiB — where the old scratch was
//! transient and pool-reused. The memory buys the throughput; it is NOT
//! free, and a run sized to the last GiB should set
//! `NSL_MATMUL_BF16_CAST_CACHE=0`. (Letting `alloc_managed`'s OOM recovery
//! drop images is a known follow-up: eviction at any moment is already
//! correct — the next acquire re-allocs and re-casts.)
//!
//! ## Concurrency
//!
//! All GPU work is single-threaded (the process invariant documented at the
//! launch path), so the map sees no concurrent GEMMs. The mutex is for the
//! free hook, which any thread may enter. Lock ordering: this module's lock
//! is always taken WITHOUT allocator locks held (the free hook runs at the
//! TOP of `free_managed`), and this module drops its lock before calling
//! `alloc_managed`/`free_managed` (whose OOM recovery can re-enter the free
//! hook — a nested probe on our own just-freed buffer must find nothing).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

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

static CACHE: LazyLock<Mutex<HashMap<u64, Entry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Fast-path gate: false until the first registration, so runs outside BF16
/// mode (and the first window inside it) pay one relaxed load per free and
/// per GEMM operand, nothing more.
static ACTIVE: AtomicBool = AtomicBool::new(false);

// Diagnostic counters (relaxed; read by the gates and the teardown line).
// `HITS` counts operand acquisitions served without a cast launch; `RECASTS`
// counts invalid->valid transitions (one per param per optimizer step, once
// warm); `EVICTIONS` counts entries dropped for ANY reason — free hook,
// write hook, or the same-pointer/different-length replacement in
// [`note_param_stepped`].
static HITS: AtomicU64 = AtomicU64::new(0);
static RECASTS: AtomicU64 = AtomicU64::new(0);
static EVICTIONS: AtomicU64 = AtomicU64::new(0);

/// `NSL_MATMUL_BF16_CAST_CACHE=0` kills the cache (fresh scratch every GEMM,
/// the pre-cache behavior). Same read-once discipline as the mode resolvers.
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::matmul_config::config().cast_cache)
}

/// The fused AdamW step just moved (or is about to move — order within the
/// step does not matter, no GEMM runs between the mark and the launch) the
/// parameter at `data`. Registers it on first sight, marks its image stale
/// always.
///
/// Call sites MUST pass only device tensors that own their storage (see the
/// module doc's ownership gate) — this function cannot check that from a
/// raw pointer. Gated internally on BF16 mode, the kill switch, and the
/// refused compositions (graphs, SR), so every other configuration stays
/// zero-cost and permanently un-armed.
pub(crate) fn note_param_stepped(data: u64, elems: usize) {
    if data == 0 || elems == 0 || !enabled() {
        return;
    }
    if super::cublas_inner::resolved_math_mode() != super::cublas_inner::CublasMathMode::Bf16 {
        return;
    }
    if super::cublas_inner::bf16_rounding() != super::cublas_inner::Bf16Rounding::Rne {
        return; // SR: per-launch dither is the contract; never cache.
    }
    if super::graph_capture::enabled() {
        // Hit-path GEMMs record no cast op, so a captured graph would be the
        // only holder of the image address — see the module doc. Refuse, and
        // say so once: a silent refusal would read as "the cache is broken"
        // in an A/B that composed the two.
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[bf16-cast-cache] disabled: --cuda-graphs is active (a cached \
                 image's address is pinned only inside captured graphs; evict \
                 would free memory a replay still reads). Use one or the other."
            );
        }
        return;
    }
    let old = {
        let mut map = CACHE.lock().unwrap();
        match map.get_mut(&data) {
            Some(e) if e.elems == elems => {
                e.valid = false;
                0
            }
            _ => {
                // New registration, or the same pointer reincarnated with a
                // different length (freed + recycled into a differently-sized
                // param without the free hook seeing it, or a reshaped
                // model). Replace wholesale; the displaced image is an
                // eviction and is freed outside the lock.
                map.insert(data, Entry { elems, buf16: 0, valid: false })
                    .map_or(0, |e| e.buf16)
            }
        }
    };
    if old != 0 {
        EVICTIONS.fetch_add(1, Ordering::Relaxed);
        super::inner::free_managed(old as *mut std::ffi::c_void);
    }
    ACTIVE.store(true, Ordering::Release);
}

/// Operand probe for `prepare_bf16_operands`. `Some((buf16, needs_cast))`
/// means the operand is a registered parameter: use `buf16` as the GEMM's
/// bf16 operand, DO NOT free it, and issue a cast into it first iff
/// `needs_cast`. `None` means take the fresh-scratch path.
///
/// Marks the entry valid on the `needs_cast` handoff. That is coherent only
/// because the caller launches the cast unconditionally on the flag with no
/// bail-out in between, and no other GEMM can interleave (single-threaded
/// GPU work) — any future early return between `acquire` and the cast
/// launch would strand a permanently-valid stale image, so keep the launch
/// adjacent to the probe.
pub(crate) fn acquire(data: u64, elems: usize) -> Option<(u64, bool)> {
    if !ACTIVE.load(Ordering::Acquire) {
        return None;
    }
    // First pass under the lock: identify, and detect a missing image.
    {
        let mut map = CACHE.lock().unwrap();
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
    // changed meanwhile (free hook, replacement), release the fresh buffer
    // rather than trusting a stale view of the entry.
    let fresh = super::inner::alloc_managed(elems * 2) as u64;
    if fresh == 0 {
        return None;
    }
    let install = {
        let mut map = CACHE.lock().unwrap();
        match map.get_mut(&data) {
            Some(e) if e.elems == elems && e.buf16 == 0 => {
                e.buf16 = fresh;
                e.valid = true;
                true
            }
            _ => false,
        }
    };
    if install {
        RECASTS.fetch_add(1, Ordering::Relaxed);
        Some((fresh, true))
    } else {
        super::inner::free_managed(fresh as *mut std::ffi::c_void);
        None
    }
}

/// Drop the entry for `data` (if any) and free its image. Wired into
/// `free_managed`'s top, the device-write transport helpers, and the
/// in-place-write externs; self-gating (one relaxed load) when the cache
/// never armed, so hot paths pay nothing outside BF16 mode.
#[inline]
pub(crate) fn evict(data: u64) {
    if !ACTIVE.load(Ordering::Acquire) || data == 0 {
        return;
    }
    let removed = CACHE.lock().unwrap().remove(&data);
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
        let mut map = CACHE.lock().unwrap();
        std::mem::take(&mut *map).into_values().collect()
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
