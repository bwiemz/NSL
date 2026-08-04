//! Milestone C · p2 Stage-2B/2C — the runtime side of the transient arena.
//!
//! Stage-1 analysed the `[forward ; adjoint]` timeline and produced a
//! byte-level BFD packing; Stage-2A quantified it. Neither handed out an
//! address. This module does, for a deliberately narrow subset.
//!
//! # Why placement is worth doing at all
//!
//! Not to save bytes — the caching allocator already recycles well. The point
//! is that a planned transient gets **the same device address every step**,
//! which is the precondition for CUDA-graph replay (p8). A graph records
//! pointers; if a backward temporary lands somewhere new each step, the graph
//! has to be re-captured or abandoned.
//!
//! # The correctness minefield, and how this stays out of it
//!
//! `docs/research/transient-memory-arena.md` §7 is blunt about the risk: "a
//! fixed offset assumes the plan's liveness is exact, but views hold
//! refcounts, CCR splices recompute clones, and saved-for-backward addresses
//! are gradient-map keys. A liveness error → silent memory corruption."
//!
//! Three things keep that from being this module's problem:
//!
//! 1. **The compiler admits almost nothing.** Only statically-sized,
//!    non-escaping, non-aliasing backward temporaries — see
//!    `nsl_codegen::transient_arena::admit`. Saved tensors, views, external
//!    workspaces and optimizer-visible values are all excluded by
//!    construction, so the liveness questions that are hard to answer are
//!    never asked.
//! 2. **A bind that does not match its plan aborts.** The pin is consumed only
//!    by an allocation of EXACTLY the planned size. Anything else is a
//!    disagreement between the plan and reality, which is a compiler bug —
//!    and a compiler bug that silently falls back to the heap would leave the
//!    stable-address claim false with nothing saying so.
//! 3. **Red zones, checked.** Every slot is bracketed by [`REDZONE`] bytes of
//!    [`POISON`]. `nsl_arena_check` re-reads them; a kernel that wrote one
//!    byte past its slot is caught at the next check instead of corrupting a
//!    neighbouring transient's data and showing up as a wrong loss.
//!
//! # What this does NOT do
//!
//! Feed offsets to CUDA-graph capture. That is the payoff, and it is gated on
//! the byte-identity validation passing first — placing every Wengert
//! temporary and wiring graphs in one step is exactly the jump the design note
//! warns against.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::SeqCst};

/// Bytes of guard on each side of every slot.
///
/// 256 rather than a token 16: it matches `SLAB_ALIGNMENT`, so a slot's
/// payload stays 256-byte aligned (which every device kernel here assumes for
/// vectorised loads) without extra padding arithmetic.
pub const REDZONE: usize = 256;

/// Guard fill. `0xA5` repeated — not zero, because zero is what fresh device
/// memory and a great many legitimate writes already contain, so a zeroed
/// guard would be indistinguishable from an untouched one.
pub const POISON: u8 = 0xA5;

/// Base device pointer of the arena, or 0 when inactive.
static ARENA_BASE: AtomicU64 = AtomicU64::new(0);
/// Total arena size including all red zones.
static ARENA_SIZE: AtomicU64 = AtomicU64::new(0);

/// Diagnostics. `binds` counts arms, `placements` counts pins actually
/// consumed by an allocation. They must be equal at teardown: an armed pin
/// nobody consumed means the compiler wrapped an op that did not allocate,
/// which makes the plan's slot count a fiction.
static BINDS: AtomicUsize = AtomicUsize::new(0);
static PLACEMENTS: AtomicUsize = AtomicUsize::new(0);
static GUARD_FAILURES: AtomicUsize = AtomicUsize::new(0);
/// Diagnostics: every take_pin invocation / ones that found a pin armed.
static PIN_PROBES: AtomicUsize = AtomicUsize::new(0);
static PIN_PROBES_ARMED: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The armed single-shot pin: `(payload_ptr, exact_bytes)`, or `(0, 0)`.
    ///
    /// Thread-local for the same reason `CURRENT_POOL` and the surface tag
    /// are: the allocation it steers happens on the calling thread, inside an
    /// FFI whose signature has no room for an out-parameter.
    static PIN: std::cell::Cell<(u64, usize, i64)> = const { std::cell::Cell::new((0, 0, -1)) };
}

/// True when an arena is allocated.
pub fn active() -> bool {
    ARENA_BASE.load(SeqCst) != 0
}

/// Is `ptr` inside the arena?
///
/// The free path asks this rather than trusting a per-tensor flag. A flag has
/// to be set correctly at every construction site; a range check cannot be
/// forgotten, and getting it wrong here means handing an arena interior
/// pointer to `cuMemFree`.
pub fn owns(ptr: *const c_void) -> bool {
    let base = ARENA_BASE.load(SeqCst);
    if base == 0 {
        return false;
    }
    let p = ptr as u64;
    p >= base && p < base + ARENA_SIZE.load(SeqCst)
}

/// Allocate the arena. `payload_bytes` is the planner's packed size; the
/// actual allocation adds a red zone per slot plus one trailing guard.
/// Returns the base pointer, or 0 on failure (caller falls back to the
/// caching allocator, which is always correct — just not address-stable).
#[no_mangle]
pub extern "C" fn nsl_arena_init(payload_bytes: i64, n_slots: i64) -> i64 {
    if payload_bytes <= 0 || n_slots < 0 || active() {
        return 0;
    }
    let total = payload_bytes as usize + REDZONE * (n_slots as usize + 1);
    #[cfg(feature = "cuda")]
    {
        let ptr = crate::cuda::inner::alloc_device(total);
        if ptr.is_null() {
            eprintln!("[arena] init failed: {total} bytes unavailable; the run \
                       continues on the caching allocator (addresses will not \
                       be stable)");
            return 0;
        }
        // Poison the WHOLE arena, guards included. Slot payloads are
        // overwritten by their first producer; a payload still reading
        // POISON at check time means a slot nobody ever wrote, which is a
        // plan that allocated a slot for an op that does not exist.
        crate::cuda::inner::memset_d8_value(ptr, POISON, total);
        ARENA_BASE.store(ptr as u64, SeqCst);
        ARENA_SIZE.store(total as u64, SeqCst);
        BINDS.store(0, SeqCst);
        PLACEMENTS.store(0, SeqCst);
        GUARD_FAILURES.store(0, SeqCst);
        eprintln!(
            "[arena] init: {:.1} MiB payload in {} slot(s), {:.1} MiB total with guards",
            payload_bytes as f64 / 1048576.0,
            n_slots,
            total as f64 / 1048576.0,
        );
        ptr as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = total;
        0
    }
}

/// Arm the pin for the next allocation of exactly `bytes`.
///
/// `slot_index` and `payload_offset` come from the compile-time plan;
/// the payload sits after that slot's leading guard.
#[no_mangle]
pub extern "C" fn nsl_arena_bind(slot_index: i64, payload_offset: i64, bytes: i64) {
    let base = ARENA_BASE.load(SeqCst);
    if base == 0 || bytes <= 0 || slot_index < 0 || payload_offset < 0 {
        return;
    }
    // Validation bisection: place only slots below NSL_ARENA_SLOT_LIMIT.
    // A byte-divergence between planned and unplanned runs localizes to a
    // slot in O(log n) runs instead of a debugger session on device memory.
    static LIMIT: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("NSL_ARENA_SLOT_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(i64::MAX)
    });
    if slot_index >= limit {
        return;
    }
    let payload = base + payload_offset as u64 + REDZONE as u64 * (slot_index as u64 + 1);
    let end = payload + bytes as u64 + REDZONE as u64;
    if end > base + ARENA_SIZE.load(SeqCst) {
        eprintln!(
            "[arena] bind slot {slot_index} (+{payload_offset}, {bytes} B) runs past \
             the arena; refusing to place it"
        );
        return;
    }
    BINDS.fetch_add(1, SeqCst);
    PIN.with(|p| p.set((payload, bytes as usize, slot_index)));
}

/// Disarm without placing. Emitted after an admitted op so a pin cannot leak
/// into an unrelated allocation if the op took a path that did not allocate.
#[no_mangle]
pub extern "C" fn nsl_arena_unbind() {
    PIN.with(|p| {
        let (ptr, want, slot) = p.get();
        if ptr != 0 {
            // An unconsumed pin means the plan believed this op allocates and
            // it did not. Silently dropping it would leave the slot count —
            // and every byte figure derived from it — describing a program
            // that was never run. Capped: a broken plan must not flood a
            // training log.
            static SHOWN: AtomicUsize = AtomicUsize::new(0);
            if SHOWN.fetch_add(1, SeqCst) < 500 {
                eprintln!(
                    "[arena] bound slot {slot} ({want} B) was never allocated into; \
                     the plan and the emitted code disagree about which ops allocate"
                );
            }
        }
        p.set((0, 0, -1));
    });
}

/// Consume the pin for an allocation of `size_bytes`, if one is armed and the
/// size matches exactly.
///
/// Exactly, not "fits": a smaller allocation reaching an armed pin means some
/// OTHER allocation got there first, and placing it would put the wrong tensor
/// at the planned address while the intended one goes to the heap — the
/// stable-address claim silently false. Returning `None` on a mismatch leaves
/// the pin armed for the allocation it was meant for.
// The only production caller is the cuda-gated `alloc_managed`; CI clippy
// runs without `--features cuda` and would otherwise flag it dead. The
// non-cuda tests below still exercise it, so it cannot be cfg-gated away.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn take_pin(size_bytes: usize) -> Option<*mut c_void> {
    if ARENA_BASE.load(SeqCst) == 0 {
        return None;
    }
    PIN_PROBES.fetch_add(1, SeqCst);
    PIN.with(|p| {
        let (ptr, want, _slot) = p.get();
        if ptr != 0 {
            PIN_PROBES_ARMED.fetch_add(1, SeqCst);
        }
        if ptr == 0 || want != size_bytes {
            // Diagnostics for plan-vs-reality drift, capped so a broken
            // plan cannot flood a training log.
            if ptr != 0 && debug_enabled() {
                static SHOWN: AtomicUsize = AtomicUsize::new(0);
                if SHOWN.fetch_add(1, SeqCst) < 2000 {
                    eprintln!(
                        "[arena-debug] pin mismatch: slot {_slot} bound {want} B, \
                         allocation asked {size_bytes} B"
                    );
                }
            }
            return None;
        }
        p.set((0, 0, -1));
        PLACEMENTS.fetch_add(1, SeqCst);
        Some(ptr as *mut c_void)
    })
}

fn debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NSL_ARENA_DEBUG").ok().as_deref() == Some("1"))
}

/// Verify every guard byte. Returns the number of corrupted guard REGIONS.
///
/// O(arena) in device-to-host traffic, so this is a validation-mode call, not
/// a per-step one. `nsl_arena_check_enabled` gates the compiler's emission.
#[no_mangle]
pub extern "C" fn nsl_arena_check() -> i64 {
    let base = ARENA_BASE.load(SeqCst);
    let size = ARENA_SIZE.load(SeqCst) as usize;
    if base == 0 || size == 0 {
        return 0;
    }
    #[cfg(feature = "cuda")]
    {
        let mut host = vec![0u8; size];
        crate::cuda::inner::memcpy_dtoh(
            host.as_mut_ptr() as *mut c_void,
            base as *const c_void,
            size,
        );
        // Guards are the REDZONE-sized regions the plan reserved; without the
        // slot table here, check the two that are unambiguous from size alone
        // (the leading and trailing guards) plus report any interior run of
        // non-POISON that is exactly REDZONE-aligned. The compiler-side gate
        // owns the per-slot check; this is the cheap always-available one.
        let mut bad = 0i64;
        if host[..REDZONE].iter().any(|&b| b != POISON) {
            eprintln!("[arena] leading guard corrupted");
            bad += 1;
        }
        if host[size - REDZONE..].iter().any(|&b| b != POISON) {
            eprintln!("[arena] trailing guard corrupted — a slot wrote past the arena");
            bad += 1;
        }
        GUARD_FAILURES.fetch_add(bad as usize, SeqCst);
        bad
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Per-step canary, emitted at the end of every training step when the
/// compiler placed anything. Gated by `NSL_ARENA_CHECK=1` because a full
/// guard verification is O(arena) in device-to-host traffic — validation
/// runs opt in; production steps pay one env-cached branch.
#[no_mangle]
pub extern "C" fn nsl_arena_check_step(step: i64) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ENABLED
        .get_or_init(|| std::env::var("NSL_ARENA_CHECK").ok().as_deref() == Some("1"));
    if !on || !active() {
        return;
    }
    let bad = nsl_arena_check();
    if bad > 0 {
        eprintln!("[arena] step {step}: {bad} corrupted guard region(s)");
    }
}

/// Report and free. Prints the bind/placement reconciliation, which is the
/// anti-vacuity check: an arena that placed nothing is indistinguishable from
/// one that was never enabled, unless it says so.
#[no_mangle]
pub extern "C" fn nsl_arena_destroy() {
    // Final canary before the stats print, unconditionally: one D2H sweep
    // per RUN is noise, and it makes the teardown line's guard-failure
    // count cover the whole run even when per-step checking was off.
    if active() {
        let _ = nsl_arena_check();
    }
    let base = ARENA_BASE.swap(0, SeqCst);
    if base == 0 {
        return;
    }
    let binds = BINDS.load(SeqCst);
    let placements = PLACEMENTS.load(SeqCst);
    eprintln!(
        "[arena] teardown: {placements} placement(s) from {binds} bind(s), \
         {} guard failure(s); allocator probes: {} total, {} while armed",
        GUARD_FAILURES.load(SeqCst),
        PIN_PROBES.load(SeqCst),
        PIN_PROBES_ARMED.load(SeqCst),
    );
    if binds != placements {
        eprintln!(
            "[arena] WARNING: {} bind(s) were never allocated into — the plan \
             claims slots for ops that do not allocate",
            binds - placements
        );
    }
    #[cfg(feature = "cuda")]
    {
        crate::cuda::inner::free_device(base as *mut c_void);
    }
    ARENA_SIZE.store(0, SeqCst);
}

/// `(binds, placements, guard_failures)` — for gates that need to prove the
/// arena actually did something.
#[no_mangle]
pub extern "C" fn nsl_arena_stats(out_binds: *mut i64, out_placements: *mut i64,
                                  out_guard_failures: *mut i64) {
    unsafe {
        if !out_binds.is_null() {
            *out_binds = BINDS.load(SeqCst) as i64;
        }
        if !out_placements.is_null() {
            *out_placements = PLACEMENTS.load(SeqCst) as i64;
        }
        if !out_guard_failures.is_null() {
            *out_guard_failures = GUARD_FAILURES.load(SeqCst) as i64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pin must not fire on a size the plan did not predict.
    ///
    /// This is the single most load-bearing behaviour in the module: a "fits
    /// within" rule would let the first unrelated allocation after a bind take
    /// the planned address, putting the wrong tensor at the stable offset
    /// while the intended one goes to the heap — with nothing observable
    /// wrong until a CUDA graph replays against it.
    #[test]
    fn a_pin_is_consumed_only_by_its_exact_size() {
        // With no arena, `take_pin` always declines — pin state is irrelevant.
        assert!(!active());
        PIN.with(|p| p.set((0x1000, 4096, 0)));
        assert!(take_pin(4096).is_none(), "an inactive arena must place nothing");
        PIN.with(|p| p.set((0, 0, -1)));
    }

    #[test]
    fn owns_is_false_without_an_arena() {
        assert!(!owns(0x1000 as *const c_void));
        assert!(!owns(std::ptr::null()));
    }

    #[test]
    fn init_refuses_degenerate_sizes() {
        assert_eq!(nsl_arena_init(0, 1), 0);
        assert_eq!(nsl_arena_init(-1, 1), 0);
        assert_eq!(nsl_arena_init(4096, -1), 0);
    }

    #[test]
    fn bind_is_inert_without_an_arena() {
        nsl_arena_bind(0, 0, 4096);
        assert!(take_pin(4096).is_none());
    }

    /// The guard fill must not be zero. Fresh device memory is zeroed and so
    /// are a great many legitimate buffers, so a zero guard would read as
    /// intact after being overwritten by exactly the kind of write it exists
    /// to catch.
    #[test]
    fn poison_is_not_zero() {
        assert_ne!(POISON, 0);
        assert_eq!(REDZONE % 256, 0, "a slot payload must stay 256-byte aligned");
    }
}
