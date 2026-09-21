//! A device memory region: a base pointer and an extent, read on hot paths.
//!
//! Two of these exist — the compile-time-planned slab (`slab.rs`) and the
//! transient arena (`transient_arena.rs`). Both hold a device pointer, so
//! roadmap A4 step 3c moved them onto the per-device `CudaContext`; this type
//! is the storage they share, and it lives in a leaf module rather than in
//! `cuda::context` because `slab.rs` and `transient_arena.rs` are NOT
//! `cuda`-gated. Their `extern "C"` rows are part of the ABI in a CPU-only
//! build too (they answer 0 / inactive there), so the type they are written
//! in terms of has to compile without the `cuda` feature.
//!
//! ## Why plain atomics rather than the context's cache map
//!
//! [`contains`](Region::contains) is on `free_managed`'s hot path — every
//! device free asks whether the pointer is arena-interior before the
//! allocator may touch it — and `base` is read once per wrapped op by
//! `nsl_arena_bind`. A mutex there would be paid on every free; two relaxed
//! loads are not. This is the same distinction step 3b drew between a
//! workspace and a cache, one level down: state whose *reads* are hot wants
//! atomics, state whose reads are rare wants the lock.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering::SeqCst};

/// Base pointer and total extent of one device region, or `(0, 0)` when no
/// region is allocated. `base == 0` is the single authority on "inactive":
/// [`take`](Region::take) clears it first so a concurrent [`contains`](
/// Region::contains) sees an inactive region rather than a stale extent.
#[derive(Default)]
pub struct Region {
    base: AtomicU64,
    size: AtomicU64,
}

impl Region {
    pub const fn new() -> Self {
        Region { base: AtomicU64::new(0), size: AtomicU64::new(0) }
    }

    /// True once a region is allocated.
    pub fn active(&self) -> bool {
        self.base.load(SeqCst) != 0
    }

    /// The base pointer, or 0 when inactive.
    pub fn base(&self) -> u64 {
        self.base.load(SeqCst)
    }

    /// The total extent in bytes, 0 when inactive.
    pub fn size(&self) -> u64 {
        self.size.load(SeqCst)
    }

    /// Publish a freshly allocated region. Size first, then base: `base` is
    /// what every reader gates on, so it must not become visible ahead of the
    /// extent that bounds it.
    pub fn set(&self, base: u64, size: u64) {
        self.size.store(size, SeqCst);
        self.base.store(base, SeqCst);
    }

    /// Retire the region and return the base that was freed (0 if there was
    /// none, in which case the caller must not free anything). Clears `base`
    /// before `size` for the reason [`set`](Region::set) orders them the
    /// other way.
    pub fn take(&self) -> u64 {
        let base = self.base.swap(0, SeqCst);
        self.size.store(0, SeqCst);
        base
    }

    /// Is `ptr` inside this region?
    ///
    /// The free path asks this rather than trusting a per-tensor flag: a flag
    /// has to be set correctly at every construction site, a range check
    /// cannot be forgotten, and getting it wrong means handing a region
    /// interior pointer to `cuMemFree`.
    pub fn contains(&self, ptr: *const c_void) -> bool {
        let base = self.base.load(SeqCst);
        if base == 0 {
            return false;
        }
        let p = ptr as u64;
        p >= base && p < base + self.size.load(SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The range check, GPU-free. `transient_arena`'s own gates used to reach
    /// straight into its statics to fake an active region; with the region on
    /// the device context that is no longer writable from a CPU-only build,
    /// so the arithmetic is proved here instead of being lost.
    #[test]
    fn contains_is_half_open_and_inactive_owns_nothing() {
        let r = Region::new();
        assert!(!r.active());
        assert!(!r.contains(0x1000 as *const c_void));
        assert!(!r.contains(std::ptr::null()));

        r.set(0x1000, 0x100);
        assert!(r.active());
        assert!(r.contains(0x1000 as *const c_void), "base is inside");
        assert!(r.contains(0x10ff as *const c_void), "last byte is inside");
        assert!(!r.contains(0x1100 as *const c_void), "one past the end is outside");
        assert!(!r.contains(0xfff as *const c_void), "one before the base is outside");
        assert!(!r.contains(std::ptr::null()), "null is never inside");

        assert_eq!(r.take(), 0x1000, "take returns the base it retired");
        assert!(!r.active());
        assert_eq!(r.size(), 0, "take clears the extent too");
        assert_eq!(r.take(), 0, "a second take frees nothing");
    }
}
