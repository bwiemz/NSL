//! M56 v1 pool of pipeline-execution contexts. Spec §3.3, §1.9.
//!
//! Invariant: "every instance handed out is either in post-reset state or
//! has been removed from the pool." On reset failure, the context's slot
//! becomes `None` (tombstone) and `effective_size` decrements. v1 does
//! NOT attempt replacement; the pool capacity erodes with accumulated
//! reset failures. Replacement is v2+ scope.

use std::collections::VecDeque;

use crate::agent::scheduler::ReactorScheduler;

#[derive(Debug)]
pub enum AcquireError {
    Exhausted,
    /// Reserved for v2 timeout semantics; v1 acquire is non-blocking.
    Timeout,
}

pub struct PipelineContext {
    pub scheduler: ReactorScheduler,
    /// Test-only flag so tests can simulate reset failure.
    #[cfg(test)]
    pub _test_reset_fails: bool,
}

impl PipelineContext {
    pub fn new(scheduler: ReactorScheduler) -> Self {
        Self {
            scheduler,
            #[cfg(test)]
            _test_reset_fails: false,
        }
    }

    #[cfg(test)]
    pub fn new_test() -> Self {
        Self::new(ReactorScheduler::new())
    }

    #[cfg(test)]
    pub fn new_test_failing_reset() -> Self {
        let mut c = Self::new_test();
        c._test_reset_fails = true;
        c
    }

    /// Returns Ok(()) on success; Err on failure. Failure causes the pool
    /// to tombstone this instance (see PipelineContextPool::release).
    pub fn reset(&mut self) -> Result<(), String> {
        #[cfg(test)]
        if self._test_reset_fails {
            return Err("simulated reset failure".into());
        }
        // Production agents: zero buffers, reinit KV-caches, drain mailboxes.
        // For v1 we only have a scheduler with no persistent state to reset
        // — this is a stub. Tasks 17–19 will populate it as agents grow
        // real state.
        Ok(())
    }
}

/// A lease handle returned by `acquire`. Linear resource: must be
/// consumed by `release` or `release_by_index`. v1 has no `Drop` impl —
/// if a `Lease` is dropped without calling release, the pool slot leaks
/// (remains permanently in the leased state and cannot be reacquired).
/// Callers are responsible for ensuring every acquired lease reaches a
/// release call. v2 may add a `Drop` impl that tombstones the slot, but
/// v1's contract is explicit-release-only.
pub struct Lease {
    index: usize,
}

impl Lease {
    pub fn index(&self) -> usize {
        self.index
    }
}

pub struct PipelineContextPool {
    /// Fixed-length Vec sized at construction. Entries become `None` (tombstoned)
    /// on reset failure and are never repopulated in v1.
    contexts: Vec<Option<PipelineContext>>,
    /// Indices of currently-available (non-leased, non-tombstoned) contexts.
    available: VecDeque<usize>,
    /// Current effective pool size — decrements on reset failure (tombstoning).
    /// Starts equal to pool_size; monotonically non-increasing in v1.
    size: usize,
}

impl PipelineContextPool {
    /// Construct a pool with `pool_size` contexts. Panics on construction
    /// failure; for fallible construction use `try_new`.
    pub fn new<F: FnMut() -> PipelineContext>(
        pool_size: usize,
        constructor: F,
    ) -> Self {
        Self::try_new(pool_size, constructor).expect("pool construction failed")
    }

    /// Fallible constructor. Returns `Err` if `pool_size` exceeds the v1
    /// hard limit (likely OOM territory). Spec §1.7: pool-size-too-large
    /// is a serve-block-construction error, not a runtime error.
    pub fn try_new<F: FnMut() -> PipelineContext>(
        pool_size: usize,
        mut constructor: F,
    ) -> Result<Self, String> {
        const HARD_LIMIT: usize = 16384;
        if pool_size > HARD_LIMIT {
            return Err(format!(
                "pool_size={} exceeds v1 hard limit ({}); likely would OOM. \
                 Production deployments should query OS/GPU memory before \
                 setting pool_size.",
                pool_size, HARD_LIMIT,
            ));
        }
        let mut contexts = Vec::with_capacity(pool_size);
        let mut available = VecDeque::with_capacity(pool_size);
        for i in 0..pool_size {
            contexts.push(Some(constructor()));
            available.push_back(i);
        }
        Ok(Self { contexts, available, size: pool_size })
    }

    /// Number of currently-available contexts (non-leased, non-tombstoned).
    pub fn available_count(&self) -> usize {
        self.available.len()
    }

    /// Current effective pool size. Decreases when reset failures tombstone
    /// contexts; never increases in v1.
    pub fn effective_size(&self) -> usize {
        self.size
    }

    /// Acquire a lease. Returns `Err(Exhausted)` if no context is available.
    pub fn acquire(&mut self) -> Result<Lease, AcquireError> {
        match self.available.pop_front() {
            Some(idx) => Ok(Lease { index: idx }),
            None => Err(AcquireError::Exhausted),
        }
    }

    /// Release a lease — run reset; on success, return the index to the
    /// available queue. On failure, tombstone the slot (set to None),
    /// decrement `size`, log the diagnostic, and do NOT re-add the index.
    pub fn release(&mut self, lease: Lease) {
        let idx = lease.index;
        let Some(slot) = self.contexts.get_mut(idx) else { return };
        let Some(ctx) = slot.as_mut() else {
            // Already tombstoned by a prior path; nothing to do.
            return;
        };
        match ctx.reset() {
            Ok(()) => {
                self.available.push_back(idx);
            }
            Err(reason) => {
                eprintln!(
                    "[m56 pool] reset failure on context {}: {}. \
                     Slot tombstoned; effective pool size now {}.",
                    idx,
                    reason,
                    self.size.saturating_sub(1),
                );
                *slot = None;
                self.size = self.size.saturating_sub(1);
            }
        }
    }

    /// FFI helper: release by raw index. Used by Task 16's FFI surface
    /// where the lease cannot be a Rust struct (it must be an i64 lease id).
    pub fn release_by_index(&mut self, idx: usize) {
        self.release(Lease { index: idx });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_default_size_one_acquire_release() {
        let mut pool = PipelineContextPool::new(1, PipelineContext::new_test);
        assert_eq!(pool.effective_size(), 1);
        assert_eq!(pool.available_count(), 1);
        let lease = pool.acquire().expect("acquire failed");
        assert_eq!(pool.available_count(), 0);
        pool.release(lease);
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.effective_size(), 1);
    }

    #[test]
    fn pool_concurrent_leases_up_to_size() {
        let mut pool = PipelineContextPool::new(4, PipelineContext::new_test);
        let l1 = pool.acquire().unwrap();
        let l2 = pool.acquire().unwrap();
        let l3 = pool.acquire().unwrap();
        let l4 = pool.acquire().unwrap();
        assert!(matches!(pool.acquire(), Err(AcquireError::Exhausted)),
            "5th acquire should fail with Exhausted");
        pool.release(l1); pool.release(l2); pool.release(l3); pool.release(l4);
        assert_eq!(pool.available_count(), 4);
    }

    #[test]
    fn pool_tombstones_on_reset_failure_and_does_not_replace() {
        let mut pool = PipelineContextPool::new(2, PipelineContext::new_test_failing_reset);
        let l1 = pool.acquire().unwrap();
        let l2 = pool.acquire().unwrap();
        pool.release(l1);
        assert_eq!(pool.effective_size(), 1, "after first reset failure, effective size = 1");
        pool.release(l2);
        assert_eq!(pool.effective_size(), 0, "after second reset failure, effective size = 0");
        // Capacity does NOT bounce back in v1 (tombstone + lazy replacement).
        assert!(matches!(pool.acquire(), Err(AcquireError::Exhausted)),
            "acquire after total tombstoning should fail");
    }

    #[test]
    fn pool_oversize_fails_at_construction() {
        let r = PipelineContextPool::try_new(usize::MAX, PipelineContext::new_test);
        assert!(r.is_err(), "memory-exceeding pool_size should fail at construction");
    }

    #[test]
    fn pool_release_by_index_works() {
        // FFI helper — the production codegen will call this.
        let mut pool = PipelineContextPool::new(2, PipelineContext::new_test);
        let l1 = pool.acquire().unwrap();
        let idx = l1.index();
        std::mem::forget(l1); // simulate FFI ownership (lease becomes raw index)
        pool.release_by_index(idx);
        assert_eq!(pool.available_count(), 2);
    }
}
