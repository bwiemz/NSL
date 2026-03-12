# Fuzz Harness for Memory Lifecycle — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic fuzz harness that generates random sequences of ~20 FFI tensor operations (including autodiff tape), executes them, and asserts that global allocation counters return to zero — catching both memory leaks and crashes.

**Architecture:** 8 atomic counters in `memory.rs` behind `#[cfg(test)]` track CPU/GPU alloc/free counts and bytes. A new `fuzz.rs` module generates seeded random op sequences against a live tensor pool with a tape state machine, then asserts counter balance after teardown. No external dependencies.

**Tech Stack:** Rust std only (`std::sync::atomic`, `rand` via manual xorshift — no new deps)

---

## File Map

| File | Responsibility |
|------|---------------|
| `crates/nsl-runtime/src/memory.rs` | Add `#[cfg(test)] pub mod stats` (8 counters + reset). Instrument `checked_alloc`, `checked_alloc_zeroed`, `checked_free`. |
| `crates/nsl-runtime/src/cuda/mod.rs` | Instrument `alloc_managed`, `free_managed` with CUDA counters. |
| `crates/nsl-runtime/src/fuzz.rs` | New file: `FuzzOp` enum, `FuzzState`, sequence generator, executor, assertions. |
| `crates/nsl-runtime/src/lib.rs` | Add `#[cfg(test)] mod fuzz;` |

---

## Chunk 1: Allocation Counters + Instrumentation

### Task 1: Add stats module to memory.rs

**Files:**
- Modify: `crates/nsl-runtime/src/memory.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/nsl-runtime/src/fuzz.rs` with a minimal test that references `stats`:

```rust
// crates/nsl-runtime/src/fuzz.rs
#[cfg(test)]
mod tests {
    use crate::memory::stats;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_stats_counter_reset() {
        stats::reset();
        assert_eq!(stats::ALLOC_COUNT.load(Ordering::SeqCst), 0);
        assert_eq!(stats::FREE_COUNT.load(Ordering::SeqCst), 0);
        assert_eq!(stats::ALLOC_BYTES.load(Ordering::SeqCst), 0);
        assert_eq!(stats::FREE_BYTES.load(Ordering::SeqCst), 0);
        assert_eq!(stats::CUDA_ALLOC_COUNT.load(Ordering::SeqCst), 0);
        assert_eq!(stats::CUDA_FREE_COUNT.load(Ordering::SeqCst), 0);
        assert_eq!(stats::CUDA_ALLOC_BYTES.load(Ordering::SeqCst), 0);
        assert_eq!(stats::CUDA_FREE_BYTES.load(Ordering::SeqCst), 0);
    }
}
```

- [ ] **Step 2: Register fuzz module in lib.rs**

Add to `crates/nsl-runtime/src/lib.rs` after the last module declaration:

```rust
#[cfg(test)]
mod fuzz;
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p nsl-runtime test_stats_counter_reset`
Expected: FAIL — `stats` module does not exist yet.

- [ ] **Step 4: Implement stats module**

Add to the **end** of `crates/nsl-runtime/src/memory.rs`:

```rust
/// Allocation statistics for fuzz testing. Only compiled in test builds.
#[cfg(test)]
pub mod stats {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static FREE_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
    pub static FREE_BYTES: AtomicUsize = AtomicUsize::new(0);

    pub static CUDA_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static CUDA_FREE_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static CUDA_ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
    pub static CUDA_FREE_BYTES: AtomicUsize = AtomicUsize::new(0);

    pub fn reset() {
        for counter in [
            &ALLOC_COUNT, &FREE_COUNT, &ALLOC_BYTES, &FREE_BYTES,
            &CUDA_ALLOC_COUNT, &CUDA_FREE_COUNT, &CUDA_ALLOC_BYTES, &CUDA_FREE_BYTES,
        ] {
            counter.store(0, Ordering::SeqCst);
        }
    }

    pub fn cpu_alloc(size: usize) {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        ALLOC_BYTES.fetch_add(size, Ordering::SeqCst);
    }

    pub fn cpu_free(size: usize) {
        FREE_COUNT.fetch_add(1, Ordering::SeqCst);
        FREE_BYTES.fetch_add(size, Ordering::SeqCst);
    }

    pub fn cuda_alloc(size: usize) {
        CUDA_ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        CUDA_ALLOC_BYTES.fetch_add(size, Ordering::SeqCst);
    }

    pub fn cuda_free(size: usize) {
        CUDA_FREE_COUNT.fetch_add(1, Ordering::SeqCst);
        CUDA_FREE_BYTES.fetch_add(size, Ordering::SeqCst);
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p nsl-runtime test_stats_counter_reset`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/memory.rs crates/nsl-runtime/src/fuzz.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(fuzz): add allocation stats counters module"
```

---

### Task 2: Instrument checked_alloc and checked_free

**Files:**
- Modify: `crates/nsl-runtime/src/memory.rs`

- [ ] **Step 1: Write the failing test**

Add to `crates/nsl-runtime/src/fuzz.rs` tests module:

```rust
#[test]
fn test_stats_track_alloc_free() {
    stats::reset();

    let ptr = crate::memory::checked_alloc(256);
    assert_eq!(stats::ALLOC_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(stats::ALLOC_BYTES.load(Ordering::SeqCst), 256);
    assert_eq!(stats::FREE_COUNT.load(Ordering::SeqCst), 0);

    unsafe { crate::memory::checked_free(ptr, 256); }
    assert_eq!(stats::FREE_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(stats::FREE_BYTES.load(Ordering::SeqCst), 256);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-runtime test_stats_track_alloc_free`
Expected: FAIL — counters stay at 0 because `checked_alloc`/`checked_free` don't bump them yet.

- [ ] **Step 3: Instrument checked_alloc**

In `crates/nsl-runtime/src/memory.rs`, modify `checked_alloc` (line ~28). Add after the null check on the returned pointer and before returning `ptr`:

```rust
pub(crate) fn checked_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        eprintln!("nsl: out of memory");
        std::process::abort();
    }
    #[cfg(test)]
    stats::cpu_alloc(size);
    ptr
}
```

Apply the same pattern to `checked_alloc_zeroed` (line ~42):

```rust
pub(crate) fn checked_alloc_zeroed(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        eprintln!("nsl: out of memory");
        std::process::abort();
    }
    #[cfg(test)]
    stats::cpu_alloc(size);
    ptr
}
```

- [ ] **Step 4: Instrument checked_free**

In `crates/nsl-runtime/src/memory.rs`, modify `checked_free` (line ~70):

```rust
pub(crate) unsafe fn checked_free(ptr: *mut u8, size: usize) {
    if !ptr.is_null() && size > 0 {
        #[cfg(test)]
        stats::cpu_free(size);
        let layout = Layout::from_size_align(size, 8).unwrap();
        unsafe { dealloc(ptr, layout) };
    }
}
```

- [ ] **Step 5: Instrument checked_realloc**

`checked_realloc` (line ~56) performs a free+alloc internally. The old size must be freed and the new size allocated in the counters:

```rust
pub(crate) unsafe fn checked_realloc(ptr: *mut u8, old_size: usize, new_size: usize) -> *mut u8 {
    if ptr.is_null() {
        return checked_alloc(new_size);
    }
    #[cfg(test)]
    {
        stats::cpu_free(old_size);
        stats::cpu_alloc(new_size);
    }
    let old_layout = Layout::from_size_align(old_size, 8).unwrap();
    let new_ptr = unsafe { std::alloc::realloc(ptr, old_layout, new_size) };
    if new_ptr.is_null() {
        eprintln!("nsl: out of memory");
        std::process::abort();
    }
    new_ptr
}
```

**Important:** `checked_realloc` calls `checked_alloc` when `ptr.is_null()`, so `checked_alloc`'s instrumentation handles that branch. The realloc branch must do its own free+alloc accounting since `std::alloc::realloc` doesn't go through our `checked_alloc`/`checked_free`.

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test -p nsl-runtime test_stats_track_alloc_free`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-runtime/src/memory.rs crates/nsl-runtime/src/fuzz.rs
git commit -m "feat(fuzz): instrument checked_alloc/free with stats counters"
```

---

### Task 3: Instrument CUDA alloc/free

**Files:**
- Modify: `crates/nsl-runtime/src/cuda/mod.rs`

**Note:** CUDA counters can only be tested on machines with CUDA hardware. The instrumentation is straightforward and mirrors the CPU pattern. This task adds the counter bumps; testing is deferred to machines with GPUs or skipped in CI.

- [ ] **Step 1: Instrument alloc_managed**

In `crates/nsl-runtime/src/cuda/mod.rs`, inside the `#[cfg(feature = "cuda")] pub(crate) mod inner` block, modify `alloc_managed` (line ~87). Add after the assert and before returning `ptr`:

```rust
pub(crate) fn alloc_managed(size_bytes: usize) -> *mut c_void {
    ensure_context();
    unsafe {
        let mut ptr: CUdeviceptr = 0;
        let result = cuMemAllocManaged(
            &mut ptr,
            size_bytes,
            CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as u32,
        );
        assert_eq!(
            result,
            CUresult::CUDA_SUCCESS,
            "cuMemAllocManaged({} bytes) failed: {:?}",
            size_bytes,
            result
        );
        #[cfg(test)]
        crate::memory::stats::cuda_alloc(size_bytes);
        ptr as *mut c_void
    }
}
```

- [ ] **Step 2: Instrument free_managed**

Modify `free_managed` (line ~108). The challenge: `free_managed` doesn't know the size. We need to track it.

**Option A (simple):** Add a thread-local `HashMap<usize, usize>` mapping `ptr_addr → size` inside the `inner` module. Record on alloc, look up on free.

**Option B (simpler):** Only track count, not bytes, for CUDA. This is acceptable since GPU tensor sizes are always `len * 4` (f32) which the fuzz harness knows.

**Use Option A** for completeness — add a size registry:

```rust
// Inside the inner module, add after the CUDA_STATE static:
#[cfg(test)]
use std::collections::HashMap as TestHashMap;
#[cfg(test)]
static CUDA_SIZE_REGISTRY: std::sync::OnceLock<std::sync::Mutex<TestHashMap<usize, usize>>> = std::sync::OnceLock::new();

#[cfg(test)]
fn cuda_size_registry() -> &'static std::sync::Mutex<TestHashMap<usize, usize>> {
    CUDA_SIZE_REGISTRY.get_or_init(|| std::sync::Mutex::new(TestHashMap::new()))
}
```

Then in `alloc_managed`, add after the `#[cfg(test)]` counter bump:

```rust
#[cfg(test)]
cuda_size_registry().lock().unwrap().insert(ptr as usize, size_bytes);
```

And in `free_managed`:

```rust
pub(crate) fn free_managed(ptr: *mut c_void) {
    ensure_context();
    #[cfg(test)]
    {
        let size = cuda_size_registry().lock().unwrap().remove(&(ptr as usize)).unwrap_or(0);
        crate::memory::stats::cuda_free(size);
    }
    unsafe {
        let result = cuMemFree_v2(ptr as CUdeviceptr);
        assert_eq!(
            result,
            CUresult::CUDA_SUCCESS,
            "cuMemFree failed: {:?}",
            result
        );
    }
}
```

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `cargo test -p nsl-runtime`
Expected: All existing tests PASS (18 runtime + whatever else exists). The CUDA instrumentation compiles but only runs with `--features cuda`.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/cuda/mod.rs
git commit -m "feat(fuzz): instrument CUDA alloc_managed/free_managed with stats counters"
```

---

### Task 4: Verify counter balance on a known-good tensor lifecycle

**Files:**
- Modify: `crates/nsl-runtime/src/fuzz.rs`

This is the smoke test proving the counters work end-to-end with real tensor operations.

- [ ] **Step 1: Write the test**

Add to `crates/nsl-runtime/src/fuzz.rs` tests module:

```rust
#[test]
fn test_tensor_lifecycle_counter_balance() {
    let _guard = super::FUZZ_LOCK.lock().unwrap();
    use crate::tensor::{nsl_tensor_zeros, nsl_tensor_add, nsl_tensor_free};
    use crate::list::{nsl_list_new, nsl_list_push, nsl_list_free};

    stats::reset();

    // Create a shape list [4, 4]
    let shape = nsl_list_new();
    nsl_list_push(shape, 4);
    nsl_list_push(shape, 4);

    // Create two tensors
    let a = nsl_tensor_zeros(shape);
    let shape2 = nsl_list_new();
    nsl_list_push(shape2, 4);
    nsl_list_push(shape2, 4);
    let b = nsl_tensor_zeros(shape2);

    // Add them
    let c = nsl_tensor_add(a, b);

    // Free everything
    nsl_tensor_free(a);
    nsl_tensor_free(b);
    nsl_tensor_free(c);
    nsl_list_free(shape);
    nsl_list_free(shape2);

    // Assert counter balance
    let allocs = stats::ALLOC_COUNT.load(Ordering::SeqCst);
    let frees = stats::FREE_COUNT.load(Ordering::SeqCst);
    let alloc_bytes = stats::ALLOC_BYTES.load(Ordering::SeqCst);
    let free_bytes = stats::FREE_BYTES.load(Ordering::SeqCst);

    assert_eq!(
        allocs, frees,
        "CPU alloc/free count mismatch: {} allocs, {} frees ({} leaked)",
        allocs, frees, allocs as isize - frees as isize
    );
    assert_eq!(
        alloc_bytes, free_bytes,
        "CPU alloc/free bytes mismatch: {} allocated, {} freed ({} bytes leaked)",
        alloc_bytes, free_bytes, alloc_bytes as isize - free_bytes as isize
    );
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p nsl-runtime test_tensor_lifecycle_counter_balance`
Expected: PASS — if it fails, there's a real leak to investigate before proceeding.

**Debugging note:** If the test fails, the assertion message shows exact counts. Common causes:
- Shape list allocations not freed (need `nsl_list_free`)
- Tensor strides allocated via `checked_alloc` but freed path mismatched
- The `nsl_tensor_add` call recording on the autodiff tape (shouldn't be, tape not started)

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/src/fuzz.rs
git commit -m "test(fuzz): verify counter balance on known-good tensor lifecycle"
```

---

## Chunk 2: Fuzz Harness — Generator + Executor

### Task 5: FuzzOp enum and FuzzState

**Files:**
- Modify: `crates/nsl-runtime/src/fuzz.rs`

- [ ] **Step 1: Define FuzzOp enum and FuzzState struct**

Replace the contents of `crates/nsl-runtime/src/fuzz.rs` (keeping existing tests at the bottom):

```rust
//! Deterministic fuzz harness for memory lifecycle testing.
//! Generates random sequences of FFI tensor operations and asserts
//! that allocation counters balance after each sequence.
//!
//! Run: `cargo test -p nsl-runtime fuzz`

#[cfg(test)]
use crate::memory::stats;

/// Fuzzable tensor operations.
#[cfg(test)]
#[derive(Debug, Clone)]
enum FuzzOp {
    // Creation
    Zeros { dims: Vec<i64> },
    Ones { dims: Vec<i64> },
    Rand { dims: Vec<i64> },
    Randn { dims: Vec<i64> },
    Arange { start: f64, stop: f64, step: f64 },

    // Unary (index into live pool)
    Clone { idx: usize },
    Reshape { idx: usize, new_dims: Vec<i64> },
    Transpose { idx: usize, dim0: i64, dim1: i64 },
    Free { idx: usize },

    // Binary (first operand from pool, second created fresh with matching shape)
    Add { idx: usize },
    Mul { idx: usize },
    MatMul { idx: usize, n: i64 },  // creates [K, n] to match [M, K]
    ScalarMul { idx: usize, scalar: f64 },

    // Reductions
    Sum { idx: usize },
    Mean { idx: usize },

    // Activations
    ReLU { idx: usize },
    Sigmoid { idx: usize },
    Tanh { idx: usize },

    // Tape operations
    TapeStart,             // start recording with random subset of live tensors as params
    TapeBackwardAndStop,   // sum a live tensor → backward → free grads → tape_stop
    TapeStop,              // aborted graph — tape_stop without backward
}

/// Minimal xorshift64 PRNG (no external deps).
#[cfg(test)]
struct Xorshift64 {
    state: u64,
}

#[cfg(test)]
impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Random u64 in [0, bound).
    fn next_bounded(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    /// Random f64 in [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Fuzz session state.
#[cfg(test)]
struct FuzzState {
    rng: Xorshift64,
    /// Live tensor pointers (i64 values returned from FFI).
    live: Vec<i64>,
    /// Shapes of live tensors (parallel to `live` vec).
    shapes: Vec<Vec<i64>>,
    /// Whether the autodiff tape is currently recording.
    tape_active: bool,
    /// Param pointers registered with tape_start (subset of `live` at time of start).
    tape_params: Vec<i64>,
    /// Gradient list pointer from last backward (needs freeing).
    last_grad_list: Option<i64>,
}

#[cfg(test)]
impl FuzzState {
    fn new(seed: u64) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            live: Vec::new(),
            shapes: Vec::new(),
            tape_active: false,
            tape_params: Vec::new(),
            last_grad_list: None,
        }
    }
}
```

- [ ] **Step 2: Run to verify it compiles**

Run: `cargo test -p nsl-runtime test_stats_counter_reset`
Expected: PASS (no new tests yet, just verifying compile).

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/src/fuzz.rs
git commit -m "feat(fuzz): define FuzzOp enum, Xorshift64 PRNG, and FuzzState"
```

---

### Task 6: Sequence generator

**Files:**
- Modify: `crates/nsl-runtime/src/fuzz.rs`

The generator must produce valid op sequences — shape-compatible binaries, valid tape state transitions, etc.

- [ ] **Step 1: Implement random shape helper**

Add after `FuzzState::new()`:

```rust
#[cfg(test)]
impl FuzzState {
    /// Generate a random shape with 1-3 dimensions, each size 1-32.
    fn random_shape(&mut self) -> Vec<i64> {
        let ndim = self.rng.next_bounded(3) as usize + 1; // 1, 2, or 3
        (0..ndim)
            .map(|_| self.rng.next_bounded(32) as i64 + 1) // 1..=32
            .collect()
    }

    /// Total elements in a shape.
    fn shape_len(shape: &[i64]) -> i64 {
        shape.iter().product()
    }

    /// Pick a random index from the live pool. Returns None if pool is empty.
    fn pick_live(&mut self) -> Option<usize> {
        if self.live.is_empty() {
            None
        } else {
            Some(self.rng.next_bounded(self.live.len() as u64) as usize)
        }
    }
}
```

- [ ] **Step 2: Implement generate_op**

Add to `FuzzState` impl:

```rust
    /// Generate one random fuzz op that is valid given current state.
    fn generate_op(&mut self) -> FuzzOp {
        // Bias: if pool is empty, always create a tensor
        if self.live.is_empty() {
            return self.random_creation_op();
        }

        // Tape transitions: handle state machine
        if self.tape_active {
            // 15% chance of tape_stop (aborted graph), 10% backward+stop, 75% normal op
            let r = self.rng.next_bounded(100);
            if r < 15 {
                return FuzzOp::TapeStop;
            } else if r < 25 {
                return FuzzOp::TapeBackwardAndStop;
            }
        } else {
            // 10% chance of starting tape if we have 2+ tensors
            if self.live.len() >= 2 && self.rng.next_bounded(100) < 10 {
                return FuzzOp::TapeStart;
            }
        }

        // Normal op distribution
        let r = self.rng.next_bounded(100);
        match r {
            0..=19 => self.random_creation_op(),
            20..=24 => {
                // Don't free tape params while recording — causes use-after-free in backward
                let idx = self.pick_live().unwrap();
                if self.tape_active && self.tape_params.contains(&self.live[idx]) {
                    FuzzOp::Clone { idx }  // safe fallback
                } else {
                    FuzzOp::Free { idx }
                }
            }
            25..=29 => FuzzOp::Clone { idx: self.pick_live().unwrap() },
            30..=34 => {
                let idx = self.pick_live().unwrap();
                let shape = &self.shapes[idx];
                let total = Self::shape_len(shape);
                // Generate a valid reshape target with same element count
                let new_dims = self.random_reshape_target(total);
                FuzzOp::Reshape { idx, new_dims }
            }
            35..=39 => {
                let idx = self.pick_live().unwrap();
                let ndim = self.shapes[idx].len() as i64;
                if ndim >= 2 {
                    FuzzOp::Transpose { idx, dim0: 0, dim1: ndim - 1 }
                } else {
                    // 1D tensor — clone instead
                    FuzzOp::Clone { idx }
                }
            }
            40..=49 => FuzzOp::Add { idx: self.pick_live().unwrap() },
            50..=59 => FuzzOp::Mul { idx: self.pick_live().unwrap() },
            60..=64 => {
                let idx = self.pick_live().unwrap();
                let shape = &self.shapes[idx];
                if shape.len() >= 2 {
                    let n = self.rng.next_bounded(16) as i64 + 1;
                    FuzzOp::MatMul { idx, n }
                } else {
                    FuzzOp::ScalarMul { idx, scalar: self.rng.next_f64() * 10.0 }
                }
            }
            65..=69 => FuzzOp::ScalarMul {
                idx: self.pick_live().unwrap(),
                scalar: self.rng.next_f64() * 10.0 - 5.0,
            },
            70..=74 => FuzzOp::Sum { idx: self.pick_live().unwrap() },
            75..=79 => FuzzOp::Mean { idx: self.pick_live().unwrap() },
            80..=86 => FuzzOp::ReLU { idx: self.pick_live().unwrap() },
            87..=93 => FuzzOp::Sigmoid { idx: self.pick_live().unwrap() },
            _ => FuzzOp::Tanh { idx: self.pick_live().unwrap() },
        }
    }

    fn random_creation_op(&mut self) -> FuzzOp {
        let dims = self.random_shape();
        match self.rng.next_bounded(5) {
            0 => FuzzOp::Zeros { dims },
            1 => FuzzOp::Ones { dims },
            2 => FuzzOp::Rand { dims },
            3 => FuzzOp::Randn { dims },
            _ => {
                let stop = self.rng.next_bounded(64) as f64 + 1.0;
                FuzzOp::Arange { start: 0.0, stop, step: 1.0 }
            }
        }
    }

    /// Find a valid reshape target for `total` elements.
    fn random_reshape_target(&mut self, total: i64) -> Vec<i64> {
        // Simple: pick 1-3 dims, last dim absorbs remainder
        let ndim = self.rng.next_bounded(3) as usize + 1;
        if ndim == 1 {
            return vec![total];
        }
        // Try to factor; fallback to [total] if unlucky
        let mut dims = Vec::new();
        let mut remaining = total;
        for _ in 0..ndim - 1 {
            // Pick a divisor of remaining
            let max_dim = remaining.min(32);
            if max_dim <= 1 {
                break;
            }
            // Try random values until we find a divisor (max 10 attempts)
            let mut found = false;
            for _ in 0..10 {
                let d = self.rng.next_bounded(max_dim as u64) as i64 + 1;
                if remaining % d == 0 {
                    dims.push(d);
                    remaining /= d;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
        dims.push(remaining);
        dims
    }
```

- [ ] **Step 3: Write a test verifying generation produces valid ops**

Add to tests:

```rust
#[test]
fn test_fuzz_generator_produces_ops() {
    let mut state = super::FuzzState::new(42);
    let mut ops = Vec::new();
    for _ in 0..100 {
        ops.push(state.generate_op());
    }
    // Should have a mix of creation and other ops
    let creation_count = ops.iter().filter(|op| matches!(op,
        super::FuzzOp::Zeros { .. } | super::FuzzOp::Ones { .. } |
        super::FuzzOp::Rand { .. } | super::FuzzOp::Randn { .. } |
        super::FuzzOp::Arange { .. }
    )).count();
    assert!(creation_count > 0, "should generate at least one creation op");
    assert!(ops.len() == 100, "should generate exactly 100 ops");
}
```

- [ ] **Step 4: Run test**

Run: `cargo test -p nsl-runtime test_fuzz_generator`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/fuzz.rs
git commit -m "feat(fuzz): implement op sequence generator with shape compatibility"
```

---

### Task 7: Sequence executor

**Files:**
- Modify: `crates/nsl-runtime/src/fuzz.rs`

The executor calls real FFI functions and maintains the live tensor pool.

- [ ] **Step 1: Implement helper to create a shape list from a Vec**

Add to `FuzzState` impl:

```rust
    /// Create an NslList shape from a Vec<i64>. Caller must free the list.
    fn make_shape_list(dims: &[i64]) -> i64 {
        let list = crate::list::nsl_list_new();
        for &d in dims {
            crate::list::nsl_list_push(list, d);
        }
        list
    }

    /// Register a new tensor in the live pool.
    fn register(&mut self, ptr: i64, shape: Vec<i64>) {
        self.live.push(ptr);
        self.shapes.push(shape);
    }

    /// Remove a tensor from the live pool by index.
    fn unregister(&mut self, idx: usize) -> i64 {
        self.shapes.swap_remove(idx);
        self.live.swap_remove(idx)
    }
```

- [ ] **Step 2: Implement execute_op**

Add to `FuzzState` impl:

```rust
    /// Execute a single fuzz op against the real FFI runtime.
    fn execute_op(&mut self, op: FuzzOp) {
        use crate::tensor::*;
        use crate::autodiff::*;
        use crate::list::*;

        match op {
            // === Creation ops ===
            FuzzOp::Zeros { dims } => {
                let shape = Self::make_shape_list(&dims);
                let t = nsl_tensor_zeros(shape);
                nsl_list_free(shape);
                self.register(t, dims);
            }
            FuzzOp::Ones { dims } => {
                let shape = Self::make_shape_list(&dims);
                let t = nsl_tensor_ones(shape);
                nsl_list_free(shape);
                self.register(t, dims);
            }
            FuzzOp::Rand { dims } => {
                let shape = Self::make_shape_list(&dims);
                let t = nsl_tensor_rand(shape);
                nsl_list_free(shape);
                self.register(t, dims);
            }
            FuzzOp::Randn { dims } => {
                let shape = Self::make_shape_list(&dims);
                let t = nsl_tensor_randn(shape);
                nsl_list_free(shape);
                self.register(t, dims);
            }
            FuzzOp::Arange { start, stop, step } => {
                let t = nsl_tensor_arange(start, stop, step);
                let len = ((stop - start) / step).ceil() as i64;
                self.register(t, vec![len]);
            }

            // === Unary ops ===
            FuzzOp::Clone { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let t = nsl_tensor_clone(ptr);
                self.register(t, shape);
            }
            FuzzOp::Reshape { idx, new_dims } => {
                let ptr = self.live[idx];
                let shape = Self::make_shape_list(&new_dims);
                let t = nsl_tensor_reshape(ptr, shape);
                nsl_list_free(shape);
                self.register(t, new_dims);
            }
            FuzzOp::Transpose { idx, dim0, dim1 } => {
                let ptr = self.live[idx];
                let mut shape = self.shapes[idx].clone();
                let t = nsl_tensor_transpose(ptr, dim0, dim1);
                shape.swap(dim0 as usize, dim1 as usize);
                self.register(t, shape);
            }
            FuzzOp::Free { idx } => {
                let ptr = self.unregister(idx);
                nsl_tensor_free(ptr);
            }

            // === Binary ops (create matching operand) ===
            FuzzOp::Add { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                // Create a matching zeros tensor for the second operand
                let shape_list = Self::make_shape_list(&shape);
                let b = nsl_tensor_zeros(shape_list);
                nsl_list_free(shape_list);
                let result = nsl_tensor_add(ptr, b);
                nsl_tensor_free(b);
                self.register(result, shape);
            }
            FuzzOp::Mul { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let shape_list = Self::make_shape_list(&shape);
                let b = nsl_tensor_ones(shape_list);
                nsl_list_free(shape_list);
                let result = nsl_tensor_mul(ptr, b);
                nsl_tensor_free(b);
                self.register(result, shape);
            }
            FuzzOp::MatMul { idx, n } => {
                let ptr = self.live[idx];
                let shape = &self.shapes[idx];
                // shape must be >= 2D; last dim is K
                let k = *shape.last().unwrap();
                let rhs_shape = vec![k, n];
                let rhs_list = Self::make_shape_list(&rhs_shape);
                let rhs = nsl_tensor_ones(rhs_list);
                nsl_list_free(rhs_list);
                let result = nsl_tensor_matmul(ptr, rhs);
                nsl_tensor_free(rhs);
                // Output shape: [shape[0..n-1], n]
                let mut out_shape = shape[..shape.len() - 1].to_vec();
                out_shape.push(n);
                self.register(result, out_shape);
            }
            FuzzOp::ScalarMul { idx, scalar } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let result = nsl_tensor_mul_scalar(ptr, scalar);
                self.register(result, shape);
            }

            // === Reductions ===
            FuzzOp::Sum { idx } => {
                let ptr = self.live[idx];
                let result = nsl_tensor_sum(ptr);
                self.register(result, vec![1]);
            }
            FuzzOp::Mean { idx } => {
                let ptr = self.live[idx];
                let result = nsl_tensor_mean(ptr);
                self.register(result, vec![1]);
            }

            // === Activations ===
            FuzzOp::ReLU { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let result = nsl_tensor_relu(ptr);
                self.register(result, shape);
            }
            FuzzOp::Sigmoid { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let result = nsl_tensor_sigmoid(ptr);
                self.register(result, shape);
            }
            FuzzOp::Tanh { idx } => {
                let ptr = self.live[idx];
                let shape = self.shapes[idx].clone();
                let result = nsl_tensor_tanh_act(ptr);
                self.register(result, shape);
            }

            // === Tape ops ===
            FuzzOp::TapeStart => {
                // Register ~50% of live tensors as params
                let param_list = nsl_list_new();
                self.tape_params.clear();
                for &ptr in &self.live {
                    if self.rng.next_bounded(2) == 0 {
                        nsl_list_push(param_list, ptr);
                        self.tape_params.push(ptr);
                    }
                }
                // Ensure at least one param
                if self.tape_params.is_empty() && !self.live.is_empty() {
                    let ptr = self.live[0];
                    nsl_list_push(param_list, ptr);
                    self.tape_params.push(ptr);
                }
                nsl_tape_start(param_list);
                nsl_list_free(param_list);
                self.tape_active = true;
            }
            FuzzOp::TapeBackwardAndStop => {
                if !self.tape_active || self.live.is_empty() || self.tape_params.is_empty() {
                    return;
                }
                // Pick a random live tensor, reduce to scalar, backward
                let idx = self.rng.next_bounded(self.live.len() as u64) as usize;
                let loss_input = self.live[idx];
                let loss = nsl_tensor_sum(loss_input);

                // Build param list for backward
                let param_list = nsl_list_new();
                for &p in &self.tape_params {
                    nsl_list_push(param_list, p);
                }

                let grad_list = nsl_tape_backward(loss, param_list);

                // Free all gradient tensors
                let grad_list_ref = crate::list::NslList::from_ptr(grad_list);
                for i in 0..grad_list_ref.len as usize {
                    let grad_ptr = unsafe { *grad_list_ref.data.add(i) };
                    nsl_tensor_free(grad_ptr);
                }
                nsl_list_free(grad_list);
                nsl_list_free(param_list);
                nsl_tensor_free(loss);

                nsl_tape_stop();
                self.tape_active = false;
                self.tape_params.clear();
            }
            FuzzOp::TapeStop => {
                if !self.tape_active {
                    return;
                }
                // Aborted graph — stop without backward
                nsl_tape_stop();
                self.tape_active = false;
                self.tape_params.clear();
            }
        }
    }
```

- [ ] **Step 3: Run to verify it compiles**

Run: `cargo test -p nsl-runtime test_stats_counter_reset`
Expected: PASS (compile check — no new tests yet).

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/fuzz.rs
git commit -m "feat(fuzz): implement sequence executor calling real FFI functions"
```

---

### Task 8: Teardown, assertion, and main fuzz test

**Files:**
- Modify: `crates/nsl-runtime/src/fuzz.rs`

- [ ] **Step 1: Implement teardown**

Add to `FuzzState` impl:

```rust
    /// Clean up all state: stop tape, free all live tensors.
    fn teardown(&mut self) {
        use crate::tensor::nsl_tensor_free;

        // Stop tape if still active
        if self.tape_active {
            crate::autodiff::nsl_tape_stop();
            self.tape_active = false;
            self.tape_params.clear();
        }

        // Free all live tensors
        let ptrs: Vec<i64> = self.live.drain(..).collect();
        self.shapes.clear();
        for ptr in ptrs {
            nsl_tensor_free(ptr);
        }
    }
```

- [ ] **Step 2: Implement assertion helper**

Add as a standalone function:

```rust
#[cfg(test)]
fn assert_counter_balance(seed: u64) {
    use std::sync::atomic::Ordering;

    let ac = stats::ALLOC_COUNT.load(Ordering::SeqCst);
    let fc = stats::FREE_COUNT.load(Ordering::SeqCst);
    let ab = stats::ALLOC_BYTES.load(Ordering::SeqCst);
    let fb = stats::FREE_BYTES.load(Ordering::SeqCst);

    if ac != fc || ab != fb {
        panic!(
            "FUZZ FAILURE at seed={}\n\
             CPU: allocated {} ({} bytes), freed {} ({} bytes)\n\
             Leaked: {} tensors, {} bytes\n\
             Replay: cargo test -p nsl-runtime fuzz_memory_lifecycle -- --seed {}",
            seed,
            ac, ab, fc, fb,
            ac as isize - fc as isize,
            ab as isize - fb as isize,
            seed
        );
    }
}
```

- [ ] **Step 3: Implement the main fuzz test**

Add to tests module:

```rust
/// Mutex to serialize fuzz tests — global atomic counters are process-wide,
/// so concurrent tests would produce false positives.
static FUZZ_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn fuzz_memory_lifecycle() {
    let _guard = FUZZ_LOCK.lock().unwrap();
    let num_seeds = 100;
    let ops_per_seed = 150;

    for seed in 1..=num_seeds {
        stats::reset();
        let mut state = super::FuzzState::new(seed);

        // Generate and execute op sequence
        for _ in 0..ops_per_seed {
            let op = state.generate_op();
            state.execute_op(op);
        }

        // Teardown: free all remaining state
        state.teardown();

        // Assert counters balance
        super::assert_counter_balance(seed);
    }
}
```

- [ ] **Step 4: Run the fuzz test**

Run: `cargo test -p nsl-runtime fuzz_memory_lifecycle -- --nocapture`
Expected: PASS for all 100 seeds.

**If it fails:** The error message includes the seed. To replay:
1. Read the seed from the failure message
2. Set `num_seeds = seed; ops_per_seed = 150` and add `eprintln!` in `execute_op` to trace which op leaked.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/fuzz.rs
git commit -m "feat(fuzz): main fuzz_memory_lifecycle test — 100 seeds × 150 ops"
```

---

### Task 9: Tape-stress targeted test

**Files:**
- Modify: `crates/nsl-runtime/src/fuzz.rs`

This test specifically stresses the autodiff tape with many start/stop/backward cycles, maximizing exposure to the refcount retention paths.

- [ ] **Step 1: Write tape-stress test**

Add to tests module:

```rust
#[test]
fn fuzz_tape_stress() {
    let _guard = FUZZ_LOCK.lock().unwrap();
    let num_seeds = 50;

    for seed in 1000..1000 + num_seeds {
        stats::reset();
        let mut state = super::FuzzState::new(seed);

        // Create a pool of tensors first
        for _ in 0..10 {
            let dims = state.random_shape();
            let op = super::FuzzOp::Rand { dims };
            state.execute_op(op);
        }

        // Alternate: tape_start → some math → backward+stop OR tape_stop
        for cycle in 0..20 {
            state.execute_op(super::FuzzOp::TapeStart);

            // Do 5-15 math ops while recording
            let n_ops = state.rng.next_bounded(11) as usize + 5;
            for _ in 0..n_ops {
                let op = state.generate_op();
                // Skip tape transitions — we control those manually
                match &op {
                    super::FuzzOp::TapeStart |
                    super::FuzzOp::TapeStop |
                    super::FuzzOp::TapeBackwardAndStop => continue,
                    _ => state.execute_op(op),
                }
            }

            // 70% backward+stop, 30% aborted graph
            if cycle % 10 < 7 {
                state.execute_op(super::FuzzOp::TapeBackwardAndStop);
            } else {
                state.execute_op(super::FuzzOp::TapeStop);
            }
        }

        state.teardown();
        super::assert_counter_balance(seed);
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p nsl-runtime fuzz_tape_stress -- --nocapture`
Expected: PASS for all 50 seeds.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/src/fuzz.rs
git commit -m "test(fuzz): add tape-stress targeted fuzz test — 50 seeds × 20 cycles"
```

---

### Task 10: Run full test suite and verify no regressions

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `cargo test --workspace`
Expected: ALL tests PASS — the original 46 tests plus the new fuzz tests.

- [ ] **Step 2: Verify fuzz tests specifically**

Run: `cargo test -p nsl-runtime fuzz -- --nocapture`
Expected: `fuzz_memory_lifecycle`, `fuzz_tape_stress`, `test_tensor_lifecycle_counter_balance`, `test_stats_counter_reset`, `test_stats_track_alloc_free`, `test_fuzz_generator_produces_ops` — all PASS.

- [ ] **Step 3: Final commit with all files**

If any files weren't committed in previous steps:

```bash
git add -A
git commit -m "feat(fuzz): complete memory lifecycle fuzz harness"
```
