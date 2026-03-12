# Fuzz Harness for Memory Lifecycle

## Goal

Deterministic fuzz harness for the NSL runtime memory lifecycle. Catches both memory leaks (allocation counter imbalance) and crashes (use-after-free, double-free) across tensor operations and the autodiff tape.

## Decisions

- **Scope:** Runtime + autodiff FFI surface only. Does not cover codegen scope teardown (loop-break cleanup) — that requires a separate compiler-level harness feeding NSL source text.
- **Engine:** Custom deterministic harness as a `#[cfg(test)]` module. No external dependencies, no nightly, works on Windows. Runs via `cargo test -p nsl-runtime fuzz`.
- **Leak detection:** Global atomic counters (monotonic gauges). 8 counters total — CPU and CUDA, count and bytes.
- **Op vocabulary:** ~20 FFI operations including autodiff tape ops.

## Architecture

### Counters (`crates/nsl-runtime/src/memory.rs`)

New `stats` submodule behind `#[cfg(test)]`:

```rust
#[cfg(test)]
pub mod stats {
    use std::sync::atomic::{AtomicUsize, Ordering};

    // CPU tracking
    pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static FREE_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
    pub static FREE_BYTES: AtomicUsize = AtomicUsize::new(0);

    // GPU tracking
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
}
```

Instrumentation points:
- `checked_alloc(size)` → increment `ALLOC_COUNT` + `ALLOC_BYTES`
- `checked_free(ptr, size)` → increment `FREE_COUNT` + `FREE_BYTES`
- `alloc_managed(size)` → increment `CUDA_ALLOC_COUNT` + `CUDA_ALLOC_BYTES`
- `free_managed(ptr)` → increment `CUDA_FREE_COUNT` + `CUDA_FREE_BYTES` (size must be tracked or passed)

### Fuzz Harness (`crates/nsl-runtime/src/fuzz.rs`)

New `#[cfg(test)]` module containing:

1. **`FuzzOp` enum** — ~20 variants representing fuzzable operations
2. **`FuzzState`** — live tensor pool, tape state machine, RNG
3. **Sequence generator** — produces valid op sequences from a seed
4. **Sequence executor** — calls real FFI functions
5. **Assertion helper** — checks counter balance after teardown

## Op Vocabulary (~20 ops)

| Category | Ops | Notes |
|----------|-----|-------|
| Creation (5) | `zeros`, `ones`, `rand`, `randn`, `arange` | Random shapes 1-3 dims, sizes 1-64 |
| Unary (4) | `clone`, `reshape`, `transpose`, `free` | Target = random live tensor |
| Binary (4) | `add`, `mul`, `matmul`, `scalar_mul` | Shape-compatible (see below) |
| Reduction (2) | `sum`, `mean` | Produces scalar tensor |
| Activation (3) | `relu`, `sigmoid`, `tanh` | Unary on random live tensor |
| Tape (3) | `tape_start`, `tape_backward_and_stop`, `tape_stop` | State-machine governed |

## Sequence Generation

- **RNG:** Seeded `StdRng` from a `u64` seed. Seed printed on failure for deterministic replay.
- **Length:** 50-200 random ops per sequence.
- **Iterations:** 100 seeds per test run (configurable).
- **Live tensor pool:** `Vec<i64>` of active tensor pointers. Creation ops push, `free` removes. Ops needing a tensor pick randomly from the pool.

### Shape Compatibility

- **`add`/`mul`:** 80% — create a fresh tensor with identical shape to the picked operand. 20% — create a broadcastable `[1]` scalar tensor.
- **`matmul`:** Pick a live tensor of shape `[M, K]`, create a fresh `[K, N]` with matching inner dimension.
- **`reshape`:** Compute a valid target shape with the same total element count.

### Tape State Machine

```
IDLE ──tape_start──▶ RECORDING
RECORDING ──tape_stop──▶ IDLE          (aborted graph, ~30%)
RECORDING ──backward+stop──▶ IDLE      (normal path, ~70%)
```

- **`tape_start`:** Collects a random subset (~50%) of live tensors as the param set.
- **`tape_backward_and_stop`:** Compound action — pick a live tensor, `sum()` to scalar, `backward(scalar)`, free all gradient tensors from returned grad-list, then `tape_stop()`.
- **`tape_stop` (aborted graph):** Tests `release_tape_op_refs()` cleanup path — the exact code path where the M17b tape leak was found. Generated ~30% of the time when tape is active.

### Gradient Cleanup

NSL stores gradients in a global `HashMap<i64, i64>` (param_ptr → grad_ptr) inside the tape. `nsl_tape_backward` returns a grad-list pointer. The fuzzer must:
1. After `backward()`: iterate the returned grad-list and `nsl_tensor_free()` each gradient tensor.
2. This prevents false-positive leak assertions from lingering gradient allocations.

## Teardown (Per Sequence)

1. If tape still active → `nsl_tape_stop()` (forces aborted-graph cleanup)
2. Free all gradient tensors from last backward (if any)
3. Free all remaining live tensors in the pool
4. Assert counter balance:
   - `ALLOC_COUNT == FREE_COUNT` (CPU tensor count)
   - `ALLOC_BYTES == FREE_BYTES` (CPU byte count)
   - `CUDA_ALLOC_COUNT == CUDA_FREE_COUNT` (GPU tensor count)
   - `CUDA_ALLOC_BYTES == CUDA_FREE_BYTES` (GPU byte count)
5. `stats::reset()` for next sequence

## Failure Output

On assertion failure, print:
```
FUZZ FAILURE at seed=12345, sequence_len=137
  CPU: allocated 15432 tensors (2.3 MB), freed 15431 tensors (2.3 MB - 4096 bytes)
  GPU: allocated 0, freed 0
  Replay: cargo test -p nsl-runtime fuzz -- --seed 12345
```

The seed enables exact replay for debugging.

## File Changes

| File | Change |
|------|--------|
| `crates/nsl-runtime/src/memory.rs` | Add `#[cfg(test)] pub mod stats` with 8 `AtomicUsize` counters + `reset()` |
| `crates/nsl-runtime/src/memory.rs` | Instrument `checked_alloc`/`checked_free` to bump counters |
| `crates/nsl-runtime/src/cuda/mod.rs` | Instrument `alloc_managed`/`free_managed` to bump CUDA counters |
| `crates/nsl-runtime/src/fuzz.rs` | New `#[cfg(test)]` module — `FuzzOp` enum, generator, executor, assertions |
| `crates/nsl-runtime/src/lib.rs` | Add `#[cfg(test)] mod fuzz;` |

## Non-Goals

- **Codegen fuzzing:** Loop-break scope teardown is not tested. Requires a separate harness that feeds NSL source text through the compiler.
- **GPU ops:** CUDA counters are wired up but no GPU ops in the vocabulary. Can be added behind `#[cfg(feature = "cuda")]` later.
- **Coverage guidance:** This is a deterministic harness, not a coverage-guided fuzzer. `cargo-fuzz` can be layered on top for Linux CI later.
- **Per-tensor tracking:** No allocation registry or backtrace capture. If a counter-based failure is hard to debug, a `HashMap<*mut, Backtrace>` registry can be added later.
