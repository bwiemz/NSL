# FASE Peak-Memory Regression Test — Design

**Date:** 2026-04-14
**Status:** Design approved, ready for implementation plan
**Depends on:** Item #4 (consume-per-param hook)
**Scoped as:** Item #5 of the FASE roadmap

## Context

Item #4 added a consume-per-param hook that consumes and frees each parameter's gradient tensor during source-AD backward lowering.  The claim: "only one gradient is live at a time" — peak gradient memory drops from `N × grad_size` to `1 × grad_size`.  Item #4's numerical test validates that the hook produces correct parameter values, but it does not measure whether the peak-memory claim holds.  Item #5 adds that measurement as a regression guard.

The test is test-only — no production FFIs, no runtime overhead in shipped binaries.  A `test-hooks` cargo feature gates both the peak-tracking atomics inside `nsl-runtime` and the two FFIs that expose peak bytes to the integration test.  The fixture compiles as a shared library (via the existing M62a `nsl build --shared-library` path), and the test loads it in-process via `libloading`, so the .dll and the test share a single copy of the peak-tracking state (inside the .dll's own nsl-runtime).

## Goals

1. Add a `test-hooks` cargo feature in `crates/nsl-runtime` that enables CPU-heap peak tracking and exposes two FFIs (`nsl_cpu_peak_reset`, `nsl_cpu_peak_bytes`).
2. Add a 4-parameter MLP fixture whose gradient sizes are large enough that the hook's peak-memory delta is well above bookkeeping noise.
3. Write an integration test that builds the fixture twice (with / without `--source-ad`), loads each as a .dll, measures peak heap bytes, and asserts the source-AD path's peak is ≥ 150 KB lower than the tape-AD path's.
4. Guarantee production builds are untouched: default `cargo test -p nsl-codegen` and default `nsl build` produce byte-identical artefacts.

## Non-Goals

- **Production user-facing peak-memory profiler.**  The GPU path already has `nsl run --profile-memory` via the caching allocator.  Item #5 is test-only.
- **Absolute memory budget.**  The test asserts a *delta* between source-AD and tape-AD, not an absolute ceiling.  Absolute budgets would be fragile to unrelated runtime changes.
- **Windows-specific symbol-export polish.**  If Windows symbol export from staticlib → cdylib proves gnarly, the test can be `#[cfg(not(windows))]`-gated as a starting point, with a follow-up issue.
- **Multi-platform CI coverage.**  Pass on developer machines first; CI integration is a separate concern.

## Design Decisions

### D1. `test-hooks` cargo feature

`crates/nsl-runtime/Cargo.toml`:

```toml
[features]
default = []
test-hooks = []
```

Default feature set is empty; shipped binaries never compile the peak-tracking code.

`crates/nsl-codegen/Cargo.toml` forwards the feature:

```toml
[features]
test-hooks = ["nsl-runtime/test-hooks"]
```

When the test suite runs with `cargo test -p nsl-codegen --features test-hooks`, the feature propagates through the workspace graph: `nsl-cli` → `nsl-codegen` → `nsl-runtime`.  The compiled .dll built via `nsl build --shared-library` statically links the test-hooks-enabled nsl-runtime, so its internal peak-tracking state lives inside the .dll.  The integration test loads that .dll and calls INTO its own peak FFIs — one copy of the state, measuring exactly the .dll's allocations.

### D2. Peak tracking in `memory.rs`

Behind `#[cfg(feature = "test-hooks")]`, add to `crates/nsl-runtime/src/memory.rs`:

```rust
#[cfg(feature = "test-hooks")]
mod peak {
    use std::sync::atomic::{AtomicUsize, Ordering};
    pub static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
    pub static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

    pub fn record_alloc(size: usize) {
        let new_current = CURRENT_BYTES.fetch_add(size, Ordering::Relaxed) + size;
        PEAK_BYTES.fetch_max(new_current, Ordering::Relaxed);
    }

    pub fn record_free(size: usize) {
        CURRENT_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
}
```

Global atomics (not thread-local), because the .dll may spawn auxiliary threads and we want the max across all of them.

`checked_alloc`, `checked_alloc_zeroed`, and `checked_free` gain `#[cfg(feature = "test-hooks")]` calls to `peak::record_alloc` / `peak::record_free`.  With the feature disabled these are no-ops and the atomic code is dead-code-eliminated.

FFI exports (same gate):

```rust
#[cfg(feature = "test-hooks")]
#[no_mangle]
pub extern "C" fn nsl_cpu_peak_reset() {
    peak::CURRENT_BYTES.store(0, Ordering::Relaxed);
    peak::PEAK_BYTES.store(0, Ordering::Relaxed);
}

#[cfg(feature = "test-hooks")]
#[no_mangle]
pub extern "C" fn nsl_cpu_peak_bytes() -> i64 {
    peak::PEAK_BYTES.load(Ordering::Relaxed) as i64
}
```

### D3. Fixture: 4-parameter MLP

Multi-parameter is essential to the test's diagnostic value — a single-param model's peak is identical with or without the hook (one gradient either way).  Four parameters with two of them substantial give the hook a clear multi-param delta to demonstrate.

`crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl`:

```nsl
from nsl.nn.losses import mse_loss

model Mlp:
    w1: Tensor = ones([256, 256])
    b1: Tensor = zeros([256])
    w2: Tensor = ones([256, 256])
    b2: Tensor = zeros([256])

    fn forward(self, x: Tensor) -> Tensor:
        let h = x @ self.w1 + self.b1
        return h @ self.w2 + self.b2

let m = Mlp()
let x = ones([1, 256])
let y = zeros([1, 256])

train(model = m, epochs = 4, grad_accumulation = 4):
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
```

Sizing (f32):
- `w1`: 256 × 256 × 4 = **262 144 bytes**
- `b1`: 256 × 4 = **1 024 bytes**
- `w2`: 256 × 256 × 4 = **262 144 bytes**
- `b2`: 256 × 4 = **1 024 bytes**

All-gradients-live (tape-AD): ≥ 526 KB peak gradient memory.
Max single gradient (source-AD + hook): 256 KB peak gradient memory.
Expected delta: ≥ 250 KB, well above bookkeeping noise floor (< 50 KB in practice).

No `model_save` call — the test reads peak via FFI, not the checkpoint.

### D4. Test harness

`crates/nsl-codegen/tests/fase_peak_memory.rs`:

```rust
//! Item #5: FASE Deferred peak-memory regression test.
//!
//! Only compiles with --features test-hooks.

#[cfg(feature = "test-hooks")]
#[path = "fase_peak_memory_impl.rs"]
mod test_impl;
```

And `crates/nsl-codegen/tests/fase_peak_memory_impl.rs`:

```rust
use libloading::{Library, Symbol};
use std::path::PathBuf;
use std::process::Command;

fn build_fixture(source_ad: bool) -> PathBuf {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fase_peak_memory.nsl");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("build").arg("--shared-library").arg(&fixture);
    if source_ad {
        cmd.arg("--source-ad");
    }
    let status = cmd.status().expect("spawn nsl build");
    assert!(status.success(), "nsl build failed");
    // Default output path from linker.rs:784 convention.
    // Adapt after reading the real path resolver.
    /* resolved dll path */
}

fn run_and_measure_peak(dll_path: &std::path::Path) -> usize {
    unsafe {
        let lib = Library::new(dll_path).expect("load .dll");
        let reset: Symbol<unsafe extern "C" fn()> =
            lib.get(b"nsl_cpu_peak_reset").expect("reset symbol");
        let peak: Symbol<unsafe extern "C" fn() -> i64> =
            lib.get(b"nsl_cpu_peak_bytes").expect("peak symbol");
        let main_fn: Symbol<unsafe extern "C" fn()> =
            lib.get(b"main").expect("main symbol");
        reset();
        main_fn();
        peak() as usize
    }
}

#[test]
fn source_ad_peak_is_lower_than_tape_ad() {
    let tape_dll = build_fixture(/*source_ad=*/ false);
    let source_dll = build_fixture(/*source_ad=*/ true);

    let peak_tape = run_and_measure_peak(&tape_dll);
    let peak_source = run_and_measure_peak(&source_dll);

    const MIN_DELTA_BYTES: usize = 150_000;
    assert!(
        peak_tape >= peak_source + MIN_DELTA_BYTES,
        "expected source-AD peak at least {} bytes below tape-AD; got tape={} source={}",
        MIN_DELTA_BYTES, peak_tape, peak_source
    );
}
```

The outer `fase_peak_memory.rs` is cfg-gated so `cargo test` without the feature is a no-op for this file (zero compile/run cost).  The inner `fase_peak_memory_impl.rs` carries the actual test body, included via `#[path = ...]`.

`libloading = "0.8"` gets added to `crates/nsl-codegen/Cargo.toml`'s `[dev-dependencies]`.

### D5. Symbol visibility in the .dll

The `#[no_mangle] pub extern "C" fn nsl_cpu_peak_*` functions live in `libnsl_runtime.{a,lib}` when test-hooks is enabled.  When that static lib gets linked into a cdylib by `nsl build --shared-library`, the linker needs to re-export the symbols so `libloading::Symbol::get` can resolve them.

Platform-specific tactics (investigate in implementation Task 1):

- **Linux / macOS:** `-Wl,--export-dynamic` or `-Wl,-export_dynamic` on the link line.  Or listing the symbols in a version script.
- **Windows MSVC:** `/EXPORT:nsl_cpu_peak_reset /EXPORT:nsl_cpu_peak_bytes` on the `link.exe` command line, or a `.def` file.
- **Rust's built-in cdylib:** `#[no_mangle]` + `pub extern "C"` from a cdylib crate auto-exports.  But here the `extern "C"` lives in a `staticlib`, not a cdylib — the re-export is the linker's responsibility.

The existing `nsl build --shared-library` path in `linker.rs:660` / `:717` assembles the link command.  Adding platform-specific `--export-dynamic` / `/EXPORT` args behind a `test-hooks`-active check is the likely fix.  If the fix turns out to be gnarly on Windows specifically, the test can start Linux/macOS-only with a follow-up issue.

### D6. Build invocation

`build_fixture()` invokes the nsl CLI binary already built by the test harness (`env!("CARGO_BIN_EXE_nsl")`).  The `--features test-hooks` flag must have been active when `nsl-cli` / `nsl-codegen` / `nsl-runtime` were compiled for the test run — which is true when the user runs `cargo test -p nsl-codegen --features test-hooks`.

No `cargo run` inside the test (would recompile and be slow).  Just `env!("CARGO_BIN_EXE_nsl")` invocation, identical to how items #2-#4 use `nsl_run`.

### D7. Components touched

| File | Change |
|---|---|
| `crates/nsl-runtime/Cargo.toml` | `[features] default = [] \n test-hooks = []`. |
| `crates/nsl-runtime/src/memory.rs` | `#[cfg(feature = "test-hooks")]` peak module + hooks in `checked_alloc{,_zeroed}` / `checked_free` + two FFI exports. ~40 LOC. |
| `crates/nsl-codegen/Cargo.toml` | Forward feature; add `libloading = "0.8"` dev-dep. |
| `crates/nsl-codegen/src/linker.rs` | Platform-specific `--export-dynamic` / `/EXPORT` when test-hooks active (confirmed required in Task 1 of plan). |
| `crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl` (new) | 4-parameter MLP fixture. |
| `crates/nsl-codegen/tests/fase_peak_memory.rs` (new) | Test entry point, cfg-gated. |
| `crates/nsl-codegen/tests/fase_peak_memory_impl.rs` (new) | Test body: build + libloading + assertion. |
| memory note | Mark item #5 shipped. |

Total: ~150 LOC across runtime, codegen, and tests.

## Architecture

### Data flow

```
cargo test -p nsl-codegen --features test-hooks
  │
  ├─ feature propagates: nsl-codegen → nsl-runtime with test-hooks on
  ├─ CARGO_BIN_EXE_nsl is pre-built with test-hooks on
  │
  └─ test source_ad_peak_is_lower_than_tape_ad:
        │
        ├─ build_fixture(source_ad=false):
        │     Command: nsl build --shared-library <fixture>
        │     Output: libfase_peak_memory.dll (tape-AD codegen)
        │
        ├─ build_fixture(source_ad=true):
        │     Command: nsl build --shared-library --source-ad <fixture>
        │     Output: libfase_peak_memory_source_ad.dll (hook active)
        │
        ├─ run_and_measure_peak(tape_dll):
        │     libloading::Library::new(tape_dll)
        │     resolve nsl_cpu_peak_reset, nsl_cpu_peak_bytes, main
        │     reset() → main() → peak() → peak_tape
        │
        ├─ run_and_measure_peak(source_dll):
        │     (same, separate .dll with its own state)
        │     → peak_source
        │
        └─ assert peak_tape >= peak_source + 150 KB
```

Two distinct Library instances, two distinct in-process memory spaces for the .dll internals (each .dll has its own copy of nsl-runtime globals).  The test only ever reads each .dll's own `nsl_cpu_peak_bytes` after resetting it — there is no cross-.dll state to worry about.

## Risks

1. **Windows symbol-export gnarliness.** D5 flags the platform-specific work.  If it turns out MSVC refuses to re-export symbols from a staticlib without a `.def` file and `.def` generation adds meaningful complexity, scope the initial landing to Linux/macOS via `#[cfg(not(windows))]` and add the Windows story as a follow-up.
2. **.dll output path.** `linker.rs:777` has a default-path function; confirm it's callable or matches the expected convention from outside.  If the CLI's `nsl build --shared-library` flag requires an explicit `-o` to produce a deterministic path, the test passes one.
3. **Noise floor.** 150 KB delta threshold assumes non-gradient bookkeeping is well below that.  Empirically verify in plan Task 3.  If the delta comes in at, say, 200 KB observed with 180 KB noise baseline, tighten the fixture (larger w1/w2) rather than loosening the threshold.
4. **`main` signature.** Whatever `nsl build` emits as the program entry — zero-arg, return void, or something else.  If it expects `argc`/`argv`, the test calls it with zero args and null argv.  Implementation plan Task 4 confirms the signature.
5. **Build artefact caching.** Two `nsl build` calls per test run = slow.  Cache the .dll builds via mtime / hash if feasible; otherwise accept the cost (this test runs behind an opt-in feature, so it's never on `cargo test`'s default hot path).
6. **GPU allocator overlap.** `peak::record_alloc` is CPU-only (lives in `memory.rs`, which is the CPU heap path).  GPU-allocated tensors go through `cuda/caching_allocator.rs` — those allocations are NOT counted by this peak.  Acceptable: the fixture is CPU-only by construction.

## Success Criteria

- `cargo test -p nsl-codegen --features test-hooks -- source_ad_peak_is_lower_than_tape_ad` passes.  Observed `peak_tape - peak_source ≥ 150 000 bytes`.
- `cargo test -p nsl-codegen` (default features) does not run the test, does not pull in `libloading`, and is byte-identical to pre-item-#5 behavior.
- `nsl build` with default features produces the same object file / executable / shared library as pre-item-#5.
- `libnsl_runtime.{a,lib}` built without `test-hooks` does not contain `nsl_cpu_peak_reset` / `nsl_cpu_peak_bytes` symbols (confirmed by `nm` / `dumpbin`).

## Follow-Ups

Closes item #5.  Remaining FASE roadmap:
- **Item #6 — `nsl check --training-report` CLI.** Pure observability, independent of everything else.
