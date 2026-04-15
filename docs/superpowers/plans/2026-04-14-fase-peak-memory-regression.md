# FASE Peak-Memory Regression Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate item #4's peak-memory claim with an in-process regression test that compiles the same MLP fixture twice (with/without `--source-ad`), loads each as a .dll via `libloading`, measures CPU heap peak via test-only FFIs, and asserts the source-AD path's peak is ≥ 150 KB lower.

**Architecture:** A `test-hooks` cargo feature in `nsl-runtime` gates `CURRENT_BYTES` + `PEAK_BYTES` atomics in the existing `checked_alloc`/`checked_free` choke-point, plus two FFIs (`nsl_cpu_peak_reset`, `nsl_cpu_peak_bytes`). The feature forwards from `nsl-codegen`. The linker paths for `nsl build --shared-library` are augmented to re-export the two peak FFIs when the feature is active. A new integration test in `crates/nsl-codegen/tests/` cfg-gates on `test-hooks`, builds two .dlls, loads them, and compares peaks.

**Tech Stack:** Rust (`AtomicUsize::fetch_max`), Cargo features, `libloading = "0.8"`, existing `nsl build --shared-library` infrastructure (M62a).

**Spec:** [docs/superpowers/specs/2026-04-14-fase-peak-memory-regression-design.md](../specs/2026-04-14-fase-peak-memory-regression-design.md)

---

## Task 1: Add `test-hooks` feature and peak-tracking in `nsl-runtime`

Add the cargo feature, the peak module, the instrumentation in `checked_alloc` / `checked_alloc_zeroed` / `checked_free`, and the two FFI exports. A unit test verifies the tracker.

**Files:**
- Modify: `crates/nsl-runtime/Cargo.toml`
- Modify: `crates/nsl-runtime/src/memory.rs`

### Steps

- [ ] **Step 1: Add the feature**

Open `crates/nsl-runtime/Cargo.toml`. Near the top under `[lib]` or after the manifest header, add (or extend if `[features]` already exists):

```toml
[features]
default = []
test-hooks = []
```

If `[features]` already exists, just add the `test-hooks = []` line.

- [ ] **Step 2: Write the failing test**

Append to the existing `#[cfg(test)] mod tests` module at the bottom of `crates/nsl-runtime/src/memory.rs` (grep for `#\[cfg(test)\]` in that file). The test is gated on the feature so it only runs when the feature is active:

```rust
#[cfg(all(test, feature = "test-hooks"))]
#[test]
fn cpu_peak_tracks_current_and_high_watermark() {
    // Reset so we're not influenced by other tests that ran before us.
    nsl_cpu_peak_reset();
    assert_eq!(nsl_cpu_peak_bytes(), 0);

    // Allocate 1 MB via the checked_alloc path (used by all tensor creation).
    let a = checked_alloc(1_048_576);
    // Peak should be at least 1 MB now.
    assert!(nsl_cpu_peak_bytes() >= 1_048_576, "peak after 1MB alloc: {}", nsl_cpu_peak_bytes());

    // Allocate another 2 MB.  Peak should rise.
    let b = checked_alloc(2_097_152);
    assert!(nsl_cpu_peak_bytes() >= 3_145_728, "peak after +2MB alloc: {}", nsl_cpu_peak_bytes());

    let peak_after_both = nsl_cpu_peak_bytes();

    // Free one.  Peak must NOT decrease (peak is high-watermark).
    unsafe { checked_free(a, 1_048_576); }
    assert_eq!(nsl_cpu_peak_bytes(), peak_after_both, "peak is a watermark; free must not lower it");

    // Free the other.
    unsafe { checked_free(b, 2_097_152); }
    assert_eq!(nsl_cpu_peak_bytes(), peak_after_both);

    // Reset zeros the watermark.
    nsl_cpu_peak_reset();
    assert_eq!(nsl_cpu_peak_bytes(), 0);
}
```

- [ ] **Step 3: Run — expect build failure**

Run: `cargo build -p nsl-runtime --features test-hooks`
Expected: FAIL (functions not yet defined).

- [ ] **Step 4: Add the peak tracking module and hooks**

Open `crates/nsl-runtime/src/memory.rs`. After the `checked_free` function (around line 130) and before the existing `#[cfg(test)] pub mod stats { ... }` block, add:

```rust
#[cfg(feature = "test-hooks")]
pub mod peak {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
    pub static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

    /// Record an allocation.  Bumps current and updates the high-watermark.
    pub fn record_alloc(size: usize) {
        let new_current = CURRENT_BYTES.fetch_add(size, Ordering::Relaxed) + size;
        PEAK_BYTES.fetch_max(new_current, Ordering::Relaxed);
    }

    /// Record a free.  Decrements current; peak is not modified.
    pub fn record_free(size: usize) {
        CURRENT_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
}

#[cfg(feature = "test-hooks")]
#[no_mangle]
pub extern "C" fn nsl_cpu_peak_reset() {
    use std::sync::atomic::Ordering;
    peak::CURRENT_BYTES.store(0, Ordering::Relaxed);
    peak::PEAK_BYTES.store(0, Ordering::Relaxed);
}

#[cfg(feature = "test-hooks")]
#[no_mangle]
pub extern "C" fn nsl_cpu_peak_bytes() -> i64 {
    use std::sync::atomic::Ordering;
    peak::PEAK_BYTES.load(Ordering::Relaxed) as i64
}
```

- [ ] **Step 5: Instrument `checked_alloc`**

Find the existing `checked_alloc` function (around line 58). It already has `#[cfg(test)] stats::cpu_alloc(size);`. Add an unconditional call to `peak::record_alloc(size)` right next to it, gated on the feature:

```rust
pub(crate) fn checked_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::NonNull::<u64>::dangling().as_ptr() as *mut u8;
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        eprintln!("nsl: out of memory");
        std::process::abort();
    }
    #[cfg(test)]
    stats::cpu_alloc(size);
    #[cfg(feature = "test-hooks")]
    peak::record_alloc(size);
    ptr
}
```

- [ ] **Step 6: Instrument `checked_alloc_zeroed`**

Same pattern at the function around line 77. Find its existing `#[cfg(test)] stats::cpu_alloc(size);` line and add the peak recording right below it:

```rust
    #[cfg(test)]
    stats::cpu_alloc(size);
    #[cfg(feature = "test-hooks")]
    peak::record_alloc(size);
```

- [ ] **Step 7: Instrument `checked_free`**

Find the function around line 120. It has existing `#[cfg(test)] stats::cpu_free(size);`. Add:

```rust
    #[cfg(test)]
    stats::cpu_free(size);
    #[cfg(feature = "test-hooks")]
    peak::record_free(size);
```

If there's a third existing alloc call site (`checked_realloc` or similar) — grep for `stats::cpu_alloc` / `stats::cpu_free` usages in `memory.rs` and mirror the feature-gated call at each one.

- [ ] **Step 8: Run the test**

Run: `cargo test -p nsl-runtime --features test-hooks --lib -- cpu_peak_tracks_current_and_high_watermark`
Expected: PASS.

- [ ] **Step 9: Verify no regression without the feature**

Run: `cargo test -p nsl-runtime --lib 2>&1 | tail -3`
Expected: all tests pass as before. The test added in Step 2 is `#[cfg(all(test, feature = "test-hooks"))]`, so it's skipped.

Also verify the feature-gated code compiles away when disabled:

```bash
cargo build -p nsl-runtime 2>&1 | grep -iE "warning|error" | head
```

Expected: no new warnings related to `peak` or `test-hooks` (unused code is gated out).

- [ ] **Step 10: Commit**

```bash
git add crates/nsl-runtime/Cargo.toml crates/nsl-runtime/src/memory.rs
git commit -m "feat(runtime): test-hooks feature + CPU heap peak tracking"
```

---

## Task 2: Forward the feature from `nsl-codegen` + add dev-deps

`crates/nsl-codegen/Cargo.toml` gets a forwarding feature plus `libloading` as a dev-dependency for Task 6's test harness.

**Files:**
- Modify: `crates/nsl-codegen/Cargo.toml`

### Steps

- [ ] **Step 1: Add the forwarding feature**

Open `crates/nsl-codegen/Cargo.toml`. Look for an existing `[features]` section. Add (or extend if present):

```toml
[features]
default = []
test-hooks = ["nsl-runtime/test-hooks"]
```

If `[features]` doesn't exist yet, add the section before `[dependencies]`. If it already has a default, preserve it and add the new entry.

- [ ] **Step 2: Add `libloading` to dev-dependencies**

Find the existing `[dev-dependencies]` block (grep `dev-dependencies`). Add:

```toml
libloading = "0.8"
```

- [ ] **Step 3: Build to verify the manifest is well-formed**

Run: `cargo build -p nsl-codegen --features test-hooks`
Expected: succeeds (the feature transitively enables nsl-runtime's test-hooks feature).

- [ ] **Step 4: Also verify `nsl-cli` inherits the feature correctly**

`CARGO_BIN_EXE_nsl` is the binary the test in Task 6 invokes via `Command`. That binary must be built with `test-hooks` so the compiled .dll has the right FFIs. Verify by running:

```bash
cargo build -p nsl-cli --features nsl-codegen/test-hooks
```

If `nsl-cli`'s Cargo.toml also needs a feature forwarder, add one:

```toml
[features]
default = []
test-hooks = ["nsl-codegen/test-hooks"]
```

Grep `grep -n "^\[features\]" crates/nsl-cli/Cargo.toml` and add the section or forward if absent.

Final build check:

```bash
cargo build -p nsl-cli --features test-hooks
```

Expected: succeeds. The resulting `nsl` binary links against the test-hooks-enabled runtime.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/Cargo.toml crates/nsl-cli/Cargo.toml
git commit -m "build: forward test-hooks feature through codegen/cli + add libloading"
```

---

## Task 3: Re-export peak FFIs from shared-library output

`nsl build --shared-library` currently links `libnsl_runtime.{a,lib}` (a staticlib) into the output .so/.dll/.dylib. Symbols from the staticlib are NOT automatically re-exported. This task adds linker flags to make `nsl_cpu_peak_reset` and `nsl_cpu_peak_bytes` visible in the .dll's export table when `test-hooks` is active.

**Files:**
- Modify: `crates/nsl-codegen/src/linker.rs`

### Steps

- [ ] **Step 1: Read the existing `link_shared_gcc` and `link_shared_msvc` functions**

Run: `sed -n '660,780p' crates/nsl-codegen/src/linker.rs`

Confirm:
- `link_shared_gcc` around line 660 — handles Linux/macOS.
- `link_shared_msvc` around line 717 — handles Windows MSVC.
- Both are called from a dispatcher (grep `link_shared_gcc\|link_shared_msvc` for callers).

- [ ] **Step 2: Plumb a `test_hooks_active` flag through to both linkers**

The cleanest way to know if test-hooks is active at codegen time: read the feature at compile time via `cfg(feature = "test-hooks")` inside `nsl-codegen`. But since nsl-codegen forwards the feature from its own `[features]`, we can compile a single linker path with cfg-gated symbol exports.

In both `link_shared_gcc` and `link_shared_msvc`, add conditional linker args at the end of each function, right before the command is run:

For `link_shared_gcc` (around line 660), add before `let status = cmd.status()...`:

```rust
    // Item #5: re-export test-hooks peak FFIs when the feature is active,
    // so the resulting .so/.dylib's export table exposes them to dlopen.
    #[cfg(feature = "test-hooks")]
    {
        if cfg!(target_os = "linux") {
            cmd.arg("-Wl,-u,nsl_cpu_peak_reset");
            cmd.arg("-Wl,-u,nsl_cpu_peak_bytes");
            cmd.arg("-Wl,--export-dynamic-symbol=nsl_cpu_peak_reset");
            cmd.arg("-Wl,--export-dynamic-symbol=nsl_cpu_peak_bytes");
        } else if cfg!(target_os = "macos") {
            cmd.arg("-Wl,-u,_nsl_cpu_peak_reset");
            cmd.arg("-Wl,-u,_nsl_cpu_peak_bytes");
            // On macOS, staticlib symbols are exported by default unless
            // -exported_symbols_list is set.  -u forces them to be linked in.
        }
    }
```

For `link_shared_msvc` (around line 717), add before `let status = cmd.status()...`:

```rust
    // Item #5: export test-hooks peak FFIs so GetProcAddress can resolve them.
    #[cfg(feature = "test-hooks")]
    {
        cmd.arg("/INCLUDE:nsl_cpu_peak_reset");
        cmd.arg("/INCLUDE:nsl_cpu_peak_bytes");
        cmd.arg("/EXPORT:nsl_cpu_peak_reset");
        cmd.arg("/EXPORT:nsl_cpu_peak_bytes");
    }
```

On MSVC, `/INCLUDE` forces the linker to pull the symbol in from the staticlib, and `/EXPORT` adds it to the DLL's export table.

- [ ] **Step 3: Build to verify no compile-time regression**

Run: `cargo build -p nsl-codegen`
Expected: succeeds (new code is cfg-gated behind test-hooks).

Run: `cargo build -p nsl-codegen --features test-hooks`
Expected: succeeds.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/linker.rs
git commit -m "feat(codegen): re-export peak FFIs from shared-library when test-hooks active"
```

---

## Task 4: Create the 4-parameter MLP fixture

**Files:**
- Create: `crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl`

### Steps

- [ ] **Step 1: Verify the NSL surface syntax for multi-param models**

Check existing examples:

```bash
grep -l "w1:\|w2:" examples/*.nsl models/**/*.nsl 2>/dev/null | head -5
```

Pick a working multi-parameter model and confirm the surface form (`model MyModel:` / `fn forward(self, x: Tensor)` / matmul via `@` / literal init via `ones([...])` and `zeros([...])`).

Also verify `x @ self.w1 + self.b1` is accepted by the parser — the prior fixtures used a single matmul; this one uses matmul + bias add. If broadcasting isn't supported, adjust to `(x @ self.w1) + self.b1` with explicit grouping, or skip the bias-add and compute `x @ self.w1` then `h + self.b1` as separate steps.

- [ ] **Step 2: Write the fixture**

Create `crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl`:

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

- [ ] **Step 3: Smoke-test the fixture compiles**

Run (from the worktree root):

```bash
cargo build --features test-hooks -p nsl-cli --bin nsl && \
./target/debug/nsl check crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl
```

(On Windows, the binary is `./target/debug/nsl.exe`.)

Expected: the `nsl check` command succeeds (exit 0) for both `--source-ad` and default modes:

```bash
./target/debug/nsl check crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl
./target/debug/nsl check --source-ad crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl
```

If either fails, fix the fixture syntax. Likely surface-level issues: bias addition shape mismatch, model field initialization style, train-block keyword ordering.

- [ ] **Step 4: Smoke-test shared-library build works**

```bash
./target/debug/nsl build --shared-library crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl
```

Expected: produces `libfase_peak_memory.{so,dylib,dll}` somewhere on disk (check `linker.rs:777` `default_shared_lib_path` for the exact convention — likely alongside the input fixture).

If this fails, the issue is in the shared-library codegen path, not the fixture. Diagnose and either fix or report BLOCKED — the rest of item #5 depends on this working.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl
git commit -m "test(fase): 4-parameter MLP fixture for peak-memory regression"
```

---

## Task 5: Write the in-process test harness

**Files:**
- Create: `crates/nsl-codegen/tests/fase_peak_memory.rs` — entry point, cfg-gated
- Create: `crates/nsl-codegen/tests/fase_peak_memory_impl.rs` — actual test body

### Steps

- [ ] **Step 1: Create the cfg-gated outer test file**

Create `crates/nsl-codegen/tests/fase_peak_memory.rs`:

```rust
//! Item #5: FASE Deferred peak-memory regression test.
//!
//! Only compiles with --features test-hooks.  Run via:
//!     cargo test -p nsl-codegen --features test-hooks --test fase_peak_memory

#[cfg(feature = "test-hooks")]
#[path = "fase_peak_memory_impl.rs"]
mod test_impl;
```

When `test-hooks` is off, this file compiles to an empty test crate — no libloading dep pulled in, no cost.

- [ ] **Step 2: Write the test body**

Create `crates/nsl-codegen/tests/fase_peak_memory_impl.rs`:

```rust
use libloading::{Library, Symbol};
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fase_peak_memory.nsl")
}

/// Invoke `nsl build --shared-library [--source-ad] <fixture>` and return
/// the path to the resulting shared library.
fn build_fixture(source_ad: bool, out_dir: &Path) -> PathBuf {
    let fixture = fixture_path();

    // Determine output suffix so two variants don't clobber each other.
    let suffix = if source_ad { "source_ad" } else { "tape_ad" };
    let out_name = format!("libfase_peak_memory_{}.{}", suffix, std::env::consts::DLL_EXTENSION);
    let out_path = out_dir.join(out_name);

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("build")
        .arg("--shared-library")
        .arg(&fixture)
        .arg("-o")
        .arg(&out_path);
    if source_ad {
        cmd.arg("--source-ad");
    }
    let status = cmd.status().expect("failed to spawn nsl build");
    assert!(
        status.success(),
        "nsl build --shared-library failed (source_ad={})",
        source_ad
    );
    assert!(
        out_path.exists(),
        "expected output at {:?} but not produced",
        out_path
    );
    out_path
}

/// Load the .dll, reset the peak counter, run `main`, read the peak.
fn run_and_measure_peak(dll_path: &Path) -> usize {
    unsafe {
        let lib = Library::new(dll_path).expect("load shared library");
        let reset: Symbol<unsafe extern "C" fn()> = lib
            .get(b"nsl_cpu_peak_reset")
            .expect("nsl_cpu_peak_reset symbol not exported from .dll");
        let peak: Symbol<unsafe extern "C" fn() -> i64> = lib
            .get(b"nsl_cpu_peak_bytes")
            .expect("nsl_cpu_peak_bytes symbol not exported from .dll");
        let main_fn: Symbol<unsafe extern "C" fn()> = lib
            .get(b"main")
            .expect("main symbol not exported from .dll");

        reset();
        main_fn();
        peak() as usize
    }
}

#[test]
fn source_ad_peak_is_lower_than_tape_ad() {
    let out_dir = tempfile::TempDir::new().expect("tempdir");

    let tape_dll = build_fixture(/*source_ad=*/ false, out_dir.path());
    let source_dll = build_fixture(/*source_ad=*/ true, out_dir.path());

    let peak_tape = run_and_measure_peak(&tape_dll);
    let peak_source = run_and_measure_peak(&source_dll);

    // Expected delta: the hook frees 3 of 4 parameter gradients before
    // their peers materialise.  The two 256×256 weights contribute ~256 KB
    // each; even if bookkeeping noise eats ~100 KB, we should see at
    // least 150 KB lower peak.
    const MIN_DELTA_BYTES: usize = 150_000;

    println!("peak_tape   = {} bytes", peak_tape);
    println!("peak_source = {} bytes", peak_source);
    println!("delta       = {} bytes", peak_tape.saturating_sub(peak_source));

    assert!(
        peak_tape >= peak_source + MIN_DELTA_BYTES,
        "expected source-AD peak at least {} bytes below tape-AD; got tape={} source={} (delta={})",
        MIN_DELTA_BYTES,
        peak_tape,
        peak_source,
        peak_tape.saturating_sub(peak_source),
    );
}
```

If `std::env::consts::DLL_EXTENSION` doesn't exist in the stable Rust the project uses, substitute platform-specific logic:

```rust
let ext = if cfg!(windows) { "dll" } else if cfg!(target_os = "macos") { "dylib" } else { "so" };
let out_name = format!("libfase_peak_memory_{}.{}", suffix, ext);
```

On Windows, the filename convention is `foo.dll` not `libfoo.dll`; adjust if the build output drops the `lib` prefix on Windows (check `default_shared_lib_path` in linker.rs:777).

If the `-o` flag isn't accepted by `nsl build --shared-library` (the plan assumes it is; verify in Task 4 smoke-test), fall back to: build with default path, then rename/move the output into `out_dir`. Or build directly in `out_dir` via `cwd`.

- [ ] **Step 3: Run the test**

Run: `cargo test -p nsl-codegen --features test-hooks --test fase_peak_memory -- source_ad_peak_is_lower_than_tape_ad --nocapture`

Possible outcomes:

- **PASS with delta ≥ 150 KB:** done. Record the observed peaks in the final report.
- **FAIL — symbol not exported:**
  - Error: `nsl_cpu_peak_reset symbol not exported from .dll`.
  - Investigate Task 3's linker fix. On Linux, run `nm -D libfase_peak_memory_tape_ad.so | grep peak` — should list both symbols. On macOS, `nm -gU`. On Windows, `dumpbin /EXPORTS libfase_peak_memory_tape_ad.dll | findstr peak`.
  - If symbols are missing on ONE platform specifically, gate the test `#[cfg(not(target_os = "windows"))]` (or whichever platform) and file a follow-up.
- **FAIL — delta too small:**
  - Print observed peaks. Common causes: the fixture's "gradients" don't dominate total allocations (e.g., the tape's internal state is much bigger than expected). Either scale up the fixture (e.g., `[512, 512]` weights = 1MB gradients each) OR adjust the threshold.
  - Don't lower the threshold blindly — establish a noise floor first by running 5+ times and measuring variance.
- **FAIL — delta is negative (source-AD peak HIGHER):**
  - Real bug. Likely item #4's hook isn't actually firing (e.g., fixture's param shape doesn't match `named_param_var_ids` resolution). Halt and report DONE_WITH_CONCERNS with the numbers.

- [ ] **Step 4: Run without the feature to confirm zero regression**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | head -5`
Expected: same pass counts as before Task 5. The new test file contributes 0 tests (it's cfg-gated out). `libloading` is not compiled in.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/tests/fase_peak_memory.rs crates/nsl-codegen/tests/fase_peak_memory_impl.rs
git commit -m "test(fase): peak-memory regression test via libloading"
```

---

## Task 6: Final verification + memory note

- [ ] **Step 1: Full workspace build (default features)**

Run: `cargo build --workspace`
Expected: succeeds with no new warnings related to the test-hooks work.

- [ ] **Step 2: Full nsl-codegen test suite (default features)**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | head -15`
Expected: all green, same counts as before the session. No `fase_peak_memory` tests run — cfg-gated out.

- [ ] **Step 3: Full nsl-codegen test suite with test-hooks**

Run: `cargo test -p nsl-codegen --features test-hooks 2>&1 | grep "^test result" | head -15`
Expected: all green. The `source_ad_peak_is_lower_than_tape_ad` test now runs and passes.

- [ ] **Step 4: Verify production .dll does NOT export peak FFIs**

Build a production .dll (no test-hooks) from the fixture:

```bash
# Linux
cargo build --workspace --release
./target/release/nsl build --shared-library crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl -o /tmp/prod.so
nm -D /tmp/prod.so | grep peak || echo "OK: no peak symbols in production build"
```

Expected: `OK: no peak symbols in production build`. Equivalent commands on macOS (`nm -gU`) and Windows (`dumpbin /EXPORTS prod.dll | findstr peak`).

If symbols ARE present in a production build, Task 1's cfg gating is broken. Investigate.

- [ ] **Step 5: Update FASE project memory note**

Edit `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_fase_deferred_integration.md`. Find the line:

```
- **Item #5:** peak-memory regression test (depends on #4, now lands)
```

Replace with:

```
- **Item #5:** ✅ shipped 2026-04-14 — test-hooks cargo feature gates CPU heap peak tracking (AtomicUsize in memory.rs) + two FFIs (`nsl_cpu_peak_reset`, `nsl_cpu_peak_bytes`). Fixture `fase_peak_memory.nsl` (4-param MLP, 256×256 weights) compiled as .dll both ways; integration test loads each via `libloading`, measures peak, asserts source-AD delta ≥ 150 KB. Production builds are byte-identical (feature off). Spec: `docs/superpowers/specs/2026-04-14-fase-peak-memory-regression-design.md`.
```

If the exact surrounding text differs, grep `Item #5` in that file and adapt.

- [ ] **Step 6: Report**

Summarize: commits shipped, observed peak numbers (tape vs source delta), remaining FASE roadmap item (#6 training-report CLI).

---

## Summary of files touched

- **Modified:** `crates/nsl-runtime/Cargo.toml` (Task 1)
- **Modified:** `crates/nsl-runtime/src/memory.rs` (Task 1)
- **Modified:** `crates/nsl-codegen/Cargo.toml` (Task 2)
- **Modified:** `crates/nsl-cli/Cargo.toml` (Task 2, if needed)
- **Modified:** `crates/nsl-codegen/src/linker.rs` (Task 3)
- **Created:** `crates/nsl-codegen/tests/fixtures/fase_peak_memory.nsl` (Task 4)
- **Created:** `crates/nsl-codegen/tests/fase_peak_memory.rs` (Task 5)
- **Created:** `crates/nsl-codegen/tests/fase_peak_memory_impl.rs` (Task 5)
- **Modified:** memory note (Task 6)
