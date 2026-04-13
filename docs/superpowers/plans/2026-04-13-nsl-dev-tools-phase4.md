# NSL Dev Tools — Phase 4 Training Health Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `nsl run --monitor` against a `train` block per `nsl dev tools.pdf` §4.3 — emit per-step gradient/weight-norm/loss/NaN hooks during training, accumulate in a runtime collector, render the live PDF §4.3 layout from JSON snapshots.

**Architecture:** Codegen `compile_train_block` emits five FFI calls per step, gated on `compile_options.health_monitor`. Runtime `HealthCollector` (single global Mutex) accumulates loss EMA + slope, per-layer grad norm, per-tensor weight Δ-from-init, NaN counter. Snapshot flushed to `<file>.nsl-health.json` every 100 steps via codegen-side step-gated branch. CLI renders the snapshot once at run end (TTY-aware ANSI, plain append fallback). FASE-aware grad-norm hook splice handles fused per-layer backward.

**Tech Stack:** Rust workspace, Cranelift IR for instrumentation, `serde_json` for snapshots, `std::io::IsTerminal` for TTY detection.

**Spec:** `docs/superpowers/specs/2026-04-13-nsl-dev-tools-phase4-design.md`

**Branch / worktree:** Continue on `feat/dev-tools-phase1` in `c:/Users/bwiem/projects/NSL/.worktrees/dev-tools-phase1`. No new worktree.

---

## Task 1: `HealthCollector` + `HealthSnapshot` types

**Files:**
- Create: `crates/nsl-runtime/src/health/mod.rs`
- Create: `crates/nsl-runtime/src/health/collector.rs`
- Modify: `crates/nsl-runtime/src/lib.rs` — add `pub mod health;`
- Test: `crates/nsl-runtime/tests/health_collector.rs` (new)

Pure data-model task. No FFI yet (Task 2). No codegen wire-up (Task 4).

- [ ] **Step 1: Write failing tests**

Create `crates/nsl-runtime/tests/health_collector.rs`:

```rust
use nsl_runtime::health::collector::HealthCollector;

#[test]
fn nan_loss_increments_counter_and_skips_ema() {
    let mut c = HealthCollector::new();
    c.record_loss(1, 1.0);
    c.record_loss(2, f64::NAN);
    c.record_loss(3, 2.0);
    let snap = c.snapshot();
    assert_eq!(snap.nan_inf_count_window, 1);
    // EMA should reflect 1.0 and 2.0 only — NaN skipped.
    assert!(snap.loss_ema.unwrap() > 1.0 && snap.loss_ema.unwrap() < 2.0);
}

#[test]
fn loss_history_capped_at_window() {
    let mut c = HealthCollector::new();
    for i in 0..150u64 {
        c.record_loss(i, i as f64);
    }
    let snap = c.snapshot();
    // LOSS_WINDOW = 100; recent value retained.
    assert_eq!(snap.steps_in_window, 100);
    assert_eq!(snap.loss, Some(149.0));
}

#[test]
fn decreasing_loss_produces_negative_slope() {
    let mut c = HealthCollector::new();
    for i in 0..20u64 {
        c.record_loss(i, 100.0 - i as f64);
    }
    let snap = c.snapshot();
    let slope = snap.loss_ema_slope.expect("slope should exist with 20 samples");
    assert!(slope < -0.5, "expected strongly negative slope, got {}", slope);
}

#[test]
fn weight_init_then_update_produces_pct_delta() {
    let mut c = HealthCollector::new();
    c.record_weight_norm("m.l0.w", 100.0, true);
    c.record_weight_norm("m.l0.w", 105.0, false);
    let snap = c.snapshot();
    let pct = snap.per_tensor_weight_pct_delta.get("m.l0.w").copied().unwrap();
    assert!((pct - 5.0).abs() < 1e-6, "expected +5%, got {}", pct);
}

#[test]
fn grad_norm_total_is_root_sum_of_squares() {
    let mut c = HealthCollector::new();
    c.record_grad_norm("m.l0", 0, 3.0);
    c.record_grad_norm("m.l1", 1, 4.0);
    let snap = c.snapshot();
    assert!((snap.grad_norm_total.unwrap() - 5.0).abs() < 1e-6, "expected sqrt(9+16)=5, got {:?}", snap.grad_norm_total);
}

#[test]
fn should_flush_gates_at_intervals() {
    let mut c = HealthCollector::new();
    c.record_loss(0, 1.0);
    assert!(c.should_flush(), "step 0 should flush (init)");
    c.record_loss(1, 1.0);
    assert!(!c.should_flush(), "step 1 should not flush");
    c.record_loss(99, 1.0);
    assert!(!c.should_flush(), "step 99 should not flush");
    c.record_loss(100, 1.0);
    assert!(c.should_flush(), "step 100 should flush");
}

#[test]
fn snapshot_serializes_to_json() {
    let mut c = HealthCollector::new();
    c.record_loss(5, 4.2);
    c.record_grad_norm("m.l0", 0, 2.5);
    let snap = c.snapshot();
    let s = serde_json::to_string(&snap).unwrap();
    assert!(s.contains("\"step\":5"));
    assert!(s.contains("\"loss\":4.2"));
}
```

- [ ] **Step 2: Run — expect fail (unresolved)**

```
cargo test -p nsl-runtime --test health_collector 2>&1 | tail -10
```

- [ ] **Step 3: Implement collector**

Create `crates/nsl-runtime/src/health/mod.rs`:

```rust
pub mod collector;
```

Create `crates/nsl-runtime/src/health/collector.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

const LOSS_WINDOW: usize = 100;
const EMA_ALPHA: f64 = 0.05;
const FLUSH_INTERVAL_DEFAULT: u64 = 100;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HealthSnapshot {
    pub step: u64,
    pub max_steps: Option<u64>,
    pub loss: Option<f64>,
    pub loss_ema: Option<f64>,
    pub loss_ema_slope: Option<f64>,
    pub grad_norm_total: Option<f64>,
    pub per_layer_grad_norm: HashMap<u32, f64>,
    pub per_tensor_weight_pct_delta: HashMap<String, f64>,
    pub nan_inf_count_window: u64,
    pub steps_in_window: u64,
}

pub struct HealthCollector {
    step: u64,
    last_flushed_step: u64,
    flush_interval: u64,
    loss_history: VecDeque<f64>,
    loss_ema: Option<f64>,
    grad_norm_per_layer: HashMap<u32, f64>,
    weight_init: HashMap<String, f64>,
    weight_current: HashMap<String, f64>,
    nan_inf_count: u64,
}

impl Default for HealthCollector {
    fn default() -> Self { Self::new() }
}

impl HealthCollector {
    pub fn new() -> Self {
        Self {
            step: 0,
            last_flushed_step: u64::MAX,
            flush_interval: FLUSH_INTERVAL_DEFAULT,
            loss_history: VecDeque::with_capacity(LOSS_WINDOW),
            loss_ema: None,
            grad_norm_per_layer: HashMap::new(),
            weight_init: HashMap::new(),
            weight_current: HashMap::new(),
            nan_inf_count: 0,
        }
    }

    pub fn set_flush_interval(&mut self, n: u64) { self.flush_interval = n.max(1); }

    pub fn record_loss(&mut self, step: u64, value: f64) {
        self.step = step;
        if value.is_nan() || value.is_infinite() {
            self.nan_inf_count += 1;
            return;
        }
        if self.loss_history.len() == LOSS_WINDOW { self.loss_history.pop_front(); }
        self.loss_history.push_back(value);
        self.loss_ema = Some(match self.loss_ema {
            None => value,
            Some(prev) => prev * (1.0 - EMA_ALPHA) + value * EMA_ALPHA,
        });
    }

    pub fn record_grad_norm(&mut self, _path: &str, layer_idx: u32, norm: f64) {
        self.grad_norm_per_layer.insert(layer_idx, norm);
    }

    pub fn record_weight_norm(&mut self, path: &str, norm: f64, is_init: bool) {
        if is_init { self.weight_init.insert(path.to_string(), norm); }
        self.weight_current.insert(path.to_string(), norm);
    }

    pub fn snapshot(&mut self) -> HealthSnapshot {
        let last = self.loss_history.back().copied();
        let slope = if self.loss_history.len() >= 2 {
            let n = self.loss_history.len() as f64;
            let mean_x = (n - 1.0) / 2.0;
            let mean_y = self.loss_history.iter().sum::<f64>() / n;
            let (mut num, mut den) = (0.0_f64, 0.0_f64);
            for (i, &y) in self.loss_history.iter().enumerate() {
                let dx = i as f64 - mean_x;
                num += dx * (y - mean_y);
                den += dx * dx;
            }
            if den > 0.0 { Some(num / den) } else { None }
        } else { None };

        let grad_norm_total = if self.grad_norm_per_layer.is_empty() { None }
            else {
                let sumsq: f64 = self.grad_norm_per_layer.values().map(|n| n * n).sum();
                Some(sumsq.sqrt())
            };

        let per_tensor_weight_pct_delta: HashMap<String, f64> = self.weight_current.iter()
            .filter_map(|(path, cur)| {
                self.weight_init.get(path).map(|init| {
                    let pct = if *init > 0.0 { (cur - init) / init * 100.0 } else { 0.0 };
                    (path.clone(), pct)
                })
            })
            .collect();

        HealthSnapshot {
            step: self.step,
            max_steps: None,
            loss: last,
            loss_ema: self.loss_ema,
            loss_ema_slope: slope,
            grad_norm_total,
            per_layer_grad_norm: self.grad_norm_per_layer.clone(),
            per_tensor_weight_pct_delta,
            nan_inf_count_window: self.nan_inf_count,
            steps_in_window: self.loss_history.len() as u64,
        }
    }

    pub fn should_flush(&mut self) -> bool {
        if self.step == 0
            || self.last_flushed_step == u64::MAX
            || self.step.saturating_sub(self.last_flushed_step) >= self.flush_interval
        {
            self.last_flushed_step = self.step;
            true
        } else { false }
    }
}
```

Add `pub mod health;` to `crates/nsl-runtime/src/lib.rs`.

If `serde` / `serde_json` aren't already dev-deps for this crate's tests, add them — they should already be in the Cargo.toml from earlier phases.

- [ ] **Step 4: Run tests — expect 7 passed**

```
cargo test -p nsl-runtime --test health_collector 2>&1 | tail -10
cargo build -p nsl-runtime 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```
git add crates/nsl-runtime/src/health/ crates/nsl-runtime/src/lib.rs \
        crates/nsl-runtime/tests/health_collector.rs
git commit -m "feat(runtime): HealthCollector + HealthSnapshot for training-step metrics"
```

---

## Task 2: Runtime FFI hooks (`health/ffi.rs`)

**Files:**
- Create: `crates/nsl-runtime/src/health/ffi.rs`
- Modify: `crates/nsl-runtime/src/health/mod.rs` — add `pub mod ffi;`
- Test: `crates/nsl-runtime/tests/health_ffi.rs` (new)

- [ ] **Step 1: Write failing tests**

Create `crates/nsl-runtime/tests/health_ffi.rs`:

```rust
use nsl_runtime::health::ffi::{
    nsl_health_record_loss, nsl_health_flush_snapshot,
    nsl_health_record_grad_norm, nsl_health_record_weight_norm,
};

// NOTE: these tests share a global static. Run sequentially with --test-threads=1
// or the assertions become flaky across tests.

fn flush_to_string() -> String {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    let bytes = path.as_bytes();
    let rc = unsafe { nsl_health_flush_snapshot(bytes.as_ptr(), bytes.len()) };
    assert_eq!(rc, 0, "flush returned non-zero: {}", rc);
    std::fs::read_to_string(tmp.path()).unwrap()
}

#[test]
fn record_loss_then_flush_writes_json_with_step_and_loss() {
    nsl_health_record_loss(2.5, 7);
    let json = flush_to_string();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["step"], 7);
    assert!((v["loss"].as_f64().unwrap() - 2.5).abs() < 1e-6);
}

#[test]
fn record_grad_norm_emits_layer_entry() {
    let path = "m.transformer.h.4.attn.wq";
    let bytes = path.as_bytes();
    unsafe { nsl_health_record_grad_norm(bytes.as_ptr(), bytes.len(), 4, 12.5); }
    let json = flush_to_string();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!((v["per_layer_grad_norm"]["4"].as_f64().unwrap() - 12.5).abs() < 1e-6);
}

#[test]
fn flush_to_null_path_returns_zero() {
    let rc = unsafe { nsl_health_flush_snapshot(std::ptr::null(), 0) };
    assert_eq!(rc, 0);
}

#[test]
fn flush_invalid_utf8_returns_2() {
    let bytes = [0xFFu8, 0xFE, 0xFD];
    let rc = unsafe { nsl_health_flush_snapshot(bytes.as_ptr(), bytes.len()) };
    assert_eq!(rc, 2);
}

#[test]
fn weight_norm_is_init_then_update_round_trips() {
    let path = "m.l0.w";
    let b = path.as_bytes();
    unsafe {
        nsl_health_record_weight_norm(b.as_ptr(), b.len(), 100.0, true);
        nsl_health_record_weight_norm(b.as_ptr(), b.len(), 110.0, false);
    }
    let json = flush_to_string();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let pct = v["per_tensor_weight_pct_delta"]["m.l0.w"].as_f64().unwrap();
    assert!((pct - 10.0).abs() < 1e-6, "expected +10%, got {}", pct);
}
```

If `tempfile` isn't a dev-dep of `nsl-runtime`, it should be from earlier phases.

- [ ] **Step 2: Run — expect fail**

```
cargo test -p nsl-runtime --test health_ffi -- --test-threads=1 2>&1 | tail -10
```

- [ ] **Step 3: Implement FFI**

Create `crates/nsl-runtime/src/health/ffi.rs`:

```rust
//! C-callable hooks emitted by codegen at training-step boundaries.
//!
//! Single global Mutex<HealthCollector> is fine today (NSL backward is
//! single-stream sequential). When CPDT lands and overlaps backward across
//! layers, the future fix is per-layer sharded collectors merged at snapshot
//! time. Don't refactor until profiling shows contention.

use once_cell::sync::Lazy;
use std::sync::Mutex;
use super::collector::HealthCollector;

static COLLECTOR: Lazy<Mutex<HealthCollector>> = Lazy::new(|| Mutex::new(HealthCollector::new()));

#[no_mangle]
pub extern "C" fn nsl_health_record_loss(value: f64, step: u64) {
    COLLECTOR.lock().unwrap().record_loss(step, value);
}

/// # Safety
/// Caller must guarantee (path_ptr, path_len) refers to valid UTF-8 bytes
/// they own for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn nsl_health_record_grad_norm(
    path_ptr: *const u8, path_len: usize,
    layer_idx: u32, norm: f64,
) {
    if path_ptr.is_null() { return; }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    if let Ok(path) = std::str::from_utf8(bytes) {
        COLLECTOR.lock().unwrap().record_grad_norm(path, layer_idx, norm);
    }
}

/// # Safety
/// Same as `nsl_health_record_grad_norm`.
#[no_mangle]
pub unsafe extern "C" fn nsl_health_record_weight_norm(
    path_ptr: *const u8, path_len: usize,
    norm: f64, is_init: bool,
) {
    if path_ptr.is_null() { return; }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    if let Ok(path) = std::str::from_utf8(bytes) {
        COLLECTOR.lock().unwrap().record_weight_norm(path, norm, is_init);
    }
}

/// Returns: 0 ok, 1 serde fail, 2 invalid UTF-8 path, 3 io fail.
///
/// # Safety
/// Same.
#[no_mangle]
pub unsafe extern "C" fn nsl_health_flush_snapshot(
    path_ptr: *const u8, path_len: usize,
) -> i32 {
    let snap = COLLECTOR.lock().unwrap().snapshot();
    let json = match serde_json::to_string(&snap) {
        Ok(s) => s,
        Err(_) => return 1,
    };
    if path_ptr.is_null() {
        println!("{}", json);
        return 0;
    }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    let s = match std::str::from_utf8(bytes) { Ok(s) => s, Err(_) => return 2 };
    match std::fs::write(std::path::Path::new(s), json) {
        Ok(_) => 0, Err(_) => 3,
    }
}

/// Codegen-emitted at train-block entry to override the default flush interval.
#[no_mangle]
pub extern "C" fn nsl_health_set_flush_interval(n: u64) {
    COLLECTOR.lock().unwrap().set_flush_interval(n);
}
```

Add `pub mod ffi;` to `crates/nsl-runtime/src/health/mod.rs`.

If `once_cell` isn't already a dep of `nsl-runtime`, it should be from Phase 2 Task 7.

- [ ] **Step 4: Run tests — expect 5 passed**

```
cargo test -p nsl-runtime --test health_ffi -- --test-threads=1 2>&1 | tail -10
cargo build -p nsl-runtime 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```
git add crates/nsl-runtime/src/health/ffi.rs \
        crates/nsl-runtime/src/health/mod.rs \
        crates/nsl-runtime/tests/health_ffi.rs
git commit -m "feat(runtime): FFI hooks for health monitor + flush_snapshot"
```

---

## Task 3: `nsl_tensor_l2_norm` exposure + builtins registration

**Files:**
- Modify: `crates/nsl-runtime/src/tensor/mod.rs` — one-line FFI lift.
- Modify: `crates/nsl-codegen/src/builtins.rs` — register six new symbols.
- Test: `crates/nsl-runtime/tests/health_ffi.rs` (extend).

- [ ] **Step 1: Locate existing tensor FFI patterns**

```
grep -n "pub extern \"C\" fn nsl_tensor_item\|fn tensor_l2_norm" crates/nsl-runtime/src/tensor/mod.rs | head -5
```

Confirm `tensor_l2_norm(t: &NslTensor) -> f64` exists privately (around line 3874) and `nsl_tensor_item` shows the i64-handle FFI pattern.

- [ ] **Step 2: Write failing test**

Append to `crates/nsl-runtime/tests/health_ffi.rs`:

```rust
#[test]
fn tensor_l2_norm_extern_returns_finite() {
    use nsl_runtime::tensor::nsl_tensor_l2_norm;
    // Construct a CPU tensor via whichever runtime helper exists. Look at
    // existing tensor unit tests for the smallest valid shape.
    let t_handle = nsl_runtime::tensor::test_helpers::cpu_tensor_from_vec(
        vec![3.0_f64, 4.0_f64], &[2],
    );
    let norm = nsl_tensor_l2_norm(t_handle);
    // sqrt(9 + 16) = 5
    assert!((norm - 5.0).abs() < 1e-6, "expected 5.0, got {}", norm);
}
```

If `cpu_tensor_from_vec` isn't a real helper, use whatever existing test in `crates/nsl-runtime/tests/` constructs a tensor — copy that pattern. The exact constructor name doesn't matter; the assertion does.

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-runtime --test health_ffi tensor_l2_norm -- --test-threads=1 2>&1 | tail -10
```

- [ ] **Step 4: Add the FFI**

In `crates/nsl-runtime/src/tensor/mod.rs`, alongside the existing private `tensor_l2_norm`:

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_l2_norm(t: i64) -> f64 {
    let tensor = NslTensor::from_ptr(t);
    tensor_l2_norm(&tensor)
}
```

Match `nsl_tensor_item`'s pattern exactly — same `from_ptr` call shape, same lack of `unsafe` on the extern (the unsafety is wrapped inside `from_ptr`), same lack of null check.

- [ ] **Step 5: Register all six new symbols in `builtins.rs`**

In `crates/nsl-codegen/src/builtins.rs::RUNTIME_FUNCTIONS`, add entries:

```rust
RuntimeFunction { name: "nsl_tensor_l2_norm",            params: &[I64], ret: Some(F64) },
RuntimeFunction { name: "nsl_health_record_loss",        params: &[F64, I64], ret: None },
RuntimeFunction { name: "nsl_health_record_grad_norm",   params: &[I64, I64, I32, F64], ret: None },
RuntimeFunction { name: "nsl_health_record_weight_norm", params: &[I64, I64, F64, I8], ret: None },
RuntimeFunction { name: "nsl_health_flush_snapshot",     params: &[I64, I64], ret: Some(I32) },
RuntimeFunction { name: "nsl_health_set_flush_interval", params: &[I64], ret: None },
```

(Adapt to the actual `RuntimeFunction` shape and signature-vec idiom — read 5 lines above to match. Pointer params are `I64` per the workspace convention.)

- [ ] **Step 6: Run tests**

```
cargo test -p nsl-runtime --test health_ffi -- --test-threads=1 2>&1 | tail -10
cargo build -p nsl-codegen -p nsl-runtime 2>&1 | tail -5
```

Expected: 6 passed in health_ffi (5 prior + tensor_l2_norm); both crates build clean.

- [ ] **Step 7: Commit**

```
git add crates/nsl-runtime/src/tensor/mod.rs \
        crates/nsl-codegen/src/builtins.rs \
        crates/nsl-runtime/tests/health_ffi.rs
git commit -m "feat(runtime,codegen): expose nsl_tensor_l2_norm FFI; register health symbols"
```

---

## Task 4: Codegen instrumentation in `compile_train_block`

**Files:**
- Modify: `crates/nsl-codegen/src/lib.rs` — add `health_monitor: bool` + `health_flush_interval: Option<u64>` to `CompileOptions`.
- Modify: `crates/nsl-codegen/src/stmt.rs::compile_train_block` — emit hooks per spec §4.3.
- Modify: every `CompileOptions { ... }` literal site — add the new fields.
- Test: `crates/nsl-codegen/tests/health_codegen.rs` (new) + extend `crates/nsl-codegen/src/test_helpers.rs`.

This is the heaviest task. Splice points depend on whether FASE per-layer fused backward is active.

- [ ] **Step 1: Locate FASE detection + per-layer backward shape**

```
grep -n "fase\|fase_enabled\|FASE\|fn compile_train_block" crates/nsl-codegen/src/stmt.rs | head -20
sed -n '2900,2970p' crates/nsl-codegen/src/stmt.rs   # train block entry
```

Identify:
- The flag/condition that distinguishes FASE per-layer fused backward from standard backward.
- The Cranelift block(s) where each layer's gradient is computed (FASE) or where the full grad list is returned (standard).
- The `step_count_var` slot location.
- How `intern_string_constant` (or equivalent) emits static UTF-8 — search:
  ```
  grep -n "intern_string_constant\|static.*UTF\|emit_string_constant\|data_id_for_string" crates/nsl-codegen/src/
  ```

- [ ] **Step 2: Add CompileOptions fields**

In `crates/nsl-codegen/src/lib.rs::CompileOptions`:

```rust
pub health_monitor: bool,
pub health_flush_interval: Option<u64>,
```

In `impl Default for CompileOptions`, add `health_monitor: false`, `health_flush_interval: None`.

Find every `CompileOptions { ... }` literal and add the two fields. Phase 2/3 patches show the pattern (literals in `crates/nsl-cli/src/main.rs:848`, `:1034`, and any others):

```
grep -rn "CompileOptions {" crates/
```

For Build path: `health_monitor: false, health_flush_interval: None`.
For Run path: `health_monitor: monitor && has_train_block, health_flush_interval: None` (Task 6 wires `has_train_block`; for now use `monitor` alone — Task 6 refines).

- [ ] **Step 3: Write failing integration test**

Create `crates/nsl-codegen/tests/health_codegen.rs`:

```rust
//! Phase 4 Task 4: compile_train_block emits health hooks when
//! compile_options.health_monitor is set.
#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;

#[test]
fn compile_with_health_monitor_writes_health_json_after_run_step() {
    // Mirror test_helpers::compile_and_capture_manifest from Phase 2 Task 5,
    // adapted for the train-block path. The helper must:
    //   1. Compile a tiny train fixture with health_monitor=true to a binary or
    //      object that can be run in-process.
    //   2. Invoke the compiled train step (one iteration) so hooks fire.
    //   3. Read <out>.nsl-health.json from disk.
    //   4. Return the parsed HealthSnapshot.

    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>, w: Tensor<[16, 16], bf16>) -> Tensor:
    return x @ w

train:
    model = forward
    step:
        let pred = forward(x, w)
        let loss = pred.sum()
"#;
    // (Adapt the fixture to whatever minimal-train-block syntax the parser
    // accepts — copy from existing train-block test fixtures in the repo.)

    let mut opts = CompileOptions::default();
    opts.health_monitor = true;
    opts.target_gpu = "h100".to_string();
    opts.dtype = "bf16".to_string();

    let snap = nsl_codegen::test_helpers::compile_and_run_one_train_step(src, &opts)
        .expect("train step should execute and write health JSON");

    assert!(snap.loss.is_some(), "loss should be recorded");
    assert!(snap.steps_in_window >= 1);
}
```

If `compile_and_run_one_train_step` is intractable to implement (training requires real CUDA / live runtime / etc.), fall back to a Cranelift IR inspection test that asserts the function body for `compile_train_block` contains call instructions for `nsl_health_record_loss` when `health_monitor=true`. Either way, the assertion proves the codegen path is wired.

If both options are too expensive, ship a minimal "the option compiles cleanly + adds the right RuntimeFunction declarations" smoke test and rely on Task 7's manual smoke for the live verification.

- [ ] **Step 4: Run — expect fail**

```
cargo test -p nsl-codegen --features test-helpers --test health_codegen 2>&1 | tail -15
```

- [ ] **Step 5: Implement instrumentation in `compile_train_block`**

In `crates/nsl-codegen/src/stmt.rs::compile_train_block`, gated on `compile_options.health_monitor`:

**5a. Loss scalar after `loss = ...`, before backward:**

```rust
if self.compile_options.health_monitor {
    let loss_scalar = builder.ins().call(
        self.func_ref("nsl_tensor_item"),
        &[loss_val],
    );
    let scalar_val = builder.inst_results(loss_scalar)[0];
    let step_val = builder.ins().load(
        cranelift_codegen::ir::types::I64,
        cranelift_codegen::ir::MemFlags::trusted(),
        step_count_var, 0,
    );
    builder.ins().call(
        self.func_ref("nsl_health_record_loss"),
        &[scalar_val, step_val],
    );
}
```

(Adapt `func_ref` / `inst_results` calls to whatever the existing builders use. Match the pattern used by Phase 2 Task 5 in `compile_kernel_call`.)

**5b. Per-layer gradient norms — FASE-aware splice:**

If FASE per-layer fused backward is active (read the flag during writing-plans to confirm name — likely `compile_options.fase_enabled` or a flag on the train-block struct):

```rust
// Inside the FASE per-layer loop, between gradient compute and optimizer step
// for each layer:
if self.compile_options.health_monitor {
    let path = &param.path;                     // owned String per layer
    let path_data = self.intern_string_constant(path);
    let path_ptr_val = builder.ins().symbol_value(I64, path_data);
    let path_len_val = builder.ins().iconst(I64, path.len() as i64);
    let layer_idx_val = builder.ins().iconst(I32, parse_layer_idx(path) as i64);
    let norm_inst = builder.ins().call(
        self.func_ref("nsl_tensor_l2_norm"),
        &[grad_ptr_val],
    );
    let norm_val = builder.inst_results(norm_inst)[0];
    builder.ins().call(
        self.func_ref("nsl_health_record_grad_norm"),
        &[path_ptr_val, path_len_val, layer_idx_val, norm_val],
    );
}
```

If FASE is not active (standard backward returns the full grad list intact), append the same loop after backward. Iterate over `enumerate_model_tensor_paths()` paired with the per-param gradient pointer values returned by `nsl_tape_backward`.

Helper at codegen time:

```rust
fn parse_layer_idx(path: &str) -> u32 {
    path.split('.').rev()
        .find_map(|seg| seg.parse::<u32>().ok())
        .unwrap_or(u32::MAX)
}
```

**5c. Weight norms — step 0 init + every 100 steps:**

After backward (regardless of FASE), emit two Cranelift branches:

```rust
if self.compile_options.health_monitor {
    let zero = builder.ins().iconst(I64, 0);
    let step_val = builder.ins().load(I64, MemFlags::trusted(), step_count_var, 0);
    let step_is_zero = builder.ins().icmp(IntCC::Equal, step_val, zero);

    let init_block = builder.create_block();
    let after_init_block = builder.create_block();
    let periodic_check_block = builder.create_block();
    let periodic_block = builder.create_block();
    let after_periodic_block = builder.create_block();

    builder.ins().brif(step_is_zero, init_block, &[], periodic_check_block, &[]);

    // init_block: emit per-param weight-norm record with is_init=true
    builder.switch_to_block(init_block);
    for (path, param_ptr_val) in enumerated_params.iter() {
        emit_weight_norm_record(self, builder, path, *param_ptr_val, /*is_init=*/ 1);
    }
    builder.ins().jump(after_init_block, &[]);
    builder.seal_block(init_block);

    builder.switch_to_block(after_init_block);
    builder.ins().jump(after_periodic_block, &[]);

    // periodic_check_block: step % 100 == 0?
    builder.switch_to_block(periodic_check_block);
    let mod_val = builder.ins().urem_imm(step_val, 100);
    let due = builder.ins().icmp(IntCC::Equal, mod_val, zero);
    builder.ins().brif(due, periodic_block, &[], after_periodic_block, &[]);

    builder.switch_to_block(periodic_block);
    for (path, param_ptr_val) in enumerated_params.iter() {
        emit_weight_norm_record(self, builder, path, *param_ptr_val, /*is_init=*/ 0);
    }
    builder.ins().jump(after_periodic_block, &[]);

    builder.seal_block(periodic_check_block);
    builder.seal_block(periodic_block);
    builder.switch_to_block(after_periodic_block);
}
```

`emit_weight_norm_record` is a small helper near the loss emission — same pattern as the grad-norm record but calling `nsl_health_record_weight_norm`.

**5d. Snapshot flush — codegen-side step-gated branch (every 100 steps):**

After all per-step body and weight-norm work, emit:

```rust
if self.compile_options.health_monitor {
    let zero = builder.ins().iconst(I64, 0);
    let step_val = builder.ins().load(I64, MemFlags::trusted(), step_count_var, 0);
    let mod_val = builder.ins().urem_imm(step_val, 100);
    let flush_due = builder.ins().icmp(IntCC::Equal, mod_val, zero);

    let do_flush_block = builder.create_block();
    let after_flush_block = builder.create_block();
    builder.ins().brif(flush_due, do_flush_block, &[], after_flush_block, &[]);

    builder.switch_to_block(do_flush_block);
    let snap_path = self.compile_options.output_path
        .as_ref()
        .map(|p| format!("{}.nsl-health.json", p.display()))
        .unwrap_or_else(|| "nsl-health.json".to_string());
    let path_data = self.intern_string_constant(&snap_path);
    let snap_path_ptr = builder.ins().symbol_value(I64, path_data);
    let snap_path_len = builder.ins().iconst(I64, snap_path.len() as i64);
    builder.ins().call(
        self.func_ref("nsl_health_flush_snapshot"),
        &[snap_path_ptr, snap_path_len],
    );
    builder.ins().jump(after_flush_block, &[]);
    builder.seal_block(do_flush_block);
    builder.switch_to_block(after_flush_block);
}
```

If `compile_options.output_path` doesn't exist yet (Phase 2 Task 6 added a `manifest_output_path` for kernel timing — different field), default the snap path to `<source-file>.nsl-health.json` derived from `compile_options.profile_source_file_name`. If that's also unset, fall back to a literal `"nsl-health.json"` in the cwd.

**5e. Optional flush-interval setter:**

If `compile_options.health_flush_interval` is `Some(n)`, emit `nsl_health_set_flush_interval(n)` once at train-block entry (before the epoch loop). Otherwise skip — runtime uses default 100.

```rust
if let Some(n) = self.compile_options.health_flush_interval {
    let n_val = builder.ins().iconst(I64, n as i64);
    builder.ins().call(self.func_ref("nsl_health_set_flush_interval"), &[n_val]);
}
```

**Scope guardrail on FASE detection:** if FASE path is genuinely too tangled to splice into during this task, ship the standard-backward path only. Document the FASE limitation in code as a `// TODO(phase4-fase): splice into FASE per-layer loop` near the train-block entry. The PDF doesn't require FASE coverage for the first ship; standard backward exercises every code path.

- [ ] **Step 6: Add the test helper if you went with Option A in Step 3**

In `crates/nsl-codegen/src/test_helpers.rs`:

```rust
#[cfg(any(test, feature = "test-helpers"))]
pub fn compile_and_run_one_train_step(
    src: &str,
    opts: &crate::CompileOptions,
) -> Result<nsl_runtime::health::collector::HealthSnapshot, String> {
    // Mirror compile_and_capture_manifest from Phase 2:
    // 1. Lex + parse + analyze.
    // 2. Compile to an in-process callable (whatever Phase 2 used).
    // 3. Call the entry function for one step.
    // 4. Read <out>.nsl-health.json from disk.
    // 5. serde_json::from_str into HealthSnapshot.
    todo!("mirror compile_and_capture_manifest pattern; if intractable, return Err and rely on smoke test")
}
```

If implementing this helper would take more than 30 minutes (e.g., compile-to-binary requires linker + spawn + wait), skip it: change the integration test to assert manifest-side behavior instead (CompileOptions.health_monitor = true → builds clean → IR contains expected calls), and leave end-to-end runtime verification to Task 7's manual smoke.

- [ ] **Step 7: Run tests**

```
cargo test -p nsl-codegen --features test-helpers --test health_codegen 2>&1 | tail -15
cargo test -p nsl-codegen --tests 2>&1 | tail -10
cargo build -p nsl-codegen -p nsl-cli 2>&1 | tail -5
```

Expected: green across the board.

- [ ] **Step 8: Commit**

```
git add crates/nsl-codegen/src/lib.rs \
        crates/nsl-codegen/src/stmt.rs \
        crates/nsl-codegen/src/test_helpers.rs \
        crates/nsl-codegen/tests/health_codegen.rs \
        crates/nsl-cli/src/main.rs   # CompileOptions literal updates
git commit -m "feat(codegen): compile_train_block emits per-step health hooks"
```

---

## Task 5: CLI renderer (`health_monitor.rs`)

**Files:**
- Create: `crates/nsl-cli/src/health_monitor.rs`
- Modify: `crates/nsl-cli/src/lib.rs` — add `pub mod health_monitor;`
- Test: `crates/nsl-cli/tests/health_monitor.rs` (new)

- [ ] **Step 1: Write failing tests**

Create `crates/nsl-cli/tests/health_monitor.rs`:

```rust
use nsl_cli::health_monitor::{HealthRenderer, group_by_layer_prefix};
use nsl_runtime::health::collector::HealthSnapshot;
use std::collections::HashMap;

fn snap_with_grad_norms(per_layer: &[(u32, f64)], nan: u64, slope: Option<f64>) -> HealthSnapshot {
    let mut s = HealthSnapshot::default();
    s.step = 100;
    s.loss = Some(4.23);
    s.grad_norm_total = Some(12.4);
    s.per_layer_grad_norm = per_layer.iter().copied().collect();
    s.nan_inf_count_window = nan;
    s.steps_in_window = 100;
    s.loss_ema_slope = slope;
    s
}

#[test]
fn renders_header_and_per_layer_block() {
    let snap = snap_with_grad_norms(&[(0, 12.1), (1, 13.4), (2, 18.7)], 0, Some(-0.032));
    let mut r = HealthRenderer::new();
    let body = r.format_block(&snap);
    assert!(body.contains("=== Training Health Monitor (live) ==="));
    assert!(body.contains("Step 100"));
    assert!(body.contains("Loss: 4.23"));
    assert!(body.contains("Grad norm: 12.4"));
    assert!(body.contains("Per-layer gradient norms:"));
    assert!(body.contains("L0:"));
    assert!(body.contains("L1:"));
    assert!(body.contains("L2:"));
    assert!(body.contains("18.7"));
    assert!(body.contains("⚠ elevated"), "L2 over threshold should be flagged: {}", body);
}

#[test]
fn nan_count_warning_renders() {
    let snap = snap_with_grad_norms(&[(0, 1.0)], 3, Some(-0.01));
    let mut r = HealthRenderer::new();
    let body = r.format_block(&snap);
    assert!(body.contains("⚠ 3 NaN/Inf"), "{}", body);
}

#[test]
fn loss_trend_classification() {
    let snap_decreasing = snap_with_grad_norms(&[], 0, Some(-0.032));
    let snap_increasing = snap_with_grad_norms(&[], 0, Some(0.05));
    let snap_flat = snap_with_grad_norms(&[], 0, Some(0.0));
    let snap_none = snap_with_grad_norms(&[], 0, None);
    let mut r = HealthRenderer::new();
    assert!(r.format_block(&snap_decreasing).contains("decreasing"));
    assert!(r.format_block(&snap_increasing).contains("increasing"));
    assert!(r.format_block(&snap_flat).contains("flat"));
    assert!(r.format_block(&snap_none).contains("insufficient"));
}

#[test]
fn group_by_layer_prefix_simple_stack() {
    let mut m = HashMap::new();
    m.insert("m.transformer.h.0.attn.wq".to_string(), 0.3);
    m.insert("m.transformer.h.0.attn.wk".to_string(), 0.2);
    m.insert("m.transformer.h.1.attn.wq".to_string(), 0.4);
    let groups = group_by_layer_prefix(&m);
    let labels: Vec<_> = groups.iter().map(|(l, _)| l.clone()).collect();
    assert!(labels.contains(&"L0".to_string()));
    assert!(labels.contains(&"L1".to_string()));
    let l0_entries = &groups.iter().find(|(l, _)| l == "L0").unwrap().1;
    assert_eq!(l0_entries.len(), 2);
}

#[test]
fn group_by_layer_prefix_encoder_decoder_split() {
    let mut m = HashMap::new();
    m.insert("m.encoder.blocks.0.attn.wq".to_string(), 0.1);
    m.insert("m.decoder.blocks.0.attn.wq".to_string(), 0.2);
    let groups = group_by_layer_prefix(&m);
    let labels: Vec<_> = groups.iter().map(|(l, _)| l.clone()).collect();
    assert!(labels.contains(&"Enc.L0".to_string()));
    assert!(labels.contains(&"Dec.L0".to_string()));
}

#[test]
fn group_by_layer_prefix_handles_no_numeric_segment() {
    let mut m = HashMap::new();
    m.insert("standalone_param".to_string(), 1.0);
    let groups = group_by_layer_prefix(&m);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].0, "misc");
}

#[test]
fn empty_grad_norms_skips_block() {
    let snap = HealthSnapshot::default();
    let mut r = HealthRenderer::new();
    let body = r.format_block(&snap);
    // No "Per-layer gradient norms:" header when no data.
    assert!(!body.contains("Per-layer gradient norms:"));
    // Still has the header and NaN watch line.
    assert!(body.contains("=== Training Health Monitor"));
    assert!(body.contains("NaN watch:"));
}
```

- [ ] **Step 2: Run — expect fail**

```
cargo test -p nsl-cli --test health_monitor 2>&1 | tail -15
```

- [ ] **Step 3: Implement renderer**

Create `crates/nsl-cli/src/health_monitor.rs`:

```rust
//! Renders HealthSnapshot in PDF §4.3 layout. TTY-aware: in-place ANSI
//! cursor restore on terminals; plain append otherwise.

use std::io::{IsTerminal, Write};
use std::collections::{BTreeMap, HashMap};
use nsl_runtime::health::collector::HealthSnapshot;

const BAR_WIDTH: usize = 18;
const GRAD_NORM_HEALTHY_MAX: f64 = 16.0;
const WEIGHT_DELTA_WARN_PCT: f64 = 2.0;

pub struct HealthRenderer {
    last_render_lines: usize,
    is_tty: bool,
}

impl Default for HealthRenderer { fn default() -> Self { Self::new() } }

impl HealthRenderer {
    pub fn new() -> Self {
        Self { last_render_lines: 0, is_tty: std::io::stderr().is_terminal() }
    }

    pub fn render(&mut self, snap: &HealthSnapshot) {
        let body = self.format_block(snap);
        let mut stderr = std::io::stderr().lock();
        if self.is_tty && self.last_render_lines > 0 {
            for _ in 0..self.last_render_lines {
                let _ = write!(stderr, "\x1b[1A\x1b[2K");
            }
        }
        let _ = write!(stderr, "{}", body);
        let _ = stderr.flush();
        self.last_render_lines = body.lines().count();
    }

    pub fn format_block(&self, snap: &HealthSnapshot) -> String {
        let mut out = String::new();
        out.push_str("=== Training Health Monitor (live) ===\n");

        let step_label = match snap.max_steps {
            Some(m) => format!("Step {}/{}", snap.step, m),
            None => format!("Step {}", snap.step),
        };
        let loss_str = snap.loss.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "—".into());
        let grad_str = snap.grad_norm_total.map(|v| format!("{:.1}", v)).unwrap_or_else(|| "—".into());
        out.push_str(&format!("{} | Loss: {} | Grad norm: {}\n\n", step_label, loss_str, grad_str));

        if !snap.per_layer_grad_norm.is_empty() {
            out.push_str("Per-layer gradient norms:\n");
            let mut layers: Vec<_> = snap.per_layer_grad_norm.iter().collect();
            layers.sort_by_key(|(idx, _)| **idx);
            let max_norm = snap.per_layer_grad_norm.values().cloned()
                .fold(0.0_f64, f64::max).max(1e-9);
            for (idx, norm) in &layers {
                let bar = render_bar(**norm, max_norm);
                let tag = if **norm > GRAD_NORM_HEALTHY_MAX { "⚠ elevated" } else { "(healthy)" };
                out.push_str(&format!(" L{}: {} {:>6.1} {}\n", idx, bar, norm, tag));
            }
            out.push('\n');
        }

        if !snap.per_tensor_weight_pct_delta.is_empty() {
            out.push_str("Per-layer weight norms (Δ from init):\n");
            for (label, entries) in group_by_layer_prefix(&snap.per_tensor_weight_pct_delta) {
                out.push_str(&format!(" {}: ", label));
                let parts: Vec<String> = entries.iter().map(|(leaf, pct)| {
                    let mark = if pct.abs() > WEIGHT_DELTA_WARN_PCT { " ⚠" } else { "" };
                    format!("{}: {:+.1}%{}", leaf, pct, mark)
                }).collect();
                out.push_str(&parts.join(" "));
                out.push('\n');
            }
            out.push('\n');
        }

        let nan_status = if snap.nan_inf_count_window == 0 {
            format!("clean (0 NaN/Inf detected in {} steps)", snap.steps_in_window)
        } else {
            format!("⚠ {} NaN/Inf detected in {} steps",
                snap.nan_inf_count_window, snap.steps_in_window)
        };
        out.push_str(&format!("NaN watch: {}\n", nan_status));

        match snap.loss_ema_slope {
            Some(slope) => {
                let trend = if slope < -0.001 { "decreasing" }
                            else if slope > 0.001 { "increasing ⚠" }
                            else { "flat" };
                out.push_str(&format!("Loss trend: {} (EMA slope: {:+.3}/step)\n", trend, slope));
            }
            None => out.push_str("Loss trend: (insufficient samples)\n"),
        }
        out
    }
}

fn render_bar(value: f64, max: f64) -> String {
    let frac = (value / max).clamp(0.0, 1.0);
    let filled = (frac * BAR_WIDTH as f64).round() as usize;
    "█".repeat(filled) + &"░".repeat(BAR_WIDTH.saturating_sub(filled))
}

pub fn group_by_layer_prefix(items: &HashMap<String, f64>)
    -> Vec<(String, Vec<(String, f64)>)>
{
    let mut groups: BTreeMap<(String, u32), Vec<(String, f64)>> = BTreeMap::new();
    for (path, value) in items {
        let segments: Vec<&str> = path.split('.').collect();
        let last_num_idx = segments.iter().enumerate().rev()
            .find_map(|(i, s)| s.parse::<u32>().ok().map(|_| i));
        match last_num_idx {
            Some(i) => {
                let prefix = segments[..i].join(".");
                let layer_idx: u32 = segments[i].parse().unwrap();
                let leaf = segments[i+1..].join(".");
                let label = make_layer_label(&prefix);
                groups.entry((label, layer_idx)).or_default().push((leaf, *value));
            }
            None => {
                groups.entry(("misc".to_string(), u32::MAX)).or_default()
                    .push((path.clone(), *value));
            }
        }
    }
    groups.into_iter().map(|((label, idx), v)| {
        let final_label = if idx == u32::MAX { "misc".to_string() }
            else if label == "L" { format!("L{}", idx) }
            else { format!("{}L{}", label, idx) };
        (final_label, v)
    }).collect()
}

fn make_layer_label(prefix: &str) -> String {
    if prefix.contains("encoder") { "Enc.".into() }
    else if prefix.contains("decoder") { "Dec.".into() }
    else { "L".into() }
}
```

Add `pub mod health_monitor;` to `crates/nsl-cli/src/lib.rs`.

- [ ] **Step 4: Run tests — expect 7 passed**

```
cargo test -p nsl-cli --test health_monitor 2>&1 | tail -15
cargo build -p nsl-cli 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```
git add crates/nsl-cli/src/health_monitor.rs \
        crates/nsl-cli/src/lib.rs \
        crates/nsl-cli/tests/health_monitor.rs
git commit -m "feat(cli): health_monitor renderer matches PDF §4.3 layout"
```

---

## Task 6: CLI integration — `--monitor` train-block detection

**Files:**
- Modify: `crates/nsl-cli/src/main.rs::Cli::Run` — detect train block, set `health_monitor`, render snapshot at end.

- [ ] **Step 1: Read current `--monitor` dispatch**

```
sed -n '1010,1050p' crates/nsl-cli/src/main.rs   # Run arm
```

Identify where `monitor: bool` is checked and `compile_options` is constructed.

- [ ] **Step 2: Implement train-block detection**

In the `Cli::Run` arm, before constructing `compile_opts`:

```rust
let has_train_block = if monitor {
    // Quick parse to detect a top-level train block. Reuse the existing
    // ShapeDebugInput::from_source pattern from Phase 1 Task 2 — it parses
    // and returns the typed AST without committing to a full pipeline.
    let src = std::fs::read_to_string(&file).unwrap_or_default();
    let detection = nsl_cli::shape_debug::ShapeDebugInput::from_source(
        &src, file.to_str().unwrap_or("<file>"),
    );
    match detection {
        Ok(input) => input.module.stmts.iter().any(|s|
            matches!(s, nsl_ast::stmt::Stmt::Train(_))   // adapt to actual variant
        ),
        Err(_) => false,
    }
} else { false };
```

(Adapt `Stmt::Train` to the actual AST variant — read `crates/nsl-ast/src/stmt.rs`.)

- [ ] **Step 3: Wire into `compile_opts`**

```rust
compile_opts.health_monitor = has_train_block;
// Don't set profile_kernels here when health_monitor is on — Phase 1/2 kernel
// timing and Phase 4 health monitor are mutually exclusive in the same run.
// User who wants both needs an explicit follow-up flag (out of Phase 4 scope).
if has_train_block { compile_opts.profile_kernels = false; }
```

- [ ] **Step 4: Render at end of run**

After the child process exits (or in-process run completes), look for `<file>.nsl-health.json`:

```rust
if monitor && has_train_block {
    let health_path = file.with_extension("nsl-health.json");
    match std::fs::read_to_string(&health_path) {
        Ok(s) => match serde_json::from_str::<nsl_runtime::health::collector::HealthSnapshot>(&s) {
            Ok(snap) => {
                let mut renderer = nsl_cli::health_monitor::HealthRenderer::new();
                renderer.render(&snap);
            }
            Err(e) => eprintln!("warning: health snapshot at {} failed to parse: {}",
                health_path.display(), e),
        },
        Err(_) => eprintln!("warning: no health snapshot at {} — run produced no metrics",
            health_path.display()),
    }
}
```

- [ ] **Step 5: Build + smoke**

```
cargo build -p nsl-cli 2>&1 | tail -5
cargo test -p nsl-cli --tests 2>&1 | tail -10
```

For a real smoke we need an NSL fixture with a train block. If `crates/nsl-cli/tests/fixtures/` has one, use it; otherwise:

```bash
cat > /tmp/tiny_train.nsl <<'EOF'
fn forward(x: Tensor<[1, 8, 16], bf16>, w: Tensor<[16, 16], bf16>) -> Tensor:
    return x @ w

train:
    model = forward
    step:
        let pred = forward(x, w)
        let loss = pred.sum()
EOF
cargo run -p nsl-cli -- run --monitor /tmp/tiny_train.nsl 2>&1 | head -30
```

Expected: at minimum, output mentions "Training Health Monitor" or the warning that no health JSON was produced. If the binary actually runs the train step, the snapshot block prints. If the train block requires more setup (DataLoader etc.) and won't execute, we still verified the codegen path compiled and the CLI detection works.

- [ ] **Step 6: Commit**

```
git add crates/nsl-cli/src/main.rs
git commit -m "feat(cli): --monitor auto-detects train block and renders health snapshot"
```

---

## Task 7: Final verification

- [ ] **Step 1: Workspace test sweep**

```
cargo test --workspace --tests -- --test-threads=1 2>&1 | grep -E "^test result:|FAILED"
```

Expected: all green. The known Windows parallel-file-lock on `e2e_m12_grad_basic_source_ad` is avoided by `--test-threads=1`.

- [ ] **Step 2: Counts**

```
cargo test --workspace --tests -- --test-threads=1 2>&1 | grep -E "^test result:" | awk '{sum+=$4} END {print "Total passing: "sum}'
```

Expected: ~2120+ (Phase 3 ended at 2102; Phase 4 adds ~20).

- [ ] **Step 3: Commit count check**

```
cd c:/Users/bwiem/projects/NSL/.worktrees/dev-tools-phase1
git log --oneline main..HEAD | head -10
```

Expected: 6 new Phase 4 commits on top of Phase 3.

- [ ] **Step 4: Manual acceptance**

If a working train fixture exists, run it:

```
cargo run -p nsl-cli -- run --monitor crates/nsl-cli/tests/fixtures/<train_fixture>.nsl
```

Expected: PDF §4.3-style block in output. If the fixture can't reach a complete training step in this environment, document the limitation in the task report and rely on the unit tests.

- [ ] **Step 5: Don't push or open PR**

Per user instruction (2026-04-12), held local until all milestones ship.

---

## Self-Review

**Spec coverage:**

- §4.1 Runtime FFI hooks → Task 2. ✅
- §4.2 `nsl_tensor_l2_norm` exposure → Task 3. ✅
- §4.3 Codegen instrumentation, FASE-aware splice, codegen-side step gating → Task 4 (with FASE fallback to standard-backward-only documented). ✅
- §4.4 `HealthCollector` (VecDeque ring, EMA, slope, NaN counter, weight init) → Task 1. ✅
- §4.5 CLI renderer with TTY-aware ANSI + group_by_layer_prefix → Task 5. ✅
- §4.6 CLI integration auto-detect train block → Task 6. ✅
- §4.7 New `CompileOptions` fields → Task 4 Step 2. ✅
- §6 error handling — warnings on missing/malformed JSON (Task 6 Step 4); NaN counter behavior (Task 1 test); no-layers skip (Task 5 test). ✅
- §7 testing — unit (Task 1, 2, 5), integration (Task 4, 6), regression (Task 7). ✅
- §8 non-goals respected — no TUI dep, no live polling, no per-layer shards, no cost-model interval. ✅

**Placeholder scan:**

Two intentional `todo!()` stubs in Task 4 Step 6 (`compile_and_run_one_train_step`) and as fallback patterns. Each is annotated "if intractable, skip and rely on Task 7's manual smoke" so the implementer doesn't get stuck. No drive-by "TBD"/"add validation" patterns elsewhere.

**Type consistency:**

- `HealthCollector`/`HealthSnapshot` defined in Task 1, used throughout.
- FFI symbols (`nsl_health_record_loss/grad_norm/weight_norm`, `nsl_health_flush_snapshot`, `nsl_health_set_flush_interval`) named consistently across Task 2, 3, 4.
- `nsl_tensor_l2_norm(t: i64) -> f64` consistent (Task 3 + Task 4 instrumentation).
- `CompileOptions { health_monitor, health_flush_interval }` defined in Task 4 Step 2, used in Task 6 Step 3.
- `HealthRenderer::format_block(&HealthSnapshot) -> String` consistent across Task 5 + Task 6.
- `group_by_layer_prefix(&HashMap<String, f64>)` consistent.

**Scope:**

7 tasks (6 impl + 1 verification). Each produces testable software. Final manual smoke matches PDF §4.3 expected output.
