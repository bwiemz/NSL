# NSL Dev Tools — Phase 4 Training Health Monitor Design

**Date:** 2026-04-13
**Status:** Design approved, ready for implementation plan
**Builds on:** `docs/superpowers/specs/2026-04-13-nsl-dev-tools-phase3-design.md`
**Branch (continued):** `feat/dev-tools-phase1`
**Source:** `nsl dev tools.pdf` §4.3

## 1. Purpose

Implement Tool 2's Training Health Monitor — the live in-flight per-layer gradient norms, weight-norm Δ-from-init, NaN/Inf watch, and loss EMA trend that the PDF describes for `nsl run --monitor` against a program containing a `train` block.

After Phase 4, running `nsl run --monitor train_script.nsl` shows the §4.3 layout updating in place during training, with sub-0.5% overhead.

## 2. Scope

**In:**

1. Codegen instrumentation in `compile_train_block` to emit per-step hooks for loss, per-layer gradient L2 norm, per-tensor weight L2 norm.
2. FASE-aware hook placement: when FASE per-layer fused backward is active, splice gradient-norm hooks *between* each layer's backward and its optimizer step. When standard backward is active, append after.
3. Public `nsl_tensor_l2_norm(t: i64) -> f64` runtime FFI (one-line lift of existing private `tensor_l2_norm`).
4. `HealthCollector` (`crates/nsl-runtime/src/health/`) — thread-local-ish global behind `Mutex<HealthCollector>`, accumulates loss EMA, slope, per-layer grad norms, per-tensor weight Δ from step-0 init snapshot, NaN/Inf counter.
5. `nsl_health_record_loss / record_grad_norm / record_weight_norm / flush_snapshot` FFI hooks.
6. CLI renderer in `crates/nsl-cli/src/health_monitor.rs`: TTY-aware in-place ANSI cursor restore when stderr is a terminal; plain append fallback otherwise. Layer grouping by string-path prefix with last-numeric-segment as layer index.
7. `nsl run --monitor` on a program containing `train { ... }` invokes the health monitor instead of (or in addition to) kernel timing.
8. Hardcoded sampling defaults: gradient norms every step, weight norms + flush every 100 steps. Optional `--health-interval=N` flag if it fits in 20 minutes; otherwise ship without and add later.

**Out:**

- Cost-model-driven sampling intervals (PDF describes the optimization story but uses fixed intervals in the example).
- TUI library dependency (no `crossterm`/`tui-rs`/`indicatif`).
- Per-layer sharded collectors for multi-stream-CPDT contention (single global Mutex with comment noting future fix).
- Stream-aware async memcpy for gradient samples (folds into Phase 5 with Tool 4).

## 3. Architecture

Five components, all small. No new modules in `nsl-codegen` (reuse `compile_train_block`); two new dirs in `nsl-runtime` and `nsl-cli`.

| Component | Location |
|---|---|
| Runtime FFI hooks | `crates/nsl-runtime/src/health/ffi.rs` (new) |
| Codegen instrumentation | `crates/nsl-codegen/src/stmt.rs::compile_train_block` (extend) |
| `nsl_tensor_l2_norm` exposure | `crates/nsl-runtime/src/tensor/mod.rs` (one-line lift) |
| `HealthCollector` | `crates/nsl-runtime/src/health/collector.rs` (new) |
| CLI renderer | `crates/nsl-cli/src/health_monitor.rs` (new) |

### 3.1 Reused infrastructure

- Existing `--monitor` flag on `Cli::Run` (Phase 1/2).
- Existing `enumerate_model_tensor_paths()` (`stmt.rs:4908`) for parameter path enumeration.
- Existing `step_count_var` Cranelift slot in `compile_train_block` (`stmt.rs:3246`).
- Existing `nsl_tensor_item(t: i64) -> f64` for loss scalar extraction.
- Existing private `tensor_l2_norm(&NslTensor) -> f64` (`tensor/mod.rs:3874`).
- Existing FASE per-layer fused-backward loop emission point (implementer locates during writing-plans).
- Existing `builtins::RUNTIME_FUNCTIONS` registry for codegen-emitted FFI symbols.

## 4. Component Designs

### 4.1 Runtime FFI hooks

`crates/nsl-runtime/src/health/mod.rs`:

```rust
pub mod collector;
pub mod ffi;
```

`crates/nsl-runtime/src/health/ffi.rs`:

```rust
//! C-callable hooks emitted by codegen at training-step boundaries.

use once_cell::sync::Lazy;
use std::sync::Mutex;
use super::collector::HealthCollector;

// NOTE: single global Mutex is fine today (NSL backward is single-stream
// sequential). When CPDT lands and overlaps backward across layers, the fix is
// per-layer sharded collectors merged at snapshot time. Don't refactor until
// profiling shows contention.
static COLLECTOR: Lazy<Mutex<HealthCollector>> = Lazy::new(|| Mutex::new(HealthCollector::new()));

#[no_mangle]
pub extern "C" fn nsl_health_record_loss(value: f64, step: u64) {
    COLLECTOR.lock().unwrap().record_loss(step, value);
}

/// path bytes are owned by the caller (codegen-emitted static .rodata).
/// Collector copies into its own String.
///
/// # Safety
/// Caller must guarantee (path_ptr, path_len) refers to valid UTF-8.
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

/// `is_init` is true on the first call per path (codegen emits at step 0);
/// subsequent calls record current weight norm and the collector computes
/// percent-delta.
///
/// # Safety
/// Same.
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

/// Flush the collector to JSON at `path_ptr` (or stdout when null).
/// Codegen emits this call inside an `if step % 100 == 0` Cranelift branch,
/// so it's never called on non-flush steps. The collector also has a
/// belt-and-suspenders `should_flush` gate, harmless when codegen already gates.
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
```

Add `pub mod health;` to `crates/nsl-runtime/src/lib.rs`.

### 4.2 `nsl_tensor_l2_norm` exposure

`crates/nsl-runtime/src/tensor/mod.rs` — alongside the existing private `tensor_l2_norm`:

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_l2_norm(t: i64) -> f64 {
    let tensor = NslTensor::from_ptr(t);
    tensor_l2_norm(&tensor)
}
```

(i64 handle in, no `unsafe` on extern, no null check — matches the existing tensor-FFI pattern. Implementer mirrors `nsl_tensor_item` exactly.)

Register in `crates/nsl-codegen/src/builtins.rs::RUNTIME_FUNCTIONS`: name `"nsl_tensor_l2_norm"`, params `[I64]`, returns `Some(F64)`.

Register the four health hooks too:
- `nsl_health_record_loss`: `[F64, I64]` → none.
- `nsl_health_record_grad_norm`: `[I64, I64, I32, F64]` → none.
- `nsl_health_record_weight_norm`: `[I64, I64, F64, I8]` → none.
- `nsl_health_flush_snapshot`: `[I64, I64]` → `Some(I32)`.

### 4.3 Codegen instrumentation

In `crates/nsl-codegen/src/stmt.rs::compile_train_block`, gated on a new `compile_options.health_monitor: bool`. Inside the per-step body:

**Loss:** after `loss = ...` and before backward.

```rust
let loss_scalar = builder.ins().call(func_ref("nsl_tensor_item"), &[loss_val]);
let step_val = builder.ins().load(I64, MemFlags::trusted(), step_count_var, 0);
builder.ins().call(func_ref("nsl_health_record_loss"), &[loss_scalar, step_val]);
```

**Per-layer gradient norms — FASE-aware placement:**

- **If FASE is active** (the per-layer fused backward-then-optimizer loop): inside that loop, between the gradient-compute and optimizer-step for each layer, emit:

  ```rust
  let path_ptr = self.intern_string_constant(&path);
  let path_len = builder.ins().iconst(I64, path.len() as i64);
  let layer_idx = builder.ins().iconst(I32, parse_layer_idx(&path) as i64);
  let norm = builder.ins().call(func_ref("nsl_tensor_l2_norm"), &[grad_ptr_val]);
  builder.ins().call(
      func_ref("nsl_health_record_grad_norm"),
      &[path_ptr, path_len, layer_idx, norm],
  );
  ```

  The exact splice point is inside the FASE-loop body in `stmt.rs` — implementer locates the `for layer in layers { backward; optimizer; free; }` shape during writing-plans. Each iteration emits one record-grad-norm call before the optimizer-step + free.

- **If FASE is not active** (standard backward, all gradients persist after backward): emit the same loop appended after backward returns the gradient list. Iterate `enumerated_param_grads` from `enumerate_model_tensor_paths()`.

**Per-layer weight norms — every 100 steps + step-0 init:**

Two Cranelift branches off `step_val`:

```rust
// is_init: step == 0 → record with is_init=true
let zero = builder.ins().iconst(I64, 0);
let step_is_zero = builder.ins().icmp(IntCC::Equal, step_val, zero);
builder.ins().brif(step_is_zero, init_block, &[], periodic_check_block, &[]);

// init_block: emit per-param record_weight_norm with is_init = 1
//   for each (path, param_ptr_val) in enumerated_params:
//     norm = nsl_tensor_l2_norm(param_ptr_val)
//     nsl_health_record_weight_norm(path_ptr, path_len, norm, /*is_init=*/ 1)

// periodic_check_block: step % 100 == 0 → record current with is_init = 0
let mod_val = builder.ins().urem_imm(step_val, 100);
let due = builder.ins().icmp(IntCC::Equal, mod_val, zero);
builder.ins().brif(due, periodic_block, &[], after_weight_block, &[]);
//   periodic_block emits the same loop with is_init = 0
```

**Snapshot flush — codegen-side step-gated:** wrap the FFI call in `if step % 100 == 0` so the FFI is never called on the 99 intervening steps (zero mutex cost, zero serde cost):

```rust
let due_val = builder.ins().urem_imm(step_val, 100);
let flush_due = builder.ins().icmp_imm(IntCC::Equal, due_val, 0);
builder.ins().brif(flush_due, do_flush_block, &[], skip_flush_block, &[]);
// do_flush_block:
let snap_path_ptr = self.intern_string_constant(&snap_path_str);
let snap_path_len = builder.ins().iconst(I64, snap_path_str.len() as i64);
builder.ins().call(
    func_ref("nsl_health_flush_snapshot"),
    &[snap_path_ptr, snap_path_len],
);
```

**Helpers:**

- `intern_string_constant(s)` — codegen has this for PTX kernel names; reuse or add. Returns a `Value` pointing at static UTF-8 bytes.
- `parse_layer_idx(path) -> u32` — runs at codegen time; finds the last numeric segment in the dotted path. Returns `u32::MAX` if none (renderer falls back gracefully).
- `enumerated_param_grads`, `enumerated_params` — derived from `enumerate_model_tensor_paths()`. Each entry pairs a path string with the IR `Value` holding the param/grad tensor pointer in scope at that Cranelift block.

**FASE detection:** check `compile_options.fase_enabled` (or whichever flag indicates the FASE per-layer loop is active for this train block — implementer confirms during writing-plans). If unclear from the existing options, look in the train-block lowering for a fork between FASE and standard backward.

### 4.4 `HealthCollector`

`crates/nsl-runtime/src/health/collector.rs`:

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
    loss_history: VecDeque<f64>,                  // O(1) push_back / pop_front
    loss_ema: Option<f64>,
    grad_norm_per_layer: HashMap<u32, f64>,
    weight_init: HashMap<String, f64>,
    weight_current: HashMap<String, f64>,
    nan_inf_count: u64,
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

### 4.5 CLI renderer

`crates/nsl-cli/src/health_monitor.rs`:

```rust
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

    fn format_block(&self, snap: &HealthSnapshot) -> String {
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
            let max_norm = snap.per_layer_grad_norm.values().cloned().fold(0.0_f64, f64::max).max(1e-9);
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
            format!("⚠ {} NaN/Inf detected in {} steps", snap.nan_inf_count_window, snap.steps_in_window)
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

### 4.6 CLI integration

In `crates/nsl-cli/src/main.rs::Cli::Run`, when `monitor` is set:

1. Detect whether the source contains a `train { ... }` block (parse-time check; reuse the existing parser output).
2. If yes → set `compile_options.health_monitor = true`, build, run. After the child process exits, locate `<file>.nsl-health.json`, read the latest snapshot, render it once at the end. (Optional follow-up: live polling — out of Phase 4 scope.)
3. If no → existing kernel-timing path (Phase 1/2 unchanged).
4. If both kernel timing and health monitor are wanted in the same run, the user can pass an additional explicit flag — out of Phase 4 scope.

Optional `--health-interval=N` flag on `Cli::Run`: if added, codegen reads it from `compile_options` and emits `nsl_health_set_flush_interval(N)` (new FFI, one-line wrapper around `HealthCollector::set_flush_interval`) at train-block entry. **Add only if the whole flag fits in 20 minutes**; otherwise ship the hardcoded default.

### 4.7 New `CompileOptions` fields

In `crates/nsl-codegen/src/lib.rs::CompileOptions`:

```rust
pub health_monitor: bool,             // default false
pub health_flush_interval: Option<u64>,  // default None → uses runtime FLUSH_INTERVAL_DEFAULT
```

Update every `CompileOptions { ... }` literal in the workspace (search via `grep -rn "CompileOptions {"`).

## 5. Data flow

```
nsl run --monitor train.nsl
   │
   ▼
parse + analyze
   │
   ├── if train block detected:
   │     opts.health_monitor = true
   │
   ▼
codegen::compile_train_block emits per step:
   ├── nsl_health_record_loss(loss_scalar, step)
   ├── per layer (FASE-aware splice or post-backward append):
   │     nsl_health_record_grad_norm(path, layer_idx, l2_norm)
   ├── if step == 0 OR step % 100 == 0:
   │     for each param: nsl_health_record_weight_norm(path, l2_norm, is_init)
   └── if step % 100 == 0:
         nsl_health_flush_snapshot("<file>.nsl-health.json")
   │
   ▼
runtime executes training loop; HealthCollector accumulates;
flushes JSON every 100 steps.
   │
   ▼
child process exits.
   │
   ▼
CLI reads <file>.nsl-health.json, renders once via HealthRenderer.
   │
   ▼
text output to stderr (TTY: in-place; non-TTY: appended).
```

**Live in-flight rendering** is out of Phase 4 scope (would require a TUI poller forking a thread to tail the JSON file). The PDF example shows a "live" view; Phase 4 ships the data pipeline + final render. Polling fold-in is a Phase 4.5 follow-up if anyone needs it.

## 6. Error handling

- Train block with no parameters: `enumerate_model_tensor_paths` returns empty; codegen emits no per-layer hooks; renderer prints "(no layers detected)".
- Loss tensor missing: `compile_train_block` already searches for the symbol "loss" and errors when absent. Existing error path; no Phase 4 change.
- `nsl_tensor_l2_norm` on a freed gradient pointer: undefined behavior, but FASE-aware splice ensures the call happens before the optimizer-step that frees. Standard backward keeps gradients alive.
- JSON file missing at render time (process crashed before first flush): CLI prints "(no health snapshot — run produced no metrics)".
- NaN/Inf loss: collector counts and skips EMA update. NaN watch line shows the count.
- Non-finite weight norm: collector keeps the value; percent-delta against finite init would produce NaN — renderer formats as "—".

## 7. Testing

- **Unit (collector):**
  - `record_loss` with NaN increments `nan_inf_count` and doesn't update EMA.
  - `record_loss` with 100+ values keeps `loss_history` capped at LOSS_WINDOW.
  - `snapshot` with monotonically decreasing loss returns negative `loss_ema_slope`.
  - `record_weight_norm(path, n, is_init=true)` followed by `record_weight_norm(path, 1.05*n, false)` produces +5% in `per_tensor_weight_pct_delta`.
  - `should_flush` returns true at step 0, false at steps 1..99, true at step 100.
- **Unit (renderer):**
  - `format_block` on a synthetic `HealthSnapshot` with 4 layers contains "L0", "L1", "L2", "L3", "Per-layer gradient norms:".
  - `format_block` with `nan_inf_count_window > 0` shows "⚠".
  - `group_by_layer_prefix` on `["m.transformer.h.0.attn.wq", "m.transformer.h.1.attn.wq"]` produces 2 groups labeled "L0" and "L1".
  - `group_by_layer_prefix` on encoder/decoder paths produces "Enc.L0" and "Dec.L0" as separate groups.
  - `format_block` on empty `per_layer_grad_norm` skips the gradient block without panicking.
- **Unit (FFI):**
  - `nsl_health_record_loss(NaN, 1)` increments NaN counter.
  - `nsl_health_flush_snapshot(null, 0)` writes JSON to stdout, returns 0.
- **Integration (codegen):** `--health-monitor` enabled compile of a tiny train-block fixture produces a binary that, when run, writes `<file>.nsl-health.json`. (CUDA-free where possible; otherwise gated `#[cfg(feature = "cuda")]`.)
- **Regression:** all Phase 1/2/2.5/3 tests pass.

## 8. Non-goals

- Cost-model-driven sampling intervals (PDF describes the optimization, but the example output uses fixed intervals).
- Live in-flight rendering via background-thread polling (Phase 4.5 if needed).
- TUI library dependency.
- Per-layer sharded collectors (Phase 5+ when CPDT lands and contention shows up in profiles).
- Stream-aware async memcpy for grad-norm computations (folds into Phase 5 with Tool 4).
- `max_steps` display value (PDF shows `Step 100/10000` but max_steps requires plumbing the train-block's max-steps from source — left at `None` until needed).

## 9. File inventory

**New:**

- `crates/nsl-runtime/src/health/mod.rs`
- `crates/nsl-runtime/src/health/collector.rs`
- `crates/nsl-runtime/src/health/ffi.rs`
- `crates/nsl-cli/src/health_monitor.rs`
- Test files: `crates/nsl-runtime/tests/health_collector.rs`, `crates/nsl-cli/tests/health_monitor.rs`, `crates/nsl-codegen/tests/health_codegen.rs`.

**Modified:**

- `crates/nsl-runtime/src/lib.rs` — add `pub mod health;`.
- `crates/nsl-runtime/src/tensor/mod.rs` — add `pub extern "C" fn nsl_tensor_l2_norm(t: i64) -> f64`.
- `crates/nsl-codegen/src/builtins.rs` — register the five new FFI symbols.
- `crates/nsl-codegen/src/stmt.rs::compile_train_block` — emit hooks per §4.3.
- `crates/nsl-codegen/src/lib.rs::CompileOptions` — `health_monitor: bool` + optional `health_flush_interval: Option<u64>`.
- `crates/nsl-cli/src/main.rs::Cli::Run` — detect train block + set `health_monitor`; render snapshot at end.
- `crates/nsl-cli/src/lib.rs` — `pub mod health_monitor;`.
- All `CompileOptions { ... }` literal sites — add the new fields.

## 10. Follow-up phases

- **Phase 4.5** (optional): live in-flight rendering via background-thread JSON poller; `--health-interval` flag if not shipped here; cost-model-driven sampling intervals.
- **Phase 5**: Tool 4 Tensor Inspector (`@inspect` decorator + async dump collector). Stream-aware profiler folds in here as a prerequisite.
