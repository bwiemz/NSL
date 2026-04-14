# NSL Dev Tools — Phase 5 Tensor Inspector Design

**Date:** 2026-04-13
**Status:** Design approved, ready for implementation plan
**Builds on:** `docs/superpowers/specs/2026-04-13-nsl-dev-tools-phase4-design.md`
**Branch (continued):** `feat/dev-tools-phase1`
**Source:** `nsl dev tools.pdf` Tool 4

## 1. Purpose

Implement Tool 4: a compile-time `@inspect(tensor, every=N | condition="...")` decorator that emits async D2H stat collection (and optionally full tensor dump) at any user-marked point in NSL source. Triggered by `nsl run --inspect`. Stats run on a dedicated CUDA stream so the compute path is uninterrupted; full dumps fire only when an explicit predicate evaluates true.

After Phase 5, all four PDF Tools ship.

## 2. Scope

**In:**

1. `@inspect(tensor, every=N)` and `@inspect(tensor, condition="...")` decorators. Combinable: `@inspect(t, every=N, condition="...")` samples stats every N steps AND dumps raw bytes whenever the predicate fires.
2. Tiny embedded predicate language: identifiers `step`, `loss`, `loss_ema`, `loss_ema_slope`, `grad_norm_total`, `nan_inf_count_window`; binary comparators `> < >= <= == !=`; boolean ops `and or not`; parens; integer/float literals. Compiled to Cranelift IR at codegen time.
3. On-device reduction kernel `nsl_tensor_stats(t, out_buf)` computing six f64 values (mean, std, min, max, nan_count, inf_count) in one pass.
4. Dedicated `inspect_stream` (lazy-init `CUstream` in TLS) carrying all inspect-side memcpy. `cuEventRecord`/`cuStreamWaitEvent` synchronize against the compute stream that produced the tensor.
5. Memory planner extension: `pinned_until_inspect_sync` annotation extends inspect-ed tensors' death points past the memcpy completion. Reuses `binary_retention.rs` extension pattern.
6. Runtime FFI: `nsl_tensor_stats`, `nsl_inspect_record_stats`, `nsl_inspect_dump_full`, `nsl_inspect_set_dir`, `nsl_health_get_loss_ema`, `nsl_health_get_loss_ema_slope`, `nsl_health_get_grad_norm_total`, `nsl_health_get_nan_inf_count_window`.
7. Binary file format `NSLI` (magic + u32 version + u64 header_len + JSON + optional raw bytes), output to `<output_dir or .nsl-inspect>/step_<N>_<name>.{stats|tensor}.bin`.
8. CLI `--inspect` flag on `nsl run`. Sets `compile_options.inspect_enabled = true`. Prints summary at end.

**Out:**

- Live in-flight visualization of stats (file output only).
- Multi-stream profiling at large (only the inspect stream is allocated).
- Predicate function calls (`abs(loss) > X` not supported).
- Dynamic decoration via `@inspect_dynamic` API.
- Tensor diff / replay tooling.

## 3. Architecture

Seven components.

| Component | Location |
|---|---|
| Decorator semantic check | `crates/nsl-semantic/src/checker/stmt.rs` (extend) |
| Predicate compiler | `crates/nsl-codegen/src/inspect/predicate.rs` (new) |
| Codegen handler | `crates/nsl-codegen/src/stmt.rs` `compile_train_block` / `let` lowering (extend) |
| Memory planner extension | `crates/nsl-codegen/src/wrga_memory.rs` or wherever `MemoryPlan` lifetimes live |
| Runtime FFI | `crates/nsl-runtime/src/inspect/` (new module) |
| Binary `NSLI` format | `crates/nsl-runtime/src/inspect/format.rs` (new) |
| CLI integration | `crates/nsl-cli/src/main.rs` `Cli::Run` |

### 3.1 Reused infrastructure

- Decorator parsing/AST (`StmtKind::Decorated { decorators, stmt }`) — already in place; `@no_grad`, `@fuse`, etc. wired similarly.
- `step_count_var` Cranelift slot in train-block scope.
- `loss_val` Cranelift Value in train-block scope.
- Phase 4 `HealthCollector` for predicate getters (read-only via new FFI).
- `binary_retention.rs` lifetime-extension pattern.
- `.nslm` checkpoint format precedent for the `NSLI` binary shape.
- `func_ref` codegen helper for FFI symbol references.
- `intern_string` for static UTF-8 in compiled code.

## 4. Component Designs

### 4.1 Decorator semantic check

In `crates/nsl-semantic/src/checker/stmt.rs`, extend the decorator validation block. For decorator name `inspect`:

1. Args must contain at least one positional `Expr` (the tensor).
2. Tensor expr must resolve to a `Tensor<...>` type. Type-error otherwise: `@inspect requires a Tensor argument; got <type>`.
3. At least one of `every=<int>` / `condition=<string>` keyword required. Both allowed.
4. `every=N` must be a positive integer literal (compile-time constant). Reject `every=step+1` etc.
5. `condition="..."` must be a string literal. The string is parsed lazily in the predicate compiler (§4.2); semantic just sniffs that it's a string.

Errors reported with span. Returns no AST mutation — codegen reads decorator args directly.

### 4.2 Predicate compiler

`crates/nsl-codegen/src/inspect/predicate.rs`:

```rust
//! Tiny predicate compiler. Accepts a string and returns Cranelift IR
//! that produces an i8 boolean.

#[derive(Debug, Clone)]
pub enum PredicateExpr {
    IntLit(i64),
    FloatLit(f64),
    Ident(String),  // "step" | "loss" | "loss_ema" | "loss_ema_slope" |
                    // "grad_norm_total" | "nan_inf_count_window"
    Cmp(Box<PredicateExpr>, CmpOp, Box<PredicateExpr>),
    And(Box<PredicateExpr>, Box<PredicateExpr>),
    Or(Box<PredicateExpr>, Box<PredicateExpr>),
    Not(Box<PredicateExpr>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CmpOp { Gt, Lt, Ge, Le, Eq, Ne }

pub fn parse_predicate(src: &str) -> Result<PredicateExpr, String>;

/// Lower a parsed predicate to Cranelift IR producing an I8 (0 or 1).
/// `ctx` provides access to the compute Cranelift Values for `step` (I64)
/// and `loss` (F64), and registered FFI refs for the getters.
pub fn lower_predicate(
    pred: &PredicateExpr,
    builder: &mut FunctionBuilder,
    ctx: &PredicateLowerCtx,
) -> Value;

pub struct PredicateLowerCtx<'a> {
    pub step_val: Value,             // I64 in scope
    pub loss_val: Value,              // F64 in scope
    pub get_loss_ema_ref: FuncRef,    // -> F64
    pub get_loss_ema_slope_ref: FuncRef,  // -> F64
    pub get_grad_norm_total_ref: FuncRef,  // -> F64
    pub get_nan_inf_count_window_ref: FuncRef,  // -> I64
    // ...references resolved via the existing func_ref helper
}
```

Recursive-descent parser; ~150 LOC including AST + parser + lowerer + tests. Operator precedence: `not` > `and` > `or` (lowest); comparators non-associative.

Identifiers map to: direct loads (step/loss) or FFI getter calls (Phase 4 collector fields). Type promotion: integer literal compared to float ident is widened to f64. Mismatched comparisons (`step > 1.5`) widen via f64 cast.

### 4.3 Codegen handler

In `compile_train_block` (and any `let`-lowering site that can see `Stmt::Decorated` with `@inspect`), gated on `compile_options.inspect_enabled`:

For each `@inspect(<tensor_ident>, every=N?, condition="..."?)`:

1. Resolve the tensor identifier to its current Cranelift `Value` (the i64 tensor handle from the most recent `let` binding of that name).
2. Mark the tensor for memory pinning: pass its NodeId to the memory planner's `pinned_until_inspect_sync` set (§4.4).
3. Intern the tensor name as a static UTF-8 buffer.
4. Emit:

   **Stats branch (if `every=N` was given):**
   ```
   if step_val % N == 0:
       // Compute stats on-device
       launch nsl_tensor_stats(tensor_handle, stats_buf_ptr) on compute stream
       cuEventRecord(stats_event, compute_stream)
       cuStreamWaitEvent(inspect_stream, stats_event)
       // Async 48-byte D2H + JSON write
       nsl_inspect_record_stats(stats_buf_ptr, step_val, name_ptr, name_len)
   ```

   `stats_buf_ptr` is a small per-inspect-site static buffer (48 bytes, `[f64; 6]`) declared in `.bss`/data segment via `intern_data_constant` or equivalent. One per inspect site. Pinned host memory if the runtime supports it; otherwise plain `malloc`'d.

   **Dump branch (if `condition="..."` was given):**
   ```
   pred_val = lower_predicate(parse_predicate(cond_str), builder, &ctx)
   if pred_val != 0:
       cuEventRecord(dump_event, compute_stream)
       cuStreamWaitEvent(inspect_stream, dump_event)
       // Full async D2H + raw-bytes write
       nsl_inspect_dump_full(tensor_handle, step_val, name_ptr, name_len)
   ```

When both `every=N` and `condition=` are present, both branches emit independently.

When `compile_options.inspect_enabled == false`, the entire decorator is a no-op (codegen skips emission, identical IR to no-decorator case).

### 4.4 Memory planner extension

In `crates/nsl-codegen/src/wrga_memory.rs` (or `binary_retention.rs` — implementer locates the actual lifetime-extension site):

```rust
pub struct MemoryPlanInputs {
    // ...existing...
    pub pinned_until_inspect_sync: HashSet<VarId>,  // NEW
}
```

The planner's death-point computation, when finalizing a slot's `death`, checks if the VarId is in `pinned_until_inspect_sync`. If yes, extends `death` to the latest program point + 1 (or some "end of step" sentinel) so the slot's allocation isn't reused before the inspect memcpy can complete.

Codegen populates this set from the `@inspect` handler in §4.3. Reuse the existing `binary_retention.rs` annotation pattern verbatim — implementer reads it and mirrors.

When `inspect_enabled == false`, the set stays empty and the planner runs unchanged.

### 4.5 Runtime FFI

`crates/nsl-runtime/src/inspect/mod.rs`:

```rust
pub mod ffi;
pub mod format;
pub mod stream;
pub mod stats_kernel;
```

`crates/nsl-runtime/src/inspect/stream.rs`:

```rust
//! Lazy-init dedicated CUstream for inspect copies.
#![cfg(feature = "cuda")]

use cudarc::driver::sys;
use std::cell::RefCell;

thread_local! {
    static INSPECT_STREAM: RefCell<Option<sys::CUstream>> = RefCell::new(None);
}

pub fn current_inspect_stream() -> sys::CUstream {
    INSPECT_STREAM.with(|s| {
        let mut g = s.borrow_mut();
        if g.is_none() {
            let mut stream: sys::CUstream = std::ptr::null_mut();
            unsafe {
                let res = sys::lib().cuStreamCreate(&mut stream, 0);
                if res.0 != 0 { panic!("cuStreamCreate failed: {:?}", res); }
            }
            *g = Some(stream);
        }
        g.unwrap()
    })
}
```

For non-cuda builds, `current_inspect_stream` returns a unit and copies fall back to host-only stat writes.

`crates/nsl-runtime/src/inspect/stats_kernel.rs`:

```rust
//! On-device reduction kernel: one pass over the tensor, writes 6 f64 stats.
//! mean, std, min, max, nan_count, inf_count.

#[no_mangle]
pub extern "C" fn nsl_tensor_stats(t: i64, out_buf: *mut f64) -> i32 {
    // Launch a single reduction kernel on the compute stream that computes
    // all six values in one pass and writes them into the 48-byte out_buf
    // on the device. Caller cuStreamWaitEvent's the inspect stream onto the
    // completion event before the async D2H of out_buf.
    //
    // Implementation note: NSL already has nsl_tensor_mean, nsl_tensor_min,
    // nsl_tensor_max in tensor/reduction.rs. NaN/Inf counts need to be added
    // to that reduction module. The reduction kernel is built using the
    // existing PTX-codegen path in nsl-codegen/src/kernel.rs.
    unimplemented!("on-device stats reduction kernel — implementer wires using existing reduction infra")
}
```

Implementer choice: either write a hand-tuned PTX that fuses all six reductions in one pass (best perf), or call the existing `nsl_tensor_mean/min/max` in sequence + one new `nsl_tensor_count_nan_inf` kernel and stash results in `out_buf` (simpler, ~2x slower but still <10µs for typical activation tensors). Per spec ship-fast policy, the latter is acceptable for the first PR.

`crates/nsl-runtime/src/inspect/ffi.rs`:

```rust
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::path::PathBuf;
use super::format;

static INSPECT_DIR: Lazy<Mutex<PathBuf>> = Lazy::new(|| Mutex::new(PathBuf::from(".nsl-inspect")));

#[no_mangle]
pub unsafe extern "C" fn nsl_inspect_set_dir(path_ptr: *const u8, path_len: usize) {
    if path_ptr.is_null() { return; }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    if let Ok(s) = std::str::from_utf8(bytes) {
        *INSPECT_DIR.lock().unwrap() = PathBuf::from(s);
    }
}

/// Stats buffer is a host-resident [f64; 6] populated by the on-device
/// reduction kernel + async D2H. Caller has already cuStreamWaitEvent'd
/// the inspect stream on the kernel completion event.
///
/// # Safety
/// Caller guarantees stats_buf_ptr points to a 48-byte readable buffer
/// and (name_ptr, name_len) is valid UTF-8 the caller owns.
#[no_mangle]
pub unsafe extern "C" fn nsl_inspect_record_stats(
    stats_buf_ptr: *const f64,
    step: u64,
    name_ptr: *const u8, name_len: usize,
) -> i32 {
    if stats_buf_ptr.is_null() || name_ptr.is_null() { return 1; }
    let stats = std::slice::from_raw_parts(stats_buf_ptr, 6);
    let name = match std::str::from_utf8(std::slice::from_raw_parts(name_ptr, name_len)) {
        Ok(s) => s, Err(_) => return 2,
    };
    let dir = INSPECT_DIR.lock().unwrap().clone();
    if let Err(_) = std::fs::create_dir_all(&dir) { return 3; }
    let path = dir.join(format!("step_{}_{}.stats.bin", step, name));
    match format::write_stats(&path, step, name, stats) {
        Ok(_) => 0, Err(_) => 4,
    }
}

/// Full-tensor dump: reads tensor handle, async D2H of full buffer via
/// the inspect stream, then writes raw bytes + JSON header.
///
/// # Safety
/// Same as record_stats; tensor handle must be valid.
#[no_mangle]
pub unsafe extern "C" fn nsl_inspect_dump_full(
    tensor_handle: i64,
    step: u64,
    name_ptr: *const u8, name_len: usize,
) -> i32 {
    // 1. Resolve tensor handle (NslTensor::from_ptr).
    // 2. Allocate host buffer matching tensor size.
    // 3. cuMemcpyAsync D->H on inspect stream.
    // 4. cuStreamSynchronize on inspect stream (we need the data on host
    //    before write).
    // 5. format::write_full(path, step, name, dtype, shape, host_buf).
    // 6. Free host buffer.
    // ...implementer wires using existing nsl_tensor_to_device pattern,
    // adapted for an explicit stream and an async memcpy.
    unimplemented!("full dump — implementer wires async D2H via inspect stream")
}

#[no_mangle]
pub extern "C" fn nsl_health_get_loss_ema() -> f64 {
    crate::health::ffi::collector_lock().snapshot().loss_ema.unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn nsl_health_get_loss_ema_slope() -> f64 {
    crate::health::ffi::collector_lock().snapshot().loss_ema_slope.unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn nsl_health_get_grad_norm_total() -> f64 {
    crate::health::ffi::collector_lock().snapshot().grad_norm_total.unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn nsl_health_get_nan_inf_count_window() -> i64 {
    crate::health::ffi::collector_lock().snapshot().nan_inf_count_window as i64
}
```

The four `nsl_health_get_*` getters need a small accessor in `health/ffi.rs` — `pub(crate) fn collector_lock() -> MutexGuard<'static, HealthCollector>` — so inspect's getters can borrow without going through the snapshot path twice. (Optimization optional; calling `snapshot()` works too.)

### 4.6 Binary `NSLI` format

`crates/nsl-runtime/src/inspect/format.rs`:

```rust
//! NSLI binary log format (mirror of NSLM checkpoint).
//!
//! Layout:
//!   [0..4]   magic = b"NSLI"
//!   [4..8]   version: u32 (1)
//!   [8..16]  header_len: u64
//!   [16..16+header_len]  JSON header (UTF-8)
//!   [aligned to 64]  raw tensor bytes (only for full dumps; empty for stats-only)

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

const MAGIC: &[u8; 4] = b"NSLI";
const VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsHeader {
    pub step: u64,
    pub tensor_name: String,
    pub kind: String,  // "stats" | "full"
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub nan_count: u64,
    pub inf_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullHeader {
    pub step: u64,
    pub tensor_name: String,
    pub kind: String,  // "full"
    pub dtype: String,
    pub shape: Vec<i64>,
    pub stats: StatsHeader,
}

pub fn write_stats(path: &Path, step: u64, name: &str, stats: &[f64]) -> std::io::Result<()> {
    let header = StatsHeader {
        step, tensor_name: name.into(), kind: "stats".into(),
        mean: stats[0], std: stats[1], min: stats[2], max: stats[3],
        nan_count: stats[4] as u64, inf_count: stats[5] as u64,
    };
    let json = serde_json::to_vec(&header)?;
    let mut f = std::fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&(json.len() as u64).to_le_bytes())?;
    f.write_all(&json)?;
    Ok(())
}

pub fn write_full(path: &Path, header: &FullHeader, data: &[u8]) -> std::io::Result<()> {
    let json = serde_json::to_vec(header)?;
    let header_len = json.len() as u64;
    let aligned_offset = ((16 + header_len + 63) / 64) * 64;
    let pad = aligned_offset - (16 + header_len);
    let mut f = std::fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(&json)?;
    f.write_all(&vec![0u8; pad as usize])?;
    f.write_all(data)?;
    Ok(())
}
```

### 4.7 CLI integration

In `crates/nsl-cli/src/main.rs::Cli::Run`:

```rust
#[arg(long)]
inspect: bool,
```

In the dispatch arm, when `inspect`:
- Set `compile_options.inspect_enabled = true`.
- Pass through to codegen.
- After child process exits, scan the inspect output dir (default `.nsl-inspect/`) for written files; print a one-line summary: `Wrote N stats records, M full dumps to <dir>/`.

`--inspect` is independent of `--monitor`. Both can be set together; both pipelines run.

Add `inspect_enabled: bool` to `CompileOptions` (default `false`). Update the existing `CompileOptions { ... }` literal sites in `nsl-cli/src/main.rs` (Build path: `inspect_enabled: false`; Run path: `inspect_enabled: inspect`).

## 5. Data flow

```
nsl run --inspect train.nsl
   │
   ▼
parse + analyze (decorator AST already built; semantic checks @inspect args)
   │
   ▼
codegen sees @inspect(t, every=N, condition="..."):
   ├── memory planner: pinned_until_inspect_sync.insert(t.var_id)
   ├── stats branch (every=N):
   │     if step % N == 0:
   │       launch nsl_tensor_stats(t, &stats_buf) on compute stream
   │       cuEventRecord(ev1, compute)
   │       cuStreamWaitEvent(inspect, ev1)
   │       call nsl_inspect_record_stats(&stats_buf, step, name)
   │           → async memcpy 48 B D->H on inspect stream
   │           → write step_<N>_<name>.stats.bin
   └── dump branch (condition="..."):
         pred = lower_predicate(parse_predicate(cond_str))
         if pred:
           cuEventRecord(ev2, compute)
           cuStreamWaitEvent(inspect, ev2)
           call nsl_inspect_dump_full(t_handle, step, name)
               → async D->H of full tensor on inspect stream
               → cuStreamSynchronize(inspect)
               → write step_<N>_<name>.tensor.bin
```

## 6. Error handling

- Bad `@inspect` args (no tensor, both keywords absent, non-int `every`, non-string `condition`): semantic error with span.
- Predicate parse error: codegen error citing the offending substring.
- `cuStreamCreate` failure: panic at first inspect-touch (matches existing CUDA init failure mode).
- `cuMemcpyAsync` / `cuStreamSynchronize` failure: warn and skip the dump; don't crash training.
- Output-dir create failure: warn and skip writes; subsequent inspect calls still fire (idempotent skip).
- Tensor handle invalid (memory freed despite pinning): undefined behavior — but the planner extension is the contract that prevents it.

## 7. Testing

- **Unit (predicate compiler):**
  - `parse_predicate("step > 500")` produces `Cmp(Ident("step"), Gt, IntLit(500))`.
  - `parse_predicate("step > 500 and loss > 5.0")` produces `And(Cmp(step, Gt, 500), Cmp(loss, Gt, 5.0))`.
  - `parse_predicate("not (step > 500 or loss > 5.0)")` parses with correct precedence.
  - Bad input `"step >"` returns `Err(...)` with non-empty message.
  - Reserved-only identifiers — `parse_predicate("foo > 1")` errors with "unknown identifier 'foo'".
- **Unit (NSLI format):**
  - `write_stats` then read-back via header parse round-trips fields.
  - `write_full` data section is 64-byte aligned.
- **Unit (semantic):**
  - `@inspect(h)` with no every/condition errors with "must specify every= or condition=".
  - `@inspect(42, every=10)` errors with "first arg must be a tensor".
- **Unit (CLI):**
  - `--inspect` flag presence sets `compile_options.inspect_enabled = true`; absence leaves it false.
- **Integration (gated `feature = "test-helpers"`):**
  - Compile a fixture with `@inspect(h, every=1)` and `inspect_enabled = true`. Verify codegen emits the stats branch.
  - Compile a fixture with `@inspect(h, condition="step > 0")` and verify the dump branch is emitted.
  - Compile with `inspect_enabled = false`: verify no inspect FFI calls in the IR.
- **Regression:** all Phase 1/2/2.5/3/4 tests pass.

## 8. Non-goals

- Live in-flight visualization of stats (file output only; analysis tools come later).
- Multi-stream profiling beyond the inspect stream.
- Function calls in predicates (`abs(loss) > X`).
- Dynamic decoration via runtime API.
- Tensor diff / replay tooling.
- Per-tensor differential file format (every dump is a fresh file).
- Stats kernel SIMD optimization beyond a clean one-pass reduction.

## 9. File inventory

**New:**

- `crates/nsl-codegen/src/inspect/mod.rs`
- `crates/nsl-codegen/src/inspect/predicate.rs`
- `crates/nsl-runtime/src/inspect/mod.rs`
- `crates/nsl-runtime/src/inspect/ffi.rs`
- `crates/nsl-runtime/src/inspect/format.rs`
- `crates/nsl-runtime/src/inspect/stream.rs` (cuda-gated)
- `crates/nsl-runtime/src/inspect/stats_kernel.rs`
- Test files:
  - `crates/nsl-codegen/tests/inspect_predicate.rs`
  - `crates/nsl-codegen/tests/inspect_codegen.rs`
  - `crates/nsl-runtime/tests/inspect_format.rs`
  - `crates/nsl-runtime/tests/inspect_ffi.rs`
  - `crates/nsl-cli/tests/inspect_cli.rs`

**Modified:**

- `crates/nsl-semantic/src/checker/stmt.rs` — `@inspect` arg validation.
- `crates/nsl-codegen/src/lib.rs` `CompileOptions` — `inspect_enabled: bool`.
- `crates/nsl-codegen/src/stmt.rs` `compile_train_block` (and any sibling `let`-lowering site) — emit hooks.
- `crates/nsl-codegen/src/wrga_memory.rs` (or `binary_retention.rs`) — `pinned_until_inspect_sync` extension.
- `crates/nsl-codegen/src/builtins.rs` — register all new FFI symbols.
- `crates/nsl-runtime/src/lib.rs` — `pub mod inspect;`.
- `crates/nsl-runtime/src/health/ffi.rs` — add `pub(crate) fn collector_lock()` accessor.
- `crates/nsl-cli/src/main.rs::Cli::Run` — `--inspect` flag + summary print.
- `crates/nsl-cli/src/main.rs` — both `CompileOptions { ... }` literal sites get `inspect_enabled: ...`.

## 10. Follow-up phases

- **Phase 5.5** (optional): live stats viewer that tails `.nsl-inspect/`.
- **Phase 5.5** (optional): inspect-result analysis CLI (`nsl inspect <dir>` to summarize trends).
- Phase 5 closes the research-document feature set.
