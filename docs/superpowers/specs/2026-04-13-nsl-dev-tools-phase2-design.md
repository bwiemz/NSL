# NSL Dev Tools — Phase 2 Real Codegen Hooks Design

**Date:** 2026-04-13
**Status:** Design approved, ready for implementation plan
**Builds on:** `docs/superpowers/specs/2026-04-12-nsl-dev-tools-phase1-design.md`
**Branch (continued):** `feat/dev-tools-phase1`

## 1. Purpose

Close the three explicit Phase 1 TODOs: emit real `nsl_profile_kernel_begin/end` hooks around every GPU kernel launch in compiled NSL programs, replace the runtime collector's host clock with real CUDA-event timing, and let codegen own the manifest so kernel IDs reflect what was actually emitted (post-fusion) rather than what the walker predicted.

After Phase 2, `nsl run --monitor` produces a fully populated predicted-vs-actual table — no more "Note: no actual timings collected" banner.

## 2. Scope

**In:**

1. Codegen pre-pass that runs the walker once to build a `HashMap<NodeId, OpCost>` of predictions.
2. `Compiler` carries a `ManifestBuilder` when `compile_options.profile_kernels` is true; each kernel launch reserves an ID, records the kernel, and emits hook constants into Cranelift IR.
3. `compile_gpu_kernel_launch` signature extended with `origin_node`, `span`, and `op_name_hint`. The existing single launch site in `expr/calls.rs` is the only code that needs span-threading.
4. After codegen, write `<out>.nsl-profile.json` from the codegen-side builder.
5. New `CudaEventClock` (gated `#[cfg(feature = "cuda")]`) replacing `NanoClock` when CUDA is enabled. Pool of `cuEvent_t` handles, drain syncs the end event then computes elapsed.
6. CLI: `--monitor` implies `profile_kernels: true`; codegen-emitted manifest takes precedence over the Phase 1 fallback.

**Out:**

- WGGO decision explainer.
- Training health monitor / `@inspect`.
- Multi-stream profiling.
- Refactor / merge of `kernel_profiler.rs` (Chrome trace path stays separate).
- CPU-op timing.

## 3. Architecture

Four discrete components. No new modules in `nsl-codegen`; one new file in `nsl-runtime`.

| Component                         | Location                                                         | Description                                                              |
|-----------------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------|
| Codegen pre-pass + state          | `crates/nsl-codegen/src/compiler/mod.rs`, `lib.rs`               | Walker run before function bodies, two new fields on `Compiler`.         |
| Hook emission at launch site      | `crates/nsl-codegen/src/expr/calls.rs::compile_gpu_kernel_launch`| One launch site, three new params, three new Cranelift `call` insns.    |
| Manifest writer                   | `crates/nsl-codegen/src/lib.rs::compile()` post-pass             | Serialize and write next to the binary.                                 |
| Real CUDA-event clock             | `crates/nsl-runtime/src/profiler/cuda_clock.rs` (new)             | `CudaEventClock` impl `ClockSource`; `cuEvent_t` pool.                  |

### 3.1 Reused infrastructure (Phase 1)

- `nsl_codegen::profiling::types::*` — `ProfileReport`, `OpCost`, `EntryKind`.
- `nsl_codegen::profiling::walker::walk_ops` — same signature: `(module, analysis, interner, entry, env, gpu, dtype) -> Result<ProfileReport, String>`.
- `nsl_codegen::profiling::shape_env::ShapeEnv::with_defaults`.
- `nsl_codegen::profiling::instrument::{ManifestBuilder, write_manifest, SourceSpanJson}` — gains two new methods (`reserve_id`, `record_kernel_at`).
- `nsl_codegen::gpu_specs::find_gpu`.
- `nsl_runtime::profiler::collector::{Collector, ClockSource, Aggregate, ActualReport}` — `ClockSource` trait already in place; just plug in a new impl.
- `nsl_runtime::profiler::ffi::{nsl_profile_kernel_begin, nsl_profile_kernel_end, nsl_profile_flush}` — symbol names unchanged; bodies adapt to use CUDA-event handles when the `cuda` feature is on.

### 3.2 Existing field/option

`crates/nsl-codegen/src/lib.rs:342::CompileOptions.profile_kernels: bool` already exists from Phase 1 — Phase 2 wires it.

## 4. Component Designs

### 4.1 Codegen pre-pass

In `crates/nsl-codegen/src/compiler/mod.rs`, extend the `Compiler` struct:

```rust
pub struct Compiler<'a> {
    // ...existing fields...
    pub prediction_map: HashMap<NodeId, OpCost>,
    pub manifest_builder: Option<ManifestBuilder>,
    pub next_kernel_id: u32,            // monotonic counter
}
```

In the codegen entry point in `crates/nsl-codegen/src/lib.rs::compile()` (or `compile_module()`), before walking function bodies:

```rust
if opts.profile_kernels {
    let env = ShapeEnv::with_defaults();
    let gpu = find_gpu(&opts.target_gpu)
        .ok_or_else(|| CodegenError::UnknownGpu(opts.target_gpu.clone()))?;
    let report = walk_ops(
        &module, &analysis, &interner,
        EntryKind::Auto, &env, gpu, &opts.dtype,
    )?;
    compiler.prediction_map = report.ops.iter()
        .filter_map(|op| op.origin_node.map(|nid| (nid, op.clone())))
        .collect();
    compiler.manifest_builder = Some(ManifestBuilder::new(&opts.target_gpu, &opts.dtype));
}
```

This requires one additive change to `OpCost`:

```rust
// crates/nsl-codegen/src/cost_model.rs
pub struct OpCost {
    // ...existing...
    pub origin_node: Option<NodeId>,    // NEW
}
```

The walker (`walker.rs`) populates it from `expr.id` in every `push_raw` call. Existing tests pass with `origin_node: None` defaults.

### 4.2 Hook emission

In `crates/nsl-codegen/src/expr/calls.rs::compile_gpu_kernel_launch`, extend the signature:

```rust
fn compile_gpu_kernel_launch(
    &mut self,
    builder: &mut FunctionBuilder,
    ptx_ptr: Value, name_ptr: Value,
    grid: [Value; 3], block: [Value; 3],
    args_ptr: Value, arg_count: Value, smem: Value,
    origin_node: NodeId,        // NEW
    span: Span,                 // NEW
    op_name_hint: &str,         // NEW
) -> CodegenResult<Value>
```

Caller (`compile_call`) already has `call_expr` in scope, so it passes `call_expr.id`, `call_expr.span`, and an op-name hint derived from the callee.

Body adds three things, all gated on `self.manifest_builder.is_some()`:

```rust
let kernel_id = if let Some(mb) = self.manifest_builder.as_mut() {
    let id = mb.reserve_id();
    let pred = self.prediction_map.get(&origin_node).cloned();
    let span_json = SourceSpanJson::from_span(span, &self.source_map);
    mb.record_kernel_at(
        id,
        op_name_hint,
        span_json,
        pred.as_ref().map(|p| p.estimated_time_us).unwrap_or(0.0),
        pred.as_ref().map(|p| p.flops).unwrap_or(0),
        pred.as_ref().map(|p| p.bytes_read + p.bytes_written).unwrap_or(0),
    );
    Some(id)
} else { None };

if let Some(id) = kernel_id {
    let id_val = builder.ins().iconst(types::I32, id as i64);
    let begin = self.func_ref("nsl_profile_kernel_begin", &[types::I32], None);
    builder.ins().call(begin, &[id_val]);
}

// Existing nsl_kernel_launch call (unchanged)
let result_call = builder.ins().call(launch_ref, &[ptx_ptr, name_ptr, /* ... */]);

if let Some(id) = kernel_id {
    let id_val = builder.ins().iconst(types::I32, id as i64);
    let end = self.func_ref("nsl_profile_kernel_end", &[types::I32], None);
    builder.ins().call(end, &[id_val]);
}
```

`SourceSpanJson::from_span(span, source_map)` is a new helper. Resolves byte positions to 1-based line numbers using whatever `SourceMap` the codegen already passes around. If no `SourceMap` is in scope, the fallback is `line: 1, end_line: 1` — degraded but not broken.

`func_ref` is the codegen helper that gets-or-creates a Cranelift `FuncRef` for an extern symbol. Pattern is identical to the existing `nsl_kernel_launch` registration; just two more registrations once.

**Two new methods on `ManifestBuilder`** (additive; Phase 1 callers unaffected):

```rust
impl ManifestBuilder {
    pub fn reserve_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
    pub fn record_kernel_at(
        &mut self, id: u32, op_name: &str, span: SourceSpanJson,
        us: f64, flops: u64, hbm: u64,
    ) {
        self.inner.kernels.push(KernelEntry {
            kernel_id: id, op_name: op_name.into(),
            source_span: span,
            predicted_us: us, predicted_flops: flops, predicted_hbm_bytes: hbm,
        });
    }
}
```

The Phase 1 `record_kernel(...)` (auto-incrementing) becomes a thin wrapper: `let id = self.reserve_id(); self.record_kernel_at(id, ...)`. Phase 1 tests still pass.

### 4.3 Manifest writer

In `crates/nsl-codegen/src/lib.rs::compile()`, after function-body codegen completes:

```rust
if let Some(mb) = compiler.manifest_builder.take() {
    let manifest = mb.finish();
    let path = opts.output_path.with_extension("nsl-profile.json");
    write_manifest(&path, &manifest)
        .map_err(|e| CodegenError::ManifestWriteFailed(e.to_string()))?;
}
```

`opts.output_path` is the path of the binary being produced; if the existing API uses a different field name (e.g., `out_dir` + `name`), adapt at implementation time.

### 4.4 Real CUDA-event clock

New file `crates/nsl-runtime/src/profiler/cuda_clock.rs` (gated `#[cfg(feature = "cuda")]`):

```rust
use cudarc::driver::sys;
use std::sync::Mutex;
use crate::profiler::collector::ClockSource;

pub struct CudaEventClock {
    pool: Mutex<Vec<sys::CUevent>>,
}

impl CudaEventClock {
    pub fn new() -> Self { Self { pool: Mutex::new(Vec::with_capacity(128)) } }

    pub fn checkout_event(&self) -> u64 {
        let mut pool = self.pool.lock().unwrap();
        if let Some(e) = pool.pop() { return e as u64; }
        unsafe {
            let mut e: sys::CUevent = std::ptr::null_mut();
            sys::lib().cuEventCreate(&mut e, 0).result().unwrap();
            e as u64
        }
    }
    fn return_event(&self, h: u64) {
        self.pool.lock().unwrap().push(h as sys::CUevent);
    }
}

impl ClockSource for CudaEventClock {
    fn elapsed_us(&self, start: u64, end: u64) -> f64 {
        unsafe {
            sys::lib().cuEventSynchronize(end as sys::CUevent).result().unwrap();
            let mut ms: f32 = 0.0;
            sys::lib().cuEventElapsedTime(&mut ms, start as sys::CUevent, end as sys::CUevent)
                .result().unwrap();
            self.return_event(start);
            self.return_event(end);
            ms as f64 * 1000.0
        }
    }
}
```

Add `pub mod cuda_clock;` to `crates/nsl-runtime/src/profiler/mod.rs` under the same cfg gate.

### 4.5 FFI hooks become event-aware

`crates/nsl-runtime/src/profiler/ffi.rs`:

```rust
#[cfg(feature = "cuda")]
static CUDA_CLOCK: Lazy<CudaEventClock> = Lazy::new(CudaEventClock::new);

#[no_mangle]
pub extern "C" fn nsl_profile_kernel_begin(kernel_id: u32) {
    #[cfg(feature = "cuda")]
    {
        let event = CUDA_CLOCK.checkout_event();
        unsafe {
            cudarc::driver::sys::lib()
                .cuEventRecord(event as _, current_stream())
                .result()
                .ok();
        }
        let mut g = COLLECTOR.lock().unwrap();
        g.get_or_insert_with(|| Collector::new_with_clock(Box::new(CudaEventClock::new())))
            .begin(kernel_id, event);
        return;
    }
    #[cfg(not(feature = "cuda"))]
    {
        let t = now_ns();
        let mut g = COLLECTOR.lock().unwrap();
        g.get_or_insert_with(|| Collector::new_with_clock(Box::new(NanoClock)))
            .begin(kernel_id, t);
    }
}

// nsl_profile_kernel_end mirrors the same shape.
```

`current_stream()` returns the same `CUstream` the runtime's `nsl_kernel_launch` uses. The runtime already maintains thread-local CUDA context state — confirm at implementation time which call returns the active stream and use it.

The `Collector` itself is unchanged. The `cuda-real-events` feature stub from Phase 1 is removed; replaced by the existing `cuda` feature.

### 4.6 CLI wiring

In `crates/nsl-cli/src/main.rs::Run` arm, when `monitor` is true, set `compile_options.profile_kernels = true` before calling into codegen. This is one line.

`monitor.rs::run_monitor` already prefers an existing manifest file — Phase 2 just means the manifest now arrives via codegen rather than via `write_manifest_beside`. The CLI's Phase 1 fallback stays as-is for the case where the user runs `--monitor` against an already-built binary that wasn't compiled with profiling.

## 5. Data flow

```
nsl run --monitor file.nsl
   │
   ▼
CLI: opts.profile_kernels = true
   │
   ▼
codegen::compile(module, opts)
   │
   ├─ pre-pass:  walk_ops → prediction_map (NodeId → OpCost)
   ├─ pre-pass:  manifest_builder = Some(ManifestBuilder::new(...))
   │
   ├─ per kernel launch:
   │     id = manifest_builder.reserve_id()
   │     pred = prediction_map.get(call_expr.id)
   │     manifest_builder.record_kernel_at(id, name, span, pred...)
   │     emit:  nsl_profile_kernel_begin(id)
   │            nsl_kernel_launch(...)        [existing]
   │            nsl_profile_kernel_end(id)
   │
   └─ post-pass: write_manifest(<out>.nsl-profile.json)
   │
   ▼
runtime (linked with cuda):
   nsl_profile_kernel_begin(id)
       → CudaEventClock.checkout_event() → cuEventRecord on current stream
       → Collector::begin(id, event_handle)
   nsl_kernel_launch        → cuLaunchKernel
   nsl_profile_kernel_end(id)
       → checkout_event → cuEventRecord → Collector::end(id, event_handle)
   on Drop / nsl_profile_flush:
       Collector::snapshot → for each pair: cuEventSynchronize(end), cuEventElapsedTime
       → return events to pool
       → write <out>.nsl-profile-actual.json
   │
   ▼
CLI reads both JSONs → render predicted-vs-actual + source view (Phase 1 monitor unchanged)
```

## 6. Error handling

| Failure                                                        | Behavior                                                                          |
|----------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Walker pre-pass fails (bad shape, unknown GPU)                 | Hard error before any IR emit; user fixes and rebuilds.                           |
| `compile_gpu_kernel_launch` called for a kernel with no `NodeId` match | Record `KernelEntry` with `predicted_us = 0.0`; monitor shows "n/a" predicted.    |
| Manifest write fails (disk full, perm denied)                  | `CodegenError::ManifestWriteFailed`; codegen aborts.                              |
| `cuEventCreate` returns error                                  | Panic during pool init — matches existing CUDA-init failure mode in the runtime.  |
| `cuEventSynchronize` / `cuEventElapsedTime` returns error      | Log warning, drop the pair, continue. Don't crash the user's training loop.       |
| Ring buffer fills before drain                                 | Phase 1 invariant holds — drain at cap 64 forces sync; no unbounded growth.       |
| `--monitor` against binary built without `profile_kernels`     | Codegen-side manifest absent → CLI falls back to Phase 1 `write_manifest_beside`. |

## 7. Testing

### Unit (host-only)

- `ManifestBuilder::reserve_id` returns 0, 1, 2, … monotonically across mixed `record_kernel` / `record_kernel_at` calls.
- `record_kernel_at` with a caller-supplied ID writes the entry verbatim, no auto-increment.
- Walker pre-pass: given a typed module with N `matmul` calls, `prediction_map` contains N entries each keyed by the corresponding `NodeId`.
- `OpCost.origin_node` is populated by every walker push site. Old tests still pass (origin_node is `Option`, defaults to `None` in test fixtures).

### Integration (`#[cfg(feature = "cuda")]`)

- Compile a tiny model with `profile_kernels: true` and verify `<out>.nsl-profile.json` exists. Round-trip through `Manifest::deserialize`. Assert `kernels.len() >= 1` and every entry has a non-empty `op_name`.
- Compile + run a tiny GPU model. Read `<out>.nsl-profile-actual.json`. Assert `aggregates.len() >= 1` and the first aggregate has `count >= 1`, `sum_us > 0.0`. **This is the first end-to-end test that closes the predicted-vs-actual loop.**
- Run with N iterations; assert `count == N` and `sum_us` scales linearly.

### Smoke

```
nsl run --monitor tests/fixtures/tiny_transformer.nsl
```

Output table has real µs numbers in the Actual column. The "Note: no actual timings collected" banner is gone.

### Regression

All Phase 1 tests pass unchanged.

## 8. Non-goals

- No merge of `kernel_profiler.rs` (Chrome trace) and `profiler/`. They stay parallel; users pick either via flags.
- No multi-stream profiling. Hooks attach to whichever stream `nsl_kernel_launch` ran on; if a future codegen variant uses multiple streams concurrently, the design needs to revisit per-stream event pools.
- No source-level CPU op timing.
- No autotuning feedback loop — predictions stay frozen at compile time.
- No Δ-vs-prediction-driven cost model retraining (a separate research project).

## 9. Open questions

None blocking. Two implementation-time confirmations:

1. Which runtime call returns the active `CUstream` for the current thread — `current_stream()` is a placeholder name in this spec.
2. Whether `CompileOptions.output_path` is the actual field name (or if it's `out_dir + name`).

Both are local lookups that don't change the design.

## 10. File inventory

**New:**

- `crates/nsl-runtime/src/profiler/cuda_clock.rs`

**Modified:**

- `crates/nsl-codegen/src/compiler/mod.rs` — three new fields on `Compiler`.
- `crates/nsl-codegen/src/cost_model.rs` — `OpCost.origin_node: Option<NodeId>`.
- `crates/nsl-codegen/src/profiling/walker.rs` — populate `origin_node`.
- `crates/nsl-codegen/src/profiling/instrument.rs` — add `reserve_id`, `record_kernel_at`, `SourceSpanJson::from_span`.
- `crates/nsl-codegen/src/lib.rs` — pre-pass + manifest write in `compile()`.
- `crates/nsl-codegen/src/expr/calls.rs` — extend `compile_gpu_kernel_launch` signature, emit hooks.
- `crates/nsl-runtime/src/profiler/ffi.rs` — event-aware FFI under `#[cfg(feature = "cuda")]`.
- `crates/nsl-runtime/src/profiler/mod.rs` — gate-publish `cuda_clock`.
- `crates/nsl-runtime/Cargo.toml` — remove `cuda-real-events` stub feature.
- `crates/nsl-cli/src/main.rs::Run` — set `profile_kernels = true` when `--monitor`.

## 11. Follow-up phases

- **Phase 3** — WGGO decision explainer (`nsl profile --explain-wggo`).
- **Phase 4** — Training health monitor (`nsl run --monitor --health`).
- **Phase 5** — Tensor inspector (`@inspect` decorator + codegen + runtime async-dump collector).
- **Multi-stream profiling** — per-stream event pools.
