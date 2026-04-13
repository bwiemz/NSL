# NSL Dev Tools — Phase 1 Design

**Date:** 2026-04-12
**Status:** Design approved, ready for implementation plan
**Scope source:** `nsl dev tools.pdf` (research proposal, April 2026)

## 1. Purpose

Deliver a compiler-native debugging and profiling toolkit that exploits the NSL compiler's complete knowledge of the program (cost model, memory planner, WGGO, fusion plan, GPU target specs) to give developers capabilities no runtime profiler can match:

- Per-op performance prediction **before** running on a GPU.
- Memory-usage timeline rendered from the allocation plan.
- Compile-time tensor-shape propagation trace with exact error attribution.
- Runtime kernel timing that compares measured latency to the cost-model's prediction.
- Source-mapped kernel view showing which NSL source lines each emitted kernel covers, and which lines were eliminated by fusion.

## 2. Scope

**In scope (Phase 1):**

1. `nsl check --shapes` — compile-time shape-propagation trace.
2. `nsl profile` — per-op cost-model report (FLOPs, HBM, arithmetic intensity, roofline bound, estimated time).
3. `nsl profile --memory` — HBM-usage timeline rendered from the memory planner.
4. `nsl run --monitor` — runtime kernel timing, predicted-vs-actual comparison table, and source-mapped kernel view.

**Explicitly deferred to later phases:**

- WGGO decision explainer (`nsl profile --explain-wggo`). The ILP solver records final decisions but not alternatives/binding constraints; instrumenting the solver is its own project.
- Training health monitor (live per-layer gradient norms, weight norms, NaN watch, loss EMA).
- Tensor inspector (`@inspect` decorator + async tensor-stat collection).

## 3. Architecture

Four CLI features share a single new module, `nsl-codegen/src/profiling/`, that produces a flat per-op prediction report from a typed AST. Each feature is a thin CLI frontend over this shared layer plus one existing subsystem.

### 3.1 Shared layer: `nsl-codegen/src/profiling/`

- `walker.rs` — `walk_ops(typed_module, &GpuSpec, dtype) -> ProfileReport`. Walks the typed AST, dispatches to the per-op cost functions in `cost_model.rs` (`matmul_cost`, `flash_attention_cost`, `softmax_cost`, `layernorm_cost`, `embedding_cost`, …), and produces:
  - `Vec<OpCost>` with `source_span` attached per entry.
  - Aggregate totals (FLOPs, HBM bytes, estimated time).
  - Roofline classification histogram.
  - `FusionPlan` (from `build_fusion_plan`) paired with the ops it affects.
  - `Vec<Recommendation>` — simple heuristics ("batch > 4 would improve memory-bound ops", "LM head dominates — consider INT4").
- `memory_timeline.rs` — consumes a `MemoryPlan`, converts `SlotAssignment { birth, death, size_bytes }` intervals into a time-ordered HBM-usage series, and annotates key phase boundaries (forward start, loss, backward begin, optimizer step) from Wengert op-kind labels.
- `instrument.rs` — gated by `CodegenOptions.profile: bool`. Wraps each emitted GPU kernel launch with calls to new runtime hooks (`nsl_profile_kernel_begin`, `nsl_profile_kernel_end`). Writes `<out>.nsl-profile.json` (the "manifest") alongside the binary.

Unknown AST ops produce an `OpCost { flops: 0, bytes: 0, note: "unknown op" }` so the report is honest about coverage rather than silently skipping.

### 3.2 Per-feature frontends

| Feature              | Entry                | Reuses                                         | New code (approx)     |
|----------------------|----------------------|-------------------------------------------------|------------------------|
| `nsl check --shapes` | `nsl-cli`            | semantic pass output                            | 200 LOC (formatter)    |
| `nsl profile`        | `nsl-cli`            | `walk_ops` + `GpuSpec` + `format_perf_table`    | 400 LOC                |
| `nsl profile --memory` | `nsl-cli`          | `walk_ops` + `MemoryPlan`                        | 200 LOC                |
| `nsl run --monitor`  | `nsl-cli` + codegen + runtime | `walk_ops` predictions + codegen kernel emit | 600 LOC                |

**Total: ~1,400 LOC.**

### 3.3 Reused existing infrastructure

| Existing component           | Location                                        |
|------------------------------|-------------------------------------------------|
| `OpCost` and per-op cost fns | `crates/nsl-codegen/src/cost_model.rs`          |
| `GpuSpec` + `GPU_DATABASE`   | `crates/nsl-codegen/src/gpu_specs.rs`           |
| `MemoryPlan`, `SlotAssignment`, `MemoryPlanStats` | `crates/nsl-codegen/src/wrga_memory.rs` |
| `FusionPlan`, `FusionDecision` | `crates/nsl-codegen/src/wrga_fusion.rs`       |
| `format_perf_table`          | `crates/nsl-codegen/src/cost_model.rs`          |
| CLI dispatch                 | `crates/nsl-cli/src/main.rs` (clap enum)         |

## 4. Feature Designs

### 4.1 `nsl check --shapes`

Extends the existing `check` subcommand with a `--shapes` flag.

**Pipeline:**

1. Run parser + semantic analyzer (existing path).
2. Walk the typed AST in source order. For each `let`-binding and top-level expression producing a `Tensor<...>`, emit a line with:
   - Source snippet (left column).
   - Inferred shape (right column).
   - ✅ if semantic analysis accepted the node, ❌ if it rejected.
3. On any shape-mismatch error, reuse the existing `nsl-errors` rendering for the "Expected / Cause / Fix / Source" block shown in §5 of the PDF.
4. Final line prints total FLOPs by calling into the §4.2 walker (read-only — no cost-model state mutation).

**No changes to the semantic analyzer.** Purely a read-only formatter.

**New files:**

- `crates/nsl-cli/src/shape_debug.rs` (~200 LOC).

**Flag wiring:** new `--shapes` bool on the `Check` clap variant; match arm calls `shape_debug::run(...)`.

### 4.2 `nsl profile`

New clap subcommand.

**Flags:**

- `--target <gpu>` — default: first entry in `GPU_DATABASE` (H100-SXM).
- `--dtype <bf16|fp16|fp8>` — default: `bf16`.
- `--fusion` / `--no-fusion` — default: on. Shows pre- and post-fusion timing.
- `--memory` — switches to §4.3 renderer. Mutually exclusive with the default per-op table output.

**Pipeline:**

1. Parse + semantic-analyze.
2. Resolve target via `find_gpu(&flag)`. Error if not found.
3. Call `walk_ops(module, gpu, dtype) -> ProfileReport`.
4. If `--fusion` is on: call `build_fusion_plan(...)` and overlay fusion groupings on the table. Print a "With fusion" section showing: fused kernel count, HBM bytes saved, per-layer before/after times.
5. Render with `format_perf_table` for the main table; append a fusion summary block and a recommendations block.

**Recommendations heuristics (kept small and explicit):**

- Any memory-bound op with AI < 2 → suggest increasing batch size.
- Any single op > 10% of total time → flag as "dominates; consider quantization".
- If `FusionPlan` eliminates ≥ 3 kernel launches per layer → "fusion strongly recommended".

**New files:**

- `crates/nsl-codegen/src/profiling/mod.rs`
- `crates/nsl-codegen/src/profiling/walker.rs` (~250 LOC)
- `crates/nsl-cli/src/profile.rs` (~150 LOC — clap variant, dispatch, renderer).

### 4.3 `nsl profile --memory`

Adds `--memory` flag to `nsl profile` (alias: `--memory-timeline`).

**Pipeline:**

1. Runs §4.2's parse/analyze/walk pipeline.
2. Builds a Wengert list from the typed AST using the same path as the WRGA flow.
3. Calls `plan_memory(...)` to get a `MemoryPlan`.
4. Passes the plan to `memory_timeline::render(plan, wengert_list)`.

**Renderer:**

- At each distinct program point (Wengert op index), sum live slot sizes to produce the HBM-usage value.
- Bucket program points into time-labeled rows (microseconds derived from the cost-model estimated_time_us, accumulated per op).
- Annotate phase boundaries (forward start/end, loss, backward begin, optimizer step) by reading op-kind labels already on Wengert ops.
- Print an ASCII bar chart matching the PDF format: `time | bar | MB | annotation`.
- Print peak memory. Print two delta lines — "With FASE" (uses `planned_peak_bytes` minus the gradient-buffer savings already tracked in the plan) and "With gradient checkpointing" (applies the static estimate already present in `MemoryPlanStats`).

**New files:**

- `crates/nsl-codegen/src/profiling/memory_timeline.rs` (~150 LOC).
- ~50 LOC wiring in `nsl-cli/src/profile.rs`.

**No changes** to `wrga_memory.rs` or the planner.

### 4.4 `nsl run --monitor` + source-mapped kernel view

Split across three crates. Overall pipeline:

```
compile:  codegen emits kernels + instrumentation + <out>.nsl-profile.json (predictions)
run:      runtime records cudaEvent timings + writes <out>.nsl-profile-actual.json
report:   CLI reads both JSONs, renders predicted-vs-actual table + source-mapped view
```

**Codegen — `nsl-codegen/src/profiling/instrument.rs` (~200 LOC):**

1. New `CodegenOptions.profile: bool`, set when CLI passes `--monitor`.
2. For each emitted GPU kernel launch, the emitter wraps the launch with `nsl_profile_kernel_begin(kernel_id)` / `nsl_profile_kernel_end(kernel_id)`. `kernel_id` is a dense u32 assigned during emission.
3. Each `kernel_id` is associated with the `OpId` (and hence source span + predicted cost) it came from. Operations eliminated by fusion do **not** get a `kernel_id`; they are recorded in a separate `eliminated_ops` list.
4. A static JSON manifest is written to `<out>.nsl-profile.json`:
   ```json
   {
     "target_gpu": "h100-sxm",
     "dtype": "bf16",
     "kernels": [
       { "kernel_id": 0, "op_name": "fused_attn", "source_span": {"file": "model.nsl", "start_line": 42, "end_line": 48}, "predicted_us": 9.2, "predicted_flops": 83900000, "predicted_hbm_bytes": 12582912 },
       ...
     ],
     "eliminated_ops": [
       { "op_name": "chunk", "source_span": {"file": "model.nsl", "start_line": 45, "end_line": 45}, "reason": "fused into fused_attn (kernel_id=0)" },
       ...
     ]
   }
   ```
5. When `profile: false`, no hooks are emitted and no manifest is written. Zero overhead.

**Runtime — `nsl-runtime/src/profiler/collector.rs` (~200 LOC):**

1. `nsl_profile_kernel_begin(kernel_id)` creates (or reuses) a `cudaEvent_t`, records it on the current stream, and pushes into a per-thread ring buffer keyed by `kernel_id`.
2. `nsl_profile_kernel_end(kernel_id)` records a matching end event.
3. On program exit (via a runtime `Drop` hook, plus an explicit `nsl_profile_flush()` FFI for host-driven flushes), the collector:
   - Synchronizes the stream.
   - Calls `cudaEventElapsedTime` on each pair.
   - Aggregates per `kernel_id`: `{ count, sum_us, min_us, max_us }`.
   - Writes `<out>.nsl-profile-actual.json` next to the binary.
4. When the binary is not instrumented, the FFI symbols are not linked; zero cost.

**CLI — `nsl-cli/src/monitor.rs` (~200 LOC):**

1. `nsl run --monitor <file>`: runs the normal `run` pipeline with `profile: true`, waits for the child process to exit, then:
2. Reads `<out>.nsl-profile.json` and `<out>.nsl-profile-actual.json`.
3. Renders the predicted-vs-actual table (Layer, Operation, Predicted µs, Actual µs, Δ%). Color / symbol threshold: warn at |Δ| > 5%, error-flag at |Δ| > 20%.
4. When |Δ| > 20% for any kernel, emits a "likely cause" heuristic line:
   - matmul whose inner dim isn't a multiple of 128 → "tile-size misalignment".
   - attention whose SMEM estimate exceeds 80% of the GPU's per-SM limit → "SMEM pressure / bank conflicts".
   - all other cases → "cause unknown; rerun with --target-profile=detailed" (detailed mode is out of scope for Phase 1; the hint is a forward-looking signpost).
5. Renders the **source-mapped kernel view**: groups kernels and eliminated ops by source file, prints the source listing with per-line annotations:
   ```
   42 | fn forward(x: Tensor<...>) -> Tensor:
   43 |     let h = self.ln1(x)            ← rmsnorm (fused, 0.3μs)
   44 |     let qkv = self.c_attn(h)       ← qkv_proj (fused, 4.1μs)
   45 |     let (q, k, v) = qkv.chunk(3)   ← eliminated (fused)
   ```

## 5. Data flow

```
           typed AST
               │
     ┌─────────┼─────────────────────────────────┐
     │         │                                 │
     ▼         ▼                                 ▼
 shape-fmt  walk_ops ─► ProfileReport ─► perf-table / memory-timeline
                            │                    │
                            │                    └──► nsl profile output
                            │
                            └──► instrument.rs ──► manifest JSON
                                         │
                                    (compile + run)
                                         │
                                         ▼
                             nsl-runtime collector
                                         │
                                         ▼
                                    actual JSON
                                         │
                                         ▼
                                 monitor.rs renderer
                                         │
                                         ▼
                               predicted-vs-actual + source view
```

## 6. Error handling

- Unknown AST op in walker: emit `OpCost` with `flops = 0, note = "unknown op"`. Do not silently skip.
- Unknown GPU target flag: hard error listing the available GPUs from `GPU_DATABASE`.
- Missing actual-JSON at monitor render time: warn and show predictions only (the run may have crashed before flush).
- Missing manifest JSON at monitor render time: hard error — the binary wasn't built with `--monitor`.
- Kernel ID present in actual but absent in manifest (or vice versa): warn per-kernel, render what's available. Do not abort.

## 7. Testing

- **Unit tests (Rust `cargo test`):**
  - `walker`: given a small typed AST (matmul + softmax + layernorm), produces the expected `OpCost` list.
  - `memory_timeline`: given a hand-built `MemoryPlan` with overlapping intervals, produces correct peak and correct ordering.
  - `instrument`: given a mock emitter, wraps each kernel launch with exactly one begin/end pair and assigns unique kernel IDs.
  - JSON round-trip for both manifest and actual.
- **Integration tests (existing `tests/` harness):**
  - Golden snapshot test per CLI feature, comparing formatted output to committed expected-output files.
  - `nsl run --monitor` integration test gated on `cfg(feature = "cuda")` — compiles a tiny model, runs it, checks both JSONs are produced and the comparator exits 0.
- **Performance regression test:** run `nsl profile` on a fixed sample model and assert wall-time < 500 ms.

## 8. Non-goals

- No cost-model feedback loop in Phase 1. The predicted-vs-actual delta is shown but not used to retrain or adjust the cost model.
- No interactive TUI. All output is batched text.
- No Chrome tracing export (the `--trace` dormant flag is left dormant).
- No WGGO decision explainer (deferred).

## 9. Open questions

None blocking. Detail decisions (exact color scheme, wall-clock bucket width for the memory timeline, JSON schema version field) are implementation concerns deferred to the plan.

## 10. File inventory (summary)

**New:**

- `crates/nsl-codegen/src/profiling/mod.rs`
- `crates/nsl-codegen/src/profiling/walker.rs`
- `crates/nsl-codegen/src/profiling/memory_timeline.rs`
- `crates/nsl-codegen/src/profiling/instrument.rs`
- `crates/nsl-cli/src/shape_debug.rs`
- `crates/nsl-cli/src/profile.rs`
- `crates/nsl-cli/src/monitor.rs`
- `crates/nsl-runtime/src/profiler/mod.rs`
- `crates/nsl-runtime/src/profiler/collector.rs`

**Edited:**

- `crates/nsl-cli/src/main.rs` — add `Profile` and `Monitor` (or `--monitor` on `Run`) clap variants.
- `crates/nsl-codegen/src/options.rs` (or equivalent) — add `profile: bool`.
- Existing codegen kernel-emit site(s) — invoke `instrument::wrap_launch` when `profile` is on.

## 11. Follow-up phases

- **Phase 2** — WGGO decision explainer (`nsl profile --explain-wggo`). Requires instrumenting `wggo_ilp.rs` to record alternatives and binding constraints.
- **Phase 3** — Training health monitor (`nsl run --monitor --health`). Live per-layer gradient/weight norms, NaN watch, loss EMA trend. Separate collector, hooks on training-loop boundaries rather than kernel launches.
- **Phase 4** — Tensor inspector (`@inspect` decorator + codegen + runtime async-dump collector).
