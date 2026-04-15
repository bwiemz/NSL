# FASE Codegen Phase 2 — Per-Layer Mode Dispatch in the Backward Loop

**Date:** 2026-04-15
**Status:** Approved for implementation
**Branch (target):** `feat/fase-codegen-phase2`
**Predecessor:** [FASE Per-Layer Mode Override (Phase 1)](2026-04-15-fase-per-layer-mode-design.md) (merged via PR #39 at `661e7d5`).

## 1. Goal

Make the FASE backward loops in `stmt.rs` consult a per-parameter mode array at runtime instead of branching on the global `fase_deferred: bool`. After this lands, WGGO's `fase_fused: bool` per-layer signal influences emitted code — not just the planner output and stderr diagnostics.

This closes out the "TODO(fase-consumer-3-phase-2)" comment Phase 1 left at `stmt.rs:3302`.

## 2. Non-Goals (Explicitly Deferred)

- Changes to the `param_paths` enumeration order. `enumerate_model_tensor_paths` is the load-bearing invariant that aligns `param_list`, `accum_list`, and the runtime `grads_list`; this plan does not touch it.
- Per-layer hyperparameter dispatch (LR, β, ε, accumulation, grad_clip — still global by design per the Phase 1 spec).
- Optimizer-state schema changes (`m_partial` layout unchanged).
- Per-layer dispatch inside the FASE source-AD hook (`fase_hook_active` branch). The hook bypasses the standard accumulation loop entirely; per-layer dispatch lands inside the hook only if a follow-up plan needs it.
- Profile-guided micro-optimization of the runtime branch (e.g. split-loop variant B from brainstorming). The current shape produces well-predicted branches because modes cluster by layer.

## 3. Critical Invariants (do not break)

### 3.1 Param-list order
`param_paths: Vec<String>` — produced once by `enumerate_model_tensor_paths(model_var_name, model_type_name)` at `stmt.rs:3370` — drives **all three** runtime list constructions in lockstep:
- `param_list` (line ~3371)
- `accum_list` (line ~3482)
- `grads_list` (line ~3661, populated by tape backward / source-AD)

So at runtime, `grads_list[i]`, `accum_list[i]`, and `param_list[i]` all correspond to `param_paths[i]`. Phase 2 leverages this: the per-param mode array is also indexed by `i`.

### 3.2 No-override fallback is byte-identical
When `fase_plan.per_layer_mode.is_none()` (the default — no WGGO active), Phase 2 emits zero new IR. The three loops keep their existing `if fase_deferred { ... } else { ... }` shape. Every existing test must produce identical Cranelift IR.

## 4. Design

### 4.1 Compile-time mode resolution

New module `crates/nsl-codegen/src/fase_codegen_table.rs` exposes a single helper:

```rust
/// Per-parameter FASE mode encoding for the runtime dispatch table.
/// Encoded as `u8` for compact `.rodata` storage and single-byte loads.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamMode {
    Passthrough = 0,
    Deferred    = 1,
    FullBuffer  = 2,
}

impl From<crate::fase::FaseMode> for ParamMode { /* obvious */ }

/// Build a per-parameter mode table aligned with `param_paths`.
///
/// Returns `None` when no WGGO per-layer overrides are active — caller
/// should skip rodata emission and use the global-mode dispatch path.
///
/// Returns `Some(modes)` of length `param_paths.len()` otherwise. Each
/// `param_paths[i]` is matched against `overrides.find_by_layer_containing`
/// to find the layer index, then `plan.per_layer_mode[layer_idx]` gives
/// the mode. Unmatched paths fall back to `plan.mode` (global default).
pub fn build_param_mode_table(
    param_paths: &[String],
    plan: &crate::fase::FasePlan,
    overrides: Option<&crate::wggo_overrides::WggoOverrides>,
) -> Option<Vec<u8>> { ... }
```

**Behavior contract:**
| `plan.per_layer_mode` | `overrides` | Result |
|---|---|---|
| `None` | any | `None` (use global dispatch) |
| `Some(_)` | `None` | `None` (no overrides supplied; use global dispatch) |
| `Some(modes)` | `Some(o)` | `Some(Vec<u8>)` of length `param_paths.len()` |

For each `path` in `param_paths`:
1. `o.find_by_layer_containing(path)` returns `Option<&PerLayerOverride>` with `.layer_index`.
2. If matched, `plan.per_layer_mode[layer_index]` gives the per-layer `FaseMode`.
3. If unmatched (param doesn't fall under any WGGO-tracked layer), use `plan.mode` (global).
4. Convert to `ParamMode as u8` and push.

### 4.2 `.rodata` emission

New helper on `Compiler`:

```rust
/// Emit a per-parameter FASE mode table into `.rodata` and return the DataId.
/// Caller wraps in a `GlobalValue` and obtains a base pointer for runtime loads.
fn emit_param_mode_table_rodata(&mut self, modes: &[u8]) -> DataId
```

Implementation mirrors the M52 weight-hash pattern at `compiler/mod.rs:1048+`:
1. `self.module.declare_data(name, Linkage::Local, false /*writable*/, false /*tls*/)` with `name = format!("nsl_fase_param_modes_{func_name}")`.
2. `DataDescription::new()` + `desc.define(modes.into())`.
3. `self.module.define_data(data_id, &desc)`.

The symbol name embeds the train-block function name to avoid collisions when multiple `@train` blocks exist in one compile.

### 4.3 Per-loop runtime dispatch

New helper on `Compiler` (in `stmt.rs` adjacent to `fase_emit_*`):

```rust
/// Emit a runtime branch on the per-parameter FASE mode.
/// Caller provides the destination blocks; this helper loads `modes[gai]`,
/// compares against `Deferred (1)`, and emits a conditional jump.
fn emit_fase_mode_branch(
    &mut self,
    builder: &mut FunctionBuilder,
    mode_table_base: Value,    // i64 base pointer from GlobalValue
    gai: Value,                 // i64 param index
    deferred_block: Block,
    fullbuffer_block: Block,
)
```

Body:
```rust
// Load modes[gai] as u8
let byte_addr = builder.ins().iadd(mode_table_base, gai);  // gai is i64; element size = 1
let mode = builder.ins().load(cl_types::I8, MemFlags::trusted(), byte_addr, 0);
let one_i8 = builder.ins().iconst(cl_types::I8, 1);
let is_deferred = builder.ins().icmp(IntCC::Equal, mode, one_i8);
builder.ins().brif(is_deferred, deferred_block, &[], fullbuffer_block, &[]);
```

### 4.4 Three loops to update

For each of the three sites, the transformation is identical in shape:

**Before:**
```rust
let grad = compile_call_by_name("nsl_list_get", &[grads_list, gai]);
if fase_deferred {
    self.fase_emit_accumulate(builder, accum_buf, grad, fase_plan.recipe.accum_scale)?;
} else {
    self.compile_call_by_name(builder, "nsl_grad_accumulate_add", &[accum_buf, grad, n_elems])?;
}
self.compile_call_by_name(builder, "nsl_tensor_free", &[grad])?;
```

**After (when `mode_table_base.is_some()`):**
```rust
let grad = compile_call_by_name("nsl_list_get", &[grads_list, gai]);
let deferred_blk  = builder.create_block();
let fullbuf_blk   = builder.create_block();
let join_blk      = builder.create_block();

self.emit_fase_mode_branch(builder, mode_table_base.unwrap(), gai, deferred_blk, fullbuf_blk);

// Deferred path
builder.switch_to_block(deferred_blk);
builder.seal_block(deferred_blk);
self.fase_emit_accumulate(builder, accum_buf, grad, fase_plan.recipe.accum_scale)?;
builder.ins().jump(join_blk, &[]);

// FullBuffer path
builder.switch_to_block(fullbuf_blk);
builder.seal_block(fullbuf_blk);
let n_elems = self.compile_call_by_name(builder, "nsl_tensor_len", &[accum_buf])?;
self.compile_call_by_name(builder, "nsl_grad_accumulate_add", &[accum_buf, grad, n_elems])?;
builder.ins().jump(join_blk, &[]);

// Join — single tensor_free regardless of path
builder.switch_to_block(join_blk);
builder.seal_block(join_blk);
self.compile_call_by_name(builder, "nsl_tensor_free", &[grad])?;
```

**After (when `mode_table_base.is_none()`):** unchanged from today's monolithic branch.

The three sites:
1. **Accumulation loop** — `stmt.rs` `ga_body` block, ~line 4869–4893.
2. **Two-phase-clip Phase A loop** — `stmt.rs` `pa_body` block, ~line 4988–5019.
3. **Optimizer step loop** — `stmt.rs` ~line 4940+ (the `fase_deferred` branch around the per-param fused step vs separate optimizer call).

The same `mode_table_base: Option<Value>` is threaded into all three loop emitters from a single allocation site near the top of `compile_train_block`:

```rust
let mode_table = crate::fase_codegen_table::build_param_mode_table(
    &param_paths, &fase_plan, self.wggo_overrides.as_ref(),
);
let mode_table_base: Option<Value> = mode_table.as_ref().map(|modes| {
    let data_id = self.emit_param_mode_table_rodata(modes);
    let global = self.module.declare_data_in_func(data_id, builder.func);
    builder.ins().symbol_value(cl_types::I64, global)
});
```

### 4.5 Encoding rationale

`u8` storage with `Deferred = 1` chosen because:
- Single byte / param → 320 bytes for a 32-layer transformer with ~10 params each. Trivial `.rodata` footprint.
- Single `load.i8` + `icmp.eq` + `brif` per iteration. No multi-way dispatch needed because Passthrough collapses to FullBuffer at runtime (both take the non-fused code path — same as today).
- Branch-predictor friendly: modes cluster by layer, so the runtime branch sees long runs of the same outcome with transitions only at layer boundaries.

## 5. Architecture Diagram

```
                                                  Compile Time
                                                  ────────────
                                  param_paths     plan          wggo_overrides
                                       │            │                  │
                                       └────────────┴──────────────────┘
                                                    │
                                                    ▼
                                       build_param_mode_table()
                                                    │
                                       ┌────────────┴───────────┐
                                       │ None (no overrides)    │ Some(modes)
                                       │                        │
                                       │                        ▼
                                       │           emit_param_mode_table_rodata()
                                       │                        │
                                       │                        ▼
                                       │           DataId → GlobalValue → base ptr
                                       │                        │
                                       └────────────┬───────────┘
                                                    │
                                       Option<mode_table_base: Value>
                                                    │
   ┌────────────────────────────────────────────────┴───────────┐
   ▼                              ▼                              ▼
ga_body                        pa_body                 optimizer_step loop
(accumulation)                 (Phase A clip)
   │                              │                              │
   │  if mode_table_base.is_some():                              │
   │     emit_fase_mode_branch(table, gai, deferred, fullbuf)   │
   │     deferred_blk:  fase_emit_accumulate(...)               │
   │     fullbuf_blk:   nsl_grad_accumulate_add(...)            │
   │     join_blk:      tensor_free(grad)                       │
   │                                                             │
   │  else:                                                      │
   │     today's monolithic branch (byte-identical)             │
   ▼                                                             ▼
loop tail                                                   loop tail


                                                  Run Time
                                                  ────────
                                       gai = 0  →  modes[0] = 1 → Deferred
                                       gai = 1  →  modes[1] = 1 → Deferred
                                       gai = 2  →  modes[2] = 2 → FullBuffer
                                       gai = 3  →  modes[3] = 2 → FullBuffer
                                       (clusters by layer; well-predicted)
```

## 6. Testing

### 6.1 `build_param_mode_table` unit tests (in `fase_codegen_table.rs`)

1. **`returns_none_when_plan_has_no_per_layer_mode`** — `plan.per_layer_mode = None` → `None`, regardless of overrides.
2. **`returns_none_when_overrides_absent`** — `plan.per_layer_mode = Some(...)` but `overrides = None` → `None`.
3. **`maps_param_paths_via_layer_prefix`** — 4 layers, 3 params each. Per-layer modes `[Deferred, FullBuffer, Deferred, FullBuffer]`. Param paths `["blocks.0.wq", "blocks.0.wk", "blocks.0.wv", "blocks.1.wq", ...]`. Assert resulting `Vec<u8>` is `[1,1,1, 2,2,2, 1,1,1, 2,2,2]`.
4. **`unmatched_param_falls_back_to_global_mode`** — A param path that doesn't match any layer (e.g. `"embedding"` when WGGO only knows `blocks.*`) takes `plan.mode`. Assert it gets the global encoding.
5. **`from_fase_mode_round_trips`** — `ParamMode::from(FaseMode::Deferred) as u8 == 1`, etc.

### 6.2 Codegen integration test (in `crates/nsl-codegen/tests/fase_codegen_phase2.rs`, new)

6. **`fallback_path_emits_no_mode_table`** — Compile a fixture without WGGO. Assert no `nsl_fase_param_modes_*` symbol appears in the emitted object (use the existing object-inspection helpers, or grep the disassembled IR text).

### 6.3 End-to-end regression

7. **All existing FASE e2e tests pass unchanged.** They don't supply WGGO overrides, so they take the fallback path and produce byte-identical IR.

### 6.4 Manual verification (in PR test plan)

- Compile a small `@train` fixture with WGGO active + mixed `fase_fused` recommendations. Inspect the emitted `.rodata` symbol with `objdump -s` (or equivalent on Windows: `dumpbin /RAWDATA`) and confirm the byte pattern matches the expected mode array. Also confirm the binary runs without crashing under a smoke training step.

**Total: 6 new tests + manual verification.**

## 7. Risks & Open Questions

- **Risk: `enumerate_model_tensor_paths` returns names that don't follow WGGO's `blocks.N.*` convention.** WGGO matches by prefix with `.` boundary. If the model's param paths use different names (e.g. `model.layers.N.attn.q_proj.weight` vs WGGO's `blocks.N`), `find_by_layer_containing` returns `None` and the param falls back to global mode. **Mitigation:** during implementation, check what `enumerate_model_tensor_paths` actually emits for a typical fixture; if the convention differs, document it as a known scope limit (overrides simply won't take effect on that model) and make the fallback explicit in test 4.
- **Risk: `ParamMode = 2 (FullBuffer)` collapses to the non-Deferred branch via `icmp.eq mode, 1`.** This is intentional — FullBuffer and Passthrough share the runtime path today. If a future refactor needs them distinguished at runtime, the table is already encoded with separate values; only the dispatch helper changes.
- **Risk: `MemFlags::trusted()` may be overly aggressive.** `trusted()` asserts no aliasing and aligned access. The mode table is `.rodata` (immutable, read-only), so `trusted()` is correct here, but verify against existing rodata-load patterns in the codebase during implementation.
- **Open: which Cranelift `Block`-creation pattern do the existing FASE loops use?** The plan's pseudocode above creates new blocks per dispatch. Existing FASE-related two-phase-clip code in `stmt.rs` already creates many blocks; mirroring its style at each of the 3 sites is the implementation's job. Defer the exact block-creation idiom to the implementation plan after reading the surrounding code.

## 8. Success Criteria

1. `build_param_mode_table(paths, plan, None)` returns `None`. So does the `plan.per_layer_mode.is_none()` case.
2. `build_param_mode_table(paths, plan_with_overrides_result, Some(overrides))` returns `Some(Vec<u8>)` of correct length and contents.
3. Compiling a fixture WITHOUT WGGO produces byte-identical Cranelift IR to today (regression-tested by all existing FASE e2e tests).
4. Compiling a fixture WITH mixed WGGO overrides emits the `nsl_fase_param_modes_<func>` `.rodata` symbol with the expected bytes, and the three backward loops contain the runtime mode dispatch.
5. The compiled binary runs a training step without crashing when modes are mixed.
6. All 6 new tests pass; no existing tests regress.
7. Memory file `project_wggo_consumers.md` updated to mark Consumer 3 fully shipped (planner + observability + codegen).

## 9. Files Touched

- `crates/nsl-codegen/src/fase_codegen_table.rs` (new) — `build_param_mode_table` + `ParamMode` enum + 5 unit tests.
- `crates/nsl-codegen/src/lib.rs` — `pub mod fase_codegen_table;`.
- `crates/nsl-codegen/src/compiler/mod.rs` — `emit_param_mode_table_rodata` helper.
- `crates/nsl-codegen/src/stmt.rs` — `emit_fase_mode_branch` helper; three loop call sites updated; `mode_table_base` plumbing at the top of `compile_train_block`; replace the `TODO(fase-consumer-3-phase-2)` comment block with a "Phase 2 shipped" comment.
- `crates/nsl-codegen/tests/fase_codegen_phase2.rs` (new) — 1 codegen-level integration test.
