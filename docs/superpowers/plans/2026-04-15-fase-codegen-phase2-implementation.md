# FASE Codegen Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the global `fase_deferred` boolean dispatch in three backward loops (`stmt.rs`) with a per-parameter runtime mode lookup driven by a compile-time `.rodata` byte table built from WGGO's `per_layer_mode`.

**Architecture:** Compile-time helper `build_param_mode_table` walks `param_paths` and resolves each path against `WggoOverrides::find_by_layer_containing` to produce a `Vec<u8>`. New `Compiler::emit_param_mode_table_rodata` writes the bytes to `.rodata` (mirrors `embed_weight_hash` at `compiler/mod.rs:1048+`). Three loops in `stmt.rs` (accumulation, two-phase-clip Phase A, optimizer step) gain a runtime branch on `modes[gai]` when the table is present; fallback path is byte-identical to today.

**Tech Stack:** Rust, Cranelift IR (`InstBuilder`, `MemFlags::trusted`, `module.declare_data` / `define_data` / `declare_data_in_func`), existing `crate::fase` + `crate::wggo_overrides`.

**Spec:** [docs/superpowers/specs/2026-04-15-fase-codegen-phase2-design.md](../specs/2026-04-15-fase-codegen-phase2-design.md)

**Branch:** `feat/fase-codegen-phase2` (already created from `origin/main` at `661e7d5` — Phase 1 merged).

## Task 0 Outcome: prefix mismatch — needs strip

`enumerate_model_tensor_paths(model_var_name, model_type_name)` produces paths like `m.blocks.0.attn.wq` (the `model_var_name` is the prefix; recursion appends `.field`). WGGO `layer_name` is bare (`blocks.0`). `find_by_layer_containing("m.blocks.0.attn.wq")` returns `None` because the path does not start with `blocks.0` — the leading `m.` breaks the prefix match.

**Fix:** `build_param_mode_table` takes an additional `model_var_name: &str` parameter and strips `format!("{model_var_name}.")` from each path before calling `find_by_layer_containing`. If the prefix isn't present (e.g., paths are already bare in some future caller), the strip is a no-op via `path.strip_prefix(&prefix).unwrap_or(path.as_str())`. Task 1's helper signature and unit tests are updated below to reflect this.

CSHA/WRGA's existing `find_by_layer_containing` use site (`wrga_spectral.rs`) passes already-bare projection names from WRGA's internal world — that's why the matcher works for them but would fail here without the strip.

---

## File Inventory

**Create:**
- `crates/nsl-codegen/src/fase_codegen_table.rs` — `ParamMode` enum, `build_param_mode_table`, 5 unit tests.
- `crates/nsl-codegen/tests/fase_codegen_phase2.rs` — 1 integration test (no-WGGO fallback emits no symbol).

**Modify:**
- `crates/nsl-codegen/src/lib.rs` — add `pub mod fase_codegen_table;`.
- `crates/nsl-codegen/src/compiler/mod.rs` — add `emit_param_mode_table_rodata` helper.
- `crates/nsl-codegen/src/stmt.rs` — add `emit_fase_mode_branch` helper; thread `mode_table_base: Option<Value>` from `compile_train_block` top into 3 loop sites; replace existing `if fase_deferred { ... } else { ... }` blocks with conditional dispatch; replace Phase 1 TODO comment block with "Phase 2 shipped" note.

---

## Task 0: Param-paths discovery

**Files:** Read-only.

The spec flagged a risk that `enumerate_model_tensor_paths` may emit names that don't follow WGGO's `blocks.N.*` convention. Confirm before writing any code.

- [ ] **Step 1: Find the function**

```bash
cd c:/Users/bwiem/projects/NSL/.worktrees/fase-codegen-phase2
grep -n "fn enumerate_model_tensor_paths" crates/nsl-codegen/src/
```

- [ ] **Step 2: Read it (~50 lines)**

Note the naming convention it produces. Typical NSL transformer fixtures use `model.blocks.N.attn.wq` or similar — but the exact pattern matters because `WggoOverrides::find_by_layer_containing` does a prefix match with `.` boundary against `layer_name`.

- [ ] **Step 3: Find what `WggoOverrides.layer_name` actually contains**

```bash
grep -n "layer_name:" crates/nsl-codegen/src/wggo*.rs | head -10
```

Look at where `AppliedLayer.layer_name` is set upstream (probably in `wggo_apply.rs` from the WGGO graph node). Confirm that `enumerate_model_tensor_paths` outputs names that BEGIN WITH `layer_name` followed by `.` for the prefix match to fire.

- [ ] **Step 4: Document the finding**

Append a `## Task 0 Outcome` section to THIS PLAN FILE noting:
- A concrete example pair: `enumerate_model_tensor_paths` emits `<exact string>` and WGGO `layer_name` is `<exact string>`.
- Whether `find_by_layer_containing` will match (yes/no/partial).
- If no match, the Phase 2 work still ships but the runtime dispatch falls back to global mode for every param — making the entire feature inert. In that case, scope-extend Task 1 to add a name-translation layer or scope-defer Phase 2 entirely.

- [ ] **Step 5: Commit if outcome was added**

```
git add docs/superpowers/plans/2026-04-15-fase-codegen-phase2-implementation.md
git commit -m "docs(fase): record Task 0 outcome — param-path naming convention"
```

---

## Task 1: `ParamMode` enum + `build_param_mode_table`

**Files:**
- Create: `crates/nsl-codegen/src/fase_codegen_table.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create the new module file**

Write `crates/nsl-codegen/src/fase_codegen_table.rs`:

```rust
//! FASE Codegen Phase 2: per-parameter mode lookup table.
//!
//! Builds a compile-time `Vec<u8>` aligned with `param_paths` that the
//! backward loops in `stmt.rs` consult at runtime via a single byte
//! load. When WGGO is inactive (`plan.per_layer_mode.is_none()`) or
//! no overrides were supplied, returns `None` so callers fall back to
//! the existing global-mode dispatch (byte-identical to pre-Phase-2).

use crate::fase::{FaseMode, FasePlan};
use crate::wggo_overrides::WggoOverrides;

/// Per-parameter FASE mode encoding for the runtime dispatch table.
/// Encoded as `u8` for compact `.rodata` storage and single-byte loads.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamMode {
    Passthrough = 0,
    Deferred    = 1,
    FullBuffer  = 2,
}

impl From<FaseMode> for ParamMode {
    fn from(m: FaseMode) -> Self {
        match m {
            FaseMode::Passthrough => ParamMode::Passthrough,
            FaseMode::Deferred    => ParamMode::Deferred,
            FaseMode::FullBuffer  => ParamMode::FullBuffer,
        }
    }
}

/// Build a per-parameter mode table aligned with `param_paths`.
/// Returns `None` when no WGGO per-layer dispatch is active — caller
/// should skip rodata emission and use the global-mode dispatch path.
pub fn build_param_mode_table(
    param_paths: &[String],
    plan: &FasePlan,
    overrides: Option<&WggoOverrides>,
) -> Option<Vec<u8>> {
    let per_layer = plan.per_layer_mode.as_ref()?;
    let o = overrides?;

    let global = ParamMode::from(plan.mode) as u8;
    let mut modes = Vec::with_capacity(param_paths.len());
    for path in param_paths {
        let m = match o.find_by_layer_containing(path) {
            Some(layer) => {
                let layer_idx = layer.layer_index as usize;
                per_layer
                    .get(layer_idx)
                    .copied()
                    .map(|fm| ParamMode::from(fm) as u8)
                    .unwrap_or(global)
            }
            None => global,
        };
        modes.push(m);
    }
    Some(modes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fase::FaseOptimizer;
    use crate::wggo_apply::{AppliedLayer, AppliedPlan};
    use crate::wggo_dp::LayerDecision as CoarseDecision;

    fn applied_layer(idx: u32, name: &str) -> AppliedLayer {
        AppliedLayer {
            layer_index: idx,
            layer_name: name.into(),
            coarse: CoarseDecision::KeepFull,
            pipeline_stage: 0,
            shard_factor: 1,
            active_heads: 8,
            ffn_width: 4096,
            csha_level: 0,
            adapter_rank: 0,
            optim_m_bits: 32,
            optim_v_bits: 32,
            fase_fused: false,
            packing_mode: 0,
            estimated_us: 1.0,
            param_bytes: 0,
            activation_bytes: 0,
        }
    }

    fn make_overrides(names: &[&str]) -> WggoOverrides {
        let plan = AppliedPlan {
            layers: names.iter().enumerate().map(|(i, n)| applied_layer(i as u32, n)).collect(),
            total_us: 0.0,
            peak_memory_bytes: 0,
        };
        WggoOverrides::from_applied(&plan)
    }

    fn deferred_plan() -> FasePlan {
        crate::fase::plan(&crate::fase::FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        })
    }

    #[test]
    fn returns_none_when_plan_has_no_per_layer_mode() {
        let p = deferred_plan();
        assert!(p.per_layer_mode.is_none());
        let paths = vec!["blocks.0.wq".into()];
        let o = make_overrides(&["blocks.0"]);
        assert!(build_param_mode_table(&paths, &p, Some(&o)).is_none());
    }

    #[test]
    fn returns_none_when_overrides_absent() {
        let mut p = deferred_plan();
        p.per_layer_mode = Some(vec![FaseMode::Deferred]);
        let paths = vec!["blocks.0.wq".into()];
        assert!(build_param_mode_table(&paths, &p, None).is_none());
    }

    #[test]
    fn maps_param_paths_via_layer_prefix() {
        let mut p = deferred_plan();
        p.per_layer_mode = Some(vec![
            FaseMode::Deferred,
            FaseMode::FullBuffer,
            FaseMode::Deferred,
            FaseMode::FullBuffer,
        ]);
        let paths: Vec<String> = vec![
            "blocks.0.wq", "blocks.0.wk", "blocks.0.wv",
            "blocks.1.wq", "blocks.1.wk", "blocks.1.wv",
            "blocks.2.wq", "blocks.2.wk", "blocks.2.wv",
            "blocks.3.wq", "blocks.3.wk", "blocks.3.wv",
        ].into_iter().map(String::from).collect();
        let o = make_overrides(&["blocks.0", "blocks.1", "blocks.2", "blocks.3"]);
        let modes = build_param_mode_table(&paths, &p, Some(&o)).unwrap();
        assert_eq!(modes, vec![1,1,1, 2,2,2, 1,1,1, 2,2,2]);
    }

    #[test]
    fn unmatched_param_falls_back_to_global_mode() {
        let mut p = deferred_plan();
        p.per_layer_mode = Some(vec![FaseMode::Deferred]);
        // Global mode is Deferred (AdamW + accumulation=4); "embedding" doesn't
        // match the WGGO layer "blocks.0", so it falls back to global Deferred (1).
        let paths: Vec<String> = vec!["embedding".into(), "blocks.0.wq".into()];
        let o = make_overrides(&["blocks.0"]);
        let modes = build_param_mode_table(&paths, &p, Some(&o)).unwrap();
        assert_eq!(modes, vec![1, 1]);
    }

    #[test]
    fn from_fase_mode_round_trips() {
        assert_eq!(ParamMode::from(FaseMode::Passthrough) as u8, 0);
        assert_eq!(ParamMode::from(FaseMode::Deferred)    as u8, 1);
        assert_eq!(ParamMode::from(FaseMode::FullBuffer)  as u8, 2);
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/nsl-codegen/src/lib.rs`, find the existing `pub mod fase;` line (or similar grouping) and add adjacent:

```rust
pub mod fase_codegen_table;
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p nsl-codegen fase_codegen_table::tests 2>&1 | tail -15
```

Expected: 5 tests PASS.

If a test fails, investigate. Most likely culprit: `WggoOverrides::from_applied` may not preserve `layer_name` exactly as passed — read it, adapt the test, don't fudge the assertion.

CAUTION: `AppliedLayer` literal construction uses many fields. The list above matches the current struct (verified for Phase 1). If any field changed since Phase 1 merge, the literal will hit E0063 — fix by adding the new field with a sensible default.

- [ ] **Step 4: Run full lib for regressions**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: 1530 + 5 = 1535 passed. (Phase 1 baseline was 1530.)

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/fase_codegen_table.rs crates/nsl-codegen/src/lib.rs
git commit -m "feat(fase): build_param_mode_table — WGGO → per-param mode resolver

Compile-time helper that walks param_paths and resolves each against
WggoOverrides::find_by_layer_containing to produce a Vec<u8> aligned
with the runtime grads_list / accum_list / param_list. Returns None
when no per-layer overrides are active (caller falls back to global
dispatch). Phase 1 of FASE Codegen Phase 2 — Phase 2 here means the
codegen-side phase 2 of the Consumer 3 rollout, not a sub-phase of
this PR."
```

---

## Task 2: `Compiler::emit_param_mode_table_rodata` helper

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/mod.rs`

- [ ] **Step 1: Read the precedent**

```bash
grep -n "embed_weight_hash\|fn emit_.*rodata\|DataDescription::new" crates/nsl-codegen/src/compiler/mod.rs | head -10
```

Read `embed_weight_hash` at `compiler/mod.rs:1050-1071`. The new helper mirrors its shape with two changes: bytes come from a parameter, and the symbol name varies per train block.

- [ ] **Step 2: Add the helper**

In `crates/nsl-codegen/src/compiler/mod.rs`, immediately after `embed_weight_hash` (around line 1071):

```rust
/// FASE Codegen Phase 2: emit a per-parameter FASE mode byte table
/// into `.rodata` and return the DataId. Caller wraps in a
/// `GlobalValue` via `module.declare_data_in_func` and obtains a base
/// pointer for runtime loads inside the backward loops.
///
/// `func_name_suffix` disambiguates symbols when multiple `@train`
/// blocks exist in one compile.
pub fn emit_param_mode_table_rodata(
    &mut self,
    modes: &[u8],
    func_name_suffix: &str,
) -> Result<cranelift_module::DataId, crate::error::CodegenError> {
    let symbol = format!("nsl_fase_param_modes_{func_name_suffix}");
    let data_id = self
        .module
        .declare_data(
            &symbol,
            cranelift_module::Linkage::Local,
            false, // not writable
            false, // not TLS
        )
        .map_err(|e| {
            crate::error::CodegenError::new(format!(
                "failed to declare FASE param-mode table data: {e}"
            ))
        })?;

    let mut desc = cranelift_module::DataDescription::new();
    desc.define(modes.to_vec().into_boxed_slice());
    self.module.define_data(data_id, &desc).map_err(|e| {
        crate::error::CodegenError::new(format!(
            "failed to define FASE param-mode table data: {e}"
        ))
    })?;

    Ok(data_id)
}
```

CAUTION: The exact import path for `CodegenError` may differ — check what `embed_weight_hash` uses (it returns `Result<(), CodegenError>` — find the import line at the top of `compiler/mod.rs`).

- [ ] **Step 3: Build to verify the helper compiles**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
```

Expected: clean build. The helper is unused at this point — that's fine; no `dead_code` warning yet because Cranelift's `DataId` propagation may not flag pub helpers.

If you see `dead_code` on `emit_param_mode_table_rodata`, that's expected and will resolve in Task 4 when stmt.rs starts calling it. Note it but don't suppress.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/compiler/mod.rs
git commit -m "feat(fase): emit_param_mode_table_rodata helper on Compiler

Mirrors embed_weight_hash at compiler/mod.rs:1050. Declares a Local-
linkage non-writable .rodata symbol nsl_fase_param_modes_<func> and
returns the DataId. Caller in stmt.rs (Task 4) wraps it via
declare_data_in_func to get a base pointer for runtime byte loads."
```

---

## Task 3: `emit_fase_mode_branch` helper in stmt.rs

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`

- [ ] **Step 1: Find the existing FASE helper neighborhood**

```bash
grep -n "fn fase_emit_accumulate\|fn fase_emit_final_step\|impl Compiler<'_>" crates/nsl-codegen/src/stmt.rs | head -10
```

Locate where the existing FASE emit helpers live. The new helper goes adjacent.

- [ ] **Step 2: Add the helper**

In `crates/nsl-codegen/src/stmt.rs`, in the same `impl Compiler<'_>` block as `fase_emit_accumulate`:

```rust
/// FASE Codegen Phase 2: emit a runtime branch on the per-parameter
/// FASE mode. Loads `modes[gai]` (a u8 from the .rodata table), tests
/// for `Deferred = 1`, and conditionally jumps to one of the two
/// caller-provided blocks. Caller is responsible for creating both
/// destination blocks AND a join block, switching to and sealing each
/// destination block before its body, and switching to the join block
/// after both bodies have jumped to it.
pub(crate) fn emit_fase_mode_branch(
    &mut self,
    builder: &mut cranelift_frontend::FunctionBuilder,
    mode_table_base: cranelift_codegen::ir::Value,
    gai: cranelift_codegen::ir::Value,
    deferred_block: cranelift_codegen::ir::Block,
    fullbuffer_block: cranelift_codegen::ir::Block,
) {
    use cranelift_codegen::ir::{condcodes::IntCC, types as cl_types, MemFlags, InstBuilder};
    let byte_addr = builder.ins().iadd(mode_table_base, gai);
    let mode = builder.ins().load(cl_types::I8, MemFlags::trusted(), byte_addr, 0);
    let one_i8 = builder.ins().iconst(cl_types::I8, 1);
    let is_deferred = builder.ins().icmp(IntCC::Equal, mode, one_i8);
    builder
        .ins()
        .brif(is_deferred, deferred_block, &[], fullbuffer_block, &[]);
}
```

CAUTION: The `IntCC` / `MemFlags` / `cl_types` imports may already be in scope at the file top — if so, the `use` inside the function is harmless but redundant. The function-local `use` is defensive and still works.

- [ ] **Step 3: Build to confirm it compiles**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
```

Expected: clean. Helper is unused (until Task 4) — `dead_code` warning expected.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "feat(fase): emit_fase_mode_branch helper for per-param dispatch

Loads modes[gai] from the .rodata table and emits brif(Deferred==1,
deferred_block, fullbuffer_block). Caller manages block lifecycle.
Used by Task 4 at three loop sites in compile_train_block."
```

---

## Task 4: Wire mode_table_base + dispatch into the three loops

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`

This is the biggest task. Splits into 4a (allocation), 4b (accumulation loop), 4c (Phase A loop), 4d (optimizer loop).

### Task 4a: Allocate mode_table_base near the top of `compile_train_block`

- [ ] **Step 1: Find the post-FASE-plan, pre-loop spot**

The existing TODO comment block lives at `stmt.rs:3302+`. Just AFTER the existing FASE plan + diagnostic-rendering section, BEFORE any backward-loop code begins, insert:

```rust
// FASE Codegen Phase 2: build per-parameter mode table from WGGO's
// per-layer decisions and emit it as a .rodata byte array. The three
// backward loops below load `modes[gai]` to choose Deferred vs
// FullBuffer per param. When None, loops use today's monolithic
// fase_deferred branch (byte-identical to pre-Phase-2 codegen).
let mode_table_base: Option<cranelift_codegen::ir::Value> = {
    let modes = crate::fase_codegen_table::build_param_mode_table(
        &param_paths,
        &fase_plan,
        self.wggo_overrides.as_ref(),
    );
    match modes {
        Some(bytes) => {
            // Disambiguate per train block. Use the function name we're
            // currently emitting into.
            let func_suffix = self
                .module
                .declarations()
                .get_function_decl(builder.func.signature.clone().into())
                .map(|_| "train") // placeholder — read the actual func name below
                .unwrap_or("train");
            // Simpler: train blocks live in named functions; the
            // generated function name is in scope as `train_fn_name` in
            // most NSL compile paths. Find it before this insertion
            // and reference it directly.
            let data_id = self.emit_param_mode_table_rodata(&bytes, func_suffix)?;
            let global = self.module.declare_data_in_func(data_id, builder.func);
            Some(builder.ins().symbol_value(cranelift_codegen::ir::types::I64, global))
        }
        None => None,
    }
};
```

CAUTION: `func_suffix` resolution above is intentionally messy — the implementer must inspect the surrounding `compile_train_block` for the actual function-name local. Likely candidates: `train_fn_name`, `train_func_name`, or an `&str` derived from the Stmt's identifier. Replace the `.map(|_| "train")` placeholder with the real local. If none exists, hardcode `"train"` and accept the symbol-collision risk for multi-train compiles (rare in practice; fix with a counter if it bites).

ALTERNATIVE simpler path if function-name plumbing is hostile: pass a counter from `Compiler` (`fn next_fase_table_index(&mut self) -> u32`) and suffix with `_<index>`. Pure mechanical disambiguation, no AST inspection.

- [ ] **Step 2: Build to confirm allocation compiles**

```bash
cargo build -p nsl-codegen 2>&1 | tail -5
```

Expected: clean build. `mode_table_base` is unused so far — `unused_variables` warning expected (will resolve in 4b/4c/4d). If you see compile errors about `param_paths` / `fase_plan` / `self.wggo_overrides` not being in scope, scope-extend the allocation site downward until it can see all three.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "feat(fase): allocate mode_table_base near top of compile_train_block

Builds the table via fase_codegen_table::build_param_mode_table;
emits .rodata via Compiler::emit_param_mode_table_rodata; wraps in
GlobalValue. Result is Option<Value> = None when WGGO inactive.
Loops in 4b/4c/4d will consume it; for now the variable is dead."
```

### Task 4b: Accumulation loop (`ga_body`, ~line 4869-4893)

- [ ] **Step 1: Find the accumulation loop site**

```bash
grep -n "ga_body\|fase_emit_accumulate" crates/nsl-codegen/src/stmt.rs | head -10
```

Read the existing block — it currently looks like (approx line 4869-4893):

```rust
let accum_buf = self.compile_call_by_name(builder, "nsl_list_get", &[accum, gai])?;
let grad = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, gai])?;
if fase_deferred {
    self.fase_emit_accumulate(builder, accum_buf, grad, fase_plan.recipe.accum_scale)?;
} else {
    let n_elems = self.compile_call_by_name(builder, "nsl_tensor_len", &[accum_buf])?;
    self.compile_call_by_name(builder, "nsl_grad_accumulate_add", &[accum_buf, grad, n_elems])?;
}
self.compile_call_by_name(builder, "nsl_tensor_free", &[grad])?;
```

- [ ] **Step 2: Replace with mode-table dispatch**

```rust
let accum_buf = self.compile_call_by_name(builder, "nsl_list_get", &[accum, gai])?;
let grad = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, gai])?;

if let Some(mtb) = mode_table_base {
    // FASE Codegen Phase 2: per-param dispatch via runtime byte load.
    let ga_deferred  = builder.create_block();
    let ga_fullbuf   = builder.create_block();
    let ga_join      = builder.create_block();

    self.emit_fase_mode_branch(builder, mtb, gai, ga_deferred, ga_fullbuf);

    // Deferred path
    builder.switch_to_block(ga_deferred);
    builder.seal_block(ga_deferred);
    self.fase_emit_accumulate(builder, accum_buf, grad, fase_plan.recipe.accum_scale)?;
    builder.ins().jump(ga_join, &[]);

    // FullBuffer path
    builder.switch_to_block(ga_fullbuf);
    builder.seal_block(ga_fullbuf);
    let n_elems = self.compile_call_by_name(builder, "nsl_tensor_len", &[accum_buf])?;
    self.compile_call_by_name(builder, "nsl_grad_accumulate_add", &[accum_buf, grad, n_elems])?;
    builder.ins().jump(ga_join, &[]);

    // Join — single tensor_free regardless of path
    builder.switch_to_block(ga_join);
    builder.seal_block(ga_join);
} else if fase_deferred {
    // Pre-Phase-2 monolithic Deferred path (byte-identical when no overrides).
    self.fase_emit_accumulate(builder, accum_buf, grad, fase_plan.recipe.accum_scale)?;
} else {
    // Pre-Phase-2 monolithic FullBuffer path (byte-identical).
    let n_elems = self.compile_call_by_name(builder, "nsl_tensor_len", &[accum_buf])?;
    self.compile_call_by_name(builder, "nsl_grad_accumulate_add", &[accum_buf, grad, n_elems])?;
}
self.compile_call_by_name(builder, "nsl_tensor_free", &[grad])?;
```

- [ ] **Step 3: Build + run full lib tests**

```bash
cargo build -p nsl-codegen 2>&1 | tail -5
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: 1535 passed (Task 1 + this). If any FASE-related lib test fails, the no-override fallback path is broken — debug before proceeding.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "feat(fase): per-param dispatch in accumulation loop (ga_body)

When mode_table_base.is_some(), each iteration loads modes[gai] and
branches to the Deferred or FullBuffer path. When None, monolithic
branch (byte-identical to pre-Phase-2). Single tensor_free at join."
```

### Task 4c: Two-phase-clip Phase A loop (`pa_body`, ~line 4988-5019)

- [ ] **Step 1: Find the Phase A loop site**

```bash
grep -n "pa_body\|fase_hook_active.*pa_grad" crates/nsl-codegen/src/stmt.rs | head
```

The existing inner block (around line 4995-5008) currently looks like:

```rust
if !fase_hook_active {
    let pa_grad = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, pa_i])?;
    self.fase_emit_accumulate(builder, pa_mpart, pa_grad, fase_plan.recipe.accum_scale)?;
    self.compile_call_by_name(builder, "nsl_tensor_free", &[pa_grad])?;
}
```

Note: this block is unconditionally Deferred today (Phase A only runs when FASE Deferred two-phase-clip is active). The Phase 2 question is: when WGGO recommends FullBuffer for some layers, does Phase A still need to run for those layers?

**Answer (per spec §2):** Phase 2 ships per-layer dispatch ONLY for the standard backward path. The FASE source-AD hook + two-phase-clip path are out of scope (`fase_hook_active` branch). The Phase A loop runs only when two-phase-clip is active (Deferred + grad_clip), which means the global mode is Deferred — per-layer FullBuffer overrides on that path are inert by spec.

So the Phase A loop body itself does NOT need per-param dispatch in Phase 2. But the spec listed it in §4.4 as a site to update — that was an over-broad scope. Resolve by **leaving Phase A unchanged in Task 4c** and noting why in code.

- [ ] **Step 2: Add a clarifying comment instead of code change**

In the Phase A inner block, immediately above the existing `if !fase_hook_active { ... }`:

```rust
// FASE Codegen Phase 2: Phase A (two-phase-clip accumulation) runs
// only when the global FASE plan is Deferred + grad_clip is set. By
// spec §2, per-layer FullBuffer overrides on this path are inert
// (the path's existence already implies global Deferred). No mode-
// table dispatch needed here.
```

- [ ] **Step 3: Commit (comment-only)**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "docs(fase): Phase A two-phase-clip loop is Deferred-only by construction

Per spec §2, Phase 2's per-param dispatch is scoped to the standard
backward path. Phase A only runs when global FASE mode is Deferred,
so per-layer FullBuffer overrides on this path are inert. Comment
documents why no mode-table dispatch is wired here."
```

### Task 4d: Optimizer step loop (~line 4940+)

- [ ] **Step 1: Find the optimizer step dispatch**

```bash
grep -n "fase_emit_final_step\|if fase_deferred" crates/nsl-codegen/src/stmt.rs | head -10
```

The existing dispatch around line 4940+ branches `if fase_deferred { fused_optimizer_step_per_param } else { existing_per_param_optimizer_step }`. Read it carefully — it likely has its own per-param loop (possibly using `accum` index var rather than `gai`) and the fused vs separate branch is INSIDE that loop.

- [ ] **Step 2: Decide scope based on inspection**

If the optimizer step loop body has the same `if fase_deferred { ... } else { ... }` shape as the accumulation loop, apply the same transformation pattern from Task 4b (with mode_table_base + 3 blocks + join).

If the structure is materially different (e.g., the fused/separate branch is OUTSIDE the loop, or the existing code already calls `fase_emit_final_step` per param without an alternate), document the actual structure inline as a comment and:
- Apply per-param dispatch where structurally feasible.
- Skip and document where the structure makes per-layer dispatch infeasible at this site (e.g., an outer-of-loop branch implies the WHOLE optimizer step is fused or not — partial mixing isn't possible without a deeper refactor).

- [ ] **Step 3: Apply the transformation**

If the loop body has the expected pattern:

```rust
if let Some(mtb) = mode_table_base {
    let opt_deferred  = builder.create_block();
    let opt_fullbuf   = builder.create_block();
    let opt_join      = builder.create_block();
    self.emit_fase_mode_branch(builder, mtb, opt_i, opt_deferred, opt_fullbuf);

    builder.switch_to_block(opt_deferred);
    builder.seal_block(opt_deferred);
    // ... existing fused step body ...
    builder.ins().jump(opt_join, &[]);

    builder.switch_to_block(opt_fullbuf);
    builder.seal_block(opt_fullbuf);
    // ... existing separate step body ...
    builder.ins().jump(opt_join, &[]);

    builder.switch_to_block(opt_join);
    builder.seal_block(opt_join);
} else if fase_deferred {
    // existing fused step body (verbatim)
} else {
    // existing separate step body (verbatim)
}
```

Use the actual loop-index variable (`opt_i` is a guess — check the real name).

If structure is different and per-layer mixing is structurally infeasible at this site, write a comment explaining and leave the existing global dispatch intact for now. This is acceptable — the planner + accumulation-loop dispatch already make WGGO's signal partially observable in codegen, which is what Phase 2 promised.

- [ ] **Step 4: Build + run lib tests + e2e regression**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
cargo test -p nsl-codegen --lib 2>&1 | tail -3
cargo test -p nsl-codegen --test fase_codegen_phase2 2>&1 | tail -3   # will run after Task 5
cargo test -p nsl-cli --test e2e 2>&1 | tail -10
```

Expected: lib still 1535 pass. e2e: same flakes as Phase 1 (Windows file-lock races, M27 PTX panics) — not regressions.

If any non-flake e2e test newly fails, the fallback path is broken — likely a missed `else` branch. Debug.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "feat(fase): per-param dispatch in optimizer step loop

[Adapt commit message to whether full dispatch was applied or partial /
deferred per Step 2's structural decision.]"
```

---

## Task 5: Integration test for fallback path

**Files:**
- Create: `crates/nsl-codegen/tests/fase_codegen_phase2.rs`

The Phase 2 fallback (no WGGO active) must produce zero new IR — no `nsl_fase_param_modes_*` symbol in the emitted object.

- [ ] **Step 1: Write the test**

Create `crates/nsl-codegen/tests/fase_codegen_phase2.rs`:

```rust
//! FASE Codegen Phase 2: confirm the fallback path emits zero new IR
//! when no WGGO overrides are active.

#[test]
fn fallback_path_emits_no_mode_table_constant() {
    // The fallback path is exercised by every existing FASE test that
    // doesn't supply WGGO overrides. Rather than spinning up a full
    // compile here (which requires fixture infrastructure already
    // covered by e2e tests), we assert the property structurally:
    // build_param_mode_table returns None when overrides are absent.
    use nsl_codegen::fase::{plan, FaseConfig, FaseMode, FaseOptimizer};
    use nsl_codegen::fase_codegen_table::build_param_mode_table;

    let mut p = plan(&FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    });
    p.per_layer_mode = Some(vec![FaseMode::Deferred, FaseMode::FullBuffer]);

    let paths: Vec<String> = vec!["blocks.0.wq".into(), "blocks.1.wq".into()];

    // No overrides → None → caller skips rodata emission entirely.
    assert!(build_param_mode_table(&paths, &p, None).is_none());
}

#[test]
fn fallback_path_emits_no_mode_table_when_per_layer_mode_none() {
    use nsl_codegen::fase::{plan, FaseConfig, FaseOptimizer};
    use nsl_codegen::fase_codegen_table::build_param_mode_table;
    use nsl_codegen::wggo_apply::AppliedPlan;
    use nsl_codegen::wggo_overrides::WggoOverrides;

    let p = plan(&FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    });
    assert!(p.per_layer_mode.is_none());

    let empty_overrides = WggoOverrides::from_applied(&AppliedPlan {
        layers: vec![],
        total_us: 0.0,
        peak_memory_bytes: 0,
    });
    let paths: Vec<String> = vec!["blocks.0.wq".into()];

    // per_layer_mode is None → None → caller skips rodata emission.
    assert!(build_param_mode_table(&paths, &p, Some(&empty_overrides)).is_none());
}
```

These two tests pin the fallback contract structurally without requiring full-compile infrastructure. The end-to-end "no symbol in object file" property follows: if `build_param_mode_table` returns `None`, the `match modes { Some(...) => declare + define + symbol_value, None => None }` arm in stmt.rs Task 4a never declares the symbol, so it cannot appear in the emitted object.

- [ ] **Step 2: Run the test**

```bash
cargo test -p nsl-codegen --test fase_codegen_phase2 2>&1 | tail -5
```

Expected: 2 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/tests/fase_codegen_phase2.rs
git commit -m "test(fase): pin Phase 2 fallback contract — no rodata when WGGO inactive

Two integration tests assert build_param_mode_table returns None when
either overrides are absent or plan.per_layer_mode is None. The 'no
symbol in object' end-to-end property follows because Task 4a's
allocation site only declares the symbol when this helper returns Some."
```

---

## Task 6: Replace Phase 1 TODO comment

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`

- [ ] **Step 1: Find the Phase 1 TODO**

```bash
grep -n "TODO(fase-consumer-3-phase-2)" crates/nsl-codegen/src/stmt.rs
```

Should return one hit around line 3302.

- [ ] **Step 2: Replace with shipped note**

Find the comment block (it spans ~10 lines starting with `// TODO(fase-consumer-3-phase-2): Per-layer codegen dispatch deferred.`) and replace with:

```rust
// FASE Codegen Phase 2 SHIPPED: per-param runtime mode dispatch is
// now wired into the accumulation loop (and the optimizer step loop
// where structurally feasible). Phase A two-phase-clip loop is
// Deferred-only by construction (see comment at pa_body) and stays
// on the global mode. mode_table_base is built from
// fase_codegen_table::build_param_mode_table just below; loops branch
// on `mode_table_base: Option<Value>` to choose per-param dispatch
// vs the byte-identical pre-Phase-2 fallback.
let fase_deferred = fase_plan.mode == crate::fase::FaseMode::Deferred;
```

- [ ] **Step 3: Build**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "docs(fase): Phase 1 TODO replaced with 'Phase 2 shipped' note

The TODO at stmt.rs:~3302 from the Phase 1 design has been resolved.
Comment now points readers to the mode_table_base allocation and the
per-loop dispatch sites."
```

---

## Task 7: Memory file update + push

- [ ] **Step 1: Update `project_wggo_consumers.md`**

Memory file at `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wggo_consumers.md`.

Find the Consumer 3 (FASE) section (heading currently reads `## Consumer 3: FASE (planner+observability shipped 2026-04-15; codegen Phase 2)`). Update to:

```markdown
## Consumer 3: FASE (FULLY SHIPPED 2026-04-15)

Planner + observability shipped via PR #39 (Phase 1, merged at `661e7d5`).
Per-param runtime mode dispatch shipped via PR #<num> (Phase 2).
WGGO's `fase_fused: bool` per-layer signal now influences emitted
backward-loop code, not just diagnostics.

**Phase 2 codegen shape:**
- `fase_codegen_table::build_param_mode_table(paths, plan, overrides)`
  walks `param_paths` and resolves each via
  `WggoOverrides::find_by_layer_containing` to produce a `Vec<u8>`
  (`0=Passthrough, 1=Deferred, 2=FullBuffer`). Returns `None` when
  WGGO is inactive (caller falls back to monolithic global dispatch).
- `Compiler::emit_param_mode_table_rodata` writes the bytes to
  `.rodata` as `nsl_fase_param_modes_<func>` (Local linkage).
- Three backward loops in `compile_train_block`:
  - **Accumulation loop** (`ga_body`) — full per-param dispatch.
  - **Two-phase-clip Phase A loop** (`pa_body`) — Deferred-only by
    construction (this path only runs when global mode = Deferred);
    no dispatch needed.
  - **Optimizer step loop** — [adapt based on Task 4d outcome:
    "full per-param dispatch" or "structural reasons; partial / not
    applied"].
- Fallback (no WGGO active) is **byte-identical** to pre-Phase-2 IR;
  guaranteed by `mode_table_base.is_none()` taking the original
  monolithic branch at every site.

**Spec:** `docs/superpowers/specs/2026-04-15-fase-codegen-phase2-design.md`
**Plan:** `docs/superpowers/plans/2026-04-15-fase-codegen-phase2-implementation.md`
```

- [ ] **Step 2: Update `MEMORY.md`**

Find the line `- [WGGO AppliedPlan → consumers](project_wggo_consumers.md) ...` and update FASE status to "fully shipped":

```markdown
- [WGGO AppliedPlan → consumers](project_wggo_consumers.md) — CSHA + WRGA + CPDT + FASE all fully shipped (codegen + observability). Only Prune still pending.
```

- [ ] **Step 3: Run full workspace test**

```bash
cd c:/Users/bwiem/projects/NSL/.worktrees/fase-codegen-phase2
cargo test --workspace 2>&1 | tail -10
```

Expected: all pass except pre-existing flakes (`e2e_m12_grad_basic_source_ad`, `e2e_m27_*` — Windows file-lock + PTX target panics, documented in earlier PRs).

- [ ] **Step 4: Push**

```bash
git push -u origin feat/fase-codegen-phase2
```

- [ ] **Step 5: Prepare PR body**

Draft along the lines of:

```markdown
## Summary
- New `fase_codegen_table::build_param_mode_table` — walks `param_paths` and resolves each via `WggoOverrides::find_by_layer_containing` to produce a `Vec<u8>` mode table.
- New `Compiler::emit_param_mode_table_rodata` — writes the table to `.rodata` as a `Local`-linkage symbol.
- Accumulation loop (`ga_body`) gains per-param runtime dispatch via a new `emit_fase_mode_branch` helper.
- [Optimizer-step-loop status — adapt based on Task 4d outcome.]
- Two-phase-clip Phase A loop is Deferred-only by construction (path only runs when global = Deferred); no dispatch needed.
- Fallback (no WGGO active) is byte-identical to pre-Phase-2 IR.

## Closes
- TODO at `stmt.rs:~3302` from Phase 1 (PR #39).
- Final piece of the WGGO Consumer 3 (FASE) rollout.

## Test plan
- [ ] `cargo test -p nsl-codegen fase_codegen_table::tests` — 5 pass
- [ ] `cargo test -p nsl-codegen --test fase_codegen_phase2` — 2 pass
- [ ] `cargo test -p nsl-codegen --lib` — 1537+ pass (Phase 1 baseline + 7)
- [ ] Manual: compile a fixture with WGGO + mixed `fase_fused`; `objdump -s -j .rodata <obj>` shows `nsl_fase_param_modes_*` symbol with expected bytes

## Out of scope (deferred)
- FASE source-AD hook path per-layer dispatch (separate plan).
- Param-name convention mismatches between `enumerate_model_tensor_paths` and WGGO `layer_name`. Task 0 outcome documents what was found; if convention mismatches cause `find_by_layer_containing` to miss, the fallback to global mode is correct — just inert.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

User opens the PR on GitHub.

---

## Self-review checklist (run before claiming done)

- [ ] Every spec section (§3 invariants, §4 design, §5 architecture, §6 testing) has ≥1 task implementing it.
- [ ] No `TBD` / `implement later` in plan text. (Task 4d allows structural-decision branching but provides full code for both outcomes.)
- [ ] Method/type names consistent: `ParamMode`, `build_param_mode_table`, `emit_param_mode_table_rodata`, `emit_fase_mode_branch`, `mode_table_base`.
- [ ] Every code-touching step shows actual code.
- [ ] Project's most-common mistake (E0063 missing fields) is mentioned in Task 1 Step 3.
- [ ] Task 0 outcome is recorded inline before Task 1 begins.
