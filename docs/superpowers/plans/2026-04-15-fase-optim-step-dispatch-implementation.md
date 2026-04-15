# FASE Optimizer-Step Per-Param Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive FASE's optimizer-step codegen from the same `.rodata` mode table Phase 2 emits for the accumulation loop. Each param's optimizer path reads the accumulator convention that matches its accumulation path, closing Phase 2's latent mixed-mode mismatch.

**Architecture:** (1) Extend `plan_with_overrides` with a two-phase-clip clamp + new `TwoPhaseClipConflict` diagnostic. (2) Extract the stdlib `<optim>_step` dispatch at `stmt.rs:5339-5439` into a reusable `emit_stdlib_optim_call` helper. (3) Add `emit_unified_optim_step_dispatch` that emits one runtime loop with per-iteration mode branch, calling `fase_emit_final_step` for Deferred params and `emit_stdlib_optim_call` for FullBuffer params. (4) Gate: use the unified path when `mode_table_base.is_some()`, preserve the monolithic fallback otherwise.

**Tech Stack:** Rust, Cranelift IR (`InstBuilder`, `Block`, `brif`), existing `crate::fase`, `crate::wggo_overrides`, `crate::stmt_fase`.

**Spec:** [docs/superpowers/specs/2026-04-15-fase-optim-step-dispatch-design.md](../specs/2026-04-15-fase-optim-step-dispatch-design.md)

**Branch:** `feat/fase-optim-step-dispatch` (already created from `origin/main` at `aa0a0c5` — Phase 2 merged).

---

## File Inventory

**Create:**
- `crates/nsl-codegen/tests/fase_optim_step_dispatch.rs` — 1 integration test.

**Modify:**
- `crates/nsl-codegen/src/wggo_overrides.rs` — add `TwoPhaseClipConflict { grad_clip_threshold: f64 }` variant + round-trip test.
- `crates/nsl-codegen/src/fase.rs` — extend `plan_with_overrides` with two-phase-clip clamp rule; 4 new unit tests.
- `crates/nsl-codegen/src/stmt_fase.rs` — add `emit_stdlib_optim_call` and `emit_unified_optim_step_dispatch` helpers.
- `crates/nsl-codegen/src/stmt.rs` — gate the optimizer step on `mode_table_base`; call the new helpers when `Some`; extend the stderr renderer with `TwoPhaseClipConflict`.

---

## Task 1: `TwoPhaseClipConflict` diagnostic variant

**Files:**
- Modify: `crates/nsl-codegen/src/wggo_overrides.rs`

- [ ] **Step 1: Read the existing enum**

```bash
cd c:/Users/bwiem/projects/NSL/.worktrees/fase-optim-step-dispatch
grep -n "pub enum OverrideRejectReason\|FaseModeInfeasible" crates/nsl-codegen/src/wggo_overrides.rs
```

Confirm `OverrideRejectReason` is a struct-variant-style enum; the existing `FaseModeInfeasible { optimizer, global_mode }` is the precedent.

- [ ] **Step 2: Write the failing test**

Add inside the `#[cfg(test)] mod tests`:

```rust
#[test]
fn two_phase_clip_conflict_round_trips_debug() {
    let r = OverrideRejectReason::TwoPhaseClipConflict {
        grad_clip_threshold: 1.5,
    };
    let s = format!("{:?}", r);
    assert!(s.contains("TwoPhaseClipConflict"));
    assert!(s.contains("1.5"));
}
```

- [ ] **Step 3: Run to confirm failure**

```bash
cargo test -p nsl-codegen wggo_overrides::tests::two_phase_clip_conflict_round_trips_debug
```

Expected: FAIL — "no variant named `TwoPhaseClipConflict`".

- [ ] **Step 4: Add the variant**

In `pub enum OverrideRejectReason`:

```rust
/// Two-phase-clip + mixed mode conflict. When `grad_clip` is set and
/// accumulation > 1 and global FASE plan is Deferred, Phase A's global
/// ||m_partial||² norm requires uniform accumulation convention across
/// params. WGGO requests of `fase_fused=false` for individual layers
/// are clamped back to Deferred to preserve the norm's validity.
TwoPhaseClipConflict {
    grad_clip_threshold: f64,
},
```

- [ ] **Step 5: Fix exhaustive match sites**

```bash
grep -rn "OverrideRejectReason::" crates/nsl-codegen/src/ | grep -v "tests\|mod tests" | head -15
```

For any `match`/`if let` site that exhaustively matches this enum (no wildcard `_` arm), add:

```rust
OverrideRejectReason::TwoPhaseClipConflict { .. } => /* fallthrough or format!("{:?}", ...) */,
```

The primary renderer in `stmt.rs` already has an `other => format!("{:?}", other)` fallback and will auto-handle the new variant — the full custom reason string is added in Task 5.

- [ ] **Step 6: Verify build + test**

```bash
cargo test -p nsl-codegen wggo_overrides::tests::two_phase_clip_conflict_round_trips_debug
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: new test PASS; full lib suite unchanged count +1.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/wggo_overrides.rs
git commit -m "feat(wggo): TwoPhaseClipConflict override reject reason

Carries grad_clip_threshold so consumer code can render a precise
stderr diagnostic when WGGO's mixed-mode request is clamped to
uniform Deferred by the two-phase-clip correctness rule."
```

---

## Task 2: Two-phase-clip clamp in `plan_with_overrides`

**Files:**
- Modify: `crates/nsl-codegen/src/fase.rs`

- [ ] **Step 1: Locate `plan_with_overrides`**

```bash
grep -n "pub fn plan_with_overrides" crates/nsl-codegen/src/fase.rs
```

Read the existing function (it should already have feasibility clamps for Lion / Unknown / `allow_v_approx=false`).

- [ ] **Step 2: Write the 4 failing tests**

Add to `fase.rs` `#[cfg(test)] mod tests`:

```rust
#[test]
fn two_phase_clip_with_mixed_modes_clamps_all_to_deferred() {
    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        grad_clip: Some(1.0),
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, false, true, false]);
    assert_eq!(p.mode, FaseMode::Deferred);
    // All clamped to Deferred.
    assert_eq!(
        p.per_layer_mode,
        Some(vec![FaseMode::Deferred, FaseMode::Deferred, FaseMode::Deferred, FaseMode::Deferred])
    );
    // Two diagnostics (one per false input).
    assert_eq!(p.override_diagnostics.len(), 2);
    let layer_indices: Vec<u32> = p.override_diagnostics.iter().map(|d| d.layer_index).collect();
    assert_eq!(layer_indices, vec![1, 3]);
    for d in &p.override_diagnostics {
        assert_eq!(d.requested, "FullBuffer");
        assert_eq!(d.applied, "Deferred");
        assert!(matches!(
            d.reason,
            crate::wggo_overrides::OverrideRejectReason::TwoPhaseClipConflict {
                grad_clip_threshold: 1.0,
            }
        ));
    }
}

#[test]
fn two_phase_clip_all_deferred_no_clamp_fires() {
    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        grad_clip: Some(1.0),
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, true, true, true]);
    assert_eq!(p.mode, FaseMode::Deferred);
    assert_eq!(p.per_layer_mode, Some(vec![FaseMode::Deferred; 4]));
    assert!(p.override_diagnostics.is_empty());
}

#[test]
fn no_clip_mixed_modes_preserved() {
    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        grad_clip: None,
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, false, true, false]);
    assert_eq!(
        p.per_layer_mode,
        Some(vec![FaseMode::Deferred, FaseMode::FullBuffer, FaseMode::Deferred, FaseMode::FullBuffer])
    );
    assert!(p.override_diagnostics.is_empty());
}

#[test]
fn two_phase_clip_diagnostic_carries_threshold() {
    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        grad_clip: Some(2.5),
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[false]);
    assert_eq!(p.override_diagnostics.len(), 1);
    assert!(matches!(
        p.override_diagnostics[0].reason,
        crate::wggo_overrides::OverrideRejectReason::TwoPhaseClipConflict {
            grad_clip_threshold: t,
        } if (t - 2.5).abs() < 1e-12
    ));
}
```

- [ ] **Step 3: Run to confirm failures**

```bash
cargo test -p nsl-codegen fase::tests::two_phase_clip 2>&1 | tail -15
```

Expected: tests 1 and 4 FAIL (wrong mode / no diagnostic). Tests 2 and 3 may already pass if existing behavior happens to match.

- [ ] **Step 4: Implement the clamp**

Modify `plan_with_overrides` in `crates/nsl-codegen/src/fase.rs`. Find the early-return for empty input AFTER the base `plan(cfg)` call:

```rust
pub fn plan_with_overrides(cfg: &FaseConfig, wggo_fused_per_layer: &[bool]) -> FasePlan {
    let mut p = plan(cfg);
    if wggo_fused_per_layer.is_empty() {
        return p;
    }

    // NEW: Two-phase-clip + mixed modes → clamp to uniform Deferred.
    // Runs BEFORE the existing per-layer feasibility clamps so the
    // global-mode feasibility check below sees the clamped inputs.
    let two_phase_clip_active = cfg.grad_clip.is_some() && cfg.accumulation > 1;
    let is_global_deferred = p.mode == FaseMode::Deferred;
    let has_any_false = wggo_fused_per_layer.iter().any(|&b| !b);

    if two_phase_clip_active && is_global_deferred && has_any_false {
        let threshold = cfg.grad_clip.expect("grad_clip checked above");
        let clamped_modes = vec![FaseMode::Deferred; wggo_fused_per_layer.len()];
        let diagnostics: Vec<crate::wggo_overrides::OverrideDiagnostic> = wggo_fused_per_layer
            .iter()
            .enumerate()
            .filter_map(|(i, &fused)| {
                if fused { return None; }
                Some(crate::wggo_overrides::OverrideDiagnostic {
                    layer_index: i as u32,
                    layer_name: format!("layer_{i}"),
                    requested: "FullBuffer".into(),
                    applied: "Deferred".into(),
                    reason: crate::wggo_overrides::OverrideRejectReason::TwoPhaseClipConflict {
                        grad_clip_threshold: threshold,
                    },
                })
            })
            .collect();
        p.per_layer_mode = Some(clamped_modes);
        p.override_diagnostics = diagnostics;
        return p;
    }

    // ... existing per-layer feasibility-clamp logic unchanged ...
}
```

IMPORTANT: Preserve whatever per-layer loop + diagnostic-accumulation code already exists in `plan_with_overrides` for the non-clip path. The new block adds a fast-return for the two-phase-clip case BEFORE the existing loop runs.

CAUTION: verify the exact `OverrideDiagnostic` field names by reading the struct at `wggo_overrides.rs`. Phase 1/2 showed: `layer_index: u32`, `layer_name: String`, `requested: String`, `applied: String`, `reason: OverrideRejectReason`.

- [ ] **Step 5: Run tests**

```bash
cargo test -p nsl-codegen fase::tests::two_phase_clip
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: all 4 new tests PASS; existing FASE tests unchanged.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/fase.rs
git commit -m "feat(fase): two-phase-clip clamp in plan_with_overrides

When grad_clip is active, accumulation > 1, and global mode is Deferred,
clamp any fase_fused=false layer to Deferred and emit TwoPhaseClipConflict
diagnostics. Preserves Phase A's global ||m_partial||² norm validity
(which requires uniform accumulation convention)."
```

---

## Task 3: Extract `emit_stdlib_optim_call` helper

**Files:**
- Modify: `crates/nsl-codegen/src/stmt_fase.rs` (add helper)
- Modify: `crates/nsl-codegen/src/stmt.rs` (replace inline match with call)

This extraction is a pure refactor — no behavior change. Both the existing monolithic path AND Task 4's new unified loop will call this helper.

- [ ] **Step 1: Read the existing match (around stmt.rs:5339-5439)**

```bash
grep -n "match optimizer_name.as_str()\|opt_fn,\|compile_call_by_name(builder, &opt_fn" crates/nsl-codegen/src/stmt.rs | head -10
```

Read the six arms (sgd, adam, adamw, lion, muon, soap) + the error arm to capture exact arg orderings.

- [ ] **Step 2: Add the helper**

In `crates/nsl-codegen/src/stmt_fase.rs`, in the same `impl Compiler<'_>` block as other FASE helpers:

```rust
/// Emit a per-parameter stdlib optimizer step call. Factored from the
/// monolithic FullBuffer optimizer-step loop in `compile_train_block`
/// so both the original loop and the unified per-param dispatch loop
/// (FASE Codegen Phase 3) can share the arg-shape logic.
///
/// Each optimizer has a distinct arg shape; the `match` arms here
/// preserve the original ordering verbatim.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_stdlib_optim_call(
    &mut self,
    builder: &mut cranelift_frontend::FunctionBuilder,
    optimizer_name: &str,
    opt_fn: &str,
    param_val: cranelift_codegen::ir::Value,
    grad_val: cranelift_codegen::ir::Value,
    s1: cranelift_codegen::ir::Value,
    s2: cranelift_codegen::ir::Value,
    lr: cranelift_codegen::ir::Value,
    momentum_const: cranelift_codegen::ir::Value,
    dampening_const: cranelift_codegen::ir::Value,
    weight_decay_const: cranelift_codegen::ir::Value,
    nesterov_const: cranelift_codegen::ir::Value,
    beta1_const: cranelift_codegen::ir::Value,
    beta2_const: cranelift_codegen::ir::Value,
    eps_const: cranelift_codegen::ir::Value,
    step_count_var: cranelift_frontend::Variable,
) -> Result<(), crate::error::CodegenError> {
    use cranelift_codegen::ir::{types as cl_types, InstBuilder};
    match optimizer_name {
        "sgd" => {
            self.compile_call_by_name(
                builder,
                opt_fn,
                &[
                    param_val, grad_val, s1,
                    lr, momentum_const, dampening_const, weight_decay_const, nesterov_const,
                ],
            )?;
        }
        "adam" | "adamw" => {
            let t_val = builder.use_var(step_count_var);
            let one = builder.ins().iconst(cl_types::I64, 1);
            let t_plus_one = builder.ins().iadd(t_val, one);
            let t_float = builder.ins().fcvt_from_sint(cl_types::F64, t_plus_one);
            self.compile_call_by_name(
                builder,
                opt_fn,
                &[
                    param_val, grad_val, s1, s2,
                    lr, beta1_const, beta2_const, eps_const, weight_decay_const, t_float,
                ],
            )?;
        }
        "lion" => {
            self.compile_call_by_name(
                builder,
                opt_fn,
                &[
                    param_val, grad_val, s1,
                    lr, beta1_const, beta2_const, weight_decay_const,
                ],
            )?;
        }
        "muon" => {
            self.compile_call_by_name(
                builder,
                opt_fn,
                &[
                    param_val, grad_val, s1,
                    lr, momentum_const, weight_decay_const, nesterov_const,
                ],
            )?;
        }
        "soap" => {
            let t_val_s = builder.use_var(step_count_var);
            let one_s = builder.ins().iconst(cl_types::I64, 1);
            let t_plus_s = builder.ins().iadd(t_val_s, one_s);
            let t_float_s = builder.ins().fcvt_from_sint(cl_types::F64, t_plus_s);
            self.compile_call_by_name(
                builder,
                opt_fn,
                &[
                    param_val, grad_val, s1, s2,
                    lr, beta1_const, beta2_const, eps_const, t_float_s,
                ],
            )?;
        }
        _ => {
            return Err(crate::error::CodegenError::new(format!(
                "unsupported optimizer '{}' in train block",
                optimizer_name
            )));
        }
    }
    Ok(())
}
```

CAUTION:
- The `s2` arg is always passed. For SGD/Lion/Muon the helper ignores it (those arms don't reference `s2`). Caller in the monolithic path can still always compute `s2` from `state_list_2` — verify by reading the monolithic code (it loads `s2` only in `"adam" | "adamw"` and `"soap"` arms, not unconditionally). To match exactly, accept `s2: Value` unconditionally and only the arms that need it use it. Callers that don't have a meaningful `s2` can pass `s1` as a placeholder — equivalent to the existing non-Adam arms.
- The imports (`cl_types`, `InstBuilder`) may already be in scope at the file top of `stmt_fase.rs`. The function-local `use` is defensive — keeps the helper portable.

- [ ] **Step 3: Replace the inline match in stmt.rs**

At `stmt.rs:5339+`, replace the `match optimizer_name.as_str() { ... }` block with a single call:

```rust
// Adam/AdamW/SOAP need s2; others ignore it. Load eagerly; the helper
// only references it in the arms that need it.
let s2 = if num_state_buffers >= 2 {
    self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, idx])?
} else {
    s1  // placeholder for non-Adam optimizers
};
self.emit_stdlib_optim_call(
    builder,
    optimizer_name.as_str(),
    &opt_fn,
    param_val, grad_val, s1, s2,
    lr, momentum_const, dampening_const, weight_decay_const, nesterov_const,
    beta1_const, beta2_const, eps_const,
    step_count_var,
)?;
```

Remove the `let s2 = ...` lines that were inline inside the individual arms — the caller now loads `s2` once before the call.

CAUTION: the original `"soap"` arm loaded `s2` inside the arm. Removing that load is fine because we hoist it above the call. The Adam/AdamW arms also loaded `s2` inside — same hoist. Make sure the unhoisted `let s2 = ...` is removed from BOTH arms' bodies.

- [ ] **Step 4: Build + run ALL tests**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: clean build; FASE test count unchanged (extraction is pure refactor).

If ANY test fails here, it likely indicates a subtle arg-order divergence during extraction — debug against the original code before proceeding. This step is the firewall.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/stmt_fase.rs crates/nsl-codegen/src/stmt.rs
git commit -m "refactor(fase): extract emit_stdlib_optim_call helper

Pure refactor — six match arms (sgd/adam/adamw/lion/muon/soap) moved
from inline in compile_train_block to a reusable method on Compiler.
Arg orderings preserved verbatim. Task 4 (unified per-param dispatch)
will call this helper from the FullBuffer iteration branch."
```

---

## Task 4: `emit_unified_optim_step_dispatch` helper

**Files:**
- Modify: `crates/nsl-codegen/src/stmt_fase.rs` (add helper)

This is the big one. The helper replaces the outer `if fase_deferred { ... } else { ... }` structure with a single loop that branches per-iteration on `modes[opt_i]`.

- [ ] **Step 1: Read the existing monolithic paths for reference**

Already done in Task 3. Additional read needed: the Phase A/B two-phase-clip emission at `stmt.rs:5042-5188` and the non-clip Deferred loop at `stmt.rs:5189-5237`.

Per spec §6, when `mode_table_base.is_some()` AND `two_phase_clip` is active, all modes are already clamped to Deferred upstream. So the unified dispatch's Deferred branch must support BOTH non-clip and clip cases, but the FullBuffer branch only sees non-clip inputs.

- [ ] **Step 2: Add the helper**

In `crates/nsl-codegen/src/stmt_fase.rs`:

```rust
/// Emit a single unified optimizer-step loop with per-param runtime
/// dispatch on the FASE mode table. Used when `mode_table_base.is_some()`.
///
/// Per spec §6, when two_phase_clip is active every mode is clamped
/// to Deferred upstream (in plan_with_overrides), so the FullBuffer
/// branch of this loop will not execute in the clip case.
///
/// Structure:
///   compute bc1_inv / bc2_inv once
///   [if two_phase_clip: run Phase A sum_sq loop, then compute clip_factor]
///   for opt_i in 0..num_params:
///     theta  = param_list[opt_i]
///     state1 = state_list_1[opt_i]
///     state2 = state_list_2[opt_i] if num_state_buffers >= 2 else state1
///     if modes[opt_i] == Deferred:
///       m_partial = accum[opt_i]
///       [if two_phase_clip: m_partial *= clip_factor]
///       fase_emit_final_step(theta, state1, m_partial, state2, recipe, Some((bc1_inv, bc2_inv)))
///     else: # FullBuffer
///       grad = opt_grads[opt_i]
///       emit_stdlib_optim_call(optimizer_name, opt_fn, theta, grad, state1, state2, ...)
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_unified_optim_step_dispatch(
    &mut self,
    builder: &mut cranelift_frontend::FunctionBuilder,
    state: &mut crate::compiler::FunctionState,
    mode_table_base: cranelift_codegen::ir::Value,
    num_params_val: cranelift_codegen::ir::Value,
    param_list: cranelift_codegen::ir::Value,
    state_list_1: cranelift_codegen::ir::Value,
    state_list_2: cranelift_codegen::ir::Value,
    num_state_buffers: usize,
    accum_list: Option<cranelift_codegen::ir::Value>,
    opt_grads: cranelift_codegen::ir::Value,
    step_count_var: cranelift_frontend::Variable,
    fase_plan: &crate::fase::FasePlan,
    optimizer_name: &str,
    opt_fn: &str,
    lr: cranelift_codegen::ir::Value,
    momentum_const: cranelift_codegen::ir::Value,
    dampening_const: cranelift_codegen::ir::Value,
    weight_decay_const: cranelift_codegen::ir::Value,
    nesterov_const: cranelift_codegen::ir::Value,
    beta1_const: cranelift_codegen::ir::Value,
    beta2_const: cranelift_codegen::ir::Value,
    eps_const: cranelift_codegen::ir::Value,
    grad_accumulation_steps: i64,
    grad_clip_threshold: f64,
) -> Result<(), crate::error::CodegenError> {
    use cranelift_codegen::ir::{condcodes::IntCC, types as cl_types, InstBuilder};

    // ── 1. Bias-correction setup (always — cheap, only used by Deferred branch)
    let sc_val = builder.use_var(step_count_var);
    let one_i64 = builder.ins().iconst(cl_types::I64, 1);
    let sc_plus_one = builder.ins().iadd(sc_val, one_i64);
    let grad_accum_const = builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
    let opt_step = builder.ins().sdiv(sc_plus_one, grad_accum_const);
    let beta1_for_bc = builder.ins().f64const(fase_plan.recipe.beta1);
    let beta2_for_bc = builder.ins().f64const(fase_plan.recipe.beta2);
    let bc1_inv = self.compile_call_by_name(
        builder, "nsl_bias_correction_inv", &[beta1_for_bc, opt_step],
    )?;
    let bc2_inv = self.compile_call_by_name(
        builder, "nsl_bias_correction_inv", &[beta2_for_bc, opt_step],
    )?;

    // ── 2. Phase A sum_sq loop + clip factor (two_phase_clip only)
    let clip_factor = if fase_plan.two_phase_clip {
        // Per §6: all modes clamped to Deferred upstream, so accum is m_partial everywhere.
        let Some(accum) = accum_list else {
            return Err(crate::error::CodegenError::new(
                "two_phase_clip requires accum_list".to_string(),
            ));
        };

        // Sum ||m_partial[i]||² across all params
        let pa_tot_var = state.new_variable();
        builder.declare_var(pa_tot_var, cl_types::F64);
        builder.def_var(pa_tot_var, builder.ins().f64const(0.0));
        let pa_i_var = state.new_variable();
        builder.declare_var(pa_i_var, cl_types::I64);
        builder.def_var(pa_i_var, builder.ins().iconst(cl_types::I64, 0));

        let pa_hdr = builder.create_block();
        let pa_body = builder.create_block();
        let pa_exit = builder.create_block();
        builder.ins().jump(pa_hdr, &[]);
        builder.switch_to_block(pa_hdr);
        let pa_i = builder.use_var(pa_i_var);
        let pa_cont = builder.ins().icmp(IntCC::SignedLessThan, pa_i, num_params_val);
        builder.ins().brif(pa_cont, pa_body, &[], pa_exit, &[]);
        builder.switch_to_block(pa_body);
        builder.seal_block(pa_body);

        let pa_mpart = self.compile_call_by_name(builder, "nsl_list_get", &[accum, pa_i])?;
        let pa_sq = self.compile_call_by_name(builder, "nsl_tensor_sum_sq", &[pa_mpart])?;
        let pa_tot_cur = builder.use_var(pa_tot_var);
        let pa_tot_new = builder.ins().fadd(pa_tot_cur, pa_sq);
        builder.def_var(pa_tot_var, pa_tot_new);
        let pa_i_next = builder.ins().iadd_imm(pa_i, 1);
        builder.def_var(pa_i_var, pa_i_next);
        builder.ins().jump(pa_hdr, &[]);
        builder.switch_to_block(pa_exit);
        builder.seal_block(pa_hdr);
        builder.seal_block(pa_exit);

        let total_sq = builder.use_var(pa_tot_var);
        let norm = builder.ins().sqrt(total_sq);
        let eps_v = builder.ins().f64const(1e-6_f64);
        let denom = builder.ins().fadd(norm, eps_v);
        let tau_v = builder.ins().f64const(grad_clip_threshold);
        let ratio = builder.ins().fdiv(tau_v, denom);
        let one_f = builder.ins().f64const(1.0_f64);
        let clip_factor = builder.ins().fmin(one_f, ratio);
        Some(clip_factor)
    } else {
        None
    };

    // ── 3. Unified per-param loop ──
    let opt_i_var = state.new_variable();
    builder.declare_var(opt_i_var, cl_types::I64);
    builder.def_var(opt_i_var, builder.ins().iconst(cl_types::I64, 0));

    let hdr = builder.create_block();
    let body = builder.create_block();
    let exit = builder.create_block();
    builder.ins().jump(hdr, &[]);
    builder.switch_to_block(hdr);
    let opt_i = builder.use_var(opt_i_var);
    let cont = builder.ins().icmp(IntCC::SignedLessThan, opt_i, num_params_val);
    builder.ins().brif(cont, body, &[], exit, &[]);
    builder.switch_to_block(body);
    builder.seal_block(body);

    // Common per-param values
    let theta = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, opt_i])?;
    let s1 = self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, opt_i])?;
    let s2 = if num_state_buffers >= 2 {
        self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, opt_i])?
    } else {
        s1
    };

    // Per-iteration mode dispatch
    let deferred_blk = builder.create_block();
    let fullbuf_blk = builder.create_block();
    let iter_join = builder.create_block();
    self.emit_fase_mode_branch(builder, mode_table_base, opt_i, deferred_blk, fullbuf_blk);

    // ── Deferred path ──
    builder.switch_to_block(deferred_blk);
    builder.seal_block(deferred_blk);
    if let Some(accum) = accum_list {
        let m_partial = self.compile_call_by_name(builder, "nsl_list_get", &[accum, opt_i])?;
        if let Some(cf) = clip_factor {
            self.compile_call_by_name(
                builder, "nsl_tensor_mul_scalar_inplace", &[m_partial, cf],
            )?;
        }
        self.fase_emit_final_step(
            builder, theta, s1, m_partial, s2, &fase_plan.recipe, Some((bc1_inv, bc2_inv)),
        )?;
    }
    builder.ins().jump(iter_join, &[]);

    // ── FullBuffer path ──
    builder.switch_to_block(fullbuf_blk);
    builder.seal_block(fullbuf_blk);
    let grad = self.compile_call_by_name(builder, "nsl_list_get", &[opt_grads, opt_i])?;
    self.emit_stdlib_optim_call(
        builder,
        optimizer_name,
        opt_fn,
        theta, grad, s1, s2,
        lr, momentum_const, dampening_const, weight_decay_const, nesterov_const,
        beta1_const, beta2_const, eps_const,
        step_count_var,
    )?;
    builder.ins().jump(iter_join, &[]);

    // ── Join ──
    builder.switch_to_block(iter_join);
    builder.seal_block(iter_join);

    // Loop tail
    let one_i64_tail = builder.ins().iconst(cl_types::I64, 1);
    let next = builder.ins().iadd(opt_i, one_i64_tail);
    builder.def_var(opt_i_var, next);
    builder.ins().jump(hdr, &[]);
    builder.seal_block(hdr);

    builder.switch_to_block(exit);
    builder.seal_block(exit);
    state.current_block = Some(exit);

    Ok(())
}
```

CAUTION:
- `FunctionState` is the state struct name — verify against the actual import in `stmt_fase.rs` (may be `crate::stmt::State` or similar). Read existing helpers to match.
- `emit_fase_mode_branch` was added in Phase 2 on `Compiler`. If it lives in `stmt_fase.rs`, use `self.emit_fase_mode_branch(...)`. If it's elsewhere, adjust the call.
- Unused `bc1_inv`/`bc2_inv` warnings expected if `accum_list` is `None` (Deferred branch short-circuits). Add `#[allow(unused_variables)]` if needed.

- [ ] **Step 3: Build**

```bash
cargo build -p nsl-codegen 2>&1 | tail -5
```

Expected: clean build. Helper is dead until Task 5 wires it.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/stmt_fase.rs
git commit -m "feat(fase): emit_unified_optim_step_dispatch helper

Single-loop optimizer step with per-iteration mode dispatch. Deferred
params go through fase_emit_final_step (with optional clip-factor
scaling on m_partial when two_phase_clip is active); FullBuffer
params go through emit_stdlib_optim_call. Bias-correction scalars
computed once before the loop. Task 5 wires this into compile_train_block."
```

---

## Task 5: Wire helpers into `compile_train_block`

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`

- [ ] **Step 1: Find the optimizer-step gate**

```bash
grep -n "if fase_deferred\|FASE Deferred: emit fused per-parameter step" crates/nsl-codegen/src/stmt.rs
```

The outer `if fase_deferred { ... } else { ... }` starts at around `stmt.rs:5002-5018` and extends through `stmt.rs:5450`.

- [ ] **Step 2: Add the `mode_table_base` gate**

Wrap the existing outer `if fase_deferred { ... } else { ... }` in a new `if let Some(mtb) = mode_table_base` arm:

```rust
// FASE Codegen Phase 3: unified per-param optimizer-step dispatch
// when a mode table is present. Fallback to the monolithic path when
// no WGGO overrides are active.
if let Some(mtb) = mode_table_base {
    let grad_clip_threshold = fase_plan
        .recipe
        .grad_clip
        .unwrap_or(1.0); // only read when two_phase_clip is true
    self.emit_unified_optim_step_dispatch(
        builder, state,
        mtb,
        num_params_val,
        param_list, state_list_1, state_list_2, num_state_buffers,
        accum_list,
        opt_grads,
        step_count_var,
        &fase_plan,
        optimizer_name.as_str(),
        &opt_fn,
        lr, momentum_const, dampening_const, weight_decay_const, nesterov_const,
        beta1_const, beta2_const, eps_const,
        grad_accumulation_steps,
        grad_clip_threshold,
    )?;

    // Post-optimizer block merge
    builder.ins().jump(post_optimizer_block, &[]);
    builder.switch_to_block(post_optimizer_block);
    builder.seal_block(post_optimizer_block);
    state.current_block = Some(post_optimizer_block);
} else if fase_deferred {
    // [EXISTING monolithic Deferred emission unchanged — do NOT edit]
} else {
    // [EXISTING monolithic FullBuffer emission unchanged — do NOT edit]
}
```

CAUTION: the existing `else if fase_deferred` and `else` branches must stay BYTE-IDENTICAL. Only add the new outer `if let Some(mtb)` arm.

Several locals used by the helper (`opt_grads`, `opt_fn`, `lr`, the per-optimizer constants) may live INSIDE the existing `else` branch today. If so, hoist them OUTSIDE the `if fase_deferred` gate so both paths can reference them. Structural hoisting is acceptable; verify behavior doesn't change by running tests.

IMPORTANT: `fase_plan.recipe.grad_clip` may not be a field — check. The grad_clip threshold may live on `fase_plan.two_phase_clip` boolean only, with the actual threshold stored in a separate variable upstream. If so, thread that variable through instead. Read the Phase A emission at the original `stmt.rs:~5123` to see where `grad_clip_threshold` comes from.

- [ ] **Step 3: Extend the stderr renderer**

Find the FASE stderr renderer added in Phase 1 (searches for `[fase] layer` or the reason-string formatter):

```bash
grep -n "\\[fase\\] layer\|FaseModeInfeasible" crates/nsl-codegen/src/stmt.rs
```

Extend the `match &diag.reason { ... }` to handle the new variant:

```rust
crate::wggo_overrides::OverrideRejectReason::TwoPhaseClipConflict {
    grad_clip_threshold,
} => format!("two_phase_clip_threshold_{grad_clip_threshold}"),
```

- [ ] **Step 4: Build + run ALL tests**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: clean build; all existing FASE tests pass (they use no WGGO overrides → fallback path taken, byte-identical).

If any test fails, something in the fallback branches was accidentally changed during hoisting — revert the hoist and keep locals inside the else branches (may require threading them into the helper call differently).

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "feat(fase): wire unified optim-step dispatch + clip conflict renderer

When mode_table_base.is_some(), call emit_unified_optim_step_dispatch
instead of the monolithic if-fase_deferred branch. Fallback path
unchanged (byte-identical for non-WGGO compiles). Extend stderr
renderer with the TwoPhaseClipConflict reason-string format."
```

---

## Task 6: Integration test for mode-table-present compile

**Files:**
- Create: `crates/nsl-codegen/tests/fase_optim_step_dispatch.rs`

- [ ] **Step 1: Write the test**

```rust
//! FASE optim-step per-param dispatch: confirm the fallback and
//! mode-table contracts at the helper level. Full end-to-end
//! behavior is exercised indirectly via the existing FASE test
//! suite (fallback path) plus the two-phase-clip unit tests
//! added to fase.rs (Task 2).

#[test]
fn two_phase_clip_mixed_produces_conflict_diagnostics() {
    use nsl_codegen::fase::{plan_with_overrides, FaseConfig, FaseMode, FaseOptimizer};
    use nsl_codegen::wggo_overrides::OverrideRejectReason;

    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        grad_clip: Some(1.5),
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, false, false, true]);

    // All clamped to Deferred
    assert_eq!(p.per_layer_mode, Some(vec![FaseMode::Deferred; 4]));

    // Two diagnostics (layers 1 and 2)
    assert_eq!(p.override_diagnostics.len(), 2);
    for d in &p.override_diagnostics {
        assert!(matches!(
            d.reason,
            OverrideRejectReason::TwoPhaseClipConflict { grad_clip_threshold: t }
                if (t - 1.5).abs() < 1e-12
        ));
    }
}

#[test]
fn no_clip_preserves_mixed_modes_for_unified_dispatch() {
    use nsl_codegen::fase::{plan_with_overrides, FaseConfig, FaseMode, FaseOptimizer};

    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        grad_clip: None,
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, false, true, false]);

    // Mixed modes preserved — the unified dispatch helper in stmt_fase
    // will branch per-param on these at runtime.
    assert_eq!(
        p.per_layer_mode,
        Some(vec![FaseMode::Deferred, FaseMode::FullBuffer, FaseMode::Deferred, FaseMode::FullBuffer])
    );
    assert!(p.override_diagnostics.is_empty());
}
```

- [ ] **Step 2: Run**

```bash
cargo test -p nsl-codegen --test fase_optim_step_dispatch
```

Expected: 2 tests PASS (both already pass if Tasks 1-2 shipped correctly).

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/tests/fase_optim_step_dispatch.rs
git commit -m "test(fase): integration tests for optim-step dispatch contract

Two tests: two_phase_clip + mixed → clamp diagnostics; no-clip mixed
→ preserved for unified dispatch. Full end-to-end runtime behavior
is covered by the existing FASE test suite (fallback path) plus
Task 2's unit tests inside fase.rs."
```

---

## Task 7: Memory update + push

- [ ] **Step 1: Update `project_wggo_consumers.md`**

In `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wggo_consumers.md`, find the Consumer 3 (FASE) section. Replace the "optimizer-step dispatch deferred to follow-up" text with:

```markdown
## Consumer 3: FASE (FULLY SHIPPED 2026-04-15)

All three phases merged:
- Phase 1 (planner + observability): PR #39 at `661e7d5`.
- Phase 2 (accumulation-loop per-param dispatch): PR #41 at `aa0a0c5`.
- Phase 3 (optimizer-step per-param dispatch): PR #<num>.

Phase 3 closed Phase 2's latent mixed-mode correctness bug: each param's
optimizer path now reads the accumulator convention that matches its
accumulation path. Also added a `TwoPhaseClipConflict` diagnostic that
clamps mixed-mode overrides to uniform Deferred when grad_clip is active,
since Phase A's global ||m_partial||² norm requires uniform accumulation.

**Phase 3 surface:**
- `OverrideRejectReason::TwoPhaseClipConflict { grad_clip_threshold: f64 }` variant.
- `plan_with_overrides` clamps when `grad_clip.is_some() && accumulation > 1 && global == Deferred && any fase_fused=false`.
- `Compiler::emit_stdlib_optim_call` — extracted stdlib-optim dispatch.
- `Compiler::emit_unified_optim_step_dispatch` — single-loop per-param dispatch when mode_table_base is present.
- Stderr reason format: `two_phase_clip_threshold_<τ>`.

**Spec/plan:** `docs/superpowers/specs/2026-04-15-fase-optim-step-dispatch-design.md` + `docs/superpowers/plans/2026-04-15-fase-optim-step-dispatch-implementation.md`.
```

- [ ] **Step 2: Update `MEMORY.md`**

Find the WGGO consumer-rollout line and update to:

```markdown
- [WGGO AppliedPlan → consumers](project_wggo_consumers.md) — CSHA + WRGA + CPDT + FASE all fully shipped. Only Prune remaining.
```

- [ ] **Step 3: Full workspace test**

```bash
cd c:/Users/bwiem/projects/NSL/.worktrees/fase-optim-step-dispatch
cargo test --workspace 2>&1 | tail -10
```

Expected: all pass except pre-existing Windows file-lock flakes (`e2e_m12_grad_basic_source_ad`, `e2e_m27_*`), which are documented in earlier PRs.

- [ ] **Step 4: Push**

```bash
git push -u origin feat/fase-optim-step-dispatch
```

- [ ] **Step 5: Prepare PR body**

```markdown
## Summary
- New `OverrideRejectReason::TwoPhaseClipConflict { grad_clip_threshold }` variant.
- `plan_with_overrides` clamps mixed-mode overrides to uniform Deferred when grad_clip is active, preserving Phase A's global norm validity.
- Extracted `Compiler::emit_stdlib_optim_call` helper (pure refactor of the six-optimizer dispatch).
- New `Compiler::emit_unified_optim_step_dispatch` — single-loop per-param optimizer dispatch.
- Gate at `compile_train_block`: `if let Some(mtb) = mode_table_base { unified } else if fase_deferred { monolithic Deferred } else { monolithic stdlib }`.
- Closes the Phase 2 follow-up. **Fixes the latent mixed-mode correctness bug** where Phase 2's per-param accumulation stored different conventions (mean for Deferred, sum for FullBuffer) but the monolithic optimizer step read only one convention.

## WGGO consumer rollout status after this PR
- ✅ CSHA, WRGA, CPDT, **FASE (all three phases)**
- ⏳ Prune (consumer 4, not started)

## Test plan
- [ ] `cargo test -p nsl-codegen fase::tests::two_phase_clip` — 4 new unit tests pass
- [ ] `cargo test -p nsl-codegen wggo_overrides::tests::two_phase_clip_conflict_round_trips_debug` — passes
- [ ] `cargo test -p nsl-codegen --test fase_optim_step_dispatch` — 2 tests pass
- [ ] `cargo test -p nsl-codegen --lib` — no regression vs pre-PR baseline
- [ ] Manual: compile a fixture with WGGO + mixed `fase_fused` (no grad_clip); objdump shows the `nsl_fase_param_modes_*` symbol and the compile succeeds

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

Open PR on GitHub.

---

## Self-review checklist

- [ ] Every spec section (§4 invariants, §5 data model + unified loop + extracted helper, §6 clamp rule, §8 testing) has ≥1 task implementing it.
- [ ] No `TBD` / `implement later`.
- [ ] Method/type names consistent: `TwoPhaseClipConflict`, `emit_stdlib_optim_call`, `emit_unified_optim_step_dispatch`, `mode_table_base`.
- [ ] Every code step shows actual code (no "similar to Task N" hand-waves).
- [ ] Project's most-common mistake (E0063 missing fields) is flagged at Task 1 Step 5 (exhaustive match sites).
- [ ] Task 3's extraction and Task 5's gate are explicit about "monolithic fallback stays byte-identical" — preserves the regression guarantee.
