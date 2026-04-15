# FASE Optimizer-Step Per-Param Dispatch — Design

**Date:** 2026-04-15
**Status:** Approved for implementation
**Branch (target):** `feat/fase-optim-step-dispatch`
**Predecessor:** [FASE Codegen Phase 2](2026-04-15-fase-codegen-phase2-design.md) (merged via PR #41 at `aa0a0c5`). Closes the Task 4d follow-up deferred in that PR.

## 1. Goal

Make FASE's optimizer-step codegen consult the same per-parameter `.rodata` mode table that Phase 2 built for the accumulation loop. After this lands:

1. **Correctness:** mixed-mode WGGO overrides no longer produce a semantic mismatch between accumulation and optimizer dispatch. Each param's optimizer step reads the accumulator convention that matches its accumulation path.
2. **Consistency:** a single `.rodata` mode table now drives both the accumulation loop (Phase 2) AND the optimizer step (this PR). Consumer 3 of the 5-consumer rollout reaches full codegen fidelity.

## 2. Correctness Problem Being Fixed

Phase 2's accumulation loop dispatches per-param: Deferred writes `m_partial += (1/N) * grad` (**mean** convention), FullBuffer writes `accum += grad` (**sum** convention). The two conventions are internally consistent per path.

Phase 2 left the optimizer step monolithic (global `fase_deferred` branch). In mixed-mode WGGO scenarios this is semantically incorrect:

- Global `Deferred` with some FullBuffer-overridden params → stdlib-free Deferred optimizer step reads their sum buffer and treats it as `m_partial` (the mean) → update applied with gradient scaled N× too large.
- Global `FullBuffer` with some Deferred-overridden params → stdlib optimizer step reads their mean buffer and treats it as a raw gradient → update applied N× too small.

Per-param optimizer dispatch closes this gap: each param's optimizer path reads the accumulator convention that matches how its accumulator was populated. No new data is needed — the `.rodata` mode table already encodes the accumulation-path decision per param.

## 3. Non-Goals (Explicitly Deferred)

- Correcting Phase A's `||m_partial||²` norm for mixed modes (it requires per-contribution rescaling by `1/N²` for FullBuffer params). This plan instead clamps two-phase-clip scenarios to uniform Deferred via a new diagnostic (see §6).
- Per-layer grad_clip thresholds — grad_clip stays global by Phase 1 spec.
- FASE source-AD hook path (`fase_hook_active` branch).
- Changing the stdlib optimizer-step FFI conventions (each optimizer's arg shape is preserved).

## 4. Scope Invariants (do not break)

### 4.1 Fallback path is byte-identical

When `mode_table_base.is_none()` (no WGGO active), the optimizer step emits today's outer `if fase_deferred { Deferred Phase A/B/fs loops } else { stdlib per-param loop }` structure verbatim. Every existing FASE test must produce identical Cranelift IR.

### 4.2 Param-list and state-list order

`param_list[i]`, `state_list_1[i]`, `state_list_2[i]`, `accum[i]`, `grads_list[i]`, and `modes[i]` all correspond to the same `param_paths[i]`. This is the Phase 1 invariant — unchanged.

### 4.3 Bias-correction setup

`bc1_inv` / `bc2_inv` must be computed once before the loop (not per iteration) when any Deferred param can exist. Computing twice or skipping when needed produces incorrect AdamW updates.

## 5. Design

### 5.1 New diagnostic variant

File: `crates/nsl-codegen/src/wggo_overrides.rs`

Add to `OverrideRejectReason`:

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

Stderr reason format: `two_phase_clip_threshold_<τ>` (e.g., `two_phase_clip_threshold_1.0`).

### 5.2 `plan_with_overrides` rule extension

File: `crates/nsl-codegen/src/fase.rs`

The existing `plan_with_overrides(cfg, wggo_fused_per_layer)` applies feasibility clamps (Lion forbids Deferred etc.). Add a new clamp:

```
When cfg.grad_clip.is_some() AND accumulation > 1 AND
global_mode == Deferred AND any layer has fase_fused == false:
    force every layer's mode to Deferred.
    For each clamped layer (those with fase_fused == false), emit a
    TwoPhaseClipConflict diagnostic carrying cfg.grad_clip.unwrap().
```

Rationale: two-phase-clip requires a uniform mode for a meaningful global norm. Clamping to all-Deferred preserves today's correctness for non-WGGO compiles and emits a visible signal in stderr when WGGO's per-layer preferences are overridden.

This clamp runs BEFORE the existing Lion/Unknown/allow_v_approx clamps. Ordering matters: if global_mode is already FullBuffer (Lion etc.), the two-phase-clip branch wouldn't be entered anyway, so the existing clamps continue to dominate there.

### 5.3 Unified optimizer-step loop

File: `crates/nsl-codegen/src/stmt.rs`

Replace the outer `if fase_deferred { complex Deferred emission } else { stdlib loop }` around `stmt.rs:5002-5400` with:

```rust
if let Some(mtb) = mode_table_base {
    emit_unified_optim_step_dispatch(
        self, builder, state,
        mtb,
        num_params_val,
        param_list, state_list_1, state_list_2, num_state_buffers,
        accum_list_or_none,      // accum for the Deferred side's m_partial
        opt_grads,               // accum_or_grads for the FullBuffer side
        step_count_var,
        fase_plan,
        &optimizer_name,
        // AdamW/Adam hyperparams
        lr_var, beta1_value, beta2_value, eps_value, weight_decay_value,
        // SGD-like hyperparams
        momentum_value, dampening_value, nesterov_value,
    )?;
} else if fase_deferred {
    // Existing monolithic Deferred emission (Phase A/B two-phase-clip OR
    // fs_body non-clip loop) — preserved byte-identically.
    // [original code here, unchanged]
} else {
    // Existing monolithic stdlib per-param loop — preserved byte-identically.
    // [original code here, unchanged]
}
```

### 5.4 `emit_unified_optim_step_dispatch` helper

File: `crates/nsl-codegen/src/stmt_fase.rs` (new helper adjacent to existing FASE emitters — keeps stmt.rs's already-large function from growing further).

```rust
/// Emit a single unified optimizer-step loop with per-param runtime
/// dispatch on the FASE mode table. Used when `mode_table_base.is_some()`.
/// The monolithic pre-Phase-2 path is NOT handled by this helper — the
/// caller keeps that path for the fallback case.
///
/// By §6, two-phase-clip + mixed mode cannot occur here: plan_with_overrides
/// clamps mixed FullBuffer requests to Deferred when grad_clip is active,
/// so when mode_table_base.is_some() AND two_phase_clip is true, every
/// param is Deferred and this helper's Deferred path behaves identically
/// to the monolithic Phase A/B emission. For simplicity and consistency,
/// the helper always uses the unified per-iteration dispatch regardless
/// of clip state.
pub(crate) fn emit_unified_optim_step_dispatch(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    state: &mut State,
    mode_table_base: Value,
    num_params_val: Value,
    param_list: Value,
    state_list_1: Value,
    state_list_2: Value,
    num_state_buffers: usize,
    accum_list: Option<Value>,   // m_partial for Deferred params
    opt_grads: Value,             // raw grads for FullBuffer params
    step_count_var: Variable,
    fase_plan: &FasePlan,
    optimizer_name: &str,
    // AdamW/Adam hyperparams (needed for stdlib call + bc setup)
    lr_var: Variable,
    beta1: f64, beta2: f64, eps: f64, weight_decay: f64,
    // SGD-like hyperparams (needed for stdlib call)
    momentum: f64, dampening: f64, nesterov: bool,
) -> Result<(), CodegenError>
```

Body shape:

```rust
// 1. Bias-correction setup (always computed — cheap, only used by Deferred)
let (bc1_inv, bc2_inv) = compute_bc_scalars(builder, step_count_var, fase_plan, beta1, beta2);

// 2. Phase A sum_sq loop (only when two_phase_clip is active).
//    Per §6, all params are Deferred here (clamped upstream), so the
//    existing pa_body shape works unchanged. Emit it verbatim.
let clip_factor = if fase_plan.two_phase_clip { emit_phase_a_sum_sq(...) } else { None };

// 3. Unified optimizer step loop
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

// Load common per-param values
let theta  = compiler.compile_call_by_name(builder, "nsl_list_get", &[param_list, opt_i])?;
let state1 = compiler.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, opt_i])?;
let state2 = if num_state_buffers >= 2 {
    compiler.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, opt_i])?
} else {
    state1   // SGD placeholder
};

// Per-iteration dispatch
let deferred_blk = builder.create_block();
let fullbuf_blk  = builder.create_block();
let iter_join    = builder.create_block();
compiler.emit_fase_mode_branch(builder, mode_table_base, opt_i, deferred_blk, fullbuf_blk);

// ── Deferred path ──
builder.switch_to_block(deferred_blk);
builder.seal_block(deferred_blk);
if let Some(accum) = accum_list {
    let m_partial = compiler.compile_call_by_name(builder, "nsl_list_get", &[accum, opt_i])?;
    if let Some(cf) = clip_factor {
        // Phase B clip-scale m_partial in place before fused step
        compiler.compile_call_by_name(builder, "nsl_tensor_mul_scalar_inplace", &[m_partial, cf])?;
    }
    compiler.fase_emit_final_step(builder, theta, state1, m_partial, state2, &fase_plan.recipe, Some((bc1_inv, bc2_inv)))?;
}
builder.ins().jump(iter_join, &[]);

// ── FullBuffer path ──
builder.switch_to_block(fullbuf_blk);
builder.seal_block(fullbuf_blk);
let grad = compiler.compile_call_by_name(builder, "nsl_list_get", &[opt_grads, opt_i])?;
emit_stdlib_optim_call(compiler, builder, optimizer_name, theta, grad, state1, state2,
    lr_var, beta1, beta2, eps, weight_decay, momentum, dampening, nesterov,
    step_count_var)?;
builder.ins().jump(iter_join, &[]);

// ── Join ──
builder.switch_to_block(iter_join);
builder.seal_block(iter_join);

// Loop tail
let one_i64 = builder.ins().iconst(cl_types::I64, 1);
let next = builder.ins().iadd(opt_i, one_i64);
builder.def_var(opt_i_var, next);
builder.ins().jump(hdr, &[]);
builder.seal_block(hdr);
builder.switch_to_block(exit);
builder.seal_block(exit);
state.current_block = Some(exit);
```

### 5.5 `emit_stdlib_optim_call` helper

File: `crates/nsl-codegen/src/stmt_fase.rs`

Factor the existing `match optimizer_name.as_str()` dispatch at `stmt.rs:5339-5400` into a helper. Each optimizer's arg shape is preserved verbatim. Takes `theta, grad, state1, state2` plus hyperparam variables/constants. Returns `Result<(), CodegenError>`.

This extraction is defensible cleanup: the unified loop needs to call this dispatch per iteration, and the monolithic path can call the same helper, reducing duplication. Both uses are covered by existing stdlib-step tests + the new integration test.

## 6. Two-Phase-Clip Clamp (§5.2 expanded)

When the following conditions hold simultaneously:

- `cfg.grad_clip.is_some()`
- `cfg.accumulation > 1`
- Global FASE mode computed by `plan(cfg)` is `Deferred`
- WGGO `wggo_fused_per_layer` contains any `false`

…then `plan_with_overrides` clamps every `fase_fused=false` layer to `Deferred` and emits one `TwoPhaseClipConflict` diagnostic per clamped layer carrying the original `grad_clip` threshold. Resulting `per_layer_mode` is `Some(vec![Deferred; N])`.

Stderr format from the `stmt.rs` renderer:
```
[fase] layer:N wggo-override-rejected requested=FullBuffer applied=Deferred reason=two_phase_clip_threshold_1.0
```

The clamp runs at `plan_with_overrides` time. By the time `compile_train_block` sees `fase_plan`, all modes are uniform if two-phase-clip is active. `emit_unified_optim_step_dispatch`'s "Phase A expects all-Deferred" invariant is upheld by this upstream guarantee.

## 7. Architecture Diagram

```
                            WGGO                  FaseConfig
                              │                       │
                              ▼                       ▼
                      plan_with_overrides (fase.rs)
                         │
            ┌────────────┼─────────────────────────────────┐
            │            │                                 │
 Lion/Unknown/           │ two_phase_clip + mixed          │ feasible mixed
 allow_v_approx=false    │ (NEW clamp §5.2)                │ (existing behavior)
 clamp                   │                                 │
            │            │                                 │
            ▼            ▼                                 ▼
  FaseModeInfeasible  TwoPhaseClipConflict       FasePlan with per_layer_mode
                                                          │
                                                          ▼
                                       build_param_mode_table → .rodata

                                                          │
                                       ┌──────────────────┼──────────────────┐
                                       ▼                                     ▼
                            Accumulation loop (Phase 2)          Optimizer step (THIS PR)
                              gai → modes[gai]                       opt_i → modes[opt_i]
                              Deferred: fase_emit_accumulate           Deferred: fase_emit_final_step
                              FullBuffer: nsl_grad_accumulate_add      FullBuffer: stdlib <optim>_step
                              
                              accum[i] = mean for Deferred,           reads matching convention
                                         sum  for FullBuffer          per param — correctness fix
```

## 8. Testing

### 8.1 `plan_with_overrides` unit tests (in `fase.rs`)

1. **`two_phase_clip_with_mixed_modes_clamps_all_to_deferred`** — `grad_clip=Some(1.0)`, `accumulation=4`, AdamW, `wggo_fused=[true, false, true, false]` → `per_layer_mode = Some(vec![Deferred; 4])`, 2 `TwoPhaseClipConflict` diagnostics (one per `false` input).
2. **`two_phase_clip_all_deferred_no_clamp_fires`** — same config with `wggo_fused=[true, true, true, true]` → no diagnostics, `per_layer_mode = Some(vec![Deferred; 4])`.
3. **`no_clip_mixed_modes_preserved`** — `grad_clip=None`, same mixed input → `per_layer_mode` preserves mixed modes, zero diagnostics.
4. **`diagnostic_carries_threshold`** — `grad_clip=Some(2.5)` + 1 clamp → diagnostic reason is `TwoPhaseClipConflict { grad_clip_threshold: 2.5 }`.

### 8.2 `OverrideRejectReason::TwoPhaseClipConflict` round-trip test

5. **`two_phase_clip_conflict_round_trips_debug`** — Debug format contains `TwoPhaseClipConflict` and the threshold value.

### 8.3 `emit_stdlib_optim_call` extraction regression

The monolithic FullBuffer path now calls the extracted helper. All existing FASE stdlib-optimizer tests (SGD, Adam, AdamW, Lion, Muon, SOAP) must produce byte-identical behavior — no test-level changes; ensure the refactor preserves arg ordering.

### 8.4 Unified-loop integration test

6. **`unified_optim_loop_emits_when_mode_table_present`** — new file `crates/nsl-codegen/tests/fase_optim_step_dispatch.rs`. Assert `build_param_mode_table` returning `Some` produces a compile that includes the dispatch — weakly, since full IR inspection is awkward. At minimum: compile a 4-param fixture with WGGO + mixed Deferred/FullBuffer non-clip config and confirm no compile-time errors and the final object contains the `nsl_fase_param_modes_*` symbol.

### 8.5 Fallback regression

All existing FASE tests (without WGGO overrides) must pass unchanged. Manual assertion: diff the emitted IR for a non-WGGO compile before vs after to confirm byte-identity.

**Total: 6 new tests + one helper-extraction regression guard.**

## 9. Risks & Open Questions

- **Risk: `emit_unified_optim_step_dispatch` grows past the "focused file" threshold.** The helper needs ~150-200 lines including the Phase A branch. Put it in `stmt_fase.rs` alongside `fase_emit_accumulate` / `fase_emit_final_step` so `stmt.rs` stays at its current size.
- **Risk: stdlib optimizer calls in the FullBuffer branch need the `step_count_var` for AdamW's `t` argument.** The monolithic path reads it at line 5359. The unified helper takes `step_count_var` as a parameter — same data flow.
- **Risk: `emit_stdlib_optim_call` extraction breaks an existing test due to subtle arg-order divergence.** Mitigation: do the extraction in a separate commit so the diff is inspectable; run the full FASE test suite after the extraction before layering the unified loop on top.
- **Risk: SOAP optimizer's stdlib call might have a different arg pattern not represented in the existing match.** The existing `match optimizer_name` covers `sgd`, `adam`, `adamw`, `lion`, `muon`, `soap`. Implementation must preserve all six arms verbatim.
- **Open: should `emit_unified_optim_step_dispatch` also subsume the monolithic fallback paths?** For this plan, NO — fallback paths stay separate to guarantee byte-identical IR for non-WGGO compiles. A future consolidation plan could merge them once the unified path has baked; that's out of scope.

## 10. Success Criteria

1. `plan_with_overrides` with two-phase-clip + mixed modes emits `TwoPhaseClipConflict` diagnostics and clamps all layers to `Deferred`.
2. When `mode_table_base.is_some()`, the optimizer step uses a single unified loop with per-iteration mode dispatch.
3. For mixed-mode non-clip WGGO compiles, each param's optimizer path matches its accumulation path (fixes the Phase 2 latent mismatch).
4. Non-WGGO compiles produce byte-identical Cranelift IR to pre-PR.
5. All 6 new tests pass; all existing FASE tests pass unchanged.
6. `project_wggo_consumers.md` updated: Consumer 3 reaches full codegen fidelity.

## 11. Files Touched

- `crates/nsl-codegen/src/wggo_overrides.rs` — add `TwoPhaseClipConflict { grad_clip_threshold: f64 }` variant + round-trip test.
- `crates/nsl-codegen/src/fase.rs` — extend `plan_with_overrides` with the two-phase-clip clamp rule; 4 new unit tests.
- `crates/nsl-codegen/src/stmt_fase.rs` — new `emit_unified_optim_step_dispatch` helper + extracted `emit_stdlib_optim_call` helper.
- `crates/nsl-codegen/src/stmt.rs` — the outer `if let Some(mtb) = mode_table_base { ... } else if fase_deferred { original } else { original }` gate at the optimizer-step site; call both helpers; update the stderr renderer reason-formatter to include `TwoPhaseClipConflict`.
- `crates/nsl-codegen/tests/fase_optim_step_dispatch.rs` (new) — 1 weak integration test.
