# FASE Two-Phase Gradient Clip Codegen — Design

**Date:** 2026-04-14
**Status:** Design approved, ready for implementation plan
**Depends on:** Item #1 (Deferred codegen), Item #2 (numerical validation), bias-correction follow-up (Tasks BC1-BC6)
**Follow-up to:** [2026-04-14-fase-deferred-codegen-integration-design.md](2026-04-14-fase-deferred-codegen-integration-design.md)

## Context

Item #1 landed a temporary downgrade in `fase::plan()` that forces `FullBuffer` mode whenever `grad_clip.is_some()`, because the two-phase clip emission wasn't implemented.  The planner's `FasePlan::two_phase_clip` flag was populated but unused; `fase_clip.rs` provides a `ClipPlan` scaffold (~229 LOC) with no consumer.

This spec removes the downgrade and implements the two-phase emission.  After it lands, users who set `train(grad_clip = 1.0, grad_accumulation = 4)` with AdamW/Adam/SGD take the Deferred path with in-line gradient clipping, rather than falling back to the pre-FASE accum-buffer path.

The semantics match PyTorch's `torch.nn.utils.clip_grad_norm_` default: global L2 norm across all parameter gradients, with `clip_factor = min(1, τ / (norm + ε))` applied uniformly.

## Goals

1. Remove the `grad_clip → FullBuffer` downgrade in `fase::plan()`.
2. Emit a two-phase final micro-batch when `two_phase_clip` is set:
   - Phase A: finish `m_partial += g/N` accumulation AND compute `total_sq = Σ_k sum(m_partial_k²)` in one pass.
   - Phase B: compute `clip_factor`, then for each parameter scale `m_partial` in place and run the existing fused optimizer step.
3. Add two runtime helpers (`nsl_tensor_sum_sq`, `nsl_tensor_mul_scalar_inplace`) required by the above.
4. Validate end-to-end with a new numerical test analogous to item #2 Task 4.

## Non-Goals

- `LinfPerParam` norm type.  The `ClipNorm` enum lists it, but the NSL surface syntax for `grad_clip` is a single float — there is no way for users to request per-parameter L∞ clipping today.  Adding it is a separate language-surface change.
- Revisiting the non-FASE clip path.  `nsl_clip_grad_norm` (runtime) and its `stmt.rs` dispatch for the `FullBuffer` path are already correct and covered by existing tests.
- SGD-specific clip logic.  SGD goes through the same `fase_emit_final_step` dispatcher as AdamW; clip applies identically at the `m_partial` scale step.

## Design Decisions

### D1. L2 global norm only

Match PyTorch default and existing non-FASE clip path.  `ClipNorm::L2Global` is the only variant emitted.  The `LinfPerParam` arm is left in `fase_clip.rs` for a future syntax addition but is never selected by the planner today.

### D2. Two-pass algorithm with Phase-A fusion

Fuse Phase A's two concerns (accumulate + sum_sq) into one loop over parameters on the final micro-batch.  Justification: `sum_sq(m_partial_k)` is valid only *after* `m_partial_k += g_k/N` on the final micro-batch (non-final micro-batches don't compute norm), so fusion is natural and safe.  No double-reads, no stale values.

Phase A pseudocode (emitted Cranelift IR):

```
total_sq = f64const(0.0)
for k in 0..num_params:
    m_partial_k = nsl_list_get(accum_list, k)
    grad_k      = nsl_list_get(grads_list, k)
    fase_emit_accumulate(builder, m_partial_k, grad_k, recipe.accum_scale)
    partial_sq  = nsl_tensor_sum_sq(m_partial_k)
    total_sq    = fadd(total_sq, partial_sq)
    nsl_tensor_free(grad_k)
```

After the loop:

```
norm        = sqrt(total_sq)
denom       = fadd(norm, 1e-6)
ratio       = fdiv(threshold, denom)
clip_factor = fmin(f64const(1.0), ratio)
```

`1e-6` matches PyTorch's conventional stability constant.

Phase B pseudocode:

```
for k in 0..num_params:
    m_partial_k = nsl_list_get(accum_list, k)
    nsl_tensor_mul_scalar_inplace(m_partial_k, clip_factor)
    theta_k, m_k, v_k = resolve state
    fase_emit_final_step(theta_k, m_k, m_partial_k, v_k, recipe, bc_params)
```

`fase_emit_final_step` already zeroes `m_partial` at end; no separate reset loop needed.

### D3. Two new runtime FFIs

Both live in `crates/nsl-runtime/src/tensor/mod.rs` next to `nsl_tensor_zero_inplace` and `nsl_clip_grad_norm`:

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_sum_sq(ptr: i64) -> f64 {
    // Walk tensor, accumulate Σ x² as f64.  GPU tensors are transferred
    // to CPU first (mirrors nsl_clip_grad_norm pattern).  Both f32 and
    // f64 dtypes supported.
}

#[no_mangle]
pub extern "C" fn nsl_tensor_mul_scalar_inplace(ptr: i64, scalar: f64) {
    // In-place elementwise scale.  No allocation, no relinquish.
}
```

Rationale for in-place scaling (vs allocate-copy-free pattern): saves one full-tensor allocation and free per parameter per final micro-batch.  For a 7B model with ~200 parameter tensors, that is 200 unnecessary heap transactions on the critical path.  Also applicable to future work (CTQS dequant scaling, CFA reference-model scaling).

Both helpers also need entries in `crates/nsl-codegen/src/builtins.rs`:

```rust
("nsl_tensor_sum_sq", &[types::I64], Some(types::F64)),
("nsl_tensor_mul_scalar_inplace", &[types::I64, types::F64], None),
```

### D4. Planner change

In `crates/nsl-codegen/src/fase.rs::plan()`, remove the temporary downgrade block added by item #1 Task 5.  Keep the existing phase-sequence construction — it already emits `BackwardPhase::FinalTwoPhase` when `grad_clip.is_some()`.

The `fase_clip.rs` `ClipPlan` scaffold already has `accumulate_during_phase_a: true` as the default — no `ClipPlan` construction changes needed.

### D5. `stmt.rs` dispatch

The existing Deferred branch around [stmt.rs:4260](crates/nsl-codegen/src/stmt.rs#L4260) (per-parameter fused-step loop) splits into two branches:

```rust
if fase_plan.two_phase_clip {
    // Phase A: fused accumulation + sum_sq loop (runtime loop over num_params_val).
    // After loop: compute clip_factor as f64 Cranelift Value.
    // Phase B: scale-then-fused-step loop (runtime loop over num_params_val).
} else {
    // Existing Item #1 path: per-parameter fused step loop (unchanged).
}
```

Bias-correction scalars `bc1_inv` / `bc2_inv` are computed the same way in both branches (from `step_count_var`), before the Phase B loop in the clipped case.  Pass `Some((bc1_inv, bc2_inv))` into `fase_emit_final_step` as in the bias-correction follow-up.

**Important:** the accumulation pass in Phase A must NOT also run on non-final micro-batches via the clip branch — non-final micro-batches continue through the existing [stmt.rs:4183](crates/nsl-codegen/src/stmt.rs#L4183) accumulation loop, which is a separate loop upstream of the final-step dispatch.  Phase A only fires on the final micro-batch.

### D6. Tests

Three tests, matching the item #2 pattern:

1. **Planner unit test** (`fase::tests::grad_clip_with_adamw_now_returns_deferred`) — asserts that `plan(FaseConfig { grad_clip: Some(1.0), optimizer: AdamW, ... })` returns `FaseMode::Deferred` (not `FullBuffer`).  Also asserts `two_phase_clip == true` and the phase sequence ends with `FinalTwoPhase`.
2. **Clip-plan shape test** (`fase_clip::tests::two_phase_plan_fuses_accumulation`) — asserts the produced `ClipPlan` has `enabled: true`, `threshold == τ`, `norm == ClipNorm::L2Global`, `accumulate_during_phase_a: true`.
3. **End-to-end numerical test** (`adamw_deferred_with_grad_clip` integration test under `crates/nsl-codegen/tests/fase_numerical_validation.rs`) — fixture: AdamW + `grad_clip = 0.01` (small enough that clipping actually fires with the fixture's gradients).  Rust reference computes expected trajectory: compute `g`, accumulate `m_partial`, compute `norm = ‖m_partial‖₂`, compute `clip_factor = min(1, 0.01 / (norm + 1e-6))`, scale `m_partial *= clip_factor`, run bias-corrected AdamW step.  Assert rel_err < 1e-5 (widen to 1e-4 if f32 chain-precision requires).

The non-FASE `FullBuffer` + grad_clip path is validated by existing snapshot/integration tests; don't duplicate.

### D7. Components touched

| File | Change |
|---|---|
| `crates/nsl-runtime/src/tensor/mod.rs` | Add `nsl_tensor_sum_sq` + `nsl_tensor_mul_scalar_inplace` (~25 LOC total). |
| `crates/nsl-codegen/src/builtins.rs` | Register 2 new FFI signatures. |
| `crates/nsl-codegen/src/fase.rs` | Remove the temporary `grad_clip → FullBuffer` downgrade; update planner tests. |
| `crates/nsl-codegen/src/fase_clip.rs` | Extend `#[cfg(test)] mod tests` with the new plan-shape assertion; no code-path changes. |
| `crates/nsl-codegen/src/stmt.rs` | Add `if fase_plan.two_phase_clip` branch with Phase A fusion + scalar computation + Phase B loop. |
| `crates/nsl-codegen/tests/fase_numerical_validation.rs` | Add `adamw_deferred_with_grad_clip` test + reference. |
| `crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_grad_clip.nsl` (new) | Fixture with `grad_clip = 0.01`. |

## Risks

1. **f32 rel_err tolerance.**  Phase A's `sqrt(total_sq)` and Phase B's scalar multiply add f32-precision rounding on top of the existing AdamW chain.  If `1e-5` tolerance proves too tight, widen to `1e-4` and add a comment citing the extra ops — don't loosen silently.
2. **Clip_factor at the boundary.**  When `norm == τ` exactly, `clip_factor` is `1.0` in exact arithmetic but may land on either side of `1.0` in f32 due to the `min` implementation.  Not an issue for tests (we choose a fixture where `norm ≫ τ` so clipping is active), but document the bound-case behavior in the code comment.
3. **GPU tensors.**  `nsl_tensor_sum_sq` must handle GPU→CPU transfer the same way `nsl_clip_grad_norm` does.  Fixtures run CPU-only (matches item #2 setup), but the code must be GPU-safe for real users.
4. **Phase A/B runtime loop structure.**  Two new runtime `for i in 0..num_params` loops using the Cranelift block-pattern already present in `stmt.rs`.  Pattern-match the existing accum loop's structure exactly to avoid block-sealing bugs.

## Success Criteria

- `cargo test -p nsl-codegen` all green, including:
  - New tests T1 / T2 / T3 all pass.
  - Existing item #1 tests, item #2 tests (SGD equivalence, AdamW pipeline, Jensen fence × 2, bias-correction shape tests) all still pass.
  - No snapshot changes except for fixtures that exercise `grad_clip + grad_accumulation > 1` (if any — none found today; re-baseline if any appear).
- A user-written `train(grad_clip=1.0, grad_accumulation=4)` with AdamW no longer falls through to the FullBuffer path; it takes Deferred with inline two-phase clipping.

## Follow-Ups

This spec closes item #3.  Remaining FASE items:

- **Item #4 (M36 memory planner).**  Subsumes what was item #1b — per-parameter gradient-slot reuse statically.
- **Item #5 (peak-memory regression test).**  Depends on #4.
- **Item #6 (`nsl check --training-report` CLI).**  Depends on nothing else; pure observability.
