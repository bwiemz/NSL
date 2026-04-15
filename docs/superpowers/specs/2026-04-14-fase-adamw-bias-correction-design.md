# FASE AdamW Bias Correction — Design

**Date:** 2026-04-14
**Status:** Design approved, ready for implementation plan
**Depends on:** Item #1 (Deferred codegen), Item #2 Task 4 (AdamW equivalence test, which surfaced this bug)
**Follow-up to:** [2026-04-14-fase-numerical-validation-design.md](2026-04-14-fase-numerical-validation-design.md)

## Context

Item #2 Task 4's AdamW equivalence test exposed that `fase_optimizer::emit_adamw` omits the `(1 - β^t)` bias-correction divisions that the stdlib `adamw_step` performs.  The test was made to pass by writing the Rust reference without bias correction — but that just encoded the bug into the test.

User-visible consequence: the same `train(optimizer: AdamW(...))` produces different parameter trajectories depending on whether `grad_accumulation=1` (Passthrough → stdlib, with bias correction) or `grad_accumulation=4` (Deferred → FASE emitter, without bias correction).  This breaks the contract that `grad_accumulation` is a compile-time optimization that does not change the optimizer's numerics.  The divergence is largest in the first ~100 steps, exactly when training is most fragile.

## Goals

Add bias correction to FASE Deferred AdamW / Adam emission so that
`grad_accumulation > 1` produces the same AdamW trajectory (up to the
Jensen v-approximation) as `grad_accumulation = 1`.

## Non-Goals

- Changing the v-approximation itself.  That is intentional and fenced
  by the Jensen test.
- Touching SGD / SGD-momentum / Lion emitters.  Those don't have bias
  correction in the stdlib either.

## Design Decisions

### D1. Compute `bc_inv` once per optimizer step, not per parameter

Bias-correction factors `bc1_inv = 1/(1 - β₁^t)` and `bc2_inv = 1/(1 - β₂^t)` are scalars that depend only on the optimizer step count `t`, not on the parameter.  Compute them once per optimizer-step firing in the stmt.rs dispatcher, then broadcast-multiply each parameter's `m` and `v` against them during the per-parameter loop.  Cost: two `powf` + two subtract/divide scalar ops per optimizer step, amortized across all parameters.

### D2. Runtime helper `nsl_bias_correction_inv`

New FFI: `nsl_bias_correction_inv(base: f64, step: i64) -> f64`.  Returns `1.0 / (1.0 - base.powf(step as f64))`.  Lives in a new file `crates/nsl-runtime/src/fase_bc.rs` (or appended to an existing math module — implementation plan picks).  ~15 LOC.

Why a runtime helper rather than inlining with Cranelift intrinsics:
Cranelift has no `pow` intrinsic; calling `libm` from emitted code requires plumbing we don't have.  One FFI call per optimizer step is negligible.

### D3. Extend `UpdateOp` with `ScalarMulByBc`

Add two things to `fase_optimizer.rs`:

1. New registers: `Register::MHat`, `Register::VHat`.  Scratch slots for bias-corrected views.  Allocated lazily in `fase_emit_final_step`.

2. New enum `BcKind { Beta1, Beta2 }` and new `UpdateOp`:

   ```rust
   /// `dst = src * bias_correction_inverse_for(kind)`.
   /// The runtime scalar is supplied to the emitter by the dispatcher
   /// (`stmt.rs`) as two f64 Values, one per `BcKind`.
   ScalarMulByBc {
       dst: Register,
       src: Register,
       kind: BcKind,
   },
   ```

Rejected alternatives:
- Inline bias correction directly in `fase_emit_final_step` without touching the recipe.  Works but couples stmt_fase.rs to recipe internals ("ops[0..=1] are m/v updates, insert bias correction after").  New UpdateOp variants are more declarative and keep the recipe a single source of truth for the optimizer math.
- Precompute `m_hat`/`v_hat` as tensor copies in the recipe as plain `ScalarMulAdd` with `a=bc_inv, b_src=None`.  Doesn't work because `a` in ScalarMulAdd is a compile-time f64 constant; bc_inv is a runtime scalar (depends on `t`).

### D4. Recipe shape after the change

`emit_adamw`'s ops become:

```
1. m = β₁·m + (1-β₁)·m_partial                  # writes M (persistent state)
2. v = β₂·v + (1-β₂)·m_partial²                # writes V (persistent state)
3. m_hat = m * bc1_inv                          # ScalarMulByBc{MHat, M, Beta1}
4. v_hat = v * bc2_inv                          # ScalarMulByBc{VHat, V, Beta2}
5. tmp = sqrt(v_hat) + eps                      # SqrtPlusEps{Tmp, VHat, eps}
6. tmp = m_hat / tmp                            # Div{Tmp, MHat, Tmp}
7. θ -= lr · (tmp + wd·θ)                      # Update{lr, wd, Tmp}
```

Steps 1-2 are unchanged; they still write persistent state.  Steps 5-6 now read `VHat`/`MHat` (the bias-corrected views) instead of `V`/`M` directly.  This also fixes a latent bug: the previous recipe wrote `sqrt(v)` into whatever v-derived buffer was in scope.  With the fix in Item #2 Task 4 we already avoid mutating persistent v; now with bias correction, v_hat is a fresh owned tensor so mutation is trivially safe.

### D5. Step counter source

`stmt.rs` already has `step_count_var` at line ~3337 — an i64 runtime var that counts micro-batches (used by the accumulation gate and by the existing FullBuffer path's t-value).  At the point the optimizer fires (`(step_count + 1) % N == 0`), the optimizer step `t = (step_count + 1) / N` is an exact 1-indexed count.  Compute `t` via Cranelift integer arithmetic; pass as i64 to `nsl_bias_correction_inv`.

No new state needed in the train block.

## Architecture

### Components touched

| File | Change |
|---|---|
| `crates/nsl-runtime/src/fase_bc.rs` (new) | Add `nsl_bias_correction_inv` FFI (~15 LOC). |
| `crates/nsl-runtime/src/lib.rs` | `mod fase_bc;` + re-export. |
| `crates/nsl-codegen/src/fase_optimizer.rs` | Add `Register::MHat`, `Register::VHat`; add `BcKind`, `UpdateOp::ScalarMulByBc`; update `emit_adamw` ops; update tests. |
| `crates/nsl-codegen/src/stmt_fase.rs` | Update `fase_emit_final_step` signature to accept `bc_params: Option<(Value, Value)>`; handle new Registers + `ScalarMulByBc`; free MHat/VHat at end. |
| `crates/nsl-codegen/src/stmt.rs` | In the Deferred optimizer branch, compute `opt_step`, call `nsl_bias_correction_inv` twice, pass results to `fase_emit_final_step`. |
| `crates/nsl-codegen/tests/fase_numerical_validation.rs` | Restore bias-correction division in `adamw_fase_deferred_reference`. |

### Data flow (one optimizer step)

```
stmt.rs (Deferred branch, on final micro-batch)
  │
  ├─ opt_step = (step_count + 1) / grad_accumulation_steps     [i64 Cranelift]
  ├─ bc1_inv = nsl_bias_correction_inv(β₁, opt_step)           [f64 FFI call]
  ├─ bc2_inv = nsl_bias_correction_inv(β₂, opt_step)           [f64 FFI call]
  │
  └─ for each parameter i:
        theta, m, m_partial, v = resolve state
        fase_emit_final_step(..., bc_params = Some((bc1_inv, bc2_inv)))
           │
           └─ processes recipe ops in order:
                op 1 (ScalarMulAdd, M): update m in place
                op 2 (SquaredAccumulate, V): update v in place
                op 3 (ScalarMulByBc, MHat, M, Beta1):
                   MHat = nsl_tensor_mul_scalar(M, bc1_inv, flags=0)  [owned alloc]
                op 4 (ScalarMulByBc, VHat, V, Beta2):
                   VHat = nsl_tensor_mul_scalar(V, bc2_inv, flags=0)  [owned alloc]
                op 5 (SqrtPlusEps, Tmp, VHat)
                op 6 (Div, Tmp, MHat, Tmp)
                op 7 (Update, lr, wd, Tmp)
              free MHat, VHat
              zero m_partial for next window
```

## Risks

1. **Register enum churn.** Adding `MHat` / `VHat` means adding arms to every `match` on `Register` in `fase_optimizer.rs` and `stmt_fase.rs`.  Rustc's exhaustiveness checker catches missed arms; implementation plan runs `cargo build` after each small change.
2. **SGD / SGD-momentum / Lion paths.** These recipes do not emit `ScalarMulByBc` and `fase_emit_final_step` is called with `bc_params = None` for them.  Need to verify non-AdamW tests (Task 3 SGD) still pass.  Easy regression gate.
3. **Tolerance after fix.** Task 4's `adamw_fase_deferred_pipeline_equivalence` test currently passes at rel_err ≈ 3.6e-6.  With bias correction added to both the emitter and the reference, the arithmetic path changes; tolerance may need to stay at 1e-5 (spec's original target) or loosen slightly.  Re-measure during implementation.
4. **Jensen fence untouched.** Task 2's Jensen tests measure the v-approximation, not bias correction.  They must still pass verbatim.  Regression gate.

## Success Criteria

- `cargo test -p nsl-codegen` passes, including:
  - All three Item #2 tests (SGD, AdamW pipeline, Jensen fence).
  - All pre-existing `fase::tests`, `fase_optimizer::tests`, `stmt_fase::tests`, snapshot tests.
- `adamw_fase_deferred_pipeline_equivalence` test's Rust reference includes the `(1 - β^t)` bias-correction divisions, and the test passes at `rel_err < 1e-5`.
- `sgd_exact_equivalence` still passes (SGD doesn't use bias correction, so no change expected).
- A user-written train block with `AdamW(lr=..., beta1=..., beta2=...)` produces numerically similar parameter trajectories at `grad_accumulation=1` and `grad_accumulation=4` (up to the Jensen approximation, which diverges by `O((1-β₂)·Var(g))` per step — small).
