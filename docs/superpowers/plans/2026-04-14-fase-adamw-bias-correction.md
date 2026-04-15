# FASE AdamW Bias Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `(1 - β^t)` bias correction to FASE Deferred AdamW so `grad_accumulation>1` produces the same trajectory (up to the Jensen v-approximation) as `grad_accumulation=1`, restoring the "grad_accumulation is a compile-time optimization" contract.

**Architecture:** New runtime FFI `nsl_bias_correction_inv(base, step)` computes `1/(1 - base^step)` as a scalar once per optimizer step. `stmt.rs` computes `opt_step` from the existing `step_count_var` and calls the FFI twice (β₁, β₂) to get `bc1_inv` / `bc2_inv` Values. These flow into `fase_emit_final_step` alongside the per-parameter state pointers. Two new `Register` variants (`MHat`, `VHat`) and one new `UpdateOp::ScalarMulByBc` let the recipe express "bias-correct this register into a scratch view" declaratively; the dispatcher lowers each to an `nsl_tensor_mul_scalar` call. Non-AdamW paths pass `bc_params = None` and are unaffected.

**Tech Stack:** Rust, Cranelift IR, existing `nsl_tensor_*` runtime helpers, new FFI with `#[no_mangle] pub extern "C" fn`.

**Spec:** [docs/superpowers/specs/2026-04-14-fase-adamw-bias-correction-design.md](../specs/2026-04-14-fase-adamw-bias-correction-design.md)

---

## Task 1: `nsl_bias_correction_inv` runtime FFI

Scalar helper that computes `1.0 / (1.0 - base^step)`. Exposed as `#[no_mangle] pub extern "C"` so the codegen can invoke it by name.

**Files:**
- Create: `crates/nsl-runtime/src/fase_bc.rs`
- Modify: `crates/nsl-runtime/src/lib.rs` — add `pub mod fase_bc;`

### Steps

- [ ] **Step 1: Write the failing test**

Create `crates/nsl-runtime/src/fase_bc.rs` with only the unit test body (leave the function unimplemented to force a failing build):

```rust
//! FASE bias-correction scalar helper.
//!
//! `nsl_bias_correction_inv(base, step)` returns `1 / (1 - base^step)` —
//! the scalar factor that turns a raw moment into a bias-corrected moment
//! in Adam/AdamW.  Computed once per optimizer step (not per parameter),
//! so a single FFI call per β is negligible cost.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_for_known_step() {
        // β₁=0.9 at step 1 → 1/(1 - 0.9) = 10.0
        let v = nsl_bias_correction_inv(0.9, 1);
        assert!((v - 10.0).abs() < 1e-12, "got {}", v);
    }

    #[test]
    fn matches_reference_for_large_step() {
        // β₂=0.999 at step 100 → 1/(1 - 0.999^100)
        let expected = 1.0 / (1.0 - 0.999_f64.powf(100.0));
        let v = nsl_bias_correction_inv(0.999, 100);
        assert!((v - expected).abs() < 1e-9, "got {} want {}", v, expected);
    }

    #[test]
    fn step_zero_produces_infinity_or_nan_without_panic() {
        // At step 0, 1 - base^0 = 0, so result is +inf.  Not something a
        // caller should do, but the FFI must not panic — it can be invoked
        // from compiled code, and unwinding across FFI is UB.
        let v = nsl_bias_correction_inv(0.9, 0);
        assert!(v.is_infinite() || v.is_nan());
    }
}
```

- [ ] **Step 2: Run the test — expect build failure**

Run: `cargo build -p nsl-runtime`
Expected: FAIL with "function `nsl_bias_correction_inv` not found".

- [ ] **Step 3: Implement the function**

Insert above the `#[cfg(test)]` block in `crates/nsl-runtime/src/fase_bc.rs`:

```rust
/// Compute `1.0 / (1.0 - base^step)` — the bias-correction divisor's
/// inverse, so callers can multiply rather than divide.
///
/// Called once per optimizer step from compiled FASE Deferred code.
/// No unwinding: returns `f64::INFINITY` or `f64::NAN` for degenerate
/// inputs rather than panicking (FFI boundary).
#[no_mangle]
pub extern "C" fn nsl_bias_correction_inv(base: f64, step: i64) -> f64 {
    let exponent = step as f64;
    let denom = 1.0 - base.powf(exponent);
    1.0 / denom
}
```

- [ ] **Step 4: Register the module**

In `crates/nsl-runtime/src/lib.rs`, find the `pub mod` declarations (around lines 6-20) and add:

```rust
pub mod fase_bc;
```

Place it in alphabetical-ish position (e.g., after `pub mod elastic;` if that module exists, or just adjacent to other short-name modules — match whatever the surrounding ordering uses).

- [ ] **Step 5: Run the tests**

Run: `cargo test -p nsl-runtime fase_bc`
Expected: 3 tests pass.

- [ ] **Step 6: Confirm no lib regressions**

Run: `cargo test -p nsl-runtime --lib 2>&1 | tail -5`
Expected: all pre-existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-runtime/src/fase_bc.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(runtime): nsl_bias_correction_inv helper for FASE Deferred AdamW"
```

---

## Task 2: Register `nsl_bias_correction_inv` in codegen FFI table

`stmt.rs` will call this helper via `compile_call_by_name("nsl_bias_correction_inv", ...)`. That lookup resolves through a signature table in `crates/nsl-codegen/src/builtins.rs` around line 230-260.

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs`

### Steps

- [ ] **Step 1: Locate the FFI signature table**

Run:

```bash
grep -n "\"nsl_tensor_mul_scalar\"" crates/nsl-codegen/src/builtins.rs
```

Expected: a match around line 248. The table entries look like:

```rust
(
    "nsl_tensor_mul_scalar",
    &[types::I64, types::F64, types::I8],
    Some(types::I64),
),
```

- [ ] **Step 2: Add the new entry**

Immediately after the `nsl_tensor_mul_scalar` entry (or in a sensible adjacent location — put it near other scalar-math FFIs), add:

```rust
// FASE Deferred bias correction: 1/(1 - base^step).  Scalar, no tensor args.
(
    "nsl_bias_correction_inv",
    &[types::F64, types::I64],
    Some(types::F64),
),
```

- [ ] **Step 3: Build the codegen crate**

Run: `cargo build -p nsl-codegen`
Expected: succeeds. If the table's type macro differs (e.g., `cl_types::F64` vs `types::F64`), match whatever the surrounding entries use.

- [ ] **Step 4: Run the full lib suite to confirm no regression**

Run: `cargo test -p nsl-codegen --lib 2>&1 | tail -3`
Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/builtins.rs
git commit -m "feat(codegen): register nsl_bias_correction_inv FFI signature"
```

---

## Task 3: Extend `Register` and `UpdateOp` enums (with stub handlers)

Add `Register::MHat`, `Register::VHat`, `BcKind`, and `UpdateOp::ScalarMulByBc`. Extend every `match` on these enums so the code compiles, but route the new variants to stub handlers that return `Err(...)`. Nothing emits the new ops yet — that lands in Task 5 — so the stubs are safely dead code.

**Files:**
- Modify: `crates/nsl-codegen/src/fase_optimizer.rs` — add variants, extend pattern matches in emitters
- Modify: `crates/nsl-codegen/src/stmt_fase.rs` — extend the `reg_ptr` resolver and the UpdateOp match

### Steps

- [ ] **Step 1: Write failing tests**

Append to `#[cfg(test)] mod tests` at the bottom of `crates/nsl-codegen/src/fase_optimizer.rs`:

```rust
#[test]
fn m_hat_v_hat_registers_exist_and_are_distinct() {
    let regs = [
        Register::Theta,
        Register::M,
        Register::MPartial,
        Register::V,
        Register::MHat,
        Register::VHat,
        Register::G,
        Register::Tmp,
    ];
    // All distinct.
    for (i, a) in regs.iter().enumerate() {
        for b in regs.iter().skip(i + 1) {
            assert_ne!(a, b, "{:?} and {:?} must be distinct", a, b);
        }
    }
}

#[test]
fn bc_kind_beta1_distinct_from_beta2() {
    assert_ne!(BcKind::Beta1, BcKind::Beta2);
}

#[test]
fn update_op_scalar_mul_by_bc_constructs() {
    let op = UpdateOp::ScalarMulByBc {
        dst: Register::MHat,
        src: Register::M,
        kind: BcKind::Beta1,
    };
    // Pattern-matches as expected.
    match op {
        UpdateOp::ScalarMulByBc { dst, src, kind } => {
            assert_eq!(dst, Register::MHat);
            assert_eq!(src, Register::M);
            assert_eq!(kind, BcKind::Beta1);
        }
        _ => panic!("variant mismatch"),
    }
}
```

- [ ] **Step 2: Run — expect failure**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests::m_hat_v_hat_registers_exist_and_are_distinct`
Expected: FAIL with "no variant or associated item named `MHat`".

- [ ] **Step 3: Add `MHat` and `VHat` to the `Register` enum**

In `crates/nsl-codegen/src/fase_optimizer.rs`, locate the `Register` enum (around line 28-41). Add two new variants after `V`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Register {
    /// Parameter tensor θ.
    Theta,
    /// First-moment state `m` for AdamW / Adam / SGD-momentum / Lion.
    M,
    /// Deferred-mode accumulator: the running mean gradient
    /// `m_partial = (1/N) Σ gᵢ` across the micro-batch window.
    MPartial,
    /// Second-moment accumulator buffer (AdamW only).
    V,
    /// Bias-corrected first moment scratch: `m_hat = m * (1/(1 - β₁^t))`.
    /// Owned tensor allocated lazily in `fase_emit_final_step`; freed at
    /// end of the per-parameter step.
    MHat,
    /// Bias-corrected second moment scratch: `v_hat = v * (1/(1 - β₂^t))`.
    /// Same lifetime as `MHat`.
    VHat,
    /// Per-micro-batch gradient, live only within one backward-step worth
    /// of register scope.
    G,
    /// Scratch (temporary) register.
    Tmp,
}
```

- [ ] **Step 4: Add `BcKind` enum**

In `crates/nsl-codegen/src/fase_optimizer.rs`, immediately after the `Register` enum definition, add:

```rust
/// Which bias-correction base to multiply by.  Identifies one of the
/// two runtime scalars (bc1_inv for β₁, bc2_inv for β₂) that the
/// dispatcher passes to `fase_emit_final_step`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BcKind {
    Beta1,
    Beta2,
}
```

- [ ] **Step 5: Add `UpdateOp::ScalarMulByBc` variant**

In the `UpdateOp` enum (in the same file), add a new variant. Match the style of surrounding variants (camelCase tags, struct-form payloads):

```rust
/// `dst = src * bc_inv[kind]`.  The runtime scalar `bc_inv` is
/// supplied to the emitter as an f64 Cranelift Value by the
/// dispatcher.  Used by AdamW / Adam to compute bias-corrected
/// moment views before the sqrt/div/update ops.
ScalarMulByBc {
    dst: Register,
    src: Register,
    kind: BcKind,
},
```

- [ ] **Step 6: Fix exhaustive matches that reference `Register`**

Rustc will now complain about non-exhaustive `match Register { ... }` in existing code. Search and add arms.

Run: `cargo build -p nsl-codegen 2>&1 | grep -E "non-exhaustive|MHat|VHat" | head -20`

Typical offenders will be in `fase_emit_final_step`'s `reg_ptr` closure in `stmt_fase.rs`. Add arms:

```rust
Register::MHat => /* delegated to lazy_alloc; see below */,
Register::VHat => /* delegated to lazy_alloc; see below */,
```

For now, make these stub arms that return `Err(...)`:

```rust
Register::MHat => Err("Register::MHat not yet handled (Task 4)".into()),
Register::VHat => Err("Register::VHat not yet handled (Task 4)".into()),
```

(If `reg_ptr` doesn't return `Result`, match its actual signature — e.g., `unreachable!("Task 4 implements MHat")`.)

Similarly for matches on `UpdateOp` inside `fase_emit_final_step`, add a stub arm:

```rust
UpdateOp::ScalarMulByBc { .. } => {
    return Err("UpdateOp::ScalarMulByBc not yet handled (Task 4)".into());
}
```

- [ ] **Step 7: Build**

Run: `cargo build -p nsl-codegen`
Expected: succeeds. If new "unused variant" or "unused arm" warnings appear for `MHat`/`VHat`/`BcKind::*`/`ScalarMulByBc`, those are expected and clear in Task 5.

- [ ] **Step 8: Run fase_optimizer tests**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests`
Expected: the 3 new tests pass; all 14 pre-existing fase_optimizer tests still pass.

- [ ] **Step 9: Run the full lib suite**

Run: `cargo test -p nsl-codegen --lib 2>&1 | tail -3`
Expected: all pre-existing tests pass — nothing actually runs the new variants yet.

- [ ] **Step 10: Commit**

```bash
git add crates/nsl-codegen/src/fase_optimizer.rs crates/nsl-codegen/src/stmt_fase.rs
git commit -m "feat(fase): add MHat/VHat/BcKind enums and ScalarMulByBc op (stubs)"
```

---

## Task 4: Implement `ScalarMulByBc` in `fase_emit_final_step`

Teach the dispatcher to (a) receive bias-correction scalars from `stmt.rs`, (b) lazily allocate MHat/VHat tensors via `nsl_tensor_mul_scalar`, (c) free them at end. No caller emits `ScalarMulByBc` yet — `emit_adamw` still emits the old 5-op sequence — so this task's behavior is dead code until Task 5. But all the plumbing lands here, unit-testable through a `bc_params=None` no-op path and (after Task 5) through the end-to-end AdamW test.

**Files:**
- Modify: `crates/nsl-codegen/src/stmt_fase.rs`

### Steps

- [ ] **Step 1: Update `fase_emit_final_step` signature**

In `crates/nsl-codegen/src/stmt_fase.rs`, find the method signature (around line 60) and add a `bc_params` parameter:

```rust
pub(crate) fn fase_emit_final_step(
    &mut self,
    builder: &mut FunctionBuilder,
    theta_ptr: Value,
    m_ptr: Value,
    m_partial_ptr: Value,
    v_ptr: Value,
    recipe: &crate::fase_optimizer::UpdateRecipe,
    bc_params: Option<(Value, Value)>,  // (bc1_inv, bc2_inv) as f64 Cranelift Values
) -> Result<(), String> {
```

- [ ] **Step 2: Update the caller in `stmt.rs` to pass `None` temporarily**

In `crates/nsl-codegen/src/stmt.rs`, find the call site around line 4287 (`self.fase_emit_final_step(builder, theta, m, m_partial, v, &fase_plan.recipe)?;`) and append `, None`:

```rust
self.fase_emit_final_step(
    builder,
    theta,
    m,
    m_partial,
    v,
    &fase_plan.recipe,
    None,  // bc_params supplied in Task 5
)?;
```

- [ ] **Step 3: Add lazy m_hat / v_hat tracking**

Inside `fase_emit_final_step` body (right after the existing `tmp_ptrs` / `tmp_val` tracking setup), add:

```rust
// Lazy MHat / VHat scratch slots for bias-corrected moment views.
// Allocated on first write by `ScalarMulByBc`; freed at end.
let mut m_hat_ptr: Option<Value> = None;
let mut v_hat_ptr: Option<Value> = None;
```

- [ ] **Step 4: Replace the `Register::MHat` / `Register::VHat` stub arms**

Find the `reg_ptr` closure (or whatever resolver Task 3 introduced). Replace the stub arms with real lookups — MHat / VHat are only valid *after* they've been allocated by a `ScalarMulByBc` write:

```rust
Register::MHat => m_hat_ptr.ok_or_else(|| {
    "Register::MHat read before ScalarMulByBc wrote it".to_string()
}),
Register::VHat => v_hat_ptr.ok_or_else(|| {
    "Register::VHat read before ScalarMulByBc wrote it".to_string()
}),
```

If the existing closure is not `FnMut` (can't capture mut references), refactor to a small helper function that takes `&m_hat_ptr` / `&v_hat_ptr` by reference.

- [ ] **Step 5: Replace the `UpdateOp::ScalarMulByBc` stub arm**

Find the stub arm added in Task 3 and replace with:

```rust
UpdateOp::ScalarMulByBc { dst, src, kind } => {
    let (bc1, bc2) = bc_params.ok_or_else(|| {
        "ScalarMulByBc emitted but bc_params is None — dispatcher must supply bias-correction scalars".to_string()
    })?;
    let bc_val = match kind {
        crate::fase_optimizer::BcKind::Beta1 => bc1,
        crate::fase_optimizer::BcKind::Beta2 => bc2,
    };
    let src_ptr = match src {
        Register::M => m_ptr,
        Register::V => v_ptr,
        other => return Err(format!(
            "ScalarMulByBc src must be M or V, got {:?}",
            other
        )),
    };

    // Allocate owned tensor: dst = src * bc_val.
    // flags = 0 forces allocation (no relinquish of src — we must keep
    // persistent m/v buffers intact).
    let flags_zero = builder.ins().iconst(cl_types::I8, 0);
    let out = self.compile_call_by_name(
        builder,
        "nsl_tensor_mul_scalar",
        &[src_ptr, bc_val, flags_zero],
    )?;

    // Free any previous allocation in this slot (shouldn't happen for
    // well-formed recipes, but defensive).
    match dst {
        Register::MHat => {
            if let Some(prev) = m_hat_ptr {
                self.compile_call_by_name(builder, "nsl_tensor_free", &[prev])?;
            }
            m_hat_ptr = Some(out);
        }
        Register::VHat => {
            if let Some(prev) = v_hat_ptr {
                self.compile_call_by_name(builder, "nsl_tensor_free", &[prev])?;
            }
            v_hat_ptr = Some(out);
        }
        other => return Err(format!(
            "ScalarMulByBc dst must be MHat or VHat, got {:?}",
            other
        )),
    }
}
```

- [ ] **Step 6: Free MHat / VHat at the end**

Before `Ok(())` at the bottom of `fase_emit_final_step`, add:

```rust
// Free bias-corrected moment scratches.
if let Some(p) = m_hat_ptr {
    self.compile_call_by_name(builder, "nsl_tensor_free", &[p])?;
}
if let Some(p) = v_hat_ptr {
    self.compile_call_by_name(builder, "nsl_tensor_free", &[p])?;
}
```

Place this adjacent to the existing `tmp_ptrs` free loop if there is one, maintaining whatever ordering the file uses.

- [ ] **Step 7: Update the stmt_fase stub tests**

The existing tests in `stmt_fase.rs`'s `#[cfg(test)] mod tests` block (`accumulate_stub_now_returns_ok`, `final_step_free_function_is_a_marker` — whatever names Task 6/8 of Item #1 left) are unaffected because they exercise the free functions, not the `Compiler` impl method. No change needed. If any test DOES call `fase_emit_final_step` directly, add `None` as the last argument.

- [ ] **Step 8: Build**

Run: `cargo build -p nsl-codegen`
Expected: succeeds.

- [ ] **Step 9: Run the full nsl-codegen test suite**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | tail -15`
Expected: all tests pass, including both integration tests (`sgd_exact_equivalence`, `adamw_fase_deferred_pipeline_equivalence`). The AdamW test still uses the pre-bias-correction behavior because `emit_adamw` hasn't been updated yet (Task 5) — its rel_err stays at ~3.6e-6.

- [ ] **Step 10: Commit**

```bash
git add crates/nsl-codegen/src/stmt_fase.rs crates/nsl-codegen/src/stmt.rs
git commit -m "feat(fase): wire ScalarMulByBc handling through fase_emit_final_step"
```

---

## Task 5: Emit bias-correction ops in `emit_adamw`; compute scalars in `stmt.rs`; restore reference

This is the behavior-changing commit. After this lands, AdamW and Adam produce bias-corrected updates. Three coordinated edits:

1. `emit_adamw` returns a 7-op recipe (adds `ScalarMulByBc` × 2, rewires SqrtPlusEps/Div to read MHat/VHat).
2. `stmt.rs` computes `opt_step`, calls `nsl_bias_correction_inv` twice, passes `Some((bc1_inv, bc2_inv))` to `fase_emit_final_step`.
3. The Rust reference in `fase_numerical_validation.rs` restores its `(1 - β^t)` divisions so the test encodes correct AdamW.

**Files:**
- Modify: `crates/nsl-codegen/src/fase_optimizer.rs` (`emit_adamw` + unit tests)
- Modify: `crates/nsl-codegen/src/stmt.rs` (Deferred dispatch block)
- Modify: `crates/nsl-codegen/tests/fase_numerical_validation.rs` (reference)

### Steps

- [ ] **Step 1: Write a failing recipe-shape test**

Append to `#[cfg(test)] mod tests` in `crates/nsl-codegen/src/fase_optimizer.rs`:

```rust
#[test]
fn adamw_recipe_emits_bias_correction() {
    let recipe = UpdateRecipe {
        optimizer: FaseOptimizer::AdamW,
        lr: 0.001,
        beta1: 0.9,
        one_minus_beta1: 0.1,
        beta2: 0.999,
        one_minus_beta2: 0.001,
        eps: 1e-8,
        weight_decay: 0.01,
        accum_scale: 0.25,
        v_uses_approx: true,
    };
    let prog = emit_adamw(&recipe, /*decoupled_wd=*/ true);

    // 7 ops: m update, v update, m_hat, v_hat, sqrt+eps, div, update.
    assert_eq!(prog.ops.len(), 7, "expected 7 ops, got {:?}", prog.ops);

    // Op 2: m_hat = m * bc1_inv
    match &prog.ops[2] {
        UpdateOp::ScalarMulByBc { dst, src, kind } => {
            assert_eq!(*dst, Register::MHat);
            assert_eq!(*src, Register::M);
            assert_eq!(*kind, BcKind::Beta1);
        }
        other => panic!("op 2 expected ScalarMulByBc, got {:?}", other),
    }

    // Op 3: v_hat = v * bc2_inv
    match &prog.ops[3] {
        UpdateOp::ScalarMulByBc { dst, src, kind } => {
            assert_eq!(*dst, Register::VHat);
            assert_eq!(*src, Register::V);
            assert_eq!(*kind, BcKind::Beta2);
        }
        other => panic!("op 3 expected ScalarMulByBc, got {:?}", other),
    }

    // Op 4: tmp = sqrt(v_hat) + eps
    match &prog.ops[4] {
        UpdateOp::SqrtPlusEps { src, .. } => {
            assert_eq!(*src, Register::VHat, "sqrt must read bias-corrected v");
        }
        other => panic!("op 4 expected SqrtPlusEps, got {:?}", other),
    }

    // Op 5: tmp = m_hat / tmp
    match &prog.ops[5] {
        UpdateOp::Div { src, .. } => {
            assert_eq!(*src, Register::MHat, "div must read bias-corrected m");
        }
        other => panic!("op 5 expected Div, got {:?}", other),
    }
}
```

- [ ] **Step 2: Run — expect failure**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests::adamw_recipe_emits_bias_correction`
Expected: FAIL (current recipe is 5 ops and reads M/V directly).

- [ ] **Step 3: Update `emit_adamw` to emit the 7-op recipe**

Replace the `ops` vector inside `emit_adamw` (around lines 163-197) with:

```rust
    let ops = vec![
        // 0. m = β₁·m_old + (1-β₁)·m_partial     [persistent state update]
        UpdateOp::ScalarMulAdd {
            dst: Register::M,
            src: Register::M,
            a: recipe.beta1,
            b_src: Some(Register::MPartial),
            b_scale: recipe.one_minus_beta1,
        },
        // 1. v = β₂·v_old + (1-β₂)·m_partial²    [persistent state update]
        UpdateOp::SquaredAccumulate {
            dst: Register::V,
            src: Register::V,
            operand: Register::MPartial,
            scale: recipe.one_minus_beta2,
        },
        // 2. m_hat = m * bc1_inv                 [bias-corrected first moment]
        UpdateOp::ScalarMulByBc {
            dst: Register::MHat,
            src: Register::M,
            kind: BcKind::Beta1,
        },
        // 3. v_hat = v * bc2_inv                 [bias-corrected second moment]
        UpdateOp::ScalarMulByBc {
            dst: Register::VHat,
            src: Register::V,
            kind: BcKind::Beta2,
        },
        // 4. tmp = sqrt(v_hat) + ε
        UpdateOp::SqrtPlusEps {
            dst: Register::Tmp,
            src: Register::VHat,
            eps: recipe.eps,
        },
        // 5. tmp = m_hat / tmp
        UpdateOp::Div {
            dst: Register::Tmp,
            src: Register::MHat,
            divisor: Register::Tmp,
        },
        // 6. θ -= lr · (tmp + wd·θ)
        UpdateOp::Update {
            lr: recipe.lr,
            wd,
            scaled_m: Register::Tmp,
        },
    ];
```

Also update the `pseudocode` string construction immediately below the `ops` vec to reflect the new sequence:

```rust
    UpdateProgram {
        optimizer: recipe.optimizer,
        ops,
        pseudocode: format!(
            "m=β₁·m+(1-β₁)·m_partial; v{}=β₂·v+(1-β₂)·m_partial²; m̂=m·bc1_inv; v̂=v·bc2_inv; θ -= lr·(m̂/(√v̂+ε) + wd·θ)",
            if recipe.v_uses_approx { "≈" } else { "=" }
        ),
    }
```

- [ ] **Step 4: Update the pre-existing `adamw_program_has_five_ops` test**

Locate the existing test (from Item #1 Task 3). Rename and update it to reflect the 7-op shape:

```rust
#[test]
fn adamw_program_has_seven_ops_with_bias_correction() {
    let recipe = UpdateRecipe {
        optimizer: FaseOptimizer::AdamW,
        lr: 0.001,
        beta1: 0.9,
        one_minus_beta1: 0.1,
        beta2: 0.999,
        one_minus_beta2: 0.001,
        eps: 1e-8,
        weight_decay: 0.01,
        accum_scale: 0.25,
        v_uses_approx: true,
    };
    let prog = emit_adamw(&recipe, true);
    assert_eq!(prog.ops.len(), 7);
}
```

If there are additional tests that assert specific op indices or specific register reads in the AdamW recipe (e.g., `adamw_reads_m_partial_for_first_and_second_moments` from Item #1), verify they still pass — ops 0 and 1 retain the same MPartial reads, so those assertions are unaffected.

- [ ] **Step 5: Run fase_optimizer tests**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests`
Expected: the new `adamw_recipe_emits_bias_correction` passes; `adamw_program_has_seven_ops_with_bias_correction` passes; `adamw_reads_m_partial_for_first_and_second_moments` passes. Pre-existing Jensen / MPartial / accumulate / reset / SGD / Lion tests all pass.

- [ ] **Step 6: Compute `opt_step` and `bc_inv` scalars in `stmt.rs`**

In the Deferred branch of the optimizer-step emission in `stmt.rs` (around line 4260), BEFORE the per-parameter loop over `fase_emit_final_step`, add:

```rust
// ── FASE Deferred: compute bias-correction scalars once per step ──
// opt_step = (step_count + 1) / grad_accumulation_steps
// bc_inv = nsl_bias_correction_inv(β, opt_step)
let sc_val = builder.use_var(step_count_var);
let one_i64 = builder.ins().iconst(cl_types::I64, 1);
let sc_plus_one = builder.ins().iadd(sc_val, one_i64);
let grad_accum_const = builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
let opt_step = builder.ins().sdiv(sc_plus_one, grad_accum_const);

let beta1_const = builder.ins().f64const(fase_plan.recipe.beta1);
let beta2_const = builder.ins().f64const(fase_plan.recipe.beta2);
let bc1_inv = self.compile_call_by_name(
    builder,
    "nsl_bias_correction_inv",
    &[beta1_const, opt_step],
)?;
let bc2_inv = self.compile_call_by_name(
    builder,
    "nsl_bias_correction_inv",
    &[beta2_const, opt_step],
)?;
```

(Use whatever identifier is already in scope for the recipe — likely `fase_plan.recipe`. Use whatever I64/F64 constant constructor style the surrounding code uses; grep the adjacent `builder.ins().iconst(cl_types::I64, ...)` usage to match.)

- [ ] **Step 7: Pass the scalars to `fase_emit_final_step`**

Update the existing call (the line modified in Task 4 Step 2):

```rust
self.fase_emit_final_step(
    builder,
    theta,
    m,
    m_partial,
    v,
    &fase_plan.recipe,
    Some((bc1_inv, bc2_inv)),
)?;
```

- [ ] **Step 8: Build and run the AdamW integration test with the new emitter but the OLD reference**

Run: `cargo test -p nsl-codegen --test fase_numerical_validation -- adamw_fase_deferred_pipeline_equivalence --nocapture`
Expected: FAIL — the emitter now does bias correction; the reference still doesn't. The observed `compiled` values should differ from `reference` by a factor related to `(1 - β^t)`.

Record the observed rel_err from the panic message — you'll use it to sanity-check Step 9.

- [ ] **Step 9: Restore bias correction in the Rust reference**

Open `crates/nsl-codegen/tests/fase_numerical_validation.rs`. Find `adamw_fase_deferred_reference`. Replace the per-window update block with:

```rust
        for j in 0..2 {
            m_state[j] = beta1 * m_state[j] + (1.0 - beta1) * m_partial[j];
            v_state[j] =
                beta2 * v_state[j] + (1.0 - beta2) * m_partial[j] * m_partial[j];
            let bc1 = 1.0 - beta1.powi(step as i32);
            let bc2 = 1.0 - beta2.powi(step as i32);
            let m_hat = m_state[j] / bc1;
            let v_hat = v_state[j] / bc2;
            w[j] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * w[j]);
        }
```

(Some version of this may already be in place from Task 4 of item #2 — check the current file state.  If the `_hat` variables are present but a subagent stubbed them out, this edit restores them.)

- [ ] **Step 10: Run the AdamW test**

Run: `cargo test -p nsl-codegen --test fase_numerical_validation -- adamw_fase_deferred_pipeline_equivalence --nocapture`

Expected: PASS at `rel_err < 1e-5`.

If the test fails at the tolerance:
- If `rel_err` is close to 1e-5 (say 1e-4), f32 precision may be the issue. Widen tolerance to `1e-4` in the test and add a comment citing the AdamW recipe's sqrt/div chain as the amplifier.
- If `rel_err` is much larger (>1e-3), suspect a real bug in the emission: check that `opt_step` is 1-indexed (not 0), that `bc_inv = 1/(1-β^t)` (not `1-β^t`), and that MHat/VHat are being read in the right ops (step 5 in the IR: `sqrt(VHat)`; step 6: `MHat / ...`).

- [ ] **Step 11: Run the full nsl-codegen suite**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | tail -15`
Expected:
- SGD exact equivalence: PASS (unchanged).
- AdamW pipeline equivalence: PASS.
- Jensen fence × 2: PASS.
- All `fase_optimizer::tests` pass.
- All `stmt_fase::tests` pass.
- All snapshots unchanged (the IR changes for AdamW snapshot fixtures, but no snapshot fixture exists for FASE Deferred AdamW AFAIK — verify with `grep -r "grad_accumulation" crates/nsl-codegen/tests/snapshots/`; if any snapshot exercises the new path, re-baseline it and note which).

- [ ] **Step 12: Commit**

```bash
git add crates/nsl-codegen/src/fase_optimizer.rs crates/nsl-codegen/src/stmt.rs crates/nsl-codegen/tests/fase_numerical_validation.rs
# If any snapshots were re-baselined:
# git add crates/nsl-codegen/tests/snapshots/
git commit -m "fix(fase): add bias correction to AdamW/Adam Deferred emission"
```

---

## Task 6: Final verification + memory

- [ ] **Step 1: Full workspace build**

Run: `cargo build --workspace`
Expected: succeeds (may emit pre-existing warnings unrelated to this work).

- [ ] **Step 2: Full `nsl-codegen` test suite**

Run: `cargo test -p nsl-codegen 2>&1 | tee /tmp/test_output.log | grep "^test result"`
Expected: all green.

- [ ] **Step 3: Update memory note**

Edit `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_fase_deferred_integration.md`. Under the "Follow-up work" section, append a new line:

```
- **AdamW bias correction (2026-04-14):** ✅ shipped — item #2 Task 4 surfaced that `fase_optimizer::emit_adamw` omitted `(1 - β^t)` bias correction, causing `grad_accumulation>1` to diverge from `grad_accumulation=1`. Fixed by adding `Register::MHat`/`VHat`, `UpdateOp::ScalarMulByBc`, and a new runtime helper `nsl_bias_correction_inv`. Spec: `docs/superpowers/specs/2026-04-14-fase-adamw-bias-correction-design.md`.
```

- [ ] **Step 4: Commit memory update**

```bash
git -C C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory add project_fase_deferred_integration.md
git -C C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory commit -m "docs(memory): FASE AdamW bias correction shipped"
```

- [ ] **Step 5: Report**

Summarize: three commits (1 runtime helper, 1 FFI register, 1 enum extension, 1 dispatcher wiring, 1 behavior change + reference restore, 1 memory); test count delta; rel_err observed after bias correction.

---

## Summary of files touched

- **Created:** `crates/nsl-runtime/src/fase_bc.rs` (Task 1)
- **Modified:** `crates/nsl-runtime/src/lib.rs` (Task 1)
- **Modified:** `crates/nsl-codegen/src/builtins.rs` (Task 2)
- **Modified:** `crates/nsl-codegen/src/fase_optimizer.rs` (Tasks 3, 5)
- **Modified:** `crates/nsl-codegen/src/stmt_fase.rs` (Tasks 3, 4)
- **Modified:** `crates/nsl-codegen/src/stmt.rs` (Tasks 4, 5)
- **Modified:** `crates/nsl-codegen/tests/fase_numerical_validation.rs` (Task 5)
- **Modified:** memory note (Task 6)
