# FASE Deferred-Mode Codegen Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `fase::plan()` into the train-block backward emitter so `grad_accumulation > 1` with AdamW/Adam/SGD emits the FASE Deferred-mode rewrite (pre-scaled `m_partial` accumulation + per-parameter fused optimizer step on the final micro-batch).

**Architecture:** The `fase*.rs` modules already produce a `FasePlan` describing the chosen mode. This plan (a) adds a `Register::MPartial` variant so the AdamW recipe can read `m` and `m_partial` as distinct operands, (b) adds a temporary `grad_clip → FullBuffer` downgrade in the planner, (c) introduces a new `stmt_fase.rs` module containing `emit_fase_deferred`, and (d) adds a dispatch site in `stmt.rs` that calls the planner and routes `Deferred` to the new module while leaving `Passthrough` and `FullBuffer` on the existing code path.

**Tech Stack:** Rust, Cranelift IR (`cranelift_frontend::FunctionBuilder`), existing NSL runtime FFI (`nsl_list_new`, `nsl_list_get`, etc.).

**Spec:** [docs/superpowers/specs/2026-04-14-fase-deferred-codegen-integration-design.md](../specs/2026-04-14-fase-deferred-codegen-integration-design.md)

---

## Task 1: Add `Register::MPartial` variant

**Files:**
- Modify: `crates/nsl-codegen/src/fase_optimizer.rs:28-41` (Register enum)
- Modify: `crates/nsl-codegen/src/fase_optimizer.rs` (in-module tests at bottom)

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block at the bottom of `fase_optimizer.rs`:

```rust
#[test]
fn m_partial_is_distinct_from_m() {
    // Register::MPartial must exist as a separate variant so AdamW's
    // final step can read m_old and m_partial simultaneously.
    let m = Register::M;
    let mp = Register::MPartial;
    assert_ne!(m, mp);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests::m_partial_is_distinct_from_m`
Expected: FAIL with "no variant named `MPartial`".

- [ ] **Step 3: Add the variant**

Replace the `Register` enum at `fase_optimizer.rs:28-41` with:

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
    /// Per-micro-batch gradient, live only within one backward-step worth
    /// of register scope.
    G,
    /// Scratch (temporary) register.
    Tmp,
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests::m_partial_is_distinct_from_m`
Expected: PASS.

- [ ] **Step 5: Verify nothing else broke**

Run: `cargo build -p nsl-codegen`
Expected: succeeds (may emit warnings about unused `MPartial` — those go away in Task 2).

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/fase_optimizer.rs
git commit -m "feat(fase): add Register::MPartial variant for deferred-mode accumulator"
```

---

## Task 2: Route the accumulator emitters through `MPartial`

The `emit_accumulate` and `emit_reset` functions currently target `Register::M`, but semantically they operate on the *accumulator* (`m_partial`), not the first-moment (`m`). After this task, those two helpers produce ops against `MPartial`.

**Files:**
- Modify: `crates/nsl-codegen/src/fase_optimizer.rs:124-147` (`emit_accumulate`, `emit_reset`)
- Modify: same file, test module

- [ ] **Step 1: Write the failing test**

Append to the `tests` module:

```rust
#[test]
fn accumulate_targets_m_partial() {
    let recipe = UpdateRecipe {
        optimizer: FaseOptimizer::AdamW,
        lr: 0.001,
        beta1: 0.9,
        one_minus_beta1: 0.1,
        beta2: 0.999,
        one_minus_beta2: 0.001,
        eps: 1e-8,
        weight_decay: 0.01,
        accum_scale: 0.25, // 1/N for N=4
        v_uses_approx: true,
    };
    let prog = emit_accumulate(&recipe);
    // m_partial += 0.25 * g
    let UpdateOp::ScalarMulAdd { dst, src, a, b_src, b_scale } = &prog.ops[0] else {
        panic!("expected ScalarMulAdd, got {:?}", prog.ops[0]);
    };
    assert_eq!(*dst, Register::MPartial);
    assert_eq!(*src, Register::MPartial);
    assert_eq!(*a, 1.0);
    assert_eq!(*b_src, Some(Register::G));
    assert!((b_scale - 0.25).abs() < 1e-12);
}

#[test]
fn reset_zeroes_m_partial() {
    let prog = emit_reset();
    assert_eq!(prog.ops.len(), 1);
    assert_eq!(prog.ops[0], UpdateOp::Zero(Register::MPartial));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests::accumulate_targets_m_partial fase_optimizer::tests::reset_zeroes_m_partial`
Expected: FAIL (both use `Register::M` today).

- [ ] **Step 3: Update `emit_accumulate`**

Replace `fase_optimizer.rs:124-137` with:

```rust
pub fn emit_accumulate(recipe: &UpdateRecipe) -> UpdateProgram {
    let ops = vec![UpdateOp::ScalarMulAdd {
        dst: Register::MPartial,
        src: Register::MPartial,
        a: 1.0,
        b_src: Some(Register::G),
        b_scale: recipe.accum_scale,
    }];
    UpdateProgram {
        optimizer: recipe.optimizer,
        ops,
        pseudocode: format!("m_partial += {} * g", recipe.accum_scale),
    }
}
```

- [ ] **Step 4: Update `emit_reset`**

Replace `fase_optimizer.rs:141-147` with:

```rust
pub fn emit_reset() -> UpdateProgram {
    UpdateProgram {
        optimizer: FaseOptimizer::Unknown,
        ops: vec![UpdateOp::Zero(Register::MPartial)],
        pseudocode: "m_partial = 0".to_string(),
    }
}
```

- [ ] **Step 5: Fix the existing reset test**

There is likely an existing test that asserts `Zero(Register::M)` against `emit_reset()`. Search `fase_optimizer.rs` for `Zero(Register::M)` inside the test module and change it to `Zero(Register::MPartial)`.

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests`
Expected: all tests in the module pass.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/fase_optimizer.rs
git commit -m "fix(fase): route emit_accumulate and emit_reset through MPartial"
```

---

## Task 3: Fix AdamW emitter to consume `MPartial` (and resolve `b_scale`)

AdamW's final step at `fase_optimizer.rs:163-197` reads `m_partial` in two places:
1. The first-moment update `m = β₁·m + (1-β₁)·m_partial` — currently `b_src: Some(Register::M)` with `b_scale: 0.0` (the bug).
2. The second-moment update `v = β₂·v + (1-β₂)·m_partial²` — currently `operand: Register::M` (wrong: it should be the accumulated mean, not the first-moment buffer).

**Files:**
- Modify: `crates/nsl-codegen/src/fase_optimizer.rs:153-206` (`emit_adamw`)
- Modify: test module

- [ ] **Step 1: Write the failing test**

Append to the `tests` module:

```rust
#[test]
fn adamw_reads_m_partial_for_first_and_second_moments() {
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

    // Op 0: m = β₁·m + (1-β₁)·m_partial
    let UpdateOp::ScalarMulAdd { dst, src, a, b_src, b_scale } = &prog.ops[0] else {
        panic!("op 0 should be ScalarMulAdd");
    };
    assert_eq!(*dst, Register::M);
    assert_eq!(*src, Register::M);
    assert!((a - 0.9).abs() < 1e-12);
    assert_eq!(*b_src, Some(Register::MPartial));
    assert!((b_scale - 0.1).abs() < 1e-12, "b_scale must be one_minus_beta1, no extra 1/N");

    // Op 1: v = β₂·v + (1-β₂)·m_partial²
    let UpdateOp::SquaredAccumulate { dst, src, operand, scale } = &prog.ops[1] else {
        panic!("op 1 should be SquaredAccumulate");
    };
    assert_eq!(*dst, Register::V);
    assert_eq!(*src, Register::V);
    assert_eq!(*operand, Register::MPartial);
    assert!((scale - 0.001).abs() < 1e-12);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests::adamw_reads_m_partial_for_first_and_second_moments`
Expected: FAIL (current `b_src` is `Some(Register::M)`, `b_scale` is `0.0`).

- [ ] **Step 3: Fix `emit_adamw`**

Replace the `ops` vector at `fase_optimizer.rs:163-197` with:

```rust
    let ops = vec![
        // m = β₁·m + (1-β₁)·m_partial
        UpdateOp::ScalarMulAdd {
            dst: Register::M,
            src: Register::M,
            a: recipe.beta1,
            b_src: Some(Register::MPartial),
            b_scale: recipe.one_minus_beta1,
        },
        // v = β₂·v + (1-β₂)·m_partial²
        UpdateOp::SquaredAccumulate {
            dst: Register::V,
            src: Register::V,
            operand: Register::MPartial,
            scale: recipe.one_minus_beta2,
        },
        // tmp = sqrt(v) + eps
        UpdateOp::SqrtPlusEps {
            dst: Register::Tmp,
            src: Register::V,
            eps: recipe.eps,
        },
        // tmp = m / tmp
        UpdateOp::Div {
            dst: Register::Tmp,
            src: Register::M,
            divisor: Register::Tmp,
        },
        // θ -= lr * (tmp + wd·θ)
        UpdateOp::Update {
            lr: recipe.lr,
            wd,
            scaled_m: Register::Tmp,
        },
    ];
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests::adamw_reads_m_partial_for_first_and_second_moments`
Expected: PASS.

- [ ] **Step 5: Run the whole fase_optimizer test module**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/fase_optimizer.rs
git commit -m "fix(fase): AdamW emitter reads MPartial with correct one_minus_beta1 scale"
```

---

## Task 4: Fix SGD emitter to consume `MPartial`

SGD is a `Deferred` path per `fase.rs:199-206`. Its emitter currently reads `Register::G`, which is the raw per-micro-batch gradient — correct for the old "re-run optimizer per micro-batch" model but wrong for Deferred, where the caller has finished accumulating and `m_partial` holds the mean gradient.

**Files:**
- Modify: `crates/nsl-codegen/src/fase_optimizer.rs:208-231` (`emit_sgd`)
- Modify: test module

- [ ] **Step 1: Write the failing test**

Append to the `tests` module:

```rust
#[test]
fn sgd_plain_uses_m_partial_in_deferred_mode() {
    let recipe = UpdateRecipe {
        optimizer: FaseOptimizer::Sgd,
        lr: 0.01,
        beta1: 0.0, beta2: 0.0,
        one_minus_beta1: 0.0, one_minus_beta2: 0.0,
        eps: 0.0, weight_decay: 0.0,
        accum_scale: 0.25,
        v_uses_approx: false,
    };
    let prog = emit_sgd(&recipe, /*with_momentum=*/ false);
    // Plain SGD: θ -= lr * m_partial (the accumulated mean gradient)
    match &prog.ops[0] {
        UpdateOp::SgdUpdate { lr } => assert!((lr - 0.01).abs() < 1e-12),
        op => panic!("expected SgdUpdate, got {:?}", op),
    }
}

#[test]
fn sgd_momentum_accumulates_from_m_partial() {
    let recipe = UpdateRecipe {
        optimizer: FaseOptimizer::SgdMomentum,
        lr: 0.01,
        beta1: 0.0, beta2: 0.0,
        one_minus_beta1: 0.0, one_minus_beta2: 0.0,
        eps: 0.0, weight_decay: 0.0,
        accum_scale: 0.25,
        v_uses_approx: false,
    };
    let prog = emit_sgd(&recipe, /*with_momentum=*/ true);
    // m = m + m_partial  (momentum is updated from the accumulated gradient)
    let UpdateOp::ScalarMulAdd { dst, src, b_src, .. } = &prog.ops[0] else {
        panic!("op 0 should be ScalarMulAdd");
    };
    assert_eq!(*dst, Register::M);
    assert_eq!(*src, Register::M);
    assert_eq!(*b_src, Some(Register::MPartial));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests::sgd_plain_uses_m_partial_in_deferred_mode fase_optimizer::tests::sgd_momentum_accumulates_from_m_partial`
Expected: the momentum test fails (currently `b_src: Some(Register::G)`). The plain SGD test already passes (its assertion doesn't depend on a changed register).

- [ ] **Step 3: Fix `emit_sgd`**

Replace `fase_optimizer.rs:208-231` with:

```rust
fn emit_sgd(recipe: &UpdateRecipe, with_momentum: bool) -> UpdateProgram {
    // In Deferred mode the caller has accumulated the mean gradient into
    // m_partial.  Plain SGD applies it directly (θ -= lr·m_partial, via the
    // SgdUpdate op which reads m_partial implicitly).  SGD+momentum folds
    // m_partial into the running momentum buffer first.
    let mut ops = Vec::new();
    if with_momentum {
        ops.push(UpdateOp::ScalarMulAdd {
            dst: Register::M,
            src: Register::M,
            a: 1.0,
            b_src: Some(Register::MPartial),
            b_scale: 1.0,
        });
    }
    ops.push(UpdateOp::SgdUpdate { lr: recipe.lr });
    UpdateProgram {
        optimizer: recipe.optimizer,
        ops,
        pseudocode: if with_momentum {
            format!("m = m + m_partial; θ -= {}·m", recipe.lr)
        } else {
            format!("θ -= {}·m_partial", recipe.lr)
        },
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p nsl-codegen --lib fase_optimizer::tests`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/fase_optimizer.rs
git commit -m "fix(fase): SGD emitter reads MPartial in deferred mode"
```

---

## Task 5: Planner downgrades `Deferred` to `FullBuffer` when `grad_clip` is set

Per spec D-"Non-Goals": this pass skips two-phase clipping. If the user sets `grad_clip`, the planner must fall back to `FullBuffer` instead of producing `Deferred` phases with `FinalTwoPhase`. Item #3 will revert this.

**Files:**
- Modify: `crates/nsl-codegen/src/fase.rs:220-240` (phase/mode selection)
- Modify: `crates/nsl-codegen/src/fase.rs` test module

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` at the bottom of `fase.rs`:

```rust
#[test]
fn grad_clip_downgrades_deferred_to_full_buffer_for_now() {
    let cfg = FaseConfig {
        accumulation: 4,
        optimizer: FaseOptimizer::AdamW,
        grad_clip: Some(1.0),
        allow_v_approx: true,
    };
    let p = plan(&cfg);
    assert_eq!(
        p.mode,
        FaseMode::FullBuffer,
        "pass #1 does not implement two-phase clipping; grad_clip must force FullBuffer"
    );
    assert!(p.rationale.contains("grad_clip"));
}

#[test]
fn grad_clip_absent_keeps_deferred_for_adamw() {
    let cfg = FaseConfig {
        accumulation: 4,
        optimizer: FaseOptimizer::AdamW,
        grad_clip: None,
        allow_v_approx: true,
    };
    let p = plan(&cfg);
    assert_eq!(p.mode, FaseMode::Deferred);
}
```

(If the existing `FaseConfig` struct differs, read `fase.rs:50-100` for its real field list and adapt the literal accordingly.)

- [ ] **Step 2: Run the tests to verify the downgrade test fails**

Run: `cargo test -p nsl-codegen --lib fase::tests::grad_clip_downgrades_deferred_to_full_buffer_for_now`
Expected: FAIL with `p.mode` being `Deferred`.

- [ ] **Step 3: Add the downgrade at the bottom of `plan()`**

Immediately before the final `FasePlan { ... }` construction at `fase.rs:232`, insert:

```rust
    // Pass #1 scope: two-phase gradient clipping is not yet emitted.
    // When the user sets grad_clip, fall back to FullBuffer so the
    // backward emitter keeps the standard accumulation+clip+step path.
    // Tracked as item #3 in the FASE roadmap.
    let (mode, rationale) = if cfg.grad_clip.is_some() && mode == FaseMode::Deferred {
        (
            FaseMode::FullBuffer,
            format!(
                "grad_clip set — two-phase clipping codegen not yet implemented; falling back to FullBuffer ({})",
                rationale
            ),
        )
    } else {
        (mode, rationale)
    };
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p nsl-codegen --lib fase::tests`
Expected: all pass (new ones green, existing ones unaffected unless one asserts `Deferred` with `grad_clip`; fix those by updating the assertion or removing `grad_clip` from the input).

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/fase.rs
git commit -m "feat(fase): planner downgrades grad_clip+Deferred to FullBuffer (temp, item #3)"
```

---

## Task 6: Scaffold `stmt_fase.rs` with the `emit_fase_deferred` stub

Create the new sibling module that will hold the Deferred-mode emitter. Task 7 fills in the per-micro-batch accumulate op; Task 8 fills in the final-step emission. This task just lands the module, the function signature, and a panic body so the dispatch site in Task 9 compiles.

**Files:**
- Create: `crates/nsl-codegen/src/stmt_fase.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` (add `pub mod stmt_fase;`)

- [ ] **Step 1: Create the new module**

Write `crates/nsl-codegen/src/stmt_fase.rs`:

```rust
//! FASE Deferred-mode emission for the train-block backward pass.
//!
//! Wired in from `stmt.rs` when `fase::plan()` returns `FaseMode::Deferred`.
//! Left deliberately thin — `stmt.rs` owns the outer micro-batch loop, the
//! parameter-list traversal, and the existing `accum_list` allocation; this
//! module only replaces:
//!
//!   1. the per-micro-batch accumulator update (was `accum += g`; now
//!      `m_partial += (1/N) * g` via the recipe from `fase_optimizer.rs`),
//!   2. the post-loop optimizer step (was "divide accum by N, run optimizer";
//!      now "run the fused per-parameter recipe, then zero m_partial").
//!
//! The buffer slot is the same allocation `stmt.rs` already makes — this
//! module does not allocate.

use crate::fase::FasePlan;

/// Emit the Deferred-mode per-micro-batch accumulator update.
///
/// Caller contract: `m_partial_buf` and `grad_buf` are runtime pointers to
/// the parameter's accumulator slot and the just-computed gradient buffer,
/// respectively.  After this call, `grad_buf` may be freed by the caller.
///
/// Not yet implemented — returns `Err` until Task 7 lands.
pub fn emit_deferred_accumulate(_plan: &FasePlan) -> Result<(), String> {
    Err("stmt_fase::emit_deferred_accumulate not yet implemented".into())
}

/// Emit the Deferred-mode fused final step (runs after the last micro-batch
/// backward): per-parameter optimizer update sourced from `m_partial`, then
/// `m_partial = 0`.
///
/// Not yet implemented — returns `Err` until Task 8 lands.
pub fn emit_deferred_final_step(_plan: &FasePlan) -> Result<(), String> {
    Err("stmt_fase::emit_deferred_final_step not yet implemented".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fase::{plan, FaseConfig};
    use crate::fase_optimizer::FaseOptimizer;

    #[test]
    fn stubs_return_err_until_implemented() {
        let p = plan(&FaseConfig {
            accumulation: 4,
            optimizer: FaseOptimizer::AdamW,
            grad_clip: None,
            allow_v_approx: true,
        });
        assert!(emit_deferred_accumulate(&p).is_err());
        assert!(emit_deferred_final_step(&p).is_err());
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/nsl-codegen/src/lib.rs`, find the block of `pub mod fase*;` lines (around lines 45-48) and append:

```rust
pub mod stmt_fase;
```

- [ ] **Step 3: Build and run the new test**

Run: `cargo test -p nsl-codegen --lib stmt_fase::tests`
Expected: `stubs_return_err_until_implemented` PASSES.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/stmt_fase.rs crates/nsl-codegen/src/lib.rs
git commit -m "feat(fase): scaffold stmt_fase module with emit_deferred_{accumulate,final_step} stubs"
```

---

## Task 7: Implement `emit_deferred_accumulate` (pre-scaled `m_partial += (1/N)·g`)

This function takes a `FunctionBuilder`, the runtime pointer to the parameter's `m_partial` buffer, and the runtime pointer to the just-computed gradient buffer, and emits an in-place tensor `axpy`-style update. The existing `stmt.rs:4165-4200` accumulation loop uses the runtime helper `nsl_tensor_add_` (in-place add) — the Deferred version uses a scaled variant.

First check whether `nsl_tensor_add_scaled_` (or equivalent) exists.

**Files:**
- Modify: `crates/nsl-codegen/src/stmt_fase.rs`
- Possibly add: runtime helper declaration (see Step 1)

- [ ] **Step 1: Locate the runtime helper**

Run:

```bash
grep -n "nsl_tensor_add\|nsl_tensor_axpy\|nsl_tensor_scale" crates/nsl-runtime/src/lib.rs crates/nsl-runtime/src/tensor.rs 2>/dev/null | head -20
```

Record which helper performs `dst += scale * src` in-place. Two cases:

- **Case A:** A single-call helper like `nsl_tensor_axpy_(dst, src, scale)` exists. Use it directly in Step 3.
- **Case B:** Only `nsl_tensor_add_(dst, src)` and `nsl_tensor_scale(dst, src, scale)` exist. Emit two calls: first compute `tmp = scale * src` via a fresh buffer (allocated by `nsl_zeros_like` then `nsl_tensor_scale_`), then `nsl_tensor_add_(dst, tmp)`, then free `tmp`. This matches the existing accum path's pattern.

Write the name you found into a comment at the top of `stmt_fase.rs`:

```rust
// Runtime helper used for m_partial += scale * g:
//   <Case A or B, exact helper name(s)>
```

- [ ] **Step 2: Write an integration smoke test**

Since this function emits Cranelift IR, a unit test that inspects IR shape is heavyweight. Instead, add a smoke test that compiles a minimal NSL program exercising the code path. Create `crates/nsl-codegen/tests/fase_deferred_smoke.rs`:

```rust
//! Smoke test: FASE Deferred mode compiles a train block with
//! grad_accumulation=4 and AdamW without crashing.  Numerical equivalence
//! is verified in item #2, not here.

use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

#[test]
fn fase_deferred_compiles_adamw_grad_accum_4() {
    let src = fixture("fase_deferred_grad_accum_4.nsl");
    // Use whatever the crate's existing integration tests use to drive the
    // compiler.  If the crate exposes `nsl_codegen::compile_file` or similar,
    // call it here.  If not, shell out to the `nsl` binary.
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("check")
        .arg(&src)
        .status()
        .expect("failed to spawn nsl");
    assert!(status.success(), "nsl check failed on {:?}", src);
}
```

Also create the fixture `crates/nsl-codegen/tests/fixtures/fase_deferred_grad_accum_4.nsl`:

```nsl
# Minimal train block with grad_accumulation=4 to exercise FASE Deferred.
model Tiny:
    w: Tensor<[4, 4], f32>

    fn forward(self, x: Tensor<[*, 4], f32>) -> Tensor<[*, 4], f32>:
        return x @ self.w

fn main():
    let m = Tiny()
    train(model=m, epochs=1, grad_accumulation=4):
        optimizer: AdamW(lr=0.001, weight_decay=0.01)
        step(batch):
            let y = m.forward(batch.input)
            let loss = (y * y).sum()
```

(Adapt syntax if the fixture above doesn't match current NSL surface — reference an existing train-block example under `examples/` or `tests/fixtures/` and copy its structure.)

- [ ] **Step 3: Run the smoke test to verify it fails**

Run: `cargo test -p nsl-codegen --test fase_deferred_smoke`
Expected: FAIL — the dispatch site in `stmt.rs` does not yet exist (Task 9), so either the build fails or the test hits the `Err` returned by the stub.

- [ ] **Step 4: Implement `emit_deferred_accumulate`**

Replace the stub body in `stmt_fase.rs`. The function signature must match what `stmt.rs` can call; follow the same pattern as `stmt.rs:4165-4200`. Exact signature (adjust imports to match the crate's `FunctionBuilder` type alias):

```rust
use cranelift_codegen::ir::Value;
use cranelift_frontend::FunctionBuilder;
use crate::compiler::Compiler;

impl Compiler<'_> {
    /// Emit `m_partial += accum_scale * grad` for a single parameter.
    ///
    /// - `m_partial_ptr`: runtime pointer to the accumulator slot (produced
    ///    by `nsl_list_get(accum_list, i)`).
    /// - `grad_ptr`: runtime pointer to the just-computed gradient.
    /// - `accum_scale`: the recipe's `accum_scale` field (1.0/N).
    ///
    /// After this call, the caller should free `grad_ptr`.
    pub(crate) fn fase_emit_accumulate(
        &mut self,
        builder: &mut FunctionBuilder,
        m_partial_ptr: Value,
        grad_ptr: Value,
        accum_scale: f64,
    ) -> Result<(), String> {
        // Prefer the runtime helper identified in Step 1.  The snippet below
        // uses Case B (scale-then-add); replace with a single axpy call if
        // Case A applied.
        let scale_val = builder.ins().f64const(accum_scale);
        // scaled = zeros_like(grad)
        let scaled = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[grad_ptr])?;
        // scaled = scale * grad          (in-place scale; nsl_tensor_scale_ signature: dst, src, scalar)
        self.compile_call_by_name(builder, "nsl_tensor_scale_", &[scaled, grad_ptr, scale_val])?;
        // m_partial += scaled
        self.compile_call_by_name(builder, "nsl_tensor_add_", &[m_partial_ptr, scaled])?;
        // free the temporary
        self.compile_call_by_name(builder, "nsl_tensor_free", &[scaled])?;
        Ok(())
    }
}
```

**If the exact helper names differ in the runtime**, run this command and substitute:

```bash
grep -n "pub extern \"C\" fn nsl_tensor_" crates/nsl-runtime/src/lib.rs crates/nsl-runtime/src/*.rs 2>/dev/null | head -40
```

Then update the free-function `emit_deferred_accumulate` to delegate:

```rust
pub fn emit_deferred_accumulate(_plan: &FasePlan) -> Result<(), String> {
    // Real emission lives on the Compiler impl (fase_emit_accumulate).
    // This free function stays as a marker hook for future non-Compiler
    // callers (e.g. a test harness).
    Ok(())
}
```

Update the stub test in `stmt_fase.rs` from `assert!(...is_err())` to `assert!(...is_ok())` for `emit_deferred_accumulate`.

- [ ] **Step 5: Run fase_optimizer + stmt_fase unit tests**

Run: `cargo test -p nsl-codegen --lib stmt_fase::tests fase_optimizer::tests`
Expected: all pass. The smoke test in `tests/fase_deferred_smoke.rs` still fails — it needs Tasks 8 and 9.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/stmt_fase.rs crates/nsl-codegen/tests/fase_deferred_smoke.rs crates/nsl-codegen/tests/fixtures/fase_deferred_grad_accum_4.nsl
git commit -m "feat(fase): emit pre-scaled m_partial accumulation op (Compiler::fase_emit_accumulate)"
```

---

## Task 8: Implement `emit_deferred_final_step` (per-parameter fused optimizer step)

This emits, for a single parameter: run the recipe's ops against that parameter's `{θ, m, m_partial, v}` buffers, then zero `m_partial`.

The `UpdateOp` IR from `fase_optimizer.rs` must be lowered to runtime calls. Each variant maps as follows (use `nsl_tensor_*` helper names actually present in the runtime — verify via the same grep as Task 7 Step 1):

| `UpdateOp` variant | Runtime call pattern |
|---|---|
| `Zero(r)` | `nsl_tensor_fill_(reg_ptr(r), 0.0)` |
| `ScalarMulAdd { dst, src, a, b_src, b_scale }` | `dst = a * src + (b_scale * b_src)` — decompose into scale + axpy like Task 7 |
| `Square { dst, src }` | `nsl_tensor_mul_(dst, src, src)` (or equivalent elementwise square) |
| `SquaredAccumulate { dst, src, operand, scale }` | 1) `tmp = operand * operand`, 2) `dst = src + scale * tmp` |
| `SqrtPlusEps { dst, src, eps }` | `nsl_tensor_sqrt_(dst, src); nsl_tensor_add_scalar_(dst, eps)` |
| `Div { dst, src, divisor }` | `nsl_tensor_div_(dst, src, divisor)` |
| `Update { lr, wd, scaled_m }` | AdamW-style fused step — emit `θ -= lr·(scaled_m + wd·θ)`; if a helper like `nsl_tensor_adamw_step_` exists, use it. Otherwise decompose. |
| `SgdUpdate { lr }` | `θ -= lr * m_partial` (axpy with negative scale) |
| `Sign { dst, src }` | Lion only — not reached in Deferred for AdamW/SGD; emit `unimplemented!("Lion not reachable in Deferred")` and leave the arm |

**Files:**
- Modify: `crates/nsl-codegen/src/stmt_fase.rs`

- [ ] **Step 1: Write the failing test (extend the smoke fixture)**

The smoke test from Task 7 already covers this path — it will succeed only once both Tasks 8 AND 9 are in. No new test file needed; the existing smoke test becomes the guard.

- [ ] **Step 2: Implement the per-parameter lowering**

Add to `crates/nsl-codegen/src/stmt_fase.rs` (inside the same `impl Compiler<'_>` block as Task 7). The signature:

```rust
pub(crate) fn fase_emit_final_step(
    &mut self,
    builder: &mut FunctionBuilder,
    theta_ptr: Value,
    m_ptr: Value,
    m_partial_ptr: Value,
    v_ptr: Value,
    recipe: &crate::fase_optimizer::UpdateRecipe,
) -> Result<(), String> {
    use crate::fase_optimizer::{emit_final_step, Register, UpdateOp};

    // Map each symbolic register to a runtime pointer.
    let reg_ptr = |r: Register| -> Value {
        match r {
            Register::Theta    => theta_ptr,
            Register::M        => m_ptr,
            Register::MPartial => m_partial_ptr,
            Register::V        => v_ptr,
            // G is not read in the final step; Tmp must be allocated fresh
            // below.  Panic if the recipe references G here (bug).
            Register::G => panic!("fase final-step recipe must not reference Register::G"),
            Register::Tmp => unreachable!("Tmp is allocated lazily below"),
        }
    };

    // Lazily allocate a scratch tensor the first time Tmp is written.
    let mut tmp_ptr: Option<Value> = None;
    let mut get_tmp = |builder: &mut FunctionBuilder, like: Value| -> Result<Value, String> {
        if let Some(t) = tmp_ptr { return Ok(t); }
        let t = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[like])?;
        tmp_ptr = Some(t);
        Ok(t)
    };

    let program = emit_final_step(recipe);
    for op in &program.ops {
        match op {
            UpdateOp::Zero(r) => {
                let p = reg_ptr(*r);
                let zero = builder.ins().f64const(0.0);
                self.compile_call_by_name(builder, "nsl_tensor_fill_", &[p, zero])?;
            }
            UpdateOp::ScalarMulAdd { dst, src, a, b_src, b_scale } => {
                let dst_p = reg_ptr(*dst);
                let src_p = reg_ptr(*src);
                let a_val = builder.ins().f64const(*a);
                // dst = a * src
                self.compile_call_by_name(builder, "nsl_tensor_scale_", &[dst_p, src_p, a_val])?;
                if let Some(br) = b_src {
                    let b_p = reg_ptr(*br);
                    let bs = builder.ins().f64const(*b_scale);
                    // dst += b_scale * b_src   (axpy)
                    self.compile_call_by_name(builder, "nsl_tensor_axpy_", &[dst_p, b_p, bs])?;
                }
            }
            UpdateOp::Square { dst, src } => {
                let dst_p = reg_ptr(*dst);
                let src_p = reg_ptr(*src);
                self.compile_call_by_name(builder, "nsl_tensor_mul_", &[dst_p, src_p, src_p])?;
            }
            UpdateOp::SquaredAccumulate { dst, src, operand, scale } => {
                let dst_p = reg_ptr(*dst);
                let src_p = reg_ptr(*src);
                let op_p  = reg_ptr(*operand);
                let tmp = get_tmp(builder, op_p)?;
                self.compile_call_by_name(builder, "nsl_tensor_mul_", &[tmp, op_p, op_p])?;
                let sc = builder.ins().f64const(*scale);
                // dst = src + scale * tmp   → copy src into dst then axpy
                self.compile_call_by_name(builder, "nsl_tensor_copy_", &[dst_p, src_p])?;
                self.compile_call_by_name(builder, "nsl_tensor_axpy_", &[dst_p, tmp, sc])?;
            }
            UpdateOp::SqrtPlusEps { dst, src, eps } => {
                let dst_p = reg_ptr(*dst);
                let src_p = reg_ptr(*src);
                let eps_v = builder.ins().f64const(*eps);
                self.compile_call_by_name(builder, "nsl_tensor_sqrt_", &[dst_p, src_p])?;
                self.compile_call_by_name(builder, "nsl_tensor_add_scalar_", &[dst_p, eps_v])?;
            }
            UpdateOp::Div { dst, src, divisor } => {
                let dst_p = reg_ptr(*dst);
                let src_p = reg_ptr(*src);
                let div_p = reg_ptr(*divisor);
                self.compile_call_by_name(builder, "nsl_tensor_div_", &[dst_p, src_p, div_p])?;
            }
            UpdateOp::Update { lr, wd, scaled_m } => {
                let sm_p  = reg_ptr(*scaled_m);
                let lr_v  = builder.ins().f64const(*lr);
                let wd_v  = builder.ins().f64const(*wd);
                // θ -= lr * (scaled_m + wd * θ)
                // = θ = θ*(1 - lr*wd) - lr*scaled_m
                let one_minus = builder.ins().f64const(1.0 - lr * wd);
                self.compile_call_by_name(builder, "nsl_tensor_scale_inplace_", &[theta_ptr, one_minus])?;
                let neg_lr = builder.ins().f64const(-*lr);
                self.compile_call_by_name(builder, "nsl_tensor_axpy_", &[theta_ptr, sm_p, neg_lr])?;
            }
            UpdateOp::SgdUpdate { lr } => {
                let neg_lr = builder.ins().f64const(-*lr);
                // θ -= lr * m_partial
                self.compile_call_by_name(builder, "nsl_tensor_axpy_", &[theta_ptr, m_partial_ptr, neg_lr])?;
            }
            UpdateOp::Sign { .. } => {
                return Err("Sign op reached in Deferred path — Lion should be FullBuffer".into());
            }
        }
    }

    // Zero m_partial for the next accumulation window.
    let zero = builder.ins().f64const(0.0);
    self.compile_call_by_name(builder, "nsl_tensor_fill_", &[m_partial_ptr, zero])?;

    if let Some(t) = tmp_ptr {
        self.compile_call_by_name(builder, "nsl_tensor_free", &[t])?;
    }
    Ok(())
}
```

Update the free-function `emit_deferred_final_step` to return `Ok(())` and update the stub test accordingly.

**Runtime-helper names:** the list above (`nsl_tensor_scale_`, `nsl_tensor_axpy_`, `nsl_tensor_mul_`, `nsl_tensor_sqrt_`, `nsl_tensor_add_scalar_`, `nsl_tensor_div_`, `nsl_tensor_copy_`, `nsl_tensor_fill_`, `nsl_tensor_scale_inplace_`, `nsl_tensor_free`, `nsl_tensor_zeros_like`) is a best guess. Re-grep the runtime crate as in Task 7 Step 1 and **substitute actual names before continuing**. If a helper is missing, the simplest fix is to add it to `nsl-runtime` (thin wrapper around existing math), not to refactor the emitter.

- [ ] **Step 3: Build**

Run: `cargo build -p nsl-codegen`
Expected: compiles. If a runtime helper is missing, the build fails with a linker error pointing to the exact symbol — add a wrapper in `nsl-runtime` and retry.

- [ ] **Step 4: Run unit tests**

Run: `cargo test -p nsl-codegen --lib stmt_fase::tests fase_optimizer::tests fase::tests`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/stmt_fase.rs
git commit -m "feat(fase): emit per-parameter fused final step from UpdateOp recipe"
```

---

## Task 9: Dispatch from `stmt.rs` train-block emitter

This is the integration. At the top of the train-block backward emitter, call `fase::plan()`. On `Deferred`, route per-parameter accumulation through `fase_emit_accumulate` and the post-loop optimizer pass through `fase_emit_final_step`. On `Passthrough`/`FullBuffer`, take the existing code paths untouched.

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs` around lines 3012-4460 (train-block emitter)

- [ ] **Step 1: Locate the five integration points**

Run to confirm the line numbers (they may have drifted):

```bash
grep -n "grad_accumulation_steps\|accum_list\|optimizer_block" crates/nsl-codegen/src/stmt.rs | head -20
```

Expected sites (approximate):
- A. `3012-3040` — parse `grad_accumulation` from train config.
- B. `3329` — allocate `accum_list`.
- C. `~4165-4200` — per-micro-batch accumulate (`accum[i] += grad[i]`).
- D. `~4231-4233` — select optimizer input (`accum` vs `grads_list`).
- E. `~4441-4460` — zero accum after optimizer step.

Record the exact line numbers observed into a scratch comment.

- [ ] **Step 2: Build the `FasePlan` at site A**

Immediately after `grad_accumulation_steps` is parsed, add:

```rust
        // FASE: plan the backward rewrite.  Passthrough (N=1) and FullBuffer
        // (Lion, Unknown, or grad_clip set) fall through to the existing
        // accum-buffer path below.  Deferred routes through stmt_fase.
        let fase_plan = crate::fase::plan(&crate::fase::FaseConfig {
            accumulation: grad_accumulation_steps,
            optimizer: crate::fase::FaseOptimizer::parse(&optimizer_name),
            grad_clip: grad_clip_threshold, // Option<f64> already in scope
            allow_v_approx: true,
        });
        let fase_deferred = fase_plan.mode == crate::fase::FaseMode::Deferred;
```

(If `grad_clip_threshold`'s actual variable name differs, grep for it near the config parse — likely `grad_clip` as an `Option<f64>`.)

- [ ] **Step 3: Per-micro-batch accumulate — site C**

Today this loop does `accum[i] += grad[i]`. Wrap the body with a dispatch:

```rust
        if fase_deferred {
            // m_partial += accum_scale * grad
            self.fase_emit_accumulate(
                builder,
                accum_buf,            // reused as m_partial
                grad_buf,
                fase_plan.recipe.accum_scale,
            )?;
        } else {
            // existing unchanged: accum += grad
            self.compile_call_by_name(builder, "nsl_tensor_add_", &[accum_buf, grad_buf])?;
        }
```

Use the exact local variable names already present in the loop (grep for `accum_buf`, `grads_list`, `gai` inside the existing loop body to confirm).

- [ ] **Step 4: Replace the optimizer step — site D + E**

The existing `optimizer_block` runs the user-level optimizer after the gated `should_step` branch. For `Deferred`, skip that path and instead loop over parameters emitting `fase_emit_final_step`. Structure:

```rust
        if fase_deferred {
            // Per-parameter fused final step: m, m_partial, v are named
            // slots in the optimizer state list (or side buffers).  Grab
            // each parameter's buffers and call fase_emit_final_step.
            // ... runtime loop over num_params_val ...
            for <each param>:
                let theta_ptr     = nsl_list_get(param_list, i);
                let m_ptr         = nsl_list_get(m_state_list, i);
                let m_partial_ptr = nsl_list_get(accum_list, i);
                let v_ptr         = nsl_list_get(v_state_list, i);
                self.fase_emit_final_step(builder, theta_ptr, m_ptr, m_partial_ptr, v_ptr, &fase_plan.recipe)?;
            // note: fase_emit_final_step already zeroes m_partial, so skip
            // the existing accum-zeroing loop at site E.
        } else {
            // existing optimizer_block + accum-zero loop, untouched
        }
```

Fill in the runtime loop using the same `builder.create_block` + `icmp` + `brif` pattern as the existing `accum_list` allocation loop at `stmt.rs:3329-3380`. The exact variable names for `m_state_list` and `v_state_list` depend on how the existing optimizer-state allocation is organized — grep for `m_list`, `v_list`, `optimizer_state` in `stmt.rs` and use the matching names.

**If the optimizer state isn't stored in parallel lists**, the Deferred path needs shape-compatible access: extract the state lookup logic used by the existing `optimizer_block` (how it maps parameter i to its `m` and `v` buffers) and reuse that exact lookup.

- [ ] **Step 5: Build**

Run: `cargo build -p nsl-codegen`
Expected: compiles. Fix any type mismatches inline.

- [ ] **Step 6: Run the full test suite**

Run: `cargo test -p nsl-codegen`
Expected: all pre-existing tests pass; smoke test `fase_deferred_smoke::fase_deferred_compiles_adamw_grad_accum_4` now PASSES.

- [ ] **Step 7: Verify no regression in Passthrough / FullBuffer paths**

Run: `cargo test -p nsl-codegen --test snapshot_tests`
Expected: all snapshots unchanged (the Passthrough and FullBuffer paths are byte-identical to before).

If any snapshot changed, inspect the diff. The only acceptable changes are in train blocks that meet ALL of: `grad_accumulation > 1`, optimizer is `AdamW`/`Adam`/`SGD`/`SGD+momentum`, and `grad_clip` is unset. For other cases the IR must be byte-identical.

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "feat(fase): dispatch Deferred-mode rewrite from train-block emitter"
```

---

## Task 10: Final verification

- [ ] **Step 1: Full workspace build**

Run: `cargo build --workspace`
Expected: succeeds.

- [ ] **Step 2: Full workspace tests**

Run: `cargo test --workspace`
Expected: all pre-existing tests pass; new FASE tests (from Tasks 1-9) pass.

- [ ] **Step 3: Spot-check the dispatch in a real training example**

Find an existing training example with `grad_accumulation` in `examples/` or `models/`:

```bash
grep -rn "grad_accumulation" examples/ models/ 2>/dev/null | head
```

Pick one and run:

```bash
cargo run -p nsl --bin nsl -- check <path>
```

Expected: succeeds. The example's train block now compiles through the Deferred path (confirm by temporarily adding an `eprintln!("fase_deferred={}", fase_deferred);` at site A; remove before commit).

- [ ] **Step 4: Update CLAUDE.md memory note**

Append to `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md` under a new FASE entry, a one-liner pointer:

```
## FASE Deferred Codegen Integration (2026-04-14)
- See [project_fase_deferred_integration.md](project_fase_deferred_integration.md) — Deferred-mode codegen landed; item #2 (numerical equivalence test) next
```

Then create `project_fase_deferred_integration.md` with a short summary of the shipped change and the remaining roadmap items (#1b, #2, #3, #4, #5, #6).

- [ ] **Step 5: Commit memory updates**

```bash
git add C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/
git commit -m "docs(memory): FASE Deferred codegen integration close-out"
```

- [ ] **Step 6: Report**

Summarize what shipped, the test count delta, and the next item (#2: numerical-equivalence test) in a final user-facing message.

---

## Summary of files touched

- **Modified:** `crates/nsl-codegen/src/fase_optimizer.rs` (Tasks 1-4)
- **Modified:** `crates/nsl-codegen/src/fase.rs` (Task 5)
- **Created:** `crates/nsl-codegen/src/stmt_fase.rs` (Tasks 6-8)
- **Modified:** `crates/nsl-codegen/src/lib.rs` (Task 6)
- **Modified:** `crates/nsl-codegen/src/stmt.rs` (Task 9)
- **Created:** `crates/nsl-codegen/tests/fase_deferred_smoke.rs` (Task 7)
- **Created:** `crates/nsl-codegen/tests/fixtures/fase_deferred_grad_accum_4.nsl` (Task 7)
- **Modified:** memory notes (Task 10)
