# FASE Two-Phase Gradient Clip Codegen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the temporary `grad_clip → FullBuffer` downgrade from item #1 and emit the two-phase clipped backward described in CFTP §2.5, so FASE Deferred handles `train(grad_clip=...)` inline.

**Architecture:** Two new runtime FFIs (`nsl_tensor_sum_sq`, `nsl_tensor_mul_scalar_inplace`) give codegen the primitives needed to compute a global L2 norm and scale `m_partial` in place. The planner's existing `two_phase_clip` flag becomes real. `stmt.rs` adds an `if fase_plan.two_phase_clip` branch that fuses accumulation with sum-of-squares in Phase A, computes the clip factor as a scalar, then runs a Phase B loop that scales `m_partial` in place and invokes the existing `fase_emit_final_step` per parameter. Non-FASE and non-clipped paths are untouched.

**Tech Stack:** Rust, Cranelift IR, existing `nsl_tensor_*` runtime helpers.

**Spec:** [docs/superpowers/specs/2026-04-14-fase-two-phase-grad-clip-design.md](../specs/2026-04-14-fase-two-phase-grad-clip-design.md)

---

## Task 1: `nsl_tensor_sum_sq` runtime helper

Compute `Σ x²` across a tensor, return as f64. Mirror the GPU→CPU transfer logic from `nsl_clip_grad_norm`. This is the primitive Phase A needs.

**Files:**
- Modify: `crates/nsl-runtime/src/tensor/mod.rs` — add helper adjacent to `nsl_clip_grad_norm` (around line 1373)

### Steps

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block at the bottom of `crates/nsl-runtime/src/tensor/mod.rs` (or wherever the file's test module lives — grep `#\[cfg(test)\]` if unclear):

```rust
#[test]
fn sum_sq_f32_known_values() {
    // Create a 4-element f32 tensor with values [1.0, 2.0, -3.0, 0.5].
    // Σ x² = 1 + 4 + 9 + 0.25 = 14.25.
    let shape_list = super::nsl_list_new();
    super::nsl_list_push_int(shape_list, 4);
    let t = super::nsl_tensor_zeros_f32(shape_list);
    let tensor = super::NslTensor::from_ptr(t);
    unsafe {
        *tensor.data_f32().add(0) = 1.0;
        *tensor.data_f32().add(1) = 2.0;
        *tensor.data_f32().add(2) = -3.0;
        *tensor.data_f32().add(3) = 0.5;
    }
    let got = super::nsl_tensor_sum_sq(t);
    assert!((got - 14.25).abs() < 1e-9, "got {}", got);
    super::nsl_tensor_free(t);
    super::nsl_list_free(shape_list);
}

#[test]
fn sum_sq_of_zero_tensor_is_zero() {
    let shape_list = super::nsl_list_new();
    super::nsl_list_push_int(shape_list, 8);
    let t = super::nsl_tensor_zeros_f32(shape_list);
    let got = super::nsl_tensor_sum_sq(t);
    assert_eq!(got, 0.0);
    super::nsl_tensor_free(t);
    super::nsl_list_free(shape_list);
}
```

If `nsl_tensor_zeros_f32` / `nsl_list_push_int` have different names or signatures (grep `pub extern "C" fn nsl_tensor_zeros` and `pub extern "C" fn nsl_list_push` in `crates/nsl-runtime/src/`), adapt. The goal is: create a small f32 tensor with known values, call `nsl_tensor_sum_sq`, assert the result.

- [ ] **Step 2: Run — expect build failure**

Run: `cargo build -p nsl-runtime`
Expected: FAIL with "function `nsl_tensor_sum_sq` not found".

- [ ] **Step 3: Implement the helper**

Insert the function in `crates/nsl-runtime/src/tensor/mod.rs` immediately before `nsl_clip_grad_norm` (around line 1373). Use the same GPU→CPU transfer style as `nsl_clip_grad_norm`:

```rust
/// Sum of squared elements: `Σ x²`, returned as f64.
///
/// Supports both f32 and f64 dtypes.  GPU tensors are transferred to
/// CPU for the reduction (mirrors `nsl_clip_grad_norm`).  Used by FASE
/// Deferred's two-phase gradient clipping to compute the global L2 norm.
#[no_mangle]
pub extern "C" fn nsl_tensor_sum_sq(tensor_ptr: i64) -> f64 {
    if tensor_ptr == 0 {
        return 0.0;
    }
    let tensor = NslTensor::from_ptr(tensor_ptr);

    // GPU tensors: transfer to CPU for reduction, free the temporary afterward.
    let (cpu_ptr, was_gpu) = if tensor.device > 0 {
        (nsl_tensor_to_device(tensor_ptr, 0), true)
    } else {
        (tensor_ptr, false)
    };

    let cpu_tensor = NslTensor::from_ptr(cpu_ptr);
    let mut acc: f64 = 0.0;
    if cpu_tensor.dtype == 1 {
        for i in 0..cpu_tensor.len as usize {
            let v = unsafe { *cpu_tensor.data_f32().add(i) } as f64;
            acc += v * v;
        }
    } else {
        for i in 0..cpu_tensor.len as usize {
            let v = unsafe { *cpu_tensor.data_f64().add(i) };
            acc += v * v;
        }
    }

    if was_gpu {
        nsl_tensor_free(cpu_ptr);
    }
    acc
}
```

If `nsl_tensor_to_device` or `NslTensor::data_f32` are namespaced differently in this file, match the surrounding conventions.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p nsl-runtime --lib tensor::tests::sum_sq`
Expected: both new tests pass.

- [ ] **Step 5: Confirm no regressions**

Run: `cargo test -p nsl-runtime --lib 2>&1 | grep "^test result" | head -3`
Expected: no pre-existing tests fail.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/tensor/mod.rs
git commit -m "feat(runtime): nsl_tensor_sum_sq for FASE two-phase grad clip"
```

---

## Task 2: `nsl_tensor_mul_scalar_inplace` runtime helper

Scale a tensor in place by an f64. No allocation. Mirrors `nsl_tensor_zero_inplace`'s shape.

**Files:**
- Modify: `crates/nsl-runtime/src/tensor/mod.rs` — add adjacent to `nsl_tensor_zero_inplace`

### Steps

- [ ] **Step 1: Write the failing test**

Append to the same `#[cfg(test)] mod tests` block as Task 1:

```rust
#[test]
fn mul_scalar_inplace_f32_scales_values() {
    let shape_list = super::nsl_list_new();
    super::nsl_list_push_int(shape_list, 3);
    let t = super::nsl_tensor_zeros_f32(shape_list);
    let tensor = super::NslTensor::from_ptr(t);
    unsafe {
        *tensor.data_f32().add(0) = 2.0;
        *tensor.data_f32().add(1) = -1.5;
        *tensor.data_f32().add(2) = 0.0;
    }
    super::nsl_tensor_mul_scalar_inplace(t, 0.5);
    unsafe {
        assert!(((*tensor.data_f32().add(0)) - 1.0).abs() < 1e-6);
        assert!(((*tensor.data_f32().add(1)) - (-0.75)).abs() < 1e-6);
        assert!((*tensor.data_f32().add(2)).abs() < 1e-6);
    }
    super::nsl_tensor_free(t);
    super::nsl_list_free(shape_list);
}

#[test]
fn mul_scalar_inplace_by_zero_clears_tensor() {
    let shape_list = super::nsl_list_new();
    super::nsl_list_push_int(shape_list, 4);
    let t = super::nsl_tensor_zeros_f32(shape_list);
    let tensor = super::NslTensor::from_ptr(t);
    unsafe {
        for i in 0..4 {
            *tensor.data_f32().add(i) = (i + 1) as f32;
        }
    }
    super::nsl_tensor_mul_scalar_inplace(t, 0.0);
    unsafe {
        for i in 0..4 {
            assert_eq!(*tensor.data_f32().add(i), 0.0);
        }
    }
    super::nsl_tensor_free(t);
    super::nsl_list_free(shape_list);
}
```

- [ ] **Step 2: Run — expect build failure**

Run: `cargo build -p nsl-runtime`
Expected: FAIL with "function `nsl_tensor_mul_scalar_inplace` not found".

- [ ] **Step 3: Implement the helper**

Insert in `crates/nsl-runtime/src/tensor/mod.rs` immediately AFTER `nsl_tensor_zero_inplace` (around line 1260, but verify — find the function that looks like `pub extern "C" fn nsl_tensor_zero_inplace` and put this one right below):

```rust
/// Elementwise in-place scale: `tensor *= scalar`.
///
/// No allocation.  GPU tensors are scaled in place on-device via
/// the existing `nsl_tensor_mul_scalar(..., flags=1)` relinquish
/// path; CPU tensors mutate their storage directly.
///
/// Used by FASE Deferred's two-phase gradient clip (Phase B) to
/// apply the global clip factor to each parameter's `m_partial`
/// without allocating a fresh tensor per parameter.
#[no_mangle]
pub extern "C" fn nsl_tensor_mul_scalar_inplace(tensor_ptr: i64, scalar: f64) {
    if tensor_ptr == 0 {
        return;
    }
    let tensor = NslTensor::from_ptr(tensor_ptr);

    if tensor.device > 0 {
        // GPU path: reuse nsl_tensor_mul_scalar with relinquish-a-and-free
        // semantics is not available as an in-place form; fall back to
        // the CPU round-trip for correctness.  If this shows up in a
        // profile, add a dedicated GPU kernel.
        #[cfg(feature = "cuda")]
        {
            let cpu_ptr = nsl_tensor_to_device(tensor_ptr, 0);
            nsl_tensor_mul_scalar_inplace(cpu_ptr, scalar);
            // Copy the scaled CPU values back to the original GPU buffer.
            let cpu_tensor = NslTensor::from_ptr(cpu_ptr);
            crate::cuda::inner::memcpy_htod(
                tensor.data,
                cpu_tensor.data as *const u8,
                (tensor.len as usize) * tensor.element_size(),
            );
            nsl_tensor_free(cpu_ptr);
            return;
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("nsl: mul_scalar_inplace: GPU path requires cuda feature");
            std::process::abort();
        }
    }

    if !tensor.has_writable_storage() {
        eprintln!("nsl: mul_scalar_inplace cannot write into borrowed CPU storage");
        std::process::abort();
    }

    if tensor.dtype == 1 {
        let s = scalar as f32;
        for i in 0..tensor.len as usize {
            unsafe {
                let p = tensor.data_f32().add(i);
                *p = *p * s;
            }
        }
    } else {
        for i in 0..tensor.len as usize {
            unsafe {
                let p = tensor.data_f64().add(i);
                *p = *p * scalar;
            }
        }
    }
}
```

If `has_writable_storage` has a different name, grep for it in surrounding code (`nsl_tensor_zero_inplace` uses the same check). Match the existing signature.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p nsl-runtime --lib tensor::tests::mul_scalar_inplace`
Expected: both new tests pass.

- [ ] **Step 5: Confirm no regressions**

Run: `cargo test -p nsl-runtime --lib 2>&1 | grep "^test result" | head -3`
Expected: no failures.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/tensor/mod.rs
git commit -m "feat(runtime): nsl_tensor_mul_scalar_inplace for FASE two-phase grad clip"
```

---

## Task 3: Register both FFIs in codegen

So codegen can resolve them via `compile_call_by_name`.

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs` around line 251 (after `nsl_tensor_mul_scalar`)

### Steps

- [ ] **Step 1: Locate the insertion point**

Run: `grep -n "\"nsl_tensor_mul_scalar\"\|\"nsl_bias_correction_inv\"" crates/nsl-codegen/src/builtins.rs`

Expected: `nsl_tensor_mul_scalar` at ~line 248, `nsl_bias_correction_inv` added by BC Task 2 nearby.

- [ ] **Step 2: Add both new entries**

Insert adjacent to the other scalar-math FFIs (alongside `nsl_bias_correction_inv`). Match the `types::F64` / `types::I64` / `None` conventions used by surrounding entries:

```rust
// FASE Deferred two-phase grad clip: sum of squared elements, in-place scale.
(
    "nsl_tensor_sum_sq",
    &[types::I64],
    Some(types::F64),
),
(
    "nsl_tensor_mul_scalar_inplace",
    &[types::I64, types::F64],
    None,
),
```

If the surrounding entries use `cl_types::` instead of `types::`, match that style.

- [ ] **Step 3: Build**

Run: `cargo build -p nsl-codegen`
Expected: succeeds.

- [ ] **Step 4: Confirm no lib regression**

Run: `cargo test -p nsl-codegen --lib 2>&1 | grep "^test result" | head -3`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/builtins.rs
git commit -m "feat(codegen): register sum_sq + mul_scalar_inplace FFIs"
```

---

## Task 4: Remove the `grad_clip → FullBuffer` downgrade in the planner

The downgrade added by item #1 Task 5 is now obsolete. Remove it so the planner returns `Deferred` for AdamW/Adam/SGD with `grad_clip.is_some()`.

**Files:**
- Modify: `crates/nsl-codegen/src/fase.rs` — remove lines ~230-246 (the `let (mode, rationale) = if cfg.grad_clip.is_some() && mode == FaseMode::Deferred { ... }` block)

### Steps

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block at the bottom of `crates/nsl-codegen/src/fase.rs`:

```rust
#[test]
fn grad_clip_with_adamw_now_returns_deferred() {
    // Reverses the item #1 Task 5 temporary downgrade.  Item #3 implements
    // two-phase clip emission, so grad_clip + Deferred is now valid.
    let cfg = FaseConfig {
        accumulation: 4,
        optimizer: FaseOptimizer::AdamW,
        grad_clip: Some(1.0),
        allow_v_approx: true,
    };
    let p = plan(&cfg);
    assert_eq!(p.mode, FaseMode::Deferred);
    assert!(p.two_phase_clip);
    assert!(matches!(
        p.backward_phases.last(),
        Some(BackwardPhase::FinalTwoPhase)
    ));
}
```

If `FaseConfig` has more fields, add `..Default::default()`.

- [ ] **Step 2: Run — expect failure**

Run: `cargo test -p nsl-codegen --lib fase::tests::grad_clip_with_adamw_now_returns_deferred`
Expected: FAIL — the downgrade is still in place, `p.mode` is `FullBuffer`.

- [ ] **Step 3: Remove the downgrade**

Open `crates/nsl-codegen/src/fase.rs`. Find the block at approximately lines 232-246 that looks like:

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

Delete this entire block.

- [ ] **Step 4: Update the test that asserted the downgrade**

Earlier in the same `tests` module, there is a test `grad_clip_downgrades_deferred_to_full_buffer_for_now` (from item #1 Task 5). Either rename it and invert the assertion, or delete it (the new test in Step 1 covers the correct behavior):

```bash
grep -n "grad_clip_downgrades_deferred_to_full_buffer_for_now" crates/nsl-codegen/src/fase.rs
```

If present, delete the entire test function. Its assertion is now wrong.

- [ ] **Step 5: Run fase tests**

Run: `cargo test -p nsl-codegen --lib fase::tests`
Expected: all pass, including the new `grad_clip_with_adamw_now_returns_deferred`.

- [ ] **Step 6: Check that the end-to-end AdamW suite still passes**

Run: `cargo test -p nsl-codegen --test fase_numerical_validation -- adamw_fase_deferred_pipeline_equivalence`
Expected: PASS (the existing test doesn't use grad_clip, so removing the downgrade doesn't affect it).

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/fase.rs
git commit -m "feat(fase): remove grad_clip→FullBuffer downgrade now that item #3 is coming"
```

Note: after this commit, any train block with `grad_clip + grad_accumulation > 1` will take the Deferred path but `stmt.rs` doesn't yet handle `two_phase_clip` — that's Task 6. Between Task 4 and Task 6, clipped training is broken. Landing Tasks 5-6 in the same session is expected.

---

## Task 5: Extend `fase_clip.rs` tests for the real ClipPlan shape

`fase_clip.rs` already has ~229 LOC of scaffolding. Add a test asserting the `ClipPlan` construction is wired for two-phase emission. No emission code changes in this task — just verifying the scaffold is ready.

**Files:**
- Modify: `crates/nsl-codegen/src/fase_clip.rs` — append to `#[cfg(test)] mod tests`

### Steps

- [ ] **Step 1: Read the existing fase_clip.rs tests to match style**

Run: `grep -n "^    fn \|#\[test\]" crates/nsl-codegen/src/fase_clip.rs | head -10`

Note what test helpers / fixtures are already in scope. The public API is likely a `ClipPlan::new_l2_global(threshold, eps)` constructor or similar.

- [ ] **Step 2: Write the new test**

Append to `#[cfg(test)] mod tests` at the bottom of `crates/nsl-codegen/src/fase_clip.rs`:

```rust
#[test]
fn two_phase_plan_fuses_accumulation() {
    // For item #3, the two-phase plan must:
    //   - be enabled with the user's threshold
    //   - use L2 global norm (matches non-FASE path)
    //   - have accumulate_during_phase_a=true so Phase A
    //     fuses m_partial accumulation with sum_sq in one loop
    //   - have eps > 0 for numerical stability when grad is zero
    let plan = ClipPlan::new_l2_global(/*threshold=*/ 1.0, /*eps=*/ 1e-6);
    assert!(plan.enabled);
    assert!((plan.threshold - 1.0).abs() < 1e-12);
    assert_eq!(plan.norm, ClipNorm::L2Global);
    assert!(plan.eps > 0.0);
    assert!(
        plan.accumulate_during_phase_a,
        "Phase A must fuse accumulation with sum_sq (plan D2)"
    );
}
```

If `ClipPlan::new_l2_global` doesn't exist — scan the file for whatever constructor IS there (likely `ClipPlan::enabled`, `ClipPlan::l2_global`, or built via struct-literal). Adapt the test to use the real constructor. If the scaffold doesn't provide a convenient constructor, add one as part of this task:

```rust
impl ClipPlan {
    pub fn new_l2_global(threshold: f64, eps: f64) -> Self {
        Self {
            enabled: true,
            threshold,
            norm: ClipNorm::L2Global,
            eps,
            accumulate_during_phase_a: true,
        }
    }
}
```

- [ ] **Step 3: Run the test**

Run: `cargo test -p nsl-codegen --lib fase_clip::tests::two_phase_plan_fuses_accumulation`
Expected: PASS.

- [ ] **Step 4: Run the whole fase_clip test module**

Run: `cargo test -p nsl-codegen --lib fase_clip::tests`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/fase_clip.rs
git commit -m "test(fase): assert ClipPlan fuses accumulation with sum_sq for item #3"
```

---

## Task 6: Emit two-phase dispatch in `stmt.rs`

The core code-change task. In the existing Deferred branch on the final micro-batch, wrap the current per-parameter fused-step loop in an `if fase_plan.two_phase_clip` branch. When true, emit:
- Phase A: fused accumulation + sum_sq loop over params, producing `total_sq` as an f64 Cranelift Value.
- Scalar block: compute `norm = sqrt(total_sq); clip_factor = min(1, τ/(norm + 1e-6))`.
- Phase B: loop over params doing `nsl_tensor_mul_scalar_inplace(m_partial, clip_factor)` then `fase_emit_final_step(...)`.

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs` around lines 4260-4300 (the existing Deferred final-step dispatch block)

### Steps

- [ ] **Step 1: Re-read the existing Deferred dispatch**

Run: `sed -n '4260,4310p' crates/nsl-codegen/src/stmt.rs`

Note the block structure: after the bias-correction scalar computation (`bc1_inv`, `bc2_inv`), there's a runtime loop (`fs_hdr`/`fs_body`/`fs_exit` blocks) that iterates `num_params_val` and calls `fase_emit_final_step` per parameter.

Task 6 wraps this loop in an `if fase_plan.two_phase_clip` branch. In the clipped branch, we have TWO runtime loops (Phase A then Phase B); in the non-clipped branch, the existing single loop stays.

- [ ] **Step 2: Write a smoke-ish failing test via fixture**

The simplest forcing function is Task 7's numerical test (which fails until Task 6 emits correct IR). To verify the Task 6 change compiles and doesn't regress un-clipped paths, add a minimal smoke assertion to the existing smoke-test file. In `crates/nsl-codegen/tests/fase_deferred_smoke.rs`, append:

```rust
#[test]
fn fase_deferred_compiles_adamw_with_grad_clip() {
    use std::process::Command;
    let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fase_deferred_grad_accum_4_clipped.nsl");

    // Fixture not created yet — skip if missing (Task 7 creates it).
    if !fixture.exists() {
        eprintln!("skipping: fixture not yet present");
        return;
    }

    let status = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("check")
        .arg(&fixture)
        .status()
        .expect("failed to spawn nsl check");
    assert!(status.success(), "nsl check failed on clipped fixture");
}
```

If the smoke test file already has similar test helpers, reuse their pattern. The "skip if missing" fallback is so Tasks 6 and 7 can land in either order without breaking the suite.

- [ ] **Step 3: Add the two-phase branch**

Open `crates/nsl-codegen/src/stmt.rs`. Find the block that starts around line 4270:

```rust
                // Per-parameter fused final step (Deferred mode).
                // accum_list is m_partial.  state_list_1 = m, state_list_2 = v.
                let fs_i_var = state.new_variable();
                // ... existing loop emission ...
```

Replace the contents of the outer `if let Some(accum) = accum_list { ... }` body with the following structure. Preserve the bc_inv scalar computation (lines ~4265-4284 from item #1 + BC follow-up). Add the new branch BELOW the scalar computation:

```rust
                // ── existing bc_inv computation (unchanged) ──
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

                if fase_plan.two_phase_clip {
                    // ─── FASE Deferred + two-phase clip (item #3) ───
                    //
                    // Phase A: fused accumulation + sum_sq loop.
                    //   total_sq = 0
                    //   for k in 0..num_params:
                    //       m_partial[k] += g[k]/N   (fase_emit_accumulate)
                    //       total_sq += nsl_tensor_sum_sq(m_partial[k])
                    //       free g[k]
                    //
                    // NOTE: This replaces the standard accumulation loop
                    // at lines ~4183-4220.  When two_phase_clip is true,
                    // the standard accumulation loop MUST be skipped on
                    // the final micro-batch — see Step 4.

                    let pa_tot_var = state.new_variable();
                    builder.declare_var(pa_tot_var, cl_types::F64);
                    let pa_zero = builder.ins().f64const(0.0);
                    builder.def_var(pa_tot_var, pa_zero);

                    let pa_i_var = state.new_variable();
                    builder.declare_var(pa_i_var, cl_types::I64);
                    let pa_i_zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(pa_i_var, pa_i_zero);

                    let pa_hdr = builder.create_block();
                    let pa_body = builder.create_block();
                    let pa_exit = builder.create_block();
                    builder.ins().jump(pa_hdr, &[]);
                    builder.switch_to_block(pa_hdr);
                    let pa_i = builder.use_var(pa_i_var);
                    let pa_cont =
                        builder.ins().icmp(IntCC::SignedLessThan, pa_i, num_params_val);
                    builder.ins().brif(pa_cont, pa_body, &[], pa_exit, &[]);
                    builder.switch_to_block(pa_body);
                    builder.seal_block(pa_body);

                    let pa_mpart =
                        self.compile_call_by_name(builder, "nsl_list_get", &[accum, pa_i])?;
                    let pa_grad = self
                        .compile_call_by_name(builder, "nsl_list_get", &[grads_list, pa_i])?;
                    self.fase_emit_accumulate(
                        builder,
                        pa_mpart,
                        pa_grad,
                        fase_plan.recipe.accum_scale,
                    )?;
                    let pa_sq = self.compile_call_by_name(
                        builder,
                        "nsl_tensor_sum_sq",
                        &[pa_mpart],
                    )?;
                    let pa_tot_cur = builder.use_var(pa_tot_var);
                    let pa_tot_new = builder.ins().fadd(pa_tot_cur, pa_sq);
                    builder.def_var(pa_tot_var, pa_tot_new);
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[pa_grad])?;
                    let pa_i_next = builder.ins().iadd_imm(pa_i, 1);
                    builder.def_var(pa_i_var, pa_i_next);
                    builder.ins().jump(pa_hdr, &[]);

                    builder.switch_to_block(pa_exit);
                    builder.seal_block(pa_hdr);
                    builder.seal_block(pa_exit);

                    // Scalar: clip_factor = min(1, τ / (sqrt(total_sq) + 1e-6))
                    let total_sq = builder.use_var(pa_tot_var);
                    let norm = builder.ins().sqrt(total_sq);
                    let eps = builder.ins().f64const(1e-6);
                    let denom = builder.ins().fadd(norm, eps);
                    let tau = builder.ins().f64const(grad_clip);
                    let ratio = builder.ins().fdiv(tau, denom);
                    let one_f = builder.ins().f64const(1.0);
                    let clip_factor = builder.ins().fmin(one_f, ratio);

                    // Phase B: scale m_partial in place, then fused optimizer step.
                    let pb_i_var = state.new_variable();
                    builder.declare_var(pb_i_var, cl_types::I64);
                    let pb_i_zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(pb_i_var, pb_i_zero);

                    let pb_hdr = builder.create_block();
                    let pb_body = builder.create_block();
                    let pb_exit = builder.create_block();
                    builder.ins().jump(pb_hdr, &[]);
                    builder.switch_to_block(pb_hdr);
                    let pb_i = builder.use_var(pb_i_var);
                    let pb_cont =
                        builder.ins().icmp(IntCC::SignedLessThan, pb_i, num_params_val);
                    builder.ins().brif(pb_cont, pb_body, &[], pb_exit, &[]);
                    builder.switch_to_block(pb_body);
                    builder.seal_block(pb_body);

                    let pb_mpart =
                        self.compile_call_by_name(builder, "nsl_list_get", &[accum, pb_i])?;
                    self.compile_call_by_name(
                        builder,
                        "nsl_tensor_mul_scalar_inplace",
                        &[pb_mpart, clip_factor],
                    )?;
                    let pb_theta = self
                        .compile_call_by_name(builder, "nsl_list_get", &[param_list, pb_i])?;
                    let pb_m = self
                        .compile_call_by_name(builder, "nsl_list_get", &[state_list_1, pb_i])?;
                    let pb_v = self
                        .compile_call_by_name(builder, "nsl_list_get", &[state_list_2, pb_i])?;
                    self.fase_emit_final_step(
                        builder,
                        pb_theta,
                        pb_m,
                        pb_mpart,
                        pb_v,
                        &fase_plan.recipe,
                        Some((bc1_inv, bc2_inv)),
                    )?;

                    let pb_i_next = builder.ins().iadd_imm(pb_i, 1);
                    builder.def_var(pb_i_var, pb_i_next);
                    builder.ins().jump(pb_hdr, &[]);

                    builder.switch_to_block(pb_exit);
                    builder.seal_block(pb_hdr);
                    builder.seal_block(pb_exit);
                } else {
                    // ─── FASE Deferred, no clip (unchanged from item #1 + BC) ───
                    let fs_i_var = state.new_variable();
                    // ... existing loop code, identical to what was there before ...
                }
```

**Verify the identifier names in scope by grepping first:** `state_list_1`, `state_list_2`, `param_list`, `grads_list`, `num_params_val`, `grad_clip` (f64 local). If any differ, adapt the code above.

For the `else` branch, **copy the existing loop body byte-for-byte** from the current implementation — don't rewrite it. The easiest way is: cut-paste the existing `let fs_i_var = ...` through `builder.seal_block(fs_exit);` into the else arm without modification.

- [ ] **Step 4: Skip the standard accumulation loop when `two_phase_clip`**

The existing accumulation loop at lines ~4183-4220 does `m_partial += g/N` and frees g. When `two_phase_clip` is active, Phase A above ALSO does this — we must not double-accumulate.

Find the existing loop (grep for `// 7e3. Gradient accumulation` or similar comment). The loop runs on EVERY micro-batch, not just the final one. For non-final micro-batches (`should_step == false`), it's still needed. For the final micro-batch with `two_phase_clip`, it must be skipped.

The current control flow already branches on `should_step`. The optimizer block (where Task 6's code lives) only runs when `should_step == true`. But the accumulation loop runs BEFORE the `should_step` branch — it runs every micro-batch.

**The fix:** change the existing per-micro-batch accumulation loop to emit only the `fase_emit_accumulate` + `free` calls, and skip them on the final micro-batch when `two_phase_clip` is active, since Phase A re-does them. Concretely, wrap the existing accumulation-loop body in a runtime conditional:

```rust
// At the call-site of the existing loop body (around line 4200-4210):
// Check at RUNTIME whether this is the final micro-batch AND two_phase_clip.
// If so, skip accumulation here — Phase A will do it.
```

Actually the simpler approach: emit the accumulation loop unconditionally on micro-batches 0..N-2 but also on N-1, AND also have Phase A do it again on N-1. That's a double-apply — WRONG.

Correct approach: gate the accumulation loop's body on `!(is_final && two_phase_clip)`. `is_final` is known at runtime as `should_step == true`; `two_phase_clip` is known at COMPILE time (so it's just a Rust `if`).

Restructure:

```rust
if !fase_plan.two_phase_clip {
    // Unchanged: per-micro-batch accumulation loop runs every micro-batch.
    // ... existing loop ...
} else {
    // Two-phase clip: accumulation on micro-batches 0..N-2 only.
    // The final micro-batch (should_step == true) skips this loop because
    // Phase A in the optimizer block re-does accumulation + sum_sq.
    // Runtime gate: if !should_step { run loop } else { skip }.

    let skip_block = builder.create_block();
    let run_block = builder.create_block();
    let accum_done = builder.create_block();
    builder.ins().brif(should_step, skip_block, &[], run_block, &[]);

    builder.switch_to_block(run_block);
    builder.seal_block(run_block);
    // ... existing per-micro-batch accumulation loop body ...
    builder.ins().jump(accum_done, &[]);

    builder.switch_to_block(skip_block);
    builder.seal_block(skip_block);
    builder.ins().jump(accum_done, &[]);

    builder.switch_to_block(accum_done);
    builder.seal_block(accum_done);
}
```

`should_step` is already computed around line 4236 for the gate that triggers the optimizer block. Reuse that value.

- [ ] **Step 5: Build**

Run: `cargo build -p nsl-codegen`
Expected: succeeds. Address any block-sealing or variable-in-scope errors.

- [ ] **Step 6: Run the full suite**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | head -15`
Expected:
- `sgd_exact_equivalence`: PASS (no clip used).
- `adamw_fase_deferred_pipeline_equivalence`: PASS (no clip used).
- Jensen fence × 2: PASS.
- `fase_deferred_compiles_adamw_with_grad_clip` smoke test: SKIP (fixture not yet present — Task 7).
- All `fase`, `fase_optimizer`, `fase_clip`, `stmt_fase` unit tests: PASS.

If any existing test fails, the `should_step` gating probably has a bug — review Step 4. The unclipped paths must be byte-identical in IR to before this commit (since the else branch is unchanged code).

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs crates/nsl-codegen/tests/fase_deferred_smoke.rs
git commit -m "feat(fase): emit two-phase clip dispatch in stmt.rs Deferred branch"
```

---

## Task 7: End-to-end AdamW + grad_clip numerical test

Fixture + reference + assertion. Matches item #2 Task 4's pattern exactly, but with `grad_clip = 0.01` (chosen so clipping actually fires — the un-clipped gradients will have norm ~4-8, well above 0.01).

**Files:**
- Create: `crates/nsl-codegen/tests/fixtures/fase_deferred_grad_accum_4_clipped.nsl`
- Modify: `crates/nsl-codegen/tests/fase_numerical_validation.rs` — append reference + test

### Steps

- [ ] **Step 1: Create the fixture**

Create `crates/nsl-codegen/tests/fixtures/fase_deferred_grad_accum_4_clipped.nsl`:

```nsl
from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()

let x = ones([4, 2])
let y = zeros([4, 1])

# 12 epochs × 1 step = 12 micro-batches, 3 optimizer windows.
# grad_clip = 0.01 — small enough that clipping fires every window.
train(model = m, epochs = 12, grad_accumulation = 4, grad_clip = 0.01):
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

model_save(m, "adamw_clipped_out.nslm")
```

Verify that the train-block parser accepts `grad_clip` as a keyword argument. Run: `grep -n "grad_clip" crates/nsl-codegen/src/stmt.rs | head`. Around line 3030 there should already be `"grad_clip" => { ... }` in the config-args loop.

- [ ] **Step 2: Verify the fixture compiles + runs**

Run from a tempdir:

```bash
mkdir -p /tmp/fase-clip-smoke
cd /tmp/fase-clip-smoke
cargo run --manifest-path=c:/Users/bwiem/projects/NSL/.worktrees/fase-deferred/Cargo.toml -p nsl-cli --bin nsl -- run c:/Users/bwiem/projects/NSL/.worktrees/fase-deferred/crates/nsl-codegen/tests/fixtures/fase_deferred_grad_accum_4_clipped.nsl
```

Expected: exit 0, `adamw_clipped_out.nslm` in cwd. Fix fixture syntax if it fails.

- [ ] **Step 3: Add the reference + test**

Append to `crates/nsl-codegen/tests/fase_numerical_validation.rs`:

```rust
/// Rust reference: FASE-Deferred AdamW with two-phase global-L2 grad clip.
/// Same fixture shape as `adamw_fase_deferred_reference`, plus clipping:
/// compute global L2 norm of m_partial across all parameters, scale by
/// clip_factor = min(1, τ / (norm + 1e-6)).
fn adamw_fase_deferred_clipped_reference(
    w_init: &[f32; 2],
    x: &[[f32; 2]; 4],
    y: &[[f32; 1]; 4],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    tau: f32,
    windows: u32,
) -> [f32; 2] {
    let mut w = *w_init;
    let mut m_state = [0.0_f32; 2];
    let mut v_state = [0.0_f32; 2];

    for step in 1..=windows {
        // Gradient (constant across window).
        let mut pred = [0.0_f32; 4];
        for i in 0..4 {
            pred[i] = x[i][0] * w[0] + x[i][1] * w[1];
        }
        let mut r = [0.0_f32; 4];
        for i in 0..4 {
            r[i] = pred[i] - y[i][0];
        }
        let n = 4.0_f32;
        let mut g = [0.0_f32; 2];
        for j in 0..2 {
            for i in 0..4 {
                g[j] += x[i][j] * r[i];
            }
            g[j] *= 2.0 / n;
        }

        // m_partial = mean(g) = g (constant inputs).
        let mut m_partial = g;

        // Phase A: global L2 norm of m_partial.
        let total_sq: f32 = m_partial.iter().map(|&v| v * v).sum();
        let norm = total_sq.sqrt();
        let clip_factor = 1.0_f32.min(tau / (norm + 1e-6));

        // Phase B: scale, then AdamW step with bias correction.
        for j in 0..2 {
            m_partial[j] *= clip_factor;
            m_state[j] = beta1 * m_state[j] + (1.0 - beta1) * m_partial[j];
            v_state[j] =
                beta2 * v_state[j] + (1.0 - beta2) * m_partial[j] * m_partial[j];
            let bc1 = 1.0 - beta1.powi(step as i32);
            let bc2 = 1.0 - beta2.powi(step as i32);
            let m_hat = m_state[j] / bc1;
            let v_hat = v_state[j] / bc2;
            w[j] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * w[j]);
        }
    }
    w
}

#[test]
fn adamw_deferred_with_grad_clip() {
    let tmp = TempDir::new().expect("tempdir");
    nsl_run(
        &fixture("fase_deferred_grad_accum_4_clipped.nsl"),
        tmp.path(),
    );

    let checkpoint = tmp.path().join("adamw_clipped_out.nslm");
    assert!(
        checkpoint.exists(),
        "expected checkpoint at {:?}",
        checkpoint
    );
    let tensors = read_nslm(&checkpoint).expect("read nslm");

    let w_compiled = tensors
        .get("w")
        .or_else(|| tensors.get("m.w"))
        .expect(&format!(
            "w tensor not in checkpoint; available: {:?}",
            tensors.keys().collect::<Vec<_>>()
        ));
    assert_eq!(w_compiled.len(), 2);

    let w_init = [1.0_f32, 1.0_f32];
    let x = [[1.0, 1.0]; 4];
    let y = [[0.0]; 4];
    let w_ref = adamw_fase_deferred_clipped_reference(
        &w_init,
        &x,
        &y,
        /*lr=*/ 0.001,
        /*beta1=*/ 0.9,
        /*beta2=*/ 0.999,
        /*eps=*/ 1e-8,
        /*wd=*/ 0.01,
        /*tau=*/ 0.01,
        /*windows=*/ 3,
    );

    for i in 0..2 {
        let diff = (w_compiled[i] - w_ref[i]).abs();
        let scale = w_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "AdamW+clip θ[{}] diverged: compiled={} reference={} rel_err={}",
            i,
            w_compiled[i],
            w_ref[i],
            diff / scale
        );
    }
}
```

- [ ] **Step 4: Run the new test**

Run: `cargo test -p nsl-codegen --test fase_numerical_validation -- adamw_deferred_with_grad_clip --nocapture`
Expected: PASS.

If it fails:
- If rel_err is 1e-4 to 5e-5: f32 precision loss through the extra scale+sqrt chain. Widen the tolerance to `1e-4` with a comment.
- If rel_err is >1e-3: bug in emission. Likely culprits: (a) clip_factor using wrong τ value (check `grad_clip` variable name in stmt.rs), (b) `nsl_tensor_mul_scalar_inplace` not actually in-place (verify by reading the runtime helper), (c) Phase A double-counting because the pre-existing accumulation loop wasn't properly gated (Task 6 Step 4).

- [ ] **Step 5: Run the full suite**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | head -15`
Expected: all tests green, no snapshot changes.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/tests/fixtures/fase_deferred_grad_accum_4_clipped.nsl crates/nsl-codegen/tests/fase_numerical_validation.rs
git commit -m "test(fase): AdamW + grad_clip Deferred equivalence test"
```

---

## Task 8: Final verification + memory

- [ ] **Step 1: Full workspace build**

Run: `cargo build --workspace`
Expected: succeeds.

- [ ] **Step 2: Full nsl-codegen test suite**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | head -20`
Expected: all green.

- [ ] **Step 3: Update FASE project memory note**

Open `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_fase_deferred_integration.md`. Find the line:

```
- **Item #3:** two-phase gradient clipping codegen, removing the temporary `grad_clip → FullBuffer` downgrade landed in item #1.
```

Replace with:

```
- **Item #3:** ✅ shipped 2026-04-14 — two-phase clip codegen (Phase A fused accumulation + sum_sq, scalar clip_factor, Phase B scale-in-place + fused optimizer step). New runtime helpers `nsl_tensor_sum_sq` and `nsl_tensor_mul_scalar_inplace`. Spec: `docs/superpowers/specs/2026-04-14-fase-two-phase-grad-clip-design.md`.
```

- [ ] **Step 4: Report**

Summarize: commits shipped, final rel_err for the new test, remaining FASE roadmap items (#4, #5, #6).

---

## Summary of files touched

- **Modified:** `crates/nsl-runtime/src/tensor/mod.rs` (Tasks 1, 2)
- **Modified:** `crates/nsl-codegen/src/builtins.rs` (Task 3)
- **Modified:** `crates/nsl-codegen/src/fase.rs` (Task 4)
- **Modified:** `crates/nsl-codegen/src/fase_clip.rs` (Task 5)
- **Modified:** `crates/nsl-codegen/src/stmt.rs` (Task 6)
- **Modified:** `crates/nsl-codegen/tests/fase_deferred_smoke.rs` (Task 6)
- **Created:** `crates/nsl-codegen/tests/fixtures/fase_deferred_grad_accum_4_clipped.nsl` (Task 7)
- **Modified:** `crates/nsl-codegen/tests/fase_numerical_validation.rs` (Task 7)
- **Modified:** memory note (Task 8)
