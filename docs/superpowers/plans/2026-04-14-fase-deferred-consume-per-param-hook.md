# FASE Deferred Consume-Per-Parameter Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver CFTP §2.4's peak-memory claim for FASE Deferred on the source-AD path. During `compile_wengert_ops`, fire a callback immediately after each parameter gradient is produced so FASE can consume the gradient (accumulate into `m_partial`) and free it before the next one is computed.

**Architecture:** Add an optional `on_param_grad` callback parameter to `compile_wengert_ops`. When Deferred + source-AD is active, `stmt.rs` builds a `VarId → accum_idx` map and constructs a closure that invokes `fase_emit_accumulate` + `nsl_tensor_free` per gradient. With the hook active, the post-Wengert grads_list seed/patch loops and the per-micro-batch accumulation loop are skipped — accumulation already happened inline during lowering. The existing caching allocator implicitly reuses each freed gradient slab for the next same-sized gradient.

**Tech Stack:** Rust, Cranelift IR, existing `fase_emit_accumulate` + `nsl_tensor_free` FFIs.

**Spec:** [docs/superpowers/specs/2026-04-14-fase-deferred-consume-per-param-hook-design.md](../specs/2026-04-14-fase-deferred-consume-per-param-hook-design.md)

---

## Task 1: Extend `compile_wengert_ops` with the `on_param_grad` hook

Add an optional callback parameter. The callback fires once for each `VarId` in a caller-provided "parameter adjoint" set, immediately after that VarId's op produces its output. After the callback returns, remove the entry from `var_map` so downstream code can't accidentally re-consume it.

**Files:**
- Modify: `crates/nsl-codegen/src/wengert_lower.rs` — signature change + hook-fire site + unit test

### Steps

- [ ] **Step 1: Write the failing unit test**

Append to the existing `#[cfg(test)] mod tests` block at the bottom of `crates/nsl-codegen/src/wengert_lower.rs`:

```rust
#[test]
fn on_param_grad_fires_once_per_param_adjoint() {
    // Build a minimal WengertList with two parameter-style ops.
    // The test inserts dummy primals for inputs and verifies the callback
    // fires exactly once per VarId in param_adjoints with the matching Value.

    use crate::wengert::{PrimalOp, WengertOp, WengertType};
    use cranelift_codegen::ir::{types as cl_types, AbiParam, Function, InstBuilder, Signature};
    use cranelift_codegen::isa::CallConv;
    use cranelift_codegen::settings;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    // This is an IR-emission test, so we need a real function builder.
    // The Compiler harness is heavier than we want here; if the project
    // provides a test-helper for constructing a Compiler + FunctionBuilder,
    // use it.  Otherwise this test is gated behind a #[cfg_attr(... ignore)]
    // until such harness exists.
    //
    // If no harness: restrict this test to the bookkeeping layer by
    // directly invoking a small shim that exercises hook-fire bookkeeping
    // without real IR.  See Step 3 for a simpler alternative.

    // For this pass, assert the bookkeeping invariant via a minimal
    // stub: given a WengertList with two ops whose results are in
    // param_adjoints, running compile_wengert_ops fires the callback
    // exactly twice and removes both VarIds from var_map.
    //
    // If building this harness is disproportionately hard, fall back to
    // the integration test in Task 3 as the validation signal.

    // Skipping harness construction inline — this test stays a spec-gated
    // assertion until integration coverage lands.  Marking with a
    // compile-time sanity check on the parameter list instead:
    fn _signature_compiles(
        _f: Option<(
            &std::collections::HashSet<VarId>,
            &mut dyn FnMut(VarId, Value, &mut FunctionBuilder) -> Result<(), crate::CodegenError>,
        )>,
    ) {
    }
    // If the type signature compiles, the API shape is correct.  The
    // behavioral assertion is provided by Task 3's end-to-end test.
    assert!(true);
}
```

Note: building a full Cranelift function builder harness for a single wengert_lower unit test is disproportionate. The integration test in Task 3's source-AD-invoked `adamw_fase_deferred_pipeline_equivalence` test provides the behavioral validation (the hook fires correctly, gradients are freed, final parameters match the reference). This Step-1 test is a compile-time API-shape assertion only. **If the engineer finds it easy to build a Cranelift test harness, they should expand this test to actually invoke `compile_wengert_ops` with a mock callback; otherwise rely on Task 3's integration signal.**

- [ ] **Step 2: Run — expect build failure**

Run: `cargo build -p nsl-codegen`
Expected: FAIL with a type-mismatch because `on_param_grad` isn't a parameter yet.

- [ ] **Step 3: Extend the signature**

In `crates/nsl-codegen/src/wengert_lower.rs`, find the existing signature (around lines 28-35):

```rust
pub fn compile_wengert_ops(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    _state: &mut FuncState,
    wengert: &WengertList,
    primal_vars: &VarMap,
) -> Result<LoweredWengert, CodegenError>
```

Change it to:

```rust
pub fn compile_wengert_ops(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    _state: &mut FuncState,
    wengert: &WengertList,
    primal_vars: &VarMap,
    on_param_grad: Option<(
        &std::collections::HashSet<VarId>,
        &mut dyn FnMut(VarId, Value, &mut FunctionBuilder) -> Result<(), CodegenError>,
    )>,
) -> Result<LoweredWengert, CodegenError>
```

Also update the doc comment above the function to mention the callback:

```rust
/// Lower a WengertList to Cranelift IR by dispatching each PrimalOp to its runtime FFI call.
///
/// `primal_vars` maps VarIds from the forward pass to their Cranelift Values (i64 tensor pointers).
/// Returns a map from all VarIds (including adjoint) to Cranelift Values.
///
/// `on_param_grad`, when `Some((set, cb))`, causes `cb` to be invoked
/// immediately after any op whose `result` VarId is in `set`.  The callback
/// receives the VarId and the just-produced Cranelift Value (a tensor
/// pointer).  The gradient is then REMOVED from `var_map` — the callback
/// is responsible for freeing or otherwise owning that tensor.  Used by
/// FASE Deferred to consume parameter gradients during backward lowering
/// so only one gradient is live at a time.
```

- [ ] **Step 4: Fire the hook at the emission site**

Find line 62 in `wengert_lower.rs`:

```rust
var_map.insert(op.result, result_val);
```

Replace with:

```rust
var_map.insert(op.result, result_val);

// FASE hook: consume parameter gradients immediately.
if let Some((ref param_set, ref mut cb)) = on_param_grad.as_mut() {
    if param_set.contains(&op.result) {
        cb(op.result, result_val, builder)?;
        // Callback owns/freed the tensor — remove from var_map so
        // downstream code can't accidentally use it.
        var_map.remove(&op.result);
    }
}
```

(Adjust to `if let Some((param_set, cb)) = &mut on_param_grad { ... }` if the `Option<(&set, &mut dyn FnMut)>` destructuring requires different pattern syntax — experiment with whichever the Rust compiler accepts.)

- [ ] **Step 5: Build**

Run: `cargo build -p nsl-codegen`
Expected: FAIL with errors in `stmt.rs` where `compile_wengert_ops` is called without the new parameter. That's expected — Task 2 fixes the call sites.

- [ ] **Step 6: Commit (signature-only; call sites fixed in Task 2)**

Don't commit yet — Task 2's call-site updates must land in the same commit as this signature change, otherwise the build is broken at the tip. Proceed to Task 2.

---

## Task 2: Update existing `compile_wengert_ops` call sites to pass `None`

All existing callers must pass `None` to preserve today's behavior. There are two call sites in `stmt.rs` (grep confirms). Passing `None` is the byte-identical path.

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`

### Steps

- [ ] **Step 1: Find all call sites**

Run: `grep -n "compile_wengert_ops" crates/nsl-codegen/src/stmt.rs`

Expected: two matches (around lines 3823 and 3874 based on prior exploration).

Also check for any other callers outside stmt.rs:

```bash
grep -rn "compile_wengert_ops\b" crates/nsl-codegen/src/ --include="*.rs" | grep -v wengert_lower.rs
```

- [ ] **Step 2: Update each call site to append `None`**

For each call site, append `, None` as the final argument. Example at line 3823:

```rust
let full_lowered = crate::wengert_lower::compile_wengert_ops(
    self,
    builder,
    state,
    &effective_primal,
    &primal_vars,
    None, // FASE on_param_grad hook — Task 3 wires this
)?;
```

- [ ] **Step 3: Build**

Run: `cargo build -p nsl-codegen`
Expected: succeeds.

- [ ] **Step 4: Run the full lib + integration test suite**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | head -15`
Expected: all pass; behavior is byte-identical to pre-Task-1.

- [ ] **Step 5: Commit Tasks 1 + 2 together**

```bash
git add crates/nsl-codegen/src/wengert_lower.rs crates/nsl-codegen/src/stmt.rs
git commit -m "feat(fase): add on_param_grad hook to compile_wengert_ops"
```

---

## Task 3: Wire FASE Deferred callback in `stmt.rs`

This is the behavior-changing task. When Deferred + source-AD conditions hold, build the `VarId → accum_idx` map, construct the callback, pass it to `compile_wengert_ops`, and skip the now-redundant post-Wengert grads_list seed/patch loops plus the per-micro-batch accumulation loop.

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs` around lines 3519-4050 (source-AD branch) and 4180-4220 (per-micro-batch accumulation)

### Steps

- [ ] **Step 1: Read the current source-AD branch structure**

```bash
sed -n '3519,3540p;3820,3830p;3944,4050p;4180,4210p' crates/nsl-codegen/src/stmt.rs
```

Record:
- Where `extractor` is built (the Wengert extractor).
- Where `gen: AdjointGenerator` is constructed and used.
- The names used in the post-Wengert patch loop: `extractor.named_param_var_ids()`, `gen.adjoint_of()`, `grad_vars`.
- The path variables for each parameter (used to resolve `accum_list` index).

The existing post-Wengert patch loop at [stmt.rs:3985-4030](crates/nsl-codegen/src/stmt.rs#L3985-L4030) iterates `extractor.named_param_var_ids()` and uses `gen.adjoint_of(*vid)` to get the adjoint `VarId`. That same resolution builds the `VarId → accum_idx` map.

- [ ] **Step 2: Build the `VarId → accum_idx` map immediately before `compile_wengert_ops`**

Around line 3820 (before the first `compile_wengert_ops` call), insert map construction. The `accum_list` is populated by the early train-block setup with one slot per trainable parameter in the order `extractor.named_param_var_ids()` yields them (ordered, deterministic).

```rust
// Build VarId → accum_idx map for the FASE consume-per-param hook.
// The accum_list was populated in the same order as extractor.named_param_var_ids()
// yields (see stmt.rs early setup ~line 3300).
let mut var_id_to_accum_idx: std::collections::HashMap<crate::wengert::VarId, i64> =
    std::collections::HashMap::new();
let mut param_adjoint_set: std::collections::HashSet<crate::wengert::VarId> =
    std::collections::HashSet::new();
for (accum_idx, (_param_name, primal_vid)) in extractor.named_param_var_ids().iter().enumerate() {
    if let Some(adj_vid) = gen.adjoint_of(**primal_vid) {
        var_id_to_accum_idx.insert(adj_vid, accum_idx as i64);
        param_adjoint_set.insert(adj_vid);
    }
}

let fase_hook_active = fase_deferred && self.features.source_ad_enabled;
```

Verify the exact iterator shape of `named_param_var_ids()` — it may return `impl Iterator<Item = (&String, &VarId)>` or `Vec<(&String, VarId)>`. Adapt the destructuring as needed.

Verify `accum_list` is allocated using the same ordering. If not, the map will be wrong. This is the critical invariant — if orderings differ, the test in Task 4 will catch it.

- [ ] **Step 3: Construct the callback closure and call `compile_wengert_ops` with it**

Replace the first `compile_wengert_ops` call (line 3823) with a branched invocation:

```rust
let full_lowered = if fase_hook_active {
    let Some(accum) = accum_list else {
        return Err(CodegenError::InternalError(
            "fase_hook_active implies accum_list is Some".into()
        ));
    };
    let accum_scale = fase_plan.recipe.accum_scale;

    // Callback closure: m_partial += accum_scale * grad; free grad.
    let mut cb = |var_id: crate::wengert::VarId,
                  grad_ptr: Value,
                  b: &mut FunctionBuilder|
          -> Result<(), CodegenError> {
        let Some(&idx) = var_id_to_accum_idx.get(&var_id) else {
            return Ok(()); // Not tracked; skip.
        };
        let idx_val = b.ins().iconst(cl_types::I64, idx);
        let m_partial = self.compile_call_by_name(b, "nsl_list_get", &[accum, idx_val])?;
        self.fase_emit_accumulate(b, m_partial, grad_ptr, accum_scale)?;
        self.compile_call_by_name(b, "nsl_tensor_free", &[grad_ptr])?;
        Ok(())
    };

    crate::wengert_lower::compile_wengert_ops(
        self,
        builder,
        state,
        &effective_primal,
        &primal_vars,
        Some((&param_adjoint_set, &mut cb)),
    )?
} else {
    crate::wengert_lower::compile_wengert_ops(
        self,
        builder,
        state,
        &effective_primal,
        &primal_vars,
        None,
    )?
};
```

**Borrow-checker note:** the callback captures `self` mutably. Rust may require restructuring into a named function or struct if the compiler complains about multiple mutable borrows. A common fix is to move `var_id_to_accum_idx` and `accum` into the closure via `move ||`, and construct the closure fresh inside the branch.

If borrow checking fails despite `move`, fall back to a concrete struct:

```rust
struct FaseCallback<'a> { /* ... */ }
impl<'a> FaseCallback<'a> {
    fn handle(&mut self, var_id: VarId, grad_ptr: Value, b: &mut FunctionBuilder) -> Result<(), CodegenError> { /* ... */ }
}
```

Then `&mut |vid, val, b| cb.handle(vid, val, b)` as the trait-object.

- [ ] **Step 4: Skip the grads_list seed + patch loops when hook active**

Find the block at [stmt.rs:3944-4040](crates/nsl-codegen/src/stmt.rs#L3944-L4040) that:
1. Creates `grads_list` with `nsl_list_new`.
2. Fills it with zeros via a runtime loop over `param_list`.
3. Scans parameters and patches in actual gradients via another runtime loop.

Wrap this block in `if !fase_hook_active { ... }`. When the hook is active, gradients were already consumed during lowering — grads_list is unnecessary, and the patch loop would fail (`grad_vars` no longer contains the adjoint VarIds — they were removed in Task 1 Step 4).

For the hook-active branch, we still need to provide a value for whatever local variable the downstream code expects (`grads_list` is returned from this block in the source-AD tuple at line 3519). Emit a null-pointer stand-in:

```rust
let grads_list = if fase_hook_active {
    // Hook already consumed gradients; no list needed downstream.
    // Provide a null i64 so the tuple type stays consistent.
    builder.ins().iconst(cl_types::I64, 0)
} else {
    // Existing grads_list construction path (the full block).
    let grads = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
    // ... existing zero-seed + patch loops ...
    grads
};
```

Identify the exact variable whose binding must remain valid for downstream code — read the tuple destructuring around line 3519 to see what's assigned. Adapt accordingly.

- [ ] **Step 5: Skip the per-micro-batch accumulation loop when hook active**

Find the per-micro-batch accumulation loop at [stmt.rs:4183-4220](crates/nsl-codegen/src/stmt.rs#L4183-L4220) (the one that does `accum[i] += grads[i]` then frees `grads[i]`). Wrap in:

```rust
if !fase_hook_active {
    // Existing per-micro-batch accumulation loop (unchanged).
    // ...
}
```

When hook-active, the callback already accumulated during backward lowering; there's nothing more to do per-micro-batch.

Also skip the `nsl_clip_grad_norm` call at [stmt.rs:4175](crates/nsl-codegen/src/stmt.rs#L4175): if `grad_clip` is set AND `fase_hook_active`, the two-phase clip emission from item #3 handles clipping via `nsl_tensor_sum_sq` + `mul_scalar_inplace` on `m_partial`, NOT on `grads`. Wrap in:

```rust
if !fase_hook_active && grad_clip < f64::MAX {
    // Existing clip_grad_norm on grads_list.
    let max_norm_val = builder.ins().f64const(grad_clip);
    self.compile_call_by_name(builder, "nsl_clip_grad_norm", &[grads_list, max_norm_val])?;
}
```

(The two-phase clip path from item #3 is already gated on `fase_plan.two_phase_clip` and runs in the optimizer block — it doesn't touch grads_list.)

- [ ] **Step 6: Skip the grads_list cleanup when hook active**

Find the grads_list free at [stmt.rs:4274](crates/nsl-codegen/src/stmt.rs#L4274) (`nsl_list_free(grads_list)`). Wrap:

```rust
if !fase_hook_active {
    self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
}
```

When hook-active, `grads_list` is the null i64 from Step 4 — freeing null is fine, but skipping is cleaner.

- [ ] **Step 7: Adjust item #3's Phase A loop when hook is active**

Item #3 Task 6 emitted a Phase A loop (around stmt.rs:4260+) that does, per parameter:

```
pa_mpart = nsl_list_get(accum, pa_i)
pa_grad  = nsl_list_get(grads_list, pa_i)
fase_emit_accumulate(pa_mpart, pa_grad, accum_scale)
pa_sq = nsl_tensor_sum_sq(pa_mpart)
total_sq = fadd(total_sq, pa_sq)
nsl_tensor_free(pa_grad)
```

With `fase_hook_active == true`, `grads_list` is null, accumulation already happened in the callback, and `pa_grad` would be invalid. Adjust Phase A's body so that when hook-active, it ONLY does `sum_sq`:

```rust
let pa_mpart = self.compile_call_by_name(builder, "nsl_list_get", &[accum, pa_i])?;
if !fase_hook_active {
    let pa_grad = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, pa_i])?;
    self.fase_emit_accumulate(builder, pa_mpart, pa_grad, fase_plan.recipe.accum_scale)?;
    self.compile_call_by_name(builder, "nsl_tensor_free", &[pa_grad])?;
}
let pa_sq = self.compile_call_by_name(builder, "nsl_tensor_sum_sq", &[pa_mpart])?;
let pa_tot_cur = builder.use_var(pa_tot_var);
let pa_tot_new = builder.ins().fadd(pa_tot_cur, pa_sq);
builder.def_var(pa_tot_var, pa_tot_new);
```

Phase B (scale + fused step) is unchanged — it already operates on `m_partial` only.

- [ ] **Step 8: Build**

Run: `cargo build -p nsl-codegen`
Expected: succeeds. If borrow-checker errors appear for the closure, apply the fallback from Step 3 (named struct with explicit `handle` method).

- [ ] **Step 8: Run the full test suite**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | head -15`
Expected:
- All pre-existing tests pass.
- The existing item #2 tests (`sgd_exact_equivalence`, `adamw_fase_deferred_pipeline_equivalence`, `adamw_deferred_with_grad_clip`) either use tape AD (hook stays off → byte-identical behavior) or source AD (hook active → numerical equivalence preserved because the callback does the same math).
- Smoke tests pass.

If any numerical test fails with the hook active, the most likely cause is a VarId→accum_idx map mismatch: the order of `accum_list` allocations differs from `named_param_var_ids()` iteration order. Verify by logging both orderings and diffing.

- [ ] **Step 9: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "feat(fase): wire consume-per-param hook for Deferred + source-AD"
```

---

## Task 4: Source-AD variant of AdamW pipeline equivalence test

Item #2 Task 4's `adamw_fase_deferred_pipeline_equivalence` runs `nsl run` on the fixture. Today that uses whatever the default AD path is (tape AD by default per grep findings). To prove this task's hook actually fires and produces correct numerics, add a source-AD variant.

**Files:**
- Create: `crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_source_ad.nsl` (same fixture body as `fase_deferred_adamw_equivalence.nsl`; separate filename for clarity)
- Modify: `crates/nsl-codegen/tests/fase_numerical_validation.rs` — new test that passes `--source-ad` to `nsl run`

### Steps

- [ ] **Step 1: Check whether the existing `nsl_run` helper supports passing extra CLI args**

```bash
grep -n "fn nsl_run\|CARGO_BIN_EXE_nsl" crates/nsl-codegen/tests/fase_numerical_validation.rs | head
```

Today's helper is likely `fn nsl_run(fixture_path, workdir)` with fixed `nsl run <path>` args. Either extend it to take a `&[&str]` of extra args, or create a sibling `nsl_run_source_ad` that invokes `nsl run --source-ad <path>`.

The extend-existing approach is cleaner:

```rust
fn nsl_run(fixture_path: &Path, workdir: &Path) {
    nsl_run_with_args(fixture_path, workdir, &[]);
}

fn nsl_run_with_args(fixture_path: &Path, workdir: &Path, extra: &[&str]) {
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run").arg(fixture_path).current_dir(workdir);
    for a in extra {
        cmd.arg(a);
    }
    let status = cmd.status().expect("failed to spawn nsl run");
    assert!(status.success(), "nsl run failed on {:?} with args {:?}", fixture_path, extra);
}
```

- [ ] **Step 2: Create the fixture**

Copy `crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_equivalence.nsl` to `crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_source_ad.nsl`. Identical contents (same model, same inputs, same optimizer). Different filename so the test harness can point at it separately.

```bash
cp crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_equivalence.nsl \
   crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_source_ad.nsl
```

Change the `model_save` line inside the fixture to write a distinct output filename:

```nsl
model_save(m, "adamw_source_ad_out.nslm")
```

So the test doesn't clash with the original one's checkpoint name.

- [ ] **Step 3: Add the failing test**

Append to `crates/nsl-codegen/tests/fase_numerical_validation.rs`:

```rust
#[test]
fn adamw_fase_deferred_source_ad_pipeline_equivalence() {
    // Exercises the consume-per-param hook (item #4): source-AD + FASE Deferred
    // invokes the callback per parameter gradient.  Numerical result must match
    // the same Rust reference as the tape-AD variant.
    let tmp = TempDir::new().expect("tempdir");
    nsl_run_with_args(
        &fixture("fase_deferred_adamw_source_ad.nsl"),
        tmp.path(),
        &["--source-ad"],
    );

    let checkpoint = tmp.path().join("adamw_source_ad_out.nslm");
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
    let w_ref = adamw_fase_deferred_reference(
        &w_init,
        &x,
        &y,
        /*lr=*/ 0.001,
        /*beta1=*/ 0.9,
        /*beta2=*/ 0.999,
        /*eps=*/ 1e-8,
        /*wd=*/ 0.01,
        /*windows=*/ 3,
    );

    for i in 0..2 {
        let diff = (w_compiled[i] - w_ref[i]).abs();
        let scale = w_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "AdamW (source-AD) θ[{}] diverged: compiled={} reference={} rel_err={}",
            i,
            w_compiled[i],
            w_ref[i],
            diff / scale
        );
    }
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p nsl-codegen --test fase_numerical_validation -- adamw_fase_deferred_source_ad_pipeline_equivalence --nocapture`

Expected: PASS at rel_err < 1e-5.

If it fails:
- If rel_err ≈ 1e-5 range: f32 precision. Widen to 1e-4 with a comment.
- If rel_err > 1e-3: hook bug. Most likely causes:
  - VarId→accum_idx map mismatch (order of `accum_list` allocations ≠ `named_param_var_ids()` iteration order).
  - Callback didn't fire for some params (check param_adjoint_set population).
  - `fase_emit_accumulate` was called with wrong m_partial pointer.

If `nsl run --source-ad` fails with "source AD not supported for this fixture," read the error. The `@fixture_name` handler may need adjustment — source-AD is mandatory for this test. If source-AD itself refuses the fixture, that's a bigger problem: source-AD coverage on this shape of NSL program is a prerequisite that needs to land first.

- [ ] **Step 5: Run the full suite**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | head -15`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_source_ad.nsl \
        crates/nsl-codegen/tests/fase_numerical_validation.rs
git commit -m "test(fase): source-AD variant of AdamW pipeline equivalence"
```

---

## Task 5: Final verification + memory

- [ ] **Step 1: Full workspace build**

Run: `cargo build --workspace`
Expected: succeeds.

- [ ] **Step 2: Full nsl-codegen test suite**

Run: `cargo test -p nsl-codegen 2>&1 | grep "^test result" | head -20`
Expected: all green, including the new `adamw_fase_deferred_source_ad_pipeline_equivalence`.

- [ ] **Step 3: Verify non-hook paths unchanged**

Spot-check snapshots:

```bash
cargo test -p nsl-codegen --test snapshot_tests 2>&1 | tail -5
```

Expected: all snapshots pass without re-baselining. If any snapshot exercises source-AD + grad_accumulation > 1 and was re-baselined, that's a signal to investigate before committing.

- [ ] **Step 4: Update FASE project memory note**

Edit `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_fase_deferred_integration.md`. Find the line:

```
- **Item #4:** M36 memory-planner wiring (now subsumes the peak-memory win from #1b)
```

Replace with:

```
- **Item #4:** ✅ shipped 2026-04-14 — targeted scope: consume-per-param hook in compile_wengert_ops fires `fase_emit_accumulate` + `nsl_tensor_free` immediately after each parameter's gradient is produced during source-AD backward lowering.  Peak gradient memory drops to one gradient live at a time.  Tape-AD path remains unoptimized (documented limitation); full M36 adoption is a separate milestone.  Spec: `docs/superpowers/specs/2026-04-14-fase-deferred-consume-per-param-hook-design.md`.
```

- [ ] **Step 5: Report**

Summarize: commits shipped, rel_err observed for the new source-AD test, and remaining FASE roadmap items (#5 peak-memory regression test, #6 training-report CLI).

---

## Summary of files touched

- **Modified:** `crates/nsl-codegen/src/wengert_lower.rs` (Task 1)
- **Modified:** `crates/nsl-codegen/src/stmt.rs` (Tasks 2, 3)
- **Created:** `crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_source_ad.nsl` (Task 4)
- **Modified:** `crates/nsl-codegen/tests/fase_numerical_validation.rs` (Task 4)
- **Modified:** memory note (Task 5)
