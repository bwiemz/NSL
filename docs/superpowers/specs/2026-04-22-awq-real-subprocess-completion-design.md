# AWQ Real-Subprocess Completion — Design Spec

**Date:** 2026-04-22
**Status:** Design approved pending review
**Branch (target):** `feat/awq-real-subprocess-completion`
**Predecessors:**
- `2026-04-13-awq-forward-pass-and-discovery-design.md` (landed via PR #28)
- `2026-04-14-awq-real-subprocess-forward-design.md` (landed via PR #29; Tasks 1-5, 7-8 shipped; Task 6 analytical end-to-end test reverted in commit `7e035855`)
- `fix/awq-retention-arena-zero` PR #98 (shipped 2026-04-21; arena-ordering gap closed)
- Memory: `project_awq_real_subprocess_blocked.md` (2026-04-21 gap diagnosis)

---

## 1. Summary

The 2026-04-14 spec's scaffolding shipped (plan-API, `nsl_awq_write_sidecar` FFI, plan-driven 2D max-abs emission, running-buffer globals, CPU-only binary). The 2026-04-21 in-tree diagnosis found that three gaps prevent the subprocess from actually producing non-zero AWQ scales:

1. **Gap 2 — no `model_forward` call.** `binary_codegen.rs::loop_body` (~line 720) calls `nsl_calibration_batch_at` then jumps straight to the plan-driven 2D max-abs reduction. There is no `model_forward(batch)` invocation between them. The reduction reads a retention arena that nothing populates in the subprocess; max-abs stays zero.
2. **Gap 3 — model not linked.** `binary_codegen.rs:864` calls `linker::link_multi(&[obj_path], ...)` with only the calibration-scaffolding object. Neither `model_forward` nor any model weights are present in the emitted binary, so Gap 2 cannot be fixed without first fixing Gap 3 — there is nothing to call.
3. **Bonus — AST pre-scan missing.** PR #98 moved `emit_retention_arena` to run BEFORE `compile_user_functions` in every entry point, correctly. But `calibration_retention` itself is populated only inside `compile_train_block` (`stmt.rs:3966`) via `discover_awq_projections` — which runs AFTER `emit_retention_arena`. Tests pre-set `calibration_retention` manually, so they work. Auto-discovery builds (the production path) silently no-op: `emit_retention_arena` fires with `calibration_retention = None`, no arena is declared, no splice is emitted.

The evidence that Gap 2 is real is the Task 6 revert: the analytical end-to-end test (`e36634a6`) was unwound by commit `7e035855` shortly after it landed. No revert reason was recorded in the commit message; the most economical explanation is that the test exposed the empty-arena pathology and was removed rather than left red. **Reverting the test also removed the end-to-end proof the subprocess produces honest scales.** This spec re-lands that test as the merge gate.

This spec extends the 2026-04-14 work — it does not replace it. The existing plan-API, FFI, and plan-driven emission stay as the load-bearing scaffolding.

---

## 2. Non-Goals

Same as the 2026-04-14 spec §2:

- GPU-side forward during calibration.
- Multiple `quant awq { ... }` blocks in one compile.
- Glob/target expansion beyond what existing discovery handles.
- Dynamic-shape calibration.

Additionally for this spec:

- **`@quantize model` decorator surface stays unchanged.** Only WHERE discovery runs changes (Gap 3), not its public contract.
- **WRGA Prune interaction.** WGGO Prune IR rewrite lives on a separate track. Calibration runs before WGGO planning, so WGGO Prune does not observe calibration state; no interaction in either direction.
- **Subprocess caching.** Digest-based subprocess reuse (existing) is unchanged. Gap 3 may widen the digest's object-list but does not change its schema.
- **Multi-architecture model objects.** Single host-architecture compile remains the assumption. Cross-compile for calibration is its own future design.

---

## 3. Context: what shipped, what didn't

### 3.1 From the 2026-04-14 plan (tracked via PR #29)

| Task | Status | Evidence |
|---|---|---|
| 1 — `observe_plan` + `finalize_plan` on `CalibrationHook` | Shipped | `crates/nsl-codegen/src/calibration/hooks.rs:53-60` |
| 2 — `nsl_awq_write_sidecar` runtime FFI | Shipped | `crates/nsl-runtime/src/awq.rs:808-848` |
| 3 — CPU-only calibration-binary guarantee | Shipped | test `calibration_emits_binary.rs` |
| 4 — `AwqCalibrationHook` implements both methods | Shipped | PR #29 commit `5b123479` |
| 5 — Plan-driven `binary_codegen` emission + host-shortcut removal | Shipped | `binary_codegen.rs:706-746`; `build_sidecar_from_forward_observation` removed |
| 6 — Analytical end-to-end test | **Reverted** (`7e035855`) | No test today asserts subprocess scales match an analytical reference |
| 7 — Cleanup — dead paths and stale stubs | Partial | `awq_full_pipeline.rs:13-14` still carries stale "Blocker A/B" comments |
| 8 — Negative tests (subprocess crash + no-CUDA linkage) | Shipped |  |

### 3.2 From PR #98 (2026-04-21)

Gap 1 (ordering) fixed: `emit_retention_arena` now runs before `compile_user_functions` in every codegen entry point. Regression test at `retention_splice_ordering.rs`.

### 3.3 What's still missing on main

- Gap 2 (no `model_forward` call).
- Gap 3 (model not linked).
- Bonus (AST pre-scan for auto-discovery).
- Task 6 (analytical end-to-end test as merge gate).
- Task 7 leftover: stale "Blocker A/B" comments in `awq_full_pipeline.rs`.

---

## 4. Design

### 4.1 Architecture

```
main compile of NSL source
  └─ AST pre-scan (NEW — bonus gap fix)
        └─ discover_awq_projections_from_ast(&ast) → Vec<DiscoveredProjection>
        └─ compile_options.calibration_retention = Some(discovered)
  └─ emit_retention_arena (existing; now sees populated calibration_retention)
  └─ compile_user_functions
        └─ model_forward's retention splice emitted per-projection

harness_entry (was real_subprocess_entry):
  1. emit_calibration_model_object(&ast, &opts) → calib_model.o
        - declares model_forward + per-weight .data globals + splice targets
        - exports model_forward AND nsl_calib_model_forward (the ABI wrapper)
        - exports __nsl_calib_retention_arena as .bss of ArenaLayout size
  2. emit_calibration_scaffolding_object(…)  (existing; renamed)
        - calibration_main imports nsl_calib_model_forward (Linkage::Import)
        - loop_body now invokes it between nsl_calibration_batch_at and the
          plan-driven max-abs loop
  3. link_multi(&[scaffolding.o, calib_model.o], …) → calibration.exe
  4. spawn subprocess + read sidecar (unchanged)
```

Key insight: the calibration-scoped model object (`calib_model.o`) is a **fresh compile pass** of the same AST the host uses, targeting calibration ABI specifically. It is NOT the main-flow model artifact reused; calibration owns its build pipeline end-to-end.

### 4.2 Gap 2 + 3 — Two-object compilation with `Linkage::Import`

**Design choice: Option B (two-object) over Option A (single-object).** The spec considered compiling `calibration_main` + `model_forward` together into one object. Rejected because:

- The main compile flow's `compile_user_functions` is a large, stateful pass. Merging `calibration_main` emission into it conflates two concerns: model codegen and calibration scaffolding. Their failure modes, test surfaces, and iteration cycles are different; keeping them in separate objects keeps the boundary clean.
- The calibration FFIs (`nsl_calibration_load`, `nsl_calibration_batch_at`, `nsl_awq_write_sidecar`) are already `Linkage::Import` in `binary_codegen.rs`. Adding `nsl_calib_model_forward` to that import list is homogeneous with the existing ABI boundary.
- Caching: a future optimization can memoize `calib_model.o` keyed on AST digest. Hard to do if calibration scaffolding is in the same object.
- Clean abort semantics: if model compilation fails, we know the failure is in the model, not in the scaffolding. Single-object mixes the two diagnostics.

The object layout on disk after `harness_entry`:

```text
/tmp/<harness-N>/
  ├─ scaffolding.o     # calibration_main, imports nsl_calib_model_forward + all nsl_calib_* FFIs
  ├─ calib_model.o     # nsl_calib_model_forward + compiled model_forward + retention splice + .bss arena
  └─ calibration.exe   # linked from the two .o + nsl-runtime staticlib (existing)
```

### 4.3 The `nsl_calib_model_forward` ABI wrapper

`model_forward` in NSL compiled form takes an `NslTensor*` (internal ABI). For calibration, the inputs come as raw `f32*` buffers from `nsl_calibration_batch_at`. The wrapper bridges the two. It is emitted as Cranelift IR inside `calib_model.o`; no new runtime FFI is required.

```rust
// signature of the wrapper as Rust-equivalent pseudocode
#[no_mangle]
pub extern "C" fn nsl_calib_model_forward(
    batch_f32_ptr: *const f32,
    batch_elem_count: u64,
) {
    // 1. Build an NslTensorDesc on the stack with the ArenaLayout's
    //    declared shape (batch × seq × channels).
    // 2. Call nsl_desc_to_tensor(&desc) — the existing M62 FFI — which
    //    constructs an NslTensor aliasing the batch buffer.
    // 3. Call model_forward(tensor_handle). Retention splice populates
    //    __nsl_calib_retention_arena.
    // 4. Free the tensor wrapper via nsl_tensor_free (borrows the data;
    //    the f32 buffer itself is owned by nsl_calibration_batch_at).
    // 5. Return (void; calibration discards the forward output).
}
```

**Reuses M62's `nsl_desc_to_tensor`** (`crates/nsl-runtime/src/c_api.rs:597`). M62's existing C-ABI wrapper FFI for `@export` functions already takes `NslTensorDesc*` and returns `*mut NslTensor` with borrow-don't-copy semantics — exactly what calibration needs. Rather than add a parallel `nsl_tensor_wrap_f32_desc` with a different signature shape (raw pointer + length + separate shape array + ndim), this spec reuses M62's existing convention: construct an `NslTensorDesc` on the Cranelift-emitted stack, fill its fields from the ArenaLayout-derived shape + the batch buffer pointer, then call the existing FFI. Zero new runtime FFIs required; perfect consistency with the M62 family.

In Cranelift IR (emitted, not source), the wrapper:

1. Imports `model_forward` (the user-compiled forward), `nsl_desc_to_tensor` (M62 FFI, Linkage::Import), and `nsl_tensor_free` (M62 FFI, Linkage::Import).
2. Allocates an `NslTensorDesc`-sized stack slot; stores `batch_f32_ptr` into the `data` field; stores ArenaLayout-derived shape pointer + `ndim` into the respective fields; stores dtype=f32 flag.
3. Calls `nsl_desc_to_tensor(desc_ptr)` → receives an `*mut NslTensor` handle.
4. Calls `model_forward(tensor_handle)`, discards the return value.
5. Calls `nsl_tensor_free(tensor_handle)` — the wrapper struct is freed but not the aliased data buffer (borrow semantics; buffer ownership stays with `nsl_calibration_batch_at`).
6. Returns void.

The wrapper is ~40 lines of Cranelift IR inside `calib_model.o`. `batch_elem_count` is checked against the ArenaLayout's declared shape product; mismatch is a structured error (§5.1).

### 4.4 Bonus — AST pre-scan extraction

Today:

```rust
// stmt.rs:3950 — inside compile_train_block
if let Some(awq_projections) = self.discover_awq_projections() {
    self.compile_options.calibration_retention = Some(awq_projections);
}
```

This runs during `compile_user_functions`, which is AFTER `emit_retention_arena` (per PR #98's ordering fix).

**Fix:** extract discovery into a free function that operates on the AST, and run it at entry-point dispatch BEFORE `emit_retention_arena`:

```rust
// crates/nsl-codegen/src/calibration/discovery.rs
pub fn pre_scan_awq_projections_from_ast(
    ast: &Ast,
) -> Vec<DiscoveredProjection> {
    // Walk the AST looking for @quantize model decorators; build the
    // same DiscoveredProjection Vec that discover_awq_projections_from_state
    // produces from mid-compile state. Idempotent; no state mutation.
}
```

Then at each entry point (`compile_returning_plan`, `compile_entry_returning_plan`, etc., mirroring PR #96's weight-loading extraction pattern):

```rust
// Before emit_retention_arena, populate calibration_retention
// from AST if not already set by the caller.
if compiler.compile_options.calibration_retention.is_none() {
    let discovered = pre_scan_awq_projections_from_ast(&ast);
    if !discovered.is_empty() {
        compiler.compile_options.calibration_retention = Some(discovered);
    }
}
compiler.emit_retention_arena()?;
```

**Interaction with existing `discover_awq_projections` at `stmt.rs:3950`.** Keep it. The in-train-block call is still the authoritative discovery for runtime-only paths (e.g., tests that don't exercise entry-point dispatch). It becomes a no-op in auto-discovery builds because `calibration_retention` is already populated by the AST pre-scan.

**DiscoveredProjection stability contract (load-bearing for the differential test).** `DiscoveredProjection` in `crates/nsl-codegen/src/calibration/discovery.rs:46-52` has exactly two fields:

```rust
pub struct DiscoveredProjection {
    pub projection: ProjectionRef,  // qualified path, e.g. "TinyMLP.up_proj"
    pub weight_shape: [u32; 2],     // [out_features, in_features]
}
```

Both fields are **pure-AST-derivable**:

- `projection` is a qualified name synthesised from the model's name (from `ModelDef`) concatenated with a field name (from the model body's tensor field declarations).
- `weight_shape` comes from parsing the literal tensor-type annotation text (`Tensor<[128, 64], f32>`), which lives in the AST at parse time. The existing `discover_awq_projections_from_state` derives it from a `tensor_shapes: HashMap<String, String>` map of stringified annotations — that map's entries are verbatim AST text, requiring no name resolution or type-checking to extract.

Therefore Option (a) — pure-AST — applies: `pre_scan_awq_projections_from_ast` and the train-block `discover_awq_projections_from_state` arrive at **byte-for-byte identical** `DiscoveredProjection` vectors for the same AST. No masked-field comparator is needed; the differential test (§6.3) uses plain `assert_eq!`. If a future refactor adds a lowered-state-dependent field to `DiscoveredProjection` (e.g., `resolved_weight_var: VarId`), the differential test breaks and forces a design decision at review time: either make the new field pure-AST-derivable, or split the struct into a pre-scan-facing subset and an enriched superset. The spec pins the current pure-AST invariant; any future erosion of it requires a spec update, not a test-side comparator workaround.

**Runtime divergence handling.** If the two paths somehow diverge at runtime despite the differential test passing on every fixture (e.g., a user's NSL source hits an AST shape no fixture exercised and exposes a discovery path bug), the refusal in §5.3 fires at compile time with a three-part error. Hard error, not silent split-brain arena.

### 4.5 Loop-body edit

`binary_codegen.rs:706-746`:

Current:

```rust
b.switch_to_block(loop_body);
b.seal_block(loop_body);
// ...stack slots for batch_at out-params...
b.ins().call(calib_batch_ref, &[lh_batches, i_cur, out_ptr_addr, out_len_addr]);

// ── Plan-driven 2D max-abs reduction per projection ──
for entry in observe_plan {
    ...
    emit_2d_max_abs_loop(&mut b, arena_base, entry.src_offset, entry.rows, entry.channels, running_base);
}
```

After:

```rust
b.switch_to_block(loop_body);
b.seal_block(loop_body);
// ...stack slots for batch_at out-params...
b.ins().call(calib_batch_ref, &[lh_batches, i_cur, out_ptr_addr, out_len_addr]);

// Load batch_ptr + batch_len back from stack.
let batch_ptr = b.ins().load(ptr_ty, MemFlags::trusted(), out_ptr_addr, 0);
let batch_len = b.ins().load(cl_types::I64, MemFlags::trusted(), out_len_addr, 0);

// ── Call model_forward via the calibration ABI wrapper ──
// calib_model_forward_ref was declared Linkage::Import at the top of
// emit_and_link_calibration_binary; resolves at link time against
// calib_model.o. Populates __nsl_calib_retention_arena via the
// model_forward retention splice.
let fwd_ref = module.declare_func_in_func(nsl_calib_model_forward_id, b.func);
b.ins().call(fwd_ref, &[batch_ptr, batch_len]);

// ── Plan-driven 2D max-abs reduction per projection ──
// (unchanged; arena is now populated by the model_forward call above)
for entry in observe_plan {
    ...
    emit_2d_max_abs_loop(&mut b, arena_base, entry.src_offset, entry.rows, entry.channels, running_base);
}
```

The plan-driven reduction loop is unchanged. Only the new call sits between `nsl_calibration_batch_at` and the reduction.

### 4.6 Object-linkage contract

`nsl_calib_model_forward` is exported by `calib_model.o` (Linkage::Export), imported by `scaffolding.o` (Linkage::Import). The linker resolves the reference in `link_multi`.

`__nsl_calib_retention_arena` is exported by `calib_model.o` (Linkage::Export as a .bss global of computed size), imported by `scaffolding.o` (Linkage::Import). Both objects agree on the arena's size and layout via `ArenaLayout`. ArenaLayout is computed once in `harness_entry` before either object is emitted; both compilations receive the same `ArenaLayout` by construction.

Model weight `.data` globals live in `calib_model.o`. They don't need scaffolding-side declarations.

---

## 5. Refusal / error surface

Every new failure mode is a first-class precondition check with a three-part error (`requested`/`expected`/`found`). This follows the transformation-precondition-refusal discipline (see `feedback_transformation_precondition_refusal.md`).

### 5.1 Batch-shape mismatch

Trigger: `nsl_calib_model_forward` receives a batch whose element count does not match the ArenaLayout's declared (batch × seq × channels) product.

```text
calibration: batch shape mismatch at subprocess model-forward call.
  requested:  run calibration forward on batch idx={i} with {actual_elems} elements
  expected:   batch elements == ArenaLayout declared shape
              ({batch} × {seq} × {channels} = {expected_elems})
  found:      {actual_elems} elements; did calibration data layout drift
              from the ArenaLayout used to emit calib_model.o?
```

Diagnostic emitted from the subprocess via `nsl_set_error_cstr` (existing FFI from M62 work), exit code = `3`.

### 5.2 Missing `nsl_calib_model_forward` symbol at link

Trigger: scaffolding.o declares `nsl_calib_model_forward` as Linkage::Import but calib_model.o did not export it (e.g., AST has no `model_forward` function; emission skipped).

```text
calibration: model-forward wrapper missing from calib_model.o.
  requested:  link scaffolding.o ← calib_model.o with nsl_calib_model_forward
              resolved
  expected:   calib_model.o exports nsl_calib_model_forward (the f32-buffer
              wrapper around model_forward)
  found:      undefined reference `nsl_calib_model_forward`. Either the
              NSL source lacks a model_forward function, or the
              calibration-scoped compile pass skipped wrapper emission.
```

Emitted at link time from `link_multi`'s error path; caught by `real_subprocess_entry` and surfaced as `HarnessError::Infrastructure`.

### 5.3 Discovery divergence

Trigger: AST pre-scan and train-block `discover_awq_projections` produce DiscoveredProjection sets that are not byte-equivalent for the same AST.

```text
calibration: discovery divergence between AST pre-scan and in-compile path.
  requested:  discover AWQ projections in {source_file}
  expected:   pre_scan_awq_projections_from_ast and
              discover_awq_projections_from_state return identical sets
  found:      pre-scan: {pre_scan_names}
              in-compile: {in_compile_names}
              ({delta_description})
```

Compile-time error during the AST pre-scan pass when divergence is detected. Tests for this live at `calibration/discovery.rs#tests`; one test per divergence flavor (different names, different shapes, different orderings).

### 5.4 Empty observe_plan with forward-pass emitted

Trigger: `needs_forward_pass` is true and Gap 2 fix emitted the `model_forward` call, but `observe_plan` is empty (no projections to reduce against). This is a silent no-op today; after Gap 2 fix, it becomes a bug.

```text
calibration: forward pass emitted but no observations declared.
  requested:  run calibration subprocess with forward pass
  expected:   observe_plan is non-empty when a model_forward call is emitted
              (otherwise the subprocess runs the model but produces no scales)
  found:      observe_plan is empty but needs_forward_pass() returned true.
              Did a hook's requires() change without updating observe_plan()?
```

Compile-time refusal in `emit_and_link_calibration_binary`; `HarnessError::Infrastructure`.

### 5.5 Auto-discovery ran but produced zero projections with `@quantize model` present

Trigger: AST contains a `@quantize model` decorator with AWQ config but pre-scan returns an empty DiscoveredProjection vec.

```text
calibration: @quantize model declared but no AWQ projections discovered.
  requested:  auto-discover AWQ projections from {source_file}
  expected:   @quantize model + AWQ config ⇒ at least one DiscoveredProjection
              (if the model has no projections to quantize, the decorator
               should be removed)
  found:      decorator present at {span}; zero projections emitted.
              Action: either remove the @quantize model decorator, or add
              the Linear/tensor projections the decorator is meant to target.
```

**Hard-fail, not warning.** An earlier draft of this spec framed §5.5 as a compile-time warning on the rationale "a user might legitimately guard the decorator behind a config flag with no Linear layers inside." That reasoning was rejected during review: the warning-and-proceed pattern is exactly the "fallback to weaker transformations with different semantics" case the spec's §11 principle #1 rejects. Compilation succeeding with no quantization when the user wrote `@quantize model` is not semantically equivalent to the quantized binary the user asked for — it's a silent semantic downgrade that would train users to ignore calibration stderr output. Symmetric hard-failure with §5.1–5.4 gives the user an actionable diagnostic at compile time; they remove the decorator or add the layers they intended. No "legitimately guarded decorator" NSL syntax exists today (there is no `@quantize model if <flag>`), so no legitimate scenario justifies the relaxation.

Variant: `HarnessError::Infrastructure` with structured diagnostic code `AwqDecoratorNoProjections`. Compilation fails; no subprocess is spawned.

---

## 6. Test plan

### 6.1 Gap 2 — loop-body emission unit test

File: `crates/nsl-codegen/src/calibration/binary_codegen.rs` `#[cfg(test)] mod forward_call_emission`.

Assert that the emitted Cranelift IR for a forward-pass calibration binary contains a call to `nsl_calib_model_forward` between `nsl_calibration_batch_at` and the first `fmax` (start of max-abs reduction). Textual check against the IR dump; not a runtime check.

```rust
#[test]
fn loop_body_calls_model_forward_between_batch_at_and_max_abs() {
    let ir = emit_calibration_scaffolding_for_test(/* AWQ observe_plan nonempty */);
    let body = ir.loop_body_text();

    let idx_batch_at = body.find("call fn nsl_calibration_batch_at")
        .expect("batch_at call missing");
    let idx_fwd = body.find("call fn nsl_calib_model_forward")
        .expect("model_forward call missing — Gap 2 regression");
    let idx_fmax = body.find("fmax").expect("max-abs reduction missing");

    assert!(idx_batch_at < idx_fwd, "model_forward must come after batch_at");
    assert!(idx_fwd < idx_fmax, "model_forward must come before max-abs");
}
```

Catches any future regression that hoists the reduction before the forward call, or silently drops the forward call (the original Gap 2 pathology).

### 6.2 Gap 3 — link-step integration test

File: `crates/nsl-codegen/tests/awq_real_subprocess_link.rs`.

Build a minimal NSL fixture with a single `@quantize model` + one Linear layer. Run `real_subprocess_entry` through `emit_calibration_model_object` + `emit_calibration_scaffolding_object` + `link_multi`. Assert:

- Both `.o` files exist on disk after compilation.
- Link succeeds.
- The final binary exports `main` and does NOT export `model_forward` (internal to the model object).
- On platforms supporting it, `nm` / `dumpbin` shows `nsl_calib_model_forward` resolved (not undefined).

**Platform guard is literally `#[cfg(any(target_os = "linux", target_os = "macos"))]`.** The 2026-04-14 spec §5.4 used the same predicate informally ("skip on Windows or use platform-conditional test"); this spec pins it explicitly so the test file is self-contained and future spec edits don't cascade through indirection. The nm-style symbol-resolution assertion and the no-CUDA-linkage `objdump`/`ldd` assertion share this single predicate — one `#[cfg]` attribute guards both. Windows exercises the link step itself (via the test's subprocess call to `link_multi`); only the post-link symbol-inspection assertions are gated off Windows because `dumpbin` availability is not guaranteed in CI environments.

### 6.3 Bonus — AST pre-scan differential test

File: `crates/nsl-codegen/src/calibration/discovery.rs` `#[cfg(test)] mod ast_prescan_vs_state`.

For each calibration fixture NSL file:

```rust
#[test]
fn pre_scan_matches_in_compile_for_calibration_mlp_fixture() {
    let ast = parse_fixture("tests/fixtures/awq_calibration_mlp.nsl");
    let pre_scan = pre_scan_awq_projections_from_ast(&ast);
    let in_compile = run_full_compile_and_capture_discovery(&ast);
    assert_eq!(pre_scan, in_compile,
        "AST pre-scan and in-compile discovery diverged; §5.3 refusal case");
}
```

One test per fixture. Catches the §5.3 divergence case at build time rather than letting it silently reach users.

### 6.4 Merge gate — re-land the analytical end-to-end test (Task 6)

File: `crates/nsl-codegen/tests/awq_full_pipeline.rs`.

Re-land the `end_to_end_real_subprocess_matches_analytical_reference` test that commit `7e035855` reverted. Shape is unchanged from `e36634a6` (the original commit), including its **5e-6** tolerance with the original ULP-derivation rationale restored as a source comment:

```rust
/// Tolerance: 5e-6 — f32 matmul over 64-length reductions accumulates
/// ~4 ULPs of round-off vs. a pairwise accumulator; the subprocess
/// runs the same fabs/fmax loop as the analytical reference, so the
/// dominant source of drift is model_forward's matmul reordering, not
/// the reduction. Fixture dimensions:
///   up_proj:   [128, 64] — K=64 reduction in the forward matmul
///   down_proj: [64, 128] — K=128 reduction, tighter but still below 5e-6
///   batch:     [8, 4, 64] = 32 rows × 64 channels of calibration data
/// Tighter (e.g. 1e-6) risks flakiness from reduction-order drift;
/// looser would mask real subprocess-pipeline bugs.
#[test]
fn end_to_end_real_subprocess_matches_analytical_reference() {
    let sidecar = compile_and_calibrate(
        Path::new("tests/fixtures/awq_calibration_mlp.nsl"),
        Path::new("tests/fixtures/awq_calib_data.safetensors"),
        Path::new("tests/fixtures/awq_calib_weights.safetensors"),
    ).expect("real subprocess pipeline runs end-to-end");

    let calib = read_safetensors_flat("tests/fixtures/awq_calib_data.safetensors", "calibration");
    let up_w  = read_safetensors_flat("tests/fixtures/awq_calib_weights.safetensors", "TinyMLP.up_proj");
    let (up_ref, down_ref) = reference_awq_scales(&calib, &up_w);

    let up_actual   = awq_scales(&sidecar, "TinyMLP.up_proj");
    let down_actual = awq_scales(&sidecar, "TinyMLP.down_proj");

    // 5e-6 tolerance: f32 matmul over length-64 inner products accumulates
    // ~4 ULPs of round-off vs. the analytical reference's sequential summation.
    assert_close(&up_actual,   &up_ref,   5e-6, "up_proj");
    assert_close(&down_actual, &down_ref, 5e-6, "down_proj");
}
```

**This test is the merge gate. Without it, the claim "subprocess produces honest scales" has no evidence.**

The `reference_awq_scales` analytical helper from the 2026-04-14 spec §5.2 is re-added alongside. The tolerance matches the reverted commit exactly (5e-6, not 1e-6); the earlier draft of this spec mis-cited 1e-6 by mistake — the reverted code's landed value was 5e-6 with the rationale inlined above.

If the test is unreachable at re-land time (e.g., because CI lacks the linker Cranelift needs on that platform), the test is gated behind a feature flag equivalent to the existing `#[cfg(feature = "real-subprocess-test")]` pattern with explicit `"ignored"` semantics — NOT deleted. Deletion without equivalent coverage would recreate the exact bug class the Task 6 revert introduced.

### 6.5 Task 7 follow-up — stale comment cleanup

The `awq_full_pipeline.rs:13-14` comment still references `build_sidecar_from_forward_observation` and Blocker A/B from the pre-PR-#29 era. Replace with a current-state comment naming the two-object architecture.

### 6.6 Negative tests — preserve the 2026-04-14 §5.4 suite

All three negative tests from the 2026-04-14 spec §5.4 (subprocess crash surfaces up, `nsl_awq_write_sidecar` returns 3 on bad path, no CUDA linkage on Linux) must continue to pass after this work. A regression on any of those three is a merge blocker.

### 6.7 Gap 2 forward-progress evidence without Task 6

The Task 6 analytical test is the merge gate but requires the full link-step to work. To surface Gap 2 progress earlier in the task sequence, Task B's unit test (§6.1) is the gate for the loop-body edit in isolation: the IR contains the new call even before the link step is wired.

---

## 7. Pipeline position and ordering invariants

### 7.1 Main compile flow

```text
parse source → AST
  └─ pre_scan_awq_projections_from_ast(&ast) → Vec<DiscoveredProjection>   ← NEW
  └─ compile_options.calibration_retention = Some(discovered) (if non-empty)
  └─ emit_retention_arena                                                  ← existing
      (sees populated calibration_retention; declares .bss arena + emits
       Linkage::Import symbol)
  └─ compile_user_functions
      └─ model_forward body emission, with retention splice per projection
```

### 7.2 Calibration harness flow (two objects)

```text
harness_entry:
  1. emit_calibration_model_object(&ast, &opts, &arena_layout)
       → calib_model.o
       — exports: model_forward, nsl_calib_model_forward, __nsl_calib_retention_arena
  2. emit_calibration_scaffolding_object(&observe_plan, &finalize_plan, &arena_layout)
       → scaffolding.o
       — imports: nsl_calib_model_forward, __nsl_calib_retention_arena,
                  nsl_calibration_load, nsl_calibration_batch_at,
                  nsl_calibration_count, nsl_awq_write_sidecar, nsl_write_file
       — exports: main
  3. link_multi(&[scaffolding.o, calib_model.o], &binary_path)
  4. run_subprocess(&binary_path, &[data, sidecar, weights])
  5. read_sidecar(&sidecar_path) → Sidecar
```

### 7.3 Ordering invariants (load-bearing)

1. **AST pre-scan runs before `emit_retention_arena`** in every entry point. Without this, auto-discovery builds hit the Bonus gap silently.
2. **`model_forward` call emission comes after `nsl_calibration_batch_at` and before the max-abs reduction** in `loop_body`. Without this, Gap 2 reappears.
3. **Canonical link-argument order is `&[scaffolding.o, calib_model.o]`.** Both GNU ld/lld and MSVC link.exe read all `.o` files up front and resolve cross-object references in a single symbol-table pass — order sensitivity for `.o` arguments is only a concern in weak-symbol override scenarios (LTO, `/ALTERNATENAME`), none of which this spec uses. `.lib`/`.a` scanning (which IS order-sensitive on MSVC single-pass `/DEFAULTLIB:` resolution) is handled by `linker::link_multi`'s existing runtime-staticlib append, unchanged by this work. The canonical order is pinned here so the implementation doesn't need a trial-and-error reversal fallback; if the link step fails at implementation time, the failure is loud (`link.exe` emits an `LNK2019` unresolved external) and the root cause gets investigated rather than papered over by flipping argument order. Removing the "handles the reversal" fallback from an earlier draft enforces the spec's overall refusal discipline (see §11 principle #1).
4. **ArenaLayout computed once, reused by both object emissions.** If they disagree, arena offsets don't match between the splice's writes and the reduction's reads — silent garbage. Enforced by constructing ArenaLayout in `harness_entry` and passing it by shared reference.

---

## 8. API surface

### 8.1 New public functions

```rust
// crates/nsl-codegen/src/calibration/discovery.rs
pub fn pre_scan_awq_projections_from_ast(
    ast: &Ast,
) -> Vec<DiscoveredProjection>;

// crates/nsl-codegen/src/calibration/binary_codegen.rs
pub(crate) fn emit_calibration_model_object(
    ast: &Ast,
    opts: &CompileOptions,
    arena_layout: &ArenaLayout,
    out_path: &Path,
) -> Result<(), HarnessError>;

// The existing `emit_and_link_calibration_binary` is split in two:
pub(crate) fn emit_calibration_scaffolding_object(
    observe_plan: &[ObservePlanEntry],
    finalize_plan: &[FinalizePlanEntry],
    arena_layout: &ArenaLayout,
    sidecar_json: &[u8],
    needs_forward: bool,
    out_path: &Path,
) -> Result<(), HarnessError>;

// And the link step:
pub(crate) fn link_calibration_binary(
    scaffolding_obj: &Path,
    calib_model_obj: &Path,
    out_binary: &Path,
) -> Result<(), HarnessError>;
```

### 8.2 No new runtime FFIs

The wrapper (§4.3) reuses the existing M62 FFIs `nsl_desc_to_tensor` and `nsl_tensor_free` (both in `crates/nsl-runtime/src/c_api.rs`) via `Linkage::Import` from within `calib_model.o`. An earlier draft of this spec proposed a new `nsl_tensor_wrap_f32_desc(data_ptr, elem_count, shape_ptr, ndim)` FFI; that design was rejected during review for unnecessarily duplicating M62's `NslTensorDesc`-based calling convention. Zero new runtime FFIs reduces the runtime's ABI surface and guarantees behavioural consistency with the M62 `@export` wrapper family (same tensor-wrapping semantics, same free semantics, same lifetime rules).

### 8.3 Renamed / deprecated symbols

- `emit_and_link_calibration_binary` is split (§8.1). The old single function is removed.
- `build_sidecar_from_forward_observation` was already removed in PR #29; no action.

---

## 9. Commit structure (informative)

Implementation is expected to land in four commits plus the Task 6 re-land:

- **Commit A — AST pre-scan (Bonus gap).** Extracts discovery into `pre_scan_awq_projections_from_ast`, wires it into every entry point before `emit_retention_arena`, adds the differential test (§6.3). Train-block `discover_awq_projections` becomes a no-op when `calibration_retention` is already set. All existing tests stay green.
- **Commit B — Two-object split (Gap 3 scaffolding).** `emit_and_link_calibration_binary` is split into `emit_calibration_model_object` + `emit_calibration_scaffolding_object` + `link_calibration_binary`. Model object is stub-sized at this commit (no model_forward body yet; just the symbol); scaffolding still imports it. Link succeeds; tests still use the observation-free path and stay green. This commit isolates the object-split refactor. **Intentional intermediate state:** between Commit B and Commit C, a hand-run calibration subprocess produces empty sidecars because model_forward is a stub — the end-to-end pipeline does not work yet. Implementers smoke-testing Commit B should expect empty scales and not panic; the end-to-end path becomes live with Commit C's wrapper + loop-body edit. Commit D adds structured refusals on top; Commit E adds the proof via the analytical test.
- **Commit C — `nsl_calib_model_forward` wrapper + `nsl_tensor_wrap_f32_desc` (Gap 2 + Gap 3 completion).** Implements the wrapper IR emission in `emit_calibration_model_object` and the runtime FFI. Loop-body edit in `emit_calibration_scaffolding_object` adds the `model_forward` call between `batch_at` and reduction. §6.1 unit test green. §6.2 link-step integration test green.
- **Commit D — Refusal surface + diagnostics.** Implements §5.1–5.5 three-part errors. Each has a unit test asserting the exact stderr text + a structured diagnostic code.
- **Commit E (merge gate) — Re-land Task 6 analytical test.** `reference_awq_scales` helper + `end_to_end_real_subprocess_matches_analytical_reference` re-added exactly as the revert removed them. Test passes bit-exact within 1e-6 tolerance. Task 7 stale-comment cleanup included here for hygiene.

Informative only; implementer splits differently if needed. The only structural requirement: Commit C's §6.1 test cannot pass without Commit B's split landing first.

---

## 10. Success criteria (merge gate)

1. **Commit E's `end_to_end_real_subprocess_matches_analytical_reference` passes within 1e-6 tolerance.** Without this the claim "subprocess produces honest scales" has no evidence. This criterion is the one the Task 6 revert cleared — restoring it closes the loop.
2. §6.1 Gap 2 unit test passes: `nsl_calib_model_forward` call appears in loop_body IR between `batch_at` and max-abs reduction.
3. §6.2 Gap 3 link-step test passes on Linux + Windows.
4. §6.3 differential test passes: AST pre-scan and in-compile discovery agree on every fixture.
5. All refusal variants (§5.1–5.5) produce their three-part error, verified by unit tests.
6. All 2026-04-14 spec §5.4 negative tests continue to pass (regression gate).
7. Existing regression tests from PR #28, PR #29, and PR #98 all pass.
8. `py -m pytest python/tests/test_m62_export.py` continues to pass (M62 work is independent but shares runtime FFI space; regression gate).
9. Task 7 cleanup lands: `awq_full_pipeline.rs:13-14` no longer references Blocker A/B.

---

## 11. Design principles cited

Inherits from three prior institutional rules:

1. **Transformation-precondition-refusal.** Each new precondition (batch shape, symbol resolution, discovery agreement, observe_plan non-empty, decorator → projections) is a first-class refusal with a three-part error. No silent no-ops; no fallback to weaker transformations. Prior instances: CPDT `one-@cpdt-per-program`, PCA Tier A convention-match, WGGO Prune v1.
2. **Pass-ordering is part of the API.** The Bonus gap exists because `calibration_retention` population was coupled to `compile_train_block` even though `emit_retention_arena` needs it earlier. Moving discovery to an AST pre-scan makes the ordering explicit in the entry-point dispatch, not hidden inside a statement-level codegen pass. Precedents: PR #98 ordering fix itself, CPDT pipeline integration's `compile_train_block` call site.
3. **Evidence over assertion.** The Task 6 revert removed the only end-to-end test. This spec treats re-landing Task 6 as the merge gate, not as a nice-to-have. A feature without a test that proves its behavior has not been delivered.

---

## 12. Out-of-scope follow-ups (v2 and beyond)

- **Object caching across calibration runs.** `calib_model.o` depends deterministically on AST digest; could be keyed and reused across subprocess invocations when the source hasn't changed. Defer until the bench shows it matters.
- **Multi-`quant awq` blocks in one compile.** Non-goal for this spec (explicit in §2). Would require a per-block ArenaLayout + per-block reduction plan — a separate design.
- **GPU-side model_forward.** The ABI wrapper emits a CPU-only call. GPU calibration needs CUDA context init, GPU weight upload, device-to-host retention-arena transfer. A separate design; not scoped here.
- **Streaming retention arenas.** For models too large to hold all activations in memory at once, the arena could be chunked and the reduction interleaved with the forward pass. Out of scope for v1.
- **Windows LTCG / cross-compilation.** Platform-specific linker oddities (object ordering, LTCG mode, cross-compile target triples) are open questions the implementation will surface case-by-case. Spec picks the most portable default (`&[scaffolding.o, calib_model.o]` order).
- **Sidecar-digest widening for Gap 3.** Today the digest covers data + weights. After Gap 3 ships, the digest should also bind against the model-object hash so a source edit invalidates the cached sidecar. Noted here; implement when the digest next turns out to be stale in practice.

---

## 13. Relationship to other in-flight work

- **WGGO Prune v1 IR rewrite** (separate branch `feat/wggo-prune-ir-rewrite`). Independent; no shared code, no shared ordering constraints. Both specs share the "transformation-precondition-refusal" discipline and cite each other as adjacent instances of the rule.
- **WGGO Phase 2 gradient scoring** (depends on this spec). Once calibration produces real scales, `wggo_head_gradients` in the sidecar starts returning populated data; `CalibratedGradientScorer` switches from silent MagnitudeFallback to real gradient-based scoring. Memory file `project_wggo_phase2.md` tracks this dependency. **Implicit behavioural shift.** The scorer's switch is automatic — no WGGO Phase 2 code changes when this spec ships. Two consequences worth surfacing before Commit E lands:
  - Any existing WGGO Phase 2 test that asserted MagnitudeFallback output will need review after this spec's merge gate clears. Not a blocker for this spec, but a follow-up sweep task on `feat/wggo-phase2`'s test suite.
  - A smoke test that compiles the same fixture twice — once under MagnitudeFallback (e.g., by stubbing the sidecar load), once under this spec's real scorer — and asserts the scorer output is both (i) different between the two and (ii) plausible under gradient-based importance rules, would catch unintended interactions. Not added to this spec's §6 test plan because the assertion lives in WGGO Phase 2's surface, not calibration's; tracked here as a downstream follow-up.
- **M62 C wrappers** (shipped via PR #48). This spec reuses M62's existing `nsl_desc_to_tensor` + `nsl_tensor_free` FFIs directly inside the calibration wrapper (see §4.3); no parallel FFI family is introduced. The runtime ABI surface stays unchanged. No merge conflicts expected.
