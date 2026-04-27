# AWQ Real-Subprocess Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three remaining gaps in the AWQ real-subprocess calibration path (no `model_forward` call, model not linked, AST pre-scan missing) and re-land Task 6's reverted analytical end-to-end test as the merge gate.

**Architecture:** Extract AST-level AWQ projection discovery into an entry-point pre-scan that runs before `emit_retention_arena`. Split `emit_and_link_calibration_binary` into two object files (`scaffolding.o` + `calib_model.o`); emit `nsl_calib_model_forward` wrapper into `calib_model.o` that reuses M62's existing `nsl_desc_to_tensor` + `nsl_tensor_free` FFIs via Linkage::Import. Edit `loop_body` in scaffolding emission to call the wrapper between `nsl_calibration_batch_at` and the plan-driven max-abs reduction. Five structured refusal variants. End-to-end analytical test restored within 5e-6 ULP tolerance.

**Tech Stack:** Rust 1.95.0, Cranelift 0.116, existing `linker::link_multi`, existing M62 FFIs in `crates/nsl-runtime/src/c_api.rs`, existing hook trait `CalibrationHook` with `observe_plan` / `finalize_plan`, `insta` snapshot testing.

**Reference spec:** [2026-04-22-awq-real-subprocess-completion-design.md](../specs/2026-04-22-awq-real-subprocess-completion-design.md) (branch `docs/awq-real-subprocess-completion-design`, head commit `b697a5a9`, 628 lines). All task-level references to §N cite that spec.

**Target branch:** `feat/awq-real-subprocess-completion`.

---

## Codebase Anchors (pin these before starting)

Verify each exists before Task 1; if any has moved, adjust the plan's references.

| Anchor | File:Line |
|---|---|
| `real_subprocess_entry` | `crates/nsl-codegen/src/calibration/binary_codegen.rs:60` |
| `emit_and_link_calibration_binary` | `crates/nsl-codegen/src/calibration/binary_codegen.rs:327` |
| `emit_2d_max_abs_loop` helper | `crates/nsl-codegen/src/calibration/binary_codegen.rs:215` |
| `loop_body` (current — missing model_forward call) | `crates/nsl-codegen/src/calibration/binary_codegen.rs:705-746` |
| `link_multi` call | `crates/nsl-codegen/src/calibration/binary_codegen.rs:864` |
| `needs_forward_pass` registry query | `crates/nsl-codegen/src/calibration/binary_codegen.rs:69` |
| `discover_awq_projections_from_state` | `crates/nsl-codegen/src/calibration/discovery.rs:334` |
| `discover_awq_projections` (in-train-block call) | `crates/nsl-codegen/src/stmt.rs:3950` |
| `DiscoveredProjection` struct (pure-AST stable) | `crates/nsl-codegen/src/calibration/discovery.rs:46` |
| `NslTensorDesc` C struct (9 fields) | `crates/nsl-runtime/src/c_api.rs:26` |
| `nsl_desc_to_tensor` M62 FFI | `crates/nsl-runtime/src/c_api.rs:597` |
| `nsl_tensor_to_desc_ffi` M62 FFI | `crates/nsl-runtime/src/c_api.rs:609` |
| `nsl_tensor_free` FFI | `crates/nsl-runtime/src/tensor/mod.rs` (existing) |
| `ArenaLayout` | `crates/nsl-codegen/src/calibration/retention.rs:74` |
| `build_arena_layout` | `crates/nsl-codegen/src/calibration/retention_pass.rs` |
| Reverted Task 6 commit | `e36634a6` (target for `git show` to recover test content verbatim) |
| `awq_full_pipeline.rs` stale "Blocker A/B" comments | `crates/nsl-codegen/tests/awq_full_pipeline.rs:13-14` |
| Entry points to wire AST pre-scan | `crates/nsl-codegen/src/compiler/entry_points.rs` |
| Fixture NSL source | `tests/fixtures/awq_calibration_mlp.nsl` |
| Fixture calibration data | `tests/fixtures/awq_calib_data.safetensors` |
| Fixture weights | `tests/fixtures/awq_calib_weights.safetensors` |

---

## File Structure

**Create:**

| File | Responsibility |
|---|---|
| `crates/nsl-codegen/tests/awq_ast_prescan_differential.rs` | Differential test: AST pre-scan vs in-compile discovery (§6.3). |
| `crates/nsl-codegen/tests/awq_real_subprocess_link.rs` | Gap 3 link integration test (§6.2). |

**Modify:**

| File | Change |
|---|---|
| `crates/nsl-codegen/src/calibration/discovery.rs` | Add `pre_scan_awq_projections_from_ast(&Ast) -> Vec<DiscoveredProjection>` (§4.4). |
| `crates/nsl-codegen/src/compiler/entry_points.rs` | Call pre-scan before `emit_retention_arena` at every entry point (§4.4). |
| `crates/nsl-codegen/src/calibration/binary_codegen.rs` | Split `emit_and_link_calibration_binary` into `emit_calibration_model_object` + `emit_calibration_scaffolding_object` + `link_calibration_binary`; add `nsl_calib_model_forward` wrapper emission; edit `loop_body` to call it; add five structured refusals (§4.1-4.6, §5.1-5.5). |
| `crates/nsl-codegen/src/calibration/mod.rs` | Re-export new public items. |
| `crates/nsl-codegen/tests/awq_full_pipeline.rs` | Re-land Task 6 analytical test + `reference_awq_scales` helper (§6.4). Cleanup stale Blocker A/B comments (§6.5). |
| `tests/fixtures/gen_awq_fixtures.py` | Confirm fixtures match reverted Task 6's shapes (8×4×64 calib, 128×64 up, 64×128 down). |

---

## Execution Order

Four commits matching spec §9, plus the merge-gate Commit E:

- **Phase A** (Commit A): AST pre-scan — Tasks 1-3
- **Phase B** (Commit B): Two-object split (stub wrapper) — Tasks 4-6
- **Phase C** (Commit C): Wrapper body + loop-body edit — Tasks 7-9
- **Phase D** (Commit D): Refusal surface — Tasks 10-14
- **Phase E** (Commit E merge gate): Task 6 re-land — Tasks 15-16

**Intentional intermediate state between Commit B and Commit C:** the pipeline is wired end-to-end but `model_forward` body is a stub; hand-running calibration produces empty sidecars. This is spec §9 Commit B's flagged intermediate state. Commit C closes the gap.

---

# Phase A — AST pre-scan (Commit A)

### Task 1: `pre_scan_awq_projections_from_ast` function

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/discovery.rs`
- Modify: `crates/nsl-codegen/src/calibration/mod.rs` (re-export)

- [ ] **Step 1: Write the failing test**

Append to `crates/nsl-codegen/src/calibration/discovery.rs` `#[cfg(test)] mod prescan_tests`:

```rust
#[test]
fn pre_scan_finds_both_projections_in_tinymlp_fixture() {
    let source = std::fs::read_to_string(
        "../../tests/fixtures/awq_calibration_mlp.nsl"
    ).expect("fixture readable");
    let ast = nsl_parser::parse(&source).expect("fixture parses");
    let discovered = pre_scan_awq_projections_from_ast(&ast);
    assert_eq!(discovered.len(), 2, "TinyMLP has up_proj + down_proj");
    assert!(discovered.iter().any(|d| d.projection.0 == "TinyMLP.up_proj"));
    assert!(discovered.iter().any(|d| d.projection.0 == "TinyMLP.down_proj"));
    let up = discovered.iter().find(|d| d.projection.0 == "TinyMLP.up_proj").unwrap();
    assert_eq!(up.weight_shape, [128, 64]);
    let down = discovered.iter().find(|d| d.projection.0 == "TinyMLP.down_proj").unwrap();
    assert_eq!(down.weight_shape, [64, 128]);
}

#[test]
fn pre_scan_returns_empty_when_no_quantize_decorator() {
    let ast = nsl_parser::parse("fn main(): return 0\n").expect("parses");
    let discovered = pre_scan_awq_projections_from_ast(&ast);
    assert!(discovered.is_empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

```
cargo test -p nsl-codegen calibration::discovery::prescan_tests
```

Expected: FAIL — `pre_scan_awq_projections_from_ast` not defined.

- [ ] **Step 3: Write minimal implementation**

In `crates/nsl-codegen/src/calibration/discovery.rs`:

```rust
/// AST-level pre-scan for AWQ-quantized model projections.
///
/// Mirrors `discover_awq_projections_from_state` but operates on raw AST
/// (no post-resolution state needed). Both fields of `DiscoveredProjection`
/// are pure-AST-derivable: `projection` from `ModelDef.name` + field name,
/// `weight_shape` from the literal `Tensor<[out, in], f32>` type annotation.
///
/// See spec §4.4 (DiscoveredProjection stability contract). The invariant is:
/// for any AST, this function and `discover_awq_projections_from_state`
/// produce byte-for-byte identical Vec<DiscoveredProjection>. Enforced by
/// the differential test in Task 3.
pub fn pre_scan_awq_projections_from_ast(ast: &Ast) -> Vec<DiscoveredProjection> {
    let mut out = Vec::new();
    for decl in ast.iter_decls() {
        let Decl::Model(model_def) = decl else { continue; };
        if !model_has_quantize_awq_decorator(model_def) { continue; }
        out.extend(enumerate_linear_fields_with_shapes(model_def));
    }
    out.sort_by(|a, b| a.projection.0.cmp(&b.projection.0));
    out
}

fn model_has_quantize_awq_decorator(model: &ModelDef) -> bool {
    model.decorators.iter().any(|d| {
        d.name == "quantize" && d.args.iter().any(|(k, v)| k == "dtype" && v == "awq4")
    })
}

fn enumerate_linear_fields_with_shapes(
    model: &ModelDef,
) -> Vec<DiscoveredProjection> {
    let model_name = model.name.as_str().to_string();
    let mut out = Vec::new();
    for field in &model.fields {
        let Some(shape) = parse_tensor_shape_from_type_expr(&field.ty) else { continue; };
        let qualified = format!("{}.{}", model_name, field.name);
        out.push(DiscoveredProjection {
            projection: ProjectionRef::new(qualified),
            weight_shape: shape,
        });
    }
    out
}
```

Re-export from `mod.rs`:

```rust
pub use discovery::pre_scan_awq_projections_from_ast;
```

- [ ] **Step 4: Run test to verify passes**

```
cargo test -p nsl-codegen calibration::discovery::prescan_tests
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(calib): pre_scan_awq_projections_from_ast — pure-AST AWQ discovery"
```

---

### Task 2: Wire pre-scan into entry points before `emit_retention_arena`

**Files:**

- Modify: `crates/nsl-codegen/src/compiler/entry_points.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/nsl-codegen/tests/awq_autodiscovery_entry_points.rs`:

```rust
//! Verifies that compile entry points populate `calibration_retention`
//! from the AST before `emit_retention_arena` runs, closing the Bonus
//! gap from spec §1 / §4.4.

use nsl_codegen::{compile_options_default, compile_returning_plan};
use std::path::Path;

#[test]
fn auto_discovery_populates_retention_arena_without_test_preset() {
    let source = std::fs::read_to_string(
        "tests/fixtures/awq_calibration_mlp.nsl"
    ).expect("fixture");
    let mut opts = compile_options_default();
    // Explicitly DO NOT pre-set calibration_retention. This is the
    // auto-discovery path that was broken before this task.
    assert!(opts.calibration_retention.is_none());

    let result = compile_returning_plan(
        &source,
        "<fixture>",
        &mut opts,
    );
    assert!(result.is_ok(), "compile succeeded: {:?}", result.err());

    // After compile, calibration_retention should be populated by the
    // pre-scan. This asserts auto-discovery works end-to-end, not just
    // the pre_scan function in isolation.
    assert!(
        opts.calibration_retention.is_some(),
        "pre-scan must populate calibration_retention before emit_retention_arena"
    );
    let projections = opts.calibration_retention.as_ref().unwrap();
    assert_eq!(projections.len(), 2, "TinyMLP has up_proj + down_proj");
}
```

- [ ] **Step 2: Run test — expect FAIL**

```
cargo test -p nsl-codegen --test awq_autodiscovery_entry_points
```

Expected: FAIL — `calibration_retention` stays `None` because pre-scan isn't wired.

- [ ] **Step 3: Wire pre-scan into every entry point**

In `crates/nsl-codegen/src/compiler/entry_points.rs`, locate each entry function (`compile_returning_plan`, `compile_with_zk_info_best_effort_plan`, `compile_standalone_best_effort_plan`, `compile_module_with_imports_best_effort_plan`, `compile_entry_returning_plan`, `compile_test`). Immediately before each `compiler.emit_retention_arena()?;` call, add:

```rust
// AST pre-scan: populate calibration_retention from the parsed AST if
// the caller did not pre-set it (auto-discovery path). Idempotent —
// pre-set values from tests / CLI are preserved.
if compiler.compile_options.calibration_retention.is_none() {
    let discovered = crate::calibration::pre_scan_awq_projections_from_ast(&ast);
    if !discovered.is_empty() {
        compiler.compile_options.calibration_retention = Some(discovered);
    }
}
compiler.emit_retention_arena()?;
```

Extract the pattern into a small helper if multiple entry points share enough structure:

```rust
// in compiler/entry_points.rs
fn populate_calibration_retention_from_ast_if_unset(
    compiler: &mut Compiler,
    ast: &Ast,
) {
    if compiler.compile_options.calibration_retention.is_some() { return; }
    let discovered = crate::calibration::pre_scan_awq_projections_from_ast(ast);
    if !discovered.is_empty() {
        compiler.compile_options.calibration_retention = Some(discovered);
    }
}
```

Call the helper before `emit_retention_arena` at every entry point. No copy-paste drift between entry points.

- [ ] **Step 4: Run test to verify passes**

```
cargo test -p nsl-codegen --test awq_autodiscovery_entry_points
cargo test -p nsl-codegen  # confirm no other regression
```

Expected: autodiscovery test passes; nothing else breaks.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(calib): wire AST pre-scan into entry points before emit_retention_arena"
```

---

### Task 3: Differential test (pure-AST vs in-compile discovery)

**Files:**

- Create: `crates/nsl-codegen/tests/awq_ast_prescan_differential.rs`

- [ ] **Step 1: Write the test (this one should pass immediately — it's the invariant guard)**

```rust
//! Spec §5.3 differential — AST pre-scan and in-compile discovery must
//! agree byte-for-byte on the DiscoveredProjection set for every fixture.
//!
//! Catches the §5.3 `DiscoveryDivergence` refusal case at build time
//! rather than letting it reach users at compile time.

use nsl_codegen::calibration::{
    discover_awq_projections_from_state,
    pre_scan_awq_projections_from_ast,
    DiscoveredProjection,
};

fn run_fixture_differential(fixture_path: &str) {
    let source = std::fs::read_to_string(fixture_path).expect("fixture");
    let ast = nsl_parser::parse(&source).expect("parse");

    let pre_scan: Vec<DiscoveredProjection> =
        pre_scan_awq_projections_from_ast(&ast);

    // Run the mid-compile discovery path — reuse whatever harness
    // compile_returning_plan uses for `discover_awq_projections_from_state`.
    // Helper `collect_in_compile_discovery` lives in this test file.
    let in_compile: Vec<DiscoveredProjection> =
        collect_in_compile_discovery(&source);

    assert_eq!(
        pre_scan, in_compile,
        "AST pre-scan and in-compile discovery diverged on {fixture_path} — §5.3 guard failed"
    );
}

#[test]
fn differential_agrees_on_awq_calibration_mlp_fixture() {
    run_fixture_differential("tests/fixtures/awq_calibration_mlp.nsl");
}

// Add one test-function per calibration fixture that exercises the
// AWQ discovery surface. As of 2026-04-22 there is exactly one fixture;
// add more as they land.
```

The `collect_in_compile_discovery` helper invokes compile up to the point where `discover_awq_projections_from_state` runs, captures the returned `Vec<DiscoveredProjection>`, and returns it. Implementation mirrors `awq_full_pipeline.rs`'s existing discovery-capturing pattern.

- [ ] **Step 2: Run test — expect PASS**

```
cargo test -p nsl-codegen --test awq_ast_prescan_differential
```

Expected: PASS. The two paths already agree per §4.4's pure-AST stability contract. If the test fails, either the pre-scan or the in-compile path has a bug Task 1's unit tests didn't catch.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "test(calib): differential test — AST pre-scan vs in-compile discovery agreement

$(cat <<'INNER'
Closes spec §5.3's DiscoveryDivergence refusal at build time. Every
fixture that exercises AWQ discovery runs through both paths; byte-for-
byte assert_eq! per §4.4's pure-AST stability contract.
INNER
)"
```

**Phase A merge gate:** Commit A lands three commits (Tasks 1-3). `cargo test -p nsl-codegen` green; auto-discovery populates `calibration_retention` correctly; differential test catches future discovery-path divergence.

---

# Phase B — Two-object split (Commit B)

Commit B refactors `emit_and_link_calibration_binary` into three smaller functions and emits a stub `nsl_calib_model_forward` symbol. The pipeline links successfully; hand-running calibration still produces empty sidecars (intentional intermediate state until Commit C).

### Task 4: `emit_calibration_model_object` with stub wrapper

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/binary_codegen.rs`

- [ ] **Step 1: Write the failing test**

Append to `binary_codegen.rs` `#[cfg(test)] mod tests`:

```rust
#[test]
fn emit_calibration_model_object_exports_wrapper_symbol() {
    let ast = parse_minimal_ast_with_awq_model();
    let arena_layout = ArenaLayout::empty();  // stub
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("calib_model.o");

    emit_calibration_model_object(
        &ast,
        &CompileOptions::default(),
        &arena_layout,
        &out_path,
    ).expect("emit succeeds");

    assert!(out_path.exists());
    // Assert nsl_calib_model_forward is in the object's exports table.
    // Implementation uses the `object` crate or direct symbol scan.
    let obj_bytes = std::fs::read(&out_path).unwrap();
    let obj = object::File::parse(&*obj_bytes).unwrap();
    assert!(
        obj.symbols().any(|s| s.name() == Ok("nsl_calib_model_forward")),
        "calib_model.o must export nsl_calib_model_forward"
    );
}
```

- [ ] **Step 2: Run test — expect FAIL**

Expected: FAIL — `emit_calibration_model_object` doesn't exist.

- [ ] **Step 3: Write `emit_calibration_model_object` with stub wrapper body**

New function in `binary_codegen.rs`:

```rust
/// Emit the calibration-scoped model object file (calib_model.o).
/// Contains: model_forward (compiled from user AST with retention splice),
/// nsl_calib_model_forward (ABI wrapper; stub body in this commit),
/// per-weight .data globals, __nsl_calib_retention_arena as .bss.
///
/// Exports: model_forward, nsl_calib_model_forward, __nsl_calib_retention_arena.
pub(crate) fn emit_calibration_model_object(
    ast: &Ast,
    opts: &CompileOptions,
    arena_layout: &ArenaLayout,
    out_path: &Path,
) -> Result<(), HarnessError> {
    // 1. Build ObjectModule with calibration target triple (CPU, no cuda).
    // 2. Compile model_forward + weight globals via the main codegen
    //    entry pointed at this module rather than the main output module.
    // 3. Declare the wrapper symbol:
    //      let mut wrapper_sig = module.make_signature();
    //      wrapper_sig.call_conv = call_conv;
    //      wrapper_sig.params.push(AbiParam::new(I64));  // batch_f32_ptr
    //      wrapper_sig.params.push(AbiParam::new(I64));  // batch_elem_count
    //      let wrapper_id = module.declare_function(
    //          "nsl_calib_model_forward", Linkage::Export, &wrapper_sig
    //      )?;
    // 4. Emit STUB body (returns immediately; no model_forward call yet):
    //      let mut ctx = Context::for_function(Function::with_name_signature(...));
    //      let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
    //      let entry = builder.create_block();
    //      builder.append_block_params_for_function_params(entry);
    //      builder.switch_to_block(entry);
    //      builder.seal_block(entry);
    //      builder.ins().return_(&[]);
    //      builder.finalize();
    //      module.define_function(wrapper_id, &mut ctx)?;
    // 5. Declare __nsl_calib_retention_arena as .bss sized per ArenaLayout.
    // 6. Finish object, write bytes to out_path.
    Ok(())
}
```

Commit C will replace the stub body (step 4) with the real wrapper IR.

- [ ] **Step 4: Run test to verify passes**

```
cargo test -p nsl-codegen binary_codegen::tests::emit_calibration_model_object_exports_wrapper_symbol
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(calib): emit_calibration_model_object — stub wrapper emission"
```

---

### Task 5: `emit_calibration_scaffolding_object` (split from `emit_and_link`)

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/binary_codegen.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn emit_calibration_scaffolding_imports_calib_model_forward() {
    let observe_plan = vec![/* one stub ObservePlanEntry */];
    let finalize_plan = vec![/* one stub FinalizePlanEntry */];
    let arena_layout = ArenaLayout::empty();
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("scaffolding.o");

    emit_calibration_scaffolding_object(
        &observe_plan,
        &finalize_plan,
        &arena_layout,
        b"{}",
        true,
        &out_path,
    ).expect("emit succeeds");

    let obj_bytes = std::fs::read(&out_path).unwrap();
    let obj = object::File::parse(&*obj_bytes).unwrap();

    // Assert scaffolding.o IMPORTS nsl_calib_model_forward (undefined symbol).
    let imports: Vec<&str> = obj.symbols()
        .filter(|s| s.is_undefined())
        .filter_map(|s| s.name().ok())
        .collect();
    assert!(
        imports.contains(&"nsl_calib_model_forward"),
        "scaffolding.o must import nsl_calib_model_forward"
    );
    // Also imports existing FFIs.
    assert!(imports.contains(&"nsl_calibration_load"));
    assert!(imports.contains(&"nsl_calibration_batch_at"));
    assert!(imports.contains(&"nsl_awq_write_sidecar"));

    // Exports `main`.
    let exports: Vec<&str> = obj.symbols()
        .filter(|s| s.is_global() && !s.is_undefined())
        .filter_map(|s| s.name().ok())
        .collect();
    assert!(exports.contains(&"main"));
}
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Extract `emit_calibration_scaffolding_object` from the current `emit_and_link_calibration_binary`**

The existing function (line 327) both emits AND links. Split into:

```rust
/// Emit the calibration-scaffolding object file (scaffolding.o).
/// Contains: main (calibration_main entry point), FFI imports for
/// nsl_calibration_*, nsl_awq_write_sidecar, nsl_write_file, and — new
/// in this task — nsl_calib_model_forward.
///
/// Imports: nsl_calib_model_forward, __nsl_calib_retention_arena,
/// nsl_calibration_load, nsl_calibration_batch_at, nsl_calibration_count,
/// nsl_awq_write_sidecar, nsl_write_file.
/// Exports: main.
pub(crate) fn emit_calibration_scaffolding_object(
    observe_plan: &[ObservePlanEntry],
    finalize_plan: &[FinalizePlanEntry],
    arena_layout: &ArenaLayout,
    sidecar_json: &[u8],
    needs_forward: bool,
    out_path: &Path,
) -> Result<(), HarnessError> {
    // Move the existing ObjectModule construction + FFI imports + main-
    // function emission here. Add ONE new Linkage::Import declaration:
    //
    //   let mut fwd_sig = module.make_signature();
    //   fwd_sig.call_conv = call_conv;
    //   fwd_sig.params.push(AbiParam::new(I64));  // batch_f32_ptr
    //   fwd_sig.params.push(AbiParam::new(I64));  // batch_elem_count
    //   let calib_model_forward_id = module.declare_function(
    //       "nsl_calib_model_forward",
    //       Linkage::Import,
    //       &fwd_sig
    //   )?;
    //
    // Pass `calib_model_forward_id` to the main-body builder.
    //
    // `__nsl_calib_retention_arena` is also declared Linkage::Import
    // (was Linkage::Export in the old single-object design).
    //
    // This task keeps loop_body unchanged — Commit C edits it. The
    // new FuncId is declared but not yet called.
    Ok(())
}
```

- [ ] **Step 4: Run test to verify passes**

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(calib): split emit_calibration_scaffolding_object from emit_and_link"
```

---

### Task 6: `link_calibration_binary` wrapping `link_multi`

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/binary_codegen.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/nsl-codegen/tests/awq_real_subprocess_link.rs`:

```rust
//! Spec §6.2 — end-to-end two-object link integration.

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn two_object_link_produces_binary_with_resolved_wrapper_symbol() {
    let tmp = tempfile::tempdir().unwrap();

    // Emit calib_model.o via Task 4's function (stub wrapper body).
    let model_obj = tmp.path().join("calib_model.o");
    let ast = parse_fixture("tests/fixtures/awq_calibration_mlp.nsl");
    let arena_layout = build_arena_layout(/* ... */);
    emit_calibration_model_object(&ast, &opts, &arena_layout, &model_obj).unwrap();

    // Emit scaffolding.o via Task 5's function.
    let scaffolding_obj = tmp.path().join("scaffolding.o");
    emit_calibration_scaffolding_object(
        &observe_plan, &finalize_plan, &arena_layout,
        b"{}", true, &scaffolding_obj,
    ).unwrap();

    // Link via Task 6's wrapper.
    let binary = tmp.path().join(if cfg!(windows) { "calib.exe" } else { "calib" });
    link_calibration_binary(&scaffolding_obj, &model_obj, &binary).unwrap();

    assert!(binary.exists());

    // Gated on Linux/macOS (per spec §6.2): verify no undefined
    // nsl_calib_model_forward symbol via nm.
    let nm_output = std::process::Command::new("nm")
        .arg(&binary).output().unwrap();
    let stdout = String::from_utf8_lossy(&nm_output.stdout);
    // If nsl_calib_model_forward is undefined (`U` in nm), linkage is broken.
    assert!(
        !stdout.lines().any(|l| l.contains("U nsl_calib_model_forward")),
        "nsl_calib_model_forward must be resolved, not undefined: nm output:\n{}",
        stdout
    );
}

// CUDA-free linkage check preserved from 2026-04-14 spec §5.4.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn binary_has_no_cuda_linkage() {
    let tmp = tempfile::tempdir().unwrap();
    /* ...build binary... */
    let ldd_output = std::process::Command::new("ldd")
        .arg(&binary).output().unwrap();
    let stdout = String::from_utf8_lossy(&ldd_output.stdout);
    assert!(
        !stdout.to_lowercase().contains("cuda"),
        "calibration binary must have zero CUDA linkage: {}", stdout
    );
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `link_calibration_binary`**

```rust
/// Link scaffolding.o + calib_model.o + runtime staticlib into the
/// calibration binary. Canonical link argument order is pinned per
/// spec §7.3: scaffolding.o first, calib_model.o second. Both GNU ld/lld
/// and MSVC link.exe resolve .o cross-references in a single up-front
/// symbol-table pass, so this order works on all targets. Link failures
/// should be loud (LNK2019 or undefined reference), not silently retried
/// with reversed order.
pub(crate) fn link_calibration_binary(
    scaffolding_obj: &Path,
    calib_model_obj: &Path,
    out_binary: &Path,
) -> Result<(), HarnessError> {
    crate::linker::link_multi(
        &[scaffolding_obj.to_path_buf(), calib_model_obj.to_path_buf()],
        out_binary,
    ).map_err(|e| HarnessError::Infrastructure {
        reason: format!("link_calibration_binary: {e}"),
    })
}
```

Update `real_subprocess_entry` (existing line 60) to call the three new functions in sequence instead of the old `emit_and_link_calibration_binary`:

```rust
// Old:
// let binary_path = emit_and_link_calibration_binary(&tmp, ...)?;

// New:
let model_obj = tmp.join("calib_model.o");
emit_calibration_model_object(&ast, &cfg.compile_options, &arena_layout, &model_obj)?;
let scaffolding_obj = tmp.join("scaffolding.o");
emit_calibration_scaffolding_object(
    &observe_plan, &finalize_plan, &arena_layout,
    &sidecar_json_for_binary, needs_forward, &scaffolding_obj,
)?;
let binary_path = tmp.join(if cfg!(target_os = "windows") { "calibration.exe" } else { "calibration" });
link_calibration_binary(&scaffolding_obj, &model_obj, &binary_path)?;
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(calib): link_calibration_binary — two-object link wrapper

Closes Commit B of spec §9. scaffolding.o + calib_model.o linked
with canonical &[scaffolding.o, calib_model.o] order per §7.3.
Intentional intermediate state: model_forward body is still a stub
(Task 4), so calibration runs produce empty sidecars until Commit C.
"
```

**Phase B merge gate:** `cargo test -p nsl-codegen --test awq_real_subprocess_link` passes on Linux/macOS. Windows builds compile and link. Existing observation-free tests still pass (they don't exercise the forward path).

---

# Phase C — Wrapper body + loop-body edit (Commit C)

Replaces the Task 4 stub `nsl_calib_model_forward` body with the real wrapper IR, and edits `loop_body` to call it between `nsl_calibration_batch_at` and the max-abs reduction.

### Task 7: `NslTensorDesc` stack-allocation helper in wrapper IR

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/binary_codegen.rs`

- [ ] **Step 1: Write the failing test (IR-shape unit test)**

```rust
#[test]
fn wrapper_ir_contains_nsl_desc_to_tensor_call() {
    // Emit nsl_calib_model_forward's body standalone (without linking)
    // and inspect the Cranelift IR text.
    let ir = emit_nsl_calib_model_forward_body_standalone_for_test();
    let body_text = ir.display().to_string();

    // Wrapper constructs NslTensorDesc on stack + calls M62 FFIs.
    assert!(body_text.contains("stack_addr"),
        "wrapper must allocate NslTensorDesc stack slot");
    assert!(body_text.contains("call fn nsl_desc_to_tensor"),
        "wrapper must call M62's nsl_desc_to_tensor");
    assert!(body_text.contains("call fn nsl_tensor_free"),
        "wrapper must free the wrapped tensor after model_forward returns");
}
```

- [ ] **Step 2: Run — expect FAIL**

Current body is the Task 4 stub (just `return`). No `stack_addr` / `nsl_desc_to_tensor`.

- [ ] **Step 3: Replace the stub body with real wrapper IR**

In `emit_calibration_model_object`, replace the stub body emission with:

```rust
// Declare M62 FFIs as Linkage::Import.
let mut desc_to_tensor_sig = module.make_signature();
desc_to_tensor_sig.call_conv = call_conv;
desc_to_tensor_sig.params.push(AbiParam::new(I64)); // NslTensorDesc*
desc_to_tensor_sig.returns.push(AbiParam::new(I64)); // NslTensor*
let desc_to_tensor_id = module.declare_function(
    "nsl_desc_to_tensor", Linkage::Import, &desc_to_tensor_sig,
)?;

let mut tensor_free_sig = module.make_signature();
tensor_free_sig.call_conv = call_conv;
tensor_free_sig.params.push(AbiParam::new(I64)); // NslTensor*
let tensor_free_id = module.declare_function(
    "nsl_tensor_free", Linkage::Import, &tensor_free_sig,
)?;

// Declare model_forward as Linkage::Local (defined elsewhere in this
// same object; emitted by compile_model_forward_into(module, ...) below).
// For the wrapper's caller-perspective: declare_func_in_func(model_forward_id, ...)
// and then `call` it.

// Wrapper body:
{
    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, /* counter */ 0),
        wrapper_sig.clone(),
    ));
    let mut fn_builder_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    let batch_ptr = b.block_params(entry)[0];
    let _batch_elem_count = b.block_params(entry)[1];  // checked via refusal §5.1 in Task 10

    // Allocate NslTensorDesc on stack. Layout from crates/nsl-runtime/src/c_api.rs:26:
    //   data: *mut c_void      (offset 0, 8 bytes)
    //   shape: *mut i64        (offset 8, 8 bytes)
    //   strides: *mut i64      (offset 16, 8 bytes)
    //   ndim: i32              (offset 24, 4 bytes)
    //   dtype: i32             (offset 28, 4 bytes)
    //   device_type: i32       (offset 32, 4 bytes)
    //   device_id: i32         (offset 36, 4 bytes)
    //   [total: 40 bytes, align 8]
    let desc_slot = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot, 40, 3,  // align = 2^3 = 8
    ));
    let desc_addr = b.ins().stack_addr(I64, desc_slot, 0);

    // Store batch_ptr into desc.data (offset 0).
    b.ins().store(MemFlags::trusted(), batch_ptr, desc_addr, 0);

    // shape: emit as a .rodata global with the ArenaLayout-derived
    // shape values (batch × seq × channels as three i64 entries).
    let shape_gv = declare_arena_shape_rodata(&mut module, arena_layout)?;
    let shape_ref = module.declare_data_in_func(shape_gv, b.func);
    let shape_ptr = b.ins().symbol_value(I64, shape_ref);
    b.ins().store(MemFlags::trusted(), shape_ptr, desc_addr, 8);

    // strides: null — contiguous layout; M62's nsl_desc_to_tensor
    // handles null-strides by computing contiguous from shape+ndim.
    let zero_i64 = b.ins().iconst(I64, 0);
    b.ins().store(MemFlags::trusted(), zero_i64, desc_addr, 16);

    // ndim: 3
    let three_i32 = b.ins().iconst(I32, 3);
    b.ins().store(MemFlags::trusted(), three_i32, desc_addr, 24);
    // dtype: 0 (f32 in C API convention)
    let zero_i32 = b.ins().iconst(I32, 0);
    b.ins().store(MemFlags::trusted(), zero_i32, desc_addr, 28);
    // device_type: 0 (CPU)
    b.ins().store(MemFlags::trusted(), zero_i32, desc_addr, 32);
    // device_id: 0
    b.ins().store(MemFlags::trusted(), zero_i32, desc_addr, 36);

    // Call nsl_desc_to_tensor(&desc) → NslTensor*.
    let desc_to_tensor_ref = module.declare_func_in_func(desc_to_tensor_id, b.func);
    let call = b.ins().call(desc_to_tensor_ref, &[desc_addr]);
    let tensor_handle = b.inst_results(call)[0];

    // Call model_forward(tensor_handle). Retention splice inside
    // populates __nsl_calib_retention_arena.
    let model_forward_ref = module.declare_func_in_func(model_forward_id, b.func);
    b.ins().call(model_forward_ref, &[tensor_handle]);
    // Return value discarded.

    // Free the wrapper tensor struct (borrow; data not freed).
    let tensor_free_ref = module.declare_func_in_func(tensor_free_id, b.func);
    b.ins().call(tensor_free_ref, &[tensor_handle]);

    b.ins().return_(&[]);
    b.finalize();
    module.define_function(wrapper_id, &mut ctx)?;
}
```

`declare_arena_shape_rodata` is a new helper in the same file that emits an `i64[3]` rodata global with the batch, seq, channels values from `ArenaLayout`.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(calib): nsl_calib_model_forward wrapper body — M62 FFI reuse"
```

---

### Task 8: Emit model_forward into calib_model.o

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/binary_codegen.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn calib_model_object_contains_model_forward_body() {
    let ast = parse_fixture("tests/fixtures/awq_calibration_mlp.nsl");
    let arena_layout = build_arena_layout(&projections, 8, 4);
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("calib_model.o");
    emit_calibration_model_object(&ast, &opts, &arena_layout, &out_path).unwrap();

    let obj_bytes = std::fs::read(&out_path).unwrap();
    let obj = object::File::parse(&*obj_bytes).unwrap();
    let exports: Vec<&str> = obj.symbols()
        .filter(|s| s.is_global() && !s.is_undefined())
        .filter_map(|s| s.name().ok())
        .collect();
    assert!(
        exports.contains(&"model_forward"),
        "calib_model.o must export model_forward (compiled from AST)"
    );
}
```

- [ ] **Step 2: Run — expect FAIL**

Task 7 declared `model_forward_id` via `declare_func_in_func` but the function was not defined. Test fails because `model_forward` isn't in the object's export table.

- [ ] **Step 3: Compile model_forward into the calib_model.o module**

Add a call before the wrapper body emission:

```rust
// Compile the user's model_forward + weight globals into this object.
// Uses the same machinery as the main codegen path, redirected to the
// calibration ObjectModule.
let model_forward_id = crate::compiler::compile_model_forward_into(
    &mut module,
    ast,
    opts,
    /* redirect retention splice to this object's __nsl_calib_retention_arena */ true,
)?;
```

`compile_model_forward_into` is a new entry point in `crates/nsl-codegen/src/compiler/`. It reuses the existing model-codegen passes but targets a caller-provided `ObjectModule` rather than the main output module, and configures the retention splice to reference `__nsl_calib_retention_arena` (declared in this same object as `.bss`).

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(calib): compile model_forward into calib_model.o"
```

---

### Task 9: `loop_body` calls wrapper between `batch_at` and max-abs reduction

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/binary_codegen.rs`

- [ ] **Step 1: Write the failing test (IR shape)**

```rust
#[test]
fn loop_body_calls_wrapper_between_batch_at_and_max_abs() {
    let ir = emit_calibration_scaffolding_body_for_test(/* AWQ observe_plan nonempty */);
    let body = ir.loop_body_text();

    let idx_batch_at = body.find("call fn nsl_calibration_batch_at")
        .expect("batch_at call missing");
    let idx_fwd = body.find("call fn nsl_calib_model_forward")
        .expect("Gap 2: nsl_calib_model_forward call missing");
    let idx_fmax = body.find("fmax").expect("max-abs fmax missing");

    assert!(idx_batch_at < idx_fwd, "model_forward after batch_at");
    assert!(idx_fwd < idx_fmax, "model_forward before max-abs reduction");
}
```

- [ ] **Step 2: Run — expect FAIL**

Current `loop_body` at line 705-746 has no wrapper call.

- [ ] **Step 3: Edit `loop_body` per spec §4.5**

In `emit_calibration_scaffolding_object`, between `nsl_calibration_batch_at` and the `for entry in observe_plan` loop:

```rust
// Load batch_ptr + batch_len back from the stack slots the
// batch_at FFI wrote to.
let batch_ptr = b.ins().load(I64, MemFlags::trusted(), out_ptr_addr, 0);
let batch_len = b.ins().load(I64, MemFlags::trusted(), out_len_addr, 0);

// ── Gap 2 fix: call nsl_calib_model_forward on the batch. ──
// Populates __nsl_calib_retention_arena via model_forward's
// retention splice; the subsequent max-abs reduction reads
// the populated arena rather than the zero-initialized one.
let fwd_ref = module.declare_func_in_func(nsl_calib_model_forward_id, b.func);
b.ins().call(fwd_ref, &[batch_ptr, batch_len]);
```

The `nsl_calib_model_forward_id` was declared as `Linkage::Import` in Task 5.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(calib): loop_body calls model_forward between batch_at and max-abs

Closes Gap 2 from spec §1. The max-abs reduction now reads a populated
retention arena instead of the empty one that motivated the Task 6
analytical-test revert (commit 7e035855).
"
```

**Phase C merge gate:** `cargo test -p nsl-codegen` green including the three new IR-shape tests. End-to-end pipeline is now functionally correct but refusal surface is still per-2026-04-14 spec (only infrastructure errors). Commits D and E complete the work.

---

# Phase D — Refusal surface (Commit D)

Five structured three-part errors per spec §5.1-5.5. Each task adds one refusal variant, its three-part error text, a `DiagnosticCode` variant, and a unit test asserting the exact stderr format.

### Task 10: §5.1 batch-shape mismatch refusal

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/binary_codegen.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn batch_shape_mismatch_returns_code_3_with_three_part_error() {
    // Build a calibration binary, mismatched calibration data (wrong
    // element count vs ArenaLayout), run it, capture stderr.
    let outcome = run_calib_subprocess_with_wrong_batch_shape();
    assert_eq!(outcome.exit_code, 3);
    let err = outcome.stderr;
    assert!(err.contains("calibration: batch shape mismatch"));
    assert!(err.contains("requested:"));
    assert!(err.contains("expected:"));
    assert!(err.contains("found:"));
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Add batch-elem-count check in `nsl_calib_model_forward`**

After reading `batch_elem_count` in the wrapper body, compare against the ArenaLayout-derived expected count:

```rust
let expected_count = arena_layout.batch * arena_layout.seq * arena_layout.channels;
let expected_const = b.ins().iconst(I64, expected_count as i64);
let is_ok = b.ins().icmp(IntCC::Equal, batch_elem_count, expected_const);
let ok_block = b.create_block();
let err_block = b.create_block();
b.ins().brif(is_ok, ok_block, &[], err_block, &[]);

b.switch_to_block(err_block);
b.seal_block(err_block);
emit_set_error_cstr(&mut b, module, "calibration: batch shape mismatch at subprocess model-forward call.\n  requested:  run calibration forward on batch with {actual} elements\n  expected:   batch elements == ArenaLayout declared shape (...)\n  found:      ..."); 
// exit code 3 returned via calibration_main
let three_i32 = b.ins().iconst(I32, 3);
b.ins().return_(&[three_i32]);   // propagate up

b.switch_to_block(ok_block);
b.seal_block(ok_block);
// ...rest of wrapper body (stack-allocate desc, call M62 FFIs, model_forward, free)
```

Error-string interpolation at compile time: substitute `{actual}` and other values known at codegen. Runtime-dependent values (actual count) need `nsl_set_error_formatted` — a new FFI OR use stderr-write directly via existing `nsl_write_file` to stderr path `/dev/stderr`.

Simpler alternative: emit three pre-formatted error strings for common mismatches (too small / too large / exact expected), pick the right one at runtime. If the alternative is chosen, document the trade-off in the task's commit message.

Add `DiagnosticCode::CalibBatchShapeMismatch` alongside in `wggo_overrides.rs`-analogous location.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(calib): §5.1 batch-shape-mismatch refusal"
```

---

### Task 11: §5.2 missing wrapper symbol at link time refusal

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/binary_codegen.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn missing_wrapper_symbol_at_link_emits_three_part_error() {
    // Force-break emit_calibration_model_object so it DOESN'T export
    // nsl_calib_model_forward (test-only flag). Then try link_calibration_binary.
    let err = run_link_with_missing_wrapper().expect_err("link should fail");
    assert!(matches!(err, HarnessError::Infrastructure { .. }));
    let err_str = format!("{err}");
    assert!(err_str.contains("calibration: model-forward wrapper missing from calib_model.o"));
    assert!(err_str.contains("requested:"));
    assert!(err_str.contains("expected:"));
    assert!(err_str.contains("found:"));
}
```

- [ ] **Step 2: Run — expect FAIL**

Current link error is `HarnessError::Infrastructure` with a generic linker-error string.

- [ ] **Step 3: Catch the link error in `link_calibration_binary` and reformat**

```rust
pub(crate) fn link_calibration_binary(...) -> Result<(), HarnessError> {
    let result = crate::linker::link_multi(&[scaffolding_obj, calib_model_obj], out_binary);
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            let err_str = format!("{e}");
            if err_str.contains("nsl_calib_model_forward") {
                // Three-part error per spec §5.2.
                Err(HarnessError::Infrastructure {
                    reason: format!(
                        "calibration: model-forward wrapper missing from calib_model.o.\n\
                         \x20 requested:  link scaffolding.o ← calib_model.o with nsl_calib_model_forward resolved\n\
                         \x20 expected:   calib_model.o exports nsl_calib_model_forward (the f32-buffer wrapper around model_forward)\n\
                         \x20 found:      {err_str}"
                    ),
                })
            } else {
                Err(HarnessError::Infrastructure { reason: err_str })
            }
        }
    }
}
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(calib): §5.2 missing-wrapper-symbol refusal at link time"
```

---

### Task 12: §5.3 discovery divergence refusal

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/discovery.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn discovery_divergence_emits_three_part_error() {
    // Synthesize an AST where the pre-scan and in-compile paths can be
    // forced to disagree (test-only injection). Assert refusal text.
    let pre_scan = vec![fake_projection("A.x", [4, 4])];
    let in_compile = vec![fake_projection("A.y", [4, 4])];
    let err = check_discovery_agreement(&pre_scan, &in_compile)
        .expect_err("must refuse");
    let err_str = format!("{err}");
    assert!(err_str.contains("calibration: discovery divergence"));
    assert!(err_str.contains("pre-scan:"));
    assert!(err_str.contains("in-compile:"));
}
```

- [ ] **Step 2: Run — expect FAIL**

`check_discovery_agreement` doesn't exist yet.

- [ ] **Step 3: Implement + wire into entry points**

```rust
pub fn check_discovery_agreement(
    pre_scan: &[DiscoveredProjection],
    in_compile: &[DiscoveredProjection],
) -> Result<(), DiscoveryError> {
    if pre_scan == in_compile {
        return Ok(());
    }
    Err(DiscoveryError::Divergence {
        pre_scan_names: pre_scan.iter().map(|p| p.projection.0.clone()).collect(),
        in_compile_names: in_compile.iter().map(|p| p.projection.0.clone()).collect(),
    })
}
```

Wire into `compile_train_block` right before the in-compile `discover_awq_projections` call: compare the result against `compiler.compile_options.calibration_retention` (set by the AST pre-scan in Task 2). Fail compilation on divergence.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(calib): §5.3 discovery-divergence refusal"
```

---

### Task 13: §5.4 empty observe_plan with forward-pass emitted (defensive invariant)

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/binary_codegen.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn empty_observe_plan_with_forward_pass_refuses() {
    // Synthetic hook registry where requires().needs_forward_pass() == true
    // but observe_plan() returns empty vec (hook-implementation drift).
    let registry = drifted_hook_registry();
    let err = emit_calibration_scaffolding_object(
        &[], &[], &arena_layout, b"{}", true, &out_path,
    ).expect_err("must refuse defensively");
    assert!(matches!(err, HarnessError::Infrastructure { .. }));
    assert!(format!("{err}").contains("calibration: forward pass emitted but no observations declared"));
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Add defensive check at top of `emit_calibration_scaffolding_object`**

```rust
if needs_forward && observe_plan.is_empty() {
    return Err(HarnessError::Infrastructure {
        reason: format!(
            "calibration: forward pass emitted but no observations declared.\n\
             \x20 requested:  run calibration subprocess with forward pass\n\
             \x20 expected:   observe_plan is non-empty when a model_forward call is emitted\n\
             \x20 found:      observe_plan is empty but needs_forward_pass() returned true.\n\
             \x20              Did a hook's requires() change without updating observe_plan()?"
        ),
    });
}
```

Comment inline: "Defensive invariant — unreachable under the current hook surface (both needs_forward_pass and observe_plan derive from the same requires() query); fires only on hook-implementation drift. Spec §5.4."

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(calib): §5.4 defensive-invariant refusal for hook drift"
```

---

### Task 14: §5.5 @quantize model decorator with zero projections refusal

**Files:**

- Modify: `crates/nsl-codegen/src/compiler/entry_points.rs`
- Modify: `crates/nsl-codegen/src/calibration/discovery.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn quantize_decorator_with_zero_linear_layers_refuses() {
    let source = r#"
        @quantize(dtype="awq4")
        model EmptyModel:
            fn forward(self, x: Tensor) -> Tensor: return x

        fn main(): let m = EmptyModel()
    "#;
    let err = compile_returning_plan(source, "<empty>", &mut Default::default())
        .expect_err("must refuse");
    assert!(format!("{err}").contains("calibration: @quantize model declared but no AWQ projections discovered"));
    assert!(format!("{err}").contains("Action:"));
}
```

- [ ] **Step 2: Run — expect FAIL**

Compilation currently succeeds silently (auto-discovery returns empty vec; no hard fail).

- [ ] **Step 3: Hard-fail when decorator is present but discovery returns empty**

In the `populate_calibration_retention_from_ast_if_unset` helper from Task 2:

```rust
fn populate_calibration_retention_from_ast_if_unset(
    compiler: &mut Compiler,
    ast: &Ast,
) -> Result<(), CodegenError> {
    if compiler.compile_options.calibration_retention.is_some() { return Ok(()); }
    let discovered = crate::calibration::pre_scan_awq_projections_from_ast(ast);
    let has_decorator = ast_has_quantize_awq_decorator(ast);
    match (has_decorator, discovered.is_empty()) {
        (true, true) => Err(CodegenError::new(format!(
            "calibration: @quantize model declared but no AWQ projections discovered.\n\
             \x20 requested:  auto-discover AWQ projections from {source_file}\n\
             \x20 expected:   @quantize model + AWQ config ⇒ at least one DiscoveredProjection\n\
             \x20              (if the model has no projections to quantize, the decorator should be removed)\n\
             \x20 found:      decorator present at {span}; zero projections emitted.\n\
             \x20              Action: either remove the @quantize model decorator, or add\n\
             \x20              the Linear/tensor projections the decorator is meant to target.",
            source_file = ast.source_name(), span = decorator_span(ast),
        ))),
        (true, false) => {
            compiler.compile_options.calibration_retention = Some(discovered);
            Ok(())
        }
        (false, _) => Ok(()),
    }
}
```

Return type changes from `()` to `Result<(), CodegenError>`; callers propagate via `?`.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(calib): §5.5 @quantize-decorator-no-projections hard-fail refusal

Spec §5.5 is a hard-fail (not warn-and-proceed). An earlier draft's
'legitimately guarded decorator' rationale was rejected — no such NSL
syntax exists, and warning-that-proceeds is the fallback-with-different-
semantics pattern the spec's §11 principle #1 rejects.
"
```

**Phase D merge gate:** 5/5 refusals have three-part errors + unit tests. `cargo test -p nsl-codegen` green. Manual smoke: compile a model with each refusal trigger, confirm all refusals print in one pass (no fix-one-recompile loop).

---

# Phase E — Merge gate: Task 6 re-land (Commit E)

Restores the analytical end-to-end test that commit `7e035855` reverted and the `reference_awq_scales` helper that went with it.

### Task 15: `reference_awq_scales` analytical helper

**Files:**

- Modify: `crates/nsl-codegen/tests/awq_full_pipeline.rs`

- [ ] **Step 1: Recover the helper from commit e36634a6**

```bash
git show e36634a6 -- crates/nsl-codegen/tests/awq_full_pipeline.rs \
    | grep -A 40 "fn reference_awq_scales"
```

Paste the recovered helper back into `awq_full_pipeline.rs`. The helper body should be a pure-Rust sequential-summation implementation of AWQ scale computation matching the 2026-04-14 spec §5.2.

- [ ] **Step 2: Verify helper compiles**

```
cargo check -p nsl-codegen --tests
```

- [ ] **Step 3: Commit (just the helper, not the test yet)**

```bash
git commit -m "test(calib): re-land reference_awq_scales helper (from reverted e36634a6)"
```

---

### Task 16: `end_to_end_real_subprocess_matches_analytical_reference` (merge gate)

**Files:**

- Modify: `crates/nsl-codegen/tests/awq_full_pipeline.rs` (re-add test + cleanup stale Blocker A/B comments per spec §6.5)

- [ ] **Step 1: Re-add the test + tolerance rationale comment**

```rust
/// Tolerance: 5e-6 — f32 matmul over 64-length reductions accumulates
/// ~4 ULPs of round-off vs. a pairwise accumulator; the subprocess
/// runs the same fabs/fmax loop as the analytical reference, so the
/// dominant source of drift is model_forward's matmul reordering, not
/// the reduction. Fixture dimensions:
///   up_proj:   [128, 64]  — K=64 reduction in the forward matmul
///   down_proj: [64, 128]  — K=128 reduction, tighter but still < 5e-6
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

    let calib = read_safetensors_flat(
        "tests/fixtures/awq_calib_data.safetensors", "calibration",
    );
    let up_w = read_safetensors_flat(
        "tests/fixtures/awq_calib_weights.safetensors", "TinyMLP.up_proj",
    );
    let (up_ref, down_ref) = reference_awq_scales(&calib, &up_w);

    let up_actual = awq_scales(&sidecar, "TinyMLP.up_proj");
    let down_actual = awq_scales(&sidecar, "TinyMLP.down_proj");

    // 5e-6 tolerance: f32 matmul over length-64 inner products accumulates
    // ~4 ULPs of round-off vs. the analytical reference's sequential summation.
    assert_close(&up_actual,   &up_ref,   5e-6, "up_proj");
    assert_close(&down_actual, &down_ref, 5e-6, "down_proj");
}
```

If the test cannot run in CI because of linker availability (common on Windows runners without MSVC), gate it behind `#[cfg_attr(not(has_msvc_linker), ignore)]` or equivalent — NEVER delete. Deletion without equivalent coverage recreates the exact bug class the Task 6 revert introduced.

- [ ] **Step 2: Update stale Blocker A/B comments at lines 13-14**

Replace:

```rust
//! Blocker A (compile_and_calibrate sidecar recovery) and Blocker B
//! (emit_observe_batch real IR) are both addressed:
//!   * Blocker A: compile_and_calibrate now constructs Compiler directly
//!     and reads back compile_options.calibration_sidecar.
//!   * Blocker B: real_subprocess_entry calls
//!     build_sidecar_from_forward_observation which loads calibration data
//!     and drives emit_observe_batch per batch with real f32 values.
```

with:

```rust
//! Architecture (post-2026-04-22 completion work):
//!   * Two-object compile: scaffolding.o (calibration_main) + calib_model.o
//!     (model_forward + nsl_calib_model_forward wrapper + retention arena)
//!     linked via link_calibration_binary. See spec/plan pair at
//!     docs/superpowers/specs/2026-04-22-awq-real-subprocess-completion-design.md.
//!   * loop_body calls nsl_calib_model_forward between nsl_calibration_batch_at
//!     and the plan-driven max-abs reduction, populating the retention arena
//!     via the model_forward splice. Empty-arena regression (the Task 6 revert
//!     signal) is impossible by construction.
```

- [ ] **Step 3: Run the test — expect PASS**

```
cargo test -p nsl-codegen --test awq_full_pipeline end_to_end_real_subprocess_matches_analytical_reference -- --nocapture
```

Expected: PASS within 5e-6 tolerance.

- [ ] **Step 4: Run the full crate test suite — expect ALL green**

```
cargo test -p nsl-codegen
```

Expected: every test passes (merge gate).

- [ ] **Step 5: Commit**

```bash
git commit -m "test(calib): re-land Task 6 analytical end-to-end + cleanup stale comments

Merge gate per spec §6.4 + §10 criterion #1. Subprocess scales match
the analytical reference within 5e-6 relative tolerance — the 4-ULP
K=64 matmul accumulation bound.

Closes the completion work started by PR #29 and blocked on the three
gaps the 2026-04-21 audit found. The reverted commit e36634a6 had no
test evidence because Gap 2 left the arena empty; with Commit C's
wrapper-call in place, the test passes and the revert's root cause
is fixed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

## Success criteria (merge gate — from spec §10)

1. Commit E's analytical test passes within 5e-6 relative tolerance.
2. §6.1 IR-shape test: wrapper call appears between batch_at and max-abs in loop_body (Task 9).
3. §6.2 two-object link test passes on Linux + macOS (Task 6).
4. §6.3 differential test: pre-scan vs in-compile agreement on all fixtures (Task 3).
5. All 5 refusal variants (§5.1-5.5) produce three-part errors verified by unit tests (Tasks 10-14).
6. All 2026-04-14 spec §5.4 negative tests continue to pass (regression gate): subprocess crash surfaces up, `nsl_awq_write_sidecar` bad-path returns 3, no CUDA linkage on Linux.
7. Existing regression tests from PR #28, PR #29, and PR #98 all pass.
8. `py -m pytest python/tests/test_m62_export.py` continues to pass (M62 regression gate).
9. `awq_full_pipeline.rs` stale "Blocker A/B" comments replaced (Task 16 Step 2).

---

## Implementer notes

- **Commit B intermediate state is load-bearing.** Between Commit B and Commit C, hand-running calibration produces empty sidecars because `nsl_calib_model_forward` is the Task 4 stub. This is documented in spec §9 Commit B. Don't panic if a smoke test shows empty scales after Commit B; Commit C closes the gap.

- **Link-order assumption.** Canonical order is `&[scaffolding.o, calib_model.o]` per spec §7.3. If link fails at implementation time (LNK2019 on Windows, undefined reference on Linux), DO NOT flip the order as a workaround — the correct answer per MSVC + GNU ld .o cross-resolution semantics is this order, so a failure indicates a missing export or mismatched signature, not a link-order bug. Fail loudly and investigate.

- **WGGO Phase 2 downstream impact.** Once Commit E lands, `wggo_head_gradients` in the sidecar starts returning populated data; `CalibratedGradientScorer` in WGGO Phase 2 silently switches from MagnitudeFallback to real gradient-based scoring. Existing WGGO Phase 2 tests that asserted MagnitudeFallback output need review as a downstream follow-up (not part of this plan's scope). See spec §13.

- **Test cost is bounded.** All integration tests run on CPU. The analytical test compiles + links + spawns a subprocess + reads a sidecar — ~5-10 seconds on a modern dev machine. No CUDA, no GPU, no network. Safe to run on every PR.
