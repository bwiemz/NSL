# M62 `@export` Model-Method Weight Loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `@export` model methods (`model Net: @export fn predict(self, x): self.W @ x`) load weights from `NslModel*` at runtime and produce correct results when called from C/Python.

**Architecture:** Three-layer change. (1) **Semantic**: require `self` on `@export` model methods; reject self-less with a pointed diagnostic; resolve `self.<field>` to a compile-time weight index. (2) **Codegen**: declare TWO functions per `@export` model method — normal `__nsl_model_<M>_<m>` (unchanged) + new `__nsl_export_impl_<M>_<m>` with leading `(weight_ptrs: i64, num_weights: i64)` args; a new `SelfResolution::WeightPtrsArray` mode lowers `self.W` to `load(weight_ptrs + field_index*8)`. (3) **Wrapper**: `ExportWrapper.is_model_method` flag; when true, wrapper emits two new FFI calls (`nsl_model_get_weight_ptrs`, `nsl_model_get_num_weights`) and prepends the results to the impl call's args.

**Tech Stack:** Rust, Cranelift, NSL runtime FFI, safetensors, ctypes.

**Reference spec:** [docs/superpowers/specs/2026-04-16-m62-weight-loading-design.md](../specs/2026-04-16-m62-weight-loading-design.md)

**Branching note:** This plan depends on the `feat/m62-c-wrappers` PR landing first (it extends `c_wrapper.rs` / `ExportWrapper`). Implement on a fresh worktree branched from `main` AFTER the c-wrappers PR merges, OR stack on top of `feat/m62-c-wrappers` during development and rebase after merge.

---

## File Structure

**Create:**
- `python/tests/fixtures/m62_predict_with_weights.nsl` — E2E NSL fixture.
- `python/tests/test_m62_weight_loading.py` — E2E Python test (+ safetensors writer helper).

**Modify:**
- `crates/nsl-runtime/src/c_api.rs` — two new FFIs.
- `crates/nsl-semantic/src/export.rs` — three-way @export rules, weight-index resolver, narrow Task 7 warning.
- `crates/nsl-codegen/src/compiler/declaration.rs` — extend is_export branch to handle model methods (two-function pattern extension).
- `crates/nsl-codegen/src/compiler/functions.rs` — new `compile_export_model_methods` that reuses method body via the new self-resolution mode.
- `crates/nsl-codegen/src/compiler/mod.rs` — `FuncState::self_resolution: SelfResolution` field.
- `crates/nsl-codegen/src/expr/access.rs` + `crates/nsl-codegen/src/expr/mod.rs` — new branch in `SelfRef` / `FieldAccess` lowering.
- `crates/nsl-codegen/src/c_wrapper.rs` — `ExportWrapper.is_model_method` + wrapper-body weight-ptrs emission.
- `crates/nsl-codegen/src/c_header.rs::emit()` — prepend "inference-only" comment to each `@export` prototype.

---

## Task 1: Runtime FFIs — `nsl_model_get_weight_ptrs` + `nsl_model_get_num_weights`

**Files:**
- Modify: `crates/nsl-runtime/src/c_api.rs`

- [ ] **Step 1: Add the two FFI functions**

In `crates/nsl-runtime/src/c_api.rs` near existing `nsl_model_*` functions (`nsl_model_create`, `nsl_model_forward`):

```rust
/// Returns a pointer to the model's weight_ptrs array. The array contains
/// `nsl_model_get_num_weights(model)` entries; each is an i64 that is a
/// *mut NslTensor. The pointer is valid for the model's lifetime.
#[no_mangle]
pub extern "C" fn nsl_model_get_weight_ptrs(model_ptr: i64) -> i64 {
    if model_ptr == 0 {
        set_error("nsl_model_get_weight_ptrs: null model\0".to_string());
        return 0;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    model.weight_ptrs.as_ptr() as i64
}

/// Returns the number of weights in the model.
#[no_mangle]
pub extern "C" fn nsl_model_get_num_weights(model_ptr: i64) -> i64 {
    if model_ptr == 0 {
        set_error("nsl_model_get_num_weights: null model\0".to_string());
        return 0;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    model.weight_ptrs.len() as i64
}
```

- [ ] **Step 2: Write tests**

Append to the existing test module in `c_api.rs` (reuse pattern from Task 3 of the c-wrappers plan — look for `nsl_set_error_cstr_sets_thread_local` as a template):

```rust
#[test]
fn nsl_model_get_weight_ptrs_returns_valid_pointer() {
    // Use the in-test constructor pattern from existing tests (grep for
    // `weight_ptrs: vec![...]` in tests; there's one around line 1001).
    let w = Box::into_raw(Box::new(dummy_nsl_tensor())) as i64;
    let model = Box::into_raw(Box::new(NslModel {
        weight_ptrs: vec![w],
        ..NslModel::empty_for_test()
    }));
    let got = nsl_model_get_weight_ptrs(model as i64);
    assert_ne!(got, 0);
    let first: i64 = unsafe { *(got as *const i64) };
    assert_eq!(first, w);
    unsafe { let _ = Box::from_raw(model); let _ = Box::from_raw(w as *mut _); }
}

#[test]
fn nsl_model_get_num_weights_returns_length() {
    let model = Box::into_raw(Box::new(NslModel {
        weight_ptrs: vec![0x1, 0x2, 0x3],
        ..NslModel::empty_for_test()
    }));
    assert_eq!(nsl_model_get_num_weights(model as i64), 3);
    unsafe { let _ = Box::from_raw(model); }
}

#[test]
fn nsl_model_get_weight_ptrs_null_returns_zero_sets_error() {
    let got = nsl_model_get_weight_ptrs(0);
    assert_eq!(got, 0);
    // Verify error string was set (existing test idiom)
}
```

If `NslModel::empty_for_test()` or `dummy_nsl_tensor()` helpers don't exist, follow the pattern of existing tests that construct `NslModel` directly (grep the test module for existing `NslModel { weight_ptrs: vec![...], ... }` literals around c_api.rs:889, 916, 1001).

- [ ] **Step 3: Run**

Run: `cargo test -p nsl-runtime nsl_model_get --lib`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/c_api.rs
git commit -m "feat(m62): nsl_model_get_weight_ptrs + nsl_model_get_num_weights FFIs"
```

---

## Task 2: Semantic — three-way `@export` rules + self-less rejection

**Files:**
- Modify: `crates/nsl-semantic/src/export.rs`

- [ ] **Step 1: Write failing tests**

In the existing `#[cfg(test)] mod tests` in `crates/nsl-semantic/src/export.rs`:

```rust
#[test]
fn export_model_method_without_self_produces_error() {
    let src = r#"
model Net:
    W: Tensor<[4, 4], f32>

    @export
    fn helper(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = parse_and_validate_for_test(src);
    let errors: Vec<_> = diags.iter().filter(|d| d.is_error()).collect();
    assert!(
        errors.iter().any(|d| {
            let m = d.message();
            m.contains("requires `self`") && m.contains("helper")
        }),
        "expected self-required error, got: {:?}",
        diags
    );
}

#[test]
fn export_model_method_with_self_accesses_weights_no_warning() {
    let src = r#"
model Net:
    W: Tensor<[4, 4], f32>

    @export
    fn predict(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return self.W @ x
"#;
    let diags = parse_and_validate_for_test(src);
    let weight_warnings: Vec<_> = diags
        .iter()
        .filter(|d| d.is_warning() && d.message().contains("weight"))
        .collect();
    assert!(
        weight_warnings.is_empty(),
        "@export model method with self should not warn; got: {:?}",
        diags
    );
}

#[test]
fn export_top_level_with_self_still_warns() {
    // Existing test path — top-level @export referencing self-like param
    // still gets the Task 7 warning.
    let src = r#"
model Net:
    W: Tensor<[4, 4], f32>

@export
fn predict_toplevel(net: Net, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return net.W @ x
"#;
    let diags = parse_and_validate_for_test(src);
    let warnings: Vec<_> = diags.iter().filter(|d| d.is_warning()).collect();
    assert!(
        warnings.iter().any(|d| d.message().contains("weight")),
        "top-level @export with model field access should still warn; got: {:?}",
        diags
    );
}
```

- [ ] **Step 2: Run → expect failure**

Run: `cargo test -p nsl-semantic export --lib`
Expected: 3 new tests fail (no validation rule emits the error/suppresses the warning).

- [ ] **Step 3: Extend `validate_exports`**

In `crates/nsl-semantic/src/export.rs`, add a new helper that walks model bodies looking for `@export`-decorated methods and validates them:

```rust
fn validate_export_on_model_methods(
    stmts: &[Stmt],
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for stmt in stmts {
        if let StmtKind::ModelDef(md) = &stmt.kind {
            for member in &md.members {
                if let ModelMember::Method(fn_def, decos) = member {
                    let has_export = decos.iter().any(|d| {
                        d.name.len() == 1 && interner.resolve(d.name[0]) == "export"
                    });
                    if !has_export { continue; }

                    let method_name = interner.resolve(fn_def.name);
                    let has_self = fn_def.params
                        .first()
                        .map(|p| interner.resolve(p.name) == "self")
                        .unwrap_or(false);

                    if !has_self {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "@export model method '{method_name}' requires `self` as first parameter"
                            ))
                            .with_label(fn_def.span, "missing self")
                            .with_note("@export on methods without `self` is not yet supported.")
                            .with_help(
                                "move this to a top-level `@export fn` (not yet supported \
                                 either; see follow-up issue) or add `self` as the first parameter."
                            ),
                        );
                    }
                }
            }
        }
    }
}
```

Adapt to the real `Diagnostic` builder API (the c-wrappers PR's `check_no_model_weight_access` in the same file shows the correct builder method names — use the same ones here).

Call `validate_export_on_model_methods` from the existing `validate_exports` entrypoint, after the current top-level validation.

- [ ] **Step 4: Narrow the Task 7 `check_no_model_weight_access` warning**

In the same file, find `check_no_model_weight_access` (from the c-wrappers PR Task 7). Today it walks every `@export` function's body. Add a guard: skip walking if the function is a model method with `self` as first param. The easiest way: the caller of `check_no_model_weight_access` is inside `validate_fn_signature`, which already knows the enclosing context. If it's a model-method context AND the function has `self` as first param, return early without walking.

If the caller doesn't have that context available, add a new parameter: `fn check_no_model_weight_access(..., is_model_method_with_self: bool)`. When `true`, skip the walk.

Adapt call sites accordingly.

- [ ] **Step 5: Run tests → expect pass**

Run: `cargo test -p nsl-semantic export --lib`
Expected: 12 passed (9 existing from c-wrappers PR + 3 new).

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-semantic/src/export.rs
git commit -m "feat(m62): @export model-method semantic rules + narrow weight-warning"
```

---

## Task 3: Semantic — resolve `self.<field>` to weight index

**Files:**
- Modify: `crates/nsl-semantic/src/export.rs` (or wherever type-map/AST annotations live)

- [ ] **Step 1: Write failing test for index resolution**

In `crates/nsl-semantic/src/export.rs::tests`:

```rust
#[test]
fn export_method_self_field_access_resolved_to_declaration_index() {
    let src = r#"
model Net:
    W: Tensor<[4, 4], f32>
    b: Tensor<[4], f32>
    gamma: Tensor<[4], f32>

    @export
    fn predict(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return (self.W @ x) + self.b
"#;
    let analyzed = parse_and_analyze_for_test(src);  // existing or new helper
    let predict_fn = find_model_method(&analyzed, "Net", "predict");
    let weight_accesses = collect_self_field_accesses(predict_fn);
    // Expected: self.W → index 0, self.b → index 1. (gamma is declared but unused; still has index 2.)
    assert!(weight_accesses.iter().any(|a| a.field == "W" && a.weight_index == 0));
    assert!(weight_accesses.iter().any(|a| a.field == "b" && a.weight_index == 1));
}

#[test]
fn export_method_accesses_non_tensor_field_errors() {
    let src = r#"
model Net:
    name: string
    W: Tensor<[4, 4], f32>

    @export
    fn predict(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return self.name  # Error: `name` is not a weight tensor
"#;
    let diags = parse_and_validate_for_test(src);
    assert!(
        diags.iter().any(|d| d.is_error() && d.message().contains("not a weight tensor")),
        "expected error on non-tensor field access, got: {:?}",
        diags
    );
}
```

If there's no existing `parse_and_analyze_for_test` or AST-annotation query helper, use whatever pattern the existing tests use to inspect post-validation AST state (e.g. query a type map).

- [ ] **Step 2: Run → expect failure**

Run: `cargo test -p nsl-semantic weight_index --lib`
Expected: failures.

- [ ] **Step 3: Add weight-index resolver**

In `validate_export_on_model_methods` from Task 2, after the `has_self` check passes, walk the method body. For each `FieldAccess` where the receiver is `SelfRef` (or ident resolving to `self`), look up the field in the enclosing `ModelDef.members`:

```rust
fn resolve_self_field_weight_indices(
    md: &ModelDef,
    fn_def: &FnDef,
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
    type_map: &mut TypeMap,  // or wherever per-expr annotations live
) {
    // Collect (declaration-order) list of tensor-field names for this model
    let mut weight_fields: Vec<String> = Vec::new();
    for member in &md.members {
        if let ModelMember::Field(field) = member {
            if field_is_tensor(field) {
                weight_fields.push(interner.resolve(field.name).to_string());
            }
        }
    }

    // Walk the method body, annotate each self.<field> access with its weight index
    walk_exprs_in_fn_body(fn_def, |expr| {
        if let ExprKind::FieldAccess { receiver, field } = &expr.kind {
            if matches!(receiver.kind, ExprKind::SelfRef) {
                let field_name = interner.resolve(*field).to_string();
                match weight_fields.iter().position(|f| f == &field_name) {
                    Some(idx) => {
                        type_map.annotate_self_field_index(expr.id, idx);
                    }
                    None => {
                        diagnostics.push(Diagnostic::error(format!(
                            "@export method references `self.{field_name}` but it is not a weight tensor"
                        )).with_label(expr.span, "not a weight tensor"));
                    }
                }
            }
        }
    });
}
```

The `type_map.annotate_self_field_index(...)` call is new — either add a hashmap `self_field_indices: HashMap<NodeId, usize>` to the existing type-map/annotation struct, or create a new side-table on the analyzed AST. Pick whichever the codebase already uses for similar per-NodeId annotations (grep for `node_id.*HashMap` patterns).

- [ ] **Step 4: Run tests → expect pass**

Run: `cargo test -p nsl-semantic --lib`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-semantic/src/
git commit -m "feat(m62): resolve @export method self.field to weight index"
```

---

## Task 4: Codegen — `SelfResolution` mode + `self.W` lowering

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/mod.rs` (add `SelfResolution` enum + field on `FuncState`)
- Modify: `crates/nsl-codegen/src/expr/mod.rs` (extend `SelfRef` branch)
- Modify: `crates/nsl-codegen/src/expr/access.rs` (extend `FieldAccess` branch)

- [ ] **Step 1: Add `SelfResolution` enum**

In `crates/nsl-codegen/src/compiler/mod.rs`, add above `FuncState`:

```rust
#[derive(Debug, Clone)]
pub enum SelfResolution {
    /// Normal model method: `self` is bound to block_params[0] as a pointer
    /// into a model-struct layout. Field accesses go through struct_layouts.
    StructPointer,
    /// @export model method: `self` is a phantom — there's no self pointer.
    /// `self.<field>` lowers to `load(weight_ptrs + field_index*8)` where
    /// `weight_ptrs_var` holds the Cranelift variable carrying the leading
    /// arg, and field indices are read from the semantic annotation.
    WeightPtrsArray {
        weight_ptrs_var: cranelift_frontend::Variable,
    },
}
```

Add to `FuncState`:

```rust
pub self_resolution: SelfResolution,
```

Initialize to `SelfResolution::StructPointer` in `FuncState::new()`.

- [ ] **Step 2: Extend `SelfRef` lowering in `expr/mod.rs`**

At `expr/mod.rs:171` (the current `SelfRef` branch):

```rust
ExprKind::SelfRef => {
    match &state.self_resolution {
        SelfResolution::StructPointer => {
            // Existing behavior — find self var in state
            for (sym, (var, _ty)) in &state.variables {
                if self.resolve_sym(*sym) == "self" {
                    return Ok(builder.use_var(*var));
                }
            }
            Err(CodegenError::new("`self` used outside of model method"))
        }
        SelfResolution::WeightPtrsArray { .. } => {
            // `self` has no value in @export methods — it's a phantom.
            // Any bare SelfRef expression (not a FieldAccess target) is an error.
            Err(CodegenError::new(
                "@export method cannot use bare `self` — only `self.<weight_field>` access is supported"
            ))
        }
    }
}
```

- [ ] **Step 3: Extend `FieldAccess` in `expr/access.rs`**

Find the existing `FieldAccess` branch (around line 59+ with `obj_val = self.compile_expr(...)`). Add a special-case BEFORE the struct-layout path:

```rust
// @export weight-array self resolution
if matches!(object.kind, ExprKind::SelfRef) {
    if let SelfResolution::WeightPtrsArray { weight_ptrs_var } = &state.self_resolution {
        let idx = self.types.get_self_field_weight_index(expr.id)
            .ok_or_else(|| CodegenError::new(format!(
                "@export method: self.{} missing weight-index annotation (semantic bug)",
                self.resolve_sym(*field)
            )))?;
        let weight_ptrs = builder.use_var(*weight_ptrs_var);
        let offset_bytes = (idx as i64) * 8;
        let tensor_ptr = builder.ins().load(
            cl_types::I64,
            MemFlags::trusted(),
            weight_ptrs,
            offset_bytes as i32,
        );
        return Ok(tensor_ptr);
    }
}
```

The call `self.types.get_self_field_weight_index(expr.id)` queries the annotation added in Task 3. Plumb this helper through `types` (or wherever the TypeMap lives).

- [ ] **Step 4: Build**

Run: `cargo build -p nsl-codegen`
Expected: clean. If `state.self_resolution` is accessed before `FuncState` default wiring is complete, fix at the call site — default is `StructPointer`.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/
git commit -m "feat(m62): SelfResolution modes + self.W codegen for @export methods"
```

---

## Task 5: Declaration — two-function pattern for `@export` model methods

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/declaration.rs`

- [ ] **Step 1: Extend the model-method declaration loop**

Find the existing loop in `declaration.rs` that iterates `ModelDef.members` and registers `__nsl_model_<M>_<m>` in `registry.functions` (look around lines 209, 247, 367). For each model method with an `@export` decorator, ALSO declare:

1. Internal impl under `__nsl_export_impl_<ModelName>_<method>` with signature `(i64, i64, ...inputs) -> output` (two leading ptrs for weight_ptrs + num_weights).
2. C-ABI wrapper under the override name or raw method name with wrapper signature (via `build_c_abi_wrapper_signature`).

Reuse the c-wrappers PR `ExportWrapper` construction path but set `is_model_method: true` (see Task 7 below for this field addition).

```rust
// In the ModelMember::Method arm, after the existing __nsl_model_... declaration:
if let Some((_is_export, override_name)) = extract_export_on_model_method(decos, self.interner) {
    // Build impl sig: (weight_ptrs, num_weights, ...inputs) -> output
    let mut impl_sig = build_method_sig_excluding_self(fn_def, self.interner, ...);
    impl_sig.params.insert(0, AbiParam::new(types::I64));  // weight_ptrs
    impl_sig.params.insert(1, AbiParam::new(types::I64));  // num_weights

    let impl_name = format!("__nsl_export_impl_{model_name}_{method_name}");
    let impl_func_id = self.module.declare_function(&impl_name, Linkage::Local, &impl_sig)?;

    // Register under a lookup key so Task 6 can find it at body-compilation time.
    self.registry.export_method_impls.insert(
        (model_name.clone(), method_name.clone()),
        (impl_func_id, impl_sig.clone()),
    );

    // Declare wrapper
    let wrapper_symbol = override_name.unwrap_or_else(|| method_name.clone());
    let info = crate::c_header::ExportInfo::from_fn_def(/* ... */);
    let call_conv = self.module.target_config().default_call_conv;
    let wrapper_sig = crate::c_wrapper::build_c_abi_wrapper_signature(&info, call_conv);
    let wrapper_func_id =
        self.module.declare_function(&wrapper_symbol, Linkage::Export, &wrapper_sig)?;

    self.features.export_wrappers.push(crate::c_wrapper::ExportWrapper {
        impl_func_id,
        impl_sig,
        wrapper_func_id,
        raw_name: method_name.clone(),
        export_info: info.clone(),
        is_model_method: true,  // see Task 7
    });
    self.features.export_functions.push(info);
}
```

`extract_export_on_model_method` mirrors `extract_export_decorator` (declaration.rs:822 from c-wrappers PR) but works on a decorator slice. `build_method_sig_excluding_self` drops the `self` struct-pointer arg that the normal method signature has; grep the existing code for how method sigs are built (likely `build_fn_signature` with a special `skip_self: true` flag or similar).

Add `export_method_impls: HashMap<(String, String), (FuncId, Signature)>` to `FunctionRegistry` (look for its definition near `pub model_methods` in `compiler/mod.rs:112`).

- [ ] **Step 2: Build**

Run: `cargo build -p nsl-codegen`
Expected: clean. Compile errors likely point at the missing `is_model_method` field — Task 7 adds it. If blocking, add that field first as a placeholder and come back.

- [ ] **Step 3: Verify no regressions on declaration**

Run: `cargo test -p nsl-codegen --lib`
Expected: still 1605 passed (Tasks 6–7 wire up body/wrapper emission; declaration-only should be additive and not break anything).

Tests that link a compiled `@export` model method won't work until Task 6 lands — if any exist and fail, `#[ignore]` with a note pointing at Task 6.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/compiler/declaration.rs
git commit -m "feat(m62): declare @export model-method impl + wrapper pair"
```

---

## Task 6: Codegen — `compile_export_model_methods`

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/functions.rs`

- [ ] **Step 1: Add the new pass**

In `crates/nsl-codegen/src/compiler/functions.rs`, alongside the existing `compile_model_methods` (line 433), add:

```rust
fn compile_export_model_methods(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
    let model_defs: Vec<_> = stmts
        .iter()
        .filter_map(|s| {
            if let StmtKind::ModelDef(md) = &s.kind { Some(md.clone()) } else { None }
        })
        .collect();

    for md in &model_defs {
        let model_name = self.resolve_sym(md.name).to_string();
        for member in &md.members {
            if let ModelMember::Method(fn_def, decos) = member {
                let is_export = decos.iter().any(|d| {
                    d.name.len() == 1 && self.resolve_sym(d.name[0]) == "export"
                });
                if !is_export { continue; }

                let method_name = self.resolve_sym(fn_def.name).to_string();
                let key = (model_name.clone(), method_name.clone());
                let (func_id, sig) = self
                    .registry
                    .export_method_impls
                    .get(&key)
                    .ok_or_else(|| CodegenError::new(format!(
                        "@export impl '{}::{}' not registered", model_name, method_name
                    )))?
                    .clone();

                let mut ctx = Context::for_function(Function::with_name_signature(
                    UserFuncName::user(0, self.next_func_index()),
                    sig.clone(),
                ));
                let mut fn_builder_ctx = FunctionBuilderContext::new();

                {
                    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
                    let mut state = FuncState::new();

                    let entry = builder.create_block();
                    builder.append_block_params_for_function_params(entry);
                    builder.switch_to_block(entry);
                    builder.seal_block(entry);
                    state.current_block = Some(entry);

                    // First two Cranelift params: weight_ptrs, num_weights
                    let weight_ptrs_val = builder.block_params(entry)[0];
                    let _num_weights_val = builder.block_params(entry)[1];

                    let weight_ptrs_var = state.new_variable();
                    builder.declare_var(weight_ptrs_var, cranelift_codegen::ir::types::I64);
                    builder.def_var(weight_ptrs_var, weight_ptrs_val);

                    state.self_resolution = SelfResolution::WeightPtrsArray {
                        weight_ptrs_var,
                    };

                    // Bind method tensor-input params (skip `self` in AST; skip first 2 cl params)
                    let mut cl_param_idx = 2usize;
                    for param in &fn_def.params {
                        let pname = self.resolve_sym(param.name).to_string();
                        if pname == "self" { continue; }
                        let param_val = builder.block_params(entry)[cl_param_idx];
                        let var = state.new_variable();
                        let ty = cranelift_codegen::ir::types::I64;  // Tensor ptrs are i64
                        builder.declare_var(var, ty);
                        builder.def_var(var, param_val);
                        state.variables.insert(param.name, (var, ty));
                        cl_param_idx += 1;
                    }

                    // Compile method body using the @export self-resolution mode
                    self.current_method_model_name = Some(model_name.clone());
                    self.compile_fn_body(&mut builder, &mut state, &fn_def.body)?;
                    self.current_method_model_name = None;

                    builder.finalize();
                }

                self.module.define_function(func_id, &mut ctx)
                    .map_err(|e| CodegenError::new(format!(
                        "define @export impl '{}::{}': {:?}", model_name, method_name, e
                    )))?;
            }
        }
    }
    Ok(())
}
```

Adapt any types / call signatures (`self.compile_fn_body`, `self.resolve_sym`, etc.) to match actual method names in the crate. The template mirrors `compile_model_methods` at line 433 — keep them structurally parallel.

- [ ] **Step 2: Wire the new pass into the compile pipeline**

`compile_export_model_methods` should run right after `compile_model_methods`. Find the caller (grep for `compile_model_methods(` in `functions.rs` or `entry_points.rs`) and add the new call immediately after:

```rust
self.compile_model_methods(stmts)?;
self.compile_export_model_methods(stmts)?;   // NEW
```

- [ ] **Step 3: Build**

Run: `cargo build -p nsl-codegen`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/compiler/functions.rs
git commit -m "feat(m62): compile_export_model_methods using WeightPtrsArray mode"
```

---

## Task 7: Wrapper — `is_model_method` + weight-ptrs threading

**Files:**
- Modify: `crates/nsl-codegen/src/c_wrapper.rs`

- [ ] **Step 1: Add `is_model_method` to `ExportWrapper`**

In `crates/nsl-codegen/src/c_wrapper.rs`, extend the struct:

```rust
#[derive(Clone, Debug)]
pub struct ExportWrapper {
    pub impl_func_id: FuncId,
    pub impl_sig: Signature,
    pub wrapper_func_id: FuncId,
    pub raw_name: String,
    pub export_info: ExportInfo,
    /// True if this wraps a model method — wrapper threads model.weight_ptrs
    /// and num_weights as leading args to the impl call.
    pub is_model_method: bool,
}
```

- [ ] **Step 2: Update wrapper-body emission**

In `emit_c_abi_wrapper`, after the null-check on `model_ptr` and BEFORE the tensor-input conversion loop, add:

```rust
// If this wraps a model method, extract weight_ptrs + num_weights from NslModel*
// and prepend them to the impl call args. @export is inference-only —
// see GRAD SCOPE note below.
let mut extra_leading_args: Vec<cranelift_codegen::ir::Value> = Vec::new();
if wrapper.is_model_method {
    let w_ptrs = call_model_get_weight_ptrs(&mut builder, &mut compiler.module, model_ptr)?;
    let n_weights = call_model_get_num_weights(&mut builder, &mut compiler.module, model_ptr)?;
    extra_leading_args.push(w_ptrs);
    extra_leading_args.push(n_weights);
}
```

Then modify the impl-call site to prepend these args:

```rust
let mut internal_args: Vec<cranelift_codegen::ir::Value> = extra_leading_args;
internal_args.extend(tensor_and_scalar_args_as_before);
```

(The existing loop already builds the tensor+scalar arg list; just extend instead of creating new.)

Add the two new helper calls below the existing `call_desc_to_tensor` / `call_tensor_to_desc_ffi` / `call_nsl_tensor_free` helpers:

```rust
fn call_model_get_weight_ptrs<M: Module + ?Sized>(
    builder: &mut FunctionBuilder,
    module: &mut M,
    model_ptr: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let fid = declare_runtime_fn(module, "nsl_model_get_weight_ptrs",
        &[cw_types::I64], &[cw_types::I64])?;
    let fref = module.declare_func_in_func(fid, builder.func);
    let call = builder.ins().call(fref, &[model_ptr]);
    Ok(builder.inst_results(call)[0])
}

fn call_model_get_num_weights<M: Module + ?Sized>(
    builder: &mut FunctionBuilder,
    module: &mut M,
    model_ptr: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let fid = declare_runtime_fn(module, "nsl_model_get_num_weights",
        &[cw_types::I64], &[cw_types::I64])?;
    let fref = module.declare_func_in_func(fid, builder.func);
    let call = builder.ins().call(fref, &[model_ptr]);
    Ok(builder.inst_results(call)[0])
}
```

- [ ] **Step 3: Add the GRAD SCOPE comment**

At the top of `emit_c_abi_wrapper`:

```rust
// GRAD SCOPE: @export dispatch is INFERENCE-ONLY. The wrapper does NOT:
//   - call nsl_tape_start over weight_ptrs
//   - save outputs to model.last_forward_outputs
//   - honor model.grad_enabled
// Calling nsl_model_enable_grad(model, 1) before an @export wrapper call
// has no effect on the @export dispatch path. For training autograd, use
// nsl_model_forward + nsl_model_backward (see the grad-context bridge fix).
```

- [ ] **Step 4: Write unit test — weight-ptrs threading IR check**

Append to the `#[cfg(test)] mod tests` in `c_wrapper.rs`:

```rust
#[test]
fn wrapper_for_model_method_threads_weight_ptrs_and_count() {
    // Build a minimal ExportWrapper with is_model_method: true
    // and synthesize a stub ExportInfo (single tensor input, tensor return).
    let info = ExportInfo {
        symbol_name: "predict".into(),
        raw_name: "predict".into(),
        params: vec![tensor_param("x", ExportDtype::F32)],
        return_type: ExportTypeInfo::Tensor {
            shape: vec!["4".into()],
            dtype: ExportDtype::F32,
            device: ExportDevice::Cpu,
        },
    };
    // Minimal compiler + module setup — mirror existing c_wrapper tests.
    let (mut compiler, impl_func_id, wrapper_func_id) = make_test_compiler_with_impl(&info);

    let wrapper = ExportWrapper {
        impl_func_id,
        impl_sig: compiler.module.declarations().get_function_decl(impl_func_id).signature.clone(),
        wrapper_func_id,
        raw_name: "predict".into(),
        export_info: info,
        is_model_method: true,
    };

    emit_c_abi_wrapper(&mut compiler, &wrapper).expect("emit ok");

    // Inspect the wrapper's function IR
    let func_ir = dump_function_ir(&compiler.module, wrapper_func_id);
    assert!(func_ir.contains("nsl_model_get_weight_ptrs"),
        "wrapper must call nsl_model_get_weight_ptrs; got:\n{func_ir}");
    assert!(func_ir.contains("nsl_model_get_num_weights"),
        "wrapper must call nsl_model_get_num_weights; got:\n{func_ir}");
}

#[test]
fn wrapper_for_non_method_does_not_call_model_accessors() {
    let info = /* same shape but is_model_method: false */;
    // ...
    let func_ir = dump_function_ir(&compiler.module, wrapper_func_id);
    assert!(!func_ir.contains("nsl_model_get_weight_ptrs"));
}

#[test]
fn wrapper_for_weightless_model_method_still_threads_args() {
    // @export fn identity(self, x): return x  — no self.<field> access
    // but is_model_method: true
    let wrapper = ExportWrapper { is_model_method: true, /* ... */ };
    emit_c_abi_wrapper(&mut compiler, &wrapper).expect("emit ok");
    let func_ir = dump_function_ir(&compiler.module, wrapper_func_id);
    assert!(func_ir.contains("nsl_model_get_weight_ptrs"),
        "weight-less @export methods must still receive weight_ptrs args");
}
```

`make_test_compiler_with_impl` and `dump_function_ir` are helpers — create them if they don't exist, mirroring patterns in other codegen tests (grep `#[cfg(test)]` in sibling files for the Cranelift `Module`-under-test pattern). If constructing a full test `Compiler` is too heavy, degrade to asserting on signature shape (test #3 would become "impl_sig.params[0].value_type == I64 && [1] == I64") and skip the IR-string check. Document the degradation in a comment.

- [ ] **Step 5: Run tests**

Run: `cargo test -p nsl-codegen c_wrapper --lib`
Expected: pre-existing 3 tests + 3 new = 6 passed.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/c_wrapper.rs
git commit -m "feat(m62): wrapper threads weight_ptrs+num_weights for model methods"
```

---

## Task 8: C header — "inference-only" comment per `@export` prototype

**Files:**
- Modify: `crates/nsl-codegen/src/c_header.rs`

- [ ] **Step 1: Update `emit()`**

In `crates/nsl-codegen/src/c_header.rs::emit()`, find the loop that emits each `@export` prototype (grep for the prototype-emission code). Prepend this comment block to each prototype:

```
/* @export dispatch is inference-only. Gradient recording does not flow
 * through this call path — use nsl_model_forward for training autograd. */
```

Code sketch (adapt to the actual emit() structure):

```rust
for export_info in &export_functions {
    writeln!(output, "/* @export dispatch is inference-only. Gradient recording does not flow")?;
    writeln!(output, " * through this call path — use nsl_model_forward for training autograd. */")?;
    // existing prototype emission:
    writeln!(output, "int {}(NslModel*, ...);", export_info.symbol_name)?;
}
```

- [ ] **Step 2: Update snapshot tests**

Run: `cargo test -p nsl-codegen c_header_snapshot --lib`
Expected: snapshot tests fail with the new comment text. If snapshots are via `insta` or similar, review + accept the diff (`cargo insta accept -p nsl-codegen` or `cargo insta review`).

If snapshots are inline-string comparisons, update the expected string to include the new comment.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/c_header.rs
git commit -m "feat(m62): @export header prototype gets inference-only comment"
```

---

## Task 9: E2E Python ctypes test — weights load + predict correct

**Files:**
- Create: `python/tests/fixtures/m62_predict_with_weights.nsl`
- Create: `python/tests/test_m62_weight_loading.py`

- [ ] **Step 1: Create the NSL fixture**

`python/tests/fixtures/m62_predict_with_weights.nsl`:

```
model Net:
    W: Tensor<[4, 4], f32>

    @export
    fn predict(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return self.W @ x
```

- [ ] **Step 2: Create the Python test**

`python/tests/test_m62_weight_loading.py`:

```python
import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import safetensors.torch as st
    import torch
except ImportError:
    pytest.skip("safetensors + torch required for this test", allow_module_level=True)


FIXTURE = Path(__file__).parent / "fixtures" / "m62_predict_with_weights.nsl"


class NslTensorDesc(ctypes.Structure):
    _fields_ = [
        ("data",        ctypes.c_void_p),
        ("shape",       ctypes.POINTER(ctypes.c_int64)),
        ("strides",     ctypes.POINTER(ctypes.c_int64)),
        ("ndim",        ctypes.c_int32),
        ("dtype",       ctypes.c_int32),
        ("device_type", ctypes.c_int32),
        ("device_id",   ctypes.c_int32),
    ]


@pytest.fixture(scope="module")
def workdir():
    with tempfile.TemporaryDirectory(prefix="m62_wl_") as d:
        yield Path(d)


@pytest.fixture(scope="module")
def W_tensor():
    # Deterministic W so we can check the matmul exactly
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((4, 4)).astype(np.float32)


@pytest.fixture(scope="module")
def weights_file(workdir, W_tensor):
    path = workdir / "weights.safetensors"
    st.save_file({"W": torch.tensor(W_tensor)}, str(path))
    return path


@pytest.fixture(scope="module")
def shared_lib(workdir):
    out = workdir / ("predict.dll" if os.name == "nt" else "predict.so")
    # Invoke the nsl binary built by cargo for this worktree
    nsl_exe = Path(__file__).parents[2] / "target" / "debug" / ("nsl.exe" if os.name == "nt" else "nsl")
    subprocess.run(
        [str(nsl_exe), "build", "--shared-lib", str(FIXTURE), "-o", str(out)],
        check=True,
    )
    return out


def _make_f32_desc(values, shape):
    data = (ctypes.c_float * len(values))(*values)
    shape_arr = (ctypes.c_int64 * len(shape))(*shape)
    desc = NslTensorDesc(
        data=ctypes.cast(data, ctypes.c_void_p),
        shape=shape_arr,
        strides=None,
        ndim=len(shape),
        dtype=0,  # f32
        device_type=0,
        device_id=0,
    )
    return desc, data, shape_arr


def test_predict_loads_weights_and_computes_W_at_x(shared_lib, weights_file, W_tensor):
    lib = ctypes.CDLL(str(shared_lib))

    lib.nsl_model_create.argtypes = [ctypes.c_int64]
    lib.nsl_model_create.restype = ctypes.c_int64
    lib.nsl_model_destroy.argtypes = [ctypes.c_int64]
    lib.nsl_model_destroy.restype = None
    lib.predict.argtypes = [
        ctypes.c_int64,                       # NslModel*
        ctypes.POINTER(NslTensorDesc),        # x
        ctypes.POINTER(NslTensorDesc),        # __ret
    ]
    lib.predict.restype = ctypes.c_int32

    # nsl_model_create takes an i64 path pointer, per existing FFI
    path_bytes = str(weights_file).encode() + b"\0"
    path_buf = (ctypes.c_char * len(path_bytes))(*path_bytes)
    model = lib.nsl_model_create(ctypes.cast(path_buf, ctypes.c_void_p).value)
    assert model != 0, "nsl_model_create failed"

    try:
        x_vals = [1.0, 2.0, 3.0, 4.0]
        x_desc, _x_buf, _x_shape = _make_f32_desc(x_vals, [4])
        ret = NslTensorDesc()

        rc = lib.predict(model, ctypes.byref(x_desc), ctypes.byref(ret))
        assert rc == 0, f"predict returned {rc}"
        assert ret.data, "ret.data is null"

        result = np.ctypeslib.as_array(
            ctypes.cast(ret.data, ctypes.POINTER(ctypes.c_float)), shape=(4,)
        ).copy()
        expected = W_tensor @ np.array(x_vals, dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)
    finally:
        lib.nsl_model_destroy(model)
```

Adapt the `nsl_model_create` argtype if the real FFI takes a `c_char_p` instead of `c_int64` pointer (grep `crates/nsl-runtime/src/c_api.rs:nsl_model_create` — the c-wrappers auto-memory notes say `i64`).

- [ ] **Step 3: Run**

Run: `py -m pytest python/tests/test_m62_weight_loading.py -v`
Expected: 1 passed.

If the shared-lib build fails, diagnose — most likely the fixture fails to compile because a prior task introduced a regression. If `predict` segfaults, suspect Task 5 (declaration) or Task 6 (body compilation) — `nsl_get_last_error` output may have clues.

If safetensors or torch are unavailable in the test env, the `pytest.skip` at the top prevents a failure — but this defeats the purpose of the test. Document in the commit if the test is skipping on CI.

- [ ] **Step 4: Commit**

```bash
git add python/tests/fixtures/m62_predict_with_weights.nsl python/tests/test_m62_weight_loading.py
git commit -m "test(m62): end-to-end @export model-method weight loading via ctypes"
```

---

## Task 10: Final regression sweep

**Files:** none (verification)

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace 2>&1 | tail -30`
Expected: no new failures beyond the 3 pre-existing CUDA panics.

- [ ] **Step 2: Full Python test**

Run: `py -m pytest python/tests/ -v 2>&1 | tail -40`
Expected: all m62 tests pass (5 from c-wrappers + 1 new E2E = 6).

- [ ] **Step 3: Walk spec §8 success criteria**

For each of spec §8's 7 success criteria, confirm it's met:
1. `cargo build --workspace` clean → yes (Steps 1-9 kept build green)
2. `@export fn predict(self, x): self.W @ x` compiles + runs + returns `W @ x` → Task 9
3. Semantic rejects self-less `@export` with clear diagnostic → Task 2
4. Weight-access warning narrows to top-level only → Task 2 step 4
5. Wrapper IR test asserts weight_ptrs + num_weights threading → Task 7 step 4
6. Pre-existing `@export` tests pass → sweep this step
7. Generated C header has "inference-only" comment → Task 8

If any fails, loop back to the owning task. Do not mark work complete while any criterion is red.

- [ ] **Step 4: No-op commit if nothing changed**

(This task has no code changes unless Step 3 surfaces an issue.)

---

## Self-Review Notes

**Spec coverage:**
- §1 Goal — Task 9 E2E closes this
- §2 Non-goals — respected throughout; no top-level @export with model param, no grad through @export, no dynamic weight access
- §3.1 Non-@export untouched — Task 4 preserves StructPointer as default; Task 5 adds a parallel declaration, doesn't replace
- §3.2 Weight ordering invariant — Task 3 resolves declaration-order; Task 9 E2E verifies nsl_model_create follows same order
- §3.3 Wrapper C-ABI unchanged — wrapper sig built via unchanged `build_c_abi_wrapper_signature`
- §3.4 Weight-less parity — Task 7 step 4 test #3
- §4.1 Semantic rules — Task 2 (three tests) + Task 3 (index resolver)
- §4.2 Internal impl signature — Task 5 (declaration) + Task 6 (compilation)
- §4.3 self.W lowering — Task 4
- §4.4 Wrapper body changes — Task 7
- §4.5 Two FFIs — Task 1
- §4.6 Declaration loop — Task 5 (uses 4.6a path)
- §5 Grad doc comment — Task 7 step 3 + Task 8
- §6.1 E2E test — Task 9
- §6.2 Semantic tests — Task 2
- §6.3 Codegen unit tests — Task 7 step 4 (three IR tests)
- §6.4 Regression guard — Task 10
- §7 Risks — acknowledged; 4.6a chosen, dual-registration treated as two declarations per risk §7

**Placeholder scan:** The word "adapt" appears 4 times — each pointing at a real codebase-dependent adjustment (existing Diagnostic builder API, AST field names, test helper APIs, nsl_model_create argtype). These are cues to read real code, not TODO markers.

**Type consistency:**
- `ExportWrapper.is_model_method: bool` (Task 7) ↔ set to `true` in Task 5 declaration path ↔ read in Task 7 wrapper emission. Consistent.
- `SelfResolution::WeightPtrsArray { weight_ptrs_var }` (Task 4) ↔ `FuncState::self_resolution` (Task 4) ↔ set in Task 6 during body compilation ↔ read in Task 4 SelfRef/FieldAccess branches. Consistent.
- `registry.export_method_impls: HashMap<(String, String), (FuncId, Signature)>` (Task 5) ↔ read in Task 6. Consistent.
- `nsl_model_get_weight_ptrs` / `nsl_model_get_num_weights` — named consistently across Tasks 1, 7, 9.

**Known adaptation points (expected, not placeholders):**
- Task 2 step 3: `Diagnostic::error(...).with_label(...).with_note(...).with_help(...)` — match real builder API
- Task 3 step 3: AST walker / type-map annotation — match crate's real pattern
- Task 5 step 1: `build_method_sig_excluding_self` sketch — use real method sig builder
- Task 6 step 1: `self.compile_fn_body` etc. — match real method names
- Task 8 step 2: snapshot tooling — `insta` or inline, depending on what's used
- Task 9 step 2: `nsl_model_create` ctypes argtype — verify against real FFI
