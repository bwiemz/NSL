# M62 `@export` Per-Function C Wrapper Emission — Design

**Date:** 2026-04-15
**Status:** Approved for implementation
**Branch (target):** `feat/m62-c-wrappers`
**Predecessors:**
- [M62 `@export` decorator](2026-04-15-m62-export-decorator-design.md) — merged via PR #45. Ships linkage override + C header emission; leaves C-wrapper emission as a documented gap.
- [M62 grad-context bridge fix](2026-04-15-m62-grad-context-bridge-design.md) — pending PR. Fixes three correctness bugs in `nsl_model_backward`; Python E2E tests currently skip because the fixture `.so` isn't callable from Python — exactly the gap this spec closes.

## 1. Goal

Make the symbols produced by `@export`-decorated NSL functions **actually callable from C / Python** using the signature the generated header promises. Today the symbol is reachable (`ctypes.CDLL(lib).forward` resolves), but the emitted function takes NSL-internal ABI (`i64` per tensor), while the header declares `int forward(NslModel*, const NslTensorDesc*, NslTensorDesc*)`. Calling the symbol with the header's signature crashes.

This spec closes the gap by emitting a **C-ABI wrapper per `@export` function** that converts `NslTensorDesc*` inputs into internal `NslTensor*`, calls the internal implementation, writes the result into the caller's `__ret` descriptor, and returns `0`/`-1`.

## 2. Non-Goals (Explicitly Deferred)

- **Weight-loading from `NslModel*` into function bodies.** The wrapper validates that `model` is non-null but does not thread weights through to the internal function. A semantic warning (§5) surfaces the case where this matters.
- **CUDA tensor support in the wrapper.** `desc_to_nsl_tensor` today assumes CPU. CUDA `@export` is a separate design.
- **Error-code taxonomy.** Success = `0`, failure = `-1` (with a thread-local error string via existing `nsl_get_last_error`). A richer taxonomy can come later.
- **Cross-module `@export`** (a function in module A being `@export`'d through module B's header). Out of scope.
- **Runtime function registry** (option B from brainstorming). Viable future work if cross-module needs emerge; YAGNI today.

## 3. Scope Invariants (do not break)

### 3.1 Fallback path is byte-identical

Non-`@export` functions continue to use `Linkage::Local` + mangled names with the internal ABI — unchanged from the current declaration loop. All existing tests that don't use `@export` produce identical Cranelift IR.

### 3.2 Exported symbol name is preserved

The exported symbol name in the `.so` is whatever the user wrote: `@export` → raw function name; `@export(name="predict")` → `"predict"`. Unchanged from today's behavior. Internal renaming of the implementation function is invisible to consumers.

### 3.3 Header output unchanged

`c_header::emit()` produces the same text as today. The header already declares the C-ABI signature; this PR makes reality match the header, not the other way around.

### 3.4 `param_paths` / `grads_list` order invariants

Unchanged. The wrapper doesn't touch any training-pipeline invariants; it only bridges the inference-call ABI.

## 4. Design

### 4.1 Two-function emission pattern

For each `@export` function, `declare_user_functions_with_linkage` declares TWO Cranelift functions:

| Role | Symbol name | Linkage | Signature |
|---|---|---|---|
| Internal implementation | `__nsl_export_impl_<raw_name>` | `Local` | NSL-internal ABI (`i64` per tensor, resolved scalars) |
| C-ABI wrapper | `<raw_name>` or `@export(name="...")` override | `Export` | `int (NslModel*, const NslTensorDesc*..., NslTensorDesc* __ret)` |

Today's `Compiler::registry.functions[raw_name]` maps `raw_name → (FuncId, Signature)` for use by `call_by_name`. The mapping now points at the internal implementation (since that's what NSL-internal callers need). The wrapper gets tracked separately in `features.export_wrappers`.

### 4.2 `ExportWrapper` struct

New type in `crates/nsl-codegen/src/c_wrapper.rs`:

```rust
pub struct ExportWrapper {
    /// FuncId of the internal implementation (Linkage::Local, mangled).
    pub impl_func_id: cranelift_module::FuncId,
    /// Signature of the internal implementation.
    pub impl_sig: cranelift_codegen::ir::Signature,
    /// FuncId of the C-ABI wrapper (Linkage::Export, exported name).
    pub wrapper_func_id: cranelift_module::FuncId,
    /// Raw NSL function name (for diagnostics).
    pub raw_name: String,
    /// The ExportInfo used for header emission. Drives wrapper signature
    /// construction so wrapper ABI matches what the header declares.
    pub export_info: crate::c_header::ExportInfo,
}
```

Stored on `FeatureConfigs.export_wrappers: Vec<ExportWrapper>`.

### 4.3 Wrapper signature construction

New helper in `c_wrapper.rs`:

```rust
pub fn build_c_abi_wrapper_signature(
    export_info: &ExportInfo,
    call_conv: CallConv,
    module: &dyn Module,
) -> Signature
```

For each `ExportParamInfo`:
- `ExportTypeInfo::Tensor { .. }` → `i64` (pointer to `NslTensorDesc`).
- `ExportTypeInfo::Scalar(dtype)` → the corresponding Cranelift type (`I32`, `I64`, `F32`, `F64`, etc.). Matches `c_type_for_scalar` in `c_header.rs`.
- `ExportTypeInfo::Tuple(_)` as input → `(i64, i32)` pair (array ptr + count). Rare; single-tensor inputs are the common case.

Prepended: `NslModel* model` as the first param (`i64`).

Return type:
- Tensor return: `i64` pointer to caller-allocated `NslTensorDesc* __ret` (wrapper writes into it; itself returns `i32` success code).
- Scalar return: `i64` pointer to caller-allocated `scalar_t* __ret`.
- Tuple return: two pointers — `NslTensorDesc* __rets, int32_t* __num_rets`.

Wrapper return type: `I32` (`0` = success, `-1` = failure).

### 4.4 Wrapper body emission

New function in `c_wrapper.rs`:

```rust
pub fn emit_c_abi_wrapper(
    compiler: &mut Compiler,
    wrapper: &ExportWrapper,
) -> Result<(), CodegenError>
```

Called once per `ExportWrapper` after all internal function bodies have been emitted. Pseudocode:

```
fn emit_c_abi_wrapper(...):
    ctx = module.make_context()
    ctx.func.signature = wrapper.wrapper_sig
    builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx)

    entry = builder.create_block()
    builder.append_block_params_for_function_params(entry)
    builder.switch_to_block(entry)
    builder.seal_block(entry)

    params = builder.block_params(entry)
    model_ptr = params[0]
    // inputs at [1..1+N], return arg at params[1+N]

    // Null-check model_ptr — set error + return -1
    null_check_block = builder.create_block()
    ok_block         = builder.create_block()
    err_return_block = builder.create_block()
    zero = builder.ins().iconst(I64, 0)
    is_null = builder.ins().icmp(IntCC::Equal, model_ptr, zero)
    builder.ins().brif(is_null, err_return_block, &[], ok_block, &[])

    builder.switch_to_block(err_return_block)
    builder.seal_block(err_return_block)
    // Call set_error-like helper (via Linkage::Import declaration)
    call_set_error(builder, "wrapper: null model pointer")
    neg_one = builder.ins().iconst(I32, -1)
    builder.ins().return_(&[neg_one])

    builder.switch_to_block(ok_block)
    builder.seal_block(ok_block)

    // Convert each tensor input desc → internal NslTensor* via desc_to_nsl_tensor_pub
    let internal_tensors = []
    for each tensor param i:
        desc_ptr = params[1 + i]
        t = call_desc_to_nsl_tensor_pub(builder, desc_ptr)
        internal_tensors.push(t)

    // Scalar inputs pass through unchanged (Cranelift args already have
    // the right type per build_c_abi_wrapper_signature).

    // Call the internal implementation.
    impl_fn_ref = module.declare_func_in_func(wrapper.impl_func_id, builder.func)
    impl_args = concatenate(internal_tensors, scalar_args)
    let result_ptr = builder.ins().call(impl_fn_ref, &impl_args)
    // For tensor return: result_ptr is an i64 → NslTensor*

    // Write result into caller's __ret descriptor.
    ret_desc_ptr = params[1 + N]   // last param
    call_nsl_tensor_to_desc(builder, result_ptr, ret_desc_ptr)

    // Free internal wrapper tensors (wrappers borrow Python-owned data;
    // freeing wrapper struct is safe and required for no leaks).
    for t in internal_tensors:
        call_nsl_tensor_free(builder, t)

    zero_ret = builder.ins().iconst(I32, 0)
    builder.ins().return_(&[zero_ret])

    builder.finalize()
    module.define_function(wrapper.wrapper_func_id, ctx)
```

**Runtime helpers the wrapper calls** — all declared via `Linkage::Import` once per compile unit:

| NSL runtime symbol | Role | Signature |
|---|---|---|
| `desc_to_nsl_tensor_pub` | Convert `NslTensorDesc*` → `NslTensor*` | `(i64) -> i64` |
| `nsl_tensor_to_desc` | Write `NslTensor*` → `NslTensorDesc*` | `(i64, i64) -> ()` |
| `nsl_tensor_free` | Free wrapper struct | `(i64) -> ()` |
| `nsl_set_error_cstr` | Set thread-local error (new — see §4.6) | `(i64) -> ()` |

### 4.5 Declaration loop rewrite in `declaration.rs`

Today:

```rust
let effective_linkage = if is_export { Linkage::Export } else { linkage };
let symbol_name = if is_export { override_name.unwrap_or(raw_name) } else { cranelift_name };
let func_id = self.module.declare_function(&symbol_name, effective_linkage, &sig)?;
self.registry.functions.insert(raw_name.clone(), (func_id, sig));
```

New shape (the `is_export` branch):

```rust
if is_export {
    // 1. Declare internal implementation
    let impl_name = format!("__nsl_export_impl_{}", raw_name);
    let impl_func_id = self.module.declare_function(&impl_name, Linkage::Local, &sig)?;
    self.registry.functions.insert(raw_name.clone(), (impl_func_id, sig.clone()));

    // 2. Build + declare C-ABI wrapper
    let info = ExportInfo::from_fn_def(fn_def, &raw_name, &wrapper_name, self.interner);
    let wrapper_sig = build_c_abi_wrapper_signature(&info, self.call_conv, &self.module);
    let wrapper_name = override_name.unwrap_or_else(|| raw_name.clone());
    let wrapper_func_id = self.module.declare_function(&wrapper_name, Linkage::Export, &wrapper_sig)?;

    // 3. Track for later body emission + header emission
    self.features.export_wrappers.push(ExportWrapper {
        impl_func_id,
        impl_sig: sig.clone(),
        wrapper_func_id,
        raw_name: raw_name.clone(),
        export_info: info.clone(),
    });
    self.features.export_functions.push(info);
} else {
    // Unchanged: mangled name, Local linkage
    let func_id = self.module.declare_function(&cranelift_name, linkage, &sig)?;
    self.registry.functions.insert(raw_name.clone(), (func_id, sig));
}
```

The existing decorator-handling loop (for `@no_grad`, `@grammar`, etc.) runs unchanged — it references `raw_name` which still maps to a valid FuncId (the internal implementation's).

### 4.6 New runtime FFI: `nsl_set_error_cstr`

Error-string setting today uses `set_error(String)` internally, but Cranelift-emitted code can't call a Rust function taking `String`. Add a tiny new `pub extern "C"` wrapper:

```rust
// In crates/nsl-runtime/src/c_api.rs
#[no_mangle]
pub extern "C" fn nsl_set_error_cstr(msg_ptr: i64) {
    if msg_ptr == 0 { return; }
    let msg_cstr = unsafe { CStr::from_ptr(msg_ptr as *const c_char) };
    let msg = msg_cstr.to_string_lossy().into_owned();
    set_error(msg);
}
```

The Cranelift wrapper passes a null-terminated string pointer from the compiled module's `.rodata` (error messages are emitted as string constants at wrapper-emission time).

### 4.7 Wrapper body emission timing

Existing pass order:
1. `declare_runtime_functions`
2. `declare_user_functions_with_linkage` — NOW also populates `export_wrappers`
3. `compile_kernels`
4. `compile_function_bodies` — internal function bodies emitted here
5. `finalize()`

New step 4b: `emit_export_wrappers` — runs AFTER 4 (internal bodies exist; their FuncIds can be resolved via `declare_func_in_func`).

```rust
impl Compiler<'_> {
    pub fn emit_export_wrappers(&mut self) -> Result<(), CodegenError> {
        for wrapper in &self.features.export_wrappers.clone() {  // clone to avoid borrow
            crate::c_wrapper::emit_c_abi_wrapper(self, wrapper)?;
        }
        Ok(())
    }
}
```

Called from the same place `compile_function_bodies` is called — likely `lib.rs`'s top-level compile function or the CLI handler.

## 5. Semantic Warning: Model-Weight References

File: `crates/nsl-semantic/src/export.rs` (extend existing).

When validating an `@export` function body, walk the body expressions and flag any access to a model-typed variable's fields. Precise check depends on how the AST represents model types — likely:

```rust
fn check_no_model_weight_access(
    fn_def: &FnDef,
    type_map: &TypeMap,
    diagnostics: &mut Vec<Diagnostic>,
) {
    fn visit_expr(expr: &Expr, type_map: &TypeMap, fn_name: &str, diagnostics: &mut Vec<Diagnostic>) {
        match &expr.kind {
            ExprKind::FieldAccess { receiver, field } => {
                if let Some(ty) = type_map.get(&receiver.id) {
                    if is_model_type(ty) {
                        diagnostics.push(Diagnostic::warning(format!(
                            "@export function '{fn_name}' references model field via '.{field}' \
                             — the exported symbol uses compile-time-baked weights, not runtime-loaded \
                             weights from NslModel*. Wait for weight-loading integration or restructure \
                             the function to be pure (inputs-only)."
                        )).with_label(expr.span, "model reference"));
                    }
                }
            }
            _ => {}
        }
        for child in expr.children() {
            visit_expr(child, type_map, fn_name, diagnostics);
        }
    }

    let fn_name = resolve(fn_def.name);
    for stmt in &fn_def.body {
        walk_stmt(stmt, |e| visit_expr(e, type_map, &fn_name, diagnostics));
    }
}
```

**This is a warning, not an error.** Compilation succeeds. Users see the signal in their diagnostic output and decide whether to wait for weight-loading integration or rewrite the function to take weights as explicit inputs.

CAUTION: `type_map` may not be accessible at the point validation runs. If the semantic pass order doesn't have type info available yet, defer this warning to a later pass OR degrade to a syntactic check ("does the function parameter list include a `self` or a parameter whose type annotation is a named model?"). Syntactic is strictly less precise but compositional.

## 6. Testing

### 6.1 Wrapper emission unit test

File: `crates/nsl-codegen/src/c_wrapper.rs` `#[cfg(test)] mod tests`.

```rust
#[test]
fn wrapper_signature_for_tensor_inputs_produces_correct_shape() {
    // Build ExportInfo for: fn add(a: Tensor<[4], f32>, b: Tensor<[4], f32>) -> Tensor<[4], f32>
    let info = /* synthesize manually */;
    let sig = build_c_abi_wrapper_signature(&info, CallConv::SystemV, &mock_module);

    // Expected: [i64 model, i64 a_desc, i64 b_desc, i64 ret_desc] -> i32
    assert_eq!(sig.params.len(), 4);
    for p in &sig.params { assert_eq!(p.value_type, I64); }
    assert_eq!(sig.returns.len(), 1);
    assert_eq!(sig.returns[0].value_type, I32);
}

#[test]
fn wrapper_signature_for_scalar_input_uses_scalar_type() {
    // fn scale(x: Tensor<[4], f32>, factor: f32) -> Tensor<[4], f32>
    let info = /* synthesize */;
    let sig = build_c_abi_wrapper_signature(&info, CallConv::SystemV, &mock_module);
    assert_eq!(sig.params[2].value_type, F32);  // factor as F32, not I64
}

#[test]
fn wrapper_signature_for_tuple_return_has_rets_and_num_rets() {
    // fn split(x: Tensor<[4], f32>) -> (Tensor<[2], f32>, Tensor<[2], f32>)
    let info = /* synthesize */;
    let sig = build_c_abi_wrapper_signature(&info, CallConv::SystemV, &mock_module);
    // Expected: [i64 model, i64 x_desc, i64 rets_array, i64 num_rets_ptr] -> i32
    assert_eq!(sig.params.len(), 4);
}
```

### 6.2 Integration test — actually call `@export fn add`

Extend `python/tests/test_m62_export.py` with a test that calls `add` via ctypes and verifies the result:

```python
def test_add_actually_computes_sum(shared_lib):
    import ctypes

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

    lib = ctypes.CDLL(str(shared_lib))
    lib.add.argtypes = [
        ctypes.c_void_p,               # NslModel* (can be non-null dummy)
        ctypes.POINTER(NslTensorDesc), # a
        ctypes.POINTER(NslTensorDesc), # b
        ctypes.POINTER(NslTensorDesc), # __ret
    ]
    lib.add.restype = ctypes.c_int32

    # Build input descriptors
    a_data = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    b_data = (ctypes.c_float * 4)(10.0, 20.0, 30.0, 40.0)
    shape = (ctypes.c_int64 * 1)(4)

    a = NslTensorDesc(
        data=ctypes.cast(a_data, ctypes.c_void_p),
        shape=shape, strides=None, ndim=1, dtype=0, device_type=0, device_id=0,
    )
    b = NslTensorDesc(
        data=ctypes.cast(b_data, ctypes.c_void_p),
        shape=shape, strides=None, ndim=1, dtype=0, device_type=0, device_id=0,
    )
    ret = NslTensorDesc()

    # Use a dummy model pointer; current wrapper only validates non-null.
    dummy_model = ctypes.c_void_p(1)  # any non-null address

    rc = lib.add(dummy_model, ctypes.byref(a), ctypes.byref(b), ctypes.byref(ret))
    assert rc == 0, f"add returned {rc}"

    # Read the result
    assert ret.data is not None
    result = ctypes.cast(ret.data, ctypes.POINTER(ctypes.c_float * 4)).contents
    assert list(result) == [11.0, 22.0, 33.0, 44.0]


def test_null_model_returns_error(shared_lib):
    # Same setup as above but pass NULL for model — wrapper should return -1.
    ...
    rc = lib.add(None, ctypes.byref(a), ctypes.byref(b), ctypes.byref(ret))
    assert rc == -1
```

### 6.3 Semantic warning test

In `crates/nsl-semantic/src/export.rs::tests`:

```rust
#[test]
fn export_function_referencing_model_field_produces_warning() {
    let src = r#"
model Net:
    W: Tensor<[4, 4], f32>

@export
fn predict(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return self.W @ x
"#;
    let diags = parse_and_validate(src);
    let warnings: Vec<_> = diags.iter().filter(|d| d.is_warning()).collect();
    assert!(
        warnings.iter().any(|d| d.message.contains("weight") && d.message.contains("predict")),
        "expected weight-reference warning, got: {:?}", diags
    );
}

#[test]
fn export_pure_function_has_no_warning() {
    let src = r#"
@export
fn add(a: Tensor<[4], f32>, b: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return a + b
"#;
    let diags = parse_and_validate(src);
    assert!(diags.iter().all(|d| !d.is_warning()), "unexpected warning: {:?}", diags);
}
```

### 6.4 Regression guard

- Non-`@export` shared-lib compiles produce byte-identical IR to pre-PR. All current tests (FASE, CSHA, WRGA, CPDT) pass unchanged.
- Test `test_m62_export.py::test_shared_lib_has_add_symbol` still passes — the `add` symbol is now the wrapper, but the symbol name is unchanged.
- Test `test_m62_export.py::test_generated_header_exists_and_declares_add` still passes — header format unchanged.

## 7. Risks & Open Questions

- **Risk: `type_map` unavailable at validation time.** If the semantic pass order prevents the precise type-based check in §5, fall back to syntactic detection (function has a `self` parameter OR a parameter with a user-defined struct type). Less precise but still catches the common case. Document the fallback in code comments if used.
- **Risk: tuple returns are more complex than specified.** The wrapper needs to take a results array + count pointer, and iterate the internal function's returned tuple (however it's represented in Cranelift — likely a packed struct or multiple return values). If internal tuples don't have a stable representation, restrict first-PR scope to single-tensor returns and defer multi-return to a follow-up. The semantic validator would enforce "only single-tensor return for `@export`" in that case.
- **Risk: `desc_to_nsl_tensor_pub` allocates.** If the tensor conversion allocates (to build the `NslTensor` wrapper struct), the wrapper needs to pair every conversion with a `nsl_tensor_free`. The pseudocode in §4.4 does this, but verify against the actual helper's ownership model before implementation.
- **Open: what if the internal function's Cranelift return type doesn't match `i64`?** Today `build_fn_signature` returns `i64` for tensor returns, but a function returning a scalar returns the scalar type directly. The wrapper's result-handling code needs to branch on tensor-vs-scalar return. `ExportInfo.return_type` already carries this distinction (`Tensor` vs `Scalar(_)`) — use it.

## 8. Success Criteria

1. `cargo build --workspace` clean; no existing tests regress.
2. `test_m62_export.py::test_add_actually_computes_sum` passes: the `add` symbol called from ctypes with 2 `NslTensorDesc*` inputs writes `[11.0, 22.0, 33.0, 44.0]` into `__ret.data`.
3. `test_m62_export.py::test_null_model_returns_error` passes: wrapper returns `-1` on null model pointer.
4. Semantic warning test passes: `@export fn predict(self, x) -> self.W @ x` produces a warning diagnostic (not error) containing "weight" and the function name.
5. No pre-existing `@export` test regresses — `test_shared_lib_has_add_symbol` and `test_generated_header_exists_and_declares_add` continue to pass.
6. Wrapper unit tests pass: signature construction correct for tensor inputs, scalar inputs, tuple returns.

## 9. Files Touched

**Create:**
- `crates/nsl-codegen/src/c_wrapper.rs` — `ExportWrapper` struct, `build_c_abi_wrapper_signature`, `emit_c_abi_wrapper`, unit tests.

**Modify:**
- `crates/nsl-codegen/src/lib.rs` — `pub mod c_wrapper;`.
- `crates/nsl-codegen/src/compiler/mod.rs` — add `export_wrappers: Vec<ExportWrapper>` on `FeatureConfigs`.
- `crates/nsl-codegen/src/compiler/declaration.rs` — rewrite the `is_export` branch to declare both internal + wrapper; populate `export_wrappers`.
- `crates/nsl-codegen/src/compiler/functions.rs` (or equivalent) — call `emit_export_wrappers` after internal function bodies are emitted.
- `crates/nsl-runtime/src/c_api.rs` — add `nsl_set_error_cstr` FFI helper.
- `crates/nsl-semantic/src/export.rs` — add `check_no_model_weight_access` warning pass + 2 new test cases.
- `python/tests/test_m62_export.py` — add `test_add_actually_computes_sum` + `test_null_model_returns_error`.

---

**After this PR lands, the `@export` story is functionally complete:** `nsl build --shared-lib model.nsl` produces a `.so` whose exported functions are callable from C and Python with the signature the header declares. The grad-context bridge's Python integration tests (`python/tests/test_autograd.py`) also unblock — they can now successfully build + load the fixture `.so` and exercise the real autograd path end-to-end.
