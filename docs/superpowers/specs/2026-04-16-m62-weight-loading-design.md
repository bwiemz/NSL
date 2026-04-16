# M62 `@export` Model-Method Weight Loading — Design

**Date:** 2026-04-16
**Status:** Approved for implementation
**Branch (target):** `feat/m62-weight-loading`
**Predecessors:**
- [M62 `@export` decorator](2026-04-15-m62-export-decorator-design.md) — merged via PR #45.
- [M62 `@export` C wrapper emission](2026-04-15-m62-c-wrappers-design.md) — `feat/m62-c-wrappers` (PR pending). Wrapper infrastructure this spec extends.
- [M62 grad-context bridge fix](2026-04-15-m62-grad-context-bridge-design.md) — handles autograd through `nsl_model_forward`. This spec leaves `@export` as an inference-only path (§5).

## 1. Goal

Enable `@export` on **model methods** so the exported C symbol loads weights from `NslModel*` at runtime and produces correct results. Today, compiling `model Net: @export fn predict(self, x) -> self.W @ x` either fails at codegen (top-level `SelfRef` has no `self` in scope) or silently produces wrong results (the existing semantic warning flags this). This spec closes that gap by threading `NslModel*.weight_ptrs` through the wrapper into the internal implementation.

## 2. Non-Goals (Explicitly Deferred)

- **`@export` on top-level functions taking a model-typed parameter** (e.g. `@export fn predict(net: Net, x)`). A follow-up issue will cover this. Semantic diagnostic points users at top-level form with "not yet supported" messaging.
- **Grad recording through `@export` dispatch.** `nsl_model_enable_grad(model, 1)` has no effect when calling `@export` wrappers. Training autograd stays on `nsl_model_forward` + `nsl_model_backward`. See §5.
- **Dynamic weight access.** `self.layers[i].W` requires symbolic index resolution; deferred.
- **Non-weight `self` fields.** Buffers, optimizer state, metadata — only tensor weights participate in the `weight_ptrs` convention today.
- **Tuple inputs/returns.** Inherited scope from the C wrapper PR.

## 3. Scope Invariants (do not break)

### 3.1 Non-`@export` paths untouched

`nsl_model_forward` / `nsl_model_set_forward` dispatch, existing `forward_fn` compilation, and non-decorated model methods produce byte-identical Cranelift IR. No regressions on WRGA, CSHA, CPDT, FASE, or autograd tape paths.

### 3.2 Weight ordering invariant

The compile-time index assigned to `self.W` at codegen MUST equal the runtime index `nsl_model_create` populates via safetensors loading. Both follow **declaration order** of `Tensor<...>` fields within the `model` block. Any change to either side must keep them in lockstep; this is load-bearing.

### 3.3 Wrapper C-ABI unchanged from caller's perspective

The exported symbol `int predict(NslModel*, const NslTensorDesc* x, NslTensorDesc* __ret)` matches the C header exactly — same as the first C-wrapper PR. Weight-pointer plumbing is invisible to C/Python callers.

### 3.4 Weight-less method parity

An `@export` method that takes `self` but does not access `self.<field>` still receives `(weight_ptrs, num_weights)` as leading args in its internal-impl signature. Every `@export` model method has the **same internal ABI shape** so the wrapper emits one dispatch pattern. Unused args are register overhead only; Cranelift's DCE elides downstream use. Without this parity, the wrapper would need per-method signature metadata — exactly the cross-pass coupling B2 was rejected for.

## 4. Design

### 4.1 Semantic rules (new)

File: `crates/nsl-semantic/src/export.rs` (extend existing `validate_exports`).

Three-way rule set for `@export` decorators:

1. **Model method with `self` as first parameter** → accepted. No warning. The existing Task 7 `check_no_model_weight_access` warning's trigger narrows: only fires on **top-level** `@export` functions with model-typed params. (New test: `@export` model method with `self.W @ x` produces NO warning.)

2. **Model method WITHOUT `self` as first parameter** → **error**, not warning:
   ```
   error: @export model method '<name>' requires `self` as first parameter
     note: @export on methods without `self` is not yet supported.
     help: move this to a top-level `@export fn` (not yet supported either;
           see follow-up issue #N) or add `self` as the first parameter.
   ```
   Rationale: methods without `self` can't access model weights; making them silently behave like top-level functions is confusing ergonomics. Rejecting with a pointer to the top-level form communicates scope clearly.

3. **Top-level function** → existing behavior unchanged. The Task 7 weight-access warning still fires for any `self`-like or model-typed reference.

Weight index resolution (also at semantic time, stored on AST / type map for codegen consumption): for each `self.<field>` inside an `@export` method body, compute the declaration-order index of `<field>` within the enclosing model's weight list (tensor fields only, in source order). Error if `<field>` is not a `Tensor<...>` field.

### 4.2 Internal impl signature — B1 convention

For `@export fn predict(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>`, the internal impl declared in `declaration.rs` has Cranelift signature:

```
__nsl_export_impl_predict(
    weight_ptrs: i64,    // *const *mut NslTensor (pointer into model.weight_ptrs)
    num_weights: i64,    // len of the weight_ptrs array
    x:           i64,    // *mut NslTensor for input x
) -> i64                 // *mut NslTensor for return
```

Weight-less methods get the same leading `(i64, i64)` pair. This matches the existing `forward_fn` dispatch convention used by `nsl_model_set_forward` / `nsl_model_forward`.

### 4.3 `self.W` codegen lowering

Inside an `@export` method body, `self.<field>` lowers to:
1. Load `weight_ptrs` from the method's second-from-input parameter.
2. Compute offset: `field_index * 8` (each entry is an `i64` NslTensor pointer).
3. Load the tensor pointer: `load.i64 (weight_ptrs + offset)`.

A new codegen mode flag (e.g. `self_resolution: SelfResolution::WeightPtrsArray { weight_ptrs_var, ... }` on the function compilation state) gates this path. Regular model-method compilation (for `forward_fn` and non-`@export` methods) uses the existing struct-layout path — unchanged. Selected at function-entry time based on whether `@export` is present.

### 4.4 Wrapper body changes

File: `crates/nsl-codegen/src/c_wrapper.rs::emit_c_abi_wrapper`.

`ExportWrapper` gains one field:

```rust
pub struct ExportWrapper {
    // ... existing fields ...
    /// True if this is an @export on a model method. Drives wrapper-body
    /// emission to thread model.weight_ptrs into the impl call.
    pub is_model_method: bool,
}
```

Set at declaration time (`declaration.rs` is_export branch) based on whether the function being processed is nested inside a `model` block AST node.

When `is_model_method = true`, the wrapper emits two extra IR steps between the null-check and the internal-impl call:

```
// After null-check on model_ptr, before converting tensor inputs:
let weight_ptrs = call nsl_model_get_weight_ptrs(model_ptr) -> i64    // *const i64
let num_weights = call nsl_model_get_num_weights(model_ptr) -> i64    // i64

// Prepended to internal_args before the tensor inputs:
internal_args = [weight_ptrs, num_weights, ...tensor_inputs]
```

When `is_model_method = false`, wrapper emission is identical to today — no weight plumbing, backwards-compatible for future top-level `@export`.

### 4.5 Two new runtime FFIs

File: `crates/nsl-runtime/src/c_api.rs`.

Two single-field accessor FFIs (chosen over a single tuple-returning FFI because Cranelift can't return tuples without out-parameters, and two sequential calls are cleaner than allocating a stack slot for an out-param):

```rust
/// Returns a pointer to the model's weight_ptrs array (valid for the model's
/// lifetime). The array contains `nsl_model_get_num_weights(model)` entries,
/// each an i64 that is a *mut NslTensor.
#[no_mangle]
pub extern "C" fn nsl_model_get_weight_ptrs(model_ptr: i64) -> i64 {
    if model_ptr == 0 {
        set_error("nsl_model_get_weight_ptrs: null model\0".into());
        return 0;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    model.weight_ptrs.as_ptr() as i64
}

/// Returns the number of weights in the model.
#[no_mangle]
pub extern "C" fn nsl_model_get_num_weights(model_ptr: i64) -> i64 {
    if model_ptr == 0 {
        set_error("nsl_model_get_num_weights: null model\0".into());
        return 0;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    model.weight_ptrs.len() as i64
}
```

Both are constant-time (single field load). Using FFI accessors rather than hard-coding `NslModel`'s field offsets into Cranelift keeps the struct layout free to evolve.

### 4.6 Declaration loop — model-method recognition

File: `crates/nsl-codegen/src/compiler/declaration.rs`.

The current `is_export` branch iterates top-level statements. For model-method `@export`, iteration also needs to descend into `model` block bodies to catch decorated methods. Two options:

- **4.6a:** Extend the existing loop to recurse into `ModelDef` body statements, treating each method with an `@export` decorator as a declaration-time `@export` target.
- **4.6b:** Add a new pass (`declare_model_export_methods`) that runs after `declare_user_functions_with_linkage` and specifically scans model bodies.

Pick **4.6a** (smaller diff, fewer passes, one source of truth for `@export` declaration logic). Risk: model methods today are probably processed by a different pass for struct/method layout — extending the `@export` branch to handle them may require coordinating with that pass's function-id registration. If the coordination gets tangled during implementation, fall back to 4.6b.

## 5. Grad scope — doc comment (D1)

In `crates/nsl-codegen/src/c_wrapper.rs`, the wrapper-emission function gets a comment block:

```rust
// GRAD SCOPE: @export dispatch is INFERENCE-ONLY. The wrapper does NOT:
//   - call nsl_tape_start over weight_ptrs
//   - save outputs to model.last_forward_outputs
//   - honor model.grad_enabled
// Calling nsl_model_enable_grad(model, 1) before an @export wrapper call
// has no effect on the @export dispatch path. For training autograd, use
// nsl_model_forward + nsl_model_backward (see the grad-context bridge fix).
```

In `crates/nsl-codegen/src/c_header.rs::emit()`, prepend a comment block to each `@export` prototype in the generated header:

```c
/* @export dispatch is inference-only. Gradient recording does not flow
 * through this call path — use nsl_model_forward for training autograd. */
int predict(NslModel* model, const NslTensorDesc* x, NslTensorDesc* __ret);
```

Without this comment, a user who reads both `nsl_model_enable_grad` and `@export` docs can't tell that the gradient plumbing is scoped to the forward-dispatch path only — they'll hit silent wrong gradients or null pointers in follow-up `nsl_model_backward` calls.

## 6. Testing

### 6.1 E2E Python ctypes test (T1)

New NSL fixture: `python/tests/fixtures/m62_predict_with_weights.nsl`:

```
model Net:
    W: Tensor<[4, 4], f32>

    @export
    fn predict(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return self.W @ x
```

Build process in test setup:
1. Write known W weights (e.g. identity matrix plus offset) to a safetensors file via `save_safetensors` helper (or directly via `safetensors.torch.save_file` in the test).
2. Compile the fixture with `nsl build --shared-lib`.

Python test (`python/tests/test_m62_weight_loading.py`):

```python
def test_predict_loads_weights_and_computes_W_at_x(shared_lib, weights_file):
    lib = ctypes.CDLL(str(shared_lib))
    # ... argtype setup for nsl_model_create / nsl_model_destroy / predict ...

    model_ptr = lib.nsl_model_create(weights_file_path_c_str)
    assert model_ptr != 0

    x = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    x_desc = _make_f32_desc(x, shape=[4])
    ret = NslTensorDesc()

    rc = lib.predict(model_ptr, ctypes.byref(x_desc), ctypes.byref(ret))
    assert rc == 0

    result = list(ctypes.cast(ret.data, ctypes.POINTER(ctypes.c_float * 4)).contents)
    expected = W_numpy @ [1.0, 2.0, 3.0, 4.0]
    assert all(abs(r - e) < 1e-5 for r, e in zip(result, expected))

    lib.nsl_model_destroy(model_ptr)
```

### 6.2 Semantic tests

File: `crates/nsl-semantic/src/export.rs::tests`.

1. **`export_model_method_without_self_produces_error`** — `model Net: @export fn helper(x): return x` → error diagnostic with message containing "requires `self`" and "not yet supported".
2. **`export_model_method_with_self_accesses_weights_no_warning`** — `model Net: W: Tensor<[4,4]>; @export fn predict(self, x): return self.W @ x` → zero diagnostics (no weight warning).
3. **Existing `export_function_referencing_model_field_produces_warning`** — continues to pass (top-level `@export` with model field access still warns).

### 6.3 Codegen unit tests

File: `crates/nsl-codegen/src/c_wrapper.rs::tests`.

1. **`wrapper_for_model_method_threads_weight_ptrs_and_count`** — compile a minimal fixture with `@export` on a model method, inspect the wrapper's Cranelift IR (via `Function::display()` or similar), assert:
   - The wrapper contains a call to `nsl_model_get_weight_ptrs`.
   - The wrapper contains a call to `nsl_model_get_num_weights`.
   - The impl call's first two args are the results of those two calls.
   This is the **cross-pass invariant guard** — catches regressions where someone "simplifies" the wrapper and drops the count.
2. **`wrapper_for_weightless_model_method_still_passes_weight_args`** — `@export fn identity(self, x): return x` — internal impl signature still has `(i64, i64)` as leading args; wrapper still emits the two accessor calls. Proves the parity invariant.
3. **`weight_index_resolution_offset_matches_declaration_order`** — a model with fields `W, b, gamma`; verify the IR loads `weight_ptrs + 0*8` for `self.W`, `+ 1*8` for `self.b`, `+ 2*8` for `self.gamma`.

### 6.4 Regression guard

Full workspace `cargo test` + `py -m pytest python/tests/` green, no pre-existing `@export` test regresses (Task 6+7+8 coverage from the C-wrapper PR).

## 7. Risks & Open Questions

- **Risk: model-method declaration coordination.** If existing model-method compilation registers function IDs in a way that conflicts with the new `@export`-mode declaration, 4.6a gets tangled. Fallback is 4.6b (separate pass). Document which path was taken during implementation.
- **Risk: `SelfRef` inside methods that previously used struct layout.** The new codegen mode flag must be selected at function-entry time. If a method is reachable via BOTH the normal `forward_fn` path AND the `@export` path, the compiler needs to emit two distinct codegen instances — or `@export` on a method conflicts with that method being a model's registered `forward`. Choose: treat them as two declarations (one via declaration.rs normal path, one via `@export` path with different name mangling `__nsl_export_impl_<name>` vs normal `<name>`). This matches existing declaration.rs two-function pattern.
- **Risk: f32 vs f64 tolerance.** The E2E test compares matmul results — f32 computation accumulates small errors. Use `1e-5` tolerance per element, or `1e-4` if matmul on our path has higher drift than numpy's.
- **Open: does `nsl_model_create` already compute weight indices in the same declaration order the compiler assigns?** Verify during implementation. If not, either fix safetensors loading OR ship a weight-name → index map as part of the compiled binary (bigger change; escalate if found).

## 8. Success Criteria

1. `cargo build --workspace` clean; no regressions.
2. `@export fn predict(self, x): self.W @ x` compiles, `nsl_model_create("weights.safetensors")` + `predict(model, x)` returns `W @ x` within f32 tolerance.
3. Semantic rejects `@export` model methods without `self` with a clear diagnostic message.
4. Existing weight-access warning narrows to top-level `@export` only (zero warnings on correct model-method `@export`).
5. Wrapper IR test asserts weight_ptrs + num_weights are threaded into the impl call (cross-pass invariant guard).
6. All pre-existing `@export` tests pass (C-wrapper PR Tasks 6–8 coverage maintained).
7. Generated C header contains the "inference-only" comment block before each `@export` prototype.

## 9. Files Touched

**Create:**
- `python/tests/fixtures/m62_predict_with_weights.nsl` — E2E fixture.
- `python/tests/test_m62_weight_loading.py` — E2E Python test.

**Modify:**
- `crates/nsl-semantic/src/export.rs` — three-way `@export` rules, weight-less rejection, narrow Task 7 warning.
- `crates/nsl-codegen/src/compiler/declaration.rs` — extend `is_export` branch to handle model methods (4.6a).
- `crates/nsl-codegen/src/c_wrapper.rs` — `ExportWrapper.is_model_method`, wrapper IR for weight-ptr threading, grad-scope doc comment, new IR tests.
- `crates/nsl-codegen/src/c_header.rs::emit()` — prepend "inference-only" comment to each `@export` prototype.
- `crates/nsl-codegen/src/expr/*.rs` (likely `access.rs` + method body codegen) — `self.W` lowering via `weight_ptrs` array when `@export` mode is active.
- `crates/nsl-runtime/src/c_api.rs` — `nsl_model_get_weight_ptrs` + `nsl_model_get_num_weights` FFIs.

---

**After this PR lands:** `@export` is production-usable for inference. Users write NSL model methods, decorate with `@export`, `nsl build --shared-lib` produces a `.so` / `.dll`, Python loads weights via `nsl_model_create`, and calls the exported symbol to get correct results. The semantic warning from Task 7 stops firing on valid code paths; it remains as a signal for the still-unsupported top-level `@export fn predict(net: Net, x)` shape.
