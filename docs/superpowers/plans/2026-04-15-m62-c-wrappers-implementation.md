# M62 `@export` C Wrapper Emission — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit per-`@export` C-ABI wrapper functions so symbols produced by `nsl build --shared-lib` are actually callable from C/Python with the signature the generated header declares.

**Architecture:** For each `@export` NSL function, declare TWO Cranelift functions: an internal implementation (`__nsl_export_impl_<raw>` / `Linkage::Local` / NSL-internal ABI) and a C-ABI wrapper (exported name / `Linkage::Export` / `int (NslModel*, NslTensorDesc*..., NslTensorDesc* __ret)`). The wrapper null-checks the model, converts `NslTensorDesc*` inputs to internal `NslTensor*` via `desc_to_nsl_tensor_pub`, calls the internal impl, writes the result into `__ret` via `nsl_tensor_to_desc`, frees wrapper tensors, returns `0`/`-1`.

**Tech Stack:** Rust, Cranelift (`cranelift-codegen`, `cranelift-module`, `cranelift-frontend`), existing NSL runtime FFI (`crates/nsl-runtime/src/c_api.rs`), ctypes (Python integration tests).

**Reference spec:** [docs/superpowers/specs/2026-04-15-m62-c-wrappers-design.md](../specs/2026-04-15-m62-c-wrappers-design.md)

---

## File Structure

**Create:**
- `crates/nsl-codegen/src/c_wrapper.rs` — `ExportWrapper` struct, `build_c_abi_wrapper_signature`, `emit_c_abi_wrapper`, unit tests.

**Modify:**
- `crates/nsl-codegen/src/lib.rs` — `pub mod c_wrapper;`.
- `crates/nsl-codegen/src/compiler/mod.rs` — add `export_wrappers: Vec<ExportWrapper>` on `FeatureConfigs`; call `emit_export_wrappers` at end of `compile_function_bodies` (or wherever body compilation terminates before `finalize`).
- `crates/nsl-codegen/src/compiler/declaration.rs` — rewrite the `is_export` branch (lines ~55–90) to declare internal impl + wrapper pair.
- `crates/nsl-runtime/src/c_api.rs` — add `nsl_set_error_cstr` FFI.
- `crates/nsl-semantic/src/export.rs` — add `check_no_model_weight_access` warning pass + 2 tests.
- `python/tests/test_m62_export.py` — add `test_add_actually_computes_sum` + `test_null_model_returns_error`.

Scope invariant: non-`@export` declaration path is untouched. Diff on that branch must be zero.

---

## Task 1: Scaffold `c_wrapper.rs` + `ExportWrapper` + `FeatureConfigs` slot

**Files:**
- Create: `crates/nsl-codegen/src/c_wrapper.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` (add `pub mod c_wrapper;`)
- Modify: `crates/nsl-codegen/src/compiler/mod.rs` (add `export_wrappers` field + init in `FeatureConfigs::default`/`::new`)

- [ ] **Step 1: Create `c_wrapper.rs` with struct + module stub**

```rust
//! Per-`@export` C-ABI wrapper emission.
//!
//! See docs/superpowers/specs/2026-04-15-m62-c-wrappers-design.md.

use crate::c_header::ExportInfo;
use cranelift_codegen::ir::Signature;
use cranelift_module::FuncId;

#[derive(Clone, Debug)]
pub struct ExportWrapper {
    pub impl_func_id: FuncId,
    pub impl_sig: Signature,
    pub wrapper_func_id: FuncId,
    pub raw_name: String,
    pub export_info: ExportInfo,
}
```

- [ ] **Step 2: Wire module into `lib.rs`**

Open `crates/nsl-codegen/src/lib.rs` and add alongside the other `pub mod` lines:

```rust
pub mod c_wrapper;
```

- [ ] **Step 3: Add `export_wrappers` to `FeatureConfigs`**

Open `crates/nsl-codegen/src/compiler/mod.rs`. Near line 265 where `export_functions` is declared:

```rust
pub export_functions: Vec<crate::c_header::ExportInfo>,
pub export_wrappers: Vec<crate::c_wrapper::ExportWrapper>,
```

And in the matching default/init block near line 321:

```rust
export_functions: Vec::new(),
export_wrappers: Vec::new(),
```

- [ ] **Step 4: Verify build**

Run: `cargo build -p nsl-codegen`
Expected: clean build (no warnings about unused `ExportWrapper` — it's `pub`).

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/c_wrapper.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/src/compiler/mod.rs
git commit -m "feat(m62): scaffold c_wrapper module + ExportWrapper struct"
```

---

## Task 2: `build_c_abi_wrapper_signature` + unit tests

**Files:**
- Modify: `crates/nsl-codegen/src/c_wrapper.rs`

- [ ] **Step 1: Write failing unit tests**

Append to `c_wrapper.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_header::{ExportDevice, ExportDtype, ExportInfo, ExportParamInfo, ExportTypeInfo};
    use cranelift_codegen::ir::types::{F32, I32, I64};
    use cranelift_codegen::isa::CallConv;

    fn tensor_param(name: &str, dtype: ExportDtype) -> ExportParamInfo {
        ExportParamInfo {
            name: name.into(),
            ty: ExportTypeInfo::Tensor {
                shape: vec![4],
                dtype,
                device: ExportDevice::Cpu,
            },
        }
    }

    #[test]
    fn wrapper_signature_for_tensor_inputs_produces_correct_shape() {
        let info = ExportInfo {
            symbol_name: "add".into(),
            raw_name: "add".into(),
            params: vec![
                tensor_param("a", ExportDtype::F32),
                tensor_param("b", ExportDtype::F32),
            ],
            return_type: ExportTypeInfo::Tensor {
                shape: vec![4],
                dtype: ExportDtype::F32,
                device: ExportDevice::Cpu,
            },
        };
        let sig = build_c_abi_wrapper_signature(&info, CallConv::SystemV);
        // [i64 model, i64 a_desc, i64 b_desc, i64 ret_desc] -> i32
        assert_eq!(sig.params.len(), 4);
        for p in &sig.params {
            assert_eq!(p.value_type, I64);
        }
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.returns[0].value_type, I32);
    }

    #[test]
    fn wrapper_signature_for_scalar_input_uses_scalar_type() {
        let info = ExportInfo {
            symbol_name: "scale".into(),
            raw_name: "scale".into(),
            params: vec![
                tensor_param("x", ExportDtype::F32),
                ExportParamInfo {
                    name: "factor".into(),
                    ty: ExportTypeInfo::Scalar(ExportDtype::F32),
                },
            ],
            return_type: ExportTypeInfo::Tensor {
                shape: vec![4],
                dtype: ExportDtype::F32,
                device: ExportDevice::Cpu,
            },
        };
        let sig = build_c_abi_wrapper_signature(&info, CallConv::SystemV);
        // [i64 model, i64 x_desc, F32 factor, i64 ret_desc] -> i32
        assert_eq!(sig.params.len(), 4);
        assert_eq!(sig.params[0].value_type, I64);
        assert_eq!(sig.params[1].value_type, I64);
        assert_eq!(sig.params[2].value_type, F32);
        assert_eq!(sig.params[3].value_type, I64);
    }
}
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cargo test -p nsl-codegen c_wrapper::tests --lib`
Expected: compile error — `build_c_abi_wrapper_signature` not defined.

- [ ] **Step 3: Implement `build_c_abi_wrapper_signature`**

Add to `c_wrapper.rs` above the test module:

```rust
use crate::c_header::{ExportDtype, ExportParamInfo, ExportTypeInfo};
use cranelift_codegen::ir::{types, AbiParam, Signature};
use cranelift_codegen::isa::CallConv;

fn cranelift_type_for_scalar(dtype: ExportDtype) -> cranelift_codegen::ir::Type {
    match dtype {
        ExportDtype::I32 => types::I32,
        ExportDtype::I64 => types::I64,
        ExportDtype::F32 => types::F32,
        ExportDtype::F64 => types::F64,
        // Low-precision scalars pass as their widened container type.
        _ => types::I64,
    }
}

pub fn build_c_abi_wrapper_signature(
    export_info: &ExportInfo,
    call_conv: CallConv,
) -> Signature {
    let mut sig = Signature::new(call_conv);

    // NslModel* first
    sig.params.push(AbiParam::new(types::I64));

    // Inputs
    for param in &export_info.params {
        let ty = match &param.ty {
            ExportTypeInfo::Tensor { .. } => types::I64,
            ExportTypeInfo::Scalar(dt) => cranelift_type_for_scalar(*dt),
            ExportTypeInfo::Tuple(_) => types::I64, // packed input — rare
        };
        sig.params.push(AbiParam::new(ty));
    }

    // Return pointer(s)
    match &export_info.return_type {
        ExportTypeInfo::Tensor { .. } | ExportTypeInfo::Scalar(_) => {
            sig.params.push(AbiParam::new(types::I64));
        }
        ExportTypeInfo::Tuple(_) => {
            // results array + num_rets pointer
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(types::I64));
        }
    }

    sig.returns.push(AbiParam::new(types::I32));
    sig
}
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `cargo test -p nsl-codegen c_wrapper::tests --lib`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/c_wrapper.rs
git commit -m "feat(m62): build_c_abi_wrapper_signature with unit tests"
```

---

## Task 3: `nsl_set_error_cstr` runtime FFI

**Files:**
- Modify: `crates/nsl-runtime/src/c_api.rs`

- [ ] **Step 1: Add the FFI function**

Locate an existing `#[no_mangle] pub extern "C" fn nsl_get_last_error` in `crates/nsl-runtime/src/c_api.rs` (grep if needed). Near it, add:

```rust
/// Set the thread-local error string from a null-terminated C string pointer.
/// Used by Cranelift-emitted `@export` wrappers; they can't call the Rust-typed
/// `set_error(String)` directly.
#[no_mangle]
pub extern "C" fn nsl_set_error_cstr(msg_ptr: i64) {
    if msg_ptr == 0 {
        return;
    }
    let msg = unsafe {
        std::ffi::CStr::from_ptr(msg_ptr as *const std::os::raw::c_char)
            .to_string_lossy()
            .into_owned()
    };
    set_error(msg);
}
```

(Imports of `std::ffi::CStr` etc. may already be present; add at top of file if not.)

- [ ] **Step 2: Write smoke test**

Append to `c_api.rs` test module:

```rust
#[test]
fn nsl_set_error_cstr_sets_thread_local() {
    let msg = std::ffi::CString::new("hello from wrapper").unwrap();
    nsl_set_error_cstr(msg.as_ptr() as i64);
    // nsl_get_last_error returns a heap-allocated copy
    let err_ptr = nsl_get_last_error();
    assert!(err_ptr != 0);
    let got = unsafe {
        std::ffi::CStr::from_ptr(err_ptr as *const std::os::raw::c_char)
            .to_string_lossy()
            .into_owned()
    };
    assert_eq!(got, "hello from wrapper");
}

#[test]
fn nsl_set_error_cstr_null_is_noop() {
    nsl_set_error_cstr(0);
    // no panic, no crash; previous error may remain (that's fine)
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p nsl-runtime nsl_set_error_cstr --lib`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/c_api.rs
git commit -m "feat(m62): nsl_set_error_cstr FFI for Cranelift wrappers"
```

---

## Task 4: Rewrite `is_export` declaration branch

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/declaration.rs` (branch near lines 55–90)

Note: read the current `is_export` block first (`declaration.rs:55-90`) to preserve surrounding code — especially how `sig`, `raw_name`, `cranelift_name`, `fn_def`, `self.interner` are in scope. The block below may need name adjustments to match actual locals.

- [ ] **Step 1: Locate the current `is_export` branch**

Run: `grep -n "if is_export" crates/nsl-codegen/src/compiler/declaration.rs | head -5`
Expected: hit around line 82 (inside `declare_user_functions_with_linkage`).

- [ ] **Step 2: Replace the is_export branch with two-function pattern**

Open `crates/nsl-codegen/src/compiler/declaration.rs`. Find the block that looks like:

```rust
let effective_linkage = if is_export { Linkage::Export } else { linkage };
let symbol_name = if is_export { override_name.clone().unwrap_or_else(|| raw_name.clone()) } else { cranelift_name.clone() };
let func_id = self.module.declare_function(&symbol_name, effective_linkage, &sig)?;
self.registry.functions.insert(raw_name.clone(), (func_id, sig.clone()));
if is_export {
    let info = ExportInfo::from_fn_def(fn_def, &raw_name, &symbol_name, self.interner);
    self.features.export_functions.push(info);
}
```

Replace with:

```rust
if is_export {
    use crate::c_wrapper::{build_c_abi_wrapper_signature, ExportWrapper};

    // 1. Internal implementation: mangled name, Local linkage, NSL-internal ABI.
    let impl_symbol = format!("__nsl_export_impl_{}", raw_name);
    let impl_func_id = self
        .module
        .declare_function(&impl_symbol, Linkage::Local, &sig)?;
    self.registry
        .functions
        .insert(raw_name.clone(), (impl_func_id, sig.clone()));

    // 2. Wrapper: exported name (user override or raw), Export linkage, C-ABI sig.
    let wrapper_symbol = override_name.clone().unwrap_or_else(|| raw_name.clone());
    let info = crate::c_header::ExportInfo::from_fn_def(
        fn_def,
        &raw_name,
        &wrapper_symbol,
        self.interner,
    );
    let call_conv = self.module.target_config().default_call_conv;
    let wrapper_sig = build_c_abi_wrapper_signature(&info, call_conv);
    let wrapper_func_id = self
        .module
        .declare_function(&wrapper_symbol, Linkage::Export, &wrapper_sig)?;

    // 3. Track for header emission + later wrapper-body emission.
    self.features.export_wrappers.push(ExportWrapper {
        impl_func_id,
        impl_sig: sig.clone(),
        wrapper_func_id,
        raw_name: raw_name.clone(),
        export_info: info.clone(),
    });
    self.features.export_functions.push(info);
} else {
    let func_id = self.module.declare_function(&cranelift_name, linkage, &sig)?;
    self.registry.functions.insert(raw_name.clone(), (func_id, sig));
}
```

If the existing code doesn't split cleanly, treat this as a hint and adapt — the key invariants are (a) `registry.functions[raw_name]` maps to the internal impl's FuncId, (b) a new `ExportWrapper` is pushed, (c) non-export path is byte-identical to before.

- [ ] **Step 3: Run existing @export tests**

Run: `cargo test -p nsl-codegen declaration --lib`
Expected: `extract_export_decorator` tests still pass (they don't touch the declaration path).

Run: `cargo build -p nsl-codegen`
Expected: clean build.

- [ ] **Step 4: Verify non-export regression hasn't hit**

Run: `cargo test -p nsl-codegen --lib`
Expected: no regressions. (Wrapper-body emission is still missing → any test that actually links a wrapper-producing `.so` will fail at `module.finalize()` with "function X has no body". That's acceptable at this task boundary; Task 5+6 fix it.)

If a test panics at finalize because of missing wrapper body, mark it `#[ignore]` with a TODO referencing Task 6, then commit. Do NOT leave the tree red.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/compiler/declaration.rs
git commit -m "feat(m62): declare internal impl + wrapper pair for @export fns"
```

---

## Task 5: `emit_c_abi_wrapper` body emission

**Files:**
- Modify: `crates/nsl-codegen/src/c_wrapper.rs`

- [ ] **Step 1: Add imports + function skeleton**

At the top of `c_wrapper.rs`:

```rust
use crate::compiler::Compiler;
use crate::error::CodegenError;
use cranelift_codegen::ir::{condcodes::IntCC, InstBuilder};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::Module;
```

Add the function body below `build_c_abi_wrapper_signature`:

```rust
pub fn emit_c_abi_wrapper(
    compiler: &mut Compiler,
    wrapper: &ExportWrapper,
) -> Result<(), CodegenError> {
    let call_conv = compiler.module.target_config().default_call_conv;
    let wrapper_sig = build_c_abi_wrapper_signature(&wrapper.export_info, call_conv);

    let mut ctx = compiler.module.make_context();
    ctx.func.signature = wrapper_sig.clone();

    let mut fn_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);

    let params: Vec<_> = builder.block_params(entry).to_vec();
    let model_ptr = params[0];

    // Null check on model
    let err_block = builder.create_block();
    let ok_block = builder.create_block();
    let zero_i64 = builder.ins().iconst(cranelift_codegen::ir::types::I64, 0);
    let is_null = builder.ins().icmp(IntCC::Equal, model_ptr, zero_i64);
    builder.ins().brif(is_null, err_block, &[], ok_block, &[]);

    // Error path
    builder.switch_to_block(err_block);
    builder.seal_block(err_block);
    emit_set_error(&mut builder, &mut compiler.module, "null model pointer")?;
    let neg_one = builder.ins().iconst(cranelift_codegen::ir::types::I32, -1);
    builder.ins().return_(&[neg_one]);

    // Happy path
    builder.switch_to_block(ok_block);
    builder.seal_block(ok_block);

    // Convert tensor-input descriptors to internal NslTensor*; pass scalars through.
    let mut internal_args: Vec<cranelift_codegen::ir::Value> = Vec::new();
    let mut tensor_inputs_to_free: Vec<cranelift_codegen::ir::Value> = Vec::new();

    for (i, param) in wrapper.export_info.params.iter().enumerate() {
        let arg_val = params[1 + i];
        match &param.ty {
            ExportTypeInfo::Tensor { .. } => {
                let tensor = call_desc_to_nsl_tensor(&mut builder, &mut compiler.module, arg_val)?;
                internal_args.push(tensor);
                tensor_inputs_to_free.push(tensor);
            }
            ExportTypeInfo::Scalar(_) => {
                internal_args.push(arg_val);
            }
            ExportTypeInfo::Tuple(_) => {
                // Packed input tuple not supported in first PR — see spec §7.
                return Err(CodegenError::unsupported(format!(
                    "@export tuple input parameter for '{}' is not yet supported",
                    wrapper.raw_name
                )));
            }
        }
    }

    // Call the internal implementation
    let impl_ref = compiler
        .module
        .declare_func_in_func(wrapper.impl_func_id, builder.func);
    let call_inst = builder.ins().call(impl_ref, &internal_args);
    let impl_rets = builder.inst_results(call_inst).to_vec();

    // Write result into caller's __ret descriptor(s)
    match &wrapper.export_info.return_type {
        ExportTypeInfo::Tensor { .. } => {
            let ret_desc_ptr = params[1 + wrapper.export_info.params.len()];
            let result_tensor = impl_rets[0];
            call_nsl_tensor_to_desc(&mut builder, &mut compiler.module, result_tensor, ret_desc_ptr)?;
        }
        ExportTypeInfo::Scalar(_) => {
            // Store scalar return through the provided pointer
            let ret_ptr = params[1 + wrapper.export_info.params.len()];
            let scalar_val = impl_rets[0];
            builder
                .ins()
                .store(cranelift_codegen::ir::MemFlags::trusted(), scalar_val, ret_ptr, 0);
        }
        ExportTypeInfo::Tuple(_) => {
            return Err(CodegenError::unsupported(format!(
                "@export tuple return for '{}' is not yet supported",
                wrapper.raw_name
            )));
        }
    }

    // Free wrapper tensor structs
    for t in tensor_inputs_to_free {
        call_nsl_tensor_free(&mut builder, &mut compiler.module, t)?;
    }

    let zero_ret = builder.ins().iconst(cranelift_codegen::ir::types::I32, 0);
    builder.ins().return_(&[zero_ret]);
    builder.finalize();

    compiler
        .module
        .define_function(wrapper.wrapper_func_id, &mut ctx)
        .map_err(|e| CodegenError::internal(format!("define wrapper fn: {e:?}")))?;
    compiler.module.clear_context(&mut ctx);
    Ok(())
}
```

- [ ] **Step 2: Add the runtime-call helpers**

Below `emit_c_abi_wrapper`:

```rust
fn declare_runtime_fn(
    module: &mut dyn Module,
    name: &str,
    params: &[cranelift_codegen::ir::Type],
    returns: &[cranelift_codegen::ir::Type],
) -> Result<cranelift_module::FuncId, CodegenError> {
    let call_conv = module.target_config().default_call_conv;
    let mut sig = cranelift_codegen::ir::Signature::new(call_conv);
    for p in params {
        sig.params.push(cranelift_codegen::ir::AbiParam::new(*p));
    }
    for r in returns {
        sig.returns.push(cranelift_codegen::ir::AbiParam::new(*r));
    }
    module
        .declare_function(name, cranelift_module::Linkage::Import, &sig)
        .map_err(|e| CodegenError::internal(format!("declare {name}: {e:?}")))
}

fn call_desc_to_nsl_tensor(
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    desc_ptr: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    use cranelift_codegen::ir::types::I64;
    let fid = declare_runtime_fn(module, "desc_to_nsl_tensor_pub", &[I64], &[I64])?;
    let fref = module.declare_func_in_func(fid, builder.func);
    let call = builder.ins().call(fref, &[desc_ptr]);
    Ok(builder.inst_results(call)[0])
}

fn call_nsl_tensor_to_desc(
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    tensor: cranelift_codegen::ir::Value,
    desc_ptr: cranelift_codegen::ir::Value,
) -> Result<(), CodegenError> {
    use cranelift_codegen::ir::types::I64;
    let fid = declare_runtime_fn(module, "nsl_tensor_to_desc", &[I64, I64], &[])?;
    let fref = module.declare_func_in_func(fid, builder.func);
    builder.ins().call(fref, &[tensor, desc_ptr]);
    Ok(())
}

fn call_nsl_tensor_free(
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    tensor: cranelift_codegen::ir::Value,
) -> Result<(), CodegenError> {
    use cranelift_codegen::ir::types::I64;
    let fid = declare_runtime_fn(module, "nsl_tensor_free", &[I64], &[])?;
    let fref = module.declare_func_in_func(fid, builder.func);
    builder.ins().call(fref, &[tensor]);
    Ok(())
}

fn emit_set_error(
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    msg: &str,
) -> Result<(), CodegenError> {
    use cranelift_codegen::ir::types::I64;
    // Allocate a data symbol for the null-terminated string.
    let mut bytes = msg.as_bytes().to_vec();
    bytes.push(0);
    let data_id = {
        let sym = format!("__nsl_wrapper_errstr_{:x}", fxhash(msg));
        let id = module
            .declare_data(&sym, cranelift_module::Linkage::Local, false, false)
            .map_err(|e| CodegenError::internal(format!("declare errstr: {e:?}")))?;
        let mut desc = cranelift_module::DataDescription::new();
        desc.define(bytes.into_boxed_slice());
        // define may fail if already defined — ignore "duplicate" by probing first
        let _ = module.define_data(id, &desc);
        id
    };
    let gv = module.declare_data_in_func(data_id, builder.func);
    let str_ptr = builder.ins().symbol_value(I64, gv);

    let fid = declare_runtime_fn(module, "nsl_set_error_cstr", &[I64], &[])?;
    let fref = module.declare_func_in_func(fid, builder.func);
    builder.ins().call(fref, &[str_ptr]);
    Ok(())
}

fn fxhash(s: &str) -> u64 {
    // Tiny stable hash so the same message produces the same symbol across
    // compile units without pulling in an extra crate.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
```

- [ ] **Step 3: Confirm build**

Run: `cargo build -p nsl-codegen`
Expected: clean. If `desc_to_nsl_tensor_pub` / `nsl_tensor_to_desc` / `nsl_tensor_free` aren't public FFI symbols, grep `crates/nsl-runtime/src/c_api.rs` to confirm their exact names and update calls. (Spec §4.4 table asserts these names.)

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/c_wrapper.rs
git commit -m "feat(m62): emit_c_abi_wrapper body + runtime call helpers"
```

---

## Task 6: Wire `emit_export_wrappers` into compile pipeline

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/mod.rs`

- [ ] **Step 1: Add the driver method on `Compiler`**

In `crates/nsl-codegen/src/compiler/mod.rs`, near the other `impl Compiler` methods (look for `compile_function_bodies` or `finalize`):

```rust
impl<'a> Compiler<'a> {
    pub fn emit_export_wrappers(&mut self) -> Result<(), crate::error::CodegenError> {
        let wrappers = self.features.export_wrappers.clone();
        for wrapper in &wrappers {
            crate::c_wrapper::emit_c_abi_wrapper(self, wrapper)?;
        }
        Ok(())
    }
}
```

- [ ] **Step 2: Call it from the compile driver**

Find where `compile_function_bodies` (or equivalent) is invoked — likely in `compiler/mod.rs::finalize` or in `entry_points.rs`. After all function bodies are defined but BEFORE `module.finish()`/`finalize()`, add:

```rust
self.emit_export_wrappers()?;
```

Candidate call sites (grep for each): `compile_main`, `compile_library`, `compile_shared_lib`, and the body-compilation loops in `entry_points.rs:806` / `entry_points.rs:932`.

- [ ] **Step 3: Build the full workspace**

Run: `cargo build --workspace`
Expected: clean.

- [ ] **Step 4: Run existing @export tests**

Run: `cargo test -p nsl-codegen --lib`
Expected: all pre-existing tests pass.

Run any previously `#[ignore]`-marked tests from Task 4 step 4 with the ignore removed:

```bash
# Remove #[ignore] lines added in Task 4 if any
```

Run: `cargo test --workspace`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/compiler/mod.rs
git commit -m "feat(m62): call emit_export_wrappers after function bodies"
```

---

## Task 7: Semantic warning — model-weight access in `@export` body

**Files:**
- Modify: `crates/nsl-semantic/src/export.rs`

- [ ] **Step 1: Write failing tests**

In `crates/nsl-semantic/src/export.rs`, inside the existing `#[cfg(test)] mod tests`:

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
    let diags = parse_and_validate_for_test(src);
    let warnings: Vec<_> = diags.iter().filter(|d| d.is_warning()).collect();
    assert!(
        warnings
            .iter()
            .any(|d| d.message.contains("weight") && d.message.contains("predict")),
        "expected weight-reference warning, got: {:?}",
        diags
    );
}

#[test]
fn export_pure_function_has_no_weight_warning() {
    let src = r#"
@export
fn add(a: Tensor<[4], f32>, b: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return a + b
"#;
    let diags = parse_and_validate_for_test(src);
    let warnings: Vec<_> = diags
        .iter()
        .filter(|d| d.is_warning() && d.message.contains("weight"))
        .collect();
    assert!(warnings.is_empty(), "unexpected weight warning: {:?}", diags);
}
```

If a `parse_and_validate_for_test` helper doesn't already exist in this file, mirror whatever pattern the existing 7 semantic tests use (grep `crates/nsl-semantic/src/export.rs` for `#[test]` to find the template).

- [ ] **Step 2: Run tests to confirm failure**

Run: `cargo test -p nsl-semantic export::tests --lib`
Expected: 2 new tests fail (no warning emitted).

- [ ] **Step 3: Implement the warning pass**

In `export.rs`, add (syntactic fallback per spec §5 CAUTION — `type_map` may not be available during semantic pass):

```rust
/// Warn if an `@export` function references a model-typed parameter's fields.
/// Uses a syntactic heuristic: any `self` parameter OR field access on a
/// parameter whose annotation is a user-defined type name.
fn check_no_model_weight_access(
    fn_def: &crate::ast::FnDef,
    interner: &crate::interner::Interner,
    diagnostics: &mut Vec<crate::diagnostic::Diagnostic>,
) {
    use crate::diagnostic::Diagnostic;

    let fn_name = interner.resolve(fn_def.name).to_string();

    // Collect parameter names that are potentially model-typed:
    //   - "self" (always model-typed)
    //   - any param with a Named type annotation (struct/model reference)
    let mut suspect_params: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in &fn_def.params {
        let name = interner.resolve(p.name).to_string();
        if name == "self" {
            suspect_params.insert(name);
            continue;
        }
        if matches!(
            p.ty_annotation.as_ref().map(|t| &t.kind),
            Some(crate::ast::TypeExprKind::Named(_))
        ) {
            suspect_params.insert(name);
        }
    }
    if suspect_params.is_empty() {
        return;
    }

    // Walk body, flag FieldAccess whose receiver is an Ident in suspect_params.
    fn visit(
        expr: &crate::ast::Expr,
        suspects: &std::collections::HashSet<String>,
        interner: &crate::interner::Interner,
        fn_name: &str,
        diagnostics: &mut Vec<crate::diagnostic::Diagnostic>,
    ) {
        use crate::ast::ExprKind;
        if let ExprKind::FieldAccess { receiver, field } = &expr.kind {
            if let ExprKind::Ident(sym) = &receiver.kind {
                let recv_name = interner.resolve(*sym);
                if suspects.contains(recv_name) {
                    let field_name = interner.resolve(*field);
                    diagnostics.push(
                        Diagnostic::warning(format!(
                            "@export function '{fn_name}' references model weight '.{field_name}' via '{recv_name}' — \
the exported symbol does not yet load weights from NslModel*. Either wait for \
weight-loading integration or restructure the function to take weights as explicit inputs."
                        ))
                        .with_label(expr.span, "model-weight reference"),
                    );
                }
            }
        }
        for child in expr.children() {
            visit(child, suspects, interner, fn_name, diagnostics);
        }
    }

    for stmt in &fn_def.body {
        crate::ast::walk_expr_in_stmt(stmt, |e| {
            visit(e, &suspect_params, interner, &fn_name, diagnostics);
        });
    }
}
```

Call it from `validate_exports` (existing function) for each `@export`-decorated function, right after the decorator validation passes. If any AST walker helpers differ from the sketch above (e.g. `Expr::children`, `walk_expr_in_stmt`), adapt to the crate's real API — grep the crate for existing walk patterns.

- [ ] **Step 4: Run tests**

Run: `cargo test -p nsl-semantic export --lib`
Expected: all 9 tests pass (7 old + 2 new).

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-semantic/src/export.rs
git commit -m "feat(m62): warn on model-weight access in @export bodies"
```

---

## Task 8: Python integration test — call `add` via ctypes

**Files:**
- Modify: `python/tests/test_m62_export.py`

- [ ] **Step 1: Add the test**

Append to `python/tests/test_m62_export.py`:

```python
import ctypes


class NslTensorDesc(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("ndim", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


def _make_f32_desc(values):
    n = len(values)
    data = (ctypes.c_float * n)(*values)
    shape = (ctypes.c_int64 * 1)(n)
    desc = NslTensorDesc(
        data=ctypes.cast(data, ctypes.c_void_p),
        shape=shape,
        strides=None,
        ndim=1,
        dtype=0,           # f32 — matches NSL runtime dtype encoding (dtype=0 CPU f64 historically, verify)
        device_type=0,     # CPU
        device_id=0,
    )
    # Keep the underlying buffers alive by returning them alongside the desc.
    return desc, data, shape


def test_add_actually_computes_sum(shared_lib):
    lib = ctypes.CDLL(str(shared_lib))
    lib.add.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NslTensorDesc),
        ctypes.POINTER(NslTensorDesc),
        ctypes.POINTER(NslTensorDesc),
    ]
    lib.add.restype = ctypes.c_int32

    a_desc, a_buf, a_shape = _make_f32_desc([1.0, 2.0, 3.0, 4.0])
    b_desc, b_buf, b_shape = _make_f32_desc([10.0, 20.0, 30.0, 40.0])
    ret = NslTensorDesc()

    dummy_model = ctypes.c_void_p(1)
    rc = lib.add(
        dummy_model,
        ctypes.byref(a_desc),
        ctypes.byref(b_desc),
        ctypes.byref(ret),
    )
    assert rc == 0, f"add returned {rc}"
    assert ret.data, "ret.data must be non-null after successful call"

    result = ctypes.cast(ret.data, ctypes.POINTER(ctypes.c_float * 4)).contents
    assert list(result) == [11.0, 22.0, 33.0, 44.0]


def test_null_model_returns_error(shared_lib):
    lib = ctypes.CDLL(str(shared_lib))
    lib.add.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NslTensorDesc),
        ctypes.POINTER(NslTensorDesc),
        ctypes.POINTER(NslTensorDesc),
    ]
    lib.add.restype = ctypes.c_int32

    a_desc, _a, _sa = _make_f32_desc([1.0, 2.0, 3.0, 4.0])
    b_desc, _b, _sb = _make_f32_desc([10.0, 20.0, 30.0, 40.0])
    ret = NslTensorDesc()

    rc = lib.add(None, ctypes.byref(a_desc), ctypes.byref(b_desc), ctypes.byref(ret))
    assert rc == -1
```

Before wiring the dtype field, grep `crates/nsl-runtime/src/c_api.rs` for the dtype enum definition to pick the correct integer for `f32` (the prior `NslTensor` extension recorded dtype=0=f64, dtype=1=f32 per auto-memory). If f32 is dtype=1, change both calls above.

- [ ] **Step 2: Run**

Run: `py -m pytest python/tests/test_m62_export.py -v`
Expected:
- 3 pre-existing tests pass.
- `test_add_actually_computes_sum` passes.
- `test_null_model_returns_error` passes.

If `shared_lib` fixture skips because of build-chain issues, diagnose — per auto-memory the fixture is the E2E gate, and this PR is supposed to close it.

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_m62_export.py
git commit -m "test(m62): end-to-end ctypes call of @export add + null-model error"
```

---

## Task 9: Final regression sweep + grad-context unblock check

**Files:** none (verification + optional test-state change)

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace`
Expected: green.

- [ ] **Step 2: Full python test**

Run: `py -m pytest python/tests/ -v`
Expected: previously-skipped `test_autograd.py::test_linear_reference`, `test_error_raises`, `test_enable_grad_pairing`, `test_matmul_chain` now run. If they pass, remove their `@pytest.mark.skip(...)` decorations. If any still skip/fail, document precisely why in the commit message — do not force-fix unrelated breakage here.

- [ ] **Step 3: Commit (only if test-state changed)**

```bash
git add python/tests/test_autograd.py
git commit -m "test(m62): unblock test_autograd.py now that @export wrappers are callable"
```

- [ ] **Step 4: Spec-compliance self-audit**

Walk spec §8 (Success Criteria) 1–6 and confirm each. If any item is missed, loop back to the owning task.

---

## Self-Review Notes

**Spec coverage:**
- §4.1 two-function emission — Task 4.
- §4.2 `ExportWrapper` struct — Task 1.
- §4.3 `build_c_abi_wrapper_signature` — Task 2.
- §4.4 `emit_c_abi_wrapper` body — Task 5.
- §4.5 declaration-loop rewrite — Task 4.
- §4.6 `nsl_set_error_cstr` — Task 3.
- §4.7 wrapper-emission timing — Task 6.
- §5 model-weight warning — Task 7.
- §6.1 signature unit tests — Task 2.
- §6.2 integration test — Task 8.
- §6.3 semantic warning test — Task 7.
- §6.4 regression guard — Task 9.
- §7 risks: tuple-input/return explicitly rejected with `CodegenError::unsupported` in Task 5, matching the spec's "defer multi-return to a follow-up" position.
- §8 success criteria 1-6 — verified by Task 9 step 4.

**Type consistency:** `ExportWrapper` field names match between Task 1 (declaration) and Task 4 (construction) and Task 5 (consumption). Method names `emit_export_wrappers` (Task 6), `emit_c_abi_wrapper` (Task 5), `build_c_abi_wrapper_signature` (Task 2) used identically throughout.

**Placeholder scan:** None detected. Every code-producing step carries the code. Risk §7 items are explicitly surfaced as `CodegenError::unsupported` returns in Task 5 step 1 rather than left as TODOs.

**Known adaptation points (not placeholders — genuinely codebase-dependent):**
- Task 4 step 2 notes the existing `is_export` block may need shape adjustments to match actual locals.
- Task 7 step 3 notes AST walker helpers may differ — pattern is to grep the crate's real API.
- Task 8 step 1 notes dtype integer may need flipping per runtime encoding.

These are honest "read the existing code" cues, not TODOs.
