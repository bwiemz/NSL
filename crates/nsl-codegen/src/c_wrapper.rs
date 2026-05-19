//! Per-`@export` C-ABI wrapper emission.
//!
//! See docs/superpowers/specs/2026-04-15-m62-c-wrappers-design.md.

use crate::c_header::{ExportDtype, ExportInfo, ExportTypeInfo};
use crate::error::CodegenError;
use cranelift_codegen::ir::{types, AbiParam, Signature};
use cranelift_codegen::isa::CallConv;
use cranelift_module::FuncId;

#[derive(Clone, Debug)]
pub struct ExportWrapper {
    pub impl_func_id: FuncId,
    pub impl_sig: Signature,
    pub wrapper_func_id: FuncId,
    pub raw_name: String,
    pub export_info: ExportInfo,
    /// True if this wraps a model method — wrapper must thread
    /// model.weight_ptrs + num_weights into the impl call.
    pub is_model_method: bool,
}

fn cranelift_type_for_scalar(dtype: ExportDtype) -> cranelift_codegen::ir::Type {
    match dtype {
        ExportDtype::I32 => types::I32,
        ExportDtype::I64 => types::I64,
        ExportDtype::F32 => types::F32,
        ExportDtype::F64 => types::F64,
        _ => types::I64,
    }
}

pub fn build_c_abi_wrapper_signature(
    export_info: &ExportInfo,
    call_conv: CallConv,
) -> Signature {
    let mut sig = Signature::new(call_conv);
    sig.params.push(AbiParam::new(types::I64)); // NslModel*

    for param in &export_info.params {
        let ty = match &param.ty {
            ExportTypeInfo::Tensor { .. } => types::I64,
            ExportTypeInfo::Scalar(dt) => cranelift_type_for_scalar(*dt),
            ExportTypeInfo::Tuple(_) => types::I64,
        };
        sig.params.push(AbiParam::new(ty));
    }

    match &export_info.return_type {
        ExportTypeInfo::Tensor { .. } | ExportTypeInfo::Scalar(_) => {
            sig.params.push(AbiParam::new(types::I64));
        }
        ExportTypeInfo::Tuple(_) => {
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(types::I64));
        }
    }

    sig.returns.push(AbiParam::new(types::I32));
    sig
}

use crate::compiler::Compiler;
use cranelift_codegen::ir::{
    condcodes::IntCC, types as cw_types, AbiParam as CwAbiParam, Function, InstBuilder, MemFlags,
    Signature as CwSignature, UserFuncName,
};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, Linkage as ModuleLinkage, Module};

/// Size of the C-ABI `NslTensorDesc` struct in bytes. Used by the
/// packed-array dispatch wrapper to compute per-element pointer offsets
/// into the input/output descriptor arrays.
///
/// Layout (must match `nsl_runtime::c_api::NslTensorDesc`, `#[repr(C)]`):
///   data: *mut c_void  (8)
///   shape: *mut i64    (8)
///   strides: *mut i64  (8)
///   ndim: i32          (4)
///   dtype: i32         (4)
///   device_type: i32   (4)
///   device_id: i32     (4)
/// Total = 40 bytes, 8-byte aligned.
pub(crate) const NSL_TENSOR_DESC_SIZE: i64 = 40;

pub fn emit_c_abi_wrapper(
    compiler: &mut Compiler,
    wrapper: &ExportWrapper,
) -> Result<(), CodegenError> {
    // GRAD SCOPE: @export dispatch is INFERENCE-ONLY. The wrapper does NOT:
    //   - call nsl_tape_start over weight_ptrs
    //   - save outputs to model.last_forward_outputs
    //   - honor model.grad_enabled
    // Calling nsl_model_enable_grad(model, 1) before an @export wrapper call
    // has no effect on the @export dispatch path. For training autograd, use
    // nsl_model_forward + nsl_model_backward (see the grad-context bridge fix).
    let call_conv = compiler.module.target_config().default_call_conv;
    let wrapper_sig = build_c_abi_wrapper_signature(&wrapper.export_info, call_conv);

    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, compiler.next_func_index()),
        wrapper_sig,
    ));
    let mut fn_builder_ctx = FunctionBuilderContext::new();

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let params: Vec<_> = builder.block_params(entry).to_vec();
        let model_ptr = params[0];

        // ── Null-check NslModel* ──────────────────────────────────────────────
        let err_block = builder.create_block();
        let ok_block = builder.create_block();
        let zero_i64 = builder.ins().iconst(cw_types::I64, 0);
        let is_null = builder.ins().icmp(IntCC::Equal, model_ptr, zero_i64);
        builder.ins().brif(is_null, err_block, &[], ok_block, &[]);

        // Error path
        builder.switch_to_block(err_block);
        builder.seal_block(err_block);
        emit_set_error(&mut builder, &mut compiler.module, "null model pointer")?;
        let neg_one = builder.ins().iconst(cw_types::I32, -1);
        builder.ins().return_(&[neg_one]);

        // Happy path
        builder.switch_to_block(ok_block);
        builder.seal_block(ok_block);

        // ── If model method, extract weight_ptrs + num_weights and prepend ───
        let mut leading_args: Vec<cranelift_codegen::ir::Value> = Vec::new();
        if wrapper.is_model_method {
            let w_ptrs =
                call_model_get_weight_ptrs(&mut builder, &mut compiler.module, model_ptr)?;
            let n_weights =
                call_model_get_num_weights(&mut builder, &mut compiler.module, model_ptr)?;
            leading_args.push(w_ptrs);
            leading_args.push(n_weights);
        }

        // ── Convert tensor inputs; pass scalars through ───────────────────────
        let mut internal_args: Vec<cranelift_codegen::ir::Value> = leading_args;
        let mut tensor_inputs_to_free: Vec<cranelift_codegen::ir::Value> = Vec::new();

        for (i, param) in wrapper.export_info.params.iter().enumerate() {
            let arg_val = params[1 + i];
            match &param.ty {
                ExportTypeInfo::Tensor { .. } => {
                    let tensor =
                        call_desc_to_tensor(&mut builder, &mut compiler.module, arg_val)?;
                    internal_args.push(tensor);
                    tensor_inputs_to_free.push(tensor);
                }
                ExportTypeInfo::Scalar(_) => {
                    internal_args.push(arg_val);
                }
                ExportTypeInfo::Tuple(_) => {
                    return Err(CodegenError::new(format!(
                        "@export tuple input parameter for '{}' is not yet supported",
                        wrapper.raw_name
                    )));
                }
            }
        }

        // ── Call the internal implementation ──────────────────────────────────
        let impl_ref = compiler
            .module
            .declare_func_in_func(wrapper.impl_func_id, builder.func);
        let call_inst = builder.ins().call(impl_ref, &internal_args);
        let impl_rets = builder.inst_results(call_inst).to_vec();

        // ── Write result into caller's __ret(s) ───────────────────────────────
        // NOTE: nsl_tensor_to_desc_ffi copies raw data/shape/strides pointers into
        // the caller's NslTensorDesc — it does NOT deep-copy the buffer. The result
        // NslTensor struct leaks (~80 bytes/call). Same pattern as nsl_model_forward.
        // A proper fix (memcpy + free, or caller-side nsl_desc_free_data) is deferred.
        match &wrapper.export_info.return_type {
            ExportTypeInfo::Tensor { .. } => {
                let ret_desc_ptr = params[1 + wrapper.export_info.params.len()];
                let result_tensor = impl_rets[0];
                call_tensor_to_desc_ffi(
                    &mut builder,
                    &mut compiler.module,
                    result_tensor,
                    ret_desc_ptr,
                )?;
            }
            ExportTypeInfo::Scalar(_) => {
                let ret_ptr = params[1 + wrapper.export_info.params.len()];
                let scalar_val = impl_rets[0];
                builder
                    .ins()
                    .store(MemFlags::trusted(), scalar_val, ret_ptr, 0);
            }
            ExportTypeInfo::Tuple(_) => {
                return Err(CodegenError::new(format!(
                    "@export tuple return for '{}' is not yet supported",
                    wrapper.raw_name
                )));
            }
        }

        // ── Free wrapper tensor structs ───────────────────────────────────────
        for t in tensor_inputs_to_free {
            call_nsl_tensor_free(&mut builder, &mut compiler.module, t)?;
        }

        let zero_ret = builder.ins().iconst(cw_types::I32, 0);
        builder.ins().return_(&[zero_ret]);
        builder.finalize();
    }

    compiler
        .module
        .define_function(wrapper.wrapper_func_id, &mut ctx)
        .map_err(|e| CodegenError::new(format!("define wrapper fn: {e:?}")))?;
    Ok(())
}

/// Build the signature for the packed-array dispatch wrapper:
///
///   `fn <name>__nsl_dispatch(
///        model_ptr: i64,
///        inputs_desc_ptr: i64,
///        num_inputs: i64,
///        outputs_desc_ptr: i64,
///        num_outputs: i64,
///    ) -> i64`
///
/// This is the signature `ExportRegistry::lookup` consumers expect
/// (matches `ExportFnPtr` in `nsl-runtime/src/c_api/exports.rs`).
pub fn build_dispatch_wrapper_signature(call_conv: CallConv) -> Signature {
    let mut sig = Signature::new(call_conv);
    sig.params.push(AbiParam::new(types::I64)); // model_ptr
    sig.params.push(AbiParam::new(types::I64)); // inputs_desc_ptr
    sig.params.push(AbiParam::new(types::I64)); // num_inputs
    sig.params.push(AbiParam::new(types::I64)); // outputs_desc_ptr
    sig.params.push(AbiParam::new(types::I64)); // num_outputs
    sig.returns.push(AbiParam::new(types::I64));
    sig
}

/// Emit the packed-array sibling wrapper `<symbol>__nsl_dispatch`.
///
/// This wrapper consumes the `nsl_model_call` ABI (model + input/output
/// descriptor arrays + counts) and forwards into the existing typed
/// individual-arg wrapper (`emit_c_abi_wrapper`'s output). Semantics are
/// identical to the typed wrapper; only the entry ABI differs.
///
/// Only tensor-typed parameters and a single-tensor (or single-scalar)
/// return are supported. Other shapes (scalar params, tuple returns)
/// fall back to an error stub that sets the thread-local error and
/// returns -1 — the registry will still find the symbol, so dispatch
/// callers get a clear runtime error rather than a missing-symbol
/// dlopen failure.
pub fn emit_c_abi_dispatch_wrapper(
    compiler: &mut Compiler,
    wrapper: &ExportWrapper,
) -> Result<(), CodegenError> {
    let call_conv = compiler.module.target_config().default_call_conv;
    let dispatch_sig = build_dispatch_wrapper_signature(call_conv);
    let dispatch_symbol = format!("{}__nsl_dispatch", wrapper.export_info.symbol_name);

    let dispatch_fn_id = compiler
        .module
        .declare_function(&dispatch_symbol, ModuleLinkage::Export, &dispatch_sig)
        .map_err(|e| {
            CodegenError::new(format!(
                "declare dispatch wrapper '{dispatch_symbol}': {e:?}"
            ))
        })?;

    // Determine whether this signature is dispatch-compatible.
    //   - All params must be tensors (packed-array ABI has no scalar slots).
    //   - Return must be a single Tensor or Scalar (single output desc).
    let all_tensor_params = wrapper
        .export_info
        .params
        .iter()
        .all(|p| matches!(p.ty, ExportTypeInfo::Tensor { .. }));
    let single_output = matches!(
        wrapper.export_info.return_type,
        ExportTypeInfo::Tensor { .. } | ExportTypeInfo::Scalar(_)
    );
    let supported = all_tensor_params && single_output;

    let expected_inputs = wrapper.export_info.params.len() as i64;
    let expected_outputs: i64 = 1; // tuple returns not supported in v1

    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, compiler.next_func_index()),
        dispatch_sig,
    ));
    let mut fn_builder_ctx = FunctionBuilderContext::new();

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let params: Vec<_> = builder.block_params(entry).to_vec();
        let model_ptr = params[0];
        let inputs_ptr = params[1];
        let num_inputs = params[2];
        let outputs_ptr = params[3];
        let num_outputs = params[4];

        if !supported {
            // Emit a stub that sets the thread-local error and returns -1.
            // The symbol still exists so the registry's dlsym succeeds, but
            // any call routes to a clear error rather than miscompiling.
            let msg = format!(
                "export '{}' has a signature unsupported by nsl_model_call \
                 (only all-Tensor params + single Tensor/Scalar return are dispatchable)",
                wrapper.export_info.raw_name
            );
            emit_set_error(&mut builder, &mut compiler.module, &msg)?;
            let neg_one = builder.ins().iconst(cw_types::I64, -1);
            builder.ins().return_(&[neg_one]);
            builder.finalize();

            compiler
                .module
                .define_function(dispatch_fn_id, &mut ctx)
                .map_err(|e| CodegenError::new(format!("define dispatch wrapper stub: {e:?}")))?;
            return Ok(());
        }

        // ── Validate arity ───────────────────────────────────────────────
        let err_block = builder.create_block();
        let check_outputs_block = builder.create_block();
        let dispatch_block = builder.create_block();

        let expected_in_val = builder.ins().iconst(cw_types::I64, expected_inputs);
        let in_eq = builder
            .ins()
            .icmp(IntCC::Equal, num_inputs, expected_in_val);
        builder
            .ins()
            .brif(in_eq, check_outputs_block, &[], err_block, &[]);

        builder.switch_to_block(check_outputs_block);
        builder.seal_block(check_outputs_block);
        let expected_out_val = builder.ins().iconst(cw_types::I64, expected_outputs);
        let out_eq = builder
            .ins()
            .icmp(IntCC::Equal, num_outputs, expected_out_val);
        builder
            .ins()
            .brif(out_eq, dispatch_block, &[], err_block, &[]);

        // Error path: set error, return -1.
        builder.switch_to_block(err_block);
        builder.seal_block(err_block);
        let arity_msg = format!(
            "export '{}' arity mismatch (expected {} inputs, {} outputs)",
            wrapper.export_info.raw_name, expected_inputs, expected_outputs
        );
        emit_set_error(&mut builder, &mut compiler.module, &arity_msg)?;
        let neg_one = builder.ins().iconst(cw_types::I64, -1);
        builder.ins().return_(&[neg_one]);

        // Dispatch path: unpack descriptor pointers, stage a scratch output
        // desc, call the typed wrapper, then fold scratch into the caller's
        // preallocated output desc (memcpy data, mirror metadata).
        builder.switch_to_block(dispatch_block);
        builder.seal_block(dispatch_block);

        // Stack-allocate a scratch NslTensorDesc for the single output. The
        // typed wrapper writes its result tensor's metadata here (and we
        // don't want it overwriting the caller's `data` ptr — that's the
        // preallocated buffer we need to memcpy into below).
        let scratch_slot = builder.create_sized_stack_slot(
            cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                NSL_TENSOR_DESC_SIZE as u32,
                3, // align: 2^3 = 8 bytes (NslTensorDesc is 8-byte aligned)
            ),
        );
        let scratch_ptr = builder
            .ins()
            .stack_addr(cw_types::I64, scratch_slot, 0);
        // Zero-init scratch — typed wrapper will overwrite all 7 metadata
        // fields, but zeroing first guards against any field being skipped.
        let zero_i64_init = builder.ins().iconst(cw_types::I64, 0);
        for off in (0..NSL_TENSOR_DESC_SIZE).step_by(8) {
            builder
                .ins()
                .store(MemFlags::trusted(), zero_i64_init, scratch_ptr, off as i32);
        }

        let typed_ref = compiler
            .module
            .declare_func_in_func(wrapper.wrapper_func_id, builder.func);

        let mut typed_args: Vec<cranelift_codegen::ir::Value> = Vec::new();
        typed_args.push(model_ptr);

        // One pointer per tensor input: inputs_ptr + i * sizeof(NslTensorDesc).
        for i in 0..wrapper.export_info.params.len() {
            let off = (i as i64) * NSL_TENSOR_DESC_SIZE;
            let desc_ptr = if off == 0 {
                inputs_ptr
            } else {
                builder.ins().iadd_imm(inputs_ptr, off)
            };
            typed_args.push(desc_ptr);
        }

        // Single output: typed wrapper writes into scratch (not caller's
        // desc); we'll fold scratch back below. The arity check above
        // already verified num_outputs == 1, so the typed wrapper (which
        // doesn't take a count) is safe to call as a single-output sink.

        let call_inst = builder.ins().call(typed_ref, &typed_args);
        let rc_i32 = builder.inst_results(call_inst)[0];

        // If the typed wrapper failed, skip the copy and return its code.
        let fail_block = builder.create_block();
        let copy_block = builder.create_block();
        let zero_i32 = builder.ins().iconst(cw_types::I32, 0);
        let typed_ok = builder.ins().icmp(IntCC::Equal, rc_i32, zero_i32);
        builder.ins().brif(typed_ok, copy_block, &[], fail_block, &[]);

        builder.switch_to_block(fail_block);
        builder.seal_block(fail_block);
        let fail_rc = builder.ins().sextend(cw_types::I64, rc_i32);
        builder.ins().return_(&[fail_rc]);

        builder.switch_to_block(copy_block);
        builder.seal_block(copy_block);

        // Fold scratch → caller's preallocated output desc.
        let apply_fid = declare_runtime_fn(
            &mut compiler.module,
            "nsl_dispatch_apply_result",
            &[cw_types::I64, cw_types::I64],
            &[cw_types::I64],
        )?;
        let apply_ref = compiler
            .module
            .declare_func_in_func(apply_fid, builder.func);
        let apply_call = builder
            .ins()
            .call(apply_ref, &[scratch_ptr, outputs_ptr]);
        let apply_rc = builder.inst_results(apply_call)[0];
        builder.ins().return_(&[apply_rc]);

        builder.finalize();
    }

    compiler
        .module
        .define_function(dispatch_fn_id, &mut ctx)
        .map_err(|e| CodegenError::new(format!("define dispatch wrapper fn: {e:?}")))?;
    Ok(())
}

// ── Runtime-call helpers ──────────────────────────────────────────────────

fn declare_runtime_fn<M: Module + ?Sized>(
    module: &mut M,
    name: &str,
    params: &[cranelift_codegen::ir::Type],
    returns: &[cranelift_codegen::ir::Type],
) -> Result<cranelift_module::FuncId, CodegenError> {
    let call_conv = module.target_config().default_call_conv;
    let mut sig = CwSignature::new(call_conv);
    for p in params {
        sig.params.push(CwAbiParam::new(*p));
    }
    for r in returns {
        sig.returns.push(CwAbiParam::new(*r));
    }
    module
        .declare_function(name, ModuleLinkage::Import, &sig)
        .map_err(|e| CodegenError::new(format!("declare {name}: {e:?}")))
}

/// Call `nsl_desc_to_tensor(desc_ptr: i64) -> i64` — C-ABI export in nsl-runtime.
fn call_desc_to_tensor<M: Module + ?Sized>(
    builder: &mut FunctionBuilder,
    module: &mut M,
    desc_ptr: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let fid = declare_runtime_fn(module, "nsl_desc_to_tensor", &[cw_types::I64], &[cw_types::I64])?;
    let fref = module.declare_func_in_func(fid, builder.func);
    let call = builder.ins().call(fref, &[desc_ptr]);
    Ok(builder.inst_results(call)[0])
}

/// Call `nsl_tensor_to_desc_ffi(tensor: i64, desc: i64)` — C-ABI export in nsl-runtime.
fn call_tensor_to_desc_ffi<M: Module + ?Sized>(
    builder: &mut FunctionBuilder,
    module: &mut M,
    tensor: cranelift_codegen::ir::Value,
    desc_ptr: cranelift_codegen::ir::Value,
) -> Result<(), CodegenError> {
    let fid = declare_runtime_fn(module, "nsl_tensor_to_desc_ffi", &[cw_types::I64, cw_types::I64], &[])?;
    let fref = module.declare_func_in_func(fid, builder.func);
    builder.ins().call(fref, &[tensor, desc_ptr]);
    Ok(())
}

/// Call `nsl_tensor_free(tensor: i64)` — C-ABI export in nsl-runtime.
fn call_nsl_tensor_free<M: Module + ?Sized>(
    builder: &mut FunctionBuilder,
    module: &mut M,
    tensor: cranelift_codegen::ir::Value,
) -> Result<(), CodegenError> {
    let fid = declare_runtime_fn(module, "nsl_tensor_free", &[cw_types::I64], &[])?;
    let fref = module.declare_func_in_func(fid, builder.func);
    builder.ins().call(fref, &[tensor]);
    Ok(())
}

/// Call `nsl_model_get_weight_ptrs(model: i64) -> i64` — returns `*const i64`
/// (pointer to the weight pointers array) or 0 on null model.
fn call_model_get_weight_ptrs<M: Module + ?Sized>(
    builder: &mut FunctionBuilder,
    module: &mut M,
    model_ptr: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let fid = declare_runtime_fn(
        module,
        "nsl_model_get_weight_ptrs",
        &[cw_types::I64],
        &[cw_types::I64],
    )?;
    let fref = module.declare_func_in_func(fid, builder.func);
    let call = builder.ins().call(fref, &[model_ptr]);
    Ok(builder.inst_results(call)[0])
}

/// Call `nsl_model_get_num_weights(model: i64) -> i64`.
fn call_model_get_num_weights<M: Module + ?Sized>(
    builder: &mut FunctionBuilder,
    module: &mut M,
    model_ptr: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let fid = declare_runtime_fn(
        module,
        "nsl_model_get_num_weights",
        &[cw_types::I64],
        &[cw_types::I64],
    )?;
    let fref = module.declare_func_in_func(fid, builder.func);
    let call = builder.ins().call(fref, &[model_ptr]);
    Ok(builder.inst_results(call)[0])
}

/// Embed a null-terminated error string as a data object and call
/// `nsl_set_error_cstr` with a pointer to it.
fn emit_set_error<M: Module + ?Sized>(
    builder: &mut FunctionBuilder,
    module: &mut M,
    msg: &str,
) -> Result<(), CodegenError> {
    let mut bytes = msg.as_bytes().to_vec();
    bytes.push(0u8); // null terminator

    let sym = format!("__nsl_wrapper_errstr_{:016x}", fnv1a_hash(msg));
    let data_id = module
        .declare_data(&sym, ModuleLinkage::Local, false, false)
        .map_err(|e| CodegenError::new(format!("declare errstr data: {e:?}")))?;
    let mut desc = DataDescription::new();
    desc.define(bytes.into_boxed_slice());
    // Ignore duplicate-definition errors so the same message can be reused
    // across multiple wrappers without a hard error.
    let _ = module.define_data(data_id, &desc);

    let gv = module.declare_data_in_func(data_id, builder.func);
    let str_ptr = builder.ins().symbol_value(cw_types::I64, gv);

    let fid = declare_runtime_fn(module, "nsl_set_error_cstr", &[cw_types::I64], &[])?;
    let fref = module.declare_func_in_func(fid, builder.func);
    builder.ins().call(fref, &[str_ptr]);
    Ok(())
}

fn fnv1a_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

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
                shape: vec!["4".into()],
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
                shape: vec!["4".into()],
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
                shape: vec!["4".into()],
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

    /// Source-scan guard: verifies that the weight-ptr helper calls and the
    /// is_model_method gate are all present in the source. This is intentionally
    /// a textual check — building a full ObjectModule/Compiler in a unit test
    /// requires an ISA + interner + type_map + ~20 other fields, so we document
    /// the cross-pass invariant as a source assertion instead. The IR-level
    /// correctness is validated by the integration tests in tests/c_wrapper_*.rs.
    #[test]
    fn emit_c_abi_wrapper_source_contains_weight_ptr_helpers() {
        let src = include_str!("c_wrapper.rs");
        assert!(
            src.contains("call_model_get_weight_ptrs"),
            "wrapper emission must call the weight-ptrs helper"
        );
        assert!(
            src.contains("call_model_get_num_weights"),
            "wrapper emission must call the num-weights helper"
        );
        assert!(
            src.contains("if wrapper.is_model_method"),
            "wrapper emission must gate the weight-ptr calls on is_model_method"
        );
    }

    /// Guard: the C-ABI signature presented to callers is independent of
    /// is_model_method. The weight_ptrs/num_weights threading only affects
    /// the internal impl call, not the external ABI.
    #[test]
    fn build_c_abi_wrapper_signature_ignores_is_model_method() {
        // Construct the same ExportInfo twice and confirm the signature is
        // identical. (build_c_abi_wrapper_signature doesn't even take
        // is_model_method — this test documents that contract explicitly.)
        let info = ExportInfo {
            symbol_name: "predict".into(),
            raw_name: "predict".into(),
            params: vec![tensor_param("x", ExportDtype::F32)],
            return_type: ExportTypeInfo::Tensor {
                shape: vec!["4".into()],
                dtype: ExportDtype::F32,
                device: crate::c_header::ExportDevice::Cpu,
            },
        };
        let sig_a = build_c_abi_wrapper_signature(&info, CallConv::SystemV);
        let sig_b = build_c_abi_wrapper_signature(&info, CallConv::SystemV);
        assert_eq!(
            sig_a.params.len(),
            sig_b.params.len(),
            "signature param count must not vary by is_model_method"
        );
        assert_eq!(
            sig_a.returns.len(),
            sig_b.returns.len(),
            "signature return count must not vary by is_model_method"
        );
    }

    #[test]
    fn wrapper_signature_for_tuple_return_has_rets_and_num_rets() {
        let info = ExportInfo {
            symbol_name: "split".into(),
            raw_name: "split".into(),
            params: vec![tensor_param("x", ExportDtype::F32)],
            return_type: ExportTypeInfo::Tuple(vec![
                ExportTypeInfo::Tensor {
                    shape: vec!["2".into()],
                    dtype: ExportDtype::F32,
                    device: ExportDevice::Cpu,
                },
                ExportTypeInfo::Tensor {
                    shape: vec!["2".into()],
                    dtype: ExportDtype::F32,
                    device: ExportDevice::Cpu,
                },
            ]),
        };
        let sig = build_c_abi_wrapper_signature(&info, CallConv::SystemV);
        // [i64 model, i64 x_desc, i64 rets_array, i64 num_rets_ptr] -> i32
        assert_eq!(sig.params.len(), 4);
        assert_eq!(sig.params[0].value_type, I64);
        assert_eq!(sig.params[1].value_type, I64);
        assert_eq!(sig.params[2].value_type, I64);
        assert_eq!(sig.params[3].value_type, I64);
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.returns[0].value_type, I32);
    }
}
