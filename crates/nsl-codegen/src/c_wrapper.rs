//! Per-`@export` C-ABI wrapper emission.
//!
//! See docs/superpowers/specs/2026-04-15-m62-c-wrappers-design.md.

use crate::c_header::{ExportDtype, ExportInfo, ExportTypeInfo};
#[cfg(test)]
use crate::c_header::ExportParamInfo;
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

pub fn emit_c_abi_wrapper(
    compiler: &mut Compiler,
    wrapper: &ExportWrapper,
) -> Result<(), CodegenError> {
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

        // ── Convert tensor inputs; pass scalars through ───────────────────────
        let mut internal_args: Vec<cranelift_codegen::ir::Value> = Vec::new();
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
