//! Per-`@export` C-ABI wrapper emission.
//!
//! See docs/superpowers/specs/2026-04-15-m62-c-wrappers-design.md.

use crate::c_header::{ExportDtype, ExportInfo, ExportTypeInfo};
#[cfg(test)]
use crate::c_header::ExportParamInfo;
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
}
