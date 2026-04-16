//! M62 `@export` C header emission + type model.
//!
//! `ExportInfo` tracks each `@export`-decorated function so the CLI
//! can emit a matching C header after shared-lib codegen completes.
//! This module also owns the C-type lowering used by the header
//! emitter (Task 5) and the C-wrapper emission (Task 4).

use serde::{Deserialize, Serialize};

use nsl_ast::decl::FnDef;
use nsl_ast::types::{DeviceExpr, DimExpr, DimValue, TypeExpr, TypeExprKind};
use nsl_lexer::Interner;

/// Per-function metadata for a single `@export` function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportInfo {
    /// The symbol name that appears in the `.so`/`.dylib`/`.dll`'s export table.
    /// Either the NSL function's raw name, or the user-provided `name="..."`.
    pub symbol_name: String,
    /// The NSL-side function name (used for diagnostics and for looking up
    /// the function body in `Compiler.registry.functions`).
    pub raw_name: String,
    /// Parameter types in declaration order.
    pub params: Vec<ExportParamInfo>,
    /// Return type. A tuple return is represented as `ExportTypeInfo::Tuple(...)`.
    pub return_type: ExportTypeInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportParamInfo {
    pub name: String,
    pub ty: ExportTypeInfo,
}

/// C-ABI-compatible type shapes that `@export` functions may use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportTypeInfo {
    /// `Tensor<[...], dtype, device>`.
    Tensor {
        /// Shape dims stringified. Named dims like `"B"` stay named;
        /// literal ints like `4` stringify to `"4"`.
        shape: Vec<String>,
        dtype: ExportDtype,
        device: ExportDevice,
    },
    /// Primitive scalar.
    Scalar(ExportDtype),
    /// Tuple of any of the above (for multi-output functions).
    Tuple(Vec<ExportTypeInfo>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportDtype {
    F32, F64, F16, BF16,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportDevice {
    Cpu,
    Cuda,
    /// Compiler chooses at call time (default for `@export` inputs that
    /// don't explicitly pin a device).
    Any,
}

/// Map an NSL dtype name (as it appears in `Tensor<[...], dtype, ...>`
/// or a bare scalar type) to the corresponding `ExportDtype`. Unknown
/// names default to `F32`; semantic validation (Task 4) rejects anything
/// unsupported before we reach this path.
fn lower_dtype_name(name: &str) -> ExportDtype {
    match name {
        "f32" | "fp32" | "float" | "float32" => ExportDtype::F32,
        "f64" | "fp64" | "double" | "float64" => ExportDtype::F64,
        "f16" | "fp16" | "half" | "float16" => ExportDtype::F16,
        "bf16" | "bfloat16" => ExportDtype::BF16,
        "i8" | "int8" => ExportDtype::I8,
        "i16" | "int16" => ExportDtype::I16,
        "i32" | "int32" | "int" => ExportDtype::I32,
        "i64" | "int64" | "long" => ExportDtype::I64,
        "u8" | "uint8" => ExportDtype::U8,
        "u16" | "uint16" => ExportDtype::U16,
        "u32" | "uint32" => ExportDtype::U32,
        "u64" | "uint64" => ExportDtype::U64,
        "bool" => ExportDtype::Bool,
        _ => ExportDtype::F32,
    }
}

fn lower_device(dev: &DeviceExpr) -> ExportDevice {
    match dev {
        DeviceExpr::Cpu => ExportDevice::Cpu,
        DeviceExpr::Cuda(_) | DeviceExpr::Rocm(_) | DeviceExpr::Metal => ExportDevice::Cuda,
        DeviceExpr::Npu(_) => ExportDevice::Any,
    }
}

fn stringify_dim(dim: &DimExpr, interner: &Interner) -> String {
    match dim {
        DimExpr::Concrete(n) => n.to_string(),
        DimExpr::Symbolic(sym) => interner.resolve(sym.0).unwrap_or("").to_string(),
        DimExpr::Named { name, value } => {
            let name_str = interner.resolve(name.0).unwrap_or("");
            if !name_str.is_empty() {
                name_str.to_string()
            } else {
                match value {
                    DimValue::String(s) => s.clone(),
                    DimValue::Int(n) => n.to_string(),
                }
            }
        }
        DimExpr::Bounded { name, .. } => interner.resolve(name.0).unwrap_or("").to_string(),
        DimExpr::Wildcard => "_".to_string(),
    }
}

/// Lower an AST `TypeExpr` into an `ExportTypeInfo`. Only the subset
/// allowed by `@export` semantic validation (Task 4) is meaningfully
/// represented; unsupported shapes fall through to `Tuple(vec![])`.
pub fn lower_type_expr(ty: &TypeExpr, interner: &Interner) -> ExportTypeInfo {
    match &ty.kind {
        TypeExprKind::Tensor {
            shape,
            dtype,
            device,
        } => {
            let shape_strs: Vec<String> =
                shape.iter().map(|d| stringify_dim(d, interner)).collect();
            let dtype_str = interner.resolve(dtype.0).unwrap_or("");
            let dev = device
                .as_ref()
                .map(lower_device)
                .unwrap_or(ExportDevice::Any);
            ExportTypeInfo::Tensor {
                shape: shape_strs,
                dtype: lower_dtype_name(dtype_str),
                device: dev,
            }
        }
        TypeExprKind::Param { shape, dtype }
        | TypeExprKind::Buffer { shape, dtype } => {
            let shape_strs: Vec<String> =
                shape.iter().map(|d| stringify_dim(d, interner)).collect();
            let dtype_str = interner.resolve(dtype.0).unwrap_or("");
            ExportTypeInfo::Tensor {
                shape: shape_strs,
                dtype: lower_dtype_name(dtype_str),
                device: ExportDevice::Any,
            }
        }
        TypeExprKind::Named(sym) => {
            let name = interner.resolve(sym.0).unwrap_or("");
            ExportTypeInfo::Scalar(lower_dtype_name(name))
        }
        TypeExprKind::Tuple(elems) => {
            ExportTypeInfo::Tuple(elems.iter().map(|t| lower_type_expr(t, interner)).collect())
        }
        _ => ExportTypeInfo::Tuple(vec![]),
    }
}

impl ExportInfo {
    /// Build an `ExportInfo` from a type-checked `FnDef` signature.
    /// Called from `declaration.rs` when a function is decorated with
    /// `@export`.
    pub fn from_fn_def(
        fn_def: &FnDef,
        raw_name: &str,
        symbol_name: &str,
        interner: &Interner,
    ) -> Self {
        let params = fn_def
            .params
            .iter()
            .map(|p| ExportParamInfo {
                name: interner.resolve(p.name.0).unwrap_or("").to_string(),
                ty: match &p.type_ann {
                    Some(ty) => lower_type_expr(ty, interner),
                    None => ExportTypeInfo::Tuple(vec![]),
                },
            })
            .collect();

        let return_type = match &fn_def.return_type {
            Some(ty) => lower_type_expr(ty, interner),
            None => ExportTypeInfo::Tuple(vec![]),
        };

        Self {
            symbol_name: symbol_name.to_string(),
            raw_name: raw_name.to_string(),
            params,
            return_type,
        }
    }
}

/// Emit a C header for the given `@export` functions. `module_name`
/// is used for the header-guard macro (e.g. "model" → `NSL_MODEL_H`).
pub fn emit(exports: &[ExportInfo], module_name: &str) -> String {
    let guard = format!("NSL_{}_H", sanitize_header_guard(module_name));
    let mut out = String::new();

    out.push_str(&format!("/* {module_name}.h — Auto-generated by NSL compiler */\n"));
    out.push_str(&format!("#ifndef {guard}\n"));
    out.push_str(&format!("#define {guard}\n\n"));
    out.push_str("#include <stdint.h>\n");
    out.push_str("#include <stddef.h>\n\n");
    out.push_str("#ifdef __cplusplus\n");
    out.push_str("extern \"C\" {\n");
    out.push_str("#endif\n\n");

    out.push_str("typedef struct NslModel NslModel;\n\n");
    out.push_str("typedef struct {\n");
    out.push_str("    void*    data;\n");
    out.push_str("    int64_t* shape;\n");
    out.push_str("    int64_t* strides;     /* NULL = contiguous */\n");
    out.push_str("    int32_t  ndim;\n");
    out.push_str("    int32_t  dtype;       /* 0=f32, 1=f64, 2=f16, 3=bf16, 4=i32, 5=i64, 6=i8, 7=u8 */\n");
    out.push_str("    int32_t  device_type; /* 0=CPU, 1=CUDA */\n");
    out.push_str("    int32_t  device_id;\n");
    out.push_str("} NslTensorDesc;\n\n");

    out.push_str("/* Lifecycle (provided by libnsl_runtime) */\n");
    out.push_str("NslModel* nsl_model_create(const char* weights_path);\n");
    out.push_str("void      nsl_model_destroy(NslModel* model);\n\n");

    out.push_str("/* @export functions */\n");
    for info in exports {
        emit_prototype(&mut out, info);
    }
    out.push('\n');

    out.push_str("#ifdef __cplusplus\n");
    out.push_str("}\n");
    out.push_str("#endif\n");
    out.push_str(&format!("#endif /* {guard} */\n"));

    out
}

fn sanitize_header_guard(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
        .collect()
}

fn emit_prototype(out: &mut String, info: &ExportInfo) {
    out.push_str("/* @export dispatch is inference-only. Gradient recording does not flow\n");
    out.push_str(" * through this call path — use nsl_model_forward for training autograd. */\n");
    out.push_str(&format!("int {}(NslModel* model", info.symbol_name));
    for param in &info.params {
        out.push_str(",\n        ");
        emit_param(out, &param.name, &param.ty);
    }
    out.push_str(",\n        ");
    emit_return_arg(out, &info.return_type);
    out.push_str(");\n");
}

fn emit_param(out: &mut String, name: &str, ty: &ExportTypeInfo) {
    match ty {
        ExportTypeInfo::Tensor { .. } => {
            out.push_str(&format!("const NslTensorDesc* {name}"));
        }
        ExportTypeInfo::Scalar(dtype) => {
            out.push_str(&format!("{} {name}", c_type_for_scalar(*dtype)));
        }
        ExportTypeInfo::Tuple(_) => {
            out.push_str(&format!("const NslTensorDesc* {name}_items, int32_t {name}_count"));
        }
    }
}

fn emit_return_arg(out: &mut String, ty: &ExportTypeInfo) {
    match ty {
        ExportTypeInfo::Tensor { .. } => {
            out.push_str("NslTensorDesc* __ret");
        }
        ExportTypeInfo::Scalar(dtype) => {
            out.push_str(&format!("{} * __ret", c_type_for_scalar(*dtype)));
        }
        ExportTypeInfo::Tuple(_) => {
            out.push_str("NslTensorDesc* __rets, int32_t* __num_rets");
        }
    }
}

fn c_type_for_scalar(dtype: ExportDtype) -> &'static str {
    match dtype {
        ExportDtype::I8  => "int8_t",
        ExportDtype::I16 => "int16_t",
        ExportDtype::I32 => "int32_t",
        ExportDtype::I64 => "int64_t",
        ExportDtype::U8  => "uint8_t",
        ExportDtype::U16 => "uint16_t",
        ExportDtype::U32 => "uint32_t",
        ExportDtype::U64 => "uint64_t",
        ExportDtype::F32 => "float",
        ExportDtype::F64 => "double",
        ExportDtype::F16 | ExportDtype::BF16 => "uint16_t",
        ExportDtype::Bool => "int32_t",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_info_basic_shape() {
        let info = ExportInfo {
            symbol_name: "forward".into(),
            raw_name: "forward".into(),
            params: vec![ExportParamInfo {
                name: "x".into(),
                ty: ExportTypeInfo::Tensor {
                    shape: vec!["B".into(), "768".into()],
                    dtype: ExportDtype::F32,
                    device: ExportDevice::Any,
                },
            }],
            return_type: ExportTypeInfo::Tensor {
                shape: vec!["B".into(), "1000".into()],
                dtype: ExportDtype::F32,
                device: ExportDevice::Any,
            },
        };
        assert_eq!(info.params.len(), 1);
        assert_eq!(info.symbol_name, "forward");
    }

    #[test]
    fn export_info_from_simple_fn_def() {
        use nsl_ast::decl::{FnDef, Param};
        use nsl_ast::types::{DimExpr, TypeExpr, TypeExprKind};
        use nsl_ast::{NodeId, Span, Symbol};
        use string_interner::StringInterner;

        let mut interner: Interner = StringInterner::new();
        let x_sym = Symbol(interner.get_or_intern("x"));
        let b_sym = Symbol(interner.get_or_intern("B"));
        let f32_sym = Symbol(interner.get_or_intern("f32"));
        let fn_name_sym = Symbol(interner.get_or_intern("forward"));

        let param_ty = TypeExpr {
            kind: TypeExprKind::Tensor {
                shape: vec![DimExpr::Symbolic(b_sym), DimExpr::Concrete(768)],
                dtype: f32_sym,
                device: None,
            },
            span: Span::dummy(),
            id: NodeId::dummy(),
        };
        let ret_ty = TypeExpr {
            kind: TypeExprKind::Tensor {
                shape: vec![DimExpr::Symbolic(b_sym), DimExpr::Concrete(1000)],
                dtype: f32_sym,
                device: None,
            },
            span: Span::dummy(),
            id: NodeId::dummy(),
        };

        let fn_def = FnDef {
            name: fn_name_sym,
            type_params: vec![],
            effect_params: vec![],
            params: vec![Param {
                name: x_sym,
                type_ann: Some(param_ty),
                default: None,
                is_variadic: false,
                span: Span::dummy(),
            }],
            return_type: Some(ret_ty),
            return_effect: None,
            body: nsl_ast::stmt::Block {
                stmts: vec![],
                span: Span::dummy(),
            },
            is_async: false,
            span: Span::dummy(),
        };

        let info = ExportInfo::from_fn_def(&fn_def, "forward", "forward", &interner);
        assert_eq!(info.symbol_name, "forward");
        assert_eq!(info.raw_name, "forward");
        assert_eq!(info.params.len(), 1);
        assert_eq!(info.params[0].name, "x");
        match &info.params[0].ty {
            ExportTypeInfo::Tensor { shape, dtype, .. } => {
                assert_eq!(shape, &vec!["B".to_string(), "768".to_string()]);
                assert_eq!(*dtype, ExportDtype::F32);
            }
            _ => panic!("expected Tensor param"),
        }
        match &info.return_type {
            ExportTypeInfo::Tensor { shape, dtype, .. } => {
                assert_eq!(shape, &vec!["B".to_string(), "1000".to_string()]);
                assert_eq!(*dtype, ExportDtype::F32);
            }
            _ => panic!("expected Tensor return"),
        }
    }
}
