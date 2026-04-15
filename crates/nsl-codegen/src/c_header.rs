//! M62 `@export` C header emission + type model.
//!
//! `ExportInfo` tracks each `@export`-decorated function so the CLI
//! can emit a matching C header after shared-lib codegen completes.
//! This module also owns the C-type lowering used by the header
//! emitter (Task 5) and the C-wrapper emission (Task 4).

use serde::{Deserialize, Serialize};

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
}
