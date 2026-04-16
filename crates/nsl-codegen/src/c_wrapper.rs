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
