//! Param block declaration and ld.param helpers.
//!
//! - emit_param_block    — emits `.visible .entry name(\n    .param ...\n)`
//! - emit_ld_param_u64   — line-level helper, caller names dest register
//! - emit_ld_param_u32   — line-level helper (covers csha_active_heads, csha_d_model)
//! - emit_ld_param_f32   — line-level helper, caller names dest register

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamTy {
    U64,
    F32,
    U32,
}

impl ParamTy {
    fn as_str(self) -> &'static str {
        match self {
            ParamTy::U64 => ".u64",
            ParamTy::F32 => ".f32",
            ParamTy::U32 => ".u32",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Param {
    pub ty: ParamTy,
    pub name: &'static str,
}

/// Emit the `.visible .entry <entry_name> ( .param ..., .param ..., ... )` block.
/// The opening brace `{` is emitted by this helper.
pub fn emit_param_block(ptx: &mut String, entry_name: &str, params: &[Param]) {
    ptx.push_str(&format!(".visible .entry {} (\n", entry_name));
    for (i, p) in params.iter().enumerate() {
        let trailing = if i + 1 < params.len() { "," } else { "" };
        ptx.push_str(&format!("    .param {} {}{}\n", p.ty.as_str(), p.name, trailing));
    }
    ptx.push_str(")\n{\n");
}

/// Emit one `ld.param.u64 <dest_reg>, [<param_name>];` line.
pub fn emit_ld_param_u64(ptx: &mut String, dest_reg: &str, param_name: &str) {
    ptx.push_str(&format!("    ld.param.u64 {}, [{}];\n", dest_reg, param_name));
}

/// Emit one `ld.param.u32 <dest_reg>, [<param_name>];` line.
/// Covers FA's csha_active_heads / csha_d_model loads.
pub fn emit_ld_param_u32(ptx: &mut String, dest_reg: &str, param_name: &str) {
    ptx.push_str(&format!("    ld.param.u32 {}, [{}];\n", dest_reg, param_name));
}

/// Emit one `ld.param.f32 <dest_reg>, [<param_name>];` line.
pub fn emit_ld_param_f32(ptx: &mut String, dest_reg: &str, param_name: &str) {
    ptx.push_str(&format!("    ld.param.f32 {}, [{}];\n", dest_reg, param_name));
}
