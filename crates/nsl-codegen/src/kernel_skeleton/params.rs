//! Param block declaration and ld.param helpers.
//!
//! - emit_param_block    — emits `.visible .entry name(\n    .param ...\n)`
//! - emit_ld_param_u64   — line-level helper, caller names dest register
//! - emit_ld_param_u32   — line-level helper (covers csha_active_heads, csha_d_model)
//! - emit_ld_param_f32   — line-level helper, caller names dest register
