//! FASE Deferred-mode emission for the train-block backward pass.
//!
//! Wired in from `stmt.rs` when `fase::plan()` returns `FaseMode::Deferred`.
//! Left deliberately thin — `stmt.rs` owns the outer micro-batch loop, the
//! parameter-list traversal, and the existing `accum_list` allocation; this
//! module only replaces:
//!
//!   1. the per-micro-batch accumulator update (was `accum += g`; now
//!      `m_partial += (1/N) * g` via the recipe from `fase_optimizer.rs`),
//!   2. the post-loop optimizer step (was "divide accum by N, run optimizer";
//!      now "run the fused per-parameter recipe, then zero m_partial").
//!
//! The buffer slot is the same allocation `stmt.rs` already makes — this
//! module does not allocate.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::fase::FasePlan;

/// Emit the Deferred-mode per-micro-batch accumulator update.
///
/// Free-function marker hook — the actual Cranelift IR emission is
/// `Compiler::fase_emit_accumulate`.
///
/// Returns `Ok(())` unconditionally; callers that only need to check whether
/// the stub is wired can call this.
pub fn emit_deferred_accumulate(_plan: &FasePlan) -> Result<(), String> {
    Ok(())
}

/// Emit the Deferred-mode fused final step (runs after the last micro-batch
/// backward): per-parameter optimizer update sourced from `m_partial`, then
/// `m_partial = 0`.
///
/// Not yet implemented — returns `Err` until Task 8 lands.
pub fn emit_deferred_final_step(_plan: &FasePlan) -> Result<(), String> {
    Err("stmt_fase::emit_deferred_final_step not yet implemented".into())
}

impl Compiler<'_> {
    /// Emit `m_partial += accum_scale * grad` for a single parameter.
    ///
    /// - `m_partial_ptr`: runtime pointer (i64) to the accumulator slot
    ///   (produced by `nsl_list_get(accum_list, i)`).
    /// - `grad_ptr`: runtime pointer (i64) to the just-computed gradient.
    /// - `accum_scale`: the recipe's `accum_scale` field (1.0/N).
    ///
    /// After this call, the caller should free `grad_ptr`.
    ///
    /// Case B (no axpy helper in nsl-runtime): scale via
    /// `nsl_tensor_mul_scalar`, then add in-place via
    /// `nsl_tensor_add_inplace`, then free the temporary.
    pub(crate) fn fase_emit_accumulate(
        &mut self,
        builder: &mut FunctionBuilder,
        m_partial_ptr: Value,
        grad_ptr: Value,
        accum_scale: f64,
    ) -> Result<(), crate::error::CodegenError> {
        // Step 1: scaled_grad = grad * accum_scale  (owned new tensor)
        let scale_val = builder.ins().f64const(accum_scale);
        // flags=0: do NOT relinquish `grad_ptr` — the caller owns it and will
        // free it after this call returns.
        let flags_zero = builder.ins().iconst(cl_types::I8, 0);
        let scaled_grad =
            self.compile_call_by_name(builder, "nsl_tensor_mul_scalar", &[grad_ptr, scale_val, flags_zero])?;

        // Step 2: m_partial += scaled_grad  (in-place, void)
        self.compile_call_by_name(builder, "nsl_tensor_add_inplace", &[m_partial_ptr, scaled_grad])?;

        // Step 3: free the temporary scaled_grad
        self.compile_call_by_name(builder, "nsl_tensor_free", &[scaled_grad])?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fase::{plan, FaseConfig, FaseOptimizer};

    #[test]
    fn accumulate_stub_now_returns_ok() {
        let p = plan(&FaseConfig {
            accumulation: 4,
            optimizer: FaseOptimizer::AdamW,
            grad_clip: None,
            allow_v_approx: true,
            ..Default::default()
        });
        assert!(emit_deferred_accumulate(&p).is_ok());
    }

    #[test]
    fn final_step_stub_still_returns_err_until_task_8() {
        let p = plan(&FaseConfig {
            accumulation: 4,
            optimizer: FaseOptimizer::AdamW,
            grad_clip: None,
            allow_v_approx: true,
            ..Default::default()
        });
        assert!(emit_deferred_final_step(&p).is_err());
    }
}
