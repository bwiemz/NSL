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

use crate::fase::FasePlan;

/// Emit the Deferred-mode per-micro-batch accumulator update.
///
/// Caller contract: `m_partial_buf` and `grad_buf` are runtime pointers to
/// the parameter's accumulator slot and the just-computed gradient buffer,
/// respectively.  After this call, `grad_buf` may be freed by the caller.
///
/// Not yet implemented — returns `Err` until Task 7 lands.
pub fn emit_deferred_accumulate(_plan: &FasePlan) -> Result<(), String> {
    Err("stmt_fase::emit_deferred_accumulate not yet implemented".into())
}

/// Emit the Deferred-mode fused final step (runs after the last micro-batch
/// backward): per-parameter optimizer update sourced from `m_partial`, then
/// `m_partial = 0`.
///
/// Not yet implemented — returns `Err` until Task 8 lands.
pub fn emit_deferred_final_step(_plan: &FasePlan) -> Result<(), String> {
    Err("stmt_fase::emit_deferred_final_step not yet implemented".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fase::{plan, FaseConfig, FaseOptimizer};

    #[test]
    fn stubs_return_err_until_implemented() {
        let p = plan(&FaseConfig {
            accumulation: 4,
            optimizer: FaseOptimizer::AdamW,
            grad_clip: None,
            allow_v_approx: true,
            ..Default::default()
        });
        assert!(emit_deferred_accumulate(&p).is_err());
        assert!(emit_deferred_final_step(&p).is_err());
    }
}
