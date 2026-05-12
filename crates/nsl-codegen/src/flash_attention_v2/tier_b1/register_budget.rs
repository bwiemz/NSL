//! Static spill analysis at codegen time. Counts live MMA accumulators
//! per warp and rejects configs predicted to exceed the 255/thread cap.
//! Per spec section 5.4 register budget. Real implementation in B1.3.

use crate::flash_attention::FlashAttentionConfig;

#[derive(Debug)]
pub struct SpillRisk(pub String);

/// Returns estimated regs/thread for the given config, or SpillRisk if
/// the estimate exceeds 240 (255 cap minus headroom).
///
/// Stub for B1.2: always returns Ok(0). B1.3 implements the real budget walk.
pub fn analyze(config: &FlashAttentionConfig) -> Result<u32, SpillRisk> {
    let _ = config;
    Ok(0)
}
