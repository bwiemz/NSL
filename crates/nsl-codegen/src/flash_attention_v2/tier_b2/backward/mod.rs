//! Tier B.2 backward kernel emitter — Phase 2 implementation.
//!
//! Phase 2 implements the three-kernel plan:
//!   - D pre-pass (`d_prepass` module, Task 6)
//!   - dQ-kernel (Tasks 8-16)
//!   - dK/dV-kernel (Phase 3)
//!
//! Phase 2 contract: `synthesize_tier_b2_backward` concatenates the PTX
//! output of `synthesize_d_prepass` + `synthesize_dq_kernel`.  Phase 3
//! will extend with dK/dV-kernel synthesis.

pub mod d_prepass;
pub mod dq;
pub mod hbm_addr;

use crate::flash_attention::FlashAttentionConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackwardSynthError {
    /// Phase 1 placeholder. Kept for the `NotImplemented` variant so that
    /// downstream match arms compile; unreachable in Phase 2+.
    NotImplemented,
    /// head_dim must be divisible by 32 for the warp-shfl reduction.
    UnsupportedHeadDim(u32),
}

impl std::fmt::Display for BackwardSynthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotImplemented => write!(
                f,
                "Tier B.2 backward emitter path not yet implemented"
            ),
            Self::UnsupportedHeadDim(hd) => write!(
                f,
                "Tier B.2 backward requires head_dim divisible by 32, got {}",
                hd
            ),
        }
    }
}

impl std::error::Error for BackwardSynthError {}

/// Phase 2 emitter: synthesize D pre-pass + dQ-kernel PTX.
///
/// Phase 3 will extend with dK/dV-kernel synthesis (separate kernel,
/// concatenated into the output).
pub fn synthesize_tier_b2_backward(
    config: &FlashAttentionConfig,
) -> Result<String, BackwardSynthError> {
    let d_prepass_ptx = d_prepass::synthesize_d_prepass(config)?;
    let dq_ptx = dq::synthesize_dq_kernel(config)?;
    Ok(format!("{}\n\n{}", d_prepass_ptx, dq_ptx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn canonical_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64, block_kv: 64, head_dim: 128,
            causal: true, paged: false,
            rope_q: false, rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false,
            gpu_sm: 80, segment_masked: false,
            csha: Some(CshaExtras { level: 2, ..Default::default() }),
        }
    }

    #[test]
    fn phase2_synthesize_concatenates_d_prepass_plus_dq_kernel() {
        let result = synthesize_tier_b2_backward(&canonical_cfg());
        let ptx = result.expect("Phase 2 emitter no longer returns NotImplemented");
        // Must contain both kernels' entry points.
        assert!(ptx.contains("tier_b2_d_prepass"),
            "expected D pre-pass entry, got:\n{ptx}");
        assert!(ptx.contains("tier_b2_dq_kernel"),
            "expected dQ kernel entry, got:\n{ptx}");
    }

    #[test]
    fn phase2_synthesize_rejects_invalid_head_dim() {
        let mut cfg = canonical_cfg();
        cfg.head_dim = 48;  // not divisible by 32
        let result = synthesize_tier_b2_backward(&cfg);
        assert_eq!(result, Err(BackwardSynthError::UnsupportedHeadDim(48)));
    }
}
