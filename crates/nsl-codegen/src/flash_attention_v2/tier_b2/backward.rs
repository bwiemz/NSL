//! Tier B.2 backward kernel emitter — Phase 1 stub.
//!
//! Phase 2 will replace this stub with the three-kernel implementation
//! (D pre-pass + dQ-kernel + dK/dV-kernel) per spec §3.
//!
//! Phase 1 contract: returns `Err(NotImplemented)` so the selector can
//! fall back to scalar v2 backward without panicking.

use crate::flash_attention::FlashAttentionConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackwardSynthError {
    /// Phase 1 placeholder. Phase 2 removes this variant when the
    /// emitter ships.
    NotImplemented,
}

impl std::fmt::Display for BackwardSynthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotImplemented => write!(
                f,
                "Tier B.2 backward emitter not yet implemented (Phase 1 stub); \
                 fall back to scalar v2 backward"
            ),
        }
    }
}

impl std::error::Error for BackwardSynthError {}

/// Phase 1 stub. Always returns `Err(NotImplemented)`.
///
/// Phase 2 replaces this with the real emitter that produces the
/// three-kernel PTX string per spec §3.
pub fn synthesize_tier_b2_backward(
    _config: &FlashAttentionConfig,
) -> Result<String, BackwardSynthError> {
    Err(BackwardSynthError::NotImplemented)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn canonical_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 80,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                ..Default::default()
            }),
        }
    }

    #[test]
    fn phase1_stub_returns_not_implemented() {
        let result = synthesize_tier_b2_backward(&canonical_cfg());
        assert_eq!(result, Err(BackwardSynthError::NotImplemented));
    }
}
