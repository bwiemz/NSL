//! Tier B.2 dispatch predicate + SMEM budget check.
//!
//! Implements the per-hd bq+chunk schedule from spec §6.4 and the
//! `predict_fallback` planner helper from spec §7.1.

use crate::flash_attention::FlashAttentionConfig;

/// Which backward kernel family the planner picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardTier {
    /// Scalar v2 backward (current default; lower throughput).
    Scalar,
    /// Tier B.2 MMA backward (Phase 2+ implementation).
    TierB2 { bq: u32, bkv: u32, chunk: u32 },
}

/// Reasons the planner rejected Tier B.2 dispatch for a given config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchReject {
    /// `csha.level < 2` — Tier B.2 only applies to Level 2 fusion.
    LevelTooLow,
    /// `gpu_sm < 80` — Tier B.2 requires Ampere or newer.
    SmTooOld,
    /// Config has no CSHA extras at all.
    NoCsha,
    /// SMEM ladder (§6.4) has no row matching this hd.
    UnsupportedHeadDim(u32),
    /// SMEM budget exceeded even at the smallest bq the ladder allows.
    SmemOverBudget { needed: u32, budget: u32 },
}

/// Per-hd bq + chunk schedule from spec §6.4.
///
/// Returns `None` if `head_dim` isn't in the supported set.
fn ladder_row(head_dim: u32) -> Option<(u32, u32)> {
    match head_dim {
        64  => Some((64, 4)),
        128 => Some((64, 4)),
        256 => Some((32, 4)),
        _ => None,
    }
}

/// SMEM bytes Tier B.2 backward (per sub-kernel) needs at a given config.
/// Matches the formula in spec §6.3 / §6.4.
pub fn tier_b2_smem_bytes(bq: u32, bkv: u32, hd: u32, chunk: u32) -> u32 {
    let fixed = bq * hd * 2   // Q
              + bkv * hd * 2  // K
              + bkv * hd * 2  // V
              + bq * hd * 2   // dO
              + bq * bkv * 4; // dS
    let chunk_staging = chunk * hd * 2 * 2   // Wk_chunk + Wv_chunk
                      + bq * chunk * 2       // x_q_chunk
                      + bkv * chunk * 2;     // x_kv_chunk
    fixed + chunk_staging
}

/// Planner predicate: can Tier B.2 dispatch handle this config?
///
/// Returns `Ok(BackwardTier::TierB2 {...})` with the planner-pinned bq/chunk
/// on success, `Err(DispatchReject)` with the rejection reason otherwise.
pub fn tier_b2_can_dispatch(
    config: &FlashAttentionConfig,
) -> Result<BackwardTier, DispatchReject> {
    use crate::flash_attention_v2::smem_layout::SMEM_DYNAMIC_BUDGET_BYTES;

    let Some(csha) = config.csha.as_ref() else {
        return Err(DispatchReject::NoCsha);
    };
    if csha.level < 2 {
        return Err(DispatchReject::LevelTooLow);
    }
    if config.gpu_sm < 80 {
        return Err(DispatchReject::SmTooOld);
    }
    let hd = config.head_dim as u32;
    let Some((bq, chunk)) = ladder_row(hd) else {
        return Err(DispatchReject::UnsupportedHeadDim(hd));
    };
    let bkv = bq;
    let needed = tier_b2_smem_bytes(bq, bkv, hd, chunk);
    if needed > SMEM_DYNAMIC_BUDGET_BYTES {
        return Err(DispatchReject::SmemOverBudget { needed, budget: SMEM_DYNAMIC_BUDGET_BYTES });
    }
    Ok(BackwardTier::TierB2 { bq, bkv, chunk })
}

/// Is the FULL hybrid backward (Tier B.2 dQ/dK/dV + scalar projection) VALIDATED for this config?
/// Bounded by the reused scalar projection emitters' smoke scope (heads==1, d_model==head_dim,
/// single Q tile) AND Tier B.2 dispatch eligibility. `tier_b2_can_dispatch` is intentionally NOT
/// modified -- the constraint is the hybrid's, not the Tier-B.2 kernels'. Widens when the
/// projection-extension follow-on lands. `batch` is a launch dim, enforced by the parity test, not here.
pub fn tier_b2_hybrid_backward_eligible(config: &FlashAttentionConfig, seq_len: u32) -> bool {
    if tier_b2_can_dispatch(config).is_err() {
        return false;
    }
    let Some(csha) = config.csha.as_ref() else { return false; };
    let hd = config.head_dim as u32;
    let block_q = config.block_q as u32;
    csha.active_heads.max(1) == 1
        && csha.d_model == hd
        && seq_len == block_q
        && !config.rope_q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn cfg(hd: i64, sm: u32, level: u8) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: hd,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: sm,
            segment_masked: false,
            csha: Some(CshaExtras {
                level,
                ..Default::default()
            }),
        }
    }

    #[test]
    fn dispatches_at_canonical_hd128_sm80_level2() {
        let result = tier_b2_can_dispatch(&cfg(128, 80, 2));
        assert_eq!(
            result,
            Ok(BackwardTier::TierB2 { bq: 64, bkv: 64, chunk: 4 })
        );
    }

    #[test]
    fn dispatches_at_hd64_sm80_level2() {
        let result = tier_b2_can_dispatch(&cfg(64, 80, 2));
        assert_eq!(
            result,
            Ok(BackwardTier::TierB2 { bq: 64, bkv: 64, chunk: 4 })
        );
    }

    #[test]
    fn rejects_sm75_even_at_level2() {
        let result = tier_b2_can_dispatch(&cfg(128, 75, 2));
        assert_eq!(result, Err(DispatchReject::SmTooOld));
    }

    #[test]
    fn rejects_level_below_2() {
        let result = tier_b2_can_dispatch(&cfg(128, 80, 1));
        assert_eq!(result, Err(DispatchReject::LevelTooLow));
    }

    #[test]
    fn downgrades_bq_to_32_at_hd256() {
        let result = tier_b2_can_dispatch(&cfg(256, 80, 2));
        assert_eq!(
            result,
            Ok(BackwardTier::TierB2 { bq: 32, bkv: 32, chunk: 4 })
        );
    }

    #[test]
    fn rejects_unsupported_head_dim() {
        let result = tier_b2_can_dispatch(&cfg(96, 80, 2));
        assert_eq!(result, Err(DispatchReject::UnsupportedHeadDim(96)));
    }

    #[test]
    fn smem_math_matches_spec_canonical() {
        // Spec §6.2 canonical: 83 KB at hd=128, bq=bkv=64, chunk=4.
        let bytes = tier_b2_smem_bytes(64, 64, 128, 4);
        // Within 1 KB of spec's pinned 83 KB.
        assert!(
            bytes >= 82 * 1024 && bytes <= 84 * 1024,
            "smem_bytes={bytes}, expected ~83 KB"
        );
    }
}
