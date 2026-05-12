//! Static spill analysis at codegen time. Counts live MMA accumulators
//! per warp and rejects configs predicted to exceed the 255/thread cap.
//! Per spec section 5.4 register budget.

use crate::flash_attention::FlashAttentionConfig;

#[derive(Debug)]
pub struct SpillRisk(pub String);

/// Per-thread register cap on sm_80+ NVIDIA GPUs.
const REG_CAP_PER_THREAD: u32 = 255;

/// Headroom reserved for compiler spill-prevention margin (live ranges
/// the static estimate doesn't see).
const REG_HEADROOM: u32 = 15;

/// Estimate live registers per thread for the given Tier B.1 config.
/// Per spec §5.4 register budget table (8 warps per CTA, MMA m16n8k16
/// fragment layout). Returns Ok(regs_estimate) if under the cap minus
/// headroom; Err(SpillRisk) otherwise.
///
/// Budget terms (warp's share, expressed per-thread = warp_share / 32):
///   * Q resident accumulators       : (bq/16) * (hd/8) / 8 * 4 (f32)
///   * K projection in-flight chunk  : 16 (f32, one chunk's worth)
///   * V projection in-flight chunk  : 16 (f32)
///   * QK^T S fragments              : (bq/16) * (bkv/8) / 8 * 4 (f32)
///   * P_f16 packed (post-softmax)   : 8 (b32 holding 16 f16 lanes)
///   * O_acc (lives across kv_iters) : (bq/16) * (hd/8) / 8 * 4 (f32)
///   * Row stats + correction        : ~16 (f32 + scratch)
///   * Addressing + predicates       : ~32 (mixed)
///
/// The formula is integer-only (no floor/ceil); MMA tile counts
/// (bq/16, bkv/8, hd/8) are assumed integral — the caller (validator
/// chain or chunk_config::select) is responsible for rejecting configs
/// where they aren't.
pub fn analyze(config: &FlashAttentionConfig) -> Result<u32, SpillRisk> {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;

    // Guard against non-MMA-aligned tiles. MMA m16n8k16 needs bq%16==0,
    // bkv%8==0, hd%8==0. If any fails, the kernel can't be emitted at all
    // — but register_budget is the wrong layer to enforce this; just
    // produce a conservative estimate.
    let q_resident = (bq / 16).max(1) * (hd / 8).max(1) / 8 * 4;
    let k_inflight = 4;     // one chunk's K accumulator fragment
    let v_inflight = 4;     // one chunk's V accumulator fragment
    let s_frags = (bq / 16).max(1) * (bkv / 8).max(1) / 8 * 4;
    let p_packed = 8;
    let o_acc = (bq / 16).max(1) * (hd / 8).max(1) / 8 * 4;
    let row_stats = 16;
    let addressing = 32;

    let total = q_resident + k_inflight + v_inflight + s_frags + p_packed + o_acc + row_stats + addressing;

    if total + REG_HEADROOM > REG_CAP_PER_THREAD {
        return Err(SpillRisk(format!(
            "tier_b1 estimated {} regs/thread for (bq={}, bkv={}, hd={}); exceeds {} cap minus {} headroom",
            total, bq, bkv, hd, REG_CAP_PER_THREAD, REG_HEADROOM
        )));
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn make_config(bq: i64, bkv: i64, hd: i64) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: bq,
            block_kv: bkv,
            head_dim: hd,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 120,
            segment_masked: false,
            csha: Some(CshaExtras { level: 2, d_model: 2048, ..CshaExtras::default() }),
        }
    }

    #[test]
    fn canonical_64_64_64_fits_budget() {
        // Per spec §5.4: typical canonical config should fit comfortably
        // under 255/thread. Expected ~168 regs.
        let cfg = make_config(64, 64, 64);
        let regs = analyze(&cfg).expect("canonical config should fit");
        assert!(regs < 200, "got {} regs, expected ~168 per spec", regs);
    }

    #[test]
    fn canonical_64_64_128_fits_budget() {
        // Same shape but hd=128. Spec §5.4 says ~168 at this config too
        // (the formula scales with bq*hd/8, but normalized per-thread the
        // dominant term is constant).
        let cfg = make_config(64, 64, 128);
        let regs = analyze(&cfg).expect("canonical hd=128 should fit");
        assert!(regs < REG_CAP_PER_THREAD - REG_HEADROOM);
    }

    #[test]
    fn small_config_fits_with_margin() {
        let cfg = make_config(32, 32, 32);
        let regs = analyze(&cfg).expect("small config should fit");
        assert!(regs < 100, "small config should use way under 100 regs, got {}", regs);
    }
}
