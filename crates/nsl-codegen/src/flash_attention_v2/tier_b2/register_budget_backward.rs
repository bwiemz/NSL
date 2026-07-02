//! Register-budget model for Tier B.2 backward sub-kernels.
//!
//! Extends `register_budget.rs` (forward, single accumulator) with multi-accumulator
//! lifetime overlap modeling and per-sub-kernel awareness.
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §3.2 + §5.2

use crate::flash_attention::FlashAttentionConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardKernel {
    DPrePass,
    DQ,
    DKDV,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BqDowngrade {
    /// Downgrade bq from 64 to 32. Triggers at hd=128 (SMEM pressure) and hd=256 (reg pressure).
    From64To32,
}

/// Conservative upper-bound count of f32-equivalent registers per lane for a
/// Tier B.2 backward sub-kernel at the given config.
///
/// The precise count is confirmed post-emission via `ptxas --ptxas-options=-v`.
/// The budget is a planner-side gate that catches obvious overruns before PTX compilation.
pub fn count_registers_backward(
    config: &FlashAttentionConfig,
    kernel: BackwardKernel,
) -> u32 {
    // `accumulator_fragments` = number of m16n8k16 C-fragments per lane for the
    // accumulator tile. Each f32-accumulator fragment occupies 4 f32 registers
    // per lane (m16n8 sub-tile of 16x8 elements distributed as 4 per lane across
    // the 32-lane warp).
    //
    // For DQ at hd=128, bq=64, 4 warps splitting hd-direction into 32-col strips:
    //   accumulator_fragments = head_dim / 32 = 4
    //   total dQ-acc registers per lane = 4 fragments × 4 f32/lane = 16 f32
    //
    // Matches the Approach A″ deliberation's "16 f32/lane at hd=128" estimate.
    let accumulator_fragments = (config.head_dim / 32) as u32;
    let accumulators = match kernel {
        BackwardKernel::DPrePass => 0,
        BackwardKernel::DQ => accumulator_fragments,
        BackwardKernel::DKDV => 2 * accumulator_fragments,
    };

    let mma_overhead = 16; // fragment + scratch lifetime overlap
    let softmax = 5;       // row_max, row_sum, correction, etc.
    let scratch = 12;      // indexing + loop counters + RoPE/segment scratch
    let d_o_indexing = 4;  // dO SMEM addressing lives across inner loop
    let csha_extras = if config.csha.is_some() { 4 } else { 0 };

    accumulators + mma_overhead + softmax + scratch + d_o_indexing + csha_extras
}

/// SM_75 register cap. Oldest supported SM generation. Informational at v1 (no
/// realistic config fails).
pub const SM75_REGISTER_CAP: u32 = 255;

/// Predict whether the planner should downgrade `bq` before emitting PTX.
///
/// Per Phase 2 spec §5.2 (extends Approach A″ from register-pressure-only to
/// SMEM-pressure as well):
///
/// - hd=128 + bq=64: SMEM-pressure trigger. Adding the Path A col-major K
///   re-stage band at bq=64 hd=128 totals exactly 99 KB (no headroom). Downgrade
///   bq → 32 to restore ~16 KB headroom.
/// - hd=256 + bq=64: register-pressure trigger (Approach A″). dV+dK at 2 ×
///   (hd/32) = 16 f32 regs/lane combined; bq=64 layout pushes register count
///   higher than budget allows.
///
/// Returns `Some(BqDowngrade::From64To32)` if a downgrade is needed; `None` otherwise.
pub fn predict_fallback(config: &FlashAttentionConfig) -> Option<BqDowngrade> {
    // SMEM-pressure fallback at hd=128 with bq=64 (per Phase 2 spec §5.2):
    // Adding the col-major K re-stage band at bq=64 hd=128 totals exactly 99 KB —
    // no headroom. Downgrade bq → 32 to restore 16 KB headroom.
    if config.head_dim == 128 && config.block_q == 64 {
        return Some(BqDowngrade::From64To32);
    }
    // Register-pressure fallback at hd=256 with bq=64 (per Approach A″):
    if config.head_dim >= 256 && config.block_q == 64 {
        return Some(BqDowngrade::From64To32);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn canonical_hd128_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64, block_kv: 64, head_dim: 128,
            causal: true, paged: false,
            rope_q: false, rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0,
            gpu_sm: 80, segment_masked: false,
            csha: Some(CshaExtras { level: 2, ..Default::default() }),
            checkpoint: None,
        }
    }

    #[test]
    fn d_prepass_uses_zero_accumulators() {
        let cfg = canonical_hd128_cfg();
        let count = count_registers_backward(&cfg, BackwardKernel::DPrePass);
        assert_eq!(count, 16 + 5 + 12 + 4 + 4);  // 41 regs
    }

    #[test]
    fn dq_uses_accumulator_fragments_for_head_dim_128() {
        let cfg = canonical_hd128_cfg();
        let count = count_registers_backward(&cfg, BackwardKernel::DQ);
        assert_eq!(count, 4 + 16 + 5 + 12 + 4 + 4);  // 45 regs
    }

    #[test]
    fn dkdv_uses_double_the_dq_accumulator_count() {
        let cfg = canonical_hd128_cfg();
        let count = count_registers_backward(&cfg, BackwardKernel::DKDV);
        assert_eq!(count, 8 + 16 + 5 + 12 + 4 + 4);  // 49 regs
    }

    #[test]
    fn dq_count_under_sm75_cap_at_all_canonical_configs() {
        for &hd in &[64u32, 128, 256] {
            let mut cfg = canonical_hd128_cfg();
            cfg.head_dim = hd as i64;
            cfg.block_q = if hd == 256 { 32 } else { 64 };
            cfg.block_kv = cfg.block_q;
            let count = count_registers_backward(&cfg, BackwardKernel::DQ);
            assert!(count < SM75_REGISTER_CAP,
                "DQ at hd={} uses {} regs, exceeds SM75 cap", hd, count);
        }
    }

    // === predict_fallback tests (spec §5.2 + Approach A″) ===

    #[test]
    fn predict_fallback_downgrades_hd128_bq64_for_smem_pressure() {
        let mut cfg = canonical_hd128_cfg();
        cfg.head_dim = 128;
        cfg.block_q = 64;
        assert_eq!(predict_fallback(&cfg), Some(BqDowngrade::From64To32),
            "hd=128 bq=64 must trigger SMEM-pressure fallback per spec §5.2");
    }

    #[test]
    fn predict_fallback_downgrades_hd256_bq64_for_register_pressure() {
        let mut cfg = canonical_hd128_cfg();
        cfg.head_dim = 256;
        cfg.block_q = 64;
        assert_eq!(predict_fallback(&cfg), Some(BqDowngrade::From64To32),
            "hd=256 bq=64 must trigger register-pressure fallback per Approach A\"");
    }

    #[test]
    fn predict_fallback_passes_hd64_canonical() {
        let mut cfg = canonical_hd128_cfg();
        cfg.head_dim = 64;
        cfg.block_q = 64;
        assert_eq!(predict_fallback(&cfg), None,
            "hd=64 bq=64 fits without fallback");
    }

    #[test]
    fn predict_fallback_no_op_at_already_downgraded_bq32() {
        let mut cfg = canonical_hd128_cfg();
        cfg.head_dim = 128;
        cfg.block_q = 32;
        assert_eq!(predict_fallback(&cfg), None,
            "bq=32 is already at the fallback target; no further downgrade");
    }
}
