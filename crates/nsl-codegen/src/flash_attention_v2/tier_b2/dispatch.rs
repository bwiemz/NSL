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
///
/// Sprint 9: hd=32 added at (bq=64, chunk=4). Matches `tier_b2_effective_bq`'s
/// no-fallback path for hd in {32, 64} (smem_layout.rs:1037), and SMEM usage at
/// (bq=bkv=64, hd=32, chunk=4) is ~33.5 KB, well under the 99 KB dynamic budget.
/// The dQ-kernel tests at `dq.rs::synthesize_dq_kernel_*` and the dQ parity sweep
/// at `tier_b2_dq_kernel_cpu_reference::tier_b2_dq_sweep_cpu_naive_forward`
/// already exercise hd=32 with bq=64.
fn ladder_row(head_dim: u32) -> Option<(u32, u32)> {
    match head_dim {
        32  => Some((64, 4)),
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
    // Sprint 9 Part B: bkv == bq is a HARD kernel invariant of the dKdV kernel
    // (dkdv.rs:40-54). The single `%band_row_base = warp_id*16` register is
    // dual-used as a q-row base (inner-loop S/dP MMA + stats load + P/dS col
    // scatter) AND a kv-row base (outer-loop dV/dK accumulator MMA + HBM
    // finalize). Asymmetric tiles would silently corrupt dK/dV. Lifting this
    // requires splitting %band_row_base into separate q/kv warp-band registers,
    // doubling warp setup, and re-auditing every q-row vs kv-row use site in
    // dkdv.rs (lines ~208, 290, 297, 303, 802-814, 885, 892, 909, finalize).
    // That's a substantial kernel rewrite; deferred. Until then the planner
    // pins bkv = bq, matching the precondition the dKdV kernel enforces at
    // synth time via tier_b2_effective_bq/bkv parity.
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
///
/// `active_heads` must be **exactly 1** (`active_heads == 1`). The `0` sentinel meaning "all heads"
/// is rejected as ambiguous — a many-head config routed through the single-head hybrid would produce
/// colliding projection gradients. Over-routing is unsafe; under-routing falls back safely to scalar.
pub fn tier_b2_hybrid_backward_eligible(config: &FlashAttentionConfig, seq_len: u32) -> bool {
    if tier_b2_can_dispatch(config).is_err() {
        return false;
    }
    let Some(csha) = config.csha.as_ref() else { return false; };
    let hd = config.head_dim as u32;
    let block_q = config.block_q as u32;
    csha.active_heads == 1  // exactly one active head; the 0="all" sentinel is rejected as ambiguous -> safe fallback to scalar
        && csha.d_model == hd
        && seq_len == block_q
        && !config.rope_q
}

/// Compile-time subset of [`tier_b2_hybrid_backward_eligible`]: all checks that
/// depend only on the `FlashAttentionConfig` (NOT on `seq_len`, which is a
/// runtime tensor shape dim).
///
/// Invariant (unit-tested below in `compile_time_plus_seq_eq_block_q_matches_full_predicate`):
///
/// ```text
/// tier_b2_hybrid_backward_eligible(config, seq_len)
///   == tier_b2_hybrid_backward_compile_time_eligible(config)
///      && seq_len == config.block_q as u32
/// ```
///
/// The wengert lowering call site uses this to decide whether to emit a
/// trivial `iconst(0)` (config-ineligible) vs. a runtime `icmp seq_len, block_q`
/// (config-eligible — the hybrid 4-kernel branch fires iff the runtime
/// tile dim matches). Sprint 1 T1.3.
pub fn tier_b2_hybrid_backward_compile_time_eligible(config: &FlashAttentionConfig) -> bool {
    if tier_b2_can_dispatch(config).is_err() {
        return false;
    }
    let Some(csha) = config.csha.as_ref() else { return false; };
    let hd = config.head_dim as u32;
    csha.active_heads == 1
        && csha.d_model == hd
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

    /// Sprint 9: hd=32 added to the SMEM ladder. The ladder pins bq=64; the
    /// dispatch helper mirrors bkv=bq (preserving the dKdV kernel's symmetric
    /// warp-band precondition at smem_layout.rs:1037 + dkdv.rs:40-54).
    #[test]
    fn dispatches_at_hd32_sm80_level2() {
        let result = tier_b2_can_dispatch(&cfg(32, 80, 2));
        assert_eq!(
            result,
            Ok(BackwardTier::TierB2 { bq: 64, bkv: 64, chunk: 4 })
        );
    }

    /// Sprint 9: the SMEM budget at hd=32, bq=bkv=64, chunk=4 is well within
    /// the 99 KB dynamic budget. Anchored numeric check guards against future
    /// silent layout regressions blowing the ladder past the budget.
    #[test]
    fn smem_math_at_hd32_fits_budget() {
        use crate::flash_attention_v2::smem_layout::SMEM_DYNAMIC_BUDGET_BYTES;
        let bytes = tier_b2_smem_bytes(64, 64, 32, 4);
        // Expected ~33.5 KB: 4*(64*32*2) + 64*64*4 + 4*32*2*2 + 64*4*2 + 64*4*2
        //                  = 16384       + 16384     + 512        + 512    + 512
        //                  = 34304 bytes.
        assert_eq!(bytes, 34304, "SMEM math drifted from spec-computed hd=32 size");
        assert!(
            bytes < SMEM_DYNAMIC_BUDGET_BYTES,
            "hd=32 ladder ({bytes} B) must fit within dynamic budget ({SMEM_DYNAMIC_BUDGET_BYTES} B)"
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

    // --- Sprint 1 T1.3: compile-time-only eligibility predicate ----------

    fn hybrid_cfg(hd: i64, heads: u32, d_model: u32, rope: bool, sm: u32, level: u8)
        -> FlashAttentionConfig
    {
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: hd,
            causal: true,
            paged: false,
            rope_q: rope,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: sm,
            segment_masked: false,
            csha: Some(CshaExtras {
                level,
                d_model,
                active_heads: heads,
                ..Default::default()
            }),
        }
    }

    #[test]
    fn compile_time_eligible_at_smoke_intersection() {
        // hd=64, heads=1, d_model=64, rope=false, sm=80, level=2.
        let c = hybrid_cfg(64, 1, 64, false, 80, 2);
        assert!(tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    #[test]
    fn compile_time_eligible_at_hd128() {
        let c = hybrid_cfg(128, 1, 128, false, 80, 2);
        assert!(tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    /// Sprint 9: hd=32 was previously rejected (UnsupportedHeadDim) by the SMEM
    /// ladder. After adding the hd=32 row, the hybrid backward eligibility
    /// follows the same predicate stack — must accept.
    #[test]
    fn compile_time_eligible_at_hd32() {
        let c = hybrid_cfg(32, 1, 32, false, 80, 2);
        assert!(tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    #[test]
    fn compile_time_rejects_heads_gt_1() {
        let c = hybrid_cfg(64, 2, 64, false, 80, 2);
        assert!(!tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    #[test]
    fn compile_time_rejects_active_heads_zero() {
        // 0 = "all heads" sentinel — ambiguous; reject for safety.
        let c = hybrid_cfg(64, 0, 64, false, 80, 2);
        assert!(!tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    #[test]
    fn compile_time_rejects_d_model_ne_head_dim() {
        let c = hybrid_cfg(64, 1, 128, false, 80, 2);
        assert!(!tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    #[test]
    fn compile_time_rejects_rope_q() {
        let c = hybrid_cfg(64, 1, 64, true, 80, 2);
        assert!(!tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    #[test]
    fn compile_time_rejects_sm_too_old() {
        // gpu_sm=75 fails tier_b2_can_dispatch (Ampere required).
        let c = hybrid_cfg(64, 1, 64, false, 75, 2);
        assert!(!tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    #[test]
    fn compile_time_rejects_level_below_2() {
        let c = hybrid_cfg(64, 1, 64, false, 80, 1);
        assert!(!tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    #[test]
    fn compile_time_rejects_unsupported_head_dim() {
        // hd=96 isn't in the per-hd ladder.
        let c = hybrid_cfg(96, 1, 96, false, 80, 2);
        assert!(!tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    #[test]
    fn compile_time_rejects_no_csha() {
        let mut c = hybrid_cfg(64, 1, 64, false, 80, 2);
        c.csha = None;
        assert!(!tier_b2_hybrid_backward_compile_time_eligible(&c));
    }

    /// Headline invariant the wengert lowering relies on:
    ///     full_eligible(c, seq) == compile_time_eligible(c) && seq == block_q
    #[test]
    fn compile_time_plus_seq_eq_block_q_matches_full_predicate() {
        // Cover both eligible and ineligible configs, with seq_len == block_q
        // and seq_len != block_q, and verify the decomposition holds.
        let configs = [
            ("smoke",         hybrid_cfg(64,  1,  64, false, 80, 2)),
            ("hd32",          hybrid_cfg(32,  1,  32, false, 80, 2)),
            ("hd128",         hybrid_cfg(128, 1, 128, false, 80, 2)),
            ("heads2",        hybrid_cfg(64,  2,  64, false, 80, 2)),
            ("active0",       hybrid_cfg(64,  0,  64, false, 80, 2)),
            ("rope",          hybrid_cfg(64,  1,  64, true,  80, 2)),
            ("dm_mismatch",   hybrid_cfg(64,  1, 128, false, 80, 2)),
            ("sm75",          hybrid_cfg(64,  1,  64, false, 75, 2)),
            ("level1",        hybrid_cfg(64,  1,  64, false, 80, 1)),
            ("hd96",          hybrid_cfg(96,  1,  96, false, 80, 2)),
        ];
        for (label, c) in &configs {
            for seq in [32u32, 64, 128] {
                let full = tier_b2_hybrid_backward_eligible(c, seq);
                let ct = tier_b2_hybrid_backward_compile_time_eligible(c);
                let expected = ct && seq == c.block_q as u32;
                assert_eq!(
                    full, expected,
                    "decomposition broken for cfg='{label}' seq={seq}: \
                     full={full}, compile_time={ct}, block_q={}",
                    c.block_q
                );
            }
        }
    }
}
