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
    // Sprint 3 cycle-2 (paper sec 3.2 asymmetric tiles): the historical
    // bkv == bq invariant was lifted in dkdv.rs by splitting the dual-use
    // %band_row_base register into axis-specific %band_row_base_q /
    // %band_row_base_kv (and similarly the idle-warp predicate). The
    // planner still pins bkv = bq today because no caller currently asks
    // for asymmetric tiles AND the per-hd SMEM ladder above pins one
    // (bq, chunk) pair per head_dim, so widening the predicate without a
    // separate kv ladder would not change observed dispatch. Once a
    // caller surfaces a genuine asymmetric demand (or a kv ladder lands),
    // the kernel is ready: dkdv synthesise no longer rejects bq != bkv,
    // and emit_warp_band_setup gates Q-axis work on bq/16 active warps
    // and KV-axis work on bkv/16 active warps independently.
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
    // Sprint 2 cycle-7 (§4.3 attention sinks v1): defense-in-depth refusal.
    // v1 is forward-only — the dQ/dKdV/proj-backward kernels do not
    // understand the persistent sink slab. Returning `false` here forces
    // the planner to pick BackwardTier::Scalar, which is ALSO sink-unaware
    // — but the scalar `synthesize_backward_with_tier_b` has its own
    // sink refusal that surfaces a clear error. This clause is the "should
    // not have been reached" defensive layer per cycle-5
    // `feedback_deferral_must_refuse`. Lift point: a future v2 backward
    // sprint that lands sink-aware kernels.
    if config.num_sink_tokens > 0 {
        return false;
    }
    // Phase 1.4b (pretraining plan): the dq/dkdv kernels silently ignore
    // `segment_ids_ptr`. Force a fallback to the scalar backward (which honors
    // the segment mask) rather than emit wrong grads on packed sequences.
    // Mirrors the synth-side refusal in `synthesize_tier_b2_backward`.
    if config.segment_masked {
        return false;
    }
    // Sprint 1 cycle-2: rope_q=true is now safe because forward and backward
    // share cos/sin via the CshaSavePointers channel. The forward call sites
    // (wengert_lower.rs and expr/advanced.rs) hoist the cos/sin Cranelift
    // Values handed to `nsl_flash_attention_csha_with_saves` into named locals
    // and stash them on the per-layer save record; the backward FFI call at
    // wengert_lower.rs:1958 reads `saves.cos`/`saves.sin` from the same record.
    // Today both are null → forward and backward both skip rotation via the
    // in-kernel null-guard (self-consistent, rope-effectively-off). When
    // future work threads non-null cos/sin into the forward, backward picks
    // them up automatically with no further edits — the H1 divergence risk
    // that gated rope_q before is now structurally impossible.
    csha.active_heads == 1  // exactly one active head; the 0="all" sentinel is rejected as ambiguous -> safe fallback to scalar
        && csha.d_model == hd
        && seq_len == block_q
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
    // Sprint 2 cycle-7 (§4.3 attention sinks v1): defense-in-depth refusal —
    // mirrors `tier_b2_hybrid_backward_eligible`. Compile-time predicates
    // gate the wengert-lowering branch that decides between the hybrid
    // 4-kernel runtime branch and the scalar fallback (Sprint 1 T1.3).
    // Forcing `false` here at `num_sink_tokens > 0` makes the wengert
    // lowering emit a trivial `iconst(0)` for the runtime-active flag, so
    // the runtime never even attempts to launch the hybrid kernels for a
    // sinks-enabled config. Lift point matches the runtime predicate.
    if config.num_sink_tokens > 0 {
        return false;
    }
    // Phase 1.4b (pretraining plan): mirror the runtime predicate — the dq/dkdv
    // kernels silently ignore `segment_ids_ptr`, so a segment-masked config must
    // fall back to the scalar backward (which honors the mask). Forcing `false`
    // here makes wengert lowering emit the scalar branch directly.
    if config.segment_masked {
        return false;
    }
    // Sprint 1 cycle-2: rope_q=true is now safe because forward and backward
    // share cos/sin via the CshaSavePointers channel. See the matching comment
    // in `tier_b2_hybrid_backward_eligible` for the full rationale.
    csha.active_heads == 1
        && csha.d_model == hd
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
            num_sink_tokens: 0,
            gpu_sm: sm,
            segment_masked: false,
            csha: Some(CshaExtras {
                level,
                ..Default::default()
            }),
            checkpoint: None,
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
    fn hybrid_ineligible_when_segment_masked() {
        // Phase 1.4b: a segment-masked config that is otherwise hybrid-eligible
        // must fall back to the scalar backward (which honors the mask).
        let mut c = cfg(64, 80, 2);
        c.csha = Some(CshaExtras { level: 2, d_model: 64, active_heads: 1, ..Default::default() });
        let seq = c.block_q as u32;
        assert!(tier_b2_hybrid_backward_eligible(&c, seq),
            "sanity: config must be eligible before enabling segment_masked");
        assert!(tier_b2_hybrid_backward_compile_time_eligible(&c),
            "sanity: compile-time predicate eligible before segment_masked");
        c.segment_masked = true;
        assert!(!tier_b2_hybrid_backward_eligible(&c, seq),
            "segment_masked must force scalar fallback");
        assert!(!tier_b2_hybrid_backward_compile_time_eligible(&c),
            "compile-time predicate must also refuse segment_masked");
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
            num_sink_tokens: 0,
            gpu_sm: sm,
            segment_masked: false,
            csha: Some(CshaExtras {
                level,
                d_model,
                active_heads: heads,
                ..Default::default()
            }),
            checkpoint: None,
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

    /// Sprint 1 cycle-2: rope_q=true is now ACCEPTED. The CshaSavePointers
    /// cos/sin channel structurally guarantees forward and backward see
    /// identical Cranelift Values for cos/sin (today both null → both skip
    /// rotation via the in-kernel null-guard, self-consistent). When future
    /// work threads non-null cos/sin into the forward call site, backward
    /// picks them up automatically — the H1 divergence risk is closed.
    #[test]
    fn compile_time_accepts_rope_q_via_saves_channel() {
        let c = hybrid_cfg(64, 1, 64, true, 80, 2);
        assert!(
            tier_b2_hybrid_backward_compile_time_eligible(&c),
            "rope_q=true must route through the hybrid now that the saves channel guarantees forward/backward cos/sin agreement"
        );
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

    // -----------------------------------------------------------------
    // Sprint 2 cycle-7 (§4.3 attention sinks v1): defense-in-depth refusal.
    // The compile-time + runtime hybrid eligibility predicates must
    // return `false` whenever `num_sink_tokens > 0` — even at the
    // otherwise-eligible smoke config. v1 is forward-only; the dQ/dKdV
    // kernels do not understand the persistent sink slab.
    // -----------------------------------------------------------------

    /// Compile-time predicate must reject the otherwise-eligible smoke
    /// config when `num_sink_tokens > 0` (defense-in-depth: the forward
    /// front door also refuses this, but the predicate is consulted at
    /// wengert lowering and would otherwise emit `iconst(1)` for the
    /// runtime-active flag — silently launching sink-unaware backward).
    #[test]
    fn compile_time_rejects_sinks_enabled_at_smoke_config() {
        let mut c = hybrid_cfg(64, 1, 64, false, 80, 2);
        c.num_sink_tokens = 4;
        assert!(
            !tier_b2_hybrid_backward_compile_time_eligible(&c),
            "compile-time hybrid predicate must refuse sinks-enabled config"
        );
    }

    /// Runtime predicate must reject the otherwise-eligible smoke config
    /// when `num_sink_tokens > 0`. Tested at `seq_len = block_q` so we
    /// know the only reason for refusal is the sinks axis.
    #[test]
    fn runtime_rejects_sinks_enabled_at_smoke_config() {
        let mut c = hybrid_cfg(64, 1, 64, false, 80, 2);
        c.num_sink_tokens = 4;
        assert!(
            !tier_b2_hybrid_backward_eligible(&c, c.block_q as u32),
            "runtime hybrid predicate must refuse sinks-enabled config"
        );
    }

    /// Same config with `num_sink_tokens = 0` must remain ACCEPTED to
    /// prove the refusal axis is sinks, not some other regression.
    /// Pins the per-axis exercise principle: a test that "rejects the
    /// config" without first proving the baseline accepts could be
    /// masking a deeper bug.
    #[test]
    fn both_predicates_accept_same_config_at_zero_sinks() {
        let mut c = hybrid_cfg(64, 1, 64, false, 80, 2);
        c.num_sink_tokens = 0;
        assert!(
            tier_b2_hybrid_backward_compile_time_eligible(&c),
            "baseline compile-time predicate must accept the zero-sinks config"
        );
        assert!(
            tier_b2_hybrid_backward_eligible(&c, c.block_q as u32),
            "baseline runtime predicate must accept the zero-sinks config"
        );
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
            // Phase 1.4b: segment_masked must preserve the decomposition invariant
            // (both predicates false → full == compile_time && seq==block_q holds).
            ("segmasked",     { let mut c = hybrid_cfg(64, 1, 64, false, 80, 2); c.segment_masked = true; c }),
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
