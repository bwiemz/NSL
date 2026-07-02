//! PCA Per-Document CTA planner (G2 Strategy 3 v1).
//!
//! Provides the [`admit`] helper that decides whether a given
//! (`PcaDetection`, `FlashAttentionConfig`, `DatasetPackingConfig`) triplet
//! is suitable for the per-CTA kernel path.  Rejection falls back to the
//! existing Tier-A segment-masked kernel.
//!
//! v1 limitations (documented in the design spec):
//!   - Causal attention only (`fa_config.causal == true` required for
//!     both forward and backward).
//!   - No CSHA fused projections (forward and backward both gate this off).
//!   - `max_doc_len_est <= fa_config.block_q` (one CTA per doc; docs
//!     exceeding the tile size fall back to Tier A).
//!   - `num_docs_est <= MAX_NUM_DOCS` (matches the pca_rope SMEM bound).
//!   - Explicitly gated behind `enable_per_doc_cta=true` in the planner
//!     config (default OFF).
//!   - Backward additionally requires `block_kv == 32` (the standard FA-2
//!     v2 backward's `ds_compute::emit` hard-asserts this — wider tiles
//!     land in T3.6+; per-doc inherits the same constraint).
//!
//! # Production wiring status (CFTP v2 follow-on Sprint 1+5)
//!
//! The forward FFI dispatch in `nsl_flash_attention_csha` / `_with_saves`
//! (Sprint 1, commit `c80312dc`) recognises the `_per_doc_cta` kernel
//! suffix and uses the new trailing `num_docs_or_zero` arg as `grid_x`.
//! The backward FFI dispatch in `nsl_flash_attention_csha_backward`
//! (Sprint 5) mirrors the same suffix-detection — when the dispatched
//! backward kernel name carries `_per_doc_cta`, it launches with
//! `grid_x = num_docs` (skipping the per-q-block outer loop entirely)
//! and the per-doc prelude derives `%q_start` / `%k_max` from the
//! doc_starts table loaded via the kernel param.
//!
//! Both the forward PTX ([`crate::flash_attention_v2::per_doc_cta::synthesize_per_doc_cta_forward`])
//! and the backward PTX ([`crate::flash_attention_v2::per_doc_cta::synthesize_per_doc_cta_backward`])
//! are shipped and GPU-validated against segmented CPU references
//! (forward: max_abs 2.08e-4; backward: dQ 3.4e-5, dK 1.3e-3, dV 2.0e-2
//! on the 4-doc fixture with seg lengths {32, 28, 24, 16}).
//!
//! However, **no production code path flips `enable_per_doc_cta=true`
//! today**:
//!
//!   - The user-facing `@pca(strategy=per_document)` decorator is
//!     parsed and validated by `nsl_semantic::cftp::validate_pca_decorator`,
//!     but the returned `PcaConfig` is currently DROPPED — the
//!     semantic checker does not yet collect it onto a list analogous
//!     to `wrga_configs` (see `crates/nsl-semantic/src/checker/stmt.rs:443`).
//!   - The planner site that would call [`admit`] from production
//!     code does not exist; `admit()` is invoked only from unit / GPU
//!     tests that hardcode `enable_per_doc_cta: true`.
//!
//! Wiring `@pca(strategy=per_document)` → `enable_per_doc_cta=true`
//! → admission → kernel-name suffix → `nsl_flash_attention_csha`
//! `grid_x` override is **Sprint 2 follow-on work** (train-block
//! decorator collection — same mechanism `@wrga`, `@freeze`, etc.
//! already use). Once decorator collection is wired, the activation
//! is a one-line `PerDocAdmitConfig` construction at the planner
//! call site; this module + the FFI dispatch are ready to receive it.

use crate::flash_attention::FlashAttentionConfig;
use crate::pca_detect::{DatasetPackingConfig, PcaDetection, PcaStrategy};

/// Maximum number of documents per packed batch supported by v1.
///
/// Matches the `pca_rope::MAX_NUM_DOCS` SMEM-bake ceiling; the per-doc
/// kernel reads `doc_starts` from HBM (not SMEM), but capping at 256 here
/// keeps the runtime CPU-walk cost bounded at 256 reads.
pub const MAX_NUM_DOCS: u32 = 256;

/// Resolved per-doc CTA kernel plan.
#[derive(Debug, Clone)]
pub struct PerDocCtaPlan {
    /// FA config the emitter must use (with `segment_masked` forced to `false`).
    pub fa_config: FlashAttentionConfig,
    /// Conservative estimate of the maximum document length in this batch.
    /// Guaranteed `<= fa_config.block_q` by [`admit`].
    pub max_doc_len: u32,
    /// Estimated number of documents per packed batch item.
    pub num_docs_per_batch: u32,
    /// Mean document length from the dataset config.
    pub mean_doc_len: u32,
    /// Human-readable routing reason for the training-report CLI.
    pub reason: String,
}

/// Admission-check configuration.
#[derive(Debug, Clone)]
pub struct PerDocAdmitConfig {
    /// Maximum `block_q` for which the per-doc path is admitted.
    /// Default 128 — no per-doc kernel for tile sizes > 128 rows.
    pub max_block_q_for_per_doc: u32,
    /// Require causal attention (v1 is causal-only).  Default true.
    pub require_causal: bool,
    /// Hard ceiling on `num_docs_per_batch`.  Default `MAX_NUM_DOCS`.
    pub max_num_docs_per_batch: u32,
    /// Whether the per-doc CTA path is enabled at all.  Default false
    /// (v1 ships gated behind an explicit opt-in).
    pub enable_per_doc_cta: bool,
}

impl Default for PerDocAdmitConfig {
    fn default() -> Self {
        Self {
            max_block_q_for_per_doc: 128,
            require_causal: true,
            max_num_docs_per_batch: MAX_NUM_DOCS,
            enable_per_doc_cta: false, // default OFF — explicit opt-in required
        }
    }
}

/// Rejection reason returned when [`admit`] returns `Err`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// The planner flag `enable_per_doc_cta` is not set.
    NotEnabled,
    /// Detection strategy is not `PerDocumentCta`.
    WrongStrategy,
    /// `fa_config.causal == false`; v1 is causal-only.
    NotCausal,
    /// CSHA fused projections are active; v1 does not compose with them.
    CshaFusedProjections,
    /// Estimated max doc length exceeds `block_q`; one-CTA-per-doc constraint.
    MaxDocLenExceedsBlockQ { est: u32, block_q: u32 },
    /// Estimated number of documents per batch exceeds `max_num_docs_per_batch`.
    TooManyDocs { est: u32, limit: u32 },
}

/// Decide whether to route to the per-document CTA kernel.
///
/// Returns `Ok(plan)` iff all admission criteria are met; `Err(reason)`
/// otherwise (caller falls back to Tier A segment-masked path).
///
/// # Admission criteria (all must hold)
/// 1. `admit_cfg.enable_per_doc_cta == true`.
/// 2. `detection.strategy == PerDocumentCta`.
/// 3. `fa_config.causal == true` (when `require_causal`).
/// 4. No CSHA fused projections.
/// 5. `max_doc_len_est = min(mean + 3*stddev, max_seq_len) <= fa_config.block_q`.
/// 6. `num_docs_est = ceil(max_seq_len / mean) <= max_num_docs_per_batch`.
pub fn admit(
    detection: &PcaDetection,
    fa_config: &FlashAttentionConfig,
    packing_cfg: &DatasetPackingConfig,
    admit_cfg: &PerDocAdmitConfig,
) -> Result<PerDocCtaPlan, RejectReason> {
    // Gate 0: explicit opt-in flag.
    if !admit_cfg.enable_per_doc_cta {
        return Err(RejectReason::NotEnabled);
    }

    // Gate 1: strategy check.
    if detection.strategy != PcaStrategy::PerDocumentCta {
        return Err(RejectReason::WrongStrategy);
    }

    // Gate 2: causal-only in v1.
    if admit_cfg.require_causal && !fa_config.causal {
        return Err(RejectReason::NotCausal);
    }

    // Gate 3: no CSHA fused projections.
    if fa_config.csha.as_ref().is_some_and(|c| c.fused_projections) {
        return Err(RejectReason::CshaFusedProjections);
    }

    let max_seq = packing_cfg.max_sequence_length;
    let mean = packing_cfg.mean_doc_length.unwrap_or(max_seq);
    let stddev = packing_cfg.doc_length_stddev.unwrap_or(0);
    let block_q = fa_config.block_q as u32;

    // Gate 4: conservative max-doc-len estimate.
    // est = min(mean + 3*stddev, max_seq_len)
    let max_doc_len_est = (mean.saturating_add(stddev.saturating_mul(3))).min(max_seq);
    if max_doc_len_est > block_q {
        return Err(RejectReason::MaxDocLenExceedsBlockQ {
            est: max_doc_len_est,
            block_q,
        });
    }

    // Gate 5: num-docs estimate.
    let num_docs_est = if mean == 0 {
        admit_cfg.max_num_docs_per_batch + 1 // reject: division by zero case
    } else {
        max_seq.div_ceil(mean)
    };
    if num_docs_est > admit_cfg.max_num_docs_per_batch {
        return Err(RejectReason::TooManyDocs {
            est: num_docs_est,
            limit: admit_cfg.max_num_docs_per_batch,
        });
    }

    // All gates passed — build the plan.
    // Clone the FA config and force segment_masked=false: the per-CTA
    // decomposition makes segment masking structurally redundant (each CTA
    // only sees its own doc's tokens) so we must NOT emit the seg_smem
    // predicate.
    let mut per_doc_fa_config = fa_config.clone();
    per_doc_fa_config.segment_masked = false;

    let reason = format!(
        "per-doc CTA admitted: mean={mean}, stddev={stddev}, max_doc_est={max_doc_len_est}, \
         block_q={block_q}, num_docs_est={num_docs_est}, max_seq={max_seq}",
    );

    Ok(PerDocCtaPlan {
        fa_config: per_doc_fa_config,
        max_doc_len: max_doc_len_est,
        num_docs_per_batch: num_docs_est,
        mean_doc_len: mean,
        reason,
    })
}

/// CPU-side helper: walk `doc_starts` (host slice, i32 elements) to find
/// the sentinel entry whose value equals `active_seq_len` (the end position
/// of the last active document).  Returns that index as `num_docs`.
///
/// Sentinel convention: `doc_starts[num_docs] == active_seq_len`.
/// Example: `doc_starts=[0,32,60,84,100,128]` with `active_seq_len=100`
/// finds `doc_starts[4]==100` and returns `4` (four real documents).
///
/// The padded `seq_len` (128 in the example) is separate and NOT used here.
/// Callers must pass `active_seq_len` = the end position of the last document,
/// NOT the padded tensor dimension.
///
/// Used by the runtime to compute `grid_x = num_docs` without a new FFI
/// parameter (ABI-stable sentinel-walk approach per the design spec).
///
/// If no sentinel is found within `max_docs+1` entries, returns
/// `min(len, max_docs)` as a conservative upper bound.
pub fn compute_per_doc_grid_x(doc_starts: &[i32], active_seq_len: i32, max_docs: u32) -> u32 {
    let limit = ((max_docs as usize) + 1).min(doc_starts.len());
    for i in 0..limit {
        if doc_starts[i] == active_seq_len {
            return i as u32;
        }
    }
    // Fallback: sentinel not found within limit — return conservative bound.
    (limit as u32).saturating_sub(1).min(max_docs)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionConfig, RopeStyle};
    use crate::pca_detect::{DatasetPackingConfig, PcaDetection, PcaStrategy};

    fn base_fa_config() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 32,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 75,
            segment_masked: true, // typical for packed training
            csha: None,
            checkpoint: None,
        }
    }

    fn per_doc_detection() -> PcaDetection {
        PcaDetection {
            strategy: PcaStrategy::PerDocumentCta,
            expected_doc_fraction: 0.0625,
            rationale: "test".to_string(),
            segment_id_bytes_per_batch: 1024,
            eliminated_mask_bytes_per_batch: 1048576,
        }
    }

    fn packing_cfg_short() -> DatasetPackingConfig {
        DatasetPackingConfig {
            enabled: true,
            max_sequence_length: 512,
            mean_doc_length: Some(32),
            doc_length_stddev: Some(4),
            separator_token_id: Some(2),
        }
    }

    fn admit_cfg_enabled() -> PerDocAdmitConfig {
        PerDocAdmitConfig {
            enable_per_doc_cta: true,
            ..PerDocAdmitConfig::default()
        }
    }

    #[test]
    fn admit_chooses_per_doc_when_short_docs_and_block_q_fits() {
        // mean=32, stddev=4, est=32+12=44 <= block_q=64
        let plan = admit(
            &per_doc_detection(),
            &base_fa_config(),
            &packing_cfg_short(),
            &admit_cfg_enabled(),
        )
        .expect("should admit");
        assert!(plan.max_doc_len <= 64, "max_doc_len={} > block_q=64", plan.max_doc_len);
        assert_eq!(plan.num_docs_per_batch, 16); // ceil(512/32)
        // segment_masked must be forced to false in the plan's FA config.
        assert!(!plan.fa_config.segment_masked);
    }

    #[test]
    fn admit_rejects_when_max_doc_len_exceeds_block_q() {
        // mean=64, stddev=16 => est=64+48=112 > block_q=64
        let cfg = DatasetPackingConfig {
            enabled: true,
            max_sequence_length: 512,
            mean_doc_length: Some(64),
            doc_length_stddev: Some(16),
            separator_token_id: Some(2),
        };
        let result = admit(
            &per_doc_detection(),
            &base_fa_config(),
            &cfg,
            &admit_cfg_enabled(),
        );
        match result {
            Err(RejectReason::MaxDocLenExceedsBlockQ { est, block_q }) => {
                assert_eq!(est, 112);
                assert_eq!(block_q, 64);
            }
            other => panic!("expected MaxDocLenExceedsBlockQ, got {:?}", other),
        }
    }

    #[test]
    fn admit_rejects_when_not_causal() {
        let mut fa = base_fa_config();
        fa.causal = false;
        let result = admit(
            &per_doc_detection(),
            &fa,
            &packing_cfg_short(),
            &admit_cfg_enabled(),
        );
        match result {
            Err(RejectReason::NotCausal) => {}
            other => panic!("expected NotCausal, got {:?}", other),
        }
    }

    #[test]
    fn admit_rejects_when_csha_fused() {
        let mut fa = base_fa_config();
        fa.csha = Some(crate::flash_attention::CshaExtras {
            level: 1,
            fused_projections: true,
            ..Default::default()
        });
        let result = admit(
            &per_doc_detection(),
            &fa,
            &packing_cfg_short(),
            &admit_cfg_enabled(),
        );
        match result {
            Err(RejectReason::CshaFusedProjections) => {}
            other => panic!("expected CshaFusedProjections, got {:?}", other),
        }
    }

    #[test]
    fn admit_rejects_when_too_many_docs() {
        // mean=2, max_seq=4096 => num_docs_est=2048 > MAX_NUM_DOCS=256
        let cfg = DatasetPackingConfig {
            enabled: true,
            max_sequence_length: 4096,
            mean_doc_length: Some(2),
            doc_length_stddev: Some(0),
            separator_token_id: Some(2),
        };
        let result = admit(
            &per_doc_detection(),
            &base_fa_config(),
            &cfg,
            &admit_cfg_enabled(),
        );
        match result {
            Err(RejectReason::TooManyDocs { est, limit }) => {
                assert!(est > limit, "est={est} should exceed limit={limit}");
            }
            other => panic!("expected TooManyDocs, got {:?}", other),
        }
    }

    #[test]
    fn admit_rejects_when_not_enabled() {
        let result = admit(
            &per_doc_detection(),
            &base_fa_config(),
            &packing_cfg_short(),
            &PerDocAdmitConfig::default(), // enable_per_doc_cta = false
        );
        match result {
            Err(RejectReason::NotEnabled) => {}
            other => panic!("expected NotEnabled, got {:?}", other),
        }
    }

    #[test]
    fn admit_rejects_wrong_strategy() {
        let mut det = per_doc_detection();
        det.strategy = PcaStrategy::SegmentIdMasked;
        let result = admit(
            &det,
            &base_fa_config(),
            &packing_cfg_short(),
            &admit_cfg_enabled(),
        );
        match result {
            Err(RejectReason::WrongStrategy) => {}
            other => panic!("expected WrongStrategy, got {:?}", other),
        }
    }

    #[test]
    fn four_short_docs_detection_admits() {
        // Fixture from the design: {32,28,24,16} packed into seq_len=128.
        // mean approx 25; stddev=0; block_q=64.
        let cfg = DatasetPackingConfig {
            enabled: true,
            max_sequence_length: 128,
            mean_doc_length: Some(25),
            doc_length_stddev: Some(0),
            separator_token_id: Some(2),
        };
        let det = PcaDetection {
            strategy: PcaStrategy::PerDocumentCta,
            expected_doc_fraction: 0.195,
            rationale: "test".to_string(),
            segment_id_bytes_per_batch: 256,
            eliminated_mask_bytes_per_batch: 32768,
        };
        let plan = admit(&det, &base_fa_config(), &cfg, &admit_cfg_enabled())
            .expect("four short docs should admit");
        assert!(plan.max_doc_len <= 64);
        assert!(!plan.fa_config.segment_masked);
    }

    // ---------------------------------------------------------------------------
    // compute_per_doc_grid_x tests
    // ---------------------------------------------------------------------------

    /// Sentinel convention: doc_starts[num_docs] == active_seq_len.
    /// For doc_starts=[0,32,60,84,100,128] and active_seq_len=100
    /// (the end of the last real document), doc_starts[4]=100 is the sentinel
    /// and num_docs=4 is returned.
    #[test]
    fn runtime_walks_doc_starts_for_grid_x() {
        let doc_starts = [0i32, 32, 60, 84, 100, 128];
        // active_seq_len = 100 (last doc ends at position 100; [100,128) is padding).
        assert_eq!(compute_per_doc_grid_x(&doc_starts, 100, 256), 4);
    }

    #[test]
    fn compute_per_doc_grid_x_simple_two_docs() {
        // Two docs: [0,32) and [32,64). doc_starts=[0,32,64]. active_seq_len=64.
        let doc_starts = [0i32, 32, 64];
        assert_eq!(compute_per_doc_grid_x(&doc_starts, 64, 256), 2);
    }

    #[test]
    fn compute_per_doc_grid_x_respects_max_docs() {
        // Sentinel not found within max_docs+1 entries; return conservative bound.
        let doc_starts = [0i32, 32, 60];
        // active_seq_len=100 not present; limit = min(6, 3) = 3; return 3-1=2.
        let result = compute_per_doc_grid_x(&doc_starts, 100, 5);
        assert_eq!(result, 2);
    }

    #[test]
    fn runtime_falls_back_when_sentinel_not_found_in_range() {
        // If doc_starts has no entry == active_seq_len, returns a conservative estimate.
        let doc_starts = [0i32, 10, 20, 30, 40]; // no entry == 50
        let result = compute_per_doc_grid_x(&doc_starts, 50, 256);
        // Not found, limit = min(257, 5) = 5, returns min(4, 256) = 4.
        assert_eq!(result, 4);
    }
}
