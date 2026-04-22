//! PCA — Packed Causal Attention: packing-configuration detection.
//!
//! Scans a [`DatasetPackingConfig`] (extracted from the `dataset … packing
//! = true` block) and decides which PCA kernel variant the compiler should
//! synthesise for the downstream attention sublayer:
//!
//!   * **SegmentIdMasked** — the general case (arbitrary document lengths).
//!     One compact `segment_ids: [u16; seq_len]` tensor per batch element
//!     replaces the dense `S×S` attention mask.
//!   * **PerDocumentCta** — documents are short and roughly even in
//!     length; each CTA processes one complete document instead of one
//!     Q-tile of a packed sequence.
//!   * **NoPacking** — packing disabled; standard FlashAttention.

use serde::Serialize;

/// Parsed packing configuration from a `dataset` block.
#[derive(Debug, Clone, Serialize)]
#[derive(Default)]
pub struct DatasetPackingConfig {
    /// Whether the dataset has `packing = true`.
    pub enabled: bool,
    /// Maximum sequence length per packed sample.
    pub max_sequence_length: u32,
    /// Mean document length across the corpus (from dataset statistics or
    /// user annotation).  Used to decide between general and per-doc CTA.
    pub mean_doc_length: Option<u32>,
    /// Standard deviation of document length.  Used to decide whether per-
    /// doc CTA scheduling will suffer from load imbalance.
    pub doc_length_stddev: Option<u32>,
    /// The separator token id used to mark document boundaries.
    pub separator_token_id: Option<i64>,
}


/// PCA strategy the compiler should apply to the attention sublayer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PcaStrategy {
    /// Packing not in use — no PCA rewrite.
    NoPacking,
    /// General segment-ID masked kernel.
    SegmentIdMasked,
    /// Per-document CTA scheduling: one CTA ↔ one complete document.
    PerDocumentCta,
}

impl PcaStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            PcaStrategy::NoPacking => "no_packing",
            PcaStrategy::SegmentIdMasked => "segment_id_masked",
            PcaStrategy::PerDocumentCta => "per_document_cta",
        }
    }
}

/// Full PCA detection result.
#[derive(Debug, Clone, Serialize)]
pub struct PcaDetection {
    pub strategy: PcaStrategy,
    /// Expected fraction of the packed sequence that sits inside a single
    /// document (used to estimate tile-skip savings).
    pub expected_doc_fraction: f64,
    /// Rationale string — kept in the plan so the `nsl check
    /// --training-report` CLI has something to print.
    pub rationale: String,
    /// Size in bytes of the segment-ID tensor per batch element (0 when
    /// PCA is disabled or when the per-doc CTA path doesn't need them).
    pub segment_id_bytes_per_batch: u64,
    /// Size in bytes of the dense `S×S` mask that PCA eliminates.
    pub eliminated_mask_bytes_per_batch: u64,
}

/// Configuration thresholds.
#[derive(Debug, Clone)]
pub struct PcaDetectConfig {
    /// Ratio of mean-doc-length over max-sequence-length below which we
    /// prefer per-document CTA scheduling.  Default 0.25 — i.e. if docs
    /// average less than a quarter of the packed sequence, we schedule
    /// per doc.
    pub per_doc_threshold: f64,
    /// Coefficient-of-variation (σ/μ) above which per-document CTA is
    /// rejected because load-imbalance penalties dominate.
    pub max_cv: f64,
}

impl Default for PcaDetectConfig {
    fn default() -> Self {
        Self {
            per_doc_threshold: 0.25,
            max_cv: 1.0,
        }
    }
}

/// Bytes consumed by a dense `S×S` attention mask for one batch element
/// (FP16 mask assumed — PyTorch uses f16 for `torch.float16` models).
fn dense_mask_bytes(seq: u32, dtype_bytes: u64) -> u64 {
    (seq as u64) * (seq as u64) * dtype_bytes
}

/// Bytes for the compact `segment_ids` tensor (`[S]` of u16). Segment
/// IDs are non-negative indices into a pack; u16 ceiling is 65535
/// segments per pack (unreachable in realistic corpora, guarded by
/// `validate_config`).
fn segment_ids_bytes(seq: u32) -> u64 {
    (seq as u64) * 2
}

/// Run the strategy selection.
pub fn detect(
    cfg: &DatasetPackingConfig,
    detect_cfg: &PcaDetectConfig,
    attention_dtype_bytes: u64,
) -> PcaDetection {
    if !cfg.enabled || cfg.max_sequence_length == 0 {
        return PcaDetection {
            strategy: PcaStrategy::NoPacking,
            expected_doc_fraction: 1.0,
            rationale: "packing disabled — standard FlashAttention".to_string(),
            segment_id_bytes_per_batch: 0,
            eliminated_mask_bytes_per_batch: 0,
        };
    }

    let seq = cfg.max_sequence_length;
    let mean = cfg.mean_doc_length.unwrap_or(seq);
    let stddev = cfg.doc_length_stddev.unwrap_or(0);
    let mean_ratio = mean as f64 / seq as f64;
    let cv = if mean > 0 {
        stddev as f64 / mean as f64
    } else {
        f64::INFINITY
    };
    let mask_bytes = dense_mask_bytes(seq, attention_dtype_bytes);
    let seg_bytes = segment_ids_bytes(seq);

    if mean_ratio < detect_cfg.per_doc_threshold && cv <= detect_cfg.max_cv {
        let expected_doc_fraction = mean_ratio;
        let rationale = format!(
            "mean_doc_len/max_seq = {:.2} < {:.2} and σ/μ = {:.2} ≤ {:.2} — per-document CTA",
            mean_ratio, detect_cfg.per_doc_threshold, cv, detect_cfg.max_cv
        );
        PcaDetection {
            strategy: PcaStrategy::PerDocumentCta,
            expected_doc_fraction,
            rationale,
            segment_id_bytes_per_batch: seg_bytes, // still used for position-reset
            eliminated_mask_bytes_per_batch: mask_bytes,
        }
    } else {
        let expected_doc_fraction = mean_ratio.min(1.0);
        let rationale = format!(
            "mean_doc_len/max_seq = {:.2} — segment-ID-masked kernel (general case)",
            mean_ratio
        );
        PcaDetection {
            strategy: PcaStrategy::SegmentIdMasked,
            expected_doc_fraction,
            rationale,
            segment_id_bytes_per_batch: seg_bytes,
            eliminated_mask_bytes_per_batch: mask_bytes,
        }
    }
}

/// Compile-time validation that a packing configuration will not
/// exceed the u16 segment-count ceiling. Spec §4 invariant #3.
pub fn validate_config(cfg: &DatasetPackingConfig) -> Result<(), String> {
    if !cfg.enabled {
        return Ok(());
    }
    let mean = cfg.mean_doc_length.unwrap_or(cfg.max_sequence_length.max(1));
    if mean == 0 {
        return Err(
            "packing.mean_doc_length must be > 0 when packing enabled".to_string(),
        );
    }
    // Worst-case segments per pack = max_sequence_length / min_doc_length.
    // We approximate min_doc_length as mean/4 (conservative lower bound).
    let min_doc = (mean / 4).max(1);
    let worst_case_segments = cfg.max_sequence_length as u64 / min_doc as u64;
    if worst_case_segments > u16::MAX as u64 {
        return Err(format!(
            "packing config may exceed u16 segment ceiling: \
             max_sequence_length={}, min_doc_length≈{}, worst-case segments={} > {}",
            cfg.max_sequence_length, min_doc, worst_case_segments, u16::MAX
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_packing_returns_no_packing() {
        let d = detect(
            &DatasetPackingConfig::default(),
            &PcaDetectConfig::default(),
            2,
        );
        assert_eq!(d.strategy, PcaStrategy::NoPacking);
        assert_eq!(d.segment_id_bytes_per_batch, 0);
        assert_eq!(d.eliminated_mask_bytes_per_batch, 0);
    }

    #[test]
    fn long_docs_go_segment_id_masked() {
        let d = detect(
            &DatasetPackingConfig {
                enabled: true,
                max_sequence_length: 1024,
                mean_doc_length: Some(512),
                doc_length_stddev: Some(200),
                separator_token_id: Some(2),
            },
            &PcaDetectConfig::default(),
            2,
        );
        assert_eq!(d.strategy, PcaStrategy::SegmentIdMasked);
    }

    #[test]
    fn short_docs_go_per_doc_cta() {
        let d = detect(
            &DatasetPackingConfig {
                enabled: true,
                max_sequence_length: 1024,
                mean_doc_length: Some(200),
                doc_length_stddev: Some(50),
                separator_token_id: Some(2),
            },
            &PcaDetectConfig::default(),
            2,
        );
        assert_eq!(d.strategy, PcaStrategy::PerDocumentCta);
    }

    #[test]
    fn highly_variable_doc_lengths_fall_back_to_seg_id() {
        let d = detect(
            &DatasetPackingConfig {
                enabled: true,
                max_sequence_length: 1024,
                mean_doc_length: Some(200),
                doc_length_stddev: Some(1000), // CV ≫ 1
                separator_token_id: Some(2),
            },
            &PcaDetectConfig::default(),
            2,
        );
        assert_eq!(d.strategy, PcaStrategy::SegmentIdMasked);
    }

    #[test]
    fn mask_bytes_match_dense_s_squared() {
        let d = detect(
            &DatasetPackingConfig {
                enabled: true,
                max_sequence_length: 2048,
                mean_doc_length: Some(1024),
                doc_length_stddev: Some(100),
                separator_token_id: Some(2),
            },
            &PcaDetectConfig::default(),
            2,
        );
        // 2048 × 2048 × 2 bytes = 8 MB.
        assert_eq!(d.eliminated_mask_bytes_per_batch, 2048u64 * 2048 * 2);
        // Segment IDs: 2048 × 2 bytes = 4 KB (u16 per position).
        assert_eq!(d.segment_id_bytes_per_batch, 2048 * 2);
    }

    #[test]
    fn segment_ids_bytes_is_u16_sized() {
        assert_eq!(segment_ids_bytes(2048), 4096);
        assert_eq!(segment_ids_bytes(1024), 2048);
    }

    #[test]
    fn validate_config_accepts_realistic_corpus() {
        let cfg = DatasetPackingConfig {
            enabled: true,
            max_sequence_length: 2048,
            mean_doc_length: Some(400),
            doc_length_stddev: Some(100),
            separator_token_id: Some(2),
        };
        assert!(validate_config(&cfg).is_ok());
    }

    #[test]
    fn validate_config_rejects_u16_overflow_case() {
        let cfg = DatasetPackingConfig {
            enabled: true,
            max_sequence_length: 65536,
            mean_doc_length: Some(4),
            doc_length_stddev: Some(1),
            separator_token_id: Some(2),
        };
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn validate_config_allows_disabled_packing() {
        assert!(validate_config(&DatasetPackingConfig::default()).is_ok());
    }

    #[test]
    fn rationale_is_never_empty_when_enabled() {
        let d = detect(
            &DatasetPackingConfig {
                enabled: true,
                max_sequence_length: 1024,
                mean_doc_length: Some(300),
                doc_length_stddev: Some(100),
                separator_token_id: Some(2),
            },
            &PcaDetectConfig::default(),
            2,
        );
        assert!(!d.rationale.is_empty());
    }
}
