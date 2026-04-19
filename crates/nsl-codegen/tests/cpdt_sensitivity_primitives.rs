//! Phase 1 Commit 2 — unit tests for cpdt_sensitivity primitives.

use nsl_codegen::cpdt_sensitivity::{
    assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
    LayerKind, ANALYSIS_VERSION, CALIB_ALPHA, CALIB_K, CALIB_T0, CALIB_T1, CALIB_T2,
};
use nsl_codegen::cpdt_tier_apply::Tier;

#[test]
fn analysis_version_pinned_to_one() {
    assert_eq!(ANALYSIS_VERSION, 1);
}

// --- position_criticality ---

#[test]
fn position_criticality_l1() {
    // L=1: only layer is first AND last → 2.0.
    assert_eq!(position_criticality(Some(0), 1, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_l2() {
    // Both layers are first or last; no middle.
    assert_eq!(position_criticality(Some(0), 2, CALIB_ALPHA), 2.0);
    assert_eq!(position_criticality(Some(1), 2, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_l3() {
    // L=3 < 4: near-extreme branch unreachable. Middle layer gets 1.0.
    assert_eq!(position_criticality(Some(0), 3, CALIB_ALPHA), 2.0);
    assert_eq!(position_criticality(Some(1), 3, CALIB_ALPHA), 1.0);
    assert_eq!(position_criticality(Some(2), 3, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_l4() {
    // First/last: 2.0. Near-extreme (l=1 or l=L-2=2): 1.0 + alpha.
    assert_eq!(position_criticality(Some(0), 4, CALIB_ALPHA), 2.0);
    assert_eq!(position_criticality(Some(1), 4, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    assert_eq!(position_criticality(Some(2), 4, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    assert_eq!(position_criticality(Some(3), 4, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_l8() {
    assert_eq!(position_criticality(Some(0), 8, CALIB_ALPHA), 2.0);
    assert_eq!(position_criticality(Some(1), 8, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    for l in 2..=5 {
        assert_eq!(position_criticality(Some(l), 8, CALIB_ALPHA), 1.0);
    }
    assert_eq!(position_criticality(Some(6), 8, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    assert_eq!(position_criticality(Some(7), 8, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_l16() {
    assert_eq!(position_criticality(Some(0), 16, CALIB_ALPHA), 2.0);
    assert_eq!(position_criticality(Some(1), 16, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    for l in 2..=13 {
        assert_eq!(position_criticality(Some(l), 16, CALIB_ALPHA), 1.0);
    }
    assert_eq!(position_criticality(Some(14), 16, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    assert_eq!(position_criticality(Some(15), 16, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_unknown_layer_borderline_high() {
    assert_eq!(position_criticality(None, 8, CALIB_ALPHA), 1.5);
}

// --- gradient_magnitude_est ---

#[test]
fn gradient_magnitude_est_none_returns_calib_k() {
    assert_eq!(gradient_magnitude_est(None), CALIB_K);
}

// --- assign_tier boundaries ---

#[test]
fn assign_tier_above_t0_high() {
    assert_eq!(assign_tier(CALIB_T0 + 1e-6, LayerKind::Generic), Tier::High);
}

#[test]
fn assign_tier_t0_exact_is_medium() {
    // assign_tier uses strict > comparisons; score exactly at boundary falls to next tier.
    assert_eq!(assign_tier(CALIB_T0, LayerKind::Generic), Tier::Medium);
}

#[test]
fn assign_tier_between_t1_t0_medium() {
    let mid = (CALIB_T1 + CALIB_T0) / 2.0;
    assert_eq!(assign_tier(mid, LayerKind::Generic), Tier::Medium);
}

#[test]
fn assign_tier_between_t2_t1_low() {
    let mid = (CALIB_T2 + CALIB_T1) / 2.0;
    assert_eq!(assign_tier(mid, LayerKind::Generic), Tier::Low);
}

#[test]
fn assign_tier_below_t2_very_low() {
    assert_eq!(assign_tier(CALIB_T2 - 1e-6, LayerKind::Generic), Tier::VeryLow);
}

// --- layer-kind overrides ---

#[test]
fn embedding_always_high_regardless_of_score() {
    assert_eq!(assign_tier(0.0, LayerKind::Embedding), Tier::High);
    assert_eq!(assign_tier(1e9, LayerKind::Embedding), Tier::High);
}

#[test]
fn norm_always_high_regardless_of_score() {
    assert_eq!(assign_tier(0.0, LayerKind::Norm), Tier::High);
}

#[test]
fn first_or_last_always_high() {
    assert_eq!(assign_tier(0.0, LayerKind::FirstOrLast), Tier::High);
}

#[test]
fn is_kind_overridden_matches_expectation() {
    assert!(LayerKind::Embedding.is_kind_overridden());
    assert!(LayerKind::Norm.is_kind_overridden());
    assert!(LayerKind::FirstOrLast.is_kind_overridden());
    assert!(!LayerKind::Generic.is_kind_overridden());
}

// --- classify_layer_kind ---

#[test]
fn classify_embedding_patterns() {
    assert_eq!(
        classify_layer_kind("tok_embeddings.weight", None, 8),
        LayerKind::Embedding
    );
    assert_eq!(
        classify_layer_kind("wte.weight", None, 8),
        LayerKind::Embedding
    );
    assert_eq!(
        classify_layer_kind("position_embedding.weight", None, 8),
        LayerKind::Embedding
    );
}

#[test]
fn classify_norm_patterns() {
    assert_eq!(
        classify_layer_kind("blocks.3.attn_norm.weight", Some(3), 8),
        LayerKind::Norm
    );
    assert_eq!(
        classify_layer_kind("norm.weight", None, 8),
        LayerKind::Norm
    );
}

#[test]
fn classify_first_or_last_layer() {
    assert_eq!(
        classify_layer_kind("blocks.0.attn.wq.weight", Some(0), 8),
        LayerKind::FirstOrLast
    );
    assert_eq!(
        classify_layer_kind("blocks.7.attn.wq.weight", Some(7), 8),
        LayerKind::FirstOrLast
    );
    assert_eq!(
        classify_layer_kind("blocks.4.attn.wq.weight", Some(4), 8),
        LayerKind::Generic
    );
}

// --- layer_of ---

#[test]
fn layer_of_recognises_patterns() {
    assert_eq!(layer_of("blocks.6.attn.wq"), Some(6));
    assert_eq!(layer_of("layers.12.norm"), Some(12));
    assert_eq!(layer_of("h.3.mlp.fc"), Some(3));
    assert_eq!(layer_of("embedding.weight"), None);
}
