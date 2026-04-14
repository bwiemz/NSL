//! WGGO Phase 2: gradient-informed head importance scoring.
//!
//! `GradientScorer` abstracts over three backends: a null scorer that
//! never contributes (consumer falls back to magnitude), a magnitude-only
//! fallback scorer (used by `CalibratedGradientScorer` for per-layer
//! fallback when the sidecar lacks entries for a given layer), and the
//! calibrated scorer that reads sidecar-produced `wggo_head_gradients`.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::wggo_cost::LayerShape;
use crate::wggo_ilp::HeadImportance;
use crate::wggo_weight_analysis::{score_layer_magnitude, WeightProvider};

/// Abstraction over gradient-based head importance scoring backends.
///
/// Consumers hold a `Box<dyn GradientScorer>` or `Arc<dyn GradientScorer>`.
/// `score_layer` returns `Some(HeadImportance)` when the scorer has a
/// signal for the layer, or `None` to signal that the consumer should
/// fall back to its own magnitude or default scoring path.
///
/// `NullGradientScorer` always returns `None`.
/// `MagnitudeFallbackScorer` always returns `Some(...)`.
/// `CalibratedGradientScorer` returns `Some(...)` from sidecar data when
/// present, or delegates to its inner `MagnitudeFallbackScorer` otherwise.
pub trait GradientScorer: std::fmt::Debug + Send + Sync {
    fn score_layer(&self, layer_key: &str, layer_shape: &LayerShape) -> Option<HeadImportance>;
}

// ---------------------------------------------------------------------------
// NullGradientScorer
// ---------------------------------------------------------------------------

/// A scorer that always returns `None`, indicating no gradient signal is
/// available.  Consumers receiving `None` should fall back to magnitude-based
/// scoring or a uniform default.
#[derive(Debug, Default)]
pub struct NullGradientScorer;

impl GradientScorer for NullGradientScorer {
    fn score_layer(&self, _layer_key: &str, _layer_shape: &LayerShape) -> Option<HeadImportance> {
        None
    }
}

// ---------------------------------------------------------------------------
// MagnitudeFallbackScorer
// ---------------------------------------------------------------------------

/// A scorer that derives head importance from weight magnitudes via
/// `score_layer_magnitude`.  Always returns `Some(HeadImportance)`.
///
/// Used directly when only weight data is available, and used internally
/// by `CalibratedGradientScorer` to handle layers missing from the sidecar.
#[derive(Debug)]
pub struct MagnitudeFallbackScorer {
    provider: Arc<dyn WeightProvider + Send + Sync>,
}

impl MagnitudeFallbackScorer {
    pub fn new(provider: Arc<dyn WeightProvider + Send + Sync>) -> Self {
        Self { provider }
    }
}

impl GradientScorer for MagnitudeFallbackScorer {
    fn score_layer(&self, key: &str, shape: &LayerShape) -> Option<HeadImportance> {
        Some(score_layer_magnitude(key, shape, &*self.provider))
    }
}

// ---------------------------------------------------------------------------
// CalibratedGradientScorer
// ---------------------------------------------------------------------------

/// A scorer that prefers sidecar-provided gradient scores and falls back to
/// magnitude-based scoring for layers absent from the sidecar.
///
/// The sidecar map is keyed by layer name (e.g. `"blocks.0"`) and stores a
/// `Vec<f32>` of per-head gradient norms gathered during a calibration forward
/// pass.  The calibrated scorer wraps a `MagnitudeFallbackScorer` so callers
/// always receive `Some(HeadImportance)` — never `None`.
#[derive(Debug)]
pub struct CalibratedGradientScorer {
    sidecar_scores: BTreeMap<String, Vec<f32>>,
    fallback: MagnitudeFallbackScorer,
}

impl CalibratedGradientScorer {
    pub fn new(
        sidecar_scores: BTreeMap<String, Vec<f32>>,
        fallback: MagnitudeFallbackScorer,
    ) -> Self {
        Self {
            sidecar_scores,
            fallback,
        }
    }
}

impl GradientScorer for CalibratedGradientScorer {
    fn score_layer(&self, key: &str, shape: &LayerShape) -> Option<HeadImportance> {
        if let Some(scores) = self.sidecar_scores.get(key) {
            Some(HeadImportance::from_per_head_f32(scores))
        } else {
            self.fallback.score_layer(key, shape)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wggo_weight_analysis::NullWeightProvider;

    fn fixture_shape() -> LayerShape {
        // d_model=256, head_dim=64 → num_heads=4; matches default_for_test_4heads.
        LayerShape::default_for_test_4heads()
    }

    #[test]
    fn null_returns_none() {
        assert!(NullGradientScorer
            .score_layer("k", &fixture_shape())
            .is_none());
    }

    #[test]
    fn magnitude_fallback_returns_some_uniform_with_null_provider() {
        let s = MagnitudeFallbackScorer::new(Arc::new(NullWeightProvider) as Arc<dyn WeightProvider + Send + Sync>);
        let hi = s.score_layer("k", &fixture_shape()).unwrap();
        assert!(
            hi.per_head.iter().all(|&v| (v - hi.per_head[0]).abs() < 1e-6),
            "uniform fallback expected with NullWeightProvider, got {:?}",
            hi.per_head
        );
    }

    #[test]
    fn calibrated_prefers_sidecar_when_key_present() {
        let mut scores = BTreeMap::new();
        scores.insert("k".into(), vec![10.0f32, 20.0, 30.0, 40.0]);
        let s = CalibratedGradientScorer::new(
            scores,
            MagnitudeFallbackScorer::new(Arc::new(NullWeightProvider) as Arc<dyn WeightProvider + Send + Sync>),
        );
        let hi = s.score_layer("k", &fixture_shape()).unwrap();
        assert_eq!(hi.per_head, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn calibrated_falls_back_to_magnitude_on_missing_layer() {
        let s = CalibratedGradientScorer::new(
            BTreeMap::new(),
            MagnitudeFallbackScorer::new(Arc::new(NullWeightProvider) as Arc<dyn WeightProvider + Send + Sync>),
        );
        // Missing key → magnitude fallback → Some(uniform)
        let hi = s.score_layer("k", &fixture_shape()).unwrap();
        assert!(
            hi.per_head.iter().all(|&v| (v - hi.per_head[0]).abs() < 1e-6),
            "expected uniform fallback, got {:?}",
            hi.per_head
        );
    }

    #[test]
    fn calibrated_never_returns_none() {
        // Even with an empty sidecar the calibrated scorer must always
        // return Some — the fallback path covers missing entries.
        let s = CalibratedGradientScorer::new(
            BTreeMap::new(),
            MagnitudeFallbackScorer::new(Arc::new(NullWeightProvider) as Arc<dyn WeightProvider + Send + Sync>),
        );
        assert!(s.score_layer("any_layer", &fixture_shape()).is_some());
    }

    #[test]
    fn trait_object_dispatch_works() {
        // Verify all three impls can be used through Box<dyn GradientScorer>.
        let scorers: Vec<Box<dyn GradientScorer>> = vec![
            Box::new(NullGradientScorer),
            Box::new(MagnitudeFallbackScorer::new(Arc::new(NullWeightProvider) as Arc<dyn WeightProvider + Send + Sync>)),
            Box::new(CalibratedGradientScorer::new(
                BTreeMap::new(),
                MagnitudeFallbackScorer::new(Arc::new(NullWeightProvider) as Arc<dyn WeightProvider + Send + Sync>),
            )),
        ];
        let shape = fixture_shape();
        let results: Vec<Option<HeadImportance>> =
            scorers.iter().map(|s| s.score_layer("l", &shape)).collect();
        assert!(results[0].is_none());  // Null
        assert!(results[1].is_some());  // Magnitude
        assert!(results[2].is_some());  // Calibrated (empty sidecar → fallback)
    }
}
