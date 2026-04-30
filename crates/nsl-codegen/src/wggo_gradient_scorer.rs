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
///
/// `has_gradient_data_for_layer` returns `true` only when the scorer has
/// *real* gradient data (from a calibration sidecar) for the given key.
/// This is used for truth-in-logging: a `CalibratedGradientScorer` that
/// fell back to magnitude for a missing layer returns `false`, preventing
/// the log from falsely labelling it `gradient (calibrated)`.
pub trait GradientScorer: std::fmt::Debug + Send + Sync {
    fn score_layer(&self, layer_key: &str, layer_shape: &LayerShape) -> Option<HeadImportance>;

    /// Returns `true` when this scorer has real sidecar-derived gradient data
    /// for `layer_key` (as opposed to deriving a score from magnitude).
    /// The default implementation returns `false` (magnitude/null scorers).
    fn has_gradient_data_for_layer(&self, _layer_key: &str) -> bool {
        false
    }
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

    /// Returns `true` only when the sidecar actually contains gradient data
    /// for this layer key — i.e., the score came from calibration, not from
    /// the magnitude fallback.
    fn has_gradient_data_for_layer(&self, layer_key: &str) -> bool {
        self.sidecar_scores.contains_key(layer_key)
    }
}

// ---------------------------------------------------------------------------
// build_scorer dispatcher
// ---------------------------------------------------------------------------

/// Constructs the appropriate `GradientScorer` given compile options and a
/// weight provider for the magnitude fallback path.
///
/// Selection logic:
/// - `Magnitude` (any calibration state) → `NullGradientScorer`
///   (pure magnitude via `run_on_wengert_with_weights` caller path)
/// - `Grad` + no calibration_sidecar → `CodegenError`
/// - `Grad`/`Auto` + calibration_sidecar present → `CalibratedGradientScorer`
///   wrapping `MagnitudeFallbackScorer`
/// - `Auto` + no calibration_sidecar → `NullGradientScorer`
pub fn build_scorer(
    opts: &crate::CompileOptions,
    weight_provider: Arc<dyn WeightProvider + Send + Sync>,
) -> Result<Box<dyn GradientScorer>, crate::CodegenError> {
    use crate::WggoImportance;
    match (opts.wggo_importance, opts.calibration_sidecar.as_ref()) {
        (WggoImportance::Magnitude, _) => Ok(Box::new(NullGradientScorer)),
        (WggoImportance::Grad, None) => Err(crate::CodegenError::new(
            "--wggo-importance=grad requires --calibration-data",
        )),
        (WggoImportance::Grad, Some(sc)) | (WggoImportance::Auto, Some(sc)) => {
            let scores: BTreeMap<String, Vec<f32>> = sc
                .wggo_head_gradients
                .as_ref()
                .map(|g| {
                    g.by_layer
                        .iter()
                        .map(|(k, v)| (k.clone(), v.per_head_score.clone()))
                        .collect()
                })
                .unwrap_or_default();
            Ok(Box::new(CalibratedGradientScorer::new(
                scores,
                MagnitudeFallbackScorer::new(weight_provider),
            )))
        }
        (WggoImportance::Auto, None) => Ok(Box::new(NullGradientScorer)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::sidecar::{PerLayerGradient, Sidecar, WggoHeadGradients};
    use crate::{CompileOptions, WggoImportance};
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

    // -----------------------------------------------------------------------
    // build_scorer tests
    // -----------------------------------------------------------------------

    fn sidecar_with_grads() -> Sidecar {
        let mut by_layer = BTreeMap::new();
        by_layer.insert(
            "x".to_string(),
            PerLayerGradient {
                per_head_score: vec![1.0],
                batches_observed: 1,
            },
        );
        Sidecar {
            version: crate::calibration::sidecar::SIDECAR_VERSION,
            checkpoint_sha256: String::new(),
            calibration_data_sha256: String::new(),
            hook_set_sha256: String::new(),
            cache_key_digest: String::new(),
            num_samples_used: 1,
            hooks: BTreeMap::new(),
            wggo_head_gradients: Some(WggoHeadGradients { by_layer }),
        }
    }

    fn opts(importance: WggoImportance, sidecar: Option<Sidecar>) -> CompileOptions {
        let mut o = CompileOptions::default();
        o.wggo_importance = importance;
        o.calibration_sidecar = sidecar;
        o
    }

    #[test]
    fn build_auto_no_calibration_yields_null() {
        let s = build_scorer(
            &opts(WggoImportance::Auto, None),
            Arc::new(NullWeightProvider),
        )
        .unwrap();
        assert!(s.score_layer("k", &fixture_shape()).is_none());
    }

    #[test]
    fn build_magnitude_override_yields_null_even_with_sidecar() {
        let s = build_scorer(
            &opts(WggoImportance::Magnitude, Some(sidecar_with_grads())),
            Arc::new(NullWeightProvider),
        )
        .unwrap();
        assert!(s.score_layer("k", &fixture_shape()).is_none());
    }

    #[test]
    fn build_grad_without_calibration_errors() {
        let err = build_scorer(
            &opts(WggoImportance::Grad, None),
            Arc::new(NullWeightProvider),
        )
        .unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("requires --calibration-data"),
            "expected --calibration-data error, got: {msg}"
        );
    }

    #[test]
    fn build_auto_with_sidecar_yields_calibrated() {
        let s = build_scorer(
            &opts(WggoImportance::Auto, Some(sidecar_with_grads())),
            Arc::new(NullWeightProvider),
        )
        .unwrap();
        // Calibrated scorer: "x" resolves from sidecar, others fall back to magnitude.
        let hit = s.score_layer("x", &fixture_shape()).unwrap();
        assert_eq!(hit.per_head, vec![1.0]);
        // Missing key falls back to magnitude — uniform with null provider.
        let miss = s.score_layer("y", &fixture_shape()).unwrap();
        assert!(miss.per_head.iter().all(|&v| (v - miss.per_head[0]).abs() < 1e-6));
    }

    #[test]
    fn build_grad_with_sidecar_yields_calibrated() {
        let s = build_scorer(
            &opts(WggoImportance::Grad, Some(sidecar_with_grads())),
            Arc::new(NullWeightProvider),
        )
        .unwrap();
        let hit = s.score_layer("x", &fixture_shape()).unwrap();
        assert_eq!(hit.per_head, vec![1.0]);
    }
}
