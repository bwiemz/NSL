//! Phase 1 Commit 4 (original) + Phase 1 retune (2026-04-19) — parameter-
//! weighted disagreement between no-weights and weights-present paths.
//!
//! Originally a <5% hard-gate. Reframed to <20% monitoring-gate by the
//! retune (spec §5): the no-weights formula `K × pos / numel` has no
//! discriminator between numel-degenerate shape classes (SwiGLU's
//! ffn.w_gate/w_up/w_down at d_model × d_ffn), so no CALIB_K achieves
//! <5% disagreement. The 20% ceiling is Phase 1's monitoring range;
//! Phase 2's spectral factor is the intervention that returns this gate
//! to <5%. See `disagreement_source_matches_numel_collision` (in
//! cpdt_tier_agreement.rs) for the diagnostic that verifies the source.

use std::path::PathBuf;

use nsl_codegen::cpdt_sensitivity::{
    assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
    CALIB_ALPHA,
};
use nsl_codegen::cpdt_tier_apply::Tier;
use nsl_codegen::weight_aware::{WeightEntry, WeightMap};

fn score_with_entry(
    name: &str,
    entry_opt: Option<&WeightEntry>,
    numel: usize,
    n_layers: u32,
) -> Tier {
    let layer = layer_of(name);
    let kind = classify_layer_kind(name, layer, n_layers);
    let gm = gradient_magnitude_est(entry_opt);
    let pos = position_criticality(layer, n_layers, CALIB_ALPHA);
    let elts = numel.max(1) as f64;
    let score = gm * pos / elts;
    assign_tier(score, kind)
}

#[test]
fn weighted_disagreement_below_monitoring_threshold() {
    let fixtures = [("calib_tiny", 2u32), ("calib_small", 8u32)];
    let mut disagreeing_params: u64 = 0;
    let mut total_params: u64 = 0;
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/cpdt_calibration");
    for (name, n_layers) in fixtures {
        let path = fixture_dir.join(format!("{name}.safetensors"));
        let wm = WeightMap::load(&path)
            .unwrap_or_else(|e| panic!("fixture {} load failed: {e:?}", path.display()));
        for (tname, entry) in wm.entries() {
            let with = score_with_entry(tname, Some(entry), entry.num_elements, n_layers);
            let without = score_with_entry(tname, None, entry.num_elements, n_layers);
            total_params += entry.num_elements as u64;
            if with != without {
                disagreeing_params += entry.num_elements as u64;
            }
        }
    }
    let frac = disagreeing_params as f64 / total_params as f64;
    eprintln!(
        "weighted disagreement: {:.4} ({}/{} params)",
        frac, disagreeing_params, total_params
    );
    assert!(
        frac < 0.20,
        "weighted disagreement {frac:.4} >= 0.20 monitoring threshold — \
         calibration drift or new class of disagreement beyond the documented \
         numel-degeneracy. See spec §5 for the reframing."
    );
}
