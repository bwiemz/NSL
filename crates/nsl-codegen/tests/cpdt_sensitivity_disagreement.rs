//! Phase 1 Commit 4 — parameter-weighted disagreement between no-weights
//! and weights-present paths on the baseline corpus. Gate: < 5%.

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
fn weighted_disagreement_below_5_percent() {
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
        frac < 0.05,
        "weighted disagreement {frac:.4} >= 0.05 on baseline corpus"
    );
}
