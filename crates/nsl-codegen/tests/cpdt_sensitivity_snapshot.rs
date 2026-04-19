//! Phase 1 Commit 3 — weights-present regression snapshot.
//!
//! Compares unified scorer output on calib_{tiny, small} against
//! tests/fixtures/cpdt_calibration/expected_weights_present.json.
//!
//! Ignored in Commit 3 and unblocked in Commit 4 after gradient_magnitude_est
//! is verified to read weights in the Some-branch. (The Some-branch already
//! exists today; Commit 4 is a green-flip, not a code-add.)

use std::path::Path;

use nsl_codegen::cpdt_sensitivity::{
    assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
    CALIB_ALPHA,
};
use nsl_codegen::weight_aware::WeightMap;
use serde::Deserialize;

#[derive(Deserialize)]
struct TierEntry {
    name: String,
    tier: String,
}

#[derive(Deserialize)]
struct FixtureSnapshot {
    fixture: String,
    tiers: Vec<TierEntry>,
}

fn load_expected() -> Vec<FixtureSnapshot> {
    let path = Path::new("tests/fixtures/cpdt_calibration/expected_weights_present.json");
    let json = std::fs::read_to_string(path).expect("expected_weights_present.json not found");
    serde_json::from_str(&json).expect("expected_weights_present.json parse error")
}

fn n_layers_for(fixture: &str) -> u32 {
    match fixture {
        "calib_tiny" => 2,
        "calib_small" => 8,
        "calib_medium" => 16,
        other => panic!("unknown fixture: {other}"),
    }
}

#[test]
#[ignore = "red: unblocked by Commit 4 (gradient_magnitude_est weights-reading path verification)"]
fn weights_present_matches_expected_snapshot() {
    let expected = load_expected();
    for fs in &expected {
        let path = format!(
            "tests/fixtures/cpdt_calibration/{}.safetensors",
            fs.fixture
        );
        let Ok(wm) = WeightMap::load(Path::new(&path)) else {
            if fs.fixture == "calib_medium" {
                continue;
            }
            panic!("fixture missing: {path}");
        };
        let n_layers = n_layers_for(&fs.fixture);
        for expected_entry in &fs.tiers {
            let entry = wm
                .get(&expected_entry.name)
                .unwrap_or_else(|| panic!("tensor missing: {}", expected_entry.name));
            let layer = layer_of(&expected_entry.name);
            let kind = classify_layer_kind(&expected_entry.name, layer, n_layers);
            let gm = gradient_magnitude_est(Some(entry));
            let pos = position_criticality(layer, n_layers, CALIB_ALPHA);
            let elts = entry.num_elements.max(1) as f64;
            let score = gm * pos / elts;
            let tier = assign_tier(score, kind);
            assert_eq!(
                tier.as_str(),
                expected_entry.tier,
                "tier mismatch on {}/{}: expected {}, got {}",
                fs.fixture,
                expected_entry.name,
                expected_entry.tier,
                tier.as_str()
            );
        }
    }
}
