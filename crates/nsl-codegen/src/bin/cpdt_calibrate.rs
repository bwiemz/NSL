//! Dev-only CPDT calibration binary. Gated behind `[features] calibrate = []`
//! so it doesn't land in release builds.
//!
//! Usage (from repo root):
//!   cargo run --features calibrate --bin cpdt_calibrate -- <fixture_dir>
//!
//! Outputs:
//!   * <fixture_dir>/baseline_heuristic.json
//!   * stdout: diff-ready Rust constants block for copy-pasting into
//!     cpdt_sensitivity.rs.

use std::path::Path;

use nsl_codegen::cpdt_sensitivity::{
    assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
    ANALYSIS_VERSION, CALIB_ALPHA, CALIB_K, CALIB_T0, CALIB_T1, CALIB_T2,
};
use nsl_codegen::cpdt_tier_apply::PrecisionConfig;
use nsl_codegen::weight_aware::WeightMap;
use serde::Serialize;

#[derive(Serialize)]
struct TierEntry {
    name: String,
    tier: &'static str,
    score: f64,
}

#[derive(Serialize)]
struct FixtureSnapshot {
    fixture: String,
    tiers: Vec<TierEntry>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: cpdt_calibrate <fixture_dir>");
        std::process::exit(1);
    }
    let fixture_dir = Path::new(&args[1]);
    let fixtures = ["calib_tiny", "calib_small", "calib_medium"];
    let mut snapshots = Vec::new();

    for fixture in &fixtures {
        let path = fixture_dir.join(format!("{fixture}.safetensors"));
        let wm = match WeightMap::load(&path) {
            Ok(w) => w,
            Err(_) if *fixture == "calib_medium" => {
                eprintln!("calib_medium missing; skipping (regenerated at test-time into target/)");
                continue;
            }
            Err(e) => {
                eprintln!("failed to load {fixture}: {e:?}");
                std::process::exit(1);
            }
        };
        let cfg = infer_config_from_fixture(fixture);
        let mut tiers = Vec::new();
        for (name, entry) in wm.entries() {
            let layer = layer_of(name);
            let kind = classify_layer_kind(name, layer, cfg.n_layers);
            let gm = gradient_magnitude_est(Some(entry));
            let pos = position_criticality(layer, cfg.n_layers, CALIB_ALPHA);
            let elts = entry.num_elements.max(1) as f64;
            let score = gm * pos / elts;
            let tier = assign_tier(score, kind);
            tiers.push(TierEntry {
                name: name.clone(),
                tier: tier.as_str(),
                score,
            });
        }
        tiers.sort_by(|a, b| a.name.cmp(&b.name));
        snapshots.push(FixtureSnapshot {
            fixture: fixture.to_string(),
            tiers,
        });
    }

    let json = serde_json::to_string_pretty(&snapshots).unwrap();
    let out_path = fixture_dir.join("baseline_heuristic.json");
    std::fs::write(&out_path, &json).unwrap();
    println!("wrote {}", out_path.display());
    println!();
    println!("pub const ANALYSIS_VERSION: u32 = {};", ANALYSIS_VERSION);
    println!("pub const CALIB_K:     f64 = {};", CALIB_K);
    println!("pub const CALIB_T0:    f64 = {};", CALIB_T0);
    println!("pub const CALIB_T1:    f64 = {};", CALIB_T1);
    println!("pub const CALIB_T2:    f64 = {};", CALIB_T2);
    println!("pub const CALIB_ALPHA: f64 = {};", CALIB_ALPHA);
}

fn infer_config_from_fixture(fixture: &str) -> PrecisionConfig {
    let n_layers = match fixture {
        "calib_tiny" => 2,
        "calib_small" => 8,
        "calib_medium" => 16,
        _ => 8,
    };
    PrecisionConfig {
        n_layers,
        ..Default::default()
    }
}
