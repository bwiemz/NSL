//! Integration tests for `nsl profile`.

use nsl_cli::profile::{run_profile, ProfileArgs};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn sample_args() -> ProfileArgs {
    ProfileArgs {
        file: fixture("tiny_transformer.nsl"),
        target: "h100".into(),
        dtype: "bf16".into(),
        batch: 1,
        seq: 2048,
        dim: vec![],
        fusion: true,
        memory: false,
        entry: "auto".into(),
        json: false,
        explain_wggo: false,
    }
}

#[test]
fn explain_wggo_flag_appends_explanation_block() {
    let mut args = sample_args();
    args.explain_wggo = true;
    let out = run_profile(&args).expect("profile should succeed");
    assert!(out.contains("=== NSL Predictive Profile ==="),
        "per-op table should still be present (additive flag)");
    assert!(out.contains("=== WGGO Decision Explanation ==="),
        "WGGO explanation block missing — flag did not append");
}

#[test]
fn explain_wggo_with_json_attaches_wggo_plan() {
    let mut args = sample_args();
    args.json = true;
    args.explain_wggo = true;
    let out = run_profile(&args).expect("profile should succeed");
    let v: serde_json::Value = serde_json::from_str(&out).expect("must be valid JSON");
    assert!(!v["wggo_explain"].is_null(),
        "wggo_explain should be populated in JSON mode, got null");
    let per_layer = &v["wggo_explain"]["per_layer"];
    assert!(per_layer.is_array(), "per_layer should be array, got {per_layer:?}");
}

#[test]
fn profile_runs_on_fixture_and_mentions_key_sections() {
    let out = run_profile(&sample_args()).expect("profile should succeed");
    assert!(out.contains("NSL Predictive Profile"), "missing header: {out}");
    assert!(out.contains("Target:"), "missing Target: {out}");
    assert!(out.contains("Totals:"), "missing Totals: {out}");
}

#[test]
fn json_flag_emits_parseable_report() {
    let args = ProfileArgs {
        json: true,
        ..sample_args()
    };
    let out = run_profile(&args).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out).expect("must be valid JSON");
    // find_gpu("h100") resolves to H100-SXM in the current GPU database.
    assert_eq!(v["target_gpu"].as_str().unwrap_or(""), "H100-SXM");
}

#[test]
fn memory_flag_appends_timeline() {
    let args = ProfileArgs {
        memory: true,
        ..sample_args()
    };
    let out = run_profile(&args).unwrap();
    assert!(out.contains("Memory Timeline"), "timeline section missing: {}", out);
    assert!(out.contains("Peak:"), "peak line missing: {}", out);
    // Per-op table still present — --memory is additive.
    assert!(out.contains("NSL Predictive Profile"));
}

#[test]
fn bad_gpu_errors_with_available_list() {
    let args = ProfileArgs {
        target: "nonsuch".into(),
        ..sample_args()
    };
    let err = run_profile(&args).unwrap_err();
    assert!(err.contains("nonsuch"), "err should mention bad target: {err}");
    assert!(
        err.to_lowercase().contains("available"),
        "err should list available: {err}"
    );
}
