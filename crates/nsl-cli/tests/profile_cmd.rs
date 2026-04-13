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
    }
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
