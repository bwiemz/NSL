//! Reference implementation self-test.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §5.
//!
//! Fixtures are NSL-reference self-anchored (spec §5.1 fallback path):
//! bitnet.cpp is not buildable on the Windows MSVC implementation
//! platform. The fixture JSON pins the reference's outputs on 10
//! canonical edge cases; future kernel changes (Tasks 4-7) validate
//! against the JSON, NOT the reference. Re-anchoring against bitnet.cpp
//! on Linux is a planned follow-on.
//!
//! Two tests live in this file:
//!
//! - [`generate_fixtures`] — `#[ignore]`'d. Run manually when the
//!   reference is deliberately changed to refresh the committed JSON.
//! - [`reference_matches_committed_fixtures`] — the GATING test for
//!   Tasks 4-7. Asserts that running the reference on each fixture's
//!   input reproduces the committed expected outputs.

#[path = "bitnet_reference_impl.rs"]
mod reference;

use reference::{forward_reference, make_f10_inputs};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
struct Fixture {
    name: String,
    input: FixtureInput,
    expected: FixtureExpected,
}

#[derive(Debug, Serialize, Deserialize)]
struct FixtureInput {
    activations: Vec<Vec<f32>>,
    weights: Vec<Vec<i8>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FixtureExpected {
    absmax_scale: Vec<f32>,
    quantized_acts: Vec<Vec<i8>>,
    gemm_output: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FixtureFile {
    source: String,
    source_revision: String,
    captured_at: String,
    notes: String,
    fixtures: Vec<Fixture>,
}

fn current_commit_sha() -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8(out.stdout)
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| String::from("unknown-utf8-decode-failed")),
        _ => String::from("unknown-git-failed"),
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bitnet_reference_outputs.json")
}

/// Helper: repeat the same `[hidden_dim]`-long row `hidden_dim` times to
/// build a square `[hidden_dim, out_dim]` weight matrix. Used for the
/// "uniform weight pattern" fixtures (f02..f07 in the brief's table).
fn repeat_weight_row(row: &[i8], times: usize) -> Vec<Vec<i8>> {
    (0..times).map(|_| row.to_vec()).collect()
}

fn build_fixture_inputs() -> Vec<(String, Vec<Vec<f32>>, Vec<Vec<i8>>)> {
    let mut out: Vec<(String, Vec<Vec<f32>>, Vec<Vec<i8>>)> = Vec::new();

    // f01_zero_row: all-zero activation row; weight is alternating ±1.
    // Expected: absmax_scale=0, q_acts all 0, gemm output all 0.
    out.push((
        "f01_zero_row".to_string(),
        vec![vec![0.0, 0.0, 0.0, 0.0]],
        repeat_weight_row(&[1, -1, 1, -1], 4),
    ));

    // f02_uniform_pos: all activations = +2.
    out.push((
        "f02_uniform_pos".to_string(),
        vec![vec![2.0, 2.0, 2.0, 2.0]],
        repeat_weight_row(&[1, -1, 1, -1], 4),
    ));

    // f03_uniform_neg: all activations = -3 with sign-flipped weight pattern.
    out.push((
        "f03_uniform_neg".to_string(),
        vec![vec![-3.0, -3.0, -3.0, -3.0]],
        repeat_weight_row(&[-1, 1, -1, 1], 4),
    ));

    // f04_mixed_sign: classic mixed-sign single row.
    out.push((
        "f04_mixed_sign".to_string(),
        vec![vec![1.5, -2.5, 0.5, -1.0]],
        repeat_weight_row(&[1, 0, -1, 1], 4),
    ));

    // f05_outlier: one large value dominates the row's absmax — checks
    // that the smaller values quantize to small int8 magnitudes.
    out.push((
        "f05_outlier".to_string(),
        vec![vec![0.1, 0.2, 10.0, 0.15]],
        repeat_weight_row(&[1, -1, 0, 1], 4),
    ));

    // f06_multi_row: four-row batch covering positive, negative, uniform,
    // and small-magnitude rows in one fixture.
    out.push((
        "f06_multi_row".to_string(),
        vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-1.0, -2.0, -3.0, -4.0],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![-0.1, 0.1, -0.1, 0.1],
        ],
        repeat_weight_row(&[1, -1, 0, 1], 4),
    ));

    // f07_zero_weight_col: all-zero weight matrix — output must be all 0.
    out.push((
        "f07_zero_weight_col".to_string(),
        vec![vec![1.0, 2.0, 3.0, 4.0]],
        repeat_weight_row(&[0, 0, 0, 0], 4),
    ));

    // f08_alternating: 8-wide alternating ±1 acts & weights.
    out.push((
        "f08_alternating".to_string(),
        vec![vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]],
        repeat_weight_row(&[1, 1, -1, -1, 1, -1, 1, -1], 8),
    ));

    // f09_unit_weights: identity weight matrix — output[r][c] should match
    // a dequantized version of activation[r][c].
    out.push((
        "f09_unit_weights".to_string(),
        vec![vec![1.0, 2.0, 3.0, 4.0]],
        vec![
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
        ],
    ));

    // f10_large_hidden: 128-element row, 128×4 ternary weights;
    // deterministic LCG seed=42 for cross-platform reproducibility.
    let (f10_acts, f10_weights) = make_f10_inputs();
    out.push(("f10_large_hidden".to_string(), vec![f10_acts], f10_weights));

    out
}

fn run_reference_on_fixture(name: &str, acts: &[Vec<f32>], weights: &[Vec<i8>]) -> Fixture {
    let (scales, q_acts, output) = forward_reference(acts, weights);
    Fixture {
        name: name.to_string(),
        input: FixtureInput {
            activations: acts.to_vec(),
            weights: weights.to_vec(),
        },
        expected: FixtureExpected {
            absmax_scale: scales,
            quantized_acts: q_acts,
            gemm_output: output,
        },
    }
}

/// Fixture generator — run manually when the reference impl is
/// deliberately changed to refresh the committed JSON.
///
/// ```text
/// cargo test -p nsl-codegen --test bitnet_reference \
///     -- --ignored generate_fixtures --nocapture
/// ```
///
/// `source_revision` is auto-populated via `git rev-parse HEAD`; if git is
/// unavailable a recognizable fallback string is written instead.
#[test]
#[ignore = "fixture generator — run manually when regenerating after deliberate ref-impl changes"]
fn generate_fixtures() {
    let inputs = build_fixture_inputs();
    let fixtures: Vec<Fixture> = inputs
        .into_iter()
        .map(|(name, acts, w)| run_reference_on_fixture(&name, &acts, &w))
        .collect();
    let file = FixtureFile {
        source: String::from("nsl-reference-self-anchor"),
        source_revision: current_commit_sha(),
        captured_at: String::from("2026-05-11"),
        notes: String::from(
            "Generated by NSL's bitnet_reference_impl.rs forward_reference; \
             bitnet.cpp unbuildable on Windows MSVC implementation platform; \
             planned re-anchor against bitnet.cpp on Linux follow-on. \
             Activation quant uses absmax per b1.58 paper (spec §4.2 \
             absmean correction).",
        ),
        fixtures,
    };
    let json = serde_json::to_string_pretty(&file).expect("serialization failed");
    let path = fixture_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create fixtures dir");
    }
    fs::write(&path, &json).expect("write fixtures failed");
    println!(
        "Wrote {} fixtures to {}",
        file.fixtures.len(),
        path.display()
    );
}

/// GATING test for Tasks 4-7.
///
/// Asserts the reference implementation reproduces the committed
/// fixture outputs. Bit-exact on `quantized_acts` (deterministic
/// integer arithmetic); 1e-6 relative or 1e-7 absolute tolerance on
/// FP32 scales and GEMM outputs — both sides run identical f32
/// arithmetic in the same order, so the tolerance is defensive padding
/// against potential rust-toolchain rounding-mode drift, not algorithmic.
#[test]
fn reference_matches_committed_fixtures() {
    let path = fixture_path();
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Could not read {}: {e}", path.display()));
    let file: FixtureFile = serde_json::from_str(&text).expect("invalid fixture JSON");
    println!(
        "Validating reference against {} ({})",
        file.source, file.source_revision
    );
    assert_eq!(
        file.fixtures.len(),
        10,
        "expected 10 fixtures, got {}",
        file.fixtures.len()
    );
    for fixture in &file.fixtures {
        let (scales, q_acts, output) =
            forward_reference(&fixture.input.activations, &fixture.input.weights);

        // FP32 tolerance: 1e-6 relative OR 1e-7 absolute (for near-zero outputs where
        // relative error explodes; absolute tolerance gates those).
        assert_eq!(
            scales.len(),
            fixture.expected.absmax_scale.len(),
            "Fixture {}: scale row count mismatch",
            fixture.name
        );
        for (i, (&actual, &expected)) in scales
            .iter()
            .zip(fixture.expected.absmax_scale.iter())
            .enumerate()
        {
            let abs_diff = (actual - expected).abs();
            let rel_diff = abs_diff / expected.abs().max(1e-30);
            assert!(
                rel_diff <= 1e-6 || abs_diff < 1e-7,
                "Fixture {}: scale[{i}] mismatch: actual={actual} expected={expected}",
                fixture.name
            );
        }

        // Quantized acts: bit-exact int8.
        assert_eq!(
            q_acts, fixture.expected.quantized_acts,
            "Fixture {}: quantized_acts mismatch",
            fixture.name
        );

        // FP32 tolerance: 1e-6 relative OR 1e-7 absolute (for near-zero outputs where
        // relative error explodes; absolute tolerance gates those).
        assert_eq!(
            output.len(),
            fixture.expected.gemm_output.len(),
            "Fixture {}: output row count mismatch",
            fixture.name
        );
        for (r, (actual_row, expected_row)) in output
            .iter()
            .zip(fixture.expected.gemm_output.iter())
            .enumerate()
        {
            assert_eq!(
                actual_row.len(),
                expected_row.len(),
                "Fixture {}: output row {r} length mismatch",
                fixture.name
            );
            for (c, (&actual, &expected)) in actual_row.iter().zip(expected_row.iter()).enumerate()
            {
                let abs_diff = (actual - expected).abs();
                let rel_diff = abs_diff / expected.abs().max(1e-30);
                assert!(
                    rel_diff <= 1e-6 || abs_diff < 1e-7,
                    "Fixture {}: gemm_output[{r}][{c}] mismatch: \
                     actual={actual} expected={expected}",
                    fixture.name
                );
            }
        }
    }
    println!("All {} fixtures validated.", file.fixtures.len());
}
