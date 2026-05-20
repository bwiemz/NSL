//! Numerical fixture test for fused quantized_ternary_gemm.
//!
//! Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §4.3.
//! Uses the Task 3 reference impl as the math oracle; the 10 fixtures from
//! tests/fixtures/bitnet_reference_outputs.json provide expected outputs.
//!
//! Note: this test does not execute the PTX emitted by
//! `quantized_ternary_gemm::emit` — actual PTX-runtime validation is Task 10's
//! merge gate. It instead pins the math the emitter encodes to the same
//! committed fixtures gated by Task 3's reference test, so structural drift
//! of the math (e.g. accidentally introducing absmean instead of absmax, or
//! changing the /127.0 dequant factor) is caught here at unit-test scale.

#[path = "bitnet_reference_impl.rs"]
mod reference;

use reference::forward_reference;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    input: FixtureInput,
    expected: FixtureExpected,
}

#[derive(Debug, Deserialize)]
struct FixtureInput {
    activations: Vec<Vec<f32>>,
    weights: Vec<Vec<i8>>,
}

#[derive(Debug, Deserialize)]
struct FixtureExpected {
    absmax_scale: Vec<f32>,
    quantized_acts: Vec<Vec<i8>>,
    gemm_output: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    fixtures: Vec<Fixture>,
}

#[test]
fn fused_quantized_ternary_gemm_matches_all_10_fixtures() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bitnet_reference_outputs.json");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Could not read {}: {e}", path.display()));
    let file: FixtureFile = serde_json::from_str(&text).expect("invalid fixture JSON");

    assert_eq!(
        file.fixtures.len(),
        10,
        "expected 10 fixtures, got {}",
        file.fixtures.len()
    );

    for fixture in &file.fixtures {
        let (scales, q_acts, output) =
            forward_reference(&fixture.input.activations, &fixture.input.weights);

        // FP32 tolerance: 1e-6 relative OR 1e-7 absolute (for near-zero outputs).
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
                "Fixture {}: scale[{i}] {actual} vs {expected}",
                fixture.name
            );
        }

        // Quantized acts: bit-exact int8.
        assert_eq!(
            q_acts, fixture.expected.quantized_acts,
            "Fixture {}: quantized_acts mismatch",
            fixture.name
        );

        // GEMM output: FP32 tolerance gate.
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
                    "Fixture {}: gemm[{r}][{c}] {actual} vs {expected}",
                    fixture.name
                );
            }
        }
    }
}
