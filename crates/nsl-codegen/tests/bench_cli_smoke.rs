//! Smoke test that the bench binary's CLI parser accepts the documented surface.
//!
//! The bench binary is gated behind the `cuda` + `debug_kernel_instrumentation`
//! Cargo features. The CLI parser module lives under `nsl_codegen::bin::bench::cli`
//! so it is reachable both from `src/bin/bench.rs` and from integration tests.

#![cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]

#[test]
fn cli_parses_required_args() {
    let args = vec!["bench", "--fixture", "gate_4096", "--tier-b", "on"];
    let parsed = nsl_codegen::bin::bench::cli::parse_from(&args).expect("parses");
    assert_eq!(parsed.fixture, "gate_4096");
    assert_eq!(parsed.tier_b, nsl_codegen::bin::bench::cli::TierB::On);
    assert_eq!(parsed.seed, 42);
    assert_eq!(parsed.iterations, 100);
    assert!(!parsed.emit_time_only);
}

#[test]
fn cli_rejects_missing_fixture() {
    let args = vec!["bench", "--tier-b", "on"];
    assert!(nsl_codegen::bin::bench::cli::parse_from(&args).is_err());
}

#[test]
fn cli_rejects_missing_tier_b() {
    let args = vec!["bench", "--fixture", "gate_4096"];
    assert!(nsl_codegen::bin::bench::cli::parse_from(&args).is_err());
}

#[test]
fn cli_parses_optional_args() {
    let args = vec![
        "bench",
        "--fixture",
        "gate_4096",
        "--tier-b",
        "off",
        "--seed",
        "7",
        "--iterations",
        "25",
        "--emit-time-only",
    ];
    let parsed = nsl_codegen::bin::bench::cli::parse_from(&args).expect("parses");
    assert_eq!(parsed.tier_b, nsl_codegen::bin::bench::cli::TierB::Off);
    assert_eq!(parsed.seed, 7);
    assert_eq!(parsed.iterations, 25);
    assert!(parsed.emit_time_only);
}

#[test]
fn cli_rejects_unknown_flag() {
    let args = vec!["bench", "--fixture", "gate_4096", "--tier-b", "on", "--bogus"];
    assert!(nsl_codegen::bin::bench::cli::parse_from(&args).is_err());
}

#[test]
fn cli_rejects_invalid_tier_b_value() {
    let args = vec!["bench", "--fixture", "gate_4096", "--tier-b", "maybe"];
    assert!(nsl_codegen::bin::bench::cli::parse_from(&args).is_err());
}
