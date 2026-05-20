//! Pins the bench binary's stdout-line format against §5.2 of
//! `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`.
//!
//! Shell scripts (`scripts/measure_tier_b_m{2,6}.sh`) and any future CI
//! consumer grep for the `tier_b_bench_result:` prefix and split on `:`.
//! Renaming or removing a key is a breaking change requiring a versioned
//! prefix (`tier_b_bench_result_v2:`).

#![cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]

use nsl_codegen::bin::bench::output::{emit_result, ResultLine};

#[test]
fn emit_result_format_matches_spec_section_5_2() {
    let line = emit_result(&ResultLine {
        fixture: "gate_4096".into(),
        tier_b_on: true,
        median_us: 234.567,
        n: 100,
        skip_ratio: 0.487,
        seed: 42,
    });
    assert_eq!(
        line,
        "tier_b_bench_result:fixture=gate_4096:tier_b=on:median_us=234.567:n=100:skip_ratio=0.487:seed=42"
    );
}

#[test]
fn emit_result_tier_b_off_label() {
    let line = emit_result(&ResultLine {
        fixture: "gate_4096".into(),
        tier_b_on: false,
        median_us: 257.0,
        n: 100,
        skip_ratio: 0.0,
        seed: 42,
    });
    assert!(
        line.contains(":tier_b=off:"),
        "missing tier_b=off label in {line}"
    );
    // Rust's default {} formatter for `f64` prints 0.0 as "0" — both shapes
    // are acceptable per the spec, so accept either.
    assert!(
        line.contains(":skip_ratio=0:") || line.contains(":skip_ratio=0.0:"),
        "missing zero skip_ratio key in {line}"
    );
}

#[test]
fn emit_result_seed_propagates() {
    let line = emit_result(&ResultLine {
        fixture: "gate_4096".into(),
        tier_b_on: true,
        median_us: 100.0,
        n: 100,
        skip_ratio: 0.5,
        seed: 1337,
    });
    assert!(
        line.ends_with(":seed=1337"),
        "seed not at end of line: {line}"
    );
}

#[test]
fn emit_result_starts_with_prefix() {
    let line = emit_result(&ResultLine {
        fixture: "any".into(),
        tier_b_on: false,
        median_us: 1.0,
        n: 1,
        skip_ratio: 0.0,
        seed: 0,
    });
    assert!(
        line.starts_with("tier_b_bench_result:"),
        "bad prefix: {line}"
    );
}
