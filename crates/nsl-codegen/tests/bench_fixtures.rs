//! Gate fixture registry tests for the `nsl-codegen-bench` binary.
//!
//! Spec §4.1 of `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`
//! pins the gate fixture dimensions: segment-masked causal, seq_len=4096,
//! head_dim=64, batch=4, block 64×64, target_sparsity=50%, sm_120.

#![cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]

use nsl_codegen::bin::bench::fixtures;

#[test]
fn gate_fixture_dims_match_spec_section_4_1() {
    let f = fixtures::lookup("gate_4096").expect("gate fixture exists");
    assert_eq!(f.config.block_q, 64, "block_q");
    assert_eq!(f.config.block_kv, 64, "block_kv");
    assert_eq!(f.config.head_dim, 64, "head_dim");
    assert!(f.config.causal, "causal");
    assert!(f.config.segment_masked, "segment_masked");
    assert_eq!(f.config.gpu_sm, 120, "gpu_sm sm_120 (Blackwell)");
    assert_eq!(f.seq_len, 4096, "seq_len");
    assert_eq!(f.batch, 4, "batch");
    assert!(
        (f.target_sparsity - 0.5).abs() < 1e-6,
        "target_sparsity 50%, got {}",
        f.target_sparsity
    );
}

#[test]
fn unknown_fixture_returns_none() {
    assert!(fixtures::lookup("nonexistent").is_none());
}

#[test]
fn registry_has_at_least_the_gate_fixture() {
    // Sentinel against accidental empty-registry regressions. The
    // sensitivity + parity fixtures land in B1.5-4 / B1.5-3 and will
    // increase this count; this test stays sound across that growth.
    assert!(
        fixtures::lookup("gate_4096").is_some(),
        "gate fixture must always be present"
    );
}
