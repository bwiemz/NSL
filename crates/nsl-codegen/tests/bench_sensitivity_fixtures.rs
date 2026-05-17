#![cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
//! Tests for Task B1.5-4: sensitivity-tier fixtures.
//!
//! Spec §4.3 of `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`
//! defines three sensitivity fixtures at sparsities {10%, 50%, 90%} that
//! share the gate fixture's other dimensions and characterize the
//! sparsity → benefit curve shape.
//!
//! §4.3.1: sensitivity_50 is structurally identical to the gate fixture
//! (redundancy / cross-check).

use nsl_codegen::bin::bench::fixtures;

#[test]
fn sensitivity_fixtures_share_gate_dims_differ_in_sparsity() {
    let gate = fixtures::lookup("gate_4096").expect("gate fixture");
    for (name, sparsity) in [
        ("sensitivity_10", 0.1),
        ("sensitivity_50", 0.5),
        ("sensitivity_90", 0.9),
    ] {
        let f = fixtures::lookup(name).expect(name);
        assert_eq!(f.config.block_q, gate.config.block_q);
        assert_eq!(f.config.block_kv, gate.config.block_kv);
        assert_eq!(f.config.head_dim, gate.config.head_dim);
        assert!(f.config.segment_masked);
        assert!(f.config.causal);
        assert_eq!(f.seq_len, gate.seq_len);
        assert_eq!(f.batch, gate.batch);
        assert!(
            (f.target_sparsity - sparsity).abs() < 1e-6,
            "{name} sparsity {} != expected {}",
            f.target_sparsity,
            sparsity
        );
    }
}

#[test]
fn sensitivity_50_matches_gate_fixture_structurally() {
    let s50 = fixtures::lookup("sensitivity_50").unwrap();
    let gate = fixtures::lookup("gate_4096").unwrap();
    assert!(
        (s50.target_sparsity - gate.target_sparsity).abs() < 1e-6,
        "sensitivity_50 is structurally identical to gate fixture per design §4.3.1"
    );
    assert_eq!(s50.config.block_q, gate.config.block_q);
    assert_eq!(s50.config.block_kv, gate.config.block_kv);
    assert_eq!(s50.config.head_dim, gate.config.head_dim);
    assert_eq!(s50.seq_len, gate.seq_len);
    assert_eq!(s50.batch, gate.batch);
}
