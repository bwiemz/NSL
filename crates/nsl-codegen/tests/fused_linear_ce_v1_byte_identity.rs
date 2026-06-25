//! Sprint-3 regression gate — the v1 single-CTA forward + backward PTX
//! emission MUST stay byte-identical to pre-Sprint-3 for every shape
//! that routes through `is_large_vocab() == false`. The Sprint-3 routing
//! split in `synthesize_fused_linear_ce_ptx` short-circuits to the v1
//! emitter for `vocab_size <= LARGE_VOCAB_THRESHOLD` (8192); this test
//! pins down that the bytes coming out of that branch have not drifted.
//!
//! If a future refactor changes the v1 emitter (even cosmetically), this
//! test catches it before it can silently change observed numerical
//! results on the existing v1 GPU test (`fused_linear_ce_numerical.rs`).
//!
//! Snapshots live in `tests/snapshots/fused_linear_ce_v1_byte_identity__*.snap`.

use nsl_codegen::fused_linear_ce::{
    Dtype, FusedLinearCEConfig,
    synthesize_fused_linear_ce_backward_ptx, synthesize_fused_linear_ce_ptx,
};

/// Strip the trailing null byte that the public synthesizers append for
/// `cuModuleLoadData`. The snapshot tests below pin the *kernel bytes*
/// (the v1 emitter contract), not the null-termination contract — that
/// contract is pinned separately by the unit tests in `fused_linear_ce.rs`
/// (`*_ptx_is_null_terminated_*`). Stripping the null here preserves the
/// snapshot semantics across the null-termination change.
fn strip_trailing_nul(ptx: &[u8]) -> &[u8] {
    assert_eq!(
        ptx.last(),
        Some(&0u8),
        "synthesizer contract: PTX must be null-terminated for cuModuleLoadData",
    );
    &ptx[..ptx.len() - 1]
}

/// The exact config used by `tests/fused_linear_ce_numerical.rs` — the v1
/// GPU numerical test. Capturing this specific shape's PTX gates that the
/// GPU test will continue to see the same kernel bytes.
fn v1_test_cfg() -> FusedLinearCEConfig {
    FusedLinearCEConfig {
        vocab_size: 4096,
        hidden_size: 128,
        seq_len: 64,
        batch_size: 2,
        vocab_tile: 1024,
        gpu_sm: 80,
        dtype: Dtype::F32,
        ignore_index: -100,
        max_vocab_v1: 8192,
    }
}

#[test]
fn v1_forward_ptx_at_v4096_matches_snapshot() {
    let cfg = v1_test_cfg();
    cfg.validate().unwrap();
    assert!(!cfg.is_large_vocab(), "this config MUST route to v1 path");

    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let s = std::str::from_utf8(strip_trailing_nul(&ptx)).expect("PTX is ASCII");
    insta::assert_snapshot!("v1_forward_v4096_h128_t1024", s);
}

#[test]
fn v1_backward_ptx_at_v4096_matches_snapshot() {
    let cfg = v1_test_cfg();
    let ptx = synthesize_fused_linear_ce_backward_ptx(&cfg);
    let s = std::str::from_utf8(strip_trailing_nul(&ptx)).expect("PTX is ASCII");
    insta::assert_snapshot!("v1_backward_v4096_h128_t1024", s);
}

/// At the routing threshold (vocab=8192 exactly) the path MUST still be
/// v1 — `is_large_vocab` uses `>`, not `>=`. Pin this so a future
/// boundary tweak doesn't silently flip the threshold.
#[test]
fn v1_forward_ptx_at_threshold_v8192_matches_snapshot() {
    let cfg = FusedLinearCEConfig {
        vocab_size: 8192,
        hidden_size: 128,
        seq_len: 32,
        batch_size: 1,
        vocab_tile: 1024,
        gpu_sm: 80,
        dtype: Dtype::F32,
        ignore_index: -100,
        max_vocab_v1: 8192,
    };
    cfg.validate().unwrap();
    assert!(!cfg.is_large_vocab(), "threshold value MUST remain v1 path");

    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let s = std::str::from_utf8(strip_trailing_nul(&ptx)).expect("PTX is ASCII");
    insta::assert_snapshot!("v1_forward_v8192_at_threshold", s);
}

/// One step past the threshold (vocab=8192+128) MUST route to large-vocab.
/// Mirror gate: the small-vocab snapshot above + this large-vocab one
/// together pin down both sides of the routing predicate.
#[test]
fn large_vocab_routing_just_above_threshold() {
    let cfg = FusedLinearCEConfig {
        vocab_size: 8192 + 128,
        hidden_size: 128,
        seq_len: 32,
        batch_size: 1,
        vocab_tile: 128,
        gpu_sm: 80,
        dtype: Dtype::F32,
        ignore_index: -100,
        max_vocab_v1: nsl_codegen::fused_linear_ce::MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().unwrap();
    assert!(cfg.is_large_vocab(), "above-threshold MUST route to large-vocab");

    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let s = std::str::from_utf8(strip_trailing_nul(&ptx)).expect("PTX is ASCII");
    // Header MUST appear exactly once (single module containing both kernels).
    assert_eq!(s.matches(".version 7.0").count(), 1);
    // Both kernels MUST be present.
    assert!(s.contains(&cfg.large_partials_kernel_name()));
    assert!(s.contains(&cfg.large_finalize_kernel_name()));
}
