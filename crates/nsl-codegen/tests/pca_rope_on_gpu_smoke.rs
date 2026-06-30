//! CFTP §4.3 — Layer 3 on-GPU smoke test scaffold.
//!
//! Launches a 2-doc packed config on a real GPU and compares against the
//! Layer 2 CPU reference. Gated on `cfg(feature = "cuda")` AND the
//! `NSL_GPU_SMOKE` env var so it doesn't run in default CI.
//!
//! The current state: the full launch + comparison is gated on a future
//! on-GPU harness hookup (the codegen path is ready but the dataloader
//! wiring for doc_starts is documented as a follow-on per T11). This
//! file lands as a scaffold so the harness target is clear.

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;
use nsl_runtime::packing::build_segment_ids_and_doc_starts;

fn smoke_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        segment_masked: true,
        csha: Some(CshaExtras::default()),
    }
}

#[test]
fn two_doc_packed_e2e_scaffold() {
    // Skip if NSL_GPU_SMOKE is not set (default — keeps CI quiet).
    if std::env::var("NSL_GPU_SMOKE").is_err() {
        eprintln!("skipped — set NSL_GPU_SMOKE=1 to run on-GPU smoke tests");
        return;
    }

    let cfg = smoke_config();
    let ptx = synthesize_flash_attention_ptx_v2(&cfg);
    assert!(!ptx.is_empty(), "PTX must synthesize");

    // Allocate 2-doc packed batch (this row's doc_lengths = [4, 4]).
    let (segment_ids, doc_starts) = build_segment_ids_and_doc_starts(&[4u32, 4u32]);
    assert_eq!(segment_ids.len(), 8);
    assert_eq!(doc_starts[0], 0);
    assert_eq!(doc_starts[1], 4);
    assert_eq!(doc_starts[2], 8);
    assert_eq!(doc_starts[3], -1);

    // Full launch + numerical comparison vs CPU reference is gated on
    // dataloader-to-FA2-call-site wiring (T11 documented as deferred —
    // bc84acf5). When that lands, this scaffold expands to a real launch
    // and a `assert!((gpu_out - cpu_ref).abs().max() < 5e-3)` style check.
    eprintln!("on-GPU smoke scaffold ready; full launch pending T11 follow-on");
}

// NOTE (A-4 Task 4 — doc_starts null-guard, Test 4): Numerical GPU validation
// of the doc_starts null-guard under `rope_q=true` (when doc_starts_ptr==0,
// emit_doc_starts_smem_load writes doc_start=0 for every position → standard
// non-packed RoPE positions) is SHIPPED in
// `pca_tier_a_forward_correctness::rope_q_forward_doc_starts_null_equals_doc_zero_baseline`
// (Item 3 / PR #274). That test launches the kernel twice — once with
// doc_starts=[0,0,...,0] and once with doc_starts_ptr=0 — and asserts
// bit-exact output equivalence (max_abs == 0.0). The backward equivalent
// ships in `pca_tier_a_backward_correctness::doc_starts_null_equals_standard_positions_backward`
// at 5e-3 tolerance.
//
// Structural validation (snapshot) also covers the doc_starts null-guard:
// the `emit_doc_starts_smem_load` guard uses the identical skip-before-dereference
// mechanism (`@%p_doc_null bra ...NULL_DOC_LD` / `mov.u32 %doc_start, 0`) as the
// seg_ids guard, which IS numerically validated end-to-end by A-4 Tests 1 and 2
// (forward + backward, both confirming masked+null == unmasked within tolerance).
// The two guards were added in the same A-3 commit; any structural regression in
// either would break the snapshot tests.

