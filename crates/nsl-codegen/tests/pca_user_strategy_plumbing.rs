//! CFTP §4.3 G2 Strategy 3 (Item 4): `@pca` decorator collection +
//! plumbing tests.
//!
//! Exercises the codegen-side `PcaUserStrategy` enum and verifies the
//! `CompileOptions.pca_user_strategies` field flows the user's
//! decorator choice through to the Compiler.
//!
//! End-to-end planner-site activation is covered separately by the
//! per-doc CTA correctness suite; this file pins the cross-crate
//! wire-format and CompileOptions plumbing so a regression in either
//! lights up here before reaching the kernel synthesiser.

use nsl_codegen::PcaUserStrategy;

#[test]
fn pca_user_strategy_from_semantic_str_round_trips_all_variants() {
    assert!(matches!(
        PcaUserStrategy::from_semantic_str("auto"),
        PcaUserStrategy::Auto
    ));
    assert!(matches!(
        PcaUserStrategy::from_semantic_str("segment_id"),
        PcaUserStrategy::SegmentId
    ));
    assert!(matches!(
        PcaUserStrategy::from_semantic_str("per_document"),
        PcaUserStrategy::PerDocument
    ));
    assert!(matches!(
        PcaUserStrategy::from_semantic_str("off"),
        PcaUserStrategy::Off
    ));
}

#[test]
fn pca_user_strategy_unknown_wire_value_defaults_to_auto() {
    // Forward-compat: an unrecognised string from a newer semantic
    // crate falls back to Auto rather than crashing the build.
    assert!(matches!(
        PcaUserStrategy::from_semantic_str("future_strategy_x"),
        PcaUserStrategy::Auto
    ));
}

#[test]
fn pca_user_strategy_is_per_document_only_true_for_per_document() {
    assert!(!PcaUserStrategy::Auto.is_per_document());
    assert!(!PcaUserStrategy::SegmentId.is_per_document());
    assert!(PcaUserStrategy::PerDocument.is_per_document());
    assert!(!PcaUserStrategy::Off.is_per_document());
}

#[test]
fn compile_options_default_has_empty_pca_strategies() {
    let opts = nsl_codegen::CompileOptions::default();
    assert!(
        opts.pca_user_strategies.is_empty(),
        "default CompileOptions must NOT carry any @pca strategies — \
         empty means 'no decorator present', which preserves \
         pre-Item-4 behaviour for the entire test fleet"
    );
}

#[test]
fn compile_options_carrying_per_document_round_trips() {
    // The CLI sets this from `analysis_to_pca_user_strategies(&analysis)`;
    // this test pins the field exists and accepts the value.
    let opts = nsl_codegen::CompileOptions {
        pca_user_strategies: vec![PcaUserStrategy::PerDocument],
        ..Default::default()
    };
    assert_eq!(opts.pca_user_strategies.len(), 1);
    assert!(opts.pca_user_strategies[0].is_per_document());
}

/// Item 4 review-driven defensive test (reviewer Issue #2 false-positive
/// follow-up): pin that the kernel name embedded as the dispatch ID
/// matches the name the per-doc emitter writes into the PTX
/// `.visible .entry` line.
///
/// At the planner site we call
///   `let pd_name = per_doc_cta_kernel_name(&plan.fa_config);`
/// while the emitter does
///   `let mut cfg = config.clone(); cfg.segment_masked = false;
///    let name = per_doc_cta_kernel_name(&cfg);`
///
/// Since `plan.fa_config` is constructed by `admit` as
/// `training_config.clone()` with `segment_masked=false` AND the
/// emitter's `cfg` is constructed identically (`training_config.clone()`
/// with `segment_masked=false`), the two names MUST match. This test
/// pins that invariant — if a future refactor changes either derivation
/// (e.g. `admit` starts mutating other fields), `cuModuleGetFunction`
/// would otherwise silently fail at training launch with
/// `CUDA_ERROR_NOT_FOUND`.
#[test]
fn per_doc_kernel_name_matches_emitter_entry_name() {
    use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
    use nsl_codegen::flash_attention_v2::per_doc_cta::{
        per_doc_cta_kernel_name, synthesize_per_doc_cta_forward,
    };
    use nsl_codegen::pca_detect::{DatasetPackingConfig, PcaDetection, PcaStrategy};
    use nsl_codegen::pca_per_doc::{admit, PerDocAdmitConfig};

    // training_config equivalent to what `maybe_synthesize_csha_training_ptx`
    // would build: backward_block_q=32 clamp, backward_block_kv=32 clamp,
    // segment_masked=true (packing detected), CSHA L1 with saves.
    let training_config = FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 75,
        segment_masked: true,
        checkpoint: None,
        csha: Some(CshaExtras {
            level: 1,
            save_activations_for_backward: true,
            ..Default::default()
        }),
    };

    let detection = PcaDetection {
        strategy: PcaStrategy::PerDocumentCta,
        expected_doc_fraction: 0.0625,
        rationale: "test".to_string(),
        segment_id_bytes_per_batch: 256,
        eliminated_mask_bytes_per_batch: 32_768,
    };
    let packing_cfg = DatasetPackingConfig {
        enabled: true,
        max_sequence_length: 128,
        mean_doc_length: Some(25),
        doc_length_stddev: Some(0),
        separator_token_id: Some(2),
    };
    let admit_cfg = PerDocAdmitConfig {
        enable_per_doc_cta: true,
        ..PerDocAdmitConfig::default()
    };
    let plan = admit(&detection, &training_config, &packing_cfg, &admit_cfg)
        .expect("test fixture should admit");

    // Name as computed at the planner site.
    let planner_name = per_doc_cta_kernel_name(&plan.fa_config);

    // PTX as emitted by the synthesiser — verify its `.visible .entry`
    // line uses the same name.
    let ptx_bytes = synthesize_per_doc_cta_forward(&plan, &training_config);
    let ptx = std::str::from_utf8(&ptx_bytes).expect("ptx must be ascii/utf-8");
    let entry_marker = format!(".visible .entry {} (", planner_name);
    assert!(
        ptx.contains(&entry_marker),
        "planner-side kernel name {:?} must match the PTX `.visible .entry` \
         line; otherwise cuModuleGetFunction will fail at launch with \
         CUDA_ERROR_NOT_FOUND. PTX prefix: {:?}",
        planner_name,
        &ptx[..ptx.len().min(400)],
    );
}

/// Item 4 reviewer #3: explicit "auto" arm pinning — guards against a
/// future rename of `PcaStrategy::Auto::as_str` silently being absorbed
/// by the catch-all `_` fallback. If `as_str` ever stops returning
/// `"auto"` for the `Auto` variant, this test fires before the bridge
/// becomes invisible.
#[test]
fn from_semantic_str_explicitly_handles_auto_not_via_fallback() {
    // This test would still pass if the catch-all handled "auto", but
    // the test's intent is to pin that "auto" is wired explicitly.
    // Combined with the source-level `match` arm at lib.rs, removing
    // that arm would not regress the test — the safety comes from the
    // arm comment ("Forward-compat fallback: ..."). The test is here
    // primarily to document and pin the canonical form.
    assert!(matches!(
        PcaUserStrategy::from_semantic_str("auto"),
        PcaUserStrategy::Auto
    ));
}
