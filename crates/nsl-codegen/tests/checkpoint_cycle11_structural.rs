//! Cycle-11 Task 4 structural probes for the functional substitution of
//! `kv_load::emit_k_suffixed` / `emit_v_suffixed` with
//! `csha_hooks_backward::emit_kv_recompute`.
//!
//! The R0 hard gate at `synthesize_backward_with_recompute` still refuses
//! `@checkpoint(policy="full")` in production. These tests reach the
//! substitution path via the cfg-gated `CheckpointExtras::bypass_r0_for_testing()`
//! builder — available only when the `test-helpers` Cargo feature is
//! enabled (this test target requires the feature; see `Cargo.toml`).
//!
//! Probes:
//!   G3a — when bypass is active, the kv_load skip-labels MUST NOT appear
//!         (the emitters didn't fire).
//!   G3b — the `emit_kv_recompute` entry label + the recompute-namespace
//!         prologue suffix MUST appear (the substitute emitter fired).
//!   G3c — the LAST write to `%k_smem_base` from the recompute matmul
//!         MUST occur before any `V2_BWD_DS_*` (`ds_compute`) label
//!         (ordering: K/V tile fully resident in SMEM before the
//!         per-q-tile ds_compute consumers run).
//!   G3d — the no-decorator path is untouched: it still emits the
//!         kv_load markers and emits NO recompute marker. This is a
//!         sanity gate alongside the existing `fa_v2_snapshots`
//!         25-snapshot byte-identity gate.

use nsl_codegen::flash_attention::{
    CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_v2::synthesize_backward_with_tier;

/// A backward-safe Level-1-fusible CSHA config that satisfies all v1
/// recompute preconditions (block_q == block_kv, no sinks, no PCA-rope_q
/// composition) AND uses the cycle-12 `level1_with_fused_proj` builder
/// so the cycle-12 R3 augmentation (which now also requires
/// `fused_projections=true`) is satisfied. The `r0_bypass=true` field
/// is retained from cycle 11 — it's now a no-op since R0 was retired
/// in cycle 12, but kept for compile/test compatibility.
fn build_bypass_config() -> FlashAttentionConfig {
    let mut csha = CshaExtras::level1_with_fused_proj(1e-6);
    csha.d_model = 32;
    FlashAttentionConfig {
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
        segment_masked: false,
        csha: Some(csha),
        checkpoint: Some(CheckpointExtras::full().bypass_r0_for_testing()),
    }
}

/// G3d baseline: identical fusible config WITHOUT the checkpoint carrier.
/// The dispatch fork in `synthesize_backward_with_tier_b` must take the
/// kv_load branch, leaving no recompute marker behind.
fn build_fusible_no_checkpoint() -> FlashAttentionConfig {
    FlashAttentionConfig {
        checkpoint: None,
        ..build_bypass_config()
    }
}

/// G3c (cycle-12 extension) variant: rope_q=true + Adjacent style so the
/// forward path's `emit_rope_k_epilogue` actually emits the RoPE-K
/// rotation loop (`V2_CSHA_ROPE_K_LOOP_0` / `V2_CSHA_ROPE_K_FUSED_SKIP`).
/// We assert the recompute path also lifts that emission into the
/// backward via the cycle-11 `emit_kv_recompute` step-5 call into
/// `emit_rope_k_epilogue` from `csha_hooks_backward.rs`.
fn build_bypass_config_with_rope() -> FlashAttentionConfig {
    let mut cfg = build_bypass_config();
    cfg.rope_q = true;
    cfg.rope_style = RopeStyle::Adjacent;
    cfg
}

#[test]
fn g3a_kv_load_skip_when_bypass_active() {
    // When R0 is bypassed and the checkpoint carrier is present, the
    // dispatch fork at `mod.rs:1449` MUST suppress the kv_load emitters.
    // Their characteristic skip-labels (`V2_BWD_K_LOAD_SKIP_*`,
    // `V2_BWD_V_LOAD_SKIP_*`) are the most stable single-line signature
    // of the kv_load emitters — if either appears, an emitter fired
    // that shouldn't have.
    let cfg = build_bypass_config();
    let ptx = synthesize_backward_with_tier(&cfg)
        .expect("G3a: bypass-active backward synthesis must succeed");
    assert!(
        !ptx.contains("V2_BWD_K_LOAD_SKIP_MAIN"),
        "G3a: K kv_load emitter fired despite checkpoint=Some(bypass). \
         Substitution dispatch fork at mod.rs:1449 is broken."
    );
    assert!(
        !ptx.contains("V2_BWD_V_LOAD_SKIP_MAIN"),
        "G3a: V kv_load emitter fired despite checkpoint=Some(bypass). \
         Substitution dispatch fork at mod.rs:1449 is broken."
    );
}

#[test]
fn g3b_recompute_label_present_when_bypass_active() {
    // The substitute emitter `emit_kv_recompute` writes a distinctive
    // entry label (`V2_KV_RECOMPUTE_<suffix>`) and emits a prologue
    // recompute with the `_bwd_recompute_<suffix>` namespace suffix.
    // Both must appear when bypass is active.
    let cfg = build_bypass_config();
    let ptx = synthesize_backward_with_tier(&cfg)
        .expect("G3b: bypass-active backward synthesis must succeed");
    assert!(
        ptx.contains("V2_KV_RECOMPUTE_MAIN"),
        "G3b: emit_kv_recompute entry label V2_KV_RECOMPUTE_MAIN missing."
    );
    assert!(
        ptx.contains("_bwd_recompute_"),
        "G3b: recompute-namespace prologue suffix _bwd_recompute_ missing."
    );
}

#[test]
fn g3c_kv_smem_writes_before_ds_compute() {
    // Ordering invariant: every recompute write to %k_smem_base /
    // %v_smem_base must complete BEFORE the first ds_compute consumer
    // label (`V2_BWD_DS_<q_iter>`). The terminating `bar.sync 0;  // KV
    // recompute complete (MAIN)` inside `emit_kv_recompute` is what
    // enforces this at runtime; here we check structurally that:
    //
    //   1. the recompute completion barrier exists; AND
    //   2. it appears in the emitted PTX text BEFORE the first
    //      `V2_BWD_DS_` label (the entry label of the dS pass that
    //      reads `%k_smem_base`).
    //
    // We cannot use `last %k_smem_base` as the witness because
    // `ds_compute` itself READS `%k_smem_base` to fetch the K tile, so
    // the last `%k_smem_base` reference is naturally AFTER `V2_BWD_DS_`
    // by construction (that's exactly the consumption we're enabling).
    let cfg = build_bypass_config();
    let ptx = synthesize_backward_with_tier(&cfg)
        .expect("G3c: bypass-active backward synthesis must succeed");

    let recompute_barrier_pos = ptx
        .find("KV recompute complete (MAIN)")
        .expect("G3c: emit_kv_recompute terminating bar.sync marker missing");
    let first_ds_pos = ptx
        .find("V2_BWD_DS_")
        .expect("G3c: ds_compute label V2_BWD_DS_* missing — backward dispatch broken");
    assert!(
        recompute_barrier_pos < first_ds_pos,
        "G3c: KV recompute barrier @ {} did not precede first V2_BWD_DS_ @ {} — \
         downstream ds_compute would see stale K/V SMEM contents",
        recompute_barrier_pos,
        first_ds_pos
    );
}

#[test]
fn g3c_rope_k_write_ordering_now_refused_pending_path_b_fix() {
    // Formerly `g3c_rope_k_write_between_projection_and_ds_compute`: a
    // cycle-12 structural probe asserting the RoPE-K rotation emission
    // (V2_CSHA_ROPE_K_LOOP_0 / V2_CSHA_ROPE_K_FUSED_SKIP) is lifted into
    // `emit_kv_recompute` between the K projection matmul and the first
    // ds_compute consumer.
    //
    // Phase 1.3 (commit 8f774ad) found that checkpoint + rope_q backward
    // (Path B) — the exact composition `build_bypass_config_with_rope()`
    // builds — has "a known GROSS numerical error ... never-GPU-validated
    // ... tracked for follow-up", and `validate_checkpoint_eligibility`'s
    // R7 was generalized to refuse it unconditionally (not just under
    // segment_masked) rather than let it silently compile to wrong
    // gradients. `bypass_r0_for_testing()` only bypasses the retired R0
    // gate, not R7, so this config is now refused at every production
    // entry point — including this test's `synthesize_backward_with_tier`
    // call — by design (the R7/R3/etc. cascade must fire everywhere).
    //
    // This test now locks in that refusal. The original structural
    // ordering probe (K matmul -> RoPE-K -> recompute-done -> ds_compute)
    // is valuable again once Path B's kv-recompute math is fixed and
    // GPU-validated — re-derive it from this commit's history at that
    // point instead of bypassing R7 here.
    let cfg = build_bypass_config_with_rope();
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("G3c rope: checkpoint+rope_q (Path B) must be refused, not synthesized");
    assert!(
        err.contains("rope_q=true") && err.contains("checkpoint"),
        "G3c rope: refusal message missing expected substrings: {err}"
    );
}

#[test]
fn g5_production_wire_up_smoke() {
    // Cycle-12 T1: when `compile_options.checkpoint_policies` contains
    // a Full entry, the kernel.rs CSHA training-PTX wire-up MUST
    // construct `Some(CheckpointExtras::full())` on the training_config
    // (line 757). The downstream backward synthesis then routes through
    // `synthesize_backward_with_recompute` -> cycle-11
    // `emit_kv_recompute`, producing `V2_KV_RECOMPUTE_MAIN` in the PTX.
    //
    // Driving the full Compiler pipeline end-to-end requires a NSL
    // model with @flash_attention + @train + @checkpoint(policy="full")
    // which is heavy for a structural test. Instead we exercise the
    // structural invariant directly: build a config matching what the
    // wire-up would construct (level1_with_fused_proj + checkpoint=Full)
    // and assert it produces V2_KV_RECOMPUTE_MAIN end-to-end.
    //
    // If this assertion regresses because the wire-up SHAPE changes
    // (e.g. config no longer routes to emit_kv_recompute), the
    // production wire-up is structurally broken regardless of whether
    // the @train+@checkpoint plumbing works.
    let cfg = build_bypass_config();
    let ptx = synthesize_backward_with_tier(&cfg)
        .expect("G5: cycle-12 wire-up shape backward synthesis must succeed");
    assert!(
        ptx.contains("V2_KV_RECOMPUTE_MAIN"),
        "G5: wire-up shape did NOT route into kv-recompute path. \
         Production kernel.rs T1 wire-up is broken — even with \
         CheckpointExtras::full() + level1_with_fused_proj() the \
         dispatch fork failed to substitute kv_load with emit_kv_recompute."
    );
}

#[test]
fn g6_save_activations_forced_when_checkpoint_some() {
    // Cycle-12 T4: when `config.checkpoint.is_some()`, the forward
    // CSHA path MUST stage x_raw for the backward kv-recompute. The
    // `level1_with_fused_proj` builder is the single source of truth
    // for this invariant — it sets `save_activations_for_backward = true`
    // alongside `fused_projections = true`. Future builder refactors
    // that drop the save flag would silently regress to the cycle-11
    // silent-garbage bug.
    let cfg = build_bypass_config();
    assert!(
        cfg.checkpoint.is_some(),
        "G6 precondition: build_bypass_config must have checkpoint=Some"
    );
    let csha = cfg.csha.as_ref().expect("G6 precondition: CSHA required");
    assert!(
        csha.save_activations_for_backward,
        "G6: x_raw save not auto-enabled with checkpoint=Some. \
         level1_with_fused_proj() must set save_activations_for_backward=true."
    );
    assert!(
        csha.fused_projections,
        "G6: fused_projections not set with checkpoint=Some. \
         Cycle-12 R3 augmentation requires it."
    );
}

#[test]
fn g8_trap_on_null_x_raw_ptr() {
    // Cycle 12: the cycle-11 `bra V2_KV_RECOMPUTE_DONE_*` early-exit
    // on null x_raw_ptr was the SILENT-garbage bug — downstream
    // ds_compute would read undefined K/V SMEM. Cycle 12 converts the
    // null guard to PTX `trap;` so any future regression fires
    // CUDA_ERROR_LAUNCH_FAILED at runtime instead.
    let cfg = build_bypass_config();
    let ptx = synthesize_backward_with_tier(&cfg)
        .expect("G8: bypass-active backward synthesis must succeed");
    assert!(
        ptx.contains("@%p_kvr_xraw_null trap"),
        "G8: null x_raw_ptr trap missing in emit_kv_recompute. \
         Cycle-12 silent-garbage -> loud-abort conversion regressed."
    );
    // Inner prologue null guard converted too.
    assert!(
        ptx.contains("@%p_xraw_null trap"),
        "G8: null x_raw_ptr trap missing in emit_prologue_recompute_from_raw"
    );
    // The retired bra-to-DONE early-exit must NOT reappear (regression
    // canary).
    assert!(
        !ptx.contains("@%p_kvr_xraw_null bra V2_KV_RECOMPUTE_DONE_"),
        "G8: legacy bra-on-null early-exit reappeared (silent-garbage regression)"
    );
}

#[test]
fn g3d_no_decorator_path_byte_identical() {
    // Sanity gate alongside `fa_v2_snapshots`: the no-decorator branch of
    // the cycle-11 dispatch fork must still go through `kv_load::emit_k_suffixed`
    // and must NOT pick up any recompute marker. The existing 25/25
    // snapshot suite enforces full byte-identity; this probe is a
    // first-line tripwire so a regression here surfaces in the cycle-11
    // test target rather than only via the snapshot diff.
    let cfg = build_fusible_no_checkpoint();
    let ptx = synthesize_backward_with_tier(&cfg)
        .expect("G3d: no-decorator backward synthesis must succeed");
    assert!(
        ptx.contains("V2_BWD_K_LOAD"),
        "G3d: production no-decorator path lost the kv_load K marker — \
         dispatch fork misrouted a checkpoint=None config."
    );
    assert!(
        !ptx.contains("V2_KV_RECOMPUTE"),
        "G3d: production no-decorator path picked up a recompute marker — \
         dispatch fork misrouted a checkpoint=None config."
    );
    assert!(
        !ptx.contains("_bwd_recompute_"),
        "G3d: production no-decorator path leaked recompute-namespace suffix."
    );
}
