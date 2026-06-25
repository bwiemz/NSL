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
/// composition) AND sets `r0_bypass=true` so the codegen path falls
/// through R0 into the cycle-11 substitution.
fn build_bypass_config() -> FlashAttentionConfig {
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
        csha: Some(CshaExtras::level1(1e-6)),
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
