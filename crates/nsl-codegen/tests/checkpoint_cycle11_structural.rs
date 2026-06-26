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
fn g3c_rope_k_write_between_projection_and_ds_compute() {
    // Cycle-12 extension to G3c: when rope_q=true, the backward
    // kv-recompute path MUST lift the forward `emit_rope_k_epilogue`
    // emission into the backward (step 5 of `emit_kv_recompute`). The
    // characteristic emission is the rope-K loop label
    // (`V2_CSHA_ROPE_K_LOOP_0`) or the fused-skip label
    // (`V2_CSHA_ROPE_K_FUSED_SKIP`) — both appear when
    // rope_q + fused_projections are set.
    //
    // Ordering invariant we extend over base G3c:
    //   1. K projection matmul (V2_RECOMPUTE_K_MATMUL_MAIN) precedes
    //   2. RoPE-K rotation emission (V2_CSHA_ROPE_K_*) precedes
    //   3. KV recompute completion barrier (V2_KV_RECOMPUTE_DONE_MAIN)
    //      precedes
    //   4. first ds_compute consumer (V2_BWD_DS_*).
    //
    // Catches the cycle-11 RoPE-K gap that the cycle-12 R3 augmentation
    // closes structurally — if `emit_rope_k_epilogue` is not invoked
    // from `emit_kv_recompute`, downstream ds_compute reads
    // un-rotated K and gradients diverge silently.
    let cfg = build_bypass_config_with_rope();
    let ptx = synthesize_backward_with_tier(&cfg)
        .expect("G3c rope: bypass-active rope_q backward synthesis must succeed");

    let k_matmul_pos = ptx
        .find("V2_RECOMPUTE_K_MATMUL_MAIN:")
        .expect("G3c rope: V2_RECOMPUTE_K_MATMUL_MAIN label missing");
    // The rope-K emitter writes the LOOP label inside the rotation body.
    let rope_k_pos = ptx
        .find("V2_CSHA_ROPE_K_LOOP_0")
        .or_else(|| ptx.find("V2_CSHA_ROPE_K_FUSED_SKIP"))
        .expect(
            "G3c rope: emit_rope_k_epilogue did not emit RoPE-K marker inside \
             emit_kv_recompute — cycle-11 RoPE-K gap NOT closed",
        );
    let recompute_done_pos = ptx
        .find("V2_KV_RECOMPUTE_DONE_MAIN")
        .expect("G3c rope: KV recompute DONE label missing");
    let first_ds_pos = ptx
        .find("V2_BWD_DS_")
        .expect("G3c rope: ds_compute label V2_BWD_DS_* missing");

    assert!(
        k_matmul_pos < rope_k_pos,
        "G3c rope: K projection matmul @ {} did not precede rope-K rotation @ {}",
        k_matmul_pos, rope_k_pos
    );
    assert!(
        rope_k_pos < recompute_done_pos,
        "G3c rope: rope-K rotation @ {} did not precede recompute DONE @ {}",
        rope_k_pos, recompute_done_pos
    );
    assert!(
        recompute_done_pos < first_ds_pos,
        "G3c rope: recompute DONE @ {} did not precede first V2_BWD_DS_ @ {}",
        recompute_done_pos, first_ds_pos
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
