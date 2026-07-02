//! Cycle-16 G16-2 / G16-3 triage harnesses.
//!
//! ## G16-2 (Path B CUDA_ERROR_ILLEGAL_ADDRESS): RESOLVED in cycle-16 Task 3
//!
//! Root cause: `shared_mem_bytes_v2_backward` (and the test harness in
//! `csha_checkpoint_recompute_gpu.rs`) omitted `recompute_extra_bytes` for
//! `checkpoint=Some(Full)` configs. The `emit_kv_recompute` path writes a
//! recomputed x_norm scratch tile starting at:
//!   `recompute_xnorm_offset = total_bytes + backward_extra_bytes`
//! which was exactly the end of the previously-granted SMEM region.
//! Fix: `shared_mem_bytes_v2_backward` now adds `recompute_extra_bytes(config)`
//! when `config.checkpoint.is_some()`.
//! For hd=64, bq=32: 32*64*2 = 4096 extra bytes (Path B SMEM 90496 -> 94592).
//!
//! ## G16-3 (A3 CUDA_ERROR_MISALIGNED_ADDRESS): DEFERRED to cycle 17
//!
//! Config: causal=true, rope_q=true, fused_projections=false, d_model=0,
//!         hd=64, S=512, bq=32, checkpoint=None (Path A).
//!
//! Static analysis findings:
//!   - emit_xnorm_recompute returns early when d_model=0: no SMEM write.
//!   - emit_dproj returns early when d_model=0: no write.
//!   - emit_drmsnorm emits label then returns: no phase 1/1b/2 emitted.
//!   - emit_kv_recompute: NOT called (Path A, checkpoint=None).
//!   - Basic dQ/dK/dV backward still runs via emit_store_kv_only / emit_store_dk_only.
//!   - Kernel runs to completion (rc=0), crash fires during cuMemFree_v2.
//!
//! Working hypothesis for cycle 17 investigation:
//!   When d_model=0, backward_dx_norm_offset == backward_rms_strip_offset ==
//!   backward_x_norm_offset (all three aliases). The rms_strip tile (bq*4=128B)
//!   at this zero-base offset may overlap with Phase-4 SMEM reads/writes,
//!   causing GPU memory corruption that manifests as MISALIGNED_ADDRESS on the
//!   runtime's dk_scratch/dv_scratch cuMemFree_v2. Needs cuda-memcheck.
//!
//! Structural tests in this file run WITHOUT --ignored (default CI).
//! GPU smoke tests require `#[cfg(feature = "cuda")]` and `#[ignore]`.

use nsl_codegen::flash_attention::{
    CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_v2::{
    shared_mem_bytes_v2_backward, smem_layout, synthesize_backward_with_tier_b,
};

/// Build the cycle-14/15 base config (hd=64, bq=32, checkpoint=Full).
fn build_cycle14_config() -> FlashAttentionConfig {
    let d_model: u32 = 64;
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 64,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            d_model,
            ..CshaExtras::level1_with_fused_proj(1e-6)
        }),
        checkpoint: Some(CheckpointExtras::full()),
    }
}

// ── G16-2 Structural Witnesses ─────────────────────────────────────────────────

/// G16-2-T1: After the cycle-16 fix, `shared_mem_bytes_v2_backward` for the
/// checkpoint=Full config must exceed the base `total_bytes + backward_extra_bytes`
/// by exactly `recompute_extra_bytes`. This is the structural proof that the
/// SMEM grant no longer truncates at the xnorm scratch boundary.
#[test]
fn g16_2_t1_smem_includes_recompute_extra_after_fix() {
    let cfg = build_cycle14_config();

    let base_smem = smem_layout::total_bytes(&cfg) + smem_layout::backward_extra_bytes(&cfg);
    let fixed_smem = shared_mem_bytes_v2_backward(&cfg);
    let expected_extra = smem_layout::recompute_extra_bytes(&cfg) as u32;

    assert!(
        fixed_smem > base_smem,
        "G16-2 fix: shared_mem_bytes_v2_backward must exceed base when \
         checkpoint=Some; base={base_smem} fixed={fixed_smem}"
    );
    assert_eq!(
        fixed_smem - base_smem,
        expected_extra,
        "G16-2 fix: SMEM delta must equal recompute_extra_bytes; \
         delta={} expected={}",
        fixed_smem - base_smem,
        expected_extra
    );
    eprintln!(
        "[g16_2_t1] base_smem={base_smem} fixed_smem={fixed_smem} recompute_extra={expected_extra}"
    );
}

/// G16-2-T2: Without checkpoint, `shared_mem_bytes_v2_backward` must equal the
/// base (no recompute overhead). Verifies the G16-2 fix does not perturb Path A.
#[test]
fn g16_2_t2_smem_no_change_without_checkpoint() {
    let mut cfg = build_cycle14_config();
    cfg.checkpoint = None;

    let base_smem = smem_layout::total_bytes(&cfg) + smem_layout::backward_extra_bytes(&cfg);
    let result = shared_mem_bytes_v2_backward(&cfg);
    assert_eq!(
        result, base_smem,
        "G16-2 fix: without checkpoint, shared_mem_bytes_v2_backward must \
         equal base; base={base_smem} result={result}"
    );
}

/// G16-2-T3: `recompute_xnorm_offset` must equal the old (pre-fix) SMEM grant
/// boundary. Proves the xnorm write started exactly at the boundary.
#[test]
fn g16_2_t3_recompute_xnorm_offset_equals_old_boundary() {
    let cfg = build_cycle14_config();
    let old_boundary = smem_layout::total_bytes(&cfg) + smem_layout::backward_extra_bytes(&cfg);
    let xnorm_off = smem_layout::recompute_xnorm_offset(&cfg);
    assert_eq!(
        xnorm_off, old_boundary,
        "G16-2 root cause: recompute_xnorm_offset must be exactly at the pre-fix \
         SMEM grant boundary. xnorm_off={xnorm_off} old_boundary={old_boundary}"
    );
    eprintln!(
        "[g16_2_t3] recompute_xnorm_offset={xnorm_off} == old_boundary={old_boundary} CONFIRMED"
    );
}

// ── G16-3 Structural Witnesses ─────────────────────────────────────────────────

/// G16-3-T1: With d_model=0, backward_{x_norm,dx_norm,rms_strip}_offset all
/// alias to the same value. Documents the zero-size tile aliasing that is the
/// suspected root cause for cycle-17 investigation.
#[test]
fn g16_3_t1_zero_d_model_smem_aliasing() {
    let mut cfg = build_cycle14_config();
    cfg.checkpoint = None; // Path A (no kv_recompute)
    if let Some(ref mut csha) = cfg.csha {
        csha.fused_projections = false;
        csha.d_model = 0;
    }

    let x_norm_off = smem_layout::backward_x_norm_offset(&cfg);
    let dx_norm_off = smem_layout::backward_dx_norm_offset(&cfg);
    let rms_strip_off = smem_layout::backward_rms_strip_offset(&cfg);

    assert_eq!(
        x_norm_off, dx_norm_off,
        "G16-3: x_norm and dx_norm offsets must alias when d_model=0; \
         x_norm={x_norm_off} dx_norm={dx_norm_off}"
    );
    assert_eq!(
        dx_norm_off, rms_strip_off,
        "G16-3: dx_norm and rms_strip offsets must alias when d_model=0; \
         dx_norm={dx_norm_off} rms_strip={rms_strip_off}"
    );
    eprintln!(
        "[g16_3_t1] d_model=0 aliasing confirmed: \
         x_norm=dx_norm=rms_strip={x_norm_off}"
    );
}

/// G16-3-T2: With d_model=0 and fused_projections=false, the A3 backward PTX
/// must synthesize without error. Kernel synthesis is not the failure point --
/// the crash fires at runtime during scratch-buffer free (deferred to cycle 17).
#[test]
fn g16_3_t2_a3_backward_ptx_synthesizes_ok() {
    let mut cfg = build_cycle14_config();
    cfg.checkpoint = None;
    if let Some(ref mut csha) = cfg.csha {
        csha.fused_projections = false;
        csha.d_model = 0;
    }

    let result = synthesize_backward_with_tier_b(&cfg, None);
    assert!(
        result.is_ok(),
        "G16-3: A3 backward PTX synthesis must succeed (crash is runtime, not PTX): {:?}",
        result.err()
    );
    let ptx = result.unwrap();
    assert!(!ptx.is_empty(), "G16-3: synthesized PTX must be non-empty");
    eprintln!(
        "[g16_3_t2] A3 backward PTX: {} bytes -- synthesis OK, crash deferred to cycle 17",
        ptx.len()
    );
}

/// G16-3-T3: SMEM for A3 must be static (< 48*1024 = 49152 bytes). The
/// cycle-16 crash fires in runtime cleanup, NOT at kernel launch, confirming
/// the SMEM grant is NOT the proximate cause (unlike G16-2 which was SMEM).
#[test]
fn g16_3_t3_a3_smem_is_static() {
    let mut cfg = build_cycle14_config();
    cfg.checkpoint = None;
    if let Some(ref mut csha) = cfg.csha {
        csha.fused_projections = false;
        csha.d_model = 0;
    }

    let total = shared_mem_bytes_v2_backward(&cfg);
    const STATIC_THRESHOLD: u32 = 48 * 1024;
    assert!(
        total <= STATIC_THRESHOLD,
        "G16-3: A3 must use static SMEM (<= {STATIC_THRESHOLD}); got {total}"
    );
    eprintln!(
        "[g16_3_t3] A3 SMEM={total} bytes (static, dyn=0) -- \
         G16-3 is NOT a SMEM-grant issue (contrast with G16-2)"
    );
}
