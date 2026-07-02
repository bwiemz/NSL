//! Structural tests for the Phase 2.6 (T2.5) B.1 save-activation retrofit.
//!
//! The Tier B.1 forward kernel was extended (`tier_b1::finalize::emit`) to
//! scatter the 5 backward-activation tensors (q_proj/k_proj/v_proj +
//! row_max/row_sum) to HBM when `save_activations_for_backward=true`. These
//! tests assert the emitted PTX has the expected save structure per the
//! V-Phase-2.6-B design doc section 6 (param loads, null-guards, store
//! counts, the col-major V read, the x_raw exclusion, and ASCII-only PTX).
//!
//! STRUCTURE-ONLY. Numerical CPU-reference / GPU validation is Task 2.7
//! (`csha-tier-b1-numerical-correctness` meta-lesson: snapshot/ptxas/SASS
//! validate structure only; end-to-end numerical comparison is mandatory
//! before claiming correctness for fused-MMA kernels).

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b1::synthesize;

/// Canonical 32x32x32 Tier B.1 config with backward saves enabled. chunk
/// 128 matches `chunk_config::select` for this geometry (mirrors
/// `tier_b1_kernel_snapshot.rs`).
fn save_config() -> FlashAttentionConfig {
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
        gpu_sm: 120,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 2048,
            save_activations_for_backward: true,
            ..CshaExtras::default()
        }),
        checkpoint: None,
    }
}

/// Same geometry but saves disabled — proves the retrofit is gated.
fn no_save_config() -> FlashAttentionConfig {
    let mut c = save_config();
    if let Some(e) = c.csha.as_mut() {
        e.save_activations_for_backward = false;
    }
    c
}

fn synth(cfg: &FlashAttentionConfig) -> String {
    let bytes = synthesize(cfg, 128);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// §6.2 — each of the 5 save pointers is `ld.param.u64`'d exactly once.
#[test]
fn param_loads_present_for_all_five_saves() {
    let ptx = synth(&save_config());
    for p in ["q_proj_ptr", "k_proj_ptr", "v_proj_ptr", "row_max_ptr", "row_sum_ptr"] {
        assert_eq!(
            ptx.matches(&format!(", [{p}];")).count(),
            1,
            "expected exactly one ld.param.u64 for {p}\n--- PTX ---\n{ptx}"
        );
    }
}

/// §6.3 — null-guards. One `setp.eq.u64` + `bra ..._SKIP` per projection
/// pointer; `setp.ne.u64` for row_max/row_sum.
#[test]
fn null_guards_present() {
    let ptx = synth(&save_config());
    for label in ["Q", "K", "V"] {
        assert!(
            ptx.contains(&format!("setp.eq.u64 %psv_{label}_null, %psv_{label}_ptr, 0;")),
            "missing setp.eq null-guard for {label}"
        );
        assert!(
            ptx.contains(&format!("@%psv_{label}_null bra V2_TIER_B1_SAVE_{label}_SKIP;")),
            "missing bra-to-SKIP for {label}"
        );
        assert!(
            ptx.contains(&format!("V2_TIER_B1_SAVE_{label}_SKIP:")),
            "missing SKIP label for {label}"
        );
    }
    assert!(
        ptx.contains("setp.ne.u64 %psv_has_row_max, %psv_row_max_base, 0;"),
        "missing row_max setp.ne null-guard"
    );
    assert!(
        ptx.contains("setp.ne.u64 %psv_has_row_sum, %psv_row_sum_base, 0;"),
        "missing row_sum setp.ne null-guard"
    );
}

/// §6.1 — save store counts. Canonical 32x32x32: rows_per_warp=4,
/// cols_per_lane=1, so each projection emits 4 f16 save stores. row_max
/// and row_sum each emit `tpw*2` gated f32 stores (tpw=1 for 32x32x32 PV).
#[test]
fn save_store_counts_match() {
    let ptx = synth(&save_config());
    assert_eq!(
        ptx.matches("// q_proj_ptr write").count(),
        4,
        "expected 4 q_proj f16 save stores"
    );
    assert_eq!(
        ptx.matches("// k_proj_ptr write").count(),
        4,
        "expected 4 k_proj f16 save stores"
    );
    assert_eq!(
        ptx.matches("// v_proj_ptr write").count(),
        4,
        "expected 4 v_proj f16 save stores"
    );
    // f32 reductions: one (max,sum) per (tile, half). 32x32x32 PV tpw=1 ->
    // 2 stores each.
    let row_max_stores = ptx.matches("// row_max_ptr write").count();
    let row_sum_stores = ptx.matches("// row_sum_ptr write").count();
    assert!(row_max_stores > 0, "expected nonzero row_max f32 stores");
    assert_eq!(
        row_max_stores, row_sum_stores,
        "row_max and row_sum must have equal store counts"
    );
    // The projection saves use st.global.b16; reductions use st.global.f32.
    assert!(
        ptx.matches("st.global.b16").count() >= 12,
        "expected at least 12 b16 save stores (4 per Q/K/V) plus the O write"
    );
}

/// §6.4 — V uses the col-major SMEM read; Q/K use row-major. The §3.3
/// transpose is the highest-risk line (R2 load_transposed bug class).
#[test]
fn v_read_is_col_major_qk_are_row_major() {
    let ptx = synth(&save_config());
    // V: multiply the COLUMN index by (bkv*2) = 64.
    assert!(
        ptx.contains("mul.lo.u64 %psv_V_smem_b, %psv_V_c64, 64;  // c * (bkv*2)"),
        "V save must read col-major SMEM (col * bkv*2)"
    );
    // V must NOT use the row-major (row-multiplied) index.
    assert!(
        !ptx.contains("mul.lo.u64 %psv_V_smem_b, %psv_V_r64,"),
        "V save must NOT use the row-major SMEM index"
    );
    // Q/K: multiply the ROW index by (hd*2) = 64.
    assert!(
        ptx.contains("mul.lo.u64 %psv_Q_smem_b, %psv_Q_r64, 64;  // r * (hd*2)"),
        "Q save must read row-major SMEM (row * hd*2)"
    );
    assert!(
        ptx.contains("mul.lo.u64 %psv_K_smem_b, %psv_K_r64, 64;  // r * (hd*2)"),
        "K save must read row-major SMEM (row * hd*2)"
    );
}

/// R1 — Q HBM seq = q_start + r; K/V HBM seq = r (absolute K position).
#[test]
fn r1_seq_index_q_vs_kv() {
    let ptx = synth(&save_config());
    assert!(
        ptx.contains("add.u64 %psv_Q_seq, %psv_Q_seq, %q_start;"),
        "Q save must add q_start to its seq index"
    );
    assert!(
        !ptx.contains("add.u64 %psv_K_seq, %psv_K_seq, %q_start"),
        "K save must NOT add q_start (uses absolute K position r)"
    );
    assert!(
        !ptx.contains("add.u64 %psv_V_seq, %psv_V_seq, %q_start"),
        "V save must NOT add q_start (uses absolute K position r)"
    );
}

/// §6.5 — the save retrofit excludes x_raw (§1: the production pre-pass
/// overwrites x on the host; the dQ-kernel does not need pre-norm x). The
/// retrofit's finalize save block must never target x_raw. NOTE: the Tier-C
/// RMSNorm prologue (`csha_hooks::emit_prologue`) independently stages x_raw
/// to HBM when it runs -- that pre-existing behavior is orthogonal to this
/// retrofit. In the production layout (`skip_rmsnorm_prologue=true`, used by
/// the dQ-kernel's caller per the #183 narrow+chunkify pre-pass) the
/// prologue is skipped and x_raw_ptr is never touched at all.
#[test]
fn retrofit_save_block_never_targets_x_raw() {
    // (a) The retrofit's own save stores never derive from x_raw_ptr,
    //     regardless of whether the prologue runs. The retrofit's save
    //     stores are tagged with 'q/k/v_proj_ptr write' / 'row_max_ptr
    //     write' / 'row_sum_ptr write' comments; an 'x_raw_ptr write'
    //     comment would only exist if the retrofit (wrongly) saved x_raw.
    //     (The prologue's own x_raw staging uses a distinct 'x_raw save'
    //     comment and is orthogonal to this retrofit.)
    let ptx = synth(&save_config());
    assert!(
        !ptx.contains("x_raw_ptr write"),
        "no retrofit save store should derive from x_raw_ptr"
    );

    // (b) Production layout: with the RMSNorm prologue skipped (the caller
    //     owns narrow+chunkify), x_raw_ptr is never loaded -- the §1
    //     exclusion holds literally.
    let mut prod = save_config();
    if let Some(e) = prod.csha.as_mut() {
        e.skip_rmsnorm_prologue = true;
    }
    let ptx_prod = synth(&prod);
    assert!(
        !ptx_prod.contains("[x_raw_ptr]"),
        "production (skip_rmsnorm_prologue) path must NOT load x_raw_ptr"
    );
    assert!(
        !ptx_prod.contains("x_raw_ptr write"),
        "production path must NOT write any x_raw save store"
    );
}

/// §6.6 — retrofit is gated. With saves disabled, NONE of the save pointers
/// are loaded, no %psv_ registers appear, no save stores emitted.
#[test]
fn no_emission_when_saves_disabled() {
    let ptx = synth(&no_save_config());
    for p in ["q_proj_ptr", "k_proj_ptr", "v_proj_ptr", "row_max_ptr", "row_sum_ptr"] {
        assert!(
            !ptx.contains(&format!(", [{p}];")),
            "{p} must not be loaded when saves disabled"
        );
    }
    assert!(!ptx.contains("%psv_"), "no %psv_ registers when saves disabled");
    assert!(
        !ptx.contains("V2_TIER_B1_SAVE_"),
        "no save SKIP labels when saves disabled"
    );
}

/// §6.6 — ASCII-only PTX (institutional pin `feedback_ptx_comment_ascii_only`:
/// Unicode in PTX comments triggers CUDA_ERROR_INVALID_PTX under cudarc JIT).
#[test]
fn emitted_ptx_is_ascii_only() {
    let bytes = synthesize(&save_config(), 128);
    // The kernel ends with a NUL terminator; allow that one non-printable
    // byte (it is < 128 anyway). Assert every byte is 7-bit ASCII.
    for (i, &b) in bytes.iter().enumerate() {
        assert!(
            b < 128,
            "non-ASCII byte 0x{b:02x} at offset {i} in emitted Tier B.1 save PTX"
        );
    }
}

/// The projection save sweep must be fenced (Q/K/V SMEM visible) and run
/// after the O/LSE loop (design §5 ordering).
#[test]
fn save_sweep_is_fenced() {
    let ptx = synth(&save_config());
    assert!(
        ptx.contains("bar.sync 0;  // FENCE: Q/K/V SMEM tiles fully visible to save sweep"),
        "projection save sweep must be preceded by a visibility fence"
    );
}
