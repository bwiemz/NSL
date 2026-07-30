//! Structure + ptxas validation for the tensor-core SDPA forward
//! (`flash_attention_v2::mma_forward`).
//!
//! The structural tests run everywhere. The ptxas tests are
//! `#[ignore]`-gated (need a CUDA toolkit):
//!     cargo test --package nsl-codegen --test fa_mma_forward_validation \
//!         -- --ignored --nocapture
//!
//! Per the Tier B.1 meta-lesson these prove STRUCTURE only — the
//! numerical gate lives in `sdpa_fused_forward_gpu_parity.rs`, which
//! exercises this body on real hardware against an f64 oracle.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::mma_forward;
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, synthesize_flash_attention_ptx_v2,
};
use std::process::Command;

/// The production shape: the SDPA fwd variant table's (64, 64) tile at
/// head_dim 64 (coder50m), causal, plain config, sm_80-class target.
fn sdpa_config(block_q: i64, block_kv: i64, head_dim: i64, causal: bool) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q,
        block_kv,
        head_dim,
        causal,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: None,
        checkpoint: None,
    }
}

fn ptx_string(config: &FlashAttentionConfig) -> String {
    let mut bytes = synthesize_flash_attention_ptx_v2(config);
    assert_eq!(bytes.last(), Some(&0), "PTX must be NUL-terminated");
    bytes.pop();
    String::from_utf8(bytes).expect("PTX must be valid UTF-8")
}

// ── admission ───────────────────────────────────────────────────────────

#[test]
fn every_sdpa_table_shape_is_admitted() {
    // Mirrors SDPA_FWD_VARIANT_HEAD_DIMS x TILE_CANDIDATES in
    // compiler/kernel.rs — every combination the table can select must
    // take the MMA body, or the dispatch silently splits by shape.
    for hd in [32i64, 64, 128] {
        for (bq, bkv) in [(64i64, 64i64), (32, 32), (32, 16), (16, 16)] {
            let cfg = sdpa_config(bq, bkv, hd, true);
            assert!(
                mma_forward::admits_shape(&cfg),
                "table shape bq={bq} bkv={bkv} hd={hd} must admit"
            );
        }
    }
}

#[test]
fn non_admissible_shapes_refuse() {
    let base = sdpa_config(64, 64, 64, true);
    let mut sm75 = base.clone();
    sm75.gpu_sm = 75;
    assert!(!mma_forward::admits_shape(&sm75), "sm_75 has no mma.sync m16n8k16");

    let mut gqa = base.clone();
    gqa.gqa_group_size = 4;
    assert!(!mma_forward::admits_shape(&gqa), "native-GQA indexing not wired in v1");

    let mut rope = base.clone();
    rope.rope_q = true;
    assert!(!mma_forward::admits_shape(&rope), "rope_q rotation not wired in v1");

    let mut seg = base.clone();
    seg.segment_masked = true;
    assert!(!mma_forward::admits_shape(&seg), "segment masking not wired in v1");

    let mut paged = base;
    paged.paged = true;
    assert!(!mma_forward::admits_shape(&paged), "paged KV not wired in v1");
}

// ── kernel name + body coherence ────────────────────────────────────────

#[test]
fn mma_body_and_name_agree() {
    let cfg = sdpa_config(64, 64, 64, true);
    let name = flash_attention_kernel_name_v2(&cfg);
    assert_eq!(
        name, "flash_attn_p0_r0_hs_g1_c1_t0_q64_kv64_v2_mma",
        "MMA suffix must appear exactly once at the end"
    );
    let ptx = ptx_string(&cfg);
    assert!(
        ptx.contains(&format!(".visible .entry {} (", name)),
        "entry name must match flash_attention_kernel_name_v2"
    );
}

#[test]
fn sm75_config_keeps_scalar_name_and_body() {
    let mut cfg = sdpa_config(64, 64, 64, true);
    cfg.gpu_sm = 75;
    assert_eq!(
        flash_attention_kernel_name_v2(&cfg),
        "flash_attn_p0_r0_hs_g1_c1_t0_q64_kv64_v2"
    );
    let ptx = ptx_string(&cfg);
    assert_eq!(
        ptx.matches("mma.sync").count(),
        0,
        "sm_75 body must stay scalar"
    );
}

// ── PTX structure ───────────────────────────────────────────────────────

#[test]
fn mma_counts_match_tile_arithmetic() {
    // (64, 64, 64): QK = k_iters(hd/16=4) x n_tiles_s(bkv/8=8) = 32,
    // PV = k_iters(bkv/16=4) x n_tiles_o(hd/8=8) = 32. The KV loop is a
    // PTX-level backedge, so the static count is exactly 64.
    let ptx = ptx_string(&sdpa_config(64, 64, 64, true));
    assert_eq!(
        ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").count(),
        64,
        "(64,64,64) must emit 32 QK + 32 PV mma.sync"
    );
    // (32, 16, 32): QK = 2 x 2 = 4, PV = 1 x 4 = 4.
    let ptx = ptx_string(&sdpa_config(32, 16, 32, true));
    assert_eq!(
        ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").count(),
        8,
        "(32,16,32) must emit 4 QK + 4 PV mma.sync"
    );
}

#[test]
fn structure_landmarks_present() {
    let ptx = ptx_string(&sdpa_config(64, 64, 64, true));
    for landmark in [
        "FMMA_LOOP_KV:",           // single KV loop (no per-4-row re-emission)
        "FMMA_K_LOAD:",            // padded K loader (bank-conflict-free)
        "FMMA_Q_LOAD:",            // full-tile Q load
        "FMMA_VT_LOAD:",           // transposed V store
        "mov.u32 %fm_smem32, shmem;", // shared-window base, NOT cvt.u32.u64
        "min.u64 %fm_kbound",      // causal early-exit bound
        "ex2.approx.f32",          // scalar-path softmax idiom
        "lg2.approx.f32",          // finalize LSE idiom
        "st.global.b16",           // f16 output staging contract
    ] {
        assert!(ptx.contains(landmark), "missing landmark {landmark:?}:\n{ptx}");
    }
    // The whole q_tile_iter unroll must be gone: exactly one K-load site.
    assert_eq!(ptx.matches("FMMA_K_LOAD").count(), 2, // label + backedge
        "MMA body must emit exactly one K tile load site");
    assert!(!ptx.contains("V2_LOOP_K_LOAD_"), "scalar K loader must not appear");
    // Non-causal body must not emit the mask block.
    let ptx_nc = ptx_string(&sdpa_config(64, 64, 64, false));
    assert!(!ptx_nc.contains("causal: k_global"), "non-causal must skip mask");
    assert!(!ptx_nc.contains("min.u64 %fm_kbound"), "non-causal loops to seq_len");
}

#[test]
fn pure_ascii_ptx() {
    // Non-ASCII bytes make cuModuleLoadData fail with CUDA_ERROR_INVALID_PTX
    // (two em-dashes have already shipped broken kernels — see
    // flash_attention_ptx_lint.rs).
    let ptx = ptx_string(&sdpa_config(64, 64, 64, true));
    assert!(ptx.is_ascii(), "MMA forward PTX must be pure ASCII");
}

// ── ptxas ───────────────────────────────────────────────────────────────

fn run_ptxas(sm: u32, config: &FlashAttentionConfig) {
    let mut ptx_bytes = synthesize_flash_attention_ptx_v2(config);
    if ptx_bytes.last() == Some(&0) {
        ptx_bytes.pop();
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "fa_mma_fwd_sm{}_q{}kv{}hd{}_{}.ptx",
        sm, config.block_q, config.block_kv, config.head_dim,
        std::process::id()
    ));
    std::fs::write(&tmp, &ptx_bytes).expect("write ptx temp");
    let out = Command::new("ptxas")
        .arg("--gpu-name")
        .arg(format!("sm_{}", sm))
        .arg("-c")
        .arg(&tmp)
        .output()
        .expect("ptxas not on PATH; install CUDA toolkit");
    assert!(
        out.status.success(),
        "ptxas sm_{} failed:\nPTX file: {}\nSTDOUT:\n{}\nSTDERR:\n{}",
        sm,
        tmp.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_file(&tmp);
}

#[test]
#[ignore = "requires ptxas on PATH; lift in CI with CUDA toolchain"]
fn mma_forward_assembles_on_sm_80_all_table_shapes() {
    for hd in [32i64, 64, 128] {
        for (bq, bkv) in [(64i64, 64i64), (32, 32), (32, 16), (16, 16)] {
            for causal in [true, false] {
                run_ptxas(80, &sdpa_config(bq, bkv, hd, causal));
            }
        }
    }
}

#[test]
#[ignore = "requires ptxas on PATH; lift in CI with CUDA toolchain"]
fn mma_forward_assembles_on_sm_120_production_shape() {
    run_ptxas(120, &sdpa_config(64, 64, 64, true));
    run_ptxas(120, &sdpa_config(64, 64, 128, true));
}
