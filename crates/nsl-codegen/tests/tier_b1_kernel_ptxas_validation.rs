//! Verify the full Tier B.1 kernel PTX (as produced by `synthesize`)
//! assembles cleanly on sm_80 and sm_120.
//!
//! These tests are `#[ignore]`-gated because they require `ptxas` from a
//! CUDA toolkit. Run with:
//!     cargo test --package nsl-codegen --test tier_b1_kernel_ptxas_validation \
//!         -- --ignored --nocapture

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b1::synthesize;
use std::process::Command;

fn canonical_config() -> FlashAttentionConfig {
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
        gpu_sm: 120,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 2048,
            ..CshaExtras::default()
        }),
    }
}

/// Canonical config + Phase 2.6 (T2.5) backward-activation saves enabled.
/// Exercises the q/k/v_proj + row_max/row_sum scatter emission in
/// `tier_b1::finalize::emit`.
fn save_config() -> FlashAttentionConfig {
    let mut c = canonical_config();
    if let Some(e) = c.csha.as_mut() {
        e.save_activations_for_backward = true;
    }
    c
}

fn run_ptxas_for_sm(sm: u32) {
    run_ptxas_for_sm_with_config(sm, &canonical_config());
}

fn run_ptxas_for_sm_with_config(sm: u32, config: &FlashAttentionConfig) {
    let config = config.clone();
    let chunk: u32 = 128; // chunk_config::select picks 128 for this 32x32x32 cfg.
    let mut ptx_bytes = synthesize(&config, chunk);
    // synthesize appends a NUL terminator; strip it so the file is valid text.
    if ptx_bytes.last() == Some(&0) {
        ptx_bytes.pop();
    }

    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "tier_b1_full_sm{}_{}.ptx",
        sm,
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
}

#[test]
#[ignore = "requires ptxas on PATH; lift in CI with CUDA toolchain"]
fn tier_b1_full_kernel_assembles_on_sm_80() {
    run_ptxas_for_sm(80);
}

#[test]
#[ignore = "requires ptxas on PATH; lift in CI with CUDA toolchain"]
fn tier_b1_full_kernel_assembles_on_sm_120() {
    run_ptxas_for_sm(120);
}

#[test]
#[ignore = "requires ptxas on PATH; lift in CI with CUDA toolchain"]
fn tier_b1_save_activations_kernel_assembles_on_sm_80() {
    run_ptxas_for_sm_with_config(80, &save_config());
}

#[test]
#[ignore = "requires ptxas on PATH; lift in CI with CUDA toolchain"]
fn tier_b1_save_activations_kernel_assembles_on_sm_120() {
    run_ptxas_for_sm_with_config(120, &save_config());
}
