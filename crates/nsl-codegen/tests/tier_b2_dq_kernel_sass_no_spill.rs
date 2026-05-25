//! Verify the Tier B.2 dQ backward kernel compiles to SASS with zero register
//! spill loads (LDL) and zero register spill stores (STL).
//!
//! Three head-dim configs are tested: hd=32, hd=64, hd=128. All are expected
//! to be spill-free at sm_80 (the lowest-common target that supports the full
//! instruction set used by the dQ emitter).
//!
//! These tests are `#[ignore]`-gated because they require both `ptxas` and
//! `cuobjdump` from a CUDA toolkit. Run with:
//!     cargo test --package nsl-codegen --test tier_b2_dq_kernel_sass_no_spill \
//!         -- --ignored --nocapture

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;
use std::process::Command;

fn cfg(bq: i64, hd: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq,
        block_kv: bq,
        head_dim: hd,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            ..Default::default()
        }),
    }
}

#[test]
#[ignore = "requires ptxas + cuobjdump on PATH; lift in CI with CUDA toolchain"]
fn tier_b2_dq_no_register_spills_all_configs() {
    // hd=32/64 -> bq=64; hd=128 -> bq=32 (effective fallback). No-spill at all 3.
    for &(bq, hd) in &[(64i64, 32i64), (64, 64), (32, 128)] {
        assert_no_spills(&cfg(bq, hd), &format!("hd{hd}_bq{bq}"));
    }
}

fn assert_no_spills(config: &FlashAttentionConfig, label: &str) {
    let ptx = synthesize_dq_kernel(config).expect("dq synth");
    // synthesize_dq_kernel returns a String (no NUL terminator to strip).

    let pid = std::process::id();
    let mut ptx_path = std::env::temp_dir();
    ptx_path.push(format!("tier_b2_dq_sass_check_{}_{}.ptx", label, pid));
    std::fs::write(&ptx_path, ptx.as_bytes()).expect("write ptx temp");

    let mut cubin_path = ptx_path.clone();
    cubin_path.set_extension("cubin");

    // Compile PTX to cubin via ptxas (sm_80 as the lowest-common target that
    // supports the full instruction set; sm_120 is the native target but
    // cuobjdump --dump-sass requires a matching toolkit version for Blackwell).
    let ptxas_out = Command::new("ptxas")
        .arg("--gpu-name")
        .arg("sm_80")
        .arg("-o")
        .arg(&cubin_path)
        .arg(&ptx_path)
        .output()
        .expect("ptxas not on PATH; install CUDA toolkit");

    assert!(
        ptxas_out.status.success(),
        "ptxas failed for SASS check [{}]:\nPTX file: {}\nSTDOUT:\n{}\nSTDERR:\n{}",
        label,
        ptx_path.display(),
        String::from_utf8_lossy(&ptxas_out.stdout),
        String::from_utf8_lossy(&ptxas_out.stderr)
    );

    // Dump SASS from the compiled cubin.
    let dump_out = Command::new("cuobjdump")
        .arg("--dump-sass")
        .arg(&cubin_path)
        .output()
        .expect("cuobjdump not on PATH; install CUDA toolkit");

    assert!(
        dump_out.status.success(),
        "cuobjdump failed [{}]:\nSTDOUT:\n{}\nSTDERR:\n{}",
        label,
        String::from_utf8_lossy(&dump_out.stdout),
        String::from_utf8_lossy(&dump_out.stderr)
    );

    let sass = String::from_utf8_lossy(&dump_out.stdout);

    // Match on space/tab boundary to avoid false positives from other opcodes
    // that share a prefix (e.g. LDS, LDG, LDGDEPBAR, LDL.LU all contain "LDL"
    // as a substring but are not register-spill loads).
    let ldl_count = sass
        .lines()
        .filter(|l| l.contains("LDL ") || l.contains("LDL\t"))
        .count();
    let stl_count = sass
        .lines()
        .filter(|l| l.contains("STL ") || l.contains("STL\t"))
        .count();

    // Print summary regardless so --nocapture shows the SASS line counts.
    let total_sass_lines = sass.lines().count();
    println!(
        "SASS summary [{}]: {} total lines, {} LDL (spill loads), {} STL (spill stores)",
        label, total_sass_lines, ldl_count, stl_count
    );

    assert_eq!(
        ldl_count,
        0,
        "tier_b2 dQ {} config emitted {} LDL (register spill loads); \
         spec requires 0 spills for admitted configs",
        label,
        ldl_count
    );
    assert_eq!(
        stl_count,
        0,
        "tier_b2 dQ {} config emitted {} STL (register spill stores); \
         spec requires 0 spills for admitted configs",
        label,
        stl_count
    );
}
