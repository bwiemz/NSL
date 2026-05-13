//! Verify the Tier B.1 canonical scaffold compiles to SASS with zero register
//! spill loads (LDL) and zero register spill stores (STL).
//!
//! Spec section 11 contingency: spill-free at all admitted V3 supported-matrix
//! configs. The uniform-SMEM placeholder architecture minimises live registers
//! per warp, so the canonical 32x32x32 causal config is expected to be clean.
//!
//! These tests are `#[ignore]`-gated because they require both `ptxas` and
//! `cuobjdump` from a CUDA toolkit. Run with:
//!     cargo test --package nsl-codegen --test tier_b1_kernel_sass_no_spill \
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

#[test]
#[ignore = "requires ptxas + cuobjdump on PATH; lift in CI with CUDA toolchain"]
fn tier_b1_canonical_no_register_spills() {
    let config = canonical_config();
    let chunk: u32 = 128;
    let mut ptx_bytes = synthesize(&config, chunk);
    // synthesize appends a NUL terminator; strip it so the file is valid text.
    if ptx_bytes.last() == Some(&0) {
        ptx_bytes.pop();
    }

    let pid = std::process::id();
    let mut ptx_path = std::env::temp_dir();
    ptx_path.push(format!("tier_b1_sass_check_{}.ptx", pid));
    std::fs::write(&ptx_path, &ptx_bytes).expect("write ptx temp");

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
        "ptxas failed for SASS check:\nPTX file: {}\nSTDOUT:\n{}\nSTDERR:\n{}",
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
        "cuobjdump failed:\nSTDOUT:\n{}\nSTDERR:\n{}",
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
        "SASS summary: {} total lines, {} LDL (spill loads), {} STL (spill stores)",
        total_sass_lines, ldl_count, stl_count
    );

    assert_eq!(
        ldl_count,
        0,
        "tier_b1 canonical config emitted {} LDL (register spill loads); \
         spec section 11 requires 0 for admitted V3 supported-matrix configs",
        ldl_count
    );
    assert_eq!(
        stl_count,
        0,
        "tier_b1 canonical config emitted {} STL (register spill stores); \
         spec section 11 requires 0 for admitted V3 supported-matrix configs",
        stl_count
    );
}
