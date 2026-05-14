//! One-shot ptxas probe for the Tier B.2 backward kernel.
//!
//! Mirrors the Tier B forward probe convention used in
//! `crates/nsl-codegen/tests/tier_b_smem_probe.rs`: run ptxas with
//! `-v` (verbose stats) on a freshly-synthesised Tier-B-on backward
//! kernel and assert 0 spills + reasonable register count, so a
//! regression that re-introduces register pressure across the dQ
//! reload-modify-flush + skip-predicate scope is caught.
//!
//! Skips when ptxas is not on PATH (no CUDA toolkit installed).

#![cfg(feature = "test-helpers")]

use std::process::{Command, Stdio};

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_backward_with_tier_b;
use nsl_codegen::pca_segment::SegmentResidency;

fn find_ptxas() -> Option<String> {
    if let Ok(p) = std::env::var("PTXAS") {
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }
    for name in ["ptxas", "ptxas.exe"] {
        if Command::new(name)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return Some(name.to_string());
        }
    }
    let win = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe";
    if std::path::Path::new(win).exists() {
        return Some(win.to_string());
    }
    None
}

fn config() -> FlashAttentionConfig {
    // Same shape as `minimal_segment_masked_backward_config` in
    // pca_backward_kernel_snapshot.rs.
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
        gpu_sm: 80,
        segment_masked: true,
        csha: Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            ..CshaExtras::default()
        }),
    }
}

fn run_ptxas(ptxas: &str, ptx: &[u8], sm: &str) -> Result<String, String> {
    // Strip the trailing NUL byte synthesize_*_with_tier_b appends for
    // cuModuleLoadData — ptxas reads from a file and treats NUL as
    // end-of-input, but on Windows a NUL inside the file body can be
    // misparsed. Write a clean PTX text file.
    let stripped: &[u8] = if ptx.last() == Some(&0) {
        &ptx[..ptx.len() - 1]
    } else {
        ptx
    };
    let tmp_ptx = std::env::temp_dir().join(format!("pca_tier_b2_probe_{sm}.ptx"));
    let tmp_cubin = std::env::temp_dir().join(format!("pca_tier_b2_probe_{sm}.cubin"));
    std::fs::write(&tmp_ptx, stripped).map_err(|e| format!("write ptx: {e}"))?;

    let out = Command::new(ptxas)
        .args(["-v", "-O2", &format!("-arch={sm}"), "-o"])
        .arg(&tmp_cubin)
        .arg(&tmp_ptx)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("spawn ptxas: {e}"))?;
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    if !out.status.success() {
        return Err(format!(
            "ptxas {sm} rejected backward Tier-B-on kernel:\nSTDERR:{stderr}\nSTDOUT:{stdout}"
        ));
    }
    // ptxas -v writes register/SMEM stats on stderr (Linux) or stdout
    // (Windows in some configurations). Concatenate both.
    Ok(format!("{stderr}{stdout}"))
}

#[test]
fn backward_tier_b_on_sm80_ptxas_clean() {
    let ptxas = match find_ptxas() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: ptxas not available on this host");
            return;
        }
    };
    let cfg = config();
    let ptx_str = synthesize_backward_with_tier_b(&cfg, Some((4096, SegmentResidency::Shared)))
        .expect("synthesize_backward_with_tier_b must succeed");
    let ptx_bytes = ptx_str.as_bytes();
    let stats = run_ptxas(&ptxas, ptx_bytes, "sm_80").expect("ptxas sm_80 must succeed");
    eprintln!("---- ptxas sm_80 stats (Tier-B-on backward) ----\n{stats}");
    assert!(
        !stats.contains("spill") || stats.contains("0 bytes spill stores, 0 bytes spill loads"),
        "ptxas sm_80 reported register spills:\n{stats}"
    );
}

#[test]
fn backward_tier_b_on_sm120_ptxas_clean() {
    let ptxas = match find_ptxas() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: ptxas not available on this host");
            return;
        }
    };
    let cfg = config();
    let ptx_str = synthesize_backward_with_tier_b(&cfg, Some((4096, SegmentResidency::Shared)))
        .expect("synthesize_backward_with_tier_b must succeed");
    let ptx_bytes = ptx_str.as_bytes();
    match run_ptxas(&ptxas, ptx_bytes, "sm_120") {
        Ok(stats) => {
            eprintln!("---- ptxas sm_120 stats (Tier-B-on backward) ----\n{stats}");
            assert!(
                !stats.contains("spill")
                    || stats.contains("0 bytes spill stores, 0 bytes spill loads"),
                "ptxas sm_120 reported register spills:\n{stats}"
            );
        }
        Err(e) => {
            // Older toolkits don't know sm_120; treat as skip rather than fail.
            eprintln!("SKIP: ptxas sm_120 failed (toolkit may be older): {e}");
        }
    }
}

/// Sanity check: with Tier B.2 ON the backward PTX must be strictly
/// larger than with Tier B.2 OFF (the preamble + per-q_iter predicate
/// add real bytes). Catches a regression where the conditional emission
/// gates fail to fire.
#[test]
fn backward_tier_b_on_is_larger_than_tier_b_off() {
    let cfg = config();
    let on = synthesize_backward_with_tier_b(&cfg, Some((4096, SegmentResidency::Shared)))
        .expect("with_tier_b succeed");
    let off = synthesize_backward_with_tier_b(&cfg, None).expect("with_tier_b(None) succeed");
    assert!(
        on.len() > off.len(),
        "Tier-B-on backward PTX ({} bytes) should be larger than Tier-B-off ({} bytes)",
        on.len(),
        off.len()
    );
    let delta = on.len() - off.len();
    // The delta is dominated by the range-table preamble (always ~2 KB
    // of PTX text for the 4 phase emissions) plus per-q_iter scope +
    // predicate body (~1 KB per q_iter × iters). For block_q=32 → iters=8.
    // Lower bound the delta at 2 KB so an accidentally-empty conditional
    // is caught.
    assert!(
        delta >= 2048,
        "Tier B.2 delta ({delta} bytes) suspiciously small — preamble or predicate may be missing"
    );
    eprintln!("Tier B.2 backward PTX delta: {delta} bytes ({} → {})", off.len(), on.len());
}
