//! ptxas-probe tests for the Tier B.2 dK/dV backward kernel (Phase 3a Task 4).
//!
//! Step 1 (TDD): these tests exist BEFORE the implementation so we can confirm
//! they fail first, then pass once S=Q*K^T + P-recompute + tile-skip are wired.
//!
//! Run with:
//!     cargo test --package nsl-codegen --test tier_b2_dkdv_ptxas_probe -- --nocapture

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::dkdv::synthesize_dkdv_kernel;
use std::process::{Command, Stdio};

fn cfg(hd: i64, bq: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq,
        block_kv: bq,
        head_dim: hd,
        causal: true,
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

fn run_ptxas(ptxas: &str, ptx: &[u8], sm: &str) -> Result<String, String> {
    let stripped: &[u8] = if ptx.last() == Some(&0) {
        &ptx[..ptx.len() - 1]
    } else {
        ptx
    };
    let tmp_ptx = std::env::temp_dir().join(format!(
        "tier_b2_dkdv_probe_{}_{}.ptx",
        sm,
        std::process::id()
    ));
    let tmp_cubin = std::env::temp_dir().join(format!(
        "tier_b2_dkdv_probe_{}_{}.cubin",
        sm,
        std::process::id()
    ));
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
            "ptxas {sm} rejected dK/dV kernel:\nSTDERR:{stderr}\nSTDOUT:{stdout}"
        ));
    }
    // ptxas -v writes register/SMEM stats on stderr (Linux) or stdout (Windows).
    Ok(format!("{stderr}{stdout}"))
}

#[test]
fn dkdv_kernel_contains_qkt_mma_and_tile_skip() {
    let ptx = synthesize_dkdv_kernel(&cfg(64, 64)).expect("synth ok");
    assert!(
        ptx.contains("mma.sync.aligned.m16n8k16"),
        "MMA-1 S=Q*K^T present: PTX must contain mma.sync.aligned.m16n8k16"
    );
    assert!(
        ptx.contains("tile_skip") || ptx.contains("p_tile_active"),
        "tile-skip predicate present: PTX must reference tile_skip_predicate or p_tile_active"
    );
}

#[test]
fn dkdv_kernel_contains_dp_mma_and_ds_combine() {
    let ptx = synthesize_dkdv_kernel(&cfg(64, 64)).expect("synth ok");
    let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
    assert!(mma_count >= 2, "expected >=2 MMAs after dP, got {mma_count}");
    assert!(ptx.contains("// === dS = P * (dP - D)"), "dS combine present");
    // The 1/sqrt(D) scale must be applied to dS (the fresh-branch fix).
    assert!(ptx.contains("mul.f32 %ds_0, %ds_0, %f_scale"),
        "dS must carry the 1/sqrt(D) f_scale factor");
}

#[test]
fn dkdv_kernel_ptxas_clean_sm80() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("SKIP: ptxas not on PATH");
        return;
    };
    let ptx = synthesize_dkdv_kernel(&cfg(128, 32)).expect("synth ok");
    let stats =
        run_ptxas(&ptxas, ptx.as_bytes(), "sm_80").expect("ptxas sm_80 must succeed");
    eprintln!("---- ptxas sm_80 stats (dK/dV Task 4) ----\n{stats}");
    assert!(
        !stats.contains("spill")
            || stats.contains("0 bytes spill stores, 0 bytes spill loads"),
        "ptxas sm_80 register spills:\n{stats}"
    );
}
