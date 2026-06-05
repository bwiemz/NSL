//! ptxas-probe test for the Tier B.2 projection-backward kernel
//! (`tier_b2_proj_backward`, Phase 3 T3). Modeled on
//! `tier_b2_dkdv_ptxas_probe.rs`.
//!
//! Asserts the emitted kernel assembles cleanly at sm_80 with 0 register
//! spills. If ptxas is not available on this machine the test self-skips
//! (it will run on the GPU host).
//!
//! Run with:
//!     cargo test -p nsl-codegen --test tier_b2_proj_backward_ptxas_probe -- --nocapture

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::proj_backward::synthesize_proj_backward;
use std::process::{Command, Stdio};

fn cfg(hd: i64, bq: i64, d_model: u32) -> FlashAttentionConfig {
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
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model,
            active_heads: 1,
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
    for v in ["v13.2", "v13.1"] {
        let win = format!(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{v}\bin\ptxas.exe"
        );
        if std::path::Path::new(&win).exists() {
            return Some(win);
        }
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
        "tier_b2_proj_backward_probe_{}_{}.ptx",
        sm,
        std::process::id()
    ));
    let tmp_cubin = std::env::temp_dir().join(format!(
        "tier_b2_proj_backward_probe_{}_{}.cubin",
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
            "ptxas {sm} rejected proj-backward kernel:\nSTDERR:{stderr}\nSTDOUT:{stdout}"
        ));
    }
    Ok(format!("{stderr}{stdout}"))
}

#[test]
fn proj_backward_ptxas_clean_sm80() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("SKIP: ptxas not on PATH (will run on GPU host)");
        return;
    };
    let ptx = synthesize_proj_backward(&cfg(64, 64, 64)).expect("synth ok");
    let stats = run_ptxas(&ptxas, ptx.as_bytes(), "sm_80")
        .expect("ptxas sm_80 must succeed");
    eprintln!("---- ptxas sm_80 stats (proj-backward T3) ----\n{stats}");
    assert!(
        !stats.contains("spill")
            || stats.contains("0 bytes spill stores, 0 bytes spill loads"),
        "ptxas sm_80 register spills:\n{stats}"
    );
}
