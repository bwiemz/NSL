//! FA-backward multi-warp campaign — real-ptxas assembly gate.
//!
//! The multi-warp (`_w4`) backward MAIN kernels partition the five MMA steps
//! across 4 warps with per-warp branch arms and element-linear scalar
//! sections. The segment-variant ptxas gate (`pca_stage_c_segment_kernels`)
//! only assembles single-warp bodies (segment configs stay warps=1), so this
//! gate assembles the three selector shapes' multi-warp modules with the real
//! `ptxas` at the sm_80 floor and the local card's sm_120. Skips when no
//! ptxas is on PATH (CI's no-cuda runners).
//!
//! Also pins the name<->warps contract: every synthesized multi-warp module
//! must carry `_w4` entries (plain + `_gqa`), and the kill-switch env is NOT
//! read here (that test needs its own binary — env races with parallel
//! synthesis).

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::flash_attention::{
    backward_select_blocks, synthesize_flash_attention_backward_ptx,
    FlashAttentionBackwardConfig,
};

fn find_ptxas() -> Option<String> {
    if let Ok(p) = std::env::var("PTXAS") {
        if std::path::Path::new(&p).is_file() {
            return Some(p);
        }
    }
    for cand in ["ptxas", "/usr/local/cuda/bin/ptxas", "/opt/cuda/bin/ptxas"] {
        if Command::new(cand).arg("--version").output().is_ok() {
            return Some(cand.into());
        }
    }
    None
}

fn assemble(ptxas: &str, ptx: &[u8], sm: &str, tag: &str) {
    let out = std::env::temp_dir().join(format!("fa_bwd_mw_{tag}_{sm}.cubin"));
    let mut cmd = Command::new(ptxas)
        .arg(format!("--gpu-name={sm}"))
        .arg("-o")
        .arg(&out)
        .arg("-")
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn ptxas");
    let end = ptx.iter().position(|&b| b == 0).unwrap_or(ptx.len());
    cmd.stdin.as_mut().unwrap().write_all(&ptx[..end]).unwrap();
    let fin = cmd.wait_with_output().unwrap();
    assert!(
        fin.status.success(),
        "ptxas rejected {tag} for {sm}:\n{}",
        String::from_utf8_lossy(&fin.stderr)
    );
    let _ = std::fs::remove_file(&out);
}

fn cfg_for(head_dim: i64, causal: bool) -> FlashAttentionBackwardConfig {
    let (block_q, block_kv) = backward_select_blocks(head_dim);
    FlashAttentionBackwardConfig {
        block_q,
        block_kv,
        head_dim,
        causal,
        gpu_sm: 80,
        segment_masked: false,
    }
}

/// Every selector shape's multi-warp phase-2 module assembles with the real
/// ptxas at the sm_80 floor and on the local card's architecture.
#[test]
fn ptxas_assembles_multiwarp_backward_variants() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("[fa-bwd-mw] ptxas not found on PATH — skipping assembly gate");
        return;
    };
    for hd in [32i64, 64, 128] {
        for causal in [true, false] {
            let config = cfg_for(hd, causal);
            let (_p1, p2) = synthesize_flash_attention_backward_ptx(&config);
            let text = String::from_utf8_lossy(&p2);
            assert!(
                text.contains("_w4"),
                "hd{hd} c{} phase-2 module lost the _w4 launch suffix — \
                 multi-warp emission did not engage:\n{}",
                causal as u8,
                &text[..text.len().min(400)]
            );
            let tag = format!("hd{hd}_c{}", causal as u8);
            assemble(&ptxas, &p2, "sm_80", &tag);
            assemble(&ptxas, &p2, "sm_120", &tag);
        }
    }
}

/// The partition scaffolding must actually fire in the multi-warp body:
/// per-warp nt arms (`_ARM0_END` labels) and the wid-derived m-loop init.
/// Vacuity guard for the whole campaign — if a refactor silently reverts to
/// the single-warp flow while keeping the name, this catches it.
#[test]
fn multiwarp_body_carries_partition_scaffolding() {
    let config = cfg_for(64, true); // 32/32: s_wnt = 2 -> arms exist
    let (_p1, p2) = synthesize_flash_attention_backward_ptx(&config);
    let text = String::from_utf8_lossy(&p2);
    assert!(
        text.contains("BWD_MMA_S_NT_ARM0_END"),
        "S-step nt arms missing — partition scaffolding did not fire"
    );
    assert!(
        text.contains("shr.u32 %bwd_mma_m_tile, %bwd_mma_wid"),
        "m-loop init is not wid-derived — warps are redundant, not partitioned"
    );
    assert!(
        text.contains("BWD_MAIN_MMA_P_ELEM"),
        "P section did not go element-linear in the multi-warp body"
    );
    // The historical warp-0 redundancy gate must be GONE from the multi-warp
    // body (ownership replaces it).
    assert!(
        !text.contains("setp.ne.u32 %bwd_mma_pw, %bwd_mma_wid, 0;"),
        "warp-0 redundancy gate survived — accumulates would double-count \
         under partitioned ownership"
    );
}
