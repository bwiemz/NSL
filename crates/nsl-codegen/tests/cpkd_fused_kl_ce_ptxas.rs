//! CPKD fused KL-CE — ptxas assembly validation (mandatory from the first
//! commit for every new PTX emitter: WRGA B.3 retrospective / composition
//! paper F-04; see also bugs.md 2026-07-10 "assemble every PTX blob in CI").
//!
//! Synthesises the forward and backward modules across representative
//! shapes (small, asymmetric teacher/student hidden dims, and the v1
//! vocab ceiling) and pipes them to `ptxas` at `sm_80`. Catches
//! undeclared registers, malformed addressing, and label mismatches that
//! structural tests can't see.
//!
//! Skipped (returns early) when `ptxas` isn't in PATH — mirrors
//! `fused_linear_ce_fp16_ptxas.rs` for tooling discovery.

use nsl_codegen::cpkd_fused_loss::{
    FusedKlCeConfig, synthesize_fused_kl_ce_backward_ptx, synthesize_fused_kl_ce_ptx,
};
use std::process::{Command, Stdio};

fn find_ptxas() -> Option<String> {
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
    let win_default =
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe";
    if std::path::Path::new(win_default).exists() {
        return Some(win_default.to_string());
    }
    None
}

fn assemble_ptx(
    ptxas_path: &str,
    ptx_bytes: &[u8],
    sm_arch: &str,
    tag: &str,
) -> Result<(), String> {
    let pid = std::process::id();
    let in_path = std::env::temp_dir().join(format!("nsl_ptxas_klce_in_{pid}_{tag}.ptx"));
    let out_path = std::env::temp_dir().join(format!("nsl_ptxas_klce_out_{pid}_{tag}.cubin"));
    // Strip the cuModuleLoadData NUL terminator — ptxas treats an embedded
    // NUL as premature EOF.
    let text_end = ptx_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(ptx_bytes.len());
    std::fs::write(&in_path, &ptx_bytes[..text_end])
        .map_err(|e| format!("failed to write PTX temp file: {e}"))?;

    let out = Command::new(ptxas_path)
        .args([
            "--gpu-name",
            sm_arch,
            "-O0",
            "-o",
            out_path.to_str().unwrap(),
            in_path.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("failed to spawn ptxas: {e}"))?;

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);

    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "ptxas rejected {tag}:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

fn shapes() -> Vec<(FusedKlCeConfig, &'static str)> {
    vec![
        (
            FusedKlCeConfig {
                vocab_size: 256,
                student_hidden: 32,
                teacher_hidden: 64,
                batch_size: 2,
                seq_len: 4,
                vocab_tile: 128,
                gpu_sm: 80,
                ignore_index: -100,
            },
            "small_asym",
        ),
        (
            FusedKlCeConfig {
                vocab_size: 4096,
                student_hidden: 128,
                teacher_hidden: 128,
                batch_size: 2,
                seq_len: 32,
                vocab_tile: 1024,
                gpu_sm: 80,
                ignore_index: -100,
            },
            "mid_symmetric",
        ),
        (
            FusedKlCeConfig {
                vocab_size: 8192,
                student_hidden: 256,
                teacher_hidden: 512,
                batch_size: 4,
                seq_len: 64,
                vocab_tile: 2048,
                gpu_sm: 80,
                ignore_index: -100,
            },
            "v1_ceiling",
        ),
    ]
}

#[test]
fn fused_kl_ce_forward_assembles() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping assembly gate");
        return;
    };
    for (cfg, tag) in shapes() {
        cfg.validate().unwrap_or_else(|e| panic!("{tag}: {e}"));
        let ptx = synthesize_fused_kl_ce_ptx(&cfg);
        assemble_ptx(&ptxas, &ptx, "sm_80", &format!("fwd_{tag}"))
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

#[test]
fn fused_kl_ce_backward_assembles() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping assembly gate");
        return;
    };
    for (cfg, tag) in shapes() {
        let ptx = synthesize_fused_kl_ce_backward_ptx(&cfg);
        assemble_ptx(&ptxas, &ptx, "sm_80", &format!("bwd_{tag}"))
            .unwrap_or_else(|e| panic!("{e}"));
    }
}
