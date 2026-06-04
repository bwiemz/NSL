//! Sprint-3 large-vocab path — ptxas-level assembly validation.
//!
//! Synthesises the two-kernel module returned by
//! `synthesize_fused_linear_ce_ptx` for vocab=49152 (NSL production) and
//! pipes it to `ptxas` targeting `sm_80`. Catches syntactic/operand defects
//! that pure-Rust structural tests can't see (undeclared registers,
//! malformed pointer arithmetic, label mismatches).
//!
//! Skipped (ignored) when `ptxas` isn't in PATH — keeps Linux dev boxes
//! and CI lanes without CUDA install green.

use nsl_codegen::fused_linear_ce::{
    Dtype, FusedLinearCEConfig, synthesize_fused_linear_ce_ptx, MAX_VOCAB_HARD_CEILING,
};
use std::io::Write;
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

fn assemble_ptx(ptxas_path: &str, ptx_bytes: &[u8], sm_arch: &str) -> Result<(), String> {
    let mut child = Command::new(ptxas_path)
        .args(["--gpu-name", sm_arch, "-O0", "-o", "-", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn ptxas: {e}"))?;

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(ptx_bytes)
        .map_err(|e| format!("failed to write PTX to ptxas stdin: {e}"))?;

    let out = child
        .wait_with_output()
        .map_err(|e| format!("ptxas wait failed: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

#[test]
fn large_vocab_two_kernel_module_assembles_for_sm80() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found in PATH; skipping");
        return;
    };

    let cfg = FusedLinearCEConfig {
        vocab_size: 49152,
        hidden_size: 128,
        seq_len: 64,
        batch_size: 2,
        vocab_tile: 128,
        gpu_sm: 80,
        dtype: Dtype::F32,
        ignore_index: -100,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().unwrap();
    assert!(cfg.is_large_vocab());

    let ptx = synthesize_fused_linear_ce_ptx(&cfg);

    // Write a copy for inspection on failure.
    let tmp = std::env::temp_dir().join("fused_linear_ce_large_v49152.ptx");
    std::fs::write(&tmp, &ptx).ok();
    eprintln!("PTX dumped to {}", tmp.display());

    match assemble_ptx(&ptxas, &ptx, "sm_80") {
        Ok(()) => eprintln!("ptxas accepted module (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected large-vocab two-kernel module:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

/// Different intermediate scale (vocab=16384, hd=64, S=32, B=1) — exercises
/// vocab_tile=256 (8192 < V < 32K — the lower end of the large-vocab path).
#[test]
fn large_vocab_intermediate_scale_assembles_for_sm80() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found in PATH; skipping");
        return;
    };

    let cfg = FusedLinearCEConfig {
        vocab_size: 16384,
        hidden_size: 64,
        seq_len: 32,
        batch_size: 1,
        vocab_tile: 256,
        gpu_sm: 80,
        dtype: Dtype::F32,
        ignore_index: -100,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().unwrap();
    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let tmp = std::env::temp_dir().join("fused_linear_ce_large_v16384_sm80.ptx");
    std::fs::write(&tmp, &ptx).ok();

    match assemble_ptx(&ptxas, &ptx, "sm_80") {
        Ok(()) => eprintln!("ptxas accepted intermediate-scale module"),
        Err(stderr) => panic!(
            "ptxas rejected sm_80 module:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}
