//! CFTP v7 — ptxas gate for the four precision-cast PTX kernels.
//!
//! Pipes each of `synthesize_{f32_to_bf16, bf16_to_f32, f32_to_fp16,
//! fp16_to_f32}_ptx()` through `ptxas -arch=sm_80 -O0`. Catches the same
//! class of bugs structural string-asserts can't: undeclared registers,
//! malformed cvt operand types, label mismatches, illegal PTX-version /
//! mnemonic combinations.
//!
//! Skipped (early return) when `ptxas` isn't in PATH — keeps non-CUDA dev
//! boxes and CI lanes green. Mirrors the skip pattern in
//! `fused_linear_ce_bf16_ptxas.rs`.

use nsl_codegen::precision_cast_ptx::{
    synthesize_bf16_to_f32_ptx, synthesize_f32_to_bf16_ptx, synthesize_f32_to_fp16_ptx,
    synthesize_fp16_to_f32_ptx,
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

/// `tag` disambiguates concurrent invocations from parallel test threads.
fn assemble_ptx(
    ptxas_path: &str,
    ptx_bytes: &[u8],
    sm_arch: &str,
    tag: &str,
) -> Result<(), String> {
    let pid = std::process::id();
    let in_path = std::env::temp_dir().join(format!("nsl_cast_ptxas_in_{pid}_{tag}.ptx"));
    let out_path =
        std::env::temp_dir().join(format!("nsl_cast_ptxas_out_{pid}_{tag}.cubin"));
    // The emitter NUL-terminates so `cuModuleLoadData` (C-string reader) is
    // happy. `ptxas` reads from a *file* (byte stream), and recent ptxas
    // versions (e.g. CUDA 13.2 ptxas V13.2.51) reject a trailing NUL as
    // "Unexpected EOF". Strip the terminator before staging the temp file.
    let staged: &[u8] = match ptx_bytes.last() {
        Some(0) => &ptx_bytes[..ptx_bytes.len() - 1],
        _ => ptx_bytes,
    };
    std::fs::write(&in_path, staged)
        .map_err(|e| format!("failed to write PTX temp file: {e}"))?;

    let mut child = Command::new(ptxas_path)
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
        .spawn()
        .map_err(|e| format!("failed to spawn ptxas: {e}"))?;

    let _ = &mut child;
    let _ = std::io::stdout().flush();

    let out = child
        .wait_with_output()
        .map_err(|e| format!("ptxas wait failed: {e}"))?;

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);

    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

#[test]
fn f32_to_bf16_assembles_for_sm80() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping");
        return;
    };
    let ptx = synthesize_f32_to_bf16_ptx();
    let tmp = std::env::temp_dir().join("nsl_cast_f32_to_bf16.ptx");
    std::fs::write(&tmp, &ptx).ok();
    match assemble_ptx(&ptxas, &ptx, "sm_80", "f32_to_bf16") {
        Ok(()) => eprintln!("ptxas accepted nsl_cast_f32_to_bf16 (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected nsl_cast_f32_to_bf16:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

#[test]
fn bf16_to_f32_assembles_for_sm80() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping");
        return;
    };
    let ptx = synthesize_bf16_to_f32_ptx();
    let tmp = std::env::temp_dir().join("nsl_cast_bf16_to_f32.ptx");
    std::fs::write(&tmp, &ptx).ok();
    match assemble_ptx(&ptxas, &ptx, "sm_80", "bf16_to_f32") {
        Ok(()) => eprintln!("ptxas accepted nsl_cast_bf16_to_f32 (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected nsl_cast_bf16_to_f32:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

#[test]
fn f32_to_fp16_assembles_for_sm80() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping");
        return;
    };
    let ptx = synthesize_f32_to_fp16_ptx();
    let tmp = std::env::temp_dir().join("nsl_cast_f32_to_fp16.ptx");
    std::fs::write(&tmp, &ptx).ok();
    match assemble_ptx(&ptxas, &ptx, "sm_80", "f32_to_fp16") {
        Ok(()) => eprintln!("ptxas accepted nsl_cast_f32_to_fp16 (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected nsl_cast_f32_to_fp16:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

#[test]
fn fp16_to_f32_assembles_for_sm80() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping");
        return;
    };
    let ptx = synthesize_fp16_to_f32_ptx();
    let tmp = std::env::temp_dir().join("nsl_cast_fp16_to_f32.ptx");
    std::fs::write(&tmp, &ptx).ok();
    match assemble_ptx(&ptxas, &ptx, "sm_80", "fp16_to_f32") {
        Ok(()) => eprintln!("ptxas accepted nsl_cast_fp16_to_f32 (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected nsl_cast_fp16_to_f32:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

#[test]
fn all_four_cast_kernels_are_ascii_only() {
    // Unicode in PTX triggers CUDA_ERROR_INVALID_PTX under cudarc JIT.
    for (tag, bytes) in [
        ("f32_to_bf16", synthesize_f32_to_bf16_ptx()),
        ("bf16_to_f32", synthesize_bf16_to_f32_ptx()),
        ("f32_to_fp16", synthesize_f32_to_fp16_ptx()),
        ("fp16_to_f32", synthesize_fp16_to_f32_ptx()),
    ] {
        for &b in &bytes {
            assert!(
                b < 128,
                "{tag} PTX must be ASCII-only; found 0x{b:02x}"
            );
        }
    }
}
