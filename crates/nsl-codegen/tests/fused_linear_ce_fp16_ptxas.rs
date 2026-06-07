//! Sprint v3-2 fp16 path — ptxas assembly validation.
//!
//! Synthesises the F16 forward, backward, and large-vocab (two-kernel)
//! modules and pipes them to `ptxas` targeting `sm_80`. Catches syntactic
//! or operand defects in the fp16 emitter that pure-Rust structural tests
//! can't see (undeclared registers, malformed pointer arithmetic, missing
//! cvt.f32.f16, label mismatches, etc.).
//!
//! Skipped (returns early) when `ptxas` isn't in PATH — keeps Linux dev
//! boxes and CI lanes without CUDA install green. Mirrors
//! `fused_linear_ce_large_vocab_ptxas.rs` for tooling-discovery style.

use nsl_codegen::fused_linear_ce::{
    Dtype, FusedLinearCEConfig, MAX_VOCAB_HARD_CEILING,
    synthesize_fused_linear_ce_backward_ptx, synthesize_fused_linear_ce_ptx,
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

/// Per-tag temp file helper — `tag` disambiguates concurrent invocations
/// from parallel test threads. Same shape as the large-vocab gate.
fn assemble_ptx(
    ptxas_path: &str,
    ptx_bytes: &[u8],
    sm_arch: &str,
    tag: &str,
) -> Result<(), String> {
    let pid = std::process::id();
    let in_path = std::env::temp_dir().join(format!("nsl_ptxas_fp16_in_{pid}_{tag}.ptx"));
    let out_path = std::env::temp_dir().join(format!("nsl_ptxas_fp16_out_{pid}_{tag}.cubin"));
    std::fs::write(&in_path, ptx_bytes)
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

fn fp16_small_cfg(vocab_size: u32, vocab_tile: u32, hidden: u32) -> FusedLinearCEConfig {
    FusedLinearCEConfig {
        vocab_size,
        hidden_size: hidden,
        seq_len: 64,
        batch_size: 2,
        vocab_tile,
        gpu_sm: 80,
        dtype: Dtype::F16,
        ignore_index: -100,
        max_vocab_v1: 8192,
    }
}

#[test]
fn fp16_forward_assembles_for_sm80_at_v4096() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found in PATH; skipping");
        return;
    };
    let cfg = fp16_small_cfg(4096, 1024, 128);
    cfg.validate().unwrap();
    assert!(!cfg.is_large_vocab());
    assert_eq!(cfg.dtype, Dtype::F16);

    let ptx = synthesize_fused_linear_ce_ptx(&cfg);

    // Sanity: the fp16 forward MUST include the .f16↔.f32 cvt instructions
    // (otherwise we accidentally emitted the F32 path). Pin this structurally
    // before we hand it to ptxas.
    let txt = std::str::from_utf8(&ptx).expect("ASCII");
    assert!(
        txt.contains("cvt.f32.f16"),
        "F16 forward MUST contain cvt.f32.f16 (load-from-HBM conversion)"
    );
    assert!(
        txt.contains("cvt.rn.f16.f32"),
        "F16 forward MUST contain cvt.rn.f16.f32 (store-to-SMEM conversion)"
    );
    assert!(
        txt.contains("ld.global.b16"),
        "F16 forward MUST load HBM as .b16"
    );
    assert!(
        txt.contains("st.shared.b16"),
        "F16 forward MUST store SMEM as .b16"
    );

    let tmp = std::env::temp_dir().join("fused_linear_ce_fp16_fwd_v4096.ptx");
    std::fs::write(&tmp, &ptx).ok();
    eprintln!("PTX dumped to {}", tmp.display());

    match assemble_ptx(&ptxas, &ptx, "sm_80", "fp16_fwd_v4096") {
        Ok(()) => eprintln!("ptxas accepted fp16 forward (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected fp16 forward kernel:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

#[test]
fn fp16_backward_assembles_for_sm80_at_v4096() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found in PATH; skipping");
        return;
    };
    let cfg = fp16_small_cfg(4096, 1024, 128);
    cfg.validate().unwrap();

    let ptx = synthesize_fused_linear_ce_backward_ptx(&cfg);
    let txt = std::str::from_utf8(&ptx).expect("ASCII");
    assert!(
        txt.contains("cvt.f32.f16"),
        "F16 backward MUST contain cvt.f32.f16 (fp16 HBM staging)"
    );
    assert!(
        txt.contains("red.global.add.f32"),
        "F16 backward dW/dx/dbias MUST use red.global.add.f32 \
         (master-grad convention; fp16 atomic adds are non-portable)"
    );
    assert!(
        txt.contains("ld.global.b16"),
        "F16 backward MUST load HBM as .b16"
    );

    let tmp = std::env::temp_dir().join("fused_linear_ce_fp16_bwd_v4096.ptx");
    std::fs::write(&tmp, &ptx).ok();

    match assemble_ptx(&ptxas, &ptx, "sm_80", "fp16_bwd_v4096") {
        Ok(()) => eprintln!("ptxas accepted fp16 backward (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected fp16 backward:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

#[test]
fn fp16_large_vocab_two_kernel_module_assembles_for_sm80_at_v49152() {
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
        dtype: Dtype::F16,
        ignore_index: -100,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().unwrap();
    assert!(cfg.is_large_vocab());

    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let txt = std::str::from_utf8(&ptx).expect("ASCII");

    // Both kernels must be in the module.
    assert!(txt.contains(&cfg.large_partials_kernel_name()));
    assert!(txt.contains(&cfg.large_finalize_kernel_name()));
    // Kernel A is fp16-staged.
    assert!(txt.contains("ld.global.b16"));
    assert!(txt.contains("cvt.f32.f16"));
    // Kernel A's partials write MUST stay f32 (per design — cross-CTA LSE
    // accuracy). Search for the canonical f32 partials store.
    assert!(
        txt.contains("st.global.f32 [%rd5],   %ftmax;"),
        "Kernel A partials MUST stay f32 (numerical robustness)"
    );

    let tmp = std::env::temp_dir().join("fused_linear_ce_fp16_large_v49152.ptx");
    std::fs::write(&tmp, &ptx).ok();
    eprintln!("PTX dumped to {}", tmp.display());

    match assemble_ptx(&ptxas, &ptx, "sm_80", "fp16_large_v49152") {
        Ok(()) => eprintln!("ptxas accepted fp16 large-vocab module"),
        Err(stderr) => panic!(
            "ptxas rejected fp16 large-vocab module:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

#[test]
fn fp16_intermediate_scale_assembles_for_sm80_at_v16384() {
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
        dtype: Dtype::F16,
        ignore_index: -100,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().unwrap();
    let ptx = synthesize_fused_linear_ce_ptx(&cfg);

    match assemble_ptx(&ptxas, &ptx, "sm_80", "fp16_v16384") {
        Ok(()) => eprintln!("ptxas accepted fp16 intermediate-scale module"),
        Err(stderr) => panic!("ptxas rejected fp16 sm_80 module:\n{stderr}"),
    }
}

#[test]
fn fp16_ptx_is_ascii_only() {
    // Unicode in PTX triggers CUDA_ERROR_INVALID_PTX under cudarc JIT
    // (same gate the F32 path's test_ptx_ascii_only enforces).
    let cfg_small = fp16_small_cfg(4096, 1024, 128);
    let fwd = synthesize_fused_linear_ce_ptx(&cfg_small);
    let bwd = synthesize_fused_linear_ce_backward_ptx(&cfg_small);
    for byte in fwd.iter().chain(bwd.iter()) {
        assert!(*byte < 128, "non-ASCII byte 0x{byte:02x} in fp16 v1 PTX");
    }

    let cfg_large = FusedLinearCEConfig {
        vocab_size: 49152,
        hidden_size: 128,
        seq_len: 64,
        batch_size: 2,
        vocab_tile: 128,
        gpu_sm: 80,
        dtype: Dtype::F16,
        ignore_index: -100,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    let ptx_large = synthesize_fused_linear_ce_ptx(&cfg_large);
    for byte in ptx_large.iter() {
        assert!(
            *byte < 128,
            "non-ASCII byte 0x{byte:02x} in fp16 large-vocab PTX"
        );
    }
}
