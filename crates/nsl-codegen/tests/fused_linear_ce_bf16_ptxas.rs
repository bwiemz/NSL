//! Sprint v4-1 bf16 path — ptxas assembly validation.
//!
//! Mirrors `fused_linear_ce_fp16_ptxas.rs` for the bf16 (Brain-float) dtype.
//! Synthesises the Bf16 forward, backward, and large-vocab (two-kernel)
//! modules and pipes them to `ptxas` targeting `sm_80`. Catches syntactic
//! or operand defects in the bf16 emitter that pure-Rust structural tests
//! can't see (undeclared registers, malformed cvt mnemonics, label
//! mismatches, the wrong -INF sentinel, etc.).
//!
//! Skipped (returns early) when `ptxas` isn't in PATH — keeps Linux dev
//! boxes and CI lanes without CUDA install green.
//!
//! ## Why bf16 needs PTX 8.0
//!
//! `cvt.f32.bf16` / `cvt.rn.bf16.f32` were introduced in PTX ISA 7.8 and
//! require `.target sm_80+`. The Bf16 path bumps the header to
//! `.version 8.0` (via `FusedLinearCEConfig::ptx_header()` dtype dispatch);
//! F32 and F16 stay at `.version 7.0` so their byte-identity snapshots
//! remain pinned to pre-v4 PTX.

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
/// from parallel test threads.
fn assemble_ptx(
    ptxas_path: &str,
    ptx_bytes: &[u8],
    sm_arch: &str,
    tag: &str,
) -> Result<(), String> {
    let pid = std::process::id();
    let in_path = std::env::temp_dir().join(format!("nsl_ptxas_bf16_in_{pid}_{tag}.ptx"));
    let out_path = std::env::temp_dir().join(format!("nsl_ptxas_bf16_out_{pid}_{tag}.cubin"));
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

fn bf16_small_cfg(vocab_size: u32, vocab_tile: u32, hidden: u32) -> FusedLinearCEConfig {
    FusedLinearCEConfig {
        vocab_size,
        hidden_size: hidden,
        seq_len: 64,
        batch_size: 2,
        vocab_tile,
        gpu_sm: 80,
        dtype: Dtype::Bf16,
        ignore_index: -100,
        max_vocab_v1: 8192,
    }
}

#[test]
fn bf16_forward_assembles_for_sm80_at_v4096() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found in PATH; skipping");
        return;
    };
    let cfg = bf16_small_cfg(4096, 1024, 128);
    cfg.validate().unwrap();
    assert!(!cfg.is_large_vocab());
    assert_eq!(cfg.dtype, Dtype::Bf16);

    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let txt = std::str::from_utf8(&ptx).expect("ASCII");

    // Structural pin: bf16 cvt mnemonics must be present, and the bf16-
    // specific PTX version bump must have taken effect.
    assert!(
        txt.contains("cvt.f32.bf16"),
        "Bf16 forward MUST contain cvt.f32.bf16 (load-from-HBM conversion)"
    );
    assert!(
        txt.contains("cvt.rn.bf16.f32"),
        "Bf16 forward MUST contain cvt.rn.bf16.f32 (store-to-SMEM conversion)"
    );
    assert!(
        txt.contains("ld.global.b16"),
        "Bf16 forward MUST load HBM as .b16 (dtype-agnostic storage class)"
    );
    assert!(
        txt.contains("st.shared.b16"),
        "Bf16 forward MUST store SMEM as .b16"
    );
    assert!(
        txt.contains(".version 8.0"),
        "Bf16 path MUST bump to .version 8.0 (bf16 cvt requires PTX 7.8+; \
         F32/F16 stay at 7.0 to preserve byte-identity)"
    );
    // Negative pin: must NOT contain the f16 cvt mnemonics (that would
    // mean we accidentally fell through to the F16 emitter).
    assert!(
        !txt.contains("cvt.f32.f16"),
        "Bf16 forward must NOT contain cvt.f32.f16 (would indicate F16 path leak)"
    );
    assert!(
        !txt.contains("cvt.rn.f16.f32"),
        "Bf16 forward must NOT contain cvt.rn.f16.f32"
    );

    let tmp = std::env::temp_dir().join("fused_linear_ce_bf16_fwd_v4096.ptx");
    std::fs::write(&tmp, &ptx).ok();
    eprintln!("PTX dumped to {}", tmp.display());

    match assemble_ptx(&ptxas, &ptx, "sm_80", "bf16_fwd_v4096") {
        Ok(()) => eprintln!("ptxas accepted bf16 forward (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected bf16 forward kernel:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

#[test]
fn bf16_backward_assembles_for_sm80_at_v4096() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found in PATH; skipping");
        return;
    };
    let cfg = bf16_small_cfg(4096, 1024, 128);
    cfg.validate().unwrap();

    let ptx = synthesize_fused_linear_ce_backward_ptx(&cfg);
    let txt = std::str::from_utf8(&ptx).expect("ASCII");
    assert!(
        txt.contains("cvt.f32.bf16"),
        "Bf16 backward MUST contain cvt.f32.bf16 (bf16 HBM staging)"
    );
    assert!(
        txt.contains("red.global.add.f32"),
        "Bf16 backward dW/dx/dbias MUST use red.global.add.f32 \
         (master-grad convention; bf16 atomic adds are non-portable)"
    );
    assert!(
        txt.contains("ld.global.b16"),
        "Bf16 backward MUST load HBM as .b16"
    );
    assert!(
        txt.contains(".version 8.0"),
        "Bf16 backward MUST bump to .version 8.0"
    );
    assert!(
        !txt.contains("cvt.f32.f16"),
        "Bf16 backward must NOT contain cvt.f32.f16"
    );

    let tmp = std::env::temp_dir().join("fused_linear_ce_bf16_bwd_v4096.ptx");
    std::fs::write(&tmp, &ptx).ok();

    match assemble_ptx(&ptxas, &ptx, "sm_80", "bf16_bwd_v4096") {
        Ok(()) => eprintln!("ptxas accepted bf16 backward (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected bf16 backward:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

#[test]
fn bf16_large_vocab_two_kernel_module_assembles_for_sm80_at_v49152() {
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
        dtype: Dtype::Bf16,
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
    // Kernel A is bf16-staged.
    assert!(txt.contains("ld.global.b16"));
    assert!(txt.contains("cvt.f32.bf16"));
    assert!(txt.contains(".version 8.0"));
    // Kernel A's partials write MUST stay f32 (per design — cross-CTA LSE
    // accuracy). Search for the canonical f32 partials store.
    assert!(
        txt.contains("st.global.f32 [%rd5],   %ftmax;"),
        "Kernel A partials MUST stay f32 (numerical robustness)"
    );

    let tmp = std::env::temp_dir().join("fused_linear_ce_bf16_large_v49152.ptx");
    std::fs::write(&tmp, &ptx).ok();
    eprintln!("PTX dumped to {}", tmp.display());

    match assemble_ptx(&ptxas, &ptx, "sm_80", "bf16_large_v49152") {
        Ok(()) => eprintln!("ptxas accepted bf16 large-vocab module"),
        Err(stderr) => panic!(
            "ptxas rejected bf16 large-vocab module:\n{stderr}\nPTX at {}",
            tmp.display()
        ),
    }
}

#[test]
fn bf16_intermediate_scale_assembles_for_sm80_at_v16384() {
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
        dtype: Dtype::Bf16,
        ignore_index: -100,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().unwrap();
    let ptx = synthesize_fused_linear_ce_ptx(&cfg);

    match assemble_ptx(&ptxas, &ptx, "sm_80", "bf16_v16384") {
        Ok(()) => eprintln!("ptxas accepted bf16 intermediate-scale module"),
        Err(stderr) => panic!("ptxas rejected bf16 sm_80 module:\n{stderr}"),
    }
}

// ─── Sprint v4-1 — Bf16 tail-zero sentinel regression guard ─────────────
//
// Mirrors `fused_linear_ce_fp16_ptxas::fp16_large_vocab_tail_zero_writes_
// fp16_neg_inf_directly`. For bf16 the correct -INF bit pattern is
// `0xFF80` (sign bit + all-ones 8-bit exponent + zero 7-bit mantissa);
// f16's `0xFC00` would round-trip through cvt.rn.bf16.f32 to a finite
// value (since the bf16 mantissa is smaller than f16's) and silently
// corrupt the per-tile max-reduce — same Finding-2 hazard class.
//
// Non-divisor V=49153 with vocab_tile=128 produces `ceil(49153/128) =
// 385` tiles where the last tile holds 1 real logit + 127 tail-zero
// slots — the exact path that exposes the sentinel bug at scale. We
// also assert via a scoped substring search that the broken f32-sentinel-
// then-cvt pattern is absent from the partials kernel.

#[test]
fn bf16_large_vocab_tail_zero_writes_bf16_neg_inf_directly() {
    let cfg = FusedLinearCEConfig {
        vocab_size: 49153,
        hidden_size: 128,
        seq_len: 32,
        batch_size: 1,
        vocab_tile: 128,
        gpu_sm: 80,
        dtype: Dtype::Bf16,
        ignore_index: -100,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().expect("V=49153 must validate (no V%vocab_tile gate)");
    assert!(cfg.is_large_vocab(), "fixture must route through the Bf16 large-vocab path");

    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let txt = std::str::from_utf8(&ptx).expect("PTX must be ASCII");

    // (1) Direct bf16 -INF write at the tail-zero label.
    assert!(
        txt.contains("LP_INNER_TAIL_ZERO:"),
        "Large-vocab bf16 partials must keep the LP_INNER_TAIL_ZERO label"
    );
    assert!(
        txt.contains("mov.b16 %h2, 0xFF80"),
        "Tail-zero branch MUST write bf16 -INF (0xFF80) directly — using \
         the fp16 -INF pattern (0xFC00) here would silently corrupt the \
         per-tile max-reduce, the same hazard class as fp16's Finding 2"
    );
    // (2) The broken f32-sentinel-then-cvt pattern must NOT appear in the
    // partials kernel section. Scope the search to the partials kernel
    // to avoid false-positives in the finalize kernel (which legitimately
    // uses 0f80800000 for its f32 max accumulator init).
    let partials_name = cfg.large_partials_kernel_name();
    let finalize_name = cfg.large_finalize_kernel_name();
    let partials_start = txt
        .find(&partials_name)
        .expect("PTX must contain the partials kernel entry");
    let partials_end = txt[partials_start..]
        .find(&finalize_name)
        .map(|off| partials_start + off)
        .unwrap_or(txt.len());
    let partials_section = &txt[partials_start..partials_end];
    assert!(
        !partials_section.contains("mov.f32 %facc, 0f80800000"),
        "Partials kernel must NOT mov 0f80800000 into %facc — that's the \
         broken f32-sentinel-then-cvt pattern that round-trips through \
         cvt.rn.bf16.f32 to a finite value and corrupts the max-reduce"
    );
    // (3) Negative pin against the fp16 sentinel value — copy-paste hazard
    // from the fp16 emitter would have left 0xFC00 in place.
    assert!(
        !partials_section.contains("mov.b16 %h2, 0xFC00"),
        "Partials kernel must NOT write 0xFC00 — that's fp16 -INF, not \
         bf16 -INF. Mixing them would silently corrupt the per-tile \
         max-reduce when all real logits in the tail-zero-bearing tile \
         are negative."
    );

    // (4) ptxas-assemble.
    if let Some(ptxas) = find_ptxas() {
        let tmp = std::env::temp_dir()
            .join("fused_linear_ce_bf16_large_v49153_sentinel.ptx");
        std::fs::write(&tmp, &ptx).ok();
        match assemble_ptx(&ptxas, &ptx, "sm_80", "bf16_large_v49153_s") {
            Ok(()) => eprintln!("ptxas accepted V=49153 bf16 module"),
            Err(stderr) => panic!(
                "ptxas rejected V=49153 bf16 module:\n{stderr}\n\
                 PTX at {}", tmp.display()
            ),
        }
    } else {
        eprintln!("ptxas not found in PATH; structural assertions still ran");
    }
}

#[test]
fn bf16_ptx_is_ascii_only() {
    // Unicode in PTX triggers CUDA_ERROR_INVALID_PTX under cudarc JIT
    // (same gate the F32 and F16 paths enforce). Sprint v3-2 caught
    // em-dashes in F16 PTX comments — repeat the discipline here.
    let cfg_small = bf16_small_cfg(4096, 1024, 128);
    let fwd = synthesize_fused_linear_ce_ptx(&cfg_small);
    let bwd = synthesize_fused_linear_ce_backward_ptx(&cfg_small);
    for byte in fwd.iter().chain(bwd.iter()) {
        assert!(*byte < 128, "non-ASCII byte 0x{byte:02x} in bf16 v1 PTX");
    }

    let cfg_large = FusedLinearCEConfig {
        vocab_size: 49152,
        hidden_size: 128,
        seq_len: 64,
        batch_size: 2,
        vocab_tile: 128,
        gpu_sm: 80,
        dtype: Dtype::Bf16,
        ignore_index: -100,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    let ptx_large = synthesize_fused_linear_ce_ptx(&cfg_large);
    for byte in ptx_large.iter() {
        assert!(
            *byte < 128,
            "non-ASCII byte 0x{byte:02x} in bf16 large-vocab PTX"
        );
    }
}
