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

// ─── Review Finding 2 — F16 tail-zero sentinel regression guard ─────────
//
// Pre-fix the large-vocab partials F16 kernel mov'd the f32 bit-pattern
// `0f80800000` (which is -1.175e-38, NOT -INF) into %facc on the
// tail-zero branch, then cvt.rn.f16.f32'd it through the shared store
// path.  The cvt mapped -1.175e-38 to fp16 0x0000 (below the fp16
// subnormal range), and the downstream max-reduce over the smem tile
// then folded 0.0 into the per-tile max — corrupting the cross-CTA LSE
// when all real logits were negative.  Latent because every existing
// fixture used cleanly-divisible vocab (vocab % vocab_tile == 0), so no
// tail slot was ever exercised by a tile that also held real logits.
//
// The fix branches around the cvt and writes the fp16 -INF bit-pattern
// (`0xFC00`) directly via `mov.b16 %h2, 0xFC00 ; st.shared.b16 …`.
//
// This regression test asserts:
//   1. The fixed sentinel pattern is structurally present.
//   2. The old broken pattern (`mov.f32 %facc, 0f80800000` immediately
//      before a fall-through into the cvt-store path) is GONE.
//   3. The kernel still assembles via ptxas.
//
// Non-divisor vocab is also accepted by the kernel — V=49153 produces
// `ceil(49153/128) = 385` tiles where the last tile holds 1 real logit
// + 127 tail-zero slots.  The PTX emitter does not parameterise on
// (V % vocab_tile), so any large-vocab F16 invocation suffices to
// exercise the structural check; we still pick a non-divisor V here so
// future GPU runs can pick this fixture up unchanged.

#[test]
fn fp16_large_vocab_tail_zero_writes_fp16_neg_inf_directly() {
    let cfg = FusedLinearCEConfig {
        vocab_size: 49153,
        hidden_size: 128,
        seq_len: 32,
        batch_size: 1,
        vocab_tile: 128,
        gpu_sm: 80,
        dtype: Dtype::F16,
        ignore_index: -100,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().expect("V=49153 must validate (no V%vocab_tile gate)");
    assert!(cfg.is_large_vocab(), "fixture must route through the F16 large-vocab path");

    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let txt = std::str::from_utf8(&ptx).expect("PTX must be ASCII");

    // (1) Direct fp16 -INF write at the tail-zero label.
    assert!(
        txt.contains("LP_INNER_TAIL_ZERO:"),
        "Large-vocab fp16 partials must keep the LP_INNER_TAIL_ZERO label"
    );
    assert!(
        txt.contains("mov.b16 %h2, 0xFC00"),
        "Tail-zero branch MUST write fp16 -INF (0xFC00) directly — the \
         previous f32 sentinel `0f80800000` was -1.175e-38, NOT -INF, \
         and cvt'd to fp16 0x0000 (Finding 2)"
    );
    // (2) The broken pattern must be GONE from the F16 large-vocab
    // partials kernel.  Search restricted: scope to the partials
    // kernel by anchoring on the kernel name and stopping at the
    // finalize entry-point.  This prevents false positives from
    // unrelated f32 sentinels in the finalize kernel (which reads
    // f32 partials and legitimately uses 0f80800000 for its f32 max
    // accumulator init).
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
        "Partials kernel must NOT mov 0f80800000 into %facc — that was \
         the broken sentinel that round-tripped through cvt.rn.f16.f32 \
         to fp16 0x0000 (Finding 2 regression guard)"
    );

    // (3) ptxas-assemble.
    if let Some(ptxas) = find_ptxas() {
        let tmp = std::env::temp_dir()
            .join("fused_linear_ce_fp16_large_v49153_finding2.ptx");
        std::fs::write(&tmp, &ptx).ok();
        match assemble_ptx(&ptxas, &ptx, "sm_80", "fp16_large_v49153_f2") {
            Ok(()) => eprintln!("ptxas accepted V=49153 fp16 module"),
            Err(stderr) => panic!(
                "ptxas rejected V=49153 fp16 module after Finding 2 fix:\n{stderr}\n\
                 PTX at {}", tmp.display()
            ),
        }
    } else {
        eprintln!("ptxas not found in PATH; structural assertions still ran");
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
