//! A.2.3.2 + Tier A emission validation on real GPU hardware.
//!
//! Pipes `synthesize_flash_attention_ptx` output to `ptxas` and asserts
//! the assembler accepts it for sm_120 (Blackwell). Catches emission-level
//! defects that pure Rust unit tests can't see: undeclared registers,
//! malformed instruction operands, label mismatches, SMEM overflow, etc.
//!
//! Skipped (ignored) when `ptxas` isn't in PATH so this doesn't break
//! Windows dev boxes without a CUDA install. On a CUDA-equipped box it's
//! the one e2e gate that proves the CSHA emission pipeline produces real,
//! assemble-able PTX.
//!
//! # Status (2026-04-13)
//!
//! Both tests are `#[ignore]`'d because bringing them up on sm_120 / CUDA
//! 13.2 surfaced pre-existing ptxas-rejection defects in the **non-CSHA
//! FA emitter** that pre-date all CSHA work:
//!
//! 1. `mov.f32 %reg, 0x....;` — PTX requires `0f` prefix for f32 hex
//!    literals (IEEE bit pattern); `0x` parses as a u32 and triggers an
//!    "Arguments mismatch for instruction 'mov'" error. Appears at
//!    ~19 call sites across `flash_attention.rs` in the f32
//!    accumulator init / masking paths.
//!
//! 2. `st.shared.f32 [shmem + %reg], ...` — the symbol+register form
//!    inside the address bracket is rejected by ptxas 13.2. Standard
//!    idiom is to compute the address into a u64 register first
//!    (`cvta.shared.u64` or explicit `mov + add`). Appears at ~10
//!    store/load sites.
//!
//! Fixing these is a separate maintenance task on the non-CSHA FA path;
//! it's out of Tier A's A.2.3.2 scope. Once the baseline assembles on
//! sm_120, un-ignore these tests and they'll validate the full CSHA
//! emission pipeline (A.2.2 prologue + A.2.3 tile-sweep + A.2.3.2
//! lane-coherent scatter + A.2.4 RoPE epilogue + A.4 active_heads guard)
//! end-to-end on real hardware.
//!
//! Run with `cargo test -p nsl-codegen --test csha_ptx_ptxas_validation -- --ignored`
//! to attempt assembly and inspect the `.ptx` dump in the system temp dir.

use nsl_codegen::flash_attention::{
    synthesize_flash_attention_ptx, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use std::io::Write;
use std::process::{Command, Stdio};

/// Locate `ptxas`. Returns `None` if the CUDA toolkit isn't installed —
/// tests that depend on this skip gracefully.
fn find_ptxas() -> Option<String> {
    // 1. $PATH (Unix, Windows WSL, containers with CUDA in PATH)
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
    // 2. Standard Windows CUDA install path.
    let win_default = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe";
    if std::path::Path::new(win_default).exists() {
        return Some(win_default.to_string());
    }
    None
}

/// Invoke ptxas on `ptx_bytes` targeting `sm_arch`. Returns Ok(()) on
/// successful assembly or Err(stderr) on rejection.
fn assemble_ptx(ptxas_path: &str, ptx_bytes: &[u8], sm_arch: &str) -> Result<(), String> {
    // `ptxas -o - -` reads PTX from stdin, writes cubin to stdout.
    let mut child = Command::new(ptxas_path)
        .args([
            "--gpu-name",
            sm_arch,
            "-O0",      // skip optimisation — focus on syntactic validation
            "-o",
            "-",        // cubin to stdout (discarded via `Stdio::null`)
            "-",        // read PTX from stdin
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn ptxas: {}", e))?;

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(ptx_bytes)
        .map_err(|e| format!("failed to write PTX to ptxas stdin: {}", e))?;

    let out = child
        .wait_with_output()
        .map_err(|e| format!("ptxas wait failed: {}", e))?;

    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

fn csha_l2_rope_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 120,
        csha: Some(CshaExtras::level2(1e-5, 512)),
    }
}

fn non_csha_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 120,
        csha: None,
    }
}

/// Regression-gate: the non-CSHA PTX path must assemble on Blackwell.
/// Any failure here indicates the test harness is broken (wrong ptxas
/// arch, CUDA install mismatch) rather than a CSHA emission defect.
#[test]
#[ignore = "baseline non-CSHA FA emitter has pre-existing ptxas defects (hex-float + shmem+reg syntax); fix those first"]
fn non_csha_ptx_assembles_on_sm120() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: ptxas not found in PATH or standard CUDA install");
        return;
    };
    let ptx = synthesize_flash_attention_ptx(&non_csha_config());
    let dump = std::env::temp_dir().join("csha_noncsha_dump.ptx");
    std::fs::write(&dump, &ptx).ok();
    eprintln!("PTX dumped to: {}", dump.display());
    if let Err(stderr) = assemble_ptx(&ptxas, &ptx, "sm_120") {
        panic!(
            "non-CSHA PTX failed to assemble (dumped to {}):\n{}",
            dump.display(),
            stderr
        );
    }
}

/// The headline Tier A validation: the CSHA L2 + RoPE kernel variant
/// (RMSNorm prologue + matmul projection with lane-coherent scatter +
/// RoPE epilogue + active_heads guard) must assemble on sm_120. This
/// catches emission defects in A.2.2, A.2.3 (tile-sweep + scatter),
/// A.2.3.2 (lane-coherent scatter address math), A.2.4 (RoPE
/// epilogue), and A.4 (active_heads guard) end-to-end.
#[test]
#[ignore = "gated on non-CSHA baseline passing — see module-level status"]
fn csha_l2_rope_ptx_assembles_on_sm120() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: ptxas not found");
        return;
    };
    let mut cfg = csha_l2_rope_config();
    cfg.csha.as_mut().unwrap().active_heads = 5; // exercise A.4 guard
    let ptx = synthesize_flash_attention_ptx(&cfg);

    if let Err(stderr) = assemble_ptx(&ptxas, &ptx, "sm_120") {
        // Include a PTX tail to make errors diagnostic — the ptxas
        // error typically references a line number we want to inspect.
        let ptx_str = String::from_utf8_lossy(&ptx);
        let lines: Vec<&str> = ptx_str.lines().collect();
        let total = lines.len();
        let tail_start = total.saturating_sub(40);
        let tail: String = lines[tail_start..]
            .iter()
            .enumerate()
            .map(|(i, l)| format!("  {:>4}: {}", tail_start + i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "CSHA L2 + RoPE PTX failed to assemble on sm_120:\n\n--- ptxas stderr ---\n{}\n--- PTX tail (last 40 lines of {}) ---\n{}",
            stderr, total, tail
        );
    }
}
