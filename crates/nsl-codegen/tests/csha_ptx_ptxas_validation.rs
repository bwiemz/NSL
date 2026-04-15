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
    // Small tile sizes keep SMEM under the default 48 KB static cap.
    // base (32+32)*32*2 = 4 KB + prologue 32*32*2 = 2 KB + projection
    // 3*32*32*2 = 6 KB + output 32*32*2 = 2 KB  ⇒  14 KB total — well
    // within the sm_120 default budget. Production configs typically
    // opt into the extended dynamic-shared regime via cudaFuncSetAttr.
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 120,
        csha: Some(CshaExtras::level2(1e-5, 32)),
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

/// v2 scalar emitter: assemble a representative corner of the supported-config
/// matrix through ptxas --gpu-name sm_75. Catches emission-level defects
/// (undeclared registers, malformed operands, invalid labels, SMEM overflow)
/// that snapshot tests cannot see. Skipped gracefully when ptxas isn't in PATH.
#[test]
#[ignore = "requires ptxas in PATH"]
fn v2_kernel_assembles_on_sm75_full_matrix() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: ptxas not found"); return;
    };
    use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;
    use nsl_codegen::flash_attention_v2::smem_layout::validate_scalar_v2_config;

    let base = FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: false, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 75, csha: None,
    };

    // Subset of the supported matrix that exercises the corners.
    let matrix: &[(i64, i64, i64)] = &[
        (4,   32,  32),
        (32,  32,  32),
        (64,  64,  128),
        (16,  16,  64),
        (128, 128, 128),
    ];

    let mut failures = Vec::new();
    for &(bq, bkv, hd) in matrix {
        let c = FlashAttentionConfig { block_q: bq, block_kv: bkv, head_dim: hd, ..base.clone() };
        if validate_scalar_v2_config(&c).is_err() { continue; }
        let ptx = synthesize_flash_attention_ptx_v2(&c);
        // Drop trailing NUL for file write; ptxas wants text input.
        let text_end = ptx.iter().position(|&b| b == 0).unwrap_or(ptx.len());
        let dump = std::env::temp_dir()
            .join(format!("v2_{}x{}x{}.ptx", bq, bkv, hd));
        std::fs::write(&dump, &ptx[..text_end]).ok();
        if let Err(stderr) = assemble_ptx(&ptxas, &ptx[..text_end], "sm_75") {
            failures.push(format!(
                "v2 PTX {}x{}x{} failed to assemble (dump: {}):\n{}",
                bq, bkv, hd, dump.display(), stderr
            ));
        }
    }
    assert!(failures.is_empty(),
        "v2 ptxas assembly failures:\n\n{}",
        failures.join("\n---\n"));
}

/// The headline Tier A validation: the CSHA L2 + RoPE kernel variant
/// (RMSNorm prologue + matmul projection with lane-coherent scatter +
/// RoPE epilogue + active_heads guard) must assemble on sm_120. This
/// catches emission defects in A.2.2, A.2.3 (tile-sweep + scatter),
/// A.2.3.2 (lane-coherent scatter address math), A.2.4 (RoPE
/// epilogue), and A.4 (active_heads guard) end-to-end.
#[test]
fn csha_l2_rope_ptx_assembles_on_sm120() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: ptxas not found");
        return;
    };
    let mut cfg = csha_l2_rope_config();
    cfg.csha.as_mut().unwrap().active_heads = 5; // exercise A.4 guard
    let ptx = synthesize_flash_attention_ptx(&cfg);
    let dump = std::env::temp_dir().join("csha_l2rope_dump.ptx");
    std::fs::write(&dump, &ptx).ok();
    eprintln!("CSHA L2 PTX dumped to: {}", dump.display());

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

/// A1: The `.target` line in synthesised PTX must reflect the compile
/// target's SM, not the old hardcoded `80`.
///
/// `emit_ptx_header` (v1) maps SM versions to three discrete PTX targets:
///   * `gpu_sm < 80`  → `.target sm_52`  (scalar path)
///   * `gpu_sm >= 80` → `.target sm_80`  (MMA path)
///   * `gpu_sm >= 90` → `.target sm_90`  (Hopper / Blackwell)
///
/// Before the A1 fix, all three kernel-build sites in `compiler/kernel.rs`
/// hard-coded `gpu_sm: 80`, so even a `sm_75` compile target would silently
/// emit MMA PTX (`.target sm_80`).  After the fix the target line must match
/// the table above.
///
/// This test exercises the actual compiler pipeline:
/// `compiler::flash_sm_for_compile_target` constructs a `Compiler` with the
/// given `compile_options.target`, runs `compile_flash_attention_kernels`
/// against a minimal `@flash_attention` stub, and returns the `gpu_sm` stored
/// in the resulting `FlashAttentionCompileContext`.  Reverting any of the
/// three fixed call-sites in `kernel.rs` back to `gpu_sm: 80` would cause
/// this test to fail for every SM other than `sm_80`.
#[test]
fn a1_gpu_sm_matches_compile_target() {
    // (compile target string, expected gpu_sm value)
    let cases: &[(&str, u32)] = &[
        ("sm_75",  75),
        ("sm_80",  80),
        ("sm_90",  90),
        ("sm_120", 120),
    ];

    for &(sm, expected_gpu_sm) in cases {
        let gpu_sm = nsl_codegen::test_helpers::flash_sm_for_compile_target(sm);
        assert_eq!(
            gpu_sm,
            expected_gpu_sm,
            "a1 [{sm}]: compile_options.target should flow into gpu_sm={expected_gpu_sm}, \
             got gpu_sm={gpu_sm} — check the three parse_gpu_sm_from_target call-sites \
             in compiler/kernel.rs (~line 638, ~690, ~756)"
        );
    }
}
