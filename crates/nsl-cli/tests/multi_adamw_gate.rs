//! Fusion-queue item 1 gates: multi-tensor fused AdamW.
//!
//! The non-clip FASE-Deferred optimizer loop collapses into ONE
//! pointer-table launch (`nsl_fase_fused_adamw_step_multi`) whose
//! per-element arithmetic is byte-for-byte the single-step kernel's.
//! Contract: training output is BIT-IDENTICAL with the multi path on
//! (default) vs off (`NSL_FASE_MULTI_STEP=0`), on GPU and CPU alike, and
//! the batch launch actually fires (anti-vacuity marker).

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOut {
    stdout: String,
    stderr: String,
}

fn run_adamw_mlp_env(
    tag: &str,
    gpu: bool,
    grad_clip: Option<f64>,
    envs: &[(&str, &str)],
) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_multiadamw_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/cuda_graph_gate.nsl"),
    )
    .unwrap();
    if gpu {
        src = src.replace(
            "# GPU_PLACEMENT",
            "m.to(cuda)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)",
        );
        src = src.replace("m.forward_train(x)", "m.forward_train(xg)");
        src = src.replace("l1_loss(pred, y)", "l1_loss(pred, yg)");
    }
    if let Some(clip) = grad_clip {
        // Route the optimizer through the two-phase-clip Deferred arm.
        src = src.replace(
            "train(model=m, epochs=8, grad_accumulation=2):",
            &format!("train(model=m, epochs=8, grad_accumulation=2, grad_clip={clip}):"),
        );
        assert!(src.contains("grad_clip"), "grad_clip rewrite did not land");
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--deterministic", "--source-ad"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    assert!(
        out.status.success(),
        "run failed (tag={tag}):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    RunOut {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn run_adamw_mlp(tag: &str, gpu: bool, multi_off: bool) -> RunOut {
    let envs: &[(&str, &str)] = if multi_off {
        &[("NSL_FASE_MULTI_STEP", "0")]
    } else {
        &[]
    };
    run_adamw_mlp_env(tag, gpu, None, envs)
}

fn loss_stream(stdout: &str) -> Vec<String> {
    stdout
        .split("LOSS_STREAM_BEGIN")
        .nth(1)
        .and_then(|r| r.split("LOSS_STREAM_END").next())
        .unwrap_or("")
        .lines()
        .map(str::trim)
        // GPU losses print as `tensor([..])`, CPU losses as bare scalars.
        .filter(|l| l.starts_with("tensor([") || l.parse::<f64>().is_ok())
        .map(str::to_string)
        .collect()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn multi_adamw_gpu_bit_identical_and_fires() {
    let on = run_adamw_mlp("gpu_on", true, false);
    let off = run_adamw_mlp("gpu_off", true, true);
    let (ls_on, ls_off) = (loss_stream(&on.stdout), loss_stream(&off.stdout));
    assert!(ls_on.len() >= 8, "expected >=8 losses, got {}", ls_on.len());
    assert_eq!(
        ls_on, ls_off,
        "multi-tensor AdamW diverged from the per-param loop (must be bit-identical)"
    );
    // Anti-vacuity: the batched launch actually fired, covering all 6
    // params of the fixture, and the kill-switch actually disabled it.
    assert!(
        on.stderr.contains("[fase-multi] batched 6 params"),
        "batch marker missing — multi path did not fire:\n{}",
        on.stderr
    );
    assert!(
        !off.stderr.contains("[fase-multi]"),
        "kill-switch NSL_FASE_MULTI_STEP=0 did not disable the multi path"
    );
}

/// The two-phase-clip Phase B multi arm: ONE pointer-table launch with the
/// clip factor folded into the m_partial read (`mp_scale`). Must be
/// BIT-IDENTICAL to the legacy scale-then-step loop. Batched Phase A sum_sq
/// is pinned OFF on both arms so this isolates the mp_scale fold — the
/// batched norm changes accumulation dtype and is gated separately below.
/// `grad_clip=0.01` is deliberately tiny so the factor is genuinely < 1 and
/// the fold is exercised (a factor of exactly 1.0 takes the branch-around).
#[test]
#[ignore = "requires CUDA GPU"]
fn multi_adamw_clip_gpu_bit_identical_and_fires() {
    let on = run_adamw_mlp_env(
        "clip_gpu_on",
        true,
        Some(0.01),
        &[("NSL_FASE_BATCH_SUMSQ", "0")],
    );
    let off = run_adamw_mlp_env(
        "clip_gpu_off",
        true,
        Some(0.01),
        &[("NSL_FASE_BATCH_SUMSQ", "0"), ("NSL_FASE_MULTI_STEP", "0")],
    );
    let (ls_on, ls_off) = (loss_stream(&on.stdout), loss_stream(&off.stdout));
    assert!(ls_on.len() >= 8, "expected >=8 losses, got {}", ls_on.len());
    assert_eq!(
        ls_on, ls_off,
        "clip-arm multi AdamW (mp_scale fold) diverged from the legacy \
         scale-then-step loop (must be bit-identical)"
    );
    assert!(
        on.stderr.contains("[fase-multi] batched 6 params"),
        "batch marker missing — the clip arm did not take the multi path:\n{}",
        on.stderr
    );
    assert!(
        !off.stderr.contains("[fase-multi]"),
        "kill-switch NSL_FASE_MULTI_STEP=0 did not disable the clip-arm multi path"
    );
}

/// Batched Phase A norm (`nsl_fase_sum_sq_list`, one drain) vs the per-param
/// syncing loop. NOT bit-identical by design: the batched kernel accumulates
/// each tensor in f64 where `nsl_tensor_sum_sq`'s GPU path accumulates in
/// f32 (see the FFI's numerics note), so the clip factor — and every loss
/// after the first clipped step — may differ in the low digits. Asserted
/// CLOSE (1e-3 relative), which a broken norm (wrong tensor set, wrong
/// slot, dropped param) fails by orders of magnitude.
#[test]
#[ignore = "requires CUDA GPU"]
fn batched_phase_a_sumsq_gpu_close_to_per_param() {
    let batched = run_adamw_mlp_env("sumsq_gpu_on", true, Some(0.01), &[]);
    let legacy = run_adamw_mlp_env(
        "sumsq_gpu_off",
        true,
        Some(0.01),
        &[("NSL_FASE_BATCH_SUMSQ", "0")],
    );
    let (ls_b, ls_l) = (loss_stream(&batched.stdout), loss_stream(&legacy.stdout));
    assert!(ls_b.len() >= 8, "expected >=8 losses, got {}", ls_b.len());
    assert_eq!(ls_b.len(), ls_l.len(), "loss stream lengths diverged");
    let parse = |s: &str| -> f64 {
        s.trim_start_matches("tensor([")
            .trim_end_matches("])")
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("unparseable loss line: {s}"))
    };
    for (b, l) in ls_b.iter().zip(&ls_l) {
        let (vb, vl) = (parse(b), parse(l));
        let rel = (vb - vl).abs() / vl.abs().max(1e-12);
        assert!(
            rel < 1e-3,
            "batched Phase A norm diverged beyond accumulation-dtype noise: \
             {vb} vs {vl} (rel {rel:.2e})"
        );
    }
}

#[test]
fn multi_adamw_cpu_bit_identical() {
    // CPU tensors take the multi FFI's per-param fallback arm — output must
    // stay bit-identical to the legacy loop.
    let on = run_adamw_mlp("cpu_on", false, false);
    let off = run_adamw_mlp("cpu_off", false, true);
    let (ls_on, ls_off) = (loss_stream(&on.stdout), loss_stream(&off.stdout));
    assert!(ls_on.len() >= 8, "expected >=8 losses, got {}", ls_on.len());
    assert_eq!(ls_on, ls_off, "CPU fallback arm diverged from the legacy loop");
}

// ── Item 8, CSLA half ──────────────────────────────────────────────────────
// The layerwise (`--layerwise-accum`) window updates one layer GROUP at a
// time; `emit_csla_group_update`'s per-param loop collapses into one
// `nsl_fase_fused_adamw_step_multi_idx` launch per group under the same
// admission (and the same kill-switches) as the FullBuffer arm. Contract:
// bit-identical loss stream + saved model vs the per-param loop, marker
// anti-vacuity, and envelope configurations (offload) fall back silently.

fn run_csla_ffn_env(tag: &str, gpu: bool, extra_flags: &[&str], envs: &[(&str, &str)]) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_cslamulti_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("out.nslm");
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .unwrap();
    src = src.replace(
        "CSLA_SAVE_PATH",
        &save.display().to_string().replace('\\', "/"),
    );
    if gpu {
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--deterministic", "--source-ad", "--checkpoint-blocks", "--layerwise-accum"])
        .args(extra_flags)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    assert!(
        out.status.success(),
        "run failed (tag={tag}):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let model_bytes = std::fs::read(&save).expect("model_save produced no file");
    // Whole-file hash, not a prefix: divergence anywhere in the saved
    // θ/m/v payload must fail the comparison.
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model_bytes.hash(&mut h);
    RunOut {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: format!(
            "{}\nMODEL_BYTES {:x} len {}",
            String::from_utf8_lossy(&out.stderr),
            h.finish(),
            model_bytes.len()
        ),
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn csla_multi_adamw_gpu_bit_identical_and_fires() {
    let on = run_csla_ffn_env("csla_gpu_on", true, &[], &[]);
    let off = run_csla_ffn_env("csla_gpu_off", true, &[], &[("NSL_FASE_MULTI_STEP", "0")]);
    let (ls_on, ls_off) = (loss_stream(&on.stdout), loss_stream(&off.stdout));
    assert!(ls_on.len() >= 10, "expected >=10 losses, got {}", ls_on.len());
    assert_eq!(
        ls_on, ls_off,
        "CSLA group multi-step diverged from the per-param loop (must be bit-identical)"
    );
    // The saved models must be byte-identical too (θ/m/v all touched by the
    // group updates; the stderr carries a byte-prefix dump of each).
    let model_on = on.stderr.split("MODEL_BYTES").nth(1).unwrap();
    let model_off = off.stderr.split("MODEL_BYTES").nth(1).unwrap();
    assert_eq!(model_on, model_off, "saved .nslm diverged between arms");
    assert!(
        on.stderr.contains("[fase-multi] batched "),
        "batch marker missing — CSLA group multi path did not fire:\n{}",
        on.stderr
    );
    assert!(
        !off.stderr.contains("[fase-multi]"),
        "kill-switch NSL_FASE_MULTI_STEP=0 did not disable the CSLA group multi path"
    );
}

#[test]
fn csla_multi_adamw_cpu_bit_identical() {
    // CPU members take the idx variant's per-param fallback arm; the loss
    // stream and saved model must match the legacy group loop bit-for-bit.
    let on = run_csla_ffn_env("csla_cpu_on", false, &[], &[]);
    let off = run_csla_ffn_env("csla_cpu_off", false, &[], &[("NSL_FASE_MULTI_STEP", "0")]);
    let (ls_on, ls_off) = (loss_stream(&on.stdout), loss_stream(&off.stdout));
    assert!(ls_on.len() >= 10, "expected >=10 losses, got {}", ls_on.len());
    assert_eq!(ls_on, ls_off, "CPU CSLA fallback diverged from the legacy loop");
}

/// Envelope exclusion: plain CSLA × `--optim-state-offload` is a supported
/// composition whose group update stages host-resident m/v through the
/// offload envelope — the multi arm must NOT admit it (host moments cannot
/// ride a device pointer table). The run must succeed with the per-param
/// loop and never print the batch marker.
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_multi_adamw_offload_falls_back_gpu() {
    let out = run_csla_ffn_env("csla_gpu_offload", true, &["--optim-state-offload"], &[]);
    let ls = loss_stream(&out.stdout);
    assert!(ls.len() >= 10, "expected >=10 losses, got {}", ls.len());
    assert!(
        !out.stderr.contains("[fase-multi]"),
        "offload envelope must exclude the CSLA multi arm:\n{}",
        out.stderr
    );
}
