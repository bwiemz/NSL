//! MFU campaign C3 gates — generic elementwise-chain fusion over the
//! adjoint tape (always-on, `NSL_FUSE_ELEMENTWISE_BWD=0` kill-switch).
//!
//! The fusion is bit-exact by construction on BOTH paths: the fused kernel
//! replays the exact tape-order `.rn` f32 ops, and the runtime falls back to
//! a decomposed replay through the original public FFIs whenever the
//! uniform-shape/device/dtype gate refuses. So the gates demand BIT-IDENTICAL
//! loss streams with fusion on vs off, plus anti-vacuity markers.
//!
//! Fixtures: `ew_chain_fusion.nsl` (decomposed-rmsnorm residual FFN — dense
//! same-shape chain population) and `ew_chain_fusion_bcast.nsl` (a shared
//! `[64]` gain whose gradient chain mixes `[B,S,64]` and `[64]` shapes —
//! forces the decomposed-replay path).

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOut {
    success: bool,
    stdout: String,
    stderr: String,
    losses: Vec<String>,
}

fn run_fixture(fixture: &str, tag: &str, gpu: bool, extra_args: &[&str], fusion_off: bool) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_ewfuse_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures").join(fixture),
    )
    .unwrap();
    if gpu {
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--deterministic", "--seed", "777"]);
    cmd.args(extra_args);
    if fusion_off {
        cmd.env("NSL_FUSE_ELEMENTWISE_BWD", "0");
    }
    let out = cmd
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let losses = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| {
            v.lines()
                .filter_map(|l| {
                    let l = l.trim();
                    if let Some(inner) =
                        l.strip_prefix("tensor([").and_then(|r| r.strip_suffix("])"))
                    {
                        Some(inner.to_string())
                    } else if l.parse::<f64>().is_ok() {
                        Some(l.to_string())
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    RunOut {
        success: out.status.success(),
        stdout,
        stderr,
        losses,
    }
}

const CHAIN_MARKER: &str = "[fuse] elementwise backward chains:";

/// CPU: fusion on == fusion off, bit-identical, and chains demonstrably fused.
#[test]
fn chain_fusion_bit_exact_cpu() {
    let on = run_fixture("ew_chain_fusion.nsl", "cpu_on", false, &[], false);
    assert!(on.success, "fusion-on run failed:\n{}", on.stderr);
    assert!(
        on.stderr.contains(CHAIN_MARKER),
        "chain fuser never fired (vacuous — fixture or matcher changed?):\n{}",
        on.stderr
    );
    let off = run_fixture("ew_chain_fusion.nsl", "cpu_off", false, &[], true);
    assert!(off.success, "fusion-off run failed:\n{}", off.stderr);
    assert!(
        !off.stderr.contains(CHAIN_MARKER),
        "kill-switch did not disable the chain fuser:\n{}",
        off.stderr
    );
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "chain fusion changed the loss stream — must be bit-exact.\non stderr:\n{}\noff stderr:\n{}",
        on.stderr, off.stderr
    );
}

/// CPU, broadcast variant: a chain mixing `[B,S,64]` and `[64]` shapes must
/// take the decomposed-replay path and still be bit-identical.
#[test]
fn chain_fusion_replay_fallback_bit_exact_cpu() {
    let on = run_fixture("ew_chain_fusion_bcast.nsl", "bcast_on", false, &[], false);
    assert!(on.success, "fusion-on run failed:\n{}", on.stderr);
    assert!(
        on.stderr.contains(CHAIN_MARKER),
        "chain fuser never fired on the broadcast fixture:\n{}",
        on.stderr
    );
    let off = run_fixture("ew_chain_fusion_bcast.nsl", "bcast_off", false, &[], true);
    assert!(off.success, "fusion-off run failed:\n{}", off.stderr);
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "replay fallback changed the loss stream — must be bit-exact"
    );
}

/// The v1 fuser must observably defer under --layerwise-accum.
#[test]
fn chain_fusion_skipped_under_csla() {
    let out = run_fixture(
        "ew_chain_fusion.nsl",
        "csla_skip",
        false,
        &["--checkpoint-blocks", "--layerwise-accum"],
        false,
    );
    assert!(out.success, "CSLA run failed:\n{}", out.stderr);
    assert!(
        out.stderr
            .contains("[fuse] elementwise backward fusion skipped (--layerwise-accum)"),
        "the CSLA deferral must be observable, not silent:\n{}",
        out.stderr
    );
    assert!(
        !out.stderr.contains(CHAIN_MARKER),
        "fuser ran under --layerwise-accum despite the v1 skip:\n{}",
        out.stderr
    );
}

/// GPU: prod-like posture (--fuse-rmsnorm-backward), fusion on vs off under
/// one binary, bit-identical, plus run-to-run determinism of the fused path.
#[test]
#[ignore = "requires CUDA GPU"]
fn chain_fusion_bit_exact_gpu() {
    let args = &["--checkpoint-blocks", "--fuse-rmsnorm-backward"][..];
    let on = run_fixture("ew_chain_fusion.nsl", "gpu_on", true, args, false);
    assert!(on.success, "fusion-on run failed:\n{}", on.stderr);
    assert!(
        on.stderr.contains(CHAIN_MARKER),
        "chain fuser never fired:\n{}",
        on.stderr
    );
    let off = run_fixture("ew_chain_fusion.nsl", "gpu_off", true, args, true);
    assert!(off.success, "fusion-off run failed:\n{}", off.stderr);
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "chain fusion changed the GPU loss stream — must be bit-exact"
    );
    let on2 = run_fixture("ew_chain_fusion.nsl", "gpu_on2", true, args, false);
    assert!(on2.success, "rerun failed:\n{}", on2.stderr);
    assert_eq!(on.losses, on2.losses, "fused path not deterministic");
}

/// GPU, broadcast variant: the replay path composes with device tensors and
/// stays bit-identical.
#[test]
#[ignore = "requires CUDA GPU"]
fn chain_fusion_replay_fallback_bit_exact_gpu() {
    let on = run_fixture("ew_chain_fusion_bcast.nsl", "gpu_bcast_on", true, &[], false);
    assert!(on.success, "fusion-on run failed:\n{}", on.stderr);
    let off = run_fixture("ew_chain_fusion_bcast.nsl", "gpu_bcast_off", true, &[], true);
    assert!(off.success, "fusion-off run failed:\n{}", off.stderr);
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "GPU replay fallback changed the loss stream — must be bit-exact"
    );
}
