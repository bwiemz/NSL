//! P4 item 14: CUDA-aware ZeRO collectives.
//!
//! The `--collectives` axis: `sim` (CPU-shm reference), `sim-gpu` (CUDA-aware
//! TEST backend — device-pointer API staged through the shm reduce; validates
//! the GPU plumbing on ONE GPU), `nccl` (real NCCL transport; multiple ranks
//! per device are refused by NCCL itself, so the ws>=2 NCCL leg is a
//! scheduled HARDWARE gate: on a single-GPU box it asserts the documented
//! symmetric refusal, on a multi-GPU box it asserts parity).
//!
//! GPU: `cargo test -p nsl-cli --features cuda --test zero_gpu_collectives_gate -- --ignored`
//! NCCL: same with `--features cuda,nccl` (plus NSL_NCCL_LIB_DIR/LD_LIBRARY_PATH).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOut {
    success: bool,
    loss: String,
    stderr: String,
}

fn run_gpu(tag: &str, extra: &[&str]) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_zgpu_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("out.nslm");
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .unwrap()
    .replace("# GPU_PLACEMENT", "m.to(cuda)")
    .replace("CSLA_SAVE_PATH", &save.display().to_string());
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--deterministic", "--zero-stage", "1"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_ZERO_COUNTER", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let loss = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default();
    RunOut {
        success: out.status.success(),
        loss,
        stderr,
    }
}

/// Unknown backend value fails fast at the CLI, before any spawn.
#[test]
fn unknown_collectives_value_refuses() {
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--collectives", "bogus", "x.nsl"])
        .output()
        .expect("spawn");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--collectives must be"),
        "wrong refusal:\n{stderr}"
    );
}

/// The CUDA-aware TEST backend: a 2-rank GPU-RESIDENT ZeRO-1 run must be
/// BIT-EXACT against the single-rank GPU baseline (replicated data:
/// (g+g)/2 == g in f32), with real device-pointer collectives counted and
/// the byte-balanced moment shards summing to the unsharded total.
#[test]
#[ignore = "requires CUDA GPU"]
fn sim_gpu_two_rank_parity_bit_exact() {
    let base = run_gpu("base", &[]);
    assert!(base.success, "GPU baseline failed:\n{}", base.stderr);
    assert!(!base.loss.is_empty(), "empty baseline loss stream");

    let spmd = run_gpu("simgpu", &["--devices", "2", "--collectives", "sim-gpu"]);
    assert!(spmd.success, "2-rank sim-gpu run failed:\n{}", spmd.stderr);
    assert_eq!(
        spmd.loss, base.loss,
        "2-rank GPU ZeRO-1 diverged from the single-rank GPU baseline\nstderr:\n{}",
        spmd.stderr
    );

    // Anti-vacuity: device-pointer collectives actually ran on both ranks
    // (8 params x 6 steps = 48 each), and the moment shards are real.
    for rank in 0..2 {
        let line = spmd
            .stderr
            .lines()
            .find(|l| l.contains(&format!("[zero] ws=2 rank={rank}")))
            .unwrap_or_else(|| panic!("no [zero] line for rank {rank}:\n{}", spmd.stderr));
        assert!(
            line.contains("all_reduce=48") && line.contains("broadcast=48"),
            "wrong collective counts: {line}"
        );
    }
    let elems = |s: &str, pat: &str| -> u64 {
        s.lines()
            .find(|l| l.contains(pat))
            .and_then(|l| l.split("optim_elems=").nth(1))
            .and_then(|r| r.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    let full = elems(&base.stderr, "[zero] ws=1 rank=0");
    let r0 = elems(&spmd.stderr, "[zero] ws=2 rank=0");
    let r1 = elems(&spmd.stderr, "[zero] ws=2 rank=1");
    assert!(full > 0 && r0 > 0 && r1 > 0, "missing optim_elems");
    assert_eq!(r0 + r1, full, "byte-balanced shards must cover the total");
    assert!(r0 < full && r1 < full, "both ranks must actually shard");
}

/// The real NCCL transport, ws=2. On a multi-GPU box: full parity. On a
/// single-GPU box NCCL refuses multiple ranks per device — the run must
/// fail FAST with the documented symmetric ncclInvalidUsage refusal on
/// every rank (never a hang, never a silent success).
#[test]
#[ignore = "requires CUDA GPU + nccl-featured build"]
fn nccl_two_rank_parity_or_documented_refusal() {
    let base = run_gpu("nbase", &[]);
    assert!(base.success, "GPU baseline failed:\n{}", base.stderr);

    let spmd = run_gpu("nccl2", &["--devices", "2", "--collectives", "nccl"]);
    if spmd.success {
        // Multi-GPU hardware: demand full parity + counters.
        assert_eq!(
            spmd.loss, base.loss,
            "2-rank NCCL ZeRO-1 diverged from the single-rank GPU baseline"
        );
        assert!(
            spmd.stderr.contains("NCCL communicator up"),
            "NCCL banner missing:\n{}",
            spmd.stderr
        );
        for rank in 0..2 {
            let line = spmd
                .stderr
                .lines()
                .find(|l| l.contains(&format!("[zero] ws=2 rank={rank}")))
                .unwrap_or_else(|| panic!("no [zero] line for rank {rank}"));
            assert!(
                line.contains("all_reduce=48") && line.contains("broadcast=48"),
                "wrong collective counts: {line}"
            );
        }
    } else {
        // Single-GPU hardware: the documented loud, symmetric refusal.
        assert!(
            spmd.stderr.contains("ncclInvalidUsage")
                || spmd.stderr.contains("NCCL backend init failed"),
            "expected the documented NCCL same-device refusal, got:\n{}",
            spmd.stderr
        );
        assert!(
            spmd.stderr.contains("rank 0") && spmd.stderr.contains("rank 1"),
            "refusal must be symmetric across ranks:\n{}",
            spmd.stderr
        );
    }
}
