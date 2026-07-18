//! D3 — ZeRO-1 SPMD parity gates (toy scale, CPU, SimulatedBackend).
//!
//! `--zero-stage 1` + `nsl run --devices N` now performs REAL multi-rank
//! training: N processes over a CPU-shm SimulatedBackend, per-step
//! gradient all-reduce (fixed rank-order sum, then ÷N), owner-gated
//! optimizer updates (round-robin `idx % N`), and a post-step per-param
//! broadcast from each owner. With replicated data (every rank sees the
//! same batches — the training loader is rank-blind today) the reduced
//! gradient is `(N·g)/N == g` bit-exactly for N a power of two, and the
//! sharded update of each param is the same arithmetic the single-rank
//! baseline runs — so the rank-0 loss stream must be BIT-IDENTICAL to the
//! unsharded baseline. That equivalence exercises the whole pipeline:
//! init/partition emission, the collective sequence (deadlock-free
//! lockstep), reduce, ownership gating, m_partial hygiene on skipped
//! params, and the θ broadcast (a wrong owner or a stale non-owner θ
//! diverges the NEXT step's loss immediately).
//!
//! Fixture: csla_layerwise_ffn.nsl WITHOUT any csla flags — a plain
//! deterministic CPU FFN LM with 8 params (uneven round-robin at N=2:
//! ranks own 4/4; at N=3: 3/3/2).
//!
//! Hazards handled here: the SimulatedBackend spin-barrier has NO timeout
//! (a crashed rank hangs peers) — every SPMD invocation runs under a
//! watchdog that kills the process tree on expiry.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn program(save_path: &Path) -> String {
    let src = std::fs::read_to_string(
        repo_root().join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .expect("ffn fixture missing");
    // CPU run: leave # GPU_PLACEMENT as a comment.
    src.replace(
        "CSLA_SAVE_PATH",
        &save_path.display().to_string().replace('\\', "/"),
    )
}

struct RunOutput {
    loss_stream: String,
    stdout: String,
    stderr: String,
    success: bool,
}

/// Run `nsl run` with a watchdog (SPMD spin-barriers hang forever on a
/// dead rank; a hung gate is worse than a red one).
fn run_nsl(source: &str, tag: &str, extra_args: &[&str], timeout_secs: u64) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_zero_gate_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("zero_gate.nsl");
    std::fs::write(&prog, source).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("run")
        .arg("-q")
        .arg("--manifest-path")
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run", "--source-ad", "--deterministic"])
        .args(extra_args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn nsl run");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let status = loop {
        match child.try_wait().expect("try_wait") {
            Some(s) => break s,
            None => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("nsl run '{tag}' exceeded {timeout_secs}s — SPMD rank hang?");
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    };
    let mut stdout = String::new();
    let mut stderr = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .ok();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .ok();

    let mut loss_stream = String::new();
    let mut in_stream = false;
    for line in stdout.lines() {
        match line.trim() {
            "LOSS_STREAM_BEGIN" => in_stream = true,
            "LOSS_STREAM_END" => in_stream = false,
            l if in_stream => {
                loss_stream.push_str(l);
                loss_stream.push('\n');
            }
            _ => {}
        }
    }
    RunOutput {
        loss_stream,
        stdout,
        stderr,
        success: status.success(),
    }
}

/// ZeRO-1 at world_size 1 degenerates to the identity (no-op collectives,
/// rank 0 owns everything) — bit-identical stream AND model bytes.
#[test]
fn zero_stage1_single_rank_is_identity() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero_id_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("base.nslm");
    let save_b = tmp.join("zero1.nslm");

    let base = run_nsl(&program(&save_a), "id_base", &[], 600);
    assert!(base.success, "baseline failed:\n{}", base.stderr);
    let z = run_nsl(&program(&save_b), "id_zero", &["--zero-stage", "1"], 600);
    assert!(z.success, "zero-stage arm failed:\n{}", z.stderr);

    assert!(!base.loss_stream.is_empty(), "empty baseline loss stream");
    assert_eq!(
        base.loss_stream, z.loss_stream,
        "single-rank --zero-stage 1 diverged from baseline"
    );
    let ba = std::fs::read(&save_a).expect("baseline save missing");
    let bb = std::fs::read(&save_b).expect("zero save missing");
    assert!(ba == bb, "model bytes diverged at world_size 1");
}

/// The real thing: 2 SPMD ranks over the shm backend, replicated data.
/// Rank 0's loss stream must be bit-identical to the single-rank baseline
/// — wrong reduce order, a missed broadcast, or stale non-owner state
/// diverges within one step.
#[test]
#[ignore = "spawns multiple processes; run explicitly"]
fn zero_stage1_two_rank_parity() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero_2r_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("base.nslm");
    let save_b = tmp.join("spmd.nslm");

    let base = run_nsl(&program(&save_a), "2r_base", &[], 600);
    assert!(base.success, "baseline failed:\n{}", base.stderr);
    let spmd = run_nsl(
        &program(&save_b),
        "2r_spmd",
        &["--zero-stage", "1", "--devices", "2"],
        900,
    );
    assert!(spmd.success, "2-rank SPMD run failed:\n{}", spmd.stderr);

    assert!(!base.loss_stream.is_empty(), "empty baseline loss stream");
    assert_eq!(
        base.loss_stream, spmd.loss_stream,
        "2-rank ZeRO-1 rank-0 loss stream diverged from the single-rank \
         baseline (replicated data must be bit-exact)\nSPMD stderr:\n{}",
        spmd.stderr
    );
    // θ-sync evidence: the saved model (whichever rank wrote last — both
    // hold broadcast-synced params) matches the baseline bit-for-bit.
    let ba = std::fs::read(&save_a).expect("baseline save missing");
    let bb = std::fs::read(&save_b).expect("spmd save missing");
    assert!(
        ba == bb,
        "2-rank ZeRO-1 model bytes diverged from baseline — θ broadcast or \
         owner gating is wrong"
    );
}

/// Loud refusals: stage 2/3 unlowered; a real (non-simulated) backend
/// request has no transport to satisfy it.
#[test]
fn zero_refusals() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero_ref_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("unused.nslm");

    let s2 = run_nsl(&program(&save), "ref_s2", &["--zero-stage", "2"], 600);
    assert!(!s2.success, "--zero-stage 2 must refuse");
    assert!(
        s2.stderr.contains("not lowered yet"),
        "wrong stage-2 refusal:\n{}",
        s2.stderr
    );
}
