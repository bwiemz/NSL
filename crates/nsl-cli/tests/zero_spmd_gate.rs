//! D3 — ZeRO-1 SPMD parity gates (toy scale, CPU, SimulatedBackend).
//!
//! `--zero-stage 1` + `nsl run --devices N` now performs REAL multi-rank
//! training: N processes over a CPU-shm SimulatedBackend, per-step
//! gradient all-reduce (fixed rank-order sum, then ÷N), owner-gated
//! optimizer updates (round-robin `idx % N`), and a post-step per-param
//! broadcast from each owner. With replicated data (every rank sees the
//! same batches — the training loader is rank-blind today) the reduced
//! gradient is `(N·g)/N == g` bit-exactly at N=2 (2·g is exact, /2 is
//! exponent-only; the gate runs N=2 — larger N is only exact up to N=4,
//! so do not generalize this to arbitrary world sizes), and the
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
//! Hazards handled here: a rank that hangs OFF the shm spin-barrier (a
//! compile/link stall, an in-process TP_CTX deadlock) is not caught by the
//! backend's own 300s barrier-timeout abort, so every SPMD invocation also
//! runs under this per-run watchdog. NOTE the watchdog only reaps the top
//! `nsl` process — its rank children and the binaries they exec are orphaned,
//! and they inherit our stdout/stderr pipe write-ends, so on a timeout we must
//! NOT join the drain threads (they would never see EOF); we snapshot what was
//! captured and panic. A red test is the goal; a hung join would defeat it.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

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
    stderr: String,
    success: bool,
}

/// Drain a child pipe on its own thread into a shared byte buffer. Bytes are
/// appended incrementally (never `read_to_string`, which drops the ENTIRE
/// capture on the first non-UTF-8 byte — a localized Windows linker diagnostic
/// would then false-red the `[zero]` counter search), so the watchdog can read
/// the partial capture WITHOUT joining the thread.
fn spawn_drain<R: Read + Send + 'static>(
    mut pipe: R,
    buf: Arc<Mutex<Vec<u8>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.lock().unwrap().extend_from_slice(&chunk[..n]),
            }
        }
    })
}

/// Run `nsl run` with a watchdog (SPMD spin-barriers hang forever on a
/// dead rank; a hung gate is worse than a red one).
fn run_nsl(source: &str, tag: &str, extra_args: &[&str], timeout_secs: u64) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_zero_gate_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("zero_gate.nsl");
    std::fs::write(&prog, source).unwrap();

    // Invoke the PRE-BUILT `nsl` binary directly (CARGO_BIN_EXE_nsl) rather
    // than `cargo run` — no build-directory lock, no per-call up-to-date
    // check. Cargo builds this package's bins before its integration tests, so
    // the path is valid; the `--devices N` self-spawn re-execs via
    // current_exe()+args().skip(1) (see run.rs), so it behaves identically.
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--deterministic"])
        .args(extra_args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        // Anti-vacuity: emit the [zero] collective-count line (ws==1 runs
        // report all_reduce=0 broadcast=0; multi-rank runs report the real
        // totals — a no-op reduce or a dropped --devices can't stay green).
        .env("NSL_ZERO_COUNTER", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn nsl run");

    // Drain stdout AND stderr on dedicated threads, concurrently with the
    // watchdog. A single `nsl run` emits several KB to stderr at compile time
    // alone (WGGO notes, the system linker's relocation warnings, the `[nsl]
    // deterministic mode` banner) plus the `[zero]` atexit line. Reading the
    // pipes only after the child exits would let a child that fills the OS pipe
    // buffer block on write forever — try_wait never sees it exit and the
    // watchdog fires a false "hang" (Linux pipes are 1 MiB and slip under it;
    // Windows anonymous pipes are far smaller and link.exe is chattier, so that
    // path deadlocked at exactly the 600 s watchdog while Linux stayed green).
    let out_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let err_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let out_reader = spawn_drain(
        child.stdout.take().expect("child stdout piped"),
        out_buf.clone(),
    );
    let err_reader = spawn_drain(
        child.stderr.take().expect("child stderr piped"),
        err_buf.clone(),
    );
    let snapshot = |buf: &Arc<Mutex<Vec<u8>>>| String::from_utf8_lossy(&buf.lock().unwrap()).into_owned();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let status = loop {
        match child.try_wait().expect("try_wait") {
            Some(s) => break s,
            None => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    // Do NOT join the drain threads here: child.kill() reaps only
                    // the top `nsl` process, but its rank children and the
                    // binaries they exec are orphaned and keep OUR pipe
                    // write-ends open, so the readers never hit EOF and a join
                    // would hang forever — defeating the whole watchdog. Snapshot
                    // the partial capture and panic; the leaked drain threads die
                    // when the test process exits.
                    panic!(
                        "nsl run '{tag}' exceeded {timeout_secs}s — SPMD rank hang?\n\
                         --- captured stdout ---\n{}\n--- captured stderr ---\n{}",
                        snapshot(&out_buf),
                        snapshot(&err_buf),
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    };
    // The child exited normally: it only returns after all its rank children
    // (and the binaries they exec) have been waited on, so every write-end is
    // closed and the readers have hit EOF — joining cannot block here.
    let _ = out_reader.join();
    let _ = err_reader.join();
    let stdout = snapshot(&out_buf);
    let stderr = snapshot(&err_buf);

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

    // Anti-vacuity: the collectives ACTUALLY ran. Replicated-data parity
    // alone is satisfied even by a no-op reduce (identical grads → sum/ws
    // == g); the counter line proves otherwise. Fixture: 8 params,
    // grad_accum=2, 13 micro-batches → 6 optimizer steps; reduce_grads and
    // sync_params each loop 8 params per step → 48 all_reduce + 48
    // broadcast per rank. Rank 0's line must show ws=2 and those exact
    // totals — a dropped --devices (children compiling world_size=1) or an
    // identity collective would report 0.
    let zero_line = spmd
        .stderr
        .lines()
        .find(|l| l.contains("[zero] ws=2 rank=0"))
        .unwrap_or_else(|| panic!("no [zero] ws=2 rank=0 line in SPMD stderr:\n{}", spmd.stderr))
        .to_string();
    assert!(
        zero_line.contains("all_reduce=48") && zero_line.contains("broadcast=48"),
        "ZeRO collective counts wrong (expected all_reduce=48 broadcast=48): {zero_line}"
    );
    // Baseline (single process, ws=1) must NOT have run any collective.
    assert!(
        base.stderr
            .lines()
            .filter(|l| l.contains("[zero]"))
            .all(|l| l.contains("all_reduce=0") && l.contains("broadcast=0")),
        "baseline ran collectives:\n{}",
        base.stderr
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
