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

/// D3 v3: the EVEN-micro-batch (12) rank-aware DP fixture. `grad_accum` picks
/// the arm: 2 = single-process baseline (consecutive pairs); 1 = per-rank DP
/// (each rank contributes one micro-batch per step, averaged by the all-reduce).
fn program_rank_aware(save_path: &Path, grad_accum: u32) -> String {
    let src = std::fs::read_to_string(
        repo_root().join("crates/nsl-cli/tests/fixtures/csla_rank_aware_dp.nsl"),
    )
    .expect("rank-aware DP fixture missing");
    let src = src.replace(
        "CSLA_SAVE_PATH",
        &save_path.display().to_string().replace('\\', "/"),
    );
    if grad_accum == 1 {
        src.replace("grad_accumulation=2", "grad_accumulation=1")
    } else {
        src
    }
}

/// Parse the newline-joined LOSS_STREAM body into per-micro-batch loss values.
fn parse_losses(stream: &str) -> Vec<f64> {
    stream
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect()
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

/// 2 SPMD ranks over the shm backend with REPLICATED data (the default —
/// `shard_by_rank` is opt-in and this fixture does not set it, so every rank
/// sees the full batch set). Rank 0's loss stream must be bit-identical to the
/// single-rank baseline: `(N·g)/N == g` exactly. This is the TIGHT check —
/// unlike the tolerance-based rank-aware gate it catches a global gradient-scale
/// error (a missing/wrong `÷world_size` in the all-reduce), which AdamW's
/// scale-invariance hides from a loss-parity comparison. `zero_stage1_rank_aware_dp_parity`
/// below is its complement: the SAME machinery under REAL (sharded) data.
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

    // Anti-vacuity: the collectives ACTUALLY ran. Replicated-data parity alone
    // is satisfied even by a no-op reduce (identical grads → sum/ws == g); the
    // counter line proves otherwise. Fixture: 8 params, grad_accum=2, 13
    // micro-batches → 6 optimizer steps; reduce_grads and sync_params each loop
    // 8 params per step → 6 bucketed all_reduce + 12 owner-bucketed broadcast per rank.
    let zero_line = spmd
        .stderr
        .lines()
        .find(|l| l.contains("[zero] ws=2 rank=0"))
        .unwrap_or_else(|| panic!("no [zero] ws=2 rank=0 line in SPMD stderr:\n{}", spmd.stderr))
        .to_string();
    assert!(
        zero_line.contains("all_reduce=6")
            && zero_line.contains("broadcast=12")
            && zero_line.contains("bucket_members=96"),
        "ZeRO collective counts wrong (expected 6 grad buckets, 12 owner \
         broadcasts, 96 bucketed members): {zero_line}"
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
    // θ-sync evidence: the saved model matches the baseline bit-for-bit.
    let ba = std::fs::read(&save_a).expect("baseline save missing");
    let bb = std::fs::read(&save_b).expect("spmd save missing");
    assert!(
        ba == bb,
        "2-rank ZeRO-1 model bytes diverged from baseline — θ broadcast or \
         owner gating is wrong"
    );
}

/// Extract `optim_elems=N` from the `[zero] ws=W rank=R ...` atexit line.
fn optim_elems(stderr: &str, ws: u64, rank: u64) -> u64 {
    let needle = format!("[zero] ws={ws} rank={rank}");
    let line = stderr
        .lines()
        .find(|l| l.contains(&needle))
        .unwrap_or_else(|| panic!("no '{needle}' line in stderr:\n{stderr}"));
    line.split_whitespace()
        .find_map(|tok| tok.strip_prefix("optim_elems="))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no optim_elems= field in line: {line}"))
}

/// G3 (D3 v2 — the MEMORY WIN): `--zero-stage 1` shards the optimizer-state
/// ALLOCATION, not just the update. v1 allocated full m/v on every rank
/// (owner-gated the update only, so zero bytes were saved); v2 gates the
/// allocation on ownership (owner = idx % world_size), so each rank holds a
/// STRICT SUBSET of the moment surface and the two together cover the whole.
///
/// This is invisible to the loss-parity gate (non-owners never read their
/// non-owned m/v, so results are identical either way) — only the per-rank
/// `optim_elems` counter proves the allocation actually shrank. Round-robin
/// by param index (rank0 {0,2,4,6} / rank1 {1,3,5,7}) with unequally-sized
/// params means the split is not exactly half, so the invariant is
/// r0+r1==full (complete, nothing lost) AND 0<r0<full AND 0<r1<full (a real
/// shrink on both ranks, neither holding everything).
#[test]
#[ignore = "spawns multiple processes; run explicitly"]
fn zero_stage1_optim_state_is_sharded() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero_shard_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_full = tmp.join("full.nslm");
    let save_spmd = tmp.join("spmd.nslm");

    // ws=1 --zero-stage 1: rank 0 owns every param → allocates the FULL
    // moment surface (2 moments × sum of all param elements).
    let full_run = run_nsl(&program(&save_full), "shard_full", &["--zero-stage", "1"], 600);
    assert!(full_run.success, "ws=1 zero run failed:\n{}", full_run.stderr);
    let full = optim_elems(&full_run.stderr, 1, 0);
    assert!(
        full > 0,
        "ws=1 --zero-stage 1 should allocate the full moment surface, got 0:\n{}",
        full_run.stderr
    );

    // --devices 2 --zero-stage 1: each rank allocates ONLY its owned shard.
    let spmd = run_nsl(
        &program(&save_spmd),
        "shard_2r",
        &["--zero-stage", "1", "--devices", "2"],
        900,
    );
    assert!(spmd.success, "2-rank zero run failed:\n{}", spmd.stderr);
    let r0 = optim_elems(&spmd.stderr, 2, 0);
    let r1 = optim_elems(&spmd.stderr, 2, 1);

    // Complete partition: every param's m/v is allocated by exactly one rank.
    assert_eq!(
        r0 + r1,
        full,
        "sharded optim elems must sum to the full surface (r0={r0} r1={r1} full={full})\n{}",
        spmd.stderr
    );
    // Real shrink on BOTH ranks: each strictly less than full, neither empty.
    assert!(
        r0 > 0 && r0 < full,
        "rank 0 optim state is not a strict shard: r0={r0} full={full}"
    );
    assert!(
        r1 > 0 && r1 < full,
        "rank 1 optim state is not a strict shard: r1={r1} full={full}"
    );
}

/// D3 v3 (RANK-AWARE DATA-PARALLEL): the loader now hands each rank a DISJOINT
/// data shard (rank r takes global micro-batches where idx % world_size == r),
/// so the ZeRO all-reduce averages gradients over the TRUE global batch instead
/// of replicated data. This is the first gate where the two ranks see DIFFERENT
/// data, so — unlike G1 — it is NOT bit-exact against the single-rank baseline
/// (real DP is numerically EQUIVALENT, not bit-identical: f32 reassociation
/// across the FASE Passthrough(accum=1) vs Deferred(accum=2) path boundary and
/// separate-process gradients leave ~1 f32 ULP of drift). We validate the
/// equivalence within tolerance instead.
///
/// Construction (proven bit-close in practice, ~1e-7 rel): baseline = 1 process,
/// grad_accumulation=2 → each optimizer step aggregates the consecutive pair
/// {2k, 2k+1}. DP = 2 ranks, grad_accumulation=1 → rank0 sees {0,2,4,6,8,10},
/// rank1 sees {1,3,5,7,9,11}; each step all-reduce-MEANS the two ranks'
/// single-micro-batch gradients — the SAME pair {2k, 2k+1} the baseline
/// averages. So rank0's k-th micro-batch loss L(θ_k, mb_{2k}) tracks the
/// baseline's (2k)-th loss.
#[test]
#[ignore = "spawns multiple processes; run explicitly"]
fn zero_stage1_rank_aware_dp_parity() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero_ra_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_base = tmp.join("ra_base.nslm");
    let save_dp = tmp.join("ra_dp.nslm");

    // Baseline: single process, grad_accum=2, rank-blind loader → 12
    // micro-batch losses (6 consecutive pairs, no trailing tail: 12 batches).
    let base = run_nsl(&program_rank_aware(&save_base, 2), "ra_base", &[], 600);
    assert!(base.success, "baseline failed:\n{}", base.stderr);
    let base_losses = parse_losses(&base.loss_stream);
    assert_eq!(
        base_losses.len(),
        12,
        "baseline should emit 12 micro-batch losses:\n{}",
        base.loss_stream
    );

    // Real DP: 2 ranks, grad_accum=1, rank-aware strided loader. rank0 (the
    // only rank whose stdout reaches us) processes ITS shard {0,2,4,6,8,10} →
    // 6 losses. Rank-blindness would instead give rank0 all 12 → 12 losses.
    let dp = run_nsl(
        &program_rank_aware(&save_dp, 1),
        "ra_dp",
        &["--zero-stage", "1", "--devices", "2"],
        900,
    );
    assert!(dp.success, "2-rank DP run failed:\n{}", dp.stderr);
    let dp_losses = parse_losses(&dp.loss_stream);
    assert_eq!(
        dp_losses.len(),
        6,
        "rank0 should emit 6 losses (its shard) not 12 (rank-blind):\n{}",
        dp.loss_stream
    );

    // Sharding + gradient-average correctness: rank0's k-th loss tracks the
    // baseline's (2k)-th loss within f32 tolerance. A wrong shard or wrong
    // averaging diverges by O(0.1+) (the distinct micro-batches differ by
    // whole loss units), far above this bound.
    for k in 0..6 {
        let got = dp_losses[k];
        let want = base_losses[2 * k];
        let rel = (got - want).abs() / want.abs().max(1.0);
        assert!(
            rel < 1e-3,
            "rank0 loss[{k}]={got} != baseline even-index[{}]={want} (rel {rel:.3e}) \
             — data shard or gradient averaging is wrong",
            2 * k
        );
    }

    // Anti-vacuity #1: rank0 did NOT see the ODD micro-batches. If the loader
    // were still rank-blind, rank0's 2nd loss would track baseline's 2nd
    // (micro-batch 1); the strided shard makes it baseline's 3rd (micro-batch
    // 2), which is a whole loss unit away.
    assert!(
        (dp_losses[1] - base_losses[1]).abs() > 0.1,
        "rank0 loss[1]={} too close to baseline loss[1]={} — loader did NOT shard \
         (still rank-blind?)",
        dp_losses[1],
        base_losses[1]
    );

    // Anti-vacuity #2: the collectives actually ran — 6 optimizer steps × 8
    // params = 6 bucketed all_reduce + 12 owner-bucketed broadcast per rank; both ranks present.
    let z0 = dp
        .stderr
        .lines()
        .find(|l| l.contains("[zero] ws=2 rank=0"))
        .unwrap_or_else(|| panic!("no [zero] ws=2 rank=0 line:\n{}", dp.stderr));
    assert!(
        z0.contains("all_reduce=6") && z0.contains("broadcast=12"),
        "wrong collective counts (expected 48/48): {z0}"
    );
    assert!(
        dp.stderr.contains("[zero] ws=2 rank=1"),
        "rank1 [zero] line missing — second rank did not run:\n{}",
        dp.stderr
    );
}

/// Loud refusals: stage 2/3 unlowered; a real (non-simulated) backend
/// request has no transport to satisfy it.
#[test]
fn zero_refusals() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero_ref_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("unused.nslm");

    let s3 = run_nsl(&program(&save), "ref_s3", &["--zero-stage", "3"], 600);
    assert!(!s3.success, "--zero-stage 3 must refuse");
    assert!(
        s3.stderr.contains("not lowered yet"),
        "wrong stage-3 refusal:\n{}",
        s3.stderr
    );
}

/// P4 item 16 (ZeRO-2): gradient partitioning. Same replicated-data parity
/// contract as stage 1 — the loss stream must be bit-identical to the
/// single-rank baseline — but grads move through owner-segmented
/// reduce_scatter collectives (each rank receives only its OWNED summed
/// gradients), so the [zero] line must show reduce_scatter>0 with
/// all_reduce=0 (no grad was all-reduced).
#[test]
#[ignore = "spawns 3 nsl processes; ~2 min. Run: cargo test --test zero_spmd_gate -- --ignored"]
fn zero_stage2_two_rank_parity_reduce_scatter() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero_s2_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("base.nslm");
    let save_b = tmp.join("s2.nslm");

    let base = run_nsl(&program(&save_a), "s2_base", &[], 600);
    assert!(base.success, "baseline failed:\n{}", base.stderr);
    let spmd = run_nsl(
        &program(&save_b),
        "s2_spmd",
        &["--zero-stage", "2", "--devices", "2"],
        900,
    );
    assert!(spmd.success, "2-rank stage-2 run failed:\n{}", spmd.stderr);
    assert!(!base.loss_stream.is_empty(), "empty baseline loss stream");
    assert_eq!(
        base.loss_stream, spmd.loss_stream,
        "stage-2 rank-0 loss stream diverged from the single-rank baseline\n{}",
        spmd.stderr
    );
    for rank in 0..2 {
        let line = spmd
            .stderr
            .lines()
            .find(|l| l.contains(&format!("[zero] ws=2 rank={rank}")))
            .unwrap_or_else(|| panic!("no [zero] line for rank {rank}:\n{}", spmd.stderr));
        assert!(
            line.contains("reduce_scatter=6")
                && line.contains("all_reduce=0")
                && line.contains("broadcast=12"),
            "stage-2 collective counts wrong (want 6 scatters, 0 all_reduce, \
             12 owner broadcasts): {line}"
        );
    }
    // Same model bytes as baseline (θ fully synced despite partitioned grads).
    let base_bytes = std::fs::read(&save_a).expect("baseline .nslm");
    let s2_bytes = std::fs::read(&save_b).expect("stage-2 .nslm");
    assert_eq!(base_bytes, s2_bytes, "stage-2 model bytes diverged");
}
