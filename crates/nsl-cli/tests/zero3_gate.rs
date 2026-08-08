//! P3 ZeRO-3 gates — tensor-granular parameter sharding on the layerwise
//! schedule (items 12-14).
//!
//! The bit-exactness argument mirrors zero_spmd_gate.rs: with a rank-blind
//! loader every rank computes identical window gradients, the per-layer
//! all-reduce averages `(g+g)/2 == g` bit-exactly at N=2, the owner's
//! update is the same arithmetic as the single-rank baseline, and
//! non-owners refetch the owner's post-update θ at the next window's
//! gather — so the rank-0 loss stream and the saved model bytes must be
//! BIT-IDENTICAL to the same config without `--zero-stage 3 --devices 2`.
//! That equivalence exercises registration (owner keeps / non-owner
//! frees), the JIT gather at every forward-segment and window-range head,
//! the release after each layer's update, the per-layer reduce, the
//! owner gate, and the teardown restore (model_save reads full replicas).

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

struct RunOutput {
    success: bool,
    stdout: String,
    stderr: String,
}

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

/// Watchdogged `nsl run` (SPMD spin-barriers hang forever on a dead rank).
fn run_nsl_with_env(
    source: &str,
    tag: &str,
    extra_args: &[&str],
    envs: &[(&str, &str)],
    timeout_secs: u64,
) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_zero3_gate_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("zero3_gate.nsl");
    std::fs::write(&prog, source).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--deterministic"])
        .args(extra_args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn nsl run");
    let out_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let err_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let out_reader = spawn_drain(child.stdout.take().unwrap(), out_buf.clone());
    let err_reader = spawn_drain(child.stderr.take().unwrap(), err_buf.clone());
    let snapshot =
        |buf: &Arc<Mutex<Vec<u8>>>| String::from_utf8_lossy(&buf.lock().unwrap()).into_owned();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => {
                out_reader.join().ok();
                err_reader.join().ok();
                return RunOutput {
                    success: status.success(),
                    stdout: snapshot(&out_buf),
                    stderr: snapshot(&err_buf),
                };
            }
            None if std::time::Instant::now() > deadline => {
                child.kill().ok();
                panic!(
                    "watchdog: nsl run '{tag}' exceeded {timeout_secs}s\nstdout:\n{}\nstderr:\n{}",
                    snapshot(&out_buf),
                    snapshot(&err_buf),
                );
            }
            None => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    }
}

fn program(save_path: &Path, gpu: bool) -> String {
    let src = std::fs::read_to_string(
        repo_root().join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .expect("ffn fixture missing");
    let src = src.replace(
        "CSLA_SAVE_PATH",
        &save_path.display().to_string().replace('\\', "/"),
    );
    if gpu {
        src.replace("# GPU_PLACEMENT", "m.to(cuda)")
    } else {
        src
    }
}

fn losses(stdout: &str) -> Vec<String> {
    stdout
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
        .unwrap_or_default()
}

/// Item 12/13 admission: stage 3 without the layerwise residency schedule
/// refuses with the actionable flag list.
#[test]
fn zero3_refuses_without_layerwise_schedule() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero3_ref_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("ref.nslm");
    let out = run_nsl_with_env(
        &program(&save, false),
        "refusal",
        &["--zero-stage", "3", "--devices", "2"],
        &[],
        300,
    );
    assert!(!out.success, "stage 3 without csla flags ran:\n{}", out.stdout);
    assert!(
        out.stderr
            .contains("--zero-stage 3 requires --layerwise-accum --weight-stream"),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// Item 11: `--zero-elementwise` on stages 1/2 would be an inert flag —
/// refuse with the actionable message instead.
#[test]
fn zero_elementwise_requires_stage3() {
    let tmp = std::env::temp_dir().join(format!("nsl_z3e_s1_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("e1.nslm");
    let out = run_nsl_with_env(
        &program(&save, false),
        "elem_s1",
        &["--zero-stage", "1", "--zero-elementwise"],
        &[],
        300,
    );
    assert!(!out.success, "stage 1 + elementwise ran:\n{}", out.stdout);
    assert!(
        out.stderr.contains("--zero-elementwise requires --zero-stage 3"),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// Item 11: Muon needs whole matrices for Newton-Schulz — elementwise
/// slices refuse it up front (gather-before-NS is the documented follow-up).
#[test]
fn zero_elementwise_refuses_muon() {
    let tmp = std::env::temp_dir().join(format!("nsl_z3e_mu_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("mu.nslm");
    let src = program(&save, false).replace(
        "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
        "Muon(lr=0.002, momentum=0.95, nesterov=true, ns_steps=5, \
         weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
    );
    let out = run_nsl_with_env(
        &src,
        "elem_muon",
        &[
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--weight-stream",
            "--zero-stage",
            "3",
            "--devices",
            "2",
            "--zero-elementwise",
        ],
        &[],
        300,
    );
    assert!(!out.success, "muon + elementwise ran:\n{}", out.stdout);
    assert!(
        out.stderr
            .contains("--zero-elementwise requires the AdamW/Adam optimizer"),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// Stage 4 does not exist.
#[test]
fn zero_stage_4_refuses() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero3_s4_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("s4.nslm");
    let out = run_nsl_with_env(
        &program(&save, false),
        "stage4",
        &["--zero-stage", "4"],
        &[],
        300,
    );
    assert!(!out.success);
    assert!(
        out.stderr.contains("--zero-stage 4 does not exist"),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// The core parity gate: stage-3 sharded 2-rank training (sim-gpu
/// collectives, one physical GPU) is BIT-IDENTICAL to the single-rank
/// run of the same layerwise config — loss stream and saved bytes — and
/// the zero3 schedule demonstrably ran (gathers/releases > 0).
#[test]
#[ignore = "requires CUDA GPU (sim-gpu collectives, 2 ranks on 1 device)"]
fn zero3_bit_exact_vs_single_rank_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero3_bx_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let flags_common = [
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
    ];

    let save_base = tmp.join("base.nslm");
    let base = run_nsl_with_env(
        &program(&save_base, true),
        "base",
        &flags_common,
        &[],
        600,
    );
    assert!(base.success, "single-rank baseline failed:\n{}", base.stderr);
    let base_losses = losses(&base.stdout);
    assert!(!base_losses.is_empty(), "empty baseline stream");

    let save_z3 = tmp.join("z3.nslm");
    let mut z3_args: Vec<&str> = flags_common.to_vec();
    // --collectives is a FLAG (the spawner deliberately overrides any
    // inherited NSL_COLLECTIVES with it — default "sim" would clobber an
    // env-only request).
    z3_args.extend_from_slice(&[
        "--zero-stage",
        "3",
        "--devices",
        "2",
        "--collectives",
        "sim-gpu",
    ]);
    let z3 = run_nsl_with_env(&program(&save_z3, true), "z3", &z3_args, &[], 900);
    assert!(z3.success, "zero3 2-rank run failed:\n{}", z3.stderr);

    // The spawner forwards RANK 0's stdout — the captured stream is one
    // rank's loss sequence and must be BIT-IDENTICAL to the baseline.
    let z3_losses = losses(&z3.stdout);
    assert_eq!(
        base_losses, z3_losses,
        "rank-0 zero3 loss stream diverged from the single-rank baseline\nstderr:\n{}",
        z3.stderr
    );

    // Anti-vacuity: the residency schedule actually gathered and released.
    assert!(
        z3.stderr.contains("[zero3] tensor-granular parameter sharding enabled"),
        "zero3 enable note missing:\n{}",
        z3.stderr
    );
    let teardown_ok = z3
        .stderr
        .lines()
        .filter(|l| l.contains("[zero3] teardown"))
        .all(|l| !l.contains("gathers=0"));
    assert!(
        teardown_ok && z3.stderr.contains("[zero3] teardown"),
        "zero3 teardown counters missing or vacuous:\n{}",
        z3.stderr
    );

    // model_save end state: full replicas, identical bytes.
    let a = std::fs::read(&save_base).expect("baseline .nslm");
    let b = std::fs::read(&save_z3).expect("zero3 .nslm");
    assert_eq!(a, b, "model bytes diverged under zero3");
}

/// Item 11: rewrite the fixture's hidden dim 64 -> 63 so the model carries
/// BOTH sharding modes at ws=2: the block norms (63 elems, odd) stay
/// tensor-granular while w_up/w_down (63x128 = 8064) go elementwise —
/// the mixed-mode plan is the production shape, not a corner case.
fn oddify(src: String) -> String {
    src.replace("ones([64])", "ones([63])")
        .replace("randn([64, 128])", "randn([63, 128])")
        .replace("randn([128, 64])", "randn([128, 63])")
        .replace("randn([64, 64]) * 0.1", "randn([64, 63]) * 0.1")
        .replace("reshape([batch_size, seq_len, 64])", "reshape([batch_size, seq_len, 63])")
}

/// Label-anchored counter field from a `[zero3] teardown:` or `[zero]` line.
fn counter_field(line: &str, label: &str) -> Option<u64> {
    line.split(&format!("{label}="))
        .nth(1)?
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

/// `optim_elems=N` off the `[zero] ws=W rank=R ...` atexit line — the
/// per-rank total of optimizer-moment ELEMENTS actually allocated. Item C's
/// only real instrument: loss parity is INVARIANT to whether moments are
/// owner-only or replicated (a non-owner never reads its non-owned m/v), so
/// a broken implementation stays bit-exact and only this number moves.
///
/// `contains`, not `starts_with`: a neighbouring fragment without a
/// trailing newline can land immediately before the (single-write) counter
/// line — same reason the all_gather parser below uses `contains`.
fn optim_elems(stderr: &str, ws: u64, rank: u64) -> u64 {
    let needle = format!("[zero] ws={ws} rank={rank}");
    let line = stderr
        .lines()
        .find(|l| l.contains(&needle))
        .unwrap_or_else(|| panic!("no '{needle}' line in stderr:\n{stderr}"));
    counter_field(line, "optim_elems")
        .unwrap_or_else(|| panic!("no optim_elems= field in line: {line}"))
}

/// Item C's shared assertion: the sharded moment surface is a COMPLETE,
/// STRICT partition across the two ranks. `full` first, as
/// zero_spmd_gate.rs does — a zero baseline would make every other test
/// here pass vacuously.
fn assert_moment_partition(full: u64, r0: u64, r1: u64, ctx: &str) {
    assert!(
        full > 0,
        "{ctx}: the ws=1 run allocated NO sharded moments — the counter is \
         dead and the partition assertions below would be vacuous"
    );
    assert_eq!(
        r0 + r1,
        full,
        "{ctx}: sharded moment elements must sum to the ws=1 surface \
         (r0={r0} r1={r1} full={full}) — a rank is holding a full replica, \
         or a slot was dropped entirely"
    );
    assert!(
        r0 > 0 && r0 < full,
        "{ctx}: rank 0 optimizer state is not a strict shard: r0={r0} full={full}"
    );
    assert!(
        r1 > 0 && r1 < full,
        "{ctx}: rank 1 optimizer state is not a strict shard: r1={r1} full={full}"
    );
}

/// Item 11 core parity gate: elementwise-sharded 2-rank training (sim-gpu)
/// is BIT-IDENTICAL to the single-rank baseline of the same layerwise
/// config — loss stream and saved bytes — and the elementwise machinery
/// demonstrably ran: params carved, every-rank slice steps executed,
/// gradients reduce_scattered (stage 3 never scatters on the
/// tensor-granular path), gathers rode all_gather. The odd norms prove
/// MIXED mode: granular gathers coexist (gathers > all_gather count).
#[test]
#[ignore = "requires CUDA GPU (sim-gpu collectives, 2 ranks on 1 device)"]
fn zero3_elementwise_bit_exact_vs_single_rank_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_z3e_bx_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let flags_common = [
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
    ];

    let save_base = tmp.join("e_base.nslm");
    let base = run_nsl_with_env(
        &oddify(program(&save_base, true)),
        "e_base",
        &flags_common,
        &[],
        600,
    );
    assert!(base.success, "single-rank baseline failed:\n{}", base.stderr);
    let base_losses = losses(&base.stdout);
    assert!(!base_losses.is_empty(), "empty baseline stream");

    let save_z3 = tmp.join("e_z3.nslm");
    let mut z3_args: Vec<&str> = flags_common.to_vec();
    z3_args.extend_from_slice(&[
        "--zero-stage",
        "3",
        "--devices",
        "2",
        "--collectives",
        "sim-gpu",
        "--zero-elementwise",
    ]);
    let z3 = run_nsl_with_env(
        &oddify(program(&save_z3, true)),
        "e_z3",
        &z3_args,
        &[("NSL_ZERO_COUNTER", "1")],
        900,
    );
    assert!(z3.success, "elementwise 2-rank run failed:\n{}", z3.stderr);

    // Parity: rank-0 loss stream and saved model bytes.
    assert_eq!(
        base_losses,
        losses(&z3.stdout),
        "elementwise loss stream diverged from the single-rank baseline\nstderr:\n{}",
        z3.stderr
    );
    let a = std::fs::read(&save_base).expect("baseline .nslm");
    let b = std::fs::read(&save_z3).expect("elementwise .nslm");
    assert_eq!(a, b, "model bytes diverged under elementwise zero3");

    // Anti-vacuity, both directions:
    assert!(
        z3.stderr.contains("[zero3] elementwise sharding armed"),
        "elementwise arming note missing:\n{}",
        z3.stderr
    );
    let teardown = z3
        .stderr
        .lines()
        .find(|l| l.contains("[zero3] teardown"))
        .unwrap_or_else(|| panic!("no teardown line:\n{}", z3.stderr));
    let elem_params = counter_field(teardown, "elem_params").expect("elem_params field");
    let elem_steps = counter_field(teardown, "elem_steps").expect("elem_steps field");
    // 2 blocks x (w_up, w_down) elementwise; the 63-elem norms are ragged
    // and MUST stay tensor-granular (mixed mode is the property under test).
    assert_eq!(
        elem_params, 4,
        "expected exactly the 4 even-sized matrices elementwise:\n{teardown}"
    );
    assert!(
        elem_steps >= elem_params,
        "elementwise steps never ran:\n{teardown}"
    );
    let gathers = counter_field(teardown, "gathers").expect("gathers field");

    // The [zero] counter lines (one per rank): reduce_scatter and
    // all_gather both nonzero on EVERY rank — and all_gather strictly
    // below the total gather count, which proves granular (broadcast)
    // gathers coexisted.
    // `contains`, not `starts_with`: the counter line itself is single-write
    // atomic, but a NEIGHBOR fragment without a trailing newline can land
    // immediately before it (review nit).
    let zero_lines: Vec<&str> = z3
        .stderr
        .lines()
        .filter(|l| l.contains("[zero] ws="))
        .collect();
    assert_eq!(zero_lines.len(), 2, "expected 2 rank counter lines:\n{}", z3.stderr);
    let ag_per_rank: Vec<u64> = zero_lines
        .iter()
        .map(|l| {
            let rs = counter_field(l, "reduce_scatter").expect("reduce_scatter field");
            let ag = counter_field(l, "all_gather").expect("all_gather field");
            assert!(rs > 0, "stage-3 elementwise must reduce_scatter: {l}");
            assert!(ag > 0, "elementwise gathers must ride all_gather: {l}");
            ag
        })
        .collect();
    // The schedule is symmetric — assert it instead of assuming it.
    assert_eq!(
        ag_per_rank[0], ag_per_rank[1],
        "asymmetric all_gather counts across ranks:\n{}",
        z3.stderr
    );
    assert!(
        ag_per_rank[0] < gathers,
        "no tensor-granular gathers — the mixed-mode arm went vacuous \
         (all_gather={}, gathers={gathers})",
        ag_per_rank[0]
    );

    // ── Item C: the moment surface is OWNER-ONLY, not replicated ────────
    //
    // Every assertion above is invariant to this: a rank never reads the
    // m/v it does not own, so replicated moments produce the identical
    // loss stream and the identical saved bytes. `optim_elems` is the only
    // number that moves — flip the fill loop's elementwise arm back to a
    // full `zeros_like` and the partition below goes red while parity
    // stays green (that is the anti-vacuity probe for this gate).
    //
    // `elem_moments` (teardown line) additionally proves the SLICE
    // allocator was the one that ran: 4 elementwise params x 2 moments.
    let elem_moments = counter_field(teardown, "elem_moments").expect("elem_moments field");
    assert_eq!(
        elem_moments,
        elem_params * 2,
        "expected one m and one v SLICE per elementwise param — a fill that \
         fell back to a full zeros_like reads 0 here:\n{teardown}"
    );

    // `full` = the same config at ws=1, where every eligible param is a
    // 1/1 slice and rank 0 therefore allocates the WHOLE sharded surface.
    let save_ws1 = tmp.join("e_ws1.nslm");
    let mut ws1_args: Vec<&str> = flags_common.to_vec();
    ws1_args.extend_from_slice(&[
        "--zero-stage",
        "3",
        "--devices",
        "1",
        "--collectives",
        "sim-gpu",
        "--zero-elementwise",
    ]);
    let ws1 = run_nsl_with_env(
        &oddify(program(&save_ws1, true)),
        "e_ws1",
        &ws1_args,
        &[("NSL_ZERO_COUNTER", "1")],
        600,
    );
    assert!(ws1.success, "ws=1 elementwise run failed:\n{}", ws1.stderr);
    let full = optim_elems(&ws1.stderr, 1, 0);
    assert_moment_partition(
        full,
        optim_elems(&z3.stderr, 2, 0),
        optim_elems(&z3.stderr, 2, 1),
        "zero3 elementwise moments",
    );
}

/// Item C on the TENSOR-GRANULAR arm (`--zero-stage 3` without
/// `--zero-elementwise`): the same owner gate stages 1/2 ship now runs at
/// the deferred fill, so each rank allocates m/v only for the sharded
/// params it owns. Parity is asserted alongside the partition for the same
/// reason as the elementwise gate — parity alone cannot see the change, and
/// the partition alone would not catch an owner gate that skipped the
/// UPDATE too.
#[test]
#[ignore = "requires CUDA GPU (sim-gpu collectives, 2 ranks on 1 device)"]
fn zero3_tensor_granular_moments_are_owner_only_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_z3g_om_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let flags_common = [
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
    ];

    let save_base = tmp.join("g_base.nslm");
    let base = run_nsl_with_env(
        &oddify(program(&save_base, true)),
        "g_base",
        &flags_common,
        &[],
        600,
    );
    assert!(base.success, "single-rank baseline failed:\n{}", base.stderr);
    let base_losses = losses(&base.stdout);
    assert!(!base_losses.is_empty(), "empty baseline stream");

    let zero3_only = |dev: &'static str| -> Vec<&'static str> {
        let mut a: Vec<&'static str> = flags_common.to_vec();
        a.extend_from_slice(&["--zero-stage", "3", "--devices", dev, "--collectives", "sim-gpu"]);
        a
    };

    let save_ws1 = tmp.join("g_ws1.nslm");
    let ws1 = run_nsl_with_env(
        &oddify(program(&save_ws1, true)),
        "g_ws1",
        &zero3_only("1"),
        &[("NSL_ZERO_COUNTER", "1")],
        600,
    );
    assert!(ws1.success, "ws=1 stage-3 run failed:\n{}", ws1.stderr);

    let save_ws2 = tmp.join("g_ws2.nslm");
    let ws2 = run_nsl_with_env(
        &oddify(program(&save_ws2, true)),
        "g_ws2",
        &zero3_only("2"),
        &[("NSL_ZERO_COUNTER", "1")],
        900,
    );
    assert!(ws2.success, "2-rank stage-3 run failed:\n{}", ws2.stderr);

    // Parity first — an owner gate that also skipped the update would shrink
    // `optim_elems` and silently stop training.
    assert_eq!(
        base_losses,
        losses(&ws2.stdout),
        "tensor-granular loss stream diverged from the single-rank baseline\nstderr:\n{}",
        ws2.stderr
    );
    assert_eq!(
        std::fs::read(&save_base).expect("baseline .nslm"),
        std::fs::read(&save_ws2).expect("zero3 .nslm"),
        "model bytes diverged under tensor-granular zero3 owner-only moments"
    );

    // No elementwise machinery on this arm — the fill must have taken the
    // owner-gated arm, not the slice allocator.
    let teardown = ws2
        .stderr
        .lines()
        .find(|l| l.contains("[zero3] teardown"))
        .unwrap_or_else(|| panic!("no teardown line:\n{}", ws2.stderr));
    assert_eq!(
        counter_field(teardown, "elem_moments"),
        Some(0),
        "tensor-granular stage 3 must not allocate slice moments:\n{teardown}"
    );

    assert_moment_partition(
        optim_elems(&ws1.stderr, 1, 0),
        optim_elems(&ws2.stderr, 2, 0),
        optim_elems(&ws2.stderr, 2, 1),
        "zero3 tensor-granular moments",
    );
}

/// Item C x Muon: the deferred fill stacks the OWNER gate on top of the
/// ROUTE gate. A Muon-routed rank-2 param's `v` is unread, so it must stay
/// null on EVERY rank; an AdamW-routed one must still be owner-only. Both
/// halves are visible in one number — `optim_elems` here is strictly less
/// than the AdamW configuration's, because the routed matrices contribute
/// m only.
///
/// Deliberately does NOT carry the θ-probing callback that
/// `zero3_muon_overlap_and_callback_touch_gpu` uses — because that gate
/// ALREADY certifies it. It runs the same owner-only deferred fill with a
/// per-micro-batch `on_step` that reads gathered θ, and its whole-stream
/// parity assertion compares those probe values, so a fill that dropped a
/// non-owner's `m` or mis-timed against the callback's read breaks it.
/// What this gate adds on top is the RAGGED (63-dim) shape and the
/// `optim_elems` partition number, neither of which that gate asserts.
///
/// (An earlier revision of this comment claimed the sibling gate was
/// broken on main with a glibc "double free or corruption". It is not:
/// it passes, and the composition it covers passes with it. The abort
/// that prompted the claim was a stale `libnsl_runtime.a` in a shared
/// CARGO_TARGET_DIR — see the mtime fallback at
/// `crates/nsl-codegen/src/linker.rs` — which surfaces as
/// "CUDA support not compiled", not as a double free.)
#[test]
#[ignore = "requires CUDA GPU (sim-gpu collectives, 2 ranks on 1 device)"]
fn zero3_muon_moments_are_owner_only_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_z3mu_om_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let muonize = |src: String| -> String {
        src.replace(
            "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
            "Muon(lr=0.002, momentum=0.95, nesterov=true, ns_steps=5, \
             weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
        )
    };
    let flags_common = [
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
    ];

    let save_base = tmp.join("mu_base.nslm");
    let base = run_nsl_with_env(
        &muonize(oddify(program(&save_base, true))),
        "mu_base",
        &flags_common,
        &[],
        600,
    );
    assert!(base.success, "muon single-rank baseline failed:\n{}", base.stderr);
    let base_losses = losses(&base.stdout);
    assert!(!base_losses.is_empty(), "empty baseline stream");

    let zero3_only = |dev: &'static str| -> Vec<&'static str> {
        let mut a: Vec<&'static str> = flags_common.to_vec();
        a.extend_from_slice(&["--zero-stage", "3", "--devices", dev, "--collectives", "sim-gpu"]);
        a
    };

    let save_ws1 = tmp.join("mu_ws1.nslm");
    let ws1 = run_nsl_with_env(
        &muonize(oddify(program(&save_ws1, true))),
        "mu_ws1",
        &zero3_only("1"),
        &[("NSL_ZERO_COUNTER", "1")],
        600,
    );
    assert!(ws1.success, "muon ws=1 stage-3 run failed:\n{}", ws1.stderr);

    let save_ws2 = tmp.join("mu_ws2.nslm");
    let ws2 = run_nsl_with_env(
        &muonize(oddify(program(&save_ws2, true))),
        "mu_ws2",
        &zero3_only("2"),
        &[("NSL_ZERO_COUNTER", "1")],
        900,
    );
    assert!(ws2.success, "muon 2-rank stage-3 run failed:\n{}", ws2.stderr);

    assert_eq!(
        base_losses,
        losses(&ws2.stdout),
        "muon x zero3 owner-only moments diverged from the single-rank \
         baseline\nstderr:\n{}",
        ws2.stderr
    );
    assert_eq!(
        std::fs::read(&save_base).expect("baseline .nslm"),
        std::fs::read(&save_ws2).expect("zero3 .nslm"),
        "model bytes diverged under muon x zero3 owner-only moments"
    );

    let full = optim_elems(&ws1.stderr, 1, 0);
    assert_moment_partition(
        full,
        optim_elems(&ws2.stderr, 2, 0),
        optim_elems(&ws2.stderr, 2, 1),
        "zero3 muon moments",
    );
    // The route gate must ALSO still bite: the 4 rank-2 matrices carry m
    // only, so the Muon surface is strictly smaller than the AdamW one
    // (which allocates m AND v for every sharded param). Without this the
    // test would pass with the route gate deleted.
    let adamw_ws1 = run_nsl_with_env(
        &oddify(program(&tmp.join("mu_adamw_ws1.nslm"), true)),
        "mu_adamw_ws1",
        &zero3_only("1"),
        &[("NSL_ZERO_COUNTER", "1")],
        600,
    );
    assert!(adamw_ws1.success, "adamw ws=1 reference run failed:\n{}", adamw_ws1.stderr);
    let adamw_full = optim_elems(&adamw_ws1.stderr, 1, 0);
    assert!(
        full < adamw_full,
        "muon-routed params must skip v even under the deferred fill \
         (muon full={full}, adamw full={adamw_full})"
    );
}

/// Compositions on the same parity bar:
/// - Muon (the tensor-granular owner gate wraps the mixed muon step);
/// - the overlap flags (--stream-arena --stream-prefetch
///   --stream-async-writeback: pack/prefetch/async-evict entry points all
///   redirect to gather/release — item 14's issue-early structure);
/// - a callback that READS model θ mid-training (the item-12 residency
///   bracket: upload_all gathers, reevict_all releases; the printed probe
///   value must match the baseline bit-for-bit).
#[test]
#[ignore = "requires CUDA GPU (sim-gpu collectives, 2 ranks on 1 device)"]
fn zero3_muon_overlap_and_callback_touch_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_zero3_mx_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let muonize = |src: String| -> String {
        let s = src.replace(
            "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
            "Muon(lr=0.002, momentum=0.95, nesterov=true, ns_steps=5, \
             weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
        );
        // Mid-training model-θ read from the callback: sums GENUINELY
        // SHARDED params (review finding 3: `embed` is tied/view-rooted and
        // stays Replicated — probing it never exercises the crash guard).
        // The blocks' w_up are layer-grouped and streamed: evicted on
        // non-owners (and, mid-window, on every rank) unless the residency
        // bracket gathers. Iteration form — `m.blocks[0]` subscripting
        // miscompiles in callbacks (pre-existing, crashes the emitted
        // program).
        s.replace(
            "on_step(step, loss):",
            "on_step(step, loss):\n            for pb in m.blocks:\n                print(sum(pb.w_up).item())",
        )
    };

    let flags: Vec<&str> = vec![
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
        "--stream-arena",
        "--stream-prefetch",
        "--stream-async-writeback",
    ];

    let save_base = tmp.join("mx_base.nslm");
    let base = run_nsl_with_env(
        &muonize(program(&save_base, true)),
        "mx_base",
        &flags,
        &[],
        600,
    );
    assert!(base.success, "muon overlap baseline failed:\n{}", base.stderr);
    let base_losses = losses(&base.stdout);
    assert!(!base_losses.is_empty(), "empty baseline stream");

    let save_z3 = tmp.join("mx_z3.nslm");
    let mut z3_args = flags.clone();
    z3_args.extend_from_slice(&[
        "--zero-stage",
        "3",
        "--devices",
        "2",
        "--collectives",
        "sim-gpu",
    ]);
    let z3 = run_nsl_with_env(
        &muonize(program(&save_z3, true)),
        "mx_z3",
        &z3_args,
        &[],
        900,
    );
    assert!(z3.success, "muon zero3 run failed:\n{}", z3.stderr);
    assert_eq!(
        base_losses,
        losses(&z3.stdout),
        "muon x zero3 x overlap-flags stream diverged (incl. the callback \
         theta probe)\nstderr:\n{}",
        z3.stderr
    );
    let a = std::fs::read(&save_base).expect("baseline .nslm");
    let b = std::fs::read(&save_z3).expect("zero3 .nslm");
    assert_eq!(a, b, "model bytes diverged under muon x zero3");
}

/// Item 16×11 refusals, both fail-closed arms:
/// - bf16-sr × tensor-granular stage 3 (no `--zero-elementwise`) refuses at
///   the envelope — the owner-gated update would leave non-owner slices
///   un-stepped;
/// - bf16-sr × elementwise with a RAGGED streamed param refuses at plan
///   derivation — the tensor-granular fallback would train that param
///   without stochastic rounding.
#[test]
fn srbf16_zero3_refuses_without_elementwise_and_on_ragged_params() {
    let tmp = std::env::temp_dir().join(format!("nsl_z3sr_ref_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("sr_ref.nslm");
    let flags_common = [
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
        "--param-dtype",
        "bf16-sr",
    ];

    let mut granular: Vec<&str> = flags_common.to_vec();
    granular.extend_from_slice(&["--zero-stage", "3", "--devices", "2"]);
    let out = run_nsl_with_env(&program(&save, false), "sr_granular", &granular, &[], 300);
    assert!(
        !out.success,
        "bf16-sr x tensor-granular stage 3 ran:\n{}",
        out.stdout
    );
    assert!(
        out.stderr
            .contains("only under --zero-elementwise"),
        "wrong refusal:\n{}",
        out.stderr
    );

    let mut ragged: Vec<&str> = flags_common.to_vec();
    ragged.extend_from_slice(&[
        "--zero-stage",
        "3",
        "--devices",
        "2",
        "--zero-elementwise",
    ]);
    let out = run_nsl_with_env(&oddify(program(&save, false)), "sr_ragged", &ragged, &[], 300);
    assert!(
        !out.success,
        "bf16-sr x elementwise with 63-elem ragged norms ran:\n{}",
        out.stdout
    );
    assert!(
        out.stderr.contains("cannot shard elementwise"),
        "wrong refusal (expected the plan-level ragged fail-closed):\n{}",
        out.stderr
    );
}

/// Item 16×11 parity: composed `--param-dtype bf16-sr --zero-stage 3
/// --zero-elementwise` 2-rank training (sim-gpu collectives, one physical
/// GPU) is BIT-IDENTICAL to single-rank PLAIN bf16-sr — loss stream and
/// saved model bytes. The chain: rank-blind loaders make both ranks'
/// window grads identical, reduce_scatter+1/ws is `(g+g)/2 == g` bit-exact
/// at ws=2, each rank's slice was carved by the same f32→bf16 cast as the
/// mirror, and the composed SR kernel draws each element's dither at
/// `param_base + rank*shard + i` — the single-rank counter for the same
/// global element. Un-oddified fixture: every streamed param divides by 2,
/// so the whole streamed set is elementwise (mixed mode is REFUSED under
/// SR — that is the ragged refusal above, not this gate).
///
/// Item C also carries the moment-partition instrument here. Two distinct
/// regressions hide from parity alone on this arm: (a) a fill that kept
/// full replicated moments for SR params — parity green, `optim_elems`
/// never shrinks, and the PR's headline claim is false for the composed
/// config; (b) an SR step that reads slice-sized moments at `+ off_bytes`
/// — every rank aborts, which parity DOES catch, but only because the
/// length assert exists. The `full` baseline is the same composed config
/// at `--devices 1`, where each slice is the whole parameter.
#[test]
#[ignore = "requires CUDA GPU (sim-gpu collectives, 2 ranks on 1 device)"]
fn srbf16_elementwise_bit_exact_vs_plain_sr_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_z3sr_bx_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let flags_common = [
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
        "--param-dtype",
        "bf16-sr",
    ];

    let save_base = tmp.join("sr_base.nslm");
    let base = run_nsl_with_env(
        &program(&save_base, true),
        "sr_base",
        &flags_common,
        &[],
        600,
    );
    assert!(base.success, "plain-SR baseline failed:\n{}", base.stderr);
    let base_losses = losses(&base.stdout);
    assert!(!base_losses.is_empty(), "empty baseline stream");
    assert!(
        base.stderr.contains("[sr-bf16] teardown:"),
        "baseline never armed the SR backend:\n{}",
        base.stderr
    );

    let save_sr = tmp.join("sr_z3.nslm");
    let mut sr_args: Vec<&str> = flags_common.to_vec();
    sr_args.extend_from_slice(&[
        "--zero-stage",
        "3",
        "--devices",
        "2",
        "--collectives",
        "sim-gpu",
        "--zero-elementwise",
    ]);
    let sr = run_nsl_with_env(
        &program(&save_sr, true),
        "sr_z3",
        &sr_args,
        &[("NSL_ZERO_COUNTER", "1")],
        900,
    );
    assert!(sr.success, "composed 2-rank run failed:\n{}", sr.stderr);

    assert_eq!(
        base_losses,
        losses(&sr.stdout),
        "composed loss stream diverged from single-rank plain SR\nstderr:\n{}",
        sr.stderr
    );
    let a = std::fs::read(&save_base).expect("baseline .nslm");
    let b = std::fs::read(&save_sr).expect("composed .nslm");
    assert_eq!(a, b, "model bytes diverged under composed bf16-sr x zero3");

    // Anti-vacuity: the SR-elementwise machinery demonstrably ran, and NO
    // streamed param silently took a non-SR arm (sr_elem == elem on both
    // counters — a composed run that trains any slice without stochastic
    // rounding must not read green here).
    let teardown = sr
        .stderr
        .lines()
        .find(|l| l.contains("[zero3] teardown"))
        .unwrap_or_else(|| panic!("no teardown line:\n{}", sr.stderr));
    let elem_params = counter_field(teardown, "elem_params").expect("elem_params field");
    let elem_steps = counter_field(teardown, "elem_steps").expect("elem_steps field");
    let sr_params = counter_field(teardown, "sr_elem_params").expect("sr_elem_params field");
    let sr_steps = counter_field(teardown, "sr_elem_steps").expect("sr_elem_steps field");
    assert!(sr_params > 0, "no bf16-sr slices carved:\n{teardown}");
    assert_eq!(
        sr_params, elem_params,
        "some elementwise params carved WITHOUT bf16-sr storage:\n{teardown}"
    );
    assert!(sr_steps > 0, "no composed SR steps ran:\n{teardown}");
    assert_eq!(
        sr_steps, elem_steps,
        "some elementwise steps took the plain f32 arm:\n{teardown}"
    );

<<<<<<< HEAD
    // The SR backend's OWN teardown must run on a composed run. It did not
    // before review: nsl_weight_stream_teardown returned after the zero3
    // arm, so SRBF16_ACTIVE leaked past the train block (inverting the
    // first-match-wins dispatch for every later weight-stream call) and
    // every NSL_SR_HIST sample the composed step recorded was discarded.
    // Both surfaces are pinned here — the line's existence, and that its
    // step count reflects the composed steps rather than reading 0 (the
    // false "stochastic rounding never ran" a certification consumer sees).
    let sr_line = sr
        .stderr
        .lines()
        .find(|l| l.contains("[sr-bf16] teardown:"))
        .unwrap_or_else(|| {
            panic!(
                "no [sr-bf16] teardown line on a composed run — the SR \
                 backend never tore down:\n{}",
                sr.stderr
            )
        });
    // Label-anchored on the field's own suffix, then the token immediately
    // before it (the line reads "..., N SR optimizer step(s), ...").
    let sr_reported: u64 = sr_line
        .split(" SR optimizer step(s)")
        .next()
        .and_then(|p| p.split_whitespace().next_back())
        .and_then(|p| p.parse().ok())
        .unwrap_or_else(|| panic!("unparseable SR step count:\n{sr_line}"));
    assert!(
        sr_reported > 0,
        "the SR backend reports 0 steps on a run that executed {sr_steps} \
         composed SR steps — tooling keyed on this counter would conclude \
         stochastic rounding never ran:\n{sr_line}"
    );


    // Item C: the SLICE allocator ran for the SR-carved params too — one m
    // and one v each. A fill that special-cased bf16-sr back onto a full
    // `zeros_like` (to keep the un-converted SR step happy) reads 0 here
    // while every assertion above stays green.
    let elem_moments = counter_field(teardown, "elem_moments").expect("elem_moments field");
    assert_eq!(
        elem_moments,
        elem_params * 2,
        "expected one m and one v SLICE per bf16-sr elementwise param — a \
         fill that kept replicated moments for SR reads 0 here:\n{teardown}"
    );

    // `full` = the same composed config at ws=1, where every streamed param
    // is a 1/1 slice and rank 0 therefore allocates the WHOLE sharded
    // optimizer surface. This is the only number that moves when the SR
    // moments stop being sharded — parity is invariant to it.
    let save_ws1 = tmp.join("sr_ws1.nslm");
    let mut ws1_args: Vec<&str> = flags_common.to_vec();
    ws1_args.extend_from_slice(&[
        "--zero-stage",
        "3",
        "--devices",
        "1",
        "--collectives",
        "sim-gpu",
        "--zero-elementwise",
    ]);
    let ws1 = run_nsl_with_env(
        &program(&save_ws1, true),
        "sr_ws1",
        &ws1_args,
        &[("NSL_ZERO_COUNTER", "1")],
        600,
    );
    assert!(ws1.success, "ws=1 composed SR run failed:\n{}", ws1.stderr);
    assert_moment_partition(
        optim_elems(&ws1.stderr, 1, 0),
        optim_elems(&sr.stderr, 2, 0),
        optim_elems(&sr.stderr, 2, 1),
        "bf16-sr x zero3 elementwise moments",
    );
}
