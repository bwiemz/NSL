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
    let zero_lines: Vec<&str> = z3
        .stderr
        .lines()
        .filter(|l| l.starts_with("[zero] ws="))
        .collect();
    assert_eq!(zero_lines.len(), 2, "expected 2 rank counter lines:\n{}", z3.stderr);
    let mut all_gather_total = 0;
    for l in &zero_lines {
        let rs = counter_field(l, "reduce_scatter").expect("reduce_scatter field");
        let ag = counter_field(l, "all_gather").expect("all_gather field");
        assert!(rs > 0, "stage-3 elementwise must reduce_scatter: {l}");
        assert!(ag > 0, "elementwise gathers must ride all_gather: {l}");
        all_gather_total = ag; // identical on both ranks (symmetric schedule)
    }
    assert!(
        all_gather_total < gathers,
        "no tensor-granular gathers — the mixed-mode arm went vacuous \
         (all_gather={all_gather_total}, gathers={gathers})"
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
