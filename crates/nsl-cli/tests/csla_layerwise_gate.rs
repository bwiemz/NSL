//! CSLA Stage-2 (D1a) — window-buffered schedule parity gates.
//!
//! `--layerwise-accum` restructures the FASE-Deferred training loop: the N
//! micro-batches of each accumulation window run their forwards first (saving
//! only the adjoint-read primal values + batch dicts), then the backward
//! phase replays the whole window through a runtime loop over the buffered
//! micro-batches. The transformation must be BIT-EXACT vs the interleaved
//! `--checkpoint-blocks` baseline: same kernels, same inputs, same
//! per-parameter accumulation order (micro-batch ascending), so the loss
//! stream and the saved model bytes must match at full precision.
//!
//! Gate discipline (campaign standard):
//!   - baseline arm FIRST, run TWICE (run-to-run determinism probe);
//!   - anti-vacuity via the NSL_CSLA_COUNTER=1 atexit report — the layerwise
//!     arm must report EXACTLY the expected number of window-backward phases
//!     and the baseline must report 0, so a silently-inert flag can't pass;
//!   - CCR must not have declined ("running without checkpointing");
//!   - refusal cases assert the loud errors, not silent fallbacks.
//!
//! Window-semantics coverage:
//!   - `csla_parity_ffn_cpu`: 13 micro-batches, N=2, epochs=1 -> 6 windows +
//!     a TRAILING PARTIAL WINDOW (teardown sweep; tail must not step);
//!   - `csla_parity_ffn_cpu_epoch_straddle`: epochs=2 -> window 7 spans the
//!     epoch boundary (global step counter, DataLoader reset between);
//!   - GPU twins (#[ignore]) + the packed-GQA composition (per-micro-batch
//!     packing-metadata re-install on the replay path).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture_src(name: &str) -> String {
    std::fs::read_to_string(repo_root().join("crates/nsl-cli/tests/fixtures").join(name))
        .unwrap_or_else(|_| panic!("{name} fixture missing"))
}

fn program(fixture: &str, gpu: bool, save_path: &Path, rewrites: &[(&str, &str)]) -> String {
    let mut src = fixture_src(fixture);
    if gpu {
        assert!(src.contains("# GPU_PLACEMENT"));
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    assert!(src.contains("CSLA_SAVE_PATH"));
    src = src.replace(
        "CSLA_SAVE_PATH",
        &save_path.display().to_string().replace('\\', "/"),
    );
    for (from, to) in rewrites {
        assert!(src.contains(from), "rewrite marker '{from}' missing");
        src = src.replace(from, to);
    }
    src
}

struct RunOutput {
    loss_stream: String,
    stderr: String,
    success: bool,
}

fn run_program(source: &str, tag: &str, cuda: bool, deterministic: bool, extra_args: &[&str]) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_csla_gate_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("csla_gate.nsl");
    std::fs::write(&prog, source).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("run").arg("-q");
    if cuda {
        cmd.args(["--features", "cuda"]);
    }
    cmd.arg("--manifest-path")
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run", "--source-ad"]);
    if deterministic {
        cmd.arg("--deterministic");
    }
    cmd.args(extra_args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        // Anti-vacuity counter report on every arm; deterministic embedding
        // scatter for the CPU cases (harmless there, load-bearing on GPU
        // EmbedCpu-style runs).
        .env("NSL_CSLA_COUNTER", "1")
        .env("NSL_EMBEDDING_BWD_CPU", "1");
    let output = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

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
        success: output.status.success(),
    }
}

/// Parse the `[csla] window backward phases: N` atexit report.
fn window_phase_count(stderr: &str) -> Option<i64> {
    stderr
        .lines()
        .find_map(|l| l.strip_prefix("[csla] window backward phases: "))
        .and_then(|n| n.trim().parse::<i64>().ok())
}

/// The parity core: baseline (`--checkpoint-blocks`) twice as a determinism
/// probe, then the layerwise arm; bit-exact loss stream + model bytes when
/// the environment is deterministic; exact window-phase count both arms.
fn parity_case(
    fixture: &str,
    cuda: bool,
    deterministic: bool,
    tag: &str,
    rewrites: &[(&str, &str)],
    common_args: &[&str],
    expected_windows: i64,
) {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_saves_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("base_a.nslm");
    let save_b = tmp.join("base_b.nslm");
    let save_c = tmp.join("csla.nslm");

    // 1. Baseline determinism probe (two identical runs, baseline arm first).
    let base_a = run_program(
        &program(fixture, cuda, &save_a, rewrites),
        &format!("{tag}_a"),
        cuda,
        deterministic,
        &[&["--checkpoint-blocks"], common_args].concat(),
    );
    assert!(base_a.success, "baseline run A failed:\n{}", base_a.stderr);
    let base_b = run_program(
        &program(fixture, cuda, &save_b, rewrites),
        &format!("{tag}_b"),
        cuda,
        deterministic,
        &[&["--checkpoint-blocks"], common_args].concat(),
    );
    assert!(base_b.success, "baseline run B failed:\n{}", base_b.stderr);
    let bytes_a = std::fs::read(&save_a).expect("baseline A model_save missing");
    let bytes_b = std::fs::read(&save_b).expect("baseline B model_save missing");
    let env_deterministic = bytes_a == bytes_b && base_a.loss_stream == base_b.loss_stream;

    // Baseline must never enter the window backward.
    assert_eq!(
        window_phase_count(&base_a.stderr),
        Some(0),
        "baseline arm reported csla window phases != 0:\n{}",
        base_a.stderr
    );

    // 2. Layerwise arm.
    let csla = run_program(
        &program(fixture, cuda, &save_c, rewrites),
        &format!("{tag}_c"),
        cuda,
        deterministic,
        &[&["--checkpoint-blocks", "--layerwise-accum"], common_args].concat(),
    );
    assert!(csla.success, "--layerwise-accum run failed:\n{}", csla.stderr);
    assert!(
        !csla.stderr.contains("running without checkpointing"),
        "CCR declined under --layerwise-accum (should have been a hard error):\n{}",
        csla.stderr
    );
    // Anti-vacuity: the window backward fired exactly once per complete
    // accumulation window (trailing partial windows never fire, matching the
    // baseline's discarded m_partial tail).
    assert_eq!(
        window_phase_count(&csla.stderr),
        Some(expected_windows),
        "csla arm window-phase count != expected {expected_windows}:\n{}",
        csla.stderr
    );
    let bytes_c = std::fs::read(&save_c).expect("csla model_save missing");

    if env_deterministic {
        assert_eq!(
            base_a.loss_stream, csla.loss_stream,
            "loss stream diverged under --layerwise-accum (must be bit-exact)"
        );
        assert!(
            bytes_a == bytes_c,
            "saved model bytes diverged under --layerwise-accum (must be bit-exact)"
        );
    } else {
        eprintln!(
            "[csla-parity] WARNING: baseline is not run-to-run deterministic \
             in this environment ({tag}); falling back to loss-prefix comparison"
        );
        let first_a = base_a.loss_stream.lines().next().unwrap_or("");
        let first_c = csla.loss_stream.lines().next().unwrap_or("");
        assert_eq!(
            first_a, first_c,
            "first loss (pure forward on deterministic init) must still match"
        );
    }
}

/// CPU gate — 6 complete windows + a trailing partial window (13 micro-
/// batches, N=2): exercises the save/replay machinery AND the teardown sweep.
#[test]
fn csla_parity_ffn_cpu() {
    parity_case("csla_layerwise_ffn.nsl", false, false, "ffn_cpu", &[], &[], 6);
}

/// CPU gate — loss tensor READ BY THE ADJOINT (review M2): tanh's backward
/// reads its own output, so wrapping the loss in tanh() makes the loss VarId
/// a window-buffered import and engages the loss_slot machinery (b<N-1 skip
/// in the replay loop + the should_step-conditional per-iteration free) that
/// the cross_entropy fixtures never touch.
#[test]
fn csla_parity_ffn_cpu_loss_read_by_adjoint() {
    parity_case(
        "csla_layerwise_ffn.nsl",
        false,
        false,
        "ffn_loss_read",
        &[(
            "let loss = cross_entropy(flat_logits, flat_labels)",
            "let loss = tanh(cross_entropy(flat_logits, flat_labels))",
        )],
        &[],
        6,
    );
}

/// CPU gate — epochs=2: 26 micro-batches -> 13 windows, one of which spans
/// the epoch boundary (buffered saves from epoch 1's tail + epoch 2's head).
#[test]
fn csla_parity_ffn_cpu_epoch_straddle() {
    parity_case(
        "csla_layerwise_ffn.nsl",
        false,
        false,
        "ffn_straddle",
        &[("epochs=1", "epochs=2")],
        &[],
        13,
    );
}

/// GPU twin of the FFN gate under `--deterministic` (M46 kernels).
/// `cargo test -p nsl-cli --features cuda --test csla_layerwise_gate -- --ignored`
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_parity_ffn_gpu() {
    parity_case("csla_layerwise_ffn.nsl", true, true, "ffn_gpu", &[], &[], 6);
}

/// Packed-GQA composition, CPU: buffered packed-batch dict state
/// (segment_ids / position_ids / input_ids per micro-batch), the per-b
/// packing-registry re-install, and the packed EmbeddingBackward reading
/// buffered input_ids at replay time. CPU because attention-on-CUDA is
/// refused in D1a (review H2: the fused SDPA dispatch's logsumexp rides an
/// SSA side-band the replay cannot see) — the GPU arm is the refusal case
/// in csla_refusals; the D1b tape-carry restores GPU attention parity.
#[test]
fn csla_parity_packed_gqa_cpu() {
    parity_case(
        "csla_layerwise_packed_gqa.nsl",
        false,
        false,
        "packed_cpu",
        &[],
        // CPU compile target: the default "cuda" target emits the fused
        // SDPA dispatch table even for CPU-placed models, which the H2
        // refusal (correctly) rejects. Both arms share the target.
        &["--target", "cpu"],
        8,
    );
}

/// Deferral-must-refuse: every unsupported composition dies loudly at
/// compile time instead of silently running the baseline schedule.
#[test]
fn csla_refusals() {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_refusals_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // (a) grad_clip: two-phase clip needs the full-window global norm.
    let clip_src = program(
        "csla_layerwise_ffn.nsl",
        false,
        &tmp.join("clip.nslm"),
        &[(
            "grad_accumulation=2)",
            "grad_accumulation=2, grad_clip=1.0)",
        )],
    );
    let clip = run_program(
        &clip_src,
        "refuse_clip",
        false,
        false,
        &["--checkpoint-blocks", "--layerwise-accum"],
    );
    assert!(!clip.success, "grad_clip + --layerwise-accum must refuse");
    assert!(
        clip.stderr.contains("incompatible with grad_clip"),
        "expected the grad_clip refusal, got:\n{}",
        clip.stderr
    );

    // (b) missing --checkpoint-blocks: clap-level `requires` error.
    let plain_src = program("csla_layerwise_ffn.nsl", false, &tmp.join("noccr.nslm"), &[]);
    let noccr = run_program(&plain_src, "refuse_noccr", false, false, &["--layerwise-accum"]);
    assert!(
        !noccr.success,
        "--layerwise-accum without --checkpoint-blocks must refuse"
    );
    assert!(
        noccr.stderr.contains("checkpoint-blocks") || noccr.stderr.contains("checkpoint_blocks"),
        "expected the clap requires error naming checkpoint-blocks, got:\n{}",
        noccr.stderr
    );

    // (b2) attention on a CUDA target (review H2): every SDPA op on GPU
    // emits the fused dispatch (runtime variant selection) whose saved
    // logsumexp travels by SSA side-band — csla must refuse at compile time.
    // CPU targets build an empty variant table, so this arm only runs under
    // the cuda feature.
    #[cfg(feature = "cuda")]
    {
        let fused_src = program(
            "csla_layerwise_packed_gqa.nsl",
            true,
            &tmp.join("fused.nslm"),
            &[],
        );
        let fused = run_program(
            &fused_src,
            "refuse_fused_sdpa",
            true,
            true,
            &["--checkpoint-blocks", "--layerwise-accum"],
        );
        assert!(
            !fused.success,
            "SDPA attention on CUDA + --layerwise-accum must refuse"
        );
        assert!(
            fused
                .stderr
                .contains("does not yet support attention on CUDA targets"),
            "expected the fused-SDPA refusal, got:\n{}",
            fused.stderr
        );
    }

    // (c) missing --source-ad: codegen admission error. Bypass the shared
    // runner (it always passes --source-ad) with a direct invocation.
    let root = repo_root();
    let prog = tmp.join("no_source_ad.nsl");
    std::fs::write(
        &prog,
        program("csla_layerwise_ffn.nsl", false, &tmp.join("nosad.nslm"), &[]),
    )
    .unwrap();
    let out = Command::new(env!("CARGO"))
        .arg("run")
        .arg("-q")
        .arg("--manifest-path")
        .arg(root.join("Cargo.toml"))
        .args([
            "-p",
            "nsl-cli",
            "--",
            "run",
            "--checkpoint-blocks",
            "--layerwise-accum",
        ])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(
        !out.status.success(),
        "--layerwise-accum without --source-ad must refuse"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("requires --source-ad"),
        "expected the source-ad refusal, got:\n{stderr}"
    );
}
