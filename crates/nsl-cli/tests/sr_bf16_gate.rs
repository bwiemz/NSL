//! P4 item 17 gates — SR-BF16 authoritative weights (`--param-dtype bf16-sr`).
//!
//! What the e2e gates prove:
//!   * the bf16 mirror schedule actually ran (teardown counters > 0) and
//!     training stayed sane (finite, decreasing loss; model saved);
//!   * the SR counter stream is DETERMINISTIC: same seed → bit-identical
//!     loss stream on rerun; different seed → different stream (the dice
//!     are real and compiler-owned, not launch-order noise);
//!   * bf16-authoritative training tracks the f32 baseline closely at this
//!     scale (quantization noise, not divergence).
//! The refusal gates pin the composition matrix (deferral-must-refuse).

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

fn run_nsl(source: &str, tag: &str, extra_args: &[&str], timeout_secs: u64) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_srbf16_gate_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("srbf16_gate.nsl");
    std::fs::write(&prog, source).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--deterministic"])
        .args(extra_args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
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

fn losses(stdout: &str) -> Vec<f64> {
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
                        inner.parse::<f64>().ok()
                    } else {
                        l.parse::<f64>().ok()
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Raw loss-stream text (for bit-identity assertions).
fn loss_text(stdout: &str) -> String {
    stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default()
}

const FULL_FLAGS: &[&str] = &[
    "--checkpoint-blocks",
    "--layerwise-accum",
    "--weight-stream",
    "--param-dtype",
    "bf16-sr",
];

/// bf16-sr without the streaming residency schedule refuses with the
/// actionable message (the mirrors ride weight streaming).
#[test]
fn srbf16_refuses_without_weight_stream() {
    let tmp = std::env::temp_dir().join(format!("nsl_srbf16_ref_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("ref.nslm");
    let out = run_nsl(
        &program(&save, false),
        "no_ws",
        &["--checkpoint-blocks", "--layerwise-accum", "--param-dtype", "bf16-sr"],
        300,
    );
    assert!(!out.success, "bf16-sr without --weight-stream ran:\n{}", out.stdout);
    assert!(
        out.stderr.contains("--param-dtype bf16-sr requires --weight-stream"),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// bf16-sr × optimizer-state offload refuses (m/v must be plain device f32).
#[test]
fn srbf16_refuses_offload() {
    let tmp = std::env::temp_dir().join(format!("nsl_srbf16_off_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("off.nslm");
    let mut args = FULL_FLAGS.to_vec();
    args.push("--optim-state-offload");
    let out = run_nsl(&program(&save, false), "offload", &args, 300);
    assert!(!out.success, "bf16-sr with offload ran:\n{}", out.stdout);
    assert!(
        out.stderr
            .contains("--param-dtype bf16-sr does not compose with --optim-state-offload"),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// bf16-sr × --training-reference refuses (needs the fused step).
#[test]
fn srbf16_refuses_training_reference() {
    let tmp = std::env::temp_dir().join(format!("nsl_srbf16_tr_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("tr.nslm");
    let mut args = FULL_FLAGS.to_vec();
    args.push("--training-reference");
    let out = run_nsl(&program(&save, false), "trref", &args, 300);
    assert!(!out.success, "bf16-sr with --training-reference ran:\n{}", out.stdout);
    assert!(
        out.stderr.contains("--training-reference"),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// Parse "[sr-bf16] teardown: N bf16-authoritative param(s), M SR optimizer
/// step(s), K widen-upload(s)".
fn teardown_counts(stderr: &str) -> Option<(u64, u64, u64)> {
    let line = stderr.lines().find(|l| l.contains("[sr-bf16] teardown:"))?;
    let nums: Vec<u64> = line
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    if nums.len() >= 4 {
        // nums[0] is the "16" from "[sr-bf16]" — skip it.
        Some((nums[1], nums[2], nums[3]))
    } else {
        None
    }
}

/// Core e2e: bf16-sr training runs the mirror schedule (counters > 0),
/// keeps the loss finite and decreasing, saves the model, and is
/// BIT-DETERMINISTIC in the seed; a different seed changes the SR dice.
#[test]
#[ignore = "requires CUDA GPU"]
fn srbf16_e2e_deterministic_and_sane_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_srbf16_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let save_a = tmp.join("a.nslm");
    let mut args = FULL_FLAGS.to_vec();
    args.extend_from_slice(&["--seed", "4242"]);
    let a = run_nsl(&program(&save_a, true), "e2e_a", &args, 600);
    assert!(a.success, "bf16-sr run failed:\nstdout:\n{}\nstderr:\n{}", a.stdout, a.stderr);

    let (params, steps, uploads) = teardown_counts(&a.stderr)
        .unwrap_or_else(|| panic!("no [sr-bf16] teardown counters in stderr:\n{}", a.stderr));
    assert!(params > 0, "no bf16-authoritative params (vacuous run)");
    assert!(steps > 0, "no SR optimizer steps fired (vacuous run)");
    assert!(uploads > 0, "no widen-uploads (schedule never streamed)");

    let la = losses(&a.stdout);
    assert!(la.len() >= 10, "loss stream too short: {la:?}");
    assert!(la.iter().all(|v| v.is_finite()), "non-finite loss: {la:?}");
    assert!(
        la.last().unwrap() < la.first().unwrap(),
        "loss did not decrease under bf16-sr: first={} last={}",
        la.first().unwrap(),
        la.last().unwrap()
    );
    assert!(save_a.exists(), "model_save did not produce a file");

    // Determinism: same seed → bit-identical loss stream.
    let save_b = tmp.join("b.nslm");
    let b = run_nsl(&program(&save_b, true), "e2e_b", &args, 600);
    assert!(b.success, "rerun failed:\n{}", b.stderr);
    assert_eq!(
        loss_text(&a.stdout),
        loss_text(&b.stdout),
        "same seed must reproduce the loss stream bit-identically"
    );

    // Seed sensitivity: the SR dice must actually be keyed on the seed.
    let save_c = tmp.join("c.nslm");
    let mut args_c = FULL_FLAGS.to_vec();
    args_c.extend_from_slice(&["--seed", "4243"]);
    let c = run_nsl(&program(&save_c, true), "e2e_c", &args_c, 600);
    assert!(c.success, "seed-4243 run failed:\n{}", c.stderr);
    assert_ne!(
        loss_text(&a.stdout),
        loss_text(&c.stdout),
        "different seed left the loss stream unchanged — SR stream is not \
         keyed on the seed (note: --seed also reseeds init, so identical \
         streams here are doubly impossible)"
    );
}

/// bf16-authoritative training tracks the f32 baseline at this scale:
/// same config minus the flag, same seed — every loss within a loose
/// quantization-noise envelope, both curves decreasing.
#[test]
#[ignore = "requires CUDA GPU"]
fn srbf16_tracks_f32_baseline_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_srbf16_base_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let save_q = tmp.join("q.nslm");
    let mut args_q = FULL_FLAGS.to_vec();
    args_q.extend_from_slice(&["--seed", "4242"]);
    let q = run_nsl(&program(&save_q, true), "trk_q", &args_q, 600);
    assert!(q.success, "bf16-sr run failed:\n{}", q.stderr);

    let save_f = tmp.join("f.nslm");
    let f = run_nsl(
        &program(&save_f, true),
        "trk_f",
        &[
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--weight-stream",
            "--seed",
            "4242",
        ],
        600,
    );
    assert!(f.success, "f32 baseline failed:\n{}", f.stderr);

    let lq = losses(&q.stdout);
    let lf = losses(&f.stdout);
    assert_eq!(lq.len(), lf.len(), "loss stream length mismatch");
    assert!(!lq.is_empty());
    for (i, (a, b)) in lq.iter().zip(lf.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.25,
            "bf16-sr diverged from f32 baseline at step {i}: {a} vs {b}\nbf16: {lq:?}\nf32:  {lf:?}"
        );
    }
    assert!(lf.last().unwrap() < lf.first().unwrap(), "baseline loss did not decrease");
}
