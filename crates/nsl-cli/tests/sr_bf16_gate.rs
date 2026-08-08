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
    run_nsl_env(source, tag, extra_args, &[], timeout_secs)
}

/// `run_nsl` plus environment overrides for the child — the batched-parity
/// gate forces the per-param arm with `NSL_FASE_MULTI_STEP=0`.
fn run_nsl_env(
    source: &str,
    tag: &str,
    extra_args: &[&str],
    envs: &[(&str, &str)],
    timeout_secs: u64,
) -> RunOutput {
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
        .envs(envs.iter().copied())
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
/// The number immediately PRECEDING `label` on the teardown line.
///
/// Label-anchored on purpose: the old positional parser split the line on
/// non-digits and indexed the pieces, with a comment skipping "the '16'
/// from '[sr-bf16]'" — but "bf16-authoritative" contributes a SECOND 16,
/// so (params, steps, uploads) actually read (params, 16, steps) for the
/// whole life of that parser. Its consumers only asserted `> 0`, which 16
/// satisfies, so nothing noticed until the 4th counter landed on the wrong
/// field and a kill-switch assert read 120.
fn teardown_field(stderr: &str, label: &str) -> Option<u64> {
    let line = stderr.lines().find(|l| l.contains("[sr-bf16] teardown:"))?;
    let end = line.find(label)?;
    line[..end]
        .trim_end()
        .rsplit(|c: char| !c.is_ascii_digit())
        .next()
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
}

fn teardown_counts(stderr: &str) -> Option<(u64, u64, u64)> {
    Some((
        teardown_field(stderr, " bf16-authoritative")?,
        teardown_field(stderr, " SR optimizer step")?,
        teardown_field(stderr, " widen-upload")?,
    ))
}

/// The 4th teardown counter (item 8): batched SR launches. 0 under the
/// per-param loop, >0 when the group update batched.
fn teardown_batched(stderr: &str) -> Option<u64> {
    teardown_field(stderr, " batched launch")
}

/// Item 8, bf16-SR arm: the batched group update
/// (`nsl_fase_fused_adamw_step_bf16sr_multi_idx`, one flat-grid launch per
/// wd bucket per group) must be BIT-IDENTICAL to the per-param SR loop it
/// replaces. This holds by construction — the SR dither is a pure function
/// of (seed, step, param_idx << SR_PARAM_SHIFT, element), independent of
/// launch shape, and the kernel body is byte-for-byte the per-param
/// arithmetic — and this gate keeps it empirically true.
///
/// The batched-launch counters make the comparison non-vacuous in BOTH
/// directions: the batched arm must actually batch (>0) and the
/// kill-switch arm must actually not (==0). Without them, an admission
/// regression sends both arms down the per-param loop and the equality
/// silently compares a path to itself (which is exactly what the first
/// draft of this feature did).
#[test]
#[ignore = "requires CUDA GPU"]
fn srbf16_batched_step_matches_per_param_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_srbf16_multi_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut args = FULL_FLAGS.to_vec();
    args.extend_from_slice(&["--seed", "4242"]);

    // Widen the FFN to 130 so w_up/w_down are 8320 elements = 32 full
    // 256-thread blocks + a PARTIAL tail block at a NONZERO element base.
    // The stock fixture's multi-block params are exact multiples of 256 (its
    // only ragged params are single-block 64-elem norms with base 0), so
    // without this a bounds-guard bug of the shape "compare tid instead of
    // bbtab[b]+tid against ntab[p]" would pass — the item-8 aligned-fixture
    // lesson, applied to this gate.
    let widen = |src: &str| -> String {
        let out = src
            .replace("randn([64, 128])", "randn([64, 130])")
            .replace("randn([128, 64])", "randn([130, 64])");
        assert_ne!(out, src, "FFN width markers missing from fixture");
        out
    };

    let batched = run_nsl(&widen(&program(&tmp.join("mb.nslm"), true)), "multi_b", &args, 600);
    assert!(
        batched.success,
        "batched bf16-sr run failed:\nstdout:\n{}\nstderr:\n{}",
        batched.stdout, batched.stderr
    );
    let nb = teardown_batched(&batched.stderr)
        .unwrap_or_else(|| panic!("no batched counter in teardown:\n{}", batched.stderr));
    assert!(
        nb > 0,
        "the batched arm never batched — admission regressed and this gate \
         is comparing the per-param loop to itself: {}",
        batched.stderr
    );

    let per_param = run_nsl_env(
        &widen(&program(&tmp.join("mp.nslm"), true)),
        "multi_p",
        &args,
        &[("NSL_FASE_MULTI_STEP", "0")],
        600,
    );
    assert!(
        per_param.success,
        "per-param bf16-sr run failed:\n{}",
        per_param.stderr
    );
    let np = teardown_batched(&per_param.stderr)
        .unwrap_or_else(|| panic!("no batched counter in teardown:\n{}", per_param.stderr));
    assert_eq!(
        np, 0,
        "NSL_FASE_MULTI_STEP=0 did not disable batching — the kill switch \
         is dead: {}",
        per_param.stderr
    );

    let lb = losses(&batched.stdout);
    assert!(lb.len() >= 10, "loss stream too short: {lb:?}");
    assert_eq!(
        loss_text(&batched.stdout),
        loss_text(&per_param.stdout),
        "batched SR step diverged from the per-param loop — the SR counter \
         stream or the update arithmetic is no longer launch-shape-independent"
    );
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

/// The bf16-sr parameter surface must be accounted as `weights`, not folded
/// into `activations` / `other`.
///
/// This is an ACCOUNTING gate, and it exists because the defect it pins was
/// invisible to every other test in this file. `alloc_managed` inherits
/// whatever allocation surface is ambient, and `srbf16_register` /
/// `srbf16_upload` run from the register belt inside the step body where
/// codegen has set `activations`. Before the fix, the 1B@2048 memory report
/// read `weights 400 MB / activations 12.23 GB` — bf16-sr appeared to make
/// activations 1.81 GB WORSE while "saving" 3.62 GB of weights, which is the
/// opposite of what it does. Correctness gates cannot catch this: the bytes
/// are in the right place, only the label is wrong.
///
/// Asserted as a RATIO against the run's own total rather than an absolute
/// byte count, so the fixture may change size freely. The failure mode being
/// pinned is "the parameter surface reports as ~0", so the bar is low and
/// one-directional on purpose.
#[test]
#[ignore = "requires GPU"]
fn srbf16_parameter_surface_is_accounted_as_weights_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_srbf16_surf_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("surf.nslm");

    let r = run_nsl_env(
        &program(&save, true),
        "surface",
        &FULL_FLAGS
            .iter()
            .copied()
            .chain(["--seed", "4242"])
            .collect::<Vec<_>>(),
        // NSL_MEMSTATS prints the end-of-run per-surface table; without it the
        // only surface output is the per-step line, which has no `weights`
        // reading at the moment the parameters are actually resident.
        &[("NSL_MEMSTATS", "1")],
        600,
    );
    assert!(r.success, "bf16-sr run failed:\n{}", r.stderr);

    // Non-vacuity first: this must be a real bf16-sr run, or "weights > 0"
    // would be satisfied by a plain f32 run that never took this path.
    let (params, steps, _uploads) = teardown_counts(&r.stderr)
        .unwrap_or_else(|| panic!("no [sr-bf16] teardown counters:\n{}", r.stderr));
    assert!(params > 0 && steps > 0, "vacuous bf16-sr run: {params} params, {steps} steps");

    // Rows look like:
    //   weights   current   16.5 KB  peak  145.5 KB  at-global-peak  81.0 KB
    //
    // The CURRENT column is the discriminating one and the PEAK column is
    // not: `weights` peak includes the initial f32 parameter allocation that
    // codegen tags correctly at train setup, in BOTH the fixed and broken
    // cases, so a peak-based assertion passes either way (verified — it did).
    // What the bug changes is where the bytes live once bf16-sr has replaced
    // that allocation with a mirror plus a widened view.
    let col = |name: &str, which: usize| -> Option<f64> {
        let line = r
            .stderr
            .lines()
            .find(|l| l.trim_start().starts_with(name) && l.contains("at-global-peak"))?;
        // ["current", V, U, "peak", V, U, "at-global-peak", V, U]
        let f: Vec<&str> = line.split_whitespace().skip(1).collect();
        let (v, u) = match which {
            0 => (f.get(1)?, f.get(2)?),
            1 => (f.get(4)?, f.get(5)?),
            _ => (f.get(7)?, f.get(8)?),
        };
        let scale = match *u {
            "GB" => 1024.0 * 1024.0 * 1024.0,
            "MB" => 1024.0 * 1024.0,
            "KB" => 1024.0,
            "B" => 1.0,
            _ => return None,
        };
        Some(v.parse::<f64>().ok()? * scale)
    };

    let weights_cur = col("weights", 0)
        .unwrap_or_else(|| panic!("no `weights` surface row in:\n{}", r.stderr));
    let other_cur = col("other", 0).unwrap_or(0.0);
    // One f32 copy of every trainable parameter — a size-independent yardstick
    // that scales with the fixture instead of a hard-coded byte count.
    let m_peak = col("optim_m", 1)
        .unwrap_or_else(|| panic!("no `optim_m` surface row in:\n{}", r.stderr));
    assert!(m_peak > 0.0, "optim_m never allocated — vacuous run:\n{}", r.stderr);

    // The widened f32 working views come from `srbf16_upload`. Untagged they
    // land on whatever is ambient there, which is `other`.
    assert!(
        weights_cur > other_cur,
        "`other` ({other_cur} B) holds more than `weights` ({weights_cur} B) \
         under bf16-sr — the widened f32 working views are untagged again \
         (measured pre-fix: weights 16.5 KB vs other 129 KB)\n{}",
        r.stderr
    );
    // The mirrors come from `srbf16_register`. Untagged they land in
    // `activations`, which drops resident `weights` to a small remainder.
    assert!(
        weights_cur >= 0.5 * m_peak,
        "resident `weights` ({weights_cur} B) is under half of one f32 \
         parameter copy ({m_peak} B) under bf16-sr — the authoritative bf16 \
         mirrors are being tagged as `activations` again (measured pre-fix: \
         16.5 KB against a 145.5 KB optim_m)\n{}",
        r.stderr
    );
}
