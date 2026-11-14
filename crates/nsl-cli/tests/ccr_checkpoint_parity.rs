//! CCR P1.a — block-granular checkpoint parity gates (CCR paper section 5.4
//! correctness test 1: bit-exact save/recompute equivalence).
//!
//! `--checkpoint-blocks` frees each transformer block's interior activations
//! after the forward and recomputes them (same ops, same order, same inputs)
//! just before that block's adjoint runs. The recompute must therefore be
//! BIT-EXACT: the loss stream must match at full printed precision and the
//! saved model files must be byte-identical.
//!
//! Determinism guard: the bit-exact cases first run the BASELINE twice. If the
//! environment itself is not run-to-run deterministic (e.g. a GPU path with
//! atomic scatter ordering), the byte-compare gate cannot distinguish
//! checkpointing effects from ambient noise. On the CPU that is a failure:
//! the CPU path is deterministic and a baseline that is not means something
//! else is broken. On a GPU the case falls back to the band the production
//! gate uses, over the WHOLE loss curve: the checkpointed run may differ from
//! the baseline by no more than a few times what the baseline differs from
//! itself (`assert_within_noise_band`). It used to compare the first loss
//! only, which is a pure forward that checkpointing cannot affect.
//!
//! A2 (post-#374): three GPU strategies, from strongest to most production-real
//!   - `Det::EmbedCpu`  — embedding backward forced to the deterministic host
//!     scatter (`NSL_EMBEDDING_BWD_CPU=1`). Historical GPU gate.
//!   - `Det::Deterministic` — `--deterministic` (M46): the embedding backward
//!     runs the NEW deterministic GPU kernel (no atomics), exercising the real
//!     device path instead of bouncing to the host.
//!   - `Det::ProductionGpu` — the default atomicAdd path, validated with a
//!     loss-curve *tolerance* gate (`ccr_production_gpu_tolerance`): CCR must
//!     not diverge from a baseline re-run by more than the ambient
//!     nondeterminism band.

use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::prelude::*;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture_src() -> String {
    std::fs::read_to_string(
        repo_root().join("crates/nsl-cli/tests/fixtures/stage_c_packed_gqa.nsl"),
    )
    .expect("stage_c_packed_gqa.nsl fixture missing")
}

fn program(gpu: bool, save_path: &Path) -> String {
    let mut src = fixture_src();
    if gpu {
        assert!(src.contains("# GPU_PLACEMENT"));
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    assert!(src.contains("STAGE_C_SAVE_PATH"));
    src.replace(
        "STAGE_C_SAVE_PATH",
        &save_path.display().to_string().replace('\\', "/"),
    )
}

/// How the run pins down (or does not) the nondeterministic embedding backward.
#[derive(Clone, Copy, PartialEq)]
enum Det {
    /// Force the deterministic host scatter (`NSL_EMBEDDING_BWD_CPU=1`).
    EmbedCpu,
    /// `--deterministic` (M46): deterministic GPU embedding kernel, no atomics.
    Deterministic,
    /// Default production path (atomicAdd embedding + flash backward).
    ProductionGpu,
}

struct RunOutput {
    loss_stream: String,
    stderr: String,
    success: bool,
}

fn run_program(source: &str, tag: &str, cuda: bool, det: Det, extra_args: &[&str]) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_ccr_parity_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("ccr_parity.nsl");
    std::fs::write(&prog, source).unwrap();

    // The binary of this test's own profile and features, not `cargo run`:
    // that built a second, debug copy of the compiler and rebuilt it whenever
    // a source file had changed since.
    let mut cmd = Command::cargo_bin("nsl").expect("the nsl binary of this test build");
    cmd.arg("run").arg("--source-ad");
    if !cuda {
        cmd.args(["--target", "cpu"]);
    }
    if det == Det::Deterministic {
        cmd.arg("--deterministic");
    }
    cmd.args(extra_args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    if det == Det::EmbedCpu {
        // Deterministic embedding backward via the host scatter (the atomicAdd
        // GPU kernel's ordering is not reproducible).
        cmd.env("NSL_EMBEDDING_BWD_CPU", "1");
    }
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

/// Extract the first float from each loss line (e.g. `tensor([5.3214])` -> 5.3214).
///
/// A NaN or infinite loss (printed `NaN` / `inf` / `-inf`) is a FAILURE, not a
/// line to skip. The digit scanner below finds no number in such a line, so it
/// used to drop it and the comparison ran over the finite remainder.
fn parse_losses(stream: &str) -> Vec<f64> {
    stream
        .lines()
        .enumerate()
        .filter_map(|(i, l)| {
            let lower = l.to_ascii_lowercase();
            assert!(
                !lower.contains("nan") && !lower.contains("inf"),
                "non-finite loss at loss-stream line {i}: {l:?}"
            );
            let mut num = String::new();
            let mut seen_digit = false;
            for c in l.chars() {
                if c.is_ascii_digit() {
                    num.push(c);
                    seen_digit = true;
                } else if (c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+') && !num.is_empty()
                {
                    num.push(c);
                } else if c == '-' && num.is_empty() {
                    num.push(c);
                } else if seen_digit {
                    break;
                }
            }
            let v = num.parse::<f64>().ok()?;
            assert!(v.is_finite(), "non-finite loss at loss-stream line {i}: {l:?}");
            Some(v)
        })
        .collect()
}

fn parity_case(cuda: bool, det: Det, tag: &str) {
    let tmp = std::env::temp_dir().join(format!("nsl_ccr_saves_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("base_a.nslm");
    let save_b = tmp.join("base_b.nslm");
    let save_c = tmp.join("ckpt.nslm");

    // 1. Baseline determinism probe (two identical runs).
    let base_a = run_program(&program(cuda, &save_a), &format!("{tag}_a"), cuda, det, &[]);
    assert!(base_a.success, "baseline run A failed:\n{}", base_a.stderr);
    let base_b = run_program(&program(cuda, &save_b), &format!("{tag}_b"), cuda, det, &[]);
    assert!(base_b.success, "baseline run B failed:\n{}", base_b.stderr);
    let bytes_a = std::fs::read(&save_a).expect("baseline A model_save missing");
    let bytes_b = std::fs::read(&save_b).expect("baseline B model_save missing");
    let env_deterministic = bytes_a == bytes_b && base_a.loss_stream == base_b.loss_stream;

    // 2. Checkpointed run.
    let ckpt = run_program(
        &program(cuda, &save_c),
        &format!("{tag}_c"),
        cuda,
        det,
        &["--checkpoint-blocks"],
    );
    assert!(ckpt.success, "--checkpoint-blocks run failed:\n{}", ckpt.stderr);
    // The transform must actually have fired — a silent decline would make
    // this whole gate vacuous.
    assert!(
        !ckpt.stderr.contains("running without checkpointing"),
        "CCR declined on the 2-block fixture — segmentation regressed:\n{}",
        ckpt.stderr
    );
    let bytes_c = std::fs::read(&save_c).expect("checkpointed model_save missing");

    // Every loss must be finite: the gates below compare the streams as
    // text, and two NaN streams compare equal. `parse_losses` refuses one.
    for stream in [&base_a.loss_stream, &base_b.loss_stream, &ckpt.loss_stream] {
        parse_losses(stream);
    }

    if env_deterministic {
        assert_eq!(
            base_a.loss_stream, ckpt.loss_stream,
            "loss stream diverged under --checkpoint-blocks (must be bit-exact)"
        );
        assert!(
            bytes_a == bytes_c,
            "saved model bytes diverged under --checkpoint-blocks (must be bit-exact)"
        );
    } else {
        assert!(
            cuda,
            "the CPU baseline is not run-to-run deterministic ({tag}): two identical \
             runs gave different losses or weights, so this gate cannot test \
             checkpointing here and something else is wrong"
        );
        // Ambient nondeterminism on the device: the strongest honest claim
        // is that the checkpointed run is indistinguishable from a baseline
        // re-run, over the whole curve.
        eprintln!(
            "[ccr-parity] WARNING: baseline is not run-to-run deterministic \
             in this environment ({tag}); comparing the loss curves within the \
             baseline's own noise band instead of bit for bit"
        );
        if let Err(why) = assert_within_noise_band(
            &parse_losses(&base_a.loss_stream),
            &parse_losses(&base_b.loss_stream),
            &parse_losses(&ckpt.loss_stream),
        ) {
            panic!("{tag}: {why}");
        }
    }
}

/// The comparison for a run-to-run nondeterministic device: at every step
/// the checkpointed loss `c` may differ from baseline `a` by at most four
/// times the largest difference between the two baselines `a` and `b`, with
/// an absolute floor of 1e-2. All three curves must have the same length and
/// at least four steps. Returns the largest deviation and the tolerance.
fn assert_within_noise_band(a: &[f64], b: &[f64], c: &[f64]) -> Result<(f64, f64), String> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(format!(
            "loss curves differ in length: baselines {} and {}, checkpointed {}",
            a.len(),
            b.len(),
            c.len()
        ));
    }
    if a.len() < 4 {
        return Err(format!("expected several loss steps, got {}", a.len()));
    }
    let mut max_noise = 0.0f64;
    let mut max_dev = 0.0f64;
    for i in 0..a.len() {
        max_noise = max_noise.max((b[i] - a[i]).abs());
        max_dev = max_dev.max((c[i] - a[i]).abs());
    }
    let tol = (max_noise * 4.0).max(1e-2);
    if max_dev > tol {
        return Err(format!(
            "checkpointed loss diverged from baseline by {max_dev:.6} > tol {tol:.6} \
             (ambient noise {max_noise:.6}) — recompute is not faithful\n\
             base_a={a:?}\nckpt ={c:?}"
        ));
    }
    Ok((max_dev, tol))
}

/// The band comparison itself, on curves written out by hand: it must
/// accept a checkpointed curve inside the baselines' own noise and reject
/// one outside it, one of a different length, and one too short to judge.
#[test]
fn the_noise_band_accepts_noise_and_rejects_a_divergence() {
    let a = [5.0, 4.0, 3.0, 2.5, 2.2];
    let b = [5.0, 4.01, 3.02, 2.49, 2.21]; // noise up to 0.02 -> tol 0.08
    let inside = [5.0, 4.03, 2.95, 2.5, 2.27];
    let (dev, tol) = assert_within_noise_band(&a, &b, &inside).expect("inside the band");
    assert!((dev - 0.07).abs() < 1e-12 && (tol - 0.08).abs() < 1e-12, "{dev} {tol}");
    // The first loss equal and a later one off by 0.2: the old first-loss
    // comparison accepted exactly this.
    let outside = [5.0, 4.0, 3.0, 2.7, 2.2];
    assert!(assert_within_noise_band(&a, &b, &outside).is_err());
    // With identical baselines the floor applies: 1e-2.
    assert!(assert_within_noise_band(&a, &a, &[5.0, 4.0, 3.0, 2.5, 2.209]).is_ok());
    assert!(assert_within_noise_band(&a, &a, &[5.0, 4.0, 3.0, 2.5, 2.22]).is_err());
    assert!(assert_within_noise_band(&a, &b, &inside[..4]).is_err());
    assert!(assert_within_noise_band(&a[..3], &b[..3], &inside[..3]).is_err());
}

/// CPU gate — ungated, runs in the default suite.
#[test]
fn ccr_blocks_parity_on_cpu() {
    parity_case(false, Det::EmbedCpu, "cpu");
}

/// GPU gate, deterministic host embedding — the historical GPU parity path.
/// `cargo test -p nsl-cli --features cuda --test ccr_checkpoint_parity -- --ignored`
#[test]
#[ignore = "requires CUDA GPU"]
fn ccr_blocks_parity_on_gpu() {
    parity_case(true, Det::EmbedCpu, "gpu");
}

/// A2: GPU gate exercising the REAL device embedding backward via the new
/// deterministic GPU kernel (`--deterministic`), not the host bounce. Proves
/// CCR recompute is bit-exact with the embedding gradient computed on-device
/// (or, if other GPU ops keep the environment noisy, falls back gracefully).
#[test]
#[ignore = "requires CUDA GPU"]
fn ccr_blocks_parity_on_gpu_deterministic() {
    parity_case(true, Det::Deterministic, "gpu_det");
}

/// A2: production-path validation. On the DEFAULT GPU path (atomicAdd embedding
/// + flash backward), the run is not bit-reproducible, so bit-exact CCR parity
/// is impossible by construction. Instead assert CCR does not DIVERGE: the
/// checkpointed loss curve must track a baseline re-run within the ambient
/// nondeterminism band (a few x the baseline-vs-baseline noise floor). This is
/// the loss-curve tolerance gate the roadmap calls for.
#[test]
#[ignore = "requires CUDA GPU"]
fn ccr_production_gpu_tolerance() {
    let tmp = std::env::temp_dir().join(format!("nsl_ccr_tol_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let det = Det::ProductionGpu;

    let a = run_program(
        &program(true, &tmp.join("a.nslm")),
        "tol_a",
        true,
        det,
        &[],
    );
    assert!(a.success, "baseline A failed:\n{}", a.stderr);
    let b = run_program(
        &program(true, &tmp.join("b.nslm")),
        "tol_b",
        true,
        det,
        &[],
    );
    assert!(b.success, "baseline B failed:\n{}", b.stderr);
    let c = run_program(
        &program(true, &tmp.join("c.nslm")),
        "tol_c",
        true,
        det,
        &["--checkpoint-blocks"],
    );
    assert!(c.success, "checkpointed run failed:\n{}", c.stderr);
    assert!(
        !c.stderr.contains("running without checkpointing"),
        "CCR declined on the 2-block fixture:\n{}",
        c.stderr
    );

    let la = parse_losses(&a.loss_stream);
    let lb = parse_losses(&b.loss_stream);
    let lc = parse_losses(&c.loss_stream);
    // Per-step: the checkpointed deviation from baseline A must stay within the
    // ambient noise (baseline B vs A) plus a small absolute floor. If CCR had a
    // real recompute bug, the checkpointed curve would drift far outside this
    // band even on a nondeterministic device.
    let (max_ckpt_dev, tol) = match assert_within_noise_band(&la, &lb, &lc) {
        Ok(band) => band,
        Err(why) => panic!("production path: {why}"),
    };
    eprintln!(
        "[ccr-tol] production-path CCR OK: max ckpt deviation {max_ckpt_dev:.6} within tol {tol:.6}"
    );
}
