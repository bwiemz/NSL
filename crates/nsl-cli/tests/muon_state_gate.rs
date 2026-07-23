//! P4 item 18 rung 2 gates — compressed Muon state (`--muon-state-dtype bf16`).
//!
//! The ladder (in order): f32 state (default, unchanged) → BF16 momentum
//! with an FP32 working buffer + counter-based SR store (THIS rung) →
//! blockwise 8-bit → 4-bit structural (both later rungs REFUSED loudly).
//!
//! e2e gates prove: the envelope actually fired (runtime banner), training
//! converges (first > 3.5 nats, final < 1.5 at Muon's spec lr over 4
//! epochs), the SR state stream is deterministic in the seed, and
//! bf16-momentum training tracks the f32-state baseline per step.

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOut {
    success: bool,
    stdout: String,
    stderr: String,
    losses: Vec<String>,
}

fn run_fixture(tag: &str, rewrites: &[(&str, &str)], extra: &[&str]) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_muonstate_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .unwrap();
    for (from, to) in rewrites {
        assert!(src.contains(from), "rewrite marker '{from}' missing in fixture");
        src = src.replace(from, to);
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--deterministic"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let losses = stdout
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
        .unwrap_or_default();
    RunOut { success: out.status.success(), stdout, stderr, losses }
}

const MUON_REWRITE: (&str, &str) = (
    "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
    "Muon(lr=0.02, momentum=0.95, nesterov=true, ns_steps=5, \
     weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
);
/// Muon needs its spec-default lr (0.02) and a few epochs to descend on the
/// FFN fixture — at the AdamW-tuned 0.002 / 6 steps it only wobbles (both
/// state dtypes identically), which would make a decrease assertion vacuous.
const EPOCHS_REWRITE: (&str, &str) = ("epochs=1", "epochs=4");
const CSLA_FLAGS: &[&str] = &["--checkpoint-blocks", "--layerwise-accum"];

/// bf16 momentum with AdamW refuses (the AdamW ladder is the CPDT/WGGO
/// moment-precision machinery).
#[test]
fn muon_state_refuses_adamw() {
    let out = run_fixture(
        "ref_adamw",
        &[("CSLA_SAVE_PATH", "ra.nslm")],
        &[CSLA_FLAGS[0], CSLA_FLAGS[1], "--muon-state-dtype", "bf16"],
    );
    assert!(!out.success, "adamw x bf16-state ran:\n{}", out.stdout);
    assert!(
        out.stderr
            .contains("--muon-state-dtype bf16 applies to the Muon optimizer only"),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// bf16 momentum without the CSLA schedule refuses (the envelope lives in
/// the group update).
#[test]
fn muon_state_refuses_without_layerwise() {
    let out = run_fixture(
        "ref_nocsla",
        &[MUON_REWRITE, ("CSLA_SAVE_PATH", "rn.nslm")],
        &["--muon-state-dtype", "bf16"],
    );
    assert!(!out.success, "bf16-state without CSLA ran:\n{}", out.stdout);
    assert!(
        out.stderr
            .contains("--muon-state-dtype bf16 requires --layerwise-accum"),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// Later ladder rungs refuse loudly with the ladder order in the message.
#[test]
fn muon_state_later_rungs_refuse() {
    for rung in ["int8-blockwise", "int4-structural"] {
        let out = run_fixture(
            &format!("ref_{}", &rung[..4]),
            &[MUON_REWRITE, ("CSLA_SAVE_PATH", "rr.nslm")],
            &[CSLA_FLAGS[0], CSLA_FLAGS[1], "--muon-state-dtype", rung],
        );
        assert!(!out.success, "{rung} ran:\n{}", out.stdout);
        assert!(
            out.stderr.contains("ladder rung not implemented"),
            "wrong refusal for {rung}:\n{}",
            out.stderr
        );
    }
}

/// Core e2e: Muon with bf16 momentum trains sanely on the CSLA schedule,
/// the SR envelope demonstrably fired, reruns are bit-deterministic, and
/// the loss tracks the f32-state baseline.
#[test]
#[ignore = "requires CUDA GPU"]
fn muon_state_bf16_e2e_gpu() {
    let gpu = ("# GPU_PLACEMENT", "m.to(cuda)");
    let mut args: Vec<&str> = CSLA_FLAGS.to_vec();
    args.extend_from_slice(&["--muon-state-dtype", "bf16", "--seed", "777"]);

    let q = run_fixture(
        "e2e_q",
        &[MUON_REWRITE, EPOCHS_REWRITE, gpu, ("CSLA_SAVE_PATH", "q.nslm")],
        &args,
    );
    assert!(q.success, "bf16-state run failed:\nstdout:\n{}\nstderr:\n{}", q.stdout, q.stderr);
    assert!(
        q.stderr.contains("[muon-state] bf16 momentum active"),
        "SR state envelope never fired:\n{}",
        q.stderr
    );
    let lq: Vec<f64> = q.losses.iter().map(|s| s.parse().unwrap()).collect();
    assert!(lq.len() >= 4, "loss stream too short: {lq:?}");
    assert!(lq.iter().all(|v| v.is_finite()), "non-finite loss: {lq:?}");
    assert!(
        *lq.first().unwrap() > 3.5 && *lq.last().unwrap() < 1.5,
        "bf16-state muon did not train (first={}, last={}): {lq:?}",
        lq.first().unwrap(),
        lq.last().unwrap()
    );

    // Determinism: identical seed → identical stream.
    let q2 = run_fixture(
        "e2e_q2",
        &[MUON_REWRITE, EPOCHS_REWRITE, gpu, ("CSLA_SAVE_PATH", "q2.nslm")],
        &args,
    );
    assert!(q2.success, "rerun failed:\n{}", q2.stderr);
    assert_eq!(q.losses, q2.losses, "same seed must reproduce bit-identically");

    // Baseline: f32 state, same everything else.
    let mut base_args: Vec<&str> = CSLA_FLAGS.to_vec();
    base_args.extend_from_slice(&["--seed", "777"]);
    let f = run_fixture(
        "e2e_f",
        &[MUON_REWRITE, EPOCHS_REWRITE, gpu, ("CSLA_SAVE_PATH", "f.nslm")],
        &base_args,
    );
    assert!(f.success, "f32-state baseline failed:\n{}", f.stderr);
    let lf: Vec<f64> = f.losses.iter().map(|s| s.parse().unwrap()).collect();
    assert_eq!(lq.len(), lf.len(), "loss stream length mismatch");
    // SR quantization noise on the momentum compounds over 4 epochs at
    // lr=0.02 — allow a loose envelope per step, plus require the baseline
    // to reach the same converged region.
    for (i, (a, b)) in lq.iter().zip(lf.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.6,
            "bf16-state diverged from f32-state at step {i}: {a} vs {b}\nbf16: {lq:?}\nf32:  {lf:?}"
        );
    }
    assert!(
        *lf.last().unwrap() < 1.5,
        "f32-state baseline did not converge: {lf:?}"
    );
    // And the state dtype actually changed something at SOME digit — a
    // bit-identical stream would mean the bf16 buffers were never really
    // in the loop (vacuous envelope).
    assert_ne!(
        q.losses, f.losses,
        "bf16-state stream is bit-identical to f32-state — envelope vacuous?"
    );
}
