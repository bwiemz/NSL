//! Muon perf campaign gates: `--muon-batch-ns` (shape-grouped batched
//! Newton-Schulz) and `--muon-resident-momentum` (device-resident Muon
//! momentum under offload).
//!
//! Contracts:
//! - batched training TRACKS the sequential path (tolerance — the batched
//!   GEMM order + TF32 are intentionally not bit-exact) and actually trains;
//! - the batched run is bit-DETERMINISTIC across identical invocations;
//! - resident momentum is bit-IDENTICAL to plain offload (it changes where
//!   m lives, not any computed value);
//! - every unsupported combination refuses loudly at compile time.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOut {
    ok: bool,
    stdout: String,
    stderr: String,
}

/// Muon-ized cuda-graph fixture: pure-tensor MLP, all rank-2 hidden weights
/// (every param Muon-routed), inputs device-resident before the train block.
fn run_muon_mlp(tag: &str, extra: &[&str]) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_muonbatch_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/cuda_graph_gate.nsl"),
    )
    .unwrap();
    src = src.replace(
        "optimizer: AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
        "optimizer: Muon(lr=0.005, momentum=0.95, nesterov=true, ns_steps=5, \
         weight_decay=0.0, adamw_lr=0.002)",
    );
    src = src.replace(
        "# GPU_PLACEMENT",
        "m.to(cuda)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)",
    );
    src = src.replace("m.forward_train(x)", "m.forward_train(xg)");
    src = src.replace("l1_loss(pred, y)", "l1_loss(pred, yg)");
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--deterministic", "--source-ad"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    RunOut {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn losses(stdout: &str) -> Vec<f64> {
    stdout
        .split("LOSS_STREAM_BEGIN")
        .nth(1)
        .and_then(|r| r.split("LOSS_STREAM_END").next())
        .unwrap_or("")
        .lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix("tensor([")?
                .strip_suffix("])")?
                .parse::<f64>()
                .ok()
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn batch_ns_tracks_sequential_and_trains() {
    let seq = run_muon_mlp("seq", &[]);
    assert!(seq.ok, "sequential run failed:\n{}", seq.stderr);
    let bat = run_muon_mlp("bat", &["--muon-batch-ns"]);
    assert!(bat.ok, "batched run failed:\n{}", bat.stderr);

    let (ls, lb) = (losses(&seq.stdout), losses(&bat.stdout));
    assert!(ls.len() >= 8, "sequential produced {} losses", ls.len());
    assert_eq!(ls.len(), lb.len(), "loss stream lengths differ");
    // Trains: the loss must actually drop on both paths (anti-vacuity).
    assert!(
        ls[0] > *ls.last().unwrap() && lb[0] > *lb.last().unwrap(),
        "loss did not decrease: seq {:?} bat {:?}",
        (ls[0], ls.last()),
        (lb[0], lb.last())
    );
    // First loss precedes any optimizer step — must be bit-identical.
    assert_eq!(
        ls[0].to_bits(),
        lb[0].to_bits(),
        "pre-step loss diverged — something besides the optimizer changed"
    );
    // Tracks within tolerance thereafter (TF32 + batched GEMM order).
    for (i, (a, b)) in ls.iter().zip(&lb).enumerate() {
        assert!(
            (a - b).abs() < 5e-3 * a.abs().max(1.0),
            "step {i}: sequential {a} vs batched {b}"
        );
    }
    // Anti-vacuity for the batch path itself: all 6 params routed to Muon.
    assert!(
        bat.stderr.contains("0 AdamW-routed, 6 Muon"),
        "routing table changed — batch path may not have been exercised:\n{}",
        bat.stderr
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn batch_ns_is_bit_deterministic() {
    let a = run_muon_mlp("det_a", &["--muon-batch-ns"]);
    let b = run_muon_mlp("det_b", &["--muon-batch-ns"]);
    assert!(a.ok && b.ok, "runs failed:\n{}\n{}", a.stderr, b.stderr);
    assert_eq!(
        losses(&a.stdout),
        losses(&b.stdout),
        "batched muon run is not deterministic"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn resident_momentum_bit_identical_to_offload() {
    let off = run_muon_mlp("off", &["--optim-state-offload"]);
    assert!(off.ok, "offload run failed:\n{}", off.stderr);
    let res = run_muon_mlp(
        "res",
        &["--optim-state-offload", "--muon-resident-momentum"],
    );
    assert!(res.ok, "resident run failed:\n{}", res.stderr);
    let (lo, lr) = (losses(&off.stdout), losses(&res.stdout));
    assert!(lo.len() >= 8, "offload produced {} losses", lo.len());
    assert_eq!(
        lo.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        lr.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "resident momentum changed training values — it must only change \
         where m lives"
    );
}

// ── Refusals (compile-time, no GPU needed) ────────────────────────────────

fn expect_refusal(tag: &str, extra: &[&str], needle: &str) {
    let out = run_muon_mlp(tag, extra);
    assert!(!out.ok, "expected refusal for {extra:?}, but the run succeeded");
    assert!(
        out.stderr.contains(needle),
        "refusal message for {extra:?} missing {needle:?}:\n{}",
        out.stderr
    );
}

#[test]
fn batch_ns_refuses_layerwise_accum() {
    expect_refusal(
        "ref_csla",
        &[
            "--muon-batch-ns",
            "--layerwise-accum",
            "--checkpoint-blocks",
        ],
        "--muon-batch-ns does not compose with --layerwise-accum",
    );
}

#[test]
fn batch_ns_refuses_offload() {
    expect_refusal(
        "ref_off",
        &["--muon-batch-ns", "--optim-state-offload"],
        "--muon-batch-ns does not compose with --optim-state-offload",
    );
}

#[test]
fn resident_momentum_refuses_without_offload() {
    expect_refusal(
        "ref_resoff",
        &["--muon-resident-momentum"],
        "--muon-resident-momentum is only meaningful with",
    );
}

#[test]
fn batch_ns_refuses_non_muon_optimizer() {
    // Plain fixture keeps AdamW — the flag must refuse.
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_muonbatch_nonmuon_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/cuda_graph_gate.nsl"),
    )
    .unwrap()
    .replace("# GPU_PLACEMENT", "");
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--deterministic", "--source-ad", "--muon-batch-ns"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(!out.status.success(), "expected non-muon refusal");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--muon-batch-ns requires the muon optimizer"),
        "missing refusal:\n{stderr}"
    );
}
