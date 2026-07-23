//! Regression gate for the mse_loss per-step tensor leak (bugs.md 2026-07-22,
//! fixed 2026-07-23): `nsl_mse_backward`'s internal sub -> mul_scalar
//! (RELINQUISH_A) chain handed back a grad carrying TWO refs, stranding the
//! (pred - target) block every training step. live_blocks grew +1/step and the
//! leaked address march defeated cuda-graph digest stability.
//!
//! The gate trains the cuda-graph MLP fixture with mse_loss under --source-ad
//! and asserts the caching allocator's live_blocks count is EXACTLY flat
//! across steps. The A4 "bounded churn" gates tolerate small leaks by design;
//! this fixture is leak-free, so it pins exact flatness per the bugs.md
//! prevention note.
//!
//! NOTE: only the source-AD path is gated. The tape-AD fallback has its own
//! (pre-existing, open) per-step leak tracked separately in bugs.md.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn mse_loss_source_ad_live_blocks_exactly_flat() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_mseleak_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Reuse the cuda-graph fixture (pure-tensor MLP, all inputs device-resident)
    // with its loss swapped to mse_loss and the GPU placement marker expanded.
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/cuda_graph_gate.nsl"),
    )
    .unwrap();
    src = src.replace("l1_loss", "mse_loss");
    src = src.replace(
        "# GPU_PLACEMENT",
        "m.to(cuda)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)",
    );
    src = src.replace("m.forward_train(x)", "m.forward_train(xg)");
    src = src.replace("mse_loss(pred, y)", "mse_loss(pred, yg)");
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--deterministic", "--source-ad"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_DEBUG_MEM_ALL", "1")
        .output()
        .expect("spawn nsl run");
    assert!(
        out.status.success(),
        "run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // Anti-vacuity: training actually ran on the source-AD path.
    assert!(
        stdout.contains("LOSS_STREAM_BEGIN") && stdout.contains("LOSS_STREAM_END"),
        "loss stream markers missing — fixture did not train"
    );
    assert!(
        stderr.contains("Using source-to-source AD"),
        "fixture silently fell off the source-AD path:\n{stderr}"
    );

    // Collect live_blocks per step. Note (review): the runtime emits TWO
    // [gpu-mem] reports per micro-step — one pre-increment at step start and
    // one post-increment after cleanup — so a given step label mixes "end of
    // k-1" and "start of k". Both are allocator-quiescent points with equal
    // counts in steady state, which is exactly what exact-flatness asserts;
    // if the step-counter increment ever moves relative to those calls,
    // revisit this parse.
    let mut per_step: Vec<(i64, i64)> = Vec::new();
    for line in stderr.lines() {
        let Some(rest) = line.strip_prefix("[gpu-mem] step=") else {
            continue;
        };
        let step: i64 = rest
            .split_whitespace()
            .next()
            .and_then(|s| s.parse().ok())
            .expect("malformed step field");
        let lb: i64 = rest
            .split("live_blocks=")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .expect("malformed live_blocks field");
        per_step.push((step, lb));
    }
    let max_step = per_step.iter().map(|(s, _)| *s).max().unwrap_or(0);
    assert!(
        max_step >= 4,
        "expected >=4 training steps of [gpu-mem] reports, got max step {max_step}"
    );

    // Exact flatness from step 1 on (step 0 may include warmup allocations).
    let baseline = per_step
        .iter()
        .find(|(s, _)| *s == 1)
        .map(|(_, lb)| *lb)
        .expect("no step-1 report");
    for (step, lb) in per_step.iter().filter(|(s, _)| *s >= 1) {
        assert_eq!(
            *lb, baseline,
            "live_blocks not flat: step {step} has {lb}, step 1 had {baseline} \
             (a +1/step march is the mse (pred-target) leak signature)"
        );
    }
}
