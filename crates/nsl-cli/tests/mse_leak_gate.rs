//! Regression gates for per-step tensor leaks in train blocks (bugs.md
//! 2026-07-22/23).
//!
//! History:
//!   - The FBIP RELINQUISH extra-ref leak (#411) made mse_loss strand its
//!     (pred - target) block every step on the source-AD path.
//!   - The tape-AD path stranded FIVE blocks per step: GPU op arms never
//!     recorded on the tape (so ALL param grads were silently zero — training
//!     moved on weight decay alone), the explicit-return path never freed
//!     function locals, and `return <call>()` results were double-owned by
//!     the conservative Unknown-retain. All three root-fixed 2026-07-23.
//!
//! Both gates train the cuda-graph MLP fixture with mse_loss and assert the
//! caching allocator's live_blocks count is EXACTLY flat once the one-time
//! optimizer-state allocation (first accumulation boundary, step 2 with
//! grad_accumulation=2) has happened. The tape gate additionally asserts the
//! loss actually descends — the zero-grad failure mode kept live_blocks
//! plausible while "training" was weight-decay-only drift.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunResult {
    per_step: Vec<(i64, i64)>,
    losses: Vec<f64>,
    stderr: String,
}

fn run_mse_fixture(extra_args: &[&str]) -> RunResult {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!(
        "nsl_mseleak_{}_{}",
        std::process::id(),
        extra_args.len()
    ));
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

    let mut args = vec!["run", "--deterministic"];
    args.extend_from_slice(extra_args);
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(&args)
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
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();

    // Anti-vacuity: training actually ran.
    assert!(
        stdout.contains("LOSS_STREAM_BEGIN") && stdout.contains("LOSS_STREAM_END"),
        "loss stream markers missing — fixture did not train"
    );

    // Parse the loss stream (`tensor([v])` on GPU, bare scalars on CPU).
    let mut losses = Vec::new();
    for line in stdout.lines() {
        let l = line.trim();
        let val = if let Some(rest) = l.strip_prefix("tensor([") {
            rest.trim_end_matches("])").parse::<f64>().ok()
        } else {
            l.parse::<f64>().ok()
        };
        if let Some(v) = val {
            losses.push(v);
        }
    }

    // Collect live_blocks per step. The runtime emits TWO [gpu-mem] reports
    // per micro-step (pre-increment at step start, post-increment after
    // cleanup); both are allocator-quiescent points with equal counts in
    // steady state, which is exactly what exact-flatness asserts.
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
    RunResult {
        per_step,
        losses,
        stderr,
    }
}

/// Assert live_blocks is exactly flat from `baseline_step` on. Steps before
/// that may include warmup allocations (fused-AdamW optimizer state lands at
/// the FIRST accumulation boundary — step 2 with grad_accumulation=2).
fn assert_flat_from(r: &RunResult, baseline_step: i64, tag: &str) {
    let max_step = r.per_step.iter().map(|(s, _)| *s).max().unwrap_or(0);
    assert!(
        max_step >= baseline_step + 3,
        "[{tag}] expected >= {} training steps of [gpu-mem] reports, got max step {max_step}",
        baseline_step + 3
    );
    let baseline = r
        .per_step
        .iter()
        .find(|(s, _)| *s == baseline_step)
        .map(|(_, lb)| *lb)
        .unwrap_or_else(|| panic!("[{tag}] no step-{baseline_step} report"));
    for (step, lb) in r.per_step.iter().filter(|(s, _)| *s >= baseline_step) {
        assert_eq!(
            *lb, baseline,
            "[{tag}] live_blocks not flat: step {step} has {lb}, step {baseline_step} had \
             {baseline} (a per-step march is a train-loop tensor leak)"
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn mse_loss_source_ad_live_blocks_exactly_flat() {
    let r = run_mse_fixture(&["--source-ad"]);
    assert!(
        r.stderr.contains("Using source-to-source AD"),
        "fixture silently fell off the source-AD path:\n{}",
        r.stderr
    );
    assert_flat_from(&r, 2, "source-ad");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn mse_loss_tape_ad_live_blocks_flat_and_grads_flow() {
    let r = run_mse_fixture(&[]);
    assert_flat_from(&r, 2, "tape-ad");

    // Grad-flow: the historical failure mode was every GPU op skipping tape
    // recording, so all param grads were zeros and the loss only drifted at
    // weight-decay scale (~0.02%/step). Real gradients drop this fixture's
    // loss by >80% over 4 optimizer steps.
    assert!(
        r.losses.len() >= 4,
        "expected >=4 parsed losses, got {:?}",
        r.losses
    );
    let first = r.losses[0];
    let last = *r.losses.last().unwrap();
    assert!(
        last < first * 0.5,
        "tape-AD loss did not descend (first={first}, last={last}) — \
         gradients are not flowing (zero-grad tape regression?)"
    );
    assert!(
        !r.stderr.contains("[tape-ad] FATAL"),
        "tape backward reported a disconnected graph:\n{}",
        r.stderr
    );
}
