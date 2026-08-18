//! Tape-AD × DataLoader: the composition must RUN, LEARN, and AGREE with
//! source AD.
//!
//! Before the item-5 campaign this composition had never executed — every
//! committed loader-driven program was run with `--source-ad`, which masked
//! two independent bugs:
//!   1. codegen: the packing-registry stash (emitted only on the loader
//!      path) terminated the batch body block and left
//!      `state.current_block` pointing at it, so `compile_stmt`'s
//!      filled-block guard silently skipped the ENTIRE step body — the
//!      tape lowering then refused with the misleading "train step body
//!      must assign to a variable named 'loss'";
//!   2. runtime: the tape's MatMul/Fp8MatMul backward skipped the
//!      broadcast reduction, so a `[b,s,d] @ [d,v]` matmul returned a
//!      batch-shaped `[b,s,v]` dW that aborted far away, in the
//!      optimizer's in-place update ("dst len N != src len M").
//!
//! The agreement leg is the CPU-f64 miniature of roadmap item 5's 50M
//! AD differential: identical model/data/order in both modes, loss
//! trajectories compared step by step.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_fixture(extra: &[&str], tmp: &std::path::Path) -> (bool, Vec<f64>, String) {
    let root = workspace_root();
    let fixture = root.join("crates/nsl-cli/tests/fixtures/tape_loader_train.nsl");
    assert!(fixture.exists(), "fixture missing: {}", fixture.display());
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .args(extra)
        .arg(&fixture)
        .current_dir(tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let losses: Vec<f64> = stdout
        .lines()
        .skip_while(|l| !l.contains("LOSS_STREAM_BEGIN"))
        .take_while(|l| !l.contains("LOSS_STREAM_END"))
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    let complete = stdout.contains("TAPE_LOADER_TRAIN_COMPLETE");
    (out.status.success() && complete, losses, stderr)
}

#[test]
fn tape_loader_train_runs_learns_and_agrees_with_source_ad() {
    let tmp = tempfile::TempDir::new().unwrap();

    // ── leg 1: bare run = tape AD; must complete and learn ─────────────
    let (tape_ok, tape_losses, tape_stderr) = run_fixture(&[], tmp.path());
    assert!(
        tape_ok,
        "loader-driven train block failed under default tape AD:\n{tape_stderr}"
    );
    // 2 epochs × 16 batches = 32 micro-steps; on_step fires per micro-batch.
    assert!(
        tape_losses.len() >= 16,
        "expected a loss stream, got {} values (stderr:\n{tape_stderr})",
        tape_losses.len()
    );
    let first = tape_losses[0];
    let last = *tape_losses.last().unwrap();
    // Uniform baseline is ln(32) ≈ 3.466 — the stream must start near it
    // (a start far below means the labels leaked; far above means a
    // broken init) and improve materially by the end.
    assert!(
        (first - 32f64.ln()).abs() < 0.35,
        "first loss {first} is not near the uniform baseline ln(32)"
    );
    assert!(
        last.is_finite() && last < first - 0.2,
        "tape-AD loss did not decrease: first={first} last={last}"
    );

    // ── leg 2: source AD on the identical program/data/order ───────────
    let (src_ok, src_losses, src_stderr) = run_fixture(&["--source-ad"], tmp.path());
    assert!(src_ok, "--source-ad control failed:\n{src_stderr}");
    assert_eq!(
        tape_losses.len(),
        src_losses.len(),
        "the two AD modes saw different numbers of micro-steps"
    );

    // ── leg 3: the miniature AD differential ───────────────────────────
    // CPU f64, identical deterministic init and batch order: the two
    // modes compute the same math and land on the same trajectory. The
    // tolerance is loose enough for benign kernel-order drift and tight
    // enough that any structural gradient bug (a missed reduction, a
    // dropped term — the class this gate exists for) diverges the
    // trajectory by orders of magnitude more within a few steps.
    for (i, (t, s)) in tape_losses.iter().zip(src_losses.iter()).enumerate() {
        let denom = s.abs().max(1e-12);
        assert!(
            ((t - s) / denom).abs() < 1e-6,
            "AD modes diverged at micro-step {i}: tape={t} source={s}"
        );
    }
}
