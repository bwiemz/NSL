//! Milestone C · p3 — stream-ordered execution gate.
//!
//! The runtime launches every kernel on the NULL stream (self-ordering) and
//! reads to host through the synchronous `cuMemcpyDtoH_v2` (a NULL-stream
//! barrier), so the per-op `cuCtxSynchronize` calls are redundant for
//! correctness. p3 gates them behind the existing `NSL_CUDA_SYNC` toggle so the
//! default path runs stream-ordered (async) and overlaps CPU launch with GPU
//! execution.
//!
//! The correctness proof is a differential test: the SAME training run under
//!   * `NSL_CUDA_SYNC=1` (eager: sync after every op — the old behavior), and
//!   * default            (stream-ordered: syncs removed)
//! must produce a **bit-identical** loss stream. If gating a sync exposed a
//! read-before-complete race, the two streams would diverge. Under
//! `--deterministic` the GPU path is bit-reproducible (A2), so the strongest
//! claim applies: exact equality at every step, not a tolerance band.
//!
//! The gate also records wall-clock for both modes so the speedup is visible in
//! the test log (not asserted — it is hardware/scheduler dependent).
//!
//! GPU-only, `#[ignore]`. Run:
//!   cargo test -p nsl-cli --features cuda --test stream_ordering_gpu_gate \
//!     -- --ignored --test-threads=1

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Run the drift fixture; `eager` sets `NSL_CUDA_SYNC=1`. Returns (loss_stream,
/// wall_clock, success, stderr).
fn run(eager: bool) -> (String, std::time::Duration, bool, String) {
    let root = repo_root();
    let fixture = root.join("crates/nsl-cli/tests/fixtures/long_run_drift.nsl");
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args([
            "-p",
            "nsl-cli",
            "--",
            "run",
            "--source-ad",
            "--deterministic",
        ])
        .arg(&fixture)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    if eager {
        cmd.env("NSL_CUDA_SYNC", "1");
    }
    let t0 = Instant::now();
    let out = cmd.output().expect("spawn nsl run");
    let dt = t0.elapsed();

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut loss = String::new();
    let mut inside = false;
    for line in stdout.lines() {
        match line.trim() {
            "LOSS_STREAM_BEGIN" => inside = true,
            "LOSS_STREAM_END" => inside = false,
            l if inside => {
                loss.push_str(l);
                loss.push('\n');
            }
            _ => {}
        }
    }
    (
        loss,
        dt,
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
#[ignore = "requires CUDA GPU (2 training runs)"]
fn stream_ordered_matches_eager_bit_exact() {
    // Eager (old behavior) first, so a cold compile doesn't skew the async time.
    let (loss_eager, t_eager, ok_eager, err_eager) = run(true);
    assert!(ok_eager, "eager (NSL_CUDA_SYNC=1) run failed:\n{err_eager}");
    let (loss_async, t_async, ok_async, err_async) = run(false);
    assert!(ok_async, "stream-ordered (default) run failed:\n{err_async}");

    let n = loss_eager.lines().filter(|l| l.contains("tensor(")).count();
    assert!(n >= 90, "expected ~96 losses, got {n}");

    assert_eq!(
        loss_eager, loss_async,
        "stream-ordered loss diverged from eager over {n} steps — gating a per-op \
         sync exposed a read-before-complete race. Re-run with NSL_CUDA_SYNC=1 to \
         confirm, then restore the sync at the offending host-read site."
    );

    eprintln!(
        "[stream-order] OK: {n} steps bit-exact. eager {:?} vs stream-ordered {:?} \
         ({:.2}x)",
        t_eager,
        t_async,
        t_eager.as_secs_f64() / t_async.as_secs_f64().max(1e-9)
    );
}
