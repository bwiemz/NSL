//! The step boundary keeps the allocator's pool in steady state, and a later
//! phase that needs that memory still gets it.
//!
//! `nsl_gpu_drain_cache` runs after every training micro-batch. Until November
//! 2026 it returned every idle transient segment to the driver each time, so
//! the next micro-batch re-allocated them: ~95 `cuMemFree` + ~95 `cuMemAlloc`
//! per micro-batch at Coder-1B, 24 ms of idle GPU each, 1.02-1.05x of
//! throughput (`models/benchmarks/NSL_VS_PYTORCH_2026_11.md`). It now releases
//! only a pool that holds markedly more than the step needed
//! (`CachingAllocator::drain_if_bloated`). Two gates:
//!
//! 1. Steady-state training allocates nothing from the driver and frees
//!    nothing to it after its first steps, at a flat reservation. (The old
//!    per-step drain fails this: its `drv_frees` grows every step.)
//! 2. The case the per-step drain protected: after the train block the cache
//!    still holds the training working set, and a tensor larger than any of its
//!    segments must still be allocatable under a memory limit that only fits
//!    it once the idle cache is released (allocation failure -> drain ->
//!    retry), with the same result as without the limit.
//!
//! GPU-only. Run:
//!   cargo test -p nsl-cli --features cuda --test step_boundary_drain_gpu_gate \
//!     -- --ignored --test-threads=1
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

fn gpu_present() -> bool {
    Command::new("nvidia-smi").arg("-L").output().map(|o| o.status.success()).unwrap_or(false)
}

struct Run {
    stdout: String,
    stderr: String,
    success: bool,
}

fn run(fixture: &str, extra_env: &[(&str, String)]) -> Run {
    let root = repo_root();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad"])
        .arg(root.join("crates/nsl-cli/tests/fixtures").join(fixture))
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_DEBUG_MEM_ALL", "1")
        .env_remove("NSL_SKIP_GPU_DRAIN")
        .env_remove("NSL_GPU_MEM_LIMIT");
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl");
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        success: out.status.success(),
    }
}

/// One `[gpu-mem] step=` report: the line itself and its continuation.
#[derive(Debug, Clone, Copy)]
struct StepMem {
    step: u64,
    reserved_mb: u64,
    drv_allocs: u64,
    drv_frees: u64,
    persistent_mb: u64,
    transient_mb: u64,
}

fn field(line: &str, key: &str) -> Option<u64> {
    let rest = &line[line.find(key)? + key.len()..];
    rest.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
}

fn step_reports(stderr: &str) -> Vec<StepMem> {
    let lines: Vec<&str> = stderr.lines().collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if !l.contains("[gpu-mem] step=") {
            continue;
        }
        let next = lines.get(i + 1).copied().unwrap_or("");
        if let (Some(step), Some(reserved_mb), Some(drv_allocs), Some(drv_frees)) =
            (field(l, "step="), field(l, "reserved="), field(l, "drv_allocs="), field(l, "drv_frees="))
        {
            out.push(StepMem {
                step,
                reserved_mb,
                drv_allocs,
                drv_frees,
                persistent_mb: field(next, "persistent=").unwrap_or(0),
                transient_mb: field(next, "transient=").unwrap_or(0),
            });
        }
    }
    out
}

/// The `[gpu-mem]` report printed after the LAST step boundary.
fn last_report(stderr: &str) -> StepMem {
    *step_reports(stderr).last().expect("no [gpu-mem] step report")
}

#[test]
#[ignore = "requires CUDA GPU"]
fn steady_training_neither_frees_nor_allocates_after_warmup() {
    if !gpu_present() {
        return;
    }
    let r = run("drain_phase_change.nsl", &[]);
    assert!(r.success, "fixture failed:\n{}", r.stderr);
    let reports = step_reports(&r.stderr);
    assert!(reports.len() >= 8, "expected a report per step, got {reports:?}");
    // Steps 0-2 are warmup: the first optimizer step (grad_accumulation=2)
    // allocates the AdamW moments.
    let steady: Vec<&StepMem> = reports.iter().filter(|m| m.step >= 3).collect();
    let first = steady[0];
    for m in &steady {
        assert_eq!(
            (m.drv_allocs, m.drv_frees),
            (first.drv_allocs, first.drv_frees),
            "step {}: the step boundary returned memory to the driver and re-allocated it \
             (cumulative driver allocs/frees moved) — the per-step drain is back.\n{reports:?}",
            m.step
        );
        assert_eq!(m.reserved_mb, first.reserved_mb, "step {}: reservation moved\n{reports:?}", m.step);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_later_phase_gets_the_idle_training_cache_back() {
    if !gpu_present() {
        return;
    }
    let free = run("drain_phase_change.nsl", &[]);
    assert!(free.success && free.stdout.contains("BIG_DONE"), "unconstrained run failed:\n{}", free.stderr);
    let m = last_report(&free.stderr);
    let big_mb = 384u64;
    // The point of the fixture: the training cache it leaves is big enough
    // that, under the limit below, the big tensor fits only once it goes.
    assert!(m.transient_mb >= 128, "training cache too small to make the case: {m:?}");
    // Persistent + the big tensor + half the idle cache: the cache cannot all
    // stay, and everything but it fits.
    let limit_mb = m.persistent_mb + big_mb + m.transient_mb / 2;
    let capped = run("drain_phase_change.nsl", &[("NSL_GPU_MEM_LIMIT", format!("{limit_mb}M"))]);
    assert!(
        capped.success && capped.stdout.contains("BIG_DONE"),
        "with NSL_GPU_MEM_LIMIT={limit_mb}M (persistent {} + big {big_mb} + half the {} MB \
         training cache) the post-training allocation must drain the idle cache and \
         succeed:\n{}",
        m.persistent_mb,
        m.transient_mb,
        capped.stderr
    );
    // Same program, same numbers: the loss stream and the big tensor's sum.
    let tail = |s: &str| s.lines().filter(|l| !l.starts_with('[')).collect::<Vec<_>>().join("\n");
    assert_eq!(tail(&free.stdout), tail(&capped.stdout), "the memory limit changed the program's output");
}
