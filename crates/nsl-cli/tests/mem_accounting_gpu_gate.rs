//! A1 unified-accounting GPU gate + A4 peak/allocation-count regression gate.
//!
//! Runs `crates/nsl-cli/tests/fixtures/mem_accounting_gate.nsl` (a small GQA
//! transformer moved to GPU with `m.to(cuda)`, then trained a few steps) and
//! asserts:
//!
//!   1. (A1) Model weights are attributed to the `weights` surface, not
//!      `other` — the fix for the #374 LOW finding. The fixture prints
//!      `WEIGHTS_PEAK` / `OTHER_PEAK` via the new in-process numeric getters.
//!   2. (A4) The global peak VRAM is bounded (catches a blowup regression).
//!   3. (A4) Steady-state allocation is stable: reserved segments do not grow
//!      across steps (no leak), and the per-step driver-alloc churn stays
//!      bounded (no accumulating per-step allocation).
//!
//! GPU-only, `#[ignore]` by default (like the other `*_gpu_e2e` tests). Run:
//!   cargo test -p nsl-cli --features cuda --test mem_accounting_gpu_gate \
//!     -- --ignored --test-threads=1
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // crates/nsl-cli
    p.pop(); // crates
    p.pop(); // repo root
    p
}

/// Skip (don't fail) when no GPU is present.
fn gpu_present() -> bool {
    Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

struct RunOutput {
    stdout: String,
    stderr: String,
    success: bool,
}

fn run_fixture() -> RunOutput {
    let root = repo_root();
    let fixture = root.join("crates/nsl-cli/tests/fixtures/mem_accounting_gate.nsl");
    // The cuda-built `nsl` binary (this test only compiles under --features
    // cuda, so CARGO_BIN_EXE_nsl points at the cuda build).
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad"])
        .arg(&fixture)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        // Deterministic embedding backward on GPU (atomicAdd scatter otherwise
        // introduces run-to-run nondeterminism; not required for the memory
        // assertions but keeps the run clean).
        .env("NSL_EMBEDDING_BWD_CPU", "1")
        // Emit the per-step `[gpu-mem]` diagnostic for every step, not just the
        // first few, so the steady-state stability check has data.
        .env("NSL_DEBUG_MEM_ALL", "1")
        .output()
        .expect("failed to spawn nsl");
    RunOutput {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        success: out.status.success(),
    }
}

/// The fixture prints a label line followed by the integer value line.
fn marker_value(stdout: &str, label: &str) -> Option<u64> {
    let lines: Vec<&str> = stdout.lines().map(|l| l.trim()).collect();
    for i in 0..lines.len() {
        if lines[i] == label {
            // Next non-empty line is the value.
            for next in &lines[i + 1..] {
                if !next.is_empty() {
                    return next.parse::<u64>().ok();
                }
            }
        }
    }
    None
}

/// Extract `(step, drv_allocs, reserved_mb)` from each `[gpu-mem] step=...`
/// summary line in stderr.
fn parse_gpu_mem_lines(stderr: &str) -> Vec<(u64, u64, u64)> {
    let field = |line: &str, key: &str| -> Option<u64> {
        // key like "drv_allocs=" -> parse the following digits.
        let idx = line.find(key)?;
        let rest = &line[idx + key.len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse::<u64>().ok()
    };
    stderr
        .lines()
        .filter(|l| l.contains("[gpu-mem] step="))
        .filter_map(|l| {
            Some((
                field(l, "step=")?,
                field(l, "drv_allocs=")?,
                field(l, "reserved=")?,
            ))
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn mem_accounting_weights_attribution_and_peak_gate() {
    if !gpu_present() {
        eprintln!("skipping: no GPU present");
        return;
    }
    let run = run_fixture();
    assert!(
        run.success,
        "fixture run failed.\nSTDOUT:\n{}\nSTDERR:\n{}",
        run.stdout, run.stderr
    );

    let weights_peak = marker_value(&run.stdout, "WEIGHTS_PEAK")
        .unwrap_or_else(|| panic!("no WEIGHTS_PEAK marker.\nSTDOUT:\n{}", run.stdout));
    let other_peak = marker_value(&run.stdout, "OTHER_PEAK")
        .unwrap_or_else(|| panic!("no OTHER_PEAK marker.\nSTDOUT:\n{}", run.stdout));
    let global_peak = marker_value(&run.stdout, "PEAK")
        .unwrap_or_else(|| panic!("no PEAK marker.\nSTDOUT:\n{}", run.stdout));

    // (A1) The #374 finding: weights must be attributed to `weights`, not
    // `other`. Before the Weights bracket around `model.to(device)`, all
    // ~1.9 MB of weights showed under `other`.
    assert!(
        weights_peak > 512 * 1024,
        "weights surface peak ({weights_peak} B) should hold the model's ~1.9 MB of \
         weights — if this is ~0, the Weights bracket around model.to(device) regressed"
    );
    assert!(
        other_peak <= weights_peak / 4,
        "weights ({weights_peak} B) must dominate `other` ({other_peak} B) — a large \
         `other` means weight allocations leaked back into the untagged surface"
    );

    // (A4) Peak VRAM regression bound: the model peaked at ~25 MB. A blowup
    // (e.g. the offload/CCR composition silently double-allocating) would
    // blow past a generous 256 MB ceiling.
    assert!(global_peak >= weights_peak, "global peak must be >= weights peak");
    assert!(
        global_peak < 256 * 1024 * 1024,
        "global peak VRAM ({global_peak} B) exceeded the 256 MB regression ceiling for \
         this tiny fixture — a peak regression"
    );

    // (A4) Steady-state allocation stability. Reserved segments must not grow
    // across steps after warmup (a leak), and per-step driver-alloc churn must
    // stay bounded (no accumulating per-step allocation).
    let mem = parse_gpu_mem_lines(&run.stderr);
    assert!(
        mem.len() >= 4,
        "expected several [gpu-mem] step lines, got {}:\n{}",
        mem.len(),
        run.stderr
    );
    // Collapse to the max drv_allocs / reserved observed per step number.
    use std::collections::BTreeMap;
    let mut by_step: BTreeMap<u64, (u64, u64)> = BTreeMap::new();
    for (step, allocs, reserved) in &mem {
        let e = by_step.entry(*step).or_insert((0, 0));
        e.0 = e.0.max(*allocs);
        e.1 = e.1.max(*reserved);
    }
    let steps: Vec<(u64, u64, u64)> = by_step.iter().map(|(s, (a, r))| (*s, *a, *r)).collect();

    // Reserved segments constant after step 0 (warmup) — no leak.
    let post_warmup: Vec<u64> = steps.iter().skip(1).map(|(_, _, r)| *r).collect();
    if let (Some(min_r), Some(max_r)) =
        (post_warmup.iter().min().copied(), post_warmup.iter().max().copied())
    {
        assert!(
            max_r - min_r <= 2,
            "reserved VRAM grew across steps ({min_r} MB → {max_r} MB) — a per-step leak.\nsteps={steps:?}"
        );
    }

    // Per-step driver-alloc churn bounded and non-accelerating: the delta
    // between consecutive steps' cumulative drv_allocs should stay small and
    // roughly constant. An accumulating per-step allocation would make the
    // deltas grow.
    let deltas: Vec<u64> = steps
        .windows(2)
        .map(|w| w[1].1.saturating_sub(w[0].1))
        .collect();
    if let Some(max_delta) = deltas.iter().max().copied() {
        assert!(
            max_delta <= 64,
            "per-step driver allocations churned {max_delta}/step — unexpectedly high, \
             suggests an accumulating per-step allocation.\ndeltas={deltas:?}"
        );
    }
    if deltas.len() >= 3 {
        // Last delta must not be dramatically larger than the first steady
        // delta (no acceleration).
        let first = deltas[deltas.len() - 3].max(1);
        let last = *deltas.last().unwrap();
        assert!(
            last <= first * 2 + 8,
            "per-step allocation churn is accelerating (first steady {first} → last {last}), \
             indicating a growing per-step allocation.\ndeltas={deltas:?}"
        );
    }

    eprintln!(
        "mem-accounting gate OK: weights_peak={weights_peak} other_peak={other_peak} \
         global_peak={global_peak} steps={steps:?}"
    );
}
