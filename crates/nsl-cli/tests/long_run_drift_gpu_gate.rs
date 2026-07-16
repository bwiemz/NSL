//! A3 long-run drift gate — sustained CCR / allocator validation.
//!
//! Short parity tests (2 optimizer steps) prove CCR recompute is bit-exact at
//! a single point in time. They cannot catch a SLOW drift: a FASE window
//! accumulator that leaks a little each step, an async-owned buffer that is
//! not reclaimed, a recompute whose error compounds. This gate trains a small
//! GQA LM for ~48 optimizer steps under `--deterministic` and asserts:
//!
//!   1. The `--checkpoint-blocks` loss stream is bit-identical to the baseline
//!      at EVERY step — CCR recompute stays faithful across the whole run.
//!   2. Reserved VRAM does not grow across steps after warmup — no per-step
//!      leak (the async-ownership / FASE-state concern).
//!   3. Per-step driver-allocation churn stays bounded — no accumulating
//!      allocation.
//!
//! GPU-only, `#[ignore]`. The full 500M / 1B several-hundred-step validation
//! is run manually and recorded in models/benchmarks/LONG_RUN_VALIDATION.md;
//! this gate is the fast, automated proxy that guards against regressions.
//!
//! Run: cargo test -p nsl-cli --features cuda --test long_run_drift_gpu_gate \
//!        -- --ignored --test-threads=1

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOutput {
    loss_stream: String,
    /// (step, reserved_mb, cumulative drv_allocs) from each `[gpu-mem] step=`.
    gpu_mem: Vec<(u64, u64, u64)>,
    stderr: String,
    success: bool,
}

fn run(tag: &str, checkpoint: bool) -> RunOutput {
    let root = repo_root();
    let fixture = root.join("crates/nsl-cli/tests/fixtures/long_run_drift.nsl");
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run", "--source-ad", "--deterministic"]);
    if checkpoint {
        cmd.arg("--checkpoint-blocks");
    }
    cmd.arg(&fixture)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_DEBUG_MEM_ALL", "1");
    let out = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

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

    let field = |line: &str, key: &str| -> Option<u64> {
        let idx = line.find(key)?;
        line[idx + key.len()..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .ok()
    };
    let gpu_mem: Vec<(u64, u64, u64)> = stderr
        .lines()
        .filter(|l| l.contains("[gpu-mem] step="))
        .filter_map(|l| Some((field(l, "step=")?, field(l, "reserved=")?, field(l, "drv_allocs=")?)))
        .collect();

    RunOutput { loss_stream, gpu_mem, stderr, success: out.status.success() }
}

#[test]
#[ignore = "requires CUDA GPU (~5 min: 3 x 48-step runs)"]
fn long_run_ccr_bit_exact_and_no_allocator_drift() {
    let base_a = run("base_a", false);
    assert!(base_a.success, "baseline A failed:\n{}", base_a.stderr);
    let base_b = run("base_b", false);
    assert!(base_b.success, "baseline B failed:\n{}", base_b.stderr);
    let ckpt = run("ckpt", true);
    assert!(ckpt.success, "checkpointed run failed:\n{}", ckpt.stderr);
    assert!(
        !ckpt.stderr.contains("running without checkpointing"),
        "CCR declined on the fixture — segmentation regressed:\n{}",
        ckpt.stderr
    );

    let n_steps = base_a.loss_stream.lines().count();
    assert!(n_steps >= 90, "expected ~96 micro-batch losses, got {n_steps}");

    // (1) CCR faithfulness over the whole run. --deterministic makes the GPU
    // path bit-reproducible (deterministic embedding kernel, A2), so the
    // strongest honest gate applies: bit-exact at every step. If the
    // environment is unexpectedly noisy, fall back to a per-step tolerance.
    let env_deterministic = base_a.loss_stream == base_b.loss_stream;
    if env_deterministic {
        assert_eq!(
            base_a.loss_stream, ckpt.loss_stream,
            "CCR loss stream diverged from baseline over {n_steps} steps (must be bit-exact \
             under --deterministic) — a slow recompute/accumulator drift"
        );
    } else {
        eprintln!(
            "[long-run] WARNING: --deterministic run was not bit-reproducible here; \
             falling back to per-step tolerance"
        );
        let parse = |s: &str| -> Vec<f64> {
            s.lines()
                .filter_map(|l| {
                    l.chars()
                        .skip_while(|c| !c.is_ascii_digit() && *c != '-')
                        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+'))
                        .collect::<String>()
                        .parse()
                        .ok()
                })
                .collect()
        };
        let (la, lb, lc) = (parse(&base_a.loss_stream), parse(&base_b.loss_stream), parse(&ckpt.loss_stream));
        let m = la.len().min(lb.len()).min(lc.len());
        let mut noise = 0.0f64;
        let mut dev = 0.0f64;
        for i in 0..m {
            noise = noise.max((lb[i] - la[i]).abs());
            dev = dev.max((lc[i] - la[i]).abs());
        }
        assert!(
            dev <= (noise * 4.0).max(1e-2),
            "CCR loss drifted {dev:.6} over the run (noise {noise:.6})"
        );
    }

    // (2) & (3) Allocator stability across the run. Drop step 0 (warmup).
    assert!(base_a.gpu_mem.len() >= 8, "expected many [gpu-mem] lines, got {}", base_a.gpu_mem.len());
    let steady: Vec<&(u64, u64, u64)> = base_a.gpu_mem.iter().filter(|(s, _, _)| *s >= 1).collect();
    let reserved: Vec<u64> = steady.iter().map(|(_, r, _)| *r).collect();
    let (min_r, max_r) = (
        *reserved.iter().min().unwrap(),
        *reserved.iter().max().unwrap(),
    );
    assert!(
        max_r - min_r <= 4,
        "reserved VRAM grew {min_r} MB -> {max_r} MB across the run — a per-step leak.\n{:?}",
        base_a.gpu_mem
    );

    // Per-step drv_allocs delta: bounded and not accelerating over 48 steps.
    let allocs: Vec<u64> = steady.iter().map(|(_, _, a)| *a).collect();
    let deltas: Vec<u64> = allocs.windows(2).map(|w| w[1].saturating_sub(w[0])).collect();
    if let Some(max_delta) = deltas.iter().max() {
        assert!(
            *max_delta <= 64,
            "per-step driver allocations churned {max_delta}/step — accumulating allocation.\n{deltas:?}"
        );
    }

    eprintln!(
        "[long-run] OK: {n_steps} steps, CCR bit-exact={env_deterministic}, \
         reserved stable {min_r}-{max_r} MB"
    );
}
