//! WRGA B.3.2 trigger measurement.
//!
//! Runs GatedLoRA forward-only and forward+backward workloads at the prescribed
//! and batch-sensitivity shapes, captures per-kernel GPU event timing via
//! `NSL_PROFILE_KERNELS`, partitions the trace into per-iteration buckets by
//! counting occurrences of the fused GatedLoRA kernel name, and computes
//! min/median/max per phase across the timed iterations.
//!
//! Emits verdict vs the 2.5x ratio trigger documented in
//! `docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md`.
//!
//! "Backward time" in this measurement means end-to-end train-iter time minus
//! forward-iter time, NOT isolated backward-kernel time. It includes source-AD
//! tape processing, the adapter-triple unfused backward kernels, allocator
//! costs, and any runtime overhead between Rust and the GPU. That is what the
//! trigger is actually gating on: real end-to-end training wall time.
//!
//! `#[ignore]`-gated. Invoke with:
//!   cargo test --features cuda --test wrga_gatedlora_backward_trigger \
//!       -- --ignored --nocapture

use assert_cmd::prelude::*;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(feature = "cuda")]
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[derive(Clone, Copy)]
struct Config {
    name: &'static str,
    batch: u64,
    seq: u64,
    dim: u64,
    rank: u64,
    alpha: u64,
}

const CONFIGS: &[Config] = &[
    Config {
        name: "prescribed_b32_r16",
        batch: 32,
        seq: 2048,
        dim: 4096,
        rank: 16,
        alpha: 32,
    },
    Config {
        name: "smaller_batch_b8_r16",
        batch: 8,
        seq: 2048,
        dim: 4096,
        rank: 16,
        alpha: 32,
    },
];

const WARMUP_ITERS: usize = 3;
const TIMED_ITERS: usize = 10;
const FUSED_MARKER_PREFIX: &str = "nsl_wrga_fused_gatedlora_";

/// Leading markers emitted by the init train block in fwd_only programs.
/// The init runs 1 epoch, contributing exactly 1 forward marker.
const INIT_MARKERS_FWD_ONLY: usize = 1;
/// fwd_bwd uses a single train block with epochs = total, so there are no
/// extra init markers beyond the timed+warmup ones.
const INIT_MARKERS_FWD_BWD: usize = 0;

fn total_iters() -> usize {
    WARMUP_ITERS + TIMED_ITERS
}

fn gen_forward_only(cfg: &Config) -> String {
    let tokens = cfg.batch * cfg.seq;
    let n = total_iters();
    // Two NSL quirks to work around:
    //   1. Forward calls against `@adapter(gatedlora, ...)` segfault unless
    //      the adapter side-table has been materialized. Materialization
    //      happens only inside a `train` block (tape_start).
    //   2. After materialization, adapter fields (lora_A, lora_B, gate) live
    //      on CPU even after `m.to(cuda)`. The fused runtime FFI guards on
    //      tensor.device != 0 and falls back to CPU math if any input is on
    //      CPU, so fused kernels never fire at realistic shapes. B.3.1
    //      fixtures work around this by manually reassigning adapter fields
    //      with `.to(cuda)` after the train block; the bench does the same.
    format!(
        r#"from nsl.nn.losses import mse_loss

model LlamaProxy:
    w: Tensor = zeros([{dim}, {dim}])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["LlamaProxy.w"], rank={rank}, alpha={alpha})
let m = LlamaProxy()
m.to(cuda)
let x = zeros([{tokens}, {dim}]).to(cuda)
let y_target = zeros([{tokens}, {dim}]).to(cuda)

# Init: materialize adapter side-table via one-epoch train block.
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)

# Reassign adapter fields onto CUDA so fused-FFI device checks pass.
m.lora_A_LlamaProxy_w__gatedlora = zeros([{dim}, {rank}]).to(cuda)
m.lora_B_LlamaProxy_w__gatedlora = zeros([{rank}, {dim}]).to(cuda)
m.gate_LlamaProxy_w__gatedlora = zeros([{dim}]).to(cuda)

# Forward-only loop: warmup + timed.
for i in range({n}):
    let _ = m.forward(x)
"#,
        dim = cfg.dim,
        rank = cfg.rank,
        alpha = cfg.alpha,
        tokens = tokens,
        n = n
    )
}

fn gen_forward_backward(cfg: &Config) -> String {
    let tokens = cfg.batch * cfg.seq;
    let n = total_iters();
    format!(
        r#"from nsl.nn.losses import mse_loss

model LlamaProxy:
    w: Tensor = zeros([{dim}, {dim}])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["LlamaProxy.w"], rank={rank}, alpha={alpha})
let m = LlamaProxy()
m.to(cuda)
let x = zeros([{tokens}, {dim}]).to(cuda)
let y_target = zeros([{tokens}, {dim}]).to(cuda)

train(model = m, epochs = {n}):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
"#,
        dim = cfg.dim,
        rank = cfg.rank,
        alpha = cfg.alpha,
        tokens = tokens,
        n = n
    )
}

#[cfg(feature = "cuda")]
fn run_and_profile(
    src: &str,
    name: &str,
) -> Result<(PathBuf, TempDir), String> {
    let tmp = TempDir::new().map_err(|e| e.to_string())?;
    let src_path = tmp.path().join(format!("{name}.nsl"));
    fs::write(&src_path, src).map_err(|e| e.to_string())?;
    let root = workspace_root();
    let stdlib = root.join("stdlib");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_PROFILE_KERNELS", "1")
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path);

    let out = cmd.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        return Err(format!(
            "nsl run failed for {name}:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        ));
    }
    let json_path = tmp.path().join("kernel_profile.json");
    if !json_path.exists() {
        return Err(format!("kernel_profile.json not found after {name}"));
    }
    Ok((json_path, tmp))
}

/// Parse kernel_profile.json, return Vec of (name, dur_us) sorted by timestamp.
fn parse_trace(path: &Path) -> Result<Vec<(String, f64)>, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let events = v
        .get("traceEvents")
        .and_then(|e| e.as_array())
        .ok_or_else(|| "missing traceEvents".to_string())?;
    let mut out: Vec<(String, f64, f64)> = Vec::with_capacity(events.len());
    for ev in events {
        let name = ev
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();
        let dur = ev.get("dur").and_then(|d| d.as_f64()).unwrap_or(0.0);
        let ts = ev.get("ts").and_then(|t| t.as_f64()).unwrap_or(0.0);
        out.push((name, dur, ts));
    }
    out.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out.into_iter().map(|(n, d, _)| (n, d)).collect())
}

/// Partition events into iterations using the fused-GatedLoRA kernel as a
/// marker for "iteration i starts here". Assumes exactly one marker per iter
/// (one forward call per step). Returns per-iter total kernel duration in us.
///
/// `min_markers_expected` asserts a lower bound; surplus markers are tolerated
/// (the caller is responsible for skipping leading init/warmup markers).
fn partition_by_marker(
    events: &[(String, f64)],
    min_markers_expected: usize,
) -> Result<Vec<f64>, String> {
    let mut marker_idxs: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(i, (n, _))| {
            if n.starts_with(FUSED_MARKER_PREFIX) {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if marker_idxs.is_empty() {
        return Err(format!(
            "no fused marker kernels '{FUSED_MARKER_PREFIX}*' found \
             in trace ({} total events) — forward fused path did not fire",
            events.len()
        ));
    }
    if marker_idxs.len() < min_markers_expected {
        return Err(format!(
            "expected >= {} fused-marker kernels, found {} \
             — at least one forward call did not fire",
            min_markers_expected,
            marker_idxs.len()
        ));
    }
    marker_idxs.push(events.len()); // sentinel for last iter end
    Ok(marker_idxs
        .windows(2)
        .map(|w| events[w[0]..w[1]].iter().map(|(_, d)| d).sum())
        .collect())
}

#[derive(Debug, Clone)]
struct Stats {
    min_ms: f64,
    median_ms: f64,
    max_ms: f64,
    per_iter_ms: Vec<f64>,
}

fn compute_stats(per_iter_us: &[f64]) -> Stats {
    let per_iter_ms: Vec<f64> = per_iter_us.iter().map(|us| us / 1000.0).collect();
    let mut sorted = per_iter_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_ms = sorted[0];
    let max_ms = *sorted.last().unwrap();
    let n = sorted.len();
    let median_ms = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    Stats {
        min_ms,
        median_ms,
        max_ms,
        per_iter_ms,
    }
}

fn spread_flag(s: &Stats) -> &'static str {
    if s.min_ms > 0.0 && s.max_ms > 1.15 * s.min_ms {
        " [SPREAD>15%]"
    } else {
        ""
    }
}

#[cfg(feature = "cuda")]
fn measure_phase(
    src: &str,
    name: &str,
    init_markers: usize,
) -> Result<Stats, String> {
    let (json_path, _tmp) = run_and_profile(src, name)?;
    let events = parse_trace(&json_path)?;
    let min_expected = init_markers + WARMUP_ITERS + TIMED_ITERS;
    let per_iter_us = partition_by_marker(&events, min_expected)?;
    // Drop init markers' buckets + warmup buckets; keep timed buckets.
    let timed_start = init_markers + WARMUP_ITERS;
    let timed_end = timed_start + TIMED_ITERS;
    if per_iter_us.len() < timed_end {
        return Err(format!(
            "insufficient marker buckets: {} < {} (init={init_markers}, warmup={WARMUP_ITERS}, timed={TIMED_ITERS})",
            per_iter_us.len(),
            timed_end,
        ));
    }
    let timed = &per_iter_us[timed_start..timed_end];
    Ok(compute_stats(timed))
}

fn is_oom(err: &str) -> bool {
    let e = err.to_lowercase();
    e.contains("out_of_memory")
        || e.contains("out of memory")
        || e.contains("cuda_error_out_of_memory")
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn wrga_b32_trigger_measurement() {
    let mut report = String::new();
    report.push_str("# WRGA B.3.2 trigger measurement\n\n");
    report.push_str(
        "Captures per-iteration GPU kernel event time for fused GatedLoRA \
         forward and unfused training (forward+backward) across two \
         configurations. Compares `ratio = fwd+bwd / fwd` against the 2.5x \
         trigger in `docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md`.\n\n",
    );
    report.push_str(&format!(
        "Warmup iters: {WARMUP_ITERS} (discarded). Timed iters: {TIMED_ITERS}. \
         Spread flag if max > 1.15 * min.\n\n",
    ));
    report.push_str(
        "\"Backward time\" here = end-to-end train-iter time minus forward-iter \
         time, NOT isolated backward-kernel time. It includes source-AD tape \
         processing, the adapter-triple unfused backward kernels, allocator \
         costs, and runtime overhead.\n\n",
    );

    for cfg in CONFIGS {
        eprintln!(
            "\n=== Config: {} (batch={}, seq={}, dim={}, rank={}) ===",
            cfg.name, cfg.batch, cfg.seq, cfg.dim, cfg.rank
        );
        report.push_str(&format!(
            "## {} (batch={}, seq={}, dim={}, rank={}, alpha={})\n\n",
            cfg.name, cfg.batch, cfg.seq, cfg.dim, cfg.rank, cfg.alpha
        ));

        let fwd_src = gen_forward_only(cfg);
        let bwd_src = gen_forward_backward(cfg);

        let fwd_res = measure_phase(&fwd_src, "fwd_only", INIT_MARKERS_FWD_ONLY);
        if fwd_res.is_err() {
            let msg = fwd_res.as_ref().err().unwrap();
            if is_oom(msg) {
                eprintln!("  forward: OOM");
                report.push_str(
                    "- forward: **OOM** — workload does not fit on this GPU\n\n",
                );
                continue;
            }
        }
        let bwd_res = measure_phase(&bwd_src, "fwd_bwd", INIT_MARKERS_FWD_BWD);
        if bwd_res.is_err() {
            let msg = bwd_res.as_ref().err().unwrap();
            if is_oom(msg) {
                eprintln!("  fwd+bwd: OOM");
                report.push_str(
                    "- fwd+bwd: **OOM** — workload does not fit on this GPU\n\n",
                );
                continue;
            }
        }

        let (fwd, bwd) = match (fwd_res, bwd_res) {
            (Ok(f), Ok(b)) => (f, b),
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("  FAILED: {e}");
                report.push_str(&format!("- **FAILED:** {e}\n\n"));
                continue;
            }
        };

        eprintln!(
            "  forward: min={:.2}ms median={:.2}ms max={:.2}ms{}",
            fwd.min_ms,
            fwd.median_ms,
            fwd.max_ms,
            spread_flag(&fwd)
        );
        eprintln!(
            "  fwd+bwd: min={:.2}ms median={:.2}ms max={:.2}ms{}",
            bwd.min_ms,
            bwd.median_ms,
            bwd.max_ms,
            spread_flag(&bwd)
        );

        let bwd_only_median = bwd.median_ms - fwd.median_ms;
        let ratio = if fwd.median_ms > 0.0 {
            bwd.median_ms / fwd.median_ms
        } else {
            0.0
        };
        let verdict = if ratio > 2.5 {
            format!("TRIGGER FIRES (ratio {ratio:.3}x > 2.5x) — B.3.2 should be scheduled")
        } else {
            format!("trigger does not fire (ratio {ratio:.3}x <= 2.5x) — B.3.2 stays deferred")
        };
        eprintln!(
            "  backward-only (median) ~= {:.2}ms   ratio = {:.3}   -> {}",
            bwd_only_median, ratio, verdict
        );

        report.push_str(&format!(
            "- forward: min={:.2}ms median={:.2}ms max={:.2}ms{}\n\
             - fwd+bwd: min={:.2}ms median={:.2}ms max={:.2}ms{}\n\
             - backward-only (median) ~= {:.2}ms; ratio = {:.3}x\n\
             - **Verdict:** {}\n\n\
             Per-iter raw (ms, iter order):\n\
             - forward: {:?}\n\
             - fwd+bwd: {:?}\n\n",
            fwd.min_ms,
            fwd.median_ms,
            fwd.max_ms,
            spread_flag(&fwd),
            bwd.min_ms,
            bwd.median_ms,
            bwd.max_ms,
            spread_flag(&bwd),
            bwd_only_median,
            ratio,
            verdict,
            fwd.per_iter_ms
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect::<Vec<_>>(),
            bwd.per_iter_ms
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect::<Vec<_>>(),
        ));
    }

    let report_path = workspace_root()
        .join("target")
        .join("wrga_b32_trigger_report.md");
    if let Some(parent) = report_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&report_path, &report).expect("write report");
    eprintln!("\nReport written to {}", report_path.display());
}

// Build-gate parity: without the cuda feature the file must still compile.
#[cfg(not(feature = "cuda"))]
#[test]
#[ignore]
fn wrga_b32_trigger_measurement_requires_cuda() {
    eprintln!("wrga_b32_trigger_measurement requires --features cuda");
}

// -------- Option 2 side measurement: unfused triple fwd vs bwd --------
//
// This is NOT the B.3.2 trigger answer. The trigger requires a fused-forward
// vs unfused-backward ratio at the prescribed shape. As of 2026-04-19, fused
// forward PTX fails ptxas (CUDA_ERROR_INVALID_PTX) at n=4096 / k=4096 / r=16
// (prescribed shape), so the fused path is unusable at the target scale.
//
// While we have the harness set up, capture the unfused-triple fwd vs bwd
// ratio as a context data point on source-AD's baseline overhead on GatedLoRA
// backward. This does NOT answer the trigger (which would need fused
// forward), but confirms the harness works and characterizes the unfused
// symmetry. We force the unfused path via `--target cuda_sm70` (the fusion
// dispatch gates on sm >= 80, so sm_70 compiles the unfused-triple AST
// directly).

#[cfg(feature = "cuda")]
fn gen_unfused_forward_only(cfg: &Config, n: u64) -> String {
    let tokens = cfg.batch * cfg.seq;
    format!(
        r#"from nsl.nn.losses import mse_loss

model LlamaProxy:
    w: Tensor = zeros([{dim}, {dim}])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["LlamaProxy.w"], rank={rank}, alpha={alpha})
let m = LlamaProxy()
m.to(cuda)
let x = zeros([{tokens}, {dim}]).to(cuda)
let y_target = zeros([{tokens}, {dim}]).to(cuda)

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)

m.lora_A_LlamaProxy_w__gatedlora = zeros([{dim}, {rank}]).to(cuda)
m.lora_B_LlamaProxy_w__gatedlora = zeros([{rank}, {dim}]).to(cuda)
m.gate_LlamaProxy_w__gatedlora = zeros([{dim}]).to(cuda)

for i in range({n}):
    let _ = m.forward(x)
"#,
        dim = cfg.dim,
        rank = cfg.rank,
        alpha = cfg.alpha,
        tokens = tokens,
        n = n,
    )
}

#[cfg(feature = "cuda")]
fn gen_unfused_forward_backward(cfg: &Config, n: u64) -> String {
    let tokens = cfg.batch * cfg.seq;
    format!(
        r#"from nsl.nn.losses import mse_loss

model LlamaProxy:
    w: Tensor = zeros([{dim}, {dim}])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["LlamaProxy.w"], rank={rank}, alpha={alpha})
let m = LlamaProxy()
m.to(cuda)
let x = zeros([{tokens}, {dim}]).to(cuda)
let y_target = zeros([{tokens}, {dim}]).to(cuda)

train(model = m, epochs = {n}):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
"#,
        dim = cfg.dim,
        rank = cfg.rank,
        alpha = cfg.alpha,
        tokens = tokens,
        n = n,
    )
}

/// Runs an NSL program once and returns subprocess wall-clock elapsed time
/// in milliseconds. Uses `Instant` around `cmd.output()`, so the figure
/// includes everything the subprocess does: startup, compile+link, CUDA
/// context init, the NSL program itself, and process teardown.
///
/// For slope extraction against two N values, the fixed-overhead portion
/// cancels and `(T_large - T_small) / (N_large - N_small)` recovers the
/// per-iter cost.
#[cfg(feature = "cuda")]
fn run_unfused_and_wall_ms(src: &str, name: &str) -> Result<f64, String> {
    let tmp = TempDir::new().map_err(|e| e.to_string())?;
    let src_path = tmp.path().join(format!("{name}.nsl"));
    fs::write(&src_path, src).map_err(|e| e.to_string())?;
    let root = workspace_root();
    let stdlib = root.join("stdlib");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", &stdlib)
        // NSL_PROFILE_KERNELS omitted: the event-based profiler reports
        // zero durations on this build (events created before context init).
        // Subprocess wall-clock timing + slope extraction bypasses that.
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm70"])
        .arg(&src_path);

    let t0 = std::time::Instant::now();
    let out = cmd.output().map_err(|e| e.to_string())?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("nsl run failed for {name}:\n{stderr}"));
    }
    Ok(elapsed_ms)
}

/// Runs an NSL program with `NSL_WRGA_FUSED_CUDA=1` + `cuda_sm80` so the
/// GatedLoRA fused-forward PTX fires, and returns subprocess wall-clock
/// elapsed time in milliseconds. Companion to `run_unfused_and_wall_ms`.
#[cfg(feature = "cuda")]
fn run_fused_and_wall_ms(src: &str, name: &str) -> Result<f64, String> {
    let tmp = TempDir::new().map_err(|e| e.to_string())?;
    let src_path = tmp.path().join(format!("{name}.nsl"));
    fs::write(&src_path, src).map_err(|e| e.to_string())?;
    let root = workspace_root();
    let stdlib = root.join("stdlib");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path);

    let t0 = std::time::Instant::now();
    let out = cmd.output().map_err(|e| e.to_string())?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("nsl run failed for {name}:\n{stderr}"));
    }
    Ok(elapsed_ms)
}

/// Forward-only NSL program that fires the fused GatedLoRA kernel.
/// Identical structure to `gen_forward_only` but factored out with an
/// explicit N parameter for slope extraction.
#[cfg(feature = "cuda")]
fn gen_fused_forward_only(cfg: &Config, n: u64) -> String {
    let tokens = cfg.batch * cfg.seq;
    format!(
        r#"from nsl.nn.losses import mse_loss

model LlamaProxy:
    w: Tensor = zeros([{dim}, {dim}])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["LlamaProxy.w"], rank={rank}, alpha={alpha})
let m = LlamaProxy()
m.to(cuda)
let x = zeros([{tokens}, {dim}]).to(cuda)
let y_target = zeros([{tokens}, {dim}]).to(cuda)

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)

m.lora_A_LlamaProxy_w__gatedlora = zeros([{dim}, {rank}]).to(cuda)
m.lora_B_LlamaProxy_w__gatedlora = zeros([{rank}, {dim}]).to(cuda)
m.gate_LlamaProxy_w__gatedlora = zeros([{dim}]).to(cuda)

for i in range({n}):
    let _ = m.forward(x)
"#,
        dim = cfg.dim,
        rank = cfg.rank,
        alpha = cfg.alpha,
        tokens = tokens,
        n = n,
    )
}

/// Fused forward + unfused backward (via source-AD through the train block).
/// At compile time, forward goes through the fused kernel; backward goes
/// through the adapter-triple source-AD path.
#[cfg(feature = "cuda")]
fn gen_fused_forward_backward(cfg: &Config, n: u64) -> String {
    let tokens = cfg.batch * cfg.seq;
    format!(
        r#"from nsl.nn.losses import mse_loss

model LlamaProxy:
    w: Tensor = zeros([{dim}, {dim}])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["LlamaProxy.w"], rank={rank}, alpha={alpha})
let m = LlamaProxy()
m.to(cuda)
let x = zeros([{tokens}, {dim}]).to(cuda)
let y_target = zeros([{tokens}, {dim}]).to(cuda)

train(model = m, epochs = {n}):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
"#,
        dim = cfg.dim,
        rank = cfg.rank,
        alpha = cfg.alpha,
        tokens = tokens,
        n = n,
    )
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn wrga_b32_fused_trigger_final() {
    // The actual B.3.2 trigger answer: fused-forward vs unfused-backward at
    // prescribed shape. Prerequisite (2026-04-19 K-loop rewrite) shipped, so
    // the fused kernel now fires at the target shape. Method mirrors the
    // unfused side measurement: subprocess slope extraction across N in
    // {3, 13} to cancel one-time setup overhead.
    let mut report = String::new();
    report.push_str("# WRGA B.3.2 trigger - FUSED forward vs UNFUSED backward\n\n");
    report.push_str(
        "Captures the trigger condition from `docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md`: \
         `backward_time > 2.5 x forward_time` at seq=2048, rank=16, Llama-3-8B-proxy dims. \
         Forward fires the B.3.1 fused GatedLoRA kernel; backward goes through \
         source-AD's adapter-triple unfused path (this is what B.3.2 would replace).\n\n",
    );
    report.push_str(&format!(
        "Warmup iters: {WARMUP_ITERS} discarded (via N=3 baseline for slope extraction). \
         Timed iters: {TIMED_ITERS} per measurement. Wall-clock via `Instant` around \
         subprocess; fixed setup/compile cost cancels in the slope.\n\n",
    ));
    report.push_str(
        "Prerequisite: 2026-04-19 K-loop rewrite (commit converting the emitter's \
         Rust-side for-loop to a PTX runtime loop). Without it, fused forward produces \
         20 MB PTX at k=4096 and ptxas rejects it - the fused path silently CPU-falls-back \
         and this measurement degenerates to the unfused side-measurement.\n\n",
    );

    for cfg in CONFIGS {
        eprintln!(
            "\n=== FUSED trigger: {} (batch={}, seq={}, dim={}, rank={}) ===",
            cfg.name, cfg.batch, cfg.seq, cfg.dim, cfg.rank
        );
        report.push_str(&format!(
            "## {} (batch={}, seq={}, dim={}, rank={}, alpha={})\n\n",
            cfg.name, cfg.batch, cfg.seq, cfg.dim, cfg.rank, cfg.alpha
        ));

        let run = |gen: &dyn Fn(&Config, u64) -> String, name: &str| -> Result<f64, String> {
            let t3 = run_fused_and_wall_ms(&gen(cfg, 3), &format!("{name}_n3"))?;
            let t13 = run_fused_and_wall_ms(&gen(cfg, 13), &format!("{name}_n13"))?;
            Ok((t13 - t3) / 10.0)
        };

        let fwd = match run(&gen_fused_forward_only, "fus_fwd") {
            Ok(v) => v,
            Err(e) => {
                if is_oom(&e) {
                    eprintln!("  forward: OOM");
                    report.push_str("- forward: **OOM**\n\n");
                    continue;
                }
                eprintln!("  forward: FAILED {e}");
                report.push_str(&format!("- forward: **FAILED:** {e}\n\n"));
                continue;
            }
        };
        let bwd = match run(&gen_fused_forward_backward, "fus_bwd") {
            Ok(v) => v,
            Err(e) => {
                if is_oom(&e) {
                    eprintln!("  fwd+bwd: OOM");
                    report.push_str("- fwd+bwd: **OOM**\n\n");
                    continue;
                }
                eprintln!("  fwd+bwd: FAILED {e}");
                report.push_str(&format!("- fwd+bwd: **FAILED:** {e}\n\n"));
                continue;
            }
        };

        let bwd_only = bwd - fwd;
        let ratio = if fwd > 0.0 { bwd / fwd } else { 0.0 };
        let verdict = if ratio > 2.5 {
            format!("**TRIGGER FIRES (ratio {ratio:.3}x > 2.5x) - B.3.2 should be scheduled**")
        } else {
            format!(
                "trigger does not fire (ratio {ratio:.3}x <= 2.5x) - B.3.2 stays deferred"
            )
        };
        eprintln!(
            "  fused fwd: {:.2}ms/iter   fwd+bwd: {:.2}ms/iter   ratio: {:.3}x   bwd-only: {:.2}ms",
            fwd, bwd, ratio, bwd_only
        );
        eprintln!("  {verdict}");
        report.push_str(&format!(
            "- fused forward: {:.2}ms/iter\n\
             - fused fwd + unfused bwd: {:.2}ms/iter\n\
             - backward-only: {:.2}ms/iter\n\
             - ratio: {:.3}x\n\
             - **Verdict:** {verdict}\n\n",
            fwd, bwd, bwd_only, ratio
        ));
    }

    let report_path = workspace_root()
        .join("target")
        .join("wrga_b32_fused_trigger_report.md");
    if let Some(parent) = report_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&report_path, &report).expect("write fused trigger report");
    eprintln!("\nFused-trigger report written to {}", report_path.display());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn wrga_b32_unfused_side_measurement() {
    let mut report = String::new();
    report.push_str("# WRGA B.3.2 — unfused-triple side measurement\n\n");
    report.push_str(
        "**CAVEAT:** this is NOT the B.3.2 trigger answer. The trigger requires \
         a fused-forward vs unfused-backward ratio. As of 2026-04-19, fused \
         forward PTX fails ptxas at the prescribed (Llama-3-8B-proxy) shape, so \
         the fused path is unusable at target scale — diagnosing that is the \
         blocker. This measurement characterizes the unfused-triple \
         forward vs unfused-triple backward symmetry on source-AD; it does NOT \
         answer the trigger question and should not be used as a B.3.2 \
         scheduling input.\n\n",
    );
    report.push_str(
        "Method: `--target cuda_sm70` forces the AST rewrite to emit unfused \
         triple (fusion dispatch requires sm >= 80). Uses slope extraction \
         across N=3 and N=13 subprocess runs to cancel out one-time setup cost \
         (compile+link, CUDA init, model placement); per-iter wall time = \
         (T_13 - T_3) / 10, measured via `Instant` around the subprocess. \
         Note: NSL's built-in `NSL_PROFILE_KERNELS` event profiler reports zero \
         durations on this build (events created before CUDA context), so \
         we fall back to wall-clock.\n\n",
    );

    for cfg in CONFIGS {
        eprintln!(
            "\n=== Unfused side: {} (batch={}, seq={}, dim={}, rank={}) ===",
            cfg.name, cfg.batch, cfg.seq, cfg.dim, cfg.rank
        );
        report.push_str(&format!(
            "## {} (batch={}, seq={}, dim={}, rank={}, alpha={})\n\n",
            cfg.name, cfg.batch, cfg.seq, cfg.dim, cfg.rank, cfg.alpha
        ));

        let run = |gen: &dyn Fn(&Config, u64) -> String, name: &str| -> Result<f64, String> {
            let t3 = run_unfused_and_wall_ms(&gen(cfg, 3), &format!("{name}_n3"))?;
            let t13 = run_unfused_and_wall_ms(&gen(cfg, 13), &format!("{name}_n13"))?;
            Ok((t13 - t3) / 10.0)
        };

        let fwd = match run(&gen_unfused_forward_only, "uf_fwd") {
            Ok(v) => v,
            Err(e) => {
                if is_oom(&e) {
                    eprintln!("  forward: OOM");
                    report.push_str("- forward: **OOM**\n\n");
                    continue;
                }
                eprintln!("  forward: FAILED {e}");
                report.push_str(&format!("- forward: **FAILED:** {e}\n\n"));
                continue;
            }
        };
        let bwd = match run(&gen_unfused_forward_backward, "uf_bwd") {
            Ok(v) => v,
            Err(e) => {
                if is_oom(&e) {
                    eprintln!("  fwd+bwd: OOM");
                    report.push_str("- fwd+bwd: **OOM**\n\n");
                    continue;
                }
                eprintln!("  fwd+bwd: FAILED {e}");
                report.push_str(&format!("- fwd+bwd: **FAILED:** {e}\n\n"));
                continue;
            }
        };

        let bwd_only = bwd - fwd;
        let ratio = if fwd > 0.0 { bwd / fwd } else { 0.0 };
        eprintln!(
            "  forward: {:.2}ms/iter   fwd+bwd: {:.2}ms/iter   ratio: {:.3}x   bwd-only: {:.2}ms",
            fwd, bwd, ratio, bwd_only
        );
        report.push_str(&format!(
            "- forward (unfused triple): {:.2}ms/iter (mean)\n\
             - fwd+bwd (unfused triple): {:.2}ms/iter (mean)\n\
             - backward-only (unfused): {:.2}ms/iter\n\
             - ratio (fwd+bwd / fwd): {:.3}x\n\n\
             **Interpretation:** symmetric unfused-triple source-AD; ratio reflects \
             the source-AD backward/forward cost ratio when NEITHER side is fused. \
             Does NOT indicate whether B.3.2 (fused backward) is justified.\n\n",
            fwd, bwd, bwd_only, ratio
        ));
    }

    let report_path = workspace_root()
        .join("target")
        .join("wrga_b32_unfused_side_report.md");
    if let Some(parent) = report_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&report_path, &report).expect("write side report");
    eprintln!("\nSide-data report written to {}", report_path.display());
}

// Compile-time smoke that the source-gen helpers produce well-formed NSL.
#[test]
fn source_gen_smoke() {
    for cfg in CONFIGS {
        let fwd = gen_forward_only(cfg);
        assert!(fwd.contains(&format!("rank={}", cfg.rank)));
        assert!(fwd.contains(&format!("alpha={}", cfg.alpha)));
        assert!(
            fwd.contains("train(model = m, epochs = 1)"),
            "fwd_only must contain init train block for adapter materialization"
        );
        assert!(fwd.contains(&format!("for i in range({})", total_iters())));
        let bwd = gen_forward_backward(cfg);
        assert!(bwd.contains(&format!("epochs = {}", total_iters())));
        assert!(bwd.contains("optimizer: SGD"));
    }
}
