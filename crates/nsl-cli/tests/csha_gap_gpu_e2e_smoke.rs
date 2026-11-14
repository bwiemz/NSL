//! CSHA Gap-suite — **end-to-end** GPU smoke of the toy program, per-op AD arm.
//!
//! This file used to have two more tests: one that trained the toy through
//! the CSHA fused forward and backward (`--csha auto`), and one that
//! compared that run's per-parameter weight movements with this arm's under
//! a loose tolerance. Training through the fused pair is now a compile
//! error, because its gradients do not match a reference (see
//! `crates/nsl-codegen/tests/csha_training_withheld.rs` and
//! `csha_toy_pretrain_gate.rs` in this directory), so both are gone. What
//! is left is the arm that still exists. The history below is kept for the
//! fixture it explains.
//!
//!
//! # What this test proves
//!
//! The existing Gap-suite tests (Gap A .. Gap I.5) all terminate at the
//! codegen layer: they inspect relocations, parse PTX, or validate SMEM
//! budgets in the emitted object file.  None of them prove that the
//! compiled toy program actually EXECUTES on a GPU, launches the fused
//! CSHA kernel, runs the optimizer step, and updates weights.
//!
//! This file closes that gap.  It drives the full CLI pipeline —
//!   parse → semantic → codegen → linker → spawn → CUDA init →
//!   GPU launch → SGD step → weight writeback
//! — and hard-asserts that EVERY trainable parameter moved between
//! before-step and after-step, then (third test) that the fused CSHA
//! path's per-parameter weight movements match the per-op AD baseline.
//!
//! # Fixture history — the degenerate-probe lesson (2026-07-23)
//!
//! The original TOY_SRC used all-ones x AND all-ones weights.  That is
//! DOUBLY degenerate:
//!   1. With V rows identical, dP is constant across keys, so the
//!      softmax Jacobian gives dS == 0 → **dQ and dK are mathematically
//!      zero**.  The old "all four params moved" assertion passed on the
//!      baseline arm by ONE f32 ULP of noise (+1.2207e-4 at sum=1024 is
//!      exactly the f32 spacing there) and the fused arm's exact zeros
//!      — the CORRECT answer — read as "wq/wk frozen".
//!   2. W=ones makes every q/k row uniform, so scores saturate softmax
//!      (exp-suppressed gradients) — a second degenerate regime.
//! On top of that the fixture never called `m.to(cuda)`, so the flash
//! FFIs got CPU tensors, every launch failed, and the pre-zeroed output
//! flowed on with exit 0.
//!
//! The current fixture avoids all three traps:
//!   - sin-varied x AND weights at O(1/sqrt(d)) scale (no saturation,
//!     dQ/dK genuinely nonzero);
//!   - `m.to(cuda)` + `.to(cuda)` inputs (kernels get device tensors);
//!   - movement is measured as `sum(abs(w_after - w_before))` against a
//!     pre-train `.clone()` — sum-of-weights deltas cancel on symmetric
//!     sin data, L1-of-difference cannot.
//!
//! # Why a shell-out rather than a JIT harness
//!
//! `nsl_codegen::compile_entry` emits a Cranelift object file.  Linking
//! it to a runnable process requires (a) the platform linker, (b) every
//! `#[no_mangle]` runtime FFI symbol, and (c) `nsl-runtime`'s atexit
//! hooks for GPU memory and profile summaries.  The CLI already does
//! all of that.  The assertion surface is stdout (`DELTA_<name>` marker
//! lines) plus the exit code; the tests print the full child
//! stdout/stderr on any assertion to keep debugging cheap.

#![cfg(feature = "cuda")]

use assert_cmd::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Toy @train program that exercises the whole flash-attention forward +
/// backward + optimizer-step chain on GPU and prints per-parameter L1
/// weight movement after one step.
///
/// Design notes:
///   - Input shape is `[1, 1, 32, 32]` (batch=1, heads=1, seq=32,
///     head_dim=32).  The CSHA kernel dispatches on rank-4 input.
///   - Weights are `sin(arange * k) * 0.17` — distinct frequencies per
///     projection, O(1/sqrt(32)) scale.  x is `sin(arange * 0.37)`.
///     This keeps scores in the soft part of softmax AND makes dQ/dK
///     mathematically nonzero (see the module doc's degenerate-probe
///     lesson).
///   - `m.to(cuda)` + `.to(cuda)` on x/y: the flash FFIs are GPU-only;
///     CPU boxes used to fail the launch and continue with zeros.
///   - `mse_loss(out, y)` with y=zeros gives well-scaled gradients
///     through SDPA → Q/K/V projections → RMSNorm.
///   - Movement probes: `sum(abs(m.<w> - <w>0))` against pre-train
///     clones — direct L1 of the SGD update, no cancellation.
const TOY_SRC: &str = r#"from nsl.nn.losses import mse_loss

model TinyAttn:
    w_norm: Tensor = ones([32])
    wq: Tensor = (tensor_sin(arange(1024.0) * 0.7) * 0.17).reshape([32, 32])
    wk: Tensor = (tensor_sin(arange(1024.0) * 1.3) * 0.17).reshape([32, 32])
    wv: Tensor = (tensor_sin(arange(1024.0) * 2.1) * 0.17).reshape([32, 32])

    @flash_attention(head_dim=32)
    fn forward(self, x: Tensor) -> Tensor:
        let x_norm = rmsnorm(x, self.w_norm, 0.00001)
        let q = x_norm @ self.wq
        let k = x_norm @ self.wk
        let v = x_norm @ self.wv
        let scale = 1.0 / sqrt(32.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
m.to(cuda)
# x MUST vary along the sequence axis. With `ones([1,1,32,32])` every row of
# V is identical, so dP_ij is constant in j and
# dS = P * (dP - sum_k P_ik dP_ik) vanishes IDENTICALLY. dQ = dS.K and
# dK = dS^T.Q are then exactly zero, so dWq and dWk are zero as a matter of
# arithmetic — and the all-four-params assertion below could never hold. The
# kernel was correct the whole time; the probe was degenerate (the same trap
# recorded for CSHA cycle 18). Both sides of the 2026-07-28 merge fixed this
# independently; the sin-based input (with the GPU placement its parity gate
# certifies against) is the survivor.
let x = ((tensor_sin(arange(1024.0) * 0.37)).reshape([1, 1, 32, 32])).to(cuda)
let y = (zeros([1, 1, 32, 32])).to(cuda)

let wq0 = m.wq.clone()
let wk0 = m.wk.clone()
let wv0 = m.wv.clone()
let wn0 = m.w_norm.clone()

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.05)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)

print("DELTA_wq")
print(sum(abs(m.wq - wq0)))
print("DELTA_wk")
print(sum(abs(m.wk - wk0)))
print("DELTA_wv")
print(sum(abs(m.wv - wv0)))
print("DELTA_w_norm")
print(sum(abs(m.w_norm - wn0)))
"#;

const PARAM_ORDER: [&str; 4] = ["wq", "wk", "wv", "w_norm"];

/// Parse the `DELTA_<NAME>` marker lines from child stdout.
///
/// The value line after each marker is either a bare float (CPU f64
/// print) or `tensor([<v>])` (GPU tensor print).  Diagnostic lines
/// (`[gpu-mem]`, `[nsl]`, ...) interleave with the markers, so we scan
/// forward to the first parsable line after each marker.
fn parse_deltas(stdout: &str) -> Result<Vec<(String, f64)>, String> {
    let lines: Vec<&str> = stdout.lines().collect();

    fn parse_value(t: &str) -> Option<f64> {
        if let Ok(v) = t.parse::<f64>() {
            return Some(v);
        }
        // GPU scalar-sum print: `tensor([22.647510528564453])`
        let inner = t.strip_prefix("tensor([")?.strip_suffix("])")?;
        inner.trim().parse::<f64>().ok()
    }

    fn next_value_after(lines: &[&str], idx: usize) -> Option<f64> {
        for line in lines.iter().skip(idx + 1) {
            let t = line.trim();
            if t.is_empty() || t.starts_with('[') {
                continue; // diagnostic like `[gpu-mem] step=0 driver=...`
            }
            return parse_value(t);
        }
        None
    }

    let mut deltas: Vec<(String, f64)> = Vec::new();
    for name in PARAM_ORDER {
        let marker = format!("DELTA_{name}");
        let idx = lines
            .iter()
            .position(|l| l.trim() == marker)
            .ok_or_else(|| format!("no {marker} marker line in stdout"))?;
        let v = next_value_after(&lines, idx)
            .ok_or_else(|| format!("no numeric line after {marker}"))?;
        deltas.push((name.to_string(), v));
    }
    Ok(deltas)
}

/// Spawn `nsl run --source-ad --target sm_89 [extra args] <toy>` and
/// return (status_success, stdout, stderr).
fn run_toy(extra_args: &[&str]) -> (bool, String, String) {
    let tmp = TempDir::new().expect("tempdir");
    let src_path = tmp.path().join("csha_toy_gpu_e2e.nsl");
    fs::write(&src_path, TOY_SRC).expect("write toy source");

    let mut cmd = Command::cargo_bin("nsl").expect("locate nsl binary");
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("run")
        // source-AD is required for the CSHA fused-backward dispatcher
        // (adjoint-generator path).  Tape-AD bypasses the Gap A..I.5
        // pipeline entirely.
        .arg("--source-ad")
        // Pin the SM the fused kernels are generated for (plain `cuda`
        // would give sm_80).
        .arg("--target")
        .arg("sm_89");
    for a in extra_args {
        cmd.arg(a);
    }
    cmd.arg(&src_path);
    let output = cmd.output().expect("spawn nsl run");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

fn is_driver_missing(stderr: &str) -> bool {
    stderr.contains("no CUDA devices")
        || stderr.contains("CUDA_ERROR_NOT_INITIALIZED")
        || stderr.contains("CUDA driver not found")
}

fn log_deltas(tag: &str, deltas: &[(String, f64)]) {
    eprintln!("[{tag}] per-parameter |w_after - w_before| L1:");
    for (name, d) in deltas {
        eprintln!("  {name:>8}: delta_l1={d:>14.8}");
    }
}

/// Every trainable parameter must move by more than this after one
/// SGD(lr=0.05) step on the sin-varied fixture.  The smallest mover is
/// wq/wk (softmax-Jacobian second-order path); the threshold is far
/// above f32 print noise but far below the real movement, so it fails
/// on "gradient silently zero" and never on rounding.
const MIN_DELTA_L1: f64 = 1e-5;

/// The bellwether end-to-end test (per-op AD baseline path).
///
/// Skips (not fails) if the host doesn't have a working CUDA driver.
///
/// On a real GPU the child must:
///   (a) exit rc=0
///   (b) print four finite DELTA_<name> L1 movements
///   (c) EVERY param must move by > MIN_DELTA_L1 — with the
///       non-degenerate fixture, dQ/dK are mathematically nonzero, so
///       a frozen wq/wk is a real bug, not fixture vacuity.
///
/// Invocation:
///   cargo test -p nsl-cli --test csha_gap_gpu_e2e_smoke \
///     --features cuda -- --ignored --nocapture
#[test]
#[ignore = "requires a CUDA-capable GPU"]
fn csha_gap_gpu_e2e_weights_change_after_one_step() {
    let (ok, stdout, stderr) = run_toy(&[]);

    if !ok {
        if is_driver_missing(&stderr) {
            eprintln!(
                "[csha-gpu-e2e] SKIP — no CUDA driver.\nstderr tail:\n{}",
                stderr.lines().rev().take(5).collect::<Vec<_>>().join("\n")
            );
            return;
        }
        panic!(
            "[csha-gpu-e2e] `nsl run` failed.\n\
             This is the full flash-attention train pipeline; the stderr \
             tail is the primary diagnostic.\n\n\
             stdout:\n{stdout}\n\nstderr:\n{stderr}"
        );
    }

    let deltas = parse_deltas(&stdout).unwrap_or_else(|e| {
        panic!(
            "[csha-gpu-e2e] failed to parse DELTA param movements: {e}\n\
             stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    });
    log_deltas("csha-gpu-e2e", &deltas);

    for (name, d) in &deltas {
        assert!(
            d.is_finite(),
            "[csha-gpu-e2e] DELTA_{name} = {d} is not finite"
        );
    }

    // The load-bearing assertion: ALL FOUR trainable params must have
    // moved.  A ~zero delta on any one means the gradient was silently
    // zero, the optimizer step never ran for that slot, or the SGD
    // write never landed in `m.<w>`'s device-backed storage.  (The
    // fixture makes dQ/dK mathematically nonzero — see module doc.)
    let stuck: Vec<&String> = deltas
        .iter()
        .filter(|(_, d)| *d <= MIN_DELTA_L1)
        .map(|(n, _)| n)
        .collect();
    assert!(
        stuck.is_empty(),
        "[csha-gpu-e2e] trainable param(s) did not move after one step: \
         {stuck:?}.  Expected ALL FOUR params (wq, wk, wv, w_norm) to \
         receive a non-zero gradient and update under SGD(lr=0.05, \
         mse_loss, sin-varied input).  Full per-param deltas above."
    );
}
