//! CSHA Gap-suite — **end-to-end** GPU smoke.
//!
//! # What this test proves
//!
//! The existing Gap-suite tests (Gap A .. Gap I.5) all terminate at the
//! codegen layer: they inspect relocations, parse PTX, or validate SMEM
//! budgets in the emitted object file.  None of them prove that the
//! compiled toy program actually EXECUTES on a GPU, launches the fused
//! CSHA kernel, runs the optimizer step, and updates weights.
//!
//! This test closes that gap.  It drives the full CLI pipeline —
//!   parse → semantic → codegen → linker → spawn → CUDA init →
//!   GPU launch → SGD step → weight writeback
//! — and hard-asserts that at least one trainable parameter moved by a
//! non-zero amount between before-step and after-step.  If the kernel-
//! level gradients were zero, if the optimizer step was skipped, if
//! the fused backward silently no-op'd, or if the CSHA chain wired
//! through but never reached `m.wq` on device memory, this test turns
//! red.
//!
//! # Why a shell-out rather than a JIT harness
//!
//! `nsl_codegen::compile_entry` emits a Cranelift object file.  Linking
//! it to a runnable process requires (a) the platform linker, (b) every
//! `#[no_mangle]` runtime FFI symbol, and (c) `nsl-runtime`'s atexit
//! hooks for GPU memory and profile summaries.  The CLI already does
//! all of that.  Rather than duplicate the `JITBuilder::finalize` +
//! symbol-table wiring in a test harness, we shell out to the built
//! `nsl` binary.  The assertion surface is stdout (`print(sum(w))`
//! lines captured before and after the training step) plus the exit
//! code.  The compromise: failure modes in the child process surface
//! as line-parse misses, not typed Rust errors.  The test prints the
//! full child stdout/stderr on any assertion to keep debugging cheap.
//!
//! # Test source: `toy_pretrain_hd32_e2e.nsl`
//!
//! A variant of `examples/csha_toy_pretrain_hd32.nsl`:
//!   - `@flash_attention(head_dim=32)` on `TinyAttn::forward`
//!   - Four trainable params: `w_norm: [32]`, `wq/wk/wv: [32, 32]`
//!   - 4-D batched input `ones([1, 1, 32, 32])` — the existing toy's
//!     2-D input fails inside the CSHA shape plumbing (asks for
//!     dim=3 of a 2-D tensor).  The 4-D shape matches
//!     [batch=1, heads=1, seq=32, head_dim=32], which is what the
//!     CSHA kernel expects.
//!   - `mse_loss(out, y)` — the original `sum(out)` loss trips a
//!     separate source-AD lowering bug (`matmul requires at least 2D
//!     tensors (got 2D and 0D)`) that is NOT CSHA-specific and is
//!     orthogonal to this test's scope.
//!   - `print(sum(m.wq))`, `print(sum(m.wk))`, `print(sum(m.wv))`,
//!     `print(sum(m.w_norm))` bracket the `train` block so we can
//!     capture before/after sums from stdout.
//!
//! # What the assertions look like
//!
//! 1. Child exits with code 0.
//! 2. All 8 BEFORE/AFTER sums parse as finite floats.
//! 3. At least one param changed (proves optimizer step ran AND
//!    received non-zero gradients — the Gap I design doc's implicit
//!    bellwether).
//! 4. Per-param: we log the delta so a human reviewer can see which
//!    of {wq, wk, wv, w_norm} moved and by how much.
//!
//! # Scope note
//!
//! This test is a **structural** e2e.  It does NOT numerically verify
//! the gradient against a CPU reference — that's what
//! `tests/csha_cuda_backward.rs::t6_3_smoke_single_config` does at the
//! kernel level.  Here, the question is "does the entire compile → run
//! → step pipeline work on real hardware?", not "are the gradients
//! correct to 5e-3?".

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

/// Toy @train program that exercises the whole CSHA fused-backward +
/// optimizer-step chain on GPU and prints trainable-param sums before
/// and after one step.
///
/// Design notes:
///   - Input shape is `[1, 1, 32, 32]` (batch=1, heads=1, seq=32,
///     head_dim=32).  The CSHA kernel dispatches on rank-4 input.
///   - `mse_loss` avoids the separate source-AD `sum(tensor)` bug
///     that surfaces on GPU (tracked separately).
///   - `y = zeros` + `x = ones` gives non-zero mse gradients that
///     propagate through SDPA → Q/K/V projections → RMSNorm.
///   - BEFORE/AFTER block structure: the `train` block runs in the
///     middle, and `m.wq` etc. are read after the step so the
///     optimizer-written values are observable.
///
/// The `before_*` / `after_*` tags are only for human readability in
/// failure diagnostics — the parser picks up the two floats after each
/// `BEFORE_<NAME>` / `AFTER_<NAME>` marker line.
const TOY_SRC: &str = r#"from nsl.nn.losses import mse_loss

model TinyAttn:
    w_norm: Tensor = ones([32])
    wq: Tensor = ones([32, 32])
    wk: Tensor = ones([32, 32])
    wv: Tensor = ones([32, 32])

    @flash_attention(head_dim=32)
    fn forward(self, x: Tensor) -> Tensor:
        let x_norm = rmsnorm(x, self.w_norm, 0.00001)
        let q = x_norm @ self.wq
        let k = x_norm @ self.wk
        let v = x_norm @ self.wv
        let scale = 1.0 / sqrt(32.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
let x = ones([1, 1, 32, 32])
let y = zeros([1, 1, 32, 32])

print("BEFORE_wq")
print(sum(m.wq))
print("BEFORE_wk")
print(sum(m.wk))
print("BEFORE_wv")
print(sum(m.wv))
print("BEFORE_w_norm")
print(sum(m.w_norm))

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)

print("AFTER_wq")
print(sum(m.wq))
print("AFTER_wk")
print(sum(m.wk))
print("AFTER_wv")
print(sum(m.wv))
print("AFTER_w_norm")
print(sum(m.w_norm))
"#;

/// Parse a BEFORE_<NAME> / AFTER_<NAME> pair from the child process
/// stdout and return `(before, after, delta)` for each trainable
/// parameter.  Missing or non-numeric lines produce an informative
/// error string.
fn parse_before_after(stdout: &str) -> Result<Vec<(String, f64, f64, f64)>, String> {
    // Collect lines; tag-line i is followed by value-line i+1.  Child
    // process also emits `[gpu-mem]` / `[gpu-alloc-summary]` /
    // `[nsl]` diagnostics interleaved with our markers, so we walk
    // the full stream and pick up the first numeric line after each
    // marker (skipping any non-numeric log lines).
    let lines: Vec<&str> = stdout.lines().collect();
    let param_order = ["wq", "wk", "wv", "w_norm"];
    let mut befores: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    let mut afters: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();

    fn next_float_after(lines: &[&str], idx: usize) -> Option<f64> {
        for line in lines.iter().skip(idx + 1) {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if t.starts_with('[') {
                // diagnostic like `[gpu-mem] step=0 driver=...`
                continue;
            }
            // Accept an int or float literal as the ENTIRE line (that's
            // what `print(sum(...))` emits — a bare number).
            if let Ok(v) = t.parse::<f64>() {
                return Some(v);
            }
            // If it's another marker or text, stop — we missed the
            // numeric line for this marker.
            return None;
        }
        None
    }

    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        for name in param_order {
            if t == format!("BEFORE_{name}") {
                if let Some(v) = next_float_after(&lines, i) {
                    befores.insert(name.to_string(), v);
                }
            } else if t == format!("AFTER_{name}") {
                if let Some(v) = next_float_after(&lines, i) {
                    afters.insert(name.to_string(), v);
                }
            }
        }
    }

    let mut out = Vec::new();
    for name in param_order {
        let before = befores.get(name).copied().ok_or_else(|| {
            format!("no BEFORE_{name} numeric line in stdout")
        })?;
        let after = afters.get(name).copied().ok_or_else(|| {
            format!("no AFTER_{name} numeric line in stdout")
        })?;
        out.push((name.to_string(), before, after, after - before));
    }
    Ok(out)
}

/// The bellwether end-to-end test.
///
/// Skips (not fails) if the host doesn't have a working CUDA driver —
/// `nsl run` prints a clear diagnostic and exits non-zero in that case,
/// and we don't want to spuriously fail a CPU-only CI box.
///
/// On a real GPU (NVIDIA sm ≥ 75) the child must:
///   (a) exit rc=0
///   (b) print eight finite BEFORE/AFTER sums
///   (c) at least one param must change by |delta| > 1e-6 (proves both
///       a non-zero gradient and a real optimizer step — a single
///       meaningful update through the whole pipeline).
///
/// The test logs per-param deltas unconditionally so a reviewer can
/// see which of {wq, wk, wv, w_norm} moved.
///
/// Invocation:
///   cargo test -p nsl-cli --test csha_gap_gpu_e2e_smoke \
///     --features cuda -- --ignored --nocapture
#[test]
#[ignore = "requires a CUDA-capable GPU"]
fn csha_gap_gpu_e2e_weights_change_after_one_step() {
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
        // `parse_gpu_sm_from_target` expects `sm_<N>`.  The default
        // `cuda` string panics inside `compile_flash_attention_kernels`.
        // sm_89 (Ada Lovelace) matches our dev boxes; any sm >= 75
        // emits valid PTX for hd=32.
        .arg("--target")
        .arg("sm_89")
        .arg(&src_path);
    let output = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        // If the failure is a missing CUDA driver, downgrade to skip
        // with a clear message — this test is explicitly a GPU smoke
        // and is `#[ignore]`-gated, so a driver-missing box isn't the
        // target population.  Any other non-zero exit is a real fail.
        let driver_missing = stderr.contains("no CUDA devices")
            || stderr.contains("CUDA_ERROR_NOT_INITIALIZED")
            || stderr.contains("CUDA driver not found");
        if driver_missing {
            eprintln!(
                "[csha-gpu-e2e] SKIP — no CUDA driver.\nstderr tail:\n{}",
                stderr.lines().rev().take(5).collect::<Vec<_>>().join("\n")
            );
            return;
        }
        panic!(
            "[csha-gpu-e2e] `nsl run` failed with exit {}.\n\
             This is the full CSHA fused-backward pipeline; if the \
             child panicked inside CUDA or the runtime, the stderr tail \
             is the primary diagnostic.\n\n\
             stdout:\n{stdout}\n\n\
             stderr:\n{stderr}",
            output.status
        );
    }

    let deltas = parse_before_after(&stdout).unwrap_or_else(|e| {
        panic!(
            "[csha-gpu-e2e] failed to parse BEFORE/AFTER param sums: {e}\n\
             The child process succeeded (rc=0) but did not emit the \
             expected print() markers — likely the `print(sum(m.<w>))` \
             lowering regressed, or the train block ran but bypassed \
             the reassignment that backs `m.wq` with updated device \
             memory.\n\n\
             stdout:\n{stdout}\n\n\
             stderr:\n{stderr}"
        )
    });

    eprintln!("[csha-gpu-e2e] per-parameter weight deltas:");
    for (name, before, after, delta) in &deltas {
        eprintln!(
            "  {name:>8}: before={before:>14.6}  after={after:>14.6}  delta={delta:>+14.6}"
        );
    }

    // Finiteness guard — NaN or Inf in any sum means a numerical
    // pathology somewhere in the fused kernel or its epilogue.
    for (name, before, after, _) in &deltas {
        assert!(
            before.is_finite(),
            "[csha-gpu-e2e] BEFORE_{name} = {before} is not finite"
        );
        assert!(
            after.is_finite(),
            "[csha-gpu-e2e] AFTER_{name} = {after} is not finite"
        );
    }

    // The load-bearing assertion: ALL FOUR trainable params must have
    // moved.  A zero delta on any one means either the optimizer step
    // never ran for that slot, the gradient was zero (e.g. a semantic
    // mismatch in the adjoint lowering silently producing a zero grad),
    // or the SGD write never landed in `m.<w>`'s device-backed storage.
    //
    // Tightened from "≥1 param changed" on 2026-04-16 as part of the
    // `w_norm` dgamma runtime fix (PR #71): the original "any changed"
    // was a fallback because Gap I.5's dgamma path was silently zero-ing
    // under mse_loss + all-ones input — the LayerNorm formulation
    // `(x - mean(x)) / std` was being applied to RMSNorm, which yields
    // x_hat = 0 for any constant-valued row.  With the per-op AD now
    // routed through `RmsNormGammaBackward` (pure `x / rms`, no mean
    // subtraction), every trainable param receives a real gradient.
    let stuck: Vec<&String> = deltas
        .iter()
        .filter(|(_, _, _, d)| d.abs() <= 1e-6)
        .map(|(n, _, _, _)| n)
        .collect();
    assert!(
        stuck.is_empty(),
        "[csha-gpu-e2e] trainable param(s) did not move after one step: \
         {stuck:?}.  If `w_norm` is among them, the RMSNorm dgamma \
         adjoint is wrong (LayerNorm formula applied to RMSNorm input). \
         Expected ALL FOUR params (wq, wk, wv, w_norm) to receive a \
         non-zero gradient and update under SGD(lr=0.001, mse_loss, \
         ones input).  Full per-param deltas above."
    );
}

/// CSHA variant: same toy program, but invokes `nsl run --csha auto` so
/// the CSHA boundary scanner + fused-backward dispatcher fire end-to-end
/// (the baseline test above takes the per-op AD path).
///
/// This is the bellwether for the runtime activation step:
///   - `nsl run --csha auto` is now wired (this PR).
///   - `csha_training_config` carries a real `d_model` (this PR).
///   - PR #74's dgamma-extract diagnostic line must show up in stderr.
///
/// **Runtime state (post-dtype-dispatch fix):** the d_model fix + the
/// CPU tensor dispatch widening f16/bf16 to f32 now land SGD updates on
/// `w_norm` + `wq`/`wk`/`wv` without panicking. The test hard-asserts
/// BOTH the d_model=0 and the data_f64-on-non-f64 / copy_data dtype
/// mismatch signatures are gone; any regression in either path turns
/// this test red.
///
/// Invocation:
///   cargo test -p nsl-cli --test csha_gap_gpu_e2e_smoke \
///     --features cuda -- --ignored --nocapture
#[test]
#[ignore = "requires a CUDA-capable GPU; runs the full CSHA fused-backward \
            pipeline end-to-end including SGD writeback on f16 grads."]
fn csha_gap_gpu_e2e_csha_fused_path() {
    let tmp = TempDir::new().expect("tempdir");
    let src_path = tmp.path().join("csha_toy_gpu_e2e_csha.nsl");
    fs::write(&src_path, TOY_SRC).expect("write toy source");

    let mut cmd = Command::cargo_bin("nsl").expect("locate nsl binary");
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("run")
        .arg("--source-ad")
        .arg("--csha")
        .arg("auto")
        .arg("--target")
        .arg("sm_89")
        .arg(&src_path);
    let output = cmd.output().expect("spawn nsl run --csha auto");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    // Compile-time bellwether: BOTH diagnostic lines that prove the
    // CSHA pass + PR #74's dgamma extract emission fired.  These are
    // independent of runtime success and should always appear once
    // `nsl run --csha auto` is honoured by the build pipeline.
    let chains_line = stderr
        .lines()
        .find(|l| l.contains("[csha] csha[auto]:") && l.contains("chains,"));
    assert!(
        chains_line.is_some(),
        "[csha-gpu-e2e-csha] missing `[csha] csha[auto]: ... chains` summary in \
         stderr — `nsl run --csha auto` flag did not reach the CSHA pass.\n\
         stderr:\n{stderr}"
    );
    eprintln!("[csha-gpu-e2e-csha] chains line: {}", chains_line.unwrap());

    let dgamma_line = stderr
        .lines()
        .find(|l| l.contains("CSHA fused backward: emitted dgamma"));
    assert!(
        dgamma_line.is_some(),
        "[csha-gpu-e2e-csha] missing `CSHA fused backward: emitted dgamma` line — \
         PR #74's emission arm did not fire.  Either `chain_varids` resolved to \
         None or EmitFused took the Fallback branch.\n\
         stderr:\n{stderr}"
    );
    eprintln!("[csha-gpu-e2e-csha] dgamma line: {}", dgamma_line.unwrap());

    // Runtime: graceful-skip on the f16/f64 dispatch assertion until
    // the downstream fix lands.  The d_model=0 signature must NOT
    // appear (regression guard).
    if !output.status.success() {
        let driver_missing = stderr.contains("no CUDA devices")
            || stderr.contains("CUDA_ERROR_NOT_INITIALIZED")
            || stderr.contains("CUDA driver not found");
        if driver_missing {
            eprintln!(
                "[csha-gpu-e2e-csha] SKIP — no CUDA driver.\nstderr tail:\n{}",
                stderr.lines().rev().take(5).collect::<Vec<_>>().join("\n")
            );
            return;
        }
        let dmodel_zero_signature = stderr.contains("dim 3: 0 vs 32")
            || stderr.contains("a_shape=[1, 1, 32, 0]");
        assert!(
            !dmodel_zero_signature,
            "[csha-gpu-e2e-csha] REGRESSION — d_model=0 crash signature is back.\n\
             stderr:\n{stderr}"
        );
        let dtype_dispatch_signature = stderr.contains("data_f64() called on non-f64")
            || stderr.contains("nsl_tensor_copy_data: dtype mismatch");
        assert!(
            !dtype_dispatch_signature,
            "[csha-gpu-e2e-csha] REGRESSION — f16/f64 dtype-dispatch crash \
             signature is back.  The CPU tensor elementwise/scalar-mul paths \
             in `crates/nsl-runtime/src/cpu.rs` + \
             `crates/nsl-runtime/src/tensor/arithmetic.rs` must widen \
             f16/bf16 inputs to f32 before computing and pick a promotion \
             rule that keeps SGD's `copy_data(param, ...)` dtype-compatible.\n\
             stderr:\n{stderr}"
        );
        panic!(
            "[csha-gpu-e2e-csha] `nsl run --csha auto` failed with exit {} for an \
             unrecognised reason (neither d_model=0 nor dtype-dispatch signature).\n\n\
             stdout:\n{stdout}\n\nstderr:\n{stderr}",
            output.status
        );
    }

    let deltas = parse_before_after(&stdout).unwrap_or_else(|e| {
        panic!(
            "[csha-gpu-e2e-csha] failed to parse BEFORE/AFTER param sums: {e}\n\
             stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    });

    eprintln!("[csha-gpu-e2e-csha] per-parameter weight deltas:");
    for (name, before, after, delta) in &deltas {
        eprintln!(
            "  {name:>8}: before={before:>14.6}  after={after:>14.6}  delta={delta:>+14.6}"
        );
    }

    for (name, before, after, _) in &deltas {
        assert!(
            before.is_finite(),
            "[csha-gpu-e2e-csha] BEFORE_{name} = {before} is not finite"
        );
        assert!(
            after.is_finite(),
            "[csha-gpu-e2e-csha] AFTER_{name} = {after} is not finite"
        );
    }

    let stuck: Vec<&String> = deltas
        .iter()
        .filter(|(_, _, _, d)| d.abs() <= 1e-6)
        .map(|(n, _, _, _)| n)
        .collect();
    assert!(
        stuck.is_empty(),
        "[csha-gpu-e2e-csha] trainable param(s) did not move after one CSHA step: \
         {stuck:?}.  Expected ALL FOUR params (wq, wk, wv, w_norm) to receive \
         a non-zero gradient under the fused backward path.  Full per-param \
         deltas above."
    );
}
