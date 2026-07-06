//! GPU gradient-accumulation e2e — the silent-frozen-params trap.
//!
//! # What this test proves
//!
//! Pre-fix, single-GPU training with `grad_accumulation > 1` and any
//! optimizer that FASE plans as `FullBuffer` (lion / muon / soap /
//! unknown) was SILENTLY WRONG: the micro-batch loop calls
//! `nsl_grad_accumulate_add(accum_buf, grad, n)` whose return code is
//! discarded by codegen, and the pre-fix runtime returned -1 for any
//! GPU-resident tensor. Accumulation buffers stayed zero, the optimizer
//! stepped on zero gradients, loss printed normally — and params never
//! moved. This test drives the full CLI pipeline (parse -> semantic ->
//! codegen -> link -> spawn -> CUDA -> accumulate -> SOAP step ->
//! writeback) and hard-asserts the trainable param actually changes.
//!
//! With the zero.rs fix (nsl_grad_accumulate_add migrates + delegates to
//! the GPU-safe nsl_tensor_add_inplace; nsl_grad_zero device-memsets),
//! the GPU run converges to the same value as the CPU reference run of
//! the identical script (verified manually: 4.0 -> 3.93921 on both).
//!
//! # Why SOAP and not Lion
//!
//! FASE plans FullBuffer for lion AND for any optimizer it does not
//! recognize (muon/soap -> `FaseOptimizer::Unknown`). The stdlib lion
//! step currently fails semantic analysis (`sign` not found in scope —
//! pre-existing, tracked separately), so SOAP is the runnable optimizer
//! that exercises the same FullBuffer accumulate path.
//!
//! # Why `--source-ad`
//!
//! Tape-AD on this GPU toy leaves params frozen for EVERY optimizer,
//! including the known-good SGD/Deferred path (pre-existing, orthogonal
//! to zero.rs — the accumulate FFI never receives a non-zero gradient).
//! Source-AD is the production configuration for GPU training (see
//! models/coder-rl/train_sft.nsl) and is what reaches the FullBuffer
//! accumulate path with real device-resident gradients.

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

/// Toy train program: GPU-resident 2x2 linear model, SOAP optimizer
/// (FASE FullBuffer), grad_accumulation=2, 4 epochs = 4 micro-batches
/// = 2 optimizer applications. `x = ones`, `y = zeros` gives a nonzero
/// mse gradient every step, so a working accumulate + step chain MUST
/// move `w` away from its all-ones init.
const TOY_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([2, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
m.to(cuda)
let x = ones([4, 2])
let y = zeros([4, 2])

print("BEFORE_w")
print(sum(m.w))

train(model = m, epochs = 4, grad_accumulation = 2):
    optimizer: SOAP(lr = 0.01, beta1 = 0.9, beta2 = 0.99)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)

print("AFTER_w")
print(sum(m.w))
"#;

/// Extract the numeric value printed after a marker line. GPU scalar
/// sums print as `tensor([4.0])`, CPU sums as a bare `4.0`; accept both.
/// `[gpu-mem]`-style diagnostics interleave with markers and are skipped.
fn value_after_marker(stdout: &str, marker: &str) -> Result<f64, String> {
    let lines: Vec<&str> = stdout.lines().collect();
    let idx = lines
        .iter()
        .position(|l| l.trim() == marker)
        .ok_or_else(|| format!("marker `{marker}` not found in stdout"))?;
    for line in lines.iter().skip(idx + 1) {
        let t = line.trim();
        if t.is_empty() || t.starts_with('[') {
            continue; // diagnostic like `[gpu-mem] step=0 ...`
        }
        let inner = t
            .strip_prefix("tensor([")
            .and_then(|s| s.strip_suffix("])"))
            .unwrap_or(t);
        return inner
            .parse::<f64>()
            .map_err(|e| format!("value line after `{marker}` is `{t}`: {e}"));
    }
    Err(format!("no value line after marker `{marker}`"))
}

/// Invocation:
///   cargo test -p nsl-cli --test zero_grad_accum_gpu_e2e \
///     --features cuda -- --ignored --nocapture
#[test]
#[ignore = "requires a CUDA-capable GPU"]
fn gpu_grad_accumulation_full_buffer_updates_params() {
    let tmp = TempDir::new().expect("tempdir");
    let src_path = tmp.path().join("zero_grad_accum_trap.nsl");
    fs::write(&src_path, TOY_SRC).expect("write toy source");

    let mut cmd = Command::cargo_bin("nsl").expect("locate nsl binary");
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("run")
        .arg("--source-ad")
        .arg(&src_path);
    let output = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        let driver_missing = stderr.contains("no CUDA devices")
            || stderr.contains("CUDA_ERROR_NOT_INITIALIZED")
            || stderr.contains("CUDA driver not found");
        if driver_missing {
            eprintln!(
                "[zero-grad-accum-e2e] SKIP — no CUDA driver.\nstderr tail:\n{}",
                stderr.lines().rev().take(5).collect::<Vec<_>>().join("\n")
            );
            return;
        }
        panic!(
            "[zero-grad-accum-e2e] `nsl run` failed with exit {}.\n\n\
             stdout:\n{stdout}\n\nstderr:\n{stderr}",
            output.status
        );
    }

    let before = value_after_marker(&stdout, "BEFORE_w").unwrap_or_else(|e| {
        panic!(
            "[zero-grad-accum-e2e] {e}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    });
    let after = value_after_marker(&stdout, "AFTER_w").unwrap_or_else(|e| {
        panic!(
            "[zero-grad-accum-e2e] {e}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    });
    let delta = after - before;
    eprintln!(
        "[zero-grad-accum-e2e] sum(w): before={before:.9} after={after:.9} \
         delta={delta:+.9}"
    );

    assert!(before.is_finite(), "BEFORE_w = {before} is not finite");
    assert!(after.is_finite(), "AFTER_w = {after} is not finite");

    // The load-bearing assertion. Pre-fix this fails with delta == 0:
    // nsl_grad_accumulate_add returned -1 (discarded) for GPU tensors,
    // the FullBuffer accumulator stayed zero, and SOAP stepped on zero
    // gradients — silent frozen-params training.
    assert!(
        delta.abs() > 1e-6,
        "[zero-grad-accum-e2e] param frozen: sum(w) {before} -> {after} \
         after 4 micro-batches (accumulation=2) of nonzero-gradient MSE \
         training. The GPU FullBuffer accumulate chain \
         (nsl_grad_accumulate_add / nsl_grad_zero in \
         crates/nsl-runtime/src/zero.rs) is silently dropping gradients \
         again.\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
    );
}
