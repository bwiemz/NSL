//! Numerical validation for FASE Deferred mode (CFTP item #2).
//!
//! This test compiles and runs a tiny NSL fixture with SGD + grad_accumulation=4,
//! reads the saved `.nslm` checkpoint, and compares the final `w` tensor against
//! an exact Rust SGD reference.

mod common {
    include!("common/mod.rs");
}

use common::nslm_reader::read_nslm;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

/// Workspace root: CARGO_MANIFEST_DIR/../..
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// Run `nsl run <fixture>` with cwd set to `workdir`.
///
/// `cargo run` is invoked with `--manifest-path` pointing at the workspace
/// Cargo.toml so cargo can find the project from any cwd.  `workdir` is
/// passed as the cwd so that `model_save(m, "file.nslm")` lands there.
/// `NSL_STDLIB_PATH` is set to the workspace `stdlib/` so that imports like
/// `from nsl.nn.losses import mse_loss` resolve correctly.
///
/// Panics on non-zero exit.
fn nsl_run(fixture: &Path, workdir: &Path) {
    let root = workspace_root();
    let cargo_toml = root.join("Cargo.toml");
    let stdlib_path = root.join("stdlib");

    let status = std::process::Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(&cargo_toml)
        .args(["-p", "nsl-cli", "--", "run"])
        .arg(fixture)
        .current_dir(workdir)
        .env("NSL_STDLIB_PATH", &stdlib_path)
        .status()
        .expect("failed to spawn cargo run nsl-cli");

    assert!(
        status.success(),
        "nsl run failed on {:?}",
        fixture
    );
}

/// Rust reference for plain SGD with grad_accumulation=4 and constant inputs
/// across all 4 micro-batches (no dataloader — step runs once per epoch, so
/// epochs=4 accumulates 4 micro-batches before the optimizer fires).
///
/// Fixture init values:
///   w = ones([2,1])  → [[1.0], [1.0]]
///   x = ones([4,2])  → all rows = [1.0, 1.0]
///   y = zeros([4,1]) → all targets = [0.0]
///
/// Forward:  pred = x @ w  → [[2.0], [2.0], [2.0], [2.0]]
/// Residual: r = pred - y  → [[2.0], [2.0], [2.0], [2.0]]
/// MSE grad: g = (2/N) * x.T @ r  where N=4
///           x.T @ r = [[8.0], [8.0]]  →  g = [[4.0], [4.0]]
/// SGD step: w_new = w - lr * g  → [[0.96], [0.96]]
///
/// Because all 4 micro-batches see the same (x, y), the FASE Deferred
/// mean(g_k) = g, so the accumulated result must exactly match a single
/// plain SGD step.
fn sgd_reference(
    w_init: &[f32; 2],
    x: &[[f32; 2]; 4],
    y: &[[f32; 1]; 4],
    lr: f32,
) -> [f32; 2] {
    // Forward: pred[i] = sum_j x[i][j] * w[j]
    let mut pred = [0.0_f32; 4];
    for i in 0..4 {
        for j in 0..2 {
            pred[i] += x[i][j] * w_init[j];
        }
    }

    // Residual: r[i] = pred[i] - y[i]
    let mut r = [0.0_f32; 4];
    for i in 0..4 {
        r[i] = pred[i] - y[i][0];
    }

    // MSE gradient wrt w: g[j] = (2/N) * sum_i x[i][j] * r[i]
    let n = 4.0_f32;
    let mut g = [0.0_f32; 2];
    for j in 0..2 {
        for i in 0..4 {
            g[j] += x[i][j] * r[i];
        }
        g[j] *= 2.0 / n;
    }

    // SGD parameter update
    [w_init[0] - lr * g[0], w_init[1] - lr * g[1]]
}

#[test]
fn sgd_exact_equivalence() {
    let tmp = TempDir::new().expect("tempdir");

    nsl_run(
        &fixture_path("fase_deferred_sgd_equivalence.nsl"),
        tmp.path(),
    );

    let checkpoint = tmp.path().join("sgd_out.nslm");
    assert!(
        checkpoint.exists(),
        "expected checkpoint at {:?} — model_save did not produce the file",
        checkpoint
    );

    let tensors = read_nslm(&checkpoint).expect("read nslm");

    let w_compiled = tensors
        .get("w")
        .or_else(|| tensors.get("m.w"))
        .unwrap_or_else(|| {
            panic!(
                "w tensor not in checkpoint; available keys: {:?}",
                tensors.keys().collect::<Vec<_>>()
            )
        });

    assert_eq!(
        w_compiled.len(),
        2,
        "w should have 2 elements (shape [2,1]), got {}",
        w_compiled.len()
    );

    // Reference inputs — must match the fixture exactly:
    //   w = ones([2,1]), x = ones([4,2]), y = zeros([4,1]), lr = 0.01
    let w_init = [1.0_f32, 1.0_f32];
    let x = [[1.0_f32, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
    let y = [[0.0_f32]; 4];
    let w_ref = sgd_reference(&w_init, &x, &y, 0.01);

    println!("SGD equivalence check:");
    println!("  compiled w[0]={} w[1]={}", w_compiled[0], w_compiled[1]);
    println!("  reference w[0]={} w[1]={}", w_ref[0], w_ref[1]);

    for i in 0..2 {
        let diff = (w_compiled[i] - w_ref[i]).abs();
        let scale = w_ref[i].abs().max(1.0);
        assert!(
            diff / scale < 1e-5,
            "SGD θ[{}] diverged: compiled={} reference={} rel_err={}",
            i,
            w_compiled[i],
            w_ref[i],
            diff / scale
        );
    }
}
