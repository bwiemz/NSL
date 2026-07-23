//! PCA Stage B: `scaled_dot_product_attention_masked` — additive-mask
//! attention for packed sequences.
//!
//! The masked form deliberately does NOT use the fused
//! `PrimalOp::ScaledDotProductAttention` op: its
//! `FlashAttentionBackwardExtract` backward only understands a `causal`
//! flag, so an additive mask through that kernel would silently produce
//! wrong dQ/dK/dV. Instead, source AD extracts the masked call as
//! DECOMPOSED primitives (transpose/matmul/mul/add/softmax/matmul) whose
//! standard adjoints handle the mask exactly; the expr (tape) path lowers
//! the same naive chain via traced runtime calls.
//!
//! 1. Forward oracle: a packed 2-doc sequence under a block-diagonal
//!    additive mask must equal the two documents attended separately.
//! 2. Backward differential: a ZERO mask makes the masked form
//!    mathematically identical to plain non-causal SDPA — trained weights
//!    must match the fused-op fixture's, cross-validating the decomposed
//!    adjoint chain against the FlashAttentionBackwardExtract reference.

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

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// Run `nsl run <fixture> [extra...]` in `workdir`; return (stdout, stderr).
fn nsl_run(fixture: &Path, workdir: &Path, extra: &[&str]) -> (String, String) {
    let root = workspace_root();
    let mut cmd = std::process::Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli"]);
    if cfg!(feature = "cuda") {
        // Feature-compatibility guard: an unfeatured `cargo run -p nsl-cli`
        // REBUILDS AND REPLACES target/debug/nsl as a non-CUDA binary while
        // the workspace suite runs, breaking every concurrently-running test
        // that spawns that path (phantom "CUDA support not compiled"
        // failures inside their trained programs). Forward cuda so the
        // spawned build matches the suite's binary (and is usually a no-op).
        cmd.args(["--features", "cuda"]);
    }
    cmd.args(["--", "run"])
        .arg(fixture)
        .current_dir(workdir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    for a in extra {
        cmd.arg(a);
    }
    let out = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "nsl run failed on {fixture:?} ({extra:?}):\n{stderr}"
    );
    (stdout, stderr)
}

/// Packed masked attention == per-document attention, exactly. The fixture
/// prints sum((packed - concat(per_doc))^2); block-diagonal -1e9 masking
/// drives cross-document attention weights to exactly 0 after softmax at
/// these magnitudes, so the difference is 0.0, not merely small.
#[test]
fn packed_masked_forward_equals_per_doc_attention() {
    let tmp = TempDir::new().unwrap();
    let (stdout, stderr) = nsl_run(
        &fixture_path("pca_masked_sdpa_forward_oracle.nsl"),
        tmp.path(),
        &[],
    );
    let val: f64 = stdout
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .next_back()
        .unwrap_or_else(|| panic!("no scalar in stdout:\n{stdout}\n{stderr}"));
    assert!(
        val < 1e-12,
        "packed vs per-doc attention diverged: sum(diff^2) = {val}"
    );
}

fn run_and_read(fixture: &str, ckpt: &str, extra: &[&str]) -> std::collections::HashMap<String, Vec<f32>> {
    let tmp = TempDir::new().unwrap();
    let (_, stderr) = nsl_run(&fixture_path(fixture), tmp.path(), extra);
    let p = tmp.path().join(ckpt);
    assert!(p.exists(), "missing checkpoint {ckpt}:\n{stderr}");
    read_nslm(&p).expect("read nslm")
}

/// Source-AD path: decomposed masked backward vs fused-op flash-extract
/// backward, identical zero-mask math. Weights must agree tightly (f64
/// op-order differences only).
#[test]
fn zero_mask_train_matches_plain_sdpa_source_ad() {
    let masked = run_and_read(
        "pca_masked_sdpa_zero_mask_train.nsl",
        "masked_sdpa_out.nslm",
        &["--source-ad"],
    );
    let plain = run_and_read(
        "pca_plain_sdpa_noncausal_train.nsl",
        "plain_sdpa_out.nslm",
        &["--source-ad"],
    );
    for name in ["wq", "wk", "wv"] {
        let a = masked.get(name).or_else(|| masked.get(&format!("m.{name}")))
            .unwrap_or_else(|| panic!("{name} missing: {:?}", masked.keys()));
        let b = plain.get(name).or_else(|| plain.get(&format!("m.{name}")))
            .unwrap_or_else(|| panic!("{name} missing: {:?}", plain.keys()));
        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(
            d <= 1e-6,
            "{name} diverged {d} between decomposed-masked and fused-plain backward"
        );
        // Training must have actually moved the weights (guards a vacuous
        // pass where gradients silently vanish on both sides).
        let init = match name {
            "wq" => 1.0f32,
            "wk" => 0.5,
            _ => 0.25,
        };
        assert!(
            a.iter().any(|v| (v - init).abs() > 1e-4),
            "{name} never moved from init — gradients did not flow"
        );
    }
}

/// Tape path: the same differential without --source-ad (the expr naive
/// lowering is traced per-primitive, so the tape backward differentiates
/// each step). Pins that both AD modes agree on the masked form.
#[test]
fn zero_mask_train_matches_plain_sdpa_tape() {
    let masked = run_and_read(
        "pca_masked_sdpa_zero_mask_train.nsl",
        "masked_sdpa_out.nslm",
        &[],
    );
    let plain = run_and_read(
        "pca_plain_sdpa_noncausal_train.nsl",
        "plain_sdpa_out.nslm",
        &[],
    );
    for name in ["wq", "wk", "wv"] {
        let a = masked.get(name).or_else(|| masked.get(&format!("m.{name}")))
            .unwrap_or_else(|| panic!("{name} missing: {:?}", masked.keys()));
        let b = plain.get(name).or_else(|| plain.get(&format!("m.{name}")))
            .unwrap_or_else(|| panic!("{name} missing: {:?}", plain.keys()));
        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(
            d <= 1e-6,
            "{name} diverged {d} between masked and plain tape backward"
        );
    }
}
