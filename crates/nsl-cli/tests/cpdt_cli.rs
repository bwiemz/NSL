//! Task 6: CPDT CLI flag behaviour tests.
//!
//! Formalizes the three failure / implication paths smoke-tested by hand in
//! Task 5:
//!
//!   1. `--cpdt-report` alone (no `--cpdt`) implies `--cpdt` in full mode, and
//!      the report header + Defaults Assumed footer both render to stdout.
//!   2. `--cpdt` without `--cpdt-num-gpus` fails fast with a clear stderr.
//!   3. `--cpdt --cpdt-num-gpus 0` fails fast with a clear stderr.
//!
//! Validation for (2) and (3) fires before any file is read, so a fixture
//! path need only be syntactically valid. (1) requires a real `@train`
//! source that WGGO can plan over, so we reuse the same inline fixture the
//! WRGA report CLI tests use, which has a one-layer Linear model + SGD train
//! block — enough for WGGO to produce an `AppliedPlan` and for CPDT to turn
//! that into a `CpdtPlan`. Since the weight-aware gate (PR #88/#90/#91) now
//! requires weights for `--cpdt full`, (1) also writes a tiny safetensors
//! fixture and passes it via `--weights`.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
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

/// Write a minimal valid safetensors file containing one f32 tensor named
/// `blocks.0.attn.wq.weight`. Mirrors the helper in `cpdt_weights_cli.rs`.
fn write_tiny_safetensors(path: &Path) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    let bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    let view = TensorView::new(Dtype::F32, vec![1], bytes.as_slice()).unwrap();
    views.insert("blocks.0.attn.wq.weight".to_string(), view);
    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}

/// Minimal `@train`-bearing fixture. Mirrors the SRC used by
/// `wrga_report_cli.rs` so we know it compiles end-to-end today.
const SRC: &str = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;

#[test]
fn bare_cpdt_report_enables_full_mode() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let weights_path = tmp.path().join("weights.safetensors");
    let out_path = tmp.path().join("t_out");
    fs::write(&src_path, SRC).unwrap();
    write_tiny_safetensors(&weights_path);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad") // WGGO only fires on the source-AD lowering path
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        // --wggo auto so WGGO produces an AppliedPlan, which CPDT then
        // consumes. Without this, cpdt_plan stays None and no report prints.
        .arg("--wggo")
        .arg("auto")
        .arg("--cpdt-report")
        .arg("--cpdt-num-gpus")
        .arg("4")
        // Weight-aware gate (PR #88/#90/#91) requires weights for full mode;
        // supply a minimal safetensors so the decision table lets the build
        // proceed and CPDT can produce its report.
        .arg("--weights")
        .arg(&weights_path);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== CPDT Training Plan ==="))
        .stdout(predicate::str::contains("Mode: full"))
        .stdout(predicate::str::contains("=== Defaults Assumed ==="));
}

#[test]
fn missing_num_gpus_fast_fails() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build").arg(&src_path).arg("--cpdt");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--cpdt requires --cpdt-num-gpus"));
}

#[test]
fn zero_num_gpus_fast_fails() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--cpdt")
        .arg("--cpdt-num-gpus")
        .arg("0");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("must be >= 1"));
}
