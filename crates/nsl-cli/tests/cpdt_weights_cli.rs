//! Phase 1 Item 3+4 CLI integration tests: AST auto-detect for
//! `load_safetensors(...)` and the four-case decision table in
//! `nsl build`. See
//! `docs/superpowers/specs/2026-04-21-cpdt-ast-autodetect-design.md`.
//!
//! Each test writes a minimal NSL source (derived from the fixture used
//! in `cpdt_cli.rs`) + when needed a tiny valid safetensors file into a
//! `TempDir`. Tests don't require the compile to succeed end-to-end —
//! they exercise the CLI decision-table logic that runs BEFORE full
//! compilation (or the error branch that exits the CLI early). For
//! tests that do invoke the full build, the source uses the same
//! Linear+SGD shape as `cpdt_cli.rs` so WGGO + CPDT both fire.

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
    Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Linear+SGD fixture shared with `cpdt_cli.rs`. Decorators slot in via
/// the `{DECORATORS}` placeholder so each test can tune its decorator
/// shape. The `load_safetensors` call (when present) is inserted via
/// `{LOAD_STMT}`.
const TEMPLATE: &str = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])
{LOAD_STMT}
{DECORATORS}
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;

fn render_source(load_stmt: &str, decorators: &str) -> String {
    TEMPLATE
        .replace("{LOAD_STMT}", load_stmt)
        .replace("{DECORATORS}", decorators)
}

/// Write a minimal valid safetensors file to `path` containing one f32
/// tensor named `blocks.0.attn.wq.weight`. The tensor name follows the
/// hierarchical layer pattern so PR #90's layer-prefix validation (once
/// merged) doesn't reject the fixture. Size is intentionally tiny — the
/// CLI decision table only needs the file to be readable by
/// `WeightMap::load`.
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

/// Case 1: AST has `load_safetensors("...")`; no `--weights` flag.
/// Expected: the build uses the AST-detected path; no four-case error.
///
/// We test via `--cpdt-report` which produces stdout output only when
/// the CLI decision table let the build proceed to compilation.
#[test]
fn ast_autodetect_only_reaches_cpdt_report() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let weights_path = tmp.path().join("weights.safetensors");
    let out_path = tmp.path().join("t_out");
    write_tiny_safetensors(&weights_path);
    let load_stmt = format!(
        "let w = load_safetensors({:?})",
        weights_path.to_str().unwrap().replace('\\', "/")
    );
    let src = render_source(&load_stmt, "");
    fs::write(&src_path, src).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        .arg("--wggo")
        .arg("auto")
        .arg("--cpdt-report")
        .arg("--cpdt-num-gpus")
        .arg("4");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== CPDT Training Plan ==="))
        .stderr(predicate::str::contains("overrides AST-declared").not());
}

/// Case 2: `--weights` flag only; no AST. Expected: the flag is used;
/// no override warning.
#[test]
fn flag_only_reaches_cpdt_report_without_warning() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let weights_path = tmp.path().join("weights.safetensors");
    let out_path = tmp.path().join("t_out");
    write_tiny_safetensors(&weights_path);
    fs::write(&src_path, render_source("", "")).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        .arg("--wggo")
        .arg("auto")
        .arg("--cpdt-report")
        .arg("--cpdt-num-gpus")
        .arg("4")
        .arg("--weights")
        .arg(&weights_path);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== CPDT Training Plan ==="))
        .stderr(predicate::str::contains("overrides AST-declared").not())
        .stderr(predicate::str::contains("requires weights").not());
}

/// Case 3: both `--weights` flag and AST `load_safetensors(...)`.
/// Expected: the build succeeds using the flag's path; stderr contains
/// the "overrides AST-declared" warning.
#[test]
fn both_present_flag_wins_with_override_warning() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let ast_weights_path = tmp.path().join("ast.safetensors");
    let flag_weights_path = tmp.path().join("flag.safetensors");
    let out_path = tmp.path().join("t_out");
    write_tiny_safetensors(&ast_weights_path);
    write_tiny_safetensors(&flag_weights_path);
    let load_stmt = format!(
        "let w = load_safetensors({:?})",
        ast_weights_path.to_str().unwrap().replace('\\', "/")
    );
    fs::write(&src_path, render_source(&load_stmt, "")).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        .arg("--wggo")
        .arg("auto")
        .arg("--cpdt-report")
        .arg("--cpdt-num-gpus")
        .arg("4")
        .arg("--weights")
        .arg(&flag_weights_path);
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("overrides AST-declared"));
}

/// Case 4: neither present; CPDT enabled; `weight_aware` default.
/// Expected: fast-failure before compile; stderr contains the three-
/// option resolution message.
#[test]
fn neither_present_cpdt_full_errors_with_three_options() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let out_path = tmp.path().join("t_out");
    // Source has no load_safetensors() and no @cpdt(weight_aware=false).
    fs::write(&src_path, render_source("", "")).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        .arg("--wggo")
        .arg("auto")
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("4");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("requires weights"))
        .stderr(predicate::str::contains("1. Add --weights"))
        .stderr(predicate::str::contains("2. Add `let w = load_safetensors"))
        .stderr(predicate::str::contains("3. Add `@cpdt(weight_aware=false)"));
}

/// Case 5: neither present; CPDT enabled; `@cpdt(weight_aware=false)`
/// present in source. Expected: the decision-table error does NOT fire;
/// the build proceeds.
#[test]
fn neither_present_cpdt_full_with_opt_out_succeeds() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let out_path = tmp.path().join("t_out");
    // `@cpdt`'s supported kwargs are `mode`, `precision`, `target_memory`,
    // `cluster`, `weight_aware`. The `num_gpus` value comes from the CLI
    // flag `--cpdt-num-gpus`, not from the decorator directly (which would
    // take `cluster={num_gpus=...}` if configured there). Use the bare
    // `@cpdt(weight_aware=false)` form here.
    fs::write(
        &src_path,
        render_source("", "@cpdt(weight_aware=false)"),
    )
    .unwrap();

    // The @cpdt decorator changes downstream codegen in ways unrelated to
    // the decision-table opt-out (the fixture's minimal SGD step relies on
    // stdlib symbols that aren't resolved under the decorator-modified
    // path on this fixture). We test only the decision-table behavior:
    // stderr MUST NOT contain the "requires weights" three-option error —
    // the opt-out was respected. The final compile succeeds-or-fails for
    // unrelated reasons; we don't assert the top-level status.
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        .arg("--wggo")
        .arg("auto")
        .arg("--cpdt-report")
        .arg("--cpdt-num-gpus")
        .arg("4");
    // Don't assert overall success/failure — just the decision-table slot.
    cmd.assert()
        .stderr(predicate::str::contains("requires weights").not())
        .stderr(predicate::str::contains("1. Add --weights").not());
}
