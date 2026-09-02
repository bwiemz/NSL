//! C1 (error model): how `nsl build` reports a codegen failure.
//!
//! Before this, `CodegenError` was a bare message with no location, its
//! `Display` carried a `codegen error:` prefix that every CLI site prefixed
//! AGAIN (`codegen error: codegen error: …`), and two refusals — the
//! `--wggo-memory-budget` floor and the CPDT weight-map validation — were
//! `eprintln!` + `process::exit(1)` from inside the nsl-codegen LIBRARY, so
//! no library caller could observe them and no test could pin their text
//! without spawning a process.
//!
//! Now an error raised anywhere under a statement leaves codegen carrying
//! the innermost statement/expression span, and the CLI renders it through
//! the same `SourceMap` the frontend's diagnostics use: `error: <message>`,
//! `file:line:col`, the excerpt with a caret, then any notes. The two
//! refusals are returned `Err`s that reach the same renderer, pointing at
//! the train header's `model = …` argument. Errors with no location keep the
//! plain `codegen error: <message>` line (pinned by the unit tests in
//! `nsl_codegen::error`); nothing prints the prefix twice.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// A located error goes through the diagnostic renderer and never through
/// the plain `codegen error: <message>` fallback — so the prefix must be
/// absent altogether, not merely not doubled (which would also hold for a
/// plain line that lost its span, and prove nothing).
fn rendered_not_plain() -> impl Predicate<str> {
    predicate::str::contains("codegen error:").not()
}

#[test]
fn a_spanned_codegen_error_renders_file_line_col_and_excerpt() {
    // `==` inside a `kernel` block: refused by the PTX kernel compiler,
    // which knows the kernel name but nothing about where in the file it
    // is — the span comes from the statement/expression dispatchers.
    let example = workspace_root().join("examples/m17_kernel_eq_condition_error.nsl");
    let tmp = TempDir::new().unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&example)
        .arg("--emit-obj")
        .arg("-o")
        .arg(tmp.path().join("out.o"));
    cmd.assert()
        .code(1)
        .stderr(predicate::str::contains(
            "error: kernel 'bad_eq': binary operator '==' is not supported in kernel code",
        ))
        // The location line names the file, and the caret sits on the
        // `i == 0` condition (line 7, column 8), not on the whole `if`.
        .stderr(predicate::str::contains("m17_kernel_eq_condition_error.nsl:7:8"))
        .stderr(predicate::str::contains("if i == 0:"))
        .stderr(rendered_not_plain());
}

/// A model WGGO's fp16-moment floor prices at ~32 MiB, against a 1 MiB cap.
const OVERSIZED: &str = r#"model Linear:
    w: Tensor = ones([2048, 2048])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2048])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let out = m.forward(x)
        let loss = sum(out)
"#;

#[test]
fn an_infeasible_wggo_budget_is_a_rendered_refusal_that_writes_nothing() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("big.nsl");
    let out_path = tmp.path().join("big_out");
    fs::write(&src_path, OVERSIZED).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        .arg("--wggo")
        .arg("full")
        .arg("--wggo-memory-budget")
        .arg("1");
    cmd.assert()
        .code(1)
        .stderr(predicate::str::contains(
            "error: --wggo-memory-budget of 1048576 bytes (1 MiB) is infeasible",
        ))
        .stderr(predicate::str::contains("even at the fp16-moment floor"))
        .stderr(predicate::str::contains("Raise the budget or shrink the model"))
        // Points at the header's `model = m` (line 10, column 7), which is
        // what the budget was checked against — not the whole train block.
        .stderr(predicate::str::contains("big.nsl:10:7"))
        .stderr(predicate::str::contains("train(model = m, epochs = 1):"))
        .stderr(rendered_not_plain());
    assert!(
        !out_path.exists(),
        "a refused compile must leave no artifact behind"
    );
}

#[test]
fn a_wrong_cpdt_checkpoint_is_a_rendered_refusal_with_the_plan_caveat_as_a_note() {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::collections::HashMap;

    // The moment-precision gate's wrong checkpoint (names that join none
    // of the fixture's `blocks.N` layers), but WITH `--wggo full` so the
    // pre-pass offers a plan and the weight map is validated against it —
    // the refusal that used to `process::exit(1)` from inside `stmt.rs`.
    let tmp = TempDir::new().unwrap();
    let wrong = tmp.path().join("wrong_model.safetensors");
    let mut raw: HashMap<String, Vec<u8>> = HashMap::new();
    for name in ["foo.0.w", "foo.1.w"] {
        raw.insert(
            name.to_string(),
            (0..64 * 64).flat_map(|_| 1e-4_f32.to_le_bytes()).collect(),
        );
    }
    let views: HashMap<String, TensorView<'_>> = raw
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                TensorView::new(Dtype::F32, vec![64, 64], v.as_slice()).unwrap(),
            )
        })
        .collect();
    fs::write(&wrong, serialize(&views, None).unwrap()).unwrap();

    let fixture = workspace_root().join("crates/nsl-codegen/tests/fixtures/cpdt_precision_fp16.nsl");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&fixture)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(tmp.path().join("out.o"))
        .arg("--wggo")
        .arg("full")
        .arg("--weights")
        .arg(&wrong)
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("2");
    cmd.assert()
        .code(1)
        .stderr(predicate::str::contains(
            "error: CPDT weight validation: weight map does not match model declaration.",
        ))
        .stderr(predicate::str::contains("Missing layers (8 of 8): blocks.0, blocks.1"))
        // The train header's `model = m`.
        .stderr(predicate::str::contains("cpdt_precision_fp16.nsl:41:7"))
        .stderr(predicate::str::contains("train(model = m, epochs = 2"))
        // The which-plan caveat is a note after the excerpt, not part of
        // the message.
        .stderr(predicate::str::contains(
            "= checked against the WGGO plan available at this point",
        ))
        .stderr(rendered_not_plain());
}
