//! `@pca(strategy = per_document)` is refused, by `nsl run` and `nsl check`.
//!
//! The decorator used to be accepted and could not run. Admission built a
//! per-document attention kernel during kernel synthesis and printed a
//! `[pca-per-doc] admitted` line; every attention call site the compiler
//! emits then passed that kernel a document count of 0; the launcher
//! refused the count and returned an error code; and the call site turned
//! the code into a trap.
//!
//! This file used to hold a test of the plumbing only: it looked for the
//! `[pca-per-doc]` marker on stderr and, in its own words, deliberately did
//! not assert the exit status, because its fixture did not compile. It
//! passed on a program that could not run. What it checks now is the
//! behaviour a user sees: a failure status and a message that names the
//! decorator, on the command that is run.
//!
//! `@pca(strategy = off)` is refused for the plainer reason that nothing
//! read it, so it did not turn packing off.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn repo_root() -> PathBuf {
    // crates/nsl-cli -> crates -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// An `@flash_attention` model method, a packed dataset and a decorated
/// train block: the combination that used to reach per-document admission.
fn program(decorator: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

dataset PackedCorpus("docs"):
    packing = true
    max_sequence_length = 64
    mean_doc_length = 16

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
let x = ones([32, 32])

{decorator}
train(model = m, epochs = 1):
    data:
        source = PackedCorpus
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, x)

print("RAN TO THE END")
"#
    )
}

fn nsl(subcommand: &[&str], source: &str) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pca_gate.nsl");
    std::fs::write(&path, source).unwrap();
    Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .current_dir(dir.path())
        .args(subcommand)
        .arg(&path)
        .output()
        .unwrap()
}

#[test]
fn per_document_is_refused_at_compile_time() {
    for subcommand in [&["check"][..], &["run", "--target", "cpu"][..]] {
        let out = nsl(subcommand, &program("@pca(strategy=per_document)"));
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "nsl {subcommand:?} accepted @pca(strategy=per_document):\n{stdout}\n{stderr}"
        );
        assert!(
            stderr.contains("@pca(strategy = per_document) is not supported"),
            "nsl {subcommand:?}: the failure does not name the decorator:\n{stderr}"
        );
        assert!(
            !stdout.contains("RAN TO THE END"),
            "nsl {subcommand:?}: the program ran"
        );
    }
}

#[test]
fn off_is_refused_at_compile_time() {
    let out = nsl(&["check"], &program("@pca(strategy=off)"));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "nsl check accepted @pca(strategy=off):\n{stderr}"
    );
    assert!(
        stderr.contains("@pca(strategy = off) is not supported"),
        "{stderr}"
    );
}

/// The strategies that name what happens anyway stay accepted by the
/// checker, so the refusals above are about those two spellings and not
/// about the decorator.
#[test]
fn the_default_strategies_are_still_accepted() {
    for decorator in [
        "@pca",
        "@pca(strategy=auto)",
        "@pca(strategy=segment_id)",
        "",
    ] {
        let out = nsl(&["check"], &program(decorator));
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "nsl check refused the program with {decorator:?}:\n{stderr}"
        );
    }
}
