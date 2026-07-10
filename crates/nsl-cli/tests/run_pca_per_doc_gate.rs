//! Regression gate for roadmap item 2.2: `nsl run` must deliver
//! `@pca(strategy=per_document)` to the per-document CTA admission gate.
//!
//! # History
//!
//! Pre-PR-#281, `nsl run` passed an empty `pca_user_strategies` list to codegen,
//! so `@pca(strategy=per_document)` was a silent no-op under `nsl run` (while
//! `nsl build` plumbed it). #281 fixed both commands by overwriting the field
//! from semantic analysis inside the shared `run_build_single`/`run_build_multi`
//! (`build/normal.rs:86`/`:349`), which `nsl run` reaches via `run_run` →
//! `run_build_inner`. This test pins that plumbing on the RUN path specifically:
//! the `[pca-per-doc]` stderr marker is printed by the admission gate inside
//! `maybe_synthesize_csha_training_ptx` (compiler/kernel.rs) — in BOTH its
//! admitted and refused branches — and the gate is only reached at all when the
//! strategy list arriving from the CLI is non-empty. Marker present ⇒ the run
//! path delivered the strategies; marker absent ⇒ the pre-#281 silent no-op
//! regressed.
//!
//! # Why exit status is deliberately NOT asserted
//!
//! The admission gate fires during kernel synthesis, well before train-block
//! lowering and execution. This fixture's shape (an `@flash_attention` method +
//! `dataset` block + non-accumulating train block) currently fails LATER in
//! compilation with `undefined function 'nsl_optim_adamw__adamw_step'` — a
//! pre-existing optimizer-resolution gap for this program shape, unrelated to
//! PCA plumbing (the same optimizer resolves fine in e.g.
//! `fase_deferred_adamw_equivalence.nsl`, and with `grad_accumulation` present
//! this fixture also gets past codegen). Asserting only the marker keeps this
//! test pinned to exactly the regression it guards; when the optimizer gap is
//! fixed, tightening this to also assert success is welcome.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    // crates/nsl-cli -> crates -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn nsl_run_delivers_pca_per_document_to_admission_gate() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_run_pca_gate_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Minimal binary corpus so the dataset's `source` points at a real file
    // (the marker does not depend on its contents — codegen never opens it —
    // but a real path keeps the fixture honest if later phases start to).
    let corpus = tmp.join("tokens.bin");
    std::fs::write(&corpus, [0u8; 128]).unwrap();

    // An @flash_attention model method + packed dataset + @pca(per_document)
    // train block: the exact combination whose admission gate prints the
    // [pca-per-doc] marker during kernel synthesis.
    //
    // Use forward slashes so the path is a valid NSL string literal on
    // Windows (backslashes in paths like C:\Users\... would be parsed as
    // escape sequences and rejected by the NSL lexer).
    let corpus_path = corpus.to_string_lossy().replace('\\', "/");
    let source = format!(
        r#"from nsl.nn.losses import mse_loss

dataset PackedCorpus("docs"):
    source = "{corpus}"
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

@pca(strategy=per_document)
train(model = m, epochs = 1):
    data:
        source = PackedCorpus
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, x)
"#,
        corpus = corpus_path
    );
    let fixture = tmp.join("pca_gate.nsl");
    std::fs::write(&fixture, source).unwrap();

    // Same invocation shape as fase_numerical_validation.rs: drive the real
    // `nsl run` CLI so the strategies travel the actual run-command path
    // (commands/run.rs dispatch -> run_build_inner -> normal.rs overwrite).
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run"])
        .arg(&fixture)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("failed to spawn nsl run");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("[pca-per-doc]"),
        "`nsl run` did not reach the per-document admission gate — the \
         @pca(strategy=per_document) strategy list was dropped on the run path \
         (pre-#281 silent-no-op regression).\n--- stderr ---\n{stderr}"
    );

    std::fs::remove_dir_all(&tmp).ok();
}
