//! Stage A of WGGO-orchestrated pretraining: surface that the WGGO plan's
//! per-layer CSHA/PCA packing decisions do not reach the decorator-free
//! `scaled_dot_product_attention` training path (the stdlib GQA shape real
//! pretraining trains through).
//!
//! `maybe_synthesize_csha_training_ptx` — home of the @pca/WGGO packing
//! admission fork — only runs when a `@flash_attention` context was built, so
//! on the decorator-free path it never fires and the plan's CSHA/PCA decisions
//! are silently dropped even though WGGO solves and reports them. The
//! reachability verdict makes that non-application visible. It is diagnostic
//! and behaviour-neutral: it prints, and it deliberately does NOT build a
//! flash context for the plain op (that field gates the forward FFI dispatch
//! and the PR #347 backward variant-table path). These pins guard the verdict
//! and its chatter-free discipline.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Compile (object only — no run, no link) and return stderr.
fn build_stderr(source: &str, tag: &str, wggo: Option<&str>) -> String {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_wggo_reach_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, source).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "build", "--source-ad"]);
    if let Some(mode) = wggo {
        cmd.args(["--wggo", mode]);
    }
    cmd.arg(&prog)
        .arg("--emit-obj")
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    let output = cmd.output().expect("spawn nsl build");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    std::fs::remove_dir_all(&tmp).ok();
    stderr
}

const DECORATOR_FREE_ATTN: &str = r#"from nsl.nn.losses import mse_loss

model TinyAttn:
    w_norm: Tensor = ones([64])
    wq: Tensor = ones([64, 64])
    wk: Tensor = ones([64, 64])
    wv: Tensor = ones([64, 64])

    fn forward(self, x: Tensor) -> Tensor:
        let x_norm = rmsnorm(x, self.w_norm, 0.00001)
        let q = x_norm @ self.wq
        let k = x_norm @ self.wk
        let v = x_norm @ self.wv
        let scale = 1.0 / sqrt(64.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
let x = ones([1, 1, 64, 64])
let y = zeros([1, 1, 64, 64])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)
"#;

/// Same model with a `@flash_attention` decorator: a context IS built, so
/// `maybe_synthesize_csha_training_ptx` runs the admission fork normally and
/// there is nothing to report.
const DECORATED_ATTN: &str = r#"from nsl.nn.losses import mse_loss

model TinyAttn:
    w_norm: Tensor = ones([64])
    wq: Tensor = ones([64, 64])
    wk: Tensor = ones([64, 64])
    wv: Tensor = ones([64, 64])

    @flash_attention(head_dim = 64, causal = false)
    fn forward(self, x: Tensor) -> Tensor:
        let x_norm = rmsnorm(x, self.w_norm, 0.00001)
        let q = x_norm @ self.wq
        let k = x_norm @ self.wk
        let v = x_norm @ self.wv
        let scale = 1.0 / sqrt(64.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
let x = ones([1, 1, 64, 64])
let y = zeros([1, 1, 64, 64])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)
"#;

/// Under `--wggo`, a decorator-free SDPA training model must report that the
/// solved plan is not applied on this path — and must name the facts a user
/// needs (that it is the plain SDPA path, and that a layer WAS planned).
#[test]
fn decorator_free_sdpa_train_reports_plan_unreachable() {
    let stderr = build_stderr(DECORATOR_FREE_ATTN, "free", Some("greedy"));
    assert!(
        stderr.contains("[wggo] plan-reachability:"),
        "expected the reachability verdict on the decorator-free SDPA path:\n{stderr}"
    );
    assert!(
        stderr.contains("scaled_dot_product_attention"),
        "verdict must name the plain SDPA path:\n{stderr}"
    );
    // The plan really did solve a layer for this model (proving something is
    // being dropped, not that there was nothing to apply).
    assert!(
        stderr.contains("layers_planned=1"),
        "verdict must report the number of planned layers:\n{stderr}"
    );
}

/// A pure-MLP training model has no attention op, so even with `--wggo` the
/// verdict must NOT appear — a plan is still solved for the train block (an
/// MLP's matmuls yield one), but the attention-only CSHA/PCA reachability claim
/// would be false. Regression guard for the attention-presence gate.
const MLP_TRAIN: &str = r#"from nsl.nn.losses import mse_loss

model TinyMlp:
    w: Tensor = ones([64, 64])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = TinyMlp()
let x = ones([4, 64])
let y = zeros([4, 64])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)
"#;

#[test]
fn mlp_train_has_no_reachability_verdict() {
    let stderr = build_stderr(MLP_TRAIN, "mlp", Some("greedy"));
    assert!(
        !stderr.contains("plan-reachability"),
        "a pure-MLP train block has no attention op; the attention-path verdict must not fire:\n{stderr}"
    );
}

/// Without `--wggo` the verdict must not appear (chatter-free discipline; no
/// plan is solved, so nothing is dropped).
#[test]
fn no_wggo_is_reachability_chatter_free() {
    let stderr = build_stderr(DECORATOR_FREE_ATTN, "nowggo", None);
    assert!(
        !stderr.contains("plan-reachability"),
        "no --wggo must be reachability-chatter-free:\n{stderr}"
    );
}

/// `--wggo off` arrives as `Some("off")`; it is NOT "enabled", so the verdict
/// must stay silent (the same trap that governs the pre-pass diagnostics).
#[test]
fn wggo_off_is_reachability_chatter_free() {
    let stderr = build_stderr(DECORATOR_FREE_ATTN, "off", Some("off"));
    assert!(
        !stderr.contains("plan-reachability"),
        "--wggo off must be reachability-chatter-free:\n{stderr}"
    );
}

/// With a `@flash_attention` decorator a context is built and the admission
/// fork runs, so no plan is dropped and the verdict must not appear.
#[test]
fn decorated_flash_attention_has_no_reachability_verdict() {
    let stderr = build_stderr(DECORATED_ATTN, "deco", Some("greedy"));
    assert!(
        !stderr.contains("plan-reachability"),
        "the decorated path runs the admission fork; no reachability verdict is expected:\n{stderr}"
    );
}
