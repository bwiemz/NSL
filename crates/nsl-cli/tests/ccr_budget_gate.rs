//! `--checkpoint-budget-mib`: what the knapsack keeps, that the budget is
//! spent once, that it is the number typed, and that it refuses a model it
//! cannot size.
//!
//! # The model and the numbers
//!
//! Four residual feed-forward blocks at batch 2, sequence 64, width 256 and
//! hidden width 1024, f32. Each block computes
//!
//!   n = rmsnorm(h)        2·64·256  elements, 128 KiB
//!   u = n @ w_up          2·64·1024 elements, 512 KiB
//!   v = silu(u)           2·64·1024 elements, 512 KiB
//!   f = v @ w_down        2·64·256  elements, 128 KiB
//!   out = h + f           the saved block boundary, 128 KiB
//!
//! What recomputing each costs, in arithmetic operations:
//!
//!   - `u`: a product over an inner dimension of 256, two operations per
//!     term: 2 · 256 · 131 072 = 67 108 864;
//!   - `v`: elementwise, four per element: 524 288;
//!   - `n`: elementwise, four per element: 131 072;
//!   - `f`: nothing. The backward of `h + f` reads `f` only for its shape,
//!     so `f` is never recomputed and keeping it would buy nothing.
//!
//! A budget of 1 MiB therefore keeps two `u` (2 × 512 KiB, 134 M operations
//! avoided; no other choice of 1 MiB comes near), and 2 MiB keeps all four.
//!
//! Under `--checkpoint-stride dp` the same number caps the projected
//! block-activation peak instead. Keeping all four block boundaries projects
//! to 4 × 128 KiB of boundaries plus one block's 1280 KiB of interiors:
//! 1792 KiB, which fits 2 MiB, and it is the partition that recomputes
//! least, so the search must keep all four.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

const MODEL: &str = r#"from nsl.nn.losses import cross_entropy

model Blk:
    g: Tensor<[256], f32> = ones([256])
    w_up: Tensor<[256, 1024], f32> = randn([256, 1024]) * 0.05
    w_down: Tensor<[1024, 256], f32> = randn([1024, 256]) * 0.05

    fn forward(self, h: Tensor<[2, 64, 256], f32>) -> Tensor<[2, 64, 256], f32>:
        let n: Tensor<[2, 64, 256], f32> = rmsnorm(h, self.g, 0.00001)
        let u: Tensor<[2, 64, 1024], f32> = n @ self.w_up
        let v: Tensor<[2, 64, 1024], f32> = silu(u)
        let f: Tensor<[2, 64, 256], f32> = v @ self.w_down
        return h + f

model LM:
    embed: Tensor<[16, 256], f32> = randn([16, 256]) * 0.1
    blocks: [Blk; 4] = Blk()
    norm_out: Tensor<[256], f32> = ones([256])

    fn forward_train(self, input_ids: Tensor) -> Tensor:
        FORWARD_HEAD
        for block in self.blocks:
            h = block.forward(h)
        let hn = rmsnorm(h, self.norm_out, 0.00001)
        return hn @ self.embed.transpose(0, 1)

let m = LM()
let unit = tensor_cat([arange(7) + full([7], 1.0), zeros([1])], 0)
let tokens = unit.reshape([1, 8]).expand([64, 8]).contiguous().reshape([512])
let loader = DataLoader(tokens, batch_size=2, seq_len=64, shuffle=false, drop_last=DROP_LAST)

print("LOSS_STREAM_BEGIN")
TRAIN_HEADER
    step(batch):
        let logits = m.forward_train(batch.input_ids)
        let flat_logits = logits.reshape([128, 16])
        let flat_labels = batch.labels.reshape([128])
        let loss = cross_entropy(flat_logits, flat_labels)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSS_STREAM_END")
"#;

const SGD: &str = "train(model=m, epochs=1):\n    optimizer: SGD(lr=0.05)";
/// Two micro-batches per step under AdamW: the configuration in which the
/// gradient buffer is deferred, and in which the parameter bytes used to be
/// added to the budget.
const ADAMW_ACCUMULATING: &str = "train(model=m, epochs=1, grad_accumulation=2):\n    \
     optimizer: AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)";

const SIZED_HEAD: &str = "let flat_ids = input_ids.reshape([128])\n        \
     let emb = embedding_lookup(self.embed, flat_ids)\n        \
     let h: Tensor<[2, 64, 256], f32> = emb.reshape([2, 64, 256])";

fn model(train_header: &str) -> String {
    MODEL
        .replace("FORWARD_HEAD", SIZED_HEAD)
        .replace("DROP_LAST", "true")
        .replace("TRAIN_HEADER", train_header)
}

/// The model with nothing a size can be read from: no shape annotation
/// inside the blocks, shapes taken from the batch at run time, and a loader
/// whose last batch may be short (so its batch shape is not proven).
fn unsized_model() -> String {
    let src = MODEL
        .replace(
            "FORWARD_HEAD",
            "let s = input_ids.shape\n        \
             let flat_ids = input_ids.reshape([s[0] * s[1]])\n        \
             let emb = embedding_lookup(self.embed, flat_ids)\n        \
             let h = emb.reshape([s[0], s[1], 256])",
        )
        .replace("DROP_LAST", "false")
        .replace("TRAIN_HEADER", SGD)
        .replace(
            "fn forward(self, h: Tensor<[2, 64, 256], f32>) -> Tensor<[2, 64, 256], f32>:",
            "fn forward(self, h: Tensor) -> Tensor:",
        )
        .replace("let n: Tensor<[2, 64, 256], f32> =", "let n =")
        .replace("let u: Tensor<[2, 64, 1024], f32> =", "let u =")
        .replace("let v: Tensor<[2, 64, 1024], f32> =", "let v =")
        .replace("let f: Tensor<[2, 64, 256], f32> =", "let f =")
        .replace("logits.reshape([128, 16])", "logits.reshape([s2[0] * s2[1], 16])")
        .replace("batch.labels.reshape([128])", "batch.labels.reshape([s2[0] * s2[1]])")
        .replace(
            "        let flat_logits =",
            "        let s2 = logits.shape\n        let flat_logits =",
        );
    assert!(!src.contains("Tensor<[2, 64"), "an annotation survived:\n{src}");
    src
}

struct Run {
    ok: bool,
    losses: String,
    stderr: String,
}

fn run(source: &str, flags: &[&str]) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ccr_budget.nsl");
    std::fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .args(["--target", "cpu", "--source-ad"])
        .args(flags)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let losses = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, rest)| rest.split_once("LOSS_STREAM_END"))
        .map(|(inner, _)| inner.trim().to_string())
        .unwrap_or_default();
    Run {
        ok: out.status.success(),
        losses,
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// A successful run's loss stream: non-empty and finite.
fn losses_of(run: &Run, what: &str) -> String {
    assert!(run.ok, "{what} failed:\n{}", run.stderr);
    assert!(!run.losses.is_empty(), "{what}: no losses printed");
    let lower = run.losses.to_ascii_lowercase();
    assert!(!lower.contains("nan") && !lower.contains("inf"), "{what}: {}", run.losses);
    run.losses.clone()
}

#[test]
fn the_budget_keeps_the_tensors_that_cost_most_to_recompute() {
    let source = model(SGD);
    let plain = losses_of(&run(&source, &[]), "the run without checkpointing");
    for (mib, kept) in [("1", 2), ("2", 4)] {
        let r = run(&source, &["--checkpoint-blocks", "--checkpoint-budget-mib", mib]);
        let losses = losses_of(&r, "the budgeted run");
        assert!(
            r.stderr
                .contains(&format!("budget {mib} MiB: {kept} tensors flipped back to SAVE")),
            "a {mib} MiB budget must keep {kept} of the 512 KiB up-projections:\n{}",
            r.stderr
        );
        assert_eq!(losses, plain, "budget {mib} MiB: recomputation must be exact");
    }
}

/// The budget is the number typed. Under FASE Deferred the parameter bytes
/// (8 MiB of block weights here) used to be added to it, which turned a
/// 1 MiB budget into 9 and kept every interior.
#[test]
fn the_budget_is_not_enlarged_under_deferred_accumulation() {
    let source = model(ADAMW_ACCUMULATING);
    let plain = losses_of(&run(&source, &[]), "the run without checkpointing");
    let r = run(&source, &["--checkpoint-blocks", "--checkpoint-budget-mib", "1"]);
    let losses = losses_of(&r, "the budgeted run");
    assert!(
        r.stderr.contains("budget 1 MiB: 2 tensors flipped back to SAVE"),
        "the budget must stay 1 MiB and keep two tensors:\n{}",
        r.stderr
    );
    assert!(!r.stderr.contains("credit"), "a credit was applied:\n{}", r.stderr);
    assert_eq!(losses, plain, "recomputation must be exact");
}

/// Under `--checkpoint-stride dp` the budget caps the projected peak and is
/// not then spent again on flipping tensors back to SAVE. With 2 MiB every
/// partition up to the per-block one fits; the per-block one recomputes
/// least.
#[test]
fn under_a_stride_search_the_budget_is_spent_once() {
    let source = model(SGD);
    let plain = losses_of(&run(&source, &[]), "the run without checkpointing");
    let r = run(
        &source,
        &["--checkpoint-blocks", "--checkpoint-stride", "dp", "--checkpoint-budget-mib", "2"],
    );
    let losses = losses_of(&r, "the dp run");
    assert!(
        r.stderr
            .contains("--checkpoint-stride dp: kept 4 of 4 block anchors [0, 1, 2, 3]"),
        "all four boundaries fit 2 MiB (1.75 MiB projected) and recompute least:\n{}",
        r.stderr
    );
    assert!(r.stderr.contains("block-activation peak 1.75 MiB"), "{}", r.stderr);
    assert!(
        !r.stderr.contains("flipped back to SAVE"),
        "the knapsack ran after the partition search:\n{}",
        r.stderr
    );
    assert_eq!(losses, plain, "recomputation must be exact");
}

/// With no static size anywhere the budget has nothing to weigh. That used
/// to be a note on stderr ("0 tensors flipped back to SAVE") and exit 0.
#[test]
fn a_budget_on_a_model_without_static_sizes_is_refused() {
    let source = unsized_model();
    // The model itself trains under plain block checkpointing.
    losses_of(&run(&source, &["--checkpoint-blocks"]), "the unsized model");
    let r = run(&source, &["--checkpoint-blocks", "--checkpoint-budget-mib", "1"]);
    assert!(!r.ok, "the budget must fail the build:\n{}", r.stderr);
    assert!(
        r.stderr.contains("--checkpoint-budget-mib cannot be honoured")
            && r.stderr.contains("static size"),
        "expected the refusal to name the flag and the reason:\n{}",
        r.stderr
    );
}
