//! Item 4 — `--fuse-lm-head`: the fused LM head without a hand-written decorator.
//!
//! # The defect this gates
//!
//! The fused linear-CE kernel deletes the `[batch*seq, vocab]` logits surface
//! and its gradient, worth ~8.5% step time on Coder-50M. Reaching it required
//! writing `@fused_lm_ce(enabled=true, vocab_size=…, hidden_size=…,
//! batch_size=…, seq_len=…)` onto the train block — four numbers the compiler
//! already knows. No production pretraining script in this repo carried it;
//! only `models/benchmarks/mfu_bench.py`, which *generates* the decorator into
//! a throwaway program. The measured win existed solely inside a benchmark
//! harness.
//!
//! # What is asserted, and what is deliberately not
//!
//! These are CPU `--emit-obj` builds: they pin WHICH DECISION the compiler
//! made, not that the kernel computes the right loss. Numerical equivalence of
//! the fused and composite paths is `fused_linear_ce_numerical.rs` and the GPU
//! cert lane's business, and it predates this item — inference reuses that
//! substitution verbatim rather than adding a second one.
//!
//! The `[lm-head-fusion]` marker is a distinct token from `[fused-lm-ce]` on
//! purpose. `[fused-lm-ce]` means "the fusion did NOT happen" and
//! `fused_lm_ce_decline_gate` asserts it ABSENT in its success cases; reusing
//! it for a success line would make every one of those negative assertions
//! pass for the wrong reason.

use assert_cmd::prelude::*;
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

/// Build `src` with `--emit-obj` (no CUDA link needed) and the given extra
/// flags; return `(exit_ok, stderr)`.
fn build(src: &str, tag: &str, flags: &[&str]) -> (bool, String) {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join(format!("{tag}.nsl"));
    let out_path = tmp.path().join(format!("{tag}_out"));
    fs::write(&src_path, src).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .arg("build")
        .arg(&src_path)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path);
    for f in flags {
        cmd.arg(f);
    }
    let output = cmd.output().expect("spawn nsl build");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// The house-style pretraining program, verbatim in shape with
/// `models/coder50m/pretrain.nsl`: a biasless weight-tied head, a runtime
/// `logits.shape` read, and the flatten built from it.
///
/// `decorator` and `loader` are the only knobs; keeping the rest identical is
/// what makes the negative cases below mean anything — if the prelude drifted,
/// every case would move together.
fn house_style(decorator: &str, loader: &str) -> String {
    format!(
        r#"from nsl.nn.losses import cross_entropy

model TinyLM:
    embed: Tensor = randn([128, 64]) * full([1], 0.02)
    norm_out: Tensor = ones([64])

    fn forward_train(self, input_ids: Tensor) -> Tensor:
        let ids_shape = input_ids.shape
        let b = ids_shape[0]
        let s = ids_shape[1]
        let flat_ids = input_ids.reshape([b * s])
        let emb = embedding_lookup(self.embed, flat_ids)
        let h = emb.reshape([b, s, 64])
        let hn = rmsnorm(h, self.norm_out, 0.00001)
        return hn @ self.embed.transpose(0, 1)

let m = TinyLM()
let tokens = arange(256)
{loader}

{decorator}train(model=m, epochs=1):
    optimizer: AdamW(lr=0.002)
    step(batch):
        let logits = m.forward_train(batch.input_ids)
        let ls = logits.shape
        let flat_logits = logits.reshape([ls[0] * ls[1], ls[2]])
        let flat_labels = batch.labels.reshape([ls[0] * ls[1]])
        let loss = cross_entropy(flat_logits, flat_labels)
    callbacks:
        on_step(step, loss):
            print(loss)
"#
    )
}

const GOOD_LOADER: &str =
    "let loader = DataLoader(tokens, batch_size=2, seq_len=64, shuffle=false, drop_last=true)";

/// The marker line a successful inference prints, minus the varying numbers.
const INFERRED: &str = "[lm-head-fusion] inferred:";

// ---------------------------------------------------------------------------
// The headline case
// ---------------------------------------------------------------------------

/// The whole point of the item: the shipped pretraining idiom fuses with no
/// decorator, no source edit, and no knowledge that the feature exists.
///
/// The `logits.shape` read in this fixture is load-bearing. It is a live
/// consumer of the `[B, S, V]` logits, so before the compile-time shape fold
/// this program declined — which would have left `--fuse-lm-head` firing only
/// on programs that already hard-code their dims, i.e. on nothing this repo
/// ships.
#[test]
fn auto_infers_the_house_style_head_through_a_runtime_shape_read() {
    let (ok, stderr) = build(
        &house_style("", GOOD_LOADER),
        "auto_house",
        &["--source-ad", "--fuse-lm-head", "auto"],
    );
    assert!(ok, "the house-style head must compile:\n{stderr}");
    assert!(
        stderr.contains(INFERRED),
        "no fused head was inferred for the house-style program:\n{stderr}"
    );
    // The four numbers, so a bug that infers a *plausible* head is still
    // caught: vocab and hidden come off the weight `[128, 64]`, the rows off
    // the DataLoader's 2x64.
    assert!(
        stderr.contains("vocab=128 hidden=64 rows=2x64=128"),
        "the inferred dims are wrong:\n{stderr}"
    );
}

/// `--pretrain-optimized` must deliver it, since "the best validated stack"
/// is the flag's entire job.
#[test]
fn pretrain_optimized_selects_auto() {
    let (ok, stderr) = build(
        &house_style("", GOOD_LOADER),
        "bundle",
        &["--pretrain-optimized"],
    );
    assert!(ok, "the bundle must compile the house-style head:\n{stderr}");
    assert!(
        stderr.contains(INFERRED),
        "--pretrain-optimized did not infer a fused LM head:\n{stderr}"
    );
}

// ---------------------------------------------------------------------------
// Authorization — the rewrite is never silent-by-default
// ---------------------------------------------------------------------------

#[test]
fn off_is_the_default() {
    let (ok, stderr) = build(&house_style("", GOOD_LOADER), "default", &["--source-ad"]);
    assert!(ok, "{stderr}");
    assert!(
        !stderr.contains("[lm-head-fusion]"),
        "inference must not run without --fuse-lm-head:\n{stderr}"
    );
}

/// `enabled = false` is a deliberate opt-out. Inference overturning it would
/// mean there was no way to say no.
#[test]
fn an_explicitly_disabled_decorator_is_not_overturned() {
    let deco = "@fused_lm_ce(enabled=false, vocab_size=128, hidden_size=64, \
                batch_size=2, seq_len=64, vocab_tile=128)\n";
    let (ok, stderr) = build(
        &house_style(deco, GOOD_LOADER),
        "optout",
        &["--source-ad", "--fuse-lm-head", "require"],
    );
    assert!(
        ok,
        "an opt-out must compile even under --fuse-lm-head require:\n{stderr}"
    );
    assert!(
        !stderr.contains("[lm-head-fusion]"),
        "inference ran on a train block that opted out:\n{stderr}"
    );
}

/// The reference arm is what the fused numerics are validated against, so it
/// must not contain the thing it is the baseline for.
#[test]
fn training_reference_suppresses_inference() {
    let (ok, stderr) = build(
        &house_style("", GOOD_LOADER),
        "refarm",
        &["--source-ad", "--fuse-lm-head", "auto", "--training-reference"],
    );
    assert!(ok, "{stderr}");
    assert!(
        !stderr.contains(INFERRED),
        "--training-reference must not contain an inferred fused head:\n{stderr}"
    );
}

// ---------------------------------------------------------------------------
// Refusal, and its `auto` counterpart
// ---------------------------------------------------------------------------

/// Without a DataLoader the token-row count is not a compile-time fact, so
/// there is nothing to bake into the op.
#[test]
fn an_unprovable_row_count_refuses_under_require_and_falls_back_under_auto() {
    // No DataLoader at all: the train block runs one step per epoch.
    let src = house_style("", "").replace("step(batch):", "step(batch):");
    let (ok_req, err_req) = build(
        &src,
        "req_norows",
        &["--source-ad", "--fuse-lm-head", "require"],
    );
    assert!(
        !ok_req,
        "--fuse-lm-head require must refuse when the row count is unprovable:\n{err_req}"
    );
    assert!(
        err_req.contains("--fuse-lm-head require"),
        "the refusal must be the item-4 one, not an unrelated failure:\n{err_req}"
    );

    let (ok_auto, err_auto) = build(&src, "auto_norows", &["--source-ad", "--fuse-lm-head", "auto"]);
    assert!(
        ok_auto,
        "auto must fall back to the composite path, not refuse:\n{err_auto}"
    );
    assert!(
        !err_auto.contains(INFERRED),
        "nothing should have been inferred:\n{err_auto}"
    );
}

/// A short final batch makes `batch_size * seq_len` not a row count. The
/// runtime does catch it (`nsl_fused_lce_targets_i64_alloc` refuses on a label
/// count that disagrees with the pinned rows) — but at the END of a run, which
/// is not the same as declining to infer.
#[test]
fn drop_last_false_declines() {
    let loader =
        "let loader = DataLoader(tokens, batch_size=2, seq_len=64, shuffle=false, drop_last=false)";
    let (ok, stderr) = build(
        &house_style("", loader),
        "droplast",
        &["--source-ad", "--fuse-lm-head", "auto"],
    );
    assert!(ok, "{stderr}");
    assert!(
        !stderr.contains(INFERRED),
        "a short final batch must not be fused with a constant row count:\n{stderr}"
    );
    assert!(
        stderr.contains("drop_last=false"),
        "the decline must name drop_last:\n{stderr}"
    );
}

/// Two differently-shaped loaders: the compiler will not guess which one feeds
/// the block.
#[test]
fn disagreeing_loaders_decline() {
    let loader = "let loader = DataLoader(tokens, batch_size=2, seq_len=64, shuffle=false, \
                  drop_last=true)\nlet other = DataLoader(tokens, batch_size=8, seq_len=64, \
                  shuffle=false, drop_last=true)";
    let (ok, stderr) = build(
        &house_style("", loader),
        "twoloaders",
        &["--source-ad", "--fuse-lm-head", "auto"],
    );
    assert!(ok, "{stderr}");
    assert!(
        !stderr.contains(INFERRED),
        "an ambiguous row count must not be inferred:\n{stderr}"
    );
}

/// A head whose weight is already `[H, V]` reaches the matmul with no
/// transpose; every fused emitter indexes `W[v*H + h]`, so this is a genuine
/// structural decline rather than a missing fact.
#[test]
fn a_pre_transposed_head_declines_and_still_compiles() {
    let src = house_style("", GOOD_LOADER)
        .replace(
            "    embed: Tensor = randn([128, 64]) * full([1], 0.02)",
            "    embed: Tensor = randn([128, 64]) * full([1], 0.02)\n    \
             head_hv: Tensor = randn([64, 128]) * full([1], 0.02)",
        )
        .replace(
            "        return hn @ self.embed.transpose(0, 1)",
            "        return hn @ self.head_hv",
        );
    let (ok, stderr) = build(
        &src,
        "notranspose",
        &["--source-ad", "--fuse-lm-head", "auto"],
    );
    assert!(ok, "auto must not refuse a head it cannot fuse:\n{stderr}");
    assert!(
        !stderr.contains(INFERRED),
        "a pre-transposed [H, V] head must not be fused:\n{stderr}"
    );
    assert!(
        stderr.contains("[lm-head-fusion] declined:"),
        "a structural decline must say so rather than being silent:\n{stderr}"
    );
}

// ---------------------------------------------------------------------------
// The shipped scripts
// ---------------------------------------------------------------------------

/// The regression that matters most: the repo's own pretraining scripts must
/// keep inferring. A change to the matcher, the loader scan, the shape fold,
/// or `models/coder50m/model.nsl`'s initializer idiom that quietly stops the
/// fusion would otherwise show up only as a slower benchmark months later.
///
/// Multi-file (`from model import NSLCoder`) on purpose: the imported-model
/// shape maps were dropped on the floor by the multi-file build path, so a
/// single-file fixture would have passed while every real model declined.
#[test]
fn the_shipped_coder50m_pretraining_scripts_infer_a_fused_head() {
    // PR #463 put an EXPLICIT `@fused_lm_ce` on `pretrain.nsl`, and an
    // explicit decorator wins over inference by design — so that script must
    // reach the fused path WITHOUT the `inferred:` marker, while the bare
    // `pretrain_cert.nsl` remains the proof that inference still carries the
    // shipped idiom on its own.
    for (script, expect_inferred) in [("pretrain.nsl", false), ("pretrain_cert.nsl", true)] {
        let dir = workspace_root().join("models").join("coder50m");
        let tmp = TempDir::new().unwrap();
        let out = tmp.path().join("out.o");
        let mut cmd = Command::cargo_bin("nsl").unwrap();
        cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
            .current_dir(&dir)
            .arg("build")
            .arg(script)
            .arg("--pretrain-optimized")
            .arg("--emit-obj")
            .arg("-o")
            .arg(&out);
        let output = cmd.output().expect("spawn nsl build");
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        assert!(
            output.status.success(),
            "models/coder50m/{script} must build under --pretrain-optimized:\n{stderr}"
        );
        assert_eq!(
            stderr.contains("[lm-head-fusion] inferred:"),
            expect_inferred,
            "models/coder50m/{script}: inference fired where the explicit \
             decorator should own the head (or stopped firing where it must):\n{stderr}"
        );
        if expect_inferred {
            assert!(
                stderr.contains("vocab=49152 hidden=512"),
                "models/coder50m/{script} inferred the wrong dims:\n{stderr}"
            );
        } else {
            // The explicit-decorator SUCCESS path is silent at compile time
            // (its `[fused-lm-ce]` marker is a runtime exec marker, asserted
            // by fused_lm_ce_hint_pin_gate); what a compile CAN prove is
            // that inference correctly stood down — no inferred line above,
            // and no decline either, since a decorated block never enters
            // the inference machinery at all.
            assert!(
                !stderr.contains("[lm-head-fusion] declined:"),
                "models/coder50m/{script}: the decorated block entered the \
                 inference machinery:\n{stderr}"
            );
        }
    }
}
