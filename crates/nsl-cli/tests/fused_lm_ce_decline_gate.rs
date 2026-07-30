//! Item 6 — `@fused_lm_ce(enabled = true)` that fuses NOTHING must refuse.
//!
//! # The defect this gates
//!
//! The decorator's purpose is to delete the `[batch*seq, vocab]`
//! logits-gradient surface. Historically its substitution declined on a
//! biasless LM head and on a reshape between the head and the loss — which
//! is exactly what `models/coder50m/model.nsl:79-80` and
//! `models/coder50m/pretrain.nsl:30-32` have. Before item 6 those declines
//! were silent: `nsl build` exited 0, said nothing, and emitted the full
//! composite path while the user believed the fused kernel was live.
//! Sprint 2.5 flipped BOTH of those shapes into substitutions (GEMM-chunked
//! path); the refusal machinery itself is still gated here via the
//! pre-transposed-weight shape, which remains a genuine decline.
//!
//! Measured on the pre-fix binary with the `declines_when_nothing_fuses`
//! fixture below: exit 0, and no line of stderr mentioning the decorator.
//!
//! # Why the refusal is scoped to "nothing fused"
//!
//! A step body may legitimately contain an auxiliary `cross_entropy` that is
//! not the LM head. Refusing on ANY decline would make the decorator unusable
//! there, so a partial decline warns on stderr and only a total failure to
//! substitute is a hard error. `partial_decline_warns_but_compiles` pins that
//! boundary — without it the refusal could be tightened to "any decline" and
//! every test here would still pass.

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

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Build `src` with source-AD and `--emit-obj` (no CUDA link needed) and
/// return `(exit_ok, stderr)`.
fn build(src: &str, tag: &str) -> (bool, String) {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join(format!("{tag}.nsl"));
    let out_path = tmp.path().join(format!("{tag}_out"));
    fs::write(&src_path, src).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path);
    let output = cmd.output().expect("spawn nsl build");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// Shared model + data prelude. `HEAD` and `LOSS` are substituted per case so
/// the only difference between the refusing and control fixtures is the LM
/// head shape — if the prelude drifted, every case would move together and
/// the comparison would stay honest.
fn fixture(head: &str, loss: &str, enabled: &str) -> String {
    format!(
        r#"from nsl.nn.losses import cross_entropy

model TinyLM:
    embed: Tensor = randn([128, 64]) * 0.1
    embed_hv: Tensor = randn([64, 128]) * 0.1
    norm_out: Tensor = ones([64])
    lm_bias: Tensor = zeros([128])

    fn forward_train(self, input_ids: Tensor) -> Tensor:
        let ids_shape = input_ids.shape
        let batch_size = ids_shape[0]
        let seq_len = ids_shape[1]
        let flat_ids = input_ids.reshape([batch_size * seq_len])
        let emb = embedding_lookup(self.embed, flat_ids)
        let h = emb.reshape([batch_size, seq_len, 64])
        let hn = rmsnorm(h, self.norm_out, 0.00001)
{head}

let m = TinyLM()
let tokens = arange(256)
let loader = DataLoader(tokens, batch_size=2, seq_len=64, shuffle=false, drop_last=true)

@fused_lm_ce(enabled={enabled}, dtype="f32", vocab_size=128, hidden_size=64, batch_size=2, seq_len=64, vocab_tile=128)
train(model=m, epochs=1):
    optimizer: AdamW(lr=0.002)
    step(batch):
        let logits = m.forward_train(batch.input_ids)
        let flat_labels = batch.labels.reshape([128])
{loss}
    callbacks:
        on_step(step, loss):
            print(loss)
"#
    )
}

/// `models/coder50m/model.nsl:79-80` — biasless weight-tied head.
/// BEHAVIOUR FLIP (Sprint 2.5): substitutes (has_bias=false, GEMM path).
const BIASLESS_HEAD: &str = "        let flat_hn = hn.reshape([batch_size * seq_len, 64])\n\
    \x20       return flat_hn @ self.embed.transpose(0, 1)";
/// A head the matcher still cannot honour: the weight reaches the matmul
/// WITHOUT a transpose (a pre-transposed `[H, V]` layout would need a new
/// emitter family) — keeps the item-6 nothing-fused refusal genuinely
/// exercised now that biasless and reshaped heads fuse.
const NO_TRANSPOSE_HEAD: &str = "        let flat_hn = hn.reshape([batch_size * seq_len, 64])\n\
    \x20       return (flat_hn @ self.embed_hv) + self.lm_bias";
/// The canonical head the matcher accepts.
const MATCHING_HEAD: &str =
    "        let flat_hn = hn.reshape([batch_size * seq_len, 64])\n\
    \x20       return (flat_hn @ self.embed.transpose(0, 1)) + self.lm_bias";
const PLAIN_LOSS: &str = "        let loss = cross_entropy(logits, flat_labels)";

// ─── The gate ──────────────────────────────────────────────────────────────

/// A head the matcher cannot honour (no transpose on the weight) under an
/// enabled, fully-hinted decorator must be a hard error naming the reason —
/// not a clean compile. (Sprint 2.5 flipped the original biasless fixture
/// into a substitution — see `biasless_head_compiles_and_fuses` — so this
/// gate now uses the pre-transposed-weight shape, which remains a genuine
/// decline.)
#[test]
fn declines_when_nothing_fuses() {
    let (ok, stderr) = build(
        &fixture(NO_TRANSPOSE_HEAD, PLAIN_LOSS, "true"),
        "no_transpose_refuses",
    );
    assert!(
        !ok,
        "a pre-transposed-weight LM head under @fused_lm_ce(enabled=true) \
         must REFUSE, but the build succeeded. stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("@fused_lm_ce(enabled = true) is active on this train block"),
        "the refusal must be the fused-CE one, not an unrelated failure. \
         stderr:\n{stderr}"
    );
    // Assert on the DECLINE PAYLOAD, not the explanatory boilerplate (the
    // original form of this test pinned `produced by Matmul` for the same
    // reason — a generic message would pass for any cause).
    assert!(
        stderr.contains("not produced by a transpose"),
        "the refusal must name the actual reason (weight not transposed). \
         stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("set enabled = false"),
        "the refusal must state the deliberate opt-out. stderr:\n{stderr}"
    );
}

/// BEHAVIOUR FLIP (Sprint 2.5): the biasless weight-tied head — the exact
/// `models/coder50m/model.nsl` shape — now compiles AND fuses.
#[test]
fn biasless_head_compiles_and_fuses() {
    let (ok, stderr) = build(
        &fixture(BIASLESS_HEAD, PLAIN_LOSS, "true"),
        "biasless_fuses",
    );
    assert!(
        ok,
        "the biasless head must compile under @fused_lm_ce(enabled=true) \
         since Sprint 2.5. stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("[fused-lm-ce]"),
        "no fused-CE fallback diagnostic may fire for the biasless head. \
         stderr:\n{stderr}"
    );
}

/// BEHAVIOUR FLIP (Sprint 2.5): a reshape between the head and the loss
/// (`coder50m/pretrain.nsl:30-32` — the shape the repo's own pretrain
/// scripts use) is now seen through and fuses.
#[test]
fn reshape_between_head_and_loss_compiles_and_fuses() {
    // 3-D logits out of the head, flattened at the loss with COMPILE-TIME
    // dims (batch 2 x seq 64 = 128 rows, vocab 128). Runtime `.shape` reads
    // of the logits are a separate, refused case — see the next test.
    let head = "        let logits3d = (hn @ self.embed.transpose(0, 1)) + self.lm_bias\n\
                \x20       return logits3d";
    let loss = "        let flat_logits = logits.reshape([128, 128])\n\
                \x20       let loss = cross_entropy(flat_logits, flat_labels)";
    let (ok, stderr) = build(&fixture(head, loss, "true"), "reshape_fuses");
    assert!(
        ok,
        "the reshape-between-head-and-loss shape must compile under \
         @fused_lm_ce(enabled=true) since Sprint 2.5 (see-through). \
         stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("[fused-lm-ce]"),
        "no fused-CE fallback diagnostic may fire for the seen-through \
         reshape. stderr:\n{stderr}"
    );
}

/// A RUNTIME consumer of the logits (`let ls = logits.shape`) between the
/// head and the loss keeps the composite chain alive after substitution —
/// the logits must materialize every step anyway (fusion defeated) and the
/// surviving chain's ghost adjoints would corrupt gradient accumulation.
/// Must be a hard refusal that names the fix, not an unresolved-adjoint ICE.
#[test]
fn live_logits_consumer_refuses_with_actionable_message() {
    let head = "        let logits3d = (hn @ self.embed.transpose(0, 1)) + self.lm_bias\n\
                \x20       return logits3d";
    let loss = "        let ls = logits.shape\n\
                \x20       let flat_logits = logits.reshape([ls[0] * ls[1], ls[2]])\n\
                \x20       let loss = cross_entropy(flat_logits, flat_labels)";
    let (ok, stderr) = build(&fixture(head, loss, "true"), "live_consumer_refuses");
    assert!(
        !ok,
        "a live logits consumer under @fused_lm_ce(enabled=true) must \
         REFUSE. stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("logits chain is still LIVE"),
        "the refusal must be the live-consumer diagnosis, not an \
         unresolved-adjoint ICE. stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("compile-time dims"),
        "the refusal must name the fix. stderr:\n{stderr}"
    );
}

// ─── Anti-vacuity: the refusal must not fire on everything ─────────────────

/// The canonical head compiles. Without this the refusal above could be an
/// unconditional error and every assertion would still pass.
#[test]
fn matching_head_still_compiles() {
    let (ok, stderr) = build(
        &fixture(MATCHING_HEAD, PLAIN_LOSS, "true"),
        "matching_ok",
    );
    assert!(
        ok,
        "the canonical Add(Matmul(x, Transpose(W)), bias) head must still \
         compile. stderr:\n{stderr}"
    );
    // Compiling is not enough — the point is that the kernel ENGAGED. Without
    // this, a change that made the matcher decline here would still pass (the
    // decline warning only fires when something else fused, and the refusal
    // only when nothing did... which would make this a refusal, so the
    // assertion above does carry weight — but assert the absence of any
    // fallback chatter explicitly rather than relying on that reasoning).
    assert!(
        !stderr.contains("[fused-lm-ce]"),
        "the matching head must produce no fused-CE fallback diagnostic at \
         all. stderr:\n{stderr}"
    );
}

/// An explicit `fused_linear_ce(...)` call with no decorator silently expands
/// to the stdlib composite. That is the same broken promise as a declined
/// auto-substitution — more so, since the user named the fused function — so
/// it must at least warn.
///
/// It cannot REFUSE: every non-substituting path out of that arm returns
/// `None`, which fails source-AD extraction outright, and the train-block
/// refusal is gated on successful extraction. A decline recorded there would
/// be silence with extra steps, which is what an earlier version of this
/// change did.
#[test]
fn explicit_fused_linear_ce_without_a_decorator_warns() {
    let src = r#"from nsl.nn.losses import fused_linear_ce

model LM:
    w: Tensor = randn([128, 64])
    bias: Tensor = zeros([128])

    fn forward(self, x: Tensor, targets: Tensor) -> Tensor:
        return fused_linear_ce(x, self.w, self.bias, targets)

let m = LM()
let x = randn([128, 64])
let targets = zeros([128])
let tokens = arange(256)
let loader = DataLoader(tokens, batch_size=2, seq_len=64, shuffle=false, drop_last=true)

train(model=m, epochs=1):
    optimizer: SGD(lr=0.01)
    step(batch):
        let loss = m.forward(x, targets)
    callbacks:
        on_step(step, loss):
            print(loss)
"#;
    let (_ok, stderr) = build(src, "explicit_no_decorator");
    assert!(
        stderr.contains("[fused-lm-ce] fused_linear_ce(...) called without an \
                         active @fused_lm_ce decorator"),
        "an explicit fused_linear_ce call with no decorator must warn that the \
         composite will run instead. stderr:\n{stderr}"
    );
}

/// `enabled = false` on the SAME non-matching head compiles clean. This is
/// what makes the refusal a broken-promise check rather than a shape check:
/// with no promise there is nothing to break.
#[test]
fn disabled_decorator_on_the_same_head_compiles() {
    let (ok, stderr) = build(
        &fixture(BIASLESS_HEAD, PLAIN_LOSS, "false"),
        "biasless_disabled",
    );
    assert!(
        ok,
        "@fused_lm_ce(enabled=false) is a deliberate opt-out and must compile \
         even on a non-matching head. stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("[fused-lm-ce]"),
        "a disabled decorator must not report a decline at all. stderr:\n{stderr}"
    );
}

/// An enabled decorator on a step body source-AD cannot extract WARNS.
///
/// This is the OTHER way the promise breaks, and the refusal does not cover
/// it: with no successful extraction there are no declines to report, so the
/// hard error is unreachable. `crates/nsl-codegen/tests/fixtures/
/// fused_lm_ce_e2e_{fp16,bf16}.nsl` both live here — their heads use
/// `bias_add(...)`, which has no source-AD handler — which is why this warns
/// rather than refuses.
#[test]
fn enabled_decorator_with_unextractable_body_warns() {
    // `bias_add` has no source-AD handler, so extraction fails and the run
    // falls back to tape AD, where the fused kernel does not exist.
    let head = "        let flat_hn = hn.reshape([batch_size * seq_len, 64])\n\
    \x20       return bias_add(flat_hn @ self.embed.transpose(0, 1), self.lm_bias)";
    let (_ok, stderr) = build(&fixture(head, PLAIN_LOSS, "true"), "unextractable");
    assert!(
        stderr.contains("source-AD extraction of the step body failed"),
        "an enabled decorator over an unextractable body must warn that the \
         fused kernel will not run. stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("[fused-lm-ce]"),
        "the warning must carry the registered marker. stderr:\n{stderr}"
    );
}

/// A step body with a matching LM head AND a second, non-matching
/// `cross_entropy` compiles — with a warning naming the fallback.
///
/// This pins the "nothing fused" scoping. If the refusal were tightened to
/// "any decline", this test flips to a failure; every other test in the file
/// would stay green, so this is the only thing standing between the current
/// behaviour and an over-strict one.
#[test]
fn partial_decline_warns_but_compiles() {
    let loss = "        let head_loss = cross_entropy(logits, flat_labels)\n\
                \x20       let aux = cross_entropy(flat_hn_aux, flat_labels)\n\
                \x20       let loss = head_loss + aux";
    // `flat_hn_aux` is a bare tensor, so its cross_entropy declines while the
    // head's fuses.
    let src = fixture(MATCHING_HEAD, loss, "true").replace(
        "        let flat_labels = batch.labels.reshape([128])",
        "        let flat_labels = batch.labels.reshape([128])\n\
         \x20       let flat_hn_aux = zeros([128, 128])",
    );
    let (ok, stderr) = build(&src, "partial_decline");
    assert!(
        ok,
        "one fused call is enough — a partial decline must not refuse. \
         stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("[fused-lm-ce] a cross_entropy call fell back"),
        "a partial decline must still WARN, or a head that quietly stopped \
         matching would be invisible. stderr:\n{stderr}"
    );
}
