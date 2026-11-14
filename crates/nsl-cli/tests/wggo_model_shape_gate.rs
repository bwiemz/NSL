//! WGGO plans the model being compiled, not a built-in layer.
//!
//! Every plan, memory figure and budget check used to be computed for one
//! fixed layer (d_model 512, 8 heads, FFN 2048, 8 layers' worth of nothing in
//! particular): a 33,344-parameter model and a 12-head, 768-wide model both
//! got "8/8 heads, FFN=2048", a 200 MiB budget on a model whose parameters
//! take 133 KB was refused as infeasible, and a weights file was sliced as 8
//! heads of 64 whatever its tensors' shapes were.
//!
//! These tests drive the real compiler on two programs whose shapes are
//! written in their source and assert the figures of those shapes, worked
//! out by hand in the comments.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

fn stdlib_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("stdlib")
}

/// 12 heads of 64 over a 768-wide stream, 4 KV heads, FFN 3072, 2 blocks,
/// through the standard library's grouped-query attention (its unpacked
/// `forward`), batch 1, 8 tokens.
const GQA12: &str = r#"from nsl.nn.gqa import GroupedQueryAttention
from nsl.nn.losses import cross_entropy

model Blk:
    norm_a: Tensor = ones([768])
    norm_f: Tensor = ones([768])
    attn: GroupedQueryAttention = GroupedQueryAttention(768, 12, 4, 64, 0.0, 16, 10000.0, 2)
    w_up: Tensor = randn([768, 3072]) * 0.02
    w_down: Tensor = randn([3072, 768]) * 0.02

    fn forward(self, h: Tensor, training: bool) -> Tensor:
        let a = self.attn.forward(rmsnorm(h, self.norm_a, 0.00001), training)
        let h1 = h + a
        let f = (rmsnorm(h1, self.norm_f, 0.00001) @ self.w_up) @ self.w_down
        return h1 + f

model LM:
    embed: Tensor = randn([64, 768]) * 0.02
    blocks: [Blk; 2] = Blk()
    norm_out: Tensor = ones([768])

    fn forward_train(self, input_ids: Tensor, training: bool) -> Tensor:
        let flat_ids = input_ids.reshape([8])
        let emb = embedding_lookup(self.embed, flat_ids)
        let h = emb.reshape([1, 8, 768])
        for block in self.blocks:
            h = block.forward(h, training)
        let hn = rmsnorm(h, self.norm_out, 0.00001)
        return hn @ self.embed.transpose(0, 1)

let m = LM()
let ids = zeros([1, 8])
let labels = zeros([8])

train(model=m, epochs=1):
    optimizer: AdamW(lr=0.001, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)
    step(batch):
        let logits = m.forward_train(ids, false)
        let flat_logits = logits.reshape([8, 64])
        let loss = cross_entropy(flat_logits, labels)
"#;

/// Eight blocks of one 64 x 64 matrix each, an 8 x 64 embedding and a
/// 64-element gain: 8 x 4096 + 512 + 64 = 33,344 parameters, 133,376 bytes.
const TINY: &str = r#"from nsl.nn.losses import mse_loss

model Block:
    w: Tensor = ones([64, 64])

    fn forward(self, h: Tensor) -> Tensor:
        return h @ self.w

model TinyNet:
    embed: Tensor = ones([8, 64])
    blocks: [Block; 8] = Block()
    final_norm: Tensor = ones([64])

    fn forward(self, x: Tensor) -> Tensor:
        let h = x @ self.embed
        for block in self.blocks:
            h = block.forward(h)
        let scale = self.final_norm
        return h * scale

let m = TinyNet()
let x = ones([2, 8])
let y = zeros([2, 64])

train(model = m, epochs = 1, grad_accumulation = 4):
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;

struct Built {
    ok: bool,
    stderr: String,
}

fn build(dir: &Path, src: &str, flags: &[&str]) -> Built {
    let src_path = dir.join("prog.nsl");
    fs::write(&src_path, src).unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(dir.join("prog.o"))
        .args(flags);
    let out = cmd.output().expect("spawn nsl build");
    Built {
        ok: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// The report line for one layer, e.g. `blocks.0: 12/12 heads, FFN=3072, …`.
fn layer_line<'a>(report: &'a str, layer: &str) -> &'a str {
    let key = format!("  {layer}: ");
    report
        .lines()
        .find(|l| l.starts_with(&key))
        .unwrap_or_else(|| panic!("no report line for `{layer}` in:\n{report}"))
}

#[test]
fn the_report_describes_the_model_that_was_compiled() {
    let tmp = TempDir::new().unwrap();
    let r = build(tmp.path(), GQA12, &["--wggo", "full", "--wggo-report"]);
    assert!(r.ok, "build failed:\n{}", r.stderr);
    assert!(
        r.stderr
            .contains("Layer shapes: from the model being compiled"),
        "{}",
        r.stderr
    );
    for layer in ["blocks.0", "blocks.1"] {
        let line = layer_line(&r.stderr, layer);
        assert!(line.contains("12/12 heads, FFN=3072"), "{line}");
    }
    assert!(!r.stderr.contains("8/8 heads"), "{}", r.stderr);
    assert!(!r.stderr.contains("FFN=2048"), "{}", r.stderr);
    // A model through the unpacked `forward` has nothing to pack: no
    // packing mode is planned and no `[pca]` verdict claims one was consumed.
    for layer in ["blocks.0", "blocks.1"] {
        assert!(
            layer_line(&r.stderr, layer).ends_with("PCA=none"),
            "{}",
            r.stderr
        );
    }
    assert!(!r.stderr.contains("wggo-override-consumed"), "{}", r.stderr);

    // Peak memory is the plan's: parameters and two 32-bit moments of every
    // layer plus activations. Parameters alone are
    //   2 blocks x (2 x 768 x 768 + 2 x 768 x 256 + 2 x 768 x 3072 + 2 x 768
    //   + 6 config scalars) + embed 64 x 768 + norm_out 768
    // which is over 12.6 million elements, 50 MB at f32 and 151 MB with the
    // moments. The report used to print 76.68 MB for every model.
    let mem_mb: f64 = r
        .stderr
        .lines()
        .find_map(|l| l.split("Peak memory: ").nth(1))
        .and_then(|rest| rest.split(' ').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no peak memory in:\n{}", r.stderr));
    assert!((151.0..165.0).contains(&mem_mb), "peak memory {mem_mb} MB");
}

#[test]
fn a_budget_is_checked_against_the_model() {
    let tmp = TempDir::new().unwrap();
    // 133 KB of parameters under a 200 MiB budget. This was refused: "needs
    // at least 296 MiB even at the fp16-moment floor".
    let r = build(
        tmp.path(),
        TINY,
        &[
            "--wggo",
            "full",
            "--wggo-report",
            "--wggo-memory-budget",
            "200",
        ],
    );
    assert!(r.ok, "a 200 MiB budget holds a 133 KB model:\n{}", r.stderr);
    for i in 0..8 {
        let line = layer_line(&r.stderr, &format!("blocks.{i}"));
        // No attention in this model, and nothing lowered when it all fits.
        assert!(
            line.contains("0/0 heads") && line.contains("m=32b v=32b"),
            "{line}"
        );
    }
    assert!(
        !r.stderr.contains("moment buffer(s) in BF16"),
        "a budget far above the model must not reduce any moment:\n{}",
        r.stderr
    );
}

/// A weights file is sliced per head of the model's own layers, and one
/// whose tensors have another shape stops the compile.
#[test]
fn a_weights_file_of_another_shape_is_refused() {
    let tmp = TempDir::new().unwrap();
    // `blocks.0.attn.wq` as an 8-head, 512-wide model would have it.
    let wrong = tmp.path().join("wrong.nslweights");
    write_nslweights(&wrong, &[("blocks.0.attn.wq", 512, 512)]);
    let r = build(
        tmp.path(),
        GQA12,
        &["--wggo", "full", "--wggo-weights", wrong.to_str().unwrap()],
    );
    assert!(
        !r.ok,
        "a weights file of another model's shape must be refused"
    );
    assert!(r.stderr.contains("blocks.0.attn.wq"), "{}", r.stderr);
    assert!(
        r.stderr.contains("[512, 512]") && r.stderr.contains("[768, 768]"),
        "{}",
        r.stderr
    );

    // The right shapes are accepted, and without a prune fraction no head
    // is pruned, whatever the scores (here the first three heads' columns
    // are a hundred times the rest). The plan used to drop a quarter of the
    // heads of any model given a weights file.
    let right = tmp.path().join("right.nslweights");
    write_nslweights(
        &right,
        &[
            ("blocks.0.attn.wq", 768, 768),
            ("blocks.1.attn.wq", 768, 768),
        ],
    );
    let r = build(
        tmp.path(),
        GQA12,
        &[
            "--wggo",
            "full",
            "--wggo-report",
            "--wggo-weights",
            right.to_str().unwrap(),
        ],
    );
    assert!(r.ok, "{}", r.stderr);
    assert!(
        layer_line(&r.stderr, "blocks.0").contains("12/12 heads"),
        "{}",
        r.stderr
    );
    assert!(
        r.stderr
            .contains("Weight analysis: 3 layers analysed, 0 without weights"),
        "{}",
        r.stderr
    );

    // With a fraction the solver may prune, but pruned heads rule out the
    // fused attention levels, and on this layer all heads at a fused level
    // cost less than nine heads without one. The solver now weighs that
    // itself: it keeps every head, and nothing is left for the conflict
    // resolver to undo. (It used to choose pruned heads with level 3, have
    // the level forced down to 1 afterwards, and print the cost from
    // before.)
    let r = build(
        tmp.path(),
        GQA12,
        &[
            "--wggo",
            "full",
            "--wggo-report",
            "--wggo-weights",
            right.to_str().unwrap(),
            "--wggo-prune-fraction",
            "0.25",
        ],
    );
    assert!(r.ok, "{}", r.stderr);
    let line = layer_line(&r.stderr, "blocks.0");
    assert!(line.contains("12/12 heads"), "{line}");
    assert!(
        line.contains("CSHA-L2") || line.contains("CSHA-L3"),
        "{line}"
    );
    assert!(r.stderr.contains("Conflicts resolved: 0"), "{}", r.stderr);
    assert!(!r.stderr.contains("Step time: inf"), "{}", r.stderr);
}

/// Write an `.nslweights` file of f32 matrices. Column block `h` of 64
/// columns is head `h`; the first three heads are 100.0 and the rest 1.0,
/// so the head scores are far apart and their order is known.
fn write_nslweights(path: &Path, tensors: &[(&str, usize, usize)]) {
    let mut params = Vec::new();
    let mut blob: Vec<u8> = Vec::new();
    for (name, rows, cols) in tensors {
        let offset = blob.len();
        for _ in 0..*rows {
            for c in 0..*cols {
                let v: f32 = if c < 3 * 64 { 100.0 } else { 1.0 };
                blob.extend_from_slice(&v.to_le_bytes());
            }
        }
        params.push(format!(
            r#"{{"name":"{name}","dtype":"f32","shape":[{rows},{cols}],"offset":{offset},"nbytes":{}}}"#,
            blob.len() - offset
        ));
    }
    let header = format!(r#"{{"params":[{}]}}"#, params.join(","));
    let mut out = Vec::new();
    out.extend_from_slice(b"NSLW");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    while out.len() % 64 != 0 {
        out.push(0);
    }
    out.extend_from_slice(&blob);
    fs::write(path, out).unwrap();
}

/// `@wggo(...)` on a train block was accepted and discarded.
#[test]
fn the_wggo_decorator_is_refused() {
    let tmp = TempDir::new().unwrap();
    let src = TINY.replace("train(model = m", "@wggo(mode=full)\ntrain(model = m");
    let src_path = tmp.path().join("deco.nsl");
    fs::write(&src_path, src).unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("@wggo").and(predicate::str::contains("--wggo")));
}
