//! Two combinations of `packing = true` and `@flash_attention` that used to
//! compile and compute the wrong attention, and are now compile-time errors.
//!
//! A `dataset` block with `packing = true` makes the `@flash_attention`
//! training kernels of its module segment-masked: they read one segment id
//! per position and mask every pair of positions in different documents.
//!
//! 1. **Fed by a packing loader.** The train block hands those kernels the
//!    raw data pointers of the batch's `segment_ids` and `doc_starts`. The
//!    kernels load segment ids as 16-bit integers and document starts as
//!    32-bit integers; the loader stores both as 32-bit floats and leaves
//!    `doc_starts` on the host. A sequence of at most one tile would have
//!    run with every id read from the wrong bytes (the float 1.0 is the two
//!    16-bit words 0x0000 and 0x3F80), and a longer one was refused by the
//!    launcher with an error code that the call site turns into a trap.
//!    Packed attention has a working route,
//!    `GroupedQueryAttention.forward_packed`, which converts the ids
//!    properly; the message names it.
//! 2. **Without the causal mask.** The forward kernel emits the document
//!    test only as an extension of the causal predicate. Built with
//!    `causal = false` it emitted neither, so a "segment-masked" kernel
//!    attended across documents with nothing reported.
//!
//! Both are refused while the module is compiled, before anything runs, so
//! these tests need no GPU: the default target synthesizes the kernels.

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

fn program(dataset_packing: bool, flash_args: &str, loader: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

dataset PackedCorpus("docs"):
    packing = {dataset_packing}
    max_sequence_length = 64
    mean_doc_length = 16

model TinyAttn:
    w_norm: Tensor = ones([32])
    wq: Tensor = ones([32, 32])
    wk: Tensor = ones([32, 32])
    wv: Tensor = ones([32, 32])

    @flash_attention(head_dim=32{flash_args})
    fn forward(self, x: Tensor) -> Tensor:
        let x_norm = rmsnorm(x, self.w_norm, 0.00001)
        let q = x_norm @ self.wq
        let k = x_norm @ self.wk
        let v = x_norm @ self.wv
        let scale = 1.0 / sqrt(32.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
let x = ones([32, 32])
{loader}

train(model = m, epochs = 1, grad_accumulation = 2):
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

const PACKING_LOADER: &str = "let toks = zeros([256])\nlet loader = DataLoader(toks, batch_size=1, seq_len=64, shuffle=false, packing=true, pack_separator=0)";
const PLAIN_LOADER: &str = "let toks = zeros([256])\nlet loader = DataLoader(toks, batch_size=1, seq_len=64, shuffle=false)";

/// `nsl build` to an object file: compiles everything and runs nothing.
fn build(source: &str) -> (bool, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("build")
        .arg(&path)
        .arg("-o")
        .arg(dir.path().join("prog.out"))
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn a_packing_loader_cannot_feed_segment_masked_flash_attention() {
    let (ok, stderr) = build(&program(true, "", PACKING_LOADER));
    assert!(!ok, "the program compiled:\n{stderr}");
    assert!(
        stderr.contains(
            "a packed batch cannot be fed to this module's `@flash_attention` training kernels"
        ),
        "{stderr}"
    );
    assert!(
        stderr.contains("GroupedQueryAttention.forward_packed"),
        "{stderr}"
    );
}

/// The refusal is about the combination. Each half on its own compiles:
/// a packing loader in a module whose kernels are not segment-masked, and
/// segment-masked kernels with a loader that does not pack.
#[test]
fn each_half_of_the_combination_still_compiles() {
    for (what, source) in [
        (
            "packing loader, dataset not packed",
            program(false, "", PACKING_LOADER),
        ),
        (
            "packed dataset, plain loader",
            program(true, "", PLAIN_LOADER),
        ),
        ("packed dataset, no loader", program(true, "", "")),
    ] {
        let (ok, stderr) = build(&source);
        assert!(ok, "{what}: {stderr}");
    }
}

#[test]
fn segment_masked_flash_attention_requires_the_causal_mask() {
    let (ok, stderr) = build(&program(true, ", causal=false", ""));
    assert!(!ok, "the program compiled:\n{stderr}");
    assert!(
        stderr.contains("packed attention without a causal mask is not supported"),
        "{stderr}"
    );
    // Not packed: a bidirectional kernel is an ordinary request.
    let (ok, stderr) = build(&program(false, ", causal=false", ""));
    assert!(ok, "causal=false without packing: {stderr}");
}
