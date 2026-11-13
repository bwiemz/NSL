//! `--checkpoint-blocks` must not change training for a block that draws
//! random numbers.
//!
//! Block checkpointing frees a block's interior activations after the
//! forward and recomputes them in the backward. That is exact for a pure
//! op and wrong for a random draw: the recomputed op is a second, different
//! sample, so the gradient is taken through activations the forward never
//! produced. Dropout was already kept out of the recompute set; `randn` was
//! not. On the program below the first loss was identical with and without
//! the flag (the forward is untouched) and the weights after one SGD step
//! were not.

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

const NOISY_BLOCK: &str = r#"from nsl.nn.losses import cross_entropy

model Blk:
    g: Tensor = ones([16])
    w_up: Tensor = randn([16, 32]) * 0.2
    w_down: Tensor = randn([32, 16]) * 0.2

    fn forward(self, h: Tensor) -> Tensor:
        let u = rmsnorm(h, self.g, 0.00001) @ self.w_up
        let v = silu(u * NOISE)
        return h + v @ self.w_down

model LM:
    embed: Tensor = randn([16, 16]) * 0.1
    blocks: [Blk; 3] = Blk()
    norm_out: Tensor = ones([16])

    fn forward_train(self, input_ids: Tensor) -> Tensor:
        let s = input_ids.shape
        let flat_ids = input_ids.reshape([s[0] * s[1]])
        let emb = embedding_lookup(self.embed, flat_ids)
        let h = emb.reshape([s[0], s[1], 16])
        for block in self.blocks:
            h = block.forward(h)
        let hn = rmsnorm(h, self.norm_out, 0.00001)
        return hn @ self.embed.transpose(0, 1)

let m = LM()
let unit = tensor_cat([arange(7) + full([7], 1.0), zeros([1])], 0)
let tokens = unit.reshape([1, 8]).expand([3, 8]).contiguous().reshape([24])
let loader = DataLoader(tokens, batch_size=2, seq_len=8, shuffle=false, drop_last=true)

train(model=m, epochs=1):
    optimizer: SGD(lr=0.05)
    step(batch):
        let logits = m.forward_train(batch.input_ids)
        let ls = logits.shape
        let flat_logits = logits.reshape([ls[0] * ls[1], ls[2]])
        let flat_labels = batch.labels.reshape([ls[0] * ls[1]])
        let loss = cross_entropy(flat_logits, flat_labels)

model_save(m, "out.nslm")
"#;

/// Trains `source` on the CPU under source AD and returns the saved model's
/// bytes and the run's stderr.
fn train(source: &str, checkpoint: bool) -> (Vec<u8>, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .arg("--target")
        .arg("cpu")
        .arg("--source-ad");
    if checkpoint {
        cmd.arg("--checkpoint-blocks");
    }
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run failed:\n{stderr}");
    (std::fs::read(dir.path().join("out.nslm")).unwrap(), stderr)
}

fn assert_checkpointing_is_exact(noise: &str) {
    let source = NOISY_BLOCK.replace("NOISE", noise);
    let (plain, _) = train(&source, false);
    let (checkpointed, stderr) = train(&source, true);
    assert!(
        stderr.contains("interior value(s) freed"),
        "--checkpoint-blocks freed nothing, so this run proves nothing:\n{stderr}"
    );
    assert!(
        plain == checkpointed,
        "the model trained with --checkpoint-blocks differs from the one \
         trained without it (noise term: `{noise}`)"
    );
}

#[test]
fn a_block_that_draws_randn_trains_the_same_with_checkpointing() {
    assert_checkpointing_is_exact("randn([2, 8, 32])");
}

/// The control: the same block with a constant in place of the draw. If
/// this ever fails, the difference above is not about randomness.
#[test]
fn the_same_block_without_a_draw_trains_the_same_with_checkpointing() {
    assert_checkpointing_is_exact("ones([2, 8, 32])");
}
