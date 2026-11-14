//! CPDT plans from facts of the model being compiled.
//!
//! Three things it used to take from constants or not report at all:
//!
//! * the number of blocks was assumed to be 8, so on a 12-block model block 7
//!   was treated as the last block (kept at full moment precision) and block
//!   11, the real last one, was reduced;
//! * `--cpdt-report` sized the model from a built-in layer ("params=12.98 MB"
//!   for a model of 0.13 MB), recommended full sharding for it, and did not
//!   say that the recommendation is not applied;
//! * an inference-only MoE program built with `--cpdt full` was refused as
//!   "silently inert" although the MoE passes had run.

use assert_cmd::prelude::*;
use safetensors::Dtype;
use safetensors::tensor::{TensorView, serialize};
use std::collections::HashMap;
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

/// `blocks` blocks of one 64 x 64 matrix, an 8 x 64 embedding and a
/// 64-element gain, trained with AdamW under gradient accumulation (the
/// configuration reduced-precision moments need).
fn tiny_net(blocks: usize) -> String {
    format!(
        r#"model Block:
    w: Tensor = ones([64, 64])

    fn forward(self, h: Tensor) -> Tensor:
        return h @ self.w

model TinyNet:
    embed: Tensor = ones([8, 64])
    blocks: [Block; {blocks}] = Block()
    final_norm: Tensor = ones([64])

    fn forward(self, x: Tensor) -> Tensor:
        let e = x @ self.embed
        let h = e * 0.0
        for block in self.blocks:
            h = h + block.forward(e)
        return h * self.final_norm

let m = TinyNet()
let x = ones([2, 8])

train(model = m, epochs = 1, grad_accumulation = 2):
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = sum(pred) * 0.0000625
"#
    )
}

/// A safetensors file holding one 64 x 64 f32 tensor of constant value.
///
/// 1e-4 puts a middle block in the Medium tier (reduced first moment): its
/// score is rms x position / elements = 1e-4 x 1.0 / 4096 = 2.44e-8, between
/// the tier bounds 2.232e-8 and 6.106e-8.
fn write_weights(path: &Path, name: &str) {
    write_tensors(path, &[(name, vec![64, 64], 1e-4)]);
}

/// A safetensors file of constant f32 tensors.
fn write_tensors(path: &Path, tensors: &[(&str, Vec<usize>, f32)]) {
    let blobs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, shape, value)| {
            std::iter::repeat_n(*value, shape.iter().product())
                .flat_map(|v| v.to_le_bytes())
                .collect()
        })
        .collect();
    let mut views = HashMap::new();
    for ((name, shape, _), bytes) in tensors.iter().zip(blobs.iter()) {
        views.insert(
            name.to_string(),
            TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
        );
    }
    fs::write(path, serialize(&views, None).unwrap()).unwrap();
}

fn run_with_weights(dir: &Path, tensor_name: &str) -> String {
    let src = dir.join("net12.nsl");
    fs::write(&src, tiny_net(12)).unwrap();
    let weights = dir.join(format!("{}.safetensors", tensor_name.replace('.', "_")));
    write_weights(&weights, tensor_name);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("run")
        .arg(&src)
        .arg("--source-ad")
        .arg("--weights")
        .arg(&weights)
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("2");
    let out = cmd.output().expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run failed:\n{stderr}");
    stderr
}

#[test]
fn the_last_block_is_the_models_own_last_block() {
    let tmp = TempDir::new().unwrap();

    // Block 7 of 12 is a middle block: its first moment is reduced. With the
    // count fixed at 8 it was taken for the last block and kept at FP32,
    // while the log said precision was active with 0 buffers.
    let stderr = run_with_weights(tmp.path(), "blocks.7.w");
    assert!(
        stderr.contains("1 first-moment buffer(s) in BF16 storage"),
        "block 7 of 12 is not the last block:\n{stderr}"
    );

    // Block 11 is the last block and stays at full precision; the log says
    // the plan reduces nothing, not that precision is active.
    let stderr = run_with_weights(tmp.path(), "blocks.11.w");
    assert!(stderr.contains("reduces no moment"), "{stderr}");
    assert!(
        !stderr.contains("first-moment buffer(s) in BF16 storage"),
        "{stderr}"
    );
}

/// A weights file whose key carries a container prefix of its own joins the
/// parameter it names. It used to join nothing: the compile was refused when
/// no name in the file matched, and with another matching name beside it the
/// prefixed one silently kept FP32 moments.
#[test]
fn a_prefixed_weights_key_joins_its_parameter() {
    let tmp = TempDir::new().unwrap();
    let stderr = run_with_weights(tmp.path(), "model.blocks.5.w");
    assert!(
        stderr.contains("1 first-moment buffer(s) in BF16 storage"),
        "`model.blocks.5.w` is this model's `blocks.5.w`:\n{stderr}"
    );
}

#[test]
fn the_cpdt_report_describes_the_model_and_says_it_is_not_applied() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("net8.nsl");
    fs::write(&src, tiny_net(8)).unwrap();
    // The CLI wants a weights file for any `--cpdt` mode; `zero_only` does
    // not read its values.
    let weights = tmp.path().join("w.safetensors");
    write_weights(&weights, "blocks.0.w");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("build")
        .arg(&src)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(tmp.path().join("net8.o"))
        .arg("--wggo")
        .arg("full")
        .arg("--weights")
        .arg(&weights)
        .arg("--cpdt")
        .arg("zero_only")
        .arg("--cpdt-num-gpus")
        .arg("8")
        .arg("--cpdt-memory-budget")
        .arg("24")
        .arg("--cpdt-report");
    let out = cmd.output().expect("spawn nsl build");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "build failed:\n{text}");

    // 8 x 4096 + 512 + 64 = 33,344 parameters at 4 bytes = 133,376 bytes.
    assert!(text.contains("params=0.13 MB"), "{text}");
    assert!(!text.contains("params=12.98 MB"), "{text}");
    // Everything fits on 8 GPUs of 24 GiB, so nothing needs sharding; the
    // planner used to recommend ZeRO-2 for this model.
    assert!(text.contains("s_p=1 s_g=1 s_os=1 (DDP)"), "{text}");
    assert!(text.contains("of a 24.00 GiB budget"), "{text}");
    // And the recommendation is labelled for what it is.
    assert!(text.contains("recommendation only, not applied"), "{text}");
    assert!(text.contains("--zero-stage is not set"), "{text}");
}

#[test]
fn an_inference_only_moe_program_compiles_under_cpdt_full() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe.nsl");
    fs::write(
        &src,
        r#"model MoEModel:
    @moe(num_experts=2, top_k=2, capacity_factor=2.0)
    experts: int = 2

    fn forward(self, x: Tensor) -> Tensor:
        let logits = ones([4, 2])
        let out = moe_dispatch(x, logits, self.experts)
        return out

let m = MoEModel()
let x = ones([4, 3])
let y = m.forward(x)
print(y)
"#,
    )
    .unwrap();
    // The router ([hidden 3, 2 experts]) and expert ([2 experts, 3 x 3])
    // weights the MoE lowering needs under `--cpdt full`. Router columns of
    // ones have norm sqrt(3), far above the dead-expert threshold.
    let weights = tmp.path().join("w.safetensors");
    write_tensors(
        &weights,
        &[
            ("MoEModel.experts.router.weight", vec![3, 2], 1.0),
            ("MoEModel.experts.experts.weight", vec![2, 9], 1.0),
        ],
    );
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        // Compile only: the refusal this test is about was raised after a
        // successful compile, before the program ran.
        .arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(tmp.path().join("moe.o"))
        .arg("--weights")
        .arg(&weights)
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("2");
    let out = cmd.output().expect("spawn nsl build");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !stderr.contains("silently inert") && !stderr.contains("recorded no disposition"),
        "the MoE passes ran; that is not an inert request:\n{stderr}"
    );
    assert!(out.status.success(), "{stderr}");
}
