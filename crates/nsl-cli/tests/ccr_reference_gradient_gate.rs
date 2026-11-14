//! Block checkpointing against gradients computed outside NSL.
//!
//! Every other CCR gate compares NSL with NSL: a checkpointed run against
//! the same program without the flag. That shows the two agree and says
//! nothing about whether either is right, and it cannot see a change that
//! moves both. This gate trains a small model for three plain-SGD steps
//! under each checkpointing configuration and compares the losses and every
//! trained weight with values PyTorch produced for the same model.
//!
//! # The model
//!
//! Three residual feed-forward blocks (`h + silu(rmsnorm(h) @ w_up) @
//! w_down`) between an embedding and a tied output head, batch 2, sequence
//! 8, width 8, hidden width 16, vocabulary 8. The weights are not random and
//! not constant: each is a sine of its flat index (`sin(i * a + b) * c`), so
//! the same tensors can be written down in Python. Learning rate 0.5, so
//! three steps move the weights by amounts far above f32 rounding.
//!
//! # The reference
//!
//! The literals below were produced by PyTorch 2.15 (CPU) in float64, with
//! `torch.autograd.grad` and a hand-written SGD update, mirroring the NSL
//! program line for line, including the DataLoader's label rule (each
//! sequence is shifted by one and its last position is ignored). The same
//! script in float32 differs from them by at most 1.6e-5, which is what f32
//! arithmetic costs over three steps; the tolerance is 1e-4.
//!
//! For each weight tensor two numbers are compared: its sum, and its dot
//! product with `sin(i * 0.61 + 0.3)` over the flat index, which is
//! sensitive to where in the tensor a change lands.

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

// Produced by PyTorch in float64; see the module header.
const LOSSES: [f64; 3] = [1.971622220786, 1.264322000565, 1.039737114485];
// (name, sum, probe)
const WEIGHTS: [(&str, f64, f64); 11] = [
    ("embed", 0.489847051058, -0.311795512898),
    ("norm_out", 7.861171703429, 1.854339930733),
    ("g0", 7.760795371298, 2.119381883363),
    ("up0", 2.177427446521, -0.320290263555),
    ("down0", 0.322081514821, -1.437877781531),
    ("g1", 8.122691247033, 2.238603031740),
    ("up1", 1.924559305776, -0.276053967681),
    ("down1", 0.313879463220, -1.613985806906),
    ("g2", 8.180574406963, 2.183845375949),
    ("up2", 1.947712700651, -0.273206042727),
    ("down2", 0.142454923700, -1.675856545657),
];

/// What f32 arithmetic costs against the float64 reference (measured:
/// 1.6e-5 at worst), with room.
const EXACT_TOLERANCE: f64 = 1e-4;

/// With `--checkpoint-compress bf16` the backward reads activations rounded
/// to bf16, which keeps 8 significant bits: a relative error of up to 2^-9
/// (0.2%) on each saved activation, and so on each gradient it enters. The
/// weights here are O(1) and move by O(0.1) over the three steps, so 0.2% of
/// the movement is a few 1e-4 per element; the sums are over up to 128
/// elements. 2e-2 bounds that with room and is still far below the O(0.1)
/// a wrong gradient moves these numbers by.
const COMPRESSED_TOLERANCE: f64 = 2e-2;

const PROGRAM: &str = r#"from nsl.nn.losses import cross_entropy

model Blk:
    g: Tensor<[8], f32> = ones([8]) + tensor_sin(arange(8) * 0.9) * 0.2
    w_up: Tensor<[8, 16], f32> = tensor_sin(arange(128) * 0.37 + full([128], 0.5)).reshape([8, 16]) * 0.4
    w_down: Tensor<[16, 8], f32> = tensor_sin(arange(128) * 0.73 + full([128], 1.1)).reshape([16, 8]) * 0.4

    fn forward(self, h: Tensor<[2, 8, 8], f32>) -> Tensor<[2, 8, 8], f32>:
        let n: Tensor<[2, 8, 8], f32> = rmsnorm(h, self.g, 0.00001)
        let u: Tensor<[2, 8, 16], f32> = n @ self.w_up
        let v: Tensor<[2, 8, 16], f32> = silu(u)
        let f: Tensor<[2, 8, 8], f32> = v @ self.w_down
        return h + f

model LM:
    embed: Tensor<[8, 8], f32> = tensor_sin(arange(64) * 0.51 + full([64], 0.2)).reshape([8, 8]) * 0.5
    blocks: [Blk; 3] = Blk()
    norm_out: Tensor<[8], f32> = ones([8])

    fn forward_train(self, input_ids: Tensor) -> Tensor:
        let flat_ids = input_ids.reshape([16])
        let emb = embedding_lookup(self.embed, flat_ids)
        let h: Tensor<[2, 8, 8], f32> = emb.reshape([2, 8, 8])
        for block in self.blocks:
            h = block.forward(h)
        let hn = rmsnorm(h, self.norm_out, 0.00001)
        return hn @ self.embed.transpose(0, 1)

fn total(t: Tensor) -> Tensor:
    return t.sum()

fn probe(t: Tensor, n: int) -> Tensor:
    let flat = t.reshape([n])
    return (flat * tensor_sin(arange(n) * 0.61 + full([n], 0.3))).sum()

let m = LM()
# GPU_PLACEMENT
let unit = tensor_cat([arange(7) + full([7], 1.0), zeros([1])], 0)
let tokens = unit.reshape([1, 8]).expand([7, 8]).contiguous().reshape([56])
let loader = DataLoader(tokens, batch_size=2, seq_len=8, shuffle=false, drop_last=true)

print("LOSSES_BEGIN")
train(model=m, epochs=1):
    optimizer: SGD(lr=0.5)
    step(batch):
        let logits = m.forward_train(batch.input_ids)
        let flat_logits = logits.reshape([16, 8])
        let flat_labels = batch.labels.reshape([16])
        let loss = cross_entropy(flat_logits, flat_labels)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSSES_END")

print("WEIGHTS_BEGIN")
let embed: Tensor = m.embed
let embed_sum: Tensor = total(embed)
print(embed_sum)
let embed_probe: Tensor = probe(embed, 64)
print(embed_probe)
let norm_out: Tensor = m.norm_out
let norm_out_sum: Tensor = total(norm_out)
print(norm_out_sum)
let norm_out_probe: Tensor = probe(norm_out, 8)
print(norm_out_probe)
BLOCK_PROBES
print("WEIGHTS_END")
"#;

/// The probe statements for block `i`: `g`, `w_up`, `w_down`, each as sum
/// then probe, in the order of `WEIGHTS`.
fn block_probes(i: usize) -> String {
    let mut out = format!("let b{i}: Blk = m.blocks[{i}]\n");
    for (field, n) in [("g", 8), ("w_up", 128), ("w_down", 128)] {
        out.push_str(&format!(
            "let {field}{i}: Tensor = b{i}.{field}\n\
             let {field}{i}_sum: Tensor = total({field}{i})\n\
             print({field}{i}_sum)\n\
             let {field}{i}_probe: Tensor = probe({field}{i}, {n})\n\
             print({field}{i}_probe)\n"
        ));
    }
    out
}

fn program(gpu: bool) -> String {
    let probes: String = (0..3).map(block_probes).collect();
    let source = PROGRAM.replace("BLOCK_PROBES\n", &probes);
    assert!(source.contains("# GPU_PLACEMENT"));
    if gpu {
        source.replace("# GPU_PLACEMENT", "m.to(cuda)")
    } else {
        source
    }
}

struct Run {
    /// The text between the markers, as printed: the bit-exact key.
    printed: String,
    losses: Vec<f64>,
    weights: Vec<f64>,
    stderr: String,
}

fn section(stdout: &str, name: &str) -> Vec<f64> {
    let begin = format!("{name}_BEGIN");
    let end = format!("{name}_END");
    let after = stdout
        .split_once(&begin)
        .unwrap_or_else(|| panic!("no {begin} in:\n{stdout}"))
        .1;
    let inner = after
        .split_once(&end)
        .unwrap_or_else(|| panic!("no {end} in:\n{stdout}"))
        .0;
    inner
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            // A CPU scalar prints bare; a device scalar prints `tensor([x])`.
            let number = l
                .strip_prefix("tensor([")
                .and_then(|rest| rest.strip_suffix("])"))
                .unwrap_or(l);
            let v: f64 = number
                .parse()
                .unwrap_or_else(|_| panic!("`{l}` in section {name} is not a number"));
            assert!(v.is_finite(), "non-finite value `{l}` in section {name}");
            v
        })
        .collect()
}

/// Train `PROGRAM` on the CPU under source AD with `flags`.
fn train(flags: &[&str]) -> Run {
    train_on(false, flags)
}

/// Train `PROGRAM` under source AD with `flags`, with the model on the GPU
/// when `gpu` (which needs a `cuda` build of this test).
fn train_on(gpu: bool, flags: &[&str]) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ccr_reference.nsl");
    std::fs::write(&path, program(gpu)).unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .arg("--source-ad");
    if !gpu {
        cmd.args(["--target", "cpu"]);
    }
    let out = cmd.args(flags).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run with {flags:?} failed:\n{stderr}");
    Run {
        printed: stdout.clone(),
        losses: section(&stdout, "LOSSES"),
        weights: section(&stdout, "WEIGHTS"),
        stderr,
    }
}

/// Compare a run with the PyTorch literals.
fn assert_matches_reference(run: &Run, flags: &[&str], tolerance: f64) {
    assert_eq!(run.losses.len(), LOSSES.len(), "{flags:?}: loss count");
    for (step, (got, want)) in run.losses.iter().zip(LOSSES).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "{flags:?}: loss at step {step} is {got}, PyTorch gives {want}"
        );
    }
    assert_eq!(
        run.weights.len(),
        2 * WEIGHTS.len(),
        "{flags:?}: probe count"
    );
    for (i, (name, want_sum, want_probe)) in WEIGHTS.into_iter().enumerate() {
        let (got_sum, got_probe) = (run.weights[2 * i], run.weights[2 * i + 1]);
        assert!(
            (got_sum - want_sum).abs() <= tolerance,
            "{flags:?}: sum of `{name}` after training is {got_sum}, PyTorch gives {want_sum}"
        );
        assert!(
            (got_probe - want_probe).abs() <= tolerance,
            "{flags:?}: probe of `{name}` after training is {got_probe}, PyTorch gives \
             {want_probe}"
        );
    }
}

/// The reference describes NSL's source AD at all: without it nothing below
/// would mean anything.
#[test]
fn training_without_checkpointing_matches_pytorch() {
    let flags: [&str; 0] = [];
    assert_matches_reference(&train(&flags), &flags, EXACT_TOLERANCE);
}

/// Every configuration that only changes what is saved and what is
/// recomputed must give PyTorch's weights, and must print exactly what the
/// run without checkpointing prints.
#[test]
fn checkpointed_training_matches_pytorch() {
    let plain = train(&[]);
    let configurations: [&[&str]; 6] = [
        &["--checkpoint-blocks"],
        &["--checkpoint-blocks", "--checkpoint-selective"],
        &["--checkpoint-blocks", "--checkpoint-stride", "2"],
        &["--checkpoint-blocks", "--checkpoint-stride", "3"],
        &["--checkpoint-blocks", "--checkpoint-stride", "dp"],
        // 1 MiB is more than every interior of this model together, so the
        // knapsack flips them all back to SAVE: the plan with nothing left
        // to recompute must still train correctly.
        &["--checkpoint-blocks", "--checkpoint-budget-mib", "1"],
    ];
    for flags in configurations {
        let run = train(flags);
        assert!(
            run.stderr.contains("[ccr]"),
            "{flags:?}: the planner printed nothing, so checkpointing did not run:\n{}",
            run.stderr
        );
        assert_matches_reference(&run, flags, EXACT_TOLERANCE);
        assert_eq!(
            run.printed, plain.printed,
            "{flags:?}: recomputation must reproduce the forward exactly"
        );
    }
}

/// Compressed saves are lossy, so they are held to the bound bf16 rounding
/// allows and must NOT reproduce the uncompressed run bit for bit (if they
/// did, nothing was compressed).
#[test]
fn compressed_saves_stay_within_bf16_rounding_of_pytorch() {
    let flags = [
        "--checkpoint-blocks",
        "--checkpoint-selective",
        "--checkpoint-compress",
        "bf16",
    ];
    let run = train(&flags);
    assert!(
        run.stderr
            .contains("compressed saves: 6 matmul-class tensors -> bf16"),
        "the two matrix products of each of the 3 blocks must be compressed:\n{}",
        run.stderr
    );
    assert_matches_reference(&run, &flags, COMPRESSED_TOLERANCE);
    // The first loss is a pure forward and is untouched by compression.
    assert!((run.losses[0] - LOSSES[0]).abs() <= EXACT_TOLERANCE);
    let exact = train(&["--checkpoint-blocks", "--checkpoint-selective"]);
    assert_ne!(
        run.printed, exact.printed,
        "the compressed run is bit-identical to the uncompressed one: nothing was rounded"
    );
}

/// `--checkpoint-compress fp16` was withdrawn: half precision overflows above
/// 65504 and the cast did not check the range, so the loss became NaN. The
/// flag must reject the value, not accept it and compress as before.
#[test]
fn fp16_is_not_an_accepted_compression_format() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ccr_reference.nsl");
    std::fs::write(&path, program(false)).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .args(["--target", "cpu", "--source-ad", "--checkpoint-blocks"])
        .args(["--checkpoint-selective", "--checkpoint-compress", "fp16"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "fp16 was accepted:\n{stderr}");
    assert!(
        stderr.contains("invalid value 'fp16'") && stderr.contains("--checkpoint-compress"),
        "expected the flag to reject the value:\n{stderr}"
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).trim().is_empty(),
        "nothing may have run"
    );
}

/// The largest absolute difference between a run and the PyTorch literals,
/// over the losses and every weight probe.
fn deviation_from_reference(run: &Run) -> f64 {
    assert_eq!(run.losses.len(), LOSSES.len());
    assert_eq!(run.weights.len(), 2 * WEIGHTS.len());
    let losses = run
        .losses
        .iter()
        .zip(LOSSES)
        .map(|(got, want)| (got - want).abs());
    let weights = WEIGHTS
        .into_iter()
        .enumerate()
        .flat_map(|(i, (_, sum, probe))| {
            [
                (run.weights[2 * i] - sum).abs(),
                (run.weights[2 * i + 1] - probe).abs(),
            ]
        });
    losses.chain(weights).fold(0.0, f64::max)
}

/// The largest absolute difference between two runs of the same program.
fn deviation_between(a: &Run, b: &Run) -> f64 {
    assert_eq!(a.losses.len(), b.losses.len());
    assert_eq!(a.weights.len(), b.weights.len());
    a.losses
        .iter()
        .chain(&a.weights)
        .zip(b.losses.iter().chain(&b.weights))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// What the device's own arithmetic costs against the float64 reference, for
/// the run WITHOUT checkpointing: measured at 6.6e-4 on an RTX PRO 4500
/// Blackwell (1.0e-4 already on the first loss, a pure forward), against
/// 1.6e-5 on the CPU. That distance belongs to the device kernels, not to checkpointing,
/// and this bound only keeps it from growing unnoticed.
const GPU_REFERENCE_TOLERANCE: f64 = 1e-3;

/// How far a checkpointed device run may be from the plain device run. They
/// execute the same kernels on the same inputs and were measured identical
/// to the last printed digit; the allowance is for the device embedding
/// backward, whose scatter order is not guaranteed to be reproducible.
const GPU_PARITY_TOLERANCE: f64 = 1e-5;

/// The same comparison with the model on the GPU: the device kernels, the
/// device shape stubs and the device bf16 casts.
///
/// Two things are asserted. The plain device run must match PyTorch within
/// `GPU_REFERENCE_TOLERANCE`, which is looser than the CPU bound because the
/// device forward itself differs more from float64.
/// And every checkpointed device run must match the plain device run within
/// `GPU_PARITY_TOLERANCE`, which is the property checkpointing owes: it is
/// far tighter than the reference bound, so a recompute that changed the
/// gradients cannot hide inside the device's distance from PyTorch.
///
/// `cargo test -p nsl-cli --features cuda --test ccr_reference_gradient_gate
/// -- --ignored --test-threads=1`
#[test]
#[ignore = "requires CUDA GPU"]
fn checkpointed_training_matches_pytorch_gpu() {
    let plain = train_on(true, &[]);
    let plain_dev = deviation_from_reference(&plain);
    eprintln!("[ccr-reference] gpu, no checkpointing: {plain_dev:.3e} from PyTorch");
    assert!(
        plain_dev <= GPU_REFERENCE_TOLERANCE,
        "the plain GPU run is {plain_dev:.3e} from PyTorch"
    );
    let configurations: [&[&str]; 4] = [
        &["--checkpoint-blocks"],
        &["--checkpoint-blocks", "--checkpoint-selective"],
        &["--checkpoint-blocks", "--checkpoint-stride", "2"],
        &["--checkpoint-blocks", "--checkpoint-stride", "dp"],
    ];
    for flags in configurations {
        let run = train_on(true, flags);
        assert!(
            run.stderr.contains("[ccr]"),
            "{flags:?}: the planner printed nothing, so checkpointing did not run:\n{}",
            run.stderr
        );
        let (from_reference, from_plain) = (
            deviation_from_reference(&run),
            deviation_between(&run, &plain),
        );
        eprintln!(
            "[ccr-reference] gpu, {flags:?}: {from_reference:.3e} from PyTorch, \
             {from_plain:.3e} from the plain GPU run"
        );
        assert!(
            from_plain <= GPU_PARITY_TOLERANCE,
            "{flags:?}: the checkpointed GPU run is {from_plain:.3e} from the plain GPU run"
        );
        assert!(
            from_reference <= GPU_REFERENCE_TOLERANCE,
            "{flags:?}: the checkpointed GPU run is {from_reference:.3e} from PyTorch"
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn compressed_saves_stay_within_bf16_rounding_of_pytorch_gpu() {
    let flags = [
        "--checkpoint-blocks",
        "--checkpoint-selective",
        "--checkpoint-compress",
        "bf16",
    ];
    let run = train_on(true, &flags);
    assert!(
        run.stderr
            .contains("compressed saves: 6 matmul-class tensors -> bf16"),
        "the two matrix products of each of the 3 blocks must be compressed:\n{}",
        run.stderr
    );
    let from_reference = deviation_from_reference(&run);
    eprintln!("[ccr-reference] gpu, bf16 compressed saves: {from_reference:.3e} from PyTorch");
    assert!(
        from_reference <= COMPRESSED_TOLERANCE,
        "the compressed GPU run is {from_reference:.3e} from PyTorch"
    );
    // The first loss is a pure forward and is untouched by compression.
    assert!((run.losses[0] - LOSSES[0]).abs() <= GPU_REFERENCE_TOLERANCE);
}
