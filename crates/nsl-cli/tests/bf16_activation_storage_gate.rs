//! `train(..., precision="bf16")`: bf16 activation storage, end to end.
//!
//! The storage plan (`nsl-codegen/src/stmt_train/storage_plan.rs`) is unit
//! tested on hand-built lists; this gate holds the real pipeline to it:
//!
//! * **Planning** (CPU build, default features): a model of two RMSNorm-fed
//!   SwiGLU blocks compiled with the production flag set plans both blocks
//!   and says so, and the same program without the key plans nothing.
//! * **Refusals**: the compositions this phase does not support are compile
//!   errors, never a silent f32 run: no `--source-ad`, a disagreeing
//!   `--matmul-mode`, `--cuda-graphs`, an unknown precision.
//! * **GPU** (`#[ignore]`, `--features cuda`): the program trains; the
//!   runtime prints the activation-storage banner; no dtype-explicit op took
//!   the widen/compute/narrow fallback; and every loss tracks the same
//!   program with f32 storage and bf16 GEMMs (`--matmul-mode bf16`) to the
//!   bf16 rounding of the activations.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

const PROD: &[&str] =
    &["--source-ad", "--checkpoint-blocks", "--checkpoint-selective", "--fuse-rmsnorm-backward", "--fuse-wgrad-accum"];

/// Two pre-norm SwiGLU blocks with residual adds, deterministic weights
/// scaled so every activation is O(1) and the loss is too. `eps` is large so
/// AdamW's update stays proportional to small gradient components instead
/// of taking their sign, which would turn a bf16 rounding into a full
/// learning-rate step. `precision` is the text after `grad_accumulation=4`.
/// `epochs=8` with accumulation 4: eight micro-batches, two optimizer steps.
fn fixture(precision: &str, cuda: bool) -> String {
    let (to_device, xd, yd) = if cuda { ("m.to(cuda)\n", "x.to(cuda)", "y.to(cuda)") } else { ("", "x", "y") };
    format!(
        r#"from nsl.nn.norms import RMSNorm
from nsl.nn.losses import mse_loss

model Block:
    norm: RMSNorm = RMSNorm(32)
    w_gate: Tensor = ((arange(2048).reshape([32, 64])) * 0.00007) - 0.06
    w_up: Tensor = ((arange(2048).reshape([32, 64])) * 0.00005) - 0.045
    w_down: Tensor = ((arange(2048).reshape([64, 32])) * 0.00004) - 0.04

    fn forward(self, x: Tensor) -> Tensor:
        let n = self.norm.forward(x)
        let gate = silu(n @ self.w_gate)
        let up = n @ self.w_up
        return x + (gate * up) @ self.w_down

model Net:
    b1: Block = Block()
    b2: Block = Block()

    fn forward(self, x: Tensor) -> Tensor:
        return self.b2.forward(self.b1.forward(x))

let m = Net()
{to_device}let x = ((arange(1024).reshape([2, 16, 32])) * 0.003) - 1.3
let y = ((arange(1024).reshape([2, 16, 32])) * 0.002) - 0.9
let xd = {xd}
let yd = {yd}

train(model=m, epochs=8, grad_accumulation=4{precision}):
    optimizer: AdamW(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-2, weight_decay=0.0)
    step(batch):
        let pred = m.forward(xd)
        let loss = mse_loss(pred, yd)
    callbacks:
        on_step(step, loss):
            print(loss)
"#
    )
}

struct Run {
    out: String,
    ok: bool,
}

fn nsl(args: &[&str], src: &str, stem: &str, cuda: bool, env: &[(&str, &str)]) -> Run {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_bf16_storage_gate_{stem}"));
    std::fs::create_dir_all(&tmp).expect("mkdir fixture dir");
    let fixture = tmp.join(format!("{stem}.nsl"));
    std::fs::write(&fixture, src).expect("write fixture");
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "-p", "nsl-cli"]);
    if cuda {
        cmd.args(["--features", "cuda"]);
    }
    cmd.arg("--manifest-path").arg(root.join("Cargo.toml")).arg("--");
    cmd.args(args).arg(&fixture);
    if args.first() == Some(&"build") {
        cmd.args(["--emit-obj", "-o"]).arg(tmp.join(format!("{stem}.o")));
    }
    cmd.current_dir(&root).env("NSL_STDLIB_PATH", root.join("stdlib")).env_remove("NSL_MATMUL_BF16");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let o = cmd.output().expect("spawn nsl");
    Run {
        out: format!("{}\n{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)),
        ok: o.status.success(),
    }
}

const BF16: &str = r#", precision="bf16""#;
const PLANNED: &str = "[bf16-storage] 2 SwiGLU MLP block(s) stored bf16";

#[test]
fn both_swiglu_blocks_are_planned_and_an_f32_program_plans_nothing() {
    let mut args = vec!["build"];
    args.extend_from_slice(PROD);
    let r = nsl(&args, &fixture(BF16, false), "plan_bf16", false, &[]);
    assert!(r.ok, "precision=\"bf16\" build failed:\n{}", r.out);
    assert!(r.out.contains(PLANNED), "the plan did not cover both blocks:\n{}", r.out);
    // Every chain still fuses: the plan must not cost the wgrad fusion.
    assert!(r.out.contains("[wgrad-fusion] 6 chain(s) fused"), "{}", r.out);

    let r = nsl(&args, &fixture("", false), "plan_f32", false, &[]);
    assert!(r.ok, "f32 build failed:\n{}", r.out);
    assert!(!r.out.contains("[bf16-storage]"), "an f32 program must not be planned:\n{}", r.out);
}

#[test]
fn compositions_this_phase_does_not_support_are_compile_errors() {
    let src = fixture(BF16, false);
    let cases: [(&[&str], &[(&str, &str)], &str, &str); 4] = [
        (&["build"], &[], "requires --source-ad", "no_source_ad"),
        (&["build", "--source-ad", "--matmul-mode", "tf32"], &[], "--matmul-mode tf32 was given", "mm_tf32"),
        (&["build", "--source-ad", "--cuda-graphs"], &[], "does not compose with --cuda-graphs", "graphs"),
        (&["build", "--source-ad"], &[("NSL_MATMUL_BF16", "0")], "NSL_MATMUL_BF16=0", "env0"),
    ];
    for (args, env, want, stem) in cases {
        let r = nsl(args, &src, stem, false, env);
        assert!(!r.ok && r.out.contains(want), "{args:?} {env:?}: expected a compile error containing {want:?}:\n{}", r.out);
    }
    let r = nsl(&["build", "--source-ad"], &fixture(r#", precision="fp8""#, false), "fp8", false, &[]);
    assert!(!r.ok && r.out.contains(r#"must be "f32" or "bf16""#), "{}", r.out);
}

fn losses(out: &str) -> Vec<f64> {
    out.lines()
        .filter_map(|l| l.trim().strip_prefix("tensor([").and_then(|r| r.strip_suffix("])")))
        .filter_map(|v| v.parse::<f64>().ok())
        .collect()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_bf16_program_trains_like_f32_storage_with_bf16_gemms() {
    let mut bf16_args = vec!["run"];
    bf16_args.extend_from_slice(PROD);
    let bf16 = nsl(&bf16_args, &fixture(BF16, true), "gpu_bf16", true, &[]);
    assert!(bf16.ok, "precision=\"bf16\" run failed:\n{}", bf16.out);
    assert!(bf16.out.contains(PLANNED), "{}", bf16.out);
    assert!(bf16.out.contains("activation storage: bf16"), "the runtime banner is missing:\n{}", bf16.out);
    assert!(
        !bf16.out.contains("widens to f32, computes and narrows"),
        "a dtype-explicit op took the fallback (strided, host or broadcast operand):\n{}",
        bf16.out
    );

    let mut ref_args = bf16_args.clone();
    ref_args.extend_from_slice(&["--matmul-mode", "bf16"]);
    let reference = nsl(&ref_args, &fixture("", true), "gpu_ref", true, &[]);
    assert!(reference.ok, "reference run failed:\n{}", reference.out);
    assert!(!reference.out.contains("activation storage: bf16"), "{}", reference.out);

    let (a, b) = (losses(&bf16.out), losses(&reference.out));
    eprintln!("losses, bf16 storage: {a:?}\nlosses, f32 storage:  {b:?}");
    assert_eq!(a.len(), 8, "one loss per micro-batch:\n{}", bf16.out);
    assert_eq!(a.len(), b.len());
    assert!(b[7] < b[0], "the reference did not train: {b:?}");
    // Both arms round every GEMM operand to bf16; this one also stores the
    // MLP's activations bf16 (one more rounding, relative 2^-9, per value),
    // which moves a mean-squared loss by about that much, not by more.
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() <= 1e-2 * y.abs(), "step {i}: bf16 storage {x} vs f32 storage {y}\n{a:?}\n{b:?}");
    }
}
