//! Two ways a `@wrga` / `@adapter` program stopped training, or stopped.
//!
//! 1. `@wrga(mode = manual)` and `@wrga(mode = hybrid, layers = [...])`
//!    trained NOTHING under `--source-ad` and exited 0. The WRGA prune pass
//!    builds its trainable set by matching the decorator's patterns against
//!    parameter names; a `Class.field` target (`Toy.w`) and a bare layer
//!    prefix (`b1`) never match an instance path (`m.w`, `m.b1.wq`), so the
//!    set was empty and the source-AD driver dropped every adjoint op.
//!    Under tape AD the mode was ignored and everything trained. The two AD
//!    modes now agree: `@freeze` alone decides what trains.
//!
//!    `mode = hybrid` is now refused outright: it asks for automatic
//!    placement, which does not exist. `budget=` and `target=` are held to
//!    what they say (see the tests of each).
//!
//! 2. Any `@adapter` on an activation of rank 3 aborted at the first
//!    optimizer step when the build pinned a target (`--target sm_80`).
//!    A pinned target alone selected the fused adapter call, whose
//!    adjoints assume 2-D activations: for `x = [2, 3, 4]` the weight
//!    gradient came out `[2, 4, 4]` and the step died on a length check.
//!    The fused path now also needs `NSL_WRGA_FUSED_CUDA=1`.

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

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run(source: &str, args: &[&str]) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .env_remove("NSL_WRGA_FUSED_CUDA")
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .args(args)
        .output()
        .unwrap();
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// The numbers printed between `<tag>_BEGIN` and `<tag>_END`.
fn section(stdout: &str, tag: &str) -> Vec<f64> {
    let begin = format!("{tag}_BEGIN");
    let end = format!("{tag}_END");
    let start = stdout.find(&begin).unwrap_or_else(|| panic!("no {begin} in:\n{stdout}"));
    let stop = stdout.find(&end).unwrap_or_else(|| panic!("no {end} in:\n{stdout}"));
    stdout[start + begin.len()..stop]
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

fn largest_change(before: &[f64], after: &[f64]) -> f64 {
    assert_eq!(before.len(), after.len());
    before
        .iter()
        .zip(after)
        .map(|(b, a)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn lora_program(decorators: &str, x_shape: &str, count: usize) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = randn([4, 4]) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

{decorators}
let m = Toy()
let w0: Tensor = m.w
print("W0_BEGIN")
print(w0)
print("W0_END")

let x = (arange({count}).reshape({x_shape})) * 0.1 + full({x_shape}, 0.1)
let y = (arange({count}).reshape({x_shape})) * 0.2 - full({x_shape}, 1.0)

train(model = m, epochs = 4):
    optimizer: SGD(lr = 0.1)
    step(batch):
        let loss = mse_loss(m.forward(x), y)

print("W_BEGIN")
print(m.w)
print("W_END")
let b: Tensor = m.lora_B_Toy_w__lora
print("B_BEGIN")
print(b)
print("B_END")
"#
    )
}

/// `mode = manual` with an adapter: the base weight and the adapter both
/// train, under source AD exactly as under tape AD.
#[test]
fn mode_manual_trains_under_both_ad_modes() {
    let source = lora_program(
        "@wrga(mode=manual)\n@adapter(type=lora, target=[\"Toy.w\"], rank=2, alpha=4)",
        "[3, 4]",
        12,
    );
    let mut finals: Vec<Vec<f64>> = Vec::new();
    for ad in [&["--target", "cpu"][..], &["--target", "cpu", "--source-ad"][..]] {
        let r = run(&source, ad);
        assert!(r.ok, "{ad:?} failed:\n{}", r.stderr);
        let w0 = section(&r.stdout, "W0");
        let w = section(&r.stdout, "W");
        let b = section(&r.stdout, "B");
        assert!(
            largest_change(&w0, &w) > 1e-3,
            "{ad:?}: the base weight did not move in 4 SGD steps"
        );
        assert!(
            b.iter().any(|v| v.abs() > 1e-4),
            "{ad:?}: LoRA B is still all zeros after training"
        );
        finals.push(w);
    }
    assert!(
        largest_change(&finals[0], &finals[1]) < 1e-5,
        "tape AD and source AD trained mode=manual to different weights"
    );
}

/// `mode = hybrid` asks the compiler to place adapters inside a layer scope.
/// Nothing places an adapter automatically, so the mode is refused; it used
/// to be accepted and to change nothing.
#[test]
fn mode_hybrid_is_refused() {
    let source = r#"from nsl.nn.losses import mse_loss

model Blk:
    wq: Tensor = randn([8, 8]) * 0.2

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.wq

model Net:
    b0: Blk = Blk()
    b1: Blk = Blk()

    fn forward(self, x: Tensor) -> Tensor:
        return self.b1.forward(self.b0.forward(x))

@wrga(mode=hybrid, layers=["b1"])
let m = Net()
print("RAN")
"#;
    for ad in [&["--target", "cpu"][..], &["--target", "cpu", "--source-ad"][..]] {
        let r = run(source, ad);
        assert!(!r.ok && !r.stdout.contains("RAN"), "{ad:?}: mode=hybrid should not build");
        assert!(
            r.stderr.contains("@wrga(mode=hybrid) is not supported"),
            "{ad:?}: the error should name the mode:\n{}",
            r.stderr
        );
    }
}

/// Six 8×8 weights, each with a rank-2 LoRA adapter: 6 × 2 × (8 + 8) = 192
/// adapter parameters.
fn budget_program(budget: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Blk:
    wq: Tensor = randn([8, 8]) * 0.2
    wk: Tensor = randn([8, 8]) * 0.2

    fn forward(self, x: Tensor) -> Tensor:
        return (x @ self.wq) @ self.wk

model Net:
    blocks: [Blk; 3] = Blk()

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for blk in self.blocks:
            h = blk.forward(h)
        return h

@wrga({budget})
@freeze
@adapter(type=lora, target=["Blk.wq", "Blk.wk"], rank=2)
let m = Net()
let x = randn([4, 8])
let y = randn([4, 8])
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.05)
    step(batch):
        let loss = mse_loss(m.forward(x), y)
print("RAN")
"#
    )
}

/// `@wrga(budget=N)` is a cap on the parameters the declared adapters hold.
/// It used to reach only the report's rank suggestions, as a number of
/// ranks, and bound nothing.
#[test]
fn a_budget_the_adapters_exceed_fails_the_build() {
    for ad in [&["--target", "cpu"][..], &["--target", "cpu", "--source-ad"][..]] {
        // 192 parameters: a budget of 192 holds them, 191 does not.
        let fits = run(&budget_program("budget=192"), ad);
        assert!(fits.ok && fits.stdout.contains("RAN"), "{ad:?}:\n{}", fits.stderr);

        let over = run(&budget_program("budget=191"), ad);
        assert!(!over.ok && !over.stdout.contains("RAN"), "{ad:?}: 192 > 191 should not build");
        assert!(
            over.stderr.contains("@wrga(budget=191)")
                && over.stderr.contains("hold 192 parameters")
                && over.stderr.contains("1 over the budget"),
            "{ad:?}: the error should give the budget and the count:\n{}",
            over.stderr
        );

        // No budget: nothing to exceed.
        let free = run(&budget_program("mode=manual"), ad);
        assert!(free.ok && free.stdout.contains("RAN"), "{ad:?}:\n{}", free.stderr);
    }
}

/// `@wrga(target=...)` names the GPU the analysis runs for. A name the GPU
/// table does not have used to fall back to the default GPU without a word.
#[test]
fn an_unknown_target_gpu_fails_the_build() {
    for ad in [&["--target", "cpu"][..], &["--target", "cpu", "--source-ad"][..]] {
        let r = run(&budget_program("target=rtx9999"), ad);
        assert!(!r.ok && !r.stdout.contains("RAN"), "{ad:?}: rtx9999 is not a GPU");
        assert!(
            r.stderr.contains("WRGA target 'rtx9999'") && r.stderr.contains("H100-SXM"),
            "{ad:?}: the error should name the target and list the known ones:\n{}",
            r.stderr
        );
        for known in ["target=h100", "target=rtx5070ti", "target=\"RTX-4090\""] {
            let ok = run(&budget_program(known), ad);
            assert!(ok.ok && ok.stdout.contains("RAN"), "{ad:?} {known}:\n{}", ok.stderr);
        }
    }
}

/// An adapter tensor read off a model is a tensor to the type checker, so an
/// expression on it prints as a number. `print(m.lora_B_... .sum())` used to
/// print the tensor's handle, a large integer.
#[test]
fn an_expression_on_an_adapter_tensor_prints_its_value() {
    let source = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = randn([4, 4]) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=ia3, target=["Toy.w"])
let m = Toy()
print("SUM_BEGIN")
print(m.ia3_scale_Toy_w__ia3.sum())
print("SUM_END")
"#;
    let r = run(source, &["--target", "cpu"]);
    assert!(r.ok, "{}", r.stderr);
    // The IA³ scale is four ones.
    assert_eq!(section(&r.stdout, "SUM"), [4.0], "stdout:\n{}", r.stdout);
}

/// A LoRA adapter on a `[batch, seq, d]` activation, built for a pinned
/// target, trains to the same weights as the unpinned build. The program
/// runs on the host either way; only the adapter lowering differs.
///
/// Built only with the `cuda` feature: a pinned `sm_NN` target is a request
/// for a CUDA build, which a runtime without it may decline.
#[cfg(feature = "cuda")]
#[test]
fn an_adapter_on_a_rank_3_activation_trains_when_the_target_is_pinned() {
    let source = lora_program(
        "@adapter(type=lora, target=[\"Toy.w\"], rank=2, alpha=4)",
        "[2, 3, 4]",
        24,
    );
    let plain = run(&source, &["--target", "cpu", "--source-ad"]);
    assert!(plain.ok, "unpinned build failed:\n{}", plain.stderr);
    let pinned = run(&source, &["--target", "sm_80", "--source-ad"]);
    assert!(
        pinned.ok,
        "the build pinned to sm_80 failed (the fused adapter adjoints assume \
         2-D activations):\n{}",
        pinned.stderr
    );
    let (w_plain, w_pinned) = (section(&plain.stdout, "W"), section(&pinned.stdout, "W"));
    assert!(largest_change(&section(&plain.stdout, "W0"), &w_plain) > 1e-3);
    assert!(
        largest_change(&w_plain, &w_pinned) < 1e-5,
        "pinned and unpinned builds trained to different weights"
    );
}
