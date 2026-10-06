//! A norm's epsilon set on the model is the one source AD trains with.
//!
//! The stdlib `LayerNorm` / `RMSNorm` pass their `eps` field to the kernel.
//! Source AD could not read a float field at compile time and baked 1e-5
//! instead, so a program that set `m.ln.eps = 0.5` trained with 0.5 on the
//! tape and with 1e-5 under `--source-ad` -- silently, since 1e-5 is also the
//! field's default (external review 2026-10-06). Source AD now reads the field
//! at run time (`NormEps::Var`).
//!
//! The program trains with SGD (no optimizer scale invariance to hide a
//! difference) in both AD modes -- and with source AD's fused RMSNorm
//! backward, whose residual fold the `h + rn(h)` shape triggers -- and all
//! must agree. A control run that leaves the eps at its default must end
//! somewhere else, so the reassignment is shown to matter at this size.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

const PROGRAM: &str = r#"
from nsl.nn.norms import LayerNorm, RMSNorm
from nsl.nn.losses import mse_loss

model Net:
    w: Tensor = (arange(16).reshape([4, 4])) * 0.05 - full([4, 4], 0.3)
    ln: LayerNorm = LayerNorm(4)
    rn: RMSNorm = RMSNorm(4)

    fn forward(self, x: Tensor) -> Tensor:
        let h = self.ln.forward(x @ self.w)
        return h + self.rn.forward(h)

let m = Net()
SET_EPS
let x = (arange(12).reshape([3, 4])) * 0.1 - full([3, 4], 0.5)
let y = (arange(12).reshape([3, 4])) * 0.2 - full([3, 4], 1.0)

train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.1)
    step(batch):
        let loss = mse_loss(m.forward(x), y)

print("W_BEGIN")
print(m.w)
print("W_END")
print("G_BEGIN")
print(m.ln.weight)
print(m.rn.weight)
print("G_END")
"#;

const SET_EPS: &str = "m.ln.eps = 0.5\nm.rn.eps = 0.25";

/// Run the program; returns (trained values, stderr).
fn run(set_eps: bool, source_ad: bool) -> (Vec<f64>, String) {
    run_with(set_eps, source_ad, &[])
}

fn run_with(set_eps: bool, source_ad: bool, flags: &[&str]) -> (Vec<f64>, String) {
    let root = repo_root();
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("norm_field_eps.nsl");
    std::fs::write(&path, PROGRAM.replace("SET_EPS", if set_eps { SET_EPS } else { "" })).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run");
    if source_ad {
        cmd.arg("--source-ad");
    }
    let out = cmd
        .args(flags)
        .arg(&path)
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "set_eps={set_eps} source_ad={source_ad} failed:\n{stderr}");
    let mut vals = Vec::new();
    for (b, e) in [("W_BEGIN", "W_END"), ("G_BEGIN", "G_END")] {
        let after = stdout.split_once(b).map(|(_, r)| r).unwrap_or("");
        let inner = after.split_once(e).map(|(l, _)| l).unwrap_or("");
        vals.extend(
            inner
                .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
                .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
                .filter_map(|t| t.parse::<f64>().ok()),
        );
    }
    assert_eq!(vals.len(), 24, "16 weights + 2x4 norm gains expected:\n{stdout}");
    (vals, stderr)
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, |m, d| if d.is_nan() || d > m { d } else { m })
}

#[test]
fn source_ad_trains_with_the_eps_the_model_was_given() {
    let (tape, _) = run(true, false);
    let (source, stderr) = run(true, true);
    assert!(
        stderr.contains("Using source-to-source AD") && !stderr.contains("falling back to tape-based AD"),
        "the source-AD run must engage source AD:\n{stderr}"
    );
    let d = max_diff(&tape, &source);
    assert!(d < 1e-5, "source AD and the tape disagree by {d:.3e}: source AD is not using the set eps");

    // The fused RMSNorm backward (dx and dgamma kernels) carries the eps as an
    // extra input (`:var`), and the residual `h + rn(h)` folds the dx into its
    // accumulate (`rmsnorm_dx_backward_add:var`).
    let (fused, stderr) = run_with(true, true, &["--fuse-rmsnorm-backward"]);
    assert!(!stderr.contains("falling back to tape-based AD"), "{stderr}");
    let d = max_diff(&tape, &fused);
    assert!(d < 1e-5, "the fused RMSNorm backward disagrees with the tape by {d:.3e}");

    // The control: the default eps trains somewhere else.
    let (control, _) = run(false, false);
    let moved = max_diff(&tape, &control);
    assert!(moved > 1e-3, "eps 0.5/0.25 vs the default moved the weights only {moved:.3e}; the check would be vacuous");
}
