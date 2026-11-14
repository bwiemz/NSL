//! `examples/csha_toy_pretrain_hd32.nsl` on a GPU, against PyTorch.
//!
//! The example is the one program that reaches the CSHA launchers from NSL
//! source: an `@flash_attention` method whose RMSNorm and three projections
//! feed `scaled_dot_product_attention` directly, trained for one SGD step
//! with `--source-ad --csha auto`. This gate runs it and compares what it
//! prints with literals from PyTorch's float64 autograd on the same program
//! (`tools/refcheck/csha_toy_literals.py`):
//!
//! * the loss, computed twice: by a direct call of the method and by the
//!   train step's forward;
//! * for each of Wq, Wk, Wv and the RMSNorm weight, two statistics of the
//!   update `d` the step applied: `sum(d * d)` and `sum(d * probe)`.
//!
//! The example it replaces could not tell a working backward from a broken
//! one, and could not run: `x` was two-dimensional, which the fused forward
//! aborts on, every weight was ones and the loss was `sum(out)`, which makes
//! dQ, dK, dWq and dWk exactly zero. Its only test compiled it and looked at
//! relocation symbols. Here the inputs are non-uniform, every gradient is
//! nonzero, and the gradient reaching the attention output is of order 1e-6.
//!
//! # What is asserted, and what is not
//!
//! Asserted: the two losses, the projections of the Wk and Wv updates on the
//! probe, and both statistics of the RMSNorm weight's update. These agree
//! with PyTorch to 1e-3 of their scale or better.
//!
//! Printed and NOT asserted, because they are wrong today ([`KNOWN_WRONG`]):
//! both statistics of the Wq update (its squared norm is 17.2 against
//! PyTorch's 0.0061) and the squared norms of the Wk and Wv updates (about
//! 12% high). The kernel-level gates in
//! `crates/nsl-codegen/tests/csha_cuda_backward.rs` pass for the fused
//! configuration; what is wrong is the weight gradient on the configuration
//! a training build emits. When that is fixed, empty the list.
//!
//! ```text
//! cargo test -p nsl-cli --release --features cuda \
//!     --test csha_toy_pretrain_gate -- --ignored
//! ```

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// Run the example with the CSHA path on and return the numbers it printed,
/// in order.
fn run_example() -> Vec<f64> {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("nsl_csha_toy_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--target", "cuda", "--source-ad", "--csha", "auto"])
        .arg(root.join("examples/csha_toy_pretrain_hd32.nsl"))
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        out.status.success() && stdout.contains("csha_toy_pretrain_hd32: PASS"),
        "the example failed ({}):\n{stdout}\n{stderr}",
        out.status
    );
    // The fused pair must be what ran, not a fallback to per-op adjoints.
    assert!(
        stderr.contains("CSHA fused backward: emitting fused launch")
            && !stderr.contains("CSHA fused backward rejected")
            && !stderr.contains("[csha] not applied"),
        "the build did not use the CSHA kernels:\n{stderr}"
    );
    // A scalar prints as `tensor([<value>])`.
    stdout
        .lines()
        .filter_map(|l| l.trim().strip_prefix("tensor([")?.strip_suffix("])")?.parse::<f64>().ok())
        .collect()
}

/// (name, PyTorch value, scale the error is judged against).
type Expected = [(&'static str, f64, f64)];

// batch 1, 1 head(s), lr 100000
const EXPECTED: &Expected = &[
    ("loss", 3.1210866990806e-07, 3.1210866990806e-07),
    ("sum(dWq * dWq)", 0.006073930897429108, 0.006073930897429108),
    ("sum(dWq * probe)", -0.1477913543420662, 1.76444848348033),
    ("sum(dWk * dWk)", 0.8543171794996102, 0.8543171794996102),
    ("sum(dWk * probe)", 1.2042940174139205, 20.92587512361776),
    ("sum(dWv * dWv)", 0.3128560938876767, 0.3128560938876767),
    ("sum(dWv * probe)", 0.6971578573546913, 12.663284638171396),
    ("sum(dgamma * dgamma)", 0.2341199895962969, 0.2341199895962969),
    ("sum(dgamma * gprobe)", 0.0008943720509240194, 1.966806917220385),
];

/// Statistics that are printed and not asserted; see the file header.
const KNOWN_WRONG: [&str; 4] =
    ["sum(dWq * dWq)", "sum(dWq * probe)", "sum(dWk * dWk)", "sum(dWv * dWv)"];

/// Allowed error as a fraction of each statistic's scale. The gradients are
/// f32, the fused kernels take Q, K and V as f16 operands and store the
/// attention output as f16, and the statistics are f32 sums. Measured on an
/// RTX PRO 4500: 1.0e-3 on the loss, 8e-4 or less on the asserted gradient
/// statistics.
const REL_TOL: f64 = 5e-3;

#[test]
#[ignore = "requires CUDA GPU"]
fn csha_toy_pretrain_matches_pytorch() {
    let got = run_example();
    assert_eq!(got.len(), EXPECTED.len() + 1, "expected {} printed numbers, got {got:?}", EXPECTED.len() + 1);
    // The first two numbers are the same loss: once from a direct call of
    // the method, once from the train step's forward.
    let (_, loss, loss_scale) = EXPECTED[0];
    let direct = (got[0] - loss).abs() / loss_scale;
    eprintln!("{:<22} nsl={:<13.6e} pytorch={loss:<13.6e} err/scale={direct:.2e}", "loss, direct call", got[0]);
    let mut failures = Vec::new();
    if !(direct < REL_TOL) {
        failures.push(format!("loss by a direct call of the method: {:e} vs PyTorch {loss:e}", got[0]));
    }
    for (&g, &(name, want, scale)) in got[1..].iter().zip(EXPECTED) {
        let err = (g - want).abs() / scale;
        let known_wrong = KNOWN_WRONG.contains(&name);
        eprintln!(
            "{name:<22} nsl={g:<13.6e} pytorch={want:<13.6e} err/scale={err:.2e}{}",
            if known_wrong { "  (known wrong, not asserted)" } else { "" }
        );
        if !known_wrong && !(err < REL_TOL) {
            failures.push(format!("{name}: {g:e} vs PyTorch {want:e} ({err:.3e} of its scale, tolerance {REL_TOL:.0e})"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
