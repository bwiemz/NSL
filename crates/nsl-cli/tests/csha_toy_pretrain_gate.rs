//! `examples/csha_toy_pretrain_hd32.nsl`: refused with `--csha`, and right
//! without it.
//!
//! The example is an `@flash_attention` method whose RMSNorm and three
//! projections feed `scaled_dot_product_attention` directly, trained for one
//! SGD step: the chain the CSHA boundary scan claims.
//!
//! **With `--csha`** the train step would go through the CSHA fused forward
//! and backward. That pair is wrong beyond one tile (dQ, dK and dV 43% to
//! 130% off the f64 reference at two), stops on a shape mismatch beyond one
//! head, and has not been shown right at one head and one tile, so the
//! compiler refuses the program. The first test checks the refusal. It
//! compiles only, and needs no GPU.
//!
//! **Without `--csha`** the train step differentiates the unfused attention.
//! The second test runs that on a GPU and compares everything the example
//! prints with literals from PyTorch's float64 autograd
//! (`tools/refcheck/csha_toy_literals.py`): the loss of the train step, and
//! for each of Wq, Wk, Wv and the RMSNorm weight two statistics of the
//! update `d` the step applied, `sum(d * d)` and `sum(d * probe)`. Every
//! statistic is asserted. The run uses `--matmul-mode f32`: the true Wq
//! gradient of this program is a small difference of large terms, and the
//! gate is about the attention's gradient, not about TF32 operand rounding.
//!
//! The example does not call its method outside the train block. Such a
//! call runs the decorated kernel, whose 64-row query tile does not divide
//! the example's sequence of 32; the kernel used to write the missing rows
//! past its output and overwrote the program's copy of Wq, so the Wq update
//! read 17.2 against 0.0061. That launch is now refused
//! (`flash_attention_ragged_seq_gate.rs`).
//!
//! The program this example replaced could not run (its `x` was
//! two-dimensional) and could not tell a working backward from a broken one
//! (all-ones inputs and a loss of `sum(out)` make dQ, dK, dWq and dWk
//! exactly zero); its only test looked at relocation symbols.
//!
//! ```text
//! cargo test -p nsl-cli --test csha_toy_pretrain_gate
//! cargo test -p nsl-cli --release --features cuda \
//!     --test csha_toy_pretrain_gate -- --ignored
//! ```

use std::path::PathBuf;
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nsl_csha_toy_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run `nsl <args>` on a copy of the example placed in `dir`, so that
/// nothing the compiler writes beside its input lands in the repository.
fn nsl(dir: &std::path::Path, args: &[&str]) -> Output {
    let root = repo_root();
    let prog = dir.join("csha_toy_pretrain_hd32.nsl");
    std::fs::copy(root.join("examples/csha_toy_pretrain_hd32.nsl"), &prog).expect("copy the example");
    Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(args)
        .arg(&prog)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl")
}

#[test]
fn training_the_example_with_csha_is_a_compile_error() {
    let dir = scratch_dir("refusal");
    let out = nsl(&dir, &["build", "--emit-obj", "--target", "cuda", "--source-ad", "--csha", "auto"]);
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!out.status.success(), "the example compiled with --csha:\n{stderr}");
    for needle in ["@flash_attention", "--csha", "withheld", "do not match the reference"] {
        assert!(stderr.contains(needle), "the error does not say {needle:?}:\n{stderr}");
    }
}

/// The same compile without `--csha` succeeds: the refusal is about the
/// fused pair, not about the program.
#[test]
fn the_example_compiles_without_csha() {
    let dir = scratch_dir("plain");
    let out = nsl(&dir, &["build", "--emit-obj", "--target", "cuda", "--source-ad"]);
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(out.status.success(), "the example does not compile without --csha:\n{stderr}");
    assert!(!stderr.contains("withheld"), "the refusal fired without --csha:\n{stderr}");
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

/// Allowed error as a fraction of each statistic's scale. Everything on
/// this path is f32; measured on an RTX PRO 4500 the worst statistic is
/// 2e-6 of its scale.
const REL_TOL: f64 = 1e-4;

#[test]
#[ignore = "requires CUDA GPU"]
fn the_example_without_csha_matches_pytorch() {
    let dir = scratch_dir("run");
    let out = nsl(&dir, &["run", "--target", "cuda", "--source-ad", "--matmul-mode", "f32"]);
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        out.status.success() && stdout.contains("csha_toy_pretrain_hd32: PASS"),
        "the example failed ({}):\n{stdout}\n{stderr}",
        out.status
    );
    // A scalar prints as `tensor([<value>])`.
    let got: Vec<f64> = stdout
        .lines()
        .filter_map(|l| l.trim().strip_prefix("tensor([")?.strip_suffix("])")?.parse::<f64>().ok())
        .collect();
    assert_eq!(got.len(), EXPECTED.len(), "expected {} printed numbers, got {got:?}", EXPECTED.len());
    let mut failures = Vec::new();
    for (&g, &(name, want, scale)) in got.iter().zip(EXPECTED) {
        let err = (g - want).abs() / scale;
        eprintln!("{name:<22} nsl={g:<13.6e} pytorch={want:<13.6e} err/scale={err:.2e}");
        if !(err < REL_TOL) {
            failures.push(format!("{name}: {g:e} vs PyTorch {want:e} ({err:.3e} of its scale, tolerance {REL_TOL:.0e})"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
