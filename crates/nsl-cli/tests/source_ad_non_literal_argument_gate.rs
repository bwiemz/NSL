//! Source AD must not guess an argument it cannot read at compile time.
//!
//! Three rules took a non-literal argument and computed something else,
//! silently:
//!
//! * `.transpose(i, j)` with dims held in variables was extracted as
//!   `Passthrough("transpose")`, which has no AD rule: the forward was right
//!   and the gradient through it was ZERO.
//! * `softmax(x, d)` / `log_softmax(x, d)` with a non-literal `d` used
//!   dim = -1 for both the forward and the adjoint — self-consistent, and the
//!   wrong function.
//! * `abs(x)` differentiated to -1 at exactly 0 (the tape, and convention,
//!   say 0), so an L1 penalty moved every zero parameter by lr * lambda.
//!
//! The first two now decline the extraction, which is the documented
//! contract (`source AD extraction failed ... falling back to tape-based
//! AD`): the tape reads the argument at run time.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// The numbers the program prints, under `--source-ad` and under `--tape-ad`.
fn both_modes(tag: &str, src: &str) -> [Vec<f64>; 2] {
    ["--source-ad", "--tape-ad"].map(|flag| {
        let dir = std::env::temp_dir().join(format!("nsl_sadarg_{}_{tag}_{}", std::process::id(), &flag[2..]));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let prog = dir.join("p.nsl");
        std::fs::write(&prog, src).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
            .args(["run", "--target", "cpu", flag])
            .arg(&prog)
            .current_dir(&dir)
            .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
            .output()
            .expect("spawn nsl run");
        let _ = std::fs::remove_dir_all(&dir);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "[{tag} {flag}] failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
        // Every number printed, tensors included (`tensor([9.0, 12.0])`).
        stdout
            .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e')))
            .filter_map(|t| t.parse().ok())
            .collect()
    })
}

fn assert_close(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: {got:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-4, "{what}: got {got:?}, expected {want:?}");
    }
}

/// d/dx sum(x^T * r) = r^T; with r = [[1,2],[3,4],[5,6]] its first row sums
/// to 9 and its second to 12. Source AD returned zeros.
#[test]
fn transpose_with_variable_dims_has_a_gradient() {
    let [source, tape] = both_modes(
        "transpose",
        "let x = arange(1.0, 7.0).reshape([2, 3])\n\
         let r = arange(1.0, 7.0).reshape([3, 2])\n\
         let i = 0\n\
         let j = 1\n\
         let (loss, grads) = grad(x):\n\
         \x20   let z = sum(x.transpose(i, j) * r)\n\
         \x20   z\n\
         print(loss.item())\n\
         print(sum(grads, 1, 0))\n",
    );
    assert_close("tape", &tape, &[86.0, 9.0, 12.0]);
    assert_close("source AD", &source, &[86.0, 9.0, 12.0]);
}

/// Softmax over dim 0 of [[0,1,2],[3,4,5]] is [sigma(-3), sigma(3)] in every
/// column, so sum(softmax * [[1,2,3],[4,5,6]]) = 14.5732. Over the last dim
/// it is 8.1504, which is what source AD printed.
#[test]
fn softmax_with_a_variable_dim_uses_that_dim() {
    for (name, call) in [("softmax", "softmax(x, d)"), ("log_softmax", "exp(log_softmax(x, d))")] {
        let [source, tape] = both_modes(
            name,
            &format!(
                "let x = arange(0.0, 6.0).reshape([2, 3])\n\
                 let r = arange(1.0, 7.0).reshape([2, 3])\n\
                 let d = 0\n\
                 let (loss, grads) = grad(x):\n\
                 \x20   let z = sum({call} * r)\n\
                 \x20   z\n\
                 print(loss.item())\n"
            ),
        );
        assert_close(&format!("{name}, tape"), &tape, &[14.573_167]);
        assert_close(&format!("{name}, source AD"), &source, &[14.573_167]);
    }
}

#[test]
fn abs_has_zero_gradient_at_zero() {
    let [source, tape] = both_modes(
        "abs",
        "let x = zeros([3])\n\
         let (loss, grads) = grad(x):\n\
         \x20   let z = sum(abs(x))\n\
         \x20   z\n\
         print(grads)\n\
         let y = arange(-1.0, 2.0)\n\
         let (loss2, grads2) = grad(y):\n\
         \x20   let z2 = sum(abs(y))\n\
         \x20   z2\n\
         print(grads2)\n",
    );
    assert_close("tape", &tape, &[0.0, 0.0, 0.0, -1.0, 0.0, 1.0]);
    assert_close("source AD", &source, &[0.0, 0.0, 0.0, -1.0, 0.0, 1.0]);
}
