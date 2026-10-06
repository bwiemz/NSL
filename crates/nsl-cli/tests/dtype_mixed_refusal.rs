//! C5 step 4 end to end: an f64 program runs in f64 under both AD modes,
//! and a program that mixes f32 with f64 is refused instead of computing in
//! whichever dtype the runtime used to pick ("f32 wins").
//!
//! The runtime half of the refusal is gated op by op in
//! `nsl-runtime/tests/mixed_dtype_refusal.rs`; this file pins that compiled
//! programs reach it -- and that nothing the compiler emits on its own (a
//! constant, a creation, a gradient seed) mixes dtypes in a program whose
//! tensors are all f64.

use std::path::Path;
use std::process::{Command, Output};

fn run(src: &str, flags: &[&str]) -> Output {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("dtype_mixed.nsl");
    std::fs::write(&path, src).unwrap();
    Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .args(flags)
        .arg(&path)
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run")
}

fn text(out: &Output) -> (String, String) {
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The numbers printed between `BEGIN_<tag>` and `END_<tag>`.
fn numbers(stdout: &str, tag: &str) -> Vec<f64> {
    let start = format!("BEGIN_{tag}\n");
    let after = stdout.split_once(&start).unwrap_or_else(|| panic!("no {tag} in:\n{stdout}")).1;
    let body = after.split_once(&format!("\nEND_{tag}")).unwrap().0;
    body.trim()
        .trim_start_matches("tensor(")
        .trim_end_matches(')')
        .split(|c: char| c == ',' || c == '[' || c == ']')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.parse().unwrap_or_else(|e| panic!("{t:?}: {e} in {body}")))
        .collect()
}

/// matmul, relu, scalar arithmetic, mean and the gradient through all of
/// them, on f64 tensors. h = relu(x @ w) * 2 + 1 = 1.6 per column, and
/// dL/dx_j = sum_k (2 h_k / 3) * 2 * w_jk = 4h = 6.4. In f64 the answer is
/// within 1e-12 of 6.4; computed in f32 anywhere along the way, it would be
/// off by about 1e-7. Under `--source-ad` the block goes to the tape (source
/// AD's constants are f32), and that hand-off is asserted, so the run cannot
/// pass by some third path.
#[test]
fn an_f64_program_trains_in_f64_under_both_ad_flags() {
    let src = r#"
let x: Tensor<[1, 3], f64> = full([1, 3], 0.1)
let w: Tensor<[3, 3], f64> = ones([3, 3])
let (l, g) = grad(x):
    let h = relu(x @ w) * 2.0 + 1.0
    mean(h * h)
print("BEGIN_g")
print(g)
print("END_g")
"#;
    for flags in [&[][..], &["--source-ad"][..]] {
        let out = run(src, flags);
        let (stdout, stderr) = text(&out);
        assert!(out.status.success(), "flags {flags:?}: the program failed:\n{stdout}\n{stderr}");
        if flags.contains(&"--source-ad") {
            assert!(
                stderr.contains("this grad block computes on f64 tensors")
                    && stderr.contains("falling back to tape-based AD"),
                "source AD must hand the f64 block to the tape:\n{stderr}"
            );
        }
        let g = numbers(&stdout, "g");
        assert_eq!(g.len(), 3, "flags {flags:?}: {stdout}");
        for v in g {
            assert!((v - 6.4).abs() < 1e-12, "flags {flags:?}: dL/dx = {v}, not f64-exact 6.4");
        }
    }
}

/// `a + b` with `a` f32 and `b` f64 used to print an f32 result. Now it
/// stops with the unsupported-dtype exit code and names the conversion.
#[test]
fn a_mixed_f32_f64_program_is_refused() {
    let src = r#"
let a = full([3], 1.0)
let b: Tensor<[3], f64> = full([3], 2.0)
print(a + b)
"#;
    let out = run(src, &[]);
    let (stdout, stderr) = text(&out);
    assert_eq!(
        out.status.code(),
        Some(nsl_runtime::fatal::NSL_EXIT_UNSUPPORTED_DTYPE),
        "a mixed-dtype add must be refused:\n{stdout}\n{stderr}"
    );
    assert!(
        stderr.contains("nsl_tensor_add: operands have different dtypes, f32 and f64"),
        "{stderr}"
    );
    assert!(stderr.contains("`.to(f64)`"), "{stderr}");
}
