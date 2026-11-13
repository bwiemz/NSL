//! `--deterministic` must not cut the tape at a GPU `sum` / `mean` over a dim.
//!
//! Under `--deterministic` codegen swaps `sum(x, dim, keepdim)` and
//! `mean(x, dim, keepdim)` for the sequential kernels in
//! `deterministic_ops.rs`. Their GPU arms returned without a tape record, so
//! tape AD saw no path from the result to the input: the loss was right and
//! the gradient was all zeros, with no diagnostic.
//!
//! Run: cargo test -p nsl-cli --features cuda --test deterministic_gpu_reduction_tape_gate
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn numbers(tag: &str, flags: &[&str]) -> Vec<f32> {
    let dir = std::env::temp_dir().join(format!("nsl_detred_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(
        &prog,
        "let x = arange(1.0, 7.0).reshape([2, 3]).to(cuda)\n\
         let w = arange(1.0, 4.0).to(cuda)\n\
         let (loss, grads) = grad(x):\n\
         \x20   let s = sum(x * x, 0, 0)\n\
         \x20   let m = mean(x, 0, 0)\n\
         \x20   let z = sum((s + m) * w)\n\
         \x20   z\n\
         print(loss.to(cpu))\n\
         print(grads.to(cpu))\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--tape-ad"])
        .args(flags)
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "[{tag}] failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    stdout
        .lines()
        .filter(|l| l.starts_with("tensor("))
        .flat_map(|l| l.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e'))))
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// loss = sum_c w_c * (sum_r x_rc^2 + mean_r x_rc) = 233, and
/// d/dx_rc = w_c * (2 x_rc + 1/2).
#[test]
#[ignore = "requires CUDA GPU"]
fn deterministic_mode_keeps_the_gradient_of_a_dim_reduction() {
    let want = [233.0, 2.5, 9.0, 19.5, 8.5, 21.0, 37.5];
    assert_eq!(numbers("plain", &[]), want, "the non-deterministic kernels");
    assert_eq!(numbers("det", &["--deterministic"]), want, "--deterministic");
}
