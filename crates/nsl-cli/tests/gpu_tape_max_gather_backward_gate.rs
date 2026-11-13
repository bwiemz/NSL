//! Tape-AD gradients of `reduce_max` and dim-0 `gather` on GPU.
//!
//! * The GPU `reduce_max` arm recorded `saved_argmax: vec![0; n]` — the
//!   kernel returns maxima, not positions — so the whole gradient went to
//!   index 0 of the reduced dim whatever the data.
//! * The GPU dim-0 `gather` is an index-select with one output row per
//!   index; its backward bounded the scatter by the number of OUTER rows
//!   (one, along dim 0), so only the first gradient row was scattered.
//!
//! Run: cargo test -p nsl-cli --features cuda --test gpu_tape_max_gather_backward_gate
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// The numbers on the LAST `tensor(...)` line the program prints.
fn last_tensor(tag: &str, src: &str) -> Vec<f32> {
    let dir = std::env::temp_dir().join(format!("nsl_maxgather_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--tape-ad"])
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .filter(|l| l.starts_with("tensor("))
        .next_back()
        .unwrap_or_else(|| panic!("[{tag}] nothing printed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr)));
    line.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e'))).filter_map(|t| t.parse().ok()).collect()
}

/// x = [[-5.5, -9, -10.5], [-10, -7.5, -3]]: the row maxima are at columns 0
/// and 2, so d/dx sum(max(x, dim 1) * [10, 20]) is [[10,0,0],[0,0,20]]. The
/// GPU returned [[10,0,0],[20,0,0]].
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_reduce_max_gradient_goes_to_the_argmax() {
    let program = |device: &str| {
        format!(
            "let b = arange(1.0, 7.0).reshape([2, 3])\n\
             let x = (b * b - b * 6.5).to({device})\n\
             let w = (arange(1.0, 3.0) * 10.0).to({device})\n\
             let (loss, grads) = grad(x):\n\
             \x20   let m = reduce_max(x, 1, 0)\n\
             \x20   let z = sum(m * w)\n\
             \x20   z\n\
             print(grads.to(cpu))\n"
        )
    };
    let want = [10.0, 0.0, 0.0, 0.0, 0.0, 20.0];
    assert_eq!(last_tensor("max_cpu", &program("cpu")), want, "the CPU reference");
    assert_eq!(last_tensor("max_gpu", &program("cuda")), want, "GPU");
}

/// idx = [2, 0, 2, 1] into a 4x3 x, upstream rows w0..w3 = [1,2,3] ..
/// [10,11,12]: dx[0] = w1, dx[1] = w3, dx[2] = w0 + w2, dx[3] = 0. The GPU
/// returned w0 in row 2 and zeros elsewhere.
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_gather_dim0_gradient_scatters_every_index() {
    let got = last_tensor(
        "gather_gpu",
        "let x = arange(1.0, 13.0).reshape([4, 3]).to(cuda)\n\
         let a = arange(0.0, 4.0)\n\
         let idx = (lt_scalar(a, 0.5) * 2.0 - lt_scalar(a, 1.5) * 2.0 + lt_scalar(a, 2.5) + 1.0).to(cuda)\n\
         let w = arange(1.0, 13.0).reshape([4, 3]).to(cuda)\n\
         let (loss, grads) = grad(x):\n\
         \x20   let g = gather(x, 0, idx)\n\
         \x20   let z = sum(g * w)\n\
         \x20   z\n\
         print(grads.to(cpu))\n",
    );
    assert_eq!(got, [4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 8.0, 10.0, 12.0, 0.0, 0.0, 0.0]);
}
