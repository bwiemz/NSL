//! CPU ops that read a strided view in storage order, and two edge cases.
//!
//! * `lt_scalar` and `multinomial` indexed `data[i]` over a view's storage,
//!   so on `x.transpose(0, 1)` they answered for the wrong elements.
//! * The fused RMSNorm dx backward (`--fuse-rmsnorm-backward`) read a strided
//!   upstream gradient the same way on its CPU arm; the sibling dgamma
//!   backward and the GPU arm already materialized it.
//! * `dropout(x, 1.0, true)` returned NaN everywhere (`x * 0 * inf`).

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// Every number the program prints, tensors included.
fn run(tag: &str, flags: &[&str], src: &str) -> Vec<f64> {
    let dir = std::env::temp_dir().join(format!("nsl_cpuedge_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--target", "cpu"])
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
        .to_lowercase()
        .replace("nan", " 1e999 ")
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e')))
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// x = [[1,2,3],[4,5,6]]; x^T < 3.5 is [[1,0],[1,0],[1,0]]. Read in storage
/// order it was [[1,1],[1,0],[0,0]].
#[test]
fn lt_scalar_reads_a_transposed_view_through_its_strides() {
    let got = run(
        "lt",
        &[],
        "let x = arange(1.0, 7.0).reshape([2, 3])\n\
         print(lt_scalar(x.transpose(0, 1), 3.5))\n",
    );
    assert_eq!(got, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

/// P = [[1,0],[0,1],[0,1]], so P^T has rows [1,0,0] and [0,1,1]: row 0 can
/// only draw index 0, and row 1 can never draw it. Read in storage order
/// row 1 was [1,0,1] and drew index 0 half the time.
#[test]
fn multinomial_reads_a_transposed_view_through_its_strides() {
    let draws = run(
        "multinomial_draws",
        &["--seed", "11"],
        "let col0 = lt_scalar(arange(0.0, 3.0), 0.5).reshape([3, 1])\n\
         let col1 = (1.0 - lt_scalar(arange(0.0, 3.0), 0.5)).reshape([3, 1])\n\
         let parts = [col0, col1]\n\
         let p = tensor_cat(parts, 1)\n\
         for i in range(40):\n\
         \x20   print(multinomial(p.transpose(0, 1), 1))\n",
    );
    assert_eq!(draws.len(), 80, "40 draws of 2 rows: {draws:?}");
    for pair in draws.chunks(2) {
        assert_eq!(pair[0], 0.0, "row 0 is [1,0,0]: only index 0 has mass");
        assert_ne!(pair[1], 0.0, "row 1 is [0,1,1]: index 0 has no mass");
    }
}

#[test]
fn dropout_at_p_one_is_zero_not_nan() {
    let got = run("dropout", &[], "let x = full([4], 3.0)\nprint(dropout(x, 1.0, true))\n");
    assert_eq!(got, [0.0, 0.0, 0.0, 0.0]);
}

/// d/dx sum(rmsnorm(x)^T * w) under the fused dx backward.
///
/// The reference is PyTorch autograd on the same program
/// (`x / sqrt(mean(x^2, -1) + eps) * g`, then `(y.T * w).sum()`), written
/// out as literals. On `main` this test compares the fused backward with
/// tape AD instead; on this branch tape AD is itself wrong for a strided
/// upstream gradient (it reads the transposed gradient in storage order,
/// giving [1.70, 4.26, 6.81, 0.77, 0.16, -0.46]). `main` fixed that in
/// "Make CPU tensor ops honour strides in forwards and tape backwards",
/// which this release branch does not carry.
#[test]
fn fused_rmsnorm_backward_handles_a_strided_gradient() {
    let src = "let x = arange(1.0, 7.0).reshape([2, 3]) * 0.5 - 1.2\n\
               let g = ones([3])\n\
               let w = arange(1.0, 7.0).reshape([3, 2])\n\
               let (loss, grads) = grad(x):\n\
               \x20   let y = rmsnorm(x, g, 0.00001)\n\
               \x20   let z = sum(y.transpose(0, 1) * w)\n\
               \x20   z\n\
               print(grads)\n";
    let reference = [
        2.696_324_6,
        6.740_865_7,
        10.785_409,
        -0.387_357_83,
        -0.079_037_905,
        0.229_281_9,
    ];
    let fused = run("rms_fused", &["--source-ad", "--fuse-rmsnorm-backward"], src);
    assert_eq!(fused.len(), 6, "{fused:?}");
    for (f, r) in fused.iter().zip(&reference) {
        assert!((f - r).abs() < 1e-5, "fused {fused:?}\n reference {reference:?}");
    }
}
