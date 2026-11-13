//! CPU `sum` / `mean` over f32 data must not accumulate in f32.
//!
//! A running f32 sum stops absorbing unit-sized terms at 2^24, so
//! `sum(ones([20000000]))` returned 16777216 and its mean 0.8388608. On
//! ordinary data the same accumulation is percent-level wrong well before
//! that: `sum(w * w)` over a `[49152, 512]` embedding at std 0.02 read 5.7%
//! low, and the mean of four million values near 10.8 read 11.0. The global
//! arms and the per-dimension arm (a long reduced dimension is the same
//! running sum) accumulate in f64 and narrow once.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
fn f32_sums_past_two_to_the_24_are_exact() {
    let dir = std::env::temp_dir().join(format!("nsl_sumprec_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(
        &prog,
        "let x = ones([20000000])\n\
         print(sum(x).item())\n\
         print(mean(x).item())\n\
         let y = ones([2, 16777300])\n\
         let rows = sum(y, 1, 0)\n\
         print(sum(rows).item())\n\
         print(\"SUM_PRECISION_DONE\")\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--target", "cpu"])
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("SUM_PRECISION_DONE"),
        "the program failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let values: Vec<f64> = stdout.lines().filter_map(|l| l.trim().parse().ok()).collect();
    assert_eq!(values.len(), 3, "three numbers expected:\n{stdout}");
    assert_eq!(values[0], 20_000_000.0, "sum(ones([2e7])) — 16777216 is the f32 running sum");
    assert_eq!(values[1], 1.0, "mean(ones([2e7])) — 0.8388608 is the f32 running sum");
    // Each row sums to 16777300, which f32 holds exactly (it is a multiple of 4);
    // the f32 running sum stalls at 16777216.
    assert_eq!(values[2], 2.0 * 16_777_300.0, "sum over a 16777300-long dimension");
}
