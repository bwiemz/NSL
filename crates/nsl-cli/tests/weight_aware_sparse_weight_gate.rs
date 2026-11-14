//! A `-w weights.safetensors` run whose model holds a mostly-zero matrix.
//!
//! Such a weight used to be routed to a CSR "sparse matmul" that had never
//! worked: on the CPU it aborted with "invalid tensor handle", and for the
//! GPU it computed `W @ x` where the program says `x @ W`. It was first
//! withheld behind `NSL_M52C_SPARSE_MATMUL=1` and is now removed. This gate
//! pins what a user sees: the product, by hand, with or without that
//! variable in the environment.

use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// A safetensors file holding one f32 tensor: an 8-byte little-endian header
/// length, a JSON header padded to a multiple of 8, then the raw values.
fn write_safetensors(path: &Path, name: &str, shape: &[usize], values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut header = format!(
        "{{\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[0,{}]}}}}",
        shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
        bytes.len()
    )
    .into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&bytes);
    std::fs::write(path, file).unwrap();
}

/// `-w` gives the compiler the checkpoint to analyse; it does not put the
/// file's values into the model. So the model builds the same matrix itself:
/// relu(i - 23) for i = 0..31 is zero in the first three rows (75% zeros)
/// and [1, 2, ..., 8] in the fourth.
const PROGRAM: &str = r#"model M(d: int):
    w: Tensor = relu(arange(0.0, 32.0).reshape([4, 8]) - 23.0)

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = M(4)
let x = arange(1.0, 9.0).reshape([2, 4])
let y: Tensor = m.forward(x)
print(y)
"#;

fn run(env: Option<(&str, &str)>) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("prog.nsl");
    std::fs::write(&source, PROGRAM).unwrap();
    let w: Vec<f32> = (0..32).map(|i| (i as f32 - 23.0).max(0.0)).collect();
    let weights = dir.path().join("w.safetensors");
    write_safetensors(&weights, "w", &[4, 8], &w);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .env_remove("NSL_M52C_SPARSE_MATMUL")
        .current_dir(dir.path())
        .arg("run")
        .arg(&source)
        .arg("--target")
        .arg("cpu")
        .arg("-w")
        .arg(&weights);
    if let Some((k, v)) = env {
        cmd.env(k, v);
    }
    cmd.output().unwrap()
}

fn printed_numbers(out: &std::process::Output) -> Vec<f64> {
    assert!(
        out.status.success(),
        "run failed: {}\n--- stdout\n{}\n--- stderr\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .filter(|tok| tok.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|tok| tok.parse::<f64>().ok())
        .collect()
}

/// x = [[1, 2, 3, 4], [5, 6, 7, 8]] and only the fourth row of w is
/// nonzero, so x @ w is x[:, 3] times [1..8]: 4 * [1..8] and 8 * [1..8].
const WANT: [f64; 16] =
    [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0];

/// The compiler did look at the weight: its report names it and says the
/// matmul is dense. (If the report line goes, this run no longer exercises
/// a weight at or above the sparse threshold.)
#[test]
fn the_weight_is_reported_as_sparse_and_its_matmul_as_dense() {
    let out = run(None);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("weight 'w' is 75.0% sparse"), "{stderr}");
    assert!(stderr.contains("its matmul is dense"), "{stderr}");
    assert!(!stderr.contains("sparse kernel"), "{stderr}");
}

#[test]
fn a_mostly_zero_weight_gives_the_product() {
    assert_eq!(printed_numbers(&run(None)), WANT);
}

/// The variable that used to select the broken path selects nothing now.
#[test]
fn the_removed_opt_in_variable_changes_nothing() {
    let out = run(Some(("NSL_M52C_SPARSE_MATMUL", "1")));
    assert_eq!(printed_numbers(&out), WANT);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("emitting CSR sparse matmul"), "{stderr}");
}
