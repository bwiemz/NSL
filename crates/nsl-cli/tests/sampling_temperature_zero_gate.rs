//! `sample_top_k` / `sample_top_p` at `temperature = 0`.
//!
//! Both divide the logits by the temperature, so zero gave infinities, the
//! softmax gave NaN, and the multinomial draw aborted the process ("sum of
//! probabilities must be > 0"). Temperature zero is the greedy limit and is
//! what callers pass to ask for it.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn temperature_zero_is_greedy() {
    let source = r#"
from nsl.inference.sampling import sample_top_k, sample_top_p
manual_seed(3)
let logits = tensor_cat([arange(0.0, 5.0), full([1], 9.0), arange(0.0, 3.0)], 0)
print(sample_top_k(logits, 3, 0.0))
print(sample_top_p(logits, 0.9, 0.0))
print(sample_top_k(logits, 1, 1.0))
"#;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .arg("--target")
        .arg("cpu")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    // The largest logit is at index 5. The third line is `k = 1` at
    // temperature 1, which is greedy too: the first two must print exactly
    // what it prints, i.e. have the shape a sampled draw has.
    let lines: Vec<&str> = std::str::from_utf8(&out.stdout).unwrap().lines().collect();
    assert_eq!(lines.len(), 3, "stdout: {lines:?}\nstderr:\n{stderr}");
    assert!(lines[2].contains("5.0"), "greedy reference printed {:?}", lines[2]);
    assert_eq!(lines[0], lines[2], "sample_top_k at temperature 0");
    assert_eq!(lines[1], lines[2], "sample_top_p at temperature 0");
}
