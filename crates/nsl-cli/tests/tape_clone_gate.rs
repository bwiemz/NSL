//! `x.clone()` must be differentiable.
//!
//! The runtime's clone records nothing on the autodiff tape, and its result
//! carries a fresh tape id, so the recorded graph stopped at every clone
//! the program wrote: `let c = w.clone()` followed by a loss on `c` gave
//! `w` a gradient of exactly zero, with the loss itself correct. Source
//! `.clone()` now lowers to a clone the tape records as an identity.

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

fn printed_numbers(source: &str) -> Vec<f64> {
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
    String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

/// `d/dw sum(c * c)` with `c = w.clone()` and `w = 1.5` is `2w = 3`, in the
/// block's own body and inside a function it calls.
#[test]
fn the_gradient_flows_through_a_clone() {
    let source = r#"
let w = full([2, 2], 1.5)
let (l, g) = grad(w):
    let c = w.clone()
    sum(c * c)
print(g)
fn f(z: Tensor) -> Tensor:
    let c = z.clone()
    return sum(c * c)
let (l2, g2) = grad(w):
    f(w)
print(g2)
"#;
    let got = printed_numbers(source);
    assert_eq!(got.len(), 8, "expected two 2x2 gradients: {got:?}");
    for (i, g) in got.iter().enumerate() {
        assert!((g - 3.0).abs() < 1e-6, "element {i} is {g}, expected 3.0 ({got:?})");
    }
}

/// A clone is still a copy: writing to it must not reach the original.
#[test]
fn a_clone_is_still_an_independent_copy() {
    let source = r#"
let w = full([2, 2], 1.5)
let c = w.clone()
c[0, 0] = 99.0
print(w.sum())
print(c.sum())
"#;
    assert_eq!(printed_numbers(source), vec![6.0, 103.5]);
}
