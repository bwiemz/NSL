//! The stdlib samplers draw from the distribution they say they draw from,
//! and a seed reproduces a draw.
//!
//! This replaces two programs under `tests/` that no harness ran and that
//! could not have failed if one had: `m19_topk_test.nsl` printed
//! "deterministic RNG: PASS" after two seeded draws it never compared, and
//! `test_sampling.nsl` printed three samples and "PASS".
//!
//! The NSL program below draws many tokens from a four-token distribution
//! with known probabilities and prints the counts; the assertions compare
//! them with the exact probabilities, each count within five standard
//! deviations of its binomial mean (so a correct sampler fails about once
//! in a million runs, and the seed is fixed besides).

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

const DRAWS: usize = 8000;

/// p = [0.5, 0.3, 0.15, 0.05], built by arithmetic; logits = ln p.
const PROGRAM: &str = r#"
from nsl.inference.sampling import sample_top_k, sample_top_p, sample_greedy

let base = arange(0.0, 4.0)
let one0 = lt_scalar(base, 0.5)
let one1 = lt_scalar(base, 1.5) - lt_scalar(base, 0.5)
let one2 = lt_scalar(base, 2.5) - lt_scalar(base, 1.5)
let one3 = full([4], 1.0) - lt_scalar(base, 2.5)
let p = one0 * 0.5 + one1 * 0.3 + one2 * 0.15 + one3 * 0.05
let logits = log(p)
let n = DRAWS

fn count(logits: Tensor, n: int, mode: int):
    let c0 = 0
    let c1 = 0
    let c2 = 0
    let c3 = 0
    for i in range(n):
        let v = 0
        if mode == 0:
            v = int(sample_top_k(logits, 4, 1.0).item())
        if mode == 1:
            v = int(sample_top_k(logits, 2, 1.0).item())
        if mode == 2:
            v = int(sample_top_p(logits, 0.8, 1.0).item())
        if mode == 3:
            v = int(sample_top_k(logits, 4, 0.5).item())
        if v == 0:
            c0 = c0 + 1
        if v == 1:
            c1 = c1 + 1
        if v == 2:
            c2 = c2 + 1
        if v == 3:
            c3 = c3 + 1
    print(c0)
    print(c1)
    print(c2)
    print(c3)

manual_seed(42)
count(logits, n, 0)
count(logits, n, 1)
count(logits, n, 2)
count(logits, n, 3)

# A seed reproduces a draw: sixteen draws, reseed, the same sixteen.
manual_seed(7)
let a = 0
for i in range(16):
    a = a * 4 + int(sample_top_k(logits, 4, 1.0).item())
manual_seed(7)
let b = 0
for i in range(16):
    b = b * 4 + int(sample_top_k(logits, 4, 1.0).item())
manual_seed(8)
let c = 0
for i in range(16):
    c = c * 4 + int(sample_top_k(logits, 4, 1.0).item())
print(a)
print(b)
print(c)
print(int(sample_greedy(logits).item()))
"#;

fn assert_counts(counts: &[usize], probs: &[f64], what: &str) {
    assert_eq!(counts.iter().sum::<usize>(), DRAWS, "{what}: {counts:?}");
    for (tok, (&c, &p)) in counts.iter().zip(probs).enumerate() {
        let mean = DRAWS as f64 * p;
        let sd = (DRAWS as f64 * p * (1.0 - p)).sqrt();
        assert!(
            (c as f64 - mean).abs() <= 5.0 * sd,
            "{what}: token {tok} was drawn {c} times of {DRAWS}; {mean:.0} expected (sd {sd:.1}). All counts: {counts:?}"
        );
    }
}

#[test]
fn stdlib_samplers_draw_the_stated_distributions_and_a_seed_reproduces_them() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, PROGRAM.replace("DRAWS", &DRAWS.to_string())).unwrap();
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
    let values: Vec<i64> = std::str::from_utf8(&out.stdout)
        .unwrap()
        .lines()
        .map(|l| l.trim().parse().unwrap_or_else(|_| panic!("not an integer: {l:?}\nstderr:\n{stderr}")))
        .collect();
    assert_eq!(values.len(), 20, "{values:?}");
    let counts = |i: usize| -> Vec<usize> { values[4 * i..4 * i + 4].iter().map(|&v| v as usize).collect() };

    // top_k = 4 at temperature 1: the distribution itself.
    assert_counts(&counts(0), &[0.5, 0.3, 0.15, 0.05], "top_k=4, T=1");
    // top_k = 2: the two most likely tokens, renormalised.
    assert_counts(&counts(1), &[0.625, 0.375, 0.0, 0.0], "top_k=2, T=1");
    // top_p = 0.8: tokens 0 and 1 reach 0.8 exactly, so the nucleus is
    // those two and no more.
    assert_counts(&counts(2), &[0.625, 0.375, 0.0, 0.0], "top_p=0.8, T=1");
    // Temperature 0.5 squares the probabilities: [25, 9, 2.25, 0.25] / 36.5.
    assert_counts(
        &counts(3),
        &[25.0 / 36.5, 9.0 / 36.5, 2.25 / 36.5, 0.25 / 36.5],
        "top_k=4, T=0.5",
    );

    // Sixteen draws packed two bits each: the same seed gives the same
    // sixteen, another seed gives others, and they are not all one token.
    let (a, b, c) = (values[16], values[17], values[18]);
    assert_eq!(a, b, "manual_seed(7) twice");
    assert_ne!(a, c, "manual_seed(8) draws something else");
    let digits: std::collections::BTreeSet<i64> = (0..16).map(|i| (a >> (2 * i)) & 3).collect();
    assert!(digits.len() > 1, "sixteen draws were all token {digits:?}");
    assert_eq!(values[19], 0, "greedy picks the most likely token");
}
