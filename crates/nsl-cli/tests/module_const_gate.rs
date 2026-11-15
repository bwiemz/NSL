//! A module-level `const` with a literal value is visible in functions,
//! model methods and field initialisers, and is accepted where a
//! `DataLoader` keyword takes a compile-time constant.
//!
//! Functions and model methods are compiled outside `main`, which holds the
//! module's top-level bindings, so `return x * K` in a function failed with
//! "undefined variable 'K'", and `DataLoader(..., batch_size=BATCH)` was
//! refused as not a compile-time constant.

use std::path::PathBuf;
use std::process::{Command, Output};

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn run(source: &str) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .args(["--target", "cpu"])
        .output()
        .unwrap()
}

fn lines(out: &Output) -> Vec<String> {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    String::from_utf8_lossy(&out.stdout).lines().map(str::to_string).collect()
}

/// By hand: `f(2) = 2 * 3 + (-3 // 2) = 6 - 2 = 4` (floor division, as
/// Python); `pi()` is the f32 constant 3.25; `full([3, 3], 2.0) * 0.5` is
/// a 3 x 3 of ones; the strings and booleans interpolate as written.
#[test]
fn consts_are_visible_in_functions_methods_and_field_initialisers() {
    let out = run(r#"const K = 3
const NEG = -K // 2
const SCALE = 0.25 * 2.0
const PI: f32 = 3.25
const ON = true
const NAME = "abc"
fn f(x: int) -> int:
    return x * K + NEG
fn pi() -> f32:
    return PI
model M:
    w: Tensor = full([K, K], 2.0)
    fn g(self) -> Tensor:
        return self.w * SCALE
    fn label(self) -> str:
        return f"{NAME} {ON}"
print(f(2))
print(pi())
let m = M()
print(m.g())
print(m.label())
"#);
    assert_eq!(
        lines(&out),
        ["4", "3.25", "tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])", "abc true"]
    );
}

/// The same loader with constants and with literals: 80 tokens in batches
/// of 2 rows of 4 is 10 batches either way.
#[test]
fn a_const_is_a_dataloader_compile_time_constant() {
    let count = |batch: &str, seq: &str, shuffle: &str| {
        lines(&run(&format!(
            r#"const BATCH = 2
const SEQ = 4
const SHUFFLE = false
let corpus = arange(0.0, 80.0)
let loader = DataLoader(corpus, batch_size={batch}, seq_len={seq}, shuffle={shuffle}, drop_last=true)
let n = 0
for b in loader:
    n = n + 1
print(n)
"#
        )))
    };
    assert_eq!(count("BATCH", "SEQ", "SHUFFLE"), ["10"]);
    assert_eq!(count("2", "4", "false"), ["10"]);
}

/// A `const` whose value is not a compile-time literal stays a variable of
/// the module's top level; reading it from a function is refused, now with
/// the reason (it was "undefined variable 'W'").
#[test]
fn a_const_that_is_not_a_literal_stays_out_of_functions() {
    let out = run(r#"const W = ones([2])
fn f() -> Tensor:
    return W
print(f())
"#);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("'W' is bound at the module's top level"), "{stderr}");
}
