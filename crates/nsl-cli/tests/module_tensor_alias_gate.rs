//! `let t = t0` gives the tensor a second owner.
//!
//! A `let` that names an existing tensor variable used to copy the handle
//! without taking a reference, so two owning names shared one reference.
//! Program teardown frees a tensor once per owning name: a module-level
//! alias of a tensor that a `train` or `distill` step had used ended the run
//! with `free(): double free detected in tcache 2`. Because the tensor
//! looked uniquely owned, an in-place update through one name could also
//! overwrite what the other name read.

use std::path::PathBuf;
use std::process::{Command, Output};

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn run(source: &str, extra_args: &[&str]) -> Output {
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
        .args(extra_args)
        .output()
        .unwrap()
}

fn numbers(out: &Output) -> Vec<f64> {
    String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

fn assert_clean_exit(out: &Output, what: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "{what}: exit {:?} (a signal means the process aborted)\n{stderr}",
        out.status.code()
    );
    assert!(!stderr.contains("double free"), "{what}:\n{stderr}");
}

const ALIAS_TRAIN: &str = r#"
from nsl.nn.losses import cross_entropy

model Student:
    xs: Tensor = tensor_sin(arange(0.0, 32.0)).reshape([8, 4])
    ws: Tensor = tensor_cos(arange(0.0, 16.0)).reshape([4, 4]) * 0.5

let student = Student()
let t0 = zeros([8])
let t = t0

train(model = student, epochs = 2):
    optimizer: SGD(lr = 0.1)
    step(batch):
        let logits = student.xs @ student.ws.transpose(0, 1)
        let loss = cross_entropy(logits, t)
print(t0)
print(t)
"#;

/// The reported program shape, on both AD paths: the run must end normally,
/// and both names still read the eight zeros.
#[test]
fn a_module_level_alias_used_in_a_train_step_exits_cleanly() {
    for args in [&[][..], &["--source-ad"][..]] {
        let out = run(ALIAS_TRAIN, args);
        assert_clean_exit(&out, &format!("train {args:?}"));
        assert_eq!(numbers(&out), vec![0.0; 16], "train {args:?}: both names print eight zeros");
    }
}

/// The same alias with a distill block in place of the train block.
#[test]
fn a_module_level_alias_used_in_a_distill_step_exits_cleanly() {
    let source = ALIAS_TRAIN
        .replace("let student = Student()", "let student = Student()\nlet teacher = Student()")
        .replace("train(model = student, epochs = 2):", "distill(teacher = teacher, student = student, epochs = 2):");
    let out = run(&source, &["--source-ad"]);
    assert_clean_exit(&out, "distill");
    assert_eq!(numbers(&out), vec![0.0; 16]);
}

/// Each name is an owner in its own right: rebinding one leaves the other
/// intact, and an update through one does not show through the other.
#[test]
fn the_two_names_are_independent_owners() {
    let out = run(
        r#"
let a = full([3], 2.0)
let b = a
let a = a * 10.0
print(b)
print(a)
let c = full([2], 1.0)
let d = c
let e = d + 5.0
print(c)
print(d)
print(e)
"#,
        &[],
    );
    assert_clean_exit(&out, "rebinding");
    assert_eq!(
        numbers(&out),
        vec![2.0, 2.0, 2.0, 20.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0, 6.0, 6.0],
        "stdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

/// An alias inside a function, of a local and of a parameter, with the
/// function called repeatedly: no abort, right values.
#[test]
fn an_alias_inside_a_function_is_freed_once() {
    let out = run(
        r#"
fn twice(x: Tensor) -> Tensor:
    let y = x
    let z = x * 2.0
    let w = z
    return w + y

let v = full([2], 1.5)
let r1: Tensor = twice(v)
let r2: Tensor = twice(v)
let r3: Tensor = twice(r2)
print(r1)
print(r3)
print(v)
"#,
        &[],
    );
    assert_clean_exit(&out, "function alias");
    assert_eq!(numbers(&out), vec![4.5, 4.5, 13.5, 13.5, 1.5, 1.5]);
}
