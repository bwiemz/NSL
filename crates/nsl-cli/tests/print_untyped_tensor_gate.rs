//! `print` of a tensor the checker could not type must print the tensor.
//!
//! `print` picks its runtime routine from the checker's type for the
//! argument, and fell back to "an integer" for anything 64 bits wide whose
//! type was unknown. A tensor is a 64-bit handle, so such a tensor printed
//! as a large integer: its address.
//!
//! The commonest way to get there was a `for` loop over a model array. The
//! loop variable was typed as the element model with no fields, so
//! `block.w` and everything built on it had no type.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn run(source: &str) -> (String, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .args(["--target", "cpu"])
        .output()
        .unwrap();
    let (stdout, stderr) =
        (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned());
    assert!(out.status.success(), "run failed:\n{stderr}");
    (stdout, stderr)
}

fn numbers(text: &str) -> Vec<f64> {
    text.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

const MODELS: &str = r#"
model Block(scale: float):
    w: Tensor = full([2, 2], scale)

    fn total(self) -> Tensor:
        return self.w.sum()

model Toy:
    blocks: [Block; 2] = Block(1.5)
"#;

/// Fields of the loop variable are typed: for each of the two blocks, the
/// sum of four elements of 1.5, the field itself, and the sum again through
/// a method. A handle would be a number in the trillions.
#[test]
fn a_loop_over_a_model_array_prints_tensor_values() {
    let source = format!(
        "{MODELS}
    fn show(self) -> int:
        for block in self.blocks:
            print(block.w.sum())
            print(block.w)
            print(block.total())
        return 0

let m = Toy()
let r = m.show()
"
    );
    let (stdout, stderr) = run(&source);
    assert_eq!(
        numbers(&stdout),
        vec![6.0, 1.5, 1.5, 1.5, 1.5, 6.0, 6.0, 1.5, 1.5, 1.5, 1.5, 6.0],
        "stdout:\n{stdout}"
    );
    assert!(!stderr.contains("printed as an integer"), "nothing here is untyped:\n{stderr}");
}

/// The same loop at module level, over the instance's array.
#[test]
fn a_module_level_loop_over_a_model_array_prints_tensor_values() {
    let source = format!(
        "{MODELS}
let m = Toy()
for block in m.blocks:
    print(block.w.sum())
"
    );
    let (stdout, _) = run(&source);
    assert_eq!(numbers(&stdout), vec![6.0, 6.0], "stdout:\n{stdout}");
}

/// Where the checker has no type at all (an unannotated parameter), `print`
/// still prints a tensor-returning call as a tensor, and says so when it
/// has to print a bare 64-bit value as an integer: for a tensor that number
/// is the handle.
#[test]
fn a_value_of_unknown_type_is_printed_with_a_warning_and_a_tensor_call_as_a_tensor() {
    let (stdout, stderr) = run(
        "fn show(x):
    print(x.sum())
    print(x)

let t = full([2], 1.5)
show(t)
",
    );
    let printed = numbers(&stdout);
    assert_eq!(printed.len(), 2, "stdout:\n{stdout}");
    assert_eq!(printed[0], 3.0, "`x.sum()` is a tensor whatever `x` was typed as; stdout:\n{stdout}");
    assert!(
        stderr.contains("print() argument has no type known at compile time and is printed as an integer"),
        "an untyped value printed as an integer must be announced:\n{stderr}"
    );
}
