//! `m.blocks[i]` on a `[Block; N]` model field.
//!
//! A model array is N element pointers stored inline. Subscripting one fell
//! through to the list path, which read the array's first words as a list
//! header, so `m.blocks[0].w` produced a garbage handle and the program
//! aborted with "invalid tensor handle" on first use. Only
//! `for block in self.blocks` could reach the elements.

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

fn run(source: &str) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .arg("--target")
        .arg("cpu")
        .output()
        .unwrap()
}

const MODEL: &str = r#"model Block:
    w: Tensor = ones([2, 2])

    fn scale(self, k: float) -> Tensor:
        return self.w * k

model Net:
    blocks: [Block; 3] = Block()

    fn bump(self) -> Tensor:
        let second: Tensor = self.blocks[1].w
        return second + 1.0

let m = Net()
"#;

#[test]
fn a_literal_or_variable_index_reaches_the_element() {
    let source = format!(
        "{MODEL}{}",
        r#"
let first: Tensor = m.blocks[0].w
print(first.sum())
print(m.blocks[2].w.sum())
let i = 1
let by_variable: Tensor = m.blocks[i].w
print(by_variable.sum())
print(m.blocks[i].scale(3.0).sum())
print(m.bump().sum())
"#
    );
    let out = run(&source);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    let printed: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    assert_eq!(printed, vec![4.0, 4.0, 4.0, 12.0, 8.0], "stderr:\n{stderr}");
}

#[test]
fn a_literal_index_out_of_range_is_a_compile_error() {
    let source = format!("{MODEL}\nlet t: Tensor = m.blocks[3].w\nprint(t.sum())\n");
    let out = run(&source);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(
        stderr.contains("index 3 is out of range for a model array of 3 element(s)"),
        "{stderr}"
    );
}

#[test]
fn a_runtime_index_out_of_range_stops_the_program() {
    let source = format!(
        "{MODEL}\nlet i = 3\nlet t: Tensor = m.blocks[i].w\nprint(t.sum())\nprint(\"unreachable\")\n"
    );
    let out = run(&source);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "an out-of-range index ran to completion");
    assert!(stderr.contains("model array index out of range"), "{stderr}");
    assert!(!String::from_utf8_lossy(&out.stdout).contains("unreachable"));
}
