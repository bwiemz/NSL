//! C5 step 3: a creation builtin's runtime dtype is the one the checker gave
//! it (`docs/superpowers/specs/2026-09-26-dtype-semantics-design.md`).
//!
//! Before step 3 the checker typed every `zeros`/`ones`/`full`/`rand`/
//! `randn`/`arange` f64 while the runtime made f32, and an `f64` annotation
//! changed the checker's type and nothing else. Now the default is f32 on
//! both sides, and an annotated declaration -- `let x: Tensor<[3], f64> =
//! full([3], 0.1)` -- stores f64.
//!
//! The probe is the printed value: f64 0.1 prints `0.1`, and f32 0.1 prints
//! `0.10000000149011612` (the f32 value, widened for display). An f64
//! `arange(0.0, 0.4, 0.1)` holds `0.30000000000000004`, which no f32 holds.

use std::path::Path;
use std::process::Command;

fn run(src: &str) -> String {
    run_with(src, &[]).0
}

/// Run `src` with extra `nsl run` flags; returns (stdout, stderr).
fn run_with(src: &str, flags: &[&str]) -> (String, String) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("dtype_creation.nsl");
    std::fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .args(flags)
        .arg(&path)
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "the program failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    (stdout, stderr)
}

/// The line printed between `BEGIN_<tag>` and `END_<tag>`.
fn between(stdout: &str, tag: &str) -> String {
    let start = format!("BEGIN_{tag}\n");
    let after = stdout.split_once(&start).unwrap_or_else(|| panic!("no {tag} in:\n{stdout}")).1;
    after.split_once(&format!("\nEND_{tag}")).unwrap().0.trim().to_string()
}

const F32_TENTH: &str = "0.10000000149011612";

#[test]
fn an_f64_annotation_stores_f64_and_the_default_is_f32() {
    let src = r#"
let a: Tensor<[3], f64> = full([3], 0.1)
let b = full([3], 0.1)
let c: Tensor<[3], f32> = full([3], 0.1)
let d: Tensor<[4], f64> = arange(0.0, 0.4, 0.1)
let e: Tensor<[2], f64> = zeros([2])
let f: Tensor<[2], f64> = ones([2])
print("BEGIN_a")
print(a)
print("END_a")
print("BEGIN_b")
print(b)
print("END_b")
print("BEGIN_c")
print(c)
print("END_c")
print("BEGIN_d")
print(d)
print("END_d")
print("BEGIN_e")
print(e + 0.1)
print("END_e")
print("BEGIN_f")
print(f - 0.9)
print("END_f")
"#;
    let out = run(src);
    let a = between(&out, "a");
    assert_eq!(a, "tensor([0.1, 0.1, 0.1])", "f64-annotated full() must hold f64 0.1");
    let b = between(&out, "b");
    assert!(b.contains(F32_TENTH), "unannotated full() is f32: {b}");
    let c = between(&out, "c");
    assert!(c.contains(F32_TENTH), "f32-annotated full() is f32: {c}");
    let d = between(&out, "d");
    assert!(d.contains("0.30000000000000004"), "f64 arange computes in f64: {d}");
    let e = between(&out, "e");
    assert_eq!(e, "tensor([0.1, 0.1])", "f64-annotated zeros() + 0.1 stays f64");
    let f = between(&out, "f");
    assert!(f.contains("0.09999999999999998"), "f64-annotated ones() - 0.9 stays f64: {f}");
}

/// Source AD lowers creation calls to the f32 FFIs and makes its constants f32
/// rank-0 tensors, so a grad block over f64 tensors is left to the tape --
/// whose codegen keeps f64 -- instead of mixing dtypes (refused since C5 step
/// 4) or silently computing in f32.
#[test]
fn source_ad_leaves_an_f64_grad_block_to_the_tape() {
    let src = r#"
let x: Tensor<[3], f64> = full([3], 2.0)
let (l, d) = grad(x):
    let c: Tensor<[3], f64> = full([3], 0.1)
    sum(x * x) + sum(c)
print("BEGIN_d")
print(d)
print("END_d")
"#;
    let (out, err) = run_with(src, &["--source-ad"]);
    assert!(
        err.contains("this grad block computes on f64 tensors, which source AD does not lower"),
        "source AD must refuse the f64 grad block:\n{err}"
    );
    assert!(err.contains("falling back to tape-based AD"), "{err}");
    assert_eq!(between(&out, "d"), "tensor([4.0, 4.0, 4.0])");
}
