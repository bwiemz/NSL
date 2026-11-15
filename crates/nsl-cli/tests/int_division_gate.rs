//! `int / int` is true division, and a shape dimension must be an int.
//!
//! The checker types `int / int` as a float (as Python 3 does), but codegen
//! emitted an integer `sdiv`. The integer then met float-typed code:
//! `let h = c / n` panicked Cranelift ("declared type of variable doesn't
//! match type of value"), `print(7 / 2)` failed the verifier, and
//! `(7 / 2) * 2.0` computed 6.0. Inside a shape list the truncated integer
//! happened to work for exact quotients; with true division a float
//! dimension is a type error that names `//`.

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

fn numbers(out: &Output) -> Vec<f64> {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    String::from_utf8_lossy(&out.stdout).lines().map(|l| l.trim().parse().unwrap()).collect()
}

/// Python 3's values: 7 / 2 = 3.5, 7 // 2 = 3, -7 // 2 = -4, -7 % 2 = 1,
/// 6 / 3 = 2.0; in a function and a model method as well as at top level.
#[test]
fn int_true_division_is_a_float() {
    let out = run(r#"fn half(c: int, n: int) -> float:
    let h = c / n
    return h
model M:
    k: int = 7
    fn q(self) -> float:
        return self.k / 2
print(7 / 2)
let a = 7
let b = 2
let h = a / b
print(h)
print((a / b) * 2.0)
print(a // b)
print(-7 // 2)
print(-7 % 2)
print(6 / 3)
print(half(9, 2))
let m = M()
print(m.q())
"#);
    assert_eq!(numbers(&out), [3.5, 3.5, 7.0, 3.0, -4.0, 1.0, 2.0, 4.5, 3.5]);
}

#[test]
fn int_division_by_zero_still_stops_the_program() {
    let out = run("let a = 7\nlet z = 0\nprint(a / z)\nprint(\"unreachable\")\n");
    assert!(!out.status.success());
    assert!(!String::from_utf8_lossy(&out.stdout).contains("unreachable"));
}

/// A float dimension is refused with the reason, wherever the shape list
/// is written; the `//` spelling it suggests builds the intended shape.
#[test]
fn a_float_shape_dimension_is_a_type_error_naming_floor_division() {
    for (call, label) in [
        ("x.reshape([2, nh, c / nh])", "use `//` for integer division"),
        ("zeros([2, c / nh])", "use `//` for integer division"),
        ("x.reshape([2, nh, hd])", "use `//` for integer division"),
    ] {
        let out = run(&format!(
            "let x = ones([2, 8])\nlet c = 8\nlet nh = 2\nlet hd = c / nh\nlet y = {call}\nprint(y)\n"
        ));
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "`{call}` compiled");
        assert!(stderr.contains("a shape dimension is a float, not an int") && stderr.contains(label), "`{call}`:\n{stderr}");
    }
    let out = run("let x = arange(0.0, 16.0).reshape([2, 8])\nlet c = 8\nlet nh = 2\nprint(x.reshape([2, nh, c // nh]))\n");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]])"
    );
}

/// `/` is true division for every integer width, not only `int`:
/// `int64 / int64` and `int32 / int32` stayed integers (`sdiv`, 3).
#[test]
fn sized_int_division_is_a_float_too() {
    let out = run(r#"let a: int64 = 7
let b: int64 = 2
print(a / b)
let c: int32 = 7
let d: int32 = 2
print(c / d)
print(a // b)
"#);
    assert_eq!(numbers(&out), [3.5, 3.5, 3.0]);
}

/// Every position that needs an int refuses a float with the `//` hint,
/// where it used to fail Cranelift's verifier without a span (`range(n /
/// 2)`, `xs[n / 2]`) or truncate without a word: `[4, n / 2]` was a list
/// of ints (typed by its first element) and reshaped a 12-element tensor
/// to [4, 3]. A list with a float element is now a list of floats.
#[test]
fn a_float_where_an_int_is_needed_is_refused() {
    let cases = [
        ("let n = 7\nfor i in range(n / 2):\n    print(i)\n", "a range() argument must be an int"),
        ("const H = 8 / 2\nfor i in range(H):\n    print(i)\n", "a range() argument must be an int"),
        ("for i in 0..7 / 2:\n    print(i)\n", "a range bound must be an int"),
        ("let xs = [10, 20, 30, 40]\nlet n = 7\nprint(xs[n / 2])\n", "an index must be an int"),
        ("let xs = [10, 20, 30, 40]\nlet n = 7\nprint(xs[1:n / 2])\n", "an index must be an int"),
        ("let t = ones([2, 2])\nlet n = 3\nprint(t[0, n / 2])\n", "an index must be an int"),
        (
            "let x = ones([2, 6])\nlet n = 7\nlet s = [4, n / 2]\nprint(x.reshape(s))\n",
            "the shape is a list of floats",
        ),
        (
            "let n = 7\nlet s = [n / 2, 4]\nprint(zeros(s))\n",
            "the shape is a list of floats",
        ),
        ("fn f(k: int) -> int:\n    return k\nprint(f(7 / 2))\n", "argument 1: expected int, got float"),
    ];
    for (src, want) in cases {
        let out = run(src);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "ran:\n{src}\n{}", String::from_utf8_lossy(&out.stdout));
        assert!(stderr.contains(want), "{src}\nwanted {want:?} in:\n{stderr}");
        assert!(stderr.contains("`//`"), "{src}\nno `//` hint in:\n{stderr}");
        assert!(!stderr.contains("Verifier"), "{src}\n{stderr}");
    }
}
