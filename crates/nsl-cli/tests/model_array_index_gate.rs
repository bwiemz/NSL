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

/// Three elements that differ, so that an index which is ignored (or
/// scaled wrongly) reads the wrong one: each block's weight is its own
/// random draw. The loop is the reference; it reached the elements
/// correctly before subscripting did. (The loop binds its sum to a typed
/// `let` before printing: a field of a loop variable has no static type,
/// and `print` of an untyped value prints the handle as an integer. That
/// is a separate, open defect.)
#[test]
fn each_index_reaches_its_own_element() {
    let source = r#"model Block:
    w: Tensor = randn([2, 2])

model Net:
    blocks: [Block; 3] = Block()

    fn sums_in_order(self) -> Tensor:
        for block in self.blocks:
            let s: Tensor = block.w.sum()
            print(s)
        return zeros([1])

let m = Net()
m.sums_in_order()
print(m.blocks[0].w.sum())
print(m.blocks[1].w.sum())
let i = 2
print(m.blocks[i].w.sum())
"#;
    let out = run(source);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    let printed: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    assert_eq!(printed.len(), 6, "stdout did not hold six sums: {printed:?}");
    let (by_loop, by_index) = printed.split_at(3);
    assert_eq!(by_loop, by_index, "indexing and iteration disagree");
    assert!(
        by_loop[0] != by_loop[1] && by_loop[1] != by_loop[2] && by_loop[0] != by_loop[2],
        "the three blocks are not distinguishable, so this proves nothing: {by_loop:?}"
    );
}

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

// ---------------------------------------------------------------------------
// A list initialiser: `blocks: [Block; 2] = [Block(1.0), Block(3.0)]`
// ---------------------------------------------------------------------------

/// Element i is list item i. The list form compiled and stored the LIST
/// pointer in every slot, so reading `block.w` aborted on "invalid tensor
/// handle". Built here with different constructor arguments, so the
/// elements differ and their order shows: x = [1, 1] through
/// `full(2x2, 1)` then `full(2x2, 3)` is [2, 2] then [12, 12].
#[test]
fn a_list_initialiser_gives_each_element_its_item() {
    let out = run(r#"model Block(k: float):
    w: Tensor = full([2, 2], k)

    fn forward(self, h: Tensor) -> Tensor:
        return h @ self.w

model Net:
    blocks: [Block; 2] = [Block(1.0), Block(3.0)]

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
let y: Tensor = m.forward(ones([1, 2]))
print(y.sum())
let w0: Tensor = m.blocks[0].w
print(w0.sum())
let w1: Tensor = m.blocks[1].w
print(w1.sum())
"#);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    let printed: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    assert_eq!(printed, vec![24.0, 4.0, 12.0], "stderr:\n{stderr}");
}

#[test]
fn a_list_initialiser_of_the_wrong_length_is_a_compile_error() {
    let out = run(r#"model Block:
    w: Tensor = ones([2, 2])

model Net:
    blocks: [Block; 3] = [Block(), Block()]

let m = Net()
"#);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(
        stderr.contains("a `[Block; 3]` field initialised with a list needs 3 elements")
            && stderr.contains("this list has 2"),
        "{stderr}"
    );
}

/// Each element must be the array's model, in either form.
#[test]
fn an_element_of_another_type_is_a_compile_error() {
    for init in ["[Block(), Other()]", "Other()", "[Block(), 2]"] {
        let out = run(&format!(
            r#"model Block:
    w: Tensor = ones([2, 2])

model Other:
    v: Tensor = ones([2, 2])

model Net:
    blocks: [Block; 2] = {init}

let m = Net()
"#
        ));
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "`= {init}` compiled and ran");
        assert!(
            stderr.contains("an element of a `[Block; 2]` field must be a `Block`"),
            "`= {init}`:\n{stderr}"
        );
    }
}

// ---------------------------------------------------------------------------
// The element model declared after the model that holds the array
// ---------------------------------------------------------------------------

fn numbers_printed(out: &std::process::Output) -> Vec<f64> {
    String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

/// The checker pre-declared every model as `Unknown` and typed it only when
/// it reached its definition, so `[Block; 2]` ahead of `model Block`
/// resolved to `Unknown`: the loop compiled as a list iteration and
/// `block.w` as a dict lookup, which panicked in `dict.rs` on a misaligned
/// pointer. A plain field `b: Block` was refused at `self.b.forward(x)` as
/// a call on an untyped value. Both now behave as in declaration order:
/// [1, 1] through two `ones` blocks is [4, 4], through one is [2, 2].
#[test]
fn a_model_declared_after_its_user_is_typed() {
    let out = run(r#"model Net:
    blocks: [Block; 2] = Block()
    b: Block = Block()

    fn through_array(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = h @ block.w
        return h

    fn through_methods(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

    fn through_field(self, x: Tensor) -> Tensor:
        return self.b.forward(x)

model Block:
    w: Tensor = ones([2, 2])

    fn forward(self, h: Tensor) -> Tensor:
        return h @ self.w

let m = Net()
let x = ones([1, 2])
print(m.through_array(x))
print(m.through_methods(x))
print(m.through_field(x))
print(m.blocks[1].w)
"#);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    assert_eq!(
        numbers_printed(&out),
        vec![4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        "stderr:\n{stderr}"
    );
    assert!(!stderr.contains("Unknown type"), "{stderr}");
}

/// One SGD step (lr 0.1) of `loss = sum(x @ w0 @ w1 @ head)` with
/// x = [[1, 2], [3, 4]], w0 = w1 = 0.5 everywhere, head = ones([2, 1]),
/// with the three models declared outermost first. By hand: dhead = 5,
/// dw1 = 5 everywhere, dw0 = [[4, 4], [6, 6]], so w0 becomes
/// [[0.1, 0.1], [-0.1, -0.1]], w1 zero and head 0.5 -- what the same
/// program gives with its models in dependency order. In both AD modes.
#[test]
fn a_train_step_does_not_depend_on_declaration_order() {
    let source = r#"model Net:
    stack: Stack = Stack()
    head: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return self.stack.forward(x) @ self.head

model Stack:
    blocks: [Block; 2] = Block()

    fn forward(self, h: Tensor) -> Tensor:
        let y = h
        for block in self.blocks:
            y = block.forward(y)
        return y

model Block:
    w: Tensor = full([2, 2], 0.5)

    fn forward(self, h: Tensor) -> Tensor:
        return h @ self.w

let m = Net()
let x = arange(1.0, 5.0).reshape([2, 2])
train(model=m, epochs=1):
    optimizer: SGD(lr=0.1)
    step(batch):
        let loss = sum(m.forward(x))
print(m.stack.blocks[0].w)
print(m.stack.blocks[1].w)
print(m.head)
"#;
    let want = [0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5];
    for source_ad in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prog.nsl");
        std::fs::write(&path, source).unwrap();
        let mut cmd = Command::cargo_bin("nsl").unwrap();
        cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
            .current_dir(dir.path())
            .arg("run")
            .arg(&path)
            .args(["--target", "cpu"]);
        if source_ad {
            cmd.arg("--source-ad");
        }
        let out = cmd.output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "source_ad={source_ad}: run failed:\n{stderr}");
        let got = numbers_printed(&out);
        assert_eq!(got.len(), want.len(), "source_ad={source_ad}: printed {got:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "source_ad={source_ad}: printed {got:?}, expected {want:?}");
        }
    }
}

/// Typing models ahead of the statements resolves each field type twice;
/// a diagnostic that raises is reported once.
#[test]
fn an_undefined_field_type_is_reported_once() {
    let out = run(r#"model Net:
    b: Nope = Block()

model Block:
    w: Tensor = ones([2, 2])

let m = Net()
"#);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert_eq!(stderr.matches("undefined type `Nope`").count(), 1, "{stderr}");
}
