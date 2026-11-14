//! `-w weights.safetensors`: the program runs with the file's values.
//!
//! `-w` used to hand the file to the compiler only. Passes then rewrote code
//! on the strength of its values (a matmul by a near-identity weight was
//! dropped, a product of two weights folded into a constant) while the model
//! still held its initializer: a `[4, 4]` weight of twos, compiled against a
//! file holding the identity, printed `x` for `x @ self.w`, and printing the
//! weight showed the twos.
//!
//! What is pinned here:
//!
//! 1. the file is loaded into the model (the weight itself is printed, so
//!    "loaded" is told apart from "the matmul was folded away");
//! 2. a parameter the program writes afterwards is loaded but not
//!    specialized on, and the report says so;
//! 3. a compiled program refuses to start on a file whose specialized
//!    tensors have changed since.
//!
//! Every expected value is written out by hand or was computed with NumPy
//! (the expression is given at each); none comes from the compiler.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// A safetensors file of f32 tensors: an 8-byte little-endian header length,
/// a JSON header padded with spaces to a multiple of 8, then the raw values.
fn write_safetensors(path: &Path, tensors: &[(&str, &[usize], &[f64])]) {
    write_safetensors_as(path, "F32", tensors);
}

/// The same in another dtype: F32, or BF16 (the top half of the f32, so
/// exact for the small integers and halves these tests use).
fn write_safetensors_as(path: &Path, dtype: &str, tensors: &[(&str, &[usize], &[f64])]) {
    let mut entries = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, values) in tensors {
        assert_eq!(shape.iter().product::<usize>(), values.len(), "{name}");
        let begin = data.len();
        for v in *values {
            let bits = (*v as f32).to_bits();
            match dtype {
                "F32" => data.extend_from_slice(&bits.to_le_bytes()),
                "BF16" => {
                    assert_eq!(bits & 0xFFFF, 0, "{v} is not exact in bf16");
                    data.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
                }
                other => panic!("no writer for {other}"),
            }
        }
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{begin},{}]}}",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(","),
            data.len()
        ));
    }
    let mut header = format!("{{{}}}", entries.join(",")).into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

fn nsl() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.env("NSL_STDLIB_PATH", repo_root().join("stdlib"));
    cmd
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `nsl run prog.nsl --target cpu [-w weights] [extra...]`.
fn run_in(dir: &Path, source: &str, weights: Option<&Path>, extra: &[&str]) -> Output {
    let file = dir.join("prog.nsl");
    std::fs::write(&file, source).unwrap();
    let mut cmd = nsl();
    cmd.current_dir(dir)
        .arg("run")
        .arg(&file)
        .args(["--target", "cpu"])
        .args(extra);
    if let Some(w) = weights {
        cmd.arg("-w").arg(w);
    }
    cmd.output().expect("spawn nsl run")
}

/// Run `source` against a weights file holding `tensors`, and require success.
fn run_ok(
    source: &str,
    tensors: Option<&[(&str, &[usize], &[f64])]>,
    extra: &[&str],
) -> (String, String) {
    let dir = tempfile::tempdir().unwrap();
    let weights = tensors.map(|t| {
        let path = dir.path().join("weights.safetensors");
        write_safetensors(&path, t);
        path
    });
    let out = run_in(dir.path(), source, weights.as_deref(), extra);
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(
        out.status.success(),
        "run failed: {}\n--- stdout\n{stdout}\n--- stderr\n{stderr}",
        out.status
    );
    (stdout, stderr)
}

/// The numbers printed between the lines `<tag>_BEGIN` and `<tag>_END`.
fn section(stdout: &str, tag: &str) -> Vec<f64> {
    let (begin, end) = (format!("{tag}_BEGIN"), format!("{tag}_END"));
    let mut lines = stdout.lines().skip_while(|l| l.trim() != begin);
    assert!(lines.next().is_some(), "no line {begin} in:\n{stdout}");
    let inner: Vec<&str> = lines.take_while(|l| l.trim() != end).collect();
    inner
        .join(" ")
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

#[track_caller]
fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: got {got:?}, want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5,
            "{what}[{i}]: got {g}, want {w}\n got {got:?}\nwant {want:?}"
        );
    }
}

/// A source line that prints a tensor between two tags.
fn show(tag: &str, expr: &str) -> String {
    format!(
        "let t_{tag}: Tensor = {expr}\nprint(\"{tag}_BEGIN\")\nprint(t_{tag})\nprint(\"{tag}_END\")\n"
    )
}

// ── The fixture ──────────────────────────────────────────────────────────────

/// A weight whose initializer is all twos, a forward pass that multiplies by
/// it, and both the product and the weight itself printed.
fn toy_program() -> String {
    format!(
        "model Toy:\n    w: Tensor = ones([4, 4]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w\n\n\
         let m = Toy()\nlet x = arange(1.0, 9.0).reshape([2, 4])\n{}{}",
        show("Y", "m.forward(x)"),
        show("W", "m.w")
    )
}

/// x = [[1, 2, 3, 4], [5, 6, 7, 8]].
const X: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

const IDENTITY: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// A matrix that is neither the initializer nor the identity.
const V: [f64; 16] = [
    1.0, 0.0, 2.0, 0.0, 0.0, -1.0, 0.0, 3.0, 0.5, 0.0, 0.0, 1.0, 0.0, 2.0, -2.0, 0.0,
];

/// x @ V, by hand. Row 1: [1 + 1.5, -2 + 8, 2 - 8, 6 + 3]; row 2:
/// [5 + 3.5, -6 + 16, 10 - 16, 18 + 7].
const X_V: [f64; 8] = [2.5, 6.0, -6.0, 9.0, 8.5, 10.0, -6.0, 25.0];

// ── 1. The file is loaded ────────────────────────────────────────────────────

/// The control: without `-w` the model holds its initializer. Every column
/// of x @ twos is twice the row sum: 2 * 10 and 2 * 26.
#[test]
fn without_the_flag_the_model_holds_its_initializer() {
    let (stdout, stderr) = run_ok(&toy_program(), None, &[]);
    assert_close(
        &section(&stdout, "Y"),
        &[20.0, 20.0, 20.0, 20.0, 52.0, 52.0, 52.0, 52.0],
        "y",
    );
    assert_close(&section(&stdout, "W"), &[2.0; 16], "w");
    assert!(!stderr.contains("weight-aware"), "{stderr}");
}

/// The case that was wrong. With an identity file the output is `x`, and it
/// is `x` because the weight is the identity: the weight is printed, and it
/// is the file's, not the twos. The matmul is also dropped, which is now
/// correct, and the report says both.
#[test]
fn an_identity_file_is_loaded_into_the_model() {
    let (stdout, stderr) = run_ok(&toy_program(), Some(&[("w", &[4, 4], &IDENTITY)]), &[]);
    assert_close(&section(&stdout, "Y"), &X, "y");
    let w = section(&stdout, "W");
    assert_close(&w, &IDENTITY, "w");
    assert!(
        (w.iter().sum::<f64>() - 4.0).abs() < 1e-9,
        "the weight sums to {}",
        w.iter().sum::<f64>()
    );
    assert!(
        stderr.contains("'Toy.w' is loaded from the file's 'w'"),
        "{stderr}"
    );
    assert!(
        stderr.contains("eliminated near-identity matmul (weight 'w')"),
        "{stderr}"
    );
}

/// A file that is neither the initializer nor a no-op: the product is x @ V
/// and the weight is V. Nothing is eliminated.
#[test]
fn a_file_that_is_not_the_identity_is_loaded_into_the_model() {
    let (stdout, stderr) = run_ok(&toy_program(), Some(&[("w", &[4, 4], &V)]), &[]);
    assert_close(&section(&stdout, "Y"), &X_V, "y");
    let w = section(&stdout, "W");
    assert_close(&w, &V, "w");
    assert!(
        (w.iter().sum::<f64>() - 6.5).abs() < 1e-9,
        "the weight sums to {}",
        w.iter().sum::<f64>()
    );
    assert!(!stderr.contains("eliminated"), "{stderr}");
}

/// A tensor named `Class.field` is loaded into that class's field, and one
/// the model has no field for is reported, not loaded anywhere.
#[test]
fn a_tensor_named_by_class_is_loaded_and_a_stray_one_is_reported() {
    let stray = [9.0; 4];
    let (stdout, stderr) = run_ok(
        &toy_program(),
        Some(&[("Toy.w", &[4, 4], &V), ("elsewhere.w", &[2, 2], &stray)]),
        &[],
    );
    assert_close(&section(&stdout, "Y"), &X_V, "y");
    assert_close(&section(&stdout, "W"), &V, "w");
    assert!(
        stderr.contains("'Toy.w' is loaded from the file's 'Toy.w'"),
        "{stderr}"
    );
    let note = stderr
        .lines()
        .find(|l| l.contains("match no"))
        .unwrap_or_else(|| panic!("{stderr}"));
    assert!(
        note.contains("elsewhere.w") && !note.contains("Toy.w"),
        "{note}"
    );
}

/// A file whose tensor has another shape than the field is an error that
/// names both, not a model left at its initializer.
#[test]
fn a_tensor_of_the_wrong_shape_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let weights = dir.path().join("weights.safetensors");
    write_safetensors(&weights, &[("w", &[2, 2], &[1.0, 0.0, 0.0, 1.0])]);
    let out = run_in(dir.path(), &toy_program(), Some(&weights), &[]);
    let stderr = text(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(
        stderr.contains("'w'") && stderr.contains("[2, 2]") && stderr.contains("[4, 4]"),
        "{stderr}"
    );
    assert!(
        !text(&out.stdout).contains("Y_BEGIN"),
        "the program ran: {}",
        text(&out.stdout)
    );
}

// ── Parameters named by their place in the model ─────────────────────────────

/// Two blocks in an array and a head: x @ blocks[0].w @ blocks[1].w @ head.
fn net_program() -> String {
    format!(
        "model Blk:\n    w: Tensor = ones([4, 4]) * 2.0\n\n    fn forward(self, h: Tensor) -> Tensor:\n        return h @ self.w\n\n\
         model Net:\n    blocks: [Blk; 2] = Blk()\n    head: Tensor = ones([4, 3]) * 2.0\n\n\
         \x20   fn forward(self, x: Tensor) -> Tensor:\n        let h = x * 1.0\n        for block in self.blocks:\n            h = block.forward(h)\n        return h @ self.head\n\n\
         let m = Net()\nlet x = arange(1.0, 9.0).reshape([2, 4])\n{}",
        show("Y", "m.forward(x)")
    )
}

/// Swaps the first two columns, doubles the third, negates the fourth.
const U: [f64; 16] = [
    0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -1.0,
];
const HEAD: [f64; 12] = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0];

/// x @ V @ U @ HEAD, by hand. x @ V @ U is [[6, 2.5, -12, -9],
/// [10, 8.5, -12, -25]] (X_V with U applied to its columns); times HEAD:
/// [6 - 12, 2.5 - 12, 6 + 2.5 - 18] and [10 - 12, 8.5 - 12, 10 + 8.5 - 50].
/// (NumPy: `x @ V @ U @ HEAD` gives the same six numbers.)
const X_V_U_HEAD: [f64; 6] = [-6.0, -9.5, -9.5, -2.0, -3.5, -31.5];

/// A parameter of a nested model is loaded from the tensor named by its path
/// from the model being constructed, in NSL's spelling (`blocks[0].w`, what
/// `model_save` writes) and in the dotted one checkpoints from elsewhere use
/// (`blocks.0.w`). The two blocks get different matrices, so one loaded into
/// both, or the two swapped, changes the answer.
#[test]
fn a_nested_parameter_is_loaded_by_its_path() {
    for names in [["blocks[0].w", "blocks[1].w"], ["blocks.0.w", "blocks.1.w"]] {
        let (stdout, stderr) = run_ok(
            &net_program(),
            Some(&[
                (names[0], &[4, 4], &V),
                (names[1], &[4, 4], &U),
                ("head", &[4, 3], &HEAD),
            ]),
            &[],
        );
        assert_close(&section(&stdout, "Y"), &X_V_U_HEAD, names[0]);
        assert!(
            stderr.contains(&format!("is loaded from the file's '{}'", names[1])),
            "{stderr}"
        );
    }
}

/// A bare name that could belong to more than one parameter is not
/// specialized on. Here the file's `w` is the identity and `Blk` has a field
/// `w`, which used to be enough to drop the matmul in `Blk.forward`; but
/// the file also names `blocks[1].w`, and that is what the second block
/// holds when it runs. The answer must still be x @ V @ U @ HEAD.
#[test]
fn a_name_that_two_parameters_could_claim_is_not_specialized_on() {
    let (stdout, stderr) = run_ok(
        &net_program(),
        Some(&[
            ("w", &[4, 4], &IDENTITY),
            ("blocks[0].w", &[4, 4], &V),
            ("blocks[1].w", &[4, 4], &U),
            ("head", &[4, 3], &HEAD),
        ]),
        &[],
    );
    assert_close(&section(&stdout, "Y"), &X_V_U_HEAD, "y");
    assert!(!stderr.contains("eliminated"), "{stderr}");
    let line = stderr
        .lines()
        .find(|l| l.contains("not specialized on 'w' (near-identity matmul elimination skipped)"))
        .unwrap_or_else(|| panic!("{stderr}"));
    assert!(line.contains("blocks["), "{line}");
}

// ── The constant fold ────────────────────────────────────────────────────────

/// Two weights multiplied together before the input reaches them.
fn product_program(tail: &str) -> String {
    format!(
        "model Two:\n    a: Tensor = ones([4, 4]) * 2.0\n    b: Tensor = ones([4, 3]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ (self.a @ self.b)\n\n\
         let m = Two()\nlet x = arange(1.0, 9.0).reshape([2, 4])\n{}{}{}{tail}",
        show("Y", "m.forward(x)"),
        show("A", "m.a"),
        show("B", "m.b")
    )
}

/// x @ V @ HEAD, by hand: X_V times HEAD is [2.5 - 6, 6 - 6, 2.5 + 6 + 18]
/// and [8.5 - 6, 10 - 6, 8.5 + 10 + 50].
const X_V_HEAD: [f64; 6] = [-3.5, 0.0, 26.5, 2.5, 4.0, 68.5];

/// The product of two weights nothing writes is folded into a constant, and
/// the constant is the product of what the model holds: both weights print
/// as the file's.
#[test]
fn a_product_of_two_loaded_weights_is_folded() {
    let (stdout, stderr) = run_ok(
        &product_program(""),
        Some(&[("a", &[4, 4], &V), ("b", &[4, 3], &HEAD)]),
        &[],
    );
    assert_close(&section(&stdout, "Y"), &X_V_HEAD, "y");
    assert_close(&section(&stdout, "A"), &V, "a");
    assert_close(&section(&stdout, "B"), &HEAD, "b");
    let line = stderr
        .lines()
        .find(|l| l.contains("constant-folded a @ b"))
        .unwrap_or_else(|| panic!("{stderr}"));
    assert!(line.contains("from the file's exact values"), "{line}");
}

/// The same program with one of the two weights assigned afterwards: the
/// fold is refused (it would freeze the old `b` into the code), and the
/// second forward pass uses the new `b`. With `b` all twos every column of
/// x @ V @ b is twice the row sum of X_V: 2 * 11.5 and 2 * 37.5.
#[test]
fn a_product_with_a_written_weight_is_not_folded() {
    let tail = format!("m.b = ones([4, 3]) * 2.0\n{}", show("Y2", "m.forward(x)"));
    let (stdout, stderr) = run_ok(
        &product_program(&tail),
        Some(&[("a", &[4, 4], &V), ("b", &[4, 3], &HEAD)]),
        &[],
    );
    assert_close(&section(&stdout, "Y"), &X_V_HEAD, "y before the assignment");
    assert_close(
        &section(&stdout, "Y2"),
        &[23.0, 23.0, 23.0, 75.0, 75.0, 75.0],
        "y after it",
    );
    assert!(!stderr.contains("constant-folded"), "{stderr}");
    let line = stderr
        .lines()
        .find(|l| l.contains("not specialized on 'b' (constant fold skipped)"))
        .unwrap_or_else(|| panic!("{stderr}"));
    assert!(line.contains("assigns to a field named 'b'"), "{line}");
}

// ── 2. A parameter the program writes ────────────────────────────────────────

/// One SGD step on the toy weight, with the weight and the forward pass
/// printed before and after.
fn training_program() -> String {
    format!(
        "from nsl.nn.losses import mse_loss\n\n\
         model Toy:\n    w: Tensor = ones([4, 4]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w\n\n\
         let m = Toy()\nlet x = arange(1.0, 9.0).reshape([2, 4]) * 0.1\n\
         let y = arange(0.0, 8.0).reshape([2, 4]) * 0.25 - full([2, 4], 1.0)\n{}{}\
         train(model = m, epochs = 1):\n    optimizer: SGD(lr = 0.5)\n    step(batch):\n        let loss = mse_loss(m.forward(x), y)\n\n{}{}",
        show("W0", "m.w"),
        show("Y0", "m.forward(x)"),
        show("W1", "m.w"),
        show("Y1", "m.forward(x)")
    )
}

/// The weight after one step from the identity, and the forward pass with
/// it. NumPy, in f64:
///
/// ```text
/// x  = np.arange(1.0, 9.0).reshape(2, 4) * 0.1
/// y  = np.arange(0.0, 8.0).reshape(2, 4) * 0.25 - 1.0
/// W1 = np.eye(4) - 0.5 * (2 / 8) * x.T @ (x @ np.eye(4) - y)
/// Y1 = x @ W1
/// ```
const W_AFTER_ONE_STEP: [f64; 16] = [
    0.955, -0.03375, -0.0225, -0.01125, -0.065, 0.95, -0.035, -0.02, -0.085, -0.06625, 0.9525,
    -0.02875, -0.105, -0.0825, -0.06, 0.9625,
];
const Y_AFTER_ONE_STEP: [f64; 8] = [
    0.015, 0.13375, 0.2525, 0.37125, 0.295, 0.44075, 0.5865, 0.73225,
];

/// A trained weight is loaded from the file and not specialized on. With an
/// identity file the matmul would be dropped if it were: the forward pass
/// would return `x` whatever the weight became, and the weight would get no
/// gradient at all. Instead the report says the rewrite was refused and why,
/// the step moves the weight from the file's identity (not from the twos:
/// from there Y1 would start 1.4325), and the output is x @ W1.
#[test]
fn a_trained_parameter_is_loaded_but_not_specialized_on() {
    let (stdout, stderr) = run_ok(&training_program(), Some(&[("w", &[4, 4], &IDENTITY)]), &[]);

    let line = stderr
        .lines()
        .find(|l| l.contains("not specialized on 'w' (near-identity matmul elimination skipped)"))
        .unwrap_or_else(|| panic!("no refusal in the report:\n{stderr}"));
    assert!(line.contains("train block"), "{line}");
    assert!(
        !stderr.contains("eliminated near-identity matmul"),
        "{stderr}"
    );
    assert!(
        stderr.contains("no rewrite depends on its values"),
        "{stderr}"
    );

    // Before the step: the file's identity, so the forward pass gives x.
    let x: Vec<f64> = X.iter().map(|v| v * 0.1).collect();
    assert_close(&section(&stdout, "W0"), &IDENTITY, "w before the step");
    assert_close(&section(&stdout, "Y0"), &x, "y before the step");

    // After: the updated weight, and the output computed with it.
    let y1 = section(&stdout, "Y1");
    assert_close(
        &section(&stdout, "W1"),
        &W_AFTER_ONE_STEP,
        "w after the step",
    );
    assert_close(&y1, &Y_AFTER_ONE_STEP, "y after the step");
    let moved = y1
        .iter()
        .zip(&x)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        moved > 0.05,
        "the output after the step is still x (moved {moved})"
    );
}

/// An assignment to the field refuses the rewrite on that field too, and
/// the assigned value is what the forward pass uses: x @ twos, not x.
#[test]
fn an_assigned_parameter_is_not_specialized_on() {
    let program = format!(
        "model Toy:\n    w: Tensor = ones([4, 4]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w\n\n\
         let m = Toy()\nlet x = arange(1.0, 9.0).reshape([2, 4])\n{}m.w = ones([4, 4]) * 2.0\n{}",
        show("Y0", "m.forward(x)"),
        show("Y1", "m.forward(x)")
    );
    let (stdout, stderr) = run_ok(&program, Some(&[("w", &[4, 4], &IDENTITY)]), &[]);
    assert_close(&section(&stdout, "Y0"), &X, "before the assignment");
    assert_close(
        &section(&stdout, "Y1"),
        &[20.0, 20.0, 20.0, 20.0, 52.0, 52.0, 52.0, 52.0],
        "after it",
    );
    let line = stderr
        .lines()
        .find(|l| l.contains("not specialized on 'w'"))
        .unwrap_or_else(|| panic!("{stderr}"));
    assert!(line.contains("assigns to a field named 'w'"), "{line}");
}

// ── 3. The compiled program is tied to the file ──────────────────────────────

/// `nsl build prog.nsl -w weights -o exe`; returns the executable.
fn build_in(dir: &Path, source: &str, weights: &Path) -> PathBuf {
    build_with(dir, source, weights, &[]).0
}

/// `nsl build prog.nsl -w weights [extra...] -o exe`; the executable and the
/// build's report (stderr).
fn build_with(dir: &Path, source: &str, weights: &Path, extra: &[&str]) -> (PathBuf, String) {
    let file = dir.join("prog.nsl");
    std::fs::write(&file, source).unwrap();
    let exe = dir.join(if cfg!(windows) { "prog.exe" } else { "prog" });
    let out = nsl()
        .current_dir(dir)
        .arg("build")
        .arg(&file)
        .args(["--target", "cpu"])
        .args(extra)
        .arg("-w")
        .arg(weights)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("spawn nsl build");
    assert!(
        out.status.success(),
        "build failed:\n{}\n{}",
        text(&out.stdout),
        text(&out.stderr)
    );
    (exe, text(&out.stderr))
}

/// The `wspec=<16 hex digits>` key of the execution fingerprint compiled
/// into a program, if it has one.
fn wspec_of(exe: &Path) -> Option<String> {
    let bytes = std::fs::read(exe).unwrap();
    let at = bytes.windows(6).position(|w| w == b"wspec=")?;
    let digest: String = bytes[at + 6..]
        .iter()
        .take_while(|b| b.is_ascii_hexdigit())
        .map(|b| *b as char)
        .collect();
    assert_eq!(digest.len(), 16, "wspec={digest}");
    Some(digest)
}

/// A program specialized on a tensor runs on the file it was compiled with,
/// and refuses to start once that tensor has changed: the error names the
/// file and the parameter, the exit status is a failure, and no user code
/// has run. (Without the check it would print `x`, the answer for the old
/// file, while holding the new weight.)
#[test]
fn a_built_program_refuses_a_file_changed_after_compiling() {
    let dir = tempfile::tempdir().unwrap();
    let weights = dir.path().join("weights.safetensors");
    write_safetensors(&weights, &[("w", &[4, 4], &IDENTITY)]);
    let exe = build_in(dir.path(), &toy_program(), &weights);

    // The run's execution fingerprint records what was specialized on.
    assert!(
        wspec_of(&exe).is_some(),
        "no wspec= key in the built program"
    );

    // The control: the file as compiled.
    let out = Command::new(&exe)
        .current_dir(dir.path())
        .output()
        .expect("run the built program");
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert_close(
        &section(&text(&out.stdout), "Y"),
        &X,
        "y on the unmodified file",
    );
    assert_close(
        &section(&text(&out.stdout), "W"),
        &IDENTITY,
        "w on the unmodified file",
    );

    // The same file rewritten byte for byte is still the file.
    write_safetensors(&weights, &[("w", &[4, 4], &IDENTITY)]);
    let out = Command::new(&exe).current_dir(dir.path()).output().unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));

    // One element changed.
    let mut edited = IDENTITY;
    edited[5] = 1.5;
    write_safetensors(&weights, &[("w", &[4, 4], &edited)]);
    let out = Command::new(&exe).current_dir(dir.path()).output().unwrap();
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(
        !out.status.success(),
        "the program ran on an edited file:\n{stdout}\n{stderr}"
    );
    assert!(
        stderr.contains("weights.safetensors"),
        "the error does not name the file: {stderr}"
    );
    assert!(
        stderr.contains("parameter 'w' has different contents"),
        "{stderr}"
    );
    assert!(
        !stdout.contains("Y_BEGIN"),
        "user code ran before the refusal:\n{stdout}"
    );

    // The file removed.
    std::fs::remove_file(&weights).unwrap();
    let out = Command::new(&exe).current_dir(dir.path()).output().unwrap();
    assert!(!out.status.success());
    assert!(
        text(&out.stderr).contains("weights.safetensors"),
        "{}",
        text(&out.stderr)
    );
}

/// Programs compiled against different contents record different digests,
/// and the same contents the same one.
#[test]
fn the_fingerprint_key_follows_the_file_s_contents() {
    let digest = |values: &[f64]| {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.safetensors");
        write_safetensors(&weights, &[("w", &[4, 4], values)]);
        wspec_of(&build_in(dir.path(), &toy_program(), &weights)).expect("a wspec= key")
    };
    let (identity, again, other) = (digest(&IDENTITY), digest(&IDENTITY), digest(&V));
    assert_eq!(identity, again);
    assert_ne!(identity, other);
}

/// A program that specialized on nothing (here because it trains the
/// weight) is not tied to the file's contents: built against the identity
/// and run after the file was changed to V, it loads V. The first forward
/// pass is x @ V / 10 (x is a tenth of [[1..4], [5..8]]).
#[test]
fn a_built_program_that_specialized_on_nothing_loads_the_file_as_it_is() {
    let dir = tempfile::tempdir().unwrap();
    let weights = dir.path().join("weights.safetensors");
    write_safetensors(&weights, &[("w", &[4, 4], &IDENTITY)]);
    let exe = build_in(dir.path(), &training_program(), &weights);
    assert_eq!(
        wspec_of(&exe),
        None,
        "a program that specialized on nothing has a wspec= key"
    );

    write_safetensors(&weights, &[("w", &[4, 4], &V)]);
    let out = Command::new(&exe).current_dir(dir.path()).output().unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));
    let stdout = text(&out.stdout);
    assert_close(&section(&stdout, "W0"), &V, "w");
    let want: Vec<f64> = X_V.iter().map(|v| v * 0.1).collect();
    assert_close(&section(&stdout, "Y0"), &want, "y");
}

// ── More than one module ─────────────────────────────────────────────────────

/// The model lives in one imported module and a function that overwrites a
/// tensor in place in another; the entry module multiplies by the weight
/// itself (`x @ m.w`). Only the entry module is compiled with the file's
/// values, so the matmul it may drop is there, and the write that must stop
/// it is in a module it imports. The constructor, compiled in the model's
/// module, loads the file all the same. (A program counts as writing when
/// any module it imports could, called or not: the analysis does not follow
/// calls.)
#[test]
fn the_decision_is_made_over_every_module() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("toymodel.nsl"),
        "model Toy:\n    w: Tensor = ones([4, 4]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tools.nsl"),
        "fn overwrite(t: Tensor):\n    copy_data(t, ones([4, 4]) * 2.0)\n",
    )
    .unwrap();
    let weights = dir.path().join("weights.safetensors");
    write_safetensors(&weights, &[("w", &[4, 4], &IDENTITY)]);

    // Only reads: loaded through the imported constructor, and the entry
    // module's `x @ m.w` is dropped, correctly.
    let reads = format!(
        "from toymodel import Toy\n\nlet m = Toy()\nlet x = arange(1.0, 9.0).reshape([2, 4])\n{}{}{}",
        show("Y", "m.forward(x)"),
        show("Z", "x @ m.w"),
        show("W", "m.w")
    );
    let out = run_in(dir.path(), &reads, Some(&weights), &[]);
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(out.status.success(), "{stderr}");
    assert_close(&section(&stdout, "Y"), &X, "y");
    assert_close(&section(&stdout, "Z"), &X, "z");
    assert_close(&section(&stdout, "W"), &IDENTITY, "w");
    assert!(
        stderr.contains("'Toy.w' is loaded from the file's 'w'"),
        "{stderr}"
    );
    assert!(
        stderr.contains("eliminated near-identity matmul (weight 'w')"),
        "{stderr}"
    );

    // The program calls the imported function on the weight: the entry
    // module's matmul stays, and after the call it multiplies by the twos.
    let writes = format!(
        "from toymodel import Toy\nfrom tools import overwrite\n\nlet m = Toy()\nlet x = arange(1.0, 9.0).reshape([2, 4])\n{}overwrite(m.w)\n{}",
        show("Z0", "x @ m.w"),
        show("Z1", "x @ m.w")
    );
    let out = run_in(dir.path(), &writes, Some(&weights), &[]);
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(out.status.success(), "{stderr}");
    assert_close(&section(&stdout, "Z0"), &X, "z before the call");
    assert_close(
        &section(&stdout, "Z1"),
        &[20.0, 20.0, 20.0, 20.0, 52.0, 52.0, 52.0, 52.0],
        "z after the call",
    );
    assert!(!stderr.contains("eliminated"), "{stderr}");
    let line = stderr
        .lines()
        .find(|l| l.contains("not specialized on 'w' (near-identity matmul elimination skipped)"))
        .unwrap_or_else(|| panic!("{stderr}"));
    assert!(line.contains("copy_data"), "{line}");
}

// ── Writes the review found missing, and the fold's dtype ─────────────────────

/// `from_hf(repo, m)` stores a checkpoint into the model's parameter slots,
/// like `model_load`, so nothing may be specialized on the `-w` file's
/// values. It used to be missed: with an identity `w` in the file the matmul
/// was dropped and the program printed `x` while the model held the HF
/// weights. The "hub" is a local hf-hub cache holding V, so no network is
/// used.
#[test]
fn a_parameter_overwritten_by_from_hf_is_not_specialized_on() {
    let dir = tempfile::tempdir().unwrap();
    let weights = dir.path().join("weights.safetensors");
    write_safetensors(&weights, &[("w", &[4, 4], &IDENTITY)]);
    let repo = dir.path().join("hf/hub/models--org--x");
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::create_dir_all(repo.join("snapshots/0123abcd")).unwrap();
    std::fs::write(repo.join("refs/main"), "0123abcd").unwrap();
    write_safetensors(
        &repo.join("snapshots/0123abcd/model.safetensors"),
        &[("w", &[4, 4], &V)],
    );
    let program = format!(
        "model Toy:\n    w: Tensor = ones([4, 4]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w\n\n\
         let m = Toy()\nfrom_hf(\"org/x\", m)\nlet x = arange(1.0, 9.0).reshape([2, 4])\n{}{}",
        show("Y", "m.forward(x)"),
        show("W", "m.w")
    );
    let file = dir.path().join("prog.nsl");
    std::fs::write(&file, program).unwrap();
    let out = nsl()
        .current_dir(dir.path())
        .env("HF_HOME", dir.path().join("hf"))
        .env("HF_HUB_OFFLINE", "1")
        .arg("run")
        .arg(&file)
        .args(["--target", "cpu", "-w"])
        .arg(&weights)
        .output()
        .expect("spawn nsl run");
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(out.status.success(), "{stderr}");
    assert_close(&section(&stdout, "W"), &V, "w after from_hf");
    assert_close(&section(&stdout, "Y"), &X_V, "y after from_hf");
    assert!(!stderr.contains("eliminated"), "{stderr}");
    let line = stderr
        .lines()
        .find(|l| l.contains("not specialized on 'w' (near-identity matmul elimination skipped)"))
        .unwrap_or_else(|| panic!("{stderr}"));
    assert!(line.contains("from_hf"), "{line}");
}

/// A fold of two BF16 weights. The fold used to keep the file's dtype (2
/// bytes per element) and label the constant f32: the product printed as
/// 4.7e32 and the program read past the constant. Now it is computed in f64
/// and given the operands' dtype when the program runs; the result is
/// x @ V @ HEAD, worked out by hand at `X_V_HEAD`.
#[test]
fn a_fold_of_bf16_weights_has_the_right_values() {
    let dir = tempfile::tempdir().unwrap();
    let weights = dir.path().join("weights.safetensors");
    write_safetensors_as(
        &weights,
        "BF16",
        &[("a", &[4, 4], &V), ("b", &[4, 3], &HEAD)],
    );
    let out = run_in(dir.path(), &product_program(""), Some(&weights), &[]);
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(out.status.success(), "{stderr}");
    assert_close(&section(&stdout, "Y"), &X_V_HEAD, "y");
    assert_close(&section(&stdout, "A"), &V, "a");
    assert_close(&section(&stdout, "B"), &HEAD, "b");
    assert!(stderr.contains("constant-folded a @ b"), "{stderr}");
}

/// A sum of two weights of different shapes is a broadcast, which the fold
/// does not implement: it is left to the runtime instead of panicking the
/// compiler ("add shape mismatch"). x @ (V + [1, 2, 3, 4]) by hand (and
/// NumPy): rows [12.5, 26, 24, 49] and [34.5, 62, 72, 129].
#[test]
fn a_broadcast_sum_of_two_weights_is_left_to_the_runtime() {
    let program = format!(
        "model Two:\n    a: Tensor = ones([4, 4]) * 2.0\n    b: Tensor = ones([4]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ (self.a + self.b)\n\n\
         let m = Two()\nlet x = arange(1.0, 9.0).reshape([2, 4])\n{}",
        show("Y", "m.forward(x)")
    );
    let (stdout, stderr) = run_ok(
        &program,
        Some(&[("a", &[4, 4], &V), ("b", &[4], &[1.0, 2.0, 3.0, 4.0])]),
        &[],
    );
    assert_close(
        &section(&stdout, "Y"),
        &[12.5, 26.0, 24.0, 49.0, 34.5, 62.0, 72.0, 129.0],
        "y",
    );
    assert!(!stderr.contains("constant-folded"), "{stderr}");
}

// ── Exact values unless the user accepts an approximation ────────────────────

/// The identity but for [0][0] = 1.0000005 (1.0000004768 in f32): within
/// the default `--dead-weight-threshold` (1e-6) of the identity, and not the
/// identity. (A deviation off the diagonal would not do: dead-weight
/// elimination zeroes a value below the threshold in the compiler's copy,
/// and the fold test below covers that copy.)
const NEAR_IDENTITY: [f64; 16] = [
    1.0000005, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// x = [1e7, 1e7, 1e7, 1e7]: x @ NEAR_IDENTITY is 1e7 everywhere but in
/// column 0, which is 1e7 * 1.0000004768 = 10000004.77 (NumPy: 10000005.0
/// in f32, 10000004.768 in f64).
fn near_program() -> String {
    format!(
        "model Toy:\n    w: Tensor = ones([4, 4]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w\n\n\
         let m = Toy()\nlet x = full([1, 4], 10000000.0)\n{}",
        show("Y", "m.forward(x)")
    )
}

/// By default a rewrite uses the file's exact values: a weight 4.8e-7 off
/// the identity is not taken for it (the default threshold, 1e-6, used to
/// make it so, and column 0 lost its 4.77), and the report says why.
#[test]
fn a_weight_near_the_identity_is_not_dropped_by_default() {
    let (stdout, stderr) = run_ok(
        &near_program(),
        Some(&[("w", &[4, 4], &NEAR_IDENTITY)]),
        &[],
    );
    let y = section(&stdout, "Y");
    assert_eq!(y.len(), 4, "{stdout}");
    assert!(
        (y[0] - 10_000_004.77).abs() < 0.5,
        "column 0 is {} (want 1e7 + 4.77)",
        y[0]
    );
    for (i, v) in y.iter().enumerate().skip(1) {
        assert!((v - 10_000_000.0).abs() < 0.5, "column {i} is {v}");
    }
    assert!(!stderr.contains("eliminated"), "{stderr}");
    let line = stderr
        .lines()
        .find(|l| l.contains("not specialized on 'w' (near-identity matmul elimination skipped)"))
        .unwrap_or_else(|| panic!("{stderr}"));
    assert!(
        line.contains("not the identity") && line.contains("--dead-weight-threshold"),
        "{line}"
    );
}

/// Given explicitly, the threshold is the user's approximation to make: the
/// matmul is dropped, the report says on what basis, and column 0 is 1e7.
#[test]
fn a_weight_near_the_identity_is_dropped_when_the_threshold_is_given() {
    let dir = tempfile::tempdir().unwrap();
    let weights = dir.path().join("weights.safetensors");
    write_safetensors(&weights, &[("w", &[4, 4], &NEAR_IDENTITY)]);
    let (exe, report) = build_with(
        dir.path(),
        &near_program(),
        &weights,
        &["--dead-weight-threshold", "1e-6"],
    );
    let line = report
        .lines()
        .find(|l| l.contains("eliminated near-identity matmul (weight 'w')"))
        .unwrap_or_else(|| panic!("{report}"));
    assert!(
        line.contains("within the --dead-weight-threshold 1e-6 given"),
        "{line}"
    );
    let out = Command::new(&exe).current_dir(dir.path()).output().unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));
    let y = section(&text(&out.stdout), "Y");
    assert!((y[0] - 10_000_000.0).abs() < 0.5, "column 0 is {}", y[0]);
}

/// Dead-weight elimination zeroes values below the threshold in the
/// compiler's copy of the file. A fold resting on that copy would freeze
/// the zeroed value into the program, so by default it is refused.
#[test]
fn a_fold_is_not_made_from_a_copy_dead_weight_elimination_changed() {
    let mut a = V;
    a[1] = 5e-7;
    let (stdout, stderr) = run_ok(
        &product_program(""),
        Some(&[("a", &[4, 4], &a), ("b", &[4, 3], &HEAD)]),
        &[],
    );
    // 5e-7 changes x @ a @ HEAD by at most 8 * 5e-7: within the tolerance.
    assert_close(&section(&stdout, "Y"), &X_V_HEAD, "y");
    assert!(!stderr.contains("constant-folded"), "{stderr}");
    let line = stderr
        .lines()
        .find(|l| l.contains("not specialized on 'a' (constant fold skipped)"))
        .unwrap_or_else(|| panic!("{stderr}"));
    assert!(line.contains("dead-weight elimination zeroed"), "{line}");
}

// ── Adapters ─────────────────────────────────────────────────────────────────

/// `-w` with a file of the base parameters alone, on a model that has
/// adapters: the base weight is loaded and the adapter is left as it was
/// initialised (LoRA's B is zero, so the adapted forward pass equals the
/// base one). The same rule as `model_load` on a base-only file.
#[test]
fn a_base_only_file_loads_into_a_model_with_adapters() {
    let program = format!(
        "from nsl.nn.losses import mse_loss\n\n\
         model Toy:\n    w: Tensor = ones([4, 4]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w\n\n\
         @adapter(type=lora, target=[\"Toy.w\"], rank=2, alpha=2)\nlet m = Toy()\n\
         let x = arange(1.0, 9.0).reshape([2, 4])\nlet y = zeros([2, 4])\n\
         train(model = m, epochs = 1):\n    optimizer: SGD(lr = 0.0)\n    step(batch):\n        let loss = mse_loss(m.forward(x), y)\n\n{}{}",
        show("Y", "m.forward(x)"),
        show("W", "m.w")
    );
    let (stdout, stderr) = run_ok(&program, Some(&[("w", &[4, 4], &V)]), &["--source-ad"]);
    assert_close(&section(&stdout, "Y"), &X_V, "y");
    assert_close(&section(&stdout, "W"), &V, "w");
    assert!(
        stderr.contains("'Toy.w' is loaded from the file's 'w'"),
        "{stderr}"
    );
}

// ── On a GPU ─────────────────────────────────────────────────────────────────

/// The model is constructed on the host, loaded from the file there, and
/// then moved: the device copy holds the file's values, for a weight that is
/// specialized on (the identity) and for one that is not (V), and a train
/// step on the device starts from the file's values.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA device: cargo test -p nsl-cli --features cuda --test weights_flag_binds_model_gate -- --include-ignored"]
fn on_a_gpu_the_moved_model_holds_the_file_s_values() {
    let run_gpu = |source: &str, values: &[f64], extra: &[&str]| {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.safetensors");
        write_safetensors(&weights, &[("w", &[4, 4], values)]);
        let file = dir.path().join("prog.nsl");
        std::fs::write(&file, source).unwrap();
        let out = nsl()
            .current_dir(dir.path())
            .arg("run")
            .args(extra)
            .arg(&file)
            .arg("-w")
            .arg(&weights)
            .output()
            .expect("spawn nsl run");
        let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
        assert!(
            out.status.success(),
            "run failed: {}\n{stdout}\n{stderr}",
            out.status
        );
        (stdout, stderr)
    };
    let forward = format!(
        "model Toy:\n    w: Tensor = ones([4, 4]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w\n\n\
         let m = Toy()\nm.to(cuda)\nlet x = arange(1.0, 9.0).reshape([2, 4]).to(cuda)\n{}{}",
        show("Y", "m.forward(x)"),
        show("W", "m.w")
    );
    let (stdout, stderr) = run_gpu(&forward, &IDENTITY, &[]);
    assert_close(&section(&stdout, "Y"), &X, "y, identity file");
    assert_close(&section(&stdout, "W"), &IDENTITY, "w, identity file");
    assert!(
        stderr.contains("eliminated near-identity matmul"),
        "{stderr}"
    );

    let (stdout, stderr) = run_gpu(&forward, &V, &[]);
    assert_close(&section(&stdout, "Y"), &X_V, "y, V file");
    assert_close(&section(&stdout, "W"), &V, "w, V file");
    assert!(!stderr.contains("eliminated"), "{stderr}");

    // One SGD step on the device, from the file's identity (the same
    // numbers as the host test; f32 on the device, hence the tolerance of
    // `assert_close`).
    let training = format!(
        "from nsl.nn.losses import mse_loss\n\n\
         model Toy:\n    w: Tensor = ones([4, 4]) * 2.0\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w\n\n\
         let m = Toy()\nm.to(cuda)\nlet x = (arange(1.0, 9.0).reshape([2, 4]) * 0.1).to(cuda)\n\
         let y = (arange(0.0, 8.0).reshape([2, 4]) * 0.25 - full([2, 4], 1.0)).to(cuda)\n\
         train(model = m, epochs = 1):\n    optimizer: SGD(lr = 0.5)\n    step(batch):\n        let loss = mse_loss(m.forward(x), y)\n\n{}{}",
        show("W1", "m.w"),
        show("Y1", "m.forward(x)")
    );
    let (stdout, stderr) = run_gpu(&training, &IDENTITY, &["--source-ad"]);
    assert_close(
        &section(&stdout, "W1"),
        &W_AFTER_ONE_STEP,
        "w after the step",
    );
    assert_close(
        &section(&stdout, "Y1"),
        &Y_AFTER_ONE_STEP,
        "y after the step",
    );
    assert!(stderr.contains("not specialized on 'w'"), "{stderr}");
}
