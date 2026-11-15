//! Source AD differentiates through a user-defined function by inlining it.
//!
//! Under `--source-ad` a call to a function the compiler has no built-in
//! rule for -- any function the user wrote -- stopped the compile-time
//! extraction: a `grad` or `train` block warned about an "unrecognized FFI
//! callee" and fell back to the tape, and a `distill` block (which refuses
//! the tape) failed to build. A loss written as a function is the ordinary
//! way to write one, so "write the loss in NSL and source AD does the rest"
//! did not hold.
//!
//! The function body is now extracted into the caller's graph. These tests
//! check the result against values worked out by hand (written in f64
//! below), and check that the block really was differentiated at compile
//! time: a tape fallback would produce the same numbers without touching
//! the code under test.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// Writes `files` (name, source) into a fresh directory, runs the first one
/// on the CPU with `--source-ad`, and returns (printed numbers, stderr).
fn run_files(files: &[(&str, &str)]) -> (Vec<f64>, String) {
    let dir = tempfile::tempdir().unwrap();
    for (name, source) in files {
        std::fs::write(dir.path().join(name), source).unwrap();
    }
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(dir.path().join(files[0].0))
        .args(["--target", "cpu", "--source-ad"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run failed:\n{stderr}");
    let numbers = String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect();
    (numbers, stderr)
}

fn run(source: &str) -> (Vec<f64>, String) {
    run_files(&[("prog.nsl", source)])
}

fn assert_compile_time(stderr: &str) {
    assert!(
        !stderr.contains("falls back to tape") && !stderr.contains("falling back to tape"),
        "the block fell back to tape AD, so inlining was not exercised:\n{stderr}"
    );
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: printed {got:?}, expected {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 2e-5, "{what}: element {i} is {g}, expected {w} (all: {got:?})");
    }
}

fn sigmoid(u: f64) -> f64 {
    1.0 / (1.0 + (-u).exp())
}

/// `c(z) = -log(0.1 * sum(z))` at four elements of 1.5: `u = 0.6`, the loss
/// is `-ln 0.6` and every element's gradient is `-0.1 / 0.6`.
#[test]
fn a_function_called_from_a_grad_block() {
    let (got, stderr) = run(
        r#"
let zw = full([2, 2], 1.5)
fn c(z: Tensor) -> Tensor:
    let u = 0.1 * sum(z)
    return neg(log(u))
let (l, g) = grad(zw):
    c(zw)
print(l)
print(g)
"#,
    );
    assert_compile_time(&stderr);
    let d = -0.1 / 0.6;
    assert_close(&got, &[-(0.6f64.ln()), d, d, d, d], "c(zw)");
}

/// A DPO-shaped loss, one function calling another:
/// `u = beta * (sum(w * xa) - sum(w * xb))`, `L = -log(sigmoid(u))`, with
/// `w = 0.2`, `xa = [0, 1, 2]`, `xb = 0.5`, `beta = 2`. So `u = 0.6` and
/// `dL/dw = -(1 - sigmoid(u)) * beta * (xa - xb)`.
#[test]
fn functions_calling_functions_with_a_scalar_parameter() {
    let (got, stderr) = run(
        r#"
let w = full([3], 0.2)
let xa = arange(0.0, 3.0)
let xb = full([3], 0.5)
fn score(w: Tensor, x: Tensor) -> Tensor:
    return sum(w * x)
fn preference_loss(w: Tensor, chosen: Tensor, rejected: Tensor, beta: float) -> Tensor:
    let u = beta * (score(w, chosen) - score(w, rejected))
    return neg(log(sigmoid(u)))
let (l, g) = grad(w):
    preference_loss(w, xa, xb, 2.0)
print(l)
print(g)
"#,
    );
    assert_compile_time(&stderr);
    let u = 2.0 * (0.2 * 3.0 - 0.2 * 1.5);
    let k = -(1.0 - sigmoid(u)) * 2.0;
    assert_close(&got, &[-(sigmoid(u).ln()), k * -0.5, k * 0.5, k * 1.5], "preference_loss");
}

/// The same loss in a train block, as a function of a model parameter:
/// three SGD steps at lr = 1 on `L = -log(sigmoid(0.1 * sum(w)))`, `w`
/// four elements of 1.5. Each step moves every element by
/// `(1 - sigmoid(u)) * 0.1`.
#[test]
fn a_loss_function_in_a_train_block() {
    let (got, stderr) = run(
        r#"
model M:
    w: Tensor = full([2, 2], 1.5)

fn my_loss(z: Tensor) -> Tensor:
    let u = 0.1 * sum(z)
    return neg(log(sigmoid(u)))

let m = M()
train(model = m, epochs = 3):
    optimizer: SGD(lr = 1.0)
    step(batch):
        let loss = my_loss(m.w)
    callbacks:
        on_step(step, loss):
            print(loss)
print(m.w)
"#,
    );
    assert_compile_time(&stderr);
    let mut w = 1.5f64;
    let mut want = Vec::new();
    for _ in 0..3 {
        let u = 0.1 * 4.0 * w;
        want.push(-(sigmoid(u).ln()));
        w += (1.0 - sigmoid(u)) * 0.1;
    }
    want.extend([w; 4]);
    assert_close(&got, &want, "train with my_loss");
}

/// A function imported from another module of the program.
#[test]
fn a_function_imported_from_another_module() {
    let (got, stderr) = run_files(&[
        (
            "prog.nsl",
            "from lossmod import half_square\n\
             let w = arange(1.0, 4.0)\n\
             let (l, g) = grad(w):\n\
             \x20   half_square(w, 3.0)\n\
             print(l)\n\
             print(g)\n",
        ),
        (
            "lossmod.nsl",
            "fn half_square(z: Tensor, scale: float) -> Tensor:\n\
             \x20   let s = z * z\n\
             \x20   return sum(s) * scale * 0.5\n",
        ),
    ]);
    assert_compile_time(&stderr);
    // 0.5 * 3 * (1 + 4 + 9) = 21; d/dz = 3 * z.
    assert_close(&got, &[21.0, 3.0, 6.0, 9.0], "half_square");
}

/// A function of this module that shares a builtin's name is the user's
/// function. Source AD used to match the name against its own table and
/// differentiate the builtin `relu` instead: for negative inputs that is a
/// loss and a gradient of zero, where the program computes `2 * x`.
#[test]
fn a_user_function_named_like_a_builtin_is_the_users() {
    let (got, stderr) = run(
        r#"
let w = full([3], -1.5)
fn relu(x: Tensor) -> Tensor:
    return x * 2.0
let (l, g) = grad(w):
    sum(relu(w))
print(l)
print(g)
"#,
    );
    assert_compile_time(&stderr);
    assert_close(&got, &[-9.0, 2.0, 2.0, 2.0], "user relu");
}

/// A function that returns from the middle of its body is not inlined: the
/// extraction has no way to say "and stop here", and would take the last
/// `return` for the result. The block falls back to the tape, says so, and
/// the answer is the taken branch's: `sum(z * z)`, gradient `2 z`.
#[test]
fn a_function_with_an_early_return_is_left_to_the_tape() {
    let (got, stderr) = run(
        r#"
let w = arange(1.0, 4.0)
fn pick(z: Tensor, squared: bool) -> Tensor:
    if squared:
        return sum(z * z)
    return sum(z)
let (l, g) = grad(w):
    pick(w, true)
print(l)
print(g)
"#,
    );
    assert!(
        stderr.contains("'pick' cannot be inlined") && stderr.contains("returns from somewhere other than the end"),
        "the reason for the fallback must be stated:\n{stderr}"
    );
    assert_close(&got, &[14.0, 2.0, 4.0, 6.0], "pick(w, true)");
}

/// A recursive function cannot be inlined either.
#[test]
fn a_recursive_function_is_left_to_the_tape() {
    let (got, stderr) = run(
        r#"
let w = arange(1.0, 4.0)
fn power(z: Tensor, n: int) -> Tensor:
    if n <= 1:
        return z
    else:
        return z * power(z, n - 1)
let (l, g) = grad(w):
    sum(power(w, 3))
print(l)
print(g)
"#,
    );
    assert!(stderr.contains("'power' cannot be inlined"), "the reason for the fallback must be stated:\n{stderr}");
    // sum(z^3) = 1 + 8 + 27; d/dz = 3 z^2.
    assert_close(&got, &[36.0, 3.0, 12.0, 27.0], "power(w, 3)");
}

// ---------------------------------------------------------------------------
// Writes the extraction does not implement
// ---------------------------------------------------------------------------

/// A write into a tensor inside a function: `y[0, 0] = 0.0` after
/// `y = x * 2` with `x = [[1, 2], [3, 4]]` leaves `y = [[0, 4], [6, 8]]`,
/// whose sum is 18. Inlined, the write used to vanish and the loss came out
/// as 20. The function is now left to the tape, which performs the write and
/// records it: the gradient is PyTorch's `[[0, 2], [2, 2]]` (the tape gave
/// `[[2, 2], [2, 2]]` before it recorded writes; tape_element_write_gate.rs).
#[test]
fn a_function_that_writes_into_a_tensor_is_left_to_the_tape() {
    let (got, stderr) = run_files(&[(
        "prog.nsl",
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
fn f(x: Tensor) -> Tensor:
    let y = x * 2.0
    y[0, 0] = 0.0
    return sum(y)
let (l, g) = grad(w):
    f(w)
print(l)
print(g)
"#,
    )]);
    assert!(
        stderr.contains("'f' cannot be inlined") && stderr.contains("writes through a subscript or a field"),
        "the reason for the fallback must be stated:\n{stderr}"
    );
    assert_close(&got, &[18.0, 0.0, 2.0, 2.0, 2.0], "loss and gradient with the element write");
}

/// A write into a PARAMETER changes the caller's tensor. With
/// `x[0, 0] = 10` the loss is `100 + 4 + 9 + 16 = 129`, and the caller's
/// `w` reads `[[10, 2], [3, 4]]` afterwards. Inlined, both used to be lost:
/// the loss was 30 and `w` was unchanged.
#[test]
fn a_write_into_a_parameter_reaches_the_caller() {
    let (got, _) = run_files(&[(
        "prog.nsl",
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
fn f(x: Tensor) -> Tensor:
    x[0, 0] = 10.0
    return sum(x * x)
let (l, g) = grad(w):
    f(w)
print(l)
print(w)
"#,
    )]);
    assert_close(&got, &[129.0, 10.0, 2.0, 3.0, 4.0], "loss, then the caller's w");
}

/// The same element write in a grad block's own body: it used to be
/// dropped there too (loss 20). The block now runs on the tape, with
/// PyTorch's gradient `[[0, 2], [2, 2]]`.
#[test]
fn an_element_write_in_a_block_body_is_left_to_the_tape() {
    let (got, stderr) = run_files(&[(
        "prog.nsl",
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
let (l, g) = grad(w):
    let y = w * 2.0
    y[0, 0] = 0.0
    sum(y)
print(l)
print(g)
"#,
    )]);
    assert!(stderr.contains("assignment through a subscript or a field"), "{stderr}");
    assert_close(&got, &[18.0, 0.0, 2.0, 2.0, 2.0], "loss and gradient with the element write");
}

// ---------------------------------------------------------------------------
// Names resolve where the inlined code was written
// ---------------------------------------------------------------------------

const MYLIB: &str = r#"
fn sq(x: Tensor) -> Tensor:
    return sum(x * x)

fn half_sq(x: Tensor) -> Tensor:
    return sq(x) * 0.5

fn relu(x: Tensor) -> Tensor:
    return x * 2.0

fn act(x: Tensor) -> Tensor:
    return sum(relu(x))

fn helper(x: Tensor) -> Tensor:
    return x * 3.0

model Scaler:
    w: Tensor = full([3], 1.0)

    fn forward(self, x: Tensor) -> Tensor:
        return sum(helper(x * self.w))
"#;

/// An imported function calls its own module's `sq`, not the `sq` the
/// importing module declares. `0.5 * sum(w * w)` at `w = [1, 2, 3]` is 7
/// with gradient `w`; the captured `sq` (`sum(x)`) gave 3 and 0.5.
#[test]
fn an_imported_function_calls_its_own_modules_function() {
    let (got, stderr) = run_files(&[
        (
            "prog.nsl",
            "from mylib import half_sq\n\
             fn sq(x: Tensor) -> Tensor:\n\
             \x20   return sum(x)\n\
             let w = arange(1.0, 4.0)\n\
             let (l, g) = grad(w):\n\
             \x20   half_sq(w)\n\
             print(l)\n\
             print(g)\n",
        ),
        ("mylib.nsl", MYLIB),
    ]);
    assert_compile_time(&stderr);
    assert_close(&got, &[7.0, 1.0, 2.0, 3.0], "half_sq");
}

/// An imported module's own `fn relu` is the function its code calls, not
/// the builtin. `sum(2 * w)` at `w = -1.5` (three elements) is -9 with
/// gradient 2; the builtin relu gave 0 and 0.
#[test]
fn an_imported_modules_function_named_like_a_builtin_is_its_own() {
    let (got, stderr) = run_files(&[
        (
            "prog.nsl",
            "from mylib import act\n\
             let w = full([3], -1.5)\n\
             let (l, g) = grad(w):\n\
             \x20   act(w)\n\
             print(l)\n\
             print(g)\n",
        ),
        ("mylib.nsl", MYLIB),
    ]);
    assert_compile_time(&stderr);
    assert_close(&got, &[-9.0, 2.0, 2.0, 2.0], "act");
}

/// A method of an imported model calls its own module's `helper`, not the
/// importing module's. `sum(3 * w * 1)` at `w = [1, 2, 3]` is 18 with
/// gradient 3; the captured `helper` (`x * 100`) gave 600 and 100.
#[test]
fn a_method_of_an_imported_model_calls_its_own_modules_function() {
    let (got, stderr) = run_files(&[
        (
            "prog.nsl",
            "from mylib import Scaler\n\
             fn helper(x: Tensor) -> Tensor:\n\
             \x20   return x * 100.0\n\
             let k = Scaler()\n\
             let w = arange(1.0, 4.0)\n\
             let (l, g) = grad(w):\n\
             \x20   k.forward(w)\n\
             print(l)\n\
             print(g)\n",
        ),
        ("mylib.nsl", MYLIB),
    ]);
    assert_compile_time(&stderr);
    assert_close(&got, &[18.0, 3.0, 3.0, 3.0], "Scaler.forward");
}

/// Two models with a `scale` method; a function takes the second one in a
/// parameter that happens to be called `m`, like the first. Inside the
/// inlined body `m.scale(x)` and `m.w` used to resolve to the model NAMED
/// `m`, whatever was passed: 12 and gradient 2 where the program computes
/// `sum(w * 5) = 30` and 5. A model argument now leaves the function to
/// the tape, and inside a function no name is resolved as a model.
#[test]
fn a_parameter_named_like_a_model_is_the_argument_passed() {
    let models = "model M:\n\
                  \x20   w: Tensor = full([3], 2.0)\n\
                  \n\
                  \x20   fn scale(self, x: Tensor) -> Tensor:\n\
                  \x20       return x * self.w\n\
                  \n\
                  model N:\n\
                  \x20   w: Tensor = full([3], 5.0)\n\
                  \n\
                  \x20   fn scale(self, x: Tensor) -> Tensor:\n\
                  \x20       return x * self.w\n\
                  \n\
                  let m = M()\n\
                  let n = N()\n\
                  let w = arange(1.0, 4.0)\n";
    for (what, body) in [("method call", "sum(m.scale(x))"), ("field read", "sum(x * m.w)")] {
        let source = format!(
            "{models}fn apply(m: N, x: Tensor) -> Tensor:\n\
             \x20   return {body}\n\
             let (l, g) = grad(w):\n\
             \x20   apply(n, w)\n\
             print(l)\n\
             print(g)\n"
        );
        let (got, stderr) = run(&source);
        assert!(stderr.contains("'apply' cannot be inlined") && stderr.contains("is a model instance"), "{what}:\n{stderr}");
        assert_close(&got, &[30.0, 5.0, 5.0, 5.0], what);
    }
}

