//! Where an `@adapter` attaches, and what it is sized from.
//!
//! An adapter used to work only on a square weight written `x @ self.w` in
//! a model defined in the file that carried the decorator:
//!
//! - the widths of `lora_A` and `lora_B` came from the weight's shape
//!   literal read as `[out, in]`, while `x @ self.w` needs `[in, out]`, so a
//!   `[4, 6]` weight got a `[6, 2]` A and the first forward pass aborted on
//!   "matmul inner dimension mismatch (3x4 @ 6x2)";
//! - `x @ self.w.transpose(0, 1)`, the standard library's `Linear`, was not
//!   matched: A and B were allocated and handed to the optimizer and never
//!   entered the forward pass;
//! - a target on an imported model (`Linear.w`) resolved to nothing, with a
//!   warning;
//! - the class in `Class.field` was ignored: with models `Aaa` and `Zzz`
//!   both owning `w`, `Zzz.w` put the adapter on `Aaa`.
//!
//! Each training fixture prints its random initial weight and A, and the
//! trained tensors are held to an f64 reference trajectory computed here
//! from those by central differences.
//!
//! The last tests are about keeping a trained adapter: `model_save` wrote
//! only the model's own fields, so a LoRA fine-tune of a frozen base had no
//! persistent output at all.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run(source: &str, source_ad: bool) -> Run {
    let tmp = tempfile::TempDir::new().unwrap();
    run_in(tmp.path(), source, source_ad)
}

/// Run `source` with `dir` as the working directory, so that the files it
/// saves and loads by relative path stay there for the test to read.
fn run_in(dir: &Path, source: &str, source_ad: bool) -> Run {
    let root = repo_root();
    let path = dir.join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run");
    if source_ad {
        cmd.arg("--source-ad");
    }
    let out = cmd
        .arg(&path)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env_remove("NSL_WRGA_FUSED_CUDA")
        .output()
        .expect("spawn nsl run");
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn run_ok(source: &str, source_ad: bool) -> String {
    let r = run(source, source_ad);
    assert!(r.ok, "the program failed (source_ad={source_ad}):\n{}", r.stderr);
    r.stdout
}

/// The numbers printed between the lines `<tag>_BEGIN` and `<tag>_END`.
fn section(stdout: &str, tag: &str) -> Vec<f64> {
    let (begin, end) = (format!("{tag}_BEGIN"), format!("{tag}_END"));
    let mut lines = stdout.lines().skip_while(|l| l.trim() != begin);
    assert!(lines.next().is_some(), "no line {begin} in:\n{stdout}");
    let inner: Vec<&str> = lines.take_while(|l| l.trim() != end).collect();
    assert!(stdout.lines().any(|l| l.trim() == end), "no line {end} in:\n{stdout}");
    inner
        .join(" ")
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

const ROWS: usize = 3;
const RANK: usize = 2;
const LR: f64 = 0.05;
const STEPS: usize = 3;

// The fixtures' data, as the NSL source builds it:
//   x = arange(ROWS * n_in).reshape([ROWS, n_in]) * 0.1 + 0.1
//   y = arange(ROWS * n_out).reshape([ROWS, n_out]) * 0.2 - 1.0
fn x_data(n_in: usize) -> Vec<f64> {
    (0..ROWS * n_in).map(|i| i as f64 * 0.1 + 0.1).collect()
}
fn y_data(n_out: usize) -> Vec<f64> {
    (0..ROWS * n_out).map(|i| i as f64 * 0.2 - 1.0).collect()
}

/// How the model's forward uses its weight.
#[derive(Clone, Copy)]
enum Form {
    /// `x @ w`, with `w` stored `[n_in, n_out]`.
    Direct,
    /// `x @ w.transpose(0, 1)`, with `w` stored `[n_out, n_in]`.
    Transposed,
}

/// `x @ W + ((x @ A) @ B) * scale (+ bias)`, written out element by element:
/// `a` is `[n_in, RANK]`, `b` is `[RANK, n_out]`, `w` as `form` stores it.
#[allow(clippy::too_many_arguments)]
fn lora_forward(
    x: &[f64],
    w: &[f64],
    a: &[f64],
    b: &[f64],
    bias: Option<&[f64]>,
    scale: f64,
    (n_in, n_out): (usize, usize),
    form: Form,
) -> Vec<f64> {
    let mut out = vec![0.0; ROWS * n_out];
    for r in 0..ROWS {
        let xa: Vec<f64> =
            (0..RANK).map(|q| (0..n_in).map(|k| x[n_in * r + k] * a[RANK * k + q]).sum()).collect();
        for j in 0..n_out {
            let base: f64 = (0..n_in)
                .map(|k| {
                    let w_kj = match form {
                        Form::Direct => w[n_out * k + j],
                        Form::Transposed => w[n_in * j + k],
                    };
                    x[n_in * r + k] * w_kj
                })
                .sum();
            let delta: f64 = (0..RANK).map(|q| xa[q] * b[n_out * q + j]).sum();
            out[n_out * r + j] = base + delta * scale + bias.map_or(0.0, |v| v[j]);
        }
    }
    out
}

/// `mse_loss(pred, y)`: the mean squared error over every cell.
fn mse(pred: &[f64], y: &[f64]) -> f64 {
    pred.iter().zip(y).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / pred.len() as f64
}

/// SGD on `loss` from `p`, each gradient by central differences.
fn sgd_reference(loss: impl Fn(&[f64]) -> f64, mut p: Vec<f64>) -> Vec<f64> {
    for _ in 0..STEPS {
        let grad: Vec<f64> = (0..p.len())
            .map(|i| {
                let (mut up, mut down) = (p.clone(), p.clone());
                up[i] += 1e-6;
                down[i] -= 1e-6;
                (loss(&up) - loss(&down)) / 2e-6
            })
            .collect();
        for (pi, gi) in p.iter_mut().zip(&grad) {
            *pi -= LR * gi;
        }
    }
    p
}

fn assert_close(label: &str, got: &[f64], want: &[f64], moved_from: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: expected {} values, got {}", want.len(), got.len());
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let travel = want.iter().zip(moved_from).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max |Δ| from the f64 reference {worst:.2e}; the reference moves {travel:.3}");
    // 500x the per-element tolerance below.
    assert!(travel > 5e-3, "{label}: the reference barely moves ({travel}); the check would be vacuous");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() < 1e-5, "{label}: [{i}] = {a}, f64 reference {b} (|Δ|={:.2e})", (a - b).abs());
    }
}

/// A one-weight model with a LoRA adapter, trained for `STEPS` steps of SGD;
/// prints the initial w and A and the trained w, A and B.
fn toy_program(w_shape: &str, forward: &str, (n_in, n_out): (usize, usize)) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = randn({w_shape}) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return {forward}

@adapter(type=lora, target=["Toy.w"], rank={RANK}, alpha=4)
let m = Toy()
let w0: Tensor = m.w
print("W0_BEGIN")
print(w0)
print("W0_END")
let a0: Tensor = m.lora_A_Toy_w__lora
print("A0_BEGIN")
print(a0)
print("A0_END")
let b0: Tensor = m.lora_B_Toy_w__lora
print("B0_BEGIN")
print(b0)
print("B0_END")

let x = (arange({nx}).reshape([{ROWS}, {n_in}])) * 0.1 + full([{ROWS}, {n_in}], 0.1)
let y = (arange({ny}).reshape([{ROWS}, {n_out}])) * 0.2 - full([{ROWS}, {n_out}], 1.0)

train(model = m, epochs = {STEPS}):
    optimizer: SGD(lr = {LR})
    step(batch):
        let loss = mse_loss(m.forward(x), y)

let w: Tensor = m.w
print("W_BEGIN")
print(w)
print("W_END")
let a: Tensor = m.lora_A_Toy_w__lora
print("A_BEGIN")
print(a)
print("A_END")
let b: Tensor = m.lora_B_Toy_w__lora
print("B_BEGIN")
print(b)
print("B_END")
"#,
        nx = ROWS * n_in,
        ny = ROWS * n_out,
    )
}

/// Run `toy_program` in both AD modes and hold w, A and B to the reference.
fn check_toy(w_shape: &str, forward: &str, dims: (usize, usize), form: Form) {
    let (n_in, n_out) = dims;
    let scale = 4.0 / RANK as f64;
    let src = toy_program(w_shape, forward, dims);
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = run_ok(&src, source_ad);
        let w0 = section(&out, "W0");
        let a0 = section(&out, "A0");
        let b0 = section(&out, "B0");
        assert_eq!(w0.len(), n_in * n_out, "{mode}: the weight has {n_in}x{n_out} cells");
        assert_eq!(a0.len(), n_in * RANK, "{mode}: A is [in, rank] = [{n_in}, {RANK}]");
        assert_eq!(b0.len(), RANK * n_out, "{mode}: B is [rank, out] = [{RANK}, {n_out}]");
        assert!(b0.iter().all(|v| *v == 0.0), "{mode}: B starts at zero");
        let (nw, na) = (w0.len(), a0.len());
        let init: Vec<f64> = w0.into_iter().chain(a0).chain(b0).collect();
        let (x, y) = (x_data(n_in), y_data(n_out));
        let loss = |p: &[f64]| {
            mse(
                &lora_forward(&x, &p[..nw], &p[nw..nw + na], &p[nw + na..], None, scale, dims, form),
                &y,
            )
        };
        let want = sgd_reference(loss, init.clone());
        for (name, range) in [("W", 0..nw), ("A", nw..nw + na), ("B", nw + na..init.len())] {
            let got = section(&out, name);
            assert_close(&format!("{mode} {name}"), &got, &want[range.clone()], &init[range]);
        }
    }
}

#[test]
fn a_lora_adapter_on_a_non_square_weight_trains_like_its_gradient() {
    // `x @ self.w` with w `[4, 6]`: 4 in, 6 out.
    check_toy("[4, 6]", "x @ self.w", (4, 6), Form::Direct);
}

#[test]
fn a_lora_adapter_on_a_transposed_weight_trains_like_its_gradient() {
    // `x @ self.w.transpose(0, 1)` with w `[6, 4]`: still 4 in, 6 out.
    check_toy("[6, 4]", "x @ self.w.transpose(0, 1)", (4, 6), Form::Transposed);
}

#[test]
fn a_lora_adapter_on_the_standard_library_linear_trains_like_its_gradient() {
    // `Linear(4, 6)` is defined in another module, takes its widths as
    // constructor arguments, stores w as `[out, in]` and computes
    // `bias_add(x @ self.w.transpose(0, 1), self.b)`.
    let (n_in, n_out) = (4, 6);
    let scale = 4.0 / RANK as f64;
    let src = format!(
        r#"from nsl.nn.losses import mse_loss
from nsl.nn.layers import Linear

model Net:
    fc: Linear = Linear({n_in}, {n_out})

    fn forward(self, x: Tensor) -> Tensor:
        return self.fc.forward(x)

@adapter(type=lora, target=["Linear.w"], rank={RANK}, alpha=4)
let m = Net()
let w0: Tensor = m.fc.w
print("W0_BEGIN")
print(w0)
print("W0_END")
let bias0: Tensor = m.fc.b
print("BIAS0_BEGIN")
print(bias0)
print("BIAS0_END")
let a0: Tensor = m.fc.lora_A_Linear_w__lora
print("A0_BEGIN")
print(a0)
print("A0_END")

let x = (arange({nx}).reshape([{ROWS}, {n_in}])) * 0.1 + full([{ROWS}, {n_in}], 0.1)
let y = (arange({ny}).reshape([{ROWS}, {n_out}])) * 0.2 - full([{ROWS}, {n_out}], 1.0)

train(model = m, epochs = {STEPS}):
    optimizer: SGD(lr = {LR})
    step(batch):
        let loss = mse_loss(m.forward(x), y)

let w: Tensor = m.fc.w
print("W_BEGIN")
print(w)
print("W_END")
let bias: Tensor = m.fc.b
print("BIAS_BEGIN")
print(bias)
print("BIAS_END")
let a: Tensor = m.fc.lora_A_Linear_w__lora
print("A_BEGIN")
print(a)
print("A_END")
let b: Tensor = m.fc.lora_B_Linear_w__lora
print("B_BEGIN")
print(b)
print("B_END")
"#,
        nx = ROWS * n_in,
        ny = ROWS * n_out,
    );
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = run_ok(&src, source_ad);
        let w0 = section(&out, "W0");
        let bias0 = section(&out, "BIAS0");
        let a0 = section(&out, "A0");
        assert_eq!(w0.len(), n_out * n_in, "{mode}: Linear's w is [out, in]");
        assert_eq!(bias0.len(), n_out);
        assert_eq!(a0.len(), n_in * RANK, "{mode}: A is [in, rank] = [{n_in}, {RANK}]");
        // w, bias, A, then B (zero).
        let (nw, nb, na) = (w0.len(), bias0.len(), a0.len());
        let init: Vec<f64> = w0
            .into_iter()
            .chain(bias0)
            .chain(a0)
            .chain(std::iter::repeat_n(0.0, RANK * n_out))
            .collect();
        let (x, y) = (x_data(n_in), y_data(n_out));
        let loss = |p: &[f64]| {
            let (w, rest) = p.split_at(nw);
            let (bias, rest) = rest.split_at(nb);
            let (a, b) = rest.split_at(na);
            mse(
                &lora_forward(&x, w, a, b, Some(bias), scale, (n_in, n_out), Form::Transposed),
                &y,
            )
        };
        let want = sgd_reference(loss, init.clone());
        let parts = [
            ("W", 0..nw),
            ("BIAS", nw..nw + nb),
            ("A", nw + nb..nw + nb + na),
            ("B", nw + nb + na..init.len()),
        ];
        for (name, range) in parts {
            let got = section(&out, name);
            assert_close(&format!("{mode} {name}"), &got, &want[range.clone()], &init[range]);
        }
    }
}

/// Two models that both own a field `w`; the adapter names one of them.
fn two_class_program(target: &str, read: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Aaa:
    w: Tensor = randn([4, 4]) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

model Zzz:
    w: Tensor = randn([4, 4]) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

model Net:
    a: Aaa = Aaa()
    z: Zzz = Zzz()

    fn forward(self, x: Tensor) -> Tensor:
        return self.z.forward(self.a.forward(x))

@freeze
@adapter(type=lora, target=["{target}"], rank=2, alpha=4)
let m = Net()
let x = (arange(12).reshape([3, 4])) * 0.1 + full([3, 4], 0.1)
let y = (arange(12).reshape([3, 4])) * 0.2 - full([3, 4], 1.0)
train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.1)
    step(batch):
        let loss = mse_loss(m.forward(x), y)
let b: Tensor = {read}
print("B_BEGIN")
print(b)
print("B_END")
"#
    )
}

#[test]
fn the_class_in_a_target_decides_which_model_gets_the_adapter() {
    for source_ad in [false, true] {
        // The adapter is on Zzz: its B exists there and has trained.
        let out = run_ok(&two_class_program("Zzz.w", "m.z.lora_B_Zzz_w__lora"), source_ad);
        let b = section(&out, "B");
        assert_eq!(b.len(), 8, "B on Zzz is [2, 4]");
        let largest = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(largest > 1e-3, "B on Zzz starts at zero and should have trained; got {b:?}");

        // ...and not on Aaa, which used to receive it as the alphabetically
        // first class with a field `w`.
        let r = run(&two_class_program("Zzz.w", "m.a.lora_B_Zzz_w__lora"), source_ad);
        assert!(!r.ok, "Aaa must not carry the adapter that targets Zzz:\n{}", r.stdout);
        assert!(
            r.stderr.contains("lora_B_Zzz_w__lora") && r.stderr.contains("Aaa"),
            "the error should name the tensor and the model that lacks it:\n{}",
            r.stderr
        );
    }
}

/// A target that attaches nowhere is a compile error naming it.
fn assert_refused(source: &str, needles: &[&str]) {
    for source_ad in [false, true] {
        let r = run(source, source_ad);
        assert!(!r.ok, "the build should fail (source_ad={source_ad}); stdout:\n{}", r.stdout);
        for needle in needles {
            assert!(
                r.stderr.contains(needle),
                "expected `{needle}` in the error (source_ad={source_ad}):\n{}",
                r.stderr
            );
        }
        assert!(
            !r.stdout.contains("RAN"),
            "the program must not run when its adapter attaches nowhere:\n{}",
            r.stdout
        );
    }
}

fn program_with(model_body: &str, decorators: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = randn([4, 4]) * 0.3
{model_body}

{decorators}
let m = Toy()
print("RAN")
"#
    )
}

const DIRECT_FORWARD: &str = "
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w";

#[test]
fn a_target_naming_no_model_is_refused() {
    // The part before the dot is the class. `m` is the variable.
    assert_refused(
        &program_with(DIRECT_FORWARD, "@adapter(type=lora, target=[\"m.w\"], rank=2)"),
        &["@adapter target 'm.w'", "no model named 'm'", "Toy"],
    );
}

#[test]
fn a_target_naming_no_field_is_refused() {
    assert_refused(
        &program_with(DIRECT_FORWARD, "@adapter(type=lora, target=[\"Toy.weight\"], rank=2)"),
        &["@adapter target 'Toy.weight'", "has no field 'weight'", "w"],
    );
}

#[test]
fn a_target_without_class_is_refused() {
    assert_refused(
        &program_with(DIRECT_FORWARD, "@adapter(type=lora, target=[\"w\"], rank=2)"),
        &["@adapter target 'w'", "Class.field"],
    );
}

/// A weight takes one adapter. A second `@adapter` on the same `Class.field`,
/// or the same target listed twice in one, used to be accepted: both sets of
/// tensors got the same names (`lora_A_Toy_w__lora` twice in a saved
/// checkpoint) and only the first entered the forward pass.
#[test]
fn a_weight_named_by_two_adapters_is_refused() {
    for decorators in [
        "@adapter(type=lora, target=[\"Toy.w\"], rank=2)\n@adapter(type=lora, target=[\"Toy.w\"], rank=4)",
        "@adapter(type=lora, target=[\"Toy.w\", \"Toy.w\"], rank=2)",
        "@adapter(type=lora, target=[\"Toy.w\"], rank=2)\n@adapter(type=ia3, target=[\"Toy.w\"])",
    ] {
        assert_refused(
            &program_with(DIRECT_FORWARD, decorators),
            &["@adapter target 'Toy.w'", "named more than once", "a weight takes one adapter"],
        );
    }
}

/// The control: two adapters on two different weights are fine.
#[test]
fn two_adapters_on_two_weights_are_accepted() {
    let body = "    v: Tensor = randn([4, 4]) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return (x @ self.w) @ self.v";
    let r = run(
        &program_with(
            body,
            "@adapter(type=lora, target=[\"Toy.w\"], rank=2)\n@adapter(type=lora, target=[\"Toy.v\"], rank=2)",
        ),
        false,
    );
    assert!(r.ok && r.stdout.contains("RAN"), "{}", r.stderr);
}

#[test]
fn a_weight_no_matmul_uses_is_refused() {
    // `self.w` is only ever added, so there is no matmul for the adapter to
    // attach to: A and B would train against nothing.
    let body = "
    fn forward(self, x: Tensor) -> Tensor:
        return x + self.w";
    assert_refused(
        &program_with(body, "@adapter(type=lora, target=[\"Toy.w\"], rank=2)"),
        &["@adapter target 'Toy.w'", "x @ self.w", "without ever entering the forward pass"],
    );
}

#[test]
fn a_weight_used_in_both_forms_is_refused() {
    let body = "
    fn forward(self, x: Tensor) -> Tensor:
        return (x @ self.w) @ self.w.transpose(0, 1)";
    assert_refused(
        &program_with(body, "@adapter(type=lora, target=[\"Toy.w\"], rank=2)"),
        &["@adapter target 'Toy.w'", "both"],
    );
}

#[test]
fn a_target_naming_no_field_of_an_imported_model_is_refused() {
    let src = r#"from nsl.nn.layers import Linear

model Net:
    fc: Linear = Linear(4, 4)

    fn forward(self, x: Tensor) -> Tensor:
        return self.fc.forward(x)

@adapter(type=lora, target=["Linear.weight"], rank=2)
let m = Net()
print("RAN")
"#;
    assert_refused(src, &["@adapter target 'Linear.weight'", "has no field 'weight'"]);
}

// ── model_save / model_load ────────────────────────────────────────────────

/// One tensor of an `.nslm` file: its name, shape and 32-bit float data.
struct SavedTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f64>,
}

/// Read an `.nslm` file as `model_save` documents it: magic, version, header
/// length, a JSON table of entries, then the data section at the next
/// multiple of 64 bytes.
fn read_nslm(path: &Path) -> Vec<SavedTensor> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let header_end = 16 + header_len;
    let header: serde_json::Value = serde_json::from_slice(&bytes[16..header_end]).unwrap();
    let data_start = header_end + (64 - header_end % 64) % 64;
    header["params"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            assert_eq!(e["dtype"], "f32", "the fixtures' tensors are 32-bit floats: {e}");
            let offset = data_start + e["offset"].as_u64().unwrap() as usize;
            let nbytes = e["nbytes"].as_u64().unwrap() as usize;
            SavedTensor {
                name: e["name"].as_str().unwrap().to_string(),
                shape: e["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
                data: bytes[offset..offset + nbytes]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                    .collect(),
            }
        })
        .collect()
}

/// A printed tensor agrees with stored 32-bit floats to the digits `print`
/// writes.
fn assert_same_values(label: &str, printed: &[f64], stored: &[f64]) {
    assert_eq!(printed.len(), stored.len(), "{label}: element count");
    for (i, (p, s)) in printed.iter().zip(stored).enumerate() {
        assert!((p - s).abs() <= 1e-5 * s.abs().max(1.0), "{label}[{i}]: printed {p}, stored {s}");
    }
}

const SAVE_MODEL: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = randn([4, 6]) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
"#;

const SAVE_DATA: &str = r#"
let x = (arange(12).reshape([3, 4])) * 0.1 + full([3, 4], 0.1)
let y = (arange(18).reshape([3, 6])) * 0.2 - full([3, 6], 1.0)
"#;

/// Print `<model>.<field>` between `<tag>_BEGIN` and `<tag>_END`.
fn show(tag: &str, expr: &str) -> String {
    format!("let t_{tag}: Tensor = {expr}\nprint(\"{tag}_BEGIN\")\nprint(t_{tag})\nprint(\"{tag}_END\")\n")
}

#[test]
fn a_trained_adapter_is_saved_with_the_model_and_loaded_back() {
    for source_ad in [false, true] {
        let dir = tempfile::TempDir::new().unwrap();
        let src = format!(
            "{SAVE_MODEL}
@freeze
@adapter(type=lora, target=[\"Toy.w\"], rank=2, alpha=4)
let m = Toy()
{SAVE_DATA}
train(model = m, epochs = 10):
    optimizer: SGD(lr = 0.1)
    step(batch):
        let loss = mse_loss(m.forward(x), y)
{}{}{}model_save(m, \"adapted.nslm\")
let m2 = Toy()
{}{}model_load(m2, \"adapted.nslm\")
{}{}{}",
            show("W", "m.w"),
            show("A", "m.lora_A_Toy_w__lora"),
            show("B", "m.lora_B_Toy_w__lora"),
            // The second model draws its own w and A, and its B is zero.
            show("FRESH_W", "m2.w"),
            show("FRESH_A", "m2.lora_A_Toy_w__lora"),
            show("W2", "m2.w"),
            show("A2", "m2.lora_A_Toy_w__lora"),
            show("B2", "m2.lora_B_Toy_w__lora"),
        );
        let r = run_in(dir.path(), &src, source_ad);
        assert!(r.ok, "source_ad={source_ad}:\n{}", r.stderr);
        let (w, a, b) = (section(&r.stdout, "W"), section(&r.stdout, "A"), section(&r.stdout, "B"));
        let b_size = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(b_size > 1e-2, "B should have trained away from zero, got {b:?}");

        // The file holds the weight and then the two adapter tensors, with
        // the values the program printed.
        let saved = read_nslm(&dir.path().join("adapted.nslm"));
        let names: Vec<&str> = saved.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["w", "lora_A_Toy_w__lora", "lora_B_Toy_w__lora"]);
        assert_eq!(saved[0].shape, [4, 6]);
        assert_eq!(saved[1].shape, [4, 2], "A is [in, rank]");
        assert_eq!(saved[2].shape, [2, 6], "B is [rank, out]");
        assert_same_values("saved w", &w, &saved[0].data);
        assert_same_values("saved A", &a, &saved[1].data);
        assert_same_values("saved B", &b, &saved[2].data);

        // Loading replaces the fresh model's own draw with the saved tensors.
        assert_ne!(section(&r.stdout, "FRESH_W"), w, "the second model starts from its own w");
        assert_ne!(section(&r.stdout, "FRESH_A"), a, "the second model starts from its own A");
        assert_eq!(section(&r.stdout, "W2"), w);
        assert_eq!(section(&r.stdout, "A2"), a);
        assert_eq!(section(&r.stdout, "B2"), b);
    }
}

#[test]
fn a_base_checkpoint_loads_into_a_model_with_adapters() {
    // What a LoRA fine-tune starts from: a file saved before the adapters
    // existed. It sets the weight and leaves A and B as constructed.
    let dir = tempfile::TempDir::new().unwrap();
    let save = format!(
        "{SAVE_MODEL}
let m = Toy()
{}model_save(m, \"base.nslm\")
",
        show("W", "m.w")
    );
    let saved = run_in(dir.path(), &save, false);
    assert!(saved.ok, "{}", saved.stderr);
    let w = section(&saved.stdout, "W");
    assert_eq!(read_nslm(&dir.path().join("base.nslm")).len(), 1, "the base file holds w alone");

    let load = format!(
        "{SAVE_MODEL}
@adapter(type=lora, target=[\"Toy.w\"], rank=2, alpha=4)
let m = Toy()
{}model_load(m, \"base.nslm\")
{}{}{}",
        show("A0", "m.lora_A_Toy_w__lora"),
        show("W", "m.w"),
        show("A", "m.lora_A_Toy_w__lora"),
        show("B", "m.lora_B_Toy_w__lora"),
    );
    let loaded = run_in(dir.path(), &load, false);
    assert!(loaded.ok, "{}", loaded.stderr);
    assert_eq!(section(&loaded.stdout, "W"), w, "the weight comes from the file");
    assert_eq!(
        section(&loaded.stdout, "A"),
        section(&loaded.stdout, "A0"),
        "A keeps the value the constructor drew"
    );
    assert_eq!(section(&loaded.stdout, "B"), vec![0.0; 12], "B stays zero");
}

#[test]
fn a_file_with_adapter_tensors_does_not_load_into_a_model_without_them() {
    let dir = tempfile::TempDir::new().unwrap();
    let save = format!(
        "{SAVE_MODEL}
@adapter(type=lora, target=[\"Toy.w\"], rank=2, alpha=4)
let m = Toy()
model_save(m, \"adapted.nslm\")
"
    );
    assert!(run_in(dir.path(), &save, false).ok);
    let load = format!(
        "{SAVE_MODEL}
let m = Toy()
model_load(m, \"adapted.nslm\")
print(\"LOADED\")
"
    );
    let r = run_in(dir.path(), &load, false);
    assert!(!r.ok && !r.stdout.contains("LOADED"), "the load should be refused:\n{}", r.stdout);
    assert!(
        r.stderr.contains("the file has 3 tensors and the model has 1") && r.stderr.contains("adapter"),
        "the refusal should give both counts and mention adapters:\n{}",
        r.stderr
    );
}

/// There is no merge built in, but an adapter's tensors are ordinary
/// tensors of the model, so a LoRA update can be folded into its weight in
/// NSL source: `W += (A @ B) * (alpha / rank)`, then B back to zero. The
/// merged weight is checked against that sum computed here in f64, and the
/// model's output must not change.
#[test]
fn a_lora_adapter_can_be_merged_into_its_weight_in_nsl_source() {
    let src = format!(
        "{SAVE_MODEL}
@adapter(type=lora, target=[\"Toy.w\"], rank=2, alpha=4)
let m = Toy()
{SAVE_DATA}
train(model = m, epochs = 6):
    optimizer: SGD(lr = 0.1)
    step(batch):
        let loss = mse_loss(m.forward(x), y)
{}{}{}{}m.w = m.w + (m.lora_A_Toy_w__lora @ m.lora_B_Toy_w__lora) * 2.0
m.lora_B_Toy_w__lora = zeros([2, 6])
{}{}{}",
        show("W", "m.w"),
        show("A", "m.lora_A_Toy_w__lora"),
        show("B", "m.lora_B_Toy_w__lora"),
        show("Y", "m.forward(x)"),
        show("MERGED_W", "m.w"),
        show("MERGED_B", "m.lora_B_Toy_w__lora"),
        show("MERGED_Y", "m.forward(x)"),
    );
    let out = run_ok(&src, false);
    let (w, a, b) = (section(&out, "W"), section(&out, "A"), section(&out, "B"));
    assert!(b.iter().any(|v| v.abs() > 1e-2), "B should have trained: {b:?}");
    // W is [4, 6], A is [4, 2], B is [2, 6]; alpha / rank = 2.
    let want: Vec<f64> = (0..4 * 6)
        .map(|i| {
            let (k, j) = (i / 6, i % 6);
            w[i] + 2.0 * (0..2).map(|q| a[2 * k + q] * b[6 * q + j]).sum::<f64>()
        })
        .collect();
    let merged = section(&out, "MERGED_W");
    assert_eq!(merged.len(), want.len());
    let moved = want.iter().zip(&w).map(|(m, w)| (m - w).abs()).fold(0.0, f64::max);
    assert!(moved > 1e-2, "the merge should change the weight (it moves {moved})");
    for (i, (got, want)) in merged.iter().zip(&want).enumerate() {
        assert!((got - want).abs() < 1e-5, "merged w[{i}] = {got}, W + 2·A·B = {want}");
    }
    assert_eq!(section(&out, "MERGED_B"), vec![0.0; 12]);
    let (y, merged_y) = (section(&out, "Y"), section(&out, "MERGED_Y"));
    assert_eq!(y.len(), 18);
    for (i, (before, after)) in y.iter().zip(&merged_y).enumerate() {
        assert!((before - after).abs() < 1e-5, "output [{i}] changed in the merge: {before} -> {after}");
    }
}
