//! WRGA paper §8.3 — `nsl check --wrga-analyze` integration tests.
//!
//! Closes audit Gap #2: the paper specifies a diagnostic CLI for emitting the
//! WRGA compilation report without running codegen. Before this PR there was
//! only `nsl build --wrga-report`, which forces a full compile + `.o`. These
//! tests pin the new short-circuiting check path.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

const SRC_WITH_DECORATORS: &str = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@freeze(include=["m.w"])
let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;

const SRC_NO_DECORATORS: &str = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;

/// Paper §8.3 form: `nsl check --wrga-analyze model.nsl` prints the report to
/// stdout (no path argument → default `-`).
#[test]
fn wrga_analyze_no_path_prints_to_stdout() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ==="))
        .stdout(predicate::str::contains("Frozen parameters"));
}

/// Explicit `-` form is identical to no-path.
#[test]
fn wrga_analyze_dash_writes_to_stdout() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze=-");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ==="));
}

/// Path argument writes the report to that file and produces nothing on
/// stdout. Verifies the file path branch of the dispatch.
#[test]
fn wrga_analyze_with_path_writes_to_file() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let report_path = tmp.path().join("wrga.txt");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg(format!("--wrga-analyze={}", report_path.display()));
    cmd.assert()
        .success()
        .stdout(predicate::str::is_empty().or(predicate::str::contains("=== WRGA").not()));

    let contents = fs::read_to_string(&report_path).unwrap();
    assert!(
        contents.contains("=== WRGA Compilation Report ==="),
        "report file missing header; got: {contents}",
    );
    assert!(contents.contains("Frozen parameters"));
}

/// WRGA paper §8.2 — `@wrga(adapter=<UserModel>)` must survive end-to-end
/// from source through the semantic checker, the codegen WrgaInputs bridge,
/// the WrgaInput→WrgaPlan plumbing, and finally render in the report's
/// header.  Pins the cross-crate bridge layer that the in-crate unit tests
/// don't exercise.  The trailing `(placement integration pending)` marker is
/// load-bearing — review finding #3 was that without it the line was
/// misleading.
#[test]
fn wrga_analyze_surfaces_custom_adapter_name_end_to_end() {
    const SRC_CUSTOM_ADAPTER: &str = r#"from nsl.nn.losses import mse_loss

model GatedLoRA:
    a: Tensor = zeros([2, 1])
    b: Tensor = zeros([1, 2])
    gate: Tensor = ones([2])

    fn forward(self, x: Tensor) -> Tensor:
        return self.gate * (x @ self.a @ self.b)

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@wrga(mode=auto, adapter=GatedLoRA, target=h100)
@freeze(include=["m.w"])
let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_CUSTOM_ADAPTER).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Custom adapter: GatedLoRA"))
        .stdout(predicate::str::contains("(placement integration pending)"));
}

/// `--wrga-target` overrides the WRGA target GPU; the report must reflect it.
#[test]
fn wrga_analyze_with_target_overrides_gpu_in_report() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze")
        .arg("--wrga-target")
        .arg("H100-SXM");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Target hardware: H100-SXM"));
}

/// No `@wrga` / `@freeze` / `@adapter` decorators in source → exit 2 with a
/// distinct error message (so CI can tell "no decorators" apart from compile
/// failure).
#[test]
fn wrga_analyze_no_decorators_exits_with_distinct_code() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_NO_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze");
    cmd.assert()
        .code(2)
        .stderr(predicate::str::contains("no @wrga / @freeze / @adapter decorators"));
}

/// `--wrga-analyze` must NOT create a `.o` next to the source. Mirrors the
/// `nsl check` side-effect-free contract — `nsl build` writes objects, `nsl
/// check` only prints.
///
/// The source lives in a dedicated subdirectory so the assertion can't pass
/// vacuously: any unexpected `.o` (whether at the documented `t.o` path or
/// anywhere else in the subdir) fails the test. This catches both the
/// multi-file build path (which uses internal temp dirs) and the single-file
/// build path (which historically wrote `.o` adjacent to the source).
#[test]
fn wrga_analyze_does_not_emit_object_file() {
    let tmp = TempDir::new().unwrap();
    let src_dir = tmp.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    let src_path = src_dir.join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    // Sanity: no stale artifacts in the source dir before the call.
    let stale: Vec<_> = fs::read_dir(&src_dir).unwrap().filter_map(Result::ok).collect();
    assert_eq!(
        stale.len(),
        1,
        "test setup leaked extra files: {stale:?}"
    );

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze");
    cmd.assert().success();

    // After the call: the only file in src_dir must still be t.nsl. Any .o
    // (or any other artifact) is a side-effect leak. We list and assert on
    // the full directory contents to catch leaks no matter the filename.
    let after: Vec<String> = fs::read_dir(&src_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name().into_string().unwrap_or_default())
        .collect();
    assert_eq!(
        after,
        vec!["t.nsl".to_string()],
        "--wrga-analyze leaked artifacts into source dir: {after:?}"
    );
}

/// An unknown GPU name is an error that lists the known ones. It used to
/// fall back to the database default without a word, so a report asked for
/// one GPU described another.
#[test]
fn wrga_analyze_with_unknown_target_is_an_error() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze")
        .arg("--wrga-target")
        .arg("definitely-not-a-real-gpu-x999");
    cmd.assert()
        .failure()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ===").not())
        .stderr(predicate::str::contains("WRGA target 'definitely-not-a-real-gpu-x999'"))
        .stderr(predicate::str::contains("H100-SXM"));
}

/// A safetensors file holding one 32-bit float tensor: an 8-byte header
/// length, a JSON header, then the data.
fn write_safetensors(path: &std::path::Path, name: &str, shape: [usize; 2], data: &[f32]) {
    assert_eq!(data.len(), shape[0] * shape[1]);
    let header = format!(
        r#"{{"{name}":{{"dtype":"F32","shape":[{},{}],"data_offsets":[0,{}]}}}}"#,
        shape[0],
        shape[1],
        data.len() * 4
    );
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

/// `--weights` reaches the analysis: the report gains its spectral line.
/// The flag used to be dropped by `nsl check --wrga-analyze`, whose report
/// was the same with and without it.
#[test]
fn wrga_analyze_uses_the_weights_it_is_given() {
    const SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = randn([16, 12]) * 0.3

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@freeze
@adapter(type=lora, target=["Toy.w"], rank=4)
let m = Toy()
let x = ones([4, 16])
let y = zeros([4, 12])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let loss = mse_loss(m.forward(x), y)
"#;
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC).unwrap();
    let weights_path = tmp.path().join("w.safetensors");
    // No structure, so no singular value dominates.
    let data: Vec<f32> = (0..16 * 12).map(|i| ((i * 37 + 11) % 101) as f32 / 101.0 - 0.5).collect();
    write_safetensors(&weights_path, "w", [16, 12], &data);

    let report = |with_weights: bool| -> String {
        let mut cmd = Command::cargo_bin("nsl").unwrap();
        cmd.env("NSL_STDLIB_PATH", stdlib_path()).arg("check").arg(&src_path).arg("--wrga-analyze");
        if with_weights {
            cmd.arg("--weights").arg(&weights_path);
        }
        let out = cmd.output().unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let (plain, weighted) = (report(false), report(true));
    // The declared adapter, at its real size either way: 4 × (16 + 12).
    for r in [&plain, &weighted] {
        assert!(r.contains("Adapter parameters: 112  |  Sites: 1"), "{r}");
    }
    assert!(!plain.contains("Spectral analysis"), "{plain}");
    assert!(
        weighted.contains("Spectral analysis: 1 of 1 sites matched a checkpoint weight"),
        "{weighted}"
    );
}
