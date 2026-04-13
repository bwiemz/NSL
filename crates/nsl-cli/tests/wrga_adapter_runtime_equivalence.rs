//! WRGA B.2.1 Task 5: runtime equivalence tests for the LoRA forward-pass
//! AST rewrite.
//!
//! Build 1 = reference (no adapter), proves the test harness + output parser.
//! Build 2 = adapter present, default init (LoRA-B=zeros).  SANITY: matches
//!   Build 1.  NOT LOAD-BEARING — if the rewrite never fires, this still
//!   passes (because `x @ zeros == 0` and `(ones @ 0 @ 0) * scale == 0`).
//! Build 3 = adapter present, LoRA-B seeded nonzero.  SANITY: output diverges
//!   from Build 1 IFF the rewrite actually fired AND seeding took effect.
//! Build 4 = A=ones, B=ones, x=ones, W=zeros, alpha=rank=2 (scale=1.0).
//!   LOAD-BEARING PROOF.  If the rewrite produced `y = y_base + x @ A^T @ B^T
//!   * scale`, every output element = 0 + 8*2*1 = 16.0.  Tolerance: 1e-5 f32.
//!
//! ### Architectural blockers discovered during Task 5 implementation
//!
//! Two interlocking constraints prevent Build 3 and Build 4 from seeding
//! the side-table from ordinary NSL source today:
//!
//!   (a) **Direct field assignment** (`m.lora_B_... = ones(...)`) errors with
//!       `no field 'lora_B_...'` — assignment wiring for synthesized adapter
//!       field names is not implemented (only read-through is, via
//!       `expr/access.rs::is_synthesized_adapter_field_name`).
//!
//!   (b) **In-place seeding via `copy_data(m.lora_B_..., ones(...))` inside a
//!       `train` step** compiles only on the source-AD path, because that is
//!       where `invoke_wrga_if_enabled` runs and populates `adapter_sites` +
//!       `last_wrga_plan`.  But the Wengert extractor rejects `copy_data` as
//!       an "unsupported function", falling back to tape-AD — which does NOT
//!       invoke WRGA.  Net: the access `m.lora_B_...` emits the codegen error
//!       `synthesized adapter field 'lora_B_...' not found for model 'Toy'
//!       in current WRGA plan`.
//!
//! A third deeper concern: `compile_user_functions` (which is where model
//! methods including `forward` are compiled) runs BEFORE `compile_main`
//! (which is where the train block runs WRGA and populates
//! `compiler.adapter_sites`).  At the time the rewrite pass consults
//! `self.adapter_sites` in `compiler/functions.rs` line 532, the vector is
//! empty, so the LoRA AST rewrite is skipped for every model method.  This
//! is the exact wiring gap the plan warned about; it will manifest as
//! Build 4 output = `y_base = zeros`, i.e. a failure by exactly 16.0 per
//! element.
//!
//! Per Task 5 guardrails: "don't modify rewrite logic".  Build 4 below is
//! written to fire the full pipeline and assert the expected value; if it
//! fails by exactly 16.0 per element, that is precisely the diagnostic the
//! plan wanted from this task.  A subsequent task owns the wiring fix.

use assert_cmd::prelude::*;
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_nsl(src_path: &Path, extra_args: &[&str]) -> (bool, String, String) {
    let root = workspace_root();
    let stdlib = root.join("stdlib");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .arg("run")
        .args(extra_args)
        .arg(src_path);
    let output = cmd.output().expect("nsl run failed to spawn");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

fn run_nsl_expect_ok(src_path: &Path, extra_args: &[&str]) -> String {
    let (ok, stdout, stderr) = run_nsl(src_path, extra_args);
    if !ok {
        panic!(
            "nsl run failed.\nstdout:\n{stdout}\nstderr:\n{stderr}",
        );
    }
    stdout
}

/// Parse a `tensor([[...], [...], ...])` line from stdout into Vec<Vec<f32>>.
fn parse_tensor_2d(s: &str) -> Vec<Vec<f32>> {
    // Locate the tensor(...) payload on any line.
    let line = s
        .lines()
        .find(|l| l.trim_start().starts_with("tensor("))
        .expect("no tensor(...) line in stdout");
    let open = line.find('(').unwrap() + 1;
    let close = line.rfind(')').unwrap();
    let inner = &line[open..close]; // "[[1,2],[3,4]]"
    let inner = inner.trim().trim_start_matches('[').trim_end_matches(']');
    inner
        .split("],")
        .map(|row| {
            row.trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .filter_map(|v| v.trim().parse::<f32>().ok())
                .collect()
        })
        .collect()
}

// ─── Build 1: reference (no adapter) ─────────────────────────────────────

const REFERENCE_SRC: &str = r#"
model Toy:
    w: Tensor = zeros([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Toy()
let x = ones([4, 8])
let y = m.forward(x)
print(y)
"#;

#[test]
fn build_1_reference_no_adapter() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build1.nsl");
    fs::write(&src_path, REFERENCE_SRC).unwrap();
    let stdout = run_nsl_expect_ok(&src_path, &["--source-ad"]);
    let tensor = parse_tensor_2d(&stdout);
    assert_eq!(tensor.len(), 4, "expected 4 rows, got {tensor:?}");
    for (i, row) in tensor.iter().enumerate() {
        assert_eq!(row.len(), 8, "row {i} wrong width: {row:?}");
        for (j, v) in row.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "Build 1: ones @ zeros must be 0 at [{i},{j}]; got {v}",
            );
        }
    }
}

// ─── Build 2: adapter present, default init (LoRA-B = zeros) ─────────────
//
// The `train` block is required so Task 2.5's init emission fires and
// `invoke_wrga_if_enabled` populates adapter_sites.  We use lr=0 and a
// zero target so the optimiser step is a no-op regardless of gradients.

const BUILD2_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = zeros([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=lora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
let x = ones([4, 8])
let y_target = zeros([4, 8])
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
let y = m.forward(x)
print(y)
"#;

#[test]
fn build_2_sanity_adapter_default_init_matches_base() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build2.nsl");
    fs::write(&src_path, BUILD2_SRC).unwrap();
    let stdout = run_nsl_expect_ok(&src_path, &["--source-ad"]);
    let tensor = parse_tensor_2d(&stdout);
    for (i, row) in tensor.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "Build 2 sanity: expected 0 at [{i},{j}]; got {v}",
            );
        }
    }
}

// ─── Build 3: adapter present, LoRA-B seeded nonzero via copy_data ───────
//
// Build 3 requires overwriting the zero-initialised LoRA-B tensor with
// nonzero values before the final forward pass.  The only code region in
// which `m.lora_B_Toy_w__lora` resolves (i.e. both `adapter_sites` and
// `last_wrga_plan` are populated) is the body of the train step when
// compiled on the source-AD path.  `copy_data` is, however, rejected by
// the Wengert extractor as unsupported, which triggers a fallback to
// tape-AD and loses the WRGA plan.
//
// Until either (a) direct field assignment `m.lora_B_... = ...` is wired,
// or (b) the Wengert extractor whitelists `copy_data` / a dedicated
// adapter-seed FFI is exposed, Build 3 cannot be expressed in surface NSL.
// Documented as `#[ignore]` so the test suite runs cleanly but the
// expectation remains on-record for the wiring-fix task.

// Build 3: adapter present, LoRA-B seeded nonzero. Output must diverge from
// Build 1. Task 5.5 fixed the two wiring issues that previously blocked
// expressing the seed in surface NSL:
//   (a) adapter_sites is now pre-populated BEFORE user-function compilation,
//   (b) field-assignment codegen now routes synthesized adapter names
//       through the side-table (mirror of read-through).
const BUILD3_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = zeros([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=lora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
let x = ones([4, 8])
let y_target = zeros([4, 8])
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_B_Toy_w__lora = ones([2, 8])
let y = m.forward(x)
print(y)
"#;

#[test]
fn build_3_sanity_b_nonzero_diverges() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build3.nsl");
    fs::write(&src_path, BUILD3_SRC).unwrap();
    let stdout = run_nsl_expect_ok(&src_path, &["--source-ad"]);
    let tensor = parse_tensor_2d(&stdout);
    // If seeding + rewrite both fire, LoRA-A is Kaiming-random so output is
    // deterministic per-seed but nonzero in at least one element. If either
    // fails, we'd see all zeros (Build 1 / Build 2).
    let any_nonzero = tensor
        .iter()
        .flat_map(|r| r.iter())
        .any(|v| v.abs() > 1e-5);
    assert!(
        any_nonzero,
        "Build 3: output must diverge from Build 1 when LoRA-B is seeded nonzero; got {tensor:?}",
    );
}

// ─── Build 4: LOAD-BEARING PROOF of the LoRA forward rewrite ─────────────
//
// Desired seeding: A = ones([2, 8]), B = ones([8, 2]).
// Expected output with the rewrite firing and scale = alpha/rank = 1.0:
//   y[i,j] = (x @ W)[i,j] + ((x @ A^T) @ B^T)[i,j] * scale
//         = 0           + ((1x8 @ 8x2) @ 2x8)[i,j] * 1.0
//         = 8 * 2 * 1   = 16.0 for every element.
//
// The seeding blocker documented on Build 3 applies here verbatim.  The
// assertion below is written against the *desired* numeric result so that
// once seeding lands, Build 4 becomes the load-bearing proof unchanged.

// Build 4: LOAD-BEARING PROOF.
//
// W=zeros, x=ones([4,8]), A=ones([2,8]), B=ones([8,2]), alpha=rank=2 ⇒
// scale=1.0.  Expected y[i,j] = (x @ W)[i,j] + (x @ A^T @ B^T)[i,j] * scale
//                             = 0 + (1x8 @ 8x2 @ 2x8)[i,j] * 1.0
//                             = 8 * 2 * 1 = 16.0 for every element.
const BUILD4_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = zeros([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=lora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
let x = ones([4, 8])
let y_target = zeros([4, 8])
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_A_Toy_w__lora = ones([8, 2])
m.lora_B_Toy_w__lora = ones([2, 8])
let y = m.forward(x)
print(y)
"#;

#[test]
fn build_4_lora_rewrite_load_bearing_proof() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build4.nsl");
    fs::write(&src_path, BUILD4_SRC).unwrap();
    let stdout = run_nsl_expect_ok(&src_path, &["--source-ad"]);
    let tensor = parse_tensor_2d(&stdout);
    assert_eq!(tensor.len(), 4, "expected 4 rows, got {tensor:?}");
    for (i, row) in tensor.iter().enumerate() {
        assert_eq!(row.len(), 8, "row {i} wrong width: {row:?}");
        for (j, v) in row.iter().enumerate() {
            assert!(
                (v - 16.0).abs() < 1e-5,
                "Build 4 load-bearing: expected 16.0 at [{i},{j}]; got {v}",
            );
        }
    }
}
