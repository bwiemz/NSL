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
//! The CPU builds above seed the adapter tensors by assignment
//! (`m.lora_A_<site> = ...`) and use constant operands, which is enough to
//! show the rewrite fires and the fused call is reached. They cannot tell a
//! right kernel from a wrong one; the tests of the synthesized kernels on
//! the GPU (`run_fused_kernel_fixture`, built with the `cuda` feature) use
//! non-constant operands and a reference computed in f64.

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
    // A build for a pinned sm target is how these tests reach the fused
    // adapter call, and that call is opt-in at compile time.
    if extra_args.iter().any(|a| a.contains("sm80") || a.contains("sm_80")) {
        cmd.env("NSL_WRGA_FUSED_CUDA", "1");
    }
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

// GPU-resident variant of BUILD4_SRC used ONLY by the `#[ignore]`-d
// hardening test `build_4_fused_cuda_actually_fires`.  Every tensor that
// feeds into `m.forward(x)` — the model weight, the input, and both
// seeded adapter tensors — is placed on CUDA so the fused FFI's "inputs
// on GPU" gate is satisfied and the real cudarc PTX launch fires.
//
// Kept separate from BUILD4_SRC because the non-ignored build_4* tests
// must continue to exercise the CPU-fallback path on machines without
// a CUDA device.
// driven by the `#[cfg(feature = "cuda")]` GPU-equivalence tests below
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const BUILD4_SRC_GPU: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = zeros([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=lora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_A_Toy_w__lora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__lora = ones([2, 8]).to(cuda)
let y = m.forward(x)
print(y)
"#;

// ─── B.3 Task 4: fused FFI reachability when target=cuda_sm80 ───────────
//
// Same load-bearing source as Build 4, but compiled with `--target
// cuda_sm80`.  The AST rewrite's fused path activates (FusionTarget =
// EpilogueFusedLora AND sm >= 80), emitting a single Call to
// `nsl_adapter_fused_lora_matmul` instead of the unfused triple.  The
// Task 4 CPU-fallback FFI computes identical math, so the numeric
// output must still be 16.0 per element — proving:
//   (a) the fused FFI is reachable (codegen produces a linkable symbol),
//   (b) the conditional rewrite fires for sm_80+ targets,
//   (c) the CPU stub's math matches B.2.1 within 1e-5 tolerance.
//
// Task 5 will replace the CPU stub with a real CUDA launch; the
// assertion stays unchanged (numeric equivalence at 1e-5).
#[test]
fn task_4_fused_ffi_is_referenced_when_target_sm80() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("task4_sm80.nsl");
    fs::write(&src_path, BUILD4_SRC).unwrap();
    let stdout =
        run_nsl_expect_ok(&src_path, &["--source-ad", "--target", "cuda_sm80"]);
    let tensor = parse_tensor_2d(&stdout);
    assert_eq!(tensor.len(), 4, "expected 4 rows, got {tensor:?}");
    for (i, row) in tensor.iter().enumerate() {
        assert_eq!(row.len(), 8, "row {i} wrong width: {row:?}");
        for (j, v) in row.iter().enumerate() {
            assert!(
                (v - 16.0).abs() < 1e-5,
                "Task 4 fused path: expected 16.0 at [{i},{j}]; got {v}",
            );
        }
    }
}

// ─── B.3 Task 5: build_4_fused — load-bearing proof at 1e-4 tolerance ───
//
// Same source as Build 4, compiled with `--target cuda_sm80` so the AST
// rewrite emits a single Call to `nsl_adapter_fused_lora_matmul` with a
// real, deterministic `kernel_handle` (Task 5 wiring).  Tolerance is
// **1e-4**, NOT 1e-3:
//   - 1e-3 would mask numeric drift indicating the PTX epilogue is doing
//     something subtly wrong.  If 1e-4 fails, **investigate before
//     loosening** — see diagnostic guidance in the panic message.
//
// As of Task 5 the FFI body is still the CPU-fallback stub (real cudarc
// launch is the test's eventual unblock target — see
// `build_5_real_cuda_launch` ignored case).  The CPU stub computes the
// exact math, so this test exercises the full dispatch + kernel_handle
// plumbing and asserts numeric correctness; the runtime-counter test
// (build_5) is what proves "exactly one kernel per site".
#[test]
fn build_4_fused() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build4_fused.nsl");
    fs::write(&src_path, BUILD4_SRC).unwrap();
    let stdout =
        run_nsl_expect_ok(&src_path, &["--source-ad", "--target", "cuda_sm80"]);
    let tensor = parse_tensor_2d(&stdout);
    assert_eq!(tensor.len(), 4, "expected 4 rows, got {tensor:?}");
    let mut max_diff: f32 = 0.0;
    for row in &tensor {
        assert_eq!(row.len(), 8);
        for v in row {
            // `max_diff.max(nan)` keeps `max_diff`: refuse NaN before folding.
            assert!(v.is_finite(), "non-finite entry in output tensor: {v} in {tensor:?}");
            max_diff = max_diff.max((v - 16.0).abs());
        }
    }
    assert!(
        max_diff < 1e-4,
        "build_4_fused: max |y - 16.0| = {max_diff:.3e}, want < 1e-4. \
         Diagnostic: \
         ~1e-3 = numeric bug (epilogue accumulator order or scale dtype); \
         ~1.0+ = scale/dim wrong (alpha/rank, m@A@B order); \
         exactly 0 = test never exercised the fused path (rewrite skipped); \
         exactly 16.0 difference (y == 0) = base matmul fired but adapter delta dropped. \
         Tensor: {tensor:?}",
    );
}

// ─── B.3 Task 5: build_5_kernel_count — exactly one launch per site ─────
//
// Runs the same single-LoRA-site source as Build 4, with the runtime
// kernel-launch counter enabled via `NSL_KERNEL_LAUNCH_COUNTER=1`.  At
// process exit the runtime prints `[nsl-kernel-count] <N>` to stderr.
// Asserts N == 1: one fused FFI call per `m.forward(x)` invocation, and
// the BUILD4_SRC source invokes forward exactly once *after* the train
// step.  (The train step itself also invokes forward, so the total is 2
// — see assertion.)
#[test]
fn build_5_fused_launches_one_kernel_per_site() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build5_count.nsl");
    fs::write(&src_path, BUILD4_SRC).unwrap();

    let root = workspace_root();
    let stdlib = root.join("stdlib");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    // NSL_WRGA_FUSED_CUDA=1: the fused adapter call is opt-in at compile
    // time (a pinned target alone no longer selects it). The tensors here
    // are host tensors, so the runtime still takes its CPU math; the
    // counter counts the fused calls either way.
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_KERNEL_LAUNCH_COUNTER", "1")
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path);
    let output = cmd.output().expect("nsl run failed to spawn");
    assert!(
        output.status.success(),
        "nsl run failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Find the count line.
    let line = stderr
        .lines()
        .find(|l| l.contains("[nsl-kernel-count]"))
        .unwrap_or_else(|| {
            panic!(
                "no [nsl-kernel-count] line in stderr; counter never fired. \
                 Either env-gating broken or atexit didn't run.\nstderr:\n{stderr}",
            )
        });
    let count: u64 = line
        .split_whitespace()
        .next_back()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("malformed count line: {line:?}"));

    // After WRGA B.3.2 Option 3 revised (phase 3e + source-AD handler
    // for the fused FFI), the in-train-step `m.forward` ALSO resolves
    // through the rewritten fused path — source-AD now owns the
    // `FusedLoraMatmul` primal and emits the FFI call during lowering.
    // BUILD4_SRC's `epochs = 1` train block contributes 1 launch and
    // the post-train `let y = m.forward(x)` contributes 1 more, so
    // `count == 2` is the correct steady-state.
    assert_eq!(
        count, 2,
        "build_5: expected exactly 2 fused launches (1 per train iter + \
         1 post-train forward), got {count}. \
         Diagnostic: \
         0 = rewrite never produced a fused Call (target_sm or fusion \
         gate failed); \
         2+ = the same site was lowered through both the fused and \
         unfused paths, or the AST rewrite duplicated the Call.",
    );
}

// ─── B.3 Task 5.6 hardening: build_4_fused_cuda_actually_fires ───────
//
// Build 4 passing at 1e-4 is consistent with EITHER the real cudarc PTX
// launch OR the CPU fallback (both produce the correct 16.0).  This
// hardening test pins down that the real CUDA path actually executes
// when `NSL_WRGA_FUSED_CUDA=1` is set, using a runtime counter that
// only increments on successful cudarc launch — never on fallback.
//
// Test is `#[cfg(feature = "cuda")]`-gated: runs automatically on CUDA
// machines when the `cuda` feature is enabled.  The fused LoRA PTX
// kernel is now valid (Tasks A-E) so the real cudarc launch fires.
//   cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence \
//     --features cuda build_4_fused_cuda_actually_fires
#[cfg(feature = "cuda")]
#[test]
fn build_4_fused_cuda_actually_fires() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build4_gpu.nsl");
    fs::write(&src_path, BUILD4_SRC_GPU).unwrap();

    let root = workspace_root();
    let stdlib = root.join("stdlib");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .env("NSL_WRGA_GPU_LAUNCH_COUNTER", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path);
    let output = cmd.output().expect("nsl run failed to spawn");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "nsl run failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        stderr,
    );

    // Parse `[nsl-gpu-launch-count] <N>`.  N > 0 proves the real cudarc
    // PTX launch fired at least once; 0 means every fused FFI call fell
    // back to CPU math.
    let line = stderr
        .lines()
        .find(|l| l.contains("[nsl-gpu-launch-count]"))
        .unwrap_or_else(|| {
            panic!(
                "no [nsl-gpu-launch-count] line in stderr; atexit hook didn't \
                 fire.\nstderr:\n{stderr}",
            )
        });
    let count: u64 = line
        .split_whitespace()
        .next_back()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("malformed gpu-count line: {line:?}"));

    assert!(
        count >= 1,
        "expected ≥1 real CUDA launch with NSL_WRGA_FUSED_CUDA=1, got {count}. \
         0 means all fused FFI calls hit the CPU fallback — check registry \
         population, cudarc PTX load, or GPU tensor placement.  See stderr \
         for fallback warnings:\n{stderr}",
    );
}

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

// ─── The synthesized kernels on the GPU ─────────────────────────────────
//
// Each test launches one synthesized kernel on non-constant operands and
// holds every element of its result to a reference computed here in f64
// from the same formulas the NSL source uses to fill the tensors.
//
// The fixtures these replace filled every operand with a constant (x, A and
// B all ones, W zeros or ones): their expected value, 16.0, is also what a
// kernel returns that drops the base product, swaps A's or B's axes, or
// reads any operand at the wrong index.
//
// Tolerance. The kernels compute in half precision: each operand is rounded
// to a 16-bit float (relative error up to 2⁻¹¹ ≈ 4.9e-4) before the tensor-
// core product, the `x @ A` intermediate is rounded again, and sums are
// kept in 32-bit. A product of two rounded operands is off by at most about
// 1e-3 of its magnitude, so an element is allowed 1.5e-3 of the sum of the
// magnitudes of its terms. That is a bound a correct half-precision kernel
// meets and a wrong one does not: dropping a term, or misreading an
// operand, moves an element by the size of the terms themselves.

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq)]
enum FusedKind {
    Lora,
    GatedLora,
    Ia3,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
struct FusedShape {
    /// Rows of x.
    m: usize,
    /// Input width: x is `[m, k]`, W is `[k, n]`.
    k: usize,
    /// Output width.
    n: usize,
    rank: usize,
    alpha: usize,
}

/// `arange(rows * cols).reshape([rows, cols]) * step - offset`, row-major.
#[cfg(feature = "cuda")]
fn ramp(len: usize, step: f64, offset: f64) -> Vec<f64> {
    (0..len).map(|i| i as f64 * step - offset).collect()
}

/// The NSL expression for the same tensor, on the GPU.
#[cfg(feature = "cuda")]
fn ramp_src(shape: &[usize], step: f64, offset: f64) -> String {
    let len: usize = shape.iter().product();
    let dims = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
    format!("((arange({len}).reshape([{dims}])) * {step} - full([{dims}], {offset})).to(cuda)")
}

/// Run one adapted matmul on the GPU with the fused path opted in and check
/// every element. `kernel_serves_it` says whether the shape is one the
/// kernel computes (one output tile, rank up to 8): then exactly one kernel
/// launch must produce the result. Otherwise the launcher must decline, say
/// so, and the ordinary operations must produce it.
#[cfg(feature = "cuda")]
fn run_fused_kernel_fixture(kind: FusedKind, s: FusedShape, kernel_serves_it: bool) {
    let FusedShape { m, k, n, rank, alpha } = s;
    // Steps chosen so that no tensor is symmetric, constant along an axis,
    // or a multiple of another.
    let (x_step, x_off) = (0.07, 0.9);
    let (w_step, w_off) = (0.013, 0.4);
    let (a_step, a_off) = (0.021, 0.3);
    let (b_step, b_off) = (0.017, 0.2);
    let (v_step, v_off) = (0.31, 1.2); // gate, or the IA³ scale

    let (decorator, seeds) = match kind {
        FusedKind::Lora => (
            format!("@adapter(type=lora, target=[\"Toy.w\"], rank={rank}, alpha={alpha})"),
            format!(
                "m.lora_A_Toy_w__lora = {}\nm.lora_B_Toy_w__lora = {}\n",
                ramp_src(&[k, rank], a_step, a_off),
                ramp_src(&[rank, n], b_step, b_off)
            ),
        ),
        FusedKind::GatedLora => (
            format!("@adapter(type=gatedlora, target=[\"Toy.w\"], rank={rank}, alpha={alpha})"),
            format!(
                "m.lora_A_Toy_w__gatedlora = {}\nm.lora_B_Toy_w__gatedlora = {}\nm.gate_Toy_w__gatedlora = {}\n",
                ramp_src(&[k, rank], a_step, a_off),
                ramp_src(&[rank, n], b_step, b_off),
                ramp_src(&[n], v_step, v_off)
            ),
        ),
        FusedKind::Ia3 => (
            "@adapter(type=ia3, target=[\"Toy.w\"])".to_string(),
            format!("m.ia3_scale_Toy_w__ia3 = {}\n", ramp_src(&[n], v_step, v_off)),
        ),
    };
    let src = format!(
        r#"model Toy:
    w: Tensor = zeros([{k}, {n}])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

{decorator}
let m = Toy()
m.to(cuda)
m.w = {w}
{seeds}let x = {x}
let y: Tensor = m.forward(x)
print(y)
"#,
        w = ramp_src(&[k, n], w_step, w_off),
        x = ramp_src(&[m, k], x_step, x_off),
    );

    let label = format!("{kind:?} {s:?}");
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("fused_kernel.nsl");
    fs::write(&src_path, &src).unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .env("NSL_WRGA_GPU_LAUNCH_COUNTER", "1")
        .arg("run")
        .args(["--target", "cuda_sm80"])
        .arg(&src_path);
    let output = cmd.output().expect("nsl run failed to spawn");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(output.status.success(), "{label}: nsl run failed.\nstdout:\n{stdout}\nstderr:\n{stderr}");

    // Where the result must come from. For a shape the kernel serves, from
    // the kernel: the ordinary operations would meet the tolerance below
    // whatever the kernel does.
    let launches: u64 = stderr
        .lines()
        .find(|l| l.contains("[nsl-gpu-launch-count]"))
        .and_then(|l| l.split_whitespace().next_back())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("{label}: no [nsl-gpu-launch-count] line.\nstderr:\n{stderr}"));
    if kernel_serves_it {
        assert_eq!(launches, 1, "{label}: expected exactly one kernel launch.\nstderr:\n{stderr}");
    } else {
        assert_eq!(launches, 0, "{label}: the kernel cannot compute this shape.\nstderr:\n{stderr}");
        assert!(
            stderr.contains("the fused adapter kernel computes one output tile"),
            "{label}: declining the kernel should be said.\nstderr:\n{stderr}"
        );
    }
    // Opting in says what was opted into.
    assert!(
        stderr.contains("half precision"),
        "{label}: the build should say the fused kernels compute in half precision.\nstderr:\n{stderr}"
    );

    // The reference, in f64.
    let x = ramp(m * k, x_step, x_off);
    let w = ramp(k * n, w_step, w_off);
    let a = ramp(k * rank, a_step, a_off);
    let b = ramp(rank * n, b_step, b_off);
    let v = ramp(n, v_step, v_off);
    let scale = alpha as f64 / rank as f64;
    let got = parse_tensor_2d(&stdout);
    assert_eq!(got.len(), m, "{label}: expected {m} rows, got {got:?}");
    let (mut worst, mut worst_allowed) = (0.0_f64, 0.0_f64);
    for i in 0..m {
        assert_eq!(got[i].len(), n, "{label}: row {i} has {} columns", got[i].len());
        for j in 0..n {
            let base: f64 = (0..k).map(|c| x[k * i + c] * w[n * c + j]).sum();
            let base_mag: f64 = (0..k).map(|c| (x[k * i + c] * w[n * c + j]).abs()).sum();
            let delta: f64 = (0..rank)
                .map(|q| (0..k).map(|c| x[k * i + c] * a[rank * c + q]).sum::<f64>() * b[n * q + j])
                .sum();
            let delta_mag: f64 = (0..rank)
                .map(|q| {
                    (0..k).map(|c| (x[k * i + c] * a[rank * c + q]).abs()).sum::<f64>() * b[n * q + j].abs()
                })
                .sum();
            let (want, magnitude) = match kind {
                FusedKind::Lora => (base + scale * delta, base_mag + scale * delta_mag),
                FusedKind::GatedLora => {
                    let gate = 1.0 / (1.0 + (-v[j]).exp());
                    (base + gate * scale * delta, base_mag + gate * scale * delta_mag)
                }
                FusedKind::Ia3 => (base * v[j], base_mag * v[j].abs()),
            };
            let value = f64::from(got[i][j]);
            assert!(value.is_finite(), "{label}: [{i}][{j}] is {value}");
            let (err, allowed) = ((value - want).abs(), 1.5e-3 * magnitude + 1e-6);
            if err / allowed > worst / worst_allowed.max(f64::MIN_POSITIVE) {
                (worst, worst_allowed) = (err, allowed);
            }
            assert!(
                err <= allowed,
                "{label}: y[{i}][{j}] = {value}, f64 reference {want} (|Δ| = {err:.3e}, allowed \
                 {allowed:.3e} for half precision)"
            );
        }
    }
    eprintln!("{label}: worst |Δ| {worst:.3e} against {worst_allowed:.3e} allowed");
}

/// One output tile exactly: 16 rows, 8 columns, one 16-wide step of k.
#[cfg(feature = "cuda")]
const WHOLE_TILE: FusedShape = FusedShape { m: 16, k: 16, n: 8, rank: 2, alpha: 4 };
/// One tile with every dimension short of it or off its grid: tail rows,
/// tail columns, a k that is not a multiple of 16, an odd rank.
#[cfg(feature = "cuda")]
const PARTIAL_TILE: FusedShape = FusedShape { m: 13, k: 20, n: 6, rank: 3, alpha: 6 };
/// The largest rank the kernel's epilogue holds, over two steps of k.
#[cfg(feature = "cuda")]
const RANK_8: FusedShape = FusedShape { m: 16, k: 32, n: 8, rank: 8, alpha: 16 };
/// More than one tile each way. The kernel used to be launched on such
/// shapes and returned the first tile's values in the others: rows 16 and
/// up and columns 8 and up were wrong by hundreds of times the tolerance.
#[cfg(feature = "cuda")]
const SEVERAL_TILES: FusedShape = FusedShape { m: 21, k: 20, n: 11, rank: 3, alpha: 6 };
/// A rank the epilogue does not hold; the kernel was wrong in every element.
#[cfg(feature = "cuda")]
const RANK_9: FusedShape = FusedShape { m: 16, k: 16, n: 8, rank: 9, alpha: 18 };

#[cfg(feature = "cuda")]
#[test]
fn the_fused_lora_kernel_matches_the_reference() {
    run_fused_kernel_fixture(FusedKind::Lora, WHOLE_TILE, true);
    run_fused_kernel_fixture(FusedKind::Lora, PARTIAL_TILE, true);
    run_fused_kernel_fixture(FusedKind::Lora, RANK_8, true);
}

#[cfg(feature = "cuda")]
#[test]
fn the_fused_gated_lora_kernel_matches_the_reference() {
    run_fused_kernel_fixture(FusedKind::GatedLora, WHOLE_TILE, true);
    run_fused_kernel_fixture(FusedKind::GatedLora, PARTIAL_TILE, true);
    run_fused_kernel_fixture(FusedKind::GatedLora, RANK_8, true);
}

#[cfg(feature = "cuda")]
#[test]
fn the_fused_ia3_kernel_matches_the_reference() {
    run_fused_kernel_fixture(FusedKind::Ia3, WHOLE_TILE, true);
    run_fused_kernel_fixture(FusedKind::Ia3, PARTIAL_TILE, true);
}

/// A shape the kernel cannot compute runs the ordinary operations, with a
/// notice, and is right.
#[cfg(feature = "cuda")]
#[test]
fn a_shape_beyond_one_tile_is_not_given_to_the_fused_kernel() {
    for kind in [FusedKind::Lora, FusedKind::GatedLora, FusedKind::Ia3] {
        run_fused_kernel_fixture(kind, SEVERAL_TILES, false);
    }
    run_fused_kernel_fixture(FusedKind::Lora, RANK_9, false);
    run_fused_kernel_fixture(FusedKind::GatedLora, RANK_9, false);
}

// ===== WRGA B.3.2 Option 3 (revised) phase 3e tests =====
//
// See `docs/superpowers/specs/2026-04-19-wrga-b32-option3-revised-design.md`
// §5 for discipline. Phase 3e extracts the adapter rewrite setup from
// `compile_user_functions` into `wrga_adapter_rewrite::rewrite_stmts_\
// for_model` and re-applies it to `compiler.models.model_method_bodies`
// immediately after prescan so source-AD's inline expansion walks the
// fused AST.
//
// Test 1 (proxy, launch-counter): asserts the fused kernel fires under a
//   train block, closing the regression "fused kernel never fires in
//   train block" observed pre-phase-3e.
// Test 2 (B.5 direct probe, 5/5 trainable params): `#[ignore]` at this
//   commit — source-AD still has no handler for the fused FFI callee
//   yet. Commit B (original option 3) removes the ignore.
// Test 3 (anti-drift): asserts the two paths (Cranelift + source-AD)
//   produce identical externally-visible behaviour.

/// Test 1 — proxy-level launch-counter (B.5 necessary-but-not-sufficient).
///
/// Catches the regression where 3e's rewrite pass doesn't fire (empty
/// `adapter_sites`, wrong insertion point, or the shared helper's
/// extraction introduced a bug). Launch-counter alone is not sufficient
/// (some other fusion-adjacent path could produce launches) — test 2
/// below is the direct probe; it is currently ignored because source-AD
/// has no handler yet for the fused FFI callee.
#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fused_fires_in_train_block() {
    let src = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = zeros([8, 8])
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = zeros([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
"#;

    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("fused_fires.nsl");
    fs::write(&src_path, src).unwrap();
    let root = workspace_root();
    let stdlib = root.join("stdlib");

    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .env("NSL_WRGA_GPU_LAUNCH_COUNTER", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path)
        .output()
        .expect("nsl run failed to spawn");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "nsl run failed:\n{stderr}");

    let launch_line = stderr
        .lines()
        .find(|l| l.contains("[nsl-gpu-launch-count]"))
        .expect("expected nsl-gpu-launch-count in stderr");
    let count: u64 = launch_line
        .split_whitespace()
        .next_back()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert!(
        count >= 3,
        "expected >=3 fused CUDA launches (1 per train epoch, 3 epochs); got {count}.\n\
         0 means the 3e rewrite didn't apply to model_method_bodies — regression.\nstderr:\n{stderr}",
    );
}

/// Test 2 — B.5-compliant direct probe (5/5 trainable tensor params).
///
/// Source-AD must report 5/5 trainable tensor params connected (x, W, A,
/// B, gate) when `@adapter(gatedlora)` is active in a train block. 5/5 is
/// a signal only the specific code path can produce — it requires the
/// Wengert extractor to have seen the fused FFI callee and decomposed its
/// 5 inputs.
///
/// IGNORED in commit A (phase 3e): at this commit source-AD has no handler
/// for `nsl_adapter_fused_gatedlora_matmul` yet, so it still falls back
/// and reports 1/1 (or less, if the fallback path also fails). Commit B
/// (original option 3) adds the extractor match arm + AD rule and removes
/// this ignore.
#[cfg(feature = "cuda")]
#[test]
fn source_ad_sees_fused_ffi_in_train_block() {
    let src = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = zeros([8, 8])
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = zeros([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
"#;

    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("source_ad_sees.nsl");
    fs::write(&src_path, src).unwrap();
    let root = workspace_root();
    let stdlib = root.join("stdlib");

    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path)
        .output()
        .expect("nsl run failed to spawn");

    let stderr = String::from_utf8_lossy(&out.stderr);
    // Expected stderr line from source-AD summary, e.g.
    //   "source AD gradient summary: 4/4 trainable tensor params connected"
    //
    // The B.5 direct-probe assertion is that every trainable model tensor
    // (base weight `W` + adapter-injected {lora_A, lora_B, gate}) reaches
    // source-AD and receives a gradient through the fused-FFI primal. For
    // the GatedLoRA setup here that's 4 params — `x` (a data Input) is not
    // counted as a param. The earlier "5/5" target in the draft spec was
    // an off-by-one; the math is {W, A, B, gate} = 4.
    //
    // The load-bearing invariant: `X/Y` where `Y == 4` (all adapter-
    // injected fields + base weight enumerated as trainable) and
    // `X == Y` (every one received a gradient). If `Y == 1`, source-AD
    // didn't see the fused-FFI inputs — the extractor or synth_member_names
    // plumbing regressed.
    let summary = stderr
        .lines()
        .find(|l| l.contains("trainable tensor params connected"))
        .expect("source AD gradient summary line missing in stderr");
    // Parse "X/Y trainable tensor params connected".
    let frac = summary
        .split_whitespace()
        .find(|w| w.contains('/'))
        .expect("no X/Y fraction in summary");
    let mut parts = frac.split('/');
    let connected: usize = parts.next().unwrap().parse().unwrap();
    let total: usize = parts.next().unwrap().parse().unwrap();
    assert!(
        total >= 4 && connected == total,
        "expected N/N >= 4/4 trainable params (W + lora_A + lora_B + gate) to \
         reach source-AD; got {connected}/{total}.\nstderr:\n{stderr}",
    );
}

/// Test 3 — anti-drift structural check.
///
/// If `compile_user_functions` and `rewrite_model_method_bodies_with_\
/// adapter_sites` use the same shared helper, both paths produce the
/// same rewritten stmts for the same model/method input. This test runs
/// the nsl binary on a program and asserts the externally-visible
/// numerical output is identical whether or not `--source-ad` is in
/// effect. If the two paths diverge (because the shared helper was
/// accidentally forked between the two call sites), the outputs would
/// differ.
///
/// Uses the fixture-A-style single-model GatedLoRA shape (same adapter
/// seeding that drives Build 4 / fixture A to y = 8 + 16 = 24.0 — we
/// only care that BOTH modes land on the same tensor value, not that it
/// matches any specific reference).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA GPU (anti-drift: Cranelift and source-AD see the same \
           rewritten AST). Parked as broken until the adapter side-table moved \
           into the model constructor; passes on sm_120 since then."]
fn cranelift_and_source_ad_see_same_rewritten_ast() {
    // We only use the `let y = m.forward(x)` output outside the train
    // block; both modes must produce the same answer. If the shared
    // helper has drifted between the Cranelift and 3e call sites,
    // `model_method_bodies` (source-AD) and `fn_def` (Cranelift) diverge
    // and the two runs produce different tensors.
    let src = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
m.lora_A_Toy_w__gatedlora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__gatedlora = ones([2, 8]).to(cuda)
m.gate_Toy_w__gatedlora = full([8], 30.0).to(cuda)
let y = m.forward(x)
print(y)
"#;

    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("same_ast.nsl");
    fs::write(&src_path, src).unwrap();
    let root = workspace_root();
    let stdlib = root.join("stdlib");

    let run = |with_source_ad: bool| -> String {
        let mut cmd = Command::cargo_bin("nsl").unwrap();
        cmd.env("NSL_STDLIB_PATH", &stdlib)
            .env("NSL_WRGA_FUSED_CUDA", "1")
            .arg("run")
            .args(["--target", "cuda_sm80"]);
        if with_source_ad {
            cmd.arg("--source-ad");
        }
        cmd.arg(&src_path);
        let out = cmd.output().expect("spawn");
        assert!(
            out.status.success(),
            "nsl run failed (source_ad={with_source_ad}):\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).to_string()
    };

    let stdout_with = run(true);
    let stdout_without = run(false);
    // Extract the final tensor rows from each; compare the trailing
    // contiguous block of `tensor(...)`-prefixed lines. This is robust
    // to extra diagnostic lines appearing before `print(y)`.
    let tensor_block = |s: &str| -> String {
        s.lines()
            .filter(|l| l.starts_with("tensor("))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let t_with = tensor_block(&stdout_with);
    let t_without = tensor_block(&stdout_without);
    assert!(
        !t_with.is_empty() && !t_without.is_empty(),
        "expected tensor output in both runs.\nwith: {stdout_with}\nwithout: {stdout_without}",
    );
    assert_eq!(
        t_with, t_without,
        "source-AD and non-source-AD paths produced different tensor output — \
         indicates model_method_bodies and Cranelift-compiled body have drifted.",
    );
}

// ─── WRGA B.3.2 Option 3 Commit B — numerical equivalence + regression ───

/// Test 4 — GatedLoRA fused-forward backward matches unfused reference.
///
/// Runs the same NSL program twice: once on `--target cuda_sm80` (source-AD
/// emits the `FusedGatedLoraMatmul` primal; the AD rule decomposes the
/// backward into primitive ops), once on `--target cuda_sm70` (the adapter
/// rewrite falls back to the unfused sigmoid+triple+scale expression and
/// the source-AD graph has separate Sigmoid/Mul/Matmul primals). After one
/// training step, compare weights element-wise.
///
/// Tolerance 1e-4 on post-training-step weight comparison.
///
///   lr = 1e-3 (pinned; see below).
///
///   Fused-forward output differs from unfused-forward by ≤ 2e-4 per
///   element (fused sigmoid ex2.approx + rcp.approx ULP ~9e-8, MMA
///   f16-accumulator noise floor ~2e-4; see B.3.1 Fixture D derivation
///   of `run_fused_kernel_fixture` in this file).
///
///   That per-element output difference ε_y ~ 2e-4 propagates into dy
///   with comparable magnitude, then into weight updates
///   Δw = -lr * dy * (...), giving weight-level discrepancy
///   ≈ lr * 2e-4 = 2e-7 per element at lr = 1e-3.
///
///   1e-4 tolerance therefore provides ~500× headroom at lr = 1e-3. If lr
///   is raised in a future refactor (e.g., to exercise harder gradient
///   signals), this tolerance needs recalibration: at lr = 1e-1 the
///   expected weight drift is ~2e-5, still inside 1e-4 but only 5×
///   headroom; at lr = 1.0 it is ~2e-4, right at the tolerance boundary.
#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fused_backward_matches_unfused_reference() {
    // Model: single GatedLoRA site at square shape (matches working B.3.1
    // fixture shape pattern). After one SGD step with lr=1e-3, read back
    // w / lora_A / lora_B / gate and compare fused (sm80) vs unfused (sm70)
    // element-wise.
    //
    // Shape rationale: w=[8,8] fits one output tile of the fused kernel.
    //
    // Initial adapter values: side-table init uses the thread-local seeded
    // RNG (seed=0) for lora_A's scaled normal draw, and deterministic ones/zeros
    // for everything else. Both sm80 and sm70 runs therefore see identical
    // adapter init, so weight-diffs at test time isolate the AD rule, not
    // initial-state drift.
    //
    // No pre-train reassignment of synth adapter fields: such assignments
    // null-deref because the side-table slot is only allocated at train
    // block entry (by emit_adapter_init_sidetable). Reassigning after the
    // train block is the working idiom used by the B.3.1 fixtures.
    let src = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
let w_out = m.w
print(w_out)
"#;

    let run = |target: &str| -> String {
        let tmp = TempDir::new().unwrap();
        let src_path = tmp.path().join("numeq.nsl");
        fs::write(&src_path, src).unwrap();
        let root = workspace_root();
        let stdlib = root.join("stdlib");
        let out = Command::cargo_bin("nsl")
            .unwrap()
            .env("NSL_STDLIB_PATH", &stdlib)
            .env("NSL_WRGA_FUSED_CUDA", "1")
            .arg("run")
            .args(["--source-ad", "--target", target])
            .arg(&src_path)
            .output()
            .expect("nsl run failed to spawn");
        assert!(
            out.status.success(),
            "nsl run failed (target={target}):\n{}",
            String::from_utf8_lossy(&out.stderr),
        );
        String::from_utf8_lossy(&out.stdout).to_string()
    };
    let stdout_fused = run("cuda_sm80");
    let stdout_unfused = run("cuda_sm70");
    // We compare the updated base weight `w` between the two paths.
    //
    // Comparing only `w` (not A/B/gate) is load-bearing here: `w`'s SGD
    // update is `w -= lr * dW`, and `dW` in the AD rule is
    // `x.T @ dy`. If the fused-forward's output differs from the unfused
    // forward by ε_y (as analysed above), that difference propagates into
    // dy with comparable magnitude and into `w` at `lr * ε_y ≈ 1e-3 * 2e-4
    // = 2e-7`. Any larger weight-diff here indicates a bug in either
    // (a) the fused FFI's forward math, or (b) the source-AD handler's
    // decomposition of the backward into primitive ops (Commit B). The
    // adapter-field tensors (A/B/gate) print as raw pointers through the
    // synth-field access path (separate pre-existing issue), so we do not
    // compare them here.
    let tensors_fused: Vec<_> = stdout_fused
        .lines()
        .filter(|l| l.trim_start().starts_with("tensor("))
        .collect();
    let tensors_unfused: Vec<_> = stdout_unfused
        .lines()
        .filter(|l| l.trim_start().starts_with("tensor("))
        .collect();
    assert_eq!(
        tensors_fused.len(),
        1,
        "expected 1 tensor print (w) in fused output; got {}.\n{stdout_fused}",
        tensors_fused.len(),
    );
    assert_eq!(tensors_unfused.len(), 1);
    let parse = |line: &str| -> Vec<f32> {
        let open = line.find('(').unwrap() + 1;
        let close = line.rfind(')').unwrap();
        let inner = &line[open..close];
        inner
            .chars()
            .filter(|c| !matches!(c, '[' | ']' | ' '))
            .collect::<String>()
            .split(',')
            .filter_map(|v| v.parse::<f32>().ok())
            .collect()
    };
    let fv = parse(tensors_fused[0]);
    let uv = parse(tensors_unfused[0]);
    assert_eq!(fv.len(), uv.len(), "w: shape differs between fused/unfused");
    let mut max_diff = 0.0_f32;
    for (i, (a, b)) in fv.iter().zip(uv.iter()).enumerate() {
        // `max_diff.max(nan)` keeps `max_diff`: refuse NaN before folding.
        assert!(
            a.is_finite() && b.is_finite(),
            "w post-step: non-finite entry at index {i} (fused {a}, unfused {b})"
        );
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff < 1e-4,
        "w post-step: max |fused-unfused| = {max_diff:.3e}, want < 1e-4.\n\
         Diagnostic: if ~1e-3 the AD rule scale is off; if ~1e-2 a \
         gradient term is missing; if ~1.0 the AD rule is wrong by a \
         full matmul.\nfused: {}\nunfused: {}",
        tensors_fused[0],
        tensors_unfused[0],
    );
}
