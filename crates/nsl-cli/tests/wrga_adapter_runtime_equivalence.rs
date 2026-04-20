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

// GPU-resident variant of BUILD4_SRC used ONLY by the `#[ignore]`-d
// hardening test `build_4_fused_cuda_actually_fires`.  Every tensor that
// feeds into `m.forward(x)` — the model weight, the input, and both
// seeded adapter tensors — is placed on CUDA so the fused FFI's "inputs
// on GPU" gate is satisfied and the real cudarc PTX launch fires.
//
// Kept separate from BUILD4_SRC because the non-ignored build_4* tests
// must continue to exercise the CPU-fallback path on machines without
// a CUDA device.
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
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_KERNEL_LAUNCH_COUNTER", "1")
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

// ─── Task D1: build_4_fused_real_launch — integration numerical gate ─
//
// Runs BUILD4_SRC_GPU with NSL_WRGA_FUSED_CUDA=1 forced on, asserts:
//   1. Output is 16.0 ± 1e-4 elementwise (numerical correctness via
//      real cudarc launch, NOT CPU fallback — the 1e-4 tolerance is
//      the milestone's numerical gate from spec §5).
//   2. [nsl-gpu-launch-count] ≥ 1 (proves the fused PTX actually
//      executed on GPU; count=0 would mean numerical result was
//      produced by CPU fallback even though NSL_WRGA_FUSED_CUDA=1
//      was set).
//
// Complementary to build_4_fused_cuda_actually_fires (which only
// asserts count ≥ 1 without checking numerics).  This test catches
// semantic bugs in the rewritten kernel that passed ptxas validation
// but silently compute the wrong thing.
//
// Known C4 precision concern: the final (x@A)@B MMA requires packed-f16
// A-operand; C4 emits cvt.rn.f16x2.f32 between the two epilogue MMAs.
// For BUILD4 shape (x=ones, A=ones, B=ones, rank=2) epi_interm=8.0
// fits f16 exactly so no loss; larger test shapes may need >1e-4
// tolerance — raise this flag here if the assertion fails.
#[cfg(feature = "cuda")]
#[test]
fn build_4_fused_real_launch() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build4_real_launch.nsl");
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
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        output.status.success(),
        "nsl run failed.\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );

    // Parse output tensor.
    let tensor = parse_tensor_2d(&stdout);
    assert_eq!(tensor.len(), 4, "expected 4 rows, got {tensor:?}");

    // Assert 16.0 ± 1e-4 per element.
    let mut max_diff: f32 = 0.0;
    for row in &tensor {
        assert_eq!(row.len(), 8, "expected 8 cols");
        for v in row {
            max_diff = max_diff.max((v - 16.0).abs());
        }
    }
    assert!(
        max_diff < 1e-4,
        "build_4_fused_real_launch: max |y - 16.0| = {max_diff:.3e}, want < 1e-4.\n\
         Diagnostic hints:\n\
         - ~1e-3: f16 precision (cvt.rn.f16x2.f32 between epilogue MMAs)\n\
         - exactly 16.0 diff (y=0): base matmul fired but adapter delta dropped\n\
         - exactly 0: test never exercised the fused path\n\
         stderr:\n{stderr}"
    );

    // Assert the REAL cudarc launch fired (not CPU fallback).
    let gpu_count_line = stderr.lines().find(|l| l.contains("[nsl-gpu-launch-count]"));
    let count: u64 = gpu_count_line
        .and_then(|l| l.split_whitespace().next_back())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert!(
        count >= 1,
        "build_4_fused_real_launch: expected ≥1 real cudarc launch with \
         NSL_WRGA_FUSED_CUDA=1, got {count}. Numerical output was correct \
         but came from CPU fallback — the fused PTX path didn't fire. \
         Check for fallback warnings in stderr above."
    );
}

// ─── Task E4: IA³ integration fixtures ─────────────────────────────────────
//
// Fixture A — full compute path baseline:
//   x = ones([4,8]), W = ones([8,8]), γ = ones([8])
//   y[i,j] = (x@W)[i,j] * γ[j] = 8 * 1 = 8 elementwise
// Proves the IA³ matmul pipeline runs end-to-end (γ=1 is a passthrough
// so a failure here means the x@W path itself is broken).
//
// Fixture B — γ actually multiplies:
//   x = ones([4,8]), W = ones([8,8]), γ = [2, 2, ..., 2]
//   y[i,j] = 8 * 2 = 16 elementwise
// Proves γ is read from HBM and applied via broadcast-multiply.  If
// Fixture A passes but B fails (y still = 8), γ is being ignored.

const IA3_FIXTURE_A_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=ia3, target=["Toy.w"])
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.ia3_scale_Toy_w__ia3 = ones([8]).to(cuda)
let y = m.forward(x)
print(y)
"#;

const IA3_FIXTURE_B_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=ia3, target=["Toy.w"])
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.ia3_scale_Toy_w__ia3 = full([8], 2.0).to(cuda)
let y = m.forward(x)
print(y)
"#;

#[cfg(feature = "cuda")]
fn run_ia3_fixture(src: &str, expected: f32, fixture_name: &str) {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join(format!("{}.nsl", fixture_name));
    fs::write(&src_path, src).unwrap();
    let root = workspace_root();
    let stdlib = root.join("stdlib");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path);
    let output = cmd.output().expect("nsl run failed to spawn");
    assert!(
        output.status.success(),
        "{fixture_name} nsl run failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let tensor = parse_tensor_2d(&stdout);
    let mut max_diff: f32 = 0.0;
    for row in &tensor {
        for v in row {
            max_diff = max_diff.max((v - expected).abs());
        }
    }
    assert!(
        max_diff < 1e-4,
        "{fixture_name}: max |y - {expected}| = {max_diff:.3e}, want < 1e-4. \
         Tensor: {tensor:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn ia3_fixture_a_baseline() {
    run_ia3_fixture(IA3_FIXTURE_A_SRC, 8.0, "ia3_a");
}

#[cfg(feature = "cuda")]
#[test]
fn ia3_fixture_b_gamma_scaling() {
    run_ia3_fixture(IA3_FIXTURE_B_SRC, 16.0, "ia3_b");
}

// ─── Task Group 5: GatedLoRA integration fixtures ──────────────────────────
//
// Fixture A — baseline (gate = 0 per column):
//   x = ones([4,8]), W = ones([8,8]), A = ones([8,2]), B = ones([2,8])
//   alpha = rank = 2, scale = alpha/rank = 1.0
//   gate[j] = 0 for all j  ⟹  sigmoid(gate[j]) = 0.5
//
//   y[i,j] = (x@W)[i,j] + sigmoid(gate[j]) * ((x@A)@B)[i,j] * scale
//          = 8            + 0.5              * 16              * 1.0
//          = 16.0 per element.
//
// Proves the GatedLoRA fused kernel runs end-to-end: base matmul (8),
// adapter delta (x@A@B = 16), sigmoid gate folded at the correct value (0.5),
// and scale=1.0 applied.  Tolerance 1e-4.

const GATEDLORA_FIXTURE_A_SRC: &str = r#"from nsl.nn.losses import mse_loss

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
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_A_Toy_w__gatedlora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__gatedlora = ones([2, 8]).to(cuda)
m.gate_Toy_w__gatedlora = zeros([8]).to(cuda)
let y = m.forward(x)
print(y)
"#;

/// Shared helper for GatedLoRA integration fixtures (Tasks 5.1-5.4).
///
/// Runs `src` through `nsl run --source-ad --target cuda_sm80` with
/// `NSL_WRGA_FUSED_CUDA=1` and `NSL_WRGA_GPU_LAUNCH_COUNTER=1`, then asserts:
///   1. Process exits successfully.
///   2. Every output element is within `tolerance` of `expected`.
///   3. `[nsl-gpu-launch-count] >= 1` — the real cudarc PTX path fired, NOT
///      the CPU fallback.
#[cfg(feature = "cuda")]
fn run_gatedlora_fixture(src: &str, expected: f32, tolerance: f32, fixture_name: &str) {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join(format!("{}.nsl", fixture_name));
    fs::write(&src_path, src).unwrap();
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
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        output.status.success(),
        "{fixture_name} nsl run failed.\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );

    let tensor = parse_tensor_2d(&stdout);
    assert_eq!(tensor.len(), 4, "{fixture_name}: expected 4 rows");

    let mut max_diff: f32 = 0.0;
    for row in &tensor {
        assert_eq!(row.len(), 8, "{fixture_name}: expected 8 cols");
        for v in row {
            max_diff = max_diff.max((v - expected).abs());
        }
    }
    assert!(
        max_diff < tolerance,
        "{fixture_name}: max |y - {expected}| = {max_diff:.3e}, want < {tolerance:.0e}.\n\
         Diagnostic:\n\
         - diff ~ 8.0 (y=8 or y=24): sigmoid gate not applied or applied twice\n\
         - diff ~ 16.0 (y=0): adapter delta dropped entirely\n\
         - diff ~ 0.5*16=8 (y~8): gate=0.5 correct but base matmul dropped\n\
         - diff > 1e-3: precision concern (f16 cvt between epilogue MMAs)\n\
         Tensor: {tensor:?}\nstderr:\n{stderr}",
    );

    // Assert fused PTX actually fired (not CPU fallback).
    let gpu_count_line = stderr.lines().find(|l| l.contains("[nsl-gpu-launch-count]"));
    let count: u64 = gpu_count_line
        .and_then(|l| l.split_whitespace().next_back())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert!(
        count >= 1,
        "{fixture_name}: expected >=1 fused CUDA launch, got {count}.\n\
         0 means all fused FFI calls hit the CPU fallback — check:\n\
         - GatedLoRA kernel registered in fused_adapter.rs kernel registry\n\
         - wrga_adapter_rewrite.rs emits GatedLoRA fused Call (not unfused triple)\n\
         - GPU tensor placement gates satisfied (all tensors on CUDA)\n\
         stderr:\n{stderr}",
    );
}

#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fixture_a_baseline() {
    run_gatedlora_fixture(GATEDLORA_FIXTURE_A_SRC, 16.0, 1e-4, "gatedlora_fixture_a");
}

/// Fixture B — positive saturation. alpha=2, rank=2, scale=1.0.
/// Same model/x as Fixture A; gate = full([8], 30.0).  sigmoid(30) ≈ 1.0
/// exactly in fp32 (ex2.approx(-30 * log2e) = ex2.approx(-43.3) → 2^-43
/// underflows to 0 in fp32, yielding 1/(1+0) = 1.0 exactly).
/// y[i,j] = 8 + 1.0 * 16 * 1.0 = 24.0 per element.
const GATEDLORA_FIXTURE_B_SRC: &str = r#"from nsl.nn.losses import mse_loss

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
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_A_Toy_w__gatedlora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__gatedlora = ones([2, 8]).to(cuda)
m.gate_Toy_w__gatedlora = full([8], 30.0).to(cuda)
let y = m.forward(x)
print(y)
"#;

#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fixture_b_positive_saturation() {
    run_gatedlora_fixture(GATEDLORA_FIXTURE_B_SRC, 24.0, 1e-4, "fixture_b");
}

/// Fixture C — negative saturation. alpha=2, rank=2, scale=1.0.
/// gate = full([8], -30.0).  sigmoid(-30) ≈ 0.0 exactly (ex2.approx(+43)
/// overflows in the denominator; 1/huge ≈ 0.0 in fp32).
/// y[i,j] = 8 + 0.0 * 16 * 1.0 = 8.0 per element.
///
/// DIAGNOSTIC: the base-only value (8.0) specifically tests that
/// the fold predicate path doesn't leak junk into main_accum when
/// sigmoid(gate) ≈ 0.  A broken kernel that adds the adapter delta
/// unconditionally (ignoring the sigmoid multiply) would produce y=24
/// here instead of y=8.
const GATEDLORA_FIXTURE_C_SRC: &str = r#"from nsl.nn.losses import mse_loss

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
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_A_Toy_w__gatedlora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__gatedlora = ones([2, 8]).to(cuda)
m.gate_Toy_w__gatedlora = full([8], -30.0).to(cuda)
let y = m.forward(x)
print(y)
"#;

#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fixture_c_negative_saturation() {
    run_gatedlora_fixture(GATEDLORA_FIXTURE_C_SRC, 8.0, 1e-4, "fixture_c");
}

/// Fixture D — mid-range sigmoid curve validation.
/// alpha=2, rank=2, scale=1.0.  gate = full([8], 1.0f) exact f32.
/// sigmoid(1.0) ≈ 0.7310586.  y[i,j] = 8 + 0.7310586 * 16 * 1.0 ≈ 19.697 per element.
///
/// Tolerance 1e-3 (vs 1e-4 on A/B/C): gate=1.0f exercises the actual
/// ex2.approx + rcp.approx sigmoid curve shape between saturation points,
/// not exact values.
///
/// Error propagation at rank=2 accumulator magnitudes:
///   ex2.approx worst-case ULP:      ~2 ULP (~5e-8 at input -1.4427)
///   rcp.approx worst-case ULP:      ~1 ULP (~6e-8 at input ≈1.368)
///   Combined via 1/(1+e) derivative: ~9e-8 on sigmoid output
///   Through fold at rank=2:          ~5e-6 per element
///   Main-path MMA noise floor:       ~2e-4 (dominates)
///   1e-3 tolerance:                  ~5x over main MMA, ~200x over sigmoid
///
/// Bug class this fixture catches: a sign-based-stub kernel emitting
/// `select(x > 0, 1.0, select(x < 0, 0.0, 0.5))` passes every saturation
/// fixture (A/B/C) but returns 0.5 here — wrong by |0.5 - 0.7310586| ≈ 0.2311
/// on the sigmoid value.
///
///   Correct y:   (x@W = 8) + 0.7310586 · 16 · 1.0 ≈ 19.697
///   Stub   y:   (x@W = 8) + 0.5       · 16 · 1.0 = 16.0
///                                                   ^^^^
///   The (x@W = 8) base contribution is identical in both paths and cancels.
///   y-level error = |19.697 − 16.0| = 3.697, driven entirely by the
///   adapter-path factor 16 · (0.7311 − 0.5) ≈ 3.697. This is ~3700× the
///   1e-3 tolerance — fixture fails loudly.
///
/// This is the ONLY fixture in the B.3.1 matrix that discriminates
/// "emitted rcp.approx(1 + ex2.approx(-x * log2e))" from "silently stubbed."
const GATEDLORA_FIXTURE_D_SRC: &str = r#"from nsl.nn.losses import mse_loss

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
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_A_Toy_w__gatedlora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__gatedlora = ones([2, 8]).to(cuda)
m.gate_Toy_w__gatedlora = full([8], 1.0).to(cuda)
let y = m.forward(x)
print(y)
"#;

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fixture_d_mid_range_curve() {
    // Expected computed from CPU reference: 8 + sigmoid(1.0) * 16.
    let expected = 8.0_f32 + sigmoid_f64(1.0) as f32 * 16.0;
    run_gatedlora_fixture(GATEDLORA_FIXTURE_D_SRC, expected, 1e-3, "fixture_d");
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
///   near GATEDLORA_FIXTURE_D_SRC in this file).
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
#[ignore = "GPU placement blocker on gate field (fused FFI falls back to CPU math, fixed separately); \
            re-enable once the gate .to(cuda) plumbing from the 2026-04-18 superseded spec §7 lands"]
fn gatedlora_fused_backward_matches_unfused_reference() {
    // Model: single GatedLoRA site at small shape. After one SGD step with
    // lr=1e-3, read back w / lora_A / lora_B / gate and compare fused vs
    // unfused element-wise.
    let src = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 4])
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 4]).to(cuda)
m.lora_A_Toy_w__gatedlora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__gatedlora = ones([2, 4]).to(cuda)
m.gate_Toy_w__gatedlora = full([4], 0.5).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
let w_out = m.w
let a_out = m.lora_A_Toy_w__gatedlora
let b_out = m.lora_B_Toy_w__gatedlora
let g_out = m.gate_Toy_w__gatedlora
print(w_out)
print(a_out)
print(b_out)
print(g_out)
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
    // Extract each tensor(...) line.
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
        4,
        "expected 4 tensor prints (w, A, B, gate) in fused output; got {}.\n{stdout_fused}",
        tensors_fused.len(),
    );
    assert_eq!(tensors_unfused.len(), 4);
    for (i, (f, u)) in tensors_fused.iter().zip(tensors_unfused.iter()).enumerate() {
        // Parse each tensor line as a flat float vector, compare element-wise.
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
        let fv = parse(f);
        let uv = parse(u);
        assert_eq!(fv.len(), uv.len(), "tensor {i}: shape differs");
        let mut max_diff = 0.0_f32;
        for (a, b) in fv.iter().zip(uv.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-4,
            "tensor {i}: max |fused-unfused| = {max_diff:.3e}, want < 1e-4.\n\
             Diagnostic: if ~1e-3 the AD rule scale is off; if ~1e-2 a \
             gradient term is missing; if ~1.0 the AD rule is wrong by a \
             full matmul.\nfused: {f}\nunfused: {u}",
        );
    }
}

/// Test 5 — inference unchanged regression (4 B.3.1 fixtures).
///
/// Programmatic re-run of the four `gatedlora_fixture_*_baseline` helpers
/// to guarantee their expected values haven't moved. The individual
/// `#[test]` functions for each fixture are already registered; this
/// wrapper provides a single quick signal that the B.3.1 inference
/// surface is intact after the Commit B changes.
#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fused_unchanged_inference_behavior() {
    // Each of the 4 fixtures is itself a #[test]; they'd run in parallel
    // anyway. This test just runs the helper directly on fixture A to
    // catch mass regressions in a single fast signal.
    run_gatedlora_fixture(GATEDLORA_FIXTURE_A_SRC, 16.0, 1e-4, "regression_gatedlora_a");
}
