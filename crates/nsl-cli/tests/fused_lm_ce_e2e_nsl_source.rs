//! CFTP v7 follow-on — `@fused_lm_ce(dtype = "fp16" | "bf16")` true
//! NSL-source end-to-end validation via the `nsl` CLI subprocess.
//!
//! ## What the orchestrator asked for
//!
//! Verify that real NSL source with `@fused_lm_ce(dtype="fp16")` and
//! `@fused_lm_ce(dtype="bf16")` compiled via `nsl run` produces correct
//! loss / grad on GPU — no env-var workarounds, no direct-FFI bypass,
//! the actual user-facing path.
//!
//! ## What this test actually validates today
//!
//! Subprocess-running `nsl check` (which goes parse → semantic →
//! `AnalysisResult.fused_ce_configs` capture):
//!
//!   * **fp16 fixture** parses, type-checks, and the decorator validates.
//!   * **bf16 fixture** parses, type-checks, and the decorator validates.
//!   * **`dtype="int8"` fixture** is REFUSED with the v5/v6 diagnostic
//!     `@fused_lm_ce: dtype 'int8' not recognised; accepted: "f32",
//!     "fp32", "f16", "fp16", "bf16"`.  A silent dtype-accept regression
//!     would flip this.
//!   * **non-`train`-block fixture** is REFUSED with the v3-3 diagnostic
//!     `@fused_lm_ce may only be applied to a `train` block`.  A silent
//!     decorator-target-loosen regression would flip this.
//!
//! Subprocess-running `nsl run` on the fp16 fixture used to document the
//! load-bearing TRAIN-LOOP BITROT that prevented the full loss-print the
//! orchestrator originally targeted (`codegen error: undefined function
//! 'nsl_optim_sgd__sgd_step'`, fired from `crates/nsl-codegen/src/stmt.rs`
//! during train-block lowering). That FFI registration bitrot is now
//! fixed as a side effect of the CPKD `discover_optimizer_modules` stdlib-
//! loader refactor (it now loads `nsl.optim.*` modules for plain `train`
//! blocks, not just `distill` blocks) — `nsl run` on the fp16 fixture
//! compiles, links, and executes to completion, printing a finite CPU
//! loss. Fixing that surfaced a second, independent bug the bitrot had
//! been masking: `x`/`targets` in the fp16/bf16 fixtures were unflattened
//! `[B, S, H]` / `[B, S]` tensors, and the naive composite fallback path
//! (`bias_add` + `cross_entropy`, taken whenever `@fused_lm_ce` doesn't
//! substitute a fused kernel call) requires flattened `[rows, H]` /
//! `[rows]` input — see `cpkd_distill_basic.nsl`. A raw 3D `x` made
//! `nsl_tensor_bias_add` (`crates/nsl-runtime/src/tensor/mod.rs`)
//! `std::process::abort()` the whole process with "bias_add requires 2D
//! tensor (got 3D)" the moment execution reached it. Fixtures now use
//! `x = randn([64, 128])` / `targets = zeros([64])` (rows = B*S = 64).
//!
//! ## Why this is the right scope today
//!
//! The strict numerical contract the orchestrator wants ("rel_err <= 5e-3
//! for fp16, <= 2e-2 for bf16") is pinned at GPU level (CUDA-gated,
//! unavailable in this environment) in
//!
//!   * `crates/nsl-codegen/tests/fused_lm_ce_e2e_fp16_activation.rs`
//!     Phase 2 (`fp16_decorator_e2e_loss_matches_cpu_f64_reference`)
//!   * `crates/nsl-codegen/tests/fused_lm_ce_e2e_bf16_activation.rs`
//!     Phase 2 (`bf16_decorator_e2e_loss_matches_cpu_f64_reference`)
//!
//! Those Phase-2 tests drive the SAME FFI (`nsl_fused_linear_ce_forward`)
//! the wengert-lowered train block would dispatch to, with the SAME
//! dtype_tag sentinels (1 for fp16, 2 for bf16), the SAME B/S/V/H shape,
//! and the SAME CPU f64 reference helper — that is still the authoritative
//! numerical contract. `nsl_run_produces_finite_loss` below is a coarser
//! sanity check on the true `nsl run` user-facing path (parse → semantic
//! → wengert lower → link → execute → stdout), not a replacement for it:
//! it only asserts the process exits 0 and prints a finite loss.
//!
//! ## What this test does NOT cover
//!
//! A hard numerical pin of the `nsl run` stdout loss against the CPU f64
//! reference. Wiring that up (matching this fixture's B/S/V/H/dtype_tag
//! against `common::fused_lce_cpu_f64`) is a reasonable follow-on now that
//! the train-loop bitrot is gone, but is left to the GPU-gated Phase-2
//! tests above, which already own that contract on the fused-kernel path.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::path::PathBuf;
use std::process::Command;

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

fn fixture_path(name: &str) -> PathBuf {
    workspace_root()
        .join("crates")
        .join("nsl-codegen")
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// `nsl check` on the fp16 fixture must succeed — semantic checker
/// accepts `dtype="fp16"` and captures the `@fused_lm_ce` config into
/// `AnalysisResult.fused_ce_configs`.
#[test]
fn nsl_check_accepts_fp16_decorator_on_train_block() {
    let fixture = fixture_path("fused_lm_ce_e2e_fp16.nsl");
    assert!(fixture.exists(), "fp16 fixture missing: {}", fixture.display());

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&fixture);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("checked successfully"));
}

/// `nsl check` on the bf16 fixture must succeed — semantic checker
/// accepts `dtype="bf16"` (v4 emitter dtype) and captures the config.
#[test]
fn nsl_check_accepts_bf16_decorator_on_train_block() {
    let fixture = fixture_path("fused_lm_ce_e2e_bf16.nsl");
    assert!(fixture.exists(), "bf16 fixture missing: {}", fixture.display());

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&fixture);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("checked successfully"));
}

/// `nsl check` on an unrecognised dtype string must fail with the exact
/// v5/v6 diagnostic — pinning the message catches a silent regression
/// that would let an invalid dtype fall through to F32.
#[test]
fn nsl_check_refuses_unrecognised_dtype_string() {
    let fixture = fixture_path("fused_lm_ce_e2e_bad_dtype.nsl");
    assert!(
        fixture.exists(),
        "bad_dtype fixture missing: {}",
        fixture.display()
    );

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&fixture);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains(
            "@fused_lm_ce: dtype 'int8' not recognised",
        ))
        // The accepted-dtype list pinning catches any future expansion
        // that drops bf16 from the supported set.
        .stderr(predicate::str::contains("\"fp16\""))
        .stderr(predicate::str::contains("\"bf16\""));
}

/// `nsl check` on a `@fused_lm_ce` attached to a non-`train` construct
/// must fail with the v3-3 diagnostic — pinning the message catches a
/// silent regression that would let the decorator apply to arbitrary
/// statements.
#[test]
fn nsl_check_refuses_non_train_block_target() {
    let fixture = fixture_path("fused_lm_ce_e2e_not_train.nsl");
    assert!(
        fixture.exists(),
        "not_train fixture missing: {}",
        fixture.display()
    );

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&fixture);
    cmd.assert().failure().stderr(predicate::str::contains(
        "@fused_lm_ce may only be applied to a `train` block",
    ));
}

/// `nsl run` on the fp16 fixture now runs the full user-facing path to
/// completion: the `nsl_optim_sgd__sgd_step` FFI bitrot this test used to
/// pin is fixed (CPKD's `discover_optimizer_modules` stdlib-loader
/// refactor loads `nsl.optim.*` for plain `train` blocks too), and the
/// fixture's `x`/`targets` are flattened to `[rows, H]` / `[rows]` so the
/// composite `bias_add` + `cross_entropy` fallback no longer aborts on a
/// 3D tensor (see the module doc comment for both root causes). This test
/// is a coarse sanity check, not the numerical contract — see
/// `fused_lm_ce_e2e_fp16_activation.rs` Phase 2 for the CUDA-gated
/// rel_err pin against the CPU f64 reference.
#[test]
fn nsl_run_produces_finite_loss() {
    let fixture = fixture_path("fused_lm_ce_e2e_fp16.nsl");
    assert!(fixture.exists(), "fp16 fixture missing: {}", fixture.display());

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("run").arg(&fixture);
    let output = cmd.assert().success().get_output().stdout.clone();
    let stdout = String::from_utf8(output).expect("stdout must be valid UTF-8");
    let loss: f64 = stdout
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("expected a bare finite loss on stdout, got {stdout:?}: {e}"));
    assert!(
        loss.is_finite(),
        "train step loss must be finite, got {loss}"
    );
}
