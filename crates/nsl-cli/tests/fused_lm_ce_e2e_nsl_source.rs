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
//! Subprocess-running `nsl run` on the fp16 fixture documents the
//! load-bearing TRAIN-LOOP BITROT that prevents the full GPU loss-print
//! the orchestrator originally targeted:
//!
//!   * `codegen error: undefined function 'nsl_optim_sgd__sgd_step'`
//!
//! This refusal fires from `crates/nsl-codegen/src/stmt.rs` during train-
//! block lowering — after the `@fused_lm_ce` semantic gate and after
//! wengert auto-substitution.  It is unrelated to the v7 GPU PTX cast
//! kernel and would block ANY train block, decorator-bearing or not.
//! v3-3 / project_cftp_v3_substitution.md tracks the optimizer-step FFI
//! registration as an open architectural follow-on.
//!
//! ## Why this is the right scope today
//!
//! The numerical contract the orchestrator wants ("rel_err <= 5e-3 for
//! fp16, <= 2e-2 for bf16") is ALREADY pinned at GPU level in
//!
//!   * `crates/nsl-codegen/tests/fused_lm_ce_e2e_fp16_activation.rs`
//!     Phase 2 (`fp16_decorator_e2e_loss_matches_cpu_f64_reference`)
//!   * `crates/nsl-codegen/tests/fused_lm_ce_e2e_bf16_activation.rs`
//!     Phase 2 (`bf16_decorator_e2e_loss_matches_cpu_f64_reference`)
//!
//! Those Phase-2 tests drive the SAME FFI (`nsl_fused_linear_ce_forward`)
//! the wengert-lowered train block would dispatch to, with the SAME
//! dtype_tag sentinels (1 for fp16, 2 for bf16), the SAME B/S/V/H shape,
//! and the SAME CPU f64 reference helper.  The compile-side path from
//! NSL source through `compile_wengert_ops` is pinned by the same files'
//! Phase 1 IR proofs.
//!
//! Once `nsl_optim_sgd__sgd_step` FFI is registered, the existing fp16/
//! bf16 fixtures here become drop-in inputs for the full GPU loss-print
//! comparison — no fixture rewrites required.
//!
//! ## What this test does NOT cover (deferral)
//!
//! Actual `nsl run` execution → stdout-loss capture → CPU f64 compare.
//! Deferred until train-block optimizer FFI bitrot is fixed.  The
//! `nsl_run_documents_train_loop_bitrot` test guards this deferral with
//! a stderr substring check so an unexpected FIX to the train-block path
//! flips a CI signal — that is the trigger to extend this file with the
//! orchestrator's loss-print numerical compare.

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

/// `nsl run` on the fp16 fixture MUST currently bottom out in the
/// optimizer-step FFI bitrot (`nsl_optim_sgd__sgd_step` undefined).
/// This test DOCUMENTS the deferral the v7 sprint inherits and arms a
/// CI signal: if the train-block optimizer-FFI registration is ever
/// fixed, this test will flip from passing to failing — and that is the
/// trigger to extend this file with the orchestrator's true `nsl run`
/// loss-print numerical compare (parse stdout → compare against the
/// CPU f64 reference in `crates/nsl-codegen/tests/common/
/// fused_lce_cpu_f64.rs`).
///
/// CFTP v3-3 `project_cftp_v3_substitution.md` tracks the optimizer-
/// step FFI registration as an open architectural follow-on.  Until
/// that lands, the IR + GPU numerical contracts are pinned via
///   * `crates/nsl-codegen/tests/fused_lm_ce_e2e_fp16_activation.rs`
///   * `crates/nsl-codegen/tests/fused_lm_ce_e2e_bf16_activation.rs`
/// (decorator → wengert lower → 3× precision_cast → fused_linear_ce
/// FFI with the right dtype_tag → CUDA-gated GPU loss vs CPU f64).
#[test]
fn nsl_run_documents_train_loop_bitrot() {
    let fixture = fixture_path("fused_lm_ce_e2e_fp16.nsl");
    assert!(fixture.exists(), "fp16 fixture missing: {}", fixture.display());

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("run").arg(&fixture);
    cmd.assert().failure().stderr(predicate::str::contains(
        "nsl_optim_sgd__sgd_step",
    )).stderr(predicate::str::contains("undefined function"));
}
