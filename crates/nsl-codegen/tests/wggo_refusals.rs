//! Refusal integration tests for WGGO Phase 2. Spec §5.4-§5.6.
//!
//! §5.3 (empty grad observations in the calibration subprocess) is a
//! harness-driver-internal check in `binary_codegen.rs`; it is covered by a
//! unit test in that module rather than here.
//!
//! All three refusals tested here are driven via `compile_with_options`, which
//! runs the full `populate_calibration_retention_from_ast_if_unset` seam
//! (including `run_pre_scan_phase` and `enforce_grad_mode_refusals`).  The
//! tests set `wggo_importance = WggoImportance::Grad` to arm the guard; in
//! `Auto` or `Magnitude` mode none of these refusals fire.

use nsl_codegen::{compile_with_options, CompileOptions, WggoImportance};

fn assert_error_contains(err_string: &str, fragment: &str) {
    assert!(
        err_string.contains(fragment),
        "expected error containing {fragment:?}; got:\n{err_string}"
    );
}

/// Build a `CompileOptions` with `wggo_importance = Grad` and no
/// `calibration_data`.  Used for §5.4 and §5.5 (both fail before reaching the
/// data check).
fn grad_mode_opts_no_data() -> CompileOptions {
    let mut opts = CompileOptions::default();
    opts.wggo_importance = WggoImportance::Grad;
    opts
}

/// Build a `CompileOptions` with `wggo_importance = Grad` and a dummy
/// `calibration_data` path.  The path does not need to exist for the refusal
/// tests — the §5.6 check only tests `is_some()`, not that the file is
/// readable.  Used for §5.6 (all preconditions met except data).
fn grad_mode_opts_with_dummy_data() -> CompileOptions {
    let mut opts = CompileOptions::default();
    opts.wggo_importance = WggoImportance::Grad;
    // Set a dummy path — the §5.6 check fires *before* any disk I/O.
    // We need a source that passes pre-scan (has decorators + reachable model)
    // but has no calibration data, so the §5.6 path fires.
    opts
}

// ── §5.4: grad mode with zero @wggo_target decorators ───────────────────────

const NO_DECORATOR_SOURCE: &str = r#"
model Plain:
    weight: Tensor = zeros([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x

fn main():
    let m = Plain(dim=4)
"#;

#[test]
fn refuse_grad_mode_with_no_decorators() {
    let opts = grad_mode_opts_no_data();
    let err = compile_with_options(NO_DECORATOR_SOURCE, &opts)
        .expect_err("must refuse when wggo_importance=Grad and no @wggo_target decorators");
    let msg = format!("{err}");
    assert_error_contains(&msg, "no @wggo_target decorators in source");
    assert_error_contains(&msg, "requested:");
    assert_error_contains(&msg, "expected:");
    assert_error_contains(&msg, "found:");
    assert_error_contains(&msg, "fix:");
}

// ── §5.5: grad mode with decorators present but model not reachable ──────────

#[test]
fn refuse_grad_mode_with_unreachable_decorators() {
    let source = include_str!("../../../tests/fixtures/orphaned_attention.nsl");
    let opts = grad_mode_opts_no_data();
    let err = compile_with_options(source, &opts).expect_err(
        "must refuse when wggo_importance=Grad and decorated model is not reachable from main",
    );
    let msg = format!("{err}");
    assert_error_contains(&msg, "no decorated model is reachable");
    assert_error_contains(&msg, "decorated classes:");
    assert_error_contains(&msg, "requested:");
    assert_error_contains(&msg, "expected:");
    assert_error_contains(&msg, "found:");
    assert_error_contains(&msg, "fix:");
}

// ── §5.6: grad mode, decorators present and reachable, but no calibration data

#[test]
fn refuse_grad_mode_with_no_calibration_data() {
    let source = include_str!("../../../tests/fixtures/wggo_attention_mlp.nsl");
    // opts has wggo_importance=Grad but calibration_data=None.
    let opts = grad_mode_opts_with_dummy_data();
    // calibration_data is still None here — that is the trigger for §5.6.
    assert!(
        opts.calibration_data.is_none(),
        "test must start with no calibration_data"
    );
    let err = compile_with_options(source, &opts).expect_err(
        "must refuse when wggo_importance=Grad and no calibration data provided",
    );
    let msg = format!("{err}");
    assert_error_contains(&msg, "requires calibration data, but none was provided");
    assert_error_contains(&msg, "requested:");
    assert_error_contains(&msg, "expected:");
    assert_error_contains(&msg, "found:");
    assert_error_contains(&msg, "fix:");
}

// ── Negative: Auto mode must NOT trigger any refusal even when no decorators ─

#[test]
fn auto_mode_does_not_refuse_without_decorators() {
    let mut opts = CompileOptions::default();
    opts.wggo_importance = WggoImportance::Auto;
    // compile_with_options may fail for other reasons (codegen on a minimal
    // fixture) but must NOT fail with the §5.4 refusal message.
    if let Err(e) = compile_with_options(NO_DECORATOR_SOURCE, &opts) {
        let msg = format!("{e}");
        assert!(
            !msg.contains("no @wggo_target decorators in source"),
            "Auto mode must not trigger the §5.4 refusal; got: {msg}"
        );
    }
}

// ── Negative: Magnitude mode must NOT trigger any refusal ────────────────────

#[test]
fn magnitude_mode_does_not_refuse_without_decorators() {
    let mut opts = CompileOptions::default();
    opts.wggo_importance = WggoImportance::Magnitude;
    if let Err(e) = compile_with_options(NO_DECORATOR_SOURCE, &opts) {
        let msg = format!("{e}");
        assert!(
            !msg.contains("no @wggo_target decorators in source"),
            "Magnitude mode must not trigger the §5.4 refusal; got: {msg}"
        );
    }
}
