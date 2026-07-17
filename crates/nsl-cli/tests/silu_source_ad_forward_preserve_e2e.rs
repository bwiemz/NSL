//! End-to-end regression for the source-AD forward in-place-mutation fix.
//!
//! Source-AD builds no tape, so it lacked tape-AD's `is_recording()` guard that
//! blocks FBIP in-place mutation. A forward activation on a uniquely-owned input
//! (`silu(x)` in `grad(x): sum(silu(x))`) therefore overwrote `x` with `silu(x)`,
//! and the input-reading `SiluBackward` adjoint then computed `silu'(silu(x))` —
//! silently wrong. The fix raises an in-place-suppression guard
//! (`nsl_set_inplace_suppressed`) around the source-AD forward pass so the primal
//! survives.
//!
//! The fixture asserts BOTH that `dx == silu'(x)` (the gradient is correct, not
//! `silu'(silu(x))`) AND that `x` is byte-identical afterward (the forward never
//! mutated it). A pass proves the fix end to end. Also guards the scalar
//! `sum`-seed broadcast path in `nsl_tensor_silu_backward` (grad `[1]` vs `x[n]`).

use std::process::Command;

fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn e2e_silu_source_ad_preserves_forward_input() {
    let root = workspace_root();
    let example = root.join("examples/silu_source_ad_grad.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run", "--source-ad"])
        .arg(&example)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run --source-ad");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Nonzero exit means either a compile error or an in-program `assert_close`
    // abort (wrong gradient, or x was mutated by the forward).
    assert!(
        output.status.success(),
        "silu source-AD forward-preserve e2e failed (exit {:?}).\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert!(
        stdout.contains("silu-source-ad-grad-preserves-input-ok"),
        "expected 'silu-source-ad-grad-preserves-input-ok' in stdout, got:\nstdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );
    // The source-AD path must actually be used — not silently fall back to tape
    // via any of the extractor/resolver bailouts (all of which would still give
    // the correct tape result and mask a regression). Assert the positive marker
    // the grad-block emitter prints, plus the absence of the extraction bailout.
    assert!(
        stderr.contains("Using source-to-source AD for grad block"),
        "expected source-AD to handle the grad block (missing marker); it fell back to tape.\nstderr:\n{}",
        stderr
    );
    assert!(
        !stderr.contains("falling back to tape-based AD"),
        "source AD bailed out to tape AD; the fix was not exercised.\nstderr:\n{}",
        stderr
    );
}

/// Run `nsl run [--source-ad] <example>` and parse the last float printed to
/// stdout (the model's `sum(m.w)` after one training step).
fn run_and_parse_sum(root: &std::path::Path, example: &std::path::Path, source_ad: bool) -> f64 {
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "-p", "nsl-cli", "--", "run"]);
    if source_ad {
        cmd.arg("--source-ad");
    }
    let output = cmd
        .arg(example)
        .current_dir(root)
        .output()
        .expect("failed to execute nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "nsl run (source_ad={source_ad}) failed (exit {:?}).\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    if source_ad {
        // The train-block must actually run source-AD (its forward is the guarded
        // path); otherwise a silent tape fallback would agree with the tape run
        // and mask a regression of the exact bug this test covers.
        assert!(
            stderr.contains("Using source-to-source AD for backward pass"),
            "train-block source-AD marker missing; it fell back to tape.\nstderr:\n{stderr}"
        );
        assert!(
            !stderr.contains("falling back to tape-based AD"),
            "source AD bailed out to tape AD.\nstderr:\n{stderr}"
        );
    }
    stdout
        .lines()
        .rev()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no float in stdout for source_ad={source_ad}:\n{stdout}"))
}

/// The critical path: `silu(x @ W)` in a `train` block. The matmul temp is
/// refcount 1, so before the in-place-suppression guard source-AD trained on
/// `silu'(silu(h))`. Assert `--source-ad` now matches tape-AD (the correct
/// gradient) after one SGD step, without hard-coding the value.
#[test]
fn e2e_silu_train_source_ad_matches_tape() {
    let root = workspace_root();
    let example = root.join("examples/silu_train_source_ad.nsl");
    let tape = run_and_parse_sum(&root, &example, false);
    let source = run_and_parse_sum(&root, &example, true);
    assert!(
        (tape - source).abs() < 1e-3,
        "source-AD sum(m.w)={source} disagrees with tape-AD {tape} \
         (pre-fix source-AD trained on silu'(silu(h)) ≈ -4.59 vs correct ≈ -4.73)"
    );
}

