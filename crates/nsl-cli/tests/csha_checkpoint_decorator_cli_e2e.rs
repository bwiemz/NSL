//! Cycle 17 T4 — CLI-level regression gates for the `@csha(...)` and
//! `@checkpoint(policy="...")` decorator handoff that the phase2.6 ← main
//! merge silently dropped.
//!
//! Background. The phase2.6 ← main merge regression deleted
//! `ModuleData.csha_configs` / `ModuleData.checkpoint_policies` and the merge
//! subagent re-initialized `CompileOptions.{csha_configs, checkpoint_policies}`
//! to empty `HashMap`s with false "populated by loader" comments. Existing CI
//! was blind to this: snapshot tests passed identical bytes, and the
//! direct-builder tests bypassed the CLI handoff entirely. Only the
//! verification workflow's regression verifier caught it.
//!
//! `pipeline.rs` now carries two in-process unit tests that pin the
//! `analysis_to_*` helper return values; this file pins the same wiring at
//! the CLI level by driving the built `nsl` binary end-to-end.
//!
//! Live regression gate. Test A uses `@csha(disable=true)` as an observable
//! decorator effect. The codegen-side hook in `nsl-codegen/src/stmt.rs` reads
//! `compile_options.csha_configs[model_type_name]` and, when `disabled=true`,
//! skips the CSHA planner entirely. The CSHA planner emits a `[csha]` summary
//! to stderr on every run AND a full "CSHA Compilation Report" when
//! `--csha-report` is set. So:
//!
//!   * Decorator HONOURED ⇒ `disabled=true` ⇒ planner skipped
//!     ⇒ NO "CSHA Compilation Report" line in stderr.
//!   * Decorator DROPPED (loader.rs:433 regression) ⇒ `csha_configs` is empty
//!     ⇒ `disabled` defaults to `false` ⇒ planner runs ⇒ "CSHA Compilation
//!     Report" appears in stderr.
//!
//! Test A asserts the first case; reverting `loader.rs:433` flips it to the
//! second case and the assertion fails. This was verified locally before
//! committing.
//!
//! Test B is a defense-in-depth gate that exercises the `nsl check` parse +
//! semantic-analysis path for `@checkpoint(policy="full")`. `nsl check`
//! deliberately does NOT route through the multi-file loader (it uses
//! `pipeline::frontend_with_flags` directly), so this test cannot by itself
//! catch a `loader.rs:439` revert — that gate is held by the in-process
//! `checkpoint_full_policy_survives_pipeline_handoff` unit test in
//! `crates/nsl-cli/src/pipeline.rs`. What Test B DOES catch is parse-level
//! or semantic-level regressions that would mark `@checkpoint(policy="...")`
//! as an unknown decorator or reject `policy="full"` as an invalid value,
//! either of which would also defeat the multi-file loader handoff
//! downstream.

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

/// Minimal `@train`-bearing fixture with `@csha(disable=true)` on the model.
/// `train(...)` triggers the multi-file build path (`needs_multi_file` in
/// `commands/build.rs:20`), which is where `module_data_to_csha_configs` is
/// called against the loader-populated `ModuleData.csha_configs` field. The
/// `disable=true` decorator must reach codegen via that field for the
/// planner-skip branch in `stmt.rs:4351` to fire.
const SRC_CSHA_DISABLED: &str = r#"from nsl.nn.losses import mse_loss

@csha(disable=true)
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

/// Minimal fixture with `@checkpoint(policy="full")` on a forward fn. Pinned
/// against the same source the in-process `pipeline.rs` unit test uses, so
/// the two layers (helper-level + CLI-level) cover identical ground.
const SRC_CHECKPOINT_FULL: &str = r#"@checkpoint(policy="full")
fn forward_step(x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
    return x
"#;

/// Test A — `@csha(disable=true)` round-trips through `nsl build
/// --csha=auto --csha-report --source-ad`. The decorator MUST take effect
/// (skip the planner) — if it does not, stderr will contain the CSHA report
/// header. This is the live regression gate for `loader.rs:433` (the
/// `ModuleData.csha_configs = analysis.csha_configs` assignment).
#[test]
fn csha_disable_decorator_round_trips_through_build_csha_report() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("csha_disabled.nsl");
    let out_path = tmp.path().join("csha_disabled_out");
    fs::write(&src_path, SRC_CSHA_DISABLED).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("build")
        .arg(&src_path)
        // --source-ad: CSHA only fires on the source-AD lowering path.
        .arg("--source-ad")
        // --emit-obj: skip the link step so this test runs on CPU-only CI.
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        // --csha=auto: enables the CSHA planner globally.
        .arg("--csha")
        .arg("auto")
        // --csha-report: makes the planner's report stream to stderr if it
        // runs. The decorator's `disable=true` must suppress this entirely.
        .arg("--csha-report");

    // Build may succeed or fail (CSHA-disabled programs still compile
    // through codegen; --emit-obj does not require CUDA link). We only
    // care about the stderr content: the decorator must have suppressed
    // the report. Use `output()` so a non-zero exit doesn't mask the
    // stderr assertion below.
    let output = cmd.output().expect("spawn nsl build");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !stderr.contains("=== CSHA Compilation Report ==="),
        "regression: `@csha(disable=true)` did not suppress the CSHA planner. \
         The report header appeared in stderr, which means `csha_configs` was \
         either empty at the codegen lookup (loader.rs:433 regression) or the \
         decorator's `disabled` field did not flow through. Full stderr:\n{}",
        stderr,
    );
    assert!(
        !stderr.contains("[csha] csha["),
        "regression: `@csha(disable=true)` did not suppress the CSHA planner. \
         The `[csha] csha[...]` summary line appeared in stderr. Full stderr:\n{}",
        stderr,
    );
}

/// Test B — `@checkpoint(policy="full")` survives `nsl check`. Defense-in-
/// depth gate for the parse + semantic-analysis layer; pairs with the
/// in-process `pipeline.rs::checkpoint_full_policy_survives_pipeline_handoff`
/// unit test that gates `loader.rs:439`.
#[test]
fn checkpoint_full_decorator_accepted_by_nsl_check() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("checkpoint_full.nsl");
    fs::write(&src_path, SRC_CHECKPOINT_FULL).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path);
    cmd.assert()
        .success()
        // `nsl check` prints "OK: ... checked successfully" on stdout when
        // semantic analysis surfaces zero error-level diagnostics. Pin the
        // exact line so a future "soft accept that warns" regression also
        // fails this test.
        .stdout(predicate::str::contains("checked successfully"))
        // The decorator must NOT be flagged as unknown. Pre-2026-04 the
        // parser accepted `@checkpoint` but the semantic validator emitted
        // `unknown decorator` if `policy=...` parsing broke; assert the
        // negative explicitly so the test stays informative on regression.
        .stderr(predicate::str::contains("unknown decorator").not())
        .stderr(predicate::str::contains("error: ").not());
}
