//! End-to-end gates for the pass manager (item 2 step 6): the per-compile
//! ordering authority, its enforcement path, and the WGGO phase-declaration
//! fix its fixture witnessed.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn loop_fixture() -> PathBuf {
    workspace_root()
        .join("crates/nsl-codegen/tests/fixtures/cpdt_precision_fp16_loop.nsl")
}

fn plain_fixture() -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures/cpdt_precision_fp16.nsl")
}

/// A train block whose model variable is loop-bound gets no WGGO pre-plan
/// (the prepass cannot type it), so WGGO's FIRST record lands at the
/// in-place replan site, under TrainBlock. Before the registry declared
/// that phase this compile printed `PHASE MISMATCH: WGGO ran in TrainBlock
/// but the registry declares [KernelPrepass]` — a REAL compile proving the
/// declaration false. The declaration now names both phases; this arm is
/// the regression pin, asserting both the attribution and the absence of
/// the mismatch line.
#[test]
fn a_loop_bound_wggo_replan_is_a_declared_phase_not_a_mismatch() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.env("NSL_PASS_TRACE", "1");
    cmd.arg("run")
        .arg(loop_fixture())
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("WGGO(OnWengert)@TrainBlock"))
        .stderr(predicate::str::contains("PHASE MISMATCH: WGGO").not());
}

/// The enforcement path, driven by the same knob convention as
/// NSL_WGGO_FORCE_STALE_TABLE: a dependency-order violation refuses the
/// compile — always-on enforcement, NOT gated on NSL_PASS_TRACE, which is
/// half the point (an advisory nobody prints is not enforcement). The
/// message names the knob so a forced refusal cannot be mistaken for a
/// real scheduling defect.
#[test]
fn a_dependency_order_violation_refuses_the_compile() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.env("NSL_FORCE_DEPENDENCY_ORDER_VIOLATION", "1");
    cmd.arg("run").arg(plain_fixture()).arg("--source-ad");
    cmd.assert().failure().stderr(predicate::str::contains(
        "dependency-order violation forced by NSL_FORCE_DEPENDENCY_ORDER_VIOLATION",
    ));
}

/// The enforcement consult sits on every successful train-block compile, so
/// it must be inert on a healthy one: same compile, no knob, green — and
/// under the trace, no DEPENDENCY ORDER advisory either.
#[test]
fn a_healthy_compile_is_untouched_by_enforcement() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.env("NSL_PASS_TRACE", "1");
    cmd.arg("run")
        .arg(plain_fixture())
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("DEPENDENCY ORDER").not());
}

#[test]
fn the_fixtures_these_gates_use_exist() {
    assert!(loop_fixture().exists(), "missing {:?}", loop_fixture());
    assert!(plain_fixture().exists(), "missing {:?}", plain_fixture());
}
