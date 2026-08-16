//! Milestone C behavioural gates: the scheduler's trace on REAL compiles.
//!
//! The static half (`pass_scheduler_coverage.rs`) pins that the schedule
//! sites exist; these pin that they FIRE — the `[pass-manager]` lines were
//! previously asserted by no test at all, so a scheduler that stopped being
//! reached would have kept every gate green.

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

fn fixture(name: &str) -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures").join(name)
}

fn nsl(fixture_name: &str) -> Command {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.env("NSL_PASS_TRACE", "1");
    cmd.arg("run").arg(fixture(fixture_name)).arg("--source-ad");
    cmd
}

/// THE Milestone C criterion, witnessed end to end: a train block nested
/// inside `fn main():` reaches the passes through compile_user_functions,
/// which installs no phase — the scheduler used to print
/// `phase=unscoped(phase check skipped)` on this exact shape and skip the
/// check. compile_train_block now scopes itself, so no production
/// scheduled pass sees phase=None: the unscoped token must be gone and
/// FASE (scheduled unconditionally on every train compile) must attribute
/// to TrainBlock, with the phase check LIVE.
#[test]
fn a_nested_train_block_schedules_with_a_phase_not_unscoped() {
    nsl("nested_train_in_fn.nsl")
        .assert()
        .success()
        .stderr(predicate::str::contains("unscoped(phase check skipped)").not())
        .stderr(predicate::str::contains("-> FASE phase=TrainBlock"))
        .stderr(predicate::str::contains("PHASE MISMATCH").not());
}

/// Both halves for a flag-gated scheduled pass: WGGO's schedule site sits
/// inside the `--wggo` mode guard, so the trace line must appear exactly
/// when the flag does. (CSHA/CPDT/FASE schedule unconditionally and decline
/// inside the body, so absence is not assertable for them — WGGO is the
/// clean witness.)
#[test]
fn a_flag_gated_pass_is_scheduled_exactly_when_its_flag_is_on() {
    nsl("cpdt_precision_fp16.nsl")
        .arg("--wggo")
        .arg("full")
        .assert()
        .success()
        .stderr(predicate::str::contains("-> WGGO"));
    nsl("cpdt_precision_fp16.nsl")
        .assert()
        .success()
        .stderr(predicate::str::contains("-> WGGO").not());
}

/// The scheduler's trace is opt-in: without NSL_PASS_TRACE=1 a compile must
/// emit no `[pass-manager]` lines at all — the enforcement runs either way,
/// but the observability contract is that stderr stays clean.
#[test]
fn the_scheduler_trace_is_opt_in() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.env_remove("NSL_PASS_TRACE");
    cmd.arg("run")
        .arg(fixture("cpdt_precision_fp16.nsl"))
        .arg("--source-ad");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("[pass-manager]").not());
}

/// Fixture-existence guard, the sibling gates' convention: a renamed
/// fixture fails once, clearly, instead of as three confusing test errors.
#[test]
fn the_fixtures_exist() {
    for f in ["nested_train_in_fn.nsl", "cpdt_precision_fp16.nsl"] {
        assert!(
            fixture(f).exists(),
            "missing fixture {} — renamed?",
            fixture(f).display()
        );
    }
}
