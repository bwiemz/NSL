//! Unit tests for `nsl_codegen::pass_trace`, deliberately in their OWN test
//! binary.
//!
//! The recorder's storage is process-global, and ~200 other nsl-codegen unit
//! tests call instrumented pass entry points directly (`fase::plan` x26,
//! `wrga::run` x35, ...). Inside the lib test binary those run concurrently
//! with these and push into the same vector, so assertions like "exactly one
//! pass ran" fail nondeterministically — reproduced at `--test-threads=64`,
//! clean at the default, i.e. it would fire on a loaded CI runner and vanish
//! on the rerun. A separate process removes the polluters entirely rather
//! than papering over them with a lock that cannot reach other tests.

use std::sync::Mutex;

/// Serializes these tests against EACH OTHER; the cross-binary problem is
/// solved by being a separate process.
static TEST_LOCK: Mutex<()> = Mutex::new(());

use nsl_codegen::pass_registry::CompilePhase;
use nsl_codegen::pass_trace::*;

fn fresh() -> std::sync::MutexGuard<'static, ()> {
    let g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    g
}

#[test]
fn record_is_idempotent_and_preserves_first_invocation_order() {
    let _g = fresh();
    record("WGGO");
    record("CCR");
    record("WGGO");
    assert_eq!(observed(), vec!["WGGO", "CCR"]);
    reset();
    assert!(observed().is_empty());
}

/// The registry order is PreExtraction < ModuleScan < OnWengert <
/// OnAdjoint < Lowering. CCR is OnAdjoint and FASE is PreExtraction, so
/// FASE-then-CCR is legal and CCR-then-FASE is not.
#[test]
fn stage_order_is_checked_in_the_direction_that_can_fail() {
    let _g = fresh();
    record("FASE");
    record("CCR");
    assert_eq!(stage_order_violation(), None);
    reset();
    record("CCR");
    record("FASE");
    assert_eq!(stage_order_violation(), Some(("CCR", "FASE")));
}

/// Five passes share `OnWengert`, so their relative order is genuinely
/// unconstrained. Pinned so nobody later reads a clean order check as
/// proof that the intra-stage sequence was verified.
#[test]
fn same_stage_passes_are_unordered_by_design() {
    let _g = fresh();
    for p in ["CPKD", "WRGA", "CSHA", "WGGO"] {
        record(p);
    }
    assert_eq!(
        stage_order_violation(),
        None,
        "OnWengert passes must not be ordered against each other — the \
         registry declares no order for them"
    );
}

/// CEP and CFIE are OutOfBand and must never produce a violation in
/// either direction; otherwise enabling inference-side work would fail a
/// training-pipeline order check for no reason.
#[test]
fn out_of_band_passes_never_violate_the_order() {
    let _g = fresh();
    record("MemoryPlanner"); // Lowering, the highest ranked stage
    record("CFIE");
    record("CEP");
    record("FASE"); // PreExtraction, the lowest — but after OutOfBand
    assert_eq!(
        stage_order_violation(),
        Some(("MemoryPlanner", "FASE")),
        "the Lowering->PreExtraction inversion should be reported, and the \
         OutOfBand passes between them should be transparent"
    );
}

#[test]
fn the_report_names_both_what_ran_and_what_did_not() {
    let _g = fresh();
    record("WGGO");
    let r = report();
    assert!(r.contains("1 pass(es) ran: WGGO(OnWengert)"), "{r}");
    assert!(r.contains("did not run:"), "{r}");
    assert!(r.contains("CCR"), "idle passes must be enumerated: {r}");
    reset();
    assert!(report().contains("NO passes ran"));
}

/// Every name this module's own tests use must be a real registered pass,
/// so the tests cannot drift into asserting on passes that no longer
/// exist.
#[test]
fn names_used_by_these_tests_are_registered() {
    for p in [
        "WGGO",
        "CCR",
        "FASE",
        "CPKD",
        "WRGA",
        "CSHA",
        "CEP",
        "CFIE",
        "MemoryPlanner",
    ] {
        assert!(
            nsl_codegen::pass_registry::pass(p).is_some(),
            "{p} is not in the pass registry"
        );
    }
}

/// The phase scope is RAII and must NEST correctly: an inner phase restores
/// the outer one on drop, not `None`.
///
/// No path in the tree nests two phases TODAY — `compile_flash_attention_
/// kernels` closes before `compile_main` opens, and neither calls the other —
/// so save/restore is currently indistinguishable from set/clear. It is
/// pinned anyway because the scopes now live at the CALLEE, where a future
/// caller could nest them, and the failure would be silent: every pass after
/// the inner scope closed would be attributed `None`.
#[test]
fn phase_scopes_nest_and_restore_the_previous_phase() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    let outer = enter_phase(CompilePhase::KernelPrepass);
    record("WGGO");
    {
        let _inner = enter_phase(CompilePhase::TrainBlock);
        record("FASE");
    }
    // Back in the outer phase.
    record("CSHA");
    drop(outer);
    record("CEP");

    let ph: Vec<_> = observed_phases();
    assert_eq!(ph[0], ("WGGO", Some(CompilePhase::KernelPrepass)));
    assert_eq!(ph[1], ("FASE", Some(CompilePhase::TrainBlock)));
    assert_eq!(
        ph[2],
        ("CSHA", Some(CompilePhase::KernelPrepass)),
        "the inner guard did not restore the outer phase"
    );
    assert_eq!(
        ph[3],
        ("CEP", None),
        "a pass recorded after every guard dropped must be unattributed"
    );
    reset();
}

/// `phase_mismatches` must actually fire. WGGO is declared KernelPrepass, so
/// recording it under TrainBlock is a mismatch; recording it under
/// KernelPrepass is not. Without both directions this could be a constant.
#[test]
fn a_pass_running_in_the_wrong_phase_is_reported() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    {
        let _p = enter_phase(CompilePhase::KernelPrepass);
        record("WGGO");
    }
    assert!(phase_mismatches().is_empty(), "correct phase flagged as a mismatch");

    reset();
    {
        let _p = enter_phase(CompilePhase::TrainBlock);
        record("WGGO");
    }
    let m = phase_mismatches();
    assert_eq!(m.len(), 1, "expected exactly one mismatch, got {m:?}");
    assert_eq!(m[0].0, "WGGO");
    assert_eq!(m[0].1, &[CompilePhase::KernelPrepass][..], "declared");
    assert_eq!(m[0].2, Some(CompilePhase::TrainBlock), "observed");
    reset();
}

/// An OutOfBand pass is DRIVEN BY A CLI SUBCOMMAND, outside the pipeline, so
/// being unattributed is correct for it — and being attributed is the
/// surprise. Pinned in both directions so the OutOfBand arm cannot silently
/// become "anything goes".
#[test]
fn out_of_band_passes_are_expected_to_be_unattributed() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    record("CEP"); // no phase scope: correct for OutOfBand
    assert!(phase_mismatches().is_empty(), "unattributed OutOfBand flagged");

    reset();
    {
        let _p = enter_phase(CompilePhase::TrainBlock);
        record("CEP");
    }
    let m = phase_mismatches();
    assert_eq!(m.len(), 1, "an OutOfBand pass inside a phase must be flagged");
    assert_eq!(m[0].0, "CEP");
    reset();
}

/// Every registered pass's declared phase must be one the driver actually
/// establishes a scope for — otherwise the declaration can never be verified
/// and is decorative, which is exactly what `pass_registry`'s header forbids.
#[test]
fn every_declared_in_pipeline_phase_is_one_the_driver_scopes() {
    use nsl_codegen::pass_registry::{CompilePhase as P, PASSES};
    let scoped = [P::KernelPrepass, P::TrainBlock, P::Analysis, P::Lowering];
    for d in PASSES {
        for &ph in d.phases {
            match ph {
                P::OutOfBand => unreachable!("OutOfBand is an empty set"),
                p => assert!(
                    scoped.contains(&p),
                    "{} declares {p:?}, which no enter_phase scope establishes",
                    d.name
                ),
            }
        }
    }
}
