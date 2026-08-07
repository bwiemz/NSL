//! Unit tests for `nsl_codegen::pass_manager` — the per-compile ordering
//! authority — in their OWN test binary for the same reason
//! `pass_trace_unit.rs` is: the recorder's storage is process-global and the
//! lib test binary's ~200 other tests drive instrumented pass entry points
//! concurrently.
//!
//! The load-bearing test here is the cross-compile split: the SAME event
//! sequence that the process-scoped advisory checker must flag (it cannot
//! know better) is provably clean per compile — which is the entire argument
//! for why enforcement lives on the manager and nowhere else.

use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

use nsl_codegen::pass_manager::PassManager;
use nsl_codegen::pass_trace::{observed, record, reset};

fn fresh() -> std::sync::MutexGuard<'static, ()> {
    let g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    g
}

#[test]
fn a_managers_view_is_its_own_compiles_only() {
    let _g = fresh();
    {
        let _m1 = PassManager::begin();
        record("CSHA");
    }
    let m2 = PassManager::begin();
    record("WGGO");
    assert_eq!(
        m2.per_compile_trace()
            .iter()
            .map(|(p, _)| *p)
            .collect::<Vec<_>>(),
        vec!["WGGO"],
        "an earlier compile's invocation leaked into this compile's view"
    );
}

/// THE test this module exists for. One process, two compiles: compile 1
/// runs only CSHA (a legitimate lone consumer — its producer was off),
/// compile 2 runs only WGGO. The process-global first-invocation order is
/// [CSHA, WGGO] — an inversion of the declared WGGO->CSHA edge, and the
/// process-scoped checker MUST report it (it has no compile boundaries to
/// know better; this is exactly why it stayed advisory). The per-compile
/// authority sees two individually well-ordered compiles and stays silent —
/// the difference that makes refusal sound.
#[test]
fn an_inversion_split_across_compiles_is_not_a_per_compile_violation() {
    let _g = fresh();
    {
        let m1 = PassManager::begin();
        record("CSHA");
        assert!(m1.dependency_order_violations().is_empty());
    }
    let m2 = PassManager::begin();
    record("WGGO");
    assert!(
        m2.dependency_order_violations().is_empty(),
        "a cross-compile interleaving was judged as if it were one compile"
    );
    assert!(
        m2.enforce_dependency_order().is_ok(),
        "enforcement refused a clean compile on another compile's evidence"
    );
    // The process-scoped advisory DOES see an inversion in the merged view —
    // deliberately pinned, because if this ever stops holding, the advisory
    // and the enforced check have diverged in edge semantics.
    assert_eq!(observed(), vec!["CSHA", "WGGO"]);
    assert_eq!(
        nsl_codegen::pass_bus::dependency_order_violations().len(),
        1,
        "the process-scoped view no longer cries wolf on the split — did the \
         shared core change?"
    );
}

#[test]
fn a_real_inversion_within_one_compile_is_caught_and_refused() {
    let _g = fresh();
    let m = PassManager::begin();
    record("CSHA");
    record("WGGO");
    let v = m.dependency_order_violations();
    assert_eq!(v.len(), 1, "expected exactly the WGGO->CSHA edge: {v:?}");
    assert_eq!((v[0].0, v[0].1), ("WGGO", "CSHA"));
    let err = m
        .enforce_dependency_order()
        .expect_err("a same-compile inversion must refuse");
    assert!(
        err.contains("CSHA was invoked before WGGO") && err.contains("wggo_overrides"),
        "the refusal must name the consumer, the producer, and the channel: {err}"
    );
}

#[test]
fn the_correct_order_and_value_ordered_edges_pass() {
    let _g = fresh();
    let m = PassManager::begin();
    // FASE before WGGO is a real compile path (the in-place replan) and its
    // edge claims ValueOrderedOnly — never an invocation violation.
    record("FASE");
    record("WGGO");
    record("CSHA");
    record("WRGA");
    assert!(m.dependency_order_violations().is_empty());
    assert!(m.enforce_dependency_order().is_ok());
}

#[test]
fn drop_restores_the_previous_epoch_for_nested_compiles() {
    let _g = fresh();
    let outer = PassManager::begin();
    record("WGGO");
    {
        let inner = PassManager::begin();
        record("CSHA");
        assert_eq!(
            inner
                .per_compile_trace()
                .iter()
                .map(|(p, _)| *p)
                .collect::<Vec<_>>(),
            vec!["CSHA"]
        );
    }
    // The inner manager dropped; recording resumes under the OUTER epoch.
    record("CSHA");
    assert_eq!(
        outer
            .per_compile_trace()
            .iter()
            .map(|(p, _)| *p)
            .collect::<Vec<_>>(),
        vec!["WGGO", "CSHA"],
        "the inner compile's drop did not restore the outer epoch"
    );
    // And the outer order is correct, so no violation.
    assert!(outer.enforce_dependency_order().is_ok());
}

/// The report surface must not change shape under epochs: a pass recorded in
/// several compiles renders once, at first occurrence — exactly what the
/// pre-epoch process-wide idempotence produced. The gate parsers in
/// pass_trace_gate.rs split those lines positionally.
#[test]
fn process_facing_views_dedupe_across_epochs() {
    let _g = fresh();
    {
        let _m1 = PassManager::begin();
        record("WGGO");
    }
    let _m2 = PassManager::begin();
    record("WGGO");
    record("CCR");
    assert_eq!(observed(), vec!["WGGO", "CCR"]);
}
