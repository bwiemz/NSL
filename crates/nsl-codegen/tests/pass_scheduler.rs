//! The `PassManager` as SCHEDULER (item 2 step 7): the refusals it adds, and
//! the declarations each one reads.
//!
//! `pass_manager.rs`'s header used to say "Not a scheduler … inverting control
//! is the step after this one". This is that step, for one pass. These tests
//! exist because an enforcement layer whose refusals are never observed firing
//! is indistinguishable from decoration — the failure mode `pass_bus.rs`'s
//! `Invariant` doc and `pass_trace`'s "a detector nobody calls" note both name.
//!
//! Each test drives one declared check to its refusal, and the positive
//! controls pin that the same check ADMITS the legitimate case.

use nsl_codegen::pass_bus::PassBus;
use nsl_codegen::pass_manager::PassManager;
use nsl_codegen::pass_registry::CompilePhase;
use nsl_codegen::pass_trace::{enter_phase, record, record_disposition, PassDisposition};
use nsl_codegen::wengert::{PrimalOp, VarId, WengertList, WengertOp};
use std::collections::HashMap;

fn tiny_tape(n: u32) -> WengertList {
    let ops: Vec<WengertOp> = (0..n)
        .map(|i| WengertOp {
            id: i,
            result: i,
            op: if i == 0 {
                PrimalOp::Input("x".into())
            } else {
                PrimalOp::Relu
            },
            inputs: if i == 0 { vec![] } else { vec![i - 1] },
            saved_for_backward: false,
            checkpointed: false,
        })
        .collect();
    WengertList {
        ops,
        output: n.saturating_sub(1),
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    }
}


/// A real `WrgaPlan`, produced by the pass itself rather than fabricated —
/// the channel's declared payload is `crate::wrga::WrgaPlan`, and publishing
/// a hand-built stand-in would test the scheduler against a value the pass
/// cannot actually produce.
fn minimal_wrga_plan(tape: &WengertList) -> nsl_codegen::wrga::WrgaPlan {
    nsl_codegen::wrga::run(nsl_codegen::wrga::WrgaInput {
        mode: nsl_ast::block::WrgaMode::Auto,
        trainable_patterns: Vec::new(),
        manual_adapter_targets: Vec::new(),
        hybrid_layers: Vec::new(),
        wengert: tape,
        loss_output: tape.output,
        weights: None,
        target: "",
        budget_params: 0,
        r_min: 1,
        r_max: 8,
        seed: 0,
        inspect_pinned_vars: Default::default(),
        wggo_overrides: None,
        ablation: Default::default(),
        custom_adapter: None,
    })
}

/// A pass name the registry does not carry cannot be checked at all, so
/// scheduling it is refused rather than silently unchecked. Without this, a
/// typo in a call site would produce an invocation with NO preconditions
/// enforced while looking exactly like a scheduled one.
#[test]
fn scheduling_an_unregistered_pass_is_refused() {
    let mgr = PassManager::begin();
    let mut ran = false;
    let err = match mgr.scheduler().schedule("NotARealPass", None, || ran = true) {
        Err(e) => e,
        Ok(_) => panic!("an unregistered pass must be refused"),
    };
    assert!(
        err.contains("not in pass_registry::PASSES"),
        "message should name the registry, got: {err}"
    );
    assert!(!ran, "the body must not run when preconditions fail");
}

/// `PassDescriptor::phases` records which driver phases invoke a pass. WRGA
/// declares `TrainBlock` only; reaching it from `KernelPrepass` means the
/// driver grew an invocation site the registry does not know about.
#[test]
fn scheduling_from_an_undeclared_phase_is_refused() {
    let mgr = PassManager::begin();
    let _phase = enter_phase(CompilePhase::KernelPrepass);
    let mut ran = false;
    let err = match mgr.scheduler().schedule("WRGA", None, || ran = true) {
        Err(e) => e,
        Ok(_) => panic!("WRGA does not declare KernelPrepass"),
    };
    assert!(
        err.contains("KernelPrepass") && err.contains("declared phases"),
        "message should name both the actual and declared phases, got: {err}"
    );
    assert!(!ran, "the body must not run when preconditions fail");
}

/// Positive control for the check above — the SAME call in the phase WRGA
/// declares must be admitted. Without this, the refusal test would pass just
/// as well against a scheduler that refused everything.
#[test]
fn scheduling_from_the_declared_phase_is_admitted() {
    let mgr = PassManager::begin();
    let _phase = enter_phase(CompilePhase::TrainBlock);
    let mut ran = false;
    let scheduled = mgr
        .scheduler()
        .schedule("WRGA", None, || {
            ran = true;
            7u32
        })
        .unwrap_or_else(|e| panic!("TrainBlock is WRGA's declared phase: {e}"));
    assert!(ran, "the body must run when preconditions hold");
    assert_eq!(
        scheduled.finish(&PassBus::default()).expect("no disposition recorded"),
        7
    );
}

/// Most compiles reach a pass with no `enter_phase` scope installed at all —
/// every unit test that drives the compiler directly, and any entry point not
/// yet wrapped. Refusing those would fire on correct behaviour, so an
/// unscoped invocation is admitted (and traced as unscoped).
#[test]
fn an_unscoped_invocation_is_admitted_rather_than_refused() {
    let mgr = PassManager::begin();
    // Deliberately NO enter_phase.
    let mut ran = false;
    let _scheduled = mgr
        .scheduler()
        .schedule("WRGA", None, || ran = true)
        .unwrap_or_else(|e| panic!("an unscoped invocation must not be refused: {e}"));
    assert!(ran);
}

/// `wrga_plan` declares `applied_implies_published: Enforced`. The bus REPORT
/// finds the violation at end of compile, where it is a line nobody blocks on;
/// the scheduler finds it at the pass boundary, where it is a refusal that
/// still names the pass.
#[test]
fn applied_with_an_empty_enforced_channel_is_refused() {
    let mgr = PassManager::begin();
    let _phase = enter_phase(CompilePhase::TrainBlock);
    let scheduled = mgr
        .scheduler()
        .schedule("WRGA", None, || {
            // Stand in for the pass body reporting that it rewrote something
            // while never publishing its plan.
            record("WRGA");
            record_disposition("WRGA", PassDisposition::Applied { rewrites: 1 });
        })
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));

    let empty = PassBus::default();
    let err = scheduled
        .finish(&empty)
        .expect_err("Applied with an empty enforced channel must be refused");
    assert!(
        err.contains("wrga_plan") && err.contains("Applied"),
        "message should name the channel and the disposition, got: {err}"
    );
}

/// Positive control: the same disposition with the channel POPULATED must be
/// admitted, or the check above would be satisfied by a scheduler that always
/// refused an Applied pass.
#[test]
fn applied_with_a_published_channel_is_admitted() {
    // Build the plan BEFORE the epoch opens. `wrga::run` records its own
    // disposition (`Declined { NoCandidates }` on a toy tape), and inside the
    // live epoch that OVERWRITES the `Applied` staged below — leaving
    // `finish` with `applied == false`, so it would skip the
    // `enforced_publish_channels` loop entirely and this "positive control"
    // would control nothing. Review caught exactly that: with the plan built
    // inside the epoch, `is_published` had no test that it ever returns true.
    let plan = minimal_wrga_plan(&tiny_tape(3));

    let mgr = PassManager::begin();
    let _phase = enter_phase(CompilePhase::TrainBlock);
    let scheduled = mgr
        .scheduler()
        .schedule("WRGA", None, || {
            record("WRGA");
            record_disposition("WRGA", PassDisposition::Applied { rewrites: 1 });
        })
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));

    // The antecedent must actually hold, or the check below is skipped
    // silently rather than satisfied.
    assert!(
        matches!(
            nsl_codegen::pass_trace::dispositions()
                .into_iter()
                .find(|(p, _)| *p == "WRGA")
                .map(|(_, d)| d),
            Some(PassDisposition::Applied { .. })
        ),
        "the staged Applied disposition must survive into finish(), or this \
         test passes without ever entering the published-channel check"
    );

    let mut bus = PassBus::default();
    assert!(
        !bus.is_published(nsl_codegen::pass_bus::Channel::WrgaPlan),
        "the channel must start empty so publishing is what changes the outcome"
    );
    bus.publish_wrga_plan(plan);
    assert!(bus.is_published(nsl_codegen::pass_bus::Channel::WrgaPlan));
    scheduled
        .finish(&bus)
        .expect("a published channel satisfies applied_implies_published");
}

/// The tape half. WRGA's plan holds `TapeRef::PositionalIndex` and is consumed
/// several hundred lines after the scan, at the `effective_primal` fork.
/// Positional references are valid only against the list state that produced
/// them, so a tape that moved in between invalidates the plan — and nothing
/// checked that before the pass was scheduled.
#[test]
fn a_tape_that_moved_between_scan_and_consumption_is_refused() {
    let mgr = PassManager::begin();
    let _phase = enter_phase(CompilePhase::TrainBlock);
    let sched = mgr.scheduler();
    let tape = tiny_tape(6);
    let _scheduled = sched
        .schedule("WRGA", Some(&tape), || ())
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));

    // Same list state: admitted.
    sched
        .assert_tape_unchanged_since("WRGA", &tape)
        .expect("an unmoved tape must be admitted");

    // A deletion that does NOT renumber — exactly what the fused-LCE prune and
    // WGGO's sub-block prune do — shifts every later position while leaving
    // ids alone. This is the mutation the positional-reference rule exists for.
    let mut moved = tape.clone();
    moved.ops.remove(2);
    let err = sched
        .assert_tape_unchanged_since("WRGA", &moved)
        .expect_err("a moved tape must be refused");
    assert!(
        err.contains("POSITIONAL") && err.contains("6 ops then, 5 now"),
        "message should explain the staleness and quantify it, got: {err}"
    );
}

/// A mutation that changes an op's IDENTITY without changing the op count must
/// also be caught: `fuse_swiglu_gate_backward` rewrites a `Passthrough`'s name
/// in place. A digest over lengths and ids alone would miss it.
#[test]
fn an_in_place_op_rewrite_is_caught_even_at_the_same_length() {
    let mgr = PassManager::begin();
    let sched = mgr.scheduler();
    let tape = tiny_tape(4);
    let _scheduled = sched
        .schedule("WRGA", Some(&tape), || ())
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));

    let mut rewritten = tape.clone();
    rewritten.ops[2].op = PrimalOp::Passthrough("silu_backward".into());
    assert_eq!(rewritten.ops.len(), tape.ops.len());
    sched
        .assert_tape_unchanged_since("WRGA", &rewritten)
        .expect_err("an in-place op rewrite must be refused");
}

/// A pass that never ran recorded no scan, so there is no captured state to
/// invalidate and the check must not manufacture a refusal.
#[test]
fn a_pass_that_never_scanned_a_tape_passes_vacuously() {
    let mgr = PassManager::begin();
    mgr.scheduler()
        .assert_tape_unchanged_since("WRGA", &tiny_tape(3))
        .expect("no recorded scan means nothing to invalidate");
}

/// Digests are retired with their compile. Without this the map grows one
/// entry per (scheduled pass x compile) for the process lifetime, and — worse
/// — a later compile could answer an earlier compile's staleness question.
#[test]
fn a_compiles_tape_digest_does_not_outlive_it() {
    let tape = tiny_tape(5);
    {
        let mgr = PassManager::begin();
        let _scheduled = mgr
            .scheduler()
            .schedule("WRGA", Some(&tape), || ())
            .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    } // manager dropped: epoch retired

    // A fresh compile must not see the previous one's scan. If it did, this
    // moved tape would be refused against a digest that belongs to a compile
    // that has ended.
    let mgr2 = PassManager::begin();
    let mut moved = tape.clone();
    moved.ops.remove(1);
    mgr2.scheduler()
        .assert_tape_unchanged_since("WRGA", &moved)
        .expect("a new compile starts with no recorded scans");
}

/// `VarId` is unused in the fixture beyond typing; keep the import honest.
const _: Option<VarId> = None;

/// A pass re-invoked in one compile WITHOUT a tape must clear its recorded
/// scan, not inherit the previous one.
///
/// Not reachable for WRGA (always scheduled with a tape), but the re-schedule
/// path exists precisely to support WGGO's in-place replan, which is a
/// re-invocation that may carry no tape. Leaving the stale digest live would
/// answer a staleness question about a scan this invocation superseded — and
/// refuse a correct build.
#[test]
fn a_reschedule_without_a_tape_clears_the_recorded_scan() {
    let mgr = PassManager::begin();
    let sched = mgr.scheduler();
    let tape = tiny_tape(6);

    let _first = sched
        .schedule("WRGA", Some(&tape), || ())
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    // Sanity: the digest IS live at this point.
    let mut moved = tape.clone();
    moved.ops.remove(2);
    sched
        .assert_tape_unchanged_since("WRGA", &moved)
        .expect_err("the first scan must still be enforced");

    // Re-invoke with no tape: the pass no longer holds positional refs.
    let _second = sched
        .schedule("WRGA", None, || ())
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    sched
        .assert_tape_unchanged_since("WRGA", &moved)
        .expect("a tape-less re-invocation supersedes the earlier scan");
}

/// A re-invocation WITH a tape rebinds to the latest scan.
#[test]
fn a_reschedule_with_a_tape_rebinds_to_the_latest_scan() {
    let mgr = PassManager::begin();
    let sched = mgr.scheduler();
    let first = tiny_tape(6);
    let second = tiny_tape(4);

    let _a = sched
        .schedule("WRGA", Some(&first), || ())
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    let _b = sched
        .schedule("WRGA", Some(&second), || ())
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));

    sched
        .assert_tape_unchanged_since("WRGA", &second)
        .expect("the latest scan is the live one");
    sched
        .assert_tape_unchanged_since("WRGA", &first)
        .expect_err("the superseded scan must no longer be enforced");
}

/// The declaration the predecessor OBSERVATION reads. Pinned because the
/// observation itself is only rendered into a trace line, so nothing else in
/// the tree would notice this edge disappearing from the bus.
#[test]
fn wrga_declares_wggo_as_an_invocation_ordered_predecessor() {
    let preds = nsl_codegen::pass_bus::required_predecessors("WRGA");
    assert!(
        preds.iter().any(|(p, c)| *p == "WGGO"
            && c.descriptor().name == "wggo_overrides"),
        "WRGA should declare WGGO on wggo_overrides, got {preds:?}"
    );
    // And the query is edge-directed, not a producer listing: WGGO does not
    // consume anything WRGA produces.
    assert!(
        !nsl_codegen::pass_bus::required_predecessors("WGGO")
            .iter()
            .any(|(p, _)| *p == "WRGA"),
        "the predecessor relation must not be symmetric"
    );
}

/// Milestone C: a MutatesInPlace pass whose body mutates the very list it
/// scanned (CCR's compressed-save append) re-digests via `rescan_tape` — the
/// consumption fork must then see the state the pass LEFT, and the state it
/// ENTERED with must no longer be the enforced one.
#[test]
fn rescan_tape_rebinds_the_digest_to_the_post_mutation_state() {
    let mgr = PassManager::begin();
    let _p = enter_phase(CompilePhase::TrainBlock);
    let sched = mgr.scheduler();
    let mut tape = tiny_tape(5);

    let scheduled = sched
        .schedule("CCR", None, || {
            // The pass's own declared mutation: append to the scanned list
            // (positions of existing ops stay valid — the CCR shape).
            tape.ops.push(WengertOp {
                id: 5,
                result: 5,
                op: PrimalOp::Relu,
                inputs: vec![4],
                saved_for_backward: false,
                checkpointed: false,
            });
        })
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    scheduled.rescan_tape(&tape);

    sched
        .assert_tape_unchanged_since("CCR", &tape)
        .expect("the post-mutation state is the enforced one after rescan");
    let entry_state = tiny_tape(5);
    sched
        .assert_tape_unchanged_since("CCR", &entry_state)
        .expect_err("the pre-mutation state must no longer satisfy the digest");

    // And a LATER mutation — someone else moving the list after the pass
    // finished — is still refused, which is the point of the whole check.
    tape.ops.remove(2);
    sched
        .assert_tape_unchanged_since("CCR", &tape)
        .expect_err("a post-rescan mutation must refuse");
    // `scheduled` drops without finish() deliberately: this test is about
    // the digest, and CCR has no Enforced channel for finish to judge.
}

/// `rescan_tape` on a pass scheduled WITHOUT a tape installs a digest that
/// was previously absent — the CCR schedule shape (tape=None because the
/// closure needs the list mutably; digest installed after the body).
#[test]
fn rescan_tape_installs_a_digest_where_none_was_captured() {
    let mgr = PassManager::begin();
    let _p = enter_phase(CompilePhase::TrainBlock);
    let sched = mgr.scheduler();
    let tape = tiny_tape(7);

    let scheduled = sched
        .schedule("CCR", None, || ())
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    // Before the rescan there is no captured state: vacuous pass.
    sched
        .assert_tape_unchanged_since("CCR", &tiny_tape(2))
        .expect("no recorded scan means nothing to invalidate");
    scheduled.rescan_tape(&tape);
    sched
        .assert_tape_unchanged_since("CCR", &tape)
        .expect("the rescan is now the enforced state");
    sched
        .assert_tape_unchanged_since("CCR", &tiny_tape(2))
        .expect_err("a different list must now refuse");
    // `scheduled` drops without finish() deliberately, as above.
}

/// `defer_postconditions` skips a finish() that WOULD have refused — proven
/// by staging exactly the state the applied⇒published check fires on
/// (Applied recorded, Enforced channel empty) and observing the deferral
/// admit it while finish on the same state refuses. This is the WGGO-prepass
/// shape: the channel is structurally not yet publishable in that phase.
#[test]
fn defer_postconditions_skips_a_finish_that_would_refuse() {
    let bus = PassBus::default();
    let mgr = PassManager::begin();
    let _p = enter_phase(CompilePhase::TrainBlock);
    let sched = mgr.scheduler();

    // Stage: WRGA records Applied, wrga_plan (Enforced) never published.
    let scheduled_a = sched
        .schedule("WRGA", None, || {
            record("WRGA");
            record_disposition("WRGA", PassDisposition::Applied { rewrites: 1 });
            41
        })
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    // Antecedent: finish on this exact state refuses.
    let err = match scheduled_a.finish(&bus) {
        Err(e) => e,
        Ok(_) => panic!("applied with an empty Enforced channel must refuse"),
    };
    assert!(err.contains("wrga_plan"), "refusal names the channel: {err}");

    // The deferral admits the same state — and returns the body's value.
    let scheduled_b = sched
        .schedule("WRGA", None, || 42)
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    let v = scheduled_b.defer_postconditions(
        "test: the channel is structurally not yet publishable here",
    );
    assert_eq!(v, 42, "the deferral unwraps the body's value");
}

/// An empty reason is the escape hatch a gate cannot distinguish from an
/// oversight — refused at the API boundary, not by convention.
#[test]
#[should_panic(expected = "non-empty reason")]
fn defer_postconditions_requires_a_reason() {
    let mgr = PassManager::begin();
    let _p = enter_phase(CompilePhase::TrainBlock);
    let scheduled = mgr
        .scheduler()
        .schedule("WRGA", None, || ())
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    scheduled.defer_postconditions("   ");
}

/// The rescan obligation is the SCHEDULER's, not the caller's memory: a pass
/// declaring MutatesInPlace + PositionalIndex (CCR) that records Applied
/// with no live digest gets refused at finish() — otherwise a forgotten
/// rescan_tape makes every consumption-fork assert vacuously pass while the
/// static coverage gate keeps certifying the assert SITE exists.
#[test]
fn an_applied_in_place_mutator_without_a_live_digest_is_refused() {
    let bus = PassBus::default();
    let mgr = PassManager::begin();
    let _p = enter_phase(CompilePhase::TrainBlock);
    let sched = mgr.scheduler();

    let scheduled = sched
        .schedule("CCR", None, || {
            record("CCR");
            record_disposition("CCR", PassDisposition::Applied { rewrites: 2 });
        })
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    let err = match scheduled.finish(&bus) {
        Err(e) => e,
        Ok(_) => panic!("Applied + MutatesInPlace + no digest must refuse"),
    };
    assert!(
        err.contains("no tape digest is live"),
        "refusal names the missing digest: {err}"
    );

    // Positive control: the same state with a rescan is admitted (CCR has
    // no Enforced channel, so the digest rule is the only live check).
    let tape = tiny_tape(3);
    let scheduled = sched
        .schedule("CCR", None, || ())
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    scheduled.rescan_tape(&tape);
    scheduled
        .finish(&bus)
        .expect("a live digest satisfies the obligation");

    // And a pass that DECLINED owes no digest: the obligation follows
    // Applied, not scheduling.
    let mgr2 = PassManager::begin();
    let _p2 = enter_phase(CompilePhase::TrainBlock);
    let scheduled = mgr2
        .scheduler()
        .schedule("CCR", None, || {
            record("CCR");
            record_disposition(
                "CCR",
                PassDisposition::Declined {
                    reason: nsl_codegen::pass_trace::DeclineReason::ModeOff,
                },
            );
        })
        .unwrap_or_else(|e| panic!("preconditions hold: {e}"));
    scheduled
        .finish(&bus)
        .expect("a declining mutator owes no digest");
}
