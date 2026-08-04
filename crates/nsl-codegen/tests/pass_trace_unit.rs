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

// ─── Dispositions (roadmap item 3) ─────────────────────────────────────────

/// A disposition presupposes an entry. Pinned as a PANIC rather than a
/// silently-dropped record: the whole point of the two-call shape is that
/// `record_disposition` cannot be used as a cheaper stand-in for `record`, and
/// a soft failure would let exactly that happen on any path a test does not
/// drive.
#[test]
#[should_panic(expected = "without a prior")]
fn a_disposition_without_an_entry_is_a_hard_error() {
    let _g = fresh();
    record_disposition("WGGO", PassDisposition::Applied { rewrites: 3 });
}

/// Repeat dispositions merge LAST-WINS. The production case is CCR under
/// `--checkpoint-stride auto`, where `ccr::plan` runs once per candidate
/// stride and only the final call describes the plan the build used.
#[test]
fn a_repeated_disposition_keeps_the_last_one() {
    let _g = fresh();
    record("CCR");
    record_disposition("CCR", PassDisposition::Declined {
        reason: DeclineReason::NoCandidates("nothing recomputable in any block segment"),
    });
    record_disposition("CCR", PassDisposition::Applied { rewrites: 6 });
    assert_eq!(
        dispositions(),
        vec![("CCR", PassDisposition::Applied { rewrites: 6 })],
        "the later disposition must replace the earlier one, not append"
    );
}

/// `reset` must clear BOTH halves. Clearing only the entry trace would leave
/// a disposition whose entry had been wiped, and the next
/// `record_disposition` in the same process would fire the entry assert on a
/// pass that had done nothing wrong.
#[test]
fn reset_clears_dispositions_as_well_as_entries() {
    let _g = fresh();
    record("FASE");
    record_disposition("FASE", PassDisposition::AdvisoryOnly);
    reset();
    assert!(dispositions().is_empty());
    assert!(observed().is_empty());
}

/// The `ran:` and `did not run:` lines must survive the addition of the
/// disposition block UNCHANGED — three gates parse them positionally by
/// splitting on `ran:` / `->` / `(` / `@` and on `did not run:` / `,`.
/// Dispositions therefore go on their own lines, under the same
/// `[pass-trace]` prefix so no new exec-marker token is introduced.
///
/// The `@Phase` suffix here is the phase-attribution layer's, not this one's:
/// nothing calls `enter_phase` in a unit test, so every pass is
/// `unattributed`. Pinning that spelling too is deliberate — it is what makes
/// this test fail if the disposition block ever starts writing INTO the
/// sequence line rather than after it.
#[test]
fn dispositions_render_on_new_lines_and_leave_the_sequence_lines_alone() {
    let _g = fresh();
    record("WGGO");
    record_disposition("WGGO", PassDisposition::Applied { rewrites: 12 });
    record("CSHA");
    record_disposition("CSHA", PassDisposition::Declined {
        reason: DeclineReason::NoCandidates("no attention boundary chain admitted a fused kernel"),
    });
    record("CPKD");
    record_disposition("CPKD", PassDisposition::AdvisoryOnly);
    let r = report();
    assert!(
        r.contains(
            "3 pass(es) ran: WGGO(OnWengert)@unattributed -> \
             CSHA(OnWengert)@unattributed -> CPKD(OnWengert)@unattributed"
        ),
        "the sequence line changed shape: {r}"
    );
    assert!(r.contains("\n[pass-trace] WGGO: applied, 12 rewrite(s)\n"), "{r}");
    assert!(
        r.contains(
            "\n[pass-trace] CSHA: declined, no candidates - no attention boundary \
             chain admitted a fused kernel\n"
        ),
        "a decline must name its category AND its detail: {r}"
    );
    assert!(r.contains("\n[pass-trace] CPKD: advisory only\n"), "{r}");
    // The disposition block is keyed off the entry trace, so a pass that was
    // never reached cannot appear in it.
    assert!(!r.contains("CCR: "), "an idle pass must not get a disposition line: {r}");
}

/// Every `DeclineReason` variant must render distinguishably. A category
/// whose text collides with another's is worse than no category: a gate
/// asserting on the collided text would pass for the wrong reason.
#[test]
fn every_decline_reason_renders_distinctly() {
    let _g = fresh();
    record("WGGO");
    let mut seen: Vec<String> = Vec::new();
    for reason in [
        DeclineReason::ModeOff,
        DeclineReason::NoCandidates("detail-a"),
        DeclineReason::PreconditionViolated("detail-b"),
        DeclineReason::FeatureDisabled("detail-c"),
        DeclineReason::BudgetInfeasible,
    ] {
        record_disposition("WGGO", PassDisposition::Declined { reason });
        let line = report()
            .lines()
            .find(|l| l.starts_with("[pass-trace] WGGO: "))
            .expect("WGGO disposition line")
            .to_string();
        assert!(!seen.contains(&line), "two DeclineReasons render identically: {line}");
        seen.push(line);
    }
    assert_eq!(seen.len(), 5, "a variant was added without a rendering");
}

/// End-to-end through REAL pass entry points, not just the recorder.
///
/// `fase::plan` is the cheapest registered pass to drive from a test, and it
/// exercises both halves: `accumulation == 1` is `FaseMode::Passthrough`,
/// which is FASE's off state (there is no separate flag), and anything above
/// that rewrites the backward into one phase per micro-batch.
///
/// This is the coverage the disposition arms would otherwise not have: the
/// production driver short-circuits `--csha off` / `--wggo off` BEFORE
/// invoking the pass, so several `ModeOff` arms are reachable only from
/// direct callers like this one.
#[test]
fn a_real_pass_reports_its_own_disposition() {
    use nsl_codegen::fase::{FaseConfig, FaseOptimizer};

    let _g = fresh();
    let _ = nsl_codegen::fase::plan(&FaseConfig {
        accumulation: 1,
        ..Default::default()
    });
    assert_eq!(
        dispositions(),
        vec![(
            "FASE",
            PassDisposition::Declined {
                reason: DeclineReason::ModeOff
            }
        )],
        "accumulation=1 is FASE declining, not FASE applying zero rewrites"
    );

    reset();
    let plan = nsl_codegen::fase::plan(&FaseConfig {
        accumulation: 4,
        optimizer: FaseOptimizer::AdamW,
        ..Default::default()
    });
    assert_eq!(plan.backward_phases.len(), 4, "fixture assumption");
    assert_eq!(
        dispositions(),
        vec![("FASE", PassDisposition::Applied { rewrites: 4 })],
        "the reported count must be the pass's real unit of work, not a constant"
    );
}

/// The same, through a pass whose `Off` arm the production driver never
/// reaches — `stmt.rs` tests `mode_str == \"off\"` and skips the call, so
/// `--csha off` renders as CSHA under "did not run". Driving `csha::run`
/// directly is what keeps that arm from being written-and-never-executed.
#[test]
fn csha_off_mode_declines_rather_than_applying_nothing() {
    use nsl_codegen::wengert::WengertList;

    let _g = fresh();
    let empty = WengertList {
        ops: Vec::new(),
        output: 0,
        var_names: std::collections::HashMap::new(),
        var_types: std::collections::HashMap::new(),
    };
    let plan = nsl_codegen::csha::run_on_wengert(&empty, "A100", "off", None, None, 8, None)
        .expect("\"off\" is a valid CshaMode");
    assert_eq!(plan.mode.as_str(), "off", "fixture assumption");
    assert_eq!(
        dispositions(),
        vec![(
            "CSHA",
            PassDisposition::Declined {
                reason: DeclineReason::ModeOff
            }
        )]
    );
}

/// A stage-rank inversion that a declared dependency edge MANDATES must say
/// so, and one no edge explains must keep the generic cross-phase wording.
///
/// WGGO(OnWengert) before FASE(PreExtraction) is the canonical case: it reads
/// as an inversion by stage rank and is exactly the order the wggo_overrides
/// edge requires. Before the edges existed this line blessed every inversion
/// with the same "expected, not a defect" shrug; pinning both wordings is
/// what keeps the upgraded arm from quietly regressing to the shrug.
#[test]
fn a_stage_inversion_mandated_by_a_dependency_edge_says_so() {
    let _g = fresh();
    record("WGGO");
    record("FASE");
    let r = report();
    assert!(
        r.contains(
            "STAGE ORDER: WGGO ran before FASE, inverting their declared \
             PipelineStage — required by the declared dependency: FASE \
             consumes wggo_overrides, which WGGO produces"
        ),
        "the mandated inversion must be attributed to its edge: {r}"
    );

    reset();
    record("CCR"); // OnAdjoint, rank 3
    record("FASE"); // PreExtraction, rank 0 — inverted, and no CCR->FASE edge
    let r = report();
    assert!(
        r.contains("STAGE ORDER: CCR ran before FASE")
            && r.contains("a cross-phase inversion is expected, not a defect"),
        "an unexplained inversion must keep the generic wording: {r}"
    );
}

/// A declared InvocationOrdered edge violated by the observed trace is
/// printed by the report — a detector with no production caller is why a
/// false claim survives, so the checker's production caller is the report
/// itself.
#[test]
fn a_dependency_order_violation_is_printed_by_the_report() {
    let _g = fresh();
    record("CSHA");
    record("WGGO");
    let r = report();
    assert!(
        r.contains(
            "DEPENDENCY ORDER: CSHA was invoked before WGGO, yet CSHA \
             consumes wggo_overrides, which WGGO produces"
        ),
        "the violation must be printed with its edge: {r}"
    );

    // The healthy order prints nothing.
    reset();
    record("WGGO");
    record("CSHA");
    assert!(
        !report().contains("DEPENDENCY ORDER"),
        "no violation, no line: {}",
        report()
    );
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
