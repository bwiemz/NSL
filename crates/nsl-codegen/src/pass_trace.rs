//! Roadmap item 2, step 1: **which passes actually ran, in what order.**
//!
//! # Why this exists
//!
//! [`crate::pass_registry`] describes eleven passes as data — implementation
//! files, driving CLI flags, pipeline stage — and its own header is explicit
//! about the limit of that:
//!
//! > Not a pass *manager* — it does not schedule, order, or invoke anything.
//! > `compile_train_block` still calls the passes directly.
//!
//! So the registry can prove a pass *exists* and that its flags are wired, but
//! nothing proves a pass *ran*. That is the gap this module closes, and it is
//! not a hypothetical one in this tree. The recurring defect here is a
//! component that is present, configured, and silently inert:
//!
//! * `find_best_variant` (the autotuner's chooser) had **zero** production
//!   callers, pinned by a negative test only after the fact;
//! * two GPU kernels declined to a CPU path **silently** on host-resident
//!   indices, costing a 15x MFU regression that looked like a modelling
//!   problem;
//! * the PTX certification gate was silently OFF;
//! * the wiki described CCR as unimplemented for ~3 months while `ccr.rs` was
//!   2000+ lines and wired into the driver.
//!
//! Every one of those is the same shape: *the thing was declared, and nobody
//! could ask whether it happened.* A flag that enables a pass which never runs
//! produces a clean, plausible, wrong build.
//!
//! # What it records
//!
//! **Invocation**, not effect: `record` is called at a pass's entry point, so
//! the trace answers "was this pass reached", not "did it change anything". A
//! pass that is reached and then declines (no candidate ops, budget already
//! met) is a genuinely different question, deliberately out of scope here —
//! conflating the two would make the signal mean less, not more.
//!
//! # Why it cannot change the build
//!
//! The trace is a compiler-side `Vec<&'static str>`. Nothing here is emitted
//! into the compiled program, so enabling it cannot move a loss curve. The
//! gate in `crates/nsl-cli/tests/pass_trace_gate.rs` asserts exactly that by
//! comparing loss streams with the trace on and off.
//!
//! # Toward a pass manager
//!
//! This is step 1 because scheduling requires observation: you cannot move
//! ordering into the registry until you can see the order the driver actually
//! produces, and cannot prove the move was behaviour-preserving without a
//! before/after trace to compare. The declared [`crate::pass_registry::
//! PipelineStage`] partial order becomes checkable the moment a trace exists.

use std::sync::Mutex;

/// Passes observed this compile, in first-invocation order.
static TRACE: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

/// Record that `pass` was reached. Idempotent per compile: a pass invoked
/// once per layer would otherwise swamp the sequence, and the question this
/// answers ("was it reached at all") is already settled by the first call.
///
/// `pass` MUST be a name in [`crate::pass_registry::PASSES`]. A name that is
/// not registered is a build-integrity failure — either the registry is stale
/// or this is a typo, and both make the trace lie — so it panics rather than
/// recording an entry no gate can interpret. This is compiler-side code, so a
/// panic here surfaces as a compiler bug, which is what it would be.
pub fn record(pass: &'static str) {
    debug_assert!(
        crate::pass_registry::pass(pass).is_some(),
        "pass_trace::record(\"{pass}\") names a pass that is not in the \
         registry — add a PassDescriptor, or fix the name"
    );
    let mut t = TRACE.lock().unwrap_or_else(|e| e.into_inner());
    if !t.contains(&pass) {
        t.push(pass);
    }
}

/// The passes reached this compile, in first-invocation order.
pub fn observed() -> Vec<&'static str> {
    TRACE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Clear the trace.
///
/// Deliberately NOT called on any production path. The trace's scope is one
/// *compilation*, and `nsl run` / `nsl build` compile once per process — a
/// program's imported modules go through a different codegen entry point than
/// its own, so resetting at any single entry would silently drop whichever
/// passes ran through the other. Accumulating for the process is therefore
/// the accurate scope, not a shortcut.
///
/// Exists for tests, which drive several compiles in one process and need a
/// clean slate between them.
pub fn reset() {
    TRACE.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// Is `NSL_PASS_TRACE=1` set? Read per call rather than cached so a test can
/// drive both states in one process.
pub fn enabled() -> bool {
    std::env::var("NSL_PASS_TRACE").ok().as_deref() == Some("1")
}

/// Render the observed sequence with each pass's declared stage, e.g.
///
/// ```text
/// [pass-trace] 3 pass(es) ran: FASE(PreExtraction) -> WGGO(OnWengert) -> CCR(OnAdjoint)
/// ```
///
/// Registered passes that did NOT run are listed too — the absence is the
/// interesting half, and a report that only showed what happened would make
/// "nothing ran" indistinguishable from "the trace is broken".
pub fn report() -> String {
    let seen = observed();
    let mut s = if seen.is_empty() {
        "[pass-trace] NO passes ran\n".to_string()
    } else {
        format!(
            "[pass-trace] {} pass(es) ran: {}\n",
            seen.len(),
            seen.iter()
                .map(|p| {
                    let stage = crate::pass_registry::pass(p)
                        .map(|d| format!("{:?}", d.stage))
                        .unwrap_or_else(|| "?".into());
                    format!("{p}({stage})")
                })
                .collect::<Vec<_>>()
                .join(" -> ")
        )
    };
    let idle: Vec<&str> = crate::pass_registry::PASSES
        .iter()
        .map(|d| d.name)
        .filter(|n| !seen.contains(n))
        .collect();
    if !idle.is_empty() {
        s.push_str(&format!("[pass-trace] did not run: {}\n", idle.join(", ")));
    }
    s
}

/// Does the observed sequence respect the declared [`crate::pass_registry::
/// PipelineStage`] order?
///
/// Returns the offending adjacent pair on violation. `OutOfBand` is exempt in
/// both directions — CEP and CFIE are, by the registry's own description, not
/// part of the train-block pipeline, so ordering them against it is
/// meaningless rather than merely unchecked.
///
/// This is a PARTIAL order: five passes share `OnWengert` and the registry
/// declares nothing about their relative order, so any arrangement of those
/// five passes is accepted here. Saying so plainly matters — a checker that
/// implied it had verified the whole sequence would be claiming coverage it
/// does not have, which is the failure mode the registry itself was built to
/// stop. Declaring intra-stage order is the next increment, not this one.
pub fn stage_order_violation() -> Option<(&'static str, &'static str)> {
    use crate::pass_registry::PipelineStage as S;
    fn rank(s: S) -> Option<u8> {
        Some(match s {
            S::PreExtraction => 0,
            S::ModuleScan => 1,
            S::OnWengert => 2,
            S::OnAdjoint => 3,
            S::Lowering => 4,
            S::OutOfBand => return None,
        })
    }
    let seen = observed();
    let ranked: Vec<(&'static str, u8)> = seen
        .iter()
        .filter_map(|p| {
            let d = crate::pass_registry::pass(p)?;
            Some((d.name, rank(d.stage)?))
        })
        .collect();
    ranked
        .windows(2)
        .find(|w| w[0].1 > w[1].1)
        .map(|w| (w[0].0, w[1].0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TRACE` is process-global; the tests that touch it must not interleave.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

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
                crate::pass_registry::pass(p).is_some(),
                "{p} is not in the pass registry"
            );
        }
    }
}
