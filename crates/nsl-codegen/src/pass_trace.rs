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
//! Read that literally for analysis-only commands: `nsl check
//! --training-report` builds its report BY running the FASE and PCA planners,
//! so those are reported as having run. That is accurate under this
//! definition, not a false positive — but it does mean "ran" must not be read
//! as "affected the emitted program".
//!
//! Where a pass entry doubles as a predicate, the record goes where the pass
//! genuinely starts, not at the shared helper: `pca_detect::detect` is also a
//! yes/no query on the FA-kernel path (it recorded PCA on every model with a
//! dataset block, packing on or off), and `wggo_apply::apply` is skipped
//! entirely by WGGO's shape-incompatibility refusal (which would have reported
//! "did not run" for a pass that ran and declined).
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

use crate::pass_registry::CompilePhase;
use std::sync::Mutex;

/// Passes observed this compile, in first-invocation order, each tagged with
/// the driver phase that was active when it was reached (`None` = the pass ran
/// outside every declared phase — see [`enter_phase`]).
static TRACE: Mutex<Vec<(&'static str, Option<CompilePhase>)>> = Mutex::new(Vec::new());

/// The phase currently executing. A plain global rather than a thread-local:
/// codegen is single-threaded, and a thread-local would silently report
/// `None` if that ever changed, which is the failure mode this is meant to
/// surface loudly.
static CURRENT_PHASE: Mutex<Option<CompilePhase>> = Mutex::new(None);

/// RAII guard restoring the previously-active phase on drop.
pub struct PhaseGuard(Option<CompilePhase>);

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        *CURRENT_PHASE.lock().unwrap_or_else(|e| e.into_inner()) = self.0;
    }
}

/// Mark the driver phase now executing, until the returned guard drops.
///
/// Wraps the statements in `compile_entry_returning_plan` that actually reach
/// a registered pass. Any pass recorded outside every scope is reported
/// `unattributed`, which is how a NEW driver — the thing that made
/// `PipelineStage` unusable as a scheduling key in the first place — announces
/// itself instead of being discovered by backtrace months later.
#[must_use = "the phase ends when the guard drops; binding to `_` ends it immediately"]
pub fn enter_phase(phase: CompilePhase) -> PhaseGuard {
    let mut cur = CURRENT_PHASE.lock().unwrap_or_else(|e| e.into_inner());
    let prev = *cur;
    *cur = Some(phase);
    PhaseGuard(prev)
}

/// The phase each observed pass ran in.
pub fn observed_phases() -> Vec<(&'static str, Option<CompilePhase>)> {
    TRACE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Passes whose observed phase differs from the registry's declaration, as
/// `(pass, declared, observed)`. A pass declared [`CompilePhase::OutOfBand`]
/// is expected to be unattributed, since those drivers are CLI subcommands
/// outside the compile pipeline.
pub fn phase_mismatches() -> Vec<(&'static str, CompilePhase, Option<CompilePhase>)> {
    observed_phases()
        .into_iter()
        .filter_map(|(name, got)| {
            let want = crate::pass_registry::pass(name)?.phase;
            let ok = match want {
                CompilePhase::OutOfBand => got.is_none(),
                w => got == Some(w),
            };
            (!ok).then_some((name, want, got))
        })
        .collect()
}

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
    // NOT debug_assert: CI ships release builds, where that would compile
    // out and let a typo'd name be recorded, rendered as `NAME(?)`, AND leave
    // the real pass listed under "did not run" — the report lying in both
    // halves at once. Once per pass per compile, so the cost is nil.
    assert!(
        crate::pass_registry::pass(pass).is_some(),
        "pass_trace::record(\"{pass}\") names a pass that is not in the \
         registry — add a PassDescriptor, or fix the name"
    );
    let phase = *CURRENT_PHASE.lock().unwrap_or_else(|e| e.into_inner());
    let mut t = TRACE.lock().unwrap_or_else(|e| e.into_inner());
    if !t.iter().any(|(p, _)| *p == pass) {
        t.push((pass, phase));
    }
}

/// The passes reached this compile, in first-invocation order.
pub fn observed() -> Vec<&'static str> {
    TRACE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(p, _)| *p)
        .collect()
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
    *CURRENT_PHASE.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Is `NSL_PASS_TRACE=1` set? Read per call rather than cached so a test can
/// drive both states in one process.
pub fn enabled() -> bool {
    std::env::var("NSL_PASS_TRACE").ok().as_deref() == Some("1")
}

/// The phase `pass` was observed running in, if it ran.
fn phase_of(pass: &str) -> Option<CompilePhase> {
    observed_phases()
        .into_iter()
        .find(|(p, _)| *p == pass)
        .and_then(|(_, c)| c)
}

/// Render the observed sequence with each pass's declared stage, e.g.
///
/// ```text
/// [pass-trace] 3 pass(es) ran: WGGO(OnWengert)@KernelPrepass -> FASE(PreExtraction)@TrainBlock -> CCR(OnAdjoint)@TrainBlock
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
                    // The phase is what explains an order that looks wrong:
                    // WGGO(OnWengert)@KernelPrepass before FASE(PreExtraction)
                    // @TrainBlock is not an inversion once both are shown.
                    let phase = phase_of(p)
                        .map(|c| format!("{c:?}"))
                        .unwrap_or_else(|| "unattributed".into());
                    format!("{p}({stage})@{phase}")
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
