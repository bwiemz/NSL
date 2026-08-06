//! Item 2 step 6: the pass-ordering DECISION, owned per compile.
//!
//! # What this is
//!
//! The bus ([`crate::pass_bus`]) DECLARES the pass-to-pass edges and their
//! order claims; the trace ([`crate::pass_trace`]) OBSERVES what ran. Both
//! are process-scoped, and #466's HIGH review finding is why that matters:
//! a per-compile ordering argument made over process-scoped state is
//! defeated by real paths — multi-module builds interleave several compiles'
//! events in one process, so "the consumer's position precedes the
//! producer's" can be true across two compiles while both compiles were
//! individually well-ordered (and vice versa). That is why every ordering
//! check before this module was ADVISORY: refusing a compile on
//! process-global evidence would refuse correct builds.
//!
//! The `PassManager` closes that gap. It is owned by the `Compiler` — one
//! per compile, like the bus — and anchors a compile EPOCH
//! ([`crate::pass_trace::begin_epoch`]) that every `pass_trace::record`
//! stamps. Its view of "what ran, in what order" is therefore exactly this
//! compile's, which is what makes the ordering decision ENFORCEABLE: a
//! declared `InvocationOrdered` edge whose consumer ran before its producer
//! *within one epoch* is a real scheduling defect, not cross-compile noise,
//! and the driver refuses the compile instead of printing a line nobody
//! reads.
//!
//! # What this is not
//!
//! Not a scheduler. The driver (`stmt.rs`, `kernel.rs`) still decides when
//! each pass runs; this module owns the JUDGMENT of that order against the
//! declared edges. Inverting control — the manager invoking passes from the
//! declarations — is the step after this one, and it should be built on the
//! per-compile substrate this module establishes, because a scheduler that
//! cannot soundly see its own compile's history cannot enforce anything.
//!
//! # Recording stays at the callee
//!
//! `pass_trace::record` remains the single choke point at each pass's entry
//! function (the step-2 lesson: call-site coverage missed 16 of 26 sites).
//! The epoch is ambient (thread-local, RAII-scoped), so the callee-side
//! recording gains per-compile attribution without any call site changing.

use crate::pass_bus::Channel;
use crate::pass_registry::CompilePhase;

/// Test-only knob, same convention as `NSL_WGGO_FORCE_STALE_TABLE` /
/// `NSL_CPDT_FORCE_STALE_PLAN`: force the refusal arm so the enforcement is
/// gate-testable without engineering a real scheduling defect (which would
/// require a broken driver by construction).
const FORCE_KNOB: &str = "NSL_FORCE_DEPENDENCY_ORDER_VIOLATION";

/// The per-compile ordering authority. One per `Compiler`, constructed with
/// it; dropping it (the compile ending, however it ends) restores the epoch
/// that was active before, so a nested compile cannot leak its epoch over
/// the tail of an outer one.
pub struct PassManager {
    epoch: u64,
    prev_epoch: u64,
}

impl PassManager {
    /// Anchor a new compile epoch on this thread.
    pub fn begin() -> Self {
        let (epoch, prev_epoch) = crate::pass_trace::begin_epoch();
        Self { epoch, prev_epoch }
    }

    /// This compile's epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The passes THIS compile reached, in first-invocation order, with the
    /// phase each was reached in.
    pub fn per_compile_trace(&self) -> Vec<(&'static str, Option<CompilePhase>)> {
        crate::pass_trace::per_compile_view(self.epoch)
    }

    /// Declared `InvocationOrdered` edges whose consumer ran before its
    /// producer WITHIN THIS COMPILE, as `(producer, consumer, channel)`.
    ///
    /// Same edge semantics as the process-scoped
    /// [`crate::pass_bus::dependency_order_violations`] (one shared core),
    /// different evidence: this view cannot be polluted by another compile
    /// in the same process, which is what upgrades the answer from an
    /// advisory report line to a refusable fact.
    pub fn dependency_order_violations(&self) -> Vec<(&'static str, &'static str, Channel)> {
        let seen: Vec<&'static str> =
            self.per_compile_trace().into_iter().map(|(p, _)| p).collect();
        crate::pass_bus::dependency_order_violations_in(&seen)
    }

    /// The enforcement decision: `Err` with the full refusal message when a
    /// declared invocation order was violated in this compile (or when the
    /// test knob forces the arm). The driver turns the message into its own
    /// error type at the single wrapper exit — this module does not know
    /// about `CodegenError`.
    pub fn enforce_dependency_order(&self) -> Result<(), String> {
        let forced = std::env::var(FORCE_KNOB).map(|v| v == "1").unwrap_or(false);
        let mut violations = self.dependency_order_violations();
        if violations.is_empty() && !forced {
            return Ok(());
        }
        if violations.is_empty() {
            // Forced with nothing real: name the knob so the message cannot
            // be mistaken for a genuine defect while still exercising the
            // exact refusal path.
            return Err(format!(
                "dependency-order violation forced by {FORCE_KNOB}=1 (no real \
                 violation exists in this compile) — this is the test arm of \
                 the pass manager's enforcement"
            ));
        }
        violations.sort_by_key(|(producer, consumer, _)| (*producer, *consumer));
        let list = violations
            .iter()
            .map(|(producer, consumer, channel)| {
                format!(
                    "{consumer} was invoked before {producer}, its declared \
                     producer on the '{}' channel",
                    channel.descriptor().name
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        Err(format!(
            "pass dependency order violated in this compile: {list}. The \
             edge is declared InvocationOrdered on the pass bus, and the \
             evidence is this compile's own invocation ledger (epoch \
             {epoch}), so this is a pass-scheduling defect in the compiler, \
             not a property of the program being compiled — please report \
             it. NSL_PASS_TRACE=1 prints the full pass trace and bus \
             report.",
            epoch = self.epoch
        ))
    }
}

impl Drop for PassManager {
    fn drop(&mut self) {
        crate::pass_trace::restore_epoch(self.prev_epoch);
    }
}
