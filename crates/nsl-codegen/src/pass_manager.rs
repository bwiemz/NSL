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
//! # Scheduling: every in-pipeline pass (Milestone C)
//!
//! The step after judgment is INVERTING CONTROL — the manager checking passes
//! from the declarations rather than grading the driver afterwards. That is
//! [`PassScheduler`]. WRGA was wired first (item 2 step 7) because its
//! declarations were richest and its invocation narrowest; Milestone C
//! completed the set: every production invocation of the nine in-pipeline
//! passes — WRGA, WGGO (both the kernel prepass and the in-place replan),
//! CSHA, CCR, CPDT (all three offer sites), CPKD, FASE, PCA (both decision
//! sites), MemoryPlanner (both drivers) — now goes through `schedule(..)` and
//! settles its postconditions with [`Scheduled::finish`] (or, for exactly one
//! site, [`Scheduled::defer_postconditions`] with its reason). What stays
//! outside by design: CEP and CFIE (OutOfBand, empty phase declarations —
//! `schedule` refuses them structurally), the PCA predicate calls (recording
//! at `pca_detect::detect` exists precisely so a predicate use is not
//! reported as the pass running), `nsl check --training-report`'s Analysis
//! walk (no Compiler exists there), and unit tests driving pass entries
//! directly. `pass_scheduler_coverage.rs` pins the set in both directions.
//!
//! The driver (stmt.rs) still owns the CALL ORDER — the bodies are
//! interleaved with the lowering's own data flow, and the value order that
//! matters is declared on the bus and enforced per-compile at the wrapper
//! exit. What stmt.rs no longer owns is the AUTHORITY: whether a pass may
//! run where it is invoked, whether its product reached its consumers, and
//! whether the tape its plan indexes is still the tape it scanned are all
//! judged HERE, from the registry and bus declarations.
//!
//! What the scheduler enforces, all of it read from the declarations rather
//! than hard-coded per pass:
//!
//! | when | check | refuses? | declared by |
//! |---|---|---|---|
//! | before | the pass is registered | yes | `pass_registry::PASSES` |
//! | before | this driver phase is one the pass declares | yes, when a phase is installed | `PassDescriptor::phases` |
//! | before | which `InvocationOrdered` predecessors have run | **no — traced** | `ChannelDescriptor::consumed_by_passes` |
//! | after | `Applied` ⇒ its `Enforced` channels are published | yes | `applied_implies_published` |
//! | later | the tape a plan's positional refs were captured against is unchanged when the plan is consumed | yes | `TapeAccess` / `TapeRef::PositionalIndex` |
//!
//! The predecessor row is an OBSERVATION, not a refusal, and the reason is
//! written out at its site: "has my producer run yet" is not soundly
//! decidable before the compile finishes, and every proxy for it that was
//! tried refuses correct compiles. Inversion enforcement stays in
//! [`PassManager::enforce_dependency_order`], which is sound because it runs
//! once both passes are in the ledger.
//!
//! The phase row is live on every production train-block shape:
//! `compile_train_block` installs its own `TrainBlock` scope at entry
//! (Milestone C), so even a `train` block nested inside `fn main():` reaches
//! the scheduler as `Some(TrainBlock)`. The check is SKIPPED only where no
//! scope exists at all — unit tests driving the compiler directly, and
//! out-of-band CLI drivers. See the `None` arm of `schedule`.
//!
//! The last one is the tape half, and it is the one no existing gate covered:
//! WRGA scans the primal, and its plan is consumed ~400 lines later where
//! `effective_primal` forks onto `plan.prune.pruned`. Positional references
//! are valid only against the list state that produced them, so anything
//! mutating that list in between silently invalidates the plan. See
//! [`PassScheduler::assert_tape_unchanged_since`].
//!
//! # Why a Copy handle and not `&self`
//!
//! A pass body needs `&mut Compiler`, and the manager LIVES on the Compiler.
//! `compiler.passes.schedule(|| … compiler …)` cannot borrow-check. The
//! handle carries the epoch by value, so the borrow of `compiler.passes` ends
//! before the body runs.
//!
//! # Recording stays at the callee
//!
//! `pass_trace::record` remains the single choke point at each pass's entry
//! function (the step-2 lesson: call-site coverage missed 16 of 26 sites).
//! The epoch is ambient (thread-local, RAII-scoped), so the callee-side
//! recording gains per-compile attribution without any call site changing.

use crate::pass_bus::{Channel, PassBus};
use crate::pass_registry::CompilePhase;
use crate::wengert::WengertList;

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
    /// `Drop` mutates a THREAD-LOCAL. A manager dropped on a different
    /// thread than `begin()` ran on would clobber the dropping thread's
    /// epoch with a foreign one and leave the origin thread on a dead
    /// epoch — records would leak out of their compile's view and
    /// violations could be MISSED. No production path moves a Compiler
    /// across threads, and this marker keeps it that way at compile time:
    /// a raw pointer makes the type `!Send + !Sync`.
    _thread_bound: std::marker::PhantomData<*const ()>,
}

impl PassManager {
    /// Anchor a new compile epoch on this thread.
    pub fn begin() -> Self {
        let (epoch, prev_epoch) = crate::pass_trace::begin_epoch();
        Self {
            epoch,
            prev_epoch,
            _thread_bound: std::marker::PhantomData,
        }
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

/// A `Copy` handle to this compile's manager, so a pass body may borrow the
/// `Compiler` mutably while a schedule call is in flight. Carries the epoch by
/// value — see the module header.
#[derive(Clone, Copy, Debug)]
pub struct PassScheduler {
    epoch: u64,
}

impl PassManager {
    /// A handle for scheduling passes in this compile.
    pub fn scheduler(&self) -> PassScheduler {
        PassScheduler { epoch: self.epoch }
    }
}

/// A pass that ran under the scheduler, awaiting its postconditions.
///
/// `#[must_use]` because dropping this instead of calling [`Self::finish`]
/// silently skips the postcondition checks — the value would still be there
/// and the compile would proceed, which is exactly the "detector nobody calls"
/// shape the bus module warns about.
#[must_use = "call finish(&bus) — dropping a Scheduled skips the pass's postconditions"]
pub struct Scheduled<R> {
    value: R,
    pass: &'static str,
    epoch: u64,
}

/// A structural digest of a tape, for the positional-reference staleness rule.
///
/// Records what a positional reference's validity depends on: the op
/// sequence and each op's identity. Deliberately NOT the name/type side
/// tables — renaming a var moves nothing. A 64-bit digest can in principle collide;
/// this is a tripwire for a compiler-scheduling defect, not a security
/// boundary, and the alternative (retaining a full clone of every scanned
/// tape for the rest of the compile) costs memory proportional to the model on
/// every build.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TapeDigest {
    len: usize,
    hash: u64,
}

impl TapeDigest {
    fn of(list: &WengertList) -> Self {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        list.ops.len().hash(&mut h);
        for op in &list.ops {
            op.id.hash(&mut h);
            op.result.hash(&mut h);
            op.inputs.hash(&mut h);
            op.saved_for_backward.hash(&mut h);
            op.checkpointed.hash(&mut h);
            // `PrimalOp` is not `Hash` (it carries an f64 in `Constant`), so
            // its `Debug` stands in. That distinguishes every variant and
            // every payload that can affect POSITIONS — in particular a
            // rewritten `Passthrough` name, which is a real in-place rewrite
            // (`fuse_swiglu_gate_backward` does exactly it) that a
            // discriminant-only digest would miss.
            //
            // It is not a total injection and is not claimed to be:
            // `Constant(NaN)` and `Constant(-NaN)` format identically. A
            // constant's sign cannot move an op's position, which is the only
            // property the positional-reference rule depends on.
            format!("{:?}", op.op).hash(&mut h);
        }
        list.output.hash(&mut h);
        Self {
            len: list.ops.len(),
            hash: h.finish(),
        }
    }
}

thread_local! {
    /// `(epoch, pass) -> digest of the tape that pass scanned`.
    ///
    /// Thread-local for the reason [`crate::pass_trace`]'s phase is: a compile
    /// never spans threads, and 15 integration-test binaries drive full
    /// compiles in parallel, so a process-global map would let one compile's
    /// digest answer another's question.
    static SCANNED_TAPES: std::cell::RefCell<Vec<((u64, &'static str), TapeDigest)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

impl PassScheduler {
    /// Invoke `body` AS the registered pass `pass`, enforcing what the
    /// registry and bus declare about it.
    ///
    /// `tape`, when supplied, is the list the pass scans; its digest is
    /// retained so [`Self::assert_tape_unchanged_since`] can check the
    /// positional-reference rule where the pass's product is consumed.
    ///
    /// `Err` means a COMPILER defect — the driver invoked a pass somewhere its
    /// declarations do not admit — so the body does not run and the message is
    /// written to be reported, not worked around.
    pub fn schedule<R>(
        self,
        pass: &'static str,
        tape: Option<&WengertList>,
        body: impl FnOnce() -> R,
    ) -> Result<Scheduled<R>, String> {
        let desc = crate::pass_registry::pass(pass).ok_or_else(|| {
            format!(
                "pass '{pass}' was scheduled but is not in pass_registry::PASSES \
                 — the scheduler cannot check declarations it does not have"
            )
        })?;

        // PRE 1: the driver phase must be one the registry declares. An empty
        // declaration means "expected unattributed" (an OutOfBand subcommand),
        // which is not a schedulable pass.
        let phase = crate::pass_trace::current_phase();
        if desc.phases.is_empty() {
            return Err(format!(
                "pass '{pass}' declares no compile phase (it is an out-of-band \
                 driver) and so cannot be scheduled"
            ));
        }
        match phase {
            Some(p) if desc.phases.contains(&p) => {}
            Some(p) => {
                return Err(format!(
                    "pass '{pass}' was scheduled from phase {p:?}, which is not \
                     among its declared phases {:?}. Either the driver grew a \
                     new invocation site or the registry is stale — both are \
                     compiler defects, please report.",
                    desc.phases
                ));
            }
            // UNSCOPED, and deliberately not a refusal.
            //
            // Only five call sites install a phase scope (`main_entry`,
            // `entry_points`, `kernel`, `standalone`, `training_report`), and
            // plenty of legitimate compiles reach a pass without one — every
            // unit test that drives the compiler directly, and any entry point
            // not yet wrapped. `pass_trace` already models this: an
            // unattributed pass is REPORTED, because "a driver nobody wrapped
            // in enter_phase" is how a new driver announces itself, not a
            // defect in the pass.
            //
            // Refusing here would fire on correct compiles, which is the
            // failure `pass_bus`'s `Invariant` doc names outright: a rule that
            // fires on correct behaviour trains its reader to stop reading it.
            //
            // Measured coverage, so the claim is not overstated: every
            // train-block shape now reaches this with `Some(TrainBlock)` —
            // `compile_train_block` installs the scope at its own entry
            // (Milestone C), so the nested-in-`fn main():` path that used to
            // arrive phase-less via `compile_user_functions` is covered by
            // construction, as are lambdas, model/agent methods, `@test`
            // fns and module compiles. What still legitimately arrives as
            // `None`: unit tests driving pass entries or the compiler
            // directly, and out-of-band CLI drivers (`nsl profile`,
            // `nsl cep`) that construct no Compiler at all. Note
            // `pass_trace::phase_mismatches` classifies the same `None` as a
            // MISMATCH for reporting; the report may say so, the scheduler
            // may not refuse on it.
            //
            // The trace line below prints which case applied, so "unscoped" is
            // visible rather than indistinguishable from a pass.
            None => {}
        }

        // PRE 2: the declared InvocationOrdered predecessors, OBSERVED — not
        // refused on.
        //
        // The refusal is not soundly decidable here, and the first version of
        // this function shipped one that was not. "Consumer ran before its
        // producer" needs to know the producer WILL run, which at schedule
        // time is undecidable: a predecessor that never runs is not an
        // inversion at all, just a feature that is off, and the consumer
        // legitimately reads the channel empty (each channel's `empty_means`
        // says what that means).
        //
        // The obvious proxy — "the channel is already published, so its
        // producer must have run" — is wrong twice over. It was first written
        // against `pass_bus::traffic`, which is PROCESS-scoped: in a
        // multi-module build one module's publish leaves `publishes > 0` for
        // every later module, so a later compile whose own epoch has no WGGO
        // would be refused. That is the #466 finding exactly. And even scoped
        // to this compile's bus it stays wrong, because publisher and
        // recorded pass legitimately differ — `adapter_prescan_plan`'s own
        // descriptor says it is "published by the driver prescan, which never
        // records WRGA as having run, so the antecedent cannot be
        // established for it".
        //
        // Enforcement therefore stays where it is sound:
        // [`PassManager::enforce_dependency_order`], which fires only when
        // BOTH passes are in this compile's ledger and the order is actually
        // inverted. The scheduler contributes the observation to the trace so
        // a missing predecessor is visible at the point of invocation.
        let seen: Vec<&'static str> = self.trace().into_iter().map(|(p, _)| p).collect();
        let missing: Vec<&'static str> = crate::pass_bus::required_predecessors(pass)
            .into_iter()
            .filter(|(producer, _)| !seen.contains(producer))
            .map(|(producer, _)| producer)
            .collect();

        // A pass legitimately re-invoked in one compile (WGGO's in-place
        // replan is the in-tree example) rescans a possibly-different tape;
        // the LATEST scan is the one its live plan refers to. A re-invocation
        // WITHOUT a tape must CLEAR the slot rather than leave the previous
        // digest live — otherwise the staleness check would answer a question
        // about a scan that this invocation superseded, and refuse a build
        // whose plan no longer holds those references at all.
        SCANNED_TAPES.with(|m| {
            let mut m = m.borrow_mut();
            let key = (self.epoch, pass);
            match (tape, m.iter().position(|(k, _)| *k == key)) {
                (Some(list), Some(i)) => m[i].1 = TapeDigest::of(list),
                (Some(list), None) => m.push((key, TapeDigest::of(list))),
                (None, Some(i)) => {
                    m.swap_remove(i);
                }
                (None, None) => {}
            }
        });

        Self::trace_line(format_args!(
            "-> {pass} phase={} epoch={} tape={} predecessors={}",
            match phase {
                Some(p) => format!("{p:?}"),
                None => "unscoped(phase check skipped)".to_string(),
            },
            self.epoch,
            tape.map(|t| t.ops.len().to_string())
                .unwrap_or_else(|| "none".into()),
            if missing.is_empty() {
                "all-ran".to_string()
            } else {
                format!("not-yet-run{missing:?}")
            },
        ));

        let value = body();

        // `dispositions_in`, NOT `dispositions()`: the process view dedupes by
        // name across every epoch, so on a multi-compile process it would
        // print ANOTHER compile's disposition for the pass that just ran —
        // cross-compile attribution in the very line a person reads while
        // diagnosing cross-compile attribution. (The old `.rev()` was also a
        // no-op once that view held at most one row per name.)
        //
        // Computed lazily: `format_args!` evaluates its arguments eagerly, so
        // building this unconditionally would scan the disposition table once
        // per scheduled pass even with tracing off.
        if crate::pass_trace::enabled() {
            let disposition = crate::pass_trace::dispositions_in(self.epoch)
                .into_iter()
                .find(|(p, _)| *p == pass)
                .map(|(_, d)| d);
            Self::trace_line(format_args!("<- {pass} disposition={disposition:?}"));
        }

        Ok(Scheduled {
            value,
            pass,
            epoch: self.epoch,
        })
    }

    /// Refuse if the tape `pass` scanned is no longer the state its plan's
    /// positional references were captured against.
    ///
    /// `WengertList`'s header states the rule: a positional index is valid
    /// only against the list state it was captured from, and "a position that
    /// must OUTLIVE the scan must be converted to the op's id at the
    /// boundary". WRGA's plan holds `TapeRef::PositionalIndex` and IS consumed
    /// after the scan — several hundred lines after, at the
    /// `effective_primal` fork. Nothing checked that the list had not moved in
    /// between; this does.
    ///
    /// A pass that recorded no scan (it did not run, or ran without a tape)
    /// vacuously passes: there is no captured state to invalidate.
    pub fn assert_tape_unchanged_since(
        self,
        pass: &'static str,
        tape: &WengertList,
    ) -> Result<(), String> {
        let recorded = SCANNED_TAPES.with(|m| {
            m.borrow()
                .iter()
                .find(|(k, _)| *k == (self.epoch, pass))
                .map(|(_, d)| *d)
        });
        let Some(before) = recorded else {
            return Ok(());
        };
        let now = TapeDigest::of(tape);
        if now == before {
            return Ok(());
        }
        Err(format!(
            "the Wengert tape '{pass}' scanned has changed before its plan was \
             consumed ({} ops then, {} now) — that plan holds POSITIONAL \
             references, which are valid only against the list state they were \
             captured from, so consuming it now would index the wrong ops. \
             Compiler scheduling defect, please report.",
            before.len, now.len
        ))
    }

    /// This compile's invocation ledger. Same view as
    /// [`PassManager::per_compile_trace`], reachable from the handle.
    fn trace(self) -> Vec<(&'static str, Option<CompilePhase>)> {
        crate::pass_trace::per_compile_view(self.epoch)
    }

    fn trace_line(args: std::fmt::Arguments<'_>) {
        if crate::pass_trace::enabled() {
            eprintln!("[pass-manager] {args}");
        }
    }
}

impl<R> Scheduled<R> {
    /// Replace the retained tape digest with the CURRENT state of `tape`.
    ///
    /// For a `TapeAccess::MutatesInPlace` pass whose scheduled body mutates
    /// the very list it scanned (CCR's `append_compressed_saves` appends the
    /// compressed-save tail inside the planning region), the digest
    /// [`PassScheduler::schedule`] captured at entry describes a state the
    /// pass itself has already superseded — an
    /// [`PassScheduler::assert_tape_unchanged_since`] at the consumption fork
    /// would then refuse every correct compile for a mutation the pass was
    /// declared to make. Re-digesting AFTER the body records the state the
    /// plan's positional references are actually valid against: "as the pass
    /// left it", which is exactly what the consumption fork must see intact.
    ///
    /// On `Scheduled` rather than the scheduler handle so a re-digest cannot
    /// exist without the scheduled invocation it re-describes.
    pub fn rescan_tape(&self, tape: &WengertList) {
        SCANNED_TAPES.with(|m| {
            let mut m = m.borrow_mut();
            let key = (self.epoch, self.pass);
            match m.iter().position(|(k, _)| *k == key) {
                Some(i) => m[i].1 = TapeDigest::of(tape),
                None => m.push((key, TapeDigest::of(tape))),
            }
        });
    }

    /// Unwrap the value WITHOUT checking the pass's declared postconditions,
    /// stating why that is sound here.
    ///
    /// Exists for exactly one shape: a pass invoked from a phase where its
    /// `applied_implies_published: Enforced` channel is structurally not yet
    /// publishable. WGGO's prepass invocation records `Applied` under
    /// `KernelPrepass`, but `wggo_overrides` is published (and cleared, and
    /// republished) per train block by the TrainBlock driver — there is no
    /// point inside the prepass where the check can hold, and downgrading the
    /// channel to `Exempt` would disarm the check at the TrainBlock boundary,
    /// the one place it IS sound. So the skip is per-INVOCATION, explicit,
    /// and carries its reason — the same shape as `pass_bus::Invariant::
    /// Exempt`, for the same reason: an escape hatch a gate cannot
    /// distinguish from an oversight is how invariants rot.
    ///
    /// The reason must be non-empty; `pass_scheduler_coverage.rs` enumerates
    /// every caller, so growing the set is a deliberate, reviewed edit.
    pub fn defer_postconditions(self, reason: &'static str) -> R {
        assert!(
            !reason.trim().is_empty(),
            "defer_postconditions requires a non-empty reason — an unexplained \
             skip is indistinguishable from a forgotten finish()"
        );
        PassScheduler::trace_line(format_args!(
            "<- {} postconditions deferred: {reason}",
            self.pass
        ));
        self.value
    }

    /// Check the pass's declared postconditions and unwrap its value.
    ///
    /// Today that is one rule, and it is the bus's own: a channel declaring
    /// `applied_implies_published: Enforced` must be occupied once its
    /// producer has recorded `Applied`. The bus REPORT already finds this at
    /// end of compile, where it is a line in a report nobody blocks on; at the
    /// pass boundary it is a refusal that still names the pass that just ran.
    pub fn finish(self, bus: &PassBus) -> Result<R, String> {
        let applied = crate::pass_trace::dispositions_in(self.epoch)
            .into_iter()
            .any(|(p, d)| {
                p == self.pass && matches!(d, crate::pass_trace::PassDisposition::Applied { .. })
            });
        if applied {
            for d in crate::pass_bus::enforced_publish_channels(self.pass) {
                if !bus.is_published(d.channel) {
                    return Err(format!(
                        "pass '{}' recorded Applied but its '{}' channel is \
                         empty, and that channel declares \
                         applied_implies_published as Enforced — the pass \
                         rewrote something and its product did not reach the \
                         consumers ({}). Compiler defect, please report.",
                        self.pass,
                        d.name,
                        d.consumers.join(", ")
                    ));
                }
            }
        }
        Ok(self.value)
    }
}

impl Drop for PassManager {
    fn drop(&mut self) {
        // The restore assumes LIFO: this manager's epoch is the thread's
        // current one. `begin()` is `pub`, so nothing else GUARANTEES the
        // discipline — check it, loudly but without panicking (a panic in
        // Drop during unwind aborts the process). A mismatch means two
        // managers on one thread were dropped out of order and the
        // survivor's tail records will mis-attribute.
        let current = crate::pass_trace::current_epoch();
        if current != self.epoch {
            eprintln!(
                "[pass-manager] BUG: non-LIFO epoch drop — dropping epoch \
                 {} while the thread is on epoch {current}; records after \
                 this point may be attributed to the wrong compile",
                self.epoch
            );
        }
        // Retire this compile's tape digests. Without it the map grows one
        // entry per (scheduled pass x compile) for the process lifetime — the
        // growth `pass_trace::TRACE` documents and tolerates for a `&'static
        // str` pair, but these carry a digest per entry and every integration
        // test binary drives many compiles.
        SCANNED_TAPES.with(|m| m.borrow_mut().retain(|((e, _), _)| *e != self.epoch));
        crate::pass_trace::restore_epoch(self.prev_epoch);
    }
}
