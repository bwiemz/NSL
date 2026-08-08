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
//! # Scheduling: one pass so far
//!
//! The step after judgment is INVERTING CONTROL — the manager invoking passes
//! from the declarations rather than grading the driver afterwards. That is
//! [`PassScheduler`], and it is deliberately wired to exactly ONE pass today
//! (WRGA). Converting the rest is mechanical once the substrate is proven;
//! converting them all at once would mean landing a scheduler whose refusals
//! nobody had watched fire.
//!
//! WRGA is the first because it is the pass whose declarations are richest and
//! whose invocation is narrowest: ONE call site
//! (`stmt::invoke_wrga_if_enabled`), an INCOMING `InvocationOrdered` edge (it
//! consumes `wggo_overrides`, so WGGO must precede it), an OUTGOING channel
//! with `applied_implies_published: Enforced` (`wrga_plan`), and a tape
//! declaration — `MutatesFork`, holding `TapeRef::PositionalIndex` — whose
//! staleness rule is real and was checked nowhere.
//!
//! What the scheduler enforces, all of it read from the declarations rather
//! than hard-coded per pass:
//!
//! | when | check | declared by |
//! |---|---|---|
//! | before | the pass is registered | `pass_registry::PASSES` |
//! | before | this driver phase is one the pass declares | `PassDescriptor::phases` |
//! | before | every `InvocationOrdered` predecessor already ran THIS compile | `ChannelDescriptor::consumed_by_passes` |
//! | after | `Applied` ⇒ its `Enforced` channels are published | `applied_implies_published` |
//! | later | the tape a plan's positional refs were captured against is unchanged when the plan is consumed | `TapeAccess` / `TapeRef::PositionalIndex` |
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
/// Records everything a positional reference's validity depends on: the op
/// sequence and each op's identity. A 64-bit digest can in principle collide;
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
            // `PrimalOp` is not `Hash` (it carries an f64 in `Constant`).
            // Its `Debug` is total and distinguishes every variant AND
            // payload, which is what the staleness question needs — a
            // discriminant-only digest would miss a rewritten `Passthrough`
            // name, and that is a real rewrite (`fuse_swiglu_gate_backward`
            // does exactly it).
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
            // So the phase check is LIVE exactly where a phase is established
            // — which includes the production `nsl build` train-block path —
            // and silent elsewhere. The trace line below says which it was, so
            // "unscoped" is visible rather than indistinguishable from a pass.
            None => {}
        }

        // PRE 2: every InvocationOrdered predecessor must already have run in
        // THIS compile. Process-scoped evidence could not support this — see
        // the module header — which is why it lives on the manager.
        let seen: Vec<&'static str> = self
            .trace()
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        for (producer, channel) in crate::pass_bus::required_predecessors(pass) {
            // A predecessor that did not run AT ALL is not an ordering
            // violation: the feature is simply off, and the consumer reads the
            // channel empty (which `empty_means` documents per channel). Only
            // an INVERSION is a defect, and at this point "producer ran" is
            // decidable while "producer will run later" is not — so the check
            // is: if the producer is going to run in this compile it must have
            // run by now. That is exactly what the post-hoc
            // `dependency_order_violations` catches after the fact; scheduling
            // upgrades it to a pre-check for the passes routed through here.
            if crate::pass_bus::traffic(channel).publishes > 0 && !seen.contains(&producer) {
                return Err(format!(
                    "pass '{pass}' was scheduled before '{producer}', its \
                     declared InvocationOrdered producer on the '{}' channel, \
                     yet that channel has already been published — the \
                     consumer is reading a value whose producing pass this \
                     compile has not run. Compiler scheduling defect, please \
                     report. NSL_PASS_TRACE=1 prints the full trace.",
                    channel.descriptor().name
                ));
            }
        }

        if let Some(list) = tape {
            let digest = TapeDigest::of(list);
            SCANNED_TAPES.with(|m| {
                let mut m = m.borrow_mut();
                let key = (self.epoch, pass);
                match m.iter_mut().find(|(k, _)| *k == key) {
                    // A pass legitimately re-invoked in one compile (WGGO's
                    // in-place replan is the in-tree example) rescans a
                    // possibly-different tape; the LATEST scan is the one its
                    // live plan refers to.
                    Some(slot) => slot.1 = digest,
                    None => m.push((key, digest)),
                }
            });
        }

        Self::trace_line(format_args!(
            "-> {pass} phase={} epoch={} tape={}",
            match phase {
                Some(p) => format!("{p:?}"),
                None => "unscoped(phase check skipped)".to_string(),
            },
            self.epoch,
            tape.map(|t| t.ops.len().to_string())
                .unwrap_or_else(|| "none".into()),
        ));

        let value = body();

        Self::trace_line(format_args!(
            "<- {pass} disposition={:?}",
            crate::pass_trace::dispositions()
                .into_iter()
                .rev()
                .find(|(p, _)| *p == pass)
                .map(|(_, d)| d)
        ));

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
