//! Roadmap item 2, step 1: **which passes actually ran, in what order.**
//!
//! # Why this exists
//!
//! [`crate::pass_registry`] describes eleven passes as data — implementation
//! files, driving CLI flags, pipeline stage — and its header at the time this
//! module was built was explicit about the limit of that:
//!
//! > Not a pass *manager* — it does not schedule, order, or invoke anything.
//! > `compile_train_block` still calls the passes directly.
//!
//! (Since Milestone C the direct calls are gone — the driver routes every
//! in-pipeline pass through `pass_manager::PassScheduler::schedule` — but the
//! observation gap this module closes predates that and remains its
//! foundation: scheduling decisions are made FROM this trace's epochs.)
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
//! Two separate facts, recorded by two separate calls.
//!
//! **Invocation** — `record` is called at a pass's entry point, so it answers
//! "was this pass reached". Conflating that with effect would make the signal
//! mean less, not more, so the two are never merged into one call.
//!
//! **Effect** — `record_disposition` is called at each of a pass's exits with
//! a [`PassDisposition`]: applied (and how many rewrites), declined (and why),
//! or advisory-only. It *presupposes* invocation — it asserts that `record`
//! already ran for the same pass — so it can never be mistaken for a cheaper
//! replacement, and a pass that declines stays clearly distinguishable from a
//! pass whose flag never arrived. That distinction is the whole point: "WGGO
//! did not run" and "WGGO ran and refused on a shape incompatibility" are the
//! same line in an invocation-only trace, and only one of them is a bug.
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
//! The trace is two compiler-side vectors of `Copy` data. Nothing here is
//! emitted into the compiled program, so enabling it cannot move a loss curve.
//! The gate in `crates/nsl-cli/tests/pass_trace_gate.rs` asserts exactly that
//! by comparing loss streams with the trace on and off.
//!
//! Both `record` and `record_disposition` run UNCONDITIONALLY — neither is
//! gated on [`enabled`]. Gating would mean the assertions inside them only
//! ever execute under `NSL_PASS_TRACE=1`, and the purity gate would then be
//! comparing two different code paths instead of proving one path pure. Every
//! disposition payload is `usize` / `&'static str` / a `Copy` enum precisely
//! so that "always on" costs an integer store and never an allocation.
//!
//! # Toward a pass manager
//!
//! This is step 1 because scheduling requires observation: you cannot move
//! ordering into the registry until you can see the order the driver actually
//! produces, and cannot prove the move was behaviour-preserving without a
//! before/after trace to compare. The declared [`crate::pass_registry::
//! PipelineStage`] partial order becomes checkable the moment a trace exists.

use crate::pass_registry::CompilePhase;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Passes observed this PROCESS, in first-invocation order per compile scope,
/// each tagged with the driver phase that was active when it was reached
/// (`None` = the pass ran outside every declared phase — see [`enter_phase`])
/// and the compile epoch it was reached in (see [`begin_epoch`]).
///
/// Process-facing readers ([`observed`], [`observed_phases`], the report)
/// dedupe by name at first occurrence, which reproduces the pre-epoch
/// semantics exactly: before epochs existed, `record` was idempotent
/// process-wide, so only the first entry per name could exist at all.
///
/// Growth is O(passes x compiles) per process where it used to be bounded
/// by pass count — irrelevant at CLI scales (one entry per pass per module
/// compile), noted for any future long-lived recompiling process, which
/// would want an epoch-retiring compaction here.
static TRACE: Mutex<Vec<(&'static str, Option<CompilePhase>, u64)>> = Mutex::new(Vec::new());

/// Source of unique compile epochs. Global, not thread-local: two threads
/// each compiling (parallel libtest binaries drive full compiles) must not
/// mint the same epoch, or their per-compile views would merge — the exact
/// cross-attribution the epoch exists to prevent.
static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);

// The compile epoch currently attributing on this THREAD. 0 = no compile
// scope installed (unit tests driving passes directly; the out-of-band CLI
// subcommand drivers). Thread-local for the same reason CURRENT_PHASE is: a
// compile never spans threads, and a process-global would let one thread's
// compile claim another's pass invocations.
thread_local! {
    static CURRENT_EPOCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Begin a per-compile attribution scope on this thread: mint a fresh epoch,
/// install it, and return `(new, previous)`. The caller is responsible for
/// [`restore_epoch`]`(previous)` when its compile ends — in production that
/// caller is exactly one place, `PassManager::begin` (RAII-restored on drop),
/// so a nested compile (should one ever exist) re-attributes to its outer
/// scope instead of leaking its own epoch over the outer compile's tail.
pub(crate) fn begin_epoch() -> (u64, u64) {
    let e = NEXT_EPOCH.fetch_add(1, Ordering::Relaxed);
    let prev = CURRENT_EPOCH.with(|c| c.replace(e));
    (e, prev)
}

/// Restore the epoch that was active before [`begin_epoch`].
pub(crate) fn restore_epoch(prev: u64) {
    CURRENT_EPOCH.with(|c| c.set(prev));
}

/// The epoch currently attributing on this thread (0 = none installed).
/// The manager's `Drop` uses it to detect a non-LIFO drop before restoring.
pub(crate) fn current_epoch() -> u64 {
    CURRENT_EPOCH.with(|c| c.get())
}

/// The entries recorded under `epoch`, in first-invocation order — the
/// per-compile view [`crate::pass_manager::PassManager`] decides from. This
/// is the sound counterpart of [`observed`]: the process-global view is for
/// REPORTING (where over-inclusion is a wording problem), the per-epoch view
/// is for ENFORCEMENT (where over-inclusion is a false refusal — the #466
/// lesson: per-compile ordering arguments over process-scoped state are
/// defeated by multi-module builds).
pub(crate) fn per_compile_view(epoch: u64) -> Vec<(&'static str, Option<CompilePhase>)> {
    TRACE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|(_, _, e)| *e == epoch)
        .map(|(p, ph, _)| (*p, *ph))
        .collect()
}

// The phase currently executing, per THREAD.
// 
// A process-global would be wrong: `enter_phase` saves and restores the
// previous value, and two threads compiling concurrently (15 integration-test
// binaries drive full compiles, and libtest runs them in parallel) can
// interleave so that one guard's restore leaks the other's phase for the rest
// of the process. This repo has already shipped that exact bug class once.
// A compile never spans threads, so thread-local is also the accurate scope.
thread_local! {
    static CURRENT_PHASE: std::cell::Cell<Option<CompilePhase>> =
        const { std::cell::Cell::new(None) };
}

/// RAII guard restoring the previously-active phase on drop.
pub struct PhaseGuard(Option<CompilePhase>);

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        CURRENT_PHASE.with(|c| c.set(self.0));
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
    PhaseGuard(CURRENT_PHASE.with(|c| c.replace(Some(phase))))
}

/// The driver phase executing on this thread right now, if any.
///
/// [`observed_phases`] answers "which phase did a pass run in" AFTER the fact,
/// from the trace. The scheduler needs the same fact BEFORE it invokes a pass,
/// so it can refuse an invocation from a phase the registry does not declare
/// instead of reporting the mismatch once the pass has already run.
pub fn current_phase() -> Option<CompilePhase> {
    CURRENT_PHASE.with(|c| c.get())
}

/// The phase each observed pass ran in — first occurrence per name, which is
/// exactly what the pre-epoch process-wide idempotence produced.
pub fn observed_phases() -> Vec<(&'static str, Option<CompilePhase>)> {
    let t = TRACE.lock().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<(&'static str, Option<CompilePhase>)> = Vec::new();
    for (p, ph, _) in t.iter() {
        if !out.iter().any(|(q, _)| q == p) {
            out.push((*p, *ph));
        }
    }
    out
}

/// Passes whose observed phase differs from the registry's declaration, as
/// `(pass, declared, observed)`. A pass declared [`CompilePhase::OutOfBand`]
/// is expected to be unattributed, since those drivers are CLI subcommands
/// outside the compile pipeline.
pub fn phase_mismatches(
) -> Vec<(&'static str, &'static [CompilePhase], Option<CompilePhase>)> {
    observed_phases()
        .into_iter()
        .filter_map(|(name, got)| {
            let want = crate::pass_registry::pass(name)?.phases;
            // Empty declaration == "expected unattributed" (an OutOfBand
            // subcommand). Otherwise the observed phase must be one of them.
            let ok = match got {
                None => want.is_empty(),
                Some(g) => want.contains(&g),
            };
            (!ok).then_some((name, want, got))
        })
        .collect()
}

/// What a pass DID once it was reached.
///
/// Deliberately three variants, and deliberately NOT four. There is no
/// `Failed`: no registered pass returns an error. Every refusal in the tree is
/// either a `None`/empty-plan return (CCR's four exits, WGGO's §2.4 shape
/// refusal) or a diagnostic the *driver* turns into a `CodegenError`
/// (`stmt.rs` ~8842). A `Failed` variant would therefore have zero producers —
/// an enum implying a distinction the compiler does not actually make, which
/// is exactly the decorative-metadata defect `pass_registry.rs` deleted its
/// `status` field to avoid. Do not add it speculatively; add it in the same
/// commit as its first real producer, or not at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassDisposition {
    /// The pass transformed something. `rewrites` is the pass's OWN unit of
    /// work and its meaning is per-pass (layer decisions for WGGO, block
    /// segments for CCR, backward phases for FASE, ...); each call site
    /// documents which quantity it is reporting. A pass that would report
    /// `rewrites: 0` should report a `Declined` with a reason instead — zero
    /// is exactly the number a hard-coded stub produces, so it must never be
    /// the way "nothing happened" is spelled.
    Applied { rewrites: usize },
    /// The pass ran, decided not to transform anything, and said why.
    Declined { reason: DeclineReason },
    /// The pass produced a report or a recommendation that no codegen stage
    /// consumes. Distinct from `Applied { rewrites: 0 }` because it is not a
    /// disappointment: it is what the pass is *for* on that path.
    AdvisoryOnly,
}

/// Why a pass declined.
///
/// Structural categories carrying a `&'static str` detail, not a fully-typed
/// enum. That is a considered limit, not laziness: every typed refusal enum in
/// the tree is PER-SITE rather than per-pass (`wggo_prune::PruneRefusal`,
/// `wggo_overrides::OverrideRejectReason`, `pca_per_doc::RejectReason`,
/// `cpdt_expert_prune::ExpertPruneRefusal`), and the genuine pass-level
/// declines — CCR's four — exist today only as `eprintln!` strings, one of
/// which interpolates runtime op indices. Categorising them buys the thing a
/// gate can assert on; spelling out every detail as a variant would buy a
/// second enum to keep in sync with the strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclineReason {
    /// The pass's own mode selector is off (`--csha off`, `WggoMode::Off`,
    /// `FaseMode::Passthrough`). Nothing is wrong; the user asked for this.
    ModeOff,
    /// The pass is on, but the program contains nothing it can work on.
    NoCandidates(&'static str),
    /// The pass is on and had candidates, but a precondition it must not
    /// weaken does not hold, so it refused rather than transforming anyway.
    PreconditionViolated(&'static str),
    /// A capability the pass needs was switched off underneath it (an
    /// ablation flag, a disabled decorator), so its stages became no-ops.
    FeatureDisabled(&'static str),
    /// A memory/parameter budget cannot be met even at the pass's floor.
    BudgetInfeasible,
}

/// Effects observed this compile, keyed by pass, in first-DISPOSITION order.
///
/// A `Vec` rather than a map so the render order is deterministic without
/// pulling in an ordered map for eleven entries.
///
/// Stamped with the compile epoch for the same reason [`TRACE`] is: the
/// process-global view is fine for REPORTING, but the scheduler turns
/// "this pass reported Applied" into a refusal, and over-inclusion in a
/// refusal is a false refusal. In a multi-module build, module A's `Applied`
/// against module B's empty channel would refuse a correct compile — the #466
/// finding, one channel over. [`dispositions`] keeps the flat process view;
/// [`dispositions_in`] is the enforceable one.
///
/// Growth is now O(passes x compiles) where it was O(passes) — the same trade
/// [`TRACE`] documents, and deliberately NOT retired per epoch: the
/// process-facing readers ([`dispositions`], [`report`]) are defined over the
/// whole history, and the report runs after that compile's `PassManager` has
/// already dropped, so retiring would erase the rows it exists to print.
/// Irrelevant at CLI scales (one row per pass per module compile); a
/// long-lived recompiling process would want the epoch-retiring compaction
/// already noted on `TRACE`.
static DISPOSITION: Mutex<Vec<(&'static str, PassDisposition, u64)>> = Mutex::new(Vec::new());

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
    let phase = CURRENT_PHASE.with(|c| c.get());
    let epoch = CURRENT_EPOCH.with(|c| c.get());
    let mut t = TRACE.lock().unwrap_or_else(|e| e.into_inner());
    // Idempotent PER COMPILE EPOCH, not per process: a pass that runs again
    // in a later compile of the same process (multi-module builds construct
    // one Compiler per module) must appear in that compile's view, or the
    // per-compile ordering authority would judge an incomplete sequence.
    // Process-facing readers dedupe by name, so the report is unchanged.
    if !t.iter().any(|(p, _, e)| *p == pass && *e == epoch) {
        t.push((pass, phase, epoch));
    }
}

/// Record what `pass` DID. Call at every exit of the pass, including the ones
/// that decline.
///
/// This is a SECOND function, not a wider `record`, and it asserts that
/// `record(pass)` already happened. Disposition therefore structurally
/// presupposes entry: there is no way to spell "WGGO applied 12 rewrites"
/// without also having said "WGGO was reached", so a future refactor cannot
/// quietly swap the cheap call for the informative one and lose the
/// invocation half of the trace. It also means a bogus name is caught by
/// `record`'s registry assert rather than needing a second copy of it here.
///
/// NOT debug_assert: CI ships release builds, where that compiles out — the
/// same reasoning written out at [`record`], and the reason `assert!` is used
/// for every production invariant in this crate.
///
/// # Merge rule: LAST WINS
///
/// A pass may be entered several times in one compile and may reach different
/// exits each time. The worked example is CCR under `--checkpoint-stride
/// auto`: `stmt.rs` calls `ccr::plan` once per candidate stride, so a single
/// build produces a whole sequence of dispositions — some declining, the
/// winner applying — and only the FINAL call corresponds to the plan the build
/// actually used. Keeping the last one is therefore the answer to "what did
/// this pass do to THIS build"; keeping the first would report a discarded
/// search step, and keeping all of them would turn an eleven-line report into
/// a log.
///
/// The same rule is what lets a driver refine a pass's own self-report with
/// something the pass could not know — see `stmt.rs`, where WGGO's
/// layer-decision count is superseded by `wggo_prune`'s tape-rewrite count.
pub fn record_disposition(pass: &'static str, d: PassDisposition) {
    let entered = {
        // TRACE holds (name, phase, epoch) triples — match on the name only;
        // neither the phase a pass was entered in nor the compile it was
        // entered by has any bearing on whether it may report a disposition.
        let t = TRACE.lock().unwrap_or_else(|e| e.into_inner());
        t.iter().any(|(p, _, _)| *p == pass)
    };
    assert!(
        entered,
        "pass_trace::record_disposition(\"{pass}\") without a prior \
         record(\"{pass}\") — a disposition presupposes entry, so this is \
         either a missing record at the pass's entry point or a disposition \
         attributed to the wrong pass"
    );
    let epoch = current_epoch();
    let mut t = DISPOSITION.lock().unwrap_or_else(|e| e.into_inner());
    // Keyed by (pass, epoch), not by pass: the pre-epoch code collapsed a
    // second compile's disposition onto the first's slot, which is harmless
    // for the report (last writer wins, and it renders one line per pass) but
    // would make the per-compile view answer with another compile's outcome.
    match t.iter_mut().find(|(p, _, e)| *p == pass && *e == epoch) {
        Some(slot) => slot.1 = d,
        None => t.push((pass, d, epoch)),
    }
}

/// The passes reached this process, in first-invocation order — deduped by
/// name, matching the pre-epoch semantics (see [`TRACE`]).
pub fn observed() -> Vec<&'static str> {
    observed_phases().into_iter().map(|(p, _)| p).collect()
}

/// What each pass that reported a disposition did, in first-disposition order.
///
/// Process-scoped, deduped by pass name so the report keeps its one-line-per-
/// pass shape across a multi-module build. For enforcement use
/// [`dispositions_in`].
pub fn dispositions() -> Vec<(&'static str, PassDisposition)> {
    let t = DISPOSITION.lock().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<(&'static str, PassDisposition)> = Vec::new();
    for (p, d, _) in t.iter() {
        match out.iter_mut().find(|(q, _)| q == p) {
            Some(slot) => slot.1 = *d,
            None => out.push((*p, *d)),
        }
    }
    out
}

/// The dispositions recorded under `epoch` — the sound counterpart of
/// [`dispositions`], for the same reason [`per_compile_view`] is the sound
/// counterpart of [`observed`].
pub(crate) fn dispositions_in(epoch: u64) -> Vec<(&'static str, PassDisposition)> {
    DISPOSITION
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|(_, _, e)| *e == epoch)
        .map(|(p, d, _)| (*p, *d))
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
    CURRENT_PHASE.with(|c| c.set(None));
    // Dispositions too, or `record_disposition`'s entry assert fires on the
    // next compile in the same process: the disposition would still be there
    // while the entry it presupposes had been wiped.
    DISPOSITION.lock().unwrap_or_else(|e| e.into_inner()).clear();
    // And the bus (step 4). Its `SILENT DEFAULT` finding asks whether the
    // producing pass ran, which it answers from TRACE — so bus traffic that
    // outlived a trace reset would be judged against the wrong compile. There
    // is no case for clearing one and not the other, so this is not left to a
    // caller to remember.
    crate::pass_bus::reset();
    // And the activation request log (Milestone A): requested decorators are
    // compile-scoped exactly like dispositions, and the reconciler joins the
    // two — clearing one without the other would reconcile this compile's
    // requests against the previous compile's answers.
    crate::activation::reset_requests();
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

/// One disposition, as it appears in the report — without the pass name or
/// the `[pass-trace]` prefix, so that every category's wording lives in one
/// `match` and two variants cannot quietly come to render the same text.
/// `pass_trace_unit.rs` pins that distinctness directly.
fn render_disposition(d: PassDisposition) -> String {
    match d {
        PassDisposition::Applied { rewrites } => format!("applied, {rewrites} rewrite(s)"),
        PassDisposition::Declined { reason } => match reason {
            DeclineReason::ModeOff => "declined, mode off".to_string(),
            DeclineReason::NoCandidates(d) => format!("declined, no candidates - {d}"),
            DeclineReason::PreconditionViolated(d) => {
                format!("declined, precondition violated - {d}")
            }
            DeclineReason::FeatureDisabled(d) => format!("declined, feature disabled - {d}"),
            DeclineReason::BudgetInfeasible => "declined, budget infeasible".to_string(),
        },
        PassDisposition::AdvisoryOnly => "advisory only".to_string(),
    }
}

/// Public form of [`render_disposition`] for the activation reconciler's
/// report: every outcome category keeps a single spelling in the tree, so
/// `[activation]` lines reuse this match instead of growing a paraphrase the
/// distinctness gate cannot see.
pub fn render_disposition_line(d: PassDisposition) -> String {
    render_disposition(d)
}

/// Render the observed sequence with each pass's declared stage and the driver
/// phase it actually ran in, then one line per pass that reported an effect:
///
/// ```text
/// [pass-trace] 3 pass(es) ran: WGGO(OnWengert)@KernelPrepass -> FASE(PreExtraction)@TrainBlock -> CCR(OnAdjoint)@TrainBlock
/// [pass-trace] WGGO: applied, 12 rewrite(s)
/// [pass-trace] FASE: applied, 4 rewrite(s)
/// [pass-trace] CCR: declined, no candidates - the tape has no blocks.N-style parameters
/// ```
///
/// Registered passes that did NOT run are listed too — the absence is the
/// interesting half, and a report that only showed what happened would make
/// "nothing ran" indistinguishable from "the trace is broken".
///
/// Dispositions go on their OWN lines rather than being folded into the
/// sequence: the gates in `crates/nsl-cli/tests/pass_trace_gate.rs` and
/// `crates/nsl-codegen/tests/pass_trace_unit.rs` parse the `ran:` and
/// `did not run:` lines positionally, by splitting on `ran:` / `->` / `(` /
/// `@` and on `did not run:` / `,`, so anything added INSIDE them breaks a
/// parser while a new line with a distinct shape does not.
///
/// They also reuse the `[pass-trace]` prefix rather than introducing a second
/// bracketed token — a new banner prefix would need an `EXEC_MARKERS` entry
/// and a `tokens::` constant in `crates/nsl-cli/src/exec_markers.rs`, or
/// `feature_composition_gate.rs` fails the moment a test asserts on it.
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
    // Effects, in INVOCATION order rather than disposition order, so the
    // effect block reads down in the same sequence as the `ran:` line above
    // it. A pass that was reached but reported no disposition is simply
    // absent here; that gap is itself a finding (an exit with no
    // `record_disposition`), and hiding it behind a placeholder would make it
    // unfindable.
    let d = dispositions();
    for p in &seen {
        if let Some((_, disp)) = d.iter().find(|(name, _)| name == p) {
            s.push_str(&format!("[pass-trace] {p}: {}\n", render_disposition(*disp)));
        }
    }
    // A detector nobody calls is decoration. Surfacing mismatches in the
    // report itself is what makes a wrong declaration visible from one run,
    // rather than only from a test that happens to cover that configuration.
    for (name, want, got) in phase_mismatches() {
        s.push_str(&format!(
            "[pass-trace] PHASE MISMATCH: {name} ran in {} but the registry \
             declares {want:?}\n",
            got.map(|c| format!("{c:?}")).unwrap_or_else(|| "no phase".into()),
        ));
    }
    if let Some((a, b)) = stage_order_violation() {
        // An inversion explained by a declared dependency edge is not merely
        // tolerable — WGGO(OnWengert) before FASE(PreExtraction) looks
        // backwards by stage rank and is exactly the order the
        // wggo_overrides edge describes. Two conditions keep this line
        // honest. The edge must have CARRIED something (publishes > 0): a
        // WGGO that ran and DECLINED publishes nothing, FASE reads the
        // channel empty and falls back, and attributing the inversion to a
        // dependency that carried nothing would be a false explanation. And
        // the wording follows the edge's own OrderClaim: only an
        // InvocationOrdered edge REQUIRES the order; a ValueOrderedOnly edge
        // merely makes the observed order the one its value flowed in.
        let edge = crate::pass_bus::CHANNELS.iter().find_map(|d| {
            if d.producer != a {
                return None;
            }
            let c = d.consumed_by_passes.iter().find(|c| c.pass == b)?;
            (crate::pass_bus::traffic(d.channel).publishes > 0).then_some((d, c))
        });
        match edge {
            Some((d, c)) => {
                let claim = match c.order {
                    crate::pass_bus::OrderClaim::InvocationOrdered => "required by",
                    crate::pass_bus::OrderClaim::ValueOrderedOnly(_) => "consistent with",
                };
                s.push_str(&format!(
                    "[pass-trace] STAGE ORDER: {a} ran before {b}, inverting \
                     their declared PipelineStage — {claim} the declared \
                     dependency: {b} consumes {}, which {a} produces\n",
                    d.name
                ));
            }
            None => s.push_str(&format!(
                "[pass-trace] STAGE ORDER: {a} ran before {b}, inverting \
                 their declared PipelineStage (see each pass's CompilePhase \
                 above — a cross-phase inversion is expected, not a defect)\n"
            )),
        }
    }
    // The invocation-order half of the declared dependency edges. Printed
    // here rather than only exposed as a function, for the reason
    // `phase_mismatches` is: a detector with no production caller is why a
    // false claim survives.
    for (producer, consumer, ch) in crate::pass_bus::dependency_order_violations() {
        s.push_str(&format!(
            "[pass-trace] DEPENDENCY ORDER: {consumer} was invoked before \
             {producer}, yet {consumer} consumes {}, which {producer} \
             produces — whatever {producer} published cannot have reached \
             this compile's {consumer} planning\n",
            ch.descriptor().name
        ));
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
/// stop. The intra-stage order that IS declared lives on the bus, where the
/// data dependencies are: `pass_bus::ChannelDescriptor::consumed_by_passes`
/// names the pass-to-pass edges, and
/// [`crate::pass_bus::dependency_order_violations`] checks the
/// invocation-ordered subset of them against this same observed sequence.
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
