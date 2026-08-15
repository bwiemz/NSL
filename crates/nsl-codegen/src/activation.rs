//! Milestone A: activation contracts — "no advertised feature can be silently
//! inert".
//!
//! # The defect class this closes
//!
//! NSL's request surface (CLI optimization flags, source decorators) is larger
//! than the surface the compiler honours, and the difference has historically
//! been silent: `@cpdt(mode = off)` on a train block compiled to a byte-identical
//! binary on every CLI entry point, `@fase(...)`'s validated config was dropped
//! at its only call site, and `@tie_weights` is advertised in the wiki while
//! appearing in the crates only inside two comments. A user cannot tell a
//! honoured request from an ignored one by exit code.
//!
//! Three of the four pieces needed to close this already existed:
//! [`crate::pass_registry`] holds per-pass `cli_flags` + `decorator_triggers`
//! as gated data; [`crate::pass_trace`] holds the outcome vocabulary
//! ([`PassDisposition`]: applied / declined / advisory) and the execution
//! witness (`NSL_PASS_TRACE=1`). This module is the missing JOIN: at the end
//! of a compile, for every surface the user actually requested, it asks "did
//! the owner record anything?" — and a silent owner becomes a hard error in
//! the CLI instead of a silently-identical binary.
//!
//! # What a contract is
//!
//! One [`Contract`] per advertised surface. The load-bearing field is
//! [`Contract::witness`]: HOW activation is observable. A surface whose
//! activation cannot be observed is not admissible here — that rule is what
//! keeps this table from decaying into decorative metadata (the defect the
//! item-20/21 registries each had to delete a field to fix).
//!
//! Witnesses come in two enforcement tiers:
//!
//! - [`Witness::Disposition`]: the owner is a registered pass that records a
//!   [`PassDisposition`] when reached. Reconciled IN-PROCESS by
//!   [`reconcile`]; a requested-but-silent owner is [`OutcomeState::Unsatisfied`]
//!   and the CLI turns that into an error (`--allow-inert-requests` demotes
//!   it to a warning).
//! - [`Witness::Marker`] / [`Witness::Config`] / [`Witness::Report`]: the
//!   surface is consumed at a named site or emits a named stderr marker.
//!   These cannot be reconciled in-process (a process cannot cheaply observe
//!   its own stderr), so they are enforced by drift gates in
//!   `tests/activation_contract_gate.rs`: the named marker/site must exist in
//!   the tree, and the flag itself must exist on the declared subcommands.
//!
//! Completeness is gated from the OTHER side (item-20 lesson: a registry must
//! gate its own incompleteness): the gate test parses `args.rs` and fails on
//! any optimization-surface flag with neither a contract nor an entry in the
//! explicit [`UNCONTRACTED_FLAGS`] allowlist, so "deliberately not covered"
//! stays a written-down decision rather than a silent omission.

use std::sync::Mutex;

use crate::pass_registry::{self, Subcommand};
use crate::pass_trace::{self, PassDisposition};

/// Decorator names present in the ENTRY module of the compile that just ran.
///
/// Entry module only, deliberately: decorators inside imported/stdlib modules
/// (`@fused_lm_ce` in `stdlib/nsl/nn/losses.nsl`, `@param_role` in
/// `muon.nsl`) are the library author's requests, exercised by the library's
/// own gates — reconciling them against every downstream user compile would
/// hold users answerable for requests they never wrote.
///
/// Same process-global pattern as [`pass_trace`]'s logs, cleared by the same
/// [`pass_trace::reset`] (one reset discipline — see the comment there on why
/// partial clears are never offered).
static REQUESTED_DECORATORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Record the entry module's decorator names for reconciliation. Called at
/// the top of each codegen entry point that receives the entry AST, so every
/// CLI path (normal, shared-lib, standalone, zk) feeds the same log without
/// threading ASTs into the drivers.
pub fn note_entry_module_decorators(module: &nsl_ast::Module, interner: &nsl_lexer::Interner) {
    let mut log = REQUESTED_DECORATORS.lock().unwrap_or_else(|e| e.into_inner());
    for u in nsl_ast::decorator_walk::collect_decorators(module) {
        if let Some(first) = u.deco.name.first() {
            let name = interner.resolve(first.0).unwrap_or("").to_string();
            if !name.is_empty() && !log.contains(&name) {
                log.push(name);
            }
        }
    }
}

/// The decorator names noted by the last compile, in first-seen order.
pub fn requested_decorators() -> Vec<String> {
    REQUESTED_DECORATORS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Clear the request log. Called from [`pass_trace::reset`] ONLY — tests and
/// multi-compile processes reset all compile-scoped globals through one door.
pub(crate) fn reset_requests() {
    REQUESTED_DECORATORS.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// A user-visible request surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// A CLI long flag, kebab-case, without the leading `--`, exactly as a
    /// user types it (matches [`pass_registry::CliFlag::flag`]).
    Flag(&'static str),
    /// A source decorator name, without the leading `@`.
    Decorator(&'static str),
}

impl Surface {
    pub fn render(&self) -> String {
        match self {
            Surface::Flag(f) => format!("--{f}"),
            Surface::Decorator(d) => format!("@{d}"),
        }
    }
}

/// HOW a surface's activation is observable. See the module doc for the two
/// enforcement tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Witness {
    /// Owner is a registered pass; activation (or a typed decline) appears in
    /// [`pass_trace::dispositions`]. Reconciled in-process.
    Disposition(&'static str),
    /// Activation emits this exact stderr marker (compile-time stderr for
    /// build-phase features). Enforced by gates, not in-process.
    Marker(&'static str),
    /// The value is consumed unconditionally at the named site — inertness is
    /// impossible by construction (a threshold, a device selector, a seed).
    /// The site is a `file::fn` breadcrumb for the gate and the reader.
    Config(&'static str),
    /// The surface exists to produce a report/artifact; a missing artifact is
    /// already loud. The site names the emitter.
    Report(&'static str),
}

/// One advertised surface and its activation contract.
#[derive(Debug, Clone, Copy)]
pub struct Contract {
    pub surface: Surface,
    /// Subcommands where the surface is accepted. For decorators this is the
    /// set of subcommands whose pipeline can honour the decorator.
    pub on: &'static [Subcommand],
    /// The component answerable for the request: a registered pass name, or a
    /// driver/frontend component name for non-pass surfaces.
    pub owner: &'static str,
    pub witness: Witness,
}

use Subcommand::{Build, Run};
const BR: &[Subcommand] = &[Build, Run];

const fn deco(
    surface: &'static str,
    on: &'static [Subcommand],
    owner: &'static str,
    witness: Witness,
) -> Contract {
    Contract { surface: Surface::Decorator(surface), on, owner, witness }
}

/// Contracts for surfaces that are NOT derivable from [`pass_registry`]:
/// driver-owned flags and decorators owned by non-pass components.
///
/// Grown family-by-family; the completeness gate forces every optimization
/// flag to appear either here, in a pass's `cli_flags`, or in
/// [`UNCONTRACTED_FLAGS`] with a reason.
pub static MANUAL_CONTRACTS: &[Contract] = &[
    // ------------------------------------------------------------------
    // Decorators owned by non-pass components (frontend / kernel driver).
    // Witness sites name the consumer the gate pins.
    // ------------------------------------------------------------------
    deco("fase", BR, "FASE", Witness::Disposition("FASE")),
    deco("autotune", BR, "autotune", Witness::Marker("[autotune]")),
    deco("flash_attention", BR, "kernel-driver", Witness::Marker("[flash-attention]")),
];

/// Optimization-surface flags DELIBERATELY without an activation contract,
/// with the reason written down. The completeness gate fails on any
/// optimization flag missing from both this list and the contract set, so
/// exclusion is a diff-visible decision.
pub static UNCONTRACTED_FLAGS: &[(&str, &str)] = &[
    // (flag, reason) — populated family-by-family alongside MANUAL_CONTRACTS.
];

/// The full contract set: every pass-registry surface (flags AND decorator
/// triggers become [`Witness::Disposition`] contracts owned by that pass),
/// then the manual table.
///
/// Derived at call time rather than stored, so the pass-registry half can
/// never drift from `PASSES` — the same reason `pass_registry` derives its
/// wiki checks from `PASSES` instead of a second list.
pub fn contracts() -> Vec<Contract> {
    let mut out = Vec::new();
    for p in pass_registry::PASSES {
        for f in p.cli_flags {
            out.push(Contract {
                surface: Surface::Flag(f.flag),
                on: f.on,
                owner: p.name,
                witness: Witness::Disposition(p.name),
            });
        }
        for d in p.decorator_triggers {
            out.push(Contract {
                surface: Surface::Decorator(d),
                // A decorator can only be honoured where the pass can run;
                // passes run on the compile pipeline (build/run), and some
                // analysis modes on check. BR is the conservative floor —
                // check-side honouring is still reported, never enforced
                // (see reconcile()).
                on: BR,
                owner: p.name,
                witness: Witness::Disposition(p.name),
            });
        }
    }
    // Manual entries may refine a generated one (e.g. @fase is generated from
    // FASE's decorator_triggers already); dedupe keeps the FIRST occurrence,
    // so generated (registry-derived) entries win and a manual duplicate is
    // rejected by the gate instead of silently shadowing.
    for c in MANUAL_CONTRACTS {
        if !out.iter().any(|e| e.surface == c.surface) {
            out.push(*c);
        }
    }
    out
}

/// Look up the contract for a requested surface, if any.
pub fn contract_for(surface: &RequestedSurface) -> Option<Contract> {
    contracts().into_iter().find(|c| match (&c.surface, surface) {
        (Surface::Flag(f), RequestedSurface::Flag(name)) => f == name,
        (Surface::Decorator(d), RequestedSurface::Decorator(name)) => d == name,
        _ => false,
    })
}

/// A surface the user actually requested on THIS invocation: a flag that was
/// explicitly present on the command line, or a decorator present in the
/// program source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestedSurface {
    Flag(String),
    Decorator(String),
}

impl RequestedSurface {
    pub fn render(&self) -> String {
        match self {
            RequestedSurface::Flag(f) => format!("--{f}"),
            RequestedSurface::Decorator(d) => format!("@{d}"),
        }
    }
}

/// What became of one requested surface, after the compile.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub surface: RequestedSurface,
    pub owner: Option<&'static str>,
    pub state: OutcomeState,
}

#[derive(Debug, Clone)]
pub enum OutcomeState {
    /// The owner recorded a disposition — the request was answered, whichever
    /// of the three answers it got. `Declined` IS a satisfied contract: the
    /// user can see why.
    Disposed(PassDisposition),
    /// The witness is structural (Marker/Config/Report) — activation is
    /// enforced by gates, not reconciled in-process. Never an error here.
    ByConstruction(Witness),
    /// The surface's contract says its owner records a disposition when
    /// reached, the surface was requested, and the owner recorded NOTHING.
    /// This is the silent-inertness condition. The CLI turns it into a hard
    /// error (or a warning under `--allow-inert-requests`).
    Unsatisfied { owner: &'static str },
    /// No contract covers this surface. Plumbing flags land here at runtime;
    /// the completeness gate bounds this set statically, so reconcile treats
    /// it as benign rather than re-deriving policy per-process.
    Uncontracted,
    /// The surface has a contract, but not for this subcommand (e.g. a
    /// build-only flag reconciled under check). Reported, never enforced:
    /// clap already rejects genuinely undeclared flags, so this arises only
    /// for surfaces accepted-but-inertly-scoped, which the entry-point
    /// declaration gate pins separately.
    OutOfScope,
}

/// Join the requested surfaces against the contract table and the disposition
/// log of the compile that just ran (the process-global [`pass_trace`] log).
///
/// Enforcement policy (error vs warn vs report) is the CLI's, which knows the
/// entry point and the escape-hatch flags.
pub fn reconcile(requested: &[RequestedSurface], entry: Subcommand) -> Vec<Outcome> {
    reconcile_with(requested, entry, &pass_trace::dispositions())
}

/// [`reconcile`] over an explicit disposition list. Pure: no globals, no I/O,
/// no exit — this is the function unit tests drive, because the global log is
/// process-wide and in-crate tests of other passes write to it concurrently.
pub fn reconcile_with(
    requested: &[RequestedSurface],
    entry: Subcommand,
    dispositions: &[(&'static str, PassDisposition)],
) -> Vec<Outcome> {
    requested
        .iter()
        .map(|req| {
            let Some(c) = contract_for(req) else {
                return Outcome { surface: req.clone(), owner: None, state: OutcomeState::Uncontracted };
            };
            if !c.on.contains(&entry) {
                return Outcome {
                    surface: req.clone(),
                    owner: Some(c.owner),
                    state: OutcomeState::OutOfScope,
                };
            }
            let state = match c.witness {
                Witness::Disposition(pass) => {
                    match dispositions.iter().find(|(p, _)| *p == pass) {
                        Some((_, d)) => OutcomeState::Disposed(*d),
                        None => OutcomeState::Unsatisfied { owner: pass },
                    }
                }
                w => OutcomeState::ByConstruction(w),
            };
            Outcome { surface: req.clone(), owner: Some(c.owner), state }
        })
        .collect()
}

/// The subset of outcomes that must fail the compile (absent the escape
/// hatch): requested, contracted for this entry, owner silent.
pub fn unsatisfied(outcomes: &[Outcome]) -> Vec<&Outcome> {
    outcomes
        .iter()
        .filter(|o| matches!(o.state, OutcomeState::Unsatisfied { .. }))
        .collect()
}

/// Render the full request→outcome table, one `[activation]`-prefixed line
/// per requested surface. Reuses the `[pass-trace]` disposition wording so
/// every outcome category keeps a single spelling in the tree.
pub fn render_report(outcomes: &[Outcome]) -> String {
    if outcomes.is_empty() {
        return "[activation] no optimization surfaces requested\n".to_string();
    }
    let mut s = String::new();
    for o in outcomes {
        let line = match &o.state {
            OutcomeState::Disposed(d) => format!(
                "[activation] {}: {} ({})",
                o.surface.render(),
                pass_trace::render_disposition_line(*d),
                o.owner.unwrap_or("?"),
            ),
            OutcomeState::ByConstruction(w) => format!(
                "[activation] {}: witnessed by construction ({})",
                o.surface.render(),
                match w {
                    Witness::Marker(m) => format!("marker {m}"),
                    Witness::Config(site) => format!("consumed at {site}"),
                    Witness::Report(site) => format!("report from {site}"),
                    Witness::Disposition(_) => unreachable!("Disposition is never ByConstruction"),
                },
            ),
            OutcomeState::Unsatisfied { owner } => format!(
                "[activation] {}: UNSATISFIED — {} recorded no disposition",
                o.surface.render(),
                owner,
            ),
            OutcomeState::Uncontracted => {
                format!("[activation] {}: no activation contract (plumbing)", o.surface.render())
            }
            OutcomeState::OutOfScope => format!(
                "[activation] {}: contracted, but not on this subcommand",
                o.surface.render(),
            ),
        };
        s.push_str(&line);
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pass-registry flag and decorator trigger must surface as a
    /// Disposition contract owned by its pass.
    #[test]
    fn registry_surfaces_become_disposition_contracts() {
        let cs = contracts();
        for p in pass_registry::PASSES {
            for f in p.cli_flags {
                let c = cs
                    .iter()
                    .find(|c| c.surface == Surface::Flag(f.flag))
                    .unwrap_or_else(|| panic!("--{} has no contract", f.flag));
                assert_eq!(c.owner, p.name, "--{} owner", f.flag);
                assert_eq!(c.witness, Witness::Disposition(p.name), "--{} witness", f.flag);
            }
            for d in p.decorator_triggers {
                assert!(
                    cs.iter().any(|c| c.surface == Surface::Decorator(d)),
                    "@{d} has no contract"
                );
            }
        }
    }

    /// A manual entry may not silently shadow a registry-derived one — the
    /// registry wins and the duplicate must be caught by review, so the
    /// derivation keeps first-wins order.
    #[test]
    fn registry_wins_over_manual_duplicates() {
        // @fase is both a FASE decorator_trigger and a MANUAL_CONTRACTS row;
        // the surviving contract must be the registry-derived one.
        let cs = contracts();
        let fase: Vec<_> = cs
            .iter()
            .filter(|c| c.surface == Surface::Decorator("fase"))
            .collect();
        assert_eq!(fase.len(), 1, "exactly one @fase contract must survive dedup");
        assert_eq!(fase[0].witness, Witness::Disposition("FASE"));
    }

    /// The reconciler's three-way split, over an explicit disposition list —
    /// the global log is process-wide and other passes' unit tests write to
    /// it concurrently, so tests never read it.
    #[test]
    fn reconcile_splits_disposed_unsatisfied_uncontracted() {
        let dispositions = [("WGGO", PassDisposition::Applied { rewrites: 3 })];
        let requested = vec![
            RequestedSurface::Flag("wggo".into()),               // disposed
            RequestedSurface::Flag("checkpoint-blocks".into()),  // CCR silent -> unsatisfied
            RequestedSurface::Flag("output".into()),             // plumbing -> uncontracted
        ];
        let outcomes = reconcile_with(&requested, Subcommand::Build, &dispositions);
        assert!(matches!(outcomes[0].state, OutcomeState::Disposed(_)), "{:?}", outcomes[0]);
        assert!(
            matches!(outcomes[1].state, OutcomeState::Unsatisfied { owner: "CCR" }),
            "{:?}",
            outcomes[1]
        );
        assert!(matches!(outcomes[2].state, OutcomeState::Uncontracted), "{:?}", outcomes[2]);
        assert_eq!(unsatisfied(&outcomes).len(), 1);
    }

    /// Check-side reconciliation of a build/run-scoped surface reports
    /// OutOfScope rather than Unsatisfied — check runs no codegen, so a
    /// silent owner there is not evidence of inertness.
    #[test]
    fn check_scope_is_reported_not_enforced() {
        let outcomes = reconcile_with(
            &[RequestedSurface::Decorator("cpdt".into())],
            Subcommand::Check,
            &[],
        );
        assert!(
            matches!(outcomes[0].state, OutcomeState::OutOfScope),
            "{:?}",
            outcomes[0]
        );
        assert!(unsatisfied(&outcomes).is_empty());
    }

    /// The report renders one line per requested surface and names the silent
    /// owner in the UNSATISFIED line — that string is the user's only pointer
    /// to WHICH component ignored them.
    #[test]
    fn report_names_the_silent_owner() {
        let outcomes = reconcile_with(
            &[RequestedSurface::Decorator("cpdt".into())],
            Subcommand::Build,
            &[],
        );
        let report = render_report(&outcomes);
        assert!(
            report.contains("@cpdt: UNSATISFIED — CPDT recorded no disposition"),
            "report was: {report}"
        );
    }

    /// A decline IS a satisfied contract — the reconciler must never treat
    /// Declined as inertness, or every legitimately-declined request would
    /// fail the build it correctly declined on.
    #[test]
    fn declined_is_satisfied_not_unsatisfied() {
        let dispositions = [(
            "CCR",
            PassDisposition::Declined { reason: crate::pass_trace::DeclineReason::ModeOff },
        )];
        let outcomes = reconcile_with(
            &[RequestedSurface::Flag("checkpoint-blocks".into())],
            Subcommand::Build,
            &dispositions,
        );
        assert!(matches!(outcomes[0].state, OutcomeState::Disposed(_)), "{:?}", outcomes[0]);
        assert!(unsatisfied(&outcomes).is_empty());
    }
}
