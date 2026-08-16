//! Milestone C coverage pins: every in-pipeline pass is SCHEDULED, every
//! scheduled invocation settles its postconditions, tape-staleness coverage
//! matches the registry's positional-reference declarations, and the ambient
//! epoch's caller surface cannot silently re-widen.
//!
//! These are the "did the milestone stick" gates. Each one is a count or an
//! enumeration pinned in BOTH directions, so reverting one conversion (a
//! schedule site disappearing), dropping one postcondition check, or adding
//! an unreviewed escape hatch fails here rather than degrading silently —
//! the declared-but-inert defect class this substrate exists to kill.

use std::path::Path;

fn src(rel: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Strip `/* block comments */`, then `//` line comments, then all
/// whitespace — the multi-line-call-chain lesson from `pass_bus_drift.rs`
/// (a line-based scan misses a call split across lines), PLUS the hole the
/// sibling gates accept and this file must not: every count here is
/// load-bearing arithmetic (schedule == finish + defer), so a
/// block-commented-out schedule/finish site counting as LIVE would let the
/// cheapest possible conversion revert — `/* */` around one wrapper — keep
/// every gate green while the pass runs unscheduled.
///
/// The block-strip is string-naive (a `/*` inside a string literal would
/// truncate to its closing `*/`) — none of the scanned needles can be
/// produced that way, and over-stripping only ever LOWERS counts, which
/// fails closed against the exact-equality pins.
fn code_only(text: &str) -> String {
    let mut no_blocks = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("/*") {
        no_blocks.push_str(&rest[..start]);
        match rest[start + 2..].find("*/") {
            Some(end) => rest = &rest[start + 2 + end + 2..],
            None => {
                rest = "";
            }
        }
    }
    no_blocks.push_str(rest);
    no_blocks
        .lines()
        .map(|l| match l.find("//") {
            Some(i) if !l[..i].contains('"') => &l[..i],
            _ => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .split_whitespace()
        .collect()
}

/// Every `.rs` file under `src/`, so a schedule site moving to a new module
/// is found rather than silently uncounted. Walked ONCE per test binary —
/// five tests share the scan (tests run as threads of one process).
fn all_src_code() -> &'static str {
    static CODE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CODE.get_or_init(|| {
        fn walk(dir: &Path, out: &mut String) {
            for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            {
                let p = e.expect("dir entry").path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push_str(&code_only(
                        &std::fs::read_to_string(&p)
                            .unwrap_or_else(|e| panic!("{}: {e}", p.display())),
                    ));
                    out.push('\n');
                }
            }
        }
        let mut out = String::new();
        walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut out);
        out
    })
}

fn count(hay: &str, needle: &str) -> usize {
    hay.match_indices(needle).count()
}

/// (pass, expected `schedule("NAME"` sites, where they live)
///
/// Pinned EXACTLY, not as a floor: a conversion reverting to a direct call
/// drops a count; an unreviewed new invocation site raises one. Both are
/// deliberate edits that must move this table in the same commit.
const SCHEDULED: &[(&str, usize, &str)] = &[
    ("WRGA", 1, "stmt.rs (invoke_wrga_if_enabled site)"),
    ("WGGO", 2, "stmt.rs in-place replan + wggo_prepass.rs prepass"),
    ("CSHA", 1, "stmt.rs (invoke_csha_if_enabled site)"),
    ("CCR", 1, "stmt.rs (the whole search+plan region)"),
    ("CPDT", 3, "stmt.rs: two wrapper offers + the in-place replan site"),
    ("CPKD", 1, "stmt.rs (cpkd::build_plan + publish)"),
    ("FASE", 1, "stmt.rs (plan + driver rewrite + disposition re-record)"),
    ("PCA", 2, "compiler/kernel.rs: segment-masked decision + per-doc admission"),
    ("MemoryPlanner", 2, "stmt.rs transient arena + entry_points.rs whole-program plan"),
];

/// Every in-pipeline pass is scheduled, at exactly its expected sites.
/// CEP and CFIE are absent by DESIGN: their `phases` declarations are empty
/// (out-of-band drivers) and `schedule()` refuses empty-phase passes
/// structurally — pinned below so the absence stays a decision.
#[test]
fn every_in_pipeline_pass_is_scheduled_at_its_expected_sites() {
    let code = all_src_code();
    let mut bad = Vec::new();
    for (pass, expected, wherever) in SCHEDULED {
        let got = count(&code, &format!("schedule(\"{pass}\""));
        if got != *expected {
            bad.push(format!(
                "{pass}: {got} schedule site(s), expected {expected} ({wherever})"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "scheduled-invocation coverage drifted:\n  {}\n\
         Reverting a pass to a direct call, or adding an invocation site, \
         must move the SCHEDULED table in the same commit.",
        bad.join("\n  ")
    );
    // The out-of-band pair stays unschedulable, not merely unscheduled.
    for oob in ["CEP", "CFIE"] {
        assert_eq!(
            count(&code, &format!("schedule(\"{oob}\"")),
            0,
            "{oob} is OutOfBand (empty phases — schedule() refuses it); a \
             schedule site for it cannot work and means the design changed"
        );
        let desc = nsl_codegen::pass_registry::pass(oob).expect("registered");
        assert!(
            desc.phases.is_empty(),
            "{oob} gained a compile phase — revisit whether it should now be \
             scheduled, and update this gate either way"
        );
    }
}

/// Every schedule() settles its postconditions: the finish(&bus) count plus
/// the defer_postconditions count equals the schedule count. `Scheduled` is
/// #[must_use], but a `let _ =` silences that warning — this pin does not
/// silence.
#[test]
fn every_scheduled_invocation_settles_its_postconditions() {
    let code = all_src_code();
    let schedules: usize = SCHEDULED.iter().map(|(_, n, _)| n).sum();
    // Anti-vacuity: the walker actually saw the sites.
    assert_eq!(
        count(code, "schedule(\""),
        schedules,
        "total schedule(\"..\") sites moved — update SCHEDULED first"
    );
    // The needle is the CALL SHAPE, not a receiver spelling: counting
    // `.finish(&self.bus)` literally would fail on a correct refactor that
    // binds the bus to a local or hoists a helper taking `compiler` (review
    // finding). `.finish(&` still requires a by-reference bus argument at
    // the call — the definition site cannot satisfy it (whitespace is
    // stripped, and pass_manager.rs's `fn finish(self, bus: &PassBus)` does
    // not contain the token).
    let finishes = count(code, ".finish(&");
    let defers = count(code, ".defer_postconditions(");
    assert_eq!(
        finishes + defers,
        schedules,
        "{schedules} scheduled invocations but {finishes} finish() + \
         {defers} defer_postconditions() — a Scheduled is being dropped \
         without its postconditions being either checked or explicitly, \
         reasonedly deferred"
    );
}

/// The postcondition-deferral surface: exactly ONE caller (WGGO's prepass,
/// where `wggo_overrides` is structurally not yet publishable), so the
/// escape hatch cannot quietly become a convention.
#[test]
fn defer_postconditions_callers_are_enumerated() {
    // Two direct pins, no allowlist machinery until a second caller exists
    // (review finding: generic scaffolding for a one-element set obscured
    // that the per-file count was hardcoded anyway). Growing the set means
    // replacing both asserts with the enumeration they pretend to be.
    assert_eq!(
        count(all_src_code(), ".defer_postconditions("),
        1,
        "a new defer_postconditions caller appeared — every deferral is a \
         reviewed decision with its reason; extend this gate with it, or \
         call finish()"
    );
    assert_eq!(
        count(&code_only(&src("src/wggo_prepass.rs")), ".defer_postconditions("),
        1,
        "the one deferral is the WGGO prepass invocation; it moved"
    );
}

/// Milestone C's tape-lifetime criterion, pinned against the registry:
/// every pass whose declared TapeAccess refs include PositionalIndex either
/// has an `assert_tape_unchanged_since("NAME"` site in the tree or is in
/// the exemption table with a reason — the `Invariant::Exempt` shape, so
/// "no assert" stays a visible decision.
#[test]
fn every_positional_ref_pass_has_a_staleness_assert_or_a_reason() {
    use nsl_codegen::pass_registry::{TapeAccess, TapeRef, PASSES};
    // (pass, why no assert exists)
    const EXEMPT: &[(&str, &str)] = &[(
        "CSHA",
        "positions are converted to OpIds AT the scan boundary inside the \
         scheduled window (collect_claimed_ops / the dispatch-map build); \
         OpIds are stable across wggo_prune's deletions, and an assert \
         against the post-prune list would refuse every prune+CSHA \
         composition for a mutation that invalidates nothing retained",
    )];
    let code = all_src_code();
    let mut bad = Vec::new();
    for d in PASSES {
        let refs = match d.tape {
            TapeAccess::Reads { refs, .. }
            | TapeAccess::MutatesInPlace { refs, .. }
            | TapeAccess::MutatesFork { refs, .. } => refs,
            TapeAccess::None => continue,
        };
        if !refs.contains(&TapeRef::PositionalIndex) {
            continue;
        }
        let has_assert = count(&code, &format!("assert_tape_unchanged_since(\"{}\"", d.name)) > 0;
        let exempt = EXEMPT.iter().find(|(n, _)| *n == d.name);
        match (has_assert, exempt) {
            (true, None) => {}
            (false, Some((_, reason))) => {
                assert!(
                    reason.len() > 30,
                    "{}: an exemption needs a real reason",
                    d.name
                );
            }
            (false, None) => bad.push(format!(
                "{}: declares PositionalIndex refs but has no staleness \
                 assert and no exemption",
                d.name
            )),
            (true, Some(_)) => bad.push(format!(
                "{}: both an assert and an exemption — the exemption is \
                 stale, delete it",
                d.name
            )),
        }
    }
    assert!(bad.is_empty(), "tape-lifetime coverage drifted:\n  {}", bad.join("\n  "));
}

/// CURRENT_EPOCH's ambient-reader surface, sealed (Milestone C criterion:
/// "CURRENT_EPOCH can begin disappearing"). The three remaining ambient
/// touchpoints are `record` / `record_disposition` (callee-side attribution
/// — removing them means threading an explicit context through every pass
/// entry, the TrainIR question) and the manager's own begin/restore/LIFO
/// check. Everything else is already explicit-epoch. A new caller of the
/// epoch machinery outside `pass_manager.rs` re-widens the ambient surface
/// and fails here.
#[test]
fn the_epoch_machinery_is_called_only_from_the_manager() {
    fn walk(dir: &Path, hits: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let p = e.expect("dir entry").path();
            if p.is_dir() {
                walk(&p, hits);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let name = p.file_name().unwrap().to_string_lossy().to_string();
                // pass_trace.rs DEFINES the machinery; pass_manager.rs is
                // the one permitted caller.
                if name == "pass_trace.rs" || name == "pass_manager.rs" {
                    continue;
                }
                let code = code_only(
                    &std::fs::read_to_string(&p)
                        .unwrap_or_else(|e| panic!("{}: {e}", p.display())),
                );
                for needle in ["begin_epoch(", "restore_epoch(", "current_epoch("] {
                    if code.contains(needle) {
                        hits.push(format!("{}: {needle}", p.display()));
                    }
                }
            }
        }
    }
    let mut hits = Vec::new();
    walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut hits);
    assert!(
        hits.is_empty(),
        "the epoch machinery gained callers outside pass_manager.rs:\n  {}\n\
         The ambient epoch is supposed to be SHRINKING — route new needs \
         through PassManager/PassScheduler, which carry the epoch by value.",
        hits.join("\n  ")
    );
    // Anti-vacuity: the permitted caller still exists and still calls it.
    let mgr = code_only(&src("src/pass_manager.rs"));
    assert!(
        mgr.contains("begin_epoch(") && mgr.contains("restore_epoch("),
        "pass_manager.rs no longer drives the epoch lifecycle — the walker \
         above is checking a machinery nobody uses"
    );
}

/// The declared InvocationOrdered edges form a DAG — the ordering the
/// manager enforces per compile is derivable from the declarations at all.
/// A cycle would make `enforce_dependency_order` refuse every compile that
/// runs the full set, and nothing else checks the declarations globally
/// (each edge is declared locally on its channel).
#[test]
fn the_declared_invocation_order_edges_form_a_dag() {
    use nsl_codegen::pass_bus::{OrderClaim, CHANNELS};
    let mut edges: Vec<(&str, &str)> = Vec::new();
    for d in CHANNELS {
        for c in d.consumed_by_passes {
            if c.order == OrderClaim::InvocationOrdered {
                edges.push((d.producer, c.pass));
            }
        }
    }
    assert!(
        edges.len() >= 2,
        "the InvocationOrdered edge set shrank below the known floor — \
         declarations lost, or the bus moved"
    );
    // Kahn: repeatedly remove nodes with no incoming edge.
    let mut nodes: Vec<&str> = edges
        .iter()
        .flat_map(|(a, b)| [*a, *b])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut remaining = edges.clone();
    while !nodes.is_empty() {
        let Some(i) = nodes
            .iter()
            .position(|n| !remaining.iter().any(|(_, to)| to == n))
        else {
            panic!(
                "the declared InvocationOrdered edges contain a cycle: \
                 {remaining:?} — enforce_dependency_order would refuse every \
                 full-pipeline compile"
            );
        };
        let n = nodes.remove(i);
        remaining.retain(|(from, _)| *from != n);
    }
}
