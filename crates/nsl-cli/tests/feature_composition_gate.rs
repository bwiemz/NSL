//! Item 20: feature-composition gates driven by `nsl_cli::feature_rules`.
//!
//! Three tiers, cheapest first. Each catches something the others cannot:
//!
//! * **A — clap drift.** `args.rs` declares every shared flag TWICE
//!   (`BuildArgs`, `RunArgs`). A `requires`/`conflicts_with` added to one and
//!   not the other silently makes `nsl run` and `nsl build` disagree about
//!   which compositions are legal. They agree today; this is what keeps them
//!   agreeing. Also cross-checks the registry against the parsed attributes in
//!   BOTH directions, so the registry cannot drift from reality either.
//! * **B — deleted refusal.** Every `Enforcement::Source` rule names a file and
//!   a message fragment. If the refusal is removed or reworded, the fragment
//!   stops matching and this fails. Without it, a deferral that "must refuse"
//!   is one edit away from becoming a silently-permitted composition — and the
//!   registry would keep asserting it was covered.
//! * **C — the refusal actually fires.** A/B are static; they prove the text is
//!   there, not that the code reaches it. Tier C runs the real compiler on a
//!   real fixture and asserts the SPECIFIC registered message, paired with a
//!   control proving the same fixture succeeds without the offending flag.
//!
//! Tier C is where the vacuity trap lives, so it is worth naming: a test that
//! asserts only "exit != 0" passes when the compile failed for an unrelated
//! reason — a typo'd fixture would "prove" every refusal at once. Matching the
//! registered fragment is what makes the assertion mean something, and the
//! control run is what proves the fixture reaches the refusal at all.

use nsl_cli::feature_rules::{
    clap_rules, flag_to_field, source_rules, Enforcement, RuleKind, FEATURE_RULES,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

// ───────────────────────────── source parsing ─────────────────────────────

/// One flag's parsed clap constraints: `(requires, conflicts)` as field names.
type Constraints = (BTreeSet<String>, BTreeSet<String>);

/// Extract `field -> (requires, conflicts)` from a `#[derive(Args)]` struct in
/// `args.rs`, by brace-matching the struct body and walking `#[arg(...)]`
/// attributes (which may span lines) up to the field they annotate.
fn parse_arg_struct(src: &str, struct_name: &str) -> BTreeMap<String, Constraints> {
    let lines: Vec<&str> = src.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.contains(&format!("struct {struct_name} {{")))
        .unwrap_or_else(|| panic!("{struct_name} not found in args.rs"));
    let mut depth = 0i32;
    let mut end = lines.len();
    for (k, l) in lines.iter().enumerate().skip(start) {
        depth += l.matches('{').count() as i32 - l.matches('}').count() as i32;
        if depth == 0 && k > start {
            end = k;
            break;
        }
    }

    let mut out = BTreeMap::new();
    let mut pending = String::new();
    let mut k = start;
    while k < end {
        let l = lines[k].trim();
        if l.starts_with("#[arg(") {
            // Attributes wrap across lines; join until the parens balance.
            let mut attr = l.to_string();
            let mut j = k;
            while attr.matches('(').count() > attr.matches(')').count() && j + 1 < end {
                j += 1;
                attr.push(' ');
                attr.push_str(lines[j].trim());
            }
            pending.push_str(&attr);
            k = j + 1;
            continue;
        }
        if let Some(field) = l
            .strip_prefix("pub(crate) ")
            .and_then(|r| r.split(':').next())
            .filter(|f| !f.is_empty() && f.chars().all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()))
        {
            out.insert(field.to_string(), extract_constraints(&pending));
            pending.clear();
        }
        k += 1;
    }
    out
}

/// Pull `requires`/`requires_all`/`conflicts_with`/`conflicts_with_all` values
/// out of a joined `#[arg(...)]` attribute.
fn extract_constraints(attr: &str) -> Constraints {
    let requires = collect_key(attr, &["requires_all", "requires"]);
    let conflicts = collect_key(attr, &["conflicts_with_all", "conflicts_with"]);
    (requires, conflicts)
}

/// Values for `key = ...` AND `key(...)`.
///
/// Both forms are honored by clap_derive — `#[arg(conflicts_with("tape_ad"))]`
/// is exactly as binding as `conflicts_with = "tape_ad"`. An earlier version
/// of this parser accepted only the `=` form and silently returned nothing for
/// the other, which made the drift gate pass while `nsl run` and `nsl build`
/// genuinely disagreed. Anything that is neither form is reported by
/// [`unparsed_constraint_values`] rather than dropped, because a value this
/// cannot read (a const, an expression) is a rule the gate cannot see.
fn collect_key(attr: &str, keys: &[&str]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for key in keys {
        let mut rest = attr;
        while let Some(pos) = rest.find(key) {
            let tail = &rest[pos + key.len()..];
            // `requires` must not match inside `requires_all`; the longer key
            // is tried first, so reject a continuation into an ident char.
            let boundary_ok = !tail
                .chars()
                .next()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
            if boundary_ok {
                let t = tail.trim_start();
                let body = if let Some(eq) = t.strip_prefix('=') {
                    Some(eq.trim_start())
                } else {
                    t.strip_prefix('(').map(|c| c.trim_start())
                };
                if let Some(body) = body {
                    read_values(body, &mut out);
                }
            }
            rest = &rest[pos + key.len()..];
        }
    }
    out
}

/// Read `"lit"`, `["a", "b"]`, or `("a")` — the value forms clap accepts.
fn read_values(body: &str, out: &mut BTreeSet<String>) {
    let body = body.trim_start();
    if let Some(list) = body.strip_prefix('[') {
        if let Some(close) = list.find(']') {
            for tok in list[..close].split(',') {
                let t = tok.trim().trim_matches('"').trim();
                if !t.is_empty() {
                    out.insert(t.to_string());
                }
            }
        }
    } else if let Some(lit) = body.strip_prefix('"') {
        if let Some(close) = lit.find('"') {
            out.insert(lit[..close].to_string());
        }
    }
}

/// `(field, joined #[arg(...)] attribute)` for one struct — the raw form, so
/// callers can inspect values `extract_constraints` chose not to keep.
fn raw_attrs(src: &str, struct_name: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = src.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.contains(&format!("struct {struct_name} {{")))
        .unwrap_or_else(|| panic!("{struct_name} not found in args.rs"));
    let mut depth = 0i32;
    let mut end = lines.len();
    for (k, l) in lines.iter().enumerate().skip(start) {
        depth += l.matches('{').count() as i32 - l.matches('}').count() as i32;
        if depth == 0 && k > start {
            end = k;
            break;
        }
    }
    let mut out = Vec::new();
    let mut pending = String::new();
    let mut k = start;
    while k < end {
        let l = lines[k].trim();
        if l.starts_with("#[arg(") {
            let mut attr = l.to_string();
            let mut j = k;
            while attr.matches('(').count() > attr.matches(')').count() && j + 1 < end {
                j += 1;
                attr.push(' ');
                attr.push_str(lines[j].trim());
            }
            pending.push_str(&attr);
            k = j + 1;
            continue;
        }
        if let Some(field) = l
            .strip_prefix("pub(crate) ")
            .and_then(|r| r.split(':').next())
            .filter(|f| {
                !f.is_empty()
                    && f.chars()
                        .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
            })
        {
            out.push((field.to_string(), std::mem::take(&mut pending)));
        }
        k += 1;
    }
    out
}

/// Constraint values this parser could NOT read — a const path, an expression,
/// anything not a string literal or literal list. Each is a composition rule
/// invisible to every gate here, so they are surfaced as a failure rather than
/// silently skipped.
fn unparsed_constraint_values(attr: &str) -> Vec<String> {
    let mut bad = Vec::new();
    for key in ["requires_all", "requires", "conflicts_with_all", "conflicts_with"] {
        let mut rest = attr;
        while let Some(pos) = rest.find(key) {
            let tail = &rest[pos + key.len()..];
            let boundary_ok = !tail
                .chars()
                .next()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
            if boundary_ok {
                let t = tail.trim_start();
                let body = t
                    .strip_prefix('=')
                    .or_else(|| t.strip_prefix('('))
                    .map(str::trim_start);
                if let Some(b) = body {
                    let mut probe = BTreeSet::new();
                    read_values(b, &mut probe);
                    if probe.is_empty() {
                        bad.push(format!("{key} -> {:.40}", b));
                    }
                }
            }
            rest = &rest[pos + key.len()..];
        }
    }
    bad
}

fn args_src() -> String {
    std::fs::read_to_string(repo_root().join("crates/nsl-cli/src/args.rs")).expect("read args.rs")
}

// ───────────────────────────── Tier A: clap drift ─────────────────────────

#[test]
fn run_and_build_declare_identical_composition_rules() {
    // THE drift gate. `args.rs` repeats every shared flag in two structs; a
    // constraint added to one and not the other means `nsl build` refuses a
    // composition that `nsl run` silently accepts (or vice versa). Nothing
    // else in the tree checks this.
    let src = args_src();
    let build = parse_arg_struct(&src, "BuildArgs");
    let run = parse_arg_struct(&src, "RunArgs");

    let mut drift = Vec::new();
    for (field, b) in &build {
        let Some(r) = run.get(field) else { continue }; // subcommand-specific
        if b != r {
            drift.push(format!(
                "  {field}:\n     BuildArgs requires={:?} conflicts={:?}\n     RunArgs   requires={:?} conflicts={:?}",
                b.0, b.1, r.0, r.1
            ));
        }
    }
    assert!(
        drift.is_empty(),
        "`nsl run` and `nsl build` disagree about {} flag(s). A composition \
         refused by one subcommand and accepted by the other is exactly the \
         drift this gate exists to catch — fix BOTH arg blocks in \
         crates/nsl-cli/src/args.rs:\n{}",
        drift.len(),
        drift.join("\n")
    );

    // ANTI-VACUITY: if the parser silently matched nothing, the loop above is
    // trivially satisfied. Both structs must have real, constrained fields.
    let shared_constrained = build
        .iter()
        .filter(|(f, c)| run.contains_key(*f) && !(c.0.is_empty() && c.1.is_empty()))
        .count();
    assert!(
        shared_constrained >= 12,
        "parsed only {shared_constrained} constrained shared flags (expected \
         12) — the args.rs parser has stopped matching, making this gate \
         vacuous. Raise this floor when a constrained flag is added, so a \
         parser that truncates part-way still trips it."
    );
}

#[test]
fn no_clap_constraint_value_is_unreadable_by_this_parser() {
    // A `requires = SOME_CONST`, or any value that is not a string literal or
    // literal list, parses to NOTHING here — so the rule stays enforced by
    // clap but invisible to every gate in this file. That is a silent coverage
    // hole, which is the failure mode this whole file exists to prevent, so it
    // fails loudly instead.
    let src = args_src();
    let mut bad = Vec::new();
    for struct_name in ["BuildArgs", "RunArgs"] {
        for (field, attr) in raw_attrs(&src, struct_name) {
            for v in unparsed_constraint_values(&attr) {
                bad.push(format!("  {struct_name}::{field}: {v}"));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "clap constraint value(s) this gate cannot read — enforced, but \
         invisible to the registry cross-check. Teach `read_values` the new \
         form, or express the constraint as a string literal:\n{}",
        bad.join("\n")
    );
}

#[test]
fn every_clap_constraint_is_registered_and_vice_versa() {
    // Both directions, because each catches a different rot:
    //   source -> registry : a new conflict lands with no registry entry, so
    //                        the deleted-refusal gate never learns about it.
    //   registry -> source : the registry claims a rule clap does not enforce,
    //                        i.e. the table lies about coverage.
    let src = args_src();
    let build = parse_arg_struct(&src, "BuildArgs");
    let run = parse_arg_struct(&src, "RunArgs");

    let mut in_source: BTreeSet<(String, &'static str, String)> = BTreeSet::new();
    for map in [&build, &run] {
        for (field, (reqs, cons)) in map {
            for r in reqs {
                in_source.insert((format!("--{}", field.replace('_', "-")), "Requires", r.clone()));
            }
            for c in cons {
                in_source.insert((format!("--{}", field.replace('_', "-")), "Conflicts", c.clone()));
            }
        }
    }

    let in_registry: BTreeSet<(String, &'static str, String)> = clap_rules()
        .map(|r| {
            (
                r.flag.to_string(),
                match r.kind {
                    RuleKind::Requires => "Requires",
                    RuleKind::Conflicts => "Conflicts",
                },
                r.other.to_string(),
            )
        })
        .collect();

    let missing: Vec<_> = in_source.difference(&in_registry).collect();
    let extra: Vec<_> = in_registry.difference(&in_source).collect();
    assert!(
        missing.is_empty(),
        "clap declares composition rules the registry does not know about — \
         add them to crates/nsl-cli/src/feature_rules.rs:\n{missing:#?}"
    );
    assert!(
        extra.is_empty(),
        "the registry claims clap-enforced rules that are NOT in args.rs. \
         Either clap lost an attribute (a real regression: the composition is \
         now permitted) or the registry is stale — in both cases the table is \
         lying about coverage:\n{extra:#?}"
    );
    // ANTI-VACUITY. Assert on the PARSER's output, per struct — not on
    // `in_registry`, which is a const array the parser cannot affect (its
    // "the parser collapsed" message would have been unreachable), and not on
    // the union, which stays complete when only ONE struct fails to parse.
    for (name, map) in [("BuildArgs", &build), ("RunArgs", &run)] {
        let n = map
            .values()
            .filter(|(r, c)| !r.is_empty() || !c.is_empty())
            .count();
        assert!(
            n >= 12,
            "parsed only {n} constrained fields from {name}; the args.rs \
             parser has stopped matching and every assertion above is vacuous"
        );
    }
}

// ─────────────────────── Tier B: the refusal still exists ─────────────────

/// Markers that begin a refusal. A fragment only counts if it appears in the
/// text FOLLOWING one of these.
const REFUSAL_MARKERS: &[&str] = &[
    "CodegenError::new(",
    "eprintln!(",
    "panic!(",
    "Err(format!(",
    "format!(",
];

/// The refusal-bearing text of a file, normalized.
///
/// Two transformations, each load-bearing:
///
/// * **Whitespace collapse.** Rust literals here wrap with `\` continuations
///   and indentation, so `--layerwise-accum requires --source-ad` is not
///   contiguous in the raw bytes.
/// * **Anchoring to refusal sites.** Searching the WHOLE file would let a
///   deleted refusal keep this gate green as long as its message survived in
///   a `//` comment or a `#[cfg(test)]` assertion — which is exactly how a
///   removed guard would look after someone "documented why it used to be
///   there". Only the span following a refusal marker counts.
///
/// This still cannot see the GUARD. Narrowing a refusal's condition (say
/// `a && (b || c)` to `a && b`) leaves the message untouched and passes here;
/// only the subprocess tier can catch that, and it covers one rule. Stated
/// plainly because a gate's limits are as important as its coverage.
fn refusal_text(path: &std::path::Path) -> String {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let code = strip_comments(&raw);
    let unwrapped = code.replace("\\\n", " ");

    let mut spans = String::new();
    for marker in REFUSAL_MARKERS {
        let mut from = 0usize;
        while let Some(pos) = unwrapped[from..].find(marker) {
            let start = from + pos + marker.len();
            // Capture exactly this call's ARGUMENTS — up to its matching close
            // paren — rather than a fixed window. A fixed window reaches past
            // the call into whatever follows, which is how the first version
            // of this gate stayed green after a refusal was deleted and its
            // message left behind nearby.
            let mut depth = 1i32;
            let mut end = start;
            for (i, c) in unwrapped[start..].char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = start + i;
                            break;
                        }
                    }
                    _ => {}
                }
                end = start + i + c.len_utf8();
            }
            spans.push_str(&unwrapped[start..end]);
            spans.push(' ');
            from = start;
        }
    }
    spans.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Remove `//` and `/* */` comments while respecting string literals, so a
/// refusal message quoted in a comment cannot stand in for the refusal.
fn strip_comments(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let (mut i, n) = (0usize, b.len());
    let (mut in_str, mut in_line, mut in_block) = (false, false, false);
    while i < n {
        let c = b[i];
        let next = if i + 1 < n { b[i + 1] } else { '\0' };
        if in_line {
            if c == '\n' {
                in_line = false;
                out.push(c);
            }
            i += 1;
        } else if in_block {
            if c == '*' && next == '/' {
                in_block = false;
                i += 2;
            } else {
                if c == '\n' {
                    out.push(c);
                }
                i += 1;
            }
        } else if in_str {
            if c == '\\' {
                out.push(c);
                if i + 1 < n {
                    out.push(next);
                }
                i += 2;
            } else {
                if c == '"' {
                    in_str = false;
                }
                out.push(c);
                i += 1;
            }
        } else if c == '/' && next == '/' {
            in_line = true;
            i += 2;
        } else if c == '/' && next == '*' {
            in_block = true;
            i += 2;
        } else {
            if c == '"' {
                in_str = true;
            }
            out.push(c);
            i += 1;
        }
    }
    out
}

#[test]
fn every_registered_refusal_message_still_exists_in_its_source() {
    // The doctrine this defends: an unsupported composition must REFUSE, and
    // a deferral with no gate is one edit away from gone. Each rule names the
    // file that is supposed to hold its refusal; if the message is deleted or
    // reworded, the composition silently becomes permitted while the registry
    // still advertises it as covered.
    let root = repo_root();
    let mut cache: BTreeMap<&'static str, String> = BTreeMap::new();
    let mut lost = Vec::new();
    let mut checked = 0usize;

    for rule in source_rules() {
        let Enforcement::Source { file, fragment } = rule.enforcement else {
            unreachable!("source_rules() filters on Enforcement::Source")
        };
        let hay = cache
            .entry(file)
            .or_insert_with(|| refusal_text(&root.join(file)));
        let needle = fragment.split_whitespace().collect::<Vec<_>>().join(" ");
        checked += 1;
        if !hay.contains(&needle) {
            lost.push(format!(
                "  {} {:?} {} — fragment not found in {file}:\n      {fragment:?}",
                rule.flag, rule.kind, rule.other
            ));
        }
    }

    assert!(
        lost.is_empty(),
        "{} registered refusal(s) no longer exist in the source. Either the \
         refusal was removed (the composition is now SILENTLY PERMITTED — a \
         `deferral must refuse` violation) or the message was reworded (update \
         the fragment in crates/nsl-cli/src/feature_rules.rs):\n{}",
        lost.len(),
        lost.join("\n")
    );
    assert!(
        checked >= 30,
        "only {checked} source-enforced rules checked; the registry has \
         shrunk unexpectedly and this gate is near-vacuous"
    );
}

#[test]
fn refusal_fragments_are_distinctive_enough_to_be_meaningful() {
    // ANTI-VACUITY for the gate above. A fragment like "requires" would match
    // any file and make the check pass forever. Every fragment must name at
    // least one flag and be long enough to be specific.
    let mut weak = Vec::new();
    for rule in source_rules() {
        let Enforcement::Source { fragment, .. } = rule.enforcement else {
            continue;
        };
        // `contains("--")` alone was near-useless: every fragment starts with
        // its own flag, so it passed unconditionally. What matters is that the
        // fragment pins the PARTNER too — otherwise a refusal can be narrowed
        // to drop `other` from its condition while the pinned text survives
        // untouched. Review found exactly that on the --param-dtype x
        // --optim-state-offload entry, whose fragment stopped one clause short
        // of naming its partner.
        let names_partner = !rule.other.starts_with("--") || fragment.contains(rule.other);
        if fragment.len() < 24 || !fragment.contains("--") || !names_partner {
            weak.push(format!(
                "  {} {:?} {} -> {fragment:?}",
                rule.flag, rule.kind, rule.other
            ));
        }
    }
    assert!(
        weak.is_empty(),
        "refusal fragments must name a flag and be specific enough not to \
         match incidentally:\n{}",
        weak.join("\n")
    );
}

#[test]
fn registry_has_no_duplicate_rules() {
    // A duplicated (flag, kind, other) triple inflates the coverage counts the
    // gates above assert on, which is the cheapest way to make them vacuous.
    let mut seen = BTreeSet::new();
    let mut dupes = Vec::new();
    for r in FEATURE_RULES {
        let key = (r.flag, format!("{:?}", r.kind), r.other, format!("{:?}", r.enforcement));
        if !seen.insert(key.clone()) {
            dupes.push(format!("  {key:?}"));
        }
    }
    assert!(dupes.is_empty(), "duplicate registry rules:\n{}", dupes.join("\n"));
}

#[test]
fn flag_to_field_matches_clap_naming() {
    assert_eq!(flag_to_field("--weight-stream"), "weight_stream");
    assert_eq!(flag_to_field("--source-ad"), "source_ad");
    assert_eq!(flag_to_field("--zero-stage"), "zero_stage");
}

// ──────────────────── Tier C: the refusal actually fires ──────────────────

/// A tier-C case: run the compiler with `flags` on the CSLA fixture and
/// require the registered `fragment` on stderr.
struct FireCase {
    flags: &'static [&'static str],
    /// Registered fragment that must appear. Kept in sync with the registry by
    /// `tier_c_fragments_are_registered` below.
    fragment: &'static str,
}

fn run_nsl(fixture: &std::path::Path, extra: &[&str]) -> (bool, String) {
    let root = repo_root();
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "-p", "nsl-cli"]);
    // Exercise the binary built with the SAME feature set as this test.
    // Without this, a `--features cuda` run silently drives the default build.
    if cfg!(feature = "cuda") {
        cmd.args(["--features", "cuda"]);
    }
    let out = cmd
        .arg("--manifest-path")
        .arg(root.join("Cargo.toml"))
        .args(["--", "run"])
        .args(extra)
        .arg(fixture)
        // Scratch cwd, NOT the repo: anything the compiled program writes
        // relatively then lands in temp. A CSLA_SAVE_PATH-shaped mistake once
        // dropped a stray file in the repo root from exactly this call.
        .current_dir(std::env::temp_dir())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let mut combined = String::from_utf8_lossy(&out.stderr).into_owned();
    combined.push_str(&String::from_utf8_lossy(&out.stdout));
    (out.status.success(), combined)
}

/// Every tier-C fragment must be one the registry actually declares —
/// otherwise tier C could assert a message that no rule pins, and tier B's
/// deleted-refusal check would never cover it.
#[test]
fn tier_c_fragments_are_registered() {
    let registered: BTreeSet<&str> = source_rules()
        .filter_map(|r| match r.enforcement {
            Enforcement::Source { fragment, .. } => Some(fragment),
            Enforcement::Clap => None,
        })
        .collect();
    assert!(
        !TIER_C.is_empty(),
        "TIER_C is empty — this test and the subprocess gate both degrade to \
         asserting nothing"
    );
    for case in TIER_C {
        assert!(
            registered.contains(case.fragment),
            "tier-C case asserts {:?}, which no registry rule declares — tier \
             B would never notice if that refusal disappeared",
            case.fragment
        );
    }
}

/// The supported baseline the control run uses. Every tier-C case is this set
/// PLUS the flag(s) under test, so a refusal can only be attributed to what
/// the case added.
const TIER_C_BASE: &[&str] = &["--source-ad", "--checkpoint-blocks", "--layerwise-accum"];

/// Scoped deliberately. A case belongs here only if the SOURCE-level refusal
/// is what the CLI actually reaches — measured, not assumed. Notably absent:
/// `--weight-stream requires --layerwise-accum`, which `stmt.rs` also refuses
/// but clap rejects first (see `feature_rules`' two-layer note), so driving
/// the binary would test clap while appearing to test `stmt.rs`.
///
/// Broadening this means adding fixtures that genuinely reach each refusal —
/// not reusing this one and accepting a failure for the wrong reason.
const TIER_C: &[FireCase] = &[FireCase {
    flags: &["--checkpoint-selective", "--checkpoint-compress", "fp16"],
    fragment: "--layerwise-accum is incompatible with --checkpoint-compress",
}];

/// Materialize the CSLA fixture into a temp file with its `CSLA_SAVE_PATH`
/// marker rewritten, following the same convention as
/// `csla_layerwise_gate.rs`. Running the fixture verbatim makes `model_save`
/// write a file literally named `CSLA_SAVE_PATH` into the CURRENT DIRECTORY —
/// which for this gate is the repo root.
fn materialize_csla_fixture() -> PathBuf {
    let root = repo_root();
    let src_path = root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", src_path.display()));
    assert!(
        src.contains("CSLA_SAVE_PATH"),
        "fixture lost its CSLA_SAVE_PATH marker; this gate would write into \
         the repo root"
    );
    // PID-suffixed, matching csla_layerwise_gate's convention: a fixed name in
    // a world-writable dir collides between concurrent `cargo test` runs, and
    // between users on a shared runner.
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let out = tmp.join(format!("nsl_feature_composition_csla_{pid}.nsl"));
    let save = tmp.join(format!("nsl_feature_composition_csla_{pid}.nslm"));
    std::fs::write(
        &out,
        src.replace("CSLA_SAVE_PATH", save.to_str().expect("utf-8 temp path")),
    )
    .expect("write temp fixture");
    out
}

#[test]
fn registered_refusals_actually_fire_on_a_real_compile() {
    let fixture = materialize_csla_fixture();

    // CONTROL FIRST. If the fixture cannot compile cleanly in its supported
    // configuration, every refusal assertion below would pass for the wrong
    // reason. This is the single most important line in tier C.
    let (ok, out) = run_nsl(&fixture, TIER_C_BASE);
    assert!(
        ok,
        "CONTROL FAILED: the fixture must compile in its SUPPORTED \
         configuration {TIER_C_BASE:?}, or the refusal assertions below prove \
         nothing (they would pass on any unrelated error).\n{out}"
    );

    for case in TIER_C {
        let mut args: Vec<&str> = TIER_C_BASE.to_vec();
        args.extend_from_slice(case.flags);
        let (ok, out) = run_nsl(&fixture, &args);
        assert!(!ok, "{:?} must be refused, but the compile succeeded", args);
        assert!(
            out.contains(case.fragment),
            "refused, but NOT for the registered reason {:?} — the compile may \
             have failed for an unrelated cause, which would make this \
             assertion vacuous. args={args:?}\n{out}",
            case.fragment
        );
    }
}

// ──────────────── Tier B2: the registry cannot be silently INCOMPLETE ──────

/// Refusals deliberately outside this registry's scope, with the reason.
/// Anything here is a conscious exclusion, not an oversight — which is the
/// distinction the sweep below exists to force someone to make.
const SWEEP_ALLOWLIST: &[(&str, &str)] = &[
    // Flag x LANGUAGE CONSTRUCT, not flag x flag: the partner is `@pipeline`
    // or a `grad_clip=` train-block argument, neither of which is a CLI flag,
    // so a pairwise CLI registry cannot express them.
    ("pipelined train", "flag x @pipeline decorator"),
    ("grad_clip", "flag x train-block argument"),
    ("CSHA-claimed fused attention", "flag x @flash_attention decorator"),
    // Diagnostics and advice, not composition refusals.
    ("re-run with --source-ad", "advice appended to an unrelated error"),
    // ADVISORY, not a refusal: --health-interval without --monitor warns and
    // continues. The registry models refusals — RuleKind has no "advisory"
    // variant — so recording it here would overstate what is enforced.
    ("has no effect without --monitor", "advisory warning, run continues"),
];

/// Two-flag refusals present in the source but absent from the registry.
///
/// Tier A is bidirectional for clap rules; without this, the SOURCE half is
/// one-way — a refusal added tomorrow would simply never be registered, and
/// therefore never gated, while `FEATURE_RULES`' doc claims to hold "every
/// composition rule the compiler enforces". This is the sweep that keeps that
/// claim honest.
#[test]
fn no_two_flag_refusal_is_missing_from_the_registry() {
    let root = repo_root();
    let files: BTreeSet<&'static str> = source_rules()
        .filter_map(|r| match r.enforcement {
            Enforcement::Source { file, .. } => Some(file),
            Enforcement::Clap => None,
        })
        .collect();

    let registered: Vec<&str> = source_rules()
        .filter_map(|r| match r.enforcement {
            Enforcement::Source { fragment, .. } => Some(fragment),
            Enforcement::Clap => None,
        })
        .collect();

    let mut unregistered = Vec::new();
    for file in &files {
        let text = refusal_text(&root.join(file));
        // Split the anchored refusal text into individual string literals.
        for lit in text.split('"') {
            let flags: BTreeSet<&str> = lit
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
                .filter(|t| t.starts_with("--") && t.len() > 4)
                .collect();
            if flags.len() < 2 {
                continue;
            }
            if registered.iter().any(|frag| {
                let f = frag.split_whitespace().collect::<Vec<_>>().join(" ");
                lit.contains(&f)
            }) {
                continue;
            }
            if SWEEP_ALLOWLIST.iter().any(|(pat, _)| lit.contains(pat)) {
                continue;
            }
            unregistered.push(format!("  {file}: {:?}\n      flags: {flags:?}", &lit[..lit.len().min(150)]));
        }
    }

    assert!(
        unregistered.is_empty(),
        "{} two-flag refusal(s) exist in the source but are NOT in \
         crates/nsl-cli/src/feature_rules.rs, so nothing gates them. Register \
         each, or add it to SWEEP_ALLOWLIST with a reason:\n{}",
        unregistered.len(),
        unregistered.join("\n")
    );
}
