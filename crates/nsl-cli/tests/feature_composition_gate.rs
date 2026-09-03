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
    } else if let Some(lit) = body.strip_prefix('"')
        && let Some(close) = lit.find('"')
    {
        out.insert(lit[..close].to_string());
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
    anchored_text(path, REFUSAL_MARKERS)
}

/// Print macros — the call sites that can put a marker on stderr.
const PRINT_MARKERS: &[&str] = &[
    "eprintln!(",
    "println!(",
    "eprint!(",
    "print!(",
    "writeln!(",
    "write!(",
    "format!(",
];

/// The argument text of every call to one of `markers`, comment-stripped and
/// whitespace-normalized. See [`refusal_text`] for why both transformations
/// are load-bearing.
fn anchored_text(path: &std::path::Path, markers: &[&str]) -> String {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    // CRLF FIRST. The continuation unwrap below matches `\` immediately
    // followed by `\n`; under a Windows checkout (`core.autocrlf` is on by
    // default there) the bytes are `\` `\r` `\n`, no match, and every
    // multi-line refusal literal stays wrapped — so the registered fragment
    // no longer occurs in the text and the sweep reports a REGISTERED refusal
    // as missing. That is what red-lined windows-latest on #481's
    // `--fuse-lm-head` rule while linux stayed green: not a rule that was
    // absent, a gate that could not read it.
    let raw = raw.replace("\r\n", "\n");
    let code = strip_comments(&raw);
    let unwrapped = code.replace("\\\n", " ");

    let mut spans = String::new();
    for marker in markers {
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
    /// What the case is, for the failure message.
    name: &'static str,
    /// The FULL argument list (fixture excluded).
    flags: &'static [&'static str],
    /// Registered fragment that must appear. Kept in sync with the registry by
    /// `tier_c_fragments_are_registered` below.
    fragment: &'static str,
    /// Args that must SUCCEED. This is what makes the refusal attributable:
    /// without it, a case passes when the compile failed for any reason at
    /// all. Every control is run (deduplicated) before any case.
    control: &'static [&'static str],
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
        !TIER_C.is_empty() && !TIER_C_GPU.is_empty(),
        "a tier-C table is empty — this test and the subprocess gates degrade \
         to asserting nothing"
    );
    for case in TIER_C.iter().chain(TIER_C_GPU) {
        assert!(
            registered.contains(case.fragment),
            "tier-C case {:?} asserts {:?}, which no registry rule declares — \
             tier B would never notice if that refusal disappeared",
            case.name,
            case.fragment
        );
        // Each case must DIFFER from its control, or the two runs are the
        // same invocation and the refusal is attributable to nothing.
        // Deliberately not "flags must be longer": a `requires` case is
        // expressed by REMOVING the prerequisite (e.g. --cuda-graphs without
        // --source-ad), which is the same length as its control.
        assert!(
            case.flags != case.control,
            "tier-C case {:?} is identical to its control — the refusal cannot \
             be attributed to anything",
            case.name
        );
    }
}

/// The supported baseline most tier-C controls use.
///
/// Most cases are this set PLUS the flag under test. One is not: a `requires`
/// rule is expressed by REMOVING the prerequisite (`--cuda-graphs` without
/// `--source-ad`), so that case differs from its control by two edits.
/// Attribution there rests on guard ORDER — stmt.rs:4656 (--cuda-graphs
/// requires --source-ad) is reached before stmt.rs:5276 (--layerwise-accum
/// requires --source-ad) — which is why the case asserts the specific
/// fragment rather than merely a nonzero exit.
const TIER_C_BASE: &[&str] = &["--source-ad", "--checkpoint-blocks", "--layerwise-accum"];

const CG: &[&str] = &[
    "--source-ad",
    "--checkpoint-blocks",
    "--layerwise-accum",
    "--cuda-graphs",
];

/// Every case here was MEASURED to reach its registered refusal, and to have a
/// control that genuinely compiles — not assumed from reading the guard.
///
/// Deliberately absent, each for a stated reason:
/// * `--weight-stream requires --layerwise-accum` — `stmt.rs` refuses it, but
///   clap rejects it first, so driving the binary would exercise clap while
///   appearing to exercise `stmt.rs` (see `feature_rules`' two-layer note).
/// * the `--param-dtype` x `--optim-state-offload` pair — its control needs
///   `--weight-stream`, which aborts at RUNTIME on a CPU-resident model
///   ("--weight-stream requires GPU placement"). It lives in `TIER_C_GPU`.
const TIER_C: &[FireCase] = &[
    FireCase {
        name: "CSLA x --checkpoint-compress",
        flags: &[
            "--source-ad",
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--checkpoint-selective",
            "--checkpoint-compress",
            "fp16",
        ],
        fragment: "--layerwise-accum is incompatible with --checkpoint-compress",
        control: TIER_C_BASE,
    },
    FireCase {
        name: "CSLA x --zero-stage 1",
        flags: &[
            "--source-ad",
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--zero-stage",
            "1",
        ],
        fragment: "--layerwise-accum is incompatible with --zero-stage 1/2",
        control: TIER_C_BASE,
    },
    FireCase {
        name: "--zero-stage 3 without --weight-stream",
        flags: &[
            "--source-ad",
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--zero-stage",
            "3",
        ],
        fragment: "--zero-stage 3 requires --layerwise-accum --weight-stream",
        control: TIER_C_BASE,
    },
    FireCase {
        // Note this rule is enforced ONLY in codegen — unlike
        // --fuse-rmsnorm-backward / --fuse-wgrad-accum, --cuda-graphs carries
        // no clap `requires = source_ad`. Dropping --source-ad therefore
        // reaches the stmt.rs refusal rather than a clap error.
        name: "--cuda-graphs without --source-ad",
        flags: &["--checkpoint-blocks", "--layerwise-accum", "--cuda-graphs"],
        fragment: "--cuda-graphs requires --source-ad",
        control: TIER_C_BASE,
    },
    FireCase {
        name: "--cuda-graphs x --zero-stage",
        flags: &[
            "--source-ad",
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--cuda-graphs",
            "--zero-stage",
            "1",
        ],
        fragment: "--cuda-graphs does not compose with --zero-stage",
        control: CG,
    },
    FireCase {
        name: "--cuda-graphs x --cuda-sync",
        flags: &[
            "--source-ad",
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--cuda-graphs",
            "--cuda-sync",
        ],
        fragment: "--cuda-graphs does not compose with --cuda-sync",
        control: CG,
    },
    FireCase {
        name: "--cuda-graphs x --profile-kernels",
        flags: &[
            "--source-ad",
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--cuda-graphs",
            "--profile-kernels",
        ],
        fragment: "--cuda-graphs does not compose with --profile-kernels",
        control: CG,
    },
    FireCase {
        name: "--param-dtype bf16-sr without --weight-stream",
        flags: &[
            "--source-ad",
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--param-dtype",
            "bf16-sr",
        ],
        fragment: "--param-dtype bf16-sr requires --weight-stream",
        control: TIER_C_BASE,
    },
];

/// Cases whose CONTROL needs a device. `--weight-stream` aborts at runtime on
/// a CPU-resident model, so anything downstream of it cannot have a passing
/// control in the CPU tier — the refusal would be indistinguishable from that
/// abort. Runs against a `m.to(cuda)` fixture under `--ignored`.
const TIER_C_GPU: &[FireCase] = &[FireCase {
    name: "--param-dtype bf16-sr x --optim-state-offload",
    flags: &[
        "--source-ad",
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
        "--param-dtype",
        "bf16-sr",
        "--optim-state-offload",
    ],
    fragment: "--param-dtype bf16-sr does not compose with --optim-state-offload",
    control: &[
        "--source-ad",
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
        "--param-dtype",
        "bf16-sr",
    ],
}];

/// Materialize the CSLA fixture into a temp file with its `CSLA_SAVE_PATH`
/// marker rewritten, following the same convention as
/// `csla_layerwise_gate.rs`. Running the fixture verbatim makes `model_save`
/// write a file literally named `CSLA_SAVE_PATH` into the CURRENT DIRECTORY —
/// which for this gate is the repo root.
fn materialize_csla_fixture(tag: &str, gpu: bool) -> PathBuf {
    let root = repo_root();
    let src_path = root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", src_path.display()));
    // Both markers are the fixture's own documented rewrite convention. Assert
    // rather than silently no-op: losing CSLA_SAVE_PATH writes a stray file
    // into the cwd, and losing GPU_PLACEMENT leaves the GPU tier running a
    // CPU-resident model, where `--weight-stream` aborts and every case in
    // that tier fails for the wrong reason.
    // Look for the markers on a NON-COMMENT line. The fixture documents its
    // own rewrite convention in a `#`-comment header that mentions both
    // tokens, so a plain `src.contains(..)` is satisfied by the documentation
    // even after the real marker is deleted — the assertion would pass while
    // `.replace()` silently no-ops. Found in review.
    let live = |needle: &str| {
        src.lines()
            .filter(|l| !l.trim_start().starts_with('#') || l.trim() == "# GPU_PLACEMENT")
            .any(|l| l.contains(needle))
    };
    assert!(
        live("CSLA_SAVE_PATH"),
        "fixture lost its CSLA_SAVE_PATH marker (found only in the comment \
         header, if at all); this gate would write a stray file into the \
         working directory"
    );
    assert!(
        live("# GPU_PLACEMENT"),
        "fixture lost its # GPU_PLACEMENT marker (found only in the comment \
         header, if at all); the GPU tier would silently run on a \
         CPU-resident model"
    );
    // PID-suffixed, matching csla_layerwise_gate's convention: a fixed name in
    // a world-writable dir collides between concurrent `cargo test` runs, and
    // between users on a shared runner.
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let out = tmp.join(format!("nsl_featcomp_csla_{tag}_{pid}.nsl"));
    let save = tmp.join(format!("nsl_featcomp_csla_{tag}_{pid}.nslm"));
    // The path is substituted into an NSL *string literal*, so a raw Windows
    // path breaks it: `save` is `C:\Users\RUNNER~1\AppData\...` and the lexer
    // reads `\U`, `\R`, `\A` … as escape sequences, failing the fixture with
    // "unknown escape sequence" — which this gate reports as `CONTROL FAILED
    // ... the fixture must compile in this SUPPORTED configuration`. Forward
    // slashes carry no escape meaning and Windows accepts them in paths, which
    // is why every other CSLA_SAVE_PATH substitution in this tree normalizes
    // the same way. Never triggered on Unix, where temp paths have no
    // backslashes.
    let save_literal = save.display().to_string().replace('\\', "/");
    let body = src
        .replace("CSLA_SAVE_PATH", &save_literal)
        .replace("# GPU_PLACEMENT", if gpu { "m.to(cuda)" } else { "" });
    std::fs::write(&out, body).expect("write temp fixture");
    out
}

#[test]
fn registered_refusals_actually_fire_on_a_real_compile() {
    let fixture = materialize_csla_fixture("cpu", false);
    run_fire_cases(&fixture, TIER_C);
}

/// Run every control (deduplicated), then every case.
///
/// CONTROLS FIRST, and this is the load-bearing part of the whole tier: if the
/// fixture cannot compile in a case's supported configuration, that case's
/// refusal assertion proves nothing — it would pass on any unrelated error.
/// Each control is run once even when several cases share it.
fn run_fire_cases(fixture: &std::path::Path, cases: &[FireCase]) {
    assert!(!cases.is_empty(), "no cases — this gate asserts nothing");

    let controls: BTreeSet<&[&str]> = cases.iter().map(|c| c.control).collect();
    for control in &controls {
        let (ok, out) = run_nsl(fixture, control);
        assert!(
            ok,
            "CONTROL FAILED for {control:?}: the fixture must compile in this \
             SUPPORTED configuration, or every refusal assertion resting on it \
             proves nothing.\n{out}"
        );
    }

    for case in cases {
        let (ok, out) = run_nsl(fixture, case.flags);
        assert!(
            !ok,
            "{}: must be refused, but the compile SUCCEEDED. args={:?}",
            case.name, case.flags
        );
        assert!(
            out.contains(case.fragment),
            "{}: refused, but NOT for the registered reason {:?} — the compile \
             may have failed for an unrelated cause, which would make this \
             assertion vacuous. args={:?}\n{out}",
            case.name,
            case.fragment,
            case.flags
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU (--weight-stream needs device-resident params)"]
fn registered_refusals_actually_fire_on_gpu_only_configurations() {
    // `--weight-stream` aborts at RUNTIME on a CPU-resident model, so every
    // rule downstream of it has no passing control in the CPU tier — the
    // refusal under test would be indistinguishable from that abort. These
    // cases get a `m.to(cuda)` fixture and a device.
    let fixture = materialize_csla_fixture("gpu", true);
    run_fire_cases(&fixture, TIER_C_GPU);
}

// ──────────────── Tier B2: the registry cannot be silently INCOMPLETE ──────

/// Refusals deliberately outside this registry's scope, with the reason.
/// Anything here is a conscious exclusion, not an oversight — which is the
/// distinction the sweep below exists to force someone to make.
/// Individual files that host composition refusals.
const SWEEP_FILES: &[&str] = &[
    "crates/nsl-codegen/src/stmt.rs",
    "crates/nsl-codegen/src/stmt_fase.rs",
    "crates/nsl-codegen/src/calibration/binary_codegen.rs",
];

/// Directories swept wholesale, so a refusal added to a NEW file is found.
const SWEEP_DIRS: &[&str] = &[
    "crates/nsl-cli/src/commands",
    // Added after review: restricting the codegen side to three named files
    // left `wggo_gradient_scorer.rs`'s `--wggo-importance=grad requires
    // --calibration-data` invisible — the same blind spot one level up.
    "crates/nsl-codegen/src",
];

/// Repo-relative path as a registry key, with `/` separators on every platform.
///
/// The registries in `nsl_cli::exec_markers` spell their paths with forward
/// slashes (`crates/nsl-codegen/src/stmt.rs`). On Windows a scanned path comes
/// back as `crates/nsl-cli/tests\foo.rs` — the literal `join("crates/...")`
/// segment keeps its slashes while the OS appends the file name with a
/// backslash — so a raw `to_string_lossy()` never matches a registered entry.
///
/// Every registered file then looked UNREGISTERED, which is why
/// `every_negative_marker_assertion_in_the_suite_is_pinned` failed on
/// windows-latest listing a dozen files that are in fact pinned. Normalise at
/// the single point where the key is built.
fn repo_rel_key(path: &std::path::Path, root: &std::path::Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Recursively collect `.rs` files under `dir` as repo-relative paths.
fn collect_rs_files(dir: &std::path::Path, out: &mut BTreeSet<String>, root: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out, root);
        } else if path.extension().is_some_and(|e| e == "rs") && path.starts_with(root) {
            out.insert(repo_rel_key(&path, root));
        }
    }
}

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
    // ADVISORY, same reasoning: --cpdt on a block with no WGGO plan now
    // plans weights-only; when arbitration still lowers nothing this notice
    // says so and the compile SUCCEEDS. It replaced the pre-weights-only
    // "CPDT planning requires a WGGO plan" skip notice (itself a replaced
    // silent skip); registering it as a refusal would overstate it.
    (
        "arbitration lowered nothing",
        "advisory not-active notice, compile succeeds",
    ),
    // A STALENESS refusal, not a composition rule: the flags compose fine —
    // what is refused is executing moment dtypes derived from a
    // fingerprint-rejected WGGO pre-plan. A pairwise flag registry cannot
    // express "this artifact went stale mid-compile".
    (
        "refusing to execute a stale precision plan",
        "staleness refusal, not a flag-pair incompatibility",
    ),
    ("warning \u{2014} --wrga-ablate=spectral", "advisory warning, run continues"),
    // ADVISORY, and pointedly so: --calibration-data is ACCEPTED and then
    // ignored, because the harness lives in `compile_and_calibrate` and no CLI
    // path calls it. The build succeeds. Registering it as a refusal would
    // claim an enforcement that does not exist -- the exact overstatement this
    // allowlist protects the registry from. It names --wggo-importance only to
    // say that grad mode stays unavailable.
    (
        "is validated but NOT consumed",
        "advisory warning, build succeeds; the corpus is dropped, not refused",
    ),
    // ADVISORY: the --wggo-importance=auto fallback note. It names
    // --calibration-data only to say that supplying it will NOT lift the note
    // (a sidecar is what is required, and only `compile_and_calibrate` makes
    // one). Compilation continues under magnitude scoring.
    (
        "fell back to magnitude scoring",
        "advisory fallback note, compile succeeds under magnitude scoring",
    ),

    ("--trace is not implemented", "unimplemented-subcommand notice, not a composition"),
    ("cannot determine export format", "diagnostic listing ways to supply a format"),
    ("usage: cpdt_", "usage banner for a dev-tool binary"),
    ("not found at {}. Run cpdt_fixture_generate", "missing-input diagnostic, not a flag composition"),
    // Unreachable-by-construction backstops, not user-facing refusals. clap
    // already conflicts `--fuse-wgrad-accum` with both partners at parse time,
    // and `--pretrain-optimized` (which sets the flag AFTER parsing, where clap
    // cannot enforce anything) suppresses it for exactly these — see
    // `WgradFusionBlockers`, gated by `both_dispatchers_pass_the_same_wgrad_blockers`.
    // These messages fire only if that enforcement develops a gap; registering
    // them as composition rules would claim a refusal a user can actually reach.
    (
        "--fuse-wgrad-accum reached lowering with --grad-integrity",
        "internal backstop for a state the CLI makes unreachable",
    ),
    (
        "--fuse-wgrad-accum reached lowering with --optim-state-offload",
        "internal backstop for a state the CLI makes unreachable",
    ),
    // ADVISORY, not a refusal: `--pretrain-optimized` sets --fuse-wgrad-accum
    // on programs that never asked for it, so a program whose train blocks all
    // lack `grad_accumulation` gets this note and the compile SUCCEEDS. The
    // TYPED form of the same precondition IS a refusal and IS registered
    // (`--fuse-wgrad-accum requires grad_accumulation >= 2 ...` in
    // feature_rules.rs); registering the advisory too would claim the bundle
    // refuses, which would be exactly backwards.
    //
    // "in this compile", not "on this train block": the note moved with the
    // refusal when both were rescoped from per-block to per-compile, so that a
    // program in which SOME block fuses is neither refused nor warned about.
    (
        "--fuse-wgrad-accum fused nothing in this compile",
        "advisory bundle-partially-disabled note, compile succeeds",
    ),
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
    // Derive the search set from a FIXED list, not from the files the registry
    // already cites. Deriving it from the registry is a bootstrapping blind
    // spot: a refusal living in a file with no entry yet is invisible to the
    // sweep whose whole job is to find exactly that. Five real rules
    // (--gpu x --perf, the cpkd/wrga ones) hid there until this changed.
    let mut files: BTreeSet<String> = SWEEP_FILES.iter().map(|f| f.to_string()).collect();
    for dir in SWEEP_DIRS {
        collect_rs_files(&root.join(dir), &mut files, &root);
    }

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

// ─────────────────── Tier D: the execution-marker vocabulary ───────────────

#[test]
fn every_exec_marker_is_still_emitted_by_its_source() {
    // Why this matters more than it looks: a POSITIVE assertion
    // (`stderr.contains("[csla]")`) fails loudly when a marker is renamed, so
    // it needs no help. A NEGATIVE one — asserting a feature did NOT engage —
    // passes forever once the token stops existing, which is exactly backwards
    // and completely silent. This pins that every registered token still has
    // an emitting call site.
    let root = repo_root();
    let mut cache: BTreeMap<&'static str, String> = BTreeMap::new();
    let mut lost = Vec::new();

    for marker in nsl_cli::exec_markers::EXEC_MARKERS {
        let found = marker.emitted_by.iter().any(|file| {
            cache
                .entry(file)
                .or_insert_with(|| anchored_text(&root.join(file), PRINT_MARKERS))
                .contains(marker.token)
        });
        if !found {
            lost.push(format!(
                "  {} — not emitted by any print macro in {:?} ({})",
                marker.token, marker.emitted_by, marker.means
            ));
        }
    }
    assert!(
        lost.is_empty(),
        "{} execution marker(s) are registered but no longer emitted. Any \
         NEGATIVE assertion on these is now vacuous — it can never fail. \
         Update crates/nsl-cli/src/exec_markers.rs, or restore the emit:\n{}",
        lost.len(),
        lost.join("\n")
    );
}

#[test]
fn exec_marker_tokens_are_well_formed_and_unique() {
    // ANTI-VACUITY for the gate above: an empty or `[`-less token would match
    // trivially, and a duplicate would inflate the coverage it appears to give.
    let mut seen = BTreeSet::new();
    for marker in nsl_cli::exec_markers::EXEC_MARKERS {
        assert!(
            marker.token.starts_with('[') && marker.token.ends_with(']') && marker.token.len() > 3,
            "malformed marker token {:?}",
            marker.token
        );
        assert!(
            seen.insert(marker.token),
            "duplicate marker token {:?}",
            marker.token
        );
    }
    assert!(
        seen.len() >= 23,
        "only {} markers registered; the vocabulary has shrunk unexpectedly",
        seen.len()
    );
}

#[test]
fn every_marker_asserted_in_the_test_suite_is_registered() {
    // The sweep, mirroring the refusal one: a marker a test asserts on but the
    // vocabulary does not know about is a token nobody is watching. Finding
    // them is the whole reason to have a vocabulary rather than 42 literals.
    let root = repo_root();
    let dir = root.join("crates/nsl-cli/tests");
    let known: BTreeSet<&str> = nsl_cli::exec_markers::EXEC_MARKERS
        .iter()
        .map(|m| m.token)
        .collect();
    // Tokens that are test-local scaffolding or third-party output, not
    // subsystem markers emitted by NSL.
    // Tokens a test may mention without them needing a vocabulary entry.
    // NOT "third-party output" — an earlier comment claimed that and was
    // wrong; these are NSL markers whose engagement no test asserts on, so
    // there is nothing for the emission gate to protect. Seven further
    // entries were removed after review because they suppressed nothing: an
    // allowlist of no-ops reads as considered exclusions and is not.
    const NOT_SUBSYSTEM_MARKERS: &[&str] = &[
        "[nsl]", // generic CLI prefix, not a subsystem engagement marker
    ];

    let mut unknown: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entry in std::fs::read_dir(&dir).expect("read tests dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap_or_default();
        // Tests read markers several ways, not just `contains("[`. Missing a
        // form is a silent coverage hole: `[cuda-graph]` went unregistered
        // because cuda_graph_gate uses `starts_with(`. Note the irony that
        // migrating an assertion to a constant (`format!("{} csha[", CSHA)`)
        // also removes it from this sweep's view — which is why
        // NEGATIVE_NEEDLES pins those separately.
        const READ_FORMS: &[&str] = &[
            "contains(\"",
            "starts_with(\"",
            "strip_prefix(\"",
            "trim_start_matches(\"",
            "splitn(2, \"",
            "split(\"",
        ];
        let mut caps: Vec<&str> = Vec::new();
        for form in READ_FORMS {
            caps.extend(src.split(form).skip(1));
        }
        for cap in caps {
            let Some(tok) = cap.split('"').next() else { continue };
            if !tok.starts_with('[') {
                continue;
            }
            let Some(close) = tok.find(']') else { continue };
            let token = &tok[..=close];
            // A marker is `[lowercase-with-hyphens]`. Without this, a shape
            // assertion like `contains("[8, 2048, 512]")` reads as a marker.
            let body = &token[1..close];
            let is_marker = !body.is_empty()
                && body.chars().next().is_some_and(|c| c.is_ascii_lowercase())
                && body
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
            if !is_marker || known.contains(token) || NOT_SUBSYSTEM_MARKERS.contains(&token) {
                continue;
            }
            unknown
                .entry(token.to_string())
                .or_default()
                .insert(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    assert!(
        unknown.is_empty(),
        "test(s) assert on execution marker(s) absent from \
         crates/nsl-cli/src/exec_markers.rs. Register each, or add it to \
         NOT_SUBSYSTEM_MARKERS with a reason:\n{}",
        unknown
            .iter()
            .map(|(t, files)| format!("  {t} — asserted in {files:?}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn every_negative_assertion_needle_is_still_emitted() {
    // The gate that `every_exec_marker_is_still_emitted_by_its_source` was not.
    //
    // Pinning a TOKEN says "[csha] appears somewhere". That is satisfied by an
    // unrelated `[csha] gpu spec:` probe while the string a negative assertion
    // actually depends on — `[csha] csha[`, assembled from a prefix in stmt.rs
    // and a summary in csha.rs — quietly stops existing. A negative assertion
    // that can no longer fail is worse than no assertion, because it reads as
    // coverage. Each needle is pinned part by part, in the file that emits it.
    let root = repo_root();
    let mut cache: BTreeMap<&'static str, String> = BTreeMap::new();
    let mut lost = Vec::new();
    let mut parts_checked = 0usize;

    for needle in nsl_cli::exec_markers::NEGATIVE_NEEDLES {
        for (literal, file) in needle.parts {
            parts_checked += 1;
            let hay = cache
                .entry(file)
                .or_insert_with(|| anchored_text(&root.join(file), PRINT_MARKERS));
            if !hay.contains(literal) {
                lost.push(format!(
                    "  {literal:?} no longer emitted by {file}\n      breaks: {} ({})",
                    needle.test, needle.asserts
                ));
            }
        }
    }
    assert!(
        lost.is_empty(),
        "{} negative-assertion needle(s) rotted. The assertions resting on \
         them can NO LONGER FAIL — they now pass unconditionally while still \
         reading as coverage:\n{}",
        lost.len(),
        lost.join("\n")
    );
    assert!(
        parts_checked >= 5,
        "only {parts_checked} needle parts checked; NEGATIVE_NEEDLES has shrunk"
    );
}

#[test]
fn every_negative_marker_assertion_in_the_suite_is_pinned() {
    // Completeness for the gate above: a negative assertion nobody registered
    // is a silently-unfailable test. Sweeps for `!<expr>.contains("[...")`.
    let root = repo_root();
    let registered: BTreeSet<&str> = nsl_cli::exec_markers::NEGATIVE_NEEDLES
        .iter()
        .map(|n| n.test)
        .collect();
    let mut unpinned = Vec::new();
    for entry in std::fs::read_dir(root.join("crates/nsl-cli/tests")).expect("read tests dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let rel = repo_rel_key(&path, &root);
        if registered.contains(rel.as_str()) {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap_or_default();
        for (i, line) in src.lines().enumerate() {
            let t = line.trim();
            // `!x.contains("[` — the negative form. The positive form fails
            // loudly on rot and needs no pin.
            //
            // The negation must be tested against the RECEIVER, not merely
            // "is there a `!` earlier on the line": `assert!(out.contains(..))`
            // is positive but has a `!` in `assert!`. Walk back over the
            // receiver expression and look at the character before it.
            let Some(pos) = t.find(".contains(\"[") else {
                continue;
            };
            if t.trim_start().starts_with("//") {
                continue;
            }
            let before = &t[..pos];
            let recv_start = before
                .rfind(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '&'))
                .map_or(0, |i| i + 1);
            let negated = before[..recv_start].trim_end().ends_with('!');
            if !negated {
                continue;
            }
            // Only bracketed lowercase tokens are subsystem markers; a shape
            // assertion like `contains("[8, 2048, 512]")` is not one.
            let after = &t[pos + ".contains(\"".len()..];
            let is_marker = after
                .split(']')
                .next()
                .and_then(|b| b.strip_prefix('['))
                .is_some_and(|b| {
                    !b.is_empty()
                        && b.chars().next().is_some_and(|c| c.is_ascii_lowercase())
                        && b.chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                });
            if is_marker {
                unpinned.push(format!("  {rel}:{}  {}", i + 1, t.trim()));
            }
        }
    }
    assert!(
        unpinned.is_empty(),
        "negative marker assertion(s) not registered in \
         exec_markers::NEGATIVE_NEEDLES. Each can silently become \
         always-true:\n{}",
        unpinned.join("\n")
    );
}

/// `#[command(group(...))]` declarations must match between the two structs.
///
/// `parse_arg_struct` walks `#[arg(...)]` attributes only, so a struct-level
/// `#[command(group(...))]` is invisible to the drift gate above. That matters
/// because clap resolves `requires = "<id>"` at RUNTIME: it looks the id up as
/// an arg, then as a group, and if NEITHER matches it drops the requirement
/// silently. So losing the group line from one struct would not error — it
/// would make `nsl build --fuse-wgrad-accum m.nsl` (no source-AD anywhere)
/// quietly accepted, with the fusion then inert. clap's own debug asserts catch
/// this in a debug build; nothing catches it in release.
#[test]
fn run_and_build_declare_identical_arg_groups() {
    let src = args_src();

    fn groups_before(src: &str, struct_name: &str) -> Vec<String> {
        let lines: Vec<&str> = src.lines().collect();
        let at = lines
            .iter()
            .position(|l| l.contains(&format!("struct {struct_name} {{")))
            .unwrap_or_else(|| panic!("{struct_name} not found in args.rs"));
        // Attributes sit immediately above the struct, after `#[derive(...)]`.
        let mut found = Vec::new();
        for l in lines[..at].iter().rev() {
            let t = l.trim();
            if t.starts_with("#[command(group(") {
                found.push(t.to_string());
            } else if !t.starts_with("#[") && !t.starts_with("///") && !t.is_empty() {
                break;
            }
        }
        found.sort();
        found
    }

    let build = groups_before(&src, "BuildArgs");
    let run = groups_before(&src, "RunArgs");
    assert_eq!(
        build, run,
        "`nsl run` and `nsl build` declare different #[command(group(...))] sets. \
         A group present on one and not the other makes a `requires = \"<group>\"` \
         silently unenforced on the other in release builds."
    );
    assert!(
        !build.is_empty(),
        "expected at least the source_ad_mode group; if groups moved out of \
         args.rs this gate has gone vacuous"
    );
}

/// The `--pretrain-optimized` blocker list must be identical at both dispatchers.
///
/// `meta_flags::expand_pretrain_optimized` is a shared helper precisely so `nsl
/// run` and `nsl build` cannot drift, but its `WgradFusionBlockers` argument is
/// constructed separately at each call site. A blocker added to one only would
/// let `nsl build --pretrain-optimized --<blocker>` keep a fusion that `nsl run`
/// drops — with both subcommands' own tests still green.
#[test]
fn both_dispatchers_pass_the_same_wgrad_blockers() {
    fn blocker_fields(path: &str) -> Vec<String> {
        let src = std::fs::read_to_string(repo_root().join(path))
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        let at = src
            .find("WgradFusionBlockers {")
            .unwrap_or_else(|| panic!("{path} does not construct WgradFusionBlockers"));
        let body = &src[at + "WgradFusionBlockers {".len()..];
        let end = body.find('}').expect("unterminated struct literal");
        let mut fields: Vec<String> = body[..end]
            .split(',')
            .map(|f| f.trim().trim_end_matches(':').trim().to_string())
            .filter(|f| !f.is_empty())
            .collect();
        fields.sort();
        fields
    }

    let run = blocker_fields("crates/nsl-cli/src/commands/run.rs");
    let build = blocker_fields("crates/nsl-cli/src/commands/build/options.rs");
    assert_eq!(
        run, build,
        "nsl run and nsl build pass different --pretrain-optimized blocker sets"
    );
    // Anti-vacuity + completeness: these are exactly `--fuse-wgrad-accum`'s
    // clap conflicts, which the bundle can no longer rely on clap to enforce.
    // CLAP-enforced conflicts specifically: those are the ones the bundle
    // disarms by setting the flag after parsing. The two source-level refusals
    // are separately safe — the CSLA window replay requires `--layerwise-accum`
    // (already in this list) and calibration-binary emission compiles with
    // `CompileOptions::default()`, so neither is reachable through the bundle.
    let expected: Vec<String> = clap_rules()
        .filter(|r| r.flag == "--fuse-wgrad-accum" && matches!(r.kind, RuleKind::Conflicts))
        .map(|r| r.other.to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    assert_eq!(
        run, expected,
        "the bundle's blocker list must equal --fuse-wgrad-accum's clap-enforced \
         conflicts: the bundle sets the flag after clap has validated, so this \
         list is the only enforcement left"
    );
}
