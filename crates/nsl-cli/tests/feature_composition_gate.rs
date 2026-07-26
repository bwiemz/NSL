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
    fn collect(attr: &str, keys: &[&str]) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for key in keys {
            let mut rest = attr;
            while let Some(pos) = rest.find(key) {
                let after = &rest[pos + key.len()..];
                let after = after.trim_start();
                // Skip past `=` then read either "lit" or [ "a", "b" ].
                let Some(after) = after.strip_prefix('=') else {
                    rest = &rest[pos + key.len()..];
                    continue;
                };
                let after = after.trim_start();
                if let Some(list) = after.strip_prefix('[') {
                    if let Some(close) = list.find(']') {
                        for tok in list[..close].split(',') {
                            let t = tok.trim().trim_matches('"').trim();
                            if !t.is_empty() {
                                out.insert(t.to_string());
                            }
                        }
                    }
                } else if let Some(lit) = after.strip_prefix('"') {
                    if let Some(close) = lit.find('"') {
                        out.insert(lit[..close].to_string());
                    }
                }
                rest = &rest[pos + key.len()..];
            }
        }
        out
    }
    // Order matters: match the `_all` forms first so the plain form's `find`
    // does not consume `requires_all=` as `requires` + junk.
    let requires = collect(attr, &["requires_all", "requires"]);
    let conflicts = collect(attr, &["conflicts_with_all", "conflicts_with"]);
    (requires, conflicts)
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
        shared_constrained >= 10,
        "parsed only {shared_constrained} constrained shared flags — the \
         args.rs parser has probably stopped matching, making this gate vacuous"
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
    assert!(
        in_registry.len() >= 12,
        "only {} clap rules registered; the parser or the registry has \
         collapsed and this gate is vacuous",
        in_registry.len()
    );
}

// ─────────────────────── Tier B: the refusal still exists ─────────────────

/// Rust string literals in these files wrap with `\` continuations and
/// indentation, so a fragment like `--layerwise-accum requires --source-ad`
/// is not contiguous in the raw bytes. Collapse all whitespace runs (and drop
/// the backslash-newline pairs) before searching.
fn normalized(path: &std::path::Path) -> String {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let unwrapped = raw.replace("\\\n", " ");
    unwrapped.split_whitespace().collect::<Vec<_>>().join(" ")
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
            .or_insert_with(|| normalized(&root.join(file)));
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
        if fragment.len() < 24 || !fragment.contains("--") {
            weak.push(format!("  {} -> {fragment:?}", rule.flag));
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
    let out = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["--", "run"])
        .args(extra)
        .arg(fixture)
        .current_dir(&root)
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
    let tmp = std::env::temp_dir();
    let out = tmp.join("nsl_feature_composition_csla.nsl");
    let save = tmp.join("nsl_feature_composition_csla.nslm");
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
