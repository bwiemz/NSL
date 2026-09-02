//! `nsl-env` — the registry of every `NSL_*` environment variable the
//! toolchain reads.
//!
//! # The problem this closes
//!
//! The compiler and runtime read ~150 `NSL_*` variables at ~300 sites, each
//! a bare `std::env::var("NSL_…")` with its own parsing and its own default,
//! documented (when at all) in a comment beside the read. Nothing listed
//! them; nothing distinguished a knob that changes the numerics of a
//! compiled program from one that prints a counter; and a run's environment
//! was never part of its record, so two runs that differed only in an
//! exported variable were indistinguishable after the fact.
//!
//! This crate is the first increment of roadmap item A5: **one declarative
//! table** ([`REGISTRY`]) naming each variable with its type, accepted
//! values, default, [`Tier`] and a one-line doc, plus a source-scanning
//! agreement gate (`tests/registry_agreement.rs`) that fails when a variable
//! is read but not registered, or registered but no longer read. `nsl env
//! list` renders the table; `nsl env current` reports which registered
//! variables are set in the calling environment and flags any `NSL_*` that
//! is set but unknown. The read sites themselves are unchanged — folding
//! them into `CompileOptions`/`RuntimeOptions` with the environment as an
//! override layer is the next increment, and this table is its input.
//!
//! Like `nsl-abi`, the crate is dependency-free and the gate parses source
//! *text*, so it stays a fast standalone check. It covers the Rust sources
//! under `crates/` (plus each crate's `build.rs`); variables that only a
//! shell script or CI workflow reads — `scripts/gpu-guard.sh`'s
//! `NSL_GPU_GUARD_*`, say — are that script's contract, not the toolchain's,
//! and are not registered here.

pub mod registry;
pub mod scan;

pub use registry::REGISTRY;

/// How a variable's value is parsed at its read site(s).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A switch. `accepted` says which spellings count as "on" — most sites
    /// compare against `"1"`, some accept any non-empty value.
    Bool,
    /// An integer (bytes, MiB, counts, seconds — the doc names the unit).
    Int,
    /// A floating-point number.
    Float,
    /// Free text.
    Str,
    /// A filesystem path.
    Path,
    /// One of a fixed set of spellings, listed in `accepted`.
    Enum,
    /// A delimited list; `accepted` names the separator and element form.
    List,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Bool => "bool",
            Kind::Int => "int",
            Kind::Float => "float",
            Kind::Str => "string",
            Kind::Path => "path",
            Kind::Enum => "enum",
            Kind::List => "list",
        }
    }
}

/// What a variable can change. The tier is the reproducibility contract: a
/// `Behavior` variable must be recorded with any result it influenced; a
/// `Diagnostic` one may not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Changes the compiled program's semantics or numerics, or what is
    /// executed: precision, algorithm and kernel choices that change bits,
    /// fallbacks, distributed roles.
    Behavior,
    /// Changes speed or memory only; numerics-preserving by the code's own
    /// claim.
    Perf,
    /// A gate, refusal, or allow-override: `NSL_ALLOW_*`, `*_FAULT`,
    /// `*_FORCE_STALE_*`. Off by default; turning one on bypasses or
    /// injects a check.
    Safety,
    /// Paths, devices, library locations, distributed topology.
    Platform,
    /// Prints, dumps, counters, traces and reports; no effect on results.
    Diagnostic,
    /// Only meaningful under the test harness.
    Test,
}

impl Tier {
    pub const ALL: [Tier; 6] = [
        Tier::Behavior,
        Tier::Perf,
        Tier::Safety,
        Tier::Platform,
        Tier::Diagnostic,
        Tier::Test,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Behavior => "behavior",
            Tier::Perf => "perf",
            Tier::Safety => "safety",
            Tier::Platform => "platform",
            Tier::Diagnostic => "diagnostic",
            Tier::Test => "test",
        }
    }

    /// Parse a tier name as `nsl env list --tier <name>` spells it.
    pub fn parse(s: &str) -> Option<Tier> {
        Tier::ALL.into_iter().find(|t| t.as_str() == s)
    }
}

/// Which process reads the variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadAt {
    /// The compiler process (`nsl build` and friends).
    Compile,
    /// `libnsl_runtime`, inside the compiled program (or `nsl run`).
    Runtime,
    Both,
    /// Only a `cargo test` binary (a re-exec'd child probe, a GPU-suite
    /// skip switch); no shipped process reads it.
    Test,
}

impl ReadAt {
    pub fn as_str(self) -> &'static str {
        match self {
            ReadAt::Compile => "compile",
            ReadAt::Runtime => "runtime",
            ReadAt::Both => "both",
            ReadAt::Test => "test",
        }
    }
}

/// One registered variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnvVar {
    pub name: &'static str,
    pub kind: Kind,
    /// The values the read site accepts, as the code spells them (`1/0`,
    /// `sr|rne`, `MiB integer`, `path`).
    pub accepted: &'static str,
    /// Behaviour when unset.
    pub default: &'static str,
    pub tier: Tier,
    pub read_at: ReadAt,
    /// One sentence, for `nsl env list`.
    pub doc: &'static str,
}

/// Find a registered variable by exact name.
pub fn lookup(name: &str) -> Option<&'static EnvVar> {
    REGISTRY.iter().find(|v| v.name == name)
}

/// The registry filtered to one tier, in registry order.
pub fn by_tier(tier: Tier) -> impl Iterator<Item = &'static EnvVar> {
    REGISTRY.iter().filter(move |v| v.tier == tier)
}

/// A registered variable that is set in the calling environment, with its
/// value, or an `NSL_*` variable that is set but not registered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Current {
    Registered { var: &'static EnvVar, value: String },
    Unregistered { name: String, value: String },
}

/// Every `NSL_*` variable set in the calling environment, registered ones
/// first (in registry order), then unknown ones (sorted). Values that are
/// not valid Unicode are rendered lossily.
pub fn current() -> Vec<Current> {
    let mut set: Vec<(String, String)> = std::env::vars_os()
        .filter_map(|(k, v)| {
            let k = k.to_string_lossy().into_owned();
            k.get(..4)
                .is_some_and(|prefix| names_match(prefix, "NSL_"))
                .then(|| (k, v.to_string_lossy().into_owned()))
        })
        .collect();
    set.sort();
    let mut out = Vec::new();
    for var in REGISTRY {
        if let Some((_, value)) = set.iter().find(|(k, _)| names_match(k, var.name)) {
            out.push(Current::Registered { var, value: value.clone() });
        }
    }
    for (name, value) in set {
        if !REGISTRY.iter().any(|v| names_match(&name, v.name)) {
            out.push(Current::Unregistered { name, value });
        }
    }
    out
}

/// Environment-variable name equality as the platform sees it: Windows
/// resolves `env::var("NSL_X")` from an exported `nsl_x`, so a
/// case-insensitive match is the one that answers "does this process honor
/// it?"; everywhere else names are exact.
fn names_match(set: &str, registered: &str) -> bool {
    if cfg!(windows) {
        set.eq_ignore_ascii_case(registered)
    } else {
        set == registered
    }
}

/// The plain-text table `nsl env list` prints, grouped by tier in
/// [`Tier::ALL`] order under a `[tier]` heading: `NAME  kind  read_at
/// default: …` and the doc on the next line.
pub fn render_table(vars: &[&EnvVar]) -> String {
    let name_w = vars.iter().map(|v| v.name.len()).max().unwrap_or(4).max(4);
    let mut out = String::new();
    for tier in Tier::ALL {
        let group: Vec<&&EnvVar> = vars.iter().filter(|v| v.tier == tier).collect();
        if group.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("[{}]\n", tier.as_str()));
        for v in group {
            out.push_str(&format!(
                "  {:<name_w$}  {:<6}  {:<8}  default: {}\n      {}\n",
                v.name,
                v.kind.as_str(),
                v.read_at.as_str(),
                v.default,
                v.doc,
            ));
        }
    }
    out
}

/// The Markdown page `nsl env list --markdown` prints (and
/// `docs/wiki/Environment-Variables.md` is generated from — the agreement
/// gate keeps the two identical).
pub fn render_markdown() -> String {
    let mut out = String::new();
    out.push_str("# Environment variables\n\n");
    out.push_str(
        "<!-- GENERATED by `nsl env list --markdown` from crates/nsl-env/src/registry.rs. \
         Do not edit by hand: `cargo test -p nsl-env` fails if this page and the registry differ. -->\n\n",
    );
    out.push_str(&format!(
        "Every `NSL_*` variable the toolchain reads ({} of them), from the registry in \
         `crates/nsl-env`. `nsl env list` prints the same table; `nsl env current` shows \
         which are set in your shell. A **behavior** variable changes the semantics or \
         numerics of what is compiled or executed and belongs in any record of a result; \
         **perf** changes speed or memory only; **safety** variables bypass or inject a \
         check; **platform** names paths, devices and topology; **diagnostic** ones print, \
         dump or count; **test** ones only matter under the test harness. \"Read at\" says \
         whether the compiler process (`compile`) or the runtime inside the compiled program \
         (`runtime`) reads it; `test` means only a `cargo test` binary does. \"Accepted\" \
         quotes what the code compares against: `1 only` means any other value, including \
         `0`, is ignored.\n\n",
        REGISTRY.len()
    ));
    for tier in Tier::ALL {
        let group: Vec<&EnvVar> = by_tier(tier).collect();
        if group.is_empty() {
            continue;
        }
        out.push_str(&format!("## {} ({})\n\n", tier.as_str(), group.len()));
        out.push_str("| Variable | Kind | Accepted | Default | Read at | What it does |\n");
        out.push_str("|---|---|---|---|---|---|\n");
        for v in group {
            out.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {} |\n",
                v.name,
                v.kind.as_str(),
                md_cell(v.accepted),
                md_cell(v.default),
                v.read_at.as_str(),
                md_cell(v.doc),
            ));
        }
        out.push('\n');
    }
    out
}

/// A JSON array, one object per variable, for `nsl env list --json`.
/// Hand-rolled (the crate is dependency-free); every string is escaped.
pub fn render_json(vars: &[&EnvVar]) -> String {
    let mut out = String::from("[\n");
    for (i, v) in vars.iter().enumerate() {
        out.push_str(&format!(
            "  {{\"name\": {}, \"kind\": {}, \"accepted\": {}, \"default\": {}, \"tier\": {}, \"read_at\": {}, \"doc\": {}}}{}\n",
            json_str(v.name),
            json_str(v.kind.as_str()),
            json_str(v.accepted),
            json_str(v.default),
            json_str(v.tier.as_str()),
            json_str(v.read_at.as_str()),
            json_str(v.doc),
            if i + 1 < vars.len() { "," } else { "" },
        ));
    }
    out.push_str("]\n");
    out
}

/// A table cell: `|` would end the cell, and `<exe>`-style placeholders
/// would be swallowed by a GFM renderer as inline HTML tags.
fn md_cell(s: &str) -> String {
    s.replace('|', "\\|").replace('<', "&lt;").replace('>', "&gt;")
}

/// A JSON string literal for `s`, escaped (shared with `nsl env`'s own
/// renderings so the crate stays dependency-free).
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_are_unique_sorted_and_nsl_prefixed() {
        let names: Vec<&str> = REGISTRY.iter().map(|v| v.name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(names, sorted, "registry must be sorted by name with no duplicates");
        for v in REGISTRY {
            assert!(v.name.starts_with("NSL_"), "{}: not an NSL_ variable", v.name);
            assert!(
                v.name.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'),
                "{}: not SCREAMING_SNAKE",
                v.name
            );
            assert!(!v.doc.is_empty() && !v.default.is_empty() && !v.accepted.is_empty(), "{}: empty field", v.name);
            assert!(v.doc.len() <= 160, "{}: doc is {} chars, keep it to one line", v.name, v.doc.len());
            assert!(!v.doc.contains('\n'), "{}: doc has a newline", v.name);
        }
    }

    #[test]
    fn lookup_and_by_tier_agree_with_the_table() {
        let first = REGISTRY[0];
        assert_eq!(lookup(first.name).map(|v| v.name), Some(first.name));
        assert!(lookup("NSL_NOT_A_VARIABLE").is_none());
        let counted: usize = Tier::ALL.iter().map(|t| by_tier(*t).count()).sum();
        assert_eq!(counted, REGISTRY.len());
    }

    #[test]
    fn tier_parse_round_trips() {
        for t in Tier::ALL {
            assert_eq!(Tier::parse(t.as_str()), Some(t));
        }
        assert_eq!(Tier::parse("bogus"), None);
    }

    #[test]
    fn json_strings_are_escaped() {
        assert_eq!(json_str("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
        assert_eq!(json_str("\u{1}"), "\"\\u0001\"");
    }

    #[test]
    fn markdown_cells_escape_pipes() {
        assert_eq!(md_cell("sr|rne"), "sr\\|rne");
    }

    #[test]
    fn renders_cover_every_variable() {
        let all: Vec<&EnvVar> = REGISTRY.iter().collect();
        let table = render_table(&all);
        let md = render_markdown();
        let json = render_json(&all);
        for v in REGISTRY {
            assert!(table.contains(v.name), "table lacks {}", v.name);
            assert!(md.contains(&format!("`{}`", v.name)), "markdown lacks {}", v.name);
            assert!(json.contains(&format!("\"name\": \"{}\"", v.name)), "json lacks {}", v.name);
        }
    }
}
