//! Roadmap item 21 — the claims in [`nsl_codegen::pass_registry::PASSES`] are
//! checked against the tree, in both directions.
//!
//! "Both directions" is meant literally and is the whole point: it is easy to
//! check that everything the registry claims is true, and that check is worth
//! little on its own, because a registry that claims nothing passes it. The
//! gates that carry the weight are the ones running tree -> registry:
//! `every_wiki_pass_section_is_registered`,
//! `every_pass_prefixed_cli_flag_is_registered`,
//! `every_codegen_pass_module_belongs_to_a_registered_pass`, and
//! `shell_pass_table_agrees_with_the_registry` (which checks both ways).
//!
//! # The failure this prevents
//!
//! `docs/wiki/Optimization-Passes.md` described CCR as "Common-kernel
//! Combination Rewriting ... no implementation file exists" for ~3 months
//! while `ccr.rs` was 2000+ lines and driven by `--checkpoint-blocks`. The
//! expansion of the acronym was wrong too. The page was simply older than the
//! implementation, and no mechanism existed that could notice.
//!
//! `scripts/check-doc-agreement.sh` already catches part of that class (broken
//! links, "unimplemented" claims about existing files, benchmark scope, CI
//! commands). What it cannot catch is INCOMPLETENESS — its `PASS_SOURCES`
//! array is itself hand-maintained, so a pass nobody added to it is invisible
//! to every check it performs. That is the same defect this repo's
//! feature-composition registry (roadmap item 20) was built to fix, and the
//! same fix applies: gate the registry's own completeness, and gate the two
//! tables against each other so they cannot disagree.
//!
//! # What each gate would have caught
//!
//! | gate | drift it catches |
//! |---|---|
//! | `every_registered_source_file_exists` | the original CCR drift, and any pass whose modules are renamed or merged |
//! | `every_full_name_appears_in_the_passs_own_source` | a WRONG acronym expansion (CCR's was wrong; WRGA's still was) |
//! | `every_documented_pass_has_its_wiki_section` | a section retitled or deleted |
//! | `every_wiki_pass_section_is_registered` | a pass documented but never registered — the incompleteness direction |
//! | `every_registered_cli_flag_is_on_exactly_its_declared_subcommands` | a flag added to `nsl build` but not `nsl run`, or silently gaining a subcommand |
//! | `every_decorator_trigger_is_recognised_somewhere` | a decorator renamed in the parser but not in the docs |
//! | `every_pass_prefixed_cli_flag_is_registered` | a pass flag ADDED to args.rs but never registered |
//! | `shell_pass_table_agrees_with_the_registry` | the two tables drifting apart, in either direction |
//! | `every_codegen_pass_module_belongs_to_a_registered_pass` | a WHOLE NEW PASS landing unregistered |
//!
//! # What is NOT asserted
//!
//! That the prose is *correct* — only that the structural claims (files exist,
//! sections exist, flags resolve, names match) hold. A wiki section can still
//! describe a pass wrongly and pass every gate here. Prose accuracy is not
//! mechanically checkable; what is checkable is that the page and the tree
//! still refer to the same things, which is where the observed drift lived.
//!
//! Two further limits, stated because they would otherwise read as covered:
//! `stage` is checked only for internal plausibility, not against where the
//! pass is actually invoked from; and the flag gate matches FIELD names, so a
//! `#[arg(long = "…")]` rename would slip it (no such override exists in
//! `args.rs` today).

use nsl_codegen::pass_registry::{
    PassDescriptor, PipelineStage, Subcommand, WikiCoverage, PASSES,
};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // crates/nsl-codegen -> crates -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("manifest dir must be two levels below the repo root")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let p = repo_root().join(rel);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// Strip `//` line comments, respecting string literals.
///
/// Without this, a flag named only inside a comment satisfies the arg-block
/// check — the exact hole that made a message-text gate vacuous in the
/// feature-composition registry (a deleted guard stayed green with its message
/// left in a comment).
fn strip_line_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let bytes = line.as_bytes();
        let mut in_str = false;
        let mut escaped = false;
        let mut cut = line.len();
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i];
            if escaped {
                escaped = false;
            } else if c == b'\\' && in_str {
                escaped = true;
            } else if c == b'"' {
                in_str = !in_str;
            } else if !in_str && c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                cut = i;
                break;
            }
            i += 1;
        }
        out.push_str(&line[..cut]);
        out.push('\n');
    }
    out
}

// ─── 1. Source files ───────────────────────────────────────────────────────

/// Every `source_files` path must exist. This alone would have caught the CCR
/// drift: the doc claimed no implementation file existed, and a registry entry
/// naming `ccr.rs` would have proved otherwise the moment it was written.
#[test]
fn every_registered_source_file_exists() {
    let root = repo_root();
    let mut missing = Vec::new();
    let mut checked = 0usize;
    for p in PASSES {
        for f in p.source_files {
            checked += 1;
            if !root.join(f).is_file() {
                missing.push(format!("{} -> {f}", p.name));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "registered source files do not exist:\n  {}",
        missing.join("\n  ")
    );
    // Anti-vacuity: an empty or near-empty registry would satisfy the loop.
    assert!(
        checked >= 50,
        "only {checked} source files checked across {} passes — the registry \
         looks truncated",
        PASSES.len()
    );
}

/// No pass may register zero source files. An entry with an empty list passes
/// the existence check trivially while claiming coverage it does not have.
#[test]
fn no_pass_registers_an_empty_source_list() {
    for p in PASSES {
        assert!(
            !p.source_files.is_empty(),
            "{} registers no source files, so nothing about it is checked",
            p.name
        );
    }
}

/// The expanded name must appear verbatim in one of the pass's own source
/// files.
///
/// This is the gate that catches the CCR class of drift at its root. The wiki
/// page called CCR "Common-kernel Combination Rewriting"; it also called WRGA
/// "Wengert-Ranked Gradient Adapters" while `wrga.rs`, `nsl-ast/src/block.rs`
/// AND `docs/wiki/Glossary.md` all say "Wengert-Pruned Roofline-Guided
/// Adaptation" — a contradiction between two pages of the same wiki that
/// nothing detected. The FIRST version of this registry copied the wrong
/// expansion straight off the wiki for four passes, which is exactly how the
/// error propagates.
///
/// Pinning the expansion to the implementation makes the code the source of
/// truth, because that is the copy a reader is least likely to trust blindly
/// and most likely to correct.
#[test]
fn every_full_name_appears_in_the_passs_own_source() {
    let mut wrong = Vec::new();
    for p in PASSES {
        let found = p.source_files.iter().any(|f| {
            std::fs::read_to_string(repo_root().join(f))
                .map(|src| src.contains(p.full_name))
                .unwrap_or(false)
        });
        if !found {
            wrong.push(format!(
                "{}: '{}' appears in none of {:?}",
                p.name, p.full_name, p.source_files
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "registered expansions do not appear in the implementation:\n  {}\n\
         The code is authoritative here — fix the registry (and any doc that \
         copied it), not the source.",
        wrong.join("\n  ")
    );
}

/// The wiki heading must agree with the registered expansion, so the page and
/// the code cannot say different things about the same acronym.
#[test]
fn wiki_headings_use_the_registered_expansion() {
    let mut wrong = Vec::new();
    for p in PASSES {
        if let WikiCoverage::Documented(h) = p.wiki {
            let expected = format!("{} — {}", p.name, p.full_name);
            if h != expected {
                wrong.push(format!("{}: heading '{h}' != '{expected}'", p.name));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "wiki headings disagree with the registered expansion:\n  {}",
        wrong.join("\n  ")
    );
}

// ─── 2. Wiki agreement, both directions ────────────────────────────────────

const WIKI: &str = "docs/wiki/Optimization-Passes.md";

/// The `### ` headings under "## Per-pass descriptions".
fn wiki_pass_sections() -> Vec<String> {
    let text = read(WIKI);
    let mut out = Vec::new();
    let mut in_section = false;
    for line in text.lines() {
        if line.starts_with("## ") {
            in_section = line.trim() == "## Per-pass descriptions";
            continue;
        }
        if in_section {
            if let Some(h) = line.strip_prefix("### ") {
                out.push(h.trim().to_string());
            }
        }
    }
    assert!(
        !out.is_empty(),
        "found no '### ' headings under '## Per-pass descriptions' in {WIKI} — \
         the parser or the page structure changed, and every wiki assertion \
         below would be vacuous"
    );
    out
}

#[test]
fn every_documented_pass_has_its_wiki_section() {
    let sections = wiki_pass_sections();
    let mut missing = Vec::new();
    for p in PASSES {
        if let WikiCoverage::Documented(heading) = p.wiki {
            if !sections.iter().any(|s| s == heading) {
                missing.push(format!("{}: expected '### {heading}'", p.name));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "registry claims wiki sections that {WIKI} does not have:\n  {}\n\
         Present sections: {sections:?}",
        missing.join("\n  ")
    );
}

/// The INCOMPLETENESS direction: a pass documented in the wiki but absent from
/// the registry would otherwise be checked by nothing at all.
#[test]
fn every_wiki_pass_section_is_registered() {
    let sections = wiki_pass_sections();
    let registered: BTreeSet<&str> = PASSES
        .iter()
        .filter_map(|p| match p.wiki {
            WikiCoverage::Documented(h) => Some(h),
            WikiCoverage::Undocumented(_) => None,
        })
        .collect();
    let unregistered: Vec<&String> = sections
        .iter()
        .filter(|s| !registered.contains(s.as_str()))
        .collect();
    assert!(
        unregistered.is_empty(),
        "{WIKI} documents pass sections with no registry entry: {unregistered:?}\n\
         Add a PassDescriptor, or the section is checked by nothing."
    );
}

/// An `Undocumented` pass must say WHY. Otherwise the variant is an escape
/// hatch indistinguishable from an oversight.
#[test]
fn undocumented_passes_carry_a_reason() {
    for p in PASSES {
        if let WikiCoverage::Undocumented(reason) = p.wiki {
            assert!(
                reason.trim().len() > 20,
                "{} is Undocumented with a uselessly short reason: {reason:?}",
                p.name
            );
        }
    }
}

// ─── 3. CLI flags resolve — in BOTH argument blocks ────────────────────────

const ARGS: &str = "crates/nsl-cli/src/args.rs";

/// Field names declared inside `struct <name>`.
///
/// `args.rs` declares every shared flag TWICE, once in `BuildArgs` and once in
/// `RunArgs`. Checking only the union would pass for a flag present in one and
/// missing from the other — which is the drift that matters, because
/// `nsl build` and `nsl run` would then disagree.
fn arg_struct_fields(src: &str, struct_name: &str) -> BTreeSet<String> {
    let needle = format!("pub(crate) struct {struct_name} {{");
    let start = src
        .find(&needle)
        .unwrap_or_else(|| panic!("{ARGS}: cannot find `{needle}`"))
        + needle.len();
    // Walk to the matching close brace.
    let bytes = src.as_bytes();
    let mut depth = 1usize;
    let mut i = start;
    while i < bytes.len() && depth > 0 {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    let body = &src[start..i];
    let mut fields = BTreeSet::new();
    for line in body.lines() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("pub(crate) ") else {
            continue;
        };
        if let Some(name) = rest.split(':').next() {
            let name = name.trim();
            if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                fields.insert(name.to_string());
            }
        }
    }
    assert!(
        fields.len() > 30,
        "{ARGS}: parsed only {} fields out of {struct_name} — the parser is \
         broken and every flag assertion would be vacuous",
        fields.len()
    );
    fields
}

/// Every registered flag must appear in EXACTLY the arg structs the registry
/// says, and in no others.
///
/// Checking "appears somewhere" would miss the drift that matters: a flag
/// gaining `RunArgs` without anyone noticing, or losing `BuildArgs` in a
/// refactor. Pinning the exact set makes either direction a failure.
#[test]
fn every_registered_cli_flag_is_on_exactly_its_declared_subcommands() {
    let src = strip_line_comments(&read(ARGS));
    let check = arg_struct_fields(&src, "CheckArgs");
    let build = arg_struct_fields(&src, "BuildArgs");
    let run = arg_struct_fields(&src, "RunArgs");
    let lookup = |sc: Subcommand| match sc {
        Subcommand::Check => &check,
        Subcommand::Build => &build,
        Subcommand::Run => &run,
    };

    let mut problems = Vec::new();
    let mut checked = 0usize;
    for p in PASSES {
        for cf in p.cli_flags {
            checked += 1;
            let field = cf.flag.replace('-', "_");
            for sc in [Subcommand::Check, Subcommand::Build, Subcommand::Run] {
                let present = lookup(sc).contains(&field);
                let declared = cf.on.contains(&sc);
                if declared && !present {
                    problems.push(format!(
                        "{}: --{} is registered on `nsl {}` but {} has no field \
                         `{field}`",
                        p.name,
                        cf.flag,
                        format!("{sc:?}").to_lowercase(),
                        sc.arg_struct(),
                    ));
                } else if !declared && present {
                    problems.push(format!(
                        "{}: --{} IS in {} but the registry does not list `nsl {}` \
                         — the flag surface widened without being recorded",
                        p.name,
                        cf.flag,
                        sc.arg_struct(),
                        format!("{sc:?}").to_lowercase(),
                    ));
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "registered CLI flags do not match args.rs:\n  {}",
        problems.join("\n  ")
    );
    assert!(
        checked >= 30,
        "only {checked} flags checked — the registry looks truncated"
    );
}

/// No flag may be registered with an empty subcommand list — that would make
/// the exactness check above pass trivially for it.
#[test]
fn every_registered_flag_is_on_at_least_one_subcommand() {
    for p in PASSES {
        for cf in p.cli_flags {
            assert!(
                !cf.on.is_empty(),
                "{}: --{} is registered on no subcommand at all",
                p.name,
                cf.flag
            );
        }
    }
}

/// Tree -> registry: a pass-prefixed flag in `args.rs` must be registered.
///
/// The gate above walks registry -> args.rs, which cannot notice a flag that
/// was ADDED. Adding `wggo_newthing` to `BuildArgs` and `RunArgs` left all
/// fifteen gates green while the registry and the wiki silently
/// under-described WGGO — the same incompleteness hole the feature-composition
/// registry (item 20) had to be reworked to close, found here by review.
#[test]
fn every_pass_prefixed_cli_flag_is_registered() {
    let src = strip_line_comments(&read(ARGS));
    let mut all: BTreeSet<String> = BTreeSet::new();
    for st in ["CheckArgs", "BuildArgs", "RunArgs"] {
        all.extend(arg_struct_fields(&src, st));
    }
    let registered: BTreeSet<String> = PASSES
        .iter()
        .flat_map(|p| p.cli_flags.iter())
        .map(|cf| cf.flag.replace('-', "_"))
        .collect();

    // Field-name prefixes that unambiguously belong to a pass. A flag starting
    // with one of these and absent from the registry is a real omission.
    // `checkpoint_` is CCR's; the rest are the acronyms.
    const PASS_PREFIXES: &[&str] = &[
        "ccr_", "fase_", "wggo_", "csha_", "wrga_", "cpdt_", "pca_", "cpkd_",
        "cep_", "cfie_", "checkpoint_",
    ];
    // Exactly-named flags that are a pass's own switch rather than prefixed.
    const PASS_EXACT: &[&str] = &["wggo", "csha", "cpdt", "cfie", "memory"];

    let mut unregistered: Vec<&String> = all
        .iter()
        .filter(|f| {
            PASS_PREFIXES.iter().any(|p| f.starts_with(p)) || PASS_EXACT.contains(&f.as_str())
        })
        .filter(|f| !registered.contains(*f))
        .collect();
    unregistered.sort();
    assert!(
        unregistered.is_empty(),
        "args.rs declares pass flags absent from the registry: {unregistered:?}\n\
         Add them to the owning PassDescriptor (and document them), or the \
         registry under-describes the pass while every other gate stays green."
    );
    // Anti-vacuity: the filter must actually be selecting flags.
    let matched = all
        .iter()
        .filter(|f| {
            PASS_PREFIXES.iter().any(|p| f.starts_with(p)) || PASS_EXACT.contains(&f.as_str())
        })
        .count();
    assert!(
        matched >= 30,
        "only {matched} pass-prefixed flags found in args.rs — the prefix \
         filter is broken and this gate checks nothing"
    );
}

// ─── 4. Decorators ─────────────────────────────────────────────────────────

/// A registered decorator must be recognised by the frontend.
///
/// There is no central decorator allowlist in this codebase — names are
/// matched at their handling sites — so "recognised" means the quoted name
/// appears in nsl-semantic or nsl-parser outside a comment. That is weaker
/// than resolving it to a specific handler, and is stated as such rather than
/// implied to be stronger.
#[test]
fn every_decorator_trigger_is_recognised_somewhere() {
    let mut haystack = String::new();
    for dir in ["crates/nsl-semantic/src", "crates/nsl-parser/src"] {
        let root = repo_root().join(dir);
        let mut stack = vec![root];
        while let Some(d) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&d) else {
                continue;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    if let Ok(s) = std::fs::read_to_string(&p) {
                        haystack.push_str(&strip_line_comments(&s));
                    }
                }
            }
        }
    }
    assert!(
        haystack.len() > 100_000,
        "decorator haystack is only {} bytes — the crawl failed and every \
         assertion below would be vacuous",
        haystack.len()
    );
    let mut unknown = Vec::new();
    let mut checked = 0usize;
    for p in PASSES {
        for d in p.decorator_triggers {
            checked += 1;
            if !haystack.contains(&format!("\"{d}\"")) {
                unknown.push(format!("{}: @{d}", p.name));
            }
        }
    }
    assert!(
        unknown.is_empty(),
        "registered decorators are not recognised by the frontend:\n  {}",
        unknown.join("\n  ")
    );
    assert!(checked >= 10, "only {checked} decorators checked");
}

// ─── 5. The two tables must agree ──────────────────────────────────────────

const SHELL: &str = "scripts/check-doc-agreement.sh";

/// `check-doc-agreement.sh` carries its own `PASS_SOURCES` array. Two
/// hand-maintained tables that can disagree are worse than one, so pin them
/// together: every pass the shell script knows about must be registered here,
/// with the same source file.
#[test]
fn shell_pass_table_agrees_with_the_registry() {
    let text = read(SHELL);
    let start = text
        .find("PASS_SOURCES=(")
        .expect("check-doc-agreement.sh must still declare PASS_SOURCES");
    let end = text[start..]
        .find("\n)")
        .map(|o| start + o)
        .expect("PASS_SOURCES array must terminate");
    let body = &text[start..end];

    let mut rows: Vec<(String, String)> = Vec::new();
    for line in body.lines().skip(1) {
        let t = line.trim().trim_matches('"');
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some((label, src)) = t.split_once(':') {
            rows.push((label.to_string(), src.to_string()));
        }
    }
    assert!(
        rows.len() >= 8,
        "parsed only {} rows out of {SHELL}'s PASS_SOURCES — the parser is \
         broken and this gate would be vacuous",
        rows.len()
    );

    let mut problems = Vec::new();
    for (label, src) in &rows {
        match PASSES.iter().find(|p| p.name == label) {
            None => problems.push(format!(
                "{SHELL} knows pass '{label}' which has no PassDescriptor"
            )),
            Some(p) => {
                if !p.source_files.contains(&src.as_str()) {
                    problems.push(format!(
                        "{SHELL} maps {label} -> {src}, which is not in that \
                         pass's registered source_files {:?}",
                        p.source_files
                    ));
                }
            }
        }
    }
    // Reverse direction: a pass registered here but missing from the shell
    // table is invisible to that script's absence-claim check. `MemoryPlanner`
    // was exactly this — 11 registry entries, 10 shell rows, gate green.
    let shell_labels: BTreeSet<&str> = rows.iter().map(|(l, _)| l.as_str()).collect();
    for p in PASSES {
        if !shell_labels.contains(p.name) {
            problems.push(format!(
                "{} is registered but absent from {SHELL}'s PASS_SOURCES, so \
                 that script never checks it for absence claims",
                p.name
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "the shell table and the Rust registry disagree:\n  {}",
        problems.join("\n  ")
    );
}

// ─── 6. Completeness: a new pass cannot land unregistered ──────────────────

/// Module-name prefixes that are NOT passes, with the reason. Anything in
/// `nsl-codegen/src` whose prefix is neither a registered pass nor listed here
/// fails the completeness gate below.
///
/// This is the allowlist that forces "not a pass" to be a written decision,
/// the same shape the feature-composition sweep uses. Without it the
/// completeness check would be unrunnable noise; with it, adding a genuinely
/// new pass module fails until someone classifies it.
const NOT_A_PASS: &[(&str, &str)] = &[
    ("ad_rules", "adjoint rules table consumed by source-AD, not a pass"),
    ("agent", "M56 agent-memory feature"),
    ("autotune", "kernel autotuner, invoked BY passes rather than being one"),
    ("backend_", "target backends (ptx/amdgpu/metal/wgsl/verilog)"),
    ("bin", "binaries, not library modules"),
    ("bitnet", "M35 BitNet quantized-model support"),
    ("builtins", "FFI symbol registration"),
    ("c_export_table", "C ABI export table"),
    ("c_header", "C header emission"),
    ("c_wrapper", "C wrapper emission"),
    ("calibration", "calibration harness feeding WGGO importance scores"),
    ("compiler", "the driver itself"),
    ("context_parallel", "M34 @context_parallel - a parallelism feature, not an IR pass"),
    ("context", "compilation context types"),
    ("cost_model", "shared cost estimation used by several passes"),
    ("dynamic_shapes", "dynamic shape support"),
    ("epilogue_fusion", "GEMM epilogue fusion inside the kernel emitters"),
    ("error", "error types"),
    ("escape", "ownership escape analysis - operates on the AST, not the WengertList"),
    ("expr", "expression lowering"),
    ("ffi_ownership", "FFI ownership annotations"),
    ("flash_attention", "attention kernel emitter + selector"),
    ("fp8", "FP8 dtype support"),
    ("fpga_error", "FPGA backend errors"),
    ("func", "function lowering"),
    ("fused_linear_ce", "fused loss kernel emitter"),
    ("fusion", "elementwise fusion (M31) - graph/report/driver"),
    ("gpu_spec", "GPU capability tables"),
    ("gpu_target", "GPU target selection"),
    ("grammar_compiler", "CFIE grammar compilation"),
    ("hir", "high-level IR"),
    ("inspect", "dev-tools inspection"),
    ("kernel", "kernel skeleton/IR/lowering machinery"),
    ("layerwise", "CSLA layerwise accumulation scheduling"),
    ("lib", "crate root"),
    ("linker", "object linking"),
    ("matmul_mma", "tensor-core matmul emitter"),
    ("moe", "mixture-of-experts kernels and HF packing"),
    ("multimodal", "multimodal input support"),
    ("muon_roles", "Muon optimizer role classification"),
    ("ownership", "ownership infrastructure"),
    ("pass_registry", "this registry"),
    ("pipeline", "pipeline-parallel scheduling"),
    ("precision_cast_ptx", "precision-cast PTX emitter"),
    ("profiling", "dev-tools capture"),
    ("ptx_metadata", "PTX metadata"),
    ("ptxas_validation", "ptxas syntax validation"),
    ("reduction_fusion", "reduction fusion inside kernel emitters"),
    ("schema_convert", "schema conversion"),
    ("serve", "serving entry point"),
    ("source_ad", "the source-AD extractor - produces the WengertList passes consume"),
    ("sparse", "sparse tensor support"),
    ("speculative", "speculative decoding support"),
    ("standalone", "standalone binary emission"),
    ("stdlib_loader", "stdlib module loading"),
    ("stmt", "statement lowering - this is where the passes are DRIVEN from"),
    ("tensor_parallel", "tensor-parallel sharding"),
    ("test_helpers", "test-only helpers"),
    ("training_report", "training report emission"),
    ("transient_arena", "transient-memory arena (roadmap item 4)"),
    ("types", "type machinery"),
    ("unikernel", "M54 unikernel targets"),
    ("use_count", "dead use-count analysis (zero readers since 5e2740bc)"),
    ("vmap", "M39 vmap"),
    ("wcet", "M53 worst-case execution time"),
    ("weight_aware", "M52 weight-aware invariants"),
    ("wengert", "the IR itself, and its Cranelift lowering"),
    ("wgrad_fusion", "roadmap item 7 weight-gradient GEMM fusion peephole"),
    ("zk", "M55 zero-knowledge proofs"),
];

#[test]
fn every_codegen_pass_module_belongs_to_a_registered_pass() {
    let dir = repo_root().join("crates/nsl-codegen/src");
    // A module belongs to a pass if it is one of that pass's registered
    // source files, or its stem matches the pass name as a module prefix.
    // Both are needed: `memory_planner.rs` is registered by path but its stem
    // does not match the pass name "MemoryPlanner" lowercased, while
    // `wggo_cost.rs` matches the "wggo_" prefix without being individually
    // registered.
    let registered: Vec<String> = PASSES.iter().map(|p| p.name.to_lowercase()).collect();
    let registered_stems: BTreeSet<String> = PASSES
        .iter()
        .flat_map(|p| p.source_files.iter())
        .filter_map(|f| {
            Path::new(f)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
        })
        .collect();

    let mut orphans = Vec::new();
    let mut seen = 0usize;
    let mut dirs_seen = 0usize;
    for e in std::fs::read_dir(&dir).expect("codegen src must be readable").flatten() {
        let p = e.path();
        // Directory modules count too. `compiler/`, `expr/`, `profiling/` are
        // modules exactly as `ccr.rs` is, and an earlier version of this walk
        // looked only at top-level `.rs` files. That made eight NOT_A_PASS
        // entries dead (matching nothing) while a new pass shipped as a
        // DIRECTORY would have slipped the completeness check entirely.
        let stem = if p.is_dir() {
            dirs_seen += 1;
            p.file_name().unwrap().to_string_lossy().to_string()
        } else if p.extension().is_some_and(|x| x == "rs") {
            p.file_stem().unwrap().to_string_lossy().to_string()
        } else {
            continue;
        };
        seen += 1;
        let is_pass = registered_stems.contains(&stem)
            || registered
                .iter()
                .any(|r| stem == *r || stem.starts_with(&format!("{r}_")));
        let is_excluded = NOT_A_PASS
            .iter()
            .any(|(prefix, _)| stem == *prefix || stem.starts_with(prefix));
        if !is_pass && !is_excluded {
            orphans.push(stem);
        }
    }
    // Two separate assertions. The total guards against a failed `read_dir`;
    // the DIRECTORY count guards against the specific regression the walk
    // above was written to fix — reverting to a `.rs`-only walk leaves the
    // total at ~167, comfortably past any total-based threshold, while
    // silently skipping every directory module.
    assert!(
        seen > 80,
        "only {seen} codegen modules seen — the directory walk failed"
    );
    assert!(
        dirs_seen >= 10,
        "only {dirs_seen} DIRECTORY modules seen — the walk has gone back to \
         looking at .rs files only, so a pass shipped as a directory would \
         slip this gate entirely"
    );
    orphans.sort();
    assert!(
        orphans.is_empty(),
        "these nsl-codegen modules belong to no registered pass and are not \
         classified in NOT_A_PASS:\n  {}\n\
         If one is a new pass, add a PassDescriptor. If it is not, add it to \
         NOT_A_PASS with a reason — so that 'not a pass' stays a decision \
         somebody made rather than an omission.",
        orphans.join("\n  ")
    );
}

/// Every `NOT_A_PASS` exclusion must carry a reason and must actually match
/// something. A stale exclusion silently widens the hole in the gate above.
#[test]
fn not_a_pass_exclusions_are_justified_and_live() {
    let dir = repo_root().join("crates/nsl-codegen/src");
    let stems: Vec<String> = std::fs::read_dir(&dir)
        .expect("codegen src must be readable")
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if p.is_dir() {
                Some(p.file_name()?.to_string_lossy().to_string())
            } else if p.extension().is_some_and(|x| x == "rs") {
                Some(p.file_stem()?.to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    let mut dead = Vec::new();
    for (prefix, reason) in NOT_A_PASS {
        assert!(
            reason.trim().len() > 5,
            "NOT_A_PASS entry '{prefix}' has no usable reason"
        );
        if !stems.iter().any(|s| s == prefix || s.starts_with(prefix)) {
            dead.push(*prefix);
        }
    }
    assert!(
        dead.is_empty(),
        "NOT_A_PASS excludes module prefixes that no longer exist: {dead:?} — \
         remove them, or the exclusion list is quietly wider than the tree"
    );
}

// ─── 7. Internal consistency ───────────────────────────────────────────────

#[test]
fn registry_is_internally_consistent() {
    let mut names = BTreeSet::new();
    for p in PASSES {
        assert!(
            names.insert(p.name),
            "duplicate pass name in the registry: {}",
            p.name
        );
        assert!(
            !p.full_name.trim().is_empty(),
            "{} has no expanded name — the original CCR drift included a WRONG \
             expansion, so this field is load-bearing",
            p.name
        );
        for cf in p.cli_flags {
            assert!(
                !cf.flag.starts_with("--"),
                "{}: cli_flags must omit the leading dashes, got {:?}",
                p.name,
                cf.flag
            );
            assert!(
                !cf.flag.contains('_'),
                "{}: cli_flags are kebab-case as the user types them, got {:?}",
                p.name,
                cf.flag
            );
        }
        for d in p.decorator_triggers {
            assert!(
                !d.starts_with('@'),
                "{}: decorator_triggers must omit the @, got {d:?}",
                p.name
            );
        }
    }
    // Anti-vacuity: pin the expected shape of the registry so a truncation
    // that leaves the loops trivially satisfied fails here instead.
    assert!(
        PASSES.len() >= 11,
        "registry has only {} passes; the compiler has at least 11",
        PASSES.len()
    );
    assert!(
        PASSES
            .iter()
            .any(|p| p.stage == PipelineStage::OnWengert),
        "no pass is registered as operating on the WengertList, which cannot \
         be right"
    );
}

/// A pass with no flags and no decorators is unreachable by any user-facing
/// control, which is almost certainly a registry error rather than a real
/// property.
#[test]
fn every_pass_has_at_least_one_trigger() {
    let mut untriggered: Vec<&PassDescriptor> = Vec::new();
    for p in PASSES {
        if p.cli_flags.is_empty() && p.decorator_triggers.is_empty() {
            untriggered.push(p);
        }
    }
    assert!(
        untriggered.is_empty(),
        "these passes have neither a CLI flag nor a decorator, so nothing \
         documented can turn them on: {:?}",
        untriggered.iter().map(|p| p.name).collect::<Vec<_>>()
    );
}
