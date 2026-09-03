//! Milestone A: total-coverage drift gates for the activation-contract table.
//!
//! The strong claims, each gated in the direction that catches tomorrow's
//! drift (item-20/21 lessons):
//!
//! 1. TOTAL COVERAGE (tree -> table): every field of CheckArgs / BuildArgs /
//!    RunArgs is covered by exactly one of {a pass's `cli_flags`,
//!    `MANUAL_CONTRACTS`, `UNCONTRACTED_FLAGS`}. A flag added tomorrow
//!    without a contract or a written-down exclusion fails here.
//! 2. NO GHOSTS (table -> tree): every manual/uncontracted row names a flag
//!    that actually exists in args.rs.
//! 3. ENTRY SETS: each manual contract's `on` EXACTLY matches the arg
//!    structs that declare the flag — the "Build/Run/Check differences are
//!    explicitly declared" exit criterion, gated bidirectionally.
//! 4. WITNESSES ARE REAL: every `Marker` string appears literally in some
//!    src file; every `Config`/`Report` site file exists. A witness that
//!    cannot be observed is decorative metadata — inadmissible.

use std::collections::BTreeMap;
use std::path::PathBuf;

use nsl_codegen::activation::{self, Surface, Witness};
use nsl_codegen::pass_registry::{self, Subcommand};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Parse args.rs: struct name -> kebab-case flag names. args.rs derives every
/// long flag from the field name (no `long = "..."` renames exist — asserted
/// below so a rename cannot silently invalidate this parser).
fn parse_arg_structs() -> BTreeMap<&'static str, Vec<String>> {
    let src = std::fs::read_to_string(
        workspace_root().join("crates/nsl-cli/src/args.rs"),
    )
    .expect("read args.rs");
    assert!(
        !src.contains("long = \""),
        "args.rs now uses an explicit `long = \"...\"` rename; this parser \
         derives flags from field names and must learn the rename first"
    );
    let mut out: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    let mut cur: Option<&'static str> = None;
    let mut flatten_next = false;
    for line in src.lines() {
        for name in ["CheckArgs", "BuildArgs", "RunArgs"] {
            if line.contains(&format!("struct {name} {{")) {
                cur = Some(name);
            }
        }
        if let Some(struct_name) = cur {
            let t = line.trim();
            if t == "}" {
                cur = None;
                continue;
            }
            if t.starts_with("#[command(flatten)]") {
                flatten_next = true;
                continue;
            }
            if let Some(rest) = t.strip_prefix("pub(crate) ")
                && let Some((field, ty)) = rest.split_once(':')
            {
                if std::mem::take(&mut flatten_next) {
                    // A flattened group contributes the INNER struct's fields.
                    // Naming the field itself would invent a flag that does not
                    // exist (`--matmul`) and hide the seven that do.
                    out.entry(struct_name).or_default().extend(flattened_fields(ty));
                } else {
                    out.entry(struct_name)
                        .or_default()
                        .push(field.trim().replace('_', "-"));
                }
            }
        }
    }
    out
}

/// The flags a `#[command(flatten)]` field contributes.
///
/// `ty` is the field's type as written, e.g. `crate::matmul_args::MatmulArgs`.
/// The module segment names the file under `crates/nsl-cli/src/`, and the last
/// segment names the struct inside it. Panics rather than returning empty: a
/// flatten this cannot follow would silently drop every flag in the group, and
/// the gate exists to make dropped flags impossible.
fn flattened_fields(ty: &str) -> Vec<String> {
    let ty = ty.trim().trim_end_matches(',').trim();
    let segs: Vec<&str> = ty.split("::").collect();
    let (struct_name, module) = match segs.as_slice() {
        [.., m, s] => (*s, *m),
        _ => panic!("cannot resolve flattened type {ty:?} to a module::Struct path"),
    };
    let path = workspace_root().join(format!("crates/nsl-cli/src/{module}.rs"));
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("flattened group {ty:?} -> {}: {e}", path.display()));
    assert!(
        !src.contains("long = \""),
        "{module}.rs uses an explicit `long = \"...\"` rename; this parser derives \
         flags from field names and must learn the rename first"
    );
    let mut fields = Vec::new();
    let mut inside = false;
    for line in src.lines() {
        if line.contains(&format!("struct {struct_name} {{")) {
            inside = true;
            continue;
        }
        if inside {
            let t = line.trim();
            if t == "}" {
                break;
            }
            if let Some(rest) = t.strip_prefix("pub(crate) ")
                && let Some((field, _)) = rest.split_once(':')
            {
                fields.push(field.trim().replace('_', "-"));
            }
        }
    }
    assert!(
        !fields.is_empty(),
        "flattened group {ty:?} contributed no flags — struct {struct_name} not found in {}",
        path.display()
    );
    fields
}

fn declared_on(structs: &BTreeMap<&'static str, Vec<String>>, flag: &str) -> Vec<Subcommand> {
    let mut on = Vec::new();
    for (name, sub) in [
        ("CheckArgs", Subcommand::Check),
        ("BuildArgs", Subcommand::Build),
        ("RunArgs", Subcommand::Run),
    ] {
        if structs.get(name).map(|v| v.iter().any(|f| f == flag)).unwrap_or(false) {
            on.push(sub);
        }
    }
    on
}

#[test]
fn every_cli_flag_is_covered_exactly_once() {
    let structs = parse_arg_structs();
    // Anti-vacuity above the collapsed value: a broken parser finds 0; the
    // union today is ~135 fields.
    let total: usize = structs.values().map(|v| v.len()).sum();
    assert!(total >= 120, "args.rs parse collapsed: {total} fields");

    let registry: Vec<&str> = pass_registry::PASSES
        .iter()
        .flat_map(|p| p.cli_flags.iter().map(|f| f.flag))
        .collect();
    let manual: Vec<&str> = activation::MANUAL_CONTRACTS
        .iter()
        .filter_map(|c| match c.surface {
            Surface::Flag(f) => Some(f),
            Surface::Decorator(_) => None,
        })
        .collect();
    let uncontracted: Vec<&str> =
        activation::UNCONTRACTED_FLAGS.iter().map(|(f, _)| *f).collect();

    let mut all_flags: Vec<String> = structs.values().flatten().cloned().collect();
    all_flags.sort();
    all_flags.dedup();

    for f in &all_flags {
        let sets = [
            registry.contains(&f.as_str()),
            manual.contains(&f.as_str()),
            uncontracted.contains(&f.as_str()),
        ];
        let n = sets.iter().filter(|b| **b).count();
        assert!(
            n != 0,
            "--{f} has no activation contract and no written-down exclusion — \
             add a MANUAL_CONTRACTS row (owner + witness), or an \
             UNCONTRACTED_FLAGS entry with the reason"
        );
        assert!(
            n == 1,
            "--{f} is claimed by {n} coverage sets (registry={}, manual={}, \
             uncontracted={}) — exactly one must own it",
            sets[0],
            sets[1],
            sets[2],
        );
    }

    // Direction 2: no ghost rows.
    for f in manual.iter().chain(uncontracted.iter()) {
        assert!(
            all_flags.iter().any(|a| a == f),
            "table names --{f} but no arg struct declares it — the flag died \
             or was renamed; remove or update the row"
        );
    }

    // Reasons must be non-empty (a written-down decision, not a shrug).
    for (f, reason) in activation::UNCONTRACTED_FLAGS {
        assert!(
            reason.len() >= 10,
            "--{f}: uncontracted reason is too short to be a decision: {reason:?}"
        );
    }
}

/// Exit criterion "Build/Run/Check differences are explicitly declared":
/// every manual contract's `on` must EXACTLY match the structs that declare
/// the flag, both directions.
#[test]
fn entry_sets_match_the_arg_structs_exactly() {
    let structs = parse_arg_structs();
    for c in activation::MANUAL_CONTRACTS {
        let Surface::Flag(f) = c.surface else { continue };
        let actual = declared_on(&structs, f);
        assert_eq!(
            c.on, &actual[..],
            "--{f}: contract declares {:?} but args.rs declares it on {:?}",
            c.on, actual
        );
    }
}

/// Every witness must be observable: markers appear literally somewhere in
/// src; Config/Report sites are real files.
#[test]
fn every_witness_is_observable() {
    let root = workspace_root();
    // Collect all src file contents once — EXCLUDING the registry files
    // that carry the marker strings as data. Review caught the first
    // version matching the contract table's own source, which made this
    // gate structurally unable to fail (the item-20 "verify the gate fails
    // for the right input" defect, shipped verbatim): "[flash-attention]"
    // was an unobservable witness and the gate was green.
    const SELF_REFERENTIAL: &[&str] = &[
        "crates/nsl-codegen/src/activation.rs",
        "crates/nsl-cli/src/exec_markers.rs",
    ];
    let mut src_blobs: Vec<(PathBuf, String)> = Vec::new();
    for crate_dir in std::fs::read_dir(root.join("crates")).unwrap() {
        let src = crate_dir.unwrap().path().join("src");
        if src.is_dir() {
            collect_rs(&src, &mut src_blobs);
        }
    }
    src_blobs.retain(|(p, _)| {
        let rel = p.strip_prefix(&root).unwrap_or(p).to_string_lossy().replace('\\', "/");
        !SELF_REFERENTIAL.iter().any(|s| rel.ends_with(s) || rel == *s)
    });
    assert!(src_blobs.len() >= 100, "src walk collapsed: {} files", src_blobs.len());

    for c in activation::MANUAL_CONTRACTS {
        match c.witness {
            Witness::Marker(m) => {
                assert!(
                    src_blobs.iter().any(|(_, s)| s.contains(m)),
                    "{}: marker {m} appears nowhere under crates/*/src — the \
                     witness is unobservable",
                    c.surface.render(),
                );
            }
            Witness::Config(site) | Witness::Report(site) => {
                let path = site.split("::").next().unwrap_or(site);
                assert!(
                    root.join(path).is_file(),
                    "{}: witness site {path} is not a file",
                    c.surface.render(),
                );
            }
            Witness::Disposition(pass) => {
                assert!(
                    pass_registry::pass(pass).is_some(),
                    "{}: disposition owner {pass} is not a registered pass",
                    c.surface.render(),
                );
            }
        }
    }
}

fn collect_rs(dir: &std::path::Path, out: &mut Vec<(PathBuf, String)>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            collect_rs(&p, out);
        } else if p.extension().map(|x| x == "rs").unwrap_or(false)
            && let Ok(s) = std::fs::read_to_string(&p)
        {
            out.push((p, s));
        }
    }
}

/// Cross-registry pins: the three tables describing decorators must agree.
/// - Every pass_registry decorator_trigger must be a KNOWN decorator, or the
///   semantic namespace close rejects a name the pass registry says it owns.
/// - Every activation decorator contract must name a KNOWN decorator (no
///   contracts for names no program can write).
#[test]
fn the_decorator_registries_agree() {
    for p in pass_registry::PASSES {
        for d in p.decorator_triggers {
            assert!(
                nsl_semantic::decorator_registry::find(d).is_some(),
                "{}: decorator_trigger @{d} is not in KNOWN_DECORATORS — the \
                 namespace close would reject the name this pass owns",
                p.name
            );
        }
    }
    for c in activation::contracts() {
        if let Surface::Decorator(d) = c.surface {
            assert!(
                nsl_semantic::decorator_registry::find(d).is_some(),
                "@{d} has an activation contract but is not a KNOWN decorator"
            );
        }
    }
}
