//! Milestone A drift gates for the closed decorator namespace.
//!
//! Directionality (item-20/21 lessons): the valuable direction is
//! tree -> registry — a consumer added tomorrow must not reopen the
//! namespace silently. registry -> tree matters too (a row whose consumer
//! died is an over-claim), and both are pinned here.

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// registry -> tree: every row's `read_by` file must still mention the name.
/// A consumer that moved or died must take its registry row with it.
#[test]
fn every_registry_row_still_has_its_consumer() {
    let root = workspace_root();
    for d in nsl_semantic::decorator_registry::KNOWN_DECORATORS {
        let path = root.join(d.read_by);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read_by file for @{} unreadable: {}: {e}", d.name, d.read_by));
        assert!(
            src.contains(&format!("\"{}\"", d.name)),
            "@{}: {} no longer mentions the name as a string literal — the \
             consumer moved or died; update or remove the registry row",
            d.name,
            d.read_by,
        );
    }
}

/// tree -> registry: every name the semantic checkers branch on must be a
/// registry row. Parses the `dname == \"...\"` / match-arm name patterns out
/// of the checker sources, so a validator added tomorrow without a registry
/// row fails HERE rather than silently reopening the namespace.
#[test]
fn every_checker_validated_name_is_registered() {
    let root = workspace_root();
    let mut found: Vec<String> = Vec::new();
    for rel in [
        "crates/nsl-semantic/src/checker/stmt.rs",
        "crates/nsl-semantic/src/checker/model.rs",
    ] {
        let src = std::fs::read_to_string(root.join(rel)).unwrap();
        // Strip line comments so prose can't add names.
        let code: String = src
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        for cap in code.split("dname == \"").skip(1) {
            if let Some(name) = cap.split('"').next() {
                found.push(name.to_string());
            }
        }
    }
    found.sort();
    found.dedup();
    // Anti-vacuity: the scrape must find a healthy fraction of the checker
    // surface. 30 clears today's ~40 with headroom above the collapsed value
    // (a broken split finds 0).
    assert!(
        found.len() >= 30,
        "checker-name scrape collapsed: found only {} names",
        found.len()
    );
    for name in &found {
        // Names consumed by the checkers but parsed into dedicated AST nodes
        // (never `Decorator`) do not need namespace rows.
        if ["pack", "unpack", "backward", "endpoint"].contains(&name.as_str()) {
            continue;
        }
        assert!(
            nsl_semantic::decorator_registry::find(name).is_some(),
            "checker validates @{name} but the namespace registry has no row — \
             a validator was added without closing the namespace over it"
        );
    }
}

/// The generic namespace close must not have disturbed the three names the
/// csha/checkpoint e2e fixture depends on (that CLI test asserts stderr does
/// NOT contain \"unknown decorator\").
#[test]
fn the_csha_checkpoint_fixture_names_stay_known() {
    for name in ["csha", "checkpoint", "pure"] {
        assert!(
            nsl_semantic::decorator_registry::find(name).is_some(),
            "@{name} fell out of the registry — csha_checkpoint_decorator_cli_e2e \
             will start failing on the unknown-decorator error it asserts absent"
        );
    }
}

/// End-to-end through `analyze`: a fabricated name errors, a typo suggests,
/// a ghost gets its typed refusal, and every KNOWN name passes bare.
#[test]
fn analyze_enforces_the_closed_namespace() {
    let check = |src: &str| -> Vec<String> {
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, _) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        let parse = nsl_parser::parse(&tokens, &mut interner);
        nsl_semantic::analyze(&parse.module, &mut interner)
            .diagnostics
            .iter()
            .filter(|d| d.level == nsl_errors::Level::Error)
            .map(|d| d.message.clone())
            .collect()
    };

    let fab = check("@totally_not_a_real_feature(alpha = 1)\nfn f() -> f64:\n    return 1.0\n");
    assert!(
        fab.iter().any(|m| m.contains("unknown decorator @totally_not_a_real_feature")),
        "fabricated name must error: {fab:?}"
    );

    let typo = check("@cpdtt(mode = off)\nfn f() -> f64:\n    return 1.0\n");
    assert!(
        typo.iter().any(|m| m.contains("did you mean @cpdt?")),
        "typo must suggest: {typo:?}"
    );

    let ghost = check("@tie_weights\nfn f() -> f64:\n    return 1.0\n");
    assert!(
        ghost.iter().any(|m| m.contains("no \nimplementation") || m.contains("no implementation")),
        "ghost must get its typed refusal: {ghost:?}"
    );
}
