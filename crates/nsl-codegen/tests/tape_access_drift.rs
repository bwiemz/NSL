//! Drift gates for the `tape:` declarations in `pass_registry.rs` — the
//! tape-mediated half of item 2's inter-pass dependency story.
//!
//! The bus's `consumed_by_passes` edges cover values passed pass-to-pass;
//! this covers the OTHER medium: passes that rewrite or scan the shared
//! `WengertList`. A declaration nobody checks is prose with extra steps, so
//! each `TapeAccess` claim is verified against the tree in both directions:
//! a pass declaring `None` must have no code-level mention of the type in
//! its registered sources, and a pass declaring access must point (`via`)
//! at real files that actually mention it. The commit-point uniqueness
//! belts and the id-minting rule are pinned textually — they are one-line
//! calls a refactor could drop with nothing else noticing.

use std::path::{Path, PathBuf};

use nsl_codegen::pass_registry::{TapeAccess, PASSES};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Drop `//` comment lines and trailing comments, then strip whitespace —
/// the multi-line lesson, plus: `cfie.rs` states "CFIE never sees a
/// WengertList" in prose, which must not count as an access.
fn code_only(text: &str) -> String {
    text.lines()
        .map(|l| match l.find("//") {
            Some(i) if !l[..i].contains('"') => &l[..i],
            _ => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .split_whitespace()
        .collect()
}

fn mentions_tape(path: &Path) -> bool {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    code_only(&text).contains("WengertList")
}

#[test]
fn every_tape_declaration_matches_the_tree() {
    let root = repo_root();
    for p in PASSES {
        match &p.tape {
            TapeAccess::None => {
                for f in p.source_files {
                    let path = root.join(f);
                    assert!(
                        !mentions_tape(&path),
                        "{}: declares TapeAccess::None but {} mentions \
                         WengertList in code — either the pass grew tape \
                         access (declare it) or the mention is dead",
                        p.name,
                        f
                    );
                }
            }
            TapeAccess::Reads { refs, via }
            | TapeAccess::MutatesInPlace { refs, via, .. }
            | TapeAccess::MutatesFork { refs, via, .. } => {
                assert!(
                    !refs.is_empty(),
                    "{}: tape access with no reference kinds says nothing \
                     a checker can use",
                    p.name
                );
                assert!(
                    !via.is_empty(),
                    "{}: tape access must name where it lives",
                    p.name
                );
                for f in *via {
                    let path = root.join(f);
                    assert!(
                        path.exists(),
                        "{}: tape via-file {} does not exist",
                        p.name,
                        f
                    );
                    assert!(
                        mentions_tape(&path),
                        "{}: tape via-file {} has no code-level WengertList \
                         mention — the declared evidence is stale",
                        p.name,
                        f
                    );
                }
            }
        }
    }
}

/// Mutator prose: every in-place mutator explains WHAT it rewrites, and the
/// two OnWengert passes that declare `None` are exactly the pair whose
/// stage/access mismatch motivated the field (CPDT, CPKD).
#[test]
fn mutators_say_what_and_the_known_none_set_is_exact() {
    let mut in_place = Vec::new();
    let mut on_wengert_none = Vec::new();
    for p in PASSES {
        match &p.tape {
            TapeAccess::MutatesInPlace { what, .. }
            | TapeAccess::MutatesFork { what, .. } => {
                assert!(
                    !what.trim().is_empty(),
                    "{}: a mutator with an empty `what` is undeclared in \
                     all but syntax",
                    p.name
                );
                if matches!(p.tape, TapeAccess::MutatesInPlace { .. }) {
                    in_place.push(p.name);
                }
            }
            TapeAccess::None => {
                if matches!(
                    p.stage,
                    nsl_codegen::pass_registry::PipelineStage::OnWengert
                ) {
                    on_wengert_none.push(p.name);
                }
            }
            TapeAccess::Reads { .. } => {}
        }
    }
    assert_eq!(
        in_place,
        vec!["CCR", "WGGO"],
        "the in-place mutator set changed — update the commit-point belts \
         (assert_unique_op_ids) alongside this list"
    );
    assert_eq!(
        on_wengert_none,
        vec!["CPDT", "CPKD"],
        "the OnWengert-but-no-tape set changed — this pair is the recorded \
         evidence that stage is not access"
    );
}

/// The commit-point uniqueness belts and the id-minting rule, pinned
/// textually (comment/whitespace-stripped): three production mutators wear
/// the belt, and CCR's append mints via `fresh_op_id`, never `ops.len()`.
#[test]
fn the_uniqueness_belts_and_the_minting_rule_hold() {
    let root = repo_root();
    let read = |rel: &str| {
        code_only(
            &std::fs::read_to_string(root.join(rel))
                .unwrap_or_else(|e| panic!("read {rel}: {e}")),
        )
    };
    for (file, needle) in [
        (
            "crates/nsl-codegen/src/ccr.rs",
            "assert_unique_op_ids(\"ccr::append_compressed_saves\")",
        ),
        (
            "crates/nsl-codegen/src/wggo_prune.rs",
            "assert_unique_op_ids(\"wggo_prune::run(post-commit)\")",
        ),
        (
            "crates/nsl-codegen/src/source_ad.rs",
            "assert_unique_op_ids(\"fused-LCEdead-chainprune\")",
        ),
    ] {
        assert!(
            read(file).contains(&needle.split_whitespace().collect::<String>()),
            "{file}: the commit-point uniqueness belt is gone (or its \
             context string changed) — update this pin with it"
        );
    }
    let ccr = read("crates/nsl-codegen/src/ccr.rs");
    assert!(
        ccr.contains("primal.fresh_op_id()"),
        "ccr::append_compressed_saves no longer mints via fresh_op_id"
    );
    assert!(
        !ccr.contains("letid=primal.ops.len()asu32"),
        "a len-minted op id is back in ccr.rs — it collides after any \
         deletion (see WengertList::fresh_op_id)"
    );
}
