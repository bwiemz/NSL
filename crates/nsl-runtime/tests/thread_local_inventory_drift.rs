//! Item 17: the thread-local inventory in
//! `docs/architecture/compiler-state.md` is machine-checked against the
//! source of EVERY workspace crate (nsl-cli housed the last MIGRATE-class
//! offenders — the retired WRGA trio — so scope matters).
//!
//! Why: the table's whole value is the classification discipline ("new
//! behavior threads through an explicit context, never a new thread-local"
//! — with a reviewer able to see the full set). Hand-maintained, it had
//! drifted by ~20 statics by 2026-08-06 — including the autodiff `TAPE`
//! itself — which is exactly the silent rot a source-scan gate exists to
//! stop (the `pass_registry_drift` / `NO_DTYPE_TENSOR` precedent).
//!
//! Two directions:
//!  1. every `thread_local!` static in any workspace crate appears in the
//!     doc's marked table (a NEW thread-local must be added
//!     there, with a class and a reason, to pass CPU CI);
//!  2. every name the table lists still exists in the file its row names
//!     (renames and deletions must update the doc in the same PR).
//!
//! Macro-declared statics (hir/ids.rs' `define_id!`) are invisible to the
//! block scan (the macro body declares `$counter`), so direction 1 finds
//! nothing there and direction 2 carries the check: the names must appear
//! at the macro's instantiation site.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
        let p = e.unwrap().path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// ALL_CAPS static names declared inside `thread_local!` blocks of `src`.
/// Brace-counted from each `thread_local!` occurrence; `static NAME:` lines
/// with an ALL_CAPS name are collected (macro `$counter` declarations have a
/// lowercase `$` head and fall out naturally).
/// `name` appears in `text` as a standalone identifier (both neighbors are
/// non-ident chars) — the fence that stops `WS` matching inside `SR_WS`.
fn contains_ident(text: &str, name: &str) -> bool {
    let bytes = text.as_bytes();
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut from = 0;
    while let Some(pos) = text[from..].find(name) {
        let s = from + pos;
        let e = s + name.len();
        let left_ok = s == 0 || !is_ident(bytes[s - 1]);
        let right_ok = e >= bytes.len() || !is_ident(bytes[e]);
        if left_ok && right_ok {
            return true;
        }
        from = s + 1;
    }
    false
}

fn scan_thread_local_statics(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut search_from = 0;
    while let Some(pos) = src[search_from..].find("thread_local!") {
        let start = search_from + pos;
        // A doc-comment MENTION of thread_local! (hir/ids.rs's module doc)
        // must not anchor a pseudo-block over whatever braces follow —
        // skip matches on comment lines (review finding).
        let line_start = src[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line_head = src[line_start..start].trim_start();
        if line_head.starts_with("//") {
            search_from = start + "thread_local!".len();
            continue;
        }
        let Some(open_rel) = src[start..].find('{') else {
            break;
        };
        // Known limit: a brace inside a string literal or comment WITHIN a
        // block would end the extraction early and silently skip later
        // statics in that block (none exist today; review finding). Full
        // lexing is not worth it for a doc gate — revisit if a block ever
        // legitimately contains a braced literal.
        let mut depth = 0usize;
        let mut end = start + open_rel;
        for (i, ch) in src[start + open_rel..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + open_rel + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let block = &src[start..end];
        for line in block.lines() {
            let t = line.trim();
            let Some(rest) = t
                .strip_prefix("pub(crate) static ")
                .or_else(|| t.strip_prefix("pub static "))
                .or_else(|| t.strip_prefix("static "))
            else {
                continue;
            };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            if !name.is_empty() && name.chars().next().unwrap().is_ascii_uppercase() {
                out.push(name);
            }
        }
        search_from = end;
    }
    out
}

#[test]
fn the_doc_inventory_matches_the_source() {
    let root = repo_root();
    let doc_path = root.join("docs/architecture/compiler-state.md");
    let doc = std::fs::read_to_string(&doc_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", doc_path.display()));
    let table = doc
        .split_once("<!-- TL-INVENTORY-BEGIN -->")
        .and_then(|(_, r)| r.split_once("<!-- TL-INVENTORY-END -->"))
        .map(|(t, _)| t)
        .expect("TL-INVENTORY markers missing from compiler-state.md");

    // Doc side: (file-suffix, name) per row — parsed from the Location and
    // State COLUMNS only (cells 1 and 2); the Notes column legitimately
    // backticks other statics' names (the first run of this gate caught its
    // own parser reading `TRANSFER_STREAM` out of a note).
    let mut doc_entries: Vec<(String, String)> = Vec::new();
    for row in table.lines().filter(|l| l.trim_start().starts_with("| `")) {
        let cells: Vec<&str> = row.split('|').collect();
        if cells.len() < 3 {
            continue;
        }
        let path_cell = cells[1];
        let state_cell = cells[2];
        let Some(path) = path_cell.split('`').nth(1) else {
            continue;
        };
        let path = &path.to_string();
        let names: Vec<&str> = state_cell.split('`').skip(1).step_by(2).collect();
        for n in &names {
            if !n.is_empty()
                && n.chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            {
                doc_entries.push((path.to_string(), n.to_string()));
            }
        }
    }
    assert!(
        doc_entries.len() > 30,
        "only {} inventory entries parsed — the table format changed and \
         this gate is scanning almost nothing",
        doc_entries.len()
    );

    // Source side: every thread_local! static in EVERY workspace crate —
    // nsl-cli housed the last MIGRATE-class offenders (the WRGA trio), so a
    // two-crate scope would leave the historical bug class unmonitored
    // (review finding).
    let mut files = Vec::new();
    for entry in std::fs::read_dir(root.join("crates")).expect("crates dir") {
        let src = entry.unwrap().path().join("src");
        if src.is_dir() {
            rs_files(&src, &mut files);
        }
    }
    let mut src_entries: Vec<(String, String)> = Vec::new();
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap();
        let rel = f
            .strip_prefix(root.join("crates"))
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        for name in scan_thread_local_statics(&src) {
            src_entries.push((rel.clone(), name));
        }
    }

    // Direction 1: every source static has a doc row whose path matches.
    let doc_set: BTreeSet<(String, String)> = doc_entries
        .iter()
        .map(|(p, n)| (p.clone(), n.clone()))
        .collect();
    let undocumented: Vec<String> = src_entries
        .iter()
        .filter(|(rel, name)| {
            !doc_set
                .iter()
                .any(|(dp, dn)| dn == name && rel.ends_with(dp.as_str()))
        })
        .map(|(rel, name)| format!("  {name} ({rel})"))
        .collect();
    assert!(
        undocumented.is_empty(),
        "{} thread-local static(s) missing from the inventory table in \
         docs/architecture/compiler-state.md — classify each (TEST / \
         FFI/RUNTIME-OK / MIGRATE) with a reason:\n{}",
        undocumented.len(),
        undocumented.join("\n")
    );

    // Direction 2: every doc name still exists in the file its row names —
    // IDENT-FENCED, not substring (raw contains let a deleted `WS` pass
    // forever on `SR_WS`, and `TAPE` on its own comment mentions; review
    // finding). A deletion whose name survives only in prose is still
    // caught unless the prose uses the exact identifier, which reads as a
    // doc bug worth a failure anyway.
    let mut stale: Vec<String> = Vec::new();
    for (path, name) in &doc_entries {
        assert!(
            path.contains("/src/"),
            "inventory row path `{path}` is not a crate-relative src path — \
             a bare filename would suffix-match the wrong file"
        );
        let full = root.join("crates").join(path);
        let ok = std::fs::read_to_string(&full)
            .map(|s| contains_ident(&s, name))
            .unwrap_or(false);
        if !ok {
            stale.push(format!("  {name} (listed for {path})"));
        }
    }
    assert!(
        stale.is_empty(),
        "{} inventory entr(ies) no longer exist where the doc says — the \
         static was renamed, moved, or deleted; update the table in the same \
         PR:\n{}",
        stale.len(),
        stale.join("\n")
    );
}

/// Anti-vacuity: the scanner itself finds a healthy population (the doc had
/// ~46 statics at refresh time; the floor is set below with room to shrink
/// through real migrations, and is meant to be RAISED never lowered).
#[test]
fn the_scanner_is_not_scanning_nothing() {
    let root = repo_root();
    let mut files = Vec::new();
    rs_files(&root.join("crates/nsl-runtime/src"), &mut files);
    let n: usize = files
        .iter()
        .map(|f| scan_thread_local_statics(&std::fs::read_to_string(f).unwrap()).len())
        .sum();
    assert!(
        n >= 25,
        "only {n} thread-local statics found in nsl-runtime — the declaration \
         style must have changed and the inventory gate is now near-vacuous"
    );
}
