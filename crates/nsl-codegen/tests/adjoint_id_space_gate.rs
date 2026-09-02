//! Static gate: adjoint tapes may only be renumbered through
//! [`nsl_codegen::wengert::renumber_adjoint_ops`].
//!
//! The primal/adjoint id split is only as strong as its narrowest hole. It
//! took five separate `for (i, op) in ops.iter_mut().enumerate() { op.id =
//! i as u32 }` loops (three in `ccr.rs`, two in `source_ad.rs`) to hold the
//! old numbering; a sixth added later would put adjoint ops back in the
//! primal id range with no test failing, because every functional gate in
//! the tree passes either way — the collision only shows up as a claim
//! table matching the wrong op, and only when CSHA, CCR and an unclaimed
//! SDPA coincide.
//!
//! So this is a source gate, not a behavioural one: it reads the codegen
//! sources and refuses raw positional id assignment outside the sanctioned
//! helper. A `debug_assert!` would be a release no-op and CI ships release
//! builds; a runtime assert only fires on the paths a test happens to
//! exercise. Scanning the source catches the site that was never run.

use std::path::{Path, PathBuf};

fn codegen_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` file under `crates/nsl-codegen/src`, recursively.
fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            out.extend(rust_sources(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out
}

/// Strip comments so prose QUOTING the forbidden shape (this file's own
/// rationale lives in several such comments) is not a hit.
///
/// Handles `//` line comments and `/* */` block comments. Block comments
/// matter in the false-POSITIVE direction: a commented-out `op.id = i as
/// u32;` inside `/* */` would fail the gate with nothing wrong in the tree.
/// Crude but sufficient — no string literal in this crate contains a comment
/// opener followed by an id assignment.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut in_block = false;
    while i < bytes.len() {
        if in_block {
            if bytes[i..].starts_with(b"*/") {
                in_block = false;
                i += 2;
            } else {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            continue;
        }
        if bytes[i..].starts_with(b"/*") {
            in_block = true;
            i += 2;
            continue;
        }
        if bytes[i..].starts_with(b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

#[test]
fn no_raw_positional_op_id_assignment_outside_the_helper() {
    // `op.id = <anything> as u32` / `as OpId` — the positional-renumber
    // shape. The helper itself writes `op.id = adjoint_op_id(i)`, which
    // has no cast and so does not match.
    let mut offenders: Vec<String> = Vec::new();
    let mut scanned_adjoint_files = 0usize;

    for path in rust_sources(&codegen_src()) {
        let name = path
            .file_name()
            .expect("file name")
            .to_string_lossy()
            .to_string();
        // `wengert.rs` DEFINES the helper; it is allowed to assign ids.
        if name == "wengert.rs" {
            continue;
        }
        let struct_scope = ADJOINT_CONSTRUCTING_FILES.contains(&name.as_str());
        if struct_scope {
            scanned_adjoint_files += 1;
        }
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let code = strip_comments(&src);
        for (lineno, line) in code.lines().enumerate() {
            if regex_lite_positional_assign_in(line, struct_scope) {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.file_name().expect("file name").to_string_lossy(),
                    lineno + 1,
                    line.trim()
                ));
            }
        }
    }

    // The scoped rule is only worth anything if its scope resolved. A renamed
    // or deleted `ccr.rs` would otherwise silently drop the struct-literal
    // half of this gate while the test stayed green.
    assert_eq!(
        scanned_adjoint_files,
        ADJOINT_CONSTRUCTING_FILES.len(),
        "ADJOINT_CONSTRUCTING_FILES names {:?} but only {scanned_adjoint_files} \
         of them exist under src/ — the struct-literal rule is not being \
         applied where it was declared to be",
        ADJOINT_CONSTRUCTING_FILES
    );

    assert!(
        offenders.is_empty(),
        "raw positional op-id assignment found — renumber adjoint tapes \
         through `wengert::renumber_adjoint_ops` so ids keep the adjoint \
         base, or the primal claim tables can alias these ops:\n  {}",
        offenders.join("\n  ")
    );
}

/// Does `rhs` convert an index-typed value into an `OpId`?
///
/// Covers the cast form and the fallible-conversion forms. A review of the
/// first version of this gate found it accepted `u32::try_from(i).unwrap()`
/// and `i.try_into().unwrap()`, which renumber exactly as `as u32` does.
fn converts_to_op_id(rhs: &str) -> bool {
    ["as u32", "as OpId", "as crate::wengert::OpId", "as _"]
        .iter()
        .any(|c| rhs.contains(c))
        || rhs.contains("u32::try_from")
        || rhs.contains("try_into")
        || rhs.contains("OpId::from")
}

/// A positional id write, in either shape the tree uses.
///
/// Two shapes, because the first version of this gate only knew one and the
/// review found a LIVE example of the other: `ccr.rs`'s early-free list built
/// `WengertOp { id: i as u32, .. }` in a struct literal, which no amount of
/// `.id =` matching sees.
///
/// * **assignment** — `op.id = <expr> as u32`
/// * **struct literal** — `id: <expr> as u32,`
fn regex_lite_positional_assign(line: &str) -> bool {
    regex_lite_positional_assign_in(line, true)
}

/// Files that CONSTRUCT ops destined for an adjoint tape, where a
/// struct-literal `id:` computed from a position is a renumber in disguise.
///
/// Scoped, because the struct-literal form alone cannot tell adjoint minting
/// from PRIMAL minting — and primal minting is correct and ubiquitous:
/// `WengertExtractor` writes `id: self.list.ops.len() as u32` at twenty-odd
/// sites, every one of them right. Flagging those would need a twenty-entry
/// allowlist, and an allowlist that large outlives the reason for its
/// entries (the failure `pass_registry.rs` deleted its `status` field to
/// avoid).
///
/// `source_ad.rs` is deliberately NOT here: its adjoint ops are all minted
/// through `AdjointGenerator::next_op`, never by struct literal, so the
/// assignment rule below is the whole story for it.
///
/// The residual gap — a NEW file constructing adjoint ops with hand-computed
/// ids — is covered at runtime instead, by the `assert_ids_in_space` entry
/// checks on all four adjoint mutators (`ccr::apply_to_adjoint`,
/// `ccr::splice_decompress`, `ccr::insert_adjoint_last_use_frees`, and the
/// two `source_ad` fusers).
const ADJOINT_CONSTRUCTING_FILES: &[&str] = &["ccr.rs"];

fn regex_lite_positional_assign_in(line: &str, struct_literal_scope: bool) -> bool {
    // Struct-literal field init. Requires a conversion on the RHS so plain
    // `id: op.id,` (a copy) and `id: ADJOINT_ID_BASE,` (the sanctioned
    // placeholder) do not match.
    if struct_literal_scope {
        let t = line.trim_start();
        if let Some(rhs) = t.strip_prefix("id:")
            && converts_to_op_id(rhs)
        {
            return true;
        }
    }

    // Assignment. Scan EVERY `.id` on the line, not just the first: the
    // review found that `let prev = op.id; op.id = i as u32;` defeated a
    // `split_once` matcher, because the first `.id`'s RHS is `;`.
    let mut hay = line;
    while let Some(pos) = hay.find(".id") {
        let rest = hay[pos + 3..].trim_start();
        if let Some(rhs) = rest.strip_prefix('=') {
            // `==` is a comparison, not an assignment.
            if !rhs.starts_with('=') && converts_to_op_id(rhs) {
                return true;
            }
        }
        hay = &hay[pos + 3..];
    }
    false
}

/// The gate is only meaningful if its matcher actually recognizes the
/// shape it forbids. Pin that directly — a matcher that matches nothing
/// makes the test above pass forever.
#[test]
fn the_matcher_recognizes_the_shape_it_forbids() {
    for line in [
        "        op.id = i as u32;",
        "op.id = idx as u32;",
        "    o.id = (i + 1) as OpId;",
        // Struct-literal minting — the shape the first version of this gate
        // missed while `ccr.rs` contained a live example.
        "            id: i as u32,",
        "    id: (n + 1) as OpId,",
        // Fallible conversions renumber exactly as a cast does.
        "        op.id = u32::try_from(i).unwrap();",
        "        op.id = i.try_into().unwrap();",
        "        op.id = i as _;",
        // An earlier `.id` READ on the line must not blind the scan.
        "        let prev = op.id; op.id = i as u32;",
    ] {
        assert!(
            regex_lite_positional_assign(line),
            "matcher missed a positional renumber: {line:?}"
        );
    }
    for line in [
        "        op.id = adjoint_op_id(i);",
        "        if op.id == other.id {",
        "    let x = op.id;",
        "    id: crate::wengert::ADJOINT_ID_BASE,",
        // A plain copy is not a renumber.
        "                id: op.id,",
        "            id: crate::wengert::adjoint_op_id(i),",
    ] {
        assert!(
            !regex_lite_positional_assign(line),
            "matcher false-positived on: {line:?}"
        );
    }
}

/// The comment stripper must not blind the gate to real code that has a
/// trailing comment.
#[test]
fn stripping_comments_does_not_hide_a_real_assignment() {
    let src = "op.id = i as u32; // renumbered\n\
               // op.id = i as u32;\n\
               /* op.id = i as u32; */\n\
               /* multi\n   op.id = i as u32;\n   line */\n";
    let code = strip_comments(src);
    let hits = code
        .lines()
        .filter(|l| regex_lite_positional_assign(l))
        .count();
    assert_eq!(
        hits, 1,
        "the code line must survive; the line comment and both block \
         comments must not. Stripped:\n{code}"
    );
}
