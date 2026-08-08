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

/// Strip `//` line comments so a doc comment QUOTING the forbidden shape
/// (this file's own rationale lives in several of them) is not a hit.
/// Crude but sufficient: no string literal in this crate contains `//`
/// followed by an `op.id =` assignment.
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn no_raw_positional_op_id_assignment_outside_the_helper() {
    // `op.id = <anything> as u32` / `as OpId` — the positional-renumber
    // shape. The helper itself writes `op.id = adjoint_op_id(i)`, which
    // has no cast and so does not match.
    let needle = regex_lite_positional_assign;
    let mut offenders: Vec<String> = Vec::new();

    for path in rust_sources(&codegen_src()) {
        // `wengert.rs` DEFINES the helper; it is allowed to assign ids.
        if path.file_name().is_some_and(|n| n == "wengert.rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let code = strip_line_comments(&src);
        for (lineno, line) in code.lines().enumerate() {
            if needle(line) {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.file_name().expect("file name").to_string_lossy(),
                    lineno + 1,
                    line.trim()
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "raw positional op-id assignment found — renumber adjoint tapes \
         through `wengert::renumber_adjoint_ops` so ids keep the adjoint \
         base, or the primal claim tables can alias these ops:\n  {}",
        offenders.join("\n  ")
    );
}

/// `op.id = <expr> as u32|OpId`, whitespace-tolerant, without pulling in a
/// regex dependency for one pattern.
fn regex_lite_positional_assign(line: &str) -> bool {
    let Some(rest) = line.split_once(".id").map(|(_, r)| r) else {
        return false;
    };
    let rest = rest.trim_start();
    let Some(rhs) = rest.strip_prefix('=') else {
        return false;
    };
    // `==` is a comparison, not an assignment.
    if rhs.starts_with('=') {
        return false;
    }
    rhs.contains("as u32") || rhs.contains("as OpId") || rhs.contains("as crate::wengert::OpId")
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
    let src = "op.id = i as u32; // renumbered\n// op.id = i as u32;\n";
    let code = strip_line_comments(src);
    let hits = code
        .lines()
        .filter(|l| regex_lite_positional_assign(l))
        .count();
    assert_eq!(hits, 1, "expected the code line to survive and the comment to not");
}
