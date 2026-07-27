//! Completeness gates for the tensor-method ownership table
//! (`expr::tensor_method_returns_owned_ref`, roadmap item 1 residual).
//!
//! The table decides whether a `tensor.method(...)` result is an owning
//! reference the caller must release. Getting it wrong is asymmetric and both
//! directions are silent:
//!
//! - **Missing an owning method** leaks. It never crashes, never fails a test,
//!   and shows up only as GPU memory that grows per call. This is what actually
//!   happened: the whole shape/view family was absent and every anonymous link
//!   of a `reshape(...).transpose(...)` chain stranded.
//! - **Wrongly marking an aliasing method owned** frees a buffer a live variable
//!   still owns — a use-after-free, which surfaces as corrupted numbers or a
//!   segfault far from the cause.
//!
//! The predecessor was a bare `matches!` allowlist whose own comment said *"this
//! allowlist is hand-maintained and drifts silently"*. These gates are what that
//! comment was asking for: the dispatcher is the source of truth, and a method
//! it can lower but the table does not classify is a hard failure.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/nsl-codegen has two ancestors")
        .to_path_buf()
}

fn advanced_rs() -> String {
    std::fs::read_to_string(repo_root().join("crates/nsl-codegen/src/expr/advanced.rs"))
        .expect("expr/advanced.rs readable")
}

fn expr_mod_rs() -> String {
    std::fs::read_to_string(repo_root().join("crates/nsl-codegen/src/expr/mod.rs"))
        .expect("expr/mod.rs readable")
}

/// Every method name the tensor-method dispatcher can lower.
///
/// Parsed from `compile_tensor_method_call`'s match arms rather than restated,
/// so the gate compares the table against the dispatcher itself and not against
/// a second hand-maintained list that could drift in lockstep.
fn dispatchable_tensor_methods() -> Vec<String> {
    let src = advanced_rs();
    let start = src
        .find("fn compile_tensor_method_call")
        .expect("compile_tensor_method_call still exists");
    // The function ends at the catch-all that reports an unknown method.
    let end = src[start..]
        .find(r#"unknown tensor method '.{method}()'"#)
        .map(|o| start + o)
        .expect("the unknown-method catch-all still exists");
    let body = &src[start..end];

    let mut methods = Vec::new();
    for line in body.lines() {
        let t = line.trim();
        // Match-arm heads look like `"name" => ...` or `"a" | "b" => ...`.
        if !t.starts_with('"') || !t.contains("=>") {
            continue;
        }
        let head = t.split("=>").next().unwrap_or("");
        for piece in head.split('|') {
            let p = piece.trim().trim_matches(',').trim();
            if let Some(name) = p.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                if !name.is_empty() && !methods.contains(&name.to_string()) {
                    methods.push(name.to_string());
                }
            }
        }
    }
    methods
}

/// The method names the ownership table classifies, and how.
///
/// Read out of the table's source because the function itself is `pub(crate)`
/// and not reachable from an integration test.
fn classified_methods() -> Vec<(String, bool)> {
    let src = expr_mod_rs();
    let start = src
        .find("pub(crate) fn tensor_method_returns_owned_ref")
        .expect("the ownership table still exists");
    let end = src[start..]
        .find("\n        _ => return None,")
        .map(|o| start + o)
        .expect("the table's unclassified fall-through still exists");
    let body = &src[start..end];

    let mut out: Vec<(String, bool)> = Vec::new();
    for line in body.lines() {
        let t = line.trim();
        if !t.starts_with('"') && !t.starts_with("| \"") {
            continue;
        }
        // An arm may wrap across lines; the verdict is on whichever line has
        // the `=>`. Collect names first, then attach the verdict when seen.
        let (names_part, verdict) = match t.split_once("=>") {
            Some((n, v)) => (n, Some(v.contains("true"))),
            None => (t, None),
        };
        let names: Vec<String> = names_part
            .split('|')
            .filter_map(|p| {
                let p = p.trim().trim_matches(',').trim();
                p.strip_prefix('"')
                    .and_then(|s| s.strip_suffix('"'))
                    .map(str::to_string)
            })
            .collect();
        match verdict {
            Some(v) => {
                // Attach this verdict to these names AND to any names still
                // waiting from previous wrapped lines.
                for (_, pending) in out.iter_mut().filter(|(_, p)| *p == PENDING) {
                    *pending = v;
                }
                for n in names {
                    out.push((n, v));
                }
            }
            None => {
                for n in names {
                    out.push((n, PENDING));
                }
            }
        }
    }
    out
}

/// Sentinel for "name parsed, verdict on a later line". `bool` has no third
/// state; a wrapped arm resolves to the verdict of the line carrying `=>`.
const PENDING: bool = false;

#[test]
fn every_tensor_method_is_classified() {
    // THE gate. A method the dispatcher can lower but the table does not name
    // falls through to `None` and is treated as non-owning — which is exactly
    // how the shape/view family leaked for as long as it did.
    let dispatchable = dispatchable_tensor_methods();
    assert!(
        dispatchable.len() >= 12,
        "only parsed {} dispatchable methods ({dispatchable:?}) — the match-arm \
         parser has stopped matching the dispatcher and this gate is vacuous",
        dispatchable.len()
    );

    let classified: Vec<String> = classified_methods().into_iter().map(|(n, _)| n).collect();
    let missing: Vec<&String> = dispatchable
        .iter()
        .filter(|m| !classified.contains(m))
        .collect();
    assert!(
        missing.is_empty(),
        "these tensor methods can be lowered but are not classified in \
         `tensor_method_returns_owned_ref`: {missing:?}.\n\
         Read the runtime implementation and add an entry. An unclassified \
         method is silently treated as non-owning, which leaks its result \
         whenever the call appears in nested position."
    );
}

#[test]
fn the_shape_and_view_family_is_classified_owning() {
    // Pins the actual fix. Each of these returns an owning reference:
    // `new_view_i64` allocates a fresh handle and bumps the root owner's
    // refcount; `expand` shares the data pointer and bumps the root; and
    // `contiguous` bumps before returning the receiver in its identity case.
    let classified = classified_methods();
    for m in [
        "reshape",
        "transpose",
        "unsqueeze",
        "expand",
        "contiguous",
        "select",
        "slice",
        "cumsum",
    ] {
        let entry = classified
            .iter()
            .find(|(n, _)| n == m)
            .unwrap_or_else(|| panic!("{m} is not classified at all"));
        assert!(
            entry.1,
            "{m} must be classified OWNING — its runtime implementation hands \
             back a counted reference, so failing to free it leaks one handle \
             per call in nested position"
        );
    }
}

#[test]
fn aliasing_methods_stay_non_owning() {
    // The other direction, and the dangerous one. `.clone()` can elide to the
    // receiver under FBIP and `.to()` hands back the receiver when the tensor is
    // already on the target device; NEITHER bumps a refcount. Marking either
    // owning would free a buffer a live variable still owns.
    let classified = classified_methods();
    for m in ["clone", "to", "item", "shape"] {
        let entry = classified
            .iter()
            .find(|(n, _)| n == m)
            .unwrap_or_else(|| panic!("{m} is not classified at all"));
        assert!(
            !entry.1,
            "{m} must stay NON-owning — it can alias the receiver without \
             taking a reference, so freeing its result is a use-after-free, \
             not a leak fix"
        );
    }
}

#[test]
fn the_table_is_the_only_tensor_method_ownership_decision() {
    // Guards against a second allowlist growing back alongside the table. The
    // predecessor's failure mode was a `matches!` list inline in the predicate;
    // if one reappears, the table stops being authoritative and the
    // completeness gate above no longer proves anything.
    let src = expr_mod_rs();
    let start = src
        .find("fn expr_result_is_owned_temporary")
        .expect("the predicate still exists");
    let end = src[start..]
        .find("\n    /// True when `expr` is a call whose result")
        .map(|o| start + o)
        .expect("the predicate's terminating doc comment still exists");
    let body = &src[start..end];

    let obj_ty_arm = body
        .find("if obj_ty.is_tensor()")
        .expect("the tensor-receiver arm still exists");
    let arm_tail = &body[obj_ty_arm..];
    let arm_end = arm_tail.find('}').unwrap_or(arm_tail.len());
    assert!(
        arm_tail[..arm_end].contains("tensor_method_returns_owned_ref"),
        "the tensor-receiver arm no longer defers to the ownership table — an \
         inline allowlist has grown back, which is the exact drift the table \
         exists to prevent"
    );
}
