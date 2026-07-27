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

/// Parse `"a" | "b" => <verdict>` match arms out of a source region.
///
/// ONE parser for both the dispatcher and the ownership table, because they
/// have the same syntax and had different bugs when written separately.
///
/// Arms wrap. rustfmt splits a long arm across lines and puts the leading `|`
/// on the continuation, so a line-at-a-time parser that requires `=>` on the
/// same line as the name drops every name on a wrapped arm. That is not
/// hypothetical: the first version of this file required both, and rewriting
/// `"contiguous" =>` as `"contiguous"\n | "materialize" =>` made BOTH names
/// invisible — `contiguous` vanished from the dispatcher set and a brand-new
/// unclassified method was never noticed, with all four gates still green. A
/// reformat could have silently disarmed the gate.
///
/// Returns `(name, Some(owned))` for a resolved arm. `None` cannot escape:
/// pending names are resolved when the `=>` line arrives, and any left over at
/// the end are returned as `None` so callers can fail loudly rather than
/// inherit a default.
fn parse_match_arms(body: &str, verdict_of: impl Fn(&str) -> bool) -> Vec<(String, Option<bool>)> {
    let indent_of = |l: &str| l.len() - l.trim_start().len();
    // Arm HEADS sit at one fixed indentation; string literals inside arm BODIES
    // sit deeper. Without this, error-message and runtime-function-name literals
    // get scraped as if they were method names — dropping the `=>`-on-the-same-
    // line requirement (needed for wrapped arms) removed the accident that used
    // to filter them out. Take the shallowest `"…" =>` line as the arm indent.
    let arm_indent = body
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with('"') && l.contains("=>")
        })
        .map(indent_of)
        .min()
        .unwrap_or(0);

    let mut out: Vec<(String, Option<bool>)> = Vec::new();
    for line in body.lines() {
        let t = line.trim();
        // A name-bearing line starts with a quote or a continuation bar...
        if !t.starts_with('"') && !t.starts_with("| \"") {
            continue;
        }
        // ...and sits at the arm indentation, not inside an arm body.
        if indent_of(line) != arm_indent {
            continue;
        }
        let (names_part, verdict) = match t.split_once("=>") {
            Some((n, v)) => (n, Some(verdict_of(v))),
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
        for n in names {
            out.push((n, verdict));
        }
        if let Some(v) = verdict {
            // Resolve every name still waiting from earlier lines of THIS arm.
            // Walking backwards and stopping at the first resolved entry keeps
            // this arm's continuation lines from reaching into a previous arm —
            // the bug in the `const PENDING: bool = false` version, where a
            // sentinel was indistinguishable from a real `false` verdict and a
            // new wrapped arm retroactively flipped every earlier `false` entry
            // to `true`, failing with a use-after-free warning about
            // perfectly correct source.
            for entry in out.iter_mut().rev() {
                if entry.1.is_some() {
                    continue;
                }
                entry.1 = Some(v);
            }
        }
    }
    out
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
    let mut methods: Vec<String> = Vec::new();
    for (name, _) in parse_match_arms(&src[start..end], |_| true) {
        if !methods.contains(&name) {
            methods.push(name);
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
    parse_match_arms(&src[start..end], |v| v.contains("true"))
        .into_iter()
        .map(|(n, v)| {
            (
                n.clone(),
                v.unwrap_or_else(|| {
                    panic!("`{n}` was parsed with no verdict — the table's arm syntax has changed")
                }),
            )
        })
        .collect()
}

#[test]
fn every_tensor_method_is_classified() {
    // THE gate. A method the dispatcher can lower but the table does not name
    // falls through to `None` and is treated as non-owning — which is exactly
    // how the shape/view family leaked for as long as it did.
    let dispatchable = dispatchable_tensor_methods();
    // Floor set just under the 14 arms present when this was written. It has
    // to bite BEFORE a parser regression can hide a method: the previous floor
    // of 12 tolerated two silently-dropped arms, which is exactly how many a
    // single wrapped arm removes.
    assert!(
        dispatchable.len() >= 14,
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
