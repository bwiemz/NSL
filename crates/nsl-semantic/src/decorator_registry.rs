//! Milestone A: the decorator namespace, closed.
//!
//! Before this module, `Decorator { name: Vec<Symbol> }` was an open string
//! namespace: `@totally_not_a_real_feature(alpha = 1)` above a train block
//! passed `nsl check` and built a byte-identical binary. Argument KEYS were
//! validated per-name (`@cpdt(mdoe = off)` errored), but the NAME itself was
//! not — a typo'd decorator name was the one misspelling the compiler never
//! caught. The only closed positions were inside `datatype` blocks and
//! `serve` blocks, both enforced by the parser before a `Decorator` node
//! exists; this registry closes everything else.
//!
//! # What a row claims
//!
//! Membership, not behaviour: "this name is part of the NSL surface, and
//! HERE is the component that reads it". The activation contract (owner +
//! witness) lives in `nsl_codegen::activation`; the drift gate in
//! `crates/nsl-semantic/tests/decorator_namespace_gate.rs` checks the
//! `read_by` breadcrumb still holds, so a consumer that moves or dies takes
//! this table with it (item-21 lesson: anchor names to CODE, and gate in the
//! direction tree -> registry).
//!
//! # The unimplemented list
//!
//! Names the DOCUMENTATION advertises with no implementation anywhere in the
//! tree get a per-name typed refusal instead of the generic unknown-name
//! error — "@tie_weights is not implemented" is actionable; "unknown
//! decorator @tie_weights" reads like a typo accusation for a name the spec
//! itself uses (deferral-must-refuse: the refusal names what was requested,
//! why it cannot be honoured, and what to do instead).

/// One name in the closed namespace and the code that reads it.
#[derive(Debug, Clone, Copy)]
pub struct KnownDecorator {
    /// The name as written after `@`, single segment.
    pub name: &'static str,
    /// Repo-relative breadcrumb of the primary consumer (validator or
    /// codegen reader). Gated: the file must still mention the name.
    pub read_by: &'static str,
}

const fn k(name: &'static str, read_by: &'static str) -> KnownDecorator {
    KnownDecorator { name, read_by }
}

/// Every decorator name the toolchain reads anywhere, from the 2026-08-15
/// empirical inventory: 34 validated in `checker/stmt.rs`, 13 in
/// `checker/model.rs`, 3 in other semantic modules, and 9 read only by
/// codegen. Names consumed by the parser before a `Decorator` node exists
/// (`pack`, `unpack`, `backward`, `*_ptx` in datatype blocks; `endpoint` in
/// serve blocks) are deliberately absent — their namespaces were already
/// closed at the parse sites. `layout` is deliberately NOT here: its
/// validator has no dispatch (see UNIMPLEMENTED_DECORATORS).
pub static KNOWN_DECORATORS: &[KnownDecorator] = &[
    // --- validated in nsl-semantic/src/checker/stmt.rs ---
    k("adapter", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("attention_sink", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("autotune", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("cep_prune", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("cep_search", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("cfie", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("checkpoint", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("cpdt", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("csha", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("deterministic", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("fase", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("flash_attention", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("freeze", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("fuse", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("fuse_graph", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("fused_kl_ce", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("fused_lm_ce", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("gqa", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("inspect", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("no_fuse", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("no_trace", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("paged_kv", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("pca", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("pure", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("rope", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("shape_assert", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("shared", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("test", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("trace_breakpoint", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("tree_mask", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("vmap", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("wggo", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("wggo_target", "crates/nsl-semantic/src/checker/stmt.rs"),
    k("wrga", "crates/nsl-semantic/src/checker/stmt.rs"),
    // --- validated in nsl-semantic/src/checker/model.rs ---
    k("context_parallel", "crates/nsl-semantic/src/checker/model.rs"),
    k("fp8_compute", "crates/nsl-semantic/src/checker/model.rs"),
    k("grammar", "crates/nsl-semantic/src/checker/model.rs"),
    k("kv_compress", "crates/nsl-semantic/src/checker/model.rs"),
    k("medusa", "crates/nsl-semantic/src/checker/model.rs"),
    k("moe", "crates/nsl-semantic/src/checker/model.rs"),
    k("multimodal", "crates/nsl-semantic/src/checker/model.rs"),
    k("perf_budget", "crates/nsl-semantic/src/checker/model.rs"),
    k("pipeline", "crates/nsl-semantic/src/checker/model.rs"),
    k("shard", "crates/nsl-semantic/src/checker/model.rs"),
    k("sparse", "crates/nsl-semantic/src/checker/model.rs"),
    k("speculative", "crates/nsl-semantic/src/checker/model.rs"),
    k("target", "crates/nsl-semantic/src/checker/model.rs"),
    // --- validated elsewhere in nsl-semantic ---
    k("auto_device_transfer", "crates/nsl-semantic/src/agent.rs"),
    k("export", "crates/nsl-semantic/src/export.rs"),
    k("pipeline_agent", "crates/nsl-semantic/src/agent.rs"),
    // --- read only by nsl-codegen ---
    k("fp4_compute", "crates/nsl-codegen/src/fp8.rs"),
    k("no_grad", "crates/nsl-codegen/src/compiler/declaration.rs"),
    k("param_role", "crates/nsl-codegen/src/compiler/collection.rs"),
    k("quantize", "crates/nsl-codegen/src/compiler/collection.rs"),
    k("real_time", "crates/nsl-codegen/src/wcet.rs"),
    k("search", "crates/nsl-codegen/src/cep_extract.rs"),
    k("wcet_budget", "crates/nsl-codegen/src/wcet.rs"),
    k("zk_lookup", "crates/nsl-codegen/src/zk/mod.rs"),
    k("zk_proof", "crates/nsl-codegen/src/zk/mod.rs"),
];

/// Documentation-advertised names with NO implementation anywhere in the
/// tree. Each gets a typed refusal that says so — being told a documented
/// name is "unknown" reads like a typo accusation; being told it is
/// unimplemented is actionable.
pub static UNIMPLEMENTED_DECORATORS: &[(&str, &str)] = &[
    (
        // Found by this registry's own drift gate: `validate_layout_decorator`
        // exists in sparse_layout.rs with unit tests, but NOTHING dispatches
        // it — no checker arm, no parser route. A validator without a
        // dispatch is an inert feature, so the name refuses rather than
        // silently ignoring the annotation.
        "layout",
        "@layout has validation logic (sparse_layout.rs) but is not dispatched \
         by the checker in this version, so it would be silently ignored; \
         remove the decorator",
    ),
    (
        "tie_weights",
        "@tie_weights is documented (spec/04-model-definition, wiki) but has no \
         implementation in this toolchain version; share weights by assigning \
         the same tensor to both fields, or remove the decorator",
    ),
    (
        "custom_vjp",
        "@custom_vjp appears in the spec's annotation grammar \
         (spec/03-automatic-differentiation) but is not implemented; use a \
         `datatype` block's @backward for custom gradients, or remove the \
         decorator",
    ),
    (
        "broadcast",
        "@broadcast is documentation prose (spec/02-tensor-type-system), not an \
         implemented decorator; broadcasting is inferred from shapes",
    ),
    (
        "mac_array",
        "@mac_array is a documented hardware intrinsic \
         (spec/09-hardware-abstraction) with no implementation; remove the \
         decorator",
    ),
    (
        "unchecked",
        "@unchecked is documentation prose (spec/02-tensor-type-system), not an \
         implemented decorator",
    ),
    (
        "torch",
        "@torch is documentation prose (spec/11-interoperability), not an \
         implemented decorator",
    ),
];

/// Is `name` in the closed namespace?
pub fn find(name: &str) -> Option<&'static KnownDecorator> {
    KNOWN_DECORATORS.iter().find(|d| d.name == name)
}

/// The refusal message for a documented-but-unimplemented name, if any.
pub fn unimplemented_refusal(name: &str) -> Option<&'static str> {
    UNIMPLEMENTED_DECORATORS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, msg)| *msg)
}

/// Nearest known name within edit distance 2, for did-you-mean. Prefers the
/// smallest distance; ties resolve to the first in registry order so the
/// suggestion is deterministic.
pub fn suggest(name: &str) -> Option<&'static str> {
    let mut best: Option<(usize, &'static str)> = None;
    for d in KNOWN_DECORATORS {
        let dist = edit_distance(name, d.name);
        if dist <= 2 && best.map_or(true, |(b, _)| dist < b) {
            best = Some((dist, d.name));
        }
    }
    best.map(|(_, n)| n)
}

/// Plain Levenshtein, small inputs only (decorator names are short; the
/// registry is 60 entries — no need for anything cleverer).
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_names_resolve_and_ghosts_do_not() {
        assert!(find("cpdt").is_some());
        assert!(find("fase").is_some());
        assert!(find("tie_weights").is_none(), "ghosts are not KNOWN");
        assert!(unimplemented_refusal("tie_weights").is_some());
        assert!(unimplemented_refusal("cpdt").is_none());
    }

    #[test]
    fn suggestion_catches_the_typo_class_that_motivated_the_close() {
        assert_eq!(suggest("cpdtt"), Some("cpdt"));
        assert_eq!(suggest("flash_atention"), Some("flash_attention"));
        assert_eq!(suggest("totally_not_a_real_feature"), None);
    }

    #[test]
    fn the_registry_has_no_duplicate_names() {
        let mut seen = std::collections::HashSet::new();
        for d in KNOWN_DECORATORS {
            assert!(seen.insert(d.name), "duplicate registry row: {}", d.name);
        }
        for (n, _) in UNIMPLEMENTED_DECORATORS {
            assert!(
                find(n).is_none(),
                "{n} cannot be both known and unimplemented"
            );
        }
    }
}
