//! Item 8 (dispatcher unification): the Muon-route contract, pinned on both
//! sides of the FFI.
//!
//! The rule "a parameter is Muon-stepped iff `route_flag == 0 && rank == 2`"
//! is expressed in three places that CANNOT reference each other:
//! codegen's `emit_muon_route_predicate` (Cranelift IR), the runtime's
//! `nsl_muon_step_batch` filter (Rust), and the stdlib's `muon_step` branch
//! (NSL). The codegen side now has ONE definition; this gate pins
//! (a) that the emission sites actually use it, and (b) the runtime filter's
//! load-bearing shape: exactly two graceful skips (route, ndim), everything
//! past them an abort.
//!
//! Why (b) matters: codegen's batch-skip site jumps over params the batch
//! call is expected to step. If a future edit turns the filter's device /
//! dtype / contiguity / empty-matrix abort into a graceful `continue`
//! (exactly the shape of `fase_multi_impl`'s demotion arm), those params are
//! silently DROPPED — never stepped by anyone — and the loss curve just
//! degrades. The asymmetry (skip vs abort) is the invariant, not the
//! predicate text.

use std::path::Path;

fn read(rel: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// (a) The four codegen emission sites route through the ONE helper, and
/// the raw predicate spelling exists only inside it.
///
/// The fourth site is item C's ZeRO-3 deferred moment fill
/// (`emit_deferred_moment_fill`): under `--zero-stage 3` the moment lists
/// are filled at the window register belt instead of at train setup, so the
/// v-allocation route gate has to be re-emitted there. It is a SECOND
/// v-allocation site, not a new kind of decision — which is exactly why it
/// must call the helper rather than re-derive the De Morgan inverse.
#[test]
fn codegen_sites_share_the_one_predicate() {
    let stmt = read("src/stmt.rs");
    let calls = stmt.matches("self.emit_muon_route_predicate(").count();
    assert_eq!(
        calls, 4,
        "expected exactly 4 emission sites calling emit_muon_route_predicate \
         (batch-skip, resident-momentum, v-allocation, zero3 deferred \
         v-allocation); found {calls}. A new site must call the helper — and \
         update this count consciously — not hand-spell the predicate"
    );
    // The raw spelling appears exactly once: inside the helper itself.
    let raw = stmt.matches("band(is_muon, is_r2)").count();
    assert_eq!(
        raw, 1,
        "the raw `route==0 && rank==2` spelling must live only inside \
         emit_muon_route_predicate; found {raw} occurrence(s)"
    );
}

/// (b) The runtime filter keeps its skip/abort asymmetry: exactly two
/// `continue`s (route, ndim) before the launch, and hard aborts for the
/// operand-shape violations.
#[test]
fn runtime_filter_keeps_the_skip_abort_asymmetry() {
    let src = read("../nsl-runtime/src/muon_batch.rs");
    // Bound the scan to nsl_muon_step_batch: from the fn to the EARLIEST
    // following top-of-line item, whichever spelling it uses (review
    // finding: preferring "\nfn " over "\npub" regardless of position
    // silently swallowed everything after the fn if the next item ever
    // became pub — including the test module's own continues/aborts).
    let start = src
        .find("fn nsl_muon_step_batch")
        .expect("nsl_muon_step_batch not found — renamed? Update this gate");
    let body = &src[start..];
    let next_fn = body[1..].find("\nfn ");
    let next_pub = body[1..].find("\npub");
    let end = match (next_fn, next_pub) {
        (Some(a), Some(b)) => a.min(b) + 1,
        (Some(a), None) | (None, Some(a)) => a + 1,
        (None, None) => body.len(),
    };
    let filter = &body[..end];
    // Bound proof: a mis-extracted span that ran into the test module would
    // carry its own continues/aborts and pass by coincidence.
    assert!(
        !filter.contains("mod tests") && !filter.contains("#[cfg(test)]"),
        "the extracted span ran past the function into the test module — \
         the bounding heuristic no longer matches the file's layout"
    );

    let continues = filter.matches("continue;").count();
    assert_eq!(
        continues, 2,
        "nsl_muon_step_batch's filter must have EXACTLY two graceful skips \
         (route != 0, ndim != 2); found {continues}. A third skip silently \
         drops params codegen's batch-skip site already jumped over — if a \
         shape class must degrade, it has to abort here and be excluded at \
         COMPILE time in emit_muon_route_predicate instead"
    );
    assert!(
        filter.contains("route != 0") || filter.contains("route == 0"),
        "the route test vanished from the filter:\n{filter:.400}"
    );
    // Count aborts strictly AFTER the last graceful skip — the fn also
    // aborts on bad ns_steps BEFORE the filter, and counting it let a
    // single post-filter abort→return conversion slip through (review
    // finding: exactly the failure class this gate exists to catch).
    let after_last_skip = filter
        .rfind("continue;")
        .map(|i| &filter[i..])
        .unwrap_or(filter);
    assert!(
        after_last_skip.matches("std::process::abort()").count() >= 2,
        "the operand-shape violations past the route/ndim tests must ABORT, \
         never skip (the skip/abort asymmetry is the contract) — expected \
         both the device/dtype/contiguity abort and the empty-matrix abort \
         after the last graceful skip"
    );
}
