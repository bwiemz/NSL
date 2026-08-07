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

/// (a) The three codegen emission sites route through the ONE helper, and
/// the raw predicate spelling exists only inside it.
#[test]
fn codegen_sites_share_the_one_predicate() {
    let stmt = read("src/stmt.rs");
    let calls = stmt.matches("self.emit_muon_route_predicate(").count();
    assert_eq!(
        calls, 3,
        "expected exactly 3 emission sites calling emit_muon_route_predicate \
         (batch-skip, resident-momentum, v-allocation); found {calls}. A new \
         site must call the helper — and update this count consciously — not \
         hand-spell the predicate"
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
    // Bound the scan to the admission loop of nsl_muon_step_batch: from the
    // fn to the grouping-key insertion that ends the filter.
    let start = src
        .find("fn nsl_muon_step_batch")
        .expect("nsl_muon_step_batch not found — renamed? Update this gate");
    let body = &src[start..];
    let end = body
        .find("fn ")
        .and_then(|_| body[1..].find("\nfn ").or_else(|| body[1..].find("\npub")))
        .map(|i| i + 1)
        .unwrap_or(body.len());
    let filter = &body[..end.min(body.len())];

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
    assert!(
        filter.matches("std::process::abort()").count() >= 2,
        "the operand-shape violations past the route/ndim tests must ABORT, \
         never skip (the skip/abort asymmetry is the contract)"
    );
}
