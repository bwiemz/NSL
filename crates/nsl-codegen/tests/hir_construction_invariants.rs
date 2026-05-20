//! Integration tests for HIR construction invariants per M57 v1 spec §3.4.

use nsl_codegen::hir::*;
use nsl_codegen::hir::module::{HirBuilderError, HirNode};

#[test]
fn invariant_1_single_driver_enforced_at_add_node() {
    let mut m = HirModule::new("inv1");
    let wid = WireId::fresh();
    let m1 = nsl_codegen::hir::nodes::Mul {
        a: SignalRef::port("a"), b: SignalRef::port("b"),
        out: wid, a_width: 8, b_width: 8, out_width: 16,
    };
    let m2 = nsl_codegen::hir::nodes::Mul {
        a: SignalRef::port("c"), b: SignalRef::port("d"),
        out: wid, a_width: 8, b_width: 8, out_width: 16,
    };
    m.add_node(HirNode::Mul(m1)).unwrap();
    let err = m.add_node(HirNode::Mul(m2)).unwrap_err();
    assert!(matches!(err, HirBuilderError::DuplicateDriver(_, _)));
}

#[test]
fn invariant_2_loops_have_constant_bounds_by_type() {
    // GenerateFor::lo, hi are i64. Cannot accept SignalRef. This is a compile-
    // time property; the test asserts the type signature compiles.
    let g = nsl_codegen::hir::control::GenerateFor::new(0, 128, vec![]);
    assert_eq!(g.hi, 128);
    // The following would not compile:
    //     GenerateFor::new(SignalRef::port("x"), 128, vec![]);
}

#[test]
fn invariant_3_generate_if_requires_both_branches() {
    use nsl_codegen::hir::control::{ConstExpr, GenerateIf};
    // Both branches required, but else may be empty Vec
    let g = GenerateIf::new(ConstExpr::True, vec![], vec![]);
    assert_eq!(g.else_body.len(), 0);
}

#[test]
fn invariant_6_max0_is_mux_not_case() {
    // §3.4 invariant 6: runtime branching is combinational mux only.
    // Max0's template is `($signed(a) > 0) ? a : '0` — a mux, not a case.
    // This test exists as a structural reminder; the actual enforcement is
    // template-side in PR 3.
    let _m = nsl_codegen::hir::nodes::Max0::new(SignalRef::port("x"), 32);
}
