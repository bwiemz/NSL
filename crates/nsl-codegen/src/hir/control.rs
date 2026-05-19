//! Generate-time control flow per spec §3.2. Both nodes resolve at HIR →
//! Verilog lowering; no runtime cost. Runtime branching is the combinational
//! mux only (Max0's `?:`); §3.4 invariant 6.

use crate::hir::ids::GenvarId;
use crate::hir::module::HirNode;

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateFor {
    pub var: GenvarId,
    pub lo: i64,
    pub hi: i64,
    pub body: Vec<HirNode>,
    // Invariant: lo, hi are compile-time constants; hi > lo. Runtime-bounded
    // iteration is unrepresentable in v1's HIR.
}

impl GenerateFor {
    pub fn new(lo: i64, hi: i64, body: Vec<HirNode>) -> Self {
        assert!(hi > lo, "GenerateFor hi={} must exceed lo={}", hi, lo);
        Self { var: GenvarId::fresh(), lo, hi, body }
    }
}

/// Compile-time boolean for `GenerateIf::cond`. The codegen evaluates the
/// condition statically before HIR construction, so v1 only ever emits `True`
/// or `False`. A future milestone that wants symbolic / parameterized
/// conditions should introduce a richer expression type at that time
/// (deliberate YAGNI — spec §1.5 deferred-roadmap).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstExpr {
    True,
    False,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateIf {
    pub cond: ConstExpr,
    pub then_body: Vec<HirNode>,
    pub else_body: Vec<HirNode>,
}

impl GenerateIf {
    pub fn new(cond: ConstExpr, then_body: Vec<HirNode>, else_body: Vec<HirNode>) -> Self {
        Self { cond, then_body, else_body }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_for_constructs_with_valid_bounds() {
        let g = GenerateFor::new(0, 128, vec![]);
        assert_eq!(g.lo, 0);
        assert_eq!(g.hi, 128);
    }

    #[test]
    #[should_panic(expected = "must exceed lo")]
    fn generate_for_rejects_hi_le_lo() {
        let _ = GenerateFor::new(5, 5, vec![]);
    }

    #[test]
    fn generate_if_allows_empty_else() {
        let g = GenerateIf::new(ConstExpr::True, vec![], vec![]);
        assert_eq!(g.cond, ConstExpr::True);
    }
}
