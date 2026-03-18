//! M38b: Ownership-aware codegen — tracks linear/shared bindings for
//! consumption-point free emission and refcount elision.
//!
//! NOTE: The compiler reference (`&mut Compiler`) will be added when actual
//! Cranelift IR emission is wired in a follow-up. Currently infrastructure-only.

use std::collections::{HashMap, HashSet};
use nsl_ast::Symbol;

/// Borrow kind for active borrows.
#[derive(Debug, Clone, PartialEq)]
pub enum BorrowKind {
    Immutable { borrower: Symbol },
    Mutable { borrower: Symbol },
}

/// Per-function ownership metadata, consumed by the codegen phase.
#[derive(Debug, Clone, Default)]
pub struct FunctionOwnership {
    /// Parameters that are consumed (moved) by this function.
    pub linear_params: Vec<Symbol>,
    /// Parameters that are borrowed (&T or &mut T).
    pub borrowed_params: Vec<(Symbol, BorrowKind)>,
    /// Parameters that are @shared (refcounted).
    pub shared_params: Vec<Symbol>,
}

impl FunctionOwnership {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a parameter is linear (consumed by the function).
    pub fn is_linear(&self, sym: &Symbol) -> bool {
        self.linear_params.contains(sym)
    }

    /// Check if a parameter is shared.
    pub fn is_shared(&self, sym: &Symbol) -> bool {
        self.shared_params.contains(sym)
    }
}

/// Tracks ownership state during codegen for a single function.
/// Used to decide when to emit `nsl_tensor_free` and when to skip refcount ops.
#[derive(Debug, Default)]
pub struct OwnershipLowering {
    /// Bindings proven linear (consumed exactly once).
    pub linear_bindings: HashSet<Symbol>,
    /// Bindings annotated @shared or model fields.
    pub shared_bindings: HashSet<Symbol>,
    /// Active borrows: lender -> borrow kind.
    pub active_borrows: HashMap<Symbol, BorrowKind>,
}

impl OwnershipLowering {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if the binding should skip refcount operations.
    pub fn should_elide_refcount(&self, sym: &Symbol) -> bool {
        self.linear_bindings.contains(sym)
    }

    /// Returns true if the binding should be freed at consumption point
    /// rather than at scope exit.
    pub fn should_free_at_consumption(&self, sym: &Symbol) -> bool {
        self.linear_bindings.contains(sym) && !self.shared_bindings.contains(sym)
    }

    /// Register a binding as linear.
    pub fn mark_linear(&mut self, sym: Symbol) {
        self.linear_bindings.insert(sym);
    }

    /// Register a binding as shared.
    pub fn mark_shared(&mut self, sym: Symbol) {
        self.shared_bindings.insert(sym);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::Symbol;

    type Interner = string_interner::StringInterner<string_interner::backend::BucketBackend<string_interner::DefaultSymbol>>;

    fn make_sym(interner: &mut Interner, name: &str) -> Symbol {
        Symbol(interner.get_or_intern(name))
    }

    #[test]
    fn test_linear_elides_refcount() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);
        assert!(lowering.should_elide_refcount(&x));
        assert!(lowering.should_free_at_consumption(&x));
    }

    #[test]
    fn test_shared_does_not_elide() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let lowering = OwnershipLowering::new();
        assert!(!lowering.should_elide_refcount(&x));
        assert!(!lowering.should_free_at_consumption(&x));
    }

    #[test]
    fn test_shared_override_prevents_early_free() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);
        lowering.mark_shared(x);
        assert!(lowering.should_elide_refcount(&x));
        assert!(!lowering.should_free_at_consumption(&x));
    }

    #[test]
    fn test_function_ownership_defaults() {
        let ownership = FunctionOwnership::new();
        assert!(ownership.linear_params.is_empty());
        assert!(ownership.borrowed_params.is_empty());
        assert!(ownership.shared_params.is_empty());
    }

    #[test]
    fn test_function_ownership_query() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let y = make_sym(&mut interner, "y");
        let mut ownership = FunctionOwnership::new();
        ownership.linear_params.push(x);
        ownership.shared_params.push(y);
        assert!(ownership.is_linear(&x));
        assert!(!ownership.is_shared(&x));
        assert!(!ownership.is_linear(&y));
        assert!(ownership.is_shared(&y));
    }
}
