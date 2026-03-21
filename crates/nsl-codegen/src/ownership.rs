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

    /// Determine the ownership action(s) for a tensor binding at a specific usage site.
    ///
    /// `sym`: the binding being used
    /// `is_consuming`: true if this use consumes (moves) the tensor
    /// `backward_access`: if inside a `grad` block, the tape op name for backward classification
    /// `debug_mode`: whether to emit poison values after moves
    pub fn decide(
        &self,
        sym: &Symbol,
        is_consuming: bool,
        backward_access: Option<&str>,
        debug_mode: bool,
    ) -> Vec<OwnershipDecision> {
        let mut decisions = Vec::new();

        // @shared bindings always use refcount management
        if self.shared_bindings.contains(sym) {
            decisions.push(OwnershipDecision::RefcountManaged);
            return decisions;
        }

        // Borrowed bindings need no ownership action
        if self.active_borrows.contains_key(sym) {
            decisions.push(OwnershipDecision::BorrowedNoAction);
            return decisions;
        }

        // Linear binding being consumed
        if self.linear_bindings.contains(sym) && is_consuming {
            // Check if the autodiff tape needs this tensor's data
            if let Some(op_name) = backward_access {
                use nsl_semantic::ownership_autodiff::{classify_backward_access, BackwardAccess};
                let access = classify_backward_access(op_name);
                match access {
                    BackwardAccess::ShapeOnly => {
                        // Safe to free immediately — backward only needs shape
                        decisions.push(OwnershipDecision::FreeAtConsumption);
                    }
                    BackwardAccess::DataRequired => {
                        // Tape saved_* holds a reference — don't free
                        decisions.push(OwnershipDecision::TapeHoldsReference);
                    }
                    BackwardAccess::AuxDataRequired => {
                        // Aux data is owned by tape, input tensor can be freed
                        decisions.push(OwnershipDecision::FreeAtConsumption);
                    }
                }
            } else {
                // Not in grad block — always free at consumption
                decisions.push(OwnershipDecision::FreeAtConsumption);
            }

            // Debug mode: poison the source slot
            if debug_mode {
                decisions.push(OwnershipDecision::PoisonAfterMove);
            }

            return decisions;
        }

        // Linear binding NOT being consumed (e.g., borrowed use)
        if self.linear_bindings.contains(sym) {
            decisions.push(OwnershipDecision::BorrowedNoAction);
            return decisions;
        }

        // Default: refcount managed (no ownership info available)
        decisions.push(OwnershipDecision::RefcountManaged);
        decisions
    }
}

/// Decision for how to handle a tensor at a usage site.
#[derive(Debug, Clone, PartialEq)]
pub enum OwnershipDecision {
    /// Tensor is linear and being consumed — emit nsl_tensor_free at this point.
    /// No refcount inc/dec needed.
    FreeAtConsumption,
    /// Tensor is linear but used in a DataRequired autodiff op — tape holds reference,
    /// do NOT free early. The tape's saved_* field keeps it alive.
    TapeHoldsReference,
    /// Tensor is @shared — use normal refcount inc/dec (current behavior).
    RefcountManaged,
    /// Tensor is borrowed — no ownership action needed (caller retains ownership).
    BorrowedNoAction,
    /// Debug mode: after consumption, zero the source pointer slot for null-on-reuse detection.
    PoisonAfterMove,
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

    // --- M38b: OwnershipDecision + decide() tests ---

    #[test]
    fn linear_consuming_no_grad_frees() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);

        let decisions = lowering.decide(&x, true, None, false);
        assert_eq!(decisions, vec![OwnershipDecision::FreeAtConsumption]);
    }

    #[test]
    fn linear_consuming_shape_only_op_frees() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);

        let decisions = lowering.decide(&x, true, Some("Add"), false);
        assert_eq!(decisions, vec![OwnershipDecision::FreeAtConsumption]);
    }

    #[test]
    fn linear_consuming_data_required_op_tape_holds() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);

        let decisions = lowering.decide(&x, true, Some("MatMul"), false);
        assert_eq!(decisions, vec![OwnershipDecision::TapeHoldsReference]);
    }

    #[test]
    fn linear_consuming_debug_adds_poison() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);

        let decisions = lowering.decide(&x, true, None, true);
        assert_eq!(decisions, vec![
            OwnershipDecision::FreeAtConsumption,
            OwnershipDecision::PoisonAfterMove,
        ]);
    }

    #[test]
    fn shared_always_refcounted() {
        let mut interner = Interner::new();
        let w = make_sym(&mut interner, "W");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_shared(w);

        let decisions = lowering.decide(&w, true, None, false);
        assert_eq!(decisions, vec![OwnershipDecision::RefcountManaged]);
    }

    #[test]
    fn borrowed_no_action() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let b = make_sym(&mut interner, "ref_x");
        let mut lowering = OwnershipLowering::new();
        lowering.active_borrows.insert(x, BorrowKind::Immutable { borrower: b });

        let decisions = lowering.decide(&x, false, None, false);
        assert_eq!(decisions, vec![OwnershipDecision::BorrowedNoAction]);
    }

    #[test]
    fn elide_refcount_for_linear() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let w = make_sym(&mut interner, "W");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);
        lowering.mark_shared(w);

        assert!(lowering.should_elide_refcount(&x));
        assert!(!lowering.should_elide_refcount(&w));
        assert!(lowering.should_free_at_consumption(&x));
        assert!(!lowering.should_free_at_consumption(&w));
    }

    #[test]
    fn mark_linear_and_shared() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let w = make_sym(&mut interner, "W");
        let mut lowering = OwnershipLowering::new();

        lowering.mark_linear(x);
        lowering.mark_shared(w);

        assert!(lowering.linear_bindings.contains(&x));
        assert!(lowering.shared_bindings.contains(&w));
    }

    #[test]
    fn autodiff_classify_coverage() {
        use nsl_semantic::ownership_autodiff::{classify_backward_access, BackwardAccess};
        assert_eq!(classify_backward_access("Add"), BackwardAccess::ShapeOnly);
        assert_eq!(classify_backward_access("MatMul"), BackwardAccess::DataRequired);
        assert_eq!(classify_backward_access("Dropout"), BackwardAccess::AuxDataRequired);
        assert_eq!(classify_backward_access("UnknownOp"), BackwardAccess::DataRequired);
    }

    #[test]
    fn linear_non_consuming_is_borrow() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);

        let decisions = lowering.decide(&x, false, None, false);
        assert_eq!(decisions, vec![OwnershipDecision::BorrowedNoAction]);
    }

    // ── Task 5: Borrowed tensors in grad() scope — codegen decisions ─

    #[test]
    fn borrowed_in_grad_scope_no_action() {
        // A borrowed tensor used in a tape-recording op: BorrowedNoAction.
        // The tape records the raw pointer (same as the underlying owned tensor),
        // and no free/refcount action is needed — the caller owns the tensor and
        // keeps it alive for the borrow's lifetime.
        let mut interner = Interner::new();
        let w = make_sym(&mut interner, "w");
        let w_ref = make_sym(&mut interner, "w_ref");
        let mut lowering = OwnershipLowering::new();

        // w is linear; w_ref is an active borrow of w
        lowering.mark_linear(w);
        lowering.active_borrows.insert(w, BorrowKind::Immutable { borrower: w_ref });

        // Using w in a grad-scope DataRequired op (MatMul) — BorrowedNoAction
        // because the borrow map entry for w takes priority.
        let decisions = lowering.decide(&w, false, Some("MatMul"), false);
        assert_eq!(decisions, vec![OwnershipDecision::BorrowedNoAction]);
    }

    #[test]
    fn borrowed_in_grad_scope_no_action_shape_only() {
        // Borrowed tensor in a ShapeOnly op (Add): still BorrowedNoAction.
        // Both ShapeOnly and DataRequired ops get BorrowedNoAction for borrows —
        // the caller keeps the tensor alive either way.
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let x_ref = make_sym(&mut interner, "x_ref");
        let mut lowering = OwnershipLowering::new();

        lowering.mark_linear(x);
        lowering.active_borrows.insert(x, BorrowKind::Immutable { borrower: x_ref });

        let decisions = lowering.decide(&x, false, Some("Add"), false);
        assert_eq!(decisions, vec![OwnershipDecision::BorrowedNoAction]);
    }

    #[test]
    fn no_grad_on_borrowed_param_frees_at_consumption() {
        // @no_grad context: backward_access = None. A linear binding being consumed
        // with no_grad active should FreeAtConsumption, same as without grad scope.
        // This documents that @no_grad (backward_access = None) + linear consume
        // = always free at consumption, regardless of borrow status of other tensors.
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();

        lowering.mark_linear(x);

        // backward_access = None simulates @no_grad context
        let decisions = lowering.decide(&x, true, None, false);
        assert_eq!(decisions, vec![OwnershipDecision::FreeAtConsumption]);
    }

    // ── Task 6: Model methods — self pointer is never linear ─────────
    //
    // Model `self` is a raw I64 pointer in the Cranelift IR. It is never
    // registered in OwnershipLowering (only tensor-typed bindings are).
    // Therefore decide() is never called for self — no ownership action.
    // The following tests verify that shared model weights (the dominant
    // pattern for model fields) behave correctly across multiple forward calls.

    #[test]
    fn model_shared_weight_multiple_forward_calls() {
        // Model weights are @shared — they can be used in forward() any number
        // of times without being consumed. Each call gets RefcountManaged.
        let mut interner = Interner::new();
        let weight = make_sym(&mut interner, "weight");
        let mut lowering = OwnershipLowering::new();

        // Model fields are @shared by convention
        lowering.mark_shared(weight);

        // Three forward() calls each use weight (e.g., MatMul in forward pass)
        for _ in 0..3 {
            let decisions = lowering.decide(&weight, true, Some("MatMul"), false);
            assert_eq!(
                decisions,
                vec![OwnershipDecision::RefcountManaged],
                "shared weight should be RefcountManaged on every forward call"
            );
        }
    }

    #[test]
    fn model_forward_borrow_then_optimizer_frees() {
        // Training loop pattern:
        //   1. weight borrowed for forward (BorrowedNoAction while borrow is active)
        //   2. Borrow released after forward
        //   3. Optimizer step: linear weight consumed → FreeAtConsumption
        let mut interner = Interner::new();
        let weight = make_sym(&mut interner, "weight");
        let w_ref = make_sym(&mut interner, "w_ref");
        let mut lowering = OwnershipLowering::new();

        lowering.mark_linear(weight);

        // Forward pass: borrow active
        lowering.active_borrows.insert(weight, BorrowKind::Immutable { borrower: w_ref });
        let fwd_decisions = lowering.decide(&weight, false, Some("MatMul"), false);
        assert_eq!(fwd_decisions, vec![OwnershipDecision::BorrowedNoAction]);

        // Release borrow (scope exit)
        lowering.active_borrows.remove(&weight);

        // Optimizer step: linear weight consumed (no grad context — @no_grad for update)
        let opt_decisions = lowering.decide(&weight, true, None, false);
        assert_eq!(opt_decisions, vec![OwnershipDecision::FreeAtConsumption]);
    }
}
