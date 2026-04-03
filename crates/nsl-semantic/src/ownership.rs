//! M38a: Linear types ownership checker — tracks tensor consumption and enforces
//! use-after-move errors at compile time. Gated by `--linear-types` flag.

use std::collections::HashMap;

use nsl_ast::Symbol;
use nsl_errors::{Diagnostic, Span};
use nsl_lexer::Interner;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Ownership state for a tensor-typed binding.
#[derive(Debug, Clone)]
pub enum OwnershipState {
    /// Tensor is live and owned — can be consumed or borrowed.
    Owned,
    /// Tensor has been consumed (moved). Any subsequent use is an error.
    Consumed { at: Span, by: String },
    /// Tensor is @shared — refcounted, multiple uses allowed.
    Shared,
    /// Tensor is a borrow (&T) — read-only access, cannot be consumed or mutated.
    Borrowed,
}

impl OwnershipState {
    pub fn is_consumed(&self) -> bool {
        matches!(self, OwnershipState::Consumed { .. })
    }

    pub fn is_shared(&self) -> bool {
        matches!(self, OwnershipState::Shared)
    }

    pub fn is_borrowed(&self) -> bool {
        matches!(self, OwnershipState::Borrowed)
    }
}

/// Tracks an active borrow on an owned variable.
#[derive(Debug, Clone)]
pub struct ActiveBorrow {
    /// Symbol of the borrow variable (the `&x` reference).
    pub borrow_sym: Symbol,
    /// Symbol of the owned variable being borrowed.
    pub owner_sym: Symbol,
    /// Where the borrow was created.
    pub created_at: Span,
}

#[derive(Clone)]
pub struct OwnershipSnapshot {
    pub states: HashMap<Symbol, OwnershipState>,
    tensor_bindings: HashMap<Symbol, Span>,
    active_borrows: HashMap<Symbol, Vec<ActiveBorrow>>,
}

// ---------------------------------------------------------------------------
// Checker
// ---------------------------------------------------------------------------

/// Walks AST after type checking to enforce linear ownership rules.
pub struct OwnershipChecker<'a> {
    interner: &'a Interner,
    states: HashMap<Symbol, OwnershipState>,
    /// Tracks which symbols are tensor-typed (only tensors are linear).
    tensor_bindings: HashMap<Symbol, Span>,
    /// Current loop depth (>0 means inside a loop body).
    loop_depth: u32,
    /// Symbols defined at each loop depth (for loop consumption checks).
    loop_defined: Vec<HashMap<Symbol, bool>>,
    /// Active borrows: owner_sym → list of borrow symbols.
    active_borrows: HashMap<Symbol, Vec<ActiveBorrow>>,
    pub diagnostics: Vec<Diagnostic>,
}

impl<'a> OwnershipChecker<'a> {
    #[allow(clippy::new_without_default)]
    pub fn new(interner: &'a Interner) -> Self {
        Self {
            interner,
            states: HashMap::new(),
            tensor_bindings: HashMap::new(),
            loop_depth: 0,
            loop_defined: Vec::new(),
            active_borrows: HashMap::new(),
            diagnostics: Vec::new(),
        }
    }

    fn resolve_sym(&self, sym: Symbol) -> &str {
        self.interner.resolve(sym.0).unwrap_or("<unknown>")
    }

    /// Register a new tensor binding as Owned (or Shared if @shared annotated).
    pub fn register_binding(&mut self, sym: Symbol, span: Span, is_shared: bool) {
        let state = if is_shared {
            OwnershipState::Shared
        } else {
            OwnershipState::Owned
        };
        self.states.insert(sym, state);
        self.tensor_bindings.insert(sym, span);
        if self.loop_depth > 0 {
            if let Some(set) = self.loop_defined.last_mut() {
                set.insert(sym, true);
            }
        }
    }

    /// Record that a tensor binding is consumed (moved) at the given span.
    pub fn consume(&mut self, sym: Symbol, at: Span, by: &str) {
        if !self.tensor_bindings.contains_key(&sym) {
            return;
        }

        let name = self.resolve_sym(sym).to_string();

        match self.states.get(&sym) {
            Some(OwnershipState::Shared) => {
                // @shared — always valid
            }
            Some(OwnershipState::Consumed { at: prev_span, by: prev_by }) => {
                self.diagnostics.push(
                    Diagnostic::error(format!("use of moved tensor '{name}'"))
                        .with_label(*prev_span, format!("value moved here, by '{prev_by}'"))
                        .with_label(at, "cannot use after move"),
                );
            }
            Some(OwnershipState::Borrowed) => {
                // Cannot consume a borrow — it's read-only
                self.diagnostics.push(
                    Diagnostic::error(format!("cannot consume borrowed tensor '{name}' — borrows are read-only"))
                        .with_label(at, "attempted to consume borrow"),
                );
            }
            Some(OwnershipState::Owned) | None => {
                // Check for active borrows — cannot consume while borrowed
                if self.has_active_borrows(sym) {
                    let borrows = self.active_borrows.get(&sym).unwrap();
                    let borrow_span = borrows[0].created_at;
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "cannot consume '{name}' — it has active borrows"
                        ))
                        .with_label(borrow_span, "borrowed here")
                        .with_label(at, "attempted consumption"),
                    );
                    return;
                }
                // Check loop consumption
                if self.loop_depth > 0 {
                    // Search ALL loop frames, not just innermost — a tensor defined
                    // in an outer loop is valid to consume in an inner loop.
                    let defined_in_loop = self
                        .loop_defined
                        .iter()
                        .any(|s| s.contains_key(&sym));
                    if !defined_in_loop {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "cannot consume '{name}' inside loop body — tensor defined outside loop"
                            ))
                            .with_label(at, "consumed in loop"),
                        );
                        return;
                    }
                }
                self.states.insert(
                    sym,
                    OwnershipState::Consumed {
                        at,
                        by: by.to_string(),
                    },
                );
            }
        }
    }

    /// Record a use of a tensor. For Owned tensors, use = consume (move semantics).
    pub fn use_binding(&mut self, sym: Symbol, at: Span, context: &str) {
        if !self.tensor_bindings.contains_key(&sym) {
            return;
        }
        if let Some(OwnershipState::Shared) = self.states.get(&sym) {
            return; // always valid
        }
        self.consume(sym, at, context);
    }

    /// Check that all owned tensors are consumed by end of scope.
    pub fn check_unconsumed(&mut self) {
        for (&sym, state) in &self.states {
            if let OwnershipState::Owned = state {
                let name = self.resolve_sym(sym).to_string();
                if let Some(&def_span) = self.tensor_bindings.get(&sym) {
                    self.diagnostics.push(
                        Diagnostic::error(format!("linear tensor '{name}' not consumed"))
                            .with_label(def_span, "defined here but never consumed"),
                    );
                }
            }
        }
    }

    /// Snapshot current ownership states (for branch analysis).
    pub fn snapshot(&self) -> HashMap<Symbol, OwnershipState> {
        self.states.clone()
    }

    /// Restore ownership states from snapshot.
    pub fn restore(&mut self, snapshot: HashMap<Symbol, OwnershipState>) {
        self.states = snapshot;
    }

    pub fn snapshot_all(&self) -> OwnershipSnapshot {
        OwnershipSnapshot {
            states: self.states.clone(),
            tensor_bindings: self.tensor_bindings.clone(),
            active_borrows: self.active_borrows.clone(),
        }
    }

    pub fn restore_all(&mut self, snapshot: OwnershipSnapshot) {
        self.states = snapshot.states;
        self.tensor_bindings = snapshot.tensor_bindings;
        self.active_borrows = snapshot.active_borrows;
    }

    pub fn check_branch_local_unconsumed(
        &mut self,
        before: &OwnershipSnapshot,
        after: &OwnershipSnapshot,
    ) {
        for (&sym, state) in &after.states {
            if before.states.contains_key(&sym) {
                continue;
            }
            if let OwnershipState::Owned = state {
                let name = self.resolve_sym(sym).to_string();
                if let Some(&def_span) = after.tensor_bindings.get(&sym) {
                    self.diagnostics.push(
                        Diagnostic::error(format!("linear tensor '{name}' not consumed"))
                            .with_label(def_span, "defined here but never consumed"),
                    );
                }
            }
        }
    }

    /// Check that branch consumption is symmetric and merge state.
    pub fn check_branch_symmetry(
        &mut self,
        before: &HashMap<Symbol, OwnershipState>,
        after_then: &HashMap<Symbol, OwnershipState>,
        after_else: &HashMap<Symbol, OwnershipState>,
        if_span: Span,
    ) {
        self.check_multi_branch_symmetry(before, &[after_then.clone(), after_else.clone()], if_span);
    }

    pub fn check_multi_branch_symmetry(
        &mut self,
        before: &HashMap<Symbol, OwnershipState>,
        branches: &[HashMap<Symbol, OwnershipState>],
        if_span: Span,
    ) {
        for (&sym, before_state) in before {
            if before_state.is_shared() || before_state.is_consumed() {
                continue;
            }
            let consumed: Vec<bool> = branches
                .iter()
                .map(|branch| branch.get(&sym).map(|s| s.is_consumed()).unwrap_or(false))
                .collect();
            let consumed_in_any = consumed.iter().any(|flag| *flag);
            let consumed_in_all = consumed.iter().all(|flag| *flag);

            if consumed_in_any != consumed_in_all {
                let name = self.resolve_sym(sym).to_string();
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "tensor '{name}' consumed in one branch but not the other"
                    ))
                    .with_label(if_span, "asymmetric consumption"),
                );
            }
        }

        // Merge: only mark consumed when BOTH branches consumed (symmetric).
        // If asymmetric, leave as Owned to avoid cascading false positives.
        for (&sym, before_state) in before {
            if before_state.is_shared() || before_state.is_consumed() {
                continue;
            }
            let consumed_in_all = branches
                .iter()
                .all(|branch| branch.get(&sym).map(|s| s.is_consumed()).unwrap_or(false));
            if consumed_in_all {
                if let Some(OwnershipState::Consumed { at, by }) = branches
                    .iter()
                    .find_map(|branch| branch.get(&sym))
                {
                    self.states
                        .insert(sym, OwnershipState::Consumed { at: *at, by: by.clone() });
                }
            }
        }
    }

    /// Enter a loop body scope.
    pub fn enter_loop(&mut self) {
        self.loop_depth += 1;
        self.loop_defined.push(HashMap::new());
    }

    /// Exit a loop body scope.
    pub fn exit_loop(&mut self) {
        self.loop_depth -= 1;
        self.loop_defined.pop();
    }

    // ─── Borrowing ──────────────────────────────────────────────────

    /// Register a borrow variable (&x) that borrows from an owned variable.
    /// The borrow is read-only and cannot be consumed.
    pub fn register_borrow(&mut self, borrow_sym: Symbol, owner_sym: Symbol, span: Span) {
        // Check owner is valid (owned or shared, not consumed)
        if let Some(OwnershipState::Consumed { at, by }) = self.states.get(&owner_sym) {
            let name = self.resolve_sym(owner_sym).to_string();
            self.diagnostics.push(
                Diagnostic::error(format!("cannot borrow '{name}' — already consumed"))
                    .with_label(*at, format!("consumed here by '{by}'"))
                    .with_label(span, "attempted borrow"),
            );
            return;
        }

        // Register the borrow variable as Borrowed state
        self.states.insert(borrow_sym, OwnershipState::Borrowed);
        self.tensor_bindings.insert(borrow_sym, span);

        // Track the active borrow on the owner
        self.active_borrows
            .entry(owner_sym)
            .or_default()
            .push(ActiveBorrow {
                borrow_sym,
                owner_sym,
                created_at: span,
            });
    }

    /// Release all borrows associated with a scope exit.
    /// Call when a borrow goes out of scope.
    pub fn release_borrow(&mut self, borrow_sym: Symbol) {
        // Find and remove from active borrows
        for borrows in self.active_borrows.values_mut() {
            borrows.retain(|b| b.borrow_sym != borrow_sym);
        }
        // Remove empty entries
        self.active_borrows.retain(|_, v| !v.is_empty());
    }

    /// Check if a variable has any active borrows.
    pub fn has_active_borrows(&self, sym: Symbol) -> bool {
        self.active_borrows
            .get(&sym)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    /// Use a borrowed variable (read-only — does not consume).
    pub fn use_borrow(&mut self, sym: Symbol, at: Span, _context: &str) {
        if !self.tensor_bindings.contains_key(&sym) {
            return;
        }
        match self.states.get(&sym) {
            Some(OwnershipState::Borrowed) => {
                // Read through borrow is always valid
            }
            _ => {
                // Not a borrow — delegate to normal use
                self.use_binding(sym, at, _context);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_lexer::Interner;

    fn sp() -> Span {
        Span::dummy()
    }

    #[test]
    fn test_basic_consumption() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.consume(x, sp(), "relu");
        assert!(checker.diagnostics.is_empty());

        checker.consume(x, sp(), "gelu");
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("moved tensor"));
    }

    #[test]
    fn test_shared_multi_use() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), true);
        checker.consume(x, sp(), "relu");
        checker.consume(x, sp(), "gelu");
        checker.consume(x, sp(), "sigmoid");
        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn test_unconsumed_linear_error() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.check_unconsumed();
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("not consumed"));
    }

    #[test]
    fn test_unconsumed_shared_ok() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), true);
        checker.check_unconsumed();
        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn test_branch_symmetric_consumption() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        let before = checker.snapshot();

        checker.consume(x, sp(), "relu");
        let after_then = checker.snapshot();

        checker.restore(before.clone());
        checker.consume(x, sp(), "gelu");
        let after_else = checker.snapshot();

        checker.check_branch_symmetry(&before, &after_then, &after_else, sp());
        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn test_branch_asymmetric_consumption() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        let before = checker.snapshot();

        checker.consume(x, sp(), "relu");
        let after_then = checker.snapshot();

        checker.restore(before.clone());
        let after_else = checker.snapshot();

        checker.check_branch_symmetry(&before, &after_then, &after_else, sp());
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("one branch"));
    }

    #[test]
    fn test_loop_consume_outer_error() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.enter_loop();
        checker.consume(x, sp(), "relu");
        checker.exit_loop();

        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("inside loop body"));
    }

    #[test]
    fn test_loop_consume_inner_ok() {
        let mut interner = Interner::new();
        let y = Symbol(interner.get_or_intern("y"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.enter_loop();
        checker.register_binding(y, sp(), false);
        checker.consume(y, sp(), "relu");
        checker.exit_loop();

        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn test_loop_shared_outer_ok() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), true);
        checker.enter_loop();
        checker.consume(x, sp(), "relu");
        checker.consume(x, sp(), "gelu");
        checker.exit_loop();

        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn test_non_tensor_ignored() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let checker = OwnershipChecker::new(&interner);

        // x not registered as tensor — consume is a no-op
        let mut checker = checker;
        checker.consume(x, sp(), "print");
        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn test_reassignment_resets_state() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.consume(x, sp(), "relu");

        checker.register_binding(x, sp(), false);
        checker.consume(x, sp(), "gelu");

        assert!(checker.diagnostics.is_empty());
    }

    // ── Borrowing tests ──────────────────────────────────────────────

    #[test]
    fn test_borrow_read_is_valid() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let x_ref = Symbol(interner.get_or_intern("x_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.register_borrow(x_ref, x, sp());
        // Read through borrow is fine
        checker.use_borrow(x_ref, sp(), "sum");
        checker.use_borrow(x_ref, sp(), "mean");
        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn test_borrow_cannot_be_consumed() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let x_ref = Symbol(interner.get_or_intern("x_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.register_borrow(x_ref, x, sp());
        // Trying to consume a borrow should fail
        checker.consume(x_ref, sp(), "relu");
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("borrowed tensor"));
    }

    #[test]
    fn test_cannot_consume_while_borrowed() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let x_ref = Symbol(interner.get_or_intern("x_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.register_borrow(x_ref, x, sp());
        // Cannot consume x while x_ref borrows it
        checker.consume(x, sp(), "relu");
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("active borrows"));
    }

    #[test]
    fn test_consume_after_borrow_released() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let x_ref = Symbol(interner.get_or_intern("x_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.register_borrow(x_ref, x, sp());
        checker.use_borrow(x_ref, sp(), "sum");
        // Release the borrow
        checker.release_borrow(x_ref);
        // Now consumption is valid
        checker.consume(x, sp(), "relu");
        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn test_multiple_borrows_allowed() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let a = Symbol(interner.get_or_intern("a"));
        let b = Symbol(interner.get_or_intern("b"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.register_borrow(a, x, sp());
        checker.register_borrow(b, x, sp());
        // Multiple immutable borrows are fine
        checker.use_borrow(a, sp(), "sum");
        checker.use_borrow(b, sp(), "mean");
        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn test_borrow_in_loop_ok() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let x_ref = Symbol(interner.get_or_intern("x_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.enter_loop();
        // Borrow in loop — reads are safe
        checker.register_borrow(x_ref, x, sp());
        checker.use_borrow(x_ref, sp(), "sum");
        checker.release_borrow(x_ref);
        checker.exit_loop();
        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn test_borrow_of_consumed_fails() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let x_ref = Symbol(interner.get_or_intern("x_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.consume(x, sp(), "relu");
        // Cannot borrow after consumption
        checker.register_borrow(x_ref, x, sp());
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("already consumed"));
    }

    #[test]
    fn test_has_active_borrows() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let x_ref = Symbol(interner.get_or_intern("x_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        assert!(!checker.has_active_borrows(x));

        checker.register_borrow(x_ref, x, sp());
        assert!(checker.has_active_borrows(x));

        checker.release_borrow(x_ref);
        assert!(!checker.has_active_borrows(x));
    }

    // ── Task 5: Autodiff + borrowed tensors ──────────────────────────
    //
    // Borrowed tensors in grad() scope:
    //   - The tape records raw i64 tensor pointers. A borrow is semantically
    //     transparent for reads — passing a &T to a tape-recording op passes
    //     the same underlying pointer as T would. No special tape handling is
    //     needed: `BorrowedNoAction` in codegen means the pointer is used
    //     directly without any refcount or free action.
    //   - @no_grad on a borrowed param: the borrow inherits the annotation.
    //     If a parameter is `&Tensor @no_grad`, it is excluded from tape
    //     recording in the same way an owned @no_grad tensor is. The borrow
    //     wrapper does not change this behaviour.
    //
    // Training pattern: borrowed weights during forward, then optimizer consumes.
    //   1. weights: Owned (e.g., model field or outer let-binding)
    //   2. weights_ref: Borrow from weights → used in forward pass (grad scope)
    //   3. Release weights_ref (end of forward)
    //   4. Optimizer consumes weights (update step) — valid because borrow released

    #[test]
    fn test_training_pattern_borrow_forward_then_optimizer_consumes() {
        // Simulate: weights owned, borrowed for forward pass, then consumed by optimizer.
        //
        //   weights: Owned
        //   weights_ref = &weights   (borrow for forward)
        //   [forward pass — reads through weights_ref, no consumption]
        //   release weights_ref
        //   optimizer consumes weights   (SGD step)
        //
        // This is the canonical training loop pattern: the model borrows its weights
        // for the forward pass, then the optimizer takes ownership to update them.
        let mut interner = Interner::new();
        let weights = Symbol(interner.get_or_intern("weights"));
        let weights_ref = Symbol(interner.get_or_intern("weights_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        // weights tensor is owned
        checker.register_binding(weights, sp(), false);

        // Borrow weights for forward pass
        checker.register_borrow(weights_ref, weights, sp());

        // Forward pass: multiple reads through borrow (matmul, relu, etc.)
        checker.use_borrow(weights_ref, sp(), "MatMul");
        checker.use_borrow(weights_ref, sp(), "BiasAdd");
        checker.use_borrow(weights_ref, sp(), "ReLU");

        // Forward complete — release the borrow (scope exit)
        checker.release_borrow(weights_ref);

        // Optimizer step: consume weights (SGD update = move the tensor into the update fn)
        checker.consume(weights, sp(), "sgd_update");

        // No errors: borrow was released before consumption
        assert!(
            checker.diagnostics.is_empty(),
            "training pattern should be clean: {:?}",
            checker.diagnostics
        );
    }

    #[test]
    fn test_training_pattern_cannot_consume_while_forward_borrows() {
        // The optimizer CANNOT consume weights while the borrow is still live.
        // This catches bugs where backward() is called while the forward borrow
        // overlaps with an optimizer step in the same scope.
        let mut interner = Interner::new();
        let weights = Symbol(interner.get_or_intern("weights"));
        let weights_ref = Symbol(interner.get_or_intern("weights_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(weights, sp(), false);
        checker.register_borrow(weights_ref, weights, sp());
        checker.use_borrow(weights_ref, sp(), "MatMul");

        // Bug: try to consume while borrow is still live
        checker.consume(weights, sp(), "sgd_update");

        assert_eq!(checker.diagnostics.len(), 1);
        assert!(
            checker.diagnostics[0].message.contains("active borrows"),
            "expected 'active borrows' error, got: {}",
            checker.diagnostics[0].message
        );
    }

    #[test]
    fn test_borrow_in_grad_scope_multiple_tape_ops() {
        // Borrowed tensor used in multiple tape-recording ops within a grad scope.
        // Since borrows are read-only and never consumed, every tape op is valid.
        // The tape stores raw pointers; the borrow's underlying tensor stays alive
        // for the entire scope (owned by the caller).
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let x_ref = Symbol(interner.get_or_intern("x_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.register_borrow(x_ref, x, sp());

        // Simulate a sequence of tape-recording ops (grad scope)
        // All are reads through the borrow — none consume x_ref
        for op in &["MatMul", "ReLU", "LayerNorm", "Softmax", "Log"] {
            checker.use_borrow(x_ref, sp(), op);
        }

        // No errors: borrow reads are always valid
        assert!(
            checker.diagnostics.is_empty(),
            "multiple tape ops on borrow should be clean: {:?}",
            checker.diagnostics
        );
    }

    // ── Task 6: Model methods use self as borrowed ────────────────────
    //
    // Model method dispatch:
    //   In the codegen layer (functions.rs), `self` in a model method is bound
    //   as a raw pointer (Cranelift I64 pointer type). It is NOT registered as a
    //   tensor-typed binding in the ownership checker — struct/model pointers are
    //   not linear types. Therefore:
    //   - Calling model.forward(x) does NOT consume the model instance.
    //   - The same model instance can be used across multiple forward calls.
    //   - This is safe by construction: the pointer is passed by value (read-only
    //     from the ownership checker's perspective), and the ownership checker
    //     only tracks tensor-typed bindings.
    //
    // These tests verify the ownership checker correctly handles the model self pattern.

    #[test]
    fn test_model_self_not_registered_as_linear() {
        // Model `self` is a pointer, not a tensor — it must NOT appear in the
        // ownership checker's tensor_bindings map. Calling forward multiple times
        // must not trigger any ownership error.
        let mut interner = Interner::new();
        let model_ptr = Symbol(interner.get_or_intern("self"));
        let checker = OwnershipChecker::new(&interner);

        // self is never registered as a tensor binding
        assert!(
            !checker.states.contains_key(&model_ptr),
            "model self should not be in ownership states"
        );
    }

    #[test]
    fn test_model_forward_multiple_calls_pattern() {
        // Simulate model.forward(x) called multiple times in a loop.
        // The model weights are @shared (or each input is borrowed). Verify that
        // a @shared weight can be used repeatedly without being consumed.
        let mut interner = Interner::new();
        let weight = Symbol(interner.get_or_intern("weight"));
        let mut checker = OwnershipChecker::new(&interner);

        // Model fields are @shared (refcounted) so they can be used across calls
        checker.register_binding(weight, sp(), true); // is_shared = true

        checker.enter_loop();
        // Simulate forward() using weight multiple times per iteration
        checker.consume(weight, sp(), "forward_matmul");
        checker.consume(weight, sp(), "forward_matmul");
        checker.consume(weight, sp(), "forward_matmul");
        checker.exit_loop();

        // No errors: @shared tensors can be consumed (refcount-managed) any number of times
        assert!(
            checker.diagnostics.is_empty(),
            "model weights @shared should support multiple forward calls: {:?}",
            checker.diagnostics
        );
    }

    #[test]
    fn test_model_forward_borrow_self_weights_pattern() {
        // A more explicit pattern: model weights are borrowed for each forward call.
        // The borrow is created at call-start and released at call-end.
        // This can repeat any number of times.
        let mut interner = Interner::new();
        let weight = Symbol(interner.get_or_intern("weight"));
        let w_ref = Symbol(interner.get_or_intern("w_ref"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(weight, sp(), false);

        // First forward call
        checker.register_borrow(w_ref, weight, sp());
        checker.use_borrow(w_ref, sp(), "MatMul");
        checker.release_borrow(w_ref);

        // Second forward call (same pattern — borrow re-created)
        checker.register_borrow(w_ref, weight, sp());
        checker.use_borrow(w_ref, sp(), "MatMul");
        checker.release_borrow(w_ref);

        // Third forward call
        checker.register_borrow(w_ref, weight, sp());
        checker.use_borrow(w_ref, sp(), "MatMul");
        checker.release_borrow(w_ref);

        assert!(
            checker.diagnostics.is_empty(),
            "model.forward() can be called multiple times via borrow pattern: {:?}",
            checker.diagnostics
        );
    }
}
