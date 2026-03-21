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

    /// Check that branch consumption is symmetric and merge state.
    pub fn check_branch_symmetry(
        &mut self,
        before: &HashMap<Symbol, OwnershipState>,
        after_then: &HashMap<Symbol, OwnershipState>,
        after_else: &HashMap<Symbol, OwnershipState>,
        if_span: Span,
    ) {
        for (&sym, before_state) in before {
            if before_state.is_shared() || before_state.is_consumed() {
                continue;
            }
            let consumed_in_then = after_then
                .get(&sym)
                .map(|s| s.is_consumed())
                .unwrap_or(false);
            let consumed_in_else = after_else
                .get(&sym)
                .map(|s| s.is_consumed())
                .unwrap_or(false);

            if consumed_in_then != consumed_in_else {
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
            let consumed_in_then = after_then.get(&sym).map(|s| s.is_consumed()).unwrap_or(false);
            let consumed_in_else = after_else.get(&sym).map(|s| s.is_consumed()).unwrap_or(false);
            if consumed_in_then && consumed_in_else {
                if let Some(OwnershipState::Consumed { at, by }) = after_then.get(&sym) {
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
}
