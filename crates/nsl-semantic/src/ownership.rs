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
}

impl OwnershipState {
    pub fn is_consumed(&self) -> bool {
        matches!(self, OwnershipState::Consumed { .. })
    }

    pub fn is_shared(&self) -> bool {
        matches!(self, OwnershipState::Shared)
    }
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
            Some(OwnershipState::Owned) | None => {
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
}
