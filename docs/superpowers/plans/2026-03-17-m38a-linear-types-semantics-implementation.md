# M38a: Linear Types Semantics — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in (`--linear-types`) ownership checker pass that tracks tensor consumption, enforces use-after-move errors, validates branch/loop consumption consistency, and supports `@shared` escape hatch — all at compile time with zero runtime changes.

**Architecture:** A single new semantic pass (`ownership.rs`) that runs after type checking on the fully-typed AST. Tracks per-binding ownership state (Owned/Consumed/Shared) via a side-table HashMap, not embedded in the Type enum (avoids breaking ~100 Type::Tensor construction sites). The pass is gated by `--linear-types` flag; without it, all tensors are implicitly Shared (current behavior). `@shared` decorator on `let`-bindings opts individual tensors into refcounted mode. Branch analysis verifies symmetric consumption. Loop analysis prevents consuming outer-scoped tensors.

**Tech Stack:** Rust (semantic analysis)

**Spec:** `docs/superpowers/specs/2026-03-15-m38-linear-types-design.md` (Sections 1, 2.1-2.3, 5)

---

## Important: Scope of This Plan

**This plan builds the ownership checker as a standalone semantic pass.** It delivers:
- `OwnershipState` enum and `OwnershipChecker` struct with full unit test coverage
- Use-after-move detection with source-location error messages
- `@shared` annotation support (multi-use tensors)
- Branch consumption symmetry checking
- Loop consumption prevention
- Unconsumed linear tensor detection
- `--linear-types` CLI flag (gating the pass)
- E2E tests for both error cases and valid programs

**The ownership checker uses a side-table approach** — ownership is tracked in `HashMap<Symbol, OwnershipState>`, NOT as a field on `Type::Tensor`. This avoids modifying the type system (which would break dozens of Type::Tensor construction sites across checker.rs, codegen, and tests). The side-table is populated during the ownership pass from decorator annotations and default rules.

**Deferred to M38b (Phase 7):** Codegen ownership lowering (refcount elision, free-at-consumption), autodiff tape safety proofs, `&`/`&mut` borrow syntax in parser/AST, `@consume` parameter annotation, closure capture semantics, `--warn-shared` audit mode, debug-mode poison values, borrow exclusivity tracking.

**Known simplification:** This plan implements ownership tracking for `let`-binding-level consumption only (not sub-expression-level). A tensor is "consumed" when its name appears as an argument to a function call or binary op. Borrows (`&x`, `&mut x`) require parser changes and are deferred.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-semantic/src/ownership.rs` | OwnershipChecker, OwnershipState, check_stmt/check_expr, branch/loop analysis, error messages | 400 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod ownership;`, invoke checker when `linear_types` flag is true |
| `crates/nsl-semantic/src/checker.rs` | Validate `@shared` decorator on VarDecl |
| `crates/nsl-cli/src/main.rs` | Add `--linear-types` flag to Build/Run/Check subcommands |
| `crates/nsl-cli/tests/e2e.rs` | Add M38a E2E tests |

---

## Phase 1: Core Ownership Checker

### Task 1: OwnershipState + OwnershipChecker Types

**Files:**
- Create: `crates/nsl-semantic/src/ownership.rs`
- Modify: `crates/nsl-semantic/src/lib.rs`

- [ ] **Step 1: Create `ownership.rs` with core types, basic consumption tracking, and tests**

```rust
//! M38a: Linear types ownership checker — tracks tensor consumption and enforces
//! use-after-move errors at compile time. Gated by `--linear-types` flag.

use std::collections::HashMap;

use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_ast::expr::{Expr, ExprKind};
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
    /// Returns an error diagnostic if the tensor was already consumed.
    pub fn consume(&mut self, sym: Symbol, at: Span, by: &str) {
        // Non-tensor bindings are ignored
        if !self.tensor_bindings.contains_key(&sym) {
            return;
        }

        let name = self.resolve_sym(sym).to_string();

        match self.states.get(&sym) {
            Some(OwnershipState::Shared) => {
                // @shared — always valid, no consumption tracking
            }
            Some(OwnershipState::Consumed { at: prev_span, by: prev_by }) => {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "use of moved tensor '{}'", name
                    ))
                    .with_label(*prev_span, format!("value moved here, by '{prev_by}'"))
                    .with_label(at, "cannot use after move"),
                );
            }
            Some(OwnershipState::Owned) | None => {
                // Check loop consumption: can't consume outer-scoped tensor in loop
                if self.loop_depth > 0 {
                    let defined_in_loop = self.loop_defined.last()
                        .map(|s| s.contains_key(&sym))
                        .unwrap_or(false);
                    if !defined_in_loop {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "cannot consume '{}' inside loop body — tensor defined outside loop",
                                name
                            ))
                            .with_label(at, "consumed in loop"),
                        );
                        return;
                    }
                }
                self.states.insert(sym, OwnershipState::Consumed {
                    at,
                    by: by.to_string(),
                });
            }
        }
    }

    /// Record a use of a tensor (read, not consume). For Shared tensors this is always valid.
    /// For Owned tensors this IS a consumption (move semantics).
    pub fn use_binding(&mut self, sym: Symbol, at: Span, context: &str) {
        if !self.tensor_bindings.contains_key(&sym) {
            return;
        }
        match self.states.get(&sym) {
            Some(OwnershipState::Shared) => {
                // Always valid
            }
            _ => {
                // For owned tensors, use = consume
                self.consume(sym, at, context);
            }
        }
    }

    /// Check that all owned tensors are consumed by end of scope.
    pub fn check_unconsumed(&mut self) {
        for (&sym, state) in &self.states {
            if let OwnershipState::Owned = state {
                let name = self.resolve_sym(sym).to_string();
                if let Some(&def_span) = self.tensor_bindings.get(&sym) {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "linear tensor '{}' not consumed", name
                        ))
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

    /// Check that branch consumption is symmetric: if a tensor is consumed in one
    /// branch, it must be consumed in all branches.
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
                        "tensor '{}' consumed in one branch but not the other",
                        name
                    ))
                    .with_label(if_span, "asymmetric consumption"),
                );
            }
        }

        // Merge: consumed in either branch means consumed in merged state
        for (&sym, before_state) in before {
            if before_state.is_shared() || before_state.is_consumed() {
                continue;
            }
            // Check both branches for consumption
            if let Some(OwnershipState::Consumed { at, by }) = after_then.get(&sym) {
                self.states.insert(sym, OwnershipState::Consumed { at: *at, by: by.clone() });
            } else if let Some(OwnershipState::Consumed { at, by }) = after_else.get(&sym) {
                self.states.insert(sym, OwnershipState::Consumed { at: *at, by: by.clone() });
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

    // NOTE: Intern all symbols FIRST, then create the checker. This avoids
    // Rust borrow conflicts (&interner immutable in checker vs &mut interner for interning).

    #[test]
    fn test_basic_consumption() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), false);
        checker.consume(x, sp(), "relu");
        assert!(checker.diagnostics.is_empty());

        // Second consume should error
        checker.consume(x, sp(), "gelu");
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("moved tensor"));
    }

    #[test]
    fn test_shared_multi_use() {
        let mut interner = Interner::new();
        let x = Symbol(interner.get_or_intern("x"));
        let mut checker = OwnershipChecker::new(&interner);

        checker.register_binding(x, sp(), true); // @shared
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

        checker.register_binding(x, sp(), true); // @shared
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
        let mut checker = OwnershipChecker::new(&interner);

        // Don't register x as tensor binding — consume is a no-op
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

        // Reassign x — fresh binding
        checker.register_binding(x, sp(), false);
        checker.consume(x, sp(), "gelu");

        assert!(checker.diagnostics.is_empty());
    }
}
```

- [ ] **Step 2: Add `pub mod ownership;` to semantic lib.rs**

- [ ] **Step 3: Run tests**

```bash
cargo test -p nsl-semantic ownership -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(m38a): add OwnershipChecker with consumption tracking, branch/loop analysis"
```

---

## Phase 2: @shared Decorator Validation

### Task 2: Checker Wiring for @shared

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 1: Wire @shared validation in StmtKind::Decorated handler**

Find the `StmtKind::Decorated` handler in checker.rs (where `@no_fuse`, `@fuse_graph`, etc. are processed). Add after the existing decorator validations:

```rust
// M38a: @shared annotation — valid on let-bindings
if dname == "shared" {
    match &stmt.kind {
        StmtKind::VarDecl { .. } => {
            // Valid — tensor will be marked Shared in ownership pass
        }
        _ => {
            self.diagnostics.push(
                Diagnostic::error(
                    "@shared can only be applied to let-bindings".to_string()
                )
                .with_label(deco.span, "invalid @shared target"),
            );
        }
    }
}
```

NOTE: Find the exact location by searching for `"no_fuse"` in the Decorated handler — add the `@shared` block nearby.

- [ ] **Step 2: Verify compilation**

```bash
cargo check -p nsl-semantic
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(m38a): validate @shared decorator on let-bindings in checker"
```

---

## Phase 3: CLI + E2E

### Task 3: --linear-types CLI Flag

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`

- [ ] **Step 1: Add `--linear-types` to Build, Run, and Check subcommands**

Add to each subcommand's struct:

```rust
        /// M38a: Enable linear types ownership checking
        #[arg(long)]
        linear_types: bool,
```

- [ ] **Step 2: Update destructuring** in match arms to include `linear_types: _linear_types`

NOTE: Same dormant-flag pattern as M36/M37 — parsed but not yet threaded through to `analyze_with_imports()`. The ownership checker invocation will be wired when CompileOptions threading is completed.

- [ ] **Step 3: Verify compilation**

```bash
cargo check --workspace
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(m38a): add --linear-types CLI flag to build/run/check subcommands"
```

---

### Task 4: E2E Tests

**Files:**
- Create: `examples/m38_shared_basic.nsl`
- Create: `tests/expected/m38_shared_basic.txt`
- Create: `examples/m38_shared_validation_error.nsl`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create basic @shared test** (compiles and runs normally — @shared is a no-op without --linear-types)

```nsl
# M38a: @shared annotation — basic test

@shared
let x = ones([2, 3])
let y = x + x
print(y)
```

Expected output (x + x = 2*ones):
```
tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
```

- [ ] **Step 2: Create @shared validation error test**

```nsl
# M38a: @shared validation error — invalid target

@shared
fn bad() -> int:
    return 0
```

- [ ] **Step 3: Add E2E tests to e2e.rs**

```rust
// ---------------------------------------------------------------------------
// M38a: Linear Types Semantics
// ---------------------------------------------------------------------------

#[test]
fn e2e_m38_shared_basic() {
    assert_output_matches("m38_shared_basic");
}

#[test]
fn e2e_m38_shared_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m38_shared_validation_error.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m38_shared_validation_error, but it succeeded"
    );
    assert!(
        stderr.contains("shared") || stderr.contains("let-binding"),
        "Expected @shared validation error in stderr, got: {}",
        stderr
    );
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p nsl-cli e2e_m38 -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git commit -m "test(m38a): add E2E tests for @shared annotation"
```

---

### Task 5: Full Verification + Clippy

- [ ] **Step 1: Run all workspace lib tests**

```bash
cargo test --workspace --lib
```

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

- [ ] **Step 3: Run M38a-specific tests**

```bash
cargo test -p nsl-semantic ownership -- --nocapture
cargo test -p nsl-cli e2e_m38 -- --nocapture
```

- [ ] **Step 4: Fix any issues, commit**

```bash
git commit -m "chore(m38a): fix clippy warnings and verify full test suite"
```

---

## Summary

| Task | Component | Tests |
|---|---|---|
| 1 | OwnershipChecker + OwnershipState + branch/loop analysis | 12 unit |
| 2 | @shared decorator validation in checker | compile check |
| 3 | --linear-types CLI flag | compile check |
| 4 | E2E tests (@shared basic + validation error) | 2 E2E |
| 5 | Full verification | all tests |

**Total: 5 tasks, ~12 unit tests + 2 E2E tests**

### Deferred to M38b (Phase 7)

- Codegen ownership lowering (`crates/nsl-codegen/src/ownership.rs`) — refcount elision, free-at-consumption
- Autodiff tape safety proofs (`ownership_autodiff.rs`) — backward access classification
- `&`/`&mut` borrow syntax — parser changes to `TypeExprKind`, borrow exclusivity tracking
- `@consume` parameter annotation
- Closure capture semantics (`||` vs `|&|`)
- `--warn-shared` audit mode
- Debug-mode poison values (zero after move)
- Ownership checker integration into `analyze_with_imports()` (requires CompileOptions threading)
- Performance benchmarks (refcount ops eliminated)
- Model weight `@shared` default inference in ownership pass
- AST walking for the ownership checker (currently only unit-tested with direct API calls)
