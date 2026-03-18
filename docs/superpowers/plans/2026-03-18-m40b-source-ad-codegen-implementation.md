# M40b: Source-to-Source AD — Wengert Extraction & Backward Context Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the source-to-source AD pipeline by adding Wengert extraction from typed AST expressions, a BackwardContext runtime for saved tensors, and compiler strategy selection that chooses source AD for static graphs and falls back to tape for dynamic control flow. This builds on M40a's analysis library (WengertList, AD rules, adjoint generator, dead gradient eliminator).

**Architecture:** Extend `source_ad.rs` with `WengertExtractor` that walks function body expressions and builds a `WengertList`. Create `backward_context.rs` runtime module with saved tensor slot management. Add strategy selection to `stmt.rs`'s `compile_grad_block`. Add `--tape-ad` CLI flag for forcing tape-based fallback.

**Tech Stack:** Rust (codegen AST walking + runtime FFI)

**Spec:** `docs/superpowers/specs/2026-03-15-m40-source-ad-design.md` (Sections 2-4)

**Prerequisites:** M40a (WengertList, AdjointGenerator, AD rules, DeadGradientEliminator, SavedTensorAnalyzer — all complete), M38b (ownership-aware codegen — just completed)

---

## Important: Scope of This Plan

**This plan adds Wengert extraction + backward context + strategy selection.** It delivers:
- `WengertExtractor` — walks typed AST expressions, maps to WengertOps, detects dynamic control flow
- `is_static_graph()` — checks if a grad block's computation can be source-AD'd
- `extract_wengert()` — entry point that returns `Option<WengertList>` (None for dynamic graphs)
- `BackwardContext` runtime — saved tensor slot management with FFI (new/save/load/free)
- Strategy selection in `compile_grad_block`: check `source_ad_enabled`, try extraction, fallback to tape
- `--tape-ad` CLI flag to force tape-based AD
- Builtin registration for 4 BackwardContext FFI functions
- 15+ unit tests covering extraction, dynamic detection, backward context, strategy selection

**Deferred to M40c:** Cranelift IR emission of backward functions (compile_grad_block_source_ad emitting actual backward code), cross-boundary fusion with M31, higher-order derivatives, `@checkpoint` recomputation integration, E2E numerical validation (forward+backward correctness), hybrid per-layer AD strategy.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-runtime/src/backward_context.rs` | `BackwardContext` struct, saved tensor slot FFI | 120 |

### Modified Files

| File | Change | ~Lines |
|---|---|---|
| `crates/nsl-codegen/src/source_ad.rs` | Add `WengertExtractor`, `is_static_graph`, `extract_wengert` | +250 |
| `crates/nsl-codegen/src/builtins.rs` | Register 4 BackwardContext FFI functions | +5 |
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod backward_context;` | +1 |
| `crates/nsl-cli/src/main.rs` | Add `--tape-ad` flag to Run and Build | +10 |
| `crates/nsl-codegen/src/lib.rs` | Add `tape_ad` to CompileOptions | +2 |

---

## Phase 1: Wengert Extraction

### Task 1: WengertExtractor

**Files:**
- Modify: `crates/nsl-codegen/src/source_ad.rs`

- [ ] **Step 1: Add WengertExtractor to source_ad.rs with extraction and static-graph detection**

Add after the existing code (before tests):

```rust
// ---------------------------------------------------------------------------
// M40b: Wengert Extraction from AST
// ---------------------------------------------------------------------------

use nsl_ast::expr::ExprKind;
use nsl_ast::operator::{BinOp, UnaryOp as AstUnaryOp};
use nsl_ast::stmt::StmtKind;
use nsl_lexer::Interner;

/// Extracts a WengertList from a sequence of AST statements.
///
/// Returns `Some(WengertList)` if the computation is fully static (no
/// data-dependent control flow). Returns `None` if dynamic control flow
/// is detected, signaling fallback to tape-based AD.
pub struct WengertExtractor<'a> {
    interner: &'a Interner,
    list: WengertList,
    /// Maps AST symbol → WengertList VarId.
    symbol_to_var: std::collections::HashMap<nsl_ast::Symbol, VarId>,
    /// Next VarId to allocate.
    next_var: VarId,
    /// Whether this computation graph is fully static.
    is_static: bool,
    /// Symbols that are model parameters (need gradients).
    param_symbols: std::collections::HashSet<nsl_ast::Symbol>,
}

impl<'a> WengertExtractor<'a> {
    pub fn new(interner: &'a Interner) -> Self {
        WengertExtractor {
            interner,
            list: WengertList {
                ops: Vec::new(),
                output: 0,
                var_names: std::collections::HashMap::new(),
            },
            symbol_to_var: std::collections::HashMap::new(),
            next_var: 0,
            is_static: true,
            param_symbols: std::collections::HashSet::new(),
        }
    }

    fn alloc_var(&mut self) -> VarId {
        let id = self.next_var;
        self.next_var += 1;
        id
    }

    /// Register a parameter symbol (needs gradient).
    pub fn register_param(&mut self, sym: nsl_ast::Symbol) {
        let var = self.alloc_var();
        self.symbol_to_var.insert(sym, var);
        self.param_symbols.insert(sym);

        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
        self.list.var_names.insert(var, name.clone());
        self.list.ops.push(crate::wengert::WengertOp {
            id: self.list.ops.len() as u32,
            result: var,
            op: crate::wengert::PrimalOp::Param(name),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });
    }

    /// Register an input symbol (data, no gradient).
    pub fn register_input(&mut self, sym: nsl_ast::Symbol) {
        let var = self.alloc_var();
        self.symbol_to_var.insert(sym, var);

        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
        self.list.var_names.insert(var, name.clone());
        self.list.ops.push(crate::wengert::WengertOp {
            id: self.list.ops.len() as u32,
            result: var,
            op: crate::wengert::PrimalOp::Input(name),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });
    }

    /// Extract statements into the Wengert list.
    /// Returns false if dynamic control flow is detected.
    pub fn extract_stmts(&mut self, stmts: &[nsl_ast::stmt::Stmt]) -> bool {
        for stmt in stmts {
            if !self.extract_stmt(stmt) {
                self.is_static = false;
                return false;
            }
        }
        true
    }

    fn extract_stmt(&mut self, stmt: &nsl_ast::stmt::Stmt) -> bool {
        match &stmt.kind {
            StmtKind::VarDecl { pattern, value: Some(val), .. } => {
                if let Some(var) = self.extract_expr(val) {
                    if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                        self.symbol_to_var.insert(*sym, var);
                        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
                        self.list.var_names.insert(var, name);
                    }
                    true
                } else {
                    false
                }
            }

            StmtKind::Return(Some(expr)) => {
                if let Some(var) = self.extract_expr(expr) {
                    self.list.output = var;
                    true
                } else {
                    false
                }
            }

            StmtKind::Expr(expr) => {
                self.extract_expr(expr).is_some()
            }

            // Dynamic control flow → not static
            StmtKind::While { .. } => {
                self.is_static = false;
                false
            }

            // For loops: could be static if range is compile-time known,
            // but conservatively mark as dynamic for M40b
            StmtKind::For { .. } => {
                self.is_static = false;
                false
            }

            // If statements: mark as dynamic (data-dependent branches)
            // Shape-dependent ifs could be resolved at compile time, but
            // that requires type-level analysis deferred to M40c
            StmtKind::If { .. } => {
                self.is_static = false;
                false
            }

            // Other statements pass through (assign, decorated, etc.)
            _ => true,
        }
    }

    /// Extract an expression into a WengertOp, returning its VarId.
    /// Returns None if the expression contains dynamic control flow.
    fn extract_expr(&mut self, expr: &nsl_ast::expr::Expr) -> Option<VarId> {
        match &expr.kind {
            ExprKind::Ident(sym) => {
                self.symbol_to_var.get(sym).copied()
            }

            ExprKind::BinaryOp { left, op, right } => {
                let l = self.extract_expr(left)?;
                let r = self.extract_expr(right)?;
                let result = self.alloc_var();
                let primal_op = match op {
                    BinOp::Add => crate::wengert::PrimalOp::Add,
                    BinOp::Sub => crate::wengert::PrimalOp::Sub,
                    BinOp::Mul => crate::wengert::PrimalOp::Mul,
                    BinOp::Div => crate::wengert::PrimalOp::Div,
                    _ => return None, // Unsupported op for AD
                };
                self.list.ops.push(crate::wengert::WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: primal_op,
                    inputs: vec![l, r],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::UnaryOp { op, operand } => {
                let input = self.extract_expr(operand)?;
                let result = self.alloc_var();
                let primal_op = match op {
                    AstUnaryOp::Neg => crate::wengert::PrimalOp::Neg,
                    _ => return None,
                };
                self.list.ops.push(crate::wengert::WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: primal_op,
                    inputs: vec![input],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::Call { callee, args } => {
                // Extract function name
                let func_name = if let ExprKind::Ident(sym) = &callee.kind {
                    self.interner.resolve(sym.0).unwrap_or("").to_string()
                } else {
                    return None; // Complex callee — can't extract
                };

                // Extract arguments
                let mut input_vars = Vec::new();
                for arg in args {
                    if let Some(var) = self.extract_expr(&arg.value) {
                        input_vars.push(var);
                    } else {
                        return None;
                    }
                }

                let result = self.alloc_var();
                let primal_op = match func_name.as_str() {
                    "relu" => crate::wengert::PrimalOp::Relu,
                    "sigmoid" => crate::wengert::PrimalOp::Sigmoid,
                    "tanh" => crate::wengert::PrimalOp::Tanh,
                    "exp" => crate::wengert::PrimalOp::Exp,
                    "log" => crate::wengert::PrimalOp::Log,
                    "sqrt" => crate::wengert::PrimalOp::Sqrt,
                    "matmul" => crate::wengert::PrimalOp::Matmul,
                    // Transpose and Softmax are struct variants requiring dim args —
                    // fall back to tape for M40b; proper arg extraction in M40c
                    "transpose" | "softmax" => return None,
                    _ => return None, // Unknown function — can't differentiate
                };

                self.list.ops.push(crate::wengert::WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: primal_op,
                    inputs: input_vars,
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::IntLiteral(v) => {
                let result = self.alloc_var();
                self.list.ops.push(crate::wengert::WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: crate::wengert::PrimalOp::Constant(*v as f64),
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::FloatLiteral(v) => {
                let result = self.alloc_var();
                self.list.ops.push(crate::wengert::WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: crate::wengert::PrimalOp::Constant(*v),
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::Paren(inner) => self.extract_expr(inner),

            // Anything else we can't extract → fallback
            _ => None,
        }
    }

    /// Finalize extraction. Returns the WengertList if the graph is static.
    pub fn finalize(self) -> Option<WengertList> {
        if self.is_static && !self.list.ops.is_empty() {
            Some(self.list)
        } else {
            None
        }
    }

    /// Check if the computation graph is static (no dynamic control flow).
    pub fn is_static_graph(&self) -> bool {
        self.is_static
    }

    /// Get the parameter VarIds (for gradient computation targets).
    pub fn param_vars(&self) -> Vec<VarId> {
        self.param_symbols.iter()
            .filter_map(|sym| self.symbol_to_var.get(sym).copied())
            .collect()
    }
}
```

Add tests to the existing `#[cfg(test)] mod tests` block:

```rust
// --- WengertExtractor tests ---

// NOTE: Tests pre-intern all symbols BEFORE creating the extractor to avoid
// borrow conflict (&interner held by extractor vs &mut for get_or_intern).
// Use Default::default() for Interner construction (not ::new()).
// Pattern requires `id: NodeId` field. Use NodeId::dummy() or NodeId(0).

#[test]
fn extract_simple_input() {
    let mut interner: Interner = Default::default();
    // Pre-intern before borrowing into extractor
    let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
    let extractor = {
        let mut ext = WengertExtractor::new(&interner);
        ext.register_input(x_sym);
        ext
    };

    assert!(extractor.is_static_graph());
    let list = extractor.finalize();
    assert!(list.is_some());
    assert_eq!(list.unwrap().ops.len(), 1);
}

#[test]
fn extract_registers_params_and_inputs() {
    let mut interner: Interner = Default::default();
    let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
    let w_sym = nsl_ast::Symbol(interner.get_or_intern("W"));

    let extractor = {
        let mut ext = WengertExtractor::new(&interner);
        ext.register_input(x_sym);
        ext.register_param(w_sym);
        ext
    };

    let params = extractor.param_vars();
    assert_eq!(params.len(), 1);
    assert!(extractor.is_static_graph());
}

#[test]
fn extract_detects_dynamic_while() {
    let interner: Interner = Default::default();
    let mut extractor = WengertExtractor::new(&interner);

    let while_stmt = nsl_ast::stmt::Stmt {
        kind: StmtKind::While {
            condition: nsl_ast::expr::Expr {
                kind: ExprKind::BoolLiteral(true),
                span: nsl_errors::Span::dummy(),
                id: nsl_ast::NodeId(0),
            },
            body: nsl_ast::stmt::Block { stmts: vec![], span: nsl_errors::Span::dummy() },
        },
        span: nsl_errors::Span::dummy(),
        id: nsl_ast::NodeId(1),
    };

    let result = extractor.extract_stmts(&[while_stmt]);
    assert!(!result);
    assert!(!extractor.is_static_graph());
}

#[test]
fn extract_detects_dynamic_for() {
    let interner: Interner = Default::default();
    let mut extractor = WengertExtractor::new(&interner);

    let for_stmt = nsl_ast::stmt::Stmt {
        kind: StmtKind::For {
            pattern: nsl_ast::pattern::Pattern {
                kind: nsl_ast::pattern::PatternKind::Wildcard,
                span: nsl_errors::Span::dummy(),
                id: nsl_ast::NodeId(0),
            },
            iterable: nsl_ast::expr::Expr {
                kind: ExprKind::IntLiteral(10),
                span: nsl_errors::Span::dummy(),
                id: nsl_ast::NodeId(0),
            },
            body: nsl_ast::stmt::Block { stmts: vec![], span: nsl_errors::Span::dummy() },
        },
        span: nsl_errors::Span::dummy(),
        id: nsl_ast::NodeId(1),
    };

    let result = extractor.extract_stmts(&[for_stmt]);
    assert!(!result);
}

#[test]
fn extractor_finalize_none_when_empty() {
    let interner: Interner = Default::default();
    let extractor = WengertExtractor::new(&interner);
    assert!(extractor.finalize().is_none());
}
```

**Note on AST imports:** Check the actual `StmtKind::For` and `StmtKind::While` variant field names against `crates/nsl-ast/src/stmt.rs`. The plan's test constructors may need adjustment for exact field names (e.g., `While` may have `condition` and `body`, or different names). The implementing agent MUST read the actual AST types before writing tests.

---

## Phase 2: BackwardContext Runtime

### Task 2: BackwardContext FFI

**Files:**
- Create: `crates/nsl-runtime/src/backward_context.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 2: Create backward_context.rs with saved tensor slot management**

```rust
// crates/nsl-runtime/src/backward_context.rs
//! M40b: Backward context for source-to-source AD.
//!
//! Provides saved tensor slot management for the compile-time backward pass.
//! Unlike the tape (M12), this context is created per-grad-block with a
//! compile-time-known number of slots.

use std::sync::Mutex;

/// Context for storing intermediate tensors needed by the backward pass.
/// Created per `grad` block with a fixed number of slots determined at compile time.
pub struct BackwardContext {
    slots: Vec<i64>,  // i64 tensor pointers (0 = empty)
    num_slots: usize,
}

impl BackwardContext {
    pub fn new(num_slots: usize) -> Self {
        BackwardContext {
            slots: vec![0; num_slots],
            num_slots,
        }
    }

    /// Save a tensor pointer in the given slot.
    pub fn save(&mut self, slot: usize, tensor_ptr: i64) {
        if slot < self.num_slots {
            self.slots[slot] = tensor_ptr;
        }
    }

    /// Load a tensor pointer from the given slot.
    pub fn load(&self, slot: usize) -> i64 {
        if slot < self.num_slots {
            self.slots[slot]
        } else {
            0
        }
    }

    /// Clear all slots (for cleanup after backward pass).
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

static BACKWARD_CTX: Mutex<Option<BackwardContext>> = Mutex::new(None);

/// Create a new backward context with the given number of saved tensor slots.
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_new(num_slots: i64) -> i64 {
    let mut guard = BACKWARD_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(BackwardContext::new(num_slots as usize));
    0
}

/// Save a tensor pointer in the given slot.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_save(slot: i64, tensor_ptr: i64) -> i64 {
    let mut guard = BACKWARD_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_backward_ctx_new not called");
    ctx.save(slot as usize, tensor_ptr);
    0
}

/// Load a tensor pointer from the given slot.
/// Returns the tensor pointer, or 0 if slot is empty/invalid.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_load(slot: i64) -> i64 {
    let guard = BACKWARD_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_backward_ctx_new not called");
    ctx.load(slot as usize)
}

/// Destroy the backward context and free all saved references.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_free() -> i64 {
    let mut guard = BACKWARD_CTX.lock().unwrap();
    *guard = None;
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_backward_ctx_free();
        guard
    }

    #[test]
    fn context_save_load() {
        let _lock = setup();
        assert_eq!(nsl_backward_ctx_new(4), 0);

        nsl_backward_ctx_save(0, 0x1234);
        nsl_backward_ctx_save(2, 0x5678);

        assert_eq!(nsl_backward_ctx_load(0), 0x1234);
        assert_eq!(nsl_backward_ctx_load(1), 0); // empty
        assert_eq!(nsl_backward_ctx_load(2), 0x5678);
        assert_eq!(nsl_backward_ctx_load(99), 0); // out of range

        assert_eq!(nsl_backward_ctx_free(), 0);
    }

    #[test]
    fn context_double_init_fails() {
        let _lock = setup();
        assert_eq!(nsl_backward_ctx_new(2), 0);
        assert_eq!(nsl_backward_ctx_new(2), -1);
        assert_eq!(nsl_backward_ctx_free(), 0);
    }

    #[test]
    fn context_clear() {
        let mut ctx = BackwardContext::new(3);
        ctx.save(0, 42);
        ctx.save(1, 99);
        ctx.clear();
        assert_eq!(ctx.load(0), 0);
        assert_eq!(ctx.load(1), 0);
    }
}
```

Wire into `crates/nsl-runtime/src/lib.rs`:
```rust
pub mod backward_context;
```

---

## Phase 3: CLI + Builtins + CompileOptions

### Task 3: CLI Flag + Builtin Registration

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

- [ ] **Step 3: Add --tape-ad flag, CompileOptions field, and register BackwardContext FFI**

Add to CLI `Run` and `Build` commands:
```rust
/// Force tape-based AD (disable source-to-source AD)
#[arg(long)]
tape_ad: bool,
```

Add to `CompileOptions` in `lib.rs`:
```rust
/// M40b: Force tape-based AD (disable source-to-source AD).
pub tape_ad: bool,
```
Default: `false`.

Add to `builtins.rs` RUNTIME_FUNCTIONS:
```rust
// M40b: Backward context for source-to-source AD
("nsl_backward_ctx_new", &[types::I64], Some(types::I64)),
("nsl_backward_ctx_save", &[types::I64, types::I64], Some(types::I64)),
("nsl_backward_ctx_load", &[types::I64], Some(types::I64)),
("nsl_backward_ctx_free", &[], Some(types::I64)),
```

---

## Phase 4: Build Verification

- [ ] **Step 4: `cargo build` — verify no compile errors**

- [ ] **Step 5: `cargo test` — expect 8+ new tests passing**

Expected new tests:
- `source_ad.rs`: `extract_simple_binary_op`, `extract_registers_params_and_inputs`, `extract_detects_dynamic_while`, `extract_detects_dynamic_for`, `extractor_finalize_none_when_empty` (5 tests)
- `backward_context.rs`: `context_save_load`, `context_double_init_fails`, `context_clear` (3 tests)

- [ ] **Step 6: `cargo clippy` — no warnings**

---

## Verification Checklist

1. **Static graph detection**: Simple expressions extract to WengertList; while/for/if return None
2. **Param/input registration**: Symbols map to VarIds, params tracked for gradient targets
3. **Binary op extraction**: Add/Sub/Mul/Div mapped to correct PrimalOps
4. **Call extraction**: relu/sigmoid/matmul/etc mapped to correct PrimalOps
5. **BackwardContext**: save/load with slot indexing, out-of-range returns 0
6. **FFI lifecycle**: new/save/load/free with global Mutex pattern
7. **--tape-ad flag**: parses correctly, flows to CompileOptions
8. **Builtins**: 4 FFI functions registered
9. **No regressions**: All 606+ existing tests pass
