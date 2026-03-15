# M28: Dynamic Shapes & Ragged Tensors Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the AOT compiler from fully static tensor shapes to supporting symbolic/dynamic dimensions that resolve at runtime, enabling variable-length sequences for inference.

**Architecture:** Extend the existing `Dim` enum with `Bounded(Symbol, i64)` and `Computed(DimExpr)` variants. Add `DimExpr` arithmetic type for tracking symbolic relationships through operations. Codegen emits runtime shape queries (`nsl_tensor_shape_dim`) for symbolic dimensions and runtime assertions for dimension unification. The runtime `NslTensor` struct already has dynamic `shape`/`strides` arrays — no struct changes needed. Static dimensions retain zero overhead (hardcoded constants in Cranelift IR).

**Tech Stack:** Rust, Cranelift, PTX (for GPU stride parameters)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/nsl-ast/src/types.rs` | Add `DimExpr::Bounded` AST variant |
| `crates/nsl-parser/src/types.rs` | Parse `SeqLen < 4096` bounded syntax |
| `crates/nsl-semantic/src/types.rs` | Add `Dim::Bounded`, `Dim::Computed`, `DimExpr` arithmetic enum |
| `crates/nsl-semantic/src/resolve.rs` | Resolve `DimExpr::Bounded` → `Dim::Bounded` |
| `crates/nsl-semantic/src/shapes.rs` | Extend `unify_dim`/`broadcast_dim` for `Bounded`/`Computed`; add `has_symbolic()` |
| `crates/nsl-semantic/src/checker.rs` | Validate symbolic dim consistency in function signatures |
| `crates/nsl-codegen/src/dynamic_shapes.rs` | **New**: symbolic dim tracking, runtime assertion codegen, shape query helpers |
| `crates/nsl-codegen/src/expr.rs` | Emit runtime shape queries for symbolic dims in tensor ops |
| `crates/nsl-codegen/src/compiler.rs` | Wire up dynamic shape context, pass symbolic info to codegen |
| `crates/nsl-runtime/src/tensor.rs` | Add `nsl_tensor_assert_dim` runtime assertion FFI |
| `crates/nsl-codegen/src/builtins.rs` | Register `nsl_tensor_assert_dim` builtin |
| `examples/m28_dynamic_shapes.nsl` | E2E test: symbolic dims with runtime varying shapes |
| `examples/m28_bounded_dims.nsl` | E2E test: bounded dims with assertion |
| `tests/expected/m28_dynamic_shapes.txt` | Expected output |
| `tests/expected/m28_bounded_dims.txt` | Expected output |
| `crates/nsl-cli/tests/e2e.rs` | E2E test entries |

---

## Chunk 1: Type System Foundation

### Task 1: Add `Bounded` variant to AST `DimExpr`

**Files:**
- Modify: `crates/nsl-ast/src/types.rs:71-81`

- [ ] **Step 1: Write the failing test**

No separate test file — this is a data type addition. Verified by parser tests in Task 2.

- [ ] **Step 2: Add `Bounded` variant to `DimExpr`**

In `crates/nsl-ast/src/types.rs`, add a new variant to the `DimExpr` enum:

```rust
#[derive(Debug, Clone, Serialize)]
pub enum DimExpr {
    /// Concrete dimension: 768
    Concrete(i64),
    /// Symbolic dimension: batch, seq
    Symbolic(Symbol),
    /// Named dimension: batch="B", heads=12
    Named { name: Symbol, value: DimValue },
    /// Bounded symbolic dimension: SeqLen < 4096
    Bounded { name: Symbol, upper_bound: i64 },
    /// Wildcard: _
    Wildcard,
}
```

- [ ] **Step 3: Verify build**

Run: `cargo build -p nsl-ast`
Expected: Compiles (may have warnings about non-exhaustive match elsewhere — that's expected and will be fixed in subsequent tasks)

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-ast/src/types.rs
git commit -m "feat(m28): add Bounded variant to AST DimExpr"
```

---

### Task 2: Parse `SeqLen < 4096` bounded dimension syntax

**Files:**
- Modify: `crates/nsl-parser/src/types.rs`

The parser's `parse_dim_expr()` function currently handles `IntLiteral`, `Underscore`, and `Ident` (with optional `=`). We need to extend the `Ident` path: after consuming an identifier, peek for `<` (less-than). If found, consume it and expect an `IntLiteral` for the upper bound.

**Important:** The `<` token is `TokenKind::Lt` in the lexer. We must be careful not to confuse it with the generic type closer `>` (`TokenKind::Gt`).

- [ ] **Step 1: Write the failing test**

Add a test to `crates/nsl-parser/src/types.rs` (or a new test file) that parses `Tensor<[Batch, SeqLen < 4096], f32>` and asserts the second dim is `DimExpr::Bounded { name: "SeqLen", upper_bound: 4096 }`.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bounded_dim() {
        // Test that "SeqLen < 4096" inside a tensor type parses as Bounded
        let source = "let x: Tensor<[Batch, SeqLen < 4096], f32> = zeros([1, 1])";
        let mut interner = nsl_lexer::Interner::new();
        let tokens = nsl_lexer::lex(source, &mut interner);
        let module = crate::parse(tokens, &mut interner);
        // Should parse without errors — the VarDecl's type annotation has Bounded dim
        assert!(module.stmts.len() == 1);
        // Extract type annotation and verify shape
        if let nsl_ast::stmt::StmtKind::VarDecl(decl) = &module.stmts[0].kind {
            if let Some(type_expr) = &decl.type_annotation {
                if let nsl_ast::types::TypeExprKind::Tensor { shape, .. } = &type_expr.kind {
                    assert_eq!(shape.len(), 2);
                    assert!(matches!(&shape[1], nsl_ast::types::DimExpr::Bounded { upper_bound: 4096, .. }));
                    return;
                }
            }
        }
        panic!("Expected VarDecl with Tensor type containing Bounded dim");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-parser -- parse_bounded_dim`
Expected: FAIL (parser doesn't handle `<` after ident in dim position)

- [ ] **Step 3: Extend `parse_dim_expr` to handle bounded syntax**

In `crates/nsl-parser/src/types.rs`, in the `parse_dim_expr()` function, modify the `Ident` arm. After consuming the ident, check for `Lt` token:

```rust
TokenKind::Ident(sym) => {
    let sym = *sym;
    p.advance();
    if p.check(&TokenKind::Eq) {
        // ... existing Named handling ...
    } else if p.check(&TokenKind::Lt) {
        // Bounded: SeqLen < 4096
        p.advance(); // consume <
        match &p.peek().kind {
            TokenKind::IntLiteral(v) => {
                let bound = *v;
                p.advance();
                DimExpr::Bounded { name: Symbol(sym.0), upper_bound: bound }
            }
            _ => {
                p.error_at_current("expected integer upper bound after '<'");
                DimExpr::Symbolic(Symbol(sym.0))
            }
        }
    } else {
        DimExpr::Symbolic(Symbol(sym.0))
    }
}
```

**Note:** Look at the existing code to match the exact token variant names. The `Ident` arm may destructure differently — adapt accordingly. The key insight: `p.check(&TokenKind::Lt)` peeks without consuming; `p.advance()` consumes.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p nsl-parser -- parse_bounded_dim`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-parser/src/types.rs
git commit -m "feat(m28): parse bounded dimension syntax (SeqLen < 4096)"
```

---

### Task 3: Add `Bounded` and `Computed` to semantic `Dim`, add `DimExpr` arithmetic type

**Files:**
- Modify: `crates/nsl-semantic/src/types.rs:135-167`

- [ ] **Step 1: Add `DimExpr` arithmetic enum**

Add below the `Dim` enum in `crates/nsl-semantic/src/types.rs`:

```rust
/// Arithmetic expression over symbolic dimensions.
/// Tracks how dimensions compose through reshape/concat/split.
#[derive(Debug, Clone, PartialEq)]
pub enum DimExpr {
    Sym(Symbol),
    Lit(i64),
    Add(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    Div(Box<DimExpr>, Box<DimExpr>),
}
```

- [ ] **Step 2: Extend `Dim` enum with `Bounded` and `Computed`**

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    /// Known concrete size.
    Concrete(i64),
    /// Symbolic: same name within a scope must unify to same size.
    Symbolic(Symbol),
    /// Named dimension with a label and optional concrete/symbolic size.
    Named { name: Symbol, size: Box<Dim> },
    /// Bounded symbolic: resolved at runtime, with compile-time upper bound.
    Bounded { name: Symbol, upper_bound: i64 },
    /// Computed: arithmetic over other dims (e.g. from reshape).
    Computed(Box<DimExpr>),
    /// Wildcard: unchecked.
    Wildcard,
}
```

- [ ] **Step 3: Add `Shape::has_symbolic()` helper**

Add to the `impl Shape` block:

```rust
/// Returns true if any dimension is symbolic, bounded, or computed.
pub fn has_symbolic(&self) -> bool {
    self.dims.iter().any(|d| matches!(d, Dim::Symbolic(_) | Dim::Bounded { .. } | Dim::Computed(_)))
}
```

- [ ] **Step 4: Verify build**

Run: `cargo build -p nsl-semantic`
Expected: Errors in `shapes.rs` and `resolve.rs` about non-exhaustive match on `Dim` — expected, fixed in next tasks.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-semantic/src/types.rs
git commit -m "feat(m28): add Bounded, Computed dims and DimExpr arithmetic type"
```

---

### Task 4: Update type resolver for `Bounded` dims

**Files:**
- Modify: `crates/nsl-semantic/src/resolve.rs`

- [ ] **Step 1: Handle `DimExpr::Bounded` in `resolve_dim()`**

In `resolve_dim()`, add a match arm for the new AST `DimExpr::Bounded` variant:

```rust
nsl_ast::types::DimExpr::Bounded { name, upper_bound } => {
    Dim::Bounded { name: *name, upper_bound: *upper_bound }
}
```

**Note:** The AST type is `nsl_ast::types::DimExpr` while the semantic type is `crate::types::Dim`. Make sure to match the correct enum.

- [ ] **Step 2: Verify build**

Run: `cargo build -p nsl-semantic`
Expected: Still may fail on `shapes.rs` match arms — that's Task 5.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-semantic/src/resolve.rs
git commit -m "feat(m28): resolve Bounded AST dims to semantic Bounded dims"
```

---

### Task 5: Extend shape checking for `Bounded` and `Computed` dims

**Files:**
- Modify: `crates/nsl-semantic/src/shapes.rs`

- [ ] **Step 1: Write failing tests**

Add tests to `crates/nsl-semantic/src/shapes.rs`:

```rust
#[test]
fn unify_bounded_with_concrete() {
    let sym = make_sym(10);
    // Bounded(SeqLen, 4096) unifies with Concrete(512) → Concrete(512)
    assert_eq!(
        unify_dim(&Dim::Bounded { name: sym, upper_bound: 4096 }, &Dim::Concrete(512)),
        Some(Dim::Concrete(512))
    );
}

#[test]
fn unify_bounded_with_concrete_exceeds() {
    let sym = make_sym(10);
    // Bounded(SeqLen, 4096) unifies with Concrete(8192) → None (exceeds bound)
    assert_eq!(
        unify_dim(&Dim::Bounded { name: sym, upper_bound: 4096 }, &Dim::Concrete(8192)),
        None
    );
}

#[test]
fn unify_bounded_with_symbolic() {
    let sym = make_sym(10);
    // Bounded(SeqLen, 4096) + Symbolic(SeqLen) → Bounded(SeqLen, 4096)
    assert_eq!(
        unify_dim(&Dim::Bounded { name: sym, upper_bound: 4096 }, &Dim::Symbolic(sym)),
        Some(Dim::Bounded { name: sym, upper_bound: 4096 })
    );
}

#[test]
fn unify_two_bounded_same_name() {
    let sym = make_sym(10);
    // Bounded(S, 4096) + Bounded(S, 2048) → Bounded(S, 2048) (tighter bound)
    assert_eq!(
        unify_dim(
            &Dim::Bounded { name: sym, upper_bound: 4096 },
            &Dim::Bounded { name: sym, upper_bound: 2048 }
        ),
        Some(Dim::Bounded { name: sym, upper_bound: 2048 })
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p nsl-semantic -- unify_bounded`
Expected: FAIL (non-exhaustive match)

- [ ] **Step 3: Extend `unify_dim()` with `Bounded` and `Computed` arms**

In `unify_dim()`, add these match arms (before the final catch-all):

```rust
// Bounded unifies with Concrete if within bound
(Dim::Bounded { upper_bound, .. }, Dim::Concrete(n))
| (Dim::Concrete(n), Dim::Bounded { upper_bound, .. }) => {
    if *n <= *upper_bound { Some(Dim::Concrete(*n)) } else { None }
}

// Bounded unifies with same-named Symbolic → keeps bound
(Dim::Bounded { name: n1, upper_bound }, Dim::Symbolic(n2))
| (Dim::Symbolic(n2), Dim::Bounded { name: n1, upper_bound }) if n1 == n2 => {
    Some(Dim::Bounded { name: *n1, upper_bound: *upper_bound })
}

// Bounded unifies with different Symbolic → None
(Dim::Bounded { .. }, Dim::Symbolic(_))
| (Dim::Symbolic(_), Dim::Bounded { .. }) => None,

// Two Bounded with same name → take tighter bound
(Dim::Bounded { name: n1, upper_bound: u1 }, Dim::Bounded { name: n2, upper_bound: u2 })
    if n1 == n2 => {
    Some(Dim::Bounded { name: *n1, upper_bound: *u1.min(u2) })
}

// Two Bounded with different names → None
(Dim::Bounded { .. }, Dim::Bounded { .. }) => None,

// Computed: treat as Wildcard for now (runtime-checked)
(Dim::Computed(_), _) | (_, Dim::Computed(_)) => Some(Dim::Wildcard),
```

- [ ] **Step 4: Update `fmt_dim()` for new variants**

```rust
Dim::Bounded { upper_bound, .. } => format!("<{}", upper_bound),
Dim::Computed(_) => "<computed>".into(),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p nsl-semantic -- unify_bounded`
Expected: PASS

- [ ] **Step 6: Verify full semantic crate builds**

Run: `cargo build -p nsl-semantic`
Expected: Clean build (all match arms covered)

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-semantic/src/shapes.rs
git commit -m "feat(m28): extend unify_dim for Bounded and Computed dimensions"
```

---

### Task 6: Fix downstream compilation (codegen, parser exhaustive matches)

**Files:**
- Modify: `crates/nsl-codegen/src/expr.rs` (any match on `Dim` or AST `DimExpr`)
- Modify: `crates/nsl-semantic/src/checker.rs` (any match on `Dim`)
- Modify: any other files with non-exhaustive matches

- [ ] **Step 1: Build full project to find all non-exhaustive match errors**

Run: `cargo build 2>&1 | grep "non-exhaustive"`
Expected: List of files/lines with non-exhaustive match on `Dim` or `DimExpr`

- [ ] **Step 2: Fix each non-exhaustive match**

For each error, add the missing arms:
- `Dim::Bounded { .. }` → treat like `Dim::Symbolic` (use shape query at runtime)
- `Dim::Computed(_)` → treat like `Dim::Wildcard` (runtime-determined)
- `DimExpr::Bounded { .. }` → similar to `DimExpr::Symbolic`

**Pattern:** In codegen matches on `Dim`, `Bounded` and `Computed` should fall through to the same path as `Symbolic` — they all resolve at runtime.

- [ ] **Step 3: Verify full build**

Run: `cargo build`
Expected: Clean build

- [ ] **Step 4: Run all tests**

Run: `cargo test`
Expected: All 148+ tests pass (no regressions)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "fix(m28): handle Bounded and Computed dims in all match expressions"
```

---

## Chunk 2: Runtime Assertions and Codegen

### Task 7: Add `nsl_tensor_assert_dim` runtime function

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

This function is called by codegen when two symbolic dims must unify at runtime. It asserts that a tensor's dimension `dim_index` equals `expected_value`, aborting with a clear error if not.

- [ ] **Step 1: Write the runtime function**

Add to `crates/nsl-runtime/src/tensor.rs`:

```rust
/// Assert that tensor dimension `dim_index` equals `expected_value`.
/// Called by codegen to enforce symbolic dimension unification at runtime.
/// If `expected_value` is -1, this is a "record" call — returns the actual dim value.
/// Otherwise, asserts equality and aborts on mismatch.
#[no_mangle]
pub extern "C" fn nsl_tensor_assert_dim(tensor_ptr: i64, dim_index: i64, expected_value: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let dim_idx = if dim_index < 0 {
        (ndim as i64 + dim_index) as usize
    } else {
        dim_index as usize
    };
    if dim_idx >= ndim {
        eprintln!(
            "nsl: assert_dim: dimension index {} out of range for rank-{} tensor",
            dim_index, ndim
        );
        std::process::abort();
    }
    let actual = unsafe { *tensor.shape.add(dim_idx) };
    if expected_value == -1 {
        // Record mode: just return the actual dim
        return actual;
    }
    if actual != expected_value {
        eprintln!(
            "nsl: dimension mismatch: expected dim[{}] = {}, got {}",
            dim_index, expected_value, actual
        );
        std::process::abort();
    }
    actual
}

/// Assert that a tensor dimension does not exceed an upper bound.
/// Used for `Bounded` dimensions (e.g., `SeqLen < 4096`).
#[no_mangle]
pub extern "C" fn nsl_tensor_assert_dim_bound(tensor_ptr: i64, dim_index: i64, upper_bound: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let dim_idx = if dim_index < 0 {
        (ndim as i64 + dim_index) as usize
    } else {
        dim_index as usize
    };
    if dim_idx >= ndim {
        eprintln!(
            "nsl: assert_dim_bound: dimension index {} out of range for rank-{} tensor",
            dim_index, ndim
        );
        std::process::abort();
    }
    let actual = unsafe { *tensor.shape.add(dim_idx) };
    if actual > upper_bound {
        eprintln!(
            "nsl: dimension bound exceeded: dim[{}] = {} exceeds upper bound {}",
            dim_index, actual, upper_bound
        );
        std::process::abort();
    }
    actual
}
```

- [ ] **Step 2: Register in codegen builtins**

In `crates/nsl-codegen/src/builtins.rs`, add:

```rust
// M28: Dynamic shape assertions
def("nsl_tensor_assert_dim", &[I64, I64, I64], Some(I64));
def("nsl_tensor_assert_dim_bound", &[I64, I64, I64], Some(I64));
```

- [ ] **Step 3: Verify build**

Run: `cargo build`
Expected: Clean build

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-codegen/src/builtins.rs
git commit -m "feat(m28): add nsl_tensor_assert_dim and assert_dim_bound runtime FFI"
```

---

### Task 8: Create `dynamic_shapes.rs` codegen module

**Files:**
- Create: `crates/nsl-codegen/src/dynamic_shapes.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

This module provides helpers that codegen calls to emit runtime shape queries and assertions for functions with symbolic parameters.

- [ ] **Step 1: Create the module**

Create `crates/nsl-codegen/src/dynamic_shapes.rs`:

```rust
//! Dynamic shape codegen helpers for M28.
//!
//! When a function has parameters with symbolic dimensions (e.g. `Tensor<[Batch, SeqLen], f32>`),
//! codegen must:
//! 1. Query the actual dimension values from the first tensor that uses each symbol
//! 2. Assert subsequent uses of the same symbol have matching dimensions
//! 3. For Bounded dims, assert the value is within the upper bound
//!
//! This module tracks which symbolic dims have been "resolved" (first seen) vs need assertion.

use std::collections::HashMap;
use nsl_ast::Symbol;
use nsl_semantic::types::{Dim, Shape};

/// Tracks symbolic dimension resolution within a function's codegen.
/// Each symbolic name maps to (tensor_param_cranelift_value, dim_index) where it was first seen.
pub struct SymbolicDimTracker {
    /// symbol_name → cranelift Value holding the resolved runtime i64 dimension value
    resolved: HashMap<Symbol, cranelift_codegen::ir::Value>,
}

impl SymbolicDimTracker {
    pub fn new() -> Self {
        Self {
            resolved: HashMap::new(),
        }
    }

    /// Check if a symbolic dim has been resolved (first tensor with this dim was seen).
    pub fn is_resolved(&self, sym: &Symbol) -> bool {
        self.resolved.contains_key(sym)
    }

    /// Record that a symbolic dim has been resolved to the given Cranelift value.
    pub fn resolve(&mut self, sym: Symbol, value: cranelift_codegen::ir::Value) {
        self.resolved.insert(sym, value);
    }

    /// Get the resolved runtime value for a symbolic dim, if available.
    pub fn get(&self, sym: &Symbol) -> Option<cranelift_codegen::ir::Value> {
        self.resolved.get(sym).copied()
    }
}

/// Analyze a function's parameter types and return all symbolic/bounded dims
/// as (param_index, dim_index, DimInfo) tuples.
pub enum DimInfo {
    /// First occurrence of this symbol — query and record
    Symbolic(Symbol),
    /// Bounded — query, record, and assert within bound
    Bounded { name: Symbol, upper_bound: i64 },
}

/// Extract symbolic dims from a shape, yielding (dim_index, DimInfo) pairs.
pub fn extract_symbolic_dims(shape: &Shape) -> Vec<(usize, DimInfo)> {
    let mut result = Vec::new();
    for (i, dim) in shape.dims.iter().enumerate() {
        match dim {
            Dim::Symbolic(sym) => {
                result.push((i, DimInfo::Symbolic(*sym)));
            }
            Dim::Bounded { name, upper_bound } => {
                result.push((i, DimInfo::Bounded { name: *name, upper_bound: *upper_bound }));
            }
            _ => {}
        }
    }
    result
}
```

- [ ] **Step 2: Add module to lib.rs**

In `crates/nsl-codegen/src/lib.rs`, add:

```rust
pub mod dynamic_shapes;
```

- [ ] **Step 3: Verify build**

Run: `cargo build -p nsl-codegen`
Expected: Clean build

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/dynamic_shapes.rs crates/nsl-codegen/src/lib.rs
git commit -m "feat(m28): create dynamic_shapes codegen module with SymbolicDimTracker"
```

---

### Task 9: Emit runtime assertions for symbolic dims in function prologues

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

When compiling a function that has parameters with symbolic dimensions, insert assertion calls at the function entry. For the first occurrence of each symbolic dim, call `nsl_tensor_assert_dim(tensor, dim_idx, -1)` to record the value. For subsequent occurrences, call `nsl_tensor_assert_dim(tensor, dim_idx, recorded_value)` to assert consistency.

**Key implementation pattern:**

In the function compilation path (look for `compile_fn_def` or similar in `compiler.rs`), after setting up parameters but before compiling the function body:

1. Scan parameter types for symbolic/bounded dims using `extract_symbolic_dims()`
2. For each symbolic dim found:
   - If first occurrence: emit `nsl_tensor_assert_dim(param, dim_idx, -1)` and store result in `SymbolicDimTracker`
   - If already resolved: emit `nsl_tensor_assert_dim(param, dim_idx, resolved_value)`
3. For bounded dims: also emit `nsl_tensor_assert_dim_bound(param, dim_idx, upper_bound)`

- [ ] **Step 1: Locate the function compilation entry point**

Search for `compile_fn_def` or `compile_function` in `compiler.rs`. The prologue assertion code goes after parameter variables are set up but before the function body is compiled.

- [ ] **Step 2: Add symbolic dim tracker to Compiler state**

In `compiler.rs`, the `Compiler` struct doesn't need a permanent field — the tracker is local to each function compilation. Create it at the start of `compile_fn_def` and pass it through.

However, `expr.rs` needs access to the tracker to use resolved symbolic values in expressions. The simplest approach: add a `symbolic_dims: Option<SymbolicDimTracker>` field to the `FuncState` in `crates/nsl-codegen/src/context.rs` (or wherever function-local state is kept).

Check the existing `FuncState` struct:

```rust
// In crates/nsl-codegen/src/context.rs
pub struct FuncState {
    // ... existing fields ...
    pub symbolic_dims: Option<crate::dynamic_shapes::SymbolicDimTracker>,
}
```

Initialize to `None` by default, set to `Some(SymbolicDimTracker::new())` when compiling functions with symbolic params.

- [ ] **Step 3: Emit assertion calls in function prologue**

In the function compilation code, after parameters are bound to variables:

```rust
// Check if any parameter has symbolic dims
let mut tracker = SymbolicDimTracker::new();
let mut has_symbolic = false;

for (param_idx, param_def) in fn_def.params.iter().enumerate() {
    if let Some(type_ann) = &param_def.type_annotation {
        let resolved_type = /* resolve the type annotation */;
        if let Type::Tensor { shape, .. } = &resolved_type {
            for (dim_idx, dim_info) in extract_symbolic_dims(shape) {
                has_symbolic = true;
                let param_val = /* get cranelift value for this param */;
                let dim_idx_const = builder.ins().iconst(I64, dim_idx as i64);

                match dim_info {
                    DimInfo::Symbolic(sym) => {
                        if tracker.is_resolved(&sym) {
                            // Assert consistency
                            let expected = tracker.get(&sym).unwrap();
                            self.compile_call_by_name(builder, "nsl_tensor_assert_dim",
                                &[param_val, dim_idx_const, expected])?;
                        } else {
                            // Record: call with -1 to get actual value
                            let neg_one = builder.ins().iconst(I64, -1);
                            let actual = self.compile_call_by_name(builder, "nsl_tensor_assert_dim",
                                &[param_val, dim_idx_const, neg_one])?;
                            tracker.resolve(sym, actual);
                        }
                    }
                    DimInfo::Bounded { name, upper_bound } => {
                        let bound_const = builder.ins().iconst(I64, upper_bound);
                        let actual = self.compile_call_by_name(builder, "nsl_tensor_assert_dim_bound",
                            &[param_val, dim_idx_const, bound_const])?;
                        if !tracker.is_resolved(&name) {
                            tracker.resolve(name, actual);
                        }
                    }
                }
            }
        }
    }
}
```

**Important:** This is pseudocode — adapt to the actual compiler API. The key functions to use are `compile_call_by_name` (already exists in expr.rs) and `builder.ins().iconst()`.

- [ ] **Step 4: Verify build**

Run: `cargo build`
Expected: Clean build

- [ ] **Step 5: Run existing tests for regression**

Run: `cargo test`
Expected: All tests pass (functions without symbolic dims should be unaffected)

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/compiler.rs crates/nsl-codegen/src/context.rs crates/nsl-codegen/src/expr.rs
git commit -m "feat(m28): emit runtime dim assertions for symbolic function parameters"
```

---

## Chunk 3: E2E Tests and Integration

### Task 10: E2E test — symbolic dims with runtime varying shapes

**Files:**
- Create: `examples/m28_dynamic_shapes.nsl`
- Create: `tests/expected/m28_dynamic_shapes.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create the test NSL file**

```nsl
# M28: Dynamic shapes — symbolic dimensions resolve at runtime

fn process(x: Tensor<[Batch, 4], f32>) -> Tensor:
    return x

# Call with different batch sizes
let a = randn([2, 4])
let b = randn([5, 4])

let out_a = process(a)
let out_b = process(b)

print("Dynamic shapes OK")
```

**Note:** `Batch` is a symbolic dim in the function signature. Calling with `[2, 4]` and `[5, 4]` should both work since the static dim (4) matches and `Batch` is unconstrained.

- [ ] **Step 2: Create expected output**

`tests/expected/m28_dynamic_shapes.txt`:
```
Dynamic shapes OK
```

- [ ] **Step 3: Add E2E test entry**

In `crates/nsl-cli/tests/e2e.rs`:

```rust
#[test]
fn e2e_m28_dynamic_shapes() {
    run_example("m28_dynamic_shapes");
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p nsl-cli --test e2e -- m28_dynamic_shapes`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/m28_dynamic_shapes.nsl tests/expected/m28_dynamic_shapes.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(m28): E2E test for symbolic dimensions with varying batch sizes"
```

---

### Task 11: E2E test — bounded dims with assertion

**Files:**
- Create: `examples/m28_bounded_dims.nsl`
- Create: `tests/expected/m28_bounded_dims.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create the test NSL file**

```nsl
# M28: Bounded dimensions — compile-time upper bound, runtime checked

fn attend(Q: Tensor<[Batch, SeqLen < 4096, 64], f32>, K: Tensor<[Batch, SeqLen < 4096, 64], f32>) -> Tensor:
    return Q

# Call within bounds
let q = randn([2, 512, 64])
let k = randn([2, 512, 64])
let out = attend(q, k)

print("Bounded dims OK")
```

- [ ] **Step 2: Create expected output**

`tests/expected/m28_bounded_dims.txt`:
```
Bounded dims OK
```

- [ ] **Step 3: Add E2E test entry**

In `crates/nsl-cli/tests/e2e.rs`:

```rust
#[test]
fn e2e_m28_bounded_dims() {
    run_example("m28_bounded_dims");
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p nsl-cli --test e2e -- m28_bounded_dims`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/m28_bounded_dims.nsl tests/expected/m28_bounded_dims.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(m28): E2E test for bounded dimensions with runtime assertion"
```

---

### Task 12: E2E test — symbolic dim unification (same symbol, multiple params)

**Files:**
- Create: `examples/m28_dim_unification.nsl`
- Create: `tests/expected/m28_dim_unification.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create the test NSL file**

```nsl
# M28: Symbolic dim unification — same symbol must match across params

fn add_tensors(a: Tensor<[N, 4], f32>, b: Tensor<[N, 4], f32>) -> Tensor:
    return a + b

let x = randn([3, 4])
let y = randn([3, 4])
let result = add_tensors(x, y)

print("Dim unification OK")
```

- [ ] **Step 2: Create expected output**

`tests/expected/m28_dim_unification.txt`:
```
Dim unification OK
```

- [ ] **Step 3: Add E2E test entry**

```rust
#[test]
fn e2e_m28_dim_unification() {
    run_example("m28_dim_unification");
}
```

- [ ] **Step 4: Run all M28 tests**

Run: `cargo test -p nsl-cli --test e2e -- m28`
Expected: All 3 M28 tests pass

- [ ] **Step 5: Commit**

```bash
git add examples/m28_dim_unification.nsl tests/expected/m28_dim_unification.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(m28): E2E test for symbolic dimension unification across parameters"
```

---

### Task 13: Final integration — full build, clippy, all tests

**Files:** None new — verification only.

- [ ] **Step 1: Full build**

Run: `cargo build`
Expected: Clean build

- [ ] **Step 2: Clippy on all crates**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: No warnings

- [ ] **Step 3: Full test suite**

Run: `cargo test`
Expected: All tests pass (148 existing + new M28 tests)

- [ ] **Step 4: Fix any issues found**

Address clippy warnings or test failures.

- [ ] **Step 5: Final commit if needed**

```bash
git add -A
git commit -m "chore(m28): final cleanup and verification"
```

---

## Deliverables Checklist

- [ ] `Tensor<[Batch, SeqLen, 4096], f16>` compiles and runs with varying shapes
- [ ] `Tensor<[Batch, SeqLen < 4096, D], f32>` compiles with bound checking
- [ ] Runtime assertions fire on dimension mismatch (same symbolic name, different values)
- [ ] Static dimensions retain zero overhead (no regression in existing tests)
- [ ] Same symbolic name across function params is unified at runtime
- [ ] All existing 148+ tests still pass
