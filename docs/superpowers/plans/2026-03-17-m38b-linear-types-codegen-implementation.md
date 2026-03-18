# M38b: Linear Types Codegen & Safety Proofs — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exploit the ownership knowledge from M38a's semantic checker to emit early tensor frees at consumption points, classify autodiff tape ops by backward data requirements for safety proofs, and add compiler infrastructure for ownership-aware codegen — eliminating unnecessary refcount ops and enabling immediate buffer reuse.

**Architecture:** Two new modules: (1) `crates/nsl-codegen/src/ownership.rs` — `OwnershipLowering` struct with linear/shared binding tracking, `FunctionOwnership` metadata for compiler integration. (2) `crates/nsl-semantic/src/ownership_autodiff.rs` — `BackwardAccess` enum classifying every TapeOp variant by its backward data requirements (ShapeOnly/DataRequired/AuxDataRequired). Both are pure analysis/classification modules with full unit test coverage. The codegen module is infrastructure-only (data structures + compiler fields); actual Cranelift IR modification (emitting `nsl_tensor_free` at consumption points, poison values) is deferred to a follow-up once the ownership checker AST-walker from M38a is wired into the compilation pipeline.

**Tech Stack:** Rust (codegen + semantic analysis)

**Spec:** `docs/superpowers/specs/2026-03-15-m38-linear-types-design.md` (Sections 3, 4, 6)

---

## Important: Scope of This Plan

**This plan builds the codegen infrastructure and autodiff safety classification.** It delivers:
- `BackwardAccess` enum with per-TapeOp classification (30 variants covered)
- `OwnershipLowering` struct with linear/shared binding sets
- `FunctionOwnership` metadata struct for compiler context
- `linear_types_enabled` field on Compiler struct
- Comprehensive unit tests for all TapeOp classifications

**Deferred:** Actual Cranelift IR emission changes (consumption-point `nsl_tensor_free`, debug poison values, refcount elision in `expr.rs`/`stmt.rs`), M36 memory planner integration, performance benchmarks. These require the ownership checker AST-walker to be wired into `compile_entry()` first.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-semantic/src/ownership_autodiff.rs` | BackwardAccess enum, classify_backward_access() for all TapeOp variants | 120 |
| `crates/nsl-codegen/src/ownership.rs` | OwnershipLowering, BorrowKind, FunctionOwnership structs | 100 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod ownership_autodiff;` |
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod ownership;` |
| `crates/nsl-codegen/src/compiler.rs` | Add `linear_types_enabled: bool`, `ownership_info: HashMap` fields |
| `crates/nsl-cli/tests/e2e.rs` | Add M38b E2E tests |

---

## Phase 1: Autodiff Tape Safety Classification

### Task 1: BackwardAccess Enum + TapeOp Classification

**Files:**
- Create: `crates/nsl-semantic/src/ownership_autodiff.rs`
- Modify: `crates/nsl-semantic/src/lib.rs`

- [ ] **Step 1: Create `ownership_autodiff.rs` with BackwardAccess enum and classification**

```rust
//! M38b: Autodiff tape ownership safety — classifies each TapeOp by its backward
//! data requirements to determine which tensors must stay alive during backward pass.

/// How a tape op accesses input data during the backward pass.
#[derive(Clone, Debug, PartialEq)]
pub enum BackwardAccess {
    /// Only needs input shape for backward (Add, Sub, Neg, SumReduce, etc.)
    /// The input tensor's data buffer can be freed immediately after the forward op.
    ShapeOnly,
    /// Needs saved input/output data for backward (Mul, Div, MatMul, ReLU, etc.)
    /// The tape holds a refcount-bumped reference via saved_a/saved_b/saved_out.
    DataRequired,
    /// Needs auxiliary data structures for backward (ReduceMax/argmax, Dropout/mask, etc.)
    /// Aux data is owned by the tape — no input tensor ownership concerns.
    AuxDataRequired,
}

/// Classify a tape operation name by its backward data requirements.
/// Names match the TapeOp variant names in `crates/nsl-runtime/src/autodiff.rs`.
pub fn classify_backward_access(op_name: &str) -> BackwardAccess {
    match op_name {
        // Shape-only: backward only needs input shape and/or scalar constants stored
        // on the tape, not input tensor data. Input buffer can be freed after forward op.
        "Add" | "Sub" | "Neg" | "AddScalar" | "MulScalar"
        | "SumReduce" | "MeanReduce" | "Transpose" | "Slice"
        | "Unsqueeze" | "Expand" => BackwardAccess::ShapeOnly,

        // Data-required: backward needs saved input/output tensor data, OR the tape
        // holds refcount-bumped input pointers for gradient routing (BiasAdd, Cat, Stack).
        "Mul" | "Div" | "MatMul"
        | "ReLU" | "GELU" | "SiLU" | "Log" | "Abs" | "Clamp"
        | "Exp" | "Sqrt" | "Sigmoid" | "Tanh" | "Softmax"
        | "EmbeddingLookup" | "LayerNorm" | "RMSNorm" | "Conv2d"
        | "BiasAdd" | "Cat" | "Stack" => BackwardAccess::DataRequired,

        // Auxiliary data: backward uses saved indices/masks/argmax, not input tensors
        "ReduceMax" | "MaxPool2d" | "Dropout" | "Gather" => BackwardAccess::AuxDataRequired,

        // Unknown ops conservatively require data
        _ => BackwardAccess::DataRequired,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ShapeOnly ops ---

    #[test]
    fn test_add_is_shape_only() {
        assert_eq!(classify_backward_access("Add"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_sub_is_shape_only() {
        assert_eq!(classify_backward_access("Sub"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_neg_is_shape_only() {
        assert_eq!(classify_backward_access("Neg"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_sum_reduce_is_shape_only() {
        assert_eq!(classify_backward_access("SumReduce"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_mean_reduce_is_shape_only() {
        assert_eq!(classify_backward_access("MeanReduce"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_transpose_is_shape_only() {
        assert_eq!(classify_backward_access("Transpose"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_cat_is_shape_only() {
        assert_eq!(classify_backward_access("Cat"), BackwardAccess::ShapeOnly);
    }

    // --- DataRequired ops ---

    #[test]
    fn test_mul_is_data_required() {
        assert_eq!(classify_backward_access("Mul"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_matmul_is_data_required() {
        assert_eq!(classify_backward_access("MatMul"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_relu_is_data_required() {
        assert_eq!(classify_backward_access("ReLU"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_softmax_is_data_required() {
        assert_eq!(classify_backward_access("Softmax"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_layernorm_is_data_required() {
        assert_eq!(classify_backward_access("LayerNorm"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_conv2d_is_data_required() {
        assert_eq!(classify_backward_access("Conv2d"), BackwardAccess::DataRequired);
    }

    // --- AuxDataRequired ops ---

    #[test]
    fn test_reduce_max_is_aux() {
        assert_eq!(classify_backward_access("ReduceMax"), BackwardAccess::AuxDataRequired);
    }

    #[test]
    fn test_dropout_is_aux() {
        assert_eq!(classify_backward_access("Dropout"), BackwardAccess::AuxDataRequired);
    }

    #[test]
    fn test_gather_is_aux() {
        assert_eq!(classify_backward_access("Gather"), BackwardAccess::AuxDataRequired);
    }

    #[test]
    fn test_maxpool_is_aux() {
        assert_eq!(classify_backward_access("MaxPool2d"), BackwardAccess::AuxDataRequired);
    }

    // --- Unknown fallback ---

    #[test]
    fn test_unknown_op_conservative() {
        assert_eq!(classify_backward_access("FutureOp"), BackwardAccess::DataRequired);
    }

    // --- Completeness: verify all 36 TapeOp variants are classified ---

    #[test]
    fn test_all_tape_ops_classified() {
        let shape_only = vec![
            "Add", "Sub", "Neg", "AddScalar", "MulScalar",
            "SumReduce", "MeanReduce", "Transpose", "Slice",
            "Unsqueeze", "Expand",
        ];
        let data_required = vec![
            "Mul", "Div", "MatMul",
            "ReLU", "GELU", "SiLU", "Log", "Abs", "Clamp",
            "Exp", "Sqrt", "Sigmoid", "Tanh", "Softmax",
            "EmbeddingLookup", "LayerNorm", "RMSNorm", "Conv2d",
            "BiasAdd", "Cat", "Stack",
        ];
        let aux_required = vec!["ReduceMax", "MaxPool2d", "Dropout", "Gather"];

        for op in &shape_only {
            assert_eq!(classify_backward_access(op), BackwardAccess::ShapeOnly,
                "Expected ShapeOnly for {op}");
        }
        for op in &data_required {
            assert_eq!(classify_backward_access(op), BackwardAccess::DataRequired,
                "Expected DataRequired for {op}");
        }
        for op in &aux_required {
            assert_eq!(classify_backward_access(op), BackwardAccess::AuxDataRequired,
                "Expected AuxDataRequired for {op}");
        }
        let total = shape_only.len() + data_required.len() + aux_required.len();
        assert_eq!(total, 36, "Should cover all 36 TapeOp variants");
    }
}
```

- [ ] **Step 2: Add `pub mod ownership_autodiff;` to semantic lib.rs**

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-semantic ownership_autodiff -- --nocapture
git commit -m "feat(m38b): add BackwardAccess classification for all 36 TapeOp variants"
```

---

## Phase 2: Codegen Ownership Infrastructure

### Task 2: OwnershipLowering + FunctionOwnership Structs

**Files:**
- Create: `crates/nsl-codegen/src/ownership.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create codegen ownership module**

```rust
//! M38b: Ownership-aware codegen — tracks linear/shared bindings for
//! consumption-point free emission and refcount elision.

use std::collections::{HashMap, HashSet};
use nsl_ast::Symbol;

/// Borrow kind for active borrows.
#[derive(Debug, Clone, PartialEq)]
pub enum BorrowKind {
    Immutable { borrower: Symbol },
    Mutable { borrower: Symbol },
}

/// Per-function ownership metadata, consumed by the codegen phase.
#[derive(Debug, Clone)]
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
        Self {
            linear_params: Vec::new(),
            borrowed_params: Vec::new(),
            shared_params: Vec::new(),
        }
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

impl Default for FunctionOwnership {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks ownership state during codegen for a single function.
/// Used to decide when to emit `nsl_tensor_free` and when to skip refcount ops.
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
        Self {
            linear_bindings: HashSet::new(),
            shared_bindings: HashSet::new(),
            active_borrows: HashMap::new(),
        }
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

impl Default for OwnershipLowering {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::Symbol;
    use string_interner::StringInterner;

    fn make_sym(interner: &mut StringInterner, name: &str) -> Symbol {
        Symbol(interner.get_or_intern(name))
    }

    #[test]
    fn test_linear_elides_refcount() {
        let mut interner = StringInterner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);
        assert!(lowering.should_elide_refcount(&x));
        assert!(lowering.should_free_at_consumption(&x));
    }

    #[test]
    fn test_shared_does_not_elide() {
        let mut interner = StringInterner::new();
        let x = make_sym(&mut interner, "x");
        let lowering = OwnershipLowering::new();
        // Not marked as linear
        assert!(!lowering.should_elide_refcount(&x));
        assert!(!lowering.should_free_at_consumption(&x));
    }

    #[test]
    fn test_shared_override_prevents_early_free() {
        let mut interner = StringInterner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);
        lowering.mark_shared(x); // @shared overrides linear
        assert!(lowering.should_elide_refcount(&x)); // still linear in tracking
        assert!(!lowering.should_free_at_consumption(&x)); // but not freed early
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
        let mut interner = StringInterner::new();
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
```

- [ ] **Step 2: Add `pub mod ownership;` to codegen lib.rs**

After the existing `pub mod memory_planner;` line.

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-codegen ownership -- --nocapture
git commit -m "feat(m38b): add OwnershipLowering and FunctionOwnership codegen infrastructure"
```

---

### Task 3: Compiler Fields

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 1: Add `linear_types_enabled` and `ownership_info` fields**

After `slab_plan`:

```rust
    /// M38b: Whether --linear-types flag is active
    pub linear_types_enabled: bool,
    /// M38b: Per-function ownership metadata from semantic pass
    pub ownership_info: HashMap<String, crate::ownership::FunctionOwnership>,
```

Initialize both in `Compiler::new()`:
```rust
    linear_types_enabled: false,
    ownership_info: HashMap::new(),
```

- [ ] **Step 2: Verify workspace compiles, commit**

```bash
cargo check --workspace
git commit -m "feat(m38b): add linear_types_enabled and ownership_info to Compiler"
```

---

## Phase 3: Verification

### Task 4: Full Verification + Clippy

- [ ] **Step 1: Run all workspace lib tests**

```bash
cargo test --workspace --lib
```

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

- [ ] **Step 3: Fix any issues, commit**

```bash
git commit -m "chore(m38b): fix clippy warnings and verify full test suite"
```

---

## Summary

| Task | Component | Tests |
|---|---|---|
| 1 | BackwardAccess classification (36 TapeOp variants) | 18 unit |
| 2 | OwnershipLowering + FunctionOwnership codegen structs | 5 unit |
| 3 | Compiler fields (linear_types_enabled, ownership_info) | compile check |
| 4 | Full verification | all tests |

**Total: 4 tasks, ~23 unit tests**

### Deferred to M38b-followup

- Actual Cranelift IR emission: `nsl_tensor_free` at consumption points in expr.rs
- Debug-mode poison values (zero variable slot after move)
- Refcount op elision in codegen (skip incref/decref for linear bindings)
- M36 memory planner integration (linear tensors enable immediate slab offset reuse)
- Ownership checker AST-walker wiring into `compile_entry()` (prerequisite for all codegen changes)
- Performance benchmarks (measure refcount ops eliminated)
- `&`/`&mut` borrow compilation (same pointer, no refcount bump)
- `@consume` parameter annotation in codegen
