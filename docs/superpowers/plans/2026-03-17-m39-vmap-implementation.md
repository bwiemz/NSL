# M39: Automatic Batching (vmap) — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compile-time `@vmap` AST-to-AST transformation that takes a per-example function and produces a batched version operating over an entire batch dimension — no runtime loop, fully vectorized.

**Architecture:** Two new modules: (1) `crates/nsl-codegen/src/vmap.rs` — `VmapTransformer` with `BatchTracker` for batch-variant/invariant classification, shape rewriting (batch dim insertion), dimension shifting for reductions/transpose, and matmul rewrite rules. (2) `@vmap` decorator semantic validation. The existing `nsl_tensor_matmul` already handles batched tensors via NumPy-style broadcasting, so no new runtime FFI is needed for the core matmul case. The transform produces a `_batched` FnDef that goes through standard compilation.

**Tech Stack:** Rust (AST transformation + semantic analysis)

**Spec:** `docs/superpowers/specs/2026-03-15-m39-vmap-design.md`

---

## Important: Scope of This Plan

**This plan builds the core vmap analysis and transformation infrastructure.** It delivers:
- `BatchTracker` with variant/invariant classification logic (fully unit-tested)
- Shape rewriting: batch dim insertion at arbitrary position
- Dimension index shifting for reductions, transpose, concat
- Matmul rewrite rules (variant x invariant, variant x variant, etc.)
- `VmapConfig` struct for compiler integration
- `@vmap` decorator semantic validation
- CLI infrastructure

**Key simplification:** `nsl_tensor_matmul` already supports batch broadcasting (3D+ tensors). The vmap transform just inserts a batch dimension into shapes — the existing matmul runtime handles the rest. No new `nsl_batched_matmul` FFI needed.

**Deferred to M39b:** AST-walking `VmapTransformer.transform()` (actual FnDef-to-FnDef rewriting), `@invariant` parameter annotation (requires adding `decorators` field to `Param` in AST + parser changes), nested vmap, reshape batch preservation, call-site auto-dispatch (original vs _batched), `nsl_vmap_check_batch` runtime assertion, E2E numerical validation tests.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-codegen/src/vmap.rs` | BatchTracker, VmapConfig, shape rewriting, dim shifting, matmul rewrite rules | 300 |
| `crates/nsl-semantic/src/vmap.rs` | @vmap decorator validation | 60 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod vmap;` |
| `crates/nsl-codegen/src/compiler.rs` | Add `vmap_configs: HashMap<String, VmapConfig>` field |
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod vmap;` |
| `crates/nsl-semantic/src/checker.rs` | Wire @vmap decorator validation |
| `crates/nsl-cli/tests/e2e.rs` | Add M39 E2E tests |

---

## Phase 1: Batch Tracking + Shape Rewriting

### Task 1: BatchTracker + Shape Rewriting + Dim Shifting

**Files:**
- Create: `crates/nsl-codegen/src/vmap.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create `vmap.rs` with core types, batch tracking, shape/dim rewriting, and tests**

```rust
//! M39: Automatic batching (vmap) — compile-time AST-to-AST batch transformation.

use std::collections::{HashMap, HashSet};
use nsl_ast::Symbol;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Batch status of a tensor binding within a vmapped function.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchStatus {
    /// Has the batch dimension (function args, derived values).
    Variant,
    /// No batch dimension (model weights, @invariant params, constants).
    Invariant,
    /// Not yet classified.
    Unknown,
}

/// Configuration for a @vmap-decorated function.
#[derive(Debug, Clone)]
pub struct VmapConfig {
    /// Position where the batch dimension is inserted (0 = leading).
    pub batch_dim: usize,
    /// Symbol for the batch dimension name (default: "Batch").
    pub batch_sym: Symbol,
    /// Parameters annotated @invariant — excluded from batch dim insertion.
    pub invariant_params: HashSet<Symbol>,
}

// ---------------------------------------------------------------------------
// Batch tracking
// ---------------------------------------------------------------------------

/// Tracks which tensor bindings are batch-variant vs batch-invariant.
pub struct BatchTracker {
    statuses: HashMap<Symbol, BatchStatus>,
}

impl BatchTracker {
    pub fn new() -> Self {
        Self {
            statuses: HashMap::new(),
        }
    }

    /// Mark a symbol as batch-variant (has batch dimension).
    pub fn mark_variant(&mut self, sym: Symbol) {
        self.statuses.insert(sym, BatchStatus::Variant);
    }

    /// Mark a symbol as batch-invariant (no batch dimension).
    pub fn mark_invariant(&mut self, sym: Symbol) {
        self.statuses.insert(sym, BatchStatus::Invariant);
    }

    /// Get the batch status of a symbol.
    pub fn status(&self, sym: &Symbol) -> BatchStatus {
        self.statuses.get(sym).cloned().unwrap_or(BatchStatus::Unknown)
    }

    /// Classify the result of a binary operation: variant if either operand is variant.
    pub fn classify_binary(&self, left: &Symbol, right: &Symbol) -> BatchStatus {
        let l = self.status(left);
        let r = self.status(right);
        if l == BatchStatus::Variant || r == BatchStatus::Variant {
            BatchStatus::Variant
        } else if l == BatchStatus::Invariant && r == BatchStatus::Invariant {
            BatchStatus::Invariant
        } else {
            BatchStatus::Unknown
        }
    }

    /// Classify a call result: variant if any argument is variant.
    pub fn classify_call(&self, args: &[Symbol]) -> BatchStatus {
        if args.iter().any(|a| self.status(a) == BatchStatus::Variant) {
            BatchStatus::Variant
        } else if args.iter().all(|a| self.status(a) == BatchStatus::Invariant) {
            BatchStatus::Invariant
        } else {
            BatchStatus::Unknown
        }
    }
}

impl Default for BatchTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shape rewriting
// ---------------------------------------------------------------------------

/// Insert a batch dimension at position `batch_dim` into a shape (as dim count vector).
/// Returns the new shape dimension count. Only applies to batch-variant tensors.
pub fn insert_batch_dim(original_ndim: usize, batch_dim: usize, status: BatchStatus) -> usize {
    if status != BatchStatus::Variant {
        return original_ndim;
    }
    original_ndim + 1
}

/// Shift a dimension index to account for an inserted batch dimension.
/// Positive dims >= batch_dim are shifted up by 1.
/// Negative dims are converted to positive using original_ndim, shifted, then converted back.
pub fn shift_dim(dim: i64, batch_dim: usize, original_ndim: usize) -> i64 {
    if dim >= 0 {
        if dim as usize >= batch_dim {
            dim + 1
        } else {
            dim
        }
    } else {
        // Convert negative to positive: dim=-1 on ndim=2 -> abs=1
        let abs_dim = (original_ndim as i64 + dim) as usize;
        // Shift the absolute dim
        let shifted = if abs_dim >= batch_dim { abs_dim + 1 } else { abs_dim };
        // Convert back to negative relative to new ndim (original + 1)
        shifted as i64 - (original_ndim as i64 + 1)
    }
}

// ---------------------------------------------------------------------------
// Matmul rewrite classification
// ---------------------------------------------------------------------------

/// How a matmul should be rewritten based on operand batch status.
#[derive(Debug, Clone, PartialEq)]
pub enum MatmulRewrite {
    /// No rewriting — both operands are invariant.
    NoRewrite,
    /// Left operand is batched, right is not: [B,M,K] @ [K,N] -> [B,M,N]
    /// Existing nsl_tensor_matmul handles this via broadcast.
    LeftBatched,
    /// Both operands are batched: [B,M,K] @ [B,K,N] -> [B,M,N]
    BothBatched,
    /// Right operand is batched, left is not: [M,K] @ [B,K,N] -> [B,M,N]
    RightBatched,
}

/// Classify how a matmul should be rewritten.
pub fn classify_matmul_rewrite(left_status: BatchStatus, right_status: BatchStatus) -> MatmulRewrite {
    match (left_status, right_status) {
        (BatchStatus::Variant, BatchStatus::Invariant) => MatmulRewrite::LeftBatched,
        (BatchStatus::Variant, BatchStatus::Variant) => MatmulRewrite::BothBatched,
        (BatchStatus::Invariant, BatchStatus::Variant) => MatmulRewrite::RightBatched,
        _ => MatmulRewrite::NoRewrite,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::Symbol;

    type Interner = string_interner::StringInterner<string_interner::backend::BucketBackend<string_interner::DefaultSymbol>>;

    fn sym(interner: &mut Interner, name: &str) -> Symbol {
        Symbol(interner.get_or_intern(name))
    }

    // --- BatchTracker ---

    #[test]
    fn test_batch_tracker_default_unknown() {
        let mut interner = Interner::new();
        let x = sym(&mut interner, "x");
        let tracker = BatchTracker::new();
        assert_eq!(tracker.status(&x), BatchStatus::Unknown);
    }

    #[test]
    fn test_batch_tracker_variant() {
        let mut interner = Interner::new();
        let x = sym(&mut interner, "x");
        let mut tracker = BatchTracker::new();
        tracker.mark_variant(x);
        assert_eq!(tracker.status(&x), BatchStatus::Variant);
    }

    #[test]
    fn test_batch_tracker_invariant() {
        let mut interner = Interner::new();
        let w = sym(&mut interner, "W");
        let mut tracker = BatchTracker::new();
        tracker.mark_invariant(w);
        assert_eq!(tracker.status(&w), BatchStatus::Invariant);
    }

    #[test]
    fn test_classify_binary_variant_propagates() {
        let mut interner = Interner::new();
        let x = sym(&mut interner, "x");
        let w = sym(&mut interner, "W");
        let mut tracker = BatchTracker::new();
        tracker.mark_variant(x);
        tracker.mark_invariant(w);
        assert_eq!(tracker.classify_binary(&x, &w), BatchStatus::Variant);
    }

    #[test]
    fn test_classify_binary_both_invariant() {
        let mut interner = Interner::new();
        let a = sym(&mut interner, "a");
        let b = sym(&mut interner, "b");
        let mut tracker = BatchTracker::new();
        tracker.mark_invariant(a);
        tracker.mark_invariant(b);
        assert_eq!(tracker.classify_binary(&a, &b), BatchStatus::Invariant);
    }

    #[test]
    fn test_classify_call_any_variant() {
        let mut interner = Interner::new();
        let x = sym(&mut interner, "x");
        let w = sym(&mut interner, "W");
        let b = sym(&mut interner, "b");
        let mut tracker = BatchTracker::new();
        tracker.mark_variant(x);
        tracker.mark_invariant(w);
        tracker.mark_invariant(b);
        assert_eq!(tracker.classify_call(&[x, w, b]), BatchStatus::Variant);
    }

    #[test]
    fn test_classify_call_all_invariant() {
        let mut interner = Interner::new();
        let w = sym(&mut interner, "W");
        let b = sym(&mut interner, "b");
        let mut tracker = BatchTracker::new();
        tracker.mark_invariant(w);
        tracker.mark_invariant(b);
        assert_eq!(tracker.classify_call(&[w, b]), BatchStatus::Invariant);
    }

    // --- Shape rewriting ---

    #[test]
    fn test_insert_batch_dim_variant() {
        // [S, D] with batch_dim=0 -> [B, S, D] (ndim 2 -> 3)
        assert_eq!(insert_batch_dim(2, 0, BatchStatus::Variant), 3);
    }

    #[test]
    fn test_insert_batch_dim_invariant_unchanged() {
        assert_eq!(insert_batch_dim(2, 0, BatchStatus::Invariant), 2);
    }

    #[test]
    fn test_insert_batch_dim_at_position_1() {
        // [H, W] with batch_dim=1 -> [H, B, W] (ndim 2 -> 3)
        assert_eq!(insert_batch_dim(2, 1, BatchStatus::Variant), 3);
    }

    // --- Dimension shifting ---

    #[test]
    fn test_shift_dim_at_or_after_batch() {
        // [S, D] (ndim=2) with batch_dim=0: dim 0 -> 1, dim 1 -> 2
        assert_eq!(shift_dim(0, 0, 2), 1);
        assert_eq!(shift_dim(1, 0, 2), 2);
    }

    #[test]
    fn test_shift_dim_before_batch() {
        // [H, W] (ndim=2) with batch_dim=1: dim 0 stays 0 (before batch)
        assert_eq!(shift_dim(0, 1, 2), 0);
        assert_eq!(shift_dim(1, 1, 2), 2); // at batch_dim, shifts
    }

    #[test]
    fn test_shift_dim_negative_batch_dim_0() {
        // [S, D] (ndim=2) with batch_dim=0 -> [B, S, D] (ndim=3)
        // dim=-1 on [S,D] refers to D (abs=1). In [B,S,D], D is abs=2 = dim -1. So -1 stays -1.
        assert_eq!(shift_dim(-1, 0, 2), -1);
        // dim=-2 on [S,D] refers to S (abs=0). In [B,S,D], S is abs=1 = dim -2. So -2 stays -2.
        assert_eq!(shift_dim(-2, 0, 2), -2);
    }

    #[test]
    fn test_shift_dim_negative_batch_dim_1() {
        // [H, W] (ndim=2) with batch_dim=1 -> [H, B, W] (ndim=3)
        // dim=-1 on [H,W] refers to W (abs=1). abs=1 >= batch_dim=1, shifts to abs=2 -> dim -1. So -1 stays -1.
        assert_eq!(shift_dim(-1, 1, 2), -1);
        // dim=-2 on [H,W] refers to H (abs=0). abs=0 < batch_dim=1, no shift -> abs=0 -> dim -3. So -2 becomes -3.
        assert_eq!(shift_dim(-2, 1, 2), -3);
    }

    // --- Matmul rewrite ---

    #[test]
    fn test_matmul_left_batched() {
        assert_eq!(
            classify_matmul_rewrite(BatchStatus::Variant, BatchStatus::Invariant),
            MatmulRewrite::LeftBatched
        );
    }

    #[test]
    fn test_matmul_both_batched() {
        assert_eq!(
            classify_matmul_rewrite(BatchStatus::Variant, BatchStatus::Variant),
            MatmulRewrite::BothBatched
        );
    }

    #[test]
    fn test_matmul_right_batched() {
        assert_eq!(
            classify_matmul_rewrite(BatchStatus::Invariant, BatchStatus::Variant),
            MatmulRewrite::RightBatched
        );
    }

    #[test]
    fn test_matmul_no_rewrite() {
        assert_eq!(
            classify_matmul_rewrite(BatchStatus::Invariant, BatchStatus::Invariant),
            MatmulRewrite::NoRewrite
        );
    }
}
```

- [ ] **Step 2: Add `pub mod vmap;` to codegen lib.rs**

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-codegen vmap -- --nocapture
git commit -m "feat(m39): add BatchTracker, shape rewriting, dim shifting, matmul rewrite rules"
```

---

## Phase 2: Semantic Validation + Compiler Integration

### Task 2: @vmap Decorator Validation

**Files:**
- Create: `crates/nsl-semantic/src/vmap.rs`
- Modify: `crates/nsl-semantic/src/lib.rs`
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 1: Create semantic vmap.rs** (same pattern as fp8.rs, perf_budget.rs)

```rust
use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Validate @vmap decorator arguments.
/// Returns batch_dim value or None on error.
pub fn validate_vmap_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<usize> {
    let mut batch_dim: Option<usize> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "batch_dim" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n < 0 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@vmap: batch_dim must be non-negative".to_string(),
                                    )
                                    .with_label(arg.span, "negative batch_dim"),
                                );
                            } else {
                                batch_dim = Some(*n as usize);
                            }
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@vmap: batch_dim must be an integer literal".to_string(),
                                )
                                .with_label(arg.span, "expected integer"),
                            );
                        }
                    }
                    "batch_size" => {
                        // Optional — symbolic or concrete batch size. Accept for now.
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "@vmap: unknown argument '{}'", aname
                            ))
                            .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    // Default batch_dim to 0 per spec (Section 1: "Default: 0")
    Some(batch_dim.unwrap_or(0))
}
```

- [ ] **Step 2: Add `pub mod vmap;` to semantic lib.rs**

- [ ] **Step 3: Wire into checker.rs** (in the `StmtKind::Decorated` handler, near `@no_fuse` and `@shared` — NOT in the model-field decorator loop)

Find the block around line 416 that handles `@no_fuse`, `@shared`, `@flash_attention` etc. for decorated statements. Add there:

```rust
// M39: @vmap decorator validation
if dname == "vmap" {
    let resolve = |s: nsl_ast::Symbol| -> String {
        self.interner
            .resolve(s.0)
            .unwrap_or("")
            .to_string()
    };
    crate::vmap::validate_vmap_decorator(
        deco,
        &resolve,
        &mut self.diagnostics,
    );
}
```

- [ ] **Step 4: Verify, commit**

```bash
cargo check -p nsl-semantic
git commit -m "feat(m39): add @vmap semantic validation"
```

---

### Task 3: Compiler Fields

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 1: Add `vmap_configs` field**

After `ownership_info`:

```rust
    /// M39: Functions with @vmap decorator and their batch configuration
    pub vmap_configs: HashMap<String, crate::vmap::VmapConfig>,
```

Initialize as `vmap_configs: HashMap::new()` in `Compiler::new()`.

- [ ] **Step 2: Verify, commit**

```bash
cargo check --workspace
git commit -m "feat(m39): add vmap_configs to Compiler struct"
```

---

## Phase 3: E2E Tests + Verification

### Task 4: E2E Tests

**Files:**
- Create: `examples/m39_vmap_validation_error.nsl`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create @vmap validation error test**

```nsl
# M39: @vmap validation error — unknown argument

@vmap(unknown_arg=42)
fn bad(x: Tensor) -> Tensor:
    return x
```

- [ ] **Step 2: Add E2E test**

```rust
// ---------------------------------------------------------------------------
// M39: Automatic Batching (vmap)
// ---------------------------------------------------------------------------

#[test]
fn e2e_m39_vmap_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m39_vmap_validation_error.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m39_vmap_validation_error, but it succeeded"
    );
    assert!(
        stderr.contains("vmap") || stderr.contains("unknown argument"),
        "Expected vmap validation error in stderr, got: {}",
        stderr
    );
}
```

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-cli e2e_m39 -- --nocapture
git commit -m "test(m39): add E2E test for @vmap validation error"
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

- [ ] **Step 3: Fix any issues, commit**

```bash
git commit -m "chore(m39): fix clippy warnings and verify full test suite"
```

---

## Summary

| Task | Component | Tests |
|---|---|---|
| 1 | BatchTracker + shape rewriting + dim shifting + matmul rewrite | 17 unit |
| 2 | @vmap semantic validation | compile check |
| 3 | Compiler fields (vmap_configs) | compile check |
| 4 | E2E test (@vmap validation error) | 1 E2E |
| 5 | Full verification | all tests |

**Total: 5 tasks, ~17 unit tests + 1 E2E test**

### Deferred to M39b

- `VmapTransformer.transform()` — actual FnDef-to-FnDef AST rewriting (walk stmts, rewrite exprs)
- `@invariant` parameter annotation (requires `decorators` field on `Param` in AST + parser changes)
- Nested vmap (inside-out decorator application)
- Reshape batch preservation (batch dim must not be merged)
- Call-site auto-dispatch (original vs `_batched` based on arg rank)
- `nsl_vmap_check_batch` runtime batch-size assertion FFI
- E2E numerical validation tests (vmapped output vs manual loop)
- Softmax/LayerNorm implicit dim handling
- Indexing/slicing dim insertion
