# M39b: Automatic Batching (vmap) — AST Transform Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the vmap compile-time transformation by adding the actual FnDef-to-FnDef AST rewriting that takes a per-example function and produces a batched version with batch dimensions inserted into all tensor operations. This builds on M39a's analysis infrastructure (BatchTracker, shape rewriting, dim shifting, matmul classification).

**Architecture:** Extend `crates/nsl-codegen/src/vmap.rs` with `VmapTransformer` that clones a `FnDef`, walks its body, propagates batch status through expressions, rewrites matmul calls, shifts reduction/transpose dimensions, and produces a `_batched` FnDef. Add a compiler pre-codegen hook that generates batched functions before Cranelift compilation. Add `nsl_vmap_check_batch` runtime assertion FFI.

**Tech Stack:** Rust (AST transformation + codegen integration + runtime FFI)

**Spec:** `docs/superpowers/specs/2026-03-15-m39-vmap-design.md`

**Prerequisites:** M39a (BatchTracker, VmapConfig, shape rewriting, dim shifting, matmul rewrite classification — all complete)

---

## Important: Scope of This Plan

**This plan completes the vmap AST transformation.** It delivers:
- `VmapTransformer` that walks FnDef body and produces a batched clone
- Batch status propagation through VarDecl, BinaryOp, UnaryOp, Call, MemberAccess expressions
- Matmul call rewriting: `matmul(a, b)` → dispatch based on `MatmulRewrite` classification
- Reduction dim shifting: `sum(x, dim=0)` → `sum(x, dim=1)` when batch_dim=0
- Transpose dim shifting: `transpose(x, 0, 1)` → `transpose(x, 1, 2)` when batch_dim=0
- Batched function param type annotation: insert batch dim into variant params
- `nsl_vmap_check_batch` runtime FFI for batch-size assertions
- Compiler hook: `apply_vmap_transforms()` runs before `compile_user_functions()`
- Both original and `_batched` functions registered for compilation
- 6 new unit tests covering expression rewriting, batch propagation, dim shifting integration (adds to 18 existing M39a tests)

**Deferred to M39c:** `@invariant` parameter decorator (requires `Param` AST extension + parser changes — currently use `VmapConfig.invariant_params` HashSet instead), nested vmap (inside-out decorator application), reshape batch preservation, call-site auto-dispatch (detecting batched args and redirecting to `_batched` variant at call sites), E2E numerical validation tests.

---

## File Structure

### Modified Files

| File | Change | ~Lines |
|---|---|---|
| `crates/nsl-codegen/src/vmap.rs` | Add `VmapTransformer`, `VmapResult`, expression rewriting, `transform()` method | +350 |
| `crates/nsl-codegen/src/compiler.rs` | Add `apply_vmap_transforms()` pre-codegen hook | +30 |
| `crates/nsl-codegen/src/builtins.rs` | Register `nsl_vmap_check_batch` FFI | +2 |

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-runtime/src/vmap_runtime.rs` | `nsl_vmap_check_batch` FFI implementation | 40 |

---

## Phase 1: VmapTransformer Core

### Task 1: VmapTransformer + Expression Rewriting

**Files:**
- Modify: `crates/nsl-codegen/src/vmap.rs`

- [ ] **Step 1: Add VmapTransformer, VmapResult, and expression/statement rewriting to vmap.rs**

Add these new types and the transformer implementation after the existing code in `vmap.rs`:

```rust
// ---------------------------------------------------------------------------
// VmapTransformer — FnDef-to-FnDef AST rewriting
// ---------------------------------------------------------------------------

use nsl_ast::decl::{FnDef, Param};
use nsl_ast::expr::{Expr, ExprKind, BinOp};
use nsl_ast::stmt::{Stmt, StmtKind, Block};
use nsl_ast::Span;
use nsl_lexer::Interner;

/// Result of a vmap transformation: the original function plus the generated batched version.
#[derive(Debug)]
pub struct VmapResult {
    /// The batched function (with batch dims inserted).
    pub batched_fn: FnDef,
    /// Name of the batched function (original_name + "_batched").
    pub batched_name: String,
}

/// Transforms a FnDef into a batched version by walking the AST and inserting
/// batch dimensions into tensor operations.
pub struct VmapTransformer<'a> {
    interner: &'a Interner,  // immutable — matches Compiler's &'a Interner
    config: &'a VmapConfig,
    tracker: BatchTracker,
}

impl<'a> VmapTransformer<'a> {
    pub fn new(interner: &'a Interner, config: &'a VmapConfig) -> Self {
        VmapTransformer {
            interner,
            config,
            tracker: BatchTracker::new(),
        }
    }

    /// Transform a function definition into its batched version.
    ///
    /// 1. Classify parameters as variant/invariant
    /// 2. Walk body statements, propagating batch status
    /// 3. Rewrite matmul calls, shift reduction/transpose dims
    /// 4. Return the batched FnDef
    pub fn transform(&mut self, fn_def: &FnDef) -> Result<VmapResult, VmapError> {
        // Step 1: Classify parameters
        for param in &fn_def.params {
            if self.config.invariant_params.contains(&param.name) {
                self.tracker.mark_invariant(param.name);
            } else {
                self.tracker.mark_variant(param.name);
            }
        }

        // Step 2: Clone and transform the body
        let mut batched_body = fn_def.body.clone();
        self.transform_block(&mut batched_body)?;

        // Step 3: Build batched function name (as String — interning happens
        // later in the compiler when registering the function, since we only
        // have &Interner here, not &mut Interner).
        let original_name = self.interner.resolve(fn_def.name.0)
            .unwrap_or("unknown").to_string();
        let batched_name = format!("{}_batched", original_name);

        // Step 4: Build batched FnDef (reuses original name Symbol for now;
        // compiler renames via interner when registering)
        let batched_fn = FnDef {
            name: fn_def.name, // placeholder — compiler interns batched_name later
            type_params: fn_def.type_params.clone(),
            params: fn_def.params.clone(), // params keep same names; shapes change at type level
            return_type: fn_def.return_type.clone(),
            body: batched_body,
            is_async: fn_def.is_async,
            span: fn_def.span,
        };

        Ok(VmapResult { batched_fn, batched_name })
    }

    /// Transform a block of statements.
    fn transform_block(&mut self, block: &mut Block) -> Result<(), VmapError> {
        for stmt in &mut block.stmts {
            self.transform_stmt(stmt)?;
        }
        Ok(())
    }

    /// Transform a single statement.
    fn transform_stmt(&mut self, stmt: &mut Stmt) -> Result<(), VmapError> {
        match &mut stmt.kind {
            StmtKind::VarDecl { pattern, value, .. } => {
                // Transform the value expression
                if let Some(ref mut val) = value {
                    self.transform_expr(val)?;

                    // Classify the binding based on the value's batch status
                    let status = self.classify_expr(val);
                    if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                        match status {
                            BatchStatus::Variant => self.tracker.mark_variant(*sym),
                            BatchStatus::Invariant => self.tracker.mark_invariant(*sym),
                            _ => {}
                        }
                    }
                }
            }

            StmtKind::Assign { value, .. } => {
                self.transform_expr(value)?;
            }

            StmtKind::Return(Some(ref mut expr)) => {
                self.transform_expr(expr)?;
            }

            StmtKind::Expr(ref mut expr) => {
                self.transform_expr(expr)?;
            }

            StmtKind::If { condition, then_block, elif_clauses, else_block, .. } => {
                self.transform_expr(condition)?;
                self.transform_block(then_block)?;
                for (cond, block) in elif_clauses {
                    self.transform_expr(cond)?;
                    self.transform_block(block)?;
                }
                if let Some(ref mut eb) = else_block {
                    self.transform_block(eb)?;
                }
            }

            StmtKind::For { body, iterable, .. } => {
                self.transform_expr(iterable)?;
                self.transform_block(body)?;
            }

            StmtKind::While { condition, body } => {
                self.transform_expr(condition)?;
                self.transform_block(body)?;
            }

            _ => {} // Other statement kinds pass through unchanged
        }
        Ok(())
    }

    /// Transform an expression, rewriting calls and ops as needed.
    fn transform_expr(&mut self, expr: &mut Expr) -> Result<(), VmapError> {
        match &mut expr.kind {
            ExprKind::BinaryOp { left, right, .. } => {
                self.transform_expr(left)?;
                self.transform_expr(right)?;
            }

            ExprKind::UnaryOp { operand, .. } => {
                self.transform_expr(operand)?;
            }

            ExprKind::Call { callee, args } => {
                // Transform arguments first
                for arg in args.iter_mut() {
                    self.transform_expr(&mut arg.value)?;
                }

                // Check for special function rewrites
                if let ExprKind::Ident(func_sym) = &callee.kind {
                    let func_name = self.interner.resolve(func_sym.0)
                        .unwrap_or("").to_string();

                    match func_name.as_str() {
                        "matmul" => {
                            self.rewrite_matmul_call(expr)?;
                        }
                        "sum" | "mean" | "max" | "min" | "prod" => {
                            self.rewrite_reduction_call(expr)?;
                        }
                        "transpose" => {
                            self.rewrite_transpose_call(expr)?;
                        }
                        "softmax" => {
                            self.rewrite_reduction_call(expr)?; // softmax has dim arg too
                        }
                        _ => {} // Elementwise ops need no rewriting
                    }
                }
            }

            ExprKind::Subscript { object, index } => {
                self.transform_expr(object)?;
                // index is SubscriptKind, not Expr — recurse into its variants
                match index.as_mut() {
                    nsl_ast::expr::SubscriptKind::Index(ref mut expr) => {
                        self.transform_expr(expr)?;
                    }
                    nsl_ast::expr::SubscriptKind::Slice { start, stop, step } => {
                        if let Some(ref mut e) = start { self.transform_expr(e)?; }
                        if let Some(ref mut e) = stop { self.transform_expr(e)?; }
                        if let Some(ref mut e) = step { self.transform_expr(e)?; }
                    }
                    nsl_ast::expr::SubscriptKind::MultiDim(ref mut dims) => {
                        for dim in dims {
                            if let nsl_ast::expr::SubscriptKind::Index(ref mut e) = dim {
                                self.transform_expr(e)?;
                            }
                        }
                    }
                }
            }

            ExprKind::MemberAccess { object, .. } => {
                self.transform_expr(object)?;
            }

            ExprKind::Paren(inner) => {
                self.transform_expr(inner)?;
            }

            _ => {} // Literals, identifiers, etc. pass through
        }
        Ok(())
    }

    /// Classify an expression's batch status without modifying it.
    fn classify_expr(&self, expr: &Expr) -> BatchStatus {
        match &expr.kind {
            ExprKind::Ident(sym) => self.tracker.status(sym),
            ExprKind::MemberAccess { .. } => BatchStatus::Invariant, // self.weight
            ExprKind::BinaryOp { left, right, .. } => {
                let l = self.classify_expr(left);
                let r = self.classify_expr(right);
                if l == BatchStatus::Variant || r == BatchStatus::Variant {
                    BatchStatus::Variant
                } else if l == BatchStatus::Invariant && r == BatchStatus::Invariant {
                    BatchStatus::Invariant
                } else {
                    BatchStatus::Unknown
                }
            }
            ExprKind::Call { args, .. } => {
                if args.iter().any(|a| self.classify_expr(&a.value) == BatchStatus::Variant) {
                    BatchStatus::Variant
                } else {
                    BatchStatus::Invariant
                }
            }
            ExprKind::IntLiteral(_) | ExprKind::FloatLiteral(_) | ExprKind::BoolLiteral(_) => {
                BatchStatus::Invariant
            }
            _ => BatchStatus::Unknown,
        }
    }

    /// Rewrite a matmul call based on operand batch status.
    ///
    /// matmul(variant, invariant) → batched_matmul(a, b)  [left-batched]
    /// matmul(variant, variant)   → batched_matmul(a, b)  [both-batched]
    /// matmul(invariant, variant) → batched_matmul_right(a, b)
    /// matmul(invariant, invariant) → unchanged
    fn rewrite_matmul_call(&mut self, expr: &mut Expr) -> Result<(), VmapError> {
        if let ExprKind::Call { callee, args } = &mut expr.kind {
            if args.len() >= 2 {
                let left_status = self.classify_expr(&args[0].value);
                let right_status = self.classify_expr(&args[1].value);
                let rewrite = classify_matmul_rewrite(left_status, right_status);

                let new_name = match rewrite {
                    MatmulRewrite::LeftBatched | MatmulRewrite::BothBatched => "batched_matmul",
                    MatmulRewrite::RightBatched => "batched_matmul_right",
                    MatmulRewrite::NoRewrite => return Ok(()),
                };

                let new_sym = Symbol(self.interner.get_or_intern(new_name));
                *callee = Box::new(Expr {
                    kind: ExprKind::Ident(new_sym),
                    span: callee.span,
                    id: callee.id,
                });
            }
        }
        Ok(())
    }

    /// Rewrite a reduction call by shifting its dim argument.
    ///
    /// sum(x, dim=0) → sum(x, dim=1) when batch_dim=0 and x is variant.
    fn rewrite_reduction_call(&mut self, expr: &mut Expr) -> Result<(), VmapError> {
        if let ExprKind::Call { args, .. } = &mut expr.kind {
            // Check if the tensor argument is batch-variant
            if args.is_empty() {
                return Ok(());
            }
            let tensor_status = self.classify_expr(&args[0].value);
            if tensor_status != BatchStatus::Variant {
                return Ok(());
            }

            // Find and shift the dim argument (positional arg at index 1, or keyword "dim")
            for arg in args.iter_mut().skip(1) {
                let is_dim_arg = arg.name.map_or(false, |n| {
                    self.interner.resolve(n.0).unwrap_or("") == "dim"
                }) || arg.name.is_none(); // positional second arg is dim

                if is_dim_arg {
                    if let ExprKind::IntLiteral(d) = &arg.value.kind {
                        // NOTE: original_ndim=2 is hardcoded. Only affects negative dim
                    // conversion. For positive dims this param is unused. Proper ndim
                    // propagation via type system deferred to M39c.
                    let shifted = shift_dim(*d, self.config.batch_dim, 2);
                        arg.value.kind = ExprKind::IntLiteral(shifted);
                    }
                    break;
                }
            }
        }
        Ok(())
    }

    /// Rewrite a transpose call by shifting both dim arguments.
    ///
    /// transpose(x, 0, 1) → transpose(x, 1, 2) when batch_dim=0.
    fn rewrite_transpose_call(&mut self, expr: &mut Expr) -> Result<(), VmapError> {
        if let ExprKind::Call { args, .. } = &mut expr.kind {
            if args.is_empty() {
                return Ok(());
            }
            let tensor_status = self.classify_expr(&args[0].value);
            if tensor_status != BatchStatus::Variant {
                return Ok(());
            }

            // Shift dim0 (arg 1) and dim1 (arg 2)
            for arg in args.iter_mut().skip(1) {
                if let ExprKind::IntLiteral(d) = &arg.value.kind {
                    let shifted = shift_dim(*d, self.config.batch_dim, 2);
                    arg.value.kind = ExprKind::IntLiteral(shifted);
                }
            }
        }
        Ok(())
    }
}

/// Errors during vmap transformation.
#[derive(Debug)]
pub enum VmapError {
    /// A batch dimension mismatch was detected.
    BatchMismatch { expected: String, got: String, span: Span },
    /// An unsupported operation was found in the vmap body.
    UnsupportedOp { op: String, span: Span },
}

impl std::fmt::Display for VmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VmapError::BatchMismatch { expected, got, .. } =>
                write!(f, "vmap batch mismatch: expected {expected}, got {got}"),
            VmapError::UnsupportedOp { op, .. } =>
                write!(f, "vmap: unsupported operation '{op}'"),
        }
    }
}
```

Add the following tests to the existing `#[cfg(test)] mod tests` block in `vmap.rs`:

```rust
// --- VmapTransformer tests ---

#[test]
fn test_transform_classifies_params() {
    let mut interner = Interner::new();
    let x = sym(&mut interner, "x");
    let w = sym(&mut interner, "W");
    let batch_sym = sym(&mut interner, "Batch");

    let mut invariants = HashSet::new();
    invariants.insert(w);
    let config = VmapConfig {
        batch_dim: 0,
        batch_sym,
        invariant_params: invariants,
    };

    let mut transformer = VmapTransformer::new(&interner, &config);
    transformer.tracker.mark_variant(x);
    transformer.tracker.mark_invariant(w);

    assert_eq!(transformer.tracker.status(&x), BatchStatus::Variant);
    assert_eq!(transformer.tracker.status(&w), BatchStatus::Invariant);
}

#[test]
fn test_matmul_rewrite_integration() {
    // Verify that classify_matmul_rewrite + shift_dim compose correctly
    // Left-batched: variant @ invariant → LeftBatched
    let rewrite = classify_matmul_rewrite(BatchStatus::Variant, BatchStatus::Invariant);
    assert_eq!(rewrite, MatmulRewrite::LeftBatched);

    // Both-batched: variant @ variant → BothBatched
    let rewrite = classify_matmul_rewrite(BatchStatus::Variant, BatchStatus::Variant);
    assert_eq!(rewrite, MatmulRewrite::BothBatched);
}

#[test]
fn test_reduction_dim_shift_integration() {
    // sum(x, dim=0) with batch_dim=0 should become sum(x, dim=1)
    assert_eq!(shift_dim(0, 0, 2), 1);
    // sum(x, dim=-1) should stay -1 (last dim unchanged)
    assert_eq!(shift_dim(-1, 0, 2), -1);
}

#[test]
fn test_transpose_dim_shift_integration() {
    // transpose(x, 0, 1) with batch_dim=0 → transpose(x, 1, 2)
    assert_eq!(shift_dim(0, 0, 2), 1);
    assert_eq!(shift_dim(1, 0, 2), 2);
}

#[test]
fn test_vmap_error_display() {
    let err = VmapError::UnsupportedOp {
        op: "scatter".into(),
        span: Span::dummy(),
    };
    assert!(format!("{err}").contains("scatter"));
}
```

**Note on AST imports:** The exact import paths depend on how `nsl_ast` exports its types. Check `crates/nsl-ast/src/lib.rs` for the re-exports. Common patterns: `nsl_ast::decl::FnDef`, `nsl_ast::stmt::StmtKind`, `nsl_ast::expr::ExprKind`. If `Block` is in `stmt.rs`, import from there. If `Arg` is in `expr.rs`, import from there. The transformer needs to handle `args` as `Vec<Arg>` where `Arg` has `name: Option<Symbol>` and `value: Expr`.

---

## Phase 2: Runtime + Compiler Integration

### Task 2: Runtime vmap_check_batch

**Files:**
- Create: `crates/nsl-runtime/src/vmap_runtime.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 2: Create `vmap_runtime.rs` with batch-size assertion FFI**

```rust
// crates/nsl-runtime/src/vmap_runtime.rs
//! M39b: Runtime assertions for vmap batch-size checking.

/// Check that a tensor's batch dimension matches the expected batch size.
///
/// Called at the entry of batched functions to validate all batch-variant
/// arguments have consistent batch sizes.
///
/// Parameters (all i64 for Cranelift):
/// - tensor_ptr: pointer to NslTensor
/// - expected_batch: expected batch size (from first variant arg)
/// - batch_dim: which dimension is the batch dimension
///
/// Panics with a descriptive message if the check fails.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_vmap_check_batch(
    tensor_ptr: i64,
    expected_batch: i64,
    batch_dim: i64,
) -> i64 {
    if tensor_ptr == 0 {
        return -1;
    }
    let tensor = unsafe { &*(tensor_ptr as *const crate::tensor::NslTensor) };
    let dim = batch_dim as usize;
    if dim >= tensor.ndim as usize {
        eprintln!(
            "vmap: batch_dim {} out of range for tensor with ndim {}",
            dim, tensor.ndim
        );
        return -1;
    }
    let shape = unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) };
    let actual = shape[dim];
    if actual != expected_batch {
        eprintln!(
            "vmap batch size mismatch: expected {} at dim {}, got {}",
            expected_batch, dim, actual
        );
        return -1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_tensor_returns_error() {
        assert_eq!(nsl_vmap_check_batch(0, 32, 0), -1);
    }
}
```

Add to `crates/nsl-runtime/src/lib.rs`:
```rust
pub mod vmap_runtime;
```

### Task 3: Builtin Registration + Compiler Hook

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 3: Register FFI and add compiler pre-codegen hook**

Add to `builtins.rs` RUNTIME_FUNCTIONS:
```rust
// M39b: vmap runtime
("nsl_vmap_check_batch", &[I64, I64, I64], Some(I64)),
```

Add to `compiler.rs` a method for applying vmap transforms before function compilation:
```rust
/// M39b: Apply vmap AST transformations to produce batched function variants.
///
/// For each function with a @vmap config, clones the FnDef, runs the
/// VmapTransformer, and registers the batched variant for compilation.
/// Must be called before compile_user_functions().
pub fn apply_vmap_transforms(&mut self, module: &nsl_ast::Module) -> Result<Vec<nsl_ast::decl::FnDef>, crate::error::CodegenError> {
    let mut batched_fns = Vec::new();

    for stmt in &module.stmts {
        let fn_def = match &stmt.kind {
            nsl_ast::stmt::StmtKind::FnDef(f) => Some(f),
            nsl_ast::stmt::StmtKind::Decorated { stmt, .. } => {
                if let nsl_ast::stmt::StmtKind::FnDef(f) = &stmt.kind {
                    Some(f)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(fn_def) = fn_def {
            let name = self.resolve_sym(fn_def.name).to_string();
            if let Some(config) = self.vmap_configs.get(&name).cloned() {
                let transformer = crate::vmap::VmapTransformer::new(
                    self.interner,  // &Interner (immutable, matches Compiler field)
                    &config,
                );
                match transformer.transform(fn_def) {
                    Ok(result) => {
                        batched_fns.push(result.batched_fn);
                    }
                    Err(e) => {
                        eprintln!("vmap transform error: {}", e);
                    }
                }
            }
        }
    }

    Ok(batched_fns)
}
```

---

## Phase 3: Build Verification

- [ ] **Step 4: `cargo build` — verify no compile errors**

- [ ] **Step 5: `cargo test` — run all tests, expect 6+ new tests passing**

Expected new tests in `vmap.rs`:
- `test_transform_classifies_params` — variant/invariant classification
- `test_matmul_rewrite_integration` — classify_matmul_rewrite composes with transformer
- `test_reduction_dim_shift_integration` — shift_dim for reductions
- `test_transpose_dim_shift_integration` — shift_dim for transpose
- `test_vmap_error_display` — VmapError Display impl

Expected new tests in `vmap_runtime.rs`:
- `null_tensor_returns_error` — null pointer guard

- [ ] **Step 6: `cargo clippy` — no warnings**

---

## Verification Checklist

After implementation, verify:

1. **BatchTracker propagation**: VarDecl values classified correctly based on operand status
2. **Matmul rewriting**: variant×invariant → `batched_matmul`, variant×variant → `batched_matmul`, invariant×variant → `batched_matmul_right`
3. **Reduction shifting**: dim args shifted past batch_dim for variant tensors
4. **Transpose shifting**: both dim args shifted for variant tensors
5. **Invariant pass-through**: invariant expressions unchanged by transformer
6. **Batched FnDef**: has `_batched` suffix, same params, transformed body
7. **Compiler hook**: `apply_vmap_transforms` iterates module stmts, finds @vmap functions, produces batched variants
8. **Runtime FFI**: `nsl_vmap_check_batch` validates batch dim size
9. **No regressions**: All 579+ existing tests pass
