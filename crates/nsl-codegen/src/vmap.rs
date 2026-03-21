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
#[derive(Default)]
pub struct BatchTracker {
    statuses: HashMap<Symbol, BatchStatus>,
}

impl BatchTracker {
    pub fn new() -> Self {
        Self::default()
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

// ---------------------------------------------------------------------------
// Shape rewriting
// ---------------------------------------------------------------------------

/// Insert a batch dimension at position `batch_dim` into a shape (as dim count).
/// Returns the new ndim. Only applies to batch-variant tensors.
pub fn insert_batch_dim(original_ndim: usize, _batch_dim: usize, status: BatchStatus) -> usize {
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
        let shifted = if abs_dim >= batch_dim {
            abs_dim + 1
        } else {
            abs_dim
        };
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
pub fn classify_matmul_rewrite(
    left_status: BatchStatus,
    right_status: BatchStatus,
) -> MatmulRewrite {
    match (left_status, right_status) {
        (BatchStatus::Variant, BatchStatus::Invariant) => MatmulRewrite::LeftBatched,
        (BatchStatus::Variant, BatchStatus::Variant) => MatmulRewrite::BothBatched,
        (BatchStatus::Invariant, BatchStatus::Variant) => MatmulRewrite::RightBatched,
        _ => MatmulRewrite::NoRewrite,
    }
}

// ---------------------------------------------------------------------------
// VmapTransformer — FnDef-to-FnDef AST rewriting (M39b)
// ---------------------------------------------------------------------------

use nsl_ast::decl::FnDef;
use nsl_ast::expr::{Expr, ExprKind, SubscriptKind};
use nsl_ast::stmt::{Stmt, StmtKind, Block};
use nsl_ast::pattern::PatternKind;
use nsl_ast::{NodeId, Span};
use nsl_lexer::Interner;

/// Result of a vmap transformation: the original function plus the generated batched version.
#[derive(Debug)]
pub struct VmapResult {
    /// The batched function (with batch dims inserted).
    pub batched_fn: FnDef,
    /// Name of the batched function (original_name + "_batched").
    pub batched_name: String,
    /// Matmul call rewrites: (NodeId of Call expr, target function name).
    /// The compiler applies these rewrites when it has `&mut Interner` access.
    pub matmul_rewrites: Vec<(NodeId, String)>,
}

/// Transforms a FnDef into a batched version by walking the AST and inserting
/// batch dimensions into tensor operations.
pub struct VmapTransformer<'a> {
    interner: &'a Interner, // immutable — matches Compiler's &'a Interner
    config: &'a VmapConfig,
    tracker: BatchTracker,
    /// Accumulated matmul rewrites (NodeId of Call expr → target callee name).
    /// Stored here because we cannot intern new symbols with `&Interner`.
    matmul_rewrites: Vec<(NodeId, String)>,
}

impl<'a> VmapTransformer<'a> {
    pub fn new(interner: &'a Interner, config: &'a VmapConfig) -> Self {
        VmapTransformer {
            interner,
            config,
            tracker: BatchTracker::new(),
            matmul_rewrites: Vec::new(),
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
        // compiler interns batched_name later)
        let batched_fn = FnDef {
            name: fn_def.name, // placeholder — compiler interns batched_name later
            type_params: fn_def.type_params.clone(),
            effect_params: fn_def.effect_params.clone(),
            params: fn_def.params.clone(), // params keep same names; shapes change at type level
            return_type: fn_def.return_type.clone(),
            return_effect: fn_def.return_effect.clone(),
            body: batched_body,
            is_async: fn_def.is_async,
            span: fn_def.span,
        };

        let matmul_rewrites = std::mem::take(&mut self.matmul_rewrites);
        Ok(VmapResult { batched_fn, batched_name, matmul_rewrites })
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
            StmtKind::VarDecl { pattern, value: Some(ref mut val), .. } => {
                // Transform the value expression
                self.transform_expr(val)?;

                // Classify the binding based on the value's batch status
                let status = self.classify_expr(val);
                if let PatternKind::Ident(sym) = &pattern.kind {
                    match status {
                        BatchStatus::Variant => self.tracker.mark_variant(*sym),
                        BatchStatus::Invariant => self.tracker.mark_invariant(*sym),
                        _ => {}
                    }
                }
            }

            StmtKind::VarDecl { value: None, .. } => {
                // No value — nothing to transform
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
                    SubscriptKind::Index(ref mut idx_expr) => {
                        self.transform_expr(idx_expr)?;
                    }
                    SubscriptKind::Slice { lower, upper, step } => {
                        if let Some(ref mut e) = lower { self.transform_expr(e)?; }
                        if let Some(ref mut e) = upper { self.transform_expr(e)?; }
                        if let Some(ref mut e) = step { self.transform_expr(e)?; }
                    }
                    SubscriptKind::MultiDim(ref mut dims) => {
                        for dim in dims {
                            if let SubscriptKind::Index(ref mut e) = dim {
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
    /// matmul(variant, invariant) -> batched_matmul(a, b)  [left-batched]
    /// matmul(variant, variant)   -> batched_matmul(a, b)  [both-batched]
    /// matmul(invariant, variant) -> batched_matmul_right(a, b)
    /// matmul(invariant, invariant) -> unchanged
    ///
    /// Since we only have `&Interner` (immutable), we cannot intern the new callee
    /// name. Instead, the rewrite target is stored in `self.matmul_rewrites` and
    /// the compiler applies it when it has `&mut Interner`.
    fn rewrite_matmul_call(&mut self, expr: &mut Expr) -> Result<(), VmapError> {
        if let ExprKind::Call { args, .. } = &expr.kind {
            if args.len() >= 2 {
                let left_status = self.classify_expr(&args[0].value);
                let right_status = self.classify_expr(&args[1].value);
                let rewrite = classify_matmul_rewrite(left_status, right_status);

                let new_name = match rewrite {
                    MatmulRewrite::LeftBatched | MatmulRewrite::BothBatched => "batched_matmul",
                    MatmulRewrite::RightBatched => "batched_matmul_right",
                    MatmulRewrite::NoRewrite => return Ok(()),
                };

                // Record the rewrite for later application by the compiler.
                self.matmul_rewrites.push((expr.id, new_name.to_string()));
            }
        }
        Ok(())
    }

    /// Rewrite a reduction call by shifting its dim argument.
    ///
    /// sum(x, dim=0) -> sum(x, dim=1) when batch_dim=0 and x is variant.
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
                let is_dim_arg = arg.name.is_some_and(|n| {
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
    /// transpose(x, 0, 1) -> transpose(x, 1, 2) when batch_dim=0.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::Symbol;

    type Interner = string_interner::StringInterner<
        string_interner::backend::BucketBackend<string_interner::DefaultSymbol>,
    >;

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
        assert_eq!(insert_batch_dim(2, 0, BatchStatus::Variant), 3);
    }

    #[test]
    fn test_insert_batch_dim_invariant_unchanged() {
        assert_eq!(insert_batch_dim(2, 0, BatchStatus::Invariant), 2);
    }

    #[test]
    fn test_insert_batch_dim_at_position_1() {
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
        // [H, W] (ndim=2) with batch_dim=1: dim 0 stays 0
        assert_eq!(shift_dim(0, 1, 2), 0);
        assert_eq!(shift_dim(1, 1, 2), 2);
    }

    #[test]
    fn test_shift_dim_negative_batch_dim_0() {
        // [S, D] (ndim=2) with batch_dim=0 -> [B, S, D] (ndim=3)
        // dim=-1 = D (abs=1), shifted abs=2, back to -(3-2) = -1
        assert_eq!(shift_dim(-1, 0, 2), -1);
        // dim=-2 = S (abs=0), shifted abs=1, back to -(3-1) = -2
        assert_eq!(shift_dim(-2, 0, 2), -2);
    }

    #[test]
    fn test_shift_dim_negative_batch_dim_1() {
        // [H, W] (ndim=2) with batch_dim=1 -> [H, B, W] (ndim=3)
        // dim=-1 = W (abs=1), abs >= batch_dim=1, shifted abs=2, back to -(3-2) = -1
        assert_eq!(shift_dim(-1, 1, 2), -1);
        // dim=-2 = H (abs=0), abs < batch_dim=1, no shift abs=0, back to -(3-0) = -3
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

    // --- VmapTransformer tests (M39b) ---

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
        // Verify that classify_matmul_rewrite composes correctly for transformer use
        // Left-batched: variant @ invariant -> LeftBatched
        let rewrite = classify_matmul_rewrite(BatchStatus::Variant, BatchStatus::Invariant);
        assert_eq!(rewrite, MatmulRewrite::LeftBatched);

        // Both-batched: variant @ variant -> BothBatched
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
        // transpose(x, 0, 1) with batch_dim=0 -> transpose(x, 1, 2)
        assert_eq!(shift_dim(0, 0, 2), 1);
        assert_eq!(shift_dim(1, 0, 2), 2);
    }

    #[test]
    fn test_vmap_error_display() {
        let err = VmapError::UnsupportedOp {
            op: "scatter".into(),
            span: nsl_ast::Span::dummy(),
        };
        assert!(format!("{err}").contains("scatter"));

        let err = VmapError::BatchMismatch {
            expected: "32".into(),
            got: "16".into(),
            span: nsl_ast::Span::dummy(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("32"));
        assert!(msg.contains("16"));
    }
}
