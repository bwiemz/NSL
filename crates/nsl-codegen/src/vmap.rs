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
}
