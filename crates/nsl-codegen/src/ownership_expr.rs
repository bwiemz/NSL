//! ELTLS — Expression-Level Tensor Lifetime System.
//!
//! Ownership states tracked in a side-channel HashMap keyed by Cranelift
//! `ir::Value`. See docs/superpowers/specs/2026-04-09-eltls-expression-lifetime-design.md
//! section 4 for the full rationale.

use cranelift_frontend::Variable;

/// Per-Value ownership state for tensor-typed Cranelift Values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ownership {
    /// Freshly-allocated tensor owned by the current scope.
    /// Producer examples: zeros/ones/randn, binary op results, most FFIs.
    /// Consumer may transfer into a variable slot, pass to an FFI with
    /// relinquish flags set, or free at last use.
    Owned,

    /// Alias of a local variable's tensor. The variable still owns the
    /// storage. Consumers that mutate must clone first; consumers that
    /// store must retain.
    BorrowedFromVar(Variable),

    /// Alias of a model weight field. Weight lifetimes are bound to the
    /// model and strictly dominate any training-step scope. Mutation
    /// requires clone; storage does not require retain.
    BorrowedWeight,

    /// Tensor produced inside a tape region by a DataRequired-classified
    /// op, OR an operand of such an op. Storage is kept alive until
    /// `nsl_tape_backward` completes. Freed by `free_tape_held_tensors`
    /// at tape-region exit. Never passed to FFIs with relinquish flags.
    TapeHeld,

    /// Explicit ignorance state. Producer failed to register ownership
    /// OR the value came from a codegen path not yet migrated to ELTLS.
    /// Consumers MUST take the conservative slow path:
    ///
    ///   - want to mutate → emit `nsl_tensor_clone` first
    ///   - want to store → emit `nsl_tensor_retain`
    ///   - at statement cleanup → leave alone; the epilog sweep will
    ///     free it via `nsl_tensor_free_if_valid`
    ///
    /// Every Unknown hit at a consumer increments
    /// `state.cleanup.unknown_ownership_count` for rollout measurement.
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ownership_variants_are_distinct() {
        let var = Variable::from_u32(0);
        assert_ne!(Ownership::Owned, Ownership::BorrowedFromVar(var));
        assert_ne!(Ownership::Owned, Ownership::BorrowedWeight);
        assert_ne!(Ownership::Owned, Ownership::TapeHeld);
        assert_ne!(Ownership::Owned, Ownership::Unknown);
        assert_ne!(Ownership::TapeHeld, Ownership::Unknown);
    }

    #[test]
    fn ownership_is_copy() {
        let own = Ownership::Owned;
        let own2 = own; // Copy
        assert_eq!(own, own2);
    }
}
