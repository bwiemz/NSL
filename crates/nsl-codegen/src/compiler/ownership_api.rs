//! ELTLS accessor API on Compiler.
//!
//! See spec §4.3. All methods filter by Cranelift value type so only I64
//! (pointer-sized) Values pollute the tracking maps.

use cranelift_codegen::ir::{self, types as cl_types};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::ownership_expr::Ownership;

impl<'a> Compiler<'a> {
    /// Read ownership state. Returns Unknown for missing entries — the
    /// strict conservative fallback (spec §4.3).
    pub fn get_ownership(&self, state: &FuncState, val: ir::Value) -> Ownership {
        state
            .cleanup
            .expr_ownership
            .get(&val)
            .copied()
            .unwrap_or(Ownership::Unknown)
    }

    /// Register ownership for a freshly-produced tensor Value.
    /// Filters by cl_types::I64 so non-tensor Values don't pollute the map.
    /// Maintains the owned_temporaries Vec invariant.
    pub fn set_ownership(
        &self,
        builder: &FunctionBuilder,
        state: &mut FuncState,
        val: ir::Value,
        own: Ownership,
    ) {
        if builder.func.dfg.value_type(val) != cl_types::I64 {
            return;
        }
        let prev = state.cleanup.expr_ownership.insert(val, own);
        match (prev, own) {
            (None, Ownership::Owned) => {
                state.cleanup.owned_temporaries.push(val);
            }
            (Some(Ownership::Owned), Ownership::TapeHeld) => {
                state.cleanup.owned_temporaries.retain(|&v| v != val);
                if !state.cleanup.tape_held.contains(&val) {
                    state.cleanup.tape_held.push(val);
                }
            }
            (None, Ownership::TapeHeld) => {
                state.cleanup.tape_held.push(val);
            }
            // Unreachable in practice: TapeHeld is monotonic within a tape
            // region. debug_assert to catch contract violations early.
            (Some(Ownership::TapeHeld), Ownership::Owned) => {
                debug_assert!(
                    false,
                    "ELTLS: illegal transition TapeHeld -> Owned on value {val:?}"
                );
            }
            _ => {}
        }
    }

    /// Register ownership for a Value produced by a tape-recordable op.
    /// Consults classify_backward_access(op_ffi_name) and promotes the
    /// result and operands to TapeHeld when the op is DataRequired and
    /// we're inside a tape region. See spec §4.3, §7.1.
    pub fn set_ownership_from_op(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        result: ir::Value,
        inputs: &[ir::Value],
        intended: Ownership,
        op_ffi_name: &str,
    ) {
        use nsl_semantic::ownership_autodiff::{classify_backward_access, BackwardAccess};
        let class = classify_backward_access(op_ffi_name);
        let tape_sensitive =
            state.flags.in_tape_region && matches!(class, BackwardAccess::DataRequired);

        if tape_sensitive {
            self.promote_to_tape_held(builder, state, result);
            for &inp in inputs {
                self.promote_to_tape_held(builder, state, inp);
            }
        } else {
            self.set_ownership(builder, state, result, intended);
        }
    }

    /// Promote a Value's storage to TapeHeld lifetime. Emits a retain call
    /// for BorrowedFromVar and Unknown prior states because the originating
    /// variable's lexical scope may end before the tape region exits (nested
    /// block trap, spec §7.2).
    pub fn promote_to_tape_held(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        val: ir::Value,
    ) {
        if builder.func.dfg.value_type(val) != cl_types::I64 {
            return;
        }
        let prev = state.cleanup.expr_ownership.get(&val).copied();
        match prev {
            Some(Ownership::BorrowedWeight) => {
                // Weights strictly dominate the tape region — no retain needed.
            }
            Some(Ownership::BorrowedFromVar(_)) => {
                // Variable scope might end before tape exit. Take an
                // independent lease.
                let _ = self.compile_call_by_name(builder, "nsl_tensor_retain", &[val]);
                if !state.cleanup.tape_held.contains(&val) {
                    state.cleanup.tape_held.push(val);
                }
            }
            Some(Ownership::TapeHeld) => {
                // Idempotent.
            }
            Some(Ownership::Owned) => {
                state.cleanup.expr_ownership.insert(val, Ownership::TapeHeld);
                state.cleanup.owned_temporaries.retain(|&v| v != val);
                if !state.cleanup.tape_held.contains(&val) {
                    state.cleanup.tape_held.push(val);
                }
            }
            Some(Ownership::Unknown) | None => {
                let _ = self.compile_call_by_name(builder, "nsl_tensor_retain", &[val]);
                state.cleanup.expr_ownership.insert(val, Ownership::TapeHeld);
                if !state.cleanup.tape_held.contains(&val) {
                    state.cleanup.tape_held.push(val);
                }
                state.cleanup.unknown_ownership_count += 1;
            }
        }
    }

    /// Mark a Value as consumed by an FFI with a relinquish flag set.
    /// Purges from all tracking queues (new and legacy) to prevent
    /// double-free during rollout. Silent no-op for Unknown or missing
    /// values (spec §4.3 consume_ownership notes + §9.1 Commit 5 tests).
    pub fn consume_ownership(&self, state: &mut FuncState, val: ir::Value) {
        debug_assert!(
            !state.cleanup.tape_held.contains(&val),
            "ELTLS: consume_ownership called on TapeHeld value {val:?}"
        );
        state.cleanup.expr_ownership.remove(&val);
        state.cleanup.owned_temporaries.retain(|&v| v != val);
        state.cleanup.tensor_temporaries.retain(|&v| v != val);
    }

    /// Record an Unknown-fallback decision at a consumer site.
    pub fn note_unknown_fallback(&self, state: &mut FuncState, _val: ir::Value) {
        state.cleanup.unknown_ownership_count += 1;
    }
}
