//! CPDT C-ABI collective wrappers.
//!
//! Every function here is `extern "C"` and accepts tensor handles as
//! `i64`, matching the rest of the NSL runtime's calling convention.
//! Single-GPU behaviour: copy-through (no-op) — the compile-time plan
//! assumes these calls are semantically correct even when the cluster
//! is a singleton.  In multi-GPU builds the stubs are replaced with
//! real NCCL / SHM implementations.
//!
//! The tensor returned by each call is freshly allocated and owned by
//! the caller.

use crate::tensor::arithmetic::{nsl_tensor_add, nsl_tensor_mul_scalar};
use crate::tensor::fbip_flags::RELINQUISH_A;
use crate::tensor::nsl_tensor_free;

// ---------------------------------------------------------------------------
// AllGather
// ---------------------------------------------------------------------------

/// Replicate a sharded tensor across the rank group.
///
/// On a single GPU this is a copy-through: the input shard *is* the
/// full tensor.  Returns a newly-allocated tensor; the caller must
/// free it.
///
/// # Safety
/// `tensor_ptr` must be a valid `*NslTensor` allocated by the NSL
/// runtime, or `0`.
#[no_mangle]
pub extern "C" fn nsl_cpdt_allgather(
    tensor_ptr: i64,
    _group_size: i64,
    _inter_node: i64,
) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    // Scale by 1.0 (no-op multiply producing a fresh tensor): keeps
    // the API's ownership semantics honest (callers free the result,
    // not the input).
    nsl_tensor_mul_scalar(tensor_ptr, 1.0, 0)
}

/// Reduce gradients across the rank group and scatter the result.
///
/// On a single GPU the input is the full gradient → output is the
/// same tensor, possibly divided by group_size (SUM reduction would
/// divide; MEAN would not).  We default to SUM semantics (no divide)
/// because CPDT's FASE integration already handles the averaging.
///
/// # Safety
/// See [`nsl_cpdt_allgather`].
#[no_mangle]
pub extern "C" fn nsl_cpdt_reducescatter(
    tensor_ptr: i64,
    _group_size: i64,
    _inter_node: i64,
) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    nsl_tensor_mul_scalar(tensor_ptr, 1.0, 0)
}

/// All-reduce (reduce followed by broadcast).  Returns a fresh tensor.
///
/// # Safety
/// See [`nsl_cpdt_allgather`].
#[no_mangle]
pub extern "C" fn nsl_cpdt_allreduce(tensor_ptr: i64, _group_size: i64, _inter_node: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    nsl_tensor_mul_scalar(tensor_ptr, 1.0, 0)
}

/// All-to-all dispatch (MoE routing).  Every GPU contributes the same
/// number of bytes and receives the same number back.  On a single
/// GPU this is a straight identity.
///
/// # Safety
/// See [`nsl_cpdt_allgather`].
#[no_mangle]
pub extern "C" fn nsl_cpdt_alltoall(tensor_ptr: i64, _group_size: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    nsl_tensor_mul_scalar(tensor_ptr, 1.0, 0)
}

/// Compose an allgather + local add: the forward path for ZeRO-3 when
/// the gathered params are accumulated into a residual stream.
///
/// # Safety
/// See [`nsl_cpdt_allgather`].
#[no_mangle]
pub extern "C" fn nsl_cpdt_allgather_add(
    shard_ptr: i64,
    residual_ptr: i64,
    group_size: i64,
    inter_node: i64,
) -> i64 {
    if shard_ptr == 0 || residual_ptr == 0 {
        return 0;
    }
    let gathered = nsl_cpdt_allgather(shard_ptr, group_size, inter_node);
    if gathered == 0 {
        return 0;
    }
    // Sum with residual; RELINQUISH_A frees `gathered` inside the add.
    nsl_tensor_add(residual_ptr, gathered, RELINQUISH_A)
}

/// Free a tensor returned by any of the `nsl_cpdt_*` functions.
///
/// # Safety
/// Pointer must be `0` or have been returned by one of these calls.
#[no_mangle]
pub extern "C" fn nsl_cpdt_result_free(tensor_ptr: i64) {
    if tensor_ptr != 0 {
        nsl_tensor_free(tensor_ptr);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_inputs_return_null() {
        assert_eq!(nsl_cpdt_allgather(0, 8, 0), 0);
        assert_eq!(nsl_cpdt_reducescatter(0, 8, 0), 0);
        assert_eq!(nsl_cpdt_allreduce(0, 8, 0), 0);
        assert_eq!(nsl_cpdt_alltoall(0, 8), 0);
        assert_eq!(nsl_cpdt_allgather_add(0, 0, 8, 0), 0);
    }

    #[test]
    fn free_null_is_safe() {
        nsl_cpdt_result_free(0);
    }
}
