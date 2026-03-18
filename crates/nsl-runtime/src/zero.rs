//! M43: ZeRO optimizer state sharding.
//!
//! Implements ZeRO-1/2/3 parameter partitioning and gradient reduction FFI stubs.
//! Real NCCL-backed implementations deferred to M43b; these stubs allow the
//! compiler and codegen to wire up the full pipeline parallelism API.

use std::sync::Mutex;

/// ZeRO optimization stage.
/// - Stage1: partition optimizer states only
/// - Stage2: partition optimizer states + gradients
/// - Stage3: partition optimizer states + gradients + parameters
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZeROStage {
    Stage1,
    Stage2,
    Stage3,
}

impl ZeROStage {
    pub fn from_i64(v: i64) -> Option<Self> {
        match v {
            1 => Some(Self::Stage1),
            2 => Some(Self::Stage2),
            3 => Some(Self::Stage3),
            _ => None,
        }
    }
}

/// Partition parameters across data-parallel ranks using round-robin.
/// Returns a vec of vec where `partitions[rank]` contains the parameter indices
/// assigned to that rank.
pub fn partition_params(num_params: usize, world_size: usize) -> Vec<Vec<usize>> {
    let mut partitions = vec![Vec::new(); world_size];
    for i in 0..num_params {
        partitions[i % world_size].push(i);
    }
    partitions
}

// ---------------------------------------------------------------------------
// ZeRO FFI stubs
// ---------------------------------------------------------------------------

static ZERO_CTX: Mutex<Option<ZeROContext>> = Mutex::new(None);

struct ZeROContext {
    _stage: ZeROStage,
    _world_size: usize,
}

/// Initialize ZeRO optimizer sharding.
/// `stage`: 1/2/3, `world_size`: number of data-parallel ranks.
/// Returns 0 on success, -1 if already initialized or invalid stage.
#[no_mangle]
pub extern "C" fn nsl_zero_init(stage: i64, world_size: i64) -> i64 {
    let mut guard = ZERO_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    let Some(s) = ZeROStage::from_i64(stage) else {
        return -1;
    };
    *guard = Some(ZeROContext {
        _stage: s,
        _world_size: world_size as usize,
    });
    0
}

/// Partition parameters for ZeRO. Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_zero_partition(_num_params: i64) -> i64 {
    0
}

/// Reduce-scatter gradients across DP ranks. Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_zero_reduce_grads(_grad_ptr: i64, _num_elems: i64) -> i64 {
    0
}

/// ZeRO optimizer step (all-gather params after update). Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_zero_step() -> i64 {
    0
}

/// Destroy ZeRO context. Returns 0 on success, -1 if not initialized.
#[no_mangle]
pub extern "C" fn nsl_zero_destroy() -> i64 {
    let mut guard = ZERO_CTX.lock().unwrap();
    if guard.is_none() {
        return -1;
    }
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// Gradient accumulation FFI stubs
// ---------------------------------------------------------------------------

/// Accumulate gradients: dst += src. Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_grad_accumulate_add(
    _dst_ptr: i64,
    _src_ptr: i64,
    _num_elems: i64,
) -> i64 {
    0
}

/// Zero out gradient buffer. Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_grad_zero(_grad_ptr: i64, _num_elems: i64) -> i64 {
    0
}

/// All-reduce gradients across all DP ranks. Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_grad_all_reduce(_grad_ptr: i64, _num_elems: i64) -> i64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_stage_parsing() {
        assert_eq!(ZeROStage::from_i64(1), Some(ZeROStage::Stage1));
        assert_eq!(ZeROStage::from_i64(2), Some(ZeROStage::Stage2));
        assert_eq!(ZeROStage::from_i64(3), Some(ZeROStage::Stage3));
        assert_eq!(ZeROStage::from_i64(0), None);
        assert_eq!(ZeROStage::from_i64(4), None);
        assert_eq!(ZeROStage::from_i64(-1), None);
    }

    #[test]
    fn test_partition_params_round_robin() {
        let parts = partition_params(10, 3);
        assert_eq!(parts.len(), 3);
        // Rank 0: params 0, 3, 6, 9
        assert_eq!(parts[0], vec![0, 3, 6, 9]);
        // Rank 1: params 1, 4, 7
        assert_eq!(parts[1], vec![1, 4, 7]);
        // Rank 2: params 2, 5, 8
        assert_eq!(parts[2], vec![2, 5, 8]);

        // All params accounted for
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_zero_ffi_lifecycle() {
        // Ensure clean state
        {
            let mut guard = ZERO_CTX.lock().unwrap();
            *guard = None;
        }

        // Init with valid stage
        assert_eq!(nsl_zero_init(2, 4), 0);
        // Double init fails
        assert_eq!(nsl_zero_init(1, 2), -1);
        // Stubs return 0
        assert_eq!(nsl_zero_partition(100), 0);
        assert_eq!(nsl_zero_reduce_grads(0, 0), 0);
        assert_eq!(nsl_zero_step(), 0);
        // Destroy succeeds
        assert_eq!(nsl_zero_destroy(), 0);
        // Double destroy fails
        assert_eq!(nsl_zero_destroy(), -1);

        // Invalid stage
        assert_eq!(nsl_zero_init(5, 2), -1);
    }
}
