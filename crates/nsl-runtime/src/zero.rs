//! M43: ZeRO optimizer state sharding.
//!
//! Implements ZeRO-1/2/3 parameter partitioning, gradient reduction, and
//! optimizer step coordination. Stage 1 partitions optimizer states only:
//! each rank updates only its assigned parameters, then all-gathers the results.

use std::sync::Mutex;

use crate::tensor::NslTensor;

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
// ZeRO FFI implementation
// ---------------------------------------------------------------------------

static ZERO_CTX: Mutex<Option<ZeROContext>> = Mutex::new(None);

#[derive(Clone)]
struct ZeROContext {
    #[allow(dead_code)]
    stage: ZeROStage,
    rank: usize,
    world_size: usize,
    /// Which parameter indices this rank owns (populated by nsl_zero_partition).
    owned_params: Vec<usize>,
    /// Total number of params (set during partition).
    num_params: usize,
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
    // In single-process mode, rank is always 0.
    // Multi-process would read NSL_LOCAL_RANK from env.
    let rank = std::env::var("NSL_LOCAL_RANK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    *guard = Some(ZeROContext {
        stage: s,
        rank,
        world_size: world_size as usize,
        owned_params: Vec::new(),
        num_params: 0,
    });
    0
}

/// Partition parameters for ZeRO. Uses round-robin partitioning.
/// Returns the number of parameters this rank owns, or -1 on error.
#[no_mangle]
pub extern "C" fn nsl_zero_partition(num_params: i64) -> i64 {
    let mut guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_mut() else {
        return -1;
    };

    let partitions = partition_params(num_params as usize, ctx.world_size);
    let my_rank = ctx.rank.min(partitions.len() - 1);
    let my_params = partitions[my_rank].clone();
    let count = my_params.len() as i64;

    ctx.owned_params = my_params;
    ctx.num_params = num_params as usize;

    count
}

/// Check if this rank owns the given parameter index.
/// Returns 1 if owned, 0 if not, -1 if ZeRO not initialized.
#[no_mangle]
pub extern "C" fn nsl_zero_owns_param(param_idx: i64) -> i64 {
    let guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };

    if ctx.owned_params.contains(&(param_idx as usize)) {
        1
    } else {
        0
    }
}

/// Reduce gradients across all DP ranks.
///
/// For Stage 1 (optimizer state sharding), all gradients are all-reduced so
/// every rank has identical gradients. The reduction averages by dividing
/// by world_size after summation.
///
/// In single-process mode (world_size=1), this is a no-op since gradients
/// are already complete.
///
/// NOTE (M43b): this path is currently DEAD in production. `nsl_zero_init`
/// is registered as a builtin but never emitted by codegen and is absent
/// from the stdlib, so `ZERO_CTX` is always `None` and this function
/// returns -1 at the context guard before touching any tensor. Wiring
/// `nsl_zero_init` emission is M43b-proper work; the body below is kept
/// correct (CPU and GPU, f64/f32) so that wiring lands on a sound runtime.
///
/// `grads_list_ptr`: pointer to an NslList of gradient tensors.
/// `num_params`: number of gradient tensors in the list.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_zero_reduce_grads(grads_list_ptr: i64, num_params: i64) -> i64 {
    let guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };

    // Single-process: gradients are already complete, no communication needed.
    if ctx.world_size <= 1 {
        return 0;
    }

    // Multi-process gradient reduction: iterate over each gradient tensor
    // and average the values. In a real NCCL backend this would call
    // all_reduce_sum then divide by world_size. For the simulated single-process
    // case we just divide by world_size (since the "all-reduce" with 1 rank is identity).
    //
    // With actual multi-process support, this would:
    // 1. For each grad tensor, call backend.all_reduce_sum(grad_data, grad_data, count, dtype, null)
    // 2. Divide each element by world_size
    //
    // For now, with world_size > 1 but single process, we normalize gradients
    // as if all ranks contributed identical gradients (i.e., divide by world_size).
    let list_ptr = grads_list_ptr as *const crate::list::NslList;
    if list_ptr.is_null() {
        return -1;
    }
    let list = unsafe { &*list_ptr };

    let ws = ctx.world_size as f64;
    let inv_ws = 1.0 / ws;
    for i in 0..num_params as usize {
        if i >= list.len as usize {
            break;
        }
        let tensor_raw = unsafe { *list.data.add(i) };
        let tensor_ptr = tensor_raw as *mut NslTensor;
        if tensor_ptr.is_null() {
            continue;
        }
        let tensor = unsafe { &*tensor_ptr };

        // Refuse loudly instead of silently skipping: a skipped tensor would
        // leave one gradient un-averaged and produce silently-wrong training.
        if tensor.device == 0 && tensor.dtype > 1 {
            eprintln!(
                "nsl: nsl_zero_reduce_grads: unsupported gradient dtype {} at \
                 index {}; only f64 (0) and f32 (1) gradients are supported",
                tensor.dtype, i
            );
            return -1;
        }
        if tensor.device > 0 && tensor.dtype != 1 {
            eprintln!(
                "nsl: nsl_zero_reduce_grads: GPU gradient at index {} has \
                 dtype {}; only the canonical GPU f32 (dtype 1) is supported",
                i, tensor.dtype
            );
            return -1;
        }

        if tensor.device == 0 {
            // CPU: device-agnostic in-place scale handles both f64 and f32.
            crate::tensor::nsl_tensor_mul_scalar_inplace(tensor_raw, inv_ws);
            continue;
        }

        // GPU: round-trip through the CPU with an explicit f64 -> f32
        // down-convert on the way back (same staged pattern as
        // nsl_tensor_add_inplace's GPU arm). NOT delegated to
        // nsl_tensor_mul_scalar_inplace: its GPU arm copies
        // `len * element_size()` bytes straight from the round-trip CPU
        // buffer, but `nsl_tensor_to_device(gpu_f32, 0)` upcasts to CPU f64,
        // so that copy would push raw f64 bytes into the f32 device buffer.
        #[cfg(feature = "cuda")]
        {
            let cpu_ptr = crate::tensor::nsl_tensor_to_device(tensor_raw, 0);
            let cpu = unsafe { &*(cpu_ptr as *const NslTensor) };
            let len = cpu.len as usize;
            let mut staged = vec![0.0f32; len];
            if cpu.dtype == 1 {
                let d = cpu.data as *const f32;
                for (j, slot) in staged.iter_mut().enumerate() {
                    *slot = unsafe { *d.add(j) } * inv_ws as f32;
                }
            } else {
                let d = cpu.data as *const f64;
                for (j, slot) in staged.iter_mut().enumerate() {
                    *slot = (unsafe { *d.add(j) } * inv_ws) as f32;
                }
            }
            crate::cuda::inner::memcpy_htod(
                tensor.data,
                staged.as_ptr() as *const std::ffi::c_void,
                len * std::mem::size_of::<f32>(),
            );
            crate::tensor::nsl_tensor_free(cpu_ptr);
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!(
                "nsl: nsl_zero_reduce_grads: GPU gradient at index {} requires \
                 the cuda feature",
                i
            );
            return -1;
        }
    }

    0
}

/// ZeRO optimizer step: after each rank updates only its owned parameters,
/// all-gather the updated parameters so every rank has the full model.
///
/// In single-process mode (world_size=1), this is a no-op since all params
/// are updated locally.
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_zero_step() -> i64 {
    let guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };

    // Single-process: all params updated locally, no communication needed.
    if ctx.world_size <= 1 {
        return 0;
    }

    // Multi-process: each rank would broadcast its updated param slices
    // via all-gather. With actual NCCL backend:
    // 1. For each rank r, broadcast params[owned_by_r] to all ranks
    // 2. Each rank copies received params into its local model
    //
    // With simulated single-process, this is a no-op since we update all
    // owned params locally and there are no other ranks to sync with.
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
// Gradient accumulation FFI
// ---------------------------------------------------------------------------

/// Accumulate gradients: dst += src element-wise.
/// Both `dst_ptr` and `src_ptr` are pointers to NslTensor.
/// `num_elems` is the number of elements to accumulate (must match tensor lengths).
///
/// Matched CPU f64/f64 and f32/f32 pairs take the original inline loops
/// (byte-identical fast path). CPU dtype mismatches convert element-wise
/// with proper casts. Any pair with a GPU-resident side — including the
/// live single-GPU FullBuffer path (GPU accum buffer += tape gradient,
/// reached by `grad_accumulation > 1` with lion/muon/soap) — migrates
/// `src` to `dst`'s device via `nsl_tensor_to_device_like` (owned ref,
/// canonical dtype: CPU=f64, GPU=f32) and delegates to the GPU-safe
/// `nsl_tensor_add_inplace` round-trip.
///
/// Non-f32/f64 dtypes refuse loudly with -1: the previous fallback promoted
/// both pointers to f64 blindly, which read out of bounds whenever `src`
/// was narrower than 8 bytes per element.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_grad_accumulate_add(
    dst_ptr: i64,
    src_ptr: i64,
    num_elems: i64,
) -> i64 {
    let dst_tensor = dst_ptr as *mut NslTensor;
    let src_tensor = src_ptr as *const NslTensor;
    if dst_tensor.is_null() || src_tensor.is_null() {
        return -1;
    }

    let dst = unsafe { &*dst_tensor };
    let src = unsafe { &*src_tensor };
    let n = num_elems as usize;

    if dst.dtype > 1 || src.dtype > 1 {
        eprintln!(
            "nsl: nsl_grad_accumulate_add: unsupported gradient dtype pair \
             (dst={}, src={}); only f64 (0) and f32 (1) are supported",
            dst.dtype, src.dtype
        );
        return -1;
    }

    if dst.device == 0 && src.device == 0 {
        match (dst.dtype, src.dtype) {
            (0, 0) => {
                // f64 += f64
                let d = dst.data as *mut f64;
                let s = src.data as *const f64;
                for i in 0..n {
                    unsafe { *d.add(i) += *s.add(i); }
                }
            }
            (1, 1) => {
                // f32 += f32
                let d = dst.data as *mut f32;
                let s = src.data as *const f32;
                for i in 0..n {
                    unsafe { *d.add(i) += *s.add(i); }
                }
            }
            (0, 1) => {
                // f64 += f32: element-wise upcast.
                let d = dst.data as *mut f64;
                let s = src.data as *const f32;
                for i in 0..n {
                    unsafe { *d.add(i) += *s.add(i) as f64; }
                }
            }
            (1, 0) => {
                // f32 += f64: element-wise downcast.
                let d = dst.data as *mut f32;
                let s = src.data as *const f64;
                for i in 0..n {
                    unsafe { *d.add(i) += *s.add(i) as f32; }
                }
            }
            _ => unreachable!("dtypes > 1 rejected above"),
        }
        return 0;
    }

    // At least one side is GPU-resident. The delegated nsl_tensor_add_inplace
    // always adds dst.len elements, so a partial `num_elems < dst.len` request
    // (honored by the CPU arms above) cannot be expressed here — refuse it
    // rather than silently over-adding.
    if n != dst.len as usize {
        eprintln!(
            "nsl: nsl_grad_accumulate_add: partial accumulation \
             (num_elems={} != dst len={}) is not supported on the GPU path; \
             refusing",
            n, dst.len
        );
        return -1;
    }

    // Migrate src to dst's device —
    // nsl_tensor_to_device_like returns an OWNED ref (refcount++ when the
    // devices already match, a fresh tensor otherwise) that must be freed
    // exactly once below (FBIP ownership, see stmt_fase.rs accumulate arm).
    // Migration canonicalizes dtype (CPU=f64, GPU=f32), so the delegated
    // nsl_tensor_add_inplace sees matching same-device operands and its
    // GPU arm performs the proven CPU round-trip with f64 -> f32 staging.
    let migrated = crate::tensor::nsl_tensor_to_device_like(src_ptr, dst_ptr);
    if migrated == 0 {
        return -1;
    }
    let mig = unsafe { &*(migrated as *const NslTensor) };
    if mig.dtype != dst.dtype || mig.len != dst.len {
        eprintln!(
            "nsl: nsl_grad_accumulate_add: post-migration mismatch \
             (dst dtype={} len={}, migrated src dtype={} len={}); refusing",
            dst.dtype, dst.len, mig.dtype, mig.len
        );
        crate::tensor::nsl_tensor_free(migrated);
        return -1;
    }
    crate::tensor::nsl_tensor_add_inplace(dst_ptr, migrated);
    crate::tensor::nsl_tensor_free(migrated);

    0
}

/// Zero out gradient buffer. `grad_ptr` is a pointer to NslTensor.
/// `num_elems` is the number of elements to zero.
///
/// GPU buffers (accumulation buffers are `zeros_like(param)`, so they live
/// on the param's device) are zeroed with a device memset, mirroring
/// `nsl_tensor_zero_inplace`. Zeroing is dtype-agnostic byte clearing.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_grad_zero(grad_ptr: i64, num_elems: i64) -> i64 {
    let tensor_ptr = grad_ptr as *mut NslTensor;
    if tensor_ptr.is_null() {
        return -1;
    }

    let tensor = unsafe { &*tensor_ptr };
    let n = num_elems as usize;

    let byte_width = match tensor.dtype {
        0 => 8usize, // f64
        1 => 4,      // f32
        2 | 3 => 2,  // f16/bf16
        4..=6 => 1, // i8/fp8
        _ => 8,      // default to f64 width
    };
    let total_bytes = n * byte_width;

    if tensor.device != 0 {
        // GPU grad buffers are always f32 (`zeros_like(param)` on GPU). The
        // byte_width table above reflects CPU storage widths — e.g. dtype 4
        // is 1 byte on CPU but allocated 4 bytes/elem on GPU by
        // nsl_tensor_to_device — so any non-f32 GPU buffer would be silently
        // under-zeroed. Refuse instead (mirrors nsl_zero_reduce_grads).
        if tensor.dtype != 1 {
            eprintln!(
                "nsl: nsl_grad_zero: GPU grad buffer must be f32 (dtype 1), \
                 got dtype {}; refusing",
                tensor.dtype
            );
            return -1;
        }
        #[cfg(feature = "cuda")]
        {
            crate::cuda::inner::memset_d8(tensor.data, total_bytes);
            return 0;
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("nsl: nsl_grad_zero: GPU tensor requires the cuda feature");
            return -1;
        }
    }

    unsafe {
        std::ptr::write_bytes(tensor.data as *mut u8, 0, total_bytes);
    }

    0
}

/// All-reduce gradients across all DP ranks (legacy API, wraps nsl_zero_reduce_grads).
/// `grad_ptr` is a single gradient tensor pointer, `num_elems` is element count.
///
/// NOTE: never-emitted scaffolding — no codegen path or stdlib function
/// references this symbol today. It intentionally performs no arithmetic:
/// a single-process all-reduce is the identity, and real multi-process
/// NCCL collectives are M43b work (see nsl_zero_reduce_grads).
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_grad_all_reduce(_grad_ptr: i64, _num_elems: i64) -> i64 {
    // Single-process: no-op (gradient is already the full gradient).
    // Multi-process would call all_reduce_sum on this single tensor.
    let guard = ZERO_CTX.lock().unwrap();
    if let Some(ctx) = guard.as_ref() {
        if ctx.world_size <= 1 {
            return 0;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    // Tests that touch ZERO_CTX race on the shared global when run in
    // parallel.  Serialize them with a module-local guard (same pattern as
    // inspect_ffi.rs / health_ffi.rs) so `cargo test` is safe without
    // requiring --test-threads=1 on the command line.
    fn zero_serial_lock() -> MutexGuard<'static, ()> {
        static SERIAL: OnceLock<Mutex<()>> = OnceLock::new();
        let m = SERIAL.get_or_init(|| Mutex::new(()));
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

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
        let _serial = zero_serial_lock();
        // Ensure clean state
        {
            let mut guard = ZERO_CTX.lock().unwrap();
            *guard = None;
        }

        // Init with valid stage
        assert_eq!(nsl_zero_init(1, 4), 0);
        // Double init fails
        assert_eq!(nsl_zero_init(1, 2), -1);

        // Partition 10 params across 4 ranks (rank 0 in single-process)
        let owned = nsl_zero_partition(10);
        // Rank 0 owns params 0, 4, 8 = 3 params
        assert_eq!(owned, 3);

        // Check ownership
        assert_eq!(nsl_zero_owns_param(0), 1);  // owned
        assert_eq!(nsl_zero_owns_param(4), 1);  // owned
        assert_eq!(nsl_zero_owns_param(8), 1);  // owned
        assert_eq!(nsl_zero_owns_param(1), 0);  // not owned
        assert_eq!(nsl_zero_owns_param(2), 0);  // not owned

        // Reduce and step are no-ops for world_size=1 processes
        // (even though world_size=4, we're single-process so reduce_grads
        // divides by world_size for averaging)
        assert_eq!(nsl_zero_step(), 0);

        // Destroy succeeds
        assert_eq!(nsl_zero_destroy(), 0);
        // Double destroy fails
        assert_eq!(nsl_zero_destroy(), -1);

        // Invalid stage
        assert_eq!(nsl_zero_init(5, 2), -1);
    }

    #[test]
    fn test_zero_owns_param_not_initialized() {
        let _serial = zero_serial_lock();
        {
            let mut guard = ZERO_CTX.lock().unwrap();
            *guard = None;
        }
        assert_eq!(nsl_zero_owns_param(0), -1);
    }

    #[test]
    fn test_grad_zero() {
        // Create a small f64 tensor manually
        let mut data = vec![1.0f64, 2.0, 3.0, 4.0];
        let shape = vec![4i64];
        let strides = vec![1i64];

        let tensor = NslTensor {
            magic: crate::tensor::TENSOR_MAGIC,
            data: data.as_mut_ptr() as *mut c_void,
            shape: shape.as_ptr() as *mut i64,
            strides: strides.as_ptr() as *mut i64,
            ndim: 1,
            len: 4,
            refcount: std::sync::atomic::AtomicI64::new(1),
            device: 0,
            dtype: 0, // f64
            owns_data: 0, // borrowed
            data_owner: 0,
            slab_managed: 0,
            tape_id: 0,
        };

        let tensor_ptr = &tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_zero(tensor_ptr, 4), 0);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_grad_accumulate_add() {
        let mut dst_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let src_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let shape = vec![4i64];
        let strides = vec![1i64];

        let dst_tensor = NslTensor {
            magic: crate::tensor::TENSOR_MAGIC,
            data: dst_data.as_mut_ptr() as *mut c_void,
            shape: shape.as_ptr() as *mut i64,
            strides: strides.as_ptr() as *mut i64,
            ndim: 1,
            len: 4,
            refcount: std::sync::atomic::AtomicI64::new(1),
            device: 0,
            dtype: 0,
            owns_data: 0,
            data_owner: 0,
            slab_managed: 0,
            tape_id: 0,
        };

        let src_tensor = NslTensor {
            magic: crate::tensor::TENSOR_MAGIC,
            data: src_data.as_ptr() as *mut c_void,
            shape: shape.as_ptr() as *mut i64,
            strides: strides.as_ptr() as *mut i64,
            ndim: 1,
            len: 4,
            refcount: std::sync::atomic::AtomicI64::new(1),
            device: 0,
            dtype: 0,
            owns_data: 0,
            data_owner: 0,
            slab_managed: 0,
            tape_id: 0,
        };

        let dst_ptr = &dst_tensor as *const NslTensor as i64;
        let src_ptr = &src_tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_accumulate_add(dst_ptr, src_ptr, 4), 0);
        assert_eq!(dst_data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// Build a borrowed (owns_data=0) CPU test tensor over caller-owned
    /// buffers. Used by the mixed-dtype accumulate tests below.
    fn borrowed_cpu_tensor(
        data: *mut c_void,
        shape: *mut i64,
        strides: *mut i64,
        len: i64,
        dtype: u16,
    ) -> NslTensor {
        NslTensor {
            magic: crate::tensor::TENSOR_MAGIC,
            data,
            shape,
            strides,
            ndim: 1,
            len,
            refcount: std::sync::atomic::AtomicI64::new(1),
            device: 0,
            dtype,
            owns_data: 0,
            data_owner: 0,
            slab_managed: 0,
            tape_id: 0,
        }
    }

    /// f64 dst += f32 src. The pre-fix code promoted BOTH pointers to f64,
    /// reading 8 bytes per element out of the 4-byte-per-element f32 src —
    /// an out-of-bounds read producing garbage sums. Now converts properly.
    #[test]
    fn test_grad_accumulate_add_f64_dst_f32_src_converts() {
        let mut dst_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut src_data = vec![0.5f32, 1.5, 2.5, 3.5];
        let mut shape = vec![4i64];
        let mut strides = vec![1i64];

        let dst_tensor = borrowed_cpu_tensor(
            dst_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            0,
        );
        let src_tensor = borrowed_cpu_tensor(
            src_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            1,
        );

        let dst_ptr = &dst_tensor as *const NslTensor as i64;
        let src_ptr = &src_tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_accumulate_add(dst_ptr, src_ptr, 4), 0);
        assert_eq!(dst_data, vec![1.5, 3.5, 5.5, 7.5]);
        // src untouched
        assert_eq!(src_data, vec![0.5, 1.5, 2.5, 3.5]);
    }

    /// f32 dst += f64 src (tape-AD gradients are CPU f64; an f32 accum
    /// buffer must receive a downcast add, not a reinterpretation).
    #[test]
    fn test_grad_accumulate_add_f32_dst_f64_src_converts() {
        let mut dst_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut src_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let mut shape = vec![4i64];
        let mut strides = vec![1i64];

        let dst_tensor = borrowed_cpu_tensor(
            dst_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            1,
        );
        let src_tensor = borrowed_cpu_tensor(
            src_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            0,
        );

        let dst_ptr = &dst_tensor as *const NslTensor as i64;
        let src_ptr = &src_tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_accumulate_add(dst_ptr, src_ptr, 4), 0);
        assert_eq!(dst_data, vec![11.0f32, 22.0, 33.0, 44.0]);
    }

    /// Non-f32/f64 dtypes must refuse (-1) and leave the buffers untouched
    /// instead of taking the old unsound promote-to-f64 arm.
    #[test]
    fn test_grad_accumulate_add_unsupported_dtype_refuses() {
        let mut dst_data = vec![0x3C00u16, 0x4000, 0x4200, 0x4400]; // f16 bits
        let mut src_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let mut shape = vec![4i64];
        let mut strides = vec![1i64];

        let dst_tensor = borrowed_cpu_tensor(
            dst_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            2, // f16
        );
        let src_tensor = borrowed_cpu_tensor(
            src_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            0,
        );

        let dst_ptr = &dst_tensor as *const NslTensor as i64;
        let src_ptr = &src_tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_accumulate_add(dst_ptr, src_ptr, 4), -1);
        assert_eq!(dst_data, vec![0x3C00, 0x4000, 0x4200, 0x4400]);
    }

    #[test]
    fn test_grad_accumulate_add_null_refuses() {
        assert_eq!(nsl_grad_accumulate_add(0, 0, 4), -1);
        assert_eq!(nsl_grad_zero(0, 4), -1);
    }

    #[test]
    fn test_grad_zero_f32() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut shape = vec![4i64];
        let mut strides = vec![1i64];

        let tensor = borrowed_cpu_tensor(
            data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            1,
        );

        let tensor_ptr = &tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_zero(tensor_ptr, 4), 0);
        assert_eq!(data, vec![0.0f32, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_partition_single_rank() {
        let _serial = zero_serial_lock();
        // Ensure clean state
        {
            let mut guard = ZERO_CTX.lock().unwrap();
            *guard = None;
        }

        assert_eq!(nsl_zero_init(1, 1), 0);
        let owned = nsl_zero_partition(5);
        assert_eq!(owned, 5); // Single rank owns all params

        // All params owned
        for i in 0..5 {
            assert_eq!(nsl_zero_owns_param(i), 1);
        }

        // reduce_grads and step are no-ops for single rank
        assert_eq!(nsl_zero_step(), 0);

        assert_eq!(nsl_zero_destroy(), 0);
    }
}
