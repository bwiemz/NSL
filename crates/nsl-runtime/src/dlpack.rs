//! M62a: DLPack v0.8 bridge for zero-copy tensor exchange.
//!
//! Implements the DLPack C ABI structs and conversion functions between
//! `NslTensor` and `DLManagedTensor`. Both directions are zero-copy — the
//! underlying data pointer is shared, not duplicated.

use std::ffi::c_void;
use std::os::raw::c_int;

use crate::memory::checked_alloc;
use crate::tensor::{NslTensor, DTYPE_F32, DTYPE_F64, DTYPE_FP16, DTYPE_BF16, DTYPE_INT8};

// ---------------------------------------------------------------------------
// DLPack v0.8 C ABI structs
// ---------------------------------------------------------------------------

/// DLPack device type code.
///
/// Deliberately a plain `c_int` (with named constants below), NOT a Rust
/// enum: this field is read out of **foreign** `DLManagedTensor` structs, and
/// a fieldless `repr(C)` enum makes any value outside its variant list
/// instant UB *before* a `match` can refuse it. Real producers emit values we
/// never listed — torch pinned memory is `kDLCUDAHost = 3` — so the type must
/// be able to *hold* every i32 and validation must happen in `match` arms.
pub type DLDeviceType = c_int;

/// CPU device.
pub const KDL_CPU: DLDeviceType = 1;
/// CUDA GPU.
pub const KDL_CUDA: DLDeviceType = 2;
/// CUDA pinned host memory (cudaMallocHost) — CPU-accessible.
pub const KDL_CUDA_HOST: DLDeviceType = 3;
/// Apple Metal.
pub const KDL_METAL: DLDeviceType = 8;
/// AMD ROCm.
pub const KDL_ROCM: DLDeviceType = 10;

/// DLPack device descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: c_int,
}

/// DLPack data type code.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDataTypeCode {
    /// Signed integer.
    KDLInt = 0,
    /// Unsigned integer.
    KDLUInt = 1,
    /// IEEE floating point.
    KDLFloat = 2,
    /// Brain floating point (bfloat16).
    KDLBfloat = 4,
}

/// DLPack data type descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

/// DLPack tensor descriptor (the core zero-copy payload).
#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: c_int,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

/// DLPack managed tensor with deleter callback.
#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

// ---------------------------------------------------------------------------
// NSL dtype <-> DLPack dtype conversion
// ---------------------------------------------------------------------------

/// Convert an NSL dtype code to a DLPack DLDataType.
///
/// Returns `None` for dtypes with no DLPack representation (fp8 variants,
/// the u16 token/segment tags, custom dtypes ≥ 256). This used to be a
/// wildcard that *labeled every unmapped dtype as float64* — a consumer
/// reading an fp8 buffer as f64 gets garbage values and a 8× out-of-bounds
/// read. Refusal over mislabeling.
fn nsl_dtype_to_dl(dtype: u16) -> Option<DLDataType> {
    match dtype {
        DTYPE_F64 => Some(DLDataType { code: DLDataTypeCode::KDLFloat as u8, bits: 64, lanes: 1 }),
        DTYPE_F32 => Some(DLDataType { code: DLDataTypeCode::KDLFloat as u8, bits: 32, lanes: 1 }),
        DTYPE_FP16 => Some(DLDataType { code: DLDataTypeCode::KDLFloat as u8, bits: 16, lanes: 1 }),
        DTYPE_BF16 => Some(DLDataType { code: DLDataTypeCode::KDLBfloat as u8, bits: 16, lanes: 1 }),
        DTYPE_INT8 => Some(DLDataType { code: DLDataTypeCode::KDLInt as u8, bits: 8, lanes: 1 }),
        crate::tensor::DTYPE_I32 => {
            Some(DLDataType { code: DLDataTypeCode::KDLInt as u8, bits: 32, lanes: 1 })
        }
        _ => None,
    }
}

/// Convert a DLPack DLDataType to an NSL dtype code. Returns `None` for unsupported types.
fn dl_dtype_to_nsl(dt: &DLDataType) -> Option<u16> {
    match (dt.code, dt.bits) {
        (c, 64) if c == DLDataTypeCode::KDLFloat as u8 => Some(DTYPE_F64),
        (c, 32) if c == DLDataTypeCode::KDLFloat as u8 => Some(DTYPE_F32),
        (c, 16) if c == DLDataTypeCode::KDLFloat as u8 => Some(DTYPE_FP16),
        (c, 16) if c == DLDataTypeCode::KDLBfloat as u8 => Some(DTYPE_BF16),
        (c, 8) if c == DLDataTypeCode::KDLInt as u8 => Some(DTYPE_INT8),
        (c, 32) if c == DLDataTypeCode::KDLInt as u8 => Some(crate::tensor::DTYPE_I32),
        _ => None,
    }
}

/// Convert an NSL device code to a DLPack DLDevice.
fn nsl_device_to_dl(device: u8) -> DLDevice {
    if device == 0 {
        DLDevice { device_type: KDL_CPU, device_id: 0 }
    } else {
        DLDevice { device_type: KDL_CUDA, device_id: (device - 1) as c_int }
    }
}

/// Convert a DLPack DLDevice to an NSL device code.
///
/// Returns `None` for device types NSL cannot address. This used to map
/// every unknown backend (Metal, ROCm, Vulkan, pinned host memory, ...) to
/// CPU — importing a Metal buffer as a CPU pointer is a wild dereference on
/// first touch. `KDL_CUDA_HOST` (pinned host memory) IS CPU-addressable, so
/// it maps to CPU legitimately.
fn dl_device_to_nsl(dev: &DLDevice) -> Option<u8> {
    match dev.device_type {
        KDL_CPU | KDL_CUDA_HOST => Some(0),
        KDL_CUDA => Some((dev.device_id + 1) as u8),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// NslTensor -> DLManagedTensor (zero-copy export)
// ---------------------------------------------------------------------------

/// Context stored in `manager_ctx` for our exported DLManagedTensors.
/// Holds the original NslTensor pointer so we can manage its lifetime.
struct ExportContext {
    /// Raw pointer to the NslTensor behind `dl_tensor.data`.
    nsl_tensor_ptr: i64,
    /// Ownership mode. `true` = the export TRANSFERRED ownership: the
    /// deleter releases the NslTensor via `nsl_tensor_free` (exactly once).
    /// `false` = borrow export (the historical `nsl_dlpack_export`
    /// semantics): the deleter frees only the wrapper structs and the
    /// NslTensor's lifetime stays managed by NSL.
    owned: bool,
    /// Copies of shape/strides for DLPack (DLPack uses its own arrays).
    shape: Vec<i64>,
    strides: Vec<i64>,
}

/// DLPack deleter callback for tensors exported from NSL.
///
/// # Safety
/// Called by the consumer when they are done with the DLManagedTensor — the
/// DLPack contract is AT MOST ONCE. For owned exports this releases the
/// NslTensor through `nsl_tensor_free`, the only correct release primitive
/// (it is refcount-, owns_data-, data_owner- and slab-aware).
///
/// The deleter field is nulled BEFORE anything is released, so a second
/// `nsl_dlpack_free` on a pointer whose box happens to still be readable
/// degrades to a no-op instead of a double `Box::from_raw`. A second call
/// after the box's memory has genuinely been recycled is undefined behavior
/// under the DLPack contract itself and cannot be guarded here; the Python
/// bridge enforces exactly-once consumption on its side (capsule rename).
unsafe extern "C" fn nsl_dlpack_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    unsafe { (*managed).deleter = None };
    let ctx_ptr = unsafe { (*managed).manager_ctx } as *mut ExportContext;
    unsafe { (*managed).manager_ctx = std::ptr::null_mut() };
    if !ctx_ptr.is_null() {
        let ctx = unsafe { Box::from_raw(ctx_ptr) };
        if ctx.owned && ctx.nsl_tensor_ptr != 0 {
            crate::tensor::nsl_tensor_free(ctx.nsl_tensor_ptr);
        }
        drop(ctx);
    }
    drop(unsafe { Box::from_raw(managed) });
}

/// Shared builder for both export modes. Returns null if the tensor's dtype
/// has no DLPack representation (see `nsl_dtype_to_dl` — refusal over the
/// old mislabel-as-f64 behavior).
fn build_dlpack(tensor: &NslTensor, tensor_ptr: i64, owned: bool) -> *mut DLManagedTensor {
    let dl_dtype = match nsl_dtype_to_dl(tensor.dtype) {
        Some(d) => d,
        None => return std::ptr::null_mut(),
    };
    let ndim = tensor.ndim as usize;

    // Copy shape and strides into owned Vecs for the DLPack struct.
    let shape: Vec<i64> = if ndim > 0 && !tensor.shape.is_null() {
        unsafe { std::slice::from_raw_parts(tensor.shape, ndim).to_vec() }
    } else {
        Vec::new()
    };
    let strides: Vec<i64> = if ndim > 0 && !tensor.strides.is_null() {
        unsafe { std::slice::from_raw_parts(tensor.strides, ndim).to_vec() }
    } else {
        Vec::new()
    };

    let mut ctx = Box::new(ExportContext {
        nsl_tensor_ptr: tensor_ptr,
        owned,
        shape,
        strides,
    });

    // SAFETY (lifetime invariant for DLPack shape pointer):
    // dl_tensor.shape points into ctx.shape (a Vec<i64> on the heap inside ExportContext).
    // ctx is moved to the heap via Box::into_raw and stored in managed.manager_ctx.
    // The deleter (nsl_dlpack_deleter) drops ctx AFTER the DLManagedTensor is freed.
    // Therefore dl_tensor.shape remains valid for the entire lifetime of managed.
    // The consumer MUST NOT cache dl_tensor.shape beyond the managed tensor's lifetime.
    let dl_tensor = DLTensor {
        data: tensor.data,
        device: nsl_device_to_dl(tensor.device),
        ndim: ndim as c_int,
        dtype: dl_dtype,
        shape: ctx.shape.as_mut_ptr(),
        strides: if ctx.strides.is_empty() { std::ptr::null_mut() } else { ctx.strides.as_mut_ptr() },
        byte_offset: 0,
    };

    let managed = Box::new(DLManagedTensor {
        dl_tensor,
        manager_ctx: Box::into_raw(ctx) as *mut c_void,
        deleter: Some(nsl_dlpack_deleter),
    });

    Box::into_raw(managed)
}

/// Does this tensor's storage bottom out in memory NSL itself allocated?
///
/// Walks the `data_owner` view chain to its root and reports whether that
/// root actually owns its buffer. A root with `owns_data == 0` is a BORROW
/// wrapper over foreign memory — the caller's input buffer, when an
/// `@export` returns one of its parameters (or a view of one) — and its
/// storage must never be handed to a DLPack consumer as owned: the consumer
/// can outlive the caller's buffer (use-after-free), and the deleter would
/// be releasing storage NSL never allocated.
///
/// Slab-managed buffers count as NSL-owned (the slab is NSL's; `nsl_tensor_free`
/// correctly declines to free individual slab offsets).
///
/// The walk is depth-bounded: a corrupted or cyclic chain returns `false`
/// (refuse) rather than spinning.
fn storage_is_nsl_owned(tensor_ptr: i64) -> bool {
    const MAX_VIEW_DEPTH: usize = 64;
    let mut cur = tensor_ptr;
    for _ in 0..MAX_VIEW_DEPTH {
        if cur == 0 {
            return false;
        }
        let t = NslTensor::from_ptr(cur);
        if t.data_owner == 0 {
            return t.owns_data == 1 || t.slab_managed == 1;
        }
        cur = t.data_owner;
    }
    false
}

/// Convert an NslTensor to a DLManagedTensor (zero-copy, BORROW export).
///
/// The returned DLManagedTensor shares the same data pointer; the NslTensor's
/// lifetime stays managed by NSL and the deleter frees only the wrapper.
/// Returns null if the dtype has no DLPack representation.
pub fn nsl_tensor_to_dlpack(tensor: &NslTensor, tensor_ptr: i64) -> *mut DLManagedTensor {
    build_dlpack(tensor, tensor_ptr, false)
}

/// Convert an NslTensor to a DLManagedTensor, TRANSFERRING ownership
/// (`nsl_model_call_alloc` output path).
///
/// On `Ok`, the consumer's deleter call releases the NslTensor via
/// `nsl_tensor_free` — exactly once. The caller must NOT free the tensor
/// itself after a successful transfer. Refuses (leaving ownership with the
/// caller) when:
///   - the tensor is GPU-resident: the deleter runs on the consumer's GC
///     thread at an arbitrary later time, and freeing device memory there
///     requires CUDA-context discipline this path does not implement (v1
///     exports CPU results only), or
///   - the dtype has no DLPack representation (fp8 / u16 tags / custom).
pub fn nsl_tensor_to_dlpack_owned(
    tensor_ptr: i64,
) -> Result<*mut DLManagedTensor, String> {
    if tensor_ptr == 0 {
        return Err("nsl_tensor_to_dlpack_owned: null tensor pointer".to_string());
    }
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if !storage_is_nsl_owned(tensor_ptr) {
        return Err(
            "output aliases memory NSL does not own (the export returned an \
             input, or a view of one). Ownership cannot be transferred: the \
             consumer would outlive the caller's buffer and the deleter would \
             free storage NSL never allocated. Return a computed tensor (e.g. \
             `x * 1.0`) or use nsl_model_call_into to copy into caller memory"
                .to_string(),
        );
    }
    if tensor.device != 0 {
        return Err(format!(
            "output tensor is GPU-resident (device={}); DLPack ownership \
             transfer supports CPU results only in v1 — the deleter runs on \
             the consumer's thread where no CUDA context is guaranteed",
            tensor.device
        ));
    }
    let managed = build_dlpack(tensor, tensor_ptr, true);
    if managed.is_null() {
        return Err(format!(
            "output tensor dtype tag {} has no DLPack representation \
             (exportable: f64, f32, f16, bf16, int8, int32)",
            tensor.dtype
        ));
    }
    Ok(managed)
}

// ---------------------------------------------------------------------------
// DLManagedTensor -> NslTensor (zero-copy import)
// ---------------------------------------------------------------------------

/// Convert a DLManagedTensor to an NslTensor (zero-copy).
///
/// The new NslTensor borrows the data (owns_data=0). The caller is
/// responsible for keeping the DLManagedTensor alive while the NslTensor
/// is in use, then calling the DLManagedTensor's deleter.
///
/// Returns null (0) if the DLPack dtype is unsupported OR the device type
/// is one NSL cannot address (see `dl_device_to_nsl` — unknown backends
/// used to silently map to CPU, turning e.g. a Metal buffer into a wild
/// CPU pointer).
pub fn dlpack_to_nsl_tensor(managed: &DLManagedTensor) -> i64 {
    let dl = &managed.dl_tensor;

    let nsl_dtype = match dl_dtype_to_nsl(&dl.dtype) {
        Some(d) => d,
        None => return 0,
    };
    let device = match dl_device_to_nsl(&dl.device) {
        Some(d) => d,
        None => return 0,
    };

    let ndim = dl.ndim as usize;

    // Compute total element count from shape.
    let len: i64 = if ndim == 0 {
        1
    } else if dl.shape.is_null() {
        0
    } else {
        let shape_slice = unsafe { std::slice::from_raw_parts(dl.shape, ndim) };
        shape_slice.iter().product()
    };

    // Copy shape into NSL-managed memory.
    let shape_ptr = if ndim > 0 && !dl.shape.is_null() {
        let ptr = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { std::ptr::copy_nonoverlapping(dl.shape, ptr, ndim); }
        ptr
    } else {
        std::ptr::null_mut()
    };

    // Copy strides or compute contiguous strides.
    let strides_ptr = if ndim > 0 {
        if !dl.strides.is_null() {
            let ptr = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
            unsafe { std::ptr::copy_nonoverlapping(dl.strides, ptr, ndim); }
            ptr
        } else {
            // Compute contiguous strides.
            NslTensor::compute_strides(shape_ptr, ndim as i64)
        }
    } else {
        std::ptr::null_mut()
    };

    // Compute the actual data pointer with byte_offset applied.
    let data = unsafe { (dl.data as *mut u8).add(dl.byte_offset as usize) as *mut c_void };

    let tensor = Box::new(NslTensor::new(
        data,
        shape_ptr,
        strides_ptr,
        ndim as i64,
        len,
        device,
        nsl_dtype,
        0,
        0,
    ));

    Box::into_raw(tensor) as i64
}

// ---------------------------------------------------------------------------
// FFI exports (Cranelift-callable, i64 params)
// ---------------------------------------------------------------------------

/// Export an NslTensor as a DLManagedTensor pointer (zero-copy).
///
/// Returns a pointer to the DLManagedTensor, or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_dlpack_export(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let tensor = NslTensor::from_ptr(tensor_ptr);
    nsl_tensor_to_dlpack(tensor, tensor_ptr) as i64
}

/// Import a DLManagedTensor as an NslTensor pointer (zero-copy).
///
/// Returns a pointer to a new NslTensor (with owns_data=0), or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_dlpack_import(dlpack_ptr: i64) -> i64 {
    if dlpack_ptr == 0 {
        return 0;
    }
    let managed = unsafe { &*(dlpack_ptr as *const DLManagedTensor) };
    dlpack_to_nsl_tensor(managed)
}

/// Free a DLManagedTensor that was created by nsl_dlpack_export.
///
/// Calls the deleter callback on the managed tensor.
#[no_mangle]
pub extern "C" fn nsl_dlpack_free(dlpack_ptr: i64) {
    if dlpack_ptr == 0 {
        return;
    }
    let managed = dlpack_ptr as *mut DLManagedTensor;
    unsafe {
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::NslTensor;
    use crate::memory::checked_alloc;
    use std::ffi::c_void;

    /// Helper: create a simple f64 tensor on CPU for testing.
    fn make_test_tensor(data: &[f64], shape: &[i64]) -> (*mut NslTensor, i64) {
        let ndim = shape.len();
        let len: i64 = shape.iter().product();
        assert_eq!(data.len(), len as usize);

        let data_ptr = checked_alloc(len as usize * std::mem::size_of::<f64>()) as *mut f64;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, len as usize); }

        let shape_ptr = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { std::ptr::copy_nonoverlapping(shape.as_ptr(), shape_ptr, ndim); }

        let strides_ptr = NslTensor::compute_strides(shape_ptr, ndim as i64);

        let tensor = Box::new(NslTensor::new(
            data_ptr as *mut c_void,
            shape_ptr,
            strides_ptr,
            ndim as i64,
            len,
            0,
            DTYPE_F64,
            1,
            0,
        ));

        let ptr = Box::into_raw(tensor);
        (ptr, ptr as i64)
    }

    /// Helper: create a simple f32 tensor on CPU.
    fn make_test_tensor_f32(data: &[f32], shape: &[i64]) -> (*mut NslTensor, i64) {
        let ndim = shape.len();
        let len: i64 = shape.iter().product();
        assert_eq!(data.len(), len as usize);

        let data_ptr = checked_alloc(len as usize * std::mem::size_of::<f32>()) as *mut f32;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, len as usize); }

        let shape_ptr = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { std::ptr::copy_nonoverlapping(shape.as_ptr(), shape_ptr, ndim); }

        let strides_ptr = NslTensor::compute_strides(shape_ptr, ndim as i64);

        let tensor = Box::new(NslTensor::new(
            data_ptr as *mut c_void,
            shape_ptr,
            strides_ptr,
            ndim as i64,
            len,
            0,
            DTYPE_F32,
            1,
            0,
        ));

        let ptr = Box::into_raw(tensor);
        (ptr, ptr as i64)
    }

    #[test]
    fn test_nsl_dtype_to_dl_f64() {
        let dl = nsl_dtype_to_dl(DTYPE_F64).expect("f64 maps");
        assert_eq!(dl.code, DLDataTypeCode::KDLFloat as u8);
        assert_eq!(dl.bits, 64);
        assert_eq!(dl.lanes, 1);
    }

    #[test]
    fn test_nsl_dtype_to_dl_f32() {
        let dl = nsl_dtype_to_dl(DTYPE_F32).expect("f32 maps");
        assert_eq!(dl.code, DLDataTypeCode::KDLFloat as u8);
        assert_eq!(dl.bits, 32);
        assert_eq!(dl.lanes, 1);
    }

    #[test]
    fn test_dl_dtype_roundtrip() {
        for &dtype in &[DTYPE_F64, DTYPE_F32, DTYPE_FP16, DTYPE_BF16, DTYPE_INT8] {
            let dl = nsl_dtype_to_dl(dtype).expect("built-in dtype maps");
            let back = dl_dtype_to_nsl(&dl).unwrap();
            assert_eq!(back, dtype, "Roundtrip failed for dtype {dtype}");
        }
        // The unmapped set REFUSES rather than mislabeling as f64.
        for dtype in [
            crate::tensor::DTYPE_FP8E4M3,
            crate::tensor::DTYPE_FP8E5M2,
            7, // u16-token
            8, // u16-segment
            256, // custom
        ] {
            assert!(nsl_dtype_to_dl(dtype).is_none(), "dtype {dtype} must refuse");
        }
    }

    #[test]
    fn test_device_mapping() {
        // CPU
        let dl_cpu = nsl_device_to_dl(0);
        assert_eq!(dl_cpu.device_type, KDL_CPU);
        assert_eq!(dl_device_to_nsl(&dl_cpu), Some(0));

        // CUDA device 0 (NSL device=1)
        let dl_cuda = nsl_device_to_dl(1);
        assert_eq!(dl_cuda.device_type, KDL_CUDA);
        assert_eq!(dl_cuda.device_id, 0);
        assert_eq!(dl_device_to_nsl(&dl_cuda), Some(1));

        // CUDA device 1 (NSL device=2)
        let dl_cuda2 = nsl_device_to_dl(2);
        assert_eq!(dl_cuda2.device_type, KDL_CUDA);
        assert_eq!(dl_cuda2.device_id, 1);
        assert_eq!(dl_device_to_nsl(&dl_cuda2), Some(2));

        // CUDA pinned host memory is CPU-addressable → CPU.
        let pinned = DLDevice { device_type: KDL_CUDA_HOST, device_id: 0 };
        assert_eq!(dl_device_to_nsl(&pinned), Some(0));

        // Unknown / unaddressable backends REFUSE rather than aliasing CPU.
        for dt in [KDL_METAL, KDL_ROCM, 4, 7, 0, -1, 12345] {
            let dev = DLDevice { device_type: dt, device_id: 0 };
            assert_eq!(dl_device_to_nsl(&dev), None, "device_type {dt} must refuse");
        }
    }

    #[test]
    fn test_export_import_roundtrip_f64() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2i64, 3];
        let (_tensor_raw, tensor_ptr) = make_test_tensor(&data, &shape);

        // Export to DLPack
        let dlpack_ptr = nsl_dlpack_export(tensor_ptr);
        assert_ne!(dlpack_ptr, 0);

        let managed = unsafe { &*(dlpack_ptr as *const DLManagedTensor) };
        assert_eq!(managed.dl_tensor.ndim, 2);
        assert_eq!(managed.dl_tensor.dtype.bits, 64);
        assert_eq!(managed.dl_tensor.dtype.code, DLDataTypeCode::KDLFloat as u8);
        assert_eq!(managed.dl_tensor.device.device_type, KDL_CPU);

        // Check shape
        let dl_shape = unsafe { std::slice::from_raw_parts(managed.dl_tensor.shape, 2) };
        assert_eq!(dl_shape, &[2, 3]);

        // Import back
        let imported_ptr = nsl_dlpack_import(dlpack_ptr);
        assert_ne!(imported_ptr, 0);
        let imported = NslTensor::from_ptr(imported_ptr);
        assert_eq!(imported.ndim, 2);
        assert_eq!(imported.len, 6);
        assert_eq!(imported.dtype, DTYPE_F64);
        assert_eq!(imported.device, 0);
        assert_eq!(imported.owns_data, 0); // borrowed

        // Verify data pointer is the same (zero-copy)
        let original = NslTensor::from_ptr(tensor_ptr);
        assert_eq!(imported.data, original.data);

        // Verify data values
        let imported_data = unsafe { std::slice::from_raw_parts(imported.data as *const f64, 6) };
        assert_eq!(imported_data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Clean up
        nsl_dlpack_free(dlpack_ptr);
        // Free the imported tensor (only the wrapper, not the data since owns_data=0)
        unsafe { drop(Box::from_raw(imported_ptr as *mut NslTensor)); }
    }

    #[test]
    fn test_export_import_roundtrip_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![4i64];
        let (_tensor_raw, tensor_ptr) = make_test_tensor_f32(&data, &shape);

        let dlpack_ptr = nsl_dlpack_export(tensor_ptr);
        assert_ne!(dlpack_ptr, 0);

        let managed = unsafe { &*(dlpack_ptr as *const DLManagedTensor) };
        assert_eq!(managed.dl_tensor.dtype.bits, 32);
        assert_eq!(managed.dl_tensor.dtype.code, DLDataTypeCode::KDLFloat as u8);
        assert_eq!(managed.dl_tensor.ndim, 1);

        let imported_ptr = nsl_dlpack_import(dlpack_ptr);
        assert_ne!(imported_ptr, 0);
        let imported = NslTensor::from_ptr(imported_ptr);
        assert_eq!(imported.dtype, DTYPE_F32);
        assert_eq!(imported.len, 4);

        let imported_data = unsafe { std::slice::from_raw_parts(imported.data as *const f32, 4) };
        assert_eq!(imported_data, &[1.0f32, 2.0, 3.0, 4.0]);

        nsl_dlpack_free(dlpack_ptr);
        unsafe { drop(Box::from_raw(imported_ptr as *mut NslTensor)); }
    }

    #[test]
    fn test_null_pointer_safety() {
        assert_eq!(nsl_dlpack_export(0), 0);
        assert_eq!(nsl_dlpack_import(0), 0);
        // Should not panic
        nsl_dlpack_free(0);
    }

    #[test]
    fn test_unsupported_dtype_returns_zero() {
        // Create a DLManagedTensor with an unsupported dtype (e.g., complex128)
        let shape = vec![2i64];
        let data = vec![0u8; 32]; // dummy data

        let dl_tensor = DLTensor {
            data: data.as_ptr() as *mut c_void,
            device: DLDevice { device_type: KDL_CPU, device_id: 0 },
            ndim: 1,
            dtype: DLDataType { code: 5, bits: 128, lanes: 1 }, // unsupported
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };

        let managed = DLManagedTensor {
            dl_tensor,
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        let result = dlpack_to_nsl_tensor(&managed);
        assert_eq!(result, 0, "Unsupported dtype should return 0");
    }

    /// The owned deleter releases the NslTensor via `nsl_tensor_free` —
    /// observed through the refcount: we hold an extra reference, so the
    /// deleter's release drops 2 → 1 instead of freeing, and the struct
    /// stays readable for the assertion.
    #[test]
    fn owned_deleter_releases_the_tensor_exactly_once() {
        use std::sync::atomic::Ordering;
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let (_raw, tensor_ptr) = make_test_tensor_f32(&data, &[4]);
        let tensor = NslTensor::from_ptr(tensor_ptr);
        // Hold a second reference so the deleter's free is observable.
        tensor.refcount.fetch_add(1, Ordering::SeqCst);
        assert_eq!(tensor.refcount.load(Ordering::SeqCst), 2);

        let managed = nsl_tensor_to_dlpack_owned(tensor_ptr).expect("owned export");
        assert!(!managed.is_null());
        // Ownership transfer must not change the refcount by itself.
        assert_eq!(tensor.refcount.load(Ordering::SeqCst), 2);

        nsl_dlpack_free(managed as i64);
        assert_eq!(
            tensor.refcount.load(Ordering::SeqCst),
            1,
            "the owned deleter must have released exactly one reference"
        );
        // Release our own reference; this frees the tensor for real.
        crate::tensor::nsl_tensor_free(tensor_ptr);
    }

    /// The BORROW export (`nsl_dlpack_export`) must keep its historical
    /// semantics: the deleter frees only the wrapper, never the tensor.
    #[test]
    fn borrow_deleter_does_not_release_the_tensor() {
        use std::sync::atomic::Ordering;
        let data = vec![1.0f32, 2.0];
        let (_raw, tensor_ptr) = make_test_tensor_f32(&data, &[2]);
        let tensor = NslTensor::from_ptr(tensor_ptr);
        assert_eq!(tensor.refcount.load(Ordering::SeqCst), 1);

        let dl = nsl_dlpack_export(tensor_ptr);
        assert_ne!(dl, 0);
        nsl_dlpack_free(dl);
        assert_eq!(
            tensor.refcount.load(Ordering::SeqCst),
            1,
            "the borrow deleter must not touch the tensor's refcount"
        );
        crate::tensor::nsl_tensor_free(tensor_ptr);
    }

    /// Owned export refuses GPU-resident tensors (the deleter runs on the
    /// consumer's thread with no CUDA context guarantee) — ownership stays
    /// with the caller on refusal.
    #[test]
    fn owned_export_refuses_gpu_resident_tensors() {
        let data = vec![1.0f32];
        let (_raw, tensor_ptr) = make_test_tensor_f32(&data, &[1]);
        {
            let tensor = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
            tensor.device = 1; // pretend CUDA; refused before any data deref
        }
        let err = nsl_tensor_to_dlpack_owned(tensor_ptr).unwrap_err();
        assert!(
            err.contains("GPU-resident"),
            "the refusal must say WHY: {err}"
        );
        {
            let tensor = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
            tensor.device = 0;
        }
        crate::tensor::nsl_tensor_free(tensor_ptr);
    }

    /// Export-side dtype refusal: fp8/u16/custom used to be silently LABELED
    /// float64 (a consumer reading 8× past the buffer); both exporters must
    /// now refuse.
    #[test]
    fn export_refuses_dtypes_without_dlpack_representation() {
        let data = vec![0u8; 8];
        let data_ptr = checked_alloc(8);
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, 8) };
        let shape_ptr = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *shape_ptr = 8 };
        let strides_ptr = NslTensor::compute_strides(shape_ptr, 1);
        let tensor = Box::new(NslTensor::new(
            data_ptr as *mut c_void,
            shape_ptr,
            strides_ptr,
            1,
            8,
            0,
            crate::tensor::DTYPE_FP8E4M3,
            1,
            0,
        ));
        let tensor_ptr = Box::into_raw(tensor) as i64;

        assert_eq!(
            nsl_dlpack_export(tensor_ptr),
            0,
            "borrow export must refuse an fp8 tensor, not mislabel it f64"
        );
        let err = nsl_tensor_to_dlpack_owned(tensor_ptr).unwrap_err();
        assert!(
            err.contains("no DLPack representation"),
            "owned export must refuse with the reason: {err}"
        );
        // NOT freed via nsl_tensor_free: `dtype_element_size` has no fp8 arm
        // and panics inside the (nounwind) free path — a pre-existing runtime
        // defect this test must not depend on. Drop only the struct box; the
        // few bytes of shape/data scratch are irrelevant to a test process.
        unsafe { drop(Box::from_raw(tensor_ptr as *mut NslTensor)) };
    }

    /// Import-side device refusal: unknown backends used to silently map to
    /// CPU — a Metal buffer became a wild CPU pointer on first touch.
    #[test]
    fn import_refuses_unaddressable_device_types() {
        let mut shape = vec![2i64];
        let data = vec![0u8; 16];
        for dt in [KDL_METAL, KDL_ROCM, 4, 7, -3] {
            let managed = DLManagedTensor {
                dl_tensor: DLTensor {
                    data: data.as_ptr() as *mut c_void,
                    device: DLDevice { device_type: dt, device_id: 0 },
                    ndim: 1,
                    dtype: DLDataType {
                        code: DLDataTypeCode::KDLFloat as u8,
                        bits: 64,
                        lanes: 1,
                    },
                    shape: shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            };
            assert_eq!(
                dlpack_to_nsl_tensor(&managed),
                0,
                "device_type {dt} must refuse on import"
            );
        }
        // Pinned host memory (kDLCUDAHost) IS CPU-addressable and imports.
        let managed = DLManagedTensor {
            dl_tensor: DLTensor {
                data: data.as_ptr() as *mut c_void,
                device: DLDevice { device_type: KDL_CUDA_HOST, device_id: 0 },
                ndim: 1,
                dtype: DLDataType {
                    code: DLDataTypeCode::KDLFloat as u8,
                    bits: 64,
                    lanes: 1,
                },
                shape: shape.as_mut_ptr(),
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };
        let imported = dlpack_to_nsl_tensor(&managed);
        assert_ne!(imported, 0, "kDLCUDAHost must import as CPU");
        let t = NslTensor::from_ptr(imported);
        assert_eq!(t.device, 0);
        unsafe { drop(Box::from_raw(imported as *mut NslTensor)) };
    }

    #[test]
    fn test_scalar_tensor() {
        // Scalar tensor: ndim=0, shape=[], len=1
        let data_ptr = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
        unsafe { *data_ptr = 42.0; }

        let tensor = Box::new(NslTensor::new(
            data_ptr as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            1,
            0,
            DTYPE_F64,
            1,
            0,
        ));

        let tensor_ptr = Box::into_raw(tensor) as i64;

        let dlpack_ptr = nsl_dlpack_export(tensor_ptr);
        assert_ne!(dlpack_ptr, 0);

        let managed = unsafe { &*(dlpack_ptr as *const DLManagedTensor) };
        assert_eq!(managed.dl_tensor.ndim, 0);

        let imported_ptr = nsl_dlpack_import(dlpack_ptr);
        assert_ne!(imported_ptr, 0);
        let imported = NslTensor::from_ptr(imported_ptr);
        assert_eq!(imported.ndim, 0);
        assert_eq!(imported.len, 1);

        let val = unsafe { *(imported.data as *const f64) };
        assert_eq!(val, 42.0);

        nsl_dlpack_free(dlpack_ptr);
        unsafe { drop(Box::from_raw(imported_ptr as *mut NslTensor)); }
    }
}
