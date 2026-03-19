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

/// DLPack device type codes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDeviceType {
    /// CPU device.
    KDLCpu = 1,
    /// CUDA GPU.
    KDLCuda = 2,
    /// Apple Metal.
    KDLMetal = 8,
    /// AMD ROCm.
    KDLRocm = 10,
}

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
fn nsl_dtype_to_dl(dtype: u16) -> DLDataType {
    match dtype {
        DTYPE_F64 => DLDataType { code: DLDataTypeCode::KDLFloat as u8, bits: 64, lanes: 1 },
        DTYPE_F32 => DLDataType { code: DLDataTypeCode::KDLFloat as u8, bits: 32, lanes: 1 },
        DTYPE_FP16 => DLDataType { code: DLDataTypeCode::KDLFloat as u8, bits: 16, lanes: 1 },
        DTYPE_BF16 => DLDataType { code: DLDataTypeCode::KDLBfloat as u8, bits: 16, lanes: 1 },
        DTYPE_INT8 => DLDataType { code: DLDataTypeCode::KDLInt as u8, bits: 8, lanes: 1 },
        _ => DLDataType { code: DLDataTypeCode::KDLFloat as u8, bits: 64, lanes: 1 },
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
        _ => None,
    }
}

/// Convert an NSL device code to a DLPack DLDevice.
fn nsl_device_to_dl(device: u8) -> DLDevice {
    if device == 0 {
        DLDevice { device_type: DLDeviceType::KDLCpu, device_id: 0 }
    } else {
        DLDevice { device_type: DLDeviceType::KDLCuda, device_id: (device - 1) as c_int }
    }
}

/// Convert a DLPack DLDevice to an NSL device code.
fn dl_device_to_nsl(dev: &DLDevice) -> u8 {
    match dev.device_type {
        DLDeviceType::KDLCpu => 0,
        DLDeviceType::KDLCuda => (dev.device_id + 1) as u8,
        // Treat unknown backends as CPU for now.
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// NslTensor -> DLManagedTensor (zero-copy export)
// ---------------------------------------------------------------------------

/// Context stored in `manager_ctx` for our exported DLManagedTensors.
/// Holds the original NslTensor pointer so we can manage its lifetime.
struct ExportContext {
    /// Raw pointer to the NslTensor that owns the data.
    /// Kept for future use (M62b: reference counting / lifetime bridging).
    #[allow(dead_code)]
    nsl_tensor_ptr: i64,
    /// Copies of shape/strides for DLPack (DLPack uses its own arrays).
    shape: Vec<i64>,
    strides: Vec<i64>,
}

/// DLPack deleter callback for tensors exported from NSL.
///
/// # Safety
/// Called by the consumer when they are done with the DLManagedTensor.
/// We drop the ExportContext and the DLManagedTensor allocation.
/// The NslTensor data remains valid (NslTensor lifetime is managed by NSL).
unsafe extern "C" fn nsl_dlpack_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    let ctx_ptr = (*managed).manager_ctx as *mut ExportContext;
    if !ctx_ptr.is_null() {
        drop(Box::from_raw(ctx_ptr));
    }
    drop(Box::from_raw(managed));
}

/// Convert an NslTensor to a DLManagedTensor (zero-copy).
///
/// The returned DLManagedTensor shares the same data pointer. The caller
/// must call the deleter when done with it.
pub fn nsl_tensor_to_dlpack(tensor: &NslTensor, tensor_ptr: i64) -> *mut DLManagedTensor {
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
        shape,
        strides,
    });

    let dl_tensor = DLTensor {
        data: tensor.data,
        device: nsl_device_to_dl(tensor.device),
        ndim: ndim as c_int,
        dtype: nsl_dtype_to_dl(tensor.dtype),
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

// ---------------------------------------------------------------------------
// DLManagedTensor -> NslTensor (zero-copy import)
// ---------------------------------------------------------------------------

/// Convert a DLManagedTensor to an NslTensor (zero-copy).
///
/// The new NslTensor borrows the data (owns_data=0). The caller is
/// responsible for keeping the DLManagedTensor alive while the NslTensor
/// is in use, then calling the DLManagedTensor's deleter.
///
/// Returns null (0) if the DLPack dtype is unsupported.
pub fn dlpack_to_nsl_tensor(managed: &DLManagedTensor) -> i64 {
    let dl = &managed.dl_tensor;

    let nsl_dtype = match dl_dtype_to_nsl(&dl.dtype) {
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

    let device = dl_device_to_nsl(&dl.device);

    let tensor = Box::new(NslTensor {
        data,
        shape: shape_ptr,
        strides: strides_ptr,
        ndim: ndim as i64,
        len,
        refcount: 1,
        device,
        dtype: nsl_dtype,
        owns_data: 0, // Borrowed — DLPack consumer owns the data.
    });

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

        let tensor = Box::new(NslTensor {
            data: data_ptr as *mut c_void,
            shape: shape_ptr,
            strides: strides_ptr,
            ndim: ndim as i64,
            len,
            refcount: 1,
            device: 0,
            dtype: DTYPE_F64,
            owns_data: 1,
        });

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

        let tensor = Box::new(NslTensor {
            data: data_ptr as *mut c_void,
            shape: shape_ptr,
            strides: strides_ptr,
            ndim: ndim as i64,
            len,
            refcount: 1,
            device: 0,
            dtype: DTYPE_F32,
            owns_data: 1,
        });

        let ptr = Box::into_raw(tensor);
        (ptr, ptr as i64)
    }

    #[test]
    fn test_nsl_dtype_to_dl_f64() {
        let dl = nsl_dtype_to_dl(DTYPE_F64);
        assert_eq!(dl.code, DLDataTypeCode::KDLFloat as u8);
        assert_eq!(dl.bits, 64);
        assert_eq!(dl.lanes, 1);
    }

    #[test]
    fn test_nsl_dtype_to_dl_f32() {
        let dl = nsl_dtype_to_dl(DTYPE_F32);
        assert_eq!(dl.code, DLDataTypeCode::KDLFloat as u8);
        assert_eq!(dl.bits, 32);
        assert_eq!(dl.lanes, 1);
    }

    #[test]
    fn test_dl_dtype_roundtrip() {
        for &dtype in &[DTYPE_F64, DTYPE_F32, DTYPE_FP16, DTYPE_BF16, DTYPE_INT8] {
            let dl = nsl_dtype_to_dl(dtype);
            let back = dl_dtype_to_nsl(&dl).unwrap();
            assert_eq!(back, dtype, "Roundtrip failed for dtype {dtype}");
        }
    }

    #[test]
    fn test_device_mapping() {
        // CPU
        let dl_cpu = nsl_device_to_dl(0);
        assert_eq!(dl_cpu.device_type, DLDeviceType::KDLCpu);
        assert_eq!(dl_device_to_nsl(&dl_cpu), 0);

        // CUDA device 0 (NSL device=1)
        let dl_cuda = nsl_device_to_dl(1);
        assert_eq!(dl_cuda.device_type, DLDeviceType::KDLCuda);
        assert_eq!(dl_cuda.device_id, 0);
        assert_eq!(dl_device_to_nsl(&dl_cuda), 1);

        // CUDA device 1 (NSL device=2)
        let dl_cuda2 = nsl_device_to_dl(2);
        assert_eq!(dl_cuda2.device_type, DLDeviceType::KDLCuda);
        assert_eq!(dl_cuda2.device_id, 1);
        assert_eq!(dl_device_to_nsl(&dl_cuda2), 2);
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
        assert_eq!(managed.dl_tensor.device.device_type, DLDeviceType::KDLCpu);

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
            device: DLDevice { device_type: DLDeviceType::KDLCpu, device_id: 0 },
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

    #[test]
    fn test_scalar_tensor() {
        // Scalar tensor: ndim=0, shape=[], len=1
        let data_ptr = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
        unsafe { *data_ptr = 42.0; }

        let tensor = Box::new(NslTensor {
            data: data_ptr as *mut c_void,
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            ndim: 0,
            len: 1,
            refcount: 1,
            device: 0,
            dtype: DTYPE_F64,
            owns_data: 1,
        });

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
