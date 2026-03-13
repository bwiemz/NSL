use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

use crate::autodiff;
use crate::list::{nsl_list_new, nsl_list_push, NslList};
use crate::memory::{checked_alloc, checked_alloc_zeroed, checked_free};

// ---------------------------------------------------------------------------
// Global training mode (thread-local)
// ---------------------------------------------------------------------------
thread_local! {
    static TRAINING_MODE: Cell<bool> = const { Cell::new(false) };
}

#[no_mangle]
pub extern "C" fn nsl_set_training_mode(mode: i8) {
    TRAINING_MODE.with(|t| t.set(mode != 0));
}

#[no_mangle]
pub extern "C" fn nsl_is_training() -> i8 {
    TRAINING_MODE.with(|t| t.get() as i8)
}

#[repr(C)]
pub struct NslTensor {
    pub(crate) data: *mut c_void,   // Opaque: CPU f64 or GPU f32
    pub(crate) shape: *mut i64,
    pub(crate) strides: *mut i64,
    pub(crate) ndim: i64,
    pub(crate) len: i64,
    pub(crate) refcount: i64,
    pub(crate) device: u8,          // 0 = CPU, 1+ = CUDA device ID
    pub(crate) dtype: u16,          // 0 = f64, 1 = f32; 256+ = custom user-defined dtypes
    pub(crate) owns_data: u8,       // 1 = heap-owned (free on drop), 0 = borrowed/mmap
}

// Built-in dtype IDs (match existing u8 values)
pub const DTYPE_F64: u16 = 0;
pub const DTYPE_F32: u16 = 1;
// Future: FP16=2, BF16=3, INT8=4, etc.

// Custom dtype IDs start at 256
pub const DTYPE_CUSTOM_START: u16 = 256;

/// Metadata for a user-defined custom datatype
pub struct CustomDtypeInfo {
    pub id: u16,
    pub name: String,
    pub bit_width: u8,
    pub block_size: u32,           // 0 = element-wise, >0 = block format
    pub element_size: usize,       // bytes per packed ELEMENT (ceil(bits/8))
    pub packed_block_size: usize,  // bytes per packed BLOCK — 0 for element-wise
    pub pack_fn: Option<*const c_void>,
    pub unpack_fn: Option<*const c_void>,
}

// SAFETY: function pointers are set once at startup, read-only after.
unsafe impl Send for CustomDtypeInfo {}
unsafe impl Sync for CustomDtypeInfo {}

/// Registry uses OnceLock — initialized once at startup, read-only after.
static CUSTOM_DTYPE_REGISTRY: OnceLock<HashMap<u16, CustomDtypeInfo>> = OnceLock::new();

fn get_registry() -> &'static HashMap<u16, CustomDtypeInfo> {
    CUSTOM_DTYPE_REGISTRY.get().expect("custom dtype registry not initialized")
}

thread_local! {
    static STAGING_REGISTRY: std::cell::RefCell<HashMap<u16, CustomDtypeInfo>>
        = std::cell::RefCell::new(HashMap::new());
}

/// Register a custom datatype. Called by codegen-generated init code at startup.
/// All params are i64 to match Cranelift calling convention.
#[no_mangle]
pub extern "C" fn nsl_register_custom_dtype(
    id: i64,
    name_ptr: i64,
    name_len: i64,
    bit_width: i64,
    block_size: i64,
    packed_block_size: i64,
    pack_fn: i64,
    unpack_fn: i64,
) {
    let id = id as u16;
    let bit_width = bit_width as u8;
    let block_size = block_size as u32;
    let packed_block_size = packed_block_size as u32;
    let name_ptr = name_ptr as *const u8;
    let pack_fn = pack_fn as *const c_void;
    let unpack_fn = unpack_fn as *const c_void;

    let name = unsafe {
        let slice = std::slice::from_raw_parts(name_ptr, name_len as usize);
        String::from_utf8_lossy(slice).into_owned()
    };
    let info = CustomDtypeInfo {
        id,
        name,
        bit_width,
        block_size,
        element_size: (bit_width as usize).div_ceil(8),
        packed_block_size: if block_size > 0 { packed_block_size as usize } else { 0 },
        pack_fn: if pack_fn.is_null() { None } else { Some(pack_fn) },
        unpack_fn: if unpack_fn.is_null() { None } else { Some(unpack_fn) },
    };

    STAGING_REGISTRY.with(|r| r.borrow_mut().insert(id, info));
}

/// Called once after all registrations. Moves staging → OnceLock.
#[no_mangle]
pub extern "C" fn nsl_finalize_dtype_registry() {
    STAGING_REGISTRY.with(|r| {
        let entries = std::mem::take(&mut *r.borrow_mut());
        if !entries.is_empty() {
            let _ = CUSTOM_DTYPE_REGISTRY.set(entries);
        }
    });
}

impl NslTensor {
    pub(crate) fn from_ptr(ptr: i64) -> &'static mut NslTensor {
        unsafe { &mut *(ptr as *mut NslTensor) }
    }

    #[inline]
    pub(crate) fn data_f64(&self) -> *mut f64 {
        assert_eq!(self.dtype, 0, "data_f64() called on non-f64 tensor (dtype={})", self.dtype);
        self.data as *mut f64
    }

    #[inline]
    pub(crate) fn data_f32(&self) -> *mut f32 {
        assert_eq!(self.dtype, 1, "data_f32() called on non-f32 tensor (dtype={})", self.dtype);
        self.data as *mut f32
    }

    #[inline]
    pub(crate) fn element_size(&self) -> usize {
        match self.dtype {
            DTYPE_F64 => std::mem::size_of::<f64>(),
            DTYPE_F32 => std::mem::size_of::<f32>(),
            id if id >= DTYPE_CUSTOM_START => {
                get_registry().get(&id).map(|info| info.element_size).unwrap_or(1)
            }
            _ => panic!("unknown dtype {}", self.dtype),
        }
    }

    pub(crate) fn compute_strides(shape: *const i64, ndim: i64) -> *mut i64 {
        let n = ndim as usize;
        let strides = checked_alloc(n * std::mem::size_of::<i64>()) as *mut i64;
        if n > 0 {
            unsafe {
                *strides.add(n - 1) = 1;
                for i in (0..n - 1).rev() {
                    *strides.add(i) = *strides.add(i + 1) * *shape.add(i + 1);
                }
            }
        }
        strides
    }

    fn total_elements(shape: *const i64, ndim: i64) -> i64 {
        let mut total: i64 = 1;
        for i in 0..ndim as usize {
            let dim = unsafe { *shape.add(i) };
            total = total.checked_mul(dim).unwrap_or_else(|| {
                eprintln!("nsl: tensor shape overflow — dimensions too large");
                std::process::abort();
            });
        }
        total
    }

    pub(crate) fn copy_shape(src: *const i64, ndim: i64) -> *mut i64 {
        let n = ndim as usize;
        if n == 0 {
            return std::ptr::null_mut();
        }
        let dst = checked_alloc(n * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { std::ptr::copy_nonoverlapping(src, dst, n); }
        dst
    }

    /// Returns true if the tensor has a contiguous row-major memory layout.
    /// A tensor is contiguous when strides match what compute_strides would produce.
    pub(crate) fn is_contiguous(&self) -> bool {
        if self.ndim <= 1 {
            return true;
        }
        let ndim = self.ndim as usize;
        unsafe {
            let mut expected_stride = 1_i64;
            for d in (0..ndim).rev() {
                if *self.strides.add(d) != expected_stride {
                    return false;
                }
                expected_stride *= *self.shape.add(d);
            }
        }
        true
    }

    /// Create a new contiguous (row-major) copy of this tensor.
    /// If the tensor is already contiguous, this behaves like clone.
    /// For strided/transposed tensors, data is re-laid-out into row-major order.
    pub(crate) fn make_contiguous(ptr: i64) -> i64 {
        let tensor = NslTensor::from_ptr(ptr);
        if tensor.is_contiguous() {
            // Fast path: just memcpy
            return nsl_tensor_clone(ptr);
        }

        let ndim = tensor.ndim as usize;
        let len = tensor.len as usize;
        let elem_size = tensor.element_size();

        let new_shape = NslTensor::copy_shape(tensor.shape, tensor.ndim);
        let new_strides = NslTensor::compute_strides(new_shape, tensor.ndim);
        let new_data = checked_alloc(len * elem_size);

        // Read shape and strides into vecs for indexing
        let shape_vec: Vec<i64> = (0..ndim).map(|i| unsafe { *tensor.shape.add(i) }).collect();
        let src_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *tensor.strides.add(i) }).collect();

        // Iterate over all elements by multi-index, copy from strided src to contiguous dst
        for flat in 0..len {
            // Convert flat index to multi-index (row-major)
            let mut remaining = flat;
            let mut src_offset = 0_usize;
            for d in 0..ndim {
                // Compute product of remaining dimensions for row-major decomposition
                let stride_in_flat: usize = if d + 1 < ndim {
                    shape_vec[d + 1..].iter().map(|&s| s as usize).product()
                } else {
                    1
                };
                let idx = remaining / stride_in_flat;
                remaining %= stride_in_flat;
                src_offset += idx * (src_strides[d] as usize);
            }

            if tensor.dtype == 1 {
                unsafe {
                    *(new_data as *mut f32).add(flat) = *tensor.data_f32().add(src_offset);
                }
            } else {
                unsafe {
                    *(new_data as *mut f64).add(flat) = *tensor.data_f64().add(src_offset);
                }
            }
        }

        let result = Box::new(NslTensor {
            data: new_data as *mut c_void,
            shape: new_shape,
            strides: new_strides,
            ndim: tensor.ndim,
            len: tensor.len,
            refcount: 1,
            device: tensor.device,
            dtype: tensor.dtype,
            owns_data: 1,
        });
        Box::into_raw(result) as i64
    }
}

/// Helper: create a tensor from a shape list, filling data with a given value (f32, dtype=1).
fn tensor_from_shape_list(shape_list: i64, fill: f64) -> i64 {
    let list = NslList::from_ptr(shape_list);
    let ndim = list.len;

    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        unsafe { *shape.add(i) = *list.data.add(i) };
    }

    let len = NslTensor::total_elements(shape, ndim);
    let fill_f32 = fill as f32;
    let data_size = (len as usize) * std::mem::size_of::<f32>();
    let data = if fill == 0.0 {
        checked_alloc_zeroed(data_size) as *mut f32
    } else {
        let data = checked_alloc(data_size) as *mut f32;
        for i in 0..len as usize {
            unsafe { *data.add(i) = fill_f32 };
        }
        data
    };

    let strides = NslTensor::compute_strides(shape, ndim);

    let tensor = Box::new(NslTensor {
        data: data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: 0,
        dtype: 1,
        owns_data: 1,
    });
    Box::into_raw(tensor) as i64
}

/// Helper: create a tensor from a shape list, filling data with a given value (f64, dtype=0).
/// Used for operations that explicitly require double precision.
#[allow(dead_code)]
fn tensor_from_shape_list_f64(shape_list: i64, fill: f64) -> i64 {
    let list = NslList::from_ptr(shape_list);
    let ndim = list.len;

    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        unsafe { *shape.add(i) = *list.data.add(i) };
    }

    let len = NslTensor::total_elements(shape, ndim);
    let data_size = (len as usize) * std::mem::size_of::<f64>();
    let data = if fill == 0.0 {
        checked_alloc_zeroed(data_size) as *mut f64
    } else {
        let data = checked_alloc(data_size) as *mut f64;
        for i in 0..len as usize {
            unsafe { *data.add(i) = fill };
        }
        data
    };

    let strides = NslTensor::compute_strides(shape, ndim);

    let tensor = Box::new(NslTensor {
        data: data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: 0,
        dtype: 0,
        owns_data: 1,
    });
    Box::into_raw(tensor) as i64
}

/// Create a 0-d scalar tensor containing a single f32 value (dtype=1).
fn create_scalar_tensor(value: f64) -> i64 {
    let data = checked_alloc(std::mem::size_of::<f32>()) as *mut f32;
    unsafe { *data = value as f32 };
    let tensor = Box::new(NslTensor {
        data: data as *mut c_void,
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        ndim: 0,
        len: 1,
        refcount: 1,
        device: 0,
        dtype: 1,
        owns_data: 1,
    });
    Box::into_raw(tensor) as i64
}

/// Create a 0-d scalar tensor with dtype-aware storage (dtype=0 → f64, dtype=1 → f32).
fn create_scalar_tensor_dtype(value: f64, dtype: u16) -> i64 {
    if dtype == 1 {
        create_scalar_tensor(value)
    } else {
        let data = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
        unsafe { *data = value };
        let tensor = Box::new(NslTensor {
            data: data as *mut c_void,
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            ndim: 0,
            len: 1,
            refcount: 1,
            device: 0,
            dtype: 0,
            owns_data: 1,
        });
        Box::into_raw(tensor) as i64
    }
}

// === Creation ===

#[no_mangle]
pub extern "C" fn nsl_tensor_zeros(shape_list: i64) -> i64 {
    tensor_from_shape_list(shape_list, 0.0)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_ones(shape_list: i64) -> i64 {
    tensor_from_shape_list(shape_list, 1.0)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_full(shape_list: i64, value: f64) -> i64 {
    tensor_from_shape_list(shape_list, value)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_rand(shape_list: i64) -> i64 {
    let ptr = tensor_from_shape_list(shape_list, 0.0);
    let tensor = NslTensor::from_ptr(ptr);
    for i in 0..tensor.len as usize {
        let val = crate::sampling::rng_f64() as f32;
        unsafe { *tensor.data_f32().add(i) = val };
    }
    ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_randn(shape_list: i64) -> i64 {
    let ptr = tensor_from_shape_list(shape_list, 0.0);
    let tensor = NslTensor::from_ptr(ptr);
    // Box-Muller transform: generate N(0,1) from uniform samples using seeded RNG
    let len = tensor.len as usize;
    let mut i = 0;
    while i + 1 < len {
        let u1 = crate::sampling::rng_f64().max(1e-15); // avoid log(0)
        let u2 = crate::sampling::rng_f64();

        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = (mag * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
        let z1 = (mag * (2.0 * std::f64::consts::PI * u2).sin()) as f32;
        unsafe {
            *tensor.data_f32().add(i) = z0;
            *tensor.data_f32().add(i + 1) = z1;
        }
        i += 2;
    }
    // If odd number of elements, generate one more pair and use first
    if i < len {
        let u1 = crate::sampling::rng_f64().max(1e-15);
        let u2 = crate::sampling::rng_f64();

        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = (mag * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
        unsafe { *tensor.data_f32().add(i) = z0 };
    }
    ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_arange(start: f64, stop: f64, step: f64) -> i64 {
    if step == 0.0 {
        eprintln!("nsl: tensor arange step cannot be zero");
        std::process::abort();
    }
    let len = ((stop - start) / step).ceil().max(0.0) as i64;

    // Create 1D tensor
    let ndim: i64 = 1;
    let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape = len };

    let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *strides = 1 };

    let data = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
    for i in 0..len as usize {
        unsafe { *data.add(i) = (start + (i as f64) * step) as f32 };
    }

    let tensor = Box::new(NslTensor {
        data: data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: 0,
        dtype: 1,
        owns_data: 1,
    });
    Box::into_raw(tensor) as i64
}

// === Element access ===

#[no_mangle]
pub extern "C" fn nsl_tensor_get(tensor_ptr: i64, indices_list: i64) -> f64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let indices = NslList::from_ptr(indices_list);

    if indices.len != tensor.ndim {
        eprintln!(
            "nsl: tensor index dimension mismatch (got {}, expected {})",
            indices.len, tensor.ndim
        );
        std::process::abort();
    }

    let mut offset: usize = 0;
    for i in 0..tensor.ndim as usize {
        let idx = unsafe { *indices.data.add(i) };
        let dim_size = unsafe { *tensor.shape.add(i) };
        if idx < 0 || idx >= dim_size {
            eprintln!(
                "nsl: tensor index out of bounds (index {} for dim {} of size {})",
                idx, i, dim_size
            );
            std::process::abort();
        }
        offset += (idx as usize) * (unsafe { *tensor.strides.add(i) } as usize);
    }

    if tensor.dtype == 1 {
        unsafe { *tensor.data_f32().add(offset) as f64 }
    } else {
        unsafe { *tensor.data_f64().add(offset) }
    }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_set(tensor_ptr: i64, indices_list: i64, value: f64) {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let indices = NslList::from_ptr(indices_list);

    if indices.len != tensor.ndim {
        eprintln!(
            "nsl: tensor index dimension mismatch (got {}, expected {})",
            indices.len, tensor.ndim
        );
        std::process::abort();
    }

    let mut offset: usize = 0;
    for i in 0..tensor.ndim as usize {
        let idx = unsafe { *indices.data.add(i) };
        let dim_size = unsafe { *tensor.shape.add(i) };
        if idx < 0 || idx >= dim_size {
            eprintln!(
                "nsl: tensor index out of bounds (index {} for dim {} of size {})",
                idx, i, dim_size
            );
            std::process::abort();
        }
        offset += (idx as usize) * (unsafe { *tensor.strides.add(i) } as usize);
    }

    if tensor.dtype == 1 {
        unsafe { *tensor.data_f32().add(offset) = value as f32 };
    } else {
        unsafe { *tensor.data_f64().add(offset) = value };
    }
}

// === Shape operations ===

#[no_mangle]
pub extern "C" fn nsl_tensor_shape(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let list = crate::list::nsl_list_new();
    for i in 0..tensor.ndim as usize {
        crate::list::nsl_list_push(list, unsafe { *tensor.shape.add(i) });
    }
    list
}

/// Return the size of a specific dimension of a tensor.
#[no_mangle]
pub extern "C" fn nsl_tensor_shape_dim(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
    if d >= ndim {
        eprintln!("nsl: shape_dim dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }
    unsafe { *tensor.shape.add(d) }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_ndim(tensor_ptr: i64) -> i64 {
    NslTensor::from_ptr(tensor_ptr).ndim
}

#[no_mangle]
pub extern "C" fn nsl_tensor_reshape(tensor_ptr: i64, new_shape_list: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let new_shape_nsl = NslList::from_ptr(new_shape_list);
    let new_ndim = new_shape_nsl.len;

    let new_shape = checked_alloc((new_ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    let mut new_len: i64 = 1;
    for i in 0..new_ndim as usize {
        let dim = unsafe { *new_shape_nsl.data.add(i) };
        unsafe { *new_shape.add(i) = dim };
        new_len *= dim;
    }

    if new_len != tensor.len {
        eprintln!(
            "nsl: cannot reshape tensor of size {} into shape of size {}",
            tensor.len, new_len
        );
        std::process::abort();
    }

    // Create a new tensor that shares data (increment refcount)
    tensor.refcount += 1;
    let strides = NslTensor::compute_strides(new_shape, new_ndim);

    let new_tensor = Box::new(NslTensor {
        data: tensor.data,
        shape: new_shape,
        strides,
        ndim: new_ndim,
        len: new_len,
        refcount: 1, // new wrapper has its own refcount
        device: tensor.device,
        dtype: tensor.dtype,
        owns_data: 1,
    });
    // Note: shared data means we need to be careful with free.
    // For M9, we just deep copy the data to keep it simple.
    let result = Box::into_raw(new_tensor) as i64;

    // Actually, let's deep copy to avoid shared ownership complexity in M9
    tensor.refcount -= 1;
    let result_tensor = NslTensor::from_ptr(result);

    // Device/dtype-aware copy
    if tensor.dtype == 1 {
        // f32 (GPU tensors use unified memory, so CPU can read/write)
        let data_size = (new_len as usize) * std::mem::size_of::<f32>();
        let new_data = if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            { crate::cuda::inner::alloc_managed(data_size) as *mut f32 }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        } else {
            checked_alloc(data_size) as *mut f32
        };
        unsafe {
            std::ptr::copy_nonoverlapping(tensor.data_f32(), new_data, new_len as usize);
        }
        result_tensor.data = new_data as *mut c_void;
    } else {
        let data_size = (new_len as usize) * std::mem::size_of::<f64>();
        let new_data = if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            { crate::cuda::inner::alloc_managed(data_size) as *mut f64 }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        } else {
            checked_alloc(data_size) as *mut f64
        };
        unsafe {
            std::ptr::copy_nonoverlapping(tensor.data_f64(), new_data, new_len as usize);
        }
        result_tensor.data = new_data as *mut c_void;
    }

    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Reshape, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }

    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_transpose(tensor_ptr: i64, dim0: i64, dim1: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);

    // Support negative dimension indices (e.g., -1 = last dim, -2 = second to last)
    let d0 = if dim0 < 0 { dim0 + tensor.ndim } else { dim0 };
    let d1 = if dim1 < 0 { dim1 + tensor.ndim } else { dim1 };

    if d0 < 0 || d0 >= tensor.ndim || d1 < 0 || d1 >= tensor.ndim {
        eprintln!(
            "nsl: transpose dimensions out of range ({}, {} for ndim {})",
            dim0, dim1, tensor.ndim
        );
        std::process::abort();
    }
    let dim0 = d0;
    let dim1 = d1;

    // Deep copy with transposed shape
    let ndim = tensor.ndim;
    let new_shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        unsafe { *new_shape.add(i) = *tensor.shape.add(i) };
    }
    // Swap dimensions
    unsafe {
        let tmp = *new_shape.add(dim0 as usize);
        *new_shape.add(dim0 as usize) = *new_shape.add(dim1 as usize);
        *new_shape.add(dim1 as usize) = tmp;
    }

    let strides = NslTensor::compute_strides(new_shape, ndim);
    let len = tensor.len;

    // Copy data with transposition
    let old_strides_arr: Vec<i64> = (0..ndim as usize)
        .map(|i| unsafe { *tensor.strides.add(i) })
        .collect();
    let new_strides_arr: Vec<i64> = (0..ndim as usize)
        .map(|i| unsafe { *strides.add(i) })
        .collect();

    // Device/dtype-aware transposed copy
    let data: *mut c_void = if tensor.dtype == 1 {
        // f32 (GPU tensors use unified memory, so CPU can read/write)
        let data = if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            { crate::cuda::inner::alloc_managed((len as usize) * std::mem::size_of::<f32>()) as *mut f32 }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        } else {
            checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32
        };
        for flat_idx in 0..len as usize {
            let mut remaining = flat_idx;
            let mut new_indices = vec![0usize; ndim as usize];
            for d in 0..ndim as usize {
                new_indices[d] = remaining / new_strides_arr[d] as usize;
                remaining %= new_strides_arr[d] as usize;
            }
            let mut old_indices = new_indices.clone();
            old_indices.swap(dim0 as usize, dim1 as usize);
            let mut old_offset = 0usize;
            for d in 0..ndim as usize {
                old_offset += old_indices[d] * old_strides_arr[d] as usize;
            }
            unsafe { *data.add(flat_idx) = *tensor.data_f32().add(old_offset) };
        }
        data as *mut c_void
    } else {
        let data = if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            { crate::cuda::inner::alloc_managed((len as usize) * std::mem::size_of::<f64>()) as *mut f64 }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        } else {
            checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64
        };
        for flat_idx in 0..len as usize {
            let mut remaining = flat_idx;
            let mut new_indices = vec![0usize; ndim as usize];
            for d in 0..ndim as usize {
                new_indices[d] = remaining / new_strides_arr[d] as usize;
                remaining %= new_strides_arr[d] as usize;
            }
            let mut old_indices = new_indices.clone();
            old_indices.swap(dim0 as usize, dim1 as usize);
            let mut old_offset = 0usize;
            for d in 0..ndim as usize {
                old_offset += old_indices[d] * old_strides_arr[d] as usize;
            }
            unsafe { *data.add(flat_idx) = *tensor.data_f64().add(old_offset) };
        }
        data as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape: new_shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: tensor.device,
        dtype: tensor.dtype,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(result) as i64;

    // Record on tape for autodiff
    if crate::autodiff::is_recording() {
        crate::autodiff::maybe_record(crate::autodiff::TapeOp::Transpose {
            a: tensor_ptr,
            out: out_ptr,
            dim0,
            dim1,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(out_ptr);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Transpose, vec![tensor_ptr], out_ptr, shape, rt.dtype, vec![]);
    }

    out_ptr
}

// ---------------------------------------------------------------------------
// Task 1: nsl_tensor_unsqueeze
// ---------------------------------------------------------------------------

/// Insert a dimension of size 1 at position `dim`.
/// Supports negative dims: dim=-1 inserts at the end (after last existing dim).
/// Example: [3,4] with dim=0 → [1,3,4]; dim=1 → [3,1,4]; dim=-1 → [3,4,1]
#[no_mangle]
pub extern "C" fn nsl_tensor_unsqueeze(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let old_ndim = tensor.ndim;
    let new_ndim = old_ndim + 1;

    // Normalize dim: valid range is [-(old_ndim+1), old_ndim]
    // After normalization, insert position must be in [0, old_ndim]
    let insert_pos = if dim < 0 {
        dim + new_ndim // e.g., dim=-1, new_ndim=3 → insert at 2 (end)
    } else {
        dim
    };

    if insert_pos < 0 || insert_pos > old_ndim {
        eprintln!(
            "nsl: unsqueeze dim {} out of range for ndim {}",
            dim, old_ndim
        );
        std::process::abort();
    }
    let insert_pos = insert_pos as usize;

    // Build new shape: copy old shape, insert 1 at insert_pos
    let new_shape = checked_alloc((new_ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..insert_pos {
        unsafe { *new_shape.add(i) = *tensor.shape.add(i) };
    }
    unsafe { *new_shape.add(insert_pos) = 1 };
    for i in insert_pos..old_ndim as usize {
        unsafe { *new_shape.add(i + 1) = *tensor.shape.add(i) };
    }

    let strides = NslTensor::compute_strides(new_shape, new_ndim);
    let len = tensor.len;

    // Deep copy data (dtype-aware, device-aware)
    let data: *mut c_void = if tensor.dtype == 1 {
        let buf = if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            { crate::cuda::inner::alloc_managed((len as usize) * std::mem::size_of::<f32>()) as *mut f32 }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        } else {
            checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32
        };
        unsafe { std::ptr::copy_nonoverlapping(tensor.data_f32(), buf, len as usize) };
        buf as *mut c_void
    } else {
        let buf = if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            { crate::cuda::inner::alloc_managed((len as usize) * std::mem::size_of::<f64>()) as *mut f64 }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        } else {
            checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64
        };
        unsafe { std::ptr::copy_nonoverlapping(tensor.data_f64(), buf, len as usize) };
        buf as *mut c_void
    };

    let out = Box::new(NslTensor {
        data,
        shape: new_shape,
        strides,
        ndim: new_ndim,
        len,
        refcount: 1,
        device: tensor.device,
        dtype: tensor.dtype,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    if autodiff::is_recording() {
        let input_shape = unsafe {
            std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize)
        }.to_vec();
        autodiff::maybe_record(autodiff::TapeOp::Unsqueeze {
            input: tensor_ptr,
            out: out_ptr,
            input_shape,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(out_ptr);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Unsqueeze, vec![tensor_ptr], out_ptr, shape, rt.dtype, vec![]);
    }

    out_ptr
}

// ---------------------------------------------------------------------------
// Task 2: nsl_tensor_select
// ---------------------------------------------------------------------------

/// Extract a hyperplane at `index` along `dim`, removing that dimension.
/// Example: [3,4] with dim=0,index=0 → [4]; dim=1,index=2 → [3]
/// Supports negative dim and index.
#[no_mangle]
pub extern "C" fn nsl_tensor_select(tensor_ptr: i64, dim: i64, index: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;

    // Normalize dim
    let d = if dim < 0 { (tensor.ndim + dim) as usize } else { dim as usize };
    if d >= ndim {
        eprintln!("nsl: select dim {} out of range for ndim {}", dim, ndim);
        std::process::abort();
    }

    let dim_size = unsafe { *tensor.shape.add(d) };

    // Normalize index
    let idx = if index < 0 { index + dim_size } else { index };
    if idx < 0 || idx >= dim_size {
        eprintln!(
            "nsl: select index {} out of range for dim {} size {}",
            index, dim, dim_size
        );
        std::process::abort();
    }
    let idx = idx as usize;

    // Build output shape: old shape minus the selected dimension
    let out_ndim = (ndim - 1) as i64;
    let out_shape = checked_alloc((ndim - 1) * std::mem::size_of::<i64>()) as *mut i64;
    let mut out_axis = 0;
    for i in 0..ndim {
        if i != d {
            unsafe { *out_shape.add(out_axis) = *tensor.shape.add(i) };
            out_axis += 1;
        }
    }

    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let out_len = NslTensor::total_elements(out_shape, out_ndim);

    // The offset into the source tensor for the selected hyperplane
    let in_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *tensor.strides.add(i) }).collect();
    let base_offset = idx * in_strides[d] as usize;

    // For the output, gather the remaining axes
    let out_stride_vec: Vec<i64> = (0..ndim - 1).map(|i| unsafe { *out_strides.add(i) }).collect();
    // Build mapping: output axis -> input axis (skipping d)
    let axis_map: Vec<usize> = (0..ndim).filter(|&i| i != d).collect();

    let data: *mut c_void = if tensor.dtype == 1 {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for flat in 0..out_len as usize {
            let mut remaining = flat;
            let mut in_offset = base_offset;
            for (oa, &ia) in axis_map.iter().enumerate() {
                let i = remaining / out_stride_vec[oa] as usize;
                remaining %= out_stride_vec[oa] as usize;
                in_offset += i * in_strides[ia] as usize;
            }
            unsafe { *buf.add(flat) = *tensor.data_f32().add(in_offset) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for flat in 0..out_len as usize {
            let mut remaining = flat;
            let mut in_offset = base_offset;
            for (oa, &ia) in axis_map.iter().enumerate() {
                let i = remaining / out_stride_vec[oa] as usize;
                remaining %= out_stride_vec[oa] as usize;
                in_offset += i * in_strides[ia] as usize;
            }
            unsafe { *buf.add(flat) = *tensor.data_f64().add(in_offset) };
        }
        buf as *mut c_void
    };

    let out = Box::new(NslTensor {
        data,
        shape: out_shape,
        strides: out_strides,
        ndim: out_ndim,
        len: out_len,
        refcount: 1,
        device: tensor.device,
        dtype: tensor.dtype,
        owns_data: 1,
    });
    // NO tape recording — select is used internally for stack backward
    Box::into_raw(out) as i64
}

// ---------------------------------------------------------------------------
// Task 3: nsl_tensor_stack
// ---------------------------------------------------------------------------

/// Stack a list of same-shape tensors along a NEW dimension at position `dim`.
/// Example: three [4] tensors stacked at dim=0 → [3,4]; at dim=1 → [4,3]
/// Input: NslList of tensor pointers (all must have identical shape).
#[no_mangle]
pub extern "C" fn nsl_tensor_stack(list_ptr: i64, dim: i64) -> i64 {
    let list = NslList::from_ptr(list_ptr);
    let num_tensors = list.len as usize;
    assert!(num_tensors > 0, "nsl_tensor_stack: empty tensor list");

    let first = NslTensor::from_ptr(unsafe { *list.data.add(0) });
    let in_ndim = first.ndim as usize;
    let out_ndim = (in_ndim + 1) as i64;

    // Normalize dim into [0, out_ndim] range
    let insert_pos = if dim < 0 {
        (dim + out_ndim) as usize
    } else {
        dim as usize
    };
    assert!(
        insert_pos <= in_ndim,
        "nsl_tensor_stack: dim {} out of range for ndim {}",
        dim, in_ndim
    );

    // Validate all tensors have the same shape
    for t_idx in 0..num_tensors {
        let t = NslTensor::from_ptr(unsafe { *list.data.add(t_idx) });
        assert_eq!(t.ndim as usize, in_ndim, "nsl_tensor_stack: ndim mismatch");
        for axis in 0..in_ndim {
            let s1 = unsafe { *first.shape.add(axis) };
            let s2 = unsafe { *t.shape.add(axis) };
            assert_eq!(s1, s2, "nsl_tensor_stack: shape mismatch at axis {}: {} vs {}", axis, s1, s2);
        }
    }

    // Build output shape: insert num_tensors at insert_pos
    let out_shape = checked_alloc((out_ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..insert_pos {
        unsafe { *out_shape.add(i) = *first.shape.add(i) };
    }
    unsafe { *out_shape.add(insert_pos) = num_tensors as i64 };
    for i in insert_pos..in_ndim {
        unsafe { *out_shape.add(i + 1) = *first.shape.add(i) };
    }

    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let out_len = NslTensor::total_elements(out_shape, out_ndim);
    let per_tensor = first.len as usize;

    // Copy data: tensor t goes into the slice [t, :, :, ...] along insert_pos
    let data: *mut c_void = if first.dtype == 1 {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let out_stride_vec: Vec<i64> = (0..out_ndim as usize)
            .map(|i| unsafe { *out_strides.add(i) })
            .collect();
        for t_idx in 0..num_tensors {
            let t = NslTensor::from_ptr(unsafe { *list.data.add(t_idx) });
            let t_strides: Vec<i64> = (0..in_ndim).map(|i| unsafe { *t.strides.add(i) }).collect();
            for flat in 0..per_tensor {
                // Decode flat index in input tensor space
                let mut remaining = flat;
                let mut in_multi: Vec<usize> = vec![0; in_ndim];
                for axis in 0..in_ndim {
                    in_multi[axis] = remaining / t_strides[axis] as usize;
                    remaining %= t_strides[axis] as usize;
                }
                // Build output multi-index: insert t_idx at insert_pos
                let mut out_offset = t_idx * out_stride_vec[insert_pos] as usize;
                for (ia, &iv) in in_multi.iter().enumerate() {
                    let oa = if ia < insert_pos { ia } else { ia + 1 };
                    out_offset += iv * out_stride_vec[oa] as usize;
                }
                unsafe { *buf.add(out_offset) = *t.data_f32().add(flat) };
            }
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        let out_stride_vec: Vec<i64> = (0..out_ndim as usize)
            .map(|i| unsafe { *out_strides.add(i) })
            .collect();
        for t_idx in 0..num_tensors {
            let t = NslTensor::from_ptr(unsafe { *list.data.add(t_idx) });
            let t_strides: Vec<i64> = (0..in_ndim).map(|i| unsafe { *t.strides.add(i) }).collect();
            for flat in 0..per_tensor {
                let mut remaining = flat;
                let mut in_multi: Vec<usize> = vec![0; in_ndim];
                for axis in 0..in_ndim {
                    in_multi[axis] = remaining / t_strides[axis] as usize;
                    remaining %= t_strides[axis] as usize;
                }
                let mut out_offset = t_idx * out_stride_vec[insert_pos] as usize;
                for (ia, &iv) in in_multi.iter().enumerate() {
                    let oa = if ia < insert_pos { ia } else { ia + 1 };
                    out_offset += iv * out_stride_vec[oa] as usize;
                }
                unsafe { *buf.add(out_offset) = *t.data_f64().add(flat) };
            }
        }
        buf as *mut c_void
    };

    let out = Box::new(NslTensor {
        data,
        shape: out_shape,
        strides: out_strides,
        ndim: out_ndim,
        len: out_len,
        refcount: 1,
        device: first.device,
        dtype: first.dtype,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    if autodiff::is_recording() {
        let ptrs: Vec<i64> = (0..num_tensors)
            .map(|i| unsafe { *list.data.add(i) })
            .collect();
        // Bump refcount on each input for tape safety
        for &tp in &ptrs {
            let t = unsafe { &mut *(tp as *mut NslTensor) };
            t.refcount += 1;
        }
        autodiff::maybe_record(autodiff::TapeOp::Stack {
            inputs: ptrs,
            out: out_ptr,
            dim,
        });
    }

    out_ptr
}

// ---------------------------------------------------------------------------
// Task 4: nsl_tensor_expand
// ---------------------------------------------------------------------------

/// Broadcast tensor to target shape by replicating data along size-1 dimensions.
/// Right-aligns source shape with target (pads left with 1s if needed).
/// For each dim: source=1 → replicate; source==target → keep; otherwise abort.
/// Example: [1,4] expand to [3,4]; [4] expand to [3,4]
/// Input: NslList of i64 for target shape.
#[no_mangle]
pub extern "C" fn nsl_tensor_expand(tensor_ptr: i64, shape_list: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let list = NslList::from_ptr(shape_list);
    let target_ndim = list.len as usize;

    // Read target shape
    let mut target_shape: Vec<i64> = Vec::with_capacity(target_ndim);
    for i in 0..target_ndim {
        target_shape.push(unsafe { *list.data.add(i) });
    }

    // Right-align source shape (pad left with 1s)
    let src_ndim = tensor.ndim as usize;
    if src_ndim > target_ndim {
        eprintln!(
            "nsl: expand: source ndim {} > target ndim {}",
            src_ndim, target_ndim
        );
        std::process::abort();
    }
    let pad = target_ndim - src_ndim;
    let src_shape: Vec<i64> = (0..target_ndim).map(|i| {
        if i < pad { 1 } else { unsafe { *tensor.shape.add(i - pad) } }
    }).collect();

    // Validate expand rules
    for i in 0..target_ndim {
        let s = src_shape[i];
        let t = target_shape[i];
        if s != 1 && s != t {
            eprintln!(
                "nsl: expand: source dim {} has size {} which cannot expand to {}",
                i, s, t
            );
            std::process::abort();
        }
    }

    // Compute output
    let out_ndim = target_ndim as i64;
    let out_shape_ptr = checked_alloc(target_ndim * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in target_shape.iter().enumerate().take(target_ndim) {
        unsafe { *out_shape_ptr.add(i) = s };
    }
    let out_strides = NslTensor::compute_strides(out_shape_ptr, out_ndim);
    let out_len = NslTensor::total_elements(out_shape_ptr, out_ndim);

    // Source strides (right-aligned, with stride=0 for padded/broadcast dims)
    let src_strides: Vec<i64> = (0..target_ndim).map(|i| {
        if i < pad {
            0 // broadcast dimension
        } else {
            let src_i = i - pad;
            if src_shape[i] == 1 && target_shape[i] > 1 {
                0 // replicate
            } else {
                unsafe { *tensor.strides.add(src_i) }
            }
        }
    }).collect();

    let out_stride_vec: Vec<i64> = (0..target_ndim)
        .map(|i| unsafe { *out_strides.add(i) })
        .collect();

    let data: *mut c_void = if tensor.dtype == 1 {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for flat in 0..out_len as usize {
            let mut remaining = flat;
            let mut src_offset: usize = 0;
            for axis in 0..target_ndim {
                let idx = remaining / out_stride_vec[axis] as usize;
                remaining %= out_stride_vec[axis] as usize;
                src_offset += idx * src_strides[axis] as usize;
            }
            unsafe { *buf.add(flat) = *tensor.data_f32().add(src_offset) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for flat in 0..out_len as usize {
            let mut remaining = flat;
            let mut src_offset: usize = 0;
            for axis in 0..target_ndim {
                let idx = remaining / out_stride_vec[axis] as usize;
                remaining %= out_stride_vec[axis] as usize;
                src_offset += idx * src_strides[axis] as usize;
            }
            unsafe { *buf.add(flat) = *tensor.data_f64().add(src_offset) };
        }
        buf as *mut c_void
    };

    let out = Box::new(NslTensor {
        data,
        shape: out_shape_ptr,
        strides: out_strides,
        ndim: out_ndim,
        len: out_len,
        refcount: 1,
        device: tensor.device,
        dtype: tensor.dtype,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    if autodiff::is_recording() {
        let original_shape = unsafe {
            std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize)
        }.to_vec();
        autodiff::maybe_record(autodiff::TapeOp::Expand {
            input: tensor_ptr,
            out: out_ptr,
            original_shape,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(out_ptr);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Expand, vec![tensor_ptr], out_ptr, shape, rt.dtype, vec![]);
    }

    out_ptr
}

// ---------------------------------------------------------------------------
// Task 5: nsl_tensor_causal_mask
// ---------------------------------------------------------------------------

/// Create a [seq_len, seq_len] causal attention mask.
/// Lower triangle (j <= i): 0.0; upper triangle (j > i): -1e9
/// Always CPU (device=0), always f64 (dtype=0). Not recorded on tape.
#[no_mangle]
pub extern "C" fn nsl_tensor_causal_mask(seq_len: i64) -> i64 {
    let n = seq_len as usize;
    let len = seq_len * seq_len;

    let out_shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = seq_len;
        *out_shape.add(1) = seq_len;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 2);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..n {
        for j in 0..n {
            let val = if j <= i { 0.0_f64 } else { -1e9_f64 };
            unsafe { *data.add(i * n + j) = val };
        }
    }

    let out = Box::new(NslTensor {
        data: data as *mut c_void,
        shape: out_shape,
        strides: out_strides,
        ndim: 2,
        len,
        refcount: 1,
        device: 0,
        dtype: 0,
        owns_data: 1,
    });
    // NOT recorded on tape — constant, no gradient
    Box::into_raw(out) as i64
}

// === Elementwise arithmetic ===

fn tensor_elementwise_op(a_ptr: i64, b_ptr: i64, op: fn(f64, f64) -> f64) -> i64 {
    crate::cpu::tensor_elementwise_op(a_ptr, b_ptr, op)
}

/// If `b` is on a different device than `a`, transfer `b` to `a`'s device.
/// Returns `(effective_b_ptr, true)` if a transfer was made (caller must free),
/// or `(b, false)` if no transfer was needed.
fn reconcile_device(a: i64, b: i64) -> (i64, bool) {
    let ta = unsafe { &*(a as *const NslTensor) };
    let tb = unsafe { &*(b as *const NslTensor) };
    if tb.device != ta.device {
        (nsl_tensor_to_device(b, ta.device as i64), true)
    } else {
        (b, false)
    }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_add(a: i64, b: i64) -> i64 {
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::ADD_F32_PTX, "nsl_add_f32\0");
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a_shape = get_shape_vec(NslTensor::from_ptr(a));
    let b_shape = get_shape_vec(NslTensor::from_ptr(b));
    let result = tensor_elementwise_op(a, b, |x, y| x + y);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Add { a, b, out: result, a_shape, b_shape });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Add, vec![a, b], result, shape, rt.dtype, vec![]);
    }
    if b_transferred { nsl_tensor_free(b); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sub(a: i64, b: i64) -> i64 {
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::SUB_F32_PTX, "nsl_sub_f32\0");
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a_shape = get_shape_vec(NslTensor::from_ptr(a));
    let b_shape = get_shape_vec(NslTensor::from_ptr(b));
    let result = tensor_elementwise_op(a, b, |x, y| x - y);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Sub { a, b, out: result, a_shape, b_shape });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Sub, vec![a, b], result, shape, rt.dtype, vec![]);
    }
    if b_transferred { nsl_tensor_free(b); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_mul(a: i64, b: i64) -> i64 {
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::MUL_F32_PTX, "nsl_mul_f32\0");
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a_shape = get_shape_vec(NslTensor::from_ptr(a));
    let b_shape = get_shape_vec(NslTensor::from_ptr(b));
    let result = tensor_elementwise_op(a, b, |x, y| x * y);
    if autodiff::is_recording() {
        NslTensor::from_ptr(a).refcount += 1;
        NslTensor::from_ptr(b).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Mul {
            a,
            b,
            out: result,
            saved_a: a,
            saved_b: b,
            a_shape,
            b_shape,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Mul, vec![a, b], result, shape, rt.dtype, vec![]);
    }
    if b_transferred { nsl_tensor_free(b); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_div(a: i64, b: i64) -> i64 {
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::DIV_F32_PTX, "nsl_div_f32\0");
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a_shape = get_shape_vec(NslTensor::from_ptr(a));
    let b_shape = get_shape_vec(NslTensor::from_ptr(b));
    let result = tensor_elementwise_op(a, b, |x, y| x / y);
    if autodiff::is_recording() {
        NslTensor::from_ptr(a).refcount += 1;
        NslTensor::from_ptr(b).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Div {
            a,
            b,
            out: result,
            saved_a: a,
            saved_b: b,
            a_shape,
            b_shape,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Div, vec![a, b], result, shape, rt.dtype, vec![]);
    }
    if b_transferred { nsl_tensor_free(b); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_neg(a_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(a_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(a_ptr, crate::cuda::kernels::NEG_F32_PTX, "nsl_neg_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(a_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = -(*a.data_f32().add(i)) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = -(*a.data_f64().add(i)) };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Neg { a: a_ptr, out: result });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Neg, vec![a_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

// === Scalar-tensor ops ===

#[no_mangle]
pub extern "C" fn nsl_tensor_add_scalar(a_ptr: i64, s: f64) -> i64 {
    {
        let ta = unsafe { &*(a_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_scalar_op(a_ptr, s as f32, crate::cuda::kernels::ADD_SCALAR_F32_PTX, "nsl_add_scalar_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(a_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = *a.data_f32().add(i) + (s as f32) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = *a.data_f64().add(i) + s };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::AddScalar { a: a_ptr, out: result });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_mul_scalar(a_ptr: i64, s: f64) -> i64 {
    {
        let ta = unsafe { &*(a_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_scalar_op(a_ptr, s as f32, crate::cuda::kernels::MUL_SCALAR_F32_PTX, "nsl_mul_scalar_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(a_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = *a.data_f32().add(i) * (s as f32) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = *a.data_f64().add(i) * s };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::MulScalar {
            a: a_ptr,
            scalar: s,
            out: result,
        });
    }
    result
}

// === Matrix multiply ===

#[no_mangle]
pub extern "C" fn nsl_tensor_matmul(a_ptr: i64, b_ptr: i64) -> i64 {
    let (b_ptr, b_transferred) = reconcile_device(a_ptr, b_ptr);
    // GPU dispatch
    {
        let a = unsafe { &*(a_ptr as *const NslTensor) };
        if a.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let result = crate::cuda::gpu_matmul_f32(a_ptr, b_ptr);
                if b_transferred { nsl_tensor_free(b_ptr); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }

    let a = NslTensor::from_ptr(a_ptr);
    let b = NslTensor::from_ptr(b_ptr);

    if a.ndim < 2 || b.ndim < 2 {
        eprintln!(
            "nsl: matmul requires at least 2D tensors (got {}D and {}D)",
            a.ndim, b.ndim
        );
        std::process::abort();
    }

    let a_shape = get_shape_vec(a);
    let b_shape = get_shape_vec(b);
    let a_nd = a.ndim as usize;
    let b_nd = b.ndim as usize;

    let m = a_shape[a_nd - 2];
    let k = a_shape[a_nd - 1];
    let k2 = b_shape[b_nd - 2];
    let n = b_shape[b_nd - 1];

    if k != k2 {
        eprintln!(
            "nsl: matmul inner dimension mismatch ({}x{} @ {}x{})",
            m, k, k2, n
        );
        std::process::abort();
    }

    // Broadcast batch dimensions (all dims before last two)
    let a_batch: Vec<i64> = a_shape[..a_nd - 2].to_vec();
    let b_batch: Vec<i64> = b_shape[..b_nd - 2].to_vec();

    // Compute broadcast batch shape
    let max_batch_nd = a_batch.len().max(b_batch.len());
    let mut out_batch: Vec<i64> = Vec::with_capacity(max_batch_nd);
    for i in 0..max_batch_nd {
        let a_dim = if i < max_batch_nd - a_batch.len() { 1 } else { a_batch[i - (max_batch_nd - a_batch.len())] };
        let b_dim = if i < max_batch_nd - b_batch.len() { 1 } else { b_batch[i - (max_batch_nd - b_batch.len())] };
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            eprintln!("nsl: matmul batch dimension mismatch at dim {}: {} vs {}", i, a_dim, b_dim);
            std::process::abort();
        }
        out_batch.push(a_dim.max(b_dim));
    }

    // Build output shape: batch_dims + [m, n]
    let out_nd = out_batch.len() + 2;
    let mut out_shape_vec: Vec<i64> = out_batch.clone();
    out_shape_vec.push(m);
    out_shape_vec.push(n);

    let total_batch: i64 = out_batch.iter().product::<i64>().max(1);
    let len = total_batch * m * n;

    let shape = checked_alloc(out_nd * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in out_shape_vec.iter().enumerate() {
        unsafe { *shape.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape, out_nd as i64);

    let a_mat_stride = (m * k) as usize;
    let b_mat_stride = (k2 * n) as usize;
    let out_mat_stride = (m * n) as usize;

    // Precompute per-dimension strides for a and b batch dims (for broadcast mapping)
    let mut a_batch_strides: Vec<usize> = vec![0; max_batch_nd];
    let mut b_batch_strides: Vec<usize> = vec![0; max_batch_nd];
    {
        let mut a_s = 1usize;
        let mut b_s = 1usize;
        for i in (0..max_batch_nd).rev() {
            let a_offset = max_batch_nd - a_batch.len();
            let b_offset = max_batch_nd - b_batch.len();
            let a_d = if i < a_offset { 1 } else { a_batch[i - a_offset] as usize };
            let b_d = if i < b_offset { 1 } else { b_batch[i - b_offset] as usize };
            a_batch_strides[i] = a_s;
            b_batch_strides[i] = b_s;
            a_s *= a_d;
            b_s *= b_d;
        }
    }

    // Dispatch based on dtype (use f32 if either input is f32)
    let out_dtype: u16 = if a.dtype == 1 || b.dtype == 1 { 1 } else { 0 };
    let elem_size = if out_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let raw_data = checked_alloc_zeroed((len as usize) * elem_size);

    for batch_idx in 0..total_batch as usize {
        // Decompose flat batch_idx into per-dimension coordinates, then map to a/b
        let mut a_batch_idx = 0usize;
        let mut b_batch_idx = 0usize;
        let mut remaining = batch_idx;
        for i in 0..max_batch_nd {
            let next_stride = if i + 1 < max_batch_nd {
                out_batch[i + 1..].iter().product::<i64>() as usize
            } else {
                1
            };
            let coord = remaining / next_stride;
            remaining %= next_stride;

            let a_offset = max_batch_nd - a_batch.len();
            let b_offset = max_batch_nd - b_batch.len();
            let a_d = if i < a_offset { 1 } else { a_batch[i - a_offset] as usize };
            let b_d = if i < b_offset { 1 } else { b_batch[i - b_offset] as usize };
            // Clamp coordinate to broadcast dim (size-1 dims broadcast to 0)
            a_batch_idx += coord.min(a_d - 1) * a_batch_strides[i];
            b_batch_idx += coord.min(b_d - 1) * b_batch_strides[i];
        }

        let a_base = a_batch_idx * a_mat_stride;
        let b_base = b_batch_idx * b_mat_stride;
        let out_base = batch_idx * out_mat_stride;

        // 2D matmul for this batch element
        if out_dtype == 1 {
            let data = raw_data as *mut f32;
            let read_a = |idx: usize| -> f32 {
                if a.dtype == 1 { unsafe { *a.data_f32().add(idx) } } else { unsafe { *a.data_f64().add(idx) as f32 } }
            };
            let read_b = |idx: usize| -> f32 {
                if b.dtype == 1 { unsafe { *b.data_f32().add(idx) } } else { unsafe { *b.data_f64().add(idx) as f32 } }
            };
            for i in 0..m as usize {
                for j in 0..k as usize {
                    let a_val = read_a(a_base + i * k as usize + j);
                    for l in 0..n as usize {
                        let b_val = read_b(b_base + j * n as usize + l);
                        unsafe { *data.add(out_base + i * n as usize + l) += a_val * b_val; }
                    }
                }
            }
        } else {
            let data = raw_data as *mut f64;
            for i in 0..m as usize {
                for j in 0..k as usize {
                    let a_val = unsafe { *a.data_f64().add(a_base + i * k as usize + j) };
                    for l in 0..n as usize {
                        let b_val = unsafe { *b.data_f64().add(b_base + j * n as usize + l) };
                        unsafe { *data.add(out_base + i * n as usize + l) += a_val * b_val; }
                    }
                }
            }
        }
    }

    let result = Box::new(NslTensor {
        data: raw_data as *mut c_void,
        shape,
        strides,
        ndim: out_nd as i64,
        len,
        refcount: 1,
        device: 0,
        dtype: out_dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(a_ptr).refcount += 1;
        NslTensor::from_ptr(b_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::MatMul {
            a: a_ptr,
            b: b_ptr,
            out: result,
            saved_a: a_ptr,
            saved_b: b_ptr,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::MatMul, vec![a_ptr, b_ptr], result, shape, rt.dtype, vec![]);
    }
    if b_transferred { nsl_tensor_free(b_ptr); }
    result
}

// === Reductions ===

/// Helper: get the shape of a tensor as a Vec<i64>.
fn get_shape_vec(tensor: &NslTensor) -> Vec<i64> {
    crate::cpu::get_shape_vec(tensor)
}

/// Helper: get the strides of a tensor as a Vec<usize>.
fn get_strides_vec(tensor: &NslTensor) -> Vec<usize> {
    crate::cpu::get_strides_vec(tensor)
}

/// Helper: create a tensor with a given shape (Rust slice).
#[allow(dead_code)]
fn create_tensor_with_shape_rs(shape: &[i64]) -> i64 {
    crate::cpu::create_tensor_with_shape_rs(shape)
}

/// Global sum reduction (backward compatible wrapper).
#[no_mangle]
pub extern "C" fn nsl_tensor_sum(tensor_ptr: i64) -> i64 {
    nsl_tensor_sum_dim(tensor_ptr, -1, 0)
}

/// Global mean reduction (backward compatible wrapper).
#[no_mangle]
pub extern "C" fn nsl_tensor_mean(tensor_ptr: i64) -> i64 {
    nsl_tensor_mean_dim(tensor_ptr, -1, 0)
}

/// Sum reduction along a dimension (dim=-1 means global).
#[no_mangle]
pub extern "C" fn nsl_tensor_sum_dim(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let input_shape = get_shape_vec(tensor);
    let keepdim_bool = keepdim != 0;

    if dim == -1 {
        // Global reduction
        let total = if tensor.dtype == 1 {
            let mut s = 0.0_f32;
            for i in 0..tensor.len as usize {
                s += unsafe { *tensor.data_f32().add(i) };
            }
            s as f64
        } else {
            let mut s = 0.0_f64;
            for i in 0..tensor.len as usize {
                s += unsafe { *tensor.data_f64().add(i) };
            }
            s
        };
        let result = create_scalar_tensor_dtype(total, tensor.dtype);
        if autodiff::is_recording() {
            autodiff::maybe_record(autodiff::TapeOp::SumReduce {
                a: tensor_ptr,
                out: result,
                dim: -1,
                keepdim: false,
                input_shape,
            });
        }
        return result;
    }

    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
    if d >= ndim {
        eprintln!("nsl: sum_dim dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let in_strides = get_strides_vec(tensor);

    // Compute output shape
    let out_shape: Vec<i64> = if keepdim_bool {
        input_shape.iter().enumerate()
            .map(|(i, &s)| if i == d { 1 } else { s })
            .collect()
    } else {
        input_shape.iter().enumerate()
            .filter(|&(i, _)| i != d)
            .map(|(_, &s)| s)
            .collect()
    };

    let result_ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&out_shape, tensor.dtype);
    let result = NslTensor::from_ptr(result_ptr);
    let out_strides = get_strides_vec(result);

    // Iterate all elements of input, accumulate into output
    let total_in = tensor.len as usize;
    for flat_in in 0..total_in {
        // Decompose flat_in into multi-index using input strides
        let mut remaining = flat_in;
        let mut indices = vec![0usize; ndim];
        for dd in 0..ndim {
            indices[dd] = remaining / in_strides[dd];
            remaining %= in_strides[dd];
        }

        // Compute output flat index (skip or collapse the reduced dim)
        let mut out_flat = 0usize;
        let mut oi = 0usize;
        for (dd, &idx) in indices.iter().enumerate().take(ndim) {
            if dd == d {
                if keepdim_bool {
                    // dim is kept as size 1, index=0
                    oi += 1;
                }
                continue;
            }
            out_flat += idx * out_strides[oi];
            oi += 1;
        }

        if tensor.dtype == 1 {
            let val = unsafe { *tensor.data_f32().add(flat_in) };
            unsafe { *result.data_f32().add(out_flat) += val };
        } else {
            let val = unsafe { *tensor.data_f64().add(flat_in) };
            unsafe { *result.data_f64().add(out_flat) += val };
        }
    }

    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::SumReduce {
            a: tensor_ptr,
            out: result_ptr,
            dim,
            keepdim: keepdim_bool,
            input_shape,
        });
    }
    result_ptr
}

/// Mean reduction along a dimension (dim=-1 means global).
#[no_mangle]
pub extern "C" fn nsl_tensor_mean_dim(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let input_shape = get_shape_vec(tensor);
    let keepdim_bool = keepdim != 0;

    if dim == -1 {
        // Global reduction
        if tensor.len == 0 {
            return create_scalar_tensor_dtype(0.0, tensor.dtype);
        }
        let num_elements = tensor.len;
        let total = if tensor.dtype == 1 {
            let mut s = 0.0_f32;
            for i in 0..num_elements as usize {
                s += unsafe { *tensor.data_f32().add(i) };
            }
            (s / num_elements as f32) as f64
        } else {
            let mut s = 0.0_f64;
            for i in 0..num_elements as usize {
                s += unsafe { *tensor.data_f64().add(i) };
            }
            s / num_elements as f64
        };
        let result = create_scalar_tensor_dtype(total, tensor.dtype);
        if autodiff::is_recording() {
            autodiff::maybe_record(autodiff::TapeOp::MeanReduce {
                a: tensor_ptr,
                out: result,
                dim: -1,
                keepdim: false,
                num_elements,
                input_shape,
            });
        }
        return result;
    }

    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
    if d >= ndim {
        eprintln!("nsl: mean_dim dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let dim_size = input_shape[d];

    // Sum first, then divide
    let sum_ptr = nsl_tensor_sum_dim(tensor_ptr, dim, keepdim);

    // Remove the SumReduce tape entry that sum_dim just recorded (we want MeanReduce instead)
    if autodiff::is_recording() {
        crate::autodiff::pop_last_op();
    }

    // Divide by dim_size
    let result = NslTensor::from_ptr(sum_ptr);
    if result.dtype == 1 {
        for i in 0..result.len as usize {
            unsafe { *result.data_f32().add(i) /= dim_size as f32 };
        }
    } else {
        for i in 0..result.len as usize {
            unsafe { *result.data_f64().add(i) /= dim_size as f64 };
        }
    }

    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::MeanReduce {
            a: tensor_ptr,
            out: sum_ptr,
            dim,
            keepdim: keepdim_bool,
            num_elements: dim_size,
            input_shape,
        });
    }
    sum_ptr
}

/// Reduce max along a dimension.
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_max(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let input_shape = get_shape_vec(tensor);
    let keepdim_bool = keepdim != 0;
    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };

    if d >= ndim {
        eprintln!("nsl: reduce_max dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let in_strides = get_strides_vec(tensor);

    // Compute output shape
    let out_shape: Vec<i64> = if keepdim_bool {
        input_shape.iter().enumerate()
            .map(|(i, &s)| if i == d { 1 } else { s })
            .collect()
    } else {
        input_shape.iter().enumerate()
            .filter(|&(i, _)| i != d)
            .map(|(_, &s)| s)
            .collect()
    };

    let result_ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&out_shape, tensor.dtype);
    let result = NslTensor::from_ptr(result_ptr);
    let out_strides = get_strides_vec(result);
    let out_total = result.len as usize;

    // Initialize with -inf
    if tensor.dtype == 1 {
        for i in 0..out_total {
            unsafe { *result.data_f32().add(i) = f32::NEG_INFINITY };
        }
    } else {
        for i in 0..out_total {
            unsafe { *result.data_f64().add(i) = f64::NEG_INFINITY };
        }
    }

    // Track argmax per output position
    let mut argmax = vec![0usize; out_total];

    // Iterate all elements of input
    let total_in = tensor.len as usize;
    for flat_in in 0..total_in {
        let mut remaining = flat_in;
        let mut indices = vec![0usize; ndim];
        for dd in 0..ndim {
            indices[dd] = remaining / in_strides[dd];
            remaining %= in_strides[dd];
        }

        // Compute output flat index
        let mut out_flat = 0usize;
        let mut oi = 0usize;
        for (dd, &idx) in indices.iter().enumerate().take(ndim) {
            if dd == d {
                if keepdim_bool {
                    oi += 1;
                }
                continue;
            }
            out_flat += idx * out_strides[oi];
            oi += 1;
        }

        if tensor.dtype == 1 {
            let val = unsafe { *tensor.data_f32().add(flat_in) };
            let cur = unsafe { *result.data_f32().add(out_flat) };
            if val > cur {
                unsafe { *result.data_f32().add(out_flat) = val };
                argmax[out_flat] = indices[d];
            }
        } else {
            let val = unsafe { *tensor.data_f64().add(flat_in) };
            let cur = unsafe { *result.data_f64().add(out_flat) };
            if val > cur {
                unsafe { *result.data_f64().add(out_flat) = val };
                argmax[out_flat] = indices[d];
            }
        }
    }

    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::ReduceMax {
            a: tensor_ptr,
            out: result_ptr,
            dim,
            keepdim: keepdim_bool,
            saved_argmax: argmax,
            input_shape,
        });
    }
    result_ptr
}

/// Gather along a dimension: output[b] = tensor[b, indices[b]] (for dim=1, 2D input).
#[no_mangle]
pub extern "C" fn nsl_tensor_gather(tensor_ptr: i64, dim: i64, indices_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let indices = NslTensor::from_ptr(indices_ptr);
    let input_shape = get_shape_vec(tensor);
    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };

    if d >= ndim {
        eprintln!("nsl: gather dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let num_indices = indices.len as usize;
    let gather_dim_size = input_shape[d] as usize;

    // Output shape: input shape with d_dim replaced by num_indices
    // For [batch, classes] with dim=1 and indices [batch] -> output [batch]
    // For [batch, seq, vocab] with dim=2 and indices [batch*seq] -> output [batch, seq]
    let out_shape: Vec<i64> = input_shape.iter().enumerate()
        .filter(|&(i, _)| i != d)
        .map(|(_, &s)| s)
        .collect();

    // Compute the number of "outer" elements (product of dims before d)
    // and "inner" elements (product of dims after d)
    let outer: usize = input_shape[..d].iter().map(|&s| s as usize).product::<usize>().max(1);
    let inner: usize = input_shape[d+1..].iter().map(|&s| s as usize).product::<usize>().max(1);

    // indices must match outer dimension count
    if num_indices != outer {
        eprintln!(
            "nsl: gather dim={} requires indices length ({}) == outer size ({})",
            d, num_indices, outer
        );
        std::process::abort();
    }

    let result_ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&out_shape, tensor.dtype);
    let result = NslTensor::from_ptr(result_ptr);

    // Helper: read index from indices tensor (always treat as integer index)
    let read_idx = |i: usize| -> usize {
        if indices.dtype == 1 {
            unsafe { *indices.data_f32().add(i) as usize }
        } else {
            unsafe { *indices.data_f64().add(i) as usize }
        }
    };

    // General N-D gather along dimension d:
    // For each outer position o and each inner position k,
    //   output[o * inner + k] = input[o * (gather_dim_size * inner) + idx * inner + k]
    for o in 0..outer {
        let idx = read_idx(o);
        if idx >= gather_dim_size {
            eprintln!(
                "nsl: gather index {} out of bounds for dim {} with size {}",
                idx, d, gather_dim_size
            );
            std::process::abort();
        }
        let in_base = o * gather_dim_size * inner + idx * inner;
        let out_base = o * inner;
        for k in 0..inner {
            if tensor.dtype == 1 {
                let val = unsafe { *tensor.data_f32().add(in_base + k) };
                unsafe { *result.data_f32().add(out_base + k) = val };
            } else {
                let val = unsafe { *tensor.data_f64().add(in_base + k) };
                unsafe { *result.data_f64().add(out_base + k) = val };
            }
        }
    }

    if autodiff::is_recording() {
        // Save indices for backward
        NslTensor::from_ptr(indices_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Gather {
            a: tensor_ptr,
            out: result_ptr,
            dim,
            indices_ptr,
            input_shape,
        });
    }
    result_ptr
}

// === Element-wise math ops ===

#[no_mangle]
pub extern "C" fn nsl_tensor_exp(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::EXP_F32_PTX, "nsl_exp_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f32().add(i)).exp() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f64().add(i)).exp() };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Exp {
            a: tensor_ptr,
            out: result,
            saved_out: result,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Exp, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_log(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::LOG_F32_PTX, "nsl_log_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f32().add(i)).ln() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f64().add(i)).ln() };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Log {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Log, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sqrt(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SQRT_F32_PTX, "nsl_sqrt_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f32().add(i)).sqrt() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f64().add(i)).sqrt() };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Sqrt {
            a: tensor_ptr,
            out: result,
            saved_out: result,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Sqrt, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_abs(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::ABS_F32_PTX, "nsl_abs_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f32().add(i)).abs() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f64().add(i)).abs() };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Abs {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sign(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SIGN_F32_PTX, "nsl_sign_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f32().add(i) };
            unsafe {
                *buf.add(i) = if val > 0.0 { 1.0f32 } else if val < 0.0 { -1.0f32 } else { 0.0f32 };
            }
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f64().add(i) };
            unsafe {
                *buf.add(i) = if val > 0.0 { 1.0 } else if val < 0.0 { -1.0 } else { 0.0 };
            }
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    // sign is non-differentiable — no tape recording
    Box::into_raw(result) as i64
}

#[no_mangle]
pub extern "C" fn nsl_tensor_clamp(tensor_ptr: i64, min_val: f64, max_val: f64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let (mn, mx) = (min_val as f32, max_val as f32);
        for i in 0..len as usize {
            let val = unsafe { *a.data_f32().add(i) };
            unsafe { *buf.add(i) = val.clamp(mn, mx) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f64().add(i) };
            unsafe { *buf.add(i) = val.clamp(min_val, max_val) };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Clamp {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
            min_val,
            max_val,
        });
    }
    result
}

/// Helper for clamp backward: produces grad * mask where mask is 1 where input is strictly
/// between min_val and max_val, and 0 otherwise.
pub(crate) fn nsl_tensor_clamp_backward(
    grad_ptr: i64,
    input_ptr: i64,
    min_val: f64,
    max_val: f64,
) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    #[cfg(feature = "cuda")]
    if grad.device > 0 {
        return crate::cuda::gpu_clamp_backward(grad_ptr, input_ptr, min_val as f32, max_val as f32);
    }
    let input = NslTensor::from_ptr(input_ptr);
    let len = input.len;
    let ndim = input.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if input.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let (mn, mx) = (min_val as f32, max_val as f32);
        for i in 0..len as usize {
            let val = unsafe { *input.data_f32().add(i) };
            let g_val = unsafe { *grad.data_f32().add(i) };
            unsafe { *buf.add(i) = if val > mn && val < mx { g_val } else { 0.0 } };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let val = unsafe { *input.data_f64().add(i) };
            let g_val = unsafe { *grad.data_f64().add(i) };
            unsafe { *buf.add(i) = if val > min_val && val < max_val { g_val } else { 0.0 } };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: input.device,
        dtype: input.dtype,
        owns_data: 1,
    });
    Box::into_raw(result) as i64
}

// === Activation functions ===

#[no_mangle]
pub extern "C" fn nsl_tensor_relu(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::RELU_F32_PTX, "nsl_relu_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f32().add(i) };
            unsafe { *buf.add(i) = if val > 0.0 { val } else { 0.0 } };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f64().add(i) };
            unsafe { *buf.add(i) = if val > 0.0 { val } else { 0.0 } };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::ReLU {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Relu, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_gelu(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        for i in 0..len as usize {
            let x = unsafe { *a.data_f32().add(i) };
            let inner = c * (x + 0.044715_f32 * x * x * x);
            unsafe { *buf.add(i) = 0.5_f32 * x * (1.0_f32 + inner.tanh()) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        let c = (2.0_f64 / std::f64::consts::PI).sqrt();
        for i in 0..len as usize {
            let x = unsafe { *a.data_f64().add(i) };
            let inner = c * (x + 0.044715 * x * x * x);
            unsafe { *buf.add(i) = 0.5 * x * (1.0 + inner.tanh()) };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::GELU {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_silu(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f32().add(i) };
            let sig = 1.0_f32 / (1.0_f32 + (-x).exp());
            unsafe { *buf.add(i) = x * sig };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f64().add(i) };
            let sig = 1.0 / (1.0 + (-x).exp());
            unsafe { *buf.add(i) = x * sig };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::SiLU {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sigmoid(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SIGMOID_F32_PTX, "nsl_sigmoid_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f32().add(i) };
            unsafe { *buf.add(i) = 1.0_f32 / (1.0_f32 + (-x).exp()) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f64().add(i) };
            unsafe { *buf.add(i) = 1.0 / (1.0 + (-x).exp()) };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Sigmoid {
            a: tensor_ptr,
            out: result,
            saved_out: result,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Sigmoid, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_tanh_act(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::TANH_F32_PTX, "nsl_tanh_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f32().add(i) };
            unsafe { *buf.add(i) = x.tanh() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f64().add(i) };
            unsafe { *buf.add(i) = x.tanh() };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Tanh {
            a: tensor_ptr,
            out: result,
            saved_out: result,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Tanh, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_softmax(tensor_ptr: i64, dim: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    // Normalize dim
    let d = if dim < 0 { (ndim + dim) as usize } else { dim as usize };

    let a_shape: Vec<i64> = (0..ndim as usize).map(|i| unsafe { *a.shape.add(i) }).collect();
    let a_strides: Vec<i64> = (0..ndim as usize).map(|i| unsafe { *a.strides.add(i) }).collect();
    let dim_size = a_shape[d] as usize;

    let num_slices = (len as usize) / dim_size;

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc_zeroed((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for slice_idx in 0..num_slices {
            let mut remaining = slice_idx;
            let mut base_offset: usize = 0;
            for axis in (0..ndim as usize).rev() {
                if axis == d { continue; }
                let idx = remaining % (a_shape[axis] as usize);
                remaining /= a_shape[axis] as usize;
                base_offset += idx * (a_strides[axis] as usize);
            }
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                let val = unsafe { *a.data_f32().add(offset) };
                if val > max_val { max_val = val; }
            }
            let mut sum = 0.0_f32;
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                let e = (unsafe { *a.data_f32().add(offset) } - max_val).exp();
                unsafe { *buf.add(offset) = e };
                sum += e;
            }
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                unsafe { *buf.add(offset) /= sum };
            }
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc_zeroed((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for slice_idx in 0..num_slices {
            let mut remaining = slice_idx;
            let mut base_offset: usize = 0;
            for axis in (0..ndim as usize).rev() {
                if axis == d { continue; }
                let idx = remaining % (a_shape[axis] as usize);
                remaining /= a_shape[axis] as usize;
                base_offset += idx * (a_strides[axis] as usize);
            }
            let mut max_val = f64::NEG_INFINITY;
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                let val = unsafe { *a.data_f64().add(offset) };
                if val > max_val { max_val = val; }
            }
            let mut sum = 0.0_f64;
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                let e = (unsafe { *a.data_f64().add(offset) } - max_val).exp();
                unsafe { *buf.add(offset) = e };
                sum += e;
            }
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                unsafe { *buf.add(offset) /= sum };
            }
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device,
        dtype: a.dtype,
        owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Softmax {
            a: tensor_ptr,
            out: result,
            saved_out: result,
            dim,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(
            crate::trace::OpType::Softmax,
            vec![tensor_ptr],
            result,
            shape,
            rt.dtype,
            vec![("axis".to_string(), crate::trace::AttrValue::Int(dim))],
        );
    }
    result
}

// === Scalar extraction ===

#[no_mangle]
pub extern "C" fn nsl_tensor_item(tensor_ptr: i64) -> f64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if tensor.len != 1 {
        eprintln!(
            "nsl: .item() requires a scalar tensor (got {} elements)",
            tensor.len
        );
        std::process::abort();
    }
    match tensor.dtype {
        1 => unsafe { *tensor.data_f32() as f64 },
        _ => unsafe { *tensor.data_f64() },
    }
}

// === Display ===

#[no_mangle]
pub extern "C" fn nsl_tensor_print(tensor_ptr: i64) {
    let tensor = NslTensor::from_ptr(tensor_ptr);

    if tensor.ndim == 0 {
        if tensor.len > 0 {
            let val = match tensor.dtype {
                1 => unsafe { *tensor.data_f32() as f64 },
                _ => unsafe { *tensor.data_f64() },
            };
            print_float_value(val);
            println!();
        } else {
            println!("tensor([])");
        }
        return;
    }

    print!("tensor(");
    print_tensor_recursive(tensor.data as *const u8, tensor.dtype, tensor.shape, tensor.strides, tensor.ndim, 0);
    println!(")");
}

fn print_float_value(v: f64) {
    if v == v.floor() && v.abs() < 1e15 {
        // Print as integer-looking: "3.0" style for tensor display
        print!("{:.1}", v);
    } else {
        print!("{}", v);
    }
}

fn print_tensor_recursive(
    data: *const u8,
    dtype: u16,
    shape: *mut i64,
    strides: *mut i64,
    ndim: i64,
    dim: usize,
) {
    let size = unsafe { *shape.add(dim) } as usize;
    let stride = unsafe { *strides.add(dim) } as usize;
    let elem_size = if dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };

    print!("[");
    if dim as i64 == ndim - 1 {
        // Last dimension: print values
        for i in 0..size {
            if i > 0 { print!(", "); }
            let val = match dtype {
                1 => unsafe { *(data as *const f32).add(i * stride) as f64 },
                _ => unsafe { *(data as *const f64).add(i * stride) },
            };
            print_float_value(val);
        }
    } else {
        // Recursive: print sub-arrays
        for i in 0..size {
            if i > 0 { print!(", "); }
            let offset_data = unsafe { data.add(i * stride * elem_size) };
            print_tensor_recursive(offset_data, dtype, shape, strides, ndim, dim + 1);
        }
    }
    print!("]");
}

// === Memory ===

#[no_mangle]
pub extern "C" fn nsl_tensor_clone(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim;
    let len = tensor.len;

    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(tensor.shape, shape, ndim as usize) };

    let strides = NslTensor::compute_strides(shape, ndim);

    let elem_size = tensor.element_size();
    let data_size = (len as usize) * elem_size;
    let data = checked_alloc(data_size);
    unsafe { std::ptr::copy_nonoverlapping(tensor.data as *const u8, data, data_size) };

    let result = Box::new(NslTensor {
        data: data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
        device: tensor.device,
        dtype: tensor.dtype,
        owns_data: 1,
    });
    Box::into_raw(result) as i64
}

#[no_mangle]
pub extern "C" fn nsl_tensor_free(tensor_ptr: i64) {
    if tensor_ptr == 0 {
        return;
    }
    let tensor = NslTensor::from_ptr(tensor_ptr);
    tensor.refcount -= 1;
    if tensor.refcount <= 0 {
        let data_size = (tensor.len as usize) * tensor.element_size();
        let shape_size = (tensor.ndim as usize) * std::mem::size_of::<i64>();
        let strides_size = shape_size;

        unsafe {
            // Only free data if this tensor owns it (not borrowed/mmap)
            if tensor.owns_data != 0 {
                // GPU tensors use CUDA unified memory — must free with cuMemFree, not CPU dealloc
                if tensor.device > 0 {
                    #[cfg(feature = "cuda")]
                    {
                        crate::cuda::inner::free_managed(tensor.data);
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        // Shouldn't happen: GPU tensor without CUDA support
                        checked_free(tensor.data as *mut u8, data_size);
                    }
                } else {
                    checked_free(tensor.data as *mut u8, data_size);
                }
            }
            // Shape and strides are always CPU-allocated
            if !tensor.shape.is_null() {
                checked_free(tensor.shape as *mut u8, shape_size);
            }
            if !tensor.strides.is_null() {
                checked_free(tensor.strides as *mut u8, strides_size);
            }
            // Free the NslTensor struct itself
            drop(Box::from_raw(tensor as *mut NslTensor));
        }
    }
}

// === In-place mutation ops (NOT taped — used outside grad blocks) ===

#[no_mangle]
pub extern "C" fn nsl_tensor_copy_data(dst_ptr: i64, src_ptr: i64) {
    let dst = NslTensor::from_ptr(dst_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    assert_eq!(
        dst.len, src.len,
        "nsl_tensor_copy_data: dst len {} != src len {}",
        dst.len, src.len
    );
    assert_eq!(
        dst.dtype, src.dtype,
        "nsl_tensor_copy_data: dtype mismatch (dst={}, src={})",
        dst.dtype, src.dtype
    );
    let byte_count = (dst.len as usize) * dst.element_size();
    unsafe {
        std::ptr::copy_nonoverlapping(src.data as *const u8, dst.data as *mut u8, byte_count);
    }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_add_inplace(dst_ptr: i64, src_ptr: i64) {
    let dst = NslTensor::from_ptr(dst_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    assert_eq!(
        dst.len, src.len,
        "nsl_tensor_add_inplace: dst len {} != src len {}",
        dst.len, src.len
    );
    assert_eq!(
        dst.dtype, src.dtype,
        "nsl_tensor_add_inplace: dtype mismatch (dst={}, src={})",
        dst.dtype, src.dtype
    );
    if dst.dtype == 1 {
        for i in 0..dst.len as usize {
            unsafe { *dst.data_f32().add(i) += *src.data_f32().add(i); }
        }
    } else {
        for i in 0..dst.len as usize {
            unsafe { *dst.data_f64().add(i) += *src.data_f64().add(i); }
        }
    }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_zero_inplace(tensor_ptr: i64) {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let byte_count = (tensor.len as usize) * tensor.element_size();
    unsafe {
        std::ptr::write_bytes(tensor.data as *mut u8, 0, byte_count);
    }
}

/// Create a zeros tensor on a specific device.
/// `device` = 0 for CPU, 1+ for CUDA device ID.
#[no_mangle]
pub extern "C" fn nsl_tensor_zeros_on(shape_list: i64, device: i64) -> i64 {
    if device == 0 {
        return nsl_tensor_zeros(shape_list);
    }
    #[cfg(feature = "cuda")]
    {
        let list = NslList::from_ptr(shape_list);
        let ndim = list.len;
        let mut len: i64 = 1;
        let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
        for i in 0..ndim as usize {
            let dim = unsafe { *list.data.add(i) };
            unsafe { *shape.add(i) = dim };
            len *= dim;
        }

        // Allocate CUDA unified memory (f32). Unified memory is zero-initialized.
        let data = crate::cuda::inner::alloc_managed((len as usize) * std::mem::size_of::<f32>());

        let strides = NslTensor::compute_strides(shape, ndim);

        let tensor = Box::new(NslTensor {
            data,
            shape,
            strides,
            ndim,
            len,
            refcount: 1,
            device: device as u8,
            dtype: 1, // f32 for GPU tensors
            owns_data: 1,
        });
        Box::into_raw(tensor) as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = shape_list;
        panic!("CUDA support not compiled. Rebuild with --features cuda");
    }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_zeros_like(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape_list = nsl_list_new();
    for i in 0..tensor.ndim as usize {
        unsafe {
            nsl_list_push(shape_list, *tensor.shape.add(i));
        }
    }
    let result = nsl_tensor_zeros_on(shape_list, tensor.device as i64);
    crate::list::nsl_list_free(shape_list);
    result
}

/// Create a ones tensor with the same shape and device as the input tensor.
#[no_mangle]
pub extern "C" fn nsl_tensor_ones_like(tensor_ptr: i64) -> i64 {
    let t = NslTensor::from_ptr(tensor_ptr);
    if t.device == 0 {
        let shape_list = nsl_list_new();
        for i in 0..t.ndim as usize {
            unsafe { nsl_list_push(shape_list, *t.shape.add(i)); }
        }
        let result = nsl_tensor_ones(shape_list);
        crate::list::nsl_list_free(shape_list);
        return result;
    }
    #[cfg(feature = "cuda")]
    {
        let shape_list = nsl_list_new();
        for i in 0..t.ndim as usize {
            unsafe { nsl_list_push(shape_list, *t.shape.add(i)); }
        }
        let result = nsl_tensor_zeros_on(shape_list, t.device as i64);
        crate::list::nsl_list_free(shape_list);
        let result_t = NslTensor::from_ptr(result);
        // Fill with 1.0f32 on CPU side (unified memory allows this)
        let data = result_t.data_f32();
        for i in 0..result_t.len as usize {
            unsafe { *data.add(i) = 1.0f32; }
        }
        // Prefetch to device for optimal GPU access
        crate::cuda::inner::prefetch_to_device(
            result_t.data,
            (result_t.len as usize) * std::mem::size_of::<f32>(),
            (t.device - 1) as i32,
        );
        result
    }
    #[cfg(not(feature = "cuda"))]
    {
        panic!("CUDA support not compiled. Rebuild with --features cuda");
    }
}

// === Gradient clipping ===

#[no_mangle]
pub extern "C" fn nsl_clip_grad_norm(grad_list_ptr: i64, max_norm: f64) {
    let list = NslList::from_ptr(grad_list_ptr);
    let num_grads = list.len as usize;

    // Compute global L2 norm: sqrt(sum of squares of all elements across all tensors)
    let mut sum_sq: f64 = 0.0;
    for g in 0..num_grads {
        let tensor_ptr = unsafe { *list.data.add(g) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if tensor.dtype == 1 {
            for i in 0..tensor.len as usize {
                let val = unsafe { *tensor.data_f32().add(i) } as f64;
                sum_sq += val * val;
            }
        } else {
            for i in 0..tensor.len as usize {
                let val = unsafe { *tensor.data_f64().add(i) };
                sum_sq += val * val;
            }
        }
    }
    let norm = sum_sq.sqrt();

    // Fast path: skip scaling when gradients are within bounds
    if norm <= max_norm {
        return;
    }

    // norm > max_norm: scale all gradients by (max_norm / (norm + 1e-6))
    {
        let scale = max_norm / (norm + 1e-6);
        for g in 0..num_grads {
            let tensor_ptr = unsafe { *list.data.add(g) };
            let tensor = NslTensor::from_ptr(tensor_ptr);
            if tensor.dtype == 1 {
                let scale_f32 = scale as f32;
                for i in 0..tensor.len as usize {
                    unsafe { *tensor.data_f32().add(i) *= scale_f32; }
                }
            } else {
                for i in 0..tensor.len as usize {
                    unsafe { *tensor.data_f64().add(i) *= scale; }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor Slice
// ---------------------------------------------------------------------------

/// Slice a tensor along a dimension: extract elements [start, end) along dim.
/// Supports negative indices (e.g., -1 means last element).
/// Input shape [d0, d1, ..., d_dim, ...] -> output shape [d0, d1, ..., (end-start), ...]
#[no_mangle]
pub extern "C" fn nsl_tensor_slice(tensor_ptr: i64, dim: i64, start: i64, end: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;

    // Normalize dim
    let d = if dim < 0 { (tensor.ndim + dim) as usize } else { dim as usize };
    assert!(d < ndim, "nsl_tensor_slice: dim {dim} out of range for ndim {ndim}");

    let dim_size = unsafe { *tensor.shape.add(d) };

    // Normalize start/end with negative index support
    let s = if start < 0 { (dim_size + start).max(0) } else { start.min(dim_size) };
    let e = if end < 0 { (dim_size + end).max(0) } else { end.min(dim_size) };
    let slice_len = (e - s).max(0);

    // Build output shape
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim {
        if i == d {
            unsafe { *out_shape.add(i) = slice_len };
        } else {
            unsafe { *out_shape.add(i) = *tensor.shape.add(i) };
        }
    }
    let out_strides = NslTensor::compute_strides(out_shape, tensor.ndim);
    let out_len = NslTensor::total_elements(out_shape, tensor.ndim);

    // Copy data: iterate over all output elements, mapping back to input
    let in_strides: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *tensor.strides.add(i) })
        .collect();
    let o_strides: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *out_strides.add(i) })
        .collect();

    let data: *mut c_void = if tensor.dtype == 1 {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for flat in 0..out_len as usize {
            let mut remaining = flat;
            let mut in_offset: usize = 0;
            for axis in 0..ndim {
                let idx = remaining / o_strides[axis] as usize;
                remaining %= o_strides[axis] as usize;
                if axis == d {
                    in_offset += (idx + s as usize) * in_strides[axis] as usize;
                } else {
                    in_offset += idx * in_strides[axis] as usize;
                }
            }
            unsafe { *buf.add(flat) = *tensor.data_f32().add(in_offset) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for flat in 0..out_len as usize {
            let mut remaining = flat;
            let mut in_offset: usize = 0;
            for axis in 0..ndim {
                let idx = remaining / o_strides[axis] as usize;
                remaining %= o_strides[axis] as usize;
                if axis == d {
                    in_offset += (idx + s as usize) * in_strides[axis] as usize;
                } else {
                    in_offset += idx * in_strides[axis] as usize;
                }
            }
            unsafe { *buf.add(flat) = *tensor.data_f64().add(in_offset) };
        }
        buf as *mut c_void
    };

    // Save input shape for backward
    let input_shape: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *tensor.shape.add(i) })
        .collect();

    let out = Box::new(NslTensor {
        data,
        shape: out_shape,
        strides: out_strides,
        ndim: tensor.ndim,
        len: out_len,
        refcount: 1,
        device: tensor.device,
        dtype: tensor.dtype,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    // Record on tape for autodiff
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Slice {
            a: tensor_ptr,
            out: out_ptr,
            dim,
            start: s,
            input_shape,
        });
    }

    out_ptr
}

// ---------------------------------------------------------------------------
// Tensor Cat
// ---------------------------------------------------------------------------

/// Concatenate a list of tensors along a dimension.
/// All tensors must have same shape except on the cat dimension.
/// Returns a new tensor with the cat dimension being the sum of all input dim sizes.
#[no_mangle]
pub extern "C" fn nsl_tensor_cat(tensor_list: i64, dim: i64) -> i64 {
    let list = NslList::from_ptr(tensor_list);
    let num_tensors = list.len as usize;
    assert!(num_tensors > 0, "nsl_tensor_cat: empty tensor list");

    let first = NslTensor::from_ptr(unsafe { *list.data.add(0) });
    let ndim = first.ndim as usize;
    let d = if dim < 0 { (first.ndim + dim) as usize } else { dim as usize };
    assert!(d < ndim, "nsl_tensor_cat: dim {dim} out of range for ndim {ndim}");

    // Collect split sizes and validate shapes
    let mut split_sizes: Vec<i64> = Vec::with_capacity(num_tensors);
    let mut total_cat_dim: i64 = 0;

    for t_idx in 0..num_tensors {
        let t = NslTensor::from_ptr(unsafe { *list.data.add(t_idx) });
        assert_eq!(t.ndim as usize, ndim, "nsl_tensor_cat: ndim mismatch");
        let cat_size = unsafe { *t.shape.add(d) };
        split_sizes.push(cat_size);
        total_cat_dim += cat_size;
        // Validate non-cat dimensions match
        for axis in 0..ndim {
            if axis != d {
                let s1 = unsafe { *first.shape.add(axis) };
                let s2 = unsafe { *t.shape.add(axis) };
                assert_eq!(s1, s2, "nsl_tensor_cat: shape mismatch at dim {axis}: {s1} vs {s2}");
            }
        }
    }

    // Build output shape
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim {
        if i == d {
            unsafe { *out_shape.add(i) = total_cat_dim };
        } else {
            unsafe { *out_shape.add(i) = *first.shape.add(i) };
        }
    }
    let out_ndim = first.ndim;
    let out_dtype = first.dtype;
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let out_len = NslTensor::total_elements(out_shape, out_ndim);

    let o_strides: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *out_strides.add(i) })
        .collect();

    let data: *mut c_void = if out_dtype == 1 {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let mut cat_offset: usize = 0;
        for (t_idx, &sz) in split_sizes.iter().enumerate().take(num_tensors) {
            let t = NslTensor::from_ptr(unsafe { *list.data.add(t_idx) });
            let t_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *t.strides.add(i) }).collect();
            for flat in 0..t.len as usize {
                let mut remaining = flat;
                let mut out_offset: usize = 0;
                for axis in 0..ndim {
                    let idx = remaining / t_strides[axis] as usize;
                    remaining %= t_strides[axis] as usize;
                    if axis == d {
                        out_offset += (idx + cat_offset) * o_strides[axis] as usize;
                    } else {
                        out_offset += idx * o_strides[axis] as usize;
                    }
                }
                unsafe { *buf.add(out_offset) = *t.data_f32().add(flat) };
            }
            cat_offset += sz as usize;
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        let mut cat_offset: usize = 0;
        for (t_idx, &sz) in split_sizes.iter().enumerate().take(num_tensors) {
            let t = NslTensor::from_ptr(unsafe { *list.data.add(t_idx) });
            let t_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *t.strides.add(i) }).collect();
            for flat in 0..t.len as usize {
                let mut remaining = flat;
                let mut out_offset: usize = 0;
                for axis in 0..ndim {
                    let idx = remaining / t_strides[axis] as usize;
                    remaining %= t_strides[axis] as usize;
                    if axis == d {
                        out_offset += (idx + cat_offset) * o_strides[axis] as usize;
                    } else {
                        out_offset += idx * o_strides[axis] as usize;
                    }
                }
                unsafe { *buf.add(out_offset) = *t.data_f64().add(flat) };
            }
            cat_offset += sz as usize;
        }
        buf as *mut c_void
    };

    let out = Box::new(NslTensor {
        data,
        shape: out_shape,
        strides: out_strides,
        ndim: out_ndim,
        len: out_len,
        refcount: 1,
        device: first.device,
        dtype: out_dtype,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    // Collect input ptrs for tape
    let input_ptrs: Vec<i64> = (0..num_tensors)
        .map(|i| unsafe { *list.data.add(i) })
        .collect();

    if autodiff::is_recording() {
        // Bump refcount on each input for tape safety (prevent use-after-free)
        for &tp in &input_ptrs {
            let t = unsafe { &mut *(tp as *mut NslTensor) };
            t.refcount += 1;
        }
    }

    #[cfg(feature = "interop")]
    let trace_input_ptrs = input_ptrs.clone();

    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Cat {
            inputs: input_ptrs,
            out: out_ptr,
            dim,
            split_sizes,
        });
    }

    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(out_ptr);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Concat, trace_input_ptrs, out_ptr, shape, rt.dtype, vec![]);
    }

    out_ptr
}

// === Embedding lookup ===

/// Look up rows from an embedding weight matrix by integer indices.
/// weight: [vocab_size, embed_dim], indices: [seq_len] (f64 values cast to integer indices)
/// output: [seq_len, embed_dim]
#[no_mangle]
pub extern "C" fn nsl_tensor_embedding_lookup(weight_ptr: i64, indices_ptr: i64) -> i64 {
    let weight = NslTensor::from_ptr(weight_ptr);
    let indices = NslTensor::from_ptr(indices_ptr);

    if weight.ndim != 2 {
        eprintln!(
            "nsl: embedding_lookup requires 2D weight tensor (got {}D)",
            weight.ndim
        );
        std::process::abort();
    }
    if indices.ndim != 1 {
        eprintln!(
            "nsl: embedding_lookup requires 1D indices tensor (got {}D)",
            indices.ndim
        );
        std::process::abort();
    }

    let vocab_size = unsafe { *weight.shape.add(0) } as usize;
    let embed_dim = unsafe { *weight.shape.add(1) } as usize;
    let seq_len = unsafe { *indices.shape.add(0) } as usize;

    // Output shape: [seq_len, embed_dim]
    let out_ndim: i64 = 2;
    let out_len = (seq_len * embed_dim) as i64;
    let out_shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = seq_len as i64;
        *out_shape.add(1) = embed_dim as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let out_dtype = weight.dtype;
    let elem_size = if out_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc((out_len as usize) * elem_size);

    // Helper to read index from indices tensor
    let read_idx = |i: usize| -> i64 {
        if indices.dtype == 1 {
            unsafe { *indices.data_f32().add(i) as i64 }
        } else {
            unsafe { *indices.data_f64().add(i) as i64 }
        }
    };

    // For each index, copy the corresponding row from weight
    for i in 0..seq_len {
        let raw_idx = read_idx(i);
        if raw_idx < 0 || raw_idx >= vocab_size as i64 {
            eprintln!(
                "nsl: embedding_lookup index {} out of bounds for vocab_size {}",
                raw_idx, vocab_size
            );
            std::process::abort();
        }
        let idx = raw_idx as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                (weight.data as *const u8).add((idx * embed_dim) * elem_size),
                out_data_raw.add(i * embed_dim * elem_size),
                embed_dim * elem_size,
            );
        }
    }

    let out = Box::new(NslTensor {
        data: out_data_raw as *mut c_void,
        shape: out_shape,
        strides: out_strides,
        ndim: out_ndim,
        len: out_len,
        refcount: 1,
        device: weight.device,
        dtype: out_dtype,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(weight_ptr).refcount += 1;
        NslTensor::from_ptr(indices_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::EmbeddingLookup {
            weight: weight_ptr,
            indices: indices_ptr,
            out: out_ptr,
            saved_weight: weight_ptr,
            saved_indices: indices_ptr,
        });
    }

    out_ptr
}

// === LayerNorm & RMSNorm (fused runtime primitives) ===

/// LayerNorm: normalize over last dimension, then scale and shift.
/// input: [*, N], weight: [N], bias: [N]
/// For each "row" (slice along last dim):
///   mean = mean(row)
///   var = mean((row - mean)^2)
///   normalized = (row - mean) / sqrt(var + eps)
///   output = normalized * weight + bias
/// Saves {input, mean, inv_std, weight} for backward.
#[no_mangle]
pub extern "C" fn nsl_tensor_layernorm(
    input_ptr: i64,
    weight_ptr: i64,
    bias_ptr: i64,
    eps: f64,
) -> i64 {
    // If input is not contiguous (e.g. transposed), make a contiguous copy first
    let input_ref = NslTensor::from_ptr(input_ptr);
    let need_contig = !input_ref.is_contiguous();
    let effective_input_ptr = if need_contig {
        NslTensor::make_contiguous(input_ptr)
    } else {
        input_ptr
    };

    let input = NslTensor::from_ptr(effective_input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);
    let bias = NslTensor::from_ptr(bias_ptr);

    let total = input.len as usize;
    let ndim = input.ndim as usize;
    let n = unsafe { *input.shape.add(ndim - 1) } as usize; // last dim size
    let num_rows = total / n;

    let in_dtype = input.dtype;
    // Allocate output with same shape as input
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, out_shape, ndim) };
    let out_strides = NslTensor::compute_strides(out_shape, ndim as i64);

    // Saved tensors for backward always use f64 for accuracy
    let mean_shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *mean_shape = num_rows as i64 };
    let mean_strides = NslTensor::compute_strides(mean_shape, 1);
    let mean_data = checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;

    let inv_std_shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *inv_std_shape = num_rows as i64 };
    let inv_std_strides = NslTensor::compute_strides(inv_std_shape, 1);
    let inv_std_data = checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;

    // Allocate output data buffer
    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc(total * elem_size);

    for row in 0..num_rows {
        let base = row * n;

        // Compute mean (always in f64 for numerical stability)
        let mut sum = 0.0_f64;
        for j in 0..n {
            let x = if in_dtype == 1 {
                unsafe { *input.data_f32().add(base + j) as f64 }
            } else {
                unsafe { *input.data_f64().add(base + j) }
            };
            sum += x;
        }
        let mean_val = sum / n as f64;

        // Compute variance
        let mut var = 0.0_f64;
        for j in 0..n {
            let x = if in_dtype == 1 {
                unsafe { *input.data_f32().add(base + j) as f64 }
            } else {
                unsafe { *input.data_f64().add(base + j) }
            };
            let diff = x - mean_val;
            var += diff * diff;
        }
        var /= n as f64;

        let inv_std_val = 1.0 / (var + eps).sqrt();

        unsafe {
            *mean_data.add(row) = mean_val;
            *inv_std_data.add(row) = inv_std_val;
        }

        // normalized * weight + bias
        for j in 0..n {
            let x = if in_dtype == 1 {
                unsafe { *input.data_f32().add(base + j) as f64 }
            } else {
                unsafe { *input.data_f64().add(base + j) }
            };
            let normalized = (x - mean_val) * inv_std_val;
            let w = if weight.dtype == 1 {
                unsafe { *weight.data_f32().add(j) as f64 }
            } else {
                unsafe { *weight.data_f64().add(j) }
            };
            let b = if bias.dtype == 1 {
                unsafe { *bias.data_f32().add(j) as f64 }
            } else {
                unsafe { *bias.data_f64().add(j) }
            };
            let result_val = normalized * w + b;
            if in_dtype == 1 {
                unsafe { *(out_data_raw as *mut f32).add(base + j) = result_val as f32 };
            } else {
                unsafe { *(out_data_raw as *mut f64).add(base + j) = result_val };
            }
        }
    }

    let out = Box::new(NslTensor {
        data: out_data_raw as *mut c_void,
        shape: out_shape,
        strides: out_strides,
        ndim: ndim as i64,
        len: total as i64,
        refcount: 1,
        device: input.device,
        dtype: in_dtype,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    let mean_tensor = Box::new(NslTensor {
        data: mean_data as *mut c_void,
        shape: mean_shape,
        strides: mean_strides,
        ndim: 1,
        len: num_rows as i64,
        refcount: 1,
        device: 0,
        dtype: 0, // saved stats always f64
        owns_data: 1,
    });
    let mean_ptr = Box::into_raw(mean_tensor) as i64;

    let inv_std_tensor = Box::new(NslTensor {
        data: inv_std_data as *mut c_void,
        shape: inv_std_shape,
        strides: inv_std_strides,
        ndim: 1,
        len: num_rows as i64,
        refcount: 1,
        device: 0,
        dtype: 0, // saved stats always f64
        owns_data: 1,
    });
    let inv_std_ptr = Box::into_raw(inv_std_tensor) as i64;

    if autodiff::is_recording() {
        // Bump refcounts for tensors that are also visible to user code
        NslTensor::from_ptr(input_ptr).refcount += 1;
        NslTensor::from_ptr(weight_ptr).refcount += 1;
        // mean and inv_std are purely internal to the tape (refcount=1).
        // Do NOT bump — no user-space variable will ever free them, so bumping
        // would cause a permanent leak on every training step.
        autodiff::maybe_record(autodiff::TapeOp::LayerNorm {
            input: input_ptr,
            weight: weight_ptr,
            bias: bias_ptr,
            out: out_ptr,
            saved_input: input_ptr,
            saved_mean: mean_ptr,
            saved_inv_std: inv_std_ptr,
            saved_weight: weight_ptr,
        });
    } else {
        // Not recording — free the saved tensors since they won't be used
        nsl_tensor_free(mean_ptr);
        nsl_tensor_free(inv_std_ptr);
    }

    // Free the contiguous copy if we made one
    if need_contig {
        nsl_tensor_free(effective_input_ptr);
    }

    out_ptr
}

/// RMSNorm: normalize by root-mean-square, scale by weight (no bias, no mean subtraction).
/// rms = sqrt(mean(x^2) + eps)
/// output = x / rms * weight
/// Saves {input, rms, weight} for backward.
#[no_mangle]
pub extern "C" fn nsl_tensor_rmsnorm(input_ptr: i64, weight_ptr: i64, eps: f64) -> i64 {
    // If input is not contiguous (e.g. transposed), make a contiguous copy first
    let input_ref = NslTensor::from_ptr(input_ptr);
    let need_contig = !input_ref.is_contiguous();
    let effective_input_ptr = if need_contig {
        NslTensor::make_contiguous(input_ptr)
    } else {
        input_ptr
    };

    let input = NslTensor::from_ptr(effective_input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);

    let total = input.len as usize;
    let ndim = input.ndim as usize;
    let n = unsafe { *input.shape.add(ndim - 1) } as usize;
    let num_rows = total / n;

    let in_dtype = input.dtype;
    // Allocate output with same shape as input
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, out_shape, ndim) };
    let out_strides = NslTensor::compute_strides(out_shape, ndim as i64);
    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc(total * elem_size);

    // Saved: rms per row [num_rows] — always f64 for accuracy
    let rms_shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *rms_shape = num_rows as i64 };
    let rms_strides = NslTensor::compute_strides(rms_shape, 1);
    let rms_data = checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;

    for row in 0..num_rows {
        let base = row * n;

        // Compute mean(x^2) in f64
        let mut sum_sq = 0.0_f64;
        for j in 0..n {
            let x = if in_dtype == 1 {
                unsafe { *input.data_f32().add(base + j) as f64 }
            } else {
                unsafe { *input.data_f64().add(base + j) }
            };
            sum_sq += x * x;
        }
        let rms_val = (sum_sq / n as f64 + eps).sqrt();

        unsafe { *rms_data.add(row) = rms_val };

        // output = x / rms * weight
        for j in 0..n {
            let x = if in_dtype == 1 {
                unsafe { *input.data_f32().add(base + j) as f64 }
            } else {
                unsafe { *input.data_f64().add(base + j) }
            };
            let w = if weight.dtype == 1 {
                unsafe { *weight.data_f32().add(j) as f64 }
            } else {
                unsafe { *weight.data_f64().add(j) }
            };
            let val = x / rms_val * w;
            if in_dtype == 1 {
                unsafe { *(out_data_raw as *mut f32).add(base + j) = val as f32 };
            } else {
                unsafe { *(out_data_raw as *mut f64).add(base + j) = val };
            }
        }
    }

    let out = Box::new(NslTensor {
        data: out_data_raw as *mut c_void,
        shape: out_shape,
        strides: out_strides,
        ndim: ndim as i64,
        len: total as i64,
        refcount: 1,
        device: input.device,
        dtype: in_dtype,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    let rms_tensor = Box::new(NslTensor {
        data: rms_data as *mut c_void,
        shape: rms_shape,
        strides: rms_strides,
        ndim: 1,
        len: num_rows as i64,
        refcount: 1,
        device: 0,
        dtype: 0, // saved stats always f64
        owns_data: 1,
    });
    let rms_ptr = Box::into_raw(rms_tensor) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(input_ptr).refcount += 1;
        NslTensor::from_ptr(weight_ptr).refcount += 1;
        // rms_tensor is purely internal to the tape (refcount=1).
        // Do NOT bump — no user-space variable will ever free it.
        autodiff::maybe_record(autodiff::TapeOp::RMSNorm {
            input: input_ptr,
            weight: weight_ptr,
            out: out_ptr,
            saved_input: input_ptr,
            saved_rms: rms_ptr,
            saved_weight: weight_ptr,
        });
    } else {
        nsl_tensor_free(rms_ptr);
    }

    // Free the contiguous copy if we made one
    if need_contig {
        nsl_tensor_free(effective_input_ptr);
    }

    out_ptr
}

// === Dropout ===

/// Dropout with inverted scaling. During training (training != 0, p > 0), randomly zero elements
/// with probability p and scale surviving elements by 1/(1-p). During eval, returns a clone.
#[no_mangle]
pub extern "C" fn nsl_tensor_dropout(tensor_ptr: i64, p: f64, training: i8) -> i64 {
    // Eval mode or p==0: identity
    if training == 0 || p == 0.0 {
        return nsl_tensor_clone(tensor_ptr);
    }

    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len as usize;
    let ndim = a.ndim;
    let in_dtype = a.dtype;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc(len * elem_size);

    // Generate dropout mask: 1.0 if kept, 0.0 if dropped (mask always f64 for backward)
    let mask_shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, mask_shape, ndim as usize) };
    let mask_strides = NslTensor::compute_strides(mask_shape, ndim);
    let mask_data = checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;

    let scale = 1.0 / (1.0 - p);

    if in_dtype == 1 {
        let out_data = out_data_raw as *mut f32;
        for i in 0..len {
            let rand_val = crate::sampling::rng_f64();
            let keep = if rand_val >= p { 1.0_f64 } else { 0.0_f64 };
            unsafe {
                *mask_data.add(i) = keep;
                *out_data.add(i) = (*a.data_f32().add(i) as f64 * keep * scale) as f32;
            }
        }
    } else {
        let out_data = out_data_raw as *mut f64;
        for i in 0..len {
            let rand_val = crate::sampling::rng_f64();
            let keep = if rand_val >= p { 1.0 } else { 0.0 };
            unsafe {
                *mask_data.add(i) = keep;
                *out_data.add(i) = *a.data_f64().add(i) * keep * scale;
            }
        }
    }

    let result = Box::new(NslTensor {
        data: out_data_raw as *mut c_void, shape, strides, ndim, len: len as i64, refcount: 1,
        device: 0,
        dtype: in_dtype,
        owns_data: 1,
    });
    let result_ptr = Box::into_raw(result) as i64;

    let mask_tensor = Box::new(NslTensor {
        data: mask_data as *mut c_void, shape: mask_shape, strides: mask_strides, ndim, len: len as i64, refcount: 1,
        device: 0,
        dtype: 0, // mask always f64 for backward
        owns_data: 1,
    });
    let mask_ptr = Box::into_raw(mask_tensor) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        NslTensor::from_ptr(mask_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Dropout {
            a: tensor_ptr,
            out: result_ptr,
            saved_mask: mask_ptr,
            scale,
        });
    } else {
        nsl_tensor_free(mask_ptr);
    }

    result_ptr
}

// === Conv2d ===

/// 2D convolution: input [N,C_in,H,W], weight [C_out,C_in,kH,kW], bias [C_out] (0 for no bias).
/// Returns output [N,C_out,H_out,W_out].
#[no_mangle]
pub extern "C" fn nsl_tensor_conv2d(
    input_ptr: i64,
    weight_ptr: i64,
    bias_ptr: i64,
    stride_h: i64,
    stride_w: i64,
    pad_h: i64,
    pad_w: i64,
) -> i64 {
    let input = NslTensor::from_ptr(input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);

    let n = unsafe { *input.shape.add(0) } as usize;
    let c_in = unsafe { *input.shape.add(1) } as usize;
    let h = unsafe { *input.shape.add(2) } as usize;
    let w = unsafe { *input.shape.add(3) } as usize;

    let c_out = unsafe { *weight.shape.add(0) } as usize;
    let kh = unsafe { *weight.shape.add(2) } as usize;
    let kw = unsafe { *weight.shape.add(3) } as usize;

    let sh = stride_h as usize;
    let sw = stride_w as usize;
    let ph = pad_h as usize;
    let pw = pad_w as usize;

    if h + 2 * ph < kh || w + 2 * pw < kw {
        eprintln!("nsl: conv2d kernel larger than padded input");
        std::process::abort();
    }
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;

    let in_dtype = input.dtype;
    let out_dtype: u16 = if in_dtype == 1 || weight.dtype == 1 { 1 } else { 0 };

    let out_len = n * c_out * h_out * w_out;
    let out_shape = checked_alloc(4 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = n as i64;
        *out_shape.add(1) = c_out as i64;
        *out_shape.add(2) = h_out as i64;
        *out_shape.add(3) = w_out as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 4);

    let read_input = |idx: usize| -> f64 {
        if in_dtype == 1 { unsafe { *input.data_f32().add(idx) as f64 } }
        else { unsafe { *input.data_f64().add(idx) } }
    };
    let read_weight = |idx: usize| -> f64 {
        if weight.dtype == 1 { unsafe { *weight.data_f32().add(idx) as f64 } }
        else { unsafe { *weight.data_f64().add(idx) } }
    };

    let elem_size = if out_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc_zeroed(out_len * elem_size);

    // Direct nested-loop convolution
    for ni in 0..n {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut val = 0.0_f64;
                    for ci in 0..c_in {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let ih = oh * sh + ky;
                                let iw = ow * sw + kx;
                                // Check padding bounds
                                if ih >= ph && iw >= pw && ih - ph < h && iw - pw < w {
                                    let input_idx = ni * (c_in * h * w) + ci * (h * w) + (ih - ph) * w + (iw - pw);
                                    let weight_idx = co * (c_in * kh * kw) + ci * (kh * kw) + ky * kw + kx;
                                    val += read_input(input_idx) * read_weight(weight_idx);
                                }
                            }
                        }
                    }
                    // Add bias if provided
                    if bias_ptr != 0 {
                        let bias = NslTensor::from_ptr(bias_ptr);
                        let bv = if bias.dtype == 1 { unsafe { *bias.data_f32().add(co) as f64 } }
                                 else { unsafe { *bias.data_f64().add(co) } };
                        val += bv;
                    }
                    let out_idx = ni * (c_out * h_out * w_out) + co * (h_out * w_out) + oh * w_out + ow;
                    if out_dtype == 1 {
                        unsafe { *(out_data_raw as *mut f32).add(out_idx) = val as f32 };
                    } else {
                        unsafe { *(out_data_raw as *mut f64).add(out_idx) = val };
                    }
                }
            }
        }
    }

    let result = Box::new(NslTensor {
        data: out_data_raw as *mut c_void, shape: out_shape, strides: out_strides, ndim: 4, len: out_len as i64, refcount: 1,
        device: 0,
        dtype: out_dtype,
        owns_data: 1,
    });
    let result_ptr = Box::into_raw(result) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(input_ptr).refcount += 1;
        NslTensor::from_ptr(weight_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Conv2d {
            input: input_ptr,
            weight: weight_ptr,
            bias: bias_ptr,
            out: result_ptr,
            saved_input: input_ptr,
            saved_weight: weight_ptr,
            stride_h: sh,
            stride_w: sw,
            pad_h: ph,
            pad_w: pw,
        });
    }

    result_ptr
}

// === MaxPool2d ===

/// 2D max pooling: input [N,C,H,W] -> output [N,C,H_out,W_out].
/// Saves argmax indices for backward gradient routing.
#[no_mangle]
pub extern "C" fn nsl_tensor_maxpool2d(
    input_ptr: i64,
    kernel_h: i64,
    kernel_w: i64,
    stride: i64,
    padding: i64,
) -> i64 {
    let input = NslTensor::from_ptr(input_ptr);

    let n = unsafe { *input.shape.add(0) } as usize;
    let c = unsafe { *input.shape.add(1) } as usize;
    let h = unsafe { *input.shape.add(2) } as usize;
    let w = unsafe { *input.shape.add(3) } as usize;

    let kh = kernel_h as usize;
    let kw = kernel_w as usize;
    let s = stride as usize;
    let pad = padding as usize;

    if h + 2 * pad < kh || w + 2 * pad < kw {
        eprintln!("nsl: maxpool2d kernel larger than padded input");
        std::process::abort();
    }
    let h_out = (h + 2 * pad - kh) / s + 1;
    let w_out = (w + 2 * pad - kw) / s + 1;

    let in_dtype = input.dtype;

    let out_len = n * c * h_out * w_out;
    let out_shape = checked_alloc(4 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = n as i64;
        *out_shape.add(1) = c as i64;
        *out_shape.add(2) = h_out as i64;
        *out_shape.add(3) = w_out as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 4);
    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc(out_len * elem_size);

    // Save argmax indices for backward
    let mut argmax_indices: Vec<usize> = vec![0; out_len];

    let read_input = |idx: usize| -> f64 {
        if in_dtype == 1 { unsafe { *input.data_f32().add(idx) as f64 } }
        else { unsafe { *input.data_f64().add(idx) } }
    };

    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx: usize = 0;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let ih = oh * s + ky;
                            let iw = ow * s + kx;
                            if ih >= pad && iw >= pad && ih - pad < h && iw - pad < w {
                                let input_idx = ni * (c * h * w) + ci * (h * w) + (ih - pad) * w + (iw - pad);
                                let val = read_input(input_idx);
                                if val > max_val {
                                    max_val = val;
                                    max_idx = input_idx;
                                }
                            }
                        }
                    }
                    let out_idx = ni * (c * h_out * w_out) + ci * (h_out * w_out) + oh * w_out + ow;
                    if in_dtype == 1 {
                        unsafe { *(out_data_raw as *mut f32).add(out_idx) = max_val as f32 };
                    } else {
                        unsafe { *(out_data_raw as *mut f64).add(out_idx) = max_val };
                    }
                    argmax_indices[out_idx] = max_idx;
                }
            }
        }
    }

    let result = Box::new(NslTensor {
        data: out_data_raw as *mut c_void, shape: out_shape, strides: out_strides, ndim: 4, len: out_len as i64, refcount: 1,
        device: 0,
        dtype: in_dtype,
        owns_data: 1,
    });
    let result_ptr = Box::into_raw(result) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(input_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::MaxPool2d {
            a: input_ptr,
            out: result_ptr,
            saved_argmax: argmax_indices,
            input_shape: vec![n as i64, c as i64, h as i64, w as i64],
        });
    }

    result_ptr
}

/// Add 1D bias [N] to 2D tensor [M, N] — broadcasts bias along rows.
/// output[i, j] = tensor[i, j] + bias[j]
#[no_mangle]
pub extern "C" fn nsl_tensor_bias_add(tensor_ptr: i64, bias_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let bias = NslTensor::from_ptr(bias_ptr);

    if tensor.ndim != 2 {
        eprintln!(
            "nsl: bias_add requires 2D tensor (got {}D)",
            tensor.ndim
        );
        std::process::abort();
    }
    if bias.ndim != 1 {
        eprintln!(
            "nsl: bias_add requires 1D bias (got {}D)",
            bias.ndim
        );
        std::process::abort();
    }

    let rows = unsafe { *tensor.shape.add(0) } as usize;
    let cols = unsafe { *tensor.shape.add(1) } as usize;
    let bias_len = unsafe { *bias.shape.add(0) } as usize;

    if cols != bias_len {
        eprintln!(
            "nsl: bias_add shape mismatch — tensor has {} cols but bias has {} elements",
            cols, bias_len
        );
        std::process::abort();
    }

    let in_dtype = tensor.dtype;
    let out_dtype: u16 = if in_dtype == 1 || bias.dtype == 1 { 1 } else { 0 };

    // Output shape: same as tensor [rows, cols]
    let out_ndim: i64 = 2;
    let out_len = (rows * cols) as i64;
    let out_shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = rows as i64;
        *out_shape.add(1) = cols as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let elem_size = if out_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc((out_len as usize) * elem_size);

    let read_t = |idx: usize| -> f64 {
        if in_dtype == 1 { unsafe { *tensor.data_f32().add(idx) as f64 } }
        else { unsafe { *tensor.data_f64().add(idx) } }
    };
    let read_b = |idx: usize| -> f64 {
        if bias.dtype == 1 { unsafe { *bias.data_f32().add(idx) as f64 } }
        else { unsafe { *bias.data_f64().add(idx) } }
    };

    for i in 0..rows {
        for j in 0..cols {
            let val = read_t(i * cols + j) + read_b(j);
            if out_dtype == 1 {
                unsafe { *(out_data_raw as *mut f32).add(i * cols + j) = val as f32 };
            } else {
                unsafe { *(out_data_raw as *mut f64).add(i * cols + j) = val };
            }
        }
    }

    let out = Box::new(NslTensor {
        data: out_data_raw as *mut c_void,
        shape: out_shape,
        strides: out_strides,
        ndim: out_ndim,
        len: out_len,
        refcount: 1,
        device: 0,
        dtype: out_dtype,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        NslTensor::from_ptr(bias_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::BiasAdd {
            tensor: tensor_ptr,
            bias: bias_ptr,
            out: out_ptr,
        });
    }

    out_ptr
}

/// Transfer a tensor to a different device.
/// CPU→GPU: converts f64 → f32, allocates unified memory, prefetches.
/// GPU→CPU: converts f32 → f64, allocates CPU memory.
#[no_mangle]
pub extern "C" fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64 {
    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    let target = target_device as u8;

    if t.device == target {
        // Same device — increment refcount and return same pointer
        let t_mut = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        t_mut.refcount += 1;
        return tensor_ptr;
    }

    #[allow(unused_variables)]
    let len = t.len as usize;

    if t.device == 0 && target > 0 {
        // CPU → GPU: convert to f32 (if already f32, copy directly)
        #[cfg(feature = "cuda")]
        {
            let dst_size = len * std::mem::size_of::<f32>();
            let dst = crate::cuda::inner::alloc_managed(dst_size);
            let dst_f32 = dst as *mut f32;

            // Convert to f32 on CPU side (unified memory)
            if t.dtype == 1 {
                let src = t.data_f32();
                unsafe { std::ptr::copy_nonoverlapping(src, dst_f32, len); }
            } else {
                let src = t.data_f64();
                for i in 0..len {
                    unsafe { *dst_f32.add(i) = *src.add(i) as f32; }
                }
            }

            // Prefetch to GPU
            crate::cuda::inner::prefetch_to_device(dst, dst_size, (target - 1) as i32);

            // Copy shape and strides
            let shape = NslTensor::copy_shape(t.shape, t.ndim);
            let strides = NslTensor::compute_strides(shape, t.ndim);

            let new_t = Box::new(NslTensor {
                data: dst,
                shape,
                strides,
                ndim: t.ndim,
                len: t.len,
                refcount: 1,
                device: target,
                dtype: 1, // f32
                owns_data: 1,
            });
            return Box::into_raw(new_t) as i64;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }

    if t.device > 0 && target == 0 {
        // GPU → CPU: f32 → f64
        #[cfg(feature = "cuda")]
        {
            // Sync to ensure GPU writes are visible
            unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

            let src = t.data_f32();
            let dst_size = len * std::mem::size_of::<f64>();
            let layout = std::alloc::Layout::from_size_align(dst_size, 8).unwrap();
            let dst = unsafe { std::alloc::alloc(layout) as *mut f64 };
            let dst_void = dst as *mut std::ffi::c_void;

            // Convert f32 → f64
            for i in 0..len {
                unsafe { *dst.add(i) = *src.add(i) as f64; }
            }

            let shape = NslTensor::copy_shape(t.shape, t.ndim);
            let strides = NslTensor::compute_strides(shape, t.ndim);

            let new_t = Box::new(NslTensor {
                data: dst_void,
                shape,
                strides,
                ndim: t.ndim,
                len: t.len,
                refcount: 1,
                device: 0,
                dtype: 0, // f64
                owns_data: 1,
            });
            return Box::into_raw(new_t) as i64;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }

    panic!("GPU-to-GPU transfer not yet supported");
}

// ---------------------------------------------------------------------------
// Slice assignment primitives (M19 data pipeline)
// ---------------------------------------------------------------------------

/// Describes one dimension of a slice assignment operation.
#[repr(C)]
pub struct NslSliceDim {
    pub is_scalar: u8, // 1 = single index, 0 = range
    pub start: i64,
    pub end: i64, // ignored if is_scalar=1
}

/// Set a single element in a tensor by flat indices.
/// `indices_ptr` points to an array of `i64` indices, one per dimension.
#[no_mangle]
pub extern "C" fn nsl_tensor_set_element(
    tensor_ptr: i64,
    indices_ptr: i64,
    num_indices: i64,
    value: f64,
) {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let n = num_indices as usize;

    if n != ndim {
        eprintln!(
            "nsl: set_element: expected {} indices, got {}",
            ndim, n
        );
        std::process::abort();
    }

    let strides = crate::cpu::get_strides_vec(tensor);
    let mut offset: usize = 0;
    for (d, &stride) in strides.iter().enumerate().take(ndim) {
        let idx = unsafe { *(indices_ptr as *const i64).add(d) } as usize;
        let dim_size = unsafe { *tensor.shape.add(d) } as usize;
        if idx >= dim_size {
            eprintln!(
                "nsl: set_element: index {} out of bounds for dim {} (size {})",
                idx, d, dim_size
            );
            std::process::abort();
        }
        offset += idx * stride;
    }

    if tensor.dtype == 1 {
        unsafe { *tensor.data_f32().add(offset) = value as f32 };
    } else {
        unsafe { *tensor.data_f64().add(offset) = value };
    }
}

/// Assign `src` tensor into a slice of `target` tensor.
/// `dims_ptr` points to an array of [`NslSliceDim`], one per dimension.
#[no_mangle]
pub extern "C" fn nsl_tensor_slice_assign(
    target_ptr: i64,
    src_ptr: i64,
    dims_ptr: i64,
    num_dims: i64,
) {
    let target = NslTensor::from_ptr(target_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    let ndim = num_dims as usize;
    let dims =
        unsafe { std::slice::from_raw_parts(dims_ptr as *const NslSliceDim, ndim) };
    let target_strides = crate::cpu::get_strides_vec(target);
    let target_shape = crate::cpu::get_shape_vec(target);

    // Compute the slice region: for each dim, determine start..end range
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(ndim);
    for d in 0..ndim {
        let dim_size = target_shape[d] as usize;
        if dims[d].is_scalar != 0 {
            let idx = if dims[d].start < 0 {
                (dim_size as i64 + dims[d].start) as usize
            } else {
                dims[d].start as usize
            };
            ranges.push((idx, idx + 1));
        } else {
            let start = if dims[d].start < 0 {
                (dim_size as i64 + dims[d].start) as usize
            } else {
                dims[d].start as usize
            };
            let end = if dims[d].end < 0 {
                (dim_size as i64 + dims[d].end) as usize
            } else {
                dims[d].end.min(dim_size as i64) as usize
            };
            ranges.push((start, end));
        }
    }

    // Iterate over all positions in the slice and copy from src
    let mut src_flat = 0usize;

    #[allow(clippy::too_many_arguments)]
    fn recurse(
        depth: usize,
        ndim: usize,
        ranges: &[(usize, usize)],
        target: &NslTensor,
        src: &NslTensor,
        target_strides: &[usize],
        target_offset: usize,
        src_flat: &mut usize,
    ) {
        if depth == ndim {
            if *src_flat < src.len as usize {
                // Read from src (handle both dtypes) and write to target (handle both dtypes)
                let val: f64 = if src.dtype == 1 {
                    unsafe { *src.data_f32().add(*src_flat) as f64 }
                } else {
                    unsafe { *src.data_f64().add(*src_flat) }
                };
                if target.dtype == 1 {
                    unsafe { *target.data_f32().add(target_offset) = val as f32 };
                } else {
                    unsafe { *target.data_f64().add(target_offset) = val };
                }
                *src_flat += 1;
            }
            return;
        }
        let (start, end) = ranges[depth];
        for i in start..end {
            recurse(
                depth + 1,
                ndim,
                ranges,
                target,
                src,
                target_strides,
                target_offset + i * target_strides[depth],
                src_flat,
            );
        }
    }

    recurse(
        0,
        ndim,
        &ranges,
        target,
        src,
        &target_strides,
        0,
        &mut src_flat,
    );
}

// ---------------------------------------------------------------------------
// Custom dtype pack/unpack conversion FFI (Task 7)
// ---------------------------------------------------------------------------

/// Helper: clone shape/strides arrays for new tensor ownership
fn clone_shape(src: *mut i64, ndim: usize) -> *mut i64 {
    let bytes = ndim * std::mem::size_of::<i64>();
    let dst = checked_alloc(bytes) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(src, dst, ndim); }
    dst
}

/// Convert a tensor to a custom dtype by calling the registered pack function.
#[no_mangle]
pub extern "C" fn nsl_tensor_to_custom_dtype(
    tensor_ptr: i64,
    target_dtype_id: i64,
) -> i64 {
    let target_dtype_id = target_dtype_id as u16;
    let tensor = unsafe { &*(tensor_ptr as *const NslTensor) };
    let registry = get_registry();

    let info = match registry.get(&target_dtype_id) {
        Some(info) => info,
        None => {
            eprintln!("nsl: unknown custom dtype id {target_dtype_id}");
            return tensor_ptr;
        }
    };

    let pack_fn = match info.pack_fn {
        Some(f) => f,
        None => {
            eprintln!("nsl: custom dtype '{}' has no pack function", info.name);
            return tensor_ptr;
        }
    };

    let num_elements = tensor.len as usize;

    if info.block_size == 0 {
        // Element-wise: pack_fn signature: extern "C" fn(i64) -> i64
        // The f64 value is passed as raw i64 bits to avoid calling-convention
        // issues with floating-point registers on Windows.
        let pack: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(pack_fn) };

        let packed_bytes = num_elements * info.element_size;
        let packed_data = checked_alloc_zeroed(packed_bytes);

        for i in 0..num_elements {
            // Read element as f64 regardless of source dtype (promote f32→f64 if needed)
            let val: f64 = if tensor.dtype == 1 {
                unsafe { *(tensor.data as *const f32).add(i) as f64 }
            } else {
                unsafe { *(tensor.data as *const f64).add(i) }
            };
            let val_bits = val.to_bits() as i64;
            let packed_val = pack(val_bits);
            let dst = unsafe { packed_data.add(i * info.element_size) };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &packed_val as *const i64 as *const u8,
                    dst,
                    info.element_size,
                );
            }
        }

        let shape_ptr = clone_shape(tensor.shape, tensor.ndim as usize);
        let strides_ptr = clone_shape(tensor.strides, tensor.ndim as usize);

        let result = Box::new(NslTensor {
            data: packed_data as *mut c_void,
            shape: shape_ptr,
            strides: strides_ptr,
            ndim: tensor.ndim,
            len: tensor.len,
            refcount: 1,
            device: tensor.device,
            dtype: target_dtype_id,
            owns_data: 1,
        });
        Box::into_raw(result) as i64
    } else {
        // Block-wise
        let block_sz = info.block_size as usize;
        let innermost = unsafe { *tensor.shape.add(tensor.ndim as usize - 1) } as usize;

        if !innermost.is_multiple_of(block_sz) {
            eprintln!("nsl: tensor dim {} not divisible by block_size {}", innermost, block_sz);
            return tensor_ptr;
        }

        let num_blocks = num_elements / block_sz;
        let pbs = info.packed_block_size;
        let packed_bytes = num_blocks * pbs;
        let packed_data = checked_alloc_zeroed(packed_bytes);

        // Block pack_fn: extern "C" fn(*const f64, i64, *mut u8)
        let pack: extern "C" fn(*const f64, i64, *mut u8) =
            unsafe { std::mem::transmute(pack_fn) };

        let src = tensor.data as *const f64;
        for b in 0..num_blocks {
            let block_ptr = unsafe { src.add(b * block_sz) };
            let dst = unsafe { packed_data.add(b * pbs) };
            pack(block_ptr, block_sz as i64, dst);
        }

        let shape_ptr = clone_shape(tensor.shape, tensor.ndim as usize);
        let strides_ptr = clone_shape(tensor.strides, tensor.ndim as usize);

        let result = Box::new(NslTensor {
            data: packed_data as *mut c_void,
            shape: shape_ptr,
            strides: strides_ptr,
            ndim: tensor.ndim,
            len: tensor.len,
            refcount: 1,
            device: tensor.device,
            dtype: target_dtype_id,
            owns_data: 1,
        });
        Box::into_raw(result) as i64
    }
}

/// Convert a custom-dtype tensor back to f64.
#[no_mangle]
pub extern "C" fn nsl_tensor_from_custom_dtype(tensor_ptr: i64) -> i64 {
    let tensor = unsafe { &*(tensor_ptr as *const NslTensor) };

    if tensor.dtype < DTYPE_CUSTOM_START {
        return tensor_ptr;
    }

    let registry = get_registry();
    let info = match registry.get(&tensor.dtype) {
        Some(info) => info,
        None => return tensor_ptr,
    };

    let unpack_fn = match info.unpack_fn {
        Some(f) => f,
        None => {
            eprintln!("nsl: custom dtype '{}' has no unpack function", info.name);
            return tensor_ptr;
        }
    };

    let num_elements = tensor.len as usize;
    let out_data = checked_alloc_zeroed(num_elements * 8) as *mut f64;

    if info.block_size == 0 {
        // Element-wise: unpack_fn signature: extern "C" fn(i64) -> i64
        // Returns f64 bits as i64 to avoid floating-point register issues.
        let unpack: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(unpack_fn) };
        let src = tensor.data as *const u8;
        for i in 0..num_elements {
            let mut packed_val: i64 = 0;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.add(i * info.element_size),
                    &mut packed_val as *mut i64 as *mut u8,
                    info.element_size,
                );
            }
            let result_bits = unpack(packed_val);
            unsafe { *out_data.add(i) = f64::from_bits(result_bits as u64); }
        }
    } else {
        let block_sz = info.block_size as usize;
        let num_blocks = num_elements / block_sz;
        let pbs = info.packed_block_size;
        let unpack: extern "C" fn(*const u8, i64, *mut f64) =
            unsafe { std::mem::transmute(unpack_fn) };
        let src = tensor.data as *const u8;
        for b in 0..num_blocks {
            let packed_ptr = unsafe { src.add(b * pbs) };
            let dst = unsafe { out_data.add(b * block_sz) };
            unpack(packed_ptr, block_sz as i64, dst);
        }
    }

    let shape_ptr = clone_shape(tensor.shape, tensor.ndim as usize);
    let strides_ptr = clone_shape(tensor.strides, tensor.ndim as usize);

    let result = Box::new(NslTensor {
        data: out_data as *mut c_void,
        shape: shape_ptr,
        strides: strides_ptr,
        ndim: tensor.ndim,
        len: tensor.len,
        refcount: 1,
        device: tensor.device,
        dtype: DTYPE_F64,
        owns_data: 1,
    });
    Box::into_raw(result) as i64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_element() {
        // Create a 2x3 zero tensor
        let shape = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape, 2);
        crate::list::nsl_list_push(shape, 3);
        let t = nsl_tensor_zeros(shape);
        let indices: [i64; 2] = [1, 2];
        nsl_tensor_set_element(t, indices.as_ptr() as i64, 2, 42.0);
        let tensor = NslTensor::from_ptr(t);
        // Element at [1, 2]: row 1 * 3 + col 2 = 5
        assert_eq!(unsafe { *tensor.data_f32().add(5) }, 42.0_f32);
    }

    #[test]
    fn test_slice_assign() {
        // Create a 2x4 zero tensor, assign [10,20,30] into [0, 0:3]
        let shape = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape, 2);
        crate::list::nsl_list_push(shape, 4);
        let target = nsl_tensor_zeros(shape);

        // Create src as an f32 tensor (use nsl_tensor_zeros then set elements)
        let src_shape = crate::list::nsl_list_new();
        crate::list::nsl_list_push(src_shape, 3);
        let src = nsl_tensor_zeros(src_shape);
        let s = NslTensor::from_ptr(src);
        unsafe {
            *s.data_f32().add(0) = 10.0_f32;
            *s.data_f32().add(1) = 20.0_f32;
            *s.data_f32().add(2) = 30.0_f32;
        }

        // Slice: [0, 0:3] => dim0: scalar index 0, dim1: range 0..3
        let dims = [
            NslSliceDim {
                is_scalar: 1,
                start: 0,
                end: 0,
            },
            NslSliceDim {
                is_scalar: 0,
                start: 0,
                end: 3,
            },
        ];
        nsl_tensor_slice_assign(target, src, dims.as_ptr() as i64, 2);

        let t = NslTensor::from_ptr(target);
        assert_eq!(unsafe { *t.data_f32().add(0) }, 10.0_f32); // [0,0]
        assert_eq!(unsafe { *t.data_f32().add(1) }, 20.0_f32); // [0,1]
        assert_eq!(unsafe { *t.data_f32().add(2) }, 30.0_f32); // [0,2]
        assert_eq!(unsafe { *t.data_f32().add(3) }, 0.0_f32); // [0,3] unchanged
        assert_eq!(unsafe { *t.data_f32().add(4) }, 0.0_f32); // [1,0] unchanged
    }
}
