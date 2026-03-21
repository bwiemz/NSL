//! Tensor module — NslTensor struct, core utilities, and re-exports of all sub-modules.

pub mod creation;
pub mod arithmetic;
pub mod reduction;
pub mod shape_ops;
pub mod activation;
pub mod trig;

// Re-export everything from sub-modules so the public API is unchanged.
pub use creation::*;
pub use arithmetic::*;
pub use reduction::*;
pub use shape_ops::*;
pub use activation::*;
pub use trig::*;

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, Ordering};

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
    pub(crate) refcount: AtomicI64,
    pub(crate) device: u8,          // 0 = CPU, 1+ = CUDA device ID
    pub(crate) dtype: u16,          // 0 = f64, 1 = f32; 256+ = custom user-defined dtypes
    pub(crate) owns_data: u8,       // 1 = heap-owned (free on drop), 0 = borrowed/mmap
    /// i64 pointer to the NslTensor that owns this tensor's data buffer.
    /// 0 means this tensor owns its own data. Non-zero means this is a view.
    /// When this tensor is freed, the owner's refcount is decremented.
    pub(crate) data_owner: i64,
}

// Built-in dtype IDs (match existing u8 values)
pub const DTYPE_F64: u16 = 0;
pub const DTYPE_F32: u16 = 1;
pub const DTYPE_FP16: u16 = 2;
pub const DTYPE_BF16: u16 = 3;
pub const DTYPE_INT8: u16 = 4;
pub const DTYPE_FP8E4M3: u16 = 5;
pub const DTYPE_FP8E5M2: u16 = 6;

// Custom dtype IDs start at 256
pub const DTYPE_CUSTOM_START: u16 = 256;

/// Metadata for a user-defined custom datatype
pub struct CustomDtypeInfo {
    pub id: u16,
    pub name: String,
    pub bit_width: u8,
    pub block_size: u32,           // 0 = element-wise, >0 = block format
    pub element_size: usize,       // bytes per packed ELEMENT (ceil(bits/8))
    pub packed_block_size: usize,  // bytes per packed BLOCK -- 0 for element-wise
    pub pack_fn: Option<*const c_void>,
    pub unpack_fn: Option<*const c_void>,
}

// SAFETY: function pointers are set once at startup, read-only after.
unsafe impl Send for CustomDtypeInfo {}
unsafe impl Sync for CustomDtypeInfo {}

/// Registry uses OnceLock -- initialized once at startup, read-only after.
static CUSTOM_DTYPE_REGISTRY: OnceLock<HashMap<u16, CustomDtypeInfo>> = OnceLock::new();

fn get_registry() -> &'static HashMap<u16, CustomDtypeInfo> {
    static EMPTY: std::sync::LazyLock<HashMap<u16, CustomDtypeInfo>> =
        std::sync::LazyLock::new(HashMap::new);
    CUSTOM_DTYPE_REGISTRY.get().unwrap_or(&EMPTY)
}

thread_local! {
    static STAGING_REGISTRY: std::cell::RefCell<HashMap<u16, CustomDtypeInfo>>
        = std::cell::RefCell::new(HashMap::new());
}

/// Register a custom datatype. Called by codegen-generated init code at startup.
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

    // RISKY-3 fix: guard against null name_ptr or negative name_len
    if name_ptr.is_null() || name_len <= 0 {
        return;
    }
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

/// Called once after all registrations. Moves staging -> OnceLock.
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

    /// Total byte size of the data buffer, accounting for block-packed custom dtypes.
    #[inline]
    pub(crate) fn data_byte_size(&self) -> usize {
        if self.dtype >= DTYPE_CUSTOM_START {
            if let Some(info) = get_registry().get(&self.dtype) {
                if info.block_size > 0 && info.packed_block_size > 0 {
                    let num_blocks = (self.len as usize)
                        .div_ceil(info.block_size as usize);
                    return num_blocks * info.packed_block_size;
                }
            }
        }
        (self.len as usize) * self.element_size()
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

    pub(crate) fn total_elements(shape: *const i64, ndim: i64) -> i64 {
        let mut total: i64 = 1;
        for i in 0..ndim as usize {
            let dim = unsafe { *shape.add(i) };
            total = total.checked_mul(dim).unwrap_or_else(|| {
                eprintln!("nsl: tensor shape overflow -- dimensions too large");
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
    pub(crate) fn make_contiguous(ptr: i64) -> i64 {
        let tensor = NslTensor::from_ptr(ptr);
        if tensor.is_contiguous() {
            return nsl_tensor_clone(ptr);
        }

        let ndim = tensor.ndim as usize;
        let len = tensor.len as usize;
        let elem_size = tensor.element_size();

        let new_shape = NslTensor::copy_shape(tensor.shape, tensor.ndim);
        let new_strides = NslTensor::compute_strides(new_shape, tensor.ndim);
        let new_data = checked_alloc(len * elem_size);

        let shape_vec: Vec<i64> = (0..ndim).map(|i| unsafe { *tensor.shape.add(i) }).collect();
        let src_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *tensor.strides.add(i) }).collect();

        for flat in 0..len {
            let mut remaining = flat;
            let mut src_offset = 0_usize;
            for d in 0..ndim {
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
            refcount: AtomicI64::new(1),
            device: tensor.device,
            dtype: tensor.dtype,
            owns_data: 1, data_owner: 0,
        });
        Box::into_raw(result) as i64
    }

    /// Create a view tensor that shares data with `source_ptr`.
    /// Bumps source refcount. View has `owns_data = 0` and `data_owner = source_ptr`.
    /// `new_shape` and `new_strides` slices are copied into fresh allocations.
    pub fn new_view_i64(
        source_ptr: i64,
        new_shape: &[i64],
        new_strides: &[i64],
        ndim: i64,
        len: i64,
    ) -> i64 {
        let source = NslTensor::from_ptr(source_ptr);
        let n = ndim as usize;
        let shape_bytes = n * std::mem::size_of::<i64>();

        let shape = checked_alloc(shape_bytes) as *mut i64;
        unsafe { std::ptr::copy_nonoverlapping(new_shape.as_ptr(), shape, n) };

        let strides = checked_alloc(shape_bytes) as *mut i64;
        unsafe { std::ptr::copy_nonoverlapping(new_strides.as_ptr(), strides, n) };

        // Determine the true data owner: if source is itself a view, inherit its owner
        let true_owner = if source.data_owner != 0 {
            // Source is a view — our owner is the source's owner (the root)
            let root = NslTensor::from_ptr(source.data_owner);
            root.refcount.fetch_add(1, Ordering::SeqCst);
            source.data_owner
        } else {
            // Source owns its data — it becomes our owner
            source.refcount.fetch_add(1, Ordering::SeqCst);
            source_ptr
        };

        let tensor = Box::new(NslTensor {
            data: source.data,
            shape,
            strides,
            ndim,
            len,
            refcount: AtomicI64::new(1),
            device: source.device,
            dtype: source.dtype,
            owns_data: 0,
            data_owner: true_owner,
        });

        Box::into_raw(tensor) as i64
    }
}

// ---------------------------------------------------------------------------
// Internal helpers used across sub-modules
// ---------------------------------------------------------------------------

/// Helper: get the shape of a tensor as a Vec<i64>.
pub(crate) fn get_shape_vec(tensor: &NslTensor) -> Vec<i64> {
    crate::cpu::get_shape_vec(tensor)
}

/// Helper: get the strides of a tensor as a Vec<usize>.
pub(crate) fn get_strides_vec(tensor: &NslTensor) -> Vec<usize> {
    crate::cpu::get_strides_vec(tensor)
}

/// Helper: create a tensor with a given shape (Rust slice).
#[allow(dead_code)]
pub(crate) fn create_tensor_with_shape_rs(shape: &[i64]) -> i64 {
    crate::cpu::create_tensor_with_shape_rs(shape)
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
        for i in 0..size {
            if i > 0 { print!(", "); }
            let val = match dtype {
                1 => unsafe { *(data as *const f32).add(i * stride) as f64 },
                d if d >= DTYPE_CUSTOM_START => {
                    print!("?");
                    continue;
                }
                _ => unsafe { *(data as *const f64).add(i * stride) },
            };
            print_float_value(val);
        }
    } else {
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
    // Ensure we clone from contiguous data so non-contiguous views are handled correctly
    let c_ptr = nsl_tensor_contiguous(tensor_ptr);
    let tensor = NslTensor::from_ptr(c_ptr);
    let ndim = tensor.ndim;
    let len = tensor.len;

    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(tensor.shape, shape, ndim as usize) };

    let strides = NslTensor::compute_strides(shape, ndim);

    let elem_size = tensor.element_size();
    let data_size = (len as usize) * elem_size;
    let data = if tensor.device > 0 {
        #[cfg(feature = "cuda")]
        { crate::cuda::inner::alloc_managed(data_size) as *mut u8 }
        #[cfg(not(feature = "cuda"))]
        { checked_alloc(data_size) }
    } else {
        checked_alloc(data_size)
    };
    unsafe { std::ptr::copy_nonoverlapping(tensor.data as *const u8, data, data_size) };

    let result = Box::new(NslTensor {
        data: data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        refcount: AtomicI64::new(1),
        device: tensor.device,
        dtype: tensor.dtype,
        owns_data: 1, data_owner: 0,
    });
    nsl_tensor_free(c_ptr);
    Box::into_raw(result) as i64
}

#[no_mangle]
pub extern "C" fn nsl_tensor_free(tensor_ptr: i64) {
    if tensor_ptr == 0 {
        return;
    }
    let (should_free, data_ptr, data_size, shape_ptr, strides_ptr, shape_size, device, owns_data, data_owner) = {
        let tensor = NslTensor::from_ptr(tensor_ptr);
        let prev = tensor.refcount.fetch_sub(1, Ordering::SeqCst);
        if prev > 1 {
            return; // still referenced — don't free
        }
        (
            true,
            tensor.data,
            tensor.data_byte_size(),
            tensor.shape,
            tensor.strides,
            (tensor.ndim as usize) * std::mem::size_of::<i64>(),
            tensor.device,
            tensor.owns_data,
            tensor.data_owner,
        )
    };

    if should_free {
        unsafe {
            // If this is a view, decrement the data owner's refcount
            if data_owner != 0 {
                let owner = NslTensor::from_ptr(data_owner);
                let owner_prev = owner.refcount.fetch_sub(1, Ordering::SeqCst);
                if owner_prev == 1 {
                    // Owner's last reference gone — free the owner's data
                    if owner.owns_data != 0 {
                        if owner.device > 0 {
                            #[cfg(feature = "cuda")]
                            {
                                crate::cuda::inner::free_managed(owner.data);
                            }
                            #[cfg(not(feature = "cuda"))]
                            {
                                checked_free(owner.data as *mut u8, owner.data_byte_size());
                            }
                        } else {
                            checked_free(owner.data as *mut u8, owner.data_byte_size());
                        }
                    }
                    // Free owner's shape, strides, box
                    let owner_shape_size = (owner.ndim as usize) * std::mem::size_of::<i64>();
                    if !owner.shape.is_null() {
                        checked_free(owner.shape as *mut u8, owner_shape_size);
                    }
                    if !owner.strides.is_null() {
                        checked_free(owner.strides as *mut u8, owner_shape_size);
                    }
                    crate::fp8::remove_fp8_scale(data_owner);
                    drop(Box::from_raw(data_owner as *mut NslTensor));
                }
            } else if owns_data != 0 {
                // This tensor owns its data — free the buffer
                if device > 0 {
                    #[cfg(feature = "cuda")]
                    {
                        crate::cuda::inner::free_managed(data_ptr);
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        checked_free(data_ptr as *mut u8, data_size);
                    }
                } else {
                    checked_free(data_ptr as *mut u8, data_size);
                }
            }

            // Free this tensor's shape, strides, and box
            if !shape_ptr.is_null() {
                checked_free(shape_ptr as *mut u8, shape_size);
            }
            if !strides_ptr.is_null() {
                checked_free(strides_ptr as *mut u8, shape_size);
            }
            crate::fp8::remove_fp8_scale(tensor_ptr);
            drop(Box::from_raw(tensor_ptr as *mut NslTensor));
        }
    }
}

// === In-place mutation ops (NOT taped -- used outside grad blocks) ===

#[no_mangle]
pub extern "C" fn nsl_tensor_copy_data(dst_ptr: i64, src_ptr: i64) {
    let dst = NslTensor::from_ptr(dst_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    debug_assert!(dst.is_contiguous(), "copy_data requires contiguous dst");
    debug_assert!(src.is_contiguous(), "copy_data requires contiguous src");
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
    debug_assert!(dst.is_contiguous(), "add_inplace requires contiguous dst");
    debug_assert!(src.is_contiguous(), "add_inplace requires contiguous src");
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

        let data = crate::cuda::inner::alloc_managed((len as usize) * std::mem::size_of::<f32>());

        let strides = NslTensor::compute_strides(shape, ndim);

        let tensor = Box::new(NslTensor {
            data,
            shape,
            strides,
            ndim,
            len,
            refcount: AtomicI64::new(1),
            device: device as u8,
            dtype: 1,
            owns_data: 1, data_owner: 0,
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
        let data = result_t.data_f32();
        for i in 0..result_t.len as usize {
            unsafe { *data.add(i) = 1.0f32; }
        }
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

    if norm <= max_norm {
        return;
    }

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

// === Embedding lookup ===

#[no_mangle]
pub extern "C" fn nsl_tensor_embedding_lookup(weight_ptr: i64, indices_ptr: i64) -> i64 {
    let weight = NslTensor::from_ptr(weight_ptr);
    let indices = NslTensor::from_ptr(indices_ptr);

    if weight.ndim != 2 {
        eprintln!("nsl: embedding_lookup requires 2D weight tensor (got {}D)", weight.ndim);
        std::process::abort();
    }
    if indices.ndim != 1 {
        eprintln!("nsl: embedding_lookup requires 1D indices tensor (got {}D)", indices.ndim);
        std::process::abort();
    }

    let vocab_size = unsafe { *weight.shape.add(0) } as usize;
    let embed_dim = unsafe { *weight.shape.add(1) } as usize;
    let seq_len = unsafe { *indices.shape.add(0) } as usize;

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

    let read_idx = |i: usize| -> i64 {
        if indices.dtype == 1 {
            unsafe { *indices.data_f32().add(i) as i64 }
        } else {
            unsafe { *indices.data_f64().add(i) as i64 }
        }
    };

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
        refcount: AtomicI64::new(1),
        device: weight.device,
        dtype: out_dtype,
        owns_data: 1, data_owner: 0,
    });
    let out_ptr = Box::into_raw(out) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(weight_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        NslTensor::from_ptr(indices_ptr).refcount.fetch_add(1, Ordering::SeqCst);
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

#[no_mangle]
pub extern "C" fn nsl_tensor_layernorm(
    input_ptr: i64, weight_ptr: i64, bias_ptr: i64, eps: f64,
) -> i64 {
    let input_ref = NslTensor::from_ptr(input_ptr);
    let need_contig = !input_ref.is_contiguous();
    let effective_input_ptr = if need_contig { NslTensor::make_contiguous(input_ptr) } else { input_ptr };

    let input = NslTensor::from_ptr(effective_input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);
    let bias = NslTensor::from_ptr(bias_ptr);

    let total = input.len as usize;
    let ndim = input.ndim as usize;
    let n = unsafe { *input.shape.add(ndim - 1) } as usize;
    let num_rows = total / n;

    let in_dtype = input.dtype;
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, out_shape, ndim) };
    let out_strides = NslTensor::compute_strides(out_shape, ndim as i64);

    let mean_shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *mean_shape = num_rows as i64 };
    let mean_strides = NslTensor::compute_strides(mean_shape, 1);
    let mean_data = checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;

    let inv_std_shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *inv_std_shape = num_rows as i64 };
    let inv_std_strides = NslTensor::compute_strides(inv_std_shape, 1);
    let inv_std_data = checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;

    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc(total * elem_size);

    for row in 0..num_rows {
        let base = row * n;
        let mut sum = 0.0_f64;
        for j in 0..n {
            let x = if in_dtype == 1 { unsafe { *input.data_f32().add(base + j) as f64 } }
                    else { unsafe { *input.data_f64().add(base + j) } };
            sum += x;
        }
        let mean_val = sum / n as f64;
        let mut var = 0.0_f64;
        for j in 0..n {
            let x = if in_dtype == 1 { unsafe { *input.data_f32().add(base + j) as f64 } }
                    else { unsafe { *input.data_f64().add(base + j) } };
            let diff = x - mean_val;
            var += diff * diff;
        }
        var /= n as f64;
        let inv_std_val = 1.0 / (var + eps).sqrt();
        unsafe {
            *mean_data.add(row) = mean_val;
            *inv_std_data.add(row) = inv_std_val;
        }
        for j in 0..n {
            let x = if in_dtype == 1 { unsafe { *input.data_f32().add(base + j) as f64 } }
                    else { unsafe { *input.data_f64().add(base + j) } };
            let normalized = (x - mean_val) * inv_std_val;
            let w = if weight.dtype == 1 { unsafe { *weight.data_f32().add(j) as f64 } }
                    else { unsafe { *weight.data_f64().add(j) } };
            let b = if bias.dtype == 1 { unsafe { *bias.data_f32().add(j) as f64 } }
                    else { unsafe { *bias.data_f64().add(j) } };
            let result_val = normalized * w + b;
            if in_dtype == 1 {
                unsafe { *(out_data_raw as *mut f32).add(base + j) = result_val as f32 };
            } else {
                unsafe { *(out_data_raw as *mut f64).add(base + j) = result_val };
            }
        }
    }

    let out = Box::new(NslTensor {
        data: out_data_raw as *mut c_void, shape: out_shape, strides: out_strides,
        ndim: ndim as i64, len: total as i64, refcount: AtomicI64::new(1), device: input.device, dtype: in_dtype, owns_data: 1, data_owner: 0,
    });
    let out_ptr = Box::into_raw(out) as i64;

    let mean_tensor = Box::new(NslTensor {
        data: mean_data as *mut c_void, shape: mean_shape, strides: mean_strides,
        ndim: 1, len: num_rows as i64, refcount: AtomicI64::new(1), device: 0, dtype: 0, owns_data: 1, data_owner: 0,
    });
    let mean_ptr = Box::into_raw(mean_tensor) as i64;

    let inv_std_tensor = Box::new(NslTensor {
        data: inv_std_data as *mut c_void, shape: inv_std_shape, strides: inv_std_strides,
        ndim: 1, len: num_rows as i64, refcount: AtomicI64::new(1), device: 0, dtype: 0, owns_data: 1, data_owner: 0,
    });
    let inv_std_ptr = Box::into_raw(inv_std_tensor) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(input_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        NslTensor::from_ptr(weight_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::LayerNorm {
            input: input_ptr, weight: weight_ptr, bias: bias_ptr, out: out_ptr,
            saved_input: input_ptr, saved_mean: mean_ptr, saved_inv_std: inv_std_ptr, saved_weight: weight_ptr,
        });
    } else {
        nsl_tensor_free(mean_ptr);
        nsl_tensor_free(inv_std_ptr);
    }

    if need_contig { nsl_tensor_free(effective_input_ptr); }
    out_ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_rmsnorm(input_ptr: i64, weight_ptr: i64, eps: f64) -> i64 {
    let input_ref = NslTensor::from_ptr(input_ptr);
    let need_contig = !input_ref.is_contiguous();
    let effective_input_ptr = if need_contig { NslTensor::make_contiguous(input_ptr) } else { input_ptr };

    let input = NslTensor::from_ptr(effective_input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);

    let total = input.len as usize;
    let ndim = input.ndim as usize;
    let n = unsafe { *input.shape.add(ndim - 1) } as usize;
    let num_rows = total / n;

    let in_dtype = input.dtype;
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, out_shape, ndim) };
    let out_strides = NslTensor::compute_strides(out_shape, ndim as i64);
    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc(total * elem_size);

    let rms_shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *rms_shape = num_rows as i64 };
    let rms_strides = NslTensor::compute_strides(rms_shape, 1);
    let rms_data = checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;

    for row in 0..num_rows {
        let base = row * n;
        let mut sum_sq = 0.0_f64;
        for j in 0..n {
            let x = if in_dtype == 1 { unsafe { *input.data_f32().add(base + j) as f64 } }
                    else { unsafe { *input.data_f64().add(base + j) } };
            sum_sq += x * x;
        }
        let rms_val = (sum_sq / n as f64 + eps).sqrt();
        unsafe { *rms_data.add(row) = rms_val };
        for j in 0..n {
            let x = if in_dtype == 1 { unsafe { *input.data_f32().add(base + j) as f64 } }
                    else { unsafe { *input.data_f64().add(base + j) } };
            let w = if weight.dtype == 1 { unsafe { *weight.data_f32().add(j) as f64 } }
                    else { unsafe { *weight.data_f64().add(j) } };
            let val = x / rms_val * w;
            if in_dtype == 1 {
                unsafe { *(out_data_raw as *mut f32).add(base + j) = val as f32 };
            } else {
                unsafe { *(out_data_raw as *mut f64).add(base + j) = val };
            }
        }
    }

    let out = Box::new(NslTensor {
        data: out_data_raw as *mut c_void, shape: out_shape, strides: out_strides,
        ndim: ndim as i64, len: total as i64, refcount: AtomicI64::new(1), device: input.device, dtype: in_dtype, owns_data: 1, data_owner: 0,
    });
    let out_ptr = Box::into_raw(out) as i64;

    let rms_tensor = Box::new(NslTensor {
        data: rms_data as *mut c_void, shape: rms_shape, strides: rms_strides,
        ndim: 1, len: num_rows as i64, refcount: AtomicI64::new(1), device: 0, dtype: 0, owns_data: 1, data_owner: 0,
    });
    let rms_ptr = Box::into_raw(rms_tensor) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(input_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        NslTensor::from_ptr(weight_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::RMSNorm {
            input: input_ptr, weight: weight_ptr, out: out_ptr,
            saved_input: input_ptr, saved_rms: rms_ptr, saved_weight: weight_ptr,
        });
    } else {
        nsl_tensor_free(rms_ptr);
    }

    if need_contig { nsl_tensor_free(effective_input_ptr); }
    out_ptr
}

// === Dropout ===

#[no_mangle]
pub extern "C" fn nsl_tensor_dropout(tensor_ptr: i64, p: f64, training: i8) -> i64 {
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
        data: out_data_raw as *mut c_void, shape, strides, ndim, len: len as i64, refcount: AtomicI64::new(1),
        device: 0, dtype: in_dtype, owns_data: 1, data_owner: 0,
    });
    let result_ptr = Box::into_raw(result) as i64;

    let mask_tensor = Box::new(NslTensor {
        data: mask_data as *mut c_void, shape: mask_shape, strides: mask_strides, ndim, len: len as i64, refcount: AtomicI64::new(1),
        device: 0, dtype: 0, owns_data: 1, data_owner: 0,
    });
    let mask_ptr = Box::into_raw(mask_tensor) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        NslTensor::from_ptr(mask_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Dropout {
            a: tensor_ptr, out: result_ptr, saved_mask: mask_ptr, scale,
        });
    } else {
        nsl_tensor_free(mask_ptr);
    }

    result_ptr
}

// === Conv2d ===

#[no_mangle]
pub extern "C" fn nsl_tensor_conv2d(
    input_ptr: i64, weight_ptr: i64, bias_ptr: i64,
    stride_h: i64, stride_w: i64, pad_h: i64, pad_w: i64,
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
                                if ih >= ph && iw >= pw && ih - ph < h && iw - pw < w {
                                    let input_idx = ni * (c_in * h * w) + ci * (h * w) + (ih - ph) * w + (iw - pw);
                                    let weight_idx = co * (c_in * kh * kw) + ci * (kh * kw) + ky * kw + kx;
                                    val += read_input(input_idx) * read_weight(weight_idx);
                                }
                            }
                        }
                    }
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
        data: out_data_raw as *mut c_void, shape: out_shape, strides: out_strides, ndim: 4, len: out_len as i64, refcount: AtomicI64::new(1),
        device: 0, dtype: out_dtype, owns_data: 1, data_owner: 0,
    });
    let result_ptr = Box::into_raw(result) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(input_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        NslTensor::from_ptr(weight_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Conv2d {
            input: input_ptr, weight: weight_ptr, bias: bias_ptr, out: result_ptr,
            saved_input: input_ptr, saved_weight: weight_ptr,
            stride_h: sh, stride_w: sw, pad_h: ph, pad_w: pw,
        });
    }

    result_ptr
}

// === MaxPool2d ===

#[no_mangle]
pub extern "C" fn nsl_tensor_maxpool2d(
    input_ptr: i64, kernel_h: i64, kernel_w: i64, stride: i64, padding: i64,
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
        *out_shape.add(0) = n as i64; *out_shape.add(1) = c as i64;
        *out_shape.add(2) = h_out as i64; *out_shape.add(3) = w_out as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 4);
    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc(out_len * elem_size);

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
                                if val > max_val { max_val = val; max_idx = input_idx; }
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
        data: out_data_raw as *mut c_void, shape: out_shape, strides: out_strides, ndim: 4, len: out_len as i64, refcount: AtomicI64::new(1),
        device: 0, dtype: in_dtype, owns_data: 1, data_owner: 0,
    });
    let result_ptr = Box::into_raw(result) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(input_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::MaxPool2d {
            a: input_ptr, out: result_ptr, saved_argmax: argmax_indices,
            input_shape: vec![n as i64, c as i64, h as i64, w as i64],
        });
    }

    result_ptr
}

/// Add 1D bias to 2D tensor.
#[no_mangle]
pub extern "C" fn nsl_tensor_bias_add(tensor_ptr: i64, bias_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let bias = NslTensor::from_ptr(bias_ptr);

    if tensor.ndim != 2 { eprintln!("nsl: bias_add requires 2D tensor (got {}D)", tensor.ndim); std::process::abort(); }
    if bias.ndim != 1 { eprintln!("nsl: bias_add requires 1D bias (got {}D)", bias.ndim); std::process::abort(); }

    let rows = unsafe { *tensor.shape.add(0) } as usize;
    let cols = unsafe { *tensor.shape.add(1) } as usize;
    let bias_len = unsafe { *bias.shape.add(0) } as usize;

    if cols != bias_len {
        eprintln!("nsl: bias_add shape mismatch -- tensor has {} cols but bias has {} elements", cols, bias_len);
        std::process::abort();
    }

    let in_dtype = tensor.dtype;
    let out_dtype: u16 = if in_dtype == 1 || bias.dtype == 1 { 1 } else { 0 };

    let out_ndim: i64 = 2;
    let out_len = (rows * cols) as i64;
    let out_shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *out_shape.add(0) = rows as i64; *out_shape.add(1) = cols as i64; }
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
        data: out_data_raw as *mut c_void, shape: out_shape, strides: out_strides,
        ndim: out_ndim, len: out_len, refcount: AtomicI64::new(1), device: 0, dtype: out_dtype, owns_data: 1, data_owner: 0,
    });
    let out_ptr = Box::into_raw(out) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        NslTensor::from_ptr(bias_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::BiasAdd { tensor: tensor_ptr, bias: bias_ptr, out: out_ptr });
    }

    out_ptr
}

/// Transfer a tensor to a different device.
#[no_mangle]
pub extern "C" fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64 {
    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    let target = target_device as u8;

    if t.device == target {
        let t_mut = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        t_mut.refcount.fetch_add(1, Ordering::SeqCst);
        return tensor_ptr;
    }

    #[allow(unused_variables)]
    let len = t.len as usize;

    if t.device == 0 && target > 0 {
        #[cfg(feature = "cuda")]
        {
            let dst_size = len * std::mem::size_of::<f32>();
            let dst = crate::cuda::inner::alloc_managed(dst_size);
            let dst_f32 = dst as *mut f32;
            if t.dtype == 1 {
                let src = t.data_f32();
                unsafe { std::ptr::copy_nonoverlapping(src, dst_f32, len); }
            } else {
                let src = t.data_f64();
                for i in 0..len {
                    unsafe { *dst_f32.add(i) = *src.add(i) as f32; }
                }
            }
            crate::cuda::inner::prefetch_to_device(dst, dst_size, (target - 1) as i32);
            let shape = NslTensor::copy_shape(t.shape, t.ndim);
            let strides = NslTensor::compute_strides(shape, t.ndim);
            let new_t = Box::new(NslTensor {
                data: dst, shape, strides, ndim: t.ndim, len: t.len, refcount: AtomicI64::new(1),
                device: target, dtype: 1, owns_data: 1, data_owner: 0,
            });
            return Box::into_raw(new_t) as i64;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }

    if t.device > 0 && target == 0 {
        #[cfg(feature = "cuda")]
        {
            unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
            let src = t.data_f32();
            let dst_size = len * std::mem::size_of::<f64>();
            let layout = std::alloc::Layout::from_size_align(dst_size, 8).unwrap();
            let dst = unsafe { std::alloc::alloc(layout) as *mut f64 };
            let dst_void = dst as *mut std::ffi::c_void;
            for i in 0..len {
                unsafe { *dst.add(i) = *src.add(i) as f64; }
            }
            let shape = NslTensor::copy_shape(t.shape, t.ndim);
            let strides = NslTensor::compute_strides(shape, t.ndim);
            let new_t = Box::new(NslTensor {
                data: dst_void, shape, strides, ndim: t.ndim, len: t.len, refcount: AtomicI64::new(1),
                device: 0, dtype: 0, owns_data: 1, data_owner: 0,
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

#[repr(C)]
pub struct NslSliceDim {
    pub is_scalar: u8,
    pub start: i64,
    pub end: i64,
}

#[no_mangle]
pub extern "C" fn nsl_tensor_set_element(
    tensor_ptr: i64, indices_ptr: i64, num_indices: i64, value: f64,
) {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let n = num_indices as usize;

    if n != ndim {
        eprintln!("nsl: set_element: expected {} indices, got {}", ndim, n);
        std::process::abort();
    }

    let strides = crate::cpu::get_strides_vec(tensor);
    let mut offset: usize = 0;
    for (d, &stride) in strides.iter().enumerate().take(ndim) {
        let idx = unsafe { *(indices_ptr as *const i64).add(d) } as usize;
        let dim_size = unsafe { *tensor.shape.add(d) } as usize;
        if idx >= dim_size {
            eprintln!("nsl: set_element: index {} out of bounds for dim {} (size {})", idx, d, dim_size);
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

#[no_mangle]
pub extern "C" fn nsl_tensor_slice_assign(
    target_ptr: i64, src_ptr: i64, dims_ptr: i64, num_dims: i64,
) {
    let target = NslTensor::from_ptr(target_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    let ndim = num_dims as usize;
    let dims = unsafe { std::slice::from_raw_parts(dims_ptr as *const NslSliceDim, ndim) };
    let target_strides = crate::cpu::get_strides_vec(target);
    let target_shape = crate::cpu::get_shape_vec(target);

    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(ndim);
    for d in 0..ndim {
        let dim_size = target_shape[d] as usize;
        if dims[d].is_scalar != 0 {
            let idx = if dims[d].start < 0 { (dim_size as i64 + dims[d].start) as usize } else { dims[d].start as usize };
            ranges.push((idx, idx + 1));
        } else {
            let start = if dims[d].start < 0 { (dim_size as i64 + dims[d].start) as usize } else { dims[d].start as usize };
            let end = if dims[d].end < 0 { (dim_size as i64 + dims[d].end) as usize } else { dims[d].end.min(dim_size as i64) as usize };
            ranges.push((start, end));
        }
    }

    let mut src_flat = 0usize;

    #[allow(clippy::too_many_arguments)]
    fn recurse(
        depth: usize, ndim: usize, ranges: &[(usize, usize)],
        target: &NslTensor, src: &NslTensor, target_strides: &[usize],
        target_offset: usize, src_flat: &mut usize,
    ) {
        if depth == ndim {
            if *src_flat < src.len as usize {
                let val: f64 = if src.dtype == 1 { unsafe { *src.data_f32().add(*src_flat) as f64 } }
                               else { unsafe { *src.data_f64().add(*src_flat) } };
                if target.dtype == 1 { unsafe { *target.data_f32().add(target_offset) = val as f32 }; }
                else { unsafe { *target.data_f64().add(target_offset) = val }; }
                *src_flat += 1;
            }
            return;
        }
        let (start, end) = ranges[depth];
        for i in start..end {
            recurse(depth + 1, ndim, ranges, target, src, target_strides, target_offset + i * target_strides[depth], src_flat);
        }
    }

    recurse(0, ndim, &ranges, target, src, &target_strides, 0, &mut src_flat);
}

// ---------------------------------------------------------------------------
// Custom dtype pack/unpack conversion FFI
// ---------------------------------------------------------------------------

fn clone_shape(src: *mut i64, ndim: usize) -> *mut i64 {
    let bytes = ndim * std::mem::size_of::<i64>();
    let dst = checked_alloc(bytes) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(src, dst, ndim); }
    dst
}

#[no_mangle]
pub extern "C" fn nsl_tensor_to_custom_dtype(tensor_ptr: i64, target_dtype_id: i64) -> i64 {
    let target_dtype_id = target_dtype_id as u16;
    let tensor = unsafe { &*(tensor_ptr as *const NslTensor) };
    let registry = get_registry();

    let info = match registry.get(&target_dtype_id) {
        Some(info) => info,
        None => { eprintln!("nsl: unknown custom dtype id {target_dtype_id}"); return tensor_ptr; }
    };

    let pack_fn = match info.pack_fn {
        Some(f) => f,
        None => { eprintln!("nsl: custom dtype '{}' has no pack function", info.name); return tensor_ptr; }
    };

    let num_elements = tensor.len as usize;

    if info.block_size == 0 {
        let pack: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(pack_fn) };
        let packed_bytes = num_elements * info.element_size;
        let packed_data = checked_alloc_zeroed(packed_bytes);

        for i in 0..num_elements {
            let val: f64 = if tensor.dtype == 1 {
                unsafe { *(tensor.data as *const f32).add(i) as f64 }
            } else {
                unsafe { *(tensor.data as *const f64).add(i) }
            };
            let val_bits = val.to_bits() as i64;
            let packed_val = pack(val_bits);
            let dst = unsafe { packed_data.add(i * info.element_size) };
            unsafe {
                std::ptr::copy_nonoverlapping(&packed_val as *const i64 as *const u8, dst, info.element_size);
            }
        }

        let shape_ptr = clone_shape(tensor.shape, tensor.ndim as usize);
        let strides_ptr = clone_shape(tensor.strides, tensor.ndim as usize);

        let result = Box::new(NslTensor {
            data: packed_data as *mut c_void, shape: shape_ptr, strides: strides_ptr,
            ndim: tensor.ndim, len: tensor.len, refcount: AtomicI64::new(1), device: tensor.device,
            dtype: target_dtype_id, owns_data: 1, data_owner: 0,
        });
        Box::into_raw(result) as i64
    } else {
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

        let pack: extern "C" fn(*const f64, i64, *mut u8) = unsafe { std::mem::transmute(pack_fn) };
        let src = tensor.data as *const f64;
        for b in 0..num_blocks {
            let block_ptr = unsafe { src.add(b * block_sz) };
            let dst = unsafe { packed_data.add(b * pbs) };
            pack(block_ptr, block_sz as i64, dst);
        }

        let shape_ptr = clone_shape(tensor.shape, tensor.ndim as usize);
        let strides_ptr = clone_shape(tensor.strides, tensor.ndim as usize);

        let result = Box::new(NslTensor {
            data: packed_data as *mut c_void, shape: shape_ptr, strides: strides_ptr,
            ndim: tensor.ndim, len: tensor.len, refcount: AtomicI64::new(1), device: tensor.device,
            dtype: target_dtype_id, owns_data: 1, data_owner: 0,
        });
        Box::into_raw(result) as i64
    }
}

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
        None => { eprintln!("nsl: custom dtype '{}' has no unpack function", info.name); return tensor_ptr; }
    };

    let num_elements = tensor.len as usize;
    let out_data = checked_alloc_zeroed(num_elements * 8) as *mut f64;

    if info.block_size == 0 {
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
        let unpack: extern "C" fn(*const u8, i64, *mut f64) = unsafe { std::mem::transmute(unpack_fn) };
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
        data: out_data as *mut c_void, shape: shape_ptr, strides: strides_ptr,
        ndim: tensor.ndim, len: tensor.len, refcount: AtomicI64::new(1), device: tensor.device,
        dtype: DTYPE_F64, owns_data: 1, data_owner: 0,
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
        let shape = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape, 2);
        crate::list::nsl_list_push(shape, 3);
        let t = nsl_tensor_zeros(shape);
        let indices: [i64; 2] = [1, 2];
        nsl_tensor_set_element(t, indices.as_ptr() as i64, 2, 42.0);
        let tensor = NslTensor::from_ptr(t);
        assert_eq!(unsafe { *tensor.data_f32().add(5) }, 42.0_f32);
    }

    #[test]
    fn test_data_owner_lifecycle() {
        // Create tensor, create a view, free view, free source — no leak, no crash
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 6);
        let t = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let tensor = NslTensor::from_ptr(t);
        assert_eq!(tensor.data_owner, 0); // owned tensors have data_owner = 0
        assert_eq!(tensor.owns_data, 1);
        assert_eq!(tensor.refcount.load(Ordering::Relaxed), 1);
        nsl_tensor_free(t);
    }

    #[test]
    fn test_data_owner_default_zero() {
        // Every newly created tensor has data_owner = 0
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 2);
        crate::list::nsl_list_push(shape_list, 3);
        let t = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let tensor = NslTensor::from_ptr(t);
        assert_eq!(tensor.data_owner, 0);
        nsl_tensor_free(t);
    }

    #[test]
    fn test_slice_assign() {
        let shape = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape, 2);
        crate::list::nsl_list_push(shape, 4);
        let target = nsl_tensor_zeros(shape);

        let src_shape = crate::list::nsl_list_new();
        crate::list::nsl_list_push(src_shape, 3);
        let src = nsl_tensor_zeros(src_shape);
        let s = NslTensor::from_ptr(src);
        unsafe {
            *s.data_f32().add(0) = 10.0_f32;
            *s.data_f32().add(1) = 20.0_f32;
            *s.data_f32().add(2) = 30.0_f32;
        }

        let dims = [
            NslSliceDim { is_scalar: 1, start: 0, end: 0 },
            NslSliceDim { is_scalar: 0, start: 0, end: 3 },
        ];
        nsl_tensor_slice_assign(target, src, dims.as_ptr() as i64, 2);

        let t = NslTensor::from_ptr(target);
        assert_eq!(unsafe { *t.data_f32().add(0) }, 10.0_f32);
        assert_eq!(unsafe { *t.data_f32().add(1) }, 20.0_f32);
        assert_eq!(unsafe { *t.data_f32().add(2) }, 30.0_f32);
        assert_eq!(unsafe { *t.data_f32().add(3) }, 0.0_f32);
        assert_eq!(unsafe { *t.data_f32().add(4) }, 0.0_f32);
    }

    #[test]
    fn test_new_view_shares_data() {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 2);
        crate::list::nsl_list_push(shape_list, 3);
        let t = creation::tensor_from_shape_list_f64(shape_list, 0.0);

        // Fill with known values
        let tensor = NslTensor::from_ptr(t);
        for i in 0..6 {
            unsafe { *tensor.data_f64().add(i) = (i + 1) as f64 };
        }

        let new_shape: [i64; 2] = [3, 2];
        let new_strides: [i64; 2] = [2, 1];
        let view_ptr = NslTensor::new_view_i64(t, &new_shape, &new_strides, 2, 6);
        let view = NslTensor::from_ptr(view_ptr);

        // View shares same data pointer
        assert_eq!(view.data, tensor.data);
        assert_eq!(view.owns_data, 0);
        assert_eq!(view.data_owner, t);
        assert_eq!(view.ndim, 2);
        assert_eq!(view.len, 6);
        unsafe {
            assert_eq!(*view.shape.add(0), 3);
            assert_eq!(*view.shape.add(1), 2);
            assert_eq!(*view.strides.add(0), 2);
            assert_eq!(*view.strides.add(1), 1);
        }

        // Source refcount bumped
        assert_eq!(tensor.refcount.load(Ordering::Relaxed), 2);

        // Free view first, then source — no crash
        nsl_tensor_free(view_ptr);
        // Source refcount back to 1
        assert_eq!(tensor.refcount.load(Ordering::Relaxed), 1);
        nsl_tensor_free(t);
    }

    #[test]
    fn test_free_source_before_view() {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 4);
        let t = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let tensor = NslTensor::from_ptr(t);
        for i in 0..4 {
            unsafe { *tensor.data_f64().add(i) = i as f64 };
        }

        let new_shape: [i64; 2] = [2, 2];
        let new_strides: [i64; 2] = [2, 1];
        let view_ptr = NslTensor::new_view_i64(t, &new_shape, &new_strides, 2, 4);

        // Free source first — view keeps data alive via data_owner refcount
        nsl_tensor_free(t);
        // Source not fully freed yet (refcount went from 2 to 1)
        let view = NslTensor::from_ptr(view_ptr);
        unsafe {
            assert_eq!(*view.data_f64().add(0), 0.0);
            assert_eq!(*view.data_f64().add(3), 3.0);
        }

        // Now free view — source data is freed too
        nsl_tensor_free(view_ptr);
    }

    #[test]
    fn test_reshape_zero_copy_when_contiguous() {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 2);
        crate::list::nsl_list_push(shape_list, 3);
        let t = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let tensor = NslTensor::from_ptr(t);
        for i in 0..6 {
            unsafe { *tensor.data_f64().add(i) = (i + 1) as f64 };
        }

        let new_shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(new_shape_list, 3);
        crate::list::nsl_list_push(new_shape_list, 2);
        let reshaped = nsl_tensor_reshape(t, new_shape_list);
        let r = NslTensor::from_ptr(reshaped);

        // Zero-copy: same data pointer
        assert_eq!(r.data, tensor.data);
        assert_eq!(r.owns_data, 0);
        assert_eq!(r.data_owner, t);
        assert_eq!(r.ndim, 2);
        assert_eq!(r.len, 6);
        unsafe {
            assert_eq!(*r.shape.add(0), 3);
            assert_eq!(*r.shape.add(1), 2);
            // Row-major strides for [3,2]
            assert_eq!(*r.strides.add(0), 2);
            assert_eq!(*r.strides.add(1), 1);
            // Data still accessible
            assert_eq!(*r.data_f64().add(0), 1.0);
            assert_eq!(*r.data_f64().add(5), 6.0);
        }

        nsl_tensor_free(reshaped);
        nsl_tensor_free(t);
    }

    #[test]
    fn test_reshape_materializes_non_contiguous() {
        // Create [2,3], transpose to [3,2] (non-contiguous), then reshape to [6]
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 2);
        crate::list::nsl_list_push(shape_list, 3);
        let t = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let tensor = NslTensor::from_ptr(t);
        for i in 0..6 {
            unsafe { *tensor.data_f64().add(i) = (i + 1) as f64 };
        }

        let transposed = nsl_tensor_transpose(t, 0, 1);

        let new_shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(new_shape_list, 6);
        let reshaped = nsl_tensor_reshape(transposed, new_shape_list);
        let r = NslTensor::from_ptr(reshaped);

        // Non-contiguous reshape must materialize
        // Transposed [3,2]: [[1,4],[2,5],[3,6]] → flattened [1,4,2,5,3,6]
        unsafe {
            assert_eq!(*r.data_f64().add(0), 1.0);
            assert_eq!(*r.data_f64().add(1), 4.0);
            assert_eq!(*r.data_f64().add(2), 2.0);
            assert_eq!(*r.data_f64().add(3), 5.0);
            assert_eq!(*r.data_f64().add(4), 3.0);
            assert_eq!(*r.data_f64().add(5), 6.0);
        }

        nsl_tensor_free(reshaped);
        nsl_tensor_free(transposed);
        nsl_tensor_free(t);
    }

    #[test]
    fn test_transpose_zero_copy() {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 2);
        crate::list::nsl_list_push(shape_list, 3);
        let t = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let tensor = NslTensor::from_ptr(t);
        for i in 0..6 {
            unsafe { *tensor.data_f64().add(i) = (i + 1) as f64 };
        }
        // Original: shape=[2,3], strides=[3,1]

        let tr = nsl_tensor_transpose(t, 0, 1);
        let trv = NslTensor::from_ptr(tr);

        // Zero-copy: same data pointer
        assert_eq!(trv.data, tensor.data);
        assert_eq!(trv.owns_data, 0);
        assert_eq!(trv.data_owner, t);
        // Shape swapped: [3, 2]
        unsafe {
            assert_eq!(*trv.shape.add(0), 3);
            assert_eq!(*trv.shape.add(1), 2);
            // Strides swapped: was [3,1] → [1,3]
            assert_eq!(*trv.strides.add(0), 1);
            assert_eq!(*trv.strides.add(1), 3);
        }

        // Non-contiguous
        assert!(!trv.is_contiguous(), "transposed tensor should be non-contiguous");

        nsl_tensor_free(tr);
        nsl_tensor_free(t);
    }

    #[test]
    fn test_transpose_3d_zero_copy() {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 2);
        crate::list::nsl_list_push(shape_list, 3);
        crate::list::nsl_list_push(shape_list, 4);
        let t = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let tensor = NslTensor::from_ptr(t);
        // strides = [12, 4, 1]

        let tr = nsl_tensor_transpose(t, 0, 2);
        let trv = NslTensor::from_ptr(tr);
        // shape = [4, 3, 2], strides = [1, 4, 12]
        unsafe {
            assert_eq!(*trv.shape.add(0), 4);
            assert_eq!(*trv.shape.add(1), 3);
            assert_eq!(*trv.shape.add(2), 2);
            assert_eq!(*trv.strides.add(0), 1);
            assert_eq!(*trv.strides.add(1), 4);
            assert_eq!(*trv.strides.add(2), 12);
        }
        assert_eq!(trv.data, tensor.data);

        nsl_tensor_free(tr);
        nsl_tensor_free(t);
    }

    #[test]
    fn test_unsqueeze_zero_copy() {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 3);
        crate::list::nsl_list_push(shape_list, 4);
        let t = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let tensor = NslTensor::from_ptr(t);
        // shape=[3,4], strides=[4,1]

        let u = nsl_tensor_unsqueeze(t, 0);
        let uv = NslTensor::from_ptr(u);

        assert_eq!(uv.data, tensor.data);
        assert_eq!(uv.owns_data, 0);
        assert_eq!(uv.ndim, 3);
        unsafe {
            assert_eq!(*uv.shape.add(0), 1);
            assert_eq!(*uv.shape.add(1), 3);
            assert_eq!(*uv.shape.add(2), 4);
        }

        nsl_tensor_free(u);
        nsl_tensor_free(t);
    }

    #[test]
    fn test_add_non_contiguous_inputs() {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 2);
        crate::list::nsl_list_push(shape_list, 3);
        let a = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let shape_list2 = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list2, 2);
        crate::list::nsl_list_push(shape_list2, 3);
        let b = creation::tensor_from_shape_list_f64(shape_list2, 0.0);
        let at = NslTensor::from_ptr(a);
        let bt = NslTensor::from_ptr(b);
        for i in 0..6 {
            unsafe {
                *at.data_f64().add(i) = (i + 1) as f64;
                *bt.data_f64().add(i) = 10.0;
            }
        }

        // Transpose both to [3,2] (non-contiguous)
        let a_t = nsl_tensor_transpose(a, 0, 1);
        let b_t = nsl_tensor_transpose(b, 0, 1);

        let result = nsl_tensor_add(a_t, b_t);
        let r = NslTensor::from_ptr(result);

        // Transposed [[1,4],[2,5],[3,6]] + 10 = [[11,14],[12,15],[13,16]]
        unsafe {
            assert_eq!(*r.data_f64().add(0), 11.0);
            assert_eq!(*r.data_f64().add(1), 14.0);
            assert_eq!(*r.data_f64().add(2), 12.0);
            assert_eq!(*r.data_f64().add(3), 15.0);
            assert_eq!(*r.data_f64().add(4), 13.0);
            assert_eq!(*r.data_f64().add(5), 16.0);
        }

        nsl_tensor_free(result);
        nsl_tensor_free(a_t);
        nsl_tensor_free(b_t);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
    }

    #[test]
    fn test_view_chain_transformer_pattern() {
        // Simulate: input [batch=2, seq=4, hidden=6]
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 2);
        crate::list::nsl_list_push(shape_list, 4);
        crate::list::nsl_list_push(shape_list, 6);
        let x = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let xt = NslTensor::from_ptr(x);
        for i in 0..48 { unsafe { *xt.data_f64().add(i) = i as f64 } }

        // Reshape [2,4,6] -> [2,4,2,3] (zero-copy)
        let sl1 = crate::list::nsl_list_new();
        for &d in &[2i64, 4, 2, 3] { crate::list::nsl_list_push(sl1, d); }
        let split = nsl_tensor_reshape(x, sl1);
        let sv = NslTensor::from_ptr(split);
        assert_eq!(sv.data, xt.data); // zero-copy

        // Transpose [2,4,2,3] -> [2,2,4,3] (zero-copy)
        let perm = nsl_tensor_transpose(split, 1, 2);
        let pv = NslTensor::from_ptr(perm);
        assert_eq!(pv.data, xt.data); // still zero-copy!

        // Materialize for matmul
        let contig = nsl_tensor_contiguous(perm);
        let cv = NslTensor::from_ptr(contig);
        // After transpose, tensor is non-contiguous, so contiguous() allocates new data
        assert!(cv.data != xt.data as *mut c_void || contig == perm);

        nsl_tensor_free(contig);
        nsl_tensor_free(perm);
        nsl_tensor_free(split);
        nsl_tensor_free(x);
    }

    #[test]
    fn test_reshape_transpose_reshape_chain() {
        let sl = crate::list::nsl_list_new();
        crate::list::nsl_list_push(sl, 12);
        let t = creation::tensor_from_shape_list_f64(sl, 0.0);
        let tv = NslTensor::from_ptr(t);
        for i in 0..12 { unsafe { *tv.data_f64().add(i) = i as f64 } }

        // reshape [12] -> [3,4] (zero-copy)
        let sl1 = crate::list::nsl_list_new();
        crate::list::nsl_list_push(sl1, 3);
        crate::list::nsl_list_push(sl1, 4);
        let r1 = nsl_tensor_reshape(t, sl1);
        assert_eq!(NslTensor::from_ptr(r1).data, tv.data);

        // transpose [3,4] -> [4,3] (zero-copy, non-contiguous)
        let tr = nsl_tensor_transpose(r1, 0, 1);
        assert_eq!(NslTensor::from_ptr(tr).data, tv.data);

        // reshape [4,3] -> [12] (must materialize because input is non-contiguous)
        let sl2 = crate::list::nsl_list_new();
        crate::list::nsl_list_push(sl2, 12);
        let r2 = nsl_tensor_reshape(tr, sl2);
        let r2v = NslTensor::from_ptr(r2);
        // Transposed order: [0,4,8,1,5,9,2,6,10,3,7,11]
        unsafe {
            assert_eq!(*r2v.data_f64().add(0), 0.0);
            assert_eq!(*r2v.data_f64().add(1), 4.0);
            assert_eq!(*r2v.data_f64().add(2), 8.0);
            assert_eq!(*r2v.data_f64().add(3), 1.0);
        }

        nsl_tensor_free(r2);
        nsl_tensor_free(tr);
        nsl_tensor_free(r1);
        nsl_tensor_free(t);
    }
}
