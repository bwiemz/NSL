//! Tensor module — NslTensor struct, core utilities, and re-exports of all sub-modules.

pub mod creation;
pub mod arithmetic;
pub mod reduction;
pub mod shape_ops;
pub mod activation;
pub mod trig;
pub mod ad_ops;

// Re-export everything from sub-modules so the public API is unchanged.
pub use creation::*;
pub use arithmetic::*;
pub use reduction::*;
pub use shape_ops::*;
pub use activation::*;
pub use trig::*;
pub use ad_ops::*;

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, Ordering};

use crate::autodiff;
use crate::list::{nsl_list_new, nsl_list_push, NslList};
use crate::memory::{checked_alloc, checked_alloc_zeroed, checked_free};

// ---------------------------------------------------------------------------
// FBIP (Functional But In-Place) metrics — enabled by NSL_FBIP_TRACE=1
// ---------------------------------------------------------------------------
static FBIP_REUSE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FBIP_ALLOC_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
pub(crate) fn fbip_record_reuse() {
    FBIP_REUSE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
pub(crate) fn fbip_record_alloc() {
    FBIP_ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Print FBIP statistics. Called at program exit when NSL_FBIP_TRACE=1.
#[no_mangle]
pub extern "C" fn nsl_fbip_report() {
    let reuse = FBIP_REUSE_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let alloc = FBIP_ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let total = reuse + alloc;
    if total > 0 {
        let pct = (reuse as f64 / total as f64) * 100.0;
        eprintln!("FBIP: {reuse}/{total} operations reused in-place ({pct:.1}%)");
    }
}

// ---------------------------------------------------------------------------
// Tensor scope — lightweight arena for training loops
// ---------------------------------------------------------------------------
// Tracks all tensors allocated since scope_begin. scope_end frees any with
// refcount==1 (temporaries). Tensors still referenced (model weights, outputs
// explicitly kept) survive because their refcount > 1.
thread_local! {
    // Using UnsafeCell for zero-overhead scope tracking (no RefCell borrow checks).
    // Safety: single-threaded access guaranteed by thread_local + no reentrancy during push.
    static TENSOR_SCOPE: Cell<*mut Vec<i64>> = const { Cell::new(std::ptr::null_mut()) };
}

/// Begin a new tensor scope. All tensors allocated after this call will be
/// tracked and eligible for cleanup when `nsl_tensor_scope_end` is called.
#[no_mangle]
pub extern "C" fn nsl_tensor_scope_begin() {
    let list = Box::new(Vec::<i64>::with_capacity(256));
    TENSOR_SCOPE.with(|s| s.set(Box::into_raw(list)));
}

/// Register a newly-allocated tensor with the active scope (if any).
#[inline]
pub(crate) fn scope_track(tensor_ptr: i64) {
    TENSOR_SCOPE.with(|s| {
        let ptr = s.get();
        if !ptr.is_null() {
            unsafe { (*ptr).push(tensor_ptr) };
        }
    });
}

/// End the current tensor scope. Frees all tracked tensors that have
/// refcount == 1 (i.e., no one else holds a reference).
#[no_mangle]
pub extern "C" fn nsl_tensor_scope_end(keep: i64) {
    let list_ptr = TENSOR_SCOPE.with(|s| {
        let p = s.get();
        s.set(std::ptr::null_mut()); // disable tracking before freeing
        p
    });
    if list_ptr.is_null() {
        eprintln!("[scope] WARNING: scope_end called with no active scope!");
        return;
    }
    let list = unsafe { *Box::from_raw(list_ptr) };
    let total = list.len();
    let mut freed = 0usize;
    let mut kept = 0usize;
    for ptr in list {
        if ptr == 0 || ptr == keep {
            continue;
        }
        let tensor = NslTensor::from_ptr(ptr);
        let rc = tensor.refcount.load(Ordering::SeqCst);
        if rc <= 1 {
            nsl_tensor_free(ptr);
            freed += 1;
        } else {
            kept += 1;
        }
    }
    if std::env::var("NSL_SCOPE_TRACE").map(|v| v == "1").unwrap_or(false) {
        eprintln!("[scope] tracked={total}, freed={freed}, kept={kept} (refcount>1)");
    }
}

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

/// Magic marker for live NslTensor instances. Spells "NSLT" in ASCII.
pub const TENSOR_MAGIC: u32 = 0x4E534C54;
/// Poison value written into the magic field when a tensor is freed.
pub const TENSOR_FREED: u32 = 0x0000DEAD;

#[repr(C)]
pub struct NslTensor {
    pub(crate) magic: u32,          // MUST be first — 0x4E534C54 ("NSLT") when live, 0x0000DEAD after free
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
    /// 1 = data is an offset into a GPU memory slab (do NOT free individually).
    /// The slab is freed once at program exit via nsl_slab_destroy.
    pub(crate) slab_managed: u8,
    /// Monotonic ID assigned during tape recording (0 = unassigned).
    /// Decouples tensor identity from memory address for safe intermediate cleanup.
    pub(crate) tape_id: i64,
}

// Built-in dtype IDs (match existing u8 values)
pub const DTYPE_F64: u16 = 0;
pub const DTYPE_F32: u16 = 1;
pub const DTYPE_FP16: u16 = 2;
pub const DTYPE_BF16: u16 = 3;
pub const DTYPE_INT8: u16 = 4;
pub const DTYPE_FP8E4M3: u16 = 5;
pub const DTYPE_FP8E5M2: u16 = 6;
pub const DTYPE_U16_TOKEN: u16 = 7;

// Custom dtype IDs start at 256
pub const DTYPE_CUSTOM_START: u16 = 256;

#[inline]
pub(crate) fn assert_elementwise_byte_copy(dtype: u16, op: &str) {
    if dtype < DTYPE_CUSTOM_START {
        return;
    }

    let supported = get_registry().get(&dtype).is_some_and(|info| {
        info.block_size == 0
            || (info.block_size == 1 && info.packed_block_size == info.element_size)
    });

    assert!(
        supported,
        "{op}: block-packed custom dtype {} is not supported by this byte-copy path",
        dtype,
    );
}

#[inline]
pub(crate) fn dtype_element_size(dtype: u16) -> usize {
    match dtype {
        DTYPE_F64 => std::mem::size_of::<f64>(),
        DTYPE_F32 => std::mem::size_of::<f32>(),
        DTYPE_FP16 | DTYPE_BF16 | DTYPE_U16_TOKEN => std::mem::size_of::<u16>(),
        4 => std::mem::size_of::<i32>(),  // i32 token IDs
        id if id >= DTYPE_CUSTOM_START => {
            get_registry().get(&id).map(|info| info.element_size).unwrap_or(1)
        }
        _ => panic!("unknown dtype {}", dtype),
    }
}

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
    /// Construct a new NslTensor, always setting `magic = TENSOR_MAGIC` and `refcount = 1`.
    /// This is the ONLY correct way to create an NslTensor — do not use struct literal syntax.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        data: *mut c_void,
        shape: *mut i64,
        strides: *mut i64,
        ndim: i64,
        len: i64,
        device: u8,
        dtype: u16,
        owns_data: u8,
        data_owner: i64,
    ) -> Self {
        Self {
            magic: TENSOR_MAGIC,
            data,
            shape,
            strides,
            ndim,
            len,
            refcount: AtomicI64::new(1),
            device,
            dtype,
            owns_data,
            data_owner,
            slab_managed: 0,
            tape_id: 0,
        }
    }

    /// Returns `true` if this tensor has the expected live magic marker.
    #[inline]
    #[allow(dead_code)] // intentional API for future safety assertions and debugging
    pub(crate) fn is_valid(&self) -> bool {
        self.magic == TENSOR_MAGIC
    }

    pub(crate) fn from_ptr(ptr: i64) -> &'static mut NslTensor {
        let t = unsafe { &mut *(ptr as *mut NslTensor) };
        debug_assert!(
            t.magic == TENSOR_MAGIC,
            "NslTensor::from_ptr: bad magic 0x{:08X} at ptr 0x{:X} — possible use-after-free or invalid pointer",
            t.magic,
            ptr
        );
        t
    }

    /// Finalize a boxed tensor: convert to raw pointer, register with scope, return i64.
    #[inline]
    pub(crate) fn publish(tensor: Box<NslTensor>) -> i64 {
        let bytes = (tensor.len as usize) * tensor.element_size();
        crate::math::track_alloc(bytes);
        let ptr = Box::into_raw(tensor) as i64;
        scope_track(ptr);
        ptr
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

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn data_i32(&self) -> *mut i32 {
        assert_eq!(self.dtype, 4, "data_i32() called on non-i32 tensor (dtype={})", self.dtype);
        self.data as *mut i32
    }

    #[inline]
    pub(crate) fn read_scalar_as_f64(&self, offset: usize) -> f64 {
        match self.dtype {
            4 => unsafe { *(self.data as *const i32).add(offset) as f64 },
            DTYPE_U16_TOKEN => unsafe { *(self.data as *const u16).add(offset) as f64 },
            1 => unsafe { *self.data_f32().add(offset) as f64 },
            0 => unsafe { *self.data_f64().add(offset) },
            _ => panic!("read_scalar_as_f64() unsupported for dtype {}", self.dtype),
        }
    }

    #[inline]
    pub(crate) fn has_writable_storage(&self) -> bool {
        if self.device != 0 {
            return false;
        }

        if self.owns_data != 0 {
            return true;
        }

        if self.data_owner != 0 {
            return NslTensor::from_ptr(self.data_owner).has_writable_storage();
        }

        false
    }

    #[inline]
    pub(crate) fn write_scalar_from_f64(&self, offset: usize, value: f64) {
        assert!(
            self.has_writable_storage(),
            "write_scalar_from_f64() cannot mutate borrowed tensor storage (dtype={})",
            self.dtype,
        );

        match self.dtype {
            4 => unsafe { *(self.data as *mut i32).add(offset) = value as i32 },
            DTYPE_U16_TOKEN => {
                assert!(
                    value.is_finite()
                        && value.fract() == 0.0
                        && (0.0..=(u16::MAX as f64)).contains(&value),
                    "write_scalar_from_f64() invalid u16 token value {}",
                    value,
                );
                unsafe { *(self.data as *mut u16).add(offset) = value as u16 };
            }
            1 => unsafe { *self.data_f32().add(offset) = value as f32 },
            0 => unsafe { *self.data_f64().add(offset) = value },
            _ => panic!("write_scalar_from_f64() unsupported for dtype {}", self.dtype),
        }
    }

    /// Read element at index `i` as an integer index value.
    /// Handles dtype 0 (f64), 1 (f32), 4 (i32), and internal u16 token buffers.
    /// Used by embedding_lookup, gather, and their backward passes.
    #[inline]
    pub(crate) fn read_index(&self, i: usize) -> i64 {
        match self.dtype {
            DTYPE_U16_TOKEN => unsafe { *(self.data as *const u16).add(i) as i64 },
            4 => unsafe { *(self.data as *const i32).add(i) as i64 },
            1 => unsafe { *(self.data as *const f32).add(i) as i64 },
            0 => unsafe { *(self.data as *const f64).add(i) as i64 },
            _ => panic!("read_index() unsupported for dtype {}", self.dtype),
        }
    }

    #[inline]
    pub(crate) fn element_size(&self) -> usize {
        dtype_element_size(self.dtype)
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

    /// Returns true if this tensor can be safely mutated in-place (FBIP).
    /// Requires: sole owner (refcount==1), owns data (not a view),
    /// no dependent views (data_owner==0), contiguous layout, CPU device,
    /// and autodiff tape is NOT recording (to preserve saved activations).
    #[inline]
    pub(crate) fn can_mutate_inplace(&self) -> bool {
        self.refcount.load(Ordering::Acquire) == 1
            && self.owns_data == 1
            && self.data_owner == 0
            && self.is_contiguous()
            && self.device == 0
            && !autodiff::is_recording()
    }

    /// Returns true if this GPU tensor can be safely mutated in-place (FBIP).
    /// Same logic as CPU but requires device > 0.
    #[inline]
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn can_mutate_inplace_gpu(&self) -> bool {
        self.refcount.load(Ordering::Acquire) == 1
            && self.owns_data == 1
            && self.data_owner == 0
            && self.device > 0
            && !autodiff::is_recording()
    }

    /// Returns true if two tensors have identical shapes.
    #[inline]
    pub(crate) fn shape_eq(&self, other: &NslTensor) -> bool {
        if self.ndim != other.ndim {
            return false;
        }
        for i in 0..self.ndim as usize {
            if unsafe { *self.shape.add(i) != *other.shape.add(i) } {
                return false;
            }
        }
        true
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
        assert_elementwise_byte_copy(tensor.dtype, "nsl_tensor_contiguous");

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

            unsafe {
                std::ptr::copy_nonoverlapping(
                    (tensor.data as *const u8).add(src_offset * elem_size),
                    (new_data as *mut u8).add(flat * elem_size),
                    elem_size,
                );
            }
        }

        let result = Box::new(NslTensor::new(
            new_data as *mut c_void,
            new_shape,
            new_strides,
            tensor.ndim,
            tensor.len,
            tensor.device,
            tensor.dtype,
            1,
            0,
        ));
        NslTensor::publish(result)
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

        let tensor = Box::new(NslTensor::new(
            source.data,
            shape,
            strides,
            ndim,
            len,
            source.device,
            source.dtype,
            0,
            true_owner,
        ));

        NslTensor::publish(tensor)
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

    tensor.read_scalar_as_f64(offset)
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

    tensor.write_scalar_from_f64(offset, value);
}

// === Scalar extraction ===

#[no_mangle]
pub extern "C" fn nsl_tensor_item(tensor_ptr: i64) -> f64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if tensor.len != 1 {
        let shape: Vec<i64> = (0..tensor.ndim as usize).map(|i| unsafe { *tensor.shape.add(i) }).collect();
        eprintln!(
            "nsl: .item() requires a scalar tensor (got {} elements, shape={:?}, ndim={})",
            tensor.len, shape, tensor.ndim
        );
        std::process::abort();
    }
    // Device memory: must copy scalar to host before reading
    if tensor.device > 0 {
        #[cfg(feature = "cuda")]
        {
            unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
            match tensor.dtype {
                4 => {
                    let mut val: i32 = 0;
                    crate::cuda::inner::memcpy_dtoh(
                        &mut val as *mut i32 as *mut std::ffi::c_void,
                        tensor.data,
                        std::mem::size_of::<i32>(),
                    );
                    return val as f64;
                }
                DTYPE_U16_TOKEN => {
                    let mut val: u16 = 0;
                    crate::cuda::inner::memcpy_dtoh(
                        &mut val as *mut u16 as *mut std::ffi::c_void,
                        tensor.data,
                        std::mem::size_of::<u16>(),
                    );
                    return val as f64;
                }
                1 => {
                    let mut val: f32 = 0.0;
                    crate::cuda::inner::memcpy_dtoh(
                        &mut val as *mut f32 as *mut std::ffi::c_void,
                        tensor.data,
                        std::mem::size_of::<f32>(),
                    );
                    return val as f64;
                }
                _ => {
                    let mut val: f64 = 0.0;
                    crate::cuda::inner::memcpy_dtoh(
                        &mut val as *mut f64 as *mut std::ffi::c_void,
                        tensor.data,
                        std::mem::size_of::<f64>(),
                    );
                    return val;
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }
    tensor.read_scalar_as_f64(0)
}

// === Display ===

#[no_mangle]
pub extern "C" fn nsl_tensor_print(tensor_ptr: i64) {
    let tensor = NslTensor::from_ptr(tensor_ptr);

    // Device memory: transfer to CPU for printing
    if tensor.device > 0 {
        let cpu_ptr = nsl_tensor_to_device(tensor_ptr, 0);
        nsl_tensor_print(cpu_ptr);
        nsl_tensor_free(cpu_ptr);
        return;
    }

    if tensor.ndim == 0 {
        if tensor.len > 0 {
            let val = tensor.read_scalar_as_f64(0);
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
    let elem_size = dtype_element_size(dtype);

    print!("[");
    if dim as i64 == ndim - 1 {
        for i in 0..size {
            if i > 0 { print!(", "); }
            let val = match dtype {
                DTYPE_U16_TOKEN => unsafe { *(data as *const u16).add(i * stride) as f64 },
                4 => unsafe { *(data as *const i32).add(i * stride) as f64 },
                1 => unsafe { *(data as *const f32).add(i * stride) as f64 },
                0 => unsafe { *(data as *const f64).add(i * stride) },
                d if d >= DTYPE_CUSTOM_START => {
                    print!("?");
                    continue;
                }
                _ => {
                    print!("?");
                    continue;
                }
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
    if tensor_ptr == 0 {
        eprintln!("nsl: clone called on null tensor");
        std::process::abort();
    }
    debug_assert!(NslTensor::from_ptr(tensor_ptr).is_valid(), "nsl_tensor_clone: invalid tensor");
    // Note: clone always allocates to maintain memory accounting invariants.
    // (FBIP for clone is Phase 2 — requires codegen-level ownership tracking.)
    // Ensure we clone from contiguous data so non-contiguous views are handled correctly
    let c_ptr = nsl_tensor_contiguous(tensor_ptr);
    let tensor = NslTensor::from_ptr(c_ptr);
    let ndim = tensor.ndim;
    let len = tensor.len;

    let shape = NslTensor::copy_shape(tensor.shape, ndim);

    let strides = NslTensor::compute_strides(shape, ndim);

    let elem_size = tensor.element_size();
    let data_size = (len as usize) * elem_size;
    let data = if tensor.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let dst = crate::cuda::inner::alloc_managed(data_size);
            // Device-to-device copy (no CPU access to device memory)
            crate::cuda::inner::memcpy_dtod(dst, tensor.data, data_size);
            dst as *mut u8
        }
        #[cfg(not(feature = "cuda"))]
        { checked_alloc(data_size) }
    } else {
        let dst = checked_alloc(data_size);
        unsafe { std::ptr::copy_nonoverlapping(tensor.data as *const u8, dst, data_size) };
        dst
    };

    let result = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        tensor.device,
        tensor.dtype,
        1,
        0,
    ));
    nsl_tensor_free(c_ptr);
    NslTensor::publish(result)
}

/// Increment refcount (used by codegen to protect multi-use variables from runtime FBIP).
#[no_mangle]
pub extern "C" fn nsl_tensor_retain(tensor_ptr: i64) {
    if tensor_ptr == 0 { return; }
    let tensor = NslTensor::from_ptr(tensor_ptr);
    tensor.refcount.fetch_add(1, Ordering::SeqCst);
}

/// Decrement refcount without freeing (paired with nsl_tensor_retain).
#[no_mangle]
pub extern "C" fn nsl_tensor_release(tensor_ptr: i64) {
    if tensor_ptr == 0 { return; }
    let tensor = NslTensor::from_ptr(tensor_ptr);
    tensor.refcount.fetch_sub(1, Ordering::SeqCst);
}

#[no_mangle]
pub extern "C" fn nsl_tensor_free(tensor_ptr: i64) {
    if tensor_ptr == 0 {
        return;
    }
    let (should_free, data_ptr, data_size, shape_ptr, strides_ptr, shape_size, device, owns_data, data_owner, slab_managed) = {
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
            tensor.slab_managed,
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
            } else if owns_data != 0 && slab_managed == 0 {
                // This tensor owns its data and is NOT slab-managed — free the buffer.
                // Slab-managed tensors have data pointing into a shared slab that is
                // freed once at program exit via nsl_slab_destroy.
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
            // Poison magic before drop so use-after-free is caught by from_ptr debug_assert.
            (*(tensor_ptr as *mut NslTensor)).magic = TENSOR_FREED;
            drop(Box::from_raw(tensor_ptr as *mut NslTensor));
        }
    }
}

/// Safe version of nsl_tensor_free that probes the magic field before freeing.
/// Used for Type::Unknown variables where we can't be sure the i64 is a tensor pointer.
#[no_mangle]
pub extern "C" fn nsl_tensor_free_if_valid(ptr: i64) {
    if ptr == 0 { return; }
    if (ptr as u64) < 0x10000 { return; }
    if (ptr as usize) % 8 != 0 { return; }
    let magic = unsafe { *(ptr as *const u32) };
    if magic == TENSOR_MAGIC {
        nsl_tensor_free(ptr);
    }
}

/// Safe version of nsl_tensor_clone that returns the original i64 when the input
/// is not a valid tensor pointer.
#[no_mangle]
pub extern "C" fn nsl_tensor_clone_if_valid(ptr: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }
    if (ptr as u64) < 0x10000 {
        return ptr;
    }
    if (ptr as usize) % 8 != 0 {
        return ptr;
    }
    let magic = unsafe { *(ptr as *const u32) };
    if magic == TENSOR_MAGIC {
        nsl_tensor_clone(ptr)
    } else {
        ptr
    }
}

// === In-place mutation ops (NOT taped -- used outside grad blocks) ===

#[no_mangle]
pub extern "C" fn nsl_tensor_copy_data(dst_ptr: i64, src_ptr: i64) {
    if dst_ptr == 0 || src_ptr == 0 {
        eprintln!("nsl: copy_data called with null ptr (dst={}, src={})", dst_ptr, src_ptr);
        return;
    }
    let dst = NslTensor::from_ptr(dst_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    if dst.data.is_null() || src.data.is_null() {
        eprintln!("nsl: copy_data null data pointer (dst.data={:?}, src.data={:?})", dst.data, src.data);
        return;
    }
    debug_assert!(dst.is_contiguous(), "copy_data requires contiguous dst");
    debug_assert!(src.is_contiguous(), "copy_data requires contiguous src");
    if dst.device == 0 && !dst.has_writable_storage() {
        eprintln!("nsl: copy_data cannot write into borrowed CPU storage");
        std::process::abort();
    }
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
    // Handle device memory: use appropriate copy method
    #[cfg(feature = "cuda")]
    if dst.device > 0 && src.device > 0 {
        crate::cuda::inner::memcpy_dtod(dst.data, src.data, byte_count);
        return;
    } else if dst.device > 0 && src.device == 0 {
        if byte_count == 12_582_912 {
            eprintln!(
                "[nsl] memcpy_htod@copy_data dst_shape={:?} src_shape={:?} dst_device={} src_device={} dst_ptr={:?} src_ptr={:?}",
                get_shape_vec(dst),
                get_shape_vec(src),
                dst.device,
                src.device,
                dst.data,
                src.data
            );
        }
        crate::cuda::inner::memcpy_htod(dst.data, src.data, byte_count);
        return;
    } else if dst.device == 0 && src.device > 0 {
        crate::cuda::inner::memcpy_dtoh(dst.data, src.data, byte_count);
        return;
    }
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
    if dst.device == 0 && !dst.has_writable_storage() {
        eprintln!("nsl: add_inplace cannot write into borrowed CPU storage");
        std::process::abort();
    }
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
    // Device memory: must transfer to CPU, compute, copy back
    #[cfg(feature = "cuda")]
    if dst.device > 0 {
        let dst_cpu = nsl_tensor_to_device(dst_ptr, 0);
        let src_cpu = if src.device > 0 {
            nsl_tensor_to_device(src_ptr, 0)
        } else { src_ptr };
        // CPU add
        let dc = NslTensor::from_ptr(dst_cpu);
        let sc = NslTensor::from_ptr(src_cpu);
        if dc.dtype == 1 {
            for i in 0..dc.len as usize {
                unsafe { *dc.data_f32().add(i) += *sc.data_f32().add(i); }
            }
        } else {
            for i in 0..dc.len as usize {
                unsafe { *dc.data_f64().add(i) += *sc.data_f64().add(i); }
            }
        }
        // Copy result back to device
        let byte_count = (dc.len as usize) * dc.element_size();
        if byte_count == 12_582_912 {
            eprintln!(
                "[nsl] memcpy_htod@add_inplace dst_shape={:?} src_shape={:?} dst_device={} src_device={} dst_ptr={:?} src_ptr={:?}",
                get_shape_vec(dst),
                get_shape_vec(dc),
                dst.device,
                dc.device,
                dst.data,
                dc.data
            );
        }
        crate::cuda::inner::memcpy_htod(dst.data, dc.data, byte_count);
        nsl_tensor_free(dst_cpu);
        if src_cpu != src_ptr { nsl_tensor_free(src_cpu); }
        return;
    }
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
    if tensor.device == 0 && !tensor.has_writable_storage() {
        eprintln!("nsl: zero_inplace cannot write into borrowed CPU storage");
        std::process::abort();
    }
    let byte_count = (tensor.len as usize) * tensor.element_size();
    // Device memory: use memset
    #[cfg(feature = "cuda")]
    if tensor.device > 0 {
        crate::cuda::inner::memset_d8(tensor.data, byte_count);
        return;
    }
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

        let data_size = (len as usize) * std::mem::size_of::<f32>();
        let data = crate::cuda::inner::alloc_managed(data_size);
        // Device memory: must use memset to zero (can't use write_bytes from CPU)
        crate::cuda::inner::memset_d8(data, data_size);

        let strides = NslTensor::compute_strides(shape, ndim);

        let tensor = Box::new(NslTensor::new(
            data,
            shape,
            strides,
            ndim,
            len,
            device as u8,
            1,
            1,
            0,
        ));
        NslTensor::publish(tensor)
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
        let len = result_t.len as usize;
        let byte_size = len * std::mem::size_of::<f32>();
        // Fill via CPU staging buffer then memcpy to device (can't write device ptr from CPU)
        let staging = crate::memory::checked_alloc(byte_size) as *mut f32;
        for i in 0..len {
            unsafe { *staging.add(i) = 1.0f32; }
        }
        if byte_size == 12_582_912 {
            eprintln!(
                "[nsl] memcpy_htod@ones_like shape={:?} device={} dst_ptr={:?} src_ptr={:?}",
                get_shape_vec(result_t),
                result_t.device,
                result_t.data,
                staging
            );
        }
        crate::cuda::inner::memcpy_htod(
            result_t.data,
            staging as *const std::ffi::c_void,
            byte_size,
        );
        unsafe { crate::memory::checked_free(staging as *mut u8, byte_size); }
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

    // Collect grad pointers, transferring GPU tensors to CPU for computation
    let mut cpu_grads: Vec<(i64, bool)> = Vec::with_capacity(num_grads); // (ptr, was_gpu)
    for g in 0..num_grads {
        let tensor_ptr = unsafe { *list.data.add(g) };
        if tensor_ptr == 0 {
            cpu_grads.push((0, false));
            continue;
        }
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if tensor.device > 0 {
            // Transfer to CPU for norm computation
            let cpu_ptr = nsl_tensor_to_device(tensor_ptr, 0);
            cpu_grads.push((cpu_ptr, true));
        } else {
            cpu_grads.push((tensor_ptr, false));
        }
    }

    let mut sum_sq: f64 = 0.0;
    for &(ptr, _) in &cpu_grads {
        if ptr == 0 { continue; }
        let tensor = NslTensor::from_ptr(ptr);
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
        // Free temporary CPU copies
        for &(ptr, was_gpu) in &cpu_grads {
            if was_gpu && ptr != 0 { nsl_tensor_free(ptr); }
        }
        return;
    }

    let scale = max_norm / (norm + 1e-8);

    // Scale the original tensors (GPU or CPU)
    for (g, &(cpu_ptr, was_gpu)) in cpu_grads.iter().enumerate() {
        let tensor_ptr = unsafe { *list.data.add(g) };
        if tensor_ptr == 0 || cpu_ptr == 0 { continue; }
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if was_gpu {
            // Scale the CPU copy, then copy back to GPU
            let cpu_t = NslTensor::from_ptr(cpu_ptr);
            if cpu_t.dtype == 1 {
                let s = scale as f32;
                for i in 0..cpu_t.len as usize {
                    unsafe { *cpu_t.data_f32().add(i) *= s; }
                }
            } else {
                for i in 0..cpu_t.len as usize {
                    unsafe { *cpu_t.data_f64().add(i) *= scale; }
                }
            }
            // Convert the scaled CPU copy back through the standard device transfer path
            // so GPU f32 gradients don't receive an oversized raw f64 memcpy.
            let scaled_gpu = nsl_tensor_to_device(cpu_ptr, tensor.device as i64);
            nsl_tensor_copy_data(tensor_ptr, scaled_gpu);
            nsl_tensor_free(scaled_gpu);
        } else if tensor.dtype == 1 {
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

    // Free temporary CPU copies
    for &(ptr, was_gpu) in &cpu_grads {
        if was_gpu && ptr != 0 { nsl_tensor_free(ptr); }
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

    // GPU path: launch fused embedding kernel when weight is on GPU.
    if weight.device > 0 {
        #[cfg(feature = "cuda")]
        {
            return crate::cuda::gpu_embedding_lookup(weight_ptr, indices_ptr);
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
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

    for i in 0..seq_len {
        let raw_idx = indices.read_index(i);
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

    // Output data was allocated via checked_alloc (CPU heap), so device must be 0.
    // Even though the weight is on GPU, the embedding lookup runs on CPU
    // (reading unified memory is fine from CPU). The output will be transferred
    // to GPU by reconcile_device when the next GPU operation uses it.
    let out = Box::new(NslTensor::new(
        out_data_raw as *mut c_void,
        out_shape,
        out_strides,
        out_ndim,
        out_len,
        0,  // CPU — data is in CPU heap, not CUDA unified memory
        out_dtype,
        1,
        0,
    ));
    let out_ptr = NslTensor::publish(out);

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
    // GPU path: native fused LayerNorm kernel.
    {
        let input_ref = NslTensor::from_ptr(input_ptr);
        if input_ref.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let target_device = input_ref.device as i64;
                // Ensure input is contiguous on GPU
                let c_input = nsl_tensor_contiguous(input_ptr);
                // Ensure gamma/beta are on GPU
                let g_gpu = nsl_tensor_to_device(weight_ptr, target_device);
                let b_gpu = nsl_tensor_to_device(bias_ptr, target_device);
                let result = crate::cuda::gpu_layernorm_f32(c_input, g_gpu, b_gpu, eps as f32);
                nsl_tensor_free(c_input);
                nsl_tensor_free(g_gpu);
                nsl_tensor_free(b_gpu);
                if autodiff::is_recording() {
                    NslTensor::from_ptr(input_ptr).refcount.fetch_add(1, Ordering::SeqCst);
                    NslTensor::from_ptr(weight_ptr).refcount.fetch_add(1, Ordering::SeqCst);
                    // For backward, we need mean/inv_std which are computed on CPU fallback.
                    // Record the tape op with the original pointers; backward will use CPU redirect.
                    let input = NslTensor::from_ptr(input_ptr);
                    let ndim = input.ndim as usize;
                    let n = unsafe { *input.shape.add(ndim - 1) } as usize;
                    let num_rows = (input.len as usize) / n;
                    // Compute mean/inv_std on CPU for the backward pass
                    let cpu_input = nsl_tensor_to_device(input_ptr, 0);
                    let ci = NslTensor::from_ptr(cpu_input);
                    let mean_shape = crate::memory::checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
                    unsafe { *mean_shape = num_rows as i64 };
                    let mean_strides = NslTensor::compute_strides(mean_shape, 1);
                    let mean_data = crate::memory::checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;
                    let inv_std_shape = crate::memory::checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
                    unsafe { *inv_std_shape = num_rows as i64 };
                    let inv_std_strides = NslTensor::compute_strides(inv_std_shape, 1);
                    let inv_std_data = crate::memory::checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;
                    for row in 0..num_rows {
                        let base = row * n;
                        let mut sum = 0.0_f64;
                        for j in 0..n {
                            let x = if ci.dtype == 1 { unsafe { *ci.data_f32().add(base + j) as f64 } }
                                    else { unsafe { *ci.data_f64().add(base + j) } };
                            sum += x;
                        }
                        let mean_val = sum / n as f64;
                        let mut var = 0.0_f64;
                        for j in 0..n {
                            let x = if ci.dtype == 1 { unsafe { *ci.data_f32().add(base + j) as f64 } }
                                    else { unsafe { *ci.data_f64().add(base + j) } };
                            let diff = x - mean_val;
                            var += diff * diff;
                        }
                        var /= n as f64;
                        unsafe {
                            *mean_data.add(row) = mean_val;
                            *inv_std_data.add(row) = 1.0 / (var + eps).sqrt();
                        }
                    }
                    nsl_tensor_free(cpu_input);
                    let mean_tensor = Box::new(NslTensor::new(
                        mean_data as *mut c_void, mean_shape, mean_strides, 1, num_rows as i64, 0, 0, 1, 0,
                    ));
                    let mean_ptr = NslTensor::publish(mean_tensor);
                    let inv_std_tensor = Box::new(NslTensor::new(
                        inv_std_data as *mut c_void, inv_std_shape, inv_std_strides, 1, num_rows as i64, 0, 0, 1, 0,
                    ));
                    let inv_std_ptr = NslTensor::publish(inv_std_tensor);
                    autodiff::maybe_record(autodiff::TapeOp::LayerNorm {
                        input: input_ptr, weight: weight_ptr, bias: bias_ptr, out: result,
                        saved_input: input_ptr, saved_mean: mean_ptr, saved_inv_std: inv_std_ptr, saved_weight: weight_ptr,
                    });
                }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }

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
    let out_shape = NslTensor::copy_shape(input.shape, ndim as i64);
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

    // Output allocated via checked_alloc (CPU heap) — device must be 0
    let out = Box::new(NslTensor::new(
        out_data_raw as *mut c_void,
        out_shape,
        out_strides,
        ndim as i64,
        total as i64,
        0,  // CPU — data is in CPU heap, not CUDA memory
        in_dtype,
        1,
        0,
    ));
    let out_ptr = NslTensor::publish(out);

    let mean_tensor = Box::new(NslTensor::new(
        mean_data as *mut c_void,
        mean_shape,
        mean_strides,
        1,
        num_rows as i64,
        0,
        0,
        1,
        0,
    ));
    let mean_ptr = NslTensor::publish(mean_tensor);

    let inv_std_tensor = Box::new(NslTensor::new(
        inv_std_data as *mut c_void,
        inv_std_shape,
        inv_std_strides,
        1,
        num_rows as i64,
        0,
        0,
        1,
        0,
    ));
    let inv_std_ptr = NslTensor::publish(inv_std_tensor);

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
    // GPU path: native fused RMSNorm kernel.
    {
        let input_ref = NslTensor::from_ptr(input_ptr);
        if input_ref.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let target_device = input_ref.device as i64;
                let c_input = nsl_tensor_contiguous(input_ptr);
                let g_gpu = nsl_tensor_to_device(weight_ptr, target_device);
                let result = crate::cuda::gpu_rmsnorm_f32(c_input, g_gpu, eps as f32);
                nsl_tensor_free(c_input);
                nsl_tensor_free(g_gpu);
                if autodiff::is_recording() {
                    NslTensor::from_ptr(input_ptr).refcount.fetch_add(1, Ordering::SeqCst);
                    NslTensor::from_ptr(weight_ptr).refcount.fetch_add(1, Ordering::SeqCst);
                    // Compute rms on CPU for backward pass
                    let input = NslTensor::from_ptr(input_ptr);
                    let ndim = input.ndim as usize;
                    let n = unsafe { *input.shape.add(ndim - 1) } as usize;
                    let num_rows = (input.len as usize) / n;
                    let cpu_input = nsl_tensor_to_device(input_ptr, 0);
                    let ci = NslTensor::from_ptr(cpu_input);
                    let rms_shape = crate::memory::checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
                    unsafe { *rms_shape = num_rows as i64 };
                    let rms_strides = NslTensor::compute_strides(rms_shape, 1);
                    let rms_data = crate::memory::checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;
                    for row in 0..num_rows {
                        let base = row * n;
                        let mut sum_sq = 0.0_f64;
                        for j in 0..n {
                            let x = if ci.dtype == 1 { unsafe { *ci.data_f32().add(base + j) as f64 } }
                                    else { unsafe { *ci.data_f64().add(base + j) } };
                            sum_sq += x * x;
                        }
                        unsafe { *rms_data.add(row) = (sum_sq / n as f64 + eps).sqrt() };
                    }
                    nsl_tensor_free(cpu_input);
                    let rms_tensor = Box::new(NslTensor::new(
                        rms_data as *mut c_void, rms_shape, rms_strides, 1, num_rows as i64, 0, 0, 1, 0,
                    ));
                    let rms_ptr = NslTensor::publish(rms_tensor);
                    autodiff::maybe_record(autodiff::TapeOp::RMSNorm {
                        input: input_ptr, weight: weight_ptr, out: result,
                        saved_input: input_ptr, saved_rms: rms_ptr, saved_weight: weight_ptr,
                    });
                }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }

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
    let out_shape = NslTensor::copy_shape(input.shape, ndim as i64);
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

    // Output allocated via checked_alloc (CPU heap) — device must be 0
    let out = Box::new(NslTensor::new(
        out_data_raw as *mut c_void,
        out_shape,
        out_strides,
        ndim as i64,
        total as i64,
        0,  // CPU — data is in CPU heap, not CUDA memory
        in_dtype,
        1,
        0,
    ));
    let out_ptr = NslTensor::publish(out);

    let rms_tensor = Box::new(NslTensor::new(
        rms_data as *mut c_void,
        rms_shape,
        rms_strides,
        1,
        num_rows as i64,
        0,
        0,
        1,
        0,
    ));
    let rms_ptr = NslTensor::publish(rms_tensor);

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

    // GPU dispatch: native dropout kernel with per-element PRNG
    {
        let t = NslTensor::from_ptr(tensor_ptr);
        if t.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let c_ptr = nsl_tensor_contiguous(tensor_ptr);
                let (result_ptr, mask_ptr) = crate::cuda::gpu_dropout_f32(c_ptr, p);
                nsl_tensor_free(c_ptr);
                // Record tape op for backward
                if autodiff::is_recording() {
                    // No refcount bump on a — identity-only (tape_id)
                    autodiff::maybe_record(autodiff::TapeOp::Dropout {
                        a: tensor_ptr, out: result_ptr, saved_mask: mask_ptr,
                        scale: 1.0 / (1.0 - p),
                    });
                } else {
                    nsl_tensor_free(mask_ptr);
                }
                return result_ptr;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }

    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len as usize;
    let ndim = a.ndim;
    let in_dtype = a.dtype;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);
    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data_raw = checked_alloc(len * elem_size);

    let mask_shape = NslTensor::copy_shape(a.shape, ndim);
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

    let result = Box::new(NslTensor::new(
        out_data_raw as *mut c_void,
        shape,
        strides,
        ndim,
        len as i64,
        0,
        in_dtype,
        1,
        0,
    ));
    let result_ptr = NslTensor::publish(result);

    let mask_tensor = Box::new(NslTensor::new(
        mask_data as *mut c_void,
        mask_shape,
        mask_strides,
        ndim,
        len as i64,
        0,
        0,
        1,
        0,
    ));
    let mask_ptr = NslTensor::publish(mask_tensor);

    if autodiff::is_recording() {
        // No refcount bump on a — identity-only. Mask still bumped (saved data).
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

    // GPU dispatch: native conv2d kernel
    if input.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let c_input = nsl_tensor_contiguous(input_ptr);
            let result = crate::cuda::gpu_conv2d_f32(
                c_input, weight_ptr, bias_ptr,
                stride_h as u64, stride_w as u64, pad_h as u64, pad_w as u64,
            );
            nsl_tensor_free(c_input);
            // Record tape op on the result (same as CPU path)
            if autodiff::is_recording() {
                NslTensor::from_ptr(input_ptr).refcount.fetch_add(1, Ordering::SeqCst);
                NslTensor::from_ptr(weight_ptr).refcount.fetch_add(1, Ordering::SeqCst);
                let w = NslTensor::from_ptr(weight_ptr);
                let sh = stride_h as usize; let sw = stride_w as usize;
                let ph = pad_h as usize; let pw = pad_w as usize;
                autodiff::maybe_record(autodiff::TapeOp::Conv2d {
                    input: input_ptr, weight: weight_ptr, bias: bias_ptr, out: result,
                    saved_input: input_ptr, saved_weight: weight_ptr,
                    stride_h: sh, stride_w: sw, pad_h: ph, pad_w: pw,
                });
            }
            return result;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }

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

    let result = Box::new(NslTensor::new(
        out_data_raw as *mut c_void,
        out_shape,
        out_strides,
        4,
        out_len as i64,
        0,
        out_dtype,
        1,
        0,
    ));
    let result_ptr = NslTensor::publish(result);

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

    // GPU dispatch: native maxpool2d kernel
    if input.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let c_input = nsl_tensor_contiguous(input_ptr);
            let (result, argmax_vec) = crate::cuda::gpu_maxpool2d_f32(
                c_input, kernel_h as u64, kernel_w as u64, stride as u64, padding as u64,
            );
            nsl_tensor_free(c_input);
            // Record tape op with argmax indices
            if autodiff::is_recording() {
                NslTensor::from_ptr(input_ptr).refcount.fetch_add(1, Ordering::SeqCst);
                let n = unsafe { *input.shape.add(0) } as i64;
                let c = unsafe { *input.shape.add(1) } as i64;
                let h = unsafe { *input.shape.add(2) } as i64;
                let w = unsafe { *input.shape.add(3) } as i64;
                let saved_argmax: Vec<usize> = argmax_vec.iter().map(|&x| x as usize).collect();
                autodiff::maybe_record(autodiff::TapeOp::MaxPool2d {
                    a: input_ptr, out: result, saved_argmax,
                    input_shape: vec![n, c, h, w],
                });
            }
            return result;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }

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

    let result = Box::new(NslTensor::new(
        out_data_raw as *mut c_void,
        out_shape,
        out_strides,
        4,
        out_len as i64,
        0,
        in_dtype,
        1,
        0,
    ));
    let result_ptr = NslTensor::publish(result);

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

    // GPU path: launch fused bias_add kernel when tensor is on GPU.
    if tensor.device > 0 {
        #[cfg(feature = "cuda")]
        {
            return crate::cuda::gpu_bias_add(tensor_ptr, bias_ptr);
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }

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

    let out = Box::new(NslTensor::new(
        out_data_raw as *mut c_void,
        out_shape,
        out_strides,
        out_ndim,
        out_len,
        0,
        out_dtype,
        1,
        0,
    ));
    let out_ptr = NslTensor::publish(out);

    if autodiff::is_recording() {
        // No refcount bumps — identity-only fields (tape_ids after assign_ids)
        autodiff::maybe_record(autodiff::TapeOp::BiasAdd { tensor: tensor_ptr, bias: bias_ptr, out: out_ptr });
    }

    out_ptr
}

/// Transfer a tensor to a different device.
#[no_mangle]
pub extern "C" fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64 {
    debug_assert!(NslTensor::from_ptr(tensor_ptr).is_valid(), "nsl_tensor_to_device: invalid tensor");
    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    let target = target_device as u8;

    if t.device == target {
        let t_mut = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        t_mut.refcount.fetch_add(1, Ordering::SeqCst);
        return tensor_ptr;
    }

    // Preserve broadcast/transpose/reshape-style CPU views on GPU when they still
    // reference the owner's base pointer. This avoids expanding large CPU views
    // just to transfer them, letting downstream GPU code materialize if needed.
    if t.device == 0 && target > 0 && !t.is_contiguous() && t.data_owner != 0 {
        #[cfg(feature = "cuda")]
        {
            let mut root_cpu_ptr = t.data_owner;
            loop {
                let owner = NslTensor::from_ptr(root_cpu_ptr);
                if owner.data_owner == 0 {
                    break;
                }
                root_cpu_ptr = owner.data_owner;
            }

            let root_cpu = NslTensor::from_ptr(root_cpu_ptr);
            if t.data == root_cpu.data {
                let root_gpu_ptr = nsl_tensor_to_device(root_cpu_ptr, target as i64);
                let root_gpu = NslTensor::from_ptr(root_gpu_ptr);
                let shape = NslTensor::copy_shape(t.shape, t.ndim);
                let strides = NslTensor::copy_shape(t.strides, t.ndim);
                let view = Box::new(NslTensor::new(
                    root_gpu.data,
                    shape,
                    strides,
                    t.ndim,
                    t.len,
                    target,
                    root_gpu.dtype,
                    0,
                    root_gpu_ptr,
                ));
                return NslTensor::publish(view);
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("CUDA support not compiled");
        }
    }

    // Materialize views before cross-device copies. Copying a non-contiguous
    // tensor's raw backing storage as if it were row-major corrupts transfers
    // for broadcast/transpose/reshape views produced heavily in source-AD.
    let transfer_src_ptr = if t.is_contiguous() {
        tensor_ptr
    } else {
        nsl_tensor_contiguous(tensor_ptr)
    };
    let transfer_src = unsafe { &*(transfer_src_ptr as *const NslTensor) };

    #[allow(unused_variables)]
    let len = transfer_src.len as usize;

    if t.device == 0 && target > 0 {
        #[cfg(feature = "cuda")]
        {
            if transfer_src.dtype == DTYPE_U16_TOKEN {
                let dst_size = len * std::mem::size_of::<i32>();
                let dst = crate::cuda::inner::alloc_managed(dst_size);
                let staging = checked_alloc(dst_size) as *mut i32;
                let src = transfer_src.data as *const u16;
                for i in 0..len {
                    unsafe { *staging.add(i) = *src.add(i) as i32; }
                }
                crate::cuda::inner::memcpy_htod(dst, staging as *const std::ffi::c_void, dst_size);
                unsafe { checked_free(staging as *mut u8, dst_size); }

                let shape = NslTensor::copy_shape(transfer_src.shape, transfer_src.ndim);
                let strides = NslTensor::compute_strides(shape, transfer_src.ndim);
                let new_t = Box::new(NslTensor::new(
                    dst,
                    shape,
                    strides,
                    transfer_src.ndim,
                    transfer_src.len,
                    target,
                    4,
                    1,
                    0,
                ));
                if transfer_src_ptr != tensor_ptr {
                    nsl_tensor_free(transfer_src_ptr);
                }
                return NslTensor::publish(new_t);
            }

            if transfer_src.dtype != 0 && transfer_src.dtype != 1 {
                assert_elementwise_byte_copy(transfer_src.dtype, "nsl_tensor_to_device");
                let dst_size = len * transfer_src.element_size();
                let dst = crate::cuda::inner::alloc_managed(dst_size);
                crate::cuda::inner::memcpy_htod(dst, transfer_src.data, dst_size);
                let shape = NslTensor::copy_shape(transfer_src.shape, transfer_src.ndim);
                let strides = NslTensor::compute_strides(shape, transfer_src.ndim);
                let new_t = Box::new(NslTensor::new(
                    dst,
                    shape,
                    strides,
                    transfer_src.ndim,
                    transfer_src.len,
                    target,
                    transfer_src.dtype,
                    1,
                    0,
                ));
                if transfer_src_ptr != tensor_ptr {
                    nsl_tensor_free(transfer_src_ptr);
                }
                return NslTensor::publish(new_t);
            }

            let dst_size = len * std::mem::size_of::<f32>();
            let dst = crate::cuda::inner::alloc_managed(dst_size);
            if dst_size == 12_582_912 {
                let shape: Vec<i64> = unsafe {
                    std::slice::from_raw_parts(transfer_src.shape, transfer_src.ndim as usize)
                }
                .to_vec();
                eprintln!(
                    "[nsl] to_device HtoD 12582912 bytes: shape={:?} dtype={} source_device={} contiguous_src={} target_device={} owns_data={} data_owner={} data_ptr={:?}",
                    shape,
                    transfer_src.dtype,
                    transfer_src.device,
                    transfer_src_ptr == tensor_ptr,
                    target,
                    transfer_src.owns_data,
                    transfer_src.data_owner,
                    transfer_src.data
                );
            }
            if transfer_src.dtype == 1 {
                // f32→f32: direct host-to-device copy
                crate::cuda::inner::memcpy_htod(dst, transfer_src.data, dst_size);
            } else {
                // f64→f32: convert on CPU into a temporary heap buffer, then copy to device.
                // Avoid the pinned staging pool here; source-AD backward performs many
                // large conversions of temporary tensors, and simple heap staging is more
                // robust than reusing pinned buffers in this hot path.
                let staging = checked_alloc(dst_size) as *mut f32;
                let src = transfer_src.data_f64();
                for i in 0..len {
                    unsafe { *staging.add(i) = *src.add(i) as f32; }
                }
                crate::cuda::inner::memcpy_htod(dst, staging as *const std::ffi::c_void, dst_size);
                unsafe { checked_free(staging as *mut u8, dst_size); }
            }
            let shape = NslTensor::copy_shape(transfer_src.shape, transfer_src.ndim);
            let strides = NslTensor::compute_strides(shape, transfer_src.ndim);
            let new_t = Box::new(NslTensor::new(
                dst, shape, strides, transfer_src.ndim, transfer_src.len, target, 1, 1, 0,
            ));
            if transfer_src_ptr != tensor_ptr {
                nsl_tensor_free(transfer_src_ptr);
            }
            return NslTensor::publish(new_t);
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }

    if t.device > 0 && target == 0 {
        #[cfg(feature = "cuda")]
        {
            unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

            if transfer_src.dtype != 0 && transfer_src.dtype != 1 {
                assert_elementwise_byte_copy(transfer_src.dtype, "nsl_tensor_to_device");
                let dst_size = len * transfer_src.element_size();
                let dst = checked_alloc(dst_size);
                crate::cuda::inner::memcpy_dtoh(
                    dst as *mut std::ffi::c_void,
                    transfer_src.data,
                    dst_size,
                );
                let shape = NslTensor::copy_shape(transfer_src.shape, transfer_src.ndim);
                let strides = NslTensor::compute_strides(shape, transfer_src.ndim);
                let new_t = Box::new(NslTensor::new(
                    dst as *mut std::ffi::c_void,
                    shape,
                    strides,
                    transfer_src.ndim,
                    transfer_src.len,
                    0,
                    transfer_src.dtype,
                    1,
                    0,
                ));
                if transfer_src_ptr != tensor_ptr {
                    nsl_tensor_free(transfer_src_ptr);
                }
                return NslTensor::publish(new_t);
            }

            // f32 (GPU) → f64 (CPU): copy to a temporary heap buffer, then convert.
            let src_size = len * std::mem::size_of::<f32>();
            let staging = checked_alloc(src_size) as *mut f32;
            crate::cuda::inner::memcpy_dtoh(
                staging as *mut std::ffi::c_void,
                transfer_src.data,
                src_size,
            );
            let dst_size = len * std::mem::size_of::<f64>();
            let dst = checked_alloc(dst_size) as *mut f64;
            for i in 0..len {
                unsafe { *dst.add(i) = *staging.add(i) as f64; }
            }
            unsafe { checked_free(staging as *mut u8, src_size); }
            let shape = NslTensor::copy_shape(transfer_src.shape, transfer_src.ndim);
            let strides = NslTensor::compute_strides(shape, transfer_src.ndim);
            let new_t = Box::new(NslTensor::new(
                dst as *mut std::ffi::c_void,
                shape, strides, transfer_src.ndim, transfer_src.len, 0, 0, 1, 0,
            ));
            if transfer_src_ptr != tensor_ptr {
                nsl_tensor_free(transfer_src_ptr);
            }
            return NslTensor::publish(new_t);
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }

    panic!("GPU-to-GPU transfer not yet supported");
}

/// Prefetch a unified-memory tensor to a GPU device asynchronously.
/// This starts the page migration before the GPU actually accesses the data,
/// reducing first-access latency from page faults. No-op on CPU tensors.
#[no_mangle]
pub extern "C" fn nsl_tensor_prefetch(tensor_ptr: i64, device: i64) {
    if tensor_ptr == 0 {
        return;
    }
    let _t = NslTensor::from_ptr(tensor_ptr);
    #[cfg(feature = "cuda")]
    {
        if device > 0 && !_t.data.is_null() {
            let elem_size = match _t.dtype {
                1 => std::mem::size_of::<f32>(),
                4 => std::mem::size_of::<i32>(),
                _ => std::mem::size_of::<f64>(),
            };
            let size_bytes = _t.len as usize * elem_size;
            if size_bytes > 0 {
                crate::cuda::inner::prefetch_to_device(
                    _t.data,
                    size_bytes,
                    (device - 1) as i32, // NSL device 1 = CUDA device 0
                );
            }
        }
    }
    let _ = device; // suppress unused warning without cuda feature
}

/// Create a tensor whose data points to static .rodata memory (compile-time constant).
/// The tensor has `owns_data = 0` — the data is never freed (it's embedded in the binary).
/// Used by M52 weight constant folding to embed folded tensors.
#[no_mangle]
pub extern "C" fn nsl_tensor_from_static(
    data_ptr: i64,
    shape_list: i64,
    dtype: i64,
) -> i64 {
    let list = crate::list::NslList::from_ptr(shape_list);
    let ndim = list.len;
    let mut len: i64 = 1;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        let dim = unsafe { *list.data.add(i) };
        unsafe { *shape.add(i) = dim };
        len *= dim;
    }
    let strides = NslTensor::compute_strides(shape, ndim);

    // owns_data = 0: static data, never freed
    let tensor = NslTensor::new(
        data_ptr as *mut c_void, shape, strides, ndim, len, 0, dtype as u16, 0, 0,
    );
    NslTensor::publish(Box::new(tensor))
}

/// Create a tensor whose data points into a pre-allocated slab.
/// The tensor is marked `slab_managed = 1` so nsl_tensor_free skips data deallocation.
/// `data_ptr` is the raw pointer (slab_base + offset), `shape_list` is a standard NslList.
#[no_mangle]
pub extern "C" fn nsl_tensor_from_slab(
    data_ptr: i64,
    shape_list: i64,
    device: i64,
    dtype: i64,
) -> i64 {
    let list = crate::list::NslList::from_ptr(shape_list);
    let ndim = list.len;
    let mut len: i64 = 1;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        let dim = unsafe { *list.data.add(i) };
        unsafe { *shape.add(i) = dim };
        len *= dim;
    }
    let strides = NslTensor::compute_strides(shape, ndim);

    let mut tensor = NslTensor::new(
        data_ptr as *mut c_void, shape, strides, ndim, len, device as u8, dtype as u16, 1, 0,
    );
    tensor.slab_managed = 1;
    NslTensor::publish(Box::new(tensor))
}

/// Transfer all tensor fields in a model struct to a target device.
/// Walks i64 fields: if a field is a live NslTensor (magic == TENSOR_MAGIC),
/// transfers it to the requested device.
/// Non-tensor fields (sub-models, scalars) are skipped.
#[no_mangle]
pub extern "C" fn nsl_model_to_device(model_ptr: i64, num_fields: i64, device: i64) {
    if model_ptr == 0 || num_fields <= 0 { return; }
    for i in 0..num_fields as usize {
        let field_addr = (model_ptr as usize + i * 8) as *mut i64;
        let field_val = unsafe { *field_addr };
        if field_val == 0 { continue; }
        // Skip values that look like small integers (not heap pointers)
        if (field_val as u64) < 0x10000 { continue; }

        // Probe: read the NslTensor header fields carefully
        let ptr = field_val as *const NslTensor;
        // Safety check: ensure the pointer is reasonably aligned (8-byte)
        #[allow(clippy::manual_is_multiple_of)]
        if (field_val as usize) % 8 != 0 { continue; }

        let t = unsafe { &*ptr };

        // Definitive tensor check: every live NslTensor has magic == TENSOR_MAGIC.
        let is_tensor = t.magic == TENSOR_MAGIC;

        if is_tensor {
            if t.device as i64 == device { continue; }
            let new_ptr = nsl_tensor_to_device(field_val, device);
            if new_ptr == 0 {
                eprintln!("[nsl] WARNING: failed to transfer tensor field {} to device {}", i, device);
                continue;
            }
            unsafe { *field_addr = new_ptr; }
            nsl_tensor_free(field_val);
        }
        // Non-tensor fields (sub-models, arrays) are skipped.
        // The codegen handles nested models by emitting recursive
        // nsl_model_to_device calls with the correct field counts.
    }

    // Sync after all transfers — surface any deferred errors from GPU copies
    #[cfg(feature = "cuda")]
    if device > 0 {
        unsafe {
            let result = cudarc::driver::sys::cuCtxSynchronize();
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                eprintln!("[nsl] WARNING: CUDA error after model transfer to device {}: {:?}",
                    device, result);
            }
        }
    }
}

/// Recursively collect all tensor parameter pointers from a model struct.
/// Walks the struct fields by probing the magic number on each i64 field.
/// Tensor fields (magic == TENSOR_MAGIC) are added to the result list.
/// Sub-model fields are recursed into using the same probing strategy.
/// FixedArray sub-model elements are walked contiguously.
///
/// `model_ptr`: pointer to the model struct (on CPU heap)
/// `num_fields`: number of i64-sized slots in the struct (total_size / 8)
///
/// Returns an NslList of tensor pointers (only actual Param/Tensor fields).
#[no_mangle]
pub extern "C" fn nsl_collect_model_params(model_ptr: i64, num_fields: i64) -> i64 {
    let result = crate::list::nsl_list_new();
    if model_ptr == 0 || num_fields <= 0 { return result; }
    collect_params_recursive(model_ptr as usize, num_fields as usize, result, 0);
    result
}

/// Maximum recursion depth for sub-model traversal.
/// Real models rarely exceed depth 4 (Model → Block → Attention → Norm).
const MAX_COLLECT_DEPTH: usize = 16;

fn collect_params_recursive(base: usize, num_slots: usize, result: i64, depth: usize) {
    if depth >= MAX_COLLECT_DEPTH { return; }
    for i in 0..num_slots {
        let field_addr = (base + i * 8) as *const i64;
        let field_val = unsafe { *field_addr };
        if field_val == 0 { continue; }
        // Skip values that look like small integers (not heap pointers)
        if (field_val as u64) < 0x10000 { continue; }
        // Alignment check
        #[allow(clippy::manual_is_multiple_of)]
        if (field_val as usize) % 8 != 0 { continue; }

        let ptr = field_val as *const NslTensor;
        let t = unsafe { &*ptr };

        if t.magic == TENSOR_MAGIC {
            // This is a tensor — add it to the param list
            crate::list::nsl_list_push(result, field_val);
        } else {
            // This might be a sub-model struct pointer — probe its first few
            // slots to see if any contain tensors. If the first word is a valid
            // heap pointer that itself contains tensor magic at depth 1, treat
            // the whole struct as a sub-model and recurse.
            //
            // Heuristic: probe up to 128 i64 slots (1KB) from this pointer.
            // Most model structs are < 1KB. This is safe because model structs
            // are always allocated via nsl_alloc on the CPU heap.
            let probe_ptr = field_val as *const i64;
            let mut found_tensor = false;
            // Quick probe: check if first field is a tensor
            let first_val = unsafe { *probe_ptr };
            if first_val != 0 && (first_val as u64) >= 0x10000 && (first_val as usize) % 8 == 0 {
                let inner = first_val as *const NslTensor;
                if unsafe { (*inner).magic } == TENSOR_MAGIC {
                    found_tensor = true;
                }
            }
            if found_tensor {
                // Recurse into sub-model — estimate slot count from a reasonable upper bound.
                // We walk until we hit a null or invalid pointer.
                let mut sub_slots = 0usize;
                for j in 0..128 {
                    let sv = unsafe { *probe_ptr.add(j) };
                    if sv == 0 { sub_slots = j; break; }
                    sub_slots = j + 1;
                }
                if sub_slots > 0 {
                    collect_params_recursive(field_val as usize, sub_slots, result, depth + 1);
                }
            }
        }
    }
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

    tensor.write_scalar_from_f64(offset, value);
}

#[no_mangle]
pub extern "C" fn nsl_tensor_slice_assign(
    target_ptr: i64, src_ptr: i64, dims_ptr: i64, num_dims: i64,
) {
    let target = NslTensor::from_ptr(target_ptr);
    let ndim = num_dims as usize;
    if ndim != target.ndim as usize {
        eprintln!(
            "nsl: slice_assign: expected {} dims, got {}",
            target.ndim,
            ndim,
        );
        std::process::abort();
    }
    let dims = unsafe { std::slice::from_raw_parts(dims_ptr as *const NslSliceDim, ndim) };
    let target_strides = crate::cpu::get_strides_vec(target);
    let target_shape = crate::cpu::get_shape_vec(target);

    let src_cpu_ptr = if NslTensor::from_ptr(src_ptr).device > 0 {
        nsl_tensor_to_device(src_ptr, 0)
    } else {
        src_ptr
    };

    let src_work_ptr = if NslTensor::from_ptr(src_cpu_ptr).is_contiguous() {
        src_cpu_ptr
    } else {
        nsl_tensor_contiguous(src_cpu_ptr)
    };
    let src = NslTensor::from_ptr(src_work_ptr);

    let normalize_index = |raw: i64, dim_size: usize, allow_endpoint: bool| -> usize {
        let dim_i64 = dim_size as i64;
        let normalized = if raw < 0 { dim_i64 + raw } else { raw };
        let upper = if allow_endpoint { dim_i64 } else { dim_i64 - 1 };
        if normalized < 0 || normalized > upper {
            eprintln!(
                "nsl: slice_assign: index {} out of bounds for dim of size {}",
                raw,
                dim_size,
            );
            std::process::abort();
        }
        normalized as usize
    };

    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(ndim);
    for d in 0..ndim {
        let dim_size = target_shape[d] as usize;
        if dims[d].is_scalar != 0 {
            let idx = normalize_index(dims[d].start, dim_size, false);
            ranges.push((idx, idx + 1));
        } else {
            let start = normalize_index(dims[d].start, dim_size, true);
            let end = normalize_index(dims[d].end, dim_size, true);
            ranges.push((start, end));
        }
    }

    let selected_len: usize = ranges.iter().map(|(start, end)| end.saturating_sub(*start)).product();
    if selected_len != src.len as usize {
        eprintln!(
            "nsl: slice_assign: source length {} does not match target slice length {}",
            src.len,
            selected_len,
        );
        std::process::abort();
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
                let val = src.read_scalar_as_f64(*src_flat);
                target.write_scalar_from_f64(target_offset, val);
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

    if src_work_ptr != src_cpu_ptr {
        nsl_tensor_free(src_work_ptr);
    }
    if src_cpu_ptr != src_ptr {
        nsl_tensor_free(src_cpu_ptr);
    }
}

// ---------------------------------------------------------------------------
// Custom dtype pack/unpack conversion FFI
// ---------------------------------------------------------------------------

fn clone_shape(src: *mut i64, ndim: usize) -> *mut i64 {
    if ndim == 0 || src.is_null() {
        return std::ptr::null_mut();
    }
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

        let result = Box::new(NslTensor::new(
            packed_data as *mut c_void,
            shape_ptr,
            strides_ptr,
            tensor.ndim,
            tensor.len,
            tensor.device,
            target_dtype_id,
            1,
            0,
        ));
        NslTensor::publish(result)
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

        let result = Box::new(NslTensor::new(
            packed_data as *mut c_void,
            shape_ptr,
            strides_ptr,
            tensor.ndim,
            tensor.len,
            tensor.device,
            target_dtype_id,
            1,
            0,
        ));
        NslTensor::publish(result)
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

    let result = Box::new(NslTensor::new(
        out_data as *mut c_void,
        shape_ptr,
        strides_ptr,
        tensor.ndim,
        tensor.len,
        tensor.device,
        DTYPE_F64,
        1,
        0,
    ));
    NslTensor::publish(result)
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

    // === FBIP (Functional But In-Place) tests ===

    /// Helper: create a 1D f64 tensor with given values, refcount=1, contiguous, CPU, owned.
    fn make_f64_tensor(values: &[f64]) -> i64 {
        let len = values.len();
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, len as i64);
        let ptr = creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let t = NslTensor::from_ptr(ptr);
        for i in 0..len {
            unsafe { *t.data_f64().add(i) = values[i] };
        }
        ptr
    }

    #[test]
    fn test_fbip_can_mutate_inplace() {
        let ptr = make_f64_tensor(&[1.0, 2.0, 3.0]);
        let t = NslTensor::from_ptr(ptr);
        assert!(t.can_mutate_inplace(), "unique owned contiguous CPU tensor should be mutable in-place");

        // Bump refcount — should no longer be mutable
        t.refcount.fetch_add(1, Ordering::SeqCst);
        assert!(!t.can_mutate_inplace(), "refcount>1 should prevent in-place mutation");
        t.refcount.fetch_sub(1, Ordering::SeqCst);
        nsl_tensor_free(ptr);
    }

    #[test]
    fn test_fbip_relu_inplace() {
        let ptr = make_f64_tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = activation::nsl_tensor_relu(ptr);
        // FBIP: same pointer returned, refcount bumped to 2
        assert_eq!(result, ptr, "relu should reuse tensor when refcount==1");
        let t = NslTensor::from_ptr(result);
        assert_eq!(t.refcount.load(Ordering::Relaxed), 2);
        let vals: Vec<f64> = (0..5).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
        nsl_tensor_free(ptr);    // input ref
        nsl_tensor_free(result); // output ref → frees
    }

    #[test]
    fn test_fbip_relu_alloc_when_shared() {
        let ptr = make_f64_tensor(&[-1.0, 2.0]);
        // Bump refcount to simulate shared reference
        NslTensor::from_ptr(ptr).refcount.fetch_add(1, Ordering::SeqCst);
        let result = activation::nsl_tensor_relu(ptr);
        assert_ne!(result, ptr, "relu should allocate new tensor when refcount>1");
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..2).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![0.0, 2.0]);
        // Original should be untouched
        let orig = NslTensor::from_ptr(ptr);
        assert_eq!(unsafe { *orig.data_f64().add(0) }, -1.0);
        nsl_tensor_free(result);
        NslTensor::from_ptr(ptr).refcount.fetch_sub(1, Ordering::SeqCst);
        nsl_tensor_free(ptr);
    }

    #[test]
    fn test_fbip_exp_inplace() {
        let ptr = make_f64_tensor(&[0.0, 1.0]);
        let result = activation::nsl_tensor_exp(ptr);
        assert_eq!(result, ptr);
        let t = NslTensor::from_ptr(result);
        assert!((unsafe { *t.data_f64().add(0) } - 1.0).abs() < 1e-10);
        assert!((unsafe { *t.data_f64().add(1) } - std::f64::consts::E).abs() < 1e-10);
        nsl_tensor_free(ptr);    // input ref
        nsl_tensor_free(result); // output ref
    }

    #[test]
    fn test_fbip_neg_inplace() {
        let ptr = make_f64_tensor(&[1.0, -2.0, 0.0]);
        let result = arithmetic::nsl_tensor_neg(ptr);
        assert_eq!(result, ptr);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![-1.0, 2.0, 0.0]);
        nsl_tensor_free(ptr);
        nsl_tensor_free(result);
    }

    #[test]
    fn test_fbip_sigmoid_inplace() {
        let ptr = make_f64_tensor(&[0.0]);
        let result = activation::nsl_tensor_sigmoid(ptr);
        assert_eq!(result, ptr);
        let t = NslTensor::from_ptr(result);
        assert!((unsafe { *t.data_f64().add(0) } - 0.5).abs() < 1e-10);
        nsl_tensor_free(ptr);
        nsl_tensor_free(result);
    }

    #[test]
    fn test_fbip_tanh_inplace() {
        let ptr = make_f64_tensor(&[0.0]);
        let result = activation::nsl_tensor_tanh_act(ptr);
        assert_eq!(result, ptr);
        let t = NslTensor::from_ptr(result);
        assert!((unsafe { *t.data_f64().add(0) }).abs() < 1e-10);
        nsl_tensor_free(ptr);
        nsl_tensor_free(result);
    }

    #[test]
    fn test_fbip_add_inplace() {
        let a = make_f64_tensor(&[1.0, 2.0, 3.0]);
        let b = make_f64_tensor(&[10.0, 20.0, 30.0]);
        let result = arithmetic::nsl_tensor_add(a, b);
        assert_eq!(result, a, "add should reuse left operand when shapes match and refcount==1");
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![11.0, 22.0, 33.0]);
        nsl_tensor_free(a);      // input ref
        nsl_tensor_free(result); // output ref
        nsl_tensor_free(b);
    }

    #[test]
    fn test_fbip_add_alloc_when_shared() {
        let a = make_f64_tensor(&[1.0, 2.0]);
        let b = make_f64_tensor(&[10.0, 20.0]);
        NslTensor::from_ptr(a).refcount.fetch_add(1, Ordering::SeqCst);
        let result = arithmetic::nsl_tensor_add(a, b);
        assert_ne!(result, a, "add should allocate new tensor when left has refcount>1");
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..2).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![11.0, 22.0]);
        nsl_tensor_free(result);
        NslTensor::from_ptr(a).refcount.fetch_sub(1, Ordering::SeqCst);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
    }

    #[test]
    fn test_fbip_mul_inplace() {
        let a = make_f64_tensor(&[2.0, 3.0, 4.0]);
        let b = make_f64_tensor(&[10.0, 10.0, 10.0]);
        let result = arithmetic::nsl_tensor_mul(a, b);
        assert_eq!(result, a);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![20.0, 30.0, 40.0]);
        nsl_tensor_free(a);
        nsl_tensor_free(result);
        nsl_tensor_free(b);
    }

    #[test]
    fn test_fbip_div_inplace() {
        let a = make_f64_tensor(&[10.0, 20.0, 30.0]);
        let b = make_f64_tensor(&[2.0, 4.0, 5.0]);
        let result = arithmetic::nsl_tensor_div(a, b);
        assert_eq!(result, a);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![5.0, 5.0, 6.0]);
        nsl_tensor_free(a);
        nsl_tensor_free(result);
        nsl_tensor_free(b);
    }

    #[test]
    fn test_fbip_sub_inplace() {
        let a = make_f64_tensor(&[10.0, 20.0]);
        let b = make_f64_tensor(&[3.0, 7.0]);
        let result = arithmetic::nsl_tensor_sub(a, b);
        assert_eq!(result, a);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..2).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![7.0, 13.0]);
        nsl_tensor_free(a);
        nsl_tensor_free(result);
        nsl_tensor_free(b);
    }

    #[test]
    fn test_clone_always_deep_copies() {
        // Clone always deep copies (FBIP clone deferred to Phase 2 — requires codegen ownership)
        let ptr = make_f64_tensor(&[1.0, 2.0, 3.0]);
        let result = nsl_tensor_clone(ptr);
        assert_ne!(result, ptr, "clone should always deep copy at runtime level");
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
        nsl_tensor_free(result);
        nsl_tensor_free(ptr);
    }

    #[test]
    fn test_fbip_add_scalar_inplace() {
        let ptr = make_f64_tensor(&[1.0, 2.0, 3.0]);
        let result = arithmetic::nsl_tensor_add_scalar(ptr, 10.0);
        assert_eq!(result, ptr);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![11.0, 12.0, 13.0]);
        nsl_tensor_free(ptr);
        nsl_tensor_free(result);
    }

    #[test]
    fn test_fbip_mul_scalar_inplace() {
        let ptr = make_f64_tensor(&[2.0, 3.0, 4.0]);
        let result = arithmetic::nsl_tensor_mul_scalar(ptr, 5.0);
        assert_eq!(result, ptr);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![10.0, 15.0, 20.0]);
        nsl_tensor_free(ptr);
        nsl_tensor_free(result);
    }

    #[test]
    fn test_fbip_chained_ops_inplace() {
        // Test that FBIP works for chained operations:
        // After free(input), refcount drops back to 1 → next op can FBIP
        let ptr = make_f64_tensor(&[-2.0, 3.0]);
        let r1 = activation::nsl_tensor_relu(ptr);
        assert_eq!(r1, ptr, "first op should FBIP");
        nsl_tensor_free(ptr); // drop input ref → refcount=1
        let r2 = activation::nsl_tensor_exp(r1);
        assert_eq!(r2, r1, "second op should also FBIP after input freed");
        nsl_tensor_free(r1); // drop r1 ref → refcount=1
        // Values: relu([-2, 3]) = [0, 3], exp([0, 3]) = [1, e^3]
        let t = NslTensor::from_ptr(r2);
        assert!((unsafe { *t.data_f64().add(0) } - 1.0).abs() < 1e-10);
        assert!((unsafe { *t.data_f64().add(1) } - (3.0_f64).exp()).abs() < 1e-10);
        nsl_tensor_free(r2);
    }

    #[test]
    fn test_fbip_autodiff_prevents_inplace() {
        // When autodiff is recording, FBIP should NOT trigger
        let ptr = make_f64_tensor(&[-1.0, 2.0]);
        let params = crate::list::nsl_list_new();
        crate::autodiff::nsl_tape_start(params);
        let result = activation::nsl_tensor_relu(ptr);
        crate::autodiff::nsl_tape_stop();
        assert_ne!(result, ptr, "FBIP should not trigger during autodiff recording");
        nsl_tensor_free(result);
        nsl_tensor_free(ptr);
    }

    #[test]
    fn test_fbip_gpu_can_mutate_inplace() {
        // CPU tensor (device=0) should NOT pass GPU check
        let ptr = make_f64_tensor(&[1.0, 2.0]);
        let t = NslTensor::from_ptr(ptr);
        assert!(!t.can_mutate_inplace_gpu(), "CPU tensor should fail GPU inplace check");
        nsl_tensor_free(ptr);
    }

    #[test]
    fn test_fbip_gpu_method_requires_device() {
        // Manually construct a tensor-like struct to verify the GPU check logic
        let ptr = make_f64_tensor(&[1.0]);
        let t = NslTensor::from_ptr(ptr);
        // CPU tensor: can_mutate_inplace() should be true, can_mutate_inplace_gpu() false
        assert!(t.can_mutate_inplace(), "CPU tensor with refcount=1 should pass CPU check");
        assert!(!t.can_mutate_inplace_gpu(), "CPU tensor should fail GPU check");
        nsl_tensor_free(ptr);
    }
}

// ---------------------------------------------------------------------------
// Training diagnostics (temporary — remove after flat-loss bug is resolved)
// ---------------------------------------------------------------------------

/// Debug: print gradient and parameter norms for the first N params.
/// Called from codegen before the optimizer step loop.
/// param_list, grads_list: NslList of tensor pointers. step: current global step.
#[no_mangle]
pub extern "C" fn nsl_debug_train_step(
    param_list: i64,
    grads_list: i64,
    step: i64,
) {
    let n_params = crate::list::nsl_list_len(param_list);
    let n_grads = crate::list::nsl_list_len(grads_list);
    if step != 0 { return; } // only print on first step

    eprintln!("[debug] step={} params={} grads={}", step, n_params, n_grads);

    let show = n_params.min(n_grads);
    for i in 0..show {
        let p_ptr = crate::list::nsl_list_get(param_list, i);
        let g_ptr = crate::list::nsl_list_get(grads_list, i);

        let p = NslTensor::from_ptr(p_ptr);
        let g = NslTensor::from_ptr(g_ptr);

        let p_norm = tensor_l2_norm(p);
        let g_norm = tensor_l2_norm(g);
        let p_device = p.device;
        let g_device = g.device;
        let p_len = p.len;
        let g_len = g.len;

        eprintln!(
            "  [{}] param: len={} dev={} norm={:.6}  grad: len={} dev={} norm={:.6}",
            i, p_len, p_device, p_norm, g_len, g_device, g_norm,
        );
    }
}

fn tensor_l2_norm(t: &NslTensor) -> f64 {
    if t.len == 0 || t.data.is_null() { return 0.0; }
    let len = t.len as usize;

    // Must read from correct device
    if t.device > 0 {
        // GPU tensor — copy to CPU for inspection
        let cpu_ptr = nsl_tensor_to_device(t as *const NslTensor as i64, 0);
        let cpu_t = NslTensor::from_ptr(cpu_ptr);
        let norm = tensor_l2_norm(cpu_t);
        nsl_tensor_free(cpu_ptr);
        return norm;
    }

    let mut sum_sq = 0.0_f64;
    match t.dtype {
        1 => { // f32
            let data = t.data as *const f32;
            for i in 0..len {
                let v = unsafe { *data.add(i) } as f64;
                sum_sq += v * v;
            }
        }
        _ => { // f64
            let data = t.data as *const f64;
            for i in 0..len {
                let v = unsafe { *data.add(i) };
                sum_sq += v * v;
            }
        }
    }
    sum_sq.sqrt()
}

/// Release idle GPU memory back to the driver. Called after each training step
/// to prevent the caching allocator from holding stale segments.
#[no_mangle]
pub extern "C" fn nsl_gpu_drain_cache() {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::inner::ensure_context();
        let mut alloc = crate::cuda::caching_allocator::CACHING_ALLOCATOR.lock().unwrap();
        alloc.drain_all();
    }
}

/// Debug: print GPU memory usage.
#[no_mangle]
pub extern "C" fn nsl_debug_gpu_mem(step: i64) {
    if step > 3 { return; }
    #[cfg(feature = "cuda")]
    {
        unsafe {
            crate::cuda::inner::ensure_context();
            let mut free: usize = 0;
            let mut total: usize = 0;
            cudarc::driver::sys::cuMemGetInfo_v2(&mut free, &mut total);
            let used_mb = (total - free) / (1024 * 1024);
            let total_mb = total / (1024 * 1024);
            eprintln!("[gpu-mem] step={} used={}MB / {}MB", step, used_mb, total_mb);
        }
    }
    #[cfg(not(feature = "cuda"))]
    { let _ = step; }
}
