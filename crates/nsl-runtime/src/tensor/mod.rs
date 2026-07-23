//! Tensor module — NslTensor struct, core utilities, and re-exports of all sub-modules.

pub mod creation;
pub mod arithmetic;
pub mod reduction;
pub mod shape_ops;
pub mod activation;
pub mod trig;
pub mod ad_ops;
pub mod fbip_flags;
pub mod precision_cast;
pub mod int8_blockwise;

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

/// ELTLS instrumentation: counts how many times `nsl_tensor_free_if_valid`
/// actually freed a valid tensor. Goal: zero for training hot path after
/// full ELTLS rollout — proves no tensor escaped producer-site tracking.
static NSL_DEBUG_EPILOG_FREE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[inline]
pub(crate) fn fbip_record_reuse() {
    FBIP_REUSE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
pub(crate) fn fbip_record_alloc() {
    FBIP_ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// In-place suppression guard (source-AD primal preservation)
// ---------------------------------------------------------------------------
// Tape-AD blocks FBIP in-place mutation while the tape is recording (see
// `can_mutate_inplace`'s `!autodiff::is_recording()`), so a forward op never
// overwrites a value the backward still needs. Source-AD builds no tape, so
// without an equivalent guard a forward op with a uniquely-owned input (e.g.
// `silu(x)` in `grad(x): sum(silu(x))`) would mutate that input in place —
// corrupting the saved primal for input-reading adjoints (silu/gelu/relu/abs;
// output-reading sigmoid/tanh are unaffected since they read the result).
//
// This depth counter is raised around the source-AD FORWARD pass only (the
// adjoint pass leaves it clear so backward FBIP still reclaims memory). It is a
// counter, not a bool, so nested grad blocks compose. Thread-local to match the
// tape and the single-threaded execution of a compiled grad block.
thread_local! {
    static INPLACE_SUPPRESS_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// True while FBIP in-place mutation is suppressed (source-AD forward pass).
///
/// Note: the guard is intentionally cleared before the adjoint pass, so CCR
/// activation *recompute* during backward runs unsuppressed. That is safe in the
/// common case (recomputed inputs are model params / refcount > 1), but a future
/// `--checkpoint-blocks` path that recomputes a uniquely-owned input feeding an
/// input-reading activation would need the guard raised around the recompute too.
#[inline]
pub(crate) fn inplace_suppressed() -> bool {
    INPLACE_SUPPRESS_DEPTH.with(|c| c.get() > 0)
}

/// Enter (`on != 0`) or leave (`on == 0`) the in-place-suppression scope.
/// Emitted by source-AD around its forward primal lowering. Paired inc/dec so
/// nested grad blocks compose; the leave saturates at zero.
#[no_mangle]
pub extern "C" fn nsl_set_inplace_suppressed(on: i64) {
    INPLACE_SUPPRESS_DEPTH.with(|c| {
        let d = c.get();
        c.set(if on != 0 { d + 1 } else { d.saturating_sub(1) });
    });
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

pub const NSL_TENSOR_DATA_OFFSET: usize = std::mem::offset_of!(NslTensor, data);

// ---------------------------------------------------------------------------
// Canonical built-in dtype IDs — the `NslTensor.dtype` (u16) wire tags.
//
// THIS is the single source of truth for the runtime tensor dtype tag space.
// Other modules that must speak these tags (tensor_parallel::collective,
// codegen::cpdt_precision_exec, dlpack, ...) are pinned to these values by
// compile-time `assert!`s; the golden `dtype_abi_lock` test below fails loudly
// if any tag moves. Add new tags at the next free slot — DO NOT reuse a value.
//
// P0.4 dtype/ABI cleanup (P4 item 16): both historical tag collisions are
// RESOLVED —
//   * i32 token tensors now carry their own tag, `DTYPE_I32` (9). Tag 4 means
//     DTYPE_INT8 and nothing else. The DataLoader, the CPU tensor factory,
//     the *i32 readers in this file, and the i32-index GPU kernels all key on
//     DTYPE_I32. `dtype_element_size(DTYPE_INT8)` is the true int8 width (1 B).
//   * The C API (`NslTensorDesc.dtype`) uses THIS canonical tag space verbatim
//     — no inverted 0=f32/1=f64 convention. `c_api::{capi_dtype_to_nsl,
//     nsl_dtype_to_capi}` survive only as validating identity chokepoints.
// See the `dtype_abi_lock` test, which pins both properties.
// ---------------------------------------------------------------------------
pub const DTYPE_F64: u16 = 0;
pub const DTYPE_F32: u16 = 1;
pub const DTYPE_FP16: u16 = 2;
pub const DTYPE_BF16: u16 = 3;
pub const DTYPE_INT8: u16 = 4;
pub const DTYPE_FP8E4M3: u16 = 5;
pub const DTYPE_FP8E5M2: u16 = 6;
pub const DTYPE_U16_TOKEN: u16 = 7;
pub const DTYPE_U16_SEGMENT: u16 = 8;
pub const DTYPE_I32: u16 = 9;

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
        DTYPE_FP16 | DTYPE_BF16 | DTYPE_U16_TOKEN | DTYPE_U16_SEGMENT => std::mem::size_of::<u16>(),
        DTYPE_INT8 => std::mem::size_of::<i8>(),
        DTYPE_I32 => std::mem::size_of::<i32>(),  // i32 token IDs
        id if id >= DTYPE_CUSTOM_START => {
            get_registry().get(&id).map(|info| info.element_size).unwrap_or(1)
        }
        _ => panic!("unknown dtype {}", dtype),
    }
}

/// Convert IEEE 754 half-precision (f16) bits to f32.
/// Dedicated bit-twiddling fallback so callers can read f16 tensors on
/// hosts where the `half` crate isn't compiled in (e.g. `--features cuda`
/// without `--features interop`). Handles subnormals, NaN, and ±Inf.
#[inline]
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let f_exp = (127 - 15 + 1 - e) << 23;
        let f_mant = (m & 0x3FF) << 13;
        f32::from_bits(sign | f_exp | f_mant)
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        let f_exp = (exp + 127 - 15) << 23;
        f32::from_bits(sign | f_exp | (mant << 13))
    }
}

/// Convert f32 to IEEE 754 half-precision (f16) bits.
///
/// # CFTP v7 follow-on — RTE + NaN preservation
///
/// Delegates to `half::f16::from_f32`, which is IEEE-754 default
/// round-to-nearest-even and bit-identical to PTX `cvt.rn.f16.f32`.
/// This closes adversarial-review findings 2 (CPU vs GPU divergence),
/// 7 (one-sided bias up to one full unit roundoff), 8 (NaN silently
/// flushed to ±Inf — previous implementation tested `exp >= 143` first
/// which covered both finite overflow AND `exp == 255` non-finite),
/// and 10 (denormal range divergence).
///
/// Concrete previously-buggy cases now correct:
/// * `f32::NAN` → quiet f16 NaN (was f16 +Inf 0x7C00)
/// * f32 0x3F80FFFF → 0x3C01 (RTE, was 0x3C00 truncating)
/// * f32 0x387FFFFF → 0x0400 (smallest normal RTE up-round, was 0x03FF)
///
/// Saturates to ±Inf on overflow, flushes to zero on underflow (same
/// IEEE-754 boundary semantics as PTX cvt.rn).
#[inline]
pub(crate) fn f32_to_f16_bits(val: f32) -> u16 {
    half::f16::from_f32(val).to_bits()
}

/// Convert bfloat16 bits to f32 — bf16 is just the top 16 bits of f32.
#[inline]
pub(crate) fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to bf16 bits.
///
/// # CFTP v7 follow-on — RTE + NaN preservation
///
/// Delegates to `half::bf16::from_f32`, which is IEEE-754 default
/// round-to-nearest-even and bit-identical to PTX `cvt.rn.bf16.f32`.
/// This closes adversarial-review findings 2 (CPU vs GPU divergence),
/// 7 (truncation bias), 8 (signaling NaNs whose payload sat in the low
/// 16 mantissa bits collapsed to bf16 +Inf), and 10 (f32 denormals
/// flushed to zero asymmetrically vs GPU).
///
/// Concrete previously-buggy cases now correct:
/// * `f32::from_bits(0x7F800001)` (sNaN with low-16 payload) → quiet
///   bf16 NaN (was bf16 +Inf 0x7F80)
/// * f32 = 0x3F80FFFF → 0x3F81 (RTE, was 0x3F80 truncating)
#[inline]
pub(crate) fn f32_to_bf16_bits(val: f32) -> u16 {
    half::bf16::from_f32(val).to_bits()
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
        let device = tensor.device;
        crate::math::track_alloc(bytes);
        if device > 0 {
            debug_track_gpu_alloc(bytes);
        }
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

    /// Raw pointer to the tensor's u16 storage (bits) for f16/bf16 tensors.
    /// Callers must convert through `f16_bits_to_f32` / `bf16_bits_to_f32`
    /// (or the `half` crate if available) before doing math.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn data_f16_bits(&self) -> *mut u16 {
        assert!(
            self.dtype == DTYPE_FP16 || self.dtype == DTYPE_BF16,
            "data_f16_bits() called on non-16-bit-float tensor (dtype={})",
            self.dtype,
        );
        self.data as *mut u16
    }

    /// Decode the tensor's raw storage into an owned `Vec<f64>` regardless of dtype.
    /// Handles f64, f32, f16, bf16, i32, and u16 token buffers. The returned vec
    /// always has exactly `self.len` elements in row-major order. Intended for
    /// callers (optimizers, CPU ops) that need to consume numeric values without
    /// hard-asserting a specific storage dtype.
    #[allow(dead_code)]
    pub(crate) fn as_f64_owned(&self) -> Vec<f64> {
        let len = self.len as usize;
        let mut out = Vec::with_capacity(len);
        match self.dtype {
            DTYPE_F64 => {
                let p = self.data as *const f64;
                for i in 0..len {
                    out.push(unsafe { *p.add(i) });
                }
            }
            DTYPE_F32 => {
                let p = self.data as *const f32;
                for i in 0..len {
                    out.push(unsafe { *p.add(i) as f64 });
                }
            }
            DTYPE_FP16 => {
                let p = self.data as *const u16;
                for i in 0..len {
                    out.push(f16_bits_to_f32(unsafe { *p.add(i) }) as f64);
                }
            }
            DTYPE_BF16 => {
                let p = self.data as *const u16;
                for i in 0..len {
                    out.push(bf16_bits_to_f32(unsafe { *p.add(i) }) as f64);
                }
            }
            DTYPE_U16_TOKEN => {
                let p = self.data as *const u16;
                for i in 0..len {
                    out.push(unsafe { *p.add(i) as f64 });
                }
            }
            DTYPE_I32 => {
                let p = self.data as *const i32;
                for i in 0..len {
                    out.push(unsafe { *p.add(i) as f64 });
                }
            }
            _ => panic!(
                "as_f64_owned() unsupported for dtype {} (len={})",
                self.dtype, self.len
            ),
        }
        out
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn data_i32(&self) -> *mut i32 {
        assert_eq!(
            self.dtype, DTYPE_I32,
            "data_i32() called on non-i32 tensor (dtype={})", self.dtype,
        );
        self.data as *mut i32
    }

    #[inline]
    pub(crate) fn read_scalar_as_f64(&self, offset: usize) -> f64 {
        match self.dtype {
            DTYPE_I32 => unsafe { *(self.data as *const i32).add(offset) as f64 },
            DTYPE_U16_TOKEN => unsafe { *(self.data as *const u16).add(offset) as f64 },
            DTYPE_FP16 => {
                let bits = unsafe { *(self.data as *const u16).add(offset) };
                f16_bits_to_f32(bits) as f64
            }
            DTYPE_BF16 => {
                let bits = unsafe { *(self.data as *const u16).add(offset) };
                bf16_bits_to_f32(bits) as f64
            }
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
            DTYPE_I32 => unsafe { *(self.data as *mut i32).add(offset) = value as i32 },
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
            DTYPE_FP16 => {
                let bits = f32_to_f16_bits(value as f32);
                unsafe { *(self.data as *mut u16).add(offset) = bits };
            }
            DTYPE_BF16 => {
                let bits = f32_to_bf16_bits(value as f32);
                unsafe { *(self.data as *mut u16).add(offset) = bits };
            }
            1 => unsafe { *self.data_f32().add(offset) = value as f32 },
            0 => unsafe { *self.data_f64().add(offset) = value },
            _ => panic!("write_scalar_from_f64() unsupported for dtype {}", self.dtype),
        }
    }

    /// Read element at index `i` as an integer index value.
    /// Handles f64, f32, DTYPE_I32, and internal u16 token buffers.
    /// Used by embedding_lookup, gather, and their backward passes.
    #[inline]
    pub(crate) fn read_index(&self, i: usize) -> i64 {
        match self.dtype {
            DTYPE_U16_TOKEN => unsafe { *(self.data as *const u16).add(i) as i64 },
            DTYPE_I32 => unsafe { *(self.data as *const i32).add(i) as i64 },
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
            && !inplace_suppressed()
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
            && !inplace_suppressed()
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
                    new_data.add(flat * elem_size),
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

    if tensor.device > 0 {
        eprintln!(
            "nsl: element read on a GPU tensor is not supported; move it to the \
             CPU first (e.g. `t.to(cpu)`) — reading device memory host-side \
             would return garbage"
        );
        std::process::abort();
    }
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

    if tensor.device > 0 {
        eprintln!(
            "nsl: element write on a GPU tensor is not supported; move it to the \
             CPU first (e.g. `t.to(cpu)`) — writing device memory host-side \
             would corrupt it"
        );
        std::process::abort();
    }
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
                DTYPE_I32 => {
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
                DTYPE_I32 => unsafe { *(data as *const i32).add(i * stride) as f64 },
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

/// Free a HOST (device==0) tensor's data buffer, routing PINNED buffers
/// (optimizer-state offload, P0.2) to `cuMemFreeHost` via the pinned
/// registry. Everything else takes the plain heap free. Pinned buffers
/// are driver allocations — `std::alloc::dealloc` on one is UB.
fn free_host_tensor_data(data: *mut c_void, size: usize) {
    #[cfg(feature = "cuda")]
    if crate::cuda::inner::is_pinned(data) {
        crate::cuda::inner::free_pinned(data);
        return;
    }
    unsafe { checked_free(data as *mut u8, size) };
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
        if tensor.device > 0 {
            debug_track_gpu_free((tensor.len as usize) * tensor.element_size());
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
                            // Routes pinned offload buffers to cuMemFreeHost.
                            free_host_tensor_data(owner.data, owner.data_byte_size());
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
                    // Routes pinned offload buffers to cuMemFreeHost.
                    free_host_tensor_data(data_ptr, data_size);
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
    if !(ptr as usize).is_multiple_of(8) { return; }
    let magic = unsafe { *(ptr as *const u32) };
    if magic == TENSOR_MAGIC {
        nsl_tensor_free(ptr);
        NSL_DEBUG_EPILOG_FREE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// ELTLS instrumentation: read the total number of actual frees performed by
/// `nsl_tensor_free_if_valid`. A non-zero value indicates tensors escaping
/// producer-site ownership tracking and being cleaned up by the runtime epilog
/// sweep — the value should trend to zero as ELTLS rollout completes.
#[no_mangle]
pub extern "C" fn nsl_debug_epilog_free_count() -> u64 {
    NSL_DEBUG_EPILOG_FREE_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// ELTLS instrumentation: reset the epilog free counter to zero.
#[no_mangle]
pub extern "C" fn nsl_debug_epilog_free_reset() {
    NSL_DEBUG_EPILOG_FREE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
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
    if !(ptr as usize).is_multiple_of(8) {
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

/// Extract the raw `data` field (device pointer) from an `NslTensor`.
///
/// Used by the train block (`stmt.rs::compile_train_block`) to forward
/// `batch["segment_ids"]` and `batch["doc_starts"]` device pointers to
/// `nsl_packing_metadata_set` per step (CFTP §4.3 activation, spec
/// `2026-05-17-pca-rope-activation-design.md`).
///
/// Returns 0 when `tensor_ptr == 0` (matches the runtime's null-passthrough
/// convention used at other FFI entry points). Returns whatever raw value
/// the `data` field holds otherwise — no validation, same risk class as
/// `csha_tensor_data_ptr`.
#[no_mangle]
pub extern "C" fn nsl_tensor_data_ptr(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    unsafe { (*(tensor_ptr as *const NslTensor)).data as i64 }
}

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
    // Self-copy is a no-op. The optimizer-state offload envelope stages
    // state with `to_device_like`, which on a same-device (CPU) run returns
    // the SAME tensor with a refcount bump — the copy-back then sees
    // dst.data == src.data, and copy_nonoverlapping would be UB.
    if dst.data == src.data {
        return;
    }
    let byte_count = (dst.len as usize) * dst.element_size();
    // Handle device memory: use appropriate copy method
    #[cfg(feature = "cuda")]
    if dst.device > 0 && src.device > 0 {
        crate::cuda::inner::memcpy_dtod(dst.data, src.data, byte_count);
        return;
    } else if dst.device > 0 && src.device == 0 {
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

// ---------------------------------------------------------------------------
// Optimizer-state offload envelope (P0.2): pinned host state, async DtoH
// copy-back on the per-thread transfer stream, and a drain point emitted
// once per optimizer step.
//
// Env kill-switches (read once, cached):
//   NSL_OFFLOAD_SYNC=1     — copy-back degrades to the synchronous path
//                            (nsl_offload_drain becomes a list-flush no-op).
//   NSL_OFFLOAD_PAGEABLE=1 — host state buffers use pageable heap memory
//                            instead of cuMemAllocHost (pinned).
// ---------------------------------------------------------------------------

/// NSL_OFFLOAD_SYNC=1 forces the offload copy-back onto the synchronous
/// path (kill-switch for the P0.2 async overlap).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))] // async path is cuda-only
pub(crate) fn offload_sync_forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| {
        std::env::var("NSL_OFFLOAD_SYNC").map(|v| v == "1").unwrap_or(false)
    })
}

/// NSL_OFFLOAD_PAGEABLE=1 forces pageable host allocation for offloaded
/// optimizer state (kill-switch for the P0.2 pinned upgrade).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))] // pinned path is cuda-only
pub(crate) fn offload_pageable_forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| {
        std::env::var("NSL_OFFLOAD_PAGEABLE").map(|v| v == "1").unwrap_or(false)
    })
}

thread_local! {
    /// Staged device tensors whose async DtoH copy-back is still in flight.
    /// Freed by `nsl_offload_drain` AFTER the transfer stream synchronizes —
    /// freeing them inline would hand the block back to the caching
    /// allocator while the copy still reads it. Thread-local to pair with
    /// the per-thread transfer stream.
    static OFFLOAD_DRAIN_TENSORS: std::cell::RefCell<Vec<i64>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(feature = "cuda")]
thread_local! {
    /// Raw transient DEVICE buffers (from `alloc_managed`, not full tensors)
    /// read by an in-flight async DtoH — the P0.3 quant-cast staging
    /// buffers. Freed via `free_managed` by `nsl_offload_drain` after sync.
    static OFFLOAD_DRAIN_DEVICE_BUFS: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Defer `nsl_tensor_free(ptr)` until the next `nsl_offload_drain()`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))] // enqueued by cuda-only paths
pub(crate) fn offload_defer_free_tensor(ptr: i64) {
    OFFLOAD_DRAIN_TENSORS.with(|l| l.borrow_mut().push(ptr));
}

/// Bound on in-flight drain-deferred device buffers (staged tensors +
/// quant-cast transients). Without a bound, EVERY parameter's staged
/// optimizer state stays device-resident until the single per-step drain
/// — at 1B/f32 that is the full m+v surface (~8 GB), exactly the memory
/// `--optim-state-offload` exists to evict, and a mid-loop OOM cannot
/// reclaim it (the caching allocator's recovery only drains POOLED
/// blocks). 4 buffers ≈ two params' (m, v) — enough pipeline depth for
/// the copy-back of param i to overlap the update of param i+1, with
/// residency bounded to O(2 params) instead of O(all params).
#[cfg(feature = "cuda")]
const OFFLOAD_MAX_INFLIGHT: usize = 4;

/// Inline flush point called BEFORE enqueueing a new async copy-back:
/// when the drain lists are saturated, synchronize the transfer stream
/// and free the deferred buffers. Flushing before (not after) the new
/// enqueue means the just-issued copy never loses its overlap window —
/// the sync only waits on copies that are `OFFLOAD_MAX_INFLIGHT` params
/// old and near-certainly complete.
#[cfg(feature = "cuda")]
pub(crate) fn offload_flush_if_saturated() {
    let inflight = OFFLOAD_DRAIN_TENSORS.with(|l| l.borrow().len())
        + OFFLOAD_DRAIN_DEVICE_BUFS.with(|l| l.borrow().len());
    if inflight >= OFFLOAD_MAX_INFLIGHT {
        nsl_offload_drain();
    }
}

/// Defer `free_managed(buf)` until the next `nsl_offload_drain()`.
#[cfg(feature = "cuda")]
pub(crate) fn offload_defer_free_device_buf(buf: *mut c_void) {
    OFFLOAD_DRAIN_DEVICE_BUFS.with(|l| l.borrow_mut().push(buf as usize));
}

/// Async sibling of `nsl_tensor_copy_data` that CONSUMES `src`.
///
/// Fast path (all of: cuda build, `src` device-resident, `dst` host-resident
/// with a PINNED data buffer, `NSL_OFFLOAD_SYNC` unset): issues the DtoH on
/// the per-thread transfer stream — ordered after the NULL-stream update
/// kernels that produced `src`, overlapping with the next parameter's
/// update — and defers `src`'s free to `nsl_offload_drain()`. The copy is
/// NOT observable in `dst` until the drain runs.
///
/// Fallback (anything else): exactly `nsl_tensor_copy_data(dst, src)`
/// followed by an inline `nsl_tensor_free(src)`.
///
/// Ownership contract: callers replace the emitted pair
/// `copy_data(dst, src); free(src)` with ONE call to this function, and
/// emit `nsl_offload_drain()` once per optimizer step after the per-param
/// loop exits.
#[no_mangle]
pub extern "C" fn nsl_tensor_copy_data_async(dst_ptr: i64, src_ptr: i64) {
    #[cfg(feature = "cuda")]
    if dst_ptr != 0 && src_ptr != 0 && !offload_sync_forced() {
        let dst = NslTensor::from_ptr(dst_ptr);
        let src = NslTensor::from_ptr(src_ptr);
        if src.device > 0
            && dst.device == 0
            && !dst.data.is_null()
            && !src.data.is_null()
            && crate::cuda::inner::is_pinned(dst.data)
        {
            // Same guards as nsl_tensor_copy_data's sync path.
            debug_assert!(dst.is_contiguous(), "copy_data_async requires contiguous dst");
            debug_assert!(src.is_contiguous(), "copy_data_async requires contiguous src");
            assert_eq!(
                dst.len, src.len,
                "nsl_tensor_copy_data_async: dst len {} != src len {}",
                dst.len, src.len
            );
            assert_eq!(
                dst.dtype, src.dtype,
                "nsl_tensor_copy_data_async: dtype mismatch (dst={}, src={})",
                dst.dtype, src.dtype
            );
            // Bound in-flight staged buffers BEFORE enqueueing (review M1:
            // otherwise the whole m+v surface stays device-resident until
            // the per-step drain, defeating the offload).
            offload_flush_if_saturated();
            let byte_count = (dst.len as usize) * dst.element_size();
            crate::cuda::inner::memcpy_dtoh_async(dst.data, src.data, byte_count);
            offload_defer_free_tensor(src_ptr);
            return;
        }
    }
    nsl_tensor_copy_data(dst_ptr, src_ptr);
    nsl_tensor_free(src_ptr);
}

/// Drain point for the offload copy-back: synchronize the calling thread's
/// transfer stream, then free every staged tensor / transient device buffer
/// whose async DtoH was in flight. Emitted by codegen ONCE per optimizer
/// step, after the per-parameter loop exits. Cheap no-op when nothing was
/// enqueued (sync fallback path, CPU runs, non-offload builds).
#[no_mangle]
pub extern "C" fn nsl_offload_drain() {
    #[cfg(feature = "cuda")]
    crate::cuda::inner::transfer_stream_synchronize();
    let tensors = OFFLOAD_DRAIN_TENSORS.with(|l| std::mem::take(&mut *l.borrow_mut()));
    for p in tensors {
        nsl_tensor_free(p);
    }
    #[cfg(feature = "cuda")]
    {
        let bufs = OFFLOAD_DRAIN_DEVICE_BUFS.with(|l| std::mem::take(&mut *l.borrow_mut()));
        for b in bufs {
            crate::cuda::inner::free_managed(b as *mut c_void);
        }
    }
}

/// Allocate a zero-filled host data buffer for offloaded optimizer state:
/// PINNED (`cuMemAllocHost_v2`, tracked in the pinned registry so
/// `nsl_tensor_free` routes it back to `cuMemFreeHost`) when a CUDA
/// context is already live and `NSL_OFFLOAD_PAGEABLE=1` is not set;
/// pageable heap otherwise (silent fallback — a GPU-less run must behave
/// exactly like one without the pinned upgrade).
pub(crate) fn alloc_host_state_buffer(bytes: usize) -> *mut u8 {
    #[cfg(feature = "cuda")]
    if !offload_pageable_forced() && crate::cuda::inner::context_initialized() {
        match crate::cuda::inner::try_alloc_pinned(bytes) {
            Some(p) => {
                // cuMemAllocHost does not zero; offloaded moments must start at 0.
                unsafe { std::ptr::write_bytes(p as *mut u8, 0, bytes) };
                return p as *mut u8;
            }
            None => {
                // Page-lock limits / fragmented host memory: degrade to
                // pageable (copy-back falls to the sync path) instead of
                // aborting a multi-GB run. Warn once per process.
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    eprintln!(
                        "[offload] cuMemAllocHost failed for a {} MB state buffer — \
                         falling back to PAGEABLE host memory (sync copy-back). \
                         Raise the process page-lock limit (ulimit -l) to restore \
                         the pinned/async path.",
                        bytes / (1024 * 1024)
                    );
                });
            }
        }
    }
    crate::memory::checked_alloc_zeroed(bytes)
}

/// Allocate a zero-filled HOST-resident f32 tensor with `template`'s shape,
/// regardless of the template's device (scaling campaign item 4 —
/// optimizer-state offload).
///
/// `nsl_tensor_zeros_like` inherits the template's placement, and the plain
/// CPU creation path allocates f64 (the CPU dtype convention) — neither
/// works for offloaded optimizer state, which must be CPU-resident f32 so
/// (a) `to_device_like` staging is a plain memcpy and (b) the DtoH
/// copy-back passes `nsl_tensor_copy_data`'s dtype-equality assert against
/// the staged GPU f32 working tensor.
#[no_mangle]
pub extern "C" fn nsl_tensor_zeros_like_host_f32(template_ptr: i64) -> i64 {
    let t = NslTensor::from_ptr(template_ptr);
    // CPU-resident template: offload is meaningless (state would sit next
    // to the params anyway) and CPU params are f64 — an f32 state buffer
    // would trip the stdlib step's copy_data dtype assert. Delegate to the
    // plain like-placement allocation so a CPU run with the offload flag
    // behaves exactly like one without it.
    if t.device == 0 {
        return nsl_tensor_zeros_like(template_ptr);
    }
    let ndim = t.ndim;
    let len = t.len as usize;
    let shape = checked_alloc((ndim as usize).max(1) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        unsafe { *shape.add(i) = *t.shape.add(i) };
    }
    let strides = NslTensor::compute_strides(shape, ndim);
    // P0.2: pinned host memory when a CUDA context is live (template is
    // device-resident here, so one always is) — enables true async DMA on
    // the copy-back. NSL_OFFLOAD_PAGEABLE=1 restores the pageable buffer.
    let data = alloc_host_state_buffer(len * std::mem::size_of::<f32>());
    let out = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        len as i64,
        0, // CPU
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_add_inplace(dst_ptr: i64, src_ptr: i64) {
    let dst = NslTensor::from_ptr(dst_ptr);
    {
        // PCA Stage C hardening: reconcile a mismatched src instead of
        // aborting. Legitimate gradients can arrive as transpose VIEWS
        // (non-contiguous) or as CPU-f64 chains on a GPU run (any adjoint
        // op that only has a CPU lowering) — the non-inplace binary ops
        // already reconcile exactly like this. The warn-once keeps the
        // perf smell visible: a converted src on every step means some
        // producer op should grow a device kernel.
        let src_probe = NslTensor::from_ptr(src_ptr);
        if src_probe.device != dst.device
            || src_probe.dtype != dst.dtype
            || !src_probe.is_contiguous()
        {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            // NSL_RECONCILE_DEBUG=1 narrates EVERY reconciliation with the
            // tensor geometry — the once-warn alone cannot identify WHICH
            // producer is emitting mismatched gradients.
            static RECONCILE_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let debug_all = *RECONCILE_DEBUG.get_or_init(|| {
                std::env::var("NSL_RECONCILE_DEBUG").ok().as_deref() == Some("1")
            });
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) || debug_all {
                let dims: Vec<i64> = (0..src_probe.ndim as usize)
                    .map(|i| unsafe { *src_probe.shape.add(i) })
                    .collect();
                eprintln!(
                    "[nsl] add_inplace: reconciling src (device {} -> {}, dtype {} -> {}, \
                     contiguous={}, shape={dims:?}) — repeated reconciliation is a perf \
                     smell (CPU-lowered producer on a GPU run?)",
                    src_probe.device, dst.device, src_probe.dtype, dst.dtype,
                    src_probe.is_contiguous()
                );
            }
            let contig = if src_probe.is_contiguous() {
                src_ptr
            } else {
                nsl_tensor_contiguous(src_ptr)
            };
            // Dtype first: `to_device_like` converts dtype only as part of a
            // CPU<->GPU transfer; a same-device dtype gap (CPU f64 grad into
            // a CPU f32 buffer read back from the GPU) needs an explicit
            // cast. nsl_tensor_cast is CPU-only, which is exactly the only
            // case where a dtype gap can exist (GPU tensors are always f32).
            let contig_probe = NslTensor::from_ptr(contig);
            let casted = if contig_probe.dtype == 0 && dst.dtype == 1 && contig_probe.device == 0 {
                // f64 host grad into an f32 buffer: plain downcast copy.
                // (nsl_tensor_cast is the CPDT F32/FP16/BF16 tool and
                // rejects f64 sources.)
                let shape: Vec<i64> = (0..contig_probe.ndim as usize)
                    .map(|i| unsafe { *contig_probe.shape.add(i) })
                    .collect();
                let out_ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&shape, 1);
                if out_ptr != 0 {
                    let out_t = NslTensor::from_ptr(out_ptr);
                    let n = contig_probe.len as usize;
                    let src_f64 = contig_probe.data as *const f64;
                    let dst_f32 = out_t.data as *mut f32;
                    for i in 0..n {
                        unsafe { *dst_f32.add(i) = *src_f64.add(i) as f32 };
                    }
                }
                out_ptr
            } else if contig_probe.dtype == 1 && dst.dtype == 0 && contig_probe.device == 0 {
                // f32 host src into an f64 buffer: plain upcast copy
                // (nsl_tensor_cast has no f64 TARGET either).
                let shape: Vec<i64> = (0..contig_probe.ndim as usize)
                    .map(|i| unsafe { *contig_probe.shape.add(i) })
                    .collect();
                let out_ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&shape, 0);
                if out_ptr != 0 {
                    let out_t = NslTensor::from_ptr(out_ptr);
                    let n = contig_probe.len as usize;
                    let src_f32 = contig_probe.data as *const f32;
                    let dst_f64 = out_t.data as *mut f64;
                    for i in 0..n {
                        unsafe { *dst_f64.add(i) = f64::from(*src_f32.add(i)) };
                    }
                }
                out_ptr
            } else if contig_probe.dtype != dst.dtype && contig_probe.device == 0 {
                crate::tensor::precision_cast::nsl_tensor_cast(contig, dst.dtype as i64)
            } else {
                contig
            };
            if casted != contig && contig != src_ptr {
                nsl_tensor_free(contig);
            }
            let migrated = nsl_tensor_to_device_like(casted, dst_ptr);
            if migrated != 0 && migrated != src_ptr {
                nsl_tensor_add_inplace(dst_ptr, migrated);
                // Refcount balance (review finding): when `to_device_like`
                // is a same-placement no-op it returns `casted` itself with
                // an EXTRA refcount — so freeing `migrated` and `casted`
                // independently is correct in BOTH cases: distinct pointers
                // get one free each; an aliased pointer gets its rc dropped
                // twice (bump + original ownership).
                nsl_tensor_free(migrated);
                if casted != src_ptr {
                    nsl_tensor_free(casted);
                }
                return;
            }
            // Migration failed (returned 0 or the raw src): drop the temp
            // and fall through to the strict asserts, which will report the
            // residual mismatch loudly.
            if casted != src_ptr {
                nsl_tensor_free(casted);
            }
        }
    }
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
    if dst.dtype != src.dtype && std::env::var("NSL_ALIGN_DEBUG").is_ok() {
        let shp = |t: &NslTensor| (0..t.ndim as usize)
            .map(|i| unsafe { *t.shape.add(i) }.to_string())
            .collect::<Vec<_>>()
            .join("x");
        eprintln!(
            "[align-debug] add_inplace mismatch: dst dev={} dtype={} shape={} | src dev={} dtype={} shape={} contig={}",
            dst.device, dst.dtype, shp(dst), src.device, src.dtype, shp(src), src.is_contiguous()
        );
    }
    assert_eq!(
        dst.dtype, src.dtype,
        "nsl_tensor_add_inplace: dtype mismatch (dst={}, src={})",
        dst.dtype, src.dtype
    );
    // Device memory: co-resident f32 operands take the elementwise add
    // kernel with the output aliased to dst — zero PCIe traffic, and
    // bit-identical to the old DtoH → f64 add → downcast → HtoD round-trip
    // (rounding the exact f64 sum of two f32 values to f32 equals the f32
    // sum). The FASE per-micro-batch accumulate (m_partial += scaled grad)
    // runs through here once per parameter per micro-batch, which made the
    // round-trip the dominant hidden host cost of large-model backwards.
    #[cfg(feature = "cuda")]
    if dst.device > 0 && src.device == dst.device && dst.dtype == 1 && src.dtype == 1 {
        crate::cuda::gpu_elementwise_binary_inplace(
            dst_ptr,
            src_ptr,
            crate::cuda::kernels::ADD_F32_PTX,
            "nsl_add_f32\0",
        );
        return;
    }
    // Residual device cases (non-f32 device tensors): transfer to CPU,
    // compute, copy back.
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
        // Copy result back to GPU. The GPU buffer is f32 (canonical GPU dtype).
        // The CPU result `dc` may be f32 or f64 depending on the migration path
        // (`nsl_tensor_to_device` upcasts GPU f32 → CPU f64). HtoD with a
        // f64-sized src would overrun the f32 dst buffer, so down-convert when
        // needed before the copy.
        let len = dc.len as usize;
        let f32_bytes = len * std::mem::size_of::<f32>();
        if dc.dtype == 1 {
            // CPU f32 → GPU f32: direct copy.
            crate::cuda::inner::memcpy_htod(dst.data, dc.data, f32_bytes);
        } else {
            // CPU f64 → GPU f32: stage the down-conversion in a heap buffer.
            let staging = checked_alloc(f32_bytes) as *mut f32;
            let src = dc.data as *const f64;
            for i in 0..len {
                unsafe { *staging.add(i) = *src.add(i) as f32; }
            }
            crate::cuda::inner::memcpy_htod(
                dst.data,
                staging as *const std::ffi::c_void,
                f32_bytes,
            );
            unsafe { checked_free(staging as *mut u8, f32_bytes); }
        }
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

/// Elementwise in-place scale: `tensor *= scalar`.
///
/// No allocation.  CPU tensors mutate their storage directly; GPU tensors
/// round-trip through CPU for correctness (if this shows up in a profile,
/// add a dedicated GPU scale kernel — for now the path is unused by
/// FASE Deferred, which runs clipping on CPU-resident accumulator slots).
///
/// Used by FASE Deferred's two-phase gradient clip (Phase B) to apply
/// the global clip factor to each parameter's `m_partial` without
/// allocating a fresh tensor per parameter.
#[no_mangle]
pub extern "C" fn nsl_tensor_mul_scalar_inplace(tensor_ptr: i64, scalar: f64) {
    if tensor_ptr == 0 {
        return;
    }
    let tensor = NslTensor::from_ptr(tensor_ptr);

    // Device-resident contiguous f32: in-place scale kernel, no PCIe.
    // The scalar rounds to f32 before the multiply (as every other GPU
    // scalar op already does); the old round-trip multiplied in f64 and
    // rounded after, so results can differ by ≤1 ulp when the scalar is
    // not exactly representable in f32 (e.g. a computed clip factor).
    // FASE's two-phase clip Phase B applies the factor through here once
    // per parameter per optimizer step.
    #[cfg(feature = "cuda")]
    if tensor.device > 0 && tensor.dtype == 1 && tensor.is_contiguous() {
        crate::cuda::gpu_scalar_op_inplace(
            tensor_ptr,
            scalar as f32,
            crate::cuda::kernels::MUL_SCALAR_F32_PTX,
            "nsl_mul_scalar_f32\0",
        );
        return;
    }

    #[cfg(feature = "cuda")]
    if tensor.device > 0 {
        // Round-trip: pull to CPU, scale, push back. `nsl_tensor_to_device`
        // upcasts the GPU f32 buffer to CPU f64, so the scaled CPU result is
        // f64. Copying f64-sized bytes back into the f32 device buffer would
        // reinterpret f64 bit patterns as f32 (garbage), so down-convert to
        // f32 before the HtoD when needed (mirrors nsl_tensor_add_inplace).
        let cpu_ptr = nsl_tensor_to_device(tensor_ptr, 0);
        nsl_tensor_mul_scalar_inplace(cpu_ptr, scalar);
        let cpu_tensor = NslTensor::from_ptr(cpu_ptr);
        let len = cpu_tensor.len as usize;
        let f32_bytes = len * std::mem::size_of::<f32>();
        if cpu_tensor.dtype == 1 {
            // CPU f32 → GPU f32: direct copy.
            crate::cuda::inner::memcpy_htod(tensor.data, cpu_tensor.data, f32_bytes);
        } else {
            // CPU f64 → GPU f32: stage the down-conversion in a heap buffer.
            let staging = checked_alloc(f32_bytes) as *mut f32;
            let src = cpu_tensor.data as *const f64;
            for i in 0..len {
                unsafe { *staging.add(i) = *src.add(i) as f32; }
            }
            crate::cuda::inner::memcpy_htod(
                tensor.data,
                staging as *const std::ffi::c_void,
                f32_bytes,
            );
            unsafe { checked_free(staging as *mut u8, f32_bytes); }
        }
        nsl_tensor_free(cpu_ptr);
        return;
    }

    #[cfg(not(feature = "cuda"))]
    if tensor.device > 0 {
        eprintln!("nsl: mul_scalar_inplace: GPU path requires cuda feature");
        std::process::abort();
    }

    if !tensor.has_writable_storage() {
        eprintln!("nsl: mul_scalar_inplace cannot write into borrowed CPU storage");
        std::process::abort();
    }

    if tensor.dtype == 1 {
        let s = scalar as f32;
        for i in 0..tensor.len as usize {
            unsafe {
                let p = tensor.data_f32().add(i);
                *p *= s;
            }
        }
    } else {
        for i in 0..tensor.len as usize {
            unsafe {
                let p = tensor.data_f64().add(i);
                *p *= scalar;
            }
        }
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

/// Create a zero-filled f16 (dtype=2, 2 bytes/element) tensor on the
/// requested device.
///
/// Used by the CSHA Tier C fused backward kernel's gradient-output
/// allocations (dq/dk/dv/dwq/dwk/dwv) — the PTX writes these via
/// `st.global.u16` at f16 element stride.  The classic `nsl_tensor_zeros_on`
/// hard-codes f32 (4 bytes/element), which leaves the upper bytes of each
/// element uninitialised and makes any subsequent f32 read interpret raw
/// f16 bits as f32 → garbage.  `dx` stays on the f32 helper because the
/// kernel writes it as f32; this helper exists specifically for the 6
/// f16-typed gradient outputs.
#[no_mangle]
pub extern "C" fn nsl_tensor_zeros_f16_on(shape_list: i64, device: i64) -> i64 {
    if device == 0 {
        return crate::tensor::creation::tensor_from_shape_list_f16(shape_list, 0.0);
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

        // 2 bytes/element for f16.
        let data_size = (len as usize) * 2;
        let data = crate::cuda::inner::alloc_managed(data_size);
        crate::cuda::inner::memset_d8(data, data_size);

        let strides = NslTensor::compute_strides(shape, ndim);

        let tensor = Box::new(NslTensor::new(
            data,
            shape,
            strides,
            ndim,
            len,
            device as u8,
            DTYPE_FP16,
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

// === Gradient utilities ===

/// Sum of squared elements: `Σ x²`, returned as f64.
///
/// Supports both f32 and f64 dtypes.  GPU tensors are transferred to
/// CPU for the reduction (mirrors `nsl_clip_grad_norm`).  Used by FASE
/// Deferred's two-phase gradient clipping to compute the global L2 norm.
#[no_mangle]
pub extern "C" fn nsl_tensor_sum_sq(tensor_ptr: i64) -> f64 {
    if tensor_ptr == 0 {
        return 0.0;
    }
    let tensor = NslTensor::from_ptr(tensor_ptr);

    // GPU f32 tensors: fused device-side sum-of-squares reduction (the RAW
    // Σx² accumulator from the stats kernel), 16-byte readback instead of a
    // full-tensor DtoH + host loop. Uses the dedicated `gpu_tensor_sum_sq_f32`
    // helper — NOT `gpu_tensor_stats_f32`, whose slot 3 is the population std
    // (mean/std transform applied), so reading it here silently returned std
    // and collapsed the FASE gradient-clip global norm by ~n on GPU. The
    // kernel accumulates in f32 where the CPU path accumulates in f64 (the
    // same accumulation dtype every other GPU reduction already uses).
    // NSL_SUM_SQ_CPU=1 restores the exact-f64 CPU reduction for bisections.
    #[cfg(feature = "cuda")]
    if tensor.device > 0 && tensor.dtype == 1 {
        static FORCE_CPU: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let force_cpu = *FORCE_CPU.get_or_init(|| {
            std::env::var("NSL_SUM_SQ_CPU").ok().as_deref() == Some("1")
        });
        if !force_cpu {
            let contig = nsl_tensor_contiguous(tensor_ptr);
            let ss = crate::cuda::gpu_tensor_sum_sq_f32(contig);
            nsl_tensor_free(contig);
            return ss;
        }
    }

    // GPU tensors: transfer to CPU for reduction.
    let (cpu_ptr, was_gpu) = if tensor.device > 0 {
        (nsl_tensor_to_device(tensor_ptr, 0), true)
    } else {
        (tensor_ptr, false)
    };

    let cpu_tensor = NslTensor::from_ptr(cpu_ptr);
    let mut acc: f64 = 0.0;
    if cpu_tensor.dtype == 1 {
        for i in 0..cpu_tensor.len as usize {
            let v = unsafe { *cpu_tensor.data_f32().add(i) } as f64;
            acc += v * v;
        }
    } else {
        for i in 0..cpu_tensor.len as usize {
            let v = unsafe { *cpu_tensor.data_f64().add(i) };
            acc += v * v;
        }
    }

    if was_gpu {
        nsl_tensor_free(cpu_ptr);
    }
    acc
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

/// Fused RMSNorm INPUT-gradient (dx) — one FFI in place of the ~11-op source-AD
/// decomposition. Computes the CORRECT RMSNorm dx (NO mean-subtract):
///   dx_j = g_j·ȳ_j/rms − x_j·Σ_k(ȳ_k·g_k·x_k)/(N·rms³),  rms=√(mean(x²)+eps)
/// recomputing rms internally (no saved-rms dependency). GPU uses the native
/// `nsl_rmsnorm_dx_bwd_f32` kernel (one block per row, one fused reduction);
/// CPU is the f64 reference. Emitted by source-AD's `RmsNormInputBackward` when
/// `--fuse-rmsnorm-backward` is set; equals the decomposition to f32 tolerance.
/// P5 item 20 slice A — fused RMSNorm GAMMA gradient (`--fuse-rmsnorm-backward`):
/// dgamma_j = sum_rows(dy_ij * x_ij / rms_i),  rms_i = sqrt(mean(x_i^2) + eps).
/// GPU: two deterministic launches (see `gpu_rmsnorm_dgamma_backward_f32`).
/// CPU: f64 reference with the same summation order.
#[no_mangle]
pub extern "C" fn nsl_rmsnorm_dgamma_backward(
    dy_ptr: i64,
    x_ptr: i64,
    gamma_ptr: i64,
    eps: f64,
) -> i64 {
    // Read only by the cuda device-dispatch below; the CPU arm re-binds `x`
    // from its contiguous copy.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    let x = NslTensor::from_ptr(x_ptr);

    #[cfg(feature = "cuda")]
    if x.device > 0 {
        // Same acquire/free discipline as nsl_rmsnorm_dx_backward: every
        // acquire returns an OWNED ref (even no-op same-device/contiguous),
        // so each gets exactly one unconditional free.
        let dy_dev = nsl_tensor_to_device(dy_ptr, x.device as i64);
        let dy_c = nsl_tensor_contiguous(dy_dev);
        let x_c = nsl_tensor_contiguous(x_ptr);
        let g_dev = nsl_tensor_to_device(gamma_ptr, x.device as i64);
        let g_c = nsl_tensor_contiguous(g_dev);
        let dg = crate::cuda::gpu_rmsnorm_dgamma_backward_f32(dy_c, x_c, g_c, eps as f32);
        nsl_tensor_free(dy_dev);
        nsl_tensor_free(dy_c);
        nsl_tensor_free(x_c);
        nsl_tensor_free(g_dev);
        nsl_tensor_free(g_c);
        return dg;
    }

    // CPU reference (f64 accumulation, row-major order). Contiguous copies
    // first — the flat indexing below is meaningless on strided views
    // (review L4); both acquires are owned refs, freed at the end.
    let dy_c = nsl_tensor_contiguous(dy_ptr);
    let x_c = nsl_tensor_contiguous(x_ptr);
    let dy = NslTensor::from_ptr(dy_c);
    let x = NslTensor::from_ptr(x_c);
    let gamma = NslTensor::from_ptr(gamma_ptr);
    let ndim = x.ndim as usize;
    let n = unsafe { *x.shape.add(ndim - 1) } as usize;
    let total = x.len as usize;
    let num_rows = total / n;
    let nf = n as f64;
    assert_eq!(
        gamma.len as usize, n,
        "rmsnorm dgamma: gamma len {} != last dim {}",
        gamma.len, n
    );

    let rd = |t: &NslTensor, i: usize| -> f64 {
        if t.dtype == 1 { unsafe { *t.data_f32().add(i) as f64 } } else { unsafe { *t.data_f64().add(i) } }
    };
    let g_shape: Vec<i64> = (0..gamma.ndim as usize)
        .map(|i| unsafe { *gamma.shape.add(i) })
        .collect();
    let dg_ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&g_shape, gamma.dtype);
    let dg = NslTensor::from_ptr(dg_ptr);

    let mut rinv = vec![0.0_f64; num_rows];
    for (row, r) in rinv.iter_mut().enumerate() {
        let base = row * n;
        let mut sum_sq = 0.0_f64;
        for j in 0..n {
            let v = rd(x, base + j);
            sum_sq += v * v;
        }
        *r = 1.0 / (sum_sq / nf + eps).sqrt();
    }
    for j in 0..n {
        let mut acc = 0.0_f64;
        for (row, r) in rinv.iter().enumerate() {
            let i = row * n + j;
            acc += rd(dy, i) * rd(x, i) * r;
        }
        if gamma.dtype == 1 {
            unsafe { *dg.data_f32().add(j) = acc as f32 };
        } else {
            unsafe { *dg.data_f64().add(j) = acc };
        }
    }
    nsl_tensor_free(dy_c);
    nsl_tensor_free(x_c);
    dg_ptr
}

/// P5 slice C — fused RMSNorm dx + residual-gradient fold:
///   out = dx(dy, x, gamma, eps) + res
/// Replaces the adjoint-accumulate Add that followed the fused dx op.
/// Bit-exact with (dx then Add): the epilogue performs the same single
/// rn-rounded add, and IEEE addition is commutative so either accumulate
/// operand order matches. Mismatched `res` falls back to dx-then-add.
#[no_mangle]
pub extern "C" fn nsl_rmsnorm_dx_backward_add(
    dy_ptr: i64,
    x_ptr: i64,
    gamma_ptr: i64,
    res_ptr: i64,
    eps: f64,
) -> i64 {
    let x = NslTensor::from_ptr(x_ptr);
    let r = NslTensor::from_ptr(res_ptr);
    // Fallback for any shape/device/dtype mismatch: the exact decomposed
    // pair the compiler emitted before the fold.
    if !r.shape_eq(x) || r.device != x.device || r.dtype != x.dtype {
        let dx = nsl_rmsnorm_dx_backward(dy_ptr, x_ptr, gamma_ptr, eps);
        let out = nsl_tensor_add(dx, res_ptr, 0);
        nsl_tensor_free(dx);
        return out;
    }

    #[cfg(feature = "cuda")]
    if x.device > 0 {
        let dy_dev = nsl_tensor_to_device(dy_ptr, x.device as i64);
        let dy_c = nsl_tensor_contiguous(dy_dev);
        let x_c = nsl_tensor_contiguous(x_ptr);
        let g_dev = nsl_tensor_to_device(gamma_ptr, x.device as i64);
        let g_c = nsl_tensor_contiguous(g_dev);
        let r_c = nsl_tensor_contiguous(res_ptr);
        let dx = crate::cuda::gpu_rmsnorm_dx_backward_add_f32(dy_c, x_c, g_c, r_c, eps as f32);
        nsl_tensor_free(dy_dev);
        nsl_tensor_free(dy_c);
        nsl_tensor_free(x_c);
        nsl_tensor_free(g_dev);
        nsl_tensor_free(g_c);
        nsl_tensor_free(r_c);
        return dx;
    }

    // CPU: exact dx reference then the same single f64 add.
    let dx = nsl_rmsnorm_dx_backward(dy_ptr, x_ptr, gamma_ptr, eps);
    let out = nsl_tensor_add(dx, res_ptr, 0);
    nsl_tensor_free(dx);
    out
}

#[no_mangle]
pub extern "C" fn nsl_rmsnorm_dx_backward(
    dy_ptr: i64,
    x_ptr: i64,
    gamma_ptr: i64,
    eps: f64,
) -> i64 {
    let x = NslTensor::from_ptr(x_ptr);

    #[cfg(feature = "cuda")]
    if x.device > 0 {
        // Native GPU kernel — contiguous f32 on the input's device.
        let dy_dev = nsl_tensor_to_device(dy_ptr, x.device as i64);
        let dy_c = nsl_tensor_contiguous(dy_dev);
        let x_c = nsl_tensor_contiguous(x_ptr);
        let g_dev = nsl_tensor_to_device(gamma_ptr, x.device as i64);
        let g_c = nsl_tensor_contiguous(g_dev);
        let dx = crate::cuda::gpu_rmsnorm_dx_backward_f32(dy_c, x_c, g_c, eps as f32);
        // Both `nsl_tensor_to_device` and `nsl_tensor_contiguous` return an
        // OWNED ref — refcount++ even on a same-device / already-contiguous
        // no-op (they return the SAME pointer). So each acquire needs exactly
        // one free, UNCONDITIONALLY: guarding the to_device free on
        // `dev != ptr` leaks the no-op's extra ref (unbounded GPU growth on the
        // per-step `dy`). Mirrors the tensor/mod.rs:~1793 refcount-balance idiom.
        nsl_tensor_free(dy_dev);
        nsl_tensor_free(dy_c);
        nsl_tensor_free(x_c);
        nsl_tensor_free(g_dev);
        nsl_tensor_free(g_c);
        return dx;
    }

    // CPU reference (f64 accumulation).
    let dy = NslTensor::from_ptr(dy_ptr);
    let gamma = NslTensor::from_ptr(gamma_ptr);
    let ndim = x.ndim as usize;
    let n = unsafe { *x.shape.add(ndim - 1) } as usize;
    let total = x.len as usize;
    let num_rows = total / n;
    let nf = n as f64;

    let rd = |t: &NslTensor, i: usize| -> f64 {
        if t.dtype == 1 { unsafe { *t.data_f32().add(i) as f64 } } else { unsafe { *t.data_f64().add(i) } }
    };
    let shape: Vec<i64> = (0..ndim).map(|i| unsafe { *x.shape.add(i) }).collect();
    let dx_ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&shape, x.dtype);
    let dx = NslTensor::from_ptr(dx_ptr);
    let write_dx = |i: usize, v: f64| {
        if x.dtype == 1 { unsafe { *dx.data_f32().add(i) = v as f32 } }
        else { unsafe { *dx.data_f64().add(i) = v } }
    };

    for row in 0..num_rows {
        let base = row * n;
        let mut sum_sq = 0.0_f64;
        let mut sum_dwx = 0.0_f64;
        for j in 0..n {
            let xj = rd(x, base + j);
            sum_sq += xj * xj;
            sum_dwx += rd(dy, base + j) * rd(gamma, j) * xj;
        }
        let rms = (sum_sq / nf + eps).sqrt().max(1e-12);
        let rms_cubed = rms * rms * rms;
        for j in 0..n {
            let xj = rd(x, base + j);
            let d = rd(gamma, j) * rd(dy, base + j) / rms - xj * sum_dwx / (nf * rms_cubed);
            write_dx(base + j, d);
        }
    }
    dx_ptr
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
                    DTYPE_I32,
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

/// Migrate `src` to match `ref_tensor`'s device. Returns a new refcounted tensor
/// (either `src` with incremented refcount if devices already match, or a
/// device-migrated copy). Used by FASE Deferred to reconcile CPU tape-AD
/// gradients with GPU-resident m_partial buffers before in-place accumulation.
#[no_mangle]
pub extern "C" fn nsl_tensor_to_device_like(src_ptr: i64, ref_ptr: i64) -> i64 {
    let r = unsafe { &*(ref_ptr as *const NslTensor) };
    nsl_tensor_to_device(src_ptr, r.device as i64)
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
                DTYPE_I32 => std::mem::size_of::<i32>(),
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
            if first_val != 0 && (first_val as u64) >= 0x10000 && (first_val as usize).is_multiple_of(8) {
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
// Test-only helpers — build/read tensors from f32 slices for integration
// tests in `tests/*.rs`. Gated on the `test-hooks` feature so production
// builds don't expose this surface.
// ---------------------------------------------------------------------------

#[cfg(feature = "test-hooks")]
pub fn test_build_tensor_2d_f32(rows: usize, cols: usize, data: &[f32]) -> i64 {
    assert_eq!(data.len(), rows * cols, "data length must equal rows*cols");
    let shape = crate::list::nsl_list_new();
    crate::list::nsl_list_push(shape, rows as i64);
    crate::list::nsl_list_push(shape, cols as i64);
    let ptr = nsl_tensor_zeros(shape);

    let t = unsafe { &*(ptr as *const NslTensor) };
    let buf = t.data as *mut f32;
    for (i, &v) in data.iter().enumerate() {
        unsafe { *buf.add(i) = v };
    }
    crate::list::nsl_list_free(shape);
    ptr
}

#[cfg(feature = "test-hooks")]
pub fn test_read_tensor_f32(ptr: i64) -> Vec<f32> {
    let t = unsafe { &*(ptr as *const NslTensor) };
    assert_eq!(t.dtype, 1, "expected f32 tensor (dtype=1), got dtype={}", t.dtype);
    let len = t.len as usize;
    let buf = t.data as *const f32;
    (0..len).map(|i| unsafe { *buf.add(i) }).collect()
}

#[cfg(feature = "test-hooks")]
pub fn test_read_tensor_f64(ptr: i64) -> Vec<f64> {
    let t = unsafe { &*(ptr as *const NslTensor) };
    assert_eq!(t.dtype, 0, "expected f64 tensor (dtype=0), got dtype={}", t.dtype);
    let len = t.len as usize;
    let buf = t.data as *const f64;
    (0..len).map(|i| unsafe { *buf.add(i) }).collect()
}

/// Read `tape_id` from an `NslTensor` pointer.
///
/// Used by `tests/desc_tape_id_roundtrip.rs` to verify that the
/// `NslTensorDesc::tape_id` field is carried verbatim across
/// `nsl_tensor_to_desc_ffi` → `nsl_desc_to_tensor`. Returns the
/// `tape_id` field as-is — `0` means "untracked".
#[cfg(feature = "test-hooks")]
pub fn test_tensor_tape_id(ptr: i64) -> i64 {
    let t = unsafe { &*(ptr as *const NslTensor) };
    t.tape_id
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// P0.4 dtype/ABI cleanup: GOLDEN LOCK on the canonical dtype tag table.
    /// A future fused kernel that renumbers or reuses a tag fails HERE instead
    /// of silently colliding on the wire. Also pins the byte widths and that
    /// the C API speaks the SAME canonical tag space (P4 item 16 removed the
    /// historical 0=f32/1=f64 inversion and the tag-4 i32/int8 overload —
    /// neither may be reintroduced).
    #[test]
    fn dtype_abi_lock() {
        // The canonical table — do not renumber; append at the next free slot.
        assert_eq!(DTYPE_F64, 0);
        assert_eq!(DTYPE_F32, 1);
        assert_eq!(DTYPE_FP16, 2);
        assert_eq!(DTYPE_BF16, 3);
        assert_eq!(DTYPE_INT8, 4);
        assert_eq!(DTYPE_FP8E4M3, 5);
        assert_eq!(DTYPE_FP8E5M2, 6);
        assert_eq!(DTYPE_U16_TOKEN, 7);
        assert_eq!(DTYPE_U16_SEGMENT, 8);
        assert_eq!(DTYPE_I32, 9);
        assert_eq!(DTYPE_CUSTOM_START, 256);

        // Byte widths. Tag 4 (int8) is 1 byte — the historical tag-4-as-i32
        // overload is gone; i32 token tensors carry DTYPE_I32 (9).
        assert_eq!(dtype_element_size(DTYPE_F64), 8);
        assert_eq!(dtype_element_size(DTYPE_F32), 4);
        assert_eq!(dtype_element_size(DTYPE_FP16), 2);
        assert_eq!(dtype_element_size(DTYPE_BF16), 2);
        assert_eq!(dtype_element_size(DTYPE_U16_TOKEN), 2);
        assert_eq!(dtype_element_size(DTYPE_U16_SEGMENT), 2);
        assert_eq!(dtype_element_size(DTYPE_INT8), 1);
        assert_eq!(dtype_element_size(DTYPE_I32), 4);

        // The C API uses the canonical tag space verbatim: the conversion
        // chokepoints are validating IDENTITY functions. Any reintroduced
        // inversion fails here.
        use crate::c_api::{capi_dtype_to_nsl, nsl_dtype_to_capi};
        for tag in [
            DTYPE_F64, DTYPE_F32, DTYPE_FP16, DTYPE_BF16, DTYPE_INT8,
            DTYPE_FP8E4M3, DTYPE_FP8E5M2, DTYPE_U16_TOKEN, DTYPE_U16_SEGMENT,
            DTYPE_I32,
        ] {
            assert_eq!(nsl_dtype_to_capi(tag), tag as i32, "C-API tag must equal canonical tag {tag}");
            assert_eq!(capi_dtype_to_nsl(tag as i32), tag, "canonical tag must round-trip {tag}");
        }
    }

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

    /// Gap I.3 (A+F): `nsl_tensor_zeros_f16_on` must return a tensor whose
    /// storage is actually f16 (dtype=2, element_size=2). Regression pin:
    /// the Tier C fused backward writes dq/dk/dv/dwq/dwk/dwv via
    /// `st.global.u16`; any f32-sized allocation would over-allocate by
    /// 2× and leave half the bytes uninitialised.
    #[test]
    fn gap_i3_zeros_f16_on_cpu_reports_f16_dtype_and_elem_size() {
        let shape = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape, 4);
        crate::list::nsl_list_push(shape, 8);
        // device=0 → CPU path.  The CUDA path is gated by cfg(feature="cuda")
        // and is exercised end-to-end by the CSHA integration tests.
        let t = nsl_tensor_zeros_f16_on(shape, 0);
        let tensor = NslTensor::from_ptr(t);
        assert_eq!(tensor.dtype, DTYPE_FP16, "dtype must be DTYPE_FP16 (2)");
        assert_eq!(tensor.element_size(), 2, "f16 element size must be 2 B");
        assert_eq!(tensor.len, 32, "shape [4,8] → len=32");
        assert_eq!(tensor.ndim, 2);
        assert_eq!(unsafe { *tensor.shape }, 4);
        assert_eq!(unsafe { *tensor.shape.add(1) }, 8);
        // All-zero: reading each of the 32 u16 slots must give 0.
        for i in 0..tensor.len as usize {
            let bits = unsafe { *(tensor.data as *const u16).add(i) };
            assert_eq!(bits, 0, "slot {} not zero-initialised", i);
        }
        nsl_tensor_free(t);
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

        let result = nsl_tensor_add(a_t, b_t, 0);
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
        use crate::tensor::fbip_flags::RELINQUISH_A;
        let a = make_f64_tensor(&[1.0, 2.0, 3.0]);
        let b = make_f64_tensor(&[10.0, 20.0, 30.0]);
        // ELTLS: in-place now requires explicit RELINQUISH_A flag.
        let result = arithmetic::nsl_tensor_add(a, b, RELINQUISH_A);
        assert_eq!(result, a, "add should reuse left operand when RELINQUISH_A is set");
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![11.0, 22.0, 33.0]);
        // RELINQUISH_A transferred the input ref into the result — one free total.
        assert_eq!(t.refcount.load(Ordering::SeqCst), 1, "in-place reuse must not add a ref");
        nsl_tensor_free(result);
        nsl_tensor_free(b);
    }

    #[test]
    fn test_fbip_add_alloc_when_shared() {
        let a = make_f64_tensor(&[1.0, 2.0]);
        let b = make_f64_tensor(&[10.0, 20.0]);
        NslTensor::from_ptr(a).refcount.fetch_add(1, Ordering::SeqCst);
        let result = arithmetic::nsl_tensor_add(a, b, 0);
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
        use crate::tensor::fbip_flags::RELINQUISH_A;
        let a = make_f64_tensor(&[2.0, 3.0, 4.0]);
        let b = make_f64_tensor(&[10.0, 10.0, 10.0]);
        // ELTLS: in-place now requires explicit RELINQUISH_A flag.
        let result = arithmetic::nsl_tensor_mul(a, b, RELINQUISH_A);
        assert_eq!(result, a);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![20.0, 30.0, 40.0]);
        assert_eq!(t.refcount.load(Ordering::SeqCst), 1, "in-place reuse must not add a ref");
        nsl_tensor_free(result);
        nsl_tensor_free(b);
    }

    #[test]
    fn test_fbip_div_inplace() {
        use crate::tensor::fbip_flags::RELINQUISH_A;
        let a = make_f64_tensor(&[10.0, 20.0, 30.0]);
        let b = make_f64_tensor(&[2.0, 4.0, 5.0]);
        // ELTLS: in-place now requires explicit RELINQUISH_A flag.
        let result = arithmetic::nsl_tensor_div(a, b, RELINQUISH_A);
        assert_eq!(result, a);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![5.0, 5.0, 6.0]);
        assert_eq!(t.refcount.load(Ordering::SeqCst), 1, "in-place reuse must not add a ref");
        nsl_tensor_free(result);
        nsl_tensor_free(b);
    }

    #[test]
    fn test_fbip_sub_inplace() {
        use crate::tensor::fbip_flags::RELINQUISH_A;
        let a = make_f64_tensor(&[10.0, 20.0]);
        let b = make_f64_tensor(&[3.0, 7.0]);
        // ELTLS: in-place now requires explicit RELINQUISH_A flag.
        let result = arithmetic::nsl_tensor_sub(a, b, RELINQUISH_A);
        assert_eq!(result, a);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..2).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![7.0, 13.0]);
        assert_eq!(t.refcount.load(Ordering::SeqCst), 1, "in-place reuse must not add a ref");
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
        use crate::tensor::fbip_flags::RELINQUISH_A;
        let ptr = make_f64_tensor(&[1.0, 2.0, 3.0]);
        // ELTLS: in-place now requires explicit RELINQUISH_A flag.
        let result = arithmetic::nsl_tensor_add_scalar(ptr, 10.0, RELINQUISH_A);
        assert_eq!(result, ptr);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![11.0, 12.0, 13.0]);
        assert_eq!(t.refcount.load(Ordering::SeqCst), 1, "in-place reuse must not add a ref");
        nsl_tensor_free(result);
    }

    #[test]
    fn test_fbip_mul_scalar_inplace() {
        use crate::tensor::fbip_flags::RELINQUISH_A;
        let ptr = make_f64_tensor(&[2.0, 3.0, 4.0]);
        // ELTLS: in-place now requires explicit RELINQUISH_A flag.
        let result = arithmetic::nsl_tensor_mul_scalar(ptr, 5.0, RELINQUISH_A);
        assert_eq!(result, ptr);
        let t = NslTensor::from_ptr(result);
        let vals: Vec<f64> = (0..3).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        assert_eq!(vals, vec![10.0, 15.0, 20.0]);
        assert_eq!(t.refcount.load(Ordering::SeqCst), 1, "in-place reuse must not add a ref");
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

    #[test]
    fn sum_sq_f32_known_values() {
        // 4-element f32 tensor with [1.0, 2.0, -3.0, 0.5]: Σx² = 14.25
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 4);
        let t = nsl_tensor_zeros(shape_list);
        let tensor = NslTensor::from_ptr(t);
        unsafe {
            *tensor.data_f32().add(0) = 1.0;
            *tensor.data_f32().add(1) = 2.0;
            *tensor.data_f32().add(2) = -3.0;
            *tensor.data_f32().add(3) = 0.5;
        }
        let got = nsl_tensor_sum_sq(t);
        assert!((got - 14.25).abs() < 1e-9, "got {}", got);
        nsl_tensor_free(t);
        crate::list::nsl_list_free(shape_list);
    }

    #[test]
    fn sum_sq_of_zero_tensor_is_zero() {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 8);
        let t = nsl_tensor_zeros(shape_list);
        let got = nsl_tensor_sum_sq(t);
        assert_eq!(got, 0.0);
        nsl_tensor_free(t);
        crate::list::nsl_list_free(shape_list);
    }

    #[test]
    fn sum_sq_null_pointer_returns_zero() {
        let got = nsl_tensor_sum_sq(0);
        assert_eq!(got, 0.0);
    }

    /// Regression gate for the review-found HIGH bug: the GPU fast path used
    /// to read `gpu_tensor_stats_f32()[3]`, which is the population STD (the
    /// helper post-processes the raw sum_sq into std), not Σx². That silently
    /// collapsed the FASE gradient-clip global norm by ~n on GPU. Assert the
    /// GPU sum_sq matches the exact CPU reduction on a multi-element f32
    /// tensor where std and Σx² differ substantially.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU: requires CUDA device"]
    fn sum_sq_gpu_matches_cpu_reference() {
        if crate::nsl_cuda_init() != 0 {
            return;
        }
        // [1, 2, -3, 0.5]: Σx² = 14.25; std ≈ 1.98 — off by ~7x, so a
        // std-vs-sumsq confusion cannot pass this by coincidence.
        let vals = [1.0f32, 2.0, -3.0, 0.5];
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, vals.len() as i64);
        let cpu = nsl_tensor_zeros(shape_list);
        {
            let t = NslTensor::from_ptr(cpu);
            for (i, &v) in vals.iter().enumerate() {
                unsafe { *t.data_f32().add(i) = v };
            }
        }
        let cpu_ss = nsl_tensor_sum_sq(cpu);
        assert!((cpu_ss - 14.25).abs() < 1e-4, "CPU sum_sq = {cpu_ss}, expected 14.25");

        let gpu = nsl_tensor_to_device(cpu, 1);
        let gpu_ss = nsl_tensor_sum_sq(gpu);
        assert!(
            (gpu_ss - cpu_ss).abs() < 1e-3,
            "GPU sum_sq = {gpu_ss} but CPU = {cpu_ss} (std would be ~1.98 — a std/sum_sq slot confusion)"
        );
        nsl_tensor_free(gpu);
        nsl_tensor_free(cpu);
    }

    #[test]
    fn mul_scalar_inplace_f32_scales_values() {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 3);
        let t = nsl_tensor_zeros(shape_list);
        let tensor = NslTensor::from_ptr(t);
        unsafe {
            *tensor.data_f32().add(0) = 2.0;
            *tensor.data_f32().add(1) = -1.5;
            *tensor.data_f32().add(2) = 0.0;
        }
        nsl_tensor_mul_scalar_inplace(t, 0.5);
        unsafe {
            assert!(((*tensor.data_f32().add(0)) - 1.0).abs() < 1e-6);
            assert!(((*tensor.data_f32().add(1)) - (-0.75)).abs() < 1e-6);
            assert!((*tensor.data_f32().add(2)).abs() < 1e-6);
        }
        nsl_tensor_free(t);
        crate::list::nsl_list_free(shape_list);
    }

    #[test]
    fn mul_scalar_inplace_by_zero_clears_tensor() {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 4);
        let t = nsl_tensor_zeros(shape_list);
        let tensor = NslTensor::from_ptr(t);
        unsafe {
            for i in 0..4 {
                *tensor.data_f32().add(i) = (i + 1) as f32;
            }
        }
        nsl_tensor_mul_scalar_inplace(t, 0.0);
        unsafe {
            for i in 0..4 {
                assert_eq!(*tensor.data_f32().add(i), 0.0);
            }
        }
        nsl_tensor_free(t);
        crate::list::nsl_list_free(shape_list);
    }

    #[test]
    fn mul_scalar_inplace_null_pointer_is_noop() {
        // Null pointer must not panic or abort.
        nsl_tensor_mul_scalar_inplace(0, 2.0);
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

#[no_mangle]
pub extern "C" fn nsl_tensor_l2_norm(t: i64) -> f64 {
    let tensor = NslTensor::from_ptr(t);
    tensor_l2_norm(tensor)
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

/// Set GPU allocator pool to Persistent (for model weights, optimizer states).
#[no_mangle]
pub extern "C" fn nsl_gpu_set_persistent_pool() {
    #[cfg(feature = "cuda")]
    crate::cuda::caching_allocator::set_alloc_pool(
        crate::cuda::caching_allocator::AllocPool::Persistent,
    );
}

/// Set GPU allocator pool to Transient (for forward/backward intermediates).
#[no_mangle]
pub extern "C" fn nsl_gpu_set_transient_pool() {
    #[cfg(feature = "cuda")]
    crate::cuda::caching_allocator::set_alloc_pool(
        crate::cuda::caching_allocator::AllocPool::Transient,
    );
}

/// Set the GPU allocator surface tag for subsequent allocations (P0.1
/// per-surface VRAM accounting). Tag values match
/// `caching_allocator::SurfaceTag`: 0=other, 1=weights, 2=optim_m,
/// 3=optim_v, 4=m_partial, 5=grads, 6=activations, 7=attn_workspace.
/// Unknown values map to `other`. Purely observational — never affects
/// allocator placement decisions (see `nsl_gpu_set_persistent_pool` for
/// those).
#[no_mangle]
pub extern "C" fn nsl_gpu_set_alloc_surface(tag: u8) {
    #[cfg(feature = "cuda")]
    crate::cuda::caching_allocator::set_alloc_surface(
        crate::cuda::caching_allocator::SurfaceTag::from_u8(tag),
    );
    #[cfg(not(feature = "cuda"))]
    let _ = tag;
}

/// Get the current GPU allocator surface tag (see `nsl_gpu_set_alloc_surface`).
/// Codegen brackets use get/set to restore the caller's surface, keeping
/// nested regions safe. Always 0 in non-CUDA builds.
#[no_mangle]
pub extern "C" fn nsl_gpu_get_alloc_surface() -> u8 {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::caching_allocator::get_alloc_surface() as u8
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// A1: set the stable `(op_id, tensor_id)` allocation identity for
/// subsequent GPU allocations (arena / CUDA-graph attribution hook). No-op
/// in non-CUDA builds.
#[no_mangle]
pub extern "C" fn nsl_gpu_set_alloc_identity(op_id: u32, tensor_id: u64) {
    #[cfg(feature = "cuda")]
    crate::cuda::caching_allocator::set_alloc_identity(op_id, tensor_id);
    #[cfg(not(feature = "cuda"))]
    let _ = (op_id, tensor_id);
}

/// A1: clear the allocation identity set by `nsl_gpu_set_alloc_identity`.
#[no_mangle]
pub extern "C" fn nsl_gpu_clear_alloc_identity() {
    #[cfg(feature = "cuda")]
    crate::cuda::caching_allocator::clear_alloc_identity();
}

/// A1: global peak allocated device bytes across every surface and every
/// allocation mechanism (pooled + async + direct). This is the first
/// in-process numeric VRAM-peak getter — regression gates and WGGO read it
/// directly instead of scraping `NSL_MEMSTATS` stderr. `0` in non-CUDA builds.
#[no_mangle]
pub extern "C" fn nsl_gpu_peak_allocated_bytes() -> i64 {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::caching_allocator::CACHING_ALLOCATOR
            .lock()
            .unwrap()
            .peak_allocated_bytes() as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// A1: cumulative allocation-event count since the last
/// `nsl_gpu_reset_mem_stats` (pooled + external). An unexpected jump between
/// steps flags a per-step allocation the arena/liveness passes missed.
#[no_mangle]
pub extern "C" fn nsl_gpu_cumulative_alloc_count() -> i64 {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::caching_allocator::CACHING_ALLOCATOR
            .lock()
            .unwrap()
            .cumulative_alloc_count() as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// A1: this surface's bytes at the moment the global allocated peak was set
/// — the per-surface decomposition of peak VRAM. Tag values match
/// `SurfaceTag` (0=other … 7=attn_workspace). `0` in non-CUDA builds.
#[no_mangle]
pub extern "C" fn nsl_gpu_surface_at_peak_bytes(tag: u8) -> i64 {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::caching_allocator::CACHING_ALLOCATOR
            .lock()
            .unwrap()
            .surface_at_global_peak(crate::cuda::caching_allocator::SurfaceTag::from_u8(tag)) as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = tag;
        0
    }
}

/// A1: this surface's own high-water mark in bytes. `0` in non-CUDA builds.
#[no_mangle]
pub extern "C" fn nsl_gpu_surface_peak_bytes(tag: u8) -> i64 {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::caching_allocator::CACHING_ALLOCATOR
            .lock()
            .unwrap()
            .surface_peak(crate::cuda::caching_allocator::SurfaceTag::from_u8(tag)) as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = tag;
        0
    }
}

/// A1: reset the peak high-water marks and cumulative allocation counters so
/// a caller can measure a fresh region (e.g. one training step). Live bytes
/// are preserved; peaks re-seed to the current live level.
#[no_mangle]
pub extern "C" fn nsl_gpu_reset_mem_stats() {
    #[cfg(feature = "cuda")]
    crate::cuda::caching_allocator::CACHING_ALLOCATOR
        .lock()
        .unwrap()
        .reset_peak_and_counts();
}

/// Release idle GPU memory back to the driver. Called after each training step
/// to prevent the caching allocator from holding stale segments.
#[no_mangle]
pub extern "C" fn nsl_gpu_drain_cache() {
    // P5 item 19: while cuda-graph capture is armed, the per-step transient
    // drain would churn every transient address (no region could ever
    // digest-stabilize) AND physically unmap memory that already-captured
    // graphs reference — skip it. OOM recovery still drains via pool_drain,
    // which taints any active region first.
    if crate::cuda::graph_capture::cuda_graphs_armed() {
        return;
    }
    // Probe gate (2026-04-23): NSL_SKIP_GPU_DRAIN=1 bypasses this workaround
    // so we can observe whether ELTLS frees intermediates at last use.
    if std::env::var("NSL_SKIP_GPU_DRAIN").ok().as_deref() != Some("1") {
        #[cfg(feature = "cuda")]
        {
            crate::cuda::inner::ensure_context();
            // Sync first to ensure all async GPU ops complete so freed blocks are actually available
            unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
            let mut alloc = crate::cuda::caching_allocator::CACHING_ALLOCATOR.lock().unwrap();
            let freed = alloc.drain_all();
            if freed > 0 {
                eprintln!("[gpu-drain] released {}MB to driver", freed / (1024 * 1024));
            }
        }
    }
}

/// Global counter for live GPU tensor allocations (debug only).
static GPU_TENSOR_LIVE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
static GPU_TENSOR_BYTES: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// Track a GPU tensor creation.
pub(crate) fn debug_track_gpu_alloc(bytes: usize) {
    GPU_TENSOR_LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GPU_TENSOR_BYTES.fetch_add(bytes as i64, std::sync::atomic::Ordering::Relaxed);
}

/// Track a GPU tensor free.
pub(crate) fn debug_track_gpu_free(bytes: usize) {
    GPU_TENSOR_LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    GPU_TENSOR_BYTES.fetch_sub(bytes as i64, std::sync::atomic::Ordering::Relaxed);
}

/// Debug: print GPU allocated block summary grouped by context.
#[no_mangle]
pub extern "C" fn nsl_debug_gpu_alloc_summary(step: i64) {
    let all = std::env::var("NSL_DEBUG_MEM_ALL").ok().as_deref() == Some("1");
    if !all && step > 2 { return; } // default: only first 3 steps
    #[cfg(feature = "cuda")]
    {
        let alloc = crate::cuda::caching_allocator::CACHING_ALLOCATOR.lock().unwrap();
        let summary = alloc.allocated_block_summary();
        eprintln!("[gpu-alloc-summary] step={} live blocks:", step);
        for (ctx, count, bytes) in &summary {
            if *bytes > 1024 {
                eprintln!("  {} — {} blocks, {}KB", ctx, count, bytes / 1024);
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    { let _ = step; }
}

/// Debug: print GPU memory usage + caching allocator stats.
#[no_mangle]
pub extern "C" fn nsl_debug_gpu_mem(step: i64) {
    let all = std::env::var("NSL_DEBUG_MEM_ALL").ok().as_deref() == Some("1");
    if !all && step > 5 { return; }
    #[cfg(feature = "cuda")]
    {
        unsafe {
            crate::cuda::inner::ensure_context();
            let mut free: usize = 0;
            let mut total: usize = 0;
            cudarc::driver::sys::cuMemGetInfo_v2(&mut free, &mut total);
            let used_mb = (total - free) / (1024 * 1024);
            let alloc = crate::cuda::caching_allocator::CACHING_ALLOCATOR.lock().unwrap();
            let stats = alloc.stats();
            let mb = |b: usize| b / (1024 * 1024);
            eprintln!(
                "[gpu-mem] step={} driver={}MB alloc={}MB reserved={}MB live_blocks={} drv_allocs={} drv_frees={}",
                step, used_mb,
                mb(stats.allocated_bytes),
                mb(stats.reserved_bytes),
                stats.num_allocs, stats.num_driver_allocs, stats.num_driver_frees,
            );
            let (p_b, p_s, t_b, t_s) = alloc.pool_breakdown();
            eprintln!(
                "[gpu-mem]    persistent={}MB ({} segs)  transient={}MB ({} segs)  free_blocks={} hits={} misses={} splits={} coalesces={}",
                mb(p_b), p_s, mb(t_b), t_s,
                stats.num_free_blocks,
                stats.num_cache_hits, stats.num_cache_misses,
                stats.num_splits, stats.num_coalesces,
            );
            // P0.1: per-surface attribution (current/peak). Zero-only
            // surfaces are elided to keep the line readable.
            let mut surface_line = String::from("[gpu-mem]    surfaces:");
            for (name, cur, peak) in alloc.surface_breakdown() {
                if cur > 0 || peak > 0 {
                    surface_line.push_str(&format!(
                        " {}={}MB(peak {}MB)",
                        name,
                        cur / (1024 * 1024),
                        peak / (1024 * 1024),
                    ));
                }
            }
            eprintln!("{}", surface_line);
            // A1: external (non-pooled) allocation breakdown — async /
            // direct-device / identity coverage. Only when any exist.
            let ext = alloc.external_summary();
            if ext.total_count > 0 {
                eprintln!(
                    "[gpu-mem]    external: async={}MB direct={}MB persistent={}MB \
                     ({} allocs, {} with op/tensor identity)",
                    ext.async_bytes / (1024 * 1024),
                    ext.direct_bytes / (1024 * 1024),
                    ext.persistent_bytes / (1024 * 1024),
                    ext.total_count,
                    ext.identified_count,
                );
            }
            drop(alloc);
            // p3-remainder: raw `alloc_device` buffers (CSHA backward saves,
            // Tier B.1 x-scratch) freed via the stream-ordered deferred path
            // are physically resident until their completion event fires and a
            // drain runs. Surface the count so the report distinguishes this
            // transient hold-over from a genuine leak.
            let pending = crate::cuda::inner::deferred_free_pending();
            if pending > 0 {
                eprintln!("[gpu-mem]    deferred-free pending: {} buffer(s)", pending);
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    { let _ = step; }
}

/// P0.1: the surface-tag FFI is a pure thread-local set/get — no driver
/// calls — so this is safe on any machine that compiles the cuda feature.
#[cfg(all(test, feature = "cuda"))]
mod alloc_surface_ffi_tests {
    #[test]
    fn surface_ffi_roundtrip_and_unknown_maps_to_other() {
        for tag in 0u8..8 {
            super::nsl_gpu_set_alloc_surface(tag);
            assert_eq!(super::nsl_gpu_get_alloc_surface(), tag);
        }
        // Unknown wire values decode to Other (0), never panic.
        super::nsl_gpu_set_alloc_surface(200);
        assert_eq!(super::nsl_gpu_get_alloc_surface(), 0);
        super::nsl_gpu_set_alloc_surface(0);
    }

    /// A1: identity setter/clearer round-trips through the thread-local and
    /// the numeric getters return sane values without panicking (they lock
    /// the global allocator and read — safe with no GPU present).
    #[test]
    fn alloc_identity_and_numeric_getters() {
        super::nsl_gpu_set_alloc_identity(7, 99);
        let (op, ten) = crate::cuda::caching_allocator::get_alloc_identity();
        assert_eq!(op, Some(7));
        assert_eq!(ten, Some(99));
        super::nsl_gpu_clear_alloc_identity();
        let (op, ten) = crate::cuda::caching_allocator::get_alloc_identity();
        assert_eq!(op, None);
        assert_eq!(ten, None);

        // Getters are non-negative and never panic (empty global allocator).
        assert!(super::nsl_gpu_peak_allocated_bytes() >= 0);
        assert!(super::nsl_gpu_cumulative_alloc_count() >= 0);
        assert!(super::nsl_gpu_surface_peak_bytes(1) >= 0);
        assert!(super::nsl_gpu_surface_at_peak_bytes(1) >= 0);
    }
}

#[cfg(test)]
mod offload_envelope_tests {
    use super::*;

    /// Helper: create a 1-D f64 tensor (CPU, dtype=0, contiguous, refcount=1).
    fn make_tensor_f64(data: &[f64]) -> i64 {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, data.len() as i64);
        let ptr = crate::tensor::creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() {
            unsafe { *t.data_f64().add(i) = *v };
        }
        ptr
    }

    /// P0.2: without an eligible async fast path (CPU tensors here), the
    /// consuming copy falls back to `copy_data` + an INLINE free of src.
    #[test]
    fn copy_data_async_cpu_fallback_copies_and_consumes() {
        let src = make_tensor_f64(&[1.0, 2.0, 3.0]);
        let dst = make_tensor_f64(&[0.0, 0.0, 0.0]);
        // rc 2 so the consuming free is observable, not destructive.
        nsl_tensor_retain(src);
        nsl_tensor_copy_data_async(dst, src);
        let rc = NslTensor::from_ptr(src).refcount.load(Ordering::SeqCst);
        assert_eq!(rc, 1, "sync fallback must free src inline (consuming contract)");
        let d = NslTensor::from_ptr(dst);
        let got: Vec<f64> = (0..3).map(|i| unsafe { *d.data_f64().add(i) }).collect();
        assert_eq!(got, vec![1.0, 2.0, 3.0], "payload must be copied");
        nsl_tensor_free(src);
        nsl_tensor_free(dst);
    }

    /// P0.2: draining with nothing enqueued must be a cheap no-op — and
    /// must NOT force-initialize CUDA (safe on GPU-less machines).
    #[test]
    fn offload_drain_is_a_noop_when_nothing_enqueued() {
        nsl_offload_drain();
        nsl_offload_drain(); // idempotent
    }
}

#[cfg(test)]
mod epilog_counter_tests {
    use super::*;

    /// Helper: create a 1-D f64 tensor (CPU, dtype=0, contiguous, refcount=1).
    fn make_tensor_f64(data: &[f64]) -> i64 {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, data.len() as i64);
        let ptr = crate::tensor::creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() {
            unsafe { *t.data_f64().add(i) = *v };
        }
        ptr
    }

    #[test]
    fn counter_starts_at_zero_after_reset() {
        nsl_debug_epilog_free_reset();
        assert_eq!(nsl_debug_epilog_free_count(), 0);
    }

    #[test]
    fn counter_increments_on_valid_free() {
        nsl_debug_epilog_free_reset();
        let t = make_tensor_f64(&[1.0, 2.0, 3.0]);
        nsl_tensor_free_if_valid(t);
        assert_eq!(nsl_debug_epilog_free_count(), 1);
    }

    #[test]
    fn counter_unchanged_on_null_ptr() {
        nsl_debug_epilog_free_reset();
        nsl_tensor_free_if_valid(0);
        assert_eq!(nsl_debug_epilog_free_count(), 0);
    }
}
