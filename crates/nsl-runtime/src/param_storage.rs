//! bf16 parameter storage on the normal training schedule:
//! `train(..., precision="bf16", param_dtype="bf16-split")`.
//!
//! Under `precision="bf16"` every GEMM that reads a weight matrix reads a
//! bf16 copy of it: the weight-cast cache's image, re-cast after every
//! optimizer step (`cuda/bf16_cast_cache.rs`). Here that copy IS the
//! parameter: the train step stores the matrices it selects (those only
//! GEMMs read) as bf16 tensors, so the storage GEMMs borrow them with no cast
//! and the image cache holds nothing for them.
//!
//! The f32 master is not dropped. Two arrangements keep it:
//!
//! * **split** (`bf16-split`, [`Mode::Split`]): θ is the master's top 16
//!   bits rounded to nearest with a tie toward the truncation
//!   (`nsl_kir::kernels::optim::split_hi`), and the master's low 16 bits
//!   live here as optimizer state. The fused AdamW kernel reassembles the
//!   exact master (`join_f32`), steps it with today's f32 arithmetic and
//!   splits it again. Memory: the same 4 bytes per element as the f32
//!   parameter, and no image. Numerics: today's, except that the GEMMs read
//!   the truncation instead of the RNE image at a tie with an odd truncation
//!   (2^-17 of elements per update).
//! * **master** (`bf16-master`, [`Mode::Master`], test-only): θ is
//!   `RNE(master)` and the whole f32 master lives here. It trains
//!   bit-identically to f32 parameters, which is what checks every path
//!   that touches a bf16 parameter before the split changes a rounding.
//!   [`Mode::MasterTz`] writes the split's image instead, the reference the
//!   split is held byte-identical to.
//!
//! Life of a stored parameter:
//!
//! 1. **Conversion** ([`nsl_param_store_bf16`]): the train step emits it at
//!    the top of every micro-step, before the forward, for its selected
//!    parameters; a parameter already stored is skipped. Its first call
//!    comes after the optimizer state is allocated (`zeros_like` of the f32
//!    parameter) and after a resume has loaded the checkpoint's f32 values.
//! 2. **Forward and backward**: the storage plan lowers every GEMM that
//!    reads the parameter dtype-explicitly; `storage_gemm::operand16`
//!    borrows the bf16 data. A GEMM the activation plan alone would have
//!    left to the eager bf16 matmul mode goes through
//!    `nsl_tensor_matmul_pdt`, which keeps that mode's rule, reading
//!    [`exact_f32`] where it chose f32 operands.
//! 3. **The optimizer** (`fase_step::fase_multi_impl`): a stored parameter
//!    joins the bucket of its mode and launches the `Theta` kernels with its
//!    state from [`lookup`].
//! 4. **Saving** (`model_save`, and through it `checkpoint_save`): the exact
//!    f32 master is written under the parameter's name, so the `.nslm` is
//!    the file an f32 run with the same master writes.
//! 5. **Teardown** ([`nsl_param_store_f32`]): back to f32 (the master, or
//!    the split reassembled), so code after the train block sees the f32
//!    tensors it always did.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use nsl_abi::wire::param_storage as wire;
use nsl_kir::kernels::optim::{Image, Theta, join_f32, split_hi, split_lo};

use crate::list::NslList;
use crate::tensor::{DTYPE_BF16, DTYPE_F32, NslTensor};

/// How a stored parameter keeps its f32 master (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Mode {
    Master,
    MasterTz,
    Split,
}

impl Mode {
    fn from_wire(code: i64) -> Option<Mode> {
        match code {
            wire::MASTER => Some(Mode::Master),
            wire::MASTER_TZ => Some(Mode::MasterTz),
            wire::SPLIT => Some(Mode::Split),
            _ => None,
        }
    }

    /// The multi-tensor AdamW member that steps this storage.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))] // read by the GPU launch only
    pub(crate) fn theta(self) -> Theta {
        match self {
            Mode::Master => Theta::Bf16Master(Image::Rne),
            Mode::MasterTz => Theta::Bf16Master(Image::HalfTowardTruncation),
            Mode::Split => Theta::Bf16Split,
        }
    }

    /// The spelling of the train-block key (`bf16-master-tz` is a test hook).
    pub(crate) fn name(self) -> &'static str {
        match self {
            Mode::Master => "bf16-master",
            Mode::MasterTz => "bf16-master-tz",
            Mode::Split => "bf16-split",
        }
    }

    /// A small nonzero code for the optimizer's bucket key (0 is f32).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))] // read by the GPU launch only
    pub(crate) fn bucket_code(self) -> u8 {
        match self {
            Mode::Master => 1,
            Mode::MasterTz => 2,
            Mode::Split => 3,
        }
    }

    /// The mode of a bucket code ([`Mode::bucket_code`]); `None` for 0, an
    /// f32 parameter.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))] // read by the GPU launch only
    pub(crate) fn from_bucket_code(code: u8) -> Option<Mode> {
        match code {
            1 => Some(Mode::Master),
            2 => Some(Mode::MasterTz),
            3 => Some(Mode::Split),
            _ => None,
        }
    }

    /// The image this mode writes for an f32 master.
    fn image(self, x: f32) -> u16 {
        match self {
            Mode::Master => half::bf16::from_f32(x).to_bits(),
            Mode::MasterTz | Mode::Split => split_hi(x.to_bits()),
        }
    }
}

/// One stored parameter.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Stored {
    pub(crate) mode: Mode,
    /// The parameter's bf16 data (its tensor's `data`), to match views of it.
    pub(crate) data: u64,
    /// Master modes: the f32 master, a tensor this module owns that holds
    /// the parameter's original f32 buffer. 0 for the split.
    master: i64,
    /// Split: the low halves (`len` u16), device or host. 0 otherwise.
    lo: u64,
    pub(crate) len: usize,
    pub(crate) device: u8,
}

impl Stored {
    /// What the AdamW kernel reads beside θ: the master's f32 data, or the
    /// split's low halves.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))] // read by the GPU launch only
    pub(crate) fn aux(&self) -> u64 {
        match self.mode {
            Mode::Split => self.lo,
            Mode::Master | Mode::MasterTz => NslTensor::from_ptr_ref(self.master).data as u64,
        }
    }
}

/// The stored parameters, by tensor handle.
static STORE: Mutex<Option<HashMap<i64, Stored>>> = Mutex::new(None);
/// Parameters converted, and fused steps taken over stored parameters
/// (per parameter, like `FASE_FUSED_STEP_COUNT`): the teardown line and the
/// gates' anti-vacuity counts.
static CONVERTED: AtomicU64 = AtomicU64::new(0);
static STEPPED: AtomicU64 = AtomicU64::new(0);
/// The storage `main` declared (`wire::*`), for the banner.
static DECLARED: AtomicI64 = AtomicI64::new(wire::F32);

fn with_store<R>(f: impl FnOnce(&mut HashMap<i64, Stored>) -> R) -> R {
    let mut g = STORE.lock().unwrap_or_else(|p| p.into_inner());
    f(g.get_or_insert_with(HashMap::new))
}

/// The stored state of the parameter tensor `tp`, if it is stored bf16.
pub(crate) fn lookup(tp: i64) -> Option<Stored> {
    with_store(|m| m.get(&tp).copied())
}

/// The stored parameter whose bf16 data starts at `data` (the parameter
/// itself or a view of it, such as its transpose).
fn lookup_data(data: u64) -> Option<Stored> {
    with_store(|m| m.values().find(|s| s.data == data).copied())
}

/// Whether `data` is a stored parameter's bf16 data (cheap until a
/// parameter has been stored).
#[cfg(feature = "cuda")]
pub(crate) fn is_stored_data(data: u64) -> bool {
    CONVERTED.load(Ordering::Relaxed) > 0 && lookup_data(data).is_some()
}

/// Count `k` fused steps over stored parameters.
pub(crate) fn note_stepped(k: u64) {
    STEPPED.fetch_add(k, Ordering::Relaxed);
}

/// Parameters converted so far (anti-vacuity for gates).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_param_store_converted() -> i64 {
    CONVERTED.load(Ordering::Relaxed) as i64
}

fn die(msg: &str) -> ! {
    crate::fatal::die(crate::fatal::Fatal::Unsupported, msg)
}

/// Declare the program's parameter storage (`wire::*`). Compiled `main` calls
/// it, beside the matmul configuration, only for a program whose train block
/// writes `param_dtype=`; the banner is what a harness checks to know the
/// arm it measured stored its parameters bf16.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_set_param_storage(mode: i64) {
    let Some(m) = Mode::from_wire(mode) else {
        die(&format!(
            "nsl_set_param_storage: unknown parameter storage {mode}"
        ));
    };
    DECLARED.store(mode, Ordering::Relaxed);
    let how = match m {
        Mode::Split => {
            "the f32 master split into the bf16 parameter and a 16-bit remainder in optimizer state"
        }
        Mode::Master | Mode::MasterTz => "the f32 master kept whole in optimizer state (test mode)",
    };
    crate::nsl_log!(
        INFO,
        "nsl-matmul",
        "[nsl-matmul] parameter storage: {} (train param_dtype: the weight matrices the GEMMs read are \
         stored bf16, {how})",
        m.name()
    );
}

/// Store the parameters `param_list[idx]`, for each index in `idx_list`, as
/// bf16 under `mode` (`wire::*`). A parameter already stored is skipped, so
/// the train step calls this at the top of every micro-step. Returns how
/// many were converted by this call.
///
/// Refused (fatal), each a contract the compiler or the model broke: an
/// unknown mode, an index out of range, a parameter stored under another
/// mode, one that is not a plain owning contiguous f32 tensor, and one whose
/// storage another parameter of the list shares (a tied weight: the
/// optimizer would step one storage twice).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_param_store_bf16(param_list: i64, idx_list: i64, mode: i64) -> i64 {
    let Some(mode) = Mode::from_wire(mode) else {
        die(&format!(
            "nsl_param_store_bf16: unknown parameter storage {mode}"
        ));
    };
    let params = NslList::from_ptr(param_list);
    let idxs = NslList::from_ptr(idx_list);
    let count = params.len as usize;
    let mut newly = 0usize;
    let mut bytes = 0usize;
    for k in 0..idxs.len as usize {
        let idx = unsafe { *idxs.data.add(k) };
        if idx < 0 || idx as usize >= count {
            die(&format!(
                "nsl_param_store_bf16: parameter index {idx} out of range 0..{count}"
            ));
        }
        let tp = unsafe { *params.data.add(idx as usize) };
        if let Some(s) = lookup(tp) {
            if s.mode != mode {
                die(&format!(
                    "nsl_param_store_bf16: parameter {idx} is stored {} and was asked for {}",
                    s.mode.name(),
                    mode.name()
                ));
            }
            continue;
        }
        let t = NslTensor::from_ptr(tp);
        if t.dtype != DTYPE_F32
            || t.owns_data == 0
            || t.data_owner != 0
            || t.slab_managed != 0
            || !t.is_contiguous()
            || t.data.is_null()
        {
            die(&format!(
                "param_dtype=\"{}\": parameter {idx} cannot be stored bf16: it is not a plain owning contiguous \
                 f32 tensor (dtype {}, owns_data {}, a view: {}, slab: {}, contiguous: {}). A CPU model holds \
                 f64 parameters; bf16 parameter storage needs the model on the GPU (`m.to(cuda)`)",
                mode.name(),
                t.dtype,
                t.owns_data,
                t.data_owner != 0,
                t.slab_managed != 0,
                t.is_contiguous()
            ));
        }
        for j in 0..count {
            let other = unsafe { *params.data.add(j) };
            if j != idx as usize && (other == tp || NslTensor::from_ptr_ref(other).data == t.data) {
                die(&format!(
                    "param_dtype=\"{}\": parameters {idx} and {j} share one storage (tied weights); the \
                     optimizer steps each list entry once, so a tied matrix cannot be stored bf16. Untie it \
                     or train with f32 parameters",
                    mode.name()
                ));
            }
        }
        let stored = if t.device > 0 {
            convert_gpu(t, mode)
        } else {
            convert_cpu(t, mode)
        };
        bytes += stored.len * 2;
        with_store(|m| m.insert(tp, stored));
        newly += 1;
    }
    if newly > 0 {
        let before = CONVERTED.fetch_add(newly as u64, Ordering::Relaxed);
        if before == 0 {
            crate::nsl_log!(
                INFO,
                "param-storage",
                "[param-storage] {newly} parameters stored {} ({:.3} GiB of bf16 parameters)",
                mode.name(),
                bytes as f64 / (1u64 << 30) as f64
            );
        }
    }
    newly as i64
}

/// A new tensor with `t`'s shape and strides over `data` (f32).
fn f32_tensor_like(t: &NslTensor, data: *mut c_void, owns: u8) -> i64 {
    let n = t.ndim as usize;
    let bytes = n * std::mem::size_of::<i64>();
    let shape = crate::memory::checked_alloc(bytes) as *mut i64;
    let strides = crate::memory::checked_alloc(bytes) as *mut i64;
    unsafe {
        std::ptr::copy_nonoverlapping(t.shape, shape, n);
        std::ptr::copy_nonoverlapping(t.strides, strides, n);
    }
    Box::into_raw(Box::new(NslTensor::new(
        data, shape, strides, t.ndim, t.len, t.device, DTYPE_F32, owns, 0,
    ))) as i64
}

#[cfg(feature = "cuda")]
fn convert_gpu(t: &mut NslTensor, mode: Mode) -> Stored {
    use crate::cuda::caching_allocator::{AllocPool, PoolGuard, SurfaceGuard, SurfaceTag};
    use crate::cuda::inner;
    inner::ensure_context();
    let len = t.len as usize;
    let old = t.data as u64;
    // The cast cache keys images by f32 data pointer. The master keeps this
    // pointer and nothing marks it stepped any more, so an image registered
    // for it would go stale; drop it now (a free evicts by itself).
    crate::cuda::bf16_cast_cache::evict(old);
    inner::set_oom_context("param_store_bf16");
    // The parameter's storage for the rest of the block: the weights surface
    // and the persistent pool, as the f32 parameter's was (the conversion
    // runs inside the step, where the transient pool is current).
    let _surface = SurfaceGuard::new(SurfaceTag::Weights);
    let _pool = PoolGuard::new(AllocPool::Persistent);
    let hi = inner::alloc_managed(len * 2) as u64;
    let (master, lo) = match mode {
        Mode::Master => {
            crate::cuda::gpu_cast_raw_f32_to_bf16(old, hi, len);
            (f32_tensor_like(t, old as *mut c_void, 1), 0)
        }
        Mode::MasterTz => {
            let scratch = inner::alloc_managed(len * 2) as u64;
            crate::cuda::param_store_gpu::gpu_param_split(old, hi, scratch, len);
            inner::free_managed(scratch as *mut c_void);
            (f32_tensor_like(t, old as *mut c_void, 1), 0)
        }
        Mode::Split => {
            let lo = inner::alloc_managed(len * 2) as u64;
            crate::cuda::param_store_gpu::gpu_param_split(old, hi, lo, len);
            inner::free_managed(old as *mut c_void);
            (0, lo)
        }
    };
    t.data = hi as *mut c_void;
    t.dtype = DTYPE_BF16;
    Stored {
        mode,
        data: hi,
        master,
        lo,
        len,
        device: t.device,
    }
}

#[cfg(not(feature = "cuda"))]
fn convert_gpu(_t: &mut NslTensor, _mode: Mode) -> Stored {
    crate::fatal::cuda_not_compiled();
}

fn convert_cpu(t: &mut NslTensor, mode: Mode) -> Stored {
    let len = t.len as usize;
    let old = t.data as *mut f32;
    let src = unsafe { std::slice::from_raw_parts(old, len) };
    let hi = crate::memory::checked_alloc(len * 2) as *mut u16;
    let his = unsafe { std::slice::from_raw_parts_mut(hi, len) };
    let (master, lo) = match mode {
        Mode::Master | Mode::MasterTz => {
            for (h, &x) in his.iter_mut().zip(src) {
                *h = mode.image(x);
            }
            (f32_tensor_like(t, old as *mut c_void, 1), 0)
        }
        Mode::Split => {
            let lo = crate::memory::checked_alloc(len * 2) as *mut u16;
            let los = unsafe { std::slice::from_raw_parts_mut(lo, len) };
            for ((h, l), &x) in his.iter_mut().zip(los.iter_mut()).zip(src) {
                *h = split_hi(x.to_bits());
                *l = split_lo(x.to_bits());
            }
            crate::tensor::free_host_tensor_data(old as *mut c_void, len * 4);
            (0, lo as u64)
        }
    };
    t.data = hi as *mut c_void;
    t.dtype = DTYPE_BF16;
    Stored {
        mode,
        data: hi as u64,
        master,
        lo,
        len,
        device: 0,
    }
}

/// The exact f32 value of the stored parameter whose data `t` (a bf16
/// tensor: the parameter or a view of it) starts at, as a new f32 tensor
/// with `t`'s shape and strides that the caller frees: the master itself
/// (not owned: it stays the store's), or the split reassembled into a new
/// buffer. `None` when `t` is not a stored parameter.
pub(crate) fn exact_f32(t_ptr: i64) -> Option<i64> {
    let t = NslTensor::from_ptr_ref(t_ptr);
    if t.dtype != DTYPE_BF16 {
        return None;
    }
    let s = lookup_data(t.data as u64)?;
    Some(match s.mode {
        Mode::Master | Mode::MasterTz => {
            f32_tensor_like(t, NslTensor::from_ptr_ref(s.master).data, 0)
        }
        Mode::Split => f32_tensor_like(t, joined(&s), 1),
    })
}

/// A new f32 buffer holding the split parameter `s` reassembled.
fn joined(s: &Stored) -> *mut c_void {
    if s.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let buf = crate::cuda::inner::alloc_managed(s.len * 4);
            crate::cuda::param_store_gpu::gpu_param_join(s.data, s.lo, buf as u64, s.len);
            return buf;
        }
        #[cfg(not(feature = "cuda"))]
        crate::fatal::cuda_not_compiled();
    }
    let buf = crate::memory::checked_alloc(s.len * 4) as *mut u32;
    let (his, los, out) = unsafe {
        (
            std::slice::from_raw_parts(s.data as *const u16, s.len),
            std::slice::from_raw_parts(s.lo as *const u16, s.len),
            std::slice::from_raw_parts_mut(buf, s.len),
        )
    };
    for ((o, &h), &l) in out.iter_mut().zip(his).zip(los) {
        *o = join_f32(h, l);
    }
    buf as *mut c_void
}

/// `model_save`'s view of a parameter list: the list itself when nothing in
/// it is stored bf16, otherwise a new list with each stored parameter
/// replaced by its exact f32 master, plus the temporaries to free.
pub(crate) fn save_view(list_ptr: i64) -> Option<(i64, Vec<i64>)> {
    let list = NslList::from_ptr(list_ptr);
    let ptrs: Vec<i64> = (0..list.len as usize)
        .map(|i| unsafe { *list.data.add(i) })
        .collect();
    if !ptrs.iter().any(|&p| p != 0 && lookup(p).is_some()) {
        return None;
    }
    let out = crate::list::nsl_list_new();
    let mut temps = Vec::new();
    for p in ptrs {
        match lookup(p).and_then(|_| exact_f32(p)) {
            Some(f) => {
                crate::list::nsl_list_push(out, f);
                temps.push(f);
            }
            None => crate::list::nsl_list_push(out, p),
        }
    }
    Some((out, temps))
}

/// Free what [`save_view`] made.
pub(crate) fn release_save_view(list_ptr: i64, temps: Vec<i64>) {
    for t in temps {
        crate::tensor::nsl_tensor_free(t);
    }
    crate::list::nsl_list_free(list_ptr);
}

/// The fused AdamW step of one stored parameter outside a batched launch:
/// its exact f32 master stepped by the f32 step (`fase_step`), then θ (and
/// the split's low half) rewritten from it. Host parameters only: on the GPU
/// every stored parameter batches (conversion refuses the tied storage that
/// could not), so reaching here with one is a bug, refused.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_unbatched(
    tp: i64,
    m: i64,
    v: i64,
    mp: i64,
    lr: f64,
    beta1: f64,
    one_minus_beta1: f64,
    beta2: f64,
    one_minus_beta2: f64,
    eps: f64,
    wd: f64,
    bc1_inv: f64,
    bc2_inv: f64,
) {
    let s = lookup(tp).expect("step_unbatched: a stored parameter");
    if s.device > 0 {
        die(&format!(
            "param_dtype=\"{}\": a bf16-stored GPU parameter missed the batched AdamW launch (its moments or \
             gradient are not uniform f32 GPU tensors of its length)",
            s.mode.name()
        ));
    }
    let f = exact_f32(tp).expect("a stored parameter has an exact f32 value");
    crate::fase_step::nsl_fase_fused_adamw_step(
        f,
        m,
        v,
        mp,
        lr,
        beta1,
        one_minus_beta1,
        beta2,
        one_minus_beta2,
        eps,
        wd,
        bc1_inv,
        bc2_inv,
    );
    let x =
        unsafe { std::slice::from_raw_parts(NslTensor::from_ptr_ref(f).data as *const f32, s.len) };
    let his = unsafe { std::slice::from_raw_parts_mut(s.data as *mut u16, s.len) };
    match s.mode {
        Mode::Master | Mode::MasterTz => {
            for (h, &xv) in his.iter_mut().zip(x) {
                *h = s.mode.image(xv);
            }
        }
        Mode::Split => {
            let los = unsafe { std::slice::from_raw_parts_mut(s.lo as *mut u16, s.len) };
            for ((h, l), &xv) in his.iter_mut().zip(los.iter_mut()).zip(x) {
                *h = split_hi(xv.to_bits());
                *l = split_lo(xv.to_bits());
            }
        }
    }
    crate::tensor::nsl_tensor_free(f);
    note_stepped(1);
}

/// Widen every stored parameter of `param_list` back to f32 (the master, or
/// the split reassembled) and forget it: the train block's teardown, so code
/// after the block sees the f32 tensors it always did. Returns how many were
/// widened, and prints the counters line when any were.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_param_store_f32(param_list: i64) -> i64 {
    let params = NslList::from_ptr(param_list);
    let mut widened = 0u64;
    let mut mode = None;
    for i in 0..params.len as usize {
        let tp = unsafe { *params.data.add(i) };
        let Some(s) = with_store(|m| m.remove(&tp)) else {
            continue;
        };
        let t = NslTensor::from_ptr(tp);
        let f32_data = match s.mode {
            Mode::Master | Mode::MasterTz => {
                let m = NslTensor::from_ptr(s.master);
                let d = m.data;
                // The master tensor gives its buffer back to the parameter.
                m.owns_data = 0;
                m.data = std::ptr::null_mut();
                crate::tensor::nsl_tensor_free(s.master);
                d
            }
            Mode::Split => {
                #[cfg(feature = "cuda")]
                let _guards = (s.device > 0).then(|| {
                    use crate::cuda::caching_allocator::{
                        AllocPool, PoolGuard, SurfaceGuard, SurfaceTag,
                    };
                    (
                        SurfaceGuard::new(SurfaceTag::Weights),
                        PoolGuard::new(AllocPool::Persistent),
                    )
                });
                let d = joined(&s);
                free_halfwords(s.lo, s.len, s.device);
                d
            }
        };
        free_halfwords(s.data, s.len, s.device);
        t.data = f32_data;
        t.dtype = DTYPE_F32;
        widened += 1;
        mode = Some(s.mode);
    }
    if let Some(mode) = mode {
        crate::nsl_log!(
            INFO,
            "param-storage",
            "[param-storage] teardown: {widened} parameters widened back to f32 from {} ({} converted, {} fused \
             steps over stored parameters, {} parameter GEMMs on f32 operands)",
            mode.name(),
            CONVERTED.load(Ordering::Relaxed),
            STEPPED.load(Ordering::Relaxed),
            crate::storage_dt::PDT_F32.load(Ordering::Relaxed)
        );
    }
    widened as i64
}

/// Free a buffer of `len` 16-bit words this module allocated.
fn free_halfwords(ptr: u64, len: usize, device: u8) {
    if device > 0 {
        #[cfg(feature = "cuda")]
        crate::cuda::inner::free_managed(ptr as *mut c_void);
        #[cfg(not(feature = "cuda"))]
        crate::fatal::cuda_not_compiled();
    } else {
        unsafe { crate::memory::checked_free(ptr as *mut u8, len * 2) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::list::{nsl_list_new, nsl_list_push};

    fn host_f32(vals: &[f32], shape: &[i64]) -> i64 {
        let p = crate::cpu::create_tensor_with_shape_rs_dtype(shape, 1);
        let t = NslTensor::from_ptr(p);
        for (i, &x) in vals.iter().enumerate() {
            unsafe { *t.data_f32().add(i) = x };
        }
        p
    }

    fn bits_of(p: i64) -> Vec<u32> {
        let t = NslTensor::from_ptr_ref(p);
        (0..t.len as usize)
            .map(|i| unsafe { *(t.data as *const u32).add(i) })
            .collect()
    }

    fn halves_of(p: i64) -> Vec<u16> {
        let t = NslTensor::from_ptr_ref(p);
        (0..t.len as usize)
            .map(|i| unsafe { *(t.data as *const u16).add(i) })
            .collect()
    }

    /// Values of every class, including the ties the split and RNE disagree on.
    fn values() -> Vec<f32> {
        let mut v: Vec<f32> = [
            0x3F80_8000u32,
            0x3F81_8000,
            0xBF81_8000,
            0x0000_8000,
            0x7F7F_FFFF,
            0x0000_0001,
            0x8000_0000,
        ]
        .iter()
        .map(|&b| f32::from_bits(b))
        .collect();
        v.extend((0..41).map(|i| ((i as f32) * 0.37).sin() * 0.03));
        v
    }

    /// Host conversion and teardown, every mode: θ is bf16 with the mode's
    /// image, the exact f32 value is the original bit for bit, `model_save`'s
    /// view holds the original, and teardown restores the f32 tensor exactly.
    #[test]
    fn host_parameters_convert_round_trip_and_widen_back_exactly() {
        for (code, mode) in [
            (wire::MASTER, Mode::Master),
            (wire::MASTER_TZ, Mode::MasterTz),
            (wire::SPLIT, Mode::Split),
        ] {
            let vals = values();
            let want: Vec<u32> = vals.iter().map(|x| x.to_bits()).collect();
            let a = host_f32(&vals, &[6, 8]);
            let b = host_f32(&vals[..8], &[8]);
            let list = nsl_list_new();
            nsl_list_push(list, a);
            nsl_list_push(list, b);
            let idx = nsl_list_new();
            nsl_list_push(idx, 0);
            assert_eq!(nsl_param_store_bf16(list, idx, code), 1);
            assert_eq!(
                nsl_param_store_bf16(list, idx, code),
                0,
                "a stored parameter is skipped"
            );
            let t = NslTensor::from_ptr_ref(a);
            assert_eq!(t.dtype, DTYPE_BF16);
            let img: Vec<u16> = vals.iter().map(|&x| mode.image(x)).collect();
            assert_eq!(
                halves_of(a),
                img,
                "{mode:?}: the parameter is its master's image"
            );
            // The odd tie 0x3F81_8000: RNE carries, the split keeps the truncation.
            assert_eq!(
                halves_of(a)[1],
                if mode == Mode::Master { 0x3F82 } else { 0x3F81 }
            );
            let f = exact_f32(a).expect("stored");
            assert_eq!(bits_of(f), want, "{mode:?}: the exact f32 value");
            crate::tensor::nsl_tensor_free(f);
            let (view, temps) = save_view(list).expect("a stored parameter in the list");
            assert_eq!(bits_of(unsafe { *NslList::from_ptr(view).data }), want);
            assert_eq!(
                unsafe { *NslList::from_ptr(view).data.add(1) },
                b,
                "an f32 parameter is saved as itself"
            );
            release_save_view(view, temps);
            assert_eq!(
                NslTensor::from_ptr_ref(b).dtype,
                DTYPE_F32,
                "only the selected index is stored"
            );
            assert_eq!(nsl_param_store_f32(list), 1);
            assert_eq!(NslTensor::from_ptr_ref(a).dtype, DTYPE_F32);
            assert_eq!(
                bits_of(a),
                want,
                "{mode:?}: teardown restores the master exactly"
            );
            assert!(lookup(a).is_none());
            crate::tensor::nsl_tensor_free(a);
            crate::tensor::nsl_tensor_free(b);
        }
    }

    /// The unbatched host step is the f32 step of the exact master, then the
    /// mode's split or image of the result.
    #[test]
    fn the_host_step_is_the_f32_step_of_the_exact_master() {
        for (code, mode) in [(wire::MASTER, Mode::Master), (wire::SPLIT, Mode::Split)] {
            let vals = values();
            let n = vals.len();
            let g: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).cos() * 1e-4).collect();
            let mk = || {
                (
                    host_f32(&vec![0.0; n], &[n as i64]),
                    host_f32(&vec![0.0; n], &[n as i64]),
                    host_f32(&g, &[n as i64]),
                )
            };
            let p = host_f32(&vals, &[n as i64]);
            let list = nsl_list_new();
            nsl_list_push(list, p);
            let idx = nsl_list_new();
            nsl_list_push(idx, 0);
            nsl_param_store_bf16(list, idx, code);
            let (m, v, mp) = mk();
            step_unbatched(
                p, m, v, mp, 1e-3, 0.9, 0.1, 0.999, 0.001, 1e-8, 0.1, 10.0, 1000.0,
            );
            let r = host_f32(&vals, &[n as i64]);
            let (rm, rv, rmp) = mk();
            crate::fase_step::nsl_fase_fused_adamw_step(
                r, rm, rv, rmp, 1e-3, 0.9, 0.1, 0.999, 0.001, 1e-8, 0.1, 10.0, 1000.0,
            );
            let want = bits_of(r);
            let f = exact_f32(p).unwrap();
            assert_eq!(bits_of(f), want, "{mode:?}: the master steps as f32");
            crate::tensor::nsl_tensor_free(f);
            let img: Vec<u16> = want
                .iter()
                .map(|&b| mode.image(f32::from_bits(b)))
                .collect();
            assert_eq!(
                halves_of(p),
                img,
                "{mode:?}: θ is the stepped master's image"
            );
            assert_eq!(bits_of(m), bits_of(rm));
            assert_eq!(bits_of(v), bits_of(rv));
            nsl_param_store_f32(list);
            for t in [p, m, v, mp, r, rm, rv, rmp] {
                crate::tensor::nsl_tensor_free(t);
            }
        }
    }

    /// On the GPU, through the runtime's launchers (their tables and argument
    /// order): each bf16 member of the AdamW family equals the f32 multi
    /// launch run on its exact f32 master, bit for bit (m, v, the zeroed
    /// gradient, the master or the split halves or the SR rounding of θ').
    /// Realistic magnitudes (weights ~0.05, gradients 1e-9..1e-3, a few
    /// corners), several blocks per parameter, ragged tails, both decay
    /// buckets, the clip fold at 1.0 and 0.37, m f32 and bf16. Then the
    /// conversion kernels over every low half of a set of high halves (ties,
    /// carries into the exponent, ±∞, NaN) against the host split.
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires CUDA GPU"]
    fn the_gpu_bf16_parameter_launches_are_the_f32_launch_of_the_exact_master() {
        use crate::cuda::inner::{alloc_managed, free_managed, memcpy_dtoh, memcpy_htod};
        use crate::cuda::param_store_gpu::{
            AdamwScalars, ParamBucket, gpu_param_join, gpu_param_split,
        };
        if crate::cuda::nsl_cuda_init() != 0 {
            crate::nsl_log!(WARN, "param-storage", "skipping: no CUDA device");
            return;
        }
        let up = |bytes: &[u8]| -> u64 {
            let d = alloc_managed(bytes.len().max(1));
            memcpy_htod(d, bytes.as_ptr() as *const c_void, bytes.len());
            d as u64
        };
        let down = |d: u64, n: usize| -> Vec<u8> {
            let mut v = vec![0u8; n];
            memcpy_dtoh(v.as_mut_ptr() as *mut c_void, d as *mut c_void, n);
            v
        };
        let le16 = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let le32 = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let w16 = |b: Vec<u8>| {
            b.chunks(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect::<Vec<u16>>()
        };
        let w32 = |b: Vec<u8>| {
            b.chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<u32>>()
        };
        let rne = |b: u32| half::bf16::from_f32(f32::from_bits(b)).to_bits();
        let nan = |b: u32| b & 0x7FFF_FFFF > 0x7F80_0000;

        let lens = [257usize, 1, 70_000, 1000, 255];
        let mut s = 0x5EEDu64;
        let mut next = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let corners = [
            0u32,
            0x8000_0000,
            0x7F80_0000,
            0x0000_0001,
            0x3F81_8000,
            0x7F7F_FFFF,
        ];
        struct P {
            theta: Vec<u32>,
            m: Vec<u32>,
            v: Vec<u32>,
            g: Vec<u32>,
        }
        let ps: Vec<P> = lens
            .iter()
            .map(|&n| {
                let theta = (0..n)
                    .map(|k| {
                        if k % 97 == 5 {
                            corners[(k / 97) % corners.len()]
                        } else {
                            (next() * 0.05).to_bits()
                        }
                    })
                    .collect();
                let scale = |k: usize| [1e-9f32, 1e-6, 1e-4, 1e-3][k % 4];
                let g: Vec<u32> = (0..n).map(|k| (next() * scale(k)).to_bits()).collect();
                let m = g
                    .iter()
                    .map(|&b| (f32::from_bits(b) * 0.3).to_bits())
                    .collect();
                let v = g
                    .iter()
                    .map(|&b| (f32::from_bits(b) * f32::from_bits(b) * 2.0).to_bits())
                    .collect();
                P { theta, m, v, g }
            })
            .collect();

        for theta in Theta::BF16 {
            for m_bf16 in [false, true] {
                for (has_wd, mp_scale) in [(false, 1.0f32), (true, 0.37)] {
                    let sc = AdamwScalars {
                        b1: 0.9,
                        omb1: 0.1,
                        b2: 0.95,
                        omb2: 0.05,
                        eps: 1e-8,
                        neg_lr: -1.5e-5,
                        neg_lr_wd: if has_wd { -1.5e-6 } else { 0.0 },
                        bc1: 1.0 / (1.0 - 0.9f32.powi(7)),
                        bc2: 1.0 / (1.0 - 0.95f32.powi(7)),
                        has_wd,
                    };
                    let key = sr_bf16_key();
                    // The f32 θ each variant's update starts from.
                    let eff = |t: &[u32]| -> Vec<u32> {
                        match theta {
                            Theta::Bf16Split => t
                                .iter()
                                .map(|&b| join_f32(split_hi(b), split_lo(b)))
                                .collect(),
                            Theta::Bf16Sr => t.iter().map(|&b| b & 0xFFFF_0000).collect(),
                            _ => t.to_vec(),
                        }
                    };
                    let m_bytes = |p: &P| {
                        if m_bf16 {
                            le16(&p.m.iter().map(|&b| rne(b)).collect::<Vec<_>>())
                        } else {
                            le32(&p.m)
                        }
                    };
                    // Reference: the f32 multi launch.
                    let r: Vec<[u64; 4]> = ps
                        .iter()
                        .map(|p| {
                            [
                                up(&le32(&eff(&p.theta))),
                                up(&m_bytes(p)),
                                up(&le32(&p.v)),
                                up(&le32(&p.g)),
                            ]
                        })
                        .collect();
                    let col =
                        |q: &[[u64; 4]], k: usize| q.iter().map(|x| x[k]).collect::<Vec<u64>>();
                    let lens32: Vec<u32> = lens.iter().map(|&n| n as u32).collect();
                    crate::cuda::gpu_fase_fused_adamw_step_multi(
                        &col(&r, 0),
                        &col(&r, 1),
                        &col(&r, 2),
                        &col(&r, 3),
                        &lens32,
                        sc.b1,
                        sc.omb1,
                        sc.b2,
                        sc.omb2,
                        sc.eps,
                        sc.neg_lr,
                        sc.neg_lr_wd,
                        sc.bc1,
                        sc.bc2,
                        sc.has_wd,
                        mp_scale,
                        m_bf16,
                    );
                    // Candidate: the variant's launch.
                    let c: Vec<[u64; 5]> = ps
                        .iter()
                        .map(|p| {
                            let (t, w) = match theta {
                                Theta::Bf16Master(_) => (
                                    le16(&p.theta.iter().map(|&b| rne(b)).collect::<Vec<_>>()),
                                    le32(&p.theta),
                                ),
                                Theta::Bf16Split => (
                                    le16(&p.theta.iter().map(|&b| split_hi(b)).collect::<Vec<_>>()),
                                    le16(&p.theta.iter().map(|&b| split_lo(b)).collect::<Vec<_>>()),
                                ),
                                _ => (
                                    le16(
                                        &p.theta
                                            .iter()
                                            .map(|&b| (b >> 16) as u16)
                                            .collect::<Vec<_>>(),
                                    ),
                                    vec![0u8; 2],
                                ),
                            };
                            [
                                up(&t),
                                up(&w),
                                up(&m_bytes(p)),
                                up(&le32(&p.v)),
                                up(&le32(&p.g)),
                            ]
                        })
                        .collect();
                    let ccol = |k: usize| c.iter().map(|x| x[k]).collect::<Vec<u64>>();
                    let ctr: Vec<u64> = (0..lens.len() as u64).map(|j| j << 40).collect();
                    let (masters, ctrs) = match theta {
                        Theta::Bf16Sr => (vec![], ctr.clone()),
                        _ => (ccol(1), vec![]),
                    };
                    crate::cuda::param_store_gpu::gpu_fase_fused_adamw_step_multi_param(
                        theta,
                        m_bf16,
                        &ParamBucket {
                            theta: &ccol(0),
                            master: &masters,
                            m: &ccol(2),
                            v: &ccol(3),
                            mp: &ccol(4),
                            ctr: &ctrs,
                            lens: &lens32,
                        },
                        sc,
                        mp_scale,
                        key,
                    );
                    let label =
                        format!("{theta:?} m_bf16={m_bf16} wd={has_wd} mp_scale={mp_scale}");
                    for (j, &n) in lens.iter().enumerate() {
                        let mb = if m_bf16 { 2 } else { 4 };
                        assert_eq!(
                            down(c[j][2], n * mb),
                            down(r[j][1], n * mb),
                            "{label}: param {j} m"
                        );
                        assert_eq!(
                            down(c[j][3], n * 4),
                            down(r[j][2], n * 4),
                            "{label}: param {j} v"
                        );
                        assert!(
                            w32(down(c[j][4], n * 4)).iter().all(|&g| g == 0),
                            "{label}: param {j} gradient not zeroed"
                        );
                        assert!(
                            w32(down(r[j][3], n * 4)).iter().all(|&g| g == 0),
                            "the reference zeroes its gradient"
                        );
                        let want = w32(down(r[j][0], n * 4));
                        let got = w16(down(c[j][0], n * 2));
                        let aux16 = if theta == Theta::Bf16Split {
                            w16(down(c[j][1], n * 2))
                        } else {
                            vec![]
                        };
                        let aux32 = if matches!(theta, Theta::Bf16Master(_)) {
                            w32(down(c[j][1], n * 4))
                        } else {
                            vec![]
                        };
                        for e in 0..n {
                            let x = want[e];
                            match theta {
                                Theta::Bf16Master(image) => {
                                    assert_eq!(aux32[e], x, "{label}: param {j}[{e}] master");
                                    let img = if image == Image::Rne {
                                        rne(x)
                                    } else {
                                        split_hi(x)
                                    };
                                    assert!(
                                        got[e] == img || (nan(x) && nan((got[e] as u32) << 16)),
                                        "{label}: param {j}[{e}] image {:#06x} of {x:#010x}",
                                        got[e]
                                    );
                                }
                                Theta::Bf16Split => {
                                    assert_eq!(
                                        (got[e], aux16[e]),
                                        (split_hi(x), split_lo(x)),
                                        "{label}: param {j}[{e}] split of {x:#010x}"
                                    );
                                    if !nan(x) {
                                        assert_eq!(
                                            join_f32(got[e], aux16[e]),
                                            x,
                                            "{label}: param {j}[{e}] reassembly"
                                        );
                                    }
                                }
                                Theta::Bf16Sr => {
                                    let d = crate::sr_bf16::sr_mix64(
                                        key,
                                        ctr[j].wrapping_add(e as u64),
                                    ) as u16;
                                    assert_eq!(
                                        got[e],
                                        crate::sr_bf16::sr_bf16_round(f32::from_bits(x), d),
                                        "{label}: param {j}[{e}] SR of {x:#010x}"
                                    );
                                }
                                Theta::F32 => unreachable!(),
                            }
                        }
                        for p in r[j].iter().chain(c[j].iter()) {
                            free_managed(*p as *mut c_void);
                        }
                    }
                }
            }
        }

        // The conversion kernels: every low half under high halves of every
        // class, against the host split; and back.
        let tops = [
            0x0000u32, 0x0001, 0x3F80, 0x3F81, 0x8000, 0xBF81, 0x7F7F, 0xFF7F, 0x7F80, 0x7FC0,
            0xFFFF, 0x0080,
        ];
        let bits: Vec<u32> = tops
            .iter()
            .flat_map(|&t| (0..=0xFFFFu32).map(move |l| (t << 16) | l))
            .collect();
        let n = bits.len();
        let (src, hi, lo, back) = (
            up(&le32(&bits)),
            alloc_managed(2 * n) as u64,
            alloc_managed(2 * n) as u64,
            alloc_managed(4 * n) as u64,
        );
        gpu_param_split(src, hi, lo, n);
        gpu_param_join(hi, lo, back, n);
        let (h, l, b) = (
            w16(down(hi, 2 * n)),
            w16(down(lo, 2 * n)),
            w32(down(back, 4 * n)),
        );
        for i in 0..n {
            assert_eq!(
                (h[i], l[i]),
                (split_hi(bits[i]), split_lo(bits[i])),
                "split {:#010x}",
                bits[i]
            );
            assert_eq!(b[i], join_f32(h[i], l[i]), "join {:#010x}", bits[i]);
            if !nan(bits[i]) {
                assert_eq!(b[i], bits[i], "round trip {:#010x}", bits[i]);
            }
        }
        for p in [src, hi, lo, back] {
            free_managed(p as *mut c_void);
        }
    }

    /// A key of the normal-schedule SR mode for the GPU test.
    #[cfg(feature = "cuda")]
    fn sr_bf16_key() -> u64 {
        crate::sr_bf16::sr_step_key_mixed(42, 7)
    }
}
