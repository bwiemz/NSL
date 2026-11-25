//! MXFP8 projection GEMMs: the forward `x @ W`, the input gradient
//! `g @ W^T` (alone, or summed into an f32 gradient), and the fused weight
//! gradient `m += s · x^T g`, each with both operands quantized to E4M3 with
//! one UE8M0 scale per 32 elements along K (`crate::mxfp8`) and multiplied by
//! cublasLt with f32 accumulation.
//!
//! Codegen sends a GEMM here only when it reads a parameter (`W`, or `W`'s
//! last-two-axes transpose) and the program asked for MXFP8. Each entry point
//! returns `None` / `false` (nothing written) for a shape outside the
//! contract, counted in [`counts`], and the caller takes the bf16 path.
//!
//! cublasLt wants both operands K-major (`D = A^T B`, column-major), so in
//! the row-major terms of `storage_gemm` (`C[m, n] = A[m, k] B[k, n]` issued
//! as `C^T = B^T A^T`):
//!
//! | GEMM | cublasLt A (`n` rows of K) | cublasLt B (`m` rows of K) |
//! |---|---|---|
//! | forward, `b = W [k, n]` | `W` blocks down its columns, transposed (`ColsT`) | `x` by rows |
//! | input gradient, `b = W^T`, `W [n, k]` | `W` by rows (`Rows`) | `g` by rows |
//! | weight gradient `[d_in, d_out]`, K = tokens | `g` down its columns (`ColsT`) | `x` down its columns (`ColsT`) |
//!
//! A parameter's two images come from the weight-cast cache
//! (`bf16_cast_cache::acquire_mx`): quantized once per optimizer step per
//! direction, invalidated by everything that stales its bf16 image. Before
//! the first step (nothing registered yet) a weight is quantized into
//! scratch on every call: the same bytes, only slower.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

use nsl_kir::kernels::mx_quant::scale_bytes;

use super::bf16_cast_cache::{self, MxOrient};
use crate::tensor::{DTYPE_BF16, DTYPE_F32, NslTensor};

static FWD: AtomicU64 = AtomicU64::new(0);
static DGRAD: AtomicU64 = AtomicU64::new(0);
static DGRAD_ACC: AtomicU64 = AtomicU64::new(0);
static WGRAD: AtomicU64 = AtomicU64::new(0);
static DECLINED: AtomicU64 = AtomicU64::new(0);

/// `[forward, input gradient, summed input gradient, weight gradient,
/// declined]` MXFP8 projection GEMMs so far.
pub(crate) fn counts() -> [u64; 5] {
    [&FWD, &DGRAD, &DGRAD_ACC, &WGRAD, &DECLINED].map(|c| c.load(Ordering::Relaxed))
}

fn decline<T>(v: T) -> T {
    DECLINED.fetch_add(1, Ordering::Relaxed);
    v
}

/// A quantized operand: scratch (freed on drop, after the GEMM is enqueued:
/// the single-stream invariant `Bf16Scratch` documents) or a borrowed weight
/// image.
struct MxOp {
    data: u64,
    scales: u64,
    owned: bool,
}

impl Drop for MxOp {
    fn drop(&mut self) {
        if self.owned {
            super::inner::free_managed(self.data as *mut c_void);
            super::inner::free_managed(self.scales as *mut c_void);
        }
    }
}

fn scratch(data_bytes: usize, scale_bytes: usize) -> MxOp {
    MxOp {
        data: super::inner::alloc_managed(data_bytes) as u64,
        scales: super::inner::alloc_managed(scale_bytes) as u64,
        owned: true,
    }
}

/// f32 or bf16 at an address the quantizers' 4-element vector loads accept
/// (16 bytes covers both): a view at an odd offset declines rather than
/// fault.
fn storage_dtype(t: &NslTensor) -> bool {
    (t.dtype == DTYPE_F32 || t.dtype == DTYPE_BF16) && (t.data as u64).is_multiple_of(16)
}

/// Quantize the contiguous `[r, c]` tensor `t` along `orient` into `into`.
fn quantize(t: &NslTensor, r: usize, c: usize, orient: MxOrient, into: &MxOp) {
    let bf16 = t.dtype == DTYPE_BF16;
    match orient {
        MxOrient::Rows => {
            super::mx_quant::quant_rows(t.data as u64, bf16, r, c, into.data, into.scales)
        }
        MxOrient::ColsT => {
            super::mx_quant::quant_cols_t(t.data as u64, bf16, r, c, into.data, into.scales)
        }
    }
}

/// `(outer, k)` of the quantized operand of a `[r, c]` tensor.
fn operand_dims(r: usize, c: usize, orient: MxOrient) -> (usize, usize) {
    match orient {
        MxOrient::Rows => (r, c),
        MxOrient::ColsT => (c, r),
    }
}

/// A transient operand: the contiguous `[r, c]` tensor `t` quantized into scratch.
fn activation(t: &NslTensor, r: usize, c: usize, orient: MxOrient) -> MxOp {
    let (outer, k) = operand_dims(r, c, orient);
    let op = scratch(r * c, scale_bytes(outer, k));
    quantize(t, r, c, orient, &op);
    op
}

/// The weight `w` (contiguous `[r, c]` storage) along `orient`: the cast
/// cache's image of a registered parameter, quantized when stale; scratch
/// for anything else.
fn weight(w: &NslTensor, r: usize, c: usize, orient: MxOrient) -> MxOp {
    let (outer, k) = operand_dims(r, c, orient);
    let sbytes = scale_bytes(outer, k);
    if w.dtype == DTYPE_F32
        && let Some((data, scales, needs)) =
            bf16_cast_cache::acquire_mx(w.data as u64, r * c, orient, r * c, sbytes)
    {
        let op = MxOp {
            data,
            scales,
            owned: false,
        };
        // `acquire_mx` marked the image valid: quantize at once.
        if needs {
            quantize(w, r, c, orient, &op);
        }
        return op;
    }
    activation(w, r, c, orient)
}

/// The dims of a GEMM this module takes: K and N multiples of 32 (the
/// quantizers' blocks and tiles), everything within cuBLAS's `i32`.
fn dims_ok(m: i64, n: i64, k: i64) -> bool {
    m > 0
        && n > 0
        && k > 0
        && k % 32 == 0
        && n % 32 == 0
        && m <= i64::from(i32::MAX)
        && n <= i64::from(i32::MAX)
        && k <= i64::from(i32::MAX)
}

/// A new contiguous tensor of `shape` and dtype `tag` on `device`.
fn new_tensor(shape: &[i64], tag: u16, device: u8) -> (i64, u64) {
    let total: i64 = shape.iter().product();
    let bytes = total as usize * if tag == DTYPE_BF16 { 2 } else { 4 };
    let data = super::inner::alloc_managed(bytes);
    let sp = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
    for (i, &d) in shape.iter().enumerate() {
        // SAFETY: `sp` holds `shape.len()` i64 slots.
        unsafe { *sp.add(i) = d };
    }
    let strides = NslTensor::compute_strides(sp, shape.len() as i64);
    let t = Box::new(NslTensor::new(
        data,
        sp,
        strides,
        shape.len() as i64,
        total,
        device,
        tag,
        1,
        0,
    ));
    (Box::into_raw(t) as i64, data as u64)
}

/// `a @ b` (plus the f32 accumulator `acc` with beta = 1), the result stored
/// `out_tag`, as one MXFP8 GEMM; `None` (nothing written) outside the
/// contract. `a` contiguous `[.., k]`; `b` a contiguous `[k, n]` weight or
/// the transposed view of a contiguous `[n, k]` one.
fn product(a_ptr: i64, b_ptr: i64, out_tag: u16, acc: Option<i64>) -> Option<i64> {
    if super::graph_capture::enabled() || !super::mx_quant::supported() {
        return decline(None);
    }
    let (a, b) = (
        NslTensor::from_ptr_ref(a_ptr),
        NslTensor::from_ptr_ref(b_ptr),
    );
    if a.device == 0
        || a.device != b.device
        || a.ndim < 2
        || b.ndim != 2
        || a.len == 0
        || b.len == 0
        || !a.is_contiguous()
        || !storage_dtype(a)
        || !storage_dtype(b)
    {
        return decline(None);
    }
    let a_shape = unsafe { std::slice::from_raw_parts(a.shape, a.ndim as usize) };
    let (b_rows, b_cols) = unsafe { (*b.shape, *b.shape.add(1)) };
    let k = a_shape[a.ndim as usize - 1];
    if k != b_rows {
        return decline(None);
    }
    let transb = if b.is_contiguous() {
        false
    } else if super::storage_gemm::is_transposed_2d(b) {
        true
    } else {
        return decline(None);
    };
    let m = a.len / k;
    let n = b_cols;
    if !dims_ok(m, n, k) {
        return decline(None);
    }
    let mut out_shape: Vec<i64> = a_shape.to_vec();
    *out_shape.last_mut().expect("ndim >= 2") = n;
    let acc = match acc {
        None => None,
        Some(c_ptr) => {
            let c = NslTensor::from_ptr_ref(c_ptr);
            let c_shape = unsafe { std::slice::from_raw_parts(c.shape, c.ndim as usize) };
            if c.device != a.device
                || c.dtype != DTYPE_F32
                || !c.is_contiguous()
                || c_shape != out_shape.as_slice()
            {
                return decline(None);
            }
            Some(c.data as u64)
        }
    };
    let (m, n, k) = (m as usize, n as usize, k as usize);
    super::inner::set_oom_context("matmul_mx");
    // The transposed view's storage is the contiguous [n, k] weight.
    let w = if transb {
        weight(b, n, k, MxOrient::Rows)
    } else {
        weight(b, k, n, MxOrient::ColsT)
    };
    let x = activation(a, m, k, MxOrient::Rows);
    let (out_ptr, out_data) = new_tensor(&out_shape, out_tag, a.device);
    let (beta, c) = match acc {
        Some(c) => (1.0, c),
        None => (0.0, out_data),
    };
    // SAFETY: `w` holds n rows of k, `x` m rows of k, with their scales; the
    // output and the accumulator hold m * n values in the stated dtypes.
    let ok = unsafe {
        super::lt_matmul::matmul_mxfp8(
            n as i32,
            m as i32,
            k as i32,
            1.0,
            w.data as *const c_void,
            w.scales as *const c_void,
            x.data as *const c_void,
            x.scales as *const c_void,
            beta,
            c as *const c_void,
            out_data as *mut c_void,
            n as i32,
            out_tag == DTYPE_BF16,
        )
    };
    drop(w);
    drop(x);
    if !ok {
        crate::tensor::nsl_tensor_free(out_ptr);
        return decline(None);
    }
    super::inner::sync_after_kernel();
    let kind = match (transb, acc.is_some()) {
        (false, _) => &FWD,
        (true, false) => &DGRAD,
        (true, true) => &DGRAD_ACC,
    };
    kind.fetch_add(1, Ordering::Relaxed);
    Some(out_ptr)
}

/// `a @ b` stored `out_tag` as an MXFP8 GEMM (`nsl_tensor_matmul_mx`).
pub(crate) fn matmul(a_ptr: i64, b_ptr: i64, out_tag: u16) -> Option<i64> {
    product(a_ptr, b_ptr, out_tag, None)
}

/// `c + a @ b` into a new f32 tensor shaped like `c`, as one MXFP8 GEMM with
/// `c` added in its epilogue (`nsl_tensor_matmul_acc` route 4).
pub(crate) fn matmul_acc(a_ptr: i64, b_ptr: i64, c_ptr: i64) -> Option<i64> {
    product(a_ptr, b_ptr, DTYPE_F32, Some(c_ptr))
}

/// `m += scale · x^T g` for f32 `m [d_in, d_out]`, `x [N, d_in]` and
/// `g [N, d_out]` contiguous (f32 or bf16) as one MXFP8 GEMM; `false`
/// (nothing written, the accumulator's fresh mark untouched) outside the
/// contract.
pub(crate) fn wgrad_accum(
    m_ptr: i64,
    x_ptr: i64,
    g_ptr: i64,
    n_rows: u64,
    d_in: u64,
    d_out: u64,
    scale: f32,
) -> bool {
    if super::graph_capture::enabled() || !super::mx_quant::supported() {
        return decline(false);
    }
    let (m, x, g) = (
        NslTensor::from_ptr_ref(m_ptr),
        NslTensor::from_ptr_ref(x_ptr),
        NslTensor::from_ptr_ref(g_ptr),
    );
    if m.dtype != DTYPE_F32 || !storage_dtype(x) || !storage_dtype(g) {
        return decline(false);
    }
    let (n, di, dout) = (n_rows as i64, d_in as i64, d_out as i64);
    // Both operands are quantized down their columns (K = tokens) by the
    // transposing quantizer, whose tiles need the tokens and both widths to be
    // multiples of 32: `dims_ok` checks K (tokens) and N (d_in), not M.
    if !dims_ok(dout, di, n) || dout % 32 != 0 || n / 32 > 65_535 {
        return decline(false);
    }
    let (n, di, dout) = (n as usize, di as usize, dout as usize);
    super::inner::set_oom_context("wgrad_accum_mx");
    let gt = activation(g, n, dout, MxOrient::ColsT);
    let xt = activation(x, n, di, MxOrient::ColsT);
    // A fresh accumulator is zero in value: overwrite it (`crate::accum_fresh`).
    let claimed = crate::accum_fresh::claim(m_ptr);
    let beta = if claimed { 0.0 } else { 1.0 };
    // Column-major `m^T [d_out, d_in] = Gt^T Xt` is row-major `m [d_in, d_out]`.
    // SAFETY: gt holds d_out rows of N, xt d_in rows of N, with their scales;
    // `m` holds d_in * d_out f32.
    let ok = unsafe {
        super::lt_matmul::matmul_mxfp8(
            dout as i32,
            di as i32,
            n as i32,
            scale,
            gt.data as *const c_void,
            gt.scales as *const c_void,
            xt.data as *const c_void,
            xt.scales as *const c_void,
            beta,
            m.data as *const c_void,
            m.data,
            dout as i32,
            false,
        )
    };
    drop(gt);
    drop(xt);
    if !ok {
        // The claim took the fresh mark: the caller's fallback adds, so the
        // buffer must hold the zeros it stands for.
        if claimed {
            super::inner::memset_d8(m.data, di * dout * 4);
        }
        return decline(false);
    }
    super::inner::sync_after_kernel();
    WGRAD.fetch_add(1, Ordering::Relaxed);
    true
}
