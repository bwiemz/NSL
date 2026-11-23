//! GEMMs for bf16 activation storage: bf16 operands, f32 accumulation, the
//! output stored bf16 (an activation: the epilogue rounds the f32 accumulator
//! once) or f32 (a weight gradient accumulated with beta = 1, or a gradient
//! flowing into an f32 value).
//!
//! An operand already stored bf16 is read as it is: no cast. An f32 operand is
//! rounded to bf16 once (round to nearest even), from the weight-cast cache
//! when it is a registered parameter (or a sub-view of one, the LM head's
//! vocabulary chunks) and into scratch otherwise, as the BF16 math mode's
//! `prepare_bf16_operands` does.
//!
//! Issued through cublasLt (`lt_matmul::matmul_bf16`, timed per shape and
//! output type) with `cublasGemmEx` as the fallback, which computes the same
//! product. Not cuda-graph aware: `precision="bf16"` refuses `--cuda-graphs`
//! at compile time, and these entry points refuse a capture at run time.

use std::ffi::c_void;

use cudarc::cublas::sys as cublas_sys;

use crate::tensor::{NslTensor, DTYPE_BF16, DTYPE_F32};

/// A bf16 operand image: borrowed (the tensor's own bf16 data, or a
/// cast-cache image) or owned scratch, freed on drop. Freeing right after the
/// GEMM is enqueued is safe by the single-stream invariant `Bf16Scratch`
/// documents in `cuda/mod.rs`.
pub(crate) struct Operand16 {
    pub ptr: u64,
    owned: bool,
}

impl Drop for Operand16 {
    fn drop(&mut self) {
        if self.owned {
            super::inner::free_managed(self.ptr as *mut c_void);
        }
    }
}

/// `t`'s first `elems` elements (from its data pointer) as bf16.
pub(crate) fn operand16(t: &NslTensor, elems: usize) -> Operand16 {
    let data = t.data as u64;
    match t.dtype {
        DTYPE_BF16 => Operand16 { ptr: data, owned: false },
        DTYPE_F32 => {
            if let Some((buf, needs_cast)) = super::bf16_cast_cache::acquire(data, elems) {
                if needs_cast {
                    super::gpu_cast_raw_f32_to_bf16(data, buf, elems);
                }
                return Operand16 { ptr: buf, owned: false };
            }
            if let Some(ptr) = super::bf16_cast_cache::acquire_sub(data, elems) {
                return Operand16 { ptr, owned: false };
            }
            let scratch = super::inner::alloc_managed(elems * 2) as u64;
            super::gpu_cast_raw_f32_to_bf16(data, scratch, elems);
            Operand16 { ptr: scratch, owned: true }
        }
        other => panic!("bf16-storage GEMM operand has dtype {other}; only f32 and bf16"),
    }
}

/// Column-major `C = alpha * op(A) op(B) + beta * C` over bf16 `A`, `B`
/// (cuBLAS argument order and meaning), `C` bf16 or f32.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm16(
    transa: bool,
    transb: bool,
    m: u64,
    n: u64,
    k: u64,
    alpha: f32,
    a16: u64,
    lda: u64,
    b16: u64,
    ldb: u64,
    beta: f32,
    c: u64,
    ldc: u64,
    c_bf16: bool,
) {
    // SAFETY: the caller's contract; C doubles as D.
    unsafe {
        gemm16_into(
            transa, transb, m, n, k, alpha, a16, lda, b16, ldb, beta, c, c, ldc, c_bf16,
        )
    }
}

/// [`gemm16`] writing `D = alpha * op(A) op(B) + beta * C` into `d`, which has C's
/// layout (`d == c`: in place). cublasLt takes the two buffers as they are; the GemmEx
/// fallback (in place only) first copies C into D, stream-ordered.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm16_into(
    transa: bool,
    transb: bool,
    m: u64,
    n: u64,
    k: u64,
    alpha: f32,
    a16: u64,
    lda: u64,
    b16: u64,
    ldb: u64,
    beta: f32,
    c_in: u64,
    c: u64,
    ldc: u64,
    c_bf16: bool,
) {
    if super::graph_capture::enabled() {
        crate::fatal::die(
            crate::fatal::Fatal::Unsupported,
            "a bf16-storage GEMM ran under --cuda-graphs; precision=\"bf16\" does not compose with \
             CUDA graph capture",
        );
    }
    super::inner::ensure_context();
    let handle = super::cublas_inner::cublas_handle();
    // SAFETY: the process's live cuBLAS handle and current stream.
    unsafe {
        cublas_sys::cublasSetStream_v2(handle, super::inner::current_stream() as cublas_sys::cudaStream_t);
    }
    let lt_applies = super::lt_matmul::enabled() || super::lt_matmul::timed_lt_applies(m, n, k);
    // SAFETY: device buffers of the sizes the layouts describe (the callers
    // allocate them from the operands' shapes).
    if lt_applies
        && unsafe {
            super::lt_matmul::matmul_bf16_into(
                i32::from(transa),
                i32::from(transb),
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a16 as *const c_void,
                lda as i32,
                b16 as *const c_void,
                ldb as i32,
                &beta,
                c_in as *mut c_void,
                c as *mut c_void,
                ldc as i32,
                c_bf16,
            )
        }
    {
        return;
    }
    if c_in != c {
        // GemmEx accumulates in place: D starts as a copy of C (stream-ordered).
        assert!(!c_bf16, "an out-of-place accumulation writes an f32 D");
        if beta != 0.0 {
            super::inner::memcpy_dtod(
                c as *mut c_void,
                c_in as *const c_void,
                (ldc * n) as usize * 4,
            );
        }
    }
    let op = |t: bool| if t { cublas_sys::cublasOperation_t::CUBLAS_OP_T } else { cublas_sys::cublasOperation_t::CUBLAS_OP_N };
    let bf16 = cublas_sys::cudaDataType_t::CUDA_R_16BF;
    // cublasGemmEx does not round a bf16 C once: measured, a 64x520x1001
    // product with a bf16 C (odd n) put 4,672 of 64,064 outputs one bf16 ulp
    // away from the correctly rounded accumulator, in both directions (a
    // second rounding in the kernel it picks, e.g. split-K partials reduced
    // in C's type). So GemmEx always writes f32 here, and a bf16 C is the RNE
    // cast of that, the one rounding the storage plan promises. This path
    // serves GEMMs below the timed-plan size and those cublasLt declines.
    let scratch = if c_bf16 {
        assert!(beta == 0.0, "a bf16 C is written, never accumulated into");
        Some(super::inner::alloc_managed((ldc * n) as usize * 4) as u64)
    } else {
        None
    };
    let c_out = scratch.unwrap_or(c);
    // SAFETY: as above.
    let st = unsafe {
        cublas_sys::cublasGemmEx(
            handle,
            op(transa),
            op(transb),
            m as i32,
            n as i32,
            k as i32,
            &alpha as *const f32 as *const c_void,
            a16 as *const c_void,
            bf16,
            lda as i32,
            b16 as *const c_void,
            bf16,
            ldb as i32,
            &beta as *const f32 as *const c_void,
            c_out as *mut c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            ldc as i32,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
        )
    };
    if st != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        crate::fatal::die(crate::fatal::Fatal::Unsupported, &format!("bf16-storage GEMM failed: {st:?}"));
    }
    if let Some(s) = scratch {
        super::gpu_cast_raw_f32_to_bf16(s, c, (ldc * n) as usize);
        // Stream-ordered after the cast (the single-stream invariant).
        super::inner::free_managed(s as *mut c_void);
    }
}

/// Is `t` a 2-D view of a contiguous `[cols, rows]` buffer transposed to
/// `[rows, cols]` (what `.transpose(0, 1)` produces)?
fn is_transposed_2d(t: &NslTensor) -> bool {
    t.ndim == 2 && unsafe { *t.strides == 1 && *t.strides.add(1) == *t.shape } && unsafe { *t.shape.add(1) } > 1
}

/// `a @ b` with the result stored as `out_tag`, when the shapes are ones a
/// single 2-D GEMM expresses (as `gpu_matmul_f32`'s 2-D and batch-collapse
/// arms): `b` 2-D, contiguous or a transposed view; `a` contiguous `[.., m,
/// k]` (leading dims collapsed into rows) or a 2-D transposed view. `None`
/// for anything else (the caller widens and takes the f32 matmul).
pub(crate) fn matmul(a_ptr: i64, b_ptr: i64, out_tag: u16) -> Option<i64> {
    let (a, b) = (NslTensor::from_ptr_ref(a_ptr), NslTensor::from_ptr_ref(b_ptr));
    if a.device == 0 || a.device != b.device || a.ndim < 2 || b.ndim != 2 || a.len == 0 || b.len == 0 {
        return None;
    }
    let a_shape = unsafe { std::slice::from_raw_parts(a.shape, a.ndim as usize) };
    let (b_rows, b_cols) = unsafe { (*b.shape, *b.shape.add(1)) };
    let k = a_shape[a.ndim as usize - 1];
    if k != b_rows {
        return None;
    }
    let transb = if b.is_contiguous() { false } else if is_transposed_2d(b) { true } else { return None };
    let transa = if a.is_contiguous() {
        false
    } else if is_transposed_2d(a) {
        true
    } else {
        return None;
    };
    let m = a.len / k; // every row of A, leading dims collapsed
    let n = b_cols;
    if m > i32::MAX as i64 || n > i32::MAX as i64 || k > i32::MAX as i64 {
        return None;
    }
    super::inner::set_oom_context("matmul_dt");
    let out_bytes = if out_tag == DTYPE_BF16 { 2 } else { 4 };
    let total = (m * n) as usize;
    let c = super::inner::alloc_managed(total * out_bytes);
    let mut out_shape: Vec<i64> = a_shape.to_vec();
    *out_shape.last_mut().expect("ndim >= 2") = n;
    let shape = crate::memory::checked_alloc(out_shape.len() * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &d) in out_shape.iter().enumerate() {
        unsafe { *shape.add(i) = d };
    }
    let strides = NslTensor::compute_strides(shape, out_shape.len() as i64);
    let out =
        Box::new(NslTensor::new(c, shape, strides, out_shape.len() as i64, total as i64, a.device, out_tag, 1, 0));
    let out_ptr = Box::into_raw(out) as i64;

    let a16 = operand16(a, a.len as usize);
    let b16 = operand16(b, b.len as usize);
    let (m, n, k) = (m as u64, n as u64, k as u64);
    // Row-major C[m,n] = A[m,k] @ B[k,n] as column-major C^T = B^T A^T: cuBLAS's
    // first operand is B, its second A (as `sgemm_row_major_t`). The leading
    // dimension is the stored row length of each operand.
    // SAFETY: a16/b16 hold a.len/b.len bf16 values, c holds m*n outputs.
    unsafe {
        gemm16(
            transb,
            transa,
            n,
            m,
            k,
            1.0,
            b16.ptr,
            if transb { k } else { n },
            a16.ptr,
            if transa { m } else { k },
            0.0,
            c as u64,
            n,
            out_tag == DTYPE_BF16,
        );
    }
    drop(a16);
    drop(b16);
    super::inner::sync_after_kernel();
    Some(out_ptr)
}

/// `c + a @ b` into a new f32 tensor shaped like `c`, as one GEMM that adds `c` in its
/// epilogue (beta = 1; `source_ad::fuse_dgrad_accumulate`), when the shapes are ones
/// [`matmul`] expresses and its product has `c`'s shape: `c` f32, contiguous, on `a`'s
/// device. The operands are the bf16 images [`matmul`] would read (the same bytes), and
/// the product runs the same plan as [`matmul`]'s for that shape, so the result is
/// `matmul(a, b, f32) + c` with the add's one rounding. `None` otherwise.
pub(crate) fn matmul_acc(a_ptr: i64, b_ptr: i64, c_ptr: i64) -> Option<i64> {
    let (a, b, cc) = (
        NslTensor::from_ptr_ref(a_ptr),
        NslTensor::from_ptr_ref(b_ptr),
        NslTensor::from_ptr_ref(c_ptr),
    );
    if a.device == 0
        || a.device != b.device
        || a.ndim < 2
        || b.ndim != 2
        || a.len == 0
        || b.len == 0
    {
        return None;
    }
    if cc.device != a.device || cc.dtype != DTYPE_F32 || !cc.is_contiguous() {
        return None;
    }
    let a_shape = unsafe { std::slice::from_raw_parts(a.shape, a.ndim as usize) };
    let (b_rows, b_cols) = unsafe { (*b.shape, *b.shape.add(1)) };
    let k = a_shape[a.ndim as usize - 1];
    if k != b_rows {
        return None;
    }
    let transb = if b.is_contiguous() {
        false
    } else if is_transposed_2d(b) {
        true
    } else {
        return None;
    };
    let transa = if a.is_contiguous() {
        false
    } else if is_transposed_2d(a) {
        true
    } else {
        return None;
    };
    let m = a.len / k;
    let n = b_cols;
    if m > i32::MAX as i64 || n > i32::MAX as i64 || k > i32::MAX as i64 {
        return None;
    }
    let mut out_shape: Vec<i64> = a_shape.to_vec();
    *out_shape.last_mut().expect("ndim >= 2") = n;
    let c_shape = unsafe { std::slice::from_raw_parts(cc.shape, cc.ndim as usize) };
    if c_shape != out_shape.as_slice() {
        return None;
    }
    super::inner::set_oom_context("matmul_acc");
    let total = (m * n) as usize;
    let d = super::inner::alloc_managed(total * 4);
    let shape =
        crate::memory::checked_alloc(out_shape.len() * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &dim) in out_shape.iter().enumerate() {
        unsafe { *shape.add(i) = dim };
    }
    let strides = NslTensor::compute_strides(shape, out_shape.len() as i64);
    let out = Box::new(NslTensor::new(
        d,
        shape,
        strides,
        out_shape.len() as i64,
        total as i64,
        a.device,
        DTYPE_F32,
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out) as i64;

    let a16 = operand16(a, a.len as usize);
    let b16 = operand16(b, b.len as usize);
    let (m, n, k) = (m as u64, n as u64, k as u64);
    // As `matmul`: row-major C[m,n] = A[m,k] @ B[k,n] as column-major C^T = B^T A^T.
    // SAFETY: a16/b16 hold a.len/b.len bf16 values; c and d hold m*n f32 values.
    unsafe {
        gemm16_into(
            transb,
            transa,
            n,
            m,
            k,
            1.0,
            b16.ptr,
            if transb { k } else { n },
            a16.ptr,
            if transa { m } else { k },
            1.0,
            cc.data as u64,
            d as u64,
            n,
            false,
        );
    }
    drop(a16);
    drop(b16);
    super::inner::sync_after_kernel();
    Some(out_ptr)
}

/// `m += scale * x^T g` for f32 `m` `[d_in, d_out]`, `x` `[N, d_in]` and `g`
/// `[N, d_out]` contiguous, each of `x` and `g` f32 or bf16: the fused weight
/// gradient (`--fuse-wgrad-accum`) with bf16 activations. Accumulation stays
/// f32 in the GEMM (beta = 1 into the f32 gradient buffer).
pub(crate) fn wgrad_accum(m_ptr: i64, x_ptr: i64, g_ptr: i64, n_rows: u64, d_in: u64, d_out: u64, scale: f32) {
    let (m, x, g) = (NslTensor::from_ptr_ref(m_ptr), NslTensor::from_ptr_ref(x_ptr), NslTensor::from_ptr_ref(g_ptr));
    assert_eq!(m.dtype, DTYPE_F32, "wgrad accumulate: the gradient buffer must be f32");
    let x16 = operand16(x, x.len as usize);
    let g16 = operand16(g, g.len as usize);
    // Column-major C_cm[d_out, d_in] = G_cm[d_out, N] * X_cm^T, as
    // `sgemm_wgrad_accum` issues it.
    // SAFETY: sizes as asserted by the caller (`nsl_tensor_wgrad_accum`).
    unsafe {
        gemm16(false, true, d_out, d_in, n_rows, scale, g16.ptr, d_out, x16.ptr, d_in, 1.0, m.data as u64, d_out, false);
    }
    drop(x16);
    drop(g16);
    super::inner::sync_after_kernel();
}
