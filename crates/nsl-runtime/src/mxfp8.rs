//! MXFP8 GEMM building blocks: quantize f32/bf16 device data to OCP E4M3
//! with one UE8M0 scale per 32 elements along K, and multiply two such
//! operands through cublasLt (`CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`).
//!
//! Phase 1 of the MXFP8 training design: the pieces, held to a host
//! reference ([`reference`]) by `tests/mx_quant_gpu.rs`. Nothing in training
//! routes here yet.
//!
//! An operand is K-major: `outer` rows of `k` bytes, its scales in
//! cublasLt's tiled layout ([`scale_offset`], [`scale_bytes`]).
//! [`quant_rows`] makes one from a row-major tensor whose rows are K;
//! [`quant_cols_t`] from one whose columns are K (it writes the transpose).
//! [`gemm`] computes `D[m, n] = alpha · A^T B + beta · C` in column-major
//! terms, `A` with `m` rows and `B` with `n` rows: for row-major outputs,
//! `D^T`, i.e. row `j` of the row-major result is `B`'s row `j` against
//! every row of `A`.

pub use nsl_kir::kernels::mx_quant::{
    MX_BLOCK, reference, scale_bytes, scale_offset, scales_padded, sf_inner,
};

/// Whether the current device runs MXFP8 GEMMs (compute capability 10.0 or
/// newer). The quantizers and [`gemm`] must not be called otherwise.
pub fn supported() -> bool {
    crate::cuda::mx_quant::supported()
}

/// Quantize row-major `[rows, k]` (f32, or bf16 when `src_bf16`), blocks
/// along each row, into `data` (`rows · k` bytes) and `scales`
/// (`scale_bytes(rows, k)` bytes, 16-byte aligned). All device pointers;
/// enqueued on the compute stream.
pub fn quant_rows(src: u64, src_bf16: bool, rows: usize, k: usize, data: u64, scales: u64) {
    crate::cuda::mx_quant::quant_rows(src, src_bf16, rows, k, data, scales);
}

/// Quantize the transpose of row-major `[r, c]`, blocks along each column,
/// into `data` (`[c, r]`) and `scales` (`scale_bytes(c, r)` bytes).
pub fn quant_cols_t(src: u64, src_bf16: bool, r: usize, c: usize, data: u64, scales: u64) {
    crate::cuda::mx_quant::quant_cols_t(src, src_bf16, r, c, data, scales);
}

/// `D = alpha · A^T B + beta · C`, column-major `[m, n]` with leading
/// dimension `ldc`, bf16 or f32 (`c_bf16`), in place in `c`. `a` is `m`
/// rows of `k` E4M3 bytes, `b` is `n` rows. Returns `false` (nothing
/// written) when cublasLt declines.
///
/// # Safety
///
/// Every pointer must be a live device buffer of the stated size, the scale
/// buffers 16-byte aligned, and `c` must not alias an operand.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: u64,
    a_scales: u64,
    b: u64,
    b_scales: u64,
    beta: f32,
    c: u64,
    ldc: usize,
    c_bf16: bool,
) -> bool {
    let dims = [m, n, k, ldc].map(i32::try_from);
    let [Ok(m), Ok(n), Ok(k), Ok(ldc)] = dims else {
        return false;
    };
    // SAFETY: forwarded under the caller's contract.
    unsafe {
        crate::cuda::lt_matmul::matmul_mxfp8(
            m,
            n,
            k,
            alpha,
            a as *const std::ffi::c_void,
            a_scales as *const std::ffi::c_void,
            b as *const std::ffi::c_void,
            b_scales as *const std::ffi::c_void,
            beta,
            c as *mut std::ffi::c_void,
            ldc,
            c_bf16,
        )
    }
}

/// `(issued, tuned shapes, declined)` MXFP8 GEMMs this process.
pub fn stats() -> (u64, u64, u64) {
    crate::cuda::lt_matmul::mx_stats()
}
