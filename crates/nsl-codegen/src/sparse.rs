//! M50: Sparse tensor compilation — format-aware kernel dispatch.

/// Sparse operation type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SparseOp {
    MatMul,           // SpMM: sparse @ dense
    MatVec,           // SpMV: sparse @ vector
    StructuredMatMul, // 2:4 mma.sp
    Add,              // sparse + sparse
    ScalarMul,        // sparse * scalar
}

/// Kernel selection for sparse operations.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelSelection {
    PtxCsrSpmm,
    PtxCooSpmm,
    PtxCscSpmm,
    PtxBsrSpmm,
    PtxCsrSpmv,
    PtxCooSpmv,
    PtxMmaSp,           // Ampere 2:4 structured sparsity
    CpuCsrSpmm,
    CpuGenericSpmm,
    CpuGenericSpmv,
    SparseAdd(u8),       // format ID
    SparseScalarMul(u8), // format ID
    Fallback,
}

/// Select optimal kernel based on operation, format, and device.
/// format: SparseFmtId as u8 (0=Coo, 1=Csr, 2=Csc, 3=Bsr)
/// device: 0=CPU, 1+=CUDA
pub fn select_sparse_kernel(op: SparseOp, format: u8, device: u8) -> KernelSelection {
    match (op, format, device > 0) {
        // GPU SpMM
        (SparseOp::MatMul, 1, true) => KernelSelection::PtxCsrSpmm,
        (SparseOp::MatMul, 0, true) => KernelSelection::PtxCooSpmm,
        (SparseOp::MatMul, 2, true) => KernelSelection::PtxCscSpmm,
        (SparseOp::MatMul, 3, true) => KernelSelection::PtxBsrSpmm,
        // GPU SpMV
        (SparseOp::MatVec, 1, true) => KernelSelection::PtxCsrSpmv,
        (SparseOp::MatVec, 0, true) => KernelSelection::PtxCooSpmv,
        // CPU
        (SparseOp::MatMul, 1, false) => KernelSelection::CpuCsrSpmm,
        (SparseOp::MatMul, _, false) => KernelSelection::CpuGenericSpmm,
        (SparseOp::MatVec, _, false) => KernelSelection::CpuGenericSpmv,
        // Structured sparsity
        (SparseOp::StructuredMatMul, _, true) => KernelSelection::PtxMmaSp,
        // Element-wise
        (SparseOp::Add, fmt, _) => KernelSelection::SparseAdd(fmt),
        (SparseOp::ScalarMul, fmt, _) => KernelSelection::SparseScalarMul(fmt),
        _ => KernelSelection::Fallback,
    }
}

/// Sparsity-preserving output type rules.
#[derive(Debug, Clone, PartialEq)]
pub enum SparseOutputType {
    Sparse(u8),   // result is sparse with given format ID
    Dense,        // result is dense
}

/// Determine output sparsity for an operation.
pub fn infer_sparse_output(op: &str, lhs_sparse: bool, rhs_sparse: bool, lhs_fmt: u8) -> SparseOutputType {
    match (op, lhs_sparse, rhs_sparse) {
        ("add" | "sub", true, true) => SparseOutputType::Sparse(lhs_fmt),
        ("mul_scalar" | "div_scalar", true, false) => SparseOutputType::Sparse(lhs_fmt),
        ("matmul", true, false) | ("matmul", false, true) => SparseOutputType::Dense,
        ("mul", true, false) | ("mul", false, true) => SparseOutputType::Dense, // elementwise
        _ => SparseOutputType::Dense,
    }
}

// ---------------------------------------------------------------------------
// Sparse cost model — density-aware format selection
// ---------------------------------------------------------------------------

use crate::gpu_specs::GpuSpec;

/// Sparse storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    /// Coordinate list: (row, col, val) triples. Best for very sparse (density < 1%).
    Coo,
    /// Compressed Sparse Row. General-purpose default for sparse matrices.
    Csr,
    /// Compressed Sparse Column.
    Csc,
    /// Block Sparse Row. Best when nonzeros cluster in dense blocks.
    Bsr,
    /// Dense representation fallback.
    Dense,
}

/// Cost estimate for a sparse operation in a given format.
#[derive(Debug, Clone)]
pub struct SparseCostEstimate {
    pub format: SparseFormat,
    pub flops: f64,
    pub memory_bytes: usize,
    pub estimated_time_us: f64,
}

/// Estimates sparse operation costs for different formats on a given GPU.
pub struct SparseCostModel<'a> {
    pub gpu: &'a GpuSpec,
}

impl<'a> SparseCostModel<'a> {
    pub fn new(gpu: &'a GpuSpec) -> Self {
        Self { gpu }
    }

    /// Estimate cost of a sparse matrix-vector/matrix multiply in the given format.
    pub fn estimate(&self, format: SparseFormat, m: usize, n: usize, nnz: usize) -> SparseCostEstimate {
        let (flops, memory_bytes) = match format {
            SparseFormat::Coo => {
                // COO SpMV: 2 * nnz FLOPs (mul + add per nonzero)
                // Memory: nnz * 12 bytes (row_idx:4 + col_idx:4 + value:4)
                (2.0 * nnz as f64, nnz * 12)
            }
            SparseFormat::Csr => {
                // CSR SpMV: 2 * nnz FLOPs
                // Memory: (m+1)*4 + nnz*4 + nnz*4 (row_ptr + col_idx + values)
                (2.0 * nnz as f64, (m + 1) * 4 + nnz * 8)
            }
            SparseFormat::Csc => {
                // CSC: same flops as CSR, memory indexed by columns
                (2.0 * nnz as f64, (n + 1) * 4 + nnz * 8)
            }
            SparseFormat::Bsr => {
                // BSR with 16x16 blocks
                let block_size = 16_usize;
                let block_area = block_size * block_size;
                let num_blocks = (nnz / block_area).max(1);
                let num_block_rows = m / block_size + 1;
                let flops = 2.0 * num_blocks as f64 * block_area as f64;
                let memory = num_block_rows * 4 + num_blocks * 4 + num_blocks * block_area * 4;
                (flops, memory)
            }
            SparseFormat::Dense => {
                // Dense: 2*M*N FLOPs, M*N*4 bytes
                (2.0 * m as f64 * n as f64, m * n * 4)
            }
        };

        let bandwidth_bytes_per_us = self.gpu.peak_bandwidth_gbs * 1e3;
        let peak_flops_per_us = self.gpu.peak_fp32_tflops * 1e6;

        let compute_time = if peak_flops_per_us > 0.0 { flops / peak_flops_per_us } else { 0.0 };
        let memory_time = if bandwidth_bytes_per_us > 0.0 { memory_bytes as f64 / bandwidth_bytes_per_us } else { 0.0 };

        SparseCostEstimate {
            format,
            flops,
            memory_bytes,
            estimated_time_us: compute_time.max(memory_time),
        }
    }
}

/// Select the optimal sparse format based on matrix dimensions, density, and GPU.
///
/// Decision tree:
/// 1. Small matrices (M*N < 4096): always Dense (sparse overhead not worth it)
/// 2. density > 0.5: Dense fallback
/// 3. density < 0.01: COO (best for very sparse, simplest access)
/// 4. density 0.01-0.5: compare CSR vs BSR via cost model
///    - BSR preferred when density > 0.1 AND dimensions are 16-aligned AND cost is lower
///    - CSR is the general-purpose default
pub fn select_optimal_format(m: usize, n: usize, nnz: usize, gpu: &GpuSpec) -> SparseFormat {
    let total = m * n;

    // Small matrices: always dense
    if total < 4096 {
        return SparseFormat::Dense;
    }

    let density = if total > 0 { nnz as f64 / total as f64 } else { 1.0 };

    // High density: dense fallback
    if density > 0.5 {
        return SparseFormat::Dense;
    }

    // Very sparse: COO
    if density < 0.01 {
        return SparseFormat::Coo;
    }

    // Medium density: CSR vs BSR
    let model = SparseCostModel::new(gpu);
    let csr_cost = model.estimate(SparseFormat::Csr, m, n, nnz);
    let bsr_cost = model.estimate(SparseFormat::Bsr, m, n, nnz);

    let block_aligned = m.is_multiple_of(16) && n.is_multiple_of(16);
    if density > 0.1 && block_aligned && bsr_cost.estimated_time_us < csr_cost.estimated_time_us {
        SparseFormat::Bsr
    } else {
        SparseFormat::Csr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_gpu_spmm() {
        assert_eq!(
            select_sparse_kernel(SparseOp::MatMul, 1, 1),
            KernelSelection::PtxCsrSpmm,
        );
    }

    #[test]
    fn coo_gpu_spmm() {
        assert_eq!(
            select_sparse_kernel(SparseOp::MatMul, 0, 1),
            KernelSelection::PtxCooSpmm,
        );
    }

    #[test]
    fn cpu_csr_spmm() {
        assert_eq!(
            select_sparse_kernel(SparseOp::MatMul, 1, 0),
            KernelSelection::CpuCsrSpmm,
        );
    }

    #[test]
    fn structured_matmul() {
        assert_eq!(
            select_sparse_kernel(SparseOp::StructuredMatMul, 0, 1),
            KernelSelection::PtxMmaSp,
        );
    }

    #[test]
    fn sparse_add() {
        assert_eq!(
            select_sparse_kernel(SparseOp::Add, 1, 0),
            KernelSelection::SparseAdd(1),
        );
        assert_eq!(
            select_sparse_kernel(SparseOp::Add, 0, 1),
            KernelSelection::SparseAdd(0),
        );
    }

    #[test]
    fn output_type_inference() {
        // sparse + sparse → sparse
        assert_eq!(
            infer_sparse_output("add", true, true, 1),
            SparseOutputType::Sparse(1),
        );
        // sparse * scalar → sparse
        assert_eq!(
            infer_sparse_output("mul_scalar", true, false, 0),
            SparseOutputType::Sparse(0),
        );
        // sparse @ dense → dense
        assert_eq!(
            infer_sparse_output("matmul", true, false, 1),
            SparseOutputType::Dense,
        );
        // dense @ sparse → dense
        assert_eq!(
            infer_sparse_output("matmul", false, true, 1),
            SparseOutputType::Dense,
        );
        // unknown op → dense
        assert_eq!(
            infer_sparse_output("unknown", true, true, 0),
            SparseOutputType::Dense,
        );
    }

    // ── Sparse cost model tests ───────────────────────────────────────

    #[test]
    fn test_sparse_cost_model_estimates() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        let model = SparseCostModel::new(gpu);

        let est = model.estimate(SparseFormat::Csr, 1024, 1024, 10240);
        assert!(est.flops > 0.0, "CSR FLOPs should be positive");
        assert!(est.memory_bytes > 0, "CSR memory should be positive");
        assert!(est.estimated_time_us > 0.0, "CSR time should be positive");
    }

    #[test]
    fn test_csr_less_memory_than_coo() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        let model = SparseCostModel::new(gpu);
        let m = 10000;
        let n = 10000;
        let nnz = 500_000;

        let coo = model.estimate(SparseFormat::Coo, m, n, nnz);
        let csr = model.estimate(SparseFormat::Csr, m, n, nnz);

        // CSR compresses row indices into row_ptr (m+1 entries vs nnz entries)
        assert!(csr.memory_bytes < coo.memory_bytes,
            "CSR ({} bytes) should use less memory than COO ({} bytes)",
            csr.memory_bytes, coo.memory_bytes);
    }

    #[test]
    fn test_dense_more_memory_than_sparse() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        let model = SparseCostModel::new(gpu);
        let m = 10000;
        let n = 10000;
        let nnz = 100_000; // 0.1% density

        let csr = model.estimate(SparseFormat::Csr, m, n, nnz);
        let dense = model.estimate(SparseFormat::Dense, m, n, nnz);

        assert!(dense.memory_bytes > csr.memory_bytes,
            "Dense ({} bytes) should use more memory than CSR ({} bytes) at 0.1% density",
            dense.memory_bytes, csr.memory_bytes);
    }

    #[test]
    fn test_select_format_very_sparse() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        // density = 500 / (10000*10000) = 0.000005 → COO
        assert_eq!(select_optimal_format(10000, 10000, 500, gpu), SparseFormat::Coo);
    }

    #[test]
    fn test_select_format_moderate_sparse() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        // density = 5M / 100M = 0.05 → CSR (medium density, general purpose)
        assert_eq!(select_optimal_format(10000, 10000, 5_000_000, gpu), SparseFormat::Csr);
    }

    #[test]
    fn test_select_format_dense_fallback() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        // density = 60M / 100M = 0.6 → Dense
        assert_eq!(select_optimal_format(10000, 10000, 60_000_000, gpu), SparseFormat::Dense);
    }

    #[test]
    fn test_select_format_small_matrix_always_dense() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        // 32x32 = 1024 < 4096 → Dense
        assert_eq!(select_optimal_format(32, 32, 100, gpu), SparseFormat::Dense);
    }

    #[test]
    fn test_select_format_single_element() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        assert_eq!(select_optimal_format(1, 1, 1, gpu), SparseFormat::Dense);
    }

    #[test]
    fn test_select_format_empty_matrix() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        // 0 nonzeros, density = 0 → COO (below 1% threshold)
        assert_eq!(select_optimal_format(1000, 1000, 0, gpu), SparseFormat::Coo);
    }

    #[test]
    fn test_select_format_fully_dense() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        assert_eq!(select_optimal_format(1000, 1000, 1_000_000, gpu), SparseFormat::Dense);
    }
}
