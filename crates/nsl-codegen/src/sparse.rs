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
}
