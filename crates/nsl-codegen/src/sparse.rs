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

// ---------------------------------------------------------------------------
// TACO-style merge lattices for sparse co-iteration
// ---------------------------------------------------------------------------

/// Level format in the iteration graph: how to traverse one dimension of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelFormat {
    /// Iterate 0..N (dense dimension).
    Dense,
    /// Iterate via ptr/idx arrays (CSR row_ptr + col_ind).
    Compressed,
    /// Iterate via coordinate array (COO).
    Singleton,
    /// Compressed with duplicate indices allowed (COO batched insertions).
    CompressedNonUnique,
    /// Hash-map backed level. O(1) lookup but non-sequential iteration.
    Hashed,
}

/// One level in the iteration graph.
#[derive(Debug, Clone)]
pub struct IterLevel {
    /// Which dimension of the tensor this level represents.
    pub dimension: usize,
    /// Storage format for this level.
    pub format: LevelFormat,
    /// Which other tensor operands co-iterate at this level.
    pub merge_with: Vec<usize>,
}

/// Iteration graph: determines traversal order based on tensor formats.
#[derive(Debug, Clone)]
pub struct IterationGraph {
    pub levels: Vec<IterLevel>,
}

/// Merge operation: intersection (mul) or union (add).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeOp {
    /// Both tensors must have non-zeros (element-wise multiply).
    Intersection,
    /// Either tensor has non-zeros (element-wise add).
    Union,
}

/// What to compute at a lattice point.
#[derive(Debug, Clone)]
pub enum MergeExpr {
    /// Both A and B contribute.
    Both,
    /// Only the first operand (A) contributes.
    OnlyA,
    /// Only the second operand (B) contributes.
    OnlyB,
}

/// A point in the merge lattice.
#[derive(Debug, Clone)]
pub struct MergeLatticePoint {
    /// Which operands participate at this point: [a_present, b_present].
    pub iterators: [bool; 2],
    /// What to compute.
    pub expression: MergeExpr,
}

/// Merge lattice: generates the correct while-loop structure for co-iterating
/// sparse dimensions.
#[derive(Debug, Clone)]
pub struct MergeLattice {
    pub points: Vec<MergeLatticePoint>,
    pub merge_op: MergeOp,
}

impl MergeLattice {
    /// Build lattice for element-wise multiplication (intersection).
    pub fn intersection() -> Self {
        MergeLattice {
            points: vec![
                MergeLatticePoint {
                    iterators: [true, true],
                    expression: MergeExpr::Both,
                },
            ],
            merge_op: MergeOp::Intersection,
        }
    }

    /// Build lattice for element-wise addition (union).
    pub fn union() -> Self {
        MergeLattice {
            points: vec![
                MergeLatticePoint {
                    iterators: [true, true],
                    expression: MergeExpr::Both,
                },
                MergeLatticePoint {
                    iterators: [true, false],
                    expression: MergeExpr::OnlyA,
                },
                MergeLatticePoint {
                    iterators: [false, true],
                    expression: MergeExpr::OnlyB,
                },
            ],
            merge_op: MergeOp::Union,
        }
    }
}

impl IterationGraph {
    /// Build iteration graph for SpMV: CSR sparse [M,K] × dense vector [K].
    pub fn spmv_csr() -> Self {
        IterationGraph {
            levels: vec![
                IterLevel {
                    dimension: 0,
                    format: LevelFormat::Compressed,
                    merge_with: vec![], // outer loop: iterate sparse rows
                },
                IterLevel {
                    dimension: 1,
                    format: LevelFormat::Dense, // inner: dense vector access
                    merge_with: vec![1], // co-iterate with vector
                },
            ],
        }
    }

    /// Build iteration graph for SpMM: CSR sparse [M,K] × dense [K,N].
    pub fn spmm_csr() -> Self {
        IterationGraph {
            levels: vec![
                IterLevel {
                    dimension: 0,
                    format: LevelFormat::Compressed,
                    merge_with: vec![],
                },
                IterLevel {
                    dimension: 1,
                    format: LevelFormat::Compressed,
                    merge_with: vec![1], // merge with dense K dimension
                },
            ],
        }
    }

    /// Build iteration graph for sparse-sparse element-wise op.
    pub fn elementwise_csr() -> Self {
        IterationGraph {
            levels: vec![
                IterLevel {
                    dimension: 0,
                    format: LevelFormat::Compressed,
                    merge_with: vec![1], // both tensors at row level
                },
                IterLevel {
                    dimension: 1,
                    format: LevelFormat::Compressed,
                    merge_with: vec![1], // both tensors at column level
                },
            ],
        }
    }
}

/// Sparse FLOP estimation for the cost model.
pub fn estimate_sparse_flops(op: SparseOp, nnz: usize, n: usize, nnz_b: usize) -> f64 {
    match op {
        SparseOp::MatVec => 2.0 * nnz as f64,           // one mul + one add per nz
        SparseOp::MatMul => 2.0 * nnz as f64 * n as f64, // SpMM: per-column
        SparseOp::Add => (nnz + nnz_b) as f64,            // merge cost
        SparseOp::ScalarMul => nnz as f64,                // one mul per nz
        SparseOp::StructuredMatMul => nnz as f64 * n as f64,
    }
}

// ---------------------------------------------------------------------------
// Format-agnostic @layout lowering (Task 2: concrete index notation)
// ---------------------------------------------------------------------------

/// An index variable in the iteration graph (e.g., i, j, k for C[i,j] = A[i,k] * B[k,j]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexVar(pub String);

/// Access pattern for a tensor in a tensor expression.
#[derive(Debug, Clone)]
pub struct TensorAccess {
    /// Index variables used to access this tensor, in dimension order.
    pub index_vars: Vec<IndexVar>,
    /// Per-dimension level formats from @layout annotation.
    pub levels: Vec<LevelFormat>,
}

/// A tensor expression to be lowered to sparse iteration code.
#[derive(Debug, Clone)]
pub enum TensorExpr {
    /// C = A @ B (matmul)
    Matmul {
        output: TensorAccess,
        left: TensorAccess,
        right: TensorAccess,
    },
    /// C = A + B (element-wise add)
    Add {
        output: TensorAccess,
        left: TensorAccess,
        right: TensorAccess,
    },
    /// C = A * B (element-wise mul)
    Mul {
        output: TensorAccess,
        left: TensorAccess,
        right: TensorAccess,
    },
}

/// Build an iteration graph from a tensor expression using TACO-style lowering.
///
/// The graph determines loop nesting order and merge strategy for each level.
/// This is the core of format-agnostic sparsity: users write `C = A @ B`,
/// the compiler inspects A's and B's @layout annotations, and generates
/// the optimal iteration structure.
pub fn build_iteration_graph_from_expr(expr: &TensorExpr) -> IterationGraph {
    match expr {
        TensorExpr::Matmul { left, right, .. } => {
            // C[i,j] = A[i,k] * B[k,j]
            // Outer loop: iterate A's row dimension (level 0)
            // Inner loop: iterate A's column dimension (level 1), merge with B's row dimension
            let row_fmt = left.levels.first().copied().unwrap_or(LevelFormat::Dense);
            let col_fmt = left.levels.get(1).copied().unwrap_or(LevelFormat::Compressed);

            IterationGraph {
                levels: vec![
                    IterLevel {
                        dimension: 0,
                        format: row_fmt,
                        merge_with: vec![],
                    },
                    IterLevel {
                        dimension: 1,
                        format: col_fmt,
                        merge_with: vec![1], // merge with right operand's row dimension
                    },
                ],
            }
        }
        TensorExpr::Add { left, right, .. } => {
            // C[i,j] = A[i,j] + B[i,j] — union merge at both levels
            let levels: Vec<IterLevel> = left
                .levels
                .iter()
                .enumerate()
                .map(|(dim, &fmt)| {
                    let right_fmt = right.levels.get(dim).copied().unwrap_or(LevelFormat::Dense);
                    // For union merge, use the more compressed format
                    let merged_fmt = if fmt == LevelFormat::Compressed || right_fmt == LevelFormat::Compressed {
                        LevelFormat::Compressed
                    } else {
                        fmt
                    };
                    IterLevel {
                        dimension: dim,
                        format: merged_fmt,
                        merge_with: vec![1],
                    }
                })
                .collect();
            IterationGraph { levels }
        }
        TensorExpr::Mul { left, right, .. } => {
            // C[i,j] = A[i,j] * B[i,j] — intersection merge at both levels
            let levels: Vec<IterLevel> = left
                .levels
                .iter()
                .enumerate()
                .map(|(dim, &fmt)| {
                    let right_fmt = right.levels.get(dim).copied().unwrap_or(LevelFormat::Dense);
                    // For intersection, use compressed if either is compressed
                    let merged_fmt = if fmt == LevelFormat::Compressed || right_fmt == LevelFormat::Compressed {
                        LevelFormat::Compressed
                    } else {
                        fmt
                    };
                    IterLevel {
                        dimension: dim,
                        format: merged_fmt,
                        merge_with: vec![1],
                    }
                })
                .collect();
            IterationGraph { levels }
        }
    }
}

/// Determine the merge lattice for a tensor expression.
pub fn build_merge_lattice_from_expr(expr: &TensorExpr) -> MergeLattice {
    match expr {
        TensorExpr::Matmul { .. } | TensorExpr::Mul { .. } => MergeLattice::intersection(),
        TensorExpr::Add { .. } => MergeLattice::union(),
    }
}

// ---------------------------------------------------------------------------
// Task 3: Automatic format selection from @layout
// ---------------------------------------------------------------------------

/// Select the optimal sparse kernel for a tensor expression based on @layout annotations.
///
/// This replaces manual kernel selection — users write standard math (`C = A @ B`)
/// and the compiler selects the best kernel from the annotated formats.
pub fn select_kernel_from_layout(
    expr: &TensorExpr,
    device: u8,
) -> KernelSelection {
    match expr {
        TensorExpr::Matmul { left, .. } => {
            // Determine format ID from left operand's layout
            let format_id = layout_to_format_id(&left.levels);
            let op = if left.levels.len() <= 1 || left.levels.iter().all(|l| *l == LevelFormat::Dense) {
                // Dense — shouldn't use sparse kernel
                return KernelSelection::Fallback;
            } else {
                SparseOp::MatMul
            };
            select_sparse_kernel(op, format_id, device)
        }
        TensorExpr::Add { left, .. } => {
            let format_id = layout_to_format_id(&left.levels);
            select_sparse_kernel(SparseOp::Add, format_id, device)
        }
        TensorExpr::Mul { left, .. } => {
            let format_id = layout_to_format_id(&left.levels);
            select_sparse_kernel(SparseOp::ScalarMul, format_id, device)
        }
    }
}

/// Map a level format list to the M50 format ID (0=COO, 1=CSR, 2=CSC, 3=BSR).
fn layout_to_format_id(levels: &[LevelFormat]) -> u8 {
    match levels {
        [LevelFormat::Dense, LevelFormat::Compressed] => 1,      // CSR
        [LevelFormat::Compressed, LevelFormat::Dense] => 2,      // CSC
        [LevelFormat::CompressedNonUnique, LevelFormat::Singleton] => 0, // COO
        [LevelFormat::Dense, LevelFormat::Compressed, LevelFormat::Dense, LevelFormat::Dense] => 3, // BSR
        _ => 1, // default to CSR for unknown layouts
    }
}

// ---------------------------------------------------------------------------
// Task 4: Auto-insertion of format conversions
// ---------------------------------------------------------------------------

/// Determine if a format conversion is needed for an operation.
///
/// Returns `Some((from_format_id, to_format_id))` if a conversion is needed,
/// or `None` if the tensor is already in the preferred format.
pub fn needs_format_conversion(
    current_levels: &[LevelFormat],
    expr: &TensorExpr,
) -> Option<(u8, u8)> {
    let current_id = layout_to_format_id(current_levels);

    let preferred_id = match expr {
        TensorExpr::Matmul { .. } => 1,  // CSR preferred for SpMM
        TensorExpr::Add { .. } => current_id, // keep current for add (sorted merge)
        TensorExpr::Mul { .. } => current_id, // keep current for mul
    };

    if current_id != preferred_id {
        Some((current_id, preferred_id))
    } else {
        None
    }
}

/// Select the conversion function name for a format pair.
pub fn conversion_function(from: u8, to: u8) -> &'static str {
    match (from, to) {
        (0, 1) => "nsl_sparse_coo_to_csr",
        (0, 2) => "nsl_sparse_coo_to_csc",
        (1, 0) => "nsl_sparse_csr_to_coo",
        (1, 2) => "nsl_sparse_csr_to_csc",
        (2, 0) => "nsl_sparse_csc_to_coo",
        (2, 1) => "nsl_sparse_csc_to_csr",
        _ => "nsl_sparse_convert",
    }
}

/// Workspace allocation hint for sparse output assembly.
///
/// The output of a sparse operation is assembled in a dense workspace first,
/// then compressed to the output format. This returns the workspace size in elements.
pub fn workspace_size_hint(output_levels: &[LevelFormat], dim_sizes: &[usize]) -> usize {
    // For CSR output: workspace is one dense row (size = num_cols)
    // For COO output: workspace is the full dense matrix (worst case)
    if output_levels.is_empty() || dim_sizes.is_empty() {
        return 0;
    }

    // CSR/CSC: workspace = last dimension size (one row/column at a time)
    if output_levels.len() >= 2 {
        match output_levels.last() {
            Some(LevelFormat::Compressed) => *dim_sizes.last().unwrap_or(&0),
            _ => dim_sizes.iter().product(),
        }
    } else {
        dim_sizes.iter().product()
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

    // ── Merge lattice tests ──────────────────────────────────────

    #[test]
    fn intersection_lattice_one_point() {
        let lat = MergeLattice::intersection();
        assert_eq!(lat.merge_op, MergeOp::Intersection);
        assert_eq!(lat.points.len(), 1);
        assert_eq!(lat.points[0].iterators, [true, true]);
    }

    #[test]
    fn union_lattice_three_points() {
        let lat = MergeLattice::union();
        assert_eq!(lat.merge_op, MergeOp::Union);
        assert_eq!(lat.points.len(), 3);
        // Both, A-only, B-only
        assert_eq!(lat.points[0].iterators, [true, true]);
        assert_eq!(lat.points[1].iterators, [true, false]);
        assert_eq!(lat.points[2].iterators, [false, true]);
    }

    #[test]
    fn spmv_iteration_graph() {
        let ig = IterationGraph::spmv_csr();
        assert_eq!(ig.levels.len(), 2);
        assert_eq!(ig.levels[0].format, LevelFormat::Compressed);
        assert_eq!(ig.levels[1].format, LevelFormat::Dense);
    }

    #[test]
    fn sparse_flop_estimation() {
        assert_eq!(estimate_sparse_flops(SparseOp::MatVec, 1000, 1, 0), 2000.0);
        assert_eq!(estimate_sparse_flops(SparseOp::MatMul, 1000, 64, 0), 128000.0);
        assert_eq!(estimate_sparse_flops(SparseOp::Add, 500, 0, 300), 800.0);
    }

    // ── Format-agnostic @layout tests ────────────────────────────────

    fn csr_access(vars: &[&str]) -> TensorAccess {
        TensorAccess {
            index_vars: vars.iter().map(|v| IndexVar(v.to_string())).collect(),
            levels: vec![LevelFormat::Dense, LevelFormat::Compressed],
        }
    }

    fn csc_access(vars: &[&str]) -> TensorAccess {
        TensorAccess {
            index_vars: vars.iter().map(|v| IndexVar(v.to_string())).collect(),
            levels: vec![LevelFormat::Compressed, LevelFormat::Dense],
        }
    }

    fn dense_access(vars: &[&str]) -> TensorAccess {
        TensorAccess {
            index_vars: vars.iter().map(|v| IndexVar(v.to_string())).collect(),
            levels: vec![LevelFormat::Dense, LevelFormat::Dense],
        }
    }

    #[test]
    fn iteration_graph_csr_matmul() {
        let expr = TensorExpr::Matmul {
            output: dense_access(&["i", "j"]),
            left: csr_access(&["i", "k"]),
            right: dense_access(&["k", "j"]),
        };
        let ig = build_iteration_graph_from_expr(&expr);
        assert_eq!(ig.levels.len(), 2);
        assert_eq!(ig.levels[0].format, LevelFormat::Dense); // CSR row: dense
        assert_eq!(ig.levels[1].format, LevelFormat::Compressed); // CSR col: compressed
    }

    #[test]
    fn iteration_graph_csc_matmul() {
        let expr = TensorExpr::Matmul {
            output: dense_access(&["i", "j"]),
            left: csc_access(&["i", "k"]),
            right: dense_access(&["k", "j"]),
        };
        let ig = build_iteration_graph_from_expr(&expr);
        assert_eq!(ig.levels[0].format, LevelFormat::Compressed); // CSC: compressed rows
        assert_eq!(ig.levels[1].format, LevelFormat::Dense); // CSC: dense cols
    }

    #[test]
    fn merge_lattice_from_add_is_union() {
        let expr = TensorExpr::Add {
            output: csr_access(&["i", "j"]),
            left: csr_access(&["i", "j"]),
            right: csr_access(&["i", "j"]),
        };
        let lattice = build_merge_lattice_from_expr(&expr);
        assert_eq!(lattice.merge_op, MergeOp::Union);
        assert_eq!(lattice.points.len(), 3);
    }

    #[test]
    fn merge_lattice_from_mul_is_intersection() {
        let expr = TensorExpr::Mul {
            output: csr_access(&["i", "j"]),
            left: csr_access(&["i", "j"]),
            right: csr_access(&["i", "j"]),
        };
        let lattice = build_merge_lattice_from_expr(&expr);
        assert_eq!(lattice.merge_op, MergeOp::Intersection);
        assert_eq!(lattice.points.len(), 1);
    }

    #[test]
    fn kernel_selection_from_csr_layout() {
        let expr = TensorExpr::Matmul {
            output: dense_access(&["i", "j"]),
            left: csr_access(&["i", "k"]),
            right: dense_access(&["k", "j"]),
        };
        // GPU
        assert_eq!(select_kernel_from_layout(&expr, 1), KernelSelection::PtxCsrSpmm);
        // CPU
        assert_eq!(select_kernel_from_layout(&expr, 0), KernelSelection::CpuCsrSpmm);
    }

    #[test]
    fn kernel_selection_from_coo_layout() {
        let coo_access = TensorAccess {
            index_vars: vec![IndexVar("i".into()), IndexVar("k".into())],
            levels: vec![LevelFormat::CompressedNonUnique, LevelFormat::Singleton],
        };
        let expr = TensorExpr::Matmul {
            output: dense_access(&["i", "j"]),
            left: coo_access,
            right: dense_access(&["k", "j"]),
        };
        assert_eq!(select_kernel_from_layout(&expr, 1), KernelSelection::PtxCooSpmm);
    }

    #[test]
    fn kernel_selection_dense_falls_back() {
        let expr = TensorExpr::Matmul {
            output: dense_access(&["i", "j"]),
            left: dense_access(&["i", "k"]),
            right: dense_access(&["k", "j"]),
        };
        assert_eq!(select_kernel_from_layout(&expr, 1), KernelSelection::Fallback);
    }

    #[test]
    fn format_conversion_coo_to_csr() {
        let coo_levels = vec![LevelFormat::CompressedNonUnique, LevelFormat::Singleton];
        let expr = TensorExpr::Matmul {
            output: dense_access(&["i", "j"]),
            left: TensorAccess {
                index_vars: vec![IndexVar("i".into()), IndexVar("k".into())],
                levels: coo_levels.clone(),
            },
            right: dense_access(&["k", "j"]),
        };
        let conv = needs_format_conversion(&coo_levels, &expr);
        assert_eq!(conv, Some((0, 1))); // COO → CSR
        assert_eq!(conversion_function(0, 1), "nsl_sparse_coo_to_csr");
    }

    #[test]
    fn no_conversion_needed_for_csr_matmul() {
        let csr_levels = vec![LevelFormat::Dense, LevelFormat::Compressed];
        let expr = TensorExpr::Matmul {
            output: dense_access(&["i", "j"]),
            left: csr_access(&["i", "k"]),
            right: dense_access(&["k", "j"]),
        };
        assert_eq!(needs_format_conversion(&csr_levels, &expr), None);
    }

    #[test]
    fn workspace_size_csr_output() {
        let csr_out = vec![LevelFormat::Dense, LevelFormat::Compressed];
        // For a 1000×500 output in CSR, workspace = 500 (one row)
        assert_eq!(workspace_size_hint(&csr_out, &[1000, 500]), 500);
    }

    #[test]
    fn workspace_size_dense_output() {
        let dense_out = vec![LevelFormat::Dense, LevelFormat::Dense];
        assert_eq!(workspace_size_hint(&dense_out, &[100, 200]), 20000);
    }

    #[test]
    fn workspace_size_empty() {
        assert_eq!(workspace_size_hint(&[], &[]), 0);
    }

    #[test]
    fn layout_to_format_id_mapping() {
        assert_eq!(layout_to_format_id(&[LevelFormat::Dense, LevelFormat::Compressed]), 1); // CSR
        assert_eq!(layout_to_format_id(&[LevelFormat::Compressed, LevelFormat::Dense]), 2); // CSC
        assert_eq!(layout_to_format_id(&[LevelFormat::CompressedNonUnique, LevelFormat::Singleton]), 0); // COO
    }
}
