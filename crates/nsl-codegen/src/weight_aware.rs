//! M52: Weight-Aware Compilation & Network Constant Folding
//!
//! Loads `.safetensors` weight files at compile time, analyzes sparsity per weight matrix,
//! constant-folds tensor operations with known operands, and eliminates dead weights.
//! Produces bespoke binaries optimized for a specific checkpoint.

use std::collections::HashMap;
use std::path::Path;
use sha2::{Sha256, Digest};

// ─────────────────────────────────────────────────────────────────────────────
// WeightAwareConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for weight-aware compilation passes.
#[derive(Debug, Clone)]
pub struct WeightAwareConfig {
    /// Threshold below which weights are considered "dead" (default: 1e-6)
    pub dead_weight_threshold: f64,
    /// Sparsity fraction above which sparse kernels are emitted (default: 0.5)
    pub sparse_threshold: f64,
    /// Enable constant propagation through matmul/add/relu
    pub constant_fold: bool,
    /// Enable dead weight elimination
    pub dead_weight_elim: bool,
    /// Enable sparsity-aware codegen
    pub sparse_codegen: bool,
}

impl Default for WeightAwareConfig {
    fn default() -> Self {
        Self {
            dead_weight_threshold: 1e-6,
            sparse_threshold: 0.5,
            constant_fold: true,
            dead_weight_elim: true,
            sparse_codegen: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WeightDType
// ─────────────────────────────────────────────────────────────────────────────

/// Supported weight data types in safetensors files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDType {
    F16,
    BF16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
    I8,
    I32,
}

impl WeightDType {
    pub fn byte_width(&self) -> usize {
        match self {
            WeightDType::F8E4M3 | WeightDType::F8E5M2 | WeightDType::I8 => 1,
            WeightDType::F16 | WeightDType::BF16 => 2,
            WeightDType::F32 | WeightDType::I32 => 4,
            WeightDType::F64 => 8,
        }
    }

    /// Convert a single element to f64 for analysis.
    pub fn to_f64(&self, bytes: &[u8]) -> f64 {
        match self {
            WeightDType::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
            WeightDType::F16 => half::f16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f64(),
            WeightDType::BF16 => half::bf16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f64(),
            WeightDType::F64 => f64::from_le_bytes(bytes[..8].try_into().unwrap()),
            WeightDType::I8 => bytes[0] as i8 as f64,
            WeightDType::I32 => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
            WeightDType::F8E4M3 | WeightDType::F8E5M2 => {
                // FP8: interpret as u8 scaled; approximation sufficient for analysis
                (bytes[0] as f64) / 127.0
            }
        }
    }

    pub fn from_safetensors_str(s: &str) -> Result<Self, WeightLoadError> {
        match s {
            "F16" => Ok(WeightDType::F16),
            "BF16" => Ok(WeightDType::BF16),
            "F32" => Ok(WeightDType::F32),
            "F64" => Ok(WeightDType::F64),
            "F8_E4M3" => Ok(WeightDType::F8E4M3),
            "F8_E5M2" => Ok(WeightDType::F8E5M2),
            "I8" => Ok(WeightDType::I8),
            "I32" => Ok(WeightDType::I32),
            other => Err(WeightLoadError::UnsupportedDType(other.to_string())),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WeightLoadError
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum WeightLoadError {
    IoError(String, std::io::Error),
    InvalidFormat(String),
    InvalidHeader(String),
    MissingField(String, &'static str),
    UnsupportedDType(String),
}

impl std::fmt::Display for WeightLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightLoadError::IoError(path, e) => write!(f, "cannot read '{}': {}", path, e),
            WeightLoadError::InvalidFormat(msg) => write!(f, "invalid safetensors format: {}", msg),
            WeightLoadError::InvalidHeader(msg) => write!(f, "invalid safetensors header: {}", msg),
            WeightLoadError::MissingField(name, field) => {
                write!(f, "tensor '{}' missing field '{}'", name, field)
            }
            WeightLoadError::UnsupportedDType(dt) => write!(f, "unsupported dtype: {}", dt),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SparsityInfo + CsrLayout
// ─────────────────────────────────────────────────────────────────────────────

/// Sparsity analysis for a single weight matrix.
#[derive(Debug, Clone)]
pub struct SparsityInfo {
    /// Fraction of elements with |value| < dead_weight_threshold
    pub near_zero_fraction: f64,
    /// Number of near-zero elements
    pub near_zero_count: usize,
    /// Fraction of exact zeros
    pub exact_zero_fraction: f64,
    /// Whether this matrix qualifies for sparse codegen
    pub use_sparse_kernel: bool,
    /// If sparse: the CSR representation for codegen
    pub csr: Option<CsrLayout>,
    /// Per-row sparsity distribution (for block-sparse analysis)
    pub row_sparsity: Vec<f64>,
    /// Max absolute value (for scaling analysis)
    pub max_abs: f64,
    /// Min non-zero absolute value (for dead weight threshold calibration)
    pub min_nonzero_abs: f64,
}

/// Compressed Sparse Row representation of a weight matrix.
#[derive(Debug, Clone)]
pub struct CsrLayout {
    /// Row pointers: row_ptrs[i] is the index in col_indices where row i starts
    pub row_ptrs: Vec<u32>,
    /// Column indices of non-zero elements
    pub col_indices: Vec<u32>,
    /// Values of non-zero elements (in native dtype bytes)
    pub values: Vec<u8>,
    /// Number of rows
    pub nrows: u32,
    /// Number of columns
    pub ncols: u32,
    /// Number of non-zero elements
    pub nnz: u32,
    /// Data type of values
    pub dtype: WeightDType,
}

// ─────────────────────────────────────────────────────────────────────────────
// WeightEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single weight tensor loaded from safetensors.
pub struct WeightEntry {
    /// Parameter name as it appears in the safetensors file
    pub name: String,
    /// Raw weight data in the file's native dtype
    pub data: Vec<u8>,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type of the weight tensor
    pub dtype: WeightDType,
    /// Number of elements
    pub num_elements: usize,
    /// Sparsity analysis results (computed lazily on first access)
    sparsity: Option<SparsityInfo>,
    /// Whether this weight has been eliminated (near-zero)
    pub eliminated: bool,
}

impl WeightEntry {
    /// Compute sparsity statistics for this weight tensor.
    /// Results are cached on first call.
    pub fn analyze_sparsity(&mut self, config: &WeightAwareConfig) -> &SparsityInfo {
        if let Some(ref info) = self.sparsity {
            return info;
        }

        let bw = self.dtype.byte_width();
        let mut near_zero_count: usize = 0;
        let mut exact_zero_count: usize = 0;
        let mut max_abs: f64 = 0.0;
        let mut min_nonzero_abs: f64 = f64::MAX;

        for i in 0..self.num_elements {
            let offset = i * bw;
            let val = self.dtype.to_f64(&self.data[offset..offset + bw]);
            let abs_val = val.abs();

            if abs_val < config.dead_weight_threshold {
                near_zero_count += 1;
            }
            if val == 0.0 {
                exact_zero_count += 1;
            }
            if abs_val > max_abs {
                max_abs = abs_val;
            }
            if abs_val > 0.0 && abs_val < min_nonzero_abs {
                min_nonzero_abs = abs_val;
            }
        }

        let near_zero_fraction = if self.num_elements > 0 {
            near_zero_count as f64 / self.num_elements as f64
        } else {
            0.0
        };
        let exact_zero_fraction = if self.num_elements > 0 {
            exact_zero_count as f64 / self.num_elements as f64
        } else {
            0.0
        };
        let use_sparse = near_zero_fraction >= config.sparse_threshold && self.shape.len() == 2;

        // Per-row sparsity (only for 2D weight matrices)
        let row_sparsity = if self.shape.len() == 2 {
            let nrows = self.shape[0];
            let ncols = self.shape[1];
            (0..nrows)
                .map(|row| {
                    let mut row_zeros = 0usize;
                    for col in 0..ncols {
                        let idx = row * ncols + col;
                        let offset = idx * bw;
                        let val = self.dtype.to_f64(&self.data[offset..offset + bw]);
                        if val.abs() < config.dead_weight_threshold {
                            row_zeros += 1;
                        }
                    }
                    row_zeros as f64 / ncols as f64
                })
                .collect()
        } else {
            Vec::new()
        };

        // Build CSR if sparse
        let csr = if use_sparse && self.shape.len() == 2 {
            Some(self.build_csr(config))
        } else {
            None
        };

        if min_nonzero_abs == f64::MAX {
            min_nonzero_abs = 0.0;
        }

        self.sparsity = Some(SparsityInfo {
            near_zero_fraction,
            near_zero_count,
            exact_zero_fraction,
            use_sparse_kernel: use_sparse,
            csr,
            row_sparsity,
            max_abs,
            min_nonzero_abs,
        });

        self.sparsity.as_ref().unwrap()
    }

    /// Read-only access to cached sparsity info (None if not yet analyzed).
    pub fn sparsity(&self) -> Option<&SparsityInfo> {
        self.sparsity.as_ref()
    }

    /// Build a Compressed Sparse Row representation, dropping near-zero elements.
    fn build_csr(&self, config: &WeightAwareConfig) -> CsrLayout {
        assert_eq!(self.shape.len(), 2, "CSR only for 2D weight matrices");
        let nrows = self.shape[0];
        let ncols = self.shape[1];
        let bw = self.dtype.byte_width();

        let mut row_ptrs: Vec<u32> = Vec::with_capacity(nrows + 1);
        let mut col_indices: Vec<u32> = Vec::new();
        let mut values: Vec<u8> = Vec::new();

        row_ptrs.push(0);

        for row in 0..nrows {
            for col in 0..ncols {
                let idx = row * ncols + col;
                let offset = idx * bw;
                let val = self.dtype.to_f64(&self.data[offset..offset + bw]);
                if val.abs() >= config.dead_weight_threshold {
                    col_indices.push(col as u32);
                    values.extend_from_slice(&self.data[offset..offset + bw]);
                }
            }
            row_ptrs.push(col_indices.len() as u32);
        }

        let nnz = col_indices.len() as u32;

        CsrLayout {
            row_ptrs,
            col_indices,
            values,
            nrows: nrows as u32,
            ncols: ncols as u32,
            nnz,
            dtype: self.dtype,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WeightMap
// ─────────────────────────────────────────────────────────────────────────────

/// Index of all weight tensors loaded from a safetensors file at compile time.
/// Each entry maps a parameter name to its raw tensor data and metadata.
pub struct WeightMap {
    /// Parameter name -> WeightEntry
    entries: HashMap<String, WeightEntry>,
    /// SHA-256 hash of the entire safetensors file
    file_hash: [u8; 32],
    /// Original file path (for error messages)
    source_path: String,
    /// Total bytes of raw tensor data loaded
    total_bytes: u64,
}

impl WeightMap {
    /// Load all tensors from a safetensors file into memory.
    /// Computes the SHA-256 hash of the entire file for integrity verification.
    ///
    /// This reuses the `safetensors` crate's `SafeTensors::deserialize()` for parsing
    /// (same approach as `nsl-runtime/src/safetensors_io.rs`), but retains raw bytes
    /// in native dtype instead of converting to f32 -- preserving dtype fidelity for
    /// sparsity analysis and codegen.
    pub fn load(path: &Path) -> Result<Self, WeightLoadError> {
        let file_bytes = std::fs::read(path)
            .map_err(|e| WeightLoadError::IoError(path.display().to_string(), e))?;

        // Compute SHA-256 of entire file
        let mut hasher = Sha256::new();
        hasher.update(&file_bytes);
        let file_hash: [u8; 32] = hasher.finalize().into();

        // Parse using the safetensors crate (same approach as nsl-runtime)
        let tensors = safetensors::SafeTensors::deserialize(&file_bytes)
            .map_err(|e| WeightLoadError::InvalidFormat(e.to_string()))?;

        let mut entries = HashMap::new();
        let mut total_bytes = 0u64;

        for (name, view) in tensors.tensors() {
            // CRITICAL FIX: match directly on safetensors::Dtype variants
            // instead of format!("{:?}") + from_safetensors_str
            use safetensors::Dtype as ST;
            let dtype = match view.dtype() {
                ST::F16 => WeightDType::F16,
                ST::BF16 => WeightDType::BF16,
                ST::F32 => WeightDType::F32,
                ST::F64 => WeightDType::F64,
                ST::I8 => WeightDType::I8,
                ST::I32 => WeightDType::I32,
                other => return Err(WeightLoadError::UnsupportedDType(format!("{:?}", other))),
            };

            let shape: Vec<usize> = view.shape().to_vec();
            let num_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
            let data = view.data().to_vec();
            total_bytes += data.len() as u64;

            entries.insert(name.clone(), WeightEntry {
                name: name.clone(),
                data,
                shape,
                dtype,
                num_elements,
                sparsity: None,
                eliminated: false,
            });
        }

        Ok(WeightMap {
            entries,
            file_hash,
            source_path: path.display().to_string(),
            total_bytes,
        })
    }

    /// Retrieve a weight tensor by parameter name (immutable).
    pub fn get(&self, name: &str) -> Option<&WeightEntry> {
        self.entries.get(name)
    }

    /// Retrieve a mutable weight entry for annotation.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut WeightEntry> {
        self.entries.get_mut(name)
    }

    /// SHA-256 hash of the original safetensors file.
    pub fn hash(&self) -> &[u8; 32] {
        &self.file_hash
    }

    /// SHA-256 hash as a hex string.
    pub fn hash_hex(&self) -> String {
        self.file_hash.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Total number of parameters across all tensors.
    pub fn total_parameters(&self) -> usize {
        self.entries.values().map(|e| e.num_elements).sum()
    }

    /// Total bytes of raw tensor data.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Source file path (for diagnostics).
    pub fn source_path(&self) -> &str {
        &self.source_path
    }

    /// Iterator over all weight names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    /// Number of weight tensors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Mutable iterator over all entries (for batch analysis).
    pub fn entries_mut(&mut self) -> impl Iterator<Item = (&String, &mut WeightEntry)> {
        self.entries.iter_mut()
    }

    /// Immutable iterator over all entries.
    pub fn entries(&self) -> impl Iterator<Item = (&String, &WeightEntry)> {
        self.entries.iter()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constant Folding
// ─────────────────────────────────────────────────────────────────────────────

/// Result of attempting to constant-fold a tensor expression.
#[derive(Debug)]
pub enum FoldResult {
    /// Expression was fully folded to a known constant tensor
    Constant(ConstTensor),
    /// Expression depends on runtime input -- cannot fold
    Runtime,
    /// Expression was partially folded (some operands known)
    Partial {
        /// Which operands are known constants (0-indexed)
        known_operands: Vec<usize>,
        /// Whether the expression was simplified
        simplified: bool,
    },
}

/// A tensor with fully known values at compile time.
#[derive(Debug, Clone)]
pub struct ConstTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: WeightDType,
}

/// Statistics from the constant folding and dead weight passes.
#[derive(Debug, Default)]
pub struct FoldStats {
    /// Number of expressions evaluated at compile time
    pub folded_ops: usize,
    /// Number of expressions that remain runtime-dependent
    pub runtime_ops: usize,
    /// Number of dead weight eliminations (individual elements)
    pub dead_weights_eliminated: usize,
    /// Number of sparse kernels annotated
    pub sparse_kernels: usize,
    /// Bytes of weight data eliminated
    pub bytes_eliminated: u64,
}

/// The constant propagation pass.
pub struct ConstantFolder<'a> {
    /// Weight map providing known weight values (used by future complex folding passes)
    #[allow(dead_code)]
    weight_map: &'a WeightMap,
    /// Statistics for reporting
    pub stats: FoldStats,
}

impl<'a> ConstantFolder<'a> {
    pub fn new(weight_map: &'a WeightMap) -> Self {
        Self {
            weight_map,
            stats: FoldStats::default(),
        }
    }

    /// Attempt to constant-fold a matmul: C = A @ B.
    /// If both A and B are known constants, compute C at compile time.
    /// If only one operand is known, return Partial.
    pub fn fold_matmul(
        &mut self,
        lhs: &FoldResult,
        rhs: &FoldResult,
    ) -> FoldResult {
        match (lhs, rhs) {
            (FoldResult::Constant(a), FoldResult::Constant(b)) => {
                let c = self.compute_matmul(a, b);
                self.stats.folded_ops += 1;
                FoldResult::Constant(c)
            }
            (FoldResult::Runtime, FoldResult::Constant(_)) => {
                // Input is runtime, weight is known -- annotate for sparsity but cannot fold
                FoldResult::Partial {
                    known_operands: vec![1],
                    simplified: false,
                }
            }
            (FoldResult::Constant(_), FoldResult::Runtime) => {
                FoldResult::Partial {
                    known_operands: vec![0],
                    simplified: false,
                }
            }
            _ => FoldResult::Runtime,
        }
    }

    /// Attempt to constant-fold element-wise addition: C = A + B.
    pub fn fold_add(
        &mut self,
        lhs: &FoldResult,
        rhs: &FoldResult,
    ) -> FoldResult {
        match (lhs, rhs) {
            (FoldResult::Constant(a), FoldResult::Constant(b)) => {
                let c = self.compute_elementwise_add(a, b);
                self.stats.folded_ops += 1;
                FoldResult::Constant(c)
            }
            _ => {
                self.stats.runtime_ops += 1;
                FoldResult::Runtime
            }
        }
    }

    /// Attempt to constant-fold ReLU: Y = relu(X).
    pub fn fold_relu(&mut self, input: &FoldResult) -> FoldResult {
        match input {
            FoldResult::Constant(x) => {
                let y = self.compute_relu(x);
                self.stats.folded_ops += 1;
                FoldResult::Constant(y)
            }
            _ => {
                self.stats.runtime_ops += 1;
                FoldResult::Runtime
            }
        }
    }

    /// Compute matmul of two constant tensors at compile time.
    /// Supports 2D [M, K] x [K, N] -> [M, N].
    /// CRITICAL FIX: use separate byte widths for each tensor.
    fn compute_matmul(&self, a: &ConstTensor, b: &ConstTensor) -> ConstTensor {
        assert_eq!(a.shape.len(), 2, "compile-time matmul requires 2D tensors");
        assert_eq!(b.shape.len(), 2, "compile-time matmul requires 2D tensors");
        let m = a.shape[0];
        let k = a.shape[1];
        assert_eq!(k, b.shape[0], "matmul dimension mismatch: {} vs {}", k, b.shape[0]);
        let n = b.shape[1];

        let bw_a = a.dtype.byte_width();
        let bw_b = b.dtype.byte_width();
        // Output uses dtype of lhs (a)
        let bw_out = a.dtype.byte_width();
        let mut result = vec![0u8; m * n * bw_out];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    let a_val = a.dtype.to_f64(&a.data[(i * k + p) * bw_a..(i * k + p + 1) * bw_a]);
                    let b_val = b.dtype.to_f64(&b.data[(p * n + j) * bw_b..(p * n + j + 1) * bw_b]);
                    sum += a_val * b_val;
                }
                write_f64_as_dtype(sum, a.dtype, &mut result[(i * n + j) * bw_out..(i * n + j + 1) * bw_out]);
            }
        }

        ConstTensor {
            data: result,
            shape: vec![m, n],
            dtype: a.dtype,
        }
    }

    /// Compute element-wise addition of two constant tensors.
    fn compute_elementwise_add(&self, a: &ConstTensor, b: &ConstTensor) -> ConstTensor {
        assert_eq!(a.shape, b.shape, "add shape mismatch");
        let bw = a.dtype.byte_width();
        let num_elements: usize = a.shape.iter().product();
        let mut result = vec![0u8; num_elements * bw];

        for i in 0..num_elements {
            let a_val = a.dtype.to_f64(&a.data[i * bw..(i + 1) * bw]);
            let b_val = b.dtype.to_f64(&b.data[i * bw..(i + 1) * bw]);
            write_f64_as_dtype(a_val + b_val, a.dtype, &mut result[i * bw..(i + 1) * bw]);
        }

        ConstTensor {
            data: result,
            shape: a.shape.clone(),
            dtype: a.dtype,
        }
    }

    /// Compute element-wise ReLU of a constant tensor.
    fn compute_relu(&self, x: &ConstTensor) -> ConstTensor {
        let bw = x.dtype.byte_width();
        let num_elements: usize = x.shape.iter().product();
        let mut result = vec![0u8; num_elements * bw];

        for i in 0..num_elements {
            let val = x.dtype.to_f64(&x.data[i * bw..(i + 1) * bw]);
            let relu_val = if val > 0.0 { val } else { 0.0 };
            write_f64_as_dtype(relu_val, x.dtype, &mut result[i * bw..(i + 1) * bw]);
        }

        ConstTensor {
            data: result,
            shape: x.shape.clone(),
            dtype: x.dtype,
        }
    }
}

/// Write an f64 value back to a byte buffer in the target dtype.
pub fn write_f64_as_dtype(val: f64, dtype: WeightDType, buf: &mut [u8]) {
    match dtype {
        WeightDType::F32 => {
            let bytes = (val as f32).to_le_bytes();
            buf[..4].copy_from_slice(&bytes);
        }
        WeightDType::F16 => {
            let bytes = half::f16::from_f64(val).to_le_bytes();
            buf[..2].copy_from_slice(&bytes);
        }
        WeightDType::BF16 => {
            let bytes = half::bf16::from_f64(val).to_le_bytes();
            buf[..2].copy_from_slice(&bytes);
        }
        WeightDType::F64 => {
            buf[..8].copy_from_slice(&val.to_le_bytes());
        }
        WeightDType::I8 => {
            buf[0] = val.round().clamp(-128.0, 127.0) as i8 as u8;
        }
        WeightDType::I32 => {
            let bytes = (val.round() as i32).to_le_bytes();
            buf[..4].copy_from_slice(&bytes);
        }
        WeightDType::F8E4M3 | WeightDType::F8E5M2 => {
            // FP8: clamp and scale back
            buf[0] = (val * 127.0).round().clamp(0.0, 255.0) as u8;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dead Weight Elimination
// ─────────────────────────────────────────────────────────────────────────────

/// Dead weight elimination pass.
pub struct DeadWeightEliminator<'a> {
    config: &'a WeightAwareConfig,
}

impl<'a> DeadWeightEliminator<'a> {
    pub fn new(config: &'a WeightAwareConfig) -> Self {
        Self { config }
    }

    /// Analyze a weight entry and zero out near-dead elements.
    /// Returns the number of elements eliminated.
    pub fn eliminate(&self, entry: &mut WeightEntry) -> usize {
        let bw = entry.dtype.byte_width();
        let mut eliminated = 0usize;

        for i in 0..entry.num_elements {
            let offset = i * bw;
            let val = entry.dtype.to_f64(&entry.data[offset..offset + bw]);
            if val.abs() < self.config.dead_weight_threshold {
                // Zero out the element in the data buffer
                for byte in &mut entry.data[offset..offset + bw] {
                    *byte = 0;
                }
                eliminated += 1;
            }
        }

        if eliminated > 0 {
            // Mark as eliminated if more than half the elements were pruned
            entry.eliminated = eliminated > entry.num_elements / 2;
        }

        eliminated
    }

    /// Check if an entire layer (weight matrix) is near-identity.
    /// A square matrix W is near-identity if max(|W - I|) < identity_threshold.
    pub fn is_near_identity(&self, entry: &WeightEntry, threshold: f64) -> bool {
        if entry.shape.len() != 2 || entry.shape[0] != entry.shape[1] {
            return false;
        }
        let n = entry.shape[0];
        let bw = entry.dtype.byte_width();

        for row in 0..n {
            for col in 0..n {
                let idx = row * n + col;
                let val = entry.dtype.to_f64(&entry.data[idx * bw..(idx + 1) * bw]);
                let expected = if row == col { 1.0 } else { 0.0 };
                if (val - expected).abs() > threshold {
                    return false;
                }
            }
        }

        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SparsityHint (consumed by codegen)
// ─────────────────────────────────────────────────────────────────────────────

/// Hint attached to a matmul operation, consumed by codegen to choose
/// between dense and sparse kernel emission.
#[derive(Debug, Clone)]
pub enum SparsityHint {
    /// Weight matrix is dense enough for standard matmul kernel.
    Dense,
    /// Weight matrix is sparse -- annotated for future CSR-based sparse matmul.
    /// M52a: annotation only. M52b: actual sparse kernel emission.
    Sparse {
        /// Weight name in the WeightMap
        weight_name: String,
        /// Number of non-zero elements
        nnz: u32,
        /// Original dense dimensions
        nrows: u32,
        ncols: u32,
        /// Sparsity fraction (for logging/diagnostics)
        sparsity: f64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// WeightIntegrity
// ─────────────────────────────────────────────────────────────────────────────

/// SHA-256 hash of the weight file, embedded in the binary for runtime verification.
#[derive(Debug, Clone)]
pub struct WeightIntegrity {
    /// SHA-256 hash of the safetensors file used during compilation
    pub compile_hash: [u8; 32],
    /// Hex-encoded hash string
    pub hash_hex: String,
}

impl WeightIntegrity {
    pub fn new(hash: [u8; 32]) -> Self {
        let hash_hex = hash.iter().map(|b| format!("{:02x}", b)).collect();
        Self {
            compile_hash: hash,
            hash_hex,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight Analysis Report
// ─────────────────────────────────────────────────────────────────────────────

/// Cloned weight entry fields used for analysis without mutating the original WeightMap.
type AnalysisEntry = (String, Vec<u8>, Vec<usize>, WeightDType, usize);

/// Run the full weight analysis pass and print a report to stderr.
/// This is invoked by `nsl check --weight-analysis`.
///
/// CRITICAL FIX: takes `&WeightMap` (immutable) and clones data internally
/// for analysis, to avoid mutating the caller's weight map.
pub fn print_weight_analysis_report(weight_map: &WeightMap, config: &WeightAwareConfig) {
    eprintln!();
    eprintln!("Weight Analysis Report for {}:", weight_map.source_path());
    eprintln!("  Total parameters: {}", format_number(weight_map.total_parameters()));
    eprintln!("  Total size: {}", format_bytes(weight_map.total_bytes()));
    eprintln!("  Weight file SHA-256: {}", weight_map.hash_hex());
    eprintln!("  Tensors: {}", weight_map.len());
    eprintln!();

    let mut sparse_count = 0usize;
    let mut dead_count = 0usize;
    let mut total_near_zero = 0usize;
    let mut total_elements = 0usize;

    // Clone entry data for analysis to avoid mutating the original WeightMap
    let mut analysis_entries: Vec<AnalysisEntry> =
        weight_map.entries().map(|(name, entry)| {
            (name.clone(), entry.data.clone(), entry.shape.clone(), entry.dtype, entry.num_elements)
        }).collect();

    eprintln!("  Per-tensor sparsity:");
    for (name, data, shape, dtype, num_elements) in &mut analysis_entries {
        total_elements += *num_elements;

        // Create a temporary WeightEntry for analysis
        let mut temp_entry = WeightEntry {
            name: name.clone(),
            data: std::mem::take(data),
            shape: shape.clone(),
            dtype: *dtype,
            num_elements: *num_elements,
            sparsity: None,
            eliminated: false,
        };

        let info = temp_entry.analyze_sparsity(config);
        total_near_zero += info.near_zero_count;

        let kernel_label = if info.use_sparse_kernel {
            sparse_count += 1;
            "SPARSE"
        } else {
            "dense"
        };

        eprintln!(
            "    {}: {:>5.1}% near-zero ({} kernel)  [shape: {:?}, dtype: {:?}]",
            name,
            info.near_zero_fraction * 100.0,
            kernel_label,
            shape,
            dtype,
        );

        // Dead weight analysis on the clone
        if config.dead_weight_elim {
            let eliminator = DeadWeightEliminator::new(config);
            let count = eliminator.eliminate(&mut temp_entry);
            dead_count += count;
        }

        // Put data back for completeness
        *data = temp_entry.data;
    }

    eprintln!();
    eprintln!("  Summary:");
    eprintln!("    Global near-zero fraction: {:.1}%",
        if total_elements > 0 { total_near_zero as f64 / total_elements as f64 * 100.0 } else { 0.0 });
    eprintln!("    Sparse-eligible tensors: {}/{}", sparse_count, analysis_entries.len());
    if config.dead_weight_elim {
        eprintln!("    Dead weights eliminated: {} (threshold: {:e})", format_number(dead_count), config.dead_weight_threshold);
    }
    eprintln!();
}

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;
    use std::io::Write;

    /// Helper: write a minimal safetensors file with one f32 tensor.
    fn write_temp_safetensors(
        name: &str,
        shape: &[usize],
        data_f32: &[f32],
    ) -> tempfile::TempPath {
        let bytes: Vec<u8> = data_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.to_vec(),
            &bytes,
        )
        .unwrap();
        let mut map = StdHashMap::new();
        map.insert(name.to_string(), view);
        let serialized = safetensors::tensor::serialize(&map, &None).unwrap();

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&serialized).unwrap();
        tmp.into_temp_path()
    }

    /// Helper: write a safetensors file with multiple f32 tensors.
    fn write_temp_safetensors_multi(
        tensors: &[(&str, &[usize], &[f32])],
    ) -> tempfile::TempPath {
        let owned_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
            .iter()
            .map(|(name, shape, data)| {
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                (name.to_string(), bytes, shape.to_vec())
            })
            .collect();

        let views: StdHashMap<String, safetensors::tensor::TensorView<'_>> = owned_bytes
            .iter()
            .map(|(name, bytes, shape)| {
                let view = safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    shape.clone(),
                    bytes,
                )
                .unwrap();
                (name.clone(), view)
            })
            .collect();

        let serialized = safetensors::tensor::serialize(&views, &None).unwrap();
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&serialized).unwrap();
        tmp.into_temp_path()
    }

    // -- WeightMap loading ---------------------------------------------------

    #[test]
    fn test_weight_map_load_single_tensor() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let path = write_temp_safetensors("weight", &[2, 2], &data);

        let wmap = WeightMap::load(path.as_ref()).unwrap();
        assert_eq!(wmap.len(), 1);
        assert_eq!(wmap.total_parameters(), 4);

        let entry = wmap.get("weight").unwrap();
        assert_eq!(entry.shape, vec![2, 2]);
        assert_eq!(entry.num_elements, 4);
        assert!(matches!(entry.dtype, WeightDType::F32));

        // Verify values
        let v0 = entry.dtype.to_f64(&entry.data[0..4]);
        let v3 = entry.dtype.to_f64(&entry.data[12..16]);
        assert!((v0 - 1.0).abs() < 1e-6);
        assert!((v3 - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_weight_map_load_multiple_tensors() {
        let path = write_temp_safetensors_multi(&[
            ("w1", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("b1", &[3], &[0.1, 0.2, 0.3]),
        ]);

        let wmap = WeightMap::load(path.as_ref()).unwrap();
        assert_eq!(wmap.len(), 2);
        assert_eq!(wmap.total_parameters(), 9); // 6 + 3

        assert!(wmap.get("w1").is_some());
        assert!(wmap.get("b1").is_some());
        assert!(wmap.get("nonexistent").is_none());
    }

    #[test]
    fn test_weight_map_hash_deterministic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let path = write_temp_safetensors("w", &[4], &data);

        let wmap1 = WeightMap::load(path.as_ref()).unwrap();
        let wmap2 = WeightMap::load(path.as_ref()).unwrap();

        assert_eq!(wmap1.hash(), wmap2.hash());
        assert!(!wmap1.hash_hex().is_empty());
        assert_eq!(wmap1.hash_hex().len(), 64); // SHA-256 = 64 hex chars
    }

    #[test]
    fn test_weight_map_hash_differs_for_different_data() {
        let path1 = write_temp_safetensors("w", &[2], &[1.0, 2.0]);
        let path2 = write_temp_safetensors("w", &[2], &[3.0, 4.0]);

        let wmap1 = WeightMap::load(path1.as_ref()).unwrap();
        let wmap2 = WeightMap::load(path2.as_ref()).unwrap();

        assert_ne!(wmap1.hash(), wmap2.hash());
    }

    // -- Sparsity analysis ---------------------------------------------------

    #[test]
    fn test_sparsity_analysis_dense() {
        // 10% zeros -- should NOT qualify for sparse kernel
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0];
        let path = write_temp_safetensors("w", &[2, 5], &data);

        let mut wmap = WeightMap::load(path.as_ref()).unwrap();
        let config = WeightAwareConfig::default(); // sparse_threshold = 0.5
        let entry = wmap.get_mut("w").unwrap();
        let info = entry.analyze_sparsity(&config);

        assert!(!info.use_sparse_kernel);
        assert!((info.exact_zero_fraction - 0.1).abs() < 0.01);
        assert!(info.csr.is_none());
    }

    #[test]
    fn test_sparsity_analysis_sparse() {
        // 60% zeros -- should qualify for sparse kernel
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let path = write_temp_safetensors("w", &[2, 5], &data);

        let mut wmap = WeightMap::load(path.as_ref()).unwrap();
        let config = WeightAwareConfig::default();
        let entry = wmap.get_mut("w").unwrap();
        let info = entry.analyze_sparsity(&config);

        // Near-zero fraction should be >= 0.5 (7 out of 10 are exactly 0 or near-zero)
        assert!(info.use_sparse_kernel);
        assert!(info.csr.is_some());

        let csr = info.csr.as_ref().unwrap();
        assert_eq!(csr.nrows, 2);
        assert_eq!(csr.ncols, 5);
        assert_eq!(csr.nnz, 3); // Only 3 non-zero elements
    }

    #[test]
    fn test_csr_construction() {
        // Known 3x3 matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 0, 0]
        let data = vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0];
        let path = write_temp_safetensors("w", &[3, 3], &data);

        let mut wmap = WeightMap::load(path.as_ref()).unwrap();
        let config = WeightAwareConfig {
            sparse_threshold: 0.3, // Lower threshold to trigger sparse
            ..Default::default()
        };
        let entry = wmap.get_mut("w").unwrap();
        let info = entry.analyze_sparsity(&config);

        assert!(info.use_sparse_kernel);
        let csr = info.csr.as_ref().unwrap();

        // row_ptrs: [0, 2, 3, 4]
        assert_eq!(csr.row_ptrs, vec![0, 2, 3, 4]);
        // col_indices: [0, 2, 2, 0]
        assert_eq!(csr.col_indices, vec![0, 2, 2, 0]);
        assert_eq!(csr.nnz, 4);
    }

    // -- Constant folding ----------------------------------------------------

    #[test]
    fn test_constant_fold_matmul() {
        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // C = A @ B = [[19, 22], [43, 50]]
        let path = write_temp_safetensors_multi(&[
            ("a", &[2, 2], &[1.0, 2.0, 3.0, 4.0]),
            ("b", &[2, 2], &[5.0, 6.0, 7.0, 8.0]),
        ]);

        let wmap = WeightMap::load(path.as_ref()).unwrap();
        let mut folder = ConstantFolder::new(&wmap);

        let a_entry = wmap.get("a").unwrap();
        let b_entry = wmap.get("b").unwrap();

        let a_const = FoldResult::Constant(ConstTensor {
            data: a_entry.data.clone(),
            shape: a_entry.shape.clone(),
            dtype: a_entry.dtype,
        });
        let b_const = FoldResult::Constant(ConstTensor {
            data: b_entry.data.clone(),
            shape: b_entry.shape.clone(),
            dtype: b_entry.dtype,
        });

        let result = folder.fold_matmul(&a_const, &b_const);

        if let FoldResult::Constant(c) = result {
            assert_eq!(c.shape, vec![2, 2]);
            let v00 = c.dtype.to_f64(&c.data[0..4]);
            let v01 = c.dtype.to_f64(&c.data[4..8]);
            let v10 = c.dtype.to_f64(&c.data[8..12]);
            let v11 = c.dtype.to_f64(&c.data[12..16]);
            assert!((v00 - 19.0).abs() < 1e-5, "expected 19.0, got {}", v00);
            assert!((v01 - 22.0).abs() < 1e-5, "expected 22.0, got {}", v01);
            assert!((v10 - 43.0).abs() < 1e-5, "expected 43.0, got {}", v10);
            assert!((v11 - 50.0).abs() < 1e-5, "expected 50.0, got {}", v11);
        } else {
            panic!("expected Constant result from fold_matmul");
        }

        assert_eq!(folder.stats.folded_ops, 1);
    }

    #[test]
    fn test_constant_fold_add() {
        let path = write_temp_safetensors_multi(&[
            ("a", &[4], &[1.0, 2.0, 3.0, 4.0]),
            ("b", &[4], &[10.0, 20.0, 30.0, 40.0]),
        ]);

        let wmap = WeightMap::load(path.as_ref()).unwrap();
        let mut folder = ConstantFolder::new(&wmap);

        let a = FoldResult::Constant(ConstTensor {
            data: wmap.get("a").unwrap().data.clone(),
            shape: vec![4],
            dtype: WeightDType::F32,
        });
        let b = FoldResult::Constant(ConstTensor {
            data: wmap.get("b").unwrap().data.clone(),
            shape: vec![4],
            dtype: WeightDType::F32,
        });

        let result = folder.fold_add(&a, &b);

        if let FoldResult::Constant(c) = result {
            assert_eq!(c.shape, vec![4]);
            let vals: Vec<f64> = (0..4)
                .map(|i| c.dtype.to_f64(&c.data[i * 4..(i + 1) * 4]))
                .collect();
            assert!((vals[0] - 11.0).abs() < 1e-5);
            assert!((vals[1] - 22.0).abs() < 1e-5);
            assert!((vals[2] - 33.0).abs() < 1e-5);
            assert!((vals[3] - 44.0).abs() < 1e-5);
        } else {
            panic!("expected Constant result");
        }
    }

    #[test]
    fn test_constant_fold_relu() {
        let path = write_temp_safetensors("x", &[4], &[-1.0, 2.0, -3.0, 4.0]);

        let wmap = WeightMap::load(path.as_ref()).unwrap();
        let mut folder = ConstantFolder::new(&wmap);

        let x = FoldResult::Constant(ConstTensor {
            data: wmap.get("x").unwrap().data.clone(),
            shape: vec![4],
            dtype: WeightDType::F32,
        });

        let result = folder.fold_relu(&x);

        if let FoldResult::Constant(c) = result {
            let vals: Vec<f64> = (0..4)
                .map(|i| c.dtype.to_f64(&c.data[i * 4..(i + 1) * 4]))
                .collect();
            assert!((vals[0] - 0.0).abs() < 1e-5, "relu(-1) should be 0");
            assert!((vals[1] - 2.0).abs() < 1e-5, "relu(2) should be 2");
            assert!((vals[2] - 0.0).abs() < 1e-5, "relu(-3) should be 0");
            assert!((vals[3] - 4.0).abs() < 1e-5, "relu(4) should be 4");
        } else {
            panic!("expected Constant result");
        }
    }

    #[test]
    fn test_fold_matmul_partial_known_rhs() {
        let path = write_temp_safetensors("w", &[3, 3], &[1.0; 9]);
        let wmap = WeightMap::load(path.as_ref()).unwrap();
        let mut folder = ConstantFolder::new(&wmap);

        let runtime = FoldResult::Runtime;
        let known = FoldResult::Constant(ConstTensor {
            data: wmap.get("w").unwrap().data.clone(),
            shape: vec![3, 3],
            dtype: WeightDType::F32,
        });

        let result = folder.fold_matmul(&runtime, &known);
        assert!(matches!(result, FoldResult::Partial { .. }));
    }

    // -- Dead weight elimination ---------------------------------------------

    #[test]
    fn test_dead_weight_elimination() {
        // Values: [0.5, 1e-7, 0.3, 1e-8]
        // With threshold 1e-6: two elements should be zeroed
        let path = write_temp_safetensors("w", &[4], &[0.5, 1e-7, 0.3, 1e-8]);

        let mut wmap = WeightMap::load(path.as_ref()).unwrap();
        let config = WeightAwareConfig::default(); // threshold = 1e-6
        let eliminator = DeadWeightEliminator::new(&config);

        let entry = wmap.get_mut("w").unwrap();
        let eliminated = eliminator.eliminate(entry);

        assert_eq!(eliminated, 2, "should eliminate 2 near-zero elements");

        // Verify non-zero values survived
        let v0 = entry.dtype.to_f64(&entry.data[0..4]);
        let v2 = entry.dtype.to_f64(&entry.data[8..12]);
        assert!((v0 - 0.5).abs() < 1e-6);
        assert!((v2 - 0.3).abs() < 1e-6);

        // Verify near-zero values were zeroed
        let v1 = entry.dtype.to_f64(&entry.data[4..8]);
        let v3 = entry.dtype.to_f64(&entry.data[12..16]);
        assert_eq!(v1, 0.0);
        assert_eq!(v3, 0.0);
    }

    #[test]
    fn test_near_identity_detection() {
        // 3x3 identity matrix with small perturbation
        let data = vec![
            1.001, 0.002, -0.001,
            0.003, 0.999, 0.002,
            -0.001, 0.001, 1.002,
        ];
        let path = write_temp_safetensors("w", &[3, 3], &data);
        let wmap = WeightMap::load(path.as_ref()).unwrap();
        let config = WeightAwareConfig::default();
        let eliminator = DeadWeightEliminator::new(&config);

        let entry = wmap.get("w").unwrap();
        assert!(eliminator.is_near_identity(entry, 0.01));
    }

    #[test]
    fn test_near_identity_rejection() {
        // 3x3 matrix far from identity
        let data = vec![
            1.0, 0.05, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let path = write_temp_safetensors("w", &[3, 3], &data);
        let wmap = WeightMap::load(path.as_ref()).unwrap();
        let config = WeightAwareConfig::default();
        let eliminator = DeadWeightEliminator::new(&config);

        let entry = wmap.get("w").unwrap();
        assert!(!eliminator.is_near_identity(entry, 0.01));
    }

    #[test]
    fn test_near_identity_non_square_rejected() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let path = write_temp_safetensors("w", &[2, 3], &data);
        let wmap = WeightMap::load(path.as_ref()).unwrap();
        let config = WeightAwareConfig::default();
        let eliminator = DeadWeightEliminator::new(&config);

        let entry = wmap.get("w").unwrap();
        assert!(!eliminator.is_near_identity(entry, 0.01));
    }

    // -- WeightDType ---------------------------------------------------------

    #[test]
    fn test_weight_dtype_byte_width() {
        assert_eq!(WeightDType::F32.byte_width(), 4);
        assert_eq!(WeightDType::F16.byte_width(), 2);
        assert_eq!(WeightDType::BF16.byte_width(), 2);
        assert_eq!(WeightDType::F64.byte_width(), 8);
        assert_eq!(WeightDType::I8.byte_width(), 1);
        assert_eq!(WeightDType::I32.byte_width(), 4);
    }

    #[test]
    fn test_weight_dtype_from_safetensors_str() {
        assert!(matches!(WeightDType::from_safetensors_str("F32"), Ok(WeightDType::F32)));
        assert!(matches!(WeightDType::from_safetensors_str("F16"), Ok(WeightDType::F16)));
        assert!(matches!(WeightDType::from_safetensors_str("BF16"), Ok(WeightDType::BF16)));
        assert!(WeightDType::from_safetensors_str("UNKNOWN").is_err());
    }

    #[test]
    fn test_write_f64_as_f32_roundtrip() {
        let val = 3.14159f64;
        let mut buf = [0u8; 4];
        write_f64_as_dtype(val, WeightDType::F32, &mut buf);
        let restored = WeightDType::F32.to_f64(&buf);
        assert!((restored - val).abs() < 1e-5);
    }

    // -- WeightIntegrity -----------------------------------------------------

    #[test]
    fn test_weight_integrity() {
        let hash = [0xABu8; 32];
        let integrity = WeightIntegrity::new(hash);
        assert_eq!(integrity.compile_hash, hash);
        assert_eq!(integrity.hash_hex.len(), 64);
        assert!(integrity.hash_hex.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
