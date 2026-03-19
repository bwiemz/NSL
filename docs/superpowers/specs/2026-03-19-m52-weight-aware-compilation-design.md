# M52: Weight-Aware Compilation & Network Constant Folding — Design Specification

**Date:** 2026-03-19
**Status:** Planned
**Milestone:** M52
**Prerequisites:** M36 (Compile-Time Memory Planning), M24 (Standalone Export), M35 (FP8/Sub-byte Quantization)
**Dependencies:** M53 (WCET Proofs) and M55 (ZK Inference Circuits) both depend on M52's weight-aware static DAG

## Overview

M52 transforms NSL from a language that *compiles model code* into one that *compiles the model itself.* By loading `.safetensors` weight files at compile time and treating every weight tensor as a known constant, the compiler can perform constant propagation through matmul, add, relu, and other tensor operations — eliminating entire subgraphs, pruning near-zero weights, and emitting sparsity-aware GPU kernels uniquely optimized for a specific checkpoint.

This is **structurally impossible in Python/PyTorch.** PyTorch's compilation boundary separates the model definition (Python code) from the model parameters (runtime data). `torch.compile()` can optimize the computation graph, but it cannot inspect weight values because they are loaded after compilation. NSL's AOT architecture and static type system mean that weight values are available during the same compilation pass that emits Cranelift IR and PTX kernels. The compiler literally sees `matmul(input, [[0.23, 0.0, -0.01], [0.0, 0.0, 0.87], ...])` and optimizes accordingly.

The practical impact is threefold:

1. **Dead weight elimination.** A 7B parameter model where 15% of weights are near-zero (common after magnitude pruning or quantization) produces a binary that physically lacks the instructions for those multiply-accumulate operations. Not masked zeros — deleted instructions.

2. **Sparsity-aware codegen.** When the compiler detects that a weight matrix has >50% structural zeros, it emits a sparse matmul kernel (CSR format) instead of a dense one. The decision is made per-layer, per-matrix, based on the actual weight values.

3. **Per-checkpoint bespoke binaries.** Two fine-tuned variants of the same architecture produce different binaries with different instruction streams, different memory layouts, and different GPU kernel configurations. Each binary is maximally optimized for its weights.

The compiler embeds a SHA-256 hash of the weight file in the binary for integrity verification at load time, ensuring the binary never runs with weights it wasn't compiled for.

---

## Section 1: Language Surface

### CLI Interface

```bash
# Basic weight-aware compilation
nsl build model.nsl --weights model.safetensors

# With dead weight threshold (default: 1e-6)
nsl build model.nsl --weights model.safetensors --dead-weight-threshold 1e-4

# With sparsity-aware codegen (default: auto at >50% zeros)
nsl build model.nsl --weights model.safetensors --sparse-threshold 0.3

# Disable specific optimizations
nsl build model.nsl --weights model.safetensors --no-constant-fold
nsl build model.nsl --weights model.safetensors --no-dead-weight
nsl build model.nsl --weights model.safetensors --no-sparse-codegen

# Weight analysis report (no binary produced)
nsl check --weight-analysis model.nsl --weights model.safetensors

# Combined with standalone export (M24)
nsl build --standalone model.nsl --weights model.safetensors -o ./optimized_model
```

### NSL Source Annotations

Weight-aware compilation requires no source changes — the `--weights` flag is sufficient. However, optional annotations give fine-grained control:

```python
model Transformer:
    # Weights loaded from safetensors at compile time
    layers: List[TransformerBlock]
    lm_head: Linear

    @no_fold   # Prevent constant folding of this layer's weights
    fn embedding(self, tokens: Tensor<[B, S], i32>) -> Tensor<[B, S, D], fp16>:
        return self.embed_table[tokens]

    @sparse(threshold=0.4)  # Override global sparse threshold for this layer
    fn feed_forward(self, x: Tensor<[B, S, D], fp16>) -> Tensor<[B, S, D], fp16>:
        return self.w2 @ relu(self.w1 @ x)

    @fold_scales  # Explicitly request quantization scale fusion
    fn quantized_attn(self, x: Tensor<[B, S, D], fp8>) -> Tensor<[B, S, D], fp8>:
        # Quantization scales folded into PTX mul instructions
        return self.attn(x)
```

### Weight Analysis Report

```
nsl check --weight-analysis model.nsl --weights llama-7b.safetensors

Weight Analysis Report for llama-7b.safetensors:
  Total parameters: 6,738,415,616
  Total size: 12.54 GB (FP16)
  Weight file SHA-256: a3f8c2...d91e

  Per-layer sparsity:
    layers.0.attn.q_proj:  12.3% near-zero  (dense matmul)
    layers.0.attn.k_proj:   8.7% near-zero  (dense matmul)
    layers.0.ffn.gate_proj: 54.2% near-zero  (SPARSE matmul — CSR kernel)
    layers.0.ffn.up_proj:   51.8% near-zero  (SPARSE matmul — CSR kernel)
    ...
    layers.31.attn.o_proj:  11.1% near-zero  (dense matmul)

  Constant folding opportunities:
    - Bias additions: 64 layers can fold bias into weight matrix
    - Scale fusion: 128 quantization scales can fold into mul instructions
    - Identity layers: layers.28.ffn near-identity (max |W - I| = 0.003)

  Estimated binary size reduction: 23.4% vs. generic compilation
  Estimated inference speedup: 1.18x (from sparsity + constant folding)
```

---

## Section 2: Architecture

### Compilation Pipeline Extension

The weight-aware pass inserts between semantic analysis and Cranelift codegen:

```
Parse → Semantic Check → Type Map → WeightAwarePass → MemoryPlanner (M36) → Cranelift Codegen
                                          ↓
                                    WeightMap loaded
                                    Constant propagation
                                    Dead weight elimination
                                    Sparsity analysis
                                    Scale fusion
                                    Modified AST + SparsityHints
```

### Core Data Structures

```rust
// crates/nsl-codegen/src/weight_aware.rs

use std::collections::HashMap;
use sha2::{Sha256, Digest};

/// Index of all weight tensors loaded from a safetensors file at compile time.
/// Each entry maps a parameter name (e.g., "layers.0.attn.q_proj.weight") to
/// its raw tensor data and metadata.
pub struct WeightMap {
    /// Parameter name -> WeightEntry
    entries: HashMap<String, WeightEntry>,
    /// SHA-256 hash of the entire safetensors file
    file_hash: [u8; 32],
    /// Original file path (for error messages)
    source_path: String,
    /// Total bytes loaded
    total_bytes: u64,
}

/// A single weight tensor loaded from safetensors.
pub struct WeightEntry {
    /// Parameter name as it appears in the safetensors file
    name: String,
    /// Raw weight data in the file's native dtype
    data: Vec<u8>,
    /// Shape dimensions
    shape: Vec<usize>,
    /// Data type of the weight tensor
    dtype: WeightDType,
    /// Number of elements
    num_elements: usize,
    /// Sparsity analysis results (computed lazily on first access)
    sparsity: Option<SparsityInfo>,
    /// Whether this weight has been constant-folded away
    folded: bool,
    /// Whether this weight has been eliminated (near-zero)
    eliminated: bool,
}

/// Supported weight data types in safetensors files.
#[derive(Debug, Clone, Copy, PartialEq)]
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
                // FP8 conversion via lookup table (256 entries)
                fp8_to_f64(bytes[0], *self)
            }
        }
    }
}

/// Sparsity analysis for a single weight matrix.
#[derive(Debug, Clone)]
pub struct SparsityInfo {
    /// Fraction of elements with |value| < threshold
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
/// Used for emitting sparse matmul kernels.
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

/// Configuration for weight-aware compilation passes.
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
    /// Enable quantization scale fusion
    pub scale_fusion: bool,
    /// Enable identity layer detection and removal
    pub identity_elim: bool,
    /// Identity layer threshold: max |W - I| for a layer to be considered identity
    pub identity_threshold: f64,
}

impl Default for WeightAwareConfig {
    fn default() -> Self {
        Self {
            dead_weight_threshold: 1e-6,
            sparse_threshold: 0.5,
            constant_fold: true,
            dead_weight_elim: true,
            sparse_codegen: true,
            scale_fusion: true,
            identity_elim: true,
            identity_threshold: 0.01,
        }
    }
}
```

### Safetensors Loader

```rust
// crates/nsl-codegen/src/weight_aware.rs (continued)

impl WeightMap {
    /// Load all tensors from a safetensors file into memory.
    /// Computes the SHA-256 hash of the entire file for integrity verification.
    pub fn load(path: &std::path::Path) -> Result<Self, WeightLoadError> {
        let file_bytes = std::fs::read(path)
            .map_err(|e| WeightLoadError::IoError(path.display().to_string(), e))?;

        // Compute SHA-256 of entire file
        let mut hasher = Sha256::new();
        hasher.update(&file_bytes);
        let file_hash: [u8; 32] = hasher.finalize().into();

        // Parse safetensors header
        // Format: 8 bytes (u64 LE) header_size, then JSON header, then raw data
        if file_bytes.len() < 8 {
            return Err(WeightLoadError::InvalidFormat("file too small".into()));
        }
        let header_size = u64::from_le_bytes(file_bytes[..8].try_into().unwrap()) as usize;
        if 8 + header_size > file_bytes.len() {
            return Err(WeightLoadError::InvalidFormat("header size exceeds file".into()));
        }

        let header_json: serde_json::Value = serde_json::from_slice(&file_bytes[8..8 + header_size])
            .map_err(|e| WeightLoadError::InvalidHeader(e.to_string()))?;

        let data_start = 8 + header_size;
        let mut entries = HashMap::new();
        let mut total_bytes = 0u64;

        if let serde_json::Value::Object(map) = &header_json {
            for (name, info) in map {
                if name == "__metadata__" { continue; }

                let dtype_str = info["dtype"].as_str()
                    .ok_or_else(|| WeightLoadError::MissingField(name.clone(), "dtype"))?;
                let dtype = WeightDType::from_safetensors_str(dtype_str)?;

                let shape: Vec<usize> = info["shape"].as_array()
                    .ok_or_else(|| WeightLoadError::MissingField(name.clone(), "shape"))?
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();

                let offsets = &info["data_offsets"];
                let start = offsets[0].as_u64().unwrap() as usize;
                let end = offsets[1].as_u64().unwrap() as usize;

                let num_elements: usize = shape.iter().product();
                let data = file_bytes[data_start + start..data_start + end].to_vec();
                total_bytes += data.len() as u64;

                entries.insert(name.clone(), WeightEntry {
                    name: name.clone(),
                    data,
                    shape,
                    dtype,
                    num_elements,
                    sparsity: None,
                    folded: false,
                    eliminated: false,
                });
            }
        }

        Ok(WeightMap {
            entries,
            file_hash,
            source_path: path.display().to_string(),
            total_bytes,
        })
    }

    /// Retrieve a weight tensor by parameter name.
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

    /// Total number of parameters across all tensors.
    pub fn total_parameters(&self) -> usize {
        self.entries.values().map(|e| e.num_elements).sum()
    }
}

#[derive(Debug)]
pub enum WeightLoadError {
    IoError(String, std::io::Error),
    InvalidFormat(String),
    InvalidHeader(String),
    MissingField(String, &'static str),
    UnsupportedDType(String),
}

impl WeightDType {
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
```

---

## Section 3: Sparsity Analysis

### Per-Weight Sparsity Computation

```rust
// crates/nsl-codegen/src/weight_aware.rs (continued)

impl WeightEntry {
    /// Compute sparsity statistics for this weight tensor.
    /// Results are cached on first call.
    pub fn analyze_sparsity(&mut self, config: &WeightAwareConfig) -> &SparsityInfo {
        if self.sparsity.is_some() {
            return self.sparsity.as_ref().unwrap();
        }

        let bw = self.dtype.byte_width();
        let mut near_zero_count: usize = 0;
        let mut exact_zero_count: usize = 0;
        let mut max_abs: f64 = 0.0;
        let mut min_nonzero_abs: f64 = f64::MAX;

        // Iterate over all elements
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

        let near_zero_fraction = near_zero_count as f64 / self.num_elements as f64;
        let exact_zero_fraction = exact_zero_count as f64 / self.num_elements as f64;
        let use_sparse = near_zero_fraction >= config.sparse_threshold && self.shape.len() == 2;

        // Compute per-row sparsity (only for 2D weight matrices)
        let row_sparsity = if self.shape.len() == 2 {
            let nrows = self.shape[0];
            let ncols = self.shape[1];
            (0..nrows).map(|row| {
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
            }).collect()
        } else {
            Vec::new()
        };

        // Build CSR layout if sparse
        let csr = if use_sparse && self.shape.len() == 2 {
            Some(self.build_csr(config))
        } else {
            None
        };

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
```

---

## Section 4: Constant Propagation Pass

### Constant Folding Through Tensor Operations

The constant propagation pass walks the AST after weight loading and attempts to evaluate tensor operations with known inputs at compile time.

```rust
// crates/nsl-codegen/src/weight_aware.rs (continued)

/// Result of attempting to constant-fold a tensor expression.
#[derive(Debug)]
pub enum FoldResult {
    /// Expression was fully folded to a known constant tensor
    Constant(ConstTensor),
    /// Expression depends on runtime input — cannot fold
    Runtime,
    /// Expression was partially folded (some operands known)
    Partial {
        /// Which operands are known constants
        known_operands: Vec<usize>,
        /// The simplified expression after partial folding
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

/// The constant propagation pass.
pub struct ConstantFolder<'a> {
    /// Weight map providing known weight values
    weight_map: &'a WeightMap,
    /// Cache of already-folded expressions (AST node ID -> FoldResult)
    fold_cache: HashMap<u64, FoldResult>,
    /// Statistics for reporting
    stats: FoldStats,
}

#[derive(Debug, Default)]
pub struct FoldStats {
    /// Number of expressions evaluated at compile time
    pub folded_ops: usize,
    /// Number of expressions that remain runtime-dependent
    pub runtime_ops: usize,
    /// Number of dead weight eliminations
    pub dead_weights_eliminated: usize,
    /// Number of sparse kernels emitted
    pub sparse_kernels: usize,
    /// Number of scale fusions performed
    pub scale_fusions: usize,
    /// Number of identity layers eliminated
    pub identity_layers: usize,
    /// Bytes of weight data eliminated
    pub bytes_eliminated: u64,
}

impl<'a> ConstantFolder<'a> {
    pub fn new(weight_map: &'a WeightMap) -> Self {
        Self {
            weight_map,
            fold_cache: HashMap::new(),
            stats: FoldStats::default(),
        }
    }

    /// Attempt to constant-fold a matmul: C = A @ B.
    /// If both A and B are known constants, compute C at compile time.
    /// If only B is known (weight matrix), annotate for sparsity-aware codegen.
    pub fn fold_matmul(
        &mut self,
        lhs: &FoldResult,
        rhs: &FoldResult,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
    ) -> FoldResult {
        match (lhs, rhs) {
            (FoldResult::Constant(a), FoldResult::Constant(b)) => {
                // Both operands known — fully evaluate at compile time
                let c = self.compute_matmul(a, b);
                self.stats.folded_ops += 1;
                FoldResult::Constant(c)
            }
            (FoldResult::Runtime, FoldResult::Constant(_)) => {
                // Input is runtime, weight is known — mark for sparsity-aware codegen
                // but cannot fold further
                FoldResult::Partial {
                    known_operands: vec![1],
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
            _ => FoldResult::Runtime,
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
            _ => FoldResult::Runtime,
        }
    }

    /// Compute matmul of two constant tensors at compile time.
    /// Supports 2D [M, K] x [K, N] and batched [B, M, K] x [B, K, N].
    fn compute_matmul(&self, a: &ConstTensor, b: &ConstTensor) -> ConstTensor {
        assert_eq!(a.shape.len(), 2, "compile-time matmul requires 2D tensors");
        assert_eq!(b.shape.len(), 2, "compile-time matmul requires 2D tensors");
        let m = a.shape[0];
        let k = a.shape[1];
        assert_eq!(k, b.shape[0], "matmul dimension mismatch");
        let n = b.shape[1];

        let bw = a.dtype.byte_width();
        let mut result = vec![0u8; m * n * bw];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    let a_val = a.dtype.to_f64(&a.data[(i * k + p) * bw..(i * k + p + 1) * bw]);
                    let b_val = b.dtype.to_f64(&b.data[(p * n + j) * bw..(p * n + j + 1) * bw]);
                    sum += a_val * b_val;
                }
                write_f64_as_dtype(sum, a.dtype, &mut result[(i * n + j) * bw..(i * n + j + 1) * bw]);
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
fn write_f64_as_dtype(val: f64, dtype: WeightDType, buf: &mut [u8]) {
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
        _ => {
            // INT8/I32/FP8: round-to-nearest
            buf[..dtype.byte_width()].copy_from_slice(&(val as i64).to_le_bytes()[..dtype.byte_width()]);
        }
    }
}
```

---

## Section 5: Dead Weight Elimination

### Near-Zero Weight Pruning

Dead weight elimination removes weight elements whose absolute value falls below a configurable threshold (default: 1e-6). For dense matmul operations, this translates to fewer multiply-accumulate instructions. For sparse-eligible matrices, dead weight elimination produces the CSR layout directly.

```rust
// crates/nsl-codegen/src/weight_aware.rs (continued)

/// Dead weight elimination pass.
pub struct DeadWeightEliminator<'a> {
    config: &'a WeightAwareConfig,
    stats: &'a mut FoldStats,
}

impl<'a> DeadWeightEliminator<'a> {
    pub fn new(config: &'a WeightAwareConfig, stats: &'a mut FoldStats) -> Self {
        Self { config, stats }
    }

    /// Analyze a weight entry and determine which elements to eliminate.
    /// Returns the number of elements eliminated.
    pub fn eliminate(&mut self, entry: &mut WeightEntry) -> usize {
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
            entry.eliminated = eliminated > entry.num_elements / 2;
            self.stats.dead_weights_eliminated += eliminated;
            self.stats.bytes_eliminated += (eliminated * bw) as u64;
        }

        eliminated
    }

    /// Check if an entire layer (weight matrix) is near-identity.
    /// A square matrix W is near-identity if max(|W - I|) < identity_threshold.
    pub fn is_near_identity(&self, entry: &WeightEntry) -> bool {
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
                if (val - expected).abs() > self.config.identity_threshold {
                    return false;
                }
            }
        }

        true
    }
}
```

---

## Section 6: Sparsity-Aware Codegen

### Dense vs. Sparse Kernel Decision

When the weight-aware pass determines that a weight matrix has sparsity exceeding the configured threshold, it annotates the corresponding matmul operation with a `SparsityHint` that the Cranelift/PTX codegen phase reads.

```rust
// crates/nsl-codegen/src/weight_aware.rs (continued)

/// Hint attached to a matmul AST node, consumed by codegen to choose
/// between dense and sparse kernel emission.
#[derive(Debug, Clone)]
pub enum SparsityHint {
    /// Weight matrix is dense enough for standard matmul kernel
    Dense,
    /// Weight matrix is sparse — use CSR-based sparse matmul
    Sparse {
        /// CSR data embedded as a constant in .rodata
        csr_symbol: String,
        /// Number of non-zero elements
        nnz: u32,
        /// Original dense dimensions
        nrows: u32,
        ncols: u32,
        /// Sparsity fraction (for logging/diagnostics)
        sparsity: f64,
    },
    /// Weight matrix has block sparsity — use block-sparse matmul
    BlockSparse {
        /// Block size (e.g., 32x32)
        block_rows: u32,
        block_cols: u32,
        /// Number of non-zero blocks
        num_nonzero_blocks: u32,
        /// Block index data embedded as .rodata constant
        block_index_symbol: String,
    },
}
```

### PTX Sparse Matmul Kernel

The sparse matmul kernel reads from CSR-formatted weight data embedded in `.rodata`:

```rust
// crates/nsl-codegen/src/weight_aware.rs (continued)

/// Generate PTX kernel code for CSR sparse matmul: Y[M, N] = X[M, K] * W_csr[K, N].
/// Each thread computes one element of the output.
pub fn emit_sparse_matmul_ptx(
    csr: &CsrLayout,
    input_rows: u32,
    output_cols: u32,
) -> String {
    format!(
        r#".version 7.0
.target sm_80
.address_size 64

.visible .entry sparse_matmul_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 csr_row_ptrs,
    .param .u64 csr_col_indices,
    .param .u64 csr_values,
    .param .u32 M,
    .param .u32 N,
    .param .u32 K
)
{{
    .reg .u32 %r<32>;
    .reg .u64 %rd<32>;
    .reg .f32 %f<16>;
    .reg .pred %p<4>;

    // Thread ID = row * N + col
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;
    add.u32 %r3, %r3, %r0;

    // row = global_id / N, col = global_id % N
    ld.param.u32 %r4, [N];
    div.u32 %r5, %r3, %r4;   // row
    rem.u32 %r6, %r3, %r4;   // col

    // Bounds check
    ld.param.u32 %r7, [M];
    setp.ge.u32 %p0, %r5, %r7;
    setp.ge.u32 %p1, %r6, %r4;
    or.pred %p2, %p0, %p1;
    @%p2 bra DONE;

    // Load row_ptrs[row] and row_ptrs[row+1]
    ld.param.u64 %rd0, [csr_row_ptrs];
    cvt.u64.u32 %rd1, %r5;
    mul.lo.u64 %rd1, %rd1, 4;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r8, [%rd2];       // start
    ld.global.u32 %r9, [%rd2 + 4];   // end

    // Accumulate: sum = 0
    mov.f32 %f0, 0.0;

    // Loop over non-zero elements in this row
    mov.u32 %r10, %r8;
LOOP:
    setp.ge.u32 %p3, %r10, %r9;
    @%p3 bra STORE;

    // Load col_index = csr_col_indices[r10]
    ld.param.u64 %rd3, [csr_col_indices];
    cvt.u64.u32 %rd4, %r10;
    mul.lo.u64 %rd4, %rd4, 4;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.u32 %r11, [%rd5];      // k = col_index

    // Load csr_value = csr_values[r10]
    ld.param.u64 %rd6, [csr_values];
    mul.lo.u64 %rd7, %rd4, 1;        // same index, f32 = 4 bytes
    add.u64 %rd8, %rd6, %rd7;
    ld.global.f32 %f1, [%rd8];       // weight value

    // Load input[k * N + col]  (transposed access pattern)
    ld.param.u64 %rd9, [input_ptr];
    cvt.u64.u32 %rd10, %r11;
    cvt.u64.u32 %rd11, %r6;
    mul.lo.u64 %rd10, %rd10, 4;
    // ... (full index computation)
    ld.global.f32 %f2, [%rd9];       // input value

    // sum += weight * input
    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r10, %r10, 1;
    bra LOOP;

STORE:
    // Store output[row * N + col] = sum
    ld.param.u64 %rd12, [output_ptr];
    cvt.u64.u32 %rd13, %r5;
    cvt.u64.u32 %rd14, %r6;
    mul.lo.u64 %rd13, %rd13, 4;
    // ... (full index computation)
    st.global.f32 [%rd12], %f0;

DONE:
    ret;
}}
"#
    )
}
```

---

## Section 7: Quantization Scale Fusion

### Folding Scales into PTX Instructions

When a quantized model (M35 FP8/INT8) performs `dequant(x) = x * scale + zero_point`, and `scale` and `zero_point` are compile-time constants (from the weight file), the compiler fuses these into the matmul kernel itself. Instead of a separate dequantization pass:

```rust
// crates/nsl-codegen/src/weight_aware.rs (continued)

/// Scale fusion information for quantized weight matrices.
#[derive(Debug, Clone)]
pub struct ScaleFusion {
    /// Per-channel scales (one scale per output channel)
    pub scales: Vec<f32>,
    /// Per-channel zero points
    pub zero_points: Vec<f32>,
    /// Granularity: per-tensor, per-channel, or per-group
    pub granularity: ScaleGranularity,
    /// Group size if per-group (e.g., 128)
    pub group_size: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub enum ScaleGranularity {
    PerTensor,
    PerChannel,
    PerGroup,
}

/// Determine if scale fusion is possible for a quantized weight + its scales.
pub fn analyze_scale_fusion(
    weight: &WeightEntry,
    scale_name: &str,
    zero_point_name: &str,
    weight_map: &WeightMap,
) -> Option<ScaleFusion> {
    let scale_entry = weight_map.get(scale_name)?;
    let zp_entry = weight_map.get(zero_point_name);

    // Determine granularity from scale shape
    let granularity = if scale_entry.num_elements == 1 {
        ScaleGranularity::PerTensor
    } else if scale_entry.shape.len() == 1 && scale_entry.shape[0] == weight.shape[0] {
        ScaleGranularity::PerChannel
    } else {
        ScaleGranularity::PerGroup
    };

    let bw = scale_entry.dtype.byte_width();
    let scales: Vec<f32> = (0..scale_entry.num_elements)
        .map(|i| scale_entry.dtype.to_f64(&scale_entry.data[i * bw..(i + 1) * bw]) as f32)
        .collect();

    let zero_points: Vec<f32> = if let Some(zp) = zp_entry {
        let zp_bw = zp.dtype.byte_width();
        (0..zp.num_elements)
            .map(|i| zp.dtype.to_f64(&zp.data[i * zp_bw..(i + 1) * zp_bw]) as f32)
            .collect()
    } else {
        vec![0.0f32; scales.len()]
    };

    let group_size = if matches!(granularity, ScaleGranularity::PerGroup) {
        Some((weight.num_elements / scale_entry.num_elements) as u32)
    } else {
        None
    };

    Some(ScaleFusion {
        scales,
        zero_points,
        granularity,
        group_size,
    })
}
```

### Fused Dequant+Matmul PTX

When scale fusion is active, the matmul kernel inlines the dequantization:

```
// Instead of:
//   dequant_weight = weight_int8 * scale + zero_point   // separate kernel launch
//   output = input @ dequant_weight                      // second kernel launch

// Fused kernel:
//   For each output element:
//     sum = 0
//     for k in 0..K:
//       w_int8 = load(weight[row, k])
//       w_fp32 = w_int8 * scale[row] + zero_point[row]  // inline dequant
//       sum += input[k, col] * w_fp32
//     output[row, col] = sum
```

This eliminates one kernel launch, one global memory read, and one global memory write per quantized matmul operation. For a 32-layer transformer, that is 128 fewer kernel launches (4 projections per layer).

---

## Section 8: Weight Hash Integrity

### SHA-256 Embedding

```rust
// crates/nsl-codegen/src/weight_aware.rs (continued)

/// Embed the weight file hash into the compiled binary's .rodata section.
/// At runtime, the binary can verify that loaded weights match the compilation hash.
pub struct WeightIntegrity {
    /// SHA-256 hash of the safetensors file used during compilation
    pub compile_hash: [u8; 32],
    /// Cranelift DataId for the .rodata symbol
    pub rodata_symbol: String,
}

/// Generate the runtime integrity check function.
/// This function is called at program startup when weights are loaded from a sidecar file.
/// If the binary was compiled with --weights, the hash is embedded; if the sidecar file
/// has a different hash, the program aborts with a clear error message.
///
/// When weights are embedded in .rodata (small model path from M24), the integrity
/// check is skipped because the weights are part of the binary itself.
pub fn emit_integrity_check_code(hash: &[u8; 32]) -> String {
    let hash_hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
    format!(
        "// Auto-generated weight integrity check\n\
         // Expected SHA-256: {hash_hex}\n\
         const EXPECTED_WEIGHT_HASH: [u8; 32] = {:?};\n",
        hash
    )
}
```

### Runtime Weight Map

```rust
// crates/nsl-runtime/src/weight_map.rs

use std::collections::HashMap;

/// Runtime weight accessor for weight-aware compiled binaries.
/// Provides read-only access to weight tensors loaded at compile time
/// and embedded in the binary or loaded from a sidecar file.
#[repr(C)]
pub struct RuntimeWeightMap {
    /// Number of weight entries
    pub num_entries: u32,
    /// Pointer to entry metadata array
    pub entries: *const RuntimeWeightEntry,
    /// Pointer to contiguous weight data block
    pub data_base: *const u8,
    /// Total size of weight data in bytes
    pub data_size: u64,
    /// SHA-256 hash of original safetensors file
    pub hash: [u8; 32],
}

#[repr(C)]
pub struct RuntimeWeightEntry {
    /// Null-terminated name string pointer
    pub name: *const u8,
    /// Byte offset into data_base
    pub data_offset: u64,
    /// Size in bytes
    pub data_size: u64,
    /// Number of dimensions
    pub ndim: u32,
    /// Pointer to shape array (u64 * ndim)
    pub shape: *const u64,
    /// Data type code (0=f16, 1=bf16, 2=f32, 3=f64, 4=f8e4m3, 5=f8e5m2, 6=i8, 7=i32)
    pub dtype: u8,
}

/// FFI: Initialize the runtime weight map from embedded .rodata or sidecar file.
#[no_mangle]
pub extern "C" fn nsl_weight_map_init(
    embedded_data: *const u8,
    embedded_size: u64,
    expected_hash: *const u8,
) -> *mut RuntimeWeightMap {
    // If embedded_data is non-null, use embedded weights
    // If null, look for sidecar file
    // In both cases, verify SHA-256 hash matches expected_hash
    todo!()
}

/// FFI: Retrieve a weight tensor pointer by name.
#[no_mangle]
pub extern "C" fn nsl_weight_map_get(
    map: *const RuntimeWeightMap,
    name: *const u8,
    name_len: u32,
) -> *const u8 {
    // Linear scan of entries (weight count is typically < 1000, not worth a hash map)
    todo!()
}

/// FFI: Verify weight integrity by computing SHA-256 of loaded data.
#[no_mangle]
pub extern "C" fn nsl_weight_map_verify(
    map: *const RuntimeWeightMap,
) -> i32 {
    // Returns 1 if hash matches, 0 if mismatch
    // On mismatch, prints error to stderr with expected vs. actual hashes
    todo!()
}

/// FFI: Free the runtime weight map.
#[no_mangle]
pub extern "C" fn nsl_weight_map_free(map: *mut RuntimeWeightMap) {
    todo!()
}
```

---

## Section 9: Codegen Changes

### WeightAwarePass Integration

The `WeightAwarePass` is invoked from the main compiler pipeline after semantic checking and before memory planning:

```rust
// Changes to crates/nsl-codegen/src/compiler.rs

/// Extended CompileOptions with weight-aware fields.
pub struct CompileOptions {
    // ... existing fields ...

    /// Path to safetensors weight file for weight-aware compilation
    pub weight_file: Option<std::path::PathBuf>,
    /// Weight-aware compilation configuration
    pub weight_config: WeightAwareConfig,
    /// Whether to emit a weight analysis report
    pub weight_analysis: bool,
}

/// Extended Compiler state with weight-aware data.
pub struct Compiler {
    // ... existing fields ...

    /// Loaded weight map (populated when --weights is passed)
    pub weight_map: Option<WeightMap>,
    /// Sparsity hints per matmul operation (AST node ID -> SparsityHint)
    pub sparsity_hints: HashMap<u64, SparsityHint>,
    /// Constant tensors from folding (AST node ID -> embedded .rodata symbol)
    pub folded_constants: HashMap<u64, String>,
    /// Scale fusion info per quantized layer
    pub scale_fusions: HashMap<String, ScaleFusion>,
    /// Weight integrity hash for embedding
    pub weight_integrity: Option<WeightIntegrity>,
}
```

### Cranelift IR Emission Changes

When the codegen encounters a matmul AST node with a `SparsityHint::Sparse`, it emits a call to the sparse matmul runtime function instead of the dense one:

```rust
// Changes to crates/nsl-codegen/src/expr.rs

/// When compiling a matmul expression, check for sparsity hints.
fn compile_matmul(&mut self, lhs: &Expr, rhs: &Expr) -> CraneliftValue {
    let node_id = rhs.node_id(); // weight is typically the RHS

    if let Some(hint) = self.sparsity_hints.get(&node_id) {
        match hint {
            SparsityHint::Dense => {
                // Normal dense matmul codegen (existing path)
                self.compile_dense_matmul(lhs, rhs)
            }
            SparsityHint::Sparse { csr_symbol, nnz, nrows, ncols, .. } => {
                // Emit call to sparse matmul runtime function
                // Load CSR data from .rodata symbol
                let csr_data = self.module.declare_data(csr_symbol, Linkage::Import, false, false).unwrap();
                // ... emit nsl_sparse_matmul call with CSR pointers ...
                self.compile_sparse_matmul(lhs, csr_symbol, *nnz, *nrows, *ncols)
            }
            SparsityHint::BlockSparse { .. } => {
                // Block-sparse matmul (similar but with block-level CSR)
                self.compile_block_sparse_matmul(lhs, hint)
            }
        }
    } else {
        self.compile_dense_matmul(lhs, rhs)
    }
}
```

### Runtime FFI: Sparse Matmul

```rust
// New functions in crates/nsl-runtime/src/tensor.rs (or new sparse.rs module)

/// Sparse matmul: Y = X @ W_csr (CSR format weight matrix).
/// On GPU: launches sparse matmul PTX kernel.
/// On CPU: uses direct CSR iteration.
#[no_mangle]
pub extern "C" fn nsl_sparse_matmul(
    input: *const NslTensor,
    csr_row_ptrs: *const u32,
    csr_col_indices: *const u32,
    csr_values: *const u8,
    csr_nnz: u32,
    weight_rows: u32,
    weight_cols: u32,
    weight_dtype: u8,
) -> *mut NslTensor {
    // Allocate output tensor [input_rows, weight_cols]
    // Dispatch to CPU or GPU based on input tensor device
    todo!()
}

/// Fused dequant + matmul: Y = X @ (W_int8 * scale + zero_point).
/// Eliminates separate dequantization kernel launch.
#[no_mangle]
pub extern "C" fn nsl_fused_dequant_matmul(
    input: *const NslTensor,
    weight_quantized: *const u8,
    scales: *const f32,
    zero_points: *const f32,
    weight_rows: u32,
    weight_cols: u32,
    scale_granularity: u8,  // 0=per-tensor, 1=per-channel, 2=per-group
    group_size: u32,
) -> *mut NslTensor {
    todo!()
}
```

---

## Section 10: Type System

### Weight Constness in the Type Map

When `--weights` is provided, the semantic checker annotates each model parameter's type with a `ConstWeight` marker:

```rust
// Extension to crates/nsl-semantic/src/types.rs

/// Marker indicating that a tensor type's values are known at compile time.
#[derive(Debug, Clone)]
pub enum TensorConstness {
    /// Values unknown at compile time (normal runtime tensor)
    Runtime,
    /// Values fully known at compile time (from --weights)
    CompileTimeKnown {
        /// Name in the weight map
        weight_name: String,
        /// SHA-256 hash of this specific tensor's data
        tensor_hash: [u8; 32],
    },
    /// Values partially known (e.g., after a matmul with one known operand)
    PartiallyKnown,
}
```

This constness propagates through expressions:
- `known_weight @ runtime_input` -> `Runtime` (output depends on input)
- `known_weight + known_bias` -> `CompileTimeKnown` (can fold)
- `relu(known_tensor)` -> `CompileTimeKnown` (can fold)

The type checker ensures that `@no_fold`-annotated functions do not have their weights folded, even if they are compile-time known.

---

## Section 11: Testing Strategy

### Unit Tests

| Test | Description |
|------|-------------|
| `test_weight_map_load` | Load a small safetensors file (3 tensors: 2x3, 3x2, 3-vector). Verify shapes, dtypes, and values are correct. |
| `test_weight_map_hash` | Load same file twice, verify SHA-256 matches. Modify one byte, verify hash differs. |
| `test_sparsity_analysis_dense` | Weight matrix with 10% zeros — verify `use_sparse_kernel = false`. |
| `test_sparsity_analysis_sparse` | Weight matrix with 60% zeros — verify `use_sparse_kernel = true`, CSR layout correct. |
| `test_csr_construction` | 4x4 matrix with known sparsity pattern. Verify `row_ptrs`, `col_indices`, `values` match expected. |
| `test_constant_fold_matmul` | Two known 3x3 matrices. Fold matmul at compile time, verify result matches naive computation. |
| `test_constant_fold_add` | Two known 4-vectors. Fold add, verify result. |
| `test_constant_fold_relu` | Known vector with positive and negative values. Fold relu, verify negatives become zero. |
| `test_dead_weight_threshold` | Weight matrix with values [0.5, 1e-7, 0.3, 1e-8]. Threshold 1e-6 eliminates two elements. |
| `test_near_identity_detection` | 4x4 matrix close to identity (max diff 0.005). Verify `is_near_identity = true`. |
| `test_near_identity_rejection` | 4x4 matrix with max diff 0.02 (> threshold 0.01). Verify `is_near_identity = false`. |
| `test_scale_fusion_per_channel` | INT8 weight + per-channel scales. Verify `ScaleFusion` has correct granularity and values. |
| `test_scale_fusion_per_group` | INT8 weight + group-128 scales. Verify group_size = 128. |

### E2E Tests

| Test | NSL Source | Description |
|------|-----------|-------------|
| `examples/m52_basic_fold.nsl` | `model TinyMLP { fn forward(self, x) = relu(self.w2 @ relu(self.w1 @ x + self.b1) + self.b2) }` | Compile with known 4x4 weights. Run inference, compare output to Python reference. |
| `examples/m52_sparse_matmul.nsl` | Model with one 70%-sparse weight matrix. | Verify sparse kernel emitted (check PTX for `sparse_matmul_kernel`). Output matches dense version. |
| `examples/m52_dead_weight.nsl` | Model with 20% near-zero weights. | Compile with `--dead-weight-threshold 1e-4`. Verify output unchanged. |
| `examples/m52_hash_verify.nsl` | Model compiled with `--weights a.safetensors`. | Verify binary contains SHA-256 hash. Tamper with weights, verify integrity check fails. |
| `examples/m52_no_fold.nsl` | Model with `@no_fold` on one layer. | Verify that layer is not constant-folded but others are. |
| `examples/m52_scale_fusion.nsl` | INT8 quantized model with per-channel scales. | Verify fused dequant+matmul emitted. Output matches separate dequant+matmul. |
| `examples/m52_weight_report.nsl` | `nsl check --weight-analysis` on a multi-layer model. | Verify report shows per-layer sparsity, folding opportunities, and size estimates. |

### Correctness Validation

The fundamental invariant: **a weight-aware compiled binary must produce identical output (within floating-point tolerance) to the same model compiled without `--weights`.** Every E2E test runs twice:

1. `nsl build model.nsl --weights weights.safetensors` (weight-aware)
2. `nsl build model.nsl` + load weights at runtime (generic)

Both binaries receive identical input. Outputs are compared with tolerance `|a - b| < 1e-5` for FP32, `|a - b| < 1e-3` for FP16.

---

## Section 12: Modified Files

### New Files

| File | Responsibility |
|------|----------------|
| `crates/nsl-codegen/src/weight_aware.rs` | WeightMap loader, sparsity analysis, constant folding, dead weight elimination, scale fusion, sparse matmul PTX generation |
| `crates/nsl-runtime/src/weight_map.rs` | RuntimeWeightMap FFI: init, get, verify, free |
| `crates/nsl-runtime/src/sparse.rs` | Sparse matmul FFI (CPU path), fused dequant+matmul |

### Modified Files

| File | Change |
|------|--------|
| `crates/nsl-codegen/src/compiler.rs` | `weight_map`, `sparsity_hints`, `folded_constants`, `scale_fusions`, `weight_integrity` fields; invoke `WeightAwarePass` before `MemoryPlanner` |
| `crates/nsl-codegen/src/expr.rs` | `compile_matmul()` checks `sparsity_hints` for sparse dispatch; `compile_add()` checks `folded_constants` for compile-time evaluation |
| `crates/nsl-codegen/src/lib.rs` | `mod weight_aware; mod sparse;` declarations; `weight_file` and `weight_config` in `CompileOptions` |
| `crates/nsl-semantic/src/checker.rs` | `check_no_fold_decorator()`, `check_sparse_decorator()`, `check_fold_scales_decorator()` |
| `crates/nsl-parser/src/parser.rs` | Parse `@no_fold`, `@sparse(threshold=N)`, `@fold_scales` decorator syntax |
| `crates/nsl-cli/src/main.rs` | `--weights`, `--dead-weight-threshold`, `--sparse-threshold`, `--no-constant-fold`, `--no-dead-weight`, `--no-sparse-codegen`, `nsl check --weight-analysis` subcommand |
| `crates/nsl-runtime/src/lib.rs` | `mod weight_map; mod sparse;` declarations |

---

## Section 13: Deliverables

1. Safetensors loader that reads weight tensors into a compile-time `WeightMap`
2. Sparsity analysis pass with per-weight near-zero fraction, CSR construction, and block-sparsity detection
3. Constant propagation through matmul, add, relu, and other elementwise ops when all operands are known
4. Dead weight elimination pass that prunes near-zero elements with configurable threshold
5. Sparsity-aware codegen that emits CSR-based sparse matmul PTX kernels for weight matrices above the sparsity threshold
6. Quantization scale fusion that folds per-channel/per-group scales into matmul instructions
7. Identity layer detection and elimination for near-identity weight matrices
8. SHA-256 weight hash embedding and runtime integrity verification
9. `nsl build --weights model.safetensors` CLI with all configuration flags
10. `nsl check --weight-analysis` diagnostic report
11. `@no_fold`, `@sparse(threshold=N)`, `@fold_scales` decorators for fine-grained control

## Out of Scope

- Dynamic weight loading (weights must be known at compile time — that is the entire point)
- Weight pruning optimization (finding which weights to prune — M52 consumes already-pruned checkpoints)
- Knowledge distillation or weight compression (orthogonal to compilation)
- Multi-file weight loading (single safetensors file per compilation; sharded models must be concatenated first)
- Weight format conversion (only `.safetensors` supported; `.bin`, `.pt`, `.ckpt` require external conversion)
- Training with weight-aware compilation (weight-aware compilation is inference-only; weights are frozen constants)
- Cross-layer weight sharing analysis (detecting that two layers have identical weights — trivial to add later)
- Profile-guided optimization using runtime traces (compile-time only, no feedback from execution)

## Success Criteria

1. A 7B parameter model compiled with `--weights` produces a binary that runs correctly and matches the non-weight-aware binary output within floating-point tolerance.
2. A model with 50%+ sparse weight matrices uses CSR sparse matmul kernels and achieves measurable inference speedup over dense kernels.
3. Dead weight elimination with threshold 1e-4 on a magnitude-pruned model produces a smaller binary with fewer GPU kernel instructions.
4. Quantization scale fusion eliminates separate dequantization kernel launches (verified by PTX inspection).
5. The SHA-256 integrity check correctly rejects weight files that do not match the compilation hash.
6. `nsl check --weight-analysis` produces a human-readable report with per-layer sparsity and optimization opportunities.
7. End-to-end latency: weight-aware compilation of a 7B model completes within 5 minutes on a workstation (weight loading + analysis + codegen).
