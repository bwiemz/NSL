//! M50: Sparse tensor runtime — storage, construction, and query FFI.

use std::sync::atomic::AtomicI64;

// NOTE: The semantic crate already has SparseFormat in types.rs (Coo, Csr, Csc, Bsr, Unknown).
// The runtime uses a u8 format ID instead of re-declaring the enum to avoid cross-crate
// collision. Conversion: Coo=0, Csr=1, Csc=2, Bsr=3. Unknown maps to None.

/// Runtime sparse format ID (matches semantic SparseFormat variant ordering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SparseFmtId {
    Coo = 0,
    Csr = 1,
    Csc = 2,
    Bsr = 3,
}

impl SparseFmtId {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Coo),
            1 => Some(Self::Csr),
            2 => Some(Self::Csc),
            3 => Some(Self::Bsr),
            _ => None, // handles semantic SparseFormat::Unknown → None
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Coo => "coo",
            Self::Csr => "csr",
            Self::Csc => "csc",
            Self::Bsr => "bsr",
        }
    }
}

/// Runtime representation of a sparse tensor.
/// Matches spec Section 3.1 with refcount and owns_data lifecycle fields.
#[repr(C)]
pub struct NslSparseTensor {
    pub format: u8,
    pub device: u8,
    pub dtype: u16,
    pub ndim: i32,
    pub nnz: i64,
    pub rows: i64,
    pub cols: i64,
    pub data: *mut u8,
    pub indices_0: *mut i64,
    pub indices_1: *mut i64,
    pub block_rows: i32,
    pub block_cols: i32,
    pub refcount: AtomicI64,  // lifecycle: same model as NslTensor
    pub owns_data: u8,  // 1 = heap-owned (free arrays on drop), 0 = borrowed
}

// SAFETY: Sparse tensors are allocated on the heap and not shared between threads
// without explicit synchronization (same as NslTensor).
unsafe impl Send for NslSparseTensor {}
unsafe impl Sync for NslSparseTensor {}

// ---------------------------------------------------------------------------
// FFI Functions (all i64 params per Cranelift convention)
// ---------------------------------------------------------------------------

/// Create a COO sparse tensor from coordinate arrays.
/// Copies the input arrays (takes ownership). Caller retains original arrays.
/// Returns pointer to NslSparseTensor (as i64), or 0 on error.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_coo(
    rows_ptr: i64, cols_ptr: i64, vals_ptr: i64,
    num_rows: i64, num_cols: i64, nnz: i64,
) -> i64 {
    if nnz < 0 || rows_ptr == 0 || cols_ptr == 0 || vals_ptr == 0 {
        return 0;
    }
    let n = nnz as usize;

    // Copy index arrays (sparse tensor owns its data)
    let mut row_copy = vec![0i64; n];
    let mut col_copy = vec![0i64; n];
    // The value buffer is an f64 array (8-aligned): every reader takes it as
    // `&[f64]`, which is undefined behaviour on a byte allocation (Miri:
    // "unaligned reference (required 8 byte alignment but found 1)").
    let mut val_copy = vec![0f64; n];
    unsafe {
        std::ptr::copy_nonoverlapping(rows_ptr as *const i64, row_copy.as_mut_ptr(), n);
        std::ptr::copy_nonoverlapping(cols_ptr as *const i64, col_copy.as_mut_ptr(), n);
        std::ptr::copy_nonoverlapping(vals_ptr as *const f64, val_copy.as_mut_ptr(), n);
    }

    let sparse = Box::new(NslSparseTensor {
        format: SparseFmtId::Coo as u8,
        device: 0,
        dtype: 0, // f64 default
        ndim: 2,
        nnz,
        rows: num_rows,
        cols: num_cols,
        data: Box::into_raw(val_copy.into_boxed_slice()) as *mut u8,
        indices_0: Box::into_raw(row_copy.into_boxed_slice()) as *mut i64,
        indices_1: Box::into_raw(col_copy.into_boxed_slice()) as *mut i64,
        block_rows: 0,
        block_cols: 0,
        refcount: AtomicI64::new(1),
        owns_data: 1,
    });
    Box::into_raw(sparse) as i64
}

/// Convert a dense tensor to COO sparse format.
/// dense_ptr: pointer to NslTensor (2D, f64 dtype=0)
/// format: 0=COO, 1=CSR
/// threshold_bits: f64 bits — values with |v| < threshold are treated as zero
/// Returns pointer to NslSparseTensor, or 0 on error.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_from_dense(dense_ptr: i64, format: i64, threshold_bits: i64) -> i64 {
    if dense_ptr == 0 { return 0; }
    let threshold = f64::from_bits(threshold_bits as u64);

    // Read dense tensor fields via raw pointer (NslTensor is pub(crate), same crate)
    let tensor = crate::tensor::NslTensor::from_ptr_ref(dense_ptr);
    if tensor.ndim != 2 { return 0; }
    let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 2) };
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    // Decode by tag: NSL-level creation builtins make f32 tensors, and a raw
    // f64 read of one walks twice its buffer.
    let data = tensor.as_f64_owned();
    // The dtype the sparse tensor records, and `nsl_sparse_to_dense` hands
    // back: its source's (f64 only for an f64 source; values are stored as
    // f64 either way). It used to record f64 whatever the source.
    let src_dtype: u16 = if tensor.dtype == 0 { 0 } else { 1 };

    // Scan for non-zeros
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = data[r * cols + c];
            if v.abs() >= threshold {
                row_indices.push(r as i64);
                col_indices.push(c as i64);
                values.push(v);
            }
        }
    }
    let nnz = values.len() as i64;

    let fmt = match format {
        1 => SparseFmtId::Csr,
        3 => SparseFmtId::Bsr,
        _ => SparseFmtId::Coo,
    };

    // BSR: block sparse row with default 16x16 blocks
    if fmt == SparseFmtId::Bsr {
        let br = 16usize; // block_rows
        let bc = 16usize; // block_cols
        let nblk_rows = rows.div_ceil(br);
        let nblk_cols = cols.div_ceil(bc);

        // Scan blocks: a block is nonzero if any element in it exceeds threshold
        let mut block_row_ptrs = vec![0i64; nblk_rows + 1];
        let mut block_col_indices: Vec<i64> = Vec::new();
        let mut block_values: Vec<f64> = Vec::new(); // stored as flat dense blocks

        for br_idx in 0..nblk_rows {
            let r_start = br_idx * br;
            let r_end = (r_start + br).min(rows);
            for bc_idx in 0..nblk_cols {
                let c_start = bc_idx * bc;
                let c_end = (c_start + bc).min(cols);
                // Check if block has any nonzero
                let mut has_nonzero = false;
                for r in r_start..r_end {
                    for c in c_start..c_end {
                        if data[r * cols + c].abs() >= threshold {
                            has_nonzero = true;
                            break;
                        }
                    }
                    if has_nonzero { break; }
                }
                if has_nonzero {
                    block_col_indices.push(bc_idx as i64);
                    // Store full block (padded with zeros if at boundary)
                    for r in 0..br {
                        for c in 0..bc {
                            let gr = r_start + r;
                            let gc = c_start + c;
                            let v = if gr < rows && gc < cols { data[gr * cols + gc] } else { 0.0 };
                            block_values.push(v);
                        }
                    }
                }
            }
            block_row_ptrs[br_idx + 1] = block_col_indices.len() as i64;
        }

        let num_blocks = block_col_indices.len();
        let val_bytes: Vec<f64> = block_values.to_vec();

        let sparse = Box::new(NslSparseTensor {
            format: SparseFmtId::Bsr as u8,
            device: 0, dtype: src_dtype, ndim: 2,
            nnz: num_blocks as i64, // nnz = number of nonzero blocks
            rows: rows as i64, cols: cols as i64,
            data: Box::into_raw(val_bytes.into_boxed_slice()) as *mut u8,
            indices_0: Box::into_raw(block_row_ptrs.into_boxed_slice()) as *mut i64,
            indices_1: Box::into_raw(block_col_indices.into_boxed_slice()) as *mut i64,
            block_rows: br as i32, block_cols: bc as i32,
            refcount: AtomicI64::new(1), owns_data: 1,
        });
        return Box::into_raw(sparse) as i64;
    }

    if fmt == SparseFmtId::Csr {
        // Convert COO to CSR: build row_ptr array
        let mut row_ptr = vec![0i64; rows + 1];
        for &r in &row_indices {
            row_ptr[r as usize + 1] += 1;
        }
        for i in 1..=rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        let val_bytes: Vec<f64> = values.to_vec();

        let sparse = Box::new(NslSparseTensor {
            format: SparseFmtId::Csr as u8,
            device: 0, dtype: src_dtype, ndim: 2,
            nnz,
            rows: rows as i64, cols: cols as i64,
            data: Box::into_raw(val_bytes.into_boxed_slice()) as *mut u8,
            indices_0: Box::into_raw(row_ptr.into_boxed_slice()) as *mut i64, // row_ptr (len = rows+1)
            indices_1: Box::into_raw(col_indices.into_boxed_slice()) as *mut i64, // col_indices (len = nnz)
            block_rows: 0, block_cols: 0,
            refcount: AtomicI64::new(1), owns_data: 1,
        });
        Box::into_raw(sparse) as i64
    } else {
        // COO format
        let val_bytes: Vec<f64> = values.to_vec();

        let sparse = Box::new(NslSparseTensor {
            format: SparseFmtId::Coo as u8,
            device: 0, dtype: src_dtype, ndim: 2,
            nnz,
            rows: rows as i64, cols: cols as i64,
            data: Box::into_raw(val_bytes.into_boxed_slice()) as *mut u8,
            indices_0: Box::into_raw(row_indices.into_boxed_slice()) as *mut i64,
            indices_1: Box::into_raw(col_indices.into_boxed_slice()) as *mut i64,
            block_rows: 0, block_cols: 0,
            refcount: AtomicI64::new(1), owns_data: 1,
        });
        Box::into_raw(sparse) as i64
    }
}

/// Convert a sparse tensor to dense NslTensor.
/// Returns pointer to a 2-D NslTensor in the sparse tensor's dtype, or 0 on
/// error (a null operand or an unknown format id).
///
/// Entries that share a coordinate ADD, as they do in every product
/// (`nsl_sparse_spmm`, `nsl_sparse_spmv`, the GPU kernels). This used to
/// keep the last duplicate instead, so one COO tensor stood for two
/// different matrices depending on which routine read it. A CSC operand
/// used to return null.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_to_dense(sparse_ptr: i64) -> i64 {
    if sparse_ptr == 0 { return 0; }
    let sparse = unsafe { &*(sparse_ptr as *const NslSparseTensor) };
    let rows = sparse.rows as usize;
    let cols = sparse.cols as usize;
    let Some(triples) = sparse_to_triples(sparse) else { return 0 };

    let mut dense = vec![0.0f64; rows * cols];
    for (r, c, v) in triples {
        if r >= 0 && c >= 0 && (r as usize) < rows && (c as usize) < cols {
            dense[r as usize * cols + c as usize] += v;
        }
    }

    let out =
        crate::tensor::creation::create_tensor_from_f64_data(&dense, &[rows as i64, cols as i64]);
    result_in_dtype(out, sparse.dtype)
}

/// Get the number of nonzero elements.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_nnz(sparse_ptr: i64) -> i64 {
    if sparse_ptr == 0 { return 0; }
    let sparse = unsafe { &*(sparse_ptr as *const NslSparseTensor) };
    sparse.nnz
}

/// Get the density (nnz / total_elements).
/// Returns f64 bits as i64.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_density(sparse_ptr: i64) -> i64 {
    if sparse_ptr == 0 { return f64::to_bits(0.0) as i64; }
    let sparse = unsafe { &*(sparse_ptr as *const NslSparseTensor) };
    let total = sparse.rows * sparse.cols;
    let density = if total > 0 { sparse.nnz as f64 / total as f64 } else { 0.0 };
    f64::to_bits(density) as i64
}

/// A sparse result (computed in f64, the values' storage type) in its
/// operand's dtype: f64 only for an f64 operand, f32 otherwise (C5: helpers
/// mint their operand's dtype). Every result goes through here, so the GPU
/// fallbacks hand back a tensor whose upload is a byte copy -- an upload does
/// not narrow (step 2b). A null result passes through.
fn result_in_dtype(result: i64, operand_dtype: u16) -> i64 {
    let target = if operand_dtype == 0 { 0 } else { 1 };
    if result == 0 || crate::tensor::NslTensor::from_ptr_ref(result).dtype == target {
        return result;
    }
    let converted = crate::tensor::precision_cast::convert_untaped(result, target);
    crate::tensor::nsl_tensor_free(result);
    converted
}

/// Sparse-dense matrix multiply: sparse(M×K) × dense(K×N) → dense(M×N).
/// sparse_ptr: NslSparseTensor (COO, CSR, CSC or BSR; a CSC operand used to
/// return null)
/// dense_ptr: NslTensor (2D, the sparse tensor's dtype)
/// Returns pointer to new NslTensor, or 0 on error.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_spmm(sparse_ptr: i64, dense_ptr: i64) -> i64 {
    if sparse_ptr == 0 || dense_ptr == 0 { return 0; }
    let sparse = unsafe { &*(sparse_ptr as *const NslSparseTensor) };
    let dense = crate::tensor::NslTensor::from_ptr_ref(dense_ptr);
    if sparse.dtype != dense.dtype {
        crate::fatal::mixed_dtypes("nsl_sparse_spmm", sparse.dtype, dense.dtype);
    }

    if dense.ndim != 2 { return 0; }
    let d_shape = unsafe { std::slice::from_raw_parts(dense.shape, 2) };
    let k = d_shape[0] as usize;
    let n = d_shape[1] as usize;
    let m = sparse.rows as usize;
    let nnz = sparse.nnz as usize;

    // Verify dimension compatibility: sparse cols == dense rows
    if sparse.cols as usize != k { return 0; }

    // While a tape is recording, the product has to be an operation the
    // tape knows, or the result is a constant to it and everything upstream
    // of the dense operand gets a gradient of exactly zero. None of the
    // sparse kernels records anything, so the sparse operand is expanded and
    // the product is the dense matmul, which records itself. (Decided here,
    // at run time: the compiler cannot know whether a tape is live.)
    if crate::autodiff::is_recording() {
        let expanded = nsl_sparse_to_dense(sparse_ptr);
        if expanded == 0 { return 0; }
        let placed = crate::tensor::nsl_tensor_to_device(expanded, dense.device as i64);
        crate::tensor::nsl_tensor_free(expanded);
        let out = crate::tensor::nsl_tensor_matmul(placed, dense_ptr, 0);
        crate::tensor::nsl_tensor_free(placed);
        return out;
    }

    // GPU dispatch: if dense tensor is on GPU, use GPU sparse kernels
    #[cfg(feature = "cuda")]
    if dense.device > 0 {
        return match SparseFmtId::from_u8(sparse.format) {
            Some(SparseFmtId::Csr) => crate::cuda::gpu_csr_spmm_f32_from_sparse(sparse, dense_ptr),
            Some(SparseFmtId::Coo) => crate::cuda::gpu_coo_spmm_f32(sparse, dense_ptr),
            Some(SparseFmtId::Bsr) => crate::cuda::gpu_bsr_spmm_f32(sparse, dense_ptr),
            _ => {
                // Unsupported format on GPU: fall through to CPU
                let cpu_dense = crate::tensor::nsl_tensor_to_device(dense_ptr, 0);
                let result = nsl_sparse_spmm(sparse_ptr, cpu_dense);
                let gpu_result = crate::tensor::nsl_tensor_to_device(result, dense.device as i64);
                crate::tensor::nsl_tensor_free(cpu_dense);
                crate::tensor::nsl_tensor_free(result);
                gpu_result
            }
        };
    }

    // Decode by tag: NSL-level creation builtins make f32 tensors, and a raw
    // f64 read of one walks twice its buffer.
    let d_data = dense.as_f64_owned();
    let s_vals = unsafe { std::slice::from_raw_parts(sparse.data as *const f64, nnz) };
    let mut output = vec![0.0f64; m * n];

    match SparseFmtId::from_u8(sparse.format) {
        Some(SparseFmtId::Coo) => {
            let row_idx = unsafe { std::slice::from_raw_parts(sparse.indices_0, nnz) };
            let col_idx = unsafe { std::slice::from_raw_parts(sparse.indices_1, nnz) };
            for i in 0..nnz {
                let r = row_idx[i] as usize;
                let c = col_idx[i] as usize;
                let v = s_vals[i];
                // output[r, :] += v * dense[c, :]
                for j in 0..n {
                    output[r * n + j] += v * d_data[c * n + j];
                }
            }
        }
        Some(SparseFmtId::Csr) => {
            let row_ptr = unsafe { std::slice::from_raw_parts(sparse.indices_0, m + 1) };
            let col_idx = unsafe { std::slice::from_raw_parts(sparse.indices_1, nnz) };
            for r in 0..m {
                let start = row_ptr[r] as usize;
                let end = row_ptr[r + 1] as usize;
                for idx in start..end {
                    let c = col_idx[idx] as usize;
                    let v = s_vals[idx];
                    for j in 0..n {
                        output[r * n + j] += v * d_data[c * n + j];
                    }
                }
            }
        }
        Some(SparseFmtId::Bsr) => {
            let br = sparse.block_rows as usize;
            let bc = sparse.block_cols as usize;
            if br == 0 || bc == 0 { return 0; }
            let nblk_rows = m.div_ceil(br);
            let num_blocks = nnz; // nnz = number of nonzero blocks
            let block_size = br * bc;
            let block_row_ptrs = unsafe { std::slice::from_raw_parts(sparse.indices_0, nblk_rows + 1) };
            let block_col_indices = unsafe { std::slice::from_raw_parts(sparse.indices_1, num_blocks) };
            let block_vals = unsafe { std::slice::from_raw_parts(sparse.data as *const f64, num_blocks * block_size) };

            for blk_r in 0..nblk_rows {
                let start = block_row_ptrs[blk_r] as usize;
                let end = block_row_ptrs[blk_r + 1] as usize;
                for (blk_idx, &blk_c_raw) in block_col_indices.iter().enumerate().take(end).skip(start) {
                    let blk_c = blk_c_raw as usize;
                    let val_base = blk_idx * block_size;
                    for r in 0..br {
                        let gr = blk_r * br + r;
                        if gr >= m { continue; }
                        for c_inner in 0..bc {
                            let gk = blk_c * bc + c_inner;
                            if gk >= k { continue; }
                            let a_val = block_vals[val_base + r * bc + c_inner];
                            if a_val == 0.0 { continue; }
                            for j in 0..n {
                                output[gr * n + j] += a_val * d_data[gk * n + j];
                            }
                        }
                    }
                }
            }
        }
        Some(SparseFmtId::Csc) => {
            // Column-compressed: walk the entries as (row, col, value).
            let Some(triples) = sparse_to_triples(sparse) else { return 0 };
            for (r, c, v) in triples {
                let (r, c) = (r as usize, c as usize);
                for j in 0..n {
                    output[r * n + j] += v * d_data[c * n + j];
                }
            }
        }
        None => return 0,
    }

    let out = crate::tensor::creation::create_tensor_from_f64_data(&output, &[m as i64, n as i64]);
    result_in_dtype(out, dense.dtype)
}

/// Free a sparse tensor and its owned arrays.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_free(sparse_ptr: i64) -> i64 {
    if sparse_ptr == 0 { return 0; }
    let sparse = unsafe { Box::from_raw(sparse_ptr as *mut NslSparseTensor) };
    if sparse.owns_data == 1 {
        let n = sparse.nnz as usize;

        // Format-aware array lengths:
        // - CSR: indices_0 = row_ptr (rows+1), indices_1 = col_indices (nnz), data = nnz * 8
        // - CSC: indices_0 = col_ptr (cols+1), indices_1 = row_indices (nnz), data = nnz * 8
        // - COO: indices_0 = row_indices (nnz), indices_1 = col_indices (nnz), data = nnz * 8
        // - BSR: indices_0 = block_row_ptrs (nblk_rows+1), indices_1 = block_col_indices (nnz),
        //        data = nnz * block_rows * block_cols * 8
        let (idx0_len, idx1_len, data_bytes) = if sparse.format == SparseFmtId::Bsr as u8 {
            let br = sparse.block_rows as usize;
            let bc = sparse.block_cols as usize;
            let nblk_rows = if br > 0 { (sparse.rows as usize).div_ceil(br) } else { 0 };
            (nblk_rows + 1, n, n * br * bc * 8)
        } else if sparse.format == SparseFmtId::Csr as u8 {
            (sparse.rows as usize + 1, n, n * 8)
        } else if sparse.format == SparseFmtId::Csc as u8 {
            (sparse.cols as usize + 1, n, n * 8)
        } else {
            (n, n, n * 8)
        };

        if !sparse.indices_0.is_null() && idx0_len > 0 {
            let _ = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(sparse.indices_0, idx0_len)) };
        }
        if !sparse.indices_1.is_null() && idx1_len > 0 {
            let _ = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(sparse.indices_1, idx1_len)) };
        }
        if !sparse.data.is_null() && data_bytes > 0 {
            // Allocated as `Box<[f64]>` (see the value-buffer note in
            // `nsl_sparse_coo`); free with the same layout.
            let _ = unsafe {
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(sparse.data as *mut f64, data_bytes / 8))
            };
        }
    }
    // Box::from_raw above already freed the NslSparseTensor struct
    0
}

// ---------------------------------------------------------------------------
// COO ↔ CSR/CSC conversion
// ---------------------------------------------------------------------------

/// Convert COO sparse tensor to CSR format.
/// Returns new CSR NslSparseTensor pointer, or 0 on error.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_coo_to_csr(coo_ptr: i64) -> i64 {
    if coo_ptr == 0 { return 0; }
    let coo = unsafe { &*(coo_ptr as *const NslSparseTensor) };
    if coo.format != SparseFmtId::Coo as u8 { return 0; }
    let nnz = coo.nnz as usize;
    let rows = coo.rows as usize;

    if nnz == 0 {
        // Empty CSR: row_ptr is all zeros
        let row_ptr = vec![0i64; rows + 1];
        let sparse = Box::new(NslSparseTensor {
            format: SparseFmtId::Csr as u8,
            device: coo.device, dtype: coo.dtype, ndim: 2,
            nnz: 0, rows: coo.rows, cols: coo.cols,
            data: std::ptr::null_mut(),
            indices_0: Box::into_raw(row_ptr.into_boxed_slice()) as *mut i64,
            indices_1: std::ptr::null_mut(),
            block_rows: 0, block_cols: 0,
            refcount: AtomicI64::new(1), owns_data: 1,
        });
        return Box::into_raw(sparse) as i64;
    }

    let row_idx = unsafe { std::slice::from_raw_parts(coo.indices_0, nnz) };
    let col_idx = unsafe { std::slice::from_raw_parts(coo.indices_1, nnz) };
    let vals = unsafe { std::slice::from_raw_parts(coo.data as *const f64, nnz) };

    // Sort by (row, col) for canonical CSR ordering
    let mut order: Vec<usize> = (0..nnz).collect();
    order.sort_by_key(|&i| (row_idx[i], col_idx[i]));

    let mut sorted_cols = vec![0i64; nnz];
    let mut sorted_vals = vec![0.0f64; nnz];
    for (new_i, &old_i) in order.iter().enumerate() {
        sorted_cols[new_i] = col_idx[old_i];
        sorted_vals[new_i] = vals[old_i];
    }

    // Build row_ptr
    let mut row_ptr = vec![0i64; rows + 1];
    for &old_i in &order {
        row_ptr[row_idx[old_i] as usize + 1] += 1;
    }
    for i in 1..=rows {
        row_ptr[i] += row_ptr[i - 1];
    }

    let val_bytes: Vec<f64> = sorted_vals.to_vec();
    let sparse = Box::new(NslSparseTensor {
        format: SparseFmtId::Csr as u8,
        device: coo.device, dtype: coo.dtype, ndim: 2,
        nnz: coo.nnz, rows: coo.rows, cols: coo.cols,
        data: Box::into_raw(val_bytes.into_boxed_slice()) as *mut u8,
        indices_0: Box::into_raw(row_ptr.into_boxed_slice()) as *mut i64,
        indices_1: Box::into_raw(sorted_cols.into_boxed_slice()) as *mut i64,
        block_rows: 0, block_cols: 0,
        refcount: AtomicI64::new(1), owns_data: 1,
    });
    Box::into_raw(sparse) as i64
}

/// Convert COO sparse tensor to CSC format.
/// Returns new CSC NslSparseTensor pointer, or 0 on error.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_coo_to_csc(coo_ptr: i64) -> i64 {
    if coo_ptr == 0 { return 0; }
    let coo = unsafe { &*(coo_ptr as *const NslSparseTensor) };
    if coo.format != SparseFmtId::Coo as u8 { return 0; }
    let nnz = coo.nnz as usize;
    let cols = coo.cols as usize;

    if nnz == 0 {
        let col_ptr = vec![0i64; cols + 1];
        let sparse = Box::new(NslSparseTensor {
            format: SparseFmtId::Csc as u8,
            device: coo.device, dtype: coo.dtype, ndim: 2,
            nnz: 0, rows: coo.rows, cols: coo.cols,
            data: std::ptr::null_mut(),
            indices_0: Box::into_raw(col_ptr.into_boxed_slice()) as *mut i64,
            indices_1: std::ptr::null_mut(),
            block_rows: 0, block_cols: 0,
            refcount: AtomicI64::new(1), owns_data: 1,
        });
        return Box::into_raw(sparse) as i64;
    }

    let row_idx = unsafe { std::slice::from_raw_parts(coo.indices_0, nnz) };
    let col_idx = unsafe { std::slice::from_raw_parts(coo.indices_1, nnz) };
    let vals = unsafe { std::slice::from_raw_parts(coo.data as *const f64, nnz) };

    // Sort by (col, row) for CSC ordering
    let mut order: Vec<usize> = (0..nnz).collect();
    order.sort_by_key(|&i| (col_idx[i], row_idx[i]));

    let mut sorted_rows = vec![0i64; nnz];
    let mut sorted_vals = vec![0.0f64; nnz];
    for (new_i, &old_i) in order.iter().enumerate() {
        sorted_rows[new_i] = row_idx[old_i];
        sorted_vals[new_i] = vals[old_i];
    }

    // Build col_ptr
    let mut col_ptr = vec![0i64; cols + 1];
    for &old_i in &order {
        col_ptr[col_idx[old_i] as usize + 1] += 1;
    }
    for i in 1..=cols {
        col_ptr[i] += col_ptr[i - 1];
    }

    let val_bytes: Vec<f64> = sorted_vals.to_vec();
    let sparse = Box::new(NslSparseTensor {
        format: SparseFmtId::Csc as u8,
        device: coo.device, dtype: coo.dtype, ndim: 2,
        nnz: coo.nnz, rows: coo.rows, cols: coo.cols,
        data: Box::into_raw(val_bytes.into_boxed_slice()) as *mut u8,
        indices_0: Box::into_raw(col_ptr.into_boxed_slice()) as *mut i64, // col_ptr
        indices_1: Box::into_raw(sorted_rows.into_boxed_slice()) as *mut i64, // row_indices
        block_rows: 0, block_cols: 0,
        refcount: AtomicI64::new(1), owns_data: 1,
    });
    Box::into_raw(sparse) as i64
}

// ---------------------------------------------------------------------------
// CSR ↔ CSC direct conversion (without going through COO)
// ---------------------------------------------------------------------------

/// Convert CSR sparse tensor to CSC format.
/// CSR: row_ptr[rows+1], col_indices[nnz], values[nnz]
/// CSC: col_ptr[cols+1], row_indices[nnz], values[nnz] (reordered)
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_csr_to_csc(csr_ptr: i64) -> i64 {
    if csr_ptr == 0 { return 0; }
    let csr = unsafe { &*(csr_ptr as *const NslSparseTensor) };
    if csr.format != SparseFmtId::Csr as u8 { return 0; }
    let rows = csr.rows as usize;
    let cols = csr.cols as usize;
    let nnz = csr.nnz as usize;

    if nnz == 0 {
        let col_ptr = vec![0i64; cols + 1];
        let sparse = Box::new(NslSparseTensor {
            format: SparseFmtId::Csc as u8,
            device: csr.device, dtype: csr.dtype, ndim: 2,
            nnz: 0, rows: csr.rows, cols: csr.cols,
            data: std::ptr::null_mut(),
            indices_0: Box::into_raw(col_ptr.into_boxed_slice()) as *mut i64,
            indices_1: std::ptr::null_mut(),
            block_rows: 0, block_cols: 0,
            refcount: AtomicI64::new(1), owns_data: 1,
        });
        return Box::into_raw(sparse) as i64;
    }

    let row_ptr = unsafe { std::slice::from_raw_parts(csr.indices_0, rows + 1) };
    let col_idx = unsafe { std::slice::from_raw_parts(csr.indices_1, nnz) };
    let vals = unsafe { std::slice::from_raw_parts(csr.data as *const f64, nnz) };

    // Count entries per column
    let mut col_counts = vec![0usize; cols];
    for &c in col_idx {
        col_counts[c as usize] += 1;
    }

    // Build col_ptr (prefix sum)
    let mut col_ptr = vec![0i64; cols + 1];
    for i in 0..cols {
        col_ptr[i + 1] = col_ptr[i] + col_counts[i] as i64;
    }

    // Fill row_indices and values in CSC order
    let mut out_row_indices = vec![0i64; nnz];
    let mut out_vals = vec![0.0f64; nnz];
    let mut write_pos = vec![0usize; cols]; // current write position per column

    for r in 0..rows {
        let start = row_ptr[r] as usize;
        let end = row_ptr[r + 1] as usize;
        for idx in start..end {
            let c = col_idx[idx] as usize;
            let pos = col_ptr[c] as usize + write_pos[c];
            out_row_indices[pos] = r as i64;
            out_vals[pos] = vals[idx];
            write_pos[c] += 1;
        }
    }

    let val_bytes: Vec<f64> = out_vals;
    let sparse = Box::new(NslSparseTensor {
        format: SparseFmtId::Csc as u8,
        device: csr.device, dtype: csr.dtype, ndim: 2,
        nnz: csr.nnz, rows: csr.rows, cols: csr.cols,
        data: Box::into_raw(val_bytes.into_boxed_slice()) as *mut u8,
        indices_0: Box::into_raw(col_ptr.into_boxed_slice()) as *mut i64,
        indices_1: Box::into_raw(out_row_indices.into_boxed_slice()) as *mut i64,
        block_rows: 0, block_cols: 0,
        refcount: AtomicI64::new(1), owns_data: 1,
    });
    Box::into_raw(sparse) as i64
}

/// Convert CSC sparse tensor to CSR format.
/// CSC: col_ptr[cols+1], row_indices[nnz], values[nnz]
/// CSR: row_ptr[rows+1], col_indices[nnz], values[nnz] (reordered)
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_csc_to_csr(csc_ptr: i64) -> i64 {
    if csc_ptr == 0 { return 0; }
    let csc = unsafe { &*(csc_ptr as *const NslSparseTensor) };
    if csc.format != SparseFmtId::Csc as u8 { return 0; }
    let rows = csc.rows as usize;
    let cols = csc.cols as usize;
    let nnz = csc.nnz as usize;

    if nnz == 0 {
        let row_ptr = vec![0i64; rows + 1];
        let sparse = Box::new(NslSparseTensor {
            format: SparseFmtId::Csr as u8,
            device: csc.device, dtype: csc.dtype, ndim: 2,
            nnz: 0, rows: csc.rows, cols: csc.cols,
            data: std::ptr::null_mut(),
            indices_0: Box::into_raw(row_ptr.into_boxed_slice()) as *mut i64,
            indices_1: std::ptr::null_mut(),
            block_rows: 0, block_cols: 0,
            refcount: AtomicI64::new(1), owns_data: 1,
        });
        return Box::into_raw(sparse) as i64;
    }

    let col_ptr = unsafe { std::slice::from_raw_parts(csc.indices_0, cols + 1) };
    let row_idx = unsafe { std::slice::from_raw_parts(csc.indices_1, nnz) };
    let vals = unsafe { std::slice::from_raw_parts(csc.data as *const f64, nnz) };

    // Count entries per row
    let mut row_counts = vec![0usize; rows];
    for &r in row_idx {
        row_counts[r as usize] += 1;
    }

    // Build row_ptr (prefix sum)
    let mut row_ptr = vec![0i64; rows + 1];
    for i in 0..rows {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i] as i64;
    }

    // Fill col_indices and values in CSR order
    let mut out_col_indices = vec![0i64; nnz];
    let mut out_vals = vec![0.0f64; nnz];
    let mut write_pos = vec![0usize; rows];

    for c in 0..cols {
        let start = col_ptr[c] as usize;
        let end = col_ptr[c + 1] as usize;
        for idx in start..end {
            let r = row_idx[idx] as usize;
            let pos = row_ptr[r] as usize + write_pos[r];
            out_col_indices[pos] = c as i64;
            out_vals[pos] = vals[idx];
            write_pos[r] += 1;
        }
    }

    let val_bytes: Vec<f64> = out_vals;
    let sparse = Box::new(NslSparseTensor {
        format: SparseFmtId::Csr as u8,
        device: csc.device, dtype: csc.dtype, ndim: 2,
        nnz: csc.nnz, rows: csc.rows, cols: csc.cols,
        data: Box::into_raw(val_bytes.into_boxed_slice()) as *mut u8,
        indices_0: Box::into_raw(row_ptr.into_boxed_slice()) as *mut i64,
        indices_1: Box::into_raw(out_col_indices.into_boxed_slice()) as *mut i64,
        block_rows: 0, block_cols: 0,
        refcount: AtomicI64::new(1), owns_data: 1,
    });
    Box::into_raw(sparse) as i64
}

/// Convert CSR to COO format.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_csr_to_coo(csr_ptr: i64) -> i64 {
    if csr_ptr == 0 { return 0; }
    let csr = unsafe { &*(csr_ptr as *const NslSparseTensor) };
    if csr.format != SparseFmtId::Csr as u8 { return 0; }
    let rows = csr.rows as usize;
    let nnz = csr.nnz as usize;

    let row_ptr = unsafe { std::slice::from_raw_parts(csr.indices_0, rows + 1) };
    let col_idx = unsafe { std::slice::from_raw_parts(csr.indices_1, nnz) };
    let vals = unsafe { std::slice::from_raw_parts(csr.data as *const f64, nnz) };

    let mut row_indices = vec![0i64; nnz];
    for r in 0..rows {
        let start = row_ptr[r] as usize;
        let end = row_ptr[r + 1] as usize;
        for slot in row_indices.iter_mut().take(end).skip(start) {
            *slot = r as i64;
        }
    }

    let val_bytes: Vec<f64> = vals.to_vec();
    let sparse = Box::new(NslSparseTensor {
        format: SparseFmtId::Coo as u8,
        device: csr.device, dtype: csr.dtype, ndim: 2,
        nnz: csr.nnz, rows: csr.rows, cols: csr.cols,
        data: Box::into_raw(val_bytes.into_boxed_slice()) as *mut u8,
        indices_0: Box::into_raw(row_indices.into_boxed_slice()) as *mut i64,
        indices_1: Box::into_raw(col_idx.to_vec().into_boxed_slice()) as *mut i64,
        block_rows: 0, block_cols: 0,
        refcount: AtomicI64::new(1), owns_data: 1,
    });
    Box::into_raw(sparse) as i64
}

/// Convert CSC to COO format.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_csc_to_coo(csc_ptr: i64) -> i64 {
    if csc_ptr == 0 { return 0; }
    let csc = unsafe { &*(csc_ptr as *const NslSparseTensor) };
    if csc.format != SparseFmtId::Csc as u8 { return 0; }
    let cols = csc.cols as usize;
    let nnz = csc.nnz as usize;

    let col_ptr = unsafe { std::slice::from_raw_parts(csc.indices_0, cols + 1) };
    let row_idx = unsafe { std::slice::from_raw_parts(csc.indices_1, nnz) };
    let vals = unsafe { std::slice::from_raw_parts(csc.data as *const f64, nnz) };

    let mut col_indices = vec![0i64; nnz];
    for c in 0..cols {
        let start = col_ptr[c] as usize;
        let end = col_ptr[c + 1] as usize;
        for slot in col_indices.iter_mut().take(end).skip(start) {
            *slot = c as i64;
        }
    }

    let val_bytes: Vec<f64> = vals.to_vec();
    let sparse = Box::new(NslSparseTensor {
        format: SparseFmtId::Coo as u8,
        device: csc.device, dtype: csc.dtype, ndim: 2,
        nnz: csc.nnz, rows: csc.rows, cols: csc.cols,
        data: Box::into_raw(val_bytes.into_boxed_slice()) as *mut u8,
        indices_0: Box::into_raw(row_idx.to_vec().into_boxed_slice()) as *mut i64,
        indices_1: Box::into_raw(col_indices.into_boxed_slice()) as *mut i64,
        block_rows: 0, block_cols: 0,
        refcount: AtomicI64::new(1), owns_data: 1,
    });
    Box::into_raw(sparse) as i64
}

// ---------------------------------------------------------------------------
// SpMV: sparse matrix × dense vector → dense vector
// ---------------------------------------------------------------------------

/// SpMV: sparse matrix [M,K] (any format) × dense vector [K] → dense vector [M].
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_spmv(sparse_ptr: i64, vec_ptr: i64) -> i64 {
    if sparse_ptr == 0 || vec_ptr == 0 { return 0; }
    let sparse = unsafe { &*(sparse_ptr as *const NslSparseTensor) };
    let vec_t = crate::tensor::NslTensor::from_ptr_ref(vec_ptr);
    if sparse.dtype != vec_t.dtype {
        crate::fatal::mixed_dtypes("nsl_sparse_spmv", sparse.dtype, vec_t.dtype);
    }

    let m = sparse.rows as usize;
    let k = sparse.cols as usize;
    let nnz = sparse.nnz as usize;

    // Vector must be 1D or 2D with second dim = 1
    let vec_len = vec_t.len as usize;
    if vec_len != k { return 0; }

    // GPU dispatch
    #[cfg(feature = "cuda")]
    if vec_t.device > 0 {
        return match SparseFmtId::from_u8(sparse.format) {
            Some(SparseFmtId::Csr) => crate::cuda::gpu_csr_spmv_f32(sparse, vec_ptr),
            Some(SparseFmtId::Coo) => crate::cuda::gpu_coo_spmv_f32(sparse, vec_ptr),
            _ => {
                // Unsupported: fall back to CPU
                let cpu_vec = crate::tensor::nsl_tensor_to_device(vec_ptr, 0);
                let result = nsl_sparse_spmv(sparse_ptr, cpu_vec);
                let gpu_result = crate::tensor::nsl_tensor_to_device(result, vec_t.device as i64);
                crate::tensor::nsl_tensor_free(cpu_vec);
                crate::tensor::nsl_tensor_free(result);
                gpu_result
            }
        };
    }

    // Decode by tag. The GPU fallback above hands this path a DOWNLOADED
    // vector for BSR, which keeps its f32 tag (C5 step 2a); a raw f64 read
    // would walk twice its buffer, as it already did for any f32 CPU operand.
    let x = vec_t.as_f64_owned();
    let s_vals = unsafe { std::slice::from_raw_parts(sparse.data as *const f64, nnz) };
    let mut output = vec![0.0f64; m];

    match SparseFmtId::from_u8(sparse.format) {
        Some(SparseFmtId::Csr) => {
            let row_ptr = unsafe { std::slice::from_raw_parts(sparse.indices_0, m + 1) };
            let col_idx = unsafe { std::slice::from_raw_parts(sparse.indices_1, nnz) };
            for r in 0..m {
                let start = row_ptr[r] as usize;
                let end = row_ptr[r + 1] as usize;
                let mut sum = 0.0;
                for j in start..end {
                    sum += s_vals[j] * x[col_idx[j] as usize];
                }
                output[r] = sum;
            }
        }
        Some(SparseFmtId::Coo) => {
            let row_idx = unsafe { std::slice::from_raw_parts(sparse.indices_0, nnz) };
            let col_idx = unsafe { std::slice::from_raw_parts(sparse.indices_1, nnz) };
            for i in 0..nnz {
                output[row_idx[i] as usize] += s_vals[i] * x[col_idx[i] as usize];
            }
        }
        Some(SparseFmtId::Bsr) => {
            let br = sparse.block_rows as usize;
            let bc = sparse.block_cols as usize;
            if br == 0 || bc == 0 { return 0; }
            let nblk_rows = m.div_ceil(br);
            let num_blocks = nnz;
            let block_size = br * bc;
            let block_row_ptrs = unsafe { std::slice::from_raw_parts(sparse.indices_0, nblk_rows + 1) };
            let block_col_indices = unsafe { std::slice::from_raw_parts(sparse.indices_1, num_blocks) };
            let block_vals = unsafe { std::slice::from_raw_parts(sparse.data as *const f64, num_blocks * block_size) };

            for blk_r in 0..nblk_rows {
                let start = block_row_ptrs[blk_r] as usize;
                let end = block_row_ptrs[blk_r + 1] as usize;
                for (blk_idx, &blk_c_raw) in block_col_indices.iter().enumerate().take(end).skip(start) {
                    let blk_c = blk_c_raw as usize;
                    let val_base = blk_idx * block_size;
                    for r in 0..br {
                        let gr = blk_r * br + r;
                        if gr >= m { continue; }
                        let mut sum = 0.0;
                        for c_inner in 0..bc {
                            let gk = blk_c * bc + c_inner;
                            if gk >= k { continue; }
                            sum += block_vals[val_base + r * bc + c_inner] * x[gk];
                        }
                        output[gr] += sum;
                    }
                }
            }
        }
        Some(SparseFmtId::Csc) => {
            let Some(triples) = sparse_to_triples(sparse) else { return 0 };
            for (r, c, v) in triples {
                output[r as usize] += v * x[c as usize];
            }
        }
        None => return 0,
    }

    let out = crate::tensor::creation::create_tensor_from_f64_data(&output, &[m as i64]);
    result_in_dtype(out, vec_t.dtype)
}

// ---------------------------------------------------------------------------
// Sparse + Sparse (union merge lattice)
// ---------------------------------------------------------------------------

/// Element-wise sparse add: sparse [M,N] + sparse [M,N] → sparse [M,N] (COO output).
/// Uses union merge: output has non-zeros where either input has non-zeros.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_add(a_ptr: i64, b_ptr: i64) -> i64 {
    if a_ptr == 0 || b_ptr == 0 { return 0; }
    let a = unsafe { &*(a_ptr as *const NslSparseTensor) };
    let b = unsafe { &*(b_ptr as *const NslSparseTensor) };
    // The result carries the operands' dtype; it used to be tagged f64 always,
    // so an f32 sum densified to f64 beside f32 tensors (C5 step 4).
    if a.dtype != b.dtype {
        crate::fatal::mixed_dtypes("nsl_sparse_add", a.dtype, b.dtype);
    }
    if a.rows != b.rows || a.cols != b.cols { return 0; }

    let rows = a.rows;
    let cols = a.cols;

    // Convert both to (row, col, val) triples. Any format: a CSC or BSR
    // operand used to come back as an empty list and count as all zeros.
    let (Some(a_triples), Some(b_triples)) = (sparse_to_triples(a), sparse_to_triples(b)) else {
        return 0;
    };

    // Merge using workspace: dense row accumulator
    let num_cols = cols as usize;
    let num_rows = rows as usize;
    let mut out_rows = Vec::new();
    let mut out_cols = Vec::new();
    let mut out_vals = Vec::new();

    let mut workspace = vec![0.0f64; num_cols];
    let mut has_value = vec![false; num_cols];

    // Group triples by row
    let mut a_by_row: Vec<Vec<(i64, f64)>> = vec![Vec::new(); num_rows];
    for &(r, c, v) in &a_triples {
        a_by_row[r as usize].push((c, v));
    }
    let mut b_by_row: Vec<Vec<(i64, f64)>> = vec![Vec::new(); num_rows];
    for &(r, c, v) in &b_triples {
        b_by_row[r as usize].push((c, v));
    }

    for r in 0..num_rows {
        // Fill workspace from A
        for &(c, v) in &a_by_row[r] {
            let ci = c as usize;
            workspace[ci] += v;
            has_value[ci] = true;
        }
        // Fill workspace from B
        for &(c, v) in &b_by_row[r] {
            let ci = c as usize;
            workspace[ci] += v;
            has_value[ci] = true;
        }
        // Compact to output
        for c in 0..num_cols {
            if has_value[c] && workspace[c] != 0.0 {
                out_rows.push(r as i64);
                out_cols.push(c as i64);
                out_vals.push(workspace[c]);
            }
            workspace[c] = 0.0;
            has_value[c] = false;
        }
    }

    build_coo_from_triples(rows, cols, &out_rows, &out_cols, &out_vals, a.dtype)
}

/// Element-wise sparse mul: sparse [M,N] * sparse [M,N] → sparse [M,N] (COO output).
/// Uses intersection merge: output has non-zeros only where both have non-zeros.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_sparse_mul(a_ptr: i64, b_ptr: i64) -> i64 {
    if a_ptr == 0 || b_ptr == 0 { return 0; }
    let a = unsafe { &*(a_ptr as *const NslSparseTensor) };
    let b = unsafe { &*(b_ptr as *const NslSparseTensor) };
    // The result carries the operands' dtype; it used to be tagged f64 always,
    // so an f32 sum densified to f64 beside f32 tensors (C5 step 4).
    if a.dtype != b.dtype {
        crate::fatal::mixed_dtypes("nsl_sparse_mul", a.dtype, b.dtype);
    }
    if a.rows != b.rows || a.cols != b.cols { return 0; }

    let rows = a.rows;
    let cols = a.cols;
    let num_rows = rows as usize;
    let num_cols = cols as usize;

    // Entries that share a coordinate add, so each operand is coalesced
    // before the product: (a1 + a2) * (b1 + b2), not a sum of whichever
    // pairs happened to be stored last.
    let (Some(a_triples), Some(b_triples)) = (sparse_to_triples(a), sparse_to_triples(b)) else {
        return 0;
    };
    let a_triples = coalesce(a_triples);
    let b_triples = coalesce(b_triples);

    // Group B by row for quick lookup
    let mut b_by_row: Vec<Vec<(i64, f64)>> = vec![Vec::new(); num_rows];
    for &(r, c, v) in &b_triples {
        b_by_row[r as usize].push((c, v));
    }

    let mut out_rows = Vec::new();
    let mut out_cols = Vec::new();
    let mut out_vals = Vec::new();

    // For intersection: use workspace to find common (row, col) pairs
    let mut b_workspace = vec![0.0f64; num_cols];
    let mut b_present = vec![false; num_cols];

    // Group A by row
    let mut a_by_row: Vec<Vec<(i64, f64)>> = vec![Vec::new(); num_rows];
    for &(r, c, v) in &a_triples {
        a_by_row[r as usize].push((c, v));
    }

    for r in 0..num_rows {
        // Mark B entries
        for &(c, v) in &b_by_row[r] {
            let ci = c as usize;
            b_workspace[ci] = v;
            b_present[ci] = true;
        }
        // Intersect with A
        for &(c, a_val) in &a_by_row[r] {
            let ci = c as usize;
            if b_present[ci] {
                let product = a_val * b_workspace[ci];
                if product != 0.0 {
                    out_rows.push(r as i64);
                    out_cols.push(c);
                    out_vals.push(product);
                }
            }
        }
        // Clear workspace
        for &(c, _) in &b_by_row[r] {
            b_workspace[c as usize] = 0.0;
            b_present[c as usize] = false;
        }
    }

    build_coo_from_triples(rows, cols, &out_rows, &out_cols, &out_vals, a.dtype)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Every stored entry of a sparse tensor as a (row, col, value) triple, in
/// any of the four formats; `None` for an unknown format id. Entries that
/// share a coordinate stay separate (see [`coalesce`]); a BSR block
/// contributes its nonzero elements.
///
/// CSC and BSR used to yield an empty list, which made `nsl_sparse_add` and
/// `nsl_sparse_mul` treat such an operand as a matrix of zeros.
fn sparse_to_triples(s: &NslSparseTensor) -> Option<Vec<(i64, i64, f64)>> {
    let fmt = SparseFmtId::from_u8(s.format)?;
    let nnz = s.nnz as usize;
    if nnz == 0 { return Some(Vec::new()); }

    Some(match fmt {
        SparseFmtId::Coo => {
            let vals = unsafe { std::slice::from_raw_parts(s.data as *const f64, nnz) };
            let rows = unsafe { std::slice::from_raw_parts(s.indices_0, nnz) };
            let cols = unsafe { std::slice::from_raw_parts(s.indices_1, nnz) };
            (0..nnz).map(|i| (rows[i], cols[i], vals[i])).collect()
        }
        SparseFmtId::Csr => {
            let vals = unsafe { std::slice::from_raw_parts(s.data as *const f64, nnz) };
            let nrows = s.rows as usize;
            let row_ptr = unsafe { std::slice::from_raw_parts(s.indices_0, nrows + 1) };
            let col_idx = unsafe { std::slice::from_raw_parts(s.indices_1, nnz) };
            let mut triples = Vec::with_capacity(nnz);
            for r in 0..nrows {
                let start = row_ptr[r] as usize;
                let end = row_ptr[r + 1] as usize;
                for j in start..end {
                    triples.push((r as i64, col_idx[j], vals[j]));
                }
            }
            triples
        }
        SparseFmtId::Csc => {
            // indices_0 = col_ptr [cols + 1], indices_1 = row index per entry.
            let vals = unsafe { std::slice::from_raw_parts(s.data as *const f64, nnz) };
            let ncols = s.cols as usize;
            let col_ptr = unsafe { std::slice::from_raw_parts(s.indices_0, ncols + 1) };
            let row_idx = unsafe { std::slice::from_raw_parts(s.indices_1, nnz) };
            let mut triples = Vec::with_capacity(nnz);
            for c in 0..ncols {
                let start = col_ptr[c] as usize;
                let end = col_ptr[c + 1] as usize;
                for j in start..end {
                    triples.push((row_idx[j], c as i64, vals[j]));
                }
            }
            triples
        }
        SparseFmtId::Bsr => {
            // `nnz` counts blocks; each is a dense block_rows x block_cols
            // tile, the last ones overhanging the matrix edge.
            let br = s.block_rows as usize;
            let bc = s.block_cols as usize;
            if br == 0 || bc == 0 { return None; }
            let (rows, cols) = (s.rows as usize, s.cols as usize);
            let nblk_rows = rows.div_ceil(br);
            let block_size = br * bc;
            let block_row_ptrs = unsafe { std::slice::from_raw_parts(s.indices_0, nblk_rows + 1) };
            let block_cols = unsafe { std::slice::from_raw_parts(s.indices_1, nnz) };
            let block_vals = unsafe { std::slice::from_raw_parts(s.data as *const f64, nnz * block_size) };
            let mut triples = Vec::new();
            for blk_r in 0..nblk_rows {
                let start = block_row_ptrs[blk_r] as usize;
                let end = block_row_ptrs[blk_r + 1] as usize;
                for blk_idx in start..end {
                    let blk_c = block_cols[blk_idx] as usize;
                    for r in 0..br {
                        for c in 0..bc {
                            let (gr, gc) = (blk_r * br + r, blk_c * bc + c);
                            let v = block_vals[blk_idx * block_size + r * bc + c];
                            if gr < rows && gc < cols && v != 0.0 {
                                triples.push((gr as i64, gc as i64, v));
                            }
                        }
                    }
                }
            }
            triples
        }
    })
}

/// Sum the entries that share a coordinate, leaving one triple per (row,
/// col) in row-major order.
fn coalesce(mut triples: Vec<(i64, i64, f64)>) -> Vec<(i64, i64, f64)> {
    triples.sort_by_key(|&(r, c, _)| (r, c));
    let mut out: Vec<(i64, i64, f64)> = Vec::with_capacity(triples.len());
    for (r, c, v) in triples {
        match out.last_mut() {
            Some(last) if last.0 == r && last.1 == c => last.2 += v,
            _ => out.push((r, c, v)),
        }
    }
    out
}

/// Build a COO NslSparseTensor from triples. `dtype` is the tag results take
/// (`result_in_dtype`); the values are stored f64 whatever it is.
fn build_coo_from_triples(rows: i64, cols: i64, r: &[i64], c: &[i64], v: &[f64], dtype: u16) -> i64 {
    let nnz = v.len();
    let val_bytes: Vec<f64> = v.to_vec();
    let sparse = Box::new(NslSparseTensor {
        format: SparseFmtId::Coo as u8,
        device: 0, dtype, ndim: 2,
        nnz: nnz as i64, rows, cols,
        data: if nnz > 0 { Box::into_raw(val_bytes.into_boxed_slice()) as *mut u8 } else { std::ptr::null_mut() },
        indices_0: if nnz > 0 { Box::into_raw(r.to_vec().into_boxed_slice()) as *mut i64 } else { std::ptr::null_mut() },
        indices_1: if nnz > 0 { Box::into_raw(c.to_vec().into_boxed_slice()) as *mut i64 } else { std::ptr::null_mut() },
        block_rows: 0, block_cols: 0,
        refcount: AtomicI64::new(1), owns_data: 1,
    });
    Box::into_raw(sparse) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_format_roundtrip() {
        assert_eq!(SparseFmtId::from_u8(0), Some(SparseFmtId::Coo));
        assert_eq!(SparseFmtId::from_u8(1), Some(SparseFmtId::Csr));
        assert_eq!(SparseFmtId::from_u8(2), Some(SparseFmtId::Csc));
        assert_eq!(SparseFmtId::from_u8(3), Some(SparseFmtId::Bsr));
        assert_eq!(SparseFmtId::from_u8(99), None);
    }

    #[test]
    fn sparse_format_names() {
        assert_eq!(SparseFmtId::Coo.name(), "coo");
        assert_eq!(SparseFmtId::Csr.name(), "csr");
    }

    #[test]
    fn coo_construction_and_query() {
        let rows = [0i64, 1, 2];
        let cols = [0i64, 1, 2];
        let vals = [1.0f64, 2.0, 3.0];

        let ptr = nsl_sparse_coo(
            rows.as_ptr() as i64, cols.as_ptr() as i64, vals.as_ptr() as i64,
            3, 3, 3,
        );
        assert_ne!(ptr, 0);

        assert_eq!(nsl_sparse_nnz(ptr), 3);

        let density_bits = nsl_sparse_density(ptr) as u64;
        let density = f64::from_bits(density_bits);
        assert!((density - 1.0 / 3.0).abs() < 0.01); // 3/9

        nsl_sparse_free(ptr);
    }

    #[test]
    fn null_sparse_queries_safe() {
        assert_eq!(nsl_sparse_nnz(0), 0);
        let density_bits = nsl_sparse_density(0) as u64;
        assert_eq!(f64::from_bits(density_bits), 0.0);
    }

    #[test]
    fn stubs_return_zero() {
        assert_eq!(nsl_sparse_from_dense(0, 1, 0), 0);
        assert_eq!(nsl_sparse_to_dense(0), 0);
        assert_eq!(nsl_sparse_spmm(0, 0), 0);
    }

    #[test]
    fn zero_nnz_sparse() {
        // Zero nonzero elements — density should be 0, not NaN
        let rows: Vec<i64> = vec![];
        let cols: Vec<i64> = vec![];
        let vals: Vec<f64> = vec![];
        let ptr = nsl_sparse_coo(
            rows.as_ptr() as i64, cols.as_ptr() as i64, vals.as_ptr() as i64,
            10, 10, 0,
        );
        // nnz=0 with null-ish pointers should still work
        assert_eq!(nsl_sparse_nnz(ptr), 0);
        if ptr != 0 { nsl_sparse_free(ptr); }
    }

    // ── COO → CSR/CSC conversion tests ──────────────────────────────

    #[test]
    fn coo_to_csr_3x3_diagonal() {
        // 3x3 diagonal: (0,0,1), (1,1,2), (2,2,3)
        let rows = [0i64, 1, 2];
        let cols = [0i64, 1, 2];
        let vals = [1.0f64, 2.0, 3.0];
        let coo = nsl_sparse_coo(
            rows.as_ptr() as i64, cols.as_ptr() as i64, vals.as_ptr() as i64,
            3, 3, 3,
        );
        let csr = nsl_sparse_coo_to_csr(coo);
        assert_ne!(csr, 0);

        let s = unsafe { &*(csr as *const NslSparseTensor) };
        assert_eq!(s.format, SparseFmtId::Csr as u8);
        assert_eq!(s.nnz, 3);

        // row_ptr should be [0, 1, 2, 3] — one element per row
        let row_ptr = unsafe { std::slice::from_raw_parts(s.indices_0, 4) };
        assert_eq!(row_ptr, &[0, 1, 2, 3]);

        // col_indices should be [0, 1, 2]
        let col_idx = unsafe { std::slice::from_raw_parts(s.indices_1, 3) };
        assert_eq!(col_idx, &[0, 1, 2]);

        // values should be [1, 2, 3]
        let v = unsafe { std::slice::from_raw_parts(s.data as *const f64, 3) };
        assert_eq!(v, &[1.0, 2.0, 3.0]);

        nsl_sparse_free(csr);
        nsl_sparse_free(coo);
    }

    #[test]
    fn coo_to_csr_unsorted_input() {
        // Unsorted COO: entries not in row order
        let rows = [2i64, 0, 1, 0];
        let cols = [1i64, 0, 2, 1];
        let vals = [7.0f64, 1.0, 5.0, 3.0];
        let coo = nsl_sparse_coo(
            rows.as_ptr() as i64, cols.as_ptr() as i64, vals.as_ptr() as i64,
            3, 3, 4,
        );
        let csr = nsl_sparse_coo_to_csr(coo);
        let s = unsafe { &*(csr as *const NslSparseTensor) };

        // row_ptr: row 0 has 2 entries, row 1 has 1, row 2 has 1 → [0, 2, 3, 4]
        let row_ptr = unsafe { std::slice::from_raw_parts(s.indices_0, 4) };
        assert_eq!(row_ptr, &[0, 2, 3, 4]);

        // Values should be sorted by (row, col)
        let col_idx = unsafe { std::slice::from_raw_parts(s.indices_1, 4) };
        assert_eq!(col_idx, &[0, 1, 2, 1], "columns sorted within rows");

        let v = unsafe { std::slice::from_raw_parts(s.data as *const f64, 4) };
        assert_eq!(v, &[1.0, 3.0, 5.0, 7.0], "values match sorted order");

        nsl_sparse_free(csr);
        nsl_sparse_free(coo);
    }

    #[test]
    fn coo_to_csc_3x3() {
        let rows = [0i64, 1, 0];
        let cols = [0i64, 1, 1];
        let vals = [1.0f64, 2.0, 3.0];
        let coo = nsl_sparse_coo(
            rows.as_ptr() as i64, cols.as_ptr() as i64, vals.as_ptr() as i64,
            2, 2, 3,
        );
        let csc = nsl_sparse_coo_to_csc(coo);
        let s = unsafe { &*(csc as *const NslSparseTensor) };
        assert_eq!(s.format, SparseFmtId::Csc as u8);

        // col_ptr: col 0 has 1, col 1 has 2 → [0, 1, 3]
        let col_ptr = unsafe { std::slice::from_raw_parts(s.indices_0, 3) };
        assert_eq!(col_ptr, &[0, 1, 3]);

        nsl_sparse_free(csc);
        nsl_sparse_free(coo);
    }

    // ── SpMV test ──────────────────────────────────────────────────

    #[test]
    fn spmv_csr_identity() {
        // 3x3 identity × [1, 2, 3] = [1, 2, 3]
        let rows = [0i64, 1, 2];
        let cols = [0i64, 1, 2];
        let vals = [1.0f64, 1.0, 1.0];
        let coo = nsl_sparse_coo(
            rows.as_ptr() as i64, cols.as_ptr() as i64, vals.as_ptr() as i64,
            3, 3, 3,
        );
        let csr = nsl_sparse_coo_to_csr(coo);

        // Create dense vector [1, 2, 3]
        let vec_data = [1.0f64, 2.0, 3.0];
        let vec_ptr = crate::tensor::creation::create_tensor_from_f64_data(&vec_data, &[3]);

        let result = nsl_sparse_spmv(csr, vec_ptr);
        assert_ne!(result, 0);

        let r = crate::tensor::NslTensor::from_ptr_ref(result);
        assert_eq!(r.len, 3);
        let out = unsafe { std::slice::from_raw_parts(r.data as *const f64, 3) };
        assert_eq!(out, &[1.0, 2.0, 3.0]);

        crate::tensor::nsl_tensor_free(result);
        crate::tensor::nsl_tensor_free(vec_ptr);
        nsl_sparse_free(csr);
        nsl_sparse_free(coo);
    }

    // ── Sparse add/mul tests ─────────────────────────────────────

    #[test]
    fn sparse_add_union() {
        // A: (0,0)=1, (1,1)=2
        // B: (0,0)=3, (1,2)=4
        // A+B: (0,0)=4, (1,1)=2, (1,2)=4
        let a_rows = [0i64, 1]; let a_cols = [0i64, 1]; let a_vals = [1.0f64, 2.0];
        let b_rows = [0i64, 1]; let b_cols = [0i64, 2]; let b_vals = [3.0f64, 4.0];
        let a = nsl_sparse_coo(a_rows.as_ptr() as i64, a_cols.as_ptr() as i64, a_vals.as_ptr() as i64, 2, 3, 2);
        let b = nsl_sparse_coo(b_rows.as_ptr() as i64, b_cols.as_ptr() as i64, b_vals.as_ptr() as i64, 2, 3, 2);

        let result = nsl_sparse_add(a, b);
        assert_ne!(result, 0);
        let s = unsafe { &*(result as *const NslSparseTensor) };
        assert_eq!(s.nnz, 3, "union should produce 3 non-zeros");

        // Convert to dense and verify
        let dense = nsl_sparse_to_dense(result);
        let d = crate::tensor::NslTensor::from_ptr_ref(dense);
        let data = unsafe { std::slice::from_raw_parts(d.data as *const f64, 6) };
        assert_eq!(data[0], 4.0); // (0,0) = 1+3
        assert_eq!(data[4], 2.0); // (1,1) = 2
        assert_eq!(data[5], 4.0); // (1,2) = 4

        crate::tensor::nsl_tensor_free(dense);
        nsl_sparse_free(result);
        nsl_sparse_free(a);
        nsl_sparse_free(b);
    }

    #[test]
    fn sparse_mul_intersection() {
        // A: (0,0)=2, (0,1)=3, (1,1)=4
        // B: (0,0)=5, (1,1)=6
        // A*B: (0,0)=10, (1,1)=24  (intersection only)
        let a_rows = [0i64, 0, 1]; let a_cols = [0i64, 1, 1]; let a_vals = [2.0f64, 3.0, 4.0];
        let b_rows = [0i64, 1]; let b_cols = [0i64, 1]; let b_vals = [5.0f64, 6.0];
        let a = nsl_sparse_coo(a_rows.as_ptr() as i64, a_cols.as_ptr() as i64, a_vals.as_ptr() as i64, 2, 2, 3);
        let b = nsl_sparse_coo(b_rows.as_ptr() as i64, b_cols.as_ptr() as i64, b_vals.as_ptr() as i64, 2, 2, 2);

        let result = nsl_sparse_mul(a, b);
        assert_ne!(result, 0);
        let s = unsafe { &*(result as *const NslSparseTensor) };
        assert_eq!(s.nnz, 2, "intersection should produce 2 non-zeros");

        let dense = nsl_sparse_to_dense(result);
        let d = crate::tensor::NslTensor::from_ptr_ref(dense);
        let data = unsafe { std::slice::from_raw_parts(d.data as *const f64, 4) };
        assert_eq!(data[0], 10.0); // 2*5
        assert_eq!(data[3], 24.0); // 4*6

        crate::tensor::nsl_tensor_free(dense);
        nsl_sparse_free(result);
        nsl_sparse_free(a);
        nsl_sparse_free(b);
    }

    // ── f32 dense operands ──────────────────────────────────────────
    //
    // NSL-level creation builtins make f32 tensors, and since C5 step 2a a
    // downloaded GPU operand stays f32 too: that is what SpMV's GPU fallback
    // (BSR, which has no GPU kernel) hands its CPU path. These paths used to
    // read every dense operand as raw f64, walking twice an f32 buffer. The
    // decode runs before the format dispatch, so CSR exercises it for both.

    fn f32_tensor(shape: &[i64], vals: &[f32]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(shape, 1);
        let t = crate::tensor::NslTensor::from_ptr(ptr);
        for (i, v) in vals.iter().enumerate() {
            unsafe { *t.data_f32().add(i) = *v };
        }
        ptr
    }

    #[test]
    fn spmm_and_spmv_read_an_f32_dense_operand_by_its_tag() {
        // A = [[1, 0, 2], [0, 3, 0]], an f32 sparse matrix the way NSL makes
        // one: from an f32 dense tensor (format 1 = CSR). `nsl_sparse_coo`
        // takes raw f64 values and tags them f64, and an f64 sparse operand
        // beside f32 dense ones is refused (C5 step 4).
        let a = f32_tensor(&[2, 3], &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        let csr = nsl_sparse_from_dense(a, 1, 0.5f64.to_bits() as i64);
        assert_ne!(csr, 0);

        // B = [[1, 2], [3, 4], [5, 6]] -> A @ B = [[11, 14], [9, 12]].
        let b = f32_tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = nsl_sparse_spmm(csr, b);
        assert_ne!(c, 0);
        let ct = crate::tensor::NslTensor::from_ptr_ref(c);
        let got: Vec<f64> = (0..4).map(|i| ct.read_scalar_as_f64(i)).collect();
        assert_eq!(got, [11.0, 14.0, 9.0, 12.0], "SpMM with an f32 dense operand");
        assert_eq!(ct.dtype, 1, "the SpMM result takes the f32 operand's dtype (C5)");

        // x = [1, 2, 3] -> A @ x = [7, 6].
        let x = f32_tensor(&[3], &[1.0, 2.0, 3.0]);
        let y = nsl_sparse_spmv(csr, x);
        assert_ne!(y, 0);
        let yt = crate::tensor::NslTensor::from_ptr_ref(y);
        let got: Vec<f64> = (0..2).map(|i| yt.read_scalar_as_f64(i)).collect();
        assert_eq!(got, [7.0, 6.0], "SpMV with an f32 dense vector");
        assert_eq!(yt.dtype, 1, "the SpMV result takes the f32 vector's dtype (C5)");

        for t in [c, b, y, x, a] {
            crate::tensor::nsl_tensor_free(t);
        }
        nsl_sparse_free(csr);
    }

    /// Sparse add/mul results keep their operands' dtype: an f32 sum
    /// densifies to f32. They were tagged f64 always.
    #[test]
    fn sparse_add_and_mul_keep_their_operands_dtype() {
        let d = f32_tensor(&[2, 2], &[1.0, 0.0, 0.0, 2.0]);
        let s1 = nsl_sparse_from_dense(d, 0, 0.5f64.to_bits() as i64);
        let s2 = nsl_sparse_from_dense(d, 0, 0.5f64.to_bits() as i64);
        for (sum, want) in [(nsl_sparse_add(s1, s2), [2.0, 0.0, 0.0, 4.0]), (nsl_sparse_mul(s1, s2), [1.0, 0.0, 0.0, 4.0])] {
            assert_ne!(sum, 0);
            let dense = nsl_sparse_to_dense(sum);
            let t = crate::tensor::NslTensor::from_ptr_ref(dense);
            assert_eq!(t.dtype, 1, "an f32 sparse result densifies to f32");
            let got: Vec<f64> = (0..4).map(|i| t.read_scalar_as_f64(i)).collect();
            assert_eq!(got, want);
            crate::tensor::nsl_tensor_free(dense);
            nsl_sparse_free(sum);
        }
        nsl_sparse_free(s1);
        nsl_sparse_free(s2);
        crate::tensor::nsl_tensor_free(d);
    }

    #[test]
    fn from_dense_reads_an_f32_tensor_by_its_tag() {
        let d = f32_tensor(&[2, 3], &[0.0, 4.5, 0.0, -2.0, 0.0, 0.25]);
        let s = nsl_sparse_from_dense(d, 0, 0.1f64.to_bits() as i64);
        assert_ne!(s, 0);
        assert_eq!(nsl_sparse_nnz(s), 3, "three entries clear the 0.1 threshold");
        let back = nsl_sparse_to_dense(s);
        let bt = crate::tensor::NslTensor::from_ptr_ref(back);
        let got: Vec<f64> = (0..6).map(|i| bt.read_scalar_as_f64(i)).collect();
        assert_eq!(got, [0.0, 4.5, 0.0, -2.0, 0.0, 0.25]);
        assert_eq!(bt.dtype, 1, "to_dense hands back the f32 source's dtype (C5)");

        crate::tensor::nsl_tensor_free(back);
        nsl_sparse_free(s);
        crate::tensor::nsl_tensor_free(d);
    }

    // -----------------------------------------------------------------------
    // One matrix, four formats, every routine
    //
    //   A = [0 5 0  0]      stored with duplicates: (0,1) = 2 and 3,
    //       [0 0 0 -1]      (2,0) = 4 and -4 (which cancel), (2,2) = 7,
    //       [0 0 7  0]      (1,3) = -1.
    //
    // Every expected value below is worked out by hand from that matrix.
    // -----------------------------------------------------------------------

    fn dup_coo() -> i64 {
        let rows = [0i64, 2, 0, 1, 2, 2];
        let cols = [1i64, 0, 1, 3, 2, 0];
        let vals = [2.0f64, 4.0, 3.0, -1.0, 7.0, -4.0];
        nsl_sparse_coo(rows.as_ptr() as i64, cols.as_ptr() as i64, vals.as_ptr() as i64, 3, 4, 6)
    }

    const A_DENSE: [f64; 12] = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 7.0, 0.0];

    fn dense_f64(shape: &[i64], vals: &[f64]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(shape, 0);
        let t = crate::tensor::NslTensor::from_ptr(ptr);
        for (i, v) in vals.iter().enumerate() {
            unsafe { *t.data_f64().add(i) = *v };
        }
        ptr
    }

    fn read_f64(ptr: i64) -> Vec<f64> {
        assert_ne!(ptr, 0, "the routine returned a null tensor");
        let t = crate::tensor::NslTensor::from_ptr(ptr);
        let v = (0..t.len as usize).map(|i| unsafe { *t.data_f64().add(i) }).collect();
        crate::tensor::nsl_tensor_free(ptr);
        v
    }

    /// The matrix in each format: COO as given (duplicates and all), CSR and
    /// CSC converted from it, BSR built from the dense matrix.
    fn all_formats() -> Vec<(&'static str, i64)> {
        let coo = dup_coo();
        let csr = nsl_sparse_coo_to_csr(coo);
        let csc = nsl_sparse_coo_to_csc(coo);
        let dense = dense_f64(&[3, 4], &A_DENSE);
        let bsr = nsl_sparse_from_dense(dense, SparseFmtId::Bsr as i64, f64::to_bits(1e-12) as i64);
        crate::tensor::nsl_tensor_free(dense);
        for (name, p, fmt) in [("csr", csr, 1u8), ("csc", csc, 2), ("bsr", bsr, 3)] {
            assert_ne!(p, 0, "{name} conversion failed");
            assert_eq!(unsafe { &*(p as *const NslSparseTensor) }.format, fmt, "{name}");
        }
        vec![("coo", coo), ("csr", csr), ("csc", csc), ("bsr", bsr)]
    }

    fn free_all(formats: Vec<(&'static str, i64)>) {
        for (_, p) in formats {
            nsl_sparse_free(p);
        }
    }

    /// Duplicates add (2 + 3 = 5, 4 - 4 = 0) in every format. COO used to
    /// keep the last duplicate (3 and -4), and CSC returned null.
    #[test]
    fn to_dense_sums_duplicates_in_every_format() {
        let formats = all_formats();
        for (name, p) in &formats {
            assert_eq!(read_f64(nsl_sparse_to_dense(*p)), A_DENSE, "{name}");
        }
        free_all(formats);
    }

    /// A @ B for B = [[1,2],[3,4],[5,6],[7,8]]:
    /// row 0 = 5*[3,4], row 1 = -1*[7,8], row 2 = 7*[5,6].
    #[test]
    fn spmm_matches_the_hand_product_in_every_format() {
        let b = dense_f64(&[4, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let formats = all_formats();
        for (name, p) in &formats {
            assert_eq!(
                read_f64(nsl_sparse_spmm(*p, b)),
                vec![15.0, 20.0, -7.0, -8.0, 35.0, 42.0],
                "{name}"
            );
        }
        free_all(formats);
        crate::tensor::nsl_tensor_free(b);
    }

    /// A @ [1,2,3,4] = [5*2, -1*4, 7*3].
    #[test]
    fn spmv_matches_the_hand_product_in_every_format() {
        let x = dense_f64(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let formats = all_formats();
        for (name, p) in &formats {
            assert_eq!(read_f64(nsl_sparse_spmv(*p, x)), vec![10.0, -4.0, 21.0], "{name}");
        }
        free_all(formats);
        crate::tensor::nsl_tensor_free(x);
    }

    /// A + C and A * C with C = [1 0 0 0; 0 0 0 1; 0 0 0 0], for every pair
    /// of formats. By hand: A + C has 1 at (0,0), keeps 5 at (0,1) and 7 at
    /// (2,2), and -1 + 1 cancels at (1,3); A * C is -1 at (1,3) alone. With
    /// a CSC or BSR operand the sum used to be the other operand unchanged
    /// and the product all zeros.
    #[test]
    fn add_and_mul_see_every_format() {
        let c_dense = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let c_formats = || {
            let rows = [0i64, 1];
            let cols = [0i64, 3];
            let vals = [1.0f64, 1.0];
            let coo = nsl_sparse_coo(rows.as_ptr() as i64, cols.as_ptr() as i64, vals.as_ptr() as i64, 3, 4, 2);
            let dense = dense_f64(&[3, 4], &c_dense);
            let bsr = nsl_sparse_from_dense(dense, SparseFmtId::Bsr as i64, f64::to_bits(1e-12) as i64);
            crate::tensor::nsl_tensor_free(dense);
            vec![("coo", coo), ("csr", nsl_sparse_coo_to_csr(coo)), ("csc", nsl_sparse_coo_to_csc(coo)), ("bsr", bsr)]
        };
        let want_sum = [1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0];
        let want_product = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0];
        let a_formats = all_formats();
        let c_formats = c_formats();
        for (a_name, a) in &a_formats {
            for (c_name, c) in &c_formats {
                for (x, y, order) in [(*a, *c, "A op C"), (*c, *a, "C op A")] {
                    let sum = nsl_sparse_add(x, y);
                    assert_eq!(read_f64(nsl_sparse_to_dense(sum)), want_sum, "{a_name} + {c_name} ({order})");
                    nsl_sparse_free(sum);
                    let product = nsl_sparse_mul(x, y);
                    assert_eq!(
                        read_f64(nsl_sparse_to_dense(product)),
                        want_product,
                        "{a_name} * {c_name} ({order})"
                    );
                    nsl_sparse_free(product);
                }
            }
        }
        free_all(a_formats);
        free_all(c_formats);
    }

    /// Duplicates are summed BEFORE a product: with B holding 2 and 3 at one
    /// coordinate and A holding 10 there, A * B is 10 * 5, not 10 * 3.
    #[test]
    fn mul_coalesces_duplicates_first() {
        let (r, c) = ([0i64, 0], [0i64, 0]);
        let b_vals = [2.0f64, 3.0];
        let b = nsl_sparse_coo(r.as_ptr() as i64, c.as_ptr() as i64, b_vals.as_ptr() as i64, 1, 1, 2);
        let a_vals = [10.0f64];
        let a = nsl_sparse_coo(r.as_ptr() as i64, c.as_ptr() as i64, a_vals.as_ptr() as i64, 1, 1, 1);
        for (x, y) in [(a, b), (b, a)] {
            let p = nsl_sparse_mul(x, y);
            assert_eq!(read_f64(nsl_sparse_to_dense(p)), vec![50.0]);
            nsl_sparse_free(p);
        }
        nsl_sparse_free(a);
        nsl_sparse_free(b);
    }

    /// Under a recording tape the product must reach the dense operand's
    /// gradient. For loss = sum(A @ B), dL/dB[k, j] is the sum of column k
    /// of A: 4 - 4 = 0, 2 + 3 = 5, 7 and -1. The sparse kernels record
    /// nothing, so this gradient used to be all zeros.
    #[test]
    fn spmm_under_a_tape_gives_the_dense_operand_its_gradient() {
        use crate::list::{nsl_list_free, nsl_list_get, nsl_list_new, nsl_list_push};
        let formats = all_formats();
        for (name, a) in &formats {
            let b = dense_f64(&[4, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let params = nsl_list_new();
            nsl_list_push(params, b);
            crate::autodiff::nsl_tape_start(params);
            let y = nsl_sparse_spmm(*a, b);
            let loss = crate::tensor::nsl_tensor_sum(y);
            let grads = crate::autodiff::nsl_tape_backward(loss, params);
            crate::autodiff::nsl_tape_stop();
            let g = nsl_list_get(grads, 0);
            let gt = crate::tensor::NslTensor::from_ptr(g);
            let got: Vec<f64> = (0..gt.len as usize).map(|i| gt.read_scalar_as_f64(i)).collect();
            assert_eq!(got, vec![0.0, 0.0, 5.0, 5.0, 7.0, 7.0, -1.0, -1.0], "{name}");
            // The forward value is unchanged by the route taken.
            let yt = crate::tensor::NslTensor::from_ptr(y);
            let fwd: Vec<f64> = (0..yt.len as usize).map(|i| yt.read_scalar_as_f64(i)).collect();
            assert_eq!(fwd, vec![15.0, 20.0, -7.0, -8.0, 35.0, 42.0], "{name}");
            nsl_list_free(params);
        }
        free_all(formats);
    }
}

