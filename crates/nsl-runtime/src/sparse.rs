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
#[no_mangle]
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
    let mut val_copy = vec![0u8; n * 8]; // f64 = 8 bytes
    unsafe {
        std::ptr::copy_nonoverlapping(rows_ptr as *const i64, row_copy.as_mut_ptr(), n);
        std::ptr::copy_nonoverlapping(cols_ptr as *const i64, col_copy.as_mut_ptr(), n);
        std::ptr::copy_nonoverlapping(vals_ptr as *const u8, val_copy.as_mut_ptr(), n * 8);
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
#[no_mangle]
pub extern "C" fn nsl_sparse_from_dense(dense_ptr: i64, format: i64, threshold_bits: i64) -> i64 {
    if dense_ptr == 0 { return 0; }
    let threshold = f64::from_bits(threshold_bits as u64);

    // Read dense tensor fields via raw pointer (NslTensor is pub(crate), same crate)
    let tensor = unsafe { &*(dense_ptr as *const crate::tensor::NslTensor) };
    if tensor.ndim != 2 { return 0; }
    let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 2) };
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    let data = unsafe { std::slice::from_raw_parts(tensor.data as *const f64, rows * cols) };

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
        _ => SparseFmtId::Coo,
    };

    if fmt == SparseFmtId::Csr {
        // Convert COO to CSR: build row_ptr array
        let mut row_ptr = vec![0i64; rows + 1];
        for &r in &row_indices {
            row_ptr[r as usize + 1] += 1;
        }
        for i in 1..=rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        let val_bytes: Vec<u8> = values.iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();

        let sparse = Box::new(NslSparseTensor {
            format: SparseFmtId::Csr as u8,
            device: 0, dtype: 0, ndim: 2,
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
        let val_bytes: Vec<u8> = values.iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();

        let sparse = Box::new(NslSparseTensor {
            format: SparseFmtId::Coo as u8,
            device: 0, dtype: 0, ndim: 2,
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
/// Returns pointer to NslTensor (f64, 2D), or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_sparse_to_dense(sparse_ptr: i64) -> i64 {
    if sparse_ptr == 0 { return 0; }
    let sparse = unsafe { &*(sparse_ptr as *const NslSparseTensor) };
    let rows = sparse.rows as usize;
    let cols = sparse.cols as usize;
    let nnz = sparse.nnz as usize;

    // Allocate dense output (zeros)
    let mut dense = vec![0.0f64; rows * cols];
    let vals = unsafe { std::slice::from_raw_parts(sparse.data as *const f64, nnz) };

    match SparseFmtId::from_u8(sparse.format) {
        Some(SparseFmtId::Coo) => {
            let row_idx = unsafe { std::slice::from_raw_parts(sparse.indices_0, nnz) };
            let col_idx = unsafe { std::slice::from_raw_parts(sparse.indices_1, nnz) };
            for i in 0..nnz {
                let r = row_idx[i] as usize;
                let c = col_idx[i] as usize;
                if r < rows && c < cols {
                    dense[r * cols + c] = vals[i];
                }
            }
        }
        Some(SparseFmtId::Csr) => {
            let row_ptr = unsafe { std::slice::from_raw_parts(sparse.indices_0, rows + 1) };
            let col_idx = unsafe { std::slice::from_raw_parts(sparse.indices_1, nnz) };
            for r in 0..rows {
                let start = row_ptr[r] as usize;
                let end = row_ptr[r + 1] as usize;
                for j in start..end {
                    let c = col_idx[j] as usize;
                    if c < cols {
                        dense[r * cols + c] = vals[j];
                    }
                }
            }
        }
        _ => return 0,
    }

    // Create NslTensor — use nsl_tensor_create_from_data
    crate::tensor::creation::create_tensor_from_f64_data(&dense, &[rows as i64, cols as i64])
}

/// Get the number of nonzero elements.
#[no_mangle]
pub extern "C" fn nsl_sparse_nnz(sparse_ptr: i64) -> i64 {
    if sparse_ptr == 0 { return 0; }
    let sparse = unsafe { &*(sparse_ptr as *const NslSparseTensor) };
    sparse.nnz
}

/// Get the density (nnz / total_elements).
/// Returns f64 bits as i64.
#[no_mangle]
pub extern "C" fn nsl_sparse_density(sparse_ptr: i64) -> i64 {
    if sparse_ptr == 0 { return f64::to_bits(0.0) as i64; }
    let sparse = unsafe { &*(sparse_ptr as *const NslSparseTensor) };
    let total = sparse.rows * sparse.cols;
    let density = if total > 0 { sparse.nnz as f64 / total as f64 } else { 0.0 };
    f64::to_bits(density) as i64
}

/// Sparse-dense matrix multiply: sparse(M×K) × dense(K×N) → dense(M×N).
/// sparse_ptr: NslSparseTensor (COO or CSR)
/// dense_ptr: NslTensor (2D, f64)
/// Returns pointer to new NslTensor, or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_sparse_spmm(sparse_ptr: i64, dense_ptr: i64) -> i64 {
    if sparse_ptr == 0 || dense_ptr == 0 { return 0; }
    let sparse = unsafe { &*(sparse_ptr as *const NslSparseTensor) };
    let dense = unsafe { &*(dense_ptr as *const crate::tensor::NslTensor) };

    if dense.ndim != 2 { return 0; }
    let d_shape = unsafe { std::slice::from_raw_parts(dense.shape, 2) };
    let k = d_shape[0] as usize;
    let n = d_shape[1] as usize;
    let m = sparse.rows as usize;
    let nnz = sparse.nnz as usize;

    // Verify dimension compatibility: sparse cols == dense rows
    if sparse.cols as usize != k { return 0; }

    let d_data = unsafe { std::slice::from_raw_parts(dense.data as *const f64, k * n) };
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
        _ => return 0,
    }

    crate::tensor::creation::create_tensor_from_f64_data(&output, &[m as i64, n as i64])
}

/// Free a sparse tensor and its owned arrays.
#[no_mangle]
pub extern "C" fn nsl_sparse_free(sparse_ptr: i64) -> i64 {
    if sparse_ptr == 0 { return 0; }
    let sparse = unsafe { Box::from_raw(sparse_ptr as *mut NslSparseTensor) };
    if sparse.owns_data == 1 {
        let n = sparse.nnz as usize;

        // BUG-1 fix: CSR indices_0 is row_ptr with length (rows+1), not nnz.
        // COO indices_0 is row_indices with length nnz.
        let idx0_len = if sparse.format == SparseFmtId::Csr as u8 {
            sparse.rows as usize + 1
        } else {
            n
        };

        if !sparse.indices_0.is_null() && idx0_len > 0 {
            let _ = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(sparse.indices_0, idx0_len)) };
        }
        if !sparse.indices_1.is_null() && n > 0 {
            let _ = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(sparse.indices_1, n)) };
        }
        if !sparse.data.is_null() && n > 0 {
            let bytes = n * 8;
            let _ = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(sparse.data, bytes)) };
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
#[no_mangle]
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

    let val_bytes: Vec<u8> = sorted_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
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
#[no_mangle]
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

    let val_bytes: Vec<u8> = sorted_vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
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
// SpMV: sparse matrix × dense vector → dense vector
// ---------------------------------------------------------------------------

/// SpMV: CSR sparse matrix [M,K] × dense vector [K] → dense vector [M].
#[no_mangle]
pub extern "C" fn nsl_sparse_spmv(sparse_ptr: i64, vec_ptr: i64) -> i64 {
    if sparse_ptr == 0 || vec_ptr == 0 { return 0; }
    let sparse = unsafe { &*(sparse_ptr as *const NslSparseTensor) };
    let vec_t = unsafe { &*(vec_ptr as *const crate::tensor::NslTensor) };

    let m = sparse.rows as usize;
    let k = sparse.cols as usize;
    let nnz = sparse.nnz as usize;

    // Vector must be 1D or 2D with second dim = 1
    let vec_len = vec_t.len as usize;
    if vec_len != k { return 0; }

    let x = unsafe { std::slice::from_raw_parts(vec_t.data as *const f64, k) };
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
        _ => return 0,
    }

    crate::tensor::creation::create_tensor_from_f64_data(&output, &[m as i64])
}

// ---------------------------------------------------------------------------
// Sparse + Sparse (union merge lattice)
// ---------------------------------------------------------------------------

/// Element-wise sparse add: sparse [M,N] + sparse [M,N] → sparse [M,N] (COO output).
/// Uses union merge: output has non-zeros where either input has non-zeros.
#[no_mangle]
pub extern "C" fn nsl_sparse_add(a_ptr: i64, b_ptr: i64) -> i64 {
    if a_ptr == 0 || b_ptr == 0 { return 0; }
    let a = unsafe { &*(a_ptr as *const NslSparseTensor) };
    let b = unsafe { &*(b_ptr as *const NslSparseTensor) };
    if a.rows != b.rows || a.cols != b.cols { return 0; }

    let rows = a.rows;
    let cols = a.cols;

    // Convert both to COO-like (row, col, val) triples
    let a_triples = sparse_to_triples(a);
    let b_triples = sparse_to_triples(b);

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

    build_coo_from_triples(rows, cols, &out_rows, &out_cols, &out_vals)
}

/// Element-wise sparse mul: sparse [M,N] * sparse [M,N] → sparse [M,N] (COO output).
/// Uses intersection merge: output has non-zeros only where both have non-zeros.
#[no_mangle]
pub extern "C" fn nsl_sparse_mul(a_ptr: i64, b_ptr: i64) -> i64 {
    if a_ptr == 0 || b_ptr == 0 { return 0; }
    let a = unsafe { &*(a_ptr as *const NslSparseTensor) };
    let b = unsafe { &*(b_ptr as *const NslSparseTensor) };
    if a.rows != b.rows || a.cols != b.cols { return 0; }

    let rows = a.rows;
    let cols = a.cols;
    let num_rows = rows as usize;
    let num_cols = cols as usize;

    let a_triples = sparse_to_triples(a);
    let b_triples = sparse_to_triples(b);

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

    build_coo_from_triples(rows, cols, &out_rows, &out_cols, &out_vals)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract (row, col, val) triples from any sparse format.
fn sparse_to_triples(s: &NslSparseTensor) -> Vec<(i64, i64, f64)> {
    let nnz = s.nnz as usize;
    if nnz == 0 { return Vec::new(); }
    let vals = unsafe { std::slice::from_raw_parts(s.data as *const f64, nnz) };

    match SparseFmtId::from_u8(s.format) {
        Some(SparseFmtId::Coo) => {
            let rows = unsafe { std::slice::from_raw_parts(s.indices_0, nnz) };
            let cols = unsafe { std::slice::from_raw_parts(s.indices_1, nnz) };
            (0..nnz).map(|i| (rows[i], cols[i], vals[i])).collect()
        }
        Some(SparseFmtId::Csr) => {
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
        _ => Vec::new(),
    }
}

/// Build a COO NslSparseTensor from triples.
fn build_coo_from_triples(rows: i64, cols: i64, r: &[i64], c: &[i64], v: &[f64]) -> i64 {
    let nnz = v.len();
    let val_bytes: Vec<u8> = v.iter().flat_map(|val| val.to_ne_bytes()).collect();
    let sparse = Box::new(NslSparseTensor {
        format: SparseFmtId::Coo as u8,
        device: 0, dtype: 0, ndim: 2,
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

        let r = unsafe { &*(result as *const crate::tensor::NslTensor) };
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
        let d = unsafe { &*(dense as *const crate::tensor::NslTensor) };
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
        let d = unsafe { &*(dense as *const crate::tensor::NslTensor) };
        let data = unsafe { std::slice::from_raw_parts(d.data as *const f64, 4) };
        assert_eq!(data[0], 10.0); // 2*5
        assert_eq!(data[3], 24.0); // 4*6

        crate::tensor::nsl_tensor_free(dense);
        nsl_sparse_free(result);
        nsl_sparse_free(a);
        nsl_sparse_free(b);
    }
}
