//! M50: Sparse tensor runtime — storage, construction, and query FFI.

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
    pub refcount: i64,  // lifecycle: same model as NslTensor
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
        refcount: 1,
        owns_data: 1,
    });
    Box::into_raw(sparse) as i64
}

/// Convert a dense tensor to sparse format.
/// Returns pointer to NslSparseTensor, or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_sparse_from_dense(dense_ptr: i64, format: i64) -> i64 {
    let _ = (dense_ptr, format);
    0 // Stub: full implementation in M50b
}

/// Convert a sparse tensor to dense.
/// Returns pointer to NslTensor, or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_sparse_to_dense(sparse_ptr: i64) -> i64 {
    let _ = sparse_ptr;
    0 // Stub: full implementation in M50b
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

/// Sparse-dense matrix multiply (SpMM).
/// Stub: returns 0 (null). Full kernel dispatch in M50b.
#[no_mangle]
pub extern "C" fn nsl_sparse_spmm(sparse_ptr: i64, dense_ptr: i64) -> i64 {
    let _ = (sparse_ptr, dense_ptr);
    0
}

/// Free a sparse tensor and its owned arrays.
#[no_mangle]
pub extern "C" fn nsl_sparse_free(sparse_ptr: i64) -> i64 {
    if sparse_ptr == 0 { return 0; }
    let sparse = unsafe { Box::from_raw(sparse_ptr as *mut NslSparseTensor) };
    if sparse.owns_data == 1 {
        // Free owned index and data arrays
        let n = sparse.nnz as usize;
        if !sparse.indices_0.is_null() && n > 0 {
            let _ = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(sparse.indices_0, n)) };
        }
        if !sparse.indices_1.is_null() && n > 0 {
            let _ = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(sparse.indices_1, n)) };
        }
        if !sparse.data.is_null() && n > 0 {
            let bytes = n * 8; // f64 = 8 bytes (approximate; exact depends on dtype)
            let _ = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(sparse.data, bytes)) };
        }
    }
    // Box::from_raw above already freed the NslSparseTensor struct
    0
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
        assert_eq!(nsl_sparse_from_dense(0, 1), 0);
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
}
