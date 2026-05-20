//! Calibration-data loader for AWQ / quantization pipelines.
//!
//! Supports two on-disk formats:
//!   - `.bin`  — NSL-native binary format (magic "NSLB" + rank + dims + f32 payload)
//!   - `.safetensors` — standard safetensors archive; must contain a tensor named "calibration"
//!
//! The public Rust API is `load(path)` and `peek_shape(path)`.
//! Four `extern "C"` functions expose the loader over FFI for the subprocess calibration binary.

use std::path::Path;

// ── Public types ──────────────────────────────────────────────────────────────

/// Contiguous collection of equal-sized calibration batches.
#[repr(C)]
pub struct Batches {
    /// Number of batches (= shape[0]).
    pub count: usize,
    /// Byte size of a single batch (= prod(shape[1..]) * 4, f32 elements).
    pub batch_nbytes: usize,
    /// Raw f32 bytes laid out as [count * batch_nbytes].
    data: Vec<u8>,
    /// Full shape: [count, dim1, dim2, ...].
    shape: Vec<u32>,
}

impl Batches {
    /// Borrow the raw bytes for batch `i`.
    pub fn batch_at(&self, i: usize) -> Option<&[u8]> {
        if i >= self.count {
            return None;
        }
        let off = i * self.batch_nbytes;
        Some(&self.data[off..off + self.batch_nbytes])
    }

    /// Shape slice: [count, dim1, dim2, ...].
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum CalibDataError {
    UnsupportedExt(String),
    Io(std::io::Error),
    BinHeader(String),
    Safetensors(String),
}

impl std::fmt::Display for CalibDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibDataError::UnsupportedExt(e) => write!(f, "unsupported extension: {e}"),
            CalibDataError::Io(e) => write!(f, "io: {e}"),
            CalibDataError::BinHeader(e) => write!(f, "bin header: {e}"),
            CalibDataError::Safetensors(e) => write!(f, "safetensors: {e}"),
        }
    }
}

impl std::error::Error for CalibDataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CalibDataError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CalibDataError {
    fn from(e: std::io::Error) -> Self {
        CalibDataError::Io(e)
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load calibration data from `path`, dispatching on file extension.
pub fn load(path: &Path) -> Result<Batches, CalibDataError> {
    match path.extension().and_then(|s| s.to_str()) {
        Some("safetensors") => load_safetensors(path),
        Some("bin") => load_bin(path),
        Some(other) => Err(CalibDataError::UnsupportedExt(other.to_string())),
        None => Err(CalibDataError::UnsupportedExt(String::new())),
    }
}

/// Return the full shape `[count, dim1, dim2, ...]` without retaining the payload.
pub fn peek_shape(path: &Path) -> Result<Vec<u32>, CalibDataError> {
    load(path).map(|b| b.shape)
}

/// For rank-3 calibration tensors `[count, seq, hidden]`, returns `(count, seq)`.
/// Errors if rank is not exactly 3.
pub fn peek_batch_seq(path: &Path) -> Result<(u32, u32), CalibDataError> {
    let shape = peek_shape(path)?;
    if shape.len() != 3 {
        return Err(CalibDataError::BinHeader(format!(
            "expected rank-3 calibration tensor, got rank {}",
            shape.len()
        )));
    }
    Ok((shape[0], shape[1]))
}

// ── .bin loader ───────────────────────────────────────────────────────────────
//
// Binary format:
//   [0..4]   magic "NSLB"
//   [4..8]   rank: u32 LE
//   [8..8+rank*4]  dims[0..rank]: u32 LE each
//   [8+rank*4 ..]  f32 payload (little-endian), count * prod(dims[1..]) * 4 bytes

fn load_bin(path: &Path) -> Result<Batches, CalibDataError> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 {
        return Err(CalibDataError::BinHeader("too short".into()));
    }
    if &bytes[0..4] != b"NSLB" {
        return Err(CalibDataError::BinHeader("bad magic".into()));
    }
    let rank = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    if rank == 0 {
        return Err(CalibDataError::BinHeader("rank must be >= 1".into()));
    }
    let dims_end = 8 + rank * 4;
    if bytes.len() < dims_end {
        return Err(CalibDataError::BinHeader("truncated dims".into()));
    }
    let shape: Vec<u32> = (0..rank)
        .map(|i| u32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap()))
        .collect();

    let count = shape[0] as usize;
    let batch_elems: u64 = shape[1..].iter().map(|&d| d as u64).product::<u64>();
    let batch_nbytes = (batch_elems * 4) as usize;
    let data = bytes[dims_end..].to_vec();

    if data.len() != count * batch_nbytes {
        return Err(CalibDataError::BinHeader(format!(
            "payload {} bytes != expected {} bytes",
            data.len(),
            count * batch_nbytes
        )));
    }
    Ok(Batches {
        count,
        batch_nbytes,
        data,
        shape,
    })
}

// ── .safetensors loader ───────────────────────────────────────────────────────
//
// Expects a tensor named "calibration" with dtype F32 and shape [count, dim1, ...].

fn load_safetensors(path: &Path) -> Result<Batches, CalibDataError> {
    let bytes = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| CalibDataError::Safetensors(format!("{e}")))?;
    let t = st
        .tensor("calibration")
        .map_err(|e| CalibDataError::Safetensors(format!("missing 'calibration' tensor: {e}")))?;

    // Verify dtype is F32 (4 bytes per element).
    if t.dtype() != safetensors::Dtype::F32 {
        return Err(CalibDataError::Safetensors(format!(
            "expected F32 tensor, got {:?}",
            t.dtype()
        )));
    }

    let dims: Vec<u32> = t.shape().iter().map(|&d| d as u32).collect();
    if dims.is_empty() {
        return Err(CalibDataError::Safetensors("zero-rank tensor".into()));
    }

    let count = dims[0] as usize;
    let batch_elems: u64 = dims[1..].iter().map(|&d| d as u64).product::<u64>();
    let batch_nbytes = (batch_elems * 4) as usize;

    Ok(Batches {
        count,
        batch_nbytes,
        data: t.data().to_vec(),
        shape: dims,
    })
}

// ── FFI ───────────────────────────────────────────────────────────────────────

/// Load calibration data from a UTF-8 file path.
/// Returns an owned `*mut Batches` on success, null on failure.
/// Caller must eventually call `nsl_calibration_free`.
#[no_mangle]
pub extern "C" fn nsl_calibration_load(path_ptr: *const u8, path_len: usize) -> *mut Batches {
    if path_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let path_bytes = unsafe { std::slice::from_raw_parts(path_ptr, path_len) };
    let path_str = match std::str::from_utf8(path_bytes) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    match load(Path::new(path_str)) {
        Ok(b) => Box::into_raw(Box::new(b)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Borrow a pointer to the raw f32 bytes of batch `i`.
/// Sets `*out_ptr` and `*out_len`; zeroes both if `i` is out of range or
/// `batches` is null.  The pointer is valid as long as `batches` is alive.
#[no_mangle]
pub extern "C" fn nsl_calibration_batch_at(
    batches: *mut Batches,
    i: usize,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) {
    if batches.is_null() || out_ptr.is_null() || out_len.is_null() {
        return;
    }
    let b = unsafe { &*batches };
    if i >= b.count {
        unsafe {
            *out_ptr = std::ptr::null();
            *out_len = 0;
        }
        return;
    }
    let off = i * b.batch_nbytes;
    unsafe {
        *out_ptr = b.data.as_ptr().add(off);
        *out_len = b.batch_nbytes;
    }
}

/// Return the number of batches in `batches` (0 if null).
#[no_mangle]
pub extern "C" fn nsl_calibration_count(batches: *mut Batches) -> usize {
    if batches.is_null() {
        return 0;
    }
    unsafe { (*batches).count }
}

/// Free a `Batches` object previously returned by `nsl_calibration_load`.
#[no_mangle]
pub extern "C" fn nsl_calibration_free(batches: *mut Batches) {
    if !batches.is_null() {
        unsafe { drop(Box::from_raw(batches)) }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── .bin tests ──────────────────────────────────────────────────────────

    #[test]
    fn bin_roundtrip_two_batches() {
        // Header: magic "NSLB" | rank=3 (u32 LE) | dims [2, 4, 4] | f32 payload
        let mut tmp = NamedTempFile::with_suffix(".bin").unwrap();
        tmp.write_all(b"NSLB").unwrap();
        tmp.write_all(&3u32.to_le_bytes()).unwrap();
        for d in [2u32, 4, 4] {
            tmp.write_all(&d.to_le_bytes()).unwrap();
        }
        let floats: Vec<f32> = (0..32).map(|i| i as f32).collect();
        for f in &floats {
            tmp.write_all(&f.to_le_bytes()).unwrap();
        }
        tmp.flush().unwrap();

        let b = load(tmp.path()).unwrap();
        assert_eq!(b.count, 2);
        assert_eq!(b.batch_nbytes, 4 * 4 * 4); // 4*4 elements * 4 bytes
        assert_eq!(b.shape, vec![2, 4, 4]);
    }

    #[test]
    fn bin_batch_at_returns_correct_slice() {
        let mut tmp = NamedTempFile::with_suffix(".bin").unwrap();
        tmp.write_all(b"NSLB").unwrap();
        tmp.write_all(&2u32.to_le_bytes()).unwrap(); // rank=2
        for d in [2u32, 4] {
            tmp.write_all(&d.to_le_bytes()).unwrap();
        }
        // batch 0: [0.0, 1.0, 2.0, 3.0], batch 1: [10.0, 11.0, 12.0, 13.0]
        let floats: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        for f in &floats {
            tmp.write_all(&f.to_le_bytes()).unwrap();
        }
        tmp.flush().unwrap();

        let b = load(tmp.path()).unwrap();
        let s1 = b.batch_at(1).unwrap();
        let got: Vec<f32> = s1
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, vec![10.0f32, 11.0, 12.0, 13.0]);
        assert!(b.batch_at(2).is_none());
    }

    #[test]
    fn bin_bad_magic_errors() {
        let mut tmp = NamedTempFile::with_suffix(".bin").unwrap();
        tmp.write_all(b"BAAD").unwrap();
        tmp.write_all(&0u32.to_le_bytes()).unwrap();
        tmp.flush().unwrap();

        assert!(matches!(
            load(tmp.path()),
            Err(CalibDataError::BinHeader(_))
        ));
    }

    #[test]
    fn bin_payload_mismatch_errors() {
        let mut tmp = NamedTempFile::with_suffix(".bin").unwrap();
        tmp.write_all(b"NSLB").unwrap();
        tmp.write_all(&2u32.to_le_bytes()).unwrap(); // rank=2
        for d in [4u32, 4] {
            tmp.write_all(&d.to_le_bytes()).unwrap();
        }
        // Only write 4 floats instead of the required 4*4=16
        for f in [0.0f32, 1.0, 2.0, 3.0] {
            tmp.write_all(&f.to_le_bytes()).unwrap();
        }
        tmp.flush().unwrap();

        assert!(matches!(
            load(tmp.path()),
            Err(CalibDataError::BinHeader(_))
        ));
    }

    // ── extension tests ─────────────────────────────────────────────────────

    #[test]
    fn rejects_txt_extension() {
        let tmp = NamedTempFile::with_suffix(".txt").unwrap();
        assert!(matches!(
            load(tmp.path()),
            Err(CalibDataError::UnsupportedExt(_))
        ));
    }

    #[test]
    fn rejects_no_extension() {
        // NamedTempFile without suffix has no extension on most platforms
        let tmp = tempfile::Builder::new()
            .prefix("calib_no_ext")
            .tempfile()
            .unwrap();
        assert!(matches!(
            load(tmp.path()),
            Err(CalibDataError::UnsupportedExt(_))
        ));
    }

    // ── peek_batch_seq tests ─────────────────────────────────────────────────

    #[test]
    fn peek_batch_seq_returns_count_and_seq_for_rank3() {
        let mut tmp = NamedTempFile::with_suffix(".bin").unwrap();
        tmp.write_all(b"NSLB").unwrap();
        tmp.write_all(&3u32.to_le_bytes()).unwrap();
        for d in [5u32, 7, 32] {
            tmp.write_all(&d.to_le_bytes()).unwrap();
        }
        let payload_floats: usize = 5 * 7 * 32;
        for i in 0..payload_floats {
            tmp.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        tmp.flush().unwrap();

        let (count, seq) = peek_batch_seq(tmp.path()).unwrap();
        assert_eq!(count, 5);
        assert_eq!(seq, 7);
    }

    #[test]
    fn peek_batch_seq_rejects_rank2() {
        let mut tmp = NamedTempFile::with_suffix(".bin").unwrap();
        tmp.write_all(b"NSLB").unwrap();
        tmp.write_all(&2u32.to_le_bytes()).unwrap();
        for d in [4u32, 8] {
            tmp.write_all(&d.to_le_bytes()).unwrap();
        }
        let payload_floats: usize = 4 * 8;
        for i in 0..payload_floats {
            tmp.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        tmp.flush().unwrap();

        match peek_batch_seq(tmp.path()) {
            Err(CalibDataError::BinHeader(msg)) => {
                assert!(msg.contains("rank"), "expected rank error, got: {msg}");
            }
            other => panic!("expected BinHeader rank error, got {other:?}"),
        }
    }

    // ── peek_shape tests ────────────────────────────────────────────────────

    #[test]
    fn peek_shape_returns_dims_without_full_load_verification() {
        let mut tmp = NamedTempFile::with_suffix(".bin").unwrap();
        tmp.write_all(b"NSLB").unwrap();
        tmp.write_all(&3u32.to_le_bytes()).unwrap();
        for d in [8u32, 4, 64] {
            tmp.write_all(&d.to_le_bytes()).unwrap();
        }
        let payload_floats: usize = 8 * 4 * 64;
        for i in 0..payload_floats {
            tmp.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        tmp.flush().unwrap();

        assert_eq!(peek_shape(tmp.path()).unwrap(), vec![8u32, 4, 64]);
    }

    // ── FFI tests ───────────────────────────────────────────────────────────

    #[test]
    fn ffi_count_and_batch_at() {
        let mut tmp = NamedTempFile::with_suffix(".bin").unwrap();
        tmp.write_all(b"NSLB").unwrap();
        tmp.write_all(&2u32.to_le_bytes()).unwrap(); // rank=2
        for d in [3u32, 2] {
            tmp.write_all(&d.to_le_bytes()).unwrap();
        }
        // 3 batches of 2 f32s each
        for i in 0..6u32 {
            tmp.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        tmp.flush().unwrap();

        let path_str = tmp.path().to_str().unwrap();
        let ptr = nsl_calibration_load(path_str.as_ptr(), path_str.len());
        assert!(!ptr.is_null());

        assert_eq!(nsl_calibration_count(ptr), 3);

        let mut out_ptr: *const u8 = std::ptr::null();
        let mut out_len: usize = 0;
        nsl_calibration_batch_at(ptr, 2, &mut out_ptr, &mut out_len);
        assert!(!out_ptr.is_null());
        assert_eq!(out_len, 8); // 2 f32s

        // Batch 3 (out of range) must return null
        nsl_calibration_batch_at(ptr, 3, &mut out_ptr, &mut out_len);
        assert!(out_ptr.is_null());
        assert_eq!(out_len, 0);

        nsl_calibration_free(ptr);
    }

    #[test]
    fn ffi_null_safety() {
        assert_eq!(nsl_calibration_count(std::ptr::null_mut()), 0);
        nsl_calibration_free(std::ptr::null_mut()); // must not panic
        assert_eq!(
            nsl_calibration_load(std::ptr::null(), 0),
            std::ptr::null_mut()
        );
    }
}
