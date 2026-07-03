//! GPUDirect Storage (GDS) integration — DMA from NVMe to GPU memory.
//!
//! On systems with cuFile (GDS), data flows NVMe → GPU without CPU bounce buffers.
//! On systems without GDS, falls back to mmap + cudaMemcpyAsync.
//!
//! The NSL compiler statically proves async GDS writes complete before tensor core
//! reads via mbarrier hardware barriers (leveraging strict aliasing from M38).

use std::io;
use std::os::raw::c_void;
use std::path::Path;

// ---------------------------------------------------------------------------
// GDS availability detection
// ---------------------------------------------------------------------------

/// Check if GPUDirect Storage is available on this system.
/// Returns true only if cuFile library is loaded and GDS driver is active.
pub fn is_gds_available() -> bool {
    // GDS requires: (1) cuFile library, (2) GDS kernel driver, (3) compatible NVMe
    // For now, check via environment variable or runtime probe
    #[cfg(feature = "cuda")]
    {
        std::env::var("NSL_GDS_ENABLED").map_or(false, |v| v == "1")
    }
    #[cfg(not(feature = "cuda"))]
    { false }
}

// ---------------------------------------------------------------------------
// GDS read (with mmap fallback)
// ---------------------------------------------------------------------------

/// Read data from a file into a GPU device buffer.
///
/// If GDS is available, uses cuFileReadAsync for zero-copy NVMe→GPU DMA.
/// Otherwise, falls back to: read the file region into a host buffer, then
/// copy host→device (`cuMemcpyHtoD`). The fallback therefore requires the
/// `cuda` feature; without it, a GPU destination cannot be filled and the
/// call refuses rather than silently leaving the buffer untouched.
///
/// `gpu_buf`: destination device pointer (must be non-null; on the GDS path it
///            must be registered with cuFileBufRegister)
/// `offset`: byte offset in the file
/// `size`: number of bytes to read (must be > 0)
///
/// Returns `Ok(())` on success, `Err` on I/O failure, a null/zero-size
/// destination, or (fallback path) a build without the `cuda` feature.
pub fn gds_read(
    path: &Path,
    gpu_buf: *mut c_void,
    offset: u64,
    size: u64,
    _stream: i64,
) -> io::Result<()> {
    if is_gds_available() {
        gds_read_cufile(path, gpu_buf, offset, size)
    } else {
        gds_read_mmap_fallback(path, gpu_buf, offset, size)
    }
}

/// GDS path: cuFileReadAsync → DMA from NVMe directly to GPU memory.
fn gds_read_cufile(
    _path: &Path,
    _gpu_buf: *mut c_void,
    _offset: u64,
    _size: u64,
) -> io::Result<()> {
    // TODO: Bind cuFile API when CUDA feature is enabled
    // cuFileDriverOpen()
    // cuFileHandleRegister(path)
    // cuFileBufRegister(gpu_buf, size)
    // cuFileReadAsync(handle, gpu_buf, size, offset, stream)
    Err(io::Error::new(io::ErrorKind::Unsupported, "cuFile not linked — use mmap fallback"))
}

/// Fallback path: read file → host staging buffer → `cuMemcpyHtoD` to the
/// device destination. Requires the `cuda` feature (a GPU destination cannot
/// be filled without it); refuses null/zero-size destinations.
fn gds_read_mmap_fallback(
    path: &Path,
    gpu_buf: *mut c_void,
    offset: u64,
    size: u64,
) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};

    // Refuse loudly rather than "succeeding" without transferring anything —
    // the previous no-op left the destination buffer untouched (garbage on
    // the device) while returning Ok.
    if gpu_buf.is_null() || size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "gds_read: destination device buffer must be non-null and size must be > 0",
        ));
    }

    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;

    let mut host_buf = vec![0u8; size as usize];
    file.read_exact(&mut host_buf)?;

    // Copy the host staging buffer into the device destination. `memcpy_htod`
    // calls `ensure_context()` internally, so no extra context management is
    // needed here (mirrors disaggregated::checkpoint model-load staging).
    #[cfg(feature = "cuda")]
    {
        crate::cuda::inner::memcpy_htod(gpu_buf, host_buf.as_ptr() as *const c_void, size as usize);
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "gds_read mmap fallback: filling a GPU destination requires the cuda feature",
        ))
    }
}

// ---------------------------------------------------------------------------
// Async prefetch hint
// ---------------------------------------------------------------------------

/// Hint to the OS / GDS driver to prefetch file data for upcoming reads.
/// On Linux, uses posix_fadvise(WILLNEED). On GDS, uses cuFileReadAsync with
/// a prefetch stream.
pub fn prefetch_hint(path: &Path, offset: u64, size: u64) {
    // Best-effort — failure is silent.
    // Use raw syscall to avoid libc dependency.
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        const POSIX_FADV_WILLNEED: i32 = 3;
        if let Ok(file) = std::fs::File::open(path) {
            unsafe {
                // libc::posix_fadvise equivalent via direct C ABI call
                extern "C" {
                    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
                }
                posix_fadvise(
                    file.as_raw_fd(),
                    offset as i64,
                    size as i64,
                    POSIX_FADV_WILLNEED,
                );
            }
        }
    }
    let _ = (path, offset, size); // suppress unused warnings on non-Linux
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

/// Read file data into a buffer. Uses GDS if available, mmap fallback otherwise.
/// `path_ptr`: C string path to file
/// `buf_ptr`: destination buffer (host or device pointer)
/// `offset`: byte offset in file
/// `size`: bytes to read
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_data_gds_read(
    path_ptr: i64,
    buf_ptr: i64,
    offset: i64,
    size: i64,
    stream: i64,
) -> i64 {
    if path_ptr == 0 || buf_ptr == 0 { return -1; }

    let path_cstr = unsafe { std::ffi::CStr::from_ptr(path_ptr as *const std::os::raw::c_char) };
    let path = match path_cstr.to_str() {
        Ok(s) => Path::new(s),
        Err(_) => return -1,
    };

    match gds_read(path, buf_ptr as *mut c_void, offset as u64, size as u64, stream) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Check if GDS is available. Returns 1 if available, 0 if not.
#[no_mangle]
pub extern "C" fn nsl_data_gds_available() -> i64 {
    if is_gds_available() { 1 } else { 0 }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use std::io::Write;

    /// A non-null but never-dereferenced sentinel destination for tests that
    /// exercise the guards / I/O errors before any device copy happens.
    const FAKE_DST: *mut c_void = 0x1000 as *mut c_void;

    #[test]
    fn test_gds_not_available_by_default() {
        // Unless NSL_GDS_ENABLED=1, GDS should not be available
        assert!(!is_gds_available());
    }

    #[test]
    fn mmap_fallback_refuses_null_destination() {
        // A null destination previously "succeeded" while copying nothing.
        let err = gds_read_mmap_fallback(Path::new("unused"), std::ptr::null_mut(), 0, 16)
            .expect_err("null destination must refuse");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn mmap_fallback_refuses_zero_size() {
        let err = gds_read_mmap_fallback(Path::new("unused"), FAKE_DST, 0, 0)
            .expect_err("zero size must refuse");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_gds_read_nonexistent_file() {
        // Non-null destination so the guard passes and the file-open failure
        // (not the null guard) is what produces the error.
        let result = gds_read(Path::new("/nonexistent/file.bin"), FAKE_DST, 0, 100, 0);
        assert!(result.is_err());
    }

    /// Without the `cuda` feature, filling a GPU destination is impossible, so
    /// the fallback must refuse loudly after the host read rather than
    /// pretending to have transferred the bytes.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn mmap_fallback_refuses_without_cuda() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("nsl_test_gds_nocuda_{}.bin", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(b"sixteen__bytes!!").unwrap();
        }
        let result = gds_read_mmap_fallback(&path, FAKE_DST, 0, 16);
        std::fs::remove_file(&path).ok();
        let err = result.expect_err("fallback must refuse a GPU destination without cuda");
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }

    /// GPU correctness: the fallback stages the file bytes into a real device
    /// buffer. Copy them back and confirm they match. Requires a GPU; run with
    /// `--features cuda -- --include-ignored`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn mmap_fallback_h2d_copies_bytes_to_device() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("nsl_test_gds_h2d_{}.bin", std::process::id()));
        let payload: [u8; 16] = *b"GDS_H2D_PAYLOAD!";
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&payload).unwrap();
        }

        let dev = crate::cuda::inner::alloc_device(payload.len());
        let result = gds_read_mmap_fallback(&path, dev, 0, payload.len() as u64);

        let mut back = [0u8; 16];
        crate::cuda::inner::memcpy_dtoh(
            back.as_mut_ptr() as *mut c_void,
            dev as *const c_void,
            payload.len(),
        );
        crate::cuda::inner::free_device(dev);
        std::fs::remove_file(&path).ok();

        assert!(result.is_ok(), "H2D fallback must succeed: {result:?}");
        assert_eq!(back, payload, "device buffer must hold the file bytes after H2D");
    }
}
