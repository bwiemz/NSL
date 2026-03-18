//! M36: Slab memory allocator for compile-time planned tensor memory.
//!
//! A slab is a single contiguous allocation. Tensors are sub-allocated as
//! pointer offsets into the slab, with `owns_data = 0` so `nsl_tensor_free`
//! does not attempt to free them individually.

/// Slab alignment — matches SLAB_ALIGNMENT in memory_planner.rs (256 bytes for GPU).
const SLAB_ALIGN: usize = 256;

/// Allocate a contiguous memory slab (zeroed) with 256-byte alignment.
/// Returns base pointer as i64.
#[no_mangle]
pub extern "C" fn nsl_slab_alloc(size_bytes: i64) -> i64 {
    if size_bytes <= 0 {
        return 0;
    }
    let layout = std::alloc::Layout::from_size_align(size_bytes as usize, SLAB_ALIGN)
        .expect("nsl: slab layout overflow");
    // SAFETY: layout has non-zero size (checked above) and valid alignment.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        eprintln!("nsl: slab allocation failed ({} bytes)", size_bytes);
        std::process::abort();
    }
    ptr as i64
}

/// Free a previously allocated slab.
/// NOTE: Takes size_bytes (unlike spec's 1-param signature) because dealloc
/// requires the allocation Layout for reconstruction.
#[no_mangle]
pub extern "C" fn nsl_slab_free(slab_ptr: i64, size_bytes: i64) {
    if slab_ptr == 0 || size_bytes <= 0 {
        return;
    }
    let layout = std::alloc::Layout::from_size_align(size_bytes as usize, SLAB_ALIGN)
        .expect("nsl: slab layout overflow in free");
    // SAFETY: slab_ptr was returned by nsl_slab_alloc with matching layout.
    unsafe {
        std::alloc::dealloc(slab_ptr as *mut u8, layout);
    }
}

/// Compute a pointer offset into the slab. Returns slab_ptr + offset.
#[no_mangle]
pub extern "C" fn nsl_slab_offset(slab_ptr: i64, offset: i64) -> i64 {
    slab_ptr + offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_alloc_free() {
        let slab = nsl_slab_alloc(4096);
        assert_ne!(slab, 0);
        unsafe {
            *(slab as *mut u8) = 42;
            *((slab as *mut u8).add(4095)) = 99;
            assert_eq!(*(slab as *mut u8), 42);
            assert_eq!(*((slab as *mut u8).add(4095)), 99);
        }
        nsl_slab_free(slab, 4096);
    }

    #[test]
    fn test_slab_alloc_zero_size() {
        assert_eq!(nsl_slab_alloc(0), 0);
        assert_eq!(nsl_slab_alloc(-1), 0);
    }

    #[test]
    fn test_slab_offset() {
        let base = 0x1000_i64;
        assert_eq!(nsl_slab_offset(base, 0), 0x1000);
        assert_eq!(nsl_slab_offset(base, 256), 0x1100);
        assert_eq!(nsl_slab_offset(base, 1024), 0x1400);
    }

    #[test]
    fn test_slab_free_null() {
        nsl_slab_free(0, 100);
        nsl_slab_free(42, 0);
        nsl_slab_free(0, 0);
    }

    #[test]
    fn test_slab_zeroed() {
        let slab = nsl_slab_alloc(1024);
        let data = unsafe { std::slice::from_raw_parts(slab as *const u8, 1024) };
        assert!(data.iter().all(|&b| b == 0), "slab should be zeroed");
        nsl_slab_free(slab, 1024);
    }
}
