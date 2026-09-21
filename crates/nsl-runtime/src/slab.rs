//! M36: Slab memory allocator for compile-time planned tensor memory.
//!
//! A slab is a single contiguous allocation. Tensors are sub-allocated as
//! pointer offsets into the slab, with `slab_managed = 1` so `nsl_tensor_free`
//! does not attempt to free them individually.
//!
//! Two allocation modes:
//! - CPU slab: `nsl_slab_alloc` uses aligned heap allocation (zeroed)
//! - GPU slab: `nsl_gpu_slab_init` uses device memory via cuMemAlloc

/// Slab alignment — matches SLAB_ALIGNMENT in memory_planner.rs (256 bytes for GPU).
const SLAB_ALIGN: usize = 256;

// ---------------------------------------------------------------------------
// CPU slab (heap memory)
// ---------------------------------------------------------------------------

/// Allocate a contiguous memory slab (zeroed) with 256-byte alignment.
/// Returns base pointer as i64.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_slab_alloc(size_bytes: i64) -> i64 {
    if size_bytes <= 0 {
        return 0;
    }
    let layout = std::alloc::Layout::from_size_align(size_bytes as usize, SLAB_ALIGN)
        .expect("nsl: slab layout overflow");
    // SAFETY: layout has non-zero size (checked above) and valid alignment.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        crate::nsl_log!(ERROR, "nsl", "nsl: slab allocation failed ({} bytes)", size_bytes);
        std::process::abort();
    }
    ptr as i64
}

/// Free a previously allocated CPU slab.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_slab_free(slab_ptr: i64, size_bytes: i64) {
    if slab_ptr == 0 || size_bytes <= 0 {
        return;
    }
    let layout = std::alloc::Layout::from_size_align(size_bytes as usize, SLAB_ALIGN)
        .expect("nsl: slab layout overflow in free");
    unsafe {
        std::alloc::dealloc(slab_ptr as *mut u8, layout);
    }
}

/// Compute a pointer offset into the slab. Returns slab_ptr + offset.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_slab_offset(slab_ptr: i64, offset: i64) -> i64 {
    slab_ptr + offset
}

// ---------------------------------------------------------------------------
// GPU slab (device memory — compile-time planned arena)
// ---------------------------------------------------------------------------

use crate::device_region::Region;

/// This device's GPU slab — base pointer (set by `nsl_gpu_slab_init`) and
/// total size, `(0, 0)` when none is allocated.
///
/// Roadmap A4 step 3c: the base is a device pointer, so it belongs to a
/// device, not to the process. Under `cuda` it lives on that device's
/// [`CudaContext`](crate::cuda::context::CudaContext); in a CPU-only build,
/// where `nsl_gpu_slab_init` cannot allocate anything at all, one
/// process-global region keeps these `extern "C"` rows answering 0 without
/// needing a device to exist.
///
/// Unlike the arena's, none of these readers is hot: the slab is allocated
/// once at program start and freed at exit, and `nsl_gpu_slab_active` /
/// `nsl_gpu_slab_size` are probes.
#[cfg(feature = "cuda")]
fn region() -> &'static Region {
    // NON-FORCING, and that is the whole point. `context::current()` would
    // lazily create the context, and these rows are reached on paths that
    // must not: `nsl_gpu_slab_destroy` is emitted at program exit
    // unconditionally, for CPU-only programs included. A
    // cuda-featured binary running a CPU-only program on a machine with no
    // driver would then abort at teardown where today it no-ops.
    //
    // No context means no device means no region, so an empty one answers
    // every read correctly. Writes are safe through the same door only
    // because both initialisers allocate device memory FIRST — which
    // initialises the context — and publish afterwards; keep that order.
    if crate::cuda::context::initialized() {
        &crate::cuda::context::current().slab
    } else {
        static NO_DEVICE: Region = Region::new();
        &NO_DEVICE
    }
}

#[cfg(not(feature = "cuda"))]
fn region() -> &'static Region {
    static CPU_ONLY: Region = Region::new();
    &CPU_ONLY
}

/// Allocate the GPU memory slab. Called once at program start.
/// `size_bytes` is the total slab size computed by the compile-time memory planner.
/// Returns the slab base pointer as i64 (0 on failure).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_gpu_slab_init(size_bytes: i64) -> i64 {
    if size_bytes <= 0 { return 0; }
    #[cfg(feature = "cuda")]
    {
        let ptr = crate::cuda::inner::alloc_device(size_bytes as usize);
        if ptr.is_null() { return 0; }
        // Zero the slab (tensors expect zero-initialized memory)
        crate::cuda::inner::memset_d8(ptr, size_bytes as usize);
        region().set(ptr as u64, size_bytes as u64);
        return ptr as i64;
    }
    #[cfg(not(feature = "cuda"))]
    { 0 }
}

/// Free the GPU memory slab. Called once at program exit.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_gpu_slab_destroy() {
    let base = region().take();
    if base == 0 { return; }
    #[cfg(feature = "cuda")]
    {
        crate::cuda::inner::free_device(base as *mut std::ffi::c_void);
    }
}

/// Returns 1 if a GPU slab is currently allocated, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_gpu_slab_active() -> i64 {
    if region().active() { 1 } else { 0 }
}

/// Returns the GPU slab total size in bytes (for diagnostics). 0 if no slab.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_gpu_slab_size() -> i64 {
    region().size() as i64
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
