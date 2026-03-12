use std::alloc::{alloc, dealloc, Layout};

#[no_mangle]
pub extern "C" fn nsl_alloc(size: i64) -> *mut u8 {
    if size <= 0 {
        return std::ptr::null_mut();
    }
    let layout = Layout::from_size_align(size as usize, 8).unwrap();
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        eprintln!("nsl: out of memory");
        std::process::abort();
    }
    ptr
}

#[no_mangle]
pub extern "C" fn nsl_free(ptr: *mut u8) {
    // We can't easily free with Layout unknown. Use libc free for compatibility.
    // In practice, we'll use Box/Vec for managed allocations.
    if !ptr.is_null() {
        // For now, leak rather than UB. Proper tracking comes with tensor refcounting.
        let _ = ptr;
    }
}

/// Internal helper: allocate uninitialized memory with given size
pub(crate) fn checked_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        eprintln!("nsl: out of memory");
        std::process::abort();
    }
    ptr
}

/// Internal helper: allocate and zero memory
pub(crate) fn checked_alloc_zeroed(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        eprintln!("nsl: out of memory");
        std::process::abort();
    }
    ptr
}

/// Internal helper: reallocate memory
pub(crate) unsafe fn checked_realloc(ptr: *mut u8, old_size: usize, new_size: usize) -> *mut u8 {
    if ptr.is_null() {
        return checked_alloc(new_size);
    }
    let old_layout = Layout::from_size_align(old_size, 8).unwrap();
    let new_ptr = unsafe { std::alloc::realloc(ptr, old_layout, new_size) };
    if new_ptr.is_null() {
        eprintln!("nsl: out of memory");
        std::process::abort();
    }
    new_ptr
}

/// Internal helper: free memory with known size
pub(crate) unsafe fn checked_free(ptr: *mut u8, size: usize) {
    if !ptr.is_null() && size > 0 {
        let layout = Layout::from_size_align(size, 8).unwrap();
        unsafe { dealloc(ptr, layout) };
    }
}

/// Allocation statistics for fuzz testing. Only compiled in test builds.
#[cfg(test)]
pub mod stats {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static FREE_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
    pub static FREE_BYTES: AtomicUsize = AtomicUsize::new(0);

    pub static CUDA_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static CUDA_FREE_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static CUDA_ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
    pub static CUDA_FREE_BYTES: AtomicUsize = AtomicUsize::new(0);

    pub fn reset() {
        for counter in [
            &ALLOC_COUNT, &FREE_COUNT, &ALLOC_BYTES, &FREE_BYTES,
            &CUDA_ALLOC_COUNT, &CUDA_FREE_COUNT, &CUDA_ALLOC_BYTES, &CUDA_FREE_BYTES,
        ] {
            counter.store(0, Ordering::SeqCst);
        }
    }

    pub fn cpu_alloc(size: usize) {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        ALLOC_BYTES.fetch_add(size, Ordering::SeqCst);
    }

    pub fn cpu_free(size: usize) {
        FREE_COUNT.fetch_add(1, Ordering::SeqCst);
        FREE_BYTES.fetch_add(size, Ordering::SeqCst);
    }

    pub fn cuda_alloc(size: usize) {
        CUDA_ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        CUDA_ALLOC_BYTES.fetch_add(size, Ordering::SeqCst);
    }

    pub fn cuda_free(size: usize) {
        CUDA_FREE_COUNT.fetch_add(1, Ordering::SeqCst);
        CUDA_FREE_BYTES.fetch_add(size, Ordering::SeqCst);
    }
}
