use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::collections::HashMap;

// Thread-local registry mapping pointers from `nsl_alloc` to their sizes,
// so `nsl_free` can reconstruct the Layout for deallocation.
thread_local! {
    static ALLOC_REGISTRY: RefCell<HashMap<usize, usize>> = RefCell::new(HashMap::new());
}

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
    ALLOC_REGISTRY.with(|reg| {
        reg.borrow_mut().insert(ptr as usize, size as usize);
    });
    ptr
}

// Safety: pointer was allocated by nsl_alloc; caller must not use it after free.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn nsl_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let size = ALLOC_REGISTRY.with(|reg| reg.borrow_mut().remove(&(ptr as usize)));
    match size {
        Some(sz) => {
            let layout = Layout::from_size_align(sz, 8).unwrap();
            unsafe { dealloc(ptr, layout) };
        }
        None => {
            eprintln!("nsl: warning: nsl_free called on untracked pointer {:p}", ptr);
        }
    }
}

/// Free a closure struct allocated by `nsl_alloc`.
///
/// Closure layout: { fn_ptr (8), num_captures (8), captures[] (8 each) }.
/// Captured values are i64 copies (not owned pointers), so no recursive free needed.
#[no_mangle]
pub extern "C" fn nsl_closure_free(ptr: i64) {
    if ptr == 0 { return; }
    nsl_free(ptr as *mut u8);
}

/// Internal helper: allocate uninitialized memory with given size
pub(crate) fn checked_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        // Return a properly aligned dangling pointer instead of null.
        // This is safe to pass to copy_nonoverlapping with count=0.
        // Use align=8 dangling pointer to match our Layout alignment.
        return std::ptr::NonNull::<u64>::dangling().as_ptr() as *mut u8;
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        eprintln!("nsl: out of memory");
        std::process::abort();
    }
    #[cfg(test)]
    stats::cpu_alloc(size);
    ptr
}

/// Internal helper: allocate and zero memory
pub(crate) fn checked_alloc_zeroed(size: usize) -> *mut u8 {
    if size == 0 {
        // Return a properly aligned dangling pointer instead of null.
        // This is safe to pass to copy_nonoverlapping with count=0.
        // Use align=8 dangling pointer to match our Layout alignment.
        return std::ptr::NonNull::<u64>::dangling().as_ptr() as *mut u8;
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        eprintln!("nsl: out of memory");
        std::process::abort();
    }
    #[cfg(test)]
    stats::cpu_alloc(size);
    ptr
}

/// Internal helper: reallocate memory
pub(crate) unsafe fn checked_realloc(ptr: *mut u8, old_size: usize, new_size: usize) -> *mut u8 {
    if ptr.is_null() || old_size == 0 {
        return checked_alloc(new_size);
    }
    if new_size == 0 {
        unsafe { checked_free(ptr, old_size) };
        // Use align=8 dangling pointer to match our Layout alignment.
        return std::ptr::NonNull::<u64>::dangling().as_ptr() as *mut u8;
    }
    let old_layout = Layout::from_size_align(old_size, 8).unwrap();
    #[cfg(test)]
    {
        stats::cpu_free(old_size);
        stats::cpu_alloc(new_size);
    }
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
        #[cfg(test)]
        stats::cpu_free(size);
        let layout = Layout::from_size_align(size, 8).unwrap();
        unsafe { dealloc(ptr, layout) };
    }
}

/// Allocation statistics for fuzz testing. Only compiled in test builds.
/// Uses thread-local counters so parallel tests don't interfere with each other.
#[cfg(test)]
pub mod stats {
    use std::cell::Cell;

    thread_local! {
        static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
        static FREE_COUNT: Cell<usize> = const { Cell::new(0) };
        static ALLOC_BYTES: Cell<usize> = const { Cell::new(0) };
        static FREE_BYTES: Cell<usize> = const { Cell::new(0) };
        static CUDA_ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
        static CUDA_FREE_COUNT: Cell<usize> = const { Cell::new(0) };
        static CUDA_ALLOC_BYTES: Cell<usize> = const { Cell::new(0) };
        static CUDA_FREE_BYTES: Cell<usize> = const { Cell::new(0) };
    }

    pub fn reset() {
        ALLOC_COUNT.with(|c| c.set(0));
        FREE_COUNT.with(|c| c.set(0));
        ALLOC_BYTES.with(|c| c.set(0));
        FREE_BYTES.with(|c| c.set(0));
        CUDA_ALLOC_COUNT.with(|c| c.set(0));
        CUDA_FREE_COUNT.with(|c| c.set(0));
        CUDA_ALLOC_BYTES.with(|c| c.set(0));
        CUDA_FREE_BYTES.with(|c| c.set(0));
    }

    pub fn cpu_alloc(size: usize) {
        ALLOC_COUNT.with(|c| c.set(c.get() + 1));
        ALLOC_BYTES.with(|c| c.set(c.get() + size));
    }

    pub fn cpu_free(size: usize) {
        FREE_COUNT.with(|c| c.set(c.get() + 1));
        FREE_BYTES.with(|c| c.set(c.get() + size));
    }

    pub fn cuda_alloc(size: usize) {
        CUDA_ALLOC_COUNT.with(|c| c.set(c.get() + 1));
        CUDA_ALLOC_BYTES.with(|c| c.set(c.get() + size));
    }

    pub fn cuda_free(size: usize) {
        CUDA_FREE_COUNT.with(|c| c.set(c.get() + 1));
        CUDA_FREE_BYTES.with(|c| c.set(c.get() + size));
    }

    pub fn alloc_count() -> usize { ALLOC_COUNT.with(|c| c.get()) }
    pub fn free_count() -> usize { FREE_COUNT.with(|c| c.get()) }
    pub fn alloc_bytes() -> usize { ALLOC_BYTES.with(|c| c.get()) }
    pub fn free_bytes() -> usize { FREE_BYTES.with(|c| c.get()) }
    pub fn cuda_alloc_count() -> usize { CUDA_ALLOC_COUNT.with(|c| c.get()) }
    pub fn cuda_free_count() -> usize { CUDA_FREE_COUNT.with(|c| c.get()) }
    pub fn cuda_alloc_bytes() -> usize { CUDA_ALLOC_BYTES.with(|c| c.get()) }
    pub fn cuda_free_bytes() -> usize { CUDA_FREE_BYTES.with(|c| c.get()) }
}
