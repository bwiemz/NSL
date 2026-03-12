#[cfg(test)]
mod tests {
    use crate::memory::stats;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_tensor_lifecycle_counter_balance() {
        use crate::tensor::{nsl_tensor_zeros, nsl_tensor_add, nsl_tensor_free};
        use crate::list::{nsl_list_new, nsl_list_push, nsl_list_free};

        stats::reset();

        // Create a shape list [4, 4]
        let shape = nsl_list_new();
        nsl_list_push(shape, 4);
        nsl_list_push(shape, 4);

        // Create two tensors
        let a = nsl_tensor_zeros(shape);
        let shape2 = nsl_list_new();
        nsl_list_push(shape2, 4);
        nsl_list_push(shape2, 4);
        let b = nsl_tensor_zeros(shape2);

        // Add them
        let c = nsl_tensor_add(a, b);

        // Free everything
        nsl_tensor_free(a);
        nsl_tensor_free(b);
        nsl_tensor_free(c);
        nsl_list_free(shape);
        nsl_list_free(shape2);

        // Assert counter balance
        let allocs = stats::ALLOC_COUNT.load(Ordering::SeqCst);
        let frees = stats::FREE_COUNT.load(Ordering::SeqCst);
        let alloc_bytes = stats::ALLOC_BYTES.load(Ordering::SeqCst);
        let free_bytes = stats::FREE_BYTES.load(Ordering::SeqCst);

        assert_eq!(
            allocs, frees,
            "CPU alloc/free count mismatch: {} allocs, {} frees ({} leaked)",
            allocs, frees, allocs as isize - frees as isize
        );
        assert_eq!(
            alloc_bytes, free_bytes,
            "CPU alloc/free bytes mismatch: {} allocated, {} freed ({} bytes leaked)",
            alloc_bytes, free_bytes, alloc_bytes as isize - free_bytes as isize
        );
    }

    #[test]
    fn test_stats_counter_reset() {
        stats::reset();
        assert_eq!(stats::ALLOC_COUNT.load(Ordering::SeqCst), 0);
        assert_eq!(stats::FREE_COUNT.load(Ordering::SeqCst), 0);
        assert_eq!(stats::ALLOC_BYTES.load(Ordering::SeqCst), 0);
        assert_eq!(stats::FREE_BYTES.load(Ordering::SeqCst), 0);
        assert_eq!(stats::CUDA_ALLOC_COUNT.load(Ordering::SeqCst), 0);
        assert_eq!(stats::CUDA_FREE_COUNT.load(Ordering::SeqCst), 0);
        assert_eq!(stats::CUDA_ALLOC_BYTES.load(Ordering::SeqCst), 0);
        assert_eq!(stats::CUDA_FREE_BYTES.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_stats_track_alloc_free() {
        stats::reset();

        let ptr = crate::memory::checked_alloc(256);
        assert_eq!(stats::ALLOC_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(stats::ALLOC_BYTES.load(Ordering::SeqCst), 256);
        assert_eq!(stats::FREE_COUNT.load(Ordering::SeqCst), 0);

        unsafe { crate::memory::checked_free(ptr, 256); }
        assert_eq!(stats::FREE_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(stats::FREE_BYTES.load(Ordering::SeqCst), 256);
    }
}
