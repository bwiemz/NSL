#[cfg(test)]
mod tests {
    use crate::memory::stats;
    use std::sync::atomic::Ordering;

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
