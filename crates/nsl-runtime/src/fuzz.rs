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
}
