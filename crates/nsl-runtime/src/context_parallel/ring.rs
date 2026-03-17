/// Next rank in the ring (send destination).
pub fn ring_next(rank: usize, ring_size: usize) -> usize {
    (rank + 1) % ring_size
}

/// Previous rank in the ring (recv source).
pub fn ring_prev(rank: usize, ring_size: usize) -> usize {
    (rank + ring_size - 1) % ring_size
}

/// Double-buffer index for a ring pass.
pub fn buffer_index(pass: usize) -> usize {
    pass % 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_next_prev() {
        assert_eq!(ring_next(2, 4), 3);
        assert_eq!(ring_next(3, 4), 0); // wraps
        assert_eq!(ring_prev(0, 4), 3); // wraps
        assert_eq!(ring_prev(2, 4), 1);
    }

    #[test]
    fn test_double_buffer_index() {
        assert_eq!(buffer_index(0), 0);
        assert_eq!(buffer_index(1), 1);
        assert_eq!(buffer_index(2), 0); // wraps
        assert_eq!(buffer_index(3), 1);
    }
}
