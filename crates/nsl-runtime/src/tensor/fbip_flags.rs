//! FBIP-3 flag byte helpers (spec §5).
//!
//! The flag byte is passed into every binary tensor FFI as the third
//! argument. Bit 0 signals the caller relinquishes operand A, bit 1
//! signals B. When a bit is set the runtime may (a) reuse the operand's
//! storage for the output if shape-compatible, or (b) free the operand
//! after using it. When a bit is clear the runtime MUST leave the
//! operand untouched.

/// Bit 0: caller relinquishes operand A.
pub const RELINQUISH_A: u8 = 0x01;
/// Bit 1: caller relinquishes operand B.
pub const RELINQUISH_B: u8 = 0x02;

#[inline]
pub fn relinquish_a(flags: u8) -> bool {
    flags & RELINQUISH_A != 0
}

#[inline]
pub fn relinquish_b(flags: u8) -> bool {
    flags & RELINQUISH_B != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_none_relinquishes_nothing() {
        assert!(!relinquish_a(0));
        assert!(!relinquish_b(0));
    }

    #[test]
    fn flags_a_only() {
        assert!(relinquish_a(RELINQUISH_A));
        assert!(!relinquish_b(RELINQUISH_A));
    }

    #[test]
    fn flags_b_only() {
        assert!(!relinquish_a(RELINQUISH_B));
        assert!(relinquish_b(RELINQUISH_B));
    }

    #[test]
    fn flags_both() {
        let f = RELINQUISH_A | RELINQUISH_B;
        assert!(relinquish_a(f));
        assert!(relinquish_b(f));
    }
}
