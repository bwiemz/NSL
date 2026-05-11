//! Trit-within-byte ordering invariant test.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §2.1.
//! Catches ordering drift in seconds, independent of full HF model load.
//! Asserts against the high-bits-first convention pinned in
//! `crates/nsl-codegen/src/bitnet/PACKED_BYTE_LAYOUT.md` (verified PI.1).

use nsl_codegen::bitnet::pack::{pack_trits, try_unpack_byte, unpack_byte};

/// Hand-constructed byte: trits [-1, 0, +1, +1] in high-bits-first ordering
/// produce byte 0x1A. The canonical wrong answer (low-bits-first) is 0x58.
#[test]
fn pack_known_trits_produces_expected_byte() {
    let trits: [i8; 4] = [-1, 0, 1, 1];
    let packed = pack_trits(&trits);
    // [-1 → 00] [0 → 01] [+1 → 10] [+1 → 10]  at bit positions [7:6][5:4][3:2][1:0]
    // byte = 0b00_01_10_10 = 0x1A.
    // If this fails with 0x58, the implementation has reverted to low-bits-first
    // ordering — check pack.rs against bitnet.cpp's high-bits-first convention.
    let expected_byte: u8 = 0x1A;
    assert_eq!(
        packed, expected_byte,
        "Trit-byte ordering mismatch — got 0x{packed:02X}, expected 0x{expected_byte:02X} \
         (high-bits-first per PACKED_BYTE_LAYOUT.md; verified against bitnet.cpp PI.1)"
    );
}

#[test]
fn unpack_byte_produces_original_trits() {
    let byte: u8 = 0x1A;
    let trits = unpack_byte(byte);
    assert_eq!(trits, [-1i8, 0, 1, 1]);
}

#[test]
fn unpack_byte_with_invalid_encoding_returns_error() {
    // 0b11 is the invalid 2-bit encoding (per spec §2.1).
    // 0xFF = all-ones = four 0b11 trits — invalid in all four positions.
    let result = try_unpack_byte(0xFF);
    assert!(
        result.is_err(),
        "0xFF contains 0b11 fields which are invalid encodings"
    );

    // 0xC0 has 0b11 in trit[0] (bits [7:6]) and 0b00 elsewhere — also invalid.
    let result = try_unpack_byte(0xC0);
    assert!(
        result.is_err(),
        "0xC0 has 0b11 in trit[0] position — invalid"
    );

    // 0x1A is the canonical valid byte — must NOT error.
    let result = try_unpack_byte(0x1A);
    assert!(
        result.is_ok(),
        "0x1A is the canonical valid test byte and must decode successfully"
    );
}

#[test]
fn pack_unpack_roundtrip_all_valid_inputs() {
    // Sweep all 3^4 = 81 valid (trit_0, trit_1, trit_2, trit_3) tuples.
    // Each must pack-then-unpack identically.
    for t0 in [-1i8, 0, 1] {
        for t1 in [-1i8, 0, 1] {
            for t2 in [-1i8, 0, 1] {
                for t3 in [-1i8, 0, 1] {
                    let trits = [t0, t1, t2, t3];
                    let packed = pack_trits(&trits);
                    let unpacked = unpack_byte(packed);
                    assert_eq!(
                        unpacked, trits,
                        "Roundtrip failed for {trits:?} → 0x{packed:02X} → {unpacked:?}"
                    );
                }
            }
        }
    }
}
