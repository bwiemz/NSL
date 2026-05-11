//! Host-side packed/unpacked ternary conversion ops.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §2.
//! Trit-within-byte ordering: HIGH-BITS-FIRST.
//! See `PACKED_BYTE_LAYOUT.md` (verified against bitnet.cpp PI.1).

/// Encode a single trit value as its 2-bit representation.
/// Spec §2.1: `{ -1 → 0b00, 0 → 0b01, +1 → 0b10 }`.
/// Panics on input outside {-1, 0, +1}.
#[inline]
fn encode_trit(t: i8) -> u8 {
    match t {
        -1 => 0b00,
        0 => 0b01,
        1 => 0b10,
        other => panic!("invalid trit value: {other}, expected -1/0/+1"),
    }
}

/// Decode a 2-bit field to its trit value, or error on the invalid 0b11 encoding.
#[inline]
fn decode_trit(bits: u8) -> Result<i8, String> {
    match bits & 0b11 {
        0b00 => Ok(-1),
        0b01 => Ok(0),
        0b10 => Ok(1),
        0b11 => Err(String::from(
            "invalid 2-bit trit encoding 0b11 (reserved per spec §2.1)",
        )),
        _ => unreachable!(),
    }
}

/// Pack four trits into one byte using the HIGH-BITS-FIRST convention.
/// `byte = (trit[0] << 6) | (trit[1] << 4) | (trit[2] << 2) | trit[3]`.
/// trit[0] lives in bits [7:6]; trit[3] lives in bits [1:0].
/// Verified against bitnet.cpp PI.1 (see PACKED_BYTE_LAYOUT.md).
pub fn pack_trits(trits: &[i8; 4]) -> u8 {
    (encode_trit(trits[0]) << 6)
        | (encode_trit(trits[1]) << 4)
        | (encode_trit(trits[2]) << 2)
        | encode_trit(trits[3])
}

/// Unpack one byte into four trits using the HIGH-BITS-FIRST convention.
/// Assumes the byte contains only valid 2-bit encodings; use `try_unpack_byte`
/// to validate. Panics on encountering 0b11 (the reserved encoding).
pub fn unpack_byte(byte: u8) -> [i8; 4] {
    [
        decode_trit(byte >> 6).expect("invalid 0b11 encoding in trit[0]"),
        decode_trit(byte >> 4).expect("invalid 0b11 encoding in trit[1]"),
        decode_trit(byte >> 2).expect("invalid 0b11 encoding in trit[2]"),
        decode_trit(byte).expect("invalid 0b11 encoding in trit[3]"),
    ]
}

/// Unpack one byte, returning an error if any of the four 2-bit fields is
/// the reserved 0b11 encoding. Use this on untrusted input (HF checkpoint
/// loader); use `unpack_byte` on data produced by `pack_trits` (the producer
/// guarantees validity).
pub fn try_unpack_byte(byte: u8) -> Result<[i8; 4], String> {
    Ok([
        decode_trit(byte >> 6)?,
        decode_trit(byte >> 4)?,
        decode_trit(byte >> 2)?,
        decode_trit(byte)?,
    ])
}

/// Pack a slice of trits into a packed byte buffer.
/// Trit count must be a multiple of 4 (caller pads with zeros if needed).
pub fn pack_trit_slice(trits: &[i8]) -> Vec<u8> {
    assert!(
        trits.len() % 4 == 0,
        "trit count must be multiple of 4, got {}",
        trits.len()
    );
    trits
        .chunks_exact(4)
        .map(|chunk| pack_trits(&[chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Unpack a packed byte buffer into a flat trit slice. Uses the unchecked
/// `unpack_byte` (assumes producer-guaranteed validity); use a manual loop
/// over `try_unpack_byte` to validate untrusted buffers.
pub fn unpack_byte_slice(bytes: &[u8]) -> Vec<i8> {
    let mut out = Vec::with_capacity(bytes.len() * 4);
    for &b in bytes {
        out.extend_from_slice(&unpack_byte(b));
    }
    out
}
