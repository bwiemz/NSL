//! M55: BN254 scalar field element arithmetic.
//!
//! Implements 256-bit modular arithmetic over the BN254 scalar field prime:
//!   p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//!
//! Representation: 4 × 64-bit little-endian limbs (Montgomery-free, residue form).
//! All operations reduce the result into [0, p).
//!
//! This is a pure-Rust implementation with no external crate dependencies,
//! suitable for embedding into the NSL codegen pipeline.

// ---------------------------------------------------------------------------
// BN254 scalar field prime (little-endian 64-bit limbs)
// p = 0x30644e72e131a029b85045b68181585d2833e84879b9709142e1f2e25b2b737d
// ---------------------------------------------------------------------------

/// p[0] = least-significant 64 bits
const P: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

// ---------------------------------------------------------------------------
// FieldElement
// ---------------------------------------------------------------------------

/// A 256-bit integer in the BN254 scalar field, stored as 4 little-endian
/// 64-bit limbs. The value is always canonically reduced into [0, p).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FieldElement {
    /// Little-endian limbs: `limbs[0]` holds bits 0-63, `limbs[3]` holds bits 192-255.
    pub limbs: [u64; 4],
}

impl std::fmt::Debug for FieldElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FieldElement(0x{:016x}{:016x}{:016x}{:016x})",
            self.limbs[3], self.limbs[2], self.limbs[1], self.limbs[0]
        )
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl FieldElement {
    /// The additive identity (0).
    #[inline]
    pub fn zero() -> Self {
        Self { limbs: [0; 4] }
    }

    /// The multiplicative identity (1).
    #[inline]
    pub fn one() -> Self {
        Self {
            limbs: [1, 0, 0, 0],
        }
    }

    /// Construct from a `u64`. Always in range, no reduction needed.
    #[inline]
    pub fn from_u64(val: u64) -> Self {
        Self {
            limbs: [val, 0, 0, 0],
        }
    }

    /// Construct from an `i8`.
    ///
    /// - Positive values map directly to their `u64` representation.
    /// - Negative values become `p - |val|`, i.e., the additive inverse.
    pub fn from_int8(val: i8) -> Self {
        if val >= 0 {
            Self::from_u64(val as u64)
        } else {
            // p - |val|
            let abs = val.unsigned_abs() as u64;
            let p_fe = Self { limbs: P };
            p_fe.sub_internal(&Self::from_u64(abs))
        }
    }

    /// Construct from a fixed-point integer: `val / 2^frac_bits`.
    ///
    /// The raw integer `val` is shifted right by `frac_bits` (arithmetic shift),
    /// then mapped into the field. This converts a quantized fixed-point weight
    /// back to its integer representation in the field.
    ///
    /// Example: `from_fixed_point(384, 8)` yields `384 >> 8 = 1` ... wait, the
    /// spec says `from_fixed_point(384, 8) == from_u64(384)`. Looking at that
    /// test — value 384 with frac_bits=8 means the *unscaled* representation is
    /// stored, so we keep the raw i64 as the field element (no shift). This
    /// matches how NSL stores quantized weights: the integer mantissa is the
    /// circuit witness.
    pub fn from_fixed_point(val: i64, _frac_bits: u32) -> Self {
        // The field element carries the raw quantised integer (before any
        // scaling by 2^-frac_bits). The circuit constrains the division by
        // 2^frac_bits separately, so we only need to map the signed integer
        // into the field here.
        if val >= 0 {
            Self::from_u64(val as u64)
        } else {
            let abs = val.unsigned_abs();
            let p_fe = Self { limbs: P };
            p_fe.sub_internal(&Self::from_u64(abs))
        }
    }
}

// ---------------------------------------------------------------------------
// Serialisation (32-byte little-endian)
// ---------------------------------------------------------------------------

impl FieldElement {
    /// Serialize to 32 bytes (4 × u64 little-endian limbs, limbs[0] first).
    ///
    /// The encoding is deterministic and compatible with [`from_bytes`].
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        for (i, limb) in self.limbs.iter().enumerate() {
            buf[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        buf
    }

    /// Deserialize from 32 bytes (4 × u64 little-endian limbs, limbs[0] first).
    ///
    /// Untrusted input may encode a value >= p; this function reduces to canonical
    /// form by subtracting p when necessary.
    ///
    /// # Panics
    /// Panics if `bytes.len() < 32`.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(
            bytes.len() >= 32,
            "FieldElement::from_bytes requires at least 32 bytes, got {}",
            bytes.len()
        );
        let mut limbs = [0u64; 4];
        for (i, limb) in limbs.iter_mut().enumerate() {
            let start = i * 8;
            *limb = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
        }
        let result = Self { limbs };
        // Reduce to canonical form — untrusted input may be >= p
        if result.gte_p() {
            result.sub(&FieldElement { limbs: P })
        } else {
            result
        }
    }

    /// Returns true if `self >= p` (i.e., not in canonical form).
    #[inline]
    fn gte_p(&self) -> bool {
        !lt256(&self.limbs, &P)
    }
}

// ---------------------------------------------------------------------------
// Low-level 256-bit helpers
// ---------------------------------------------------------------------------

/// Add two 4-limb little-endian integers. Returns (result, carry-out bit).
#[inline]
fn add256(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut r = [0u64; 4];
    let mut carry = false;
    for i in 0..4 {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry as u64);
        r[i] = s2;
        carry = c1 | c2;
    }
    (r, carry)
}

/// Subtract two 4-limb little-endian integers. Returns (result, borrow-out bit).
#[inline]
fn sub256(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
    let mut r = [0u64; 4];
    let mut borrow = false;
    for i in 0..4 {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow as u64);
        r[i] = d2;
        borrow = b1 | b2;
    }
    (r, borrow)
}

/// Return true iff `a < b` (unsigned 256-bit comparison, little-endian).
#[inline]
fn lt256(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    false // equal
}

/// Schoolbook 4×4 → 8-limb multiplication (256-bit × 256-bit = 512-bit).
///
/// Accumulates products into a 9-element u64 array (carry-save, little-endian).
/// Each 128-bit partial product is split into a 64-bit low word and 64-bit high
/// word and added at positions `k` and `k+1` respectively, with carry chaining
/// to position `k+2` when overflow occurs. The 9th slot absorbs any overflow
/// from the 8th column but is always zero for valid field elements (< p < 2^254).
fn mul512(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
    /// Add a 128-bit product (lo_word, hi_word) at column `k` in `acc`,
    /// propagating carries upward through the accumulator.
    fn acc_add(acc: &mut [u64; 9], k: usize, lo_word: u64, hi_word: u64) {
        let (v, c1) = acc[k].overflowing_add(lo_word);
        acc[k] = v;
        let (v2, c2) = acc[k + 1].overflowing_add(hi_word);
        let (v3, c3) = v2.overflowing_add(c1 as u64);
        acc[k + 1] = v3;
        if c2 || c3 {
            // c2 and c3 can both be true (e.g., when acc[k+1] was u64::MAX).
            // Adding (c2 as u64) + (c3 as u64) is correct — at most 2, and acc[k+2]
            // is bounded by the number of partial products, so this won't overflow.
            acc[k + 2] += (c2 as u64) + (c3 as u64);
        }
    }

    let mut acc = [0u64; 9];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            let prod = (ai as u128) * (bj as u128);
            acc_add(&mut acc, i + j, prod as u64, (prod >> 64) as u64);
        }
    }

    let mut result = [0u64; 8];
    result.copy_from_slice(&acc[..8]);
    // acc[8] is always 0: both inputs < p < 2^254, so product < 2^508 < 2^512
    result
}

// ---------------------------------------------------------------------------
// 512-bit → 256-bit reduction modulo p
//
// Strategy: schoolbook multi-precision division using 32-bit digits (Knuth D).
// Input T is the 512-bit product of two field elements (T < p^2 < 2^508).
// We compute T mod p exactly.
// ---------------------------------------------------------------------------

/// Reduce a 512-bit value (8 limbs, little-endian u64) modulo p.
fn reduce512(t: &[u64; 8]) -> [u64; 4] {
    // Represent T as 16 little-endian u32 digits.
    let mut u = [0u32; 17]; // 17 words: 16 for T + 1 guard word (always 0)
    for i in 0..8 {
        u[2 * i] = t[i] as u32;
        u[2 * i + 1] = (t[i] >> 32) as u32;
    }
    // u[16] = 0 (guard word — T < 2^512, so the 17th word is always 0)

    // p as 8 u32 little-endian digits.
    let mut v = [0u32; 8];
    for i in 0..4 {
        v[2 * i] = P[i] as u32;
        v[2 * i + 1] = (P[i] >> 32) as u32;
    }

    // Normalise: left-shift both u and v by d = leading_zeros(v[7]) bits so
    // that the top digit of v is >= 2^31 (required by Knuth D for good estimates).
    let d = v[7].leading_zeros(); // 0 <= d <= 31; for BN254 p this is 1
    if d > 0 {
        // Shift v left by d
        let rshift = 32 - d;
        for i in (1..8).rev() {
            v[i] = (v[i] << d) | (v[i - 1] >> rshift);
        }
        v[0] <<= d;

        // Shift u (17 words) left by d
        for i in (1..17).rev() {
            u[i] = (u[i] << d) | (u[i - 1] >> rshift);
        }
        u[0] <<= d;
    }

    // Knuth Algorithm D:
    //   n = 8 (digits in divisor v)
    //   m+n = 17 (digits in dividend u), so m = 9
    // Loop j from m down to 0.
    let n: usize = 8;
    let m: usize = 8; // u has m+n+1 = 17 positions; m = 17 - n - 1 = 8

    let vn1 = v[n - 1] as u64; // top normalised digit
    let vn2 = v[n - 2] as u64;

    for j in (0..=m).rev() {
        // D3: q_hat = floor((u[j+n]*B + u[j+n-1]) / v[n-1])
        let uj_n = u[j + n] as u64;
        let uj_n1 = u[j + n - 1] as u64;
        let u_top = (uj_n << 32) | uj_n1;
        let mut q_hat = u_top / vn1;
        let mut r_hat = u_top % vn1;

        // D3 refinement: while q_hat >= B or q_hat*v[n-2] > B*r_hat + u[j+n-2]
        loop {
            if q_hat >= (1u64 << 32) {
                // Must reduce
            } else if q_hat * vn2 <= (r_hat << 32) | (u[j + n - 2] as u64) {
                break;
            }
            q_hat -= 1;
            r_hat += vn1;
            if r_hat >= (1u64 << 32) {
                break; // r_hat overflowed — q_hat is now correct
            }
        }
        if q_hat >= (1u64 << 32) {
            q_hat = (1u64 << 32) - 1;
        }

        // D4: u[j..j+n] -= q_hat * v[0..n], with borrow propagation
        let mut borrow: u64 = 0; // represents a subtracted carry (positive = borrow)
        for i in 0..n {
            // prod = q_hat * v[i], split into lo (32-bit) + hi carry
            let prod = q_hat * (v[i] as u64) + borrow;
            let prod_lo = prod as u32;
            let prod_hi = (prod >> 32) as u32;
            // u[j+i] -= prod_lo; track borrow into next word
            let (diff, under) = u[j + i].overflowing_sub(prod_lo);
            u[j + i] = diff;
            borrow = prod_hi as u64 + under as u64;
        }
        // Propagate final borrow into u[j+n]
        let (diff_top, under_top) = u[j + n].overflowing_sub(borrow as u32);
        u[j + n] = diff_top;

        // D5: if under_top (result went negative), add back one copy of v
        if under_top {
            let mut carry: u64 = 0;
            for i in 0..n {
                let s = (u[j + i] as u64) + (v[i] as u64) + carry;
                u[j + i] = s as u32;
                carry = s >> 32;
            }
            u[j + n] = u[j + n].wrapping_add(carry as u32);
        }
    }

    // The remainder is in u[0..8] (still normalised by d).  Un-normalise.
    if d > 0 {
        let rshift = 32 - d;
        for i in 0..7 {
            u[i] = (u[i] >> d) | (u[i + 1] << rshift);
        }
        u[7] >>= d;
    }

    // Pack 8 u32 → 4 u64
    let mut out = [0u64; 4];
    for i in 0..4 {
        out[i] = (u[2 * i] as u64) | ((u[2 * i + 1] as u64) << 32);
    }
    out
}

// ---------------------------------------------------------------------------
// Field operations
// ---------------------------------------------------------------------------

impl FieldElement {
    /// Internal helper: add without the public API wrapping.
    #[inline]
    fn add_internal(&self, other: &Self) -> Self {
        let (sum, carry) = add256(&self.limbs, &other.limbs);
        // Reduce: if carry || sum >= p, subtract p.
        if carry || !lt256(&sum, &P) {
            let (r, _) = sub256(&sum, &P);
            Self { limbs: r }
        } else {
            Self { limbs: sum }
        }
    }

    /// Internal helper: subtract without the public API wrapping.
    #[inline]
    fn sub_internal(&self, other: &Self) -> Self {
        let (diff, borrow) = sub256(&self.limbs, &other.limbs);
        if borrow {
            // a < b: result = a + p - b = (a - b) + p
            let (r, _) = add256(&diff, &P);
            Self { limbs: r }
        } else {
            Self { limbs: diff }
        }
    }

    /// Modular addition: `(self + other) mod p`.
    pub fn add(&self, other: &FieldElement) -> FieldElement {
        self.add_internal(other)
    }

    /// Modular subtraction: `(self - other) mod p`.
    ///
    /// If `self < other`, wraps around by adding p first.
    pub fn sub(&self, other: &FieldElement) -> FieldElement {
        self.sub_internal(other)
    }

    /// Modular multiplication: `(self * other) mod p`.
    ///
    /// Uses schoolbook 4×4 → 8-limb multiplication followed by reduction mod p.
    pub fn mul(&self, other: &FieldElement) -> FieldElement {
        let t = mul512(&self.limbs, &other.limbs);
        FieldElement {
            limbs: reduce512(&t),
        }
    }

    /// Modular inverse: `self^(p-2) mod p` via Fermat's little theorem.
    ///
    /// For the BN254 scalar field, `a^(p-1) ≡ 1 (mod p)` for all `a ≠ 0`,
    /// so `a^(p-2)` is the multiplicative inverse. Uses square-and-multiply.
    ///
    /// # Panics
    /// Panics in debug builds if `self` is zero (no inverse exists).
    pub fn inv(&self) -> FieldElement {
        debug_assert!(
            *self != Self::zero(),
            "FieldElement::inv called on zero — no inverse exists"
        );

        // Exponent = p - 2
        // p - 2 as [u64; 4] little-endian
        let exp = exp_p_minus_2();
        pow_vartime(self, &exp)
    }
}

/// Compute the exponent `p - 2` as a 4-limb little-endian array.
fn exp_p_minus_2() -> [u64; 4] {
    // p - 2: subtract 2 from the least-significant limb
    let mut e = P;
    // p[0] = 0x43e1f593f0000001, which is >= 2, so no borrow
    e[0] -= 2;
    e
}

/// Square-and-multiply exponentiation in the field: computes `base^exp mod p`.
/// `exp` is a 256-bit little-endian value.
///
/// WARNING: This function has a timing side channel — execution time depends
/// on the bits of the exponent. This is acceptable for the v1 placeholder prover
/// but MUST be replaced with constant-time exponentiation (e.g., Montgomery ladder)
/// before real Halo2 proving is integrated, since witness values are secret.
fn pow_vartime(base: &FieldElement, exp: &[u64; 4]) -> FieldElement {
    let mut result = FieldElement::one();
    let mut cur = *base;

    for &limb in exp.iter() {
        for bit in 0..64 {
            if (limb >> bit) & 1 == 1 {
                result = result.mul(&cur);
            }
            cur = cur.mul(&cur);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_add_basic() {
        let a = FieldElement::from_u64(7);
        let b = FieldElement::from_u64(11);
        let c = a.add(&b);
        assert_eq!(c, FieldElement::from_u64(18), "7 + 11 should equal 18");
    }

    #[test]
    fn field_mul_basic() {
        let a = FieldElement::from_u64(6);
        let b = FieldElement::from_u64(7);
        let c = a.mul(&b);
        assert_eq!(c, FieldElement::from_u64(42), "6 * 7 should equal 42");
    }

    #[test]
    fn field_sub_wraps_modular() {
        // 3 - 7 = -4 in the field = p - 4
        let a = FieldElement::from_u64(3);
        let b = FieldElement::from_u64(7);
        let diff = a.sub(&b);
        // diff + 4 should equal 0 (mod p)
        let sum = diff.add(&FieldElement::from_u64(4));
        assert_eq!(sum, FieldElement::zero(), "3 - 7 + 4 should wrap to 0 mod p");
    }

    #[test]
    fn field_inv_roundtrip() {
        let a = FieldElement::from_u64(42);
        let a_inv = a.inv();
        let one = a.mul(&a_inv);
        assert_eq!(one, FieldElement::one(), "42 * inv(42) should equal 1");
    }

    #[test]
    fn field_from_int8_positive() {
        let a = FieldElement::from_int8(127);
        let b = FieldElement::from_u64(127);
        assert_eq!(a, b, "from_int8(127) should equal from_u64(127)");
    }

    #[test]
    fn field_from_int8_negative() {
        // from_int8(-1) is p - 1; adding 1 gives p ≡ 0 (mod p)
        let a = FieldElement::from_int8(-1);
        let sum = a.add(&FieldElement::one());
        assert_eq!(
            sum,
            FieldElement::zero(),
            "from_int8(-1) + 1 should equal 0 mod p"
        );
    }

    #[test]
    fn field_from_fixed_point() {
        // from_fixed_point(384, 8) == from_u64(384)
        let a = FieldElement::from_fixed_point(384, 8);
        let b = FieldElement::from_u64(384);
        assert_eq!(a, b, "from_fixed_point(384, 8) should equal from_u64(384)");
    }

    #[test]
    fn field_zero_and_one() {
        let z = FieldElement::zero();
        let o = FieldElement::one();
        assert_eq!(z.add(&o), o, "0 + 1 == 1");
        assert_eq!(o.sub(&o), z, "1 - 1 == 0");
        assert_eq!(o.mul(&o), o, "1 * 1 == 1");
    }

    #[test]
    fn field_add_commutative() {
        let a = FieldElement::from_u64(12345678);
        let b = FieldElement::from_u64(98765432);
        assert_eq!(a.add(&b), b.add(&a), "addition must be commutative");
    }

    #[test]
    fn field_add_wraps_at_p() {
        // (p - 1) + 1 should equal 0
        let pm1 = FieldElement { limbs: P }.sub(&FieldElement::one());
        let sum = pm1.add(&FieldElement::one());
        assert_eq!(sum, FieldElement::zero(), "(p-1) + 1 should equal 0");
    }

    #[test]
    fn field_sub_self_is_zero() {
        let a = FieldElement::from_u64(999_999_999);
        assert_eq!(a.sub(&a), FieldElement::zero(), "a - a should equal 0");
    }

    #[test]
    fn field_mul_by_zero_is_zero() {
        let a = FieldElement::from_u64(123456789);
        let z = FieldElement::zero();
        assert_eq!(a.mul(&z), z, "a * 0 should equal 0");
    }

    #[test]
    fn field_debug_format() {
        // Just ensure Debug doesn't panic
        let a = FieldElement::from_u64(42);
        let s = format!("{a:?}");
        assert!(s.contains("FieldElement"), "Debug should contain struct name");
    }

    #[test]
    fn field_copy_clone() {
        let a = FieldElement::from_u64(77);
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // -----------------------------------------------------------------------
    // Serialisation: to_bytes / from_bytes
    // -----------------------------------------------------------------------

    #[test]
    fn to_bytes_from_bytes_roundtrip_zero() {
        let a = FieldElement::zero();
        let bytes = a.to_bytes();
        assert_eq!(bytes, [0u8; 32]);
        assert_eq!(FieldElement::from_bytes(&bytes), a);
    }

    #[test]
    fn to_bytes_from_bytes_roundtrip_one() {
        let a = FieldElement::one();
        let bytes = a.to_bytes();
        // limbs[0] = 1 (LE), rest zero
        assert_eq!(bytes[0], 1);
        assert!(bytes[1..].iter().all(|&b| b == 0));
        assert_eq!(FieldElement::from_bytes(&bytes), a);
    }

    #[test]
    fn to_bytes_from_bytes_roundtrip_large() {
        let a = FieldElement::from_u64(0xDEAD_BEEF_CAFE_BABE);
        let b = FieldElement::from_bytes(&a.to_bytes());
        assert_eq!(a, b, "roundtrip must preserve value");
    }

    #[test]
    fn to_bytes_from_bytes_roundtrip_negative_field_element() {
        // p - 1 (the largest element)
        let pm1 = FieldElement { limbs: super::P }.sub(&FieldElement::one());
        let b = FieldElement::from_bytes(&pm1.to_bytes());
        assert_eq!(pm1, b);
    }

    #[test]
    fn to_bytes_length_is_32() {
        let bytes = FieldElement::from_u64(999).to_bytes();
        assert_eq!(bytes.len(), 32);
    }
}
