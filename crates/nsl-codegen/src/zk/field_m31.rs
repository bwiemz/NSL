//! Mersenne-31 field element arithmetic (p = 2^31 - 1).
//!
//! Single-limb 32-bit arithmetic, ~10x faster than BN254 for ZK proving.
//! All operations use u64 intermediates to avoid overflow, then reduce mod p.
//!
//! Used by the folding backend for high-throughput proving of large models.

use super::field::Field;

/// The Mersenne prime p = 2^31 - 1 = 2147483647.
const P: u64 = (1u64 << 31) - 1;

/// A field element in the Mersenne-31 field (p = 2^31 - 1).
///
/// Stored as a canonical `u32` in [0, p). All operations reduce to canonical form.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mersenne31Field(pub u32);

impl std::fmt::Debug for Mersenne31Field {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "M31({})", self.0)
    }
}

/// Reduce a u64 value mod p = 2^31 - 1.
///
/// Uses the identity: for p = 2^31 - 1, `x mod p = (x >> 31) + (x & p)`,
/// then a conditional subtraction if the result >= p.
#[inline]
fn reduce(x: u64) -> u32 {
    // First reduction: split into high and low 31-bit halves
    let mut r = (x >> 31) + (x & P);
    // Second reduction (needed when first result >= p, e.g., 0xFFFFFFFF)
    r = (r >> 31) + (r & P);
    // Final conditional subtraction
    if r >= P {
        r -= P;
    }
    r as u32
}

impl Mersenne31Field {
    /// The additive identity (0).
    #[inline]
    pub fn zero() -> Self {
        Self(0)
    }

    /// The multiplicative identity (1).
    #[inline]
    pub fn one() -> Self {
        Self(1)
    }

    /// Construct from a u64 value.
    #[inline]
    pub fn from_u64(val: u64) -> Self {
        Self(reduce(val))
    }

    /// Modular addition.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let sum = self.0 as u64 + other.0 as u64;
        Self(reduce(sum))
    }

    /// Modular subtraction.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        // Add p to avoid underflow
        let diff = self.0 as u64 + P - other.0 as u64;
        Self(reduce(diff))
    }

    /// Modular multiplication.
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        let prod = self.0 as u64 * other.0 as u64;
        Self(reduce(prod))
    }

    /// Modular multiplicative inverse via Fermat's little theorem.
    ///
    /// a^(p-1) = 1 mod p, so a^(p-2) = a^(-1) mod p.
    /// p - 2 = 2^31 - 3 = 2147483645.
    pub fn inv(&self) -> Self {
        debug_assert!(self.0 != 0, "Mersenne31Field::inv called on zero");
        self.pow(P as u32 - 2)
    }

    /// Square-and-multiply exponentiation.
    fn pow(&self, mut exp: u32) -> Self {
        let mut base = *self;
        let mut result = Self::one();
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }
        result
    }

    /// Additive inverse (negation).
    #[inline]
    pub fn neg(&self) -> Self {
        if self.0 == 0 {
            Self(0)
        } else {
            Self(P as u32 - self.0)
        }
    }

    /// Serialize to 4 bytes (little-endian u32).
    pub fn to_bytes(&self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    /// Deserialize from 4 bytes (little-endian u32), reducing mod p.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= 4, "Mersenne31Field::from_bytes requires at least 4 bytes");
        let val = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        Self(reduce(val as u64))
    }
}

impl Field for Mersenne31Field {
    #[inline] fn zero() -> Self { Mersenne31Field::zero() }
    #[inline] fn one() -> Self { Mersenne31Field::one() }
    #[inline] fn field_add(&self, other: &Self) -> Self { self.add(other) }
    #[inline] fn field_sub(&self, other: &Self) -> Self { self.sub(other) }
    #[inline] fn field_mul(&self, other: &Self) -> Self { self.mul(other) }
    #[inline] fn field_inv(&self) -> Self { self.inv() }
    #[inline] fn field_neg(&self) -> Self { self.neg() }
    #[inline] fn from_u64(val: u64) -> Self { Mersenne31Field::from_u64(val) }
    fn to_bytes_vec(&self) -> Vec<u8> { self.to_bytes().to_vec() }
    fn field_name() -> &'static str { "Mersenne31" }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_one() {
        assert_eq!(Mersenne31Field::zero().0, 0);
        assert_eq!(Mersenne31Field::one().0, 1);
    }

    #[test]
    fn test_add() {
        let a = Mersenne31Field::from_u64(100);
        let b = Mersenne31Field::from_u64(200);
        assert_eq!(a.add(&b).0, 300);
    }

    #[test]
    fn test_add_wraps() {
        let a = Mersenne31Field::from_u64(P - 1);
        let b = Mersenne31Field::from_u64(2);
        assert_eq!(a.add(&b).0, 1); // (p-1) + 2 = p + 1 ≡ 1 mod p
    }

    #[test]
    fn test_sub() {
        let a = Mersenne31Field::from_u64(300);
        let b = Mersenne31Field::from_u64(100);
        assert_eq!(a.sub(&b).0, 200);
    }

    #[test]
    fn test_sub_wraps() {
        let a = Mersenne31Field::from_u64(1);
        let b = Mersenne31Field::from_u64(3);
        // 1 - 3 = -2 ≡ p - 2 mod p
        assert_eq!(a.sub(&b).0, P as u32 - 2);
    }

    #[test]
    fn test_mul() {
        let a = Mersenne31Field::from_u64(12345);
        let b = Mersenne31Field::from_u64(67890);
        let expected = (12345u64 * 67890) % P;
        assert_eq!(a.mul(&b).0, expected as u32);
    }

    #[test]
    fn test_mul_large() {
        let a = Mersenne31Field::from_u64(P - 1);
        let b = Mersenne31Field::from_u64(P - 1);
        // (p-1)^2 mod p = (-1)^2 mod p = 1
        assert_eq!(a.mul(&b).0, 1);
    }

    #[test]
    fn test_inv_identity() {
        // a * inv(a) == 1 for various values
        for val in [1, 2, 3, 7, 42, 12345, 999999, P as u64 - 1] {
            let a = Mersenne31Field::from_u64(val);
            let a_inv = a.inv();
            let product = a.mul(&a_inv);
            assert_eq!(product.0, 1, "a={}, inv(a)={}, a*inv(a)={}", val, a_inv.0, product.0);
        }
    }

    #[test]
    fn test_neg_identity() {
        // a + neg(a) == 0
        for val in [0, 1, 2, 100, P as u64 - 1] {
            let a = Mersenne31Field::from_u64(val);
            let neg_a = a.neg();
            assert_eq!(a.add(&neg_a).0, 0, "a={}, -a={}", val, neg_a.0);
        }
    }

    #[test]
    fn test_from_u64_reduces() {
        // Values >= p should be reduced
        let a = Mersenne31Field::from_u64(P);
        assert_eq!(a.0, 0);
        let b = Mersenne31Field::from_u64(P + 1);
        assert_eq!(b.0, 1);
        let c = Mersenne31Field::from_u64(2 * P);
        assert_eq!(c.0, 0);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let a = Mersenne31Field::from_u64(42);
        let bytes = a.to_bytes();
        let b = Mersenne31Field::from_bytes(&bytes);
        assert_eq!(a, b);
    }

    #[test]
    fn test_field_trait() {
        // Test through the Field trait interface
        let a = <Mersenne31Field as Field>::from_u64(7);
        let b = <Mersenne31Field as Field>::from_u64(5);
        let sum = a.field_add(&b);
        assert_eq!(sum, <Mersenne31Field as Field>::from_u64(12));
        let prod = a.field_mul(&b);
        assert_eq!(prod, <Mersenne31Field as Field>::from_u64(35));
        assert_eq!(Mersenne31Field::field_name(), "Mersenne31");
    }

    #[test]
    fn test_distributive() {
        // a * (b + c) == a*b + a*c
        let a = Mersenne31Field::from_u64(12345);
        let b = Mersenne31Field::from_u64(67890);
        let c = Mersenne31Field::from_u64(11111);
        let lhs = a.mul(&b.add(&c));
        let rhs = a.mul(&b).add(&a.mul(&c));
        assert_eq!(lhs, rhs);
    }
}
