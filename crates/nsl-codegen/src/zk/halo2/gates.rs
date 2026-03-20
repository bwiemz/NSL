//! Halo2 custom gate definitions for ZK inference circuits.
//!
//! Defines the gate configurations used in the Halo2 PLONKish arithmetization
//! of NSL inference graphs. Each gate type corresponds to a common ML operation
//! and encodes the constraint structure needed to verify it in zero knowledge.
//!
//! ## Gate types
//!
//! - [`DotProductGate`]: inner product of two vectors (core of matmul)
//! - [`FixedMulGate`]: fixed-point multiplication with rescaling
//! - [`RequantizeGate`]: accumulator narrowing for layer boundaries
//! - [`LayerNormGate`]: fused layer normalization (mean, variance, scale)
//!
//! ## Cost model
//!
//! Each gate reports its constraint count via `num_constraints()`. This is used
//! by the circuit compiler to estimate the total number of PLONKish rows and
//! to choose the circuit size parameter `k` (where the table has `2^k` rows).

// ---------------------------------------------------------------------------
// DotProductGate
// ---------------------------------------------------------------------------

/// Configuration for a dot-product custom gate.
///
/// Computes: `acc_next = acc_curr + a[i] * b[i]` for each row `i`.
/// For a dot product of length N, uses N rows — one multiplication gate per
/// element pair. The final accumulator wire holds the result.
///
/// In Halo2, this maps to a custom gate with 4 advice columns:
/// `a`, `b`, `acc_curr`, `acc_next`, constrained by:
///   `acc_next - acc_curr - a * b = 0`
#[derive(Debug, Clone)]
pub struct DotProductGate {
    /// Number of element pairs in the dot product.
    pub length: usize,
}

impl DotProductGate {
    /// Create a new dot-product gate for vectors of the given length.
    pub fn new(length: usize) -> Self {
        Self { length }
    }

    /// Number of PLONKish constraints (= number of multiplication rows).
    ///
    /// Each element pair `(a[i], b[i])` requires one constraint row:
    ///   `acc[i+1] = acc[i] + a[i] * b[i]`
    pub fn num_constraints(&self) -> u64 {
        self.length as u64
    }

    /// Number of advice columns required by this gate.
    ///
    /// 4 columns: `a`, `b`, `acc_curr`, `acc_next`.
    pub fn num_advice_columns(&self) -> usize {
        4
    }
}

// ---------------------------------------------------------------------------
// FixedMulGate
// ---------------------------------------------------------------------------

/// Configuration for fixed-point multiplication.
///
/// Computes: `a * b = result * 2^frac_bits + remainder`
///
/// Constrains:
///   1. `a * b - result * scale - remainder = 0` (where `scale = 2^frac_bits`)
///   2. `0 <= remainder < 2^frac_bits` (range check)
///
/// This gate implements the rescaling step after integer multiplication in
/// quantized inference. The range check on the remainder ensures that the
/// division-by-power-of-two is exact (no cheating on rounding).
#[derive(Debug, Clone)]
pub struct FixedMulGate {
    /// Number of fractional bits in the fixed-point representation.
    pub frac_bits: u32,
}

impl FixedMulGate {
    /// Create a new fixed-point multiplication gate.
    pub fn new(frac_bits: u32) -> Self {
        Self { frac_bits }
    }

    /// Number of PLONKish constraints.
    ///
    /// 2 constraints:
    ///   1. Multiplication + rescale: `a * b = result * 2^frac + remainder`
    ///   2. Range check on remainder: `0 <= remainder < 2^frac`
    pub fn num_constraints(&self) -> u64 {
        2
    }

    /// Number of advice columns required.
    ///
    /// 4 columns: `a`, `b`, `result`, `remainder`.
    pub fn num_advice_columns(&self) -> usize {
        4
    }

    /// The scaling factor: `2^frac_bits`.
    pub fn scale(&self) -> u64 {
        1u64 << self.frac_bits
    }
}

// ---------------------------------------------------------------------------
// RequantizeGate
// ---------------------------------------------------------------------------

/// Configuration for requantization (accumulator narrowing).
///
/// Computes: `out = clamp(round((input - zero_point) * scale), 0, 2^target_bits - 1)`
///
/// Constrains:
///   1. Affine transform: `shifted = input - zero_point`
///   2. Scaled product: `scaled = shifted * scale`
///   3. Range check: `0 <= out < 2^target_bits`
///
/// This gate narrows a wide accumulator (e.g., 32-bit from matmul) back to a
/// smaller bit-width (e.g., 8-bit) for the next layer's lookup table.
#[derive(Debug, Clone)]
pub struct RequantizeGate {
    /// Target bit-width after requantization.
    pub target_bits: u32,
}

impl RequantizeGate {
    /// Create a new requantization gate.
    pub fn new(target_bits: u32) -> Self {
        Self { target_bits }
    }

    /// Number of PLONKish constraints.
    ///
    /// 3 constraints:
    ///   1. Subtraction of zero point: `shifted = input - zero_point`
    ///   2. Multiplication by scale factor: `scaled = shifted * scale`
    ///   3. Range check on output: `0 <= out < 2^target_bits`
    pub fn num_constraints(&self) -> u64 {
        3
    }

    /// Number of advice columns required.
    ///
    /// 4 columns: `input`, `shifted`, `scaled`, `out`.
    pub fn num_advice_columns(&self) -> usize {
        4
    }

    /// Maximum value the output can take: `2^target_bits - 1`.
    pub fn max_output(&self) -> u64 {
        (1u64 << self.target_bits) - 1
    }
}

// ---------------------------------------------------------------------------
// LayerNormGate
// ---------------------------------------------------------------------------

/// Configuration for layer normalization.
///
/// Fuses the following operations:
///   1. **Mean** (free — sum of N additions, no mul gates)
///   2. **Variance**: N squaring multiplications
///   3. **Inverse square root**: 1 lookup table evaluation
///   4. **Scale**: N fixed-point multiplications
///
/// Total constraint cost: `N` (variance squares) + `1` (rsqrt lookup) + `N` (scale muls)
/// = `2*N + 1`.
///
/// This fused gate avoids materializing intermediate results as separate
/// circuit regions, saving advice columns and reducing the total row count.
#[derive(Debug, Clone)]
pub struct LayerNormGate {
    /// Dimension of the normalization (number of elements in the vector).
    pub dim: usize,
}

impl LayerNormGate {
    /// Create a new layer normalization gate.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Number of PLONKish constraints.
    ///
    /// - N squaring constraints (variance computation)
    /// - 1 lookup constraint (rsqrt)
    /// - N fixed-mul constraints (scale by normalized weight)
    /// - Total: `2 * dim + 1`
    pub fn num_constraints(&self) -> u64 {
        2 * self.dim as u64 + 1
    }

    /// Number of advice columns required.
    ///
    /// 6 columns: `input`, `mean`, `diff`, `diff_sq`, `rsqrt`, `output`.
    pub fn num_advice_columns(&self) -> usize {
        6
    }
}

// ---------------------------------------------------------------------------
// GateType — enum wrapper for polymorphic gate handling
// ---------------------------------------------------------------------------

/// Enum wrapping all custom gate types for polymorphic constraint counting.
#[derive(Debug, Clone)]
pub enum GateType {
    /// Dot-product custom gate.
    DotProduct(DotProductGate),
    /// Fixed-point multiplication gate.
    FixedMul(FixedMulGate),
    /// Requantization (accumulator narrowing) gate.
    Requantize(RequantizeGate),
    /// Fused layer normalization gate.
    LayerNorm(LayerNormGate),
}

impl GateType {
    /// Total number of constraints for this gate instance.
    pub fn num_constraints(&self) -> u64 {
        match self {
            GateType::DotProduct(g) => g.num_constraints(),
            GateType::FixedMul(g) => g.num_constraints(),
            GateType::Requantize(g) => g.num_constraints(),
            GateType::LayerNorm(g) => g.num_constraints(),
        }
    }

    /// Number of advice columns required by this gate.
    pub fn num_advice_columns(&self) -> usize {
        match self {
            GateType::DotProduct(g) => g.num_advice_columns(),
            GateType::FixedMul(g) => g.num_advice_columns(),
            GateType::Requantize(g) => g.num_advice_columns(),
            GateType::LayerNorm(g) => g.num_advice_columns(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_gate_constraint_count() {
        let gate = DotProductGate::new(768);
        assert_eq!(gate.num_constraints(), 768);
    }

    #[test]
    fn dot_product_gate_zero_length() {
        let gate = DotProductGate::new(0);
        assert_eq!(gate.num_constraints(), 0);
    }

    #[test]
    fn dot_product_gate_advice_columns() {
        let gate = DotProductGate::new(128);
        assert_eq!(gate.num_advice_columns(), 4);
    }

    #[test]
    fn fixed_mul_gate_constraint_count() {
        let gate = FixedMulGate::new(8);
        assert_eq!(gate.num_constraints(), 2);
    }

    #[test]
    fn fixed_mul_gate_scale() {
        let gate = FixedMulGate::new(8);
        assert_eq!(gate.scale(), 256);
        let gate16 = FixedMulGate::new(16);
        assert_eq!(gate16.scale(), 65536);
    }

    #[test]
    fn fixed_mul_gate_advice_columns() {
        let gate = FixedMulGate::new(8);
        assert_eq!(gate.num_advice_columns(), 4);
    }

    #[test]
    fn requantize_gate_constraint_count() {
        let gate = RequantizeGate::new(8);
        assert_eq!(gate.num_constraints(), 3);
    }

    #[test]
    fn requantize_gate_max_output() {
        let gate = RequantizeGate::new(8);
        assert_eq!(gate.max_output(), 255);
        let gate4 = RequantizeGate::new(4);
        assert_eq!(gate4.max_output(), 15);
    }

    #[test]
    fn requantize_gate_advice_columns() {
        let gate = RequantizeGate::new(8);
        assert_eq!(gate.num_advice_columns(), 4);
    }

    #[test]
    fn layer_norm_gate_constraint_count() {
        let gate = LayerNormGate::new(768);
        // 2 * 768 + 1 = 1537
        assert_eq!(gate.num_constraints(), 1537);
    }

    #[test]
    fn layer_norm_gate_small_dim() {
        let gate = LayerNormGate::new(1);
        assert_eq!(gate.num_constraints(), 3); // 2*1 + 1
    }

    #[test]
    fn layer_norm_gate_advice_columns() {
        let gate = LayerNormGate::new(128);
        assert_eq!(gate.num_advice_columns(), 6);
    }

    #[test]
    fn gate_type_enum_constraints() {
        let gates: Vec<GateType> = vec![
            GateType::DotProduct(DotProductGate::new(100)),
            GateType::FixedMul(FixedMulGate::new(8)),
            GateType::Requantize(RequantizeGate::new(8)),
            GateType::LayerNorm(LayerNormGate::new(64)),
        ];
        assert_eq!(gates[0].num_constraints(), 100);
        assert_eq!(gates[1].num_constraints(), 2);
        assert_eq!(gates[2].num_constraints(), 3);
        assert_eq!(gates[3].num_constraints(), 129); // 2*64 + 1
    }

    #[test]
    fn gate_type_enum_advice_columns() {
        assert_eq!(GateType::DotProduct(DotProductGate::new(10)).num_advice_columns(), 4);
        assert_eq!(GateType::FixedMul(FixedMulGate::new(8)).num_advice_columns(), 4);
        assert_eq!(GateType::Requantize(RequantizeGate::new(8)).num_advice_columns(), 4);
        assert_eq!(GateType::LayerNorm(LayerNormGate::new(64)).num_advice_columns(), 6);
    }

    #[test]
    fn total_constraints_for_transformer_layer() {
        // Realistic scenario: a transformer layer with:
        //   - 2 matmuls (Q*K^T and attn*V), each 768-element dot products × 768 rows = 2 × 768
        //   - 768 fixed-mul rescales
        //   - 1 layer norm (dim=768)
        //   - 768 activation lookups (counted separately, not gate constraints)
        let qk_dot = DotProductGate::new(768);
        let av_dot = DotProductGate::new(768);
        let rescales: u64 = 768 * FixedMulGate::new(8).num_constraints();
        let ln = LayerNormGate::new(768);

        let total = qk_dot.num_constraints()
            + av_dot.num_constraints()
            + rescales
            + ln.num_constraints();

        // 768 + 768 + 768*2 + 2*768+1 = 768 + 768 + 1536 + 1537 = 4609
        assert_eq!(total, 4609);
    }
}
