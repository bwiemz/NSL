//! Bit-exact CPU reference functions for the v1 MLP forward pass.
//!
//! Each function matches the per-op arithmetic that the Verilator-simulated
//! Verilog must produce and produces the same per-element output the Verilator
//! harness reads from the corresponding test-tap port.
//!
//! All additions use `wrapping_add` per precision Pin 1: signed integer
//! addition with two's-complement wraparound on overflow. v1's accumulators
//! have sufficient headroom that overflow doesn't occur for the v1 fixture
//! (see spec §4.6), but the wrapping semantics are pinned in the contract so:
//!   (a) future fixtures that approach the overflow bound produce the same
//!       wrapped result on both hardware and CPU reference;
//!   (b) a future codegen change that accidentally introduces saturation is
//!       caught by the bit-exact gate because the CPU reference's wrapping
//!       semantics no longer match.

use crate::fixture::FixtureFile;

// ---------------------------------------------------------------------------
// Per-op reference functions
// ---------------------------------------------------------------------------

/// Layer 1 matmul: i8 × i8 → i16 (intermediate) → i32 accumulator.
///
/// w1 is stored row-major as `[784, 128]`; indexing `w1[i * n_out + o]`.
pub fn cpu_reference_layer1_matmul(x: &[i8], w1: &[i8]) -> Vec<i32> {
    let (n_in, n_out) = (784usize, 128usize);
    assert_eq!(x.len(), n_in, "expected 784-element input");
    assert_eq!(w1.len(), n_in * n_out, "expected 784*128 W1 weights");

    let mut out = vec![0i32; n_out];
    for o in 0..n_out {
        let mut acc: i32 = 0;
        for i in 0..n_in {
            let prod: i16 = (x[i] as i16) * (w1[i * n_out + o] as i16);
            acc = acc.wrapping_add(prod as i32);
        }
        out[o] = acc;
    }
    out
}

/// Layer 1 bias-add: i32 + i32 → i32.
pub fn cpu_reference_layer1_bias(pre: &[i32], b1: &[i32]) -> Vec<i32> {
    pre.iter()
        .zip(b1)
        .map(|(a, b)| a.wrapping_add(*b))
        .collect()
}

/// Layer 1 ReLU: max(0, x) preserving i32 width.
pub fn cpu_reference_layer1_relu(post: &[i32]) -> Vec<i32> {
    post.iter().map(|x| (*x).max(0)).collect()
}

/// Layer 2 matmul: i32 × i8 (sign-extended to i32) → i64 accumulator.
///
/// w2 is stored row-major as `[128, 10]`; indexing `w2[i * n_out + o]`.
/// The i8 weights are sign-extended to i32 before multiplication (matching
/// the `localparam signed [31:0]` in the emitted Verilog).
pub fn cpu_reference_layer2_matmul(h: &[i32], w2: &[i8]) -> Vec<i64> {
    let (n_in, n_out) = (128usize, 10usize);
    assert_eq!(h.len(), n_in, "expected 128-element hidden layer");
    assert_eq!(w2.len(), n_in * n_out, "expected 128*10 W2 weights");

    let mut out = vec![0i64; n_out];
    for o in 0..n_out {
        let mut acc: i64 = 0;
        for i in 0..n_in {
            let w_i32: i32 = w2[i * n_out + o] as i32; // i8 → i32 sign-extend
            let prod: i64 = (h[i] as i64) * (w_i32 as i64);
            acc = acc.wrapping_add(prod);
        }
        out[o] = acc;
    }
    out
}

/// Layer 2 bias-add: i64 + i64 → i64.
pub fn cpu_reference_layer2_bias(pre: &[i64], b2: &[i64]) -> Vec<i64> {
    pre.iter()
        .zip(b2)
        .map(|(a, b)| a.wrapping_add(*b))
        .collect()
}

/// Layer 2 ReLU: max(0, x) preserving i64 width.
pub fn cpu_reference_layer2_relu(post: &[i64]) -> Vec<i64> {
    post.iter().map(|x| (*x).max(0)).collect()
}

// ---------------------------------------------------------------------------
// Full MLP composite
// ---------------------------------------------------------------------------

/// Intermediate per-op outputs for a single input vector.
///
/// All per-op tap values are recorded so Layer 2 tests can compare against
/// Verilator-simulated tap port values.
#[derive(Debug, Clone)]
pub struct MlpOutput {
    /// Layer 1 matmul output (pre-bias), length 128.
    pub l1_matmul: Vec<i32>,
    /// Layer 1 post-bias output, length 128.
    pub l1_bias: Vec<i32>,
    /// Layer 1 ReLU output (= hidden layer input to L2), length 128.
    pub l1_relu: Vec<i32>,
    /// Layer 2 matmul output (pre-bias), length 10.
    pub l2_matmul: Vec<i64>,
    /// Layer 2 post-bias output, length 10.
    pub l2_bias: Vec<i64>,
    /// Layer 2 ReLU output (= final output), length 10.
    pub l2_relu: Vec<i64>,
}

/// Run the full v1 MLP forward pass on one 784-element i8 input.
pub fn cpu_reference_v1_mlp(x: &[i8], fixture: &FixtureFile) -> MlpOutput {
    let w1 = fixture.blocks[0].as_i8(); // block 0: W1
    let b1 = fixture.blocks[1].as_i32(); // block 1: b1
    let w2 = fixture.blocks[2].as_i8(); // block 2: W2
    let b2 = fixture.blocks[3].as_i64(); // block 3: b2

    let l1_matmul = cpu_reference_layer1_matmul(x, &w1);
    let l1_bias = cpu_reference_layer1_bias(&l1_matmul, &b1);
    let l1_relu = cpu_reference_layer1_relu(&l1_bias);
    let l2_matmul = cpu_reference_layer2_matmul(&l1_relu, &w2);
    let l2_bias = cpu_reference_layer2_bias(&l2_matmul, &b2);
    let l2_relu = cpu_reference_layer2_relu(&l2_bias);

    MlpOutput {
        l1_matmul,
        l1_bias,
        l1_relu,
        l2_matmul,
        l2_bias,
        l2_relu,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer1_matmul_zero_input() {
        let x = vec![0i8; 784];
        let w1 = vec![127i8; 784 * 128];
        let out = cpu_reference_layer1_matmul(&x, &w1);
        assert_eq!(out.len(), 128);
        assert!(out.iter().all(|&v| v == 0), "zero input => zero matmul");
    }

    #[test]
    fn layer1_matmul_single_active_input() {
        // x[0] = 1, all others 0; W1[0, o] = o as i8 (mod 64 to stay in range)
        let mut x = vec![0i8; 784];
        x[0] = 1;
        let mut w1 = vec![0i8; 784 * 128];
        for o in 0..128 {
            w1[0 * 128 + o] = (o % 64) as i8;
        }
        let out = cpu_reference_layer1_matmul(&x, &w1);
        for o in 0..128 {
            assert_eq!(out[o], (o % 64) as i32, "mismatch at output {o}");
        }
    }

    #[test]
    fn layer1_relu_clamps_negatives() {
        let post = vec![-100i32, 0, 100, i32::MIN, i32::MAX];
        let out = cpu_reference_layer1_relu(&post);
        assert_eq!(out, vec![0, 0, 100, 0, i32::MAX]);
    }

    #[test]
    fn layer2_matmul_sign_extension() {
        // w2 = -1 (i8 = 0xFF); h = [1, 0, ...0]; expect acc = -1
        let mut h = vec![0i32; 128];
        h[0] = 1;
        let mut w2 = vec![0i8; 128 * 10];
        w2[0 * 10 + 0] = -1i8;
        let out = cpu_reference_layer2_matmul(&h, &w2);
        assert_eq!(out[0], -1i64, "i8 -1 should sign-extend to i64 -1");
        assert_eq!(out[1], 0i64);
    }

    #[test]
    fn wrapping_add_semantics_preserved() {
        // Verify that wrapping_add is used (not saturating) for bias-add.
        let pre = vec![i32::MAX];
        let bias = vec![1i32];
        let result = cpu_reference_layer1_bias(&pre, &bias);
        assert_eq!(result[0], i32::MIN, "should wrap around, not saturate");
    }
}
