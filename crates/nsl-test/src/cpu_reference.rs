//! Bit-exact CPU reference functions for the v1 MLP forward pass.
//!
//! Each function matches the per-op arithmetic that the Verilator-simulated
//! Verilog must produce and produces the same per-element output the Verilator
//! harness reads from the corresponding test-tap port.
//!
//! ## M57.1 §3.5 — bias-as-seed fold
//!
//! Post-Task-3.4 the hardware initializes the MAC ripple-chain with
//! `acc[o][0] = b[o]` and accumulates products on top, rather than computing
//! a pure matmul and then performing a post-MAC bias-add. By integer
//! addition's associativity these are bit-exactly equivalent, but the CPU
//! reference must mirror the hardware structurally so per-op tap comparisons
//! line up. The standalone bias-add references that existed pre-§3.5
//! (`cpu_reference_layer1_bias` / `cpu_reference_layer2_bias`) are removed —
//! they have no callers post-bias-as-seed.
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
// Per-op reference functions (M57.1 §3.5 bias-as-seed)
// ---------------------------------------------------------------------------

/// Layer 1 matmul-with-bias: i8 × i8 → i16 (intermediate) → i32 accumulator,
/// seeded with bias per M57.1 §3.5.
///
/// `w1` is stored row-major as `[784, 128]`; indexing `w1[i * n_out + o]`.
/// The accumulator is initialized to `b1[o]` and products are summed on top,
/// matching the post-§3.5 ripple-chain seed in `lower_matmul_with_optional_bias`.
pub fn cpu_reference_layer1_matmul_with_bias(
    x: &[i8],
    w1: &[i8],
    b1: &[i32],
) -> Vec<i32> {
    let (n_in, n_out) = (784usize, 128usize);
    assert_eq!(x.len(), n_in, "expected 784-element input");
    assert_eq!(w1.len(), n_in * n_out, "expected 784*128 W1 weights");
    assert_eq!(b1.len(), n_out, "expected 128-element bias");

    // M57.1 §3.5: acc = bias (ripple seed). Integer addition is associative,
    // so this is bit-exactly equivalent to summing products first and adding
    // bias last; the associativity regression test below pins this property.
    let mut out: Vec<i32> = b1.to_vec();
    for o in 0..n_out {
        for i in 0..n_in {
            let prod: i16 = (x[i] as i16) * (w1[i * n_out + o] as i16);
            out[o] = out[o].wrapping_add(prod as i32);
        }
    }
    out
}

/// Layer 1 ReLU: max(0, x) preserving i32 width.
pub fn cpu_reference_layer1_relu(post: &[i32]) -> Vec<i32> {
    post.iter().map(|x| (*x).max(0)).collect()
}

/// Layer 2 matmul-with-bias: i32 × i8 (sign-extended to i32) → i64
/// accumulator, seeded with bias per M57.1 §3.5.
///
/// `w2` is stored row-major as `[128, 10]`; indexing `w2[i * n_out + o]`.
/// The i8 weights are sign-extended to i32 before multiplication (matching
/// the `localparam signed [31:0]` in the emitted Verilog). The accumulator
/// is initialized to `b2[o]` (i64) and the widened products are summed on top.
pub fn cpu_reference_layer2_matmul_with_bias(
    h: &[i32],
    w2: &[i8],
    b2: &[i64],
) -> Vec<i64> {
    let (n_in, n_out) = (128usize, 10usize);
    assert_eq!(h.len(), n_in, "expected 128-element hidden layer");
    assert_eq!(w2.len(), n_in * n_out, "expected 128*10 W2 weights");
    assert_eq!(b2.len(), n_out, "expected 10-element bias");

    // M57.1 §3.5: acc = bias (ripple seed).
    let mut out: Vec<i64> = b2.to_vec();
    for o in 0..n_out {
        for i in 0..n_in {
            let w_i32: i32 = w2[i * n_out + o] as i32; // i8 → i32 sign-extend
            let prod: i64 = (h[i] as i64) * (w_i32 as i64);
            out[o] = out[o].wrapping_add(prod);
        }
    }
    out
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
///
/// M57.1 §3.5 (Concern #4): the standalone bias-tap outputs
/// (`l1_bias` / `l2_bias`) no longer exist post-bias-as-seed; the matmul
/// fields hold the post-bias accumulator value directly. The bias-tap
/// descriptors in `fpga_harness.rs` are removed by Task 3.6.
#[derive(Debug, Clone)]
pub struct MlpOutput {
    /// Layer 1 matmul-with-bias output (acc = bias + Σ products), length 128.
    pub l1_matmul: Vec<i32>,
    /// Layer 1 ReLU output (= hidden layer input to L2), length 128.
    pub l1_relu: Vec<i32>,
    /// Layer 2 matmul-with-bias output (acc = bias + Σ products), length 10.
    pub l2_matmul: Vec<i64>,
    /// Layer 2 ReLU output (= final output), length 10.
    pub l2_relu: Vec<i64>,
}

/// Run the full v1 MLP forward pass on one 784-element i8 input.
pub fn cpu_reference_v1_mlp(x: &[i8], fixture: &FixtureFile) -> MlpOutput {
    let w1 = fixture.blocks[0].as_i8(); // block 0: W1
    let b1 = fixture.blocks[1].as_i32(); // block 1: b1
    let w2 = fixture.blocks[2].as_i8(); // block 2: W2
    let b2 = fixture.blocks[3].as_i64(); // block 3: b2

    let l1_matmul = cpu_reference_layer1_matmul_with_bias(x, &w1, &b1);
    let l1_relu = cpu_reference_layer1_relu(&l1_matmul);
    let l2_matmul = cpu_reference_layer2_matmul_with_bias(&l1_relu, &w2, &b2);
    let l2_relu = cpu_reference_layer2_relu(&l2_matmul);

    MlpOutput {
        l1_matmul,
        l1_relu,
        l2_matmul,
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
    fn layer1_with_bias_zero_input_yields_bias() {
        // M57.1 §3.5: bias-as-seed means zero input yields bias, not zero.
        let x = vec![0i8; 784];
        let w = vec![1i8; 784 * 128];
        let bias = vec![42i32; 128];
        let out = cpu_reference_layer1_matmul_with_bias(&x, &w, &bias);
        assert_eq!(out, vec![42i32; 128]);
    }

    #[test]
    fn layer1_with_bias_single_element_yields_bias_plus_prod() {
        let mut x = vec![0i8; 784];
        x[0] = 1;
        let mut w = vec![0i8; 784 * 128];
        w[0] = 64;
        let mut bias = vec![10i32; 128];
        bias[0] = 10;
        let out = cpu_reference_layer1_matmul_with_bias(&x, &w, &bias);
        assert_eq!(out[0], 10 + 64);
        for o in 1..128 {
            assert_eq!(out[o], 10);
        }
    }

    #[test]
    fn relu_clips_negative() {
        let v = vec![5i32, -3, 0, 17, -1];
        let r = cpu_reference_layer1_relu(&v);
        assert_eq!(r, vec![5, 0, 0, 17, 0]);
    }

    #[test]
    fn associativity_old_vs_new_bias_handling() {
        // M57.1 §3.5 regression test: bias-as-seed and post-MAC bias-add produce
        // identical bit-exact results for integer accumulation (associativity holds).
        let x = vec![3i8; 784];
        let w: Vec<i8> = (0..784 * 128).map(|i| (i % 7) as i8 - 3).collect();
        let bias: Vec<i32> = (0..128).map(|i| (i * 17) as i32 - 1000).collect();

        let new_out = cpu_reference_layer1_matmul_with_bias(&x, &w, &bias);

        let mut old_acc: Vec<i32> = vec![0; 128];
        for o in 0..128 {
            for i in 0..784 {
                let prod = (x[i] as i16) * (w[i * 128 + o] as i16);
                old_acc[o] = old_acc[o].wrapping_add(prod as i32);
            }
        }
        let old_out: Vec<i32> = old_acc
            .iter()
            .zip(&bias)
            .map(|(a, b)| a.wrapping_add(*b))
            .collect();

        assert_eq!(
            new_out, old_out,
            "bias-as-seed must match post-MAC bias-add bit-exactly"
        );
    }

    #[test]
    fn layer2_widening_to_i64() {
        let h = vec![1i32 << 20; 128];
        let mut w = vec![0i8; 128 * 10];
        w[0] = 64;
        let bias = vec![0i64; 10];
        let out = cpu_reference_layer2_matmul_with_bias(&h, &w, &bias);
        assert_eq!(out[0], (1i64 << 20) * 64);
    }
}
