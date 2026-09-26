//! Ground-truth scalar FP8 reference for integration tests.
//!
//! Every helper here is independent of the runtime's FP8 implementation.
//! If a test fails because the reference and the runtime disagree, triage
//! by inspecting the reference first (it must be bit-correct against the
//! published FP8 format spec) before assuming the runtime is at fault.

use nsl_runtime::fp8::{
    compute_scale, fp8_matmul_cpu, FP8E4M3_MAX, FP8E5M2_MAX, FP8_FORMAT_E4M3, FP8_FORMAT_E5M2,
};

// Not every test binary uses both helpers (fp8_scale only builds tensors;
// fp8_dispatcher's no-cuda path uses neither) — silence the unused-import
// warnings this produces in those binaries.
#[allow(unused_imports)]
pub use nsl_runtime::tensor::{test_build_tensor_2d_f32 as make_tensor_2d_f32,
                               test_read_tensor_f32 as read_tensor_f32};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    E4M3,
    E5M2,
}

impl Fp8Format {
    pub fn dtype_code(self) -> i64 {
        match self {
            Fp8Format::E4M3 => FP8_FORMAT_E4M3,
            Fp8Format::E5M2 => FP8_FORMAT_E5M2,
        }
    }

    pub fn max_repr(self) -> f32 {
        match self {
            Fp8Format::E4M3 => FP8E4M3_MAX,
            Fp8Format::E5M2 => FP8E5M2_MAX,
        }
    }
}

/// Runtime FP8 matmul vs this reference: both multiply the SAME FP8-rounded
/// operands, so the only difference is the runtime's f32 accumulation against
/// the reference's f64 one. Relative to the largest |output|, that stays
/// around 1e-7 at K = 128; 1e-5 leaves headroom and is still thousands of
/// times smaller than one FP8 rounding step (2^-4 for E4M3, 2^-3 for E5M2).
pub const E4M3_REL_TOL: f32 = 1e-5;
pub const E5M2_REL_TOL: f32 = 1e-5;
pub const DISPATCH_ABS_TOL: f32 = 1e-5;

impl Fp8Format {
    /// Decode one FP8 byte per the OCP 8-bit floating point spec (v1.0).
    /// `None` for the NaN codes (and E5M2's infinities).
    pub fn decode(self, code: u8) -> Option<f64> {
        let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
        let (exp_bits, man_bits, bias) = match self {
            Fp8Format::E4M3 => (4u32, 3u32, 7i32),
            Fp8Format::E5M2 => (5, 2, 15),
        };
        let exp = ((code as u32 >> man_bits) & ((1 << exp_bits) - 1)) as i32;
        let man = (code as u32 & ((1 << man_bits) - 1)) as f64;
        let man_scale = (1u32 << man_bits) as f64;
        let exp_max = (1i32 << exp_bits) - 1;
        match self {
            // E4M3 has no infinities; only S.1111.111 is NaN.
            Fp8Format::E4M3 if exp == exp_max && man == 7.0 => return None,
            // E5M2 is IEEE-like: the top exponent is Inf/NaN.
            Fp8Format::E5M2 if exp == exp_max => return None,
            _ => {}
        }
        let magnitude = if exp == 0 {
            man / man_scale * 2f64.powi(1 - bias)
        } else {
            (1.0 + man / man_scale) * 2f64.powi(exp - bias)
        };
        Some(sign * magnitude)
    }

    /// The finite value nearest to `x` among all 256 codes; a tie goes to the
    /// code with an even mantissa. `x` beyond the maximum saturates to it.
    pub fn nearest(self, x: f64) -> f64 {
        if x.is_nan() {
            return x;
        }
        let x = x.clamp(-(self.max_repr() as f64), self.max_repr() as f64);
        let mut best: Option<(f64, u8)> = None;
        for code in 0..=255u8 {
            let Some(v) = self.decode(code) else { continue };
            let better = match best {
                None => true,
                Some((b, bcode)) => {
                    let (d, bd) = ((v - x).abs(), (b - x).abs());
                    d < bd || (d == bd && v != b && code & 1 == 0 && bcode & 1 == 1)
                }
            };
            if better {
                best = Some((v, code));
            }
        }
        let v = best.unwrap().0;
        // Both zero codes are finite; keep the sign of the input like IEEE.
        if v == 0.0 { 0.0f64.copysign(x) } else { v }
    }

    /// Half the spacing between FP8 neighbours around `|x|` — the largest
    /// round-to-nearest error at that magnitude, in scaled units.
    pub fn half_ulp(self, x: f64) -> f64 {
        let (man_bits, min_normal_exp) = match self {
            Fp8Format::E4M3 => (3, -6),
            Fp8Format::E5M2 => (2, -14),
        };
        let a = x.abs().min(self.max_repr() as f64);
        let e = if a == 0.0 { min_normal_exp } else { (a.log2().floor() as i32).max(min_normal_exp) };
        2f64.powi(e - man_bits) / 2.0
    }
}

/// Reference quantization, independent of the runtime: `x / scale` onto the
/// FP8 grid by exhaustive search over the decoded codes.
pub fn quantize(x: f32, scale: f32, fmt: Fp8Format) -> f32 {
    fmt.nearest(x as f64 / scale as f64) as f32
}

pub fn dequantize(q: f32, scale: f32) -> f32 {
    (q as f64 * scale as f64) as f32
}

/// `x` after an FP8 round trip at `scale`, computed in f64 the way the
/// runtime's `nsl_fp8_cast` does, so the two agree bit for bit.
pub fn round_trip(x: f32, scale: f32, fmt: Fp8Format) -> f32 {
    (fmt.nearest(x as f64 / scale as f64) * scale as f64) as f32
}

/// The largest round-trip error allowed at `x` (half an FP8 ulp at
/// `x / scale`, back in unscaled units).
pub fn quantization_step(x: f32, scale: f32, fmt: Fp8Format) -> f32 {
    (fmt.half_ulp(x as f64 / scale as f64) * scale as f64) as f32
}

pub struct Fp8ReferenceMatmul {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub format: Fp8Format,
    pub scale: f32,
}

impl Fp8ReferenceMatmul {
    pub fn compute_f32(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), self.m * self.k, "A shape mismatch");
        assert_eq!(b.len(), self.k * self.n, "B shape mismatch");

        let dq_a: Vec<f64> = a
            .iter()
            .map(|&v| round_trip(v, self.scale, self.format) as f64)
            .collect();
        let dq_b: Vec<f64> = b
            .iter()
            .map(|&v| round_trip(v, self.scale, self.format) as f64)
            .collect();

        fp8_matmul_cpu(&dq_a, &dq_b, self.m, self.k, self.n)
            .into_iter()
            .map(|v| v as f32)
            .collect()
    }
}

pub fn seeded_input(len: usize, seed: u64) -> Vec<f32> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(-1.0_f32..1.0)).collect()
}

pub const FIXTURE_SHAPES: &[(usize, usize, usize)] = &[
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
];

pub fn compute_pertensor_scale(a: &[f32], b: &[f32], fmt: Fp8Format) -> f32 {
    let combined: Vec<f64> = a.iter().chain(b.iter()).map(|&v| v as f64).collect();
    compute_scale(&combined, fmt.dtype_code()) as f32
}

pub fn assert_rel_err_le(test: &[f32], reference: &[f32], tol: f32, label: &str) {
    assert_eq!(test.len(), reference.len(), "{label}: length mismatch");
    let max_ref_abs = reference.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_ref_abs < 1e-12 {
        let max_abs_err = test.iter().zip(reference).map(|(t, r)| (t - r).abs()).fold(0.0_f32, f32::max);
        assert!(max_abs_err < tol, "{label}: reference is ~zero; max abs err {max_abs_err} > tol {tol}");
        return;
    }
    let max_rel_err = test.iter().zip(reference).map(|(t, r)| (t - r).abs() / max_ref_abs).fold(0.0_f32, f32::max);
    assert!(max_rel_err <= tol, "{label}: max rel err {max_rel_err} > tol {tol} (max_ref_abs = {max_ref_abs})");
}

pub fn assert_abs_err_le(test: &[f32], reference: &[f32], tol: f32, label: &str) {
    assert_eq!(test.len(), reference.len(), "{label}: length mismatch");
    let max_abs_err = test.iter().zip(reference).map(|(t, r)| (t - r).abs()).fold(0.0_f32, f32::max);
    assert!(max_abs_err <= tol, "{label}: max abs err {max_abs_err} > tol {tol}");
}
