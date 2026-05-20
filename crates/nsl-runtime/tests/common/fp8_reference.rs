//! Ground-truth scalar FP8 reference for integration tests.
//!
//! Every helper here is independent of the runtime's FP8 implementation.
//! If a test fails because the reference and the runtime disagree, triage
//! by inspecting the reference first (it must be bit-correct against the
//! published FP8 format spec) before assuming the runtime is at fault.

use nsl_runtime::fp8::{
    compute_scale, dequantize_fp8, fp8_matmul_cpu, quantize_fp8, FP8E4M3_MAX, FP8E5M2_MAX,
    FP8_FORMAT_E4M3, FP8_FORMAT_E5M2,
};

// Not every test binary uses both helpers (fp8_scale only builds tensors;
// fp8_dispatcher's no-cuda path uses neither) — silence the unused-import
// warnings this produces in those binaries.
#[allow(unused_imports)]
pub use nsl_runtime::tensor::{
    test_build_tensor_2d_f32 as make_tensor_2d_f32, test_read_tensor_f32 as read_tensor_f32,
};

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

pub const E4M3_REL_TOL: f32 = 0.02;
pub const E5M2_REL_TOL: f32 = 0.10;
pub const DISPATCH_ABS_TOL: f32 = 1e-5;

pub fn quantize(x: f32, scale: f32, fmt: Fp8Format) -> f32 {
    quantize_fp8(x as f64, scale as f64, fmt.dtype_code()) as f32
}

pub fn dequantize(q: f32, scale: f32) -> f32 {
    dequantize_fp8(q as f64, scale as f64) as f32
}

pub fn round_trip(x: f32, scale: f32, fmt: Fp8Format) -> f32 {
    dequantize(quantize(x, scale, fmt), scale)
}

pub fn quantization_step(x: f32, scale: f32, fmt: Fp8Format) -> f32 {
    let precision = match fmt {
        Fp8Format::E4M3 => 0.125,
        Fp8Format::E5M2 => 0.5,
    };
    let _ = x; // magnitude-dependent bound could vary; uniform bound chosen for simplicity
    (precision as f32) * scale / 2.0
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
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
}

pub const FIXTURE_SHAPES: &[(usize, usize, usize)] =
    &[(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128)];

pub fn compute_pertensor_scale(a: &[f32], b: &[f32], fmt: Fp8Format) -> f32 {
    let combined: Vec<f64> = a.iter().chain(b.iter()).map(|&v| v as f64).collect();
    compute_scale(&combined, fmt.dtype_code()) as f32
}

pub fn assert_rel_err_le(test: &[f32], reference: &[f32], tol: f32, label: &str) {
    assert_eq!(test.len(), reference.len(), "{label}: length mismatch");
    let max_ref_abs = reference.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_ref_abs < 1e-12 {
        let max_abs_err = test
            .iter()
            .zip(reference)
            .map(|(t, r)| (t - r).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_err < tol,
            "{label}: reference is ~zero; max abs err {max_abs_err} > tol {tol}"
        );
        return;
    }
    let max_rel_err = test
        .iter()
        .zip(reference)
        .map(|(t, r)| (t - r).abs() / max_ref_abs)
        .fold(0.0_f32, f32::max);
    assert!(
        max_rel_err <= tol,
        "{label}: max rel err {max_rel_err} > tol {tol} (max_ref_abs = {max_ref_abs})"
    );
}

pub fn assert_abs_err_le(test: &[f32], reference: &[f32], tol: f32, label: &str) {
    assert_eq!(test.len(), reference.len(), "{label}: length mismatch");
    let max_abs_err = test
        .iter()
        .zip(reference)
        .map(|(t, r)| (t - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_err <= tol,
        "{label}: max abs err {max_abs_err} > tol {tol}"
    );
}
