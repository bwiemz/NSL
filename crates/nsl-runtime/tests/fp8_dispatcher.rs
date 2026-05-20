#![cfg(feature = "test-hooks")]

//! Integration tests for FP8 matmul dispatcher — MMA kernel output and
//! CPU fallback output must agree within DISPATCH_ABS_TOL (1e-5) at all
//! tested shapes. Divergence > 1e-5 indicates an algorithmic difference,
//! not f32 accumulation-ordering physics.

mod common;
// Needed by `cuda_tests` via `super::*`. When cuda is off, the submodule
// is compiled out and the glob looks unused — the attribute silences the
// warning without moving the glob (which would complicate super reach-
// through if additional tests land here later).
#[allow(unused_imports)]
use common::fp8_reference::*;

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use nsl_runtime::fp8::{nsl_fp8_cast, nsl_fp8_matmul, FP8_FORMAT_E4M3};
    use nsl_runtime::tensor::nsl_tensor_free;

    /// Returns true if the current device supports FP8 mma.sync (sm_89+).
    /// Uses the runtime's `test_detect_sm_version` helper added in Step 2.
    fn is_sm_89_plus() -> bool {
        nsl_runtime::test_detect_sm_version() >= 89
    }

    fn run_dispatch_at_shape(m: usize, k: usize, n: usize) {
        if !is_sm_89_plus() {
            eprintln!("skipping {m}x{k}x{n}: device < sm_89");
            return;
        }
        let a_data = seeded_input(m * k, 21);
        let b_data = seeded_input(k * n, 22);
        let scale = compute_pertensor_scale(&a_data, &b_data, Fp8Format::E4M3);

        let a_f32 = make_tensor_2d_f32(m, k, &a_data);
        let b_f32 = make_tensor_2d_f32(k, n, &b_data);
        let a_fp8_mma = nsl_fp8_cast(a_f32, FP8_FORMAT_E4M3, scale as f64);
        let b_fp8_mma = nsl_fp8_cast(b_f32, FP8_FORMAT_E4M3, scale as f64);
        let out_mma = nsl_fp8_matmul(a_fp8_mma, b_fp8_mma, scale as f64, scale as f64);
        let v_mma = read_tensor_f32(out_mma);

        let reference = Fp8ReferenceMatmul {
            m,
            n,
            k,
            format: Fp8Format::E4M3,
            scale,
        };
        let v_fallback = reference.compute_f32(&a_data, &b_data);

        assert_abs_err_le(
            &v_mma,
            &v_fallback,
            DISPATCH_ABS_TOL,
            &format!("MMA vs fallback {m}x{k}x{n}"),
        );

        nsl_tensor_free(a_f32);
        nsl_tensor_free(b_f32);
        nsl_tensor_free(a_fp8_mma);
        nsl_tensor_free(b_fp8_mma);
        nsl_tensor_free(out_mma);
    }

    #[test]
    fn dispatch_16x16x16() {
        run_dispatch_at_shape(16, 16, 16);
    }

    #[test]
    fn dispatch_32x32x32() {
        run_dispatch_at_shape(32, 32, 32);
    }

    #[test]
    fn dispatch_64x64x64() {
        run_dispatch_at_shape(64, 64, 64);
    }

    #[test]
    fn dispatch_128x128x128() {
        run_dispatch_at_shape(128, 128, 128);
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
#[ignore = "FP8 dispatcher tests require the cuda feature + sm_89+"]
fn cuda_feature_required() {
    eprintln!("compile with --features cuda to run dispatcher tests");
}
