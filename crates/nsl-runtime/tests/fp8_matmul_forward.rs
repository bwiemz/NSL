#![cfg(feature = "test-hooks")]

//! Integration tests for E4M3 forward matmul — output of nsl_fp8_matmul
//! agrees with the scalar reference within E4M3_REL_TOL. Both sides multiply
//! the same FP8-rounded operands, so the tolerance only covers f32-vs-f64
//! accumulation, not FP8 rounding.

mod common;
use common::fp8_reference::*;
use nsl_runtime::fp8::{nsl_fp8_cast, nsl_fp8_matmul, FP8_FORMAT_E4M3};
use nsl_runtime::tensor::nsl_tensor_free;

fn run_e4m3_at_shape(m: usize, k: usize, n: usize) {
    let a_data = seeded_input(m * k, 1);
    let b_data = seeded_input(k * n, 2);
    let scale = compute_pertensor_scale(&a_data, &b_data, Fp8Format::E4M3);

    let reference = Fp8ReferenceMatmul {
        m,
        n,
        k,
        format: Fp8Format::E4M3,
        scale,
    };
    let ref_out = reference.compute_f32(&a_data, &b_data);

    let a_f32 = make_tensor_2d_f32(m, k, &a_data);
    let b_f32 = make_tensor_2d_f32(k, n, &b_data);
    let a_fp8 = nsl_fp8_cast(a_f32, FP8_FORMAT_E4M3, scale as f64);
    let b_fp8 = nsl_fp8_cast(b_f32, FP8_FORMAT_E4M3, scale as f64);

    let out_ptr = nsl_fp8_matmul(a_fp8, b_fp8, scale as f64, scale as f64);
    let test_out = read_tensor_f32(out_ptr);

    assert_rel_err_le(
        &test_out,
        &ref_out,
        E4M3_REL_TOL,
        &format!("E4M3 matmul {m}x{k}x{n}"),
    );

    nsl_tensor_free(a_f32);
    nsl_tensor_free(b_f32);
    nsl_tensor_free(a_fp8);
    nsl_tensor_free(b_fp8);
    nsl_tensor_free(out_ptr);
}

#[test]
fn e4m3_matmul_16x16x16() {
    run_e4m3_at_shape(16, 16, 16);
}

#[test]
fn e4m3_matmul_32x32x32() {
    run_e4m3_at_shape(32, 32, 32);
}

#[test]
fn e4m3_matmul_64x64x64() {
    run_e4m3_at_shape(64, 64, 64);
}

#[test]
fn e4m3_matmul_128x128x128() {
    run_e4m3_at_shape(128, 128, 128);
}

#[test]
fn e4m3_matmul_non_square() {
    run_e4m3_at_shape(64, 32, 16);
}

// The shared comparators' own guard. Not FP8 tests — they pin the property
// every tolerance assertion over `common::fp8_reference` depends on: an
// all-NaN result must fail, not score 0.0 (`f32::max` returns the non-NaN
// operand, so a bare fold drops it). They live here because this binary
// needs no device.

#[test]
#[should_panic(expected = "non-finite")]
fn rel_err_comparator_rejects_nan_instead_of_scoring_it_perfect() {
    let reference = vec![1.0f32, -2.0, 3.0, 4.0];
    assert_rel_err_le(&[f32::NAN; 4], &reference, 1e-6, "all-NaN test");
}

#[test]
#[should_panic(expected = "non-finite")]
fn rel_err_comparator_rejects_nan_in_the_reference() {
    let test = vec![1.0f32, -2.0, 3.0, 4.0];
    let mut reference = test.clone();
    reference[1] = f32::NAN;
    assert_rel_err_le(&test, &reference, 1e-6, "NaN reference");
}

#[test]
#[should_panic(expected = "non-finite")]
fn abs_err_comparator_rejects_nan_instead_of_scoring_it_perfect() {
    let reference = vec![1.0f32, -2.0, 3.0, 4.0];
    let mut test = reference.clone();
    test[2] = f32::NAN;
    assert_abs_err_le(&test, &reference, 1e-6, "one-NaN test");
}

#[test]
#[should_panic(expected = "finite disagreement: max rel err")]
fn rel_err_comparator_still_reports_finite_disagreement() {
    let reference = vec![1.0f32, -2.0, 3.0, 4.0];
    let mut test = reference.clone();
    test[2] = 3.5;
    assert_rel_err_le(&reference, &reference, 0.0, "identical");
    assert_rel_err_le(&test, &reference, 1e-3, "finite disagreement");
}

#[test]
#[should_panic(expected = "finite disagreement: max abs err")]
fn abs_err_comparator_still_reports_finite_disagreement() {
    let reference = vec![1.0f32, -2.0, 3.0, 4.0];
    let mut test = reference.clone();
    test[0] = 1.5;
    assert_abs_err_le(&reference, &reference, 0.0, "identical");
    assert_abs_err_le(&test, &reference, 1e-3, "finite disagreement");
}
