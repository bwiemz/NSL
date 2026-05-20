#![cfg(feature = "test-hooks")]

//! Integration tests for nsl_fp8_compute_scale — verifies scale =
//! max_abs / FP8_MAX for both formats, plus edge cases.

mod common;
use common::fp8_reference::*;
use nsl_runtime::fp8::{
    nsl_fp8_compute_scale, FP8E4M3_MAX, FP8E5M2_MAX, FP8_FORMAT_E4M3, FP8_FORMAT_E5M2,
};
use nsl_runtime::tensor::nsl_tensor_free;

#[test]
fn e4m3_scale_matches_max_div_format_max() {
    let data: Vec<f32> = vec![1.0, -4.0, 2.0, 0.5];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3) as f32;
    let expected = 4.0_f32 / FP8E4M3_MAX;
    assert!(
        (scale - expected).abs() <= f32::EPSILON,
        "E4M3 scale {scale} != expected {expected}"
    );

    nsl_tensor_free(ptr);
}

#[test]
fn e5m2_scale_matches_max_div_format_max() {
    let data: Vec<f32> = vec![1.0, -10000.0, 2.0, 50.0];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E5M2) as f32;
    let expected = 10000.0_f32 / FP8E5M2_MAX;
    assert!(
        (scale - expected).abs() <= f32::EPSILON,
        "E5M2 scale {scale} != expected {expected}"
    );

    nsl_tensor_free(ptr);
}

#[test]
fn all_zero_input_returns_sentinel_one() {
    let data: Vec<f32> = vec![0.0; 16];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3);
    assert_eq!(scale, 1.0, "all-zero input must return sentinel 1.0");

    nsl_tensor_free(ptr);
}

#[test]
fn single_element_e4m3() {
    let data: Vec<f32> = vec![3.5];
    let ptr = make_tensor_2d_f32(1, 1, &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3) as f32;
    let expected = 3.5_f32 / FP8E4M3_MAX;
    assert!((scale - expected).abs() <= f32::EPSILON);

    nsl_tensor_free(ptr);
}

#[test]
fn negative_only_input_uses_magnitude() {
    let data: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3) as f32;
    let expected = 4.0_f32 / FP8E4M3_MAX;
    assert!((scale - expected).abs() <= f32::EPSILON);

    nsl_tensor_free(ptr);
}

#[test]
fn large_magnitude_e4m3_clamps_to_max() {
    let data: Vec<f32> = vec![10000.0, -5000.0];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3) as f32;
    let scaled_max = 10000.0_f32 / scale;
    assert!(
        (scaled_max - FP8E4M3_MAX).abs() <= 1e-3,
        "scaled max {scaled_max} should equal {FP8E4M3_MAX}"
    );

    nsl_tensor_free(ptr);
}

#[test]
fn f32_tensor_dtype_path() {
    let data: Vec<f32> = vec![2.0, 1.0, -1.0, -2.0];
    let ptr = make_tensor_2d_f32(1, data.len(), &data);

    let scale = nsl_fp8_compute_scale(ptr, FP8_FORMAT_E4M3) as f32;
    let expected = 2.0_f32 / FP8E4M3_MAX;
    assert!((scale - expected).abs() <= f32::EPSILON);

    nsl_tensor_free(ptr);
}
