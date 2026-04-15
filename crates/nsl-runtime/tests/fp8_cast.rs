#![cfg(feature = "test-hooks")]

//! Integration tests for nsl_fp8_cast — f32 → FP8 → f32 round-trip
//! correctness across E4M3 and E5M2 formats.

mod common;
use common::fp8_reference::*;
use nsl_runtime::fp8::{nsl_fp8_cast, FP8_FORMAT_E4M3, FP8_FORMAT_E5M2};
use nsl_runtime::tensor::nsl_tensor_free;

#[test]
fn e4m3_cast_round_trip_within_step() {
    let data: Vec<f32> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 100.0];
    let scale = compute_pertensor_scale(&data, &[], Fp8Format::E4M3);

    let input_ptr = make_tensor_2d_f32(1, data.len(), &data);
    let output_ptr = nsl_fp8_cast(input_ptr, FP8_FORMAT_E4M3, scale as f64);
    let output = read_tensor_f32(output_ptr);

    for (i, (&orig, &deq)) in data.iter().zip(&output).enumerate() {
        let bound = quantization_step(orig, scale, Fp8Format::E4M3);
        let err = (orig - deq).abs();
        assert!(
            err <= bound + 1e-6,
            "E4M3 cast at index {i}: orig={orig} deq={deq} err={err} bound={bound}"
        );
    }

    nsl_tensor_free(input_ptr);
    nsl_tensor_free(output_ptr);
}

#[test]
fn e5m2_cast_round_trip_within_step() {
    let data: Vec<f32> = vec![-1000.0, -10.0, 0.0, 0.5, 1.0, 100.0, 5000.0, 50000.0];
    let scale = compute_pertensor_scale(&data, &[], Fp8Format::E5M2);

    let input_ptr = make_tensor_2d_f32(1, data.len(), &data);
    let output_ptr = nsl_fp8_cast(input_ptr, FP8_FORMAT_E5M2, scale as f64);
    let output = read_tensor_f32(output_ptr);

    for (i, (&orig, &deq)) in data.iter().zip(&output).enumerate() {
        let bound = quantization_step(orig, scale, Fp8Format::E5M2);
        let err = (orig - deq).abs();
        assert!(
            err <= bound + 1e-6,
            "E5M2 cast at index {i}: orig={orig} deq={deq} err={err} bound={bound}"
        );
    }

    nsl_tensor_free(input_ptr);
    nsl_tensor_free(output_ptr);
}

#[test]
fn cast_auto_scale_matches_explicit_scale() {
    let data: Vec<f32> = seeded_input(64, 42);
    let explicit_scale = compute_pertensor_scale(&data, &[], Fp8Format::E4M3);

    let input_a = make_tensor_2d_f32(1, data.len(), &data);
    let out_explicit = nsl_fp8_cast(input_a, FP8_FORMAT_E4M3, explicit_scale as f64);

    let input_b = make_tensor_2d_f32(1, data.len(), &data);
    let out_auto = nsl_fp8_cast(input_b, FP8_FORMAT_E4M3, 0.0);

    let v_explicit = read_tensor_f32(out_explicit);
    let v_auto = read_tensor_f32(out_auto);

    assert_abs_err_le(&v_auto, &v_explicit, 1e-6,
        "auto-scale must match explicit scale exactly");

    nsl_tensor_free(input_a);
    nsl_tensor_free(input_b);
    nsl_tensor_free(out_explicit);
    nsl_tensor_free(out_auto);
}
