#![cfg(feature = "test-hooks")]

//! Integration tests for E4M3 forward matmul — output of nsl_fp8_matmul
//! agrees with the scalar reference within E4M3_REL_TOL (2%).

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
