//! Integration tests for E5M2 backward matmul — output of
//! fp8_matmul_e5m2_backward agrees with the scalar reference within
//! E5M2_REL_TOL (10%).

mod common;
use common::fp8_reference::*;
use nsl_runtime::fp8::fp8_matmul_e5m2_backward;
use nsl_runtime::tensor::nsl_tensor_free;

fn run_e5m2_at_shape(m: usize, k: usize, n: usize) {
    let a_data = seeded_input(m * k, 11);
    let b_data = seeded_input(k * n, 12);
    let scale = compute_pertensor_scale(&a_data, &b_data, Fp8Format::E5M2);

    let reference = Fp8ReferenceMatmul {
        m,
        n,
        k,
        format: Fp8Format::E5M2,
        scale,
    };
    let ref_out = reference.compute_f32(&a_data, &b_data);

    let a_ptr = make_tensor_2d_f32(m, k, &a_data);
    let b_ptr = make_tensor_2d_f32(k, n, &b_data);

    let out_ptr = fp8_matmul_e5m2_backward(a_ptr, b_ptr, scale, scale);
    let test_out = read_tensor_f32(out_ptr);

    assert_rel_err_le(
        &test_out,
        &ref_out,
        E5M2_REL_TOL,
        &format!("E5M2 backward matmul {m}x{k}x{n}"),
    );

    nsl_tensor_free(a_ptr);
    nsl_tensor_free(b_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn e5m2_backward_16x16x16() {
    run_e5m2_at_shape(16, 16, 16);
}

#[test]
fn e5m2_backward_32x32x32() {
    run_e5m2_at_shape(32, 32, 32);
}

#[test]
fn e5m2_backward_64x64x64() {
    run_e5m2_at_shape(64, 64, 64);
}

#[test]
fn e5m2_backward_128x128x128() {
    run_e5m2_at_shape(128, 128, 128);
}
