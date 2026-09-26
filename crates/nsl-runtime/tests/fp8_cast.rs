#![cfg(feature = "test-hooks")]

//! Integration tests for nsl_fp8_cast — f32 → FP8 → f32 round-trip
//! correctness across E4M3 and E5M2 formats, against a reference that decodes
//! the OCP FP8 codes itself (`common::fp8_reference`).

mod common;
use common::fp8_reference::*;
use nsl_runtime::fp8::{nsl_fp8_cast, round_to_fp8, FP8_FORMAT_E4M3, FP8_FORMAT_E5M2};
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
            err <= bound * (1.0 + 1e-6),
            "E4M3 cast at index {i}: orig={orig} deq={deq} err={err} bound={bound}"
        );
        let expected = round_trip(orig, scale, Fp8Format::E4M3);
        assert_eq!(
            deq.to_bits(),
            expected.to_bits(),
            "E4M3 cast at index {i}: orig={orig} got {deq}, the FP8 grid gives {expected}"
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
            err <= bound * (1.0 + 1e-6),
            "E5M2 cast at index {i}: orig={orig} deq={deq} err={err} bound={bound}"
        );
        let expected = round_trip(orig, scale, Fp8Format::E5M2);
        assert_eq!(
            deq.to_bits(),
            expected.to_bits(),
            "E5M2 cast at index {i}: orig={orig} got {deq}, the FP8 grid gives {expected}"
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

fn formats() -> [(Fp8Format, i64); 2] {
    [(Fp8Format::E4M3, FP8_FORMAT_E4M3), (Fp8Format::E5M2, FP8_FORMAT_E5M2)]
}

/// Every finite FP8 value is a fixed point of the runtime's rounding, and
/// every other input lands on the value the exhaustive-search reference
/// picks: just inside and just outside each midpoint between neighbours, the
/// midpoints themselves (ties to even), and beyond the maximum (saturation).
#[test]
fn runtime_rounding_matches_the_fp8_grid_everywhere() {
    for (fmt, code) in formats() {
        let mut grid: Vec<f64> = (0..=255u8).filter_map(|c| fmt.decode(c)).collect();
        grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        grid.dedup();
        for &v in &grid {
            assert_eq!(round_to_fp8(v, code), v, "{fmt:?}: {v} is on the grid");
        }
        let mut probes = Vec::new();
        for w in grid.windows(2) {
            let mid = (w[0] + w[1]) / 2.0;
            let eps = (w[1] - w[0]) * 1e-6;
            probes.extend([mid, mid - eps, mid + eps, w[0] + (w[1] - w[0]) * 0.3]);
        }
        let max = fmt.max_repr() as f64;
        probes.extend([max * 1.01, -max * 1.5, 1e30, -1e30, f64::INFINITY, f64::NEG_INFINITY, 1e-300]);
        for &x in &probes {
            let got = round_to_fp8(x, code);
            let want = fmt.nearest(x);
            assert_eq!(got.to_bits(), want.to_bits(), "{fmt:?}: round_to_fp8({x}) = {got}, the grid gives {want}");
        }
        assert!(round_to_fp8(f64::NAN, code).is_nan());
    }
}

/// Hand-checked points of the OCP spec, so a shared misreading of it in both
/// the runtime and the reference cannot pass unnoticed.
#[test]
fn fp8_grid_spot_values() {
    let e4 = FP8_FORMAT_E4M3;
    let e5 = FP8_FORMAT_E5M2;
    // E4M3: 8 values per binade. [0.25, 0.5) steps by 1/32.
    assert_eq!(round_to_fp8(0.3, e4), 0.3125);
    // [256, 448] steps by 32; 1000 saturates to 448.
    assert_eq!(round_to_fp8(300.0, e4), 288.0);
    assert_eq!(round_to_fp8(1000.0, e4), 448.0);
    // Tie between 288 (mantissa 001) and 320 (010) goes to the even 320.
    assert_eq!(round_to_fp8(304.0, e4), 320.0);
    // Tie between 256 (000) and 288 (001) goes to 256.
    assert_eq!(round_to_fp8(272.0, e4), 256.0);
    // Subnormals step by 2^-9; half of the smallest one ties to zero.
    assert_eq!(round_to_fp8(2f64.powi(-9), e4), 2f64.powi(-9));
    assert_eq!(round_to_fp8(2f64.powi(-10), e4), 0.0);
    assert_eq!(round_to_fp8(3.0 * 2f64.powi(-10), e4), 2.0 * 2f64.powi(-9));
    // E5M2: 4 values per binade. [512, 1024) steps by 128.
    assert_eq!(round_to_fp8(1000.0, e5), 1024.0);
    assert_eq!(round_to_fp8(700.0, e5), 640.0);
    assert_eq!(round_to_fp8(100_000.0, e5), 57344.0);
    // Subnormals step by 2^-16.
    assert_eq!(round_to_fp8(3.0 * 2f64.powi(-17), e5), 2.0 * 2f64.powi(-16));
    // The sign carries through, including on zero.
    assert_eq!(round_to_fp8(-0.3, e4), -0.3125);
    assert!(round_to_fp8(-1e-9, e4).is_sign_negative());
}

/// Two mantissa bits are coarse: a whole tensor round-tripped through E5M2
/// keeps each element within 2^-3 relative error (normal range) — and some
/// element of a random tensor really does get close to that bound, which the
/// old uniform grid (error <= 0.25 * scale) never produced.
#[test]
fn e5m2_relative_error_is_two_mantissa_bits() {
    let data = seeded_input(4096, 7);
    let scale = compute_pertensor_scale(&data, &[], Fp8Format::E5M2);
    let input = make_tensor_2d_f32(1, data.len(), &data);
    let out = nsl_fp8_cast(input, FP8_FORMAT_E5M2, scale as f64);
    let deq = read_tensor_f32(out);
    let mut worst = 0.0f32;
    for (&x, &q) in data.iter().zip(&deq) {
        if x.abs() / scale > 2f32.powi(-14) {
            worst = worst.max((x - q).abs() / x.abs());
        }
    }
    assert!(worst <= 0.125 * (1.0 + 1e-6), "worst relative error {worst} > 2^-3");
    assert!(worst > 0.09, "worst relative error {worst}: E5M2 should be about 2^-3 coarse");
    nsl_tensor_free(input);
    nsl_tensor_free(out);
}
