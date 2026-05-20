//! Spec §4.2 required fixture tests for absmax activation quantization.
//!
//! The four fixtures validate per-row absmax behavior at the math level
//! (via the Task 3 reference impl). Structural PTX assertions live in
//! tests/bitnet_ptx_snapshots.rs.

#[path = "bitnet_reference_impl.rs"]
mod reference;

use reference::{absmax_scale_row, quantize_row_int8};

#[test]
fn fixture_zero_magnitude_row() {
    let row = vec![0.0_f32; 8];
    let scale = absmax_scale_row(&row);
    assert_eq!(scale, 0.0, "zero-magnitude row scale must be 0.0");
    let q = quantize_row_int8(&row, scale);
    assert_eq!(
        q,
        vec![0i8; 8],
        "zero-magnitude row must produce all-zero quantized output"
    );
}

#[test]
fn fixture_uniform_magnitude_row() {
    let row = vec![1.5_f32; 8];
    let scale = absmax_scale_row(&row);
    assert!(
        (scale - 1.5).abs() < 1e-6,
        "absmax of uniform 1.5 must be 1.5"
    );
    let q = quantize_row_int8(&row, scale);
    // All values map to scale, so q = +127 across the row.
    assert_eq!(
        q,
        vec![127i8; 8],
        "uniform-positive row should clip to +127 across"
    );
}

#[test]
fn fixture_mixed_sign_mixed_magnitude_row() {
    let row = vec![1.5_f32, -2.5, 0.5, -1.0];
    let scale = absmax_scale_row(&row);
    assert!((scale - 2.5).abs() < 1e-6, "absmax must be 2.5");
    let q = quantize_row_int8(&row, scale);
    // q = round(x * 127 / 2.5) = round([76.2, -127.0, 25.4, -50.8]) = [76, -127, 25, -51]
    assert_eq!(
        q,
        vec![76, -127, 25, -51],
        "mixed-sign quantization mismatch"
    );
}

#[test]
fn fixture_single_outlier_row() {
    let row = vec![0.1_f32, 0.2, 10.0, 0.15];
    let scale = absmax_scale_row(&row);
    assert!((scale - 10.0).abs() < 1e-6, "outlier dominates absmax");
    let q = quantize_row_int8(&row, scale);
    // q = round(x * 127 / 10.0)
    //   = round([1.27, 2.54, 127, 1.905]) = [1, 3, 127, 2]
    assert_eq!(
        q,
        vec![1, 3, 127, 2],
        "outlier-row quantization mismatch (non-outliers compressed)"
    );
}
