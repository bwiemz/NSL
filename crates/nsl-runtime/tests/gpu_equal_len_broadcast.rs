//! GPU elementwise binary ops must broadcast operands of equal LENGTH but
//! different SHAPE.
//!
//! `gpu_elementwise_binary` entered its broadcast path only when the two
//! lengths differed (or an operand was strided). A column against a row —
//! `[3, 1]` and `[1, 3]`, three elements each — therefore ran the flat
//! kernel and came back as a `[3, 1]` tensor of three element-wise results,
//! where the CPU returns the 3x3 outer result. Measured on an RTX PRO 4500:
//! `[[1],[2],[3]] - [[10,20,30]]` gave `[[-9],[-18],[-27]]`. The tape then
//! un-broadcast gradients against the shapes it had recorded, so the
//! gradient was wrong-shaped as well.
//!
//! Run: cargo test -p nsl-runtime --features cuda --test gpu_equal_len_broadcast
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_add, nsl_tensor_data_ptr, nsl_tensor_div, nsl_tensor_free, nsl_tensor_len, nsl_tensor_mul,
    nsl_tensor_ndim, nsl_tensor_shape_dim, nsl_tensor_sub, nsl_tensor_zeros_on,
};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d};

fn cuda_available() -> bool {
    std::env::var("NSL_SKIP_CUDA_TESTS").is_err() && nsl_cuda_init() == 0
}

fn gpu_tensor(vals: &[f32], shape: &[i64]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 1 /* GPU */);
    nsl_list_free(shape_list);
    nsl_test_cuda_h2d(nsl_tensor_data_ptr(t), vals.as_ptr() as i64, (vals.len() * 4) as i64);
    t
}

fn shape_and_values(t: i64) -> (Vec<i64>, Vec<f32>) {
    let shape = (0..nsl_tensor_ndim(t)).map(|d| nsl_tensor_shape_dim(t, d)).collect();
    let mut out = vec![0.0f32; nsl_tensor_len(t) as usize];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, nsl_tensor_data_ptr(t), (out.len() * 4) as i64);
    (shape, out)
}

type BinaryOp = extern "C" fn(i64, i64, u8) -> i64;

const OPS: [(&str, BinaryOp, fn(f32, f32) -> f32); 4] = [
    ("add", nsl_tensor_add, |a, b| a + b),
    ("sub", nsl_tensor_sub, |a, b| a - b),
    ("mul", nsl_tensor_mul, |a, b| a * b),
    ("div", nsl_tensor_div, |a, b| a / b),
];

/// A `[3, 1]` column against a `[1, 3]` row is the 3x3 outer result.
#[test]
#[ignore = "requires CUDA GPU"]
fn column_against_row_is_the_outer_result() {
    if !cuda_available() {
        return;
    }
    let (col, row) = ([1.0f32, 2.0, 3.0], [10.0f32, 20.0, 30.0]);
    for (name, op, f) in OPS {
        let (a, b) = (gpu_tensor(&col, &[3, 1]), gpu_tensor(&row, &[1, 3]));
        let out = op(a, b, 0);
        let (shape, got) = shape_and_values(out);
        let want: Vec<f32> = col.iter().flat_map(|&c| row.iter().map(move |&r| f(c, r))).collect();
        assert_eq!(shape, [3, 3], "{name}: a [3,1] and a [1,3] operand broadcast to [3,3]");
        // The GPU divide is within an ulp of the correctly rounded quotient,
        // not always equal to it; this gate is about which elements meet.
        let close = got.iter().zip(&want).all(|(g, w)| (g - w).abs() <= 2.0 * f32::EPSILON * w.abs());
        assert!(close, "{name}: the outer result\n got {got:?}\nwant {want:?}");
        for t in [a, b, out] {
            nsl_tensor_free(t);
        }
    }
}

/// A `[3]` vector against a `[3, 1]` column is 3x3 as well: the vector is
/// the trailing dimension.
#[test]
#[ignore = "requires CUDA GPU"]
fn vector_against_column_is_the_outer_result() {
    if !cuda_available() {
        return;
    }
    let (vec3, col) = ([10.0f32, 20.0, 30.0], [1.0f32, 2.0, 3.0]);
    let (a, b) = (gpu_tensor(&vec3, &[3]), gpu_tensor(&col, &[3, 1]));
    let out = nsl_tensor_add(a, b, 0);
    let (shape, got) = shape_and_values(out);
    let want: Vec<f32> = col.iter().flat_map(|&c| vec3.iter().map(move |&v| v + c)).collect();
    assert_eq!(shape, [3, 3]);
    assert_eq!(got, want);
    for t in [a, b, out] {
        nsl_tensor_free(t);
    }
}

/// Two operands of one shape still take the flat kernel and keep that shape.
#[test]
#[ignore = "requires CUDA GPU"]
fn equal_shapes_are_untouched() {
    if !cuda_available() {
        return;
    }
    let (x, y) = ([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0]);
    let (a, b) = (gpu_tensor(&x, &[2, 3]), gpu_tensor(&y, &[2, 3]));
    let out = nsl_tensor_sub(a, b, 0);
    let (shape, got) = shape_and_values(out);
    assert_eq!(shape, [2, 3]);
    assert_eq!(got, [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]);
    for t in [a, b, out] {
        nsl_tensor_free(t);
    }
}
