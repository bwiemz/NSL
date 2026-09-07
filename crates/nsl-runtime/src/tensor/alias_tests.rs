//! Aliasing-input probes (roadmap C2, step 6).
//!
//! Every entry point here takes two or more tensor handles, and each probe
//! passes the SAME handle for all of them: `cat([x, x])`, `x == x`,
//! `where(c, x, x)`, `x += x`, and so on — all of which a program can write.
//! Natively they are ordinary tests of the arithmetic. Under Miri
//! (`scripts/miri-cpu-tensor.sh tensor::alias_tests`) they are the check
//! that an op does not derive two `&mut NslTensor` from one handle: the
//! second `from_ptr` invalidates the first under Stacked Borrows, and the
//! next read through the first is undefined behaviour that runs "fine"
//! natively. The `matmul(x, x)` and `mul(y, y)` cases were found by the
//! module sweep; this module makes the whole family a standing gate.

use super::*;

fn make_tensor_f64(data: &[f64]) -> i64 {
    let shape_list = crate::list::nsl_list_new();
    crate::list::nsl_list_push(shape_list, data.len() as i64);
    let ptr = crate::tensor::creation::tensor_from_shape_list_f64(shape_list, 0.0);
    let t = NslTensor::from_ptr(ptr);
    for (i, v) in data.iter().enumerate() {
        unsafe { *t.data_f64().add(i) = *v };
    }
    crate::list::nsl_list_free(shape_list);
    ptr
}

fn make_tensor_2d_f64(rows: usize, cols: usize, data: &[f64]) -> i64 {
    assert_eq!(rows * cols, data.len());
    let shape_list = crate::list::nsl_list_new();
    crate::list::nsl_list_push(shape_list, rows as i64);
    crate::list::nsl_list_push(shape_list, cols as i64);
    let ptr = crate::tensor::creation::tensor_from_shape_list_f64(shape_list, 0.0);
    let t = NslTensor::from_ptr(ptr);
    for (i, v) in data.iter().enumerate() {
        unsafe { *t.data_f64().add(i) = *v };
    }
    crate::list::nsl_list_free(shape_list);
    ptr
}

fn values(ptr: i64) -> Vec<f64> {
    let t = NslTensor::from_ptr_ref(ptr);
    assert_eq!(t.dtype, DTYPE_F64, "probe tensors are CPU f64");
    (0..t.len as usize).map(|i| unsafe { *(t.data as *const f64).add(i) }).collect()
}

fn shape(ptr: i64) -> Vec<i64> {
    let t = NslTensor::from_ptr_ref(ptr);
    (0..t.ndim as usize).map(|i| unsafe { *t.shape.add(i) }).collect()
}

fn list_of(ptrs: &[i64]) -> i64 {
    let l = crate::list::nsl_list_new();
    for &p in ptrs {
        crate::list::nsl_list_push(l, p);
    }
    l
}

#[test]
fn add_sub_mul_div_same_handle() {
    let x = make_tensor_f64(&[1.0, 2.0, 4.0]);
    let s = arithmetic::nsl_tensor_add(x, x, 0);
    let d = arithmetic::nsl_tensor_sub(x, x, 0);
    let m = arithmetic::nsl_tensor_mul(x, x, 0);
    let q = arithmetic::nsl_tensor_div(x, x, 0);
    assert_eq!(values(s), [2.0, 4.0, 8.0]);
    assert_eq!(values(d), [0.0, 0.0, 0.0]);
    assert_eq!(values(m), [1.0, 4.0, 16.0]);
    assert_eq!(values(q), [1.0, 1.0, 1.0]);
    for p in [s, d, m, q, x] {
        nsl_tensor_free(p);
    }
}

#[test]
fn matmul_same_handle() {
    let x = make_tensor_2d_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let y = arithmetic::nsl_tensor_matmul(x, x, 0);
    assert_eq!(values(y), [7.0, 10.0, 15.0, 22.0]);
    nsl_tensor_free(y);
    nsl_tensor_free(x);
}

#[test]
fn compare_same_handle() {
    let x = make_tensor_f64(&[1.0, -2.0, 3.5]);
    // kind 4 is `==` (within 1e-12): x == x is all ones; kind 0 is `>`: none.
    let eq = ad_ops::nsl_tensor_compare(x, x, 4);
    let gt = ad_ops::nsl_tensor_compare(x, x, 0);
    assert_eq!(values(eq), [1.0, 1.0, 1.0]);
    assert_eq!(values(gt), [0.0, 0.0, 0.0]);
    nsl_tensor_free(eq);
    nsl_tensor_free(gt);
    nsl_tensor_free(x);
}

#[test]
fn where_same_handle_for_every_input() {
    let x = make_tensor_f64(&[0.0, 2.0, -3.0]);
    // where(x, x, x): nonzero picks the true branch (x), zero the false (x).
    let y = ad_ops::nsl_tensor_where(x, x, x);
    assert_eq!(values(y), [0.0, 2.0, -3.0]);
    nsl_tensor_free(y);
    nsl_tensor_free(x);
}

#[test]
fn cat_and_stack_same_handle_twice() {
    let x = make_tensor_f64(&[1.0, 2.0]);
    let l = list_of(&[x, x]);
    let c = shape_ops::nsl_tensor_cat(l, 0);
    assert_eq!(values(c), [1.0, 2.0, 1.0, 2.0]);
    assert_eq!(shape(c), [4]);
    let s = shape_ops::nsl_tensor_stack(l, 0);
    assert_eq!(values(s), [1.0, 2.0, 1.0, 2.0]);
    assert_eq!(shape(s), [2, 2]);
    nsl_tensor_free(c);
    nsl_tensor_free(s);
    crate::list::nsl_list_free(l);
    nsl_tensor_free(x);
}

#[test]
fn add_inplace_onto_itself() {
    // `x += x` — dst and src are one handle.
    let x = make_tensor_f64(&[1.0, 2.5, -4.0]);
    nsl_tensor_add_inplace(x, x);
    assert_eq!(values(x), [2.0, 5.0, -8.0]);
    nsl_tensor_free(x);
}

#[test]
fn copy_data_onto_itself_is_a_no_op() {
    let x = make_tensor_f64(&[3.0, 1.0]);
    nsl_tensor_copy_data(x, x);
    assert_eq!(values(x), [3.0, 1.0]);
    nsl_tensor_free(x);
}

#[test]
fn scalar_mul_add_inplace_onto_itself() {
    // m = m + s * m
    let m = make_tensor_f64(&[1.0, -2.0, 4.0]);
    arithmetic::nsl_tensor_scalar_mul_add_inplace(m, m, 0.5);
    assert_eq!(values(m), [1.5, -3.0, 6.0]);
    nsl_tensor_free(m);
}

#[test]
fn wgrad_accum_with_activation_aliasing_grad() {
    // m[d, o] += s * x^T g with x == g (both [n, d] = [n, o]).
    let m = make_tensor_2d_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let x = make_tensor_2d_f64(1, 2, &[2.0, 3.0]);
    arithmetic::nsl_tensor_wgrad_accum(m, x, x, 1.0);
    // x^T x = [[4, 6], [6, 9]]
    assert_eq!(values(m), [5.0, 8.0, 9.0, 13.0]);
    nsl_tensor_free(x);
    nsl_tensor_free(m);
}

#[test]
fn reduce_to_shape_onto_own_shape() {
    let g = make_tensor_f64(&[1.0, 2.0, 3.0]);
    let r = ad_ops::nsl_tensor_reduce_to_shape(g, g);
    assert_eq!(values(r), [1.0, 2.0, 3.0]);
    if r != g {
        nsl_tensor_free(r);
    }
    nsl_tensor_free(g);
}

#[test]
fn activation_backward_with_grad_aliasing_input() {
    let x = make_tensor_f64(&[0.0, 1.0, -1.0]);
    // Each backward reads both handles; the values are checked against the
    // closed forms with the same tolerance the module tests use.
    let silu = activation::nsl_tensor_silu_backward(x, x);
    let gelu = activation::nsl_tensor_gelu_backward(x, x);
    let sigm = activation::nsl_tensor_sigmoid_backward(x, x);
    let tanh = activation::nsl_tensor_tanh_backward(x, x);
    let xs = [0.0f64, 1.0, -1.0];
    for (i, &v) in xs.iter().enumerate() {
        let sg = 1.0 / (1.0 + (-v).exp());
        let silu_ref = v * (sg * (1.0 + v * (1.0 - sg)));
        assert!((values(silu)[i] - silu_ref).abs() < 1e-6, "silu at {i}");
        // sigmoid_backward(grad, y) = grad * y * (1 - y); tanh_backward = grad * (1 - y^2)
        assert!((values(sigm)[i] - v * v * (1.0 - v)).abs() < 1e-12, "sigmoid at {i}");
        assert!((values(tanh)[i] - v * (1.0 - v * v)).abs() < 1e-12, "tanh at {i}");
    }
    // gelu'(0) = 0.5, so grad(0) * 0.5 = 0; the other two are finite.
    assert_eq!(values(gelu)[0], 0.0);
    assert!(values(gelu)[1].is_finite() && values(gelu)[2].is_finite());
    for p in [silu, gelu, sigm, tanh, x] {
        nsl_tensor_free(p);
    }
}

#[test]
fn rmsnorm_with_weight_aliasing_input() {
    let x = make_tensor_f64(&[3.0, 4.0]);
    let y = nsl_tensor_rmsnorm(x, x, 0.0);
    // rms = sqrt((9 + 16) / 2) = sqrt(12.5); y_i = x_i / rms * x_i
    let rms = 12.5f64.sqrt();
    let got = values(y);
    assert!((got[0] - 9.0 / rms).abs() < 1e-9);
    assert!((got[1] - 16.0 / rms).abs() < 1e-9);
    nsl_tensor_free(y);
    nsl_tensor_free(x);
}

#[test]
fn layernorm_with_weight_and_bias_aliasing_input() {
    let x = make_tensor_f64(&[1.0, 3.0]);
    let y = nsl_tensor_layernorm(x, x, x, 0.0);
    // mean 2, var 1: normalized = [-1, 1]; y = n * x + x = [-1 + 1, 3 + 3]
    let got = values(y);
    assert!((got[0] - 0.0).abs() < 1e-9, "{got:?}");
    assert!((got[1] - 6.0).abs() < 1e-9, "{got:?}");
    nsl_tensor_free(y);
    nsl_tensor_free(x);
}

#[test]
fn to_device_like_with_itself_on_cpu() {
    let x = make_tensor_f64(&[1.0]);
    let y = nsl_tensor_to_device_like(x, x);
    assert_eq!(values(y), [1.0]);
    if y != x {
        nsl_tensor_free(y);
    } else {
        // Same-device returns the same tensor with a refcount bump.
        nsl_tensor_free(y);
    }
    nsl_tensor_free(x);
}

#[test]
fn cast_into_itself() {
    // `cast_into` takes f32 / f16 / bf16 sources, so the probe is an f32
    // tensor cast onto itself.
    let x = crate::cpu::create_tensor_with_shape_rs_dtype(&[2], DTYPE_F32);
    {
        let t = NslTensor::from_ptr(x);
        unsafe {
            *t.data_f32() = 2.0;
            *t.data_f32().add(1) = -2.0;
        }
    }
    precision_cast::nsl_tensor_cast_into(x, x);
    let t = NslTensor::from_ptr_ref(x);
    assert_eq!(unsafe { *t.data_f32() }, 2.0);
    assert_eq!(unsafe { *t.data_f32().add(1) }, -2.0);
    nsl_tensor_free(x);
}

#[test]
fn mse_backward_with_pred_aliasing_target_and_grad() {
    let x = make_tensor_f64(&[1.0, 2.0]);
    let g = ad_ops::nsl_mse_backward(x, x, x);
    // d/dpred mean((pred - target)^2) = 2 (pred - target) / n = 0, times grad.
    assert_eq!(values(g), [0.0, 0.0]);
    nsl_tensor_free(g);
    nsl_tensor_free(x);
}
