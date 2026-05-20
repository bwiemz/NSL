//! Spec C §2.3 — unit tests for the ORT→NSL dtype conversion helper.
//!
//! The full mock `OrtKernelContext` end-to-end test is deferred to T6 (real
//! ORT in the gated CI job). Here we just verify the dtype mapping uses the
//! `NslTensorDesc` C-API convention (0=f32, 1=f64, ...), NOT the inverted
//! `NslTensor` internal convention (0=f64, 1=f32). See `c_api/mod.rs:29-40`.

#![cfg(feature = "onnx-rt-op")]

use nsl_runtime::onnx_rt_op::{kernel, vendored::ONNXTensorElementDataType};

#[test]
fn ort_dtype_maps_to_nsl_dtype_for_f32() {
    let dtype = kernel::ort_element_type_to_nsl_dtype(ONNXTensorElementDataType::FLOAT);
    assert_eq!(dtype, 0); // C-API convention: 0=f32
}

#[test]
fn ort_dtype_maps_to_nsl_dtype_for_f64() {
    let dtype = kernel::ort_element_type_to_nsl_dtype(ONNXTensorElementDataType::DOUBLE);
    assert_eq!(dtype, 1); // C-API convention: 1=f64
}

#[test]
fn ort_dtype_maps_to_nsl_dtype_for_f16() {
    let dtype = kernel::ort_element_type_to_nsl_dtype(ONNXTensorElementDataType::FLOAT16);
    assert_eq!(dtype, 2);
}

#[test]
fn ort_dtype_maps_to_nsl_dtype_for_bf16() {
    let dtype = kernel::ort_element_type_to_nsl_dtype(ONNXTensorElementDataType::BFLOAT16);
    assert_eq!(dtype, 3);
}

#[test]
fn ort_dtype_maps_to_nsl_dtype_for_i32() {
    let dtype = kernel::ort_element_type_to_nsl_dtype(ONNXTensorElementDataType::INT32);
    assert_eq!(dtype, 4);
}

#[test]
fn ort_dtype_maps_to_nsl_dtype_for_i64() {
    let dtype = kernel::ort_element_type_to_nsl_dtype(ONNXTensorElementDataType::INT64);
    assert_eq!(dtype, 5);
}

#[test]
fn ort_dtype_maps_to_nsl_dtype_for_i8() {
    let dtype = kernel::ort_element_type_to_nsl_dtype(ONNXTensorElementDataType::INT8);
    assert_eq!(dtype, 6);
}

#[test]
fn ort_dtype_maps_to_nsl_dtype_for_u8() {
    let dtype = kernel::ort_element_type_to_nsl_dtype(ONNXTensorElementDataType::UINT8);
    assert_eq!(dtype, 7);
}

#[test]
fn ort_dtype_unsupported_returns_minus_one() {
    // COMPLEX64 is a real ORT enum value but not in our supported mapping.
    let dtype = kernel::ort_element_type_to_nsl_dtype(ONNXTensorElementDataType::COMPLEX64);
    assert_eq!(dtype, -1);
}

#[test]
fn ort_dtype_undefined_returns_minus_one() {
    let dtype = kernel::ort_element_type_to_nsl_dtype(ONNXTensorElementDataType::UNDEFINED);
    assert_eq!(dtype, -1);
}
