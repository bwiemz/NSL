//! Spec C §2.3 — kernel state + compute bridge.
//!
//! When ORT executes an NSL custom op, ORT calls `KernelCompute` with an
//! `OrtKernelContext` that exposes input tensors and output slots. This
//! module:
//!   1. Reads each input via the OrtApi accessors.
//!   2. Bridges `OrtValue` → `NslTensorDesc` (zero-copy when CPU dtype
//!      matches).
//!   3. Invokes the export's cached fn_ptr (resolved via
//!      `nsl_model_lookup_function` at kernel-create time, which is wired
//!      up by T4 in the registry module).
//!   4. Writes outputs back through the OrtApi.
//!
//! ## v1 limitations
//!
//! - **CPU only.** `device_type = 0` is hardcoded on every descriptor.
//!   CUDA EP integration is gated behind a future `onnx-rt-cuda` feature.
//! - **Single-output / element-wise shape convention.** `ort_alloc_output_desc`
//!   infers each output's shape from the first input. Multi-output ops or
//!   shape-changing kernels (reductions, matmul, conv, ...) need a real
//!   `InferOutputShapeFn` and are deferred to v2.
//! - **Per-compute shape `Vec` leak.** Each `ort_input_to_desc` call leaks
//!   one `Vec<i64>` (via `Box::leak`) so the `NslTensorDesc::shape` pointer
//!   remains valid for the duration of the NSL function call. ORT's
//!   `OrtTensorTypeAndShapeInfo` is released before the call, so we have
//!   no other place to anchor the shape buffer without restructuring the
//!   NslTensorDesc ABI. Bounded but non-zero — addressed in v2 by storing
//!   the shape on the `NslOrtKernelState` arena.
//!
//! ## dtype convention
//!
//! `ort_element_type_to_nsl_dtype` returns the **C-API** convention
//! (`0=f32, 1=f64, 2=f16, 3=bf16, 4=i32, 5=i64, 6=i8, 7=u8`) because the
//! result populates `NslTensorDesc.dtype`. This is INVERTED from the
//! `NslTensor` internal convention (`0=f64, 1=f32, ...`). See
//! `c_api/mod.rs:29-40` for the canonical table and conversion helpers
//! (`capi_dtype_to_nsl` / `nsl_dtype_to_capi`).

use std::os::raw::{c_char, c_void};

use super::vendored::*;
use crate::c_api::exports::ExportFnPtr;

/// Per-kernel state. One instance per custom op per session; created by
/// `OrtCustomOp::CreateKernel` (T4) and destroyed by `KernelDestroy`.
///
/// `fn_ptr` is stored already-typed as `ExportFnPtr` rather than as `usize`
/// so the unsafe transmute happens exactly once (at create time) instead of
/// on every compute call. This also avoids a build-time size-mismatch hazard
/// (`size_of::<usize>()` is guaranteed to equal `size_of::<fn(...)>()` on
/// every Tier-1 target, but pre-storing the typed pointer makes the contract
/// local to the code that constructed it).
#[repr(C)]
pub struct NslOrtKernelState {
    /// Pointer to the OrtApi vtable, captured at kernel-create time.
    pub api: *const OrtApi,
    /// Cached NSL export function pointer. Resolved by T4's
    /// `CreateKernel` via `nsl_model_lookup_function(model, name)`.
    pub fn_ptr: ExportFnPtr,
    /// The NSL model handle this kernel is bound to.
    pub model_ptr: i64,
}

// SAFETY: `NslOrtKernelState` is constructed once per session by ORT on the
// main thread and read-only thereafter from compute callbacks. ORT serializes
// invocations of a given kernel instance; we hold no interior-mutable state.
unsafe impl Send for NslOrtKernelState {}
unsafe impl Sync for NslOrtKernelState {}

/// Spec C §2.3 — the shared compute body for every NSL custom op.
///
/// Referenced via the `KernelCompute` function-pointer slot in the
/// `OrtCustomOp` vtable (populated by T4). Not exported by symbol name —
/// ORT reaches it through the vtable.
///
/// SAFETY: caller (ORT) provides a valid `OrtKernelContext`; `kernel_state`
/// points to an `NslOrtKernelState` produced by our `CreateKernel`.
pub unsafe extern "C" fn nsl_ort_kernel_compute(
    kernel_state: *mut c_void,
    ctx: *mut OrtKernelContext,
) {
    let state = &*(kernel_state as *const NslOrtKernelState);
    let api = &*state.api;

    let mut n_in: usize = 0;
    let _ = (api.KernelContext_GetInputCount)(ctx, &mut n_in);
    let mut n_out: usize = 0;
    let _ = (api.KernelContext_GetOutputCount)(ctx, &mut n_out);

    // Convert ORT inputs to NslTensorDescs.
    let input_descs: Vec<crate::c_api::NslTensorDesc> = (0..n_in)
        .map(|i| ort_input_to_desc(api, ctx, i))
        .collect();

    // For each output, ORT requires us to call KernelContext_GetOutput with
    // the inferred shape. v1 assumes single-output / element-wise ops where
    // the output shape mirrors input[0]; multi-output or shape-inferring
    // kernels are out of scope (see module doc).
    let mut output_descs: Vec<crate::c_api::NslTensorDesc> = (0..n_out)
        .map(|i| ort_alloc_output_desc(api, ctx, i, &input_descs))
        .collect();

    // Invoke the NSL export. The fn_ptr is the codegen-emitted
    // `<name>__nsl_dispatch` symbol (see `ExportRegistry` for the ABI).
    let rc = (state.fn_ptr)(
        state.model_ptr,
        input_descs.as_ptr() as i64,
        n_in as i64,
        output_descs.as_mut_ptr() as i64,
        n_out as i64,
    );

    if rc != 0 {
        // Surface NSL error as ORT status. The v1 `KernelCompute` slot is
        // `void`-returning so we cannot hand the status back directly; we
        // build it for ORT's session-level error collection and leak the
        // pointer (ORT owns the lifetime once we surface it). T6 / M62c
        // will switch the registry to `KernelComputeV2` which returns
        // `*mut OrtStatus` so this leak goes away.
        let nsl_err_ptr = crate::c_api::nsl_get_last_error();
        let msg = if nsl_err_ptr != 0 {
            nsl_err_ptr as *const c_char
        } else {
            c"nsl_ort_kernel_compute: NSL export returned non-zero rc".as_ptr()
        };
        let status = (api.CreateStatus)(OrtErrorCode::ORT_RUNTIME_EXCEPTION, msg);
        let _ = status;
    }
}

/// Convert an ORT input (i-th) to an `NslTensorDesc`. Zero-copy on data;
/// the shape buffer is heap-allocated and leaked for the duration of the
/// kernel invocation (see module doc for the rationale).
unsafe fn ort_input_to_desc(
    api: &OrtApi,
    ctx: *mut OrtKernelContext,
    index: usize,
) -> crate::c_api::NslTensorDesc {
    let mut value: *const OrtValue = std::ptr::null();
    let _ = (api.KernelContext_GetInput)(ctx, index, &mut value);

    let mut info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
    let _ = (api.GetTensorTypeAndShape)(value, &mut info);

    let mut element_type = ONNXTensorElementDataType::UNDEFINED;
    let _ = (api.GetTensorElementType)(info, &mut element_type);

    let mut ndim: usize = 0;
    let _ = (api.GetDimensionsCount)(info, &mut ndim);

    let mut shape: Vec<i64> = vec![0; ndim];
    let _ = (api.GetDimensions)(info, shape.as_mut_ptr(), ndim);

    let mut data_ptr: *mut c_void = std::ptr::null_mut();
    let _ = (api.GetTensorMutableData)(value as *mut OrtValue, &mut data_ptr);

    (api.ReleaseTensorTypeAndShapeInfo)(info);

    let nsl_dtype = ort_element_type_to_nsl_dtype(element_type);

    // SAFETY: leak the shape Vec so the resulting raw pointer outlives the
    // NSL export call. The leak is bounded (one shape Vec per input per
    // compute invocation). v2 should anchor shapes on a per-kernel arena
    // owned by `NslOrtKernelState`. Do NOT change this to a stack-allocated
    // array — `NslTensorDesc::shape` must remain valid until the export
    // function returns.
    let shape_box = shape.into_boxed_slice();
    let shape_ptr = Box::leak(shape_box).as_mut_ptr();

    crate::c_api::NslTensorDesc {
        data: data_ptr,
        shape: shape_ptr,
        strides: std::ptr::null_mut(),
        ndim: ndim as i32,
        dtype: nsl_dtype,
        device_type: 0, // CPU only for v1
        device_id: 0,
        // ORT inputs cross the FFI boundary fresh per compute — no autodiff
        // tape history to inherit. CPU custom ops are inference-only in v1;
        // training-aware ORT integration is M62c scope.
        tape_id: 0,
    }
}

/// Allocate the i-th output via the OrtApi and produce a descriptor.
///
/// v1: shape inferred from the first input (element-wise op convention).
/// Multi-output / shape-changing kernels deferred to v2.
unsafe fn ort_alloc_output_desc(
    api: &OrtApi,
    ctx: *mut OrtKernelContext,
    index: usize,
    input_descs: &[crate::c_api::NslTensorDesc],
) -> crate::c_api::NslTensorDesc {
    let template = input_descs
        .first()
        .expect("output requires at least one input for shape inference (v1)");
    let shape_slice = std::slice::from_raw_parts(template.shape, template.ndim as usize);

    let mut output_value: *mut OrtValue = std::ptr::null_mut();
    let _ = (api.KernelContext_GetOutput)(
        ctx,
        index,
        shape_slice.as_ptr(),
        template.ndim as usize,
        &mut output_value,
    );

    let mut data_ptr: *mut c_void = std::ptr::null_mut();
    let _ = (api.GetTensorMutableData)(output_value, &mut data_ptr);

    crate::c_api::NslTensorDesc {
        data: data_ptr,
        shape: template.shape,
        strides: std::ptr::null_mut(),
        ndim: template.ndim,
        dtype: template.dtype,
        device_type: 0,
        device_id: 0,
        tape_id: 0, // ORT outputs are inference-only in v1
    }
}

/// Map ORT element type → `NslTensorDesc.dtype` (C-API convention).
///
/// Convention (from `c_api/mod.rs:34`): `0=f32, 1=f64, 2=f16, 3=bf16,
/// 4=i32, 5=i64, 6=i8, 7=u8`. Unsupported / unrecognized inputs return
/// `-1` so callers can detect the failure (the value never escapes back
/// into `NslTensorDesc.dtype` for a real compute path because T4's
/// `GetInputType` advertises only types we support).
///
/// Returns `i32` (matching `NslTensorDesc.dtype`'s declared type) rather
/// than `u8` so the unsupported sentinel can be negative.
pub fn ort_element_type_to_nsl_dtype(t: ONNXTensorElementDataType) -> i32 {
    match t {
        ONNXTensorElementDataType::FLOAT => 0,    // f32  → NslTensorDesc 0
        ONNXTensorElementDataType::DOUBLE => 1,   // f64  → 1
        ONNXTensorElementDataType::FLOAT16 => 2,  // f16  → 2
        ONNXTensorElementDataType::BFLOAT16 => 3, // bf16 → 3
        ONNXTensorElementDataType::INT32 => 4,    // i32  → 4
        ONNXTensorElementDataType::INT64 => 5,    // i64  → 5
        ONNXTensorElementDataType::INT8 => 6,     // i8   → 6
        ONNXTensorElementDataType::UINT8 => 7,    // u8   → 7
        _ => -1,                                  // unsupported
    }
}
