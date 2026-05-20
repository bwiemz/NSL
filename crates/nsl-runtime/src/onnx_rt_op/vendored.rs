//! Vendored ORT C API struct definitions — `ORT_API_VERSION = 22` (ORT 1.22.x).
//!
//! These types are `#[repr(C)]` mirrors of the upstream
//! `onnxruntime_c_api.h`. The `verify-ort-vendoring.sh` script (run in CI by
//! the `test-onnx-rt` job) diffs symbols here against bindgen output from the
//! vendored header at `third_party/onnxruntime-1.22.0/include/onnxruntime_c_api.h`
//! — any drift fails CI.
//!
//! **Do not edit without re-running `tools/verify-ort-vendoring.sh` and
//! re-counting opaque slot sizes against the upstream header.** Mis-sized
//! opaque arrays in `OrtApi` will silently shift typed function pointers to
//! the wrong slot, causing reads of garbage when ORT calls these functions.
//!
//! ## Sizing audit for `OrtApi`
//!
//! Inspected `third_party/onnxruntime-1.22.0/include/onnxruntime_c_api.h`
//! lines 808..5254 (the `struct OrtApi { ... };` block). The full struct
//! contains **318 function-pointer fields**. We type the 16 calls Spec C
//! actually invokes and leave the rest as opaque `*const c_void` arrays
//! sized to the exact gap counts. Position numbers below are 1-indexed
//! within the OrtApi struct as enumerated in upstream order:
//!
//! ```text
//!   1   CreateStatus                              (typed)
//!  2..  GetErrorCode, GetErrorMessage, ...        (opaque _gap0[26], 26 fps)
//!  27   CreateCustomOpDomain                      (typed)
//!  28   CustomOpDomain_Add                        (typed)
//!  29   AddCustomOpDomain                         (typed)
//!  30..51  RegisterCustomOpsLibrary, ...           (opaque _gap1[22], 22 fps)
//!  52   GetTensorMutableData                      (typed)
//!  53..60  FillStringTensor, ...                   (opaque _gap2[8], 8 fps)
//!  61   GetTensorElementType                      (typed)
//!  62   GetDimensionsCount                        (typed)
//!  63   GetDimensions                             (typed)
//!  64   GetSymbolicDimensions                     (opaque _gap3[1], 1 fp)
//!  65   GetTensorShapeElementCount                (typed)
//!  66   GetTensorTypeAndShape                     (typed)
//!  67..88  GetTypeInfo, ...                        (opaque _gap4[22], 22 fps)
//!  89   KernelContext_GetInputCount               (typed)
//!  90   KernelContext_GetOutputCount              (typed)
//!  91   KernelContext_GetInput                    (typed)
//!  92   KernelContext_GetOutput                   (typed)
//!  93   ReleaseEnv                                (opaque _gap5[1], 1 fp)
//!  94   ReleaseStatus                             (typed)
//!  95..99  ReleaseMemoryInfo, ...                  (opaque _gap6[5], 5 fps)
//!  100  ReleaseTensorTypeAndShapeInfo             (typed)
//!  101..318  ReleaseSessionOptions, ...            (opaque _gap7[218], 218 fps)
//! ```
//!
//! Total typed: 16. Total opaque: 26+22+8+1+22+1+5+218 = 303. Sum: 319 — wait,
//! that's 1 over the 318 count. The discrepancy: `GetSymbolicDimensions` is
//! at position 64 between `GetDimensions` (63) and `GetTensorShapeElementCount`
//! (65), so the math is 26 + 22 + 8 + 1 + 22 + 1 + 5 + 218 = 303 opaque slots,
//! plus 16 typed slots = 319 total. Re-counting upstream functions
//! confirms there are exactly **318** function-pointer fields, so one of the
//! gap sizes is off by one. Cross-check with the enumeration in
//! `/tmp/ort_api_fns.txt` (generated from upstream) shows:
//!
//! - positions 95..99 = 5 functions: ReleaseMemoryInfo, ReleaseSession,
//!   ReleaseValue, ReleaseRunOptions, ReleaseTypeInfo. Correct.
//! - positions 101..318 = 218 functions. Correct.
//!
//! The off-by-one is in `_gap0`: positions 2..26 = 25 functions (not 26),
//! since position 27 is `CreateCustomOpDomain` (typed). Fixed below to 25.
//!
//! **Final tally:** 25 + 22 + 8 + 1 + 22 + 1 + 5 + 218 = 302 opaque + 16 typed
//! = 318 total. Matches upstream.
//!
//! ## Sizing audit for `OrtCustomOp`
//!
//! Inspected lines 5281..5366. The struct has **1 `u32` (`version`) and 25
//! function-pointer fields** = 26 fields total. All are modeled below.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![allow(clippy::missing_safety_doc)]

use std::os::raw::{c_char, c_void};

/// `ORT_API_VERSION` value that this vendored layout was generated against.
/// Used in `RegisterCustomOps` as the argument to `OrtApiBase::GetApi(...)`;
/// a NULL return signals a version mismatch and surfaces an
/// `OrtVersionUnsupported` status to ORT.
pub const EXPECTED_ORT_API_VERSION: u32 = 22;

// --- Opaque handles --------------------------------------------------------
//
// Each ORT opaque struct is modeled as a newtype around `*mut c_void`. We
// only ever pass them around by pointer; their interior layout is owned by
// ORT and is not part of this vendoring.

#[repr(C)]
pub struct OrtSession(pub *mut c_void);

#[repr(C)]
pub struct OrtSessionOptions(pub *mut c_void);

#[repr(C)]
pub struct OrtStatus(pub *mut c_void);

#[repr(C)]
pub struct OrtCustomOpDomain(pub *mut c_void);

#[repr(C)]
pub struct OrtKernelInfo(pub *mut c_void);

#[repr(C)]
pub struct OrtKernelContext(pub *mut c_void);

#[repr(C)]
pub struct OrtValue(pub *mut c_void);

#[repr(C)]
pub struct OrtMemoryInfo(pub *mut c_void);

#[repr(C)]
pub struct OrtAllocator(pub *mut c_void);

#[repr(C)]
pub struct OrtTensorTypeAndShapeInfo(pub *mut c_void);

#[repr(C)]
pub struct OrtShapeInferContext(pub *mut c_void);

// --- Enums ----------------------------------------------------------------

/// Mirror of `enum OrtErrorCode` (upstream lines 245..258).
///
/// Stored as `u32` rather than `#[repr(u32)] enum` to avoid the
/// `conflicting-repr-hints` issue when combined with `#[derive]` on Rust
/// editions that emit hidden discriminant traits.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrtErrorCode(pub u32);

impl OrtErrorCode {
    pub const ORT_OK: OrtErrorCode = OrtErrorCode(0);
    pub const ORT_FAIL: OrtErrorCode = OrtErrorCode(1);
    pub const ORT_INVALID_ARGUMENT: OrtErrorCode = OrtErrorCode(2);
    pub const ORT_NO_SUCHFILE: OrtErrorCode = OrtErrorCode(3);
    pub const ORT_NO_MODEL: OrtErrorCode = OrtErrorCode(4);
    pub const ORT_ENGINE_ERROR: OrtErrorCode = OrtErrorCode(5);
    pub const ORT_RUNTIME_EXCEPTION: OrtErrorCode = OrtErrorCode(6);
    pub const ORT_INVALID_PROTOBUF: OrtErrorCode = OrtErrorCode(7);
    pub const ORT_MODEL_LOADED: OrtErrorCode = OrtErrorCode(8);
    pub const ORT_NOT_IMPLEMENTED: OrtErrorCode = OrtErrorCode(9);
    pub const ORT_INVALID_GRAPH: OrtErrorCode = OrtErrorCode(10);
    pub const ORT_EP_FAIL: OrtErrorCode = OrtErrorCode(11);
}

/// Mirror of `enum ONNXTensorElementDataType` (upstream lines 187..210).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ONNXTensorElementDataType(pub u32);

impl ONNXTensorElementDataType {
    pub const UNDEFINED: ONNXTensorElementDataType = ONNXTensorElementDataType(0);
    pub const FLOAT: ONNXTensorElementDataType = ONNXTensorElementDataType(1); // f32
    pub const UINT8: ONNXTensorElementDataType = ONNXTensorElementDataType(2);
    pub const INT8: ONNXTensorElementDataType = ONNXTensorElementDataType(3);
    pub const UINT16: ONNXTensorElementDataType = ONNXTensorElementDataType(4);
    pub const INT16: ONNXTensorElementDataType = ONNXTensorElementDataType(5);
    pub const INT32: ONNXTensorElementDataType = ONNXTensorElementDataType(6);
    pub const INT64: ONNXTensorElementDataType = ONNXTensorElementDataType(7);
    pub const STRING: ONNXTensorElementDataType = ONNXTensorElementDataType(8);
    pub const BOOL: ONNXTensorElementDataType = ONNXTensorElementDataType(9);
    pub const FLOAT16: ONNXTensorElementDataType = ONNXTensorElementDataType(10);
    pub const DOUBLE: ONNXTensorElementDataType = ONNXTensorElementDataType(11);
    pub const UINT32: ONNXTensorElementDataType = ONNXTensorElementDataType(12);
    pub const UINT64: ONNXTensorElementDataType = ONNXTensorElementDataType(13);
    pub const COMPLEX64: ONNXTensorElementDataType = ONNXTensorElementDataType(14);
    pub const COMPLEX128: ONNXTensorElementDataType = ONNXTensorElementDataType(15);
    pub const BFLOAT16: ONNXTensorElementDataType = ONNXTensorElementDataType(16);
}

/// Mirror of `enum OrtCustomOpInputOutputCharacteristic` (upstream lines
/// 5271..5275). Returned by `OrtCustomOp::GetInputCharacteristic` /
/// `GetOutputCharacteristic`. Default `INPUT_OUTPUT_REQUIRED` = 0.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrtCustomOpInputOutputCharacteristic(pub u32);

impl OrtCustomOpInputOutputCharacteristic {
    pub const INPUT_OUTPUT_REQUIRED: Self = Self(0);
    pub const INPUT_OUTPUT_OPTIONAL: Self = Self(1);
    pub const INPUT_OUTPUT_VARIADIC: Self = Self(2);
}

/// Mirror of `enum OrtMemType` (upstream lines 395..400). Note the signed
/// values; modeled here as `i32` for that reason.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrtMemType(pub i32);

impl OrtMemType {
    pub const CPU_INPUT: OrtMemType = OrtMemType(-2);
    pub const CPU_OUTPUT: OrtMemType = OrtMemType(-1);
    pub const CPU: OrtMemType = OrtMemType(-1); // alias for CPU_OUTPUT
    pub const DEFAULT: OrtMemType = OrtMemType(0);
}

// --- OrtApiBase: stable version-negotiation handshake ---------------------
//
// Upstream layout (lines 738..755):
//   const OrtApi*(* GetApi)(uint32_t version);
//   const char*  (* GetVersionString)(void);

#[repr(C)]
pub struct OrtApiBase {
    /// Returns a pointer to the OrtApi struct for the requested version,
    /// or NULL if the requested version is not supported.
    pub GetApi: unsafe extern "C" fn(version: u32) -> *const OrtApi,
    /// Returns a null-terminated string of the ORT version (e.g. "1.22.0").
    pub GetVersionString: unsafe extern "C" fn() -> *const c_char,
}

// --- OrtCustomOp: vtable for each registered custom op --------------------
//
// Upstream layout (lines 5281..5366). 1 u32 + 25 function pointers.
// We model EVERY field — ORT writes/reads through these slots and shorting
// the struct would cause it to read past the end.

#[repr(C)]
pub struct OrtCustomOp {
    /// Must equal `EXPECTED_ORT_API_VERSION` when this op is registered.
    pub version: u32,

    /// v1 kernel constructor. Spec C uses this slot (returns a `Box<…>`-style
    /// kernel state pointer cast to `*mut c_void`). v2 alternative below is
    /// stubbed.
    pub CreateKernel: unsafe extern "C" fn(
        op: *const OrtCustomOp,
        api: *const OrtApi,
        info: *const OrtKernelInfo,
    ) -> *mut c_void,

    pub GetName:
        unsafe extern "C" fn(op: *const OrtCustomOp) -> *const c_char,
    pub GetExecutionProviderType:
        unsafe extern "C" fn(op: *const OrtCustomOp) -> *const c_char,

    pub GetInputType: unsafe extern "C" fn(
        op: *const OrtCustomOp,
        index: usize,
    ) -> ONNXTensorElementDataType,
    pub GetInputTypeCount:
        unsafe extern "C" fn(op: *const OrtCustomOp) -> usize,
    pub GetOutputType: unsafe extern "C" fn(
        op: *const OrtCustomOp,
        index: usize,
    ) -> ONNXTensorElementDataType,
    pub GetOutputTypeCount:
        unsafe extern "C" fn(op: *const OrtCustomOp) -> usize,

    /// v1 compute callback. Spec C uses this slot.
    pub KernelCompute: unsafe extern "C" fn(
        op_kernel: *mut c_void,
        context: *mut OrtKernelContext,
    ),
    pub KernelDestroy: unsafe extern "C" fn(op_kernel: *mut c_void),

    pub GetInputCharacteristic: unsafe extern "C" fn(
        op: *const OrtCustomOp,
        index: usize,
    ) -> OrtCustomOpInputOutputCharacteristic,
    pub GetOutputCharacteristic: unsafe extern "C" fn(
        op: *const OrtCustomOp,
        index: usize,
    ) -> OrtCustomOpInputOutputCharacteristic,

    pub GetInputMemoryType: unsafe extern "C" fn(
        op: *const OrtCustomOp,
        index: usize,
    ) -> OrtMemType,

    pub GetVariadicInputMinArity:
        unsafe extern "C" fn(op: *const OrtCustomOp) -> i32,
    pub GetVariadicInputHomogeneity:
        unsafe extern "C" fn(op: *const OrtCustomOp) -> i32,
    pub GetVariadicOutputMinArity:
        unsafe extern "C" fn(op: *const OrtCustomOp) -> i32,
    pub GetVariadicOutputHomogeneity:
        unsafe extern "C" fn(op: *const OrtCustomOp) -> i32,

    /// v2 kernel constructor (status-returning). Spec C registers v1
    /// (`CreateKernel`) and leaves this populated with an unreachable stub
    /// (set by T4 registry code).
    pub CreateKernelV2: unsafe extern "C" fn(
        op: *const OrtCustomOp,
        api: *const OrtApi,
        info: *const OrtKernelInfo,
        kernel_out: *mut *mut c_void,
    ) -> *mut OrtStatus,

    /// v2 compute callback (status-returning). Same v1/v2 story — Spec C sets
    /// `KernelCompute` (v1) and stubs this.
    pub KernelComputeV2: unsafe extern "C" fn(
        op_kernel: *mut c_void,
        context: *mut OrtKernelContext,
    ) -> *mut OrtStatus,

    pub InferOutputShapeFn: unsafe extern "C" fn(
        op: *const OrtCustomOp,
        ctx: *mut OrtShapeInferContext,
    ) -> *mut OrtStatus,

    pub GetStartVersion:
        unsafe extern "C" fn(op: *const OrtCustomOp) -> i32,
    pub GetEndVersion:
        unsafe extern "C" fn(op: *const OrtCustomOp) -> i32,

    /// May-inplace API (ORT 1.16+). Spec C does not use inplace; the slot is
    /// populated with a stub returning 0.
    pub GetMayInplace: unsafe extern "C" fn(
        input_index: *mut *mut i32,
        output_index: *mut *mut i32,
    ) -> usize,
    pub ReleaseMayInplace: unsafe extern "C" fn(
        input_index: *mut i32,
        output_index: *mut i32,
    ),

    /// Alias-map API (ORT 1.16+). Spec C does not use aliasing; slots are
    /// populated with stubs returning 0.
    pub GetAliasMap: unsafe extern "C" fn(
        input_index: *mut *mut i32,
        output_index: *mut *mut i32,
    ) -> usize,
    pub ReleaseAliasMap: unsafe extern "C" fn(
        input_index: *mut i32,
        output_index: *mut i32,
    ),
}

// --- OrtApi: the versioned function table ---------------------------------
//
// See module-level comment for the full gap-sizing audit. Only the 16
// functions Spec C invokes are typed; the rest are reserved as opaque
// `*const c_void` arrays so ORT's internal callers can still find their
// targets at the correct offsets.
//
// Each opaque slot is exactly one function pointer worth of storage.
// `*const c_void` is the natural unit because (a) all OrtApi fields are
// function pointers (`size_of::<fn(..)>() == size_of::<*const c_void>()`
// on every Tier-1 target), and (b) using `c_void` rather than `()` keeps
// `repr(C)` semantics unambiguous.

#[repr(C)]
pub struct OrtApi {
    // Position 1.
    /// Allocate and initialize an OrtStatus with the given code and message.
    pub CreateStatus: unsafe extern "C" fn(
        code: OrtErrorCode,
        msg: *const c_char,
    ) -> *mut OrtStatus,

    // Positions 2..26 (25 fps): GetErrorCode, GetErrorMessage, CreateEnv,
    // CreateEnvWithCustomLogger, EnableTelemetryEvents,
    // DisableTelemetryEvents, CreateSession, CreateSessionFromArray, Run,
    // CreateSessionOptions, SetOptimizedModelFilePath, CloneSessionOptions,
    // SetSessionExecutionMode, EnableProfiling, DisableProfiling,
    // EnableMemPattern, DisableMemPattern, EnableCpuMemArena,
    // DisableCpuMemArena, SetSessionLogId, SetSessionLogVerbosityLevel,
    // SetSessionLogSeverityLevel, SetSessionGraphOptimizationLevel,
    // SetIntraOpNumThreads, SetInterOpNumThreads.
    pub _gap0: [*const c_void; 25],

    // Position 27.
    /// Create a custom-op domain (collection of related custom ops).
    pub CreateCustomOpDomain: unsafe extern "C" fn(
        domain: *const c_char,
        out: *mut *mut OrtCustomOpDomain,
    ) -> *mut OrtStatus,

    // Position 28.
    /// Add a custom op to a domain.
    pub CustomOpDomain_Add: unsafe extern "C" fn(
        custom_op_domain: *mut OrtCustomOpDomain,
        op: *const OrtCustomOp,
    ) -> *mut OrtStatus,

    // Position 29.
    /// Attach a custom-op domain to session options.
    pub AddCustomOpDomain: unsafe extern "C" fn(
        options: *mut OrtSessionOptions,
        custom_op_domain: *mut OrtCustomOpDomain,
    ) -> *mut OrtStatus,

    // Positions 30..51 (22 fps): RegisterCustomOpsLibrary, SessionGetInputCount,
    // SessionGetOutputCount, SessionGetOverridableInitializerCount,
    // SessionGetInputTypeInfo, SessionGetOutputTypeInfo,
    // SessionGetOverridableInitializerTypeInfo, SessionGetInputName,
    // SessionGetOutputName, SessionGetOverridableInitializerName,
    // CreateRunOptions, RunOptionsSetRunLogVerbosityLevel,
    // RunOptionsSetRunLogSeverityLevel, RunOptionsSetRunTag,
    // RunOptionsGetRunLogVerbosityLevel, RunOptionsGetRunLogSeverityLevel,
    // RunOptionsGetRunTag, RunOptionsSetTerminate, RunOptionsUnsetTerminate,
    // CreateTensorAsOrtValue, CreateTensorWithDataAsOrtValue, IsTensor.
    pub _gap1: [*const c_void; 22],

    // Position 52.
    /// Get the data pointer from an OrtValue (typed by element type).
    /// Spec C reads f32 tensors via this call.
    pub GetTensorMutableData: unsafe extern "C" fn(
        value: *mut OrtValue,
        out: *mut *mut c_void,
    ) -> *mut OrtStatus,

    // Positions 53..60 (8 fps): FillStringTensor, GetStringTensorDataLength,
    // GetStringTensorContent, CastTypeInfoToTensorInfo,
    // GetOnnxTypeFromTypeInfo, CreateTensorTypeAndShapeInfo,
    // SetTensorElementType, SetDimensions.
    pub _gap2: [*const c_void; 8],

    // Position 61.
    pub GetTensorElementType: unsafe extern "C" fn(
        info: *const OrtTensorTypeAndShapeInfo,
        out: *mut ONNXTensorElementDataType,
    ) -> *mut OrtStatus,

    // Position 62.
    pub GetDimensionsCount: unsafe extern "C" fn(
        info: *const OrtTensorTypeAndShapeInfo,
        out: *mut usize,
    ) -> *mut OrtStatus,

    // Position 63.
    pub GetDimensions: unsafe extern "C" fn(
        info: *const OrtTensorTypeAndShapeInfo,
        dim_values: *mut i64,
        dim_values_length: usize,
    ) -> *mut OrtStatus,

    // Position 64 (1 fp): GetSymbolicDimensions.
    pub _gap3: [*const c_void; 1],

    // Position 65.
    pub GetTensorShapeElementCount: unsafe extern "C" fn(
        info: *const OrtTensorTypeAndShapeInfo,
        out: *mut usize,
    ) -> *mut OrtStatus,

    // Position 66.
    pub GetTensorTypeAndShape: unsafe extern "C" fn(
        value: *const OrtValue,
        out: *mut *mut OrtTensorTypeAndShapeInfo,
    ) -> *mut OrtStatus,

    // Positions 67..88 (22 fps): GetTypeInfo, GetValueType, CreateMemoryInfo,
    // CreateCpuMemoryInfo, CompareMemoryInfo, MemoryInfoGetName,
    // MemoryInfoGetId, MemoryInfoGetMemType, MemoryInfoGetType,
    // AllocatorAlloc, AllocatorFree, AllocatorGetInfo,
    // GetAllocatorWithDefaultOptions, AddFreeDimensionOverride, GetValue,
    // GetValueCount, CreateValue, CreateOpaqueValue, GetOpaqueValue,
    // KernelInfoGetAttribute_float, KernelInfoGetAttribute_int64,
    // KernelInfoGetAttribute_string.
    pub _gap4: [*const c_void; 22],

    // Position 89.
    pub KernelContext_GetInputCount: unsafe extern "C" fn(
        context: *const OrtKernelContext,
        out: *mut usize,
    ) -> *mut OrtStatus,

    // Position 90.
    pub KernelContext_GetOutputCount: unsafe extern "C" fn(
        context: *const OrtKernelContext,
        out: *mut usize,
    ) -> *mut OrtStatus,

    // Position 91.
    pub KernelContext_GetInput: unsafe extern "C" fn(
        context: *const OrtKernelContext,
        index: usize,
        out: *mut *const OrtValue,
    ) -> *mut OrtStatus,

    // Position 92.
    pub KernelContext_GetOutput: unsafe extern "C" fn(
        context: *mut OrtKernelContext,
        index: usize,
        dim_values: *const i64,
        dim_count: usize,
        out: *mut *mut OrtValue,
    ) -> *mut OrtStatus,

    // Position 93 (1 fp): ReleaseEnv.
    pub _gap5: [*const c_void; 1],

    // Position 94.
    pub ReleaseStatus: unsafe extern "C" fn(status: *mut OrtStatus),

    // Positions 95..99 (5 fps): ReleaseMemoryInfo, ReleaseSession,
    // ReleaseValue, ReleaseRunOptions, ReleaseTypeInfo.
    pub _gap6: [*const c_void; 5],

    // Position 100.
    pub ReleaseTensorTypeAndShapeInfo: unsafe extern "C" fn(
        info: *mut OrtTensorTypeAndShapeInfo,
    ),

    // Positions 101..318 (218 fps): ReleaseSessionOptions,
    // ReleaseCustomOpDomain, GetDenotationFromTypeInfo,
    // CastTypeInfoToMapTypeInfo, CastTypeInfoToSequenceTypeInfo, and all
    // post-V1 additions through GetEpApi (position 318). Audited against
    // `third_party/onnxruntime-1.22.0/include/onnxruntime_c_api.h`
    // lines 808..5254.
    pub _back: [*const c_void; 218],
}

// --- Compile-time layout assertion -----------------------------------------
//
// On every Tier-1 target `*const c_void` and a function pointer are both
// pointer-sized (`size_of::<usize>()`), so `OrtApi` has exactly
// `318 * size_of::<usize>()` bytes. If a future change miscounts a gap, this
// const_assert will fire at build time before any ORT call can read garbage.

const _: () = {
    let expected_bytes = 318 * std::mem::size_of::<*const c_void>();
    let actual_bytes = std::mem::size_of::<OrtApi>();
    assert!(
        actual_bytes == expected_bytes,
        "OrtApi layout drift: vendored struct does not match the 318 \
         function-pointer slots in ORT 1.22.0 onnxruntime_c_api.h. \
         Re-run tools/verify-ort-vendoring.sh and re-audit gap sizes."
    );
};

// `OrtCustomOp` is sized as: 1 u32 + 1 padding-to-pointer + 25 function
// pointers = 8 (header on 64-bit) + 25*ptr. The assertion below checks that
// the field count matches; the byte size depends on platform alignment so we
// only assert "at least the 25 fps fit", not a precise byte count.

const _: () = {
    let min_bytes = 8 + 25 * std::mem::size_of::<*const c_void>();
    let actual_bytes = std::mem::size_of::<OrtCustomOp>();
    assert!(
        actual_bytes >= min_bytes,
        "OrtCustomOp layout drift: vendored struct is smaller than the \
         26 fields documented in ORT 1.22.0 onnxruntime_c_api.h."
    );
};
