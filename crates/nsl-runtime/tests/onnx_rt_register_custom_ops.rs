//! Spec C §2.1 — `RegisterCustomOps` successfully creates a `com.nsl` domain
//! and registers one `OrtCustomOp` per `@export`.
//!
//! Runs **without** linking real ORT. We construct a mock `OrtApiBase`/`OrtApi`
//! whose 318 function-pointer slots are populated entry-by-entry, then call
//! `RegisterCustomOps` and assert it returns `null` (success) on a matching
//! API version and non-null on a version mismatch.
//!
//! The test binary has no `@export`-decorated NSL functions, so
//! `nsl_get_num_exports()` returns 0 and `OPS_ADDED` stays at 0. The
//! `register_creates_one_op_per_export` test therefore only asserts that the
//! status return is null — the actual op-count coverage lives in T6's gated
//! Python E2E job.

#![cfg(feature = "onnx-rt-op")]
#![allow(non_snake_case)]
// Bit-pattern-1 sentinels are intentional: they're never dereferenced, but
// they're distinguishable from null and from any real allocator return —
// exactly what ORT's "non-null = failure" contract wants from a mock.
#![allow(clippy::manual_dangling_ptr)]
// The `as *const OrtApi` casts work around clippy's confusion between the
// `MOCK_API.0` projection's type and the static's wrapper type — leaving
// the cast keeps the intent explicit.
#![allow(clippy::unnecessary_cast)]

use std::os::raw::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use nsl_runtime::onnx_rt_op::{
    vendored::*,
    RegisterCustomOps,
};

// ---------------------------------------------------------------------------
// Mock observation state
// ---------------------------------------------------------------------------

static OPS_ADDED: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// Mock function pointers — only the 16 typed OrtApi slots are observed.
// The 302 gap pointers are null; ORT (here: our test) never calls through
// them, but the layout must be correct for `&MOCK_API` to be ABI-compatible.
// ---------------------------------------------------------------------------

unsafe extern "C" fn mock_get_api(version: u32) -> *const OrtApi {
    if version != EXPECTED_ORT_API_VERSION {
        return std::ptr::null();
    }
    std::ptr::addr_of!(MOCK_API.0) as *const OrtApi
}

unsafe extern "C" fn mock_get_version_string() -> *const c_char {
    c"1.22.0-mock".as_ptr()
}

// Non-null sentinel — never dereferenced by RegisterCustomOps. Cast via
// `as` to avoid a const-context provenance issue (we just need a non-null
// pointer to signal a status to ORT in some paths).
const MOCK_STATUS_SENTINEL: *mut OrtStatus = 1usize as *mut OrtStatus;

unsafe extern "C" fn mock_create_status(
    _code: OrtErrorCode,
    _msg: *const c_char,
) -> *mut OrtStatus {
    MOCK_STATUS_SENTINEL
}

unsafe extern "C" fn mock_release_status(_status: *mut OrtStatus) {}

unsafe extern "C" fn mock_create_custom_op_domain(
    _name: *const c_char,
    out: *mut *mut OrtCustomOpDomain,
) -> *mut OrtStatus {
    // Non-null sentinel — never dereferenced.
    *out = 1usize as *mut OrtCustomOpDomain;
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_custom_op_domain_add(
    _domain: *mut OrtCustomOpDomain,
    _op: *const OrtCustomOp,
) -> *mut OrtStatus {
    OPS_ADDED.fetch_add(1, Ordering::SeqCst);
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_add_custom_op_domain(
    _opts: *mut OrtSessionOptions,
    _domain: *mut OrtCustomOpDomain,
) -> *mut OrtStatus {
    std::ptr::null_mut()
}

// The remaining typed slots are KernelContext / TensorTypeAndShape helpers —
// RegisterCustomOps does not touch them, but their function-pointer types
// must be populated for the OrtApi struct to be initializable.

unsafe extern "C" fn mock_kc_get_input_count(
    _ctx: *const OrtKernelContext,
    out: *mut usize,
) -> *mut OrtStatus {
    *out = 0;
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_kc_get_output_count(
    _ctx: *const OrtKernelContext,
    out: *mut usize,
) -> *mut OrtStatus {
    *out = 0;
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_kc_get_input(
    _ctx: *const OrtKernelContext,
    _index: usize,
    out: *mut *const OrtValue,
) -> *mut OrtStatus {
    *out = std::ptr::null();
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_kc_get_output(
    _ctx: *mut OrtKernelContext,
    _index: usize,
    _dim_values: *const i64,
    _dim_count: usize,
    out: *mut *mut OrtValue,
) -> *mut OrtStatus {
    *out = std::ptr::null_mut();
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_get_tensor_data(
    _value: *mut OrtValue,
    out: *mut *mut c_void,
) -> *mut OrtStatus {
    *out = std::ptr::null_mut();
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_get_tensor_type_and_shape(
    _value: *const OrtValue,
    out: *mut *mut OrtTensorTypeAndShapeInfo,
) -> *mut OrtStatus {
    *out = std::ptr::null_mut();
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_get_tensor_element_type(
    _info: *const OrtTensorTypeAndShapeInfo,
    out: *mut ONNXTensorElementDataType,
) -> *mut OrtStatus {
    *out = ONNXTensorElementDataType::UNDEFINED;
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_get_dimensions_count(
    _info: *const OrtTensorTypeAndShapeInfo,
    out: *mut usize,
) -> *mut OrtStatus {
    *out = 0;
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_get_dimensions(
    _info: *const OrtTensorTypeAndShapeInfo,
    _dim_values: *mut i64,
    _dim_values_length: usize,
) -> *mut OrtStatus {
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_get_tensor_shape_element_count(
    _info: *const OrtTensorTypeAndShapeInfo,
    out: *mut usize,
) -> *mut OrtStatus {
    *out = 0;
    std::ptr::null_mut()
}

unsafe extern "C" fn mock_release_tensor_type_and_shape_info(
    _info: *mut OrtTensorTypeAndShapeInfo,
) {
}

// ---------------------------------------------------------------------------
// Mock OrtApi — every one of the 318 slots populated.
// ---------------------------------------------------------------------------
//
// Position 1: CreateStatus (typed).
// Positions 2..26 (25): _gap0 — opaque nulls.
// Position 27: CreateCustomOpDomain (typed).
// Position 28: CustomOpDomain_Add (typed).
// Position 29: AddCustomOpDomain (typed).
// Positions 30..51 (22): _gap1 — opaque nulls.
// Position 52: GetTensorMutableData (typed).
// Positions 53..60 (8): _gap2 — opaque nulls.
// Position 61: GetTensorElementType (typed).
// Position 62: GetDimensionsCount (typed).
// Position 63: GetDimensions (typed).
// Position 64 (1): _gap3 — opaque null.
// Position 65: GetTensorShapeElementCount (typed).
// Position 66: GetTensorTypeAndShape (typed).
// Positions 67..88 (22): _gap4 — opaque nulls.
// Position 89: KernelContext_GetInputCount (typed).
// Position 90: KernelContext_GetOutputCount (typed).
// Position 91: KernelContext_GetInput (typed).
// Position 92: KernelContext_GetOutput (typed).
// Position 93 (1): _gap5 — opaque null.
// Position 94: ReleaseStatus (typed).
// Positions 95..99 (5): _gap6 — opaque nulls.
// Position 100: ReleaseTensorTypeAndShapeInfo (typed).
// Positions 101..318 (218): _back — opaque nulls.
//
// Total: 16 typed + 302 nulls = 318. Matches `vendored.rs` exactly.

// `OrtApi` (and `OrtApiBase`) contain `*const c_void` slots that aren't
// `Sync`, but they're statically-initialized constant function pointers and
// never mutated. Wrap them in `SyncWrap` so we can keep them in `static`.

#[repr(transparent)]
struct SyncWrap<T>(T);
unsafe impl<T> Sync for SyncWrap<T> {}

static MOCK_API: SyncWrap<OrtApi> = SyncWrap(OrtApi {
    CreateStatus: mock_create_status,
    _gap0: [std::ptr::null(); 25],
    CreateCustomOpDomain: mock_create_custom_op_domain,
    CustomOpDomain_Add: mock_custom_op_domain_add,
    AddCustomOpDomain: mock_add_custom_op_domain,
    _gap1: [std::ptr::null(); 22],
    GetTensorMutableData: mock_get_tensor_data,
    _gap2: [std::ptr::null(); 8],
    GetTensorElementType: mock_get_tensor_element_type,
    GetDimensionsCount: mock_get_dimensions_count,
    GetDimensions: mock_get_dimensions,
    _gap3: [std::ptr::null(); 1],
    GetTensorShapeElementCount: mock_get_tensor_shape_element_count,
    GetTensorTypeAndShape: mock_get_tensor_type_and_shape,
    _gap4: [std::ptr::null(); 22],
    KernelContext_GetInputCount: mock_kc_get_input_count,
    KernelContext_GetOutputCount: mock_kc_get_output_count,
    KernelContext_GetInput: mock_kc_get_input,
    KernelContext_GetOutput: mock_kc_get_output,
    _gap5: [std::ptr::null(); 1],
    ReleaseStatus: mock_release_status,
    _gap6: [std::ptr::null(); 5],
    ReleaseTensorTypeAndShapeInfo: mock_release_tensor_type_and_shape_info,
    _back: [std::ptr::null(); 218],
});

static MOCK_API_BASE: SyncWrap<OrtApiBase> = SyncWrap(OrtApiBase {
    GetApi: mock_get_api,
    GetVersionString: mock_get_version_string,
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn register_creates_one_op_per_export() {
    // The test binary has no `@export` NSL functions, so the codegen
    // FFIs `nsl_get_num_exports` returns 0 and no ops are registered.
    // We only assert RegisterCustomOps succeeds; op-count coverage
    // lives in the gated Python E2E test (T6).
    OPS_ADDED.store(0, Ordering::SeqCst);
    let mut opts = OrtSessionOptions(std::ptr::null_mut());
    let rc = unsafe {
        RegisterCustomOps(
            &mut opts as *mut OrtSessionOptions,
            std::ptr::addr_of!(MOCK_API_BASE.0),
        )
    };
    assert!(
        rc.is_null(),
        "RegisterCustomOps should succeed with mock OrtApiBase (got non-null status)"
    );
    // OPS_ADDED reflects the test binary's export count, which is 0.
    // No stricter assertion — the plan explicitly accepts this vacuous pass.
}

#[test]
fn register_rejects_wrong_api_version() {
    unsafe extern "C" fn bad_get_api(_v: u32) -> *const OrtApi {
        std::ptr::null()
    }
    let bad_api_base = OrtApiBase {
        GetApi: bad_get_api,
        GetVersionString: mock_get_version_string,
    };
    let mut opts = OrtSessionOptions(std::ptr::null_mut());
    let rc = unsafe {
        RegisterCustomOps(
            &mut opts as *mut OrtSessionOptions,
            &bad_api_base as *const OrtApiBase,
        )
    };
    assert!(
        !rc.is_null(),
        "RegisterCustomOps must return non-null status when GetApi returns NULL"
    );
}
