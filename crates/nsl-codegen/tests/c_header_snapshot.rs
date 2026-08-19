//! M62 C header emission snapshot test.

use nsl_codegen::c_header::{
    emit, ExportDevice, ExportDtype, ExportInfo, ExportParamInfo, ExportTypeInfo,
};

fn sample_export_functions() -> Vec<ExportInfo> {
    vec![
        ExportInfo {
            symbol_name: "forward".into(),
            raw_name: "forward".into(),
            params: vec![ExportParamInfo {
                name: "x".into(),
                ty: ExportTypeInfo::Tensor {
                    shape: vec!["B".into(), "768".into()],
                    dtype: ExportDtype::F32,
                    device: ExportDevice::Any,
                },
            }],
            return_type: ExportTypeInfo::Tensor {
                shape: vec!["B".into(), "1000".into()],
                dtype: ExportDtype::F32,
                device: ExportDevice::Any,
            },
        },
        ExportInfo {
            symbol_name: "predict".into(),
            raw_name: "inference_forward".into(),
            params: vec![ExportParamInfo {
                name: "x".into(),
                ty: ExportTypeInfo::Tensor {
                    shape: vec!["B".into(), "768".into()],
                    dtype: ExportDtype::F32,
                    device: ExportDevice::Any,
                },
            }],
            return_type: ExportTypeInfo::Tensor {
                shape: vec!["B".into(), "10".into()],
                dtype: ExportDtype::F32,
                device: ExportDevice::Any,
            },
        },
    ]
}

#[test]
fn header_contains_expected_prototypes() {
    let exports = sample_export_functions();
    let header = emit(&exports, "model");

    // Header guard
    assert!(header.contains("#ifndef NSL_MODEL_H"), "missing header guard open: {header}");
    assert!(header.contains("#define NSL_MODEL_H"), "missing header guard define: {header}");
    assert!(header.contains("#endif"), "missing header guard close: {header}");

    // Required types
    assert!(header.contains("typedef struct NslModel NslModel"), "missing NslModel typedef: {header}");
    assert!(header.contains("NslTensorDesc"), "missing NslTensorDesc: {header}");
    // tape_id field carries autodiff identity across desc round-trips
    // (per-call grad context, Spec B). Pinning the field here so a
    // future struct-shrinking refactor is caught before the runtime
    // layout silently diverges.
    assert!(
        header.contains("int64_t  tape_id;"),
        "missing tape_id field in NslTensorDesc emission: {header}"
    );

    // Lifecycle
    assert!(header.contains("nsl_model_create"), "missing nsl_model_create proto: {header}");
    assert!(header.contains("nsl_model_destroy"), "missing nsl_model_destroy proto: {header}");

    // Item-7 ownership models + introspection, WITH their ownership
    // commentary — the comments are the contract a C host reads.
    assert!(
        header.contains("int64_t   nsl_model_call_into(NslModel* model, const char* name,"),
        "missing nsl_model_call_into proto: {header}"
    );
    assert!(
        header.contains("const uint64_t* out_capacities);"),
        "call_into must declare the capacity array: {header}"
    );
    assert!(
        header.contains("int64_t   nsl_model_call_alloc(NslModel* model, const char* name,"),
        "missing nsl_model_call_alloc proto: {header}"
    );
    assert!(
        header.contains("DLManagedTensor** out_dl"),
        "call_alloc must take DLManagedTensor** slots: {header}"
    );
    assert!(
        header.contains("typedef struct DLManagedTensor DLManagedTensor;"),
        "the opaque DLManagedTensor typedef must precede its use: {header}"
    );
    assert!(
        header.contains("deleter releases the underlying"),
        "the alloc prototype must document the exactly-once deleter contract: {header}"
    );
    assert!(
        header.contains("const char* nsl_model_get_export_signature(NslModel* model, const char* name);"),
        "missing nsl_model_get_export_signature proto: {header}"
    );
    assert!(
        header.contains("valid until\n * nsl_model_destroy") || header.contains("valid until"),
        "the signature prototype must document the borrowed-pointer lifetime: {header}"
    );

    // Export functions — prototypes
    assert!(header.contains("int forward(NslModel"), "missing forward prototype: {header}");
    assert!(header.contains("const NslTensorDesc*"), "missing const NslTensorDesc param type: {header}");
    assert!(header.contains("NslTensorDesc* __ret"), "missing __ret out-param: {header}");

    // Renamed symbol
    assert!(header.contains("int predict(NslModel"), "missing predict prototype: {header}");

    // The raw NSL name should NOT appear as a prototype (predict replaces it)
    assert!(!header.contains("int inference_forward("), "raw name leaked into header: {header}");
}

#[test]
fn header_has_extern_c_guards() {
    let exports = sample_export_functions();
    let header = emit(&exports, "model");
    assert!(header.contains("#ifdef __cplusplus"), "missing extern C guard: {header}");
    assert!(header.contains("extern \"C\""), "missing extern C block: {header}");
}
