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
    assert!(
        header.contains("#ifndef NSL_MODEL_H"),
        "missing header guard open: {header}"
    );
    assert!(
        header.contains("#define NSL_MODEL_H"),
        "missing header guard define: {header}"
    );
    assert!(
        header.contains("#endif"),
        "missing header guard close: {header}"
    );

    // Required types
    assert!(
        header.contains("typedef struct NslModel NslModel"),
        "missing NslModel typedef: {header}"
    );
    assert!(
        header.contains("NslTensorDesc"),
        "missing NslTensorDesc: {header}"
    );

    // Lifecycle
    assert!(
        header.contains("nsl_model_create"),
        "missing nsl_model_create proto: {header}"
    );
    assert!(
        header.contains("nsl_model_destroy"),
        "missing nsl_model_destroy proto: {header}"
    );

    // Export functions — prototypes
    assert!(
        header.contains("int forward(NslModel"),
        "missing forward prototype: {header}"
    );
    assert!(
        header.contains("const NslTensorDesc*"),
        "missing const NslTensorDesc param type: {header}"
    );
    assert!(
        header.contains("NslTensorDesc* __ret"),
        "missing __ret out-param: {header}"
    );

    // Renamed symbol
    assert!(
        header.contains("int predict(NslModel"),
        "missing predict prototype: {header}"
    );

    // The raw NSL name should NOT appear as a prototype (predict replaces it)
    assert!(
        !header.contains("int inference_forward("),
        "raw name leaked into header: {header}"
    );
}

#[test]
fn header_has_extern_c_guards() {
    let exports = sample_export_functions();
    let header = emit(&exports, "model");
    assert!(
        header.contains("#ifdef __cplusplus"),
        "missing extern C guard: {header}"
    );
    assert!(
        header.contains("extern \"C\""),
        "missing extern C block: {header}"
    );
}
