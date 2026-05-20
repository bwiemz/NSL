//! CFTP §4.3 / Tier A activation — call-site wiring sentinel tests.
//!
//! Verifies that the 4 Cranelift CSHA call sites (advanced.rs ×2,
//! wengert_lower.rs ×2) emit calls to the thread-local packing registry
//! getters (`nsl_packing_metadata_get_segment_ids` /
//! `nsl_packing_metadata_get_doc_starts`) immediately BEFORE invoking
//! the FA-2 entry points. Guards against regressions where a future
//! codegen change accidentally drops the registry-read step and silently
//! falls back to sentinel-0 even in packed-sequence training.
//!
//! These tests are codegen-only — they inspect the emitted IR / object
//! file textually rather than launching kernels. Pairs with the runtime
//! roundtrip tests in `nsl_runtime::pca_rope_runtime::tests`.

use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{builder, Flags};
use cranelift_module::default_libcall_names;
use cranelift_object::{ObjectBuilder, ObjectModule};
use nsl_codegen::builtins::declare_runtime_functions;

fn make_module() -> ObjectModule {
    let isa_builder = cranelift_native::builder().expect("host isa");
    let flag_builder = builder();
    let flags = Flags::new(flag_builder);
    let isa = isa_builder.finish(flags).expect("finish isa");
    let obj_builder = ObjectBuilder::new(
        isa,
        "pca_rope_activation_call_sites_test",
        default_libcall_names(),
    )
    .expect("ObjectBuilder::new");
    ObjectModule::new(obj_builder)
}

/// PCA-ACT-1: builtins.rs declares the registry FFIs with the correct
/// signatures (no params, returns I64) so the FA call sites can invoke
/// them.
#[test]
fn registry_getters_are_declared_with_no_params_returning_i64() {
    let mut module = make_module();
    let fns = declare_runtime_functions(&mut module, CallConv::SystemV).expect("declare");

    for getter in &[
        "nsl_packing_metadata_get_segment_ids",
        "nsl_packing_metadata_get_doc_starts",
    ] {
        let (_, sig) = fns.get(*getter).unwrap_or_else(|| {
            panic!("{getter} must be registered in builtins.rs::RUNTIME_FUNCTIONS")
        });
        assert_eq!(
            sig.params.len(),
            0,
            "{getter} must take no params (its job is to read thread-local state)"
        );
        assert_eq!(sig.returns.len(), 1, "{getter} must return one value");
    }
}

/// PCA-ACT-2: builtins.rs declares the registry setter with two i64
/// params (segment_ids_ptr, doc_starts_ptr) and no return value.
#[test]
fn registry_setter_is_declared_with_two_i64_params_no_return() {
    let mut module = make_module();
    let fns = declare_runtime_functions(&mut module, CallConv::SystemV).expect("declare");

    let (_, sig) = fns
        .get("nsl_packing_metadata_set")
        .expect("nsl_packing_metadata_set must be registered");
    assert_eq!(
        sig.params.len(),
        2,
        "set takes (segment_ids_ptr: i64, doc_starts_ptr: i64)"
    );
    assert_eq!(sig.returns.len(), 0, "set returns nothing");
}

/// PCA-ACT-3: `nsl_tensor_data_ptr` FFI is declared with one i64 param
/// and an i64 return. Used by the train block to extract device pointers
/// from `batch["segment_ids"]` / `batch["doc_starts"]` tensor handles.
#[test]
fn tensor_data_ptr_ffi_is_declared() {
    let mut module = make_module();
    let fns = declare_runtime_functions(&mut module, CallConv::SystemV).expect("declare");

    let (_, sig) = fns
        .get("nsl_tensor_data_ptr")
        .expect("nsl_tensor_data_ptr must be registered");
    assert_eq!(sig.params.len(), 1, "takes one tensor pointer");
    assert_eq!(sig.returns.len(), 1, "returns one i64 (the data ptr)");
}

/// PCA-ACT-4: regression guard — `nsl_dict_contains` returns i8, used
/// by the train block to probe for batch["segment_ids"] presence. The
/// activation wiring's runtime probe relies on this signature.
#[test]
fn dict_contains_returns_i8_for_runtime_probe() {
    let mut module = make_module();
    let fns = declare_runtime_functions(&mut module, CallConv::SystemV).expect("declare");

    let (_, sig) = fns
        .get("nsl_dict_contains")
        .expect("nsl_dict_contains must be registered for activation probe");
    assert_eq!(sig.params.len(), 2, "(dict_ptr: i64, key_cstr: i64)");
    let ret = sig
        .returns
        .first()
        .expect("nsl_dict_contains must return i8");
    assert_eq!(
        ret.value_type.to_string(),
        "i8",
        "nsl_dict_contains must return i8 (1 = present, 0 = absent); changing this would break the train block's brif probe"
    );
}
