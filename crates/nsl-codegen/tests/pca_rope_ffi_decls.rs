//! PCA §4.3 — codegen-side FFI decl sentinel test.
//!
//! Verifies the three Cranelift CSHA FFI signature declarations in
//! `builtins.rs::RUNTIME_FUNCTIONS` have the trailing `doc_starts_ptr: I64`
//! parameter. Counts are stable in spec v3:
//!
//!   * `nsl_flash_attention_csha`            — 35 params
//!   * `nsl_flash_attention_csha_with_saves` — 41 params
//!   * `nsl_flash_attention_csha_backward`   — 50 params
//!
//! Each count = base + segment_ids (Tier A) + doc_starts (PCA §4.3).
//!
//! A regression here means a Cranelift call site emitting the call with
//! N args would now hit a sig-mismatch at IR-finalize time. The targeted
//! count assertions surface the regression before that downstream failure.

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
        "pca_rope_ffi_decls_test",
        default_libcall_names(),
    )
    .expect("ObjectBuilder::new");
    ObjectModule::new(obj_builder)
}

#[test]
fn csha_forward_decl_has_doc_starts_trailing_param() {
    let mut module = make_module();
    let fns = declare_runtime_functions(&mut module, CallConv::SystemV).expect("declare");
    let (_, sig) = fns
        .get("nsl_flash_attention_csha")
        .expect("nsl_flash_attention_csha registered");
    // 24 base + 9 CSHA extras + 1 segment_ids + 1 doc_starts = 35 i64 params.
    assert_eq!(
        sig.params.len(),
        35,
        "nsl_flash_attention_csha must accept 35 i64 params (PCA §4.3 Task 3)"
    );
}

#[test]
fn csha_with_saves_decl_has_doc_starts_trailing_param() {
    let mut module = make_module();
    let fns = declare_runtime_functions(&mut module, CallConv::SystemV).expect("declare");
    let (_, sig) = fns
        .get("nsl_flash_attention_csha_with_saves")
        .expect("nsl_flash_attention_csha_with_saves registered");
    // 24 base + 9 CSHA extras + 6 save pointers + 1 segment_ids + 1 doc_starts = 41.
    assert_eq!(
        sig.params.len(),
        41,
        "nsl_flash_attention_csha_with_saves must accept 41 i64 params (PCA §4.3 Task 3)"
    );
}

#[test]
fn csha_backward_decl_has_doc_starts_trailing_param() {
    let mut module = make_module();
    let fns = declare_runtime_functions(&mut module, CallConv::SystemV).expect("declare");
    let (_, sig) = fns
        .get("nsl_flash_attention_csha_backward")
        .expect("nsl_flash_attention_csha_backward registered");
    // Authoritative count comes from RUNTIME_FUNCTIONS in builtins.rs:
    // 33 base (forward-side, includes the explicit `wo` slot) + 6 saves
    //   + 9 grad outputs (dO + dq/dk/dv + dwq/dwk/dwv + dx + dx_norm)
    //   + 1 segment_ids + 1 doc_starts = 50.
    assert_eq!(
        sig.params.len(),
        50,
        "nsl_flash_attention_csha_backward must accept 50 i64 params (PCA §4.3 Task 3)"
    );
}
