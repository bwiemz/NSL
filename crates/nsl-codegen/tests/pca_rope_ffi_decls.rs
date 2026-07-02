//! PCA §4.3 — codegen-side FFI decl sentinel test.
//!
//! Verifies the three Cranelift CSHA FFI signature declarations in
//! `builtins.rs::RUNTIME_FUNCTIONS` have the correct trailing parameters:
//! segment_ids_ptr (Tier A), tier_b_ptx_ptr + tier_b_name_ptr (Tier B),
//! doc_starts_ptr (PCA §4.3), and num_docs_or_zero (PCA per-doc CTA
//! Strategy 3 v1, forward FFIs only). Counts are:
//!
//!   * `nsl_flash_attention_csha`            — 38 params (+1 num_docs_or_zero)
//!   * `nsl_flash_attention_csha_with_saves` — 44 params (+1 num_docs_or_zero)
//!   * `nsl_flash_attention_csha_backward`   — 54 params (+1 tier_b2_active
//!     from CSHA Tier B.2 Phase 3 T6 AND +1 num_docs_or_zero added
//!     Sprint 5; backward kernel currently honours num_docs == 0 only,
//!     non-zero is a Sprint 6 follow-on per-doc backward)
//!
//! Each forward count = base + segment_ids (Tier A) + 2 tier_b + doc_starts
//!                       (PCA §4.3) + num_docs_or_zero (per-doc CTA).
//! Backward count    = base + segment_ids (Tier A) + 2 tier_b + doc_starts
//!                     + tier_b2_active (CSHA Tier B.2 Phase 3 T6)
//!                     + num_docs_or_zero (slot wired, value gated to 0).
//!
//! A regression here means a Cranelift call site emitting the call with
//! N args would now hit a sig-mismatch at IR-finalize time. The targeted
//! count assertions surface the regression before that downstream failure.

#[cfg(feature = "csha_cycle19_probe")]
use cranelift_codegen::ir::types;
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
    // 24 base + 9 CSHA extras + 1 segment_ids + 2 tier_b + 1 doc_starts + 1 num_docs_or_zero
    //   = 38 i64 params (PCA per-doc CTA Strategy 3 v1).
    assert_eq!(
        sig.params.len(),
        38,
        "nsl_flash_attention_csha must accept 38 i64 params (PCA §4.3 Task 3 + Tier B + per-doc CTA)"
    );
}

#[test]
fn csha_with_saves_decl_has_doc_starts_trailing_param() {
    let mut module = make_module();
    let fns = declare_runtime_functions(&mut module, CallConv::SystemV).expect("declare");
    let (_, sig) = fns
        .get("nsl_flash_attention_csha_with_saves")
        .expect("nsl_flash_attention_csha_with_saves registered");
    // 24 base + 9 CSHA extras + 6 save pointers + 1 segment_ids + 2 tier_b
    //   + 1 doc_starts + 1 num_docs_or_zero = 44.
    assert_eq!(
        sig.params.len(),
        44,
        "nsl_flash_attention_csha_with_saves must accept 44 i64 params (PCA §4.3 Task 3 + Tier B + per-doc CTA)"
    );
}

/// CSHA cycle 19 T1 (variant-B) — additive arity assertion for the NEW
/// `nsl_flash_attention_csha_backward_probe` symbol. Registered only under
/// the `csha_cycle19_probe` Cargo feature. The count is the existing 54
/// backward params + 2 trailing probe pointers = 56. This test intentionally
/// leaves the 54-arity assertion on the non-probe symbol UNTOUCHED (per
/// cycle-19 T1 spec: "DO NOT edit the existing `54` literal").
#[cfg(feature = "csha_cycle19_probe")]
#[test]
fn csha_backward_probe_decl_has_expected_arity() {
    let mut module = make_module();
    let fns = declare_runtime_functions(&mut module, CallConv::SystemV).expect("declare");
    let (_, sig) = fns
        .get("nsl_flash_attention_csha_backward_probe")
        .expect("nsl_flash_attention_csha_backward_probe registered under csha_cycle19_probe");
    assert_eq!(
        sig.params.len(),
        56,
        "nsl_flash_attention_csha_backward_probe must accept 56 i64 params \
         (54 original + probe_ds_out_ptr + probe_dv_out_ptr)"
    );
    // Defence-in-depth (R3): every param must be i64. A stray non-i64 slot
    // would silently mis-encode pointer args at the Cranelift call site
    // before the arity assertion caught it.
    for (i, p) in sig.params.iter().enumerate() {
        assert_eq!(
            p.value_type,
            types::I64,
            "probe backward FFI param #{i} must be i64, got {:?}",
            p.value_type
        );
    }
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
    //   + 1 segment_ids + 2 tier_b + 1 doc_starts + 1 tier_b2_active
    //   + 1 num_docs_or_zero = 54. (tier_b2_active added in CSHA Tier B.2
    // Phase 3 T6; num_docs_or_zero added in PCA per-doc CTA Sprint 5.)
    assert_eq!(
        sig.params.len(),
        54,
        "nsl_flash_attention_csha_backward must accept 54 i64 params \
         (PCA §4.3 Task 3 + Tier B + CSHA Tier B.2 tier_b2_active + per-doc CTA num_docs_or_zero)"
    );
}
