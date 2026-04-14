//! Dev Tools Phase 2, Task 5: smoke test for `nsl_profile_kernel_begin/end`
//! emission scaffolding.
//!
//! Full end-to-end verification (compile a `kernel { ... }` program with
//! `profile_kernels = true`, link, run, and assert manifest IDs match
//! runtime collector IDs) is deferred to Task 9.  Here we only assert the
//! pieces Task 5 owns:
//!
//! 1. The runtime functions `nsl_profile_kernel_begin` and
//!    `nsl_profile_kernel_end` are declared so `compile_call_by_name`
//!    can resolve them when the kernel-launch wrapper fires.
//! 2. The pre-pass test helper still works after Task 5's plumbing
//!    additions (no regression on Compiler::new field defaults).
#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;

#[test]
fn profile_kernel_runtime_fns_are_declared() {
    use nsl_lexer::Interner;
    use nsl_semantic::checker::TypeMap;

    let interner = Interner::new();
    let type_map: TypeMap = std::collections::HashMap::new();
    let opts = CompileOptions::default();
    let compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed with default options");

    // `declare_runtime_functions` registers everything in RUNTIME_FUNCTIONS.
    // After Task 5, the begin/end pair must be present under runtime_fns.
    let mut compiler = compiler;
    compiler
        .declare_runtime_functions()
        .expect("declare_runtime_functions should succeed");

    assert!(
        compiler
            .registry
            .runtime_fns
            .contains_key("nsl_profile_kernel_begin"),
        "nsl_profile_kernel_begin must be declared as a runtime function"
    );
    assert!(
        compiler
            .registry
            .runtime_fns
            .contains_key("nsl_profile_kernel_end"),
        "nsl_profile_kernel_end must be declared as a runtime function"
    );
}

#[test]
fn compiler_has_source_text_and_file_name_fields() {
    use nsl_lexer::Interner;
    use nsl_semantic::checker::TypeMap;

    let interner = Interner::new();
    let type_map: TypeMap = std::collections::HashMap::new();
    let opts = CompileOptions::default();
    let compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed");

    // Defaults are empty strings — populated lazily by the codegen entry
    // when callers plumb the original source through.
    assert_eq!(compiler.source_text, "");
    assert_eq!(compiler.source_file_name, "");
}
