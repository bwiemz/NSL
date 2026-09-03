//! Allocation primitives.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_MEMORY: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Memory
    ("nsl_alloc", &[types::I64], Some(types::I64)),
    ("nsl_free", &[types::I64], None),
    ("nsl_closure_free", &[types::I64], None),
];
