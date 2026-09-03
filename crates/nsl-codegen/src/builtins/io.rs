//! Printing, stdin, files and process arguments.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_IO: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Print
    ("nsl_print_int", &[types::I64], None),
    ("nsl_print_float", &[types::F64], None),
    ("nsl_print_str", &[types::I64], None),
    ("nsl_print_bool", &[types::I8], None),
    // Stdin I/O
    ("nsl_read_line", &[], Some(types::I64)),
    // File I/O
    ("nsl_read_file", &[types::I64], Some(types::I64)),
    ("nsl_write_file", &[types::I64, types::I64], None),
    ("nsl_append_file", &[types::I64, types::I64], None),
    ("nsl_file_exists", &[types::I64], Some(types::I8)),
    // Command-line args
    ("nsl_args_init", &[types::I32, types::I64], None),
    ("nsl_args", &[], Some(types::I64)),
];
