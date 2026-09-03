//! Scalar arithmetic, conversions and assertions.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_SCALAR: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Power
    ("nsl_pow_int", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_pow_float", &[types::F64, types::F64], Some(types::F64)),
    // Type conversions
    ("nsl_str_to_int", &[types::I64], Some(types::I64)),
    ("nsl_str_to_float", &[types::I64], Some(types::F64)),
    ("nsl_str_len", &[types::I64], Some(types::I64)),
    // Math
    ("nsl_sqrt", &[types::F64], Some(types::F64)),
    ("nsl_log", &[types::F64], Some(types::F64)),
    ("nsl_exp", &[types::F64], Some(types::F64)),
    ("nsl_sin", &[types::F64], Some(types::F64)),
    ("nsl_cos", &[types::F64], Some(types::F64)),
    ("nsl_abs_float", &[types::F64], Some(types::F64)),
    ("nsl_abs_int", &[types::I64], Some(types::I64)),
    ("nsl_min_int", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_max_int", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_min_float", &[types::F64, types::F64], Some(types::F64)),
    ("nsl_max_float", &[types::F64, types::F64], Some(types::F64)),
    // Assert & Exit
    ("nsl_assert", &[types::I8, types::I64], None),
    ("nsl_exit", &[types::I64], None),
    // Scalar math (M14)
    ("nsl_floor", &[types::F64], Some(types::F64)),
    // Assert functions (M15 test framework)
    (
        "nsl_assert_eq_int",
        &[types::I64, types::I64, types::I64, types::I64],
        None,
    ),
    (
        "nsl_assert_eq_float",
        &[types::F64, types::F64, types::I64, types::I64],
        None,
    ),
    (
        "nsl_assert_close",
        &[
            types::I64,
            types::I64,
            types::F64,
            types::F64,
            types::I64,
            types::I64,
        ],
        None,
    ),
];
