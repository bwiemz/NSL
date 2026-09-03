//! Lists, dicts, strings, ranges and higher-order functions.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_COLLECTIONS: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // List
    ("nsl_list_new", &[], Some(types::I64)),
    ("nsl_list_push", &[types::I64, types::I64], None),
    ("nsl_list_get", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_list_len", &[types::I64], Some(types::I64)),
    ("nsl_list_set", &[types::I64, types::I64, types::I64], None),
    (
        "nsl_list_contains",
        &[types::I64, types::I64],
        Some(types::I8),
    ),
    ("nsl_list_free", &[types::I64], None),
    // String
    (
        "nsl_str_concat",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_int_to_str", &[types::I64], Some(types::I64)),
    ("nsl_float_to_str", &[types::F64], Some(types::I64)),
    ("nsl_bool_to_str", &[types::I8], Some(types::I64)),
    // Range
    (
        "nsl_range",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // Dict
    ("nsl_dict_new", &[], Some(types::I64)),
    (
        "nsl_dict_set_str",
        &[types::I64, types::I64, types::I64],
        None,
    ),
    (
        "nsl_dict_get_str",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_dict_len", &[types::I64], Some(types::I64)),
    (
        "nsl_dict_contains",
        &[types::I64, types::I64],
        Some(types::I8),
    ),
    ("nsl_dict_keys", &[types::I64], Some(types::I64)),
    ("nsl_dict_free", &[types::I64], None),
    ("nsl_dict_free_tensor_values", &[types::I64], None),
    // String comparison
    ("nsl_str_eq", &[types::I64, types::I64], Some(types::I64)),
    // String repeat & slice
    (
        "nsl_str_repeat",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_list_slice",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_str_slice",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // String methods
    ("nsl_str_upper", &[types::I64], Some(types::I64)),
    ("nsl_str_lower", &[types::I64], Some(types::I64)),
    ("nsl_str_strip", &[types::I64], Some(types::I64)),
    ("nsl_str_split", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_str_join", &[types::I64, types::I64], Some(types::I64)),
    (
        "nsl_str_replace",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_str_find", &[types::I64, types::I64], Some(types::I64)),
    (
        "nsl_str_startswith",
        &[types::I64, types::I64],
        Some(types::I8),
    ),
    (
        "nsl_str_endswith",
        &[types::I64, types::I64],
        Some(types::I8),
    ),
    (
        "nsl_str_contains",
        &[types::I64, types::I64],
        Some(types::I8),
    ),
    // Higher-order functions. Third arg = ret_is_bool: the function
    // pointer is invoked as fn(i64)->i64, but a bool-returning NSL fn
    // compiles to an I8 return whose upper register bits are undefined —
    // the runtime masks to the low byte when this flag is set (hof.rs).
    ("nsl_map", &[types::I64, types::I64, types::I64], Some(types::I64)),
    (
        "nsl_filter",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_enumerate", &[types::I64], Some(types::I64)),
    ("nsl_zip", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_sorted", &[types::I64], Some(types::I64)),
    ("nsl_reversed", &[types::I64], Some(types::I64)),
    // String deallocation (M15)
    ("nsl_string_free", &[types::I64], None),
];
