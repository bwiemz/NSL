//! The C-API table: every runtime symbol a *host* calls — a C program
//! through the generated header, or the Python package through `ctypes` —
//! declared ONCE (roadmap A3, step 2 of the design in
//! `docs/superpowers/specs/2026-09-08-a3-abi-extern-table-design.md`).
//!
//! [`for_each_runtime_fn!`](crate::for_each_runtime_fn) is the surface the
//! *compiler* emits calls to. This is the other surface: the model
//! lifecycle, the named-dispatch and ownership entry points, DLPack, the
//! per-call grad context and the error slot. The two overlap only where a
//! host and a program both call a function; where they do, the tests below
//! pin the two rows equal.
//!
//! Three consumers:
//!
//! * `nsl-runtime` (`abi_check.rs`) renders each row as a compile-time
//!   assertion that the named implementation has that signature, exactly as
//!   it does for the runtime-function table;
//! * `nsl-codegen` (`c_header.rs`) prints each row's C prototype into every
//!   generated header, so the header cannot declare a signature the runtime
//!   does not implement;
//! * `nsl abi python` renders [`render_python`] — `python/nslpy/_abi.py`,
//!   one `argtypes`/`restype` pair per row — which `nslpy` binds from and
//!   `cargo test -p nsl-abi` pins against this table.
//!
//! # Row grammar
//!
//! ```text
//! name(param, …) -> ret = module::path::name : "C prototype";
//! ```
//!
//! `param`/`ret` and the path are as in the runtime-function table. The C
//! prototype is the exact text the header prints for the row, trailing
//! comment included; it is optional — a row without one is a host entry
//! point the header does not (yet) declare, bound by the Python mirror
//! only. Every prototype is parsed back by
//! [`parse_c_prototypes`](crate::parse_c_prototypes) and must lower, type
//! by type, to the row's scalars: a pointer or `int64_t` where the row says
//! `i64`, `void` where it says `()`.

use std::fmt::Write as _;

use crate::AbiScalar;

/// One row of the C-API table, as data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CApiDecl {
    /// The link symbol.
    pub name: &'static str,
    /// Parameter register classes, in order.
    pub params: &'static [AbiScalar],
    /// Return register class; `None` for no return value.
    pub ret: Option<AbiScalar>,
    /// The implementation's path inside `nsl-runtime`, as segments after
    /// `crate`.
    pub path: &'static [&'static str],
    /// The prototype the generated C header prints, if the header declares
    /// this symbol.
    pub c_prototype: Option<&'static str>,
}

#[doc(hidden)]
#[macro_export]
macro_rules! __capi_proto {
    () => { ::core::option::Option::None };
    ($c:literal) => { ::core::option::Option::Some($c) };
}

macro_rules! __decl_capi_abi {
    ($($n:ident ($($p:ident),*) -> $r:tt = $($seg:ident)::+ $(: $c:literal)? ;)*) => {
        /// The C-API table as data: one [`CApiDecl`] per row, in row order.
        pub static CAPI_ABI: &[CApiDecl] = &[
            $(CApiDecl {
                name: stringify!($n),
                params: &[$($crate::abi_scalar!($p)),*],
                ret: $crate::abi_ret!($r),
                path: &[$(stringify!($seg)),+],
                c_prototype: $crate::__capi_proto!($($c)?),
            },)*
        ];
    };
}

/// The C-API table. Invoke with the name of a `macro_rules!` that accepts
/// the row grammar in the module docs; it receives every row in one
/// invocation.
#[macro_export]
macro_rules! for_each_capi_fn {
    ($m:ident) => {
        $m! {
            // ── Version and errors ──
            nsl_abi_version() -> i64 = c_api::nsl_abi_version
                : "int64_t   nsl_abi_version(void); /* (major<<16)|minor; cf. NSL_ABI_VERSION */";
            nsl_get_last_error() -> i64 = c_api::nsl_get_last_error
                : "const char* nsl_get_last_error(void);";
            nsl_clear_error() -> i64 = c_api::nsl_clear_error
                : "int64_t     nsl_clear_error(void); /* returns 0 */";

            // ── Model lifecycle ──
            nsl_model_create(i64) -> i64 = c_api::nsl_model_create
                : "NslModel* nsl_model_create(const char* weights_path);";
            nsl_model_create_with_lib(i64, i64) -> i64 = c_api::nsl_model_create_with_lib;
            nsl_model_destroy(i64) -> i64 = c_api::nsl_model_destroy
                : "int64_t   nsl_model_destroy(NslModel* model); /* returns 0 */";
            nsl_model_get_version() -> i64 = c_api::nsl_model_get_version;
            nsl_model_num_weights(i64) -> i64 = c_api::nsl_model_num_weights;
            nsl_model_export_count(i64) -> i64 = c_api::nsl_model_export_count;
            nsl_model_lookup_function(i64, i64) -> i64 = c_api::nsl_model_lookup_function;

            // ── Named dispatch (M62b) and the item-7 ownership models ──
            nsl_model_call(i64, i64, i64, i64, i64, i64) -> i64 = c_api::nsl_model_call
                : "int64_t   nsl_model_call(NslModel* model, const char* name,\n                          const NslTensorDesc* inputs, int64_t n_inputs,\n                          NslTensorDesc* outputs, int64_t n_outputs);";
            nsl_model_call_into(i64, i64, i64, i64, i64, i64, i64) -> i64 = c_api::nsl_model_call_into
                : "int64_t   nsl_model_call_into(NslModel* model, const char* name,\n                          const NslTensorDesc* inputs, int64_t n_inputs,\n                          NslTensorDesc* outputs, int64_t n_outputs,\n                          const uint64_t* out_capacities);";
            nsl_model_call_alloc(i64, i64, i64, i64, i64, i64) -> i64 = c_api::nsl_model_call_alloc
                : "int64_t   nsl_model_call_alloc(NslModel* model, const char* name,\n                          const NslTensorDesc* inputs, int64_t n_inputs,\n                          DLManagedTensor** out_dl, int64_t n_outputs);";
            nsl_model_call_dlpack(i64, i64, i64, i64, i64, i64) -> i64 = c_api::nsl_model_call_dlpack;
            nsl_model_get_export_signature(i64, i64) -> i64 = c_api::nsl_model_get_export_signature
                : "const char* nsl_model_get_export_signature(NslModel* model, const char* name);";

            // ── Forward and DLPack ──
            nsl_model_forward_dlpack(i64, i64, i64, i64, i64) -> i64 = c_api::nsl_model_forward_dlpack;
            nsl_dlpack_export(i64) -> i64 = dlpack::nsl_dlpack_export;
            nsl_dlpack_import(i64) -> i64 = dlpack::nsl_dlpack_import;
            nsl_dlpack_free(i64) -> () = dlpack::nsl_dlpack_free;

            // ── Per-call grad context (Spec B) ──
            nsl_model_forward_grad(i64, i64, i64, i64, i64, i64) -> i64 = grad_context::nsl_model_forward_grad;
            nsl_model_backward(i64, i64, i64, i64, i64) -> i64 = grad_context::nsl_model_backward;
            nsl_grad_context_destroy(i64) -> () = grad_context::nsl_grad_context_destroy;
        }
    };
}

for_each_capi_fn!(__decl_capi_abi);

/// The row for `name`, if the C API declares it.
pub fn lookup(name: &str) -> Option<&'static CApiDecl> {
    CAPI_ABI.iter().find(|d| d.name == name)
}

/// The prototype the generated C header prints for `name`; `None` when the
/// symbol is not in the table or the table carries no prototype for it.
pub fn c_prototype(name: &str) -> Option<&'static str> {
    lookup(name).and_then(|d| d.c_prototype)
}

/// The `ctypes` spelling of a register class. Every C-API slot is an
/// integer-sized value or a pointer passed as one, so the mirror never needs
/// a pointer type: a host hands the runtime addresses as `int`.
fn ctypes_name(s: AbiScalar) -> &'static str {
    match s {
        AbiScalar::Int(64) => "ctypes.c_int64",
        AbiScalar::Int(32) => "ctypes.c_int32",
        AbiScalar::Int(16) => "ctypes.c_int16",
        AbiScalar::Int(8) => "ctypes.c_int8",
        AbiScalar::Float(64) => "ctypes.c_double",
        AbiScalar::Float(32) => "ctypes.c_float",
        AbiScalar::Int(w) | AbiScalar::Float(w) => panic!("no ctypes spelling for a {w}-bit slot"),
    }
}

/// Render `python/nslpy/_abi.py`: the table as `ctypes` signatures plus the
/// `bind` helper `nslpy` calls on every library handle it opens.
pub fn render_python() -> String {
    let mut out = String::new();
    out.push_str(
        "\"\"\"ctypes signatures of the NSL runtime C API.\n\
         \n\
         GENERATED by `nsl abi python` from crates/nsl-abi/src/capi.rs. Do not\n\
         edit by hand: `cargo test -p nsl-abi` fails if this file and the table\n\
         differ. Every slot is an integer-sized value or a pointer passed as one\n\
         (the runtime's C API takes addresses as int64_t), so the mirror never\n\
         needs a ctypes pointer type.\n\
         \"\"\"\n\
         \n\
         import ctypes\n\
         \n\
         SIGNATURES = {\n",
    );
    for d in CAPI_ABI {
        let params: Vec<&str> = d.params.iter().map(|&p| ctypes_name(p)).collect();
        let ret = d.ret.map_or("None", ctypes_name);
        let _ = writeln!(out, "    \"{}\": ([{}], {ret}),", d.name, params.join(", "));
    }
    out.push_str(
        "}\n\
         \n\
         \n\
         def bind(lib, names=None):\n    \
             \"\"\"Set argtypes/restype on `lib` for the table symbols it exports.\n\
         \n    \
             `names` restricts the binding to a subset; a symbol absent from `lib`\n    \
             is skipped, so one call serves both the standalone runtime and an\n    \
             @export shared library that statically links it (which may predate\n    \
             some of the entry points).\n    \
             \"\"\"\n    \
             for name in SIGNATURES if names is None else names:\n        \
                 if hasattr(lib, name):\n            \
                     fn = getattr(lib, name)\n            \
                     fn.argtypes, fn.restype = SIGNATURES[name]\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{abi_from_c, parse_c_prototypes, ParsedType, RUNTIME_ABI};

    #[test]
    fn names_are_unique_and_paths_end_in_the_symbol() {
        let mut seen = std::collections::HashSet::new();
        for d in CAPI_ABI {
            assert!(seen.insert(d.name), "duplicate C-API row {}", d.name);
            assert_eq!(d.path.last().copied(), Some(d.name), "{}: path must end in the symbol", d.name);
        }
        assert_eq!(CAPI_ABI.len(), 22);
    }

    /// Each C prototype, parsed back as the header gate parses the emitted
    /// header, lowers to exactly the row's scalars — so the header and the
    /// runtime's typed assertion describe one signature.
    #[test]
    fn every_c_prototype_lowers_to_its_row() {
        let mut checked = 0;
        for d in CAPI_ABI {
            let Some(proto) = d.c_prototype else { continue };
            let parsed = parse_c_prototypes(proto);
            assert_eq!(parsed.len(), 1, "{}: prototype must parse as one declaration:\n{proto}", d.name);
            let sig = &parsed[0];
            assert_eq!(sig.name, d.name, "prototype names {} but the row is {}", sig.name, d.name);
            let lower = |t: &ParsedType| match t {
                ParsedType::Known(s) => *s,
                ParsedType::Unknown(s) => panic!("{}: C type `{s}` is not modelled by abi_from_c", d.name),
            };
            let params: Vec<AbiScalar> = sig.params.iter().map(lower).collect();
            assert_eq!(params, d.params, "{}: parameter slots", d.name);
            assert_eq!(sig.ret.as_ref().map(lower), d.ret, "{}: return slot", d.name);
            checked += 1;
        }
        assert_eq!(checked, 9, "the header declares nine C-API prototypes");
        // The mapping the gate relies on, spelled out.
        assert_eq!(abi_from_c("NslModel*"), Some(AbiScalar::Int(64)));
        assert_eq!(abi_from_c("const uint64_t*"), Some(AbiScalar::Int(64)));
    }

    /// A symbol in both tables has one signature.
    #[test]
    fn rows_shared_with_the_runtime_function_table_agree() {
        for d in CAPI_ABI {
            if let Some(r) = RUNTIME_ABI.iter().find(|r| r.name == d.name) {
                assert_eq!((r.params, r.ret, r.path), (d.params, d.ret, d.path), "{}", d.name);
            }
        }
    }

    #[test]
    fn the_python_mirror_binds_every_row_with_integer_slots() {
        let py = render_python();
        for d in CAPI_ABI {
            assert!(py.contains(&format!("\"{}\": (", d.name)), "{} missing from the mirror", d.name);
        }
        assert!(py.contains("\"nsl_dlpack_free\": ([ctypes.c_int64], None)"));
        assert!(py.contains(
            "\"nsl_model_call_into\": ([ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, \
             ctypes.c_int64, ctypes.c_int64, ctypes.c_int64], ctypes.c_int64)"
        ));
        assert!(py.starts_with("\"\"\"ctypes signatures"));
        assert!(py.ends_with("fn.argtypes, fn.restype = SIGNATURES[name]\n"));
    }
}
