//! Compile-time check that every runtime function the codegen calls has the
//! signature the ABI table says it has (roadmap A3).
//!
//! `nsl_abi::for_each_runtime_fn!` hands this module every row of the table
//! in `crates/nsl-abi/src/table.rs`; each row becomes one `const _: ()` that
//! casts the named implementation to `unsafe extern "C" fn(_, …) -> _` — so
//! `rustc` infers the implementation's real parameter types — and compares
//! the register class and width of every slot with the row through
//! `nsl_abi::typed::assert_sig`. A row whose arity, slot class or path
//! disagrees with the implementation fails this crate's build with the
//! function's name (`evaluation panicked: nsl_…`). Nothing here exists at
//! run time.
//!
//! An `[interop]` row names an implementation behind the `interop` feature;
//! without the feature the symbol is the stub of the same name in
//! `interop_stubs`, which is checked instead, so both builds verify the
//! surface they link.
//!
//! The C-API table (`nsl_abi::for_each_capi_fn!`, the surface a host calls
//! through the generated header or the Python package) is checked the same
//! way; its rows carry no feature flag.

use nsl_abi::typed::assert_sig;

/// `_` in type position, one per table parameter, so the cast infers the
/// implementation's own types.
macro_rules! infer {
    ($t:tt) => {
        _
    };
}

macro_rules! assert_abi_row {
    ([interop] $n:ident ($($p:ident),*) -> $r:tt = $($seg:ident)::+) => {
        #[cfg(feature = "interop")]
        const _: () = assert_sig(
            crate::$($seg)::+ as unsafe extern "C" fn($(infer!($p)),*) -> _,
            stringify!($n),
            &[$(nsl_abi::abi_scalar!($p)),*],
            nsl_abi::abi_ret!($r),
        );
        #[cfg(not(feature = "interop"))]
        const _: () = assert_sig(
            crate::interop_stubs::$n as unsafe extern "C" fn($(infer!($p)),*) -> _,
            stringify!($n),
            &[$(nsl_abi::abi_scalar!($p)),*],
            nsl_abi::abi_ret!($r),
        );
    };
    ([] $n:ident ($($p:ident),*) -> $r:tt = $($seg:ident)::+) => {
        const _: () = assert_sig(
            crate::$($seg)::+ as unsafe extern "C" fn($(infer!($p)),*) -> _,
            stringify!($n),
            &[$(nsl_abi::abi_scalar!($p)),*],
            nsl_abi::abi_ret!($r),
        );
    };
}

macro_rules! assert_runtime_abi {
    ($([$g:ident] $n:ident ($($p:ident),*) -> $r:tt = $($seg:ident)::+ $([$f:ident])? ;)*) => {
        $( assert_abi_row! { [$($f)?] $n ($($p),*) -> $r = $($seg)::+ } )*
    };
}

nsl_abi::for_each_runtime_fn!(assert_runtime_abi);

macro_rules! assert_capi_abi {
    ($($n:ident ($($p:ident),*) -> $r:tt = $($seg:ident)::+ $(: $c:literal)? ;)*) => {
        $( assert_abi_row! { [] $n ($($p),*) -> $r = $($seg)::+ } )*
    };
}

nsl_abi::for_each_capi_fn!(assert_capi_abi);
