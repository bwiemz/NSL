//! The runtime-function table: every `extern "C"` symbol the codegen can
//! emit a call to, declared ONCE, in a form each consumer expands without
//! parsing text (roadmap A3, step 1 of the design in
//! `docs/superpowers/specs/2026-09-08-a3-abi-extern-table-design.md`).
//!
//! The table is the X-macro [`for_each_runtime_fn!`]: a consumer passes the
//! name of its own `macro_rules!` and receives every row as tokens. Three
//! consumers exist today:
//!
//! * `nsl-abi` itself builds [`RUNTIME_ABI`], the table as data (`FnDecl`),
//!   for the cross-check, the generators and the tests;
//! * `nsl-codegen` (`builtins/mod.rs`) renders each row as a Cranelift
//!   signature — the declarations it used to keep in fourteen hand-written
//!   `RUNTIME_FUNCTIONS*` tables;
//! * `nsl-runtime` (`abi_check.rs`) renders each row as a compile-time
//!   assertion that the named implementation has that signature, checked by
//!   `rustc` through [`crate::typed`].
//!
//! # Row grammar
//!
//! ```text
//! [group] name(param, …) -> ret = module::path::name [interop];
//! ```
//!
//! * `group` — the subsystem the row belongs to; the groups are the former
//!   table files (`memory`, `io`, `scalar`, `collections`, `tensor` for what
//!   a program calls directly; `abi_tensor`, `training`, `optimizer`,
//!   `distributed`, `inference`, `quantization`, `diagnostics`,
//!   `abi_memory`, `interop` for the implementation surface behind it).
//! * `param`/`ret` — the ABI scalar, spelled `i64`, `i32`, `i8`, `f64`,
//!   `f32`; `()` for no return value. This is the register class and width
//!   ([`AbiScalar`]), which is what the calling convention sees: a runtime
//!   implementation may spell an `i64` slot as `u64`, `usize` or a raw
//!   pointer, and the typed check accepts those (see [`crate::typed`]).
//! * `module::path::name` — the implementation's path inside `nsl-runtime`,
//!   relative to the crate root, so the runtime's rendering can name it.
//! * `[interop]` — the implementation lives in a module behind the runtime's
//!   `interop` feature; without the feature the symbol is the stub of the
//!   same name in `interop_stubs`.
//!
//! Row order is the order the codegen declares the functions in, which the
//! emitted CLIF never observes (see `builtins/mod.rs`); the grouping and the
//! comments are carried over from the tables the rows were generated from.
//!
//! # Adding a runtime function
//!
//! Implement the `#[unsafe(no_mangle)] extern "C" fn` in `nsl-runtime`, then
//! add one row here, in the group its subject belongs to. Nothing else is
//! edited: the codegen declares it, the runtime's build checks it, and
//! `signature_agreement` cross-checks the runtime's `extern "C"` items
//! against this table by text as belt-and-braces.

use crate::AbiScalar;

/// One row of the table, as data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FnDecl {
    /// The subsystem group (see the module docs).
    pub group: &'static str,
    /// The link symbol.
    pub name: &'static str,
    /// Parameter register classes, in order.
    pub params: &'static [AbiScalar],
    /// Return register class; `None` for no return value.
    pub ret: Option<AbiScalar>,
    /// The implementation's path inside `nsl-runtime`, as segments after
    /// `crate`.
    pub path: &'static [&'static str],
    /// Whether the implementation is behind the runtime's `interop` feature.
    pub interop: bool,
}

/// Map a row's scalar spelling to an [`AbiScalar`] value.
#[macro_export]
macro_rules! abi_scalar {
    (i64) => { $crate::AbiScalar::Int(64) };
    (i32) => { $crate::AbiScalar::Int(32) };
    (i16) => { $crate::AbiScalar::Int(16) };
    (i8) => { $crate::AbiScalar::Int(8) };
    (f64) => { $crate::AbiScalar::Float(64) };
    (f32) => { $crate::AbiScalar::Float(32) };
}

/// Map a row's return spelling to an `Option<AbiScalar>` value.
#[macro_export]
macro_rules! abi_ret {
    (()) => { ::core::option::Option::None };
    ($t:ident) => { ::core::option::Option::Some($crate::abi_scalar!($t)) };
}

/// `true` when a row carries the `[interop]` flag.
#[doc(hidden)]
#[macro_export]
macro_rules! __abi_is_interop {
    () => { false };
    (interop) => { true };
}

macro_rules! __decl_runtime_abi {
    ($([$g:ident] $n:ident ($($p:ident),*) -> $r:tt = $($seg:ident)::+ $([$f:ident])? ;)*) => {
        /// The table as data: one [`FnDecl`] per row, in row order.
        pub static RUNTIME_ABI: &[FnDecl] = &[
            $(FnDecl {
                group: stringify!($g),
                name: stringify!($n),
                params: &[$($crate::abi_scalar!($p)),*],
                ret: $crate::abi_ret!($r),
                path: &[$(stringify!($seg)),+],
                interop: $crate::__abi_is_interop!($($f)?),
            },)*
        ];
    };
}

/// The table. Invoke with the name of a `macro_rules!` that accepts the row
/// grammar in the module docs; it receives every row in one invocation.
#[macro_export]
macro_rules! for_each_runtime_fn {
    ($m:ident) => {
        $m! {

            // ── memory (was builtins/memory.rs) ──
            // Memory
            [memory] nsl_alloc(i64) -> i64 = memory::nsl_alloc;
            [memory] nsl_free(i64) -> () = memory::nsl_free;
            [memory] nsl_closure_free(i64) -> () = memory::nsl_closure_free;

            // ── io (was builtins/io.rs) ──
            // Print
            [io] nsl_print_int(i64) -> () = print::nsl_print_int;
            [io] nsl_print_float(f64) -> () = print::nsl_print_float;
            [io] nsl_print_str(i64) -> () = print::nsl_print_str;
            [io] nsl_print_bool(i8) -> () = print::nsl_print_bool;
            // Stdin I/O
            [io] nsl_read_line() -> i64 = io::nsl_read_line;
            // File I/O
            [io] nsl_read_file(i64) -> i64 = file_io::nsl_read_file;
            [io] nsl_write_file(i64, i64) -> () = file_io::nsl_write_file;
            [io] nsl_append_file(i64, i64) -> () = file_io::nsl_append_file;
            [io] nsl_file_exists(i64) -> i8 = file_io::nsl_file_exists;
            // Command-line args
            [io] nsl_args_init(i32, i64) -> () = args::nsl_args_init;
            [io] nsl_args() -> i64 = args::nsl_args;

            // ── scalar (was builtins/scalar.rs) ──
            // Power
            [scalar] nsl_pow_int(i64, i64) -> i64 = power::nsl_pow_int;
            [scalar] nsl_pow_float(f64, f64) -> f64 = power::nsl_pow_float;
            // Type conversions
            [scalar] nsl_str_to_int(i64) -> i64 = string::nsl_str_to_int;
            [scalar] nsl_str_to_float(i64) -> f64 = string::nsl_str_to_float;
            [scalar] nsl_str_len(i64) -> i64 = string::nsl_str_len;
            // Math
            [scalar] nsl_sqrt(f64) -> f64 = math::nsl_sqrt;
            [scalar] nsl_log(f64) -> f64 = math::nsl_log;
            [scalar] nsl_exp(f64) -> f64 = math::nsl_exp;
            [scalar] nsl_sin(f64) -> f64 = math::nsl_sin;
            [scalar] nsl_cos(f64) -> f64 = math::nsl_cos;
            [scalar] nsl_abs_float(f64) -> f64 = math::nsl_abs_float;
            [scalar] nsl_abs_int(i64) -> i64 = math::nsl_abs_int;
            [scalar] nsl_min_int(i64, i64) -> i64 = math::nsl_min_int;
            [scalar] nsl_max_int(i64, i64) -> i64 = math::nsl_max_int;
            [scalar] nsl_min_float(f64, f64) -> f64 = math::nsl_min_float;
            [scalar] nsl_max_float(f64, f64) -> f64 = math::nsl_max_float;
            // Assert & Exit
            [scalar] nsl_assert(i8, i64) -> () = assert::nsl_assert;
            [scalar] nsl_exit(i64) -> () = assert::nsl_exit;
            // Scalar math (M14)
            [scalar] nsl_floor(f64) -> f64 = math::nsl_floor;
            // Assert functions (M15 test framework)
            [scalar] nsl_assert_eq_int(i64, i64, i64, i64) -> () = assert::nsl_assert_eq_int;
            [scalar] nsl_assert_eq_float(f64, f64, i64, i64) -> () = assert::nsl_assert_eq_float;
            [scalar] nsl_assert_close(i64, i64, f64, f64, i64, i64) -> () = assert::nsl_assert_close;

            // ── collections (was builtins/collections.rs) ──
            // List
            [collections] nsl_list_new() -> i64 = list::nsl_list_new;
            [collections] nsl_list_push(i64, i64) -> () = list::nsl_list_push;
            [collections] nsl_list_get(i64, i64) -> i64 = list::nsl_list_get;
            [collections] nsl_list_len(i64) -> i64 = list::nsl_list_len;
            [collections] nsl_list_set(i64, i64, i64) -> () = list::nsl_list_set;
            [collections] nsl_list_contains(i64, i64) -> i8 = list::nsl_list_contains;
            [collections] nsl_list_free(i64) -> () = list::nsl_list_free;
            // String
            [collections] nsl_str_concat(i64, i64) -> i64 = string::nsl_str_concat;
            [collections] nsl_int_to_str(i64) -> i64 = string::nsl_int_to_str;
            [collections] nsl_float_to_str(f64) -> i64 = string::nsl_float_to_str;
            [collections] nsl_bool_to_str(i8) -> i64 = string::nsl_bool_to_str;
            // Range
            [collections] nsl_range(i64, i64, i64) -> i64 = range::nsl_range;
            // Dict
            [collections] nsl_dict_new() -> i64 = dict::nsl_dict_new;
            [collections] nsl_dict_set_str(i64, i64, i64) -> () = dict::nsl_dict_set_str;
            [collections] nsl_dict_get_str(i64, i64) -> i64 = dict::nsl_dict_get_str;
            [collections] nsl_dict_len(i64) -> i64 = dict::nsl_dict_len;
            [collections] nsl_dict_contains(i64, i64) -> i8 = dict::nsl_dict_contains;
            [collections] nsl_dict_keys(i64) -> i64 = dict::nsl_dict_keys;
            [collections] nsl_dict_free(i64) -> () = dict::nsl_dict_free;
            [collections] nsl_dict_free_tensor_values(i64) -> () = dict::nsl_dict_free_tensor_values;
            // String comparison
            [collections] nsl_str_eq(i64, i64) -> i64 = string_ops::nsl_str_eq;
            // String repeat & slice
            [collections] nsl_str_repeat(i64, i64) -> i64 = string_ops::nsl_str_repeat;
            [collections] nsl_list_slice(i64, i64, i64, i64) -> i64 = list::nsl_list_slice;
            [collections] nsl_str_slice(i64, i64, i64, i64) -> i64 = string_ops::nsl_str_slice;
            // String methods
            [collections] nsl_str_upper(i64) -> i64 = string::nsl_str_upper;
            [collections] nsl_str_lower(i64) -> i64 = string::nsl_str_lower;
            [collections] nsl_str_strip(i64) -> i64 = string::nsl_str_strip;
            [collections] nsl_str_split(i64, i64) -> i64 = string::nsl_str_split;
            [collections] nsl_str_join(i64, i64) -> i64 = string::nsl_str_join;
            [collections] nsl_str_replace(i64, i64, i64) -> i64 = string::nsl_str_replace;
            [collections] nsl_str_find(i64, i64) -> i64 = string::nsl_str_find;
            [collections] nsl_str_startswith(i64, i64) -> i8 = string::nsl_str_startswith;
            [collections] nsl_str_endswith(i64, i64) -> i8 = string::nsl_str_endswith;
            [collections] nsl_str_contains(i64, i64) -> i8 = string::nsl_str_contains;
            // Higher-order functions. Third arg = ret_is_bool: the function
            // pointer is invoked as fn(i64)->i64, but a bool-returning NSL fn
            // compiles to an I8 return whose upper register bits are undefined —
            // the runtime masks to the low byte when this flag is set (hof.rs).
            [collections] nsl_map(i64, i64, i64) -> i64 = hof::nsl_map;
            [collections] nsl_filter(i64, i64, i64) -> i64 = hof::nsl_filter;
            [collections] nsl_enumerate(i64) -> i64 = hof::nsl_enumerate;
            [collections] nsl_zip(i64, i64) -> i64 = hof::nsl_zip;
            [collections] nsl_sorted(i64) -> i64 = hof::nsl_sorted;
            [collections] nsl_reversed(i64) -> i64 = hof::nsl_reversed;
            // String deallocation (M15)
            [collections] nsl_string_free(i64) -> () = string::nsl_string_free;

            // ── tensor (was builtins/tensor.rs) ──
            // Tensor creation
            [tensor] nsl_tensor_zeros(i64) -> i64 = tensor::creation::nsl_tensor_zeros;
            [tensor] nsl_tensor_ones(i64) -> i64 = tensor::creation::nsl_tensor_ones;
            [tensor] nsl_tensor_rand(i64) -> i64 = tensor::creation::nsl_tensor_rand;
            [tensor] nsl_tensor_randn(i64) -> i64 = tensor::creation::nsl_tensor_randn;
            // Tensor element access
            [tensor] nsl_tensor_get(i64, i64) -> f64 = tensor::nsl_tensor_get;
            [tensor] nsl_tensor_set(i64, i64, f64) -> () = tensor::nsl_tensor_set;
            // Tensor shape ops
            [tensor] nsl_tensor_shape(i64) -> i64 = tensor::shape_ops::nsl_tensor_shape;
            [tensor] nsl_tensor_shape_dim(i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_shape_dim;
            // M28: Dynamic shape assertions
            [tensor] nsl_tensor_assert_dim(i64, i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_assert_dim;
            [tensor] nsl_tensor_assert_dim_bound(i64, i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_assert_dim_bound;
            [tensor] nsl_tensor_ndim(i64) -> i64 = tensor::shape_ops::nsl_tensor_ndim;
            // PCA Stage C: non-aborting shape probe (0 for out-of-range dims).
            [tensor] nsl_tensor_dim_or_zero(i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_dim_or_zero;
            [tensor] nsl_tensor_len(i64) -> i64 = tensor::shape_ops::nsl_tensor_len;
            [tensor] nsl_tensor_get_dtype(i64) -> i64 = tensor::shape_ops::nsl_tensor_get_dtype;
            [tensor] nsl_tensor_reshape(i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_reshape;
            [tensor] nsl_tensor_transpose(i64, i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_transpose;
            // Tensor arithmetic (elementwise)
            [tensor] nsl_tensor_add(i64, i64, i8) -> i64 = tensor::arithmetic::nsl_tensor_add;
            [tensor] nsl_tensor_sub(i64, i64, i8) -> i64 = tensor::arithmetic::nsl_tensor_sub;
            [tensor] nsl_tensor_mul(i64, i64, i8) -> i64 = tensor::arithmetic::nsl_tensor_mul;
            [tensor] nsl_tensor_div(i64, i64, i8) -> i64 = tensor::arithmetic::nsl_tensor_div;
            [tensor] nsl_tensor_neg(i64) -> i64 = tensor::arithmetic::nsl_tensor_neg;
            // Tensor scalar ops
            [tensor] nsl_tensor_add_scalar(i64, f64, i8) -> i64 = tensor::arithmetic::nsl_tensor_add_scalar;
            [tensor] nsl_tensor_mul_scalar(i64, f64, i8) -> i64 = tensor::arithmetic::nsl_tensor_mul_scalar;
            // Scalar-immediate siblings (MFU campaign C3): x / s and x - s with the
            // scalar as an argument instead of a broadcast-materialized tensor.
            [tensor] nsl_tensor_div_scalar(i64, f64, i8) -> i64 = tensor::arithmetic::nsl_tensor_div_scalar;
            [tensor] nsl_tensor_sub_scalar(i64, f64, i8) -> i64 = tensor::arithmetic::nsl_tensor_sub_scalar;
            // Dispatching scalar-immediate entry (review fix): f32 -> the dedicated
            // scalar kernel; any other dtype -> the literal decomposed baseline
            // (preserves the mixed-dtype "f32 wins" narrowing AND output dtype).
            // (tensor, f64 scalar, descriptor-v1 opcode) -> result handle.
            [tensor] nsl_tensor_scalar_rhs(i64, f64, i64) -> i64 = tensor::fused_chain::nsl_tensor_scalar_rhs;
            // Tensor matmul
            [tensor] nsl_tensor_matmul(i64, i64, i8) -> i64 = tensor::arithmetic::nsl_tensor_matmul;
            // Tensor reductions (return scalar tensor ptr, not f64)
            [tensor] nsl_tensor_sum(i64) -> i64 = tensor::reduction::nsl_tensor_sum;
            [tensor] nsl_tensor_mean(i64) -> i64 = tensor::reduction::nsl_tensor_mean;
            // Tensor scalar extraction
            [tensor] nsl_tensor_item(i64) -> f64 = tensor::nsl_tensor_item;
            [tensor] nsl_tensor_l2_norm(i64) -> f64 = tensor::nsl_tensor_l2_norm;
            // Tensor display
            [tensor] nsl_tensor_print(i64) -> () = tensor::nsl_tensor_print;
            // Tensor memory
            [tensor] nsl_tensor_clone(i64) -> i64 = tensor::nsl_tensor_clone;
            [tensor] nsl_tensor_clone_if_valid(i64) -> i64 = tensor::nsl_tensor_clone_if_valid;
            [tensor] nsl_tensor_free(i64) -> () = tensor::nsl_tensor_free;
            [tensor] nsl_tensor_free_if_valid(i64) -> () = tensor::nsl_tensor_free_if_valid;
            [tensor] nsl_tensor_free_transient(i64) -> () = tensor::nsl_tensor_free_transient;
            [tensor] nsl_tensor_retain(i64) -> () = tensor::nsl_tensor_retain;
            [tensor] nsl_tensor_release(i64) -> () = tensor::nsl_tensor_release;
            [tensor] nsl_tensor_scope_begin() -> () = tensor::nsl_tensor_scope_begin;
            [tensor] nsl_tensor_scope_end(i64) -> () = tensor::nsl_tensor_scope_end;
            // Element-wise tensor ops (M14)
            [tensor] nsl_tensor_exp(i64) -> i64 = tensor::activation::nsl_tensor_exp;
            [tensor] nsl_tensor_log(i64) -> i64 = tensor::activation::nsl_tensor_log;
            [tensor] nsl_tensor_sqrt(i64) -> i64 = tensor::activation::nsl_tensor_sqrt;
            [tensor] nsl_tensor_abs(i64) -> i64 = tensor::activation::nsl_tensor_abs;
            [tensor] nsl_tensor_sign(i64) -> i64 = tensor::activation::nsl_tensor_sign;
            [tensor] nsl_tensor_clamp(i64, f64, f64) -> i64 = tensor::activation::nsl_tensor_clamp;
            // Dimensional reductions (M14)
            [tensor] nsl_tensor_sum_dim(i64, i64, i64) -> i64 = tensor::reduction::nsl_tensor_sum_dim;
            [tensor] nsl_tensor_mean_dim(i64, i64, i64) -> i64 = tensor::reduction::nsl_tensor_mean_dim;
            [tensor] nsl_tensor_reduce_max(i64, i64, i64) -> i64 = tensor::reduction::nsl_tensor_reduce_max;
            [tensor] nsl_tensor_gather(i64, i64, i64) -> i64 = tensor::reduction::nsl_tensor_gather;
            // In-place mutation ops (M14)
            [tensor] nsl_tensor_copy_data(i64, i64) -> () = tensor::nsl_tensor_copy_data;
            [tensor] nsl_tensor_add_inplace(i64, i64) -> () = tensor::nsl_tensor_add_inplace;
            [tensor] nsl_tensor_zero_inplace(i64) -> () = tensor::nsl_tensor_zero_inplace;
            [tensor] nsl_tensor_zeros_like(i64) -> i64 = tensor::nsl_tensor_zeros_like;
            // M52b: Create tensor from static .rodata data (compile-time constant folded)
            [tensor] nsl_tensor_from_static(i64, i64, i64) -> i64 = tensor::nsl_tensor_from_static;
            // Activation functions (M15)
            [tensor] nsl_tensor_relu(i64) -> i64 = tensor::activation::nsl_tensor_relu;
            [tensor] nsl_tensor_gelu(i64) -> i64 = tensor::activation::nsl_tensor_gelu;
            [tensor] nsl_tensor_silu(i64) -> i64 = tensor::activation::nsl_tensor_silu;
            [tensor] nsl_tensor_sigmoid(i64) -> i64 = tensor::activation::nsl_tensor_sigmoid;
            [tensor] nsl_tensor_tanh_act(i64) -> i64 = tensor::activation::nsl_tensor_tanh_act;
            // Tensor trig (RoPE support)
            [tensor] nsl_tensor_sin(i64) -> i64 = tensor::trig::nsl_tensor_sin;
            [tensor] nsl_tensor_cos(i64) -> i64 = tensor::trig::nsl_tensor_cos;
            // Fused rotate_half (RoPE support)
            [tensor] nsl_tensor_rotate_half(i64) -> i64 = tensor::shape_ops::nsl_tensor_rotate_half;
            // Slice & Cat (M15)
            [tensor] nsl_tensor_slice(i64, i64, i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_slice;
            [tensor] nsl_tensor_cat(i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_cat;
            // Embedding lookup (M15)
            [tensor] nsl_tensor_embedding_lookup(i64, i64) -> i64 = tensor::nsl_tensor_embedding_lookup;
            // LayerNorm & RMSNorm (M15)
            [tensor] nsl_tensor_layernorm(i64, i64, i64, f64) -> i64 = tensor::nsl_tensor_layernorm;
            [tensor] nsl_tensor_rmsnorm(i64, i64, f64) -> i64 = tensor::nsl_tensor_rmsnorm;
            // Dropout, Conv2d, MaxPool2d (M15)
            [tensor] nsl_tensor_dropout(i64, f64, i8) -> i64 = tensor::nsl_tensor_dropout;
            // Bias add (M15 — broadcast 1D bias over 2D tensor)
            [tensor] nsl_tensor_bias_add(i64, i64) -> i64 = tensor::nsl_tensor_bias_add;
            // Tensor creation helpers (M17)
            [tensor] nsl_tensor_zeros_on(i64, i64) -> i64 = tensor::nsl_tensor_zeros_on;
            // CSHA Gap I.3 (A+F): f16 (dtype=2, 2 bytes/element) zeros allocator.
            // The Tier C backward kernel writes dq/dk/dv/dwq/dwk/dwv via
            // `st.global.u16`; the f32 `_zeros_on` variant over-allocates by 2×
            // and leaves every second byte uninitialised → host-side f32 reads
            // then interpret raw f16 bits as f32 → garbage → weight corruption.
            // `dx` stays on `_zeros_on` because the kernel writes it as f32.
            [tensor] nsl_tensor_zeros_f16_on(i64, i64) -> i64 = tensor::nsl_tensor_zeros_f16_on;
            [tensor] nsl_tensor_ones_like(i64) -> i64 = tensor::nsl_tensor_ones_like;
            // Shape manipulation ops (M18a)
            [tensor] nsl_tensor_unsqueeze(i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_unsqueeze;
            [tensor] nsl_tensor_select(i64, i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_select;
            [tensor] nsl_tensor_stack(i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_stack;
            [tensor] nsl_tensor_expand(i64, i64) -> i64 = tensor::shape_ops::nsl_tensor_expand;
            [tensor] nsl_tensor_contiguous(i64) -> i64 = tensor::shape_ops::nsl_tensor_contiguous;
            [tensor] nsl_tensor_causal_mask(i64) -> i64 = tensor::shape_ops::nsl_tensor_causal_mask;
            // Sampling primitives (M19)
            [tensor] nsl_manual_seed(i64) -> () = sampling::nsl_manual_seed;
            [tensor] nsl_tensor_topk(i64, i64, i64) -> i64 = sampling::nsl_tensor_topk;
            [tensor] nsl_tensor_multinomial(i64, i64) -> i64 = sampling::nsl_tensor_multinomial;
            [tensor] nsl_tensor_argmax(i64, i64) -> i64 = sampling::nsl_tensor_argmax;
            [tensor] nsl_tensor_cumsum(i64, i64) -> i64 = sampling::nsl_tensor_cumsum;
            [tensor] nsl_tensor_lt_scalar(i64, f64) -> i64 = sampling::nsl_tensor_lt_scalar;
            // Tensor mutation (M19)
            [tensor] nsl_tensor_set_element(i64, i64, i64, f64) -> () = tensor::nsl_tensor_set_element;
            [tensor] nsl_tensor_slice_assign(i64, i64, i64, i64) -> () = tensor::nsl_tensor_slice_assign;
            // BatchNorm + AvgPool2d (proper implementations replacing approximations)
            [tensor] nsl_tensor_batchnorm(i64, i64, i64, f64, i64) -> i64 = tensor::ad_ops::nsl_tensor_batchnorm;
            [tensor] nsl_tensor_avgpool2d(i64, i64, i64, i64, i64) -> i64 = tensor::ad_ops::nsl_tensor_avgpool2d;

            // ── abi_tensor (was runtime_abi/tensor.rs) ──
            // Fused elementwise-chain launcher (MFU campaign C3):
            // (ptx, kname, descriptor, desc_len, in0..in5, n_inputs) -> result handle.
            [abi_tensor] nsl_fused_ew_chain(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = tensor::fused_chain::nsl_fused_ew_chain;
            // FASE fused scaled-add epilogue (p4): m += s * g, in place, void.
            [abi_tensor] nsl_tensor_scalar_mul_add_inplace(i64, i64, f64) -> () = tensor::arithmetic::nsl_tensor_scalar_mul_add_inplace;
            // Item 7 fused weight-gradient accumulate: m += s * (x^T @ g), in place,
            // void. Args (m_partial, x, g, s) — x/g are the PRE-transpose activation
            // and the upstream gradient; the transpose is folded into the GEMM.
            [abi_tensor] nsl_tensor_wgrad_accum(i64, i64, i64, f64) -> () = tensor::arithmetic::nsl_tensor_wgrad_accum;
            // WRGA B.3 Task 4: fused LoRA/IA³ adapter matmul FFIs.
            // LoRA args: (x_ptr, w_ptr, a_ptr, b_ptr, scale_f64, kernel_handle_i64).
            // The scale is f64 at the FFI boundary because NSL FloatLiteral is f64;
            // the runtime narrows to f32 internally.
            [abi_tensor] nsl_adapter_fused_lora_matmul(i64, i64, i64, i64, f64, i64) -> i64 = fused_adapter::nsl_adapter_fused_lora_matmul;
            // IA³ args: (x_ptr, w_ptr, ia3_scale_ptr, kernel_handle_i64)
            [abi_tensor] nsl_adapter_fused_ia3_matmul(i64, i64, i64, i64) -> i64 = fused_adapter::nsl_adapter_fused_ia3_matmul;
            // B.3.1 GatedLoRA args: (x_ptr, w_ptr, a_ptr, b_ptr, scale_f64, gate_ptr, kernel_handle_i64).
            // Body is a stub returning the base matmul (x @ W) until Task 5.0.c registers the real
            // fused PTX kernel.  The FFI declaration is needed here so compile_call resolves the
            // callee symbol and falls through to compile_traced_call rather than compile_indirect_call.
            [abi_tensor] nsl_adapter_fused_gatedlora_matmul(i64, i64, i64, i64, f64, i64, i64) -> i64 = fused_adapter::nsl_adapter_fused_gatedlora_matmul;
            // WRGA B.3 Task 5.6: fused-PTX runtime registry registration.
            // Args: (handle_i64, ptx_ptr_i64, ptx_len_i64, name_ptr_i64, name_len_i64).
            // Called from `main` preamble, one call per unique (m,n,k,rank,sm) key.
            [abi_tensor] nsl_wrga_register_fused_ptx(i64, i64, i64, i64, i64) -> () = fused_adapter::nsl_wrga_register_fused_ptx;
            // Fused elementwise operations (M31 fusion lowering)
            [abi_tensor] nsl_fused_elementwise_2(i64, i64, i64, i64) -> i64 = cpu::nsl_fused_elementwise_2;
            [abi_tensor] nsl_fused_elementwise_1(i64, i64, i64) -> i64 = cpu::nsl_fused_elementwise_1;
            [abi_tensor] nsl_fused_matmul_epilogue(i64, i64, i64, i64, i64) -> i64 = cpu::nsl_fused_matmul_epilogue;
            // FBIP Phase 2: unconditional in-place variants (compiler-guaranteed single-use)
            [abi_tensor] nsl_tensor_relu_inplace(i64) -> i64 = tensor::activation::nsl_tensor_relu_inplace;
            [abi_tensor] nsl_tensor_exp_inplace(i64) -> i64 = tensor::activation::nsl_tensor_exp_inplace;
            [abi_tensor] nsl_tensor_log_inplace(i64) -> i64 = tensor::activation::nsl_tensor_log_inplace;
            [abi_tensor] nsl_tensor_sqrt_inplace(i64) -> i64 = tensor::activation::nsl_tensor_sqrt_inplace;
            [abi_tensor] nsl_tensor_abs_inplace(i64) -> i64 = tensor::activation::nsl_tensor_abs_inplace;
            [abi_tensor] nsl_tensor_sigmoid_inplace(i64) -> i64 = tensor::activation::nsl_tensor_sigmoid_inplace;
            [abi_tensor] nsl_tensor_tanh_inplace(i64) -> i64 = tensor::activation::nsl_tensor_tanh_inplace;
            [abi_tensor] nsl_tensor_neg_inplace(i64) -> i64 = tensor::activation::nsl_tensor_neg_inplace;
            [abi_tensor] nsl_tensor_sign_inplace(i64) -> i64 = tensor::activation::nsl_tensor_sign_inplace;
            [abi_tensor] nsl_tensor_gelu_inplace(i64) -> i64 = tensor::activation::nsl_tensor_gelu_inplace;
            [abi_tensor] nsl_tensor_silu_inplace(i64) -> i64 = tensor::activation::nsl_tensor_silu_inplace;
            // CFTP §4.3 / Tier A activation — extract raw device pointer from
            // NslTensor* for use by nsl_packing_metadata_set. Returns 0 when
            // tensor_ptr == 0. See spec 2026-05-17-pca-rope-activation-design.md.
            [abi_tensor] nsl_tensor_data_ptr(i64) -> i64 = tensor::nsl_tensor_data_ptr;
            // CFTP §4.3 / Tier A activation — thread-local registry for the
            // segment_ids/doc_starts pointers. Train block sets per step;
            // CSHA call sites read.
            [abi_tensor] nsl_packing_metadata_set(i64, i64) -> () = pca_rope_runtime::nsl_packing_metadata_set;
            [abi_tensor] nsl_packing_metadata_get_segment_ids() -> i64 = pca_rope_runtime::nsl_packing_metadata_get_segment_ids;
            [abi_tensor] nsl_packing_metadata_get_doc_starts() -> i64 = pca_rope_runtime::nsl_packing_metadata_get_doc_starts;
            // PCA Tier A (spec §6.1) — mismatch warning: warns once if a
            // segment-masked module sees no segment_ids in the first N steps.
            [abi_tensor] nsl_pca_packing_mismatch_check(i64) -> () = pca_rope_runtime::nsl_pca_packing_mismatch_check;
            // CPDT precision-adaptive optimizer: cast / zeros helpers
            [abi_tensor] nsl_tensor_cast(i64, i64) -> i64 = tensor::precision_cast::nsl_tensor_cast;
            [abi_tensor] nsl_tensor_cast_into(i64, i64) -> () = tensor::precision_cast::nsl_tensor_cast_into;
            [abi_tensor] nsl_tensor_zeros_like_dtype(i64, i64) -> i64 = tensor::precision_cast::nsl_tensor_zeros_like_dtype;
            // Optimizer-state offload (scaling campaign item 4): host-resident f32
            // zeros with the template's shape, regardless of the template's device.
            [abi_tensor] nsl_tensor_zeros_like_host_f32(i64) -> i64 = tensor::nsl_tensor_zeros_like_host_f32;
            // Offload P0.2: async copy-back (CONSUMES src — replaces the emitted
            // copy_data+free pair) + the once-per-step drain point.
            [abi_tensor] nsl_tensor_copy_data_async(i64, i64) -> () = tensor::nsl_tensor_copy_data_async;
            [abi_tensor] nsl_offload_drain() -> () = tensor::nsl_offload_drain;
            // Offload P0.3 (offload x reduced-precision composition): host state
            // at the planned dtype + the cross-device quant/dequant envelope.
            [abi_tensor] nsl_tensor_zeros_like_host_dtype(i64, i64) -> i64 = tensor::precision_cast::nsl_tensor_zeros_like_host_dtype;
            [abi_tensor] nsl_tensor_cast_to_host_into(i64, i64) -> () = tensor::precision_cast::nsl_tensor_cast_to_host_into;
            [abi_tensor] nsl_tensor_cast_from_host(i64, i64) -> i64 = tensor::precision_cast::nsl_tensor_cast_from_host;
            // CFTP v6 forward inline-cast wrappers: src_ptr -> new tensor (scope-tracked).
            [abi_tensor] nsl_tensor_to_bf16(i64) -> i64 = tensor::precision_cast::nsl_tensor_to_bf16;
            [abi_tensor] nsl_tensor_to_fp16(i64) -> i64 = tensor::precision_cast::nsl_tensor_to_fp16;
            [abi_tensor] nsl_tensor_to_f32(i64) -> i64 = tensor::precision_cast::nsl_tensor_to_f32;
            // M52c: CSR sparse matmul (row_ptrs, col_indices, values, B, nrows, ncols, nnz) -> C
            [abi_tensor] nsl_sparse_matmul(i64, i64, i64, i64, i64, i64, i64) -> i64 = tensor::arithmetic::nsl_sparse_matmul;
            // Fused RoPE backward: -rotate_half(dy) in one launch (bit-exact)
            [abi_tensor] nsl_tensor_rotate_half_neg(i64) -> i64 = tensor::shape_ops::nsl_tensor_rotate_half_neg;
            [abi_tensor] nsl_tensor_softmax(i64, i64) -> i64 = tensor::reduction::nsl_tensor_softmax;
            // Item 9: fused RMSNorm input-gradient (dy, x, gamma, eps) -> dx.
            [abi_tensor] nsl_rmsnorm_dx_backward(i64, i64, i64, f64) -> i64 = tensor::nsl_rmsnorm_dx_backward;
            // P5 slice C: fused RMSNorm dx + residual fold (dy, x, gamma, res, eps) -> dx+res.
            [abi_tensor] nsl_rmsnorm_dx_backward_add(i64, i64, i64, i64, f64) -> i64 = tensor::nsl_rmsnorm_dx_backward_add;
            // P5 item 20 slice A: fused RMSNorm gamma-gradient (dy, x, gamma, eps) -> dgamma.
            [abi_tensor] nsl_rmsnorm_dgamma_backward(i64, i64, i64, f64) -> i64 = tensor::nsl_rmsnorm_dgamma_backward;
            // Source AD: reduce gradient to match parameter shape (matmul broadcast backward)
            [abi_tensor] nsl_tensor_reduce_to_shape(i64, i64) -> i64 = tensor::ad_ops::nsl_tensor_reduce_to_shape;
            // Source-AD dropout forward: returns an NslList [out, mask] so the
            // compiled backward can consume the exact RNG mask (two-op split,
            // wengert.rs DropoutMask).
            [abi_tensor] nsl_tensor_dropout_fwd_mask(i64, f64) -> i64 = tensor::nsl_tensor_dropout_fwd_mask;
            [abi_tensor] nsl_tensor_conv2d(i64, i64, i64, i64, i64, i64, i64) -> i64 = tensor::nsl_tensor_conv2d;
            // Source-AD conv2d backward FFIs: (grad, input, weight, sh, sw, ph, pw) -> grad.
            [abi_tensor] nsl_conv2d_input_backward(i64, i64, i64, i64, i64, i64, i64) -> i64 = autodiff::backward::nsl_conv2d_input_backward;
            [abi_tensor] nsl_conv2d_weight_backward(i64, i64, i64, i64, i64, i64, i64) -> i64 = autodiff::backward::nsl_conv2d_weight_backward;
            [abi_tensor] nsl_conv2d_bias_backward(i64, i64, i64, i64, i64, i64, i64) -> i64 = autodiff::backward::nsl_conv2d_bias_backward;
            // Reify grad_output to the conv2d output shape once per node, shared by
            // the 3 FFIs above: (grad, input, weight, sh, sw, ph, pw) -> grad.
            [abi_tensor] nsl_materialize_conv_output_grad(i64, i64, i64, i64, i64, i64, i64) -> i64 = autodiff::backward::nsl_materialize_conv_output_grad;
            [abi_tensor] nsl_tensor_maxpool2d(i64, i64, i64, i64, i64) -> i64 = tensor::nsl_tensor_maxpool2d;
            // FlashAttention-2 launch wrappers (M27)
            [abi_tensor] nsl_flash_attention(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = flash_attention::nsl_flash_attention;
            // CSHA Tier A.1: FA launcher variant carrying per-layer CSHA extras.
            // Same 24-arg prelude as `nsl_flash_attention`, then 9 CSHA args:
            //   x, norm_weight, Wq, Wk, Wv, Wo, rmsnorm_eps_bits, active_heads, d_model.
            // Stub today — forwards to `nsl_flash_attention`; A.2 will light up
            // the CSHA PTX body.
            [abi_tensor] nsl_flash_attention_csha(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = flash_attention::nsl_flash_attention_csha;
            // Gap A: CSHA FA launcher with activation-save pointers for source-AD
            // backward. Identical to `nsl_flash_attention_csha` plus 6 trailing
            // save-pointer args (q_proj, k_proj, v_proj, row_max, row_sum, x_raw).
            // Emitted by `compile_flash_attention_call` when
            // `CshaExtras.save_activations_for_backward` is true (i.e. inside a
            // `@train` block with CSHA fused).
            [abi_tensor] nsl_flash_attention_csha_with_saves(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = flash_attention::nsl_flash_attention_csha_with_saves;
            // Gap A: codegen-side allocator for the 6 CSHA backward-activation
            // HBM buffers. Writes 6 device-pointer i64s contiguously into
            // `out_ptr` (caller-supplied stack-slot). Returns 0 on success.
            [abi_tensor] nsl_csha_alloc_backward_activations_into(i64, i64, i64, i64, i64) -> i64 = flash_attention::nsl_csha_alloc_backward_activations_into;
            // Gap A: codegen-side free helper. Takes 6 i64 pointers matching the
            // layout written by `nsl_csha_alloc_backward_activations_into`.
            [abi_tensor] nsl_csha_free_backward_activations_from(i64, i64, i64, i64, i64, i64) -> () = flash_attention::nsl_csha_free_backward_activations_from;
            // Gap D / Tier C (extended by Gap I.5 Option A): CSHA fused backward
            // launch. i64 args matching the wengert_lower.rs
            // `PrimalOp::FusedCshaBackward` emission order, plus the trailing
            // tier_b2_active flag (CSHA Tier B.2 Phase 3 T6):
            //   36-arg forward-side prelude mirrored off `_with_saves`,
            //   + 6 forward-saved activation pointers,
            //   + dO input + 8 gradient outputs
            //     (dq, dk, dv, dwq, dwk, dwv, dx, dx_norm).
            // First surfaced as "undefined function" in the Gap I.3 smoke once
            // A+F let the backward launch actually fire. Gap I.5 appended the
            // 8th output (`dx_norm`) so the AD-side `RmsNormGammaBackward` gets
            // the correct `dy_norm` input.
            [abi_tensor] nsl_flash_attention_csha_backward(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = flash_attention::nsl_flash_attention_csha_backward;
            // FlashAttention-2 backward (M27 backward pass)
            // Returns NslList [dQ, dK, dV]. When logsumexp_ptr == 0, auto-computes lse.
            //
            // The trailing Tier-B sentinel pair (planner spec §4) was added to the
            // runtime FFI but this declaration and the wengert_lower call site were
            // never extended — Cranelift emitted 16-arg calls against the 18-param C
            // function, so the runtime read undefined stack/registers for the pair
            // and `assert_tier_b_sentinels` aborted the process at the FIRST plain-
            // SDPA training backward (found by the roadmap-4.2 pretrain e2e; no prior
            // test ran the compiler-EMITTED call — GPU parity tests build FFI args by
            // hand). Keep this in lock-step with `nsl_flash_attention_backward` in
            // nsl-runtime/src/flash_attention.rs.
            [abi_tensor] nsl_flash_attention_backward(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = flash_attention::nsl_flash_attention_backward;
            // PCA Stage C: plain fused SDPA forward with saves. Launches the v2
            // scalar forward (csha: None) selected by the wengert lowering's
            // per-head_dim variant table; returns an NslList* [out, lse] or 0 to
            // DECLINE (caller's decomposed fallback runs). segment_ids != 0 selects
            // the segment-masked kernel family (packed attention); the Tier-B pair
            // is the tile-skip variant behind the runtime gate. Keep in lock-step
            // with `nsl_sdpa_fused_forward` in nsl-runtime/src/flash_attention.rs.
            [abi_tensor] nsl_sdpa_fused_forward(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = flash_attention::nsl_sdpa_fused_forward;
            // PCA Stage C: align packed-batch mask/segment tensors to the params'
            // device at step start (train-block batch prep). No-op for CPU models
            // and unpacked batches. Keep in lock-step with
            // `nsl_packed_batch_align_device` in nsl-runtime/src/packing.rs.
            [abi_tensor] nsl_packed_batch_align_device(i64, i64) -> i64 = packing::nsl_packed_batch_align_device;
            // Campaign item 5: derive the dense [b,1,s,s] packed mask from
            // segment_ids at the decomposed-fallback site (the DataLoader no
            // longer ships attention_mask by default). Keep in lock-step with
            // `nsl_packed_mask_from_segment_ids` in nsl-runtime/src/packing.rs.
            [abi_tensor] nsl_packed_mask_from_segment_ids(i64) -> i64 = packing::nsl_packed_mask_from_segment_ids;
            // --- M46: Deterministic kernel variants ---
            [abi_tensor] nsl_tensor_reduce_sum_deterministic(i64, i64, i64) -> i64 = deterministic_ops::nsl_tensor_reduce_sum_deterministic;
            [abi_tensor] nsl_tensor_reduce_mean_deterministic(i64, i64, i64) -> i64 = deterministic_ops::nsl_tensor_reduce_mean_deterministic;
            [abi_tensor] nsl_tensor_scatter_add_deterministic(i64, i64, i64) -> i64 = deterministic_ops::nsl_tensor_scatter_add_deterministic;
            // --- M46: Global deterministic mode flag + RNG seeding ---
            [abi_tensor] nsl_set_deterministic(i64) -> i64 = deterministic_ops::nsl_set_deterministic;
            [abi_tensor] nsl_rng_seed(i64) -> i64 = deterministic_ops::nsl_rng_seed;
            // Item 4 (2026-09-02): the matmul arithmetic configuration, promoted from
            // the NSL_MATMUL_BF16* environment family. Installed before user code, from
            // the same compile options the execution fingerprint renders -- so the
            // runtime and the fingerprint cannot disagree about which arithmetic ran.
            // (mode, rounding, min_ratio, cast_cache, lt, lt_workspace_mib, lt_tune)
            [abi_tensor] nsl_set_matmul_config(i64, i64, f64, i64, i64, i64, i64) -> i64 = matmul_config::nsl_set_matmul_config;
            // --- M50: Sparse tensors ---
            [abi_tensor] nsl_sparse_coo(i64, i64, i64, i64, i64, i64) -> i64 = sparse::nsl_sparse_coo;
            [abi_tensor] nsl_sparse_from_dense(i64, i64, i64) -> i64 = sparse::nsl_sparse_from_dense;
            // M50b: +threshold
            [abi_tensor] nsl_sparse_to_dense(i64) -> i64 = sparse::nsl_sparse_to_dense;
            [abi_tensor] nsl_sparse_nnz(i64) -> i64 = sparse::nsl_sparse_nnz;
            [abi_tensor] nsl_sparse_density(i64) -> i64 = sparse::nsl_sparse_density;
            [abi_tensor] nsl_sparse_spmm(i64, i64) -> i64 = sparse::nsl_sparse_spmm;
            [abi_tensor] nsl_sparse_spmv(i64, i64) -> i64 = sparse::nsl_sparse_spmv;
            [abi_tensor] nsl_sparse_coo_to_csr(i64) -> i64 = sparse::nsl_sparse_coo_to_csr;
            [abi_tensor] nsl_sparse_coo_to_csc(i64) -> i64 = sparse::nsl_sparse_coo_to_csc;
            [abi_tensor] nsl_sparse_csr_to_csc(i64) -> i64 = sparse::nsl_sparse_csr_to_csc;
            [abi_tensor] nsl_sparse_csc_to_csr(i64) -> i64 = sparse::nsl_sparse_csc_to_csr;
            [abi_tensor] nsl_sparse_csr_to_coo(i64) -> i64 = sparse::nsl_sparse_csr_to_coo;
            [abi_tensor] nsl_sparse_csc_to_coo(i64) -> i64 = sparse::nsl_sparse_csc_to_coo;
            [abi_tensor] nsl_sparse_add(i64, i64) -> i64 = sparse::nsl_sparse_add;
            [abi_tensor] nsl_sparse_mul(i64, i64) -> i64 = sparse::nsl_sparse_mul;
            [abi_tensor] nsl_sparse_free(i64) -> i64 = sparse::nsl_sparse_free;
            // --- M40: Source AD runtime helpers ---
            [abi_tensor] nsl_tensor_compare(i64, i64, i64) -> i64 = tensor::ad_ops::nsl_tensor_compare;
            [abi_tensor] nsl_tensor_where(i64, i64, i64) -> i64 = tensor::ad_ops::nsl_tensor_where;
            [abi_tensor] nsl_tensor_scalar(f64, i64) -> i64 = tensor::ad_ops::nsl_tensor_scalar;
            [abi_tensor] nsl_tensor_pad_zero(i64, i64, i64, i64) -> i64 = tensor::ad_ops::nsl_tensor_pad_zero;
            [abi_tensor] nsl_tensor_scatter_add(i64, i64, i64) -> i64 = tensor::ad_ops::nsl_tensor_scatter_add;
            [abi_tensor] nsl_embedding_backward(i64, i64, i64) -> i64 = tensor::ad_ops::nsl_embedding_backward;
            [abi_tensor] nsl_cross_entropy_backward(i64, i64, i64) -> i64 = tensor::ad_ops::nsl_cross_entropy_backward;
            [abi_tensor] nsl_mse_backward(i64, i64, i64) -> i64 = tensor::ad_ops::nsl_mse_backward;
            [abi_tensor] nsl_l1_backward(i64, i64, i64) -> i64 = tensor::ad_ops::nsl_l1_backward;
            // p4 slice 2: fused SiLU backward — grad * σ(x)*(1 + x*(1-σ(x))).
            [abi_tensor] nsl_tensor_silu_backward(i64, i64) -> i64 = tensor::activation::nsl_tensor_silu_backward;
            // P5 item 20 slice B: fused SwiGLU gate backward (y_bar, up, gate_in) -> dgate.
            [abi_tensor] nsl_tensor_swiglu_gate_backward(i64, i64, i64) -> i64 = tensor::activation::nsl_tensor_swiglu_gate_backward;
            // p4 slice 3: fused Sigmoid backward — grad * y*(1-y), y = σ output.
            [abi_tensor] nsl_tensor_sigmoid_backward(i64, i64) -> i64 = tensor::activation::nsl_tensor_sigmoid_backward;
            // p4 slice 3: fused Tanh backward — grad * (1 - y*y), y = tanh output.
            [abi_tensor] nsl_tensor_tanh_backward(i64, i64) -> i64 = tensor::activation::nsl_tensor_tanh_backward;
            // p4 slice 4: source-AD in-place suppression guard (on!=0 enter / on==0 leave)
            // — raised around the source-AD forward so FBIP preserves primal inputs.
            [abi_tensor] nsl_set_inplace_suppressed(i64) -> () = tensor::nsl_set_inplace_suppressed;
            // p4 GELU fix: fused GELU backward — grad * gelu'(x), per-device derivative.
            [abi_tensor] nsl_tensor_gelu_backward(i64, i64) -> i64 = tensor::activation::nsl_tensor_gelu_backward;
            // LSE tape-carry gates: fused-SDPA launch counter (0 = base fwd kernel,
            // 1 = Tier-B tile-skip) — NSL-callable as sdpa_fused_launch_count(v).
            [abi_tensor] nsl_sdpa_fused_launch_count(i64) -> i64 = flash_attention::nsl_sdpa_fused_launch_count;
            // --- M39b: vmap runtime ---
            [abi_tensor] nsl_vmap_check_batch(i64, i64, i64) -> i64 = vmap_runtime::nsl_vmap_check_batch;

            // ── training (was runtime_abi/training.rs) ──
            // Training mode
            [training] nsl_set_training_mode(i8) -> () = tensor::nsl_set_training_mode;
            [training] nsl_is_training() -> i8 = tensor::nsl_is_training;
            [training] nsl_tensor_full(i64, f64) -> i64 = tensor::creation::nsl_tensor_full;
            [training] nsl_tensor_arange(f64, f64, f64) -> i64 = tensor::creation::nsl_tensor_arange;
            // Autodiff tape management
            [training] nsl_tape_start(i64) -> () = autodiff::nsl_tape_start;
            [training] nsl_tape_stop() -> () = autodiff::nsl_tape_stop;
            [training] nsl_tape_backward(i64, i64) -> i64 = autodiff::backward::nsl_tape_backward;
            [training] nsl_tape_backward_train(i64, i64) -> i64 = autodiff::backward::nsl_tape_backward_train;
            [training] nsl_tape_pause() -> () = autodiff::nsl_tape_pause;
            [training] nsl_tape_resume() -> () = autodiff::nsl_tape_resume;
            // Gradient checkpointing
            [training] nsl_checkpoint_record(i64, i64, i64, i64) -> () = autodiff::nsl_checkpoint_record;
            // Gradient clipping (M14)
            [training] nsl_clip_grad_norm(i64, f64) -> () = tensor::nsl_clip_grad_norm;
            // Collect all tensor params from a model struct (recursive, magic-probed)
            [training] nsl_collect_model_params(i64, i64) -> i64 = tensor::nsl_collect_model_params;
            // Debug training: gradient checksum (--debug-training)
            [training] nsl_debug_grad_checksum(i64, i64) -> () = autodiff::backward::nsl_debug_grad_checksum;
            // P0.3 gradient-integrity gate (--grad-integrity)
            [training] nsl_grad_integrity_arm() -> () = grad_integrity::nsl_grad_integrity_arm;
            [training] nsl_grad_integrity_check(i64, i64) -> () = grad_integrity::nsl_grad_integrity_check;
            // (num_params, expected_notes_per_param) — the second argument is what
            // lets the runtime distinguish "every micro-batch reached this param"
            // from "k of N did", which the CSLA window bracket otherwise merges away.
            [training] nsl_grad_integrity_step_begin(i64, i64) -> () = grad_integrity::nsl_grad_integrity_step_begin;
            [training] nsl_grad_integrity_note(i64, i64) -> () = grad_integrity::nsl_grad_integrity_note;
            [training] nsl_grad_integrity_step_end() -> () = grad_integrity::nsl_grad_integrity_step_end;
            // Prefetch tensor to GPU asynchronously
            [training] nsl_tensor_prefetch(i64, i64) -> () = tensor::nsl_tensor_prefetch;
            // Item 2 (2026-08-25): refuse a host-resident dense-float step input on
            // a GPU-parameter train — the left-operand device-reconciliation rule
            // would otherwise silently drag the whole graph to the host.
            [training] nsl_train_input_device_guard(i64, i64) -> () = tensor::nsl_train_input_device_guard;
            // Checkpoint I/O (M14)
            [training] nsl_model_save(i64, i64, i64, i64) -> () = checkpoint::nsl_model_save;
            [training] nsl_model_load(i64, i64, i64) -> () = checkpoint::nsl_model_load;
            // Milestone B + item 8: full training-state checkpoint (θ .nslm + .optim
            // sidecar with m/v moments, micro-batch step counter, data position and
            // RNG state). Save: (path_ptr, path_len, names_list, param_list,
            // state_list_1, state_list_2, step_count, dataloader_handle_or_0,
            // train_epoch). Load: (path_ptr, path_len, param_list, state_list_1,
            // state_list_2, dataloader_handle_or_0) -> saved step counter; the
            // restored training epoch comes back through nsl_train_resume_epoch.
            [training] nsl_train_checkpoint_save(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> () = checkpoint::nsl_train_checkpoint_save;
            [training] nsl_train_checkpoint_load(i64, i64, i64, i64, i64, i64, i64) -> i64 = checkpoint::nsl_train_checkpoint_load;
            [training] nsl_train_resume_epoch() -> i64 = checkpoint::nsl_train_resume_epoch;
            // Data sources (M19)
            [training] nsl_load_jsonl(i64, i64, i64, i64) -> i64 = data_source::nsl_load_jsonl;
            [training] nsl_load_csv(i64, i64, i64, i64) -> i64 = data_source::nsl_load_csv;
            [training] nsl_load_mmap(i64, i64, i64) -> i64 = data_source::nsl_load_mmap;
            // DataLoader (M19)
            [training] nsl_dataloader_create(i64, i64, i64, i64) -> i64 = dataloader::nsl_dataloader_create;
            [training] nsl_dataloader_start(i64) -> () = dataloader::nsl_dataloader_start;
            [training] nsl_dataloader_next_batch(i64) -> i64 = dataloader::nsl_dataloader_next_batch;
            [training] nsl_dataloader_reset(i64) -> () = dataloader::nsl_dataloader_reset;
            [training] nsl_dataloader_stop(i64) -> () = dataloader::nsl_dataloader_stop;
            [training] nsl_dataloader_free(i64) -> () = dataloader::nsl_dataloader_free;
            // Item 8: resumable data position. `slot` is the loader's next DELIVERY
            // slot (not a batch count — the ragged-packed-tail sentinel makes those
            // differ), `identity` fingerprints the corpus + geometry it indexes.
            [training] nsl_dataloader_epoch(i64) -> i64 = dataloader::nsl_dataloader_epoch;
            [training] nsl_dataloader_slot(i64) -> i64 = dataloader::nsl_dataloader_slot;
            [training] nsl_dataloader_identity(i64) -> i64 = dataloader::nsl_dataloader_identity;
            [training] nsl_dataloader_resume_to(i64, i64, i64) -> () = dataloader::nsl_dataloader_resume_to;
            // Packing efficiency (M19)
            [training] nsl_packing_efficiency(i64) -> f64 = packing::nsl_packing_efficiency;
            // CFTP §4.4 G3 (Sprint 4): fused linear-CE FFI signatures.
            // Sprint v3-2 added trailing `dtype_tag` (0=F32 sentinel preserves
            // pre-v3-2 ABI; 1=F16). Sprint v4-1 extended the sentinel space
            // with 2=Bf16 (same single-i64 trailing arg — no ABI bump).
            // The Cranelift IR call sites in wengert_lower.rs derive the tag
            // from the @fused_lm_ce(dtype=...) decorator via
            // `fused_ce_dtype_for_compiler`. Note: v4-2 wengert refuses
            // tag != 0 pending precision_cast plumbing (see review Finding 2);
            // direct FFI tests with caller-managed 16-bit HBM allocation
            // exercise tags 1 and 2 end-to-end.
            // Forward v1 (small vocab, single CTA per row).
            [training] nsl_fused_linear_ce_forward(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = fused_linear_ce::nsl_fused_linear_ce_forward;
            // Forward large-vocab (Sprint 3 two-kernel path, vocab > 8192).
            [training] nsl_fused_linear_ce_forward_large(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = fused_linear_ce::nsl_fused_linear_ce_forward_large;
            // Backward (shared between v1 and large-vocab forward paths).
            [training] nsl_fused_linear_ce_backward(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = fused_linear_ce::nsl_fused_linear_ce_backward;
            // Sprint 2.5: GEMM-chunked fused linear-CE (production path for large
            // vocab and every biasless head). No PTX/kname/smem args — chunk
            // kernels are runtime constants, heavy math is cuBLAS. bias/dbias
            // pointers are the literal 0 when has_bias == 0.
            [training] nsl_fused_linear_ce_forward_gemm(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = fused_linear_ce::nsl_fused_linear_ce_forward_gemm;
            [training] nsl_fused_linear_ce_backward_gemm(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = fused_linear_ce::nsl_fused_linear_ce_backward_gemm;
            // CPKD: fused KL-CE distillation loss (forward + backward).
            //
            // ABI LOCK-STEP: these declarations, the call sites in
            // wengert_lower.rs (lower_fused_kl_ce_forward / _backward_extract),
            // and the runtime extern "C" fns in
            // crates/nsl-runtime/src/fused_kl_ce.rs must agree on arg count and
            // order BY HAND — there is no compile-time cross-check (see the
            // 16-vs-18-arg Tier-B lesson above nsl_flash_attention_backward).
            // Forward = 20 args; backward = 23 args.
            [training] nsl_fused_kl_ce_forward(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = fused_kl_ce::nsl_fused_kl_ce_forward;
            [training] nsl_fused_kl_ce_backward(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = fused_kl_ce::nsl_fused_kl_ce_backward;
            // --- M40b: Backward context for source-to-source AD (handle-based) ---
            [training] nsl_backward_ctx_new(i64) -> i64 = backward_context::nsl_backward_ctx_new;
            [training] nsl_backward_ctx_save(i64, i64, i64) -> i64 = backward_context::nsl_backward_ctx_save;
            [training] nsl_backward_ctx_load(i64, i64) -> i64 = backward_context::nsl_backward_ctx_load;
            [training] nsl_backward_ctx_free(i64) -> i64 = backward_context::nsl_backward_ctx_free;
            // --- M43: Gradient accumulation (ABI-fixed) ---
            [training] nsl_grad_accumulate_add(i64, i64, i64) -> i64 = zero::nsl_grad_accumulate_add;
            // (dst, src, num_elems)
            [training] nsl_grad_zero(i64, i64) -> i64 = zero::nsl_grad_zero;
            // (grad_ptr, num_elems)
            [training] nsl_grad_all_reduce(i64, i64) -> i64 = zero::nsl_grad_all_reduce;
            // Item 4 (2026-08-25): the resolved train/optimizer/scheduler record,
            // installed at TRAIN-BLOCK entry (per block, not per program — a module
            // can hold several train blocks). Joins the .optim sidecar as checkpoint
            // identity; see nsl-runtime/src/train_config_record.rs for the policy.
            [training] nsl_set_train_config_record(i64, i64) -> i64 = train_config_record::nsl_set_train_config_record;
            // D1 (CSLA Stage-2): window-backward anti-vacuity counter — mark once per
            // accumulation window at the head of the buffered backward phase, plus the
            // in-process getter for gates.
            [training] nsl_csla_window_mark() -> () = csla_stat::nsl_csla_window_mark;
            [training] nsl_csla_window_count() -> i64 = csla_stat::nsl_csla_window_count;
            // D1b: one-time pointer-tie guard over param_list (aborts loudly on any
            // aliased pair — per-layer in-place updates would corrupt the alias).
            [training] nsl_csla_assert_params_unaliased(i64) -> () = csla_stat::nsl_csla_assert_params_unaliased;
            // Fused-CE tape-carry gates: fused linear-CE launch counter (0 =
            // forward, 1 = forward_large, 2 = backward) — NSL-callable as
            // fused_lce_launch_count(k).
            [training] nsl_fused_lce_launch_count(i64) -> i64 = fused_linear_ce::nsl_fused_lce_launch_count;
            // Milestone B: weight-stream stat surface — NSL-callable as
            // weight_stream_stat(kind); kinds documented on the runtime fn.
            [training] nsl_weight_stream_stat(i64) -> i64 = weight_stream::nsl_weight_stream_stat;
            // Fused-CE targets dtype bridge: the kernels read targets as s64 but
            // NSL GPU labels are f32 — materialize/free a device i64 copy around
            // each fused forward/backward FFI.
            // Second arg: expected row count (decorator batch*seq) — the runtime
            // aborts loudly on mismatch instead of overreading the staging buffer.
            [training] nsl_fused_lce_targets_i64_alloc(i64, i64) -> i64 = fused_linear_ce::nsl_fused_lce_targets_i64_alloc;
            // (x_tensor, w_tensor, batch, seq, vocab_size, hidden_size, site_code)
            // -> void. Aborts when the decorator hints disagree with the head
            // tensors. batch and seq stay SEPARATE: collapsing them to rows here
            // would let a swapped pair through, and the backward builds dx from the
            // pair. site_code names the caller in the diagnostic (0 = @fused_lm_ce,
            // 1/2 = @fused_kl_ce student/teacher) so a refusal never blames the
            // wrong decorator or the wrong hint name.
            [training] nsl_fused_lce_pin_hint_extents(i64, i64, i64, i64, i64, i64, i64) -> () = fused_linear_ce::nsl_fused_lce_pin_hint_extents;
            [training] nsl_fused_lce_targets_i64_free(i64) -> () = fused_linear_ce::nsl_fused_lce_targets_i64_free;
            // D2b weight streaming: pointer-identity host offload of model params
            // (side-table mirrors; tensor pointers never change).
            [training] nsl_weight_stream_register(i64) -> () = weight_stream::nsl_weight_stream_register;
            [training] nsl_weight_stream_upload(i64) -> () = weight_stream::nsl_weight_stream_upload;
            [training] nsl_weight_stream_evict(i64, i64) -> () = weight_stream::nsl_weight_stream_evict;
            [training] nsl_weight_stream_upload_all() -> () = weight_stream::nsl_weight_stream_upload_all;
            // Item 12: re-evict everything after a scoped `upload_all` around a
            // model-touching callback. Arg = writeback (1 if the callback may mutate).
            [training] nsl_weight_stream_reevict_all(i64) -> () = weight_stream::nsl_weight_stream_reevict_all;
            // Item 10: contiguous layer-pack transfers. Arg = NslList of the pack's
            // param tensor pointers; evict also takes writeback.
            [training] nsl_weight_stream_upload_pack(i64) -> () = weight_stream::nsl_weight_stream_upload_pack;
            [training] nsl_weight_stream_evict_pack(i64, i64) -> () = weight_stream::nsl_weight_stream_evict_pack;
            // Item 11: async double-buffer prefetch + event-ordered await.
            [training] nsl_weight_stream_prefetch_pack(i64) -> () = weight_stream::nsl_weight_stream_prefetch_pack;
            [training] nsl_weight_stream_evict_pack_async(i64) -> () = weight_stream::nsl_weight_stream_evict_pack_async;
            [training] nsl_weight_stream_await_pack(i64) -> () = weight_stream::nsl_weight_stream_await_pack;
            [training] nsl_weight_stream_teardown() -> () = weight_stream::nsl_weight_stream_teardown;
            [training] nsl_weight_stream_upload_count() -> i64 = weight_stream::nsl_weight_stream_upload_count;
            [training] nsl_tensor_logsoftmax(i64, i64) -> i64 = tensor::ad_ops::nsl_tensor_logsoftmax;
            [training] nsl_tensor_repeat(i64, i64) -> i64 = tensor::ad_ops::nsl_tensor_repeat;
            [training] nsl_tensor_rope_inverse(i64, i64) -> i64 = tensor::ad_ops::nsl_tensor_rope_inverse;

            // ── optimizer (was runtime_abi/optimizer.rs) ──
            // P1 Muon items 8+10: planned Newton-Schulz orthogonalization primitive
            // (no .item() sync; device-resident norm; materialized tall/wide).
            [optimizer] nsl_tensor_muon_orthogonalize(i64, f64) -> i64 = muon::nsl_tensor_muon_orthogonalize;
            // Muon perf campaign: batched momentum + Newton-Schulz + param update
            // over (params, grads, m, routes) lists; f64 lr/momentum/wd, i64
            // nesterov flag, f64 ns_steps.
            [optimizer] nsl_muon_step_batch(i64, i64, i64, i64, f64, f64, f64, i64, f64) -> () = nsl_muon_step_batch;
            // Muon internal profiler (perf-campaign item 2): explicit begin/end
            // region markers + on-demand report, no-ops unless NSL_MUON_PROF is set.
            [optimizer] nsl_muon_prof_begin(i64) -> () = muon_prof::nsl_muon_prof_begin;
            [optimizer] nsl_muon_prof_end(i64) -> () = muon_prof::nsl_muon_prof_end;
            [optimizer] nsl_muon_prof_report() -> () = muon_prof::nsl_muon_prof_report;
            // Fusion item 1: multi-tensor fused AdamW final step over the whole
            // param/m/v/m_partial lists (one pointer-table launch, bit-identical).
            //
            // The two trailing I64s are the AdamW parameter groups (0 = no no_decay);
            // the final F64 is mp_scale, the two-phase-clip factor folded into the
            // m_partial read, 1.0 = unclipped (exact legacy behaviour, branched around
            // in-kernel). The two features are independent and BOTH are load-bearing —
            // this list, the emission order in `stmt.rs`, and the Rust signature in
            // `fase_step.rs` must stay in lockstep or arguments silently shift.
            [optimizer] nsl_fase_fused_adamw_step_multi(i64, i64, i64, i64, f64, f64, f64, f64, f64, f64, f64, f64, f64, i64, i64, f64) -> () = fase_step::nsl_fase_fused_adamw_step_multi;
            // Item 8, CSLA half: subset variant — same contract with an extra NslList
            // of i64 param indices (arg 5) selecting which parameters to step. The
            // CSLA layerwise window emits one call per layer group. Same lockstep
            // warning as above: this list, the `stmt.rs` emission order, and the Rust
            // signature in `fase_step.rs` must not drift.
            [optimizer] nsl_fase_fused_adamw_step_multi_idx(i64, i64, i64, i64, i64, f64, f64, f64, f64, f64, f64, f64, f64, f64, i64, i64, f64) -> () = fase_step::nsl_fase_fused_adamw_step_multi_idx;
            // Item 8, bf16-SR arm: the SR twin of the idx entry above. Same contract
            // minus mp_scale (the SR path has no clip fold; the layerwise schedule
            // refuses grad_clip) plus the trailing `step` the SR counter stream is
            // keyed on. No full-list twin exists: bf16-sr requires --weight-stream,
            // which requires --layerwise-accum, so SR only ever batches from the
            // CSLA group update. Same lockstep warning: this list, the `stmt.rs`
            // emission order, and the Rust signature in `sr_bf16.rs` must not drift.
            [optimizer] nsl_fase_fused_adamw_step_bf16sr_multi_idx(i64, i64, i64, i64, i64, f64, f64, f64, f64, f64, f64, f64, f64, f64, i64, i64, i64) -> () = sr_bf16::nsl_fase_fused_adamw_step_bf16sr_multi_idx;
            // FASE two-phase-clip Phase A: global sum-of-squares over an NslList of
            // m_partial tensors with ONE pipeline drain (batched device reduction).
            [optimizer] nsl_fase_sum_sq_list(i64) -> f64 = fase_step::nsl_fase_sum_sq_list;
            // FASE Deferred bias correction: 1/(1 - base^step).  Scalar, no tensor args.
            [optimizer] nsl_bias_correction_inv(f64, i64) -> f64 = fase_bc::nsl_bias_correction_inv;
            // AdamW parameter groups: resolve ONE param's weight decay from its
            // compile-time role flag + its runtime rank.
            // (param, static_exempt, exempt_non_rank2, wd) -> wd_for_this_param
            [optimizer] nsl_optim_param_wd(i64, i64, i64, f64) -> f64 = optim_groups::nsl_optim_param_wd;
            // FASE Deferred two-phase grad clip: sum of squared elements, in-place scale.
            [optimizer] nsl_tensor_sum_sq(i64) -> f64 = tensor::nsl_tensor_sum_sq;
            [optimizer] nsl_tensor_mul_scalar_inplace(i64, f64) -> () = tensor::nsl_tensor_mul_scalar_inplace;
            // Item 3: the compiled ParameterPlan, cross-checked against the three
            // residency tables at run time. LOCKSTEP: the flag bits are
            // `nsl_runtime::param_plan::PLAN_*`, re-exported by
            // `crate::parameter_plan` — there is no second copy to keep in sync.
            [optimizer] nsl_param_plan_declare(i64, i64, i64) -> i64 = param_plan::nsl_param_plan_declare;
            // (tensor, idx, flags)
            [optimizer] nsl_param_plan_verify() -> i64 = param_plan::nsl_param_plan_verify;
            [optimizer] nsl_param_plan_teardown() -> i64 = param_plan::nsl_param_plan_teardown;
            // P4 item 17: SR-BF16 authoritative weights
            [optimizer] nsl_sr_bf16_enable() -> () = sr_bf16::nsl_sr_bf16_enable;
            [optimizer] nsl_sr_bf16_note_param(i64, i64) -> () = sr_bf16::nsl_sr_bf16_note_param;
            // (tensor, idx)
            // P4 item 18 rung 2: (src_f32, dst_bf16, step, param_idx)
            [optimizer] nsl_muon_state_sr_store(i64, i64, i64, i64) -> () = sr_bf16::nsl_muon_state_sr_store;
            [optimizer] nsl_sr_bf16_step_adamw(i64, i64, i64, i64, f64, f64, f64, f64, f64, f64, f64, f64, f64, i64) -> () = sr_bf16::nsl_sr_bf16_step_adamw;
            [optimizer] nsl_zero3_reduce_grad_slot(i64, i64) -> i64 = zero::nsl_zero3_reduce_grad_slot;
            // (list, idx)
            // p9: fused per-param FASE-Deferred AdamW step — one launch for the whole
            // m/v/θ update. (theta, m, v, m_partial, lr, β₁, 1-β₁, β₂, 1-β₂, ε, wd,
            // bc1_inv, bc2_inv) → void.
            [optimizer] nsl_fase_fused_adamw_step(i64, i64, i64, i64, f64, f64, f64, f64, f64, f64, f64, f64, f64) -> () = fase_step::nsl_fase_fused_adamw_step;

            // ── distributed (was runtime_abi/distributed.rs) ──
            // --- M41: Disaggregated inference ---
            [distributed] nsl_disagg_init(i64, i64, i64, i64) -> i64 = disaggregated::router::nsl_disagg_init;
            [distributed] nsl_disagg_get_role() -> i64 = disaggregated::router::nsl_disagg_get_role;
            [distributed] nsl_disagg_get_rank() -> i64 = disaggregated::router::nsl_disagg_get_rank;
            [distributed] nsl_disagg_destroy() -> i64 = disaggregated::router::nsl_disagg_destroy;
            [distributed] nsl_disagg_worker_init(i64, i64, i64) -> i64 = disaggregated::worker::nsl_disagg_worker_init;
            [distributed] nsl_disagg_worker_destroy() -> i64 = disaggregated::worker::nsl_disagg_worker_destroy;
            [distributed] nsl_disagg_prefill_loop(i64) -> i64 = disaggregated::worker::nsl_disagg_prefill_loop;
            [distributed] nsl_disagg_decode_loop(i64) -> i64 = disaggregated::worker::nsl_disagg_decode_loop;
            // --- M41b: KV transfer backends ---
            [distributed] nsl_kv_transfer_init(i64, i64) -> i64 = disaggregated::kv_transfer::nsl_kv_transfer_init;
            [distributed] nsl_kv_transfer_send(i64, i64, i64, i64, i64) -> i64 = disaggregated::kv_transfer::nsl_kv_transfer_send;
            [distributed] nsl_kv_transfer_recv(i64, i64, i64, i64) -> i64 = disaggregated::kv_transfer::nsl_kv_transfer_recv;
            [distributed] nsl_kv_transfer_destroy() -> i64 = disaggregated::kv_transfer::nsl_kv_transfer_destroy;
            // --- M30: Tensor parallelism ---
            [distributed] nsl_tp_init() -> i64 = tensor_parallel::ffi::nsl_tp_init;
            [distributed] nsl_tp_rank() -> i64 = tensor_parallel::ffi::nsl_tp_rank;
            [distributed] nsl_tp_world_size() -> i64 = tensor_parallel::ffi::nsl_tp_world_size;
            [distributed] nsl_tp_all_reduce_sum(i64, i64, i64, i64, i64) -> i64 = tensor_parallel::ffi::nsl_tp_all_reduce_sum;
            [distributed] nsl_tp_all_gather(i64, i64, i64, i64, i64) -> i64 = tensor_parallel::ffi::nsl_tp_all_gather;
            [distributed] nsl_tp_broadcast(i64, i64, i64, i64, i64) -> i64 = tensor_parallel::ffi::nsl_tp_broadcast;
            [distributed] nsl_tp_barrier() -> i64 = tensor_parallel::ffi::nsl_tp_barrier;
            [distributed] nsl_tp_destroy() -> i64 = tensor_parallel::ffi::nsl_tp_destroy;
            // --- M43: Pipeline parallelism ---
            [distributed] nsl_pipeline_init(i64, i64, i64) -> i64 = pipeline::comm::nsl_pipeline_init;
            [distributed] nsl_pipeline_send(i64, i64, i64, i64) -> i64 = pipeline::comm::nsl_pipeline_send;
            [distributed] nsl_pipeline_recv(i64, i64, i64, i64, i64, i64) -> i64 = pipeline::comm::nsl_pipeline_recv;
            [distributed] nsl_pipeline_send_grad(i64, i64, i64, i64) -> i64 = pipeline::comm::nsl_pipeline_send_grad;
            [distributed] nsl_pipeline_recv_grad(i64, i64, i64, i64, i64, i64) -> i64 = pipeline::comm::nsl_pipeline_recv_grad;
            [distributed] nsl_pipeline_barrier() -> i64 = pipeline::comm::nsl_pipeline_barrier;
            [distributed] nsl_pipeline_destroy() -> i64 = pipeline::comm::nsl_pipeline_destroy;
            // --- M43: ZeRO optimizer (ABI-fixed: match runtime signatures exactly) ---
            [distributed] nsl_zero_init(i64, i64) -> i64 = zero::nsl_zero_init;
            // (stage, world_size)
            [distributed] nsl_zero_partition(i64) -> i64 = zero::nsl_zero_partition;
            // (num_params)
            [distributed] nsl_zero_partition_bytes(i64, i64) -> i64 = zero::nsl_zero_partition_bytes;
            // (param_list, num_params)
            [distributed] nsl_zero_reduce_grads(i64, i64) -> i64 = zero::nsl_zero_reduce_grads;
            // (grad_ptr, num_elems)
            [distributed] nsl_zero_step() -> i64 = zero::nsl_zero_step;
            // ()
            // D3 (ZeRO-1): post-step parameter sync — broadcast each param from
            // its owner rank (idx % world_size) so all ranks hold the full model.
            [distributed] nsl_zero_sync_params(i64, i64) -> i64 = zero::nsl_zero_sync_params;
            // (param_list, num_params)
            [distributed] nsl_zero_destroy() -> i64 = zero::nsl_zero_destroy;
            [distributed] nsl_zero_owns_param(i64) -> i64 = zero::nsl_zero_owns_param;
            // (param_idx) -> 1 if owned
            // (accum_list, num_params) -> NslList of owned indices; zeroes non-owners'
            // m_partial. Caller frees the list.
            [distributed] nsl_zero_owned_step_indices(i64, i64) -> i64 = zero::nsl_zero_owned_step_indices;
            // P3 ZeRO-3: tensor-granular parameter sharding (items 12-14).
            [distributed] nsl_zero3_enable() -> i64 = zero::nsl_zero3_enable;
            [distributed] nsl_zero3_note_param(i64, i64) -> i64 = zero::nsl_zero3_note_param;
            // (tensor, idx)
            // Item 11: elementwise 1/ws sharding (--zero-elementwise). Mark is
            // plan-driven and precedes the registration belt (the sr-note pattern);
            // the step runs on EVERY rank over its own slice — scalar order mirrors
            // nsl_fase_fused_adamw_step.
            // (tensor, idx, sr) — `sr` is the plan entry's storage decision (item 16x11)
            [distributed] nsl_zero3_mark_elementwise(i64, i64, i64) -> i64 = zero::nsl_zero3_mark_elementwise;
            // Owner-only moments: THIS RANK's slice-sized m/v for an elementwise
            // param. Sized from the carved ElemShard, so it MUST be emitted after
            // the weight-stream register belt (it aborts otherwise). Notes its own
            // element count against `optim_elems`.
            [distributed] nsl_zero3_alloc_elem_moment(i64, i64) -> i64 = zero::nsl_zero3_alloc_elem_moment;
            // (theta, idx)
            [distributed] nsl_zero3_elem_adamw_step(i64, i64, i64, i64, f64, f64, f64, f64, f64, f64, f64, f64, f64) -> i64 = zero::nsl_zero3_elem_adamw_step;
            // Item 16×11: the composed bf16-sr elementwise step — same scalar order
            // plus the trailing optimizer step the SR counter stream is keyed on.
            [distributed] nsl_zero3_elem_sr_adamw_step(i64, i64, i64, i64, f64, f64, f64, f64, f64, f64, f64, f64, f64, i64) -> i64 = zero::nsl_zero3_elem_sr_adamw_step;
            // D3 v2: record an owned optimizer-moment allocation's element count so the
            // G3 gate can prove per-rank optimizer state shrank to ~1/world_size.
            [distributed] nsl_zero_note_optim_alloc(i64) -> i64 = zero::nsl_zero_note_optim_alloc;
            // (tensor_ptr) -> running elems
            // (tensor_ptr) -> running REPLICA elems; the MomentFill::Full half that
            // nsl_zero_note_optim_alloc deliberately excludes.
            [distributed] nsl_zero_note_replicated_optim_alloc(i64) -> i64 = zero::nsl_zero_note_replicated_optim_alloc;
            // --- M34: Context parallelism (ring attention) ---
            // Extern signatures REMOVED across two cycles:
            //   * CPDT Part III v2.22 unlinked the ring FFI chain from codegen
            //     (`nsl_cp_init` / `nsl_sequence_partition` / `nsl_ring_attention`
            //     / `nsl_ring_send_recv` / `nsl_sequence_gather` / `nsl_cp_destroy`)
            //     and fell `@context_parallel` through to naive attention.
            //   * M34 v1 (this cycle) deleted the six runtime stubs themselves from
            //     `crates/nsl-runtime/src/context_parallel/ffi.rs` (they were dead
            //     symbols with wrong positional layouts) and shipped the
            //     single-node ring-attention composer
            //     (`run_ring_attention_full`) verified against `naive_attention`
            //     on a matrix of shapes and ring sizes. Multi-device distribution
            //     is deferred until NCCL/IPC lands.
            // When multi-device distribution lands, a fresh runtime FFI shape gets
            // designed and the extern table + emission + runtime impl all get
            // wired together against the new shape.

            // ── inference (was runtime_abi/inference.rs) ──
            // Standalone weight provider and arg parser (M24)
            [inference] nsl_standalone_init_embedded(i64, i64) -> () = weight_provider::nsl_standalone_init_embedded;
            [inference] nsl_standalone_init_sidecar(i64, i64) -> () = weight_provider::nsl_standalone_init_sidecar;
            [inference] nsl_standalone_has_weights() -> i64 = weight_provider::nsl_standalone_has_weights;
            [inference] nsl_standalone_args_init(i64, i64) -> () = weight_provider::nsl_standalone_args_init;
            [inference] nsl_standalone_arg_str(i64, i64) -> i64 = weight_provider::nsl_standalone_arg_str;
            [inference] nsl_standalone_arg_str_default(i64, i64, i64, i64) -> i64 = weight_provider::nsl_standalone_arg_str_default;
            [inference] nsl_standalone_arg_int(i64, i64) -> i64 = weight_provider::nsl_standalone_arg_int;
            [inference] nsl_standalone_arg_int_default(i64, i64, i64) -> i64 = weight_provider::nsl_standalone_arg_int_default;
            [inference] nsl_standalone_arg_float(i64, i64) -> i64 = weight_provider::nsl_standalone_arg_float;
            [inference] nsl_standalone_arg_float_default(i64, i64, i64) -> i64 = weight_provider::nsl_standalone_arg_float_default;
            [inference] nsl_standalone_args_finish() -> () = weight_provider::nsl_standalone_args_finish;
            // Paged KV-cache (M25)
            [inference] nsl_kv_cache_init(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = paged_kv::manager::nsl_kv_cache_init;
            [inference] nsl_kv_cache_init_gpu(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = paged_kv::manager::nsl_kv_cache_init_gpu;
            [inference] nsl_kv_cache_alloc_seq(i64) -> i64 = paged_kv::manager::nsl_kv_cache_alloc_seq;
            [inference] nsl_kv_cache_append(i64, i64) -> i64 = paged_kv::manager::nsl_kv_cache_append;
            [inference] nsl_kv_cache_k_ptr(i64, i64, i64) -> i64 = paged_kv::manager::nsl_kv_cache_k_ptr;
            [inference] nsl_kv_cache_v_ptr(i64, i64, i64) -> i64 = paged_kv::manager::nsl_kv_cache_v_ptr;
            [inference] nsl_kv_cache_free_seq(i64, i64) -> () = paged_kv::manager::nsl_kv_cache_free_seq;
            [inference] nsl_kv_cache_seq_len(i64, i64) -> i64 = paged_kv::manager::nsl_kv_cache_seq_len;
            [inference] nsl_kv_cache_seq_blocks(i64, i64) -> i64 = paged_kv::manager::nsl_kv_cache_seq_blocks;
            [inference] nsl_kv_cache_seq_num_blocks(i64, i64) -> i64 = paged_kv::manager::nsl_kv_cache_seq_num_blocks;
            [inference] nsl_kv_cache_utilization(i64) -> f64 = paged_kv::manager::nsl_kv_cache_utilization;
            [inference] nsl_kv_cache_destroy(i64) -> () = paged_kv::manager::nsl_kv_cache_destroy;
            // M29: Serving engine
            [inference] nsl_serve_init(i64, i64, i64, i64) -> i64 = serving::ffi::nsl_serve_init;
            [inference] nsl_serve_enqueue(i64, i64, i64, f64, f64) -> i64 = serving::ffi::nsl_serve_enqueue;
            [inference] nsl_serve_step() -> i64 = serving::ffi::nsl_serve_step;
            [inference] nsl_serve_record_token(i64, i64) -> i64 = serving::ffi::nsl_serve_record_token;
            [inference] nsl_serve_drain_completed() -> i64 = serving::ffi::nsl_serve_drain_completed;
            [inference] nsl_serve_has_work() -> i64 = serving::ffi::nsl_serve_has_work;
            [inference] nsl_serve_completed_count() -> i64 = serving::ffi::nsl_serve_completed_count;
            [inference] nsl_serve_preempt(i64) -> i64 = serving::ffi::nsl_serve_preempt;
            [inference] nsl_serve_destroy() -> i64 = serving::ffi::nsl_serve_destroy;
            // --- CFIE: continuous-batching request ring + grammar-table helper ---
            [inference] nsl_cfie_ring_init(i64) -> i64 = cfie::ffi::nsl_cfie_ring_init;
            [inference] nsl_cfie_ring_push(i64, i64, i64, i64, i64, i64) -> i64 = cfie::ffi::nsl_cfie_ring_push;
            [inference] nsl_cfie_ring_pop(i64, i64, i64, i64, i64, i64) -> i64 = cfie::ffi::nsl_cfie_ring_pop;
            [inference] nsl_cfie_ring_len() -> i64 = cfie::ffi::nsl_cfie_ring_len;
            [inference] nsl_cfie_grammar_transition(i64, i64, i64, i64, i64) -> i64 = cfie::ffi::nsl_cfie_grammar_transition;
            // --- CFIE: KV sequence-slot free-list ---
            [inference] nsl_cfie_kv_slots_init(i64, i64) -> i64 = cfie::ffi::nsl_cfie_kv_slots_init;
            [inference] nsl_cfie_kv_slot_acquire() -> i64 = cfie::ffi::nsl_cfie_kv_slot_acquire;
            [inference] nsl_cfie_kv_slot_release(i64) -> i64 = cfie::ffi::nsl_cfie_kv_slot_release;
            [inference] nsl_cfie_kv_slot_advance(i64, i64) -> i64 = cfie::ffi::nsl_cfie_kv_slot_advance;
            [inference] nsl_cfie_kv_slot_rollback(i64, i64) -> i64 = cfie::ffi::nsl_cfie_kv_slot_rollback;
            [inference] nsl_cfie_kv_slots_active() -> i64 = cfie::ffi::nsl_cfie_kv_slots_active;
            [inference] nsl_cfie_kv_attach_device(i64, i64) -> i64 = cfie::ffi::nsl_cfie_kv_attach_device;
            // --- CFIE Cycle 6: compiled-engine registration/lifecycle + launch
            // FFIs (frozen ABI).  All params/returns i64; f32 kernel params are
            // passed as f32::to_bits in the LOW 32 bits.  Kernel kinds:
            // 0=decode_attn, 1=fused_sample, 2=decode_block, 3=spec_verify,
            // 4=spec_reject, 5=quant_attn (layer_idx meaningful only for 5). ---
            [inference] nsl_cfie_register_kernel(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_register_kernel;
            [inference] nsl_cfie_kv_pool_alloc(i64) -> i64 = cfie::engine::nsl_cfie_kv_pool_alloc;
            // bytes
            [inference] nsl_cfie_engine_finalize() -> i64 = cfie::engine::nsl_cfie_engine_finalize;
            [inference] nsl_cfie_engine_destroy() -> i64 = cfie::engine::nsl_cfie_engine_destroy;
            // --- CFIE Cycle 9: runtime weight binding (production upload FFIs).
            // Cast host f32 [out][in] row-major -> device f16/f32, persistent
            // pool, engine-tracked; reset frees them. ---
            [inference] nsl_cfie_upload_weight_f16(i64, i64) -> i64 = cfie::engine::nsl_cfie_upload_weight_f16;
            [inference] nsl_cfie_upload_weight_f32(i64, i64) -> i64 = cfie::engine::nsl_cfie_upload_weight_f32;
            [inference] nsl_cfie_weights_reset() -> i64 = cfie::engine::nsl_cfie_weights_reset;
            [inference] nsl_cfie_launch_decode_attn(i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_launch_decode_attn;
            [inference] nsl_cfie_launch_fused_sample(i64, i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_launch_fused_sample;
            [inference] nsl_cfie_launch_decode_block(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_launch_decode_block;
            [inference] nsl_cfie_launch_spec_verify(i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_launch_spec_verify;
            [inference] nsl_cfie_launch_spec_reject(i64, i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_launch_spec_reject;
            [inference] nsl_cfie_launch_quant_attn(i64, i64, i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_launch_quant_attn;
            [inference] nsl_cfie_decode_step(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_decode_step;
            // --- CFIE Cycle 10: model binding + generation driver. bind_model
            // resolves an NslModel's host f32 weights by the HF-Llama name
            // convention, uploads them, and records the device weight table;
            // generate drives the decode loop over a prompt; generate_reset
            // clears the binding without freeing the weight buffers. ---
            [inference] nsl_cfie_bind_model(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_bind_model;
            [inference] nsl_cfie_generate(i64, i64, i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_generate;
            [inference] nsl_cfie_generate_reset() -> i64 = cfie::engine::nsl_cfie_generate_reset;
            // --- CFIE Cycle 12: host token-buffer <-> tokenizer-tensor bridge.
            // tokens_to_tensor turns generate's out-buffer into the 1-D f64
            // tensor nsl_tokenizer_decode consumes (text output); tensor_to_tokens
            // turns nsl_tokenizer_encode's tensor into generate's host i64 prompt
            // array (runtime-encoded prompt). ---
            [inference] nsl_cfie_tokens_to_tensor(i64, i64) -> i64 = cfie::bridge::nsl_cfie_tokens_to_tensor;
            [inference] nsl_cfie_tensor_to_tokens(i64, i64, i64) -> i64 = cfie::bridge::nsl_cfie_tensor_to_tokens;
            // --- CFIE Cycle 13 (G15 draft-model-in-binary): draft-model binding
            // + engine-held draft KV pool + speculative decode driver.  The
            // serve wiring emits bind_draft_model/draft_pool_alloc at serve init
            // (after the target bind) and speculative_generate from the
            // endpoint's generate() when the speculative draft is configured;
            // the launch FFIs are the kind-6/7/8 wrappers the driver uses
            // internally (registered for ABI completeness + direct testing). ---
            [inference] nsl_cfie_bind_draft_model(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_bind_draft_model;
            [inference] nsl_cfie_draft_pool_alloc(i64) -> i64 = cfie::engine::nsl_cfie_draft_pool_alloc;
            // bytes
            [inference] nsl_cfie_draft_reset() -> i64 = cfie::engine::nsl_cfie_draft_reset;
            [inference] nsl_cfie_launch_draft_block(i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_launch_draft_block;
            [inference] nsl_cfie_launch_draft_sample(i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_launch_draft_sample;
            [inference] nsl_cfie_launch_verify_probs(i64, i64) -> i64 = cfie::engine::nsl_cfie_launch_verify_probs;
            [inference] nsl_cfie_speculative_generate(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = cfie::engine::nsl_cfie_speculative_generate;
            // --- M32: MoE runtime functions ---
            [inference] nsl_moe_route(i64, i64, i64, i64, i64, i64) -> i64 = moe::ffi::nsl_moe_route;
            [inference] nsl_moe_scatter(i64, i64, i64, i64, i64, i64) -> i64 = moe::ffi::nsl_moe_scatter;
            [inference] nsl_expert_parallel_matmul(i64, i64, i64, i64, i64, i64, i64) -> i64 = moe::ffi::nsl_expert_parallel_matmul;
            [inference] nsl_moe_gather(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = moe::ffi::nsl_moe_gather;
            [inference] nsl_moe_all_to_all(i64, i64, i64, i64, i64, i64, i64) -> i64 = moe::ffi::nsl_moe_all_to_all;
            [inference] nsl_moe_aux_loss(i64, i64, i64, i64, i64, i64) -> i64 = moe::ffi::nsl_moe_aux_loss;
            [inference] nsl_moe_dispatch_full(i64, i64, i64, i64, i64) -> i64 = moe::ffi::nsl_moe_dispatch_full;
            // CPDT Part III v1 production-forward (M32 gap closure): same as v1
            // plus `experts_ptr`, `hidden_dim`, `intermediate_dim` (3 extra i64
            // args, total 8). Returns NslTensor `[total_tokens, intermediate_dim]`
            // (note: trailing dim differs from v1's `[total_tokens, hidden_dim]`
            // identity output). See crates/nsl-runtime/src/moe/ffi.rs.
            [inference] nsl_moe_dispatch_full_v2(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = moe::ffi::nsl_moe_dispatch_full_v2;
            // CPDT Part III v2.2 paper-faithful MoE FFN: per-expert kernel is
            // `up → SiLU → down` instead of v2's single matmul. 10 i64 args:
            // tokens, logits, experts_up, experts_down, num_experts, top_k,
            // capacity_factor_bits, hidden_dim, intermediate_dim, activation_kind,
            // experts_up_bias_ptr, experts_down_bias_ptr (v2.11: bias args are
            // nullable — pass 0 for no bias).
            // Returns NslTensor `[total_tokens, hidden_dim]` (back to hidden,
            // unlike v2's intermediate). See nsl-runtime/src/moe/ffi.rs.
            [inference] nsl_moe_dispatch_full_v3(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = moe::ffi::nsl_moe_dispatch_full_v3;
            // CPDT Part III v2.5+v2.8 Mixtral gated MoE FFN: per-expert kernel is
            // `gate_act(gate) * up → down` where gate_act is selected by
            // gate_activation_kind. 11 i64 args: tokens, logits, experts_gate,
            // experts_up, experts_down, num_experts, top_k, capacity_factor_bits,
            // hidden_dim, intermediate_dim, gate_activation_kind (v2.8: 1=SwiGLU,
            // 2=GeGLU, 3=ReGLU). Output shape matches v3
            // `[total_tokens, hidden_dim]`. See nsl-runtime/src/moe/ffi.rs.
            [inference] nsl_moe_dispatch_full_v4(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = moe::ffi::nsl_moe_dispatch_full_v4;
            // --- M33: Speculative decoding runtime functions ---
            [inference] nsl_speculative_draft(i64, i64, i64, i64, i64, i64, i64) -> i64 = speculative::ffi::nsl_speculative_draft;
            [inference] nsl_speculative_verify(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = speculative::ffi::nsl_speculative_verify;
            [inference] nsl_speculative_build_tree(i64, i64, i64, i64, i64) -> i64 = speculative::ffi::nsl_speculative_build_tree;
            [inference] nsl_speculative_verify_tree(i64, i64, i64, i64, i64, i64) -> i64 = speculative::ffi::nsl_speculative_verify_tree;
            [inference] nsl_page_branch(i64, i64) -> i64 = speculative::ffi::nsl_page_branch;
            [inference] nsl_page_cow_copy(i64, i64, i64) -> i64 = speculative::ffi::nsl_page_cow_copy;
            [inference] nsl_tree_attention(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = speculative::ffi::nsl_tree_attention;
            [inference] nsl_speculative_cleanup(i64, i64) -> i64 = speculative::ffi::nsl_speculative_cleanup;
            [inference] nsl_speculative_decode_step(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = speculative::ffi::nsl_speculative_decode_step;
            // --- M44: Constrained decoding (grammar FSM) ---
            [inference] nsl_grammar_init(i64, i64, i64) -> i64 = grammar::nsl_grammar_init;
            [inference] nsl_grammar_step(i64, i64) -> i64 = grammar::nsl_grammar_step;
            [inference] nsl_grammar_apply_mask(i64, i64) -> i64 = grammar::nsl_grammar_apply_mask;
            [inference] nsl_grammar_is_accept(i64) -> i64 = grammar::nsl_grammar_is_accept;
            [inference] nsl_grammar_start_state() -> i64 = grammar::nsl_grammar_start_state;
            [inference] nsl_grammar_destroy() -> i64 = grammar::nsl_grammar_destroy;
            // M44b: Constrained decoding serve integration
            [inference] nsl_serve_apply_grammar(i64, i64) -> i64 = serving::ffi::nsl_serve_apply_grammar;
            [inference] nsl_serve_advance_grammar(i64, i64) -> i64 = serving::ffi::nsl_serve_advance_grammar;
            [inference] nsl_serve_set_grammar(i64, i64) -> i64 = serving::ffi::nsl_serve_set_grammar;

            // ── quantization (was runtime_abi/quantization.rs) ──
            // CPDT §3.2: INT8 blockwise quantization (the headline 4× memory result)
            [quantization] nsl_tensor_quant_int8_blockwise(i64, i64) -> i64 = tensor::int8_blockwise::nsl_tensor_quant_int8_blockwise;
            [quantization] nsl_tensor_dequant_int8_blockwise(i64) -> i64 = tensor::int8_blockwise::nsl_tensor_dequant_int8_blockwise;
            // Quantization (M16)
            [quantization] nsl_qtensor_quantize(i64, i64, i64, i64, i64) -> i64 = quantize::nsl_qtensor_quantize;
            [quantization] nsl_qtensor_dequantize(i64) -> i64 = quantize::nsl_qtensor_dequantize;
            [quantization] nsl_qtensor_matmul_mixed(i64, i64) -> i64 = quantize::nsl_qtensor_matmul_mixed;
            [quantization] nsl_qtensor_free(i64) -> () = quantize::nsl_qtensor_free;
            [quantization] nsl_qtensor_addref(i64) -> () = quantize::nsl_qtensor_addref;
            [quantization] nsl_qtensor_release(i64) -> () = quantize::nsl_qtensor_release;
            [quantization] nsl_qtensor_dtype(i64) -> i64 = quantize::nsl_qtensor_dtype;
            [quantization] nsl_qtensor_shape(i64) -> i64 = quantize::nsl_qtensor_shape;
            // Custom dtype registry (M23)
            [quantization] nsl_register_custom_dtype(i64, i64, i64, i64, i64, i64, i64, i64) -> () = tensor::nsl_register_custom_dtype;
            [quantization] nsl_finalize_dtype_registry() -> () = tensor::nsl_finalize_dtype_registry;
            [quantization] nsl_tensor_to_custom_dtype(i64, i64) -> i64 = tensor::nsl_tensor_to_custom_dtype;
            [quantization] nsl_tensor_from_custom_dtype(i64) -> i64 = tensor::nsl_tensor_from_custom_dtype;
            // M42b: Quantized FlashAttention (KV-cache in INT8/FP8)
            [quantization] nsl_flash_attention_quantized(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = flash_attention::nsl_flash_attention_quantized;
            [quantization] nsl_rope_cache_write(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = flash_attention::nsl_rope_cache_write;
            // --- M35: FP8 compute ---
            [quantization] nsl_fp8_cast(i64, i64, f64) -> i64 = fp8::nsl_fp8_cast;
            [quantization] nsl_fp8_matmul(i64, i64, f64, f64) -> i64 = fp8::nsl_fp8_matmul;
            [quantization] nsl_fp8_matmul_training(i64, i64, i8) -> i64 = fp8::nsl_fp8_matmul_training;
            [quantization] nsl_fp8_compute_scale(i64, i64) -> f64 = fp8::nsl_fp8_compute_scale;
            [quantization] nsl_fp8_quantize_e5m2(i64, f64) -> i64 = fp8::nsl_fp8_quantize_e5m2;
            [quantization] nsl_fp8_gradient_scale(i64) -> f64 = fp8::nsl_fp8_gradient_scale;
            [quantization] nsl_fp8_cache_e5m2_ptx(i64, i64) -> () = fp8::nsl_fp8_cache_e5m2_ptx;
            [quantization] nsl_fp8_update_calibration(i64, i64, f64) -> f64 = fp8::nsl_fp8_update_calibration;
            // --- M35: AWQ 4-bit quantization ---
            [quantization] nsl_awq_quantize(i64, i64, i64) -> i64 = awq::nsl_awq_quantize;
            [quantization] nsl_awq_matmul(i64, i64, i64) -> i64 = awq::nsl_awq_matmul;
            [quantization] nsl_awq_free(i64) -> () = awq::nsl_awq_free;
            // AWQ calibration sidecar: apply per-channel scales to weight tensor before quantizing.
            // Signature: (weight_ptr, scales_ptr, scales_len, alpha) -> scaled_weight_ptr
            [quantization] nsl_awq_pre_scale_weight(i64, i64, i64, f64) -> i64 = awq::nsl_awq_pre_scale_weight;
            // --- M35: GPTQ quantization ---
            [quantization] nsl_gptq_quantize(i64, i64, i64, i64) -> i64 = gptq::nsl_gptq_quantize;
            [quantization] nsl_gptq_quantize_ext(i64, i64, i64, i64, i64, i64, i64) -> i64 = gptq::nsl_gptq_quantize_ext;
            [quantization] nsl_gptq_matmul(i64, i64, i64, i64) -> i64 = gptq::nsl_gptq_matmul;
            [quantization] nsl_gptq_free(i64) -> () = gptq::nsl_gptq_free;
            [quantization] nsl_gptq_hessian_init(i64) -> i64 = gptq::nsl_gptq_hessian_init;
            [quantization] nsl_gptq_hessian_add_batch(i64) -> i64 = gptq::nsl_gptq_hessian_add_batch;
            [quantization] nsl_gptq_hessian_finalize() -> i64 = gptq::nsl_gptq_hessian_finalize;
            // --- M42: KV-cache compression ---
            [quantization] nsl_kv_quantize_and_store(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = kv_compress::ffi::nsl_kv_quantize_and_store;
            [quantization] nsl_kv_sliding_window_init(i64, i64, i64) -> i64 = kv_compress::ffi::nsl_kv_sliding_window_init;
            [quantization] nsl_kv_sliding_window_check(i64, i64, i64, i64) -> i64 = kv_compress::ffi::nsl_kv_sliding_window_check;
            [quantization] nsl_kv_sliding_window_destroy() -> i64 = kv_compress::ffi::nsl_kv_sliding_window_destroy;
            [quantization] nsl_kv_h2o_init(i64, i64, i64) -> i64 = kv_compress::ffi::nsl_kv_h2o_init;
            [quantization] nsl_kv_h2o_accumulate(i64, i64, i64) -> i64 = kv_compress::ffi::nsl_kv_h2o_accumulate;
            [quantization] nsl_kv_h2o_check(i64, i64, i64, i64) -> i64 = kv_compress::ffi::nsl_kv_h2o_check;
            [quantization] nsl_kv_h2o_remove_sequence(i64) -> i64 = kv_compress::ffi::nsl_kv_h2o_remove_sequence;
            [quantization] nsl_kv_h2o_destroy() -> i64 = kv_compress::ffi::nsl_kv_h2o_destroy;
            [quantization] nsl_kv_compress_ratio(i64) -> i64 = kv_compress::ffi::nsl_kv_compress_ratio;

            // ── diagnostics (was runtime_abi/diagnostics.rs) ──
            // Training diagnostics (temporary)
            [diagnostics] nsl_debug_train_step(i64, i64, i64) -> () = tensor::nsl_debug_train_step;
            [diagnostics] nsl_debug_gpu_mem(i64) -> () = tensor::nsl_debug_gpu_mem;
            [diagnostics] nsl_gpu_drain_cache() -> () = tensor::nsl_gpu_drain_cache;
            [diagnostics] nsl_gpu_set_persistent_pool() -> () = tensor::nsl_gpu_set_persistent_pool;
            [diagnostics] nsl_gpu_set_transient_pool() -> () = tensor::nsl_gpu_set_transient_pool;
            // P0.1 per-surface VRAM accounting (tag values: caching_allocator::SurfaceTag)
            [diagnostics] nsl_gpu_set_alloc_surface(i8) -> () = tensor::nsl_gpu_set_alloc_surface;
            [diagnostics] nsl_gpu_get_alloc_surface() -> i8 = tensor::nsl_gpu_get_alloc_surface;
            // A1 unified accounting: numeric VRAM getters (peak / counts / per-surface)
            // + stable allocation identity. First in-process VRAM-peak API — gates and
            // WGGO read these instead of scraping NSL_MEMSTATS stderr.
            [diagnostics] nsl_gpu_peak_allocated_bytes() -> i64 = tensor::nsl_gpu_peak_allocated_bytes;
            [diagnostics] nsl_flash_bwd_det_routed_count() -> i64 = flash_attention::nsl_flash_bwd_det_routed_count;
            [diagnostics] nsl_gpu_cumulative_alloc_count() -> i64 = tensor::nsl_gpu_cumulative_alloc_count;
            [diagnostics] nsl_gpu_surface_peak_bytes(i8) -> i64 = tensor::nsl_gpu_surface_peak_bytes;
            [diagnostics] nsl_gpu_surface_at_peak_bytes(i8) -> i64 = tensor::nsl_gpu_surface_at_peak_bytes;
            [diagnostics] nsl_gpu_reset_mem_stats() -> () = tensor::nsl_gpu_reset_mem_stats;
            [diagnostics] nsl_gpu_set_alloc_identity(i32, i64) -> () = tensor::nsl_gpu_set_alloc_identity;
            [diagnostics] nsl_gpu_clear_alloc_identity() -> () = tensor::nsl_gpu_clear_alloc_identity;
            [diagnostics] nsl_debug_gpu_alloc_summary(i64) -> () = tensor::nsl_debug_gpu_alloc_summary;
            // Health monitor FFI (dev-tools phase 4)
            [diagnostics] nsl_health_record_loss(f64, i64) -> () = health::ffi::nsl_health_record_loss;
            [diagnostics] nsl_health_record_grad_norm(i64, i64, i32, f64) -> () = health::ffi::nsl_health_record_grad_norm;
            [diagnostics] nsl_health_record_weight_norm(i64, i64, f64, i8) -> () = health::ffi::nsl_health_record_weight_norm;
            [diagnostics] nsl_health_flush_snapshot(i64, i64) -> i32 = health::ffi::nsl_health_flush_snapshot;
            [diagnostics] nsl_health_set_flush_interval(i64) -> () = health::ffi::nsl_health_set_flush_interval;
            // Inspector FFI (dev-tools phase 5)
            [diagnostics] nsl_tensor_stats(i64, i64) -> i32 = inspect::stats_kernel::nsl_tensor_stats;
            [diagnostics] nsl_inspect_record_stats(i64, i64, i64, i64) -> i32 = inspect::ffi::nsl_inspect_record_stats;
            [diagnostics] nsl_inspect_dump_full(i64, i64, i64, i64) -> i32 = inspect::ffi::nsl_inspect_dump_full;
            [diagnostics] nsl_inspect_set_dir(i64, i64) -> () = inspect::ffi::nsl_inspect_set_dir;
            [diagnostics] nsl_health_get_last_loss() -> f64 = health::ffi::nsl_health_get_last_loss;
            [diagnostics] nsl_health_get_loss_ema() -> f64 = health::ffi::nsl_health_get_loss_ema;
            [diagnostics] nsl_health_get_loss_ema_slope() -> f64 = health::ffi::nsl_health_get_loss_ema_slope;
            [diagnostics] nsl_health_get_grad_norm_total() -> f64 = health::ffi::nsl_health_get_grad_norm_total;
            [diagnostics] nsl_health_get_nan_inf_count_window() -> i64 = health::ffi::nsl_health_get_nan_inf_count_window;
            // Timing and allocation tracking
            [diagnostics] nsl_clock() -> f64 = math::nsl_clock;
            // NSL_PHASE_TIMING train-block instrumentation (deferral-closure
            // 2026-07-14): device sync + per-phase wall-clock report lines.
            [diagnostics] nsl_cuda_device_synchronize() -> () = math::nsl_cuda_device_synchronize;
            [diagnostics] nsl_phase_fwd_bwd_report(f64, f64) -> () = math::nsl_phase_fwd_bwd_report;
            [diagnostics] nsl_phase_optim_report(f64) -> () = math::nsl_phase_optim_report;
            [diagnostics] nsl_alloc_reset() -> i64 = math::nsl_alloc_reset;
            [diagnostics] nsl_alloc_count() -> i64 = math::nsl_alloc_count;
            [diagnostics] nsl_alloc_bytes() -> i64 = math::nsl_alloc_bytes;
            [diagnostics] nsl_model_to_device(i64, i64, i64) -> () = tensor::nsl_model_to_device;
            // Memory profiler (M25)
            [diagnostics] nsl_profiler_start(i64) -> () = profiling::nsl_profiler_start;
            [diagnostics] nsl_profiler_stop() -> () = profiling::nsl_profiler_stop;
            [diagnostics] nsl_profiler_dump(i64, i64) -> () = profiling::nsl_profiler_dump;
            [diagnostics] nsl_profiler_peak() -> i64 = profiling::nsl_profiler_peak;
            // Dev Tools Phase 2, Task 5: kernel-launch profile hooks.
            // Emitted around every GPU `kernel { ... }` launch when codegen runs
            // with `profile_kernels` enabled. Take a single i32 kernel_id matching
            // the dense ids assigned by ManifestBuilder::reserve_id().
            [diagnostics] nsl_profile_kernel_begin(i32) -> () = profiler::ffi::nsl_profile_kernel_begin;
            [diagnostics] nsl_profile_kernel_end(i32) -> () = profiler::ffi::nsl_profile_kernel_end;
            // Kernel profiler (M26) — flush is NOT registered here (Rust-only atexit call)
            [diagnostics] nsl_kernel_profiler_start() -> () = kernel_profiler::nsl_kernel_profiler_start;
            [diagnostics] nsl_kernel_profiler_stop() -> () = kernel_profiler::nsl_kernel_profiler_stop;
            // Execution fingerprint: (ptr, len) of a .rodata k=v record naming the
            // compile flags that decide training arithmetic. Installed before user
            // code so a checkpoint written later carries it.
            [diagnostics] nsl_set_exec_fingerprint(i64, i64) -> i64 = exec_fingerprint::nsl_set_exec_fingerprint;
            // --- M45: Tensor debugger trace ---
            [diagnostics] nsl_trace_init() -> i64 = tensor_trace::nsl_trace_init;
            [diagnostics] nsl_trace_record_op(i64, i64, i64, i64) -> i64 = tensor_trace::nsl_trace_record_op;
            [diagnostics] nsl_trace_suppress() -> i64 = tensor_trace::nsl_trace_suppress;
            [diagnostics] nsl_trace_unsuppress() -> i64 = tensor_trace::nsl_trace_unsuppress;
            [diagnostics] nsl_trace_breakpoint() -> i64 = tensor_trace::nsl_trace_breakpoint;
            [diagnostics] nsl_trace_flush() -> i64 = tensor_trace::nsl_trace_flush;
            [diagnostics] nsl_trace_destroy() -> i64 = tensor_trace::nsl_trace_destroy;
            [diagnostics] nsl_trace_nan_warning(i64, i64) -> i64 = tensor_trace::nsl_trace_nan_warning;

            // ── abi_memory (was runtime_abi/memory.rs) ──
            // Milestone C p2 Stage-2B: the placed transient arena. `bind` arms a
            // single-shot, size-exact pin that the device allocator consumes; `unbind`
            // disarms it so an op that took a non-allocating path cannot leak the pin
            // into an unrelated allocation.
            [abi_memory] nsl_arena_init(i64, i64) -> i64 = transient_arena::nsl_arena_init;
            [abi_memory] nsl_arena_bind(i64, i64, i64) -> () = transient_arena::nsl_arena_bind;
            [abi_memory] nsl_arena_unbind() -> () = transient_arena::nsl_arena_unbind;
            [abi_memory] nsl_arena_unbind_verify(i64) -> () = transient_arena::nsl_arena_unbind_verify;
            [abi_memory] nsl_arena_declare_slot(i64, i64) -> () = transient_arena::nsl_arena_declare_slot;
            [abi_memory] nsl_arena_check() -> i64 = transient_arena::nsl_arena_check;
            [abi_memory] nsl_arena_check_step(i64) -> () = transient_arena::nsl_arena_check_step;
            [abi_memory] nsl_arena_destroy() -> () = transient_arena::nsl_arena_destroy;
            // M36: GPU memory slab (compile-time planned device memory arena)
            [abi_memory] nsl_gpu_slab_init(i64) -> i64 = slab::nsl_gpu_slab_init;
            [abi_memory] nsl_slab_offset(i64, i64) -> i64 = slab::nsl_slab_offset;
            [abi_memory] nsl_gpu_slab_destroy() -> () = slab::nsl_gpu_slab_destroy;
            [abi_memory] nsl_gpu_slab_active() -> i64 = slab::nsl_gpu_slab_active;
            [abi_memory] nsl_tensor_from_slab(i64, i64, i64, i64) -> i64 = tensor::nsl_tensor_from_slab;
            // P5 item 19: opportunistic per-region CUDA graph capture/replay
            [abi_memory] nsl_cuda_graphs_enable(i64) -> () = cuda::graph_capture::nsl_cuda_graphs_enable;
            // (accum_window)
            [abi_memory] nsl_cuda_graph_region_begin(i64) -> () = cuda::graph_capture::nsl_cuda_graph_region_begin;
            // (region_id)
            [abi_memory] nsl_cuda_graph_region_end(i64) -> () = cuda::graph_capture::nsl_cuda_graph_region_end;
            // (region_id)
            [abi_memory] nsl_cuda_graphs_report() -> () = cuda::graph_capture::nsl_cuda_graphs_report;

            // ── interop (was runtime_abi/interop.rs) ──
            // Tokenizer functions (M15)
            [interop] nsl_byte_tokenizer_new() -> i64 = tokenizer::nsl_byte_tokenizer_new;
            [interop] nsl_bpe_train(i64, i64, i64, i64) -> i64 = tokenizer::nsl_bpe_train;
            [interop] nsl_tokenizer_load(i64) -> i64 = tokenizer::nsl_tokenizer_load;
            [interop] nsl_tokenizer_save(i64, i64) -> () = tokenizer::nsl_tokenizer_save;
            [interop] nsl_tokenizer_encode(i64, i64) -> i64 = tokenizer::nsl_tokenizer_encode;
            [interop] nsl_tokenizer_decode(i64, i64) -> i64 = tokenizer::nsl_tokenizer_decode;
            [interop] nsl_tokenizer_vocab_size(i64) -> i64 = tokenizer::nsl_tokenizer_vocab_size;
            [interop] nsl_tokenizer_encode_batch(i64, i64, i8, i8, i64) -> i64 = tokenizer::nsl_tokenizer_encode_batch;
            // GPU runtime functions (M17)
            [interop] nsl_cuda_init() -> i64 = cuda::nsl_cuda_init;
            [interop] nsl_kernel_launch(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = cuda::nsl_kernel_launch;
            // User `kernel` block launch: args array holds NslTensor handles; the
            // runtime extracts each `.data` device pointer and builds the kernelParams.
            [interop] nsl_kernel_launch_tensors(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 = cuda::nsl_kernel_launch_tensors;
            [interop] nsl_tensor_to_device(i64, i64) -> i64 = tensor::nsl_tensor_to_device;
            [interop] nsl_tensor_to_device_like(i64, i64) -> i64 = tensor::nsl_tensor_to_device_like;
            // Safetensors I/O (M18b)
            [interop] nsl_safetensors_load(i64, i64, i64) -> i64 = safetensors_io::nsl_safetensors_load [interop];
            [interop] nsl_safetensors_save(i64, i64, i64) -> () = safetensors_io::nsl_safetensors_save [interop];
            // HuggingFace Hub download + weight loading (M18b)
            [interop] nsl_hf_load(i64, i64, i64, i64, i64, i64) -> i64 = huggingface::nsl_hf_load [interop];
            // Trace infrastructure for ONNX export (M18b Task 7)
            [interop] nsl_trace_start() -> () = trace::nsl_trace_start [interop];
            [interop] nsl_trace_register_input(i64, i64) -> () = trace::nsl_trace_register_input [interop];
            [interop] nsl_trace_register_output(i64, i64) -> () = trace::nsl_trace_register_output [interop];
            [interop] nsl_trace_stop() -> i64 = trace::nsl_trace_stop [interop];
            // ONNX export (M18b Tasks 9-10)
            [interop] nsl_onnx_export(i64, i64, i64) -> () = onnx::nsl_onnx_export [interop];
            // --- M48: Multimodal primitives ---
            [interop] nsl_patch_embed(i64, i64, i64) -> i64 = multimodal::nsl_patch_embed;
            [interop] nsl_mel_spectrogram(i64, i64, i64, i64) -> i64 = multimodal::nsl_mel_spectrogram;
            // Explicit-sample-rate variant; the 4-arg form assumes 16 kHz.
            [interop] nsl_mel_spectrogram_sr(i64, i64, i64, i64, i64) -> i64 = multimodal::nsl_mel_spectrogram_sr;
            [interop] nsl_cross_attention(i64, i64, i64, i64) -> i64 = multimodal::nsl_cross_attention;
            [interop] nsl_image_resize(i64, i64, i64) -> i64 = multimodal::nsl_image_resize;
            [interop] nsl_image_normalize(i64, i64, i64) -> i64 = multimodal::nsl_image_normalize;
            [interop] nsl_stft(i64, i64, i64) -> i64 = multimodal::nsl_stft;
            [interop] nsl_audio_resample(i64, i64, i64) -> i64 = multimodal::nsl_audio_resample;
            // --- M62: Legacy Interop — DLPack bridge + C API ---
            [interop] nsl_dlpack_export(i64) -> i64 = dlpack::nsl_dlpack_export;
            [interop] nsl_dlpack_import(i64) -> i64 = dlpack::nsl_dlpack_import;
            [interop] nsl_dlpack_free(i64) -> () = dlpack::nsl_dlpack_free;
            [interop] nsl_model_create(i64) -> i64 = c_api::nsl_model_create;
            [interop] nsl_model_destroy(i64) -> i64 = c_api::nsl_model_destroy;
            [interop] nsl_model_forward(i64, i64, i64, i64, i64) -> i64 = c_api::nsl_model_forward;
            [interop] nsl_model_forward_dlpack(i64, i64, i64, i64, i64) -> i64 = c_api::nsl_model_forward_dlpack;
            [interop] nsl_model_backward(i64, i64, i64, i64, i64) -> i64 = grad_context::nsl_model_backward;
            [interop] nsl_model_get_version() -> i64 = c_api::nsl_model_get_version;
            [interop] nsl_get_last_error() -> i64 = c_api::nsl_get_last_error;
            [interop] nsl_clear_error() -> i64 = c_api::nsl_clear_error;
            // --- M54b: Unikernel runtime ---
            [interop] nsl_unikernel_init(i64, i64) -> i64 = unikernel::nsl_unikernel_init;
            [interop] nsl_unikernel_model_alloc(i64, i64) -> i64 = unikernel::nsl_unikernel_model_alloc;
            [interop] nsl_unikernel_kv_alloc(i64, i64) -> i64 = unikernel::nsl_unikernel_kv_alloc;
            [interop] nsl_unikernel_model_pool_stats() -> i64 = unikernel::nsl_unikernel_model_pool_stats;
            [interop] nsl_unikernel_shutdown() -> i64 = unikernel::nsl_unikernel_shutdown;
            [interop] nsl_unikernel_gpu_init(i64) -> i64 = unikernel::gpu_init::nsl_unikernel_gpu_init;
            [interop] nsl_unikernel_gpu_ready() -> i64 = unikernel::gpu_init::nsl_unikernel_gpu_ready;
            [interop] nsl_unikernel_gpu_device_id() -> i64 = unikernel::gpu_init::nsl_unikernel_gpu_device_id;
            // --- M56 v1 agent runtime FFI (Task 16). Signatures from spec §3.4. ---
            // All raw pointers are I64 per the workspace convention; time: u64 is also I64.
            [interop] nsl_agent_pool_new(i64, i64) -> i64 = agent::ffi::nsl_agent_pool_new;
            [interop] nsl_agent_pool_destroy(i64) -> () = agent::ffi::nsl_agent_pool_destroy;
            [interop] nsl_agent_pool_acquire(i64, i64) -> i64 = agent::ffi::nsl_agent_pool_acquire;
            [interop] nsl_agent_pool_release(i64, i64) -> () = agent::ffi::nsl_agent_pool_release;
            [interop] nsl_agent_scheduler_step(i64) -> i32 = agent::ffi::nsl_agent_scheduler_step;
            [interop] nsl_agent_mailbox_write(i64, i64, i64) -> i32 = agent::ffi::nsl_agent_mailbox_write;
            [interop] nsl_agent_mailbox_read(i64) -> i64 = agent::ffi::nsl_agent_mailbox_read;

        }
    };
}

for_each_runtime_fn!(__decl_runtime_abi);

/// Look a row up by symbol name.
pub fn lookup(name: &str) -> Option<&'static FnDecl> {
    RUNTIME_ABI.iter().find(|d| d.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn names_are_unique_and_prefixed() {
        let mut seen = BTreeSet::new();
        for d in RUNTIME_ABI {
            assert!(d.name.starts_with("nsl_"), "{}", d.name);
            assert!(seen.insert(d.name), "duplicate row: {}", d.name);
            assert_eq!(d.path.last().copied(), Some(d.name), "path must end in the symbol: {}", d.name);
        }
    }

    #[test]
    fn interop_rows_live_in_interop_modules() {
        const INTEROP: &[&str] = &["safetensors_io", "huggingface", "weight_map", "trace", "onnx", "onnx_proto"];
        for d in RUNTIME_ABI {
            let in_interop = INTEROP.contains(&d.path[0]);
            assert_eq!(d.interop, in_interop, "{}: [interop] flag must match its module", d.name);
        }
    }

    #[test]
    fn table_is_the_recorded_size() {
        // The count is pinned so a row dropped by a bad merge is noticed; move
        // it with a row that is deliberately added or removed.
        assert_eq!(RUNTIME_ABI.len(), 682);
    }
}
