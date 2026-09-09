//! The runtime-function registry: every `extern "C"` symbol the codegen can
//! emit a call to, with its Cranelift signature.
//!
//! # Where the declarations live
//!
//! The registry is rendered from the ABI table in `crates/nsl-abi/src/table.rs`
//! (roadmap A3): `nsl_abi::for_each_runtime_fn!` hands
//! [`render_runtime_functions`] every row, and each becomes one
//! `(name, params, ret)` of [`RUNTIME_FUNCTIONS`]. The same table renders,
//! in `nsl-runtime`, into a compile-time check that each implementation has
//! that signature, so the two sides cannot drift — the fourteen hand-written
//! `RUNTIME_FUNCTIONS*` tables this module used to keep (split by PR #600
//! along "what the language exposes vs what the runtime implements") are the
//! table's groups now.
//!
//! What stays here is the lowering: `builtins/{memory,io,scalar,collections,
//! tensor}.rs` hold the code that emits calls to the language-facing
//! functions, and [`declare_runtime_functions`] declares every row as an
//! import.
//!
//! # Declaration order is not load-bearing
//!
//! [`declare_runtime_functions`] walks the rows in table order, so `FuncId`s
//! are assigned in that order. That order does NOT reach the emitted CLIF: it
//! names its runtime callees (`fn0 = nsl_alloc`) and numbers funcrefs per
//! function by order of first use, so the global `FuncId` never appears.
//! Verified by moving `nsl_alloc` from index 6 to index 0 before the split:
//! all 26 `train_clif_snapshots` stayed byte-identical. That is what makes
//! regrouping the table free, and why a row may be moved between groups on
//! the strength of where it belongs.
//!
//! It does reach the OBJECT FILE — `cranelift-object` calls `add_symbol`
//! eagerly from `declare_function`, `Linkage::Import` included, so the `.o`
//! symbol table follows table order. Nothing checked in depends on that today.
//!
//! # Adding a runtime function
//!
//! Implement the `extern "C" fn` in `nsl-runtime`, then add one row to the
//! table in `nsl-abi` (the grammar and the groups are in its module docs).
//! Nothing here changes: this file declares whatever the table says. A row
//! must appear once — `no_runtime_function_is_declared_twice` checks the
//! rendering here and `nsl-abi`'s own tests check the table — and its
//! signature must match the implementation, which the runtime's build now
//! enforces (`abi_check.rs`) and `nsl-abi`'s `signature_agreement` gate
//! cross-checks by text as belt-and-braces.

use cranelift_codegen::ir::{types, AbiParam, Signature};
use cranelift_codegen::isa::CallConv;
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::ObjectModule;
use std::collections::HashMap;

use crate::error::CodegenError;

pub(crate) mod collections;
pub(crate) mod io;
pub(crate) mod memory;
pub(crate) mod scalar;
pub(crate) mod tensor;

/// Runtime function info: (name, params, returns).
type RuntimeFn = (&'static str, &'static [types::Type], Option<types::Type>);

/// The row scalars of the ABI table as Cranelift types.
macro_rules! cranelift_type {
    (i64) => { types::I64 };
    (i32) => { types::I32 };
    (i16) => { types::I16 };
    (i8) => { types::I8 };
    (f64) => { types::F64 };
    (f32) => { types::F32 };
}

macro_rules! cranelift_ret {
    (()) => { None };
    ($t:ident) => { Some(cranelift_type!($t)) };
}

/// One `(name, params, ret)` per row of `nsl_abi::for_each_runtime_fn!`,
/// in row order. This is the registry: every runtime function the codegen
/// can emit a call to, rendered from the one table in
/// `crates/nsl-abi/src/table.rs` (roadmap A3).
macro_rules! render_runtime_functions {
    ($([$g:ident] $n:ident ($($p:ident),*) -> $r:tt = $($seg:ident)::+ $([$f:ident])? ;)*) => {
        const RUNTIME_FUNCTIONS: &[RuntimeFn] = &[
            $((stringify!($n), &[$(cranelift_type!($p)),*], cranelift_ret!($r)),)*
        ];
    };
}

nsl_abi::for_each_runtime_fn!(render_runtime_functions);

/// Every runtime function, in table order.
pub(crate) fn all_runtime_functions() -> impl Iterator<Item = &'static RuntimeFn> {
    RUNTIME_FUNCTIONS.iter()
}

/// Declare all runtime functions as imports in the module.
pub fn declare_runtime_functions(
    module: &mut ObjectModule,
    call_conv: CallConv,
) -> Result<HashMap<String, (FuncId, Signature)>, CodegenError> {
    let mut fns = HashMap::new();

    for &(name, params, ret) in all_runtime_functions() {
        let mut sig = module.make_signature();
        sig.call_conv = call_conv;
        for &p in params {
            sig.params.push(AbiParam::new(p));
        }
        if let Some(r) = ret {
            sig.returns.push(AbiParam::new(r));
        }

        let func_id = module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| {
                CodegenError::new(format!("failed to declare runtime fn '{name}': {e}"))
            })?;

        fns.insert(name.to_string(), (func_id, sig));
    }

    // CSHA cycle 19 T1 (variant-B): register the new probe FFI symbol behind
    // the `csha_cycle19_probe` feature. Signature = 54 original i64 params
    // (byte-identical to `nsl_flash_attention_csha_backward`) + 2 trailing
    // i64 probe pointers = 56. Non-default; wired only by c19 probe tests.
    // See `docs/superpowers` c19 T1 spec + project_csha_paper_completion_cycle18.md.
    #[cfg(feature = "csha_cycle19_probe")]
    {
        let mut sig = module.make_signature();
        sig.call_conv = call_conv;
        for _ in 0..56 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));

        let func_id = module
            .declare_function(
                "nsl_flash_attention_csha_backward_probe",
                Linkage::Import,
                &sig,
            )
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to declare runtime fn 'nsl_flash_attention_csha_backward_probe': {e}"
                ))
            })?;

        fns.insert(
            "nsl_flash_attention_csha_backward_probe".to_string(),
            (func_id, sig),
        );
    }

    Ok(fns)
}

#[cfg(test)]
mod tests {
    use super::{all_runtime_functions, RuntimeFn};
    use cranelift_codegen::ir::types;

    /// The registry IS the ABI table: every row of `nsl_abi::RUNTIME_ABI` is
    /// declared here with the same signature, in the same order, and nothing
    /// else is. Both are renderings of one macro, so they cannot drift apart;
    /// what this pins is the rendering itself — that every scalar spelling
    /// maps to the right Cranelift type and every row is read (a macro arm
    /// that silently matched nothing would surface here as a length gap).
    #[test]
    fn registry_is_the_abi_table() {
        use nsl_abi::AbiScalar;

        fn scalar(t: types::Type) -> AbiScalar {
            match t {
                types::I8 => AbiScalar::Int(8),
                types::I16 => AbiScalar::Int(16),
                types::I32 => AbiScalar::Int(32),
                types::I64 => AbiScalar::Int(64),
                types::F32 => AbiScalar::Float(32),
                types::F64 => AbiScalar::Float(64),
                other => panic!("unmapped Cranelift type {other}"),
            }
        }

        let rendered: Vec<&RuntimeFn> = all_runtime_functions().collect();
        assert_eq!(rendered.len(), nsl_abi::RUNTIME_ABI.len(), "row count");
        for (r, d) in rendered.iter().zip(nsl_abi::RUNTIME_ABI) {
            assert_eq!(r.0, d.name, "row order");
            let params: Vec<AbiScalar> = r.1.iter().map(|t| scalar(*t)).collect();
            assert_eq!(params, d.params, "{}: params", d.name);
            assert_eq!(r.2.map(scalar), d.ret, "{}: return", d.name);
        }
    }

    /// No runtime function may be declared twice.
    ///
    /// `declare_runtime_functions` calls `Module::declare_function` once per
    /// entry; Cranelift accepts a repeat declaration when the signature is
    /// identical, so a duplicate is invisible at build time — and a duplicate
    /// whose signatures DISAGREE surfaces far from the edit that caused it.
    /// This keeps the table a set, which is what makes it safe to regroup.
    ///
    /// Declaration ORDER, by contrast, does not reach the CLIF: emitted CLIF
    /// names its runtime callees (`fn0 = nsl_alloc`) and numbers funcrefs per
    /// function by order of first use, so the global `FuncId` never appears.
    /// Verified by moving `nsl_alloc` — a callee the snapshots do reference —
    /// from index 6 to index 0: all 26 `train_clif_snapshots` stayed
    /// byte-identical. The table may therefore be regrouped by domain freely.
    ///
    /// It does reach the OBJECT FILE, though: `cranelift-object` calls
    /// `add_symbol` eagerly from `declare_function`, including for
    /// `Linkage::Import`, so every runtime symbol lands in the emitted `.o`
    /// symbol table in table order whether it is referenced or not. Nothing
    /// checked in depends on that today — no `.o`/`.a` goldens exist — but a
    /// future byte-identity gate over object output would need regenerating
    /// after a regrouping.
    #[test]
    fn no_runtime_function_is_declared_twice() {
        let mut seen = std::collections::BTreeMap::<&str, usize>::new();
        for (name, _, _) in all_runtime_functions() {
            *seen.entry(*name).or_default() += 1;
        }
        let dupes: Vec<String> = seen
            .iter()
            .filter(|(_, n)| **n > 1)
            .map(|(name, n)| format!("{name} ({n}x)"))
            .collect();
        assert!(
            dupes.is_empty(),
            "runtime function(s) declared more than once: {}",
            dupes.join(", ")
        );
    }

    #[test]
    fn precision_cast_ops_have_signatures() {
        let names: Vec<&str> = all_runtime_functions().map(|(n, _, _)| *n).collect();
        assert!(names.contains(&"nsl_tensor_cast"), "nsl_tensor_cast missing");
        assert!(names.contains(&"nsl_tensor_cast_into"), "nsl_tensor_cast_into missing");
        assert!(
            names.contains(&"nsl_tensor_zeros_like_dtype"),
            "nsl_tensor_zeros_like_dtype missing"
        );
    }

    #[test]
    fn int8_blockwise_ops_have_signatures() {
        // CPDT §3.2 — the headline 4× memory result. These signatures must
        // match the runtime exports in nsl-runtime/src/tensor/int8_blockwise.rs
        // and the ownership table in ffi_ownership.rs (both produce new owned
        // tensors).
        let table: Vec<(&str, &[cranelift_codegen::ir::Type], Option<cranelift_codegen::ir::Type>)> =
            all_runtime_functions()
                .filter(|(n, _, _)| {
                    *n == "nsl_tensor_quant_int8_blockwise"
                        || *n == "nsl_tensor_dequant_int8_blockwise"
                })
                .map(|(n, p, r)| (*n, *p, *r))
                .collect();
        assert_eq!(table.len(), 2, "INT8 blockwise op pair missing");
        for (name, params, ret) in &table {
            assert_eq!(*ret, Some(cranelift_codegen::ir::types::I64), "{name} must return i64");
            assert!(params.iter().all(|t| *t == cranelift_codegen::ir::types::I64),
                "{name} params must all be I64");
        }
    }

    /// CFTP v6: forward inline-cast wrapper FFIs are registered with the
    /// correct Cranelift signature ([I64] -> I64). Required so wengert_lower
    /// can emit calls to them from compiled NSL.
    #[test]
    fn cftp_v6_cast_wrappers_have_signatures() {
        use cranelift_codegen::ir::types;
        for &name in &["nsl_tensor_to_bf16", "nsl_tensor_to_fp16", "nsl_tensor_to_f32"] {
            let entry = all_runtime_functions()
                .find(|(n, _, _)| *n == name)
                .unwrap_or_else(|| panic!("{name} missing from the runtime registry"));
            assert_eq!(
                entry.1,
                &[types::I64],
                "{name}: expected params [I64], got {:?}",
                entry.1
            );
            assert_eq!(
                entry.2,
                Some(types::I64),
                "{name}: expected return I64, got {:?}",
                entry.2
            );
        }
    }

    /// CFIE Cycle 6: the engine registration/lifecycle + launch FFIs
    /// are declared with the frozen ABI's arities — all-i64 params,
    /// i64 return — so `declare_runtime_functions` picks them up and
    /// the serve emission can `compile_call_by_name` them.
    #[test]
    fn cfie_cycle6_engine_ffis_have_frozen_abi_signatures() {
        use cranelift_codegen::ir::types;
        let arity = |name: &str| -> usize {
            let entry = all_runtime_functions()
                .find(|(n, _, _)| *n == name)
                .unwrap_or_else(|| panic!("{name} missing from the runtime registry"));
            assert!(
                entry.1.iter().all(|&t| t == types::I64),
                "{name}: every param must be I64 (frozen ABI), got {:?}",
                entry.1
            );
            assert_eq!(
                entry.2,
                Some(types::I64),
                "{name}: must return I64, got {:?}",
                entry.2
            );
            entry.1.len()
        };
        assert_eq!(arity("nsl_cfie_register_kernel"), 9);
        assert_eq!(arity("nsl_cfie_kv_pool_alloc"), 1);
        assert_eq!(arity("nsl_cfie_engine_finalize"), 0);
        assert_eq!(arity("nsl_cfie_engine_destroy"), 0);
        assert_eq!(arity("nsl_cfie_upload_weight_f16"), 2);
        assert_eq!(arity("nsl_cfie_upload_weight_f32"), 2);
        assert_eq!(arity("nsl_cfie_weights_reset"), 0);
        assert_eq!(arity("nsl_cfie_launch_decode_attn"), 5);
        assert_eq!(arity("nsl_cfie_launch_fused_sample"), 6);
        assert_eq!(arity("nsl_cfie_launch_decode_block"), 14);
        assert_eq!(arity("nsl_cfie_launch_spec_verify"), 5);
        assert_eq!(arity("nsl_cfie_launch_spec_reject"), 6);
        assert_eq!(arity("nsl_cfie_launch_quant_attn"), 7);
        assert_eq!(arity("nsl_cfie_decode_step"), 11);
        assert_eq!(arity("nsl_cfie_bind_model"), 8);
        assert_eq!(arity("nsl_cfie_generate"), 7);
        assert_eq!(arity("nsl_cfie_generate_reset"), 0);
        assert_eq!(arity("nsl_cfie_tokens_to_tensor"), 2);
        assert_eq!(arity("nsl_cfie_tensor_to_tokens"), 3);
        // CFIE Cycle 13 (G15): draft binding + pool + launch trio +
        // the speculative decode driver — arity-pinned against the
        // frozen all-i64 engine ABI.
        assert_eq!(arity("nsl_cfie_bind_draft_model"), 8);
        assert_eq!(arity("nsl_cfie_draft_pool_alloc"), 1);
        assert_eq!(arity("nsl_cfie_draft_reset"), 0);
        assert_eq!(arity("nsl_cfie_launch_draft_block"), 4);
        assert_eq!(arity("nsl_cfie_launch_draft_sample"), 4);
        assert_eq!(arity("nsl_cfie_launch_verify_probs"), 2);
        assert_eq!(arity("nsl_cfie_speculative_generate"), 8);
    }
}
