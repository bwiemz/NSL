//! The runtime-function registry: every `extern "C"` symbol the codegen can
//! emit a call to, with its Cranelift signature.
//!
//! # Why this is split, and along which line
//!
//! The registry was one 3,500-line table. The split is between what the
//! LANGUAGE exposes and what the RUNTIME implements:
//!
//! * [`mod@crate::builtins`] — the surface a program writes directly: printing,
//!   scalar arithmetic, collections, and the tensor operations that have NSL
//!   syntax or a stdlib spelling. Adding one of these widens the language.
//! * [`mod@crate::runtime_abi`] — the implementation surface behind it: fused
//!   kernels, the training loop's plumbing, optimizers, parallelism, serving,
//!   quantization, diagnostics. Adding one of these does not change what a
//!   program can say, only how it runs.
//!
//! The line is "semantic primitive vs composition", not "could this be written
//! in NSL?" — plenty of `builtins` entries could be, and stay here because the
//! compiler needs to know their shape, dtype or aliasing behaviour.
//!
//! # Declaration order is not load-bearing
//!
//! [`declare_runtime_functions`] walks the tables in the order of
//! [`RUNTIME_TABLES`], so `FuncId`s are assigned in that order. That order does
//! NOT reach the emitted CLIF: it names its runtime callees (`fn0 = nsl_alloc`)
//! and numbers funcrefs per function by order of first use, so the global
//! `FuncId` never appears. Verified by moving `nsl_alloc` from index 6 to index
//! 0 before this split: all 26 `train_clif_snapshots` stayed byte-identical.
//! That is what makes regrouping the registry free, and why an entry may be
//! moved between these files on the strength of where it belongs.
//!
//! It does reach the OBJECT FILE — `cranelift-object` calls `add_symbol`
//! eagerly from `declare_function`, `Linkage::Import` included, so the `.o`
//! symbol table follows table order. Nothing checked in depends on that today.
//!
//! # Adding a runtime function
//!
//! Put it in the file its subject matter belongs to, in any position. It must
//! appear exactly once across every table — `no_runtime_function_is_declared_twice`
//! checks that here, and `nsl-abi`'s `DuplicateDecl` checks it across files —
//! and its signature must match the `extern "C" fn` in `nsl-runtime`, which
//! `nsl-abi`'s `signature_agreement` gate enforces. The two are linked by
//! symbol name only; nothing else in the build catches a drift.

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

/// Every table in the registry.
///
/// A table missing from this list is silently never declared, and the calls
/// the codegen emits to it fail to link — so a new file must be added here as
/// well as to its `mod` declaration.
///
/// `every_declared_table_is_reachable` catches that: it parses every table
/// under `src/` and asserts the names found there are exactly the names
/// reachable through this list. (`dead_code = "deny"` usually gets there
/// first, rejecting the unused const at compile time; the test covers the
/// case rustc cannot see, a file that is never `mod`-declared.)
const RUNTIME_TABLES: &[&[RuntimeFn]] = &[
    memory::RUNTIME_FUNCTIONS_MEMORY,
    io::RUNTIME_FUNCTIONS_IO,
    scalar::RUNTIME_FUNCTIONS_SCALAR,
    collections::RUNTIME_FUNCTIONS_COLLECTIONS,
    tensor::RUNTIME_FUNCTIONS_TENSOR,
    crate::runtime_abi::tensor::RUNTIME_FUNCTIONS_ABI_TENSOR,
    crate::runtime_abi::training::RUNTIME_FUNCTIONS_ABI_TRAINING,
    crate::runtime_abi::optimizer::RUNTIME_FUNCTIONS_ABI_OPTIMIZER,
    crate::runtime_abi::distributed::RUNTIME_FUNCTIONS_ABI_DISTRIBUTED,
    crate::runtime_abi::inference::RUNTIME_FUNCTIONS_ABI_INFERENCE,
    crate::runtime_abi::quantization::RUNTIME_FUNCTIONS_ABI_QUANTIZATION,
    crate::runtime_abi::diagnostics::RUNTIME_FUNCTIONS_ABI_DIAGNOSTICS,
    crate::runtime_abi::memory::RUNTIME_FUNCTIONS_ABI_MEMORY,
    crate::runtime_abi::interop::RUNTIME_FUNCTIONS_ABI_INTEROP,
];

/// Every runtime function, across every table.
pub(crate) fn all_runtime_functions() -> impl Iterator<Item = &'static RuntimeFn> {
    RUNTIME_TABLES.iter().flat_map(|t| t.iter())
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
    use super::all_runtime_functions;

    /// Every table that EXISTS is reachable from [`super::RUNTIME_TABLES`].
    ///
    /// Splitting the registry introduced a failure mode the single table did
    /// not have: a new file can declare a `RUNTIME_FUNCTIONS*` const, be added
    /// to its `mod` list so it compiles, and be left out of `RUNTIME_TABLES`.
    /// Nothing then declares those functions, and every call the codegen emits
    /// to one fails at LINK time, in whatever unrelated test happens to link
    /// first. `nsl-abi` cannot catch it either: it parses the sources, so it
    /// sees the orphaned table and is satisfied.
    ///
    /// So this reads the registry's own sources and compares the names it
    /// finds against the names actually reachable at runtime.
    ///
    /// The reading is done by `nsl-abi`, which is the parser the ABI gate
    /// already uses on these same files. A second hand-rolled scanner lived
    /// here first and was wrong in a way worth recording: it recognised only
    /// `const` and `pub(crate) const`, so a table spelled `pub(super) const`
    /// — arguably the more idiomatic visibility, since these are read only by
    /// the parent — would have been invisible to it. Its names would never
    /// enter `on_disk`, they would not be reachable either, and BOTH
    /// assertions below would have passed over exactly the hole this test
    /// exists to close. Sharing the parser also means the two cannot drift.
    #[test]
    fn every_declared_table_is_reachable() {
        use std::collections::BTreeSet;

        fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {dir:?}: {e}")) {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    rust_files(&path, out);
                } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                    out.push(path);
                }
            }
        }

        // The WHOLE crate source, not just the two registry directories: a
        // table in a third directory, or nested one level deeper, is one
        // `nsl-abi` would still find and signature-check while nothing
        // declared it.
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_files(&src, &mut files);
        files.sort();

        let mut on_disk: BTreeSet<String> = BTreeSet::new();
        let (mut found, mut parsed) = (0usize, 0usize);
        for path in &files {
            let text =
                std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
            let label = path.display().to_string();
            let (sigs, seen, ok) = nsl_abi::parse_runtime_functions_table_in_file(&text, &label);
            found += seen;
            parsed += ok;
            on_disk.extend(sigs.into_iter().map(|s| s.name));
        }
        assert_eq!(
            found, parsed,
            "{found} table declaration(s) found under src/ but {parsed} parsed — one was \
             recognised and then not read, so the comparison below is over an incomplete set."
        );

        let reachable: BTreeSet<String> =
            all_runtime_functions().map(|(n, _, _)| (*n).to_string()).collect();

        let orphaned: Vec<&String> = on_disk.difference(&reachable).collect();
        assert!(
            orphaned.is_empty(),
            "{} runtime function(s) are declared in a table that RUNTIME_TABLES does not \
             list, so nothing declares them and every emitted call to one fails at LINK \
             time. Add the table to RUNTIME_TABLES in builtins/mod.rs:\n  {:?}",
            orphaned.len(),
            orphaned
        );
        // The converse would mean the scanner missed a table it should have
        // found, which would make the check above vacuous.
        let unseen: Vec<&String> = reachable.difference(&on_disk).collect();
        assert!(
            unseen.is_empty(),
            "the source scan missed {} reachable function(s) — this gate is not seeing \
             the tables it claims to check:\n  {:?}",
            unseen.len(),
            unseen
        );
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
