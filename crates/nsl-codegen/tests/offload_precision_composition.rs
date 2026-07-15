//! P0.2/P0.3 — optimizer-state offload emission + composition with
//! reduced-precision moments.
//!
//! Until the 2026-07-15 pretraining memory-reduction pass, combining
//! `--optim-state-offload` with a CPDT/WGGO reduced-precision moment plan
//! was a HARD compile error at train-block setup ("cannot combine with
//! reduced-precision optimizer moments ... the requant copy-back cannot
//! cross devices"). P0.3 replaced the co-resident requant with the
//! cross-device cast envelope (`nsl_tensor_cast_from_host` /
//! `nsl_tensor_cast_to_host_into`), so the combination now compiles and
//! allocates HOST state at the planned dtypes
//! (`nsl_tensor_zeros_like_host_dtype`).
//!
//! Both tests compile through `compile_entry` (NOT the module-library
//! path: `compile_module_with_imports_*` never lowers top-level
//! statements, and the train block here lives in `fn main()` so lowering
//! actually happens) and assert at the RELOCATION level — symbol-table
//! membership alone is meaningless because `declare_runtime_functions`
//! declares every runtime symbol unconditionally; only a relocation
//! proves the compile emitted a call (pattern from
//! `csha_gap_f_toy_pretrain_smoke.rs`).
//!
//! The refusal predicate was `cpdt_precision_dtypes.is_some()` — tier
//! CONTENT was irrelevant — so an active plan of any tier mix reproduces
//! the old hard error. Reduced-precision NUMERICS for the combined
//! envelope are pinned at the runtime level in `precision_cast.rs`'s
//! `tests::gpu` module; this file pins the codegen emission + the
//! refusal removal. The pipelined-path refusal is NOT relaxed (offload
//! remains refused there) — see `compile_train_block_pipelined`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use nsl_codegen::CompileOptions;
use object::{Object, ObjectSymbol};

/// Toy AdamW + grad_accumulation=4 training program (FASE Deferred).
/// The train block sits inside `fn main()` so `compile_entry` lowers it.
const TRAIN_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

fn main():
    let m = Tiny()
    let x = ones([2, 4])
    let y = zeros([2, 4])
    train(model = m, epochs = 2, grad_accumulation = 4):
        optimizer: AdamW(lr = 0.001, weight_decay = 0.01)
        step(batch):
            let pred = m.forward(x)
            let loss = mse_loss(pred, y)
"#;

/// Same program with TWO identical train blocks. The CPDT plan is stashed
/// on the compiler DURING train-block lowering (`invoke_cpdt_if_enabled`
/// runs after the block's own dtype resolution), so block 1 seeds
/// `compiler.cpdt_plan` and block 2's `cpdt_precision_dtypes` resolution
/// sees an ACTIVE plan — the deterministic single-compile route to the
/// offload x precision composition (mirrors the CLI's plan persistence
/// across blocks; no cost-model-dependent memory budget needed).
const TRAIN_SRC_TWO_BLOCKS: &str = r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

fn main():
    let m = Tiny()
    let x = ones([2, 4])
    let y = zeros([2, 4])
    train(model = m, epochs = 2, grad_accumulation = 4):
        optimizer: AdamW(lr = 0.001, weight_decay = 0.01)
        step(batch):
            let pred = m.forward(x)
            let loss = mse_loss(pred, y)
    train(model = m, epochs = 2, grad_accumulation = 4):
        optimizer: AdamW(lr = 0.001, weight_decay = 0.01)
        step(batch):
            let pred = m.forward(x)
            let loss = mse_loss(pred, y)
"#;

/// Parse + semantic-check + stdlib-load + `compile_entry`.
fn compile_src(src: &str, opts: &CompileOptions) -> Result<Vec<u8>, String> {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    assert!(
        !lex_diags.iter().any(|d| matches!(d.level, nsl_errors::Level::Error)),
        "lex errors: {lex_diags:?}"
    );
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        !parsed.diagnostics.iter().any(|d| matches!(d.level, nsl_errors::Level::Error)),
        "parse errors: {:?}",
        parsed.diagnostics
    );
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        !analysis.diagnostics.iter().any(|d| matches!(d.level, nsl_errors::Level::Error)),
        "semantic errors: {:?}",
        analysis.diagnostics
    );
    let imported_fns = nsl_codegen::stdlib_loader::build_imported_fns_for_entry(
        &parsed.module,
        &mut interner,
        &analysis.type_map,
        opts,
    )
    .map_err(|e| e.message)?;
    nsl_codegen::compile_entry(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &imported_fns,
        HashMap::new(),
        HashSet::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        false,
        opts,
    )
    .map_err(|e| e.message)
}

/// Symbols referenced by at least one relocation in the object.
fn relocation_symbol_set(obj_bytes: &[u8]) -> HashSet<String> {
    use object::ObjectSection;
    let file = object::File::parse(obj_bytes).expect("object::File::parse");
    let name_by_index: HashMap<_, _> = file
        .symbols()
        .filter_map(|s| {
            let n = s.name().ok()?;
            Some((
                s.index(),
                nsl_codegen::linker::strip_host_symbol_prefix(n).to_string(),
            ))
        })
        .collect();
    let mut out = HashSet::new();
    for section in file.sections() {
        for (_, reloc) in section.relocations() {
            if let object::RelocationTarget::Symbol(idx) = reloc.target() {
                if let Some(name) = name_by_index.get(&idx) {
                    out.insert(name.clone());
                }
            }
        }
    }
    out
}

/// Write a safetensors file whose single tensor matches the toy model's
/// parameter (`w: [4, 4]`) so the CPDT weight-aware cascade produces a
/// NON-EMPTY precision plan. The lone param maps to the non-hierarchical
/// "other" layer, so `cpdt_sensitivity::validate` (which hard-exits the
/// process on mismatched HIERARCHICAL layers) checks zero layers and
/// passes.
fn write_matching_weights() -> PathBuf {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;

    let vals: Vec<f32> = vec![1e-4_f32; 16];
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    views.insert(
        "w".to_string(),
        TensorView::new(Dtype::F32, vec![4, 4], &bytes).unwrap(),
    );
    let serialized = serialize(&views, &None).unwrap();
    let out = std::env::temp_dir().join(format!(
        "nsl_offload_precision_composition_{}.safetensors",
        std::process::id()
    ));
    std::fs::write(&out, serialized).unwrap();
    out
}

/// P0.2 emission pin: offload WITHOUT a precision plan compiles, and the
/// object actually CALLS the new envelope FFIs — host-f32 state alloc,
/// the consuming async copy-back, and the once-per-step drain.
#[test]
fn offload_only_emits_async_copy_back_and_drain() {
    let opts = CompileOptions {
        source_ad: true,
        optim_state_offload: true,
        ..Default::default()
    };
    let bytes = compile_src(TRAIN_SRC, &opts)
        .unwrap_or_else(|e| panic!("offload-only train block must compile: {e}"));
    let relocs = relocation_symbol_set(&bytes);
    for sym in [
        "nsl_tensor_zeros_like_host_f32",
        "nsl_tensor_copy_data_async",
        "nsl_offload_drain",
    ] {
        assert!(
            relocs.contains(sym),
            "offload-only object must reference {sym} (P0.2 emission); \
             referenced runtime symbols: {:?}",
            {
                let mut v: Vec<_> = relocs.iter().filter(|s| s.starts_with("nsl_")).collect();
                v.sort();
                v
            }
        );
    }
    // P0.2 switched the stage-out to the CONSUMING copy: the plain
    // copy_data must no longer be emitted by the offload envelope.
    // (nsl_tensor_copy_data can legitimately appear from OTHER lowerings,
    // so no negative assert — the positive asserts above are the pin.)
}

/// THE P0.3 pin: offload + an ACTIVE CPDT precision moment plan compiles
/// (pre-P0.3: hard CodegenError), allocates host state at planned dtypes,
/// and stages through the cross-device cast envelope.
#[test]
fn offload_composes_with_cpdt_precision_plan() {
    let weights = write_matching_weights();
    let plan_slot: Arc<Mutex<Option<nsl_codegen::cpdt::CpdtPlan>>> = Arc::new(Mutex::new(None));
    let opts = CompileOptions {
        source_ad: true,
        optim_state_offload: true,
        weight_file: Some(weights.clone()),
        wggo: nsl_codegen::WggoOptions {
            // CPDT only runs when WGGO produced an AppliedPlan.
            mode: Some("full".to_string()),
            ..Default::default()
        },
        cpdt: nsl_codegen::CpdtOptions {
            mode: nsl_codegen::cpdt::CpdtMode::Full,
            cluster: Some(nsl_codegen::cpdt_zero::ClusterSpec::default()),
            plan_out: Some(plan_slot.clone()),
            ..Default::default()
        },
        ..Default::default()
    };

    let result = compile_src(TRAIN_SRC_TWO_BLOCKS, &opts);
    let _ = std::fs::remove_file(&weights);
    let bytes = match result {
        Ok(b) => b,
        Err(msg) => {
            assert!(
                !msg.contains("cannot combine with reduced-precision"),
                "P0.3 regression: the offload x precision refusal is back: {msg}"
            );
            panic!("offload + CPDT precision plan must compile after P0.3, got: {msg}");
        }
    };

    // Guard against a vacuous pass: the precision cascade must actually
    // have fired (non-empty per-param tier plan). If this trips, the test
    // stopped exercising the composition — fix the activation plumbing
    // rather than deleting the assert.
    {
        let plan = plan_slot.lock().unwrap();
        let plan = plan
            .as_ref()
            .expect("CpdtMode::Full + cluster + WGGO plan must produce a CpdtPlan");
        assert!(
            !plan.precision.params.is_empty(),
            "precision plan is empty — the composition path was not exercised"
        );
    }

    // Emission pin: host state allocated at the PLANNED dtype and staged
    // through the cross-device cast envelope (not the f32-only pair).
    let relocs = relocation_symbol_set(&bytes);
    for sym in [
        "nsl_tensor_zeros_like_host_dtype",
        "nsl_tensor_cast_from_host",
        "nsl_tensor_cast_to_host_into",
        "nsl_offload_drain",
    ] {
        assert!(
            relocs.contains(sym),
            "composed object must reference {sym} (P0.3 combined envelope); \
             referenced runtime symbols: {:?}",
            {
                let mut v: Vec<_> = relocs.iter().filter(|s| s.starts_with("nsl_")).collect();
                v.sort();
                v
            }
        );
    }
}
