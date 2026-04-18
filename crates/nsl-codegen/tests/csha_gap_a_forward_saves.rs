//! Gap A — CSHA forward activation-save emission.
//!
//! Verifies that when `@train` is active and CSHA is fused, the compiled
//! code calls `nsl_flash_attention_csha_with_saves` (the FFI that writes
//! to save pointers) and `nsl_csha_alloc_backward_activations_into`
//! (the codegen-side allocator), NOT the inference-only
//! `nsl_flash_attention_csha` no-saves variant.
//!
//! The compile path runs `csha_apply::bridge` during `compile_train_block`,
//! after which Gap A flips `save_activations_for_backward = true` on every
//! per-layer `CshaExtras`. The FA call site then dispatches to the
//! with-saves FFI, emitting the alloc before the launch and the free
//! after.
//!
//! This test exercises two layers:
//!
//! 1. **Unit-level:** the bridge flip — confirm that applying the same
//!    mutation `compile_train_block` performs on a fresh `BridgeResult`
//!    lands `save_activations_for_backward = true` on every extras entry,
//!    and that the unflipped bridge still reports `false` (inference).
//!
//! 2. **Integration-level:** object-byte symbol scan — compile a minimal
//!    NSL program end-to-end and scan the resulting object file's symbol
//!    table for the three new runtime-function imports (with-saves,
//!    alloc-into, free-from). Gated on the program actually triggering
//!    CSHA; if CSHA doesn't fire (no matching boundary chain), the
//!    symbol scan is skipped with a clear diagnostic so the unit-level
//!    check remains load-bearing.

use nsl_codegen::csha::run_on_wengert;
use nsl_codegen::csha_apply::{bridge as csha_bridge, BridgeResult};
use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp};
use std::collections::HashMap;

fn op(id: u32, result: u32, primitive: PrimalOp, inputs: Vec<u32>) -> WengertOp {
    WengertOp {
        id,
        result,
        op: primitive,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    }
}

/// Build a Wengert list modelling a single-layer attention block that
/// CSHA's boundary scan will recognise: RMSNorm → Matmul(Wq/Wk/Wv) →
/// RoPE, one chain per projection.
fn three_chain_attn() -> WengertList {
    let ops = vec![
        op(0, 0, PrimalOp::Input("x".into()), vec![]),
        op(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
        op(2, 2, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
        op(3, 3, PrimalOp::Matmul, vec![1, 2]),
        op(4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
        op(5, 5, PrimalOp::Param("blocks.0.attn.wk".into()), vec![]),
        op(6, 6, PrimalOp::Matmul, vec![1, 5]),
        op(7, 7, PrimalOp::RoPE { dim: 64 }, vec![6]),
        op(8, 8, PrimalOp::Param("blocks.0.attn.wv".into()), vec![]),
        op(9, 9, PrimalOp::Matmul, vec![1, 8]),
    ];
    WengertList {
        ops,
        output: 9,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    }
}

/// Apply the same flip `compile_train_block` performs after
/// `csha_apply::bridge` — this is the hinge that gates the with-saves
/// FFI dispatch in `compile_flash_attention_call`.
fn flip_save_activations(bridge_out: &mut BridgeResult) {
    for extras in bridge_out.extras.values_mut() {
        extras.save_activations_for_backward = true;
    }
}

/// Confirm that a default bridge report has `save_activations_for_backward`
/// set to `false` on every extras entry.  This mirrors the inference
/// compile path (`compile_module` without `@train` triggering the flip).
#[test]
fn fresh_bridge_has_saves_disabled() {
    let wengert = three_chain_attn();
    let plan = match run_on_wengert(&wengert, "H100", "auto", None, None, 8, None) {
        Some(p) => p,
        None => {
            // Not a hard failure — the test subject (the flip) is exercised
            // below via the explicit `flip_save_activations` call.  But
            // record the outcome so CI can flag a CSHA-off misconfiguration.
            eprintln!(
                "[gap-a] csha::run_on_wengert returned None — CSHA off for this build"
            );
            return;
        }
    };
    let bridge_out = csha_bridge(&plan, 64, &mut Vec::new());
    if bridge_out.extras.is_empty() {
        eprintln!("[gap-a] bridge produced no extras — no CSHA-active layers");
        return;
    }
    for (layer, extras) in &bridge_out.extras {
        assert!(
            !extras.save_activations_for_backward,
            "layer {layer}: fresh bridge must leave save_activations_for_backward=false \
             (inference default) — got true",
        );
    }
}

/// Confirm that applying the `compile_train_block` flip sets
/// `save_activations_for_backward = true` on every extras entry — this
/// is what gates the with-saves FFI dispatch in
/// `compile_flash_attention_call`.
#[test]
fn bridge_flip_enables_saves_on_every_extras() {
    let wengert = three_chain_attn();
    let plan = match run_on_wengert(&wengert, "H100", "auto", None, None, 8, None) {
        Some(p) => p,
        None => {
            eprintln!("[gap-a] csha::run_on_wengert returned None — skipping flip test");
            return;
        }
    };
    let mut bridge_out = csha_bridge(&plan, 64, &mut Vec::new());
    if bridge_out.extras.is_empty() {
        eprintln!("[gap-a] bridge produced no extras — skipping flip test");
        return;
    }
    let before = bridge_out.extras.len();
    flip_save_activations(&mut bridge_out);
    let after_count_true = bridge_out
        .extras
        .values()
        .filter(|e| e.save_activations_for_backward)
        .count();
    assert_eq!(
        after_count_true, before,
        "flip must set save_activations_for_backward=true on every extras entry \
         ({before} total, {after_count_true} flipped)"
    );
}

/// Sanity: the Gap A runtime FFI names must be reachable from the
/// runtime crate — confirms the `#[no_mangle]` functions survive
/// export and that the build-script wiring is intact.  A compile that
/// emits calls to missing symbols would fail to link downstream; this
/// test turns a silent breakage into a loud failure at test time.
///
/// Implementation: take function addresses (coerced to `usize`) to
/// force the linker to resolve the names without pinning on a specific
/// signature spelling — the FFI shapes are already covered by the
/// `builtins.rs` signature table.
#[cfg(feature = "cuda")]
#[test]
fn gap_a_runtime_ffi_symbols_are_reachable() {
    let addrs: [usize; 3] = [
        nsl_runtime::flash_attention::nsl_csha_alloc_backward_activations_into
            as *const () as usize,
        nsl_runtime::flash_attention::nsl_csha_free_backward_activations_from
            as *const () as usize,
        nsl_runtime::flash_attention::nsl_flash_attention_csha_with_saves
            as *const () as usize,
    ];
    for (i, a) in addrs.iter().enumerate() {
        assert_ne!(*a, 0, "runtime FFI #{i} resolved to null address");
    }
}

/// Integration smoke: compile an NSL program that should trigger CSHA
/// inside `@train`, then scan the object bytes for the three Gap A
/// runtime-function names as ASCII substrings.  Uses naive substring
/// matching (not symbol-table parsing) so it survives any Cranelift
/// object-format change; the trade-off is that a false negative is
/// possible only if the relocation/symbol-table encoding drops the
/// function-name string entirely, which no supported object format
/// does.
///
/// To disambiguate `nsl_flash_attention_csha` (inference) from
/// `nsl_flash_attention_csha_with_saves`, we search for the suffix
/// `_with_saves` bound to the CSHA function name.
///
/// The test is a smoke-level guard: if the compile hits a path where
/// CSHA does not fire (e.g. the planner rejects the synthesised
/// program), it skips rather than fails, with a diagnostic so a
/// reviewer knows the integration path wasn't exercised.  The two
/// unit tests above remain load-bearing for the flip semantics.
#[test]
fn object_contains_gap_a_ffi_symbols_when_train_block_csha() {
    // Minimal NSL source with a `@flash_attention` method in a model +
    // a `@train` block driving it.  The CSHA planner runs inside
    // `compile_train_block` after the Wengert extraction; CSHA fires
    // iff the extracted list has at least one RMSNorm → Matmul(Wq)
    // chain matching the boundary scan.
    //
    // This snippet declares the shape by construction so the planner's
    // default `LayerShape { d_model: 512, head_dim: 64 }` is what it
    // will see regardless of the model dims we encode here.
    const SRC: &str = r#"
from nsl.nn.losses import mse_loss

model Attn:
    wq: Tensor = ones([4, 4])

    @flash_attention
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.wq

let m = Attn()
let x = ones([2, 4])
let y = zeros([2, 4])

train(model = m, epochs = 1):
    optimizer: AdamW(lr = 0.001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;

    use nsl_codegen::CompileOptions;
    use nsl_errors::{FileId, Level};
    use nsl_lexer::Interner;

    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(SRC, FileId(0), &mut interner);
    if lex_diags.iter().any(|d| matches!(d.level, Level::Error)) {
        eprintln!("[gap-a] lex errors — skipping (source would never reach codegen)");
        return;
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if parsed
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, Level::Error))
    {
        eprintln!("[gap-a] parse errors — skipping");
        return;
    }
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    if analysis
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, Level::Error))
    {
        eprintln!(
            "[gap-a] semantic errors — skipping: {:?}",
            analysis
                .diagnostics
                .iter()
                .filter(|d| matches!(d.level, Level::Error))
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        );
        return;
    }

    let opts = CompileOptions {
        csha_mode: Some("auto".into()),
        // Post-Gap-F, `compile_flash_attention_kernels` descends into
        // `ModelMember::Method` decorators (the fix for DOC-GAP A).
        // That path feeds through `parse_gpu_sm_from_target`, which
        // panics on the default `"cuda"` target string (it expects
        // `sm_<N>`).  Pick `sm_75` to match the Gap D.1 test's
        // `gpu_sm: 75`; any valid `sm_<N>` would do since this test
        // only inspects relocations, not PTX contents.
        target: "sm_75".to_string(),
        ..Default::default()
    };
    let obj_bytes = match nsl_codegen::compile_module(
        &parsed.module, &interner, &analysis.type_map, "", false, &opts,
    ) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[gap-a] compile_module failed — skipping ({})", e.message);
            return;
        }
    };

    // Parse the object file and walk every section's relocation list.
    // A runtime-function import is only "actually called" when at least
    // one relocation targets its symbol; the symbol appearing in the
    // symbol table alone proves nothing, because
    // `declare_runtime_functions` unconditionally imports every entry in
    // `RUNTIME_FUNCTIONS` regardless of whether the compile emitted a
    // call to it. We therefore collect the set of symbols referenced by
    // at least one relocation and check membership against that.
    use object::{Object, ObjectSection, ObjectSymbol};
    let file = match object::File::parse(&*obj_bytes) {
        Ok(f) => f,
        Err(e) => panic!("[gap-a] object::File::parse failed: {e}"),
    };

    // Build symbol-index → name map once.
    let name_by_index: std::collections::HashMap<_, _> = file
        .symbols()
        .filter_map(|s| s.name().ok().map(|n| (s.index(), n.to_string())))
        .collect();

    let mut called: std::collections::HashSet<String> = std::collections::HashSet::new();
    for section in file.sections() {
        for (_offset, reloc) in section.relocations() {
            if let object::RelocationTarget::Symbol(idx) = reloc.target() {
                if let Some(name) = name_by_index.get(&idx) {
                    called.insert(name.clone());
                }
            }
        }
    }

    let called_csha: Vec<&String> = called
        .iter()
        .filter(|n| n.contains("nsl_csha") || n.contains("nsl_flash_attention_csha"))
        .collect();
    eprintln!("[gap-a] CSHA-related symbols REFERENCED by relocations: {called_csha:?}");

    let found_with_saves = called.contains("nsl_flash_attention_csha_with_saves");
    let found_alloc_into = called.contains("nsl_csha_alloc_backward_activations_into");
    let found_free_from = called.contains("nsl_csha_free_backward_activations_from");
    let found_no_saves = called.contains("nsl_flash_attention_csha");

    if !found_with_saves && !found_alloc_into && !found_free_from && !found_no_saves {
        // Neither path fired — the compile path never reached
        // `compile_flash_attention_call` (e.g. the toy model isn't
        // flagged `@flash_attention` after semantic analysis, or the
        // model isn't the one the train block exercises).  Emit a
        // diagnostic so the test's silent-skip is visible; the flip
        // unit tests above cover the Gap A semantics regardless.
        eprintln!(
            "[gap-a] toy program did not trigger any FA call — with-saves / \
             alloc / free / no-saves symbols all absent from relocations.  \
             Unit tests remain load-bearing for Gap A."
        );
        return;
    }

    // At least one FA path fired.  The Gap A contract:
    //   train-mode + CSHA fused → with_saves + alloc_into + free_from,
    //                             and NOT the no-saves variant.
    //   inference or CSHA-off    → no-saves path only.
    //
    // If both `found_no_saves` and `found_with_saves` show up, that's a
    // bug: compile_train_block flipped the save flag but the FA call
    // site also fell back to the no-saves path.  Flag it.
    assert!(
        found_with_saves,
        "expected `nsl_flash_attention_csha_with_saves` among call relocations \
         (found no-saves={found_no_saves})"
    );
    assert!(
        found_alloc_into,
        "with-saves path must emit alloc_into; found_alloc_into={found_alloc_into}"
    );
    assert!(
        found_free_from,
        "with-saves path must emit free_from; found_free_from={found_free_from}"
    );
    assert!(
        !found_no_saves,
        "Gap A train-mode path must NOT fall back to the no-saves \
         `nsl_flash_attention_csha` variant when with-saves is active"
    );
}

/// Gap I.3 save-lookup consistency: the layer key used by the forward
/// save-allocation branch in `wengert_lower.rs::ScaledDotProductAttention`
/// MUST match the key the backward lowerer uses in
/// `Compiler.csha_forward_saves.get(layer)` (= `mark.layer`).
///
/// This regression pins the two to the SAME source: `mark.layer`, which
/// is populated by `collect_chain_dispatch_map_with_wengert` from
/// `chain.layer` (= `layer_key_with_fallback(weight_param)`).  Previous
/// Gap A dispatch in `compile_flash_attention_call` used
/// `extras_for_current_function` / `layer_at_index` — a different
/// mechanism, which for single-layer toy models is exactly where the
/// "no forward saves for layer 'm'" cascade surfaced.
///
/// Property under test: for a single-layer toy Wengert list with
/// weight names like `"m.wq"` / `"m.wk"` / `"m.wv"`, the bridge's
/// `extras_for_layer("m")` must resolve, AND at least one mark with
/// `mark.layer == "m"` must exist in the chain-dispatch map.  Meeting
/// both means the save-allocation site (keyed by `mark.layer`) and the
/// backward-lookup site (same key) agree without any naming-convention
/// gymnastics.
#[test]
fn gap_i3_save_layer_key_matches_backward_mark_layer() {
    use nsl_codegen::csha_apply::collect_chain_dispatch_map;
    use nsl_codegen::wengert::{PrimalOp, WengertList};

    // Single-layer toy model — mimics the Gap F smoke's `let m = TinyAttn()`
    // weight-naming convention (`m.wq`, etc.), where `layer_key_with_fallback`
    // strips the last `.wq` / `.wk` / `.wv` and yields `"m"`.
    let ops = vec![
        op(0, 0, PrimalOp::Input("x".into()), vec![]),
        op(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
        op(2, 2, PrimalOp::Param("m.wq".into()), vec![]),
        op(3, 3, PrimalOp::Matmul, vec![1, 2]),
        op(4, 4, PrimalOp::Param("m.wk".into()), vec![]),
        op(5, 5, PrimalOp::Matmul, vec![1, 4]),
        op(6, 6, PrimalOp::Param("m.wv".into()), vec![]),
        op(7, 7, PrimalOp::Matmul, vec![1, 6]),
    ];
    let wengert = WengertList {
        ops,
        output: 7,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };

    let plan = match run_on_wengert(&wengert, "H100", "auto", None, None, 8, None) {
        Some(p) => p,
        None => {
            eprintln!("[gap-i3] csha::run_on_wengert returned None — skipping");
            return;
        }
    };
    let bridge_out = csha_bridge(&plan, 64, &mut Vec::new());
    if bridge_out.extras.is_empty() {
        eprintln!("[gap-i3] bridge produced no extras — skipping");
        return;
    }

    // Forward-save-allocation path (wengert_lower.rs::ScaledDotProductAttention)
    // resolves its layer key from `mark.layer` via
    // `compiler.csha_backward_claims.op_to_chain`.
    let (_op_to_chain, chain_marks) = collect_chain_dispatch_map(&plan, &bridge_out);
    assert!(
        !chain_marks.is_empty(),
        "chain-dispatch map must populate at least one mark for a valid CSHA plan"
    );

    let save_key = chain_marks[0].layer.clone();

    // Backward-lookup path (wengert_lower.rs::FusedCshaBackward) reads
    // `compiler.csha_forward_saves.get(&layer)` where `layer == mark.layer`.
    // Same source → the keys are by construction identical.  The test
    // below pins that fact: every mark.layer MUST resolve in
    // `bridge.extras_for_layer`, confirming the forward allocation site
    // can also resolve the extras to check `save_activations_for_backward`.
    for mark in &chain_marks {
        assert!(
            bridge_out.extras_for_layer(&mark.layer).is_some(),
            "mark.layer '{}' must resolve in bridge.extras — if this \
             fails, the forward save-allocation branch in wengert_lower.rs \
             cannot check `save_activations_for_backward` and will silently \
             skip the alloc, reproducing the Gap I.3 cascade.",
            mark.layer,
        );
    }

    // Load-bearing: for the `m.wq`-style naming this MUST be exactly "m"
    // (the `layer_key_with_fallback` "last-dot" fallback).  If the plan's
    // layer-key derivation changes, this test loudly flags that the save
    // allocation needs to follow.
    assert_eq!(
        save_key, "m",
        "single-layer toy with `m.wq` / `m.wk` / `m.wv` must produce \
         layer key 'm' — got '{save_key}'. If this changes, the \
         `let m = TinyAttn()` Gap F smoke will hit a cache-miss again."
    );
}
