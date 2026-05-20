//! Gap C (CSHA fused backward multi-result primitive) structural tests.
//!
//! Gap C adds a structural AD primitive so the Gap D EmitFused path can
//! represent the CSHA fused-backward outputs inside a Wengert graph whose
//! `WengertOp.result` is a single `VarId`. Gap I.5 Option-A extended the
//! output count from seven to **eight**: dq, dk, dv, dwq, dwk, dwv, dx,
//! dx_norm — the 8th (`dx_norm`) is the gradient w.r.t. the RMSNorm
//! OUTPUT, fed to the AD-side `RmsNormGammaBackward` for correct dgamma
//! under the fused CSHA dispatcher claim path.
//!
//! The primitive mirrors `FlashAttentionBackwardExtract`: a single
//! `PrimalOp::CshaFusedBackwardExtract { component }` variant with eight
//! component values, plus an eight-slot cache on `Compiler` keyed by the
//! extract's first-input Cranelift `Value` ("chain key").
//!
//! These tests verify the primitive's shape — the Wengert-level variant
//! and the `Compiler` cache field — without touching the real lowerer
//! (which needs a full `FunctionBuilder`; exercised by Gap D's
//! integration tests once the EmitFused path emits real extract ops).
#![cfg(feature = "test-helpers")]

use cranelift_codegen::ir::Value;
use nsl_codegen::wengert::PrimalOp;
use nsl_codegen::CompileOptions;
use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;
use std::collections::HashMap;

/// `Compiler::new` must initialize the Gap C cache empty so Gap D can
/// start inserting into it on the first fused-backward chain without
/// observing stale entries from a prior compile.
#[test]
fn compiler_new_initialises_csha_fused_bwd_cache_empty() {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions::default();
    let compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed with default options");
    assert!(
        compiler.csha_fused_bwd_cache.is_empty(),
        "csha_fused_bwd_cache must start empty"
    );
}

/// The cache must round-trip an eight-slot entry under the chain-key
/// `Value` that the extract ops will use as a lookup key.  This is the
/// exact shape Gap D now writes after the Gap I.5 Option-A fix: one
/// launch → eight output Values → one `insert(key, [v0..v7])`.
#[test]
fn csha_fused_bwd_cache_roundtrips_seven_slot_entry() {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions::default();
    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed with default options");
    let key = Value::from_u32(100);
    let slots: [Value; 8] = [
        Value::from_u32(200),
        Value::from_u32(201),
        Value::from_u32(202),
        Value::from_u32(203),
        Value::from_u32(204),
        Value::from_u32(205),
        Value::from_u32(206),
        Value::from_u32(207),
    ];
    compiler.csha_fused_bwd_cache.insert(key, slots);
    let fetched = compiler
        .csha_fused_bwd_cache
        .get(&key)
        .copied()
        .expect("insert/get round-trip");
    // All eight slots preserved in order — lowerer's `component as usize`
    // indexes this array, so ordering matters.
    for (i, v) in slots.iter().enumerate() {
        assert_eq!(fetched[i], *v, "slot {i} must preserve the stored Value");
    }
}

/// Gap D emits eight `CshaFusedBackwardExtract` ops sharing one chain-key
/// VarId.  Each op carries a distinct `component` in 0..=7. This test
/// pins the mapping so Gap D can rely on it:
///   0 = dq, 1 = dk, 2 = dv, 3 = dwq, 4 = dwk, 5 = dwv, 6 = dx, 7 = dx_norm.
/// Component 7 is the Gap I.5 Option-A addition (gradient w.r.t. the
/// RMSNorm output, consumed by `RmsNormGammaBackward`).
#[test]
fn extract_op_component_range_matches_seven_outputs() {
    for c in 0u8..=7 {
        // Constructing succeeds; PartialEq distinguishes distinct components.
        let op = PrimalOp::CshaFusedBackwardExtract { component: c };
        let same = PrimalOp::CshaFusedBackwardExtract { component: c };
        assert_eq!(op, same);
        if c > 0 {
            let prev = PrimalOp::CshaFusedBackwardExtract { component: c - 1 };
            assert_ne!(op, prev, "component {c} must not equal {}", c - 1);
        }
    }
}

/// Gap I.B: `clear_csha_per_function_caches` empties both
/// `csha_fused_bwd_cache` and `csha_forward_saves` so a subsequent
/// train/grad block in the same module starts with no aliased entries.
///
/// Cranelift `Value` IDs restart at 0 for every new function — if a prior
/// function left `Value::from_u32(42) → [v0..v7]` behind, a later
/// function's first cache lookup against its own `Value::from_u32(42)`
/// would return ghost outputs that don't belong to it.  The helper makes
/// the post-function state byte-equivalent to a fresh `Compiler::new()`
/// for these two fields.
#[test]
fn clear_csha_per_function_caches_empties_both_maps() {
    use nsl_codegen::csha_apply::CshaSavePointers;

    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions::default();
    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed with default options");

    // Populate `csha_fused_bwd_cache` the way `FusedCshaBackward` does.
    let key = Value::from_u32(42);
    let slots: [Value; 8] = [
        Value::from_u32(300),
        Value::from_u32(301),
        Value::from_u32(302),
        Value::from_u32(303),
        Value::from_u32(304),
        Value::from_u32(305),
        Value::from_u32(306),
        Value::from_u32(307),
    ];
    compiler.csha_fused_bwd_cache.insert(key, slots);

    // Populate `csha_forward_saves` the way the FA call site does.
    compiler.csha_forward_saves.insert(
        "TinyAttn".to_string(),
        CshaSavePointers {
            q_proj: Value::from_u32(400),
            k_proj: Value::from_u32(401),
            v_proj: Value::from_u32(402),
            row_max: Value::from_u32(403),
            row_sum: Value::from_u32(404),
            x_raw: Value::from_u32(405),
            backward_ptx_data_id: None,
            backward_name_data_id: None,
        },
    );

    assert!(!compiler.csha_fused_bwd_cache.is_empty(), "precondition");
    assert!(!compiler.csha_forward_saves.is_empty(), "precondition");

    compiler.clear_csha_per_function_caches();

    assert!(
        compiler.csha_fused_bwd_cache.is_empty(),
        "csha_fused_bwd_cache must be empty after clear_csha_per_function_caches; \
         otherwise a later function's `Value::from_u32(N)` lookup could alias \
         against this function's stale entry"
    );
    assert!(
        compiler.csha_forward_saves.is_empty(),
        "csha_forward_saves must be empty after clear_csha_per_function_caches; \
         otherwise a later function's backward emission could read save pointers \
         that point into the previous function's stack frame"
    );
}
