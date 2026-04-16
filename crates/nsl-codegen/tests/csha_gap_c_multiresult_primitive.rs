//! Gap C (CSHA fused backward multi-result primitive) structural tests.
//!
//! Gap C adds a structural AD primitive so the forthcoming Gap D EmitFused
//! path can represent the **seven** CSHA fused-backward outputs (dq, dk,
//! dv, dwq, dwk, dwv, dx) inside a Wengert graph whose `WengertOp.result`
//! is a single `VarId`.
//!
//! The primitive mirrors `FlashAttentionBackwardExtract`: a single
//! `PrimalOp::CshaFusedBackwardExtract { component }` variant with seven
//! component values, plus a seven-slot cache on `Compiler` keyed by the
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

/// The cache must round-trip a seven-slot entry under the chain-key
/// `Value` that the extract ops will use as a lookup key.  This is the
/// exact shape Gap D will write: one launch → seven output Values →
/// one `insert(key, [v0..v6])`.
#[test]
fn csha_fused_bwd_cache_roundtrips_seven_slot_entry() {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions::default();
    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed with default options");
    let key = Value::from_u32(100);
    let slots: [Value; 7] = [
        Value::from_u32(200),
        Value::from_u32(201),
        Value::from_u32(202),
        Value::from_u32(203),
        Value::from_u32(204),
        Value::from_u32(205),
        Value::from_u32(206),
    ];
    compiler.csha_fused_bwd_cache.insert(key, slots);
    let fetched = compiler
        .csha_fused_bwd_cache
        .get(&key)
        .copied()
        .expect("insert/get round-trip");
    // All seven slots preserved in order — lowerer's `component as usize`
    // indexes this array, so ordering matters.
    for (i, v) in slots.iter().enumerate() {
        assert_eq!(
            fetched[i], *v,
            "slot {i} must preserve the stored Value"
        );
    }
}

/// Gap D will emit seven `CshaFusedBackwardExtract` ops sharing one
/// chain-key VarId.  Each op carries a distinct `component` in 0..=6.
/// This test pins the mapping so Gap D can rely on it:
///   0 = dq, 1 = dk, 2 = dv, 3 = dwq, 4 = dwk, 5 = dwv, 6 = dx.
#[test]
fn extract_op_component_range_matches_seven_outputs() {
    for c in 0u8..=6 {
        // Constructing succeeds; PartialEq distinguishes distinct components.
        let op = PrimalOp::CshaFusedBackwardExtract { component: c };
        let same = PrimalOp::CshaFusedBackwardExtract { component: c };
        assert_eq!(op, same);
        if c > 0 {
            let prev =
                PrimalOp::CshaFusedBackwardExtract { component: c - 1 };
            assert_ne!(op, prev, "component {c} must not equal {}", c - 1);
        }
    }
}
