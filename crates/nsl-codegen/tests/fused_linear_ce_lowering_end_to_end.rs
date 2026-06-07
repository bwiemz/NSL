//! CFTP §4.4 G3 (Sprint 4) — end-to-end integration test for the
//! `fused_linear_ce` compiler-side surface.
//!
//! This test exercises the compiler decisions WITHOUT spinning up a
//! Cranelift JIT or GPU launch.  Reasons:
//!
//! * GPU numerical correctness is already covered by
//!   `fused_linear_ce_numerical.rs` (V=4096) and
//!   `fused_linear_ce_large_vocab_numerical.rs` (V=49152).  Both run the
//!   same FFI that Sprint 4 dispatches into; numerical equivalence vs
//!   CPU reference is already locked in there.
//! * The Sprint 4 deliverable is the COMPILER-side wiring: builtin
//!   recognition, PrimalOp construction with correct shape facts, AD rule
//!   that produces three FusedLinearCeBackwardExtract ops, the
//!   composite-fallback regression invariant, and the routing decision
//!   at the LARGE_VOCAB_THRESHOLD boundary.
//!
//! The test asserts each of these decisions independently using the same
//! plumbing path a real compile would take.

use nsl_codegen::ad_rules::{apply_ad_rule, AdjointExpr};
use nsl_codegen::fused_linear_ce::LARGE_VOCAB_THRESHOLD;
use nsl_codegen::wengert::{PrimalOp, WengertOp};
use nsl_codegen::FusedCeDecoratorConfig;
use nsl_codegen::source_ad::WengertExtractor;
use nsl_lexer::Interner;

/// Extract the Wengert list for a single-fn test program, optionally with
/// a `@fused_lm_ce` decorator config plumbed in.  Returns the extracted
/// list (when extraction succeeded).
fn extract_first_fn(
    src: &str,
    cfg: Option<FusedCeDecoratorConfig>,
) -> Option<nsl_codegen::wengert::WengertList> {
    let mut interner_box = Box::new(Interner::new());
    let (tokens, _) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner_box);
    let parsed = nsl_parser::parse(&tokens, &mut interner_box);
    let module = parsed.module;
    // SAFETY: leak the interner so the returned references live for the
    // test duration. This is a one-off test pattern.
    let interner_static: &'static Interner = Box::leak(interner_box);
    let mut extractor = WengertExtractor::new(interner_static).with_fused_ce_config(cfg);
    // Find the first FnDef in the module.
    let fn_def = module
        .stmts
        .iter()
        .find_map(|s| match &s.kind {
            nsl_ast::stmt::StmtKind::FnDef(f) => Some(f),
            _ => None,
        })
        .expect("test src must contain a fn");
    // Pre-register the params by symbol.
    for param in &fn_def.params {
        extractor.register_input(param.name);
    }
    let ok = extractor.extract_stmts(&fn_def.body.stmts);
    if !ok {
        return None;
    }
    extractor.finalize()
}

// ─── Test 1: no decorator → composite fallback (no FusedLinearCe emitted) ─

#[test]
fn no_decorator_falls_through_to_composite() {
    let src = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    return fused_linear_ce(x, w, bias, targets)
"#;
    let list = extract_first_fn(src, None);
    // Either extraction succeeded but produced NO FusedLinearCe ops, or
    // extraction failed (the stdlib fn body wasn't visible and the
    // extractor bailed) — both cases satisfy the regression invariant:
    // the fused PrimalOp must not appear without the decorator.
    if let Some(list) = list {
        let has_fused = list
            .ops
            .iter()
            .any(|op| matches!(op.op, PrimalOp::FusedLinearCe { .. }));
        assert!(
            !has_fused,
            "no-decorator path must not emit PrimalOp::FusedLinearCe"
        );
    }
}

// ─── Test 2: decorator with shapes → builtin fires; small vocab → !is_large ─

#[test]
fn decorator_with_shapes_emits_fused_op_at_small_vocab() {
    let src = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    return fused_linear_ce(x, w, bias, targets)
"#;
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(64),
    };
    let list = extract_first_fn(src, Some(cfg)).expect("extraction must succeed");
    let fused: Vec<_> = list
        .ops
        .iter()
        .filter_map(|op| match &op.op {
            PrimalOp::FusedLinearCe {
                vocab_size,
                hidden_size,
                batch_size,
                seq_len,
                vocab_tile,
                is_large,
                ignore_index,
            } => Some((
                *vocab_size,
                *hidden_size,
                *batch_size,
                *seq_len,
                *vocab_tile,
                *is_large,
                *ignore_index,
            )),
            _ => None,
        })
        .collect();
    assert_eq!(fused.len(), 1, "expected exactly one FusedLinearCe op");
    let (v, h, b, s, vt, is_large, ignore) = fused[0];
    assert_eq!(v, 4096);
    assert_eq!(h, 128);
    assert_eq!(b, 2);
    assert_eq!(s, 64);
    assert_eq!(vt, 1024);
    assert!(
        !is_large,
        "V=4096 < {LARGE_VOCAB_THRESHOLD} must route to small-vocab path"
    );
    assert_eq!(ignore, -100, "default ignore_index must be -100");
}

// ─── Test 3: large vocab → is_large=true ─────────────────────────────────

#[test]
fn decorator_with_large_vocab_routes_to_two_kernel_path() {
    let src = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    return fused_linear_ce(x, w, bias, targets)
"#;
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(128),
        vocab_size: Some(49152),
        hidden_size: Some(512),
        batch_size: Some(1),
        seq_len: Some(64),
    };
    let list = extract_first_fn(src, Some(cfg)).expect("extraction must succeed");
    let is_large = list
        .ops
        .iter()
        .find_map(|op| match &op.op {
            PrimalOp::FusedLinearCe { is_large, .. } => Some(*is_large),
            _ => None,
        })
        .expect("expected FusedLinearCe");
    assert!(
        is_large,
        "V=49152 > {LARGE_VOCAB_THRESHOLD} must route to large-vocab path"
    );
}

// ─── Test 4: decorator with enabled=false → composite fallback ──────────

#[test]
fn decorator_enabled_false_falls_through_to_composite() {
    let src = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    return fused_linear_ce(x, w, bias, targets)
"#;
    let cfg = FusedCeDecoratorConfig {
        enabled: false,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(64),
    };
    let list = extract_first_fn(src, Some(cfg));
    if let Some(list) = list {
        let has_fused = list
            .ops
            .iter()
            .any(|op| matches!(op.op, PrimalOp::FusedLinearCe { .. }));
        assert!(
            !has_fused,
            "decorator with enabled=false must fall through to composite"
        );
    }
}

// ─── Test 5: decorator with missing shape hints → composite fallback ────

#[test]
fn decorator_missing_shape_hints_falls_through_to_composite() {
    let src = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    return fused_linear_ce(x, w, bias, targets)
"#;
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: None,
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(64),
    };
    let list = extract_first_fn(src, Some(cfg));
    if let Some(list) = list {
        let has_fused = list
            .ops
            .iter()
            .any(|op| matches!(op.op, PrimalOp::FusedLinearCe { .. }));
        assert!(
            !has_fused,
            "decorator with missing vocab_size must fall through to composite"
        );
    }
}

// ─── Test 6: AD rule produces three components with correct shapes ──────

#[test]
fn fused_linear_ce_ad_rule_emits_three_components() {
    let fwd_op = WengertOp {
        id: 0,
        result: 100,
        op: PrimalOp::FusedLinearCe {
            vocab_size: 49152,
            hidden_size: 512,
            batch_size: 8,
            seq_len: 128,
            vocab_tile: 256,
            ignore_index: -100,
            is_large: true,
        },
        inputs: vec![10, 11, 12, 13], // x, W, bias, targets
        saved_for_backward: false,
        checkpointed: false,
    };
    let adjoints = apply_ad_rule(&fwd_op, 200);
    assert_eq!(adjoints.len(), 3, "must emit dx, dW, dbias");
    let mut components_seen = std::collections::BTreeSet::new();
    for adj in &adjoints {
        match &adj.expr {
            AdjointExpr::FusedLinearCeBackward {
                grad,
                x,
                w,
                bias,
                targets,
                fwd_result,
                component,
                vocab_size,
                hidden_size,
                batch_size,
                seq_len,
                vocab_tile,
                ignore_index,
            } => {
                assert_eq!(*grad, 200);
                assert_eq!(*x, 10);
                assert_eq!(*w, 11);
                assert_eq!(*bias, 12);
                assert_eq!(*targets, 13);
                assert_eq!(*fwd_result, 100);
                assert_eq!(*vocab_size, 49152);
                assert_eq!(*hidden_size, 512);
                assert_eq!(*batch_size, 8);
                assert_eq!(*seq_len, 128);
                assert_eq!(*vocab_tile, 256);
                assert_eq!(*ignore_index, -100);
                components_seen.insert(*component);
            }
            other => panic!("unexpected adjoint expr: {:?}", other),
        }
    }
    let want: std::collections::BTreeSet<u8> = [0u8, 1, 2].into_iter().collect();
    assert_eq!(components_seen, want);
    let by_component: std::collections::HashMap<_, _> = adjoints
        .iter()
        .map(|a| {
            let c = match &a.expr {
                AdjointExpr::FusedLinearCeBackward { component, .. } => *component,
                _ => panic!(),
            };
            (c, a.input_var)
        })
        .collect();
    assert_eq!(by_component[&0], 10, "component 0 -> dx -> input_var x");
    assert_eq!(by_component[&1], 11, "component 1 -> dW -> input_var W");
    assert_eq!(by_component[&2], 12, "component 2 -> dbias -> input_var bias");
}

// ─── Test 7: routing threshold is exactly LARGE_VOCAB_THRESHOLD ─────────

#[test]
fn routing_threshold_at_8192_keeps_v1_path() {
    let src = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    return fused_linear_ce(x, w, bias, targets)
"#;
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(128),
        vocab_size: Some(LARGE_VOCAB_THRESHOLD),
        hidden_size: Some(128),
        batch_size: Some(1),
        seq_len: Some(64),
    };
    let list = extract_first_fn(src, Some(cfg)).expect("extraction must succeed");
    let is_large = list
        .ops
        .iter()
        .find_map(|op| match &op.op {
            PrimalOp::FusedLinearCe { is_large, .. } => Some(*is_large),
            _ => None,
        })
        .expect("expected FusedLinearCe at threshold");
    assert!(
        !is_large,
        "V == LARGE_VOCAB_THRESHOLD must route to v1 path (strict >)"
    );
}
