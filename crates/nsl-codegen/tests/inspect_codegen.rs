//! Phase 5 Task 7: verify predicate parsing and lowering wiring.
//!
//! Full end-to-end verification (JIT execution of the compiled stats/dump
//! hooks) is deferred to Task 9's smoke test.  These tests just check that
//! the predicate AST → lowering surfaces link and that predicate grammar
//! accepts the PDF §5.3 examples.

use nsl_codegen::inspect::predicate::{parse_predicate, CmpOp, PredicateExpr};

#[test]
fn parse_compound_and_lower_types_link() {
    let ast = parse_predicate("step > 500 and loss > 5.0").unwrap();
    match ast {
        PredicateExpr::And(_, _) => {}
        other => panic!("expected And, got {:?}", other),
    }
}

#[test]
fn health_ident_parses_into_cmp_variant() {
    let ast = parse_predicate("nan_inf_count_window > 0").unwrap();
    match ast {
        PredicateExpr::Cmp(_, CmpOp::Gt, _) => {}
        other => panic!("expected Cmp(_, Gt, _), got {:?}", other),
    }
}

#[test]
fn or_and_not_compose() {
    let ast = parse_predicate("not (loss_ema_slope < 0.0) or grad_norm_total > 100.0").unwrap();
    match ast {
        PredicateExpr::Or(_, _) => {}
        other => panic!("expected Or, got {:?}", other),
    }
}

#[test]
fn unknown_ident_rejected() {
    // Identifiers outside the allow-list are rejected at parse time so
    // codegen never sees them.
    assert!(parse_predicate("batch_idx > 10").is_err());
}

#[test]
fn lower_predicate_symbol_is_public() {
    // Link check — lower_predicate must be publicly reachable, otherwise
    // the codegen @inspect emitter cannot call it.  This test exercises
    // the symbol via a runtime value probe without constructing a
    // FunctionBuilder (which would require a full Cranelift context).
    let _lowerer: fn(
        &PredicateExpr,
        &mut cranelift_frontend::FunctionBuilder,
        &nsl_codegen::inspect::predicate::PredicateLowerCtx,
    ) -> cranelift_codegen::ir::Value = nsl_codegen::inspect::predicate::lower_predicate;
}
