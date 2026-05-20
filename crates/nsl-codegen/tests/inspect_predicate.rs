use nsl_codegen::inspect::predicate::{parse_predicate, CmpOp, PredicateExpr};

#[test]
fn parses_simple_int_comparison() {
    let p = parse_predicate("step > 500").unwrap();
    match p {
        PredicateExpr::Cmp(l, op, r) => {
            assert!(matches!(*l, PredicateExpr::Ident(ref s) if s == "step"));
            assert_eq!(op, CmpOp::Gt);
            assert!(matches!(*r, PredicateExpr::IntLit(500)));
        }
        _ => panic!("expected Cmp, got {:?}", p),
    }
}

#[test]
fn parses_float_comparison() {
    let p = parse_predicate("loss > 5.0").unwrap();
    match p {
        PredicateExpr::Cmp(_, op, r) => {
            assert_eq!(op, CmpOp::Gt);
            assert!(matches!(*r, PredicateExpr::FloatLit(v) if (v - 5.0).abs() < 1e-9));
        }
        _ => panic!(),
    }
}

#[test]
fn parses_and() {
    let p = parse_predicate("step > 500 and loss > 5.0").unwrap();
    assert!(matches!(p, PredicateExpr::And(_, _)));
}

#[test]
fn parses_or_with_lower_precedence() {
    // a and b or c => (a and b) or c
    let p = parse_predicate("step > 1 and step > 2 or loss > 0.0").unwrap();
    match p {
        PredicateExpr::Or(l, _) => {
            assert!(matches!(*l, PredicateExpr::And(_, _)));
        }
        _ => panic!("expected Or at root, got {:?}", p),
    }
}

#[test]
fn parses_not_with_highest_precedence() {
    // not a and b => (not a) and b
    let p = parse_predicate("not step > 1 and loss > 0.0").unwrap();
    match p {
        PredicateExpr::And(l, _) => {
            assert!(matches!(*l, PredicateExpr::Not(_)));
        }
        _ => panic!("expected And at root, got {:?}", p),
    }
}

#[test]
fn parses_parens() {
    let p = parse_predicate("not (step > 1 or loss > 0.0)").unwrap();
    assert!(matches!(p, PredicateExpr::Not(_)));
}

#[test]
fn rejects_unknown_identifier() {
    let err = parse_predicate("foo > 1").unwrap_err();
    assert!(err.contains("unknown identifier"), "got {:?}", err);
    assert!(
        err.contains("foo"),
        "should mention the bad ident: {:?}",
        err
    );
}

#[test]
fn rejects_truncated_expression() {
    assert!(parse_predicate("step >").is_err());
    assert!(parse_predicate("").is_err());
    assert!(parse_predicate("(step > 1").is_err());
}

#[test]
fn accepts_all_comparators() {
    for op in &[">", "<", ">=", "<=", "==", "!="] {
        let src = format!("step {} 1", op);
        assert!(parse_predicate(&src).is_ok(), "should parse {:?}", src);
    }
}

#[test]
fn accepts_all_health_idents() {
    for ident in &[
        "step",
        "loss",
        "loss_ema",
        "loss_ema_slope",
        "grad_norm_total",
        "nan_inf_count_window",
    ] {
        let src = format!("{} > 0", ident);
        assert!(parse_predicate(&src).is_ok(), "should parse {:?}", src);
    }
}
