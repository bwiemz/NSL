//! Compile-time AST evaluator for constructor-arg binding during
//! calibration pre-scan.
//!
//! Per the WGGO Phase 2 backward-pass calibration spec §4.7
//! ("Constructor-arg binding gap"), NSL's existing AWQ pre-scan requires
//! literal `IntLiteral`s in field initializers (e.g.
//! `randn([8, 8])`) — production transformers using `randn([dim, dim])`
//! patterns are not handled. This evaluator closes that gap by resolving
//! a small, closed set of expression shapes against a binding scope built
//! at the constructor-call site.
//!
//! Supported shapes (verified by grep over `examples/` + stdlib):
//!
//! 1. `IntLiteral(n)` → `Int(n)`
//! 2. `Ident(sym)` → scope lookup
//! 3. `BinaryOp { Add | Sub | Mul | Div | FloorDiv | Mod, .. }` over `Int`
//! 4. `ListLiteral([items..])` where each item evaluates to `Int` → `IntList`
//! 5. `Call { callee: Ident(fname), args }` for `fname` in the
//!    tensor-init set `{zeros, ones, randn, rand, full, arange}` —
//!    returns the result of evaluating `args[0].value` (the shape list)
//!
//! Anything else returns `EvalError::UnsupportedExpr`.

use std::collections::HashMap;

use nsl_ast::decl::Param;
use nsl_ast::expr::{Arg, Expr, ExprKind};
use nsl_ast::operator::{BinOp, UnaryOp};
use nsl_ast::{Span, Symbol};
use nsl_lexer::Interner;

/// Result of evaluating a field-initializer expression at calibration
/// pre-scan time.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalValue {
    /// Scalar integer (literal, identifier-bound integer, or arithmetic).
    Int(i64),
    /// Shape vector from a list literal or tensor-init call like
    /// `randn([dim, dim])`.
    IntList(Vec<i64>),
}

/// Reasons an expression cannot be evaluated. Each variant carries the
/// span of the offending sub-expression so the pre-scan can emit a
/// three-part diagnostic (`requested` / `expected` / `found`).
#[derive(Debug)]
pub enum EvalError {
    /// The `ExprKind` (or operator) is outside the supported subset.
    UnsupportedExpr {
        kind_summary: &'static str,
        span: Span,
    },
    /// An `Ident` had no entry in the supplied scope (or a constructor
    /// param had no matching arg, no positional fallback, and no default).
    UnboundIdent { name: String, span: Span },
    /// An operand was the wrong `EvalValue` kind for its context (e.g. a
    /// list used as the left side of an arithmetic operator, or a list
    /// element that did not evaluate to `Int`).
    TypeMismatch {
        expected: &'static str,
        got: &'static str,
        span: Span,
    },
    /// Right operand of `Div`, `FloorDiv`, or `Mod` was zero.
    DivByZero { span: Span },
    /// Integer arithmetic overflow during evaluation. `op` is a static
    /// label naming the offending operation (e.g. `"addition"`,
    /// `"negation"`) so a three-part diagnostic can render
    /// `requested`/`expected`/`found` consistently with the other
    /// `EvalError` variants.
    Overflow { span: Span, op: &'static str },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedExpr { kind_summary, .. } => {
                write!(
                    f,
                    "unsupported expression in calibration pre-scan: {kind_summary}"
                )
            }
            Self::UnboundIdent { name, .. } => {
                write!(
                    f,
                    "unbound identifier '{name}' in calibration pre-scan scope"
                )
            }
            Self::TypeMismatch { expected, got, .. } => {
                write!(
                    f,
                    "type mismatch in calibration pre-scan: expected {expected}, got {got}"
                )
            }
            Self::DivByZero { .. } => write!(f, "division by zero in calibration pre-scan"),
            Self::Overflow { op, .. } => {
                write!(f, "calibration pre-scan: integer overflow during {op}")
            }
        }
    }
}

impl std::error::Error for EvalError {}

/// Binding environment keyed by constructor-param symbol.
pub type Scope = HashMap<Symbol, EvalValue>;

/// Evaluate `expr` against `scope`. Returns `Int` for integer
/// expressions, `IntList` for list literals and tensor-init calls.
pub fn evaluate_expr(
    expr: &Expr,
    scope: &Scope,
    interner: &Interner,
) -> Result<EvalValue, EvalError> {
    match &expr.kind {
        ExprKind::IntLiteral(n) => Ok(EvalValue::Int(*n)),

        ExprKind::Ident(sym) => scope
            .get(sym)
            .cloned()
            .ok_or_else(|| EvalError::UnboundIdent {
                name: interner.resolve(sym.0).unwrap_or("?").to_string(),
                span: expr.span,
            }),

        ExprKind::BinaryOp { left, op, right } => {
            let l = expect_int(evaluate_expr(left, scope, interner)?, left.span)?;
            let r = expect_int(evaluate_expr(right, scope, interner)?, right.span)?;
            // `checked_*` everywhere: in debug builds raw `+/-/*` panics
            // on overflow (kills the compiler with no diagnostic span);
            // in release builds it wraps silently (produces wrong shape
            // values that propagate into far-away codegen errors). Both
            // are wrong for a user-controlled AST.
            let result = match op {
                BinOp::Add => l.checked_add(r).ok_or(EvalError::Overflow {
                    span: expr.span,
                    op: "addition",
                })?,
                BinOp::Sub => l.checked_sub(r).ok_or(EvalError::Overflow {
                    span: expr.span,
                    op: "subtraction",
                })?,
                BinOp::Mul => l.checked_mul(r).ok_or(EvalError::Overflow {
                    span: expr.span,
                    op: "multiplication",
                })?,
                BinOp::Div | BinOp::FloorDiv => {
                    if r == 0 {
                        return Err(EvalError::DivByZero { span: expr.span });
                    }
                    // `i64::checked_div` returns `None` only on the
                    // single overflow case `i64::MIN / -1`; the
                    // explicit `r == 0` check above still covers
                    // standard div-by-zero. `Div` and `FloorDiv` are
                    // equivalent here because this evaluator only
                    // operates on i64 — there is no float domain.
                    // NSL `/` over floats parses to the same `BinOp::Div`
                    // but a float operand would have been rejected as
                    // `UnsupportedExpr` in `evaluate_expr`'s float arm
                    // before reaching here, so we can safely treat
                    // `Div` and `FloorDiv` as integer-floor division.
                    l.checked_div(r).ok_or(EvalError::Overflow {
                        span: expr.span,
                        op: "division",
                    })?
                }
                BinOp::Mod => {
                    if r == 0 {
                        return Err(EvalError::DivByZero { span: expr.span });
                    }
                    l.checked_rem(r).ok_or(EvalError::Overflow {
                        span: expr.span,
                        op: "modulo",
                    })?
                }
                _ => {
                    return Err(EvalError::UnsupportedExpr {
                        kind_summary: describe_binop(*op),
                        span: expr.span,
                    });
                }
            };
            Ok(EvalValue::Int(result))
        }

        ExprKind::UnaryOp { op, operand } => {
            // `-1` parses as `UnaryOp { Neg, IntLiteral(1) }`, NOT as
            // `IntLiteral(-1)`. Without this arm, every negative literal
            // (`pad_id: int = -1`, `[batch, -1]`) refuses with
            // `UnsupportedExpr`. `Not` falls through to the catch-all.
            let v = evaluate_expr(operand, scope, interner)?;
            match (op, v) {
                (UnaryOp::Neg, EvalValue::Int(n)) => {
                    n.checked_neg()
                        .map(EvalValue::Int)
                        .ok_or(EvalError::Overflow {
                            span: expr.span,
                            op: "negation",
                        })
                }
                (UnaryOp::Neg, EvalValue::IntList(_)) => Err(EvalError::TypeMismatch {
                    expected: "int",
                    got: "list",
                    span: operand.span,
                }),
                _ => Err(EvalError::UnsupportedExpr {
                    kind_summary: "non-arithmetic unary op",
                    span: expr.span,
                }),
            }
        }

        ExprKind::Paren(inner) => evaluate_expr(inner, scope, interner),

        ExprKind::ListLiteral(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let v = evaluate_expr(item, scope, interner)?;
                out.push(expect_int(v, item.span)?);
            }
            Ok(EvalValue::IntList(out))
        }

        ExprKind::Call { callee, args } => {
            let ExprKind::Ident(callee_sym) = &callee.kind else {
                return Err(EvalError::UnsupportedExpr {
                    kind_summary: "non-identifier callee",
                    span: expr.span,
                });
            };
            let fname = interner.resolve(callee_sym.0).unwrap_or("");
            if !is_tensor_init(fname) {
                return Err(EvalError::UnsupportedExpr {
                    kind_summary: "non-tensor-init function call",
                    span: expr.span,
                });
            }
            let Some(first) = args.first() else {
                return Err(EvalError::UnsupportedExpr {
                    kind_summary: "tensor-init with no shape argument",
                    span: expr.span,
                });
            };
            evaluate_expr(&first.value, scope, interner)
        }

        _ => Err(EvalError::UnsupportedExpr {
            kind_summary: describe_expr_kind(&expr.kind),
            span: expr.span,
        }),
    }
}

/// Bind constructor args at an instantiation site.
///
/// For each `Param` in declaration order, pick a value expression by:
/// 1. matching name against any keyword-style `Arg` (`Arg.name = Some(_)`),
///    else
/// 2. taking the next positional `Arg` (`Arg.name = None`), else
/// 3. using `param.default` if present, else
/// 4. returning `EvalError::UnboundIdent`.
///
/// Each value is evaluated against the scope built from earlier params,
/// so a default like `num_heads: int = dim / 128` can reference an
/// earlier `dim` param.
pub fn bind_constructor_args(
    params: &[Param],
    args: &[Arg],
    interner: &Interner,
) -> Result<Scope, EvalError> {
    let mut scope = Scope::new();

    let mut by_name: HashMap<Symbol, &Expr> = HashMap::new();
    for arg in args {
        if let Some(name) = arg.name {
            by_name.insert(name, &arg.value);
        }
    }
    let positional: Vec<&Expr> = args
        .iter()
        .filter(|a| a.name.is_none())
        .map(|a| &a.value)
        .collect();
    let mut positional_idx: usize = 0;

    for param in params {
        let value_expr: &Expr = if let Some(e) = by_name.get(&param.name) {
            e
        } else if positional_idx < positional.len() {
            let e = positional[positional_idx];
            positional_idx += 1;
            e
        } else if let Some(d) = &param.default {
            d
        } else {
            return Err(EvalError::UnboundIdent {
                name: interner.resolve(param.name.0).unwrap_or("?").to_string(),
                span: param.span,
            });
        };

        let value = evaluate_expr(value_expr, &scope, interner)?;
        scope.insert(param.name, value);
    }

    Ok(scope)
}

// ── helpers ────────────────────────────────────────────────────────────

fn expect_int(v: EvalValue, span: Span) -> Result<i64, EvalError> {
    match v {
        EvalValue::Int(n) => Ok(n),
        EvalValue::IntList(_) => Err(EvalError::TypeMismatch {
            expected: "int",
            got: "list",
            span,
        }),
    }
}

fn is_tensor_init(name: &str) -> bool {
    matches!(
        name,
        "zeros" | "ones" | "randn" | "rand" | "full" | "arange"
    )
}

/// Stable, exhaustive label for every `BinOp` variant. Matches the
/// "exhaustive-variant pattern" used by Task 3's `describe_expr_kind`
/// so a new operator added to `BinOp` becomes a hard compile error
/// here, not a silent fallthrough.
fn describe_binop(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
        BinOp::FloorDiv => "floordiv",
        BinOp::Mod => "mod",
        BinOp::Pow => "non-arithmetic binary op (pow)",
        BinOp::MatMul => "non-arithmetic binary op (matmul)",
        BinOp::Eq => "non-arithmetic binary op (eq)",
        BinOp::NotEq => "non-arithmetic binary op (noteq)",
        BinOp::Lt => "non-arithmetic binary op (lt)",
        BinOp::Gt => "non-arithmetic binary op (gt)",
        BinOp::LtEq => "non-arithmetic binary op (lteq)",
        BinOp::GtEq => "non-arithmetic binary op (gteq)",
        BinOp::And => "non-arithmetic binary op (and)",
        BinOp::Or => "non-arithmetic binary op (or)",
        BinOp::Is => "non-arithmetic binary op (is)",
        BinOp::In => "non-arithmetic binary op (in)",
        BinOp::BitOr => "non-arithmetic binary op (bitor)",
        BinOp::BitAnd => "non-arithmetic binary op (bitand)",
    }
}

/// Stable label for every `ExprKind` variant outside the supported
/// subset. The supported variants (`IntLiteral`, `Ident`, `BinaryOp`,
/// `ListLiteral`, `Call`) are matched directly above and never reach
/// this function — they are explicitly listed here only so adding a
/// new variant to `ExprKind` is a hard compile error.
fn describe_expr_kind(kind: &ExprKind) -> &'static str {
    match kind {
        ExprKind::IntLiteral(_) => "int literal (supported)",
        ExprKind::FloatLiteral(_) => "float literal",
        ExprKind::StringLiteral(_) => "string literal",
        ExprKind::FString(_) => "f-string",
        ExprKind::BoolLiteral(_) => "bool literal",
        ExprKind::NoneLiteral => "none literal",
        ExprKind::ListLiteral(_) => "list literal (supported)",
        ExprKind::TupleLiteral(_) => "tuple literal",
        ExprKind::DictLiteral(_) => "dict literal",
        ExprKind::Ident(_) => "identifier (supported)",
        ExprKind::SelfRef => "self reference",
        ExprKind::BinaryOp { .. } => "binary op (supported)",
        ExprKind::UnaryOp { .. } => "unary op",
        ExprKind::Pipe { .. } => "pipe",
        ExprKind::MemberAccess { .. } => "member access",
        ExprKind::Subscript { .. } => "subscript",
        ExprKind::Call { .. } => "call (supported subset)",
        ExprKind::Lambda { .. } => "lambda",
        ExprKind::BlockExpr(_) => "block expression",
        ExprKind::ListComp { .. } => "list comprehension",
        ExprKind::IfExpr { .. } => "if expression",
        ExprKind::MatchExpr { .. } => "match expression",
        ExprKind::Range { .. } => "range",
        ExprKind::Paren(_) => "parenthesized expression",
        ExprKind::Await(_) => "await expression",
        ExprKind::Error => "parse error expression",
    }
}

// ── tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::expr::Arg;
    use nsl_ast::{NodeId, Span};

    fn dummy_span() -> Span {
        Span::dummy()
    }

    fn make_expr(kind: ExprKind) -> Expr {
        Expr {
            kind,
            span: dummy_span(),
            id: NodeId::next(),
        }
    }

    fn intern(interner: &mut Interner, s: &str) -> Symbol {
        Symbol(interner.get_or_intern(s))
    }

    fn int_lit(n: i64) -> Expr {
        make_expr(ExprKind::IntLiteral(n))
    }

    fn ident(sym: Symbol) -> Expr {
        make_expr(ExprKind::Ident(sym))
    }

    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        make_expr(ExprKind::BinaryOp {
            left: Box::new(l),
            op,
            right: Box::new(r),
        })
    }

    fn call(callee_sym: Symbol, args: Vec<Expr>) -> Expr {
        let arg_vec = args
            .into_iter()
            .map(|value| Arg {
                name: None,
                value,
                span: dummy_span(),
            })
            .collect();
        make_expr(ExprKind::Call {
            callee: Box::new(make_expr(ExprKind::Ident(callee_sym))),
            args: arg_vec,
        })
    }

    fn list_lit(items: Vec<Expr>) -> Expr {
        make_expr(ExprKind::ListLiteral(items))
    }

    fn unary_neg(operand: Expr) -> Expr {
        make_expr(ExprKind::UnaryOp {
            op: UnaryOp::Neg,
            operand: Box::new(operand),
        })
    }

    fn paren(inner: Expr) -> Expr {
        make_expr(ExprKind::Paren(Box::new(inner)))
    }

    // ── evaluate_expr coverage ────────────────────────────────────────

    #[test]
    fn evaluates_int_literal() {
        let interner = Interner::new();
        let scope = Scope::new();
        assert_eq!(
            evaluate_expr(&int_lit(42), &scope, &interner).unwrap(),
            EvalValue::Int(42)
        );
    }

    #[test]
    fn evaluates_add() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = bin(BinOp::Add, int_lit(2), int_lit(3));
        assert_eq!(
            evaluate_expr(&e, &scope, &interner).unwrap(),
            EvalValue::Int(5)
        );
    }

    #[test]
    fn evaluates_sub() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = bin(BinOp::Sub, int_lit(10), int_lit(4));
        assert_eq!(
            evaluate_expr(&e, &scope, &interner).unwrap(),
            EvalValue::Int(6)
        );
    }

    #[test]
    fn evaluates_mul() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = bin(BinOp::Mul, int_lit(3), int_lit(4));
        assert_eq!(
            evaluate_expr(&e, &scope, &interner).unwrap(),
            EvalValue::Int(12)
        );
    }

    #[test]
    fn evaluates_floor_div() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = bin(BinOp::FloorDiv, int_lit(64), int_lit(8));
        assert_eq!(
            evaluate_expr(&e, &scope, &interner).unwrap(),
            EvalValue::Int(8)
        );
    }

    #[test]
    fn evaluates_div_as_integer_floor() {
        let interner = Interner::new();
        let scope = Scope::new();
        // 10 / 3 -> 3 in i64 floor semantics.
        let e = bin(BinOp::Div, int_lit(10), int_lit(3));
        assert_eq!(
            evaluate_expr(&e, &scope, &interner).unwrap(),
            EvalValue::Int(3)
        );
    }

    #[test]
    fn evaluates_mod() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = bin(BinOp::Mod, int_lit(17), int_lit(5));
        assert_eq!(
            evaluate_expr(&e, &scope, &interner).unwrap(),
            EvalValue::Int(2)
        );
    }

    #[test]
    fn unsupported_op_errors() {
        let interner = Interner::new();
        let scope = Scope::new();
        // Equality: not supported by the evaluator.
        let e = bin(BinOp::Eq, int_lit(1), int_lit(2));
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::UnsupportedExpr { kind_summary, .. }) => {
                assert!(
                    kind_summary.contains("eq"),
                    "expected error to mention 'eq', got {kind_summary:?}"
                );
            }
            other => panic!("expected UnsupportedExpr for Eq, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_pow_errors() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = bin(BinOp::Pow, int_lit(2), int_lit(8));
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::UnsupportedExpr { kind_summary, .. }) => {
                assert!(kind_summary.contains("pow"));
            }
            other => panic!("expected UnsupportedExpr for Pow, got {other:?}"),
        }
    }

    #[test]
    fn unbound_ident_errors() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let scope = Scope::new();
        let e = ident(dim);
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::UnboundIdent { name, .. }) => assert_eq!(name, "dim"),
            other => panic!("expected UnboundIdent, got {other:?}"),
        }
    }

    #[test]
    fn evaluates_int_list() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = list_lit(vec![int_lit(1), int_lit(2), int_lit(3)]);
        assert_eq!(
            evaluate_expr(&e, &scope, &interner).unwrap(),
            EvalValue::IntList(vec![1, 2, 3])
        );
    }

    #[test]
    fn list_with_non_int_element_errors() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let mut scope = Scope::new();
        // Bind dim to an IntList — using it as a list element should be
        // a TypeMismatch, not a silent flatten.
        scope.insert(dim, EvalValue::IntList(vec![4, 4]));
        let e = list_lit(vec![ident(dim)]);
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::TypeMismatch { expected, got, .. }) => {
                assert_eq!(expected, "int");
                assert_eq!(got, "list");
            }
            other => panic!("expected TypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn evaluates_randn_shape_with_bound_dim() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let randn = intern(&mut interner, "randn");
        let mut scope = Scope::new();
        scope.insert(dim, EvalValue::Int(4096));
        // randn([dim, dim])
        let shape = list_lit(vec![ident(dim), ident(dim)]);
        let e = call(randn, vec![shape]);
        assert_eq!(
            evaluate_expr(&e, &scope, &interner).unwrap(),
            EvalValue::IntList(vec![4096, 4096])
        );
    }

    #[test]
    fn evaluates_zeros_with_constructor_param() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let zeros = intern(&mut interner, "zeros");
        let mut scope = Scope::new();
        scope.insert(dim, EvalValue::Int(8));
        // zeros([dim, dim])
        let shape = list_lit(vec![ident(dim), ident(dim)]);
        let e = call(zeros, vec![shape]);
        assert_eq!(
            evaluate_expr(&e, &scope, &interner).unwrap(),
            EvalValue::IntList(vec![8, 8])
        );
    }

    #[test]
    fn evaluates_each_tensor_init_name() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let mut scope = Scope::new();
        scope.insert(dim, EvalValue::Int(2));
        for name in &["zeros", "ones", "randn", "rand", "full", "arange"] {
            let fname = intern(&mut interner, name);
            let shape = list_lit(vec![ident(dim), ident(dim)]);
            let e = call(fname, vec![shape]);
            let got = evaluate_expr(&e, &scope, &interner).unwrap();
            assert_eq!(
                got,
                EvalValue::IntList(vec![2, 2]),
                "tensor-init {name} should return shape list"
            );
        }
    }

    #[test]
    fn evaluates_head_dim_field_init() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let num_heads = intern(&mut interner, "num_heads");
        let mut scope = Scope::new();
        scope.insert(dim, EvalValue::Int(4096));
        scope.insert(num_heads, EvalValue::Int(32));
        // dim // num_heads
        let e = bin(BinOp::FloorDiv, ident(dim), ident(num_heads));
        assert_eq!(
            evaluate_expr(&e, &scope, &interner).unwrap(),
            EvalValue::Int(128)
        );
    }

    #[test]
    fn unsupported_callee_kind_errors() {
        let interner = Interner::new();
        let scope = Scope::new();
        // Call where callee is itself an IntLiteral (parser wouldn't
        // emit this, but the evaluator must refuse rather than panic).
        let weird = make_expr(ExprKind::Call {
            callee: Box::new(int_lit(7)),
            args: vec![],
        });
        match evaluate_expr(&weird, &scope, &interner) {
            Err(EvalError::UnsupportedExpr { kind_summary, .. }) => {
                assert!(kind_summary.contains("non-identifier callee"));
            }
            other => panic!("expected UnsupportedExpr, got {other:?}"),
        }
    }

    #[test]
    fn unknown_function_call_errors() {
        let mut interner = Interner::new();
        let custom = intern(&mut interner, "my_custom_init");
        let scope = Scope::new();
        let e = call(custom, vec![list_lit(vec![int_lit(2)])]);
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::UnsupportedExpr { kind_summary, .. }) => {
                assert!(kind_summary.contains("non-tensor-init"));
            }
            other => panic!("expected UnsupportedExpr for unknown call, got {other:?}"),
        }
    }

    #[test]
    fn tensor_init_without_args_errors() {
        let mut interner = Interner::new();
        let randn = intern(&mut interner, "randn");
        let scope = Scope::new();
        let e = make_expr(ExprKind::Call {
            callee: Box::new(make_expr(ExprKind::Ident(randn))),
            args: vec![],
        });
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::UnsupportedExpr { kind_summary, .. }) => {
                assert!(kind_summary.contains("no shape argument"));
            }
            other => panic!("expected UnsupportedExpr for arg-less randn, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_expr_kind_float_errors() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = make_expr(ExprKind::FloatLiteral(2.5));
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::UnsupportedExpr { kind_summary, .. }) => {
                assert!(kind_summary.contains("float"));
            }
            other => panic!("expected UnsupportedExpr for float, got {other:?}"),
        }
    }

    #[test]
    fn div_by_zero_errors() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = bin(BinOp::Div, int_lit(4), int_lit(0));
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::DivByZero { .. }) => {}
            other => panic!("expected DivByZero, got {other:?}"),
        }
    }

    #[test]
    fn floor_div_by_zero_errors() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = bin(BinOp::FloorDiv, int_lit(4), int_lit(0));
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::DivByZero { .. }) => {}
            other => panic!("expected DivByZero for FloorDiv, got {other:?}"),
        }
    }

    #[test]
    fn mod_by_zero_errors() {
        let interner = Interner::new();
        let scope = Scope::new();
        let e = bin(BinOp::Mod, int_lit(4), int_lit(0));
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::DivByZero { .. }) => {}
            other => panic!("expected DivByZero for Mod, got {other:?}"),
        }
    }

    #[test]
    fn arithmetic_on_list_left_errors() {
        let mut interner = Interner::new();
        let v = intern(&mut interner, "v");
        let mut scope = Scope::new();
        scope.insert(v, EvalValue::IntList(vec![1, 2]));
        let e = bin(BinOp::Add, ident(v), int_lit(1));
        match evaluate_expr(&e, &scope, &interner) {
            Err(EvalError::TypeMismatch { expected, got, .. }) => {
                assert_eq!(expected, "int");
                assert_eq!(got, "list");
            }
            other => panic!("expected TypeMismatch, got {other:?}"),
        }
    }

    // ── overflow / Paren / UnaryOp coverage (Tasks I-1, I-2, I-3) ─────

    #[test]
    fn arithmetic_overflow_errors() {
        // i64::MAX + 1 must surface as `Overflow`, NOT panic (debug) or
        // wrap silently (release).
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = bin(BinOp::Add, int_lit(i64::MAX), int_lit(1));
        match evaluate_expr(&expr, &scope, &interner) {
            Err(EvalError::Overflow { op, .. }) => assert_eq!(op, "addition"),
            other => panic!("expected Overflow, got {other:?}"),
        }
    }

    #[test]
    fn multiplication_overflow_errors() {
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = bin(BinOp::Mul, int_lit(i64::MAX), int_lit(2));
        match evaluate_expr(&expr, &scope, &interner) {
            Err(EvalError::Overflow { op, .. }) => assert_eq!(op, "multiplication"),
            other => panic!("expected Overflow for multiplication, got {other:?}"),
        }
    }

    #[test]
    fn subtraction_overflow_errors() {
        // i64::MIN - 1 must surface as `Overflow`.  Mirrors the
        // addition test — `Sub` and `Add` go through different
        // `checked_*` arms.
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = bin(BinOp::Sub, int_lit(i64::MIN), int_lit(1));
        match evaluate_expr(&expr, &scope, &interner) {
            Err(EvalError::Overflow { op, .. }) => assert_eq!(op, "subtraction"),
            other => panic!("expected Overflow for subtraction, got {other:?}"),
        }
    }

    #[test]
    fn division_overflow_errors() {
        // The single `i64::checked_div` overflow case: `i64::MIN / -1`
        // (the result would be `i64::MAX + 1`, which doesn't fit).
        // Must surface as `Overflow`, NOT `DivByZero` (`r != 0` here).
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = bin(BinOp::Div, int_lit(i64::MIN), int_lit(-1));
        match evaluate_expr(&expr, &scope, &interner) {
            Err(EvalError::Overflow { op, .. }) => assert_eq!(op, "division"),
            other => panic!("expected Overflow for division, got {other:?}"),
        }
    }

    #[test]
    fn floor_division_overflow_errors() {
        // Same overflow case as `Div` since the evaluator collapses
        // `Div` and `FloorDiv` to integer-floor `checked_div`.
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = bin(BinOp::FloorDiv, int_lit(i64::MIN), int_lit(-1));
        match evaluate_expr(&expr, &scope, &interner) {
            Err(EvalError::Overflow { op, .. }) => assert_eq!(op, "division"),
            other => panic!("expected Overflow for floor division, got {other:?}"),
        }
    }

    #[test]
    fn negation_overflow_errors() {
        // i64::MIN.checked_neg() is None — only overflow case for neg.
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = unary_neg(int_lit(i64::MIN));
        assert!(matches!(
            evaluate_expr(&expr, &scope, &interner),
            Err(EvalError::Overflow { op: "negation", .. })
        ));
    }

    #[test]
    fn paren_unwraps_inner_expr() {
        // `(3 + 4)` must evaluate identically to `3 + 4`.
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = paren(bin(BinOp::Add, int_lit(3), int_lit(4)));
        assert_eq!(
            evaluate_expr(&expr, &scope, &interner).unwrap(),
            EvalValue::Int(7)
        );
    }

    #[test]
    fn paren_around_list_preserves_list() {
        // `([1, 2, 3])` should yield the IntList unchanged.
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = paren(list_lit(vec![int_lit(1), int_lit(2), int_lit(3)]));
        assert_eq!(
            evaluate_expr(&expr, &scope, &interner).unwrap(),
            EvalValue::IntList(vec![1, 2, 3])
        );
    }

    #[test]
    fn unary_neg_negates_int() {
        // `-5` parses as UnaryOp{Neg, IntLiteral(5)} — must yield -5.
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = unary_neg(int_lit(5));
        assert_eq!(
            evaluate_expr(&expr, &scope, &interner).unwrap(),
            EvalValue::Int(-5)
        );
    }

    #[test]
    fn unary_neg_in_list_evaluates() {
        // `[batch, -1]` is a common reshape pattern — UnaryOp::Neg
        // inside a list literal must evaluate to -1, not refuse.
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = list_lit(vec![int_lit(2), unary_neg(int_lit(1))]);
        assert_eq!(
            evaluate_expr(&expr, &scope, &interner).unwrap(),
            EvalValue::IntList(vec![2, -1])
        );
    }

    #[test]
    fn unary_neg_on_list_errors() {
        // Negating a list value is a TypeMismatch, not silent dropping.
        let mut interner = Interner::new();
        let v = intern(&mut interner, "v");
        let mut scope = Scope::new();
        scope.insert(v, EvalValue::IntList(vec![1, 2]));
        let expr = unary_neg(ident(v));
        assert!(matches!(
            evaluate_expr(&expr, &scope, &interner),
            Err(EvalError::TypeMismatch {
                expected: "int",
                got: "list",
                ..
            })
        ));
    }

    #[test]
    fn unary_not_unsupported() {
        // `not x` falls through to UnsupportedExpr — the calibration
        // pre-scan never sees boolean expressions.
        let interner = Interner::new();
        let scope = Scope::new();
        let expr = make_expr(ExprKind::UnaryOp {
            op: UnaryOp::Not,
            operand: Box::new(int_lit(0)),
        });
        match evaluate_expr(&expr, &scope, &interner) {
            Err(EvalError::UnsupportedExpr { kind_summary, .. }) => {
                assert!(kind_summary.contains("non-arithmetic unary op"));
            }
            other => panic!("expected UnsupportedExpr for UnaryOp::Not, got {other:?}"),
        }
    }

    // ── bind_constructor_args coverage ────────────────────────────────

    fn make_param(name: Symbol, default: Option<Expr>) -> Param {
        Param {
            name,
            type_ann: None,
            default,
            is_variadic: false,
            span: dummy_span(),
        }
    }

    fn arg_named(name: Symbol, value: Expr) -> Arg {
        Arg {
            name: Some(name),
            value,
            span: dummy_span(),
        }
    }

    fn arg_pos(value: Expr) -> Arg {
        Arg {
            name: None,
            value,
            span: dummy_span(),
        }
    }

    #[test]
    fn bind_constructor_args_named() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let num_heads = intern(&mut interner, "num_heads");
        let params = vec![make_param(dim, None), make_param(num_heads, None)];
        let args = vec![
            arg_named(dim, int_lit(4096)),
            arg_named(num_heads, int_lit(32)),
        ];
        let scope = bind_constructor_args(&params, &args, &interner).unwrap();
        assert_eq!(scope.get(&dim), Some(&EvalValue::Int(4096)));
        assert_eq!(scope.get(&num_heads), Some(&EvalValue::Int(32)));
    }

    #[test]
    fn bind_constructor_args_positional() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let num_heads = intern(&mut interner, "num_heads");
        let params = vec![make_param(dim, None), make_param(num_heads, None)];
        let args = vec![arg_pos(int_lit(4096)), arg_pos(int_lit(32))];
        let scope = bind_constructor_args(&params, &args, &interner).unwrap();
        assert_eq!(scope.get(&dim), Some(&EvalValue::Int(4096)));
        assert_eq!(scope.get(&num_heads), Some(&EvalValue::Int(32)));
    }

    #[test]
    fn bind_constructor_args_mixed_named_and_positional() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let num_heads = intern(&mut interner, "num_heads");
        let params = vec![make_param(dim, None), make_param(num_heads, None)];
        // Named for the SECOND param; positional for the first.
        let args = vec![arg_named(num_heads, int_lit(8)), arg_pos(int_lit(512))];
        let scope = bind_constructor_args(&params, &args, &interner).unwrap();
        assert_eq!(scope.get(&dim), Some(&EvalValue::Int(512)));
        assert_eq!(scope.get(&num_heads), Some(&EvalValue::Int(8)));
    }

    #[test]
    fn bind_constructor_args_missing_no_default() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let params = vec![make_param(dim, None)];
        let args: Vec<Arg> = vec![];
        match bind_constructor_args(&params, &args, &interner) {
            Err(EvalError::UnboundIdent { name, .. }) => assert_eq!(name, "dim"),
            other => panic!("expected UnboundIdent, got {other:?}"),
        }
    }

    #[test]
    fn bind_constructor_args_uses_default() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let params = vec![make_param(dim, Some(int_lit(1024)))];
        let args: Vec<Arg> = vec![];
        let scope = bind_constructor_args(&params, &args, &interner).unwrap();
        assert_eq!(scope.get(&dim), Some(&EvalValue::Int(1024)));
    }

    #[test]
    fn bind_default_can_reference_earlier_param() {
        // Attention(dim: int = 4096, num_heads: int = dim // 128)
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let num_heads = intern(&mut interner, "num_heads");
        let params = vec![
            make_param(dim, Some(int_lit(4096))),
            make_param(
                num_heads,
                Some(bin(BinOp::FloorDiv, ident(dim), int_lit(128))),
            ),
        ];
        let args: Vec<Arg> = vec![];
        let scope = bind_constructor_args(&params, &args, &interner).unwrap();
        assert_eq!(scope.get(&dim), Some(&EvalValue::Int(4096)));
        assert_eq!(scope.get(&num_heads), Some(&EvalValue::Int(32)));
    }

    #[test]
    fn bind_named_arg_overrides_default() {
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let params = vec![make_param(dim, Some(int_lit(1024)))];
        let args = vec![arg_named(dim, int_lit(2048))];
        let scope = bind_constructor_args(&params, &args, &interner).unwrap();
        assert_eq!(scope.get(&dim), Some(&EvalValue::Int(2048)));
    }

    #[test]
    fn bind_evaluates_arg_expression_against_earlier_params() {
        // Attention(dim: int, head_dim: int) called with
        // dim=4096, head_dim=dim // 32 (positional).
        let mut interner = Interner::new();
        let dim = intern(&mut interner, "dim");
        let head_dim = intern(&mut interner, "head_dim");
        let params = vec![make_param(dim, None), make_param(head_dim, None)];
        let args = vec![
            arg_pos(int_lit(4096)),
            arg_pos(bin(BinOp::FloorDiv, ident(dim), int_lit(32))),
        ];
        let scope = bind_constructor_args(&params, &args, &interner).unwrap();
        assert_eq!(scope.get(&dim), Some(&EvalValue::Int(4096)));
        assert_eq!(scope.get(&head_dim), Some(&EvalValue::Int(128)));
    }
}
