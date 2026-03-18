use nsl_ast::expr::*;
use nsl_ast::operator::UnaryOp;
use nsl_errors::Diagnostic;
use nsl_lexer::TokenKind;

use crate::parser::Parser;
use crate::pratt;

/// Parse an expression with minimum binding power 0.
pub fn parse_expr(p: &mut Parser) -> Expr {
    parse_expr_bp(p, 0)
}

/// Parse an expression using Pratt parsing with the given minimum binding power.
fn parse_expr_bp(p: &mut Parser, min_bp: u8) -> Expr {
    let mut lhs = parse_prefix_or_atom(p);

    loop {
        // Try postfix (member access, subscript, call)
        if let Some(bp) = pratt::postfix_binding_power(p.peek()) {
            if bp < min_bp {
                break;
            }
            lhs = parse_postfix(p, lhs);
            continue;
        }

        // Try pipe operator specially
        if let TokenKind::Pipe = p.peek() {
            if let Some((l_bp, r_bp)) = pratt::infix_binding_power(p.peek()) {
                if l_bp < min_bp {
                    break;
                }
                let start = lhs.span;
                p.advance(); // consume |>
                let rhs = parse_expr_bp(p, r_bp);
                let span = start.merge(rhs.span);
                lhs = Expr {
                    kind: ExprKind::Pipe {
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    },
                    span,
                    id: p.next_node_id(),
                };
                continue;
            }
        }

        // Try range operators
        if matches!(p.peek(), TokenKind::DotDot | TokenKind::DotDotEq) {
            if let Some((l_bp, r_bp)) = pratt::infix_binding_power(p.peek()) {
                if l_bp < min_bp {
                    break;
                }
                let start = lhs.span;
                let inclusive = matches!(p.peek(), &TokenKind::DotDotEq);
                p.advance();
                // Range end is optional
                let end = if can_start_expr(p.peek()) {
                    Some(Box::new(parse_expr_bp(p, r_bp)))
                } else {
                    None
                };
                let span = end
                    .as_ref()
                    .map(|e| start.merge(e.span))
                    .unwrap_or(start);
                lhs = Expr {
                    kind: ExprKind::Range {
                        start: Some(Box::new(lhs)),
                        end,
                        inclusive,
                    },
                    span,
                    id: p.next_node_id(),
                };
                continue;
            }
        }

        // Try infix
        if let Some((l_bp, r_bp)) = pratt::infix_binding_power(p.peek()) {
            if l_bp < min_bp {
                break;
            }
            let op_kind = p.peek().clone();
            if let Some(op) = pratt::token_to_binop(&op_kind) {
                let start = lhs.span;
                p.advance();
                let rhs = parse_expr_bp(p, r_bp);
                let span = start.merge(rhs.span);
                lhs = Expr {
                    kind: ExprKind::BinaryOp {
                        left: Box::new(lhs),
                        op,
                        right: Box::new(rhs),
                    },
                    span,
                    id: p.next_node_id(),
                };
                continue;
            }
        }

        break;
    }

    lhs
}

fn parse_prefix_or_atom(p: &mut Parser) -> Expr {
    match p.peek().clone() {
        // Unary minus
        TokenKind::Minus => {
            let start = p.current_span();
            p.advance();
            let bp = pratt::prefix_binding_power(&TokenKind::Minus).unwrap();
            let operand = parse_expr_bp(p, bp);
            let span = start.merge(operand.span);
            Expr {
                kind: ExprKind::UnaryOp {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                },
                span,
                id: p.next_node_id(),
            }
        }

        // Logical not
        TokenKind::Not => {
            let start = p.current_span();
            p.advance();
            let bp = pratt::prefix_binding_power(&TokenKind::Not).unwrap();
            let operand = parse_expr_bp(p, bp);
            let span = start.merge(operand.span);
            Expr {
                kind: ExprKind::UnaryOp {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                },
                span,
                id: p.next_node_id(),
            }
        }

        // Literals
        TokenKind::IntLiteral(v) => {
            let span = p.advance().span;
            Expr {
                kind: ExprKind::IntLiteral(v),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::FloatLiteral(v) => {
            let span = p.advance().span;
            Expr {
                kind: ExprKind::FloatLiteral(v),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::StringLiteral(s) => {
            let s = s.clone();
            let span = p.advance().span;
            Expr {
                kind: ExprKind::StringLiteral(s),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::FStringStart => parse_fstring(p),
        TokenKind::True => {
            let span = p.advance().span;
            Expr {
                kind: ExprKind::BoolLiteral(true),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::False => {
            let span = p.advance().span;
            Expr {
                kind: ExprKind::BoolLiteral(false),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::None => {
            let span = p.advance().span;
            Expr {
                kind: ExprKind::NoneLiteral,
                span,
                id: p.next_node_id(),
            }
        }

        // Identifiers
        TokenKind::Ident(_) => {
            let (sym, span) = p.expect_ident();
            Expr {
                kind: ExprKind::Ident(sym),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::SelfKw => {
            let span = p.advance().span;
            Expr {
                kind: ExprKind::SelfRef,
                span,
                id: p.next_node_id(),
            }
        }

        // Parenthesized expression or tuple
        TokenKind::LeftParen => parse_paren_or_tuple(p),

        // List literal or comprehension
        TokenKind::LeftBracket => parse_list_or_comp(p),

        // Dict literal
        TokenKind::LeftBrace => parse_dict(p),

        // Lambda: |params| expr
        TokenKind::Bar => parse_lambda(p),

        // If expression
        TokenKind::If => parse_if_expr(p),

        // Await expression
        TokenKind::Await => {
            let start = p.current_span();
            p.advance();
            let expr = parse_expr_bp(p, 27);
            let span = start.merge(expr.span);
            Expr {
                kind: ExprKind::Await(Box::new(expr)),
                span,
                id: p.next_node_id(),
            }
        }

        _ => {
            let span = p.current_span();
            p.diagnostics.push(
                Diagnostic::error(format!("expected expression, found {}", p.peek()))
                    .with_label(span, "expected expression"),
            );
            p.advance();
            Expr {
                kind: ExprKind::Error,
                span,
                id: p.next_node_id(),
            }
        }
    }
}

fn parse_postfix(p: &mut Parser, lhs: Expr) -> Expr {
    match p.peek() {
        TokenKind::Dot => {
            let start = lhs.span;
            p.advance(); // consume .
            let (member, member_span) = p.expect_ident();
            let span = start.merge(member_span);
            Expr {
                kind: ExprKind::MemberAccess {
                    object: Box::new(lhs),
                    member,
                },
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::LeftBracket => {
            let start = lhs.span;
            p.advance(); // consume [
            let index = parse_subscript(p);
            let end = p.expect(&TokenKind::RightBracket);
            let span = start.merge(end);
            Expr {
                kind: ExprKind::Subscript {
                    object: Box::new(lhs),
                    index: Box::new(index),
                },
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::LeftParen => {
            let start = lhs.span;
            p.advance(); // consume (
            let args = parse_args(p);
            let end = p.expect(&TokenKind::RightParen);
            let span = start.merge(end);
            Expr {
                kind: ExprKind::Call {
                    callee: Box::new(lhs),
                    args,
                },
                span,
                id: p.next_node_id(),
            }
        }
        _ => lhs,
    }
}

fn parse_subscript(p: &mut Parser) -> SubscriptKind {
    // Could be: index, slice (a:b:c), or multi-dim (a, b:c)
    let mut dims = Vec::new();

    loop {
        let dim = parse_single_subscript(p);
        dims.push(dim);

        if !p.eat(&TokenKind::Comma) {
            break;
        }
    }

    if dims.len() == 1 {
        dims.into_iter().next().unwrap()
    } else {
        SubscriptKind::MultiDim(dims)
    }
}

fn parse_single_subscript(p: &mut Parser) -> SubscriptKind {
    // Check for slice syntax: :, a:, :b, a:b, a:b:c
    if p.at(&TokenKind::Colon) {
        p.advance();
        let upper = if can_start_expr(p.peek()) && !p.at(&TokenKind::Colon) {
            Some(parse_expr(p))
        } else {
            None
        };
        let step = if p.eat(&TokenKind::Colon) {
            if can_start_expr(p.peek()) {
                Some(parse_expr(p))
            } else {
                None
            }
        } else {
            None
        };
        return SubscriptKind::Slice {
            lower: None,
            upper,
            step,
        };
    }

    let expr = parse_expr(p);

    if p.at(&TokenKind::Colon) {
        p.advance();
        let upper = if can_start_expr(p.peek()) && !p.at(&TokenKind::Colon) {
            Some(parse_expr(p))
        } else {
            None
        };
        let step = if p.eat(&TokenKind::Colon) {
            if can_start_expr(p.peek()) {
                Some(parse_expr(p))
            } else {
                None
            }
        } else {
            None
        };
        SubscriptKind::Slice {
            lower: Some(expr),
            upper,
            step,
        }
    } else {
        SubscriptKind::Index(expr)
    }
}

pub fn parse_args(p: &mut Parser) -> Vec<Arg> {
    let mut args = Vec::new();
    if p.at(&TokenKind::RightParen) {
        return args;
    }

    loop {
        p.skip_newlines();
        if p.at(&TokenKind::RightParen) {
            break;
        }

        let start = p.current_span();

        // Check for keyword argument: name=value
        // Also handle language keywords used as kwarg names (e.g. model=m in train blocks)
        let kwarg_sym = match p.peek().clone() {
            TokenKind::Ident(sym) => Some(sym),
            TokenKind::Model if matches!(p.peek_at(1), &TokenKind::Eq) => {
                Some(p.interner.get_or_intern("model"))
            }
            _ => None,
        };
        if let Some(sym) = kwarg_sym {
            if matches!(p.peek_at(1), &TokenKind::Eq) {
                let name = sym;
                p.advance(); // consume ident/keyword
                p.advance(); // consume =
                let value = parse_expr(p);
                let span = start.merge(value.span);
                args.push(Arg {
                    name: Some(name.into()),
                    value,
                    span,
                });

                if !p.eat(&TokenKind::Comma) {
                    break;
                }
                continue;
            }
        }

        // Positional argument
        let value = parse_expr(p);
        let span = start.merge(value.span);
        args.push(Arg {
            name: None,
            value,
            span,
        });

        if !p.eat(&TokenKind::Comma) {
            break;
        }
    }

    p.skip_newlines();
    args
}

fn parse_paren_or_tuple(p: &mut Parser) -> Expr {
    let start = p.current_span();
    p.advance(); // consume (

    if p.at(&TokenKind::RightParen) {
        // Empty tuple ()
        let end = p.advance().span;
        return Expr {
            kind: ExprKind::TupleLiteral(Vec::new()),
            span: start.merge(end),
            id: p.next_node_id(),
        };
    }

    let first = parse_expr(p);

    if p.eat(&TokenKind::Comma) {
        // Tuple
        let mut items = vec![first];
        while !p.at(&TokenKind::RightParen) && !p.at(&TokenKind::Eof) {
            p.skip_newlines();
            if p.at(&TokenKind::RightParen) {
                break;
            }
            items.push(parse_expr(p));
            if !p.eat(&TokenKind::Comma) {
                break;
            }
        }
        let end = p.expect(&TokenKind::RightParen);
        Expr {
            kind: ExprKind::TupleLiteral(items),
            span: start.merge(end),
            id: p.next_node_id(),
        }
    } else {
        // Parenthesized expression
        let end = p.expect(&TokenKind::RightParen);
        Expr {
            kind: ExprKind::Paren(Box::new(first)),
            span: start.merge(end),
            id: p.next_node_id(),
        }
    }
}

fn parse_list_or_comp(p: &mut Parser) -> Expr {
    let start = p.current_span();
    p.advance(); // consume [

    if p.at(&TokenKind::RightBracket) {
        let end = p.advance().span;
        return Expr {
            kind: ExprKind::ListLiteral(Vec::new()),
            span: start.merge(end),
            id: p.next_node_id(),
        };
    }

    let first = parse_expr(p);

    // Check for list comprehension: [expr for x in iter]
    if p.at(&TokenKind::For) {
        let generators = parse_comp_generators(p);
        let end = p.expect(&TokenKind::RightBracket);
        return Expr {
            kind: ExprKind::ListComp {
                element: Box::new(first),
                generators,
            },
            span: start.merge(end),
            id: p.next_node_id(),
        };
    }

    // Regular list
    let mut items = vec![first];
    while p.eat(&TokenKind::Comma) {
        p.skip_newlines();
        if p.at(&TokenKind::RightBracket) {
            break;
        }
        items.push(parse_expr(p));
    }
    let end = p.expect(&TokenKind::RightBracket);
    Expr {
        kind: ExprKind::ListLiteral(items),
        span: start.merge(end),
        id: p.next_node_id(),
    }
}

fn parse_comp_generators(p: &mut Parser) -> Vec<CompGenerator> {
    let mut generators = Vec::new();
    while p.eat(&TokenKind::For) {
        let pattern = crate::pattern::parse_pattern(p);
        p.expect(&TokenKind::In);
        let iterable = parse_expr(p);
        let mut conditions = Vec::new();
        while p.at(&TokenKind::If) && !matches!(p.peek_at(1), &TokenKind::Else) {
            p.advance();
            conditions.push(parse_expr(p));
        }
        generators.push(CompGenerator {
            pattern,
            iterable,
            conditions,
        });
    }
    generators
}

fn parse_dict(p: &mut Parser) -> Expr {
    let start = p.current_span();
    p.advance(); // consume {

    if p.at(&TokenKind::RightBrace) {
        let end = p.advance().span;
        return Expr {
            kind: ExprKind::DictLiteral(Vec::new()),
            span: start.merge(end),
            id: p.next_node_id(),
        };
    }

    let mut entries = Vec::new();
    loop {
        p.skip_newlines();
        if p.at(&TokenKind::RightBrace) {
            break;
        }
        let key = parse_expr(p);
        p.expect(&TokenKind::Colon);
        let value = parse_expr(p);
        entries.push((key, value));
        if !p.eat(&TokenKind::Comma) {
            break;
        }
    }
    p.skip_newlines();
    let end = p.expect(&TokenKind::RightBrace);
    Expr {
        kind: ExprKind::DictLiteral(entries),
        span: start.merge(end),
        id: p.next_node_id(),
    }
}

fn parse_lambda(p: &mut Parser) -> Expr {
    let start = p.current_span();
    p.advance(); // consume |

    let mut params = Vec::new();
    if !p.at(&TokenKind::Bar) {
        loop {
            let param_start = p.current_span();
            let (name, _) = p.expect_ident();
            let type_ann = if p.eat(&TokenKind::Colon) {
                Some(crate::types::parse_primary_type(p))
            } else {
                None
            };
            params.push(LambdaParam {
                name,
                type_ann,
                span: param_start,
            });
            if !p.eat(&TokenKind::Comma) {
                break;
            }
        }
    }
    p.expect(&TokenKind::Bar);

    let body = parse_expr(p);
    let span = start.merge(body.span);
    Expr {
        kind: ExprKind::Lambda {
            params,
            body: Box::new(body),
        },
        span,
        id: p.next_node_id(),
    }
}

fn parse_if_expr(p: &mut Parser) -> Expr {
    let start = p.current_span();
    p.advance(); // consume 'if'
    let condition = parse_expr(p);

    // Parse the then-branch block (colon + indented body).
    p.expect(&TokenKind::Colon);
    let then_block = p.parse_block();

    // Extract the last expression from the then-block as the if-expression's value.
    // KNOWN LIMITATION: If the block is empty or the last statement is not an expression,
    // we produce `NoneLiteral` as a placeholder. This means `if cond: pass` silently
    // evaluates to `None` rather than being a type error. A future semantic pass should
    // reject if-expressions whose branches don't yield a value.
    let then_expr = if let Some(last) = then_block.stmts.last() {
        if let nsl_ast::stmt::StmtKind::Expr(e) = &last.kind {
            e.clone()
        } else {
            Expr {
                kind: ExprKind::NoneLiteral,
                span: start,
                id: p.next_node_id(),
            }
        }
    } else {
        Expr {
            kind: ExprKind::NoneLiteral,
            span: start,
            id: p.next_node_id(),
        }
    };

    // Parse the else-branch if present.
    // KNOWN LIMITATION: If-expressions used as values (e.g., `let x = if cond: a else: b`)
    // require an else branch to be well-typed, but the parser currently accepts a missing
    // else clause and fills in `NoneLiteral`. This is intentional at the parser level:
    // if-statements (not used as values) legitimately omit `else`. Distinguishing the two
    // contexts requires type information, so enforcement is deferred to the semantic checker.
    // TODO(semantic): Reject if-expressions in value position that lack an else branch.
    let else_expr = if p.eat(&TokenKind::Else) {
        p.expect(&TokenKind::Colon);
        let else_block = p.parse_block();
        if let Some(last) = else_block.stmts.last() {
            if let nsl_ast::stmt::StmtKind::Expr(e) = &last.kind {
                e.clone()
            } else {
                Expr {
                    kind: ExprKind::NoneLiteral,
                    span: start,
                    id: p.next_node_id(),
                }
            }
        } else {
            Expr {
                kind: ExprKind::NoneLiteral,
                span: start,
                id: p.next_node_id(),
            }
        }
    } else {
        Expr {
            kind: ExprKind::NoneLiteral,
            span: start,
            id: p.next_node_id(),
        }
    };

    let span = start.merge(else_expr.span);
    Expr {
        kind: ExprKind::IfExpr {
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        },
        span,
        id: p.next_node_id(),
    }
}

fn parse_fstring(p: &mut Parser) -> Expr {
    let start = p.current_span();
    p.advance(); // consume FStringStart

    let mut parts = Vec::new();

    loop {
        match p.peek() {
            TokenKind::FStringText(text) => {
                let text = text.clone();
                p.advance();
                parts.push(FStringPart::Text(text));
            }
            TokenKind::FStringExprStart => {
                p.advance();
                let expr = parse_expr(p);
                parts.push(FStringPart::Expr(expr));
                p.expect(&TokenKind::FStringExprEnd);
            }
            TokenKind::FStringEnd => {
                let end = p.advance().span;
                return Expr {
                    kind: ExprKind::FString(parts),
                    span: start.merge(end),
                    id: p.next_node_id(),
                };
            }
            _ => {
                let span = p.current_span();
                p.diagnostics.push(
                    Diagnostic::error("unexpected token in f-string")
                        .with_label(span, "here"),
                );
                return Expr {
                    kind: ExprKind::Error,
                    span: start,
                    id: p.next_node_id(),
                };
            }
        }
    }
}

fn can_start_expr(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::IntLiteral(_)
            | TokenKind::FloatLiteral(_)
            | TokenKind::StringLiteral(_)
            | TokenKind::FStringStart
            | TokenKind::True
            | TokenKind::False
            | TokenKind::None
            | TokenKind::Ident(_)
            | TokenKind::SelfKw
            | TokenKind::LeftParen
            | TokenKind::LeftBracket
            | TokenKind::LeftBrace
            | TokenKind::Bar
            | TokenKind::Minus
            | TokenKind::Not
            | TokenKind::If
            | TokenKind::Await
    )
}
