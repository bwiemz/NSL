use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::pattern::*;
use nsl_lexer::TokenKind;

use crate::parser::Parser;

/// Parse a pattern (used in let, for, match, destructuring).
pub fn parse_pattern(p: &mut Parser) -> Pattern {
    let mut pat = parse_primary_pattern(p);

    // Or pattern: a | b
    if p.at(&TokenKind::Bar) {
        let mut patterns = vec![pat];
        while p.eat(&TokenKind::Bar) {
            patterns.push(parse_primary_pattern(p));
        }
        let span = patterns.first().unwrap().span.merge(patterns.last().unwrap().span);
        pat = Pattern {
            kind: PatternKind::Or(patterns),
            span,
            id: p.next_node_id(),
        };
    }

    pat
}

fn parse_primary_pattern(p: &mut Parser) -> Pattern {
    match p.peek().clone() {
        // Wildcard
        TokenKind::Underscore => {
            let span = p.advance().span;
            Pattern {
                kind: PatternKind::Wildcard,
                span,
                id: p.next_node_id(),
            }
        }

        // Literal patterns
        TokenKind::IntLiteral(v) => {
            let span = p.advance().span;
            Pattern {
                kind: PatternKind::Literal(Box::new(Expr {
                    kind: ExprKind::IntLiteral(v),
                    span,
                    id: p.next_node_id(),
                })),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::FloatLiteral(v) => {
            let span = p.advance().span;
            Pattern {
                kind: PatternKind::Literal(Box::new(Expr {
                    kind: ExprKind::FloatLiteral(v),
                    span,
                    id: p.next_node_id(),
                })),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::StringLiteral(s) => {
            let s = s.clone();
            let span = p.advance().span;
            Pattern {
                kind: PatternKind::Literal(Box::new(Expr {
                    kind: ExprKind::StringLiteral(s),
                    span,
                    id: p.next_node_id(),
                })),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::True => {
            let span = p.advance().span;
            Pattern {
                kind: PatternKind::Literal(Box::new(Expr {
                    kind: ExprKind::BoolLiteral(true),
                    span,
                    id: p.next_node_id(),
                })),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::False => {
            let span = p.advance().span;
            Pattern {
                kind: PatternKind::Literal(Box::new(Expr {
                    kind: ExprKind::BoolLiteral(false),
                    span,
                    id: p.next_node_id(),
                })),
                span,
                id: p.next_node_id(),
            }
        }
        TokenKind::None => {
            let span = p.advance().span;
            Pattern {
                kind: PatternKind::Literal(Box::new(Expr {
                    kind: ExprKind::NoneLiteral,
                    span,
                    id: p.next_node_id(),
                })),
                span,
                id: p.next_node_id(),
            }
        }

        // Identifier or constructor pattern
        TokenKind::Ident(sym) => {
            let sym = nsl_ast::Symbol::from(sym);
            let span = p.advance().span;

            if p.at(&TokenKind::LeftParen) {
                // Constructor pattern: Name(a, b)
                p.advance();
                let mut args = Vec::new();
                while !p.at(&TokenKind::RightParen) && !p.at(&TokenKind::Eof) {
                    args.push(parse_pattern(p));
                    if !p.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                let end = p.expect(&TokenKind::RightParen);
                Pattern {
                    kind: PatternKind::Constructor {
                        path: vec![sym],
                        args,
                    },
                    span: span.merge(end),
                    id: p.next_node_id(),
                }
            } else if p.at(&TokenKind::Dot) {
                // Qualified constructor: Enum.Variant(args)
                let mut path = vec![sym];
                while p.eat(&TokenKind::Dot) {
                    let (seg, _) = p.expect_ident();
                    path.push(seg);
                }
                if p.at(&TokenKind::LeftParen) {
                    p.advance();
                    let mut args = Vec::new();
                    while !p.at(&TokenKind::RightParen) && !p.at(&TokenKind::Eof) {
                        args.push(parse_pattern(p));
                        if !p.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    let end = p.expect(&TokenKind::RightParen);
                    Pattern {
                        kind: PatternKind::Constructor { path, args },
                        span: span.merge(end),
                        id: p.next_node_id(),
                    }
                } else {
                    // Just a qualified path as constructor with no args
                    Pattern {
                        kind: PatternKind::Constructor {
                            path,
                            args: Vec::new(),
                        },
                        span: span.merge(p.prev_span()),
                        id: p.next_node_id(),
                    }
                }
            } else {
                // Simple identifier binding
                Pattern {
                    kind: PatternKind::Ident(sym),
                    span,
                    id: p.next_node_id(),
                }
            }
        }

        // Tuple pattern: (a, b, c)
        TokenKind::LeftParen => {
            let start = p.advance().span;
            let mut patterns = Vec::new();
            while !p.at(&TokenKind::RightParen) && !p.at(&TokenKind::Eof) {
                patterns.push(parse_pattern(p));
                if !p.eat(&TokenKind::Comma) {
                    break;
                }
            }
            let end = p.expect(&TokenKind::RightParen);
            Pattern {
                kind: PatternKind::Tuple(patterns),
                span: start.merge(end),
                id: p.next_node_id(),
            }
        }

        // List pattern: [a, b, c]
        TokenKind::LeftBracket => {
            let start = p.advance().span;
            let mut patterns = Vec::new();
            while !p.at(&TokenKind::RightBracket) && !p.at(&TokenKind::Eof) {
                patterns.push(parse_pattern(p));
                if !p.eat(&TokenKind::Comma) {
                    break;
                }
            }
            let end = p.expect(&TokenKind::RightBracket);
            Pattern {
                kind: PatternKind::List(patterns),
                span: start.merge(end),
                id: p.next_node_id(),
            }
        }

        // Struct pattern: {field1, field2}
        TokenKind::LeftBrace => {
            let start = p.advance().span;
            let mut fields = Vec::new();
            let mut rest = None;

            while !p.at(&TokenKind::RightBrace) && !p.at(&TokenKind::Eof) {
                if p.at(&TokenKind::DotDot) {
                    p.advance();
                    if let TokenKind::Ident(sym) = p.peek().clone() {
                        rest = Some(sym.into());
                        p.advance();
                    }
                    break;
                }

                let field_start = p.current_span();
                let (name, _) = p.expect_ident();
                let pattern = if p.eat(&TokenKind::Colon) {
                    Some(parse_pattern(p))
                } else {
                    None
                };
                fields.push(FieldPattern {
                    name,
                    pattern,
                    span: field_start.merge(p.prev_span()),
                });

                if !p.eat(&TokenKind::Comma) {
                    break;
                }
            }

            let end = p.expect(&TokenKind::RightBrace);
            Pattern {
                kind: PatternKind::Struct { fields, rest },
                span: start.merge(end),
                id: p.next_node_id(),
            }
        }

        // Rest/spread: *name
        TokenKind::Star => {
            let start = p.advance().span;
            let name = if let TokenKind::Ident(sym) = p.peek().clone() {
                p.advance();
                Some(sym.into())
            } else {
                None
            };
            Pattern {
                kind: PatternKind::Rest(name),
                span: start.merge(p.prev_span()),
                id: p.next_node_id(),
            }
        }

        _ => {
            let span = p.current_span();
            p.diagnostics.push(
                nsl_errors::Diagnostic::error(format!("expected pattern, found {}", p.peek()))
                    .with_label(span, "expected pattern"),
            );
            p.advance();
            Pattern {
                kind: PatternKind::Wildcard,
                span,
                id: p.next_node_id(),
            }
        }
    }
}
