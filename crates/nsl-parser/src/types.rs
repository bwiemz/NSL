use nsl_ast::types::*;
use nsl_ast::{Span, Symbol};
use nsl_lexer::TokenKind;

use crate::parser::Parser;

/// Parse a type expression.
pub fn parse_type(p: &mut Parser) -> TypeExpr {
    let mut ty = parse_primary_type(p);

    // Union: A | B | C
    while p.at(&TokenKind::Bar) {
        p.advance();
        let right = parse_primary_type(p);
        let span = ty.span.merge(right.span);
        let types = match ty.kind {
            TypeExprKind::Union(mut existing) => {
                existing.push(right);
                existing
            }
            _ => vec![ty.clone(), right],
        };
        ty = TypeExpr {
            kind: TypeExprKind::Union(types),
            span,
            id: p.next_node_id(),
        };
    }

    ty
}

/// Parse a single (non-union) type expression. Public so lambda params can
/// use this to avoid consuming `|` as a union-type separator.
pub fn parse_primary_type(p: &mut Parser) -> TypeExpr {
    match p.peek().clone() {
        TokenKind::Ident(sym) => {
            let name: Symbol = sym.into();
            let start = p.advance().span;

            // Check for generic: Name<...>
            if p.at(&TokenKind::Lt) {
                parse_generic_or_tensor_type(p, name, start)
            } else {
                TypeExpr {
                    kind: TypeExprKind::Named(name),
                    span: start,
                    id: p.next_node_id(),
                }
            }
        }

        // Keywords that are valid type names
        TokenKind::SelfKw => {
            let start = p.advance().span;
            let sym = p.intern("self");
            TypeExpr {
                kind: TypeExprKind::Named(sym),
                span: start,
                id: p.next_node_id(),
            }
        }

        // Function type: (A, B) -> C
        TokenKind::LeftParen => parse_function_or_tuple_type(p),

        // Wildcard: _
        TokenKind::Underscore => {
            let span = p.advance().span;
            TypeExpr {
                kind: TypeExprKind::Wildcard,
                span,
                id: p.next_node_id(),
            }
        }

        // Fixed-size array type: [Type; N]
        TokenKind::LeftBracket => {
            let start = p.advance().span; // consume [
            let elem_type = parse_type(p);
            p.expect(&TokenKind::Semicolon);
            let size = match p.peek().clone() {
                TokenKind::IntLiteral(n) => {
                    p.advance();
                    n
                }
                _ => {
                    p.diagnostics.push(
                        nsl_errors::Diagnostic::error("expected integer literal in fixed array size")
                            .with_label(p.current_span(), "expected integer"),
                    );
                    1
                }
            };
            let end = p.expect(&TokenKind::RightBracket);
            TypeExpr {
                kind: TypeExprKind::FixedArray {
                    element_type: Box::new(elem_type),
                    size,
                },
                span: start.merge(end),
                id: p.next_node_id(),
            }
        }

        _ => {
            let span = p.current_span();
            p.diagnostics.push(
                nsl_errors::Diagnostic::error(format!("expected type, found {}", p.peek()))
                    .with_label(span, "expected type"),
            );
            p.advance();
            let sym = p.intern("error");
            TypeExpr {
                kind: TypeExprKind::Named(sym),
                span,
                id: p.next_node_id(),
            }
        }
    }
}

fn parse_generic_or_tensor_type(p: &mut Parser, name: Symbol, start: Span) -> TypeExpr {
    // Resolve well-known type names
    let name_str = p.resolve(name).unwrap_or("").to_string();

    match name_str.as_str() {
        "Tensor" => parse_tensor_type(p, start, TypeExprKind::Tensor {
            shape: Vec::new(),
            dtype: name, // placeholder
            device: None,
        }),
        "Param" => parse_param_type(p, start),
        "Buffer" => parse_buffer_type(p, start),
        "Sparse" => parse_sparse_type(p, start),
        _ => {
            // Generic type: Name<A, B, C>
            p.advance(); // consume <
            let mut args = Vec::new();
            while !p.at(&TokenKind::Gt) && !p.at(&TokenKind::Eof) {
                args.push(parse_type(p));
                if !p.eat(&TokenKind::Comma) {
                    break;
                }
            }
            let end = p.expect(&TokenKind::Gt);
            TypeExpr {
                kind: TypeExprKind::Generic { name, args },
                span: start.merge(end),
                id: p.next_node_id(),
            }
        }
    }
}

fn parse_tensor_type(p: &mut Parser, start: Span, _placeholder: TypeExprKind) -> TypeExpr {
    p.advance(); // consume <

    // Shape: [dim, dim, ...]
    p.expect(&TokenKind::LeftBracket);
    let shape = parse_dim_list(p);
    p.expect(&TokenKind::RightBracket);

    // dtype
    p.expect(&TokenKind::Comma);
    let (dtype, _) = p.expect_ident();

    // Optional device
    let device = if p.eat(&TokenKind::Comma) {
        Some(parse_device_expr(p))
    } else {
        None
    };

    let end = p.expect(&TokenKind::Gt);
    TypeExpr {
        kind: TypeExprKind::Tensor {
            shape,
            dtype,
            device,
        },
        span: start.merge(end),
        id: p.next_node_id(),
    }
}

fn parse_param_type(p: &mut Parser, start: Span) -> TypeExpr {
    p.advance(); // consume <
    p.expect(&TokenKind::LeftBracket);
    let shape = parse_dim_list(p);
    p.expect(&TokenKind::RightBracket);
    p.expect(&TokenKind::Comma);
    let (dtype, _) = p.expect_ident();
    let end = p.expect(&TokenKind::Gt);
    TypeExpr {
        kind: TypeExprKind::Param { shape, dtype },
        span: start.merge(end),
        id: p.next_node_id(),
    }
}

fn parse_buffer_type(p: &mut Parser, start: Span) -> TypeExpr {
    p.advance(); // consume <
    p.expect(&TokenKind::LeftBracket);
    let shape = parse_dim_list(p);
    p.expect(&TokenKind::RightBracket);
    p.expect(&TokenKind::Comma);
    let (dtype, _) = p.expect_ident();
    let end = p.expect(&TokenKind::Gt);
    TypeExpr {
        kind: TypeExprKind::Buffer { shape, dtype },
        span: start.merge(end),
        id: p.next_node_id(),
    }
}

fn parse_sparse_type(p: &mut Parser, start: Span) -> TypeExpr {
    p.advance(); // consume <
    p.expect(&TokenKind::LeftBracket);
    let shape = parse_dim_list(p);
    p.expect(&TokenKind::RightBracket);
    p.expect(&TokenKind::Comma);
    let (dtype, _) = p.expect_ident();
    p.expect(&TokenKind::Comma);
    let (format, _) = p.expect_ident();
    let end = p.expect(&TokenKind::Gt);
    TypeExpr {
        kind: TypeExprKind::Sparse {
            shape,
            dtype,
            format,
        },
        span: start.merge(end),
        id: p.next_node_id(),
    }
}

fn parse_dim_list(p: &mut Parser) -> Vec<DimExpr> {
    let mut dims = Vec::new();
    while !p.at(&TokenKind::RightBracket) && !p.at(&TokenKind::Eof) {
        dims.push(parse_dim_expr(p));
        if !p.eat(&TokenKind::Comma) {
            break;
        }
    }
    dims
}

fn parse_dim_expr(p: &mut Parser) -> DimExpr {
    match p.peek().clone() {
        TokenKind::IntLiteral(v) => {
            p.advance();
            DimExpr::Concrete(v)
        }
        TokenKind::Underscore => {
            p.advance();
            DimExpr::Wildcard
        }
        TokenKind::Ident(sym) => {
            p.advance();
            // Check for named dim: name="B" or name=123
            if p.eat(&TokenKind::Eq) {
                match p.peek().clone() {
                    TokenKind::StringLiteral(s) => {
                        p.advance();
                        DimExpr::Named {
                            name: sym.into(),
                            value: DimValue::String(s),
                        }
                    }
                    TokenKind::IntLiteral(v) => {
                        p.advance();
                        DimExpr::Named {
                            name: sym.into(),
                            value: DimValue::Int(v),
                        }
                    }
                    _ => DimExpr::Symbolic(sym.into()),
                }
            } else {
                DimExpr::Symbolic(sym.into())
            }
        }
        _ => {
            p.diagnostics.push(
                nsl_errors::Diagnostic::error(format!(
                    "expected dimension (integer, identifier, or _), found {}",
                    p.peek()
                ))
                .with_label(p.current_span(), "expected dimension"),
            );
            p.advance();
            DimExpr::Wildcard
        }
    }
}

fn parse_device_expr(p: &mut Parser) -> DeviceExpr {
    if let TokenKind::Ident(sym) = p.peek().clone() {
        let name = p.interner.resolve(sym).unwrap_or("").to_string();
        p.advance();
        match name.as_str() {
            "cpu" => DeviceExpr::Cpu,
            "cuda" => {
                if p.at(&TokenKind::LeftParen) {
                    p.advance();
                    let idx = crate::expr::parse_expr(p);
                    p.expect(&TokenKind::RightParen);
                    DeviceExpr::Cuda(Some(Box::new(idx)))
                } else {
                    DeviceExpr::Cuda(None)
                }
            }
            "metal" => DeviceExpr::Metal,
            "rocm" => {
                if p.at(&TokenKind::LeftParen) {
                    p.advance();
                    let idx = crate::expr::parse_expr(p);
                    p.expect(&TokenKind::RightParen);
                    DeviceExpr::Rocm(Some(Box::new(idx)))
                } else {
                    DeviceExpr::Rocm(None)
                }
            }
            _ => {
                // npu<Target> or just a named device
                if p.at(&TokenKind::Lt) {
                    p.advance();
                    let (target, _) = p.expect_ident();
                    p.expect(&TokenKind::Gt);
                    DeviceExpr::Npu(target)
                } else {
                    // Unknown device, treat as Npu
                    DeviceExpr::Npu(sym.into())
                }
            }
        }
    } else {
        p.diagnostics.push(
            nsl_errors::Diagnostic::error(format!("expected device, found {}", p.peek()))
                .with_label(p.current_span(), "expected device"),
        );
        DeviceExpr::Cpu
    }
}

fn parse_function_or_tuple_type(p: &mut Parser) -> TypeExpr {
    let start = p.current_span();
    p.advance(); // consume (

    let mut types = Vec::new();
    while !p.at(&TokenKind::RightParen) && !p.at(&TokenKind::Eof) {
        types.push(parse_type(p));
        if !p.eat(&TokenKind::Comma) {
            break;
        }
    }
    p.expect(&TokenKind::RightParen);

    // Check for function type: (...) -> RetType
    if p.eat(&TokenKind::Arrow) {
        let ret = parse_type(p);
        let span = start.merge(ret.span);
        TypeExpr {
            kind: TypeExprKind::Function {
                params: types,
                ret: Box::new(ret),
            },
            span,
            id: p.next_node_id(),
        }
    } else {
        // Tuple type
        let span = start.merge(p.prev_span());
        TypeExpr {
            kind: TypeExprKind::Tuple(types),
            span,
            id: p.next_node_id(),
        }
    }
}
