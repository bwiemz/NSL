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
    // Immutable borrow: &Type
    if p.at(&TokenKind::Ampersand) {
        let start = p.advance().span;
        // Reject &mut — mutable borrows are not supported yet
        if p.at(&TokenKind::Mut) {
            let end = p.advance().span;
            p.diagnostics.push(
                nsl_errors::Diagnostic::error("mutable borrows (`&mut T`) are not supported — use `&T` for read-only borrowing")
                    .with_label(start.merge(end), "use `&T` instead of `&mut T`"),
            );
            // Parse the inner type anyway so the parser can continue
            let inner = parse_primary_type(p);
            let span = start.merge(inner.span);
            return TypeExpr {
                kind: TypeExprKind::Borrow(Box::new(inner)),
                span,
                id: p.next_node_id(),
            };
        }
        let inner = parse_primary_type(p);
        // Reject &&T — nested borrows are not supported
        if matches!(inner.kind, TypeExprKind::Borrow(_)) {
            p.diagnostics.push(
                nsl_errors::Diagnostic::error("nested borrows (`&&T`) are not supported — use a single `&T` borrow")
                    .with_label(start.merge(inner.span), "remove the extra `&`"),
            );
        }
        let span = start.merge(inner.span);
        return TypeExpr {
            kind: TypeExprKind::Borrow(Box::new(inner)),
            span,
            id: p.next_node_id(),
        };
    }

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
                    _ => {
                        p.diagnostics.push(
                            nsl_errors::Diagnostic::error(format!(
                                "expected string or integer after '=' in dimension, found {}",
                                p.peek()
                            ))
                            .with_label(p.current_span(), "expected value"),
                        );
                        p.advance();
                        DimExpr::Symbolic(sym.into())
                    }
                }
            } else if p.eat(&TokenKind::Lt) {
                // Bounded dim: SeqLen < 4096
                match p.peek().clone() {
                    TokenKind::IntLiteral(v) => {
                        p.advance();
                        DimExpr::Bounded {
                            name: sym.into(),
                            upper_bound: v,
                        }
                    }
                    _ => {
                        p.diagnostics.push(
                            nsl_errors::Diagnostic::error(format!(
                                "expected integer literal after '<' in bounded dimension, found {}",
                                p.peek()
                            ))
                            .with_label(p.current_span(), "expected integer"),
                        );
                        p.advance();
                        DimExpr::Symbolic(sym.into())
                    }
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

/// Parse a type expression, but reject borrow types (`&T`) with an error.
/// Used for return type positions where borrows cannot escape.
pub fn parse_type_no_borrow(p: &mut Parser) -> TypeExpr {
    let ty = parse_type(p);
    reject_borrow_in_type(&ty, p);
    ty
}

/// Recursively check for borrow types and emit an error if found.
/// Walks into compound types (tuples, optionals, unions, functions, etc.)
/// so that `(int, &str)` or `fn() -> &int` in return position are also caught.
fn reject_borrow_in_type(ty: &TypeExpr, p: &mut Parser) {
    match &ty.kind {
        TypeExprKind::Borrow(_) => {
            p.diagnostics.push(
                nsl_errors::Diagnostic::error("borrow types (`&T`) cannot appear in return position — borrows cannot escape function scope")
                    .with_label(ty.span, "borrow in return type"),
            );
        }
        TypeExprKind::Tuple(elems) => {
            for elem in elems {
                reject_borrow_in_type(elem, p);
            }
        }
        TypeExprKind::Union(variants) => {
            for variant in variants {
                reject_borrow_in_type(variant, p);
            }
        }
        TypeExprKind::Generic { args, .. } => {
            for arg in args {
                reject_borrow_in_type(arg, p);
            }
        }
        TypeExprKind::Function { params, ret, .. } => {
            // Borrows in function *param* types are allowed; only reject in the return type.
            reject_borrow_in_type(ret, p);
            let _ = params; // params are fine
        }
        TypeExprKind::FixedArray { element_type, .. } => {
            reject_borrow_in_type(element_type, p);
        }
        // Named, Tensor, Param, Buffer, Sparse, Wildcard — leaf types, never borrow
        _ => {}
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
                effect: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::types::TypeExprKind;
    use nsl_errors::FileId;
    use nsl_lexer;

    fn parse_type_str(src: &str) -> (TypeExpr, Vec<nsl_errors::Diagnostic>) {
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, _) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
        let mut p = crate::parser::Parser::new(&tokens, &mut interner);
        let ty = parse_type(&mut p);
        let diags = p.diagnostics;
        (ty, diags)
    }

    #[test]
    fn test_parse_borrow_named_type() {
        let (ty, diags) = parse_type_str("&int");
        assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
        match &ty.kind {
            TypeExprKind::Borrow(inner) => {
                assert!(matches!(inner.kind, TypeExprKind::Named(_)));
            }
            other => panic!("expected Borrow, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_borrow_tensor_type() {
        let (ty, diags) = parse_type_str("&Tensor<[4], f32>");
        assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
        match &ty.kind {
            TypeExprKind::Borrow(inner) => {
                assert!(matches!(inner.kind, TypeExprKind::Tensor { .. }));
            }
            other => panic!("expected Borrow(Tensor), got {:?}", other),
        }
    }

    #[test]
    fn test_parse_non_borrow_type() {
        let (ty, diags) = parse_type_str("Tensor<[4], f32>");
        assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
        assert!(matches!(ty.kind, TypeExprKind::Tensor { .. }));
    }

    #[test]
    fn test_borrow_return_type_error() {
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, _) = nsl_lexer::tokenize("&Tensor<[4], f32>", FileId(0), &mut interner);
        let mut p = crate::parser::Parser::new(&tokens, &mut interner);
        let ty = parse_type_no_borrow(&mut p);
        assert!(
            matches!(ty.kind, TypeExprKind::Borrow(_)),
            "should still parse the borrow"
        );
        assert!(
            !p.diagnostics.is_empty(),
            "should produce a diagnostic for borrow in return position"
        );
        assert!(
            format!("{:?}", p.diagnostics[0]).contains("borrow"),
            "diagnostic should mention 'borrow', got: {:?}",
            p.diagnostics[0]
        );
    }
}
