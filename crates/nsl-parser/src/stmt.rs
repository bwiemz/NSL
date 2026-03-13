use nsl_ast::decl::Decorator;
use nsl_ast::operator::AssignOp;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_lexer::TokenKind;

use crate::expr::{parse_args, parse_expr};
use crate::parser::Parser;

/// Parse a single statement.
pub fn parse_stmt(p: &mut Parser) -> Stmt {
    // Handle decorators
    if p.at(&TokenKind::At) {
        return parse_decorated_stmt(p);
    }

    p.skip_newlines();

    match p.peek() {
        TokenKind::Let | TokenKind::Const => parse_var_decl(p),
        TokenKind::Fn | TokenKind::Async => crate::decl::parse_fn_def_stmt(p),
        TokenKind::Model => crate::decl::parse_model_def_stmt(p),
        TokenKind::Struct => crate::decl::parse_struct_def_stmt(p),
        TokenKind::Enum => crate::decl::parse_enum_def_stmt(p),
        TokenKind::Trait => crate::decl::parse_trait_def_stmt(p),
        TokenKind::If => parse_if_stmt(p),
        TokenKind::For => parse_for_stmt(p),
        TokenKind::While => parse_while_stmt(p),
        TokenKind::Match => parse_match_stmt(p),
        TokenKind::Return => parse_return_stmt(p),
        TokenKind::Break => parse_break_stmt(p),
        TokenKind::Continue => parse_continue_stmt(p),
        TokenKind::Yield => parse_yield_stmt(p),
        TokenKind::Import => crate::decl::parse_import_stmt(p),
        TokenKind::From => crate::decl::parse_from_import_stmt(p),
        TokenKind::Train => crate::block::parse_train_block_stmt(p),
        TokenKind::Grad => crate::block::parse_grad_block_stmt(p),
        TokenKind::Quant => crate::block::parse_quant_block_stmt(p),
        TokenKind::Kernel => crate::block::parse_kernel_def_stmt(p),
        TokenKind::Tokenizer => crate::block::parse_tokenizer_def_stmt(p),
        TokenKind::Dataset => crate::block::parse_dataset_def_stmt(p),
        TokenKind::Datatype => crate::block::parse_datatype_def_stmt(p),
        TokenKind::Pub | TokenKind::Priv => parse_visibility_prefixed(p),
        _ => parse_expr_or_assign(p),
    }
}

fn parse_var_decl(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    let is_const = p.at(&TokenKind::Const);
    p.advance(); // consume let/const

    let pattern = crate::pattern::parse_pattern(p);

    let type_ann = if p.eat(&TokenKind::Colon) {
        Some(crate::types::parse_type(p))
    } else {
        None
    };

    if p.eat(&TokenKind::Eq) {
        // Check for `let pattern = grad(targets): body`
        if p.at(&TokenKind::Grad) {
            p.advance(); // consume 'grad'
            p.expect(&TokenKind::LeftParen);
            let targets = parse_expr(p);
            p.expect(&TokenKind::RightParen);
            p.expect(&TokenKind::Colon);
            let body = p.parse_block();
            let span = start.merge(body.span);
            return Stmt {
                kind: StmtKind::GradBlock(nsl_ast::block::GradBlock {
                    outputs: Some(pattern),
                    targets,
                    body: body.clone(),
                    span,
                }),
                span,
                id: p.next_node_id(),
            };
        }

        let value = Some(parse_expr(p));
        let span = value
            .as_ref()
            .map(|v| start.merge(v.span))
            .or(type_ann.as_ref().map(|t| start.merge(t.span)))
            .unwrap_or(start.merge(pattern.span));
        p.expect_end_of_stmt();

        Stmt {
            kind: StmtKind::VarDecl {
                is_const,
                pattern,
                type_ann,
                value,
            },
            span,
            id: p.next_node_id(),
        }
    } else {
        let span = type_ann
            .as_ref()
            .map(|t| start.merge(t.span))
            .unwrap_or(start.merge(pattern.span));
        p.expect_end_of_stmt();

        Stmt {
            kind: StmtKind::VarDecl {
                is_const,
                pattern,
                type_ann,
                value: None,
            },
            span,
            id: p.next_node_id(),
        }
    }
}

fn parse_if_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'if'

    let condition = parse_expr(p);
    p.expect(&TokenKind::Colon);
    let then_block = p.parse_block();

    let mut elif_clauses = Vec::new();
    p.skip_newlines();
    while p.at(&TokenKind::Elif) {
        p.advance();
        let cond = parse_expr(p);
        p.expect(&TokenKind::Colon);
        let block = p.parse_block();
        elif_clauses.push((cond, block));
        p.skip_newlines();
    }

    p.skip_newlines();
    let else_block = if p.at(&TokenKind::Else) {
        p.advance();
        p.expect(&TokenKind::Colon);
        Some(p.parse_block())
    } else {
        None
    };

    let end = else_block
        .as_ref()
        .map(|b| b.span)
        .or(elif_clauses.last().map(|(_, b)| b.span))
        .unwrap_or(then_block.span);

    Stmt {
        kind: StmtKind::If {
            condition,
            then_block,
            elif_clauses,
            else_block,
        },
        span: start.merge(end),
        id: p.next_node_id(),
    }
}

fn parse_for_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'for'

    let pattern = crate::pattern::parse_pattern(p);
    p.expect(&TokenKind::In);
    let iterable = parse_expr(p);
    p.expect(&TokenKind::Colon);
    let body = p.parse_block();

    Stmt {
        kind: StmtKind::For {
            pattern,
            iterable,
            body: body.clone(),
        },
        span: start.merge(body.span),
        id: p.next_node_id(),
    }
}

fn parse_while_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'while'

    // Check for while-let
    if p.at(&TokenKind::Let) {
        p.advance();
        let pattern = crate::pattern::parse_pattern(p);
        p.expect(&TokenKind::Eq);
        let expr = parse_expr(p);
        p.expect(&TokenKind::Colon);
        let body = p.parse_block();
        return Stmt {
            kind: StmtKind::WhileLet {
                pattern,
                expr,
                body: body.clone(),
            },
            span: start.merge(body.span),
            id: p.next_node_id(),
        };
    }

    let condition = parse_expr(p);
    p.expect(&TokenKind::Colon);
    let body = p.parse_block();

    Stmt {
        kind: StmtKind::While {
            condition,
            body: body.clone(),
        },
        span: start.merge(body.span),
        id: p.next_node_id(),
    }
}

fn parse_match_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'match'

    let subject = parse_expr(p);
    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut arms = Vec::new();
    while p.at(&TokenKind::Case) {
        let arm_start = p.current_span();
        p.advance(); // consume 'case'

        let pattern = crate::pattern::parse_pattern(p);

        // Optional guard: `if condition`
        let guard = if p.at(&TokenKind::If) {
            p.advance();
            Some(parse_expr(p))
        } else {
            None
        };

        // case Pattern: \n INDENT stmts DEDENT
        p.expect(&TokenKind::Colon);
        let body = p.parse_block();

        arms.push(nsl_ast::expr::MatchArm {
            pattern,
            guard,
            body: body.clone(),
            span: arm_start.merge(body.span),
        });
        p.skip_newlines();
    }

    p.eat(&TokenKind::Dedent);

    Stmt {
        kind: StmtKind::Match { subject, arms },
        span: start.merge(p.prev_span()),
        id: p.next_node_id(),
    }
}

fn parse_return_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'return'

    let value = if !p.at(&TokenKind::Newline)
        && !p.at(&TokenKind::Eof)
        && !p.at(&TokenKind::Dedent)
    {
        Some(parse_expr(p))
    } else {
        None
    };

    let span = value
        .as_ref()
        .map(|v| start.merge(v.span))
        .unwrap_or(start);
    p.expect_end_of_stmt();

    Stmt {
        kind: StmtKind::Return(value),
        span,
        id: p.next_node_id(),
    }
}

fn parse_break_stmt(p: &mut Parser) -> Stmt {
    let span = p.current_span();
    p.advance();
    p.expect_end_of_stmt();
    Stmt {
        kind: StmtKind::Break,
        span,
        id: p.next_node_id(),
    }
}

fn parse_continue_stmt(p: &mut Parser) -> Stmt {
    let span = p.current_span();
    p.advance();
    p.expect_end_of_stmt();
    Stmt {
        kind: StmtKind::Continue,
        span,
        id: p.next_node_id(),
    }
}

fn parse_yield_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'yield'

    let value = if !p.at(&TokenKind::Newline)
        && !p.at(&TokenKind::Eof)
        && !p.at(&TokenKind::Dedent)
    {
        Some(parse_expr(p))
    } else {
        None
    };

    let span = value
        .as_ref()
        .map(|v| start.merge(v.span))
        .unwrap_or(start);
    p.expect_end_of_stmt();

    Stmt {
        kind: StmtKind::Yield(value),
        span,
        id: p.next_node_id(),
    }
}

/// Check whether an expression is a valid assignment target.
fn is_valid_assign_target(expr: &nsl_ast::expr::Expr) -> bool {
    use nsl_ast::expr::ExprKind;
    match &expr.kind {
        ExprKind::Ident(_)
        | ExprKind::MemberAccess { .. }
        | ExprKind::Subscript { .. } => true,
        ExprKind::TupleLiteral(items) | ExprKind::ListLiteral(items) => {
            items.iter().all(is_valid_assign_target)
        }
        _ => false,
    }
}

fn parse_expr_or_assign(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    let expr = parse_expr(p);

    // Check for assignment operators
    let op = match p.peek() {
        TokenKind::Eq => Some(AssignOp::Assign),
        TokenKind::PlusEq => Some(AssignOp::AddAssign),
        TokenKind::MinusEq => Some(AssignOp::SubAssign),
        TokenKind::StarEq => Some(AssignOp::MulAssign),
        TokenKind::SlashEq => Some(AssignOp::DivAssign),
        _ => None,
    };

    if let Some(op) = op {
        // Validate that the target is assignable
        if !is_valid_assign_target(&expr) {
            p.diagnostics.push(
                nsl_errors::Diagnostic::error("invalid assignment target")
                    .with_label(expr.span, "not assignable"),
            );
        }
        p.advance(); // consume assignment operator
        let value = parse_expr(p);
        let span = start.merge(value.span);
        p.expect_end_of_stmt();
        Stmt {
            kind: StmtKind::Assign {
                target: expr,
                op,
                value,
            },
            span,
            id: p.next_node_id(),
        }
    } else {
        let span = start.merge(expr.span);
        p.expect_end_of_stmt();
        Stmt {
            kind: StmtKind::Expr(expr),
            span,
            id: p.next_node_id(),
        }
    }
}

fn parse_decorated_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    let mut decorators = Vec::new();

    while p.at(&TokenKind::At) {
        decorators.push(parse_decorator(p));
        p.skip_newlines();
    }

    let stmt = parse_stmt(p);
    let span = start.merge(stmt.span);

    Stmt {
        kind: StmtKind::Decorated {
            decorators,
            stmt: Box::new(stmt),
        },
        span,
        id: p.next_node_id(),
    }
}

fn parse_decorator(p: &mut Parser) -> Decorator {
    let start = p.current_span();
    p.advance(); // consume @

    let mut name = Vec::new();
    let (first, _) = p.expect_ident();
    name.push(first);
    while p.eat(&TokenKind::Dot) {
        let (segment, _) = p.expect_ident();
        name.push(segment);
    }

    let args = if p.at(&TokenKind::LeftParen) {
        p.advance();
        let args = parse_args(p);
        p.expect(&TokenKind::RightParen);
        Some(args)
    } else {
        None
    };

    Decorator {
        name,
        args,
        span: start.merge(p.prev_span()),
    }
}

fn parse_visibility_prefixed(p: &mut Parser) -> Stmt {
    // pub/priv before a declaration — just skip it for now and parse the next stmt
    let start = p.current_span();
    p.advance(); // consume pub/priv

    let mut decorators = Vec::new();
    while p.at(&TokenKind::At) {
        decorators.push(parse_decorator(p));
        p.skip_newlines();
    }

    let inner = parse_stmt(p);

    if decorators.is_empty() {
        // Just return the inner stmt (visibility handling is for later phases)
        inner
    } else {
        Stmt {
            kind: StmtKind::Decorated {
                decorators,
                stmt: Box::new(inner),
            },
            span: start.merge(p.prev_span()),
            id: p.next_node_id(),
        }
    }
}
